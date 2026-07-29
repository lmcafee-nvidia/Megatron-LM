# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from pathlib import Path

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.mamba_slot_allocator import (
    MAX_INTERMEDIATE_OFFSETS_PER_REQUEST,
)
from tests.test_utils.python_scripts.prefix_cache_coverage import (
    generated_matrix_cases,
    load_manifest,
)
from tests.unit_tests.inference.contexts.test_dynamic_prefix_caching import PrefixCachingTestBase

pytestmark = pytest.mark.internal

_GIB = 1024**3
_CAPACITY_BUFFER_GB = 1 / 1024
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = load_manifest(_REPO_ROOT / "tests/test_utils/prefix_cache_coverage.yaml")
_POLICIES = {
    "ref_zero": PrefixCachingEvictionPolicy.REF_ZERO,
    "lru": PrefixCachingEvictionPolicy.LRU,
}
_BOUNDARY_PROFILES = {
    "bs64_chunk64": (64, 64),
    "bs128_chunk64": (128, 64),
    "bs256_chunk128": (256, 128),
}


class TestHybridMambaPrefixCacheStress(PrefixCachingTestBase):

    def _typed_mamba_config(self, *, mamba_chunk_size, dtype_name):
        config = self._mamba_config(mamba_chunk_size=mamba_chunk_size)
        dtype = getattr(torch, dtype_name)
        config.conv_states_dtype = dtype
        config.ssm_states_dtype = dtype
        return config

    @staticmethod
    def _mamba_slot_bytes(mamba_config):
        mamba_layers = sum(layer_type == "M" for layer_type in mamba_config.layer_type_list)
        conv_values = math.prod(mamba_config.conv_states_shape)
        ssm_values = math.prod(mamba_config.ssm_states_shape)
        return mamba_layers * (
            conv_values * mamba_config.conv_states_dtype.itemsize
            + ssm_values * mamba_config.ssm_states_dtype.itemsize
        )

    @staticmethod
    def _kv_block_bytes(block_size_tokens):
        # PrefixCachingTestBase builds two attention layers, two KV heads, and
        # eight values per head in float32, with separate K and V storage.
        return 4 * 2 * 2 * block_size_tokens * 2 * 8

    def _expected_kv_pool_size(
        self, *, buffer_size_gb, block_size_tokens, max_requests, mamba_config
    ):
        buffer_bytes = int(buffer_size_gb * _GIB)
        paused_bytes = int(0.2 * buffer_size_gb * _GIB)
        live_mamba_bytes = max_requests * self._mamba_slot_bytes(mamba_config)
        live_ratio = live_mamba_bytes / (buffer_bytes + paused_bytes)
        adjusted_buffer_bytes = int(buffer_bytes * (1.0 - live_ratio))
        return max(2, adjusted_buffer_bytes // self._kv_block_bytes(block_size_tokens))

    def _mamba_budget_gb(
        self, *, durable_slots, block_size_tokens, max_tokens, max_requests, mamba_config
    ):
        scratch_slots = min(
            max(0, math.ceil(max_tokens / block_size_tokens) - 1),
            MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * max_requests,
        )
        return (scratch_slots + durable_slots) * self._mamba_slot_bytes(mamba_config) / _GIB

    def _ctx_with_slots(
        self,
        *,
        durable_slots,
        block_size_tokens=64,
        max_tokens=512,
        max_requests=4,
        buffer_size_gb=_CAPACITY_BUFFER_GB,
        policy=PrefixCachingEvictionPolicy.LRU,
        mamba_chunk_size=64,
        dtype_name="float32",
    ):
        mamba_config = self._typed_mamba_config(
            mamba_chunk_size=mamba_chunk_size, dtype_name=dtype_name
        )
        mamba_gb = self._mamba_budget_gb(
            durable_slots=durable_slots,
            block_size_tokens=block_size_tokens,
            max_tokens=max_tokens,
            max_requests=max_requests,
            mamba_config=mamba_config,
        )
        ctx = self._ctx(
            buffer_size_gb=buffer_size_gb,
            block_size_tokens=block_size_tokens,
            max_sequence_length=block_size_tokens * 8,
            rounder=1,
            max_tokens=max_tokens,
            max_requests=max_requests,
            prefix_caching_eviction_policy=policy,
            mamba_config=mamba_config,
            prefix_caching_mamba_gb=mamba_gb,
        )

        expected_pool_size = self._expected_kv_pool_size(
            buffer_size_gb=buffer_size_gb,
            block_size_tokens=block_size_tokens,
            max_requests=max_requests,
            mamba_config=mamba_config,
        )
        assert ctx.kv_block_allocator.pool_size == expected_pool_size
        assert ctx.mamba_slot_allocator.max_slots == durable_slots
        assert ctx.mamba_slot_allocator.conv_states.dtype == getattr(torch, dtype_name)
        assert ctx.mamba_slot_allocator.ssm_states.dtype == getattr(torch, dtype_name)
        return ctx

    @staticmethod
    def _store_distinct_states(ctx, block_ids, slots, generation):
        msa = ctx.mamba_slot_allocator
        slot_tensor = torch.tensor(slots, dtype=torch.int64, device=msa.conv_states.device)
        conv_values = generation * 16 + torch.arange(
            1, len(slots) + 1, dtype=msa.conv_states.dtype, device=msa.conv_states.device
        )
        ssm_values = (
            generation * 16
            + 64
            + torch.arange(
                1, len(slots) + 1, dtype=msa.ssm_states.dtype, device=msa.ssm_states.device
            )
        )
        conv_expected = conv_values.view(1, -1, 1, 1).expand(
            msa.num_mamba_layers, -1, *msa.conv_states.shape[2:]
        )
        ssm_expected = ssm_values.view(1, -1, 1, 1).expand(
            msa.num_mamba_layers, -1, *msa.ssm_states.shape[2:]
        )
        msa.conv_states[:, slot_tensor] = conv_expected
        msa.ssm_states[:, slot_tensor] = ssm_expected
        assert torch.equal(msa.conv_states[:, slot_tensor], conv_expected)
        assert torch.equal(msa.ssm_states[:, slot_tensor], ssm_expected)
        return conv_expected.clone(), ssm_expected.clone()

    @pytest.mark.parametrize(
        "case_id,row",
        generated_matrix_cases("mamba_match_depth", _MANIFEST["matrices"]["mamba_match_depth"]),
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_executable_kv_mamba_depths_repeat_real_reuse(self, case_id, row):
        """Every depth repeatedly pins KV and either restores or recomputes Mamba."""
        del case_id
        policy = _POLICIES[row["eviction_policy"]]
        dtype_name = row["mamba_state_dtype"]
        block_size = 64
        max_tokens = block_size * 8
        max_requests = 4
        mamba_config = self._typed_mamba_config(mamba_chunk_size=64, dtype_name=dtype_name)
        mamba_depth = {"memory_only": None, "kv_only": 0, "mamba_shorter": 2, "equal": 3}[
            row["mamba_match_mode"]
        ]
        prefix_caching_mamba_gb = None
        if mamba_depth is not None:
            prefix_caching_mamba_gb = self._mamba_budget_gb(
                durable_slots=4,
                block_size_tokens=block_size,
                max_tokens=max_tokens,
                max_requests=max_requests,
                mamba_config=mamba_config,
            )

        ctx = self._ctx(
            buffer_size_gb=_CAPACITY_BUFFER_GB,
            block_size_tokens=block_size,
            max_sequence_length=block_size * 4,
            rounder=1,
            max_tokens=max_tokens,
            max_requests=max_requests,
            prefix_caching_eviction_policy=policy,
            mamba_config=mamba_config,
            prefix_caching_mamba_gb=prefix_caching_mamba_gb,
        )
        assert ctx.mamba_conv_states.dtype == getattr(torch, dtype_name)
        assert ctx.mamba_ssm_states.dtype == getattr(torch, dtype_name)
        prompt = self._prompt(3 * block_size + 5)
        seed = self._req(ctx, prompt.clone(), request_id=0)
        ctx.add_request(seed)
        seed_blocks = self._block_ids(ctx, 0, 4)
        canonical_blocks = seed_blocks[:3]
        alloc = ctx.kv_block_allocator

        expected_conv = expected_ssm = None
        if mamba_depth:
            msa = ctx.mamba_slot_allocator
            state_blocks = canonical_blocks[:mamba_depth]
            slots = msa.allocate_slots_batch(state_blocks)
            expected_conv, expected_ssm = self._store_distinct_states(
                ctx, state_blocks, slots, generation=1
            )
            state_hashes = [alloc.block_hashes[block_id].item() for block_id in state_blocks]
            msa.register_block_hashes_batch(state_blocks, state_hashes)
            assert msa.hash_to_block_id == dict(zip(state_hashes, state_blocks))

        expected_query_length = len(prompt) - (mamba_depth or 0) * block_size
        expected_ref_counts = alloc.block_ref_counts[canonical_blocks].clone()
        for cycle in range(3):
            follower = self._req(ctx, prompt.clone(), request_id=cycle + 1)
            match = ctx._compute_prefix_match(follower, len(prompt))
            assert match[0] == canonical_blocks
            assert match[4] == (mamba_depth or 0) * block_size
            assert match[5] == expected_query_length

            ctx.add_request(follower)
            assert follower._mamba_num_matched_blocks == (mamba_depth or 0)
            assert ctx.request_query_lengths[1].item() == expected_query_length
            assert torch.equal(alloc.block_ref_counts[canonical_blocks], expected_ref_counts + 1)

            ctx.initialize_attention_state()
            follower_mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[1].item()
            if mamba_depth:
                restore_slot = ctx.mamba_slot_allocator.get_slot(canonical_blocks[mamba_depth - 1])
                assert torch.equal(
                    ctx.mamba_conv_states[:, follower_mamba_idx],
                    ctx.mamba_slot_allocator.conv_states[:, restore_slot],
                )
                assert torch.equal(
                    ctx.mamba_ssm_states[:, follower_mamba_idx],
                    ctx.mamba_slot_allocator.ssm_states[:, restore_slot],
                )
                state_slots = [
                    ctx.mamba_slot_allocator.get_slot(block_id)
                    for block_id in canonical_blocks[:mamba_depth]
                ]
                state_slot_tensor = torch.tensor(
                    state_slots,
                    dtype=torch.int64,
                    device=ctx.mamba_slot_allocator.conv_states.device,
                )
                assert torch.equal(
                    ctx.mamba_slot_allocator.conv_states[:, state_slot_tensor], expected_conv
                )
                assert torch.equal(
                    ctx.mamba_slot_allocator.ssm_states[:, state_slot_tensor], expected_ssm
                )
            else:
                assert torch.count_nonzero(ctx.mamba_conv_states[:, follower_mamba_idx]) == 0
                assert torch.count_nonzero(ctx.mamba_ssm_states[:, follower_mamba_idx]) == 0

            ctx.update_requests(
                torch.tensor([1, 0], dtype=torch.int32),
                torch.tensor([10_000 + cycle, 0], dtype=torch.int64),
            )
            assert ctx.total_request_count == 1
            assert torch.equal(alloc.block_ref_counts[canonical_blocks], expected_ref_counts)

    @pytest.mark.parametrize(
        "case_id,row",
        generated_matrix_cases(
            "mamba_relative_capacity", _MANIFEST["matrices"]["mamba_relative_capacity"]
        ),
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_relative_kv_mamba_capacities_churn_three_generations(self, case_id, row):
        """The independently sized smaller cache becomes the real limiting pool."""
        del case_id
        policy = _POLICIES[row["eviction_policy"]]
        dtype_name = row["mamba_state_dtype"]
        mamba_config = self._typed_mamba_config(mamba_chunk_size=64, dtype_name=dtype_name)
        expected_pool_size = self._expected_kv_pool_size(
            buffer_size_gb=_CAPACITY_BUFFER_GB,
            block_size_tokens=64,
            max_requests=3,
            mamba_config=mamba_config,
        )
        kv_usable = expected_pool_size - 1
        durable_slots = {
            "mamba_limited": kv_usable - 1,
            "balanced": kv_usable,
            "kv_limited": kv_usable + 2,
        }[row["relative_capacity"]]
        ctx = self._ctx_with_slots(
            durable_slots=durable_slots,
            max_tokens=192,
            max_requests=3,
            policy=policy,
            dtype_name=dtype_name,
        )
        alloc = ctx.kv_block_allocator
        msa = ctx.mamba_slot_allocator
        block_ids = alloc.allocate_memory_blocks(kv_usable)
        assert block_ids is not None
        assert alloc.pool_avail == 0
        previous_slot_states = {}

        for generation in range(3):
            bids = block_ids.tolist()
            hashes = [100_000 * (generation + 1) + index + 1 for index in range(kv_usable)]
            parent_hashes = [0, *hashes[:-1]]
            alloc.register_kv_block_hashes(bids, hashes, parent_hashes)

            state_count = min(kv_usable, durable_slots)
            state_blocks = bids[:state_count]
            state_hashes = hashes[:state_count]
            slots = msa.allocate_slots_batch(state_blocks)
            assert msa.hash_to_block_id == {}
            if previous_slot_states:
                assert set(slots) == set(previous_slot_states)
                for slot in slots:
                    previous_conv, previous_ssm = previous_slot_states[slot]
                    assert torch.equal(msa.conv_states[:, slot], previous_conv)
                    assert torch.equal(msa.ssm_states[:, slot], previous_ssm)

            expected_conv, expected_ssm = self._store_distinct_states(
                ctx, state_blocks, slots, generation
            )
            if previous_slot_states:
                for index, slot in enumerate(slots):
                    previous_conv, previous_ssm = previous_slot_states[slot]
                    assert not torch.equal(expected_conv[:, index], previous_conv)
                    assert not torch.equal(expected_ssm[:, index], previous_ssm)
            msa.register_block_hashes_batch(state_blocks, state_hashes)

            assert msa.free_count == max(0, durable_slots - kv_usable)
            assert msa.hash_to_block_id == dict(zip(state_hashes, state_blocks))
            assert alloc.kv_hash_to_block_id == dict(zip(hashes, bids))
            assert torch.equal(
                msa.slot_to_block[
                    torch.tensor(slots, dtype=torch.int64, device=msa.slot_to_block.device)
                ],
                torch.tensor(state_blocks, dtype=torch.int32),
            )

            if row["relative_capacity"] == "mamba_limited":
                mappings_before = msa.block_to_slot.clone()
                with pytest.raises(RuntimeError, match="No evictable Mamba"):
                    msa.allocate_slots_batch([bids[-1]])
                assert torch.equal(msa.block_to_slot, mappings_before)
                assert msa.get_allocatable_slot_count() == 0
            elif row["relative_capacity"] == "balanced":
                assert alloc.allocate_memory_blocks(1) is None
                assert msa.get_allocatable_slot_count() == 0
            else:
                assert row["relative_capacity"] == "kv_limited"
                assert alloc.allocate_memory_blocks(1) is None
                assert msa.get_allocatable_slot_count() == 2

            previous_slot_states = {
                slot: (expected_conv[:, index].clone(), expected_ssm[:, index].clone())
                for index, slot in enumerate(slots)
            }
            alloc.release_memory_blocks(block_ids)
            if policy == PrefixCachingEvictionPolicy.REF_ZERO:
                assert alloc.pool_avail == kv_usable
                assert msa.free_count == durable_slots
            else:
                assert alloc.pool_avail == 0
                assert int(alloc.get_evictable_block_count()) == kv_usable

            if generation < 2:
                block_ids = alloc.allocate_memory_blocks(kv_usable)
                assert block_ids is not None
                assert alloc.physical_block_reuse_count == (generation + 1) * kv_usable
                assert set(block_ids.tolist()) == set(bids)
                assert alloc.kv_hash_to_block_id == {}
                assert msa.hash_to_block_id == {}
                assert msa.free_count == durable_slots

        assert alloc.physical_block_reuse_count == 2 * kv_usable
        if policy == PrefixCachingEvictionPolicy.REF_ZERO:
            assert alloc.deregistered_block_count == 3 * kv_usable

    @pytest.mark.parametrize(
        "policy",
        [PrefixCachingEvictionPolicy.REF_ZERO, PrefixCachingEvictionPolicy.LRU],
        ids=["refzero", "lru"],
    )
    @pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
    def test_pinned_slot_pressure_is_transactional_then_reuses_physical_state(
        self, policy, dtype_name, monkeypatch
    ):
        """Pinned durable bytes survive pressure; one released slot is reused safely."""
        mamba_config = self._typed_mamba_config(mamba_chunk_size=64, dtype_name=dtype_name)
        expected_pool_size = self._expected_kv_pool_size(
            buffer_size_gb=_CAPACITY_BUFFER_GB,
            block_size_tokens=64,
            max_requests=3,
            mamba_config=mamba_config,
        )
        kv_usable = expected_pool_size - 1
        ctx = self._ctx_with_slots(
            durable_slots=2, max_tokens=192, max_requests=3, policy=policy, dtype_name=dtype_name
        )
        alloc = ctx.kv_block_allocator
        msa = ctx.mamba_slot_allocator
        all_blocks = alloc.allocate_memory_blocks(kv_usable)
        assert all_blocks is not None and alloc.pool_avail == 0
        bids = all_blocks.tolist()
        hashes = [800_000 + index for index in range(kv_usable)]
        alloc.register_kv_block_hashes(bids, hashes, [0] * kv_usable)

        anchor_a, anchor_b, candidate = bids[:3]
        slot_a, slot_b = msa.allocate_slots_batch([anchor_a, anchor_b])
        self._store_distinct_states(ctx, [anchor_a, anchor_b], [slot_a, slot_b], generation=1)
        msa.register_block_hashes_batch([anchor_a, anchor_b], hashes[:2])
        anchor_b_conv = msa.conv_states[:, slot_b].clone()
        anchor_b_ssm = msa.ssm_states[:, slot_b].clone()
        mappings_before = msa.block_to_slot.clone()

        for _ in range(3):
            with pytest.raises(RuntimeError, match="No evictable Mamba"):
                msa.allocate_slots_batch([candidate])
            assert msa.free_count == 0
            assert torch.equal(msa.block_to_slot, mappings_before)
            assert torch.equal(msa.conv_states[:, slot_b], anchor_b_conv)
            assert torch.equal(msa.ssm_states[:, slot_b], anchor_b_ssm)

        # Snapshot publication is optional under pressure. Exercise the real
        # commit/preflight path three times and prove it skips rather than
        # consuming or replacing pinned durable state.
        monkeypatch.setattr(
            msa, "_collect_commit_data", lambda: ([], [], [candidate], [0], [hashes[2]])
        )
        for _ in range(3):
            msa.commit_intermediate_states()
            assert msa.commit_count == 0
            assert msa.free_count == 0
            assert torch.equal(msa.block_to_slot, mappings_before)
            assert torch.equal(msa.conv_states[:, slot_b], anchor_b_conv)
            assert torch.equal(msa.ssm_states[:, slot_b], anchor_b_ssm)

        alloc.release_memory_blocks(torch.tensor([anchor_a], dtype=torch.int32))
        existing_slot, candidate_slot, duplicate_slot = msa.allocate_slots_batch(
            [anchor_b, candidate, candidate]
        )
        assert existing_slot == slot_b
        assert duplicate_slot == candidate_slot
        assert candidate_slot == slot_a
        if policy == PrefixCachingEvictionPolicy.LRU:
            assert msa.eviction_count == 1
        msa.register_block_hashes_batch([candidate], [hashes[2]])
        self._store_distinct_states(ctx, [candidate], [candidate_slot], generation=2)

        recycled = alloc.allocate_memory_blocks(1)
        assert recycled is not None and recycled.tolist() == [anchor_a]
        recycled_block = recycled.item()

        alloc.release_memory_blocks(torch.tensor([candidate], dtype=torch.int32))
        recycled_slot = msa.allocate_slots_batch([recycled_block])[0]
        assert recycled_slot == candidate_slot
        if policy == PrefixCachingEvictionPolicy.LRU:
            assert msa.eviction_count == 2

        # Put the candidate KV block back in use without giving it a Mamba slot.
        # This leaves the recycled block as the sole evictable KV block in each
        # generation below, so physical-ID reuse is deterministic under both
        # policies.
        active_candidate = alloc.allocate_memory_blocks(1)
        assert active_candidate is not None and active_candidate.item() == candidate

        for generation in range(3):
            recycled_hash = 900_000 + generation
            alloc.register_kv_block_hashes([recycled_block], [recycled_hash], [0])
            msa.register_block_hashes_batch([recycled_block], [recycled_hash])
            self._store_distinct_states(
                ctx, [recycled_block], [recycled_slot], generation=generation + 3
            )
            assert torch.equal(msa.conv_states[:, slot_b], anchor_b_conv)
            assert torch.equal(msa.ssm_states[:, slot_b], anchor_b_ssm)

            prior_reuse_count = alloc.physical_block_reuse_count
            alloc.release_memory_blocks(torch.tensor([recycled_block], dtype=torch.int32))
            next_generation = alloc.allocate_memory_blocks(1)
            assert next_generation is not None and next_generation.item() == recycled_block
            assert alloc.physical_block_reuse_count == prior_reuse_count + 1
            recycled_slot = msa.allocate_slots_batch([recycled_block])[0]
            assert recycled_slot == candidate_slot

    @pytest.mark.parametrize(
        "case_id,row",
        generated_matrix_cases(
            "mamba_boundary_extraction", _MANIFEST["matrices"]["mamba_boundary_extraction"]
        ),
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_chunk_boundary_extraction_commits_and_restores_three_times(self, case_id, row):
        """Both extraction sources publish real state that a follower restores."""
        del case_id
        policy = _POLICIES[row["eviction_policy"]]
        block_size, mamba_chunk_size = _BOUNDARY_PROFILES[row["mamba_block_chunk_profile"]]
        boundary_kind = row["mamba_boundary_source"]
        scratch_limit = row["mamba_scratch_limit"]
        dtype_name = row["mamba_state_dtype"]
        tail = 5
        prompt_length = 3 * block_size + (tail if boundary_kind == "continuation" else 0)
        if scratch_limit == "tokens":
            max_requests = 4
            max_tokens = 5 * block_size + 2 * tail
        else:
            max_requests = 2
            max_tokens = 10 * block_size
        ctx = self._ctx_with_slots(
            durable_slots=8,
            block_size_tokens=block_size,
            max_tokens=max_tokens,
            max_requests=max_requests,
            policy=policy,
            mamba_chunk_size=mamba_chunk_size,
            dtype_name=dtype_name,
        )
        token_scratch_limit = max(0, math.ceil(max_tokens / block_size) - 1)
        request_scratch_limit = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * max_requests
        expected_scratch_slots = min(token_scratch_limit, request_scratch_limit)
        assert ctx.max_mamba_intermediate_states_per_step == expected_scratch_slots
        assert ctx.mamba_slot_allocator.max_intermediate_count == expected_scratch_slots
        if scratch_limit == "tokens":
            assert token_scratch_limit < request_scratch_limit
        else:
            assert request_scratch_limit < token_scratch_limit

        for cycle in range(3):
            ctx.reset()
            self._saturate_mamba_scratch(
                ctx, expected_scratch_slots, request_id_base=10_000 + cycle * 100
            )
            ctx.reset()
            prompt = self._prompt(prompt_length, offset=cycle * 100_000)
            seed = self._req(ctx, prompt.clone(), request_id=2 * cycle)

            if boundary_kind == "continuation":
                first_chunk_length = 2 * block_size
                ctx.add_request(seed, prefill_chunk_length=first_chunk_length)
                seed.remaining_prompt_tokens = seed.remaining_prompt_tokens[first_chunk_length:]
                seed.finished_chunk_token_count = first_chunk_length
                ctx.chunked_prefill_request_id = seed.request_id
                ctx.initialize_attention_state()
                ctx.update_requests(
                    torch.tensor([1], dtype=torch.int32), torch.tensor([0], dtype=torch.int64)
                )
                assert ctx.total_request_count == 0

                # The final chunk returns to the retained request row exactly as
                # schedule_chunked_prefill does when it reaches the prompt end.
                ctx.chunked_prefill_request_id = -1
                ctx.add_request(seed)
                msa = ctx.mamba_slot_allocator
                assert msa._intermediate_counts_cpu[0].item() == 1
                assert msa._intermediate_offsets_cpu[0, 0].item() == block_size
                expected_depth = 3
                expected_query_length = tail
            else:
                ctx.add_request(seed)
                msa = ctx.mamba_slot_allocator
                assert msa._eos_cache_block_id_cpu[0].item() >= 0
                expected_depth = 3
                expected_query_length = block_size

            ctx.initialize_attention_state()
            ctx.transfer_bookkeeping_to_gpu()
            state_value = 10.0 + cycle
            if ctx.mamba_metadata.intermediate_count:
                count = ctx.mamba_metadata.intermediate_count
                msa.intermediate_conv_out[:, :count] = state_value
                msa.intermediate_ssm_out[:, :count] = state_value + 100
            live_idx = ctx.mamba_metadata.request_to_mamba_state_idx[0].item()
            ctx.mamba_conv_states[:, live_idx] = state_value
            ctx.mamba_ssm_states[:, live_idx] = state_value + 100
            msa.commit_intermediate_states()

            boundary_hash = seed.precomputed_block_hashes[expected_depth - 1]
            assert boundary_hash in msa.hash_to_block_id
            assert msa.commit_count >= 1
            ctx.reset_attention_state()

            follower = self._req(ctx, prompt.clone(), request_id=2 * cycle + 1)
            match = ctx._compute_prefix_match(follower, len(prompt))
            assert len(match[0]) == expected_depth
            assert match[4] == len(prompt) - expected_query_length
            ctx.add_request(follower)
            assert follower._mamba_num_matched_blocks == expected_depth
            assert ctx.request_query_lengths[1].item() == expected_query_length

            ctx.initialize_attention_state()
            follower_idx = ctx.mamba_metadata.request_to_mamba_state_idx[1].item()
            assert torch.all(ctx.mamba_conv_states[:, follower_idx] == state_value)
            assert torch.all(ctx.mamba_ssm_states[:, follower_idx] == state_value + 100)
            assert msa.restore_hit_count == cycle + 1

            ctx.update_requests(
                torch.tensor([1, 0], dtype=torch.int32),
                torch.tensor([20_000 + cycle, 0], dtype=torch.int64),
            )
