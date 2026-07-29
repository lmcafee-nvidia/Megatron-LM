# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Admission-failure stress for matched prefix-cache requests."""

from pathlib import Path

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.dynamic_context import BlockOverflowError
from tests.test_utils.python_scripts.prefix_cache_coverage import (
    generated_matrix_cases,
    load_manifest,
)
from tests.unit_tests.inference.contexts.test_dynamic_prefix_caching import (
    PrefixCachingTestBase as _PrefixCachingTestBase,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = load_manifest(_REPO_ROOT / "tests/test_utils/prefix_cache_coverage.yaml")


def _logical_snapshot(ctx, req):
    """Capture state that remains observable after a rejected admission."""

    allocator = ctx.kv_block_allocator
    current_id = ctx.total_request_count
    result = {
        "pool_avail": allocator.pool_avail,
        "hash_map": dict(allocator.kv_hash_to_block_id),
        "ref_counts": allocator.block_ref_counts.clone(),
        "never_allocated": allocator.never_allocated_count,
        "physical_reuses": allocator.physical_block_reuse_count,
        "deregistered": allocator.deregistered_block_count,
        "lru_evicted": allocator.lru_evicted_block_count,
        "context_counts": (
            ctx.total_request_count,
            ctx.active_token_count,
            ctx.num_prefill_requests,
            ctx.prefix_cache_hits,
            ctx.prefix_cache_blocks_matched,
            ctx.prefix_cache_prefill_computed_tokens,
            ctx.prefix_cache_prefill_skipped_tokens,
        ),
        "request_row": ctx.request_to_kv_block_ids[current_id].clone(),
        "request_cached_tokens": req.num_cached_tokens,
        "request_mamba_match": getattr(req, "_mamba_num_matched_blocks", None),
        "mamba_live_free": ctx.mamba_metadata.mamba_state_free_slot_count,
        "mamba_prefix_map": dict(ctx.mamba_slot_allocator.hash_to_block_id),
        "pending_mamba_zeros": list(ctx._pending_mamba_zeros),
        "pending_mamba_restores": list(ctx._pending_mamba_restores),
    }
    if allocator.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
        result["timestamps"] = allocator.block_timestamps.clone()
    return result


class TestPrefixCacheAdmissionRollback(_PrefixCachingTestBase):
    """A real capacity rejection must not partially admit a matching request."""

    def _build_exhausted_context(self, policy):
        ctx = self._ctx(
            buffer_size_gb=0.002,
            block_size_tokens=256,
            max_sequence_length=1024,
            rounder=1,
            max_tokens=1024,
            max_requests=4,
            prefix_caching_eviction_policy=policy,
            mamba_config=self._mamba_config(),
            prefix_caching_mamba_gb=0.001,
        )
        block_size = ctx.block_size_tokens
        seed_prompt = self._prompt(2 * block_size + 1)
        seed = self._req(ctx, seed_prompt, request_id=1)
        ctx.add_request(seed)
        seed_blocks = self._block_ids(ctx, 0, 3)

        # Publish executable Mamba state at the first boundary. The follower
        # shares this block, diverges at block two, and therefore needs two new
        # blocks after skipping one.
        mamba_allocator = ctx.mamba_slot_allocator
        mamba_allocator.allocate_slots_batch(seed_blocks[:1])
        mamba_allocator.register_block_hashes_batch(
            seed_blocks[:1], seed.precomputed_block_hashes[:1]
        )

        allocator = ctx.kv_block_allocator
        if policy == PrefixCachingEvictionPolicy.LRU:
            ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
            # Remove the cached second-block child, leaving only the first block
            # as the follower's match. Its returned storage is consumed below.
            assert allocator.evict_lru_blocks(1)
            ctx.reset_metadata(preserve_prefix_cache=True)
            assert set(allocator.kv_hash_to_block_id) == {seed.precomputed_block_hashes[0]}

        pressure_blocks = allocator.allocate_memory_blocks(allocator.pool_avail)
        assert pressure_blocks is not None
        assert allocator.pool_avail == 0

        follower_prompt = seed_prompt.clone()
        follower_prompt[block_size:] += 10_000
        return ctx, follower_prompt, pressure_blocks, seed_blocks[0]

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "case_id,row",
        generated_matrix_cases("admission_rollback", _MANIFEST["matrices"]["admission_rollback"]),
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_exhausted_match_pin_rolls_back_before_retry(self, case_id, row):
        """Repeat real match/pin/capacity failures, then fund and admit the request."""

        del case_id
        assert row["pressure_shape"] == "exhausted"
        policy = {
            "ref_zero": PrefixCachingEvictionPolicy.REF_ZERO,
            "lru": PrefixCachingEvictionPolicy.LRU,
        }[row["eviction_policy"]]
        for cycle in range(3):
            ctx, follower_prompt, pressure_blocks, matched_block = self._build_exhausted_context(
                policy
            )
            follower = self._req(ctx, follower_prompt, request_id=100 + cycle)
            before = _logical_snapshot(ctx, follower)

            with pytest.raises(BlockOverflowError):
                ctx.add_request(follower)

            after = _logical_snapshot(ctx, follower)
            assert after.keys() == before.keys()
            for key in before:
                if isinstance(before[key], torch.Tensor):
                    assert torch.equal(after[key], before[key]), key
                else:
                    assert after[key] == before[key], key

            # Supply exactly the two blocks the divergent tail needs and retry
            # the same request through the real KV+Mamba prefix-hit path.
            ctx.kv_block_allocator.release_memory_blocks(pressure_blocks[:2])
            ctx.add_request(follower)

            assert follower.num_cached_tokens == ctx.block_size_tokens
            assert follower._mamba_num_matched_blocks == 1
            assert ctx.prefix_cache_hits == 1
            assert ctx.prefix_cache_blocks_matched == 1
            assert ctx.request_query_lengths[ctx.total_request_count - 1].item() == (
                ctx.block_size_tokens + 1
            )
            expected_ref_count = 2 if policy == PrefixCachingEvictionPolicy.REF_ZERO else 1
            assert (
                ctx.kv_block_allocator.block_ref_counts[matched_block].item() == expected_ref_count
            )
            assert len(ctx._pending_mamba_restores) == 1
