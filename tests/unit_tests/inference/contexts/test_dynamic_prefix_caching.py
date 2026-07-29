# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
from collections import deque

import numpy as np
import pytest
import torch

from megatron.core.inference.config import (
    InferenceConfig,
    KVCacheManagementMode,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.contexts.dynamic_context import (
    DynamicInferenceContext,
    RequestOverflowError,
    TokenOverflowError,
)
from megatron.core.inference.contexts.mamba_slot_allocator import (
    MAX_INTERMEDIATE_OFFSETS_PER_REQUEST,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class PrefixCachingTestBase:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def _mamba_config(mamba_chunk_size=128):
        from megatron.core.inference.config import MambaInferenceStateConfig

        return MambaInferenceStateConfig(
            layer_type_list=["*", "M", "*", "M"],
            conv_states_shape=(4, 8),
            ssm_states_shape=(4, 16),
            conv_states_dtype=torch.float32,
            ssm_states_dtype=torch.float32,
            mamba_chunk_size=mamba_chunk_size,
        )

    def _ctx(
        self,
        *,
        buffer_size_gb=0.1,
        block_size_tokens=32,
        max_sequence_length=512,
        rounder=64,
        enable_prefix_caching=True,
        max_tokens=None,
        max_requests=None,
        num_speculative_tokens=0,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
        mamba_config=None,
        prefix_caching_mamba_gb=None,
        kv_cache_management_mode=KVCacheManagementMode.PERSIST,
        static_kv_memory_pointers=False,
        unified_memory_level=0,
    ):
        DynamicInferenceContext.ROUNDER = rounder
        DynamicInferenceContext.TOKEN_ROUNDER = rounder
        DynamicInferenceContext.REQUEST_ROUNDER = rounder

        transformer_config = TransformerConfig(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            hidden_size=16,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_cpu_initialization=True,
        )
        inference_config = InferenceConfig(
            max_sequence_length=max_sequence_length,
            buffer_size_gb=buffer_size_gb,
            paused_buffer_size_gb=0.2 * buffer_size_gb,
            block_size_tokens=block_size_tokens,
            max_tokens=max_tokens,
            max_requests=max_requests,
            num_speculative_tokens=num_speculative_tokens,
            mamba_inference_state_config=mamba_config,
            use_flashinfer_fused_rope=None,
            unified_memory_level=unified_memory_level,
            kv_cache_management_mode=kv_cache_management_mode,
            static_kv_memory_pointers=static_kv_memory_pointers,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_eviction_policy=prefix_caching_eviction_policy,
            prefix_caching_mamba_gb=prefix_caching_mamba_gb,
        )
        return DynamicInferenceContext(
            model_config=transformer_config, inference_config=inference_config
        )

    @staticmethod
    def _req(ctx, prompt_tokens, request_id=1, *, enable_prefix_caching=True, sampling_params=None):
        return DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            sampling_params=(
                sampling_params
                if sampling_params is not None
                else SamplingParams(num_tokens_to_generate=10)
            ),
            block_size_tokens=ctx.block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
        )

    @staticmethod
    def _prompt(num_tokens, offset=0):
        return torch.arange(offset, offset + num_tokens, device=torch.cuda.current_device())

    @staticmethod
    def _block_ids(ctx, req_idx, n):
        return [ctx.request_to_kv_block_ids[req_idx][i].item() for i in range(n)]

    @staticmethod
    def _fill_pool_with_one_evictable_block(ctx):
        """Exhaust the free pool while leaving exactly one LRU block evictable."""
        alloc = ctx.kv_block_allocator
        drained_block_ids = alloc.allocate_memory_blocks(alloc.pool_avail)
        assert drained_block_ids is not None and drained_block_ids.numel() > 0

        cached_block_id = drained_block_ids[0].item()
        cached_hash = 1
        while cached_hash in alloc.kv_hash_to_block_id:
            cached_hash += 1
        alloc.register_kv_block_hashes([cached_block_id], [cached_hash], parent_hashes=[0])
        alloc.release_memory_blocks(drained_block_ids[:1])

        assert alloc.pool_avail == 0
        assert alloc.get_allocatable_count() == 1
        assert int(alloc.get_evictable_block_count()) == 1
        return cached_block_id, cached_hash

    @staticmethod
    def _mamba_allocate_and_register(ctx, bids):
        """Allocate Mamba cache slots and register hashes for a list of block IDs."""
        msa = ctx.mamba_slot_allocator
        alloc = ctx.kv_block_allocator
        slots = msa.allocate_slots_batch(bids)
        bid_tensor = torch.tensor(bids, dtype=torch.int64, device=alloc.block_hashes.device)
        hashes = alloc.block_hashes[bid_tensor].tolist()
        msa.register_block_hashes_batch(bids, hashes)
        return slots

    def _saturate_mamba_scratch(self, ctx, expected_count, *, request_id_base):
        """Fill every reachable extraction slot through real request admission."""
        assert 0 < expected_count <= MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * ctx.max_requests
        block_size = ctx.block_size_tokens
        two_offset_count, one_offset_count = divmod(expected_count, 2)
        requests = []

        for index in range(two_offset_count):
            prompt = self._prompt(
                2 * block_size + 1, offset=request_id_base * 10_000 + index * 1_000
            )
            requests.append(self._req(ctx, prompt, request_id=request_id_base + index))

        for index in range(one_offset_count):
            prompt = self._prompt(
                block_size + 1, offset=request_id_base * 10_000 + (two_offset_count + index) * 1_000
            )
            requests.append(
                self._req(ctx, prompt, request_id=request_id_base + two_offset_count + index)
            )

        # A one-block KV-only match supplies the second distinct extraction
        # boundary for each long request without skipping any Mamba execution.
        seed_blocks = ctx.kv_block_allocator.allocate_memory_blocks(two_offset_count)
        assert seed_blocks is not None
        if two_offset_count:
            seed_hashes = [
                request.precomputed_block_hashes[0] for request in requests[:two_offset_count]
            ]
            ctx.kv_block_allocator.register_kv_block_hashes(
                seed_blocks.tolist(), seed_hashes, [0] * two_offset_count
            )

        for request in requests:
            ctx.add_request(request)

        ctx.initialize_attention_state()
        ctx.transfer_bookkeeping_to_gpu()
        expected_per_request = [2] * two_offset_count + [1] * one_offset_count
        metadata = ctx.mamba_metadata
        assert metadata.per_request_intermediate_counts == expected_per_request
        assert metadata.intermediate_count == expected_count

        allocator = ctx.mamba_slot_allocator
        assert allocator.hash_to_block_id == {}
        commits_before = allocator.commit_count
        allocator.intermediate_conv_out[:, :expected_count].fill_(request_id_base + 1)
        allocator.intermediate_ssm_out[:, :expected_count].fill_(request_id_base + 101)
        allocator.commit_intermediate_states()
        assert allocator.commit_count == commits_before + expected_count
        assert len(allocator.hash_to_block_id) == expected_count

        cached_blocks = list(allocator.hash_to_block_id.values())
        cached_slots = [allocator.get_slot(block_id) for block_id in cached_blocks]
        assert all(
            torch.all(allocator.conv_states[:, slot] == request_id_base + 1)
            for slot in cached_slots
        )
        assert all(
            torch.all(allocator.ssm_states[:, slot] == request_id_base + 101)
            for slot in cached_slots
        )


class _StubEngine(DynamicInferenceEngine):

    def __init__(self, context: DynamicInferenceContext, *, enable_chunked_prefill=False):
        self.context = context
        self.enable_chunked_prefill = enable_chunked_prefill
        self.cuda_graph_all_prefills = False
        self._prefix_coordination_waits = 0
        self._prefix_cache_hits = 0
        self._prefix_cache_blocks_matched = 0
        self._prefill_tokens_computed = 0
        self._prefill_tokens_skipped = 0
        self._loop = asyncio.new_event_loop()
        self.waiting_request_ids: deque = deque()
        self.requests = {}
        self._generation_epoch = None


class TestPrefixCachingCore(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_registration_and_discovery(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator

        # blocks discoverable after add_request
        prompt = self._prompt(bs * 2)
        req = self._req(ctx, prompt)
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)
        h0, h1 = req.precomputed_block_hashes
        assert alloc.kv_hash_to_block_id.get(h0) == b0
        assert alloc.kv_hash_to_block_id.get(h1) == b1
        assert alloc.block_hashes[b0].item() == h0 and alloc.block_hashes[b1].item() == h1

        # partial block not registered
        ctx2 = self._ctx()
        alloc2 = ctx2.kv_block_allocator
        ctx2.add_request(self._req(ctx2, self._prompt(bs + bs // 2)))
        pb0, pb1 = self._block_ids(ctx2, 0, 2)
        assert alloc2.block_hashes[pb0].item() != -1
        assert alloc2.block_hashes[pb1].item() == -1

        # decode does not register completed blocks
        ctx3 = self._ctx()
        alloc3 = ctx3.kv_block_allocator
        ctx3.add_request(self._req(ctx3, self._prompt(bs + (bs - 1))))
        db0, db1 = self._block_ids(ctx3, 0, 2)
        assert alloc3.block_hashes[db0].item() != -1 and alloc3.block_hashes[db1].item() == -1
        active_mask = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        ctx3.update_requests(active_mask, new_tokens)
        assert alloc3.block_hashes[db1].item() == -1

        # second request finds registered blocks
        ctx4 = self._ctx()
        alloc4 = ctx4.kv_block_allocator
        p4 = self._prompt(bs * 3)
        ctx4.add_request(self._req(ctx4, p4.clone()))
        req2 = self._req(ctx4, p4.clone(), request_id=2)
        for h in req2.precomputed_block_hashes:
            assert h in alloc4.kv_hash_to_block_id

    @pytest.mark.internal
    def test_block_sharing_patterns(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator

        # N=10 identical prompts share all blocks
        prompt = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 3)
        avail_after_first = alloc.pool_avail
        for i in range(2, 11):
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=i))
        assert alloc.pool_avail == avail_after_first
        for req_idx in range(1, 10):
            assert self._block_ids(ctx, req_idx, 3) == first_blocks
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 10

        # divergent suffix shares common prefix
        ctx2 = self._ctx()
        alloc2 = ctx2.kv_block_allocator
        p1 = self._prompt(bs * 3)
        ctx2.add_request(self._req(ctx2, p1))
        r1 = self._block_ids(ctx2, 0, 3)
        p2 = p1.clone()
        p2[bs * 2 :] += 1000
        ctx2.add_request(self._req(ctx2, p2, request_id=2))
        r2 = self._block_ids(ctx2, 1, 3)
        assert r2[0] == r1[0] and r2[1] == r1[1] and r2[2] != r1[2]
        assert alloc2.block_ref_counts[r1[0]].item() == 2
        assert alloc2.block_ref_counts[r1[2]].item() == 1

        # broken chain stops sharing: [X,W,Z] vs [X,Y,Z]
        ctx3 = self._ctx()
        alloc3 = ctx3.kv_block_allocator
        p3a = self._prompt(bs * 3)
        ctx3.add_request(self._req(ctx3, p3a))
        r3a = self._block_ids(ctx3, 0, 3)
        p3b = p3a.clone()
        p3b[bs : bs * 2] += 5000
        ctx3.add_request(self._req(ctx3, p3b, request_id=2))
        r3b = self._block_ids(ctx3, 1, 3)
        assert r3b[0] == r3a[0] and r3b[1] != r3a[1] and r3b[2] != r3a[2]
        assert alloc3.block_ref_counts[r3a[0]].item() == 2

    @pytest.mark.internal
    def test_prefill_token_savings(self):
        bs = 32

        # enabled vs disabled – use a non-block-aligned prompt so the second
        # request's effective prefill chunk after prefix skipping is > 1, which
        # avoids the single-token-chunk clamp in _compute_prefix_match.
        tail = 5
        ctx_on = self._ctx()
        prompt = self._prompt(bs * 4 + tail)
        ctx_on.add_request(self._req(ctx_on, prompt.clone()))
        ctx_on.add_request(self._req(ctx_on, prompt.clone(), request_id=2))
        ctx_off = self._ctx(enable_prefix_caching=False)
        ctx_off.add_request(self._req(ctx_off, prompt.clone(), enable_prefix_caching=False))
        ctx_off.add_request(
            self._req(ctx_off, prompt.clone(), request_id=2, enable_prefix_caching=False)
        )
        # With caching: first request prefills all tokens, second skips 4 full blocks.
        assert ctx_on.lifetime_prefill_token_count == (bs * 4 + tail) + tail
        assert ctx_off.lifetime_prefill_token_count == (bs * 4 + tail) * 2

        # partial match reduces proportionally
        ctx2 = self._ctx()
        p2a = self._prompt(bs * 3)
        ctx2.add_request(self._req(ctx2, p2a.clone()))
        p2b = p2a.clone()
        p2b[bs * 2 :] += 1000
        ctx2.add_request(self._req(ctx2, p2b, request_id=2))
        assert ctx2.lifetime_prefill_token_count == bs * 3 + bs

        # full match: duplicates skip all full cached blocks
        tail = 5
        ctx3 = self._ctx()
        alloc3 = ctx3.kv_block_allocator
        p3 = self._prompt(bs * 3 + tail)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        tokens_after = ctx3.active_token_count
        first_blocks = self._block_ids(ctx3, 0, 3)
        for i in range(5):
            ctx3.add_request(self._req(ctx3, p3.clone(), request_id=i + 2))
            assert ctx3.request_query_lengths[i + 1].item() == tail
        assert ctx3.active_token_count - tokens_after == 5 * tail
        for bid in first_blocks:
            assert alloc3.block_ref_counts[bid].item() == 6
        assert ctx3.lifetime_prefill_token_count == (bs * 3 + tail) + 5 * tail

        # no match: full prompt added
        ctx4 = self._ctx()
        ctx4.add_request(self._req(ctx4, self._prompt(bs * 2)))
        ctx4.add_request(self._req(ctx4, self._prompt(bs * 2, offset=9000), request_id=2))
        assert ctx4.lifetime_prefill_token_count == bs * 2 + bs * 2

    @pytest.mark.internal
    @pytest.mark.parametrize("top_n_logprobs", [0, 5])
    def test_prompt_logprobs_recompute_without_mutating_discoverable_prefixes(self, top_n_logprobs):
        """Prompt-logprob requests execute every prompt token under cache churn."""
        ctx = self._ctx()
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 3 + 5)

        seed = self._req(ctx, prompt.clone())
        ctx.add_request(seed)
        seed_blocks = set(self._block_ids(ctx, 0, 4))
        seed_hash_map = dict(alloc.kv_hash_to_block_id)
        avail_after_seed = alloc.pool_avail

        sampling_params = SamplingParams(
            num_tokens_to_generate=10,
            return_log_probs=True,
            skip_prompt_log_probs=False,
            top_n_logprobs=top_n_logprobs,
        )
        follower = self._req(ctx, prompt.clone(), request_id=2, sampling_params=sampling_params)
        match = ctx._compute_prefix_match(follower, len(prompt))
        assert match[0] == []
        assert match[4] == 0
        assert match[5] == len(prompt)

        ctx.add_request(follower)
        follower_blocks = set(self._block_ids(ctx, 1, 4))
        assert follower.num_cached_tokens == 0
        assert ctx.request_query_lengths[1].item() == len(prompt)
        assert follower_blocks.isdisjoint(seed_blocks)
        assert alloc.pool_avail == avail_after_seed - 4
        assert alloc.kv_hash_to_block_id == seed_hash_map
        assert all(alloc.block_hashes[block_id].item() == -1 for block_id in follower_blocks)

    @pytest.mark.internal
    def test_block_allocation_with_prefix(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator

        # matched blocks not allocated from pool
        prompt = self._prompt(bs * 4)
        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 4)
        avail = alloc.pool_avail
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert self._block_ids(ctx, 1, 4) == first_blocks and alloc.pool_avail == avail

        # extended prompt allocates only new blocks
        ctx2 = self._ctx()
        alloc2 = ctx2.kv_block_allocator
        p2a = self._prompt(bs * 3)
        ctx2.add_request(self._req(ctx2, p2a))
        avail2 = alloc2.pool_avail
        p2b = torch.cat([p2a, self._prompt(bs * 2, offset=1000)])
        ctx2.add_request(self._req(ctx2, p2b, request_id=2))
        assert alloc2.pool_avail == avail2 - 2

        # check_availability accounts for prefix match
        ctx3 = self._ctx(buffer_size_gb=0.01, rounder=1)
        alloc3 = ctx3.kv_block_allocator
        p3 = self._prompt(ctx3.block_size_tokens * 2)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        while alloc3.pool_avail > 0:
            alloc3.allocate_memory_blocks(1)
        _, _, kv_available = ctx3.check_availability(self._req(ctx3, p3.clone(), request_id=2))
        assert kv_available

    @pytest.mark.internal
    def test_ref_count_lru(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator
        prompt = self._prompt(bs * 2)

        # decrement preserves cached blocks
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        assert alloc.block_ref_counts[b0].item() == 2
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1 and b0_hash in alloc.kv_hash_to_block_id
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0 and b0_hash in alloc.kv_hash_to_block_id

        # cached blocks reused by new request
        ctx2 = self._ctx()
        alloc2 = ctx2.kv_block_allocator
        p2 = self._prompt(bs * 2)
        ctx2.add_request(self._req(ctx2, p2.clone()))
        cb0, cb1 = self._block_ids(ctx2, 0, 2)
        ctx2.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx2.total_request_count = 0
        assert alloc2.block_ref_counts[cb0].item() == 0
        ctx2.add_request(self._req(ctx2, p2.clone(), request_id=2))
        assert self._block_ids(ctx2, 0, 2) == [cb0, cb1]
        assert alloc2.block_ref_counts[cb0].item() == 1

        # eviction frees oldest cached first
        ctx3 = self._ctx(buffer_size_gb=0.01, rounder=1)
        alloc3 = ctx3.kv_block_allocator
        ctx3.add_request(self._req(ctx3, self._prompt(bs * 2)))
        active_blocks = ctx3.request_to_kv_block_ids[0][:2].clone()
        ctx3.add_request(self._req(ctx3, self._prompt(bs * 2, offset=5000), request_id=2))
        ctx3.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        ctx3.total_request_count = 1
        for i in range(20):
            try:
                ctx3.add_request(
                    self._req(
                        ctx3, self._prompt(bs * 2, offset=(i + 10) * 1000), request_id=i + 100
                    )
                )
            except Exception:
                break
        for bid in active_blocks:
            assert alloc3.block_ref_counts[bid.item()].item() == 1

    @pytest.mark.internal
    def test_add_request_full_cache_partial_hit_pins_matched_blocks(self):
        """On a partial prefix hit against a FULL cache, the matched
        blocks must be pinned before allocation so LRU eviction cannot reclaim
        one of them for the new (non-matched) block.

        Scenario (mirrors the descendant-first LRU edge case): a cached chain
        H0/S0 -> H1/S1 (older) plus an unrelated cached root HX/SX (newer) fill
        the pool. An incoming prompt H0 -> H1 -> H2 matches [S0, S1] and needs one
        new block. If S0/S1 are not pinned first, descendant-first LRU evicts the
        older leaf S1 and immediately reuses it, yielding block_table [S0, S1, S1]
        and a dangling H2 -> missing H1 chain. Correct behavior evicts SX and
        yields [S0, S1, SX] with a contiguous H0 -> H1 -> H2 chain.
        """
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator

        # Cached chain H0/S0 -> H1/S1, seeded with an OLD timestamp.
        ctx.prefix_cache_lru_clock = 1
        req_chain = self._req(ctx, self._prompt(bs * 2))
        ctx.add_request(req_chain)
        s0, s1 = self._block_ids(ctx, 0, 2)
        h0, h1 = req_chain.precomputed_block_hashes[0], req_chain.precomputed_block_hashes[1]

        # Unrelated cached root HX/SX, seeded with a NEWER timestamp, so a naive
        # oldest-first / descendant-first eviction would prefer the chain leaf.
        ctx.prefix_cache_lru_clock = 10
        ctx.add_request(self._req(ctx, self._prompt(bs, offset=9000), request_id=2))
        (sx,) = self._block_ids(ctx, 1, 1)

        # All three slots are distinct and now cached (ref_count drops to 0).
        assert len({s0, s1, sx}) == 3
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0, 1]))
        ctx.total_request_count = 0
        assert alloc.block_ref_counts[s0].item() == 0
        assert alloc.block_ref_counts[s1].item() == 0
        assert alloc.block_ref_counts[sx].item() == 0

        # Force a full pool: the new block for H2 can only come from eviction.
        alloc.pool_avail = 0

        # Incoming prompt H0 -> H1 -> H2: first two blocks match the cached chain,
        # the third (H2) is new and must trigger a single eviction.
        ctx.prefix_cache_lru_clock = 20
        req_new = self._req(ctx, self._prompt(bs * 3), request_id=3)
        ctx.add_request(req_new)
        h2 = req_new.precomputed_block_hashes[2]

        block_table = self._block_ids(ctx, 0, 3)

        # Matched blocks are preserved and SX (the unrelated root) is evicted/reused.
        assert block_table == [s0, s1, sx]
        # All three block IDs are distinct — no duplicate from a reclaimed match.
        assert len(set(block_table)) == 3
        # Matched blocks stay pinned for the new request.
        assert alloc.block_ref_counts[s0].item() == 1
        assert alloc.block_ref_counts[s1].item() == 1
        assert alloc.block_ref_counts[sx].item() == 1
        # Contiguous H0 -> H1 -> H2 hash chain over [S0, S1, SX].
        assert alloc.block_hashes[s0].item() == h0
        assert alloc.block_hashes[s1].item() == h1
        assert alloc.block_hashes[sx].item() == h2
        # Parent bookkeeping is stored as resolved block ids: S1's parent is S0
        # and SX's parent is S1 along the H0 -> H1 -> H2 chain.
        assert alloc.block_parent_id[s1].item() == s0
        assert alloc.block_parent_id[sx].item() == s1
        assert alloc.kv_hash_to_block_id[h1] == s1

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "error_type,ctx_kwargs",
        [
            pytest.param(
                RequestOverflowError, {"max_requests": 1, "max_tokens": 128}, id="request-capacity"
            ),
            pytest.param(
                TokenOverflowError, {"max_requests": 2, "max_tokens": 40}, id="token-capacity"
            ),
        ],
    )
    def test_failed_admission_does_not_pin_allocate_or_publish_hashes(self, error_type, ctx_kwargs):
        """Repeated failed admissions are transactionally invisible to the cache."""
        ctx = self._ctx(rounder=1, **ctx_kwargs)
        alloc = ctx.kv_block_allocator
        prompt = self._prompt(ctx.block_size_tokens)
        ctx.add_request(self._req(ctx, prompt.clone()))

        expected_pool_avail = alloc.pool_avail
        expected_ref_counts = alloc.block_ref_counts.clone()
        expected_hash_map = dict(alloc.kv_hash_to_block_id)
        expected_hits = ctx.prefix_cache_hits
        expected_matched = ctx.prefix_cache_blocks_matched
        expected_request_table = ctx.request_to_kv_block_ids.clone()

        for request_id in range(2, 7):
            with pytest.raises(error_type):
                ctx.add_request(self._req(ctx, prompt.clone(), request_id=request_id))

            assert alloc.pool_avail == expected_pool_avail
            assert torch.equal(alloc.block_ref_counts, expected_ref_counts)
            assert alloc.kv_hash_to_block_id == expected_hash_map
            assert ctx.prefix_cache_hits == expected_hits
            assert ctx.prefix_cache_blocks_matched == expected_matched
            assert torch.equal(ctx.request_to_kv_block_ids, expected_request_table)

    @pytest.mark.internal
    def test_check_availability_excludes_already_pinned_matches(self):
        """check_availability reserves only matched blocks that are currently
        evictable (ref_count == 0). A matched prefix already pinned by an
        in-flight request frees no capacity when re-pinned, so reserving it would
        under-report availability and needlessly defer shared-prefix requests."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator

        # Request A stays active, pinning the shared prefix H0/S0 -> H1/S1.
        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        s0, s1 = self._block_ids(ctx, 0, 2)
        assert alloc.block_ref_counts[s0].item() == 1
        assert alloc.block_ref_counts[s1].item() == 1

        # One unrelated block is cached and evictable (ref_count == 0).
        ctx.add_request(self._req(ctx, self._prompt(bs, offset=9000), request_id=2))
        (sx,) = self._block_ids(ctx, 1, 1)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[sx].item() == 0
        assert int(alloc.get_evictable_block_count()) == 1

        # Free pool exhausted: the one new block B needs (H2) can only come from
        # evicting SX. The already-pinned matches S0/S1 must not be reserved.
        alloc.pool_avail = 0

        # Request B shares H0/H1 with A and needs one new block for H2.
        req_b = self._req(ctx, self._prompt(bs * 3), request_id=3)
        matched, num_from_pool, *_ = ctx._compute_prefix_match(req_b, req_b.remaining_prompt_length)
        assert matched == [s0, s1]
        assert num_from_pool == 1

        _, _, kv_cache_available = ctx.check_availability(req_b)
        # SX (the sole evictable block) can satisfy H2; reserving the pinned
        # matches would wrongly report the request as un-addable.
        assert kv_cache_available is True

    @pytest.mark.internal
    def test_resume_boundary_crossing_evicts_lru_block_when_free_pool_empty(self):
        """A boundary-crossing request resumes when only LRU capacity remains."""
        ctx = self._ctx(buffer_size_gb=0.01, rounder=1)
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens

        ctx.add_request(self._req(ctx, self._prompt(bs)))
        original_block_id = self._block_ids(ctx, 0, 1)[0]
        assert ctx.request_last_kv_block_offset[0].item() == bs - 1
        assert alloc.block_ref_counts[original_block_id].item() == 1

        cached_block_id, cached_hash = self._fill_pool_with_one_evictable_block(ctx)

        result = ctx.update_requests(
            torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32),
            torch.tensor([123], device=torch.cuda.current_device()),
        )

        new_block_id = ctx.request_last_kv_block_id[0].item()
        assert result["newly_paused_request_ids"].numel() == 0
        assert result["evict_request_ids"] is None
        assert ctx.paused_request_count == 0
        assert ctx.total_request_count == 1
        assert ctx.request_kv_block_counts[0].item() == 2
        assert new_block_id == cached_block_id
        assert new_block_id != original_block_id
        assert cached_hash not in alloc.kv_hash_to_block_id
        assert alloc.block_hashes[new_block_id].item() == -1
        assert alloc.block_ref_counts[original_block_id].item() == 1
        assert alloc.block_ref_counts[new_block_id].item() == 1
        assert alloc.pool_avail == 0
        assert alloc.get_allocatable_count() == 0
        assert int(alloc.get_evictable_block_count()) == 0
        assert ctx.token_to_block_idx[0].item() == new_block_id

    @pytest.mark.internal
    def test_resume_counts_new_blocks_independently_from_requests(self):
        """With LIFO needs [0, 1, 1], one evictable block resumes exactly two requests."""
        ctx = self._ctx(buffer_size_gb=0.01, rounder=1)
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens

        ctx.add_request(self._req(ctx, self._prompt(bs), request_id=1))
        ctx.add_request(self._req(ctx, self._prompt(bs, offset=1000), request_id=2))
        ctx.add_request(self._req(ctx, self._prompt(bs - 1, offset=2000), request_id=3))
        original_last_block_ids = ctx.request_last_kv_block_id[:3].clone()

        ctx.paused_request_count = 3
        needs_new_block_lifo = ctx.request_last_kv_block_offset[:3].flip(dims=[0]) >= bs - 1
        assert needs_new_block_lifo.tolist() == [False, True, True]

        cached_block_id, cached_hash = self._fill_pool_with_one_evictable_block(ctx)
        active_request_count, newly_paused_request_ids = ctx.resume_paused_requests(0, None)

        assert active_request_count == 2
        assert newly_paused_request_ids is None
        assert ctx.paused_request_count == 1
        assert ctx.total_request_count == 3
        assert ctx.request_kv_block_counts[:3].tolist() == [1, 2, 1]
        assert ctx.request_last_kv_block_id[0].item() == original_last_block_ids[0].item()
        assert ctx.request_last_kv_block_id[1].item() == cached_block_id
        assert ctx.request_last_kv_block_id[2].item() == original_last_block_ids[2].item()
        assert cached_hash not in alloc.kv_hash_to_block_id
        assert alloc.block_ref_counts[cached_block_id].item() == 1
        assert alloc.pool_avail == 0
        assert alloc.get_allocatable_count() == 0
        assert int(alloc.get_evictable_block_count()) == 0

    @pytest.mark.internal
    def test_ref_count_refzero(self):
        bs = 32

        # deregisters on last release
        ctx = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        alloc = ctx.kv_block_allocator
        prompt = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        avail_before = alloc.pool_avail
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1 and b0_hash in alloc.kv_hash_to_block_id
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0 and b0_hash not in alloc.kv_hash_to_block_id
        assert alloc.block_hashes[b0].item() == -1 and alloc.block_hashes[b1].item() == -1
        assert alloc.pool_avail == avail_before + 2

        # released blocks not discoverable
        ctx2 = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        alloc2 = ctx2.kv_block_allocator
        p2 = self._prompt(bs * 2)
        ctx2.add_request(self._req(ctx2, p2.clone()))
        ctx2.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx2.total_request_count = 0
        ctx2.add_request(self._req(ctx2, p2.clone(), request_id=2))
        new_blocks = self._block_ids(ctx2, 0, 2)
        assert alloc2.block_ref_counts[new_blocks[0]].item() == 1


class TestEngineScheduling(PrefixCachingTestBase):

    def _engine(self, ctx, **kwargs):
        return _StubEngine(ctx, **kwargs)

    def _add_to_waiting(self, engine, ctx, req):
        request_id = req.request_id
        engine.requests[request_id] = type(
            "Entry",
            (),
            {
                "record": DynamicInferenceRequestRecord.from_request(req),
                "future": engine._loop.create_future(),
            },
        )()
        req.status = Status.ACTIVE_AND_GENERATING_TOKENS
        req.sampling_params.num_tokens_to_generate = 10
        engine.waiting_request_ids.append(request_id)

    @pytest.mark.internal
    def test_scheduling_deferral_and_resolution(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        # shared prefix defers second request
        req1 = self._req(ctx, prompt.clone())
        self._add_to_waiting(engine, ctx, req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)
        engine.schedule_non_chunked_prefill()
        assert ctx.total_request_count == 1
        assert len(engine.waiting_request_ids) == 1 and engine.waiting_request_ids[0] == 2
        assert engine._prefix_coordination_waits == 1

        # deferred request schedulable after registration (round 2)
        engine.schedule_non_chunked_prefill()
        assert ctx.total_request_count == 2 and len(engine.waiting_request_ids) == 0
        metrics = engine.get_prefix_cache_metrics()
        assert metrics["enabled"]
        assert metrics["hits"] >= 1
        assert metrics["blocks_matched"] >= 1
        assert metrics["kv_blocks_cached"] >= 2

        # skip deferred to schedule non-conflicting
        ctx2 = self._ctx()
        engine2 = self._engine(ctx2)
        pa = self._prompt(bs * 2)
        pb = self._prompt(bs * 2, offset=5000)
        self._add_to_waiting(engine2, ctx2, self._req(ctx2, pa.clone()))
        self._add_to_waiting(engine2, ctx2, self._req(ctx2, pa.clone(), request_id=2))
        self._add_to_waiting(engine2, ctx2, self._req(ctx2, pb.clone(), request_id=3))
        engine2.schedule_non_chunked_prefill()
        assert ctx2.total_request_count == 2
        assert len(engine2.waiting_request_ids) == 1 and engine2.waiting_request_ids[0] == 2

        # registered prefix allows immediate scheduling
        ctx3 = self._ctx()
        engine3 = self._engine(ctx3)
        p3 = self._prompt(bs * 2)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        self._add_to_waiting(engine3, ctx3, self._req(ctx3, p3.clone(), request_id=2))
        engine3.schedule_non_chunked_prefill()
        assert len(engine3.waiting_request_ids) == 0 and ctx3.total_request_count == 2
        assert engine3._prefix_coordination_waits == 0

        # metrics track deferrals
        ctx4 = self._ctx()
        engine4 = self._engine(ctx4)
        p4 = self._prompt(bs * 2)
        assert engine4.get_prefix_coordination_metrics() == {"waits": 0}
        self._add_to_waiting(engine4, ctx4, self._req(ctx4, p4.clone()))
        self._add_to_waiting(engine4, ctx4, self._req(ctx4, p4.clone(), request_id=2))
        engine4.schedule_non_chunked_prefill()
        assert engine4.get_prefix_coordination_metrics() == {"waits": 1}
        engine4.schedule_non_chunked_prefill()
        assert engine4.get_prefix_coordination_metrics() == {"waits": 1}
        assert len(engine4.waiting_request_ids) == 0

    @pytest.mark.internal
    def test_chunked_prefill_deferral(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx, enable_chunked_prefill=True)
        prompt = self._prompt(bs * 2)
        self._add_to_waiting(engine, ctx, self._req(ctx, prompt.clone()))
        self._add_to_waiting(engine, ctx, self._req(ctx, prompt.clone(), request_id=2))
        engine.schedule_chunked_prefill()
        assert ctx.total_request_count == 1
        assert len(engine.waiting_request_ids) == 1 and engine._prefix_coordination_waits == 1

    @pytest.mark.internal
    @pytest.mark.parametrize("model_type", ["gpt", "hybrid"])
    def test_epoch_change_between_prefill_chunks_quarantines_mixed_state(self, model_type):
        """An in-flight prefix must never become reusable across a weight update."""
        is_hybrid = model_type == "hybrid"
        block_size = 256 if is_hybrid else 32
        ctx = self._ctx(
            block_size_tokens=block_size,
            max_sequence_length=block_size * 8,
            max_tokens=block_size,
            rounder=block_size,
            mamba_config=self._mamba_config() if is_hybrid else None,
            prefix_caching_mamba_gb=0.01 if is_hybrid else None,
        )
        engine = self._engine(ctx, enable_chunked_prefill=True)
        engine._generation_epoch = 6
        prompt = self._prompt(block_size * 3)
        in_flight = self._req(ctx, prompt.clone(), request_id=1)
        unadmitted = self._req(ctx, prompt.clone(), request_id=2)
        for request in (in_flight, unadmitted):
            request.set_prefix_cache_namespace(6)
            request.policy_epoch = [(0, 6)]
            request.kv_cache_epoch = [(0, 6)]
        self._add_to_waiting(engine, ctx, in_flight)
        self._add_to_waiting(engine, ctx, unadmitted)

        # Execute the first chunk and then hide it exactly as the normal
        # post-forward bookkeeping does.
        engine.schedule_chunked_prefill()
        assert in_flight.finished_chunk_token_count == block_size
        assert ctx.chunked_prefill_request_id == in_flight.request_id
        assert in_flight.precomputed_block_hashes[0] in ctx.kv_block_allocator.kv_hash_to_block_id
        ctx.update_requests(
            torch.ones(1, dtype=torch.bool, device=torch.cuda.current_device()),
            torch.zeros(1, dtype=torch.long, device=torch.cuda.current_device()),
        )
        assert ctx.total_request_count == 0

        engine._apply_generation_epoch(7)

        # The never-admitted request can safely use the new namespace. The
        # partially executed request must continue uncached because its state
        # already depends on the previous weights.
        assert unadmitted.prefix_cache_namespace == 7
        assert unadmitted.enable_prefix_caching
        assert unadmitted.policy_epoch == [(0, 7)]
        assert unadmitted.kv_cache_epoch == [(0, 7)]
        assert in_flight.prefix_cache_namespace == 6
        assert not in_flight.enable_prefix_caching
        assert in_flight.policy_epoch == [(0, 6), (block_size - 1, 7)]
        assert in_flight.kv_cache_epoch == [(0, 6), (block_size - 1, 7)]
        assert ctx.kv_block_allocator.kv_hash_to_block_id == {}
        if is_hybrid:
            assert ctx.mamba_slot_allocator.hash_to_block_id == {}

        # Execute another chunk under pressure. Neither its completed KV block
        # nor its Mamba snapshot may be published under either epoch.
        engine.schedule_chunked_prefill()
        assert in_flight.finished_chunk_token_count == block_size * 2
        assert ctx.kv_block_allocator.kv_hash_to_block_id == {}
        allocated_blocks = self._block_ids(ctx, 0, 2)
        assert all(
            ctx.kv_block_allocator.block_hashes[block_id].item() == -1
            for block_id in allocated_blocks
        )
        if is_hybrid:
            assert ctx.mamba_slot_allocator.hash_to_block_id == {}
            assert ctx.mamba_slot_allocator._intermediate_counts_cpu[0].item() == 0


class TestMambaPrefixCaching(PrefixCachingTestBase):

    def _mctx(self, **kwargs):
        defaults = dict(
            mamba_config=self._mamba_config(),
            prefix_caching_mamba_gb=0.01,
            block_size_tokens=256,
            max_sequence_length=4096,
        )
        defaults.update(kwargs)
        return self._ctx(**defaults)

    @pytest.mark.internal
    def test_hybrid_memory_only(self):
        # hybrid model: no prefill skipping, but blocks reused for memory savings
        ctx = self._ctx(mamba_config=self._mamba_config())
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator
        prompt = self._prompt(bs * 3)
        assert ctx.is_hybrid_model

        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)
        first_blocks = self._block_ids(ctx, 0, 3)
        avail = alloc.pool_avail
        tokens_after = ctx.active_token_count

        req2 = self._req(ctx, prompt.clone(), request_id=2)
        # no prefill skipping
        (matched, _, _, _, prefix_skip, eff_chunk) = ctx._compute_prefix_match(req2, len(prompt))
        assert len(matched) == 3 and prefix_skip == 0 and eff_chunk == len(prompt)

        ctx.add_request(req2)
        # blocks reused (pool unchanged), ref counts incremented
        assert alloc.pool_avail == avail
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 2
        # all tokens processed (none skipped)
        assert ctx.active_token_count - tokens_after == len(prompt)
        assert ctx.request_kv_length_offsets[1].item() == 0

    @pytest.mark.internal
    def test_mamba_cache_lifecycle(self):
        ctx = self._mctx()
        bs = ctx.block_size_tokens

        # allocated when prefix_caching_mamba_gb is set
        assert ctx.mamba_slot_allocator.max_slots > 0
        assert ctx.mamba_slot_allocator.conv_states is not None
        assert ctx.mamba_slot_allocator.free_count == ctx.mamba_slot_allocator.max_slots

        # not allocated when None
        ctx_none = self._mctx(prefix_caching_mamba_gb=None)
        assert ctx_none.mamba_slot_allocator is None

        # store and restore round-trips
        prompt = self._prompt(bs * 2)
        req = self._req(ctx, prompt.clone())
        ctx.add_request(req)
        block_id = ctx.request_to_kv_block_ids[0][0].item()
        slot = ctx.mamba_slot_allocator.allocate_slots_batch([block_id])[0]
        for layer_idx in range(ctx.num_mamba_layers):
            ssm = torch.ones_like(ctx.mamba_slot_allocator.ssm_states[layer_idx, slot]) * (
                layer_idx + 1
            )
            conv = torch.ones_like(ctx.mamba_slot_allocator.conv_states[layer_idx, slot]) * (
                layer_idx + 10
            )
            ctx.mamba_slot_allocator.store_from_tensors(block_id, layer_idx, ssm, conv)
        assert ctx.mamba_slot_allocator.has_state(block_id)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        ctx.add_request(req2)
        assert ctx.mamba_slot_allocator.restore_to_live(1, block_id)
        mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[1].item()
        for layer_idx in range(ctx.num_mamba_layers):
            assert torch.allclose(
                ctx.mamba_ssm_states[layer_idx, mamba_idx],
                torch.ones_like(ctx.mamba_ssm_states[layer_idx, mamba_idx]) * (layer_idx + 1),
            )

        # invalidate frees slot
        ctx3 = self._mctx()
        p3 = self._prompt(bs * 2)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        bid3 = ctx3.request_to_kv_block_ids[0][0].item()
        ctx3.mamba_slot_allocator.allocate_slots_batch([bid3])
        assert ctx3.mamba_slot_allocator.has_state(bid3)
        free_before = ctx3.mamba_slot_allocator.free_count
        ctx3.mamba_slot_allocator.invalidate_block(bid3)
        assert (
            not ctx3.mamba_slot_allocator.has_state(bid3)
            and ctx3.mamba_slot_allocator.free_count == free_before + 1
        )

        # slot reuse for same block
        ctx4 = self._mctx()
        ctx4.add_request(self._req(ctx4, self._prompt(bs * 2)))
        bid4 = ctx4.request_to_kv_block_ids[0][0].item()
        s1, s2 = ctx4.mamba_slot_allocator.allocate_slots_batch([bid4, bid4])
        assert s1 == s2

        # two-map hash design: kv and mamba maps are independent
        ctx5 = self._mctx()
        alloc5 = ctx5.kv_block_allocator
        p5 = self._prompt(bs * 3)
        ctx5.add_request(self._req(ctx5, p5.clone()))
        msa5 = ctx5.mamba_slot_allocator
        assert len(alloc5.kv_hash_to_block_id) == 3 and len(msa5.hash_to_block_id) == 0
        assert msa5.cache_state_version == 0
        self._mamba_allocate_and_register(ctx5, self._block_ids(ctx5, 0, 3)[:2])
        assert len(alloc5.kv_hash_to_block_id) == 3 and len(msa5.hash_to_block_id) == 2
        assert msa5.cache_state_version == 1
        cached_mamba_hashes = list(msa5.hash_to_block_id)
        cached_mamba_blocks = [
            msa5.hash_to_block_id[block_hash] for block_hash in cached_mamba_hashes
        ]
        msa5.register_block_hashes_batch(cached_mamba_blocks, cached_mamba_hashes)
        assert msa5.cache_state_version == 1
        kv_version_before_epoch = alloc5.cache_state_version
        mamba_free_before_epoch = msa5.free_count
        assert alloc5.invalidate_prefix_cache() == 3
        assert alloc5.cache_state_version == kv_version_before_epoch + 1
        assert msa5.cache_state_version == 2
        assert alloc5.kv_hash_to_block_id == {}
        assert msa5.hash_to_block_id == {}
        assert msa5.free_count == mamba_free_before_epoch + 2

        # find_mamba_match_count
        ctx6 = self._mctx()
        alloc6 = ctx6.kv_block_allocator
        p6 = self._prompt(bs * 4)
        ctx6.add_request(self._req(ctx6, p6.clone()))
        msa6 = ctx6.mamba_slot_allocator
        self._mamba_allocate_and_register(ctx6, self._block_ids(ctx6, 0, 4)[:2])
        req6 = self._req(ctx6, p6.clone(), request_id=2)
        assert ctx6._find_mamba_match_count(req6, 0, len(req6.precomputed_block_hashes)) == 2
        # no match when no mamba hashes registered
        ctx7 = self._mctx()
        ctx7.add_request(self._req(ctx7, self._prompt(bs * 3)))
        req7 = self._req(ctx7, self._prompt(bs * 3), request_id=2)
        assert ctx7._find_mamba_match_count(req7, 0, len(req7.precomputed_block_hashes)) == 0

        # allocate, free, re-allocate
        ctx8 = self._mctx()
        ctx8.add_request(self._req(ctx8, self._prompt(bs * 3)))
        bids8 = self._block_ids(ctx8, 0, 3)
        initial_free = ctx8.mamba_slot_allocator.free_count
        ctx8.mamba_slot_allocator.allocate_slots_batch(bids8)
        assert ctx8.mamba_slot_allocator.free_count == initial_free - 3
        ctx8.mamba_slot_allocator.invalidate_block(bids8[0])
        assert (
            ctx8.mamba_slot_allocator.free_count == initial_free - 2
            and not ctx8.mamba_slot_allocator.has_state(bids8[0])
        )
        ctx8.mamba_slot_allocator.allocate_slots_batch([bids8[0]])
        assert (
            ctx8.mamba_slot_allocator.free_count == initial_free - 3
            and ctx8.mamba_slot_allocator.has_state(bids8[0])
        )

    @pytest.mark.internal
    def test_mamba_prefix_cache_survives_repeated_manual_offload_cycles(self):
        """OFFLOAD manages durable and scratch Mamba cache tensors as one state."""
        ctx = self._mctx(
            buffer_size_gb=0.001,
            prefix_caching_mamba_gb=0.001,
            max_tokens=64,
            max_requests=2,
            kv_cache_management_mode=KVCacheManagementMode.OFFLOAD,
        )
        msa = ctx.mamba_slot_allocator
        tensor_names = (
            "conv_states",
            "ssm_states",
            "_intermediate_offsets_gpu",
            "_intermediate_counts_gpu",
            "intermediate_ssm_out",
            "intermediate_conv_out",
        )
        aliases = {name: f"mamba_prefix_cache_{name.lstrip('_')}" for name in tensor_names}
        for name, context_name in aliases.items():
            assert getattr(ctx, context_name) is getattr(msa, name)
            assert context_name in ctx._offloadable_tensor_names

        msa.conv_states.fill_(1.25)
        msa.ssm_states.fill_(-2.5)
        expected_conv = msa.conv_states.clone()
        expected_ssm = msa.ssm_states.clone()

        for _ in range(3):
            ctx.deallocate_inference_state_buffers()
            assert all(getattr(msa, name).storage().size() == 0 for name in tensor_names)

            ctx.reinitialize_inference_state_buffers()
            assert torch.equal(msa.conv_states, expected_conv)
            assert torch.equal(msa.ssm_states, expected_ssm)

    @pytest.mark.internal
    def test_mamba_prefix_cache_recompute_rebuilds_without_stale_state(self):
        """RECOMPUTE destroys and rebuilds all nested prefix-cache tensors."""
        ctx = self._mctx(
            buffer_size_gb=0.001,
            prefix_caching_mamba_gb=0.001,
            max_tokens=64,
            max_requests=2,
            kv_cache_management_mode=KVCacheManagementMode.RECOMPUTE,
        )

        previous_allocator = ctx.mamba_slot_allocator
        for _ in range(3):
            ctx.deallocate_inference_state_buffers()
            assert ctx.mamba_slot_allocator is None
            assert ctx.kv_block_allocator.on_blocks_deregistered is None

            ctx.reinitialize_inference_state_buffers()
            assert ctx.mamba_slot_allocator is not None
            assert ctx.mamba_slot_allocator is not previous_allocator
            assert ctx.mamba_slot_allocator.hash_to_block_id == {}
            assert ctx.mamba_slot_allocator.free_count == ctx.mamba_slot_allocator.max_slots
            previous_allocator = ctx.mamba_slot_allocator

    @pytest.mark.internal
    def test_mamba_prefill_skip_and_zero_prefill(self):
        # mamba match limits prefill skip
        ctx = self._mctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator
        msa = ctx.mamba_slot_allocator
        prompt = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt.clone()))
        self._mamba_allocate_and_register(ctx, self._block_ids(ctx, 0, 3)[:1])
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        req2._mamba_num_matched_blocks = 1
        (matched, _, _, _, prefix_skip, eff_chunk) = ctx._compute_prefix_match(req2, len(prompt))
        assert len(matched) == 3 and prefix_skip == bs and eff_chunk == len(prompt) - bs

        # no mamba match means no skip
        ctx2 = self._mctx()
        p2 = self._prompt(bs * 3)
        ctx2.add_request(self._req(ctx2, p2.clone()))
        req2b = self._req(ctx2, p2.clone(), request_id=2)
        req2b._mamba_num_matched_blocks = 0
        (m2, _, _, _, ps2, ec2) = ctx2._compute_prefix_match(req2b, len(p2))
        assert len(m2) == 3 and ps2 == 0 and ec2 == len(p2)

        # zero prefill for hybrid (mamba-cached, block-aligned)
        ctx3 = self._mctx()
        p3 = self._prompt(bs * 3)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        self._mamba_allocate_and_register(ctx3, self._block_ids(ctx3, 0, 3))
        req3 = self._req(ctx3, p3.clone(), request_id=2)
        req3._mamba_num_matched_blocks = 3
        (m3, _, _, _, ps3, ec3) = ctx3._compute_prefix_match(req3, len(p3))
        assert len(m3) == 3 and ps3 == 2 * bs and ec3 == bs

        # KV-only prefix skip with non-block-aligned prompt: all 3 full blocks
        # are skipped and only the trailing tokens remain for prefill.
        ctx4 = self._ctx()
        bs4 = ctx4.block_size_tokens
        tail = 5
        p4 = self._prompt(bs4 * 3 + tail)
        req4a = self._req(ctx4, p4.clone())
        ctx4.add_request(req4a)
        req4b = self._req(ctx4, p4.clone(), request_id=2)
        (m4, _, _, _, ps4, ec4) = ctx4._compute_prefix_match(req4b, len(p4))
        assert len(m4) == 3 and ps4 == 3 * bs4 and ec4 == tail
        ctx4.add_request(req4b)

        # KV eviction invalidates mamba
        ctx5 = self._mctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        alloc5 = ctx5.kv_block_allocator
        msa5 = ctx5.mamba_slot_allocator
        p5 = self._prompt(bs * 2)
        ctx5.add_request(self._req(ctx5, p5.clone()))
        bid5 = ctx5.request_to_kv_block_ids[0][0].item()
        bh5 = alloc5.block_hashes[bid5].item()
        self._mamba_allocate_and_register(ctx5, [bid5])
        assert msa5.has_state(bid5) and bh5 in msa5.hash_to_block_id
        ctx5.release_memory_blocks_from_request_indexes([0])
        assert not msa5.has_state(bid5) and bh5 not in msa5.hash_to_block_id

    @pytest.mark.internal
    def test_mamba_intermediate_offsets(self):
        bs = 256

        # KV divergence offsets
        ctx = self._mctx(block_size_tokens=bs)
        prompt = self._prompt(bs * 4)
        ctx.add_request(self._req(ctx, prompt.clone()))
        msa = ctx.mamba_slot_allocator
        self._mamba_allocate_and_register(ctx, self._block_ids(ctx, 0, 4)[:2])
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        req2._mamba_num_matched_blocks = 2
        (matched, _, _, overall, prefix_skip, _) = ctx._compute_prefix_match(req2, len(prompt))
        # Copy block IDs to slot 1 so compute_and_store_offsets can resolve EOS block
        ctx.request_to_kv_block_ids[1] = ctx.request_to_kv_block_ids[0]
        msa.compute_and_store_offsets(
            req2,
            1,
            prefix_skip,
            len(prompt),
            len(matched),
            [ctx.request_to_kv_block_ids[0][i].item() for i in range(len(matched))],
            overall,
        )
        # Penultimate block offset (block 2 boundary) is a valid intermediate
        count = msa._intermediate_counts_cpu[1].item()
        if count > 0:
            offsets = msa._intermediate_offsets_cpu[1, :count].tolist()
            for o in offsets:
                assert o > 0 and o % 128 == 0
        assert msa._eos_cache_block_id_cpu[1].item() >= 0

        # non-aligned prompt produces last_aligned intermediate offset
        ctx2 = self._mctx(block_size_tokens=bs)
        prompt_len = bs * 3 + bs // 2
        p2 = self._prompt(prompt_len)
        ctx2.add_request(self._req(ctx2, p2.clone()))
        msa2 = ctx2.mamba_slot_allocator
        self._mamba_allocate_and_register(ctx2, self._block_ids(ctx2, 0, 3)[:2])
        req2b = self._req(ctx2, p2.clone(), request_id=2)
        req2b._mamba_num_matched_blocks = 2
        ctx2.add_request(req2b)
        count2 = msa2._intermediate_counts_cpu[1].item()
        if count2 > 0:
            offsets = msa2._intermediate_offsets_cpu[1, :count2].tolist()
            for o in offsets:
                assert o > 0 and o % 128 == 0
        assert msa2._eos_cache_block_id_cpu[1].item() < 0

        # block-aligned prompts set EOS cache block ID
        ctx3 = self._mctx(block_size_tokens=bs)
        p3 = self._prompt(bs * 3)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        req3 = self._req(ctx3, p3.clone(), request_id=2)
        req3._mamba_num_matched_blocks = 0
        ctx3.add_request(req3)
        # Deferred Mamba ops execute during transfer.
        ctx3.initialize_attention_state()
        ctx3.transfer_bookkeeping_to_gpu()
        assert ctx3.mamba_slot_allocator._eos_cache_block_id_cpu[1].item() >= 0

        # intermediate output buffers are pre-allocated
        ctx4 = self._mctx()
        msa4 = ctx4.mamba_slot_allocator
        assert msa4.intermediate_ssm_out.shape[0] == ctx4.num_mamba_layers
        assert msa4.intermediate_conv_out.shape[0] == ctx4.num_mamba_layers
        assert msa4.intermediate_ssm_out.shape[1] == msa4.max_intermediate_count

        # store_from_live copies all layers
        ctx5 = self._mctx()
        msa5 = ctx5.mamba_slot_allocator
        p5 = self._prompt(ctx5.block_size_tokens * 2)
        ctx5.add_request(self._req(ctx5, p5.clone()))
        bid5 = ctx5.request_to_kv_block_ids[0][0].item()
        slot5 = msa5.allocate_slots_batch([bid5])[0]
        mamba_idx = ctx5.mamba_metadata.request_to_mamba_state_idx[0].item()
        for layer in range(ctx5.num_mamba_layers):
            ctx5.mamba_conv_states[layer, mamba_idx] = layer + 1.0
            ctx5.mamba_ssm_states[layer, mamba_idx] = layer + 100.0
        msa5.store_from_live_batch([slot5], [0])
        for layer in range(ctx5.num_mamba_layers):
            assert torch.allclose(
                ctx5.mamba_slot_allocator.conv_states[layer, slot5],
                torch.full_like(ctx5.mamba_slot_allocator.conv_states[layer, slot5], layer + 1.0),
            )

    @pytest.mark.parametrize(
        "max_requests,max_tokens,limiting_budget",
        [
            pytest.param(4, 5 * 256 + 3, "tokens", id="token-limited"),
            pytest.param(2, 5 * 256 + 1, "requests", id="request-limited"),
        ],
    )
    @pytest.mark.internal
    def test_intermediate_extraction_stresses_scratch_sizing(
        self, max_requests, max_tokens, limiting_budget
    ):
        """Packed prefills exercise both scratch-buffer sizing limits."""
        import math

        bs = 256
        ctx = self._mctx(
            block_size_tokens=bs,
            max_tokens=max_tokens,
            max_requests=max_requests,
            max_sequence_length=4096,
            rounder=1,
        )
        token_budget = max(0, math.ceil(max_tokens / bs) - 1)
        request_budget = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * max_requests
        expected_budget = min(token_budget, request_budget)
        if limiting_budget == "tokens":
            assert expected_budget == token_budget < request_budget
        else:
            assert expected_budget == request_budget < token_budget

        assert ctx.max_mamba_intermediate_states_per_step == expected_budget
        assert ctx.mamba_slot_allocator.max_intermediate_count == expected_budget
        assert ctx.mamba_metadata.max_intermediate_count == expected_budget
        assert ctx.mamba_slot_allocator.intermediate_ssm_out.shape[1] == expected_budget

        self._saturate_mamba_scratch(ctx, expected_budget, request_id_base=10_000)
        assert ctx.mamba_metadata.intermediate_count == expected_budget

    @pytest.mark.internal
    def test_intermediate_offsets_use_configured_mamba_chunk_size(self):
        # Regression guard for the fix that reads mamba_chunk_size from the model
        # config instead of hardcoding 128 in compute_and_store_offsets.
        #
        # With a 64-token mamba chunk and 64-token blocks, a 65-token prompt's
        # block boundary at token 64 is a valid mamba-chunk multiple
        # (64 % 64 == 0), so its state must be extracted and cached. The old
        # hardcoded filter (64 % 128 != 0) would have wrongly skipped it, caching
        # nothing and leaving no resume point for a later turn.
        bs = 64
        ctx = self._mctx(
            mamba_config=self._mamba_config(mamba_chunk_size=64),
            block_size_tokens=bs,
            max_sequence_length=512,
        )
        assert ctx.mamba_chunk_size == 64
        msa = ctx.mamba_slot_allocator

        # Fresh 65-token prompt: crosses exactly the block boundary at token 64.
        ctx.add_request(self._req(ctx, self._prompt(bs + 1)))

        count = msa._intermediate_counts_cpu[0].item()
        # The boundary at token 64 was recorded -- would be 0 under the old
        # hardcoded-128 filter, since 64 % 128 != 0.
        assert count == 1
        offsets = msa._intermediate_offsets_cpu[0, :count].tolist()
        assert offsets == [bs]


class TestMixedCachedAndFreshPrefill(PrefixCachingTestBase):

    def _setup_mixed_batch(self, model_type):
        """Set up mixed batch: req0 (decode), reqs 1-4 (mixed cached/fresh prefill).

        Uses 2-block + tail prompts so cached requests skip 2 full blocks and
        prefill only the tail, avoiding the single-token-chunk clamp while still
        producing distinct query lengths for cached vs fresh requests.
        """
        if model_type == "gpt":
            ctx = self._ctx(block_size_tokens=32)
        else:
            ctx = self._ctx(
                mamba_config=self._mamba_config(),
                prefix_caching_mamba_gb=0.01,
                block_size_tokens=256,
                max_sequence_length=4096,
            )
        bs = ctx.block_size_tokens
        tail = 5
        prompt_len = bs * 2 + tail

        prompt0 = self._prompt(prompt_len)
        req0 = self._req(ctx, prompt0.clone())
        ctx.add_request(req0)

        vocab_size = prompt_len + 50
        block_hash = req0.precomputed_block_hashes[0]

        if model_type == "hybrid":
            block_ids_0 = self._block_ids(ctx, 0, 2)
            for bid in block_ids_0:
                bh = ctx.kv_block_allocator.block_hashes[bid].item()
                ctx.mamba_slot_allocator.register_block_hashes_batch([bid], [bh])

        ctx.request_kv_length_offsets[0] += prompt_len
        ctx.request_query_lengths[0] = 1
        ctx.request_last_kv_block_offset[0] = 0
        ctx.num_prefill_requests = 0
        ctx.active_token_count = 1
        ctx.token_to_input_ids[0] = 42
        ctx.token_to_pos_ids[0] = prompt_len
        ctx.token_to_request_idx[0] = 0

        req1 = self._req(ctx, prompt0.clone(), request_id=2)
        req2 = self._req(ctx, self._prompt(prompt_len, offset=50), request_id=3)
        req3 = self._req(ctx, prompt0.clone(), request_id=4)
        req4 = self._req(ctx, self._prompt(prompt_len, offset=40), request_id=5)

        if model_type == "hybrid":
            req1._mamba_num_matched_blocks = 2
            req2._mamba_num_matched_blocks = 0
            req3._mamba_num_matched_blocks = 2
            req4._mamba_num_matched_blocks = 0

        for r in [req1, req2, req3, req4]:
            ctx.add_request(r)

        return ctx, bs, tail, prompt_len, vocab_size, block_hash

    @pytest.mark.parametrize("model_type", ["gpt", "hybrid"])
    @pytest.mark.internal
    def test_mixed_batch(self, model_type):
        ctx, bs, tail, prompt_len, vocab_size, block_hash = self._setup_mixed_batch(model_type)

        # Cached requests (req1/req3) skip 2 full blocks → query_length == tail.
        # Fresh requests (req2/req4) have no match → query_length == prompt_len.
        cached_ql = tail
        fresh_ql = prompt_len

        # query lengths: decode=1, cached=tail, fresh=prompt_len
        assert ctx.request_query_lengths[0].item() == 1
        assert ctx.request_query_lengths[1].item() == cached_ql
        assert ctx.request_query_lengths[2].item() == fresh_ql
        assert ctx.request_query_lengths[3].item() == cached_ql
        assert ctx.request_query_lengths[4].item() == fresh_ql
        assert ctx.active_token_count == 1 + 2 * cached_ql + 2 * fresh_ql

        # last_token_logits
        ctx.initialize_attention_state()
        ctx.transfer_bookkeeping_to_gpu()
        logits = torch.randn(
            1, ctx.padded_active_token_count, vocab_size, device=torch.cuda.current_device()
        )
        result = ctx.last_token_logits(logits)
        assert result.shape == (5, vocab_size)

        # calculate_log_probs
        new_tokens = torch.randint(0, vocab_size, (5,), device=torch.cuda.current_device())
        log_probs_list, _ = ctx.calculate_log_probs(logits, new_tokens)
        assert len(log_probs_list) == 5
        assert len(log_probs_list[0]) == 1
        assert len(log_probs_list[1]) == cached_ql
        assert len(log_probs_list[2]) == fresh_ql
        assert len(log_probs_list[3]) == cached_ql
        assert len(log_probs_list[4]) == fresh_ql


class TestMambaSlotAllocator(PrefixCachingTestBase):

    def _mctx(self, **kwargs):
        defaults = dict(
            mamba_config=self._mamba_config(),
            prefix_caching_mamba_gb=0.01,
            block_size_tokens=256,
            max_sequence_length=4096,
        )
        defaults.update(kwargs)
        return self._ctx(**defaults)

    @pytest.mark.internal
    def test_allocate_slots_batch(self, monkeypatch):
        ctx = self._mctx()
        bs = ctx.block_size_tokens
        msa = ctx.mamba_slot_allocator

        # Basic batch: allocate 3 new slots, verify unique slots and mappings
        prompt = self._prompt(bs * 4)
        ctx.add_request(self._req(ctx, prompt.clone()))
        bids = self._block_ids(ctx, 0, 3)
        initial_free = msa.free_count
        slots = msa.allocate_slots_batch(bids)
        assert len(slots) == 3
        assert len(set(slots)) == 3  # all unique
        assert msa.free_count == initial_free - 3
        for bid, slot in zip(bids, slots):
            assert msa.block_to_slot[bid].item() == slot
            assert msa.slot_to_block[slot].item() == bid

        # Existing slots: same block_ids return same slots without consuming pool
        free_before = msa.free_count
        slots2 = msa.allocate_slots_batch(bids)
        assert slots2 == slots
        assert msa.free_count == free_before

        # Dedup: same block_id twice, only one free slot consumed
        ctx2 = self._mctx()
        ctx2.add_request(self._req(ctx2, self._prompt(bs * 2)))
        bid_new = self._block_ids(ctx2, 0, 1)[0]
        msa2 = ctx2.mamba_slot_allocator
        free_before2 = msa2.free_count
        dup_slots = msa2.allocate_slots_batch([bid_new, bid_new])
        assert dup_slots[0] == dup_slots[1]
        assert msa2.free_count == free_before2 - 1

        # Mixed: pre-allocated + new in one call
        ctx3 = self._mctx()
        ctx3.add_request(self._req(ctx3, self._prompt(bs * 3)))
        bids3 = self._block_ids(ctx3, 0, 3)
        msa3 = ctx3.mamba_slot_allocator
        pre_slot = msa3.allocate_slots_batch([bids3[0]])[0]
        free_before3 = msa3.free_count
        mixed_slots = msa3.allocate_slots_batch(bids3)
        assert mixed_slots[0] == pre_slot
        assert msa3.free_count == free_before3 - 2  # only 2 new

        # Eviction: exhaust free pool, verify eviction fires and returns valid slots.
        # Budget must cover the CUDA-graph extraction scratch (2 * max_requests
        # slots) plus the durable cache; a budget too small to fit the scratch now
        # raises (see test_mamba_cache_budget_too_small_raises).
        ctx4 = self._mctx(prefix_caching_mamba_gb=0.01)
        msa4 = ctx4.mamba_slot_allocator
        total_slots = msa4.max_slots
        ctx4.add_request(self._req(ctx4, self._prompt(bs * 4)))
        bids4 = self._block_ids(ctx4, 0, 4)
        # Allocate all available slots by filling the free pool
        fill_bids = bids4[: min(total_slots, 4)]
        fill_slots = msa4.allocate_slots_batch(fill_bids)
        assert len(fill_slots) == len(fill_bids)
        # If we can exhaust the pool, test eviction
        if total_slots <= 4:
            assert msa4.free_count == 0
            # Set ref counts to 0 so blocks are evictable
            for bid in fill_bids:
                ctx4.kv_block_allocator.block_ref_counts[bid] = 0
            # Invalidate old slots, then reallocate to test eviction path
            for bid in fill_bids:
                msa4.invalidate_block(bid)
            evict_slots = msa4.allocate_slots_batch(fill_bids)
            assert len(evict_slots) == len(fill_bids)

        # All-pinned pressure is transactional. Repeated optional commits skip
        # their new snapshot without losing the already exhausted free stack;
        # once one durable slot becomes evictable, the next commit succeeds.
        ctx5 = self._mctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU)
        ctx5.add_request(self._req(ctx5, self._prompt(bs * 3)))
        msa5 = ctx5.mamba_slot_allocator
        bids5 = self._block_ids(ctx5, 0, 3)
        msa5.free_slots[:2] = torch.tensor([0, 1], dtype=torch.int32)
        msa5.free_count = 2
        first_slots = msa5.allocate_slots_batch(bids5[:2])
        assert msa5.get_allocatable_slot_count() == 0
        mappings_before = msa5.block_to_slot.clone()

        with pytest.raises(RuntimeError, match="No evictable Mamba"):
            msa5.allocate_slots_batch([bids5[2]])
        assert msa5.free_count == 0
        assert torch.equal(msa5.block_to_slot, mappings_before)

        clear_count = 0

        def clear_intermediate_state():
            nonlocal clear_count
            clear_count += 1

        monkeypatch.setattr(msa5, "_clear_intermediate_state", clear_intermediate_state)
        block_hash = ctx5.kv_block_allocator.block_hashes[bids5[2]].item()
        monkeypatch.setattr(
            msa5, "_collect_commit_data", lambda: ([], [], [bids5[2]], [0], [block_hash])
        )
        for _ in range(3):
            msa5.commit_intermediate_states()
            assert msa5.free_count == 0
            assert torch.equal(msa5.block_to_slot, mappings_before)
        assert clear_count == 3
        assert msa5.commit_count == 0

        ctx5.kv_block_allocator.block_ref_counts[bids5[0]] = 0
        msa5.commit_intermediate_states()
        assert msa5.block_to_slot[bids5[2]].item() == first_slots[0]
        assert msa5.block_to_slot[bids5[0]].item() == -1
        assert msa5.eviction_count == 1
        assert msa5.commit_count == 1

        # A mixed lookup protects requested existing state even when that state
        # is ref-zero and older than every other eviction candidate.
        ctx6 = self._mctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU)
        ctx6.add_request(self._req(ctx6, self._prompt(bs * 3)))
        msa6 = ctx6.mamba_slot_allocator
        bids6 = self._block_ids(ctx6, 0, 3)
        msa6.free_slots[:2] = torch.tensor([0, 1], dtype=torch.int32)
        msa6.free_count = 2
        slot_a, slot_c = msa6.allocate_slots_batch([bids6[0], bids6[2]])
        ctx6.kv_block_allocator.block_ref_counts[bids6[0]] = 0
        ctx6.kv_block_allocator.block_ref_counts[bids6[2]] = 0
        ctx6.kv_block_allocator.block_timestamps[bids6[0]] = 0
        ctx6.kv_block_allocator.block_timestamps[bids6[2]] = 1

        existing_slot, new_slot = msa6.allocate_slots_batch([bids6[0], bids6[1]])
        assert existing_slot == slot_a
        assert new_slot == slot_c
        assert msa6.block_to_slot[bids6[0]].item() == slot_a
        assert msa6.block_to_slot[bids6[1]].item() == slot_c
        assert msa6.block_to_slot[bids6[2]].item() == -1

    @pytest.mark.internal
    def test_commit_intermediate_states_batched(self):
        ctx = self._mctx(block_size_tokens=256)
        bs = ctx.block_size_tokens
        msa = ctx.mamba_slot_allocator
        alloc = ctx.kv_block_allocator
        metadata = ctx.mamba_metadata

        # Set up context with a prefill request that has block-aligned prompt
        prompt = self._prompt(bs * 3)
        req = self._req(ctx, prompt.clone())
        req._mamba_num_matched_blocks = 0
        ctx.add_request(req)

        # initialize_attention_state sets batch_dimensions and mamba metadata
        ctx.initialize_attention_state()

        # Determine prefill_start for this batch
        prefill_start = ctx.paused_request_count + ctx.batch_dimensions.decode_req_count
        ctx_idx = prefill_start  # first prefill request

        # Write known patterns to intermediate output buffers
        for layer in range(ctx.num_mamba_layers):
            msa.intermediate_ssm_out[layer, 0] = layer + 1.0
            msa.intermediate_conv_out[layer, 0] = layer + 100.0

        # Set up intermediate offsets: 1 intermediate at src_offset=0
        bid0 = ctx.request_to_kv_block_ids[ctx_idx][0].item()
        msa._intermediate_block_ids_cpu[ctx_idx, 0] = bid0
        msa._intermediate_offsets_cpu[ctx_idx, 0] = 128
        msa._intermediate_counts_cpu[ctx_idx] = 1
        msa._has_intermediates = True

        # Set metadata fields that would normally be set by _update_intermediate_offsets
        metadata.intermediate_count = 1
        metadata.per_request_intermediate_counts = [1]

        # Set up EOS block (block-aligned prompt)
        eos_bid = ctx.request_to_kv_block_ids[ctx_idx][2].item()
        msa._eos_cache_block_id_cpu[ctx_idx] = eos_bid

        # Write known patterns to live mamba state for EOS copy
        mamba_idx = metadata.request_to_mamba_state_idx[ctx_idx].item()
        for layer in range(ctx.num_mamba_layers):
            ctx.mamba_conv_states[layer, mamba_idx] = layer + 200.0
            ctx.mamba_ssm_states[layer, mamba_idx] = layer + 300.0

        # Call the batched commit
        msa.commit_intermediate_states()

        # Verify intermediate state was copied to correct slot
        slot0 = msa.block_to_slot[bid0].item()
        assert slot0 >= 0
        for layer in range(ctx.num_mamba_layers):
            assert torch.allclose(
                msa.ssm_states[layer, slot0],
                torch.full_like(msa.ssm_states[layer, slot0], layer + 1.0),
            )
            assert torch.allclose(
                msa.conv_states[layer, slot0],
                torch.full_like(msa.conv_states[layer, slot0], layer + 100.0),
            )

        # Verify EOS state was copied from live buffer
        eos_slot = msa.block_to_slot[eos_bid].item()
        assert eos_slot >= 0
        for layer in range(ctx.num_mamba_layers):
            assert torch.allclose(
                msa.conv_states[layer, eos_slot],
                torch.full_like(msa.conv_states[layer, eos_slot], layer + 200.0),
            )
            assert torch.allclose(
                msa.ssm_states[layer, eos_slot],
                torch.full_like(msa.ssm_states[layer, eos_slot], layer + 300.0),
            )

        # Verify hash_to_block_id updated for valid hashes
        bid0_hash = alloc.block_hashes[bid0].item()
        eos_hash = alloc.block_hashes[eos_bid].item()
        if bid0_hash > 0:
            assert msa.hash_to_block_id.get(bid0_hash) == bid0
        if eos_hash > 0:
            assert msa.hash_to_block_id.get(eos_hash) == eos_bid

        # Verify _has_intermediates cleared
        assert not msa._has_intermediates

    @pytest.mark.parametrize("follower_count", [1, 4], ids=["serial", "concurrent"])
    @pytest.mark.internal
    def test_hybrid_followers_preserve_published_kv_and_mamba_state(
        self, follower_count, monkeypatch
    ):
        """Recomputed followers may read a published prefix but never write it."""
        ctx = self._mctx(
            block_size_tokens=256,
            max_requests=8,
            max_tokens=4096,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
        )
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 3)

        ctx.add_request(self._req(ctx, prompt.clone(), request_id=0))
        canonical_bids = self._block_ids(ctx, 0, 3)
        msa = ctx.mamba_slot_allocator
        slots = self._mamba_allocate_and_register(ctx, canonical_bids)

        # Give both durable caches recognizable bytes before any follower runs.
        ctx.memory_buffer[:, :, canonical_bids] = 7.0
        slot_tensor = torch.tensor(slots, dtype=torch.int64, device=msa.conv_states.device)
        msa.conv_states[:, slot_tensor] = 11.0
        msa.ssm_states[:, slot_tensor] = 13.0
        kv_before = ctx.memory_buffer[:, :, canonical_bids].clone()
        conv_before = msa.conv_states[:, slot_tensor].clone()
        ssm_before = msa.ssm_states[:, slot_tensor].clone()
        dummy = ctx.kv_block_allocator.dummy_block_idx
        ctx.memory_buffer[:, :, dummy] = 0.0

        follower_token_ranges = []
        for follower_idx in range(follower_count):
            token_start = ctx.active_token_count
            request_idx = ctx.total_request_count
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=request_idx))
            query_length = ctx.request_query_lengths[request_idx].item()
            follower_token_ranges.append(slice(token_start, token_start + query_length))

            # Attention still reads the canonical blocks through the request table.
            assert self._block_ids(ctx, request_idx, 3) == canonical_bids
            # The final matched block is recomputed for live Mamba state. Its KV
            # appends must be isolated in the dummy block.
            assert query_length == bs
            assert torch.all(
                ctx.token_to_block_idx[follower_token_ranges[-1]]
                == ctx.kv_block_allocator.dummy_block_idx
            )

        ctx.initialize_attention_state()

        # Exercise the real KV append path. The anchor rewrites the same bytes;
        # followers write distinct values, which would corrupt the final
        # canonical block if their append metadata still pointed at it.
        key = torch.full(
            (
                ctx.padded_active_token_count,
                1,
                ctx.num_attention_heads_per_partition,
                ctx.hidden_size_per_attention_head,
            ),
            7.0,
            dtype=ctx.params_dtype,
            device=torch.cuda.current_device(),
        )
        value = key.clone()
        for follower_idx, token_range in enumerate(follower_token_ranges):
            key[token_range] = 31.0 + follower_idx
            value[token_range] = 61.0 + follower_idx
        ctx.append_key_value_cache(layer_number=1, key=key, value=value)

        assert torch.equal(ctx.memory_buffer[:, :, canonical_bids], kv_before)
        assert torch.count_nonzero(ctx.memory_buffer[:, :, dummy]) > 0

        # Repeatedly present serial/concurrent followers as producers for the
        # already-published EOS state. Every attempt must leave the durable
        # Mamba bytes unchanged.
        eos_bid = canonical_bids[-1]
        eos_hash = ctx.kv_block_allocator.block_hashes[eos_bid].item()
        producer_indices = list(range(ctx.total_request_count))
        monkeypatch.setattr(
            msa,
            "_collect_commit_data",
            lambda: (
                [],
                [],
                [eos_bid] * len(producer_indices),
                producer_indices,
                [eos_hash] * len(producer_indices),
            ),
        )
        for cycle in range(3):
            for request_idx in producer_indices:
                mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[request_idx].item()
                ctx.mamba_conv_states[:, mamba_idx] = 100.0 * cycle + request_idx
                ctx.mamba_ssm_states[:, mamba_idx] = 200.0 * cycle + request_idx
            msa.commit_intermediate_states()
            assert torch.equal(msa.conv_states[:, slot_tensor], conv_before)
            assert torch.equal(msa.ssm_states[:, slot_tensor], ssm_before)

    @pytest.mark.internal
    def test_concurrent_mamba_producers_stably_deduplicate_across_slot_reuse(self, monkeypatch):
        """One stable producer materializes each destination once per slot lifetime."""
        ctx = self._mctx(
            block_size_tokens=256,
            max_requests=8,
            max_tokens=4096,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
        )
        bs = ctx.block_size_tokens
        prompt_a = self._prompt(bs * 2)
        prompt_b = self._prompt(bs * 2, offset=10_000)

        for request_id, prompt in enumerate((prompt_a, prompt_a, prompt_b, prompt_b)):
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=request_id))
        ctx.initialize_attention_state()

        msa = ctx.mamba_slot_allocator
        destinations = [
            ctx.request_to_kv_block_ids[0, 1].item(),
            ctx.request_to_kv_block_ids[0, 1].item(),
            ctx.request_to_kv_block_ids[2, 1].item(),
            ctx.request_to_kv_block_ids[2, 1].item(),
        ]
        assert destinations[0] != destinations[2]
        hashes = [ctx.kv_block_allocator.block_hashes[bid].item() for bid in destinations]
        producer_indices = [0, 1, 2, 3]
        monkeypatch.setattr(
            msa, "_collect_commit_data", lambda: ([], [], destinations, producer_indices, hashes)
        )

        store_calls = []
        original_store = msa.store_from_live_batch

        def tracked_store(slots, request_indices):
            store_calls.append((list(slots), list(request_indices)))
            original_store(slots, request_indices)

        monkeypatch.setattr(msa, "store_from_live_batch", tracked_store)

        reused_slot_set = None
        for cycle in range(3):
            for request_idx in producer_indices:
                mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[request_idx].item()
                ctx.mamba_conv_states[:, mamba_idx] = 100.0 * cycle + request_idx
                ctx.mamba_ssm_states[:, mamba_idx] = 200.0 * cycle + request_idx

            calls_before = len(store_calls)
            msa.commit_intermediate_states()
            assert len(store_calls) == calls_before + 1
            # The first producer for each new destination wins, and both copies
            # remain in one vectorized batch.
            assert store_calls[-1][1] == [0, 2]

            unique_bids = [destinations[0], destinations[2]]
            slots = [msa.get_slot(bid) for bid in unique_bids]
            if reused_slot_set is not None:
                assert set(slots) == reused_slot_set
            reused_slot_set = set(slots)

            for bid, producer_idx in zip(unique_bids, [0, 2]):
                slot = msa.get_slot(bid)
                assert torch.all(msa.conv_states[:, slot] == 100.0 * cycle + producer_idx)
                assert torch.all(msa.ssm_states[:, slot] == 200.0 * cycle + producer_idx)

            conv_before = msa.conv_states[:, slots].clone()
            ssm_before = msa.ssm_states[:, slots].clone()
            calls_before = len(store_calls)
            for request_idx in producer_indices:
                mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[request_idx].item()
                ctx.mamba_conv_states[:, mamba_idx] = -1000.0 - request_idx
                ctx.mamba_ssm_states[:, mamba_idx] = -2000.0 - request_idx
            msa.commit_intermediate_states()
            assert len(store_calls) == calls_before
            assert torch.equal(msa.conv_states[:, slots], conv_before)
            assert torch.equal(msa.ssm_states[:, slots], ssm_before)

            for bid in unique_bids:
                msa.invalidate_block(bid)


class TestPerBlockRouting(PrefixCachingTestBase):
    """Tests for per-block routing storage and reconstruction."""

    @pytest.mark.internal
    def test_routing_cleared_across_reuse_and_reset_cycles(self):
        """Reused blocks never inherit routing from an earlier lifetime."""
        ctx = self._ctx(enable_prefix_caching=False)
        alloc = ctx.kv_block_allocator

        current_ids = alloc.allocate_memory_blocks(1)
        bid = current_ids[0].item()
        positions = np.array([0])
        for cycle in range(3):
            routing = np.full((1, 4, 2), cycle + 1, dtype=np.int16)
            alloc.store_block_routing(bid, positions, routing)
            assert np.array_equal(alloc.get_block_routing(bid)[positions], routing)

            alloc.release_memory_blocks(current_ids)
            assert np.array_equal(alloc.get_block_routing(bid)[positions], routing)

            current_ids = alloc.allocate_memory_blocks(1)
            assert current_ids.tolist() == [bid]
            assert alloc.get_block_routing(bid) is None

        alloc.store_block_routing(bid, positions, np.ones((1, 4, 2), dtype=np.int16))
        alloc.reset()
        assert alloc.get_block_routing(bid) is None
        assert len(alloc.block_routing) == 0

    @pytest.mark.internal
    def test_routing_persists_through_deregister(self):
        """Routing data persists through block deregister (needed for reconstruction)."""
        ctx = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens

        # Add a request so blocks get allocated and registered
        prompt = self._prompt(bs * 2)
        req = self._req(ctx, prompt)
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)

        # Store routing for both blocks.
        routing_by_block = {
            bid: np.full((bs, 4, 2), fill_value=index + 1, dtype=np.int16)
            for index, bid in enumerate((b0, b1))
        }
        for bid, routing in routing_by_block.items():
            alloc.store_block_routing(bid, np.arange(bs), routing)

        # Release blocks (REF_ZERO deregisters immediately)
        blocks = ctx.request_to_kv_block_ids[0]
        valid_blocks = blocks[blocks >= 0]
        alloc.release_memory_blocks(valid_blocks)

        # Deregistration removes hash ownership but preserves enough routing state
        # to reconstruct the entire prefix.
        reconstructed = alloc.reconstruct_routing_from_blocks([b0, b1], 2 * bs)
        assert reconstructed is not None
        assert np.array_equal(
            reconstructed, np.concatenate([routing_by_block[b0], routing_by_block[b1]])
        )

    @pytest.mark.internal
    def test_reconstruct_routing_from_blocks(self):
        """Test reconstruction of routing indices from per-block storage."""
        ctx = self._ctx()
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens
        num_layers, topk = 4, 2

        # Allocate 3 blocks
        block_ids = alloc.allocate_memory_blocks(3)
        bids = block_ids.tolist()

        # Store routing for all positions in first two blocks (full)
        for bid in bids[:2]:
            alloc.store_block_routing(
                bid,
                np.arange(bs),
                np.arange(bs * num_layers * topk, dtype=np.int16).reshape(bs, num_layers, topk)
                + bid,
            )

        # Store routing for partial last block (e.g., 5 tokens)
        partial = 5
        expected_partial = (
            np.arange(partial * num_layers * topk, dtype=np.int16).reshape(
                partial, num_layers, topk
            )
            + bids[2]
        )
        alloc.store_block_routing(bids[2], np.arange(partial), expected_partial)
        stored_partial = alloc.get_block_routing(bids[2])
        assert isinstance(stored_partial, np.ndarray)
        assert stored_partial.shape == (bs, num_layers, topk)
        assert np.allclose(stored_partial[:partial], expected_partial)
        assert (stored_partial[partial:] == 0).all()

        # total_routing_tokens = 2 full blocks + 5 partial = 2*bs + 5
        total_routing_tokens = 2 * bs + partial

        result = alloc.reconstruct_routing_from_blocks(bids, total_routing_tokens)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (total_routing_tokens, num_layers, topk)

        # Verify content: first block
        expected_b0 = (
            np.arange(bs * num_layers * topk, dtype=np.int16).reshape(bs, num_layers, topk)
            + bids[0]
        )
        assert np.allclose(result[:bs], expected_b0)

        # Verify content: partial last block
        assert np.allclose(result[2 * bs :], expected_partial)

    @pytest.mark.internal
    def test_reconstruct_returns_none_for_missing_block(self):
        """Reconstruction returns None if a block has no routing data."""
        ctx = self._ctx()
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens

        block_ids = alloc.allocate_memory_blocks(2)
        bids = block_ids.tolist()

        # Only store routing for the first block
        alloc.store_block_routing(
            bids[0], np.arange(bs), np.random.randint(-100, 100, size=(bs, 4, 2), dtype=np.int16)
        )

        result = alloc.reconstruct_routing_from_blocks(bids, 2 * bs)
        assert result is None

    @pytest.mark.internal
    def test_routing_survives_prefix_match_lru(self):
        """In LRU mode, matched blocks' routing persists for the new request."""
        ctx = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU)
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens

        # First request: 2 full blocks
        prompt = self._prompt(bs * 2)
        req1 = self._req(ctx, prompt, request_id=1)
        ctx.add_request(req1)
        b0, b1 = self._block_ids(ctx, 0, 2)

        # Store routing for both blocks
        routing_b0 = np.random.randint(-100, 100, size=(bs, 4, 2), dtype=np.int16)
        routing_b1 = np.random.randint(-100, 100, size=(bs, 4, 2), dtype=np.int16)
        alloc.store_block_routing(b0, np.arange(bs), routing_b0)
        alloc.store_block_routing(b1, np.arange(bs), routing_b1)

        # Release first request's blocks (LRU: blocks stay cached)
        blocks = ctx.request_to_kv_block_ids[0]
        valid_blocks = blocks[blocks >= 0]
        active_mask = torch.zeros(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        ctx.update_requests(active_mask, new_tokens)

        # Second request with same prefix should match
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        ctx.add_request(req2)

        # The matched blocks should still have routing data
        assert alloc.get_block_routing(b0) is not None
        assert np.allclose(alloc.get_block_routing(b0), routing_b0)
        assert alloc.get_block_routing(b1) is not None
        assert np.allclose(alloc.get_block_routing(b1), routing_b1)


class TestPrefixCacheReuse(PrefixCachingTestBase):
    """Cross-request prefix reuse on hybrid (Mamba) models:

    - reset(preserve_prefix_cache=True) keeps the cache; a plain reset() clears it.
    - Per-context prefill token accounting (computed vs skipped).
    - Mamba state is extracted for the last complete block of a multi-chunk prompt.
    """

    @pytest.mark.internal
    def test_reset_preserves_prefix_cache_when_requested(self):
        # Repeated preserving resets must keep serving real hits. Repeated
        # clearing resets must force the whole prompt through prefill again.
        ctx = self._ctx(enable_prefix_caching=True)
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 2 + 5)
        ctx.add_request(self._req(ctx, prompt.clone()))
        cached = dict(ctx.kv_block_allocator.kv_hash_to_block_id)
        assert len(cached) == 2

        for request_id in range(2, 5):
            ctx.reset(preserve_prefix_cache=True)
            follower = self._req(ctx, prompt.clone(), request_id=request_id)
            ctx.add_request(follower)
            assert ctx.kv_block_allocator.kv_hash_to_block_id == cached
            assert follower.num_cached_tokens == 2 * bs
            assert ctx.request_query_lengths[0].item() == 5

        for request_id in range(5, 8):
            ctx.reset()
            assert not ctx.kv_block_allocator.kv_hash_to_block_id
            recomputed = self._req(ctx, prompt.clone(), request_id=request_id)
            ctx.add_request(recomputed)
            assert recomputed.num_cached_tokens == 0
            assert ctx.request_query_lengths[0].item() == len(prompt)
            assert len(ctx.kv_block_allocator.kv_hash_to_block_id) == 2

    @pytest.mark.internal
    def test_prefill_computed_and_skipped_counters(self):
        # A second request that shares a cached prefix should skip that prefix's
        # prefill; the per-context counters must reflect computed vs skipped tokens.
        ctx = self._ctx(enable_prefix_caching=True)
        bs = ctx.block_size_tokens

        ctx.add_request(self._req(ctx, self._prompt(bs * 4), request_id=1))
        assert ctx.prefix_cache_prefill_skipped_tokens == 0
        assert ctx.prefix_cache_prefill_computed_tokens == bs * 4

        # request 2 shares the first 4 blocks, adds 2 new blocks
        req2 = self._req(ctx, self._prompt(bs * 6), request_id=2)
        (matched, _, _, _, prefix_skip, _) = ctx._compute_prefix_match(req2, bs * 6)
        assert len(matched) == 4 and prefix_skip == bs * 4
        ctx.add_request(req2)

        assert ctx.prefix_cache_prefill_skipped_tokens == bs * 4
        assert ctx.prefix_cache_prefill_computed_tokens == bs * 6  # 4bs + 2bs

    @pytest.mark.internal
    def test_mamba_extraction_covers_last_block_of_continuation_chunk(self):
        # For a non-block-aligned, multi-chunk prompt, the last complete block lies
        # in a continuation chunk. Extraction offsets are chunk-relative, so that
        # boundary's Mamba state is recorded when its chunk is scheduled.
        ctx = self._ctx(
            mamba_config=self._mamba_config(),
            prefix_caching_mamba_gb=0.01,
            block_size_tokens=256,
            max_sequence_length=4096,
        )  # mamba prefix caching enabled
        bs = ctx.block_size_tokens
        assert bs == 256
        msa = ctx.mamba_slot_allocator

        prompt_len = bs * 3 + 64  # 3 complete blocks + a 64-token remainder
        req = self._req(ctx, self._prompt(prompt_len))
        ctx.add_request(req)  # populates request_to_kv_block_ids[0]
        overall_blocks = ctx.request_kv_block_counts[0].item()
        assert overall_blocks == 4  # ceil(832 / 256)

        # Simulate the continuation chunk that covers tokens [2*bs, prompt_len):
        # finished=2*bs, no prefix skip, the rest of the prompt as the chunk.
        req.finished_chunk_token_count = 2 * bs
        cont_chunk = prompt_len - 2 * bs
        msa.compute_and_store_offsets(
            req,
            current_id=0,
            skip_tokens=0,
            prefill_chunk_length=cont_chunk,
            num_matched_blocks=0,
            matched_block_ids=[],
            overall_required_blocks=overall_blocks,
        )

        # last complete block boundary = 3*bs (768); chunk-relative offset = 768-512=256.
        last_aligned_abs = (prompt_len // bs) * bs
        expected_offset = last_aligned_abs - 2 * bs
        count = msa._intermediate_counts_cpu[0].item()
        assert count >= 1
        recorded = msa._intermediate_offsets_cpu[0, :count].tolist()
        assert expected_offset in recorded
        # the recorded boundary maps to the last complete block (index 2)
        idx = recorded.index(expected_offset)
        assert (
            msa._intermediate_block_ids_cpu[0, idx].item()
            == ctx.request_to_kv_block_ids[0][last_aligned_abs // bs - 1].item()
        )
