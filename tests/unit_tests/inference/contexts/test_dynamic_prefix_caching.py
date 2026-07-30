# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import gc
import json
import socket
import time
import urllib.request
from collections import deque

import numpy as np
import pytest
import torch

from megatron.core.inference.config import (
    InferenceConfig,
    KVCacheManagementMode,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.async_zmq_communicator import AsyncZMQCommunicator
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
    compute_block_hashes_batched,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (
    HAS_BACKEND,
    start_text_gen_server,
    stop_text_gen_server,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicInferenceEngineTestBase,
)
from tests.unit_tests.test_utilities import Utils

try:
    import zmq

    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False


class _NumericTokenizer:
    """Picklable tokenizer used by the real coordinator and HTTP subprocesses."""

    vocab_size = 100
    bos = None
    eod = 0
    pad = 0

    def tokenize(self, prompt):
        return [int(token) % self.vocab_size for token in prompt.split()]

    def detokenize(self, tokens, skip_special_tokens=False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if skip_special_tokens:
            tokens = [token for token in tokens if token != self.eod]
        return "".join(f"{token} " for token in tokens)


def _http_json(url, payload=None):
    """Issue one real HTTP request from a worker thread."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


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
            unified_memory_level=0,
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


class _StubEngine(DynamicInferenceEngine):

    def __init__(self, context: DynamicInferenceContext, *, enable_chunked_prefill=False):
        self.context = context
        self.enable_chunked_prefill = enable_chunked_prefill
        self.cuda_graph_all_prefills = False
        self._prefix_coordination_waits = 0
        self._loop = asyncio.new_event_loop()
        self.waiting_request_ids: deque = deque()
        self.requests = {}
        self._generation_epoch = None


class TestPrefixCachingCore(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_hash_computation(self):
        # determinism and range
        tokens = self._prompt(32)
        h1 = compute_block_hashes_batched(tokens, 32)
        h2 = compute_block_hashes_batched(tokens, 32)
        assert h1 == h2 and len(h1) == 1 and h1[0] >= 1
        assert compute_block_hashes_batched(self._prompt(32, offset=1), 32)[0] != h1[0]

        # parent chaining: 4 blocks of all-zero tokens produce distinct hashes
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        zeros = torch.zeros(bs * 4, device=torch.cuda.current_device(), dtype=torch.long)
        hashes = compute_block_hashes_batched(zeros, bs)
        assert len(hashes) == 4 and len(set(hashes)) == 4

        # edge cases: short, empty, long
        assert compute_block_hashes_batched(self._prompt(bs // 2), bs) == []
        empty = torch.tensor([], device=torch.cuda.current_device(), dtype=torch.long)
        assert compute_block_hashes_batched(empty, bs) == []
        long_h = compute_block_hashes_batched(
            torch.arange(bs * 120, device=torch.cuda.current_device(), dtype=torch.long), bs
        )
        assert len(long_h) == 120 and all(v > 0 for v in long_h)

    @pytest.mark.internal
    def test_hash_collision_resistance(self):
        """Regression tests: old polynomial collision attacks must fail with SHA-256."""
        bs = 32

        # V2 regression: algebraic attack (token[j] += 31, token[j+1] -= 1)
        # This was a zero-delta exploit against the old polynomial hash.
        tokens = self._prompt(bs)
        collision = tokens.clone()
        collision[0] += 31
        collision[1] -= 1
        h_orig = compute_block_hashes_batched(tokens, bs)
        h_coll = compute_block_hashes_batched(collision, bs)
        assert h_orig != h_coll, "V2 algebraic collision: token[j]+=31, token[j+1]-=1"

        # V2 at different positions within the block
        for j in range(bs - 1):
            c = tokens.clone()
            c[j] += 31
            c[j + 1] -= 1
            assert compute_block_hashes_batched(c, bs) != h_orig, f"V2 at position {j}"

        # V2 across multiple blocks: modify one block, verify all downstream hashes change
        tokens_multi = self._prompt(bs * 4)
        h_multi = compute_block_hashes_batched(tokens_multi, bs)
        modified = tokens_multi.clone()
        modified[0] += 31
        modified[1] -= 1
        h_mod = compute_block_hashes_batched(modified, bs)
        assert h_mod[0] != h_multi[0], "modified block hash must differ"
        # Parent chaining: all subsequent blocks must also differ
        for i in range(1, 4):
            assert h_mod[i] != h_multi[i], f"parent chain: block {i} must differ"

        # V2 generalized: arbitrary linear combinations (token[j] += k*31, token[j+1] -= k)
        for k in [1, 2, 5, 100]:
            c = tokens.clone()
            c[0] += k * 31
            c[1] -= k
            assert compute_block_hashes_batched(c, bs) != h_orig, f"V2 generalized k={k}"

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
    def test_duplicate_registration_does_not_duplicate_parent_edge(self):
        """Re-registering one physical child must not strand its parent in the LRU forest."""
        ctx = self._ctx(rounder=1)
        alloc = ctx.kv_block_allocator
        block_ids = alloc.allocate_memory_blocks(2)
        assert block_ids is not None
        parent_id, child_id = block_ids.tolist()
        parent_hash, child_hash = 101, 102

        alloc.register_kv_block_hashes(
            [parent_id, child_id], [parent_hash, child_hash], parent_hashes=[0, parent_hash]
        )
        assert alloc.block_child_count[parent_id].item() == 1

        # Hybrid prefix recovery can recompute a KV-matched block after its
        # corresponding Mamba state was evicted. That path re-registers the
        # same physical child and parent edge.
        alloc.register_kv_block_hashes([child_id], [child_hash], parent_hashes=[parent_hash])
        assert alloc.block_child_count[parent_id].item() == 1

        alloc.release_memory_blocks(block_ids)
        assert alloc.evict_lru_blocks(2)
        assert alloc.kv_hash_to_block_id == {}

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


class TestDisabledAndEngineScheduling(PrefixCachingTestBase):

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
    @pytest.mark.parametrize("hybrid", [False, True], ids=["gpt", "hybrid"])
    def test_epoch_change_invalidates_cache_without_disrupting_live_request(self, hybrid):
        ctx = self._ctx(
            max_tokens=256,
            max_requests=8,
            mamba_config=self._mamba_config() if hybrid else None,
            prefix_caching_mamba_gb=0.01 if hybrid else None,
        )
        alloc = ctx.kv_block_allocator
        engine = self._engine(ctx)
        bs = ctx.block_size_tokens

        live = self._req(ctx, self._prompt(2 * bs), request_id=1)
        ctx.add_request(live)
        self._add_to_waiting(engine, ctx, live)
        live_blocks = self._block_ids(ctx, 0, 2)
        live_hashes = set(live.precomputed_block_hashes)

        cached = self._req(ctx, self._prompt(bs, offset=10_000), request_id=2)
        ctx.add_request(cached)
        (cached_block,) = self._block_ids(ctx, 1, 1)
        cached_hash = cached.precomputed_block_hashes[0]
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[cached_block].item() == 0

        routing = np.ones((bs, 1, 1), dtype=np.int64)
        alloc.block_routing[live_blocks[0]] = routing.copy()
        alloc.block_routing[cached_block] = routing.copy()
        pool_avail_before = alloc.pool_avail
        if hybrid:
            mamba_alloc = ctx.mamba_slot_allocator
            assert mamba_alloc._has_intermediates

        engine._set_generation_epoch(1)

        assert live.enable_prefix_caching is False
        assert live.precomputed_block_hashes == []
        assert live.kv_cache_epoch == [(0, 1)]
        assert live_hashes.isdisjoint(alloc.kv_hash_to_block_id)
        assert cached_hash not in alloc.kv_hash_to_block_id
        assert all(alloc.block_hashes[block_id].item() == -1 for block_id in live_blocks)
        assert all(alloc.block_ref_counts[block_id].item() == 1 for block_id in live_blocks)
        assert alloc.pool_avail == pool_avail_before + 1
        assert live_blocks[0] in alloc.block_routing
        assert cached_block not in alloc.block_routing
        replacement = alloc.allocate_memory_blocks(1)
        assert replacement is not None and replacement.item() == cached_block

        if hybrid:
            assert not mamba_alloc._has_intermediates
            assert mamba_alloc.free_count == mamba_alloc.max_slots
            durable_free_before = mamba_alloc.free_count
            uncacheable = self._req(
                ctx, self._prompt(2 * bs, offset=20_000), request_id=3, enable_prefix_caching=False
            )
            ctx.add_request(uncacheable)
            assert not mamba_alloc._has_intermediates
            mamba_alloc.commit_intermediate_states()
            assert mamba_alloc.free_count == durable_free_before

    @pytest.mark.internal
    def test_disabled_mode(self):
        # no sharing
        ctx = self._ctx(enable_prefix_caching=False)
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt.clone(), enable_prefix_caching=False))
        r1 = set(self._block_ids(ctx, 0, 2))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2, enable_prefix_caching=False))
        r2 = set(self._block_ids(ctx, 1, 2))
        assert r1.isdisjoint(r2)

        # no caching attrs on disabled allocator
        alloc_d = ctx.kv_block_allocator
        assert not hasattr(alloc_d, 'block_hashes')
        assert not hasattr(alloc_d, 'kv_hash_to_block_id')
        assert not hasattr(alloc_d, 'block_ref_counts')

        # REF_ZERO lacks timestamps
        ctx_rz = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        assert not hasattr(ctx_rz.kv_block_allocator, 'block_timestamps')

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
        self._mamba_allocate_and_register(ctx5, self._block_ids(ctx5, 0, 3)[:2])
        assert len(alloc5.kv_hash_to_block_id) == 3 and len(msa5.hash_to_block_id) == 2

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
    def test_hybrid_prefix_caching_without_mamba_budget_warns(self, caplog):
        # Memory-only mode: prefix caching on a hybrid model without a Mamba cache
        # budget is allowed (KV prefixes deduplicated for memory savings) but must
        # warn that Mamba state caching and prefill skipping are disabled, and must
        # not allocate a slot allocator.
        import logging as _logging

        with caplog.at_level(_logging.WARNING):
            ctx = self._ctx(
                mamba_config=self._mamba_config(),
                enable_prefix_caching=True,
                prefix_caching_mamba_gb=None,
            )
        assert ctx.is_hybrid_model
        assert ctx.mamba_slot_allocator is None
        assert "memory-only" in caplog.text

    @pytest.mark.internal
    def test_mamba_cache_budget_too_small_raises(self):
        # The CUDA-graph extraction scratch (sized to the per-step token-budget
        # cap, max_mamba_intermediate_states_per_step) is reserved from
        # prefix_caching_mamba_gb before the durable cache is sized. A budget too
        # small to fit the scratch plus at least one durable slot is a hard
        # configuration error, not a silent over-allocation (which previously
        # could OOM at startup).
        with pytest.raises(ValueError, match="prefix cache budget"):
            self._mctx(prefix_caching_mamba_gb=1e-5)

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

    @pytest.mark.internal
    def test_max_intermediate_states_per_step_formula(self):
        # The extraction buffers are sized by the tighter of two per-step bounds:
        #   token-based:   ceil(max_tokens / block_size) + 1
        #   request-based: MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * max_requests
        import math

        from megatron.core.inference.contexts.mamba_slot_allocator import (
            MAX_INTERMEDIATE_OFFSETS_PER_REQUEST,
        )

        def token_based(ctx):
            return math.ceil(ctx.max_tokens / ctx.block_size_tokens)

        def request_based(ctx):
            return MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * ctx.max_requests

        # Token-limited regime: many requests, so the token budget is tighter.
        ctx = self._mctx(block_size_tokens=256, max_tokens=2048)
        assert ctx.max_requests >= 3  # ensure this regime is actually token-limited
        expected = min(token_based(ctx), request_based(ctx))
        assert expected == token_based(ctx)  # token bound wins here
        assert ctx.max_mamba_intermediate_states_per_step == expected
        # The single value is shared everywhere it's consumed.
        assert ctx.mamba_slot_allocator.max_intermediate_count == expected
        assert ctx.mamba_metadata.max_intermediate_count == expected
        assert ctx.mamba_slot_allocator.intermediate_ssm_out.shape[1] == expected

        # Request-limited regime: few requests but a large token budget, so
        # 3 * max_requests is the tighter bound. This is the case the token-only
        # formula over-allocated for (e.g. 1 request + 16384 tokens once reserved
        # 65 scratch slots a single request could never fill).
        ctx2 = self._mctx(block_size_tokens=256, max_tokens=2048, max_requests=2)
        expected2 = min(token_based(ctx2), request_based(ctx2))
        assert expected2 == request_based(ctx2)  # request bound wins here
        assert expected2 < token_based(ctx2)  # ...and it is strictly tighter
        assert ctx2.max_mamba_intermediate_states_per_step == expected2
        assert ctx2.mamba_slot_allocator.max_intermediate_count == expected2
        assert ctx2.mamba_metadata.max_intermediate_count == expected2
        assert ctx2.mamba_slot_allocator.intermediate_ssm_out.shape[1] == expected2

    @pytest.mark.internal
    def test_intermediate_count_bounded_by_token_budget(self):
        # Claim: a single engine step emits at most max_tokens / block_size Mamba
        # intermediate states, regardless of how many prefill requests it packs.
        # Fill the token budget with fresh multi-block prefills and confirm the
        # extracted count never exceeds the scratch buffer.
        bs = 256
        ctx = self._mctx(block_size_tokens=bs, max_tokens=2048, max_sequence_length=4096)
        budget = ctx.max_mamba_intermediate_states_per_step

        # Non-block-aligned 2.5-block prefills (each crosses a block boundary on a
        # mamba-chunk multiple -> one intermediate offset). Distinct content so
        # they never prefix-match one another.
        per_req = bs * 2 + bs // 2  # 640 tokens
        n = ctx.max_tokens // per_req
        assert n >= 2
        for i in range(n):
            ctx.add_request(
                self._req(ctx, self._prompt(per_req, offset=i * 100000), request_id=i + 1)
            )

        # Drive the step's metadata computation (populates intermediate_count).
        ctx.initialize_attention_state()
        ctx.transfer_bookkeeping_to_gpu()

        md = ctx.mamba_metadata
        # Extraction actually fired (guards against a silent no-op test)...
        assert md.intermediate_count > 0
        # ...and the packed step never exceeds the token-budget bound.
        assert md.intermediate_count <= budget
        assert md.intermediate_count == sum(md.per_request_intermediate_counts)

    @pytest.mark.internal
    def test_intermediate_count_fills_scratch_buffer(self):
        # Reviewer follow-up: drive a single step that consumes nearly the whole
        # scratch buffer (not just a few slots) and confirm it is never overrun.
        #
        # Realistic config: attention block_size=256, mamba_chunk_size=128. Since
        # 256 is a multiple of 128, a 256-aligned block boundary is also a mamba
        # chunk boundary (extractable); the mamba boundary at 128 is NOT a block
        # boundary, so it is never a candidate.
        #
        # Each request is a fresh 257-token prompt (one token past a block): it
        # crosses exactly one block boundary at token 256 -> exactly one
        # intermediate offset, while consuming ~one block of tokens. Packing the
        # token budget with these drives intermediate_count to max_tokens // 257,
        # close to the token-budget bound -- filling nearly every scratch slot,
        # unlike test_intermediate_count_bounded_by_token_budget (~1/3 of them).
        import math

        bs = 256
        ctx = self._mctx(block_size_tokens=bs, max_tokens=2048, max_sequence_length=4096)
        assert ctx.mamba_chunk_size == 128  # bs=256 is a multiple of mamba chunk 128
        budget = ctx.max_mamba_intermediate_states_per_step

        per_req = bs + 1  # 257: crosses the block boundary at bs=256 (a mamba-chunk multiple)
        n = ctx.max_tokens // per_req
        assert n >= 2
        assert n <= ctx.max_requests  # all packed into a single step
        for i in range(n):
            ctx.add_request(
                self._req(ctx, self._prompt(per_req, offset=i * 100000), request_id=i + 1)
            )

        ctx.initialize_attention_state()
        ctx.transfer_bookkeeping_to_gpu()

        md = ctx.mamba_metadata
        # Every request contributed exactly one intermediate offset...
        assert md.intermediate_count == n
        assert md.intermediate_count == sum(md.per_request_intermediate_counts)
        # ...the step never overruns the scratch buffer...
        assert md.intermediate_count <= budget
        # ...and it fills all but a small, *derived* deficit: the block-spillover
        # (per_req > bs, so fewer requests fit than there are blocks). n <=
        # max_requests forces the token-based bound, so budget == ceil(max_tokens / bs).
        assert budget == math.ceil(ctx.max_tokens / bs)
        expected_unfilled = math.ceil(ctx.max_tokens / bs) - n
        assert budget - md.intermediate_count == expected_unfilled

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
    def test_allocate_slots_batch(self):
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
        # Budget must cover the CUDA-graph extraction scratch (3 * max_requests
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


class TestPerBlockRouting(PrefixCachingTestBase):
    """Tests for per-block routing storage and reconstruction."""

    @pytest.mark.internal
    def test_store_and_get_block_routing(self):
        """Verify store_block_routing / get_block_routing round-trip."""
        ctx = self._ctx()
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens
        num_layers, topk = 4, 2

        # Allocate a block
        block_ids = alloc.allocate_memory_blocks(1)
        bid = block_ids[0].item()

        # Store routing for some positions
        positions = np.array([0, 1, 2])
        routing = np.random.randint(-100, 100, size=(3, num_layers, topk), dtype=np.int16)
        alloc.store_block_routing(bid, positions, routing)

        # Retrieve and verify
        stored = alloc.get_block_routing(bid)
        assert stored is not None
        assert isinstance(stored, np.ndarray)
        assert stored.shape == (bs, num_layers, topk)
        assert np.allclose(stored[:3], routing)
        # Remaining positions should be zero
        assert (stored[3:] == 0).all()

    @pytest.mark.internal
    def test_routing_cleared_on_allocate(self):
        """Routing data is cleared when a block is re-allocated."""
        ctx = self._ctx(enable_prefix_caching=False)
        alloc = ctx.kv_block_allocator

        # Allocate, store routing, release, re-allocate
        block_ids = alloc.allocate_memory_blocks(1)
        bid = block_ids[0].item()
        positions = np.array([0])
        routing = np.random.randint(-100, 100, size=(1, 4, 2), dtype=np.int16)
        alloc.store_block_routing(bid, positions, routing)
        assert alloc.get_block_routing(bid) is not None

        alloc.release_memory_blocks(block_ids)
        # After release, routing still present (persists until re-alloc)
        assert alloc.get_block_routing(bid) is not None

        # Re-allocate the same block
        new_ids = alloc.allocate_memory_blocks(1)
        new_bid = new_ids[0].item()
        # The re-allocated block should have routing cleared
        assert alloc.get_block_routing(new_bid) is None

    @pytest.mark.internal
    def test_routing_cleared_on_reset(self):
        """Routing data is cleared on allocator reset."""
        ctx = self._ctx()
        alloc = ctx.kv_block_allocator

        block_ids = alloc.allocate_memory_blocks(1)
        bid = block_ids[0].item()
        alloc.store_block_routing(
            bid, np.array([0]), np.random.randint(-100, 100, size=(1, 4, 2), dtype=np.int16)
        )
        assert alloc.get_block_routing(bid) is not None

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

        # Store routing for both blocks
        for bid in [b0, b1]:
            alloc.store_block_routing(
                bid, np.arange(bs), np.random.randint(-100, 100, size=(bs, 4, 2), dtype=np.int16)
            )

        # Release blocks (REF_ZERO deregisters immediately)
        blocks = ctx.request_to_kv_block_ids[0]
        valid_blocks = blocks[blocks >= 0]
        alloc.release_memory_blocks(valid_blocks)

        # Routing data should still be present
        assert alloc.get_block_routing(b0) is not None
        assert alloc.get_block_routing(b1) is not None

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
        alloc.store_block_routing(
            bids[2],
            np.arange(partial),
            np.arange(partial * num_layers * topk, dtype=np.int16).reshape(
                partial, num_layers, topk
            )
            + bids[2],
        )

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
        expected_partial = (
            np.arange(partial * num_layers * topk, dtype=np.int16).reshape(
                partial, num_layers, topk
            )
            + bids[2]
        )
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
        # LRU + prefix caching enabled: reset(preserve_prefix_cache=True) keeps the
        # KV hash index (so an idle dummy_forward does not wipe cross-request reuse),
        # while a plain reset() clears it.
        ctx = self._ctx(enable_prefix_caching=True)
        bs = ctx.block_size_tokens
        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        cached = dict(ctx.kv_block_allocator.kv_hash_to_block_id)
        assert len(cached) == 2

        ctx.reset(preserve_prefix_cache=True)
        assert ctx.kv_block_allocator.kv_hash_to_block_id == cached  # preserved

        ctx.reset()  # default: full reset
        assert len(ctx.kv_block_allocator.kv_hash_to_block_id) == 0  # cleared

    @pytest.mark.internal
    def test_reset_disabled_ignores_preserve_flag(self):
        # When prefix caching is disabled, preserve_prefix_cache=True still performs
        # a full reset: step_count returns to 0.
        ctx_off = self._ctx(enable_prefix_caching=False)
        ctx_off.step_count = 7
        ctx_off.reset(preserve_prefix_cache=True)
        assert ctx_off.step_count == 0

        # With caching ON, preserve keeps step_count monotonic (for logging cadence).
        ctx_on = self._ctx(enable_prefix_caching=True)
        ctx_on.step_count = 7
        ctx_on.reset(preserve_prefix_cache=True)
        assert ctx_on.step_count == 7

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


# Each ownership entry names an assertion returned by the executed row.  Keeping
# this table beside the matrix makes additions reviewable: a feature/policy pair
# is not owned merely because a flag was passed to a constructor.
PREFIX_CACHE_CONTEXT_PAIR_OWNERS = {
    "exact-prefix×lru": "exact-prefix-hit",
    "exact-prefix×ref-zero": "exact-prefix-hit",
    "partial-prefix×lru": "partial-prefix-hit",
    "partial-prefix×ref-zero": "partial-prefix-hit",
    "missing-prefix×lru": "missing-prefix-observed",
    "missing-prefix×ref-zero": "missing-prefix-observed",
    "full-block-boundary×lru": "full-block-clamp-executed",
    "full-block-boundary×ref-zero": "full-block-clamp-executed",
    "concurrent-sharing×lru": "concurrent-refcounts-churned",
    "concurrent-sharing×ref-zero": "concurrent-refcounts-churned",
    "mixed-cached-fresh×lru": "mixed-query-lengths-executed",
    "mixed-cached-fresh×ref-zero": "mixed-query-lengths-executed",
}

PREFIX_CACHE_CONTEXT_CASES = [
    pytest.param(feature, policy, id=f"{feature}-{policy.name.lower()}")
    for feature in (
        "exact-prefix",
        "partial-prefix",
        "missing-prefix",
        "full-block-boundary",
        "concurrent-sharing",
        "mixed-cached-fresh",
    )
    for policy in (PrefixCachingEvictionPolicy.LRU, PrefixCachingEvictionPolicy.REF_ZERO)
]


class TestPrefixCachePolicyStressMatrix(PrefixCachingTestBase):
    """Small allocator matrix; real-model combinations live in the engine matrix below."""

    def _run_policy_cycles(self, policy):
        ctx = self._ctx(
            buffer_size_gb=0.001,
            rounder=1,
            max_requests=12,
            max_tokens=512,
            prefix_caching_eviction_policy=policy,
        )
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens
        activations = set()
        prior_pressure_hashes = set()
        ref_zero_reuse_events = 0
        for cycle in range(3):
            prompt = self._prompt(2 * bs, offset=cycle * 10_000)
            producer = self._req(ctx, prompt.clone(), request_id=cycle * 10)
            producer_idx = ctx.total_request_count
            ctx.add_request(producer)
            producer_blocks = self._block_ids(ctx, producer_idx, 2)
            assert not prior_pressure_hashes.intersection(producer.precomputed_block_hashes)

            if policy == PrefixCachingEvictionPolicy.LRU:
                ctx.release_memory_blocks_from_request_indexes(torch.tensor([producer_idx]))
                assert all(alloc.block_ref_counts[bid].item() == 0 for bid in producer_blocks)
                assert all(
                    block_hash in alloc.kv_hash_to_block_id
                    for block_hash in producer.precomputed_block_hashes
                )
                activations.add("lru-retained-reuse")

            # Pinned filler blocks leave no free capacity.  The cache transition
            # below therefore cannot pass without sharing or policy-driven reuse.
            filler = alloc.allocate_memory_blocks(alloc.pool_avail)
            assert filler is not None and alloc.pool_avail == 0
            # Allocator returns a view into block_bag, which later eviction mutates.
            filler = filler.clone()
            activations.add("pool-exhausted")

            follower = self._req(ctx, prompt.clone(), request_id=cycle * 10 + 1)
            follower_idx = ctx.total_request_count
            ctx.add_request(follower)
            assert self._block_ids(ctx, follower_idx, 2) == producer_blocks
            assert follower.num_cached_tokens == 2 * bs
            assert alloc.pool_avail == 0

            if policy == PrefixCachingEvictionPolicy.REF_ZERO:
                assert all(alloc.block_ref_counts[bid].item() == 2 for bid in producer_blocks)
                activations.add("ref-zero-live-sharing")
                ctx.release_memory_blocks_from_request_indexes(torch.tensor([follower_idx]))
                ctx.release_memory_blocks_from_request_indexes(torch.tensor([producer_idx]))
                assert not any(
                    block_hash in alloc.kv_hash_to_block_id
                    for block_hash in producer.precomputed_block_hashes
                )
                assert all(alloc.block_hashes[bid].item() == -1 for bid in producer_blocks)
            else:
                assert all(alloc.block_ref_counts[bid].item() == 1 for bid in producer_blocks)
                ctx.release_memory_blocks_from_request_indexes(torch.tensor([follower_idx]))

            # Allocate an unrelated two-block prefix.  REF_ZERO must physically
            # recycle the just-released blocks; LRU must evict the retained prefix.
            cached_blocks_before = dict(alloc.kv_hash_to_block_id)
            pressure_prompt = self._prompt(2 * bs, offset=100_000 + cycle * 10_000)
            pressure = self._req(ctx, pressure_prompt, request_id=cycle * 10 + 2)
            pressure_idx = ctx.total_request_count
            ctx.add_request(pressure)
            pressure_blocks = self._block_ids(ctx, pressure_idx, 2)
            activations.add("physical-blocks-reused")

            if policy == PrefixCachingEvictionPolicy.LRU:
                removed_ids = {
                    block_id
                    for block_hash, block_id in cached_blocks_before.items()
                    if block_hash not in alloc.kv_hash_to_block_id
                }
                assert set(pressure_blocks) == removed_ids
                revisit = self._req(ctx, prompt.clone(), request_id=cycle * 10 + 3)
                matched, *_ = ctx._compute_prefix_match(revisit, len(prompt))
                if set(producer_blocks) == removed_ids:
                    assert matched == []
                activations.add("lru-prefix-evicted")
            else:
                assert set(pressure_blocks) == set(producer_blocks)
                assert all(alloc.block_ref_counts[bid].item() == 1 for bid in pressure_blocks)
                ref_zero_reuse_events += 1

            ctx.release_memory_blocks_from_request_indexes(torch.tensor([pressure_idx]))
            alloc.release_memory_blocks(filler)
            prior_pressure_hashes = set(pressure.precomputed_block_hashes)

        assert ctx.total_request_count == 9
        if policy == PrefixCachingEvictionPolicy.REF_ZERO:
            assert ref_zero_reuse_events == 3
            assert alloc.kv_hash_to_block_id == {}
            activations.add("three-ref-zero-deregister-recycle-events")
        activations.add("three-policy-cycles")
        return activations

    def _apply_local_churn(self, ctx, producer, producer_blocks):
        """Force a policy-specific block recycle after a feature assertion."""
        alloc = ctx.kv_block_allocator
        policy = alloc.prefix_caching_eviction_policy
        live_indexes = torch.arange(ctx.total_request_count)
        ctx.release_memory_blocks_from_request_indexes(live_indexes)

        if policy == PrefixCachingEvictionPolicy.LRU:
            cached_before = dict(alloc.kv_hash_to_block_id)
            assert cached_before
            filler = alloc.allocate_memory_blocks(alloc.pool_avail)
            assert filler is not None and alloc.pool_avail == 0
        else:
            assert alloc.kv_hash_to_block_id == {}
            assert all(alloc.block_hashes[bid].item() == -1 for bid in producer_blocks)
            filler = alloc.allocate_memory_blocks(alloc.pool_avail)
            assert filler is not None and alloc.pool_avail == 0
            producer_tensor = torch.tensor(producer_blocks, dtype=torch.int32)
            assert set(producer_blocks) <= set(filler.tolist())
            alloc.release_memory_blocks(producer_tensor)
            assert alloc.pool_avail == len(producer_blocks)

        pressure = self._req(
            ctx, self._prompt(3 * ctx.block_size_tokens, offset=200_000), request_id=100
        )
        pressure_idx = ctx.total_request_count
        ctx.add_request(pressure)
        pressure_blocks = set(self._block_ids(ctx, pressure_idx, 3))

        if policy == PrefixCachingEvictionPolicy.LRU:
            removed_ids = {
                block_id
                for block_hash, block_id in cached_before.items()
                if block_hash not in alloc.kv_hash_to_block_id
            }
            assert pressure_blocks == removed_ids
            return "lru-local-eviction"

        assert pressure_blocks == set(producer_blocks)
        assert not any(
            block_hash in alloc.kv_hash_to_block_id
            for block_hash in producer.precomputed_block_hashes
        )
        return "ref-zero-local-recycle"

    def _run_feature_probe(self, feature, policy):
        ctx = self._ctx(
            buffer_size_gb=0.001,
            rounder=1,
            max_requests=12,
            max_tokens=512,
            prefix_caching_eviction_policy=policy,
        )
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens
        prompt = self._prompt(3 * bs)
        producer = self._req(ctx, prompt.clone())
        ctx.add_request(producer)
        producer_blocks = self._block_ids(ctx, 0, 3)
        if policy == PrefixCachingEvictionPolicy.LRU:
            ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))

        if feature == "exact-prefix":
            probe = self._req(ctx, prompt.clone(), request_id=2)
            matched, *_ = ctx._compute_prefix_match(probe, len(prompt))
            assert matched == producer_blocks
            ctx.add_request(probe)
            assert probe.num_cached_tokens == 3 * bs
            activation = "exact-prefix-hit"

        elif feature == "partial-prefix":
            partial = torch.cat((prompt[: 2 * bs], self._prompt(bs, offset=50_000)))
            probe = self._req(ctx, partial, request_id=2)
            matched, *_ = ctx._compute_prefix_match(probe, len(partial))
            assert matched == producer_blocks[:2]
            ctx.add_request(probe)
            assert probe.num_cached_tokens == 2 * bs
            activation = "partial-prefix-hit"

        elif feature == "missing-prefix":
            probe = self._req(ctx, self._prompt(3 * bs, offset=50_000), request_id=2)
            matched, *_ = ctx._compute_prefix_match(probe, 3 * bs)
            assert matched == []
            ctx.add_request(probe)
            assert probe.num_cached_tokens == 0
            activation = "missing-prefix-observed"

        elif feature == "full-block-boundary":
            probe = self._req(ctx, prompt.clone(), request_id=2)
            matched, _, _, _, skipped, effective = ctx._compute_prefix_match(probe, len(prompt))
            assert matched == producer_blocks
            assert skipped == 2 * bs
            assert effective == bs
            ctx.add_request(probe)
            activation = "full-block-clamp-executed"

        elif feature == "concurrent-sharing":
            for request_id in range(2, 5):
                ctx.add_request(self._req(ctx, prompt.clone(), request_id=request_id))
            expected_refs = 3 + int(policy == PrefixCachingEvictionPolicy.REF_ZERO)
            assert all(
                alloc.block_ref_counts[bid].item() == expected_refs for bid in producer_blocks
            )
            ctx.release_memory_blocks_from_request_indexes(torch.tensor([1, 2, 3]))
            remaining_refs = int(policy == PrefixCachingEvictionPolicy.REF_ZERO)
            assert all(
                alloc.block_ref_counts[bid].item() == remaining_refs for bid in producer_blocks
            )
            activation = "concurrent-refcounts-churned"

        else:
            assert feature == "mixed-cached-fresh"
            cached = self._req(ctx, prompt.clone(), request_id=2)
            fresh = self._req(ctx, self._prompt(3 * bs, offset=50_000), request_id=3)
            ctx.add_request(cached)
            ctx.add_request(fresh)
            assert ctx.request_query_lengths[1].item() == bs
            assert ctx.request_query_lengths[2].item() == 3 * bs
            assert cached.num_cached_tokens == 3 * bs
            assert fresh.num_cached_tokens == 0
            activation = "mixed-query-lengths-executed"

        churn = self._apply_local_churn(ctx, producer, producer_blocks)
        return activation, churn

    @pytest.mark.internal
    @pytest.mark.parametrize("feature,policy", PREFIX_CACHE_CONTEXT_CASES)
    def test_context_policy_matrix(self, feature, policy):
        activation, churn = self._run_feature_probe(feature, policy)
        pair = f"{feature}×{'lru' if policy == PrefixCachingEvictionPolicy.LRU else 'ref-zero'}"
        assert PREFIX_CACHE_CONTEXT_PAIR_OWNERS[pair] == activation
        expected_churn = (
            "lru-local-eviction"
            if policy == PrefixCachingEvictionPolicy.LRU
            else "ref-zero-local-recycle"
        )
        assert churn == expected_churn

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "policy",
        [PrefixCachingEvictionPolicy.LRU, PrefixCachingEvictionPolicy.REF_ZERO],
        ids=["lru", "ref-zero"],
    )
    def test_policy_pressure_cycles(self, policy):
        """Exercise each eviction policy once instead of repeating it for every feature row."""
        policy_activations = self._run_policy_cycles(policy)
        assert "three-policy-cycles" in policy_activations
        if policy == PrefixCachingEvictionPolicy.LRU:
            assert {"lru-retained-reuse", "lru-prefix-evicted"} <= policy_activations
        else:
            assert {
                "ref-zero-live-sharing",
                "physical-blocks-reused",
                "three-ref-zero-deregister-recycle-events",
            } <= policy_activations


PREFIX_CACHE_ENGINE_CASES = [
    pytest.param(dict(name="gpt-dp8", model="gpt", feature="base"), id="gpt-dp8-output"),
    pytest.param(dict(name="hybrid-dp8", model="hybrid", feature="base"), id="hybrid-dp8-output"),
    pytest.param(dict(name="gpt-tp4", model="gpt", feature="tp", tp=4), id="gpt-tp4"),
    pytest.param(dict(name="hybrid-tp4", model="hybrid", feature="tp", tp=4), id="hybrid-tp4"),
    pytest.param(dict(name="gpt-pp4", model="gpt", feature="pp", pp=4), id="gpt-pp4"),
    pytest.param(
        dict(name="hybrid-pp2", model="hybrid", feature="pp", pp=2, num_tokens_to_generate=3),
        id="hybrid-pp2",
    ),
    pytest.param(
        dict(name="gpt-tp2-pp2-sp", model="gpt", feature="mixed-parallel", tp=2, pp=2, sp=True),
        id="gpt-tp2-pp2-sp",
    ),
    pytest.param(
        dict(
            name="hybrid-tp2-pp2-sp", model="hybrid", feature="mixed-parallel", tp=2, pp=2, sp=True
        ),
        id="hybrid-tp2-pp2-sp",
    ),
    pytest.param(dict(name="gpt-ep4", model="gpt", feature="moe", ep=4), id="gpt-ep4-moe"),
    pytest.param(dict(name="hybrid-ep2", model="hybrid", feature="moe", ep=2), id="hybrid-ep2-moe"),
    pytest.param(dict(name="gpt-chunked", model="gpt", feature="chunked"), id="gpt-chunked"),
    pytest.param(
        dict(name="hybrid-chunked", model="hybrid", feature="chunked"), id="hybrid-chunked"
    ),
    pytest.param(
        dict(name="gpt-cuda-graph", model="gpt", feature="cuda-graph"), id="gpt-cuda-graph"
    ),
    pytest.param(
        dict(name="hybrid-cuda-graph", model="hybrid", feature="cuda-graph"), id="hybrid-cuda-graph"
    ),
    pytest.param(dict(name="gpt-mtp", model="gpt", feature="mtp"), id="gpt-mtp"),
    pytest.param(dict(name="hybrid-mtp", model="hybrid", feature="mtp"), id="hybrid-mtp"),
    pytest.param(dict(name="gpt-logprobs", model="gpt", feature="logprobs"), id="gpt-logprobs"),
    pytest.param(
        dict(name="hybrid-logprobs", model="hybrid", feature="logprobs"), id="hybrid-logprobs"
    ),
    pytest.param(dict(name="gpt-offload", model="gpt", feature="offload"), id="gpt-offload-resume"),
    pytest.param(
        dict(name="hybrid-offload", model="hybrid", feature="offload"), id="hybrid-offload-resume"
    ),
    pytest.param(
        dict(name="gpt-recompute", model="gpt", feature="recompute"), id="gpt-recompute-resume"
    ),
    pytest.param(
        dict(name="hybrid-recompute", model="hybrid", feature="recompute"),
        id="hybrid-recompute-resume",
    ),
    pytest.param(dict(name="gpt-epoch", model="gpt", feature="epoch"), id="gpt-epoch-invalidation"),
    pytest.param(
        dict(name="hybrid-epoch", model="hybrid", feature="epoch"), id="hybrid-epoch-invalidation"
    ),
    pytest.param(
        dict(name="gpt-request-eviction", model="gpt", feature="request-eviction"),
        id="gpt-request-eviction-checkpoint-resume",
    ),
    pytest.param(dict(name="gpt-uvm", model="gpt", feature="uvm"), id="gpt-uvm-backed-lifecycle"),
]

PREFIX_CACHE_ENGINE_LRU_CASES = {
    "gpt-dp8",
    "hybrid-tp4",
    "gpt-pp4",
    "hybrid-tp2-pp2-sp",
    "gpt-ep4",
    "hybrid-chunked",
    "gpt-cuda-graph",
    "hybrid-mtp",
    "gpt-logprobs",
    "hybrid-offload",
    "gpt-recompute",
    "gpt-epoch",
    "hybrid-epoch",
    "gpt-request-eviction",
    "gpt-uvm",
    "gpt-http-zmq",
}

PREFIX_CACHE_ENGINE_PAIR_OWNERS = {
    "gpt-dp8×lru": "real-engine-output-parity",
    "hybrid-dp8×ref-zero": "real-engine-output-parity",
    "gpt-tp4×ref-zero": "tensor-parallel-prefix-hit",
    "hybrid-tp4×lru": "tensor-parallel-prefix-hit",
    "gpt-pp4×lru": "pipeline-parallel-prefix-hit",
    "hybrid-pp2×ref-zero": "pipeline-parallel-prefix-hit",
    "gpt-tp2-pp2-sp×ref-zero": "mixed-parallel-prefix-hit",
    "hybrid-tp2-pp2-sp×lru": "mixed-parallel-prefix-hit",
    "gpt-ep4×lru": "moe-expert-parallel-forward-with-prefix-hits",
    "hybrid-ep2×ref-zero": "moe-expert-parallel-forward-with-prefix-hits",
    "gpt-chunked×ref-zero": "chunked-prefix-reuse",
    "hybrid-chunked×lru": "chunked-prefix-reuse",
    "gpt-cuda-graph×lru": "cuda-graph-replay-with-prefix-hits",
    "hybrid-cuda-graph×ref-zero": "cuda-graph-replay-with-prefix-hits",
    "gpt-mtp×ref-zero": "mtp-speculative-proposals-with-prefix-hits",
    "hybrid-mtp×lru": "mtp-speculative-proposals-with-prefix-hits",
    "gpt-logprobs×lru": "logprob-parity",
    "hybrid-logprobs×ref-zero": "logprob-parity",
    "gpt-offload×ref-zero": "offload-prefix-resume",
    "hybrid-offload×lru": "offload-prefix-resume",
    "gpt-recompute×lru": "recompute-prefix-resume",
    "hybrid-recompute×ref-zero": "recompute-prefix-resume",
    "gpt-epoch×lru": "epoch-signal-invalidation-rebuild",
    "hybrid-epoch×lru": "epoch-signal-invalidation-rebuild",
    "gpt-request-eviction×lru": "request-eviction-checkpoint-resume",
    "gpt-uvm×lru": "uvm-backed-prefix-lifecycle",
}


class TestPrefixCacheRealEngineMatrix(DynamicInferenceEngineTestBase):
    """Real model/engine pair coverage; every row executes three sharing waves."""

    @classmethod
    def _build_inference_context(
        cls, test_config, transformer_config, requests, mamba_inference_state_config=None
    ):
        # The shared engine harness intentionally defaults hybrid prefix caching
        # to KV-only mode.  This matrix supplies a durable Mamba budget so the
        # hybrid rows execute KV matching, state extraction, and state restore.
        return DynamicInferenceContext(
            model_config=transformer_config,
            inference_config=InferenceConfig(
                max_sequence_length=test_config.max_sequence_length,
                num_cuda_graphs=test_config.num_cuda_graphs,
                use_cuda_graphs_for_non_decode_steps=test_config.use_cuda_graphs_for_non_decode_steps,
                cuda_graph_all_prefills=test_config.cuda_graph_all_prefills,
                buffer_size_gb=test_config.context_buffer_size_gb,
                paused_buffer_size_gb=test_config.context_paused_buffer_size_gb,
                block_size_tokens=test_config.context_block_size_tokens,
                max_requests=test_config.context_max_requests,
                max_tokens=test_config.context_max_tokens,
                mamba_inference_state_config=mamba_inference_state_config,
                materialize_only_last_token_logits=test_config.materialize_only_last_token_logits,
                kv_cache_management_mode=KVCacheManagementMode(
                    test_config.kv_cache_management_mode
                ),
                static_kv_memory_pointers=test_config.static_kv_memory_pointers,
                enable_chunked_prefill=test_config.enable_chunked_prefill,
                enable_prefix_caching=test_config.enable_prefix_caching,
                prefix_caching_eviction_policy=test_config.prefix_caching_eviction_policy,
                prefix_caching_mamba_gb=(
                    0.02 if mamba_inference_state_config is not None else None
                ),
                use_flashinfer_fused_rope=None,
                unified_memory_level=getattr(test_config, "unified_memory_level", 0),
                track_generated_token_events=test_config.track_generated_token_events,
                num_speculative_tokens=test_config.num_speculative_tokens,
                sampling_backend=test_config.sampling_backend,
                async_sched_mode=test_config.async_sched_mode,
            ),
        )

    @staticmethod
    def _case_config(case, *, enable_prefix_caching):
        feature = case["feature"]
        block_size = 256  # FlashAttention requires paged KV blocks divisible by 256.
        prompt_length = 3 * block_size - 2 if feature == "request-eviction" else 2 * block_size + 5
        num_speculative_tokens = 2 if feature == "mtp" else 0
        kwargs = dict(
            num_requests=0,
            min_prompt_length=prompt_length,
            max_prompt_length=prompt_length,
            num_tokens_to_generate=case.get("num_tokens_to_generate", 4),
            max_sequence_length=prompt_length + 8,
            context_buffer_size_gb=0.02,
            context_block_size_tokens=block_size,
            context_max_requests=8 if case["model"] == "hybrid" else 32,
            context_max_tokens=4096,
            tensor_model_parallel_size=case.get("tp", 1),
            pipeline_model_parallel_size=case.get("pp", 1),
            expert_model_parallel_size=case.get("ep", 1),
            sequence_parallel=case.get("sp", False),
            model_provider=case["model"],
            enable_prefix_caching=enable_prefix_caching,
            num_speculative_tokens=num_speculative_tokens,
            materialize_only_last_token_logits=(feature not in ("mtp", "logprobs")),
            return_log_probs=(feature in ("logprobs", "request-eviction")),
            skip_prompt_log_probs=(feature in ("logprobs", "request-eviction")),
            enable_chunked_prefill=(feature == "chunked"),
        )
        if feature == "chunked":
            kwargs["context_max_tokens"] = block_size + 8
            kwargs["use_cuda_graphs_for_non_decode_steps"] = False
        elif feature == "cuda-graph":
            kwargs.update(
                num_cuda_graphs=2,
                force_build_cuda_graphs=True,
                use_cuda_graphs_for_non_decode_steps=False,
                inference_cuda_graph_scope=InferenceCudaGraphScope.block,
            )
        elif feature in ("offload", "recompute"):
            kwargs.update(
                kv_cache_management_mode=feature,
                static_kv_memory_pointers=False,
                suspend_resume_interval=2,
            )
        elif feature == "request-eviction":
            kwargs["context_paused_buffer_size_gb"] = 0
        elif feature == "uvm":
            # PERSIST does not migrate KV storage during suspend/resume.  This
            # row verifies UVM-backed allocation plus repeated lifecycle state
            # transitions and stable backing storage, not offload/restore.
            kwargs.update(
                kv_cache_management_mode="persist",
                static_kv_memory_pointers=True,
                suspend_resume_interval=2,
            )
        config = DynamicEngineTestConfig(**kwargs)
        if feature == "uvm":
            config.unified_memory_level = 1
        config.prefix_caching_eviction_policy = (
            PrefixCachingEvictionPolicy.LRU
            if case["name"] in PREFIX_CACHE_ENGINE_LRU_CASES
            else PrefixCachingEvictionPolicy.REF_ZERO
        )
        return config

    @staticmethod
    def _make_engine_request(
        ctx,
        request_id,
        prompt,
        *,
        enable_prefix_caching,
        return_log_probs,
        skip_prompt_log_probs=True,
        top_n_logprobs=0,
        num_tokens_to_generate=4,
    ):
        return DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=num_tokens_to_generate,
                termination_id=-1,
                top_k=1,
                return_log_probs=return_log_probs,
                skip_prompt_log_probs=skip_prompt_log_probs,
                top_n_logprobs=top_n_logprobs,
            ),
            block_size_tokens=ctx.block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
        )

    def _assert_cached_prompt_logprobs_rejected(self, case):
        config = self._case_config(case, enable_prefix_caching=True)
        env = self._build_test_env(config)
        env.engine.controller.tokenizer = _NumericTokenizer()
        prompt = torch.arange(
            2 * config.context_block_size_tokens + 5, device=torch.cuda.current_device()
        ) % (config.vocab_size - 1)
        request = self._make_engine_request(
            env.engine.context,
            -1,
            prompt,
            enable_prefix_caching=True,
            return_log_probs=True,
            skip_prompt_log_probs=False,
            top_n_logprobs=5,
        )
        with pytest.raises(ValueError, match=r"^Prompt log probabilities"):
            env.engine._add_request(request)
        assert request.request_id not in env.engine.requests

    @torch.inference_mode()
    def _run_engine_session(self, case, *, enable_prefix_caching):
        config = self._case_config(case, enable_prefix_caching=enable_prefix_caching)
        env = self._build_test_env(config)
        engine = env.engine
        engine.controller.tokenizer = _NumericTokenizer()
        ctx = engine.context
        alloc = ctx.kv_block_allocator
        kv_pool_size = alloc.pool_size
        kv_storage_bytes = ctx.memory_buffer.untyped_storage().nbytes()
        mamba_cache_slots = (
            ctx.mamba_slot_allocator.max_slots if ctx.mamba_slot_allocator is not None else 0
        )
        finished = {}
        expected_epoch = {}
        wave_allocated_blocks = []
        all_wave_hashes = set()
        min_pool_avail = alloc.pool_avail
        saw_chunk = False
        cuda_graph_step_count = 0
        saw_mixed_batch = False
        suspend_count = 0
        uvm_pointer_stability_checks = 0
        step_count = 0
        mamba_commit_calls = 0
        mamba_restore_hits = 0
        tracked_mamba_allocator = None
        instrumented_mamba_allocators = []
        paused_overflow_calls = 0
        checkpointed_records = 0
        checkpoint_configs = []
        epoch_invalidation_count = 0
        epoch_rebuild_count = 0
        prior_ref_zero_blocks = None
        ref_zero_reuse_transitions = 0

        original_evict_overflow = ctx.evict_overflow_paused_requests

        def tracked_evict_overflow(*args, **kwargs):
            nonlocal paused_overflow_calls
            paused_overflow_calls += int(ctx.paused_request_count > 0)
            return original_evict_overflow(*args, **kwargs)

        ctx.evict_overflow_paused_requests = tracked_evict_overflow

        def instrument_mamba_allocator():
            nonlocal tracked_mamba_allocator
            allocator = ctx.mamba_slot_allocator
            if allocator is None or allocator is tracked_mamba_allocator:
                return
            tracked_mamba_allocator = allocator
            original_commit = allocator.commit_intermediate_states
            original_restore = allocator.restore_to_live
            instrumented_mamba_allocators.append((allocator, original_commit, original_restore))

            def tracked_commit():
                nonlocal mamba_commit_calls
                mamba_commit_calls += 1
                return original_commit()

            def tracked_restore(request_idx, block_id):
                nonlocal mamba_restore_hits
                restored = original_restore(request_idx, block_id)
                mamba_restore_hits += int(restored)
                return restored

            allocator.commit_intermediate_states = tracked_commit
            allocator.restore_to_live = tracked_restore

        instrument_mamba_allocator()
        if case["feature"] == "uvm":
            if ctx.unified_memory_level != 1 or not hasattr(ctx, "unified_memory_mempool"):
                pytest.fail(
                    "UVM-backed prefix-cache coverage is required on DFW, but the context "
                    "fell back to GPU memory because the UVM allocator was unavailable"
                )
            uvm_memory_buffer_ptr = ctx.memory_buffer.data_ptr()

        for cycle in range(3):
            block_size = ctx.block_size_tokens
            prompt_length = (
                3 * block_size - 2 if case["feature"] == "request-eviction" else 2 * block_size + 5
            )
            prompt_offset = 0 if case["feature"] == "epoch" else cycle * 17
            base = (
                torch.arange(prompt_length, device=torch.cuda.current_device()) + prompt_offset
            ) % (config.vocab_size - 1)
            pressure = (base + 37) % (config.vocab_size - 1)

            if case["feature"] == "epoch":
                if cycle > 0:
                    if enable_prefix_caching:
                        challenge_hashes = set(compute_block_hashes_batched(base, block_size))
                        assert challenge_hashes <= alloc.kv_hash_to_block_id.keys()
                engine._set_generation_epoch(cycle)
                if cycle > 0:
                    if enable_prefix_caching:
                        assert challenge_hashes.isdisjoint(alloc.kv_hash_to_block_id)
                        epoch_invalidation_count += 1

            # Cache-on needs seven blocks for producer + follower tail + pressure.
            # Cache-off needs nine unless chunked staging limits live demand to seven.
            target_allocatable = 7 if enable_prefix_caching or case["feature"] == "chunked" else 9
            allocatable = alloc.get_allocatable_count()
            if allocatable > target_allocatable:
                filler = alloc.allocate_memory_blocks(allocatable - target_allocatable)
                assert filler is not None
            assert alloc.get_allocatable_count() == target_allocatable

            wave_ids = {3 * cycle, 3 * cycle + 1, 3 * cycle + 2}
            requests = []
            for request_id, prompt in (
                (3 * cycle, base),
                (3 * cycle + 1, base.clone()),
                (3 * cycle + 2, pressure),
            ):
                request = self._make_engine_request(
                    ctx,
                    request_id,
                    prompt,
                    enable_prefix_caching=enable_prefix_caching,
                    return_log_probs=(case["feature"] in ("logprobs", "request-eviction")),
                    skip_prompt_log_probs=True,
                    top_n_logprobs=5 if case["feature"] == "logprobs" else 0,
                    num_tokens_to_generate=case.get("num_tokens_to_generate", 4),
                )
                requests.append(request)
                if case["feature"] == "epoch":
                    expected_epoch[request_id] = cycle

            if enable_prefix_caching:
                for request in requests:
                    engine._add_request(request)
            else:
                # Cache-on's hash coordination defers the duplicate follower
                # during the first step. Mirror that execution order explicitly
                # in the cache-off baseline so BF16 comparisons use equal batches.
                engine._add_request(requests[0])
                if case["feature"] != "chunked":
                    engine._add_request(requests[2])
            baseline_follower_pending = not enable_prefix_caching

            if enable_prefix_caching:
                for request in requests:
                    all_wave_hashes.update(request.precomputed_block_hashes)

            hits_before = engine._prefix_cache_hits
            blocks_this_wave = set()
            while wave_ids - finished.keys():
                result = engine.step_modern()
                step_count += 1
                if baseline_follower_pending and (
                    case["feature"] != "chunked" or ctx.chunked_prefill_request_id == -1
                ):
                    engine._add_request(requests[1])
                    if case["feature"] == "chunked":
                        engine._add_request(requests[2])
                    baseline_follower_pending = False
                min_pool_avail = min(min_pool_avail, alloc.pool_avail)
                active_blocks = ctx.request_to_kv_block_ids[: ctx.total_request_count]
                blocks_this_wave.update(active_blocks[active_blocks >= 0].tolist())
                saw_chunk |= ctx.chunked_prefill_request_id != -1
                cuda_graph_step_count += int(ctx.using_cuda_graph_this_step())
                saw_mixed_batch |= (
                    ctx.batch_dimensions.prefill_req_count > 0
                    and ctx.batch_dimensions.decode_req_count > 0
                )
                for record in result["finished_request_records"]:
                    if len(record.requests) > 1:
                        checkpointed_records += 1
                        for request in record.requests[1:]:
                            checkpoint_configs.append(
                                (request.enable_prefix_caching, request.block_size_tokens)
                            )
                            expected_hashes = (
                                compute_block_hashes_batched(
                                    request.prompt_tokens, request.block_size_tokens
                                )
                                if request.enable_prefix_caching
                                else []
                            )
                            assert request.precomputed_block_hashes == expected_hashes
                    merged = record.merge()
                    finished[merged.request_id] = merged

                if (
                    config.suspend_resume_interval is not None
                    and engine.has_unfinished_requests()
                    and step_count % config.suspend_resume_interval == 0
                ):
                    engine.suspend()
                    assert not ctx.is_tensor_state_allocated
                    engine.resume()
                    assert ctx.is_tensor_state_allocated
                    if case["feature"] == "uvm":
                        assert ctx.memory_buffer.data_ptr() == uvm_memory_buffer_ptr
                        uvm_pointer_stability_checks += 1
                    alloc = ctx.kv_block_allocator
                    instrument_mamba_allocator()
                    suspend_count += 1
                assert step_count < 512, f"{case['name']} did not converge"

            torch.cuda.synchronize()
            used_blocks = alloc.get_total_used()
            assert alloc.pool_size == kv_pool_size
            assert ctx.memory_buffer.untyped_storage().nbytes() == kv_storage_bytes
            assert (
                ctx.mamba_slot_allocator.max_slots if ctx.mamba_slot_allocator is not None else 0
            ) == mamba_cache_slots
            assert alloc.pool_avail + used_blocks == alloc.pool_size - 1
            assert 0 <= used_blocks <= alloc.pool_size - 1
            wave_allocated_blocks.append(used_blocks)
            if enable_prefix_caching:
                assert engine._prefix_cache_hits > hits_before
                if config.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
                    assert alloc.kv_hash_to_block_id == {}
                    assert all(alloc.block_hashes[bid].item() == -1 for bid in blocks_this_wave)
                    assert all(alloc.block_ref_counts[bid].item() == 0 for bid in blocks_this_wave)
                    free_ids = set(alloc.block_bag[: alloc.pool_avail].tolist())
                    assert blocks_this_wave <= free_ids
                    if prior_ref_zero_blocks is not None:
                        assert len(blocks_this_wave & prior_ref_zero_blocks) >= 2
                        ref_zero_reuse_transitions += 1
                    prior_ref_zero_blocks = blocks_this_wave
                elif case["feature"] == "epoch":
                    challenge_hashes = set(compute_block_hashes_batched(base, block_size))
                    assert challenge_hashes <= alloc.kv_hash_to_block_id.keys()
                    assert requests[0].num_cached_tokens == 0
                    assert requests[1].num_cached_tokens >= 2 * block_size
                    epoch_rebuild_count += 1

        assert len(finished) == 9
        assert len(wave_allocated_blocks) == 3
        assert max(wave_allocated_blocks) <= alloc.pool_size - 1
        if enable_prefix_caching:
            assert min_pool_avail == 0
            assert engine._prefix_cache_hits >= 3
            assert engine._prefix_cache_blocks_matched >= 6
            assert engine._prefix_coordination_waits >= 3
            if config.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                if case["feature"] == "epoch":
                    assert epoch_invalidation_count == 2
                else:
                    assert all_wave_hashes - alloc.kv_hash_to_block_id.keys()
            else:
                assert ref_zero_reuse_transitions == 2
        else:
            assert min_pool_avail <= 1
            assert engine._prefix_cache_hits == 0

        stats = dict(
            config=config,
            saw_chunk=saw_chunk,
            cuda_graph_step_count=cuda_graph_step_count,
            saw_mixed_batch=saw_mixed_batch,
            suspend_count=suspend_count,
            uvm_pointer_stability_checks=uvm_pointer_stability_checks,
            mamba_commit_calls=mamba_commit_calls,
            mamba_restore_hits=mamba_restore_hits,
            paused_overflow_calls=paused_overflow_calls,
            checkpointed_records=checkpointed_records,
            checkpoint_configs=checkpoint_configs,
            evicted_request_count=engine.evicted_request_count,
            expected_epoch=expected_epoch,
            epoch_invalidation_count=epoch_invalidation_count,
            epoch_rebuild_count=epoch_rebuild_count,
            ref_zero_reuse_transitions=ref_zero_reuse_transitions,
            mtp_tokens_proposed=int(engine._spec_tokens_proposed_per_pos.sum()),
            mtp_num_layers=engine.controller.inference_wrapped_model.model.config.mtp_num_layers,
            num_moe_experts=engine.controller.inference_wrapped_model.model.config.num_moe_experts,
        )
        ctx.evict_overflow_paused_requests = original_evict_overflow
        for allocator, original_commit, original_restore in instrumented_mamba_allocators:
            allocator.commit_intermediate_states = original_commit
            allocator.restore_to_live = original_restore
        return finished, stats

    @staticmethod
    def _clear_engine_runtime():
        torch.cuda.synchronize()
        delete_cuda_graphs()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @staticmethod
    def _assert_top_n_parity(cached_values, baseline_values, expected_length, atol):
        assert cached_values is not None and baseline_values is not None
        assert len(cached_values) == len(baseline_values) == expected_length
        for cached_top_n, baseline_top_n in zip(cached_values, baseline_values):
            assert isinstance(cached_top_n, dict) and isinstance(baseline_top_n, dict)
            assert len(cached_top_n) == len(baseline_top_n) == 5
            assert tuple(cached_top_n) == tuple(baseline_top_n)
            assert np.allclose(
                list(cached_top_n.values()), list(baseline_top_n.values()), atol=atol, rtol=0
            )

    @torch.inference_mode()
    def _run_engine_case(self, case):
        self._clear_engine_runtime()
        try:
            if case["feature"] == "logprobs":
                self._assert_cached_prompt_logprobs_rejected(case)
                self._clear_engine_runtime()
            cached, stats = self._run_engine_session(case, enable_prefix_caching=True)
            self._clear_engine_runtime()
            baseline, baseline_stats = self._run_engine_session(case, enable_prefix_caching=False)

            tokenizer = _NumericTokenizer()
            num_tokens_to_generate = case.get("num_tokens_to_generate", 4)
            for request_id in range(9):
                cached_request = cached[request_id]
                baseline_request = baseline[request_id]
                assert cached_request.generated_tokens == baseline_request.generated_tokens
                assert cached_request.generated_text == baseline_request.generated_text
                assert cached_request.generated_text == tokenizer.detokenize(
                    cached_request.generated_tokens, skip_special_tokens=True
                )
                assert len(cached_request.generated_tokens) == num_tokens_to_generate

            activations = {
                "cache-off-on-output-parity",
                "real-engine-output-parity",
                "three-engine-sharing-waves",
                "forced-kv-pool-pressure",
                "kv-pool-capacity-bounded-under-pressure",
                "prefix-scheduling-deferrals-observed",
            }
            config = stats["config"]
            policy = config.prefix_caching_eviction_policy
            if policy == PrefixCachingEvictionPolicy.LRU:
                activations.add("lru-retained-prefix-evicted")
            else:
                assert stats["ref_zero_reuse_transitions"] == 2
                activations.add("ref-zero-physical-block-reuse")

            if case["model"] == "hybrid":
                assert stats["mamba_commit_calls"] > 0
                assert stats["mamba_restore_hits"] >= 3
                activations.add("hybrid-kv-mamba-reuse")
            feature = case["feature"]
            if feature == "tp":
                assert config.tensor_model_parallel_size > 1
                activations.add("tensor-parallel-prefix-hit")
            elif feature == "pp":
                assert config.pipeline_model_parallel_size > 1
                activations.add("pipeline-parallel-prefix-hit")
            elif feature == "mixed-parallel":
                assert config.sequence_parallel
                assert config.tensor_model_parallel_size > 1
                assert config.pipeline_model_parallel_size > 1
                activations.add("mixed-parallel-prefix-hit")
            elif feature == "moe":
                assert config.expert_model_parallel_size > 1
                assert stats["num_moe_experts"] == config.expert_model_parallel_size
                activations.add("moe-expert-parallel-forward-with-prefix-hits")
            elif feature == "chunked":
                assert stats["saw_chunk"]
                activations.add("chunked-prefix-reuse")
            elif feature == "cuda-graph":
                assert config.num_cuda_graphs == 2
                assert config.inference_cuda_graph_scope == InferenceCudaGraphScope.block
                assert stats["cuda_graph_step_count"] > 0
                activations.add("cuda-graph-replay-with-prefix-hits")
            elif feature == "mtp":
                assert stats["mtp_num_layers"] == config.num_speculative_tokens == 2
                assert stats["mtp_tokens_proposed"] > 0
                activations.add("mtp-speculative-proposals-with-prefix-hits")
            elif feature in ("logprobs", "request-eviction"):
                # Prefix restore changes hybrid BF16 batch shapes; the observed drift is < 0.0041.
                logprob_atol = 5e-3 if case["model"] == "hybrid" else 1e-3
                for request_id in range(9):
                    cached_request = cached[request_id]
                    baseline_request = baseline[request_id]
                    assert cached_request.generated_log_probs is not None
                    assert baseline_request.generated_log_probs is not None
                    assert len(cached_request.generated_log_probs) == num_tokens_to_generate
                    assert len(baseline_request.generated_log_probs) == num_tokens_to_generate
                    assert np.allclose(
                        cached_request.generated_log_probs,
                        baseline_request.generated_log_probs,
                        atol=logprob_atol if feature == "logprobs" else 1e-5,
                        rtol=0,
                    )
                    if feature == "logprobs":
                        self._assert_top_n_parity(
                            cached_request.generated_top_n_logprobs,
                            baseline_request.generated_top_n_logprobs,
                            num_tokens_to_generate,
                            logprob_atol,
                        )
                if feature == "logprobs":
                    activations.add("logprob-parity")
                else:
                    assert stats["paused_overflow_calls"] > 0
                    assert baseline_stats["paused_overflow_calls"] > 0
                    assert stats["evicted_request_count"] > 0
                    assert baseline_stats["evicted_request_count"] > 0
                    assert stats["checkpointed_records"] > 0
                    assert baseline_stats["checkpointed_records"] > 0
                    assert all(
                        enabled and block_size == stats["config"].context_block_size_tokens
                        for enabled, block_size in stats["checkpoint_configs"]
                    )
                    assert all(not enabled for enabled, _ in baseline_stats["checkpoint_configs"])
                    activations.add("request-eviction-checkpoint-resume")
            elif feature in ("offload", "recompute"):
                assert stats["suspend_count"] >= 1
                activations.add(f"{feature}-prefix-resume")
            elif feature == "uvm":
                assert stats["config"].unified_memory_level == 1
                assert stats["suspend_count"] >= 3
                assert stats["uvm_pointer_stability_checks"] == stats["suspend_count"]
                activations.add("uvm-backed-prefix-lifecycle")
            elif feature == "epoch":
                assert stats["epoch_invalidation_count"] == 2
                assert stats["epoch_rebuild_count"] == 3
                for request_id, request in cached.items():
                    assert request.kv_cache_epoch == [(0, stats["expected_epoch"][request_id])]
                activations.add("epoch-signal-invalidation-rebuild")

            assert stats["saw_mixed_batch"]
            return activations
        finally:
            self._clear_engine_runtime()

    @pytest.mark.internal
    @pytest.mark.parametrize("case", PREFIX_CACHE_ENGINE_CASES)
    def test_real_engine_pair_matrix(self, case):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=case.get("tp", 1),
            pipeline_model_parallel_size=case.get("pp", 1),
            expert_model_parallel_size=case.get("ep", 1),
            expert_tensor_parallel_size=1,
        )
        try:
            activations = self._run_engine_case(case)
            matrix_pairs = []
            for parameter in PREFIX_CACHE_ENGINE_CASES:
                matrix_case = parameter.values[0]
                matrix_policy = (
                    "lru" if matrix_case["name"] in PREFIX_CACHE_ENGINE_LRU_CASES else "ref-zero"
                )
                matrix_pairs.append(f"{matrix_case['name']}×{matrix_policy}")
            assert len(matrix_pairs) == len(set(matrix_pairs))
            assert set(matrix_pairs) == PREFIX_CACHE_ENGINE_PAIR_OWNERS.keys()

            policy = (
                PrefixCachingEvictionPolicy.LRU
                if case["name"] in PREFIX_CACHE_ENGINE_LRU_CASES
                else PrefixCachingEvictionPolicy.REF_ZERO
            )
            policy_name = "lru" if policy == PrefixCachingEvictionPolicy.LRU else "ref-zero"
            owner = PREFIX_CACHE_ENGINE_PAIR_OWNERS[f"{case['name']}×{policy_name}"]
            assert owner in activations
            assert "three-engine-sharing-waves" in activations
            expected_policy_activation = (
                "lru-retained-prefix-evicted"
                if policy == PrefixCachingEvictionPolicy.LRU
                else "ref-zero-physical-block-reuse"
            )
            assert expected_policy_activation in activations
        finally:
            delete_cuda_graphs()
            DynamicInferenceContext.ROUNDER = 64
            DynamicInferenceContext.TOKEN_ROUNDER = 64
            DynamicInferenceContext.REQUEST_ROUNDER = 64
            Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.asyncio
    async def test_real_http_zmq_prefix_pressure(self):
        with torch.inference_mode():
            await self._run_real_http_zmq_prefix_pressure()

    async def _run_real_http_zmq_prefix_pressure(self):
        """Drive real HTTP, coordinator, client, and model-engine processes through cache churn."""
        if not HAVE_ZMQ or not HAS_BACKEND:
            pytest.skip("pyzmq, Quart, and Hypercorn are required")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        rank = torch.distributed.get_rank()
        sync_context = zmq.Context()
        sync = AsyncZMQCommunicator(sync_context, process_group=None, hostname="127.0.0.1")
        engine = None
        control_client = None
        control_ready = False
        server_socket = None
        server_started = False
        try:
            self._clear_engine_runtime()
            case = dict(name="gpt-http-zmq", model="gpt", feature="base")
            baseline_config = self._case_config(case, enable_prefix_caching=False)
            baseline_env = self._build_test_env(baseline_config)
            baseline_env.engine.controller.tokenizer = _NumericTokenizer()
            prompt = [
                (index % 89) + 1
                for index in range(2 * baseline_config.context_block_size_tokens + 5)
            ]
            baseline_request = self._make_engine_request(
                baseline_env.engine.context,
                0,
                torch.tensor(prompt, device=torch.cuda.current_device()),
                enable_prefix_caching=False,
                return_log_probs=True,
            )
            baseline_env.engine._add_request(baseline_request)
            baseline_output = None
            while baseline_output is None:
                result = await baseline_env.engine.async_step()
                if result["finished_request_records"]:
                    baseline_output = result["finished_request_records"][0].merge()
            baseline_tokens = list(baseline_output.generated_tokens)
            baseline_log_probs = list(baseline_output.generated_log_probs)
            del baseline_output, baseline_request, baseline_env
            self._clear_engine_runtime()

            config = self._case_config(case, enable_prefix_caching=True)
            assert config.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU
            env = self._build_test_env(config)
            engine = env.engine
            tokenizer = _NumericTokenizer()
            engine.controller.tokenizer = tokenizer

            allocator = engine.context.kv_block_allocator
            filler = allocator.allocate_memory_blocks(allocator.get_allocatable_count() - 7)
            assert filler is not None
            assert allocator.get_allocatable_count() == 7

            coordinator_addr = await engine.start_listening_to_data_parallel_coordinator(
                launch_inference_coordinator=True, hostname="127.0.0.1"
            )
            control_error = None
            if rank == 0:
                try:
                    control_client = InferenceClient(coordinator_addr)
                    control_client.start()
                    control_ready = True
                except Exception as error:
                    control_error = error
            control_status = torch.tensor(
                int(control_ready), dtype=torch.int64, device=torch.cuda.current_device()
            )
            torch.distributed.broadcast(control_status, src=0)
            control_ready = bool(control_status.item())
            if control_error is not None:
                raise control_error
            if not control_ready:
                raise RuntimeError("rank 0 could not start the coordinator control client")
            await asyncio.wait_for(sync.all_reduce_max(1), timeout=30)

            prompt_hashes = compute_block_hashes_batched(
                torch.tensor(prompt), config.context_block_size_tokens
            )
            reference_choices = []
            if rank == 0:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind(("127.0.0.1", 0))
                server_port = server_socket.getsockname()[1]
                start_text_gen_server(
                    coordinator_addr,
                    tokenizer,
                    rank,
                    server_port,
                    num_replicas=1,
                    hostname="127.0.0.1",
                    sock=server_socket,
                )
                server_started = True
                base_url = f"http://127.0.0.1:{server_port}"
                deadline = time.monotonic() + 30
                while True:
                    try:
                        assert (await asyncio.to_thread(_http_json, f"{base_url}/v1/health"))[
                            "ready"
                        ]
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise
                        await asyncio.sleep(0.1)

                request = {
                    "prompt": prompt,
                    "max_tokens": 4,
                    "temperature": 0,
                    "logprobs": 5,
                    "ignore_eos": True,
                }
                for _ in range(2):
                    response = await asyncio.to_thread(
                        _http_json, f"{base_url}/v1/completions", request
                    )
                    assert response["usage"]["prompt_tokens"] == len(prompt)
                    reference_choices.append(response["choices"][0])

            await asyncio.wait_for(sync.all_reduce_max(1), timeout=90)
            cached_before_epoch = torch.tensor(
                int(
                    any(block_hash in allocator.kv_hash_to_block_id for block_hash in prompt_hashes)
                ),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(cached_before_epoch)
            assert cached_before_epoch.item() >= 1

            if rank == 0:
                control_client.set_generation_epoch(7)
            deadline = time.monotonic() + 30
            while engine._generation_epoch != 7:
                if time.monotonic() >= deadline:
                    raise TimeoutError("SET_GENERATION_EPOCH did not reach every engine")
                await asyncio.sleep(0.02)
            assert all(
                block_hash not in allocator.kv_hash_to_block_id for block_hash in prompt_hashes
            )
            await asyncio.wait_for(sync.all_reduce_max(1), timeout=90)

            if rank == 0:
                for _ in range(2):
                    response = await asyncio.to_thread(
                        _http_json, f"{base_url}/v1/completions", request
                    )
                    choice = response["choices"][0]
                    assert choice["kv_cache_epoch"] == [[0, 7]]
                    reference_choices.append(choice)

            await asyncio.wait_for(sync.all_reduce_max(1), timeout=90)
            rebuilt_after_epoch = torch.tensor(
                int(
                    any(block_hash in allocator.kv_hash_to_block_id for block_hash in prompt_hashes)
                ),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(rebuilt_after_epoch)
            assert rebuilt_after_epoch.item() >= 1

            if rank == 0:
                pressure_prompts = [
                    [((index + 3 * group) % 89) + 1 for index in range(len(prompt))]
                    for group in range(1, 33)
                ]
                pressure = dict(request, prompt=pressure_prompts, logprobs=None)
                response = await asyncio.to_thread(
                    _http_json, f"{base_url}/v1/completions", pressure
                )
                assert len(response["choices"]) == len(pressure_prompts)

            await asyncio.wait_for(sync.all_reduce_max(1), timeout=90)
            pressure_stats = torch.tensor(
                [
                    engine._prefix_cache_hits,
                    engine._prefix_cache_blocks_matched,
                    int(
                        any(
                            block_hash in allocator.kv_hash_to_block_id
                            for block_hash in prompt_hashes
                        )
                    ),
                ],
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(pressure_stats)
            assert pressure_stats[0].item() >= 1
            assert pressure_stats[1].item() >= len(prompt_hashes)
            assert pressure_stats[2].item() == 0, "pressure did not evict the original prefix"

            if rank == 0:
                for _ in range(2):
                    response = await asyncio.to_thread(
                        _http_json, f"{base_url}/v1/completions", request
                    )
                    reference_choices.append(response["choices"][0])
                expected = reference_choices[0]
                assert expected["generation_token_ids"] == baseline_tokens
                assert np.allclose(
                    expected["generation_log_probs"], baseline_log_probs, atol=1e-5, rtol=0
                )
                for choice in reference_choices[1:]:
                    assert choice["generation_token_ids"] == expected["generation_token_ids"]
                    assert np.allclose(
                        choice["generation_log_probs"],
                        expected["generation_log_probs"],
                        atol=1e-5,
                        rtol=0,
                    )

            await asyncio.wait_for(sync.all_reduce_max(1), timeout=90)
            final_hits = torch.tensor(
                engine._prefix_cache_hits, dtype=torch.int64, device=torch.cuda.current_device()
            )
            torch.distributed.all_reduce(final_hits)
            assert final_hits.item() >= pressure_stats[0].item() + 1
        finally:
            if rank == 0 and server_started:
                try:
                    stop_text_gen_server()
                except Exception:
                    pass
            if server_socket is not None and server_socket.fileno() != -1:
                server_socket.close()
            task = getattr(engine, "engine_loop_task", None) if engine is not None else None
            if task is not None and not task.done():
                graceful_stop = control_ready
                if rank == 0 and control_client is not None:
                    try:
                        control_client.pause_engines()
                    except Exception:
                        graceful_stop = False
                if graceful_stop:
                    try:
                        if engine.state not in (EngineState.PAUSED, EngineState.STOPPED):
                            await asyncio.wait_for(
                                engine.wait_until(EngineState.PAUSED), timeout=15
                            )
                        if rank == 0 and control_client is not None and not task.done():
                            control_client.stop_engines()
                        await asyncio.wait_for(asyncio.shield(task), timeout=15)
                    except Exception:
                        graceful_stop = False
                if not graceful_stop and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            if rank == 0 and control_client is not None:
                try:
                    control_client.shutdown_coordinator()
                except Exception:
                    pass
            if (
                rank == 0
                and engine is not None
                and hasattr(engine, "inference_coordinator_process")
            ):
                process = engine.inference_coordinator_process
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            if rank == 0 and control_client is not None:
                try:
                    control_client.stop()
                except Exception:
                    pass
            try:
                sync.close()
            finally:
                sync_context.term()
            self._clear_engine_runtime()
            DynamicInferenceContext.ROUNDER = 64
            DynamicInferenceContext.TOKEN_ROUNDER = 64
            DynamicInferenceContext.REQUEST_ROUNDER = 64
            Utils.destroy_model_parallel()
