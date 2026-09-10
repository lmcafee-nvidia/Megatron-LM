# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import copy

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model, until


async def _submit_paused(h, prompt, params):
    before = set(h.service.coordinator.request_id_to_rank)
    future = h.clients[0].add_request(prompt, copy.deepcopy(params))
    await until(lambda: len(h.service.coordinator.request_id_to_rank) == len(before) + 1)
    return (set(h.service.coordinator.request_id_to_rank) - before).pop(), future


@pytest.mark.internal
@pytest.mark.asyncio
async def test_routed_chunk_partition_beside_decode(monkeypatch):
    prompt, params = list(range(4, 69)), greedy_params(num_tokens_to_generate=16)
    async with routed_model(monkeypatch, enable_chunked_prefill=True, context_max_tokens=32) as h:
        assert h.dp_size == 2
        direct = await h.direct(prompt, params)
        await h.start()
        parts = []
        model = h.engine.controller.inference_wrapped_model.model
        forward = model.forward

        def observe(*args, **kwargs):
            c = h.engine.context
            rows = slice(c.paused_request_count, c.total_request_count)
            tensors = (c.request_ids, c.request_query_lengths, c.request_in_prefill_status_tensor)
            snapshot = list(zip(*(tensor[rows].tolist() for tensor in tensors)))
            result = forward(*args, **kwargs)
            parts.extend(
                (rid, count, c.num_decode_requests) for rid, count, prefill in snapshot if prefill
            )
            return result

        monkeypatch.setattr(model, "forward", observe)
        await h.pause()
        target = [None]
        if h.rank == 0:
            first, a = await _submit_paused(h, [4, 5, 6, 7], params)
            _, b = await _submit_paused(h, [8, 9, 10, 11], params)
            target[0], future = await _submit_paused(h, prompt, params)
            routes = h.service.coordinator.request_id_to_rank
            assert routes[first] == routes[target[0]]
        torch.distributed.broadcast_object_list(target, src=0)
        await h.unpause()
        if h.rank == 0:
            _, _, final = await asyncio.wait_for(asyncio.gather(a, b, future), timeout=60)
            assert final["generated_tokens"] == direct["generated_tokens"]
        await h.barrier()
        chunks = [(count, decode) for rid, count, decode in parts if rid == target[0]]
        if chunks:
            assert len(chunks) >= 3 and sum(count for count, _ in chunks) == len(prompt)
            assert all(0 < count <= 32 for count, _ in chunks)
            assert any(decode > 0 for _, decode in chunks)
        assert await h.sync.all_reduce_max(int(bool(chunks))) == 1
        h.assert_retired()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["affinity", "load", "expired"])
async def test_routed_affinity_competes_with_load_and_expiry(monkeypatch, choice):
    prompt, params = list(range(4, 37)), greedy_params()
    options = dict(
        prefix_caching_routing_alpha=100 if choice == "load" else 0,
        prefix_cache_ttl_seconds=0 if choice == "expired" else 300,
    )
    async with routed_model(
        monkeypatch,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
        coordinator_options=options,
    ) as h:
        assert h.dp_size == 2
        direct = await h.direct(prompt, params)
        await h.start()
        await h.pause()
        if h.rank == 0:
            _, filler = await _submit_paused(h, [91, 92, 93, 94], params)
            warm_id, warm = await _submit_paused(h, prompt, params)
            warm_owner = h.service.coordinator.request_id_to_rank[warm_id]
        await h.unpause()
        if h.rank == 0:
            await asyncio.wait_for(asyncio.gather(filler, warm), timeout=60)
            if choice == "expired":
                # Unrelated hash-bearing traffic triggers the production TTL sweep.
                await asyncio.wait_for(h.clients[0].add_request(list(range(60, 77)), params), 60)
        await h.barrier()
        await h.pause()
        target = [None]
        if h.rank == 0:
            pending = []
            if choice != "expired":
                _, blocker = await _submit_paused(h, prompt, params)
                pending.append(blocker)
            target[0], future = await _submit_paused(h, prompt, params)
            owner = h.service.coordinator.request_id_to_rank[target[0]]
            assert (owner == warm_owner) == (choice == "affinity")
        torch.distributed.broadcast_object_list(target, src=0)
        await h.unpause()
        if h.rank == 0:
            outputs = await asyncio.wait_for(asyncio.gather(*pending, future), timeout=60)
            assert outputs[-1]["generated_tokens"] == direct["generated_tokens"]
        await h.barrier()
        steps = [s for s in h.witnesses if target[0] in s["ids"]]
        if steps:
            assert (max(s["cached"][target[0]] for s in steps) >= 16) == (choice == "affinity")
        assert await h.sync.all_reduce_max(int(bool(steps))) == 1
        h.assert_retired()
