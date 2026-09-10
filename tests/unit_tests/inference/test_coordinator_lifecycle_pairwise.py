# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Routed control transitions with live request ownership and actual KV state."""

import asyncio
import copy

import pytest

from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.headers import Headers
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model, until


async def _pause_active(h, monkeypatch, prompt, params):
    """Hold exactly after a real step until the coordinator has broadcast PAUSE."""
    arrived, release = asyncio.Event(), asyncio.Event()
    step = h.engine.async_step

    async def gated_step():
        result = await step()
        if h.rank == 0 and not arrived.is_set() and 0 in h.engine.requests:
            if len(h.engine.get_request(0).generated_tokens) >= 2:
                arrived.set()
                await asyncio.wait_for(release.wait(), timeout=30)
        return result

    monkeypatch.setattr(h.engine, "async_step", gated_step)
    future = None
    if h.rank == 0:
        future = h.clients[0].add_request(prompt, copy.deepcopy(params))
        await asyncio.wait_for(arrived.wait(), timeout=30)
        h.clients[0].pause_engines()
        await until(lambda: any(e["header"] == Headers.PAUSE for e in h.service.events))
        release.set()
    await asyncio.wait_for(h.engine.wait_until(EngineState.PAUSED), timeout=60)
    await h.barrier()
    if h.rank == 0:
        request = h.engine.get_request(0)
        assert 0 < len(request.generated_tokens) < params.num_tokens_to_generate
        assert h.witnesses and not future.done()
    return future


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(AsyncScheduleMode))
@pytest.mark.parametrize("residency", list(KVCacheManagementMode))
async def test_routed_suspend_resumes_active_request(monkeypatch, mode, residency):
    """All three residency transitions preserve a partly generated routed result."""
    prompt, params = list(range(4, 20)), greedy_params(num_tokens_to_generate=16)
    async with routed_model(
        monkeypatch,
        async_sched_mode=mode,
        kv_cache_management_mode=residency.value,
        static_kv_memory_pointers=False,
    ) as h:
        direct = await h.direct(prompt, params)
        await h.start()
        future = await _pause_active(h, monkeypatch, prompt, params)
        pointer = h.engine.context.memory_buffer.data_ptr()
        if h.rank == 0:
            before = list(h.engine.get_request(0).generated_tokens)
            route = dict(h.service.coordinator.request_id_to_rank)
            h.clients[0].suspend_engines()
        await asyncio.wait_for(h.engine.wait_until(EngineState.SUSPENDED), timeout=60)
        context = h.engine.context
        assert not context.is_tensor_state_allocated
        if residency == KVCacheManagementMode.PERSIST:
            assert context.memory_buffer.data_ptr() == pointer
        elif residency == KVCacheManagementMode.OFFLOAD:
            assert context.memory_buffer.untyped_storage().nbytes() == 0
            assert context._offloadable_cpu_backups["memory_buffer"].device.type == "cpu"
        else:
            assert not hasattr(context, "memory_buffer")
            assert not h.engine.controller._async_sched_logits.is_valid
        await h.barrier()
        if h.rank == 0:
            assert h.service.coordinator.request_id_to_rank == route
            assert not future.done()
            h.clients[0].resume_engines()
        await asyncio.wait_for(h.engine.wait_until(EngineState.RESUMED), timeout=60)
        assert h.engine.context.is_tensor_state_allocated
        assert h.engine.state == EngineState.PAUSED
        assert h.engine._weight_epoch == 1
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            final = await asyncio.wait_for(future, timeout=60)
            assert final["generated_tokens"][: len(before)] == before
            assert final["generated_tokens"] == direct["generated_tokens"]
            assert final["status"] == direct["status"]
            if residency == KVCacheManagementMode.RECOMPUTE:
                assert sum(bool(s["prefill"]) for s in h.witnesses) >= 2
        await h.barrier()
        h.assert_retired()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("active", [False, True], ids=["queued", "active"])
async def test_routed_abort_retires_only_cancelled_request(monkeypatch, active):
    """Abort names the client-local ID; another client's ID zero remains valid."""
    prompt, params = list(range(4, 20)), greedy_params(num_tokens_to_generate=16)
    async with routed_model(monkeypatch) as h:
        direct = await h.direct(prompt, params)
        await h.start()
        if active:
            cancelled = await _pause_active(h, monkeypatch, prompt, params)
        else:
            await h.pause()
            if h.rank == 0:
                cancelled = h.clients[0].add_request(prompt, copy.deepcopy(params))
                await until(lambda: 0 in h.engine.requests)
                assert not h.witnesses
        survivor = None
        if h.rank == 0:
            survivor_id, survivor = h.clients[1].add_request_with_id(prompt, copy.deepcopy(params))
            assert survivor_id == 0
            await until(lambda: len(h.service.coordinator.request_id_to_rank) == 2)
            h.clients[0].abort_request(0)
            assert cancelled.cancelled()
            await until(lambda: any(e["header"] == Headers.ABORT_REQUEST for e in h.service.events))
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            final = await asyncio.wait_for(survivor, timeout=60)
            assert final["generated_tokens"] == direct["generated_tokens"]
            await until(lambda: not h.service.coordinator.request_id_to_rank)
            await until(lambda: not h.clients[0].aborted_request_ids)
        await h.barrier()
        h.assert_retired()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_routed_epoch_change_reaches_all_engines(monkeypatch, synchronous):
    """A paused generation update reaches idle peers and the active request."""
    prompt, params = list(range(4, 20)), greedy_params(num_tokens_to_generate=16)
    async with routed_model(monkeypatch) as h:
        # This field selects the actual world barrier execution, not a mocked gate.
        h.engine.use_synchronous_zmq_collectives = synchronous
        direct = await h.direct(prompt, params)
        await h.start()
        future = await _pause_active(h, monkeypatch, prompt, params)
        if h.rank == 0:
            h.clients[0].set_generation_epoch(7)
        await until(lambda: h.engine._generation_epoch == 7)
        if h.rank == 0:
            request = h.engine.get_request(0)
            assert request.policy_epoch == [(0, 7)]
            assert request.kv_cache_epoch == [(0, 7)]
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            final = await asyncio.wait_for(future, timeout=60)
            assert final["generated_tokens"] == direct["generated_tokens"]
            assert final["policy_epoch"] == [[0, 7]]
            assert final["kv_cache_epoch"] == [[0, 7]]
        await h.barrier()
        h.assert_retired()
