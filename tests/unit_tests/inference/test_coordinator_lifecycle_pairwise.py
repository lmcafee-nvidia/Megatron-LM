# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import asyncio
import copy
from unittest import mock

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.contexts import dynamic_context as dynamic_context_module
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.headers import Headers
from megatron.core.transformer.cuda_graphs import _CudaGraphRunner
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model, until


async def _pause_active(h, monkeypatch, prompt, params):
    arrived, release = asyncio.Event(), asyncio.Event()
    step = h.engine.async_step

    async def gated_step():
        if arrived.is_set():
            return None
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
    monkeypatch.setattr(h.engine, "async_step", step)
    await h.barrier()
    if h.rank == 0:
        assert 0 < len(h.engine.get_request(0).generated_tokens) < params.num_tokens_to_generate
        assert h.witnesses and not future.done()
    return future


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(AsyncScheduleMode))
@pytest.mark.parametrize("residency", list(KVCacheManagementMode))
async def test_routed_suspend_resumes_active_request(monkeypatch, mode, residency):
    """Supported residency transitions preserve a partly generated routed result."""
    prompt, params = list(range(4, 20)), greedy_params(num_tokens_to_generate=16)
    offload = residency == KVCacheManagementMode.OFFLOAD
    if offload:
        saver = dynamic_context_module.torch_memory_saver
        for name in ("region", "pause", "resume"):
            monkeypatch.setattr(saver, name, mock.Mock(wraps=getattr(saver, name)))
    async with routed_model(
        monkeypatch,
        async_sched_mode=mode,
        kv_cache_management_mode=residency.value,
        static_kv_memory_pointers=offload,
        num_cuda_graphs=1 if offload else None,
        force_build_cuda_graphs=offload,
        use_cuda_graphs_for_non_decode_steps=not offload,
    ) as h:
        direct = await h.direct(prompt, params)
        await h.start()
        context, replays = h.engine.context, []
        if offload:
            replay = _CudaGraphRunner.replay_graph_capture

            def observed_replay(runner, *args, **kwargs):
                result = replay(runner, *args, **kwargs)
                active = slice(context.paused_request_count, context.total_request_count)
                ids = context.request_ids[active]
                live = 0 in h.engine.requests and not context._bookkeeping_no_real_work
                replays.extend([True] if live and 0 in ids else [])
                return result

            monkeypatch.setattr(_CudaGraphRunner, "replay_graph_capture", observed_replay)
        future = await _pause_active(h, monkeypatch, prompt, params)
        pointer = context.memory_buffer.data_ptr()
        if h.rank == 0:
            route = dict(h.service.coordinator.request_id_to_rank)
            assert route[0] == f"mp-coord-{h.rank}".encode() and 0 in h.engine.requests
            row = context.request_ids.tolist().index(0)
            blocks = context.request_to_kv_block_ids[row]
            blocks = blocks[: context.request_kv_block_counts[row]].long()
            target_kv = context.memory_buffer[:, :, blocks].view(torch.int16).clone()
            before = list(h.engine.get_request(0).generated_tokens)
            h.clients[0].suspend_engines()
        await asyncio.wait_for(h.engine.wait_until(EngineState.SUSPENDED), timeout=60)
        assert not context.is_tensor_state_allocated
        if residency == KVCacheManagementMode.PERSIST:
            assert context.memory_buffer.data_ptr() == pointer
        elif offload:
            saver.region.assert_called_once_with(tag=context.TMS_TAG, enable_cpu_backup=True)
            saver.pause.assert_called_once_with(context.TMS_TAG)
        else:
            assert not hasattr(context, "memory_buffer")
            assert not h.engine.controller._async_sched_logits.is_valid
        await h.barrier()
        if h.rank == 0:
            assert h.service.coordinator.request_id_to_rank == route and not future.done()
            h.clients[0].resume_engines()
        await asyncio.wait_for(h.engine.wait_until(EngineState.RESUMED), timeout=60)
        assert context.is_tensor_state_allocated and h.engine.state == EngineState.PAUSED
        assert h.engine._weight_epoch == 1
        if offload:
            assert context.memory_buffer.data_ptr() == pointer
            saver.resume.assert_called_once_with(context.TMS_TAG)
            if h.rank == 0:
                assert context.memory_buffer[:, :, blocks].view(torch.int16).equal(target_kv)
            replays[:] = h.witnesses[:] = []
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            final = await asyncio.wait_for(future, timeout=60)
            assert final["request_id"] == 0 and final["generated_tokens"][: len(before)] == before
            assert final["generated_tokens"] == direct["generated_tokens"]
            assert final["status"] == direct["status"]
            if residency == KVCacheManagementMode.RECOMPUTE:
                assert sum(bool(s["prefill"]) for s in h.witnesses) >= 2
        if offload:
            assert await h.sync.all_reduce_max(int(bool(replays))) == 1
        await h.barrier()
        h.assert_retired()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("active", [False, True], ids=["queued", "active"])
async def test_routed_abort_retires_only_cancelled_request(monkeypatch, active):
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
        direct = await h.direct(prompt, params)
        await h.start()
        h.engine.use_synchronous_zmq_collectives = synchronous
        calls, operation = [], h.engine.world_zmq_communicator.all_reduce_max

        async def observed_reduce(*args, **kwargs):
            result = await operation(*args, **kwargs)
            calls.append(kwargs["async_op"])
            return result

        monkeypatch.setattr(h.engine.world_zmq_communicator, "all_reduce_max", observed_reduce)
        future = await _pause_active(h, monkeypatch, prompt, params)
        if h.rank == 0:
            h.clients[0].set_generation_epoch(7)
        await until(lambda: h.engine._generation_epoch == 7)
        if h.rank == 0:
            request = h.engine.get_request(0)
            assert (request.policy_epoch, request.kv_cache_epoch) == ([(0, 7)], [(0, 7)])
        await h.barrier()
        await h.unpause()
        assert len(calls) >= 2 and set(calls) == {not synchronous}
        if h.rank == 0:
            final = await asyncio.wait_for(future, timeout=60)
            assert final["generated_tokens"] == direct["generated_tokens"]
            assert (final["policy_epoch"], final["kv_cache_epoch"]) == ([[0, 7]], [[0, 7]])
        await h.barrier()
        h.assert_retired()
