# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-engine, real-transport handoff fixtures; no transfer or forward substitutes."""

import asyncio
import gc
from contextlib import contextmanager
from dataclasses import replace
from unittest import mock

import torch
import torch.distributed as dist

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.disaggregation.engine import DisaggDynamicInferenceEngine
from megatron.core.inference.inference_request import Status
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from tests.unit_tests.inference.engines import test_dynamic_engine as dynamic_tests
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicInferenceEngineTestBase,
)


def disagg_config(**changes):
    return DynamicEngineTestConfig(
        **(
            dict(
                num_requests=0,
                max_sequence_length=512,
                context_buffer_size_gb=0.02,
                context_block_size_tokens=16,
                context_max_requests=8,
                context_max_tokens=256,
                num_cuda_graphs=None,
                enable_prefix_caching=True,
                transformer_impl="transformer_engine",
                async_sched_mode=AsyncScheduleMode.LEGACY,
            )
            | changes
        )
    )


def sampling(count=7, **changes):
    return SamplingParams(top_k=1, termination_id=-1, num_tokens_to_generate=count, **changes)


def prompt(length):
    return [3 + (index * 7) % 83 for index in range(length)]


@contextmanager
def real_engine(config, *, role=None, backend="nccl", weights=None):
    # Reuse the existing tiny-model factory, substituting only its production
    # engine composition. The model, controller, allocator and transport are real.
    with mock.patch.object(dynamic_tests, "DynamicInferenceEngine", DisaggDynamicInferenceEngine):
        engine = DynamicInferenceEngineTestBase._build_test_env(config).engine
    model = engine.controller.inference_wrapped_model.model
    if weights is not None:
        model.load_state_dict(weights)
        for name, value in model.state_dict().items():
            if torch.is_tensor(value):
                assert torch.equal(value.cpu(), weights[name]), name
    if role is not None:
        engine.setup_kv_transfer(role, backend=backend)
    try:
        yield engine
    finally:
        # Safety assertions belong to the caller, before this fixture teardown.
        for agent in [engine._kv_transfer_agent, *engine._ssm_transfer_agents.values()]:
            close = getattr(agent, "close", None)
            if close is not None:
                close()
        for socket in getattr(engine, "zmq_sockets", []):
            socket.close(linger=0)
        delete_cuda_graphs()
        del model, engine
        gc.collect()
        torch.cuda.empty_cache()


def run_to_completion(engine, future, *, limit=256):
    for _ in range(limit):
        if future.done():
            result = future.result()
            request = result.merge()
            assert request.status == Status.COMPLETED
            return request
        engine.step_modern()
    raise AssertionError("Request did not complete within the bounded real-engine schedule")


def collocated_reference(config, tokens, params):
    # Broadcast actual weights; equal seeds alone are not an identity oracle.
    with real_engine(
        replace(config, num_cuda_graphs=None, force_build_cuda_graphs=False)
    ) as engine:
        model = engine.controller.inference_wrapped_model.model
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
        weights = {
            name: value.detach().cpu().clone() if torch.is_tensor(value) else value
            for name, value in model.state_dict().items()
        }
        result = run_to_completion(engine, engine.add_request(101, tokens, params))
        return weights, list(result.generated_tokens)


class ForwardWitness:
    """Observe the target's real compute inputs, including graph replay."""

    def __init__(self, engine, request_id=101):
        self.engine = engine
        self.request_id = request_id
        self.steps = []
        self.graph_replays = 0
        self._patches = []

    def _snapshot(self):
        context = self.engine.context
        ids = context.request_ids[: context.total_request_count].tolist()
        if self.request_id not in ids:
            return None
        row = ids.index(self.request_id)
        return (
            int(context.request_kv_length_offsets[row]),
            int(context.request_query_lengths[row]),
            bool(context.request_in_prefill_status_tensor[row]),
            tuple(ids),
        )

    def __enter__(self):
        wrapper = self.engine.controller.inference_wrapped_model
        original_forward = wrapper.run_one_forward_step
        original_replay = torch.cuda.CUDAGraph.replay

        def forward(*args, **kwargs):
            snapshot = self._snapshot()
            result = original_forward(*args, **kwargs)
            if snapshot is not None:
                self.steps.append(snapshot)
            return result

        def replay(graph, *args, **kwargs):
            snapshot = self._snapshot()
            result = original_replay(graph, *args, **kwargs)
            if snapshot is not None:
                self.graph_replays += 1
                self.steps.append(snapshot)
            return result

        self._patches = [
            mock.patch.object(wrapper, "run_one_forward_step", side_effect=forward),
            mock.patch.object(torch.cuda.CUDAGraph, "replay", replay),
        ]
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.stop()


def exchange(value, group):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value, group=group)
    return values[dist.get_rank() ^ 1]


def snapshot_source(engine, request):
    metadata = request.disaggregated_params
    blocks = metadata["block_ids"]
    context = engine.context
    state = {"kv": context.memory_buffer[:, :, blocks].detach().cpu().clone()}
    if context.is_hybrid_model:
        slot = engine._pinned_handoff_ssm_slots[101]
        state["conv"] = context.mamba_conv_states[:, slot].detach().cpu().clone()
        state["recurrent"] = context.mamba_ssm_states[:, slot].detach().cpu().clone()
    assert engine._pinned_handoff_blocks[101] == blocks
    assert (context.kv_block_allocator.block_ref_counts[blocks] > 0).all()
    return metadata, state


def decode_peer_meta(engine, pending):
    metadata = dict(engine._kv_transfer_agent.export_meta(), block_ids=pending.local_blocks)
    if pending.ssm is not None:
        metadata["ssm"] = {
            kind: dict(agent.export_meta(), block_ids=[pending.ssm.live_slot])
            for kind, agent in engine._ssm_transfer_agents.items()
        }
    return metadata


def assert_import_equal(engine, pending, source, prompt_length):
    # The last physical block can include uninitialized padding: compare only
    # tokens belonging to the transferred prompt, never the allocator tail.
    block_size = engine.context.block_size_tokens
    for index, block in enumerate(pending.local_blocks):
        count = min(block_size, prompt_length - index * block_size)
        actual = engine.context.memory_buffer[:, :, block, :count].cpu()
        assert torch.equal(actual, source["kv"][:, :, index, :count])
    if pending.ssm is not None:
        slot = pending.ssm.live_slot
        assert torch.equal(engine.context.mamba_conv_states[:, slot].cpu(), source["conv"])
        assert torch.equal(engine.context.mamba_ssm_states[:, slot].cpu(), source["recurrent"])


def complete_transfer(engine, metadata, tokens, params, control):
    """Drive real P2P/pull while both peers keep their storage alive."""
    is_source = dist.get_rank() % 2 == 0
    pending = None
    future = None
    if not is_source:
        future = engine.add_request_with_kv_handoff(
            101, tokens, params, metadata["kv_meta"], metadata["block_ids"]
        )
        assert len(engine._pending_kv_imports) == 1
        pending = engine._pending_kv_imports[0]
    peer = exchange(None if is_source else decode_peer_meta(engine, pending), control)
    if is_source and engine._kv_transfer_agent.is_push:
        engine.push_handoff_kv(101, [peer])
    handles = (
        [handle for _, group in engine._pending_kv_pushes for handle in group]
        if is_source
        else engine._pending_transfer_handles(pending)
    )
    for handle in handles:
        handle.wait()
        assert handle.poll()
    dist.barrier(group=control)
    return pending, future


def admit_import(engine):
    assert engine._poll_pending_kv_imports() == 1
    assert engine._admit_pending_kv_imports() == 1
    engine._loop.run_until_complete(asyncio.sleep(0))
    assert not engine._pending_kv_imports
    assert not engine.waiting_request_ids
    assert not engine._handoff_completion_notifications


def assert_released(engine):
    assert not engine._pending_kv_imports
    assert not engine._deferred_kv_handoffs
    assert not engine._quarantined_kv_imports
    assert not engine._pinned_handoff_blocks
    assert not engine._pinned_handoff_ssm_slots
    assert not engine.requests
    assert not engine.waiting_request_ids
    assert engine.context.total_request_count == 0
    assert (engine.context.kv_block_allocator.block_ref_counts == 0).all()
