# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Collocated parity and exact transfer witnesses for real disaggregated engines."""

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.config import AsyncScheduleMode
from tests.unit_tests.inference.engines.disagg_test_utils import (
    ForwardWitness,
    admit_import,
    assert_import_equal,
    assert_released,
    collocated_reference,
    complete_transfer,
    disagg_config,
    exchange,
    prompt,
    real_engine,
    run_to_completion,
    sampling,
    snapshot_source,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def transport_world():
    Utils.initialize_model_parallel()
    assert dist.get_world_size() >= 2 and dist.get_world_size() % 2 == 0
    assert dist.get_backend() == "nccl"
    control = dist.new_group(backend="gloo")
    assert dist.get_backend(control) == "gloo"
    yield control
    dist.destroy_process_group(control)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "backend,length,changes,count",
    [
        pytest.param("nccl", 32, {}, 7, id="nccl-boundary"),
        pytest.param("nccl", 33, {}, 7, id="nccl-partial-tail"),
        pytest.param("nixl", 33, {}, 7, id="nixl-partial-tail"),
        pytest.param(
            "nccl", 33, {"hidden_size": 256, "flash_attention_version": 4}, 7, id="fa4-d64"
        ),
        pytest.param(
            "nccl", 33, {"async_sched_mode": AsyncScheduleMode.ASYNC}, 7, id="async-decode"
        ),
        pytest.param(
            "nccl",
            129,
            {"enable_chunked_prefill": True, "context_max_tokens": 64},
            7,
            id="chunked-prefill",
        ),
        pytest.param("nccl", 33, {"sampling_backend": "flashinfer"}, 7, id="flashinfer"),
        pytest.param("nccl", 33, {}, 1, id="terminal-first-token"),
        pytest.param("nccl", 33, {"model_provider": "hybrid"}, 7, id="mamba-exact-state"),
        pytest.param(
            "nccl",
            33,
            {"num_cuda_graphs": 2, "force_build_cuda_graphs": True},
            7,
            id="decode-graph",
        ),
    ],
)
@torch.inference_mode()
def test_disagg_real_engine_parity(transport_world, backend, length, changes, count):
    config = disagg_config(**changes)
    tokens = prompt(length)
    reference_config = replace(config, async_sched_mode=AsyncScheduleMode.LEGACY)
    weights, expected = collocated_reference(reference_config, tokens, sampling(count))
    assert exchange(expected, transport_world) == expected
    source = dist.get_rank() % 2 == 0
    async_decode = config.async_sched_mode == AsyncScheduleMode.ASYNC
    chunked = changes.get("enable_chunked_prefill")
    neighbor_tokens = [91, 92, 93, 94]
    if async_decode or chunked:
        with real_engine(reference_config, weights=weights) as reference:
            future = reference.add_request(102, neighbor_tokens, sampling(16))
            neighbor_expected = run_to_completion(reference, future).generated_tokens
    role = "prefill" if source else "decode"
    with real_engine(config, role=role, backend=backend, weights=weights) as engine:
        with ForwardWitness(engine) as witness:
            neighbor = None
            if async_decode:
                metadata = None
                if source:
                    seed = engine.add_request(102, neighbor_tokens, sampling(1, do_kv_handoff=True))
                    metadata = run_to_completion(engine, seed).disaggregated_params
                metadata = exchange(metadata, transport_world)
                _, seeded = complete_transfer(
                    engine, metadata, neighbor_tokens, sampling(16), transport_world, 102
                )
                if source:
                    engine._poll_pending_kv_pushes()
                    engine.release_handoff_blocks(102)
                else:
                    admit_import(engine)
                    neighbor = seeded
                    engine.step_modern()
                    assert engine.controller._async_sched_logits.is_valid
            elif source and chunked:
                neighbor = engine.add_request(102, neighbor_tokens, sampling(16))
                engine.step_modern()
                assert not neighbor.done() and engine.context.is_decode_only()
            metadata = state = None
            if source:
                request = run_to_completion(
                    engine, engine.add_request(101, tokens, sampling(1, do_kv_handoff=True))
                )
                assert request.generated_tokens == expected[:1]
                metadata, state = snapshot_source(engine, request)
                assert len(metadata["block_ids"]) == (length + 15) // 16
                assert witness.steps and all(step[2] for step in witness.steps)
                if chunked:
                    assert len({step[0] for step in witness.steps}) >= 3
                    assert sum(102 in step[3] for step in witness.steps) >= 3
                with pytest.raises(RuntimeError, match="handoff state remains pinned"):
                    engine.reset()
            transferred = exchange((metadata, state) if source else None, transport_world)
            if not source:
                metadata, state = transferred
            pending, future = complete_transfer(
                engine, metadata, tokens, sampling(count), transport_world
            )
            if not source:
                assert_import_equal(engine, pending, state, length)
                if length == 32:
                    block = pending.local_blocks[0]
                    transferred = engine.context.memory_buffer[0, 0, block, 0, 0, 0]
                    saved = transferred.clone()
                    transferred.copy_(saved + 1)
                    with pytest.raises(AssertionError):
                        assert_import_equal(engine, pending, state, length)
                    transferred.copy_(saved)
                    assert_import_equal(engine, pending, state, length)
                assert pending.resume_tokens == expected[:1]
                if neighbor is None:
                    assert not engine.context.total_request_count
                    admit_import(engine)
                else:
                    assert engine._poll_pending_kv_imports() == 1
                    engine.step_modern()  # Resolve the existing chain before normal import admission.
                if count > 1:
                    row = engine.context.request_ids.tolist().index(101)
                    if neighbor is None:
                        assert engine.context.request_kv_length_offsets[row].item() == length
                    assert not engine.context.request_in_prefill_status_tensor[row].item()
                    result = run_to_completion(engine, future)
                    assert witness.steps and all(not step[2] for step in witness.steps)
                    assert witness.steps[0][0] == length
                    if changes.get("force_build_cuda_graphs"):
                        assert witness.graph_replays > 0
                    if config.flash_attention_version == 4:
                        assert witness.fa4_calls > 0
                    if neighbor is not None:
                        assert witness.pending_forwards > 0
                        assert any(102 in step[3] for step in witness.steps)
                else:
                    assert future.done()
                    result = future.result().merge()
                    assert not witness.steps
                assert result.generated_tokens == expected
                assert result.num_cached_tokens == length
                if neighbor is None:
                    assert_released(engine)
            if neighbor is not None:
                assert run_to_completion(engine, neighbor).generated_tokens == neighbor_expected
                if not source:
                    assert_released(engine)
            dist.barrier(group=transport_world)
            if source:
                engine._poll_pending_kv_pushes()
                assert not engine._pending_kv_pushes
                engine.release_handoff_blocks(101)
                engine.release_handoff_blocks(101)
                assert_released(engine)
