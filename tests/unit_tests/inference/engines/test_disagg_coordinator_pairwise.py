# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real coordinator-native prefill-to-decode handoff coverage."""

import asyncio
from copy import copy
from dataclasses import replace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.inference_client import InferenceClient
from tests.unit_tests.inference.engines.disagg_test_utils import (
    ForwardWitness,
    assert_released,
    collocated_reference,
    disagg_config,
    prompt,
    real_engine,
    sampling,
)
from tests.unit_tests.test_utilities import Utils


class RequestForwardWitness(ForwardWitness):
    def _snapshot(self):
        # Idle dummy rows reuse ID zero but have no live request owner.
        return super()._snapshot() if self.request_id in self.engine.requests else None


@pytest.fixture
def coordinator_world():
    """Build singleton model replicas and two disjoint role-DP groups."""
    Utils.initialize_model_parallel()
    assert dist.get_world_size() == 4
    assert dist.get_backend() == "nccl"
    control = dist.new_group(ranks=[0, 1, 2, 3], backend="gloo")
    prefill_dp = dist.new_group(ranks=[0, 2])
    decode_dp = dist.new_group(ranks=[1, 3])
    rank = dist.get_rank()
    yield control, prefill_dp if rank % 2 == 0 else decode_dp
    dist.destroy_process_group(prefill_dp if rank % 2 == 0 else decode_dp)
    dist.destroy_process_group(control)
    Utils.destroy_model_parallel()


async def _async_barrier(group, timeout=180):
    """Synchronize without starving a local engine task."""
    work = dist.barrier(group=group, async_op=True)

    async def poll():
        while not work.is_completed():
            await asyncio.sleep(0.01)
        work.wait()

    await asyncio.wait_for(poll(), timeout=timeout)


async def _wait_for(predicate, timeout=60):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


async def _collect_stream(stream):
    events = []
    async for event in stream:
        events.append(event)
    return events


def _valid_kv(engine, blocks, length):
    return engine.context.memory_buffer[:, :, blocks].flatten(2, 3)[:, :, :length].cpu().clone()


def _assert_same_kv(source, imported):
    assert source.shape == imported.shape
    assert source.dtype == imported.dtype
    assert torch.equal(source, imported), "transferred KV bytes differ from the pinned source"


async def _stop_services(engine, control, clients):
    """Stop both role fleets and their coordinators in production order."""
    rank = dist.get_rank()
    if rank == 0:
        for client in clients:
            client.pause_engines()
    await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=60)
    await _async_barrier(control)
    if rank == 0:
        for client in clients:
            client.stop_engines()
    await asyncio.wait_for(asyncio.shield(engine.engine_loop_task), timeout=60)
    await _async_barrier(control)
    if rank == 0:
        for client in clients:
            client.shutdown_coordinator()
        for client in clients:
            client.stop()
    await _async_barrier(control)
    process = getattr(engine, "inference_coordinator_process", None)
    if process is not None:
        process.join(timeout=30)
        assert not process.is_alive()


@pytest.mark.parametrize(
    "mode,streaming",
    [
        pytest.param(AsyncScheduleMode.LEGACY, False, id="eager"),
        pytest.param(AsyncScheduleMode.ASYNC, True, id="async-streaming"),
    ],
)
def test_two_coordinator_nixl_handoff(coordinator_world, mode, streaming):
    """Run a client-visible prefill -> NIXL import -> decode -> release flow."""
    config = disagg_config(async_sched_mode=mode)
    tokens = prompt(33)
    with torch.inference_mode():
        reference = replace(config, async_sched_mode=AsyncScheduleMode.LEGACY)
        weights, expected = collocated_reference(reference, tokens, sampling(7))
        asyncio.run(
            _run_coordinator_handoff(
                coordinator_world, config, tokens, weights, expected, streaming
            )
        )


async def _run_coordinator_handoff(coordinator_world, config, tokens, weights, expected, streaming):
    control, role_dp = coordinator_world
    rank = dist.get_rank()
    role = "prefill" if rank % 2 == 0 else "decode"
    with torch.inference_mode():
        local = {"source": None, "imported": None, "transfer": None}
        with real_engine(config, role=role, backend="nixl", weights=weights) as engine:
            # Coordinator setup reads pg_collection.dp directly. Preserve every
            # model group from the real factory and replace only the role's DP group.
            engine.pg_collection = copy(engine.pg_collection)
            engine.pg_collection.dp = role_dp
            assert dist.get_world_size(engine.pg_collection.tp) == 1
            assert dist.get_world_size(engine.pg_collection.pp) == 1
            assert dist.get_world_size(engine.pg_collection.mp) == 1
            assert dist.get_world_size(engine.pg_collection.dp) == 2

            original_capture = engine._capture_handoff_meta
            original_finalize = engine._finalize_kv_handoff_import

            def capture_source(request, prepared):
                original_capture(request, prepared)
                blocks = engine._pinned_handoff_blocks[request.request_id]
                local["source"] = _valid_kv(engine, blocks, len(tokens))
                local["source_id"] = request.request_id

            def capture_import(pending):
                assert pending.handle.poll()
                local["imported"] = _valid_kv(engine, pending.local_blocks, len(tokens))
                local["transfer"] = {
                    "request_id": pending.request_id,
                    "cached_blocks": pending.cached_prefix_block_count,
                    "xfers": len(pending.handle.xfers),
                }
                original_finalize(pending)

            with (
                RequestForwardWitness(engine, request_id=0) as witness,
                mock.patch.object(engine, "_capture_handoff_meta", side_effect=capture_source),
                mock.patch.object(
                    engine, "_finalize_kv_handoff_import", side_effect=capture_import
                ),
            ):
                # The shared tiny-engine factory uses an unpicklable lambda tokenizer.
                # This token-ID-only flow disables coordinator detokenization.
                engine.controller.tokenizer = None
                role_addr = await engine.start_listening_to_data_parallel_coordinator(
                    launch_inference_coordinator=True, hostname="127.0.0.1"
                )

                # The engine task has only just been created and cannot run until
                # this coroutine yields, so exchange both role addresses first.
                advertised = [None] * dist.get_world_size(control)
                dist.all_gather_object(advertised, (role, role_addr), group=control)
                addresses = {}
                for advertised_role, address in advertised:
                    addresses.setdefault(advertised_role, set()).add(address)
                assert set(addresses) == {"prefill", "decode"}
                assert all(len(role_addresses) == 1 for role_addresses in addresses.values())
                addresses = {key: value.pop() for key, value in addresses.items()}
                assert addresses["prefill"] != addresses["decode"]

                clients = ()
                client_result = None
                if rank == 0:
                    prefill_client = InferenceClient(addresses["prefill"])
                    decode_client = InferenceClient(addresses["decode"])
                    clients = (prefill_client, decode_client)
                    for client in clients:
                        client.start(connect_timeout_seconds=30)

                    prefill_result = await asyncio.wait_for(
                        prefill_client.add_request(
                            tokens, sampling(1, do_kv_handoff=True, detokenize_generations=False)
                        ),
                        timeout=120,
                    )
                    assert prefill_result["generated_tokens"] == expected[:1]
                    handoff = prefill_result["disaggregated_params"]
                    source_id = handoff["request_id"]
                    assert source_id == prefill_result["request_id"]
                    assert len(handoff["block_ids"]) == 3
                    assert handoff["kv_meta"]["agent_name"] == "prefill-rank0"
                    assert handoff["kv_meta"]["agent_metadata_b64"]

                    decode_params = sampling(7, streaming_interval=2, detokenize_generations=False)
                    if streaming:
                        stream = decode_client.add_request_with_kv_handoff_streaming(
                            tokens, decode_params, handoff["kv_meta"], handoff["block_ids"]
                        )
                        events = await asyncio.wait_for(_collect_stream(stream), timeout=120)
                        partials = [event["partial"]["new_tokens"] for event in events[:-1]]
                        decode_result = events[-1]["final"]
                        incremental = [token for part in partials for token in part]
                        assert [len(part) for part in partials] == [2, 2, 2]
                        assert incremental == expected[:6]
                    else:
                        decode_result = await asyncio.wait_for(
                            decode_client.add_request_with_kv_handoff(
                                tokens, decode_params, handoff["kv_meta"], handoff["block_ids"]
                            ),
                            timeout=120,
                        )
                    assert decode_result["generated_tokens"] == expected
                    assert decode_result["num_cached_tokens"] == len(tokens)
                    prefill_client.release_handoff(source_id)
                    await _wait_for(lambda: not engine._pinned_handoff_blocks)
                    client_result = {
                        "source_id": source_id,
                        "prefill_tokens": prefill_result["generated_tokens"],
                        "decode_tokens": decode_result["generated_tokens"],
                    }

                # Ranks without clients keep yielding to their local engine task.
                await _async_barrier(control)
                await asyncio.sleep(0.05)
                assert_released(engine)
                evidence = {
                    "rank": rank,
                    "role": role,
                    "role_dp_rank": dist.get_rank(role_dp),
                    "is_dp_coordinator": engine.is_dp_coordinator,
                    "steps": list(witness.steps),
                    "pending": list(witness.pending_forwards),
                    "source": local["source"],
                    "source_id": local.get("source_id"),
                    "imported": local["imported"],
                    "transfer": local["transfer"],
                    "client_result": client_result,
                }
                gathered = [None] * dist.get_world_size(control)
                dist.all_gather_object(gathered, evidence, group=control)

                source_owners = [item for item in gathered if item["source"] is not None]
                decode_owners = [item for item in gathered if item["imported"] is not None]
                assert [item["rank"] for item in source_owners] == [0]
                assert [item["rank"] for item in decode_owners] == [1]
                assert all(
                    item["is_dp_coordinator"] == (item["role_dp_rank"] == 0) for item in gathered
                )
                assert all(not item["steps"] for item in gathered if item["rank"] in (2, 3))
                assert source_owners[0]["steps"] == [(0, len(tokens), True, (0,))]
                assert decode_owners[0]["steps"]
                assert bool(decode_owners[0]["pending"]) == streaming
                assert all(not step[2] for step in decode_owners[0]["steps"])
                assert decode_owners[0]["steps"][0][0] == len(tokens)
                assert decode_owners[0]["transfer"] == {
                    "request_id": 0,
                    "cached_blocks": 0,
                    "xfers": 1,
                }
                source_kv = source_owners[0]["source"]
                imported_kv = decode_owners[0]["imported"]
                corrupted = imported_kv.clone()
                corrupted.view(torch.uint8).flatten()[0] ^= 1
                with pytest.raises(AssertionError, match="transferred KV bytes"):
                    _assert_same_kv(source_kv, corrupted)
                _assert_same_kv(source_kv, imported_kv)
                result = next(item["client_result"] for item in gathered if item["client_result"])
                assert result["source_id"] == source_owners[0]["source_id"]
                assert result["prefill_tokens"] == expected[:1]
                assert result["decode_tokens"] == expected

                await _stop_services(engine, control, clients)
