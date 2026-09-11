# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native source-lifetime safety at destructive engine boundaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest import mock

import torch
import torch.distributed as dist

from tests.unit_tests.inference.engines.disagg_test_utils import (
    admit_import,
    assert_import_equal,
    assert_released,
    disagg_config,
    exchange,
    prompt,
    real_engine,
    run_to_completion,
    sampling,
    snapshot_source,
)
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401


def _terminal_error(future):
    try:
        future.result(timeout=30)
    except Exception as error:  # noqa: BLE001 - the exact boundary error is the oracle below
        return type(error).__name__, str(error)
    return None, ""


@torch.inference_mode()
def test_recompute_suspend_rejects_registered_nixl_source_pin(transport_world):
    """A registered request pin rejects suspend before context deallocation."""

    rank = dist.get_rank()
    assert dist.get_world_size() == 4, "safety pair requires two explicit source/decode pairs"
    source = rank % 2 == 0
    tokens = prompt(33)
    config = disagg_config(kv_cache_management_mode="recompute", static_kv_memory_pointers=False)
    with real_engine(config, role="prefill" if source else "decode", backend="nixl") as engine:
        metadata = state = None
        if source:
            request = run_to_completion(
                engine, engine.add_request(101, tokens, sampling(1, do_kv_handoff=True))
            )
            metadata, state = snapshot_source(engine, request)
            assert metadata["kv_meta"]["base_addr"] == engine.context.memory_buffer.data_ptr()
        peer = exchange((metadata, state) if source else None, transport_world)
        if not source:
            metadata, state = peer

        entered = Event()
        refuse_deallocation = Event()
        suspend_task = suspend_pool = None
        if source:

            def deallocate_boundary():
                entered.set()
                if not refuse_deallocation.wait(timeout=30):
                    raise TimeoutError("test deallocation boundary was not released")
                raise RuntimeError("test refused registered source deallocation")

            def suspend_source():
                torch.cuda.set_device(rank)
                with mock.patch.object(
                    engine.context,
                    "deallocate_inference_state_buffers",
                    side_effect=deallocate_boundary,
                ):
                    engine.suspend()

            suspend_pool = ThreadPoolExecutor(max_workers=1)
            suspend_task = suspend_pool.submit(suspend_source)
            entered.wait(timeout=10)

        local_boundary = (entered.is_set(), suspend_task.done()) if source else None
        peer_boundary = exchange(local_boundary, transport_world)
        native = []
        pending = future = None
        start_error = ""
        if not source:
            backend = engine._kv_transfer_agent
            begin_transfer = backend._begin_transfer

            def witnessed_read(*args, **kwargs):
                xfer, context = begin_transfer(*args, **kwargs)
                native.append((context, backend._agent.check_xfer_state(xfer)))
                return xfer, context

            try:
                with mock.patch.object(backend, "_begin_transfer", side_effect=witnessed_read):
                    future = engine.add_request_with_kv_handoff(
                        101, tokens, sampling(1), metadata["kv_meta"], metadata["block_ids"]
                    )
                pending = engine._pending_kv_imports[0]
                owned = torch.tensor(
                    pending.local_blocks + pending.continuation_blocks, dtype=torch.int64
                )
                assert (engine.context.kv_block_allocator.block_ref_counts[owned] > 0).all()
            except Exception as error:  # ensure the source boundary is always released safely
                start_error = repr(error)

        read_witness = exchange((native, start_error) if not source else None, transport_world)
        if source:
            refuse_deallocation.set()
            suspend_error = _terminal_error(suspend_task)
            suspend_pool.shutdown()
        else:
            suspend_error = None
        peer_suspend = exchange(suspend_error if source else None, transport_world)

        if not source and pending is not None:
            for handle in engine._pending_transfer_handles(pending):
                handle.wait()
                assert handle.poll()
            assert_import_equal(engine, pending, state, len(tokens))
            admit_import(engine)
            assert future.done() and future.result().merge().generated_tokens == [
                *metadata["kv_meta"]["resume_tokens"]
            ]
        dist.barrier(group=transport_world)
        if source:
            engine.release_handoff_blocks(101)
        assert_released(engine)

        observed_native, native_error = read_witness if source else (native, start_error)
        observed_suspend = suspend_error if source else peer_suspend
        observed_boundary = local_boundary if source else peer_boundary
        print(
            f"NIXL_SOURCE_SAFETY rank={rank} role={'source' if source else 'decode'} "
            f"boundary={observed_boundary} native={observed_native} suspend={observed_suspend}",
            flush=True,
        )
        assert not native_error
        assert observed_native and all(state not in ("DONE", "ERR") for _, state in observed_native)
        assert not observed_boundary[0], (
            "DynamicInferenceEngine.suspend entered destructive context deallocation "
            "for a registered request pin; during the held call a native NIXL READ "
            f"entered its live state: {observed_native}"
        )
        assert observed_suspend[0] == "RuntimeError"
        assert "handoff state remains pinned" in observed_suspend[1]
