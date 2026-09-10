# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real allocator/backpressure and message-level disaggregation regressions."""

import uuid

import msgpack
import pytest
import torch
import torch.distributed as dist
import zmq

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.headers import Headers
from tests.unit_tests.inference.engines.disagg_test_utils import (
    ForwardWitness,
    assert_import_equal,
    assert_released,
    collocated_reference,
    decode_peer_meta,
    disagg_config,
    exchange,
    prompt,
    real_engine,
    run_to_completion,
    sampling,
    snapshot_source,
)
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("policy", list(PrefixCachingEvictionPolicy))
@torch.inference_mode()
def test_real_handoff_capacity_fifo(transport_world, policy):
    config = disagg_config(prefix_caching_eviction_policy=policy)
    tokens = prompt(33)
    weights, expected = collocated_reference(config, tokens, sampling())
    source = dist.get_rank() % 2 == 0
    with real_engine(config, role="prefill" if source else "decode", weights=weights) as engine:
        metadata = state = None
        if source:
            result = run_to_completion(
                engine, engine.add_request(101, tokens, sampling(1, do_kv_handoff=True))
            )
            metadata, state = snapshot_source(engine, result)
        peer = exchange((metadata, state) if source else None, transport_world)
        allocator = engine.context.kv_block_allocator
        held = None
        if not source:
            metadata, state = peer
            # Consume actual capacity. No allocator predicate or backend is mocked.
            held = allocator.allocate_memory_blocks(allocator.pool_avail).clone()
            futures = [
                engine.add_request_with_kv_handoff(
                    rid, tokens, sampling(), metadata["kv_meta"], metadata["block_ids"]
                )
                for rid in (101, 102)
            ]
            assert [item.request_id for item in engine._deferred_kv_handoffs] == [101, 102]
            assert not engine._pending_kv_imports
            allocator.release_memory_blocks(held[:2])
            assert engine._drain_deferred_kv_handoffs() == 0
            allocator.release_memory_blocks(held[2:3])
            assert engine._drain_deferred_kv_handoffs() == 1
            assert [item.request_id for item in engine._deferred_kv_handoffs] == [102]
        for index, rid in enumerate((101, 102)):
            pending = None
            if not source:
                if index:
                    assert engine._drain_deferred_kv_handoffs() == 1
                assert len(engine._pending_kv_imports) == 1
                pending = engine._pending_kv_imports[0]
                assert pending.request_id == rid
            peer = exchange(None if source else decode_peer_meta(engine, pending), transport_world)
            if source:
                engine.push_handoff_kv(101, [peer])
            handles = (
                [handle for _, group in engine._pending_kv_pushes for handle in group]
                if source
                else engine._pending_transfer_handles(pending)
            )
            for handle in handles:
                handle.wait()
                assert handle.poll()
            if not source:
                assert_import_equal(engine, pending, state, len(tokens))
                assert engine._poll_pending_kv_imports() == 1
                assert engine._admit_pending_kv_imports() == 1
                with ForwardWitness(engine, rid) as witness:
                    result = run_to_completion(engine, futures[index])
                assert result.generated_tokens == expected
                assert witness.steps and all(not step[2] for step in witness.steps)
            else:
                assert engine._poll_pending_kv_pushes() == 1
            dist.barrier(group=transport_world)
        if source:
            engine.release_handoff_blocks(101)
        else:
            allocator.release_memory_blocks(held[3:])
        assert_released(engine)


@torch.inference_mode()
def test_abort_capacity_queued_handoff_resolves_without_transfer():
    """A delivered ABORT must reach handoffs not yet present in engine.requests.

    This is a focused negative/lifecycle regression, not positive transfer credit.
    """
    Utils.initialize_model_parallel()
    try:
        with real_engine(disagg_config(), role="decode") as engine:
            allocator = engine.context.kv_block_allocator
            held = allocator.allocate_memory_blocks(allocator.pool_avail).clone()
            future = engine.add_request_with_kv_handoff(
                101, prompt(33), sampling(), {"resume_tokens": [9]}, [1, 2, 3]
            )
            assert not engine.requests and not engine._pending_kv_imports
            assert [item.request_id for item in engine._deferred_kv_handoffs] == [101]
            context = zmq.Context()
            receiver, sender, publisher = (
                context.socket(zmq.PAIR),
                context.socket(zmq.PAIR),
                context.socket(zmq.PUB),
            )
            address = f"inproc://abort-handoff-{uuid.uuid4().hex}"
            receiver.bind(address)
            sender.connect(address)
            engine.is_mp_coordinator = True
            engine.socket_for_receiving_requests = receiver
            engine.model_parallel_publisher_socket = publisher
            try:
                sender.send_multipart(
                    [msgpack.packb([Headers.ABORT_REQUEST.value, 101], use_bin_type=True)]
                )
                assert receiver.poll(1000)
                with ForwardWitness(engine) as witness:
                    assert engine.schedule_requests() == 1
                # Assert while capacity is still exhausted, before teardown can
                # cancel anything or release the independently held blocks.
                resolved = future.done()
                retired = not engine._deferred_kv_handoffs and not engine._pending_kv_imports
                assert not witness.steps
                assert torch.equal(allocator.block_ref_counts[held], torch.ones_like(held))
            finally:
                for socket in (receiver, sender, publisher):
                    socket.close(linger=0)
                context.term()
                allocator.release_memory_blocks(held)
                engine._reset_pending_kv_imports()
            assert resolved, "Delivered ABORT left a capacity-queued handoff future unresolved"
            assert retired, "Aborted handoff remained eligible for a later real transfer"
    finally:
        Utils.destroy_model_parallel()
