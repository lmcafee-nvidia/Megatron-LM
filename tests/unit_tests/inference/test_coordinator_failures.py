# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import queue
import threading
import time
from collections import Counter
from types import SimpleNamespace

import msgpack
import pytest
import torch
import zmq
from zmq.utils.monitor import recv_monitor_message

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams


class _ObservedSocket:
    """Count actual terminal frames without replacing transport or client parsing."""

    def __init__(self, socket):
        self.socket, self.replies = socket, Counter()

    def __getattr__(self, name):
        return getattr(self.socket, name)

    def recv_multipart(self, *args, **kwargs):
        frames = self.socket.recv_multipart(*args, **kwargs)
        metadata = msgpack.unpackb(frames[0], raw=False)
        if metadata[0] == Headers.ENGINE_REPLY.value:
            self.replies[metadata[1]] += 1
        return frames


class _CoordinatorRuntime:
    def __init__(self):
        self.addresses = queue.Queue()
        self.errors = queue.Queue()
        self.processed = Counter()
        self.ready = threading.Event()
        self.monitor_endpoint = "inproc://coordinator-monitor"
        self.coordinator = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        assert self.ready.wait(timeout=5.0)
        self.address = self.addresses.get(timeout=1.0)
        self.monitor = self.coordinator.context.socket(zmq.PAIR)
        self.monitor.connect(self.monitor_endpoint)

    def _run(self):
        try:
            coordinator = DataParallelInferenceCoordinator(
                SimpleNamespace(send=self.addresses.put, close=lambda: None),
                data_parallel_size=0,
                tokenizer=None,
                max_requests=8,
                hostname="127.0.0.1",
            )
            self.coordinator = coordinator
            for header, handler in list(coordinator._handlers.items()):

                def observed(*args, header=header, handler=handler):
                    result = handler(*args)
                    self.processed[header] += 1
                    return result

                coordinator._handlers[header] = observed
            coordinator.router_socket.monitor(self.monitor_endpoint, zmq.EVENT_ALL)
            self.ready.set()
            coordinator.start()
        except BaseException as error:  # Surface background failures in the test thread.
            self.errors.put(error)
            self.ready.set()
        finally:
            if self.coordinator is not None:
                self.coordinator.router_socket.disable_monitor()
                self.coordinator.stop()

    def assert_healthy(self):
        if not self.errors.empty():
            raise AssertionError("coordinator thread failed") from self.errors.get_nowait()
        assert self.thread.is_alive(), "coordinator loop exited unexpectedly"

    def shutdown(self):
        if self.thread.is_alive():
            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            socket.connect(self.address)
            socket.send(msgpack.packb([Headers.CONNECT.value], use_bin_type=True))
            if socket.poll(5000) & zmq.POLLIN:
                socket.recv_multipart()
                socket.send(msgpack.packb([Headers.SHUTDOWN.value], use_bin_type=True))
            socket.close(linger=0)
            context.term()
        self.monitor.close(linger=0)
        self.thread.join(timeout=5.0)
        assert not self.thread.is_alive(), "coordinator thread did not stop"
        if not self.errors.empty():
            raise AssertionError("coordinator thread failed") from self.errors.get_nowait()

    async def wait_for_disconnect(self):
        def disconnected():
            while self.monitor.poll(0):
                if recv_monitor_message(self.monitor)["event"] == zmq.EVENT_DISCONNECTED:
                    return True

        await _eventually(disconnected, "coordinator did not observe disconnect")

    async def wait_for_engine(self, engine):
        await _eventually(
            lambda: engine.identity in self.coordinator.identities_of_data_parallel_ranks,
            "engine did not register",
        )


class _EnginePeer:
    def __init__(self, address, identity):
        self.identity = identity
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, identity)
        self.socket.connect(address)
        self.socket.send(b"")

    def receive_request(self):
        assert self.socket.poll(5000) & zmq.POLLIN, "engine did not receive request"
        frames = self.socket.recv_multipart()
        metadata = msgpack.unpackb(frames[0], raw=False)
        assert Headers(metadata[0]) in (Headers.SUBMIT_REQUEST, Headers.SUBMIT_REQUEST_WITH_KV)
        return metadata[1]

    def disconnect(self):
        self.socket.send(msgpack.packb([Headers.DISCONNECT.value], use_bin_type=True))

    def send_partial(self, request_id):
        metadata = msgpack.packb(
            [Headers.ENGINE_REPLY_PARTIAL.value, [request_id]], use_bin_type=True
        )
        body = msgpack.packb({"request_id": request_id, "new_tokens": [1]}, use_bin_type=True)
        self.socket.send_multipart([metadata, body])

    def send_final(self, request_id):
        metadata = msgpack.packb(
            [Headers.ENGINE_REPLY.value, [[request_id, False]]], use_bin_type=True
        )
        completed_request = DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=torch.empty(0, dtype=torch.int64),
            sampling_params=SamplingParams(detokenize_generations=False),
            status=Status.COMPLETED,
        )
        body = msgpack.packb(completed_request.serialize(), use_bin_type=True)
        self.socket.send_multipart([metadata, body])

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


@pytest.fixture
def coordinator_runtime():
    runtime = _CoordinatorRuntime()
    yield runtime
    runtime.shutdown()


def _start_client(runtime, *, deserialize=False):
    client = InferenceClient(runtime.address, deserialize=deserialize)
    client.socket = _ObservedSocket(client.socket)
    client.start(connect_timeout_seconds=5.0)
    return client


async def _eventually(predicate, message):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(message)


def _assert_reply(reply, status=Status.FAILED):
    if isinstance(reply, DynamicInferenceRequest):
        reply = reply.serialize()
    assert reply["status"] == status.name
    if status == Status.FAILED:
        assert [event["type"] for event in reply["events"]].count("FAIL") == 1


def _assert_no_requests(coordinator):
    for field in ("client_id", "client_request_id", "rank", "sampling_params"):
        assert not getattr(coordinator, f"request_id_to_{field}", {}), field
    assert coordinator.client_request_to_request_id == {}


@pytest.mark.asyncio
async def test_engine_disconnect_fails_owned_request_and_ignores_late_reply(coordinator_runtime):
    runtime = coordinator_runtime
    engine = _EnginePeer(runtime.address, b"removed-engine")
    first_client, second_client = [_start_client(runtime) for _ in range(2)]
    try:
        await runtime.wait_for_engine(engine)
        futures = [
            first_client.add_request([1, 2], SamplingParams(num_tokens_to_generate=1)),
            second_client.add_request([3, 4], SamplingParams(num_tokens_to_generate=1)),
        ]
        resolved = []
        for future in futures:
            future.add_done_callback(resolved.append)
        server_request_ids = [engine.receive_request(), engine.receive_request()]
        await _eventually(
            lambda: len(runtime.coordinator.request_id_to_rank) == 2, "unowned requests"
        )
        assert set(runtime.coordinator.request_id_to_rank.values()) == {engine.identity}
        assert runtime.coordinator._pending_counts.tolist() == [2]

        engine.disconnect()
        replies = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
        for reply in replies:
            _assert_reply(reply)
        assert {reply["request_id"] for reply in replies} == set(server_request_ids)
        await _eventually(
            lambda: engine.identity not in runtime.coordinator.identities_of_data_parallel_ranks,
            "engine was not removed",
        )
        _assert_no_requests(runtime.coordinator)
        assert runtime.coordinator._pending_counts.size == 0

        for server_request_id in server_request_ids:
            engine.send_final(server_request_id)
        await _eventually(
            lambda: runtime.processed[Headers.ENGINE_REPLY] == len(server_request_ids),
            "late finals were not processed",
        )
        runtime.assert_healthy()
        assert Counter(resolved) == Counter(futures)
        for client in (first_client, second_client):
            client._connect_with_inference_coordinator(timeout_seconds=5)
            assert not client.listener_task.done()
        assert first_client.socket.replies == second_client.socket.replies == {0: 1}
        _assert_no_requests(runtime.coordinator)
    finally:
        first_client.stop()
        second_client.stop()
        engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["plain", "kv-handoff", "deserialize"])
async def test_no_engines_fails_submission_but_keeps_loop_alive(coordinator_runtime, kind):
    runtime = coordinator_runtime
    client = _start_client(runtime, deserialize=kind == "deserialize")
    engine = _EnginePeer(runtime.address, b"unreachable-engine")
    try:
        await runtime.wait_for_engine(engine)
        old = client.add_request([7, 8], SamplingParams(num_tokens_to_generate=1))
        old_server_id = engine.receive_request()
        await _eventually(
            lambda: runtime.coordinator.request_id_to_rank.get(old_server_id) == engine.identity,
            "old request unowned",
        )
        engine.close()
        engine = None
        await runtime.wait_for_disconnect()
        params = SamplingParams(num_tokens_to_generate=1)
        if kind == "kv-handoff":
            future = client.add_request_with_kv_handoff([1, 2], params, {}, [])
        else:
            future = client.add_request([1, 2], params)
        resolved = []
        for pending in (old, future):
            pending.add_done_callback(resolved.append)
        replies = await asyncio.wait_for(asyncio.gather(old, future), timeout=5.0)
        for reply in replies:
            _assert_reply(reply)
        assert Counter(resolved) == Counter([old, future])
        assert client.socket.replies == {0: 1, 1: 1}
        _assert_no_requests(runtime.coordinator)
        runtime.assert_healthy()
        engine = _EnginePeer(runtime.address, b"replacement-engine")
        await runtime.wait_for_engine(engine)
        future = client.add_request([3, 4], SamplingParams(num_tokens_to_generate=1))
        server_request_id = engine.receive_request()
        engine.send_final(server_request_id)
        reply = await asyncio.wait_for(future, timeout=5.0)
        _assert_reply(reply, Status.COMPLETED)
        assert client.socket.replies == {0: 1, 1: 1, 2: 1}
        _assert_no_requests(runtime.coordinator)
    finally:
        client.stop()
        if engine is not None:
            engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["partial", "final"])
async def test_closed_client_delivery_does_not_disrupt_live_client(coordinator_runtime, delivery):
    runtime = coordinator_runtime
    engine = _EnginePeer(runtime.address, b"live-engine")
    closed_client, live_client = [_start_client(runtime) for _ in range(2)]
    closed = False
    try:
        await runtime.wait_for_engine(engine)
        closed_stream = closed_client.add_request_streaming(
            [1, 2], SamplingParams(num_tokens_to_generate=2)
        )
        closed_request_id = engine.receive_request()
        live_future = live_client.add_request([3, 4], SamplingParams(num_tokens_to_generate=1))
        resolved = []
        live_future.add_done_callback(resolved.append)
        live_request_id = engine.receive_request()
        assert closed_request_id != live_request_id
        closed_client.stop()
        closed = True
        await runtime.wait_for_disconnect()
        if delivery == "partial":
            engine.send_partial(closed_request_id)
        else:
            engine.send_final(closed_request_id)
        # Same-peer FIFO makes this live reply a barrier after the failed delivery.
        engine.send_final(live_request_id)
        reply = await asyncio.wait_for(live_future, timeout=5.0)
        assert reply["status"] == Status.COMPLETED.name
        runtime.assert_healthy()
        assert resolved == [live_future]
        assert live_client.socket.replies == {0: 1}

        if delivery == "partial":
            # A later final still owns cleanup even though its client is gone.
            engine.send_final(closed_request_id)
            await _eventually(
                lambda: not runtime.coordinator.request_id_to_client_id,
                "closed client request was not cleaned up",
            )
        _assert_no_requests(runtime.coordinator)
        assert runtime.coordinator._pending_counts.tolist() == [0]
        assert closed_stream.finished
    finally:
        if not closed:
            closed_client.stop()
        live_client.stop()
        engine.close()
