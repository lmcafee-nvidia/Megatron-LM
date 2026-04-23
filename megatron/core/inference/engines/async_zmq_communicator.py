# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import os
import socket
import struct
import uuid

import torch.distributed as dist

try:
    import zmq

    HAVE_ZMQ = True
except ImportError:
    from unittest.mock import MagicMock

    zmq = MagicMock()
    HAVE_ZMQ = False


logger = logging.getLogger(__name__)


def _pick_transport(process_group: dist.ProcessGroup) -> str:
    """Return "ipc" when all ranks in the group share a host, else "tcp".

    Overridden by env MCORE_EP_ZMQ_TRANSPORT in {tcp, ipc, auto}.
    """
    override = os.environ.get("MCORE_EP_ZMQ_TRANSPORT", "auto").lower()
    if override == "tcp" or override == "ipc":
        return override
    if override not in ("auto", ""):
        logger.warning(
            "Unknown MCORE_EP_ZMQ_TRANSPORT=%r, falling back to auto", override
        )

    if process_group is None:
        world_size = dist.get_world_size()
    else:
        world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        # Single-rank group never talks over the socket; transport is moot.
        return "ipc"

    my_hostname = socket.gethostname()
    hostnames = [None] * world_size
    dist.all_gather_object(hostnames, my_hostname, group=process_group)
    return "ipc" if all(h == my_hostname for h in hostnames) else "tcp"


class AsyncZMQCommunicator:
    """
    An asyncio-friendly communicator abstraction using ZMQ.
    Can be used to implement collective operations like all-reduce,
    and bcast which are asyncio friendly on top of ZMQ sockets.
    Only to be used with small amounts of data (e.g., 1 integer)
    on the CPU.
    """

    def __init__(
        self,
        zmq_context: zmq.Context,
        process_group: dist.ProcessGroup,
        hostname: str | None = None,
    ):
        """
        Constructor for AsyncZMQCommunicator. Sets up ZMQ sockets
        for communication among ranks in the given process group.
        Args:
            zmq_context (zmq.Context): ZMQ context to create sockets.
            process_group (dist.ProcessGroup): Process group for communication.
            hostname (str | None): Hostname or IP address to use for ZMQ socket binding.
                If None, defaults to socket.gethostname().
        """
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.is_leader = self.rank == 0
        # Get the global rank of the leader (first rank in the process group)
        src_rank = dist.get_process_group_ranks(process_group)[0]

        transport = _pick_transport(process_group)
        # Track IPC file paths so we can unlink them on close.
        self._ipc_paths: list[str] = []

        if self.is_leader:
            gather_socket_addr, bcast_socket_addr = self._leader_bind(
                zmq_context, transport, hostname
            )

            # Share the socket addresses with all peers
            dist.broadcast_object_list(
                [gather_socket_addr, bcast_socket_addr], src=src_rank, group=process_group
            )

        else:
            bcast_output = [None, None]
            dist.broadcast_object_list(bcast_output, src=src_rank, group=process_group)
            gather_socket_addr, bcast_socket_addr = bcast_output
            self.gather_sock = zmq_context.socket(zmq.PUSH)
            self.gather_sock.connect(gather_socket_addr)
            self.bcast_sock = zmq_context.socket(zmq.SUB)
            self.bcast_sock.connect(bcast_socket_addr)
            self.bcast_sock.setsockopt_string(zmq.SUBSCRIBE, "")

    def _leader_bind(
        self, zmq_context: "zmq.Context", transport: str, hostname: str | None
    ) -> tuple[str, str]:
        """Bind the leader's gather/bcast sockets; return their endpoint URIs.

        Tries ipc:// when transport=="ipc"; falls back to tcp:// on bind failure.
        """
        self.gather_sock = zmq_context.socket(zmq.PULL)
        self.bcast_sock = zmq_context.socket(zmq.PUB)

        if transport == "ipc":
            suffix = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
            gather_path = f"/tmp/mcore-ep-{suffix}-gather.ipc"
            bcast_path = f"/tmp/mcore-ep-{suffix}-bcast.ipc"
            try:
                self.gather_sock.bind(f"ipc://{gather_path}")
                self.bcast_sock.bind(f"ipc://{bcast_path}")
                self._ipc_paths = [gather_path, bcast_path]
                return (
                    self.gather_sock.getsockopt_string(zmq.LAST_ENDPOINT),
                    self.bcast_sock.getsockopt_string(zmq.LAST_ENDPOINT),
                )
            except zmq.ZMQError as e:
                logger.warning(
                    "IPC bind failed (%s); falling back to TCP transport", e
                )
                # Fall through to TCP below. Recreate sockets since the failed
                # bind left them in an inconsistent state.
                self.gather_sock.close(linger=0)
                self.bcast_sock.close(linger=0)
                self.gather_sock = zmq_context.socket(zmq.PULL)
                self.bcast_sock = zmq_context.socket(zmq.PUB)
                self._ipc_paths = []

        # TCP bind (default / fallback).
        local_ip = hostname or socket.gethostname()
        self.gather_sock.bind_to_random_port(f"tcp://{local_ip}")
        self.bcast_sock.bind_to_random_port(f"tcp://{local_ip}")
        return (
            self.gather_sock.getsockopt_string(zmq.LAST_ENDPOINT),
            self.bcast_sock.getsockopt_string(zmq.LAST_ENDPOINT),
        )

    async def all_reduce_max(self, *local_vals: int, async_op=True) -> int | tuple[int, ...]:
        """Element-wise all-reduce max of one or more integers.

        Packs all values into a single message so the communication cost
        is independent of the number of values.

        Returns a single int when called with one argument, otherwise a tuple.
        """
        n = len(local_vals)
        if n == 0:
            raise ValueError("all_reduce_max requires at least one value")

        if self.world_size <= 1:
            return local_vals[0] if n == 1 else local_vals

        fmt = f'!{n}i'
        payload = struct.pack(fmt, *local_vals)

        if self.is_leader:
            rows = [local_vals]

            while len(rows) < self.world_size:
                try:
                    if async_op:
                        msg = self.gather_sock.recv(flags=zmq.NOBLOCK)
                    else:
                        msg = self.gather_sock.recv()
                    try:
                        rows.append(struct.unpack(fmt, msg))
                    except struct.error:
                        # Diagnostic: size mismatch means a peer sent a message
                        # from a different collective phase than the leader expected.
                        logger.error(
                            "all_reduce_max leader size mismatch: "
                            "my_rank=%d, world_size=%d, fmt=%r (expect %d bytes), "
                            "got %d bytes, rows_collected=%d/%d",
                            self.rank, self.world_size, fmt, struct.calcsize(fmt),
                            len(msg), len(rows), self.world_size,
                        )
                        raise
                except zmq.Again:
                    await asyncio.sleep(0.001)

            maxes = tuple(max(row[i] for row in rows) for i in range(n))
            self.bcast_sock.send(struct.pack(fmt, *maxes))
            if not async_op:
                await asyncio.sleep(
                    0
                )  # Yield control once to ensure that other coroutines can run.
                # This might be needed for colocated RL.
            return maxes[0] if n == 1 else maxes

        else:
            self.gather_sock.send(payload)

            while True:
                try:
                    if async_op:
                        msg = self.bcast_sock.recv(flags=zmq.NOBLOCK)
                    else:
                        msg = self.bcast_sock.recv()
                    result = struct.unpack(fmt, msg)
                    if not async_op:
                        await asyncio.sleep(
                            0
                        )  # Yield control once to ensure that other coroutines can run.
                        # This might be needed for colocated RL.
                    return result[0] if n == 1 else result
                except zmq.Again:
                    await asyncio.sleep(0.001)

    def sync_all_reduce_max(self, *local_vals: int) -> int | tuple[int, ...]:
        """Synchronous (non-asyncio) variant of all_reduce_max.

        Uses blocking ZMQ sends/recvs so it can be called from synchronous
        call sites that need a CPU-only MAX reduction across the process
        group. Intended for tiny payloads (e.g. a few integers) that would
        otherwise force a NCCL AllReduce kernel on the compute stream.

        Note: when called from inside a running asyncio event loop, the
        blocking recv will pause other coroutines on this rank until all
        peers respond. This is acceptable here because every rank reaches
        the call simultaneously and the message size is trivial.

        Returns a single int when called with one argument, otherwise a tuple.
        """
        n = len(local_vals)
        if n == 0:
            raise ValueError("sync_all_reduce_max requires at least one value")

        if self.world_size <= 1:
            return local_vals[0] if n == 1 else local_vals

        fmt = f'!{n}i'
        payload = struct.pack(fmt, *local_vals)

        if self.is_leader:
            rows = [local_vals]
            while len(rows) < self.world_size:
                msg = self.gather_sock.recv()
                rows.append(struct.unpack(fmt, msg))
            maxes = tuple(max(row[i] for row in rows) for i in range(n))
            self.bcast_sock.send(struct.pack(fmt, *maxes))
            return maxes[0] if n == 1 else maxes
        else:
            self.gather_sock.send(payload)
            msg = self.bcast_sock.recv()
            result = struct.unpack(fmt, msg)
            return result[0] if n == 1 else result

    def close(self):
        """
        Close the ZMQ sockets.
        """
        # linger=0: discard unsent messages immediately on close rather than blocking until sent.
        # The ZMQ default is to not allow `close` until all messages have been successfully sent.
        self.gather_sock.close(linger=0)
        self.bcast_sock.close(linger=0)
        for path in self._ipc_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._ipc_paths = []
