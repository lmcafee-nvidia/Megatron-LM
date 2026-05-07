# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import socket
import struct

import torch.distributed as dist

try:
    import zmq

    HAVE_ZMQ = True
except ImportError:
    from unittest.mock import MagicMock

    zmq = MagicMock()
    HAVE_ZMQ = False


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

        if self.is_leader:
            local_ip = hostname or socket.gethostname()
            self.gather_sock = zmq_context.socket(zmq.PULL)
            self.gather_sock.bind_to_random_port(f"tcp://{local_ip}")
            gather_socket_addr = self.gather_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            # PUB/SUB can drop a collective result if a peer is not already
            # receiving when the leader broadcasts. Use one reliable PUSH/PULL
            # result channel per non-leader rank instead.
            self.result_socks = []
            result_socket_addrs = [None] * self.world_size
            for peer_rank in range(1, self.world_size):
                result_sock = zmq_context.socket(zmq.PUSH)
                result_sock.bind_to_random_port(f"tcp://{local_ip}")
                result_socket_addrs[peer_rank] = result_sock.getsockopt_string(
                    zmq.LAST_ENDPOINT
                )
                self.result_socks.append(result_sock)

            # Share the socket addresses with all peers.
            dist.broadcast_object_list(
                [gather_socket_addr, result_socket_addrs], src=src_rank, group=process_group
            )

        else:
            bcast_output = [None, None]
            dist.broadcast_object_list(bcast_output, src=src_rank, group=process_group)
            gather_socket_addr, result_socket_addrs = bcast_output
            self.gather_sock = zmq_context.socket(zmq.PUSH)
            self.gather_sock.connect(gather_socket_addr)
            self.result_sock = zmq_context.socket(zmq.PULL)
            self.result_sock.connect(result_socket_addrs[self.rank])

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
                    rows.append(struct.unpack(fmt, msg))
                except zmq.Again:
                    await asyncio.sleep(0.001)

            maxes = tuple(max(row[i] for row in rows) for i in range(n))
            result_payload = struct.pack(fmt, *maxes)
            for result_sock in self.result_socks:
                result_sock.send(result_payload)
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
                        msg = self.result_sock.recv(flags=zmq.NOBLOCK)
                    else:
                        msg = self.result_sock.recv()
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
            result_payload = struct.pack(fmt, *maxes)
            for result_sock in self.result_socks:
                result_sock.send(result_payload)
            return maxes[0] if n == 1 else maxes
        else:
            self.gather_sock.send(payload)
            msg = self.result_sock.recv()
            result = struct.unpack(fmt, msg)
            return result[0] if n == 1 else result

    def close(self):
        """
        Close the ZMQ sockets.
        """
        # linger=0: discard unsent messages immediately on close rather than blocking until sent.
        # The ZMQ default is to not allow `close` until all messages have been successfully sent.
        self.gather_sock.close(linger=0)
        if self.is_leader:
            for result_sock in self.result_socks:
                result_sock.close(linger=0)
        else:
            self.result_sock.close(linger=0)
