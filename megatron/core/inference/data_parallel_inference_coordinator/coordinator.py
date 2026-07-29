# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import errno
import faulthandler
import json
import logging
import signal
import socket
from collections import deque
from multiprocessing import Event
from multiprocessing.connection import Connection

import numpy as np
import torch

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

from .handlers import HANDLERS
from .state import CoordinatorState

try:
    import zmq

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

try:
    import msgpack

    HAVE_MSGPACK = True
except:
    HAVE_MSGPACK = False

# Register faulthandler to emit stack traces upon process kill.
faulthandler.enable()
faulthandler.register(signal.SIGTERM, all_threads=False, chain=True)
faulthandler.register(signal.SIGINT, all_threads=False, chain=True)


class DataParallelInferenceCoordinator:
    """
    Coordinates inference requests between clients and distributed model engines.

    This class acts as a central server. It uses a ZMQ ROUTER socket to manage
    communication flows between multiple clients and multiple data parallel ranks.

    The coordinator's main responsibilities are:
    1.  **Worker Registration**: It waits for a specified number of data parallel ranks
        (representing distributed model instances) to connect and register themselves.
    2.  **Client Connection**: It accepts connections from external clients, like
        `InferenceClient`, and performs a simple handshake.
    3.  **Request Forwarding**: It receives inference requests from clients, assigns a
        unique server-side request ID, tokenizes the prompt, and forwards the request
        to one of the available data parallel ranks using load-balanced (and,
        when prefix caching is enabled, prefix-affinity-aware) routing.
    4.  **Response Routing**: It receives completed results from
        the data parallel ranks and routes them back to the original client that made the
        request.
    5.  **Control Signal Broadcasting**: It relays control signals (e.g., PAUSE, STOP)
        from a client to all connected data parallel ranks.

    Message handling is split out into handlers.py: the event loop in start()
    dispatches each message to the handler registered for its header, so
    supporting a new message type requires no changes here.

    Attributes:
        router_socket (zmq.Socket): The central ZMQ ROUTER socket for all communication.
        data_parallel_size (int): The number of data parallel workers to expect.
        identities_of_data_parallel_ranks (deque): A deque holding the ZMQ
            identities of connected data parallel instances, used for request routing.
        request_id_to_client_id (dict): Maps server-side request IDs to the ZMQ
            identity of the client that initiated the request.
        request_id_to_client_request_id (dict): Maps server-side request IDs to the
            original request ID provided by the client.
        next_request_id (int): A counter for generating unique server-side request IDs.
    """

    def __init__(
        self,
        pipe_connection: Connection,
        data_parallel_size: int,
        tokenizer,
        max_requests,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ),
        prefix_caching_routing_alpha: float = 0.5,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
    ):
        """
        Initializes the inference coordinator.

        This sets up the ZMQ context and a ROUTER socket, binding it to the given
        port. It then enters a blocking loop to wait for all expected data parallel
        ranks to connect before proceeding.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            data_parallel_size (int): The number of data parallel instances that are
                expected to connect.
            tokenizer: The tokenizer to use for prompt tokenization and detokenization.
            inference_coordinator_port (Optional[int]): The TCP port number to bind the server to.
            prefix_caching_routing_alpha (float): Weight for prefix-aware routing score:
                score = alpha * match + (1 - alpha) * normalized_load.
            max_requests (int): Max concurrent requests per rank, used to
                compute normalized_load for prefix-aware scoring.
        """
        assert HAVE_ZMQ, (
            "please install the pyzmq library to use DataParallelInferenceCoordinator\n"
            "pip install pyzmq"
        )
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use DataParallelInferenceCoordinator\n"
            "pip install msgpack"
        )
        self.pipe_connection = pipe_connection
        self.data_parallel_size = data_parallel_size
        self.context = zmq.Context()

        # This is the central router socket
        # 1. data parallel ranks connect to this socket to register themselves
        # 2. Users connect to this socket and submit their requests. We transmit them to
        #    data parallel ranks in a round robin fashion
        # 3. data parallel ranks return completed requests to this socket. We route them back to
        #    the user that had submitted the request originally.

        # Get local IP.
        local_ip = hostname or socket.gethostname()

        self.router_socket = self.context.socket(zmq.ROUTER)
        # Raise error if the other side of the connection has dropped.
        self.router_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        is_bound = False
        if inference_coordinator_port is not None:
            try:
                self.router_socket.bind(f"tcp://{local_ip}:{inference_coordinator_port}")
                is_bound = True
            except zmq.error.ZMQError as e:
                if e.errno == errno.EADDRINUSE:
                    logging.warning(
                        f"Port {inference_coordinator_port} is already in use. "
                        "Binding to a random available port instead."
                    )
            except Exception:
                logging.warning(
                    f"Unknown error when binding to port {inference_coordinator_port}. "
                    "Attempting to bind to a random available port instead."
                )
        if not is_bound:
            self.router_socket.bind_to_random_port(f"tcp://{local_ip}")
        self.addr = self.router_socket.getsockopt_string(zmq.LAST_ENDPOINT)

        # Send the address to the parent process.
        self.pipe_connection.send(self.addr)
        self.pipe_connection.close()

        logging.info("Inference Coordinator: waiting for connections from data parallel ranks...")
        # First wait for all data parallel ranks to establish connections.
        self.identities_of_data_parallel_ranks = deque([])
        initial_rank_stream_ids = {}
        # time.sleep(5)  # Give data parallel ranks time to spawn and connect.
        for _ in range(data_parallel_size):
            identity, serialized_payload = self.router_socket.recv_multipart()
            payload = msgpack.unpackb(serialized_payload, raw=False)
            if (
                len(payload) != 2
                or Headers(payload[0]) != Headers.PREFIX_CACHE_REGISTER
                or not isinstance(payload[1], str)
                or not payload[1]
            ):
                raise ValueError("engine sent a malformed prefix cache registration")
            assert identity not in self.identities_of_data_parallel_ranks
            self.identities_of_data_parallel_ranks.append(identity)
            initial_rank_stream_ids[identity] = payload[1]
        logging.info("Inference Coordinator: Connected with data parallel ranks...")

        # In deterministic mode, sort identities for consistent scheduling order.
        if deterministic_mode:
            self.identities_of_data_parallel_ranks = deque(
                sorted(self.identities_of_data_parallel_ranks)
            )

        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}
        self.request_id_to_rank = {}  # Maps request_id → rank identity for pending count tracking

        self.next_request_id = 0
        self.tokenizer = tokenizer
        self.state = CoordinatorState.RUNNING

        # Prefix caching state for routing.
        self.block_size_tokens = block_size_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.generation_epoch = None
        self.prefix_caching_coordinator_policy = prefix_caching_coordinator_policy
        self.prefix_caching_routing_alpha = prefix_caching_routing_alpha
        self.max_requests = max_requests
        assert self.max_requests is not None and self.max_requests > 0

        # Schedule recording.
        self.schedule_output_path = schedule_output_path
        self.schedule_records = [] if schedule_output_path else None

        # Deterministic rank index mapping (sorted identity -> 0-based index).
        sorted_identities = sorted(self.identities_of_data_parallel_ranks)
        self.identity_to_rank_index = {
            identity: idx for idx, identity in enumerate(sorted_identities)
        }

        # Numpy arrays for vectorized scoring (indexed by rank index).
        n_ranks = len(sorted_identities)
        self._identities_list = list(sorted_identities)  # rank_index → identity
        self._pending_counts = np.zeros(n_ranks, dtype=np.int32)

        # Hash → {rank_idx: timestamp} dict for prefix cache affinity routing.
        # Each key is a block hash; each value maps rank indices to assignment
        # timestamps (positive int).  Missing entries are implicitly zero.
        self._hash_table: dict[int, dict[int, int]] = {}
        self._hash_assignment_counter = 0
        self._rank_hashes: dict[bytes, set[int]] = {
            identity: set() for identity in sorted_identities
        }
        self._rank_cache_epochs: dict[bytes, int | None] = {
            identity: None for identity in sorted_identities
        }
        self._rank_cache_sequences: dict[bytes, int | None] = {
            identity: None for identity in sorted_identities
        }
        self._rank_cache_stream_ids: dict[bytes, str] = {
            identity: initial_rank_stream_ids[identity] for identity in sorted_identities
        }
        self._rank_cache_resync_ids: dict[bytes, int] = {
            identity: 0 for identity in sorted_identities
        }
        self._rank_cache_needs_full_snapshot: dict[bytes, bool] = {
            identity: True for identity in sorted_identities
        }

        # Clients that have completed the CONNECT handshake.
        self.known_clients = set()

        # Header -> handler dispatch table, sourced from the handler registry.
        self._handlers = dict(HANDLERS)

        for identity in sorted_identities:
            self._request_prefix_cache_resync(identity)

    def get_least_loaded_data_parallel_rank(self):
        """
        Selects the data parallel rank with the fewest in-flight requests.

        Ties are broken by lowest rank index for deterministic behavior.

        Returns:
            bytes: The ZMQ identity of the least-loaded data parallel rank.
        """
        if not self._identities_list:
            raise RuntimeError("No engines connected")
        best_idx = int(np.argmin(self._pending_counts))
        return self._identities_list[best_idx]

    def _register_rank_identity(self, identity, stream_id):
        """Register a new rank identity in the scoring data structures.

        Called when a rank dynamically connects to a running coordinator
        (e.g. in tests that spawn the coordinator with data_parallel_size=0
        and let engines register after the fact).
        """
        if identity in self.identity_to_rank_index:
            return
        new_idx = len(self._identities_list)
        self.identity_to_rank_index[identity] = new_idx
        self._identities_list.append(identity)
        self._pending_counts = np.append(self._pending_counts, np.int32(0))
        self._rank_hashes[identity] = set()
        self._rank_cache_epochs[identity] = None
        self._rank_cache_sequences[identity] = None
        self._rank_cache_stream_ids[identity] = stream_id
        self._rank_cache_resync_ids[identity] = 0
        self._rank_cache_needs_full_snapshot[identity] = True
        logging.info(
            "Coordinator: registered engine %s as rank index %d (now %d engines)",
            identity,
            new_idx,
            len(self._identities_list),
        )

    def _remove_engine(self, identity):
        """Remove a disconnected engine from all routing bookkeeping.
        Called both during shutdown and when an engine becomes unreachable mid-operation
        (e.g. zmq.EHOSTUNREACH in _send_to_engine). The O(n) index-shifting and hash-table
        rebuild are acceptable because the number of connected engines is small; optimize
        only if dynamic registration/deregistration at high engine counts becomes a use case.
        """
        self.identities_of_data_parallel_ranks.remove(identity)
        idx = self.identity_to_rank_index.pop(identity, None)
        if idx is None:
            return
        self._rank_hashes.pop(identity, None)
        self._rank_cache_epochs.pop(identity, None)
        self._rank_cache_sequences.pop(identity, None)
        self._rank_cache_stream_ids.pop(identity, None)
        self._rank_cache_resync_ids.pop(identity, None)
        self._rank_cache_needs_full_snapshot.pop(identity, None)
        self._identities_list.pop(idx)
        self._pending_counts = np.delete(self._pending_counts, idx)
        # Shift indices for engines that came after the removed slot.
        for ident in self.identity_to_rank_index:
            if self.identity_to_rank_index[ident] > idx:
                self.identity_to_rank_index[ident] -= 1
        # Drop hash-table entries for the removed rank; shift indices above it.
        new_hash_table = {}
        for h, rank_ts in self._hash_table.items():
            # h is hash index
            # rank_ts is a dict mapping rank_idx → timestamp
            new_row = {}
            for r, ts in rank_ts.items():
                if r == idx:
                    # skip this rank as it is removed
                    continue
                new_r = r - 1 if r > idx else r
                new_row[new_r] = ts
            if new_row:
                new_hash_table[h] = new_row
        self._hash_table = new_hash_table
        logging.warning(
            "Coordinator: removed engine %s (now %d engines)",
            identity,
            len(self.identities_of_data_parallel_ranks),
        )

    def _send_to_engine(self, identity, payload):
        """Send payload to an engine, removing it from the pool if unreachable.

        Returns:
            True if the send succeeded, False if the engine was unreachable and removed.
        """
        try:
            self.router_socket.send_multipart([identity, payload])
            return True
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EHOSTUNREACH:
                self._remove_engine(identity)
                return False
            raise

    def _broadcast_to_engines(self, payload):
        """Send a deserialized payload to every connected data parallel rank."""
        serialized = msgpack.packb(payload, use_bin_type=True)
        for data_parallel_rank_id in list(self.identities_of_data_parallel_ranks):
            self._send_to_engine(data_parallel_rank_id, serialized)

    def compute_request_hashes(self, prompt, add_BOS: bool = False):
        """Compute block hashes for a prompt on CPU.

        Args:
            prompt: Either a string (to be tokenized) or a list of token IDs.
            add_BOS: Whether the engine will prepend the tokenizer's BOS token.
                This applies only to string prompts; pre-tokenized prompts are
                passed to the engine unchanged.

        Returns:
            List of integer block hashes, or empty list if prefix caching is disabled.
        """
        if not self.enable_prefix_caching or self.block_size_tokens is None:
            return []
        if isinstance(prompt, str):
            tokens = TextGenerationController.tokenize_prompt(
                self.tokenizer, prompt, add_BOS=add_BOS
            )
        else:
            tokens = list(prompt)
        token_tensor = torch.tensor(tokens, dtype=torch.int64)
        return compute_block_hashes_batched(
            token_tensor, self.block_size_tokens, namespace=self.generation_epoch
        )

    def get_best_data_parallel_rank(self, request_hashes):
        """Select the best DP rank based on prefix cache affinity and load.

        Uses a scoring function: score = alpha * match + (1 - alpha) * normalized_load
        where *match* is a policy-dependent affinity score in [0, 1] (binary for
        ``first_prefix_block``, normalized prefix depth for ``longest_prefix``)
        and normalized_load = free_slots / max_requests (higher means more free
        capacity).

        Args:
            request_hashes: List of block hashes for the request.

        Returns:
            bytes: The ZMQ identity of the selected data parallel rank.
        """
        if self.prefix_caching_coordinator_policy == PrefixCachingCoordinatorPolicy.LOAD_BALANCED:
            return self.get_least_loaded_data_parallel_rank()

        # Without prefix caching (or when the request has no hashes to match on)
        # fall back to load-balanced routing.
        if not self.enable_prefix_caching or not request_hashes:
            return self.get_least_loaded_data_parallel_rank()

        match, recency = self._match_vector(request_hashes)

        alpha = self.prefix_caching_routing_alpha

        # Vectorized score: alpha * match + (1-alpha) * free_capacity_fraction.
        free_slots = np.maximum(0, self.max_requests - self._pending_counts).astype(np.float64)
        scores = alpha * match + (1.0 - alpha) * (free_slots / self.max_requests)

        # Tiebreak: highest score, then highest recency, then lowest rank index.
        n_ranks = len(self._identities_list)
        order = np.lexsort((np.arange(n_ranks), -recency, -scores))
        best_idx = int(order[0])
        return self._identities_list[best_idx]

    def _update_rank_hashes(self, rank_identity, request_hashes):
        """Record newly observed hashes for a rank.

        Args:
            rank_identity: ZMQ identity of the target rank.
            request_hashes: List of block hashes confirmed by the engine.
        """
        rank_idx = self.identity_to_rank_index[rank_identity]
        self._hash_assignment_counter += 1
        ts = self._hash_assignment_counter
        for h in request_hashes:
            self._hash_table.setdefault(h, {})[rank_idx] = ts

    def _remove_rank_hashes(self, rank_identity, hashes) -> None:
        """Remove confirmed hashes for one rank from the reverse routing index."""
        rank_idx = self.identity_to_rank_index[rank_identity]
        for block_hash in hashes:
            rank_ts = self._hash_table.get(block_hash)
            if rank_ts is None:
                continue
            rank_ts.pop(rank_idx, None)
            if not rank_ts:
                del self._hash_table[block_hash]

    def _replace_rank_hashes(self, rank_identity, request_hashes):
        """Replace one rank's prefix-cache ownership with an engine snapshot.

        Retained hashes keep their existing timestamps. Newly materialized hashes
        receive a shared, monotonic timestamp, and hashes absent from the snapshot
        are removed. This makes routing reflect actual allocator state rather than
        requests that were merely assigned to the rank.

        Args:
            rank_identity: ZMQ identity of the reporting engine.
            request_hashes: Complete iterable of hashes currently discoverable.
        """
        confirmed_hashes = set(request_hashes)
        previous_hashes = self._rank_hashes[rank_identity]

        self._remove_rank_hashes(rank_identity, previous_hashes - confirmed_hashes)
        newly_confirmed = confirmed_hashes - previous_hashes
        if newly_confirmed:
            self._update_rank_hashes(rank_identity, newly_confirmed)
        self._rank_hashes[rank_identity] = confirmed_hashes

    def _reset_prefix_cache_protocol(self) -> None:
        """Clear confirmed ownership and require a full snapshot from every rank."""
        self._hash_table.clear()
        for rank_identity in self.identity_to_rank_index:
            self._rank_hashes[rank_identity] = set()
            self._rank_cache_epochs[rank_identity] = None
            self._rank_cache_sequences[rank_identity] = None
            self._rank_cache_needs_full_snapshot[rank_identity] = True

    def _reset_rank_prefix_cache_protocol(self, rank_identity) -> None:
        """Clear one rank's ownership and require its next update to be full."""
        self._remove_rank_hashes(rank_identity, self._rank_hashes.get(rank_identity, set()))
        self._rank_hashes[rank_identity] = set()
        self._rank_cache_epochs[rank_identity] = None
        self._rank_cache_sequences[rank_identity] = None
        self._rank_cache_needs_full_snapshot[rank_identity] = True

    def _request_prefix_cache_resync(self, rank_identity) -> None:
        """Discard one rank's ownership and request a new full snapshot."""
        self._reset_rank_prefix_cache_protocol(rank_identity)
        self._rank_cache_resync_ids[rank_identity] += 1
        payload = msgpack.packb(
            [
                Headers.PREFIX_CACHE_RESYNC.value,
                self._rank_cache_stream_ids[rank_identity],
                self._rank_cache_resync_ids[rank_identity],
            ],
            use_bin_type=True,
        )
        self._send_to_engine(rank_identity, payload)

    def _apply_prefix_cache_state(
        self,
        rank_identity,
        epoch: int | None,
        stream_id: str,
        resync_id: int,
        sequence: int,
        is_full_snapshot: bool,
        added_hashes,
        removed_hashes,
    ) -> bool:
        """Apply an ordered, confirmed prefix-cache state update from one rank.

        A newly registered rank, or any rank after a generation-epoch change,
        must start with a full snapshot. A full snapshot carries the complete
        state in ``added_hashes`` and an empty ``removed_hashes`` list. Later
        messages may carry set deltas. The stream identifies one engine
        incarnation, the resync identifies one recovery cycle within that
        incarnation, and the sequence is contiguous within a resync cycle.

        Returns:
            True when the update was accepted and applied, otherwise False.
        """
        if rank_identity not in self.identity_to_rank_index:
            logging.warning("Coordinator: ignoring prefix cache state from unknown engine.")
            return False
        if epoch != self.generation_epoch:
            logging.warning(
                "Coordinator: ignoring prefix cache state from engine %s at epoch %r "
                "(current epoch %r).",
                rank_identity,
                epoch,
                self.generation_epoch,
            )
            return False
        if (
            not isinstance(stream_id, str)
            or not stream_id
            or not isinstance(resync_id, int)
            or isinstance(resync_id, bool)
            or resync_id <= 0
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence <= 0
            or not isinstance(is_full_snapshot, bool)
        ):
            logging.warning("Coordinator: ignoring malformed prefix cache state metadata.")
            return False

        expected_stream_id = self._rank_cache_stream_ids[rank_identity]
        expected_resync_id = self._rank_cache_resync_ids[rank_identity]
        if stream_id != expected_stream_id or resync_id != expected_resync_id:
            logging.warning(
                "Coordinator: ignoring stale prefix cache state from engine %s "
                "(stream/resync %r/%r, expected %r/%r).",
                rank_identity,
                stream_id,
                resync_id,
                expected_stream_id,
                expected_resync_id,
            )
            return False

        last_sequence = self._rank_cache_sequences[rank_identity]
        if last_sequence is not None and sequence <= last_sequence:
            logging.warning(
                "Coordinator: ignoring stale prefix cache state from engine %s "
                "(sequence %d <= %d).",
                rank_identity,
                sequence,
                last_sequence,
            )
            return False
        needs_full = self._rank_cache_needs_full_snapshot[rank_identity]
        if needs_full and sequence != 1:
            logging.warning(
                "Coordinator: engine %s did not restart its prefix cache sequence "
                "for a full resync.",
                rank_identity,
            )
            self._request_prefix_cache_resync(rank_identity)
            return False
        if not needs_full:
            assert last_sequence is not None
        if not needs_full and sequence != last_sequence + 1:
            logging.warning(
                "Coordinator: detected a prefix cache sequence gap from engine %s "
                "(sequence %d after %d); requesting a full snapshot.",
                rank_identity,
                sequence,
                last_sequence,
            )
            self._request_prefix_cache_resync(rank_identity)
            return False

        try:
            added = set(added_hashes)
            removed = set(removed_hashes)
        except TypeError:
            logging.warning("Coordinator: ignoring malformed prefix cache hash collections.")
            return False
        if added & removed:
            logging.warning("Coordinator: ignoring prefix cache delta with overlapping sets.")
            return False

        if is_full_snapshot:
            if removed:
                logging.warning(
                    "Coordinator: ignoring full prefix cache snapshot with removed hashes."
                )
                return False
            self._replace_rank_hashes(rank_identity, added)
        else:
            if needs_full:
                logging.warning(
                    "Coordinator: ignoring prefix cache delta from engine %s before "
                    "its full snapshot.",
                    rank_identity,
                )
                return False
            current = self._rank_hashes[rank_identity]
            self._remove_rank_hashes(rank_identity, current & removed)
            newly_confirmed = added - current
            if newly_confirmed:
                self._update_rank_hashes(rank_identity, newly_confirmed)
            self._rank_hashes[rank_identity] = (current - removed) | added

        self._rank_cache_epochs[rank_identity] = epoch
        self._rank_cache_sequences[rank_identity] = sequence
        self._rank_cache_needs_full_snapshot[rank_identity] = False
        return True

    def _match_vector(self, hashes):
        """Return ``(match, recency)`` vectors of shape ``(n_ranks,)``.

        *match* is each rank's deepest cached prefix depth normalized by the
        request length. *recency* is that matched block's raw assignment
        timestamp (0 for non-matching ranks).

        For ``FIRST_PREFIX_BLOCK`` the caller already truncates *hashes* to a
        single element, so the same logic yields a binary 0/1 match score.
        """
        n_ranks = len(self._identities_list)
        n = len(hashes)
        match = np.zeros(n_ranks, dtype=np.float64)
        recency = np.zeros(n_ranks, dtype=np.float64)
        if n == 0:
            return match, recency

        unmatched = np.ones(n_ranks, dtype=bool)
        for i in range(n - 1, -1, -1):
            row = self._hash_table.get(hashes[i])
            if row is None:
                continue
            rank_idxs = np.fromiter(row.keys(), dtype=np.intp)
            rank_idxs = rank_idxs[unmatched[rank_idxs]]
            if rank_idxs.size == 0:
                continue
            match[rank_idxs] = (i + 1.0) / n
            recency[rank_idxs] = np.fromiter(
                (row[int(rank_idx)] for rank_idx in rank_idxs), dtype=np.float64
            )
            unmatched[rank_idxs] = False
            if not unmatched.any():
                break
        return match, recency

    def start(self):
        """
        Starts the main event loop for the coordinator.

        This method runs an infinite loop, continuously listening for incoming
        messages on the ZMQ ROUTER socket. It reads the message header and
        dispatches to the handler registered for it (see handlers.py).
        A handler that returns a truthy value stops the loop.
        """
        # Todo [Siddharth]: Make this more robust to handle invalid messages.
        while True:
            sender_identity, serialized_payload = self.router_socket.recv_multipart()

            deserialized_payload = msgpack.unpackb(serialized_payload, raw=False)
            header = Headers(deserialized_payload[0])

            handler = self._handlers.get(header)
            if handler is None:
                raise UnknownHeaderError(header)
            if handler(self, sender_identity, deserialized_payload):
                break

    def _handle_rank_registration(self, sender_identity, stream_id):
        """Register a data parallel rank that connected to a running coordinator."""
        if sender_identity not in self.identities_of_data_parallel_ranks:
            self.identities_of_data_parallel_ranks.append(sender_identity)
            self._register_rank_identity(sender_identity, stream_id)
        else:
            self._reset_rank_prefix_cache_protocol(sender_identity)
            if stream_id != self._rank_cache_stream_ids[sender_identity]:
                self._rank_cache_stream_ids[sender_identity] = stream_id
                self._rank_cache_resync_ids[sender_identity] = 0

        # A late or restarted rank must enter the coordinator's current namespace
        # before the event loop can route another request to it.
        if self.generation_epoch is not None:
            payload = msgpack.packb(
                [Headers.SET_GENERATION_EPOCH.value, self.generation_epoch], use_bin_type=True
            )
            self._send_to_engine(sender_identity, payload)
        self._request_prefix_cache_resync(sender_identity)

    def detokenize(self, finished_request):
        """
        Detokenizes the generated tokens in the finished request.

        This method uses the coordinator's tokenizer to convert the list of
        generated token IDs back into human-readable text.

        Args:
            finished_request (dict): The serialized merged request containing the
                generated tokens to be detokenized. It is modified in place.
        """
        detokenize_stop_sequence = (finished_request.get("sampling_params", {}) or {}).get(
            "detokenize_stop_sequence", False
        )
        finished_request["generated_text"] = TextGenerationController.detokenize(
            self.tokenizer,
            finished_request["generated_tokens"],
            remove_EOD=not detokenize_stop_sequence,
        )

    @classmethod
    def entrypoint(
        cls,
        pipe_connection: Connection,
        ready_event: Event,
        data_parallel_size: int,
        tokenizer,
        max_requests,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ),
        prefix_caching_routing_alpha: float = 0.5,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
    ):
        """
        Class method to instantiate and run the coordinator, for use in a separate process.

        This method initializes the coordinator, signals a `ready_event` to indicate
        that it is fully initialized and listening, and then starts the main event loop.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            ready_event (Event): A threading or multiprocessing event object that is set()
                once the coordinator is ready to accept connections.
            inference_coordinator_port (int): The port to bind to.
            data_parallel_size (int): The number of expected data parallel instances.
            deterministic_mode (bool): Whether to enable deterministic scheduling.
            block_size_tokens (Optional[int]): Token block size for prefix caching hashing.
            enable_prefix_caching (bool): Whether prefix caching is enabled.
            prefix_caching_coordinator_policy (PrefixCachingCoordinatorPolicy): Routing policy.
            schedule_output_path (Optional[str]): Path to write scheduling decisions JSON.
            prefix_caching_routing_alpha (float): Weight for prefix-aware routing score.
            max_requests (int): Max concurrent requests per rank.
        """
        coordinator = cls(
            pipe_connection,
            data_parallel_size,
            tokenizer,
            max_requests,
            inference_coordinator_port,
            deterministic_mode=deterministic_mode,
            block_size_tokens=block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_coordinator_policy=prefix_caching_coordinator_policy,
            prefix_caching_routing_alpha=prefix_caching_routing_alpha,
            schedule_output_path=schedule_output_path,
            hostname=hostname,
        )
        ready_event.set()
        try:
            coordinator.start()
        except KeyboardInterrupt:
            logging.info("Coordinator process interrupted. Exiting...")
        coordinator.stop()
        logging.info("Inference Coordinator: shut down successfully.")

    def stop(self):
        """
        Stops the inference coordinator, performing any necessary cleanup operations.
        """
        if self.schedule_output_path and self.schedule_records:
            schedule_data = {
                "policy": self.prefix_caching_coordinator_policy.value,
                "data_parallel_size": self.data_parallel_size,
                "num_requests": len(self.schedule_records),
                "records": self.schedule_records,
            }
            with open(self.schedule_output_path, "w") as f:
                json.dump(schedule_data, f, indent=2)
        self.router_socket.close()
        self.context.term()
