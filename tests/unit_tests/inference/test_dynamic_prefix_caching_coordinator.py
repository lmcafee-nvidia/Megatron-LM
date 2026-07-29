# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for prefix-cache-aware coordinator routing.

Validates that the DataParallelInferenceCoordinator correctly computes block
hashes from prompts, routes requests to the DP rank with the longest consecutive
prefix match, and maintains confirmed per-rank cache state and timestamps.
"""

from unittest.mock import MagicMock

import msgpack
import torch

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.data_parallel_inference_coordinator.handlers import (
    handle_control_signal,
    handle_prefix_cache_register,
    handle_prefix_cache_state,
    handle_submit_request,
)
from megatron.core.inference.data_parallel_inference_coordinator.state import CoordinatorState
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.inference.sampling_params import SamplingParams

# ============================================================================
# Shared fixtures and helpers
# ============================================================================

BLOCK_SIZE = 4


class DummyTokenizer:
    """Dummy tokenizer that splits on whitespace and converts to ints."""

    def __init__(self, vocab_size: int = 10, bos: int | None = None, eod: int = 0, pad: int = 0):
        self.vocab_size = vocab_size
        self.bos = bos
        self.eod = eod
        self.pad = pad

    def tokenize(self, prompt):
        if isinstance(prompt, str):
            return [int(tok) % self.vocab_size for tok in prompt.strip().split()]
        return list(prompt)

    def detokenize(self, tokens, skip_special_tokens: bool = False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if skip_special_tokens and self.eod in tokens:
            tokens = [tok for tok in tokens if tok != self.eod]
        return " ".join(str(tok) for tok in tokens)


def make_coordinator_direct(
    data_parallel_size=2,
    block_size_tokens=BLOCK_SIZE,
    enable_prefix_caching=True,
    deterministic_mode=True,
    prefix_caching_routing_alpha=0.5,
    max_requests=10,
):
    """Create a coordinator with mock ZMQ, for unit testing routing logic.

    Thin wrapper around the shared helper in coordinator_test_utils.py that
    supplies a DummyTokenizer and this module's BLOCK_SIZE default.
    """
    from tests.unit_tests.inference.coordinator_test_utils import (
        make_coordinator_direct as _make_coordinator,
    )

    return _make_coordinator(
        data_parallel_size=data_parallel_size,
        block_size_tokens=block_size_tokens,
        enable_prefix_caching=enable_prefix_caching,
        deterministic_mode=deterministic_mode,
        prefix_caching_routing_alpha=prefix_caching_routing_alpha,
        max_requests=max_requests,
        tokenizer=DummyTokenizer(),
    )


def _apply_cache_state(
    coordinator,
    rank,
    epoch,
    sequence,
    is_full_snapshot,
    added_hashes,
    removed_hashes,
    *,
    stream_id=None,
    resync_id=None,
):
    """Apply state using the rank's current stream and resync by default."""
    if stream_id is None:
        stream_id = coordinator._rank_cache_stream_ids[rank]
    if resync_id is None:
        resync_id = coordinator._rank_cache_resync_ids[rank]
    return coordinator._apply_prefix_cache_state(
        rank, epoch, stream_id, resync_id, sequence, is_full_snapshot, added_hashes, removed_hashes
    )


# ============================================================================
# Test classes
# ============================================================================


class TestCoordinatorHashComputation:
    """Stress coordinator/engine hash parity across input and epoch boundaries."""

    def test_hash_chain_stresses_boundaries_mutation_bos_and_epoch(self):
        """Long chains remain exact while mutations, BOS, and refits change ownership."""
        coordinator = make_coordinator_direct()
        coordinator.tokenizer = DummyTokenizer(vocab_size=10_000)
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        tokens = list(range(1, BLOCK_SIZE * 120 + 1))
        old_hashes = coordinator.compute_request_hashes(tokens)
        expected = compute_block_hashes_batched(torch.tensor(tokens), BLOCK_SIZE, namespace=None)
        assert old_hashes == expected
        assert old_hashes == coordinator.compute_request_hashes(
            " ".join(str(token) for token in tokens)
        )
        assert len(old_hashes) == 120

        for token_count, expected_count in (
            (BLOCK_SIZE - 1, 0),
            (BLOCK_SIZE, 1),
            (BLOCK_SIZE + 1, 1),
        ):
            assert len(coordinator.compute_request_hashes(tokens[:token_count])) == expected_count

        mutated = tokens.copy()
        mutation_block = 60
        mutated[mutation_block * BLOCK_SIZE] += 10_000
        mutated_hashes = coordinator.compute_request_hashes(mutated)
        assert mutated_hashes[:mutation_block] == old_hashes[:mutation_block]
        assert all(
            current != original
            for current, original in zip(
                mutated_hashes[mutation_block:], old_hashes[mutation_block:]
            )
        )

        coordinator.tokenizer = DummyTokenizer(vocab_size=1024, bos=1023)
        bos_prompt = "1 2 3 4 5 6 7"
        bos_hashes = coordinator.compute_request_hashes(bos_prompt, add_BOS=True)
        assert bos_hashes == compute_block_hashes_batched(
            torch.tensor([1023, 1, 2, 3, 4, 5, 6, 7]), BLOCK_SIZE, namespace=None
        )
        assert bos_hashes != coordinator.compute_request_hashes(bos_prompt, add_BOS=False)

        assert _apply_cache_state(coordinator, rank_0, None, 1, True, old_hashes, [])
        coordinator.generation_epoch = 17
        coordinator._reset_prefix_cache_protocol()
        new_hashes = coordinator.compute_request_hashes(tokens)
        assert new_hashes != old_hashes
        assert coordinator._hash_table == {}
        assert coordinator.get_best_data_parallel_rank(new_hashes) == min(
            coordinator.identities_of_data_parallel_ranks
        )
        assert _apply_cache_state(coordinator, rank_0, 17, 1, True, new_hashes, [])
        assert coordinator.get_best_data_parallel_rank(new_hashes) == rank_0


class TestCoordinatorCacheState:
    """Stress confirmed rank cache ownership through replacement and churn."""

    def test_delta_protocol_stresses_churn_and_rejects_stale_or_wrong_epoch(self):
        """Large confirmed-state churn stays exact across ranks and stale updates."""
        coordinator = make_coordinator_direct()
        rank_0, rank_1 = coordinator.identities_of_data_parallel_ranks
        idx_0 = coordinator.identity_to_rank_index[rank_0]
        idx_1 = coordinator.identity_to_rank_index[rank_1]
        initial = set(range(1, 2049))
        shared = set(range(1025, 1537))

        handle_prefix_cache_state(
            coordinator,
            rank_0,
            [
                Headers.PREFIX_CACHE_STATE.value,
                None,
                coordinator._rank_cache_stream_ids[rank_0],
                coordinator._rank_cache_resync_ids[rank_0],
                1,
                True,
                sorted(initial),
                [],
            ],
        )
        assert coordinator._rank_hashes[rank_0] == initial
        assert _apply_cache_state(coordinator, rank_1, None, 1, True, shared, [])

        removed = set(range(1, 1025))
        added = set(range(3001, 4025))
        handle_prefix_cache_state(
            coordinator,
            rank_0,
            [
                Headers.PREFIX_CACHE_STATE.value,
                None,
                coordinator._rank_cache_stream_ids[rank_0],
                coordinator._rank_cache_resync_ids[rank_0],
                2,
                False,
                sorted(added),
                sorted(removed),
            ],
        )
        expected_rank_0 = (initial - removed) | added
        assert coordinator._rank_hashes[rank_0] == expected_rank_0
        assert all(idx_0 not in coordinator._hash_table.get(h, {}) for h in removed)
        assert all(coordinator._hash_table[h][idx_0] > 0 for h in added)
        assert all(coordinator._hash_table[h][idx_1] > 0 for h in shared)

        snapshot = {
            block_hash: dict(rank_timestamps)
            for block_hash, rank_timestamps in coordinator._hash_table.items()
        }
        assert not _apply_cache_state(coordinator, rank_0, None, 2, False, [9999], [])
        assert not _apply_cache_state(coordinator, rank_0, 7, 3, False, [9999], [])
        assert coordinator._hash_table == snapshot

    def test_delta_requires_full_snapshot_after_epoch_change(self):
        """An epoch reset rejects deltas until a full current-epoch snapshot."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        assert _apply_cache_state(coordinator, rank_0, None, 1, True, [10, 20], [])

        coordinator.generation_epoch = 9
        coordinator._reset_prefix_cache_protocol()
        assert not _apply_cache_state(coordinator, rank_0, 9, 1, False, [30], [10])
        assert coordinator._hash_table == {}
        assert coordinator._rank_cache_needs_full_snapshot[rank_0]

        assert _apply_cache_state(coordinator, rank_0, 9, 1, True, [30, 40], [])
        assert coordinator._rank_hashes[rank_0] == {30, 40}
        assert coordinator._rank_cache_epochs[rank_0] == 9
        assert coordinator._rank_cache_sequences[rank_0] == 1

    def test_sequence_gap_clears_ownership_and_requires_explicit_resync(self):
        """A missing delta cannot leave partially trusted ownership routable."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        stream_id = coordinator._rank_cache_stream_ids[rank_0]
        old_resync_id = coordinator._rank_cache_resync_ids[rank_0]
        assert _apply_cache_state(coordinator, rank_0, None, 1, True, [10, 20], [])
        coordinator.router_socket.reset_mock()

        assert not _apply_cache_state(coordinator, rank_0, None, 3, False, [30], [])
        assert coordinator._rank_hashes[rank_0] == set()
        assert coordinator._rank_cache_needs_full_snapshot[rank_0]
        assert coordinator._rank_cache_resync_ids[rank_0] == old_resync_id + 1
        destination, packed_resync = coordinator.router_socket.send_multipart.call_args.args[0]
        assert destination == rank_0
        assert msgpack.unpackb(packed_resync, raw=False) == [
            Headers.PREFIX_CACHE_RESYNC.value,
            stream_id,
            old_resync_id + 1,
        ]

        assert not _apply_cache_state(
            coordinator, rank_0, None, 2, False, [30], [], resync_id=old_resync_id
        )
        assert _apply_cache_state(coordinator, rank_0, None, 1, True, [20, 30], [])
        assert coordinator._rank_hashes[rank_0] == {20, 30}


class TestCoordinatorCacheProtocolLifecycle:
    """Exercise epoch, registration, and request-tokenization protocol boundaries."""

    def test_generation_epoch_change_clears_once_and_duplicate_is_idempotent(self):
        """Repeated epoch delivery neither clears current state nor rebroadcasts."""
        coordinator = make_coordinator_direct()
        rank_0 = coordinator.identities_of_data_parallel_ranks[0]
        client = b"client"
        coordinator.known_clients = {client}
        coordinator.state = CoordinatorState.RUNNING
        coordinator._broadcast_to_engines = MagicMock()
        assert _apply_cache_state(coordinator, rank_0, None, 1, True, [10, 20], [])

        epoch_payload = [Headers.SET_GENERATION_EPOCH.value, 7]
        handle_control_signal(coordinator, client, epoch_payload)

        assert coordinator.generation_epoch == 7
        assert coordinator._hash_table == {}
        assert coordinator._rank_hashes[rank_0] == set()
        assert coordinator._rank_cache_needs_full_snapshot[rank_0]
        coordinator._broadcast_to_engines.assert_called_once_with(epoch_payload)

        assert _apply_cache_state(coordinator, rank_0, 7, 1, True, [30], [])
        handle_control_signal(coordinator, client, epoch_payload)

        assert coordinator._rank_hashes[rank_0] == {30}
        assert coordinator._rank_cache_sequences[rank_0] == 1
        coordinator._broadcast_to_engines.assert_called_once_with(epoch_payload)

    def test_rank_reregistration_clears_only_that_rank_and_sends_current_epoch(self):
        """A restarted rank cannot route from stale state before its new full snapshot."""
        coordinator = make_coordinator_direct()
        rank_0, rank_1 = coordinator.identities_of_data_parallel_ranks
        coordinator.generation_epoch = 12
        coordinator.router_socket = MagicMock()
        old_stream = coordinator._rank_cache_stream_ids[rank_0]
        old_resync = coordinator._rank_cache_resync_ids[rank_0]
        assert _apply_cache_state(coordinator, rank_0, 12, 1, True, [10, 20], [])
        assert _apply_cache_state(coordinator, rank_1, 12, 1, True, [20, 30], [])

        handle_prefix_cache_register(
            coordinator, rank_0, [Headers.PREFIX_CACHE_REGISTER.value, "replacement-stream"]
        )

        assert coordinator._rank_hashes[rank_0] == set()
        assert coordinator._rank_cache_needs_full_snapshot[rank_0]
        assert coordinator._rank_cache_stream_ids[rank_0] == "replacement-stream"
        assert coordinator._rank_cache_resync_ids[rank_0] == 1
        assert coordinator._rank_hashes[rank_1] == {20, 30}
        idx_0 = coordinator.identity_to_rank_index[rank_0]
        idx_1 = coordinator.identity_to_rank_index[rank_1]
        assert idx_0 not in coordinator._hash_table[20]
        assert coordinator._hash_table[20][idx_1] > 0

        epoch_call, resync_call = coordinator.router_socket.send_multipart.call_args_list[-2:]
        destination, packed_epoch = epoch_call.args[0]
        assert destination == rank_0
        assert msgpack.unpackb(packed_epoch, raw=False) == [Headers.SET_GENERATION_EPOCH.value, 12]
        destination, packed_resync = resync_call.args[0]
        assert destination == rank_0
        assert msgpack.unpackb(packed_resync, raw=False) == [
            Headers.PREFIX_CACHE_RESYNC.value,
            "replacement-stream",
            1,
        ]

        assert not _apply_cache_state(
            coordinator, rank_0, 12, 2, False, [40], [], stream_id=old_stream, resync_id=old_resync
        )
        assert _apply_cache_state(coordinator, rank_0, 12, 1, True, [40], [])
        assert coordinator._rank_hashes[rank_0] == {40}

    def test_submit_handler_routes_bos_hit_then_bypasses_prompt_logprob_reuse(self):
        """Production submit routing uses BOS parity and prompt-logprob eligibility."""
        coordinator = make_coordinator_direct(prefix_caching_routing_alpha=1.0)
        coordinator.tokenizer = DummyTokenizer(vocab_size=128, bos=127)
        rank_0, rank_1 = coordinator.identities_of_data_parallel_ranks
        client = b"client"
        coordinator.known_clients = {client}
        coordinator.next_request_id = 0
        coordinator.request_id_to_client_id = {}
        coordinator.request_id_to_client_request_id = {}
        coordinator.request_id_to_rank = {}
        coordinator.schedule_records = None
        coordinator._send_to_engine = MagicMock(return_value=True)
        coordinator._pending_counts[:] = 1
        prompt = "1 2 3 4 5 6 7"
        bos_hashes = coordinator.compute_request_hashes(prompt, add_BOS=True)
        assert _apply_cache_state(coordinator, rank_1, None, 1, True, bos_hashes, [])

        hit_sampling_params = SamplingParams(
            num_tokens_to_generate=1, add_BOS=True, skip_prompt_log_probs=True
        )
        handle_submit_request(
            coordinator,
            client,
            [Headers.SUBMIT_REQUEST.value, 99, prompt, hit_sampling_params.serialize()],
        )
        assert coordinator._send_to_engine.call_args.args[0] == rank_1

        # Prompt-position log probabilities require execution of every prompt
        # position, so the same cached BOS prefix is deliberately ignored. The
        # first submission made rank 1 more loaded, causing load fallback to rank 0.
        prompt_logprob_params = SamplingParams(
            num_tokens_to_generate=1,
            add_BOS=True,
            return_log_probs=True,
            skip_prompt_log_probs=False,
        )
        handle_submit_request(
            coordinator,
            client,
            [Headers.SUBMIT_REQUEST.value, 100, prompt, prompt_logprob_params.serialize()],
        )
        assert coordinator._send_to_engine.call_args.args[0] == rank_0
