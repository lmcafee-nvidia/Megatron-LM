# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Direct-handler stress coverage for prefix-cache-aware coordinator routing."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.data_parallel_inference_coordinator.handlers import (
    handle_control_signal,
    handle_engine_reply,
    handle_prefix_cache_state,
    handle_submit_request,
)
from megatron.core.inference.data_parallel_inference_coordinator.state import CoordinatorState
from megatron.core.inference.headers import Headers
from megatron.core.inference.sampling_params import SamplingParams
from tests.test_utils.python_scripts.prefix_cache_coverage import (
    generated_matrix_cases,
    load_manifest,
)
from tests.unit_tests.inference.coordinator_test_utils import make_coordinator_direct

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = load_manifest(_REPO_ROOT / "tests/test_utils/prefix_cache_coverage.yaml")
_COORDINATOR_CASES = generated_matrix_cases("coordinator", _MANIFEST["matrices"]["coordinator"])
_BLOCK_SIZE = 4
_RANK_COUNT = 3
_MAX_REQUESTS = 16
_CLIENT = b"client"
_POLICIES = {
    "longest_prefix": PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
    "first_prefix_block": PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK,
}
_ROUTING_WEIGHTS = {"load_only": 0.0, "balanced": 0.5, "prefix_only": 1.0}
_WORKER_STATES = ("empty", "matched", "evicted", "reconnected", "refit")
_STATE_TRANSITIONS = {
    "empty": ("matched", "empty", "matched", "empty", "matched", "empty"),
    "matched": ("empty", "matched", "evicted", "matched", "empty", "matched"),
    "evicted": ("matched", "evicted", "matched", "evicted", "matched", "empty"),
    "reconnected": ("matched", "reconnected", "matched", "reconnected", "empty", "matched"),
    "refit": ("matched", "refit", "matched", "refit", "empty", "matched"),
}
_REPEATS_PER_ROW = 6
_LOGICAL_CASE_COUNT = len(_COORDINATOR_CASES) * _REPEATS_PER_ROW
_DECISIONS_PER_CASE = 6


class _IntegerTokenizer:
    """Tokenize whitespace-separated integers without changing their values."""

    bos = None
    eod = 0
    pad = 0
    vocab_size = 65_536

    def tokenize(self, prompt):
        return [int(token) for token in prompt.split()]

    def detokenize(self, tokens, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(token) for token in tokens)


@dataclass
class _ScalarCacheState:
    """Small independent model of confirmed worker state and publication recency."""

    ranks: tuple[bytes, ...]
    epoch: int | None
    assignment_counter: int
    hashes: dict[bytes, set[int]]
    timestamps: dict[bytes, dict[int, int]]
    stream_ids: dict[bytes, str]
    resync_ids: dict[bytes, int]
    sequences: dict[bytes, int | None]
    needs_full_snapshot: dict[bytes, bool]

    @classmethod
    def empty(cls, ranks):
        ranks = tuple(ranks)
        return cls(
            ranks=ranks,
            epoch=None,
            assignment_counter=0,
            hashes={rank: set() for rank in ranks},
            timestamps={rank: {} for rank in ranks},
            stream_ids={rank: f"stream-{idx}" for idx, rank in enumerate(ranks)},
            resync_ids={rank: 1 for rank in ranks},
            sequences={rank: None for rank in ranks},
            needs_full_snapshot={rank: True for rank in ranks},
        )

    def apply(self, rank, epoch, sequence, is_full_snapshot, added_hashes, removed_hashes):
        """Apply one valid full/delta message to the scalar state model."""
        assert epoch == self.epoch
        assert sequence == (1 if self.sequences[rank] is None else self.sequences[rank] + 1)
        added = set(added_hashes)
        removed = set(removed_hashes)
        assert not added & removed

        previous = self.hashes[rank]
        if is_full_snapshot:
            assert not removed
            current = added
        else:
            assert not self.needs_full_snapshot[rank]
            current = (previous - removed) | added

        newly_confirmed = current - previous
        if newly_confirmed:
            self.assignment_counter += 1
        retained_timestamps = {
            block_hash: timestamp
            for block_hash, timestamp in self.timestamps[rank].items()
            if block_hash in current
        }
        retained_timestamps.update(
            {block_hash: self.assignment_counter for block_hash in newly_confirmed}
        )
        self.hashes[rank] = current
        self.timestamps[rank] = retained_timestamps
        self.sequences[rank] = sequence
        self.needs_full_snapshot[rank] = False

    def reset_rank(self, rank):
        self.hashes[rank] = set()
        self.timestamps[rank] = {}
        self.sequences[rank] = None
        self.needs_full_snapshot[rank] = True

    def restart_rank(self, rank, stream_id):
        self.reset_rank(rank)
        self.stream_ids[rank] = stream_id
        self.resync_ids[rank] = 1

    def reset_epoch(self, epoch):
        self.epoch = epoch
        for rank in self.ranks:
            self.reset_rank(rank)
            self.resync_ids[rank] += 1


def _make_stress_coordinator(policy, alpha):
    coordinator = make_coordinator_direct(
        data_parallel_size=_RANK_COUNT,
        block_size_tokens=_BLOCK_SIZE,
        prefix_caching_routing_alpha=alpha,
        max_requests=_MAX_REQUESTS,
        policy=policy,
        tokenizer=_IntegerTokenizer(),
    )
    coordinator.known_clients = {_CLIENT}
    coordinator.next_request_id = 0
    coordinator.request_id_to_client_id = {}
    coordinator.request_id_to_client_request_id = {}
    coordinator.request_id_to_rank = {}
    coordinator.schedule_records = []
    coordinator.state = CoordinatorState.RUNNING
    coordinator._send_to_engine = MagicMock(return_value=True)
    coordinator._broadcast_to_engines = MagicMock()
    coordinator.detokenize = MagicMock()
    coordinator.router_socket = MagicMock()
    return coordinator


def _assert_protocol_state(coordinator, scalar_state):
    """Compare every confirmed-state index with the independent scalar model."""
    assert coordinator.generation_epoch == scalar_state.epoch
    assert coordinator._rank_hashes == scalar_state.hashes
    assert coordinator._rank_cache_stream_ids == scalar_state.stream_ids
    assert coordinator._rank_cache_resync_ids == scalar_state.resync_ids
    assert coordinator._rank_cache_sequences == scalar_state.sequences
    assert coordinator._rank_cache_needs_full_snapshot == scalar_state.needs_full_snapshot
    assert coordinator._rank_cache_epochs == {
        rank: (scalar_state.epoch if not scalar_state.needs_full_snapshot[rank] else None)
        for rank in scalar_state.ranks
    }

    expected_reverse_index = {}
    for rank_index, rank in enumerate(scalar_state.ranks):
        for block_hash, timestamp in scalar_state.timestamps[rank].items():
            expected_reverse_index.setdefault(block_hash, {})[rank_index] = timestamp
    assert coordinator._hash_table == expected_reverse_index


def _publish_state(coordinator, scalar_state, rank, target_hashes):
    """Publish the shortest exact full/delta update needed for ``target_hashes``."""
    target_hashes = set(target_hashes)
    current_hashes = scalar_state.hashes[rank]
    is_full_snapshot = scalar_state.needs_full_snapshot[rank]
    sequence = 1 if is_full_snapshot else scalar_state.sequences[rank] + 1
    added_hashes = target_hashes if is_full_snapshot else target_hashes - current_hashes
    removed_hashes = set() if is_full_snapshot else current_hashes - target_hashes
    payload = [
        Headers.PREFIX_CACHE_STATE.value,
        scalar_state.epoch,
        scalar_state.stream_ids[rank],
        scalar_state.resync_ids[rank],
        sequence,
        is_full_snapshot,
        sorted(added_hashes),
        sorted(removed_hashes),
    ]

    handle_prefix_cache_state(coordinator, rank, payload)
    scalar_state.apply(
        rank, scalar_state.epoch, sequence, is_full_snapshot, added_hashes, removed_hashes
    )
    _assert_protocol_state(coordinator, scalar_state)


def _publish_match_profile(
    coordinator, scalar_state, request_hashes, target_rank_index, target_depth
):
    """Make the target deepest and most recent, with a shallower contender."""
    ranks = scalar_state.ranks
    target_rank = ranks[target_rank_index]
    contender_rank = ranks[(target_rank_index + 1) % len(ranks)]

    # Clear first so retained timestamps cannot hide a recency-ordering bug.
    for rank in ranks:
        _publish_state(coordinator, scalar_state, rank, ())
    _publish_state(
        coordinator, scalar_state, contender_rank, request_hashes[: max(0, target_depth - 1)]
    )
    _publish_state(coordinator, scalar_state, target_rank, request_hashes[:target_depth])


def _transition_worker_state(
    coordinator, scalar_state, state_name, prompt, request_hashes, target_rank_index, target_depth
):
    """Drive one worker lifecycle transition and return current-epoch hashes."""
    ranks = scalar_state.ranks
    target_rank = ranks[target_rank_index]
    previous_rank = ranks[(target_rank_index - 1) % len(ranks)]

    if state_name == "empty":
        for rank in ranks:
            _publish_state(coordinator, scalar_state, rank, ())
    elif state_name == "matched":
        _publish_match_profile(
            coordinator, scalar_state, request_hashes, target_rank_index, target_depth
        )
    elif state_name == "evicted":
        _publish_state(coordinator, scalar_state, previous_rank, request_hashes[:target_depth])
        _publish_state(coordinator, scalar_state, previous_rank, ())
        _publish_match_profile(
            coordinator, scalar_state, request_hashes, target_rank_index, target_depth
        )
    elif state_name == "reconnected":
        for rank in ranks:
            _publish_state(coordinator, scalar_state, rank, ())
        _publish_state(coordinator, scalar_state, previous_rank, request_hashes[:target_depth])
        replacement_stream = f"{scalar_state.stream_ids[previous_rank]}-restart"
        coordinator._handle_rank_registration(previous_rank, replacement_stream)
        scalar_state.restart_rank(previous_rank, replacement_stream)
        _assert_protocol_state(coordinator, scalar_state)
        _publish_state(
            coordinator,
            scalar_state,
            ranks[(target_rank_index + 1) % len(ranks)],
            request_hashes[: max(0, target_depth - 1)],
        )
        _publish_state(coordinator, scalar_state, target_rank, request_hashes[:target_depth])
    elif state_name == "refit":
        next_epoch = 1 if scalar_state.epoch is None else scalar_state.epoch + 1
        handle_control_signal(
            coordinator, _CLIENT, [Headers.SET_GENERATION_EPOCH.value, next_epoch]
        )
        scalar_state.reset_epoch(next_epoch)
        _assert_protocol_state(coordinator, scalar_state)
        request_hashes = coordinator.compute_request_hashes(prompt)
        _publish_state(
            coordinator,
            scalar_state,
            ranks[(target_rank_index + 1) % len(ranks)],
            request_hashes[: max(0, target_depth - 1)],
        )
        _publish_state(coordinator, scalar_state, target_rank, request_hashes[:target_depth])
    else:
        raise AssertionError(f"unknown worker state: {state_name}")

    return request_hashes


def _scalar_route(policy, alpha, request_hashes, scalar_state, pending_counts):
    """Return the winning rank using scalar policy code independent of production."""
    effective_hashes = (
        request_hashes[:1]
        if policy == PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        else request_hashes
    )
    candidates = []
    for rank_index, rank in enumerate(scalar_state.ranks):
        depth = 0
        recency = 0
        for block_index in range(len(effective_hashes) - 1, -1, -1):
            block_hash = effective_hashes[block_index]
            if block_hash in scalar_state.hashes[rank]:
                depth = block_index + 1
                recency = scalar_state.timestamps[rank][block_hash]
                break
        match_fraction = depth / len(effective_hashes) if effective_hashes else 0.0
        free_fraction = max(0, _MAX_REQUESTS - pending_counts[rank_index]) / _MAX_REQUESTS
        score = alpha * match_fraction + (1.0 - alpha) * free_fraction
        candidates.append((score, recency, -rank_index))
    winner = max(range(len(candidates)), key=candidates.__getitem__)
    return scalar_state.ranks[winner], candidates


def _install_background_load(coordinator, load_rank_index, outstanding_requests):
    """Install a rotating background load while retaining measured arrivals."""
    background = [9, 8, 7]
    background[load_rank_index] = 0
    outstanding_by_rank = [
        sum(rank == candidate for _, rank in outstanding_requests)
        for candidate in coordinator._identities_list
    ]
    coordinator._pending_counts[:] = [
        base + outstanding for base, outstanding in zip(background, outstanding_by_rank)
    ]


def _submit_and_check(coordinator, scalar_state, policy, alpha, prompt, request_hashes):
    pending_before = tuple(int(value) for value in coordinator._pending_counts)
    expected_rank, scalar_scores = _scalar_route(
        policy, alpha, request_hashes, scalar_state, pending_before
    )
    request_id = coordinator.next_request_id
    sampling_params = SamplingParams(num_tokens_to_generate=1, skip_prompt_log_probs=True)
    handle_submit_request(
        coordinator,
        _CLIENT,
        [Headers.SUBMIT_REQUEST.value, request_id, prompt, sampling_params.serialize()],
    )
    actual_rank = coordinator.request_id_to_rank[request_id]
    assert actual_rank == expected_rank, (
        f"policy={policy.value}, alpha={alpha}, pending={pending_before}, "
        f"scores={scalar_scores}"
    )
    assert coordinator._pending_counts[coordinator.identity_to_rank_index[actual_rank]] == (
        pending_before[coordinator.identity_to_rank_index[actual_rank]] + 1
    )
    return request_id, actual_rank


def _complete_request(coordinator, request_id, rank):
    handle_engine_reply(
        coordinator, rank, [Headers.ENGINE_REPLY.value, [{"request_id": request_id}]]
    )


class TestCoordinatorStressMatrix:
    """Stress routing, cache-state churn, and arrivals in one stable test node."""

    @pytest.mark.parametrize(
        "case_id,row",
        generated_matrix_cases("coordinator", _MANIFEST["matrices"]["coordinator"]),
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_coordinator_pairwise_routing_stress(self, case_id, row):
        """Exercise 36 independently checked decisions for one concise pairwise row."""
        del case_id
        policy = _POLICIES[row["coordinator_policy"]]
        weight_name = row["routing_alpha"]
        alpha = _ROUTING_WEIGHTS[weight_name]
        arrival_pattern = row["arrival_pattern"]
        worker_state = row["worker_cache_state"]

        for repeat in range(_REPEATS_PER_ROW):
            coordinator = _make_stress_coordinator(policy, alpha)
            ranks = tuple(coordinator._identities_list)
            scalar_state = _ScalarCacheState.empty(ranks)
            prompt = list(range(1, _BLOCK_SIZE * 6 + 1))
            request_hashes = coordinator.compute_request_hashes(prompt)
            worker_state_index = _WORKER_STATES.index(worker_state)
            state_sequence = _STATE_TRANSITIONS[worker_state]
            outstanding_requests = []
            routed_ranks = []
            conflicting_signal_decisions = 0

            for transition_index, state_name in enumerate(state_sequence):
                target_rank_index = (worker_state_index + transition_index + repeat) % len(ranks)
                load_rank_index = (
                    target_rank_index + (transition_index + repeat) % len(ranks)
                ) % len(ranks)
                if state_name != "empty" and target_rank_index != load_rank_index:
                    conflicting_signal_decisions += 1
                target_depth = 1 + (worker_state_index + 2 * transition_index + repeat) % len(
                    request_hashes
                )
                request_hashes = _transition_worker_state(
                    coordinator,
                    scalar_state,
                    state_name,
                    prompt,
                    request_hashes,
                    target_rank_index,
                    target_depth,
                )
                _install_background_load(coordinator, load_rank_index, outstanding_requests)
                request_id, routed_rank = _submit_and_check(
                    coordinator, scalar_state, policy, alpha, prompt, request_hashes
                )
                outstanding_requests.append((request_id, routed_rank))
                routed_ranks.append(routed_rank)

                if arrival_pattern == "serial" or (
                    arrival_pattern == "staggered" and transition_index % 2 == 1
                ):
                    completed_request = (
                        outstanding_requests.pop()
                        if arrival_pattern == "serial"
                        else outstanding_requests.pop(0)
                    )
                    _complete_request(coordinator, *completed_request)

            route_changes = sum(
                previous != current for previous, current in zip(routed_ranks, routed_ranks[1:])
            )
            assert route_changes >= 2, (
                f"case={(policy.value, weight_name, arrival_pattern, worker_state)} "
                f"only changed route {route_changes} times: {routed_ranks}"
            )
            assert conflicting_signal_decisions >= 1
            assert len(state_sequence) >= 3
            for request in outstanding_requests:
                _complete_request(coordinator, *request)
            assert not coordinator.request_id_to_rank

        assert _LOGICAL_CASE_COUNT == 90
        assert _LOGICAL_CASE_COUNT * _DECISIONS_PER_CASE == 540
