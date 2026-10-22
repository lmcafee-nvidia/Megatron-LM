# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generated cache-admission histories; install the pilot's pinned Hypothesis dependency."""

import json
import os
from collections import Counter
from pathlib import Path

import pytest
import torch
from hypothesis import Phase, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from megatron.core.inference.contexts.dynamic_context import BlockOverflowError
from tests.unit_tests.inference.contexts.test_dynamic_prefix_caching import PrefixCachingTestBase


class AdmissionMachine(RuleBasedStateMachine):
    """One request slot, two prompt families, and four usable KV blocks.

    The reference ledger predicts prefix reuse from token-family identities,
    not production's prefix-matching helper. Physical block IDs are opaque
    allocation results; eviction choice is intentionally left to production.
    Every prompt ends in a partial block, so all matched full blocks can be skipped.
    """

    metrics = Counter()
    histories = []
    failures = []
    capacity = 4

    def __init__(self):
        super().__init__()
        self.helper = PrefixCachingTestBase()
        self.ctx = self.helper._ctx(buffer_size_gb=0.002, rounder=1, max_requests=8, max_tokens=512)
        self.alloc = self.ctx.kv_block_allocator
        # Use the real allocator to reserve surplus capacity, without changing its internals.
        self.reserved = self.alloc.allocate_memory_blocks(
            self.alloc.pool_avail - self.capacity
        ).clone()
        self.held = []
        self.active = None
        self.waiting = None
        self.cache = {}  # (prompt family, full-block index) -> (physical ID, hash)
        self.expected_hits = 0
        self.expected_matched = 0
        self.next_request = 1
        self.trace = []
        self.metrics['histories_started'] += 1

    def _record(self, action, **details):
        self.trace.append(dict(action=action, **details))
        self.metrics[action] += 1

    def _raw_free(self):
        occupied = set(self.held) | {bid for bid, _ in self.cache.values()}
        if self.active is not None:
            occupied.update(self.active[1])
        return self.capacity - len(occupied)

    def _admit(self, family, full_blocks, request, action):
        matched = []
        for index in range(full_blocks):
            if (family, index) not in self.cache:
                break
            matched.append(self.cache[family, index][0])
        should_succeed = full_blocks + 1 <= self.capacity - len(self.held)
        self._record(
            action,
            family=family,
            full_blocks=full_blocks,
            matched=len(matched),
            held=len(self.held),
            expected='accepted' if should_succeed else 'rejected',
        )
        try:
            self.ctx.add_request(request)
        except BlockOverflowError:
            assert not should_succeed, 'admission rejected despite sufficient modeled capacity'
            self.metrics['rejected_with_prefix' if matched else 'rejected_without_prefix'] += 1
            self.waiting = (family, full_blocks, request)
            return
        assert should_succeed, 'admission exceeded modeled capacity'
        blocks = self.helper._block_ids(self.ctx, 0, full_blocks + 1)
        assert len(set(blocks)) == len(blocks), 'duplicate physical blocks in one request'
        assert blocks[: len(matched)] == matched, 'matched prefix was not retained'
        assert not set(blocks).intersection(self.held + self.reserved.tolist())
        assert request.num_cached_tokens == len(matched) * self.ctx.block_size_tokens
        # Reassigned IDs evict their previous logical contents. No LRU algorithm is copied.
        new_ids = set(blocks[len(matched) :])
        self.cache = {key: value for key, value in self.cache.items() if value[0] not in new_ids}
        for index in range(full_blocks):
            self.cache[family, index] = (blocks[index], request.precomputed_block_hashes[index])
        self.expected_hits += bool(matched)
        self.expected_matched += len(matched)
        self.active = (request, blocks)
        self.waiting = None
        self.metrics['accepted_with_prefix' if matched else 'accepted_without_prefix'] += 1
        if action == 'retry':
            self.metrics['successful_retries'] += 1

    @precondition(lambda self: self.active is None)
    @rule(family=st.integers(0, 1), full_blocks=st.integers(0, 3))
    def submit(self, family, full_blocks):
        request = self.helper._req(
            self.ctx,
            self.helper._prompt(full_blocks * self.ctx.block_size_tokens + 1, offset=family * 1000),
            request_id=self.next_request,
        )
        self.next_request += 1
        self._admit(family, full_blocks, request, 'submit')

    @precondition(lambda self: self.active is None and self.waiting is not None)
    @rule()
    def retry(self):
        self._admit(*self.waiting, 'retry')

    @precondition(lambda self: self.active is not None)
    @rule()
    def finish(self):
        self._record('finish')
        self.ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        self.ctx.reset_metadata(preserve_prefix_cache=True, preserve_counters=True)
        self.active = None

    @precondition(lambda self: self._raw_free() > 0)
    @rule(amount=st.integers(1, 4))
    def reserve(self, amount):
        amount = min(amount, self._raw_free())
        self._record('reserve', amount=amount)
        blocks = self.alloc.allocate_memory_blocks(amount)
        assert blocks is not None
        self.held.extend(blocks.tolist())

    @precondition(lambda self: bool(self.held))
    @rule(amount=st.integers(1, 4))
    def unreserve(self, amount):
        amount = min(amount, len(self.held))
        self._record('unreserve', amount=amount)
        self.alloc.release_memory_blocks(torch.tensor(self.held[-amount:], dtype=torch.int32))
        del self.held[-amount:]

    @invariant()
    def bookkeeping_matches_reference(self):
        try:
            assert self.ctx.prefix_cache_hits == self.expected_hits, (
                f'cache-hit accounting: expected {self.expected_hits}, '
                f'observed {self.ctx.prefix_cache_hits}; trace={self.trace}'
            )
            assert self.ctx.prefix_cache_blocks_matched == self.expected_matched
            assert self.ctx.total_request_count == int(self.active is not None)
            if self.waiting is not None:
                assert (
                    self.waiting[2].num_cached_tokens == 0
                ), 'rejected request reports cached tokens'
            refs = torch.zeros_like(self.alloc.block_ref_counts)
            refs[self.reserved] = 1
            if self.held:
                refs[self.held] = 1
            if self.active is not None:
                refs[self.active[1]] += 1
            assert torch.equal(self.alloc.block_ref_counts, refs), 'block ownership mismatch'
            assert self.alloc.pool_avail == self._raw_free(), 'raw free-pool mismatch'
            assert self.alloc.kv_hash_to_block_id == {
                hash_: bid for bid, hash_ in self.cache.values()
            }, 'cache identity ledger mismatch'
        except AssertionError as error:
            self.failures.append(dict(trace=list(self.trace), error=str(error)))
            raise

    def teardown(self):
        self.histories.append(list(self.trace))
        if self.active is not None:
            self.ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        if self.held:
            self.alloc.release_memory_blocks(torch.tensor(self.held, dtype=torch.int32))
        self.alloc.release_memory_blocks(self.reserved)
        del self.ctx, self.alloc, self.active, self.waiting
        self.helper.teardown_method(None)


TestPrefixAdmissionStateful = AdmissionMachine.TestCase
TestPrefixAdmissionStateful.settings = settings(
    max_examples=100,
    stateful_step_count=30,
    deadline=None,
    database=None,
    phases=(Phase.generate, Phase.shrink),
    report_multiple_bugs=False,
)
TestPrefixAdmissionStateful = pytest.mark.internal(TestPrefixAdmissionStateful)


@pytest.mark.internal
@pytest.mark.skipif(
    'STATEFUL_REPLAY_FILE' not in os.environ, reason='requires a discovered history'
)
def test_replay_discovered_history():
    """Replay a generator-produced artifact without invoking search or shrinking."""
    history = json.loads(Path(os.environ['STATEFUL_REPLAY_FILE']).read_text())
    machine = AdmissionMachine()
    try:
        machine.bookkeeping_matches_reference()
        for step in history:
            action = step['action']
            if action == 'submit':
                machine.submit(step['family'], step['full_blocks'])
            elif action in ('reserve', 'unreserve'):
                getattr(machine, action)(step['amount'])
            else:
                assert action in ('finish', 'retry')
                getattr(machine, action)()
            machine.bookkeeping_matches_reference()
    finally:
        machine.teardown()


@pytest.fixture(scope='module', autouse=True)
def stateful_context_lifecycle():
    """Use normal distributed setup/teardown and optionally retain discovery evidence."""
    AdmissionMachine.metrics.clear()
    AdmissionMachine.histories.clear()
    AdmissionMachine.failures.clear()
    PrefixCachingTestBase.setup_class()
    try:
        yield
    finally:
        PrefixCachingTestBase.teardown_class()
        output = os.environ.get('STATEFUL_EVIDENCE_DIR')
        if output:
            destination = Path(output)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / 'histories.json').write_text(
                json.dumps(
                    dict(
                        metrics=AdmissionMachine.metrics,
                        histories=AdmissionMachine.histories,
                        failures=AdmissionMachine.failures,
                    ),
                    indent=2,
                )
                + '\n'
            )
