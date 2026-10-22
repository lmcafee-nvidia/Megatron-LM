# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explore two live requests against real prefix-cache admission and ownership code."""

import gzip
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


class SharingMachine(RuleBasedStateMachine):
    """Six blocks, two live requests, and prompts on either side of block boundaries.

    The driver supplies completion decisions without computing KV contents. Real
    resolve_requests compacts surviving rows; synthetic sampled tokens establish
    the next input count. No model outputs, scheduler races, or LRU optimality
    are claimed. Prefix identity and capacity are predicted independently.
    """

    capacity = 6
    prompt_lengths = (1, 31, 32, 33, 63, 64, 65, 95, 96, 97)
    metrics = Counter()
    histories = []
    failures = []

    def __init__(self):
        super().__init__()
        self.helper = PrefixCachingTestBase()
        self.ctx = self.helper._ctx(buffer_size_gb=0.002, rounder=1, max_requests=8, max_tokens=512)
        self.alloc = self.ctx.kv_block_allocator
        self.reserved = self.alloc.allocate_memory_blocks(
            self.alloc.pool_avail - self.capacity
        ).clone()
        self.held = []
        self.active = []  # Stable logical records, in independently predicted row order.
        self.waiting = None
        self.cache = {}  # Logical token-prefix tuple -> (physical block ID, hash).
        self.expected_hits = 0
        self.expected_matched = 0
        self.next_request = 1
        self.trace = []
        self.metrics['histories_started'] += 1

    @staticmethod
    def _key(family, index):
        # Families 0 and 1 share their first block, then diverge; family 2 is disjoint.
        return (2 if family == 2 else 0,) + (family,) * index

    def _live_blocks(self):
        return {bid for record in self.active for bid in record['blocks']}

    def _raw_free(self):
        occupied = set(self.held) | self._live_blocks() | {bid for bid, _ in self.cache.values()}
        return self.capacity - len(occupied)

    def _record(self, action, **details):
        self.trace.append(dict(action=action, **details))
        self.metrics[action] += 1

    def _admit(self, family, num_tokens, request, action):
        bs = self.ctx.block_size_tokens
        full_blocks = num_tokens // bs
        required_blocks = (num_tokens + bs - 1) // bs
        matched = []
        for index in range(full_blocks):
            key = self._key(family, index)
            if key not in self.cache:
                break
            matched.append(self.cache[key][0])
        live = self._live_blocks()
        unavailable = live | set(self.held) | set(matched)
        should_succeed = required_blocks - len(matched) <= self.capacity - len(unavailable)
        self._record(action, family=family, num_tokens=num_tokens, expected=should_succeed)
        # Full cached blocks save their tokens, except the final prompt token:
        # it must still produce the first output logits, even on an exact hit.
        computed_positions = [
            pos for pos in range(num_tokens) if pos // bs >= len(matched) or pos == num_tokens - 1
        ]
        skipped_tokens = num_tokens - len(computed_positions)
        # The component's availability query must agree even when matches are already pinned.
        requests_ok, tokens_ok, blocks_ok = self.ctx.check_availability(request)
        assert requests_ok and tokens_ok, 'fixture unexpectedly exhausted request/token capacity'
        assert bool(blocks_ok) == should_succeed, 'availability disagrees with independent capacity'
        row = len(self.active)
        token_start = self.ctx.active_token_count
        try:
            self.ctx.add_request(request)
        except BlockOverflowError:
            assert not should_succeed, 'admission rejected despite sufficient modeled capacity'
            self.waiting = (family, num_tokens, request)
            self.metrics['rejected_with_prefix' if matched else 'rejected_without_prefix'] += 1
            if self.active:
                self.metrics['rejected_with_live_request'] += 1
            return
        assert should_succeed, 'admission exceeded modeled capacity'
        blocks = self.helper._block_ids(self.ctx, row, required_blocks)
        assert len(set(blocks)) == len(blocks), 'duplicate physical blocks in one request'
        assert blocks[: len(matched)] == matched, 'matched prefix was not retained'
        assert not set(blocks).intersection(self.held + self.reserved.tolist())
        assert not set(blocks[len(matched) :]).intersection(live), 'new blocks alias a live request'
        assert request.num_cached_tokens == skipped_tokens, 'skipped-token accounting mismatch'
        token_end = self.ctx.active_token_count
        assert token_end - token_start == len(computed_positions), 'computed-token count mismatch'
        assert self.ctx.request_query_lengths[row].item() == len(computed_positions)
        assert self.ctx.request_kv_length_offsets[row].item() == skipped_tokens
        assert self.ctx.token_to_pos_ids[token_start:token_end].tolist() == computed_positions
        # Recomputed cached tokens must not overwrite a block another owner can read.
        expected_writes = [
            self.alloc.dummy_block_idx if pos // bs < len(matched) else blocks[pos // bs]
            for pos in computed_positions
        ]
        assert (
            self.ctx.token_to_block_idx[token_start:token_end].tolist() == expected_writes
        ), 'cached-token write protection mismatch'
        new_ids = set(blocks[len(matched) :])
        before = len(self.cache)
        self.cache = {key: value for key, value in self.cache.items() if value[0] not in new_ids}
        self.metrics['evicted_cache_entries'] += before - len(self.cache)
        for index in range(full_blocks):
            self.cache[self._key(family, index)] = (
                blocks[index],
                request.precomputed_block_hashes[index],
            )
        self.expected_hits += bool(matched)
        self.expected_matched += len(matched)
        self.active.append(
            dict(request=request, blocks=blocks, cached_tokens=request.num_cached_tokens)
        )
        self.waiting = None
        self.metrics['accepted_with_prefix' if matched else 'accepted_without_prefix'] += 1
        self.metrics[f'accepted_length_{num_tokens}'] += 1
        if num_tokens % bs == 0:
            self.metrics['aligned_admissions'] += 1
        if len(matched) == required_blocks:
            self.metrics['fully_cached_aligned_admissions'] += 1
            self.metrics['cached_token_write_redirects'] += 1
            if self._raw_free() == 0:
                self.metrics['fully_cached_with_empty_raw_pool'] += 1
        if len(self.active) == 2:
            self.metrics['two_live_requests'] += 1
        if set(blocks).intersection(live):
            self.metrics['shared_live_admissions'] += 1
        if action == 'retry':
            self.metrics['successful_retries'] += 1
            if row:
                self.metrics['successful_retries_with_survivor'] += 1

    @precondition(lambda self: len(self.active) < 2)
    @rule(family=st.integers(0, 2), num_tokens=st.sampled_from(prompt_lengths))
    def submit(self, family, num_tokens):
        bs = self.ctx.block_size_tokens
        tokens = self.helper._prompt(num_tokens, offset=2000 if family == 2 else 0)
        if family == 1:
            tokens[bs:] += 1000
        request = self.helper._req(self.ctx, tokens, request_id=self.next_request)
        self.next_request += 1
        self._admit(family, num_tokens, request, 'submit')

    @precondition(lambda self: len(self.active) < 2 and self.waiting is not None)
    @rule()
    def retry(self):
        self._admit(*self.waiting, 'retry')

    @precondition(lambda self: bool(self.active))
    @rule(slot=st.integers(0, 1))
    def finish(self, slot):
        slot %= len(self.active)
        self._record('finish', slot=slot)
        before = list(self.active)
        removed = self.active.pop(slot)
        mask = torch.ones(len(before), dtype=torch.int32)
        mask[slot] = 0
        finished, survivor_rows = self.ctx.resolve_requests(mask)
        assert finished.tolist() == [removed['request'].request_id], 'wrong request completed'
        assert survivor_rows.tolist() == [i for i in range(len(before)) if i != slot]
        if self.active:
            self.metrics['finish_with_survivor'] += 1
            self.metrics[f'finished_row_{slot}_with_survivor'] += 1
            if set(removed['blocks']).intersection(self._live_blocks()):
                self.metrics['shared_release_with_survivor'] += 1
            # This is a stub decoder result, not a model computation. It resets the
            # next input count through the production API, without resetting live rows.
            self.ctx.commit_sampled_tokens(torch.zeros(len(self.active), dtype=torch.int64))
        else:
            self.ctx.reset_metadata(preserve_prefix_cache=True, preserve_counters=True)

    @precondition(lambda self: self._raw_free() > 0)
    @rule(amount=st.integers(1, 6))
    def reserve(self, amount):
        amount = min(amount, self._raw_free())
        self._record('reserve', amount=amount)
        blocks = self.alloc.allocate_memory_blocks(amount)
        assert blocks is not None
        self.held.extend(blocks.tolist())

    @precondition(lambda self: bool(self.held))
    @rule(amount=st.integers(1, 6))
    def unreserve(self, amount):
        amount = min(amount, len(self.held))
        self._record('unreserve', amount=amount)
        self.alloc.release_memory_blocks(torch.tensor(self.held[-amount:], dtype=torch.int32))
        del self.held[-amount:]

    @invariant()
    def bookkeeping_matches_reference(self):
        try:
            assert self.ctx.prefix_cache_hits == self.expected_hits, 'cache-hit accounting mismatch'
            assert self.ctx.prefix_cache_blocks_matched == self.expected_matched
            assert self.ctx.total_request_count == len(self.active), 'active count mismatch'
            refs = torch.zeros_like(self.alloc.block_ref_counts)
            refs[self.reserved] = 1
            if self.held:
                refs[self.held] = 1
            for row, record in enumerate(self.active):
                blocks = record['blocks']
                refs[blocks] += 1
                assert self.ctx.request_ids[row].item() == record['request'].request_id
                assert self.helper._block_ids(self.ctx, row, len(blocks)) == blocks
                assert self.ctx.request_kv_block_counts[row].item() == len(blocks)
                assert record['request'].num_cached_tokens == record['cached_tokens']
            if self.waiting is not None:
                assert (
                    self.waiting[2].num_cached_tokens == 0
                ), 'rejected request reports cached tokens'
            assert torch.equal(self.alloc.block_ref_counts, refs), 'block ownership mismatch'
            assert self.alloc.pool_avail == self._raw_free(), 'raw free-pool mismatch'
            expected_available = self.capacity - len(set(self.held) | self._live_blocks())
            assert self.alloc.get_allocatable_count() == expected_available
            assert self.alloc.kv_hash_to_block_id == {
                hash_: bid for bid, hash_ in self.cache.values()
            }, 'cache identity ledger mismatch'
            children = Counter()
            for key, (bid, hash_) in self.cache.items():
                assert self.alloc.block_hashes[bid].item() == hash_
                parent = -1 if len(key) == 1 else self.cache[key[:-1]][0]
                assert self.alloc.block_parent_id[bid].item() == parent, 'cache parent mismatch'
                if parent >= 0:
                    children[parent] += 1
            for bid, _ in self.cache.values():
                assert self.alloc.block_child_count[bid].item() == children[bid]
        except (AssertionError, KeyError) as error:
            self.failures.append(dict(trace=list(self.trace), error=str(error)))
            raise

    def teardown(self):
        self.histories.append(list(self.trace))
        if self.active:
            self.ctx.release_memory_blocks_from_request_indexes(torch.arange(len(self.active)))
        if self.held:
            self.alloc.release_memory_blocks(torch.tensor(self.held, dtype=torch.int32))
        self.alloc.release_memory_blocks(self.reserved)
        del self.ctx, self.alloc, self.active, self.waiting
        self.helper.teardown_method(None)


TestPrefixSharingStateful = SharingMachine.TestCase
TestPrefixSharingStateful.settings = settings(
    max_examples=int(os.environ.get('STATEFUL_MAX_EXAMPLES', '100')),
    stateful_step_count=int(os.environ.get('STATEFUL_MAX_STEPS', '80')),
    deadline=None,
    database=None,
    phases=(Phase.generate, Phase.shrink),
    report_multiple_bugs=False,
)
TestPrefixSharingStateful = pytest.mark.internal(TestPrefixSharingStateful)


@pytest.mark.internal
@pytest.mark.skipif(
    'STATEFUL_REPLAY_FILE' not in os.environ, reason='requires a discovered history'
)
def test_replay_sharing_history():
    history = json.loads(Path(os.environ['STATEFUL_REPLAY_FILE']).read_text())
    machine = SharingMachine()
    try:
        machine.bookkeeping_matches_reference()
        for step in history:
            if step['action'] == 'submit':
                machine.submit(step['family'], step['num_tokens'])
            elif step['action'] == 'finish':
                machine.finish(step['slot'])
            elif step['action'] in ('reserve', 'unreserve'):
                getattr(machine, step['action'])(step['amount'])
            else:
                assert step['action'] == 'retry'
                machine.retry()
            machine.bookkeeping_matches_reference()
    finally:
        machine.teardown()


@pytest.fixture(scope='module', autouse=True)
def sharing_context_lifecycle():
    SharingMachine.metrics.clear()
    SharingMachine.histories.clear()
    SharingMachine.failures.clear()
    PrefixCachingTestBase.setup_class()
    try:
        yield
    finally:
        PrefixCachingTestBase.teardown_class()
        output = os.environ.get('STATEFUL_EVIDENCE_DIR')
        if output:
            destination = Path(output)
            destination.mkdir(parents=True, exist_ok=True)
            with gzip.open(destination / 'histories.json.gz', 'wt') as stream:
                json.dump(
                    dict(
                        metrics=SharingMachine.metrics,
                        histories=SharingMachine.histories,
                        failures=SharingMachine.failures,
                    ),
                    stream,
                )
