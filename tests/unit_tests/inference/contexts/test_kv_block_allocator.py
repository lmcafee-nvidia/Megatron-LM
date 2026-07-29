# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator

POOL_SIZE = 10
PAUSED_LIMIT = 2
MAX_REQUESTS = 8
MAX_BLOCKS_PER_REQ = 4


def _make_context(
    paused_request_count=0,
    total_request_count=0,
    request_kv_block_counts=None,
    request_to_kv_block_ids=None,
    prefix_cache_lru_clock=0,
):
    """Build a minimal DynamicInferenceContext-like fake for the allocator."""
    if request_kv_block_counts is None:
        request_kv_block_counts = torch.zeros(MAX_REQUESTS, dtype=torch.int32)
    if request_to_kv_block_ids is None:
        request_to_kv_block_ids = -torch.ones((MAX_REQUESTS, MAX_BLOCKS_PER_REQ), dtype=torch.int32)
    return SimpleNamespace(
        paused_request_count=paused_request_count,
        total_request_count=total_request_count,
        request_kv_block_counts=request_kv_block_counts,
        request_to_kv_block_ids=request_to_kv_block_ids,
        prefix_cache_lru_clock=prefix_cache_lru_clock,
    )


def test_allocate_release_reset_round_trip_no_prefix_caching():
    """End-to-end exercise of the no-prefix-caching path: allocate from the
    bag (popping IDs off the top), release returns them, reset rewinds.

    Also covers the surrounding invariants the allocator must preserve:
    pool_avail bookkeeping, paused-limit headroom validation, the computed
    allocatable count, the is_memory_available fast-path + no-eviction fallback,
    and the noop behaviour of release([]).
    """
    ctx = _make_context()

    # The paused limit must leave one usable non-dummy block for liveness.
    with pytest.raises(AssertionError):
        KVBlockAllocator(ctx, pool_size=3, paused_limit=2)
    with pytest.raises(AssertionError):
        KVBlockAllocator(ctx, pool_size=3, paused_limit=-1)
    with pytest.raises(AssertionError):
        KVBlockAllocator(ctx, pool_size=1, paused_limit=0)

    a = KVBlockAllocator(ctx, pool_size=POOL_SIZE, paused_limit=PAUSED_LIMIT)
    # Initial state: POOL_SIZE - 1 (dummy block) available, nothing used.
    assert a.pool_avail == POOL_SIZE - 1
    assert a.get_allocatable_count() == POOL_SIZE - 1
    assert a.get_total_used() == 0
    assert not hasattr(a, "active_count")
    assert not hasattr(a, "get_active_avail")
    assert not hasattr(a, "get_paused_avail")
    assert not hasattr(a, "get_allocatable_block_count")
    assert str(a) == "blocks: occupied 0/9; allocatable 9; active-used 0; paused-used 0/2"
    # is_memory_available short-circuits True when free pool has enough.
    assert a.is_memory_available(5) is True

    # Allocate 3 → pop IDs off the top of the bag.
    ids = a.allocate_memory_blocks(3)
    assert ids is not None and ids.numel() == 3
    assert a.pool_avail == POOL_SIZE - 1 - 3
    assert a.get_allocatable_count() == POOL_SIZE - 1 - 3

    # Empty release is a no-op; non-empty release returns IDs to the bag.
    before = a.pool_avail
    a.release_memory_blocks(torch.tensor([], dtype=torch.int32))
    assert a.pool_avail == before
    a.release_memory_blocks(ids)
    assert a.pool_avail == before + 3
    assert a.get_allocatable_count() == before + 3

    # Free pool exhausted: without prefix caching there's no eviction path,
    # so both is_memory_available and allocate_memory_blocks return failure.
    small_alloc = KVBlockAllocator(ctx, pool_size=4, paused_limit=1)
    assert small_alloc.pool_avail == 3
    assert small_alloc.get_allocatable_count() == 3
    assert small_alloc.is_memory_available(5) is False
    assert small_alloc.allocate_memory_blocks(5) is None

    # reset rewinds the bag back to arange(pool_size) and clears routing state.
    a.allocate_memory_blocks(4)
    a.reset()
    assert a.pool_avail == POOL_SIZE - 1
    assert a.get_allocatable_count() == POOL_SIZE - 1
    assert a.block_bag.tolist() == list(range(POOL_SIZE))
    assert a.block_routing == {}


@pytest.mark.parametrize(
    "scope,paused,total,counts,expected_active,expected_paused",
    [
        # active_used = sum over [paused:total]; paused_used = sum over [:paused].
        ("nonempty", 1, 4, [1, 2, 3, 4, 0, 0, 0, 0], 9, 1),
        ("paused_only", 2, 2, [5, 7, 0, 0, 0, 0, 0, 0], 0, 12),
    ],
)
def test_block_usage_counts_no_prefix_caching(
    scope, paused, total, counts, expected_active, expected_paused
):
    """get_active_used / get_paused_used sum request_kv_block_counts over the
    [paused:total] and [:paused] slices respectively."""
    ctx = _make_context(
        paused_request_count=paused,
        total_request_count=total,
        request_kv_block_counts=torch.tensor(counts, dtype=torch.int32),
    )
    a = KVBlockAllocator(ctx, pool_size=POOL_SIZE, paused_limit=3)
    assert a.get_active_used() == expected_active
    assert a.get_paused_used() == expected_paused
