# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Execution-stress matrices for prefix-cache block allocation."""

import heapq
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator
from tests.test_utils.python_scripts.prefix_cache_coverage import (
    generated_matrix_cases,
    load_manifest,
)
from tests.unit_tests.inference.contexts.test_dynamic_prefix_caching import (
    PrefixCachingTestBase as _PrefixCachingTestBase,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = load_manifest(_REPO_ROOT / "tests/test_utils/prefix_cache_coverage.yaml")
_POOL_SIZE = 18


def _context():
    return SimpleNamespace(
        paused_request_count=0,
        total_request_count=0,
        request_kv_block_counts=torch.zeros(8, dtype=torch.int32),
        request_to_kv_block_ids=torch.full((8, 8), -1, dtype=torch.int32),
        prefix_cache_lru_clock=0,
    )


def _policy(name):
    return {
        "lru": PrefixCachingEvictionPolicy.LRU,
        "ref_zero": PrefixCachingEvictionPolicy.REF_ZERO,
    }[name]


def _assert_allocator_state(allocator):
    """Check the allocator against independently reconstructed scalar state."""
    usable_ids = set(range(allocator.pool_size - 1))
    free_ids = allocator.block_bag[: allocator.pool_avail].tolist()
    assert len(free_ids) == len(set(free_ids))
    assert set(free_ids) <= usable_ids

    occupied_ids = usable_ids - set(free_ids)
    assert allocator.get_total_used() == len(occupied_ids)
    assert torch.all(allocator.block_ref_counts >= 0)
    assert all(allocator.block_ref_counts[block_id].item() == 0 for block_id in free_ids)
    assert all(allocator.block_hashes[block_id].item() == -1 for block_id in free_ids)

    registered = {
        allocator.block_hashes[block_id].item(): block_id
        for block_id in occupied_ids
        if allocator.block_hashes[block_id].item() > 0
    }
    assert allocator.kv_hash_to_block_id == registered

    evictable = sum(
        allocator.block_ref_counts[block_id].item() == 0
        and allocator.block_hashes[block_id].item() > 0
        for block_id in occupied_ids
    )
    expected_allocatable = allocator.pool_avail
    if allocator.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
        expected_allocatable += evictable
        expected_children = {block_id: 0 for block_id in occupied_ids}
        for block_id in occupied_ids:
            if allocator.block_hashes[block_id].item() < 0:
                continue
            parent_id = allocator.block_parent_id[block_id].item()
            if parent_id >= 0:
                assert parent_id in occupied_ids
                assert allocator.block_hashes[parent_id].item() > 0
                expected_children[parent_id] += 1
        assert all(
            allocator.block_child_count[block_id].item() == child_count
            for block_id, child_count in expected_children.items()
        )
    assert allocator.get_allocatable_count() == expected_allocatable


def _assert_policy_layout(allocator):
    """Check policy-specific state before the stress workload mutates it."""
    assert torch.all(allocator.block_hashes == -1)
    assert torch.all(allocator.block_ref_counts == 0)
    assert allocator.kv_hash_to_block_id == {}
    has_lru_state = allocator.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU
    assert hasattr(allocator, "block_timestamps") is has_lru_state
    assert hasattr(allocator, "block_parent_id") is has_lru_state
    assert hasattr(allocator, "block_child_count") is has_lru_state
    if has_lru_state:
        assert torch.all(allocator.block_parent_id == -1)
        assert torch.all(allocator.block_child_count == 0)


def _longest_match(allocator, hashes):
    """Reconstruct the longest consecutive match without using context code."""
    for end in range(len(hashes), 0, -1):
        if hashes[end - 1] in allocator.kv_hash_to_block_id:
            return [allocator.kv_hash_to_block_id[value] for value in hashes[:end]]
    return []


def _drain_to(allocator, raw_free_target):
    drain_count = allocator.pool_avail - raw_free_target
    assert drain_count >= 0
    if drain_count == 0:
        return torch.empty(0, dtype=torch.int32)
    drained = allocator.allocate_memory_blocks(drain_count)
    assert drained is not None
    return drained


def _release_some(allocator, blocks, count):
    released = blocks[:count]
    retained = blocks[count:]
    allocator.release_memory_blocks(released)
    return retained


def _run_shared_reference_probe(allocator, hash_value, arrival_pattern, follower_count):
    """Run serial or overlapping followers through real reference transitions."""
    block = allocator.allocate_memory_blocks(1)
    assert block is not None
    block_id = block.item()
    raw_avail_after_allocate = allocator.pool_avail
    allocator.register_kv_block_hashes([block_id], [hash_value], [0])

    if arrival_pattern == "serial":
        for _ in range(follower_count):
            allocator.block_ref_counts[block_id] += 1
            assert allocator.block_ref_counts[block_id].item() == 2
            allocator.release_memory_blocks(block)
        peak_followers = 1
    else:
        allocator.block_ref_counts[block_id] += follower_count
        assert allocator.block_ref_counts[block_id].item() == follower_count + 1
        if arrival_pattern == "staggered":
            for _ in range(follower_count):
                allocator.release_memory_blocks(block)
        else:
            allocator.release_memory_blocks(block.repeat(follower_count))
        peak_followers = follower_count
    allocator.release_memory_blocks(block)
    assert allocator.block_ref_counts[block_id].item() == 0
    if allocator.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
        assert allocator.pool_avail == raw_avail_after_allocate + 1
        assert hash_value not in allocator.kv_hash_to_block_id
    else:
        assert allocator.pool_avail == raw_avail_after_allocate
        assert allocator.kv_hash_to_block_id[hash_value] == block_id
        assert allocator.get_allocatable_count() == raw_avail_after_allocate + 1
    return peak_followers, 2 * (follower_count + 1)


def _run_cycle(allocator, row, cycle):
    allocator.context.prefix_cache_lru_clock += 10
    hash_base = 10_000 * (cycle + 1)
    producer_hashes = [hash_base + 1, hash_base + 2, hash_base + 3]
    producer = allocator.allocate_memory_blocks(3)
    assert producer is not None
    assert torch.all(allocator.block_ref_counts[producer] == 1)
    allocator.register_kv_block_hashes(
        producer.tolist(), producer_hashes, [0, producer_hashes[0], producer_hashes[1]]
    )

    matched_count = {"none": 0, "branching": 1, "partial": 2, "exact": 3}[row["match_shape"]]
    follower_hashes = producer_hashes[:matched_count] + [
        hash_base + 100 + index for index in range(3 - matched_count)
    ]
    matched = _longest_match(allocator, follower_hashes)
    assert matched == producer[:matched_count].tolist()

    follower_count = 4 if row["arrival_pattern"] != "serial" else 1
    if matched:
        matched_tensor = torch.tensor(matched, dtype=torch.int32)
        allocator.block_ref_counts[matched_tensor] += follower_count
    allocator.release_memory_blocks(producer)
    _assert_allocator_state(allocator)

    required_new = 3 - matched_count
    pressure_demand = max(required_new, 1)
    cached_probe = None
    if (
        row["pressure_shape"] == "one_over"
        and allocator.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU
    ):
        cached_probe = allocator.allocate_memory_blocks(1)
        assert cached_probe is not None
        allocator.register_kv_block_hashes(
            cached_probe.tolist(), [hash_base + 500], parent_hashes=[0]
        )
        allocator.release_memory_blocks(cached_probe)

    pressure_pins = torch.empty(0, dtype=torch.int32)
    pinned_probe = torch.empty(0, dtype=torch.int32)
    exhaustion_filler = torch.empty(0, dtype=torch.int32)
    if row["pressure_shape"] == "pinned":
        # Keep an explicitly registered block live under both policies, then
        # pin every LRU entry that would otherwise be evictable.
        pinned_probe = allocator.allocate_memory_blocks(1)
        assert pinned_probe is not None
        allocator.register_kv_block_hashes(
            pinned_probe.tolist(), [hash_base + 700], parent_hashes=[0]
        )
        pressure_pins = torch.nonzero(
            (allocator.block_ref_counts == 0) & (allocator.block_hashes > 0), as_tuple=True
        )[0].to(torch.int32)
        allocator.block_ref_counts[pressure_pins] += 1
    elif (
        row["pressure_shape"] == "exhausted"
        and allocator.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU
    ):
        # Consume all evictable cache entries into unregistered live allocations.
        # This leaves a genuinely exhausted pool, rather than a pinned cache.
        exhaustion_filler = _drain_to(allocator, 0)
        evictable_count = int(allocator.get_evictable_block_count())
        if evictable_count:
            evicted_allocations = allocator.allocate_memory_blocks(evictable_count)
            assert evicted_allocations is not None
            assert torch.all(allocator.block_hashes[evicted_allocations] == -1)
            exhaustion_filler = torch.cat((exhaustion_filler, evicted_allocations))

    raw_target = {
        "exact_capacity": pressure_demand,
        "one_over": max(pressure_demand - 1, 0),
        "pinned": 0,
        "exhausted": 0,
    }[row["pressure_shape"]]
    filler = torch.cat((exhaustion_filler, _drain_to(allocator, raw_target)))
    assert allocator.pool_avail == raw_target
    if row["pressure_shape"] == "pinned":
        assert pinned_probe.numel() == 1
        assert allocator.block_hashes[pinned_probe].item() > 0
        assert allocator.block_ref_counts[pinned_probe].item() == 1
        assert int(allocator.get_evictable_block_count()) == 0
    elif row["pressure_shape"] == "exhausted":
        assert pinned_probe.numel() == 0
        assert int(allocator.get_evictable_block_count()) == 0

    should_succeed = row["pressure_shape"] == "exact_capacity" or (
        row["pressure_shape"] == "one_over"
        and allocator.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU
    )
    assert allocator.is_memory_available(pressure_demand) is should_succeed
    allocated = allocator.allocate_memory_blocks(pressure_demand)
    assert (allocated is not None) is should_succeed
    if allocated is None:
        needed = pressure_demand - allocator.pool_avail
        filler = _release_some(allocator, filler, needed)
        allocated = allocator.allocate_memory_blocks(pressure_demand)
        assert allocated is not None
    _assert_allocator_state(allocator)

    new_blocks = allocated[:required_new]
    if required_new:
        parent_hashes = [
            follower_hashes[index - 1] if index > 0 else 0 for index in range(matched_count, 3)
        ]
        allocator.register_kv_block_hashes(
            new_blocks.tolist(), follower_hashes[matched_count:], parent_hashes
        )
    if allocated.numel() > required_new:
        allocator.release_memory_blocks(allocated[required_new:])

    if row["arrival_pattern"] == "staggered" and matched:
        matched_tensor = torch.tensor(matched, dtype=torch.int32)
        for _ in range(follower_count):
            allocator.release_memory_blocks(matched_tensor)
    elif matched:
        allocator.release_memory_blocks(
            torch.tensor(matched, dtype=torch.int32).repeat(follower_count)
        )
    allocator.release_memory_blocks(new_blocks)
    allocator.release_memory_blocks(pressure_pins)
    allocator.release_memory_blocks(pinned_probe)
    allocator.release_memory_blocks(filler)

    follower_count = (1, 10, 64)[cycle]
    shared_followers, reference_transitions = _run_shared_reference_probe(
        allocator, hash_base + 900, row["arrival_pattern"], follower_count
    )

    # Every LRU row performs an actual leaf eviction with the raw pool empty.
    if allocator.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
        eviction_filler = _drain_to(allocator, 0)
        candidate_ids = torch.nonzero(
            (allocator.block_ref_counts == 0)
            & (allocator.block_hashes > 0)
            & (allocator.block_child_count == 0),
            as_tuple=True,
        )[0].tolist()
        assert candidate_ids
        evictable_count = int(allocator.get_evictable_block_count())
        assert allocator.is_memory_available(evictable_count)
        assert not allocator.is_memory_available(evictable_count, potential_matched_count=1)
        assert not allocator.is_memory_available(1, potential_matched_count=evictable_count)
        expected_victim = min(
            candidate_ids,
            key=lambda block_id: (allocator.block_timestamps[block_id].item(), block_id),
        )
        evicted_reuse = allocator.allocate_memory_blocks(1)
        assert evicted_reuse is not None
        assert evicted_reuse.item() == expected_victim
        allocator.release_memory_blocks(evicted_reuse)
        allocator.release_memory_blocks(eviction_filler)

    # Epoch-style invalidation makes every retained LRU block undiscoverable and
    # returns it to the same physical pool for the next generation.
    allocator.invalidate_prefix_cache()
    _assert_allocator_state(allocator)
    assert allocator.pool_avail == allocator.pool_size - 1
    return shared_followers, reference_transitions


@pytest.mark.internal
@pytest.mark.parametrize(
    "case_id,row",
    generated_matrix_cases("allocator_execution", _MANIFEST["matrices"]["allocator_execution"]),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_allocator_execution_pairwise_stress(case_id, row):
    """Every row runs three real cache generations under forced pool pressure."""
    del case_id
    allocator = KVBlockAllocator(
        _context(),
        pool_size=_POOL_SIZE,
        paused_limit=4,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=_policy(row["eviction_policy"]),
    )
    _assert_policy_layout(allocator)

    max_followers = 0
    reference_transitions = 0
    for cycle in range(3):
        followers, transitions = _run_cycle(allocator, row, cycle)
        max_followers = max(max_followers, followers)
        reference_transitions += transitions

    if row["arrival_pattern"] == "serial":
        assert max_followers == 1
    else:
        assert max_followers == 64
    assert reference_transitions >= 8
    assert allocator.physical_block_reuse_count >= 3
    assert allocator.deregistered_block_count >= 3
    if row["eviction_policy"] == "lru":
        assert allocator.lru_evicted_block_count >= 3
    else:
        assert allocator.lru_evicted_block_count == 0


_NAMED_LRU_TOPOLOGIES = [
    pytest.param(
        {
            "hashes": [10, 20, 30],
            "parents": [0, 10, 20],
            "timestamps": [1, 1, 5],
            "evict": 1,
            "expected_hashes": {10, 20},
        },
        id="never-orphan-child",
    ),
    pytest.param(
        {
            "hashes": [10, 20, 30],
            "parents": [0, 10, 20],
            "timestamps": [1, 1, 5],
            "evict": 2,
            "expected_hashes": {10},
        },
        id="cascade-up-chain",
    ),
    pytest.param(
        {
            "hashes": [10, 20, 30],
            "parents": [0, 10, 20],
            "timestamps": [9, 9, 3],
            "evict": 1,
            "expected_hashes": {10, 20},
        },
        id="oldest-leaf-first",
    ),
    pytest.param(
        {
            "hashes": [10, 20, 30],
            "parents": [0, 10, 10],
            "timestamps": [1, 2, 8],
            "evict": 2,
            "expected_hashes": {10},
        },
        id="branching-parent-last",
    ),
    pytest.param(
        {
            "hashes": [10, 20, 30],
            "parents": [0, 10, 0],
            "timestamps": [0, 1, 9],
            "references": [1, 0, 0],
            "evict": 2,
            "expected_hashes": {10},
            "post_evict_failure": True,
        },
        id="pinned-parent-is-root",
    ),
    pytest.param(
        {
            "hashes": [10, 20, 30],
            "parents": [0, 10, 20],
            "timestamps": [1, 2, 3],
            "evict": 2,
            "expected_hashes": {10},
        },
        id="partial-chain-peel",
    ),
    pytest.param(
        {
            "hashes": [10, 20],
            "parents": [0, 10],
            "timestamps": [1, 2],
            "evict": 3,
            "expected_hashes": {10, 20},
            "expected_success": False,
        },
        id="insufficient-cache-is-atomic",
    ),
    pytest.param(
        {
            "hashes": [10, 20, 30, 40, 50, 60],
            "parents": [0, 10, 20, 20, 10, 50],
            "timestamps": [1, 2, 5, 3, 3, 5],
            "evict": 3,
            "expected_hashes": {10, 50, 60},
        },
        id="hot-leaf-over-cold-interior",
    ),
    pytest.param(
        {
            "hashes": [10, 20],
            "parents": [20, 10],
            "timestamps": [1, 2],
            "evict": 1,
            "expected_hashes": {10, 20},
            "expected_assertion": True,
        },
        id="cyclic-parent-graph-fails",
    ),
    pytest.param(None, id="generated-forest-200"),
]


def _seed_lru_topology(topology):
    allocator = KVBlockAllocator(
        _context(),
        pool_size=len(topology["hashes"]) + 1,
        paused_limit=1,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
    )
    blocks = allocator.allocate_memory_blocks(len(topology["hashes"]))
    assert blocks is not None and allocator.pool_avail == 0
    allocator.register_kv_block_hashes(blocks.tolist(), topology["hashes"], topology["parents"])
    references = topology.get("references", [0] * len(topology["hashes"]))
    cached_positions = [
        position for position, reference_count in enumerate(references) if reference_count == 0
    ]
    if cached_positions:
        allocator.release_memory_blocks(blocks[cached_positions])
    allocator.block_timestamps[blocks] = torch.tensor(topology["timestamps"], dtype=torch.int64)
    _assert_allocator_state(allocator)
    return allocator, blocks


def _run_named_lru_topology(topology):
    for _ in range(3):
        allocator, _ = _seed_lru_topology(topology)
        before_map = dict(allocator.kv_hash_to_block_id)
        before_pool_avail = allocator.pool_avail

        if topology.get("expected_assertion", False):
            with pytest.raises(AssertionError, match="parent graph is not a forest"):
                allocator.allocate_memory_blocks(topology["evict"])
            assert allocator.kv_hash_to_block_id == before_map
            assert allocator.pool_avail == before_pool_avail
        else:
            expected_success = topology.get("expected_success", True)
            reused = allocator.allocate_memory_blocks(topology["evict"])
            assert (reused is not None) is expected_success
            assert set(allocator.kv_hash_to_block_id) == topology["expected_hashes"]
            if expected_success:
                evicted_ids = set(before_map.values()) - set(allocator.kv_hash_to_block_id.values())
                assert len(evicted_ids) == topology["evict"]
                assert set(reused.tolist()) == evicted_ids
                assert allocator.physical_block_reuse_count == topology["evict"]
                assert torch.all(allocator.block_hashes[reused] == -1)
                allocator.release_memory_blocks(reused)
            else:
                assert allocator.kv_hash_to_block_id == before_map
                assert allocator.pool_avail == before_pool_avail

        if topology.get("post_evict_failure", False):
            assert allocator.evict_lru_blocks(1) is False
        _assert_allocator_state(allocator)


def _reference_leaf_peel(block_ids, hashes, parents, timestamps, cached_block_ids, requested_count):
    """Independent scalar leaf peel over only zero-reference registered blocks."""
    cached = set(cached_block_ids)
    hash_to_id = dict(zip(hashes, block_ids))
    timestamp_by_id = dict(zip(block_ids, timestamps))
    child_count = {block_id: 0 for block_id in cached}
    parent_by_id = {}
    for block_id, parent_hash in zip(block_ids, parents):
        if block_id not in cached:
            continue
        parent_id = hash_to_id.get(parent_hash)
        if parent_id not in cached:
            parent_id = None
        parent_by_id[block_id] = parent_id
        if parent_id is not None:
            child_count[parent_id] += 1

    leaves = [
        (timestamp_by_id[block_id], block_id)
        for block_id, count in child_count.items()
        if count == 0
    ]
    heapq.heapify(leaves)
    evicted = set()
    while leaves and len(evicted) < requested_count:
        _, block_id = heapq.heappop(leaves)
        evicted.add(block_id)
        parent_id = parent_by_id[block_id]
        if parent_id is not None:
            child_count[parent_id] -= 1
            if child_count[parent_id] == 0:
                heapq.heappush(leaves, (timestamp_by_id[parent_id], parent_id))
    return evicted


def _run_generated_lru_forests():
    generator = torch.Generator().manual_seed(17)
    request_shapes = ("zero", "one", "half", "all", "impossible")
    pinned_fractions = (0.0, 0.25, 0.75)

    for case_index in range(200):
        block_count = 2 + case_index % 127
        allocator = KVBlockAllocator(
            _context(),
            pool_size=block_count + 1,
            paused_limit=1,
            enable_prefix_caching=True,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
        )
        blocks = allocator.allocate_memory_blocks(block_count)
        assert blocks is not None and allocator.pool_avail == 0
        block_ids = blocks.tolist()
        hashes = [100_000 * (case_index + 1) + index + 1 for index in range(block_count)]
        parents = []
        for index in range(block_count):
            is_root = index == 0 or int(torch.randint(0, 3, (), generator=generator)) == 0
            if is_root:
                parents.append(0)
            else:
                parent_index = int(torch.randint(0, index, (), generator=generator))
                parents.append(hashes[parent_index])
        allocator.register_kv_block_hashes(block_ids, hashes, parents)

        pinned_count = int(block_count * pinned_fractions[case_index % 3])
        cached_blocks = blocks[pinned_count:]
        allocator.release_memory_blocks(cached_blocks)
        timestamps = torch.randperm(block_count, generator=generator).add(1)
        allocator.block_timestamps[blocks] = timestamps
        _assert_allocator_state(allocator)

        cached_count = cached_blocks.numel()
        request_shape = request_shapes[case_index % len(request_shapes)]
        requested_count = {
            "zero": 0,
            "one": 1,
            "half": max(1, cached_count // 2),
            "all": cached_count,
            "impossible": cached_count + 1,
        }[request_shape]
        expected_evicted = _reference_leaf_peel(
            block_ids,
            hashes,
            parents,
            timestamps.tolist(),
            cached_blocks.tolist(),
            min(requested_count, cached_count),
        )
        before_map = dict(allocator.kv_hash_to_block_id)

        if request_shape == "impossible":
            assert allocator.allocate_memory_blocks(requested_count) is None
            assert allocator.kv_hash_to_block_id == before_map
            assert allocator.pool_avail == 0
        elif request_shape == "zero":
            assert allocator.evict_lru_blocks(0) is True
            assert allocator.kv_hash_to_block_id == before_map
            assert allocator.pool_avail == 0
        else:
            reused = allocator.allocate_memory_blocks(requested_count)
            assert reused is not None
            retained_ids = set(allocator.kv_hash_to_block_id.values())
            assert retained_ids == set(block_ids) - expected_evicted
            assert allocator.lru_evicted_block_count == requested_count
            assert set(reused.tolist()) == expected_evicted
            assert allocator.physical_block_reuse_count == requested_count
            allocator.release_memory_blocks(reused)
        _assert_allocator_state(allocator)


@pytest.mark.internal
@pytest.mark.parametrize("topology", _NAMED_LRU_TOPOLOGIES)
def test_lru_forest_property_stress(topology):
    """Replay named regressions and 200 generated full-pool forest cases."""
    if topology is None:
        _run_generated_lru_forests()
    else:
        _run_named_lru_topology(topology)


@pytest.mark.internal
@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(PrefixCachingEvictionPolicy.REF_ZERO, id="ref-zero"),
        pytest.param(PrefixCachingEvictionPolicy.LRU, id="lru"),
    ],
)
def test_epoch_invalidation_lifecycle_stress(policy):
    """Invalidate live and cached blocks, then rebuild on recycled IDs for three epochs."""
    allocator = KVBlockAllocator(
        _context(),
        pool_size=4,
        paused_limit=1,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=policy,
    )
    _assert_policy_layout(allocator)

    for cycle in range(3):
        hash_base = 10_000 * (cycle + 1)
        old_hashes = [hash_base + 1, hash_base + 2, hash_base + 3]
        blocks = allocator.allocate_memory_blocks(3)
        assert blocks is not None and allocator.pool_avail == 0
        version_before_register = allocator.cache_state_version
        allocator.register_kv_block_hashes(
            blocks.tolist(), old_hashes, [0, old_hashes[0], old_hashes[1]]
        )
        assert allocator.cache_state_version == version_before_register + 1

        # Repeated and empty writes do not report a discoverable-set change.
        allocator.register_kv_block_hashes([blocks[0].item()], [old_hashes[0]], [0])
        allocator.register_kv_block_hashes([], [])
        assert allocator.cache_state_version == version_before_register + 1
        with pytest.raises(AssertionError):
            allocator.register_kv_block_hashes(
                [blocks[0].item()], [old_hashes[0]], [0, old_hashes[0]]
            )
        assert allocator.cache_state_version == version_before_register + 1

        matched = _longest_match(allocator, old_hashes)
        assert matched == blocks.tolist()
        allocator.block_ref_counts[blocks] += 1
        allocator.block_ref_counts[blocks[2]] += 1
        allocator.release_memory_blocks(blocks)
        assert not allocator.is_memory_available(1)
        assert allocator.allocate_memory_blocks(1) is None
        allocator.release_memory_blocks(blocks[1:2])

        callback_calls = []
        allocator.on_blocks_deregistered = lambda block_ids, hashes: callback_calls.append(
            (set(block_ids), set(hashes))
        )
        expected_invalidated_ids = (
            {blocks[0].item(), blocks[2].item()}
            if policy == PrefixCachingEvictionPolicy.REF_ZERO
            else set(blocks.tolist())
        )
        expected_invalidated_hashes = {
            old_hashes[blocks.tolist().index(block_id)] for block_id in expected_invalidated_ids
        }
        version_before_invalidation = allocator.cache_state_version
        assert allocator.invalidate_prefix_cache() == len(expected_invalidated_ids)
        assert allocator.cache_state_version == version_before_invalidation + 1
        assert callback_calls == [(expected_invalidated_ids, expected_invalidated_hashes)]
        assert allocator.epoch_invalidated_block_count == len(expected_invalidated_ids) * (
            cycle + 1
        )
        assert allocator.kv_hash_to_block_id == {}
        assert _longest_match(allocator, old_hashes) == []
        assert allocator.pool_avail == 1
        _assert_allocator_state(allocator)

        allocator.on_blocks_deregistered = None
        allocator.release_memory_blocks(
            torch.tensor([blocks[0].item(), blocks[2].item(), blocks[2].item()], dtype=torch.int32)
        )
        assert allocator.pool_avail == allocator.pool_size - 1

        reused = allocator.allocate_memory_blocks(3)
        assert reused is not None
        assert set(reused.tolist()) == set(blocks.tolist())
        assert torch.all(allocator.block_hashes[reused] == -1)
        assert allocator.physical_block_reuse_count == 3 * (cycle + 1)
        new_hashes = [hash_base + 101, hash_base + 102, hash_base + 103]
        allocator.register_kv_block_hashes(
            reused.tolist(), new_hashes, [0, new_hashes[0], new_hashes[1]]
        )
        assert _longest_match(allocator, new_hashes) == reused.tolist()

        version_before_reset = allocator.cache_state_version
        allocator.reset()
        assert allocator.cache_state_version == version_before_reset + 1
        allocator.reset()
        assert allocator.cache_state_version == version_before_reset + 1
        _assert_allocator_state(allocator)


def _paused_rows_for_topology(blocks, paused_topology):
    if paused_topology == "shared_prefix":
        return [[blocks[0], blocks[index + 1]] for index in range(4)]
    if paused_topology == "uneven_tail":
        return [[blocks[0], blocks[1]], [blocks[2], blocks[3]], [blocks[4], blocks[5]], [blocks[6]]]
    if paused_topology == "sparse_singletons":
        return [[blocks[index]] for index in range(4)]
    assert paused_topology == "disjoint_pairs"
    return [
        [blocks[0], blocks[1]],
        [blocks[2], blocks[3]],
        [blocks[4], blocks[5]],
        [blocks[6], blocks[7]],
    ]


def _retained_request_count(request_rows, paused_limit):
    seen = set()
    retained = 0
    for block_ids in request_rows:
        seen.update(block_ids)
        if len(seen) > paused_limit:
            break
        retained += 1
    return retained


def _releasable_suffix_count(request_rows, block_ref_counts, suffix_count):
    selected = Counter(
        block_id
        for block_ids in request_rows[len(request_rows) - suffix_count :]
        for block_id in block_ids
    )
    return sum(
        selected_count == block_ref_counts[block_id]
        for block_id, selected_count in selected.items()
    )


class TestPausedRequestRebalance(_PrefixCachingTestBase):
    """Generated real-context cases for paused eviction and reactivation."""

    def _build_paused_context(self, row, cycle):
        policy = _policy(row["eviction_policy"])
        paused_limit = {"zero": 0, "insufficient": 1, "exact": 2}[row["paused_budget"]]
        ctx = self._ctx(
            rounder=1,
            buffer_size_gb=0.01,
            block_size_tokens=16,
            max_sequence_length=128,
            max_tokens=64,
            max_requests=8,
            prefix_caching_eviction_policy=policy,
        )
        ctx.kv_block_allocator = KVBlockAllocator(
            ctx,
            pool_size=9,
            paused_limit=paused_limit,
            enable_prefix_caching=True,
            prefix_caching_eviction_policy=policy,
        )
        allocator = ctx.kv_block_allocator
        allocated = allocator.allocate_memory_blocks(8)
        assert allocated is not None and allocator.pool_avail == 0
        blocks = allocated.tolist()
        request_rows = _paused_rows_for_topology(blocks, row["paused_topology"])

        assigned_counts = Counter(block_id for block_ids in request_rows for block_id in block_ids)
        for block_id in blocks:
            allocator.block_ref_counts[block_id] = assigned_counts.get(block_id, 1)
        assigned_blocks = sorted(assigned_counts)
        hashes = [100_000 * (cycle + 1) + block_id + 1 for block_id in assigned_blocks]
        allocator.register_kv_block_hashes(
            assigned_blocks, hashes, parent_hashes=[0] * len(assigned_blocks)
        )

        ctx.total_request_count = len(request_rows)
        ctx.paused_request_count = len(request_rows)
        ctx.request_ids[:4] = torch.arange(
            10 * (cycle + 1), 10 * (cycle + 1) + 4, dtype=torch.int32
        )
        ctx.request_to_kv_block_ids[:4] = -1
        for request_idx, block_ids in enumerate(request_rows):
            ctx.request_to_kv_block_ids[request_idx, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )
            ctx.request_kv_block_counts[request_idx] = len(block_ids)
            ctx.request_last_kv_block_id[request_idx] = block_ids[-1]
        ctx.request_last_kv_block_offset[:4] = ctx.block_size_tokens - 1
        assert allocator.get_active_used() == 0
        assert allocator.get_paused_used() == len(assigned_counts)
        return ctx, request_rows

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "case_id,row",
        generated_matrix_cases(
            "paused_rebalance_local", _MANIFEST["matrices"]["paused_rebalance_local"]
        ),
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_paused_rebalance_pairwise_stress(self, case_id, row):
        """Three full-pool generations must evict exactly the scalar-minimal suffix."""
        del case_id
        total_mutations = 0
        total_reference_transitions = 0
        for cycle in range(3):
            ctx, request_rows = self._build_paused_context(row, cycle)
            allocator = ctx.kv_block_allocator
            assert allocator.pool_avail == 0
            assert allocator.get_allocatable_count() == 0

            retained = _retained_request_count(request_rows, allocator.paused_limit)
            overflow = len(request_rows) - retained
            refs = allocator.block_ref_counts.tolist()
            expected_evict_count = None
            for candidate in range(overflow + 1):
                survivor_count = overflow - candidate
                survivor_new_blocks = survivor_count
                releasable = _releasable_suffix_count(request_rows, refs, candidate)
                if survivor_new_blocks <= releasable:
                    expected_evict_count = candidate
                    break
            assert expected_evict_count is not None
            expected_ids = ctx.request_ids[
                len(request_rows) - expected_evict_count : len(request_rows)
            ].tolist()

            evicted = ctx.evict_overflow_paused_requests(
                active_request_count=0,
                next_tokens=torch.arange(len(request_rows), dtype=torch.int64),
            )
            actual_ids = [] if evicted is None else evicted.tolist()
            assert actual_ids == expected_ids

            released_capacity = _releasable_suffix_count(request_rows, refs, expected_evict_count)
            remaining_request_count = len(request_rows) - expected_evict_count
            expected_resume_count = min(remaining_request_count, released_capacity)
            active_count, newly_paused = ctx.resume_paused_requests(0, None)
            assert newly_paused is None
            assert active_count == expected_resume_count
            assert ctx.paused_request_count == remaining_request_count - expected_resume_count
            assert ctx.total_request_count == remaining_request_count
            assert torch.all(
                ctx.request_kv_block_counts[ctx.paused_request_count : ctx.total_request_count] >= 2
            )
            paused_ids = ctx.request_to_kv_block_ids[: ctx.paused_request_count]
            active_ids = ctx.request_to_kv_block_ids[
                ctx.paused_request_count : ctx.total_request_count
            ]
            expected_paused_used = (
                len(set(paused_ids[paused_ids >= 0].tolist())) if paused_ids.numel() else 0
            )
            expected_active_used = (
                len(set(active_ids[active_ids >= 0].tolist())) if active_ids.numel() else 0
            )
            assert allocator.get_paused_used() == expected_paused_used
            assert allocator.get_active_used() == expected_active_used

            # If a row evicted every overflow request, force one allocation from
            # the released LRU set so the policy still executes its mutation path.
            if (
                row["eviction_policy"] == "lru"
                and allocator.lru_evicted_block_count == 0
                and allocator.get_evictable_block_count().item() > 0
            ):
                probe = allocator.allocate_memory_blocks(1)
                assert probe is not None
                allocator.release_memory_blocks(probe)

            mutations = allocator.deregistered_block_count + allocator.lru_evicted_block_count
            assert mutations > 0
            total_mutations += mutations
            total_reference_transitions += sum(map(len, request_rows))

        assert total_mutations >= 3
        assert total_reference_transitions >= 3
