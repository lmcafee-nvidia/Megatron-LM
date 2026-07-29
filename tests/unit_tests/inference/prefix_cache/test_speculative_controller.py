# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Speculative-rewind stress tests coupled to the prefix-cache allocator."""

import random
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator
from megatron.core.inference.text_generation_controllers.mtp_utils_pytorch import rewind_kv_cache

_BLOCK_SIZE = 8
_FOLLOWER_COUNT = 4
_NUM_SPECULATIVE_TOKENS = 3
_POOL_SIZE = 10
_PROPERTY_CASES = 200


def _make_allocator(policy):
    context = SimpleNamespace(
        paused_request_count=0,
        total_request_count=0,
        request_kv_block_counts=torch.zeros(_FOLLOWER_COUNT, dtype=torch.int32),
        request_to_kv_block_ids=torch.full((_FOLLOWER_COUNT, 4), -1, dtype=torch.int32),
        prefix_cache_lru_clock=0,
    )
    return KVBlockAllocator(
        context,
        pool_size=_POOL_SIZE,
        paused_limit=0,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=policy,
    )


def _assert_allocator_matches_scalar_state(allocator):
    """Reconstruct the free, referenced, and registered sets independently."""

    usable_ids = set(range(allocator.pool_size - 1))
    free_ids = allocator.block_bag[: allocator.pool_avail].tolist()
    assert len(free_ids) == len(set(free_ids))
    assert set(free_ids) <= usable_ids

    owned_ids = usable_ids - set(free_ids)
    assert allocator.get_total_used() == len(owned_ids)
    assert torch.all(allocator.block_ref_counts >= 0)
    assert all(allocator.block_ref_counts[block_id].item() == 0 for block_id in free_ids)
    assert all(allocator.block_hashes[block_id].item() == -1 for block_id in free_ids)

    expected_hash_map = {
        allocator.block_hashes[block_id].item(): block_id
        for block_id in owned_ids
        if allocator.block_hashes[block_id].item() > 0
    }
    assert allocator.kv_hash_to_block_id == expected_hash_map


def _scalar_rewind(
    accepted_counts, last_offsets, length_offsets, block_counts, last_block_ids, block_tables
):
    """Independent list-based model of one speculative rewind."""

    expected = []
    for accepted, last_offset, length, count, last_block, table in zip(
        accepted_counts, last_offsets, length_offsets, block_counts, last_block_ids, block_tables
    ):
        rewind_count = _NUM_SPECULATIVE_TOKENS - accepted
        offset_delta = last_offset - rewind_count
        crosses_boundary = offset_delta < 0
        new_count = count - int(crosses_boundary)
        new_table = list(table)
        if crosses_boundary:
            new_table[new_count] = -1
        expected.append(
            {
                "last_offset": offset_delta % _BLOCK_SIZE,
                "length": length - rewind_count,
                "block_count": new_count,
                "last_block": new_table[new_count - 1] if crosses_boundary else last_block,
                "block_table": new_table,
                "released_block": last_block,
                "remove": crosses_boundary,
            }
        )
    return expected


@pytest.mark.parametrize(
    "eviction_policy",
    [PrefixCachingEvictionPolicy.REF_ZERO, PrefixCachingEvictionPolicy.LRU],
    ids=["ref_zero", "lru"],
)
def test_speculative_rewind_state_machine_stress(eviction_policy):
    """Run 200 shared-prefix rewind/release/reuse traces per eviction policy."""

    rng = random.Random(20260729 + int(eviction_policy == PrefixCachingEvictionPolicy.LRU))

    for case_index in range(_PROPERTY_CASES):
        allocator = _make_allocator(eviction_policy)
        allocator.context.prefix_cache_lru_clock = case_index + 1

        prefix_blocks = allocator.allocate_memory_blocks(2)
        assert prefix_blocks is not None
        prefix_hashes = [10_000 + case_index * 2, 10_001 + case_index * 2]
        allocator.register_kv_block_hashes(
            prefix_blocks.tolist(), prefix_hashes, [0, prefix_hashes[0]]
        )

        # Four followers acquire both published prefix blocks while their
        # speculative tails occupy distinct physical blocks.
        allocator.block_ref_counts[prefix_blocks] += _FOLLOWER_COUNT
        speculative_tail_blocks = allocator.allocate_memory_blocks(_FOLLOWER_COUNT)
        assert speculative_tail_blocks is not None
        allocator.release_memory_blocks(prefix_blocks)
        assert torch.all(allocator.block_ref_counts[prefix_blocks] == _FOLLOWER_COUNT)

        block_tables = [
            [*prefix_blocks.tolist(), int(speculative_tail_blocks[index]), -1]
            for index in range(_FOLLOWER_COUNT)
        ]
        accepted_counts = [
            0,
            _NUM_SPECULATIVE_TOKENS,
            rng.randrange(_NUM_SPECULATIVE_TOKENS + 1),
            rng.randrange(_NUM_SPECULATIVE_TOKENS + 1),
        ]
        last_offsets = [
            0,
            rng.randrange(_BLOCK_SIZE),
            rng.randrange(_BLOCK_SIZE),
            rng.randrange(_BLOCK_SIZE),
        ]
        length_offsets = [2 * _BLOCK_SIZE + offset + 1 for offset in last_offsets]
        block_counts = [3] * _FOLLOWER_COUNT
        last_block_ids = speculative_tail_blocks.tolist()

        expected = _scalar_rewind(
            accepted_counts,
            last_offsets,
            length_offsets,
            block_counts,
            last_block_ids,
            block_tables,
        )

        accepted_tensor = torch.tensor(accepted_counts, dtype=torch.int64)
        prefill_status = torch.zeros(_FOLLOWER_COUNT, dtype=torch.int32)
        last_offset_tensor = torch.tensor(last_offsets, dtype=torch.int32)
        length_tensor = torch.tensor(length_offsets, dtype=torch.int32)
        block_count_tensor = torch.tensor(block_counts, dtype=torch.int32)
        last_block_tensor = torch.tensor(last_block_ids, dtype=torch.int32)
        block_table_tensor = torch.tensor(block_tables, dtype=torch.int32)

        blocks_to_release, remove_mask = rewind_kv_cache(
            accepted_tensor,
            prefill_status,
            last_offset_tensor,
            length_tensor,
            block_count_tensor,
            last_block_tensor,
            block_table_tensor,
            _NUM_SPECULATIVE_TOKENS,
            _BLOCK_SIZE,
        )

        assert remove_mask.tolist() == [row["remove"] for row in expected]
        assert last_offset_tensor.tolist() == [row["last_offset"] for row in expected]
        assert length_tensor.tolist() == [row["length"] for row in expected]
        assert block_count_tensor.tolist() == [row["block_count"] for row in expected]
        assert last_block_tensor.tolist() == [row["last_block"] for row in expected]
        assert block_table_tensor.tolist() == [row["block_table"] for row in expected]
        assert blocks_to_release.tolist() == [row["released_block"] for row in expected]

        # Apply the controller's release result to the real prefix allocator,
        # then release the four followers in a staggered order.
        allocator.release_memory_blocks(blocks_to_release[remove_mask])
        for request_index in rng.sample(range(_FOLLOWER_COUNT), _FOLLOWER_COUNT):
            live_blocks = block_table_tensor[request_index]
            allocator.release_memory_blocks(live_blocks[live_blocks >= 0])
            _assert_allocator_matches_scalar_state(allocator)

        assert torch.all(allocator.block_ref_counts[prefix_blocks] == 0)
        if eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
            assert allocator.kv_hash_to_block_id == {}
        else:
            assert set(allocator.kv_hash_to_block_id) == set(prefix_hashes)

        # Exhaust the pool once more. LRU must peel the published child before
        # its parent; REF_ZERO has already returned both blocks directly.
        recycled = allocator.allocate_memory_blocks(_POOL_SIZE - 1)
        assert recycled is not None
        assert set(recycled.tolist()) == set(range(_POOL_SIZE - 1))
        assert allocator.kv_hash_to_block_id == {}
        assert allocator.physical_block_reuse_count >= 2 + _FOLLOWER_COUNT
        _assert_allocator_matches_scalar_state(allocator)

        allocator.release_memory_blocks(recycled)
        assert allocator.pool_avail == _POOL_SIZE - 1
        _assert_allocator_matches_scalar_state(allocator)
