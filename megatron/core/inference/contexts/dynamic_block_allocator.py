# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
from torch import Tensor


class BlockAllocator:
    """Allocator that manages blocks of memory for the KV cache.

    This allocator is responsible for:
    - Initializing a pool of block IDs
    - Allocating blocks from the pool
    - Releasing blocks back to the pool

    Block metadata (hashes, ref_counts, timestamps) is managed by the PrefixTree
    in the context, not by this allocator.

    Args:
        context (DynamicInferenceContext): Dynamic inference context.
        total_count (int): Total number of blocks in the buffer.
        paused_count (int): Number of paused blocks in the buffer. Must be less
            than `total_count`.
    """

    def __init__(self, context: "DynamicInferenceContext", total_count: int, paused_count: int):

        self.context = context

        self.total_count = total_count
        self.total_avail = total_count - 1  # -1 for dummy_block_idx (see below)
        self.paused_count = paused_count
        self.active_count = total_count - paused_count - 1  # -1 for dummy_block_idx
        assert self.active_count >= 1  # ensures paused_count < total_count - 1
        self.dummy_block_idx = self.total_count - 1

        # Initialize block pool as a "stack" data structure
        self.block_bag = torch.arange(
            self.total_count, dtype=torch.int32, device=torch.cuda.current_device()
        )

    def __str__(self):
        return (
            f"using: total {self.get_total_used()}/{self.total_count - 1}"
            f"; active {self.get_active_used()}/{self.active_count}"
            f"; paused {self.get_paused_used()}/{self.paused_count}"
        )

    def get_total_used(self):
        """Compute number of total blocks used."""
        return self.total_count - self.total_avail - 1

    def get_active_used(self):
        """Compute number of active blocks used."""
        return (
            self.context.request_kv_block_counts[
                self.context.paused_request_count : self.context.total_request_count
            ]
            .sum()
            .item()
        )

    def get_paused_used(self):
        """Compute number of paused blocks used."""
        return (
            self.context.request_kv_block_counts[: self.context.paused_request_count].sum().item()
        )

    def get_active_avail(self):
        """Compute number of active blocks available."""
        return self.active_count - self.get_active_used()

    def get_paused_avail(self):
        """Compute number of paused blocks available."""
        return self.paused_count - self.get_paused_used()

    def is_memory_available(self, num_blocks: int) -> bool:
        """Check if memory blocks are available.

        Includes both free pool blocks and evictable cached blocks (ref_count == 0).

        Args:
            num_blocks (int): Number of blocks to check.

        Return:
            (bool) Is memory available?
        """
        # Fast path: avoid expensive evictable count computation when free pool suffices
        if self.total_avail >= num_blocks:
            return True
        # Also count evictable cached blocks from prefix tree
        evictable_count = self.context.prefix_tree.get_evictable_block_count()
        return (self.total_avail + evictable_count) >= num_blocks

    def allocate_memory_blocks(self, num_blocks: int) -> Optional[Tensor]:
        """Allocate memory blocks if available, else return None.

        Will attempt LRU eviction of cached blocks if the free pool is insufficient.

        Args:
            num_blocks (int): Number of blocks to allocate.

        Return:
            (Optional[Tensor]) Allocated block IDs.
        """
        # Try to evict cached blocks if free pool is insufficient
        if self.total_avail < num_blocks:
            blocks_needed_from_eviction = num_blocks - self.total_avail
            if not self.evict_lru_blocks(blocks_needed_from_eviction):
                return None  # Not enough blocks even after eviction

        # Now allocate from the free pool
        self.total_avail -= num_blocks
        block_ids = self.block_bag[self.total_avail : (self.total_avail + num_blocks)]
        assert num_blocks == block_ids.numel()

        # Note: ref_counts and timestamps are managed by PrefixTree
        # They will be set when blocks are inserted into the tree

        return block_ids

    def release_memory_blocks(self, blocks: Tensor) -> None:
        """Release memory blocks by decrementing reference counts.

        Blocks with ref_count == 0 remain cached (in prefix tree) for potential reuse.
        They will be evicted via LRU when space is needed.

        Args:
            blocks (Tensor): Block IDs to release.

        Return:
            None
        """
        if blocks.numel() == 0:
            return

        # Decrement reference counts via prefix tree - blocks stay cached for prefix reuse
        block_ids_list = blocks.tolist()
        self.context.prefix_tree.decrement_ref_counts(block_ids_list)

    def reset(self) -> None:
        """Reset the allocator to initial state.

        This resets the available block count to the entire memory pool
        (except for the dummy block).
        """

        # Reset block bag to so we start consuming from the beginning of the pool
        # for UVM performance.
        # *Note*: Resetting the block bag is essential because if engine has been
        # suspended, then the block bag contains non-unique IDs since the
        # right-most IDs have been 'popped' off and are owned by the context.
        # Without resetting the block bag, context request memory will clash and
        # requests will point to each other's memory blocks, resulting in faulty
        # generations.
        self.block_bag = torch.arange(
            self.total_count, dtype=torch.int32, device=torch.cuda.current_device()
        )

        self.total_avail = self.total_count - 1

        # Reset prefix tree
        self.context.prefix_tree.reset()

    def evict_lru_blocks(self, num_blocks_needed: int) -> bool:
        """Evict LRU cached blocks to free up space in the pool.

        Evicts leaf nodes from the prefix tree with ref_count == 0,
        starting with oldest timestamps.

        Args:
            num_blocks_needed: Number of blocks to evict.

        Returns:
            True if enough blocks were evicted, False otherwise.
        """
        # Delegate to prefix tree for leaf-node eviction
        evicted_block_ids = self.context.prefix_tree.evict_leaf_nodes(num_blocks_needed)

        if len(evicted_block_ids) < num_blocks_needed:
            return False  # Not enough blocks could be evicted

        # Add evicted blocks back to free pool
        evicted_tensor = torch.tensor(
            evicted_block_ids, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.block_bag[self.total_avail : self.total_avail + len(evicted_block_ids)] = evicted_tensor
        self.total_avail += len(evicted_block_ids)

        return True

    def lookup_block_by_hash(self, block_hash: int) -> Optional[int]:
        """Look up a cached block by its hash.

        Delegates to the prefix tree.

        Args:
            block_hash: The hash value to look up.

        Returns:
            Block ID if found, None otherwise.
        """
        return self.context.prefix_tree.lookup_block_by_hash(block_hash)

    def get_block_hash(self, block_id: int) -> int:
        """Get the hash for a block.

        Delegates to the prefix tree.

        Args:
            block_id: The block ID to get hash for.

        Returns:
            Hash value (-1 if not found).
        """
        return self.context.prefix_tree.get_block_hash(block_id)

    def compute_block_hash(self, parent_hash: int, token_ids: Tensor) -> int:
        """Compute hash for a block from (parent_hash, token_ids).

        Delegates to the prefix tree.

        Args:
            parent_hash: Hash of parent block (0 for first block in sequence).
            token_ids: Token IDs in this block.

        Returns:
            Computed hash value.
        """
        return self.context.prefix_tree.compute_block_hash(parent_hash, token_ids)

    def get_block_ref_count(self, block_id: int) -> int:
        """Get the reference count for a block.

        Delegates to the prefix tree.

        Args:
            block_id: The block ID to get ref count for.

        Returns:
            Reference count (0 if not found).
        """
        return self.context.prefix_tree.get_block_ref_count(block_id)

    def get_evictable_block_count(self) -> int:
        """Get the number of evictable cached blocks.

        Delegates to the prefix tree.

        Returns:
            Number of blocks in evictable leaf nodes.
        """
        return self.context.prefix_tree.get_evictable_block_count()

    @property
    def hash_to_block_id(self):
        """Access the hash-to-block-id mapping.

        Delegates to the prefix tree.

        Returns:
            Dict mapping hash -> block_id.
        """
        return self.context.prefix_tree.hash_to_block_id

    def get_block_timestamp(self, block_id: int) -> int:
        """Get the timestamp for a block.

        Delegates to the prefix tree.

        Args:
            block_id: The block ID to get timestamp for.

        Returns:
            Timestamp (0 if not found).
        """
        return self.context.prefix_tree.get_block_timestamp(block_id)
