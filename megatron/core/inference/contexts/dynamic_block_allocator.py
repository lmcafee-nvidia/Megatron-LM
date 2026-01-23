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

        # Block hash tracking for prefix caching: -1 = uncomputed, positive = valid hash
        self.block_hashes = torch.full(
            (self.total_count,), -1, dtype=torch.int64, device=torch.cuda.current_device()
        )

        # Store token IDs per block for hash computation
        # Shape: [total_count, block_size_tokens]
        self.block_to_token_ids = torch.full(
            (self.total_count, context.block_size_tokens),
            -1,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
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

        Args:
            num_blocks (int): Number of blocks to check.

        Return:
            (bool) Is memory available?
        """
        return self.total_avail >= num_blocks

    def allocate_memory_blocks(self, num_blocks: int) -> Optional[Tensor]:
        """Allocate memory blocks if available, else return None.

        Args:
            num_blocks (int): Number of blocks to allocate.

        Return:
            (Optional[Tensor]) Allocated block IDs.
        """
        if self.is_memory_available(num_blocks):
            self.total_avail -= num_blocks
            block_ids = self.block_bag[self.total_avail : (self.total_avail + num_blocks)]
            assert num_blocks == block_ids.numel()
            return block_ids
        else:
            return None

    def release_memory_blocks(self, blocks: Tensor) -> None:
        """Release memory blocks.

        Args:
            blocks (Tensor): Block IDs to release.

        Return:
            None
        """
        num_blocks = blocks.size(dim=0)
        self.block_bag[self.total_avail : (self.total_avail + num_blocks)] = blocks
        self.total_avail += num_blocks

        # Reset block hashes and token storage to -1 (invalid)
        self.block_hashes[blocks] = -1
        self.block_to_token_ids[blocks] = -1

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

        # Reset all block hashes and token storage
        self.block_hashes.fill_(-1)
        self.block_to_token_ids.fill_(-1)

    # Constants for hash computation
    HASH_PRIME = 1000000007
    HASH_BASE = 31

    def compute_block_hash(self, parent_hash: int, token_ids: Tensor) -> int:
        """Compute hash for a block from (parent_hash, token_ids).

        Uses a GPU-based polynomial rolling hash combined with the parent hash.

        Args:
            parent_hash: Hash of parent block (0 for first block in sequence).
            token_ids: Token IDs in this block, shape [block_size_tokens].

        Returns:
            Positive integer hash value (1 to HASH_PRIME).
        """
        block_size = token_ids.shape[0]
        positions = torch.arange(block_size, device=token_ids.device, dtype=torch.int64)
        powers = torch.pow(self.HASH_BASE, positions).to(torch.int64) % self.HASH_PRIME
        token_hash = ((token_ids.to(torch.int64) * powers).sum() % self.HASH_PRIME).item()

        # Combine with parent hash
        combined = (parent_hash * self.HASH_BASE + token_hash) % self.HASH_PRIME
        return combined + 1  # Ensure positive (1 to HASH_PRIME)

    def set_block_hash(self, block_id: int, hash_value: int) -> None:
        """Set the hash for a specific block.

        Args:
            block_id: The block ID to set hash for.
            hash_value: The hash value to store.
        """
        self.block_hashes[block_id] = hash_value

    def get_block_hash(self, block_id: int) -> int:
        """Get the hash for a block.

        Args:
            block_id: The block ID to get hash for.

        Returns:
            Hash value (-1 if not computed).
        """
        return self.block_hashes[block_id].item()
