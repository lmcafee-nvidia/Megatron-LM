# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from .gpu_hash_table import GPUHashTable


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

    def __init__(
        self,
        context: "DynamicInferenceContext",
        total_count: int,
        paused_count: int,
        enable_prefix_caching: bool = False,
        block_evict_lru: bool = False,
    ):

        self.context = context
        self.enable_prefix_caching = enable_prefix_caching
        self.block_evict_lru = block_evict_lru

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

        if self.enable_prefix_caching:
            device = torch.cuda.current_device()

            # Block hash tracking for prefix caching: -1 = uncomputed, positive = valid hash
            self.block_hashes = torch.full(
                (self.total_count,), -1, dtype=torch.int64, device=device
            )

            # GPU hash table for O(1) prefix lookup (replaces CPU hash_to_block_id)
            self.gpu_hash_table = GPUHashTable(max_entries=total_count, device=device)

            # Reference count per block: 0 = cached (evictable), >0 = actively used
            self.block_ref_counts = torch.zeros(
                (self.total_count,), dtype=torch.int32, device=device
            )

            # LRU timestamps for eviction ordering (higher = more recently used)
            # Only needed in LRU mode; RZ mode evicts immediately on ref_count==0
            if self.block_evict_lru:
                self.block_timestamps = torch.zeros(
                    (self.total_count,), dtype=torch.int64, device=device
                )

            # Pending block hashes for prefix caching coordination
            # -1 = not pending, positive = hash registered but KV not yet computed
            self._pending_block_hashes = torch.full(
                (self.total_count,), -1, dtype=torch.int64, device=device
            )

            # GPU pending bitmap: True if block is registered but KV not yet computed
            self.pending_bitmap = torch.zeros(self.total_count, dtype=torch.bool, device=device)

            # GPU scalar counters for blocks_with_refs and total_ref_count.
            # Used for metrics and evictable estimate (read post-forward with .item()).
            self._gpu_blocks_with_refs = torch.zeros(1, dtype=torch.int32, device=device)
            self._gpu_total_ref_count = torch.zeros(1, dtype=torch.int32, device=device)

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
        if not self.enable_prefix_caching:
            return False
        if not self.block_evict_lru:
            return False  # RZ: no cached blocks to evict

        # Exact GPU count (1 sync via get_evictable_block_count)
        evictable_count = self.get_evictable_block_count()
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
            if not self.enable_prefix_caching or not self.block_evict_lru:
                return None  # RZ: no eviction path; disabled: no cached blocks
            blocks_needed_from_eviction = num_blocks - self.total_avail
            if not self.evict_lru_blocks(blocks_needed_from_eviction):
                return None  # Not enough blocks even after eviction

        # Now allocate from the free pool
        self.total_avail -= num_blocks
        block_ids = self.block_bag[self.total_avail : (self.total_avail + num_blocks)]
        assert num_blocks == block_ids.numel()

        if self.enable_prefix_caching:
            # Initialize ref counts for newly allocated blocks
            self.block_ref_counts[block_ids] = 1
            self._gpu_blocks_with_refs += num_blocks
            self._gpu_total_ref_count += num_blocks
            if self.block_evict_lru:
                self.update_timestamps(block_ids)

        return block_ids

    def release_memory_blocks(self, blocks: Tensor) -> None:
        """Release memory blocks by decrementing reference counts.

        Blocks with ref_count == 0 remain cached (in hash map) for potential reuse.
        They will be evicted via LRU when space is needed.

        Args:
            blocks (Tensor): Block IDs to release.

        Return:
            None
        """
        if blocks.numel() == 0:
            return

        if self.enable_prefix_caching:
            self.block_ref_counts[blocks] -= 1
            self._gpu_total_ref_count -= blocks.numel()

            zero_mask = self.block_ref_counts[blocks] == 0
            # Use .sum() on GPU to count zero-ref blocks (avoids boolean indexing sync)
            num_zero = zero_mask.sum()
            self._gpu_blocks_with_refs -= num_zero

            if not self.block_evict_lru:
                # RZ: immediately deregister zero-ref blocks
                # Use torch.where to extract block IDs at fixed size, then slice
                # by the GPU-computed count (requires one sync for _deregister_blocks)
                num_zero_cpu = num_zero.item()
                if num_zero_cpu > 0:
                    zero_blocks = blocks[zero_mask]
                    self._deregister_blocks(zero_blocks)
            # LRU: blocks stay cached (hash remains) for potential reuse
        else:
            num_blocks = blocks.numel()
            self.block_bag[self.total_avail : self.total_avail + num_blocks] = blocks
            self.total_avail += num_blocks

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

        if self.enable_prefix_caching:
            # Reset all block hashes
            self.block_hashes.fill_(-1)

            # Reset prefix caching state
            self.gpu_hash_table.clear()
            self._pending_block_hashes.fill_(-1)
            self.pending_bitmap.fill_(False)
            self.block_ref_counts.fill_(0)
            self._gpu_blocks_with_refs.fill_(0)
            self._gpu_total_ref_count.fill_(0)
            if self.block_evict_lru:
                self.block_timestamps.fill_(0)

    # =========================================================================
    # Prefix caching methods
    # =========================================================================

    def register_block_hashes(self, block_ids: list[int], block_hashes: list[int]) -> None:
        """Register blocks in the hash-to-block mapping for discovery (batch).

        NOTE: Does NOT mark blocks as computed. Call mark_blocks_computed() after
        KV is computed. This two-phase approach enables prefix caching coordination
        where subsequent requests wait for blocks to be computed before reusing.

        Args:
            block_ids: List of block IDs.
            block_hashes: List of computed hash values (same length as block_ids).
        """
        if not block_ids:
            return
        device = self._pending_block_hashes.device
        id_tensor = torch.tensor(block_ids, dtype=torch.int64, device=device)
        hash_tensor = torch.tensor(block_hashes, dtype=torch.int64, device=device)
        self._pending_block_hashes[id_tensor] = hash_tensor

        # GPU hash table insert
        id_tensor_i32 = torch.tensor(block_ids, dtype=torch.int32, device=device)
        self.gpu_hash_table.insert_batch(hash_tensor, id_tensor_i32)
        self.pending_bitmap[id_tensor] = True

    def register_block_hashes_gpu(self, block_ids: Tensor, block_hashes: Tensor) -> None:
        """Register blocks using GPU tensors directly (no CPU conversion).

        GPU-resident version of register_block_hashes(). Both inputs are
        already GPU tensors, avoiding CPU list construction.

        Args:
            block_ids: int32 GPU tensor of block IDs.
            block_hashes: int64 GPU tensor of hash values (same length).
        """
        if block_ids.numel() == 0:
            return
        id_tensor_i64 = block_ids.to(torch.int64)
        self._pending_block_hashes[id_tensor_i64] = block_hashes
        self.gpu_hash_table.insert_batch(block_hashes, block_ids.to(torch.int32))
        self.pending_bitmap[id_tensor_i64] = True

    def increment_ref_counts(self, matched_block_ids: list[int], matched_tensor: Tensor) -> None:
        """Increment ref counts for matched prefix blocks.

        Args:
            matched_block_ids: CPU list of matched block IDs.
            matched_tensor: GPU tensor of matched block IDs (same content).
        """
        num_matched = len(matched_block_ids)
        if num_matched == 0:
            return

        # In LRU mode, some matched blocks might be cached (ref=0 → ref=1)
        if self.block_evict_lru:
            newly_active = (self.block_ref_counts[matched_tensor] == 0).sum()
            self._gpu_blocks_with_refs += newly_active

        self._gpu_total_ref_count += num_matched
        self.block_ref_counts[matched_tensor] += 1

        if self.block_evict_lru:
            self.update_timestamps(matched_tensor)

    def increment_ref_counts_gpu(self, matched_tensor: Tensor) -> None:
        """Increment ref counts for matched prefix blocks (GPU-only version).

        GPU-resident version that avoids the CPU list parameter.

        Args:
            matched_tensor: GPU tensor of matched block IDs.
        """
        num_matched = matched_tensor.numel()
        if num_matched == 0:
            return

        if self.block_evict_lru:
            newly_active = (self.block_ref_counts[matched_tensor] == 0).sum()
            self._gpu_blocks_with_refs += newly_active

        self._gpu_total_ref_count += num_matched
        self.block_ref_counts[matched_tensor] += 1

        if self.block_evict_lru:
            self.update_timestamps(matched_tensor)

    def mark_blocks_computed(self, block_ids: list[int]) -> None:
        """Mark blocks as having their KV computed (batch).

        Called after prefill completes for blocks that were registered.
        This sets block_hashes[block_id] to the actual hash value,
        signaling that the KV cache for these blocks is ready for reuse.

        Args:
            block_ids: List of block IDs to mark as computed.
        """
        if not block_ids:
            return
        id_tensor = torch.tensor(
            block_ids, dtype=torch.int64, device=self._pending_block_hashes.device
        )
        self.block_hashes[id_tensor] = self._pending_block_hashes[id_tensor]
        self._pending_block_hashes[id_tensor] = -1
        self.pending_bitmap[id_tensor] = False

    def mark_blocks_computed_gpu(self, block_ids: Tensor) -> None:
        """Mark blocks as computed using a GPU tensor (no CPU conversion).

        GPU-resident version of mark_blocks_computed().

        Args:
            block_ids: int32 GPU tensor of block IDs to mark as computed.
        """
        if block_ids.numel() == 0:
            return
        id_tensor_i64 = block_ids.to(torch.int64)
        self.block_hashes[id_tensor_i64] = self._pending_block_hashes[id_tensor_i64]
        self._pending_block_hashes[id_tensor_i64] = -1
        self.pending_bitmap[id_tensor_i64] = False

    def _deregister_blocks(self, block_ids: Tensor) -> None:
        """Remove blocks from prefix caching state and return to free pool.

        Shared cleanup logic for both LRU eviction and RZ proactive eviction.
        All operations are GPU-resident tensor ops + hash table rebuild.

        Args:
            block_ids: Tensor of block IDs to deregister.
        """
        num_blocks = block_ids.numel()
        if num_blocks == 0:
            return

        # Reset GPU state (batched tensor ops)
        block_ids_i64 = block_ids.to(torch.int64)
        self._pending_block_hashes[block_ids_i64] = -1
        self.pending_bitmap[block_ids_i64] = False
        self.block_hashes[block_ids] = -1
        self.block_ref_counts[block_ids] = 0
        if self.block_evict_lru:
            self.block_timestamps[block_ids] = 0

        # Return blocks to free pool
        self.block_bag[self.total_avail : self.total_avail + num_blocks] = block_ids
        self.total_avail += num_blocks

        # Rebuild GPU hash table from remaining valid entries.
        # This is the "rebuild after eviction" strategy — simpler and correct
        # compared to per-element deletion with linear probing.
        # Must include both computed blocks (block_hashes != -1) and pending
        # blocks (_pending_block_hashes != -1) since both are discoverable.
        computed_mask = self.block_hashes != -1
        pending_mask = self._pending_block_hashes != -1

        # Computed blocks: hash from block_hashes
        computed_ids = torch.nonzero(computed_mask, as_tuple=True)[0]
        # Pending blocks: hash from _pending_block_hashes
        pending_ids = torch.nonzero(pending_mask, as_tuple=True)[0]

        if computed_ids.numel() > 0 or pending_ids.numel() > 0:
            all_ids_list = []
            all_hashes_list = []
            if computed_ids.numel() > 0:
                all_ids_list.append(computed_ids.to(torch.int32))
                all_hashes_list.append(self.block_hashes[computed_ids])
            if pending_ids.numel() > 0:
                all_ids_list.append(pending_ids.to(torch.int32))
                all_hashes_list.append(self._pending_block_hashes[pending_ids])

            all_ids = torch.cat(all_ids_list) if len(all_ids_list) > 1 else all_ids_list[0]
            all_hashes = (
                torch.cat(all_hashes_list) if len(all_hashes_list) > 1 else all_hashes_list[0]
            )
            self.gpu_hash_table.rebuild(all_hashes, all_ids)
        else:
            self.gpu_hash_table.clear()

    def update_timestamps(self, block_ids: Tensor) -> None:
        """Update LRU timestamps for accessed blocks. No-op in RZ mode.

        Args:
            block_ids: Tensor of block IDs that were accessed.
        """
        if not self.block_evict_lru or block_ids.numel() == 0:
            return
        self.block_timestamps[block_ids] = self.context.step_count

    def get_evictable_block_count(self) -> Tensor:
        """Get count of cached blocks that can be evicted (ref_count == 0, hash set).

        Returns:
            Scalar tensor with the number of evictable cached blocks.
        """
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        return cached_mask.sum()

    def evict_lru_blocks(self, num_blocks_needed: int) -> bool:
        """Evict LRU cached blocks to free up space in the pool.

        Evicts blocks with ref_count == 0, starting with oldest timestamps.

        Args:
            num_blocks_needed: Number of blocks to evict.

        Returns:
            True if enough blocks were evicted, False otherwise.
        """
        # Find all cached blocks (ref_count == 0, hash != -1)
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)

        # nonzero() syncs to determine output size; .numel() reads shape (no sync).
        # Combined into one sync instead of separate sum() + nonzero().
        cached_block_ids = torch.nonzero(cached_mask, as_tuple=True)[0]
        if cached_block_ids.numel() < num_blocks_needed:
            return False  # Not enough cached blocks to evict

        # Sort by timestamp (ascending = oldest first)
        cached_timestamps = self.block_timestamps[cached_block_ids]
        sorted_indices = torch.argsort(cached_timestamps)
        blocks_to_evict = cached_block_ids[sorted_indices[:num_blocks_needed]]

        self._deregister_blocks(blocks_to_evict)

        return True
