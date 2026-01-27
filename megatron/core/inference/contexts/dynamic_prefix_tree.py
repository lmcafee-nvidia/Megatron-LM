# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class BlockMetadata:
    """Metadata for a single block in the prefix tree."""

    block_id: int
    hash: int


class PrefixNode:
    """Node in prefix tree containing blocks and child references.

    Each node represents a sequence of blocks that share a common prefix path.
    The root node has no blocks and serves as the entry point for traversal.

    For hybrid (Mamba) models, nodes also store mamba states at the end of the
    last block in the node. These states enable skipping computation for cached
    prefix tokens when a new request matches this prefix.
    """

    def __init__(
        self,
        parent: Optional['PrefixNode'] = None,
        children: Optional[Dict[int, 'PrefixNode']] = None,
        blocks: Optional[List[BlockMetadata]] = None,
        mamba_conv_states: Optional[Tensor] = None,
        mamba_ssm_states: Optional[Tensor] = None,
        ref_count: int = 0,
        timestamp: int = 0,
    ):
        self.parent = parent
        self.children = children if children is not None else {}
        self.blocks = blocks if blocks is not None else []
        # Mamba states at the end of the last block in this node (hybrid models only)
        # Shape: [num_mamba_layers, *state_shape]
        self.mamba_conv_states = mamba_conv_states
        self.mamba_ssm_states = mamba_ssm_states
        # Reference count and LRU timestamp (formerly on BlockMetadata)
        self.ref_count = ref_count
        self.timestamp = timestamp

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this node is the root (no parent)."""
        return self.parent is None

    def get_all_block_ids(self) -> List[int]:
        """Get all block IDs in this node."""
        return [b.block_id for b in self.blocks]


class PrefixTree:
    """Radix tree for prefix caching block metadata.

    The tree structure enables efficient prefix matching and block sharing:
    - Root node has no blocks, serves as entry point
    - Each non-root node contains a sequence of blocks
    - Children are indexed by the hash of their first block
    - When divergence occurs mid-node, the node is split

    Attributes:
        root: Root node of the tree (no parent, no blocks)
        hash_to_block_id: O(1) lookup from hash to block_id
        block_id_to_metadata: O(1) lookup from block_id to BlockMetadata
        global_timestamp: Counter for LRU ordering
    """

    def __init__(self, mamba_state_bytes_per_node: int = 0):
        self.root = PrefixNode()
        self.hash_to_block_id: Dict[int, int] = {}
        self.block_id_to_metadata: Dict[int, BlockMetadata] = {}
        self.block_id_to_node: Dict[int, PrefixNode] = {}
        self.global_timestamp: int = 0
        # Mamba state memory tracking
        self.mamba_state_bytes_per_node = mamba_state_bytes_per_node
        self.mamba_state_memory_bytes: int = 0

    def reset(self) -> None:
        """Reset tree to initial empty state."""
        self.root = PrefixNode()
        self.hash_to_block_id.clear()
        self.block_id_to_metadata.clear()
        self.block_id_to_node.clear()
        self.global_timestamp = 0
        self.mamba_state_memory_bytes = 0

    def lookup_block_by_hash(self, block_hash: int) -> Optional[int]:
        """Look up block ID by content hash.

        Args:
            block_hash: Hash of the block content.

        Returns:
            Block ID if found, None otherwise.
        """
        return self.hash_to_block_id.get(block_hash)

    def compute_block_hash(self, parent_hash: int, token_ids: Tensor) -> int:
        """Compute hash for a block from (parent_hash, token_ids).

        Uses a GPU-based polynomial rolling hash combined with the parent hash.

        Args:
            parent_hash: Hash of parent block (0 for first block in sequence).
            token_ids: Token IDs in this block, shape [block_size_tokens].

        Returns:
            Computed hash value (always positive).
        """
        HASH_PRIME = 1000000007
        HASH_BASE = 31

        # Compute polynomial hash of token IDs on GPU
        block_size = token_ids.shape[0]
        positions = torch.arange(block_size, device=token_ids.device, dtype=torch.int64)
        powers = torch.pow(
            torch.tensor(HASH_BASE, device=token_ids.device, dtype=torch.int64),
            block_size - 1 - positions,
        )
        token_hash = (token_ids.to(torch.int64) * powers).sum().item() % HASH_PRIME

        # Combine with parent hash for position-dependent hashing
        combined = (parent_hash * HASH_BASE + token_hash) % HASH_PRIME

        # Ensure positive hash (add 1 since 0 could indicate "no hash")
        return combined + 1

    def find_matching_prefix(
        self, prompt_tokens: Tensor, block_size: int
    ) -> Tuple[List[int], int, PrefixNode, int, Optional[Tensor], Optional[Tensor]]:
        """Find cached blocks matching the prompt prefix.

        Traverses the tree following block hashes until divergence or end.

        Args:
            prompt_tokens: All prompt tokens for the request.
            block_size: Number of tokens per block.

        Returns:
            Tuple of:
                - matched_block_ids: List of block IDs that match the prefix
                - parent_hash: Hash of last matched block (for computing next hash)
                - insertion_node: Node where new blocks should be inserted
                - insertion_idx: Block index within node for insertion/split
                - matched_mamba_conv_states: Mamba conv states at end of matched prefix (or None)
                - matched_mamba_ssm_states: Mamba SSM states at end of matched prefix (or None)
        """
        matched_block_ids: List[int] = []
        parent_hash = 0
        current_node = self.root
        block_idx_in_node = 0
        # Track the last node whose ALL blocks were matched (for mamba state retrieval)
        last_fully_matched_node: Optional[PrefixNode] = None

        num_complete_blocks = len(prompt_tokens) // block_size

        for block_pos in range(num_complete_blocks):
            start = block_pos * block_size
            end = start + block_size
            block_tokens = prompt_tokens[start:end]

            # Compute hash for this block
            block_hash = self.compute_block_hash(parent_hash, block_tokens)

            # Check if we're still traversing within current node's blocks
            if block_idx_in_node < len(current_node.blocks):
                existing_block = current_node.blocks[block_idx_in_node]
                if existing_block.hash == block_hash:
                    # Match within current node
                    matched_block_ids.append(existing_block.block_id)
                    parent_hash = block_hash
                    block_idx_in_node += 1
                    # Check if we just finished matching all blocks in this node
                    if block_idx_in_node == len(current_node.blocks):
                        last_fully_matched_node = current_node
                    continue
                else:
                    # Divergence within current node - need to split
                    # Return mamba state from last fully matched node (not current node)
                    conv_states, ssm_states = self._get_mamba_states(last_fully_matched_node)
                    return (matched_block_ids, parent_hash, current_node, block_idx_in_node,
                            conv_states, ssm_states)

            # Finished current node's blocks, check for child with this hash
            if block_hash in current_node.children:
                # Move to child node
                current_node = current_node.children[block_hash]
                block_idx_in_node = 0

                # First block of child should match
                if current_node.blocks and current_node.blocks[0].hash == block_hash:
                    matched_block_ids.append(current_node.blocks[0].block_id)
                    parent_hash = block_hash
                    block_idx_in_node = 1
                    # Check if this single-block child is now fully matched
                    if block_idx_in_node == len(current_node.blocks):
                        last_fully_matched_node = current_node
                    continue
                else:
                    # Child exists but hash mismatch (shouldn't happen in valid tree)
                    conv_states, ssm_states = self._get_mamba_states(last_fully_matched_node)
                    return (matched_block_ids, parent_hash, current_node, 0,
                            conv_states, ssm_states)
            else:
                # No matching child - this is where we insert
                conv_states, ssm_states = self._get_mamba_states(last_fully_matched_node)
                return (matched_block_ids, parent_hash, current_node, len(current_node.blocks),
                        conv_states, ssm_states)

        # All blocks matched
        conv_states, ssm_states = self._get_mamba_states(last_fully_matched_node)
        return (matched_block_ids, parent_hash, current_node, block_idx_in_node,
                conv_states, ssm_states)

    def _get_mamba_states(
        self, node: Optional[PrefixNode]
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Get mamba states from a node if available.

        Args:
            node: PrefixNode to get states from (or None).

        Returns:
            Tuple of (mamba_conv_states, mamba_ssm_states), both None if not available.
        """
        if node is None:
            return None, None
        return node.mamba_conv_states, node.mamba_ssm_states

    def insert_blocks(
        self,
        block_metadatas: List[BlockMetadata],
        insertion_node: PrefixNode,
        insertion_idx: int,
    ) -> None:
        """Insert new blocks into the tree.

        Handles three cases:
        1. Append to existing node (insertion_idx == len(node.blocks))
        2. Split node and add as new child (insertion_idx < len(node.blocks))
        3. Add as new child of node (insertion_idx == len(node.blocks) and node is leaf)

        Args:
            block_metadatas: List of BlockMetadata for new blocks.
            insertion_node: Node where insertion should occur.
            insertion_idx: Index within node's blocks for insertion point.
        """
        if not block_metadatas:
            return

        # Register all blocks in lookup dicts (block_id_to_node set after node assignment)
        for block in block_metadatas:
            self.hash_to_block_id[block.hash] = block.block_id
            self.block_id_to_metadata[block.block_id] = block

        first_block_hash = block_metadatas[0].hash

        # Case 1: Need to split the node (divergence mid-node)
        if insertion_idx < len(insertion_node.blocks):
            self._split_node(insertion_node, insertion_idx)
            # After split, insertion_node now ends at insertion_idx
            # Add new blocks as a new child

        # Case 2: Add as child (either after split or natural end of node)
        if insertion_idx == len(insertion_node.blocks):
            # Check if there's already a child with this hash
            if first_block_hash in insertion_node.children:
                # Shouldn't happen if find_matching_prefix is correct
                # But handle gracefully by appending to existing child
                existing_child = insertion_node.children[first_block_hash]
                existing_child.blocks.extend(block_metadatas)
                # Update block_id_to_node for appended blocks
                for block in block_metadatas:
                    self.block_id_to_node[block.block_id] = existing_child
            else:
                # Create new child node with ref_count=1 (this request references it)
                self.global_timestamp += 1
                new_node = PrefixNode(
                    parent=insertion_node,
                    children={},
                    blocks=block_metadatas,
                    ref_count=1,
                    timestamp=self.global_timestamp,
                )
                insertion_node.children[first_block_hash] = new_node
                # Update block_id_to_node for new blocks
                for block in block_metadatas:
                    self.block_id_to_node[block.block_id] = new_node

    def _split_node(self, node: PrefixNode, split_idx: int) -> None:
        """Split node at split_idx.

        The node is truncated to blocks[:split_idx], and a new child is created
        with blocks[split_idx:] plus the original children.

        The new child inherits the parent's ref_count and timestamp because any
        requests still using the original node are also using the split-off blocks.

        Args:
            node: Node to split.
            split_idx: Index at which to split.
        """
        if split_idx >= len(node.blocks) or split_idx <= 0:
            return  # Nothing to split or invalid index

        # Get hash of first block in continuation (before truncating)
        continuation_hash = node.blocks[split_idx].hash

        # Create child node with remaining blocks and original children
        # Inherit ref_count and timestamp - original requests still use these blocks
        new_child = PrefixNode(
            parent=node,
            children=node.children,
            blocks=node.blocks[split_idx:],
            ref_count=node.ref_count,
            timestamp=node.timestamp,
        )

        # Update parent references in moved children
        for child in new_child.children.values():
            child.parent = new_child

        # Update block_id_to_node for blocks moved to the new child
        for block in new_child.blocks:
            self.block_id_to_node[block.block_id] = new_child

        # Truncate original node and set new child
        node.blocks = node.blocks[:split_idx]
        node.children = {continuation_hash: new_child}

    def increment_ref_counts(self, block_ids: List[int]) -> None:
        """Increment reference count for nodes containing specified blocks.

        Args:
            block_ids: List of block IDs to increment.
        """
        # Collect unique nodes to avoid incrementing the same node multiple times
        nodes = set()
        for block_id in block_ids:
            if block_id in self.block_id_to_node:
                nodes.add(self.block_id_to_node[block_id])
        for node in nodes:
            node.ref_count += 1

    def decrement_ref_counts(self, block_ids: List[int]) -> None:
        """Decrement reference count for nodes containing specified blocks.

        Args:
            block_ids: List of block IDs to decrement.
        """
        # Collect unique nodes to avoid decrementing the same node multiple times
        nodes = set()
        for block_id in block_ids:
            if block_id in self.block_id_to_node:
                nodes.add(self.block_id_to_node[block_id])
        for node in nodes:
            node.ref_count -= 1

    def update_timestamps(self, block_ids: List[int]) -> None:
        """Update timestamps for nodes containing specified blocks (LRU tracking).

        Args:
            block_ids: List of block IDs to update.
        """
        self.global_timestamp += 1
        # Collect unique nodes to avoid updating the same node multiple times
        nodes = set()
        for block_id in block_ids:
            if block_id in self.block_id_to_node:
                nodes.add(self.block_id_to_node[block_id])
        for node in nodes:
            node.timestamp = self.global_timestamp

    def get_evictable_block_count(self) -> int:
        """Count blocks in evictable leaf nodes.

        A leaf node is evictable if it has ref_count == 0.

        Returns:
            Total number of blocks in evictable leaf nodes.
        """
        count = 0
        for node in self._find_leaf_nodes():
            if node.blocks and node.ref_count == 0:
                count += len(node.blocks)
        return count

    def evict_leaf_nodes(self, num_blocks_needed: int) -> List[int]:
        """Evict oldest leaf nodes until enough blocks are freed.

        Only evicts complete leaf nodes (node must have ref_count == 0).
        Evicts in LRU order (oldest timestamp first).

        Args:
            num_blocks_needed: Number of blocks to free.

        Returns:
            List of evicted block IDs.
        """
        evicted: List[int] = []

        while len(evicted) < num_blocks_needed:
            # Find all leaf nodes (excluding root)
            leaf_nodes = self._find_leaf_nodes()

            # Filter to evictable (node ref_count == 0)
            evictable = [n for n in leaf_nodes if n.blocks and n.ref_count == 0]

            if not evictable:
                break  # No more evictable nodes

            # Sort by oldest timestamp (node-level timestamp)
            evictable.sort(key=lambda n: n.timestamp)

            # Evict oldest
            node = evictable[0]
            for block in node.blocks:
                del self.hash_to_block_id[block.hash]
                del self.block_id_to_metadata[block.block_id]
                del self.block_id_to_node[block.block_id]
                evicted.append(block.block_id)

            self._remove_node(node)

        return evicted

    def evict_for_mamba_memory(self, bytes_needed: int, memory_limit_bytes: int) -> bool:
        """Evict unused leaf nodes with mamba states until memory is under limit.

        Evicts entire nodes (not just mamba states) to maintain tree consistency
        with KV cache eviction behavior.

        Args:
            bytes_needed: Bytes about to be stored.
            memory_limit_bytes: Maximum allowed mamba state memory.

        Returns:
            True if enough memory was freed, False if unable to evict enough.
        """
        target_memory = memory_limit_bytes - bytes_needed

        while self.mamba_state_memory_bytes > target_memory:
            # Find leaf nodes with mamba states that are evictable (ref_count=0)
            evictable = [
                n for n in self._find_leaf_nodes()
                if n.mamba_conv_states is not None and n.blocks and n.ref_count == 0
            ]
            if not evictable:
                return False

            # Sort by LRU (oldest timestamp first)
            evictable.sort(key=lambda n: n.timestamp)

            # Evict oldest node entirely (removes blocks too)
            node = evictable[0]
            for block in node.blocks:
                del self.hash_to_block_id[block.hash]
                del self.block_id_to_metadata[block.block_id]
                del self.block_id_to_node[block.block_id]
            self._remove_node(node)

        return True

    def _find_leaf_nodes(self) -> List[PrefixNode]:
        """Find all leaf nodes in the tree (excluding root).

        Returns:
            List of leaf nodes.
        """
        leaves: List[PrefixNode] = []

        def traverse(node: PrefixNode) -> None:
            if node.is_leaf() and not node.is_root():
                leaves.append(node)
            for child in node.children.values():
                traverse(child)

        traverse(self.root)
        return leaves

    def _remove_node(self, node: PrefixNode) -> None:
        """Remove a leaf node from the tree.

        Frees any associated mamba states when the node is removed.

        Args:
            node: Leaf node to remove (must have no children).
        """
        if not node.is_leaf() or node.is_root():
            return  # Can only remove leaf nodes, not root

        # Update mamba memory tracking before clearing
        if node.mamba_conv_states is not None:
            self.mamba_state_memory_bytes -= self.mamba_state_bytes_per_node

        # Free mamba states (Python GC will reclaim memory)
        node.mamba_conv_states = None
        node.mamba_ssm_states = None

        if node.parent:
            # Find and remove from parent's children
            for hash_key, child in list(node.parent.children.items()):
                if child is node:
                    del node.parent.children[hash_key]
                    break

    def get_block_hash(self, block_id: int) -> int:
        """Get hash for a block by ID.

        Args:
            block_id: Block ID to look up.

        Returns:
            Hash value, or -1 if not found.
        """
        metadata = self.block_id_to_metadata.get(block_id)
        return metadata.hash if metadata else -1

    def get_block_ref_count(self, block_id: int) -> int:
        """Get reference count for a block by ID.

        Args:
            block_id: Block ID to look up.

        Returns:
            Reference count of the node containing this block, or 0 if not found.
        """
        node = self.block_id_to_node.get(block_id)
        return node.ref_count if node else 0

    def get_block_timestamp(self, block_id: int) -> int:
        """Get timestamp for a block by ID.

        Args:
            block_id: Block ID to look up.

        Returns:
            Timestamp of the node containing this block, or 0 if not found.
        """
        node = self.block_id_to_node.get(block_id)
        return node.timestamp if node else 0
