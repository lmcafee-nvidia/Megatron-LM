# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamic-context prefix matching across continued prefill chunks."""

from pathlib import Path

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from tests.test_utils.python_scripts.prefix_cache_coverage import (
    generated_matrix_cases,
    load_manifest,
)
from tests.unit_tests.inference.contexts.test_dynamic_prefix_caching import PrefixCachingTestBase

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = load_manifest(_REPO_ROOT / "tests/test_utils/prefix_cache_coverage.yaml")
_BLOCK_SIZE = 32
_PROMPT_BLOCK_COUNT = 4
_CYCLE_COUNT = 3

_POLICIES = {
    "lru": PrefixCachingEvictionPolicy.LRU,
    "ref_zero": PrefixCachingEvictionPolicy.REF_ZERO,
}
_CHUNK_CUTS = {"before": _BLOCK_SIZE - 1, "exact": _BLOCK_SIZE, "after": _BLOCK_SIZE + 1}


class TestChunkContinuationStress(PrefixCachingTestBase):
    """Exercise the hidden continuation row on every local boundary pair."""

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "case_id,row",
        generated_matrix_cases(
            "context_continuation", _MANIFEST["matrices"]["context_continuation"]
        ),
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_continuation_discovers_additional_cached_blocks(self, case_id, row):
        """A later chunk must acquire the next cached blocks without duplicating them."""
        del case_id
        speculative_token_count = 2 if row["speculation"] == "mtp" else 0
        ctx = self._ctx(
            block_size_tokens=_BLOCK_SIZE,
            max_sequence_length=8 * _BLOCK_SIZE,
            max_requests=8,
            max_tokens=16 * _BLOCK_SIZE,
            num_speculative_tokens=speculative_token_count,
            prefix_caching_eviction_policy=_POLICIES[row["eviction_policy"]],
        )
        first_chunk_length = _CHUNK_CUTS[row["chunk_cut"]]
        second_chunk_length = 2 * _BLOCK_SIZE

        total_matched_blocks = 0
        total_reference_transitions = 0
        for cycle in range(_CYCLE_COUNT):
            if cycle:
                ctx.reset()

            prompt = self._prompt(_PROMPT_BLOCK_COUNT * _BLOCK_SIZE, offset=cycle * 10_000)
            seed = self._req(ctx, prompt.clone(), request_id=2 * cycle)
            follower = self._req(ctx, prompt.clone(), request_id=2 * cycle + 1)
            ctx.add_request(seed)
            seed_blocks = self._block_ids(ctx, 0, _PROMPT_BLOCK_COUNT)
            assert len(set(seed_blocks)) == _PROMPT_BLOCK_COUNT

            hits_before = ctx.prefix_cache_hits
            blocks_before = ctx.prefix_cache_blocks_matched
            ctx.chunked_prefill_request_id = follower.request_id
            ctx.add_request(follower, prefill_chunk_length=first_chunk_length)

            first_block_count = (first_chunk_length + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            assert self._block_ids(ctx, 1, first_block_count) == seed_blocks[:first_block_count]
            assert ctx.request_query_lengths[1].item() == first_chunk_length
            assert ctx.request_kv_length_offsets[1].item() == 0

            active_mask = torch.ones(2, dtype=torch.int32, device='cpu')
            new_tokens = torch.tensor([100 + cycle, 200 + cycle], dtype=torch.int64, device='cpu')
            new_speculative_tokens = None
            if speculative_token_count:
                new_speculative_tokens = torch.tensor(
                    [[300 + cycle, 400 + cycle], [500 + cycle, 600 + cycle]],
                    dtype=torch.int64,
                    device='cpu',
                )
            ctx.update_requests(
                active_mask, new_tokens, new_speculative_tokens=new_speculative_tokens
            )

            # The engine hides an unfinished chunk just beyond total_request_count,
            # then add_request restores that same row for the continuation.
            assert ctx.total_request_count == 1
            assert ctx.request_ids[1].item() == follower.request_id
            follower.remaining_prompt_tokens = follower.remaining_prompt_tokens[first_chunk_length:]
            follower.finished_chunk_token_count = first_chunk_length
            continuation_token_start = ctx.active_token_count
            ctx.add_request(follower, prefill_chunk_length=second_chunk_length)

            already_allocated = (first_chunk_length + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            overall_required = (
                first_chunk_length + second_chunk_length + _BLOCK_SIZE - 1
            ) // _BLOCK_SIZE
            newly_matched = seed_blocks[already_allocated:overall_required]
            assert len(newly_matched) == 2
            assert (
                self._block_ids(ctx, 1, overall_required)[already_allocated:overall_required]
                == newly_matched
            )

            expected_skip = _BLOCK_SIZE if row["chunk_cut"] == "exact" else 0
            assert ctx.request_query_lengths[1].item() == second_chunk_length - expected_skip
            assert ctx.request_kv_length_offsets[1].item() == first_chunk_length + expected_skip
            assert ctx.request_kv_block_counts[1].item() == overall_required
            assert (
                ctx.request_last_kv_block_offset[1].item()
                == (first_chunk_length + second_chunk_length - 1) % _BLOCK_SIZE
            )
            continuation_token_end = ctx.active_token_count
            assert torch.all(
                ctx.token_to_block_idx[continuation_token_start:continuation_token_end]
                == ctx.kv_block_allocator.dummy_block_idx
            )

            touched_blocks = set(seed_blocks[:first_block_count] + newly_matched)
            for block_id in seed_blocks:
                expected_refs = 2 if block_id in touched_blocks else 1
                assert ctx.kv_block_allocator.block_ref_counts[block_id].item() == expected_refs

            matched_this_cycle = first_block_count + len(newly_matched)
            assert ctx.prefix_cache_hits - hits_before == 2
            assert ctx.prefix_cache_blocks_matched - blocks_before == matched_this_cycle
            assert follower.num_cached_tokens == matched_this_cycle * _BLOCK_SIZE
            total_matched_blocks += matched_this_cycle
            total_reference_transitions += len(touched_blocks)

        assert total_matched_blocks >= 9
        assert total_reference_transitions >= 9
