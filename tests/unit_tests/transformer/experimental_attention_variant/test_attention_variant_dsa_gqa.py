# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    HAVE_TRITON,
    _build_shifted_causal_mask,
    _gather_block_cache_sequence,
    compute_gqa_dsa_indexer_loss,
    grouped_dsa_fn,
    paged_gqa_dsa_indexer_topk_fn,
    paged_grouped_dsa_fn,
    triton_grouped_dsa_fn,
    triton_paged_gqa_dsa_indexer_topk_fn,
    triton_paged_grouped_dsa_fn,
    unfused_grouped_dsa_fn,
)


class _DummyTPGroup:
    def size(self):
        return 1


class _DummyPGCollection:
    tp = _DummyTPGroup()


def _fill_paged_cache_from_sequence(sequence, block_table_row, block_size_tokens, num_blocks=None):
    if num_blocks is None:
        num_blocks = int(block_table_row.max().item()) + 1
    cache = torch.zeros(
        num_blocks, block_size_tokens, *sequence.shape[1:], dtype=sequence.dtype, device=sequence.device
    )
    for token_idx in range(sequence.size(0)):
        block_id = block_table_row[token_idx // block_size_tokens]
        local_idx = token_idx % block_size_tokens
        cache[block_id, local_idx] = sequence[token_idx]
    return cache


def test_compute_gqa_dsa_indexer_loss_dense_and_sparse():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32)
    causal_mask = torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), dtype=torch.float32), diagonal=1
    )
    topk_indices = (index_scores + causal_mask).topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    dense_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores.clone(),
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=False,
        pg_collection=pg_collection,
    )
    sparse_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores.clone(),
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
    )

    assert dense_loss.ndim == 0
    assert sparse_loss.ndim == 0
    assert torch.isfinite(dense_loss)
    assert torch.isfinite(sparse_loss)


def test_unfused_grouped_dsa_fn_output_shape():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    topk_indices = torch.randint(0, seqlen, (batch_size, seqlen, topk))

    output = unfused_grouped_dsa_fn(
        query=query, key=key, value=value, topk_indices=topk_indices, softmax_scale=head_dim**-0.5
    )

    assert output.shape == (seqlen, batch_size, num_heads * head_dim)
    assert output.dtype == query.dtype


def test_grouped_dsa_fn_uses_reference_on_cpu():
    torch.manual_seed(123)

    seqlen = 5
    batch_size = 2
    num_heads = 4
    num_query_groups = 2
    head_dim = 8
    topk = 3

    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    causal_mask = torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), dtype=torch.float32), diagonal=1
    )
    topk_indices = (torch.randn(batch_size, seqlen, seqlen) + causal_mask).topk(topk, dim=-1).indices

    expected = unfused_grouped_dsa_fn(query, key, value, topk_indices, head_dim**-0.5)
    actual = grouped_dsa_fn(query, key, value, topk_indices, head_dim**-0.5)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA/Triton required")
def test_triton_grouped_dsa_fn_matches_reference():
    torch.manual_seed(123)

    seqlen = 7
    batch_size = 2
    num_heads = 4
    num_query_groups = 2
    head_dim = 16
    topk = 4

    query = torch.randn(seqlen, batch_size, num_heads, head_dim, device="cuda")
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, device="cuda")
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim, device="cuda")
    causal_mask = torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), device="cuda"), diagonal=1
    )
    topk_indices = (
        torch.randn(batch_size, seqlen, seqlen, device="cuda") + causal_mask
    ).topk(topk, dim=-1).indices

    expected = unfused_grouped_dsa_fn(query, key, value, topk_indices, head_dim**-0.5)
    actual = triton_grouped_dsa_fn(query, key, value, topk_indices, head_dim**-0.5)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA/Triton required")
def test_triton_paged_grouped_dsa_fn_matches_gather_reference():
    torch.manual_seed(123)

    sequence_length = 7
    query_length = 2
    query_start_position = 5
    block_size_tokens = 4
    num_cache_blocks = 3
    num_heads = 4
    num_query_groups = 2
    head_dim = 16
    topk = 4
    block_table_row = torch.tensor([2, 0], dtype=torch.long, device="cuda")

    sequence_key = torch.randn(sequence_length, 1, num_query_groups, head_dim, device="cuda")
    sequence_value = torch.randn(sequence_length, 1, num_query_groups, head_dim, device="cuda")
    key_cache = torch.zeros(
        num_cache_blocks, block_size_tokens, num_query_groups, head_dim, device="cuda"
    )
    value_cache = torch.zeros_like(key_cache)
    for token_idx in range(sequence_length):
        block_id = block_table_row[token_idx // block_size_tokens]
        local_idx = token_idx % block_size_tokens
        key_cache[block_id, local_idx] = sequence_key[token_idx, 0]
        value_cache[block_id, local_idx] = sequence_value[token_idx, 0]

    query = torch.randn(query_length, 1, num_heads, head_dim, device="cuda")
    request_mask = _build_shifted_causal_mask(
        query_length, sequence_length, query_start_position, query.device
    )
    topk_indices = (
        torch.randn(1, query_length, sequence_length, device="cuda") + request_mask.view(1, query_length, sequence_length)
    ).topk(topk, dim=-1).indices

    request_key = _gather_block_cache_sequence(
        key_cache, block_table_row, sequence_length, block_size_tokens
    ).unsqueeze(1)
    request_value = _gather_block_cache_sequence(
        value_cache, block_table_row, sequence_length, block_size_tokens
    ).unsqueeze(1)
    expected = grouped_dsa_fn(
        query, request_key, request_value, topk_indices, head_dim**-0.5, mask=request_mask
    )
    actual = triton_paged_grouped_dsa_fn(
        query,
        key_cache,
        value_cache,
        block_table_row,
        topk_indices,
        head_dim**-0.5,
        query_start_position,
        sequence_length,
        block_size_tokens,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA/Triton required")
def test_triton_paged_gqa_dsa_indexer_topk_matches_gather_reference():
    key_length = 7
    block_size_tokens = 4
    index_heads = 4
    index_dim = 16
    topk = 3
    query_start_position = key_length - 1
    block_table_row = torch.tensor([2, 0], dtype=torch.long, device="cuda")

    sequence_key = (
        torch.arange(1, key_length + 1, dtype=torch.float32, device="cuda")
        .view(key_length, 1)
        .expand(key_length, index_dim)
        .contiguous()
    )
    dsa_key_cache = _fill_paged_cache_from_sequence(
        sequence_key, block_table_row, block_size_tokens, num_blocks=3
    )
    q_index = torch.ones(1, 1, index_heads, index_dim, dtype=torch.float32, device="cuda")
    weights = torch.arange(1, index_heads + 1, dtype=torch.float32, device="cuda").view(
        1, 1, index_heads
    )
    request_index_key = _gather_block_cache_sequence(
        dsa_key_cache, block_table_row, key_length, block_size_tokens
    ).unsqueeze(1)
    request_mask = _build_shifted_causal_mask(1, key_length, query_start_position, q_index.device)
    _, expected = fused_qk_topk_naive(q_index, request_index_key, weights, topk, request_mask)

    actual = triton_paged_gqa_dsa_indexer_topk_fn(
        q_index,
        dsa_key_cache,
        block_table_row,
        weights,
        topk,
        query_start_position,
        key_length,
        block_size_tokens,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA/Triton required")
def test_triton_paged_gqa_dsa_indexer_topk_caps_topk_by_key_length():
    key_length = 5
    block_size_tokens = 4
    index_heads = 2
    index_dim = 8
    topk = 8
    query_start_position = key_length - 1
    block_table_row = torch.tensor([1, 0], dtype=torch.long, device="cuda")

    sequence_key = (
        torch.arange(1, key_length + 1, dtype=torch.float32, device="cuda")
        .view(key_length, 1)
        .expand(key_length, index_dim)
        .contiguous()
    )
    dsa_key_cache = _fill_paged_cache_from_sequence(
        sequence_key, block_table_row, block_size_tokens, num_blocks=2
    )
    q_index = torch.ones(1, 1, index_heads, index_dim, dtype=torch.float32, device="cuda")
    weights = torch.ones(1, 1, index_heads, dtype=torch.float32, device="cuda")
    request_index_key = _gather_block_cache_sequence(
        dsa_key_cache, block_table_row, key_length, block_size_tokens
    ).unsqueeze(1)
    request_mask = _build_shifted_causal_mask(1, key_length, query_start_position, q_index.device)
    _, expected = fused_qk_topk_naive(q_index, request_index_key, weights, topk, request_mask)

    actual = triton_paged_gqa_dsa_indexer_topk_fn(
        q_index,
        dsa_key_cache,
        block_table_row,
        weights,
        topk,
        query_start_position,
        key_length,
        block_size_tokens,
    )

    assert actual.shape == (1, 1, key_length)
    assert torch.equal(actual, expected)


def test_paged_gqa_dsa_indexer_topk_falls_back_on_cpu():
    torch.manual_seed(123)

    key_length = 5
    query_length = 2
    query_start_position = 3
    block_size_tokens = 4
    index_heads = 4
    index_dim = 8
    topk = 3
    block_table_row = torch.tensor([1, 0], dtype=torch.long)

    sequence_key = torch.randn(key_length, index_dim)
    dsa_key_cache = _fill_paged_cache_from_sequence(sequence_key, block_table_row, block_size_tokens)
    q_index = torch.randn(query_length, 1, index_heads, index_dim)
    weights = torch.randn(query_length, 1, index_heads)
    request_index_key = _gather_block_cache_sequence(
        dsa_key_cache, block_table_row, key_length, block_size_tokens
    ).unsqueeze(1)
    request_mask = _build_shifted_causal_mask(
        query_length, key_length, query_start_position, q_index.device
    )
    _, expected = fused_qk_topk_naive(q_index, request_index_key, weights, topk, request_mask)

    actual = paged_gqa_dsa_indexer_topk_fn(
        q_index,
        dsa_key_cache,
        block_table_row,
        weights,
        topk,
        query_start_position,
        key_length,
        block_size_tokens,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA/Triton required")
def test_paged_gqa_dsa_decode_dispatch_matches_gather_reference():
    key_length = 9
    query_length = 1
    query_start_position = key_length - 1
    block_size_tokens = 4
    num_heads = 4
    num_query_groups = 2
    head_dim = 16
    index_heads = 4
    index_dim = 16
    topk = 4
    block_table_row = torch.tensor([2, 0, 1], dtype=torch.long, device="cuda")

    sequence_index_key = (
        torch.arange(1, key_length + 1, dtype=torch.float32, device="cuda")
        .view(key_length, 1)
        .expand(key_length, index_dim)
        .contiguous()
    )
    dsa_key_cache = _fill_paged_cache_from_sequence(
        sequence_index_key, block_table_row, block_size_tokens, num_blocks=3
    )
    q_index = torch.ones(query_length, 1, index_heads, index_dim, dtype=torch.float32, device="cuda")
    weights = torch.arange(1, index_heads + 1, dtype=torch.float32, device="cuda").view(
        1, 1, index_heads
    )

    sequence_key = torch.randn(key_length, num_query_groups, head_dim, device="cuda")
    sequence_value = torch.randn_like(sequence_key)
    key_cache = _fill_paged_cache_from_sequence(
        sequence_key, block_table_row, block_size_tokens, num_blocks=3
    )
    value_cache = _fill_paged_cache_from_sequence(
        sequence_value, block_table_row, block_size_tokens, num_blocks=3
    )
    query = torch.randn(query_length, 1, num_heads, head_dim, device="cuda")

    request_index_key = _gather_block_cache_sequence(
        dsa_key_cache, block_table_row, key_length, block_size_tokens
    ).unsqueeze(1)
    request_mask = _build_shifted_causal_mask(
        query_length, key_length, query_start_position, query.device
    )
    _, expected_topk = fused_qk_topk_naive(
        q_index, request_index_key, weights, topk, request_mask
    )
    actual_topk = paged_gqa_dsa_indexer_topk_fn(
        q_index,
        dsa_key_cache,
        block_table_row,
        weights,
        topk,
        query_start_position,
        key_length,
        block_size_tokens,
    )

    request_key = _gather_block_cache_sequence(
        key_cache, block_table_row, key_length, block_size_tokens
    ).unsqueeze(1)
    request_value = _gather_block_cache_sequence(
        value_cache, block_table_row, key_length, block_size_tokens
    ).unsqueeze(1)
    expected = grouped_dsa_fn(
        query, request_key, request_value, expected_topk, head_dim**-0.5, mask=request_mask
    )
    actual = paged_grouped_dsa_fn(
        query,
        key_cache,
        value_cache,
        block_table_row,
        actual_topk,
        head_dim**-0.5,
        query_start_position,
        key_length,
        block_size_tokens,
    )

    assert torch.equal(actual_topk, expected_topk)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_paged_grouped_dsa_fn_falls_back_on_cpu():
    torch.manual_seed(123)

    sequence_length = 5
    query_length = 2
    query_start_position = 3
    block_size_tokens = 4
    num_heads = 4
    num_query_groups = 2
    head_dim = 8
    topk = 3
    block_table_row = torch.tensor([1, 0], dtype=torch.long)

    sequence_key = torch.randn(sequence_length, 1, num_query_groups, head_dim)
    sequence_value = torch.randn(sequence_length, 1, num_query_groups, head_dim)
    key_cache = torch.zeros(2, block_size_tokens, num_query_groups, head_dim)
    value_cache = torch.zeros_like(key_cache)
    for token_idx in range(sequence_length):
        block_id = block_table_row[token_idx // block_size_tokens]
        local_idx = token_idx % block_size_tokens
        key_cache[block_id, local_idx] = sequence_key[token_idx, 0]
        value_cache[block_id, local_idx] = sequence_value[token_idx, 0]

    query = torch.randn(query_length, 1, num_heads, head_dim)
    request_mask = _build_shifted_causal_mask(
        query_length, sequence_length, query_start_position, query.device
    )
    topk_indices = (
        torch.randn(1, query_length, sequence_length) + request_mask.view(1, query_length, sequence_length)
    ).topk(topk, dim=-1).indices

    request_key = _gather_block_cache_sequence(
        key_cache, block_table_row, sequence_length, block_size_tokens
    ).unsqueeze(1)
    request_value = _gather_block_cache_sequence(
        value_cache, block_table_row, sequence_length, block_size_tokens
    ).unsqueeze(1)
    expected = grouped_dsa_fn(
        query, request_key, request_value, topk_indices, head_dim**-0.5, mask=request_mask
    )
    actual = paged_grouped_dsa_fn(
        query,
        key_cache,
        value_cache,
        block_table_row,
        topk_indices,
        head_dim**-0.5,
        query_start_position,
        sequence_length,
        block_size_tokens,
    )

    assert torch.equal(actual, expected)


def test_fused_qk_topk_naive_caps_topk_by_key_length():
    torch.manual_seed(123)

    q = torch.randn(2, 1, 4, 8, dtype=torch.float32)
    k = torch.randn(5, 1, 8, dtype=torch.float32)
    weights = torch.randn(2, 1, 4, dtype=torch.float32)

    _, topk_indices = fused_qk_topk_naive(q=q, k=k, weights=weights, index_topk=4)

    assert topk_indices.shape == (1, 2, 4)
    assert torch.all((topk_indices >= 0) & (topk_indices < 5))


def test_build_shifted_causal_mask_respects_query_offset():
    mask = _build_shifted_causal_mask(
        query_length=2, key_length=5, query_start_position=3, device=torch.device("cpu")
    )

    expected = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, float("-inf")], [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    assert torch.equal(mask, expected)
