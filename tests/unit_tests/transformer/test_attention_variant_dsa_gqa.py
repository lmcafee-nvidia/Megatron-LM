import torch

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGroupedSelfAttention,
    _build_shifted_causal_mask,
    compute_gqa_dsa_indexer_loss,
    unfused_grouped_dsa_fn,
)


class _DummyTPGroup:
    def size(self):
        return 1


class _DummyPGCollection:
    tp = _DummyTPGroup()


def test_mamba_stack_spec_uses_dsa_grouped_self_attention():
    attention_module = mamba_stack_spec.submodules.attention_layer.submodules.self_attention.module
    assert attention_module is DSGroupedSelfAttention


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
    index_scores = index_scores + causal_mask.view(1, seqlen, seqlen)
    topk_indices = index_scores.topk(topk, dim=-1).indices
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
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
    )

    assert output.shape == (seqlen, batch_size, num_heads * head_dim)
    assert output.dtype == query.dtype


def test_fused_qk_topk_naive_caps_topk_by_key_length():
    torch.manual_seed(123)

    q = torch.randn(2, 1, 4, 8, dtype=torch.float32)
    k = torch.randn(5, 1, 8, dtype=torch.float32)
    weights = torch.randn(2, 1, 4, dtype=torch.float32)

    _, topk_indices = fused_qk_topk_naive(q=q, k=k, weights=weights, index_topk=4)

    assert topk_indices.shape == (1, 2, 4)
    assert torch.all((topk_indices >= 0) & (topk_indices < 5))


def test_build_shifted_causal_mask_respects_query_offset():
    mask = _build_shifted_causal_mask(query_length=2, key_length=5, query_start_position=3, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(mask, expected)
