# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    fused_qk_topk_naive,
    rotate_activation,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_using_quantization_scales


def _repeat_grouped_key_value(key: torch.Tensor, value: torch.Tensor, num_query_heads: int):
    """Expand grouped keys/values to per-query-head layout for reference attention math."""
    num_query_groups = key.size(2)
    assert num_query_heads % num_query_groups == 0, (
        f"num_query_heads ({num_query_heads}) must be divisible by num_query_groups "
        f"({num_query_groups})."
    )
    repeat_factor = num_query_heads // num_query_groups
    if repeat_factor == 1:
        return key, value
    key = key.repeat_interleave(repeat_factor, dim=2)
    value = value.repeat_interleave(repeat_factor, dim=2)
    return key, value


def _gather_block_cache_sequence(
    cache: torch.Tensor, block_table_row: torch.Tensor, sequence_length: int, block_size_tokens: int
) -> torch.Tensor:
    """Materialize a per-request sequence from a paged block cache."""
    if sequence_length == 0:
        return cache.new_empty((0,) + cache.shape[2:])
    positions = torch.arange(sequence_length, device=cache.device, dtype=torch.long)
    block_ids = block_table_row[(positions // block_size_tokens).to(block_table_row.device)].long()
    local_positions = positions % block_size_tokens
    return cache[block_ids, local_positions]


def _build_shifted_causal_mask(
    query_length: int, key_length: int, query_start_position: int, device: torch.device
) -> torch.Tensor:
    """Build a causal mask for a query chunk that starts at a non-zero KV offset."""
    if query_length == 0 or key_length == 0:
        return torch.empty((query_length, key_length), dtype=torch.float32, device=device)
    query_positions = torch.arange(
        query_start_position, query_start_position + query_length, device=device, dtype=torch.long
    )
    key_positions = torch.arange(key_length, device=device, dtype=torch.long)
    invalid = key_positions.view(1, key_length) > query_positions.view(query_length, 1)
    return torch.zeros((query_length, key_length), dtype=torch.float32, device=device).masked_fill(
        invalid, float("-inf")
    )


def compute_gqa_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """Compute DSA indexer KL loss for grouped-query attention."""
    sq, b, np, hn = query.size()
    sk, _, ng, _ = key.size()
    assert sq == index_scores.size(1), "Query sequence length must match index_scores."
    assert sk == index_scores.size(2), "Key sequence length must match index_scores."

    if np != ng:
        assert np % ng == 0, f"num_query_heads ({np}) must be divisible by num_query_groups ({ng})."
        repeat_factor = np // ng
        key = key.repeat_interleave(repeat_factor, dim=2)

    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
    ).scatter_(-1, topk_indices, 0)

    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        index_scores = index_scores + index_mask

    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )
    return kl_per_element.sum(dim=-1).mean() * loss_coeff


def unfused_grouped_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    mask: Optional[torch.Tensor] = None,
):
    """Reference grouped-query sparse attention with one top-k set per token."""
    sq, b, np, hn = query.size()
    skv = key.size(0)

    key, value = _repeat_grouped_key_value(key, value, np)
    hnv = value.size(3)

    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, skv)
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    attention_scores = attention_scores.reshape(b, np, sq, skv)

    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    index_mask.scatter_(-1, topk_indices, 0)
    if mask is None:
        mask = torch.triu(
            torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=index_mask.device),
            diagonal=1,
        )
    index_mask = index_mask + mask.view(1, sq, skv)
    attention_scores = attention_scores + index_mask.unsqueeze(1)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    value = value.permute(1, 2, 0, 3).reshape(b * np, skv, hnv)
    attention_scores = attention_scores.reshape(b * np, sq, skv)
    output = torch.bmm(attention_scores.to(value.dtype), value)
    output = output.reshape(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
    return output.reshape(sq, b, np * hnv)


@dataclass
class DSGQAIndexerSubmodules:
    linear_q: Union[ModuleSpec, type] = None
    linear_k: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class DSGQAAttentionSubmodules:
    indexer: Union[ModuleSpec, type] = None


class DSGQAIndexer(MegatronModule):
    """Token-level DSA indexer for grouped-query attention."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSGQAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)
        self.hidden_size = config.hidden_size
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.index_rotary_dim = int(self.index_head_dim * config.rotary_percent)
        self.index_rotary_dim -= self.index_rotary_dim % 2

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.rotary_pos_emb = None
        if self.index_rotary_dim > 0:
            if config.rope_type == 'rope':
                self.rotary_pos_emb = RotaryEmbedding(
                    self.index_rotary_dim,
                    rotary_percent=1.0,
                    rotary_base=config.rotary_base,
                    cp_group=self.pg_collection.cp,
                )
            elif config.rope_type == 'yarn':
                self.rotary_pos_emb = YarnRotaryEmbedding(
                    self.index_rotary_dim,
                    rotary_base=config.rotary_base,
                    scaling_factor=config.rotary_scaling_factor,
                    original_max_position_embeddings=config.original_max_position_embeddings,
                    beta_fast=config.beta_fast,
                    beta_slow=config.beta_slow,
                    mscale=config.mscale,
                    mscale_all_dim=config.mscale_all_dim,
                    cp_group=self.pg_collection.cp,
                )

        self.linear_q = build_module(
            submodules.linear_q,
            self.hidden_size,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_k = build_module(
            submodules.linear_k,
            self.hidden_size,
            self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=config.layernorm_epsilon,
        )
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _apply_rope(self, x: torch.Tensor, use_rope: bool, packed_seq_params=None):
        if not use_rope or self.rotary_pos_emb is None or self.index_rotary_dim == 0:
            return x

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        x_nope, x_pe = torch.split(
            x, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        return torch.cat([x_nope, x_pe], dim=-1)

    def _get_dynamic_rotary_pos_emb(self, inference_context) -> Tuple[torch.Tensor, float]:
        n = inference_context.padded_active_token_count
        if n == 0:
            rotary_seq_len = 1
        else:
            rotary_seq_len = (
                int(inference_context.token_to_position_in_request[:n].max().item()) + 1
            )
        if self.config.rope_type == "rope":
            return self.rotary_pos_emb(rotary_seq_len, packed_seq=False), 1.0
        return self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

    def _apply_rope_dynamic(self, q: torch.Tensor, k: torch.Tensor, inference_context):
        if self.rotary_pos_emb is None or self.index_rotary_dim == 0:
            return q, k

        rotary_pos_emb, mscale = self._get_dynamic_rotary_pos_emb(inference_context)
        q_nope, q_pe = torch.split(
            q, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )
        k_nope, k_pe = torch.split(
            k, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )
        cu_seqlens_q, _ = inference_context.cu_query_lengths()
        q_pe = inference_context.apply_rotary_emb_query(
            q_pe,
            rotary_pos_emb,
            self.config,
            cu_seqlens_q,
            self.pg_collection.cp,
            mscale=mscale,
        )
        k_pe = inference_context.apply_rotary_emb_key(
            k_pe, rotary_pos_emb, self.config, self.pg_collection.cp, mscale=mscale
        )
        return torch.cat([q_nope, q_pe], dim=-1), torch.cat([k_nope, k_pe], dim=-1)

    def forward_before_topk(
        self,
        hidden_states: torch.Tensor,
        use_rope: bool,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )

        seqlen, batch_size, _ = hidden_states.size()

        q, _ = self.linear_q(hidden_states)
        q = q.reshape(seqlen, batch_size, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, use_rope=use_rope, packed_seq_params=packed_seq_params)

        k, _ = self.linear_k(hidden_states)
        k = self.k_norm(k)
        k = k.reshape(seqlen, batch_size, 1, self.index_head_dim)
        k = self._apply_rope(k, use_rope=use_rope, packed_seq_params=packed_seq_params)
        k = k.reshape(seqlen, batch_size, self.index_head_dim)

        if self.config.dsa_indexer_use_hadamard:
            q = rotate_activation(q)
            k = rotate_activation(k)

        weights, _ = self.linear_weights_proj(hidden_states)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale
        return q, k, weights

    def forward_with_scores(
        self,
        hidden_states: torch.Tensor,
        use_rope: bool,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert packed_seq_params is None, "Packed sequence is not supported for DSA-GQA."
        q, k, weights = self.forward_before_topk(hidden_states, use_rope, packed_seq_params)
        return fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

    def forward_before_topk_dynamic(
        self,
        hidden_states: torch.Tensor,
        use_rope: bool,
        inference_context,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )

        seqlen, batch_size, _ = hidden_states.size()
        assert batch_size == 1, "Dynamic DSA-GQA expects batch=1 flattened token layout."

        q, _ = self.linear_q(hidden_states)
        q = q.reshape(seqlen, batch_size, self.index_n_heads, self.index_head_dim)

        k, _ = self.linear_k(hidden_states)
        k = self.k_norm(k)
        k = k.reshape(seqlen, batch_size, 1, self.index_head_dim)

        if use_rope:
            q, k = self._apply_rope_dynamic(q, k, inference_context)

        k = k.reshape(seqlen, batch_size, self.index_head_dim)

        if self.config.dsa_indexer_use_hadamard:
            q = rotate_activation(q)
            k = rotate_activation(k)

        weights, _ = self.linear_weights_proj(hidden_states)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale
        return q, k, weights


class DSGQACoreAttention(MegatronModule):
    """Token-level DSA core attention for grouped-query attention."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSGQAAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.indexer = build_module(
            submodules.indexer, config=config, pg_collection=pg_collection
        )
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        use_indexer_rope: bool = False,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        assert attention_bias is None, "attention_bias is not supported for DSA-GQA."
        assert packed_seq_params is None, "Packed sequence is not supported for DSA-GQA."
        if key.size(0) != hidden_states.size(0):
            if self.config.sequence_parallel:
                raise NotImplementedError(
                    "DSA-GQA does not currently support sequence parallelism."
                )
            raise NotImplementedError(
                "DSA-GQA currently supports full-sequence attention only. Decode-time "
                "indexer caching is not implemented yet."
            )

        sq, b, _, _ = query.size()
        skv = key.size(0)

        hidden_states = hidden_states.detach()

        if attn_mask_type is not None:
            assert attn_mask_type == AttnMaskType.causal, 'Only causal mask is supported for now'
            float_mask = torch.triu(
                torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=query.device),
                diagonal=1,
            )
        else:
            assert attention_mask.shape == (b, 1, sq, skv), 'attention_mask shape mismatch'
            mask = attention_mask.squeeze(1)
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, float('-inf'))

        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0) or 0.0
        if self.training and torch.is_grad_enabled():
            q_index, k_index, weights = self.indexer.forward_before_topk(
                hidden_states, use_rope=use_indexer_rope, packed_seq_params=packed_seq_params
            )
            index_scores, topk_indices = fused_qk_topk_naive(
                q_index, k_index, weights, self.indexer.index_topk, float_mask
            )

            indexer_loss = None
            if indexer_loss_coeff > 0:
                indexer_loss = compute_gqa_dsa_indexer_loss(
                    index_scores,
                    topk_indices,
                    query.detach(),
                    key.detach(),
                    self.softmax_scale,
                    indexer_loss_coeff,
                    getattr(self.config, "dsa_indexer_use_sparse_loss", False),
                    self.indexer.pg_collection,
                )
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers,
                )

            output = unfused_grouped_dsa_fn(query, key, value, topk_indices, self.softmax_scale)
            if indexer_loss is not None:
                output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            return output

        _, topk_indices = self.indexer.forward_with_scores(
            hidden_states,
            use_rope=use_indexer_rope,
            mask=float_mask,
            packed_seq_params=packed_seq_params,
        )
        return unfused_grouped_dsa_fn(query, key, value, topk_indices, self.softmax_scale)

    def forward_dynamic(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        hidden_states: torch.Tensor,
        inference_context,
        provider_layer_number: int,
        use_indexer_rope: bool = False,
    ) -> torch.Tensor:
        assert not self.training, "Dynamic DSA-GQA inference only supports eval mode."
        assert value_cache is not None, "Dynamic DSA-GQA requires value cache."

        q_index, k_index_current, weights = self.indexer.forward_before_topk_dynamic(
            hidden_states,
            use_rope=use_indexer_rope,
            inference_context=inference_context,
        )
        inference_context.append_dsa_key_cache(provider_layer_number, k_index_current)
        dsa_key_cache, block_table = inference_context.dsa_key_cache(provider_layer_number)

        query_lengths = inference_context.active_attn_metadata["mha_metadata"].state_data["query_lengths"]
        kv_lengths = inference_context.active_attn_metadata["mha_metadata"].state_data["kv_seq_lengths"]
        kv_offsets = inference_context.request_kv_length_offsets[
            inference_context.paused_request_count : inference_context.total_request_count
        ]

        sq, b, np, _ = query.size()
        value_head_dim = value_cache.size(-1)
        output = value_cache.new_zeros((sq, b, np * value_head_dim))

        q_cursor = 0
        block_size_tokens = inference_context.block_size_tokens
        num_requests = inference_context.padded_active_request_count

        for request_idx in range(num_requests):
            query_length = int(query_lengths[request_idx].item())
            if query_length == 0:
                continue

            key_length = int(kv_lengths[request_idx].item())
            query_start = q_cursor
            query_end = q_cursor + query_length
            q_cursor = query_end

            if key_length == 0:
                continue

            block_table_row = block_table[request_idx]
            request_key = _gather_block_cache_sequence(
                key_cache, block_table_row, key_length, block_size_tokens
            ).unsqueeze(1)
            request_value = _gather_block_cache_sequence(
                value_cache, block_table_row, key_length, block_size_tokens
            ).unsqueeze(1)
            request_index_key = _gather_block_cache_sequence(
                dsa_key_cache, block_table_row, key_length, block_size_tokens
            ).unsqueeze(1)

            request_query = query[query_start:query_end]
            request_q_index = q_index[query_start:query_end]
            request_weights = weights[query_start:query_end]
            request_offset = int(kv_offsets[request_idx].item()) if request_idx < kv_offsets.numel() else 0
            request_mask = _build_shifted_causal_mask(
                query_length, key_length, request_offset, request_query.device
            )

            _, topk_indices = fused_qk_topk_naive(
                request_q_index,
                request_index_key,
                request_weights,
                self.indexer.index_topk,
                request_mask,
            )
            output[query_start:query_end] = unfused_grouped_dsa_fn(
                request_query,
                request_key,
                request_value,
                topk_indices,
                self.softmax_scale,
                mask=request_mask,
            )

        if q_cursor != inference_context.active_token_count:
            raise RuntimeError(
                f"DSA-GQA dynamic inference consumed {q_cursor} query tokens but context has "
                f"{inference_context.active_token_count} active tokens."
            )

        if is_using_quantization_scales(self.config):
            output[inference_context.padding_slice] = 0.0

        return output


class DSGroupedSelfAttention(SelfAttention):
    """Self-attention that swaps in token-level DSA for grouped-query attention."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
    ):
        if config.experimental_attention_variant == "dsa":
            submodules = copy.copy(submodules)
            submodules.core_attention = ModuleSpec(
                module=DSGQACoreAttention,
                submodules=DSGQAAttentionSubmodules(
                    indexer=ModuleSpec(
                        module=DSGQAIndexer,
                        submodules=DSGQAIndexerSubmodules(
                            linear_q=ModuleSpec(module=TELinear),
                            linear_k=ModuleSpec(module=TELinear),
                            k_norm=ModuleSpec(module=TENorm),
                            linear_weights_proj=ModuleSpec(module=TELinear),
                        ),
                    )
                ),
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )

    def _use_indexer_rope(self, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin) -> bool:
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            return False

        position_embedding_type = getattr(self.config, "position_embedding_type", None)
        if position_embedding_type is not None:
            return position_embedding_type in ("rope", "yarn")
        return any(
            tensor is not None
            for tensor in (rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin)
        )

    def _get_core_attention_extra_kwargs(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_context,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        rotary_pos_cos_sin,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams],
    ) -> dict:
        if self.config.experimental_attention_variant != "dsa":
            return {}
        return {
            "hidden_states": hidden_states,
            "use_indexer_rope": self._use_indexer_rope(
                rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin
            ),
        }

    def _dynamic_core_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_context,
        block_table: torch.Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams],
        hidden_states: torch.Tensor,
        use_indexer_rope: bool,
    ) -> torch.Tensor:
        if self.config.experimental_attention_variant != "dsa":
            return super()._dynamic_core_attention_forward(
                query,
                key,
                value,
                attention_mask,
                inference_context,
                block_table,
                attn_mask_type,
                attention_bias,
                packed_seq_params,
                hidden_states=hidden_states,
                use_indexer_rope=use_indexer_rope,
            )
        if packed_seq_params is not None:
            raise NotImplementedError("Packed sequence is not supported for DSA-GQA dynamic inference.")
        if inference_context.using_cuda_graph_this_step():
            raise NotImplementedError("DSA-GQA dynamic inference does not yet support CUDA graphs.")

        provider_layer_number = self.layer_number - self._get_pp_layer_offset_for_inference()
        return self.core_attention.forward_dynamic(
            query=query,
            key_cache=key,
            value_cache=value,
            hidden_states=hidden_states,
            inference_context=inference_context,
            provider_layer_number=provider_layer_number,
            use_indexer_rope=use_indexer_rope,
        )
