# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Microbenchmark DSA-GQA stages against explicit baseline backends."""

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from megatron.core.transformer.experimental_attention_variant import dsa_gqa
from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    _build_shifted_causal_mask,
    _gather_block_cache_sequence,
    grouped_dsa_fn,
    paged_grouped_dsa_fn,
    triton_grouped_dsa_fn,
    triton_paged_grouped_dsa_fn,
    unfused_grouped_dsa_fn,
)


@dataclass
class BenchConfig:
    batch_size: int
    block_size_tokens: int
    dtype: torch.dtype
    head_dim: int
    hidden_size: int
    index_head_dim: int
    index_heads: int
    iters: int
    key_length: int
    num_query_groups: int
    num_query_heads: int
    query_length: int
    sequence_length: int
    topk: int
    warmup: int


def _time_ms(fn: Callable[[], object], config: BenchConfig, device: torch.device) -> float:
    if device.type == "cuda":
        for _ in range(config.warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(config.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return statistics.median(times)

    for _ in range(config.warmup):
        fn()
    times = []
    for _ in range(config.iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def _make_config(args: argparse.Namespace) -> BenchConfig:
    if args.quick:
        args.sequence_length = min(args.sequence_length, 128)
        args.key_length = min(args.key_length, 512)
        args.query_length = min(args.query_length, 2)
        args.hidden_size = min(args.hidden_size, 1024)
        args.index_heads = min(args.index_heads, 8)
        args.index_head_dim = min(args.index_head_dim, 64)
        args.num_query_heads = min(args.num_query_heads, 8)
        args.num_query_groups = min(args.num_query_groups, 2)
        args.head_dim = min(args.head_dim, 64)
        args.topk = min(args.topk, 16)
        args.iters = min(args.iters, 10)
        args.warmup = min(args.warmup, 3)
    return BenchConfig(
        batch_size=args.batch_size,
        block_size_tokens=args.block_size_tokens,
        dtype=getattr(torch, args.dtype),
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        index_head_dim=args.index_head_dim,
        index_heads=args.index_heads,
        iters=args.iters,
        key_length=args.key_length,
        num_query_groups=args.num_query_groups,
        num_query_heads=args.num_query_heads,
        query_length=args.query_length,
        sequence_length=args.sequence_length,
        topk=args.topk,
        warmup=args.warmup,
    )


def _projection_inputs(config: BenchConfig, rows: int, batch_size: int, device: torch.device):
    hidden = torch.randn(rows, batch_size, config.hidden_size, device=device, dtype=config.dtype)
    linear_q_weight = torch.randn(
        config.index_heads * config.index_head_dim,
        config.hidden_size,
        device=device,
        dtype=config.dtype,
    ) / math.sqrt(config.hidden_size)
    linear_k_weight = torch.randn(
        config.index_head_dim, config.hidden_size, device=device, dtype=config.dtype
    ) / math.sqrt(config.hidden_size)
    linear_weight_proj = torch.randn(
        config.index_heads, config.hidden_size, device=device, dtype=config.dtype
    ) / math.sqrt(config.hidden_size)
    return hidden, linear_q_weight, linear_k_weight, linear_weight_proj


def _project_indexer(
    hidden: torch.Tensor,
    linear_q_weight: torch.Tensor,
    linear_k_weight: torch.Tensor,
    linear_weight_proj: torch.Tensor,
    config: BenchConfig,
):
    q_index = F.linear(hidden, linear_q_weight).reshape(
        hidden.size(0), hidden.size(1), config.index_heads, config.index_head_dim
    )
    k_index = F.layer_norm(F.linear(hidden, linear_k_weight), (config.index_head_dim,)).reshape(
        hidden.size(0), hidden.size(1), config.index_head_dim
    )
    weights = F.linear(hidden, linear_weight_proj)
    weights = weights * (config.index_heads**-0.5) * (config.index_head_dim**-0.5)
    return q_index, k_index, weights


def _require_backend(name: str):
    fn = getattr(dsa_gqa, name, None)
    if fn is None:
        raise RuntimeError(f"Requested backend requires {name}, but it is not available in this checkout.")
    return fn


def _score_topk_static(
    backend: str,
    q_index: torch.Tensor,
    k_index: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    config: BenchConfig,
):
    if backend in ("torch", "dispatch"):
        return fused_qk_topk_naive(q_index, k_index, weights, config.topk, mask)
    raise RuntimeError(f"{backend} score/top-k backend is only valid for decode profiles.")


def _score_topk_decode(
    backend: str,
    q_index: torch.Tensor,
    dsa_key_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    weights: torch.Tensor,
    query_start_position: int,
    config: BenchConfig,
):
    if backend == "torch":
        request_index_key = _gather_block_cache_sequence(
            dsa_key_cache, block_table_row, config.key_length, config.block_size_tokens
        ).unsqueeze(1)
        mask = _build_shifted_causal_mask(
            q_index.size(0), config.key_length, query_start_position, q_index.device
        )
        return fused_qk_topk_naive(q_index, request_index_key, weights, config.topk, mask)
    if backend == "dispatch":
        fn = getattr(dsa_gqa, "paged_gqa_dsa_indexer_topk_fn", None)
        if fn is None:
            return _score_topk_decode(
                "torch",
                q_index,
                dsa_key_cache,
                block_table_row,
                weights,
                query_start_position,
                config,
            )
        return None, fn(
            q_index,
            dsa_key_cache,
            block_table_row,
            weights,
            config.topk,
            query_start_position,
            config.key_length,
            config.block_size_tokens,
        )
    if backend == "triton-paged":
        fn = _require_backend("triton_paged_gqa_dsa_indexer_topk_fn")
        return None, fn(
            q_index,
            dsa_key_cache,
            block_table_row,
            weights,
            config.topk,
            query_start_position,
            config.key_length,
            config.block_size_tokens,
        )
    raise RuntimeError(f"Unknown score/top-k backend: {backend}")


def _attention_static(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    config: BenchConfig,
):
    scale = config.head_dim**-0.5
    if backend == "torch":
        return unfused_grouped_dsa_fn(query, key, value, topk_indices, scale)
    if backend == "dispatch":
        return grouped_dsa_fn(query, key, value, topk_indices, scale)
    if backend == "triton":
        return triton_grouped_dsa_fn(query, key, value, topk_indices, scale)
    if backend == "triton-gqa-tiled":
        fn = _require_backend("triton_grouped_dsa_tiled_fn")
        return fn(query, key, value, topk_indices, scale)
    raise RuntimeError(f"{backend} attention backend is not valid for static profiles.")


def _attention_decode(
    backend: str,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    topk_indices: torch.Tensor,
    query_start_position: int,
    config: BenchConfig,
):
    scale = config.head_dim**-0.5
    if backend == "torch":
        request_key = _gather_block_cache_sequence(
            key_cache, block_table_row, config.key_length, config.block_size_tokens
        ).unsqueeze(1)
        request_value = _gather_block_cache_sequence(
            value_cache, block_table_row, config.key_length, config.block_size_tokens
        ).unsqueeze(1)
        mask = _build_shifted_causal_mask(
            query.size(0), config.key_length, query_start_position, query.device
        )
        return unfused_grouped_dsa_fn(query, request_key, request_value, topk_indices, scale, mask=mask)
    if backend == "dispatch":
        return paged_grouped_dsa_fn(
            query,
            key_cache,
            value_cache,
            block_table_row,
            topk_indices,
            scale,
            query_start_position,
            config.key_length,
            config.block_size_tokens,
        )
    if backend == "triton-paged":
        return triton_paged_grouped_dsa_fn(
            query,
            key_cache,
            value_cache,
            block_table_row,
            topk_indices,
            scale,
            query_start_position,
            config.key_length,
            config.block_size_tokens,
        )
    if backend == "triton-gqa-tiled":
        fn = _require_backend("triton_paged_grouped_dsa_tiled_fn")
        return fn(
            query,
            key_cache,
            value_cache,
            block_table_row,
            topk_indices,
            scale,
            query_start_position,
            config.key_length,
            config.block_size_tokens,
        )
    raise RuntimeError(f"{backend} attention backend is not valid for decode profiles.")


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def _print_result(title: str, result: dict, baseline: dict | None = None):
    print(f"\n{title}")
    total = result["projection_ms"] + result["score_topk_ms"] + result["attention_ms"]
    if "indexer_cache_ms" in result:
        total += result["indexer_cache_ms"]
    result["total_ms"] = total

    for key in ("projection_ms", "indexer_cache_ms", "score_topk_ms", "attention_ms", "total_ms"):
        if key in result:
            print(f"  {key}: {result[key]:.4f}")

    if baseline is not None:
        baseline_total = baseline["total_ms"]
        print(f"  speedup_vs_baseline: {baseline_total / total:.3f}x")
        print(f"  topk_match: {result['topk_match']}")
        print(f"  output_max_abs_diff: {result['output_max_abs_diff']:.6g}")


def run_static(
    score_backend: str,
    attention_backend: str,
    config: BenchConfig,
    device: torch.device,
    baseline: dict | None = None,
):
    torch.manual_seed(1234)
    hidden, linear_q_weight, linear_k_weight, linear_weight_proj = _projection_inputs(
        config, config.sequence_length, config.batch_size, device
    )
    query = torch.randn(
        config.sequence_length,
        config.batch_size,
        config.num_query_heads,
        config.head_dim,
        device=device,
        dtype=config.dtype,
    )
    key = torch.randn(
        config.sequence_length,
        config.batch_size,
        config.num_query_groups,
        config.head_dim,
        device=device,
        dtype=config.dtype,
    )
    value = torch.randn_like(key)
    mask = torch.triu(
        torch.full(
            (config.sequence_length, config.sequence_length),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        ),
        diagonal=1,
    )

    q_index, k_index, weights = _project_indexer(
        hidden, linear_q_weight, linear_k_weight, linear_weight_proj, config
    )
    _, topk_indices = _score_topk_static(score_backend, q_index, k_index, weights, mask, config)
    output = _attention_static(attention_backend, query, key, value, topk_indices, config)

    result = {
        "projection_ms": _time_ms(
            lambda: _project_indexer(hidden, linear_q_weight, linear_k_weight, linear_weight_proj, config),
            config,
            device,
        ),
        "score_topk_ms": _time_ms(
            lambda: _score_topk_static(score_backend, q_index, k_index, weights, mask, config),
            config,
            device,
        ),
        "attention_ms": _time_ms(
            lambda: _attention_static(attention_backend, query, key, value, topk_indices, config),
            config,
            device,
        ),
        "topk_indices": topk_indices,
        "output": output,
    }
    if baseline is not None:
        result["topk_match"] = torch.equal(result["topk_indices"], baseline["topk_indices"])
        result["output_max_abs_diff"] = _max_abs_diff(result["output"], baseline["output"])
    return result


def _make_block_cache(sequence: torch.Tensor, block_table_row: torch.Tensor, block_size_tokens: int):
    num_blocks = int(block_table_row.max().item()) + 1
    cache = torch.zeros(
        num_blocks,
        block_size_tokens,
        *sequence.shape[1:],
        device=sequence.device,
        dtype=sequence.dtype,
    )
    for token_idx in range(sequence.size(0)):
        block_id = block_table_row[token_idx // block_size_tokens]
        local_idx = token_idx % block_size_tokens
        cache[block_id, local_idx] = sequence[token_idx]
    return cache


def run_decode(
    score_backend: str,
    attention_backend: str,
    cache_backend: str,
    config: BenchConfig,
    device: torch.device,
    baseline: dict | None = None,
):
    if cache_backend == "paged-direct" and score_backend == "torch":
        raise RuntimeError("paged-direct cache backend requires dispatch or triton-paged score/top-k.")

    torch.manual_seed(5678)
    hidden, linear_q_weight, linear_k_weight, linear_weight_proj = _projection_inputs(
        config, config.query_length, 1, device
    )
    q_index, _, weights = _project_indexer(
        hidden, linear_q_weight, linear_k_weight, linear_weight_proj, config
    )
    dsa_key_sequence = torch.randn(
        config.key_length, config.index_head_dim, device=device, dtype=config.dtype
    )
    num_blocks = math.ceil(config.key_length / config.block_size_tokens)
    block_table_row = torch.arange(num_blocks, device=device, dtype=torch.long)
    dsa_key_cache = _make_block_cache(dsa_key_sequence, block_table_row, config.block_size_tokens)
    key_sequence = torch.randn(
        config.key_length, config.num_query_groups, config.head_dim, device=device, dtype=config.dtype
    )
    value_sequence = torch.randn_like(key_sequence)
    key_cache = _make_block_cache(key_sequence, block_table_row, config.block_size_tokens)
    value_cache = _make_block_cache(value_sequence, block_table_row, config.block_size_tokens)
    query = torch.randn(
        config.query_length,
        1,
        config.num_query_heads,
        config.head_dim,
        device=device,
        dtype=config.dtype,
    )
    query_start_position = config.key_length - config.query_length

    _, topk_indices = _score_topk_decode(
        score_backend,
        q_index,
        dsa_key_cache,
        block_table_row,
        weights,
        query_start_position,
        config,
    )
    output = _attention_decode(
        attention_backend,
        query,
        key_cache,
        value_cache,
        block_table_row,
        topk_indices,
        query_start_position,
        config,
    )

    result = {
        "projection_ms": _time_ms(
            lambda: _project_indexer(hidden, linear_q_weight, linear_k_weight, linear_weight_proj, config),
            config,
            device,
        ),
        "indexer_cache_ms": 0.0
        if cache_backend == "paged-direct"
        else _time_ms(
            lambda: _gather_block_cache_sequence(
                dsa_key_cache, block_table_row, config.key_length, config.block_size_tokens
            ),
            config,
            device,
        ),
        "score_topk_ms": _time_ms(
            lambda: _score_topk_decode(
                score_backend,
                q_index,
                dsa_key_cache,
                block_table_row,
                weights,
                query_start_position,
                config,
            ),
            config,
            device,
        ),
        "attention_ms": _time_ms(
            lambda: _attention_decode(
                attention_backend,
                query,
                key_cache,
                value_cache,
                block_table_row,
                topk_indices,
                query_start_position,
                config,
            ),
            config,
            device,
        ),
        "topk_indices": topk_indices,
        "output": output,
    }
    if baseline is not None:
        result["topk_match"] = torch.equal(result["topk_indices"], baseline["topk_indices"])
        result["output_max_abs_diff"] = _max_abs_diff(result["output"], baseline["output"])
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-backend", default="dispatch", choices=["torch", "dispatch", "triton", "triton-paged", "triton-gqa-tiled"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size-tokens", type=int, default=64)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--index-head-dim", type=int, default=128)
    parser.add_argument("--index-heads", type=int, default=32)
    parser.add_argument("--indexer-cache-backend", default="gather", choices=["gather", "paged-direct"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--key-length", type=int, default=8192)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--profile", default="both", choices=["static", "decode", "both"])
    parser.add_argument("--query-length", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--score-topk-backend", default="dispatch", choices=["torch", "dispatch", "triton-paged"])
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    config = _make_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.dtype == torch.bfloat16 and device.type == "cpu":
        config.dtype = torch.float32
    print(f"device: {device}")
    print(f"dtype: {config.dtype}")

    if args.profile in ("static", "both"):
        baseline = None
        if args.compare_baseline:
            baseline = run_static("torch", "torch", config, device)
            _print_result("static baseline score=torch attention=torch", baseline)
        result = run_static(args.score_topk_backend, args.attention_backend, config, device, baseline)
        _print_result(
            f"static score={args.score_topk_backend} attention={args.attention_backend}",
            result,
            baseline,
        )

    if args.profile in ("decode", "both"):
        baseline = None
        if args.compare_baseline:
            baseline = run_decode("torch", "torch", "gather", config, device)
            _print_result("decode baseline score=torch cache=gather attention=torch", baseline)
        result = run_decode(
            args.score_topk_backend,
            args.attention_backend if args.attention_backend != "triton" else "dispatch",
            args.indexer_cache_backend,
            config,
            device,
            baseline,
        )
        _print_result(
            "decode "
            f"score={args.score_topk_backend} cache={args.indexer_cache_backend} "
            f"attention={args.attention_backend}",
            result,
            baseline,
        )


if __name__ == "__main__":
    main()
