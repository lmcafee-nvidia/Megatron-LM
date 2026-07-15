# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Temporary differential diagnostics for async-scheduling investigation."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor

_REFERENCE_TOKENS = (
    2,
    4568,
    1584,
    89474,
    1454,
    11851,
    1261,
    6165,
    7430,
    5510,
    1455,
    9985,
    8616,
    1317,
    4292,
    1261,
    3310,
    1044,
    1321,
    1278,
    5510,
    2715,
    1850,
    1278,
)
_DEFAULT_CAPTURE_POSITIONS = frozenset((376, 377, 393, 394, 395))


@lru_cache(maxsize=1)
def _forced_tokens() -> tuple[int, ...]:
    value = os.environ.get("MCORE_ASYNC_DIAG_FORCE_TOKENS", "")
    return tuple(int(token) for token in value.split(",") if token) or _REFERENCE_TOKENS


@lru_cache(maxsize=1)
def _capture_positions() -> frozenset[int]:
    value = os.environ.get("MCORE_ASYNC_DIAG_CAPTURE_POSITIONS", "")
    return (
        frozenset(int(position) for position in value.split(",") if position)
        or _DEFAULT_CAPTURE_POSITIONS
    )


def _diagnostic_directory() -> Optional[Path]:
    run_dir = os.environ.get("MCORE_BENCH_RUN_DIR")
    if not run_dir or not _capture_positions():
        return None

    directory = Path(run_dir) / "artifacts" / "async-sched-diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _is_capture_forward(context: Any) -> bool:
    position = context._async_diag_forward_position
    if position not in _capture_positions() or not context.is_decode_only():
        return False
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return False
    return context.get_active_request_count() == 1


def _to_cpu(tensor: Tensor) -> Tensor:
    return tensor.detach().cpu().clone()


def force_async_sched_sample(context: Any, sampled_tokens: Tensor) -> None:
    """Force the configured reference token at the current inference step."""
    tokens = _forced_tokens()
    step = context.step_count
    if tokens and context.get_active_request_count() == 1 and step < len(tokens):
        sampled_tokens[:1].fill_(tokens[step])


def _capture_context(context: Any) -> dict[str, Any]:
    active_slice = slice(context.paused_request_count, context.total_request_count)
    active_token_slice = slice(0, context.active_token_count)

    request_fields = (
        "request_ids",
        "request_kv_length_offsets",
        "request_query_lengths",
        "request_output_lengths",
        "request_in_prefill_status_tensor",
        "request_kv_block_counts",
        "request_last_kv_block_id",
        "request_last_kv_block_offset",
        "request_to_kv_block_ids",
    )
    token_fields = (
        "token_to_input_ids",
        "token_to_pos_ids",
        "token_to_request_idx",
        "token_to_position_in_request",
        "token_to_local_position_within_kv_block",
        "token_to_block_idx",
    )

    payload: dict[str, Any] = {
        "step": context.step_count,
        "forward_position": context._async_diag_forward_position,
        "mode": context.config.async_sched_mode.value,
        "counts": {
            "total_request_count": context.total_request_count,
            "paused_request_count": context.paused_request_count,
            "active_token_count": context.active_token_count,
            "num_prefill_requests": context.num_prefill_requests,
            "num_decode_requests": context.num_decode_requests,
            "padded_active_token_count": context.padded_active_token_count,
            "padded_active_request_count": context.padded_active_request_count,
        },
        "requests": {
            name: _to_cpu(context.__dict__[name][active_slice]) for name in request_fields
        },
        "request_metadata": {
            name: _to_cpu(tensor[active_slice])
            for name, tensor in context.request_metadata.items()
        },
        "tokens": {
            name: _to_cpu(context.__dict__[name][active_token_slice]) for name in token_fields
        },
        "cpu_bookkeeping": _to_cpu(context._cpu_bookkeeping_buf),
        "gpu_bookkeeping": _to_cpu(context.gpu_view._buf),
    }

    block_counts = context.request_kv_block_counts[active_slice].tolist()
    block_rows = context.request_to_kv_block_ids[active_slice]
    block_ids = sorted(
        {
            int(block_id)
            for row, count in zip(block_rows, block_counts)
            for block_id in row[:count].tolist()
            if block_id >= 0
        }
    )
    payload["kv_block_ids"] = block_ids
    if block_ids:
        if context.cache_mla_latent:
            payload["kv_state"] = _to_cpu(context.memory_buffer[:, block_ids])
        else:
            payload["kv_state"] = _to_cpu(context.memory_buffer[:, :, block_ids])

    if context.is_hybrid_model:
        mamba_state_indices = context.mamba_metadata.request_to_mamba_state_idx[
            active_slice
        ].tolist()
        valid_state_indices = sorted({int(index) for index in mamba_state_indices if index >= 0})
        payload["mamba_state_indices"] = mamba_state_indices
        payload["mamba_gpu_decode_indices"] = _to_cpu(
            context.mamba_metadata.batch_indices_decode[: context.padded_active_request_count]
        )
        if valid_state_indices:
            payload["mamba_conv_state"] = _to_cpu(
                context.mamba_conv_states[:, valid_state_indices]
            )
            payload["mamba_ssm_state"] = _to_cpu(
                context.mamba_ssm_states[:, valid_state_indices]
            )

    return payload


def capture_async_sched_forward_state(
    context: Any,
    input_ids: Tensor,
    position_ids: Tensor,
    phase: str,
    logits: Optional[Tensor] = None,
) -> None:
    """Persist all relevant state around a selected decode forward."""
    directory = _diagnostic_directory()
    context._async_diag_forward_position = int(position_ids.flatten()[0].item())
    if directory is None or not _is_capture_forward(context):
        return

    payload = _capture_context(context)
    payload["input_ids"] = _to_cpu(input_ids)
    payload["position_ids"] = _to_cpu(position_ids)
    if logits is not None:
        payload["logits"] = _to_cpu(logits)
    position = context._async_diag_forward_position
    torch.save(
        payload,
        directory / f"position-{position:04d}-step-{context.step_count:04d}-forward-{phase}.pt",
    )


def install_async_sched_layer_hooks(model: Any, context: Any) -> None:
    """Capture each hybrid-layer input and output at selected decode steps."""
    if _diagnostic_directory() is None or not context.is_hybrid_model:
        return

    for layer in model.decoder.layers:
        layer_number = layer.layer_number
        layer_type = type(layer).__name__

        def capture_layer(
            _module: Any,
            _args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
            *,
            number: int = layer_number,
            name: str = layer_type,
        ) -> None:
            if not _is_capture_forward(context):
                return

            hidden_input = kwargs["hidden_states"]
            hidden_output = output[0] if isinstance(output, tuple) else output
            payload = {
                "step": context.step_count,
                "forward_position": context._async_diag_forward_position,
                "mode": context.config.async_sched_mode.value,
                "layer_number": number,
                "layer_type": name,
                "input": _to_cpu(hidden_input),
                "output": _to_cpu(hidden_output),
            }
            directory = _diagnostic_directory()
            assert directory is not None
            position = context._async_diag_forward_position
            torch.save(
                payload,
                directory
                / f"position-{position:04d}-step-{context.step_count:04d}-layer-{number:02d}.pt",
            )

        layer.register_forward_hook(capture_layer, with_kwargs=True)
