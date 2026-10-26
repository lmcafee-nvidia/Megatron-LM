# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Repro for defect d1: fused-TP inference drops the residual on every non-fused path.

With ``inference_fuse_tp_communication=True`` and ``InferenceMode`` active,
``TransformerLayer`` skips its own residual add for attention and MLP
(``transformer_layer.py:1040-1054``, ``:1582-1604``) and relies on
``InferenceRowParallelLinear`` to add it. The row-parallel layer adds the residual only
inside the fused NVLS branch of ``_matmul_reduce_scatter`` (``inference_layers.py:482-520``).
The NCCL fallback (``:521-525``) and the ``tp_size == 1`` short-circuit in ``forward``
(``:549-551``) return the bare projection, so the residual stream is lost.

Each test builds two identical 2-layer inference-optimized GPT ``TransformerBlock``s
(``get_gpt_layer_with_inference_spec``), one with the fuse flag off and one with it on,
with the same weights, and runs them on the same input under ``InferenceMode``.
The sub-layer outputs are about 0.1-0.2 in magnitude, while the residual input is N(0, 1).
A dropped residual therefore shows as ``||on - off|| / ||residual|| ~ 1``, far outside the
tolerance used.

Launch: ``python -m torch.distributed.run --standalone --nproc-per-node=2 -m pytest <file> -v``.
TP=2 cases skip when the world size is below 2. The NVLS-dependent cases skip when
symmetric memory or multicast is unavailable.
"""

import contextlib
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils

REQUIRED_GPUS = 2

# Problem size: [S, B, H] activations, 2 layers.
_S, _B, _H = 64, 2, 128
_FFN = 512
_HEADS = 4
_NUM_LAYERS = 2
_INIT_STD = 0.05

# Far tighter than the residual (N(0,1), max |x| ~ 4); loose enough for bf16 rounding of
# the residual add at |x| ~ 4 (1 ulp = 0.03) and for kernel-choice differences.
_ATOL = 2e-2
_RTOL = 2e-2


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------


def _require_runtime():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("transformer_engine", reason="inference_optimized requires TE")


@contextlib.contextmanager
def _model_parallel(tp: int):
    """Initialize TP=tp model parallelism and tear it down (and the 'tp' symm buffer)."""
    _require_runtime()
    if Utils.world_size < tp or Utils.world_size % tp != 0:
        pytest.skip(f"requires a world size divisible by TP={tp} (world size {Utils.world_size})")

    from megatron.core import parallel_state
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    SymmetricMemoryManager.destroy()
    try:
        model_parallel_cuda_manual_seed(123)
        yield parallel_state.get_tensor_model_parallel_group()
    finally:
        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


def _build_block(tp: int, *, fuse: bool, disable_nvls: bool):
    """Smallest inference-optimized GPT decoder block (no final norm), built on GPU.

    The block is built on GPU (``use_cpu_initialization=False``) as production does.
    ``_setup_fused_tp_communication`` stores the next layer's norm weights as ``.data``
    aliases, and a later ``.cuda()`` move would leave those aliases on the CPU.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_inference_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_H,
        ffn_hidden_size=_FFN,
        num_attention_heads=_HEADS,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,  # required by the inference linears at TP > 1
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.bfloat16,
        bf16=True,
        init_method_std=_INIT_STD,
        use_cpu_initialization=False,
        inference_fuse_tp_communication=fuse,
        inference_disable_triton_nvls_kernels=disable_nvls,
    )
    model_parallel_cuda_manual_seed(123)
    block = TransformerBlock(config, get_gpt_layer_with_inference_spec(), post_layer_norm=False)
    assert block.final_layernorm is None
    assert all(p.is_cuda for p in block.parameters())
    block.eval()
    return block


def _build_reference(tp: int, *, disable_nvls: bool):
    """Flag-off block with non-trivial norm gains, so a norm applied with the wrong weights
    is visible."""
    off = _build_block(tp, fuse=False, disable_nvls=disable_nvls)
    gen = torch.Generator(device="cpu").manual_seed(4321)
    with torch.no_grad():
        for name, p in off.named_parameters():
            if name.endswith("layer_norm_weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
    return off


def _build_pair(tp: int, *, disable_nvls: bool):
    """(flag-off block, flag-on block) with identical weights."""
    off = _build_reference(tp, disable_nvls=disable_nvls)
    on = _build_block(tp, fuse=True, disable_nvls=disable_nvls)
    off_params = dict(off.named_parameters())
    on_params = dict(on.named_parameters())
    assert off_params.keys() == on_params.keys(), "flag on/off blocks have different parameters"
    with torch.no_grad():
        # In-place copy keeps the fused wiring's `.data` aliases of the norm weights valid.
        for name, p in on_params.items():
            p.copy_(off_params[name])
    return off, on


def _input_shard(tp: int) -> torch.Tensor:
    """This rank's sequence-parallel shard of a fixed N(0,1) [S, B, H] input."""
    from megatron.core import parallel_state

    gen = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(_S, _B, _H, generator=gen).to(torch.bfloat16)
    rank = parallel_state.get_tensor_model_parallel_rank()
    return x.chunk(tp, dim=0)[rank].contiguous().cuda()


def _forward(block, x: torch.Tensor, *, inference: bool = True) -> torch.Tensor:
    """Run the block on a private copy of x (the fused kernel updates the residual in place)."""
    from megatron.core.inference.utils import InferenceMode

    inp = x.clone()
    with torch.no_grad():
        if inference:
            block.eval()
            with InferenceMode.active():
                out = block(hidden_states=inp, attention_mask=None)
        else:
            # Training path: TE modules + ordinary bias-dropout-add, InferenceMode inactive.
            block.train()
            try:
                out = block(hidden_states=inp, attention_mask=None)
            finally:
                block.eval()
    out = out.float().clone()
    # Keep ranks in step between forwards so one rank's next symmetric-memory collective
    # cannot overlap a peer's previous one (an ordering question outside d1).
    torch.cuda.synchronize()
    dist.barrier()
    return out


@contextlib.contextmanager
def _count_fused_kernel_calls():
    """Count calls of the fused RS+add+norm+AG kernel without changing its behavior."""
    import megatron.core.tensor_parallel.inference_layers as il

    with mock.patch.object(
        il, "fused_multimem_rs_add_norm_ag", wraps=il.fused_multimem_rs_add_norm_ag
    ) as spy:
        yield spy


def _all_reduce(value: float, op) -> float:
    t = torch.tensor([float(value)], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=op)
    return t.item()


def _comparison(out: torch.Tensor, ref: torch.Tensor, residual: torch.Tensor):
    """World-reduced stats; identical on every rank so all ranks pass or fail together."""
    residual = residual.float()
    diff = out - ref
    ok = bool(torch.allclose(out, ref, atol=_ATOL, rtol=_RTOL))
    stats = {
        "ok": _all_reduce(1.0 if ok else 0.0, dist.ReduceOp.MIN) == 1.0,
        "max_diff": _all_reduce(diff.abs().max().item(), dist.ReduceOp.MAX),
        "max_res": _all_reduce(residual.abs().max().item(), dist.ReduceOp.MAX),
        "diff_sq": _all_reduce(diff.pow(2).sum().item(), dist.ReduceOp.SUM),
        "res_sq": _all_reduce(residual.pow(2).sum().item(), dist.ReduceOp.SUM),
        "out_sq": _all_reduce(out.pow(2).sum().item(), dist.ReduceOp.SUM),
        "ref_sq": _all_reduce(ref.pow(2).sum().item(), dist.ReduceOp.SUM),
    }
    stats["diff_over_res"] = (stats["diff_sq"] / stats["res_sq"]) ** 0.5
    stats["out_over_ref"] = (stats["out_sq"] / max(stats["ref_sq"], 1e-30)) ** 0.5
    stats["out_over_res"] = (stats["out_sq"] / stats["res_sq"]) ** 0.5
    stats["ref_over_res"] = (stats["ref_sq"] / stats["res_sq"]) ** 0.5
    return stats


def _fmt(s) -> str:
    return (
        f"max|out-ref|={s['max_diff']:.4f}, max|residual|={s['max_res']:.4f}, "
        f"||out-ref||/||residual||={s['diff_over_res']:.3f}, ||out||/||ref||={s['out_over_ref']:.3f} "
        f"(atol={_ATOL}, rtol={_RTOL}; a ratio ~1 means the residual stream was dropped)"
    )


def _forward_with_buffer_fills(block, x, buf, fills=(0.0, 1.0)):
    """Outputs of two forwards whose only difference is the prior symmetric-buffer contents."""
    outs = []
    for value in fills:
        buf.symm_buffer.view(torch.bfloat16).fill_(value)
        torch.cuda.synchronize()
        dist.barrier()
        outs.append(_forward(block, x))
    return outs


def _symm_buffer(tp_group):
    """Create (collectively) and return the shared 'tp' symmetric-memory buffer."""
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    return SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)


def _require_symm_handle(tp_group):
    buf = _symm_buffer(tp_group)
    ok = _all_reduce(1.0 if buf.symm_mem_hdl is not None else 0.0, dist.ReduceOp.MIN) == 1.0
    if not ok:
        pytest.skip(f"symmetric memory unavailable: {buf.init_failure_reason}")
    return buf


def _require_nvls(tp_group):
    """Skip unless the NVLS multicast kernels can run on every rank."""
    from megatron.core.inference.communication.torch_symm_triton.utils import (
        is_device_nvls_capable,
    )

    buf = _symm_buffer(tp_group)  # collective; call on every rank before any skip
    capable = is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device()))
    hdl = buf.symm_mem_hdl
    multicast = hdl is not None and bool(getattr(hdl, "multicast_ptr", 0))
    ok = _all_reduce(1.0 if (capable and multicast) else 0.0, dist.ReduceOp.MIN) == 1.0
    if not ok:
        pytest.skip(
            "NVLS unavailable (needs SM>=9, a symmetric-memory handle and a multicast pointer; "
            f"capable={capable}, handle={hdl is not None}, multicast={multicast}, "
            f"reason={buf.init_failure_reason})"
        )
    return buf


# --------------------------------------------------------------------------------------
# CONTROL tests (must pass on main)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("tp", [1, 2], ids=["tp1", "tp2"])
def test_control_flag_off_matches_training_path(tp):
    """With the fuse flag off, the inference forward equals the TE training forward.

    This validates the flag-off output used as the reference by the defect tests. The
    training forward runs ``block.train()`` with InferenceMode inactive: the TE
    LayerNormLinear/RowParallel modules plus the ordinary bias-dropout-add (p=0).
    """
    with _model_parallel(tp):
        off = _build_reference(tp, disable_nvls=True)
        x = _input_shard(tp)
        out_inf = _forward(off, x, inference=True)
        out_train = _forward(off, x, inference=False)
        s = _comparison(out_inf, out_train, x)
        assert s["ok"], f"harness: flag-off inference != training path at TP={tp}: {_fmt(s)}"
        # Sanity: the sub-layers contribute visibly, but the residual dominates the output
        # (expected ||out - x|| / ||x|| of about 0.3), so the "~1" signature in the defect
        # messages identifies a dropped residual.
        sub = _comparison(out_inf, x.float(), x)["diff_over_res"]
        assert 0.01 < sub < 0.9, f"harness: ||out - x|| / ||x|| = {sub:.3f} outside (0.01, 0.9)"


def test_control_tp2_fused_nvls_path_matches_flag_off():
    """When the fused NVLS kernel actually runs, flag-on equals flag-off.

    This shows that the flag-on vs flag-off comparison passes on a path that handles the
    residual. Needs 2 GPUs with NVLS multicast.
    """
    tp = 2
    with _model_parallel(tp) as tp_group:
        _require_nvls(tp_group)
        off, on = _build_pair(tp, disable_nvls=False)
        for layer in on.layers:
            proj_w = layer.self_attention.linear_proj.next_layer_norm_weights
            fc1_w = layer.mlp.linear_fc1.layer_norm_weight
            assert proj_w.data_ptr() == fc1_w.data_ptr(), "harness: fused norm-weight wiring"
        x = _input_shard(tp)
        out_off = _forward(off, x)
        with _count_fused_kernel_calls() as spy:
            out_on = _forward(on, x)
        assert spy.call_count == 2 * _NUM_LAYERS, (
            f"harness: expected the fused NVLS kernel on every linear_proj/linear_fc2 "
            f"({2 * _NUM_LAYERS} calls), saw {spy.call_count}"
        )
        s = _comparison(out_on, out_off, x)
        assert s["ok"], f"harness: fused NVLS path != flag-off reference: {_fmt(s)}"


def test_control_tp2_flag_off_output_independent_of_symm_buffer():
    """With the fuse flag off and NVLS disabled, the output ignores the symmetric buffer."""
    tp = 2
    with _model_parallel(tp) as tp_group:
        buf = _require_symm_handle(tp_group)
        off = _build_reference(tp, disable_nvls=True)
        x = _input_shard(tp)
        out_a, out_b = _forward_with_buffer_fills(off, x, buf)
        s = _comparison(out_b, out_a, x)
        assert s["ok"], f"harness: flag-off output depends on symmetric-buffer contents: {_fmt(s)}"


# --------------------------------------------------------------------------------------
# DEFECT tests (must fail on main)
# --------------------------------------------------------------------------------------


def test_defect_tp1_fused_flag_drops_residual():
    """(a) TP=1: the row-parallel ``tp_size == 1`` short-circuit never adds the residual."""
    tp = 1
    with _model_parallel(tp):
        off, on = _build_pair(tp, disable_nvls=False)
        x = _input_shard(tp)
        out_off = _forward(off, x)
        with _count_fused_kernel_calls() as spy:
            out_on = _forward(on, x)
        assert spy.call_count == 0, "harness: fused kernel unexpectedly ran at TP=1"
        s = _comparison(out_on, out_off, x)
        assert s["ok"], (
            "DEFECT d1: TP=1 with inference_fuse_tp_communication=True: TransformerLayer skips "
            "its residual add (transformer_layer.py:1050-1054, 1600-1604) but "
            "InferenceRowParallelLinear.forward's tp_size==1 short-circuit "
            "(inference_layers.py:549-551) never adds it; fused-TP output != unfused output: "
            + _fmt(s)
        )


def test_defect_tp2_nccl_fallback_drops_residual():
    """(b) TP=2, ``inference_disable_triton_nvls_kernels=True``: NCCL fallback, no residual."""
    tp = 2
    with _model_parallel(tp) as tp_group:
        off, on = _build_pair(tp, disable_nvls=True)
        x = _input_shard(tp)
        out_off = _forward(off, x)
        # Fallback column layers may consume the symmetric buffer without writing it (see the
        # stale-buffer test). Zero it so this case is deterministic and shows only the
        # residual signature.
        buf = _symm_buffer(tp_group)
        if buf.symm_buffer is not None:
            buf.symm_buffer.zero_()
        with _count_fused_kernel_calls() as spy:
            out_on = _forward(on, x)
        assert spy.call_count == 0, "harness: fused kernel ran although NVLS kernels are disabled"
        s = _comparison(out_on, out_off, x)
        assert s["ok"], (
            "DEFECT d1: TP=2, inference_disable_triton_nvls_kernels=True, fuse flag on: "
            "_matmul_reduce_scatter's NCCL fallback (inference_layers.py:521-525) returns the "
            "bare reduce-scatter without the residual that TransformerLayer skipped "
            "(transformer_layer.py:1050-1054, 1600-1604); fused-TP output != unfused output: "
            + _fmt(s)
        )


def test_defect_tp2_batch_invariant_fallback_drops_residual():
    """(c) TP=2, NVLS available, batch-invariant mode forces the NCCL RS: no residual."""
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    tp = 2
    with _model_parallel(tp) as tp_group:
        _require_nvls(tp_group)
        off, on = _build_pair(tp, disable_nvls=False)
        x = _input_shard(tp)
        # The runtime global mode is what _matmul_reduce_scatter reads (inference_layers.py:487).
        # te_native keeps native TE GEMM/RMSNorm kernels (no DeepGEMM / FA3 needed).
        with set_batch_invariant_mode(True, backend="te_native"):
            out_off = _forward(off, x)
            with _count_fused_kernel_calls() as spy:
                out_on = _forward(on, x)
        assert spy.call_count == 0, "harness: fused kernel ran in batch-invariant mode"
        s = _comparison(out_on, out_off, x)
        assert s["ok"], (
            "DEFECT d1: TP=2 in batch-invariant mode, fuse flag on: can_use_nvls is forced False "
            "(inference_layers.py:487), so the NCCL fallback (inference_layers.py:521-525) "
            "returns without the residual TransformerLayer skipped; fused-TP output != "
            "unfused output: " + _fmt(s)
        )


def test_defect_tp2_nccl_fallback_reads_stale_symm_buffer():
    """TP=2 NCCL fallback, fuse flag on: the next column layer reads an unwritten buffer.

    ``InferenceLayerNormColumnParallelLinear`` decides it is "fused" from its own flag and
    the buffer handle only (``inference_layers.py:265-271``), so after a fallback RS it
    consumes ``symm_mem_buffer['tensor']`` that no kernel wrote this step. Correct code
    produces an output independent of the buffer's prior contents.
    """
    tp = 2
    with _model_parallel(tp) as tp_group:
        buf = _require_symm_handle(tp_group)
        on = _build_block(tp, fuse=True, disable_nvls=True)
        x = _input_shard(tp)
        with _count_fused_kernel_calls() as spy:
            out_a, out_b = _forward_with_buffer_fills(on, x, buf)
        assert spy.call_count == 0, "harness: fused kernel ran although NVLS kernels are disabled"
        s = _comparison(out_b, out_a, x)
        assert s["ok"], (
            "DEFECT d1: TP=2 NCCL fallback with the fuse flag on: the output depends on stale "
            "symmetric-memory contents (buffer pre-filled with 1.0 vs 0.0: "
            f"max|delta|={s['max_diff']:.4f}, ||out(0.0)||/||x||={s['ref_over_res']:.3f}, "
            f"||out(1.0)||/||x||={s['out_over_res']:.3f}). "
            "InferenceLayerNormColumnParallelLinear (inference_layers.py:265-271) treats itself "
            "as fused and consumes symm_mem_buffer['tensor'], which the fallback reduce-scatter "
            "never wrote"
        )
