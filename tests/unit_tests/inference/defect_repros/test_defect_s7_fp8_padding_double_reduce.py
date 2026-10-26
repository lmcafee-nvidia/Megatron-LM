# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime check s7: the FP8 inference padding wrapper reduces a row-parallel output twice.

Code read at ed9c2c8c1:

* ``prepare_model_for_fp8_inference`` (``megatron/core/fp8_utils.py:1046-1065``) wraps every TE
  linear (``TE_LINEAR_TYPES``, ``:82-87``) with ``_wrap_te_linear_for_padding`` (``:962-1044``).
  ``AbstractModelInferenceWrapper.__init__`` applies it whenever ``config.fp8`` is set and
  ``transformer_impl != "inference_optimized"`` (``abstract_model_inference_wrapper.py:87-88``).
* Inside an FP8 context with ``module.sequence_parallel`` set, the wrapper sets
  ``module.sequence_parallel = False`` (``:1001``), pads the sequence, calls the original forward,
  unpads, and then, for a row-parallel module, reduce-scatters the output itself (``:1028-1036``).
* Megatron's ``TERowParallelLinear`` builds ``te.pytorch.Linear`` with ``parallel_mode="row"``,
  ``tp_size=TP`` (``megatron/core/extensions/transformer_engine.py:2086-2108`` ->
  ``TELinear.__init__`` ``:1377-1431``) and its forward calls TE's forward unchanged (``:1464-1477``).
* TE v2.20 (``transformer_engine/pytorch/module/linear.py`` at tag ``v2.20``, read from GitHub;
  not the container's build) passes ``sequence_parallel=self.sequence_parallel`` and
  ``tensor_parallel=self.tp_size > 1`` into the forward on every call (``:2524-2525``), and the
  forward does ``if sequence_parallel: reduce_scatter ... elif tensor_parallel: allreduce``
  for row-parallel (``:664-673``).

So with the wrapper, TE all-reduces the partial products and the wrapper then reduce-scatters the
already-reduced output: every output element is ``TP x`` the correct value. The wrapper relies on
TE doing *no* reduction when ``sequence_parallel`` is False, which by reading is false.

Tests (TP=2, sequence parallel, one ``TERowParallelLinear`` with fixed weights):

* ``test_control_bf16_row_parallel_sp_matches_reference`` -- no FP8, no wrapper: output equals a
  plain-torch fp64 reference of the TP-sharded matmul + sequence reduce-scatter. Must PASS.
* ``test_control_wrapper_is_noop_outside_fp8`` -- wrapper installed with the real
  ``prepare_model_for_fp8_inference``, run outside an FP8 context: passes straight through. PASS.
* ``test_control_fp8_unwrapped_matches_reference`` -- FP8 (tensorwise current scaling) without the
  wrapper: within FP8 tolerance of the reference, and different from the bf16 output (FP8 ran).
  PASS.
* ``test_defect_fp8_padding_wrapper_reduces_row_parallel_once`` -- the property the wrapper relies
  on: with the wrapper installed, the FP8 output equals the unwrapped FP8 output. Expected to FAIL
  on ``main`` with ``DEFECT s7: ...`` and a wrapped/unwrapped ratio of 2.0. Parametrized on a
  sequence length that needs no padding (32) and one the wrapper pads (40 -> 48).

Requirements: 2 GPUs (``REQUIRED_GPUS = 2``), Transformer Engine with FP8 support (H100). Skips
otherwise. Launch:
``python -m torch.distributed.run --standalone --nproc-per-node=2 -m pytest <file> -v``.
"""

import contextlib

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils

REQUIRED_GPUS = 2

TP = 2
BATCH = 2
H_IN = 256  # row-parallel input features (128 per rank); FP8 needs the last dim % 16 == 0
H_OUT = 128
W_STD = 0.05  # output std ~ sqrt(256) * 0.05 = 0.8

# Sequence lengths divisible by TP (sequence parallel) with S*B % 8 == 0 (TE FP8 GEMM, so the
# unwrapped FP8 forward runs). 32 needs no FP8 padding; the wrapper pads 40 to 48.
SEQ_LENS = [pytest.param(32, id="seq32-unpadded"), pytest.param(40, id="seq40-padded-to-48")]

BF16_ATOL = 2e-2
BF16_RTOL = 2e-2
# E4M3 relative quantization error is at most 2^-4 per operand; a K=128 dot product lands near 5%
# relative Frobenius error. A double reduction gives a relative error of TP - 1 = 1.
FP8_REF_REL_TOL = 0.15
# Wrapped vs unwrapped FP8: the same per-tensor scales (padding adds zero rows), so only GEMM
# algorithm and bf16 reduction rounding differ.
WRAPPED_REL_TOL = 0.05


# --------------------------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------------------------- #


@contextlib.contextmanager
def _tp_sp():
    """TP=2 model parallel state (pattern of test_defect_d1_fused_tp_residual.py::_model_parallel)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("transformer_engine", reason="TERowParallelLinear requires TE")
    if Utils.world_size < TP or Utils.world_size % TP != 0:
        pytest.skip(f"requires a world size divisible by TP={TP} (world size {Utils.world_size})")

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(tensor_model_parallel_size=TP)
    try:
        model_parallel_cuda_manual_seed(123)
        yield parallel_state.get_tensor_model_parallel_group()
    finally:
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()


def _require_fp8():
    try:
        from transformer_engine.pytorch.fp8 import check_fp8_support
    except ImportError:  # newer TE layout
        from transformer_engine.pytorch.quantization import check_fp8_support
    from megatron.core import fp8_utils

    ok, reason = check_fp8_support()
    if not ok:
        pytest.skip(f"FP8 unsupported here: {reason}")
    if fp8_utils.Fp8Padding is None or fp8_utils.Fp8Unpadding is None:
        pytest.skip("this TE has no Fp8Padding/Fp8Unpadding; the wrapper cannot be installed")


def _config(fp8: bool):
    from megatron.core.transformer.transformer_config import TransformerConfig

    extra = dict(fp8="e4m3", fp8_recipe="tensorwise") if fp8 else {}
    return TransformerConfig(
        num_layers=1,
        hidden_size=H_OUT,
        num_attention_heads=4,
        tensor_model_parallel_size=TP,
        sequence_parallel=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        **extra,
    )


def _operands(seq_len):
    """Full input [S, B, H_IN] and weight [H_OUT, H_IN] in bf16, identical on every rank."""
    g = torch.Generator().manual_seed(1000 + seq_len)
    x = torch.randn(seq_len, BATCH, H_IN, generator=g).to(torch.bfloat16)
    w = (torch.randn(H_OUT, H_IN, generator=g) * W_STD).to(torch.bfloat16)
    return x, w


def _row_parallel(config, tp_group, w_full):
    """TERowParallelLinear as built in tests/unit_tests/tensor_parallel/test_initialization.py,
    with this rank's input-dimension shard of w_full copied in before any forward."""
    from megatron.core.extensions.transformer_engine import TERowParallelLinear

    module = TERowParallelLinear(
        H_IN,
        H_OUT,
        config=config,
        init_method=config.init_method,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
        tp_group=tp_group,
    )
    rank = dist.get_rank(tp_group)
    shard = H_IN // TP
    with torch.no_grad():
        module.weight.copy_(w_full[:, rank * shard : (rank + 1) * shard].to("cuda"))
    assert module.sequence_parallel, "harness: TE module was built without sequence_parallel"
    return module.eval()


def _local_input(x_full, tp_group):
    rank = dist.get_rank(tp_group)
    shard = H_IN // TP
    return x_full[:, :, rank * shard : (rank + 1) * shard].contiguous().to("cuda")


def _reference(x_full, w_full, tp_group):
    """Plain-torch fp64 result of the TP-sharded matmul + reduce-scatter along the sequence:
    rank r's chunk of sum_k x[..., k] * w[:, k], computed from the full operands (no collectives)."""
    rank = dist.get_rank(tp_group)
    seq_len = x_full.shape[0]
    chunk = seq_len // TP
    full = torch.einsum("sbk,nk->sbn", x_full.double(), w_full.double())
    return full[rank * chunk : (rank + 1) * chunk].to("cuda")


def _forward(module, x_local, fp8_config=None):
    from megatron.core.fp8_utils import get_fp8_context

    ctx = get_fp8_context(fp8_config) if fp8_config is not None else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        out, bias = module(x_local)
    assert bias is None, "harness: unexpected bias output"
    return out


def _stats(out, ref, tp_group):
    """TP-group-reduced relative error and projection ratio <out, ref> / <ref, ref>."""
    o = out.double()
    r = ref.double()
    t = torch.stack([((o - r) ** 2).sum(), (r * r).sum(), (o * r).sum()])
    dist.all_reduce(t, group=tp_group)
    err2, ref2, dot = t.tolist()
    return dict(rel_err=(err2 / ref2) ** 0.5, ratio=dot / ref2)


def _install_wrapper(module):
    """Install the FP8 padding wrapper with the real installer (what the inference wrapper calls)."""
    from megatron.core import fp8_utils

    container = torch.nn.Module()
    container.proj = module
    unwrapped_forward = module.forward
    fp8_utils.prepare_model_for_fp8_inference(container)
    assert module in fp8_utils._fp8_inference_wrapped_modules, "harness: wrapper not installed"
    assert module.forward is not unwrapped_forward, "harness: forward was not replaced"
    return fp8_utils


def _expected_shape(seq_len):
    return (seq_len // TP, BATCH, H_OUT)


# --------------------------------------------------------------------------------------------- #
# CONTROL (must pass on main)
# --------------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_control_bf16_row_parallel_sp_matches_reference(seq_len):
    with _tp_sp() as tp_group:
        x_full, w_full = _operands(seq_len)
        module = _row_parallel(_config(fp8=False), tp_group, w_full)
        out = _forward(module, _local_input(x_full, tp_group))
        ref = _reference(x_full, w_full, tp_group)
        assert tuple(out.shape) == _expected_shape(seq_len), f"harness: shape {tuple(out.shape)}"
        torch.testing.assert_close(out.double(), ref, atol=BF16_ATOL, rtol=BF16_RTOL)
        s = _stats(out, ref, tp_group)
        assert abs(s["ratio"] - 1.0) < 0.01, f"harness: bf16 output/reference ratio {s}"


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_control_wrapper_is_noop_outside_fp8(seq_len, monkeypatch):
    """Wrapped module outside an FP8 context: the wrapper calls the original forward directly."""
    with _tp_sp() as tp_group:
        _require_fp8()
        x_full, w_full = _operands(seq_len)
        module = _row_parallel(_config(fp8=True), tp_group, w_full)
        x_local = _local_input(x_full, tp_group)
        before = _forward(module, x_local)
        fp8_utils = _install_wrapper(module)
        rs_calls = []
        original_rs = fp8_utils.reduce_scatter_to_sequence_parallel_region
        monkeypatch.setattr(
            fp8_utils,
            "reduce_scatter_to_sequence_parallel_region",
            lambda *a, **k: rs_calls.append(1) or original_rs(*a, **k),
        )
        try:
            after = _forward(module, x_local)
        finally:
            fp8_utils._fp8_inference_wrapped_modules.discard(module)
        assert not rs_calls, "harness: the wrapper took its FP8 path outside an FP8 context"
        assert module.sequence_parallel, "harness: wrapper left sequence_parallel cleared"
        torch.testing.assert_close(
            after, before, atol=1e-3, rtol=1e-3, msg="harness: wrapper changed the bf16 output"
        )
        torch.testing.assert_close(
            after.double(), _reference(x_full, w_full, tp_group), atol=BF16_ATOL, rtol=BF16_RTOL
        )


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_control_fp8_unwrapped_matches_reference(seq_len):
    """FP8 row-parallel forward without the wrapper is within FP8 tolerance (validates the
    tolerance and the unwrapped FP8 output the defect test compares against)."""
    with _tp_sp() as tp_group:
        _require_fp8()
        config = _config(fp8=True)
        x_full, w_full = _operands(seq_len)
        module = _row_parallel(config, tp_group, w_full)
        x_local = _local_input(x_full, tp_group)
        out_fp8 = _forward(module, x_local, fp8_config=config)
        out_bf16 = _forward(module, x_local)
        ref = _reference(x_full, w_full, tp_group)
        assert tuple(out_fp8.shape) == _expected_shape(seq_len)
        s = _stats(out_fp8, ref, tp_group)
        assert s["rel_err"] < FP8_REF_REL_TOL, f"harness: unwrapped FP8 vs reference {s}"
        assert abs(s["ratio"] - 1.0) < 0.05, f"harness: unwrapped FP8 vs reference {s}"
        assert not torch.equal(out_fp8, out_bf16), "harness: the FP8 context did not change the GEMM"


# --------------------------------------------------------------------------------------------- #
# DEFECT / runtime check (fails on main if the wrapper reduces twice)
# --------------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_defect_fp8_padding_wrapper_reduces_row_parallel_once(seq_len, monkeypatch):
    with _tp_sp() as tp_group:
        _require_fp8()
        config = _config(fp8=True)
        x_full, w_full = _operands(seq_len)
        module = _row_parallel(config, tp_group, w_full)
        x_local = _local_input(x_full, tp_group)
        ref = _reference(x_full, w_full, tp_group)

        unwrapped = _forward(module, x_local, fp8_config=config)
        fp8_utils = _install_wrapper(module)

        # Spy on the wrapper's own reduce-scatter (looked up as a module global at call time), so
        # the test cannot pass vacuously by the wrapper falling through to the unwrapped forward.
        rs_calls = []
        sp_seen_by_te = []
        original_rs = fp8_utils.reduce_scatter_to_sequence_parallel_region

        def rs_spy(*args, **kwargs):
            rs_calls.append(1)
            sp_seen_by_te.append(module.sequence_parallel)  # still cleared at this point
            return original_rs(*args, **kwargs)

        monkeypatch.setattr(fp8_utils, "reduce_scatter_to_sequence_parallel_region", rs_spy)
        try:
            wrapped = _forward(module, x_local, fp8_config=config)
        finally:
            fp8_utils._fp8_inference_wrapped_modules.discard(module)
        assert rs_calls == [1] and sp_seen_by_te == [False], (
            f"harness: the wrapper did not take its FP8 sequence-parallel path "
            f"(reduce-scatter calls {len(rs_calls)}, sequence_parallel during it {sp_seen_by_te})"
        )

        assert tuple(wrapped.shape) == tuple(unwrapped.shape) == _expected_shape(seq_len), (
            f"DEFECT s7: wrapped output shape {tuple(wrapped.shape)} != unwrapped "
            f"{tuple(unwrapped.shape)}"
        )
        vs_unwrapped = _stats(wrapped, unwrapped, tp_group)
        vs_ref = _stats(wrapped, ref, tp_group)
        assert vs_unwrapped["rel_err"] < WRAPPED_REL_TOL, (
            f"DEFECT s7: FP8 inference padding wrapper (fp8_utils._wrap_te_linear_for_padding) on "
            f"TERowParallelLinear, TP={TP}, sequence_parallel=True, S={seq_len}: wrapped output != "
            f"unwrapped FP8 output (rel err {vs_unwrapped['rel_err']:.3f}, ratio "
            f"<wrapped, unwrapped>/<unwrapped, unwrapped> = {vs_unwrapped['ratio']:.3f}; vs the "
            f"exact reference: rel err {vs_ref['rel_err']:.3f}, ratio {vs_ref['ratio']:.3f}). A "
            f"ratio of {TP} means the output was reduced twice: the wrapper sets "
            f"module.sequence_parallel=False, so TE's row-parallel forward all-reduces, and the "
            f"wrapper then reduce-scatters the reduced output again."
        )
        assert module.sequence_parallel, (
            "DEFECT s7: the wrapper did not restore module.sequence_parallel after the forward"
        )
