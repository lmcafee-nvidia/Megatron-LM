# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime check s9: the module-level batch-invariant RMSNorm patch drops zero-centered gamma.

``_te_patch_for_batch_invariant`` (``megatron/core/transformer/custom_layers/
batch_invariant_kernels.py:632-764``) patches three normalization entry points when batch-invariant
mode is enabled with the ``"triton"`` or ``"deepgemm"`` backend (``te_native`` skips all of them,
``:1866-1874``):

1. ``transformer_engine.pytorch.RMSNorm.forward`` -> ``_te_rmsnorm_forward_patched``
   (``:1646-1655``), which reads ``self.zero_centered_gamma`` and passes it to
   ``BatchInvariantRMSNormFn``;
2. ``transformer_engine.pytorch.module._common.apply_normalization`` (and its re-imports in
   ``layernorm_linear`` / ``layernorm_mlp``) -> ``_te_apply_normalization_patched``
   (``:1215-1274``), which adds 1 to gamma when ``zero_centered_gamma``;
3. any of ``rmsnorm`` / ``rmsnorm_forward`` / ``rmsnorm_fwd`` found on
   ``transformer_engine.pytorch.module.layernorm`` -> ``_make_rmsnorm_patched`` (``:709-730``,
   applied by the loop at ``:732-736``), which extracts only ``x``, ``weight`` and ``eps`` and
   calls ``rmsnorm_batch_invariant``, which hard-codes ``zero_centered_gamma=False``
   (``:1390-1397``).

Path 3 is the suspected bug. It can only matter if the installed TE exposes one of those names.
By reading TE ``v2.20`` (``transformer_engine/pytorch/module/layernorm.py`` defines only the
``LayerNorm`` class), none exists, so the patch would be dead code; the probe test records what the
container's build actually has.

Tests: the two live paths (1, 2) are CONTROLS against a plain-torch fp64 reference
``x * rsqrt(mean(x^2) + eps) * (1 + gamma)``; the DEFECT test checks path 3 for each name that
exists and SKIPs (with the probe result) when none does.

Needs one GPU (``mean_dim`` is a Triton kernel) and Transformer Engine.
"""

import pytest
import torch

import megatron.core.transformer.custom_layers.batch_invariant_kernels as bik

try:
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.module._common as te_common
    import transformer_engine.pytorch.module.layernorm as te_layernorm_mod

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

REQUIRED_GPUS = 1

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="s9 check needs one CUDA GPU"),
    pytest.mark.skipif(not HAVE_TE, reason="s9 check needs Transformer Engine"),
]

MODULE_LEVEL_NAMES = ("rmsnorm", "rmsnorm_forward", "rmsnorm_fwd")
# Backend whose enable path runs the RMSNorm patches (te_native skips them); needs no DeepGEMM.
BACKEND = "triton"
HIDDEN = 512
ROWS = 96
EPS = 1e-5
# bf16 output vs fp64 reference: a few bf16 ulps. The zero-centered vs plain gamma gap is O(1).
ATOL = 3e-2
RTOL = 2e-2


def _present_names():
    """Module-level RMSNorm names TE exposes *before* any batch-invariant patching."""
    return [n for n in MODULE_LEVEL_NAMES if hasattr(te_layernorm_mod, n)]


def _inputs(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(ROWS, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    gamma = (0.5 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    return x, gamma


def _reference(x, gamma, zero_centered):
    """Plain-torch fp64 RMSNorm, computed outside batch-invariant mode (aten::mean is patched
    inside it)."""
    x64 = x.double()
    w64 = gamma.double() + (1.0 if zero_centered else 0.0)
    return x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS) * w64


def _errs(out, x, gamma):
    """(max err vs zero-centered reference, max err vs plain-gamma reference)."""
    out64 = out.double()
    return (
        (out64 - _reference(x, gamma, True)).abs().max().item(),
        (out64 - _reference(x, gamma, False)).abs().max().item(),
    )


def _close(out, x, gamma):
    return torch.allclose(out.double(), _reference(x, gamma, True), atol=ATOL, rtol=RTOL)


@pytest.fixture
def bi_off():
    if bik.is_batch_invariant_mode_enabled():
        pytest.skip("batch-invariant mode already enabled in this process (leaked state)")
    yield
    assert not bik.is_batch_invariant_mode_enabled()


# ---------------------------------------------------------------------------------------------
# Probe and controls
# ---------------------------------------------------------------------------------------------


def test_control_probe_module_level_rmsnorm_names(bi_off, record_property):
    """CONTROL/probe: record which module-level RMSNorm names TE exposes and which the
    batch-invariant enable path actually wraps."""
    present = _present_names()
    with bik.set_batch_invariant_mode(True, backend=BACKEND):
        wrapped = sorted(bik._TE_RMSNORM_FUNC_ORIGS)
        class_patched = te.RMSNorm.forward is bik._te_rmsnorm_forward_patched
        apply_norm_patched = te_common.apply_normalization is bik._te_apply_normalization_patched
    msg = (
        f"s9 probe: transformer_engine {getattr(transformer_engine, '__version__', '?')}; "
        f"module.layernorm exposes {present or 'none'} of {list(MODULE_LEVEL_NAMES)}; "
        f"wrapped by BI patch: {wrapped or 'none'}; RMSNorm.forward patched: {class_patched}; "
        f"_common.apply_normalization patched: {apply_norm_patched}"
    )
    record_property("s9_probe", msg)
    print(msg)  # pyproject addopts include -s
    assert wrapped == sorted(present)


def test_control_class_rmsnorm_patch_honors_zero_centered_gamma(bi_off):
    """CONTROL: te.RMSNorm(zero_centered_gamma=True) under batch-invariant mode runs the patched
    forward and matches the (1 + gamma) reference."""
    x, gamma = _inputs(seed=1)
    norm = te.RMSNorm(HIDDEN, eps=EPS, zero_centered_gamma=True, device="cuda", dtype=x.dtype)
    with torch.no_grad():
        norm.weight.copy_(gamma)
    with torch.no_grad(), bik.set_batch_invariant_mode(True, backend=BACKEND):
        assert type(norm).forward is bik._te_rmsnorm_forward_patched, "class patch not installed"
        out = norm(x)
    err_zc, err_plain = _errs(out, x, gamma)
    assert err_plain > 0.25, "harness: gamma does not distinguish the two references"
    assert _close(out, x, gamma), f"err vs (1+gamma) {err_zc:.3g}, vs gamma {err_plain:.3g}"


def test_control_apply_normalization_patch_honors_zero_centered_gamma(bi_off):
    """CONTROL: the fused-module entry (apply_normalization, used by LayerNormLinear/MLP) under
    batch-invariant mode matches the (1 + gamma) reference."""
    x, gamma = _inputs(seed=2)
    with torch.no_grad(), bik.set_batch_invariant_mode(True, backend=BACKEND):
        fn = te_common.apply_normalization
        assert fn is bik._te_apply_normalization_patched, "apply_normalization patch not installed"
        out, mu, rsigma = fn(x, None, gamma, None, EPS, None, x.dtype, "RMSNorm", 0, True)
    assert mu is None and rsigma.shape == (ROWS,)
    err_zc, err_plain = _errs(out, x, gamma)
    assert err_plain > 0.25, "harness: gamma does not distinguish the two references"
    assert _close(out, x, gamma), f"err vs (1+gamma) {err_zc:.3g}, vs gamma {err_plain:.3g}"


# ---------------------------------------------------------------------------------------------
# Defect (runtime check of the module-level patch)
# ---------------------------------------------------------------------------------------------


def test_defect_module_level_rmsnorm_patch_honors_zero_centered_gamma(bi_off):
    """DEFECT/runtime check: every module-level RMSNorm function the batch-invariant mode wraps
    must honour zero_centered_gamma=True. SKIPs when TE exposes none (the patch is dead code)."""
    present = _present_names()
    if not present:
        pytest.skip(
            "s9 unreachable with this TE "
            f"({getattr(transformer_engine, '__version__', '?')}): "
            "transformer_engine.pytorch.module.layernorm exposes none of "
            f"{list(MODULE_LEVEL_NAMES)}, so the module-level batch-invariant RMSNorm patch "
            "(batch_invariant_kernels.py:709-736) wraps nothing; it is dead code here"
        )
    x, gamma = _inputs(seed=3)
    outs = {}
    with torch.no_grad(), bik.set_batch_invariant_mode(True, backend=BACKEND):
        for name in present:
            fn = getattr(te_layernorm_mod, name)
            assert name in bik._TE_RMSNORM_FUNC_ORIGS, f"harness: {name} was not wrapped"
            out = fn(x, gamma, EPS, zero_centered_gamma=True)
            outs[name] = (out[0] if isinstance(out, (tuple, list)) else out).clone()
    # Compare outside the scope: batch-invariant mode reroutes aten::mean used by the reference.
    problems = []
    for name, out in outs.items():
        if not _close(out, x, gamma):
            err_zc, err_plain = _errs(out, x, gamma)
            problems.append(
                f"{name}: max err vs (1+gamma) reference {err_zc:.3g}, "
                f"vs plain-gamma reference {err_plain:.3g}"
            )
    assert not problems, (
        "DEFECT s9: the batch-invariant wrapper of transformer_engine.pytorch.module.layernorm "
        "RMSNorm functions ignores zero_centered_gamma=True (it forwards only x, weight, eps to "
        "rmsnorm_batch_invariant, which hard-codes zero_centered_gamma=False): "
        + "; ".join(problems)
    )
