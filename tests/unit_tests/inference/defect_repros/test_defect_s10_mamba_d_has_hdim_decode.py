# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for suspected bug s10: Mamba decode applies the wrong skip term when ``D_has_hdim=True``.

With ``D_has_hdim=True`` the skip parameter ``MambaMixer.D`` has one entry per channel,
shape ``(nheads * headdim,)`` (``megatron/core/ssm/mamba_mixer.py:418-424``). The training /
prefill paths reshape it to ``(nheads, headdim)`` (``:686-691``, ``:753-758``, ``:1000-1005``), and
so does the batch-invariant decoder (``megatron/core/ssm/ops/mamba2/batch_invariant_decode.py:
478-480``). The default Triton decode step instead builds
``D = repeat(self.D, "h -> h p", p=headdim)`` (``mamba_mixer.py:1277``), i.e. a
``(nheads * headdim, headdim)`` view with strides ``(1, 0)``. ``selective_state_update`` accepts
any 2-D ``D`` (``ops/mamba2/mamba_ssm.py:350-351``) and the kernel reads
``D_ptr + pid_h * stride_D_head + m * stride_D_dim`` (``:191-193``), i.e. ``self.D[head]`` for
every channel of the head instead of ``self.D[head * headdim + m]``. No shape error is raised;
the decode output is wrong whenever ``D`` is not one constant. (The native-PyTorch fallback
branches at ``:1238`` and ``:1265`` have the same per-head assumption but are unreachable: the
in-tree Triton ``selective_state_update`` is always importable.)

Oracle: a plain-PyTorch fp64 single-token SSM step written from the Mamba-2 recurrence
(conv-state shift + depthwise conv + SiLU, ``dt = softplus(dt + dt_bias)``,
``h' = exp(dt * A) h + dt * B x``, ``y = C h' + D x``, gated by ``silu(z)``), using the mixer's own
parameters. ``rmsnorm=False`` keeps the gate inside the kernel so the oracle needs no norm.

Construction follows ``tests/unit_tests/ssm/test_mamba_mixer.py::TestMambaMixer.get_mixer`` and
``test_defect_d6_static_mamba_batch.py`` (``hybrid_stack_spec`` mixer submodules, TP=1, fp32,
default Mamba dims: d_state 128, headdim 64, 8 groups, d_inner 512 -> 8 heads).

Needs one GPU, Transformer Engine (in/out projections of ``hybrid_stack_spec``) and ``mamba_ssm``
(``MambaMixer.__init__`` requires it). The decode kernels are Megatron's in-tree Triton kernels.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import mamba_ssm  # noqa: F401

    HAVE_MAMBA_SSM = True
except ImportError:
    HAVE_MAMBA_SSM = False

try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

REQUIRED_GPUS = 1

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="s10 repro needs one CUDA GPU"),
    pytest.mark.skipif(not HAVE_MAMBA_SSM, reason="MambaMixer.__init__ requires mamba_ssm"),
    pytest.mark.skipif(not HAVE_TE, reason="hybrid_stack_spec in_proj/out_proj need TE"),
]

HIDDEN = 256
BATCH = 2
# fp32 kernel vs fp64 oracle; state sums run over d_state=128 terms of O(0.1-1) values.
ATOL = 1e-4
RTOL = 1e-4


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _mixer(d_has_hdim: bool) -> MambaMixer:
    config = TransformerConfig(
        hidden_size=HIDDEN, num_layers=1, num_attention_heads=1, use_cpu_initialization=True
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    m = MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        D_has_hdim=d_has_hdim,
        rmsnorm=False,
        layer_number=1,
        pg_collection=pg_collection,
    )
    return m.cuda().eval()


def reference_decode_step(m, zxBCdt, conv_state, ssm_state):
    """Plain-PyTorch fp64 Mamba-2 decode step for batch-first zxBCdt of shape (b, 1, proj).

    Returns (y (b, 1, d_inner), new conv_state, new ssm_state); inputs are not modified.
    """
    d_inner, G, N = m.d_inner_local_tp, m.ngroups_local_tp, m.d_state
    H, P = m.nheads_local_tp, m.headdim
    b = zxBCdt.shape[0]
    zx = zxBCdt[:, 0].double()
    z, xBC, dt = torch.split(zx, [d_inner, d_inner + 2 * G * N, H], dim=-1)

    # Causal depthwise conv over the rolling window: shift left, append the new input.
    conv = torch.cat([conv_state.double()[:, :, 1:], xBC.unsqueeze(-1)], dim=-1)
    w = m.conv1d_weight.double()[:, 0, :]  # (conv_dim, d_conv)
    xBC = F.silu((conv * w).sum(-1) + m.conv1d_bias.double())
    x, B, C = torch.split(xBC, [d_inner, G * N, G * N], dim=-1)
    x = x.reshape(b, H, P)
    B = B.reshape(b, G, N).repeat_interleave(H // G, dim=1)  # head h uses group h // (H/G)
    C = C.reshape(b, G, N).repeat_interleave(H // G, dim=1)

    dt = F.softplus(dt + m.dt_bias.double())  # (b, H)
    A = -torch.exp(m.A_log.double())  # (H,)
    dA = torch.exp(dt * A)
    state = ssm_state.double() * dA[:, :, None, None] + (dt[:, :, None] * x)[..., None] * B[
        :, :, None, :
    ]
    y = (state * C[:, :, None, :]).sum(-1)  # (b, H, P)
    D = m.D.double().reshape(H, P) if m.D_has_hdim else m.D.double()[:, None]
    y = (y + D * x) * F.silu(z.reshape(b, H, P))
    return y.reshape(b, 1, H * P), conv, state


def _inputs(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    proj = m.in_proj.weight.shape[0]
    conv_shape, ssm_shape = m.mamba_state_shapes_per_request()
    zxBCdt = torch.randn(BATCH, 1, proj, generator=g, device="cuda")
    conv = torch.randn(BATCH, *conv_shape, generator=g, device="cuda")
    ssm = 0.1 * torch.randn(BATCH, *ssm_shape, generator=g, device="cuda")
    return zxBCdt, conv, ssm


def _decode(m, zxBCdt, conv0, ssm0):
    conv, ssm = conv0.clone(), ssm0.clone()
    with torch.no_grad(), InferenceMode.active():
        y = m.ssm_decode(zxBCdt, conv_state=conv, ssm_state=ssm, batch_indices=None)
    return y, conv, ssm


def _mismatches(m, zxBCdt, conv0, ssm0):
    y, conv, ssm = _decode(m, zxBCdt, conv0, ssm0)
    y_ref, conv_ref, ssm_ref = reference_decode_step(m, zxBCdt, conv0, ssm0)
    out = {}
    checks = (("y", y, y_ref), ("conv_state", conv, conv_ref), ("ssm_state", ssm, ssm_ref))
    for name, got, want in checks:
        assert got.shape == want.shape, f"{name} shape {tuple(got.shape)} vs {tuple(want.shape)}"
        if not torch.allclose(got.double(), want, atol=ATOL, rtol=RTOL):
            out[name] = (got.double() - want).abs().max().item()
    return out, y


def _set_D(m, values):
    with torch.no_grad():
        m.D.copy_(values.to(m.D.dtype))


# ---------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------


def test_control_per_head_D_matches_reference(model_parallel):
    """CONTROL: D_has_hdim=False with a random per-head D: the decode step equals the oracle in
    output and both states. Validates the oracle and harness."""
    m = _mixer(d_has_hdim=False)
    g = torch.Generator(device="cuda").manual_seed(1)
    _set_D(m, torch.randn(m.nheads_local_tp, generator=g, device="cuda"))
    zxBCdt, conv0, ssm0 = _inputs(m, seed=2)
    bad, _ = _mismatches(m, zxBCdt, conv0, ssm0)
    assert not bad, f"per-head decode vs oracle mismatches (max abs err): {bad}"


def test_control_d_has_hdim_uniform_D_matches_reference(model_parallel):
    """CONTROL: D_has_hdim=True with D one constant (!= 1): every indexing of D agrees, and the
    decode step equals the oracle. The D_has_hdim mixer is otherwise sound."""
    m = _mixer(d_has_hdim=True)
    assert m.D.shape == (m.nheads_local_tp * m.headdim,)
    _set_D(m, torch.full_like(m.D, 0.37))
    zxBCdt, conv0, ssm0 = _inputs(m, seed=3)
    bad, _ = _mismatches(m, zxBCdt, conv0, ssm0)
    assert not bad, f"uniform-D decode vs oracle mismatches (max abs err): {bad}"


# ---------------------------------------------------------------------------------------------
# Defect
# ---------------------------------------------------------------------------------------------


def test_defect_d_has_hdim_decode_uses_per_channel_D(model_parallel):
    """DEFECT: with D_has_hdim=True and a per-channel D, the decode step must apply
    D[head * headdim + dim] to each channel."""
    m = _mixer(d_has_hdim=True)
    g = torch.Generator(device="cuda").manual_seed(4)
    _set_D(m, torch.randn(m.nheads_local_tp * m.headdim, generator=g, device="cuda"))
    zxBCdt, conv0, ssm0 = _inputs(m, seed=5)
    bad, y = _mismatches(m, zxBCdt, conv0, ssm0)

    # Attribution: what the decode would produce if it read D[head] (the main-branch indexing).
    H, P = m.nheads_local_tp, m.headdim
    D_saved = m.D.detach().clone()
    _set_D(m, D_saved[:H].repeat_interleave(P))
    y_head_indexed, _, _ = reference_decode_step(m, zxBCdt, conv0, ssm0)
    _set_D(m, D_saved)
    err_head_indexed = (y.double() - y_head_indexed).abs().max().item()

    assert not bad, (
        "DEFECT s10: MambaMixer.ssm_decode with D_has_hdim=True applies the wrong skip term: "
        f"max abs err vs the per-channel oracle {bad} (states should match; only y is "
        "affected). mamba_mixer.py:1277 builds repeat(self.D, 'h -> h p') from a (nheads*headdim,) "
        "D, so selective_state_update reads self.D[head] for every channel; the decode output "
        f"matches that D[head]-indexed formula to {err_head_indexed:.2e}"
    )
