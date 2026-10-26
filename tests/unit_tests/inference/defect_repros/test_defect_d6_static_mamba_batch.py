# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for defect d6: static-batching Mamba decode with batch > 1.

``MambaMixer._static_decode`` (``megatron/core/ssm/mamba_mixer.py:573-591``) projects the
sequence-first decode input ``(1, b, d_model)`` to ``zxBCdt`` of shape ``(1, b, proj)`` and passes
it unchanged to ``ssm_decode``, whose contract is batch-first ``(batch, seq, d)``
(``mamba_mixer.py:1107-1134``). With ``b > 1`` the b requests are therefore decoded as one
request's b-token (speculative) sequence on state slot 0; slots 1..b-1 are never touched. The
output shape ``(1, b, d_inner)`` is still what ``out_proj`` expects, so nothing fails loudly.
GDP's ``_static_decode`` transposes to batch-first (``gated_delta_product.py:1077-1084``).

Oracle (metamorphic): one static decode step over b=2 requests, with distinct inputs and distinct
per-request conv/SSM states, must equal two independent b=1 static decode steps, in outputs and in
the updated states. At b=1 the two layouts coincide, so the b=1 runs are correct by construction.

Construction follows ``tests/unit_tests/ssm/test_mamba_mixer.py::TestMambaMixer.get_mixer``
(``hybrid_stack_spec`` mixer submodules, TP=1). The static context's per-layer state cache is
seeded directly, as a prior static prefill would leave it, so the prefill kernels are not needed.

Needs one GPU, Transformer Engine (``hybrid_stack_spec`` in_proj/out_proj) and ``mamba_ssm`` (the
mixer's gated RMSNorm). The decode kernels are Megatron's in-tree Triton kernels.
"""

from unittest import mock

import pytest
import torch

from megatron.core.inference.contexts.static_context import StaticInferenceContext
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
    pytest.mark.skipif(not torch.cuda.is_available(), reason="d6 repro needs one CUDA GPU"),
    pytest.mark.skipif(
        not HAVE_MAMBA_SSM, reason="MambaMixer(rmsnorm=True) needs mamba_ssm's gated RMSNorm"
    ),
    pytest.mark.skipif(not HAVE_TE, reason="hybrid_stack_spec in_proj/out_proj need TE"),
]

HIDDEN = 256
# fp32 throughout. The only cross-batch-size arithmetic difference is the in/out projection GEMM
# with M=2 vs M=1; the decode kernels run one program per request row.
ATOL = 1e-4
RTOL = 1e-4


@pytest.fixture
def mixer():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        hidden_size=HIDDEN, num_layers=1, num_attention_heads=1, use_cpu_initialization=True
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    m = MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        layer_number=1,
        pg_collection=pg_collection,
    )
    m.cuda()
    m.eval()
    yield m
    Utils.destroy_model_parallel()


def _random_states(m: MambaMixer, batch: int, seed: int):
    """Distinct per-request conv/SSM states, shaped and typed as _get_states_from_cache makes them."""
    conv_shape, ssm_shape = m.mamba_state_shapes_per_request()
    g = torch.Generator(device="cuda").manual_seed(seed)
    conv = torch.randn(batch, *conv_shape, generator=g, device="cuda").to(m.conv1d_weight.dtype)
    ssm = torch.randn(batch, *ssm_shape, generator=g, device="cuda").to(m.in_proj.weight.dtype)
    return conv, ssm


def _random_hidden(batch: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, batch, HIDDEN, generator=g, device="cuda")


def _static_decode_step(m: MambaMixer, hidden, conv_state, ssm_state, offset: int = 5):
    """One production static-batching decode step through MambaMixer.forward.

    Seeds the StaticInferenceContext cache with (conv_state, ssm_state), which the step updates
    in place, and spies on _static_decode to prove the decode branch ran.
    """
    batch = hidden.shape[1]
    ctx = StaticInferenceContext(max_batch_size=batch, max_sequence_length=16)
    ctx.sequence_len_offset = offset
    # HybridStack.forward mirrors these two attributes for static batching
    # (megatron/core/models/hybrid/hybrid_block.py:604-610); MambaMixer.forward reads seqlen_offset.
    ctx.max_seqlen = ctx.max_sequence_length
    ctx.seqlen_offset = ctx.sequence_len_offset
    # What a prior static prefill of `batch` requests leaves behind: _get_states_from_cache returns
    # this pair untouched when the batch size matches and sequence_len_offset > 0.
    ctx.key_value_memory_dict[m.layer_number] = (conv_state, ssm_state)
    m.cached_batch_size = batch

    with mock.patch.object(m, "_static_decode", wraps=m._static_decode) as spy:
        with torch.no_grad(), InferenceMode.active():
            out, _ = m(hidden, inference_context=ctx)
    assert spy.call_count == 1, "harness: static decode branch was not taken"
    got_conv, got_ssm = ctx.key_value_memory_dict[m.layer_number]
    assert got_conv is conv_state and got_ssm is ssm_state, "harness: state cache was reallocated"
    return out


def _batch_first_decode_step(m: MambaMixer, hidden, conv_state, ssm_state):
    """Same step through ssm_decode's documented batch-first contract (b, s, d), the layout the
    dynamic path builds (ssm_inference.py:237-247) and GDP's static decode transposes to."""
    with torch.no_grad(), InferenceMode.active():
        zxBCdt, _ = m.in_proj(hidden)  # (1, b, proj)
        y = m.ssm_decode(
            zxBCdt.transpose(0, 1).contiguous(),  # (b, 1, proj)
            conv_state=conv_state,
            ssm_state=ssm_state,
            batch_indices=None,
        )  # (b, 1, d_inner)
        out, _ = m.out_proj(y.transpose(0, 1))
    return out


def _max_err(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def _close(a, b, atol=ATOL, rtol=RTOL) -> bool:
    return bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))


def test_control_ssm_decode_batch_first_is_batch_separable(mixer):
    """CONTROL: given the batch-first layout it documents, ssm_decode over 2 requests equals two
    1-request calls (outputs and both state slots), and slot 1's state visibly changes."""
    m = mixer
    proj = m.in_proj.weight.shape[0]
    g = torch.Generator(device="cuda").manual_seed(7)
    zxBCdt = torch.randn(2, 1, proj, generator=g, device="cuda")
    conv0, ssm0 = _random_states(m, 2, seed=11)

    conv2, ssm2 = conv0.clone(), ssm0.clone()
    with torch.no_grad(), InferenceMode.active():
        y2 = m.ssm_decode(zxBCdt, conv_state=conv2, ssm_state=ssm2, batch_indices=None)
        for i in range(2):
            conv1, ssm1 = conv0[i : i + 1].clone(), ssm0[i : i + 1].clone()
            y1 = m.ssm_decode(
                zxBCdt[i : i + 1], conv_state=conv1, ssm_state=ssm1, batch_indices=None
            )
            assert _close(y2[i : i + 1], y1, 1e-5, 1e-5), f"y[{i}] err {_max_err(y2[i:i+1], y1)}"
            assert _close(conv2[i : i + 1], conv1, 1e-5, 1e-5), f"conv[{i}]"
            assert _close(ssm2[i : i + 1], ssm1, 1e-5, 1e-5), f"ssm[{i}]"
    assert not torch.equal(conv2[1], conv0[1]), "conv slot 1 not updated"
    assert not torch.equal(ssm2[1], ssm0[1]), "ssm slot 1 not updated"


def test_control_static_decode_batch1_matches_batch_first_contract(mixer):
    """CONTROL: at b=1 (where sequence-first and batch-first coincide) the production static
    decode step equals the batch-first reference step, outputs and states. Validates the harness
    (context seeding, seqlen_offset, InferenceMode, spy)."""
    m = mixer
    hidden = _random_hidden(1, seed=3)
    conv0, ssm0 = _random_states(m, 1, seed=5)

    conv_s, ssm_s = conv0.clone(), ssm0.clone()
    out_s = _static_decode_step(m, hidden, conv_s, ssm_s)
    conv_r, ssm_r = conv0.clone(), ssm0.clone()
    out_r = _batch_first_decode_step(m, hidden, conv_r, ssm_r)

    assert out_s.shape == (1, 1, HIDDEN)
    torch.testing.assert_close(out_s, out_r, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(conv_s, conv_r, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ssm_s, ssm_r, atol=1e-6, rtol=1e-6)
    assert not torch.equal(ssm_s, ssm0), "the decode step did not update the SSM state"


def test_defect_static_decode_batch2_equals_two_batch1(mixer):
    """DEFECT: a static decode step over 2 requests must equal two independent 1-request static
    decode steps, and must update both requests' state slots."""
    m = mixer
    batch = 2
    hidden = _random_hidden(batch, seed=17)
    conv0, ssm0 = _random_states(m, batch, seed=19)

    conv_b, ssm_b = conv0.clone(), ssm0.clone()
    out_b = _static_decode_step(m, hidden, conv_b, ssm_b)
    assert out_b.shape == (1, batch, HIDDEN)  # shape is right on main; values are not

    problems = []
    for i in range(batch):
        conv_1, ssm_1 = conv0[i : i + 1].clone(), ssm0[i : i + 1].clone()
        out_1 = _static_decode_step(m, hidden[:, i : i + 1], conv_1, ssm_1)
        checks = [
            ("output", out_b[:, i : i + 1], out_1),
            ("conv_state", conv_b[i : i + 1], conv_1),
            ("ssm_state", ssm_b[i : i + 1], ssm_1),
        ]
        for name, got, want in checks:
            if not _close(got, want):
                problems.append(f"request {i} {name} max err {_max_err(got, want):.3g}")
    for name, after, before in (("conv_state", conv_b, conv0), ("ssm_state", ssm_b, ssm0)):
        if torch.equal(after[1], before[1]):
            problems.append(f"slot 1 {name} never updated")

    assert not problems, (
        "DEFECT d6: static-batching Mamba decode with batch=2 is not two independent "
        "requests: _static_decode passes the sequence-first (1, b, proj) projection to "
        "ssm_decode, which reads it as (batch=1, seq=b), so request 1 is decoded as a draft "
        "token of request 0 on state slot 0. Mismatches vs two batch=1 steps: "
        + "; ".join(problems)
    )
