# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Repro for defect d9: inference-optimized MoE crashes with top-1 routing.

``InferenceTopKRouter.forward`` ends with ``return probs.squeeze(1), top_indices.squeeze(1)``
(``megatron/core/transformer/moe/router.py:1487``). The routing helper already returns
``[num_tokens, topk]``, so with ``moe_router_topk=1`` the squeeze removes the top-k dimension
and the router returns ``[num_tokens]``. Every consumer indexes ``routing_map.shape[1]``
(``vllm_fused_moe.py:756``, ``permute.py:385``) and raises ``IndexError``. Found while running
the d2 repro (``moe_apply_probs_on_input`` requires topk=1).

Single GPU; the layer cases need Transformer Engine (TEGroupedMLP experts).
"""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

REQUIRED_GPUS = 1

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="d9 repro needs a CUDA GPU")

SEED = 9901
NUM_TOKENS = 64
NUM_EXPERTS = 8
LAYER_TOL = 2e-2
LAYER_BACKENDS = ["vllm", "torch"]
ROUTER_MODES = ["compiled", "eager"]

_BASE = dict(
    num_layers=4,
    hidden_size=128,
    ffn_hidden_size=128,
    num_attention_heads=4,
    num_query_groups=2,
    num_moe_experts=NUM_EXPERTS,
    moe_ffn_hidden_size=128,
    moe_router_dtype="fp32",
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=False,
    moe_router_pre_softmax=False,
    moe_grouped_gemm=True,
    moe_token_dispatcher_type="alltoall",
    moe_aux_loss_coeff=0.0,
    activation_func=squared_relu,
    gated_linear_unit=False,
    normalization="RMSNorm",
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    transformer_impl="inference_optimized",
    expert_tensor_parallel_size=1,
    use_cpu_initialization=True,
    attention_backend=AttnBackend.local,
    init_method_std=0.1,
)


def setup_module(module):
    if torch.cuda.is_available():
        Utils.initialize_model_parallel(1, 1)


def teardown_module(module):
    if torch.cuda.is_available():
        Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def _isolated_inference_dispatcher_state():
    """Private valid-token scalar and no leaked NVLS buffers between cases."""
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
        NVLSAllGatherVDispatcher,
    )

    base, nvls = InferenceAllGatherDispatcherBase, NVLSAllGatherVDispatcher
    saved = (
        base._valid_tokens_tensor,
        base._host_valid_tokens_estimate,
        nvls._real_token_count_tensor,
        nvls._symm_rsv,
    )
    base.allocate_valid_tokens_tensor()
    nvls._real_token_count_tensor = None
    nvls._symm_rsv = None
    try:
        yield
    finally:
        (
            base._valid_tokens_tensor,
            base._host_valid_tokens_estimate,
            nvls._real_token_count_tensor,
            nvls._symm_rsv,
        ) = saved


def _seed():
    from megatron.training.initialize import _set_random_seed

    _set_random_seed(seed_=SEED, data_parallel_random_init=False)


def _config(topk: int, **overrides) -> TransformerConfig:
    return TransformerConfig(**{**_BASE, "moe_router_topk": topk, **overrides})


def _hidden(hidden_size: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
    return torch.randn(NUM_TOKENS, 1, hidden_size, device="cuda", generator=gen).to(torch.bfloat16)


def _reference_routing(logits: torch.Tensor, topk: int):
    """Sigmoid top-k routing from its definition (no expert bias); ids sorted per token."""
    scores = torch.sigmoid(logits.float())
    ids = scores.topk(topk, dim=1).indices
    probs = scores.gather(1, ids)
    if topk > 1:
        probs = probs / probs.sum(dim=-1, keepdim=True)
    order = ids.argsort(dim=1)
    return probs.gather(1, order), ids.gather(1, order)


def _route(router, hidden, mode, monkeypatch):
    import megatron.core.transformer.moe.router as router_module

    if mode == "eager":
        monkeypatch.setattr(router_module, "is_batch_invariant_mode_enabled", lambda: True)
    with torch.no_grad(), InferenceMode.active():
        probs, ids = router(hidden)
    with torch.no_grad():
        logits = router.gating(hidden).reshape(hidden.shape[0], -1)
    return probs, ids, logits


def _router_case(topk, mode, monkeypatch):
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.router import InferenceTopKRouter

    _seed()
    config = _config(topk)
    router = InferenceTopKRouter(config=config, pg_collection=get_default_pg_collection())
    router = router.cuda().eval()
    probs, ids, logits = _route(router, _hidden(config.hidden_size), mode, monkeypatch)
    return probs, ids, _reference_routing(logits, topk)


def _require_layer_backend(backend):
    pytest.importorskip("transformer_engine", reason="TEGroupedMLP experts need Transformer Engine")
    if backend == "torch":
        from megatron.core.inference.moe.fused_moe import HAVE_GROUPED_MM

        if not HAVE_GROUPED_MM or torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("inference_grouped_gemm_backend='torch' needs torch grouped_mm on SM90+")


def _layer_case(topk, backend):
    """Inference MoE layer output and a plain-torch MoE reference with the same weights."""
    from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec

    _seed()
    config = _config(topk, inference_grouped_gemm_backend=backend)
    layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
    hidden = _hidden(config.hidden_size)
    with torch.no_grad(), InferenceMode.active():
        output, _ = layer(hidden.clone())
    output = output.float().reshape(NUM_TOKENS, -1)
    with torch.no_grad():
        logits = layer.router.gating(hidden).reshape(NUM_TOKENS, -1)
        ref_probs, ref_ids = _reference_routing(logits, topk)
        x = hidden.reshape(NUM_TOKENS, -1).float()
        reference = torch.zeros_like(x)
        for e in range(NUM_EXPERTS):
            w1 = getattr(layer.experts.linear_fc1, f"weight{e}").detach().float()
            w2 = getattr(layer.experts.linear_fc2, f"weight{e}").detach().float()
            weight = (ref_probs * (ref_ids == e)).sum(dim=1, keepdim=True)
            reference += weight * (torch.relu(x @ w1.t()).square() @ w2.t())
    return output, reference


def _rel_err(actual, reference):
    return ((actual - reference).norm() / reference.norm().clamp_min(1e-30)).item()


# ------------------------------------------------------------------ CONTROL (must pass)


@pytest.mark.parametrize("mode", ROUTER_MODES)
def test_control_topk2_router_returns_topk_columns(mode, monkeypatch):
    """CONTROL: with topk=2 the inference router returns [T, 2] matching the reference."""
    probs, ids, (ref_probs, ref_ids) = _router_case(2, mode, monkeypatch)
    assert tuple(probs.shape) == (NUM_TOKENS, 2) and tuple(ids.shape) == (NUM_TOKENS, 2)
    order = ids.long().argsort(dim=1)
    assert torch.equal(ids.long().gather(1, order), ref_ids)
    torch.testing.assert_close(probs.float().gather(1, order), ref_probs, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("backend", LAYER_BACKENDS)
def test_control_topk2_moe_layer_matches_reference(backend):
    """CONTROL: the inference MoE layer with topk=2 matches the plain-torch MoE."""
    _require_layer_backend(backend)
    output, reference = _layer_case(2, backend)
    assert reference.std().item() > 5e-2, f"reference too small: {reference.std().item():.3e}"
    err = _rel_err(output, reference)
    assert err <= LAYER_TOL, f"CONTROL failed ({backend}, topk=2): relative error {err:.3e}"


# ------------------------------------------------------------------ DEFECT (fail on main)


@pytest.mark.parametrize("mode", ROUTER_MODES)
def test_defect_topk1_router_returns_topk_column(mode, monkeypatch):
    """DEFECT d9: with topk=1 the inference router must still return [T, 1]."""
    probs, ids, (ref_probs, ref_ids) = _router_case(1, mode, monkeypatch)
    assert tuple(probs.shape) == (NUM_TOKENS, 1) and tuple(ids.shape) == (NUM_TOKENS, 1), (
        f"DEFECT d9: InferenceTopKRouter(moe_router_topk=1, {mode}) returned probs "
        f"{tuple(probs.shape)} and indices {tuple(ids.shape)}; expected [num_tokens, 1] = "
        f"({NUM_TOKENS}, 1). router.py:1487 squeezes dim 1, which is the top-k dimension "
        f"when topk == 1."
    )
    assert torch.equal(ids.long(), ref_ids)
    torch.testing.assert_close(probs.float(), ref_probs, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("backend", LAYER_BACKENDS)
def test_defect_topk1_moe_layer_runs_and_matches_reference(backend):
    """DEFECT d9: the inference MoE layer with topk=1 must run and match the plain-torch MoE."""
    _require_layer_backend(backend)
    try:
        output, reference = _layer_case(1, backend)
    except (IndexError, RuntimeError, AssertionError) as exc:
        pytest.fail(
            f"DEFECT d9: inference-optimized MoE layer ({backend}, moe_router_topk=1) raised "
            f"{type(exc).__name__}: {exc}. InferenceTopKRouter returns 1-D routing for topk=1 "
            f"(router.py:1487), and the expert backends index routing_map.shape[1]."
        )
    err = _rel_err(output, reference)
    assert err <= LAYER_TOL, (
        f"DEFECT d9: inference MoE layer ({backend}, topk=1) ran but is wrong: relative error "
        f"{err:.3e} > {LAYER_TOL}"
    )
