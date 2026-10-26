# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for defect d4: ``InferenceTopKRouter`` with ``moe_router_fusion=True`` returns
``[T, E]`` tensors instead of ``[T, topk]``.

``InferenceTopKRouter._forward`` (``megatron/core/transformer/moe/router.py:1442-1487``) always
asks for ``dense_output=True`` and passes ``fused=config.moe_router_fusion``. In
``topk_routing_with_score_function`` (``megatron/core/transformer/moe/moe_utils.py:792``) the
fused branch returns ``fused_topk_with_score_function(**kwargs)`` directly (``:891-925``); it
neither forwards ``dense_output`` nor reaches the ``if dense_output: return probs, top_indices``
at ``:1036``. TE's fused router returns sparse ``probs [T, E]`` and a boolean
``routing_map [T, E]``, so the inference router hands the MoE layer ``[T, E]`` probabilities and
a ``[T, E]`` bool map where ``[T, topk]`` fp32 probabilities and ``[T, topk]`` int64 expert ids are
expected. (The training ``TopKRouter`` handles fusion + dense output itself,
``router.py:1147-1160``.)
In batch-invariant mode the router calls the eager function (``:1468-1472``); otherwise the
``@torch.compile``'d ``_compiled_topk_routing`` (``:1412-1440``). Both are exercised.

Downstream (traced by reading, see audit/step1/d4-router-fusion-shape.md): no shape check
exists on the inference MoE path at EP=1. The vllm and torch backends take ``topk`` from
``routing_map.shape[1]`` (= E) and read the bool map as expert ids 0/1, so every token is
computed by expert 1 with weight ``sum_j p_tj`` (plus zero-weighted expert-0 work). The
layer-level defect test checks the output and reports its distance to that prediction.

References are plain torch written from the routing definition, applied to the router's own
gating logits (gating is not under test). Router construction follows
``tests/unit_tests/inference/test_moe_dispatching_and_routing.py::TestInferenceTopKRouter``; the
layer construction follows ``...::TestNVLSAllGatherVDispatcher::test_batch_invariant_moe_matches_
training`` at EP=1.

Single GPU. Fused cases need Transformer Engine's ``fused_topk_with_score_function`` (TE >= 2.7)
and are skipped, not passed, without it. Layer-level cases need TE (TEGroupedMLP), Triton (vllm)
and torch grouped_mm on SM90+ (torch).
"""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Max GPUs any test in this file needs.
REQUIRED_GPUS = 1

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="d4 repro needs one CUDA GPU (H100 target)"
)

try:
    import transformer_engine  # noqa: F401  pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

SEED = 4321

# Router-level shapes.
NUM_TOKENS = 32
NUM_EXPERTS = 16
TOPK = 4

# Layer-level shapes (small nanov3-like layer, squared ReLU experts, no shared experts).
LAYER_TOKENS = 64
LAYER_EXPERTS = 8
LAYER_TOPK = 2
LAYER_BACKENDS = ["vllm", "torch"]

# Router probabilities are fp32 end to end; TE's kernel and torch differ by a few fp32 ULPs.
PROB_ATOL = 1e-5
PROB_RTOL = 1e-4
# Normwise relative error of the BF16 inference MoE vs an fp32 plain-torch MoE (~5e-3 noise).
LAYER_TOL = 2e-2

_BASE = dict(
    num_layers=4,
    hidden_size=128,
    ffn_hidden_size=128,
    num_attention_heads=4,
    num_query_groups=2,
    num_moe_experts=NUM_EXPERTS,
    moe_ffn_hidden_size=128,
    moe_router_topk=TOPK,
    moe_router_dtype="fp32",
    moe_shared_expert_intermediate_size=None,
    moe_shared_expert_overlap=False,
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
    # Router logits with std ~1.1, so top-k gaps are wide and probabilities are well spread.
    init_method_std=0.1,
)

# The two score functions InferenceTopKRouter accepts for ordinary (non-hash) layers.
SCORE_CASES = {
    # nanov3 / DeepSeek style: sigmoid + expert bias (selection only) + normalization + scaling.
    "sigmoid": dict(
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_topk_scaling_factor=2.5,
        moe_router_pre_softmax=False,
    ),
    # Mixtral / Qwen style: top-k on logits, softmax over the selected k.
    "softmax": dict(
        moe_router_score_function="softmax",
        moe_router_enable_expert_bias=False,
        moe_router_topk_scaling_factor=None,
        moe_router_pre_softmax=False,
    ),
}
ROUTER_MODES = ["compiled", "eager"]


def setup_module(module):
    if torch.cuda.is_available():
        Utils.initialize_model_parallel(1, 1)


def teardown_module(module):
    if torch.cuda.is_available():
        Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def _isolated_inference_dispatcher_state():
    """Private valid-token scalar, no leaked NVLS buffers (see the d2 repro for the rationale)."""
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


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _require_te_fused_router():
    from megatron.core.transformer.moe import moe_utils

    if not moe_utils.HAVE_TE or moe_utils.fused_topk_with_score_function is None:
        pytest.skip("TE fused_topk_with_score_function unavailable (needs TE >= 2.7)")


def _require_layer_backend(backend: str) -> None:
    if not HAVE_TE:
        pytest.skip("Transformer Engine is required for TEGroupedMLP experts")
    if backend == "vllm":
        try:
            import triton  # noqa: F401  pylint: disable=unused-import
        except ImportError:
            pytest.skip("inference_grouped_gemm_backend='vllm' needs Triton")
    elif backend == "torch":
        from megatron.core.inference.moe.fused_moe import HAVE_GROUPED_MM

        if not HAVE_GROUPED_MM:
            pytest.skip("inference_grouped_gemm_backend='torch' needs torch grouped_mm (>= 2.10)")
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("torch BF16 grouped_mm needs SM90+")


def _seed() -> None:
    from megatron.training.initialize import _set_random_seed

    _set_random_seed(seed_=SEED, data_parallel_random_init=False)


def _make_config(score: str, fusion: bool, **overrides) -> TransformerConfig:
    return TransformerConfig(
        **{**_BASE, **SCORE_CASES[score], "moe_router_fusion": fusion, **overrides}
    )


def _set_expert_bias(router: torch.nn.Module) -> None:
    """Nonzero bias so selection (score + bias) and probabilities (score) differ."""
    if router.expert_bias is None:
        return
    gen = torch.Generator(device="cuda").manual_seed(SEED + 7)
    bias = 0.05 * torch.randn(router.expert_bias.numel(), device="cuda", generator=gen)
    router.expert_bias.copy_(bias.to(router.expert_bias.dtype))


def _make_router(config: TransformerConfig) -> torch.nn.Module:
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.router import InferenceTopKRouter

    router = InferenceTopKRouter(config=config, pg_collection=get_default_pg_collection())
    router = router.cuda().eval()
    _set_expert_bias(router)
    return router


def _hidden(num_tokens: int, hidden_size: int, offset: int = 1) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(SEED + offset)
    # [seq, batch=1, hidden]: InferenceTopKRouter squeezes dim 1.
    x = torch.randn(num_tokens, 1, hidden_size, device="cuda", generator=gen)
    return x.to(torch.bfloat16)


def _routing_kwargs(config: TransformerConfig, expert_bias) -> dict:
    return dict(
        topk=config.moe_router_topk,
        use_pre_softmax=config.moe_router_pre_softmax,
        num_groups=None,
        group_topk=None,
        scaling_factor=config.moe_router_topk_scaling_factor,
        score_function=config.moe_router_score_function,
        expert_bias=expert_bias,
    )


def _reference_topk(
    logits,
    topk,
    use_pre_softmax,
    scaling_factor,
    score_function,
    expert_bias,
    num_groups=None,
    group_topk=None,
):
    """Dense top-k routing from its definition; returns (probs, ids) sorted by expert id."""
    assert num_groups is None and group_topk is None, "group-limited routing not modelled"
    logits = logits.float()
    if score_function == "softmax":
        if use_pre_softmax:
            probs, ids = torch.softmax(logits, dim=-1).topk(topk, dim=1)
        else:
            top_logits, ids = logits.topk(topk, dim=1)
            probs = torch.softmax(top_logits, dim=-1)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits)
        selection = scores if expert_bias is None else scores + expert_bias.float()
        ids = selection.topk(topk, dim=1).indices  # bias steers selection only
        probs = scores.gather(1, ids)
        if topk > 1:
            probs = probs / probs.sum(dim=-1, keepdim=True)
    else:
        raise AssertionError(f"no reference for score function {score_function!r}")
    if scaling_factor:
        probs = probs * scaling_factor
    return _canonical(probs, ids)


def _canonical(probs: torch.Tensor, ids: torch.Tensor):
    """Order each token's (prob, expert) pairs by expert id (inference top-k is unsorted)."""
    order = ids.argsort(dim=1)
    return probs.gather(1, order), ids.gather(1, order)


def _assert_dense_routing(probs, ids, ref_probs, ref_ids, what: str, prefix: str) -> None:
    """Shape [T, topk], integer ids, same expert set and probabilities as the reference."""
    shape = tuple(ref_ids.shape)
    assert tuple(probs.shape) == shape and tuple(ids.shape) == shape, (
        f"{prefix}{what} returned probs {tuple(probs.shape)} ({probs.dtype}) and indices "
        f"{tuple(ids.shape)} ({ids.dtype}); expected [num_tokens, topk] = {shape}. The TE fused "
        f"branch of topk_routing_with_score_function returns its sparse [T, E] outputs before "
        f"dense_output is honoured (moe_utils.py:891-925 vs :1036)."
    )
    assert not ids.dtype.is_floating_point and ids.dtype != torch.bool, (
        f"{prefix}{what} returned expert ids of dtype {ids.dtype}; expected an integer dtype"
    )
    got_probs, got_ids = _canonical(probs.float(), ids.long())
    bad_rows = (got_ids != ref_ids).any(dim=1)
    assert not bad_rows.any(), (
        f"{prefix}{what} selected different experts than the reference for "
        f"{int(bad_rows.sum())} of {shape[0]} tokens (first: token "
        f"{int(bad_rows.nonzero()[0])}: {got_ids[bad_rows][0].tolist()} vs "
        f"{ref_ids[bad_rows][0].tolist()})"
    )
    torch.testing.assert_close(
        got_probs,
        ref_probs,
        atol=PROB_ATOL,
        rtol=PROB_RTOL,
        msg=lambda m: f"{prefix}{what} probabilities differ from the reference.\n{m}",
    )


def _route(router, hidden, mode: str, monkeypatch):
    """Run the inference router; 'eager' takes the batch-invariant (non-compiled) branch."""
    import megatron.core.transformer.moe.router as router_module

    if mode == "eager":
        # Only selects the eager routing call at router.py:1468-1472; gating and the routing
        # function itself are unchanged (moe_utils reads its own batch-invariant flag).
        monkeypatch.setattr(router_module, "is_batch_invariant_mode_enabled", lambda: True)
    with torch.no_grad(), InferenceMode.active():
        probs, ids = router(hidden)
    with torch.no_grad():
        logits = router.gating(hidden).reshape(hidden.shape[0], -1)
    return probs, ids, logits


# ──────────────────────────────────────────────────────────────────────
# Router-level tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ROUTER_MODES)
@pytest.mark.parametrize("score", list(SCORE_CASES))
def test_control_unfused_inference_router_matches_reference(score, mode, monkeypatch):
    """CONTROL: moe_router_fusion=False gives [T, topk] probs/ids equal to the reference."""
    _seed()
    config = _make_config(score, fusion=False)
    router = _make_router(config)
    probs, ids, logits = _route(router, _hidden(NUM_TOKENS, config.hidden_size), mode, monkeypatch)
    ref_probs, ref_ids = _reference_topk(logits, **_routing_kwargs(config, router.expert_bias))
    _assert_dense_routing(
        probs, ids, ref_probs, ref_ids, f"unfused InferenceTopKRouter ({mode})", "CONTROL: "
    )


@pytest.mark.parametrize("score", list(SCORE_CASES))
def test_control_te_fused_topk_sparse_output_matches_reference(score):
    """CONTROL: TE's fused kernel itself is right; it just returns the sparse [T, E] form.

    Pins the TE contract the defect relies on (bool [T, E] map, sparse [T, E] probs) and shows
    that densifying it gives the reference, so the defect is the missing densification only.
    """
    _require_te_fused_router()
    from megatron.core.transformer.moe import moe_utils

    _seed()
    config = _make_config(score, fusion=True)
    router = _make_router(config)
    with torch.no_grad():
        logits = router.gating(_hidden(NUM_TOKENS, config.hidden_size)).reshape(NUM_TOKENS, -1)
        kwargs = _routing_kwargs(config, router.expert_bias)
        probs_te, map_te = moe_utils.fused_topk_with_score_function(logits=logits, **kwargs)
    ref_probs, ref_ids = _reference_topk(logits, **kwargs)

    assert map_te.shape == (NUM_TOKENS, NUM_EXPERTS) and map_te.dtype == torch.bool, (
        f"CONTROL: TE fused router contract changed: routing map {tuple(map_te.shape)} "
        f"{map_te.dtype}, expected bool [T, E]; re-check the d4 premise"
    )
    assert probs_te.shape == (NUM_TOKENS, NUM_EXPERTS), tuple(probs_te.shape)
    counts = map_te.sum(dim=1)
    assert (counts == TOPK).all(), f"CONTROL: TE selected {counts.tolist()} experts per token"
    assert (probs_te[~map_te] == 0).all(), "CONTROL: TE probs nonzero outside the routing map"
    # nonzero() is row-major, so each row's expert ids come out ascending.
    ids = map_te.nonzero()[:, 1].view(NUM_TOKENS, TOPK)
    _assert_dense_routing(
        probs_te.gather(1, ids), ids, ref_probs, ref_ids, "densified TE fused router", "CONTROL: "
    )


@pytest.mark.parametrize("score", list(SCORE_CASES))
def test_defect_fused_topk_routing_honours_dense_output(score):
    """DEFECT d4: topk_routing_with_score_function(fused=True, dense_output=True) -> [T, topk]."""
    _require_te_fused_router()
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

    gen = torch.Generator(device="cuda").manual_seed(SEED + 3)
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, device="cuda", generator=gen)
    config = _make_config(score, fusion=True)
    expert_bias = None
    if config.moe_router_enable_expert_bias:
        expert_bias = 0.05 * torch.randn(NUM_EXPERTS, device="cuda", generator=gen)
    kwargs = _routing_kwargs(config, expert_bias)
    ref_probs, ref_ids = _reference_topk(logits, **kwargs)

    with torch.no_grad():
        probs, ids = topk_routing_with_score_function(
            logits, fused=True, dense_output=True, **kwargs
        )
    _assert_dense_routing(
        probs,
        ids,
        ref_probs,
        ref_ids,
        "topk_routing_with_score_function(fused=True, dense_output=True)",
        "DEFECT d4: ",
    )


@pytest.mark.parametrize("mode", ROUTER_MODES)
@pytest.mark.parametrize("score", list(SCORE_CASES))
def test_defect_fused_inference_router_returns_topk(score, mode, monkeypatch):
    """DEFECT d4: InferenceTopKRouter(moe_router_fusion=True) must keep the [T, topk] contract."""
    _require_te_fused_router()
    _seed()
    config = _make_config(score, fusion=True)
    router = _make_router(config)
    probs, ids, logits = _route(router, _hidden(NUM_TOKENS, config.hidden_size), mode, monkeypatch)
    ref_probs, ref_ids = _reference_topk(logits, **_routing_kwargs(config, router.expert_bias))
    _assert_dense_routing(
        probs,
        ids,
        ref_probs,
        ref_ids,
        f"InferenceTopKRouter(moe_router_fusion=True, {mode})",
        "DEFECT d4: ",
    )


# ──────────────────────────────────────────────────────────────────────
# Layer-level tests: what the inference MoE does with the router output
# ──────────────────────────────────────────────────────────────────────


def _layer_case(backend: str, fusion: bool):
    """Inference MoE layer output, a plain-torch MoE reference, and the misroute prediction."""
    from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec

    _seed()
    config = _make_config(
        "sigmoid",
        fusion=fusion,
        num_moe_experts=LAYER_EXPERTS,
        moe_router_topk=LAYER_TOPK,
        inference_grouped_gemm_backend=backend,
    )
    layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
    _set_expert_bias(layer.router)
    hidden = _hidden(LAYER_TOKENS, config.hidden_size, offset=11)

    with torch.no_grad(), InferenceMode.active():
        output, _ = layer(hidden.clone())
    output = output.float().reshape(LAYER_TOKENS, -1)

    with torch.no_grad():
        logits = layer.router.gating(hidden).reshape(LAYER_TOKENS, -1)
        ref_probs, ref_ids = _reference_topk(
            logits, **_routing_kwargs(config, layer.router.expert_bias)
        )
        x = hidden.reshape(LAYER_TOKENS, -1).float()
        expert_out = []
        for e in range(LAYER_EXPERTS):
            w1 = getattr(layer.experts.linear_fc1, f"weight{e}").detach().float()  # [ffn, H]
            w2 = getattr(layer.experts.linear_fc2, f"weight{e}").detach().float()  # [H, ffn]
            expert_out.append(torch.relu(x @ w1.t()).square() @ w2.t())  # squared-ReLU expert
        reference = torch.zeros_like(x)
        for e in range(LAYER_EXPERTS):
            weight = (ref_probs * (ref_ids == e)).sum(dim=1, keepdim=True)
            reference += weight * expert_out[e]
        # Bool [T, E] map read as expert ids: True -> expert 1 with the selected prob, False ->
        # expert 0 with prob 0. So out[t] = (sum_j p_tj) * E_1(x_t).
        misroute = ref_probs.sum(dim=1, keepdim=True) * expert_out[1]
    return output, reference, misroute


def _rel_err(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual - reference).norm() / reference.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("backend", LAYER_BACKENDS)
def test_control_unfused_router_moe_layer_matches_reference(backend):
    """CONTROL: inference MoE layer with the unfused router matches a plain-torch MoE."""
    _require_layer_backend(backend)
    output, reference, misroute = _layer_case(backend, fusion=False)
    assert reference.std().item() > 5e-2, f"reference too small: {reference.std().item():.3e}"
    err = _rel_err(output, reference)
    assert err <= LAYER_TOL, (
        f"CONTROL failed: unfused-router inference MoE ({backend}) vs plain-torch MoE: normwise "
        f"relative error {err:.3e} > {LAYER_TOL}"
    )
    # The misroute prediction must be far from the correct output, or the defect test below
    # could not tell the two apart.
    separation = _rel_err(misroute, reference)
    assert separation > 10 * LAYER_TOL, (
        f"HARNESS: misroute prediction is only {separation:.3e} away from the correct output"
    )


@pytest.mark.parametrize("backend", LAYER_BACKENDS)
def test_defect_fused_router_moe_layer_matches_reference(backend):
    """DEFECT d4: with moe_router_fusion=True the inference MoE output must still be correct."""
    _require_layer_backend(backend)
    _require_te_fused_router()
    output, reference, misroute = _layer_case(backend, fusion=True)
    err = _rel_err(output, reference)
    err_misroute = _rel_err(output, misroute)
    assert err <= LAYER_TOL, (
        f"DEFECT d4: inference MoE layer ({backend}, EP=1) with moe_router_fusion=True is wrong: "
        f"normwise relative error {err:.3e} vs the plain-torch MoE (tolerance {LAYER_TOL}). "
        f"Distance to the predicted misroute (bool [T, E] map read as expert ids, every token "
        f"computed by expert 1 with weight sum_j p_tj): {err_misroute:.3e}. No exception was "
        f"raised: the wrong routing is silent."
    )
