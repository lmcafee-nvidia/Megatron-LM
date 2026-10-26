# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for defect d2: inference-optimized MoE silently ignores gated-linear-unit options.

``InferenceGroupedMLP`` (``megatron/core/transformer/moe/experts.py``) maps every gated SiLU
configuration to plain SwiGLU in ``_resolve_mcore_activation_type`` (``:1428-1436``), and its
``vllm`` (default) and ``torch`` forwards (``_vllm_forward`` / ``_mcore_fused_moe_forward``,
``:1712-1748``) pass only ``activation_type`` and the tanh ``activation_clamp_scale`` to the
kernels. The following options are therefore dropped without an error:

* ``activation_func_clamp_value``  (training: ``glu`` clamps gate <= v and |linear| <= v)
* ``glu_linear_offset``            (training: ``silu(gate) * (linear + offset)``)
* ``moe_mlp_glu_interleave_size``  (training: ``_remove_glu_interleaving`` before the GLU)
* ``moe_apply_probs_on_input``     (training: ``x * p`` before FC1, unit weight after; topk=1)

The training path (``TEGroupedMLP.forward``, ``experts.py:1093-1181``) honours all four. Config
validation rejects them only inside the ``inference_moe_nvfp4_recipe`` block
(``transformer_config.py:2199-2208``).

Oracle: the training path of the *same* layer. ``InferenceGroupedMLP`` subclasses
``TEGroupedMLP`` and, outside ``InferenceMode``, delegates to it; ``InferenceTopKRouter`` likewise
delegates to ``TopKRouter``, and ``MoELayer`` switches to its training token dispatcher. Running
one layer object with and without ``InferenceMode.active()`` therefore compares the inference
MoE against training with identical weights. This is the construction used by
``tests/unit_tests/inference/test_moe_dispatching_and_routing.py::TestNVLSAllGatherVDispatcher::
test_batch_invariant_moe_matches_training``, here at EP=1 and outside batch-invariant mode.

Single GPU, EP=1, BF16. Needs Transformer Engine (TEGroupedMLP), Triton (vllm backend) and
``torch.nn.functional.grouped_mm`` / ``torch._grouped_mm`` on SM90+ (torch backend).
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Max GPUs any test in this file needs.
REQUIRED_GPUS = 1

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="d2 repro needs one CUDA GPU (H100 target)"
)

try:
    import transformer_engine  # noqa: F401  pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

BACKENDS = ["vllm", "torch"]

NUM_TOKENS = 64
SEED = 1234

# Normwise relative error ||inference - training|| / ||training||. BF16 rounding differences
# between the two paths (different GEMMs, activation rounding order, fp32 vs bf16 combine) are a
# few 1e-3 here; the option effects below are >= 2e-1 (checked per test, see GUARD).
TOL = 2e-2
# Each option must change the *training* output by at least this much relative to plain SwiGLU,
# otherwise the defect test could pass vacuously.
GUARD = 2e-1

# Small nanov3-like layer (tests/unit_tests/inference/test_moe_dispatching_and_routing.py
# NANOV3_BASE), switched to gated SiLU, without shared experts, and with init_method_std=0.1 so
# FC1 pre-activations have std ~1.1: large enough that the clamp, offset, interleave and
# probability-placement options all move the result far beyond BF16 noise (the 0.02 default
# gives std ~0.2, where e.g. SiLU is near-linear and a 0.5 clamp rarely bites).
_BASE = dict(
    num_layers=4,
    hidden_size=128,
    ffn_hidden_size=128,
    num_attention_heads=4,
    num_query_groups=2,
    num_moe_experts=8,
    moe_ffn_hidden_size=128,
    moe_router_topk=2,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_topk_scaling_factor=2.5,
    moe_router_dtype="fp32",
    moe_shared_expert_intermediate_size=None,
    moe_shared_expert_overlap=False,
    moe_grouped_gemm=True,
    moe_token_dispatcher_type="alltoall",
    moe_aux_loss_coeff=0.0,
    activation_func=F.silu,
    gated_linear_unit=True,
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

# option id -> (option overrides, routing overrides shared with the plain-SwiGLU baseline).
# All combinations are legal under TransformerConfig validation on main:
# - clamp value: finite > 0, MoE, glu_linear_offset == 0, use_te_activation_func False
#   (transformer_config.py:3436-3461);
# - glu_linear_offset != 0: bias_activation_fusion stays False (:3358-3362);
# - interleave: BF16 params (:3315-3321), 2*ffn divisible by 2*size (128 % 32 == 0);
# - moe_apply_probs_on_input: topk must be 1 (runtime assert, experts.py:1094-1096). Routing is
#   pre-softmax top-1 without scaling (softmax + topk=1 requires pre-softmax, :3656-3664), so the
#   routed probability is the max of an 8-way softmax, typically 0.3-0.6. A top-1 sigmoid would
#   select the largest of 8 logits and give p ~ 0.85, too close to 1 for a clear effect.
_OPTIONS = {
    "activation_func_clamp_value": ({"activation_func_clamp_value": 0.5}, {}),
    "glu_linear_offset": ({"glu_linear_offset": 1.0}, {}),
    "moe_mlp_glu_interleave_size": ({"moe_mlp_glu_interleave_size": 32}, {}),
    "moe_apply_probs_on_input": (
        {"moe_apply_probs_on_input": True},
        {
            "moe_router_topk": 1,
            "moe_router_score_function": "softmax",
            "moe_router_pre_softmax": True,
            "moe_router_enable_expert_bias": False,
            "moe_router_topk_scaling_factor": None,
        },
    ),
}


def setup_module(module):
    if torch.cuda.is_available():
        Utils.initialize_model_parallel(1, 1)


def teardown_module(module):
    if torch.cuda.is_available():
        Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def _isolated_inference_dispatcher_state():
    """Give each test a private valid-token scalar and no leaked NVLS buffers.

    The inference dispatchers keep class-level state. In production the dynamic context
    allocates the valid-token scalar (``allocate_valid_tokens_tensor``); here we do it per test.
    A real-token-count tensor or RSV buffer leaked by another test in the same process would
    mask routing rows or redirect the expert output, so both are cleared and restored after.
    """
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


def _require_backend(backend: str) -> None:
    if not HAVE_TE:
        pytest.skip("Transformer Engine is required (TEGroupedMLP is the training oracle)")
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


def _make_config(backend: str, **overrides) -> TransformerConfig:
    return TransformerConfig(**{**_BASE, "inference_grouped_gemm_backend": backend, **overrides})


def _build_layer(config: TransformerConfig) -> torch.nn.Module:
    from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec

    return get_inference_optimized_moe_spec()(config=config).cuda().eval()


def _hidden(config: TransformerConfig) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
    # [seq, batch=1, hidden]: InferenceTopKRouter squeezes dim 1.
    return torch.randn(NUM_TOKENS, 1, config.hidden_size, device="cuda", generator=gen).to(
        torch.bfloat16
    )


def _training_forward(layer: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    assert not InferenceMode.is_active()
    with torch.no_grad():
        out, _ = layer(hidden.clone())
    return out.float()


def _inference_forward(layer: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), InferenceMode.active():
        out, _ = layer(hidden.clone())
    return out.float()


def _rel_err(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual - reference).norm() / reference.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("backend", BACKENDS)
def test_control_plain_swiglu_inference_matches_training(backend):
    """CONTROL: plain SwiGLU, inference-optimized MoE == training MoE (same weights)."""
    _require_backend(backend)
    _seed()
    config = _make_config(backend)
    layer = _build_layer(config)
    hidden = _hidden(config)

    reference = _training_forward(layer, hidden)
    output = _inference_forward(layer, hidden)

    assert output.shape == reference.shape, (output.shape, reference.shape)
    assert torch.isfinite(output).all() and torch.isfinite(reference).all()
    # Non-trivial magnitudes, so TOL is meaningful (unlike 0.01-scaled weights).
    assert reference.std().item() > 5e-2, f"reference too small: std {reference.std().item():.3e}"
    err = _rel_err(output, reference)
    assert err <= TOL, (
        f"CONTROL failed: plain SwiGLU, backend={backend!r}: inference vs training normwise "
        f"relative error {err:.3e} > {TOL}. The harness or a different defect is at fault; the "
        f"d2 defect cases for this backend cannot be interpreted."
    )


@pytest.mark.parametrize("option", list(_OPTIONS))
@pytest.mark.parametrize("backend", BACKENDS)
def test_defect_glu_option_honoured_by_inference_moe(backend, option):
    """DEFECT d2: a legal GLU option must give the same output in inference as in training.

    Also passes if the inference configuration rejects the option loudly (config or layer
    construction raises an error naming it): that is an acceptable fix for a silent drop.
    """
    _require_backend(backend)
    option_overrides, routing_overrides = _OPTIONS[option]
    option_name = next(iter(option_overrides))
    _seed()

    plain_config = _make_config(backend, **routing_overrides)
    try:
        config = _make_config(backend, **routing_overrides, **option_overrides)
        layer = _build_layer(config)
    except (ValueError, AssertionError, NotImplementedError) as exc:
        # Loud rejection fixes the silent drop. Guard against unrelated failures.
        assert option_name in str(exc), (
            f"HARNESS: building the {option} layer failed for an unrelated reason: {exc!r}"
        )
        return

    plain_layer = _build_layer(plain_config)
    plain_layer.load_state_dict(layer.state_dict())
    hidden = _hidden(config)

    reference = _training_forward(layer, hidden)  # honours the option
    plain_reference = _training_forward(plain_layer, hidden)  # same weights, option off
    effect = _rel_err(plain_reference, reference)
    assert effect >= GUARD, (
        f"HARNESS: {option} changes the training output by only {effect:.3e} (< {GUARD}); "
        f"the defect test would not discriminate"
    )

    output = _inference_forward(layer, hidden)
    assert output.shape == reference.shape, (output.shape, reference.shape)
    err = _rel_err(output, reference)
    err_vs_plain = _rel_err(output, plain_reference)
    assert err <= TOL, (
        f"DEFECT d2: inference_grouped_gemm_backend={backend!r} ignores "
        f"{option}={option_overrides[option_name]!r}: inference vs training normwise relative "
        f"error {err:.3e} (tolerance {TOL}; the option itself moves the training output by "
        f"{effect:.3e}). Inference vs plain-SwiGLU training: {err_vs_plain:.3e}, i.e. inference "
        f"computed plain SwiGLU. InferenceGroupedMLP passes only activation_type and the tanh "
        f"clamp to the kernels (experts.py:1428-1436, :1712-1748)."
    )
