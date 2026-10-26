# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for defect d5: dynamic-batching MLA never runs a DSA core attention.

``hybrid_inference_stack_spec.dsa_layer`` (``megatron/core/models/hybrid/hybrid_layer_specs.py``,
``dsa_layer``) builds ``MLASelfAttention`` with ``DSAttention`` (indexer + top-k sparse attention)
as its core attention. ``MultiLatentAttention.forward``
(``megatron/core/transformer/multi_latent_attention.py:576-607``) calls ``self._run_core_attention``
only when ``inference_context is None or inference_context.is_static_batching()``. Dynamic batching
requires ``cache_mla_latents`` (asserted at ``:496-499``) and its branch calls
``self.flash_decode_and_prefill`` (dense FlashAttention / FlashMLA) directly, so the DSA indexer and
top-k selection never run: a DSA layer computes dense attention.

Harness: the module comes from the real inference spec; the context is a real
``DynamicInferenceContext`` (one 16-token prefill request, MLA latent cache, block size 64, as
``tests/unit_tests/inference/contexts/test_dynamic_context.py`` builds it). Only the attention
*kernel* ``flash_decode_and_prefill`` is replaced, by a stub that returns correctly shaped zeros, so
no FlashAttention / FlashMLA is needed. ``DSAttention.forward`` and
``DSAIndexer.forward_before_topk`` are observed with ``wraps=`` spies.

Needs one GPU and Transformer Engine. ``dsa_indexer_rotate_activation=False`` avoids the optional
``fast_hadamard_transform`` dependency; ``dsa_kernel_backend`` stays at its default (PyTorch path).
"""

import traceback
from unittest import mock

import pytest
import torch

import megatron.core.transformer.multi_latent_attention as mla_module
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention, DSAIndexer
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

REQUIRED_GPUS = 1

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="d5 repro needs one CUDA GPU"),
    pytest.mark.skipif(not HAVE_TE, reason="hybrid_inference_stack_spec needs Transformer Engine"),
]

PROMPT_LEN = 16  # > DSA_TOPK, the regime where sparse and dense attention differ
DSA_TOPK = 4


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _config(*, dsa: bool, cache_mla_latents: bool) -> MLATransformerConfig:
    """Small DeepSeek-V3.2-shaped proxy (tests/unit_tests/models/test_dsa_gpt_mamba_equivalence.py)."""
    kwargs = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=16,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        qk_layernorm=True,  # DeepSeek-V3/V3.2 normalize the q/kv latents
        normalization="RMSNorm",
        add_bias_linear=False,
        # resolve_hybrid_stack_spec (megatron/training/models/hybrid.py) selects
        # hybrid_inference_stack_spec exactly when transformer_impl is inference_optimized.
        transformer_impl="inference_optimized",
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        # GPU init: prepare_for_absorption builds new TE modules mid-forward on the config's
        # init device, so CPU initialization would leave them on the CPU.
        use_cpu_initialization=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        cache_mla_latents=cache_mla_latents,
    )
    if dsa:
        kwargs.update(
            experimental_attention_variant="dsa",  # what the hybrid 'D' symbol implies
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=DSA_TOPK,
            dsa_indexer_rotate_activation=False,
        )
    return MLATransformerConfig(**kwargs)


def _build_attention(config: MLATransformerConfig, layer: str) -> MLASelfAttention:
    """Self-attention module of hybrid_inference_stack_spec.<layer>, as HybridStack builds it."""
    spec = getattr(hybrid_inference_stack_spec.submodules, layer).submodules.self_attention
    attn = build_module(spec, config=config, layer_number=1, pp_layer_offset=0)
    assert isinstance(attn, MLASelfAttention)
    return attn.cuda().eval()


def _dynamic_prefill_context(model_config: MLATransformerConfig) -> DynamicInferenceContext:
    """A real dynamic context holding one PROMPT_LEN-token prefill request."""
    ctx = DynamicInferenceContext(
        model_config=model_config,
        inference_config=InferenceConfig(
            max_sequence_length=256,
            buffer_size_gb=0.01,
            paused_buffer_size_gb=0.002,
            block_size_tokens=64,  # required with cache_mla_latents (dynamic_context.py:423-426)
            max_tokens=128,
            max_requests=8,
            num_cuda_graphs=None,
            use_flashinfer_fused_rope=False,
            unified_memory_level=0,
        ),
    )
    ctx.add_request(
        DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.arange(1, PROMPT_LEN + 1, device="cpu"),
            sampling_params=SamplingParams(num_tokens_to_generate=4, termination_id=-1),
        )
    )
    ctx.initialize_attention_state()
    assert ctx.is_dynamic_batching() and not ctx.is_decode_only()
    return ctx


def _kernel_stub(config: MLATransformerConfig):
    """Stand-in for Attention.flash_decode_and_prefill: correctly shaped zeros.

    Prefill/mixed returns the dense FA layout [s, b, h, v_head_dim]; decode-only returns FlashMLA's
    latent layout [s, b, h, kv_lora_rank] (multi_latent_attention.py:608-628).
    """

    def stub(q, *args, **kwargs):
        # forward passes is_decode_only as the 10th positional argument (:596-607).
        decode = kwargs.get("is_decode_only", args[-1] if args else False)
        d_out = config.kv_lora_rank if decode else config.v_head_dim
        return torch.zeros(*q.shape[:3], d_out, dtype=q.dtype, device=q.device)

    return stub


def _absorption_split_without_norm(attn: MLASelfAttention):
    """Test-side stand-in for split_te_layernorm_column_parallel_linear.

    prepare_for_absorption (multi_latent_attention.py:1308-1362) assumes linear_kv_up_proj is a
    fused norm+linear and splits it. DSA forbids that fusion (QKNormConfigResolver), so its
    linear_kv_up_proj is a plain linear and its KV norm is already the standalone kv_layernorm;
    there is nothing to split. This returns (existing kv_layernorm, the linear), the split result
    a DSA-aware prepare_for_absorption would produce, so the dispatch in forward can be reached.
    """

    def split(fused_layer, config, init_method=None, tp_group=None):
        assert fused_layer is attn.linear_kv_up_proj
        return attn.kv_layernorm, fused_layer

    return split


def _run_dynamic_forward(attn: MLASelfAttention, ctx: DynamicInferenceContext, config):
    """One dynamic-batching forward with spies on the DSA core / indexer and a stubbed kernel."""
    hidden = torch.randn(
        ctx.padded_active_token_count, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    spies = {}
    with mock.patch.object(
        attn, "flash_decode_and_prefill", side_effect=_kernel_stub(config)
    ) as kernel:
        spies["kernel"] = kernel
        core = attn.core_attention
        with mock.patch.object(core, "forward", wraps=core.forward) as core_spy:
            spies["core"] = core_spy
            indexer = getattr(core, "indexer", None)
            if indexer is not None:
                with mock.patch.object(
                    indexer, "forward_before_topk", wraps=indexer.forward_before_topk
                ) as idx_spy:
                    spies["indexer"] = idx_spy
                    with torch.no_grad(), InferenceMode.active():
                        out, _ = attn(hidden, attention_mask=None, inference_context=ctx)
            else:
                with torch.no_grad(), InferenceMode.active():
                    out, _ = attn(hidden, attention_mask=None, inference_context=ctx)
    return out, hidden, spies


# ---------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------


def test_control_training_path_invokes_dsa_core_and_indexer():
    """CONTROL: with no inference context (training / non-dynamic path) the DSA layer of the
    inference spec runs its DSAttention core and the core runs its indexer."""
    config = _config(dsa=True, cache_mla_latents=False)
    attn = _build_attention(config, "dsa_layer")
    core = attn.core_attention
    assert isinstance(core, DSAttention) and isinstance(core.indexer, DSAIndexer)

    hidden = torch.randn(PROMPT_LEN, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with (
        mock.patch.object(core, "forward", wraps=core.forward) as core_spy,
        mock.patch.object(
            core.indexer, "forward_before_topk", wraps=core.indexer.forward_before_topk
        ) as idx_spy,
        torch.no_grad(),
    ):
        out, _ = attn(hidden, attention_mask=None)

    assert core_spy.call_count == 1
    assert idx_spy.call_count == 1
    assert out.shape == (PROMPT_LEN, 1, config.hidden_size)
    assert torch.isfinite(out).all()


def test_control_dynamic_mla_requires_latent_cache():
    """CONTROL: dynamic batching without cache_mla_latents is refused, so the latent-cache branch
    (the one under test) is the only dynamic MLA branch."""
    ctx = _dynamic_prefill_context(_config(dsa=True, cache_mla_latents=True))
    config = _config(dsa=True, cache_mla_latents=False)
    attn = _build_attention(config, "dsa_layer")
    hidden = torch.randn(
        ctx.padded_active_token_count, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    with pytest.raises(AssertionError, match="cache mla latents must be true"):
        with torch.no_grad(), InferenceMode.active():
            attn(hidden, attention_mask=None, inference_context=ctx)


def test_control_dynamic_harness_reaches_attention_kernel():
    """CONTROL: the same dynamic harness drives a plain MLA layer of the inference spec through
    latent caching, cache uncompression and kernel dispatch to an output of the right shape."""
    config = _config(dsa=False, cache_mla_latents=True)
    attn = _build_attention(config, "mla_layer")
    ctx = _dynamic_prefill_context(config)
    out, hidden, spies = _run_dynamic_forward(attn, ctx, config)
    assert spies["kernel"].call_count == 1
    q = spies["kernel"].call_args.args[0]
    assert q.shape[0] == hidden.shape[0]
    assert out.shape == hidden.shape


# ---------------------------------------------------------------------------------------------
# Defects
# ---------------------------------------------------------------------------------------------


def _assert_dsa_ran(spies):
    kernel_calls = spies["kernel"].call_count
    assert spies["core"].call_count >= 1 and spies["indexer"].call_count >= 1, (
        "DEFECT d5: dynamic-batching MLA forward of a DSA layer never invoked the DSAttention "
        f"core (calls={spies['core'].call_count}) or its indexer "
        f"(calls={spies['indexer'].call_count}); the cache_mla_latents branch called "
        f"flash_decode_and_prefill {kernel_calls}x instead, i.e. dense attention over all "
        f"{PROMPT_LEN} keys rather than the top-{DSA_TOPK} sparse selection"
    )


def test_defect_dynamic_dsa_forward_invokes_dsa_core():
    """DEFECT (end to end): a dynamic-batching prefill through the inference spec's DSA layer must
    run the DSAttention core and its indexer."""
    config = _config(dsa=True, cache_mla_latents=True)
    attn = _build_attention(config, "dsa_layer")
    ctx = _dynamic_prefill_context(config)
    try:
        _, _, spies = _run_dynamic_forward(attn, ctx, config)
    except Exception as exc:  # noqa: BLE001 - report where the dynamic DSA path dies
        frames = traceback.extract_tb(exc.__traceback__)
        megatron_frames = [f for f in frames if "/megatron/core/" in f.filename]
        frame = (megatron_frames or frames)[-1]  # deepest Megatron frame, not torch internals
        pytest.fail(
            "DEFECT d5: dynamic-batching forward of the inference spec's DSA layer raised "
            f"{type(exc).__name__}: {exc} (at {frame.filename}:{frame.lineno} in {frame.name}) "
            "before any attention ran; the DSAttention core and its indexer were never invoked"
        )
    _assert_dsa_ran(spies)


def test_defect_dynamic_dispatch_bypasses_dsa_core():
    """DEFECT (dispatch only): with absorption prepared the way a DSA layer needs, the
    dynamic-batching branch of MultiLatentAttention.forward must still route to the DSA core."""
    config = _config(dsa=True, cache_mla_latents=True)
    attn = _build_attention(config, "dsa_layer")
    ctx = _dynamic_prefill_context(config)
    with mock.patch.object(
        mla_module,
        "split_te_layernorm_column_parallel_linear",
        _absorption_split_without_norm(attn),
    ):
        _, _, spies = _run_dynamic_forward(attn, ctx, config)
    _assert_dsa_ran(spies)
