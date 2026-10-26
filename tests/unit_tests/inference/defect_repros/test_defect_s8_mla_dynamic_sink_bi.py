# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for suspected bug s8: dynamic-batching MLA drops the sink offset and is not
batch-invariant in FlashMLA decode.

Reachability (checked at ed9c2c8c1): no validation rejects either combination.
``MLATransformerConfig(softmax_type="off-by-one"|"learnable", cache_mla_latents=True)`` and
``MLATransformerConfig(batch_invariant_mode=True, attention_backend=flash,
flash_attention_version=3, cache_mla_latents=True)`` both construct (``transformer_config.py``
``__post_init__`` at ``:4421-4530`` and ``:4698-...`` has no MLA check; nothing in
``megatron/core/inference`` or ``megatron/training`` does either). No in-tree recipe combines them.

1. Sink: ``MultiLatentAttention.forward``'s dynamic branch (``multi_latent_attention.py:587-607``)
   calls ``self.flash_decode_and_prefill(...)`` without ``softmax_offset``, so ``need_lse`` is
   False (``attention.py:1130``) and no sink correction is applied on the FA prefill path
   (``:1208-1213``) or the FlashMLA decode path (``:1275-1279``). Standard attention passes
   ``softmax_offset=self._get_inference_softmax_offset()`` (``attention.py:1748``). The static path
   (TE core attention built with ``softmax_type``) applies the sink, so static and dynamic MLA
   disagree.
2. Batch invariance: in decode-only steps the FlashMLA branch (``attention.py:1236-1274``) asks
   ``get_mla_metadata`` for a split-KV schedule computed from every request's cache length and
   hands it to ``flash_mla_with_kvcache``. Every other decode/prefill branch pins
   ``num_splits=1`` under ``batch_invariant_mode`` (``:1153, 1300, 1323``); this one has no guard.

Harness (sink): the module comes from ``hybrid_inference_stack_spec.mla_layer`` and the context is
a real ``DynamicInferenceContext`` with one 16-token prefill request; only
``flash_decode_and_prefill`` is replaced by a shape-correct stub whose call arguments are
inspected (pattern from ``test_defect_d5_dsa_dynamic_bypass.py``). No FlashAttention or FlashMLA.

Harness (batch invariance): ``Attention.flash_decode_and_prefill`` is called directly in
decode-only mode on the same spec's MLA module built with a batch-invariant config.
``_resolve_flash_version`` is stubbed to "FA3" because the pinned version is asserted installed
before the branch, and the MLA branch never runs FlashAttention. The structural test replaces
``get_mla_metadata`` / ``flash_mla_with_kvcache`` with recorders; the runtime tests use the real
FlashMLA and SKIP without it.

Needs one GPU and Transformer Engine; the two runtime tests also need ``flash_mla``.
"""

import math
from unittest import mock

import pytest
import torch

import megatron.core.transformer.attention as attention_module
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
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
    pytest.mark.skipif(not torch.cuda.is_available(), reason="s8 repro needs one CUDA GPU"),
    pytest.mark.skipif(not HAVE_TE, reason="hybrid_inference_stack_spec needs Transformer Engine"),
]

needs_flash_mla = pytest.mark.skipif(
    not attention_module.HAVE_FMLA, reason="needs FlashMLA (flash_mla) for the real decode kernel"
)

PROMPT_LEN = 16
# FlashMLA dense decode shapes (DeepSeek-V3 per-TP-rank): latent 512 + rope 64, 16 query heads.
FMLA_KV_LORA = 512
FMLA_ROPE = 64
FMLA_HEADS = 16
FMLA_BLOCK = 64


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _sink_config(softmax_type: str) -> MLATransformerConfig:
    """Small MLA proxy from test_defect_d5_dsa_dynamic_bypass.py, with a sink softmax."""
    return MLATransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=16,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        qk_layernorm=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        transformer_impl="inference_optimized",
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,  # prepare_for_absorption builds TE modules mid-forward
        hidden_dropout=0.0,
        attention_dropout=0.0,
        cache_mla_latents=True,
        softmax_type=softmax_type,
    )


def _fmla_config(batch_invariant: bool) -> MLATransformerConfig:
    """MLA config with FlashMLA-compatible latent sizes; optionally batch-invariant."""
    kwargs = dict(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=FMLA_HEADS,
        q_lora_rank=256,
        kv_lora_rank=FMLA_KV_LORA,
        qk_head_dim=128,
        qk_pos_emb_head_dim=FMLA_ROPE,
        v_head_dim=128,
        qk_layernorm=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        transformer_impl="inference_optimized",
        rope_type="rope",
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        cache_mla_latents=True,
    )
    if batch_invariant:
        # The minimum batch_invariant_mode validation asks for (transformer_config.py:4421-4467).
        kwargs.update(
            batch_invariant_mode=True,
            attention_backend=AttnBackend.flash,
            flash_attention_version=3,
        )
    return MLATransformerConfig(**kwargs)


def _build_attention(config: MLATransformerConfig) -> MLASelfAttention:
    """Self-attention of hybrid_inference_stack_spec.mla_layer, as HybridStack builds it."""
    spec = hybrid_inference_stack_spec.submodules.mla_layer.submodules.self_attention
    attn = build_module(spec, config=config, layer_number=1, pp_layer_offset=0)
    assert isinstance(attn, MLASelfAttention)
    return attn.cuda().eval()


# ---------------------------------------------------------------------------------------------
# Sink harness
# ---------------------------------------------------------------------------------------------


def _dynamic_prefill_context(model_config: MLATransformerConfig) -> DynamicInferenceContext:
    """A real dynamic context holding one PROMPT_LEN-token prefill request (d5 pattern)."""
    ctx = DynamicInferenceContext(
        model_config=model_config,
        inference_config=InferenceConfig(
            max_sequence_length=256,
            buffer_size_gb=0.01,
            paused_buffer_size_gb=0.002,
            block_size_tokens=64,  # required with cache_mla_latents
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
    """Shape-correct stand-in for flash_decode_and_prefill (zeros)."""

    def stub(q, *args, **kwargs):
        decode = kwargs.get("is_decode_only", args[8] if len(args) > 8 else False)
        d_out = config.kv_lora_rank if decode else config.v_head_dim
        return torch.zeros(*q.shape[:3], d_out, dtype=q.dtype, device=q.device)

    return stub


def _run_dynamic_forward(attn, ctx, config):
    hidden = torch.randn(
        ctx.padded_active_token_count, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    with mock.patch.object(
        attn, "flash_decode_and_prefill", side_effect=_kernel_stub(config)
    ) as kernel:
        with torch.no_grad(), InferenceMode.active():
            out, _ = attn(hidden, attention_mask=None, inference_context=ctx)
    return out, hidden, kernel


def _passed_softmax_offset(kernel):
    """softmax_offset as flash_decode_and_prefill received it (kwarg or 11th positional)."""
    args, kwargs = kernel.call_args
    if "softmax_offset" in kwargs:
        return kwargs["softmax_offset"]
    return args[10] if len(args) > 10 else None


def _sink_attention(softmax_type):
    config = _sink_config(softmax_type)
    attn = _build_attention(config)
    if softmax_type == "learnable":
        offset = getattr(attn.core_attention, "softmax_offset", None)
        if isinstance(offset, torch.Tensor):
            with torch.no_grad():
                offset.copy_(torch.linspace(-1.0, 1.0, offset.numel(), device=offset.device))
    return config, attn


# ---------------------------------------------------------------------------------------------
# FlashMLA decode harness
# ---------------------------------------------------------------------------------------------


def _decode_batch(lengths, seed_req0=7, seed_rest=11):
    """Decode-only inputs for flash_decode_and_prefill. Request 0's query and latent cache depend
    only on seed_req0 and its length, so it is identical in every batch it appears in."""
    B = len(lengths)
    dim = FMLA_KV_LORA + FMLA_ROPE
    nb = math.ceil(max(lengths) / FMLA_BLOCK)
    g0 = torch.Generator(device="cuda").manual_seed(seed_req0)
    q0 = torch.randn(1, 1, FMLA_HEADS, dim, generator=g0, device="cuda").to(torch.bfloat16)
    kv0 = torch.randn(nb, FMLA_BLOCK, dim, generator=g0, device="cuda").to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(seed_rest)
    q = torch.randn(B, 1, FMLA_HEADS, dim, generator=g, device="cuda").to(torch.bfloat16)
    kv = torch.randn(B * nb, FMLA_BLOCK, dim, generator=g, device="cuda").to(torch.bfloat16)
    q[0:1] = q0
    kv[:nb] = kv0
    seqlens_k = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cu_k = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    cu_k[1:] = torch.cumsum(seqlens_k, 0)
    return dict(
        q=q,  # (B * 1, 1, H, D): one decode token per request
        k=kv,  # paged latent cache (num_blocks, block, D)
        v=None,
        max_seqlen_q=1,
        max_seqlen_k=max(lengths),
        cu_seqlens_q=torch.arange(B + 1, dtype=torch.int32, device="cuda"),
        cu_seqlens_k=cu_k,
        seqlens_k=seqlens_k,
        block_table=torch.arange(B * nb, dtype=torch.int32, device="cuda").view(B, nb),
        is_decode_only=True,
    )


def _decode(attn, inputs):
    # The pinned FlashAttention version is asserted installed before the MLA branch, which never
    # runs FlashAttention; report "FA3" so the FlashMLA branch is reached without it.
    with mock.patch.object(attn, "_resolve_flash_version", return_value=(False, True)):
        with torch.no_grad():
            return attn.flash_decode_and_prefill(**inputs)


class _FakeFlashMLA:
    """Recording stand-ins for get_mla_metadata / flash_mla_with_kvcache."""

    def __init__(self):
        self.metadata_calls = []
        self.kernel_calls = []

    def get_mla_metadata(self, cache_seqlens, num_heads_per_head_k, num_heads_k):
        splits = 2 * torch.arange(cache_seqlens.numel() + 1, dtype=torch.int32, device="cuda")
        meta = torch.zeros(8, 8, dtype=torch.int32, device="cuda")
        self.metadata_calls.append(dict(cache_seqlens=cache_seqlens.clone(), splits=splits))
        return meta, splits

    def flash_mla_with_kvcache(
        self, q, kv_cache, block_table, cache_seqlens, head_dim_v, meta, num_splits, **kwargs
    ):
        self.kernel_calls.append(dict(num_splits=num_splits, kwargs=kwargs))
        B, S, H, _ = q.shape
        out = torch.zeros(B, S, H, head_dim_v, dtype=q.dtype, device=q.device)
        return out, torch.zeros(B, H, S, dtype=torch.float32, device=q.device)


def _patched_fake_flash_mla():
    fake = _FakeFlashMLA()
    return fake, (
        mock.patch.object(attention_module, "get_mla_metadata", fake.get_mla_metadata),
        mock.patch.object(
            attention_module, "flash_mla_with_kvcache", fake.flash_mla_with_kvcache
        ),
    )


def _real_split_recorder():
    """Wrap the real get_mla_metadata, recording the per-request split counts it returns."""
    real = attention_module.get_mla_metadata
    record = []

    def wrapper(*args, **kwargs):
        meta, splits = real(*args, **kwargs)
        if isinstance(splits, torch.Tensor) and splits.dim() == 1:
            record.append((splits[1:] - splits[:-1]).tolist())
        else:
            record.append(repr(splits))
        return meta, splits

    return record, mock.patch.object(attention_module, "get_mla_metadata", wrapper)


# ---------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("softmax_type", ["off-by-one", "learnable"])
def test_control_mla_sink_layer_exposes_offset(softmax_type):
    """CONTROL: an MLA layer built with a sink softmax has a per-head offset that the inference
    helper returns (the value the dynamic path should hand to the kernel)."""
    config, attn = _sink_attention(softmax_type)
    offset = attn._get_inference_softmax_offset()
    assert isinstance(offset, torch.Tensor)
    assert offset.shape == (attn.num_attention_heads_per_partition,)


def test_control_dynamic_sink_harness_reaches_kernel():
    """CONTROL: a dynamic-batching prefill through the sink-configured MLA layer reaches the
    (stubbed) attention kernel once and returns [T, 1, hidden]."""
    config, attn = _sink_attention("learnable")
    ctx = _dynamic_prefill_context(config)
    out, hidden, kernel = _run_dynamic_forward(attn, ctx, config)
    assert kernel.call_count == 1
    assert out.shape == hidden.shape


def test_control_flashmla_decode_branch_reached_with_recorders():
    """CONTROL: without batch-invariant mode, a decode-only call reaches the FlashMLA branch,
    which schedules the whole batch with get_mla_metadata and runs the kernel with that
    schedule. Validates the FlashMLA-free harness."""
    attn = _build_attention(_fmla_config(batch_invariant=False))
    lengths = [300, 4096, 4096, 1000]
    inputs = _decode_batch(lengths)
    fake, patches = _patched_fake_flash_mla()
    with patches[0], patches[1]:
        out = _decode(attn, inputs)
    assert len(fake.metadata_calls) == 1 and len(fake.kernel_calls) == 1
    assert fake.metadata_calls[0]["cache_seqlens"].tolist() == lengths
    assert fake.kernel_calls[0]["num_splits"] is fake.metadata_calls[0]["splits"]
    assert out.shape == (len(lengths), 1, FMLA_HEADS, FMLA_KV_LORA)


@needs_flash_mla
def test_control_flashmla_bi_decode_repeatable():
    """CONTROL (real FlashMLA): the same batch decoded twice in batch-invariant config is
    bitwise identical, so a co-batch difference in the defect test is not run-to-run noise."""
    attn = _build_attention(_fmla_config(batch_invariant=True))
    inputs = _decode_batch([4096] * 32)
    a = _decode(attn, inputs).clone()
    b = _decode(attn, inputs)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------------------------
# Defects
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("softmax_type", ["off-by-one", "learnable"])
def test_defect_dynamic_mla_passes_sink_offset_to_kernel(softmax_type):
    """DEFECT: the dynamic-batching MLA branch must hand the layer's sink offset to
    flash_decode_and_prefill, as standard attention does (attention.py:1748)."""
    config, attn = _sink_attention(softmax_type)
    expected = attn._get_inference_softmax_offset()
    ctx = _dynamic_prefill_context(config)
    _, _, kernel = _run_dynamic_forward(attn, ctx, config)
    assert kernel.call_count == 1, "harness: kernel not reached"
    passed = _passed_softmax_offset(kernel)
    ok = isinstance(passed, torch.Tensor) and torch.equal(passed, expected)
    assert ok, (
        f"DEFECT s8: dynamic-batching MLA with softmax_type={softmax_type!r} called "
        f"flash_decode_and_prefill with softmax_offset={passed!r} instead of the layer's per-head "
        f"sink {expected.float().tolist()}; no sink correction is applied, so dynamic attention "
        "uses exp(qk)/sum(exp(qk)) where the static path uses exp(qk)/(exp(sink)+sum(exp(qk))) "
        "(multi_latent_attention.py:596-607 omits softmax_offset=)"
    )


def test_defect_bi_flashmla_decode_does_not_use_batch_schedule():
    """DEFECT (structural, no FlashMLA needed): in batch-invariant mode the FlashMLA decode
    branch must not run the kernel on the split-KV schedule get_mla_metadata computes from the
    whole batch's cache lengths (it must refuse, or use a batch-independent schedule)."""
    attn = _build_attention(_fmla_config(batch_invariant=True))
    assert attn.batch_invariant_mode
    lengths = [300, 4096, 4096, 1000]
    inputs = _decode_batch(lengths)
    fake, patches = _patched_fake_flash_mla()
    refused = None
    with patches[0], patches[1]:
        try:
            _decode(attn, inputs)
        except (AssertionError, RuntimeError, NotImplementedError, ValueError) as exc:
            refused = exc
    if refused is not None:
        assert "invariant" in str(refused).lower(), f"unexpected error: {refused!r}"
        return
    batch_schedule_used = any(
        call["num_splits"] is meta["splits"] and len(meta["cache_seqlens"]) > 1
        for call in fake.kernel_calls
        for meta in fake.metadata_calls
    )
    assert not batch_schedule_used, (
        "DEFECT s8: with batch_invariant_mode=True the FlashMLA decode branch "
        "(attention.py:1236-1274) ran flash_mla_with_kvcache on the split-KV schedule that "
        f"get_mla_metadata computed from all {len(lengths)} requests' cache lengths {lengths}; "
        "a request's split count, and so its reduction order, then depends on its co-batch. "
        "Every other decode/prefill branch forces num_splits=1 in this mode"
    )


@needs_flash_mla
def test_defect_bi_flashmla_decode_row_independent_of_cobatch():
    """DEFECT (runtime, real FlashMLA): in batch-invariant config a request's decode output must
    be bitwise identical whether it is decoded alone or with 31 long co-batched requests."""
    attn = _build_attention(_fmla_config(batch_invariant=True))
    splits, recorder = _real_split_recorder()
    with recorder:
        alone = _decode(attn, _decode_batch([4096]))[0].clone()
        batched = _decode(attn, _decode_batch([4096] * 32))[0].clone()
    per_request = [s[0] if isinstance(s, list) else s for s in splits]  # request 0's entry
    split_note = f"get_mla_metadata split count for request 0 (alone, co-batched): {per_request}"
    assert torch.equal(alone, batched), (
        "DEFECT s8: FlashMLA decode under batch_invariant_mode is not batch-invariant: request "
        "0's output differs between batch size 1 and 32 (max abs diff "
        f"{(alone.float() - batched.float()).abs().max().item():.3g}); {split_note}. "
        "get_mla_metadata schedules split-KV over the whole batch (attention.py:1255-1263)"
    )
