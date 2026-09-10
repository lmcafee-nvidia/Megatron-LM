# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real-engine batch-invariance fixtures; backend groups run in fresh interpreters."""

import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import AsyncScheduleMode, InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import attention
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.enums import AttnBackend, InferenceCudaGraphScope
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.models.test_gpt_model_batch_invariant import DummyTokenizer
from tests.unit_tests.test_utilities import Utils

TARGET = 101
VOCAB = 128


@dataclass(frozen=True)
class Case:
    """One supported feature interaction, not an environment fallback recipe."""

    name: str
    model: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    prompt_length: int = 17
    warm_prefix: bool = False
    tp: int = 1
    pp: int = 1
    sp: bool = False


@contextmanager
def invariant_runtime(case):
    """Pin process-global mode; launch each selected backend in a fresh process."""
    backend = os.environ.get("MCORE_BI_TEST_BACKEND", "triton")
    fa_version = int(os.environ.get("MCORE_BI_TEST_FA_VERSION", "3"))
    assert backend in ("triton", "te_native", "deepgemm")
    assert not bik.is_batch_invariant_mode_enabled(), "inherited a live backend from another test"
    if backend == "te_native":
        assert (
            os.environ.get("CUBLASLT_WORKSPACE_SIZE") == "0"
        ), "Set CUBLASLT_WORKSPACE_SIZE=0 before starting the distributed interpreter"
    assert fa_version in (3, 4)
    assert attention.HAVE_FA3 if fa_version == 3 else attention.HAVE_FA4
    assert bik.te_supports_batch_invariant_attention()
    Utils.initialize_model_parallel(case.tp, case.pp)
    # Another test module can change these class attributes during collection.
    old_rounders = DynamicInferenceContext.TOKEN_ROUNDER, DynamicInferenceContext.REQUEST_ROUNDER
    DynamicInferenceContext.TOKEN_ROUNDER = 64
    DynamicInferenceContext.REQUEST_ROUNDER = 4
    with pytest.MonkeyPatch.context() as patch:
        for name, value in {
            "NVTE_FUSED_ATTN": "0",
            "NVTE_FLASH_ATTN": "1",
            "NVTE_UNFUSED_ATTN": "0",
        }.items():
            patch.setenv(name, value)
        bik.enable_batch_invariant_mode(backend=backend, collective="ordered")
        try:
            yield backend, fa_version
        finally:
            InferenceMode.unset_active()
            bik.disable_batch_invariant_mode()
            DynamicInferenceContext.TOKEN_ROUNDER, DynamicInferenceContext.REQUEST_ROUNDER = (
                old_rounders
            )
            gc.collect()
            torch.cuda.empty_cache()
            Utils.destroy_model_parallel()


def build_engine(case, backend, fa_version):
    """Build the same tiny, real model for every ordering of a case."""
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(
        321, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
    )
    graphed = case.context.get("num_cuda_graphs") is not None
    model_options = dict(
        num_layers=2 if case.pp == 1 else 4,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        batch_invariant_backend=backend,
        flash_attention_version=fa_version,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_backend=AttnBackend.flash,
        transformer_impl="transformer_engine",
        nccl_all_reduce_for_prefill=False,
        inference_rng_tracker=True,
        inference_sampling_seed=333,
        tensor_model_parallel_size=case.tp,
        pipeline_model_parallel_size=case.pp,
        sequence_parallel=case.sp,
        pipeline_dtype=torch.bfloat16,
        cuda_graph_impl="local" if graphed else "none",
        inference_cuda_graph_scope=(
            InferenceCudaGraphScope.block if graphed else InferenceCudaGraphScope.none
        ),
    )
    model_options.update(case.model)
    cfg = TransformerConfig(**model_options)
    factories = {
        "transformer_engine": get_gpt_layer_with_transformer_engine_spec,
        "local": get_gpt_layer_local_spec,
        "inference_optimized": get_gpt_layer_with_inference_spec,
    }
    spec_options = {"normalization": cfg.normalization} if cfg.transformer_impl == "local" else {}
    model = (
        GPTModel(
            config=cfg,
            transformer_layer_spec=factories[cfg.transformer_impl](**spec_options),
            vocab_size=VOCAB,
            max_sequence_length=1024,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            position_embedding_type="rope",
        )
        .cuda()
        .eval()
    )
    model = Float16Module(cfg, model).eval()
    options = dict(
        max_sequence_length=1024,
        buffer_size_gb=0.125,
        block_size_tokens=256,
        max_requests=68,
        max_tokens=512,
        num_cuda_graphs=None,
        materialize_only_last_token_logits=False,
        use_cuda_graphs_for_non_decode_steps=False,
        unified_memory_level=0,
        async_sched_mode=AsyncScheduleMode.LEGACY,
    )
    options.update(case.context)
    ctx = DynamicInferenceContext(cfg, InferenceConfig(**options))
    tokenizer = DummyTokenizer(VOCAB, bos=1, eod=VOCAB - 1)
    wrapper = GPTInferenceWrapper(model, ctx)
    return DynamicInferenceEngine(TextGenerationController(wrapper, tokenizer), ctx)


class ForwardWitness:
    """Capture target-owned tensors before the next step can compact or reuse them."""

    def __init__(self, engine, patch):
        self.engine = engine
        self.steps = []
        self.current = None
        self.sample_steps = []
        controller, ctx = engine.controller, engine.context
        original = controller._dynamic_step_forward_logits

        def forward(input_ids, position_ids):
            n = ctx.active_token_count
            req_idxs = ctx.token_to_request_idx[:n].long()
            ids = ctx.request_ids[req_idxs]
            rows = (ids == TARGET).nonzero().flatten()
            self.current = None
            if rows.numel():
                request_idx = int(req_idxs[rows[0]])
                self.current = dict(
                    positions=ctx.token_to_pos_ids[rows].clone(),
                    tokens=ctx.token_to_input_ids[rows].clone(),
                    logical=n,
                    physical=int(input_ids.shape[1]),
                    decode=ctx.is_decode_only(),
                    target_query=int(ctx.request_query_lengths[request_idx]),
                    target_offset=int(ctx.request_kv_length_offsets[request_idx]),
                    requests=ctx.total_request_count - ctx.paused_request_count,
                    graph=ctx.using_cuda_graph_this_step(),
                    replay=0,
                    attention=[],
                    gemms=[],
                    target_row=int(rows[0]),
                )
            try:
                result = original(input_ids, position_ids)
                if self.current is not None:
                    if ctx.config.materialize_only_last_token_logits:
                        mapped = ctx.active_logit_idxs[: ctx.num_last_token_logits].cpu().long()
                        selected = torch.isin(mapped, rows)
                        rows = mapped[selected]
                        logits = controller._all_logits_cuda[0, : ctx.num_last_token_logits][
                            selected.to("cuda")
                        ]
                        self.current["positions"] = ctx.token_to_pos_ids[rows].clone()
                        self.current["tokens"] = ctx.token_to_input_ids[rows].clone()
                    else:
                        logits = controller._all_logits_cuda[0, rows.to("cuda")]
                    self.current["logits"] = logits.detach().clone().cpu()
                    self.steps.append(self.current)
                return result
            finally:
                self.current = None

        patch.setattr(controller, "_dynamic_step_forward_logits", forward)

        def observe(owner, name, kind):
            fn = getattr(owner, name, None)
            if fn is None:
                return

            @wraps(fn)
            def call(*args, **kwargs):
                result = fn(*args, **kwargs)
                if self.current is not None:
                    if kind == "attention":
                        self.current[kind].append((name, kwargs.get("num_splits")))
                    else:
                        tensors = [x for x in args if isinstance(x, torch.Tensor)]
                        self.current[kind].append(
                            (name, tuple(tensors[0].shape) if tensors else ())
                        )
                return result

            patch.setattr(owner, name, call)

        for name in ("_flash_attn_forward", "flash_attn3_with_kvcache", "flash_attn4_varlen_func"):
            observe(attention, name, "attention")
        for name in ("matmul_persistent", "_mm_deepgemm"):
            observe(bik, name, "gemms")
        # Native TE modules bind their GEMM entrypoint locally, rather than looking
        # it up through the shared cpp_extensions package at each invocation.
        if bik.get_batch_invariant_backend() == "te_native":
            from transformer_engine.pytorch.module import layernorm_linear, linear

            observe(linear, "general_gemm", "gemms")
            observe(layernorm_linear, "general_gemm", "gemms")
        replay = torch.cuda.CUDAGraph.replay

        def graph_replay(graph):
            result = replay(graph)
            if self.current is not None:
                self.current["replay"] += 1
            return result

        patch.setattr(torch.cuda.CUDAGraph, "replay", graph_replay)
        sampler = controller._sampling
        sample_kernel = sampler.sample_kernel

        @wraps(sample_kernel)
        def sample(logits, n, context, **kwargs):
            result = sample_kernel(logits, n, context, **kwargs)
            ids = context.request_ids[context.paused_request_count : context.total_request_count]
            target_rows = (ids == TARGET).nonzero().flatten()
            if target_rows.numel() and kwargs.get("token_to_request_index") is None:
                target_row = int(target_rows[0])
                indices = kwargs.get("gather_indices")
                selected = logits[:n] if indices is None else logits[indices[:n].long()]
                processed = sampler.log_probs_kernel(selected, context)
                token = int(result[target_row])
                self.sample_steps.append(
                    dict(
                        logits=selected[target_row].detach().clone().cpu(),
                        log_probs=processed[target_row].detach().clone().cpu(),
                        token=token,
                        sampler=type(sampler).__name__,
                        requests=n,
                        filters=context.active_sampling_filter_flags(n),
                    )
                )
            return result

        patch.setattr(sampler, "sample_kernel", sample)

    def assert_active(self, fa_version):
        """Reject configured-only backends and observations from other requests."""
        assert self.steps, "target never executed a forward"
        calls = [call for step in self.steps for call in step["attention"]]
        assert calls, "no real target-containing attention call"
        expected = (
            {"_flash_attn_forward", "flash_attn3_with_kvcache"}
            if fa_version == 3
            else {"flash_attn4_varlen_func"}
        )
        assert all(name in expected and splits == 1 for name, splits in calls), calls
        assert any(step["gemms"] for step in self.steps), "no real target-containing GEMM call"
        assert all(step["physical"] % 64 == 0 for step in self.steps)


def target_prompt(length):
    return [(7 + i * 13) % (VOCAB - 2) + 1 for i in range(length)]


def run_order(case, backend, fa_version, order, *, sampling=None):
    """Run identical target histories alone, moved in a batch, or amid arrivals."""
    engine = build_engine(case, backend, fa_version)
    params = dict(
        num_tokens_to_generate=6,
        top_k=1,
        termination_id=-1,
        return_log_probs=True,
        return_prompt_tokens=True,
    )
    params.update(case.sampling)
    params.update(sampling or {})
    target_params = SamplingParams(**params)
    target = target_prompt(case.prompt_length)
    finished = {}

    def step():
        result = engine.step_modern()
        for record in result["finished_request_records"]:
            req = record.merge(engine.controller.tokenizer)
            finished[req.request_id] = req

    def drain():
        for _ in range(256):
            if not engine.has_unfinished_requests():
                break
            step()
        assert not engine.has_unfinished_requests(), "bounded engine drain exhausted"

    if case.warm_prefix:
        engine.add_request(
            99, target, SamplingParams(num_tokens_to_generate=1, top_k=1, termination_id=-1)
        )
        drain()
        assert engine.context.kv_block_allocator.enable_prefix_caching
    with pytest.MonkeyPatch.context() as patch:
        witness = ForwardWitness(engine, patch)
        neighbors = list(range(201, 265))

        def add_neighbors():
            for rid in neighbors:
                # Real heterogeneous dispatch: stochastic target rows share their
                # launch with both top-k and top-p requests, not neutral padding.
                filters = dict(top_k=1)
                if target_params.top_k != 1:
                    filters = dict(top_k=7) if rid % 2 else dict(top_p=0.8)
                neighbor = SamplingParams(num_tokens_to_generate=6, termination_id=-1, **filters)
                engine.add_request(rid, [rid % (VOCAB - 1)], neighbor)

        if order == "back":
            add_neighbors()
        engine.add_request(TARGET, target, target_params)
        if order == "staggered":
            step()
            add_neighbors()
        elif order == "front":
            add_neighbors()
        else:
            assert order in ("solo", "back")
        drain()
        witness.assert_active(fa_version)
        assert TARGET in finished
        # Observe retirement before cleanup can make a leaking run appear green.
        assert engine.context.total_request_count == 0
        assert engine.context.kv_block_allocator.get_active_used() == 0
        if not case.warm_prefix:
            assert engine.context.kv_block_allocator.get_total_used() == 0
        if case.context.get("num_cuda_graphs") is not None:
            assert any(
                s["graph"] and s["replay"] for s in witness.steps
            ), "target never replayed a graph"
        if case.context.get("enable_chunked_prefill"):
            chunks = [s for s in witness.steps if int(s["positions"][0]) < case.prompt_length - 1]
            assert len(chunks) >= 2, "target did not execute multiple prefill chunks"
        if case.warm_prefix:
            assert (
                min(int(s["positions"].min()) for s in witness.steps) >= 256
            ), "target did not reuse its prefix"
    return finished[TARGET], witness


def assert_same_target(reference, actual, *, require_trajectory=True):
    """Check every shared target position, never unrelated padded rows."""
    ref_req, ref_witness = reference
    req, witness = actual
    if require_trajectory:
        for field_name in (
            "generated_tokens",
            "prompt_log_probs",
            "generated_log_probs",
            "prompt_top_n_logprobs",
            "generated_top_n_logprobs",
        ):
            assert getattr(req, field_name, None) == getattr(ref_req, field_name, None), field_name
    reference_rows = {}
    for step in ref_witness.steps:
        for pos, token, logits in zip(step["positions"], step["tokens"], step["logits"]):
            key = int(pos), int(token)
            if key in reference_rows:
                assert torch.equal(reference_rows[key], logits), (
                    "reference repeated position",
                    key,
                )
            reference_rows[key] = logits
    comparisons = 0
    for step in witness.steps:
        for pos, token, logits in zip(step["positions"], step["tokens"], step["logits"]):
            key = int(pos), int(token)
            assert key in reference_rows, ("target history changed", key)
            assert torch.equal(reference_rows[key], logits), (
                "target logits changed",
                key,
                step["physical"],
            )
            comparisons += 1
    assert comparisons > 0
