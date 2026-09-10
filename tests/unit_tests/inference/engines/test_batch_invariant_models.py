# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Recurrent-state and real-inner-MTP complements to the dense BI campaign."""

from functools import partial

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, InferenceCudaGraphScope
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.engines import batch_invariant_test_utils as fixture
from tests.unit_tests.models.test_gpt_model_batch_invariant import DummyTokenizer


def _build_model_engine(case, backend, version, *, mamba, engines, patch):
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(321, inference_rng_tracker=True, force_reset_rng=True)
    depth = case.context.get("num_speculative_tokens", 0)
    graph = case.context.get("num_cuda_graphs") is not None
    cfg = TransformerConfig(
        num_layers=3 if mamba else 2,
        hidden_size=256 if mamba else 128,
        num_attention_heads=8 if mamba else 4,
        mamba_num_heads=16,
        mamba_head_dim=32,
        mamba_num_groups=8,
        mamba_state_dim=16,
        is_hybrid_model=mamba,
        mtp_num_layers=depth or None,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.flash,
        flash_attention_version=version,
        batch_invariant_mode=True,
        batch_invariant_backend=backend,
        transformer_impl="transformer_engine",
        tensor_model_parallel_size=case.tp,
        pipeline_model_parallel_size=case.pp,
        sequence_parallel=case.sp,
        inference_rng_tracker=True,
        inference_sampling_seed=333,
        nccl_all_reduce_for_prefill=False,
        cuda_graph_impl="local" if graph else "none",
        inference_cuda_graph_scope=(
            InferenceCudaGraphScope.block if graph else InferenceCudaGraphScope.none
        ),
    )
    options = dict(
        config=cfg,
        vocab_size=fixture.VOCAB,
        max_sequence_length=1024,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )
    if mamba:
        model = HybridModel(
            **options, hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="M*-"
        )
    else:
        spec = get_gpt_layer_with_transformer_engine_spec()
        model = GPTModel(
            **options,
            transformer_layer_spec=spec,
            mtp_block_spec=get_gpt_mtp_block_spec(cfg, spec, use_transformer_engine=True),
        )
    model = model.cuda().eval()
    state_config = MambaInferenceStateConfig.from_model(model)
    context_options = dict(
        max_sequence_length=1024,
        max_requests=68,
        max_tokens=512,
        block_size_tokens=256,
        buffer_size_gb=0.25,
        mamba_inference_state_config=state_config,
        mamba_memory_ratio=0.5 if mamba else None,
        materialize_only_last_token_logits=False,
        use_cuda_graphs_for_non_decode_steps=False,
        async_sched_mode="legacy",
    )
    context_options.update(case.context)
    ctx = DynamicInferenceContext(cfg, InferenceConfig(**context_options))
    controller = TextGenerationController(
        GPTInferenceWrapper(Float16Module(cfg, model).eval(), ctx), DummyTokenizer(fixture.VOCAB)
    )
    evidence = dict(states=[], inner_calls={}, captured_inner_calls={}, replay=0)

    def target_index():
        selected = (
            (ctx.request_ids[: ctx.total_request_count] == fixture.TARGET).nonzero().flatten()
        )
        return int(selected[0]) if selected.numel() else None

    if depth:
        for d, layer in enumerate(model.mtp.layers):

            def inner_hook(module, args, kwargs, *, d=d):
                key = "inner_calls" if target_index() is not None else "captured_inner_calls"
                evidence[key][d] = evidence[key].get(d, 0) + 1

            layer.mtp_model_layer.register_forward_pre_hook(inner_hook, with_kwargs=True)

    engine = DynamicInferenceEngine(controller, ctx)
    engine._bi_model_evidence = evidence
    engines.append(engine)
    original = controller._dynamic_step_forward_logits

    def forward(inputs, positions):
        idx = target_index()
        # Read identity/offset before a following async bookkeeping phase can
        # recycle the request's recurrent slot for a neighboring request.
        end = (
            int(ctx.request_kv_length_offsets[idx] + ctx.request_query_lengths[idx])
            if idx is not None
            else None
        )
        result = original(inputs, positions)
        if mamba and idx is not None:
            slot = int(ctx.mamba_metadata.request_to_mamba_state_idx[idx])
            counts = []
            for mixer in model.modules():
                if isinstance(mixer, MambaMixer):
                    decoder = mixer._batch_invariant_decoder
                    count = int(decoder.buffers.num_buffered[slot])
                    assert count == end % mixer.chunk_size, (count, end, mixer.chunk_size)
                    counts.append(count)
            assert counts, "target never reached a buffered Mamba decoder"
            evidence["states"].append(
                (
                    end,
                    ctx.is_decode_only(),
                    counts,
                    ctx.mamba_ssm_states[:, slot].detach().clone().cpu(),
                )
            )
        return result

    patch.setattr(controller, "_dynamic_step_forward_logits", forward)
    if depth and graph:
        manager = model._mtp_cudagraph_manager
        for runner in manager.cudagraph_runners:
            original_replay = runner.fwd_graph.replay

            def replay(*, original_replay=original_replay):
                result = original_replay()
                if target_index() is not None:
                    evidence["replay"] += 1
                return result

            patch.setattr(runner.fwd_graph, "replay", replay)
    return engine


MAMBA_CASES = [
    fixture.Case("mamba-partial", prompt_length=129),
    fixture.Case(
        "mamba-chunked",
        prompt_length=401,
        context={"enable_chunked_prefill": True, "max_tokens": 256},
    ),
    fixture.Case("mamba-graph", prompt_length=129, context={"num_cuda_graphs": 4}),
    fixture.Case("mamba-tp2", prompt_length=129, tp=2),
]


@pytest.mark.parametrize("case", MAMBA_CASES, ids=lambda case: case.name)
def test_mamba_dynamic_batch_invariance(case):
    engines = []
    with (
        fixture.invariant_runtime(case) as (backend, version),
        pytest.MonkeyPatch.context() as patch,
    ):
        patch.setattr(
            fixture,
            "build_engine",
            partial(_build_model_engine, mamba=True, engines=engines, patch=patch),
        )
        reference = fixture.run_order(case, backend, version, "solo")
        actual = fixture.run_order(case, backend, version, "back")
        fixture.assert_same_target(reference, actual)
        expected = {end: state for end, _, _, state in engines[0]._bi_model_evidence["states"]}
        compared = 0
        for end, decode, _, state in engines[1]._bi_model_evidence["states"]:
            if end in expected:
                assert torch.equal(state, expected[end]), ("Mamba target state changed", end)
                compared += int(decode)
        assert compared > 0, "no target decode-state comparison"
        assert any(
            s["decode"] and s["requests"] == 65 and s["physical"] == 128 for s in actual[1].steps
        )
        print("BI_MAMBA_WITNESS", case.name, backend, version, "decode_state_comparisons", compared)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("graph_mode", ["eager", "decode", "mixed"])
def test_mtp_dynamic_batch_invariance(depth, graph_mode):
    context = dict(num_speculative_tokens=depth)
    if graph_mode != "eager":
        context.update(num_cuda_graphs=4, max_tokens=256)
    if graph_mode == "mixed":
        context.update(
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=True,
            cuda_graph_mixed_prefill_count=2,
            cuda_graph_all_prefills=True,
        )
    case = fixture.Case("mtp", context=context, prompt_length=401 if graph_mode == "mixed" else 17)
    engines = []
    with (
        fixture.invariant_runtime(case) as (backend, version),
        pytest.MonkeyPatch.context() as patch,
    ):
        patch.setattr(
            fixture,
            "build_engine",
            partial(_build_model_engine, mamba=False, engines=engines, patch=patch),
        )
        reference = fixture.run_order(case, backend, version, "solo")
        actual = fixture.run_order(case, backend, version, "back")
        fixture.assert_same_target(reference, actual)
        evidence = engines[1]._bi_model_evidence
        if graph_mode == "eager":
            assert set(evidence["inner_calls"]) == set(
                range(depth)
            ), "a real inner MTP depth never executed with the target"
        else:
            assert set(evidence["captured_inner_calls"]) == set(range(depth))
            assert evidence["replay"] > 0, "target did not replay the actual MTP graph"
        expected_width = ((65 * (depth + 1) + 63) // 64) * 64
        assert any(
            s["decode"] and s["requests"] == 65 and s["physical"] == expected_width
            for s in actual[1].steps
        )
        print(
            "BI_MTP_WITNESS",
            depth,
            graph_mode,
            backend,
            version,
            "physical",
            expected_width,
            "replay",
            evidence["replay"],
        )
