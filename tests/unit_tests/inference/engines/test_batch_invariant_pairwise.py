# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Target-owned batch invariance across executed dynamic-inference features.

Run backend/version groups in fresh torch.distributed.run processes using
MCORE_BI_TEST_BACKEND and MCORE_BI_TEST_FA_VERSION. Start te_native with
CUBLASLT_WORKSPACE_SIZE=0; missing required capabilities must fail, not skip.
"""

import pytest

from megatron.core.inference.config import AsyncScheduleMode, CudaGraphSizingDistribution
from tests.unit_tests.inference.engines import batch_invariant_test_utils as bi
from tests.unit_tests.inference.engines.batch_invariant_test_utils import (
    Case,
    assert_same_target,
    invariant_runtime,
    run_order,
)

MLA_CASE = Case(
    "mla",
    model=dict(
        multi_latent_attention=True,
        cache_mla_latents=True,
        num_attention_heads=64,
        qk_layernorm=True,
    ),
    context={"block_size_tokens": 64},
)

DENSE_CASES = [
    Case("dense"),
    Case("async", context={"async_sched_mode": AsyncScheduleMode.ASYNC}),
    Case("local", model={"transformer_impl": "local"}),
    Case("optimized", model={"transformer_impl": "inference_optimized", "add_bias_linear": False}),
    Case(
        "last-logits",
        context={"materialize_only_last_token_logits": True},
        sampling={"return_log_probs": False},
    ),
    Case("fused-rope", context={"use_flashinfer_fused_rope": True}),
    Case("chunked", context={"enable_chunked_prefill": True, "max_tokens": 128}, prompt_length=273),
    Case(
        "chunked-async",
        context={
            "enable_chunked_prefill": True,
            "max_tokens": 128,
            "async_sched_mode": AsyncScheduleMode.ASYNC,
        },
        prompt_length=273,
    ),
    Case(
        "prefix",
        context={"enable_prefix_caching": True},
        sampling={"return_log_probs": False},
        prompt_length=273,
        warm_prefix=True,
    ),
    Case("decode-graph", context={"num_cuda_graphs": 4, "max_tokens": 128}),
    Case(
        "decode-graph-linear",
        context={
            "num_cuda_graphs": 4,
            "max_tokens": 128,
            "cuda_graph_sizing_distribution": CudaGraphSizingDistribution.LINEAR,
        },
    ),
    Case(
        "mixed-graph",
        context={
            "num_cuda_graphs": 4,
            "max_tokens": 128,
            "use_cuda_graphs_for_non_decode_steps": True,
            "cuda_graph_all_prefills": True,
            "cuda_graph_mixed_prefill_count": 2,
            "enable_chunked_prefill": True,
        },
        prompt_length=273,
    ),
]


@pytest.mark.parametrize("case", DENSE_CASES, ids=lambda case: case.name)
def test_dense_dynamic_feature_batch_invariance(case):
    with invariant_runtime(case) as (backend, version):
        reference = run_order(case, backend, version, "solo")
        contrasts = []
        contrast_runs = {}
        for order in ("front", "back", "staggered"):
            result = run_order(case, backend, version, order)
            assert_same_target(reference, result)
            contrasts.extend(result[1].steps)
            contrast_runs[order] = result[1]
            assert any(
                s["decode"] and s["physical"] == 128 and s["requests"] == 65
                for s in result[1].steps
            ), (case.name, order)
        # Witness target movement and changed decode shape during execution with live neighbors.
        assert any(step["target_row"] > 0 for step in contrasts)
        ref_decode = {s["physical"] for s in reference[1].steps if s["decode"]}
        wide_decode = {s["physical"] for s in contrasts if s["decode"] and s["requests"] > 64}
        assert ref_decode == {64}, ref_decode
        assert 128 in wide_decode, wide_decode
        assert any(
            step["neighbor_queries"].get(202) == 33 for step in contrasts
        ), "long neighbor prompt never shared a target forward"
        assert any(
            event["target_active"] and event["generated"] == 2
            for event in contrast_runs["staggered"].retirement_events
        )
        if case.prompt_length == 17:
            assert any(not s["decode"] and s["physical"] == 128 for s in contrasts)
        print("BI_WITNESS", case.name, backend, version, "decode_shapes", ref_decode, wide_decode)


@pytest.mark.parametrize(
    "case",
    [Case("tp2-sp", tp=2, sp=True), Case("pp2", pp=2), Case("tp2-pp2-sp", tp=2, pp=2, sp=True)],
    ids=lambda case: case.name,
)
def test_parallel_batch_invariance(case):
    with invariant_runtime(case) as (backend, version):
        reference = run_order(case, backend, version, "solo")
        actual = run_order(case, backend, version, "back")
        assert_same_target(reference, actual)
        assert any(
            s["decode"] and s["physical"] == 128 and s["requests"] == 65 for s in actual[1].steps
        )


def test_mla_split_device_first_prefill():
    with invariant_runtime(MLA_CASE) as (backend, version):
        engine = bi.build_engine(MLA_CASE, backend, version)
        model = engine.controller.inference_wrapped_model.model
        assert version == 3 and model.config.use_cpu_initialization
        layers = [layer for layer in model.modules() if isinstance(layer, bi.MLASelfAttention)]
        sources = [layer.linear_kv_up_proj for layer in layers]
        assert len(sources) == 2 and all(source.weight.is_cuda for source in sources)
        params = bi.SamplingParams(num_tokens_to_generate=1, top_k=1, termination_id=-1)
        engine.add_request(bi.TARGET, bi.target_prompt(MLA_CASE.prompt_length), params)
        records = engine.step_modern()["finished_request_records"]
        assert len(records) == 1 and not engine.has_unfinished_requests()
        for layer, source in zip(layers, sources):
            norm, linear = layer.kv_layernorm, layer.linear_kv_up_proj_linear
            pairs = [(norm.weight, source.layer_norm_weight), (linear.weight, source.weight)]
            for actual, expected in pairs:
                assert actual.device == expected.device == source.weight.device
                assert actual.dtype == expected.dtype == bi.torch.bfloat16
                assert bi.torch.equal(actual, expected)
