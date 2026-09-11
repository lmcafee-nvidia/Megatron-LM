# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial

import pytest

from megatron.core.transformer import attention
from tests.unit_tests.inference import test_coordinator_features_pairwise as features
from tests.unit_tests.inference.engines import test_dynamic_engine as engines


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "version", "symbols"),
    [
        ("optimized", None, ()),
        ("fa2", 2, ("flash_attn_varlen_func", "flash_attn_with_kvcache")),
        ("fa4", 4, ("flash_attn4_varlen_func",)),
    ],
)
async def test_routed_dense_backends_match_direct(monkeypatch, backend, version, symbols):
    constructor = partial(engines.TransformerConfig, flash_attention_version=version or 3)
    monkeypatch.setattr(engines, "TransformerConfig", constructor)
    build = engines.DynamicInferenceEngineTestBase._build_test_env
    routed = partial(features.routed_model, engine_factory=lambda config: build(config).engine)
    monkeypatch.setattr(features, "routed_model", routed)
    instrument = features._instrument_scenario_runtime

    def instrument_native(env, scenario, runtime):
        instrument(env, scenario, runtime)
        for symbol in symbols:
            kernel = getattr(attention, symbol)

            def observed(*args, _kernel=kernel, **kwargs):
                result = _kernel(*args, **kwargs)
                phase = "decode" if env.engine.context.is_decode_only() else "prefill"
                runtime[f"{backend}-{phase}"] += 1
                return result

            monkeypatch.setattr(attention, symbol, observed)

    monkeypatch.setattr(features, "_instrument_scenario_runtime", instrument_native)
    options = {"transformer_impl": "inference_optimized" if version is None else "local"}
    if version == 2:
        options.update(context_block_size_tokens=256, context_max_tokens=256)
    if version == 4:
        options["hidden_size"] = 128
    signals = ("inference-optimized",) if version is None else ()
    required = (
        ("module-forward:inference-optimized",)
        if version is None
        else (f"{backend}-prefill", f"{backend}-decode")
    )
    await features.test_routed_model_kernels_match_direct(monkeypatch, options, signals, required)
