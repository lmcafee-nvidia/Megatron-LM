# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Output, terminal, residency, and explicit unsupported BI contracts."""

from dataclasses import replace

import pytest
import torch

from megatron.core.inference.config import KVCacheManagementMode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.engines import batch_invariant_test_utils as fixture


@pytest.mark.parametrize(
    "invalid,reason",
    [
        ({"params_dtype": torch.float16}, "BF16 model parameters"),
        ({"flash_attention_version": 2}, "flash-attention-version 3 or 4"),
        ({"context_parallel_size": 2}, "does not support context parallelism"),
        ({"attention_dropout": 0.1}, "does not support attention dropout"),
    ],
)
def test_batch_invariant_support_boundary(invalid, reason):
    # Negative support classification only: these cases deliberately execute no
    # model and receive no positive runtime pair-coverage credit.
    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        batch_invariant_mode=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        flash_attention_version=3,
        attention_dropout=0.0,
    )
    kwargs.update(invalid)
    with pytest.raises(AssertionError, match=reason):
        TransformerConfig(**kwargs)


@pytest.mark.parametrize("output", ["top-n", "skip-prompt", "total-length", "bos"])
def test_batch_invariant_output_contract(output):
    options = dict(top_n_logprobs=3)
    if output == "skip-prompt":
        options["skip_prompt_log_probs"] = True
    if output == "total-length":
        options.update(num_tokens_to_generate=None, num_tokens_total=23)
    if output == "bos":
        options["add_BOS"] = True
    case = fixture.Case("output", sampling=options)
    with (
        fixture.invariant_runtime(case) as (backend, version),
        pytest.MonkeyPatch.context() as patch,
    ):
        if output == "bos":
            original_prompt = fixture.target_prompt
            patch.setattr(
                fixture, "target_prompt", lambda n: " ".join(map(str, original_prompt(n)))
            )
        reference = fixture.run_order(case, backend, version, "solo")
        actual = fixture.run_order(case, backend, version, "back")
        fixture.assert_same_target(reference, actual)
        request = actual[0]
        assert len(request.generated_tokens) == 6
        assert len(request.generated_top_n_logprobs) == 6
        assert all(len(row) == 3 for row in request.generated_top_n_logprobs)
        if output == "skip-prompt":
            assert not request.prompt_log_probs and not request.prompt_top_n_logprobs
        else:
            assert len(request.prompt_top_n_logprobs) == len(request.prompt_tokens) - 1
        if output == "bos":
            assert request.prompt_tokens[0] == 1 and len(request.prompt_tokens) == 18
        assert actual[1].sample_steps[0]["requests"] == 65


@pytest.mark.parametrize("terminal", ["eos", "stop-kept", "stop-trimmed"])
def test_batch_invariant_terminal_contract(terminal):
    case = fixture.Case("terminal")
    with fixture.invariant_runtime(case) as (backend, version):
        baseline = fixture.run_order(case, backend, version, "solo")
        tokens = baseline[0].generated_tokens
        assert len(tokens) == 6
        if terminal == "eos":
            params = dict(termination_id=tokens[0])
        else:
            # A stop is derived from an unmodified real greedy forward, never by
            # replacing model logits or forcing the sampler's returned token.
            params = dict(
                stop_words=[" ".join(map(str, tokens[:2]))],
                detokenize_stop_sequence=terminal == "stop-kept",
            )
        limited = replace(case, sampling=params)
        reference = fixture.run_order(limited, backend, version, "solo")
        actual = fixture.run_order(limited, backend, version, "back")
        fixture.assert_same_target(reference, actual)
        fixture.assert_same_target(baseline, actual, require_trajectory=False)
        expected_count = {"eos": 1, "stop-kept": 2, "stop-trimmed": 0}[terminal]
        assert len(actual[0].generated_tokens) == expected_count
        assert len(actual[0].generated_log_probs or []) == expected_count
        assert actual[1].sample_steps[0]["requests"] == 65


@pytest.mark.parametrize("mode", list(KVCacheManagementMode))
@pytest.mark.parametrize("static_pointers", [False, True])
def test_batch_invariant_suspend_resume(mode, static_pointers):
    case = fixture.Case(
        "suspend",
        context={
            "kv_cache_management_mode": mode,
            "static_kv_memory_pointers": static_pointers,
            "num_cuda_graphs": 4,
            "max_tokens": 128,
        },
    )
    transitions = []
    with (
        fixture.invariant_runtime(case) as (backend, version),
        pytest.MonkeyPatch.context() as patch,
    ):
        reference = fixture.run_order(case, backend, version, "solo")
        build = fixture.build_engine

        def resumed_engine(*args, **kwargs):
            engine = build(*args, **kwargs)
            original = engine.step_modern
            suspended = False

            def step(*args, **kwargs):
                nonlocal suspended
                result = original(*args, **kwargs)
                if not suspended and fixture.TARGET in engine.requests:
                    request = engine.get_request(fixture.TARGET)
                    if len(request.generated_tokens or []) >= 2:
                        before = tuple(request.generated_tokens)
                        pointer = engine.context.memory_buffer.data_ptr()
                        active = engine.context.total_request_count
                        assert active > 1, "suspension did not overlap live neighbors"
                        engine.suspend()
                        engine.resume()
                        # RECOMPUTE puts prior output into a checkpoint record;
                        # the new live segment correctly has no generated tokens.
                        merged = engine.requests[fixture.TARGET].record.merge()
                        assert tuple(merged.generated_tokens or []) == before
                        if static_pointers:
                            assert engine.context.memory_buffer.data_ptr() == pointer
                        transitions.append((mode.value, active, len(before)))
                        suspended = True
                return result

            patch.setattr(engine, "step_modern", step)
            return engine

        patch.setattr(fixture, "build_engine", resumed_engine)
        actual = fixture.run_order(case, backend, version, "back")
        fixture.assert_same_target(reference, actual)
        assert len(transitions) == 1
        assert any(
            s["decode"] and s["graph"] and s["replay"] and int(s["positions"][0]) >= 19
            for s in actual[1].steps
        )
        print("BI_RESIDENCY_WITNESS", transitions, "static_pointers", static_pointers)
