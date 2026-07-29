# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-model cache-off/cache-on execution stress for prefix caching."""

import gc
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from tests.test_utils.python_scripts.prefix_cache_coverage import (
    generated_matrix_cases,
    load_manifest,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig as _DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase as _DynamicInferenceEngineTestBase,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder
from tests.unit_tests.test_utilities import Utils

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = load_manifest(_REPO_ROOT / "tests/test_utils/prefix_cache_coverage.yaml")


@dataclass
class _PrefixCacheExecutionConfig(_DynamicEngineTestConfig):
    """Arguments used only by the generated prefix-cache workload."""

    stress_output_mode: str = "tokens"
    stress_sampling_control: str = "greedy"
    stress_match_shape: str = "exact"
    stress_gap_steps: int = 1
    stress_cycle_count: int = 3
    stress_requests_per_cycle: int = 8
    stress_shared_request_count: int = 3
    stress_termination: str = "length"
    stress_termination_tokens: tuple[int, ...] | None = None
    stress_stop_word_ids: tuple[tuple[int, ...], ...] | None = None
    stress_capture_base_samples: bool = False
    stress_reseed_each_cycle: bool = False
    stress_pool_size: int | None = None


def _sampling_kwargs(control):
    return {
        "greedy": {"top_k": 1},
        "top_k": {"top_k": 10},
        "top_p": {"top_p": 0.9},
        "temperature": {"temperature": 0.7},
    }[control]


def _request_sampling_params(config, request_id):
    """Keep non-target traffic deterministic while the final request exercises sampling."""
    output_mode = config.stress_output_mode
    cycle, local_request_id = divmod(request_id, config.stress_requests_per_cycle)
    is_target = local_request_id == config.stress_requests_per_cycle - 2
    return_log_probs = output_mode in {"generated_logprobs", "prompt_logprobs", "top_n_logprobs"}
    top_n = 3 if output_mode == "top_n_logprobs" else 0
    needs_prompt_values = is_target and (
        output_mode == "prompt_logprobs"
        or (output_mode == "top_n_logprobs" and not config.materialize_only_last_token_logits)
    )
    termination_id = -1
    if is_target and config.stress_termination == "termination_token":
        assert config.stress_termination_tokens is not None
        termination_id = config.stress_termination_tokens[cycle]
    return SamplingParams(
        num_tokens_to_generate=config.num_tokens_to_generate,
        termination_id=termination_id,
        return_log_probs=return_log_probs,
        skip_prompt_log_probs=not needs_prompt_values,
        top_n_logprobs=top_n,
        detokenize_stop_sequence=True,
        return_prompt_tokens=True,
        **_sampling_kwargs(config.stress_sampling_control if is_target else "greedy"),
    )


def _prompt(config, request_id):
    block_size = config.context_block_size_tokens
    prompt_length = config.min_prompt_length
    cycle, local_request_id = divmod(request_id, config.stress_requests_per_cycle)
    is_target = local_request_id == config.stress_requests_per_cycle - 2
    is_unrelated = local_request_id in {
        config.stress_requests_per_cycle - 3,
        config.stress_requests_per_cycle - 1,
    }
    if is_target:
        # Keep the sampling/termination target within one chunk while giving it
        # enough complete blocks to execute every match-shape row itself.
        prompt_length = 3 * block_size

    shared = (
        torch.arange(prompt_length, dtype=torch.int64, device=torch.cuda.current_device())
        + cycle * 97
    ) % (config.vocab_size - 1)
    if is_unrelated:
        result = torch.full_like(shared, config.vocab_size - 1)
        result[0] = request_id
        return result
    if local_request_id < config.stress_shared_request_count:
        return shared

    matched_blocks = {"branching": 1, "partial": 2, "exact": 3}[config.stress_match_shape]
    divergence = matched_blocks * block_size
    if divergence == prompt_length:
        return shared
    result = shared.clone()
    result[divergence:] = (
        torch.arange(
            prompt_length - divergence, dtype=torch.int64, device=torch.cuda.current_device()
        )
        .add_(17 * local_request_id + 31 + cycle * 97)
        .remainder_(config.vocab_size - 1)
    )
    return result


def _float_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    return [float(item) for item in value]


def _top_n_list(value):
    if value is None:
        return None
    return [
        sorted((str(token), float(logprob)) for token, logprob in position.items())
        for position in value
    ]


def _snapshot(request):
    return {
        "status": request.status,
        "prompt_tokens": (
            None if request.prompt_tokens is None else request.prompt_tokens.detach().cpu().tolist()
        ),
        "generated_tokens": list(request.generated_tokens),
        "prompt_log_probs": _float_list(request.prompt_log_probs),
        "generated_log_probs": _float_list(request.generated_log_probs),
        "prompt_top_n_logprobs": _top_n_list(request.prompt_top_n_logprobs),
        "generated_top_n_logprobs": _top_n_list(request.generated_top_n_logprobs),
        "num_cached_tokens": request.num_cached_tokens,
    }


def _collect_finished(result, finished):
    for record in result["finished_request_records"]:
        request = record.merge()
        assert request.request_id not in finished
        finished[request.request_id] = _snapshot(request)


def _run_workload(config):
    env = TestPrefixCacheEngineExecution._build_test_env(config)
    env.engine.controller.tokenizer.detokenize = lambda tokens, **_: " ".join(
        f"token-{int(token)}" for token in tokens
    )
    if config.stress_pool_size is not None:
        context = env.engine.context
        previous_allocator = context.kv_block_allocator
        context.kv_block_allocator = KVBlockAllocator(
            context,
            pool_size=config.stress_pool_size,
            paused_limit=min(previous_allocator.paused_limit, config.stress_pool_size - 2),
            enable_prefix_caching=previous_allocator.enable_prefix_caching,
            prefix_caching_eviction_policy=previous_allocator.prefix_caching_eviction_policy,
        )
        if context.mamba_slot_allocator is not None:
            context.kv_block_allocator.on_blocks_deregistered = (
                context.mamba_slot_allocator.on_kv_blocks_deregistered
            )
    finished = {}
    chunk_continuations = 0
    base_samples = {}
    release_checks = 0
    target_match_depth_checks = 0
    rewind_metrics = {"calls": 0, "tokens": 0, "blocks_released": 0}

    if config.stress_capture_base_samples:
        original_post_process = env.engine.post_process_requests

        def capture_base_samples(request_ids, *args, **kwargs):
            sample = args[3]
            for request_id, token in zip(request_ids.tolist(), sample.tolist()):
                if request_id == env.engine.context.chunked_prefill_request_id:
                    continue
                if isinstance(token, list):
                    assert len(token) == 1
                    token = token[0]
                base_samples.setdefault(request_id, []).append(int(token))
            return original_post_process(request_ids, *args, **kwargs)

        env.engine.post_process_requests = capture_base_samples

    if config.num_speculative_tokens > 0:
        controller = env.engine.controller
        original_rewind = controller._rewind_kv_cache

        def observe_rewind():
            context = controller.inference_wrapped_model.inference_context
            active_count = context.total_request_count - context.paused_request_count
            active_slice = slice(context.paused_request_count, context.total_request_count)
            prefill = context.request_in_prefill_status_tensor[active_slice]
            accepted = controller._accepted_token_counts_per_request[:active_count].cpu()
            decode = prefill == 0
            before_offsets = context.request_kv_length_offsets[active_slice].clone()
            before_last_block_offsets = context.request_last_kv_block_offset[active_slice].clone()
            expected_rewind = torch.where(
                decode, config.num_speculative_tokens - accepted, torch.zeros_like(accepted)
            ).clamp_min_(0)
            rewind_metrics["calls"] += 1
            blocks, remove_mask = original_rewind()
            actual_rewind = before_offsets - context.request_kv_length_offsets[active_slice]
            assert torch.equal(actual_rewind, expected_rewind.to(actual_rewind.dtype))
            assert torch.equal(
                remove_mask, decode & (before_last_block_offsets - expected_rewind < 0)
            )
            rewind_metrics["tokens"] += int(actual_rewind.sum().item())
            rewind_metrics["blocks_released"] += int(remove_mask.sum().item())
            return blocks, remove_mask

        controller._rewind_kv_cache = observe_rewind

    for cycle in range(config.stress_cycle_count):
        if config.stress_reseed_each_cycle:
            env.engine.controller.sampling_rng.manual_seed(config.random_seed + cycle)
        for local_request_id in range(config.stress_requests_per_cycle):
            request_id = cycle * config.stress_requests_per_cycle + local_request_id
            if local_request_id == config.stress_requests_per_cycle - 1:
                # Finish the parity group before admitting one isolated,
                # unrelated pressure request. It creates three additional cached
                # blocks per cycle without increasing live-request capacity.
                while env.engine.has_unfinished_requests():
                    _collect_finished(env.engine.step_modern(), finished)
                    chunk_continuations += int(env.engine.context.chunked_prefill_request_id != -1)
            env.engine.add_request(
                request_id=request_id,
                prompt=_prompt(config, request_id),
                sampling_params=_request_sampling_params(config, request_id),
            )
            if (
                local_request_id == config.stress_requests_per_cycle - 2
                and env.engine.context.enable_prefix_caching
            ):
                request = env.engine.get_request(request_id)
                routable_hashes = set(env.engine.context.kv_block_allocator.kv_hash_to_block_id)
                if env.engine.context.is_hybrid_model:
                    routable_hashes &= set(env.engine.context.mamba_slot_allocator.hash_to_block_id)
                available_match_depth = 0
                for block_hash in request.precomputed_block_hashes:
                    if block_hash not in routable_hashes:
                        break
                    available_match_depth += 1
                expected_match_depth = {"branching": 1, "partial": 2, "exact": 3}[
                    config.stress_match_shape
                ]
                assert available_match_depth == expected_match_depth
                target_match_depth_checks += 1
            if (
                local_request_id == config.stress_requests_per_cycle - 2
                and config.stress_termination == "stop_word"
            ):
                assert config.stress_stop_word_ids is not None
                env.engine.get_request(request_id).stop_word_ids = [
                    list(config.stress_stop_word_ids[cycle])
                ]
            for _ in range(config.stress_gap_steps):
                _collect_finished(env.engine.step_modern(), finished)
                chunk_continuations += int(env.engine.context.chunked_prefill_request_id != -1)

        step_count = 0
        while env.engine.has_unfinished_requests():
            _collect_finished(env.engine.step_modern(), finished)
            chunk_continuations += int(env.engine.context.chunked_prefill_request_id != -1)
            step_count += 1
            assert step_count < 2_000, "prefix-cache execution workload did not converge"

        allocator = env.engine.context.kv_block_allocator
        if allocator.enable_prefix_caching:
            assert int(allocator.block_ref_counts.sum().item()) == 0
        else:
            assert allocator.pool_avail == allocator.pool_size - 1
        release_checks += 1

    request_count = config.stress_cycle_count * config.stress_requests_per_cycle
    assert set(finished) == set(range(request_count))
    assert env.engine.evicted_request_count == 0
    metrics = env.engine.get_prefix_cache_metrics()
    speculative_metrics = {
        "steps": int(env.engine._spec_steps),
        "proposed": int(sum(env.engine._spec_tokens_proposed_per_pos)),
        "accepted": int(sum(env.engine._spec_tokens_accepted_per_pos)),
        "rewind_calls": rewind_metrics["calls"],
        "rewound_tokens": rewind_metrics["tokens"],
        "rewind_blocks_released": rewind_metrics["blocks_released"],
        "release_checks": release_checks,
        "target_match_depth_checks": target_match_depth_checks,
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return finished, metrics, chunk_continuations, speculative_metrics, base_samples


def _assert_float_parity(reference, actual):
    assert (reference is None) is (actual is None)
    if reference is None:
        return
    torch.testing.assert_close(
        torch.tensor(actual, dtype=torch.float32),
        torch.tensor(reference, dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
        equal_nan=True,
    )


def _assert_top_n_parity(reference, actual):
    assert (reference is None) is (actual is None)
    if reference is None:
        return
    assert len(actual) == len(reference)
    for expected_position, actual_position in zip(reference, actual):
        assert [token for token, _ in actual_position] == [token for token, _ in expected_position]
        _assert_float_parity(
            [value for _, value in expected_position], [value for _, value in actual_position]
        )


def _assert_output_parity(reference, actual):
    assert set(actual) == set(reference)
    for request_id in reference:
        expected = reference[request_id]
        observed = actual[request_id]
        assert observed["status"] == expected["status"]
        assert observed["prompt_tokens"] == expected["prompt_tokens"]
        assert observed["generated_tokens"] == expected["generated_tokens"]
        _assert_float_parity(expected["prompt_log_probs"], observed["prompt_log_probs"])
        _assert_float_parity(expected["generated_log_probs"], observed["generated_log_probs"])
        _assert_top_n_parity(expected["prompt_top_n_logprobs"], observed["prompt_top_n_logprobs"])
        _assert_top_n_parity(
            expected["generated_top_n_logprobs"], observed["generated_top_n_logprobs"]
        )


def _assert_requested_outputs(outputs, config):
    """Prove each output mode produced the fields whose parity is asserted."""
    for request_id, output in outputs.items():
        _, local_request_id = divmod(request_id, config.stress_requests_per_cycle)
        is_target = local_request_id == config.stress_requests_per_cycle - 2
        generated_count = len(output["generated_tokens"])

        if config.stress_output_mode == "tokens":
            assert output["prompt_log_probs"] is None
            assert output["generated_log_probs"] is None
            assert output["prompt_top_n_logprobs"] is None
            assert output["generated_top_n_logprobs"] is None
            continue

        assert output["generated_log_probs"] is not None
        assert len(output["generated_log_probs"]) == generated_count
        expects_prompt_values = is_target and (
            config.stress_output_mode == "prompt_logprobs"
            or (
                config.stress_output_mode == "top_n_logprobs"
                and not config.materialize_only_last_token_logits
            )
        )
        if expects_prompt_values:
            assert output["prompt_log_probs"] is not None
            assert len(output["prompt_log_probs"]) > 0
        else:
            assert output["prompt_log_probs"] is None

        if config.stress_output_mode == "top_n_logprobs":
            generated_top_n = output["generated_top_n_logprobs"]
            assert generated_top_n is not None
            assert len(generated_top_n) == generated_count
            assert all(len(position) == 3 for position in generated_top_n)
            if expects_prompt_values:
                prompt_top_n = output["prompt_top_n_logprobs"]
                assert prompt_top_n is not None
                assert len(prompt_top_n) == len(output["prompt_log_probs"])
                assert all(len(position) == 3 for position in prompt_top_n)
            else:
                assert output["prompt_top_n_logprobs"] is None
        else:
            assert output["prompt_top_n_logprobs"] is None
            assert output["generated_top_n_logprobs"] is None


def _target_request_ids(config):
    return tuple(
        cycle * config.stress_requests_per_cycle + config.stress_requests_per_cycle - 2
        for cycle in range(config.stress_cycle_count)
    )


def _termination_controls(config, probe_outputs, probe_base_samples):
    """Select controls observed before the configured early completion point."""
    target_ids = _target_request_ids(config)
    if config.stress_termination == "termination_token":
        result = []
        for request_id in target_ids:
            samples = probe_base_samples[request_id]
            assert len(samples) >= 3
            result.append(samples[min(2, len(samples) - 2)])
        return {"stress_termination_tokens": tuple(result)}

    if config.stress_termination == "stop_word":
        result = []
        for request_id in target_ids:
            tokens = probe_outputs[request_id]["generated_tokens"]
            assert len(tokens) >= 5
            start = min(2, len(tokens) - 3)
            result.append(tuple(tokens[start : start + 2]))
        return {"stress_stop_word_ids": tuple(result)}

    assert config.stress_termination == "length"
    return {}


def _assert_termination_path(outputs, config):
    target_ids = _target_request_ids(config)
    for cycle, request_id in enumerate(target_ids):
        generated = outputs[request_id]["generated_tokens"]
        if config.stress_termination == "length":
            assert len(generated) == config.num_tokens_to_generate
        elif config.stress_termination == "termination_token":
            assert 0 < len(generated) < config.num_tokens_to_generate
            assert generated[-1] == config.stress_termination_tokens[cycle]
        else:
            assert config.stress_termination == "stop_word"
            stop_word = list(config.stress_stop_word_ids[cycle])
            assert 0 < len(generated) < config.num_tokens_to_generate
            assert generated[-len(stop_word) :] == stop_word
    return len(target_ids)


class TestPrefixCacheEngineExecution(_DynamicInferenceEngineTestBase):
    """Single-rank real execution; distributed and graph backends stay functional-owned."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "case_id,row",
        generated_matrix_cases(
            "engine_execution_local", _MANIFEST["matrices"]["engine_execution_local"]
        ),
        ids=lambda value: value if isinstance(value, str) else None,
    )
    @torch.inference_mode()
    def test_execution_pairwise_stress(self, case_id, row):
        """Compare three stressed request groups on independent cache-off/on engines."""
        del case_id
        if row["architecture"] == "hybrid":
            sequence_packing_available, reason = _check_mamba_sequence_packing_support()
            assert sequence_packing_available, reason

        block_size = 128 if row["architecture"] == "hybrid" else 16
        prompt_length = 3 * block_size + block_size // 2
        chunk_tokens = 3 * block_size
        num_speculative_tokens = 2 if row["speculation"] == "mtp" else 0
        common = dict(
            num_requests=0,
            min_prompt_length=prompt_length,
            max_prompt_length=prompt_length,
            num_tokens_to_generate=12,
            max_sequence_length=prompt_length + 12 + num_speculative_tokens,
            context_block_size_tokens=block_size,
            context_max_requests=8,
            context_max_tokens=(
                chunk_tokens if row["execution_mode"] == "chunked_prefill" else 2 * prompt_length
            ),
            context_buffer_size_gb=0.1,
            context_paused_buffer_size_gb=0.0,
            model_provider=row["architecture"],
            materialize_only_last_token_logits=row["logit_materialization"] == "last_token",
            enable_chunked_prefill=row["execution_mode"] == "chunked_prefill",
            prefix_caching_eviction_policy={
                "lru": PrefixCachingEvictionPolicy.LRU,
                "ref_zero": PrefixCachingEvictionPolicy.REF_ZERO,
            }[row["eviction_policy"]],
            logprobs_mode=(
                "raw_logprobs"
                if row["logprob_representation"] == "none"
                else f"{row['logprob_representation']}_logprobs"
            ),
            use_flashinfer_fused_rope=False,
            sampling_backend="torch",
            num_speculative_tokens=num_speculative_tokens,
            stress_output_mode=row["output_mode"],
            stress_sampling_control=row["sampling_control"],
            stress_match_shape=row["match_shape"],
            stress_termination=row["termination"],
            stress_reseed_each_cycle=True,
            stress_pool_size=23,
        )

        reference_config = _PrefixCacheExecutionConfig(
            **common, enable_prefix_caching=False, prefix_caching_mamba_gb=None
        )
        if row["termination"] != "length":
            probe_config = replace(
                reference_config, stress_termination="length", stress_capture_base_samples=True
            )
            probe_outputs, _, _, _, probe_base_samples = _run_workload(probe_config)
            reference_config = replace(
                reference_config,
                **_termination_controls(reference_config, probe_outputs, probe_base_samples),
            )

        (reference, reference_metrics, reference_chunks, reference_speculation, _) = _run_workload(
            reference_config
        )
        cache_config = replace(
            reference_config,
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=(0.01 if row["architecture"] == "hybrid" else None),
        )
        actual, cache_metrics, cache_chunks, cache_speculation, _ = _run_workload(cache_config)

        _assert_requested_outputs(reference, reference_config)
        _assert_requested_outputs(actual, cache_config)
        _assert_output_parity(reference, actual)
        assert _assert_termination_path(reference, reference_config) == 3
        assert _assert_termination_path(actual, cache_config) == 3
        expected_matched_blocks = {"branching": 1, "partial": 2, "exact": 3}[row["match_shape"]]
        for cycle in range(cache_config.stress_cycle_count):
            cycle_start = cycle * cache_config.stress_requests_per_cycle
            follower_id = cycle_start + cache_config.stress_shared_request_count
            unrelated_ids = (
                cycle_start + cache_config.stress_requests_per_cycle - 3,
                cycle_start + cache_config.stress_requests_per_cycle - 1,
            )
            target_id = cycle_start + cache_config.stress_requests_per_cycle - 2
            assert reference[follower_id]["num_cached_tokens"] == 0
            assert actual[follower_id]["num_cached_tokens"] == (
                expected_matched_blocks * block_size
            )
            for unrelated_id in unrelated_ids:
                assert reference[unrelated_id]["num_cached_tokens"] == 0
                assert actual[unrelated_id]["num_cached_tokens"] == 0
            assert reference[target_id]["num_cached_tokens"] == 0
            target_bypasses_cache = row["output_mode"] == "prompt_logprobs" or (
                row["output_mode"] == "top_n_logprobs" and row["logit_materialization"] == "full"
            )
            assert actual[target_id]["num_cached_tokens"] == (
                0 if target_bypasses_cache else expected_matched_blocks * block_size
            )
        assert reference_metrics["enabled"] is False
        assert reference_metrics["hits"] == 0
        assert cache_metrics["enabled"] is True
        assert cache_metrics["hits"] >= 12
        assert cache_metrics["blocks_matched"] >= 12
        assert cache_metrics["prefill_tokens_skipped"] >= 3 * block_size
        assert cache_metrics["prefill_tokens_computed"] > 0
        assert cache_metrics["kv_physical_reuses"] >= 3
        if row["eviction_policy"] == "lru":
            assert cache_metrics["kv_lru_evictions"] >= 3
        else:
            assert cache_metrics["kv_lru_evictions"] == 0
            assert cache_metrics["kv_blocks_deregistered"] >= 3
        if row["speculation"] == "mtp":
            assert reference_speculation["steps"] >= 3
            assert cache_speculation["steps"] >= 3
            assert reference_speculation["proposed"] > 0
            assert cache_speculation["proposed"] > 0
            assert reference_speculation["accepted"] < reference_speculation["proposed"]
            assert cache_speculation["accepted"] < cache_speculation["proposed"]
            assert reference_speculation["rewind_calls"] > 0
            assert cache_speculation["rewind_calls"] > 0
            assert reference_speculation["rewound_tokens"] > 0
            assert cache_speculation["rewound_tokens"] > 0
        else:
            for observed in (reference_speculation, cache_speculation):
                assert observed["steps"] == 0
                assert observed["proposed"] == 0
                assert observed["accepted"] == 0
                assert observed["rewind_calls"] == 0
                assert observed["rewound_tokens"] == 0
                assert observed["rewind_blocks_released"] == 0
        assert reference_speculation["release_checks"] == 3
        assert cache_speculation["release_checks"] == 3
        assert reference_speculation["target_match_depth_checks"] == 0
        assert cache_speculation["target_match_depth_checks"] == 3
        if row["architecture"] == "hybrid":
            assert cache_metrics["mamba_commits"] >= 1
            assert cache_metrics["mamba_restore_hits"] >= 1
        if row["execution_mode"] == "chunked_prefill":
            assert reference_chunks >= 3
            assert cache_chunks >= 3
        else:
            assert reference_chunks == 0
            assert cache_chunks == 0
