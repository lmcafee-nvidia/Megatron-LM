# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging
import math
import os
from pathlib import Path
from statistics import median

import numpy as np
import pytest
import yaml

from tests.test_utils.python_scripts.prefix_cache_coverage import (
    assert_runtime_evidence,
    load_manifest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_NON_REQUEST_TOP_LEVEL_KEYS = {
    # System-level metrics
    "throughput",
    "lifetime_prefill_token_count",
    "async_sched_step_count",
    "async_sched_compaction_step_count",
    "prefix_cache_observations",
    "prefix_cache_comparison",
    "memory_phases",
    # Peak memory metrics (added by inference scripts; optionally checked if present in golden values)
    "mem-max-allocated-bytes",
}

_STRICT_LOGPROB_ABS_TOL = 1.0e-6
_GOLDEN_LOGPROB_ABS_TOL = 1.0e-3
_PREFIX_CACHE_MANIFEST = (
    Path(__file__).resolve().parents[2] / "test_utils/prefix_cache_coverage.yaml"
)


def _median_as_float(value):
    """Convert scalar or list metric to a single float (median).

    For list metrics (e.g., per-request throughput), treat the first element as
    warmup if length > 1, matching existing throughput behavior.
    """
    if isinstance(value, list):
        assert len(value) > 0, "Metric list is empty."
        values = [float(v) for v in value]
        if len(values) > 1:
            values = values[1:]
        return float(median(values))
    return float(value)


def _bytes_to_gib(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0**3)


def _assert_finite_close(
    expected: float, current: float, *, label: str, abs_tol: float, rel_tol: float
) -> None:
    """Compare two finite scalar values with an explicit tolerance."""
    expected = float(expected)
    current = float(current)
    assert math.isfinite(expected), f"{label}: reference value is not finite: {expected}"
    assert math.isfinite(current), f"{label}: current value is not finite: {current}"
    assert math.isclose(expected, current, abs_tol=abs_tol, rel_tol=rel_tol), (
        f"{label}: expected {expected:.9g}, got {current:.9g}; "
        f"abs diff {abs(expected - current):.9g} exceeds "
        f"abs_tol={abs_tol:.3g}, rel_tol={rel_tol:.3g}"
    )


def _assert_logprob_sequence(
    expected, current, *, label: str, abs_tol: float, rel_tol: float
) -> None:
    """Compare a logprob sequence position by position."""
    assert (expected is None) == (current is None), (
        f"{label}: one result omitted the sequence: expected={expected is None}, "
        f"current={current is None}"
    )
    if expected is None:
        return
    assert len(expected) == len(
        current
    ), f"{label}: position count differs: {len(expected)} vs {len(current)}"
    for position, (expected_value, current_value) in enumerate(zip(expected, current)):
        _assert_finite_close(
            expected_value,
            current_value,
            label=f"{label}[{position}]",
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )


def _assert_top_n_sequence(
    expected, current, *, label: str, abs_tol: float, rel_tol: float
) -> None:
    """Compare top-N token/logprob mappings at each token position."""
    assert (expected is None) == (current is None), (
        f"{label}: one result omitted the sequence: expected={expected is None}, "
        f"current={current is None}"
    )
    if expected is None:
        return
    assert len(expected) == len(
        current
    ), f"{label}: position count differs: {len(expected)} vs {len(current)}"
    for position, (expected_top_n, current_top_n) in enumerate(zip(expected, current)):
        assert set(expected_top_n) == set(current_top_n), (
            f"{label}[{position}]: token keys differ: "
            f"{sorted(expected_top_n)} vs {sorted(current_top_n)}"
        )
        for token, expected_value in expected_top_n.items():
            _assert_finite_close(
                expected_value,
                current_top_n[token],
                label=f"{label}[{position}][{token!r}]",
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )


def _compare_request_outputs(
    expected: dict, current: dict, *, label: str, abs_tol: float, rel_tol: float
) -> None:
    """Compare one cache-off/cache-on scenario at identical output positions."""
    assert expected["scenario_id"] == current["scenario_id"], (
        f"{label}: scenario identity differs: "
        f"{expected['scenario_id']!r} vs {current['scenario_id']!r}"
    )
    assert (
        expected["scenario_prompt"] == current["scenario_prompt"]
    ), f"{label}: scenario prompt differs"
    assert expected["prompt_token_count"] == current["prompt_token_count"], (
        f"{label}: prompt token count differs: "
        f"{expected['prompt_token_count']} vs {current['prompt_token_count']}"
    )
    assert expected["skip_prompt_log_probs"] == current["skip_prompt_log_probs"]
    assert expected["generated_tokens"] == current["generated_tokens"], (
        f"{label}: generated token mismatch:\n"
        f"cache_off: {expected['generated_tokens']}\n"
        f"cache_on:  {current['generated_tokens']}"
    )
    assert (
        expected["generated_text"] == current["generated_text"]
    ), f"{label}: generated text differs between cache-off and cache-on execution"
    assert expected["output_length"] == current["output_length"]
    assert expected["request_status"] == current["request_status"]
    if "routing_indices" in expected or "routing_indices" in current:
        assert expected.get("routing_indices") == current.get("routing_indices"), (
            f"{label}: reconstructed routing indices differ between cache-off "
            "and cache-on execution"
        )

    for field in ("prompt_logprobs", "generated_logprobs"):
        if field in expected or field in current:
            _assert_logprob_sequence(
                expected.get(field),
                current.get(field),
                label=f"{label}.{field}",
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )

    generated_count = len(current["generated_tokens"])
    for field in ("generated_logprobs", "generated_top_n_logprobs"):
        values = current.get(field)
        if values is not None:
            assert len(values) == generated_count, (
                f"{label}.{field}: expected {generated_count} generated positions, "
                f"got {len(values)}"
            )

    prompt_position_count = max(0, int(current["prompt_token_count"]) - 1)
    for field in ("prompt_logprobs", "prompt_top_n_logprobs"):
        values = current.get(field)
        if values is not None:
            expected_count = 0 if current["skip_prompt_log_probs"] else prompt_position_count
            assert len(values) == expected_count, (
                f"{label}.{field}: expected {expected_count} prompt positions, "
                f"got {len(values)}"
            )

    for field in ("prompt_top_n_logprobs", "generated_top_n_logprobs"):
        if field in expected or field in current:
            _assert_top_n_sequence(
                expected.get(field),
                current.get(field),
                label=f"{label}.{field}",
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )


def _compare_prefix_cache_phases(payload: dict, comparison_config: dict) -> None:
    """Validate direct cache-off/cache-on parity and cache-on activation."""
    assert (
        payload.get("schema_version") == 1
    ), f"Unsupported prefix_cache_comparison schema: {payload.get('schema_version')}"
    cache_off = payload["cache_off"]
    cache_on = payload["cache_on"]
    assert cache_off["prefix_caching_enabled"] is False
    assert cache_on["prefix_caching_enabled"] is True
    if "expected_lifecycle_modes" in comparison_config:
        assert cache_on.get("lifecycle_modes") == comparison_config["expected_lifecycle_modes"], (
            f"cache_on lifecycle modes were {cache_on.get('lifecycle_modes')!r}, "
            f"expected {comparison_config['expected_lifecycle_modes']!r}"
        )

    scenario_ids = [str(scenario_id) for scenario_id in payload["scenario_ids"]]
    assert set(cache_off["requests"]) == set(
        scenario_ids
    ), "cache_off scenario set does not match declared scenario_ids"
    cycles = cache_on["cycles"]
    assert cycles, "cache_on contains no stress cycles"
    supplemental_outputs = cache_on.get("supplemental_outputs", [])
    if "generated_output_count" in cache_off:
        assert cache_off["generated_output_count"] == len(cache_off["requests"])
    if "generated_output_count" in cache_on:
        accounted_outputs = sum(len(cycle["requests"]) for cycle in cycles) + len(
            supplemental_outputs
        )
        assert cache_on["generated_output_count"] == accounted_outputs, (
            "cache_on generated outputs were not all included in parity comparisons: "
            f"generated {cache_on['generated_output_count']}, accounted for {accounted_outputs}"
        )
    assert any(
        int(request.get("num_cached_tokens", 0)) > 0
        for cycle in cycles
        for request in cycle["requests"].values()
    ), "cache_on reported no request with cached prompt tokens"

    abs_tol = float(comparison_config.get("prefix_cache_logprob_abs_tol", _STRICT_LOGPROB_ABS_TOL))
    rel_tol = float(comparison_config.get("prefix_cache_logprob_rel_tol", 0.0))
    supplemental_labels = [entry["label"] for entry in supplemental_outputs]
    assert len(supplemental_labels) == len(set(supplemental_labels))
    for entry in supplemental_outputs:
        reference_scenario_id = str(entry["reference_scenario_id"])
        assert reference_scenario_id in cache_off["requests"], (
            f"supplemental output {entry['label']!r} names missing cache-off "
            f"scenario {reference_scenario_id!r}"
        )
        _compare_request_outputs(
            cache_off["requests"][reference_scenario_id],
            entry["request"],
            label=f"supplemental[{entry['label']}]",
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    cache_off_by_lifecycle = payload.get("cache_off_by_lifecycle", {"default": cache_off})
    for cycle in cycles:
        reference_key = cycle.get("reference_key", "default")
        assert reference_key in cache_off_by_lifecycle, (
            f"cache_on cycle {cycle['cycle_index']} names missing cache_off "
            f"reference {reference_key!r}"
        )
        cycle_reference = cache_off_by_lifecycle[reference_key]
        assert cycle_reference["prefix_caching_enabled"] is False
        if "cache_memory_overhead_bytes" in comparison_config:
            allowed_overhead = int(comparison_config["cache_memory_overhead_bytes"])
            assert allowed_overhead >= 0
            reference_peak = int(cycle_reference["memory"]["peak_allocated_bytes"])
            cache_peak = int(cycle["memory"]["peak_allocated_bytes"])
            assert cache_peak <= reference_peak + allowed_overhead, (
                f"cache_on cycle {cycle['cycle_index']} peak allocation "
                f"{_bytes_to_gib(cache_peak):.3f} GiB exceeds its cache_off "
                f"reference {_bytes_to_gib(reference_peak):.3f} GiB plus the "
                f"declared cache overhead {_bytes_to_gib(allowed_overhead):.3f} GiB"
            )
        current_requests = cycle["requests"]
        assert set(current_requests) == set(
            scenario_ids
        ), f"cache_on cycle {cycle['cycle_index']} scenario set differs from cache_off"
        for scenario_id in scenario_ids:
            _compare_request_outputs(
                cycle_reference["requests"][scenario_id],
                current_requests[scenario_id],
                label=f"cycle[{cycle['cycle_index']}].scenario[{scenario_id}]",
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        cycle_observations = cycle["observations"]
        assert (
            cycle_observations["prefix_cache_hits"] > 0
        ), f"cache_on cycle {cycle['cycle_index']} did not execute a prefix-cache hit"
        assert (
            cycle_observations["prefix_cache_blocks_matched"] > 0
        ), f"cache_on cycle {cycle['cycle_index']} did not match a prefix block"
        if payload.get("require_prefill_skip", True):
            assert (
                cycle_observations["prefill_tokens_skipped"] > 0
            ), f"cache_on cycle {cycle['cycle_index']} skipped no prefill tokens"
        for name, minimum in comparison_config.get("minimum_cycle_observations", {}).items():
            assert name in cycle_observations, (
                f"cache_on cycle {cycle['cycle_index']} did not report required "
                f"observation {name!r}"
            )
            assert float(cycle_observations[name]) >= float(minimum), (
                f"cache_on cycle {cycle['cycle_index']} observation {name!r} was "
                f"{cycle_observations[name]}, below required minimum {minimum}"
            )

    for reference_key, reference in cache_off_by_lifecycle.items():
        reference_observations = reference["observations"]
        assert reference_observations["prefix_caching_enabled"] is False, reference_key
        assert reference_observations["prefix_cache_hits"] == 0, reference_key
        assert reference_observations["prefix_cache_blocks_matched"] == 0, reference_key

    observations = cache_on["aggregate_observations"]
    assert observations["prefix_caching_enabled"] is True
    assert (
        observations["prefix_cache_hits"] > 0
    ), "cache_on did not record a prefix-cache hit; the workload did not activate caching"
    assert (
        observations["prefix_cache_blocks_matched"] > 0
    ), "cache_on did not match a block; the workload did not activate caching"
    if payload.get("require_prefill_skip", True):
        assert (
            observations["prefill_tokens_skipped"] > 0
        ), "cache_on matched blocks but skipped no prefill tokens"

    for name, minimum in comparison_config.get("minimum_observations", {}).items():
        assert name in observations, f"cache_on did not report required observation {name!r}"
        assert float(observations[name]) >= float(minimum), (
            f"cache_on observation {name!r} was {observations[name]}, "
            f"below required minimum {minimum}"
        )

    for name, expected in comparison_config.get("expected_observations", {}).items():
        assert name in observations, f"cache_on did not report required observation {name!r}"
        assert observations[name] == expected, (
            f"cache_on observation {name!r} was {observations[name]!r}, " f"expected {expected!r}"
        )


def _prefix_cache_runtime_evidence(payload: dict, output_current: dict) -> dict:
    """Translate harness counters into the committed stress-contract vocabulary."""
    observations = payload["cache_on"]["aggregate_observations"]
    return {
        "cache_off_on_comparisons": 1,
        "cache_state_updates": int(observations.get("seed_ranks_with_routable_prefixes", 0))
        + int(observations.get("routing_ranks_cleared", 0)),
        "cache_mutations": sum(
            int(observations.get(name, 0))
            for name in (
                "kv_deregistered_blocks",
                "kv_lru_evicted_blocks",
                "kv_physical_block_reuses",
                "kv_epoch_invalidated_blocks",
            )
        ),
        "chunk_boundaries_crossed": int(observations.get("chunk_boundaries_crossed", 0)),
        "concurrent_followers": int(observations.get("concurrent_followers", 0)),
        "epoch_invalidations": int(observations.get("kv_epoch_invalidated_blocks", 0)),
        "eligible_cache_followers": int(observations.get("eligible_cache_followers", 0)),
        "graph_replays": int(observations.get("cuda_graph_steps", 0)),
        "lru_evictions": int(observations.get("kv_lru_evicted_blocks", 0)),
        "mamba_restorations": int(observations.get("mamba_restore_hits", 0)),
        "mamba_slot_evictions": int(observations.get("mamba_evictions", 0)),
        "matched_blocks": int(observations.get("prefix_cache_blocks_matched", 0)),
        "memory_phases_measured": len(output_current.get("memory_phases", [])),
        "physical_reuses": int(observations.get("kv_physical_block_reuses", 0)),
        "pool_exhaustions": int(observations.get("kv_lru_evicted_blocks", 0) > 0),
        "prefill_tokens_computed": int(observations.get("prefill_tokens_computed", 0)),
        "prefill_tokens_skipped": int(observations.get("prefill_tokens_skipped", 0)),
        "prompt_logprob_bypass_requests": int(
            observations.get("prompt_logprob_bypass_requests", 0)
        ),
        "prompt_logprob_cache_immutability_checks": int(
            observations.get("prompt_logprob_cache_immutability_checks", 0)
        ),
        "producer_follower_cycles": int(observations.get("producer_follower_cycles", 0)),
        "refzero_deregistrations": int(observations.get("kv_deregistered_blocks", 0)),
        "route_changes": int(observations.get("routing_owner_mismatches", 0))
        + int(observations.get("routing_removal_checks", 0)),
        "routing_matched_requests": int(observations.get("routing_matched_requests", 0)),
        "routing_owner_matches": int(observations.get("routing_owner_matches", 0)),
        "routing_owner_mismatches": int(observations.get("routing_owner_mismatches", 0)),
        "routing_removal_checks": int(observations.get("routing_removal_checks", 0)),
        "routing_seed_distinct_ranks": int(observations.get("routing_seed_distinct_ranks", 0)),
        "routing_reconstructions": int(observations.get("routing_reconstruction_requests", 0)),
        "suspend_resume_cycles": min(
            int(observations.get("suspend_count", 0)), int(observations.get("resume_count", 0))
        ),
    }


def _check_memory_phase_leak(payload: dict, comparison_config: dict) -> None:
    """Bound end-of-cycle allocation growth across cache-on stress cycles."""
    cycles = payload["cache_on"]["cycles"]
    assert len(cycles) >= 2, "memory_phase_leak requires at least two cache_on cycles"
    assert (
        "memory_phase_leak_fraction" in comparison_config
        or "memory_phase_leak_bytes" in comparison_config
    ), (
        "memory_phase_leak requires an explicit memory_phase_leak_fraction or "
        "memory_phase_leak_bytes allowance in INFERENCE_COMPARISON"
    )
    end_allocations = [int(cycle["memory"]["end_allocated_bytes"]) for cycle in cycles]
    baseline = end_allocations[0]
    fraction = float(comparison_config.get("memory_phase_leak_fraction", 0.0))
    extra_bytes = int(comparison_config.get("memory_phase_leak_bytes", 0))
    upper_bound = baseline * (1.0 + fraction) + extra_bytes
    for cycle_index, allocated in enumerate(end_allocations[1:], start=1):
        assert allocated <= upper_bound, (
            f"cache_on cycle {cycle_index} ends with {allocated} allocated bytes "
            f"({_bytes_to_gib(allocated):.3f} GiB), above the phase-0 leak bound "
            f"{upper_bound:.0f} bytes ({_bytes_to_gib(upper_bound):.3f} GiB)"
        )


def test_inference_pipeline(
    golden_values_path: str, test_values_path: str, model_config_path: str
) -> None:
    if os.getenv("ENABLE_LIGHTWEIGHT_MODE") == "true":
        pytest.skip("Lightweight mode enabled. Skipping test.")

    with (
        open(golden_values_path, 'r') as f1,
        open(test_values_path, 'r') as f2,
        open(model_config_path, 'r') as f3,
    ):
        golden_values_content = f1.read()
        tensorboard_content = f2.read()
        model_config_content = f3.read()

    model_config = yaml.safe_load(model_config_content)
    metrics = set(model_config["METRICS"] or [])
    comparison_config = model_config.get("INFERENCE_COMPARISON", {})
    if not metrics:
        print("No metrics defined in model_config.yaml, skipping validation.")
        return

    output_groundtruth = json.loads(golden_values_content)

    if isinstance(output_groundtruth, str):
        # Handle JSONL output, assume only one line in this case.
        output_groundtruth = json.loads(output_groundtruth)

    output_current = json.loads(tensorboard_content)
    if isinstance(output_current, str):
        # Handle JSONL output, assume only one line in this case.
        output_current = json.loads(output_current)

    golden_metrics = metrics - {"prefix_cache_comparison", "memory_phase_leak"}
    if golden_metrics:
        assert len(output_groundtruth) > 0, "No golden-backed test performed for output"
    groundtruth_request_ids = set(output_groundtruth) - _NON_REQUEST_TOP_LEVEL_KEYS
    current_request_ids = set(output_current) - _NON_REQUEST_TOP_LEVEL_KEYS
    request_metric_names = {
        "generated_tokens",
        "logprobs",
        "prompt_logprobs",
        "generated_logprobs",
        "top_n_logprobs",
        "generated_top_n_logprobs",
        "prompt_top_n_logprobs",
        "generated_text",
        "routing_indices",
    }
    selected_request_metrics = metrics & request_metric_names
    if selected_request_metrics:
        assert groundtruth_request_ids.issuperset(current_request_ids), (
            "Some request IDs from groundtruth are missing in current or current has "
            "unexpected IDs: "
            f"{sorted(groundtruth_request_ids)} vs {sorted(current_request_ids)}"
        )
        if groundtruth_request_ids != current_request_ids:
            logger.warning(
                "Some request IDs from groundtruth are missing in output; only the "
                "current subset will be tested: %s vs %s",
                sorted(groundtruth_request_ids),
                sorted(current_request_ids),
            )

    # Throughput assertions.
    performed_any_comparison = False
    if "throughput" in output_groundtruth and "throughput" in metrics:
        # First warmup iteration is excluded from throughput statistics.
        current_throughput = output_current["throughput"]
        golden_throughput = output_groundtruth["throughput"]
        if isinstance(current_throughput, (list, tuple)) and len(current_throughput) > 1:
            current_throughput = current_throughput[1:]
        if isinstance(golden_throughput, (list, tuple)) and len(golden_throughput) > 1:
            golden_throughput = golden_throughput[1:]
        throughput_sampled = _median_as_float(current_throughput)
        throughput_golden = _median_as_float(golden_throughput)

        # 10% is empirically observed to be within hardware variance.
        assert throughput_sampled >= 0.9 * throughput_golden, (
            "Throughput is slower than expected! Expected to be within 10% of "
            f"~{throughput_golden} tok/s but benchmarked {output_current['throughput']} tok/s"
        )

        # If throughput is significantly improved (> 20%), update golden values accordingly.
        assert throughput_sampled < throughput_golden * 1.2, (
            f"Throughput has been improved from expected ~{throughput_golden} tok/s "
            f"to {output_current['throughput']} tok/s. Please update golden values "
            "in the functional tests."
        )
        performed_any_comparison = True

    # Peak memory is a one-sided regression bound. Improvements are accepted.
    memory_key = "mem-max-allocated-bytes"
    if memory_key in metrics:
        assert (
            memory_key in output_groundtruth
        ), "METRICS requests mem-max-allocated-bytes but the golden output omits it"
        assert memory_key in output_current, (
            "Golden values include mem-max-allocated-bytes but current output does "
            "not. Ensure the inference script records memory metrics."
        )
        sampled = _median_as_float(output_current[memory_key])
        golden = _median_as_float(output_groundtruth[memory_key])
        assert golden > 0, f"Golden mem_max_allocated_bytes must be > 0, got {golden}."
        regression_fraction = float(comparison_config.get("memory_regression_fraction", 0.05))
        regression_bytes = int(comparison_config.get("memory_regression_bytes", 0))
        high = golden * (1.0 + regression_fraction) + regression_bytes
        assert sampled <= high, (
            f"Memory regression for {memory_key}: allowed at most {high:.0f} bytes "
            f"({_bytes_to_gib(high):.3f} GiB) from golden {golden:.0f} bytes, "
            f"but got {sampled:.0f} bytes ({_bytes_to_gib(sampled):.3f} GiB)."
        )
        performed_any_comparison = True

    lptc_key = "lifetime_prefill_token_count"
    if lptc_key in metrics:
        assert lptc_key in output_groundtruth, f"Golden output is missing {lptc_key}"
        assert lptc_key in output_current, f"Current output is missing {lptc_key}"
        assert int(output_current[lptc_key]) == int(output_groundtruth[lptc_key]), (
            f"{lptc_key} differs: expected {output_groundtruth[lptc_key]}, "
            f"got {output_current[lptc_key]}"
        )
        performed_any_comparison = True

    if "prefix_cache_comparison" in metrics:
        assert (
            "prefix_cache_comparison" in output_current
        ), "Current output is missing the direct cache_off/cache_on comparison artifact"
        prefix_cache_payload = output_current["prefix_cache_comparison"]
        _compare_prefix_cache_phases(prefix_cache_payload, comparison_config)
        scenario_id = model_config.get("PREFIX_CACHE_SCENARIO")
        assert scenario_id, (
            "prefix_cache_comparison requires PREFIX_CACHE_SCENARIO so runtime "
            "evidence is checked against the committed coverage contract"
        )
        manifest = load_manifest(_PREFIX_CACHE_MANIFEST)
        assert_runtime_evidence(
            manifest,
            scenario_id,
            _prefix_cache_runtime_evidence(prefix_cache_payload, output_current),
        )
        performed_any_comparison = True

    if "memory_phase_leak" in metrics:
        assert (
            "prefix_cache_comparison" in output_current
        ), "memory_phase_leak requires prefix_cache_comparison output"
        _check_memory_phase_leak(output_current["prefix_cache_comparison"], comparison_config)
        performed_any_comparison = True

    golden_logprob_abs_tol = float(
        comparison_config.get("logprob_abs_tol", _GOLDEN_LOGPROB_ABS_TOL)
    )
    golden_logprob_rel_tol = float(comparison_config.get("logprob_rel_tol", 0.0))

    for request_id in sorted(current_request_ids if selected_request_metrics else []):
        groundtruth_results = output_groundtruth[request_id]
        current_results = output_current[request_id]

        if "generated_tokens" in groundtruth_results and "generated_tokens" in metrics:
            performed_any_comparison = True
            tokens_groundtruth = groundtruth_results["generated_tokens"]
            tokens_current = current_results["generated_tokens"]
            assert tokens_groundtruth == tokens_current, (
                f"Token mismatch:\nGround truth: {tokens_groundtruth}\n"
                f"Current: {tokens_current}"
            )

        if "logprobs" in groundtruth_results and "logprobs" in metrics:
            performed_any_comparison = True
            _assert_logprob_sequence(
                groundtruth_results["logprobs"],
                current_results["logprobs"],
                label=f"request[{request_id}].logprobs",
                abs_tol=golden_logprob_abs_tol,
                rel_tol=golden_logprob_rel_tol,
            )

        for field in ("prompt_logprobs", "generated_logprobs"):
            if field in groundtruth_results and field in metrics:
                performed_any_comparison = True
                _assert_logprob_sequence(
                    groundtruth_results[field],
                    current_results[field],
                    label=f"request[{request_id}].{field}",
                    abs_tol=golden_logprob_abs_tol,
                    rel_tol=golden_logprob_rel_tol,
                )

        top_n_fields = ("top_n_logprobs", "generated_top_n_logprobs", "prompt_top_n_logprobs")
        for field in top_n_fields:
            if field in groundtruth_results and field in metrics:
                performed_any_comparison = True
                _assert_top_n_sequence(
                    groundtruth_results[field],
                    current_results[field],
                    label=f"request[{request_id}].{field}",
                    abs_tol=golden_logprob_abs_tol,
                    rel_tol=golden_logprob_rel_tol,
                )

        if "generated_text" in groundtruth_results and "generated_text" in metrics:
            performed_any_comparison = True
            generated_text_groundtruth = groundtruth_results["generated_text"]
            generated_text_current = current_results["generated_text"]
            min_len = min(len(generated_text_groundtruth), len(generated_text_current))
            assert min_len > 0, (
                "Generated text mismatch:"
                f"\nGround truth: {generated_text_groundtruth}\nCurrent: {generated_text_current}"
            )
            assert generated_text_groundtruth[:min_len] == generated_text_current[:min_len], (
                "Generated text mismatch:"
                f"\nGround truth (truncated to {min_len} chars): {generated_text_groundtruth[:min_len]}"
                f"\nCurrent (truncated to {min_len} chars): {generated_text_current[:min_len]}"
            )

        if "routing_indices" in groundtruth_results and "routing_indices" in metrics:
            performed_any_comparison = True
            token_indices = groundtruth_results.get("routing_indices_token_indices")
            current_routing = np.array(current_results["routing_indices"])
            assert token_indices is not None
            current_routing = current_routing[token_indices]
            routing_indices_groundtruth = np.sort(
                np.array(groundtruth_results["routing_indices"]), axis=-1
            )
            routing_indices_current = np.sort(current_routing, axis=-1)
            assert np.array_equal(
                routing_indices_groundtruth, routing_indices_current
            ), f"Routing indices mismatch:\nGround truth: {routing_indices_groundtruth}\nCurrent: {routing_indices_current}"

    if not performed_any_comparison:
        raise AssertionError(
            f"No requested metric was compared. Selected metrics: {sorted(metrics)}"
        )
