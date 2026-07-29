# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lightweight meta-tests for the prefix-cache execution-stress contract."""

import copy
from pathlib import Path

import pytest

from tests.test_utils.python_scripts.prefix_cache_coverage import (
    assert_runtime_evidence,
    coverage_summary,
    generated_matrix_cases,
    load_manifest,
    pair_ownership,
    required_pairs,
    scenario_pairs,
    validate_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/test_utils/prefix_cache_coverage.yaml"


def test_prefix_cache_coverage_contract_is_valid():
    """Check catalogs, stress thresholds, ownership, redundancy, and test references."""

    manifest = load_manifest(MANIFEST_PATH)
    errors = validate_manifest(manifest, REPO_ROOT)

    assert not errors, "\n".join(errors)


def test_every_supported_pair_has_one_stable_active_owner():
    """Each local pair resolves to the first declared active scenario that executes it."""

    manifest = load_manifest(MANIFEST_PATH)

    for matrix_id, matrix in manifest["matrices"].items():
        ownership = pair_ownership(manifest, matrix_id)
        cases = generated_matrix_cases(matrix_id, matrix)
        declared_cases = sum(
            manifest["scenarios"][scenario_id]["planned_cases"]
            for scenario_id in matrix["scenarios"]
        )

        assert set(ownership) == required_pairs(matrix)
        for pair, owner in ownership.items():
            covering_scenarios = [
                scenario_id
                for scenario_id in matrix["scenarios"]
                if pair in scenario_pairs(manifest["scenarios"][scenario_id])
            ]
            assert covering_scenarios
            assert owner == covering_scenarios[0]
            assert manifest["scenarios"][owner]["status"] == "active"
        assert len({case_id for case_id, _ in cases}) == len(cases)
        assert declared_cases == len(cases)


def test_supported_catalog_contains_only_concrete_active_coverage():
    """Planning records and count-credit shims cannot inflate supported coverage."""

    manifest = load_manifest(MANIFEST_PATH)
    summary = coverage_summary(manifest)

    assert all(behavior["status"] == "active" for behavior in manifest["behaviors"].values())
    assert all(matrix["status"] == "active" for matrix in manifest["matrices"].values())
    assert all(scenario["status"] == "active" for scenario in manifest["scenarios"].values())
    assert all(
        "counts_toward_design" not in scenario for scenario in manifest["scenarios"].values()
    )
    assert summary["supported_scenarios"] == len(manifest["scenarios"])
    assert summary["supported_behaviors"] == len(manifest["behaviors"])
    assert summary["supported_matrices"] == len(manifest["matrices"])


def test_every_supported_behavior_and_matrix_is_completely_owned():
    """Supported means every behavior and every valid local pair has an active owner."""

    manifest = load_manifest(MANIFEST_PATH)

    for behavior_id in manifest["behaviors"]:
        owners = [
            scenario_id
            for scenario_id, scenario in manifest["scenarios"].items()
            if behavior_id in scenario["behaviors"]
        ]
        assert owners
        assert all(manifest["scenarios"][owner]["status"] == "active" for owner in owners)

    for matrix_id, matrix in manifest["matrices"].items():
        assert set(pair_ownership(manifest, matrix_id)) == required_pairs(matrix)


def test_every_output_producing_scenario_compares_cache_off_and_on():
    """No generated token, logprob, text, status, or routing claim is cache-on-only."""

    manifest = load_manifest(MANIFEST_PATH)

    for scenario_id, scenario in manifest["scenarios"].items():
        output = scenario["output"]
        if output["produces"]:
            assert output["cache_off_on"] is True, scenario_id
            assert output["fields"], scenario_id
            assert scenario["runtime_minimums"]["cache_off_on_comparisons"] >= 1


def test_unsupported_and_external_gaps_are_visible_but_earn_no_coverage():
    """Known gaps remain explicit and cannot inflate supported counts or pair ownership."""

    manifest = load_manifest(MANIFEST_PATH)
    required_gaps = {
        "async_scheduler",
        "below_one_mamba_slot",
        "http_api",
        "mla_layout",
        "request_cancellation",
        "rl_refit",
        "uvm_backing",
    }
    assert required_gaps.issubset(manifest["unsupported"])
    assert {gap["category"] for gap in manifest["unsupported"].values()} == {
        "external",
        "unsupported",
    }

    summary = coverage_summary(manifest)
    without_gaps = copy.deepcopy(manifest)
    without_gaps["unsupported"] = {}
    without_gap_summary = coverage_summary(without_gaps)
    assert summary["unsupported_entries"] == len(manifest["unsupported"])
    assert {key: value for key, value in summary.items() if key != "unsupported_entries"} == {
        key: value for key, value in without_gap_summary.items() if key != "unsupported_entries"
    }


def test_validator_rejects_duplicate_supported_scenarios():
    """Exact duplicates cannot hide behind a second test name or a planning label."""

    manifest = load_manifest(MANIFEST_PATH)
    duplicated = copy.deepcopy(manifest)
    duplicated["scenarios"]["pc.unit.hash_lifecycle_duplicate"] = copy.deepcopy(
        duplicated["scenarios"]["pc.unit.hash_lifecycle"]
    )

    errors = validate_manifest(duplicated, REPO_ROOT)
    assert any("duplicates scenario 'pc.unit.hash_lifecycle' exactly" in error for error in errors)


def test_validator_requires_a_reason_for_a_subsumed_scenario():
    """A broader stress owner cannot silently make a narrower scenario redundant."""

    manifest = load_manifest(MANIFEST_PATH)
    subsumed = copy.deepcopy(manifest)
    broad_id = "pc.active.hybrid_immutability_broad"
    narrow_id = "pc.active.hybrid_immutability_serial"
    subsumed["scenarios"][broad_id] = copy.deepcopy(subsumed["scenarios"][narrow_id])
    subsumed["scenarios"][broad_id]["values"]["arrival_pattern"] = ["serial", "staggered"]

    errors = validate_manifest(subsumed, REPO_ROOT)
    assert any(
        error.startswith(f"scenarios.{narrow_id}: dominated by {broad_id!r}") for error in errors
    )

    subsumed["scenarios"][narrow_id][
        "nonredundancy_reason"
    ] = "The narrow row isolates serial-release behavior."
    errors = validate_manifest(subsumed, REPO_ROOT)
    assert not any(
        error.startswith(f"scenarios.{narrow_id}: dominated by {broad_id!r}") for error in errors
    )


def test_runtime_contract_rejects_missing_stress_evidence():
    """Passing a configuration flag or isolated hit cannot satisfy a stress profile."""

    manifest = load_manifest(MANIFEST_PATH)
    scenario_id = "pc.active.hybrid_immutability_concurrent"
    observed = {
        "concurrent_followers": 4,
        "producer_follower_cycles": 3,
        "state_checksum_checks": 3,
    }

    assert_runtime_evidence(manifest, scenario_id, observed)
    with pytest.raises(AssertionError, match="state_checksum_checks"):
        assert_runtime_evidence(
            manifest, scenario_id, {"concurrent_followers": 4, "producer_follower_cycles": 3}
        )
    with pytest.raises(AssertionError, match="unknown runtime observations"):
        assert_runtime_evidence(manifest, scenario_id, {**observed, "enable_prefix_caching": True})
