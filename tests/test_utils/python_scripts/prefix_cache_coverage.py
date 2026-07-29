# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validation helpers for the prefix-cache execution-stress contract.

This module deliberately does not import Megatron or torch.  The coverage
contract is checked in lightweight CI, while the GPU tests named by the
contract run in their normal Slurm buckets.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from yaml.constructor import ConstructorError

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*$")
VALID_STATUSES = {"active"}
VALID_UNSUPPORTED_CATEGORIES = {"external", "unsupported"}
VALID_LAYERS = {"unit", "engine", "functional"}
VALID_TEST_KINDS = {"pytest", "functional"}
PAIR_OWNERSHIP_RULE = "first_listed_covering_scenario"
TOP_LEVEL_KEYS = {
    "schema_version",
    "contract",
    "observations",
    "output_fields",
    "oracles",
    "stress_profiles",
    "behaviors",
    "features",
    "matrices",
    "scenarios",
    "unsupported",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silently overwritten mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a coverage manifest, rejecting a non-mapping document."""

    data = yaml.load(
        path.read_text(),
        Loader=_UniqueKeyLoader,  # noqa: S506 - This loader inherits yaml.SafeLoader.
    )
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the document root")
    return data


def _is_id(value: Any) -> bool:
    return isinstance(value, str) and ID_PATTERN.fullmatch(value) is not None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _check_keys(
    errors: list[str], path: str, value: Any, required: set[str], allowed: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected a mapping")
        return {}
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        errors.append(f"{path}: missing keys {missing}")
    if unknown:
        errors.append(f"{path}: unknown keys {unknown}")
    return value


def _check_id_catalog(errors: list[str], name: str, catalog: Any) -> Mapping[str, Any]:
    if not isinstance(catalog, dict) or not catalog:
        errors.append(f"{name}: expected a non-empty mapping")
        return {}
    for item_id, description in catalog.items():
        if not _is_id(item_id):
            errors.append(f"{name}: invalid ID {item_id!r}")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{name}.{item_id}: expected a non-empty description")
    return catalog


def _check_threshold(
    errors: list[str], path: str, actual: Any, minimum: Any, *, runtime: bool = False
) -> None:
    label = "runtime observation" if runtime else "parameter"
    if isinstance(minimum, bool):
        if actual is not minimum:
            errors.append(f"{path}: {label} must be {minimum!r}, got {actual!r}")
    elif isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or actual < minimum:
            errors.append(f"{path}: {label} must be >= {minimum!r}, got {actual!r}")
    elif isinstance(minimum, list):
        if not isinstance(actual, list) or not set(minimum).issubset(actual):
            errors.append(f"{path}: {label} must contain {minimum!r}, got {actual!r}")
    elif actual != minimum:
        errors.append(f"{path}: {label} must equal {minimum!r}, got {actual!r}")


def _normalize_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _pair_key(axis_a: str, value_a: Any, axis_b: str, value_b: Any) -> tuple[str, str]:
    tokens = (f"{axis_a}.{value_a}", f"{axis_b}.{value_b}")
    return tuple(sorted(tokens))


def _excluded_pairs(matrix: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for exclusion in _as_list(matrix.get("exclusions")):
        values = _as_mapping(_as_mapping(exclusion).get("values"))
        if len(values) != 2:
            continue
        (axis_a, value_a), (axis_b, value_b) = values.items()
        result[_pair_key(axis_a, value_a, axis_b, value_b)] = str(
            _as_mapping(exclusion).get("reason", "")
        )
    return result


def required_pairs(matrix: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Expand all valid strength-2 obligations for one local matrix."""

    axes = _as_mapping(matrix.get("axes"))
    result: set[tuple[str, str]] = set()
    excluded = _excluded_pairs(matrix)
    for axis_a, axis_b in itertools.combinations(axes, 2):
        for value_a, value_b in itertools.product(_as_list(axes[axis_a]), _as_list(axes[axis_b])):
            pair = _pair_key(axis_a, value_a, axis_b, value_b)
            if pair not in excluded:
                result.add(pair)
    return result


def generated_matrix_cases(
    matrix_id: str, matrix: Mapping[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Build a deterministic concise strength-2 covering array.

    Case IDs contain the complete assignment rather than a row number, so an
    unrelated row insertion cannot silently retarget a coverage reference.
    """

    axes = _as_mapping(matrix.get("axes"))
    explicit_rows = matrix.get("rows")
    if explicit_rows is not None:
        if not isinstance(explicit_rows, list) or not explicit_rows:
            raise ValueError(f"{matrix_id}: rows must be a non-empty list")
        excluded = set(_excluded_pairs(matrix))
        required = required_pairs(matrix)
        covered: set[tuple[str, str]] = set()
        result = []
        seen_assignments: set[tuple[Any, ...]] = set()
        for row_index, raw_assignment in enumerate(explicit_rows):
            assignment = _as_mapping(raw_assignment)
            if set(assignment) != set(axes):
                raise ValueError(f"{matrix_id}: row {row_index} must assign exactly {sorted(axes)}")
            assignment_values = tuple(assignment[axis] for axis in axes)
            if assignment_values in seen_assignments:
                raise ValueError(f"{matrix_id}: row {row_index} duplicates an earlier row")
            seen_assignments.add(assignment_values)
            for axis, value in assignment.items():
                if value not in _as_list(axes[axis]):
                    raise ValueError(
                        f"{matrix_id}: row {row_index} uses unknown value {axis}.{value}"
                    )
            pairs = {
                _pair_key(axis_a, assignment[axis_a], axis_b, assignment[axis_b])
                for axis_a, axis_b in itertools.combinations(axes, 2)
            }
            invalid = sorted(pairs & excluded)
            if invalid:
                raise ValueError(
                    f"{matrix_id}: row {row_index} contains excluded pair "
                    f"{'+'.join(invalid[0])}"
                )
            covered.update(pairs)
            case_id = ".".join([matrix_id, *(str(assignment[axis]) for axis in axes)])
            result.append((case_id, dict(assignment)))
        missing = sorted(required - covered)
        if missing:
            preview = ", ".join("+".join(pair) for pair in missing[:8])
            raise ValueError(
                f"{matrix_id}: explicit rows miss {len(missing)} pair obligations ({preview})"
            )
        return result

    excluded = set(_excluded_pairs(matrix))
    required = required_pairs(matrix)
    candidates: list[tuple[tuple[Any, ...], dict[str, Any], set[tuple[str, str]]]] = []
    for assignment_values in itertools.product(*(_as_list(values) for values in axes.values())):
        assignment = dict(zip(axes, assignment_values))
        pairs = {
            _pair_key(axis_a, assignment[axis_a], axis_b, assignment[axis_b])
            for axis_a, axis_b in itertools.combinations(axes, 2)
        }
        if pairs & excluded:
            continue
        candidates.append((assignment_values, assignment, pairs))

    uncovered = set(required)
    selected: list[tuple[tuple[Any, ...], dict[str, Any], set[tuple[str, str]]]] = []
    while uncovered:
        gains = [
            (len(pairs & uncovered), values, assignment, pairs)
            for values, assignment, pairs in candidates
        ]
        best_gain = max((gain for gain, *_ in gains), default=0)
        if best_gain == 0:
            missing = ", ".join("+".join(pair) for pair in sorted(uncovered)[:8])
            raise ValueError(f"{matrix_id}: no valid assignment covers {missing}")
        _, values, assignment, pairs = min(
            (item for item in gains if item[0] == best_gain), key=lambda item: item[1]
        )
        selected.append((values, assignment, pairs))
        uncovered -= pairs

    # Remove rows that became redundant after later greedy selections.
    changed = True
    while changed:
        changed = False
        for index in range(len(selected) - 1, -1, -1):
            other_pairs = set().union(
                *(
                    pairs
                    for other_index, (_, _, pairs) in enumerate(selected)
                    if other_index != index
                )
            )
            if required.issubset(other_pairs):
                selected.pop(index)
                changed = True
                break

    result = []
    for _, assignment, _ in selected:
        case_id = ".".join([matrix_id, *(str(assignment[axis]) for axis in axes)])
        result.append((case_id, assignment))
    return result


def scenario_pairs(scenario: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Expand the pairs a scenario promises to execute, not merely configure."""

    values = _as_mapping(scenario.get("values"))
    result: set[tuple[str, str]] = set()
    for axis_a, axis_b in itertools.combinations(values, 2):
        for value_a, value_b in itertools.product(
            _normalize_values(values[axis_a]), _normalize_values(values[axis_b])
        ):
            result.add(_pair_key(axis_a, value_a, axis_b, value_b))
    return result


def pair_ownership(manifest: Mapping[str, Any], matrix_id: str) -> dict[tuple[str, str], str]:
    """Return the compact table's deterministic, single owner for every pair."""

    matrix = _as_mapping(_as_mapping(manifest.get("matrices")).get(matrix_id))
    scenarios = _as_mapping(manifest.get("scenarios"))
    ownership: dict[tuple[str, str], str] = {}
    excluded = set(_excluded_pairs(matrix))
    for scenario_id in _as_list(matrix.get("scenarios")):
        for pair in scenario_pairs(_as_mapping(scenarios.get(scenario_id))):
            if pair not in excluded:
                ownership.setdefault(pair, scenario_id)
    return ownership


def _test_ast_node(tree: ast.Module, node_ref: str) -> ast.AST | None:
    parts = [part.split("[", 1)[0] for part in node_ref.split("::") if part]
    if not parts:
        return None
    candidates: Iterable[ast.AST] = tree.body
    current: ast.AST | None = None
    for part in parts:
        current = next(
            (
                node
                for node in candidates
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if current is None:
            return None
        candidates = current.body if isinstance(current, ast.ClassDef) else ()
    return current


def _decorator_text(node: ast.AST) -> str:
    decorators = getattr(node, "decorator_list", ())
    return " ".join(ast.unparse(decorator) for decorator in decorators)


def _validate_active_test_ref(
    errors: list[str],
    scenario_id: str,
    test: Mapping[str, Any],
    repo_root: Path,
    matrix_id: str | None,
) -> None:
    path_value = test.get("path")
    if not isinstance(path_value, str) or not path_value:
        errors.append(f"scenarios.{scenario_id}.test.path: expected a non-empty path")
        return
    candidate = (repo_root / path_value).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        errors.append(f"scenarios.{scenario_id}.test.path: path escapes the repository")
        return
    if not candidate.is_file():
        errors.append(f"scenarios.{scenario_id}.test.path: active test file does not exist")
        return

    kind = test.get("kind")
    if kind == "pytest":
        node_ref = test.get("node")
        if not isinstance(node_ref, str) or not node_ref:
            errors.append(f"scenarios.{scenario_id}.test.node: expected a pytest node")
            return
        try:
            tree = ast.parse(candidate.read_text(), filename=str(candidate))
        except SyntaxError as exc:
            errors.append(f"scenarios.{scenario_id}.test.path: cannot parse Python: {exc}")
            return
        node = _test_ast_node(tree, node_ref)
        if node is None:
            errors.append(
                f"scenarios.{scenario_id}.test.node: {node_ref!r} was not found without imports"
            )
            return
        decorators = _decorator_text(node)
        parameter_match = re.search(r"\[([^\]]+)\]$", node_ref)
        if parameter_match and parameter_match.group(1) not in decorators:
            errors.append(
                f"scenarios.{scenario_id}.test.node: parameter ID "
                f"{parameter_match.group(1)!r} is absent from the test decorators"
            )
        if matrix_id is not None:
            node_text = ast.unparse(node)
            if "generated_matrix_cases" not in node_text or repr(matrix_id) not in node_text:
                errors.append(
                    f"scenarios.{scenario_id}.test.node: active matrix tests must obtain "
                    f"their rows from generated_matrix_cases({matrix_id!r}, ...)"
                )
        disabled_markers = ("skip", "xfail", "flaky", "flaky_in_dev")
        if any(marker in decorators for marker in disabled_markers):
            errors.append(
                f"scenarios.{scenario_id}.test.node: active coverage cannot be disabled "
                f"({decorators})"
            )
    elif kind == "functional":
        case = test.get("case")
        if not isinstance(case, str) or not case:
            errors.append(f"scenarios.{scenario_id}.test.case: expected a functional case ID")
            return
        contents = candidate.read_text()
        if case not in contents:
            errors.append(f"scenarios.{scenario_id}.test.case: {case!r} is absent from the recipe")
        if f"{case}-broken" in contents or "mr-github-broken" in contents:
            errors.append(
                f"scenarios.{scenario_id}.test.case: active functional coverage is broken"
            )


def _scenario_fingerprint(scenario: Mapping[str, Any]) -> tuple[Any, ...]:
    def freeze(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((key, freeze(item)) for key, item in value.items()))
        if isinstance(value, list):
            return tuple(sorted((freeze(item) for item in value), key=repr))
        return value

    return (
        scenario.get("status"),
        scenario.get("layer"),
        scenario.get("matrix"),
        freeze(scenario.get("values")),
        freeze(scenario.get("behaviors")),
        freeze(scenario.get("stress")),
        freeze(scenario.get("runtime_minimums")),
        freeze(scenario.get("oracles")),
        freeze(scenario.get("output")),
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Conservatively detect a scenario that makes another scenario redundant."""

    if (
        left.get("status") != right.get("status")
        or left.get("layer") != right.get("layer")
        or left.get("matrix") != right.get("matrix")
        or _as_mapping(left.get("stress")).get("profile")
        != _as_mapping(right.get("stress")).get("profile")
        or left.get("output") != right.get("output")
    ):
        return False

    left_values = {
        key: set(_normalize_values(value)) for key, value in _as_mapping(left.get("values")).items()
    }
    right_values = {
        key: set(_normalize_values(value))
        for key, value in _as_mapping(right.get("values")).items()
    }
    if set(left_values) != set(right_values):
        return False
    if not all(left_values[key].issuperset(right_values[key]) for key in left_values):
        return False
    if not set(_as_list(left.get("behaviors"))).issuperset(_as_list(right.get("behaviors"))):
        return False
    if not set(_as_list(left.get("oracles"))).issuperset(_as_list(right.get("oracles"))):
        return False

    left_runtime = _as_mapping(left.get("runtime_minimums"))
    right_runtime = _as_mapping(right.get("runtime_minimums"))
    if set(left_runtime) != set(right_runtime):
        return False
    for key, right_value in right_runtime.items():
        left_value = left_runtime[key]
        if isinstance(right_value, (int, float)) and not isinstance(right_value, bool):
            if not isinstance(left_value, (int, float)) or left_value < right_value:
                return False
        elif left_value != right_value:
            return False
    return _scenario_fingerprint(left) != _scenario_fingerprint(right)


def validate_manifest(manifest: Mapping[str, Any], repo_root: Path | None = None) -> list[str]:
    """Return every contract error instead of failing at the first one."""

    errors: list[str] = []
    if set(manifest) != TOP_LEVEL_KEYS:
        missing = sorted(TOP_LEVEL_KEYS - set(manifest))
        unknown = sorted(set(manifest) - TOP_LEVEL_KEYS)
        if missing:
            errors.append(f"manifest: missing top-level keys {missing}")
        if unknown:
            errors.append(f"manifest: unknown top-level keys {unknown}")
    if manifest.get("schema_version") != 1:
        errors.append("schema_version: expected 1")

    contract = _check_keys(
        errors,
        "contract",
        manifest.get("contract"),
        {
            "active_only_counts",
            "pair_ownership",
            "flag_only_coverage",
            "isolated_hit_coverage",
            "unsupported_coverage",
        },
        {
            "active_only_counts",
            "pair_ownership",
            "flag_only_coverage",
            "isolated_hit_coverage",
            "unsupported_coverage",
        },
    )
    if contract.get("active_only_counts") is not True:
        errors.append("contract.active_only_counts: must be true")
    if contract.get("pair_ownership") != PAIR_OWNERSHIP_RULE:
        errors.append(f"contract.pair_ownership: must be {PAIR_OWNERSHIP_RULE!r}")
    if contract.get("flag_only_coverage") != "zero":
        errors.append("contract.flag_only_coverage: must be 'zero'")
    if contract.get("isolated_hit_coverage") != "zero":
        errors.append("contract.isolated_hit_coverage: must be 'zero'")
    if contract.get("unsupported_coverage") != "zero":
        errors.append("contract.unsupported_coverage: must be 'zero'")

    observations = _check_id_catalog(errors, "observations", manifest.get("observations"))
    output_fields = _check_id_catalog(errors, "output_fields", manifest.get("output_fields"))
    oracles = _check_id_catalog(errors, "oracles", manifest.get("oracles"))

    profiles = _as_mapping(manifest.get("stress_profiles"))
    if not profiles:
        errors.append("stress_profiles: expected a non-empty mapping")
    for profile_id, raw_profile in profiles.items():
        if not _is_id(profile_id):
            errors.append(f"stress_profiles: invalid ID {profile_id!r}")
        profile = _check_keys(
            errors,
            f"stress_profiles.{profile_id}",
            raw_profile,
            {"description", "minimums", "runtime_minimums"},
            {"description", "minimums", "runtime_minimums"},
        )
        if (
            not isinstance(profile.get("description"), str)
            or not profile.get("description", "").strip()
        ):
            errors.append(f"stress_profiles.{profile_id}.description: must be non-empty")
        if not isinstance(profile.get("minimums"), dict) or not profile.get("minimums"):
            errors.append(f"stress_profiles.{profile_id}.minimums: must be non-empty")
        runtime_minimums = _as_mapping(profile.get("runtime_minimums"))
        if not runtime_minimums:
            errors.append(f"stress_profiles.{profile_id}.runtime_minimums: must be non-empty")
        for observation_id in runtime_minimums:
            if observation_id not in observations:
                errors.append(
                    f"stress_profiles.{profile_id}.runtime_minimums: unknown observation "
                    f"{observation_id!r}"
                )

    behaviors = _as_mapping(manifest.get("behaviors"))
    if not behaviors:
        errors.append("behaviors: expected a non-empty mapping")
    for behavior_id, raw_behavior in behaviors.items():
        if not _is_id(behavior_id):
            errors.append(f"behaviors: invalid ID {behavior_id!r}")
        behavior = _check_keys(
            errors,
            f"behaviors.{behavior_id}",
            raw_behavior,
            {"description", "status", "stress_profiles", "runtime_observations"},
            {"description", "status", "stress_profiles", "runtime_observations"},
        )
        if behavior.get("status") not in VALID_STATUSES:
            errors.append(f"behaviors.{behavior_id}.status: supported behaviors must be active")
        for profile_id in _as_list(behavior.get("stress_profiles")):
            if profile_id not in profiles:
                errors.append(
                    f"behaviors.{behavior_id}.stress_profiles: unknown profile {profile_id!r}"
                )
        if not _as_list(behavior.get("stress_profiles")):
            errors.append(f"behaviors.{behavior_id}.stress_profiles: must be non-empty")
        for observation_id in _as_list(behavior.get("runtime_observations")):
            if observation_id not in observations:
                errors.append(
                    f"behaviors.{behavior_id}.runtime_observations: unknown observation "
                    f"{observation_id!r}"
                )
        if not _as_list(behavior.get("runtime_observations")):
            errors.append(f"behaviors.{behavior_id}.runtime_observations: must be non-empty")

    features = _as_mapping(manifest.get("features"))
    if not features:
        errors.append("features: expected a non-empty mapping")
    for feature_id, raw_feature in features.items():
        if not _is_id(feature_id):
            errors.append(f"features: invalid ID {feature_id!r}")
        feature = _check_keys(
            errors,
            f"features.{feature_id}",
            raw_feature,
            {"description", "values"},
            {"description", "values"},
        )
        if (
            not isinstance(feature.get("description"), str)
            or not feature.get("description", "").strip()
        ):
            errors.append(f"features.{feature_id}.description: must be non-empty")
        values = _check_id_catalog(errors, f"features.{feature_id}.values", feature.get("values"))
        if len(values) < 2:
            errors.append(f"features.{feature_id}.values: expected at least two values")

    unsupported = _as_mapping(manifest.get("unsupported"))
    if not unsupported:
        errors.append("unsupported: expected a non-empty mapping")
    for gap_id, raw_gap in unsupported.items():
        if not _is_id(gap_id):
            errors.append(f"unsupported: invalid ID {gap_id!r}")
        gap = _check_keys(
            errors,
            f"unsupported.{gap_id}",
            raw_gap,
            {"category", "description", "reason", "values"},
            {"category", "description", "reason", "values", "tracking"},
        )
        if gap.get("category") not in VALID_UNSUPPORTED_CATEGORIES:
            errors.append(
                f"unsupported.{gap_id}.category: expected one of "
                f"{sorted(VALID_UNSUPPORTED_CATEGORIES)}"
            )
        for field in ("description", "reason"):
            if not isinstance(gap.get(field), str) or not gap.get(field, "").strip():
                errors.append(f"unsupported.{gap_id}.{field}: must be non-empty")
        tracking = gap.get("tracking")
        if tracking is not None and (not isinstance(tracking, str) or not tracking.strip()):
            errors.append(f"unsupported.{gap_id}.tracking: must be non-empty")
        values = _as_mapping(gap.get("values"))
        if not values:
            errors.append(f"unsupported.{gap_id}.values: must be non-empty")
        for feature_id, selected in values.items():
            if feature_id not in features:
                errors.append(f"unsupported.{gap_id}.values: unknown feature {feature_id!r}")
                continue
            known_values = _as_mapping(_as_mapping(features[feature_id]).get("values"))
            for value in _normalize_values(selected):
                if value not in known_values:
                    errors.append(
                        f"unsupported.{gap_id}.values: unknown value " f"{feature_id}.{value}"
                    )

    scenarios = _as_mapping(manifest.get("scenarios"))
    if not scenarios:
        errors.append("scenarios: expected a non-empty mapping")

    matrices = _as_mapping(manifest.get("matrices"))
    if not matrices:
        errors.append("matrices: expected a non-empty mapping")
    for matrix_id, raw_matrix in matrices.items():
        if not _is_id(matrix_id):
            errors.append(f"matrices: invalid ID {matrix_id!r}")
        matrix = _check_keys(
            errors,
            f"matrices.{matrix_id}",
            raw_matrix,
            {"description", "status", "axes", "exclusions", "scenarios"},
            {"description", "status", "axes", "exclusions", "scenarios", "rows"},
        )
        if matrix.get("status") not in VALID_STATUSES:
            errors.append(f"matrices.{matrix_id}.status: supported matrices must be active")
        axes = _as_mapping(matrix.get("axes"))
        if len(axes) < 2:
            errors.append(f"matrices.{matrix_id}.axes: expected at least two axes")
        for feature_id, allowed_values in axes.items():
            if feature_id not in features:
                errors.append(f"matrices.{matrix_id}.axes: unknown feature {feature_id!r}")
                continue
            if not isinstance(allowed_values, list) or not allowed_values:
                errors.append(f"matrices.{matrix_id}.axes.{feature_id}: expected a non-empty list")
                continue
            if len(allowed_values) != len(set(allowed_values)):
                errors.append(f"matrices.{matrix_id}.axes.{feature_id}: duplicate values")
            known_values = _as_mapping(_as_mapping(features[feature_id]).get("values"))
            for value in allowed_values:
                if value not in known_values:
                    errors.append(
                        f"matrices.{matrix_id}.axes.{feature_id}: unknown value {value!r}"
                    )
        seen_exclusions: set[tuple[str, str]] = set()
        for index, raw_exclusion in enumerate(_as_list(matrix.get("exclusions"))):
            exclusion = _check_keys(
                errors,
                f"matrices.{matrix_id}.exclusions[{index}]",
                raw_exclusion,
                {"values", "reason"},
                {"values", "reason", "test"},
            )
            values = _as_mapping(exclusion.get("values"))
            if len(values) != 2:
                errors.append(
                    f"matrices.{matrix_id}.exclusions[{index}].values: "
                    "pairwise exclusions require exactly two values"
                )
                continue
            for feature_id, value in values.items():
                if feature_id not in axes or value not in _as_list(axes.get(feature_id)):
                    errors.append(
                        f"matrices.{matrix_id}.exclusions[{index}].values: "
                        f"unknown matrix value {feature_id}.{value}"
                    )
            pair = _pair_key(*itertools.chain.from_iterable(values.items()))
            if pair in seen_exclusions:
                errors.append(f"matrices.{matrix_id}.exclusions[{index}]: duplicate exclusion")
            seen_exclusions.add(pair)
            if (
                not isinstance(exclusion.get("reason"), str)
                or not exclusion.get("reason", "").strip()
            ):
                errors.append(f"matrices.{matrix_id}.exclusions[{index}].reason: must be non-empty")
        matrix_scenarios = _as_list(matrix.get("scenarios"))
        if not matrix_scenarios:
            errors.append(f"matrices.{matrix_id}.scenarios: must be non-empty")
        if len(matrix_scenarios) != len(set(matrix_scenarios)):
            errors.append(f"matrices.{matrix_id}.scenarios: duplicate scenario IDs")
        for scenario_id in matrix_scenarios:
            if scenario_id not in scenarios:
                errors.append(f"matrices.{matrix_id}.scenarios: unknown scenario {scenario_id!r}")
        try:
            generated_count = len(generated_matrix_cases(matrix_id, matrix))
        except ValueError as exc:
            errors.append(f"matrices.{matrix_id}: {exc}")
        else:
            declared_count = sum(
                int(_as_mapping(scenarios.get(scenario_id)).get("planned_cases", 0))
                for scenario_id in matrix_scenarios
            )
            if declared_count != generated_count:
                errors.append(
                    f"matrices.{matrix_id}: concise covering array requires "
                    f"{generated_count} collected cases, but scenarios declare {declared_count}"
                )

    for scenario_id, raw_scenario in scenarios.items():
        if not _is_id(scenario_id):
            errors.append(f"scenarios: invalid ID {scenario_id!r}")
        scenario = _check_keys(
            errors,
            f"scenarios.{scenario_id}",
            raw_scenario,
            {
                "description",
                "status",
                "layer",
                "test",
                "matrix",
                "values",
                "behaviors",
                "stress",
                "runtime_minimums",
                "oracles",
                "output",
                "planned_cases",
            },
            {
                "description",
                "status",
                "layer",
                "test",
                "matrix",
                "values",
                "behaviors",
                "stress",
                "runtime_minimums",
                "oracles",
                "output",
                "planned_cases",
                "nonredundancy_reason",
            },
        )
        status = scenario.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"scenarios.{scenario_id}.status: supported scenarios must be active")
        if scenario.get("layer") not in VALID_LAYERS:
            errors.append(f"scenarios.{scenario_id}.layer: invalid layer")
        if (
            not isinstance(scenario.get("description"), str)
            or not scenario.get("description", "").strip()
        ):
            errors.append(f"scenarios.{scenario_id}.description: must be non-empty")
        planned_cases = scenario.get("planned_cases")
        if (
            not isinstance(planned_cases, int)
            or isinstance(planned_cases, bool)
            or planned_cases < 1
        ):
            errors.append(f"scenarios.{scenario_id}.planned_cases: must be a positive integer")
        reason = scenario.get("nonredundancy_reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            errors.append(f"scenarios.{scenario_id}.nonredundancy_reason: must be non-empty")

        test = _check_keys(
            errors,
            f"scenarios.{scenario_id}.test",
            scenario.get("test"),
            {"kind", "path"},
            {"kind", "path", "node", "case"},
        )
        if test.get("kind") not in VALID_TEST_KINDS:
            errors.append(f"scenarios.{scenario_id}.test.kind: invalid kind")
        if not isinstance(test.get("path"), str) or not test.get("path", "").strip():
            errors.append(f"scenarios.{scenario_id}.test.path: must be non-empty")
        if test.get("kind") == "pytest" and (
            not isinstance(test.get("node"), str) or not test.get("node", "").strip()
        ):
            errors.append(f"scenarios.{scenario_id}.test.node: must be non-empty")
        if test.get("kind") == "functional" and (
            not isinstance(test.get("case"), str) or not test.get("case", "").strip()
        ):
            errors.append(f"scenarios.{scenario_id}.test.case: must be non-empty")
        matrix_id = scenario.get("matrix")
        if matrix_id is not None and matrix_id not in matrices:
            errors.append(f"scenarios.{scenario_id}.matrix: unknown matrix {matrix_id!r}")
        if status == "active" and repo_root is not None:
            _validate_active_test_ref(errors, scenario_id, test, repo_root, matrix_id)
        values = _as_mapping(scenario.get("values"))
        if matrix_id in matrices:
            axes = _as_mapping(_as_mapping(matrices[matrix_id]).get("axes"))
            if set(values) != set(axes):
                errors.append(
                    f"scenarios.{scenario_id}.values: must specify exactly matrix axes "
                    f"{sorted(axes)}"
                )
            for feature_id, selected in values.items():
                for value in _normalize_values(selected):
                    if value not in _as_list(axes.get(feature_id)):
                        errors.append(
                            f"scenarios.{scenario_id}.values: unknown matrix value "
                            f"{feature_id}.{value}"
                        )
            if scenario_id not in _as_list(_as_mapping(matrices[matrix_id]).get("scenarios")):
                errors.append(
                    f"scenarios.{scenario_id}: matrix {matrix_id!r} does not list this scenario"
                )
        else:
            for feature_id, selected in values.items():
                if feature_id not in features:
                    errors.append(f"scenarios.{scenario_id}.values: unknown feature {feature_id!r}")
                    continue
                known_values = _as_mapping(_as_mapping(features[feature_id]).get("values"))
                for value in _normalize_values(selected):
                    if value not in known_values:
                        errors.append(
                            f"scenarios.{scenario_id}.values: unknown value "
                            f"{feature_id}.{value}"
                        )

        scenario_behaviors = _as_list(scenario.get("behaviors"))
        if not scenario_behaviors:
            errors.append(f"scenarios.{scenario_id}.behaviors: must be non-empty")
        for behavior_id in scenario_behaviors:
            if behavior_id not in behaviors:
                errors.append(
                    f"scenarios.{scenario_id}.behaviors: unknown behavior {behavior_id!r}"
                )

        stress = _check_keys(
            errors,
            f"scenarios.{scenario_id}.stress",
            scenario.get("stress"),
            {"profile", "parameters"},
            {"profile", "parameters"},
        )
        profile_id = stress.get("profile")
        if profile_id not in profiles:
            errors.append(f"scenarios.{scenario_id}.stress.profile: unknown profile {profile_id!r}")
        parameters = _as_mapping(stress.get("parameters"))
        if profile_id in profiles:
            minimums = _as_mapping(_as_mapping(profiles[profile_id]).get("minimums"))
            for parameter, minimum in minimums.items():
                if parameter not in parameters:
                    errors.append(
                        f"scenarios.{scenario_id}.stress.parameters: missing {parameter!r}"
                    )
                else:
                    _check_threshold(
                        errors,
                        f"scenarios.{scenario_id}.stress.parameters.{parameter}",
                        parameters[parameter],
                        minimum,
                    )
            for behavior_id in scenario_behaviors:
                behavior = _as_mapping(behaviors.get(behavior_id))
                if profile_id not in _as_list(behavior.get("stress_profiles")):
                    errors.append(
                        f"scenarios.{scenario_id}: profile {profile_id!r} cannot cover "
                        f"behavior {behavior_id!r}"
                    )

        runtime_minimums = _as_mapping(scenario.get("runtime_minimums"))
        if not runtime_minimums:
            errors.append(f"scenarios.{scenario_id}.runtime_minimums: must be non-empty")
        for observation_id in runtime_minimums:
            if observation_id not in observations:
                errors.append(
                    f"scenarios.{scenario_id}.runtime_minimums: unknown observation "
                    f"{observation_id!r}"
                )
        if profile_id in profiles:
            for observation_id, minimum in _as_mapping(
                _as_mapping(profiles[profile_id]).get("runtime_minimums")
            ).items():
                if observation_id not in runtime_minimums:
                    errors.append(
                        f"scenarios.{scenario_id}.runtime_minimums: missing profile "
                        f"observation {observation_id!r}"
                    )
                else:
                    _check_threshold(
                        errors,
                        f"scenarios.{scenario_id}.runtime_minimums.{observation_id}",
                        runtime_minimums[observation_id],
                        minimum,
                        runtime=True,
                    )
        for behavior_id in scenario_behaviors:
            for observation_id in _as_list(
                _as_mapping(behaviors.get(behavior_id)).get("runtime_observations")
            ):
                if observation_id not in runtime_minimums:
                    errors.append(
                        f"scenarios.{scenario_id}.runtime_minimums: behavior "
                        f"{behavior_id!r} requires {observation_id!r}"
                    )

        scenario_oracles = _as_list(scenario.get("oracles"))
        if not scenario_oracles:
            errors.append(f"scenarios.{scenario_id}.oracles: must be non-empty")
        for oracle_id in scenario_oracles:
            if oracle_id not in oracles:
                errors.append(f"scenarios.{scenario_id}.oracles: unknown oracle {oracle_id!r}")

        output = _check_keys(
            errors,
            f"scenarios.{scenario_id}.output",
            scenario.get("output"),
            {"produces", "cache_off_on", "fields"},
            {"produces", "cache_off_on", "fields"},
        )
        produces = output.get("produces")
        if not isinstance(produces, bool):
            errors.append(f"scenarios.{scenario_id}.output.produces: expected bool")
        if not isinstance(output.get("cache_off_on"), bool):
            errors.append(f"scenarios.{scenario_id}.output.cache_off_on: expected bool")
        fields = _as_list(output.get("fields"))
        for field_id in fields:
            if field_id not in output_fields:
                errors.append(f"scenarios.{scenario_id}.output.fields: unknown field {field_id!r}")
        if produces and (output.get("cache_off_on") is not True or not fields):
            errors.append(
                f"scenarios.{scenario_id}.output: output-producing scenarios require "
                "cache-off/cache-on comparison fields"
            )
        if not produces and (output.get("cache_off_on") is not False or fields):
            errors.append(
                f"scenarios.{scenario_id}.output: non-output scenarios cannot claim "
                "output comparison"
            )
        if produces and "cache_off_on_comparisons" not in runtime_minimums:
            errors.append(
                f"scenarios.{scenario_id}.runtime_minimums: output scenarios require "
                "'cache_off_on_comparisons'"
            )

    # Keep every catalog closed: an entry must be exercised by supported
    # coverage or explicitly listed as an unsupported/external gap.
    used_profiles = {
        _as_mapping(_as_mapping(scenario).get("stress")).get("profile")
        for scenario in scenarios.values()
    }
    for profile_id in profiles:
        if profile_id not in used_profiles:
            errors.append(f"stress_profiles.{profile_id}: no active scenario uses this profile")

    used_observations: set[str] = set()
    for profile in profiles.values():
        used_observations.update(_as_mapping(_as_mapping(profile).get("runtime_minimums")))
    for behavior in behaviors.values():
        used_observations.update(_as_list(_as_mapping(behavior).get("runtime_observations")))
    for scenario in scenarios.values():
        used_observations.update(_as_mapping(_as_mapping(scenario).get("runtime_minimums")))
    for observation_id in observations:
        if observation_id not in used_observations:
            errors.append(f"observations.{observation_id}: unused catalog entry")

    used_oracles = {
        oracle_id
        for scenario in scenarios.values()
        for oracle_id in _as_list(_as_mapping(scenario).get("oracles"))
    }
    for oracle_id in oracles:
        if oracle_id not in used_oracles:
            errors.append(f"oracles.{oracle_id}: unused catalog entry")

    used_output_fields = {
        field_id
        for scenario in scenarios.values()
        for field_id in _as_list(_as_mapping(_as_mapping(scenario).get("output")).get("fields"))
    }
    for field_id in output_fields:
        if field_id not in used_output_fields:
            errors.append(f"output_fields.{field_id}: unused catalog entry")

    supported_feature_values: dict[str, set[Any]] = {feature_id: set() for feature_id in features}
    for matrix in matrices.values():
        for feature_id, selected in _as_mapping(_as_mapping(matrix).get("axes")).items():
            supported_feature_values.setdefault(feature_id, set()).update(_as_list(selected))
    for scenario in scenarios.values():
        for feature_id, selected in _as_mapping(_as_mapping(scenario).get("values")).items():
            supported_feature_values.setdefault(feature_id, set()).update(
                _normalize_values(selected)
            )
    unsupported_feature_values: dict[str, set[Any]] = {feature_id: set() for feature_id in features}
    for gap in unsupported.values():
        for feature_id, selected in _as_mapping(_as_mapping(gap).get("values")).items():
            unsupported_feature_values.setdefault(feature_id, set()).update(
                _normalize_values(selected)
            )
    for feature_id, feature in features.items():
        known_values = set(_as_mapping(_as_mapping(feature).get("values")))
        unaccounted = sorted(
            known_values
            - supported_feature_values.get(feature_id, set())
            - unsupported_feature_values.get(feature_id, set())
        )
        if unaccounted:
            errors.append(
                f"features.{feature_id}: values are neither actively covered nor "
                f"unsupported/external {unaccounted}"
            )

    # Every supported behavior must be owned by a concrete active stress scenario.
    for behavior_id, behavior in behaviors.items():
        owners = [
            scenario_id
            for scenario_id, scenario in scenarios.items()
            if behavior_id in _as_list(_as_mapping(scenario).get("behaviors"))
        ]
        if not owners:
            errors.append(f"behaviors.{behavior_id}: no active stress owner")
            continue
        if any(_as_mapping(scenarios[owner]).get("status") != "active" for owner in owners):
            errors.append(f"behaviors.{behavior_id}: every supported owner must be active")

    # Every supported local pair must have a concrete active owner.
    for matrix_id, matrix in matrices.items():
        required = required_pairs(_as_mapping(matrix))
        ownership = pair_ownership(manifest, matrix_id)
        missing = sorted(required - set(ownership))
        if missing:
            preview = ", ".join("+".join(pair) for pair in missing[:8])
            errors.append(
                f"matrices.{matrix_id}: {len(missing)} pair obligations lack an active "
                f"owner ({preview})"
            )
        inactive = [
            pair
            for pair, owner in ownership.items()
            if _as_mapping(scenarios.get(owner)).get("status") != "active"
        ]
        if inactive:
            errors.append(f"matrices.{matrix_id}: {len(inactive)} pairs lack an active owner")

    fingerprints: dict[tuple[Any, ...], str] = {}
    for scenario_id, scenario in scenarios.items():
        fingerprint = _scenario_fingerprint(_as_mapping(scenario))
        previous = fingerprints.get(fingerprint)
        if previous is not None:
            errors.append(f"scenarios.{scenario_id}: duplicates scenario {previous!r} exactly")
        else:
            fingerprints[fingerprint] = scenario_id

    scenario_items = list(scenarios.items())
    for (left_id, left), (right_id, right) in itertools.permutations(scenario_items, 2):
        if _dominates(_as_mapping(left), _as_mapping(right)) and not _as_mapping(right).get(
            "nonredundancy_reason"
        ):
            errors.append(
                f"scenarios.{right_id}: dominated by {left_id!r}; merge it or provide "
                "a nonredundancy_reason"
            )

    return errors


def coverage_summary(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Summarize concrete supported coverage and separately visible gaps."""

    scenarios = _as_mapping(manifest.get("scenarios"))
    matrices = _as_mapping(manifest.get("matrices"))
    behaviors = _as_mapping(manifest.get("behaviors"))
    supported_pairs = 0
    for matrix_id, matrix in matrices.items():
        ownership = pair_ownership(manifest, matrix_id)
        supported_pairs += len(ownership)
    return {
        "supported_scenarios": len(scenarios),
        "supported_behaviors": len(behaviors),
        "supported_matrices": len(matrices),
        "supported_pairs": supported_pairs,
        "collected_cases": sum(
            int(_as_mapping(scenario).get("planned_cases", 0))
            for scenario in scenarios.values()
            if _as_mapping(scenario).get("layer") != "functional"
        ),
        "functional_jobs": sum(
            1
            for scenario in scenarios.values()
            if _as_mapping(scenario).get("layer") == "functional"
        ),
        "functional_cases": sum(
            int(_as_mapping(scenario).get("planned_cases", 0))
            for scenario in scenarios.values()
            if _as_mapping(scenario).get("layer") == "functional"
        ),
        "unsupported_entries": len(_as_mapping(manifest.get("unsupported"))),
    }


def assert_runtime_evidence(
    manifest: Mapping[str, Any], scenario_id: str, observations: Mapping[str, Any]
) -> None:
    """Assert the runtime events required by one declared stress scenario."""

    scenarios = _as_mapping(manifest.get("scenarios"))
    if scenario_id not in scenarios:
        raise AssertionError(f"unknown prefix-cache scenario {scenario_id!r}")
    scenario = _as_mapping(scenarios[scenario_id])
    profile_id = _as_mapping(scenario.get("stress")).get("profile")
    profile = _as_mapping(_as_mapping(manifest.get("stress_profiles")).get(profile_id))
    required = dict(_as_mapping(profile.get("runtime_minimums")))
    for observation_id, minimum in _as_mapping(scenario.get("runtime_minimums")).items():
        current = required.get(observation_id)
        if (
            isinstance(current, (int, float))
            and not isinstance(current, bool)
            and isinstance(minimum, (int, float))
            and not isinstance(minimum, bool)
        ):
            required[observation_id] = max(current, minimum)
        else:
            required[observation_id] = minimum

    errors: list[str] = []
    unknown = sorted(set(observations) - set(_as_mapping(manifest.get("observations"))))
    if unknown:
        errors.append(f"unknown runtime observations {unknown}")
    for observation_id, minimum in required.items():
        if observation_id not in observations:
            errors.append(f"missing runtime observation {observation_id!r}")
            continue
        _check_threshold(
            errors, observation_id, observations[observation_id], minimum, runtime=True
        )
    if errors:
        raise AssertionError(f"{scenario_id}: " + "; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    errors = validate_manifest(manifest, args.repo_root)
    if errors:
        print("\n".join(f"- {error}" for error in errors))
        return 1
    summary = coverage_summary(manifest)
    print("prefix-cache coverage contract is valid")
    for name, value in summary.items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
