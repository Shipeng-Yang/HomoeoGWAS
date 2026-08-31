"""Deterministic replicate-to-table aggregation for the v2.0.1 benchmark.

This module deliberately treats replicate JSON as evidence, not as a convenient
cache.  It validates the design lock and registry before it emits any table and
never edits either inputs or immutable shards.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .comparators import METHOD_NAMES
from .contracts import canonical_json, sha256_payload
from .scenarios import build_scenarios

MASTER_SEED = 20260830
ACCEPTANCE_RULES = {
    "wilson_level": 0.95,
    "core_fwer_upper_wilson_max": 0.075,
    "core_failure_rate_max": 0.01,
    "stress_contributes_to_overall": False,
    "pilot_inference_status": "noninferential_do_not_threshold",
}


class BenchmarkAggregateError(RuntimeError):
    """Raised when immutable benchmark evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class RegistryRow:
    scenario_id: str
    track: str
    stage: str
    replicates: int
    bootstrap_B: int
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedEvidence:
    root: Path
    stage: str
    design_lock: Mapping[str, Any]
    design_hash: str
    registry_sha256: str
    registry: tuple[RegistryRow, ...]
    shards: tuple[tuple[Path, Mapping[str, Any]], ...]


@dataclass(frozen=True)
class AggregateResult:
    table_paths: tuple[Path, ...]
    table_sha256: Mapping[str, str]
    evidence: LoadedEvidence


_COMMON = (
    "stage", "inference_status", "track", "scenario_id", "replicate",
    "design_hash", "request_hash", "context_fingerprint", "failed",
    "failure_type", "runtime_seconds",
)
_ROBUSTNESS_METHODS = ("omnib", "minor_burden", "pc1", "kernel_hadamard")

TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "fit_pve_recovery.tsv": _COMMON + (
        "panel", "allocation", "subgenome", "true_pve", "estimated_pve",
        "bias", "absolute_error", "squared_error", "replicate_rmse",
        "replicate_spearman", "optimizer_status", "dominant_correct", "boundary_fit",
    ),
    "fit_pve_coverage.tsv": _COMMON + (
        "panel", "allocation", "subgenome", "target_pve", "interval_low", "interval_high",
        "covered", "interval_width", "bootstrap_failures", "successes", "total",
        "estimate", "ci_low", "ci_high", "failures",
    ),
    "fit_scan_metrics.tsv": _COMMON + (
        "panel", "scan_arm", "method", "scan_pve", "placement", "minimum_p",
        "rejected", "causal_detected", "lead_distance", "localization_correct",
        "distance_unit", "lambda_gc", "successes", "total", "estimate", "ci_low",
        "ci_high", "failures",
    ),
    "omnib_null_replicates.tsv": _COMMON + (
        "method", "null_model", "stress", "bank_role", "response_index",
        "minimum_p", "threshold", "rejected", "family_size", "family_hash",
        "family_order_hash", "score_family_size", "tested_family_size",
        "tested_family_hash", "response_seed_id", "calibration_seed_id",
        "successes", "total", "estimate", "ci_low", "ci_high",
        "failures",
    ),
    "omnib_power_replicates.tsv": _COMMON + (
        "method", "architecture", "control_type", "interaction_pve", "causal_groups",
        "response_index", "detected", "false_positive", "specificity", "recall", "threshold", "family_hash",
        "calibration_response_hash", "target_response_hash",
        "null_calibration_gate", "eligible_for_power_summary", "successes", "total",
        "estimate", "ci_low", "ci_high", "failures",
    ),
    "omnib_encoding_robustness.tsv": _COMMON + (
        "perturbation", "check_kind", "required", "status", "architecture", "method",
        "observed_arrays_identical", "adjusted_decisions_identical",
        "ranking_hash_identical", "rejection_sets_identical", "rank_correlation",
        "top_k_jaccard", "non_estimable_rate", "absolute_power_regret",
        "fwer_successes", "fwer_total", "fwer_estimate", "fwer_ci_low",
        "fwer_ci_high", "fwer_failures", "power_successes", "power_total",
        "power_estimate", "power_ci_low", "power_ci_high", "power_failures",
        "qa_fwer_successes", "qa_fwer_total", "qa_fwer_estimate",
        "qa_fwer_ci_low", "qa_fwer_ci_high", "qa_fwer_failures",
        "qa_power_successes", "qa_power_total", "qa_power_estimate",
        "qa_power_ci_low", "qa_power_ci_high", "qa_power_failures",
        "non_estimable_count", "non_estimable_total", "realized_marker_design",
    ),
    "omnib_family_manifest.tsv": _COMMON + (
        "method", "hypothesis_unit", "family_scope", "family_size",
        "ordered_ids", "family_hash", "family_order_hash", "tested_family_hash",
        "callable", "requested_jobs", "effective_jobs", "backend", "worker_pids",
    ),
    "scaling_runs.tsv": _COMMON + (
        "anchor", "jobs", "repeat", "effective_jobs", "backend", "worker_pids",
        "wall_seconds", "cpu_seconds", "aggregate_cpu_percent",
        "peak_aggregate_pss_bytes", "peak_aggregate_rss_bytes", "result_sha256",
        "family_sha256", "ranking_sha256", "exact_result_identity",
        "exact_family_identity", "exact_ranking_identity", "speedup_vs_serial",
        "parallel_efficiency", "parallel_comparable", "parallel_exclusion_reasons",
        "native_thread_contract_valid", "parallel_execution_contract_valid",
    ),
    "cross_species_application.tsv": _COMMON + (
        "analysis_id", "species", "panel", "ploidy", "subgenomes", "sample_count",
        "marker_count", "group_family_count", "edge_family_count", "primary_unit",
        "requested_jobs", "effective_jobs", "backend", "worker_pids",
        "calibration_method", "calibration_B", "adjusted_discovery_count",
        "adjusted_discoveries", "negative_result", "component_driver_distribution",
        "audit_status", "family_hash", "limitations", "status", "repair_required",
        "repair_reason",
    ),
    "benchmark_acceptance.tsv": (
        "stage", "inference_status", "gate_id", "track", "scenario_id",
        "gate_kind", "successes", "total", "estimate", "ci_low", "ci_high",
        "failures", "failure_rate", "passed", "status", "evidence_path", "reason",
    ),
}

_PILOT_FORMAL_FIELDS = {
    "fit_pve_coverage.tsv": {"covered"},
    "fit_scan_metrics.tsv": {"rejected", "causal_detected"},
    "omnib_null_replicates.tsv": {"threshold", "rejected"},
    "omnib_power_replicates.tsv": {
        "detected", "false_positive", "specificity", "recall", "threshold",
        "null_calibration_gate", "eligible_for_power_summary",
    },
    "benchmark_acceptance.tsv": {"passed"},
    "omnib_encoding_robustness.tsv": {
        "fwer_successes", "fwer_total", "fwer_estimate", "fwer_ci_low",
        "fwer_ci_high", "fwer_failures", "power_successes", "power_total",
        "power_estimate", "power_ci_low", "power_ci_high", "power_failures",
    },
}

_FORMAL_QA_FIELDS = {
    "omnib_encoding_robustness.tsv": {
        "qa_fwer_successes", "qa_fwer_total", "qa_fwer_estimate",
        "qa_fwer_ci_low", "qa_fwer_ci_high", "qa_fwer_failures",
        "qa_power_successes", "qa_power_total", "qa_power_estimate",
        "qa_power_ci_low", "qa_power_ci_high", "qa_power_failures",
    },
}


def table_schemas(stage: str) -> dict[str, tuple[str, ...]]:
    """Return stage-appropriate table headers without pilot inference fields."""

    if stage not in {"pilot", "formal"}:
        raise ValueError("table stage must be pilot or formal")
    if stage == "formal":
        return {
            name: tuple(
                field for field in fields
                if field not in _FORMAL_QA_FIELDS.get(name, set())
            )
            for name, fields in TABLE_SCHEMAS.items()
        }
    return {
        name: tuple(
            field for field in fields
            if field not in _PILOT_FORMAL_FIELDS.get(name, set())
        )
        for name, fields in TABLE_SCHEMAS.items()
    }

_HEX = set("0123456789abcdef")
_REGISTRY_HEADER = (
    "scenario_id", "track", "stage", "replicates", "bootstrap_B", "parameters",
)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        encoded = path.read_text(encoding="utf-8")
        value = json.loads(
            encoded, parse_constant=reject, object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise BenchmarkAggregateError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise BenchmarkAggregateError(f"JSON evidence is not an object: {path}")
    if encoded != canonical_json(value) + "\n":
        raise BenchmarkAggregateError(f"JSON evidence is not canonically encoded: {path}")
    return value


def _sha256_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        raise BenchmarkAggregateError(f"invalid {label}")
    return value


def _read_registry(path: Path) -> tuple[RegistryRow, ...]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != _REGISTRY_HEADER:
                raise BenchmarkAggregateError("scenario registry schema mismatch")
            raw_rows = list(reader)
    except OSError as error:
        raise BenchmarkAggregateError("scenario registry is missing or unreadable") from error
    rows: list[RegistryRow] = []
    for raw in raw_rows:
        def reject_constant(value: str) -> None:
            raise ValueError(value)

        def unique_parameters(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate registry parameter: {key}")
                value[key] = item
            return value

        try:
            parameters = json.loads(
                raw["parameters"], parse_constant=reject_constant,
                object_pairs_hook=unique_parameters,
            )
            replicates = int(raw["replicates"])
            bootstrap_b = int(raw["bootstrap_B"])
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise BenchmarkAggregateError("invalid scenario registry row") from error
        if (
            not raw["scenario_id"] or raw["track"] not in {"fit", "omnib", "scaling", "application"}
            or raw["stage"] not in {"pilot", "formal"} or replicates < 1
            or bootstrap_b < 0 or not isinstance(parameters, dict)
            or raw["parameters"] != canonical_json(parameters)
        ):
            raise BenchmarkAggregateError("invalid scenario registry row")
        rows.append(RegistryRow(
            raw["scenario_id"], raw["track"], raw["stage"], replicates,
            bootstrap_b, parameters,
        ))
    ids = [row.scenario_id for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise BenchmarkAggregateError("scenario registry is empty or has duplicate IDs")
    stages = {row.stage for row in rows}
    if len(stages) != 1:
        raise BenchmarkAggregateError("scenario registry mixes stages")
    return tuple(rows)


def _expected_replicates(row: RegistryRow) -> range:
    experiment = row.parameters.get("experiment")
    if row.track in {"scaling", "application"} or experiment in {
        "conditional", "global_vc", "family_size",
    }:
        return range(1)
    return range(row.replicates)


def _file_sha256(path: Path, label: str) -> str:
    try:
        if not path.is_file() or path.is_symlink():
            raise OSError("not a regular file")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise BenchmarkAggregateError(f"{label} is missing or unreadable") from error


def _uses_internal_symlink(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return any(
        (root / Path(*relative.parts[:index])).is_symlink()
        for index in range(1, len(relative.parts) + 1)
    )


def _validate_config_manifest(root: Path, path: Path) -> dict[str, str]:
    if _uses_internal_symlink(root, path):
        raise BenchmarkAggregateError("config manifest must not be a symlink")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["path", "sha256"]:
                raise BenchmarkAggregateError("config manifest schema mismatch")
            rows = list(reader)
    except OSError as error:
        raise BenchmarkAggregateError("config manifest is missing or unreadable") from error
    if not rows:
        raise BenchmarkAggregateError("config manifest is empty")
    hashes: dict[str, str] = {}
    for row in rows:
        relative = row["path"]
        if (
            not relative
            or relative in hashes
            or Path(relative).suffix.lower() not in {".yaml", ".yml"}
        ):
            raise BenchmarkAggregateError("config manifest has invalid or duplicate paths")
        unresolved = root / relative
        if _uses_internal_symlink(root, unresolved) or not unresolved.is_file():
            raise BenchmarkAggregateError("declared config is not a regular file")
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise BenchmarkAggregateError("config manifest path escapes benchmark root") from error
        digest = _sha256_hex(row["sha256"], "config hash")
        if _file_sha256(candidate, "declared config") != digest:
            raise BenchmarkAggregateError("declared config hash mismatch")
        try:
            parsed = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise BenchmarkAggregateError("declared config YAML is invalid") from error
        if not isinstance(parsed, Mapping) or not parsed:
            raise BenchmarkAggregateError(
                "declared config YAML must be a nonempty mapping"
            )
        interact = parsed.get("interact")
        if interact is not None:
            exact = {
                "mode": "group", "statistic": "omniB",
                "hypothesis_unit": "group", "subset_order": 2,
                "family_scope": "primary_only", "primary_transform": "INT",
                "primary_multiplicity": "bootstrap_minp",
            }
            subgenomes = interact.get("subgenomes") if isinstance(interact, Mapping) else None
            calibration = interact.get("calibration") if isinstance(interact, Mapping) else None
            burden = interact.get("burden") if isinstance(interact, Mapping) else None
            path_fields = ("groups", "phenotype", "sample_col", "trait")
            if (
                not isinstance(interact, Mapping)
                or any(interact.get(field) != value for field, value in exact.items())
                or not isinstance(subgenomes, list) or len(subgenomes) not in {2, 3, 4}
                or len(set(subgenomes)) != len(subgenomes)
                or any(not isinstance(interact.get(field), str) or not interact[field]
                       for field in path_fields)
                or any(
                    not isinstance(interact.get(field), Mapping)
                    or set(interact[field]) != set(subgenomes)
                    or any(not isinstance(value, str) or not value
                           for value in interact[field].values())
                    for field in ("genotype", "snp_to_gene")
                )
                or interact.get("grm") != {
                    "method": "grm_from_X", "maf_min": 0.01,
                    "scope": "all_subgenomes",
                }
                or burden != {
                    "cap": 150, "min_snp": 3, "maf_min": 0.01, "n_pc": 3,
                }
                or not isinstance(calibration, Mapping)
                or calibration.get("method") != "bootstrap"
                or isinstance(calibration.get("B"), bool)
                or not isinstance(calibration.get("B"), int)
                or calibration["B"] not in {199, 2_000}
                or isinstance(calibration.get("seed"), bool)
                or not isinstance(calibration.get("seed"), int)
                or not isinstance(calibration.get("qa_only"), bool)
                or calibration["qa_only"] is not (calibration["B"] == 199)
            ):
                raise BenchmarkAggregateError(
                    "declared interaction config is not canonical group-omniB"
                )
        hashes[relative] = digest
    actual_configs = {
        item.relative_to(root).as_posix()
        for item in (root / "configs").rglob("*")
        if item.is_file() and item.name != "manifest.tsv"
    }
    if set(hashes) != actual_configs:
        raise BenchmarkAggregateError("config directory differs from manifest")
    return hashes


def _validate_input_manifest(root: Path, path: Path) -> dict[str, dict[str, Any]]:
    if _uses_internal_symlink(root, path):
        raise BenchmarkAggregateError("input manifest must not be a symlink")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["path", "size", "sha256", "type"]:
                raise BenchmarkAggregateError("input manifest schema mismatch")
            rows = list(reader)
    except OSError as error:
        raise BenchmarkAggregateError("input manifest is missing or unreadable") from error
    if not rows:
        raise BenchmarkAggregateError("input manifest is empty")
    records: dict[str, dict[str, Any]] = {}
    seen_resolved: set[Path] = set()
    root = root.resolve()
    for row in rows:
        declared = row["path"]
        if not declared or declared in records or not row["type"]:
            raise BenchmarkAggregateError("input manifest has invalid or duplicate paths")
        declared_path = Path(declared)
        if declared_path.is_absolute():
            candidate = declared_path
            if candidate.is_symlink():
                raise BenchmarkAggregateError("declared input is not a regular file")
            resolved = candidate.resolve()
            if str(candidate) != str(resolved):
                raise BenchmarkAggregateError(
                    "external input path must be a normalized absolute path"
                )
            record_key = str(resolved)
        else:
            if not declared.startswith("inputs/"):
                raise BenchmarkAggregateError(
                    "internal input path must be rooted below inputs/"
                )
            candidate = root / declared_path
            if _uses_internal_symlink(root, candidate):
                raise BenchmarkAggregateError("declared input is not a regular file")
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise BenchmarkAggregateError(
                    "input manifest path escapes benchmark root"
                ) from error
            record_key = declared
        if record_key in records:
            raise BenchmarkAggregateError("input manifest has invalid or duplicate paths")
        if candidate.is_symlink() or not candidate.is_file():
            raise BenchmarkAggregateError("declared input is not a regular file")
        if resolved in seen_resolved:
            raise BenchmarkAggregateError("input manifest has invalid or duplicate paths")
        seen_resolved.add(resolved)
        try:
            size = int(row["size"])
        except (TypeError, ValueError) as error:
            raise BenchmarkAggregateError("input manifest size is invalid") from error
        digest = _sha256_hex(row["sha256"], "input hash")
        if size < 0 or resolved.stat().st_size != size or _file_sha256(
            resolved, "declared input"
        ) != digest:
            raise BenchmarkAggregateError("declared input size/hash mismatch")
        records[record_key] = {
            "size": size, "sha256": digest, "type": row["type"],
        }
    declared_inputs = {
        relative for relative in records if relative.startswith("inputs/")
    }
    actual_inputs = {
        item.relative_to(root).as_posix()
        for item in (root / "inputs").rglob("*")
        if item.is_file() and item.name != "manifest.tsv"
    }
    if declared_inputs != actual_inputs:
        raise BenchmarkAggregateError("input directory differs from manifest")
    return records


def _validate_context_artifact(
    root: Path,
    context_key: str,
    record: Mapping[str, Any],
    input_records: Mapping[str, Mapping[str, Any]],
) -> None:
    backbone, _, family_label = context_key.partition(":g")
    expected_group_count = int(family_label) if family_label else 80
    required_record = {
        "artifact_path", "artifact_sha256", "group_count",
        "ordered_family_ids", "ordered_family_ids_hash",
        "context_fingerprint", "family_hash",
    }
    if not isinstance(record, Mapping) or set(record) != required_record:
        raise BenchmarkAggregateError(
            f"design backbone context is invalid: {context_key}"
        )
    artifact_path = record.get("artifact_path")
    input_record = input_records.get(str(artifact_path))
    if (
        not isinstance(artifact_path, str)
        or Path(artifact_path).is_absolute()
        or not artifact_path.startswith("inputs/")
        or not isinstance(input_record, Mapping)
        or input_record.get("type") != "omnib_context"
        or input_record.get("sha256") != record.get("artifact_sha256")
    ):
        raise BenchmarkAggregateError(
            f"context artifact is not bound to the input manifest: {context_key}"
        )
    artifact_file = root / artifact_path
    if _uses_internal_symlink(root, artifact_file):
        raise BenchmarkAggregateError(
            f"context artifact must be an internal regular file: {context_key}"
        )
    artifact = _strict_json(artifact_file)
    expected_artifact_fields = {
        "schema", "backbone", "context_manifest", "context_fingerprint",
        "family_manifest", "family_hash", "group_count", "ordered_family_ids",
        "ordered_family_ids_hash", "source_inputs",
    }
    family = artifact.get("family_manifest")
    context = artifact.get("context_manifest")
    source_inputs = artifact.get("source_inputs")
    context_subgenomes = context.get("subgenomes") if isinstance(context, Mapping) else None
    sample_idx = context.get("sample_idx") if isinstance(context, Mapping) else None
    phenotype = context.get("phenotype") if isinstance(context, Mapping) else None
    if (
        set(artifact) != expected_artifact_fields
        or artifact.get("schema") != "homoeogwas-v201-omnib-context-v1"
        or artifact.get("backbone") != backbone
        or not isinstance(family, Mapping)
        or not isinstance(context, Mapping)
        or context.get("family") != family
        or not isinstance(context_subgenomes, list)
        or len(context_subgenomes) != len(family.get("subgenomes", []))
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("label"), str)
            or not isinstance(item.get("X"), Mapping)
            or not isinstance(item.get("gene_snp"), list)
            or not isinstance(item.get("samples"), list)
            or not item.get("samples")
            for item in context_subgenomes
        )
        or not isinstance(sample_idx, Mapping)
        or not isinstance(sample_idx.get("values"), list)
        or not sample_idx.get("values")
        or not isinstance(phenotype, Mapping)
        or not isinstance(phenotype.get("shape"), list)
        or artifact.get("family_hash") != sha256_payload(family)
        or artifact.get("context_fingerprint") != sha256_payload(context)
        or family.get("group_ids") != artifact.get("ordered_family_ids")
        or artifact.get("group_count") != len(artifact.get("ordered_family_ids", []))
        or artifact.get("group_count") != expected_group_count
        or artifact.get("ordered_family_ids_hash")
        != sha256_payload(artifact.get("ordered_family_ids"))
        or not isinstance(source_inputs, list)
        or not source_inputs
    ):
        raise BenchmarkAggregateError(f"context artifact is invalid: {context_key}")
    seen: set[str] = set()
    for source in source_inputs:
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256", "type"}:
            raise BenchmarkAggregateError(f"context source input is invalid: {context_key}")
        source_path = source.get("path")
        manifest_source = input_records.get(str(source_path))
        if (
            not isinstance(source_path, str)
            or source_path == artifact_path
            or source_path in seen
            or not isinstance(manifest_source, Mapping)
            or dict(source) != {
                "path": source_path,
                "sha256": manifest_source.get("sha256"),
                "type": manifest_source.get("type"),
            }
        ):
            raise BenchmarkAggregateError(f"context source input is unbound: {context_key}")
        seen.add(source_path)
    expected_record = {
        "artifact_path": artifact_path,
        "artifact_sha256": input_record.get("sha256"),
        "group_count": artifact["group_count"],
        "ordered_family_ids": artifact["ordered_family_ids"],
        "ordered_family_ids_hash": artifact["ordered_family_ids_hash"],
        "context_fingerprint": artifact["context_fingerprint"],
        "family_hash": artifact["family_hash"],
    }
    if dict(record) != expected_record:
        raise BenchmarkAggregateError(f"context lock differs from artifact: {context_key}")


def _current_git_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BenchmarkAggregateError("cannot resolve current software commit") from error
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in _HEX for character in commit):
        raise BenchmarkAggregateError("current software commit is invalid")
    return commit


def _validate_locked_root(
    root: Path,
    lock: Mapping[str, Any],
    registry: Sequence[RegistryRow],
    registry_hash: str,
) -> None:
    stage = registry[0].stage
    canonical = build_scenarios(stage)
    actual_rows = [
        {
            "scenario_id": row.scenario_id, "track": row.track, "stage": row.stage,
            "replicates": row.replicates, "bootstrap_B": row.bootstrap_B,
            "parameters": dict(row.parameters),
        }
        for row in registry
    ]
    expected_rows = [row.to_dict() for row in canonical]
    if actual_rows != expected_rows:
        raise BenchmarkAggregateError("scenario registry differs from canonical design")
    expected_ids = [row.scenario_id for row in canonical]
    if lock.get("stage") != stage:
        raise BenchmarkAggregateError("design lock stage differs from registry")
    if lock.get("ordered_scenario_ids") != expected_ids:
        raise BenchmarkAggregateError("design scenario order mismatch")
    if lock.get("scenario_registry_sha256") != registry_hash:
        raise BenchmarkAggregateError("scenario registry hash mismatch")
    if lock.get("master_seed") != MASTER_SEED:
        raise BenchmarkAggregateError("design master seed mismatch")
    target = lock.get("target_release")
    harness = lock.get("harness")
    if not isinstance(target, Mapping) or not isinstance(harness, Mapping):
        raise BenchmarkAggregateError("release target/harness identity is missing")
    from homoeogwas import __version__

    if (
        target != {
            "version": "2.0.1", "tag": "v2.0.1",
            "git_commit": "015e023439addf2cca658cd01720ff6f856023be",
        }
        or __version__ != "2.0.1"
        or harness.get("git_commit") != _current_git_commit()
    ):
        raise BenchmarkAggregateError("release target/harness identity mismatch")
    locked_files = {
        "input_manifest_sha256": root / "inputs" / "manifest.tsv",
        "comparator_preflight_sha256": root / "inputs" / "comparator_preflight.tsv",
        "config_manifest_sha256": root / "configs" / "manifest.tsv",
    }
    for field, path in locked_files.items():
        digest = _file_sha256(path, field)
        if lock.get(field) != digest:
            raise BenchmarkAggregateError(f"{field} mismatch")
    input_records = _validate_input_manifest(root, locked_files["input_manifest_sha256"])
    if lock.get("input_records") != input_records:
        raise BenchmarkAggregateError("design input records mismatch")
    config_hashes = _validate_config_manifest(root, locked_files["config_manifest_sha256"])
    if lock.get("config_hashes") != config_hashes:
        raise BenchmarkAggregateError("design config hashes mismatch")
    scenario_configs = lock.get("scenario_config_bindings")
    if (
        not isinstance(scenario_configs, Mapping)
        or set(scenario_configs) != set(expected_ids)
        or any(
            not isinstance(path, str) or path not in config_hashes
            for path in scenario_configs.values()
        )
    ):
        raise BenchmarkAggregateError("design scenario/config bindings mismatch")
    contexts = lock.get("benchmark_contexts")
    required_backbones = {
        str(row.parameters["backbone"])
        for row in canonical
        if row.track == "omnib" and "backbone" in row.parameters
    }
    required_contexts = required_backbones | {
        f"{row.parameters['backbone']}:g{row.parameters['family_size']}"
        for row in canonical
        if row.track == "omnib" and row.parameters.get("experiment") == "family_size"
    }
    if not isinstance(contexts, Mapping) or set(contexts) != required_contexts:
        raise BenchmarkAggregateError("design backbone contexts are incomplete")
    for context_key, record in contexts.items():
        _validate_context_artifact(root, context_key, record, input_records)
        backbone, _, family_label = context_key.partition(":g")
        bound_scenarios = [
            row for row in canonical
            if row.track == "omnib"
            and row.parameters.get("backbone") == backbone
            and (
                (family_label and row.parameters.get("experiment") == "family_size"
                 and row.parameters.get("family_size") == int(family_label))
                or (not family_label and row.parameters.get("experiment") != "family_size")
            )
        ]
        config_paths = {
            str(scenario_configs[row.scenario_id]) for row in bound_scenarios
        }
        if not bound_scenarios or not config_paths:
            raise BenchmarkAggregateError(
                f"context has no scenario-bound config: {context_key}"
            )
        artifact = _strict_json(root / str(record["artifact_path"]))
        artifact_family = artifact.get("family_manifest")
        expected_subgenomes = (
            artifact_family.get("subgenomes")
            if isinstance(artifact_family, Mapping) else None
        )
        if not isinstance(expected_subgenomes, list):
            raise BenchmarkAggregateError(
                f"context subgenomes are invalid: {context_key}"
            )
        manifest_hashes = {
            str((root / relative).resolve()): str(item["sha256"])
            for relative, item in input_records.items()
        }
        manifest_hashes.update({
            str((root / relative).resolve()): digest
            for relative, digest in config_hashes.items()
        })
        from .track_omnib import validate_real_omnib_context

        for relative in sorted(config_paths):
            try:
                validate_real_omnib_context(
                    root / relative,
                    context_artifact_path=root / str(record["artifact_path"]),
                    input_manifest=manifest_hashes,
                    stage=stage,
                    backbone=backbone,
                    family_size=int(family_label) if family_label else 80,
                    expected_subgenomes=expected_subgenomes,
                    manifest_root=root,
                )
            except (OSError, ValueError) as error:
                raise BenchmarkAggregateError(
                    f"strict real context validation failed: {context_key}"
                ) from error
    loco_ids = {
        row.scenario_id for row in canonical
        if row.track == "fit" and row.parameters.get("experiment") == "loco"
    }
    loco_artifacts = lock.get("loco_truth_artifacts")
    if loco_ids and (
        not isinstance(loco_artifacts, Mapping) or set(loco_artifacts) != loco_ids
    ):
        raise BenchmarkAggregateError("design LOCO truth artifacts are incomplete")
    required_loco = {
        "path", "sha256", "truth_hash", "source", "seed",
        "generated_config_sha256", "phenotype_sha256",
    }
    for scenario_id, record in (loco_artifacts or {}).items():
        if (
            not isinstance(record, Mapping) or set(record) != required_loco
            or record.get("path") not in input_records
            or input_records[str(record["path"])].get("sha256") != record.get("sha256")
            or input_records[str(record["path"])].get("type") != "loco_truth"
            or any(
                not isinstance(record.get(field), str)
                or len(record[field]) != 64
                or any(character not in _HEX for character in record[field])
                for field in (
                    "sha256", "truth_hash", "generated_config_sha256",
                    "phenotype_sha256",
                )
            )
            or not isinstance(record.get("source"), str) or not record["source"]
            or isinstance(record.get("seed"), bool) or not isinstance(record["seed"], int)
        ):
            raise BenchmarkAggregateError(
                f"design LOCO truth artifact is invalid: {scenario_id}"
            )
    if lock.get("acceptance_rules") != ACCEPTANCE_RULES:
        raise BenchmarkAggregateError("design acceptance rules mismatch")


def load_evidence(root: str | Path) -> LoadedEvidence:
    """Load and validate a complete immutable evidence set in canonical order."""

    requested_root = Path(root)
    if requested_root.is_symlink():
        raise BenchmarkAggregateError("benchmark root must not be a symlink")
    benchmark_root = requested_root.resolve()
    lock_path = benchmark_root / "design_lock.json"
    registry_path = benchmark_root / "scenario_registry.tsv"
    if (
        _uses_internal_symlink(benchmark_root, lock_path)
        or _uses_internal_symlink(benchmark_root, registry_path)
    ):
        raise BenchmarkAggregateError("design lock/registry must not be symlinks")
    lock = _strict_json(lock_path)
    if lock.get("schema") != "homoeogwas-v201-benchmark-lock-v1":
        raise BenchmarkAggregateError("design lock schema mismatch")
    design_hash = _sha256_hex(lock.get("design_hash"), "design hash")
    lock_payload = {key: value for key, value in lock.items() if key != "design_hash"}
    if sha256_payload(lock_payload) != design_hash:
        raise BenchmarkAggregateError("design hash mismatch")
    registry = _read_registry(registry_path)
    stage = registry[0].stage
    registry_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    _validate_locked_root(benchmark_root, lock, registry, registry_hash)

    expected: dict[tuple[str, str, int], RegistryRow] = {}
    expected_paths: dict[tuple[str, str, int], Path] = {}
    for row in registry:
        for replicate in _expected_replicates(row):
            key = (row.track, row.scenario_id, replicate)
            expected[key] = row
            expected_paths[key] = (
                benchmark_root / stage / row.track / row.scenario_id
                / f"replicate-{replicate:06d}.json"
            )
    all_json = sorted(
        [
            path
            for candidate_stage in ("pilot", "formal")
            for path in (benchmark_root / candidate_stage).rglob("*.json")
        ],
        key=lambda path: path.relative_to(benchmark_root).as_posix(),
    )
    discovered = []
    for path in all_json:
        relative = path.relative_to(benchmark_root)
        if not path.name.startswith("replicate-") or not path.name.endswith(".json"):
            raise BenchmarkAggregateError(f"unexpected JSON evidence: {path}")
        if path.is_symlink() or any(
            (benchmark_root / Path(*relative.parts[:index])).is_symlink()
            for index in range(1, len(relative.parts))
        ):
            raise BenchmarkAggregateError(f"symlink shard/output path is forbidden: {path}")
        discovered.append(path)
    by_key: dict[tuple[str, str, int], tuple[Path, Mapping[str, Any]]] = {}
    for path in discovered:
        payload = _strict_json(path)
        replicate = payload.get("replicate")
        key = (payload.get("track"), payload.get("scenario_id"), replicate)
        if (
            not isinstance(key[0], str) or not isinstance(key[1], str)
            or isinstance(replicate, bool) or not isinstance(replicate, int)
        ):
            raise BenchmarkAggregateError(f"invalid shard identity: {path}")
        if key in by_key:
            raise BenchmarkAggregateError(f"duplicate shard identity: {key}")
        if key not in expected or path != expected_paths[key]:
            raise BenchmarkAggregateError(f"unexpected shard: {path}")
        by_key[key] = (path, payload)
    missing = [key for key in expected if key not in by_key]
    if missing:
        raise BenchmarkAggregateError(f"missing shard: {missing[0]}")

    ordered: list[tuple[Path, Mapping[str, Any]]] = []
    for key in sorted(expected):
        path, payload = by_key[key]
        row = expected[key]
        if payload.get("design_hash") != design_hash:
            raise BenchmarkAggregateError(f"shard design hash mismatch: {path}")
        if payload.get("stage") != stage:
            raise BenchmarkAggregateError(f"shard stage mismatch: {path}")
        if payload.get("formal") is not (stage == "formal"):
            raise BenchmarkAggregateError(f"shard formal marker mismatch: {path}")
        if payload.get("qa_only") is not (stage == "pilot"):
            raise BenchmarkAggregateError(f"shard QA marker mismatch: {path}")
        _sha256_hex(payload.get("request_hash"), "request hash")
        _sha256_hex(payload.get("context_fingerprint"), "context fingerprint")
        if payload.get("experiment") != row.parameters.get("experiment"):
            raise BenchmarkAggregateError(f"shard experiment mismatch: {path}")
        backbone = row.parameters.get("backbone")
        if row.track == "omnib" and isinstance(backbone, str):
            context_key = (
                f"{backbone}:g{row.parameters['family_size']}"
                if row.parameters.get("experiment") == "family_size" else backbone
            )
            locked_context = lock["benchmark_contexts"].get(context_key)
            if (
                not isinstance(locked_context, Mapping)
                or payload.get("context_fingerprint")
                != locked_context.get("context_fingerprint")
                or payload.get("family_hash") != locked_context.get("family_hash")
                or payload.get("family_ids")
                != locked_context.get("ordered_family_ids")
            ):
                raise BenchmarkAggregateError(
                    f"shard context differs from design backbone: {path}"
                )
        ordered.append((path, payload))
    return LoadedEvidence(
        benchmark_root, stage, lock, design_hash, registry_hash, registry,
        tuple(ordered),
    )


def _json_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise BenchmarkAggregateError("non-finite table value")
    return value


def _atomic_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise BenchmarkAggregateError(f"table row schema mismatch: {path.name}")
            writer.writerow({field: _json_cell(row[field]) for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _failure(payload: Mapping[str, Any]) -> tuple[bool, str | None]:
    failure = payload.get("failure")
    if not isinstance(failure, Mapping) or not isinstance(failure.get("failed"), bool):
        raise BenchmarkAggregateError("shard failure state is missing or invalid")
    return bool(failure["failed"]), failure.get("error_type")


def _common(payload: Mapping[str, Any]) -> dict[str, Any]:
    failed, failure_type = _failure(payload)
    inference = payload.get("inference_status")
    if payload["stage"] == "pilot":
        inference = "noninferential_do_not_threshold"
    elif inference is None:
        inference = "formal"
    return {
        "stage": payload["stage"], "inference_status": inference,
        "track": payload["track"], "scenario_id": payload["scenario_id"],
        "replicate": payload["replicate"], "design_hash": payload["design_hash"],
        "request_hash": payload["request_hash"],
        "context_fingerprint": payload["context_fingerprint"], "failed": failed,
        "failure_type": failure_type, "runtime_seconds": payload.get("runtime_seconds"),
    }


def _binomial_cells(prefix: str, decisions: Sequence[bool], failures: int) -> dict[str, Any]:
    total = len(decisions)
    if total < 1 or failures < 0 or failures > total:
        raise BenchmarkAggregateError("robustness binomial denominator is invalid")
    successes = sum(value is True for value in decisions)
    estimate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {
        f"{prefix}_successes": successes, f"{prefix}_total": total,
        f"{prefix}_estimate": estimate,
        f"{prefix}_ci_low": max(0.0, center - half),
        f"{prefix}_ci_high": min(1.0, center + half),
        f"{prefix}_failures": failures,
    }


def _empty(name: str, common: Mapping[str, Any], **values: Any) -> dict[str, Any]:
    row = {field: None for field in TABLE_SCHEMAS[name]}
    row.update(common)
    row.update(values)
    return row


def _fit_rows(payload: Mapping[str, Any], registry: RegistryRow) -> tuple[str, list[dict[str, Any]]]:
    common = _common(payload)
    experiment = str(payload["experiment"])
    panel = registry.parameters.get("panel")
    allocation = registry.parameters.get("allocation")
    if experiment == "recovery":
        name = "fit_pve_recovery.tsv"
        true = payload.get("true_pve") or payload.get("target_pve") or {}
        estimated = payload.get("estimated_pve") or {}
        derived = payload.get("audit_derived") or {}
        rows = []
        for subgenome in sorted((set(true) | set(estimated)) - {"e"}):
            true_value, estimate = true.get(subgenome), estimated.get(subgenome)
            rows.append(_empty(name, common, panel=panel, allocation=allocation,
                subgenome=subgenome, true_pve=true_value, estimated_pve=estimate,
                bias=(estimate - true_value if isinstance(estimate, (int, float))
                      and isinstance(true_value, (int, float)) else None),
                absolute_error=(abs(estimate - true_value)
                    if isinstance(estimate, (int, float))
                    and isinstance(true_value, (int, float)) else None),
                squared_error=((estimate - true_value) ** 2
                    if isinstance(estimate, (int, float))
                    and isinstance(true_value, (int, float)) else None),
                replicate_rmse=derived.get("rmse"),
                replicate_spearman=derived.get("spearman"),
                optimizer_status=payload.get("optimizer_status"),
                dominant_correct=derived.get("dominant_correct"),
                boundary_fit=derived.get("boundary_fit")))
        return name, rows or [_empty(name, common, panel=panel, allocation=allocation)]
    if experiment == "coverage":
        name = "fit_pve_coverage.tsv"
        uncertainty = payload.get("pve_uncertainty") or payload.get("pve_bootstrap") or {}
        target = payload.get("target_pve") or {}
        rows = []
        if isinstance(uncertainty, Mapping):
            intervals = uncertainty.get("intervals", uncertainty)
            if isinstance(intervals, Mapping):
                for subgenome in sorted(intervals):
                    if subgenome == "e":
                        continue
                    record = intervals[subgenome]
                    if not isinstance(record, Mapping):
                        continue
                    low = record.get("ci_low", record.get("low"))
                    high = record.get("ci_high", record.get("high"))
                    truth = (
                        sum(value for name, value in target.items() if name != "e")
                        if subgenome == "total_genetic" else target.get(subgenome)
                    )
                    covered = (
                        low <= truth <= high if all(isinstance(x, (int, float))
                        for x in (low, truth, high)) else None
                    )
                    coverage = uncertainty.get("coverage", {})
                    if isinstance(coverage, Mapping) and subgenome in coverage:
                        covered = coverage[subgenome]
                    rows.append(_empty(name, common, panel=panel, allocation=allocation,
                        subgenome=subgenome, target_pve=truth,
                        interval_low=low, interval_high=high,
                        covered=covered,
                        interval_width=(high - low if isinstance(low, (int, float))
                                        and isinstance(high, (int, float)) else None),
                        bootstrap_failures=uncertainty.get("B_failed"),
                        successes=int(bool(covered)), total=1,
                        estimate=float(bool(covered)), failures=0))
        return name, rows or [_empty(name, common, panel=panel, allocation=allocation)]
    name = "fit_scan_metrics.tsv"
    comparators = payload.get("comparators") or {}
    derived = payload.get("audit_derived") or {}
    rows = []
    expected_methods = (
        ("canonical_multi_kernel",)
        if experiment == "loco"
        else ("canonical_multi_kernel", "pooled_trace_sum", "independent_subgenome")
    )
    if common["failed"]:
        return name, [
            _empty(name, common, panel=panel,
                scan_arm=("loco_sensitivity" if experiment == "loco" else "primary"),
                method=method, scan_pve=registry.parameters.get("scan_pve"),
                placement=registry.parameters.get("placement"), distance_unit="bp")
            for method in expected_methods
        ]
    if isinstance(comparators, Mapping):
        for method in sorted(comparators):
            record = comparators[method]
            fwer = derived.get(method, {}) if isinstance(derived, Mapping) else {}
            rows.append(_empty(name, common, panel=panel,
                scan_arm=payload.get("scan_arm", experiment), method=method,
                scan_pve=registry.parameters.get("scan_pve"),
                placement=registry.parameters.get("placement"),
                minimum_p=fwer.get("minimum_p"), rejected=fwer.get("rejected"),
                causal_detected=fwer.get("causal_detected"),
                lead_distance=fwer.get("lead_distance"),
                localization_correct=fwer.get("localization_correct"),
                distance_unit="bp",
                lambda_gc=fwer.get("lambda_gc")))
    return name, rows or [_empty(name, common, panel=panel,
        scan_arm=payload.get("scan_arm", experiment),
        scan_pve=registry.parameters.get("scan_pve"),
        placement=registry.parameters.get("placement"))]


def _omnib_rows(payload: Mapping[str, Any], registry: RegistryRow) -> dict[str, list[dict[str, Any]]]:
    output = {name: [] for name in TABLE_SCHEMAS if name.startswith("omnib_")}
    common = _common(payload)
    if payload.get("experiment") == "global_vc":
        name = "omnib_null_replicates.tsv"
        calibration = payload.get("calibration_p_values")
        target = payload.get("target_p_values")
        if not isinstance(calibration, list) or not isinstance(target, list):
            raise BenchmarkAggregateError("global VC p-value evidence is missing")
        formal = payload.get("stage") == "formal"
        threshold = payload.get("threshold") if formal else None
        decisions = payload.get("rejected") if formal else [None] * len(target)
        failed_indices = set(payload.get("failed_target_response_indices", []))
        lrt_evidence = payload.get("target_lrt_evidence", [])
        for index, value in enumerate(target):
            failed = index in failed_indices
            failure_type = (
                lrt_evidence[index].get("error_type")
                if failed and index < len(lrt_evidence)
                and isinstance(lrt_evidence[index], Mapping) else None
            )
            output[name].append(_empty(
                name, common, method="global_hadamard_variance_component",
                null_model=registry.parameters.get("null_model", "gaussian"),
                stress=False, bank_role=payload.get("target_role"),
                response_index=index, minimum_p=value, threshold=threshold,
                rejected=decisions[index], family_size=1,
                family_hash=payload["kernel_manifest"]["global_hadamard_sha256"],
                score_family_size=1, tested_family_size=1,
                tested_family_hash=payload["kernel_manifest"]["global_hadamard_sha256"],
                response_seed_id=(payload.get("target_response_ids") or [None])[index],
                failed=failed, failure_type=failure_type,
            ))
        return output
    family_ids = payload.get("family_ids")
    if not isinstance(family_ids, list) or not all(isinstance(item, str) and item for item in family_ids):
        raise BenchmarkAggregateError("omniB shard family IDs are missing")
    family_hash = _sha256_hex(payload.get("family_hash"), "family hash")
    order_hash = payload.get("family_order_hash") or sha256_payload(family_ids)
    if order_hash != sha256_payload(family_ids):
        raise BenchmarkAggregateError("family order hash mismatch")
    manifest = "omnib_family_manifest.tsv"
    output[manifest].append(_empty(manifest, common, method="group_omniB",
        hypothesis_unit=payload.get("hypothesis_unit", "group"),
        family_scope=payload.get("family_scope", "primary_only"),
        family_size=len(family_ids), ordered_ids=family_ids, family_hash=family_hash,
        family_order_hash=order_hash, callable=not common["failed"],
        requested_jobs=payload.get("requested_jobs"),
        effective_jobs=payload.get("effective_jobs"),
        backend=payload.get("parallel_backend"), worker_pids=payload.get("worker_pids")))
    experiment = payload["experiment"]
    if experiment == "end2end":
        name = "omnib_null_replicates.tsv"
        values = payload.get("observed_group_p")
        adjusted = payload.get("adjusted_p")
        decisions = payload.get(
            "adjusted_decisions" if payload.get("stage") == "formal"
            else "qa_adjusted_diagnostic_decisions"
        )
        if common["failed"]:
            output[name].append(_empty(name, common, method="group_omniB",
                null_model=registry.parameters.get("null_model"), stress=False,
                bank_role="heldout", response_index=payload["replicate"],
                family_size=len(family_ids), family_hash=family_hash,
                family_order_hash=order_hash,
                response_seed_id=payload.get("response_seed_id"),
                calibration_seed_id=payload.get("calibration_seed_id")))
            return output
        if not isinstance(values, list) or not isinstance(adjusted, list) or not isinstance(decisions, list):
            raise BenchmarkAggregateError("end-to-end decision evidence is incomplete")
        minimum = min(values) if values else None
        rejected = bool(payload.get("formal_rejections") if payload["stage"] == "formal"
                        else payload.get("qa_diagnostic_rejections"))
        output[name].append(_empty(name, common, method="group_omniB",
            null_model=registry.parameters.get("null_model"), stress=False,
            bank_role="heldout", response_index=payload["replicate"],
            minimum_p=minimum,
            threshold=((payload.get("bootstrap_minp") or {}).get("threshold")
                       if payload["stage"] == "formal" else None),
            rejected=(rejected if payload["stage"] == "formal" else None),
            family_size=len(family_ids), family_hash=family_hash,
            family_order_hash=order_hash,
            score_family_size=len(family_ids), tested_family_size=len(family_ids),
            tested_family_hash=family_hash,
            response_seed_id=payload.get("response_seed_id"),
            calibration_seed_id=payload.get("calibration_seed_id")))
    elif experiment == "conditional":
        name = "omnib_null_replicates.tsv"
        bank = payload.get("bank")
        if common["failed"] and not isinstance(bank, Mapping):
            output[name].append(_empty(name, common, method="group_omniB",
                null_model=registry.parameters.get("null_model", "gaussian"),
                stress=bool(registry.parameters.get("stress", False)),
                bank_role=registry.parameters.get("bank"),
                family_size=len(family_ids), family_hash=family_hash,
                family_order_hash=order_hash))
            return output
        if not isinstance(bank, Mapping):
            raise BenchmarkAggregateError("conditional bank evidence is missing")
        matrices = bank.get("p_by_method")
        if not isinstance(matrices, Mapping):
            raise BenchmarkAggregateError("conditional score matrices are missing")
        seed_ids = bank.get("seed_ids")
        if not isinstance(seed_ids, list):
            raise BenchmarkAggregateError("conditional seed IDs are missing")
        if set(matrices) != set(METHOD_NAMES):
            raise BenchmarkAggregateError(
                "conditional method set differs from locked comparators"
            )
        tested_members = bank.get("tested_family_members")
        tested_sizes = bank.get("tested_family_sizes")
        tested_hashes = bank.get("tested_family_hashes")
        hypothesis_units = bank.get("score_hypothesis_units")
        if not all(
            isinstance(value, Mapping)
            and set(value) == set(METHOD_NAMES)
            for value in (tested_members, tested_sizes, tested_hashes, hypothesis_units)
        ):
            raise BenchmarkAggregateError(
                "conditional tested-family evidence is incomplete"
            )
        for method in METHOD_NAMES:
            matrix = matrices[method]
            members = tested_members[method]
            size = tested_sizes[method]
            tested = tested_hashes[method]
            if not isinstance(matrix, list) or not matrix:
                raise BenchmarkAggregateError("conditional score matrix is invalid")
            for index in range(len(seed_ids)):
                column = [row[index] for row in matrix]
                finite = [value for value in column if value is not None]
                output[name].append(_empty(name, common, method=method,
                    null_model=registry.parameters.get("null_model", "gaussian"),
                    stress=bool(registry.parameters.get("stress", False)),
                    bank_role=bank.get("canonical_role"), response_index=index,
                    minimum_p=min(finite) if finite else None,
                    family_size=len(matrix), family_hash=family_hash,
                    score_family_size=len(matrix), tested_family_size=size,
                    tested_family_hash=tested,
                    family_order_hash=order_hash, response_seed_id=seed_ids[index]))
            if not isinstance(members, list) or size != len(members):
                raise BenchmarkAggregateError(
                    "conditional tested-family manifest mismatch"
                )
            output[manifest].append(_empty(manifest, common, method=method,
                hypothesis_unit=hypothesis_units[method], family_scope="primary_only",
                family_size=size, ordered_ids=members,
                family_hash=family_hash, family_order_hash=order_hash,
                tested_family_hash=tested, callable=not common["failed"],
                requested_jobs=bank.get("requested_jobs"),
                effective_jobs=bank.get("effective_jobs"),
                backend=bank.get("parallel_backend"), worker_pids=bank.get("worker_pids")))
    elif experiment == "family_size":
        name = "omnib_null_replicates.tsv"
        rejections = payload.get(
            "rejections_by_method" if payload.get("stage") == "formal"
            else "qa_rejections_by_method"
        )
        minima = payload.get("target_minima_by_method")
        thresholds = payload.get(
            "thresholds" if payload.get("stage") == "formal" else "qa_cutoffs"
        )
        tested_sizes = payload.get("tested_family_sizes")
        tested_hashes = payload.get("tested_family_hashes")
        if not all(isinstance(value, Mapping) and value for value in (
            rejections, minima, thresholds, tested_sizes, tested_hashes,
        )):
            raise BenchmarkAggregateError("family-size FWER evidence is missing")
        for method, decisions in sorted(rejections.items()):
            if not isinstance(decisions, list) or not isinstance(minima.get(method), list):
                raise BenchmarkAggregateError("family-size response evidence is invalid")
            for index, rejected in enumerate(decisions):
                output[name].append(_empty(
                    name, common, method=method, null_model="gaussian", stress=True,
                    bank_role="heldout", response_index=index,
                    minimum_p=minima[method][index],
                    threshold=(thresholds[method]
                               if payload["stage"] == "formal" else None),
                    rejected=(bool(rejected) if payload["stage"] == "formal" else None),
                    family_size=payload.get("family_size"),
                    score_family_size=payload.get("group_count"),
                    tested_family_size=tested_sizes[method],
                    tested_family_hash=tested_hashes[method],
                    family_hash=family_hash, family_order_hash=order_hash,
                ))
    elif experiment == "power":
        name = "omnib_power_replicates.tsv"
        negative_control = registry.parameters.get("control_type") == "negative"
        formal = payload.get("stage") == "formal"
        decisions = payload.get(
            ("specificity_by_method" if formal else "qa_specificity_by_method")
            if negative_control else (
                "causal_detection_by_method" if formal
                else "qa_causal_detection_by_method"
            )
        )
        if common["failed"] and not isinstance(decisions, Mapping):
            for method in METHOD_NAMES:
                output[name].append(_empty(name, common, method=method,
                    architecture=payload.get(
                        "architecture", registry.parameters.get("architecture")
                    ), interaction_pve=payload.get(
                        "interaction_pve", registry.parameters.get("interaction_pve")
                    ), causal_groups=registry.parameters.get("causal_groups"),
                    response_index=payload["replicate"], family_hash=family_hash))
            return output
        if not isinstance(decisions, Mapping):
            raise BenchmarkAggregateError("power decision evidence is missing")
        if set(decisions) != set(METHOD_NAMES):
            raise BenchmarkAggregateError("power method set differs from locked comparators")
        thresholds = payload.get(
            "thresholds" if formal else "qa_cutoffs_by_method", {}
        )
        for method in METHOD_NAMES:
            method_decisions = decisions[method]
            if not isinstance(method_decisions, list):
                raise BenchmarkAggregateError("power decisions are invalid")
            for index, detected in enumerate(method_decisions):
                output[name].append(_empty(name, common, method=method,
                    architecture=payload.get("architecture"),
                    control_type=payload.get("control_type", "positive"),
                    interaction_pve=payload.get("interaction_pve"),
                    causal_groups=len(payload.get("causal_group_ids", [])),
                    response_index=index,
                    detected=(None if negative_control else detected),
                    false_positive=(not detected if negative_control else None),
                    specificity=(detected if negative_control else None),
                    recall=(None if negative_control else
                            (payload.get(
                                "recall_by_method" if payload.get("stage") == "formal"
                                else "qa_recall_by_method", {}
                            ).get(method)
                             or [None])[index]),
                    threshold=(thresholds.get(method)
                               if payload["stage"] == "formal" else None),
                    family_hash=family_hash,
                    calibration_response_hash=payload.get("calibration_response_hash"),
                    target_response_hash=payload.get("target_response_hash")))
    elif experiment == "encoding":
        name = "omnib_encoding_robustness.tsv"
        for perturbation, record in sorted(payload.get("exact_checks", {}).items()):
            output[name].append(_empty(name, common, perturbation=perturbation,
                check_kind="exact", required=record.get("required"),
                status=record.get("status"),
                observed_arrays_identical=record.get("observed_arrays_identical"),
                adjusted_decisions_identical=record.get("adjusted_decisions_identical"),
                ranking_hash_identical=record.get("ranking_hash_identical"),
                rejection_sets_identical=record.get("rejection_sets_identical")))
        for perturbation, record in sorted(payload.get("robustness_checks", {}).items()):
            pilot = payload.get("stage") == "pilot"
            strata = record.get(
                "qa_power_by_architecture" if pilot else "power_by_architecture"
            )
            heldout = record.get("heldout")
            if record.get("status") != "completed":
                output[name].append(_empty(name, common, perturbation=perturbation,
                    check_kind="robustness", required=record.get("required"),
                    status=record.get("status"),
                    realized_marker_design=record.get("realized_marker_design")))
                continue
            if not isinstance(strata, Mapping) or not isinstance(heldout, Mapping):
                raise BenchmarkAggregateError("robustness raw response evidence is missing")
            heldout_decisions = heldout.get(
                "qa_rejections_by_method" if pilot else "rejections_by_method"
            )
            heldout_scores = heldout.get("p_by_method")
            if not isinstance(heldout_decisions, Mapping) or not isinstance(
                heldout_scores, Mapping
            ):
                raise BenchmarkAggregateError("robustness heldout evidence is invalid")
            for architecture, stratum in sorted(strata.items()):
                if not isinstance(stratum, Mapping):
                    raise BenchmarkAggregateError("robustness architecture evidence is invalid")
                power_decisions = stratum.get(
                    "qa_detection_by_method" if pilot else "detection_by_method"
                )
                power_scores = stratum.get("p_by_method")
                if not isinstance(power_decisions, Mapping) or not isinstance(
                    power_scores, Mapping
                ):
                    raise BenchmarkAggregateError("robustness power evidence is invalid")
                correlations_by_method = stratum.get("rank_correlation_by_method")
                overlaps_by_method = stratum.get("top_k_jaccard_by_method")
                regret_by_method = stratum.get(
                    "qa_absolute_power_regret_by_method"
                    if pilot else "absolute_power_regret_by_method"
                )
                if not all(
                    isinstance(value, Mapping)
                    and set(value) == set(_ROBUSTNESS_METHODS)
                    for value in (
                        correlations_by_method, overlaps_by_method, regret_by_method,
                    )
                ):
                    raise BenchmarkAggregateError(
                        "robustness method-specific metrics are invalid"
                    )
                for method in sorted(power_decisions):
                    fwer_flags = heldout_decisions.get(method)
                    power_flags = power_decisions.get(method)
                    if (
                        not isinstance(fwer_flags, list)
                        or any(not isinstance(value, bool) for value in fwer_flags)
                        or not isinstance(power_flags, list)
                        or any(not isinstance(value, bool) for value in power_flags)
                    ):
                        raise BenchmarkAggregateError(
                            "robustness response decisions are invalid"
                        )
                    non_estimable = 0
                    non_estimable_total = 0
                    failure_by_bank: list[int] = []
                    for score_bank in (heldout_scores.get(method), power_scores.get(method)):
                        if not isinstance(score_bank, list) or not score_bank:
                            raise BenchmarkAggregateError(
                                "robustness score matrix is invalid"
                            )
                        values = np.asarray([
                            [np.nan if value is None else value for value in row]
                            for row in score_bank
                        ], dtype=float)
                        if values.ndim != 2:
                            raise BenchmarkAggregateError(
                                "robustness score matrix is invalid"
                            )
                        failures = int((~np.isfinite(values).any(axis=0)).sum())
                        failure_by_bank.append(failures)
                        non_estimable += failures
                        non_estimable_total += values.shape[1]
                    output[name].append(_empty(
                        name, common, perturbation=perturbation,
                        check_kind="robustness", required=record.get("required"),
                        status=record.get("status"), architecture=architecture,
                        method=method,
                        rank_correlation=(
                            sum(correlations) / len(correlations)
                            if (correlations := [
                                float(value)
                                for value in correlations_by_method[method]
                                if value is not None
                            ]) else None
                        ),
                        top_k_jaccard=(
                            sum(overlaps) / len(overlaps)
                            if (overlaps := [
                                float(value)
                                for value in overlaps_by_method[method]
                                if value is not None
                            ]) else None
                        ),
                        non_estimable_rate=(non_estimable / non_estimable_total),
                        non_estimable_count=non_estimable,
                        non_estimable_total=non_estimable_total,
                        absolute_power_regret=regret_by_method[method],
                        realized_marker_design=record.get("realized_marker_design"),
                        **_binomial_cells(
                            "fwer" if payload.get("stage") == "formal" else "qa_fwer",
                            fwer_flags, failure_by_bank[0],
                        ),
                        **_binomial_cells(
                            "power" if payload.get("stage") == "formal" else "qa_power",
                            power_flags, failure_by_bank[1],
                        ),
                    ))
    return output


def _scaling_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    name = "scaling_runs.tsv"
    common = _common(payload)
    summary = payload.get("summary", payload)
    anchor = summary.get("anchor", {})
    runs = summary.get("runs", {})
    rows: list[dict[str, Any]] = []
    for jobs in sorted(runs, key=int):
        job = runs[jobs]
        for record in job.get("measured_repeats", []):
            rows.append(_empty(name, common, anchor=anchor.get("anchor_id"),
                jobs=record.get("jobs"), repeat=record.get("repeat"),
                effective_jobs=record.get("effective_jobs"), backend=record.get("backend"),
                worker_pids=record.get("worker_pids"), wall_seconds=record.get("wall_seconds"),
                cpu_seconds=record.get("cpu_seconds"),
                aggregate_cpu_percent=record.get("aggregate_cpu_percent"),
                peak_aggregate_pss_bytes=record.get("peak_aggregate_pss_bytes"),
                peak_aggregate_rss_bytes=record.get("peak_aggregate_rss_bytes"),
                result_sha256=record.get("result_sha256"),
                family_sha256=record.get("family_sha256"),
                ranking_sha256=record.get("ranking_sha256"),
                exact_result_identity=summary.get("exact_result_hash_identity"),
                exact_family_identity=summary.get("exact_family_hash_identity"),
                exact_ranking_identity=summary.get("exact_ranking_hash_identity"),
                speedup_vs_serial=job.get("speedup_vs_serial"),
                parallel_efficiency=job.get("parallel_efficiency"),
                parallel_comparable=job.get("parallel_comparable"),
                parallel_exclusion_reasons=job.get("parallel_exclusion_reasons"),
                native_thread_contract_valid=summary.get(
                    "native_thread_contract_valid"
                ), parallel_execution_contract_valid=summary.get(
                    "parallel_execution_contract_valid"
                )))
    return rows or [_empty(name, common, anchor=anchor.get("anchor_id"))]


def _application_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    name = "cross_species_application.tsv"
    common = _common(payload)
    records = payload.get("rows", payload.get("application_rows"))
    if common["failed"] and not isinstance(records, list):
        return [_empty(name, common, analysis_id=payload["scenario_id"],
            status="FAILED", repair_required=True,
            repair_reason=common["failure_type"] or "application export failed")]
    if not isinstance(records, list):
        raise BenchmarkAggregateError("application rows are missing")
    output = []
    application_fields = [field for field in TABLE_SCHEMAS[name] if field not in _COMMON]
    for record in records:
        if not isinstance(record, Mapping):
            raise BenchmarkAggregateError("application row is invalid")
        output.append(_empty(name, common, **{
            field: record.get(field) for field in application_fields
        }))
    return output


def build_table_rows(evidence: LoadedEvidence) -> dict[str, list[dict[str, Any]]]:
    """Build strict-schema rows from already validated immutable evidence."""

    tables = {name: [] for name in TABLE_SCHEMAS if name != "benchmark_acceptance.tsv"}
    registry = {row.scenario_id: row for row in evidence.registry}
    context_by_scenario: dict[str, str] = {}
    family_by_scenario: dict[str, str] = {}
    for _path, payload in evidence.shards:
        scenario = registry[payload["scenario_id"]]
        context = payload["context_fingerprint"]
        previous_context = context_by_scenario.setdefault(scenario.scenario_id, context)
        if previous_context != context:
            raise BenchmarkAggregateError("scenario context fingerprint changed")
        if payload["track"] == "fit":
            name, rows = _fit_rows(payload, scenario)
            tables[name].extend(rows)
        elif payload["track"] == "omnib":
            family = _sha256_hex(payload.get("family_hash"), "family hash")
            previous_family = family_by_scenario.setdefault(scenario.scenario_id, family)
            if previous_family != family:
                raise BenchmarkAggregateError("scenario family hash changed")
            for name, rows in _omnib_rows(payload, scenario).items():
                tables[name].extend(rows)
        elif payload["track"] == "scaling":
            tables["scaling_runs.tsv"].extend(_scaling_rows(payload))
        else:
            tables["cross_species_application.tsv"].extend(_application_rows(payload))
    def numeric(value: Any) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else -1

    for rows in tables.values():
        rows.sort(key=lambda row: (
            str(row.get("track", "")), str(row.get("scenario_id", "")),
            numeric(row.get("replicate")), str(row.get("method", "")),
            str(row.get("subgenome", "")), numeric(row.get("response_index")),
            numeric(row.get("jobs")), numeric(row.get("repeat")),
            str(row.get("analysis_id", "")),
        ))
    return tables


def write_tables(
    evidence: LoadedEvidence,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    acceptance_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Path, ...], dict[str, str]]:
    table_root = evidence.root / "tables"
    paths: list[Path] = []
    hashes: dict[str, str] = {}
    all_rows = dict(rows)
    all_rows["benchmark_acceptance.tsv"] = list(acceptance_rows)
    for name, fields in table_schemas(evidence.stage).items():
        path = table_root / name
        projected = [
            {field: row.get(field) for field in fields}
            for row in all_rows.get(name, [])
        ]
        _atomic_tsv(path, fields, projected)
        paths.append(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return tuple(paths), hashes


def aggregate_benchmark(root: str | Path) -> AggregateResult:
    """Validate shards and write deterministic evidence tables.

    Acceptance rows are produced by :func:`audit_benchmark`; importing it here
    keeps aggregation usable in isolation while avoiding a module-level cycle.
    """

    from .audit import audit_benchmark

    report = audit_benchmark(root)
    return AggregateResult(report.table_paths, report.table_sha256, report.evidence)
