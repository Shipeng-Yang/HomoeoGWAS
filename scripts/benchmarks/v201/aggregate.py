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

TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "fit_pve_recovery.tsv": _COMMON + (
        "panel", "allocation", "subgenome", "true_pve", "estimated_pve",
        "bias", "optimizer_status", "dominant_correct", "boundary_fit",
    ),
    "fit_pve_coverage.tsv": _COMMON + (
        "panel", "allocation", "subgenome", "target_pve", "interval_low", "interval_high",
        "covered", "interval_width", "bootstrap_failures", "successes", "total",
        "estimate", "ci_low", "ci_high", "failures",
    ),
    "fit_scan_metrics.tsv": _COMMON + (
        "panel", "scan_arm", "method", "scan_pve", "placement", "minimum_p",
        "rejected", "causal_detected", "lead_distance", "localization_correct",
        "lambda_gc", "successes", "total", "estimate", "ci_low",
        "ci_high", "failures",
    ),
    "omnib_null_replicates.tsv": _COMMON + (
        "method", "null_model", "stress", "bank_role", "response_index",
        "minimum_p", "threshold", "rejected", "family_size", "family_hash",
        "family_order_hash", "response_seed_id", "calibration_seed_id",
        "successes", "total", "estimate", "ci_low", "ci_high",
        "failures",
    ),
    "omnib_power_replicates.tsv": _COMMON + (
        "method", "architecture", "interaction_pve", "causal_groups",
        "response_index", "detected", "recall", "threshold", "family_hash",
        "calibration_response_hash", "target_response_hash",
        "null_calibration_gate", "eligible_for_power_summary", "successes", "total",
        "estimate", "ci_low", "ci_high", "failures",
    ),
    "omnib_encoding_robustness.tsv": _COMMON + (
        "perturbation", "check_kind", "required", "status",
        "observed_arrays_identical", "adjusted_decisions_identical",
        "ranking_hash_identical", "rejection_sets_identical", "rank_correlation",
        "top_k_jaccard", "non_estimable_rate", "absolute_power_regret",
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
        "exact_family_identity", "exact_ranking_identity",
    ),
    "cross_species_application.tsv": _COMMON + (
        "analysis_id", "species", "panel", "ploidy", "subgenomes", "sample_count",
        "marker_count", "group_family_count", "edge_family_count", "primary_unit",
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

_HEX = set("0123456789abcdef")
_REGISTRY_HEADER = (
    "scenario_id", "track", "stage", "replicates", "bootstrap_B", "parameters",
)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite constant {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise BenchmarkAggregateError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise BenchmarkAggregateError(f"JSON evidence is not an object: {path}")
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
        try:
            parameters = json.loads(raw["parameters"])
            replicates = int(raw["replicates"])
            bootstrap_b = int(raw["bootstrap_B"])
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise BenchmarkAggregateError("invalid scenario registry row") from error
        if (
            not raw["scenario_id"] or raw["track"] not in {"fit", "omnib", "scaling", "application"}
            or raw["stage"] not in {"pilot", "formal"} or replicates < 1
            or bootstrap_b < 0 or not isinstance(parameters, dict)
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
    if row.track in {"scaling", "application"} or experiment == "conditional":
        return range(1)
    return range(row.replicates)


def _file_sha256(path: Path, label: str) -> str:
    try:
        if not path.is_file() or path.is_symlink():
            raise OSError("not a regular file")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise BenchmarkAggregateError(f"{label} is missing or unreadable") from error


def _validate_config_manifest(root: Path, path: Path) -> dict[str, str]:
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
        if not relative or relative in hashes:
            raise BenchmarkAggregateError("config manifest has invalid or duplicate paths")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise BenchmarkAggregateError("config manifest path escapes benchmark root") from error
        digest = _sha256_hex(row["sha256"], "config hash")
        if _file_sha256(candidate, "declared config") != digest:
            raise BenchmarkAggregateError("declared config hash mismatch")
        hashes[relative] = digest
    return hashes


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
    software = lock.get("software")
    if not isinstance(software, Mapping):
        raise BenchmarkAggregateError("design software identity is missing")
    from homoeogwas import __version__

    if software.get("version") != __version__ or software.get("git_commit") != _current_git_commit():
        raise BenchmarkAggregateError("design software version/commit mismatch")
    locked_files = {
        "input_manifest_sha256": root / "inputs" / "manifest.tsv",
        "comparator_preflight_sha256": root / "inputs" / "comparator_preflight.tsv",
        "config_manifest_sha256": root / "configs" / "manifest.tsv",
    }
    for field, path in locked_files.items():
        digest = _file_sha256(path, field)
        if lock.get(field) != digest:
            raise BenchmarkAggregateError(f"{field} mismatch")
    config_hashes = _validate_config_manifest(root, locked_files["config_manifest_sha256"])
    if lock.get("config_hashes") != config_hashes:
        raise BenchmarkAggregateError("design config hashes mismatch")
    if lock.get("acceptance_rules") != ACCEPTANCE_RULES:
        raise BenchmarkAggregateError("design acceptance rules mismatch")


def load_evidence(root: str | Path) -> LoadedEvidence:
    """Load and validate a complete immutable evidence set in canonical order."""

    benchmark_root = Path(root).resolve()
    lock_path = benchmark_root / "design_lock.json"
    registry_path = benchmark_root / "scenario_registry.tsv"
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
    discovered = sorted(
        [
            path
            for candidate_stage in ("pilot", "formal")
            for path in (benchmark_root / candidate_stage).rglob("replicate-*.json")
        ],
        key=lambda path: path.relative_to(benchmark_root).as_posix(),
    )
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
        if key not in expected or path.resolve() != expected_paths[key].resolve():
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
                lambda_gc=fwer.get("lambda_gc")))
    return name, rows or [_empty(name, common, panel=panel,
        scan_arm=payload.get("scan_arm", experiment),
        scan_pve=registry.parameters.get("scan_pve"),
        placement=registry.parameters.get("placement"))]


def _omnib_rows(payload: Mapping[str, Any], registry: RegistryRow) -> dict[str, list[dict[str, Any]]]:
    output = {name: [] for name in TABLE_SCHEMAS if name.startswith("omnib_")}
    common = _common(payload)
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
        decisions = payload.get("adjusted_decisions")
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
            threshold=(payload.get("bootstrap_minp") or {}).get("threshold"),
            rejected=rejected, family_size=len(family_ids), family_hash=family_hash,
            family_order_hash=order_hash,
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
        for method in sorted(matrices):
            matrix = matrices[method]
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
                    family_order_hash=order_hash, response_seed_id=seed_ids[index]))
            tested = bank.get("tested_family_hashes", {}).get(method)
            output[manifest].append(_empty(manifest, common, method=method,
                hypothesis_unit="group", family_scope="primary_only",
                family_size=len(matrix), ordered_ids=bank.get("family_ids"),
                family_hash=family_hash, family_order_hash=order_hash,
                tested_family_hash=tested, callable=not common["failed"],
                requested_jobs=bank.get("requested_jobs"),
                effective_jobs=bank.get("effective_jobs"),
                backend=bank.get("parallel_backend"), worker_pids=bank.get("worker_pids")))
    elif experiment == "power":
        name = "omnib_power_replicates.tsv"
        decisions = payload.get("causal_detection_by_method")
        if common["failed"] and not isinstance(decisions, Mapping):
            output[name].append(_empty(name, common, method="group_omniB",
                architecture=payload.get("architecture", registry.parameters.get("architecture")),
                interaction_pve=payload.get(
                    "interaction_pve", registry.parameters.get("interaction_pve")),
                causal_groups=registry.parameters.get("causal_groups"),
                response_index=payload["replicate"], family_hash=family_hash))
            return output
        if not isinstance(decisions, Mapping):
            raise BenchmarkAggregateError("power decision evidence is missing")
        thresholds = payload.get("thresholds", {})
        for method in sorted(decisions):
            method_decisions = decisions[method]
            if not isinstance(method_decisions, list):
                raise BenchmarkAggregateError("power decisions are invalid")
            for index, detected in enumerate(method_decisions):
                output[name].append(_empty(name, common, method=method,
                    architecture=payload.get("architecture"),
                    interaction_pve=payload.get("interaction_pve"),
                    causal_groups=len(payload.get("causal_group_ids", [])),
                    response_index=index, detected=detected,
                    recall=(payload.get("recall_by_method", {}).get(method) or [None])[index],
                    threshold=thresholds.get(method), family_hash=family_hash,
                    calibration_response_hash=payload.get("calibration_response_hash"),
                    target_response_hash=payload.get("target_response_hash")))
    elif experiment == "encoding":
        name = "omnib_encoding_robustness.tsv"
        for kind, checks in (("exact", payload.get("exact_checks", {})),
                             ("robustness", payload.get("robustness_checks", {}))):
            for perturbation in sorted(checks):
                record = checks[perturbation]
                output[name].append(_empty(name, common, perturbation=perturbation,
                    check_kind=kind, required=record.get("required"),
                    status=record.get("status"),
                    observed_arrays_identical=record.get("observed_arrays_identical"),
                    adjusted_decisions_identical=record.get("adjusted_decisions_identical"),
                    ranking_hash_identical=record.get("ranking_hash_identical"),
                    rejection_sets_identical=record.get("rejection_sets_identical"),
                    rank_correlation=record.get("rank_correlation"),
                    top_k_jaccard=record.get("top_k_jaccard"),
                    non_estimable_rate=record.get("non_estimable_rate"),
                    absolute_power_regret=record.get("absolute_power_regret")))
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
                exact_ranking_identity=summary.get("exact_ranking_hash_identity")))
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
    for name, fields in TABLE_SCHEMAS.items():
        path = table_root / name
        _atomic_tsv(path, fields, all_rows.get(name, []))
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
