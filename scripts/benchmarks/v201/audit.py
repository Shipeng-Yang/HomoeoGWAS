"""Independent, fail-closed acceptance audit for v2.0.1 benchmark evidence."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from homoeogwas.diagnostics import NestedREMLComparison, boundary_lrt

from .aggregate import (
    TABLE_SCHEMAS,
    BenchmarkAggregateError,
    LoadedEvidence,
    build_table_rows,
    load_evidence,
    table_schemas,
    write_tables,
)
from .comparators import METHOD_NAMES
from .contracts import ScalingAnchor, canonical_json, derive_seed, sha256_payload
from .track_scaling import ScalingAnchorRun, summarize_anchor

_APPLICATION_ACCEPTED = {
    "AUDIT_COMPLETE", "INTERNAL_DISCOVERY_REPLICATION_REQUIRED", "REVIEW_REQUIRED",
}
_APPLICATION_FAILURE_STATUSES = {
    "ANALYSIS_INVALID", "MISSING_AUTHORITATIVE_OUTPUT",
    "UNREADABLE_AUTHORITATIVE_OUTPUT", "UNDECLARED_AUTHORITATIVE_INVENTORY",
    "AMBIGUOUS_AUTHORITATIVE_INVENTORY", "AUTHORITATIVE_PATH_ESCAPE",
    "AUTHORITATIVE_ARTIFACT_MISMATCH", "SCHEMA_INCOMPLETE", "SEMANTIC_MISMATCH",
    "REGISTRY_MISMATCH", "UNSUPPORTED_ANALYSIS_SHAPE",
}
_APPLICATION_ROW_FIELDS = {
    "analysis_id", "analysis_shape", "panel", "species", "ploidy", "subgenomes",
    "sample_count", "marker_count", "group_family_count", "edge_family_count",
    "requested_jobs", "effective_jobs", "backend", "worker_pids", "primary_unit",
    "calibration_method", "calibration_B", "adjusted_discovery_count",
    "adjusted_discoveries", "negative_result", "component_driver_distribution",
    "audit_status", "family_hash", "limitations", "status", "repair_required",
    "repair_reason",
}
_APPLICATION_SPECIES = {
    "wheat": {"wheat", "triticum aestivum"},
    "cotton": {"cotton", "gossypium hirsutum"},
    "rapeseed": {"rapeseed", "brassica napus"},
    "peanut": {"peanut", "arachis hypogaea"},
}

_ENCODING_EXACT_CHECKS = {
    "allele_flip_25pct", "allele_flip_50pct", "allele_flip_100pct",
    "within_gene_snp_column_permutation", "group_row_permutation_restored_ids",
    "serial_vs_admitted_parallel",
}
_ENCODING_ROBUSTNESS_CHECKS = {
    "missingness_2pct", "missingness_5pct", "miscoding_1pct",
    "markers_per_gene_3", "markers_per_gene_10", "markers_per_gene_30",
    "maf_0p01_0p05", "maf_0p05_0p20", "maf_above_0p20",
    "unbalanced_marker_counts",
}
_ENCODING_EXACT_SCHEMA = {
    "status", "required", "observed_arrays_identical",
    "adjusted_decisions_identical", "ranking_hash_identical",
    "rejection_sets_identical", "baseline_observed_hash",
    "candidate_observed_hash", "requested_jobs", "effective_jobs",
    "backend", "worker_pids", "skip_reason",
}
_ENCODING_ROBUSTNESS_SCHEMA_BASE = {
    "status", "error_type", "message",
    "rank_correlation", "top_k", "top_k_jaccard", "non_estimable_rate",
    "note",
    "design_ruling", "calibration", "heldout",
    "realized_marker_design",
}
_ROBUSTNESS_METHODS = ("omnib", "minor_burden", "pc1", "kernel_hadamard")
_CANONICAL_NULL_KIND = {
    "gaussian": "gaussian", "student_t5": "t5", "t5": "t5",
    "heteroscedastic_pc1": "heteroscedastic_pc1",
    "contamination_1pct_6sd": "contamination_1pct_6sd",
    "additive_only": "additive_only", "structure_aligned": "additive_only",
    "omitted_kernel": "omitted_kernel",
}


class BenchmarkAuditError(RuntimeError):
    """Raised when evidence cannot support a reproducible benchmark audit."""


@dataclass(frozen=True)
class AuditGate:
    gate_id: str
    track: str
    scenario_id: str
    gate_kind: str
    successes: int | None
    total: int | None
    estimate: float | None
    lower_ci: float | None
    upper_ci: float | None
    failures: int
    failure_rate: float | None
    passed: bool | None
    qa_passed: bool | None
    status: str
    evidence_path: str
    reason: str


@dataclass(frozen=True)
class AuditReport:
    stage: str
    inference_status: str
    formal_overall_passed: bool | None
    qa_overall_passed: bool | None
    gates: Mapping[str, AuditGate]
    shard_manifest: tuple[Mapping[str, str], ...]
    table_paths: tuple[Path, ...]
    table_sha256: Mapping[str, str]
    evidence: LoadedEvidence


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the standard two-sided 95% Wilson score interval."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("binomial counts must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("binomial counts must be integers")
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def summarize_binomial(successes: int, total: int) -> dict[str, int | float]:
    low, high = wilson_interval(successes, total)
    return {
        "successes": int(successes), "total": int(total),
        "estimate": float(successes / total), "ci_low": float(low),
        "ci_high": float(high),
    }


def core_fwer_gate(
    scenario_id: str,
    rejections: int,
    total: int,
    failures: int,
    *,
    stage: str,
    evidence_path: str,
) -> AuditGate:
    summary = summarize_binomial(rejections, total)
    if isinstance(failures, bool) or not isinstance(failures, int) or not 0 <= failures <= total:
        raise ValueError("invalid failed-replicate count")
    failure_rate = failures / total
    scientific_pass = summary["ci_high"] <= 0.075 and failure_rate <= 0.01
    if stage == "formal":
        passed, qa_passed = scientific_pass, None
        status = "PASS" if scientific_pass else "FAIL"
    else:
        # Pilot output is schema/runtime QA only.  The formal Wilson acceptance
        # threshold is reported numerically but is never applied as a pilot gate.
        qa_gate = failure_rate <= 0.01
        passed, qa_passed = None, qa_gate
        status = "QA_PASS" if qa_gate else "QA_FAIL"
    return AuditGate(
        f"B.core_fwer.{scenario_id}", "omnib", scenario_id, "core_fwer",
        rejections, total, float(summary["estimate"]), float(summary["ci_low"]),
        float(summary["ci_high"]), failures, failure_rate, passed, qa_passed,
        status, evidence_path,
        (
            "FWER upper Wilson <= 0.075 and failed replicates / declared replicates <= 0.01"
            if stage == "formal"
            else "QA only: failed replicates / declared replicates <= 0.01; Wilson interval is descriptive"
        ),
    )


def _strict_old_audit(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    def reject(value: str) -> None:
        raise ValueError(value)

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
        raise BenchmarkAuditError("existing benchmark audit is unreadable") from error
    if not isinstance(value, dict):
        raise BenchmarkAuditError("existing benchmark audit is invalid")
    if encoded != canonical_json(value) + "\n":
        raise BenchmarkAuditError("existing benchmark audit is not canonical")
    return value


def _shard_manifest(evidence: LoadedEvidence) -> tuple[dict[str, str], ...]:
    return tuple({
        "path": path.relative_to(evidence.root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path, _payload in evidence.shards)


def _verify_old_seal(
    old: Mapping[str, Any] | None,
    manifest: Sequence[Mapping[str, str]],
    evidence: LoadedEvidence,
) -> None:
    if old is None:
        return
    declared_self_hash = old.get("audit_sha256")
    if (
        not isinstance(declared_self_hash, str)
        or declared_self_hash != sha256_payload({
            key: value for key, value in old.items() if key != "audit_sha256"
        })
    ):
        raise BenchmarkAuditError("existing audit self-hash mismatch")
    old_manifest = old.get("shard_manifest")
    if old_manifest != list(manifest):
        raise BenchmarkAuditError("shard hash mismatch against first successful audit seal")
    if (
        old.get("schema") != "homoeogwas-v201-benchmark-audit-v1"
        or old.get("design_hash") != evidence.design_hash
        or old.get("scenario_registry_sha256") != evidence.registry_sha256
    ):
        raise BenchmarkAuditError("existing audit design/registry seal mismatch")
    table_hashes = old.get("table_sha256")
    if not isinstance(table_hashes, Mapping) or set(table_hashes) != set(TABLE_SCHEMAS):
        raise BenchmarkAuditError("existing audit table seal is incomplete")
    for name, digest in table_hashes.items():
        path = evidence.root / "tables" / name
        if path.is_symlink() or _file_digest(path) != digest:
            raise BenchmarkAuditError("aggregate table hash mismatch against sealed audit")


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    except OSError:
        return None


def _failed(payload: Mapping[str, Any]) -> bool:
    failure = payload.get("failure")
    if not isinstance(failure, Mapping) or not isinstance(failure.get("failed"), bool):
        raise BenchmarkAuditError("missing or invalid failure state")
    return bool(failure["failed"])


def _finite_probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkAuditError(f"{label} is not a probability")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise BenchmarkAuditError(f"{label} is not a probability")
    return numeric


def _self_hash(record: Any, label: str) -> None:
    if record is None:
        return
    if not isinstance(record, Mapping):
        raise BenchmarkAuditError(f"{label} is not a manifest")
    declared = record.get("sha256")
    manifest = {key: value for key, value in record.items() if key != "sha256"}
    if declared != sha256_payload(manifest):
        raise BenchmarkAuditError(f"{label} hash mismatch")


def _audit_scan_fwer(fwer: Any, truth: Any, *, formal: bool) -> dict[str, Any]:
    if not isinstance(fwer, Mapping) or not isinstance(truth, Mapping):
        raise BenchmarkAuditError("scan FWER/truth evidence is missing")
    ordered = fwer.get("ordered_family")
    raw, adjusted, rejected = (
        fwer.get("raw_p"), fwer.get("adjusted_p"), fwer.get("rejected")
    )
    if (
        not isinstance(ordered, list) or not ordered
        or not all(isinstance(value, Mapping) for value in ordered)
        or not all(isinstance(value, Mapping) for value in (raw, adjusted, rejected))
        or fwer.get("distance_unit") != "bp"
        or fwer.get("family_size") != len(ordered)
        or fwer.get("planned_count") != len(ordered)
        or fwer.get("ordered_family_hash") != sha256_payload(ordered)
    ):
        raise BenchmarkAuditError("scan ordered family contract is invalid")
    if formal and any(
        isinstance(item.get("position_bp"), bool)
        or not isinstance(item.get("position_bp"), int)
        or item["position_bp"] < 0 for item in ordered
    ):
        raise BenchmarkAuditError("formal scan lacks explicit bp positions")
    names = list(raw)
    if set(adjusted) != set(names) or set(rejected) != set(names):
        raise BenchmarkAuditError("scan FWER subgenomes differ")
    flattened: list[tuple[str, str, int, float]] = []
    alpha = _finite_probability(fwer.get("alpha"), "scan alpha")
    family_size = len(ordered)
    cursor = 0
    for name in names:
        raw_values, adj_values, decisions = raw[name], adjusted[name], rejected[name]
        if not all(isinstance(values, list) for values in (raw_values, adj_values, decisions)):
            raise BenchmarkAuditError("scan FWER arrays are invalid")
        if not len(raw_values) == len(adj_values) == len(decisions):
            raise BenchmarkAuditError("scan FWER arrays differ in length")
        for pvalue, adj, decision in zip(raw_values, adj_values, decisions, strict=True):
            member = ordered[cursor]
            cursor += 1
            if member.get("subgenome") != name:
                raise BenchmarkAuditError("scan family order differs from raw arrays")
            p = _finite_probability(pvalue, "scan raw p")
            expected_adj = min(p * family_size, 1.0)
            if adj != expected_adj or decision is not (expected_adj <= alpha):
                raise BenchmarkAuditError("scan adjusted p/rejection differs from raw p")
            marker = member.get("variant_id")
            position = member.get("position_bp")
            if not isinstance(marker, str) or not marker:
                raise BenchmarkAuditError("scan family variant ID is invalid")
            if isinstance(position, bool) or not isinstance(position, int):
                raise BenchmarkAuditError("scan family position is invalid")
            flattened.append((name, marker, position, p))
    if cursor != family_size or len({item[1] for item in flattened}) != family_size:
        raise BenchmarkAuditError("scan family members are incomplete or duplicate")
    causal = truth.get("causal_variant_ids")
    if not isinstance(causal, list):
        records = truth.get("causal_variants", [])
        causal = [
            record.get("variant_id") for record in records
            if isinstance(record, Mapping)
        ] if isinstance(records, list) else []
    if not isinstance(causal, list) or any(not isinstance(value, str) for value in causal):
        raise BenchmarkAuditError("scan causal truth is invalid")
    causal_set = set(causal)
    rejected_ids: set[str] = set()
    offsets = {name: 0 for name in names}
    for name, marker, _position, _p in flattened:
        index = offsets[name]
        offsets[name] += 1
        if rejected[name][index]:
            rejected_ids.add(marker)
    # Stable family order resolves exact p-value ties.
    lead = min(flattened, key=lambda item: item[3])
    causal_members = [item for item in flattened if item[1] in causal_set]
    lead_distance = (
        min(abs(lead[2] - item[2]) for item in causal_members if item[0] == lead[0])
        if any(item[0] == lead[0] for item in causal_members) else None
    )
    from homoeogwas.interact import lambda_gc

    return {
        "minimum_p": lead[3],
        "rejected": bool(rejected_ids),
        "causal_detected": bool(rejected_ids & causal_set),
        "lead_distance": lead_distance,
        "localization_correct": (
            lead[0] in {item[0] for item in causal_members}
            if causal_set else False
        ),
        "lambda_gc": float(lambda_gc([item[3] for item in flattened])),
    }


def _audit_fit_scenario(
    payload: Mapping[str, Any], scenario: Any, *, preflight_sha256: str | None = None,
) -> bool:
    context = payload.get("fit_context_manifest")
    request = payload.get("request_manifest")
    if not isinstance(context, Mapping) or payload.get("context_fingerprint") != sha256_payload(context):
        raise BenchmarkAuditError("fit context manifest hash mismatch")
    if not isinstance(request, Mapping) or payload.get("request_hash") != sha256_payload(request):
        raise BenchmarkAuditError("fit request manifest hash mismatch")
    expected_request = {
        "design_hash": payload.get("design_hash"),
        "context_fingerprint": payload.get("context_fingerprint"),
        "scenario": scenario.to_dict() if hasattr(scenario, "to_dict") else {
            "scenario_id": scenario.scenario_id, "track": scenario.track,
            "stage": scenario.stage, "replicates": scenario.replicates,
            "bootstrap_B": scenario.bootstrap_B,
            "parameters": dict(scenario.parameters),
        },
        "replicate": payload.get("replicate"), "seed": payload.get("seed"),
        "source": payload.get("result_source"),
    }
    for key, value in expected_request.items():
        if request.get(key) != value:
            raise BenchmarkAuditError(f"fit request {key} binding mismatch")
    for request_key, payload_key in (
        ("released_source_manifest", "released_source_manifest"),
        ("released_request_binding", "released_request_binding"),
        ("released_scan_truth_binding", "released_scan_truth_binding"),
        ("coverage_request_binding", "coverage_request_binding"),
        ("truth_hash", "truth_hash"), ("truth_manifest", "truth_manifest"),
        ("scan_context_fingerprint", "scan_context_fingerprint"),
        ("scan_request_manifest", "scan_request_manifest"),
        ("comparator_preflight", "comparator_preflight_evidence"),
    ):
        if request.get(request_key) != payload.get(payload_key):
            raise BenchmarkAuditError(f"fit request {request_key} binding mismatch")
    for label in (
        "released_source_manifest", "released_request_binding",
        "released_scan_truth_binding", "coverage_request_binding",
    ):
        _self_hash(payload.get(label), label)
    truth_manifest, truth_hash = payload.get("truth_manifest"), payload.get("truth_hash")
    if truth_manifest is not None:
        if not isinstance(truth_manifest, Mapping) or truth_hash != sha256_payload(truth_manifest):
            raise BenchmarkAuditError("fit truth manifest hash mismatch")
        truth = payload.get("truth") or payload.get("scan_truth")
        if not isinstance(truth, Mapping) or truth.get("truth_hash") != truth_hash:
            raise BenchmarkAuditError("fit truth payload differs from request truth")
        if {key: value for key, value in truth.items() if key != "truth_hash"} != dict(truth_manifest):
            raise BenchmarkAuditError("fit truth payload differs from truth manifest")
    failed = _failed(payload)
    if failed:
        return False
    experiment = scenario.parameters.get("experiment")
    if (
        (truth_manifest is None or truth_hash is None)
        and not (experiment == "loco" and scenario.stage == "pilot")
    ):
        raise BenchmarkAuditError("successful fit evidence lacks frozen truth")
    if preflight_sha256 is not None and experiment == "scan":
        preflight = payload.get("comparator_preflight_evidence")
        if (
            payload.get("comparator_preflight_hash") != preflight_sha256
            or not isinstance(preflight, Mapping)
            or preflight.get("expected_hash") != preflight_sha256
            or preflight.get("sha256") != preflight_sha256
        ):
            raise BenchmarkAuditError("fit comparator preflight binding mismatch")
    derived: dict[str, Any] = {}
    if experiment == "recovery":
        true, estimated, bias = payload.get("true_pve"), payload.get("estimated_pve"), payload.get("pve_bias")
        if not all(isinstance(value, Mapping) for value in (true, estimated, bias)) or set(true) != set(estimated) or set(true) != set(bias):
            raise BenchmarkAuditError("fit recovery components are incomplete")
        frozen_target = truth_manifest.get("target_pve") if truth_manifest else None
        if (
            not isinstance(frozen_target, Mapping)
            or dict(true) != dict(frozen_target)
            or payload.get("target_pve") != frozen_target
            or truth_manifest.get("allocation") != scenario.parameters.get("allocation")
        ):
            raise BenchmarkAuditError("fit recovery PVE differs from frozen truth")
        genetic_total = sum(
            float(value) for name, value in frozen_target.items() if name != "e"
        )
        if not math.isclose(
            genetic_total, float(scenario.parameters.get("total_pve", genetic_total)),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise BenchmarkAuditError("fit recovery target differs from scenario")
        for name in true:
            expected = float(estimated[name]) - float(true[name])
            if not math.isfinite(expected) or bias[name] != expected:
                raise BenchmarkAuditError("fit recovery bias differs from PVE values")
        expected_boundary = [
            name for name, value in estimated.items() if float(value) < 1e-3
        ]
        if payload.get("boundary_components") != expected_boundary:
            raise BenchmarkAuditError(
                "fit recovery boundary components differ from estimated PVE"
            )
        genetic = [name for name in true if name != "e"]
        true_max = max(float(true[name]) for name in genetic)
        estimate_max = max(float(estimated[name]) for name in genetic)
        dominant_correct = (
            {
                name for name in genetic if float(true[name]) == true_max
            } == {name for name in genetic if float(estimated[name]) == estimate_max}
            if scenario.parameters.get("allocation") in {"single_dominant", "two_dominant"}
            else None
        )
        true_vector = [float(true[name]) for name in genetic]
        estimate_vector = [float(estimated[name]) for name in genetic]
        spearman_value = (
            float(spearmanr(
                true_vector, estimate_vector,
            ).statistic)
            if len(genetic) > 1 and np.ptp(true_vector) > 0
            and np.ptp(estimate_vector) > 0 else math.nan
        )
        derived = {
            "dominant_correct": dominant_correct,
            "boundary_fit": bool(expected_boundary),
            "rmse": float(math.sqrt(np.mean([
                (float(estimated[name]) - float(true[name])) ** 2
                for name in genetic
            ]))),
            "spearman": spearman_value if math.isfinite(spearman_value) else None,
        }
    elif experiment == "coverage":
        uncertainty, target = payload.get("pve_bootstrap"), payload.get("target_pve")
        if not isinstance(uncertainty, Mapping) or not isinstance(target, Mapping):
            raise BenchmarkAuditError("fit coverage evidence is missing")
        frozen_target = truth_manifest.get("target_pve") if truth_manifest else None
        if (
            not isinstance(frozen_target, Mapping)
            or dict(target) != dict(frozen_target)
            or truth_manifest.get("allocation") != scenario.parameters.get("allocation")
        ):
            raise BenchmarkAuditError("fit coverage target differs from frozen truth")
        genetic_total = sum(
            float(value) for name, value in frozen_target.items() if name != "e"
        )
        if not math.isclose(
            genetic_total, float(scenario.parameters.get("total_pve", genetic_total)),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise BenchmarkAuditError("fit coverage target differs from scenario")
        if (
            uncertainty.get("B_requested") != scenario.bootstrap_B
            or not isinstance(uncertainty.get("B_success"), int)
            or not isinstance(uncertainty.get("B_failed"), int)
            or uncertainty["B_success"] + uncertainty["B_failed"] != scenario.bootstrap_B
        ):
            raise BenchmarkAuditError("fit coverage bootstrap counts differ from registry")
        intervals, coverage = uncertainty.get("intervals"), uncertainty.get("coverage")
        expected = set(target) | {"total_genetic"}
        if not isinstance(intervals, Mapping) or set(intervals) != expected or not isinstance(coverage, Mapping) or set(coverage) != expected:
            raise BenchmarkAuditError("fit coverage intervals/components differ from truth")
        for name, record in intervals.items():
            if not isinstance(record, Mapping):
                raise BenchmarkAuditError("fit coverage interval is invalid")
            low, high = record.get("ci_low"), record.get("ci_high")
            truth_value = sum(float(value) for key, value in target.items() if key != "e") if name == "total_genetic" else float(target[name])
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (low, high)) or not 0 <= low <= high <= 1 or coverage[name] is not (low <= truth_value <= high):
                raise BenchmarkAuditError("fit coverage decision differs from interval/truth")
    elif experiment in {"scan", "loco"}:
        scan_manifest = payload.get("scan_request_manifest")
        if scan_manifest is not None:
            if not isinstance(scan_manifest, Mapping) or scan_manifest.get("sha256") != payload.get("scan_context_fingerprint"):
                raise BenchmarkAuditError("fit scan request manifest is invalid")
            _self_hash(scan_manifest, "scan request manifest")
        if scenario.stage == "formal" and (
            not isinstance(scan_manifest, Mapping)
            or scan_manifest.get("bp_positions_explicit") is not True
        ):
            raise BenchmarkAuditError("formal scan lacks explicit bp positions")
        if experiment == "loco":
            binding = payload.get("released_scan_truth_binding")
            source = payload.get("released_source_manifest")
            released_request = payload.get("released_request_binding")
            if scenario.stage == "formal" and (
                payload.get("result_source") != "homoeogwas_outputs"
                or not isinstance(binding, Mapping)
                or not isinstance(source, Mapping)
                or not isinstance(released_request, Mapping)
                or binding.get("truth_hash") != truth_hash
                or binding.get("released_source_manifest_sha256")
                != source.get("sha256")
                or binding.get("released_request_binding_sha256")
                != released_request.get("sha256")
                or binding.get("distance_unit") != "bp"
                or binding.get("ordered_family")
                != scan_manifest.get("ordered_family")
                or binding.get("ordered_family_hash")
                != scan_manifest.get("ordered_family_hash")
            ):
                raise BenchmarkAuditError("formal LOCO released truth binding is invalid")
            if truth_manifest is not None:
                causal = truth_manifest.get("causal_variants")
                expected_analysis_context = {
                    "analysis_sample_ids_sha256": released_request.get(
                        "analysis_sample_ids_sha256"
                    ),
                    "joined_phenotype": released_request.get("joined_phenotype"),
                    "kernel_order": released_request.get("kernel_order"),
                    "kernel_fingerprints": released_request.get(
                        "kernel_fingerprints"
                    ),
                    "generated_config_sha256": source.get("files", {}).get(
                        "generated_config", {}
                    ).get("sha256"),
                }
                if (
                    truth_manifest.get("scan_pve")
                    != scenario.parameters.get("scan_pve")
                    or truth_manifest.get("distance_unit") != "bp"
                    or not isinstance(causal, list)
                    or (scenario.parameters.get("scan_pve") == 0.0)
                    is not (len(causal) == 0)
                    or truth_manifest.get("analysis_context")
                    != expected_analysis_context
                    or released_request.get("truth_hash") != truth_hash
                ):
                    raise BenchmarkAuditError("LOCO truth differs from scenario")
        comparators = payload.get("comparators")
        if not isinstance(comparators, Mapping) or not comparators:
            raise BenchmarkAuditError("fit scan comparator evidence is missing")
        truth = payload.get("scan_truth") or {"causal_variant_ids": []}
        if experiment == "loco" and truth_manifest is not None:
            truth = {
                **dict(truth),
                "causal_variant_ids": [
                    item["variant_id"]
                    for item in truth_manifest.get("causal_variants", [])
                ],
            }
        derived = {
            method: _audit_scan_fwer(record.get("fwer") if isinstance(record, Mapping) else None, truth, formal=scenario.stage == "formal")
            for method, record in comparators.items()
        }
    else:
        raise BenchmarkAuditError("unknown fit experiment")
    if isinstance(payload, dict):
        payload["audit_derived"] = derived
    return True


def _audit_end2end_decisions(payload: Mapping[str, Any]) -> bool:
    family_ids = payload.get("family_ids")
    observed = payload.get("observed_group_p")
    adjusted = payload.get("adjusted_p")
    adjusted_decisions = payload.get(
        "adjusted_decisions" if payload.get("stage") == "formal"
        else "qa_adjusted_diagnostic_decisions"
    )
    calibration = payload.get("bootstrap_minp")
    null_minima = payload.get("null_minima")
    if (
        not isinstance(family_ids, list) or not isinstance(observed, list)
        or not isinstance(adjusted, list) or not isinstance(adjusted_decisions, list)
        or not isinstance(calibration, Mapping) or not isinstance(null_minima, list)
        or not len(family_ids) == len(observed) == len(adjusted) == len(adjusted_decisions)
    ):
        raise BenchmarkAuditError("end-to-end decision fields are incomplete")
    observed_values = [_finite_probability(value, "observed p") for value in observed]
    adjusted_values = [_finite_probability(value, "adjusted p") for value in adjusted]
    minima = [_finite_probability(value, "bootstrap null minimum") for value in null_minima]
    declared_b = calibration.get("B")
    if (
        isinstance(declared_b, bool) or not isinstance(declared_b, int)
        or declared_b < 1 or declared_b != len(minima)
        or payload.get("bootstrap_B") != declared_b
    ):
        raise BenchmarkAuditError("bootstrap null-minimum count differs from B")
    if not all(isinstance(value, bool) for value in adjusted_decisions):
        raise BenchmarkAuditError("serialized adjusted decisions are invalid")
    alpha = _finite_probability(calibration.get("alpha", 0.05), "alpha")
    k = int(math.floor(alpha * (declared_b + 1)))
    recomputed_threshold = sorted(minima)[k - 1] if k >= 1 else None
    threshold = calibration.get("threshold")
    if threshold != recomputed_threshold:
        raise BenchmarkAuditError("bootstrap threshold differs from null minima")
    threshold_indices: set[int]
    if threshold is None:
        threshold_indices = set()
    else:
        threshold_value = _finite_probability(threshold, "threshold")
        threshold_indices = {
            index for index, value in enumerate(observed_values)
            if value < threshold_value
        }
    adjusted_indices = {
        index for index, value in enumerate(adjusted_values) if value <= alpha
    }
    serialized_indices = {
        index for index, value in enumerate(adjusted_decisions) if value
    }
    rejected_local = calibration.get("rejected_local")
    if (
        not isinstance(rejected_local, list)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in rejected_local)
    ):
        raise BenchmarkAuditError("bootstrap rejection indices are invalid")
    bootstrap_indices = set(rejected_local)
    bootstrap_adjusted = calibration.get("adjusted_p_local")
    recomputed_adjusted = [
        (1 + sum(minimum <= value for minimum in minima)) / (declared_b + 1)
        for value in observed_values
    ]
    empirical_p = (
        1 + sum(minimum <= min(observed_values) for minimum in minima)
    ) / (declared_b + 1)
    if (
        not isinstance(bootstrap_adjusted, list)
        or [float(v) for v in bootstrap_adjusted] != recomputed_adjusted
        or adjusted_values != recomputed_adjusted
        or calibration.get("empirical_p") != empirical_p
        or calibration.get("rejected") is not (empirical_p <= alpha)
    ):
        raise BenchmarkAuditError("bootstrap and serialized adjusted p-values differ")
    expected_ids = {family_ids[index] for index in threshold_indices}
    serialized_ids = payload.get(
        "formal_rejections" if payload["stage"] == "formal"
        else "qa_diagnostic_rejections"
    )
    if not isinstance(serialized_ids, list) or set(serialized_ids) != expected_ids:
        raise BenchmarkAuditError("decision disagreement: serialized rejection IDs differ")
    if payload["stage"] == "pilot" and any(
        field in payload for field in ("formal_rejections", "adjusted_decisions")
    ):
        raise BenchmarkAuditError("pilot contains formal rejection claims")
    if not (
        threshold_indices == adjusted_indices == serialized_indices == bootstrap_indices
    ):
        raise BenchmarkAuditError(
            "decision disagreement among threshold, adjusted-p and serialized decisions"
        )
    return bool(threshold_indices)


def _matrix(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        raise BenchmarkAuditError(f"{label} is not a nonempty matrix")
    width = len(value[0])
    if width < 1 or any(len(row) != width for row in value):
        raise BenchmarkAuditError(f"{label} matrix shape is inconsistent")
    return [[_finite_probability(cell, label) for cell in row] for row in value]


def _validate_bank_envelope(
    bank: Mapping[str, Any], expected: int, role: str,
) -> tuple[list[int], list[str]]:
    if bank.get("canonical_role") != role:
        raise BenchmarkAuditError("conditional bank role mismatch")
    seeds, seed_ids = bank.get("seeds"), bank.get("seed_ids")
    metadata, shape = bank.get("response_metadata"), bank.get("response_shape")
    if (
        not isinstance(seeds, list) or not isinstance(seed_ids, list)
        or not isinstance(metadata, list) or len(seeds) != expected
        or len(seed_ids) != expected or len(metadata) != expected
        or len(set(seeds)) != expected or len(set(seed_ids)) != expected
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or any(not isinstance(seed_id, str) or not seed_id for seed_id in seed_ids)
        or not isinstance(shape, list) or len(shape) != 2
        or isinstance(shape[0], bool) or not isinstance(shape[0], int) or shape[0] < 1
        or shape[1] != expected
    ):
        raise BenchmarkAuditError("conditional bank response count/identity is invalid")
    for index, record in enumerate(metadata):
        if (
            not isinstance(record, Mapping) or record.get("seed") != seeds[index]
            or record.get("seed_id") != seed_ids[index]
        ):
            raise BenchmarkAuditError("conditional response metadata differs from seed order")
    failure = bank.get("failure")
    if not isinstance(failure, Mapping) or not isinstance(failure.get("failed"), bool):
        raise BenchmarkAuditError("conditional bank failure provenance is invalid")
    failed = failure.get("failed_response_indices")
    if (
        not isinstance(failed, list)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in failed)
        or len(failed) != len(set(failed))
        or any(index < 0 or index >= expected for index in failed)
        or failure["failed"] is not bool(failed)
    ):
        raise BenchmarkAuditError("conditional failed response indices are invalid")
    return failed, seed_ids


def _conditional_score_matrices(
    bank: Mapping[str, Any], expected: int, failed: Sequence[int],
) -> dict[str, list[list[float | None]]]:
    scores = bank.get("p_by_method")
    members = bank.get("tested_family_members")
    sizes = bank.get("tested_family_sizes")
    hashes = bank.get("tested_family_hashes")
    score_hashes = bank.get("score_matrix_hashes")
    if not all(isinstance(value, Mapping) for value in (scores, members, sizes, hashes, score_hashes)):
        raise BenchmarkAuditError("conditional tested-family evidence is incomplete")
    locked_methods = set(METHOD_NAMES)
    if not (
        set(scores) == set(members) == set(sizes) == set(hashes)
        == set(score_hashes) == locked_methods
    ):
        raise BenchmarkAuditError("conditional tested-family methods differ")
    family_ids = bank.get("family_ids")
    if (
        not isinstance(family_ids, list) or not family_ids
        or any(not isinstance(item, str) or not item for item in family_ids)
        or len(family_ids) != len(set(family_ids))
    ):
        raise BenchmarkAuditError("conditional group family IDs are invalid")
    failed_set = set(failed)
    output: dict[str, list[list[float | None]]] = {}
    computed_nonfinite: dict[str, list[int]] = {}
    computed_all_nan: dict[str, list[int]] = {}
    for method in sorted(scores):
        matrix = scores[method]
        ordered = members[method]
        if (
            not isinstance(matrix, list) or not matrix
            or not isinstance(ordered, list) or not ordered
            or len(matrix) != len(family_ids) or sizes[method] != len(ordered)
            or any(not isinstance(item, str) or not item for item in ordered)
            or len(ordered) != len(set(ordered))
            or hashes[method] != sha256_payload({
                "method": method, "ordered_member_ids": ordered,
            })
        ):
            raise BenchmarkAuditError("conditional tested-family manifest mismatch")
        checked: list[list[float | None]] = []
        for hypothesis in matrix:
            if not isinstance(hypothesis, list) or len(hypothesis) != expected:
                raise BenchmarkAuditError("conditional score matrix width mismatch")
            checked_row: list[float | None] = []
            for index, value in enumerate(hypothesis):
                if value is None:
                    if index not in failed_set:
                        raise BenchmarkAuditError("undeclared conditional non-finite score")
                    checked_row.append(None)
                else:
                    checked_row.append(_finite_probability(value, "conditional p"))
            checked.append(checked_row)
        output[str(method)] = checked
        if score_hashes[method] != sha256_payload(checked):
            raise BenchmarkAuditError("conditional score matrix hash mismatch")
        if not isinstance(score_hashes[method], str) or len(score_hashes[method]) != 64:
            raise BenchmarkAuditError("conditional score matrix hash is invalid")
        computed_nonfinite[method] = [
            index for index in range(expected)
            if any(row[index] is None for row in checked)
        ]
        computed_all_nan[method] = [
            index for index in range(expected)
            if all(row[index] is None for row in checked)
        ]
    failure = bank.get("failure", {})
    if (
        failure.get("nonfinite_response_indices_by_method") != computed_nonfinite
        or failure.get("all_nan_response_indices_by_method") != computed_all_nan
        or failed_set != {
            index for indices in computed_nonfinite.values() for index in indices
        }
    ):
        raise BenchmarkAuditError("conditional partial-failure matrix evidence differs")
    if bank.get("canonical_role") == "calibration":
        artifact = _validate_snpxsnp_calibration_artifact(
            bank.get("snpxsnp_calibration_artifact"),
            calibration_count=expected,
        )
        if artifact.get("calibration_p_sha256") != bank.get(
            "snpxsnp_calibration_reference", {}
        ).get("calibration_p_sha256"):
            raise BenchmarkAuditError("conditional SNPxSNP calibration reference differs")
    return output


def _empirical_threshold(matrix: Sequence[Sequence[float]], alpha: float = 0.05) -> float | None:
    minima = sorted(min(row[index] for row in matrix) for index in range(len(matrix[0])))
    k = int(math.floor(alpha * (len(minima) + 1)))
    return None if k < 1 else minima[k - 1]


def _conditional_gates(
    evidence: LoadedEvidence,
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, AuditGate]:
    calibration: dict[str, Mapping[str, Any]] = {}
    heldout: dict[str, Mapping[str, Any]] = {}
    failed_by_base: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for _path, payload in evidence.shards:
        if payload["track"] != "omnib" or payload.get("experiment") != "conditional":
            continue
        scenario_id = str(payload["scenario_id"])
        base = scenario_id
        for suffix in (".calibration", ".heldout", ".evaluation"):
            if base.endswith(suffix):
                base = base.removesuffix(suffix)
                break
        bank = payload.get("bank")
        if not isinstance(bank, Mapping):
            if _failed(payload):
                failed_by_base[base].append(payload)
                continue
            raise BenchmarkAuditError("conditional bank is missing")
        role = bank.get("canonical_role")
        if role == "calibration":
            calibration[base] = payload
        elif role == "heldout":
            heldout[base] = payload
        else:
            raise BenchmarkAuditError("conditional bank has invalid role")
    gates: dict[str, AuditGate] = {}
    all_bases = set(calibration) | set(heldout) | set(failed_by_base)
    for base in sorted(all_bases):
        if failed_by_base.get(base):
            declared = [
                row.replicates for row in evidence.registry
                if row.scenario_id.startswith(base)
                and row.parameters.get("bank") in {"heldout", "evaluation"}
            ]
            total = declared[0] if declared else 1
            gate = core_fwer_gate(
                base, 0, total, total, stage=evidence.stage,
                evidence_path="tables/omnib_null_replicates.tsv",
            )
            failed_registry = [
                row for row in evidence.registry
                if row.scenario_id.startswith(base)
            ]
            if failed_registry and all(
                row.parameters.get("stress") is True for row in failed_registry
            ):
                gate = AuditGate(
                    gate.gate_id.replace("core_fwer", "stress_fwer"), gate.track,
                    gate.scenario_id, "stress_fwer", gate.successes, gate.total,
                    gate.estimate, gate.lower_ci, gate.upper_ci, gate.failures,
                    gate.failure_rate, None, None, "DESCRIPTIVE_STRESS",
                    gate.evidence_path, "failed stress bank remains descriptive",
                )
            gates[gate.gate_id] = gate
            continue
        if base not in calibration or base not in heldout:
            raise BenchmarkAuditError("conditional calibration/heldout pair is incomplete")
        cal_payload, held_payload = calibration[base], heldout[base]
        cal_bank, held_bank = cal_payload["bank"], held_payload["bank"]
        cal_registry = next(
            row for row in evidence.registry if row.scenario_id == cal_payload["scenario_id"]
        )
        held_registry = next(
            row for row in evidence.registry if row.scenario_id == held_payload["scenario_id"]
        )
        if cal_registry.replicates != held_registry.replicates:
            raise BenchmarkAuditError("conditional response-bank declarations differ")
        expected = held_registry.replicates
        cal_failed, cal_seed_order = _validate_bank_envelope(
            cal_bank, expected, "calibration",
        )
        held_failed, held_seed_order = _validate_bank_envelope(
            held_bank, expected, "heldout",
        )
        cal_scores = _conditional_score_matrices(cal_bank, expected, cal_failed)
        held_scores = _conditional_score_matrices(held_bank, expected, held_failed)
        declared_null = held_registry.parameters.get("null_model")
        expected_null = _CANONICAL_NULL_KIND.get(str(declared_null))
        if expected_null is None or any(
            not isinstance(metadata, Mapping)
            or metadata.get("kind") != declared_null
            or metadata.get("canonical_kind") != expected_null
            for bank in (cal_bank, held_bank)
            for metadata in bank.get("response_metadata", [])
        ):
            raise BenchmarkAuditError("conditional null metadata differs from registry")
        cal_seeds, held_seeds = set(cal_seed_order), set(held_seed_order)
        reference = held_bank.get("calibration_reference")
        if not isinstance(reference, Mapping):
            raise BenchmarkAuditError("heldout calibration reference is missing")
        if cal_seeds != set(reference.get("seed_ids", [])):
            raise BenchmarkAuditError("heldout calibration seed IDs differ from frozen bank")
        if cal_seeds & held_seeds:
            raise BenchmarkAuditError("seed roles overlap")
        if cal_bank.get("response_hash") != reference.get("response_hash"):
            raise BenchmarkAuditError("heldout calibration response hash differs")
        if cal_bank.get("response_hash") == held_bank.get("response_hash"):
            raise BenchmarkAuditError("calibration and heldout response hashes overlap")
        if (
            cal_payload.get("family_hash") != held_payload.get("family_hash")
            or cal_payload.get("family_ids") != held_payload.get("family_ids")
        ):
            raise BenchmarkAuditError("conditional family differs across response roles")
        if set(cal_scores) != set(held_scores):
            raise BenchmarkAuditError("conditional methods differ across response roles")
        for method in sorted(cal_scores):
            cal_matrix = cal_scores[method]
            held_matrix = held_scores[method]
            if len(cal_matrix) != len(held_matrix):
                raise BenchmarkAuditError("conditional tested family size changed")
            calibration_valid = not cal_failed
            cal_minima = [
                min(value for row in cal_matrix if (value := row[index]) is not None)
                for index in range(expected)
            ] if calibration_valid else []
            if not calibration_valid or not cal_minima:
                threshold = None
            else:
                k = int(math.floor(0.05 * (len(cal_minima) + 1)))
                threshold = None if k < 1 else sorted(cal_minima)[k - 1]
            decisions = [
                False if index in set(held_failed) else (
                    threshold is not None
                    and min(
                        value for row in held_matrix if (value := row[index]) is not None
                    ) < threshold
                )
                for index in range(expected)
            ]
            # Held-out failure rate uses its own fixed denominator. Calibration
            # failures invalidate the frozen threshold rather than being merged
            # by coincident numeric response indices.
            failures = len(held_failed)
            scenario_key = f"{held_payload['scenario_id']}.{method}"
            gate = core_fwer_gate(
                scenario_key, sum(decisions), len(decisions), failures,
                stage=evidence.stage,
                evidence_path="tables/omnib_null_replicates.tsv",
            )
            if not calibration_valid:
                gate = AuditGate(
                    gate.gate_id, gate.track, gate.scenario_id, gate.gate_kind,
                    gate.successes, gate.total, gate.estimate, gate.lower_ci,
                    gate.upper_ci, gate.failures, gate.failure_rate,
                    False if evidence.stage == "formal" else None,
                    False if evidence.stage == "pilot" else None,
                    "FAIL" if evidence.stage == "formal" else "QA_FAIL",
                    gate.evidence_path,
                    "calibration response failure invalidates the frozen threshold",
                )
            if held_registry.parameters.get("stress") is True:
                gate = AuditGate(
                    gate.gate_id.replace("core_fwer", "stress_fwer"), gate.track,
                    gate.scenario_id, "stress_fwer", gate.successes, gate.total,
                    gate.estimate, gate.lower_ci, gate.upper_ci, gate.failures,
                    gate.failure_rate, None, None, "DESCRIPTIVE_STRESS",
                    gate.evidence_path,
                    "predeclared stress null is descriptive and excluded from acceptance",
                )
            gates[gate.gate_id] = gate
            for row in rows["omnib_null_replicates.tsv"]:
                if (
                    evidence.stage == "formal"
                    and row["scenario_id"] == held_payload["scenario_id"]
                    and row["method"] == method
                ):
                    index = int(row["response_index"])
                    row["threshold"] = threshold
                    row["rejected"] = decisions[index]
    return gates


def _audit_seed_roles(evidence: LoadedEvidence) -> None:
    calibration: set[str] = set()
    targets: set[str] = set()

    def add(values: Any, destination: set[str], label: str) -> None:
        if values is None:
            return
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
            raise BenchmarkAuditError(f"invalid {label} seed IDs")
        destination.update(values)

    for _path, payload in evidence.shards:
        add(payload.get("calibration_seed_id"), calibration, "calibration")
        add(payload.get("calibration_seed_ids"), calibration, "calibration")
        add(payload.get("response_seed_id"), targets, "heldout")
        add(payload.get("target_seed_ids"), targets, "target")
        bank = payload.get("bank")
        if isinstance(bank, Mapping):
            role = bank.get("canonical_role")
            add(bank.get("seed_ids"), calibration if role == "calibration" else targets, str(role))
            reference = bank.get("calibration_reference")
            if isinstance(reference, Mapping):
                add(reference.get("seed_ids"), calibration, "calibration reference")
        for name in ("calibration_bank", "target_bank"):
            nested = payload.get(name)
            if isinstance(nested, Mapping):
                role = nested.get("canonical_role")
                add(nested.get("seed_ids"), calibration if role == "calibration" else targets, str(role))
    overlap = calibration & targets
    if overlap:
        raise BenchmarkAuditError(f"seed roles overlap: {sorted(overlap)[0]}")


def _expected_seed_id(
    design_hash: str, scenario_id: str, replicate: int, stage: str, role: str,
) -> tuple[int, str]:
    namespace = f"{stage}:{role}"
    seed = derive_seed(design_hash, "omnib", scenario_id, replicate, namespace)
    return seed, f"{namespace}:{scenario_id}:{replicate}:{seed:016x}"


def _audit_derived_seeds_and_requests(evidence: LoadedEvidence) -> None:
    registry = {row.scenario_id: row for row in evidence.registry}
    for _path, payload in evidence.shards:
        scenario = registry[str(payload["scenario_id"])]
        replicate = int(payload["replicate"])
        if payload["track"] == "fit":
            expected = derive_seed(
                evidence.design_hash, "fit", scenario.scenario_id, replicate,
                evidence.stage,
            )
            if payload.get("seed") != expected:
                raise BenchmarkAuditError("fit seed differs from deterministic derivation")
            continue
        if payload["track"] != "omnib":
            continue
        experiment = scenario.parameters.get("experiment")
        bank = scenario.parameters.get("bank")
        canonical_bank = (
            "heldout" if bank == "evaluation" else bank
        ) if experiment == "conditional" else None
        request = {
            "entrypoint": "run_omnib_replicate",
            "scenario": {
                "scenario_id": scenario.scenario_id, "track": scenario.track,
                "stage": scenario.stage, "replicates": scenario.replicates,
                "bootstrap_B": scenario.bootstrap_B,
                "parameters": dict(scenario.parameters),
            },
            "replicate": replicate, "experiment": experiment,
            "canonical_bank": canonical_bank,
            "n_jobs": payload.get("requested_jobs"),
        }
        if experiment == "power":
            request["calibration_bank_manifest_hash"] = payload.get(
                "calibration_bank_manifest_hash"
            )
        expected_request = sha256_payload({
            "design_hash": evidence.design_hash,
            "context_fingerprint": payload["context_fingerprint"],
            "request": request,
        })
        if payload.get("request_hash") != expected_request:
            raise BenchmarkAuditError("omniB request hash mismatch")
        if experiment == "end2end":
            if _failed(payload) and not any(
                field in payload for field in (
                    "response_seed", "response_seed_id", "calibration_seed",
                    "calibration_seed_id",
                )
            ):
                continue
            for role, seed_field, id_field in (
                ("heldout", "response_seed", "response_seed_id"),
                ("calibration", "calibration_seed", "calibration_seed_id"),
            ):
                seed, seed_id = _expected_seed_id(
                    evidence.design_hash, scenario.scenario_id, replicate,
                    evidence.stage, role,
                )
                if payload.get(seed_field) != seed or payload.get(id_field) != seed_id:
                    raise BenchmarkAuditError(f"{role} seed differs from deterministic derivation")

        nested_banks = []
        if isinstance(payload.get("bank"), Mapping):
            nested_banks.append(payload["bank"])
        for field in ("calibration_bank", "target_bank"):
            if isinstance(payload.get(field), Mapping):
                nested_banks.append(payload[field])
        for nested in nested_banks:
            role = nested.get("canonical_role")
            seed_ids, seeds = nested.get("seed_ids"), nested.get("seeds")
            if role not in {"calibration", "heldout", "power"}:
                raise BenchmarkAuditError("invalid serialized seed role")
            if not isinstance(seed_ids, list) or not isinstance(seeds, list) or len(seed_ids) != len(seeds):
                raise BenchmarkAuditError("serialized seeds are incomplete")
            for seed_id, seed in zip(seed_ids, seeds, strict=True):
                parts = str(seed_id).split(":")
                if len(parts) < 5 or parts[0] != evidence.stage or parts[1] != role:
                    raise BenchmarkAuditError("serialized seed ID is malformed")
                try:
                    nested_replicate = int(parts[-2])
                except ValueError as error:
                    raise BenchmarkAuditError("serialized seed replicate is malformed") from error
                nested_scenario = ":".join(parts[2:-2])
                expected_seed, expected_id = _expected_seed_id(
                    evidence.design_hash, nested_scenario, nested_replicate,
                    evidence.stage, role,
                )
                if seed != expected_seed or seed_id != expected_id:
                    raise BenchmarkAuditError("serialized seed differs from deterministic derivation")


def _compact_vector(
    value: Any, expected: int, failed: set[int], label: str,
) -> list[float | None]:
    if not isinstance(value, list) or len(value) != expected:
        raise BenchmarkAuditError(f"{label} response count mismatch")
    checked: list[float | None] = []
    for index, item in enumerate(value):
        if item is None:
            if index not in failed:
                raise BenchmarkAuditError(f"{label} has undeclared missing evidence")
            checked.append(None)
        else:
            checked.append(_finite_probability(item, label))
    return checked


def _index_set(value: Any, expected: int, label: str) -> set[int]:
    if (
        not isinstance(value, list)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in value)
        or len(value) != len(set(value))
        or any(index < 0 or index >= expected for index in value)
    ):
        raise BenchmarkAuditError(f"invalid {label} failure indices")
    return set(value)


def _validate_snpxsnp_calibration_reference(
    value: Any, *, calibration_count: int,
) -> Mapping[str, Any]:
    expected_fields = {
        "schema", "hypothesis_unit", "member_ids", "member_ids_sha256", "group_memberships",
        "calibration_shape", "calibration_p_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise BenchmarkAuditError("SNPxSNP frozen calibration reference is invalid")
    members = value.get("member_ids")
    memberships = value.get("group_memberships")
    shape = value.get("calibration_shape")
    if (
        value.get("schema") != "snpxsnp_calibration_v1"
        or value.get("hypothesis_unit") != "snp_pair_within_group"
        or not isinstance(members, list) or not members
        or any(not isinstance(item, str) or not item for item in members)
        or len(set(members)) != len(members)
        or value.get("member_ids_sha256") != sha256_payload(members)
        or not isinstance(memberships, list) or len(memberships) != len(members)
        or any(
            not isinstance(item, list) or not item
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0
                   for index in item)
            for item in memberships
        )
        or shape != [len(members), calibration_count]
        or not isinstance(value.get("calibration_p_sha256"), str)
        or len(value["calibration_p_sha256"]) != 64
        or any(character not in "0123456789abcdef"
               for character in value["calibration_p_sha256"])
    ):
        raise BenchmarkAuditError("SNPxSNP frozen calibration reference is invalid")
    return value


def _numeric_array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_snpxsnp_calibration_artifact(
    value: Any, *, calibration_count: int,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "hypothesis_unit", "member_ids", "member_ids_sha256", "group_memberships",
        "calibration_shape", "calibration_p_sha256", "calibration_p",
        "artifact_sha256",
    }:
        raise BenchmarkAuditError("SNPxSNP frozen calibration artifact is invalid")
    reference = {
        key: value[key] for key in (
            "schema", "hypothesis_unit", "member_ids", "member_ids_sha256", "group_memberships",
            "calibration_shape", "calibration_p_sha256",
        )
    }
    _validate_snpxsnp_calibration_reference(
        reference, calibration_count=calibration_count,
    )
    matrix = value.get("calibration_p")
    if (
        not isinstance(matrix, list) or len(matrix) != len(reference["member_ids"])
        or any(not isinstance(row, list) or len(row) != calibration_count for row in matrix)
    ):
        raise BenchmarkAuditError("SNPxSNP frozen calibration artifact is invalid")
    numeric = np.asarray(matrix, dtype=float)
    if (
        not np.all(np.isfinite(numeric))
        or np.any((numeric < 0.0) | (numeric > 1.0))
        or reference["calibration_p_sha256"] != _numeric_array_hash(numeric)
        or value.get("artifact_sha256") != sha256_payload({
            key: item for key, item in value.items() if key != "artifact_sha256"
        })
    ):
        raise BenchmarkAuditError("SNPxSNP frozen calibration artifact is invalid")
    return value


def _audit_power_evidence(payload: Mapping[str, Any], scenario: Any) -> None:
    negative_control = scenario.parameters.get("control_type") == "negative"
    if payload.get("stage") == "pilot" and any(
        field in payload for field in (
            "thresholds", "rejections_by_method", "causal_detection_by_method",
            "recall_by_method", "adjusted_decisions", "formal_rejections",
            "false_positive_by_method", "specificity_by_method",
        )
    ):
        raise BenchmarkAuditError("pilot power payload contains formal decision fields")
    if payload.get("stage") == "formal" and any(
        field in payload for field in (
            "qa_cutoffs_by_method", "qa_rejections_by_method",
            "qa_causal_detection_by_method", "qa_recall_by_method",
            "qa_false_positive_by_method", "qa_specificity_by_method",
        )
    ):
        raise BenchmarkAuditError("formal power payload contains QA decision fields")
    calibration_count = scenario.parameters.get("calibration_count")
    if isinstance(calibration_count, bool) or not isinstance(calibration_count, int):
        raise BenchmarkAuditError("power calibration count is invalid")
    if (
        scenario.parameters.get("response_count") != 1
        or payload.get("architecture") != scenario.parameters.get("architecture")
        or payload.get("interaction_pve") != scenario.parameters.get("interaction_pve")
        or payload.get("calibration_scenario_id")
        != scenario.parameters.get("calibration_scenario_id")
        or scenario.parameters.get("null_model") != "gaussian"
    ):
        raise BenchmarkAuditError("power request metadata differs from registry")
    cal_ids, target_ids = payload.get("calibration_response_ids"), payload.get("target_response_ids")
    if (
        not isinstance(cal_ids, list) or len(cal_ids) != calibration_count
        or cal_ids != payload.get("calibration_seed_ids")
        or not isinstance(target_ids, list) or len(target_ids) != 1
        or target_ids != payload.get("target_seed_ids")
        or len(set(cal_ids)) != len(cal_ids) or len(set(target_ids)) != len(target_ids)
        or set(cal_ids) & set(target_ids)
    ):
        raise BenchmarkAuditError("power response IDs are incomplete or overlap")
    response_count = len(target_ids)
    cal_failed = _index_set(
        payload.get("failed_calibration_response_indices"), calibration_count,
        "power calibration",
    )
    target_failed = _index_set(
        payload.get("failed_target_response_indices"), response_count,
        "power target",
    )
    names = [
        "calibration_minima_by_method", "target_minima_by_method",
        "calibration_minima_hashes", "target_minima_hashes",
    ]
    cutoff_name = "thresholds" if payload.get("stage") == "formal" else "qa_cutoffs_by_method"
    rejection_name = (
        "rejections_by_method"
        if payload.get("stage") == "formal" else "qa_rejections_by_method"
    )
    names += [cutoff_name, rejection_name]
    causal_detection_name = (
        "causal_detection_by_method"
        if payload.get("stage") == "formal" else "qa_causal_detection_by_method"
    )
    recall_name = (
        "recall_by_method"
        if payload.get("stage") == "formal" else "qa_recall_by_method"
    )
    false_positive_name = (
        "false_positive_by_method"
        if payload.get("stage") == "formal" else "qa_false_positive_by_method"
    )
    specificity_name = (
        "specificity_by_method"
        if payload.get("stage") == "formal" else "qa_specificity_by_method"
    )
    names += (
        [false_positive_name, specificity_name]
        if negative_control else [
            "causal_minima_by_method", "causal_minima_hashes",
            causal_detection_name, recall_name,
        ]
    )
    maps = {name: payload.get(name) for name in names}
    if not all(isinstance(value, Mapping) for value in maps.values()):
        raise BenchmarkAuditError("power compact score evidence is incomplete")
    methods = set(maps[cutoff_name])
    if methods != set(METHOD_NAMES) or any(
        set(value) != methods for value in maps.values()
    ):
        raise BenchmarkAuditError("power compact score methods differ")
    causal_ids = payload.get("causal_group_ids")
    if (
        not isinstance(causal_ids, list)
        or len(causal_ids) != len(set(causal_ids))
        or len(causal_ids) != (
            0 if negative_control else scenario.parameters.get("causal_groups")
        )
    ):
        raise BenchmarkAuditError("power causal group IDs are invalid")
    calibration_bank = payload.get("calibration_bank")
    target_bank = payload.get("target_bank")
    artifact = payload.get("calibration_artifact")
    if (
        not isinstance(calibration_bank, Mapping)
        or payload.get("calibration_bank_manifest_hash")
        != sha256_payload(artifact)
        or not isinstance(artifact, Mapping)
        or artifact.get("artifact_schema") != "conditional_calibration_v1"
        or artifact.get("seed_ids") != cal_ids
        or artifact.get("family_hash") != calibration_bank.get("family_hash")
        or artifact.get("score_matrix_hashes") != calibration_bank.get("score_matrix_hashes")
        or artifact.get("tested_family_hashes") != calibration_bank.get("tested_family_hashes")
        or artifact.get("failed_response_indices") != payload.get(
            "failed_calibration_response_indices"
        )
        or calibration_bank.get("canonical_role") != "calibration"
        or len(calibration_bank.get("seed_ids", [])) != calibration_count
        or calibration_bank.get("failure", {}).get("failed") is not False
        or any(
            not isinstance(item, Mapping)
            or item.get("canonical_kind") != "gaussian"
            for item in calibration_bank.get("response_metadata", [])
        )
        or not isinstance(target_bank, Mapping)
        or target_bank.get("canonical_role") != "power"
        or target_bank.get("response_shape", [None, None])[1] != 1
    ):
        raise BenchmarkAuditError("power frozen calibration binding is invalid")
    target_scores = target_bank.get("p_by_method")
    target_score_hashes = target_bank.get("score_matrix_hashes")
    family_ids = target_bank.get("family_ids")
    if (
        not isinstance(target_scores, Mapping)
        or set(target_scores) != set(METHOD_NAMES)
        or not isinstance(target_score_hashes, Mapping)
        or set(target_score_hashes) != set(METHOD_NAMES)
        or not isinstance(family_ids, list)
        or bool(causal_ids and not set(causal_ids) <= set(family_ids))
    ):
        raise BenchmarkAuditError("power target score evidence is incomplete")
    target_arrays: dict[str, np.ndarray] = {}
    for method in METHOD_NAMES:
        raw = target_scores[method]
        if (
            not isinstance(raw, list) or len(raw) != len(family_ids)
            or any(not isinstance(row, list) or len(row) != response_count for row in raw)
            or target_score_hashes[method] != sha256_payload(raw)
        ):
            raise BenchmarkAuditError("power target score hash/shape is invalid")
        values = np.asarray(raw, dtype=float)
        if np.any(np.isinf(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise BenchmarkAuditError("power target score values are invalid")
        target_arrays[method] = values
    artifact_reference = _validate_snpxsnp_calibration_reference(
        artifact.get("snpxsnp_calibration_reference"),
        calibration_count=calibration_count,
    )
    if (
        calibration_bank.get("snpxsnp_calibration_reference") != artifact_reference
        or target_bank.get("snpxsnp_calibration_reference") != artifact_reference
        or "snpxsnp_calibration_artifact" in calibration_bank
        or "snpxsnp_calibration_artifact" in target_bank
    ):
        raise BenchmarkAuditError("power frozen calibration binding is invalid")
    if artifact.get("method_minima") != payload.get("calibration_minima_by_method"):
        raise BenchmarkAuditError("power compact minima detach from calibration artifact")
    if artifact.get("method_minima_hashes") != payload.get("calibration_minima_hashes"):
        raise BenchmarkAuditError("power compact minima hashes detach from artifact")
    target_metadata = target_bank.get("response_metadata")
    if (
        not isinstance(target_metadata, list) or len(target_metadata) != 1
        or target_metadata[0].get("architecture")
        != scenario.parameters.get("architecture")
        or target_metadata[0].get("causal_group_ids") != causal_ids
        or target_metadata[0].get("pve", {}).get("target_pve")
        != scenario.parameters.get("interaction_pve")
    ):
        raise BenchmarkAuditError("power target truth differs from registry")
    for method in sorted(methods):
        calibration = _compact_vector(
            maps["calibration_minima_by_method"][method], calibration_count,
            cal_failed, "power calibration minimum",
        )
        target = _compact_vector(
            maps["target_minima_by_method"][method], response_count,
            target_failed, "power target minimum",
        )
        recomputed_target = np.where(
            np.isfinite(target_arrays[method]).any(axis=0),
            np.where(np.isfinite(target_arrays[method]), target_arrays[method], np.inf).min(axis=0),
            np.nan,
        ).tolist()
        if target != recomputed_target:
            raise BenchmarkAuditError("power target minima detach from score matrix")
        causal: list[list[float | None]] = []
        if not negative_control:
            causal_raw = maps["causal_minima_by_method"][method]
            if not isinstance(causal_raw, list) or len(causal_raw) != len(causal_ids):
                raise BenchmarkAuditError("power causal-minimum group count mismatch")
            causal = [
                _compact_vector(row, response_count, target_failed, "power causal minimum")
                for row in causal_raw
            ]
            causal_indices = [family_ids.index(group_id) for group_id in causal_ids]
            recomputed_causal = target_arrays[method][causal_indices].tolist()
            if causal != recomputed_causal:
                raise BenchmarkAuditError("power causal minima detach from score matrix")
        compact_values = [("calibration", calibration), ("target", target)]
        if not negative_control:
            compact_values.append(("causal", causal))
        for name, compact in compact_values:
            hashes = maps[f"{name}_minima_hashes"]
            if hashes[method] != sha256_payload(compact):
                raise BenchmarkAuditError(f"power {name} compact hash mismatch")
        usable_calibration = [
            value for index, value in enumerate(calibration)
            if index not in cal_failed and value is not None
        ]
        k = int(math.floor(0.05 * (len(usable_calibration) + 1)))
        threshold = None if k < 1 else sorted(usable_calibration)[k - 1]
        if maps[cutoff_name][method] != threshold:
            raise BenchmarkAuditError("power threshold differs from calibration minima")
        rejected = [
            False if index in target_failed or threshold is None else bool(
                target[index] is not None and target[index] < threshold
            )
            for index in range(response_count)
        ]
        detected = [] if negative_control else [
            False if index in target_failed or threshold is None else any(
                row[index] is not None and row[index] < threshold for row in causal
            )
            for index in range(response_count)
        ]
        recall = [] if negative_control else [
            0.0 if index in target_failed or threshold is None else sum(
                row[index] is not None and row[index] < threshold for row in causal
            ) / len(causal)
            for index in range(response_count)
        ]
        decisions_ok = maps[rejection_name][method] == rejected
        if negative_control:
            decisions_ok = decisions_ok and maps[false_positive_name][method] == rejected
            decisions_ok = decisions_ok and maps[specificity_name][method] == [
                not value for value in rejected
            ]
        else:
            decisions_ok = decisions_ok and maps[causal_detection_name][method] == detected
            decisions_ok = decisions_ok and maps[recall_name][method] == recall
        if not decisions_ok:
            raise BenchmarkAuditError("power decisions differ from compact minima")


def _audit_robustness_record(
    perturbation: str, record: Mapping[str, Any], payload: Mapping[str, Any],
) -> None:
    ruling = record.get("design_ruling")
    calibration = record.get("calibration")
    heldout = record.get("heldout")
    pilot = payload.get("stage") == "pilot"
    strata = record.get(
        "qa_power_by_architecture" if pilot else "power_by_architecture"
    )
    architectures = [
        "minor_burden_aligned", "pc1_distributed", "kernel_multidimensional",
        "single_snp_pair", "mixed_sign",
    ]
    if payload.get("pair_edges_per_group") != 1:
        architectures.append("multi_edge_group")
    response_count = 20 if pilot else 500
    calibration_count = payload.get("bootstrap_B")
    if (
        isinstance(calibration_count, bool)
        or not isinstance(calibration_count, int)
        or calibration_count != (199 if pilot else 2_000)
    ):
        raise BenchmarkAuditError("robustness calibration count is invalid")
    if (
        ruling != {
            "interaction_pve": 0.05, "causal_groups": 1,
            "architectures": architectures, "calibration_count": calibration_count,
            "heldout_count": response_count,
            "power_count_per_architecture": response_count,
            "stratify_by_architecture": True,
            "component_regret_reference": [
                "minor_burden", "pc1", "kernel_hadamard",
            ],
        }
        or not isinstance(calibration, Mapping)
        or not isinstance(heldout, Mapping)
        or not isinstance(strata, Mapping) or list(strata) != architectures
    ):
        raise BenchmarkAuditError("robustness frozen design ruling is invalid")
    cutoff_name = "qa_cutoffs_by_method" if pilot else "thresholds"
    rejection_name = "qa_rejections_by_method" if pilot else "rejections_by_method"
    detection_name = "qa_detection_by_method" if pilot else "detection_by_method"
    power_name = "qa_power_by_method" if pilot else "power_by_method"
    regret_name = (
        "qa_absolute_power_regret_by_method"
        if pilot else "absolute_power_regret_by_method"
    )
    if set(calibration) != {
        "response_ids", "seeds", "response_hash", "p_by_method", "p_hashes",
        cutoff_name,
    } or set(heldout) != {
        "response_ids", "seeds", "response_hash", "p_by_method", "p_hashes",
        rejection_name,
    }:
        raise BenchmarkAuditError("robustness raw stage schema is invalid")
    marker_design = record.get("realized_marker_design")
    maf_bands = {
        "maf_0p01_0p05": (0.01, True, 0.05, False),
        "maf_0p05_0p20": (0.05, True, 0.20, True),
        "maf_above_0p20": (0.20, False, 0.50, True),
    }
    if perturbation in maf_bands:
        if not isinstance(marker_design, Mapping):
            raise BenchmarkAuditError("MAF robustness realized marker design is missing")
        lower, lower_inclusive, upper, upper_inclusive = maf_bands[perturbation]
        ids = marker_design.get("selected_marker_ids")
        observed_min = marker_design.get("realized_maf_min")
        observed_max = marker_design.get("realized_maf_max")
        if (
            marker_design.get("band") != {
                "lower": lower, "lower_inclusive": lower_inclusive,
                "upper": upper, "upper_inclusive": upper_inclusive,
            }
            or not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or marker_design.get("selected_marker_count") != len(ids)
            or marker_design.get("selected_marker_ids_hash") != sha256_payload(ids)
            or not isinstance(observed_min, (int, float))
            or not isinstance(observed_max, (int, float))
            or (observed_min < lower if lower_inclusive else observed_min <= lower)
            or (observed_max > upper if upper_inclusive else observed_max >= upper)
        ):
            raise BenchmarkAuditError("MAF robustness realized band is invalid")
    elif marker_design is not None:
        raise BenchmarkAuditError("non-MAF robustness has an unexpected MAF design")

    def score_matrices(bank: Mapping[str, Any], count: int) -> dict[str, np.ndarray]:
        ids, seeds = bank.get("response_ids"), bank.get("seeds")
        matrices, hashes = bank.get("p_by_method"), bank.get("p_hashes")
        if (
            not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count
            or not isinstance(seeds, list) or len(seeds) != count
            or len(set(seeds)) != count
            or not isinstance(matrices, Mapping) or set(matrices) != set(_ROBUSTNESS_METHODS)
            or not isinstance(hashes, Mapping) or set(hashes) != set(_ROBUSTNESS_METHODS)
        ):
            raise BenchmarkAuditError("robustness response bank is incomplete")
        output: dict[str, np.ndarray] = {}
        family_count = len(payload.get("family_ids", []))
        for method in _ROBUSTNESS_METHODS:
            raw = matrices[method]
            if (
                not isinstance(raw, list) or len(raw) != family_count
                or any(not isinstance(row, list) or len(row) != count for row in raw)
                or hashes[method] != sha256_payload(raw)
            ):
                raise BenchmarkAuditError("robustness score commitment is invalid")
            output[method] = np.asarray(raw, dtype=float)
        return output

    calibration_scores = score_matrices(calibration, calibration_count)
    heldout_scores = score_matrices(heldout, response_count)
    if set(calibration["response_ids"]) & set(heldout["response_ids"]):
        raise BenchmarkAuditError("robustness null banks overlap")
    thresholds = {
        method: (
            lambda minima: sorted(minima)[int(math.floor(0.05 * (len(minima) + 1))) - 1]
        )(_finite_minima(calibration_scores[method]))
        for method in _ROBUSTNESS_METHODS
    }
    if calibration.get(cutoff_name) != thresholds:
        raise BenchmarkAuditError("robustness threshold differs from calibration")
    heldout_decisions = {
        method: (_finite_minimum_array(values) < thresholds[method]).tolist()
        for method, values in heldout_scores.items()
    }
    if heldout.get(rejection_name) != heldout_decisions:
        raise BenchmarkAuditError("robustness heldout decisions do not recompute")
    record_fwer_name = "qa_fwer" if pilot else "fwer"
    if record.get(record_fwer_name) != float(np.mean(heldout_decisions["omnib"])):
        raise BenchmarkAuditError("robustness FWER does not recompute")
    family_ids = payload["family_ids"]
    all_correlations: list[float] = []
    all_top: list[float] = []
    for architecture in architectures:
        arm = strata[architecture]
        if not isinstance(arm, Mapping) or set(arm) != {
            "response_ids", "seeds", "response_hash", "causal_group_ids",
            "p_by_method", "p_hashes", "baseline_p_by_method",
            "baseline_p_hashes", "rank_correlation_by_method",
            "top_k_jaccard_by_method", detection_name, power_name, regret_name,
        }:
            raise BenchmarkAuditError("robustness power stratum is invalid")
        scores = score_matrices(arm, response_count)
        if set(calibration["response_ids"]) & set(arm["response_ids"]):
            raise BenchmarkAuditError("robustness power/calibration banks overlap")
        causal = arm.get("causal_group_ids")
        if not isinstance(causal, list) or len(causal) != 1 or causal[0] not in family_ids:
            raise BenchmarkAuditError("robustness causal truth is invalid")
        index = family_ids.index(causal[0])
        detection = {
            method: (values[index] < thresholds[method]).tolist()
            for method, values in scores.items()
        }
        powers = {method: float(np.mean(values)) for method, values in detection.items()}
        if arm.get(detection_name) != detection or arm.get(power_name) != powers:
            raise BenchmarkAuditError("robustness power decisions do not recompute")
        expected_regret = {
            "omnib": abs(
                powers["omnib"] - max(
                    powers["minor_burden"], powers["pc1"],
                    powers["kernel_hadamard"],
                )
            ),
            "minor_burden": None,
            "pc1": None,
            "kernel_hadamard": None,
        }
        if arm.get(regret_name) != expected_regret:
            raise BenchmarkAuditError("robustness component regret does not recompute")
        baseline_raw = arm.get("baseline_p_by_method")
        baseline_hashes = arm.get("baseline_p_hashes")
        if (
            not isinstance(baseline_raw, Mapping)
            or set(baseline_raw) != set(_ROBUSTNESS_METHODS)
            or not isinstance(baseline_hashes, Mapping)
            or set(baseline_hashes) != set(_ROBUSTNESS_METHODS)
        ):
            raise BenchmarkAuditError("robustness baseline rank commitment is invalid")
        correlations_by_method: dict[str, list[float | None]] = {}
        overlaps_by_method: dict[str, list[float]] = {}
        k = min(10, len(family_ids))
        for method in _ROBUSTNESS_METHODS:
            baseline_method = baseline_raw[method]
            if (
                not isinstance(baseline_method, list)
                or len(baseline_method) != len(family_ids)
                or any(
                    not isinstance(row, list) or len(row) != response_count
                    for row in baseline_method
                )
                or baseline_hashes[method] != sha256_payload(baseline_method)
            ):
                raise BenchmarkAuditError(
                    "robustness baseline rank commitment is invalid"
                )
            baseline = np.asarray(baseline_method, dtype=float)
            correlations: list[float | None] = []
            overlaps: list[float] = []
            for column in range(response_count):
                left, right = baseline[:, column], scores[method][:, column]
                finite = np.isfinite(left) & np.isfinite(right)
                correlation = None
                if finite.sum() >= 2:
                    lrank = np.argsort(
                        np.argsort(left[finite], kind="stable"), kind="stable"
                    )
                    rrank = np.argsort(
                        np.argsort(right[finite], kind="stable"), kind="stable"
                    )
                    value = np.corrcoef(lrank, rrank)[0, 1]
                    correlation = float(value) if np.isfinite(value) else None
                correlations.append(correlation)
                ltop = set(np.argsort(
                    np.where(np.isfinite(left), left, np.inf)
                )[:k])
                rtop = set(np.argsort(
                    np.where(np.isfinite(right), right, np.inf)
                )[:k])
                overlaps.append(
                    len(ltop & rtop) / len(ltop | rtop) if ltop | rtop else 1.0
                )
            correlations_by_method[method] = correlations
            overlaps_by_method[method] = overlaps
        if (
            arm.get("rank_correlation_by_method") != correlations_by_method
            or arm.get("top_k_jaccard_by_method") != overlaps_by_method
        ):
            raise BenchmarkAuditError("robustness ranking metrics do not recompute")
        all_correlations.extend(
            value for value in correlations_by_method["omnib"] if value is not None
        )
        all_top.extend(overlaps_by_method["omnib"])
    if (
        record.get("rank_correlation")
        != (float(np.mean(all_correlations)) if all_correlations else None)
        or record.get("top_k_jaccard") != float(np.mean(all_top))
        or record.get("non_estimable_rate")
        != float(np.mean(~np.isfinite(heldout_scores["omnib"])))
    ):
        raise BenchmarkAuditError("robustness aggregate metrics do not recompute")


def _finite_minimum_array(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    return np.where(finite.any(axis=0), np.where(finite, values, np.inf).min(axis=0), np.nan)


def _finite_minima(values: np.ndarray) -> list[float]:
    minima = _finite_minimum_array(values)
    if not np.all(np.isfinite(minima)):
        raise BenchmarkAuditError("robustness calibration has non-estimable responses")
    return minima.tolist()


def _audit_families_and_parallel(evidence: LoadedEvidence) -> None:
    tested_hashes: dict[tuple[str, str], tuple[int, str]] = {}
    power_calibration_by_backbone: dict[str, str] = {}
    global_calibration_banks: dict[str, Mapping[str, Any]] = {}
    conditional_calibration_banks: dict[str, Mapping[str, Any]] = {}
    for _path, candidate in evidence.shards:
        if (
            candidate.get("track") == "omnib"
            and candidate.get("experiment") == "global_vc"
            and candidate.get("target_role") == "calibration"
        ):
            scenario_id = candidate.get("scenario_id")
            if not isinstance(scenario_id, str) or scenario_id in global_calibration_banks:
                raise BenchmarkAuditError("global VC calibration bank identity is invalid")
            global_calibration_banks[scenario_id] = candidate
        if (
            candidate.get("track") == "omnib"
            and candidate.get("experiment") == "conditional"
            and isinstance(candidate.get("bank"), Mapping)
            and candidate["bank"].get("canonical_role") == "calibration"
        ):
            scenario_id = candidate.get("scenario_id")
            if not isinstance(scenario_id, str) or scenario_id in conditional_calibration_banks:
                raise BenchmarkAuditError("conditional calibration bank identity is invalid")
            conditional_calibration_banks[scenario_id] = candidate["bank"]
    for _path, payload in evidence.shards:
        if payload.get("track") == "fit" and payload.get("experiment") == "loco":
            artifacts = evidence.design_lock.get("loco_truth_artifacts")
            stage_records = (
                artifacts.get(str(payload.get("stage")))
                if isinstance(artifacts, Mapping) else None
            )
            scenario_records = (
                stage_records.get(str(payload.get("scenario_id")))
                if isinstance(stage_records, Mapping) else None
            )
            record = (
                scenario_records.get(str(payload.get("replicate")))
                if isinstance(scenario_records, Mapping) else None
            )
            binding = payload.get("released_scan_truth_binding")
            truth_manifest = payload.get("truth_manifest")
            if (
                not isinstance(record, Mapping)
                or not isinstance(binding, Mapping)
                or binding.get("truth_artifact_path") != record.get("path")
                or binding.get("truth_artifact_sha256") != record.get("sha256")
                or binding.get("truth_hash") != record.get("truth_hash")
                or binding.get("source") != record.get("source")
                or binding.get("seed") != record.get("seed")
                or binding.get("generated_config_sha256")
                != record.get("generated_config_sha256")
                or binding.get("phenotype_sha256") != record.get("phenotype_sha256")
                or not isinstance(truth_manifest, Mapping)
                or truth_manifest.get("source") != record.get("source")
                or truth_manifest.get("seed") != record.get("seed")
            ):
                raise BenchmarkAuditError("LOCO truth is not presealed in design lock")
        if payload["track"] == "omnib":
            if payload.get("experiment") == "global_vc":
                if (
                    payload.get("method") != "global_hadamard_variance_component"
                    or payload.get("hypothesis_unit") != "global"
                    or payload.get("detection_only") is not True
                    or any(field in payload for field in (
                        "causal_group_ids", "causal_recall", "recall_by_method",
                    ))
                ):
                    raise BenchmarkAuditError("global VC detection-only contract is invalid")
                calibration = payload.get("calibration_p_values")
                target = payload.get("target_p_values")
                calibration_failed = _index_set(
                    payload.get("failed_calibration_response_indices"),
                    len(calibration) if isinstance(calibration, list) else 0,
                    "global VC calibration",
                )
                target_failed = _index_set(
                    payload.get("failed_target_response_indices"),
                    len(target) if isinstance(target, list) else 0,
                    "global VC target",
                )
                if (
                    not isinstance(calibration, list) or not calibration
                    or not isinstance(target, list) or not target
                    or any(
                        (value is None) is not (index in calibration_failed)
                        or (value is not None and (
                            isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or not 0 <= value <= 1
                        ))
                        for index, value in enumerate(calibration)
                    )
                    or any(
                        (value is None) is not (index in target_failed)
                        or (value is not None and (
                            isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or not 0 <= value <= 1
                        ))
                        for index, value in enumerate(target)
                    )
                    or payload.get("calibration_p_hash") != sha256_payload(calibration)
                    or payload.get("target_p_hash") != sha256_payload(target)
                ):
                    raise BenchmarkAuditError("global VC p-value evidence is invalid")
                calibration_lrt = payload.get("calibration_lrt_evidence")
                target_lrt = payload.get("target_lrt_evidence")
                if (
                    not isinstance(calibration_lrt, list)
                    or len(calibration_lrt) != len(calibration)
                    or not isinstance(target_lrt, list)
                    or len(target_lrt) != len(target)
                ):
                    raise BenchmarkAuditError("global VC LRT evidence is incomplete")
                for rows, reported in ((calibration_lrt, calibration), (target_lrt, target)):
                    failed_indices = calibration_failed if rows is calibration_lrt else target_failed
                    for index, (row, reported_p) in enumerate(zip(rows, reported, strict=True)):
                        if not isinstance(row, Mapping):
                            raise BenchmarkAuditError("global VC LRT evidence is invalid")
                        if index in failed_indices:
                            if (
                                reported_p is not None
                                or row.get("status") != "failed"
                                or not isinstance(row.get("error_type"), str)
                                or not isinstance(row.get("message"), str)
                            ):
                                raise BenchmarkAuditError(
                                    "global VC failed-response evidence is invalid"
                                )
                            continue
                        if (
                            row.get("status") != "completed"
                            or row.get("error_type") is not None
                            or row.get("message") is not None
                        ):
                            raise BenchmarkAuditError(
                                "global VC successful-response status is invalid"
                            )
                        null_name, alt_name = row.get("null_model"), row.get("alt_model")
                        if not isinstance(null_name, str) or not isinstance(alt_name, str):
                            raise BenchmarkAuditError("global VC LRT model identity is invalid")
                        null_components = [
                            item for item in null_name.split("+") if item != "e"
                        ]
                        comparison = NestedREMLComparison(
                            fits={
                                null_name: SimpleNamespace(
                                    log_lik=row.get("ll_null"),
                                    optimizer_status=row.get("both_converged"),
                                    boundary_components=list(
                                        row.get("null_boundary_components", [])
                                    ),
                                ),
                                alt_name: SimpleNamespace(
                                    log_lik=row.get("ll_alt"),
                                    optimizer_status=row.get("both_converged"),
                                ),
                            },
                            model_specs={
                                null_name: null_components,
                                alt_name: [*null_components, "hom"],
                            },
                            likelihood_table=None,
                            kernels_used=[*null_components, "hom"],
                        )
                        try:
                            recomputed = boundary_lrt(comparison, null_name, alt_name)
                        except (TypeError, ValueError) as error:
                            raise BenchmarkAuditError(
                                "global VC LRT evidence is invalid"
                            ) from error
                        expected_lrt = {
                            "statistic": recomputed.statistic,
                            "statistic_raw": recomputed.statistic_raw,
                            "df_added": recomputed.df_added,
                            "p_mixture": recomputed.p_mixture,
                            "clipped": recomputed.clipped,
                            "both_converged": recomputed.both_converged,
                        }
                        if (
                            any(row.get(key) != value for key, value in expected_lrt.items())
                            or reported_p != recomputed.p_mixture
                            or recomputed.clipped
                            or not recomputed.both_converged
                        ):
                            raise BenchmarkAuditError("global VC LRT does not recompute")
                calibration_ids = payload.get("calibration_response_ids")
                target_ids = payload.get("target_response_ids")
                role = payload.get("target_role")
                if (
                    role not in {"calibration", "heldout", "power"}
                    or not isinstance(calibration_ids, list)
                    or not isinstance(target_ids, list)
                    or len(calibration_ids) != len(calibration)
                    or len(target_ids) != len(target)
                    or len(set(calibration_ids)) != len(calibration_ids)
                    or len(set(target_ids)) != len(target_ids)
                ):
                    raise BenchmarkAuditError("global VC response-bank identity is invalid")
                if role == "calibration":
                    if (
                        payload.get("calibration_scenario_id") != payload.get("scenario_id")
                        or calibration_ids != target_ids
                        or calibration != target
                        or payload.get("calibration_p_hash") != payload.get("target_p_hash")
                        or calibration_lrt != target_lrt
                    ):
                        raise BenchmarkAuditError("global VC calibration bank is inconsistent")
                else:
                    frozen = global_calibration_banks.get(
                        str(payload.get("calibration_scenario_id"))
                    )
                    if (
                        frozen is None
                        or set(calibration_ids) & set(target_ids)
                        or calibration_ids != frozen.get("target_response_ids")
                        or calibration != frozen.get("target_p_values")
                        or payload.get("calibration_p_hash") != frozen.get("target_p_hash")
                        or calibration_lrt != frozen.get("target_lrt_evidence")
                    ):
                        raise BenchmarkAuditError(
                            "global VC calibration bank is detached from heldout evidence"
                        )
                    if role == "power":
                        positive = payload.get("positive_signal")
                        flags = payload.get(
                            "rejected" if payload.get("stage") == "formal"
                            else "qa_detection_flags"
                        )
                        if (
                            positive != {
                                "architecture": "kernel_multidimensional",
                                "interaction_pve": 0.05,
                                "causal_groups": 1,
                            }
                            or not isinstance(flags, list)
                            or len(flags) != len(target)
                            or payload.get("detection_power")
                            != float(np.mean(np.asarray(flags, dtype=bool)))
                        ):
                            raise BenchmarkAuditError(
                                "global VC detection-only power evidence is invalid"
                            )
                usable_calibration = [
                    float(value) for value in calibration if value is not None
                ]
                expected = _empirical_threshold([usable_calibration])
                expected_decisions = [
                    False if value is None or expected is None else value < expected
                    for value in target
                ]
                if payload.get("stage") == "formal":
                    if (
                        payload.get("threshold") != expected
                        or payload.get("rejected") != expected_decisions
                    ):
                        raise BenchmarkAuditError("global VC formal decisions do not recompute")
                elif (
                    payload.get("inference_status") != "noninferential_do_not_threshold"
                    or any(field in payload for field in ("threshold", "rejected", "passed"))
                    or payload.get("qa_calibration_cutoff") != expected
                    or payload.get("qa_detection_flags") != expected_decisions
                ):
                    raise BenchmarkAuditError("global VC pilot contains formal inference fields")
                failure = payload.get("failure")
                any_failure = bool(calibration_failed or target_failed)
                if (
                    not isinstance(failure, Mapping)
                    or failure.get("failed") is not any_failure
                    or failure.get("status") != (
                        "partial_failure" if any_failure else "completed"
                    )
                    or failure.get("failed_calibration_response_indices")
                    != sorted(calibration_failed)
                    or failure.get("failed_target_response_indices")
                    != sorted(target_failed)
                ):
                    raise BenchmarkAuditError("global VC failure envelope is invalid")
                manifest = payload.get("kernel_manifest")
                family_manifest = payload.get("family_manifest")
                family_ids = payload.get("family_ids")
                if (
                    not isinstance(manifest, Mapping)
                    or manifest.get("construction") != "hadamard_product"
                    or manifest.get("normalization") != "trace"
                    or not isinstance(manifest.get("global_hadamard_sha256"), str)
                    or len(manifest["global_hadamard_sha256"]) != 64
                    or not isinstance(family_manifest, Mapping)
                    or not isinstance(family_ids, list)
                    or family_manifest.get("group_ids") != family_ids
                    or payload.get("family_hash") != sha256_payload(family_manifest)
                ):
                    raise BenchmarkAuditError("global VC kernel commitment is invalid")
                continue
            canonical = {
                "mode": "group", "statistic": "omniB",
                "hypothesis_unit": "group", "family_scope": "primary_only",
                "subset_order": 2, "direct_higher_order_term": False,
            }
            for field, expected in canonical.items():
                if payload.get(field) != expected:
                    raise BenchmarkAuditError(f"noncanonical omniB {field}")
            family_manifest = payload.get("family_manifest")
            context_manifest = payload.get("context_manifest")
            if not isinstance(family_manifest, Mapping) or not isinstance(context_manifest, Mapping):
                raise BenchmarkAuditError("omniB family/context manifest is missing")
            if payload.get("family_hash") != sha256_payload(family_manifest):
                raise BenchmarkAuditError("omniB family manifest hash mismatch")
            if payload.get("context_fingerprint") != sha256_payload(context_manifest):
                raise BenchmarkAuditError("omniB context manifest hash mismatch")
            if context_manifest.get("family") != family_manifest:
                raise BenchmarkAuditError("omniB context/family manifest mismatch")
            ids = payload.get("family_ids")
            if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
                raise BenchmarkAuditError("invalid ordered family IDs")
            if family_manifest.get("group_ids") != ids:
                raise BenchmarkAuditError("omniB ordered family IDs differ from manifest")
            subgenomes, genes = family_manifest.get("subgenomes"), family_manifest.get("genes")
            if (
                not isinstance(subgenomes, list) or len(subgenomes) < 2
                or len(subgenomes) != len(set(subgenomes))
                or not isinstance(genes, list) or len(genes) != len(ids)
                or any(not isinstance(row, list) or len(row) != len(subgenomes) for row in genes)
            ):
                raise BenchmarkAuditError("omniB family manifest structure is invalid")
            expected_order = sha256_payload(ids)
            if payload.get("family_order_hash", expected_order) != expected_order:
                raise BenchmarkAuditError("family order hash mismatch")
            family_hash = payload.get("family_hash")
            if (
                not isinstance(family_hash, str) or len(family_hash) != 64
                or any(character not in "0123456789abcdef" for character in family_hash)
            ):
                raise BenchmarkAuditError("invalid family hash")
            if payload.get("experiment") == "end2end":
                registry_row = next(
                    row for row in evidence.registry
                    if row.scenario_id == payload["scenario_id"]
                )
                if payload.get("bootstrap_B") != registry_row.bootstrap_B:
                    raise BenchmarkAuditError("end-to-end bootstrap B differs from registry")
            if payload.get("experiment") == "family_size":
                registry_row = next(
                    row for row in evidence.registry
                    if row.scenario_id == payload["scenario_id"]
                )
                declared = registry_row.parameters.get("family_size")
                methods = payload.get("methods")
                expected_methods = set(METHOD_NAMES) if declared == 80 else set(METHOD_NAMES) - {"snpxsnp"}
                fwer = payload.get("fwer")
                calibration_count = payload.get("calibration_count")
                response_count = payload.get("response_count")
                calibration_minima = payload.get("calibration_minima_by_method")
                target_minima = payload.get("target_minima_by_method")
                calibration_hashes = payload.get("calibration_minima_hashes")
                target_hashes = payload.get("target_minima_hashes")
                rejections = payload.get(
                    "rejections_by_method"
                    if payload.get("stage") == "formal"
                    else "qa_rejections_by_method"
                )
                tested_sizes = payload.get("tested_family_sizes")
                tested_members = payload.get("tested_family_members")
                tested_family_hashes = payload.get("tested_family_hashes")
                hypothesis_units = payload.get("score_hypothesis_units")
                thresholds = payload.get(
                    "thresholds" if payload.get("stage") == "formal" else "qa_cutoffs"
                )
                calibration_ids = payload.get("calibration_response_ids")
                target_ids = payload.get("target_response_ids")
                if (
                    payload.get("family_size") != declared
                    or payload.get("group_count") != declared
                    or set(methods or []) != expected_methods
                    or set(payload.get("tested_family_sizes") or {}) != expected_methods
                    or not isinstance(fwer, Mapping) or set(fwer) != expected_methods
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) or not 0 <= value <= 1
                           for value in fwer.values())
                    or payload.get("snpxsnp_status") != (
                        "applicable" if declared == 80 else "not_applicable_above_80"
                    )
                    or isinstance(calibration_count, bool)
                    or not isinstance(calibration_count, int) or calibration_count < 1
                    or isinstance(response_count, bool)
                    or not isinstance(response_count, int) or response_count < 1
                    or not all(isinstance(value, Mapping) for value in (
                        calibration_minima, target_minima, calibration_hashes,
                        target_hashes, rejections, thresholds, tested_sizes,
                        tested_members, tested_family_hashes, hypothesis_units,
                    ))
                    or any(set(value) != expected_methods for value in (
                        calibration_minima, target_minima, calibration_hashes,
                        target_hashes, rejections, thresholds, tested_sizes,
                        tested_members, tested_family_hashes, hypothesis_units,
                    ))
                    or not isinstance(calibration_ids, list)
                    or len(calibration_ids) != calibration_count
                    or len(set(calibration_ids)) != calibration_count
                    or not isinstance(target_ids, list)
                    or len(target_ids) != response_count
                    or len(set(target_ids)) != response_count
                    or bool(set(calibration_ids) & set(target_ids))
                    or len(payload.get("calibration_seeds", [])) != calibration_count
                    or len(payload.get("target_seeds", [])) != response_count
                    or not isinstance(payload.get("calibration_response_hash"), str)
                    or not isinstance(payload.get("target_response_hash"), str)
                    or payload.get("calibration_response_hash")
                    == payload.get("target_response_hash")
                ):
                    raise BenchmarkAuditError("family-size statistical stress evidence is invalid")
                for method in sorted(expected_methods):
                    if hypothesis_units[method] != (
                        "snp_pair_within_group"
                        if method == "snpxsnp" else "group_score"
                    ):
                        raise BenchmarkAuditError(
                            "family-size hypothesis unit is invalid"
                        )
                    ordered_members = tested_members[method]
                    if (
                        not isinstance(ordered_members, list) or not ordered_members
                        or len(ordered_members) != tested_sizes[method]
                        or len(set(ordered_members)) != len(ordered_members)
                        or tested_family_hashes[method] != sha256_payload({
                            "method": method,
                            "ordered_member_ids": ordered_members,
                        })
                    ):
                        raise BenchmarkAuditError(
                            "family-size tested-family evidence is invalid"
                        )
                    calibration_values = _compact_vector(
                        calibration_minima[method], calibration_count, set(),
                        "family-size calibration minimum",
                    )
                    target_values = _compact_vector(
                        target_minima[method], response_count, set(),
                        "family-size target minimum",
                    )
                    if (
                        calibration_hashes[method] != sha256_payload(calibration_values)
                        or target_hashes[method] != sha256_payload(target_values)
                    ):
                        raise BenchmarkAuditError(
                            "family-size compact evidence hash mismatch"
                        )
                    k = int(math.floor(0.05 * (calibration_count + 1)))
                    expected_threshold = (
                        None if k < 1 else sorted(calibration_values)[k - 1]
                    )
                    expected_rejections = [
                        False if expected_threshold is None
                        else bool(value < expected_threshold)
                        for value in target_values
                    ]
                    if (
                        thresholds[method] != expected_threshold
                        or rejections[method] != expected_rejections
                        or fwer[method] != sum(expected_rejections) / response_count
                    ):
                        raise BenchmarkAuditError(
                            "family-size response-level decisions do not recompute"
                        )
            execution = payload.get("bank") if isinstance(payload.get("bank"), Mapping) else payload
            requested = execution.get("requested_jobs", payload.get("requested_jobs"))
            effective = execution.get("effective_jobs")
            backend = execution.get("parallel_backend")
            worker_pids = execution.get("worker_pids")
            if (
                isinstance(requested, bool) or not isinstance(requested, int) or requested < 1
                or isinstance(effective, bool) or not isinstance(effective, int)
                or not 0 <= effective <= requested
                or backend not in {"serial", "fork_shared_memory", "failed"}
                or not isinstance(worker_pids, list)
            ):
                raise BenchmarkAuditError("invalid omniB parallel execution provenance")
            if backend == "failed":
                if effective != 0 or worker_pids:
                    raise BenchmarkAuditError("failed omniB execution metadata is inconsistent")
            elif effective < 1 or len(set(worker_pids)) != effective:
                raise BenchmarkAuditError("omniB worker PID/effective job mismatch")
            nested_banks = [
                value for field in ("bank", "calibration_bank", "target_bank")
                if isinstance((value := payload.get(field)), Mapping)
            ]
            for bank in nested_banks:
                sizes = bank.get("tested_family_sizes")
                hashes = bank.get("tested_family_hashes")
                members = bank.get("tested_family_members")
                score_hashes = bank.get("score_matrix_hashes")
                units = bank.get("score_hypothesis_units")
                if not all(isinstance(value, Mapping) for value in (
                    sizes, hashes, members, score_hashes, units,
                )):
                    raise BenchmarkAuditError("tested family provenance is incomplete")
                if not (
                    set(sizes) == set(hashes) == set(members)
                    == set(score_hashes) == set(units) == set(METHOD_NAMES)
                ):
                    raise BenchmarkAuditError("tested family size/hash methods differ")
                for method in sizes:
                    size, digest = sizes[method], hashes[method]
                    ordered_members = members[method]
                    if (
                        isinstance(size, bool) or not isinstance(size, int) or size < 1
                        or not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)
                        or not isinstance(ordered_members, list)
                        or len(ordered_members) != size
                        or len(set(ordered_members)) != size
                        or any(not isinstance(item, str) or not item for item in ordered_members)
                        or digest != sha256_payload({
                            "method": method,
                            "ordered_member_ids": ordered_members,
                        })
                        or not isinstance(score_hashes[method], str)
                        or len(score_hashes[method]) != 64
                        or any(c not in "0123456789abcdef" for c in score_hashes[method])
                        or units[method] != (
                            "snp_pair_within_group"
                            if method == "snpxsnp" else "group_score"
                        )
                    ):
                        raise BenchmarkAuditError("invalid tested family provenance")
                    key = (str(payload["context_fingerprint"]), str(method))
                    previous = tested_hashes.setdefault(key, (size, digest))
                    if previous != (size, digest):
                        raise BenchmarkAuditError("tested family hash changed across responses")
            if payload.get("experiment") == "power" and not _failed(payload):
                calibration_bank = payload.get("calibration_bank")
                target_bank = payload.get("target_bank")
                if not isinstance(calibration_bank, Mapping) or not isinstance(target_bank, Mapping):
                    raise BenchmarkAuditError("power bank provenance is incomplete")
                if (
                    calibration_bank.get("family_hash") != target_bank.get("family_hash")
                    or calibration_bank.get("family_ids") != target_bank.get("family_ids")
                ):
                    raise BenchmarkAuditError("power banks use different families")
                if calibration_bank.get("response_hash") == target_bank.get("response_hash"):
                    raise BenchmarkAuditError("power calibration and target response hashes overlap")
                registry_row = next(
                    row for row in evidence.registry
                    if row.scenario_id == payload["scenario_id"]
                )
                _audit_power_evidence(payload, registry_row)
                calibration_scenario_id = payload.get("calibration_scenario_id")
                frozen_bank = conditional_calibration_banks.get(
                    str(calibration_scenario_id)
                )
                artifact = payload.get("calibration_artifact")
                if frozen_bank is None or not isinstance(artifact, Mapping):
                    raise BenchmarkAuditError(
                        "power frozen calibration has no registered conditional artifact"
                    )
                frozen_count = len(frozen_bank.get("seed_ids", []))
                full_snpxsnp = _validate_snpxsnp_calibration_artifact(
                    frozen_bank.get("snpxsnp_calibration_artifact"),
                    calibration_count=frozen_count,
                )
                frozen_reference = {
                    key: full_snpxsnp[key] for key in (
                        "schema", "member_ids", "member_ids_sha256",
                        "group_memberships", "calibration_shape",
                        "calibration_p_sha256",
                    )
                }
                direct_fields = (
                    "stage", "canonical_role", "seed_ids", "seeds",
                    "response_hash", "response_metadata", "family_ids",
                    "family_hash", "design_hash", "tested_family_sizes",
                    "tested_family_members", "tested_family_hashes",
                    "score_matrix_hashes", "failed_response_indices",
                )
                if (
                    artifact.get("snpxsnp_calibration_reference") != frozen_reference
                    or any(
                        artifact.get(field) != (
                            frozen_bank.get("failure", {}).get(field)
                            if field == "failed_response_indices"
                            else frozen_bank.get(field)
                        )
                        for field in direct_fields
                    )
                ):
                    raise BenchmarkAuditError(
                        "power frozen calibration differs from registered conditional artifact"
                    )
                score_matrices = frozen_bank.get("p_by_method")
                if not isinstance(score_matrices, Mapping):
                    raise BenchmarkAuditError(
                        "registered conditional calibration scores are missing"
                    )
                frozen_minima: dict[str, list[float | None]] = {}
                for method in METHOD_NAMES:
                    matrix = score_matrices.get(method)
                    if not isinstance(matrix, list) or not matrix:
                        raise BenchmarkAuditError(
                            "registered conditional calibration scores are invalid"
                        )
                    values = np.asarray([
                        [np.nan if item is None else item for item in row]
                        for row in matrix
                    ], dtype=float)
                    minima = np.where(
                        np.isfinite(values).any(axis=0),
                        np.where(np.isfinite(values), values, np.inf).min(axis=0),
                        np.nan,
                    )
                    frozen_minima[method] = [
                        None if not math.isfinite(float(value)) else float(value)
                        for value in minima
                    ]
                if (
                    artifact.get("method_minima") != frozen_minima
                    or artifact.get("method_minima_hashes") != {
                        method: sha256_payload(values)
                        for method, values in frozen_minima.items()
                    }
                ):
                    raise BenchmarkAuditError(
                        "power frozen calibration minima differ from conditional artifact"
                    )
                backbone = registry_row.parameters.get("backbone")
                digest = payload.get("calibration_bank_manifest_hash")
                if not isinstance(backbone, str) or not isinstance(digest, str):
                    raise BenchmarkAuditError("power backbone calibration binding is missing")
                previous = power_calibration_by_backbone.setdefault(backbone, digest)
                if previous != digest:
                    raise BenchmarkAuditError(
                        "power cells do not share the frozen backbone calibration bank"
                    )
        if payload.get("experiment") == "encoding":
            registry_row = next(
                row for row in evidence.registry
                if row.scenario_id == payload["scenario_id"]
            )
            if payload.get("bootstrap_B") != registry_row.bootstrap_B:
                raise BenchmarkAuditError(
                    "encoding robustness calibration B differs from registry"
                )
            checks = payload.get("exact_checks")
            robustness = payload.get("robustness_checks")
            if (
                not isinstance(checks, Mapping)
                or set(checks) != _ENCODING_EXACT_CHECKS
                or not isinstance(robustness, Mapping)
                or set(robustness) != _ENCODING_ROBUSTNESS_CHECKS
            ):
                raise BenchmarkAuditError("encoding checks are missing")
            recomputed_all_required = True
            for check in checks.values():
                if (
                    not isinstance(check, Mapping)
                    or set(check) != _ENCODING_EXACT_SCHEMA
                    or check.get("status") != "completed"
                    or not isinstance(check.get("required"), bool)
                    or any(not isinstance(check.get(field), bool) for field in (
                        "observed_arrays_identical", "adjusted_decisions_identical",
                        "ranking_hash_identical", "rejection_sets_identical",
                    ))
                    or any(
                        not isinstance(check.get(field), str)
                        or len(check[field]) != 64
                        or any(character not in "0123456789abcdef"
                               for character in check[field])
                        for field in ("baseline_observed_hash", "candidate_observed_hash")
                    )
                    or not isinstance(check.get("worker_pids"), list)
                ):
                    raise BenchmarkAuditError("encoding check is invalid")
                if check.get("required"):
                    recomputed_all_required = recomputed_all_required and all(
                        check.get(field) is True for field in (
                            "observed_arrays_identical", "adjusted_decisions_identical",
                            "ranking_hash_identical", "rejection_sets_identical",
                        )
                    )
            if payload.get("all_required_exact") is not recomputed_all_required:
                raise BenchmarkAuditError("encoding aggregate decision differs from checks")
            robustness_stage_fields = (
                {
                    "qa_fwer", "qa_power", "qa_absolute_power_regret",
                    "qa_power_by_architecture",
                }
                if payload.get("stage") == "pilot"
                else {
                    "fwer", "power", "absolute_power_regret",
                    "power_by_architecture",
                }
            )
            for perturbation, record in robustness.items():
                if (
                    not isinstance(record, Mapping)
                    or set(record)
                    != (_ENCODING_ROBUSTNESS_SCHEMA_BASE | robustness_stage_fields)
                    or record.get("status") not in {"completed", "failed"}
                    or not isinstance(record.get("note"), str)
                ):
                    raise BenchmarkAuditError("encoding robustness schema is invalid")
                if record["status"] == "completed":
                    if record.get("error_type") is not None or record.get("message") is not None:
                        raise BenchmarkAuditError("completed robustness record has an error")
                    _audit_robustness_record(perturbation, record, payload)
                elif (
                    not isinstance(record.get("error_type"), str)
                    or not isinstance(record.get("message"), str)
                ):
                    raise BenchmarkAuditError("failed robustness record lacks an error")
                for field, lower, upper in (
                    (("qa_fwer" if payload.get("stage") == "pilot" else "fwer"),
                     0.0, 1.0),
                    (("qa_power" if payload.get("stage") == "pilot" else "power"),
                     0.0, 1.0),
                    ("rank_correlation", -1.0, 1.0),
                    ("top_k_jaccard", 0.0, 1.0),
                    ("non_estimable_rate", 0.0, 1.0),
                    (("qa_absolute_power_regret"
                      if payload.get("stage") == "pilot"
                      else "absolute_power_regret"), 0.0, 1.0),
                ):
                    value = record.get(field)
                    if value is not None and (
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not lower <= value <= upper
                    ):
                        raise BenchmarkAuditError("encoding robustness metric is invalid")
                top_k = record.get("top_k")
                if top_k is not None and (
                    isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
                ):
                    raise BenchmarkAuditError("encoding robustness top_k is invalid")


def _audit_scaling_scenario(
    payload: Mapping[str, Any], scenario: Any,
) -> dict[str, Any]:
    if _failed(payload):
        raise BenchmarkAuditError("scaling scenario is an explicit failed shard")
    anchor_value = scenario.parameters.get("anchor")
    if not isinstance(anchor_value, Mapping):
        raise BenchmarkAuditError("scaling registry anchor is invalid")
    try:
        anchor = ScalingAnchor(
            str(anchor_value["anchor_id"]), int(anchor_value["n"]),
            int(anchor_value["groups"]), int(anchor_value["copies"]),
            int(anchor_value["edges"]), int(anchor_value["bootstrap_B"]),
            tuple(int(value) for value in anchor_value["jobs"]),
            int(anchor_value["repeats"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkAuditError("scaling registry anchor is invalid") from error
    expected_repeats = 1 if scenario.stage == "pilot" else 3
    if anchor.repeats != expected_repeats or scenario.replicates != expected_repeats:
        raise BenchmarkAuditError("scaling stage/repeat contract mismatch")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise BenchmarkAuditError("scaling raw summary evidence is missing")
    runs = summary.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != {str(job) for job in anchor.jobs}:
        raise BenchmarkAuditError("scaling jobs matrix is incomplete")
    measured: list[ScalingAnchorRun] = []
    for jobs in anchor.jobs:
        job_record = runs[str(jobs)]
        if not isinstance(job_record, Mapping):
            raise BenchmarkAuditError("scaling jobs record is invalid")
        repetitions = job_record.get("measured_repeats")
        if not isinstance(repetitions, list) or len(repetitions) != expected_repeats:
            raise BenchmarkAuditError("scaling repeat matrix is incomplete")
        for expected_repeat, raw in enumerate(repetitions):
            if not isinstance(raw, Mapping) or raw.get("repeat") != expected_repeat:
                raise BenchmarkAuditError("scaling repeat order is invalid")
            required_fields = {
                "jobs", "effective_jobs", "backend", "worker_pids",
                "wall_seconds", "cpu_seconds", "aggregate_cpu_percent",
                "peak_aggregate_pss_bytes", "peak_aggregate_rss_bytes",
                "result_sha256", "family_sha256", "ranking_sha256",
                "max_process_threads_by_pid", "numeric_threadpool_info",
                "numeric_threadpool_max_threads", "numeric_thread_env",
                "runtime_oversubscription_guard_passed", "command",
            }
            if not required_fields <= set(raw):
                raise BenchmarkAuditError("scaling raw run lacks required metrics")
            for field in ("result_sha256", "family_sha256", "ranking_sha256"):
                digest = raw[field]
                if (
                    not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise BenchmarkAuditError("scaling identity hash is invalid")
            adapted = dict(raw)
            adapted.setdefault("cpu_percent", adapted.get("aggregate_cpu_percent"))
            adapted.setdefault("peak_rss_bytes", adapted.get("peak_aggregate_pss_bytes"))
            try:
                measured.append(ScalingAnchorRun.from_record(
                    anchor.anchor_id, expected_repeat, adapted,
                ))
            except (KeyError, TypeError, ValueError) as error:
                raise BenchmarkAuditError("scaling raw run is invalid") from error
    recomputed = summarize_anchor(anchor, measured)
    normalized_summary = json.loads(canonical_json(dict(summary)))
    normalized_recomputed = json.loads(canonical_json(recomputed))
    if normalized_summary != normalized_recomputed:
        raise BenchmarkAuditError("scaling summary differs from raw-run recomputation")
    return recomputed


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BenchmarkAuditError(f"invalid application {label}")
    return value


def _audit_application_scenario(
    payload: Mapping[str, Any], scenario: Any,
) -> bool:
    if scenario.parameters.get("read_only") is not True or scenario.parameters.get("rescan") is not False:
        raise BenchmarkAuditError("application scenario is not frozen read-only evidence")
    rows = payload.get("rows", payload.get("application_rows"))
    if _failed(payload) and not isinstance(rows, list):
        return False
    if not isinstance(rows, list) or not rows:
        raise BenchmarkAuditError("application evidence rows are missing")
    species_key = str(scenario.parameters.get("species", ""))
    aliases = _APPLICATION_SPECIES.get(species_key)
    if aliases is None or scenario.scenario_id != f"D.{species_key}":
        raise BenchmarkAuditError("application species/scenario binding is invalid")
    seen: set[str] = set()
    all_accepted = True
    for row in rows:
        if not isinstance(row, Mapping):
            raise BenchmarkAuditError("application row is not an object")
        if set(row) != _APPLICATION_ROW_FIELDS:
            raise BenchmarkAuditError("application row schema is incomplete or has extra fields")
        analysis_id = row.get("analysis_id")
        if not isinstance(analysis_id, str) or not analysis_id or analysis_id in seen:
            raise BenchmarkAuditError("application analysis ID is invalid or duplicate")
        seen.add(analysis_id)
        species = row.get("species")
        if not isinstance(species, str) or species.strip().lower() not in aliases:
            raise BenchmarkAuditError("application row species differs from scenario")
        repair = row.get("repair_required")
        if not isinstance(repair, bool):
            raise BenchmarkAuditError("application repair_required is not boolean")
        status, audit_status = row.get("status"), row.get("audit_status")
        if repair:
            if (
                status not in _APPLICATION_FAILURE_STATUSES
                or not isinstance(row.get("repair_reason"), str)
                or not row["repair_reason"]
            ):
                raise BenchmarkAuditError("application repair row is incomplete")
            all_accepted = False
            continue
        if status not in _APPLICATION_ACCEPTED or audit_status not in _APPLICATION_ACCEPTED:
            raise BenchmarkAuditError("application row has unauthoritative audit status")
        family_hash = row.get("family_hash")
        if (
            not isinstance(family_hash, str) or len(family_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in family_hash)
        ):
            raise BenchmarkAuditError("application family hash is invalid")
        for field in (
            "sample_count", "marker_count", "group_family_count", "edge_family_count",
            "requested_jobs", "effective_jobs", "calibration_B",
        ):
            _positive_int(row.get(field), field)
        if row["effective_jobs"] > row["requested_jobs"]:
            raise BenchmarkAuditError("application effective jobs exceed request")
        backend, pids = row.get("backend"), row.get("worker_pids")
        if backend not in {"serial", "fork_shared_memory"} or not isinstance(pids, list):
            raise BenchmarkAuditError("application backend/PIDs are invalid")
        if (
            len(pids) != row["effective_jobs"] or len(set(pids)) != len(pids)
            or any(isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 for pid in pids)
            or (backend == "serial" and row["effective_jobs"] != 1)
        ):
            raise BenchmarkAuditError("application worker PID contract is invalid")
        discoveries = row.get("adjusted_discoveries")
        count = row.get("adjusted_discovery_count")
        if (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            or not isinstance(discoveries, list) or len(discoveries) != count
            or row.get("negative_result") is not (count == 0)
        ):
            raise BenchmarkAuditError("application discovery summary is inconsistent")
    return all_accepted


def _annotate_binomial_rows(rows: dict[str, list[dict[str, Any]]]) -> None:
    field_by_table = {
        "fit_pve_coverage.tsv": "covered",
        "omnib_null_replicates.tsv": "rejected",
        "omnib_power_replicates.tsv": "detected",
    }
    for name, outcome_field in field_by_table.items():
        for row in rows[name]:
            outcome = row.get(outcome_field)
            failed = row.get("failed") is True
            if not isinstance(outcome, bool) and not failed:
                continue
            summary = summarize_binomial(int(outcome) if isinstance(outcome, bool) else 0, 1)
            row["successes"] = summary["successes"]
            row["total"] = summary["total"]
            row["estimate"] = summary["estimate"]
            row["ci_low"] = summary["ci_low"]
            row["ci_high"] = summary["ci_high"]
            row["failures"] = int(failed)
    for row in rows["fit_scan_metrics.tsv"]:
        outcome = row.get(
            "rejected" if row.get("scan_pve") == 0.0 else "causal_detected"
        )
        failed = row.get("failed") is True
        if not isinstance(outcome, bool) and not failed:
            continue
        summary = summarize_binomial(int(outcome) if isinstance(outcome, bool) else 0, 1)
        row.update(
            successes=summary["successes"], total=1,
            estimate=summary["estimate"], ci_low=summary["ci_low"],
            ci_high=summary["ci_high"], failures=int(failed),
        )


def _annotate_power_eligibility(
    stage: str,
    rows: dict[str, list[dict[str, Any]]],
    gates: Mapping[str, AuditGate],
) -> None:
    for row in rows["omnib_power_replicates.tsv"]:
        parts = str(row["scenario_id"]).split(".")
        backbone = parts[2] if len(parts) > 2 else "unknown"
        gate_id = (
            f"B.core_fwer.B.conditional.{backbone}.gaussian.heldout.{row['method']}"
        )
        gate = gates.get(gate_id)
        row["null_calibration_gate"] = gate.status if gate is not None else "MISSING"
        row["eligible_for_power_summary"] = (
            gate.passed if stage == "formal" and gate is not None else None
        )


def _omnib_power_gates(
    evidence: LoadedEvidence,
    rows: dict[str, list[dict[str, Any]]],
    existing_gates: Mapping[str, AuditGate],
) -> dict[str, AuditGate]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows["omnib_power_replicates.tsv"]:
        grouped[(str(row["scenario_id"]), str(row["method"]))].append(row)
    gates: dict[str, AuditGate] = {}
    for scenario in evidence.registry:
        if scenario.track != "omnib" or scenario.parameters.get("experiment") != "power":
            continue
        observed = {
            method for scenario_id, method in grouped
            if scenario_id == scenario.scenario_id
        }
        if observed != set(METHOD_NAMES):
            raise BenchmarkAuditError("power canonical method rows are incomplete")
        backbone = str(scenario.parameters["backbone"])
        negative_control = scenario.parameters.get("control_type") == "negative"
        for method in METHOD_NAMES:
            method_rows = grouped[(scenario.scenario_id, method)]
            if len(method_rows) != scenario.replicates:
                raise BenchmarkAuditError("power method denominator differs from registry")
            failures = sum(row.get("failed") is True for row in method_rows)
            outcome_field = "specificity" if negative_control else "detected"
            if any(
                not isinstance(row.get(outcome_field), bool)
                for row in method_rows if row.get("failed") is not True
            ):
                raise BenchmarkAuditError(
                    "negative-control specificity evidence is missing"
                    if negative_control else "power detection evidence is missing"
                )
            if negative_control and any(
                row.get("detected") is not None
                or not isinstance(row.get("false_positive"), bool)
                or row.get("specificity") is row.get("false_positive")
                for row in method_rows if row.get("failed") is not True
            ):
                raise BenchmarkAuditError("negative-control evidence has causal semantics")
            successes = sum(row.get(outcome_field) is True for row in method_rows)
            summary = summarize_binomial(successes, scenario.replicates)
            failure_rate = failures / scenario.replicates
            null_id = (
                f"B.core_fwer.B.conditional.{backbone}.gaussian.heldout.{method}"
            )
            null_gate = existing_gates.get(null_id)
            null_ok = (
                null_gate is not None
                and (null_gate.passed is True if evidence.stage == "formal"
                     else null_gate.qa_passed is True)
            )
            ok = failure_rate <= 0.01 and null_ok
            gate_id = (
                f"B.specificity.{scenario.scenario_id}.{method}"
                if negative_control else f"B.power.{scenario.scenario_id}.{method}"
            )
            gates[gate_id] = AuditGate(
                gate_id, "omnib", scenario.scenario_id,
                ("negative_control_specificity" if negative_control
                 else "power_descriptive"), successes, scenario.replicates,
                float(summary["estimate"]), float(summary["ci_low"]),
                float(summary["ci_high"]), failures, failure_rate,
                ok if evidence.stage == "formal" else None,
                ok if evidence.stage == "pilot" else None,
                ("PASS" if ok else "FAIL") if evidence.stage == "formal"
                else ("QA_PASS" if ok else "QA_FAIL"),
                "tables/omnib_power_replicates.tsv",
                ("negative-control specificity uses fixed denominator, frozen Gaussian "
                 "null calibration and failures <= 1%" if negative_control else
                 "power is descriptive; frozen Gaussian null gate passes and failures <= 1%"),
            )
    return gates


def _global_vc_gates(evidence: LoadedEvidence) -> dict[str, AuditGate]:
    gates: dict[str, AuditGate] = {}
    for _path, payload in evidence.shards:
        if payload.get("experiment") != "global_vc":
            continue
        role = payload.get("target_role")
        if role not in {"calibration", "heldout", "power"}:
            continue
        flags = payload.get(
            "rejected" if evidence.stage == "formal" else "qa_detection_flags"
        )
        if not isinstance(flags, list) or any(not isinstance(value, bool) for value in flags):
            raise BenchmarkAuditError("global VC gate decisions are missing")
        target_failures = payload.get("failed_target_response_indices")
        calibration_failures = payload.get("failed_calibration_response_indices")
        if not isinstance(target_failures, list) or not isinstance(
            calibration_failures, list
        ):
            raise BenchmarkAuditError("global VC failure indices are missing")
        failures = len(target_failures)
        successes = sum(flags)
        scenario_id = str(payload["scenario_id"])
        calibration_failure_rate = len(calibration_failures) / len(
            payload["calibration_response_ids"]
        )
        if role == "calibration":
            ok = failures / len(flags) <= 0.01
            summary = summarize_binomial(len(flags) - failures, len(flags))
            gate = AuditGate(
                f"B.global_vc_calibration.{scenario_id}", "omnib", scenario_id,
                "global_vc_calibration_failure", int(summary["successes"]),
                len(flags), float(summary["estimate"]), float(summary["ci_low"]),
                float(summary["ci_high"]), failures, failures / len(flags),
                ok if evidence.stage == "formal" else None,
                ok if evidence.stage == "pilot" else None,
                ("PASS" if ok else "FAIL") if evidence.stage == "formal"
                else ("QA_PASS" if ok else "QA_FAIL"),
                "tables/omnib_null_replicates.tsv",
                "global VC calibration retains the fixed response denominator and <=1% failures",
            )
            gates[gate.gate_id] = gate
            continue
        if role == "heldout":
            gate = core_fwer_gate(
                scenario_id, successes, len(flags), failures,
                stage=evidence.stage,
                evidence_path="tables/omnib_null_replicates.tsv",
            )
            gate = replace(
                gate, gate_id=f"B.global_vc_fwer.{scenario_id}",
                gate_kind="global_vc_fwer",
                passed=(gate.passed and calibration_failure_rate <= 0.01
                        if evidence.stage == "formal" else None),
                qa_passed=(gate.qa_passed and calibration_failure_rate <= 0.01
                           if evidence.stage == "pilot" else None),
                status=(
                    ("PASS" if gate.passed and calibration_failure_rate <= 0.01 else "FAIL")
                    if evidence.stage == "formal" else
                    ("QA_PASS" if gate.qa_passed and calibration_failure_rate <= 0.01
                     else "QA_FAIL")
                ),
                reason="standalone global VC heldout FWER uses a frozen calibration bank; "
                "both failure rates are <=1%",
            )
        else:
            summary = summarize_binomial(successes, len(flags))
            ok = failures / len(flags) <= 0.01 and calibration_failure_rate <= 0.01
            gate = AuditGate(
                f"B.global_vc_power.{scenario_id}", "omnib", scenario_id,
                "global_vc_detection_power", successes, len(flags),
                float(summary["estimate"]), float(summary["ci_low"]),
                float(summary["ci_high"]), failures, failures / len(flags),
                ok if evidence.stage == "formal" else None,
                ok if evidence.stage == "pilot" else None,
                ("PASS" if ok else "FAIL") if evidence.stage == "formal"
                else ("QA_PASS" if ok else "QA_FAIL"),
                "tables/omnib_null_replicates.tsv",
                "global VC detection-only power is descriptive; calibration and target "
                "failure rates are <= 1%",
            )
        gates[gate.gate_id] = gate
    return gates


def _provenance_gate(evidence: LoadedEvidence) -> AuditGate:
    passed = True if evidence.stage == "formal" else None
    qa_passed = True if evidence.stage == "pilot" else None
    return AuditGate(
        "all.provenance", "all", "*", "provenance", None, None, None, None,
        None, 0, None, passed, qa_passed,
        "PASS" if evidence.stage == "formal" else "QA_PASS",
        "design_lock.json;scenario_registry.tsv", "design, registry, shard keys and hashes agree",
    )


def _boolean_gate(
    *, gate_id: str, track: str, scenario_id: str, kind: str, ok: bool,
    stage: str, evidence_path: str, reason: str,
) -> AuditGate:
    return AuditGate(
        gate_id, track, scenario_id, kind, int(ok), 1, float(ok),
        *wilson_interval(int(ok), 1), 0, 0.0,
        ok if stage == "formal" else None,
        ok if stage == "pilot" else None,
        ("PASS" if ok else "FAIL") if stage == "formal"
        else ("QA_PASS" if ok else "QA_FAIL"),
        evidence_path, reason,
    )


def _pilot_and_engineering_gates(
    evidence: LoadedEvidence,
) -> dict[str, AuditGate]:
    gates: dict[str, AuditGate] = {}
    pve_checks: list[bool] = []
    edge_checks: list[bool] = []
    scaling_payloads: list[tuple[Mapping[str, Any], Any, Mapping[str, Any]]] = []
    application_acceptance: dict[str, bool] = {}
    encoding_exact: dict[str, bool] = {}
    registry = {row.scenario_id: row for row in evidence.registry}
    for _path, payload in evidence.shards:
        scenario = registry[str(payload["scenario_id"])]
        if payload["track"] == "fit":
            panel_preflights = evidence.design_lock.get("comparator_preflights")
            panel = str(scenario.parameters.get("panel", ""))
            panel_record = (
                panel_preflights.get(panel)
                if isinstance(panel_preflights, Mapping) else None
            )
            preflight_sha256 = (
                str(panel_record["sha256"])
                if isinstance(panel_record, Mapping)
                else str(evidence.design_lock.get("comparator_preflight_sha256"))
            )
            _audit_fit_scenario(
                payload, scenario,
                preflight_sha256=preflight_sha256,
            )
            truth = payload.get("truth") or payload.get("scan_truth")
            if isinstance(truth, Mapping):
                if "observed_total_genetic_pve" in truth:
                    pve_checks.append(abs(
                        float(truth["observed_total_genetic_pve"])
                        - float(scenario.parameters.get("total_pve", 0.4))
                    ) <= 0.01)
                if "realized_signal_pve" in truth:
                    pve_checks.append(abs(
                        float(truth["realized_signal_pve"])
                        - float(scenario.parameters.get("scan_pve", 0.0))
                    ) <= 0.01)
        elif payload["track"] == "omnib":
            expected_edges = scenario.parameters.get("edges_per_group")
            if expected_edges is not None:
                edge_checks.append(payload.get("pair_edges_per_group") == expected_edges)
            target = payload.get("target_bank")
            if isinstance(target, Mapping):
                for metadata in target.get("response_metadata", []):
                    pve = metadata.get("pve") if isinstance(metadata, Mapping) else None
                    if isinstance(pve, Mapping):
                        pve_checks.append(abs(
                            float(pve["realized_pve"]) - float(pve["target_pve"])
                        ) <= 0.01)
            if scenario.parameters.get("experiment") == "encoding":
                encoding_exact[scenario.scenario_id] = (
                    payload.get("all_required_exact") is True
                    and payload.get("failure", {}).get("failed") is False
                )
        elif payload["track"] == "scaling":
            scaling_payloads.append((
                payload, scenario, _audit_scaling_scenario(payload, scenario),
            ))
        elif payload["track"] == "application":
            application_acceptance[scenario.scenario_id] = (
                _audit_application_scenario(payload, scenario)
            )
    if edge_checks:
        gates["B.edge_expansion"] = _boolean_gate(
            gate_id="B.edge_expansion", track="omnib", scenario_id="*",
            kind="edge_expansion", ok=all(edge_checks), stage=evidence.stage,
            evidence_path="tables/omnib_family_manifest.tsv",
            reason="every declared 2/3/4-copy group expands to 1/3/6 pair edges",
        )
    if pve_checks:
        gates["all.realized_pve"] = _boolean_gate(
            gate_id="all.realized_pve", track="fit+omnib", scenario_id="*",
            kind="realized_pve", ok=all(pve_checks), stage=evidence.stage,
            evidence_path="tables/fit_pve_recovery.tsv;tables/omnib_power_replicates.tsv",
            reason="realized signal PVE differs from target by at most 0.01",
        )
    for scenario_id, ok in sorted(encoding_exact.items()):
        gate_id = f"B.encoding_exact.{scenario_id}"
        gates[gate_id] = _boolean_gate(
            gate_id=gate_id, track="omnib", scenario_id=scenario_id,
            kind="released_encoding_exact_invariance", ok=ok,
            stage=evidence.stage,
            evidence_path="tables/omnib_encoding_robustness.tsv",
            reason="all six required released-scorer exact checks pass",
        )
    for payload, _scenario, summary in scaling_payloads:
        anchor = summary.get("anchor", {}).get("anchor_id", payload["scenario_id"])
        exact = all(summary.get(field) is True for field in (
            "exact_result_hash_identity", "exact_family_hash_identity",
            "exact_ranking_hash_identity",
        ))
        gate_id = f"C.exact_identity.{anchor}"
        gates[gate_id] = _boolean_gate(
            gate_id=gate_id, track="scaling", scenario_id=str(payload["scenario_id"]),
            kind="serial_parallel_identity", ok=exact, stage=evidence.stage,
            evidence_path="tables/scaling_runs.tsv",
            reason="statistical arrays, ordered family and rankings match across jobs",
        )
        contract_id = f"C.execution_contract.{anchor}"
        gates[contract_id] = _boolean_gate(
            gate_id=contract_id, track="scaling",
            scenario_id=str(payload["scenario_id"]), kind="scaling_execution_contract",
            ok=(summary.get("native_thread_contract_valid") is True
                and summary.get("parallel_execution_contract_valid") is True),
            stage=evidence.stage, evidence_path="tables/scaling_runs.tsv",
            reason="native one-thread limits, backend, effective jobs and worker PIDs agree",
        )
        if anchor == "small_qa":
            accepted = summary.get("release_policy", {}).get("accepted") is True
            gates["C.small_qa_release"] = _boolean_gate(
                gate_id="C.small_qa_release", track="scaling",
                scenario_id=str(payload["scenario_id"]), kind="speed_memory_release",
                ok=accepted, stage=evidence.stage,
                evidence_path="tables/scaling_runs.tsv",
                reason="four workers faster than serial and aggregate PSS <= 1.35x serial",
            )
    for scenario_id, ok in sorted(application_acceptance.items()):
        gate_id = f"D.audited.{scenario_id}"
        gates[gate_id] = _boolean_gate(
            gate_id=gate_id, track="application", scenario_id=scenario_id,
            kind="read_only_application_audit", ok=ok, stage=evidence.stage,
            evidence_path="tables/cross_species_application.tsv",
            reason="all frozen application rows retain an authoritative successful audit",
        )
    return gates


def _fit_scan_gates(
    evidence: LoadedEvidence,
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, AuditGate]:
    """Aggregate Track A scan calibration/power over frozen scenario denominators."""

    registry = {row.scenario_id: row for row in evidence.registry}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows["fit_scan_metrics.tsv"]:
        grouped[(str(row["scenario_id"]), str(row["method"]))].append(row)
    for scenario in evidence.registry:
        if scenario.track != "fit" or scenario.parameters.get("experiment") not in {
            "scan", "loco",
        }:
            continue
        expected_methods = (
            {"canonical_multi_kernel"}
            if scenario.parameters.get("experiment") == "loco"
            else {
                "canonical_multi_kernel", "pooled_trace_sum",
                "independent_subgenome",
            }
        )
        observed_methods = {
            method for scenario_id, method in grouped
            if scenario_id == scenario.scenario_id
        }
        if observed_methods != expected_methods:
            raise BenchmarkAuditError("Track A scan canonical method set is incomplete")
    gates: dict[str, AuditGate] = {}
    null_gate_by_key: dict[tuple[str, str | None, str, str], AuditGate] = {}
    for (scenario_id, method), method_rows in sorted(grouped.items()):
        scenario = registry[scenario_id]
        expected = scenario.replicates
        if len(method_rows) != expected:
            raise BenchmarkAuditError("Track A scan method denominator differs from registry")
        panel = str(scenario.parameters.get("panel"))
        placement = scenario.parameters.get("placement")
        arm = "loco_sensitivity" if scenario.parameters.get("experiment") == "loco" else "primary"
        if any(row.get("scan_arm") != arm for row in method_rows):
            raise BenchmarkAuditError("Track A scan arm differs from registry")
        failures = sum(row.get("failed") is True for row in method_rows)
        pve = float(scenario.parameters.get("scan_pve", -1.0))
        outcome_field = "rejected" if pve == 0.0 else "causal_detected"
        if any(
            not isinstance(row.get(outcome_field), bool)
            for row in method_rows if row.get("failed") is not True
        ):
            raise BenchmarkAuditError("Track A scan outcome is missing")
        successes = sum(row.get(outcome_field) is True for row in method_rows)
        summary = summarize_binomial(successes, expected)
        failure_rate = failures / expected
        key = (panel, placement, arm, method)
        if pve == 0.0:
            scientific = summary["ci_high"] <= 0.075 and failure_rate <= 0.01
            gate_id = f"A.scan_fwer.{scenario_id}.{method}"
            gate = AuditGate(
                gate_id, "fit", scenario_id, "scan_fwer", successes, expected,
                float(summary["estimate"]), float(summary["ci_low"]),
                float(summary["ci_high"]), failures, failure_rate,
                scientific if evidence.stage == "formal" else None,
                (failure_rate <= 0.01) if evidence.stage == "pilot" else None,
                ("PASS" if scientific else "FAIL") if evidence.stage == "formal"
                else ("QA_PASS" if failure_rate <= 0.01 else "QA_FAIL"),
                "tables/fit_scan_metrics.tsv",
                "null any-rejection FWER upper Wilson <= 0.075; failures <= 1%",
            )
            null_gate_by_key[key] = gate
            gates[gate_id] = gate
        else:
            null_gate = null_gate_by_key.get(key)
            null_ok = (
                null_gate is not None
                and (
                    null_gate.passed is True if evidence.stage == "formal"
                    else null_gate.qa_passed is True
                )
            )
            scientific = failure_rate <= 0.01 and null_ok
            gate_id = f"A.scan_power.{scenario_id}.{method}"
            gates[gate_id] = AuditGate(
                gate_id, "fit", scenario_id, "scan_power_descriptive",
                successes, expected, float(summary["estimate"]),
                float(summary["ci_low"]), float(summary["ci_high"]),
                failures, failure_rate,
                scientific if evidence.stage == "formal" else None,
                scientific if evidence.stage == "pilot" else None,
                ("PASS" if scientific else "FAIL") if evidence.stage == "formal"
                else ("QA_PASS" if scientific else "QA_FAIL"),
                "tables/fit_scan_metrics.tsv",
                "power is descriptive and eligible only when its matching null method passes",
            )
    return gates


def _fit_non_scan_failure_gates(evidence: LoadedEvidence) -> dict[str, AuditGate]:
    registry = {row.scenario_id: row for row in evidence.registry}
    payloads: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for _path, payload in evidence.shards:
        if payload["track"] == "fit":
            payloads[str(payload["scenario_id"])].append(payload)
    gates: dict[str, AuditGate] = {}
    for scenario_id, scenario_payloads in sorted(payloads.items()):
        scenario = registry[scenario_id]
        if scenario.parameters.get("experiment") in {"scan", "loco"}:
            continue
        if len(scenario_payloads) != scenario.replicates:
            raise BenchmarkAuditError("Track A fixed denominator differs from registry")
        failures = sum(_failed(payload) for payload in scenario_payloads)
        rate = failures / scenario.replicates
        ok = rate <= 0.01
        gate_id = f"A.failure_rate.{scenario_id}"
        gates[gate_id] = AuditGate(
            gate_id, "fit", scenario_id, "fit_failure_rate",
            scenario.replicates - failures, scenario.replicates,
            (scenario.replicates - failures) / scenario.replicates,
            *wilson_interval(scenario.replicates - failures, scenario.replicates),
            failures, rate, ok if evidence.stage == "formal" else None,
            ok if evidence.stage == "pilot" else None,
            ("PASS" if ok else "FAIL") if evidence.stage == "formal"
            else ("QA_PASS" if ok else "QA_FAIL"),
            f"{evidence.stage}/fit/{scenario_id}",
            "successful fixed-denominator Track A shards; failures <= 1%",
        )
    return gates


def _acceptance_rows(
    stage: str, gates: Mapping[str, AuditGate],
) -> list[dict[str, Any]]:
    inference = "formal" if stage == "formal" else "noninferential_do_not_threshold"
    rows: list[dict[str, Any]] = []
    for gate_id in sorted(gates):
        gate = gates[gate_id]
        rows.append({
            "stage": stage, "inference_status": inference, "gate_id": gate.gate_id,
            "track": gate.track, "scenario_id": gate.scenario_id,
            "gate_kind": gate.gate_kind, "successes": gate.successes,
            "total": gate.total, "estimate": gate.estimate, "ci_low": gate.lower_ci,
            "ci_high": gate.upper_ci, "failures": gate.failures,
            "failure_rate": gate.failure_rate,
            "passed": gate.passed if stage == "formal" else None,
            "status": gate.status, "evidence_path": gate.evidence_path,
            "reason": gate.reason,
        })
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(value) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False,
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _verify_written_tables(
    paths: Sequence[Path], expected_hashes: Mapping[str, str], stage: str,
) -> None:
    schemas = table_schemas(stage)
    for path in paths:
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)
                list(reader)
        except (OSError, UnicodeError, StopIteration, csv.Error) as error:
            raise BenchmarkAuditError(f"aggregate table is unreadable: {path.name}") from error
        if header != list(schemas[path.name]):
            raise BenchmarkAuditError(f"aggregate table schema mismatch: {path.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hashes[path.name]:
            raise BenchmarkAuditError(f"aggregate table hash mismatch: {path.name}")


def _restore_publication(snapshots: Mapping[Path, bytes | None]) -> None:
    """Restore every live artifact after a failed locked publication."""

    errors: list[str] = []
    for path, previous in snapshots.items():
        try:
            if previous is None:
                if path.exists() and not path.is_symlink():
                    path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.rollback-", dir=path.parent
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(previous)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        except OSError as error:
            errors.append(f"{path}: {error}")
    if errors:
        raise BenchmarkAuditError(
            "benchmark publication rollback failed: " + "; ".join(errors)
        )


def _audit_benchmark_locked(root: str | Path) -> AuditReport:
    """Recompute all acceptance evidence and seal the first successful audit.

    The first invocation can only seal the current outcome bytes: a mutation made
    before this seal is not distinguishable from original output.  Every later
    invocation checks the frozen ordered path/SHA-256 manifest before rewriting
    any aggregate artifact.
    """

    benchmark_root = Path(root).resolve()
    for output_name in ("audit", "tables"):
        if (benchmark_root / output_name).is_symlink():
            raise BenchmarkAuditError("benchmark output directories must not be symlinks")
    audit_path = benchmark_root / "audit" / "benchmark_audit.json"
    if audit_path.is_symlink():
        raise BenchmarkAuditError("benchmark audit document must not be a symlink")
    old = _strict_old_audit(audit_path)
    try:
        evidence = load_evidence(root)
    except BenchmarkAggregateError as error:
        raise BenchmarkAuditError(str(error)) from error
    manifest = _shard_manifest(evidence)
    _verify_old_seal(old, manifest, evidence)
    _audit_seed_roles(evidence)
    _audit_derived_seeds_and_requests(evidence)
    _audit_families_and_parallel(evidence)
    gates: dict[str, AuditGate] = {"all.provenance": _provenance_gate(evidence)}
    gates.update(_pilot_and_engineering_gates(evidence))
    rows = build_table_rows(evidence)
    gates.update(_fit_non_scan_failure_gates(evidence))
    gates.update(_fit_scan_gates(evidence, rows))
    payloads_by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for _path, payload in evidence.shards:
        payloads_by_scenario[str(payload["scenario_id"])].append(payload)
    for scenario in evidence.registry:
        gate_id = f"all.evidence.{scenario.scenario_id}"
        # Loading has already established exact shard completeness and schema.
        # Explicit failures remain evidence and are assessed by statistical gates.
        ok = bool(payloads_by_scenario[scenario.scenario_id])
        gate = _boolean_gate(
            gate_id=gate_id, track=scenario.track,
            scenario_id=scenario.scenario_id, kind="scenario_evidence",
            ok=ok, stage=evidence.stage,
            evidence_path=(
                f"{evidence.stage}/{scenario.track}/{scenario.scenario_id}"
            ),
            reason="canonical scenario has its complete immutable shard contract",
        )
        if scenario.parameters.get("stress") is True:
            gate = AuditGate(
                gate.gate_id, gate.track, gate.scenario_id, "stress_evidence",
                gate.successes, gate.total, gate.estimate, gate.lower_ci,
                gate.upper_ci, gate.failures, gate.failure_rate, None, None,
                "DESCRIPTIVE_STRESS", gate.evidence_path,
                "predeclared stress scenario is descriptive and excluded from acceptance",
            )
        gates[gate_id] = gate

    registry = {row.scenario_id: row for row in evidence.registry}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for _path, payload in evidence.shards:
        grouped[str(payload["scenario_id"])].append(payload)
    for scenario_id, payloads in sorted(grouped.items()):
        row = registry[scenario_id]
        if row.track != "omnib" or row.parameters.get("experiment") != "end2end":
            continue
        null_model = row.parameters.get("null_model")
        if any(
            payload.get("null_model") != null_model
            or not isinstance(payload.get("null_generation"), Mapping)
            or payload["null_generation"].get("kind") != null_model
            or payload["null_generation"].get("canonical_kind")
            != _CANONICAL_NULL_KIND.get(str(null_model))
            for payload in payloads if not _failed(payload)
        ):
            raise BenchmarkAuditError("end-to-end null metadata differs from registry")
        stress = bool(row.parameters.get("stress", False)) or null_model in {
            "student_t5", "heteroskedastic_pc1", "contamination_1pct",
            "omitted_background_kernel",
        }
        failures = sum(_failed(payload) for payload in payloads)
        rejections = sum(
            _audit_end2end_decisions(payload)
            for payload in payloads if not _failed(payload)
        )
        gate = core_fwer_gate(
            scenario_id, rejections, len(payloads), failures,
            stage=evidence.stage,
            evidence_path="tables/omnib_null_replicates.tsv",
        )
        if stress:
            gate = AuditGate(
                gate.gate_id.replace("core_fwer", "stress_fwer"), gate.track,
                gate.scenario_id, "stress_fwer", gate.successes, gate.total,
                gate.estimate, gate.lower_ci, gate.upper_ci, gate.failures,
                gate.failure_rate, None, gate.qa_passed if evidence.stage == "pilot" else None,
                "DESCRIPTIVE_STRESS", gate.evidence_path,
                "stress null is descriptive and never contributes to acceptance",
            )
        gates[gate.gate_id] = gate
    gates.update(_conditional_gates(evidence, rows))
    gates.update(_global_vc_gates(evidence))
    gates.update(_omnib_power_gates(evidence, rows, gates))
    expected_minimum = {
        "fit_pve_recovery.tsv": sum(
            row.replicates for row in evidence.registry
            if row.track == "fit" and row.parameters.get("experiment") == "recovery"
        ),
        "fit_pve_coverage.tsv": sum(
            row.replicates for row in evidence.registry
            if row.track == "fit" and row.parameters.get("experiment") == "coverage"
        ),
        "fit_scan_metrics.tsv": sum(
            row.replicates for row in evidence.registry
            if row.track == "fit" and row.parameters.get("experiment") in {"scan", "loco"}
        ),
        "omnib_null_replicates.tsv": sum(
            row.replicates for row in evidence.registry
            if row.track == "omnib" and row.parameters.get("experiment") in {
                "end2end", "conditional", "global_vc", "family_size",
            }
        ),
        "omnib_power_replicates.tsv": sum(
            row.replicates for row in evidence.registry
            if row.track == "omnib" and row.parameters.get("experiment") == "power"
        ),
        "omnib_encoding_robustness.tsv": sum(
            len(_ENCODING_EXACT_CHECKS) + len(_ENCODING_ROBUSTNESS_CHECKS)
            for row in evidence.registry
            if row.track == "omnib" and row.parameters.get("experiment") == "encoding"
        ),
        "omnib_family_manifest.tsv": sum(
            len(payloads_by_scenario[row.scenario_id]) for row in evidence.registry
            if row.track == "omnib" and row.parameters.get("experiment") != "global_vc"
        ),
        "scaling_runs.tsv": sum(
            len(row.parameters["anchor"]["jobs"])
            * (1 if row.stage == "pilot" else 3)
            for row in evidence.registry if row.track == "scaling"
        ),
        "cross_species_application.tsv": sum(
            1 for row in evidence.registry if row.track == "application"
        ),
    }
    evidence_tables_complete = all(
        len(rows[name]) >= expected_minimum[name]
        for name in TABLE_SCHEMAS if name != "benchmark_acceptance.tsv"
    )
    gates["all.tables_complete"] = _boolean_gate(
        gate_id="all.tables_complete", track="all", scenario_id="*",
        kind="table_completeness", ok=evidence_tables_complete,
        stage=evidence.stage, evidence_path="tables/",
        reason="all nine predeclared evidence tables contain canonical rows",
    )
    _annotate_power_eligibility(evidence.stage, rows, gates)
    _annotate_binomial_rows(rows)

    acceptance_rows = _acceptance_rows(evidence.stage, gates)
    publication_snapshots: dict[Path, bytes | None] = {}
    with tempfile.TemporaryDirectory(
        prefix=".audit-staging-", dir=benchmark_root,
    ) as staging_directory:
        staging_root = Path(staging_directory)
        staged_evidence = replace(evidence, root=staging_root)
        staged_paths, table_hashes = write_tables(
            staged_evidence, rows, acceptance_rows
        )
        _verify_written_tables(staged_paths, table_hashes, evidence.stage)
        if old is not None and old.get("table_sha256") != table_hashes:
            raise BenchmarkAuditError("aggregate table hash mismatch against sealed audit")
        live_table_root = benchmark_root / "tables"
        live_table_root.mkdir(parents=True, exist_ok=True)
        paths_list: list[Path] = []
        try:
            for staged_path in staged_paths:
                live_path = live_table_root / staged_path.name
                if live_path.is_symlink():
                    raise BenchmarkAuditError("aggregate table path must not be a symlink")
                publication_snapshots[live_path] = (
                    live_path.read_bytes() if live_path.exists() else None
                )
                os.replace(staged_path, live_path)
                paths_list.append(live_path)
        except Exception:
            _restore_publication(publication_snapshots)
            raise
        paths = tuple(paths_list)

    required = [
        gate for gate in gates.values()
        if gate.gate_kind not in {"stress_fwer", "stress_evidence"}
    ]
    formal_overall = (
        all(gate.passed is True for gate in required)
        if evidence.stage == "formal" else None
    )
    qa_overall = (
        all(gate.qa_passed is True for gate in required)
        if evidence.stage == "pilot" else None
    )
    inference = (
        "formal" if evidence.stage == "formal"
        else "noninferential_do_not_threshold"
    )
    serialized_gates = {}
    for gate_id, gate in sorted(gates.items()):
        serialized = asdict(gate)
        serialized.pop("qa_passed" if evidence.stage == "formal" else "passed")
        serialized_gates[gate_id] = serialized
    document = {
        "schema": "homoeogwas-v201-benchmark-audit-v1",
        "stage": evidence.stage,
        "inference_status": inference,
        "design_hash": evidence.design_hash,
        "scenario_registry_sha256": evidence.registry_sha256,
        "gates": serialized_gates,
        "shard_manifest": list(manifest),
        "shard_manifest_sha256": sha256_payload(list(manifest)),
        "table_sha256": table_hashes,
        "first_audit_seal_boundary": (
            "The first successful audit seals current ordered shard bytes; "
            "pre-seal mutation is not distinguishable from original output."
        ),
    }
    if evidence.stage == "formal":
        document["formal_overall_passed"] = formal_overall
    else:
        document["qa_overall_passed"] = qa_overall
    document["audit_sha256"] = sha256_payload(document)
    publication_snapshots[audit_path] = (
        audit_path.read_bytes() if audit_path.exists() else None
    )
    try:
        _atomic_json(audit_path, document)
    except Exception:
        _restore_publication(publication_snapshots)
        raise
    return AuditReport(
        evidence.stage, inference, formal_overall, qa_overall, gates,
        tuple(manifest), paths, table_hashes, evidence,
    )


def audit_benchmark(root: str | Path) -> AuditReport:
    """Run one serialized audit so concurrent invocations cannot interleave."""

    requested = Path(root)
    if requested.is_symlink():
        raise BenchmarkAuditError("benchmark root must not be a symlink")
    benchmark_root = requested.resolve()
    lock_path = benchmark_root / ".benchmark-audit.lock"
    if lock_path.is_symlink():
        raise BenchmarkAuditError("benchmark audit lock must not be a symlink")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise BenchmarkAuditError("cannot open benchmark audit lock") from error
    with os.fdopen(descriptor, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            return _audit_benchmark_locked(benchmark_root)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "AuditGate", "AuditReport", "BenchmarkAuditError", "audit_benchmark",
    "core_fwer_gate", "summarize_binomial", "wilson_interval",
]
