"""Independent, fail-closed acceptance audit for v2.0.1 benchmark evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .aggregate import (
    TABLE_SCHEMAS,
    BenchmarkAggregateError,
    LoadedEvidence,
    build_table_rows,
    load_evidence,
    write_tables,
)
from .contracts import canonical_json, derive_seed, sha256_payload


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

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise BenchmarkAuditError("existing benchmark audit is unreadable") from error
    if not isinstance(value, dict):
        raise BenchmarkAuditError("existing benchmark audit is invalid")
    return value


def _shard_manifest(evidence: LoadedEvidence) -> tuple[dict[str, str], ...]:
    return tuple({
        "path": path.relative_to(evidence.root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path, _payload in evidence.shards)


def _verify_old_seal(
    old: Mapping[str, Any] | None,
    manifest: Sequence[Mapping[str, str]],
) -> None:
    if old is None:
        return
    old_manifest = old.get("shard_manifest")
    if old_manifest != list(manifest):
        raise BenchmarkAuditError("shard hash mismatch against first successful audit seal")


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


def _audit_end2end_decisions(payload: Mapping[str, Any]) -> bool:
    family_ids = payload.get("family_ids")
    observed = payload.get("observed_group_p")
    adjusted = payload.get("adjusted_p")
    adjusted_decisions = payload.get("adjusted_decisions")
    calibration = payload.get("bootstrap_minp")
    if (
        not isinstance(family_ids, list) or not isinstance(observed, list)
        or not isinstance(adjusted, list) or not isinstance(adjusted_decisions, list)
        or not isinstance(calibration, Mapping)
        or not len(family_ids) == len(observed) == len(adjusted) == len(adjusted_decisions)
    ):
        raise BenchmarkAuditError("end-to-end decision fields are incomplete")
    observed_values = [_finite_probability(value, "observed p") for value in observed]
    adjusted_values = [_finite_probability(value, "adjusted p") for value in adjusted]
    if not all(isinstance(value, bool) for value in adjusted_decisions):
        raise BenchmarkAuditError("serialized adjusted decisions are invalid")
    alpha = _finite_probability(calibration.get("alpha", 0.05), "alpha")
    threshold = calibration.get("threshold")
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
    if not isinstance(bootstrap_adjusted, list) or [float(v) for v in bootstrap_adjusted] != adjusted_values:
        raise BenchmarkAuditError("bootstrap and serialized adjusted p-values differ")
    expected_ids = {family_ids[index] for index in threshold_indices}
    serialized_ids = payload.get(
        "formal_rejections" if payload["stage"] == "formal"
        else "qa_diagnostic_rejections"
    )
    if not isinstance(serialized_ids, list) or set(serialized_ids) != expected_ids:
        raise BenchmarkAuditError("decision disagreement: serialized rejection IDs differ")
    if payload["stage"] == "pilot" and payload.get("formal_rejections"):
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
            gates[gate.gate_id] = gate
            continue
        if base not in calibration or base not in heldout:
            raise BenchmarkAuditError("conditional calibration/heldout pair is incomplete")
        cal_payload, held_payload = calibration[base], heldout[base]
        cal_bank, held_bank = cal_payload["bank"], held_payload["bank"]
        cal_seeds, held_seeds = set(cal_bank.get("seed_ids", [])), set(held_bank.get("seed_ids", []))
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
        cal_scores, held_scores = cal_bank.get("p_by_method"), held_bank.get("p_by_method")
        if not isinstance(cal_scores, Mapping) or not isinstance(held_scores, Mapping):
            raise BenchmarkAuditError("conditional score evidence is missing")
        if set(cal_scores) != set(held_scores):
            raise BenchmarkAuditError("conditional methods differ across response roles")
        for method in sorted(cal_scores):
            cal_matrix = _matrix(cal_scores[method], "calibration p")
            held_matrix = _matrix(held_scores[method], "heldout p")
            if len(cal_matrix) != len(held_matrix):
                raise BenchmarkAuditError("conditional tested family size changed")
            threshold = _empirical_threshold(cal_matrix)
            decisions = [
                threshold is not None and min(row[index] for row in held_matrix) < threshold
                for index in range(len(held_matrix[0]))
            ]
            failures = len(set(held_bank.get("failure", {}).get("failed_response_indices", [])))
            scenario_key = f"{held_payload['scenario_id']}.{method}"
            gate = core_fwer_gate(
                scenario_key, sum(decisions), len(decisions), failures,
                stage=evidence.stage,
                evidence_path="tables/omnib_null_replicates.tsv",
            )
            gates[gate.gate_id] = gate
            for row in rows["omnib_null_replicates.tsv"]:
                if row["scenario_id"] == held_payload["scenario_id"] and row["method"] == method:
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


def _audit_families_and_parallel(evidence: LoadedEvidence) -> None:
    tested_hashes: dict[tuple[str, str], tuple[int, str]] = {}
    for _path, payload in evidence.shards:
        if payload["track"] == "omnib":
            canonical = {
                "mode": "group", "statistic": "omniB",
                "hypothesis_unit": "group", "family_scope": "primary_only",
                "subset_order": 2, "direct_higher_order_term": False,
            }
            for field, expected in canonical.items():
                if payload.get(field) != expected:
                    raise BenchmarkAuditError(f"noncanonical omniB {field}")
            ids = payload.get("family_ids")
            if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
                raise BenchmarkAuditError("invalid ordered family IDs")
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
                if not isinstance(sizes, Mapping) or not isinstance(hashes, Mapping):
                    raise BenchmarkAuditError("tested family provenance is incomplete")
                if set(sizes) != set(hashes):
                    raise BenchmarkAuditError("tested family size/hash methods differ")
                for method in sizes:
                    size, digest = sizes[method], hashes[method]
                    if (
                        isinstance(size, bool) or not isinstance(size, int) or size < 1
                        or not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)
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
                methods = set(payload.get("thresholds", {}))
                if methods != set(payload.get("rejections_by_method", {})) or methods != set(
                    payload.get("causal_detection_by_method", {})
                ):
                    raise BenchmarkAuditError("power method decision fields differ")
                for method in methods:
                    threshold = payload["thresholds"][method]
                    if threshold is not None:
                        _finite_probability(threshold, "power threshold")
                    for field in ("rejections_by_method", "causal_detection_by_method"):
                        decisions = payload[field][method]
                        if not isinstance(decisions, list) or not all(
                            isinstance(value, bool) for value in decisions
                        ):
                            raise BenchmarkAuditError("power decisions are invalid")
        if payload["track"] == "scaling":
            summary = payload.get("summary", payload)
            for field in (
                "exact_result_hash_identity", "exact_family_hash_identity",
                "exact_ranking_hash_identity",
            ):
                if summary.get(field) is not True:
                    raise BenchmarkAuditError("serial/parallel array/family/ranking hash mismatch")
        if payload.get("experiment") == "encoding":
            checks = payload.get("exact_checks")
            if not isinstance(checks, Mapping):
                raise BenchmarkAuditError("encoding checks are missing")
            for check in checks.values():
                if check.get("required") and not all(check.get(field) is True for field in (
                    "observed_arrays_identical", "adjusted_decisions_identical",
                    "ranking_hash_identical", "rejection_sets_identical",
                )):
                    raise BenchmarkAuditError("required encoding identity check failed")


def _annotate_binomial_rows(rows: dict[str, list[dict[str, Any]]]) -> None:
    field_by_table = {
        "fit_pve_coverage.tsv": "covered",
        "fit_scan_metrics.tsv": "causal_detected",
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


def _annotate_power_eligibility(
    stage: str,
    rows: dict[str, list[dict[str, Any]]],
    gates: Mapping[str, AuditGate],
) -> None:
    for row in rows["omnib_power_replicates.tsv"]:
        parts = str(row["scenario_id"]).split(".")
        backbone = parts[2] if len(parts) > 2 else "unknown"
        gate_id = (
            f"B.core_fwer.B.conditional.{backbone}.heldout.{row['method']}"
        )
        gate = gates.get(gate_id)
        row["null_calibration_gate"] = gate.status if gate is not None else "MISSING"
        row["eligible_for_power_summary"] = (
            gate.passed if stage == "formal" and gate is not None else None
        )


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
    table_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, AuditGate]:
    gates: dict[str, AuditGate] = {}
    pve_checks: list[bool] = []
    edge_checks: list[bool] = []
    scaling_payloads: list[Mapping[str, Any]] = []
    registry = {row.scenario_id: row for row in evidence.registry}
    for _path, payload in evidence.shards:
        scenario = registry[str(payload["scenario_id"])]
        if payload["track"] == "fit":
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
        elif payload["track"] == "scaling":
            scaling_payloads.append(payload)
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
    for payload in scaling_payloads:
        summary = payload.get("summary", payload)
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
        if anchor == "small_qa":
            accepted = summary.get("release_policy", {}).get("accepted") is True
            gates["C.small_qa_release"] = _boolean_gate(
                gate_id="C.small_qa_release", track="scaling",
                scenario_id=str(payload["scenario_id"]), kind="speed_memory_release",
                ok=accepted, stage=evidence.stage,
                evidence_path="tables/scaling_runs.tsv",
                reason="four workers faster than serial and aggregate PSS <= 1.35x serial",
            )
    by_application: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in table_rows["cross_species_application.tsv"]:
        by_application[str(row["scenario_id"])].append(row)
    for scenario_id, application_rows in sorted(by_application.items()):
        ok = bool(application_rows) and all(
            row.get("repair_required") is False and bool(row.get("audit_status"))
            for row in application_rows
        )
        gate_id = f"D.audited.{scenario_id}"
        gates[gate_id] = _boolean_gate(
            gate_id=gate_id, track="application", scenario_id=scenario_id,
            kind="read_only_application_audit", ok=ok, stage=evidence.stage,
            evidence_path="tables/cross_species_application.tsv",
            reason="all frozen application rows retain an authoritative successful audit",
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
    paths: Sequence[Path], expected_hashes: Mapping[str, str],
) -> None:
    for path in paths:
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)
                list(reader)
        except (OSError, UnicodeError, StopIteration, csv.Error) as error:
            raise BenchmarkAuditError(f"aggregate table is unreadable: {path.name}") from error
        if header != list(TABLE_SCHEMAS[path.name]):
            raise BenchmarkAuditError(f"aggregate table schema mismatch: {path.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hashes[path.name]:
            raise BenchmarkAuditError(f"aggregate table hash mismatch: {path.name}")


def audit_benchmark(root: str | Path) -> AuditReport:
    """Recompute all acceptance evidence and seal the first successful audit.

    The first invocation can only seal the current outcome bytes: a mutation made
    before this seal is not distinguishable from original output.  Every later
    invocation checks the frozen ordered path/SHA-256 manifest before rewriting
    any aggregate artifact.
    """

    audit_path = Path(root).resolve() / "audit" / "benchmark_audit.json"
    old = _strict_old_audit(audit_path)
    try:
        evidence = load_evidence(root)
    except BenchmarkAggregateError as error:
        raise BenchmarkAuditError(str(error)) from error
    manifest = _shard_manifest(evidence)
    _verify_old_seal(old, manifest)
    _audit_seed_roles(evidence)
    _audit_derived_seeds_and_requests(evidence)
    _audit_families_and_parallel(evidence)
    rows = build_table_rows(evidence)
    gates: dict[str, AuditGate] = {"all.provenance": _provenance_gate(evidence)}
    gates.update(_pilot_and_engineering_gates(evidence, rows))

    registry = {row.scenario_id: row for row in evidence.registry}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for _path, payload in evidence.shards:
        grouped[str(payload["scenario_id"])].append(payload)
    for scenario_id, payloads in sorted(grouped.items()):
        row = registry[scenario_id]
        if row.track != "omnib" or row.parameters.get("experiment") != "end2end":
            continue
        null_model = row.parameters.get("null_model")
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
    _annotate_power_eligibility(evidence.stage, rows, gates)
    _annotate_binomial_rows(rows)

    acceptance_rows = _acceptance_rows(evidence.stage, gates)
    paths, table_hashes = write_tables(evidence, rows, acceptance_rows)
    _verify_written_tables(paths, table_hashes)
    if old is not None and old.get("table_sha256") != table_hashes:
        raise BenchmarkAuditError("aggregate table hash mismatch against sealed audit")

    required = [gate for gate in gates.values() if gate.gate_kind != "stress_fwer"]
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
    _atomic_json(audit_path, document)
    return AuditReport(
        evidence.stage, inference, formal_overall, qa_overall, gates,
        tuple(manifest), paths, table_hashes, evidence,
    )


__all__ = [
    "AuditGate", "AuditReport", "BenchmarkAuditError", "audit_benchmark",
    "core_fwer_gate", "summarize_binomial", "wilson_interval",
]
