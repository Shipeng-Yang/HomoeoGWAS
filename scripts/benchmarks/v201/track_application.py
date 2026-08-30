"""Read-only export of frozen cross-species HomoeoGWAS application evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from homoeogwas.run_registry import RegistryRun, load_registry

_VALID_AUDIT_STATUSES = {
    "AUDIT_COMPLETE",
    "INTERNAL_DISCOVERY_REPLICATION_REQUIRED",
    "REVIEW_REQUIRED",
}
_VALID_RECORD_STATUSES = {
    "ANALYSIS_INVALID",
    "INTERNAL_DISCOVERY_REPLICATION_REQUIRED",
    "INTERNAL_DISCOVERY_REPLICATION_RECORDED",
    "NO_FAMILYWISE_DISCOVERY_REVIEW_REQUIRED",
    "NO_FAMILYWISE_DISCOVERY",
}


class _ExportFailure(Exception):
    def __init__(
        self,
        status: str,
        reason: str,
        *,
        audit_status: str | None = None,
        partial: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.audit_status = audit_status
        self.partial = dict(partial or {})


def _empty_row(run: RegistryRun) -> dict[str, Any]:
    return {
        "analysis_id": run.id,
        "analysis_shape": run.analysis_shape,
        "panel": run.panel,
        "species": run.species,
        "ploidy": 2 * len(run.subgenomes),
        "subgenomes": list(run.subgenomes),
        "sample_count": None,
        "marker_count": None,
        "group_family_count": None,
        "edge_family_count": None,
        "requested_jobs": None,
        "effective_jobs": None,
        "backend": None,
        "worker_pids": None,
        "primary_unit": None,
        "calibration_method": None,
        "calibration_B": None,
        "adjusted_discovery_count": None,
        "adjusted_discoveries": None,
        "negative_result": None,
        "component_driver_distribution": None,
        "audit_status": None,
        "family_hash": None,
        "limitations": [],
        "status": None,
        "repair_required": True,
        "repair_reason": None,
    }


def _failure_row(run: RegistryRun, failure: _ExportFailure) -> dict[str, Any]:
    row = _empty_row(run)
    row.update(failure.partial)
    row.update({
        "status": failure.status,
        "repair_required": True,
        "repair_reason": failure.reason,
    })
    if failure.audit_status is not None:
        row["audit_status"] = failure.audit_status
    return row


def _read_json_object(body: bytes, path: Path, run_id: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            body.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise _ExportFailure(
            "UNREADABLE_AUTHORITATIVE_OUTPUT",
            f"run {run_id!r}: unreadable authoritative JSON {path}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise _ExportFailure(
            "UNREADABLE_AUTHORITATIVE_OUTPUT",
            f"run {run_id!r}: authoritative JSON is not an object: {path}",
        )
    return value


def _read_declared_artifacts(
    run: RegistryRun,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], dict[str, str]]:
    root = run.result_root
    if root is None or not root.exists() or not root.is_dir():
        raise _ExportFailure(
            "MISSING_AUTHORITATIVE_OUTPUT",
            f"run {run.id!r}: authoritative result root is missing: {root}",
        )
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _ExportFailure(
            "UNREADABLE_AUTHORITATIVE_OUTPUT",
            f"run {run.id!r}: cannot resolve authoritative result root {root}: {exc}",
        ) from exc
    if not run.artifact_inventory:
        raise _ExportFailure(
            "UNDECLARED_AUTHORITATIVE_INVENTORY",
            f"run {run.id!r}: no frozen artifact_inventory is declared",
        )
    by_role: dict[str, list[Mapping[str, Any]]] = {"result": [], "audit": []}
    for record in run.artifact_inventory:
        by_role[str(record["role"])].append(record)
    if len(by_role["result"]) != 1 or len(by_role["audit"]) != 1:
        raise _ExportFailure(
            "AMBIGUOUS_AUTHORITATIVE_INVENTORY",
            f"run {run.id!r}: exporter requires exactly one declared result and "
            "one declared audit, observed result={len(by_role['result'])}, "
            f"audit={len(by_role['audit'])}",
        )

    resolved: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for role in ("result", "audit"):
        record = by_role[role][0]
        path = root / str(record["path"])
        try:
            resolved_path = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise _ExportFailure(
                "MISSING_AUTHORITATIVE_OUTPUT",
                f"run {run.id!r}: declared {role} artifact is missing: {path}",
            ) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise _ExportFailure(
                "UNREADABLE_AUTHORITATIVE_OUTPUT",
                f"run {run.id!r}: cannot resolve declared {role} artifact "
                f"{path}: {exc}",
            ) from exc
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise _ExportFailure(
                "AUTHORITATIVE_PATH_ESCAPE",
                f"run {run.id!r}: declared {role} artifact resolves outside "
                f"result_root: {path} -> {resolved_path}",
            ) from exc
        try:
            body = resolved_path.read_bytes()
        except OSError as exc:
            raise _ExportFailure(
                "UNREADABLE_AUTHORITATIVE_OUTPUT",
                f"run {run.id!r}: cannot read declared {role} artifact "
                f"{resolved_path}: {exc}",
            ) from exc
        observed_size = len(body)
        observed_hash = hashlib.sha256(body).hexdigest()
        if observed_size != record["size"] or observed_hash != record["sha256"]:
            raise _ExportFailure(
                "AUTHORITATIVE_ARTIFACT_MISMATCH",
                f"run {run.id!r}: declared {role} artifact identity mismatch at "
                f"{path} (size={observed_size}, sha256={observed_hash})",
            )
        resolved[role] = (
            resolved_path,
            _read_json_object(body, resolved_path, run.id),
            observed_hash,
        )
    result_path, result, result_hash = resolved["result"]
    audit_path, audit, audit_hash = resolved["audit"]
    return result_path, result, audit_path, audit, {
        "result": result_hash, "audit": audit_hash,
    }


def _integer(value: Any, label: str, run_id: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run_id!r}: {label} must be an integer >= {minimum}",
        )
    return value


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _primary_result(result: Mapping[str, Any], run_id: str) -> Mapping[str, Any]:
    provenance = result.get("provenance")
    results = result.get("results")
    if not isinstance(provenance, Mapping) or not isinstance(results, Mapping):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run_id!r}: result lacks provenance/results objects",
        )
    transform = provenance.get("primary_transform", "INT")
    primary = results.get(transform)
    if not isinstance(primary, Mapping):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run_id!r}: result lacks declared primary transform {transform!r}",
        )
    return primary


def _primary_unit(
    run: RegistryRun, result: Mapping[str, Any], primary: Mapping[str, Any],
) -> str | None:
    provenance = result["provenance"]
    unit = provenance.get("hypothesis_unit") or result.get("hypothesis_unit")
    if unit is None:
        mode = str(provenance.get("mode") or result.get("mode") or "").lower()
        if mode == "pairwise" and "pairwise" in str(run.analysis_shape).lower():
            unit = "edge"
        elif (
            mode == "triad"
            and run.expected.get("primary_unit") == "group"
            and "group" in str(run.analysis_shape).lower()
        ):
            unit = "group"
    if unit is None:
        fwer = (primary.get("model_diagnostics") or {}).get("bootstrap_fwer")
        if isinstance(fwer, Mapping):
            unit = fwer.get("declared_hypothesis_unit") or fwer.get("family_id")
    return str(unit).lower() if unit in {"edge", "group"} else None


def _marker_count(
    run: RegistryRun, provenance: Mapping[str, Any], primary: Mapping[str, Any],
    audit_status: str | None = None,
) -> int | None:
    diagnostics = provenance.get("grm_filter_diagnostics")
    if not isinstance(diagnostics, Mapping):
        model_diagnostics = primary.get("model_diagnostics")
        if isinstance(model_diagnostics, Mapping):
            diagnostics = model_diagnostics.get("grm_provenance")
    if not isinstance(diagnostics, Mapping):
        return None
    subgenomes = diagnostics.get("subgenomes")
    if not isinstance(subgenomes, Mapping) or not subgenomes:
        return None
    if set(subgenomes) != set(run.subgenomes):
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run.id!r}: marker diagnostic subgenomes do not match registry",
            audit_status=audit_status,
        )
    counts = []
    for value in subgenomes.values():
        if not isinstance(value, Mapping):
            return None
        count = value.get("n_variants_input")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        counts.append(count)
    return sum(counts)


def _adjusted_discoveries(
    primary: Mapping[str, Any], run_id: str, audit_status: str | None = None,
) -> tuple[int, list[dict[str, Any]], dict[str, int]]:
    count = _integer(primary.get("n_sig"), "n_sig", run_id)
    significant = primary.get("sig")
    if not isinstance(significant, list) or len(significant) != count:
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run_id!r}: discovery count does not match significant-unit list",
            audit_status=audit_status,
        )
    exported = []
    drivers: Counter[str] = Counter()
    for index, unit in enumerate(significant):
        if not isinstance(unit, Mapping):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run_id!r}: significant unit {index} is malformed",
                audit_status=audit_status,
            )
        identifier = unit.get("hypothesis_id") or unit.get("edge_id") or unit.get("pair")
        adjusted = unit.get("p_adjusted_bootstrap_minp")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or not isinstance(adjusted, (int, float))
            or isinstance(adjusted, bool)
            or not math.isfinite(adjusted)
            or not 0.0 <= float(adjusted) <= 1.0
        ):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run_id!r}: significant unit {index} lacks finite adjusted p-value "
                "or identifier",
                audit_status=audit_status,
            )
        exported.append({
            "hypothesis_id": identifier.strip(), "adjusted_p": float(adjusted),
        })
        driver = unit.get("smallest_component")
        if isinstance(driver, str) and driver:
            drivers[driver] += 1
    return count, exported, dict(drivers)


def _audit_input_is_bound(
    value: str, result_path: Path, result_root: Path,
) -> bool:
    supplied = Path(value)
    resolved_result = result_path.resolve(strict=True)
    resolved_root = result_root.resolve(strict=True)
    if supplied.is_absolute():
        return supplied.resolve(strict=True) in {resolved_result, resolved_root}
    parts = supplied.parts
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        return False
    return any(
        len(target.parts) >= len(parts)
        and target.parts[-len(parts):] == parts
        for target in (resolved_result, resolved_root)
    )


def _audit_status_and_record(
    run: RegistryRun,
    result_path: Path,
    result_hash: str,
    audit: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any] | None]:
    if "overall_status" in audit:
        schema_version = audit.get("audit_schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: production audit_schema_version must equal 1",
            )
        created_utc = audit.get("created_utc")
        try:
            if not isinstance(created_utc, str) or not created_utc.strip():
                raise ValueError("empty timestamp")
            datetime.fromisoformat(created_utc.strip())
        except ValueError as exc:
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: production audit created_utc is not ISO time",
            ) from exc
        audit_input = audit.get("input")
        if not isinstance(audit_input, str) or not audit_input.strip():
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: production audit input is not a path",
            )
        try:
            input_is_bound = _audit_input_is_bound(
                audit_input, result_path, run.result_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: production audit input cannot be resolved: {exc}",
            ) from exc
        if not input_is_bound:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: production audit input is not bound to the "
                "declared result or result_root",
            )

        status = str(audit.get("overall_status", "")).strip().upper()
        records = audit.get("records")
        if (
            not isinstance(records, list)
            or len(records) != 1
            or audit.get("n_results") != 1
        ):
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: audit must contain exactly one record and n_results=1",
                audit_status=status,
            )
        record = records[0]
        if not isinstance(record, Mapping):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: audit record must be an object",
                audit_status=status,
            )
        required = {
            "source", "command", "trait", "mode", "statistic", "status",
            "discovery_count", "n", "n_planned", "n_valid", "calibration",
            "replication_status", "flags", "evidence_boundary",
        }
        missing = sorted(required - set(record))
        if missing:
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: audit record lacks required fields: "
                + ", ".join(missing),
                audit_status=status,
            )
        text_fields = (
            "source", "trait", "mode", "statistic", "status", "calibration",
            "replication_status",
        )
        invalid_text = [
            field for field in text_fields
            if not isinstance(record[field], str) or not record[field].strip()
        ]
        if invalid_text:
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: audit record text fields are invalid: "
                + ", ".join(invalid_text),
                audit_status=status,
            )
        if record["command"] != "interact":
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: audit command must be interact",
                audit_status=status,
            )
        try:
            source = Path(record["source"]).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: audit source is invalid: {exc}",
                audit_status=status,
            ) from exc
        if source != result_path.resolve():
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: audit source does not bind declared result",
                audit_status=status,
            )
        integer_fields = {
            "discovery_count": 0, "n": 1, "n_planned": 1, "n_valid": 0,
        }
        invalid_integers = [
            field for field, minimum in integer_fields.items()
            if isinstance(record[field], bool)
            or not isinstance(record[field], int)
            or record[field] < minimum
        ]
        if invalid_integers:
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: audit integer fields are invalid: "
                + ", ".join(invalid_integers),
                audit_status=status,
            )
        record_status = record["status"]
        if record_status not in _VALID_RECORD_STATUSES:
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: audit record status is unknown: {record_status}",
                audit_status=status,
            )
        if record["replication_status"] not in {"NOT_ASSESSED", "RECORDED"}:
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: audit replication_status is invalid",
                audit_status=status,
            )
        flags = record["flags"]
        if not isinstance(flags, list) or any(
            not isinstance(flag, Mapping)
            or set(flag) != {"code", "severity", "message"}
            or any(
                not isinstance(flag.get(field), str) or not flag[field].strip()
                for field in ("code", "severity", "message")
            )
            or flag.get("severity") not in {"info", "review", "error"}
            for flag in flags
        ):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: audit flags are malformed",
                audit_status=status,
            )
        boundary = record["evidence_boundary"]
        if (
            not isinstance(boundary, list)
            or not boundary
            or any(not isinstance(value, str) or not value.strip() for value in boundary)
        ):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: audit evidence_boundary is malformed",
                audit_status=status,
            )
        has_error = any(flag["severity"] == "error" for flag in flags)
        has_review = any(flag["severity"] == "review" for flag in flags)
        discovery_count = record["discovery_count"]
        replication_status = record["replication_status"]
        if has_error:
            computed_record = "ANALYSIS_INVALID"
        elif discovery_count > 0 and replication_status == "NOT_ASSESSED":
            computed_record = "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
        elif discovery_count > 0:
            computed_record = "INTERNAL_DISCOVERY_REPLICATION_RECORDED"
        elif has_review:
            computed_record = "NO_FAMILYWISE_DISCOVERY_REVIEW_REQUIRED"
        else:
            computed_record = "NO_FAMILYWISE_DISCOVERY"
        if computed_record == "ANALYSIS_INVALID":
            computed_overall = "ANALYSIS_INVALID"
        elif computed_record == "INTERNAL_DISCOVERY_REPLICATION_REQUIRED":
            computed_overall = computed_record
        elif "REVIEW_REQUIRED" in computed_record:
            computed_overall = "REVIEW_REQUIRED"
        else:
            computed_overall = "AUDIT_COMPLETE"
        if computed_overall == "ANALYSIS_INVALID" or status == "ANALYSIS_INVALID":
            raise _ExportFailure(
                "AUDIT_FAILED",
                f"run {run.id!r}: production audit recomputes to ANALYSIS_INVALID",
                audit_status="ANALYSIS_INVALID",
            )
        if status not in _VALID_AUDIT_STATUSES:
            raise _ExportFailure(
                "AUDIT_FAILED",
                f"run {run.id!r}: frozen audit failed with status {status or 'MISSING'}",
                audit_status=status or None,
            )
        if record_status != computed_record or status != computed_overall:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: declared audit record/overall status disagrees "
                "with production status recomputation",
                audit_status=status,
            )
        return status, record

    status = str(audit.get("status", "")).strip().upper()
    if status != "PASS":
        raise _ExportFailure(
            "AUDIT_FAILED",
            f"run {run.id!r}: frozen custom audit failed with status "
            f"{status or 'MISSING'}",
            audit_status=status or None,
        )
    hashes = audit.get("output_sha256")
    if not isinstance(hashes, Mapping) or result_hash not in hashes.values():
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run.id!r}: custom audit is not bound to declared result hash",
            audit_status=status,
        )
    return status, None


def _same_metric(
    run_id: str, label: str, expected: Any, observed: Any, audit_status: str,
) -> None:
    if expected is not None and observed is not None and expected != observed:
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run_id!r}: {label} mismatch, expected {expected!r}, "
            f"observed {observed!r}",
            audit_status=audit_status,
        )


def _validate_family_and_calibration(
    run: RegistryRun,
    result: Mapping[str, Any],
    primary: Mapping[str, Any],
    audit_record: Mapping[str, Any] | None,
    *,
    audit_status: str,
    unit: str | None,
    n_planned: int,
    n_valid: int,
    group_count: Any,
    edge_count: Any,
    calibration_b: Any,
    family_hash: Any,
) -> None:
    if n_valid > n_planned:
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run.id!r}: n_valid exceeds n_planned",
            audit_status=audit_status,
        )
    for label, value in (
        ("group_family_count", group_count), ("edge_family_count", edge_count),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: {label} must be a non-negative integer",
                audit_status=audit_status,
            )
    if family_hash is not None and not _is_sha256_hex(family_hash):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: family_hash is not 64 hexadecimal digits",
            audit_status=audit_status,
        )
    if calibration_b is not None and (
        isinstance(calibration_b, bool)
        or not isinstance(calibration_b, int)
        or calibration_b < 1
    ):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: calibration_B must be a positive integer",
            audit_status=audit_status,
        )

    diagnostics = primary.get("model_diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    fwer = diagnostics.get("bootstrap_fwer")
    if isinstance(fwer, Mapping):
        declared = fwer.get("declared_hypothesis_unit") or fwer.get("family_id")
        _same_metric(run.id, "bootstrap family unit", unit, declared, audit_status)
        _same_metric(
            run.id, "bootstrap family count", n_planned,
            fwer.get("n_hypotheses"), audit_status)
        _same_metric(
            run.id, "bootstrap calibration B", calibration_b,
            fwer.get("B"), audit_status)

    provenance = result["provenance"]
    primary_statistic = primary.get("statistic")
    provenance_statistic = provenance.get("statistic")
    if (
        not isinstance(primary_statistic, str)
        or not primary_statistic.strip()
        or not isinstance(provenance_statistic, str)
        or primary_statistic.lower() != provenance_statistic.lower()
    ):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: statistic is missing or inconsistent across layers",
            audit_status=audit_status,
        )
    primary_method = primary.get("calibration_method")
    provenance_method = provenance.get("calibration_method")
    if (
        not isinstance(primary_method, str)
        or not primary_method.strip()
        or not isinstance(provenance_method, str)
        or primary_method.lower() != provenance_method.lower()
    ):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: calibration_method is missing or inconsistent",
            audit_status=audit_status,
        )
    statistic = primary_statistic.lower()
    method = primary_method.lower()
    if (statistic, method) not in {
        ("omnib", "bootstrap"),
        ("triad3", "bootstrap"),
        ("burden", "permutation"),
    }:
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run.id!r}: unsupported statistic/calibration combination "
            f"{primary_statistic}+{primary_method}",
            audit_status=audit_status,
        )

    scope = provenance.get("family_scope", "primary_only")
    if scope not in {"primary_only", "joint"}:
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: family_scope is invalid",
            audit_status=audit_status,
        )
    if unit not in {"edge", "group"}:
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: primary family unit is missing",
            audit_status=audit_status,
        )
    required_counts = (
        (group_count, edge_count)
        if scope == "joint"
        else (edge_count,) if unit == "edge" else (group_count,)
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in required_counts
    ):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: unit-specific frozen family count is missing or invalid",
            audit_status=audit_status,
        )
    expected_family_count = (
        group_count + edge_count
        if scope == "joint"
        else edge_count if unit == "edge" else group_count
    )
    primary_g = primary.get("G")
    if (
        isinstance(primary_g, bool)
        or not isinstance(primary_g, int)
        or primary_g < 1
    ):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: primary G is missing or invalid",
            audit_status=audit_status,
        )
    if primary_g != expected_family_count or n_planned != expected_family_count:
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run.id!r}: frozen family count disagrees with G/n_planned",
            audit_status=audit_status,
        )
    if not _is_sha256_hex(family_hash):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: unit-specific frozen family hash is missing or invalid",
            audit_status=audit_status,
        )

    if method == "bootstrap":
        primary_b = primary.get("bootstrap_B")
        if (
            isinstance(primary_b, bool)
            or not isinstance(primary_b, int)
            or primary_b < 1
        ):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: primary bootstrap_B must be a positive integer",
                audit_status=audit_status,
            )
        if not isinstance(fwer, Mapping):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: bootstrap_fwer is missing",
                audit_status=audit_status,
            )
        fwer_b = fwer.get("B")
        if (
            isinstance(fwer_b, bool)
            or not isinstance(fwer_b, int)
            or fwer_b < 1
        ):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: bootstrap_fwer.B must be a positive integer",
                audit_status=audit_status,
            )
        if fwer_b != primary_b or calibration_b != primary_b:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: primary/FWER bootstrap B values disagree",
                audit_status=audit_status,
            )
        for layer_name, layer in (("result", result), ("provenance", provenance)):
            if "bootstrap_B" not in layer:
                continue
            layer_b = layer["bootstrap_B"]
            if (
                isinstance(layer_b, bool)
                or not isinstance(layer_b, int)
                or layer_b < 1
            ):
                raise _ExportFailure(
                    "SCHEMA_INCOMPLETE",
                    f"run {run.id!r}: {layer_name}.bootstrap_B is invalid",
                    audit_status=audit_status,
                )
            if layer_b != primary_b:
                raise _ExportFailure(
                    "SEMANTIC_MISMATCH",
                    f"run {run.id!r}: {layer_name}.bootstrap_B disagrees with primary",
                    audit_status=audit_status,
                )
        fwer_count = fwer.get("n_hypotheses")
        if (
            isinstance(fwer_count, bool)
            or not isinstance(fwer_count, int)
            or fwer_count < 1
        ):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: bootstrap_fwer.n_hypotheses is missing or invalid",
                audit_status=audit_status,
            )
        if fwer_count != expected_family_count:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: bootstrap FWER family count disagrees",
                audit_status=audit_status,
            )
        if "family_scope" in fwer and fwer["family_scope"] != scope:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: bootstrap FWER family_scope disagrees",
                audit_status=audit_status,
            )
    else:
        permutation_fwer = diagnostics.get("permutation_fwer")
        permutation_is_primary = (
            provenance.get("primary_multiplicity") == "permutation_minp")
        if permutation_is_primary and not isinstance(permutation_fwer, Mapping):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: primary permutation_fwer is missing",
                audit_status=audit_status,
            )
        if isinstance(permutation_fwer, Mapping):
            if (
                "family_scope" in permutation_fwer
                and permutation_fwer["family_scope"] != scope
            ):
                raise _ExportFailure(
                    "SEMANTIC_MISMATCH",
                    f"run {run.id!r}: permutation FWER family_scope disagrees",
                    audit_status=audit_status,
                )
            if permutation_fwer.get("method") != (
                "freedman_lane_permutation_minp_plus_one"
            ):
                raise _ExportFailure(
                    "SEMANTIC_MISMATCH",
                    f"run {run.id!r}: permutation FWER method is invalid",
                    audit_status=audit_status,
                )
            if "n_hypotheses" in permutation_fwer and (
                permutation_fwer["n_hypotheses"] != expected_family_count
            ):
                raise _ExportFailure(
                    "SEMANTIC_MISMATCH",
                    f"run {run.id!r}: permutation FWER family count disagrees",
                    audit_status=audit_status,
                )

    canonical = result.get("mode") == "group" and statistic == "omnib"
    if canonical:
        if provenance.get("mode") != "group":
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: canonical result/provenance modes disagree",
                audit_status=audit_status,
            )
        family = diagnostics.get("family_provenance")
        if not isinstance(family, Mapping):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: canonical family_provenance is missing",
                audit_status=audit_status,
            )
        family_fields = (
            "group_family_sha256", "edge_family_sha256",
            "n_groups_raw", "n_unique_edges",
        )
        if any(field not in family or field not in provenance for field in family_fields):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: canonical family layers lack required fields",
                audit_status=audit_status,
            )
        if any(family[field] != provenance[field] for field in family_fields):
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: canonical family provenance layers disagree",
                audit_status=audit_status,
            )
        if any(
            isinstance(provenance[field], bool)
            or not isinstance(provenance[field], int)
            or provenance[field] < 1
            for field in ("n_groups_raw", "n_unique_edges")
        ):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: canonical family counts must be positive integers",
                audit_status=audit_status,
            )
        for field in ("group_family_sha256", "edge_family_sha256"):
            value = provenance[field]
            if not _is_sha256_hex(value):
                raise _ExportFailure(
                    "SCHEMA_INCOMPLETE",
                    f"run {run.id!r}: canonical {field} is invalid",
                    audit_status=audit_status,
                )
        if not isinstance(fwer, Mapping):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: canonical bootstrap FWER object is missing",
                audit_status=audit_status,
            )
        scope = provenance.get("family_scope")
        if unit not in {"edge", "group"} or scope not in {"primary_only", "joint"}:
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run.id!r}: canonical unit/family_scope is invalid",
                audit_status=audit_status,
            )
        expected_family_id = "joint" if scope == "joint" else unit
        if (
            fwer.get("declared_hypothesis_unit") != unit
            or fwer.get("family_id") != expected_family_id
            or fwer.get("family_scope") != scope
        ):
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: canonical FWER unit/family_scope mismatch",
                audit_status=audit_status,
            )
        expected = (
            group_count + edge_count
            if scope == "joint"
            else edge_count if unit == "edge" else group_count
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (expected, primary.get("G"), primary.get("n_planned"),
                          fwer.get("n_hypotheses"))
        ) or not (
            primary.get("G") == primary.get("n_planned")
            == fwer.get("n_hypotheses") == expected
        ):
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: canonical family/FWER hypothesis counts disagree",
                audit_status=audit_status,
            )
        if (
            method != "bootstrap"
            or fwer.get("method") != "parametric_bootstrap_minp_plus_one"
            or provenance.get("primary_multiplicity") != "bootstrap_minp"
            or provenance.get("primary_transform") != "INT"
        ):
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: canonical omniB bootstrap contract is invalid",
                audit_status=audit_status,
            )

    if audit_record is not None:
        _same_metric(
            run.id, "audit trait", result.get("trait"),
            audit_record.get("trait"), audit_status)
        _same_metric(
            run.id, "audit mode", result.get("mode"),
            audit_record.get("mode"), audit_status)
        _same_metric(
            run.id, "audit statistic", primary_statistic,
            audit_record.get("statistic"), audit_status)
        calibration = audit_record.get("calibration")
        match = re.fullmatch(
            rf"{re.escape(primary_statistic)}\+{re.escape(primary_method)}"
            r"\(B=(\d+)\)",
            calibration,
        )
        if match is None:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: audit calibration string is inconsistent",
                audit_status=audit_status,
            )
        _same_metric(
            run.id, "audit calibration B", calibration_b,
            int(match.group(1)), audit_status)


def _validate_parallel_execution(
    run: RegistryRun,
    provenance: Mapping[str, Any],
    primary: Mapping[str, Any],
    result: Mapping[str, Any],
    audit_status: str,
) -> Mapping[str, Any]:
    parallel = provenance.get("parallel_execution")
    if not isinstance(parallel, Mapping):
        parallel = result.get("parallel_execution")
    if not isinstance(parallel, Mapping):
        return {}
    diagnostics = primary.get("model_diagnostics")
    nested = diagnostics.get("parallel_execution") if isinstance(
        diagnostics, Mapping) else None
    if nested is not None and (not isinstance(nested, Mapping) or dict(nested) != dict(parallel)):
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run.id!r}: parallel execution layers disagree",
            audit_status=audit_status,
        )
    requested = parallel.get("requested_jobs")
    effective = parallel.get("effective_jobs")
    backend = parallel.get("backend")
    pids = parallel.get("worker_pids")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (requested, effective)
    ) or effective > requested:
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: requested/effective jobs are invalid",
            audit_status=audit_status,
        )
    if backend not in {"serial", "fork_shared_memory"}:
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: parallel backend is invalid",
            audit_status=audit_status,
        )
    if (
        not isinstance(pids, list)
        or len(pids) != effective
        or len(set(pids)) != len(pids)
        or any(isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 for pid in pids)
    ):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: worker_pids do not match effective jobs",
            audit_status=audit_status,
        )
    process_model = parallel.get("process_model")
    inner_threads = parallel.get("inner_threads")
    fallback = parallel.get("fallback_reason")
    parent_pid = parallel.get("parent_pid")
    if inner_threads is not None and (
        isinstance(inner_threads, bool)
        or not isinstance(inner_threads, int)
        or inner_threads != 1
    ):
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run.id!r}: inner_threads must equal one",
            audit_status=audit_status,
        )
    if parent_pid is not None and (
        isinstance(parent_pid, bool) or not isinstance(parent_pid, int) or parent_pid < 1
    ):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: parent_pid is invalid",
            audit_status=audit_status,
        )
    if backend == "fork_shared_memory":
        valid = (
            effective > 1
            and (process_model is None or process_model == "processes")
            and fallback is None
            and (parent_pid is None or parent_pid not in pids)
        )
    else:
        valid = (
            effective == 1
            and (process_model is None or process_model == "serial")
            and (parent_pid is None or pids == [parent_pid])
            and (
                (requested == 1 and fallback is None)
                or (requested > 1 and isinstance(fallback, str) and fallback.strip())
            )
        )
    if not valid:
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run.id!r}: backend/process/PID/fallback semantics disagree",
            audit_status=audit_status,
        )
    return parallel


def _limitations(
    run: RegistryRun,
    audit: Mapping[str, Any],
    audit_record: Mapping[str, Any] | None,
) -> list[str]:
    values: list[Any] = []
    declared = run.expected.get("limitations")
    if isinstance(declared, list):
        values.extend(declared)
    if audit_record is not None:
        boundary = audit_record.get("evidence_boundary")
        if isinstance(boundary, list):
            values.extend(boundary)
        flags = audit_record.get("flags")
        if isinstance(flags, list):
            values.extend(
                flag.get("message") for flag in flags if isinstance(flag, Mapping))
        if audit_record.get("replication_status") == "NOT_ASSESSED":
            values.append("Independent external replication has not been assessed.")
    custom = audit.get("limitations")
    if isinstance(custom, list):
        values.extend(custom)
    result = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    return result


def _application_row(run: RegistryRun) -> dict[str, Any]:
    result_path, result, _audit_path, audit, hashes = _read_declared_artifacts(run)
    if result.get("tool") != "homoeogwas" or result.get("command") != "interact":
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: result is not a HomoeoGWAS interact artifact",
        )
    if result.get("subgenomes") != list(run.subgenomes):
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run.id!r}: result subgenomes do not match registry declaration",
        )
    primary = _primary_result(result, run.id)
    provenance = result["provenance"]
    audit_status, audit_record = _audit_status_and_record(
        run, result_path, hashes["result"], audit)

    unit = _primary_unit(run, result, primary)
    sample_count = _integer(
        provenance.get("n_samples", primary.get("n")), "sample_count", run.id,
        minimum=1)
    _same_metric(run.id, "result sample count", sample_count, primary.get("n"), audit_status)
    discoveries, adjusted, drivers = _adjusted_discoveries(
        primary, run.id, audit_status)
    if not drivers:
        custom_drivers = audit.get("component_driver_counts_among_formal_hits")
        if isinstance(custom_drivers, Mapping) and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in custom_drivers.values()
        ):
            drivers = dict(custom_drivers)
    if sum(drivers.values()) != discoveries:
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: component driver distribution does not cover all "
            "adjusted discoveries",
            audit_status=audit_status,
        )

    n_planned = _integer(primary.get("n_planned"), "n_planned", run.id)
    n_valid = _integer(primary.get("n_valid"), "n_valid", run.id)
    diagnostics = primary.get("model_diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    family_layer = diagnostics.get("family_provenance")
    if not isinstance(family_layer, Mapping):
        family_layer = {}
    group_count = provenance.get("n_groups_raw")
    edge_count = provenance.get("n_unique_edges")
    if group_count is None:
        group_count = family_layer.get("n_groups_raw")
    if edge_count is None:
        edge_count = family_layer.get("n_unique_edges")
    if unit == "edge" and edge_count is None:
        edge_count = provenance.get("n_units_raw", result.get("n_units_raw"))
    if unit == "group" and group_count is None:
        group_count = provenance.get("n_units_raw", result.get("n_units_raw"))

    method = primary.get("calibration_method")
    if method == "bootstrap":
        calibration_b = primary.get("bootstrap_B")
    elif method == "permutation":
        permutation = primary.get("permutation")
        if isinstance(permutation, Mapping):
            calibration_b = permutation.get("B_requested")
        else:
            calibration_b = primary.get("perm_B", provenance.get("perm_B"))
    else:
        calibration_b = None
    family_hash = None
    if unit == "edge":
        family_hash = (
            provenance.get("edge_family_sha256")
            or family_layer.get("edge_family_sha256")
        )
    elif unit == "group":
        family_hash = (
            provenance.get("group_family_sha256")
            or family_layer.get("group_family_sha256")
        )
    family_hash = (
        family_hash or provenance.get("family_sha256") or result.get("family_hash")
    )

    _validate_family_and_calibration(
        run, result, primary, audit_record,
        audit_status=audit_status,
        unit=unit,
        n_planned=n_planned,
        n_valid=n_valid,
        group_count=group_count,
        edge_count=edge_count,
        calibration_b=calibration_b,
        family_hash=family_hash,
    )
    parallel = _validate_parallel_execution(
        run, provenance, primary, result, audit_status)

    if audit_record is not None:
        _same_metric(
            run.id, "audit discovery count", discoveries,
            audit_record.get("discovery_count"), audit_status)
        _same_metric(
            run.id, "audit sample count", sample_count,
            audit_record.get("n"), audit_status)
        _same_metric(
            run.id, "audit planned count", n_planned,
            audit_record.get("n_planned"), audit_status)
        _same_metric(
            run.id, "audit valid count", n_valid,
            audit_record.get("n_valid"), audit_status)
        record_status = audit_record["status"]
        replication_status = audit_record["replication_status"]
        status_consistent = (
            discoveries > 0
            and record_status == "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
            and replication_status == "NOT_ASSESSED"
        ) or (
            discoveries > 0
            and record_status == "INTERNAL_DISCOVERY_REPLICATION_RECORDED"
            and replication_status == "RECORDED"
        ) or (
            discoveries == 0
            and record_status in {
                "NO_FAMILYWISE_DISCOVERY",
                "NO_FAMILYWISE_DISCOVERY_REVIEW_REQUIRED",
            }
        )
        if not status_consistent:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: audit record status/replication/discovery values "
                "are inconsistent",
                audit_status=audit_status,
            )
    _same_metric(
        run.id, "registry discovery count", run.expected.get("n_significant"),
        discoveries, audit_status)
    _same_metric(
        run.id, "registry primary unit", run.expected.get("primary_unit"),
        unit, audit_status)
    _same_metric(
        run.id, "registry planned count", run.expected.get("n_planned"),
        n_planned, audit_status)
    _same_metric(
        run.id, "registry valid count", run.expected.get("n_valid"),
        n_valid, audit_status)

    row = _empty_row(run)
    row.update({
        "sample_count": sample_count,
        "marker_count": _marker_count(run, provenance, primary, audit_status),
        "group_family_count": group_count,
        "edge_family_count": edge_count,
        "requested_jobs": parallel.get("requested_jobs"),
        "effective_jobs": parallel.get("effective_jobs"),
        "backend": parallel.get("backend"),
        "worker_pids": parallel.get("worker_pids"),
        "primary_unit": unit,
        "calibration_method": method,
        "calibration_B": calibration_b,
        "adjusted_discovery_count": discoveries,
        "adjusted_discoveries": adjusted,
        "negative_result": discoveries == 0,
        "component_driver_distribution": drivers,
        "audit_status": audit_status,
        "family_hash": family_hash,
        "limitations": _limitations(run, audit, audit_record),
        "status": audit_status,
        "repair_required": False,
        "repair_reason": None,
    })
    required = (
        "sample_count", "marker_count", "requested_jobs", "effective_jobs",
        "backend", "worker_pids",
        "primary_unit", "calibration_method", "calibration_B", "family_hash",
    )
    missing = [name for name in required if row[name] is None]
    if missing:
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: critical application fields are absent: "
            + ", ".join(missing),
            audit_status=audit_status,
            partial=row,
        )
    return row


def export_application_rows(registry_path: str | Path) -> list[dict[str, Any]]:
    """Export one immutable row per registry analysis without dispatching a run.

    Only explicitly inventoried historical result and audit JSON files are read.
    Planned/current runs without a frozen inventory are reported for repair rather
    than scanned, executed, or resolved through filename guessing.
    """
    registry = load_registry(registry_path)
    rows = []
    for run in registry.runs:
        if run.kind != "historical":
            rows.append(_failure_row(run, _ExportFailure(
                "UNDECLARED_AUTHORITATIVE_INVENTORY",
                f"run {run.id!r}: Track D accepts frozen historical inventory only",
            )))
            continue
        try:
            rows.append(_application_row(run))
        except _ExportFailure as failure:
            rows.append(_failure_row(run, failure))
    return rows
