"""Read-only export of frozen cross-species HomoeoGWAS application evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from homoeogwas.run_registry import RegistryRun, load_registry

_VALID_AUDIT_STATUSES = {
    "PASS",
    "AUDIT_COMPLETE",
    "INTERNAL_DISCOVERY_REPLICATION_REQUIRED",
    "REVIEW_REQUIRED",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_json_object(path: Path, run_id: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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
        if not path.exists() or not path.is_file():
            raise _ExportFailure(
                "MISSING_AUTHORITATIVE_OUTPUT",
                f"run {run.id!r}: declared {role} artifact is missing: {path}",
            )
        observed_size = int(path.stat().st_size)
        observed_hash = _sha256_file(path)
        if observed_size != record["size"] or observed_hash != record["sha256"]:
            raise _ExportFailure(
                "AUTHORITATIVE_ARTIFACT_MISMATCH",
                f"run {run.id!r}: declared {role} artifact identity mismatch at "
                f"{path} (size={observed_size}, sha256={observed_hash})",
            )
        resolved[role] = (path, _read_json_object(path, run.id), observed_hash)
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
    provenance: Mapping[str, Any], primary: Mapping[str, Any],
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
    primary: Mapping[str, Any], run_id: str,
) -> tuple[int, list[dict[str, Any]], dict[str, int]]:
    count = _integer(primary.get("n_sig"), "n_sig", run_id)
    significant = primary.get("sig")
    if not isinstance(significant, list) or len(significant) != count:
        raise _ExportFailure(
            "SEMANTIC_MISMATCH",
            f"run {run_id!r}: discovery count does not match significant-unit list",
        )
    exported = []
    drivers: Counter[str] = Counter()
    for index, unit in enumerate(significant):
        if not isinstance(unit, Mapping):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run_id!r}: significant unit {index} is malformed",
            )
        identifier = unit.get("hypothesis_id") or unit.get("edge_id") or unit.get("pair")
        adjusted = unit.get("p_adjusted_bootstrap_minp")
        if identifier is None or not isinstance(adjusted, (int, float)) \
                or isinstance(adjusted, bool) or not math.isfinite(adjusted):
            raise _ExportFailure(
                "SCHEMA_INCOMPLETE",
                f"run {run_id!r}: significant unit {index} lacks finite adjusted p-value "
                "or identifier",
            )
        exported.append({"hypothesis_id": identifier, "adjusted_p": float(adjusted)})
        driver = unit.get("smallest_component")
        if isinstance(driver, str) and driver:
            drivers[driver] += 1
    return count, exported, dict(drivers)


def _audit_status_and_record(
    run: RegistryRun,
    result_path: Path,
    result_hash: str,
    audit: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any] | None]:
    if "overall_status" in audit:
        status = str(audit.get("overall_status", "")).strip().upper()
        if status not in _VALID_AUDIT_STATUSES:
            raise _ExportFailure(
                "AUDIT_FAILED",
                f"run {run.id!r}: frozen audit failed with status {status or 'MISSING'}",
                audit_status=status or None,
            )
        records = audit.get("records")
        if not isinstance(records, list) or audit.get("n_results") != len(records):
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: audit records/count are malformed",
                audit_status=status,
            )
        matches = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            try:
                source = Path(str(record.get("source"))).resolve()
            except (TypeError, ValueError):
                continue
            if source == result_path.resolve():
                matches.append(record)
        if len(matches) != 1:
            raise _ExportFailure(
                "SEMANTIC_MISMATCH",
                f"run {run.id!r}: audit does not bind exactly one declared result",
                audit_status=status,
            )
        return status, matches[0]

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
    if family_hash is not None and (
        not isinstance(family_hash, str)
        or len(family_hash) != 64
        or any(character not in "0123456789abcdef" for character in family_hash)
    ):
        raise _ExportFailure(
            "SCHEMA_INCOMPLETE",
            f"run {run.id!r}: family_hash is not 64 lowercase hexadecimal digits",
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
    fwer = diagnostics.get("bootstrap_fwer") if isinstance(diagnostics, Mapping) else None
    if isinstance(fwer, Mapping):
        declared = fwer.get("declared_hypothesis_unit") or fwer.get("family_id")
        _same_metric(run.id, "bootstrap family unit", unit, declared, audit_status)
        _same_metric(
            run.id, "bootstrap family count", n_planned,
            fwer.get("n_hypotheses"), audit_status)
        _same_metric(
            run.id, "bootstrap calibration B", calibration_b,
            fwer.get("B"), audit_status)

    if audit_record is not None:
        _same_metric(
            run.id, "audit trait", result.get("trait"),
            audit_record.get("trait"), audit_status)
        _same_metric(
            run.id, "audit mode", result.get("mode"),
            audit_record.get("mode"), audit_status)
        calibration = audit_record.get("calibration")
        if isinstance(calibration, str):
            match = re.search(r"\bB=(\d+)\b", calibration)
            if match is not None:
                _same_metric(
                    run.id, "audit calibration B", calibration_b,
                    int(match.group(1)), audit_status)


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
    discoveries, adjusted, drivers = _adjusted_discoveries(primary, run.id)
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
    group_count = provenance.get("n_groups_raw")
    edge_count = provenance.get("n_unique_edges")
    if unit == "edge" and edge_count is None:
        edge_count = provenance.get("n_units_raw", primary.get("G"))
    if group_count is None and unit == "edge" and len(run.subgenomes) == 2:
        group_count = primary.get("G")
    if group_count is None and unit == "group":
        group_count = primary.get("G")
    if edge_count is None and isinstance(group_count, int):
        edge_count = group_count * math.comb(len(run.subgenomes), 2)

    parallel = provenance.get("parallel_execution")
    if not isinstance(parallel, Mapping):
        parallel = result.get("parallel_execution")
    if not isinstance(parallel, Mapping):
        parallel = {}
    method = primary.get("calibration_method") or provenance.get("calibration_method")
    calibration_b = primary.get("bootstrap_B")
    if calibration_b is None:
        fwer = (primary.get("model_diagnostics") or {}).get("bootstrap_fwer")
        if isinstance(fwer, Mapping):
            calibration_b = fwer.get("B")
    family_hash = None
    if unit == "edge":
        family_hash = provenance.get("edge_family_sha256")
    elif unit == "group":
        family_hash = provenance.get("group_family_sha256")
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
        "marker_count": _marker_count(provenance, primary),
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
        "sample_count", "marker_count", "group_family_count", "edge_family_count",
        "requested_jobs", "effective_jobs", "backend", "worker_pids",
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
