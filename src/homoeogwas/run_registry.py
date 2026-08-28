"""Species-independent production registry for HomoeoGWAS workflows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RegistryError(ValueError):
    """A registry cannot be interpreted without changing its declared analysis."""


@dataclass(frozen=True)
class RegistryRun:
    id: str
    kind: str
    species: str
    panel: str
    subgenomes: tuple[str, ...]
    out_dir: Path | None
    phenotype: Path | None = None
    sample_col: str | None = None
    trait: str | None = None
    bed_prefixes: Mapping[str, Path] = field(default_factory=dict)
    snp_to_gene: Mapping[str, Path] = field(default_factory=dict)
    groups: Path | None = None
    hypothesis_unit: str = "group"
    family_scope: str = "primary_only"
    bootstrap_B: int = 2000
    n_jobs: int = 8
    include_hadamard: bool = False
    loco: bool = False
    run_plots: bool = True
    result_root: Path | None = None
    analysis_shape: str | None = None
    artifact_inventory: tuple[Mapping[str, Any], ...] = ()
    expected: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRegistry:
    registry_version: int
    name: str
    index_dir: Path
    source_path: Path
    runs: tuple[RegistryRun, ...]


_ROOT_FIELDS = {"registry_version", "name", "index_dir", "runs"}
_RUN_FIELDS = {
    "id", "kind", "species", "panel", "subgenomes", "out_dir",
    "phenotype", "sample_col", "trait", "bed_prefixes", "snp_to_gene",
    "groups", "hypothesis_unit", "family_scope", "bootstrap_B", "n_jobs",
    "include_hadamard", "loco", "run_plots", "result_root",
    "analysis_shape", "artifact_inventory", "expected",
}


def _resolve(base: Path, value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise RegistryError("registry paths must be non-empty strings")
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _mapping_paths(base: Path, value: Any, label: str) -> dict[str, Path]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RegistryError(f"{label} must be a subgenome-to-path mapping")
    return {str(key): _resolve(base, path) for key, path in value.items()}


def _require_text(raw: Mapping[str, Any], key: str, run_id: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"run {run_id!r}: {key} must be a non-empty string")
    return value.strip()


def _optional_text(
    raw: Mapping[str, Any], key: str, run_id: str, default: str | None = None,
) -> str | None:
    if key not in raw:
        return default
    return _require_text(raw, key, run_id)


def _boolean(raw: Mapping[str, Any], key: str, run_id: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise RegistryError(f"run {run_id!r}: {key} must be true or false")
    return value


def _parse_artifact_inventory(value: Any, run_id: str) -> tuple[dict, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise RegistryError(
            f"run {run_id!r}: artifact_inventory must be a non-empty list")
    parsed = []
    seen = set()
    for row, record in enumerate(value):
        if not isinstance(record, Mapping) or set(record) != {
            "role", "path", "size", "sha256"
        }:
            raise RegistryError(
                f"run {run_id!r}: artifact_inventory[{row}] is malformed")
        role = record["role"]
        relative_text = record["path"]
        size = record["size"]
        digest = record["sha256"]
        if role not in {"result", "audit"}:
            raise RegistryError(
                f"run {run_id!r}: artifact role must be result or audit")
        if not isinstance(relative_text, str) or not relative_text.strip():
            raise RegistryError(
                f"run {run_id!r}: artifact path must be a non-empty string")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise RegistryError(
                f"run {run_id!r}: artifact path must stay below result_root")
        normalized = relative.as_posix()
        if normalized in seen:
            raise RegistryError(
                f"run {run_id!r}: duplicate artifact path {normalized!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RegistryError(
                f"run {run_id!r}: artifact size must be a non-negative integer")
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise RegistryError(
                f"run {run_id!r}: artifact sha256 must be 64 lowercase hex digits")
        seen.add(normalized)
        parsed.append({
            "role": role, "path": normalized, "size": size, "sha256": digest,
        })
    return tuple(parsed)


def _parse_run(raw: Any, base: Path, row: int) -> RegistryRun:
    if not isinstance(raw, Mapping):
        raise RegistryError(f"runs[{row}] must be a mapping")
    unknown = sorted(set(raw) - _RUN_FIELDS)
    if unknown:
        raise RegistryError(
            f"run {raw.get('id', row)!r}: unsupported fields: {', '.join(unknown)}")
    run_id = _require_text(raw, "id", f"row-{row}")
    kind = _require_text(raw, "kind", run_id).lower()
    if kind not in {"gwas", "interaction", "historical"}:
        raise RegistryError(
            f"run {run_id!r}: kind must be gwas, interaction, or historical")
    subgenomes_raw = raw.get("subgenomes")
    if not isinstance(subgenomes_raw, list) or not all(
        isinstance(value, str) and value.strip() for value in subgenomes_raw
    ):
        raise RegistryError(f"run {run_id!r}: subgenomes must be a list of labels")
    subgenomes = tuple(value.strip() for value in subgenomes_raw)
    if kind == "interaction" and (
        len(subgenomes) < 2 or len(set(subgenomes)) != len(subgenomes)
    ):
        raise RegistryError(
            f"run {run_id!r}: interaction needs at least two unique subgenomes")
    if kind != "interaction" and len(set(subgenomes)) != len(subgenomes):
        raise RegistryError(f"run {run_id!r}: subgenomes must be unique")

    expected_raw = raw.get("expected", {})
    if expected_raw is None:
        expected_raw = {}
    if not isinstance(expected_raw, Mapping):
        raise RegistryError(f"run {run_id!r}: expected must be a mapping")
    run = RegistryRun(
        id=run_id,
        kind=kind,
        species=_require_text(raw, "species", run_id),
        panel=_require_text(raw, "panel", run_id),
        subgenomes=subgenomes,
        out_dir=_resolve(base, raw.get("out_dir")),
        phenotype=_resolve(base, raw.get("phenotype")),
        sample_col=_optional_text(raw, "sample_col", run_id),
        trait=_optional_text(raw, "trait", run_id),
        bed_prefixes=_mapping_paths(
            base, raw.get("bed_prefixes"), f"run {run_id!r} bed_prefixes"),
        snp_to_gene=_mapping_paths(
            base, raw.get("snp_to_gene"), f"run {run_id!r} snp_to_gene"),
        groups=_resolve(base, raw.get("groups")),
        hypothesis_unit=(
            _optional_text(raw, "hypothesis_unit", run_id, "group") or "group"
        ).lower(),
        family_scope=(
            _optional_text(raw, "family_scope", run_id, "primary_only")
            or "primary_only"
        ).lower(),
        bootstrap_B=raw.get("bootstrap_B", 2000),
        n_jobs=raw.get("n_jobs", 8),
        include_hadamard=_boolean(raw, "include_hadamard", run_id, False),
        loco=_boolean(raw, "loco", run_id, False),
        run_plots=_boolean(raw, "run_plots", run_id, True),
        result_root=_resolve(base, raw.get("result_root")),
        analysis_shape=_optional_text(raw, "analysis_shape", run_id),
        artifact_inventory=_parse_artifact_inventory(
            raw.get("artifact_inventory"), run_id),
        expected=dict(expected_raw),
    )
    _validate_run(run)
    return run


def _validate_run(run: RegistryRun) -> None:
    if run.kind in {"gwas", "interaction"}:
        missing = [
            name for name, value in (
                ("out_dir", run.out_dir), ("phenotype", run.phenotype),
                ("sample_col", run.sample_col), ("trait", run.trait),
            ) if not value
        ]
        if missing:
            raise RegistryError(
                f"run {run.id!r}: missing required fields: {', '.join(missing)}")
        absent_beds = sorted(set(run.subgenomes) - set(run.bed_prefixes))
        if absent_beds:
            raise RegistryError(
                f"run {run.id!r}: bed_prefixes missing subgenomes {absent_beds}")
    if run.kind == "interaction":
        absent_maps = sorted(set(run.subgenomes) - set(run.snp_to_gene))
        if absent_maps:
            raise RegistryError(
                f"run {run.id!r}: snp_to_gene missing subgenomes {absent_maps}")
        if run.groups is None:
            raise RegistryError(f"run {run.id!r}: interaction requires groups")
        if run.hypothesis_unit not in {"edge", "group"}:
            raise RegistryError(
                f"run {run.id!r}: hypothesis_unit must be edge or group")
        if run.family_scope not in {"primary_only", "joint"}:
            raise RegistryError(
                f"run {run.id!r}: family_scope must be primary_only or joint")
        if isinstance(run.bootstrap_B, bool) or not isinstance(run.bootstrap_B, int) \
                or run.bootstrap_B < 19:
            raise RegistryError(f"run {run.id!r}: bootstrap_B must be an integer >= 19")
    if isinstance(run.n_jobs, bool) or not isinstance(run.n_jobs, int) or run.n_jobs < 1:
        raise RegistryError(f"run {run.id!r}: n_jobs must be an integer >= 1")
    if run.kind == "historical" and (
        run.result_root is None or not isinstance(run.analysis_shape, str)
        or not run.analysis_shape.strip() or not run.artifact_inventory
    ):
        raise RegistryError(
            f"run {run.id!r}: historical entries require result_root, analysis_shape, "
            "and artifact_inventory")
    if run.kind != "historical" and run.artifact_inventory:
        raise RegistryError(
            f"run {run.id!r}: artifact_inventory is only valid for historical entries")


def validate_registry(registry: RunRegistry) -> None:
    if (
        isinstance(registry.registry_version, bool)
        or not isinstance(registry.registry_version, int)
        or registry.registry_version != 1
    ):
        raise RegistryError(
            f"registry_version must be 1, got {registry.registry_version!r}")
    if not registry.runs:
        raise RegistryError("registry runs must not be empty")
    ids = [run.id for run in registry.runs]
    duplicates = sorted({run_id for run_id in ids if ids.count(run_id) > 1})
    if duplicates:
        raise RegistryError(f"duplicate run id: {', '.join(duplicates)}")


def load_registry(path: str | Path) -> RunRegistry:
    """Load and validate a versioned registry, resolving paths from its directory."""
    import yaml

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise RegistryError(f"registry not found: {source}")
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, Mapping):
        raise RegistryError("registry YAML must be a mapping")
    unknown = sorted(set(raw) - _ROOT_FIELDS)
    if unknown:
        raise RegistryError(f"registry has unsupported fields: {', '.join(unknown)}")
    base = source.parent
    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, list):
        raise RegistryError("registry runs must be a list")
    version = raw.get("registry_version")
    name = raw.get("name")
    if isinstance(version, bool) or not isinstance(version, int):
        raise RegistryError("registry_version must be the integer 1")
    if not isinstance(name, str) or not name.strip():
        raise RegistryError("registry name must be a non-empty string")
    registry = RunRegistry(
        registry_version=version,
        name=name.strip(),
        index_dir=_resolve(base, raw.get("index_dir") or "registry-index"),
        source_path=source,
        runs=tuple(_parse_run(item, base, row) for row, item in enumerate(runs_raw)),
    )
    validate_registry(registry)
    return registry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path | None, *, required: bool = True) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        if required:
            raise RegistryError(f"identity input is missing: {path}")
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _bed_identity(prefix: Path) -> dict:
    bed = Path(str(prefix) + ".bed")
    bim = Path(str(prefix) + ".bim")
    fam = Path(str(prefix) + ".fam")
    missing = [str(path) for path in (bed, bim, fam) if not path.exists()]
    if missing:
        raise RegistryError(
            "identity PLINK inputs are missing: " + ", ".join(missing))
    return {
        "prefix": str(prefix),
        "bed": {"path": str(bed), "size": int(bed.stat().st_size)},
        "bim": _file_identity(bim),
        "fam": _file_identity(fam),
    }


def canonical_run_identity(run: RegistryRun) -> dict:
    """Return the canonical, input-bound identity payload for one run."""
    from . import __version__

    payload: dict[str, Any] = {
        "identity_schema": "homoeogwas-registry-run-identity-v1",
        "homoeogwas_version": __version__,
        "id": run.id,
        "kind": run.kind,
        "species": run.species,
        "panel": run.panel,
        "subgenomes": list(run.subgenomes),
        "trait": run.trait,
        "sample_col": run.sample_col,
        "out_dir": str(run.out_dir) if run.out_dir else None,
    }
    if run.kind == "historical":
        artifact_audit = audit_historical_artifacts(run)
        payload.update({
            "result_root": str(run.result_root),
            "analysis_shape": run.analysis_shape,
            "expected": dict(run.expected),
            "artifact_inventory": artifact_audit["artifacts"],
        })
        return payload
    payload.update({
        "phenotype": _file_identity(run.phenotype),
        "bed_prefixes": {
            sub: _bed_identity(run.bed_prefixes[sub])
            for sub in sorted(run.subgenomes)
        },
    })
    if run.kind == "gwas":
        payload["gwas"] = {
            "include_hadamard": run.include_hadamard,
            "loco": run.loco,
            "run_plots": run.run_plots,
        }
    else:
        payload["interaction"] = {
            "mode": "group",
            "statistic": "omniB",
            "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
            "subset_order": 2,
            "family_scope": run.family_scope,
            "hypothesis_unit": run.hypothesis_unit,
            "bootstrap_B": run.bootstrap_B,
            "bootstrap_seed": 2026,
            "groups": _file_identity(run.groups),
            "snp_to_gene": {
                sub: _file_identity(run.snp_to_gene[sub])
                for sub in sorted(run.subgenomes)
            },
        }
    return payload


def run_identity_sha256(run: RegistryRun) -> str:
    payload = json.dumps(
        canonical_run_identity(run), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def audit_historical_artifacts(run: RegistryRun) -> dict:
    """Verify the declared immutable result/audit inventory of a legacy run."""
    root = run.result_root
    if root is None or not root.exists() or not root.is_dir():
        raise RegistryError(
            f"run {run.id!r}: historical result root is missing: {root}")
    declared = {record["path"]: dict(record) for record in run.artifact_inventory}
    observed = {
        path.relative_to(root).as_posix()
        for path in [
            *root.glob("interact_*.json"), *((root / "audit").glob("*.json"))
        ]
        if path.is_file()
    }
    if set(declared) != observed:
        raise RegistryError(
            f"run {run.id!r}: declared historical artifact inventory does not "
            "match result/audit files")
    artifacts = []
    result_payloads = []
    result_paths = []
    audit_payloads = []
    for relative, record in sorted(declared.items()):
        path = root / relative
        if not path.exists() or not path.is_file():
            raise RegistryError(
                f"run {run.id!r}: declared historical artifact is missing: {path}")
        observed_size = int(path.stat().st_size)
        observed_digest = _sha256_file(path)
        if (
            observed_size != record["size"]
            or observed_digest != record["sha256"]
        ):
            raise RegistryError(
                f"run {run.id!r}: historical artifact identity mismatch: {relative}")
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(
                f"run {run.id!r}: unreadable historical artifact {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise RegistryError(
                f"run {run.id!r}: historical artifact is not a JSON object: {path}")
        resolved_record = {**record, "path": str(path.resolve())}
        artifacts.append(resolved_record)
        if record["role"] == "result":
            result_payloads.append(value)
            result_paths.append(path.resolve())
        else:
            audit_payloads.append(value)
    if not result_payloads or not audit_payloads:
        raise RegistryError(
            f"run {run.id!r}: artifact inventory needs result and audit roles")

    result_path_set = set(result_paths)
    audited_sources = set()
    custom_bound_results = set()
    allowed_overall = {
        "AUDIT_COMPLETE", "INTERNAL_DISCOVERY_REPLICATION_REQUIRED",
        "REVIEW_REQUIRED",
    }
    for audit in audit_payloads:
        if "overall_status" in audit:
            status = str(audit.get("overall_status", "")).strip().upper()
            if status not in allowed_overall:
                raise RegistryError(
                    f"run {run.id!r}: historical audit status is invalid: {status}")
            records = audit.get("records")
            if not isinstance(records, list) or audit.get("n_results") != len(records):
                raise RegistryError(
                    f"run {run.id!r}: historical audit records are malformed")
            for audit_record in records:
                if not isinstance(audit_record, Mapping):
                    raise RegistryError(
                        f"run {run.id!r}: historical audit record is malformed")
                try:
                    audited_sources.add(Path(str(audit_record["source"])).resolve())
                except (KeyError, TypeError, ValueError) as exc:
                    raise RegistryError(
                        f"run {run.id!r}: historical audit source is invalid") from exc
        elif str(audit.get("status", "")).strip().upper() == "PASS":
            hashes = audit.get("output_sha256")
            if not isinstance(hashes, Mapping):
                raise RegistryError(
                    f"run {run.id!r}: custom historical audit lacks output hashes")
            declared_result_hashes = {
                record["sha256"] for record in declared.values()
                if record["role"] == "result"
            }
            bound = set(hashes.values()) & declared_result_hashes
            if not bound:
                raise RegistryError(
                    f"run {run.id!r}: custom historical audit is not bound to its result")
            custom_bound_results.update(bound)
        else:
            raise RegistryError(
                f"run {run.id!r}: historical audit status is unknown or invalid")
    if audited_sources and audited_sources != result_path_set:
        raise RegistryError(
            f"run {run.id!r}: historical audit sources do not bind the result inventory")
    if not audited_sources and not custom_bound_results:
        raise RegistryError(
            f"run {run.id!r}: historical audits do not bind any declared result")
    primary_results = []
    for payload in result_payloads:
        results = payload.get("results") or {}
        preferred = str((payload.get("provenance") or {}).get(
            "primary_transform", "INT"))
        primary = results.get(preferred) or results.get("INT")
        if isinstance(primary, Mapping):
            primary_results.append(primary)
    expected_fields = {
        "n_significant": "n_sig",
        "n_planned": "n_planned",
        "n_valid": "n_valid",
    }
    for expected_key, result_key in expected_fields.items():
        if expected_key not in run.expected:
            continue
        observed = {
            value for primary in primary_results
            if (value := primary.get(result_key)) is not None
        }
        if observed != {run.expected[expected_key]}:
            raise RegistryError(
                f"run {run.id!r}: historical {expected_key} expected "
                f"{run.expected[expected_key]!r}, observed {sorted(observed)!r}")
    return {"status": "PASS", "artifacts": artifacts}


def load_run_state(out_dir: Path) -> dict | None:
    path = Path(out_dir) / "registry_run.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read registry state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"registry state {path} must contain a JSON object")
    return value


def write_run_state(out_dir: Path, payload: Mapping[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "registry_run.json"
    body = json.dumps(
        dict(payload), indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        dir=out_dir, prefix=".registry_run.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(body)
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def resume_decision(
    run: RegistryRun,
    existing_state: Mapping[str, Any] | None,
    *,
    resume: bool,
) -> str:
    """Return RUN or SKIP, failing closed on an occupied output identity."""
    if existing_state is None:
        return "RUN"
    if not resume:
        raise RegistryError(
            f"run {run.id!r}: registry state exists and resume is disabled")
    observed = existing_state.get("identity")
    expected = run_identity_sha256(run)
    if observed != expected:
        raise RegistryError(
            f"run {run.id!r}: output identity mismatch; use a new out_dir "
            "instead of overwriting an existing analysis")
    return "SKIP" if existing_state.get("status") == "COMPLETE" else "RUN"


def _atomic_text(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(body)
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def _atomic_tsv(path: Path, table) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            table.to_csv(handle, sep="\t", index=False, na_rep="NA")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def _summary_fields(summary: Mapping[str, Any] | None) -> dict:
    summary = dict(summary or {})
    return {
        "n_significant": summary.get("n_significant"),
        "audit_status": summary.get("audit_status"),
        "primary_unit": summary.get("primary_unit"),
        "n_planned": summary.get("n_planned"),
        "n_valid": summary.get("n_valid"),
    }


def _record_for_run(
    run: RegistryRun,
    *,
    status: str,
    identity: str | None,
    summary: Mapping[str, Any] | None = None,
    reason: str | None = None,
) -> dict:
    output = run.result_root if run.kind == "historical" else run.out_dir
    return {
        "run_id": run.id,
        "kind": run.kind,
        "species": run.species,
        "panel": run.panel,
        "trait": run.trait,
        "subgenomes": list(run.subgenomes),
        "identity": identity,
        "status": status,
        "output_root": str(output) if output else None,
        "analysis_shape": run.analysis_shape,
        **_summary_fields(summary),
        "reason": reason,
        "next_action": (
            "migrate and rerun canonically" if status == "HISTORICAL"
            else ("inspect failure and repair inputs" if status.startswith("FAILED")
                  else ("follow-up formal discoveries" if
                        (_summary_fields(summary)["n_significant"] or 0) > 0
                        else "none"))
        ),
    }


def write_registry_indexes(
    registry: RunRegistry,
    records: list[Mapping[str, Any]],
) -> dict[str, str]:
    """Write deterministic JSON, TSV and breeder-readable Markdown indexes."""
    import pandas as pd
    import yaml

    out = registry.index_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry_version": registry.registry_version,
        "name": registry.name,
        "source": str(registry.source_path),
        "runs": [dict(record) for record in records],
    }
    json_path = _atomic_text(
        out / "run_index.json",
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    columns = [
        "run_id", "kind", "species", "panel", "trait", "subgenomes",
        "status", "identity", "output_root", "analysis_shape",
        "primary_unit", "n_planned", "n_valid", "n_significant",
        "audit_status", "reason", "next_action",
    ]
    table = pd.DataFrame.from_records(records)
    for column in columns:
        if column not in table:
            table[column] = None
    table = table[columns].copy()
    table["subgenomes"] = table["subgenomes"].map(
        lambda value: ",".join(value) if isinstance(value, list) else value)
    tsv_path = _atomic_tsv(out / "run_index.tsv", table)
    header = (
        "| Run | Species | Trait | Copies | Status | Significant | Audit | Next action |\n"
        "|---|---|---|---|---|---:|---|---|\n")
    rows = []
    for record in records:
        significant = record.get("n_significant")
        rows.append(
            f"| {record['run_id']} | {record['species']} | "
            f"{record.get('trait') or 'NA'} | {','.join(record['subgenomes'])} | "
            f"{record['status']} | {significant if significant is not None else 'NA'} | "
            f"{record.get('audit_status') or 'NA'} | {record['next_action']} |")
    markdown_path = _atomic_text(
        out / "run_index.md",
        f"# {registry.name}\n\n" + header + "\n".join(rows) + "\n")
    resolved = {
        "registry_version": registry.registry_version,
        "name": registry.name,
        "index_dir": str(registry.index_dir),
        "runs": [
            {
                "id": run.id,
                "kind": run.kind,
                "species": run.species,
                "panel": run.panel,
                "subgenomes": list(run.subgenomes),
                "out_dir": str(run.out_dir) if run.out_dir else None,
                "result_root": str(run.result_root) if run.result_root else None,
            }
            for run in registry.runs
        ],
    }
    resolved_path = _atomic_text(
        out / "registry.resolved.yaml",
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    return {
        "json": str(json_path),
        "tsv": str(tsv_path),
        "markdown": str(markdown_path),
        "resolved": str(resolved_path),
    }


def execute_registry(
    path: str | Path,
    *,
    only=(),
    resume: bool = True,
    fail_fast: bool = False,
    dry_run: bool = False,
    gwas_runner=None,
    interaction_runner=None,
) -> dict:
    """Execute independent registered workflows and write a cross-run index."""
    from . import workflow

    registry = load_registry(path)
    gwas_runner = gwas_runner or workflow.run_gwas
    interaction_runner = interaction_runner or workflow.run_interaction
    selected = {str(value) for value in only}
    unknown = sorted(selected - {run.id for run in registry.runs})
    if unknown:
        raise RegistryError(f"--only contains unknown run IDs: {', '.join(unknown)}")
    records: list[dict] = []
    for run in registry.runs:
        if selected and run.id not in selected:
            continue
        if run.kind == "historical":
            try:
                artifact_audit = audit_historical_artifacts(run)
                identity = run_identity_sha256(run)
            except Exception as exc:  # noqa: BLE001 - isolate registry entries
                records.append(_record_for_run(
                    run, status="FAILED_HISTORICAL_AUDIT", identity=None,
                    summary=run.expected,
                    reason=f"{type(exc).__name__}: {exc}"))
                if fail_fast:
                    break
                continue
            records.append(_record_for_run(
                run, status="HISTORICAL", identity=identity,
                summary=run.expected,
                reason=f"artifact audit {artifact_audit['status']}"))
            continue
        try:
            identity = run_identity_sha256(run)
        except Exception as exc:  # noqa: BLE001 - filesystem identity is fallible
            records.append(_record_for_run(
                run, status="FAILED_IDENTITY_INPUT", identity=None,
                reason=f"{type(exc).__name__}: {exc}"))
            if fail_fast:
                break
            continue
        try:
            state = load_run_state(run.out_dir)
            decision = resume_decision(run, state, resume=resume)
        except RegistryError as exc:
            status = (
                "BLOCKED_IDENTITY_MISMATCH"
                if "identity mismatch" in str(exc) or "resume is disabled" in str(exc)
                else "FAILED"
            )
            record = _record_for_run(
                run, status=status, identity=identity,
                reason=str(exc))
            records.append(record)
            if fail_fast:
                break
            continue
        if decision == "SKIP":
            records.append(_record_for_run(
                run, status="SKIPPED_COMPLETE", identity=identity,
                summary=state.get("summary")))
            continue
        try:
            write_run_state(run.out_dir, {
                "run_id": run.id, "status": "PLANNED", "identity": identity})
            write_run_state(run.out_dir, {
                "run_id": run.id, "status": "RUNNING", "identity": identity})
            if run.kind == "gwas":
                result = gwas_runner(
                    phenotype=str(run.phenotype), sample_col=run.sample_col,
                    trait=run.trait, subgenomes=run.subgenomes,
                    out_dir=str(run.out_dir),
                    bed_prefixes={
                        key: str(value) for key, value in run.bed_prefixes.items()},
                    include_hadamard=run.include_hadamard, loco=run.loco,
                    run_plots=run.run_plots, dry_run=dry_run)
            else:
                result = interaction_runner(
                    phenotype=str(run.phenotype), sample_col=run.sample_col,
                    trait=run.trait, subgenomes=run.subgenomes,
                    bed_prefixes={
                        key: str(value) for key, value in run.bed_prefixes.items()},
                    snp_to_gene={
                        key: str(value) for key, value in run.snp_to_gene.items()},
                    out_dir=str(run.out_dir), groups=str(run.groups),
                    hypothesis_unit=run.hypothesis_unit,
                    subset_order=2, family_scope=run.family_scope,
                    perm_b=run.bootstrap_B, n_jobs=run.n_jobs,
                    statistic="omniB", dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 - isolate independent registry runs
            reason = f"{type(exc).__name__}: {exc}"
            try:
                write_run_state(run.out_dir, {
                    "run_id": run.id, "status": "FAILED", "identity": identity,
                    "summary": {}, "reason": reason,
                })
            except Exception as state_exc:  # noqa: BLE001 - retain index failure
                reason += (
                    f"; failed to record run state: {type(state_exc).__name__}: "
                    f"{state_exc}")
            records.append(_record_for_run(
                run, status="FAILED", identity=identity, reason=reason))
            if fail_fast:
                break
            continue
        summary = result.get("summary") or {}
        if result.get("ok"):
            status = "DRY_RUN" if dry_run else "COMPLETE"
            reason = None
        else:
            status = "FAILED"
            reason = result.get("reason") or "workflow returned ok=false"
        try:
            write_run_state(run.out_dir, {
                "run_id": run.id, "status": status, "identity": identity,
                "summary": summary, "reason": reason,
            })
        except Exception as exc:  # noqa: BLE001 - isolate registry entries
            reason = f"failed to record final run state: {type(exc).__name__}: {exc}"
            records.append(_record_for_run(
                run, status="FAILED", identity=identity,
                summary=summary, reason=reason))
            if fail_fast:
                break
            continue
        records.append(_record_for_run(
            run, status=status, identity=identity, summary=summary, reason=reason))
        if status == "FAILED" and fail_fast:
            break
    indexes = write_registry_indexes(registry, records)
    failed_statuses = {
        "FAILED", "FAILED_HISTORICAL_AUDIT", "FAILED_IDENTITY_INPUT",
        "BLOCKED_IDENTITY_MISMATCH"}
    return {
        "ok": not any(record["status"] in failed_statuses for record in records),
        "registry": str(registry.source_path),
        "runs": records,
        "indexes": indexes,
        "dry_run": dry_run,
    }


def add_registry_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "registry",
        help="validate or execute a cross-species HomoeoGWAS run registry")
    actions = parser.add_subparsers(dest="registry_action", required=True)
    validate = actions.add_parser(
        "validate", help="validate registry structure without running analyses")
    validate.add_argument("-c", "--config", required=True, help="registry YAML path")
    run = actions.add_parser(
        "run", help="execute registered workflows and write a status index")
    run.add_argument("-c", "--config", required=True, help="registry YAML path")
    run.add_argument(
        "--only", nargs="*", default=(), metavar="RUN_ID",
        help="execute only the listed run IDs")
    resume_group = run.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="resume or skip matching registered output roots (default)")
    resume_group.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="reject any output root with existing registry state")
    run.add_argument(
        "--fail-fast", action="store_true",
        help="stop after the first failed or identity-blocked run")
    run.add_argument(
        "--dry-run", action="store_true",
        help="generate configs and command plans without long computation")


def cmd_registry(args) -> int:
    try:
        if args.registry_action == "validate":
            registry = load_registry(args.config)
            print(
                f"[registry] registry schema OK: {registry.name} "
                f"({len(registry.runs)} runs)")
            return 0
        result = execute_registry(
            args.config, only=args.only, resume=args.resume,
            fail_fast=args.fail_fast, dry_run=args.dry_run)
    except RegistryError as exc:
        print(f"ERROR: registry: {exc}")
        return 1
    for record in result["runs"]:
        print(
            f"[registry] {record['run_id']}: {record['status']} "
            f"significant={record.get('n_significant')}")
    print(f"[registry] index: {result['indexes']['json']}")
    return 0 if result["ok"] else 1
