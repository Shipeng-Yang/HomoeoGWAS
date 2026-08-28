"""Species-independent production registry for HomoeoGWAS workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


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
    "analysis_shape", "expected",
}


def _resolve(base: Path, value: Any) -> Path | None:
    if value is None:
        return None
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

    run = RegistryRun(
        id=run_id,
        kind=kind,
        species=_require_text(raw, "species", run_id),
        panel=_require_text(raw, "panel", run_id),
        subgenomes=subgenomes,
        out_dir=_resolve(base, raw.get("out_dir")),
        phenotype=_resolve(base, raw.get("phenotype")),
        sample_col=raw.get("sample_col"),
        trait=raw.get("trait"),
        bed_prefixes=_mapping_paths(
            base, raw.get("bed_prefixes"), f"run {run_id!r} bed_prefixes"),
        snp_to_gene=_mapping_paths(
            base, raw.get("snp_to_gene"), f"run {run_id!r} snp_to_gene"),
        groups=_resolve(base, raw.get("groups")),
        hypothesis_unit=str(raw.get("hypothesis_unit", "group")).lower(),
        family_scope=str(raw.get("family_scope", "primary_only")).lower(),
        bootstrap_B=raw.get("bootstrap_B", 2000),
        n_jobs=raw.get("n_jobs", 8),
        include_hadamard=raw.get("include_hadamard", False),
        loco=raw.get("loco", False),
        run_plots=raw.get("run_plots", True),
        result_root=_resolve(base, raw.get("result_root")),
        analysis_shape=raw.get("analysis_shape"),
        expected=dict(raw.get("expected") or {}),
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
        or not run.analysis_shape.strip()
    ):
        raise RegistryError(
            f"run {run.id!r}: historical entries require result_root and analysis_shape")


def validate_registry(registry: RunRegistry) -> None:
    if registry.registry_version != 1:
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
    registry = RunRegistry(
        registry_version=raw.get("registry_version"),
        name=str(raw.get("name", "")).strip(),
        index_dir=_resolve(base, raw.get("index_dir") or "registry-index"),
        source_path=source,
        runs=tuple(_parse_run(item, base, row) for row, item in enumerate(runs_raw)),
    )
    if not registry.name:
        raise RegistryError("registry name must be a non-empty string")
    validate_registry(registry)
    return registry
