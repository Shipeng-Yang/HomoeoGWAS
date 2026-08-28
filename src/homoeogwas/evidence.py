"""Species-independent, provenance-bound candidate evidence table adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class EvidenceError(ValueError):
    """An evidence source cannot be joined without guessing its meaning."""


@dataclass(frozen=True)
class EvidenceSource:
    name: str
    kind: str
    path: Path
    gene_col: str
    support_col: str | None = None
    citation: str | None = None
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceManifest:
    evidence_version: int
    source_path: Path
    sources: tuple[EvidenceSource, ...]


_KINDS = {"annotation", "orthology", "expression", "qtl", "functional", "literature"}
_SOURCE_FIELDS = {
    "name", "kind", "path", "gene_col", "support_col", "citation", "columns"}
_TRUTHY = {"1", "true", "yes", "supported", "positive"}


def load_evidence_manifest(path: str | Path) -> EvidenceManifest:
    import yaml

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise EvidenceError(f"evidence manifest not found: {source_path}")
    raw = yaml.safe_load(source_path.read_text())
    if not isinstance(raw, dict):
        raise EvidenceError("evidence manifest must be a YAML mapping")
    unknown_root = sorted(set(raw) - {"evidence_version", "sources"})
    if unknown_root:
        raise EvidenceError(
            "evidence manifest has unsupported fields: " + ", ".join(unknown_root))
    if raw.get("evidence_version") != 1:
        raise EvidenceError("evidence_version must be 1")
    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise EvidenceError("evidence sources must be a non-empty list")
    sources = []
    for index, value in enumerate(raw_sources):
        if not isinstance(value, dict):
            raise EvidenceError(f"evidence source {index} must be a mapping")
        unknown = sorted(set(value) - _SOURCE_FIELDS)
        if unknown:
            raise EvidenceError(
                f"evidence source {index} has unsupported fields: {', '.join(unknown)}")
        name = str(value.get("name", "")).strip()
        kind = str(value.get("kind", "")).strip().lower()
        gene_col = str(value.get("gene_col", "")).strip()
        raw_path = value.get("path")
        if not name or not gene_col or raw_path is None:
            raise EvidenceError(
                f"evidence source {index} requires name, path and gene_col")
        if kind not in _KINDS:
            raise EvidenceError(
                f"evidence source {name!r} has unsupported kind {kind!r}")
        table_path = Path(str(raw_path)).expanduser()
        if not table_path.is_absolute():
            table_path = source_path.parent / table_path
        table_path = table_path.resolve()
        if not table_path.exists():
            raise EvidenceError(
                f"evidence source {name!r} table not found: {table_path}")
        columns_raw = value.get("columns") or ()
        if not isinstance(columns_raw, (list, tuple)) or not all(
            isinstance(column, str) and column.strip() for column in columns_raw
        ):
            raise EvidenceError(
                f"evidence source {name!r} columns must be a list of names")
        support = value.get("support_col")
        sources.append(EvidenceSource(
            name=name, kind=kind, path=table_path, gene_col=gene_col,
            support_col=(str(support).strip() if support is not None else None),
            citation=(str(value["citation"]).strip()
                      if value.get("citation") is not None else None),
            columns=tuple(column.strip() for column in columns_raw),
        ))
    names = [source.name for source in sources]
    duplicate = sorted({name for name in names if names.count(name) > 1})
    if duplicate:
        raise EvidenceError(
            "evidence source names must be unique: " + ", ".join(duplicate))
    return EvidenceManifest(1, source_path, tuple(sources))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _supported(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value == 1)
    return str(value).strip().lower() in _TRUTHY


def join_candidate_evidence(
    candidate_genes: pd.DataFrame,
    manifest: EvidenceManifest,
) -> tuple[pd.DataFrame, dict]:
    """Left-join every declared source while retaining unsupported candidates."""
    from .io import read_delimited

    if "gene_id" not in candidate_genes:
        raise EvidenceError("candidate gene table requires gene_id")
    candidates = candidate_genes.copy()
    candidates["gene_id"] = candidates["gene_id"].astype(str)
    frames = []
    provenance = {
        "evidence_version": manifest.evidence_version,
        "manifest": str(manifest.source_path),
        "sources": [],
    }
    for source in manifest.sources:
        table = read_delimited(source.path, dtype={source.gene_col: "string"})
        if source.gene_col not in table:
            raise EvidenceError(
                f"source {source.name!r} gene column {source.gene_col!r} is missing")
        if source.support_col and source.support_col not in table:
            raise EvidenceError(
                f"source {source.name!r} support column "
                f"{source.support_col!r} is missing")
        selected = list(source.columns) if source.columns else list(table.columns)
        required = [source.gene_col]
        if source.support_col:
            required.append(source.support_col)
        missing_selected = sorted(set(selected + required) - set(table.columns))
        if missing_selected:
            raise EvidenceError(
                f"source {source.name!r} selected columns are missing: "
                + ", ".join(missing_selected))
        selected = list(dict.fromkeys([*required, *selected]))
        source_table = table[selected].copy().rename(
            columns={source.gene_col: "gene_id"})
        source_table["gene_id"] = source_table["gene_id"].astype("string")
        merged = candidates.merge(
            source_table, on="gene_id", how="left", sort=False, indicator=True)
        merged.insert(1, "source", source.name)
        merged.insert(2, "kind", source.kind)
        merged.insert(3, "citation", source.citation)
        merged["matched"] = merged.pop("_merge").eq("both")
        merged["supported"] = (
            merged[source.support_col].map(_supported)
            if source.support_col else False)
        frames.append(merged)
        provenance["sources"].append({
            "name": source.name,
            "kind": source.kind,
            "path": str(source.path),
            "sha256": _sha256(source.path),
            "row_count": int(len(table)),
            "gene_col": source.gene_col,
            "support_col": source.support_col,
            "selected_columns": selected,
            "citation": source.citation,
        })
    return pd.concat(frames, ignore_index=True, sort=False), provenance


def assign_evidence_tiers(
    formal_units: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Assign the highest deterministic evidence tier across each formal unit's genes."""
    if "hypothesis_id" not in formal_units:
        raise EvidenceError("formal unit table requires hypothesis_id")
    required = {"gene_id", "kind", "matched", "supported"}
    if not required <= set(evidence.columns):
        raise EvidenceError(
            "joined evidence is missing columns: "
            + ", ".join(sorted(required - set(evidence.columns))))
    gene_columns = [column for column in formal_units if column.startswith("gene_")]
    records = []
    for formal in formal_units.to_dict(orient="records"):
        genes = {
            str(formal[column]) for column in gene_columns
            if formal.get(column) is not None and not pd.isna(formal[column])
        }
        linked = evidence.loc[evidence["gene_id"].astype(str).isin(genes)]
        supported = linked.loc[linked["supported"].astype(bool)]
        supported_kinds = set(supported["kind"].astype(str))
        matched_kinds = set(
            linked.loc[linked["matched"].astype(bool), "kind"].astype(str))
        if "functional" in supported_kinds:
            tier = "direct functional evidence"
        elif supported_kinds & {"qtl", "literature"}:
            tier = "QTL or literature support"
        elif "expression" in supported_kinds:
            tier = "matched-tissue expression support"
        elif matched_kinds & {"annotation", "orthology"}:
            tier = "orthology/domain annotation only"
        else:
            tier = "no linked external evidence"
        citations = sorted({
            str(value) for value in linked.get("citation", pd.Series(dtype=object)).dropna()
            if str(value).strip()
        })
        sources = sorted({
            str(value) for value in linked.get("source", pd.Series(dtype=object)).dropna()
            if str(value).strip()
        })
        records.append({
            "hypothesis_id": formal["hypothesis_id"],
            "candidate_genes": ";".join(sorted(genes)),
            "evidence_tier": tier,
            "evidence_sources": ";".join(sources),
            "citations": ";".join(citations),
        })
    return pd.DataFrame.from_records(records)
