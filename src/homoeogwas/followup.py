"""Candidate-only stability follow-up for canonical group omniB results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class FollowupError(ValueError):
    """A formal result cannot be followed up without changing its meaning."""


@dataclass(frozen=True)
class FollowupInputs:
    results_dir: Path
    config_path: Path
    ranking_path: Path
    result_path: Path
    audit_path: Path
    config: dict
    subgenomes: tuple[str, ...]
    trait: str
    hypothesis_unit: str
    ranking: pd.DataFrame
    formal_hits: pd.DataFrame
    formal_contract: dict


@dataclass(frozen=True)
class PreparedFollowup:
    inputs: FollowupInputs
    family: object
    expanded: object
    subdata: dict
    scores: object
    analyzed_samples: tuple[str, ...]
    sample_idx: np.ndarray
    y_raw: np.ndarray
    phenotype_rows: pd.DataFrame
    replay: pd.DataFrame
    formal_replay_max_abs_error: float


_RANKING_REQUIRED = {
    "rank", "hypothesis_id", "hypothesis_unit", "p_interaction",
    "p_adjusted_bootstrap_minp", "primary_sig",
}


def _one_path(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FollowupError(f"cannot find {label}")
    if len(paths) > 1:
        raise FollowupError(
            f"multiple {label} files found; pass an explicit path: "
            + ", ".join(str(path) for path in paths))
    return paths[0]


def load_followup_inputs(
    results_dir: str | Path,
    config: str | Path | None = None,
    ranking: str | Path | None = None,
) -> FollowupInputs:
    """Locate and validate the frozen canonical config and complete INT ranking."""
    import yaml

    from .interact import normalize_interact_config, validate_interact_config

    root = Path(results_dir).expanduser().resolve()
    if config is None:
        config_path = _one_path(
            sorted((root / "configs").glob("interact.generated.group.omnib.yaml")),
            "canonical generated interaction config")
    else:
        config_path = Path(config).expanduser().resolve()
    if not config_path.exists():
        raise FollowupError(f"config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("interact"), dict):
        raise FollowupError(f"interaction config is not a YAML mapping: {config_path}")
    raw_interact = raw["interact"]
    if (
        str(raw_interact.get("mode", "")).lower() != "group"
        or "groups" not in raw_interact
    ):
        raise FollowupError(
            "legacy interaction result requires explicit canonical migration; "
            "it will not be relabelled by follow-up")
    cfg = normalize_interact_config(raw)
    try:
        validate_interact_config(cfg)
    except (ValueError, SystemExit) as exc:
        raise FollowupError(f"canonical interaction config is invalid: {exc}") from exc
    ic = cfg["interact"]
    requirements = {
        "statistic": str(ic.get("statistic", "")).lower() == "omnib",
        "primary_transform": str(ic.get("primary_transform", "")).upper() == "INT",
        "primary_multiplicity": (
            str(ic.get("primary_multiplicity", "")).lower() == "bootstrap_minp"),
        "subset_order": ic.get("subset_order") == 2,
    }
    failed = [name for name, passed in requirements.items() if not passed]
    if failed:
        raise FollowupError(
            "follow-up requires canonical group omniB fields: " + ", ".join(failed))
    hypothesis_unit = str(ic.get("hypothesis_unit", "")).lower()
    if hypothesis_unit not in {"edge", "group"}:
        raise FollowupError("canonical hypothesis_unit must be edge or group")
    trait = str(ic.get("trait", "")).strip()
    if not trait:
        raise FollowupError("canonical interaction config is missing trait")
    if ic.get("covariates"):
        raise FollowupError(
            "covariate-bearing canonical results are not yet supported by "
            "deletion follow-up; rerun is refused rather than changing the null model")
    if ranking is None:
        ranking_path = _one_path(
            sorted(root.glob(f"interact_{trait}_ranking_group_INT.tsv")),
            "canonical complete INT ranking")
    else:
        ranking_path = Path(ranking).expanduser().resolve()
    if not ranking_path.exists():
        raise FollowupError(f"ranking not found: {ranking_path}")
    frame = pd.read_csv(ranking_path, sep="\t")
    missing = sorted(_RANKING_REQUIRED - set(frame.columns))
    if missing:
        raise FollowupError(
            "canonical ranking is missing columns: " + ", ".join(missing))
    result_path = _one_path(
        sorted(root.glob(f"interact_{trait}.json")),
        "formal interaction result JSON")
    audit_path = root / "audit" / "homoeogwas_audit.json"
    if not audit_path.exists():
        raise FollowupError(
            f"canonical follow-up requires a completed audit: {audit_path}")
    try:
        formal_contract = _validate_formal_contract(
            cfg, config_path, frame, ranking_path, result_path, audit_path)
    except FollowupError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise FollowupError(
            f"formal result contract validation failed: {exc}") from exc
    primary = pd.to_numeric(frame["primary_sig"], errors="coerce").fillna(0).astype(int)
    formal = frame.loc[primary == 1].copy().reset_index(drop=True)
    if formal["hypothesis_id"].duplicated().any():
        raise FollowupError("formal ranking contains duplicate significant hypothesis IDs")
    return FollowupInputs(
        results_dir=root,
        config_path=config_path,
        ranking_path=ranking_path,
        result_path=result_path,
        audit_path=audit_path,
        config=cfg,
        subgenomes=tuple(str(value) for value in ic["subgenomes"]),
        trait=trait,
        hypothesis_unit=hypothesis_unit,
        ranking=frame,
        formal_hits=formal,
        formal_contract=formal_contract,
    )


def _resolve_config_path(value, config_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    beside = (config_dir / path).resolve()
    if beside.exists():
        return beside
    below_result_root = (config_dir.parent / path).resolve()
    return below_result_root if below_result_root.exists() else path.resolve()


def _read_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FollowupError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FollowupError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_formal_contract(
    cfg: dict,
    config_path: Path,
    ranking: pd.DataFrame,
    ranking_path: Path,
    result_path: Path,
    audit_path: Path,
) -> dict:
    """Bind discovery labels to the full audited canonical family."""
    from .group_family import expand_pair_edges, load_master_group_family
    from .omnib_family import _family_provenance

    ic = cfg["interact"]
    subgenomes = tuple(str(value) for value in ic["subgenomes"])
    declared_unit = str(ic["hypothesis_unit"]).lower()
    family_scope = str(ic.get("family_scope", "primary_only")).lower()
    family = load_master_group_family(
        _resolve_config_path(ic["groups"], config_path.parent), subgenomes)
    expanded = expand_pair_edges(family)
    computed_family = _family_provenance(family, expanded)
    edge_ids = [f"edge:{edge.edge_id}" for edge in expanded.edges]
    group_ids = [f"group:{group_id}" for group_id in family.group_ids]
    expected_ids = (
        edge_ids + group_ids if family_scope == "joint"
        else edge_ids if declared_unit == "edge" else group_ids)
    expected_units = (
        {"edge", "group"} if family_scope == "joint" else {declared_unit})

    ranks = pd.to_numeric(ranking["rank"], errors="coerce")
    if ranks.isna().any() or not np.array_equal(
        ranks.to_numpy(int), np.arange(len(ranking))
    ):
        raise FollowupError("formal ranking ranks must be contiguous from zero")
    observed_ids = ranking["hypothesis_id"].astype(str).tolist()
    if len(observed_ids) != len(set(observed_ids)):
        raise FollowupError("formal ranking contains duplicate hypothesis IDs")
    if len(observed_ids) != len(expected_ids) or set(observed_ids) != set(expected_ids):
        raise FollowupError(
            "formal ranking does not contain the exact full hypothesis inventory")
    units = set(ranking["hypothesis_unit"].dropna().astype(str).str.lower())
    if units != expected_units:
        raise FollowupError(
            f"ranking hypothesis units {sorted(units)} do not match "
            f"the declared family {sorted(expected_units)}")
    p_values = pd.to_numeric(ranking["p_interaction"], errors="coerce")
    unestimable = pd.to_numeric(
        ranking["p_unestimable"], errors="coerce").fillna(1).astype(int)
    valid_p = p_values[unestimable == 0]
    if valid_p.isna().any() or not valid_p.is_monotonic_increasing:
        raise FollowupError(
            "formal ranking is not a complete ascending raw-P ranking")

    payload = _read_json_object(result_path, "formal interaction result")
    provenance = payload.get("provenance") or {}
    primary = (payload.get("results") or {}).get("INT") or {}
    diagnostics = primary.get("model_diagnostics") or {}
    result_family = diagnostics.get("family_provenance") or {}
    fwer = diagnostics.get("bootstrap_fwer") or {}
    expected_contract = {
        "mode": "group", "statistic": "omniB",
        "primary_transform": "INT",
        "primary_multiplicity": "bootstrap_minp",
        "hypothesis_unit": declared_unit,
        "family_scope": family_scope,
        "subset_order": 2,
    }
    mismatched = [
        key for key, expected in expected_contract.items()
        if provenance.get(key) != expected]
    if payload.get("command") != "interact" or payload.get("trait") != ic["trait"]:
        mismatched.extend(["command_or_trait"])
    if mismatched:
        raise FollowupError(
            "formal result provenance does not match the canonical config: "
            + ", ".join(mismatched))
    config_sha256 = _sha256_file(config_path)
    reported_config_sha256 = provenance.get("config_sha256")
    if reported_config_sha256 is None:
        manifest_ref = (cfg.get("provenance") or {}).get("pre_run_manifest")
        if manifest_ref is None:
            raise FollowupError(
                "formal result does not bind the exact generated config SHA-256")
        manifest_path = _resolve_config_path(manifest_ref, config_path.parent)
        manifest = _read_json_object(manifest_path, "formal pre-run manifest")
        reported_config_sha256 = (
            manifest.get("config_sha256")
            or (manifest.get("config") or {}).get("sha256"))
        for key in ("group_family_sha256", "edge_family_sha256"):
            if manifest.get(key) != computed_family[key]:
                raise FollowupError(
                    f"formal pre-run manifest family mismatch for {key}")
    if reported_config_sha256 != config_sha256:
        raise FollowupError(
            "formal result config SHA-256 does not match the generated config")
    for key, expected in computed_family.items():
        if provenance.get(key) != expected or result_family.get(key) != expected:
            raise FollowupError(
                f"formal result family provenance mismatch for {key}")
    expected_count = len(expected_ids)
    n_valid = int((unestimable == 0).sum())
    if primary.get("n_planned") != expected_count or primary.get("n_valid") != n_valid:
        raise FollowupError(
            "formal result planned/valid counts do not match the complete ranking")
    if fwer.get("family_id") != (
        "joint" if family_scope == "joint" else declared_unit
    ) or fwer.get("declared_hypothesis_unit") != declared_unit:
        raise FollowupError("formal bootstrap family declaration is inconsistent")
    threshold = fwer.get("threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise FollowupError("formal bootstrap-minP threshold is missing or invalid")
    labels = pd.to_numeric(
        ranking["primary_sig"], errors="coerce").fillna(-1).astype(int)
    if not labels.isin([0, 1]).all():
        raise FollowupError("formal discovery labels must be binary")
    expected_labels = ((unestimable == 0) & (p_values < float(threshold))).astype(int)
    if not np.array_equal(labels.to_numpy(), expected_labels.to_numpy()):
        raise FollowupError(
            "formal discovery labels do not match the audited bootstrap-minP threshold")
    significant_ids = set(ranking.loc[labels == 1, "hypothesis_id"].astype(str))
    result_sig = fwer.get("sig")
    if not isinstance(result_sig, list):
        result_sig = primary.get("sig")
    if not isinstance(result_sig, list):
        raise FollowupError("formal result is missing its rejected hypothesis records")
    result_by_id = {
        str(record.get("hypothesis_id")): record
        for record in result_sig if isinstance(record, dict)
    }
    if set(result_by_id) != significant_ids or fwer.get("n_rejected") != len(
        significant_ids
    ):
        raise FollowupError(
            "formal discovery labels do not match the result JSON rejection set")
    ranking_by_id = ranking.set_index(ranking["hypothesis_id"].astype(str))
    for hypothesis_id, record in result_by_id.items():
        row = ranking_by_id.loc[hypothesis_id]
        for column in ("p_interaction", "p_adjusted_bootstrap_minp"):
            if not np.isclose(
                float(row[column]), float(record[column]), rtol=1e-12, atol=1e-15
            ):
                raise FollowupError(
                    f"formal rejected-unit {column} differs between ranking and result JSON")

    audit = _read_json_object(audit_path, "formal audit")
    if str(audit.get("overall_status", "")).upper() in {"INVALID", "FAILED"}:
        raise FollowupError(
            f"formal audit status is not follow-up eligible: "
            f"{audit.get('overall_status')}")
    records = audit.get("records")
    if audit.get("n_results") != 1 or not isinstance(records, list) or len(records) != 1:
        raise FollowupError("formal audit must contain exactly one interaction result")
    record = records[0]
    try:
        audited_source = Path(str(record.get("source"))).resolve()
    except (TypeError, ValueError) as exc:
        raise FollowupError("formal audit source path is invalid") from exc
    if (
        audited_source != result_path.resolve()
        or record.get("command") != "interact"
        or record.get("mode") != "group"
        or record.get("statistic") != "omniB"
        or record.get("n_planned") != expected_count
        or record.get("n_valid") != n_valid
        or record.get("discovery_count") != len(significant_ids)
    ):
        raise FollowupError(
            "formal audit does not bind the result, family counts, and discoveries")
    return {
        **computed_family,
        "primary_family": "joint" if family_scope == "joint" else declared_unit,
        "n_planned": expected_count,
        "n_valid": n_valid,
        "n_significant": len(significant_ids),
        "bootstrap_B": fwer.get("B", primary.get("bootstrap_B")),
        "bootstrap_threshold": float(threshold),
        "config_sha256": config_sha256,
        "ranking_sha256": _sha256_file(ranking_path),
        "result_sha256": _sha256_file(result_path),
        "audit_sha256": _sha256_file(audit_path),
    }


def _load_verified_subgenomes(interact: dict, subgenomes, config_dir: Path) -> dict:
    from .interact import _load_subgenome

    return {
        sub: _load_subgenome(
            str(_resolve_config_path(interact["genotype"][sub], config_dir)),
            str(_resolve_config_path(interact["snp_to_gene"][sub], config_dir)),
            verify_mapping=True,
        )
        for sub in subgenomes
    }


def extract_primary_scores(subset_scores, prepared: PreparedFollowup) -> pd.DataFrame:
    """Extract only frozen formal IDs from one edge/group subset score matrix."""
    from .omnib_family import OMNIB_COMPONENT_NAMES

    edge_by_id = {
        edge.edge_id: index for index, edge in enumerate(prepared.expanded.edges)
    }
    group_by_id = {
        group_id: index
        for index, group_id in enumerate(prepared.family.group_ids)
    }
    records = []
    for formal in prepared.inputs.formal_hits.to_dict(orient="records"):
        hypothesis_id = str(formal["hypothesis_id"])
        driving_edge = None
        component_name = None
        component_p = np.nan
        if hypothesis_id.startswith("edge:"):
            edge_id = hypothesis_id.removeprefix("edge:")
            if edge_id not in edge_by_id:
                raise FollowupError(
                    f"formal edge is absent from the master family: {edge_id}")
            edge_index = edge_by_id[edge_id]
            p_value = float(subset_scores.edge_p[edge_index])
            components = subset_scores.edge_components[edge_index]
            if np.isfinite(components).any():
                component_index = int(np.nanargmin(components))
                component_name = OMNIB_COMPONENT_NAMES[component_index]
                component_p = float(components[component_index])
            driving_edge = edge_id
        elif hypothesis_id.startswith("group:"):
            group_id = hypothesis_id.removeprefix("group:")
            if group_id not in group_by_id:
                raise FollowupError(
                    f"formal group is absent from the master family: {group_id}")
            group_index = group_by_id[group_id]
            p_value = float(subset_scores.group_p[group_index])
            edge_indices = np.asarray(
                prepared.expanded.group_edge_indices[group_index], int)
            finite = edge_indices[
                np.isfinite(subset_scores.edge_p[edge_indices])]
            if finite.size:
                edge_index = int(finite[
                    np.argmin(subset_scores.edge_p[finite])])
                edge = prepared.expanded.edges[edge_index]
                driving_edge = edge.edge_id
                components = subset_scores.edge_components[edge_index]
                if np.isfinite(components).any():
                    component_index = int(np.nanargmin(components))
                    component_name = OMNIB_COMPONENT_NAMES[component_index]
                    component_p = float(components[component_index])
        else:
            raise FollowupError(
                f"unsupported canonical hypothesis ID: {hypothesis_id}")
        records.append({
            "rank": int(formal["rank"]),
            "hypothesis_id": hypothesis_id,
            "p_interaction": p_value,
            "driving_edge": driving_edge,
            "driving_component": component_name,
            "driving_component_p": component_p,
        })
    return pd.DataFrame.from_records(records)


def prepare_formal_replay(
    inputs: FollowupInputs,
    *,
    n_jobs: int = 8,
) -> PreparedFollowup:
    """Load exact analysis inputs, freeze features, and reproduce formal hits."""
    from .group_family import load_master_group_family
    from .io import read_delimited
    from .omnib_family import score_omnib_family, score_omnib_subset

    ic = inputs.config["interact"]
    config_dir = inputs.config_path.parent
    family = load_master_group_family(
        _resolve_config_path(ic["groups"], config_dir), inputs.subgenomes)
    subdata = _load_verified_subgenomes(ic, inputs.subgenomes, config_dir)
    samples = tuple(str(value) for value in subdata[inputs.subgenomes[0]].samples)
    if len(samples) != len(set(samples)):
        raise FollowupError("genotype sample IDs are not unique")
    for sub in inputs.subgenomes[1:]:
        observed = tuple(str(value) for value in subdata[sub].samples)
        if observed != samples:
            raise FollowupError(
                f"subgenome sample order mismatch between {inputs.subgenomes[0]} and {sub}")
    sample_col = str(ic.get("sample_col", "sample"))
    phenotype_path = _resolve_config_path(ic["phenotype"], config_dir)
    phenotype_rows = read_delimited(
        phenotype_path, dtype={sample_col: "string"})
    if sample_col not in phenotype_rows or inputs.trait not in phenotype_rows:
        raise FollowupError(
            f"phenotype must contain {sample_col!r} and {inputs.trait!r}")
    phenotype_rows = phenotype_rows.loc[
        phenotype_rows[sample_col].notna()].copy()
    phenotype_rows[sample_col] = phenotype_rows[sample_col].astype(str)
    phenotype_rows[inputs.trait] = pd.to_numeric(
        phenotype_rows[inputs.trait], errors="coerce")
    phenotype = phenotype_rows.groupby(sample_col, sort=False)[inputs.trait].mean()
    analyzed = tuple(
        sample for sample in samples
        if sample in phenotype.index and pd.notna(phenotype.loc[sample]))
    if len(analyzed) < 10:
        raise FollowupError(
            f"only {len(analyzed)} samples overlap genotype and phenotype; need at least 10")
    sample_lookup = {sample: index for index, sample in enumerate(samples)}
    sample_idx = np.asarray([sample_lookup[sample] for sample in analyzed], int)
    y_raw = phenotype.loc[list(analyzed)].to_numpy(float)
    burden = ic.get("burden") or {}
    grm = ic.get("grm") or {}
    calibration = ic.get("calibration") or {}
    scores, expanded = score_omnib_family(
        subdata, family, y_raw, sample_idx,
        cap=int(burden.get("cap", 150)),
        n_pc=int(burden.get("n_pc", 3)),
        transform="INT", bootstrap_B=0,
        bootstrap_seed=int(calibration.get("seed", 2026)),
        n_jobs=n_jobs, grm_method=str(grm.get("method", "grm_from_X")),
        maf_min=float(grm.get("maf_min", 0.01)),
        burden_maf=float(burden.get("maf_min", 0.01)),
        min_snp=int(burden.get("min_snp", 3)),
        covariates=None,
    )
    placeholder = PreparedFollowup(
        inputs=inputs, family=family, expanded=expanded,
        subdata=subdata, scores=scores, analyzed_samples=analyzed,
        sample_idx=sample_idx, y_raw=y_raw, phenotype_rows=phenotype_rows,
        replay=pd.DataFrame(), formal_replay_max_abs_error=float("nan"))
    subset = score_omnib_subset(
        scores, family, expanded, np.arange(len(analyzed)), y_raw,
        n_jobs=n_jobs)
    replay = extract_primary_scores(subset, placeholder)
    joined = replay.merge(
        inputs.formal_hits[[
            "hypothesis_id", "p_interaction", "driving_edge",
            "driving_component"]],
        on="hypothesis_id", suffixes=("_replay", "_formal"),
        validate="one_to_one")
    if len(joined) != len(inputs.formal_hits):
        raise FollowupError("formal hit identity reproduction failed")
    p_error = float(np.max(np.abs(
        joined["p_interaction_replay"] - joined["p_interaction_formal"])))
    if not np.allclose(
        joined["p_interaction_replay"], joined["p_interaction_formal"],
        rtol=1e-6, atol=1e-10):
        raise FollowupError(
            f"formal p reproduction failed: max_abs_error={p_error}")
    for column in ("driving_edge", "driving_component"):
        formal_col = f"{column}_formal"
        replay_col = f"{column}_replay"
        comparable = joined[formal_col].notna()
        if comparable.any() and not joined.loc[comparable, replay_col].equals(
            joined.loc[comparable, formal_col]
        ):
            raise FollowupError(f"formal {column} reproduction failed")
    return PreparedFollowup(
        inputs=inputs, family=family, expanded=expanded,
        subdata=subdata, scores=scores, analyzed_samples=analyzed,
        sample_idx=sample_idx, y_raw=y_raw, phenotype_rows=phenotype_rows,
        replay=replay, formal_replay_max_abs_error=p_error)


def deterministic_folds(
    samples: Sequence[str],
    *,
    n_folds: int = 20,
    seed: int = 2026,
) -> tuple[tuple[str, ...], ...]:
    values = sorted(str(value) for value in samples)
    if len(values) != len(set(values)):
        raise FollowupError("sample IDs must be unique")
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) \
            or n_folds < 2 or n_folds > len(values):
        raise FollowupError(
            "material folds must be between 2 and the number of samples")
    ordered = sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{seed}\0{value}".encode()).digest(),
    )
    return tuple(tuple(ordered[index::n_folds]) for index in range(n_folds))


def run_material_deletion(
    prepared: PreparedFollowup,
    *,
    n_folds: int = 20,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """Score formal units after deterministic, disjoint sample-fold deletion."""
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    from .omnib_family import score_omnib_subset

    folds = deterministic_folds(
        prepared.analyzed_samples, n_folds=n_folds, seed=2026)
    sample_to_local = {
        sample: index for index, sample in enumerate(prepared.analyzed_samples)
    }

    def score(index: int, deleted: tuple[str, ...]) -> pd.DataFrame:
        deleted_set = set(deleted)
        keep = np.asarray([
            local for sample, local in sample_to_local.items()
            if sample not in deleted_set
        ], int)
        subset = score_omnib_subset(
            prepared.scores, prepared.family, prepared.expanded,
            keep, prepared.y_raw[keep], n_jobs=1)
        frame = extract_primary_scores(subset, prepared)
        frame.insert(0, "deleted_fold", f"fold_{index + 1:02d}")
        frame.insert(1, "n", int(keep.size))
        frame.insert(2, "deleted_samples", ";".join(deleted))
        return frame

    with threadpool_limits(limits=1):
        frames = Parallel(n_jobs=int(n_jobs), backend="threading") (
            delayed(score)(index, deleted)
            for index, deleted in enumerate(folds)
        )
    return pd.concat(frames, ignore_index=True)


def run_environment_deletion(
    prepared: PreparedFollowup,
    environment_col: str | None,
    *,
    n_jobs: int = 8,
) -> tuple[pd.DataFrame, str]:
    """Delete phenotype records by environment and reaggregate repeated samples."""
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    from .omnib_family import score_omnib_subset

    if not environment_col:
        return pd.DataFrame(), "NOT_AVAILABLE_NO_ENVIRONMENT_COLUMN"
    rows = prepared.phenotype_rows
    if environment_col not in rows.columns:
        return pd.DataFrame(), "NOT_AVAILABLE_NO_ENVIRONMENT_COLUMN"
    levels = sorted(rows[environment_col].dropna().astype(str).unique())
    if not levels:
        return pd.DataFrame(), "NOT_AVAILABLE_NO_ENVIRONMENT_LEVELS"
    sample_col = str(prepared.inputs.config["interact"].get(
        "sample_col", "sample"))
    analyzed_lookup = {
        sample: index for index, sample in enumerate(prepared.analyzed_samples)
    }

    def score(level: str):
        environment = rows[environment_col].astype("string")
        remaining = rows.loc[environment.isna() | (environment.astype(str) != level)]
        phenotype = remaining.groupby(sample_col, sort=False)[
            prepared.inputs.trait].mean()
        kept_samples = [
            sample for sample in prepared.analyzed_samples
            if sample in phenotype.index and pd.notna(phenotype.loc[sample])
        ]
        if len(kept_samples) < 10:
            unavailable = prepared.inputs.formal_hits[[
                "rank", "hypothesis_id"]].copy()
            unavailable.insert(0, "deleted_environment", level)
            unavailable.insert(1, "n", int(len(kept_samples)))
            unavailable.insert(2, "available", 0)
            unavailable.insert(3, "unavailable_reason", "LT10_RETAINED_SAMPLES")
            unavailable["p_interaction"] = np.nan
            unavailable["driving_edge"] = None
            unavailable["driving_component"] = None
            unavailable["driving_component_p"] = np.nan
            return unavailable
        keep = np.asarray([analyzed_lookup[sample] for sample in kept_samples], int)
        y_raw = phenotype.loc[kept_samples].to_numpy(float)
        subset = score_omnib_subset(
            prepared.scores, prepared.family, prepared.expanded,
            keep, y_raw, n_jobs=1)
        frame = extract_primary_scores(subset, prepared)
        frame.insert(0, "deleted_environment", level)
        frame.insert(1, "n", int(keep.size))
        frame.insert(2, "available", 1)
        frame.insert(3, "unavailable_reason", None)
        return frame

    with threadpool_limits(limits=1):
        values = Parallel(n_jobs=int(n_jobs), backend="threading") (
            delayed(score)(level) for level in levels)
    frame = pd.concat(values, ignore_index=True)
    n_available = int(frame.loc[frame["available"] == 1,
                                "deleted_environment"].nunique())
    if n_available == 0:
        return frame, "NOT_AVAILABLE_LT10_SAMPLES"
    status = "COMPLETED" if n_available == len(levels) \
        else "COMPLETED_WITH_UNAVAILABLE_LEVELS"
    return frame, status


def summarize_stability(
    prepared: PreparedFollowup,
    material: pd.DataFrame,
    environment: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for formal in prepared.inputs.formal_hits.to_dict(orient="records"):
        hypothesis_id = str(formal["hypothesis_id"])
        material_values = material.loc[
            material["hypothesis_id"] == hypothesis_id]
        environment_values = (
            environment.loc[environment["hypothesis_id"] == hypothesis_id]
            if not environment.empty else pd.DataFrame())
        if not environment_values.empty and "available" in environment_values:
            environment_values = environment_values.loc[
                environment_values["available"] == 1]
        record = {
            "rank": int(formal["rank"]),
            "hypothesis_id": hypothesis_id,
            "formal_p": float(formal["p_interaction"]),
            "formal_p_fwer": float(formal["p_adjusted_bootstrap_minp"]),
            "material_p_median": float(material_values["p_interaction"].median()),
            "material_p_max": float(material_values["p_interaction"].max()),
            "material_nominal_support_fraction": float(
                (material_values["p_interaction"] < 0.05).mean()),
            "material_driver_agreement_fraction": float(
                (material_values["driving_component"]
                 == formal.get("driving_component")).mean()),
            "material_driving_edge_agreement_fraction": float(
                (material_values["driving_edge"]
                 == formal.get("driving_edge")).mean()),
            "material_min_n": int(material_values["n"].min()),
            "interpretation": (
                "candidate-only internal sensitivity; not a new discovery "
                "family or independent replication"),
        }
        if not environment_values.empty:
            record.update({
                "environment_p_median": float(
                    environment_values["p_interaction"].median()),
                "environment_p_max": float(
                    environment_values["p_interaction"].max()),
                "environment_nominal_support_fraction": float(
                    (environment_values["p_interaction"] < 0.05).mean()),
                "environment_driver_agreement_fraction": float(
                    (environment_values["driving_component"]
                     == formal.get("driving_component")).mean()),
                "environment_driving_edge_agreement_fraction": float(
                    (environment_values["driving_edge"]
                     == formal.get("driving_edge")).mean()),
                "environment_min_n": int(environment_values["n"].min()),
            })
        records.append(record)
    return pd.DataFrame.from_records(records)


def candidate_coordinates(prepared: PreparedFollowup) -> pd.DataFrame:
    """Attach exact analysis-build gene coordinates to each formal unit."""
    rows = []
    for formal in prepared.inputs.formal_hits.to_dict(orient="records"):
        hypothesis_id = str(formal["hypothesis_id"])
        if hypothesis_id.startswith("edge:"):
            copy_set = (str(formal["sub_x"]), str(formal["sub_y"]))
        elif hypothesis_id.startswith("group:"):
            copy_set = prepared.inputs.subgenomes
        else:
            raise FollowupError(
                f"unsupported canonical hypothesis ID: {hypothesis_id}")
        record = dict(formal)
        record["copy_set"] = tuple(copy_set)
        for sub in copy_set:
            gene = formal.get(f"gene_{sub}")
            if gene is None or pd.isna(gene) or not str(gene).strip():
                raise FollowupError(
                    f"formal unit {hypothesis_id} is missing gene_{sub}")
            gene = str(gene)
            data = prepared.subdata[sub]
            indices = prepared.scores.gated_snp.get((sub, gene))
            if indices is None:
                indices = data.gene_snp.get(gene)
            if indices is None or len(indices) == 0:
                raise FollowupError(
                    f"formal gene has no exact analysis SNP block: {sub}:{gene}")
            if data.chunk is None or not hasattr(data.chunk, "chrom") \
                    or not hasattr(data.chunk, "pos"):
                raise FollowupError(
                    f"analysis genotype lacks chromosome coordinates for {sub}:{gene}")
            indices = np.asarray(indices, int)
            chrom = np.asarray(data.chunk.chrom, dtype=object)[indices].astype(str)
            positions = np.asarray(data.chunk.pos, int)[indices]
            unique, counts = np.unique(chrom, return_counts=True)
            selected_chrom = str(unique[int(np.argmax(counts))])
            selected_positions = positions[chrom == selected_chrom]
            record[f"chrom_{sub}"] = selected_chrom
            record[f"pos_{sub}"] = int(np.median(selected_positions))
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def _pc1_lookup(feature_pc1, sub: str, left: str, right: str) -> float:
    value = feature_pc1.get((sub, left, right))
    if value is None:
        value = feature_pc1.get((sub, right, left))
    return float(value) if value is not None else float("nan")


def cluster_formal_units(
    hits: pd.DataFrame,
    feature_pc1,
    *,
    max_distance_bp: int = 1_000_000,
    pc1_r2_threshold: float = 0.64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge formal units by an all-compared-copy physical or LD rule."""
    required = {"rank", "hypothesis_id", "copy_set"}
    missing = sorted(required - set(hits.columns))
    if missing:
        raise FollowupError(
            "coordinate table is missing columns: " + ", ".join(missing))
    result = hits.copy().reset_index(drop=True)
    parent = list(range(len(result)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    links = []
    for left in range(len(result)):
        for right in range(left + 1, len(result)):
            first, second = result.iloc[left], result.iloc[right]
            copies_left = tuple(first["copy_set"])
            copies_right = tuple(second["copy_set"])
            record = {
                "rank_i": int(first["rank"]),
                "rank_j": int(second["rank"]),
                "hypothesis_id_i": first["hypothesis_id"],
                "hypothesis_id_j": second["hypothesis_id"],
                "merge": False,
                "merge_basis": "none",
            }
            if copies_left != copies_right:
                record["merge_reason"] = "different_copy_set"
                links.append(record)
                continue
            if len(copies_left) < 2:
                record["merge_reason"] = "fewer_than_two_comparable_copies"
                links.append(record)
                continue
            physical = True
            high_ld = True
            for sub in copies_left:
                same_chrom = str(first[f"chrom_{sub}"]) == str(
                    second[f"chrom_{sub}"])
                distance = (
                    abs(int(first[f"pos_{sub}"]) - int(second[f"pos_{sub}"]))
                    if same_chrom else None)
                r2 = _pc1_lookup(
                    feature_pc1, sub, str(first[f"gene_{sub}"]),
                    str(second[f"gene_{sub}"]))
                record[f"distance_{sub}_bp"] = distance
                record[f"pc1_r2_{sub}"] = r2
                physical = physical and distance is not None \
                    and distance <= max_distance_bp
                high_ld = high_ld and np.isfinite(r2) \
                    and r2 >= pc1_r2_threshold
            merge = bool(physical or high_ld)
            record["merge"] = merge
            if physical and high_ld:
                record["merge_basis"] = "physical_and_ld"
                record["merge_reason"] = (
                    f"all_compared_copies_within_{max_distance_bp}bp_and_"
                    f"pc1_r2_ge_{pc1_r2_threshold:g}")
            elif physical:
                record["merge_basis"] = "physical"
                record["merge_reason"] = (
                    f"all_compared_copies_within_{max_distance_bp}bp")
            elif high_ld:
                record["merge_basis"] = "ld"
                record["merge_reason"] = (
                    f"pc1_r2_ge_{pc1_r2_threshold:g}_in_all_compared_copies")
            else:
                record["merge_reason"] = "separate"
            if merge:
                union(left, right)
            links.append(record)

    roots = sorted(
        {find(index) for index in range(len(result))},
        key=lambda root: int(result.iloc[root]["rank"]))
    locus_by_root = {
        root: f"HOMEO_LOCUS_{index + 1:02d}"
        for index, root in enumerate(roots)
    }
    result["locus_id"] = [
        locus_by_root[find(index)] for index in range(len(result))]
    result["n_units_in_locus"] = result.groupby("locus_id")[
        "hypothesis_id"].transform("size")
    return result, pd.DataFrame.from_records(links)


def _atomic_text(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(body)
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _write_tsv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    try:
        frame.to_csv(
            temporary, sep="\t", index=False, na_rep="NA",
            float_format="%.17g")
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _followup_identity(
    inputs: FollowupInputs,
    *,
    material_folds: int,
    environment_col: str | None,
    evidence_manifest,
) -> tuple[str, dict]:
    from . import __version__, omnib_family
    from . import evidence as evidence_module

    evidence_identity = None
    if evidence_manifest is not None:
        evidence_identity = {
            "manifest": str(evidence_manifest.source_path),
            "manifest_sha256": _sha256_file(evidence_manifest.source_path),
            "sources": [
                {
                    "name": source.name,
                    "path": str(source.path),
                    "sha256": _sha256_file(source.path),
                }
                for source in evidence_manifest.sources
            ],
        }
    payload = {
        "identity_schema": "homoeogwas-group-omnib-followup-identity-v1",
        "homoeogwas_version": __version__,
        "implementation_sha256": {
            "followup.py": _sha256_file(Path(__file__)),
            "evidence.py": _sha256_file(Path(evidence_module.__file__)),
            "omnib_family.py": _sha256_file(Path(omnib_family.__file__)),
        },
        "formal_contract": inputs.formal_contract,
        "material_folds": int(material_folds),
        "material_fold_seed": 2026,
        "environment_col": environment_col,
        "region_rule": {"max_distance_bp": 1_000_000, "pc1_r2": 0.64},
        "evidence": evidence_identity,
    }
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest(), payload


def _open_followup_generation(
    out: Path,
    identity: str,
    payload: dict,
) -> dict | None:
    """Create one immutable output generation or return a completed match."""
    identity_path = out / "followup_identity.json"
    summary_path = out / "followup_summary.json"
    if out.exists() and any(out.iterdir()):
        if not identity_path.exists():
            raise FollowupError(
                f"follow-up output is non-empty without an identity: {out}; "
                "choose a new --out-dir")
        existing = _read_json_object(identity_path, "follow-up output identity")
        if existing.get("identity") != identity:
            raise FollowupError(
                "follow-up output identity mismatch; choose a new --out-dir")
        if existing.get("status") == "COMPLETE" and summary_path.exists():
            return _read_json_object(summary_path, "completed follow-up summary")
        raise FollowupError(
            "matching follow-up output is incomplete; preserve it for diagnosis "
            "and choose a new --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        identity_path,
        json.dumps({
            "identity": identity, "status": "RUNNING", "payload": payload,
        }, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return None


def _complete_followup_generation(out: Path, identity: str, payload: dict) -> None:
    _atomic_text(
        out / "followup_identity.json",
        json.dumps({
            "identity": identity, "status": "COMPLETE", "payload": payload,
        }, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _pc1_map(prepared: PreparedFollowup, coordinates: pd.DataFrame) -> dict:
    values = {}
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            first, second = coordinates.iloc[left], coordinates.iloc[right]
            copies = tuple(first["copy_set"])
            if copies != tuple(second["copy_set"]):
                continue
            for sub in copies:
                gene_left = str(first[f"gene_{sub}"])
                gene_right = str(second[f"gene_{sub}"])
                feature_left = prepared.scores.feature_cache.get(
                    (sub, gene_left))
                feature_right = prepared.scores.feature_cache.get(
                    (sub, gene_right))
                if feature_left is None or feature_right is None:
                    continue
                pc_left = np.asarray(feature_left[1], float).reshape(-1)
                pc_right = np.asarray(feature_right[1], float).reshape(-1)
                if np.std(pc_left) <= 1e-12 or np.std(pc_right) <= 1e-12:
                    r2 = float("nan")
                else:
                    r2 = float(np.corrcoef(pc_left, pc_right)[0, 1] ** 2)
                values[(sub, gene_left, gene_right)] = r2
    return values


def _candidate_genes(coordinates: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in coordinates.to_dict(orient="records"):
        for sub in tuple(row["copy_set"]):
            records.append({
                "gene_id": str(row[f"gene_{sub}"]),
                "subgenome": sub,
            })
    return pd.DataFrame.from_records(records).drop_duplicates(
        ["gene_id", "subgenome"], keep="first").reset_index(drop=True)


def _followup_audit(
    prepared: PreparedFollowup,
    material: pd.DataFrame,
    loci: pd.DataFrame,
    candidate_genes: pd.DataFrame,
    tiers: pd.DataFrame,
) -> dict:
    fold_manifest = material[["deleted_fold", "deleted_samples"]].drop_duplicates()
    sample_counts = Counter()
    for value in fold_manifest["deleted_samples"]:
        sample_counts.update(str(value).split(";"))
    formal_ids = set(prepared.inputs.formal_hits["hypothesis_id"].astype(str))
    tier_ids = set(tiers["hypothesis_id"].astype(str))
    checks = {
        "formal_contract": dict(prepared.inputs.formal_contract),
        "formal_hit_count": int(len(prepared.inputs.formal_hits)),
        "formal_replay_max_abs_error": prepared.formal_replay_max_abs_error,
        "formal_replay_allowed_abs_error": float(
            1e-10 + 1e-6 * prepared.inputs.formal_hits[
                "p_interaction"].abs().max()),
        "material_deletion": {
            "fold_count": int(material["deleted_fold"].nunique()),
            "unique_samples": int(len(sample_counts)),
            "samples_deleted_exactly_once": int(sum(
                count == 1 for count in sample_counts.values())),
            "candidate_fold_rows": int(len(material)),
        },
        "locus_clustering": {
            "candidate_units": int(len(loci)),
            "independent_loci": int(loci["locus_id"].nunique()),
        },
        "functional_evidence": {
            "candidate_genes": int(len(candidate_genes)),
            "candidate_unit_tiers": int(len(tiers)),
            "candidate_keys_match_formal_hits": tier_ids == formal_ids,
        },
    }
    expected_samples = len(prepared.analyzed_samples)
    passed = (
        checks["formal_replay_max_abs_error"]
        <= checks["formal_replay_allowed_abs_error"]
        and checks["material_deletion"]["unique_samples"] == expected_samples
        and checks["material_deletion"]["samples_deleted_exactly_once"] == expected_samples
        and checks["functional_evidence"]["candidate_keys_match_formal_hits"]
    )
    if not passed:
        raise FollowupError(
            "independent follow-up consistency audit failed; outputs are incomplete")
    return {
        "audit": "canonical_group_omnib_followup_consistency",
        "status": "PASS",
        **checks,
        "scope": (
            "Candidate-only internal sensitivity and evidence consistency; "
            "not independent replication or causal validation."),
    }


def run_followup(
    results_dir: str | Path,
    *,
    config: str | Path | None = None,
    ranking: str | Path | None = None,
    out_dir: str | Path | None = None,
    material_folds: int = 20,
    environment_col: str | None = None,
    evidence: str | Path | None = None,
    n_jobs: int = 8,
    grm_blas_threads: int = 1,
) -> dict:
    """Build a complete candidate-only stability and evidence dossier."""
    from threadpoolctl import threadpool_limits

    from .evidence import (
        assign_evidence_tiers,
        join_candidate_evidence,
        load_evidence_manifest,
    )

    inputs = load_followup_inputs(results_dir, config=config, ranking=ranking)
    out = (Path(out_dir).expanduser().resolve() if out_dir is not None
           else inputs.results_dir / "followup")
    manifest = load_evidence_manifest(evidence) if evidence is not None else None
    identity, identity_payload = _followup_identity(
        inputs, material_folds=material_folds,
        environment_col=environment_col, evidence_manifest=manifest)
    completed = _open_followup_generation(out, identity, identity_payload)
    if completed is not None:
        return completed
    if inputs.formal_hits.empty:
        summary = {
            "analysis": "canonical_group_omnib_followup",
            "status": "NO_FORMAL_DISCOVERY",
            "trait": inputs.trait,
            "formal_hits": 0,
            "interpretation": (
                "The formal family has no FWER discovery; candidate follow-up "
                "was not started and no candidates were manufactured."),
        }
        _atomic_text(
            out / "followup_summary.json",
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        _complete_followup_generation(out, identity, identity_payload)
        return summary

    with threadpool_limits(limits=int(grm_blas_threads)):
        prepared = prepare_formal_replay(inputs, n_jobs=n_jobs)
    coordinates = candidate_coordinates(prepared)
    loci, links = cluster_formal_units(
        coordinates, _pc1_map(prepared, coordinates))
    material = run_material_deletion(
        prepared, n_folds=material_folds, n_jobs=n_jobs)
    environment, environment_status = run_environment_deletion(
        prepared, environment_col, n_jobs=n_jobs)
    stability = summarize_stability(prepared, material, environment)
    candidate_genes = _candidate_genes(coordinates)
    if manifest is not None:
        evidence_frame, evidence_provenance = join_candidate_evidence(
            candidate_genes, manifest)
    else:
        evidence_frame = pd.DataFrame(columns=[
            "gene_id", "subgenome", "source", "kind", "citation",
            "matched", "supported"])
        evidence_provenance = {
            "evidence_version": 1,
            "manifest": None,
            "sources": [],
            "status": "NO_EVIDENCE_MANIFEST_PROVIDED",
        }
    tiers = assign_evidence_tiers(coordinates, evidence_frame)
    outputs = {
        "formal_reproduction": _write_tsv(
            prepared.replay, out / "formal_hit_reproduction.tsv"),
        "material_deletion": _write_tsv(
            material, out / "material_deletion.tsv"),
        "stability": _write_tsv(
            stability, out / "stability_summary.tsv"),
        "locus_links": _write_tsv(links, out / "locus_links.tsv"),
        "independent_loci": _write_tsv(
            loci, out / "independent_loci.tsv"),
        "candidate_evidence": _write_tsv(
            evidence_frame, out / "candidate_evidence.tsv"),
        "evidence_tiers": _write_tsv(tiers, out / "evidence_tiers.tsv"),
    }
    if not environment.empty:
        outputs["environment_deletion"] = _write_tsv(
            environment, out / "environment_deletion.tsv")
    _atomic_text(
        out / "evidence_provenance.json",
        json.dumps(evidence_provenance, indent=2, ensure_ascii=False) + "\n")
    audit = _followup_audit(
        prepared, material, loci, candidate_genes, tiers)
    audit_path = _atomic_text(
        out / "independent_audit.json",
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "analysis": "canonical_group_omnib_followup",
        "status": "COMPLETED",
        "audit_status": audit["status"],
        "trait": inputs.trait,
        "primary_unit": inputs.hypothesis_unit,
        "formal_hits": int(len(inputs.formal_hits)),
        "independent_loci": int(loci["locus_id"].nunique()),
        "material_folds": int(material["deleted_fold"].nunique()),
        "environment_deletion": environment_status,
        "formal_reproduction_max_abs_error": (
            prepared.formal_replay_max_abs_error),
        "interpretation": (
            "Encoding-robust omnibus pairwise interaction evidence with "
            "candidate-only internal deletion sensitivity. This does not "
            "establish a higher-order, physical, or causal mechanism and is "
            "not independent replication."),
        "outputs": {name: str(path) for name, path in outputs.items()}
        | {"independent_audit": str(audit_path)},
    }
    summary_path = out / "followup_summary.json"
    markdown = (
        f"# Follow-up summary: {inputs.trait}\n\n"
        f"- Primary unit: {inputs.hypothesis_unit}\n"
        f"- Formal discoveries: {len(inputs.formal_hits)}\n"
        f"- Independent reporting regions: {summary['independent_loci']}\n"
        f"- Material folds: {summary['material_folds']}\n"
        f"- Environment deletion: {environment_status}\n"
        f"- Independent consistency audit: PASS\n\n"
        "These are encoding-robust omnibus pairwise interaction candidates. "
        "Deletion analyses are internal sensitivity only; they do not prove "
        "higher-order interaction, physical interaction, causality, or "
        "independent replication.\n")
    markdown_path = _atomic_text(out / "FOLLOWUP_SUMMARY.md", markdown)
    summary["outputs"].update({
        "summary_json": str(summary_path),
        "summary_markdown": str(markdown_path),
        "followup_identity": str(out / "followup_identity.json"),
    })
    _atomic_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _complete_followup_generation(out, identity, identity_payload)
    return summary


def add_followup_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "follow-up",
        help="candidate-only stability and evidence follow-up for group omniB")
    parser.add_argument("results_dir", help="completed canonical interaction result")
    parser.add_argument("--config", default=None, help="explicit canonical config")
    parser.add_argument("--ranking", default=None, help="explicit complete INT ranking")
    parser.add_argument("--out-dir", default=None, help="follow-up output directory")
    parser.add_argument("--material-folds", type=int, default=20)
    parser.add_argument("--environment-col", default=None)
    parser.add_argument("--evidence", default=None, help="evidence manifest YAML")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--grm-blas-threads", type=int, default=1,
        help=("BLAS threads during formal replay; keep at 1 when --n-jobs "
              "is greater than 1 to avoid nested oversubscription"))


def cmd_followup(args) -> int:
    from .evidence import EvidenceError

    try:
        result = run_followup(
            args.results_dir, config=args.config, ranking=args.ranking,
            out_dir=args.out_dir, material_folds=args.material_folds,
            environment_col=args.environment_col, evidence=args.evidence,
            n_jobs=args.n_jobs, grm_blas_threads=args.grm_blas_threads)
    except (FollowupError, EvidenceError) as exc:
        print(f"ERROR: follow-up: {exc}")
        return 1
    print(
        f"[follow-up] {result['status']} "
        f"audit={result.get('audit_status', 'NA')}")
    return 0
