"""Candidate-only stability follow-up for canonical group omniB results."""

from __future__ import annotations

import hashlib
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
    config: dict
    subgenomes: tuple[str, ...]
    trait: str
    hypothesis_unit: str
    ranking: pd.DataFrame
    formal_hits: pd.DataFrame


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
    units = set(frame["hypothesis_unit"].dropna().astype(str).str.lower())
    if units != {hypothesis_unit}:
        raise FollowupError(
            f"ranking hypothesis unit {sorted(units)} does not match {hypothesis_unit!r}")
    primary = pd.to_numeric(frame["primary_sig"], errors="coerce").fillna(0).astype(int)
    formal = frame.loc[primary == 1].copy().reset_index(drop=True)
    if formal["hypothesis_id"].duplicated().any():
        raise FollowupError("formal ranking contains duplicate significant hypothesis IDs")
    return FollowupInputs(
        results_dir=root,
        config_path=config_path,
        ranking_path=ranking_path,
        config=cfg,
        subgenomes=tuple(str(value) for value in ic["subgenomes"]),
        trait=trait,
        hypothesis_unit=hypothesis_unit,
        ranking=frame,
        formal_hits=formal,
    )


def _resolve_config_path(value, config_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    beside = (config_dir / path).resolve()
    return beside if beside.exists() else path.resolve()


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
            return None
        keep = np.asarray([analyzed_lookup[sample] for sample in kept_samples], int)
        y_raw = phenotype.loc[kept_samples].to_numpy(float)
        subset = score_omnib_subset(
            prepared.scores, prepared.family, prepared.expanded,
            keep, y_raw, n_jobs=1)
        frame = extract_primary_scores(subset, prepared)
        frame.insert(0, "deleted_environment", level)
        frame.insert(1, "n", int(keep.size))
        return frame

    with threadpool_limits(limits=1):
        values = Parallel(n_jobs=int(n_jobs), backend="threading") (
            delayed(score)(level) for level in levels)
    frames = [value for value in values if value is not None]
    if not frames:
        return pd.DataFrame(), "NOT_AVAILABLE_LT10_SAMPLES"
    status = "COMPLETED" if len(frames) == len(levels) \
        else "COMPLETED_WITH_UNAVAILABLE_LEVELS"
    return pd.concat(frames, ignore_index=True), status


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
