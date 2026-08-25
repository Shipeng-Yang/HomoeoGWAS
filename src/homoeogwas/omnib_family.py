"""Shared edge/group omniB score matrices for canonical homoeolog families."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .group_family import ExpandedEdgeFamily, MasterGroupFamily, expand_pair_edges

OMNIB_COMPONENT_NAMES = ("minor_burden", "pc1", "kernel_hadamard")


@dataclass
class OmniBFamilyScores:
    """Observed/bootstrap scores retaining the complete predeclared family."""

    edge_p: np.ndarray
    group_p: np.ndarray
    edge_components_obs: np.ndarray
    edge_estimable: np.ndarray
    group_estimable: np.ndarray
    W: np.ndarray
    y: np.ndarray
    covariance_components: dict[str, float]
    group_partial: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    gated_snp: dict = field(default_factory=dict, repr=False)
    feature_cache: dict = field(default_factory=dict, repr=False)
    covariate_block: np.ndarray | None = field(default=None, repr=False)
    covariate_metadata: dict = field(default_factory=dict)


def omnib_components_over_Y(Wh, Yw, Cw, gsx, gsy):
    """Return minor-burden, PC1 and kernel-Hadamard p-values by response."""
    # Delayed import avoids a module cycle while keeping the established
    # nested-F implementation as the single numerical primitive.
    from . import interact as I

    bx, p1x, PX = gsx
    by, p1y, PY = gsy
    components = []
    for ax, ay in ((bx, by), (p1x, p1y), (PX, PY)):
        reduced = np.column_stack([Cw, Wh @ ax, Wh @ ay])
        cross = (ax[:, :, None] * ay[:, None, :]).reshape(ax.shape[0], -1)
        components.append(I._batch_nested_f(Yw, reduced, Wh @ cross))
    return np.vstack(components)


def _edge_design_estimable(gsx, gsy, C: np.ndarray) -> bool:
    """Predeclare target rank using genotype/covariates only."""
    from scipy.linalg import orth

    bx, p1x, PX = gsx
    by, p1y, PY = gsy
    n = C.shape[0]
    for ax, ay in ((bx, by), (p1x, p1y), (PX, PY)):
        reduced = np.column_stack([C, ax, ay])
        Qr = orth(reduced)
        cross = (ax[:, :, None] * ay[:, None, :]).reshape(n, -1)
        residual = cross - Qr @ (Qr.T @ cross) if Qr.size else cross
        Qa = orth(residual)
        if Qa.shape[1] and n - Qr.shape[1] - Qa.shape[1] > 0:
            return True
    return False


def _null_bootstrap_responses(
    V: np.ndarray,
    beta: np.ndarray,
    C: np.ndarray,
    B: int,
    seed: int,
) -> np.ndarray:
    """Draw all null responses from the already-fitted covariance."""
    values, vectors = np.linalg.eigh(0.5 * (V + V.T))
    root = (vectors * np.sqrt(np.clip(values, 1e-12, None))) @ vectors.T
    rng = np.random.default_rng(seed)
    # Preserve the frozen legacy stream exactly: the historical implementation
    # drew one length-n vector per replicate in a Python loop.  NumPy fills a
    # (B, n) array in that same row-major order; drawing (n, B) would assign a
    # different random stream to every replicate despite using the same seed.
    return C @ beta[:, None] + root @ rng.standard_normal((B, V.shape[0])).T


def score_omnib_family(
    subdata: dict,
    family: MasterGroupFamily,
    y_raw: np.ndarray,
    sample_idx: np.ndarray,
    *,
    cap: int = 150,
    n_pc: int = 3,
    transform: str = "INT",
    bootstrap_B: int = 2000,
    bootstrap_seed: int = 2026,
    n_jobs: int = 8,
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
    burden_maf: float = 0.01,
    min_snp: int = 3,
    covariates: dict = None,
) -> tuple[OmniBFamilyScores, ExpandedEdgeFamily]:
    """Score unique edges once, then ACAT-reduce the shared matrix by group."""
    from joblib import Parallel, delayed

    from . import interact as I

    sample_idx = np.asarray(sample_idx, int)
    y_raw = np.asarray(y_raw, float)
    if y_raw.ndim != 1 or y_raw.size != sample_idx.size:
        raise ValueError("y_raw must be one-dimensional and aligned to sample_idx")
    if not np.all(np.isfinite(y_raw)):
        raise ValueError("phenotype contains non-finite values")
    if isinstance(bootstrap_B, bool) or int(bootstrap_B) != bootstrap_B:
        raise ValueError("bootstrap_B must be an integer")
    bootstrap_B = int(bootstrap_B)
    if bootstrap_B < 0:
        raise ValueError("bootstrap_B must be >= 0")
    if isinstance(n_jobs, bool) or int(n_jobs) != n_jobs or int(n_jobs) < 1:
        raise ValueError("n_jobs must be an integer >= 1")
    n_jobs = int(n_jobs)
    missing = [sub for sub in family.subgenomes if sub not in subdata]
    if missing:
        raise ValueError(
            "master family references missing subgenomes: " + ", ".join(missing))

    expanded = expand_pair_edges(family)
    if not expanded.edges:
        raise ValueError("master homoeolog family contains no pair edges")

    # All supplied subgenomes enter one null fit. This preserves the historical
    # pair wrapper's all-kernel behavior and gives canonical AB/AD/BD edges the
    # exact same whitener and bootstrap response columns.
    subs = list(subdata)
    n = sample_idx.size
    kernels = {
        sub: I._build_grm(subdata[sub], sample_idx, grm_method, maf_min)
        for sub in subs
    }
    C = None
    covariate_metadata = {"policy": "none"}
    if covariates:
        C, covariate_metadata = I.build_covariate_block(
            kernels, n, n_pcs=int(covariates.get("n_pcs", 0)),
            extra=covariates.get("extra"))
    C_design = np.ones((n, 1)) if C is None else np.asarray(C, float).reshape(n, -1)
    y = I.rank_int(y_raw) if transform == "INT" else y_raw.astype(float)
    W, V, beta, covariance_components = I.null_lmm_fit(
        kernels, y, C, seed=42)
    Cw = W @ C_design

    feature_rng = np.random.default_rng(bootstrap_seed)
    gated: dict[tuple[str, str], np.ndarray] = {}
    features: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def gated_snps(sub: str, gene: str) -> np.ndarray | None:
        key = (sub, gene)
        if key in gated:
            return gated[key]
        if gene not in subdata[sub].gene_snp:
            return None
        indices = np.asarray(subdata[sub].gene_snp[gene], int)
        means = np.nanmean(
            subdata[sub].X[np.ix_(sample_idx, indices)], axis=0) / 2.0
        gated[key] = indices[np.minimum(means, 1.0 - means) >= burden_maf]
        return gated[key]

    def feature(sub: str, gene: str, indices: np.ndarray):
        key = (sub, gene)
        if key not in features:
            Xg = subdata[sub].X[np.ix_(sample_idx, indices)]
            local = np.arange(indices.size)
            burden = I.block_burden_capped(
                Xg, local, cap, feature_rng, minor=True).reshape(-1, 1)
            pcs = I.gene_pc_scores(Xg, local, cap, feature_rng, n_pc)
            features[key] = burden, pcs[:, :1], pcs
        return features[key]

    edge_estimable = np.zeros(len(expanded.edges), bool)
    for edge_index, edge in enumerate(expanded.edges):
        ix = gated_snps(edge.sub_x, edge.gene_x)
        iy = gated_snps(edge.sub_y, edge.gene_y)
        if ix is None or iy is None or ix.size < min_snp or iy.size < min_snp:
            continue
        fx = feature(edge.sub_x, edge.gene_x, ix)
        fy = feature(edge.sub_y, edge.gene_y, iy)
        edge_estimable[edge_index] = _edge_design_estimable(fx, fy, C_design)

    response_count = bootstrap_B + 1
    responses = np.empty((n, response_count), float)
    responses[:, 0] = y
    if bootstrap_B:
        responses[:, 1:] = _null_bootstrap_responses(
            V, np.asarray(beta, float), C_design, bootstrap_B, bootstrap_seed)
    whitened_responses = W @ responses

    edge_p = np.full((len(expanded.edges), response_count), np.nan)
    edge_components_obs = np.full(
        (len(expanded.edges), len(OMNIB_COMPONENT_NAMES)), np.nan)
    valid_indices = np.flatnonzero(edge_estimable)

    def score_block(bounds):
        lo, hi = bounds
        indices = valid_indices[lo:hi]
        values = np.full((indices.size, response_count), np.nan)
        observed = np.full((indices.size, len(OMNIB_COMPONENT_NAMES)), np.nan)
        for local, edge_index in enumerate(indices):
            edge = expanded.edges[int(edge_index)]
            components = I._omnib_components_over_Y(
                W, whitened_responses, Cw,
                features[(edge.sub_x, edge.gene_x)],
                features[(edge.sub_y, edge.gene_y)])
            values[local] = np.asarray(
                [I.acat(components[:, col]) for col in range(response_count)], float)
            observed[local] = components[:, 0]
        return indices, values, observed

    if valid_indices.size:
        step = max(1, valid_indices.size // (n_jobs * 8))
        blocks = [
            (lo, min(lo + step, valid_indices.size))
            for lo in range(0, valid_indices.size, step)
        ]
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(score_block)(block) for block in blocks)
        for indices, values, observed in results:
            edge_p[indices] = values
            edge_components_obs[indices] = observed

    failed = edge_estimable & ~np.isfinite(edge_p[:, 0])
    if failed.any():
        failed_ids = [expanded.edges[i].edge_id for i in np.flatnonzero(failed)[:5]]
        raise RuntimeError(
            "design-valid edge produced a post-whitening non-finite observed "
            f"omniB score: {', '.join(failed_ids)}")

    group_p = np.full((len(family.group_ids), response_count), np.nan)
    group_partial = np.zeros(len(family.group_ids), bool)
    for group_index, edge_indices in enumerate(expanded.group_edge_indices):
        indices = np.asarray(edge_indices, int)
        valid_count = int(edge_estimable[indices].sum())
        group_partial[group_index] = 0 < valid_count < indices.size
        if indices.size == 1:
            group_p[group_index] = edge_p[indices[0]]
        else:
            for column in range(response_count):
                group_p[group_index, column] = I.acat(edge_p[indices, column])

    return OmniBFamilyScores(
        edge_p=edge_p,
        group_p=group_p,
        edge_components_obs=edge_components_obs,
        edge_estimable=edge_estimable,
        group_estimable=np.isfinite(group_p[:, 0]),
        W=W,
        y=y,
        covariance_components={
            str(name): float(value)
            for name, value in covariance_components.items()
        },
        group_partial=group_partial,
        gated_snp=gated,
        feature_cache=features,
        covariate_block=C,
        covariate_metadata=covariate_metadata,
    ), expanded


def _interact_result(interact_module, **values):
    """Construct against both the frozen and extended compatibility schemas."""
    allowed = interact_module.InteractResult.__dataclass_fields__
    return interact_module.InteractResult(
        **{key: value for key, value in values.items() if key in allowed})


def _callable_group_rows(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    min_snp: int,
) -> np.ndarray:
    """Return master rows whose every declared copy passes the SNP gate."""
    callable_rows = []
    for group_index, genes in enumerate(family.genes):
        if all(
            (sub, gene) in scores.gated_snp
            and scores.gated_snp[(sub, gene)].size >= min_snp
            for sub, gene in zip(family.subgenomes, genes, strict=True)
        ):
            callable_rows.append(group_index)
    return np.asarray(callable_rows, int)


def _tsv_p(interact_module, value) -> str:
    """Serialize a probability without turning non-estimability into a number."""
    if value is None:
        return "NA"
    if hasattr(interact_module, "_tsv_p"):
        try:
            return interact_module._tsv_p(value)
        except (TypeError, ValueError):
            return "NA"
    try:
        return repr(float(value)) if np.isfinite(value) else "NA"
    except (TypeError, ValueError):
        return "NA"


def bootstrap_minp_calibration(
    p_obs: np.ndarray,
    p_null: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict:
    """Calibrate one fixed hypothesis matrix by plus-one single-step min-P."""
    p_obs = np.asarray(p_obs, float)
    p_null = np.asarray(p_null, float)
    if p_obs.ndim != 1:
        raise ValueError("p_obs must be one-dimensional")
    if p_null.ndim != 2 or p_null.shape[0] != p_obs.size:
        raise ValueError(
            "p_obs and p_null must contain the same fixed hypothesis family")
    if not np.isfinite(p_obs).all():
        raise ValueError("observed min-P family contains non-finite statistics")
    B = int(p_null.shape[1])
    if B < 1:
        raise ValueError("bootstrap min-P calibration requires at least one replicate")

    finite_null = np.isfinite(p_null)
    degenerate = ~finite_null.all(axis=0)
    safe = np.where(finite_null, p_null, np.inf)
    null_min = safe.min(axis=0)
    null_min[degenerate] = 0.0
    minp_obs = float(p_obs.min())
    empirical_p = float(
        (1 + int((null_min <= minp_obs).sum())) / (B + 1))
    k = int(np.floor(alpha * (B + 1)))
    threshold = float(np.sort(null_min)[k - 1]) if k >= 1 else None
    rejected_local = (
        np.flatnonzero(p_obs < threshold).astype(int).tolist()
        if threshold is not None else [])
    adjusted = (
        (1 + (null_min[None, :] <= p_obs[:, None]).sum(axis=1))
        / (B + 1)
    ).astype(float)
    rejected = bool(empirical_p <= alpha)
    if rejected != bool(rejected_local):
        raise RuntimeError(
            "bootstrap global decision and single-step rejection set disagree")
    if set(rejected_local) != set(np.flatnonzero(adjusted <= alpha).tolist()):
        raise RuntimeError(
            "bootstrap adjusted-p and threshold rejection sets disagree")
    return {
        "alpha": float(alpha),
        "method": "parametric_bootstrap_minp_plus_one",
        "B": B,
        "empirical_p": empirical_p,
        "threshold": threshold,
        "threshold_comparator": "strict_less_than",
        "rejected": rejected,
        "rejected_local": rejected_local,
        "adjusted_p_local": adjusted,
        "n_degenerate_replicates": int(degenerate.sum()),
        "degenerate_policy": "any_nonfinite_statistic_sets_null_min_to_zero",
    }


def _component_localization(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    component_p = {
        name: (float(values[index]) if np.isfinite(values[index]) else None)
        for index, name in enumerate(OMNIB_COMPONENT_NAMES)
    }
    finite = np.isfinite(values)
    smallest = int(np.nanargmin(values)) if finite.any() else None
    return {
        "component_p": component_p,
        "smallest_component": (
            OMNIB_COMPONENT_NAMES[smallest] if smallest is not None else None),
        "smallest_component_p": (
            float(values[smallest]) if smallest is not None else None),
    }


def _edge_identity_record(
    scores: OmniBFamilyScores,
    expanded: ExpandedEdgeFamily,
    index: int,
) -> dict:
    edge = expanded.edges[index]
    return {
        "hypothesis_id": f"edge:{edge.edge_id}",
        "hypothesis_unit": "edge",
        "edge_id": edge.edge_id,
        "group_ids": list(edge.source_group_ids),
        "direction": edge.direction,
        "sub_x": edge.sub_x,
        "sub_y": edge.sub_y,
        "gene_x": edge.gene_x,
        "gene_y": edge.gene_y,
    } | _component_localization(scores.edge_components_obs[index])


def _group_identity_record(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    index: int,
) -> dict:
    edge_indices = np.asarray(expanded.group_edge_indices[index], int)
    edge_values = scores.edge_p[edge_indices, 0]
    finite_edges = np.isfinite(edge_values)
    driving_index = (
        int(edge_indices[int(np.nanargmin(edge_values))])
        if finite_edges.any() else None)
    driving = (
        _edge_identity_record(scores, expanded, driving_index)
        if driving_index is not None else {})
    return {
        "hypothesis_id": f"group:{family.group_ids[index]}",
        "hypothesis_unit": "group",
        "group_id": family.group_ids[index],
        "genes": {
            sub: gene
            for sub, gene in zip(
                family.subgenomes, family.genes[index], strict=True)
        },
        "ordered_genes": list(family.genes[index]),
        "driving_edge": driving.get("edge_id"),
        "driving_direction": driving.get("direction"),
        "driving_component": driving.get("smallest_component"),
        "driving_component_p": driving.get("smallest_component_p"),
        "edge_localization": [
            {
                "edge_id": expanded.edges[int(edge_index)].edge_id,
                "direction": expanded.edges[int(edge_index)].direction,
                "p_omnib": (
                    float(scores.edge_p[int(edge_index), 0])
                    if np.isfinite(scores.edge_p[int(edge_index), 0]) else None),
            }
            for edge_index in edge_indices
        ],
    }


def _select_primary_family(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    hypothesis_unit: str,
    family_scope: str,
) -> tuple[np.ndarray, list[dict], str, list[str]]:
    edge_records = [
        _edge_identity_record(scores, expanded, index)
        for index in range(len(expanded.edges))
    ]
    group_records = [
        _group_identity_record(scores, family, expanded, index)
        for index in range(len(family.group_ids))
    ]
    if family_scope == "joint":
        return (
            np.vstack([scores.edge_p, scores.group_p]),
            edge_records + group_records,
            "joint",
            ["edge", "group"],
        )
    if hypothesis_unit == "edge":
        return scores.edge_p, edge_records, "edge", ["edge"]
    return scores.group_p, group_records, "group", ["group"]


def _ranking_rows(
    interact_module,
    records: list[dict],
    order: list[int],
    family: MasterGroupFamily,
) -> tuple[list[str], list[list]]:
    header = [
        "rank", "hypothesis_id", "family_id", "hypothesis_unit",
        "group_id", "edge_id", "group_ids", "direction", "sub_x", "sub_y",
        *[f"gene_{sub}" for sub in family.subgenomes],
        "p_interaction", "p_adjusted_bootstrap_minp", "primary_sig",
        "p_unestimable", "driving_edge", "driving_component",
        "driving_component_p",
    ]
    rows = []
    for rank, index in enumerate(order):
        record = records[index]
        genes = record.get("genes", {})
        if record["hypothesis_unit"] == "edge":
            genes = {
                record["sub_x"]: record["gene_x"],
                record["sub_y"]: record["gene_y"],
            }
        group_ids = record.get("group_ids", [])
        rows.append([
            rank,
            record["hypothesis_id"],
            record["family_id"],
            record["hypothesis_unit"],
            record.get("group_id") or "|".join(group_ids) or "NA",
            record.get("edge_id") or "NA",
            "|".join(group_ids) or "NA",
            record.get("direction") or "NA",
            record.get("sub_x") or "NA",
            record.get("sub_y") or "NA",
            *[genes.get(sub, "NA") for sub in family.subgenomes],
            _tsv_p(interact_module, record["p_interaction"]),
            _tsv_p(
                interact_module, record["p_adjusted_bootstrap_minp"]),
            int(record["primary_sig"]),
            int(record["p_unestimable"]),
            record.get("driving_edge") or record.get("edge_id") or "NA",
            record.get("driving_component")
            or record.get("smallest_component") or "NA",
            _tsv_p(
                interact_module,
                record.get("driving_component_p")
                if record["hypothesis_unit"] == "group"
                else record.get("smallest_component_p")),
        ])
    return header, rows


def _family_provenance(
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
) -> dict:
    """Return full ordered hashes for the biological and tested families."""
    group_records = [
        "\t".join((group_id, *genes))
        for group_id, genes in zip(family.group_ids, family.genes, strict=True)
    ]
    edge_records = [
        "\t".join((
            edge.edge_id, edge.sub_x, edge.sub_y, edge.gene_x, edge.gene_y,
            *edge.source_group_ids,
        ))
        for edge in expanded.edges
    ]
    return {
        "group_family_sha256": hashlib.sha256(
            "\x00".join(group_records).encode()).hexdigest(),
        "edge_family_sha256": hashlib.sha256(
            "\x00".join(edge_records).encode()).hexdigest(),
        "n_groups_raw": len(family.group_ids),
        "n_unique_edges": len(expanded.edges),
    }


def run_group_scan_omnib(
    subdata,
    family: MasterGroupFamily,
    y_raw,
    sample_idx,
    *,
    hypothesis_unit="group",
    family_scope="primary_only",
    cap=150,
    n_pc=3,
    transform="INT",
    bootstrap_B=2000,
    bootstrap_seed=2026,
    n_jobs=8,
    grm_method="grm_from_X",
    maf_min=0.01,
    burden_maf=0.01,
    min_snp=3,
    covariates=None,
    full_dump_path=None,
    alpha=0.05,
):
    """Run the formal edge, group or jointly calibrated omniB family."""
    from . import interact as I

    hypothesis_unit = str(hypothesis_unit).lower()
    family_scope = str(family_scope).lower()
    if hypothesis_unit not in {"edge", "group"}:
        raise ValueError("hypothesis_unit must be edge or group")
    if family_scope not in {"primary_only", "joint"}:
        raise ValueError("family_scope must be primary_only or joint")
    if str(transform).upper() != "INT":
        raise ValueError("formal omniB family calibration requires transform='INT'")
    if isinstance(bootstrap_B, bool) or int(bootstrap_B) != bootstrap_B:
        raise ValueError("formal calibration bootstrap_B must be an integer")
    bootstrap_B = int(bootstrap_B)
    if bootstrap_B < 1:
        raise ValueError("formal calibration requires at least one bootstrap replicate")

    scores, expanded = score_omnib_family(
        subdata, family, y_raw, sample_idx, cap=cap, n_pc=n_pc,
        transform="INT", bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed, n_jobs=n_jobs,
        grm_method=grm_method, maf_min=maf_min, burden_maf=burden_maf,
        min_snp=min_snp, covariates=covariates)
    primary_p, identities, family_id, calibrated_layers = _select_primary_family(
        scores, family, expanded, hypothesis_unit, family_scope)
    if primary_p.ndim != 2 or primary_p.shape[1] != bootstrap_B + 1:
        raise RuntimeError(
            "omniB scorer returned an invalid observed/bootstrap matrix shape")
    observed = primary_p[:, 0]
    finite = np.isfinite(observed)
    if not finite.any():
        raise ValueError("declared omniB primary family has no estimable hypotheses")
    finite_indices = np.flatnonzero(finite)

    # This is deliberately the sole calibration call for primary_only and joint.
    calibration = bootstrap_minp_calibration(
        observed[finite], primary_p[finite, 1:], alpha=alpha)
    adjusted = np.full(observed.size, np.nan)
    adjusted[finite] = calibration["adjusted_p_local"]
    rejected_indices = [
        int(finite_indices[int(local)])
        for local in calibration["rejected_local"]
    ]
    rejected_set = set(rejected_indices)

    records = []
    for index, identity in enumerate(identities):
        records.append(identity | {
            "family_id": family_id,
            "p": float(observed[index]) if finite[index] else None,
            "p_interaction": (
                float(observed[index]) if finite[index] else None),
            "p_adjusted_bootstrap_minp": (
                float(adjusted[index]) if np.isfinite(adjusted[index]) else None),
            "primary_sig": index in rejected_set,
            "p_unestimable": not bool(finite[index]),
        })
    order = np.argsort(
        np.where(finite, observed, np.inf), kind="stable").astype(int).tolist()
    sig = [records[index] for index in order if index in rejected_set]
    top = [records[index] for index in order if finite[index]][:5]

    hypothesis_ids = [record["hypothesis_id"] for record in records]
    fwer = {
        key: value
        for key, value in calibration.items()
        if key not in {"rejected_local", "adjusted_p_local"}
    } | {
        "family_id": family_id,
        "family_scope": family_scope,
        "declared_hypothesis_unit": hypothesis_unit,
        "calibrated_layers": calibrated_layers,
        "formal_discovery_layer": True,
        "inferential": True,
        "n_hypotheses": int(observed.size),
        "n_calibrated": int(finite.sum()),
        "n_unestimable": int((~finite).sum()),
        "hypothesis_ids": hypothesis_ids,
        "family_order_sha256": hashlib.sha256(
            "\x00".join(hypothesis_ids).encode()).hexdigest(),
        "observed_p": [
            float(value) if np.isfinite(value) else None for value in observed
        ],
        "adjusted_p": [
            float(value) if np.isfinite(value) else None for value in adjusted
        ],
        "rejected_indices": [
            int(index) for index in order if index in rejected_set
        ],
        "rejected_hypothesis_ids": [
            hypothesis_ids[index] for index in order if index in rejected_set
        ],
        "n_rejected": len(sig),
        "sig": sig,
        "note": (
            "This bootstrap min-P object is the sole calibrated discovery "
            "layer; edge and component decompositions are descriptive unless "
            "included in family_scope=joint."),
    }

    flags = omnib_fwer_consistency_flags({
        "n_sig": len(sig),
        "sig": sig,
        "minp_boot_rejected": bool(calibration["rejected"]),
        "minp_boot_emp": float(calibration["empirical_p"]),
        "minp_boot_threshold": calibration["threshold"],
        "bootstrap_B": bootstrap_B,
        "model_diagnostics": {"bootstrap_fwer": fwer},
    })
    if flags:
        raise RuntimeError(
            "internal omniB FWER serialization inconsistency: " + ", ".join(flags))

    if full_dump_path:
        header, rows = _ranking_rows(I, records, order, family)
        I._write_ranking_tsv(full_dump_path, header, rows)

    G = int(observed.size)
    bonferroni_alpha = float(alpha / G)
    analytic_indices = [
        index for index in order
        if finite[index] and observed[index] < bonferroni_alpha
    ]
    analytic = [records[index] for index in analytic_indices]
    return _interact_result(
        I,
        trait="",
        transform="INT",
        n=int(np.asarray(sample_idx).size),
        G=G,
        pair_acat=float(I.acat(observed[finite])),
        pair_acat_emp=float("nan"),
        min_p=float(observed[finite].min()),
        lambda_gc_obs=float(I.lambda_gc(observed[finite])),
        lambda_gc_perm_median=float("nan"),
        bonferroni_alpha=bonferroni_alpha,
        n_sig=len(sig),
        sig=sig,
        top=top,
        sigma_hat={
            sub: float(scores.covariance_components.get(sub, 0.0))
            for sub in family.subgenomes
        } | {"e": float(scores.covariance_components.get("e", 0.0))},
        weighted=None,
        covariates=scores.covariate_metadata,
        n_planned=G,
        n_valid=int(finite.sum()),
        n_unestimable=int((~finite).sum()),
        statistic="omniB",
        calibration_method="bootstrap",
        bootstrap_B=bootstrap_B,
        bootstrap_seed=int(bootstrap_seed),
        minp_boot_emp=float(calibration["empirical_p"]),
        minp_boot_threshold=calibration["threshold"],
        minp_boot_rejected=bool(calibration["rejected"]),
        component_diagnostics={
            "role": "descriptive_localization",
            "calibrated_layers": calibrated_layers,
        },
        model_diagnostics={
            "bootstrap_fwer": fwer,
            "family_provenance": _family_provenance(family, expanded),
        },
        analytic_screen_n=len(analytic),
        analytic_screen_sig=analytic,
    )


def omnib_fwer_consistency_flags(payload: dict) -> tuple[str, ...]:
    """Return stable audit codes for a serialized formal omniB result record."""
    flags: list[str] = []
    try:
        fwer = payload["model_diagnostics"]["bootstrap_fwer"]
    except (KeyError, TypeError):
        return ("OMNIB_FWER_OBJECT_MISSING",)
    if not isinstance(fwer, dict):
        return ("OMNIB_FWER_OBJECT_MISSING",)

    family_scope = fwer.get("family_scope")
    declared = fwer.get("declared_hypothesis_unit")
    expected_family = "joint" if family_scope == "joint" else declared
    if fwer.get("family_id") != expected_family:
        flags.append("OMNIB_FWER_FAMILY_ID_MISMATCH")

    expected_layers = (
        ["edge", "group"] if family_scope == "joint" else [declared])
    if fwer.get("calibrated_layers") != expected_layers:
        flags.append("OMNIB_FWER_UNCALIBRATED_SECOND_PRIMARY_LAYER")

    n_rejected = fwer.get("n_rejected")
    if payload.get("n_sig") != n_rejected:
        flags.append("OMNIB_FWER_TOPLEVEL_COUNT_MISMATCH")

    fwer_rejected = fwer.get("rejected")
    empirical_p = fwer.get("empirical_p")
    alpha = fwer.get("alpha")
    if not (
        isinstance(fwer_rejected, bool)
        and payload.get("minp_boot_rejected") == fwer_rejected
        and isinstance(n_rejected, int)
        and fwer_rejected == (n_rejected > 0)
        and isinstance(empirical_p, (int, float))
        and isinstance(alpha, (int, float))
        and fwer_rejected == (float(empirical_p) <= float(alpha))
    ):
        flags.append("OMNIB_FWER_TOPLEVEL_DECISION_MISMATCH")
    if payload.get("minp_boot_emp") != empirical_p:
        flags.append("OMNIB_FWER_TOPLEVEL_EMPIRICAL_P_MISMATCH")
    if payload.get("minp_boot_threshold") != fwer.get("threshold"):
        flags.append("OMNIB_FWER_TOPLEVEL_THRESHOLD_MISMATCH")
    if payload.get("bootstrap_B") != fwer.get("B"):
        flags.append("OMNIB_FWER_TOPLEVEL_BOOTSTRAP_B_MISMATCH")

    ids = fwer.get("hypothesis_ids")
    observed = fwer.get("observed_p")
    adjusted = fwer.get("adjusted_p")
    if not (
        isinstance(ids, list)
        and isinstance(observed, list)
        and isinstance(adjusted, list)
        and all(isinstance(value, str) for value in ids)
        and len(ids) == len(observed) == len(adjusted)
        and fwer.get("n_hypotheses") == len(ids)
    ):
        flags.append("OMNIB_FWER_VECTOR_LENGTH_MISMATCH")
        return tuple(dict.fromkeys(flags))

    expected_hash = hashlib.sha256("\x00".join(ids).encode()).hexdigest()
    if fwer.get("family_order_sha256") != expected_hash:
        flags.append("OMNIB_FWER_FAMILY_HASH_MISMATCH")

    missing_adjusted = any(
        value is not None and adjusted[index] is None
        for index, value in enumerate(observed)
    )
    if missing_adjusted:
        flags.append("OMNIB_FWER_ADJUSTED_P_MISSING")

    threshold = fwer.get("threshold")
    adjusted_ids = {
        ids[index]
        for index, value in enumerate(adjusted)
        if value is not None and alpha is not None and value <= alpha
    }
    threshold_ids = {
        ids[index]
        for index, value in enumerate(observed)
        if value is not None and threshold is not None and value < threshold
    }
    rejected_ids_raw = fwer.get("rejected_hypothesis_ids")
    serialized_ids = set(rejected_ids_raw or [])
    fwer_sig = fwer.get("sig")
    sig_ids = {
        record.get("hypothesis_id")
        for record in (fwer_sig or [])
        if isinstance(record, dict)
    }
    if not (
        adjusted_ids == threshold_ids == serialized_ids == sig_ids
        and n_rejected == len(serialized_ids)
    ):
        flags.append("OMNIB_FWER_THRESHOLD_HIT_MISMATCH")

    def ordered_hit_ids(value):
        if not isinstance(value, list):
            return None
        result = []
        for record in value:
            if not isinstance(record, dict) or not isinstance(
                record.get("hypothesis_id"), str
            ):
                return None
            result.append(record["hypothesis_id"])
        return result

    top_sig_ids = ordered_hit_ids(payload.get("sig"))
    fwer_sig_ids = ordered_hit_ids(fwer_sig)
    rejected_ids = (
        rejected_ids_raw
        if isinstance(rejected_ids_raw, list)
        and all(isinstance(value, str) for value in rejected_ids_raw)
        else None
    )
    if not (
        top_sig_ids is not None
        and fwer_sig_ids is not None
        and rejected_ids is not None
        and top_sig_ids == fwer_sig_ids == rejected_ids
    ):
        flags.append("OMNIB_FWER_TOPLEVEL_SIG_MISMATCH")

    rejected_indices = fwer.get("rejected_indices")
    valid_indices = (
        isinstance(rejected_indices, list)
        and all(
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(ids)
            for index in rejected_indices
        )
        and len(set(rejected_indices)) == len(rejected_indices)
    )
    mapped_ids = (
        [ids[index] for index in rejected_indices]
        if valid_indices else None
    )
    if mapped_ids is None or mapped_ids != rejected_ids:
        flags.append("OMNIB_FWER_REJECTED_INDEX_MISMATCH")

    top_sig = payload.get("sig")
    hit_records_ok = isinstance(top_sig, list) and top_sig == fwer_sig
    if (
        valid_indices
        and isinstance(fwer_sig, list)
        and len(fwer_sig) == len(rejected_indices)
    ):
        for full_index, hit in zip(rejected_indices, fwer_sig, strict=True):
            if not isinstance(hit, dict):
                hit_records_ok = False
                break
            try:
                hit_legacy_p = float(hit.get("p"))
                hit_p = float(hit.get("p_interaction"))
                hit_adjusted = float(hit.get("p_adjusted_bootstrap_minp"))
                expected_p = float(observed[full_index])
                expected_adjusted = float(adjusted[full_index])
            except (TypeError, ValueError):
                hit_records_ok = False
                break
            if not (
                hit.get("hypothesis_id") == ids[full_index]
                and np.isfinite(hit_legacy_p)
                and hit_legacy_p == expected_p
                and np.isfinite(hit_p)
                and np.isfinite(expected_p)
                and hit_p == expected_p
                and np.isfinite(hit_adjusted)
                and np.isfinite(expected_adjusted)
                and hit_adjusted == expected_adjusted
                and hit.get("primary_sig") is True
                and hit.get("p_unestimable") is False
            ):
                hit_records_ok = False
                break
    elif valid_indices:
        hit_records_ok = False
    if not hit_records_ok:
        flags.append("OMNIB_FWER_HIT_RECORD_MISMATCH")
    return tuple(dict.fromkeys(flags))


def run_pair_scan_omnib(
    subdata,
    pairs,
    y_raw,
    sample_idx,
    *,
    cap=150,
    n_pc=3,
    transform="INT",
    bootstrap_B=2000,
    bootstrap_seed=2026,
    n_jobs=8,
    pair_subs=None,
    grm_method="compute_grm_maf",
    maf_min=0.01,
    burden_maf=0.01,
    min_snp=3,
    covariates=None,
    tail_thresholds=(1e-2, 1e-3, 1e-4, 1e-5),
    inferential=True,
    full_dump_path=None,
    burden_dump_path=None,
    top_k_burden=5,
    primary_multiplicity="bonferroni",
):
    """Compatibility pair wrapper selecting the shared scorer's edge matrix."""
    from . import interact as I

    primary_multiplicity = str(primary_multiplicity).lower()
    if primary_multiplicity != "bonferroni":
        raise ValueError(
            "Task 3 pairwise omniB compatibility supports only "
            "primary_multiplicity='bonferroni'; bootstrap_minp becomes a "
            "formal discovery layer in Task 4")
    sx, sy = pair_subs
    family = MasterGroupFamily(
        (sx, sy), tuple(f"pair_{i}" for i in range(len(pairs))),
        tuple(tuple(pair) for pair in pairs))
    scores, expanded = score_omnib_family(
        subdata, family, y_raw, sample_idx, cap=cap, n_pc=n_pc,
        transform=transform, bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed, n_jobs=n_jobs,
        grm_method=grm_method, maf_min=maf_min, burden_maf=burden_maf,
        min_snp=min_snp, covariates=covariates)
    selected = _callable_group_rows(scores, family, min_snp)
    if not selected.size:
        raise ValueError(
            "no homoeolog pairs retained: each copy must be present with "
            f">= {min_snp} SNPs passing burden MAF >= {burden_maf}")
    # A pair master row has exactly one edge membership.  Index through the
    # memberships instead of selecting unique primitive edges so duplicate
    # planned rows remain distinct hypotheses in the legacy wrapper family.
    edge_indices = np.asarray([
        expanded.group_edge_indices[int(group_index)][0]
        for group_index in selected
    ], int)
    kept = [family.genes[int(group_index)] for group_index in selected]
    P = scores.edge_p[edge_indices]
    component_obs = scores.edge_components_obs[edge_indices]
    observed = P[:, 0]
    finite = np.isfinite(observed)
    if not finite.any():
        raise ValueError("no estimable pairs (every omniB edge was non-estimable)")
    order_all = np.argsort(
        np.where(finite, observed, np.inf), kind="stable").astype(int).tolist()
    order = [index for index in order_all if finite[index]]
    G = len(kept)
    bonferroni = 0.05 / G

    def hit(index):
        record = I._omnib_component_record(component_obs[index]) if hasattr(
            I, "_omnib_component_record") else {}
        return {
            "pair": kept[index], "p": float(observed[index]),
            "p_adjusted_bonferroni": float(min(observed[index] * G, 1.0)),
        } | record

    analytic_indices = [i for i in order if observed[i] < bonferroni]
    minp_emp = threshold = None
    tail_excess = None
    if bootstrap_B:
        null_family = P[finite, 1:]
        null_min = np.nanmin(null_family, axis=0)
        minp_emp = float(
            (1 + int((null_min <= np.nanmin(observed[finite])).sum()))
            / (bootstrap_B + 1))
        threshold = float(np.quantile(null_min, 0.05))
        tail_excess = {}
        for cutoff in tail_thresholds:
            counts = (null_family < cutoff).sum(axis=0)
            observed_count = int((observed[finite] < cutoff).sum())
            tail_excess[f"n_below_{cutoff:g}"] = {
                "observed": observed_count,
                "null_mean": float(counts.mean()),
                "null_q95": float(np.quantile(counts, 0.95)),
                "empirical_p": float(
                    (1 + int((counts >= observed_count).sum()))
                    / (bootstrap_B + 1)),
            }
    analytic = [hit(i) for i in analytic_indices]
    sig = analytic if inferential else None

    if full_dump_path:
        rows = []
        for rank, index in enumerate(order_all):
            record = (
                I._omnib_component_record(component_obs[index])
                if hasattr(I, "_omnib_component_record") else {})
            components = record.get("component_p", {})
            rows.append([
                rank, kept[index][0], kept[index][1], sx, sy,
                _tsv_p(I, observed[index]),
                _tsv_p(I, components.get("minor_burden", np.nan)),
                _tsv_p(I, components.get("pc1", np.nan)),
                _tsv_p(I, components.get("kernel_hadamard", np.nan)),
                record.get("smallest_component") or "NA",
                int(not finite[index]),
            ])
        I._write_ranking_tsv(
            full_dump_path,
            ["rank", f"gene_{sx}", f"gene_{sy}", "sub_x", "sub_y",
             "p_interaction", "p_minor_burden", "p_pc1",
             "p_kernel_hadamard", "smallest_component", "p_unestimable"], rows)

    if burden_dump_path:
        features = scores.feature_cache
        rows = []
        for pair_rank, index in enumerate(order[:max(int(top_k_burden), 1)]):
            gx, gy = kept[index]
            bx = features[(sx, gx)][0][:, 0]
            by = features[(sy, gy)][0][:, 0]
            if scores.covariate_block is None:
                residual = scores.y - float(np.mean(scores.y))
            else:
                coef, *_ = np.linalg.lstsq(
                    scores.covariate_block, scores.y, rcond=None)
                residual = scores.y - scores.covariate_block @ coef
            for sample_row in range(sample_idx.size):
                rows.append([
                    pair_rank, gx, gy, sx, sy, sample_row,
                    repr(float(bx[sample_row])), repr(float(by[sample_row])),
                    repr(float(scores.y[sample_row])),
                    repr(float(residual[sample_row])),
                ])
        I._write_ranking_tsv(
            burden_dump_path,
            ["pair_rank", "gene_x", "gene_y", "sub_x", "sub_y",
             "sample_row", "minor_burden_x", "minor_burden_y", "phenotype",
             "resid"],
            rows)

    diagnostics = (
        I._omnib_component_summary(component_obs)
        if hasattr(I, "_omnib_component_summary") else None)
    return _interact_result(
        I, trait="", transform=transform, n=int(sample_idx.size), G=G,
        pair_acat=float(I.acat(observed)), pair_acat_emp=float("nan"),
        min_p=float(np.nanmin(observed)),
        lambda_gc_obs=float(I.lambda_gc(observed[finite])),
        lambda_gc_perm_median=float("nan"), bonferroni_alpha=bonferroni,
        n_sig=(len(sig) if sig is not None else None), sig=sig,
        top=[hit(i) for i in order[:5]],
        sigma_hat={sub: float(scores.covariance_components.get(sub, 0.0))
                   for sub in subdata} | {
                       "e": float(scores.covariance_components.get("e", 0.0))},
        weighted=None, covariates=scores.covariate_metadata,
        n_planned=G, n_valid=int(finite.sum()), n_unestimable=int((~finite).sum()),
        statistic="omniB", calibration_method=("bootstrap" if bootstrap_B else "none"),
        bootstrap_B=int(bootstrap_B), bootstrap_seed=int(bootstrap_seed),
        minp_boot_emp=minp_emp, minp_boot_threshold=threshold,
        minp_boot_rejected=None,
        tail_excess=tail_excess, component_diagnostics=diagnostics,
        analytic_screen_n=len(analytic), analytic_screen_sig=analytic,
        model_diagnostics={
            "bootstrap_fwer": {
                "role": "diagnostic_only_task3_legacy_compatibility",
                "formal_discovery_layer": False,
            } if bootstrap_B else None})


def run_clique_scan_omnib(
    subdata,
    groups,
    y_raw,
    sample_idx,
    *,
    cap=150,
    n_pc=3,
    transform="INT",
    bootstrap_B=2000,
    bootstrap_seed=2026,
    n_jobs=8,
    grm_method="compute_grm_maf",
    maf_min=0.01,
    burden_maf=0.01,
    min_snp=3,
    covariates=None,
    tail_thresholds=(1e-2, 1e-3, 1e-4, 1e-5),
    inferential=True,
    full_dump_path=None,
):
    """Compatibility clique wrapper selecting the shared scorer's group matrix."""
    from . import interact as I

    subs = tuple(subdata)
    family = MasterGroupFamily(
        subs, tuple(f"group_{i}" for i in range(len(groups))),
        tuple(tuple(group) for group in groups))
    scores, expanded = score_omnib_family(
        subdata, family, y_raw, sample_idx, cap=cap, n_pc=n_pc,
        transform=transform, bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed, n_jobs=n_jobs,
        grm_method=grm_method, maf_min=maf_min, burden_maf=burden_maf,
        min_snp=min_snp, covariates=covariates)
    selected = _callable_group_rows(scores, family, min_snp)
    if not selected.size:
        raise ValueError(
            "no homoeolog groups retained (need all copies present with >= min_snp SNPs)")
    kept = [family.genes[i] for i in selected]
    P = scores.group_p[selected]
    observed = P[:, 0]
    finite = np.isfinite(observed)
    if not finite.any():
        raise ValueError("no estimable groups (every omniB group was non-estimable)")
    order_all = np.argsort(
        np.where(finite, observed, np.inf), kind="stable").astype(int).tolist()
    order = [index for index in order_all if finite[index]]
    pair_defs = list(__import__("itertools").combinations(subs, 2))
    pair_labels = [f"{left}{right}" for left, right in pair_defs]
    pair_obs = []
    components = []
    for family_index in selected:
        indices = np.asarray(expanded.group_edge_indices[family_index], int)
        pair_obs.append(scores.edge_p[indices, 0])
        components.append(scores.edge_components_obs[indices])
    pair_obs = np.asarray(pair_obs)
    components = np.asarray(components)
    G = len(kept)
    bonferroni = 0.05 / G

    def hit(index):
        pairwise = {}
        for pair_index, label in enumerate(pair_labels):
            record = (
                I._omnib_component_record(components[index, pair_index])
                if hasattr(I, "_omnib_component_record") else {})
            pairwise[label] = {"p_omnib": float(pair_obs[index, pair_index])} | record
        flat = components[index].reshape(-1)
        finite_flat = np.isfinite(flat)
        smallest_pair = smallest_component = None
        smallest_p = None
        if finite_flat.any():
            flat_index = int(np.nanargmin(np.where(finite_flat, flat, np.nan)))
            pair_index, component_index = np.unravel_index(
                flat_index, components[index].shape)
            smallest_pair = pair_labels[pair_index]
            smallest_component = OMNIB_COMPONENT_NAMES[component_index]
            smallest_p = float(flat[flat_index])
        return {
            "pair": kept[index], "p": float(observed[index]),
            "p_adjusted_bonferroni": float(min(observed[index] * G, 1.0)),
            "pairwise": pairwise,
            "smallest_pair": smallest_pair,
            "smallest_component": smallest_component,
            "smallest_component_p": smallest_p,
        }

    sig = [hit(i) for i in order if observed[i] < bonferroni] if inferential else None
    minp_emp = threshold = None
    tail_excess = None
    if bootstrap_B:
        null_family = P[finite, 1:]
        null_min = np.nanmin(null_family, axis=0)
        minp_emp = float(
            (1 + int((null_min <= np.nanmin(observed[finite])).sum()))
            / (bootstrap_B + 1))
        threshold = float(np.quantile(null_min, 0.05))
        tail_excess = {}
        for cutoff in tail_thresholds:
            counts = (null_family < cutoff).sum(axis=0)
            observed_count = int((observed[finite] < cutoff).sum())
            tail_excess[f"n_below_{cutoff:g}"] = {
                "observed": observed_count,
                "null_mean": float(counts.mean()),
                "null_q95": float(np.quantile(counts, 0.95)),
                "empirical_p": float(
                    (1 + int((counts >= observed_count).sum()))
                    / (bootstrap_B + 1)),
            }

    if full_dump_path:
        rows = []
        for rank, index in enumerate(order_all):
            record = hit(index)
            row = [
                rank, *kept[index], _tsv_p(I, observed[index]),
                record["smallest_pair"] or "NA",
                record["smallest_component"] or "NA",
                _tsv_p(I, record["smallest_component_p"]),
                int(not finite[index]),
            ]
            for label in pair_labels:
                pair_record = record["pairwise"][label]
                component_p = pair_record.get("component_p", {})
                row.extend([
                    _tsv_p(I, pair_record["p_omnib"]),
                    _tsv_p(I, component_p.get("minor_burden", np.nan)),
                    _tsv_p(I, component_p.get("pc1", np.nan)),
                    _tsv_p(I, component_p.get("kernel_hadamard", np.nan)),
                ])
            rows.append(row)
        I._write_ranking_tsv(
            full_dump_path,
            ["rank", *[f"gene_{sub}" for sub in subs], "p_interaction",
             "smallest_pair", "smallest_component", "smallest_component_p",
             "p_unestimable"]
            + [value for label in pair_labels for value in (
                f"p_omnib_{label}", f"p_minor_burden_{label}",
                f"p_pc1_{label}", f"p_kernel_hadamard_{label}")],
            rows)

    component_diagnostics = {
        "pair_labels": pair_labels,
        "per_pair": {
            label: (
                I._omnib_component_summary(components[:, pair_index, :])
                if hasattr(I, "_omnib_component_summary") else None)
            for pair_index, label in enumerate(pair_labels)
        },
        "interpretation": (
            "The group p-value is ACAT across pairwise omniB tests. Pair and "
            "component decomposition is descriptive localization, not an "
            "additional rejection family."),
    }

    return _interact_result(
        I, trait="", transform=transform, n=int(sample_idx.size), G=G,
        pair_acat=float(I.acat(observed)), pair_acat_emp=float("nan"),
        min_p=float(np.nanmin(observed)),
        lambda_gc_obs=float(I.lambda_gc(observed[finite])),
        lambda_gc_perm_median=float("nan"), bonferroni_alpha=bonferroni,
        n_sig=(len(sig) if sig is not None else None), sig=sig,
        top=[hit(i) for i in order[:5]],
        sigma_hat={sub: float(scores.covariance_components.get(sub, 0.0))
                   for sub in subdata} | {
                       "e": float(scores.covariance_components.get("e", 0.0))},
        weighted=None, covariates=scores.covariate_metadata,
        n_planned=G, n_valid=int(finite.sum()), n_unestimable=int((~finite).sum()),
        statistic="omniB", calibration_method=("bootstrap" if bootstrap_B else "none"),
        bootstrap_B=int(bootstrap_B), bootstrap_seed=int(bootstrap_seed),
        minp_boot_emp=minp_emp, minp_boot_threshold=threshold,
        tail_excess=tail_excess, component_diagnostics=component_diagnostics)
