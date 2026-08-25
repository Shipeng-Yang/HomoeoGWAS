"""Shared edge/group omniB score matrices for canonical homoeolog families."""

from __future__ import annotations

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
    return C @ beta[:, None] + root @ rng.standard_normal((V.shape[0], B))


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


def _bootstrap_minp(interact_module, p_obs, p_null, *, alpha=0.05):
    """Use the engine calibration helper, with an exact compatibility fallback."""
    if hasattr(interact_module, "_bootstrap_minp_calibration"):
        return interact_module._bootstrap_minp_calibration(
            p_obs, p_null, alpha=alpha)

    p_obs = np.asarray(p_obs, float)
    p_null = np.asarray(p_null, float)
    if p_null.ndim != 2 or p_null.shape[0] != p_obs.size:
        raise ValueError(
            "observed and bootstrap p-values must contain the same fixed family")
    if not np.isfinite(p_obs).all() or p_null.shape[1] < 1:
        raise ValueError("bootstrap min-P needs finite observations and >= 1 replicate")
    finite_null = np.isfinite(p_null)
    degenerate = ~finite_null.all(axis=0)
    null_min = np.where(finite_null, p_null, np.inf).min(axis=0)
    null_min[degenerate] = 0.0
    B = p_null.shape[1]
    empirical_p = float((1 + (null_min <= p_obs.min()).sum()) / (B + 1))
    k = int(np.floor(alpha * (B + 1)))
    threshold = float(np.sort(null_min)[k - 1]) if k >= 1 else None
    rejected_local = (
        np.flatnonzero(p_obs < threshold).astype(int).tolist()
        if threshold is not None else [])
    rejected = bool(empirical_p <= alpha)
    if rejected != bool(rejected_local):
        raise RuntimeError(
            "bootstrap global decision and single-step rejection set disagree")
    return {
        "alpha": float(alpha),
        "method": "parametric_bootstrap_minp_plus_one",
        "B": int(B),
        "empirical_p": empirical_p,
        "threshold": threshold,
        "threshold_comparator": "strict_less_than",
        "rejected": rejected,
        "rejected_local": rejected_local,
        "adjusted_p_local": (
            (1 + (null_min[None, :] <= p_obs[:, None]).sum(axis=1))
            / (B + 1)
        ).astype(float),
        "n_degenerate_replicates": int(degenerate.sum()),
        "degenerate_policy": "any_nonfinite_statistic_sets_null_min_to_zero",
    }


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
    selected = np.flatnonzero(scores.edge_estimable)
    if not selected.size:
        raise ValueError(
            "no homoeolog pairs retained: each copy must be present with "
            f">= {min_snp} SNPs passing burden MAF >= {burden_maf}")
    kept = [
        (expanded.edges[i].gene_x, expanded.edges[i].gene_y)
        for i in selected
    ]
    P = scores.edge_p[selected]
    component_obs = scores.edge_components_obs[selected]
    observed = P[:, 0]
    finite = np.isfinite(observed)
    order = [
        int(i) for i in np.argsort(np.where(finite, observed, np.inf))
        if finite[i]
    ]
    G = len(kept)
    bonferroni = 0.05 / G
    primary_multiplicity = str(primary_multiplicity).lower()
    if primary_multiplicity not in {"bonferroni", "bootstrap_minp"}:
        raise ValueError(
            "pairwise omniB primary_multiplicity must be bonferroni or bootstrap_minp")
    adjusted = np.full(G, np.nan)

    def hit(index):
        record = I._omnib_component_record(component_obs[index]) if hasattr(
            I, "_omnib_component_record") else {}
        return {
            "pair": kept[index], "p": float(observed[index]),
            "p_adjusted_bonferroni": float(min(observed[index] * G, 1.0)),
            "p_adjusted_bootstrap_minp": (
                float(adjusted[index]) if np.isfinite(adjusted[index]) else None),
        } | record

    analytic_indices = [i for i in order if observed[i] < bonferroni]
    minp_emp = threshold = None
    minp_rejected = None
    formal_indices = None
    bootstrap_fwer = None
    tail_excess = None
    if bootstrap_B:
        family_indices = np.flatnonzero(finite)
        calibration = _bootstrap_minp(
            I, observed[finite], P[finite, 1:], alpha=0.05)
        adjusted[finite] = calibration["adjusted_p_local"]
        minp_emp = calibration["empirical_p"]
        threshold = calibration["threshold"]
        if primary_multiplicity == "bootstrap_minp":
            minp_rejected = calibration["rejected"] if inferential else None
            formal_indices = (
                [int(family_indices[i]) for i in calibration["rejected_local"]]
                if inferential else None)
        bootstrap_fwer = {
            key: value for key, value in calibration.items()
            if key not in {"adjusted_p_local", "rejected_local"}
        } | {
            "inferential": bool(
                inferential and primary_multiplicity == "bootstrap_minp"),
            "rejected": minp_rejected,
            "n_rejected": (
                len(formal_indices) if formal_indices is not None else None),
        }
        tail_excess = {}
        null_family = P[finite, 1:]
        for cutoff in tail_thresholds:
            counts = (null_family < cutoff).sum(axis=0).astype(float)
            counts[~np.isfinite(null_family).all(axis=0)] = float(finite.sum())
            observed_count = int((observed[finite] < cutoff).sum())
            tail_excess[f"n_below_{cutoff:g}"] = {
                "observed": observed_count,
                "null_mean": float(counts.mean()),
                "null_q95": float(np.quantile(counts, 0.95)),
                "empirical_p": float(
                    (1 + int((counts >= observed_count).sum()))
                    / (bootstrap_B + 1)),
                "role": "descriptive_tail_diagnostic_not_a_discovery_test",
            }
    elif primary_multiplicity == "bootstrap_minp" and inferential:
        raise ValueError(
            "formal pairwise omniB bootstrap_minp requires bootstrap_B >= 1")

    if primary_multiplicity == "bonferroni" and inferential:
        formal_indices = analytic_indices
    analytic = [hit(i) for i in analytic_indices]
    sig = ([hit(i) for i in formal_indices]
           if formal_indices is not None else None)
    if bootstrap_fwer is not None:
        bootstrap_fwer["sig"] = sig if primary_multiplicity == "bootstrap_minp" else None
        bootstrap_fwer["note"] = (
            "This bootstrap min-P object is the sole calibrated discovery layer. "
            "Bonferroni fields are a descriptive analytic screen."
            if primary_multiplicity == "bootstrap_minp" else
            "Bootstrap min-P is diagnostic because Bonferroni was the "
            "predeclared discovery layer.")

    if full_dump_path:
        rows = []
        for rank, index in enumerate(order):
            record = hit(index)
            components = record.get("component_p", {})
            rows.append([
                rank, kept[index][0], kept[index][1], sx, sy,
                repr(float(observed[index])),
                (repr(float(adjusted[index]))
                 if np.isfinite(adjusted[index]) else "NA"),
                repr(float(components.get("minor_burden", np.nan))),
                repr(float(components.get("pc1", np.nan))),
                repr(float(components.get("kernel_hadamard", np.nan))),
                record.get("smallest_component") or "NA",
            ])
        I._write_ranking_tsv(
            full_dump_path,
            ["rank", f"gene_{sx}", f"gene_{sy}", "sub_x", "sub_y",
             "p_interaction", "p_adjusted_bootstrap_minp",
             "p_minor_burden", "p_pc1",
             "p_kernel_hadamard", "smallest_component"], rows)

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
        minp_boot_rejected=minp_rejected,
        tail_excess=tail_excess, component_diagnostics=diagnostics,
        analytic_screen_n=len(analytic), analytic_screen_sig=analytic,
        model_diagnostics={"bootstrap_fwer": bootstrap_fwer})


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
    selected = np.flatnonzero(scores.group_estimable & ~scores.group_partial)
    if not selected.size:
        raise ValueError(
            "no homoeolog groups retained (need all copies present with >= min_snp SNPs)")
    kept = [family.genes[i] for i in selected]
    P = scores.group_p[selected]
    observed = P[:, 0]
    finite = np.isfinite(observed)
    order = [
        int(i) for i in np.argsort(np.where(finite, observed, np.inf))
        if finite[i]
    ]
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
        calibration = _bootstrap_minp(
            I, observed[finite], P[finite, 1:], alpha=0.05)
        minp_emp = calibration["empirical_p"]
        threshold = calibration["threshold"]
        tail_excess = {}
        null_family = P[finite, 1:]
        for cutoff in tail_thresholds:
            counts = (null_family < cutoff).sum(axis=0).astype(float)
            counts[~np.isfinite(null_family).all(axis=0)] = float(finite.sum())
            observed_count = int((observed[finite] < cutoff).sum())
            tail_excess[f"n_below_{cutoff:g}"] = {
                "observed": observed_count,
                "null_mean": float(counts.mean()),
                "null_q95": float(np.quantile(counts, 0.95)),
                "empirical_p": float(
                    (1 + int((counts >= observed_count).sum()))
                    / (bootstrap_B + 1)),
                "role": "descriptive_tail_diagnostic_not_a_discovery_test",
            }

    if full_dump_path:
        rows = []
        for rank, index in enumerate(order):
            record = hit(index)
            row = [
                rank, *kept[index], repr(float(observed[index])),
                record["smallest_pair"] or "NA",
                record["smallest_component"] or "NA",
                (repr(float(record["smallest_component_p"]))
                 if record["smallest_component_p"] is not None else "NA"),
            ]
            for label in pair_labels:
                pair_record = record["pairwise"][label]
                component_p = pair_record.get("component_p", {})
                row.extend([
                    repr(float(pair_record["p_omnib"])),
                    repr(float(component_p.get("minor_burden", np.nan))),
                    repr(float(component_p.get("pc1", np.nan))),
                    repr(float(component_p.get("kernel_hadamard", np.nan))),
                ])
            rows.append(row)
        I._write_ranking_tsv(
            full_dump_path,
            ["rank", *[f"gene_{sub}" for sub in subs], "p_interaction",
             "smallest_pair", "smallest_component", "smallest_component_p"]
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
