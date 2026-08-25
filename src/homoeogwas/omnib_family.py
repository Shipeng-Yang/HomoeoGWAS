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
