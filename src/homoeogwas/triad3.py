"""Experimental exact-three-copy conditional interaction estimator.

Kept separate from the canonical pair-edge group omniB engine: triad3 tests a
conditional third-order burden coefficient and is never used for 4+ copies.
"""

from __future__ import annotations

import itertools

import numpy as np

from .interact import (
    ESTIMABILITY_POLICY,
    TRIAD3_FORMAL_BOOTSTRAP_MIN_B,
    InteractResult,
    SubgenomeData,
    _batch_nested_f,
    _build_grm,
    _gene_coord,
    _neglog10,
    _rank_with_ties,
    _tsv_p,
    _write_ranking_tsv,
    acat,
    block_burden_capped,
    build_covariate_block,
    lambda_gc,
    null_lmm_fit,
    null_replicates,
    rank_int,
    scols_safe,
)


def threeway_design_mask(*args, **kwargs):
    from . import interact
    return interact.threeway_design_mask(*args, **kwargs)


def _bootstrap_minp_calibration(*args, **kwargs):
    from . import interact
    return interact._bootstrap_minp_calibration(*args, **kwargs)



def run_triad3_scan(
    subdata: dict[str, SubgenomeData],
    triads: list,
    y_raw: np.ndarray,
    sample_idx: np.ndarray,
    *,
    cap: int = 150,
    transform: str = "INT",
    bootstrap_B: int = 2000,
    bootstrap_seed: int = 2026,
    n_jobs: int = 8,
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
    burden_maf: float = 0.01,
    min_snp: int = 3,
    covariates: dict = None,
    tail_thresholds: tuple = (1e-2, 1e-3, 1e-4, 1e-5),
    inferential: bool = True,
    full_dump_path: str = None,
) -> InteractResult:
    """Experimental conditional three-way homoeolog-burden scan.

    For exactly three subgenomes, fit the hierarchy-preserving model

    ``y ~ C + A + B + D + A:B + A:D + B:D + A:B:D``

    and test the final coefficient.  Burdens are minor-allele oriented before
    centering/scaling, so the statistic is invariant to PLINK REF/ALT recoding.
    The test is deliberately separate from ``omniB``: it asks whether a genuine
    third-order term remains after every lower-order burden term has been
    conditioned out.

    Bonferroni is a descriptive analytic screen only.  For an inferential run,
    the mandatory parametric-bootstrap min-P procedure is the sole formal
    experiment-wide discovery layer.  This is an experimental statistic, not
    a claim of a physical three-protein complex.
    """
    from joblib import Parallel, delayed

    subs = list(subdata)
    if len(subs) != 3:
        raise ValueError(
            f"triad3 requires exactly 3 subgenomes; got {len(subs)} ({subs})")
    if isinstance(bootstrap_B, bool) or int(bootstrap_B) != bootstrap_B:
        raise ValueError("triad3 bootstrap_B must be an integer")
    bootstrap_B = int(bootstrap_B)
    if bootstrap_B < 0:
        raise ValueError("triad3 bootstrap_B must be >= 0")
    if inferential and bootstrap_B < TRIAD3_FORMAL_BOOTSTRAP_MIN_B:
        raise ValueError(
            "formal inferential triad3 requires bootstrap_B >= "
            f"{TRIAD3_FORMAL_BOOTSTRAP_MIN_B}; use inferential=False for "
            "B>=19 QA runs")
    rng = np.random.default_rng(bootstrap_seed)
    n_t = int(sample_idx.size)
    kernels = {
        s: _build_grm(subdata[s], sample_idx, grm_method, maf_min)
        for s in subs
    }
    C = None
    cov_meta = dict(policy="none")
    if covariates:
        C, cov_meta = build_covariate_block(
            kernels, n_t, n_pcs=int(covariates.get("n_pcs", 0)),
            extra=covariates.get("extra"))
    y = rank_int(y_raw) if transform == "INT" else np.asarray(y_raw, float)
    W, V, beta, cv = null_lmm_fit(kernels, y, C, seed=42)

    def _gene_snp_gated(sub, gid):
        idx = np.asarray(subdata[sub].gene_snp[gid], int)
        mu = np.nanmean(
            subdata[sub].X[np.ix_(sample_idx, idx)], axis=0) / 2.0
        return idx[np.minimum(mu, 1.0 - mu) >= burden_maf]

    cols = {s: [] for s in subs}
    nsnp = {s: [] for s in subs}
    gated_by_gene = {}
    burden_by_gene = {}
    kept = []
    for triad in triads:
        gmap = dict(zip(subs, triad, strict=True))
        if not all(gmap[s] in subdata[s].gene_snp for s in subs):
            continue
        gated = {s: _gene_snp_gated(s, gmap[s]) for s in subs}
        if any(gated[s].size < min_snp for s in subs):
            continue
        for s in subs:
            key = (s, gmap[s])
            if key not in burden_by_gene:
                Xg = subdata[s].X[np.ix_(sample_idx, gated[s])]
                burden_by_gene[key] = block_burden_capped(
                    Xg, np.arange(gated[s].size), cap, rng, minor=True)
            cols[s].append(burden_by_gene[key])
            nsnp[s].append(int(gated[s].size))
            gated_by_gene[key] = gated[s]
        kept.append(tuple(triad))
    G = len(kept)
    if G < 1:
        raise ValueError(
            "no triad retained (need all three copies with >= min_snp "
            "minor-allele-frequency-gated SNPs)")

    burden = {s: scols_safe(np.column_stack(cols[s])) for s in subs}
    BA, BB, BD = (burden[s] for s in subs)
    design_mask, exclusions, design_diag = threeway_design_mask(
        BA, BB, BD, C=C)
    if not design_mask.any():
        raise ValueError(
            "no triad has an estimable A×B×D term after conditioning on all "
            "main effects and pairwise interactions")

    ncol = 1 + max(int(bootstrap_B), 0)
    Yall = np.empty((n_t, ncol))
    Yall[:, 0] = y
    if bootstrap_B and bootstrap_B > 0:
        ystars, _W2, _cv2 = null_replicates(
            kernels, y, C=C, B=bootstrap_B, method="bootstrap",
            seed=int(bootstrap_seed), null_fit=(W, V, beta, cv))
        Yall[:, 1:] = np.column_stack(
            [np.asarray(value, float) for value in ystars])
    Yw = W @ Yall
    Cw = (W @ (np.ones(n_t) if C is None else C)).reshape(n_t, -1)

    def _block(lo_hi):
        lo, hi = lo_hi
        out = np.full((hi - lo, ncol), np.nan)
        a, b, d = BA[:, lo:hi], BB[:, lo:hi], BD[:, lo:hi]
        # Whiten whole blocks with BLAS matrix-matrix products.  Whitening
        # seven vectors separately for every triad makes a genome-wide wheat
        # scan orders of magnitude slower.
        Aw, Bw, Dw = W @ a, W @ b, W @ d
        ABw, ADw, BDw = W @ (a * b), W @ (a * d), W @ (b * d)
        ABDw = W @ (a * b * d)
        for j, gi in enumerate(range(lo, hi)):
            if not design_mask[gi]:
                continue
            nuisance = np.column_stack([
                Cw, Aw[:, j], Bw[:, j], Dw[:, j],
                ABw[:, j], ADw[:, j], BDw[:, j],
            ])
            out[j] = _batch_nested_f(
                Yw, nuisance, ABDw[:, j].reshape(-1, 1))
        return lo, out

    step = max(1, G // (max(n_jobs, 1) * 8))
    blocks = [(i, min(i + step, G)) for i in range(0, G, step)]
    block_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_block)(block) for block in blocks)
    P = np.full((G, ncol), np.nan)
    for lo, arr in block_results:
        P[lo:lo + arr.shape[0]] = arr

    late_fail = design_mask & ~np.isfinite(P[:, 0])
    if late_fail.any():
        raise ValueError(
            f"{int(late_fail.sum())} raw-design-estimable triad3 tests lost "
            "their statistic after whitening")
    p_obs = P[:, 0]
    finite = np.isfinite(p_obs)
    minp_obs = float(np.nanmin(p_obs))
    bonf = 0.05 / G
    order = [
        int(i) for i in np.argsort(np.where(finite, p_obs, np.inf))
        if finite[i]
    ]

    residual_ratio = design_diag["target_residual_ratio"]
    target_sd = design_diag["target_sd"]
    information_max = design_diag["target_information_max_fraction"]
    information_top10 = design_diag["target_information_top10_fraction"]
    information_effective_n = design_diag["target_information_effective_n"]

    minp_adjusted = np.full(G, np.nan)

    def _hit(i):
        return {
            "triad": kept[i],
            "p": float(p_obs[i]),
            "p_adjusted_bonferroni": float(min(p_obs[i] * G, 1.0)),
            "p_adjusted_bootstrap_minp": (
                float(minp_adjusted[i])
                if np.isfinite(minp_adjusted[i]) else None),
            "target_residual_ratio": (
                float(residual_ratio[i]) if np.isfinite(residual_ratio[i])
                else None),
            "target_sd": (
                float(target_sd[i]) if np.isfinite(target_sd[i]) else None),
            "target_information_max_fraction": (
                float(information_max[i])
                if np.isfinite(information_max[i]) else None),
            "target_information_top10_fraction": (
                float(information_top10[i])
                if np.isfinite(information_top10[i]) else None),
            "target_information_effective_n": (
                float(information_effective_n[i])
                if np.isfinite(information_effective_n[i]) else None),
        }

    analytic_indices = [i for i in order if p_obs[i] < bonf]
    analytic_sig = [_hit(i) for i in analytic_indices]

    minp_boot_emp = minp_boot_threshold = None
    minp_boot_rejected = None
    bootstrap_fwer = None
    tail_excess = None
    formal_indices = None
    if bootstrap_B and bootstrap_B > 0:
        family_indices = np.flatnonzero(finite)
        cal = _bootstrap_minp_calibration(
            p_obs[finite], P[finite, 1:], alpha=0.05)
        minp_adjusted[finite] = cal["adjusted_p_local"]
        minp_boot_emp = cal["empirical_p"]
        minp_boot_threshold = cal["threshold"]
        minp_boot_rejected = cal["rejected"] if inferential else None
        formal_indices = (
            [int(family_indices[i]) for i in cal["rejected_local"]]
            if inferential else None)
        bootstrap_fwer = {
            key: value for key, value in cal.items()
            if key not in {"rejected_local", "adjusted_p_local"}
        } | {
            "inferential": bool(inferential),
            "rejected": minp_boot_rejected,
            "n_rejected": (
                len(formal_indices) if formal_indices is not None else None),
            "sig": (
                [_hit(i) for i in formal_indices]
                if formal_indices is not None else None),
            "note": (
                "This bootstrap min-P object is the sole calibrated discovery "
                "layer. Bonferroni fields are a descriptive analytic screen."),
        }
        tail_excess = {}
        for threshold in tail_thresholds:
            obs_ct = int((p_obs[finite] < threshold).sum())
            null_family = P[finite, 1:]
            null_ct = (null_family < threshold).sum(0).astype(float)
            degenerate = ~np.isfinite(null_family).all(axis=0)
            null_ct[degenerate] = float(finite.sum())
            tail_excess[f"n_below_{threshold:g}"] = {
                "observed": obs_ct,
                "null_mean": float(null_ct.mean()),
                "null_q95": float(np.quantile(null_ct, 0.95)),
                "empirical_p": float(
                    (1 + int((null_ct >= obs_ct).sum()))
                    / (null_ct.size + 1)),
                "role": "descriptive_tail_diagnostic_not_a_discovery_test",
            }

    sig = (
        [_hit(i) for i in formal_indices]
        if formal_indices is not None else None)
    n_sig = len(sig) if sig is not None else None
    top = [_hit(i) for i in order[:5]]

    if full_dump_path:
        ns = {s: np.asarray(nsnp[s], int) for s in subs}
        coords = {
            s: [
                _gene_coord(
                    subdata[s],
                    gated_by_gene[
                        (s, dict(zip(subs, triad, strict=True))[s])])
                for triad in kept
            ]
            for s in subs
        }
        formal_set = set(formal_indices or [])
        analytic_set = set(analytic_indices)
        sort_key = np.where(finite, p_obs, np.inf)
        _, rank_of, tie_of = _rank_with_ties(sort_key, kept)
        rows = []
        for i in np.argsort(sort_key, kind="stable"):
            rows.append([
                int(rank_of[i]), *kept[i], _tsv_p(p_obs[i]),
                _tsv_p(_neglog10(np.array([p_obs[i]]))[0]),
                _tsv_p(residual_ratio[i]), _tsv_p(target_sd[i]),
                _tsv_p(information_max[i]),
                _tsv_p(information_top10[i]),
                _tsv_p(information_effective_n[i]),
                *[int(ns[s][i]) for s in subs],
                int(sum(ns[s][i] for s in subs)),
                int(tie_of[i]), int(not finite[i]),
                (int(i in formal_set) if inferential else "NA"),
                int(i in analytic_set),
                *[value for s in subs for value in coords[s][i]],
            ])
        header = (
            ["rank", *[f"gene_{s}" for s in subs],
             "p_threeway", "neglog10p", "target_residual_ratio",
             "target_sd", "target_information_max_fraction",
             "target_information_top10_fraction",
             "target_information_effective_n",
             *[f"n_snp_{s}" for s in subs],
             "n_snp_triad", "tie_group", "p_unestimable",
             "primary_sig", "analytic_screen_sig"]
            + [value for s in subs
               for value in (f"chrom_{s}", f"pos_{s}")]
        )
        _write_ranking_tsv(full_dump_path, header, rows)

    main_terms = " + ".join(subs)
    pair_terms = " + ".join(":".join(pair)
                            for pair in itertools.combinations(subs, 2))
    tested_term = ":".join(subs)
    return InteractResult(
        trait="", transform=transform, n=n_t, G=G,
        pair_acat=float(acat(p_obs)), pair_acat_emp=float("nan"),
        min_p=minp_obs, lambda_gc_obs=float(lambda_gc(p_obs[finite])),
        lambda_gc_perm_median=float("nan"), bonferroni_alpha=float(bonf),
        n_sig=n_sig, sig=sig, top=top,
        sigma_hat={
            s: float(cv.get(s, 0.0)) for s in subs
        } | {"e": float(cv.get("e", 0.0))},
        weighted=None, covariates=cov_meta,
        n_planned=G, n_valid=int(finite.sum()),
        n_unestimable=int((~design_mask).sum()),
        estimability={
            **ESTIMABILITY_POLICY,
            "decided_on": "raw_hierarchical_design",
            "n_planned": G,
            "n_valid": int(design_mask.sum()),
            "n_unestimable": int((~design_mask).sum()),
            "n_late_fail": int(late_fail.sum()),
            "exclusions": [
                {"triad": kept[i], "reason": reason}
                for i, reason in sorted(exclusions.items())
            ],
        },
        statistic="triad3",
        calibration_method=("bootstrap" if bootstrap_B else "none"),
        bootstrap_B=int(bootstrap_B), bootstrap_seed=int(bootstrap_seed),
        minp_boot_emp=minp_boot_emp,
        minp_boot_threshold=minp_boot_threshold,
        minp_boot_rejected=minp_boot_rejected,
        tail_excess=tail_excess,
        analytic_screen_n=len(analytic_sig),
        analytic_screen_sig=analytic_sig,
        model_diagnostics={
            "formula": (
                f"y ~ C + {main_terms} + {pair_terms} + {tested_term}"),
            "tested_term": tested_term,
            "encoding": "minor_allele_burden_centered_scaled",
            "target_residual_ratio_min": (
                float(np.nanmin(residual_ratio[design_mask]))),
            "target_residual_ratio_median": (
                float(np.nanmedian(residual_ratio[design_mask]))),
            "target_information_max_fraction_median": (
                float(np.nanmedian(information_max[design_mask]))),
            "target_information_effective_n_median": (
                float(np.nanmedian(information_effective_n[design_mask]))),
            "bootstrap_fwer": bootstrap_fwer,
            "analytic_screen": {
                "method": "bonferroni",
                "alpha": 0.05,
                "per_test_alpha": float(bonf),
                "n_screened": len(analytic_sig),
                "sig": analytic_sig,
                "role": "descriptive_candidate_screen_not_formal_discovery",
            },
            "inference_plan": {
                "formal_discovery": "parametric_bootstrap_minp_plus_one",
                "analytic_bonferroni": "descriptive_screen_only",
                "inferential": bool(inferential),
            },
            "interpretation": (
                "A significant term is statistical third-order non-additivity "
                "conditional on every lower-order burden term; it is not evidence "
                "by itself for a physical three-protein complex."),
        },
    )
