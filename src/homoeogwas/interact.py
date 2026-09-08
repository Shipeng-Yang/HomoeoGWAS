"""Gene-resolution homoeolog interaction scans.

The production default, omniB, combines three encoding-robust conditional
interaction tests (minor-burden, PC1 and kernel-Hadamard) and calibrates the
experiment-wide minimum with a kinship-preserving parametric bootstrap. The
experimental ``triad3`` statistic instead tests the hierarchy-preserving
``A:B:D`` minor-burden coefficient conditional on all main and pairwise terms.
legacy burden path fits, for each homoeolog pair, the whitened GLS

    y_w ~ 1 + b_X + b_Y + b_X * b_Y

where b_* are gene burdens (capped block-mean of standardized dosages) and the whitener
comes from a subgenome-stratified null LMM (multi-kernel REML over {K_sub}); it then tests
the interaction coefficient and aggregates per-pair p-values with ACAT. INT is the primary
transform; ``raw`` is a sensitivity transform. The legacy empirical calibration uses
Freedman-Lane permutation. Optional y-independent extensions include frozen pair weights and
multi-trait ACAT over a predeclared trait set.

Config (YAML)::

    interact:
      subgenomes: [A, D]               # 2 (disomic) or 3 (hexaploid)
      statistic: omniB
      primary_transform: INT
      genotype: {A: <plink_prefix>, D: <plink_prefix>}
      snp_to_gene: {A: <npz>, D: <npz>}    # gene_ids + snp_idx into that subgenome's X
      pairs: <tsv>                          # columns gene_<S> per subgenome (e.g. gene_A,gene_D)
      phenotype: <tsv>
      sample_col: sample
      trait: <name>                         # single-trait; OR predeclare a frozen set below
      # multi_trait: [t1, t2, t3]           # pairwise-only pleiotropy: ACAT across a FROZEN trait set
      burden: {cap: 150, min_snp: 3}
      calibration: {method: bootstrap, B: 2000, seed: 2026}
    outputs: {out_dir: <dir>, full_ranking: true}
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import omnib_family as _family_score
from .formal_provenance import FormalLaunchError, verify_formal_launch
from .group_family import (
    ExpandedEdgeFamily,
    MasterGroupFamily,
    expand_pair_edges,
    load_master_group_family,
)
from .interaction_config import normalize_interact_config


def _acat_tan_terms(p: np.ndarray) -> np.ndarray:
    """Cauchy terms tan((0.5-p)*pi) with precision-safe branches at the extremes (Liu & Xie 2020).

    For p < 1e-15, ``(0.5 - p)`` rounds to exactly 0.5 in float64 and ``tan`` saturates, so we use
    the cotangent identity tan((0.5-p)*pi) = cot(p*pi) ~ 1/(p*pi), with a symmetric branch near 1.
    In the normal range the plain ``tan`` term is returned."""
    small = p < 1e-15
    big = p > 1.0 - 1e-15
    t = np.tan((0.5 - p) * np.pi)
    t = np.where(small, 1.0 / (p * np.pi), t)
    t = np.where(big, -1.0 / ((1.0 - p) * np.pi), t)
    return t


def _acat_p_from_t(tbar: float) -> float:
    """Invert the mean Cauchy statistic to a p-value, with |t|-large branches to avoid arctan
    saturation when one extreme p dominates the combination."""
    if tbar > 1e15:
        return float(1.0 / (tbar * np.pi))
    if tbar < -1e15:
        return float(1.0 - 1.0 / (abs(tbar) * np.pi))
    return float(0.5 - np.arctan(tbar) / np.pi)


def acat(pvals: np.ndarray) -> float:
    """Aggregated Cauchy association test combination of p-values (equal weights)."""
    p = np.asarray(pvals, float)
    p = np.clip(p[np.isfinite(p)], 1e-300, 1.0 - 1e-16)
    if p.size == 0:
        return float("nan")
    return _acat_p_from_t(float(np.mean(_acat_tan_terms(p))))


def acat_weighted(pvals: np.ndarray, weights: np.ndarray) -> float:
    """Weighted ACAT combination. Weights must be y-independent (e.g. zero-shot DL variant priors
    or homoeolog expression bias, frozen before the association test) to stay calibrated."""
    p = np.asarray(pvals, float)
    w = np.asarray(weights, float)
    m = np.isfinite(p) & np.isfinite(w) & (w > 0)
    p = np.clip(p[m], 1e-300, 1.0 - 1e-16)
    w = w[m]
    if p.size == 0:
        return float("nan")
    w = w / w.sum()
    return _acat_p_from_t(float(np.sum(w * _acat_tan_terms(p))))


def lambda_gc(pvals: np.ndarray) -> float:
    """Genomic-control lambda from p-values via the median chi-square(1) statistic."""
    p = np.asarray(pvals, float)
    p = np.clip(p[np.isfinite(p)], 1e-300, 1.0)
    if p.size == 0:
        return float("nan")
    chi2 = stats.norm.isf(p / 2.0) ** 2
    return float(np.median(chi2) / 0.4549364)


def rank_int(y: np.ndarray) -> np.ndarray:
    """Rank-based inverse-normal transform (Blom)."""
    y = np.asarray(y, float)
    n = y.size
    r = stats.rankdata(y, method="average")
    return stats.norm.ppf((r - 0.5) / n)


def block_burden_capped(X: np.ndarray, snp_idx, cap: int, rng, minor: bool = False) -> np.ndarray:
    """Mean of column-standardized dosages over a gene's SNPs; subsample to ``cap`` if larger.

    ``minor=True`` first orients every SNP to its minor allele (:func:`orient_minor`), a coding fixed
    by allele frequency rather than the arbitrary REF/ALT label. The resulting burden is invariant to
    a REF/ALT swap (unlike the default), so it can enter a STRICTLY encoding-invariant omnibus while
    retaining a burden's power for direction-coherent signals (the fast canonicalization path)."""
    idx = np.asarray(snp_idx, int)
    if cap and idx.size > cap:
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    M = X[:, idx].astype(float)
    if minor:
        M = orient_minor(M)
    mu = np.nanmean(M, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Mi = np.where(np.isnan(M), mu, M)
    sd = Mi.std(0, ddof=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (Mi - mu) / sd_safe
    Z[:, sd <= 1e-12] = 0.0
    return Z.mean(1)


def orient_minor(M: np.ndarray) -> np.ndarray:
    """Orient each dosage column to the MINOR allele (count of the less-frequent allele).

    The coding is set by the integer allele count in the sample, not by the reference-genome REF/ALT
    label, so a REF/ALT swap (``x -> 2-x``, which sends the ALT count ``n_alt -> n_ref``) re-orients to
    the SAME physical allele and leaves the output unchanged. Integer counts (not a float frequency)
    are used so the decision cannot flip on rounding noise near ``freq == 0.5``. A SNP whose two
    alleles are exactly equinumerous has no defined minor allele (REF/ALT-symmetric): it is neutralised
    to a constant column (zero variance -> dropped from the burden), which is itself invariant under
    ``x -> 2-x``. This makes the minor-allele burden STRICTLY invariant to any REF/ALT recoding."""
    M = np.asarray(M, float)
    nonmiss = np.sum(~np.isnan(M), axis=0)
    n_alt = np.nansum(M, axis=0)                       # ALT-allele count; n_ref = 2*nonmiss - n_alt
    flip = n_alt > nonmiss + 1e-9                      # ALT is the major allele -> orient to minor
    tie = np.abs(n_alt - nonmiss) <= 1e-9             # exact allele tie -> minor undefined
    out = M.copy()
    out[:, flip] = 2.0 - out[:, flip]
    out[:, tie] = 1.0                                  # neutralise symmetric SNPs (invariant: 2-1=1)
    return out


def null_lmm_fit(kernels: dict[str, np.ndarray], y: np.ndarray, C: np.ndarray = None, seed: int = 42):
    """Fit the null LMM y = C.beta + sum_k g_k + e and return (W, V, beta, cv).

    ``W = V^-1/2`` is the whitener implied by the FITTED variance components; ``V`` is the null
    covariance that any valid null replicate must reproduce."""
    from .lmm import fit_multi_reml

    n = next(iter(kernels.values())).shape[0]
    if C is None:
        C = np.ones((n, 1))
    res = fit_multi_reml(y, C, kernels, n_starts=3, random_state=int(seed) % 100000)
    cv = res.component_var
    V = max(cv.get("e", 1e-6), 1e-6) * np.eye(n)
    for name, K in kernels.items():
        V = V + cv.get(name, 0.0) * K
    V = 0.5 * (V + V.T)
    w, Q = np.linalg.eigh(V)
    w = np.clip(w, 1e-10, None)
    W = (Q * (1.0 / np.sqrt(w))) @ Q.T
    beta, *_ = np.linalg.lstsq(W @ C, W @ y, rcond=None)      # GLS fixed effects
    return W, V, beta, cv


def null_replicates(kernels: dict[str, np.ndarray], y: np.ndarray, C: np.ndarray = None,
                    B: int = 1000, method: str = "bootstrap", seed: int = 0,
                    null_fit: tuple | None = None):
    """``B`` null phenotypes for experiment-wide calibration under KINSHIP.

    A raw ``y``-shuffle is NOT a valid null here: samples are not exchangeable under a GRM, and the
    shuffled phenotype carries no kinship signal, so a re-fitted REML drives the genetic variance
    components to zero and the permuted tests end up effectively UNWHITENED while the observed test
    is whitened. Two kinship-preserving alternatives are provided:

    ``bootstrap`` (recommended primary FWER basis)
        Parametric bootstrap from the fitted null model: ``y* = C.beta_hat + V_hat^{1/2} z``,
        ``z ~ N(0, I)``. The null covariance ``V_hat`` (hence the kinship) is reproduced exactly.

    ``whitened``
        Freedman-Lane on the WHITENED residuals: whiten with the FIXED observed-fit ``W``, project
        out ``C`` in the whitened space, permute the residuals there (they are exchangeable under the
        null LMM), then map back. The kinship covariance is preserved by construction.

    ``yshuffle``
        The legacy raw permutation, kept only so the mis-calibration can be quantified.

    Returns ``(list_of_y_star, W_obs, cv_obs)``; the replicates live in the ORIGINAL space, so the
    caller whitens them exactly as it whitens the observed phenotype."""
    n = next(iter(kernels.values())).shape[0]
    if C is None:
        C = np.ones((n, 1))
    if null_fit is None:
        W, V, beta, cv = null_lmm_fit(kernels, y, C, seed=42)
    else:
        W, V, beta, cv = null_fit
    rng = np.random.default_rng(seed)
    fit = C @ beta
    out = []
    if method == "bootstrap":
        w_, Q_ = np.linalg.eigh(0.5 * (V + V.T))
        L = (Q_ * np.sqrt(np.clip(w_, 1e-12, None))) @ Q_.T          # V^{1/2}
        for _ in range(B):
            out.append(fit + L @ rng.standard_normal(n))
    elif method == "whitened":
        yw = W @ y
        Cw = W @ C
        bw, *_ = np.linalg.lstsq(Cw, yw, rcond=None)
        fitw = Cw @ bw
        residw = yw - fitw
        w_, Q_ = np.linalg.eigh(0.5 * (V + V.T))
        L = (Q_ * np.sqrt(np.clip(w_, 1e-12, None))) @ Q_.T          # W^{-1} = V^{1/2}
        for _ in range(B):
            out.append(L @ (fitw + residw[rng.permutation(n)]))
    elif method == "yshuffle":
        for _ in range(B):
            out.append(y[rng.permutation(n)])
    else:
        raise ValueError(f"unknown null method: {method}")
    return out, W, cv


def null_replicates_by_index(
    kernels: dict[str, np.ndarray],
    y: np.ndarray,
    C: np.ndarray = None,
    *,
    indices,
    base_seed: int,
    null_fit: tuple | None = None,
):
    """Generate indexed parametric-bootstrap phenotypes from one frozen null fit.

    Each replicate owns a SHA-256-derived RNG stream keyed only by
    ``(base_seed, replicate_index)``.  Returned phenotypes remain on the input
    analysis scale; callers must not apply INT again.
    """
    from .resampling_checkpoint import replicate_seed

    n = next(iter(kernels.values())).shape[0]
    if C is None:
        C = np.ones((n, 1))
    C = np.asarray(C, float).reshape(n, -1)
    if null_fit is None:
        W, V, beta, cv = null_lmm_fit(kernels, y, C, seed=42)
    else:
        W, V, beta, cv = null_fit
    requested = []
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("replicate indices must be non-negative integers")
        value = int(value)
        if value < 0:
            raise ValueError("replicate indices must be non-negative")
        requested.append(value)
    if len(set(requested)) != len(requested):
        raise ValueError("replicate indices must be unique")

    values, vectors = np.linalg.eigh(0.5 * (V + V.T))
    root = (vectors * np.sqrt(np.clip(values, 1e-12, None))) @ vectors.T
    fit = C @ np.asarray(beta, float)
    out = [
        fit + root @ np.random.default_rng(
            replicate_seed(base_seed, index)).standard_normal(n)
        for index in requested
    ]
    return out, W, cv


def scols_safe(M: np.ndarray) -> np.ndarray:
    mu = M.mean(0)
    sd = M.std(0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (M - mu) / sd


def grm_from_X(
    X: np.ndarray,
    maf_min: float = 0.0,
    *,
    return_provenance: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Build the standardized additive GRM and optionally bind its SNP filter.

    A direct call with the default ``maf_min=0`` retains the historical
    all-column implementation.  Canonical callers request provenance, which
    activates finite-value handling and the declared inclusive MAF filter on
    the already selected analysis-sample matrix.
    """
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("GRM dosage matrix must be two-dimensional")
    if X.shape[0] < 1:
        raise ValueError("GRM dosage matrix must contain at least one sample")
    if not return_provenance and float(maf_min) == 0.0:
        # Frozen direct-call compatibility path.
        mu = np.nanmean(X, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        Xi = np.where(np.isnan(X), mu, X)
        sd = Xi.std(0, ddof=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Z = (Xi - mu) / sd
        K = Z @ Z.T / Z.shape[1]
        K = K / (np.trace(K) / K.shape[0])
        w, Q = np.linalg.eigh(0.5 * (K + K.T))
        w = np.clip(w, 1e-6, None)
        K = (Q * w) @ Q.T
        return K / (np.trace(K) / K.shape[0])

    maf_min = float(maf_min)
    if not np.isfinite(maf_min) or not 0.0 <= maf_min <= 0.5:
        raise ValueError("maf_min must be finite and in [0, 0.5]")
    finite = np.isfinite(X)
    finite_count = finite.sum(axis=0)
    finite_sum = np.where(finite, X, 0.0).sum(axis=0)
    means = np.divide(
        finite_sum,
        finite_count,
        out=np.full(X.shape[1], np.nan, float),
        where=finite_count > 0,
    )
    allele_frequency = means / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    retained = (
        (finite_count > 0)
        & np.isfinite(allele_frequency)
        & (maf >= maf_min)
    )
    n_used = int(retained.sum())
    if n_used == 0:
        raise ValueError(
            "zero variants survive the analysis-sample GRM filter "
            f"(n_variants_input={X.shape[1]}, maf_min={maf_min:g})"
        )
    selected = X[:, retained]
    selected_means = means[retained]
    selected = np.where(np.isfinite(selected), selected, selected_means)
    sd = selected.std(axis=0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (selected - selected_means) / sd
    K = Z @ Z.T / n_used
    trace_scale = np.trace(K) / K.shape[0]
    if not np.isfinite(trace_scale) or trace_scale <= 0.0:
        raise ValueError(
            "surviving analysis-sample GRM variants have zero standardized "
            f"variance (n_variants_used={n_used}, maf_min={maf_min:g})"
        )
    K = K / trace_scale
    w, Q = np.linalg.eigh(0.5 * (K + K.T))
    w = np.clip(w, 1e-6, None)
    K = (Q * w) @ Q.T
    K = K / (np.trace(K) / K.shape[0])
    mask_bytes = np.ascontiguousarray(retained.astype(np.uint8)).tobytes()
    provenance = {
        "n_variants_input": int(X.shape[1]),
        "n_variants_used": n_used,
        "maf_min": maf_min,
        "maf_boundary": "inclusive_greater_than_or_equal",
        "missing_value_policy": "analysis_sample_finite_mean_imputation",
        "retained_variant_mask_encoding": "uint8_input_variant_order",
        "retained_variant_mask": retained.astype(np.uint8).tolist(),
        "retained_variant_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
    }
    return (K, provenance) if return_provenance else K


def whiten_multi(kernels: dict[str, np.ndarray], y: np.ndarray, X: np.ndarray = None, seed: int = 42):
    """Subgenome-stratified whitener from a multi-kernel null LMM (REML variance components).

    ``X`` is the fixed-effect mean model for the null LMM; default ``None`` => intercept-only.
    Passing a covariate block ``[1, PCs, covariates]`` estimates the variance components
    conditional on those fixed effects, so structure is not absorbed into sigma^2_sub."""
    from .lmm import fit_multi_reml

    n = next(iter(kernels.values())).shape[0]
    if X is None:
        X = np.ones((n, 1))
    res = fit_multi_reml(y, X, kernels, n_starts=3, random_state=int(seed) % 100000)
    cv = res.component_var
    V = max(cv.get("e", 1e-6), 1e-6) * np.eye(n)
    for name, K in kernels.items():
        V = V + cv.get(name, 0.0) * K
    w, Q = np.linalg.eigh(0.5 * (V + V.T))
    w = np.clip(w, 1e-10, None)
    return (Q * (1.0 / np.sqrt(w))) @ Q.T, cv


TARGET_ESTIMABILITY_RTOL = float(np.sqrt(np.finfo(np.float64).eps))   # 1.4901161193847656e-08
TRIAD3_FORMAL_BOOTSTRAP_MIN_B = 999
PAIRWISE_OMNIB_FORMAL_BOOTSTRAP_MIN_B = 999
PAIRWISE_BURDEN_FORMAL_PERMUTATION_MIN_B = 999
ESTIMABILITY_POLICY = dict(
    method="frisch_waugh_lovell_target_residual_ratio",
    nuisance_rank_rtol_formula="max(n, q_active) * float64_eps",
    target_estimability_rtol=TARGET_ESTIMABILITY_RTOL)


DESIGN_EXCLUSION_REASONS = ("nonfinite_design", "zero_target",
                            "target_nonestimable", "insufficient_df")


def _fwl_design(Zw: np.ndarray, xw: np.ndarray, *,
                target_rtol: float = TARGET_ESTIMABILITY_RTOL) -> tuple:
    """Design-only half of the FWL test: nuisance column space plus the residualised target.

    Decided WITHOUT the response, so this is the part that defines which hypotheses exist."""
    if not (np.all(np.isfinite(Zw)) and np.all(np.isfinite(xw))):
        return None, 0, None, 0.0, "nonfinite_design"
    xn = float(np.sqrt(xw @ xw))
    if not (xn > 0):
        return None, 0, None, 0.0, "zero_target"
    rank_z, Qz = 0, None
    if Zw.size:
        dz = np.sqrt(np.einsum("ij,ij->j", Zw, Zw))
        act = dz > 0
        if act.any():
            Zeq = Zw[:, act] / dz[act]
            U, s, _ = np.linalg.svd(Zeq, full_matrices=False)   # a failure here is not
            #   "unestimable" but "not determined": let it propagate rather than retire a hypothesis
            if s[0] > 0:
                rank_z = int(np.count_nonzero(
                    s > s[0] * max(Zeq.shape) * np.finfo(np.float64).eps))
            Qz = U[:, :rank_z]
    xr = xw - Qz @ (Qz.T @ xw) if rank_z else xw
    xrn = float(np.sqrt(xr @ xr))
    if not np.isfinite(xrn) or xrn / xn <= target_rtol:
        return Qz, rank_z, xr, xrn, "target_nonestimable"
    if xw.shape[0] - rank_z - 1 < 1:
        return Qz, rank_z, xr, xrn, "insufficient_df"
    return Qz, rank_z, xr, xrn, None


def pairwise_design_mask(BX: np.ndarray, BY: np.ndarray, C: np.ndarray = None,
                         dominance_adjust: bool = False) -> tuple:
    """Which pairs are testable, decided on the RAW (unwhitened) design.

    The whitener is fitted to the phenotype, so deciding estimability after whitening would let the
    tested FAMILY depend on y -- and a family that moves with the response cannot calibrate a
    permutation null. Whether the product is estimable given the two main effects is a rank
    property, invariant under the invertible whitening transform, so the raw design answers the same
    question without that dependence."""
    n, G = BX.shape
    base = np.ones((n, 1)) if C is None else np.asarray(C, float).reshape(n, -1)
    INT = BX * BY
    BX2, BY2 = (BX * BX, BY * BY) if dominance_adjust else (None, None)
    mask = np.zeros(G, bool)
    excl = {}
    for g in range(G):
        cols = [base, BX[:, g].reshape(-1, 1), BY[:, g].reshape(-1, 1)]
        if dominance_adjust:
            cols += [BX2[:, g].reshape(-1, 1), BY2[:, g].reshape(-1, 1)]
        why = _fwl_design(np.column_stack(cols), INT[:, g])[4]
        mask[g] = why is None
        if why is not None:
            excl[g] = why
    return mask, excl


def _coef_pval_fwl(Zw: np.ndarray, xw: np.ndarray, yw: np.ndarray, *,
                   target_rtol: float = TARGET_ESTIMABILITY_RTOL) -> tuple:
    """Two-sided p for the coefficient of ``xw`` given nuisance block ``Zw`` (Frisch-Waugh-Lovell).

    Estimability is a property of the TESTED column, not of the whole design: two duplicated
    nuisance columns leave the interaction perfectly estimable, so gating on the condition number of
    ``[Zw, xw]`` discards valid results. Here only ``xw``'s component orthogonal to the numerical
    column space of ``Zw`` is required to survive, and ``df = n - rank(Zw) - 1`` follows the actual
    nuisance rank. The two tolerances answer different questions: ``max(n, q) * eps`` is a
    backward-error threshold for the dimension of that column space, while ``sqrt(eps)`` is a
    coefficient-stability policy on the tested direction. RSS is the explicit residual norm, never
    ``y'y - (U'y)'(U'y)`` -- on a good fit that subtraction cancels catastrophically and can return a
    negative value, which a clamp then turns into a NaN model. A test that cannot be run returns NaN
    with a reason, never ``p = 1.0``, which ACAT and genomic control would treat as a real
    maximally-non-significant observation.

    Returns ``(p, reason, design_ok)``. ``design_ok`` is False only for the reasons in
    :data:`DESIGN_EXCLUSION_REASONS`, all of which are decided WITHOUT looking at ``yw`` -- that is
    what makes the estimability mask outcome-independent and therefore reusable as a frozen
    permutation family. A response-dependent failure keeps ``design_ok`` True so the caller can treat
    it as an analysis error rather than silently retire the hypothesis."""
    Qz, rank_z, xr, xrn, why = _fwl_design(Zw, xw, target_rtol=target_rtol)
    if why is not None:
        return float("nan"), why, False
    n = yw.shape[0]
    df = n - rank_z - 1
    # everything below depends on yw, so a failure here is an analysis error on a design-valid
    # hypothesis, NOT an unestimable hypothesis
    yr = yw - Qz @ (Qz.T @ yw) if rank_z else yw
    u = xr / xrn
    beta_eq = float(u @ yr)
    resid = yr - beta_eq * u
    sigma_eq = float(np.sqrt(float(resid @ resid) / df))
    if not np.isfinite(sigma_eq) or sigma_eq <= 0:
        return float("nan"), "zero_residual_variance", True
    t = beta_eq / sigma_eq
    if not np.isfinite(t):
        return float("nan"), "nonfinite_statistic", True
    return float(2.0 * stats.t.sf(abs(t), df)), None, True


def pairwise_pvals(Wh: np.ndarray, y: np.ndarray, BX: np.ndarray, BY: np.ndarray,
                   C: np.ndarray = None, dominance_adjust: bool = False, *,
                   fixed_mask: np.ndarray = None, return_diag: bool = False):
    """Per-pair interaction p (whitened GLS t-test on the b_X*b_Y coefficient).

    ``C`` is the fixed-effect covariate block (n, p_c) that already includes the intercept column
    (e.g. ``[1, PC1..PCk, env...]``); default ``None`` => intercept-only, i.e. design
    ``[1, b_X, b_Y, b_X*b_Y]``. With ``C`` given the design is ``[C, b_X, b_Y, b_X*b_Y]`` and the
    residual df is ``n - (p_c + 3)``.

    ``dominance_adjust`` (default ``False`` => byte-identical to the legacy design) appends the two
    per-gene squared burdens, giving ``[C, b_X, b_Y, b_X^2, b_Y^2, b_X*b_Y]`` (df ``n-(p_c+5)``).
    Because the product ``b_X*b_Y`` becomes collinear with ``b_X^2`` (and ``b_Y^2``) as the two
    homoeolog burdens become correlated, an unmodelled per-copy dominance/curvature main effect can
    otherwise leak into the interaction term under homoeolog collinearity; conditioning on the
    squared burdens removes that leak. The tested coefficient is then the product effect CONDITIONAL
    on the additive AND per-gene quadratic terms — not a generic dominance test. Costs 2 df and
    makes the product harder to estimate at extreme collinearity.

    ``fixed_mask`` restricts the scan to a pre-computed estimability mask so that a permutation
    replicate tests exactly the family the observed scan tested; ``return_diag`` additionally returns
    the mask and the per-unit exclusion reasons."""
    n, G = BX.shape
    yw = Wh @ y
    if not np.all(np.isfinite(yw)):
        raise ValueError("whitened response contains non-finite values")
    BXw = Wh @ BX
    BYw = Wh @ BY
    INTw = Wh @ (BX * BY)
    if dominance_adjust:
        BX2w = Wh @ (BX * BX)
        BY2w = Wh @ (BY * BY)
    if C is None:
        Cw = (Wh @ np.ones(n)).reshape(-1, 1)
    else:
        Cw = Wh @ np.asarray(C, float).reshape(n, -1)
    pv = np.full(G, np.nan)
    design_ok = np.zeros(G, bool) if fixed_mask is None else np.asarray(fixed_mask, bool).copy()
    excl, fail = {}, {}
    for g in range(G):
        if fixed_mask is not None and not fixed_mask[g]:
            continue
        cols = [Cw, BXw[:, g].reshape(-1, 1), BYw[:, g].reshape(-1, 1)]
        if dominance_adjust:
            cols += [BX2w[:, g].reshape(-1, 1), BY2w[:, g].reshape(-1, 1)]
        pv[g], why, ok = _coef_pval_fwl(np.column_stack(cols), INTw[:, g], yw)
        design_ok[g] = ok
        if why is not None:
            (fail if ok else excl)[g] = why
    if return_diag:
        return pv, dict(n_planned=int(G), estimable=design_ok, exclusions=excl, failures=fail)
    return pv


def _threeway_nuisance(A: np.ndarray, B: np.ndarray, D: np.ndarray,
                       C: np.ndarray = None) -> np.ndarray:
    """Hierarchy-preserving nuisance block for a conditional three-way test.

    The tested model is

    ``y ~ C + A + B + D + A:B + A:D + B:D + A:B:D``.

    This helper returns every term below ``A:B:D``.  Keeping all lower-order
    terms is mandatory: without them, a strong pairwise interaction can leak
    into the nominal three-way coefficient.
    """
    n = A.shape[0]
    base = np.ones((n, 1)) if C is None else np.asarray(C, float).reshape(n, -1)
    return np.column_stack([base, A, B, D, A * B, A * D, B * D])


def threeway_design_mask(BA: np.ndarray, BB: np.ndarray, BD: np.ndarray,
                         C: np.ndarray = None) -> tuple:
    """Outcome-independent estimability mask for conditional ``A×B×D`` tests.

    Estimability is frozen on the raw design, before the phenotype-dependent
    whitener is fitted.  The returned diagnostics include the fraction of the
    raw three-way column that survives projection on all lower-order terms;
    values near the global target tolerance are numerically fragile even when
    they remain formally testable.
    """
    n, G = BA.shape
    if BB.shape != (n, G) or BD.shape != (n, G):
        raise ValueError("threeway burden matrices must have identical shapes")
    mask = np.zeros(G, bool)
    excl = {}
    residual_ratio = np.full(G, np.nan)
    target_sd = np.full(G, np.nan)
    information_max_fraction = np.full(G, np.nan)
    information_top10_fraction = np.full(G, np.nan)
    information_effective_n = np.full(G, np.nan)
    for g in range(G):
        a, b, d = BA[:, g], BB[:, g], BD[:, g]
        target = a * b * d
        nuisance = _threeway_nuisance(a, b, d, C=C)
        _Q, _rank, xr, xrn, why = _fwl_design(nuisance, target)
        xn = float(np.linalg.norm(target))
        target_sd[g] = float(np.std(target, ddof=0))
        if xn > 0 and np.isfinite(xrn):
            residual_ratio[g] = xrn / xn
        if xr is not None and np.isfinite(xr).all():
            information = xr ** 2
            total = float(information.sum())
            if total > 0:
                share = information / total
                k = min(10, share.size)
                information_max_fraction[g] = float(share.max())
                information_top10_fraction[g] = float(
                    np.partition(share, share.size - k)[-k:].sum())
                information_effective_n[g] = float(
                    1.0 / np.sum(share ** 2))
        mask[g] = why is None
        if why is not None:
            excl[g] = why
    return mask, excl, {
        "target_residual_ratio": residual_ratio,
        "target_sd": target_sd,
        "target_information_max_fraction": information_max_fraction,
        "target_information_top10_fraction": information_top10_fraction,
        "target_information_effective_n": information_effective_n,
    }


def threeway_pvals(Wh: np.ndarray, y: np.ndarray, BA: np.ndarray, BB: np.ndarray,
                   BD: np.ndarray, C: np.ndarray = None, *,
                   fixed_mask: np.ndarray = None, return_diag: bool = False):
    """Conditional three-way burden p-values under a whitened GLS model.

    For every homoeolog triad this tests only the ``A×B×D`` coefficient while
    retaining the three main effects and all three pairwise products.  ``BA``,
    ``BB`` and ``BD`` are expected to be centered/scaled gene burden matrices.
    A fixed raw-design mask should be supplied for resampling analyses so that
    the tested family cannot change with the phenotype.
    """
    n, G = BA.shape
    if BB.shape != (n, G) or BD.shape != (n, G):
        raise ValueError("threeway burden matrices must have identical shapes")
    yw = Wh @ np.asarray(y, float)
    if not np.all(np.isfinite(yw)):
        raise ValueError("whitened response contains non-finite values")
    if C is None:
        Cw = (Wh @ np.ones(n)).reshape(-1, 1)
    else:
        Cw = Wh @ np.asarray(C, float).reshape(n, -1)
    Aw, Bw, Dw = Wh @ BA, Wh @ BB, Wh @ BD
    ABw, ADw, BDw = Wh @ (BA * BB), Wh @ (BA * BD), Wh @ (BB * BD)
    ABDw = Wh @ (BA * BB * BD)
    pv = np.full(G, np.nan)
    design_ok = np.zeros(G, bool) if fixed_mask is None else np.asarray(fixed_mask, bool).copy()
    excl, fail = {}, {}
    for g in range(G):
        if fixed_mask is not None and not fixed_mask[g]:
            continue
        nuisance = np.column_stack([
            Cw, Aw[:, g], Bw[:, g], Dw[:, g],
            ABw[:, g], ADw[:, g], BDw[:, g],
        ])
        pv[g], why, ok = _coef_pval_fwl(nuisance, ABDw[:, g], yw)
        design_ok[g] = ok
        if why is not None:
            (fail if ok else excl)[g] = why
    if return_diag:
        return pv, dict(n_planned=int(G), estimable=design_ok,
                        exclusions=excl, failures=fail)
    return pv


def marginal_pvals(Wh: np.ndarray, y: np.ndarray, B: np.ndarray,
                   C: np.ndarray = None, *, return_diag: bool = False):
    """Per-gene SINGLE-burden marginal p (whitened GLS t-test on b alone).

    Same whitener/covariates as :func:`pairwise_pvals`, but the design is
    ``[C, b]`` (no product term), so the test asks whether a single homoeolog
    burden associates on its own. Comparing this against the pair interaction p
    is the "signal invisible to single-locus tests" contrast.
    """
    n, G = B.shape
    yw = Wh @ y
    if not np.all(np.isfinite(yw)):
        raise ValueError("whitened response contains non-finite values")
    Bw = Wh @ B
    if C is None:
        Cw = (Wh @ np.ones(n)).reshape(-1, 1)
    else:
        Cw = Wh @ np.asarray(C, float).reshape(n, -1)
    pv = np.full(G, np.nan)
    design_ok = np.zeros(G, bool)
    excl, fail = {}, {}
    for g in range(G):
        pv[g], why, ok = _coef_pval_fwl(Cw, Bw[:, g], yw)
        design_ok[g] = ok
        if why is not None:
            (fail if ok else excl)[g] = why
    if return_diag:
        return pv, dict(n_planned=int(G), estimable=design_ok, exclusions=excl, failures=fail)
    return pv


def gene_pc_scores(X: np.ndarray, snp_idx, cap: int, rng, n_pc: int = 1) -> np.ndarray:
    """Top-``n_pc`` standardized principal-component scores (n, k) of a gene's SNP block.

    Encoding-invariant companion to :func:`block_burden_capped`: a per-SNP REF/ALT swap sends the
    standardized dosage column ``z -> -z`` (i.e. ``Z -> Z diag(+-1)``), which flips the right
    singular vectors but leaves ``U`` and the singular values unchanged, so the PC scores are
    numerically identical. The burden ``mean(Z)`` is NOT invariant (a signed average that assumes a
    coherent within-gene direction and a meaningful REF/ALT sign); the PC-score interaction removes
    that dependence on the arbitrary reference-allele coding. Each score column is sign-fixed by its
    own largest-magnitude entry (itself flip-invariant) and scaled to unit variance."""
    idx = np.asarray(snp_idx, int)
    if cap and idx.size > cap:
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    M = X[:, idx].astype(float)
    mu = np.nanmean(M, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Mi = np.where(np.isnan(M), mu, M)
    sd = Mi.std(0, ddof=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (Mi - mu) / sd_safe
    Z[:, sd <= 1e-12] = 0.0
    return pc_scores_std(Z, n_pc)


def pc_scores_std(Z: np.ndarray, n_pc: int = 1) -> np.ndarray:
    """Top-``n_pc`` sign-fixed, unit-variance PC scores of an ALREADY-standardized block ``Z`` (n, m).
    Split out from :func:`gene_pc_scores` so callers holding a pre-standardized SNP block (e.g. the
    benchmark's ``scols_safe`` columns) reuse the same encoding-invariant scores. Directions below the
    numerical-rank tolerance are dropped so unit-scaling never amplifies arbitrary noise."""
    Z = np.asarray(Z, float)
    if not np.all(np.isfinite(Z)):
        raise ValueError("standardized gene block contains non-finite values before PCA")
    k = int(min(n_pc, Z.shape[1], max(1, Z.shape[0] - 1)))
    try:
        U, S, _Vt = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        # NumPy normally calls divide-and-conquer GESDD, which can very rarely
        # fail on a finite, highly collinear real gene block. Classical GESVD
        # is slower but more robust and is used only for that exceptional
        # block; returning zeros/NA would silently change the omniB estimand.
        from scipy.linalg import svd

        try:
            U, S, _Vt = svd(
                Z, full_matrices=False, check_finite=True, lapack_driver="gesvd")
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"gene-block PCA failed with both gesdd and gesvd "
                f"(shape={Z.shape})") from exc
    tol = max(Z.shape) * np.finfo(float).eps * (S[0] if S.size else 0.0)
    rank = int(np.sum(S > tol))
    if rank == 0:
        return np.zeros((Z.shape[0], 1))
    k = int(min(k, rank, S.size))
    scores = U[:, :k] * S[:k]
    for j in range(k):
        col = scores[:, j]
        s = np.sign(col[int(np.argmax(np.abs(col)))])       # sign from the score (flip-invariant)
        if s != 0:
            scores[:, j] = col * s
    sc_sd = scores.std(0, ddof=0)
    sc_sd = np.where(sc_sd > 1e-12, sc_sd, 1.0)
    return (scores - scores.mean(0)) / sc_sd


def gene_pc1_matrix(sd_X, genes, snp_idx_map, cap: int, rng) -> np.ndarray:
    """(n, G) matrix of gene PC1 scores, aligned to ``genes`` order (drop-in for a burden matrix in
    :func:`pairwise_pvals`, giving the encoding-invariant PC1 x PC1 interaction)."""
    cols = [gene_pc_scores(sd_X, snp_idx_map[g], cap, rng, n_pc=1)[:, 0] for g in genes]
    return np.column_stack(cols)


def kernel_interaction_pvals(Wh: np.ndarray, y: np.ndarray, PX: list, PY: list,
                             C: np.ndarray = None) -> np.ndarray:
    """Per-pair low-rank pairwise-kernel interaction p (whitened GLS F-test).

    ``PX``/``PY`` are length-G lists of per-gene top-k PC score blocks (n, k_x)/(n, k_y) from
    :func:`gene_pc_scores`. For each pair the design is ``[C, PX, PY, PX (x) PY]`` and the test is the
    joint F-test of the ``k_x * k_y`` cross-product columns conditional on the two main-effect PC
    blocks. This is the rank-k eigen-approximation of the Hadamard interaction of the two linear gene
    kernels ``(Z_X Z_X') o (Z_Y Z_Y')`` (PC1 x PC1 is the k=1 case), and is encoding-invariant for the
    same reason the scores are. F-based, so it drops into the existing permutation/ACAT null."""
    n = y.shape[0]
    G = len(PX)
    yw = Wh @ y
    if C is None:
        Cw = (Wh @ np.ones(n)).reshape(-1, 1)
    else:
        Cw = Wh @ np.asarray(C, float).reshape(n, -1)
    pv = np.empty(G)
    for g in range(G):
        Ax = np.asarray(PX[g], float).reshape(n, -1)
        Ay = np.asarray(PY[g], float).reshape(n, -1)
        cross = (Ax[:, :, None] * Ay[:, None, :]).reshape(n, -1)     # k_x * k_y cross-products
        Axw = Wh @ Ax
        Ayw = Wh @ Ay
        crossw = Wh @ cross
        Xred = np.column_stack([Cw, Axw, Ayw])
        Xfull = np.column_stack([Xred, crossw])
        br, _r, rank_r, _s = np.linalg.lstsq(Xred, yw, rcond=None)
        bf, _r2, rank_f, _s2 = np.linalg.lstsq(Xfull, yw, rcond=None)
        dfd = n - rank_f
        if rank_f <= rank_r or dfd < 1:                              # interaction block not estimable:
            pv[g] = np.nan                                           # NaN (not p=1) so ACAT omits it
            continue
        rss_r = float(((yw - Xred @ br) ** 2).sum())
        rss_f = float(((yw - Xfull @ bf) ** 2).sum())
        dfn = rank_f - rank_r                                        # estimable interaction df
        denom = rss_f / dfd
        if denom <= 1e-300:                                         # full model fits ~perfectly
            pv[g] = 0.0 if rss_r - rss_f > 1e-300 else np.nan
            continue
        F = ((rss_r - rss_f) / dfn) / denom
        pv[g] = float(stats.f.sf(max(F, 0.0), dfn, dfd))
    return pv


def omnibus_pvals(*pval_arrays: np.ndarray) -> np.ndarray:
    """Per-pair ACAT (Cauchy) combination of the encoding-dependent burden-product p and the
    encoding-invariant PC/kernel interaction p, giving an adaptive test that keeps the burden's power
    for direction-coherent signals while covering mixed-direction / haplotype interactions.

    A non-estimable component (``NaN`` or the conservative ``p==1`` sentinel) is dropped from the
    combination rather than clipped to ``1-eps``: the latter injects a large negative Cauchy term
    that would spuriously drag the omnibus toward the null and mask a valid component."""
    arrs = [np.asarray(p, float).ravel() for p in pval_arrays]
    L = arrs[0].size
    if not arrs or any(a.size != L for a in arrs):
        raise ValueError("omnibus_pvals: components must be non-empty and equal length")
    stack = np.vstack(arrs)
    stack = np.where(stack >= 1.0 - 1e-12, np.nan, stack)
    return np.array([acat(stack[:, j]) for j in range(L)])


def _gene_coord(sub, snp_idx) -> tuple:
    """(chrom, pos) for a gene = first SNP's chrom + median SNP bp (NA/-1 if none)."""
    idx = np.asarray(snp_idx, int)
    chunk = getattr(sub, "chunk", None)
    pos = getattr(chunk, "pos", None)
    chrom = getattr(chunk, "chrom", None)
    if pos is None or chrom is None or idx.size == 0:
        return "NA", -1
    pos, chrom = np.asarray(pos), np.asarray(chrom)
    idx = idx[(idx >= 0) & (idx < pos.size)]          # drop sentinels / OOB
    if idx.size == 0:
        return "NA", -1
    c = chrom[idx]
    # the dominant chromosome (genes should be on one; guards mixed mappings)
    vals, counts = np.unique(c.astype(str), return_counts=True)
    return str(vals[int(counts.argmax())]), int(np.median(pos[idx]))


def _neglog10(p: np.ndarray) -> np.ndarray:
    """-log10(p) with p floored at 1e-300 (consistent finite cap; never +inf)."""
    return -np.log10(np.clip(np.asarray(p, float), 1e-300, 1.0))


def _decile_bin(x: np.ndarray) -> np.ndarray:
    """Decile bin (0..9) of a 1-D vector by rank (descriptive column only). Ties share a bin;
    an all-constant vector maps to bin 0."""
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return np.zeros(0, int)
    r = stats.rankdata(x, method="average")          # 1..n, ties averaged
    return np.clip(((r - 0.5) / n * 10).astype(int), 0, 9)


def _normalize_weights(raw, G: int, label: str) -> np.ndarray:
    """Scale y-independent prior weights to sum ``G``. Weighted Bonferroni spends alpha*w_i/G per
    test, so FWER control needs sum(w) <= G; equality makes the procedure invariant to any rescaling
    of the supplied weights."""
    w = np.asarray(raw, float)
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError(f"{label} must be finite and non-negative")
    peak = float(w.max()) if w.size else 0.0
    if not peak > 0:
        raise ValueError(f"{label} are all zero")
    ws = w / peak                                    # scale first: sum() alone can overflow to inf
    tot = float(ws.sum())
    if not (np.isfinite(tot) and tot > 0):
        raise ValueError(f"{label} do not sum to a finite positive total")
    return ws * (G / tot)


def _tsv_p(v: float) -> str:
    """Round-trippable p for a ranking dump; a test that was not run is NA, never a number."""
    if v is None:
        return "NA"
    try:
        return repr(float(v)) if np.isfinite(v) else "NA"
    except (TypeError, ValueError):
        return "NA"


def _json_safe(obj):
    """Recursively map non-finite floats to ``None``. Bare ``NaN``/``Infinity`` are not valid
    RFC 8259 JSON and strict parsers reject them, so a non-estimable test must serialize as null."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _write_ranking_tsv(path, header: list, rows: list) -> None:
    """Write the full genome-wide ranking (every callable unit), deterministically ordered."""
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        wtr = csv.writer(fh, delimiter="\t")
        wtr.writerow(header)
        wtr.writerows(rows)


def _rank_with_ties(pv: np.ndarray, keys: list) -> tuple:
    """Deterministic ascending-p order (stable sort) plus a tie-group id per row (rows sharing an
    identical p get the same tie_group). Returns (order, rank_of_row, tie_group_of_row), with
    rank/tie indexed by original row position."""
    pv = np.asarray(pv, float)
    G = pv.size
    order = np.argsort(pv, kind="stable")            # stable => ties keep original (gene) order
    rank_of = np.empty(G, int)
    tie_of = np.empty(G, int)
    rank_of[order] = np.arange(G)
    tg = 0
    prev = None
    for j, i in enumerate(order):
        if prev is None or pv[i] != prev:
            tg = j                                   # tie group = rank of the first member
            prev = pv[i]
        tie_of[i] = tg
    return order, rank_of, tie_of


@dataclass
class SubgenomeData:
    X: np.ndarray                       # (n_samples, n_snp) for this subgenome
    gene_snp: dict                      # gene_id -> snp_idx (into X)
    samples: list
    chunk: object = None                # GenoChunk (for grm.compute_grm)


@dataclass(frozen=True)
class FrozenTraitSet:
    """Immutable, ordered, predeclared trait set for multi-trait (pleiotropy) scans.

    Freezing the trait list in a hashable object (order preserved, content digest recorded) keeps
    the "multiplicity = G, not G x T" claim auditable: the set is declared before the association
    test and cannot be silently reordered, deduplicated, or expanded."""

    traits: tuple
    digest: str

    @classmethod
    def from_list(cls, traits) -> FrozenTraitSet:
        ts = list(traits)
        if not ts:
            raise ValueError("multi_trait must be a non-empty ordered list of trait names")
        if any(not isinstance(t, str) for t in ts):
            raise ValueError(f"multi_trait must be trait-name strings; got {ts}")
        dup = sorted({t for t in ts if ts.count(t) > 1})
        if dup:
            raise ValueError(f"multi_trait has duplicate traits {dup}; the trait set must be unique")
        import hashlib

        digest = hashlib.sha256("\x00".join(ts).encode()).hexdigest()[:16]
        return cls(traits=tuple(ts), digest=digest)


@dataclass
class InteractResult:
    trait: str
    transform: str
    n: int
    G: int
    pair_acat: float
    pair_acat_emp: float
    min_p: float
    lambda_gc_obs: float
    lambda_gc_perm_median: float
    bonferroni_alpha: float
    n_sig: int
    sig: list = field(default_factory=list)
    top: list = field(default_factory=list)
    sigma_hat: dict = field(default_factory=dict)
    weighted: dict = None
    covariates: dict = None
    minp_perm_emp: float = float("nan")        # empirical p of the observed min pair-p (permutation FWER)
    minp_perm_threshold: float = float("nan")  # k-th SMALLEST permuted min-p, k = floor(0.05*(B+1));
    #   reject iff min_p is STRICTLY BELOW it -- equality at the k-th order statistic does not reject,
    #   which matters when replicates tie at zero. `minp_perm_rejected` is the authoritative decision.
    minp_perm_threshold_comparator: str = "strict_less_than"
    minp_perm_rejected: bool = None
    n_planned: int = None                      # hypotheses the design declared
    n_valid: int = None                        # hypotheses actually estimable (design-determined)
    n_unestimable: int = None
    estimability: dict = None                  # policy, tolerances and the excluded unit ids
    permutation: dict = None                   # requested/used/degenerate replicate counts
    inference_plan: dict = None                # which procedure was predeclared to spend alpha
    # --- omniB + parametric-bootstrap production path (None on the legacy burden+permutation path,
    #     so a legacy result serializes byte-identically once these are dropped from the summary) ---
    statistic: str = "burden"                  # "omniB" (encoding-invariant primary) | "burden" (legacy)
    calibration_method: str = "permutation"    # "bootstrap" (kinship-preserving) | "permutation" (legacy FL)
    bootstrap_B: int = None
    bootstrap_seed: int = None
    minp_boot_emp: float = None                # experiment-wide FWER: empirical p of observed min-p under bootstrap null
    minp_boot_threshold: float = None          # 5th-percentile of bootstrap min-p (alpha=0.05 experiment-wide cutoff)
    minp_boot_rejected: bool = None            # exact plus-one min-P decision when defined
    tail_excess: dict = None                   # aggregate tail-excess: observed vs bootstrap null (per threshold)
    component_diagnostics: dict = None         # omniB component names/driver counts + interpretation guard
    model_diagnostics: dict = None             # optional design/support diagnostics for experimental models
    analytic_screen_n: int = None              # descriptive Bonferroni screen; never a triad3 discovery
    analytic_screen_sig: list = None


def _validate_snp_mapping(
        plink_prefix: str, npz_path: str, expected_subgenome: str | None = None) -> None:
    """Verify that a SNP-to-gene NPZ belongs to the exact PLINK BIM in use."""
    from .io import plink_bim_sha256

    z = np.load(npz_path, allow_pickle=True)
    required = {"gene_ids", "snp_idx", "bim_sha256", "n_variants"}
    missing = sorted(required - set(z.files))
    if missing:
        raise ValueError(
            f"{npz_path} is an unverified/legacy snp_to_gene NPZ (missing {missing}); "
            "rerun `homoeogwas prep-snps` with the BED used for this interaction")
    expected = str(np.asarray(z["bim_sha256"]).item())
    observed = plink_bim_sha256(plink_prefix)
    if expected != observed:
        raise ValueError(
            f"snp_to_gene/BIM fingerprint mismatch for {plink_prefix}: the NPZ was "
            "built from a different BIM or variant order; rerun `homoeogwas prep-snps`")
    n_variants = int(np.asarray(z["n_variants"]).item())
    if n_variants < 1:
        raise ValueError(f"{npz_path} records invalid n_variants={n_variants}")
    gene_ids = [str(value) for value in z["gene_ids"].tolist()]
    indices = z["snp_idx"]
    if len(gene_ids) != len(indices):
        raise ValueError(
            f"{npz_path} gene_ids/snp_idx length mismatch: "
            f"{len(gene_ids)} != {len(indices)}")
    if len(set(gene_ids)) != len(gene_ids):
        raise ValueError(f"{npz_path} contains duplicate gene IDs")
    if expected_subgenome is not None and "subgenome" in z.files:
        recorded = str(np.asarray(z["subgenome"]).item())
        if recorded != expected_subgenome:
            raise ValueError(
                f"{npz_path} records subgenome {recorded!r}, expected "
                f"{expected_subgenome!r}")
    for i, raw in enumerate(indices):
        idx = np.asarray(raw, dtype=np.int64)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= n_variants):
            raise ValueError(
                f"{npz_path} contains out-of-range SNP indices for gene row {i}; "
                f"valid BED-column range is [0, {n_variants})")


def _load_subgenome(
        plink_prefix: str, npz_path: str, *, verify_mapping: bool = True) -> SubgenomeData:
    from .io import load_bed_hardcall

    if verify_mapping:
        _validate_snp_mapping(plink_prefix, npz_path)

    bed = load_bed_hardcall(plink_prefix)
    X = np.asarray(bed.dosage, dtype=np.float64)
    samples = [str(s) for s in np.asarray(bed.samples)]
    z = np.load(npz_path, allow_pickle=True)
    n_variants = int(np.asarray(z["n_variants"]).item())
    if X.shape[1] != n_variants:
        raise ValueError(
            f"snp_to_gene records {n_variants} variants but BED has {X.shape[1]}")
    gene_ids = z["gene_ids"].tolist()
    snp_idx = z["snp_idx"]
    gene_snp = {g: np.asarray(snp_idx[i], int) for i, g in enumerate(gene_ids)}
    return SubgenomeData(X=X, gene_snp=gene_snp, samples=samples, chunk=bed)


def _build_grm(
    sd: SubgenomeData,
    sample_idx: np.ndarray,
    method: str,
    maf_min: float,
    *,
    return_provenance: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Subgenome GRM restricted to valid samples, trace-normed. ``compute_grm_maf`` reuses the
    package GRM; ``grm_from_X`` is the all-SNP PSD-clipped variant (sensitivity)."""
    n_t = sample_idx.size
    if method == "compute_grm_maf":
        from .grm import compute_grm
        K, info = compute_grm(sd.chunk, maf_min=maf_min)
        K = np.asarray(K)[np.ix_(sample_idx, sample_idx)]
        K = K / (np.trace(K) / n_t)
        return (K, dict(info)) if return_provenance else K
    if method == "grm_from_X":
        # Selection precedes every missingness, allele-frequency and MAF
        # decision. Held-out FAM rows therefore cannot alter the formal null.
        analysis_X = np.asarray(sd.X, float)[np.asarray(sample_idx, int), :]
        K, provenance = grm_from_X(
            analysis_X, maf_min=maf_min, return_provenance=True)
        K = K / (np.trace(K) / n_t)
        return (K, provenance) if return_provenance else K
    raise ValueError(f"unknown grm.method '{method}' (use compute_grm_maf | grm_from_X)")


def genotype_pcs(kernels: dict[str, np.ndarray], n_pcs: int) -> np.ndarray:
    """Top-``n_pcs`` genotype principal components from the combined (mean) subgenome GRM.

    Reuses the already-built per-subgenome GRMs (no genotype re-read): K_comb = mean_s K_s shares
    its leading eigenvectors with standard genotype PCA. PCs are y-independent (genotype only) and
    standardized to unit scale."""
    n = next(iter(kernels.values())).shape[0]
    if n_pcs <= 0:
        return np.empty((n, 0))
    K = np.mean([np.asarray(K_, float) for K_ in kernels.values()], axis=0)
    w, Q = np.linalg.eigh(0.5 * (K + K.T))
    order = np.argsort(w)[::-1]
    n_pos = int(np.sum(w > 1e-8 * max(abs(w).max(), 1.0)))   # positive-eigenvalue capacity
    k = min(n_pcs, n_pos, max(n - 4, 0))                     # cap by rank AND leave design headroom
    idx = order[:k]
    P = Q[:, idx]
    sd = P.std(0, ddof=0)
    return P / np.where(sd > 1e-12, sd, 1.0)


def build_covariate_block(kernels: dict[str, np.ndarray], n_t: int, *, n_pcs: int = 0,
                          extra: np.ndarray = None) -> tuple:
    """Assemble the fixed-effect covariate design ``C = [1, PC1..PCk, extra...]`` (intercept first).

    With ``n_pcs==0`` and ``extra is None`` returns ``(None, ...)`` so the engine takes the
    intercept-only path. ``extra`` is an optional sample-aligned, y-independent covariate matrix
    (n_t, q) (e.g. environment/batch), standardized here."""
    blocks = [np.ones((n_t, 1))]
    cols = ["intercept"]
    actual_pcs = 0
    if n_pcs and n_pcs > 0:
        P = genotype_pcs(kernels, n_pcs)            # may be clamped to positive-eigenvalue capacity
        actual_pcs = int(P.shape[1])
        if actual_pcs:
            blocks.append(P)
            cols += [f"PC{i+1}" for i in range(actual_pcs)]
    n_extra = 0
    if extra is not None:
        E = np.asarray(extra, float).reshape(n_t, -1)
        sd = E.std(0, ddof=0)
        keep = sd > 1e-12                            # drop zero-variance (constant) covariates
        E = E[:, keep]
        if E.shape[1]:
            E = (E - E.mean(0)) / E.std(0, ddof=0)
            blocks.append(E)
            n_extra = int(E.shape[1])
            cols += [f"cov{i+1}" for i in range(n_extra)]
    if len(blocks) == 1:                             # nothing usable => legacy intercept-only
        return None, dict(policy="none", n_pcs=0, n_extra=0, columns=cols, rank=1)
    C = np.column_stack(blocks)
    rank = int(np.linalg.matrix_rank(C))
    if rank < C.shape[1]:                            # rank-deficient covariate block is a config error
        raise ValueError(f"covariate block is rank-deficient (rank {rank} < {C.shape[1]} cols "
                         f"{cols[1:]}); covariates must be linearly independent (a covariate may "
                         "duplicate a genotype PC or another covariate)")
    meta = dict(policy=f"pcs={actual_pcs}+extra={n_extra}", n_pcs=actual_pcs, n_extra=n_extra,
                n_pcs_requested=int(n_pcs), columns=cols, n_cols=int(C.shape[1]), rank=rank)
    return C, meta


def run_pair_scan(
    subdata: dict[str, SubgenomeData],
    pairs: list[tuple],                 # list of dict {sub: gene_id} OR (sx, gx, sy, gy)
    y_raw: np.ndarray,
    sample_idx: np.ndarray,             # rows (into subdata samples) for valid phenotype
    *,
    cap: int = 150,
    transform: str = "INT",
    perm_B: int = 2000,
    n_jobs: int = 8,
    seed: int = 7,
    pair_subs: tuple = None,            # (sx, sy) for 2-col pair tuples
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
    min_snp: int = 1,
    pair_weights: dict = None,          # {(gx,gy): w}  y-INDEPENDENT prior (DL/HEB), frozen
    covariates: dict = None,            # {n_pcs:int, extra:(n_t,q) array} fixed effects; None=legacy
    dominance_adjust: bool = False,     # add per-gene b^2 covariates to the interaction test
    primary_weighting: str = "unweighted",   # which weighting spends alpha
    primary_multiplicity: str = "bonferroni",  # bonferroni | permutation_minp
    inferential: bool = True,           # False => sensitivity run, emits no rejection set
    full_dump_path: str = None,         # if set, write FULL per-pair ranking TSV (else top-N only)
    burden_dump_path: str = None,       # if set, write per-sample burdens for the top-K pairs
    top_k_burden: int = 5,              # number of top pairs to export burdens for
) -> InteractResult:
    """Whitened per-pair burden-product interaction scan with ACAT + permutation calibration.

    Optional ``pair_weights`` (y-independent DL/HEB priors) enable weighted-hypothesis testing
    (weighted Bonferroni p_i < alpha*w_i/G and weighted ACAT), reported alongside the unweighted
    results; FWER stays controlled because weights do not depend on y. Optional ``covariates``
    (y-independent genotype PCs and/or extra fixed effects) enter both the null-LMM mean model and
    the per-pair GLS design; ``None`` => intercept-only. Permutation switches to Freedman-Lane
    (residualize on C, permute residuals), which reduces to y-shuffle when C is intercept-only."""
    from joblib import Parallel, delayed

    if primary_weighting not in ("unweighted", "weighted"):
        raise ValueError("primary_weighting must be 'unweighted' or 'weighted'")
    if primary_multiplicity not in ("bonferroni", "permutation_minp"):
        raise ValueError("primary_multiplicity must be 'bonferroni' or 'permutation_minp'")
    if inferential and primary_multiplicity == "permutation_minp":
        if primary_weighting == "weighted":
            raise ValueError("weighted permutation min-P is not implemented; the permuted "
                             "statistic would have to be the weighted one")
        if not perm_B or perm_B < PAIRWISE_BURDEN_FORMAL_PERMUTATION_MIN_B:
            raise ValueError(
                "primary_multiplicity='permutation_minp' needs "
                f"perm_B >= {PAIRWISE_BURDEN_FORMAL_PERMUTATION_MIN_B} for formal inference; "
                f"got {perm_B}. Use inferential=False for a >=19-replicate QA run")
    rng = np.random.default_rng(seed)
    subs = list(subdata.keys())
    n_t = sample_idx.size
    kernels = {s: _build_grm(subdata[s], sample_idx, grm_method, maf_min) for s in subs}
    # fixed-effect covariate block C = [1, PCs, extra] (None = intercept-only)
    cov_meta = dict(policy="none")
    C = None
    if covariates:
        C, cov_meta = build_covariate_block(kernels, n_t, n_pcs=int(covariates.get("n_pcs", 0)),
                                            extra=covariates.get("extra"))

    # build paired burden matrices BX (subgenome sx) / BY (subgenome sy)
    sx, sy = pair_subs
    bx_cols, by_cols, kept_pairs = [], [], []
    nsnp_x, nsnp_y = [], []                          # callable SNP count per gene (= len snp_idx)
    for p in pairs:
        gx, gy = p
        if (gx in subdata[sx].gene_snp and gy in subdata[sy].gene_snp
                and np.asarray(subdata[sx].gene_snp[gx]).size >= min_snp
                and np.asarray(subdata[sy].gene_snp[gy]).size >= min_snp):
            bx_cols.append(block_burden_capped(subdata[sx].X, subdata[sx].gene_snp[gx], cap, rng)[sample_idx])
            by_cols.append(block_burden_capped(subdata[sy].X, subdata[sy].gene_snp[gy], cap, rng)[sample_idx])
            kept_pairs.append((gx, gy))
            nsnp_x.append(int(np.asarray(subdata[sx].gene_snp[gx]).size))
            nsnp_y.append(int(np.asarray(subdata[sy].gene_snp[gy]).size))
    G = len(kept_pairs)
    if G < 1:
        raise ValueError(
            f"no homoeolog pairs retained with >= {min_snp} SNPs in both copies")
    BX = scols_safe(np.column_stack(bx_cols))
    BY = scols_safe(np.column_stack(by_cols))

    # y-independent prior weights aligned to kept pairs, normalized to sum G (missing -> 1). Invalid
    # weights raise instead of being silently rewritten to 1.0, which would fake a uniform prior.
    w = None
    if pair_weights:
        w = _normalize_weights([pair_weights.get(kp, 1.0) for kp in kept_pairs], G, "pair weights")
    if primary_weighting == "weighted" and w is None:
        raise ValueError("primary_weighting='weighted' requires pair_weights; otherwise the "
                         "unweighted rejections are suppressed and no primary procedure runs")

    y = rank_int(y_raw) if transform == "INT" else y_raw.astype(float)
    Wh, cv = whiten_multi(kernels, y, X=C, seed=42)
    # estimability is decided on the RAW design so the tested family cannot move with y through the
    # fitted whitener; the same mask is then used by the observed scan and every permutation
    design_mask, design_excl = pairwise_design_mask(BX, BY, C=C, dominance_adjust=dominance_adjust)
    pv, diag = pairwise_pvals(Wh, y, BX, BY, C=C, dominance_adjust=dominance_adjust,
                              fixed_mask=design_mask, return_diag=True)
    n_late = int((design_mask & ~np.isfinite(pv)).sum())
    if n_late:
        raise ValueError(f"{n_late} pairs estimable on the raw design lost their statistic after "
                         "whitening; the inferential family would be incomplete")
    # non-estimable pairs stay NaN: acat/lambda_gc drop them, and folding them to 1.0 would enter
    # them into those combinations as real maximally-non-significant observations. The mask is a
    # property of the design, so the permutation family is frozen to exactly this set.
    est = design_mask
    if (inferential and primary_weighting == "weighted" and w is not None
            and not (est & (w > 0)).any()):
        raise ValueError("every estimable pair has weight zero: no hypothesis receives any alpha "
                         "under weighted primary")
    if not est.any():
        raise ValueError("no estimable pair in this scan; an empirical p computed against an "
                         "undefined observed statistic would report the smallest possible value")
    if diag["failures"]:
        raise ValueError(
            f"{len(diag['failures'])} design-valid pairs produced no statistic "
            f"({sorted(set(diag['failures'].values()))}); this is an analysis failure, not an "
            "unestimable hypothesis, and silently retiring them would shrink the tested family")
    n_valid = int(est.sum())
    p_acat_obs = acat(pv)
    minp_obs = float(pv[est].min()) if n_valid else float("nan")
    lam_obs = lambda_gc(pv)
    # the denominator is the PLANNED family: estimability is design-determined, but shrinking the
    # denominator to the tests that happened to be runnable can only loosen the threshold, so the
    # conservative count is authoritative and the exact one is reported alongside.
    bonf = 0.05 / G
    order = [int(i) for i in np.argsort(np.where(est, pv, np.inf)) if est[i]]
    # only the predeclared procedure emits rejections: two alpha-level procedures over the same
    # hypotheses do not control alpha if the reader may take whichever one rejects
    bonf_is_primary = inferential and primary_multiplicity == "bonferroni"
    analytic_sig = [dict(pair=kept_pairs[i], p=float(pv[i]),
                         p_adjusted_bonferroni=float(min(pv[i] * G, 1.0)))
                    for i in order if pv[i] < bonf]
    sig = (analytic_sig
           if (bonf_is_primary and primary_weighting == "unweighted") else None)
    top = [dict(pair=kept_pairs[i], p=float(pv[i])) for i in order[:5]]
    estimability = dict(ESTIMABILITY_POLICY, decided_on="raw_design", n_planned=int(G),
                        n_valid=n_valid, n_unestimable=int(G - n_valid), n_late_fail=n_late,
                        per_test_alpha_valid=float(0.05 / n_valid) if n_valid else None,
                        excluded=[dict(pair=kept_pairs[i], reason=r)
                                  for i, r in sorted(design_excl.items())])

    # per-gene single-burden marginal p (the "invisible to single-locus" contrast) and gene
    # coordinates are computed once here when any dump is requested (reuse Wh/C from above).
    if full_dump_path or burden_dump_path:
        pmx, dmx = marginal_pvals(Wh, y, BX, C=C, return_diag=True)
        pmy, dmy = marginal_pvals(Wh, y, BY, C=C, return_diag=True)
        if dmx["failures"] or dmy["failures"]:
            raise ValueError(
                "single-burden marginal test failed on a design-valid gene "
                f"({sorted(set(dmx['failures'].values()) | set(dmy['failures'].values()))}); the "
                "interaction-vs-marginal contrast would silently read NA")
        coords = [(_gene_coord(subdata[sx], subdata[sx].gene_snp[gx]),
                   _gene_coord(subdata[sy], subdata[sy].gene_snp[gy]))
                  for gx, gy in kept_pairs]

    if full_dump_path:
        # Cache the y-independent/descriptive columns now; the TSV is written after permutation so
        # its primary_sig and adjusted-p columns reflect the declared multiplicity procedure.
        nx, ny = np.asarray(nsnp_x), np.asarray(nsnp_y)
        sort_key = np.where(est, pv, np.inf)              # unestimable rows stay, but sort last
        _, rank_of, tie_of = _rank_with_ties(sort_key, kept_pairs)
        nl = _neglog10(pv)
        nlmx, nlmy = _neglog10(pmx), _neglog10(pmy)
        cbin = _decile_bin(nx + ny)

    if burden_dump_path:
        # per-sample burdens + phenotype for the top-K pairs -> A×D interaction-surface plot.
        # resid removes the covariate fit (== y - mean when intercept-only).
        if C is None:
            resid_y = y - float(np.mean(y))
        else:
            b_c, *_ = np.linalg.lstsq(C, y, rcond=None)
            resid_y = y - C @ b_c
        brows = []
        for kr, i in enumerate(order[:max(int(top_k_burden), 1)]):
            gx, gy = kept_pairs[int(i)]
            for s in range(n_t):
                brows.append([kr, gx, gy, sx, sy, int(s),
                              repr(float(BX[s, i])), repr(float(BY[s, i])),
                              repr(float(y[s])), repr(float(resid_y[s]))])
        _write_ranking_tsv(burden_dump_path,
                           ["pair_rank", "gene_x", "gene_y", "sub_x", "sub_y", "sample_row",
                            "burden_x", "burden_y", "phenotype", "resid"], brows)

    weighted = None
    if w is not None:
        wsig = ([dict(pair=kept_pairs[i], p=float(pv[i]), weight=float(w[i]),
                      p_weighted=float(pv[i] / w[i]) if w[i] > 0 else float("inf"))
                 for i in order if w[i] > 0 and pv[i] < bonf * w[i]]
                if (bonf_is_primary and primary_weighting == "weighted") else None)
        n_cov = int(sum(kp in pair_weights for kp in kept_pairs))
        weighted = dict(
            role=("primary" if primary_weighting == "weighted" else
                  "exploratory_no_rejections_emitted"),
            acat_weighted=float(acat_weighted(pv, w)),
            bonferroni_n_sig=(len(wsig) if wsig is not None else None), sig=wsig,
            audit=dict(n_pairs=int(G), n_covered=n_cov, n_missing_default1=int(G - n_cov),
                       weight_min=float(w.min()), weight_mean=float(w.mean()),
                       weight_max=float(w.max())),
            note=("weighted Bonferroni controls pair-level FWER only if this procedure was "
                  "predeclared as THE primary one -- reporting it beside the unweighted rejections "
                  "and taking either is a union of two alpha-level procedures. Weighted ACAT is a "
                  "y-independent prior-weighted omnibus. Valid ONLY if weights were frozen and "
                  "are y-independent (DL zero-shot / HEB). Missing pairs default to weight 1; after "
                  "normalization (sum w = G) up-weighting prioritized pairs reallocates alpha from "
                  "the rest (their effective weight < 1)."))

    # Freedman-Lane permutation: y* = C@beta_hat + (y - C@beta_hat)[perm]. With C=None this is
    # exactly y[perm] (y-shuffle); with covariates it preserves the covariate mean structure.
    if C is None:
        fl_fit, fl_resid = None, None
    else:
        b_fl, *_ = np.linalg.lstsq(C, y, rcond=None)
        fl_fit = C @ b_fl
        fl_resid = y - fl_fit

    def _perm(seed_i):
        r = np.random.default_rng(seed_i)
        perm = r.permutation(n_t)
        ys = y[perm] if C is None else (fl_fit + fl_resid[perm])
        try:
            Whp, _ = whiten_multi(kernels, ys, X=C, seed=seed_i % 100000)
            pp = pairwise_pvals(Whp, ys, BX, BY, C=C, dominance_adjust=dominance_adjust,
                                fixed_mask=est)
            # the null family is frozen to the observed estimable set. A replicate in which a
            # mask-true test yields no statistic is counted as MAXIMALLY extreme rather than
            # dropped: those failures are degenerate fits (t -> inf), i.e. exactly the tail of the
            # null, so deleting them would shrink the numerator of the empirical p and raise the
            # min-p cutoff -- anticonservative in both directions.
        except np.linalg.LinAlgError:
            # the null model itself degenerated for this replicate: conservative extreme tuple.
            # Any OTHER exception is a bug or a data error and must not be laundered into evidence.
            return 0.0, float("nan"), 0.0, 1
        if not np.isfinite(pp[est]).all():
            return 0.0, float("nan"), 0.0, 1
        return acat(pp), lambda_gc(pp), float(pp[est].min()), 0

    p_acat_emp = float("nan")
    lam_perm = float("nan")
    minp_perm_emp = float("nan")
    minp_perm_threshold = float("nan")
    minp_perm_rejected = None
    p_adjusted_perm = np.full(G, np.nan, dtype=float)
    perm_status = dict(status="not_run", B_requested=int(perm_B or 0), n_used=0, n_degenerate=0,
                       note="no resampling was run; empirical p fields are null, not 1.0")
    if perm_B and perm_B > 0:
        res = Parallel(n_jobs=n_jobs)(delayed(_perm)(900000 + i) for i in range(perm_B))
        n_deg = int(sum(r[3] for r in res))
        perm_status = dict(
            status="completed" if n_deg == 0 else "completed_with_degenerate_replicates",
            B_requested=int(perm_B), n_used=len(res), n_degenerate=n_deg,
            note=None if n_deg == 0 else
            f"{n_deg} replicate(s) had a design-valid test with no statistic and were counted as "
            "maximally extreme (conservative); a large count means the null model is degenerate")
        if res:
            ap = np.array([r[0] for r in res])
            lam = np.array([r[1] for r in res])
            mp = np.array([r[2] for r in res])                  # permuted min pair-p distribution
            if np.isfinite(p_acat_obs):
                p_acat_emp = float((1 + int((ap <= p_acat_obs).sum())) / (len(ap) + 1))
            lam_perm = float(np.nanmedian(lam)) if np.isfinite(lam).any() else float("nan")
            # experiment-wide FWER from the SAME plus-one test as the cutoff, so the two can never
            # disagree: reject iff (1 + #{null <= obs})/(B+1) <= alpha, i.e. iff obs is strictly
            # below the k-th smallest null value with k = floor(alpha*(B+1)). An interpolated
            # quantile is not a valid permutation cutoff and mishandles ties at zero.
            if np.isfinite(minp_obs):
                minp_perm_emp = float((1 + int((mp <= minp_obs).sum())) / (len(mp) + 1))
            k = int(np.floor(0.05 * (len(mp) + 1)))
            minp_perm_threshold = float(np.sort(mp)[k - 1]) if k >= 1 else float("nan")
            mp_sorted = np.sort(mp)
            p_adjusted_perm[est] = (
                1 + np.searchsorted(mp_sorted, pv[est], side="right")) / (len(mp) + 1)
            # only the declared procedure emits a rejection decision: Bonferroni and permutation
            # min-P each spend the full alpha over the same hypotheses
            if (inferential and primary_multiplicity == "permutation_minp"
                    and primary_weighting == "unweighted"):
                minp_perm_rejected = bool(np.isfinite(minp_perm_emp) and minp_perm_emp <= 0.05)

    permutation_is_primary = (
        inferential and primary_multiplicity == "permutation_minp"
        and primary_weighting == "unweighted")
    if permutation_is_primary:
        sig = [dict(pair=kept_pairs[i], p=float(pv[i]),
                    p_adjusted_permutation_minp=float(p_adjusted_perm[i]))
               for i in order if p_adjusted_perm[i] <= 0.05]
    top = [dict(pair=kept_pairs[i], p=float(pv[i]),
                **({"p_adjusted_permutation_minp": float(p_adjusted_perm[i])}
                   if np.isfinite(p_adjusted_perm[i]) else {}))
           for i in order[:5]]
    primary_index = {
        kept_pairs.index(tuple(hit["pair"])) for hit in (sig or [])
    } if sig is not None else set()

    permutation_fwer = dict(
        alpha=0.05,
        method="freedman_lane_permutation_minp_plus_one",
        inferential=bool(permutation_is_primary),
        rejected=(bool(sig) if permutation_is_primary else None),
        n_rejected=(len(sig) if permutation_is_primary else None),
        empirical_p=(float(minp_perm_emp) if np.isfinite(minp_perm_emp) else None),
        threshold=(float(minp_perm_threshold)
                   if np.isfinite(minp_perm_threshold) else None),
        threshold_comparator="strict_less_than",
        n_degenerate_replicates=int(perm_status["n_degenerate"]),
        sig=(sig if permutation_is_primary else None),
    )

    if full_dump_path:
        rows = [[int(rank_of[i]), kept_pairs[i][0], kept_pairs[i][1], sx, sy,
                 _tsv_p(pv[i]), _tsv_p(nl[i]), _tsv_p(p_adjusted_perm[i]),
                 int(nx[i]), int(ny[i]), int(nx[i] + ny[i]),
                 int(cbin[i]), int(tie_of[i]), int(not est[i]),
                 (int(i in primary_index) if sig is not None else "NA"),
                 int(est[i] and pv[i] < bonf),
                 "NA", "NA", "NA",
                 coords[i][0][0], coords[i][0][1], coords[i][1][0], coords[i][1][1],
                 _tsv_p(pmx[i]), _tsv_p(pmy[i]),
                 _tsv_p(nlmx[i]), _tsv_p(nlmy[i])]
                for i in np.argsort(sort_key, kind="stable")]
        _write_ranking_tsv(
            full_dump_path,
            ["rank", f"gene_{sx}", f"gene_{sy}", "sub_x", "sub_y", "p_interaction",
             "neglog10p", "p_adjusted_permutation_minp",
             f"n_snp_{sx}", f"n_snp_{sy}", "n_snp_pair",
             "callable_snp_decile", "tie_group", "p_unestimable", "primary_sig",
             "analytic_screen_sig", f"gene_len_{sx}", f"gene_len_{sy}",
             "gene_len_pair_sum", "chrom_x", "pos_x", "chrom_y", "pos_y",
             "p_marginal_x", "p_marginal_y", "neglog10p_marginal_x",
             "neglog10p_marginal_y"], rows)

    return InteractResult(
        trait="", transform=transform, n=int(n_t), G=int(G),
        pair_acat=float(p_acat_obs), pair_acat_emp=p_acat_emp, min_p=minp_obs,
        lambda_gc_obs=float(lam_obs), lambda_gc_perm_median=lam_perm,
        bonferroni_alpha=float(bonf), n_sig=(len(sig) if sig is not None else None), sig=sig,
        top=top,
        sigma_hat={s: float(cv.get(s, 0.0)) for s in subs} | {"e": float(cv.get("e", 0.0))},
        weighted=weighted, covariates=cov_meta,
        minp_perm_emp=minp_perm_emp, minp_perm_threshold=minp_perm_threshold,
        minp_perm_rejected=minp_perm_rejected,
        n_planned=int(G), n_valid=n_valid, n_unestimable=int(G - n_valid),
        estimability=estimability, permutation=perm_status,
        model_diagnostics={"permutation_fwer": permutation_fwer},
        analytic_screen_n=len(analytic_sig), analytic_screen_sig=analytic_sig,
        inference_plan=dict(
            primary_weighting=primary_weighting, primary_multiplicity=primary_multiplicity,
            inferential=bool(inferential),
            note="rejection fields are emitted for the predeclared procedure only and are null "
                 "elsewhere -- a suppressed procedure must not be readable as one that found "
                 "nothing. Bonferroni and permutation min-P each spend the full alpha over the "
                 "same hypotheses, so exactly one of them is inferential"))


def _batch_nested_f(Yw: np.ndarray, Xred: np.ndarray, Xadd: np.ndarray) -> np.ndarray:
    """Whitened nested-F p for the ``Xadd`` block over EVERY column of ``Yw`` at once.

    The designs depend only on genotypes, so a single orthonormalization serves all phenotype columns
    (the observed phenotype plus every bootstrap null), which is what makes the bootstrap affordable.
    With one added column F == t^2, so this reproduces the per-pair burden/PC1 t-test exactly and
    generalizes to the multi-column kernel block. Non-estimable interaction blocks return NaN (not
    p=1) so ACAT omits them rather than injecting a spurious large-p term."""
    from scipy.linalg import orth

    n = Yw.shape[0]
    Qr = orth(Xred)
    Xadd_r = Xadd - Qr @ (Qr.T @ Xadd) if Qr.size else Xadd
    Qa = orth(Xadd_r)
    dfn = Qa.shape[1]
    dfd = n - Qr.shape[1] - dfn
    if dfn < 1 or dfd < 1:
        return np.full(Yw.shape[1], np.nan)
    # Explicit residual norms avoid catastrophic cancellation from
    # ||Y||^2 - ||Q'Y||^2 when the fitted model explains almost all variation.
    Yres = Yw - Qr @ (Qr.T @ Yw) if Qr.size else Yw.copy()
    added_ss = ((Qa.T @ Yres) ** 2).sum(0)
    full_resid = Yres - Qa @ (Qa.T @ Yres)
    rss_f = (full_resid ** 2).sum(0)
    denom = rss_f / dfd
    bad = denom <= 1e-300
    f = added_ss / dfn / np.where(bad, 1.0, denom)
    p = stats.f.sf(np.maximum(f, 0.0), dfn, dfd)
    # A truly perfect full-model fit with positive added signal has p=0;
    # an entirely degenerate response remains undefined.
    return np.where(bad, np.where(added_ss > 1e-300, 0.0, np.nan), p)


def _bootstrap_minp_calibration(
    p_obs: np.ndarray,
    p_null: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict:
    """Single-step plus-one min-P calibration with conservative degeneracy handling.

    ``p_null`` contains only the fixed, design-estimable hypothesis family.
    If any statistic in a null replicate is non-finite, that replicate's
    minimum is set to zero.  This cannot create a false discovery; silently
    dropping the failed test could make the null minimum too large.
    """
    p_obs = np.asarray(p_obs, float)
    p_null = np.asarray(p_null, float)
    if p_null.ndim != 2:
        raise ValueError("p_null must be a hypothesis-by-bootstrap matrix")
    if p_null.shape[0] != p_obs.size:
        raise ValueError(
            "p_obs and p_null must contain the same fixed hypothesis family")
    B = int(p_null.shape[1])
    if B < 1:
        raise ValueError("bootstrap min-P calibration requires at least one replicate")
    if not np.isfinite(p_obs).all():
        raise ValueError("observed min-P family contains non-finite statistics")

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
    rejected = bool(empirical_p <= alpha)
    rejected_local = (
        np.flatnonzero(p_obs < threshold).astype(int).tolist()
        if threshold is not None else []
    )
    if rejected != bool(rejected_local):
        raise RuntimeError(
            "bootstrap global decision and single-step rejection set disagree")
    return {
        "alpha": float(alpha),
        "method": "parametric_bootstrap_minp_plus_one",
        "B": B,
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


OMNIB_COMPONENT_NAMES = ("minor_burden", "pc1", "kernel_hadamard")


@dataclass
class OmniBFamilyScores:
    """Observed/bootstrap omniB matrices for one predeclared group family.

    Rows in ``edge_p`` retain the deterministic unique-edge order from
    :func:`expand_pair_edges`; rows in ``group_p`` retain the master-table
    order.  Column zero is observed and the remaining columns are shared null
    responses.  Non-estimable edges remain present as all-NaN rows so family
    membership never depends on the phenotype.
    """

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


def _omnib_edge_design_estimable(gsx, gsy, C: np.ndarray) -> bool:
    """Predeclare whether at least one omniB component has target rank.

    This check is genotype/covariate-only and happens before the phenotype is
    whitened.  It separates structural non-estimability (which keeps an NaN
    row) from a numerical failure after whitening (which aborts the scan).
    """
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


def _score_omnib_family(
    subdata: dict[str, SubgenomeData],
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
    """Score every unique edge once and derive every group from that matrix.

    One all-subgenome null fit and one observed/bootstrap response matrix are
    shared by all directions.  Feature blocks are cached by
    ``(subgenome, gene_id)``.  The returned matrices preserve the complete
    predeclared family, including structurally non-estimable edges.
    """
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
    missing_subgenomes = [s for s in family.subgenomes if s not in subdata]
    if missing_subgenomes:
        raise ValueError(
            "master family references missing subgenomes: "
            + ", ".join(missing_subgenomes))

    expanded = expand_pair_edges(family)
    if not expanded.edges:
        raise ValueError("master homoeolog family contains no pair edges")

    # Include every supplied subgenome kernel in the common null, even when a
    # compatibility pair wrapper selects only one edge direction.
    subs = list(subdata)
    n_t = sample_idx.size
    kernels = {
        s: _build_grm(subdata[s], sample_idx, grm_method, maf_min)
        for s in subs
    }
    C = None
    cov_meta = {"policy": "none"}
    if covariates:
        C, cov_meta = build_covariate_block(
            kernels, n_t, n_pcs=int(covariates.get("n_pcs", 0)),
            extra=covariates.get("extra"))
    C_design = np.ones((n_t, 1)) if C is None else np.asarray(C, float).reshape(n_t, -1)
    y = rank_int(y_raw) if transform == "INT" else y_raw.astype(float)
    W, V, beta, cv = null_lmm_fit(kernels, y, C, seed=42)
    Cw = W @ C_design

    rng = np.random.default_rng(bootstrap_seed)
    gated: dict[tuple[str, str], np.ndarray] = {}
    feats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _gated_snps(sub: str, gene: str) -> np.ndarray | None:
        key = (sub, gene)
        if key in gated:
            return gated[key]
        if gene not in subdata[sub].gene_snp:
            return None
        idx = np.asarray(subdata[sub].gene_snp[gene], int)
        mu = np.nanmean(subdata[sub].X[np.ix_(sample_idx, idx)], axis=0) / 2.0
        gated[key] = idx[np.minimum(mu, 1.0 - mu) >= burden_maf]
        return gated[key]

    def _feature(sub: str, gene: str, idx: np.ndarray):
        key = (sub, gene)
        if key not in feats:
            Xg = subdata[sub].X[np.ix_(sample_idx, idx)]
            local_idx = np.arange(idx.size)
            burden = block_burden_capped(
                Xg, local_idx, cap, rng, minor=True).reshape(-1, 1)
            pcs = gene_pc_scores(Xg, local_idx, cap, rng, n_pc)
            feats[key] = burden, pcs[:, :1], pcs
        return feats[key]

    edge_estimable = np.zeros(len(expanded.edges), bool)
    for ei, edge in enumerate(expanded.edges):
        ix = _gated_snps(edge.sub_x, edge.gene_x)
        iy = _gated_snps(edge.sub_y, edge.gene_y)
        if ix is None or iy is None or ix.size < min_snp or iy.size < min_snp:
            continue
        fx = _feature(edge.sub_x, edge.gene_x, ix)
        fy = _feature(edge.sub_y, edge.gene_y, iy)
        edge_estimable[ei] = _omnib_edge_design_estimable(fx, fy, C_design)

    ncol = bootstrap_B + 1
    Yall = np.empty((n_t, ncol), float)
    Yall[:, 0] = y
    if bootstrap_B:
        ystars, _W2, _cv2 = null_replicates(
            kernels, y, C=C, B=bootstrap_B, method="bootstrap",
            seed=bootstrap_seed, null_fit=(W, V, beta, cv))
        Yall[:, 1:] = np.column_stack(
            [np.asarray(value, float) for value in ystars])
    Yw = W @ Yall

    from joblib import Parallel, delayed

    edge_p = np.full((len(expanded.edges), ncol), np.nan)
    edge_components_obs = np.full(
        (len(expanded.edges), len(OMNIB_COMPONENT_NAMES)), np.nan)
    valid_indices = np.flatnonzero(edge_estimable)

    def _block(bounds):
        lo, hi = bounds
        indices = valid_indices[lo:hi]
        out = np.full((indices.size, ncol), np.nan)
        observed = np.full((indices.size, len(OMNIB_COMPONENT_NAMES)), np.nan)
        for local, edge_index in enumerate(indices):
            edge = expanded.edges[int(edge_index)]
            components = _omnib_components_over_Y(
                W, Yw, Cw,
                feats[(edge.sub_x, edge.gene_x)],
                feats[(edge.sub_y, edge.gene_y)])
            out[local] = np.asarray(
                [acat(components[:, col]) for col in range(ncol)], float)
            observed[local] = components[:, 0]
        return indices, out, observed

    if valid_indices.size:
        step = max(1, valid_indices.size // (n_jobs * 8))
        blocks = [
            (lo, min(lo + step, valid_indices.size))
            for lo in range(0, valid_indices.size, step)
        ]
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_block)(block) for block in blocks)
        for indices, values, components in results:
            edge_p[indices] = values
            edge_components_obs[indices] = components

    failed_observed = edge_estimable & ~np.isfinite(edge_p[:, 0])
    if failed_observed.any():
        failed_ids = [
            expanded.edges[i].edge_id
            for i in np.flatnonzero(failed_observed)[:5]
        ]
        raise RuntimeError(
            "design-valid edge produced a post-whitening non-finite observed "
            f"omniB score: {', '.join(failed_ids)}")

    group_p = np.full((len(family.group_ids), ncol), np.nan)
    group_partial = np.zeros(len(family.group_ids), bool)
    for gi, edge_indices in enumerate(expanded.group_edge_indices):
        idx = np.asarray(edge_indices, int)
        valid_count = int(edge_estimable[idx].sum())
        group_partial[gi] = 0 < valid_count < idx.size
        if idx.size == 1:
            # Required bit-exact two-copy reduction; a trigonometric ACAT
            # round-trip need not reproduce the input's final bits.
            group_p[gi] = edge_p[idx[0]]
        else:
            for col in range(ncol):
                group_p[gi, col] = acat(edge_p[idx, col])
    group_estimable = np.isfinite(group_p[:, 0])

    scores = OmniBFamilyScores(
        edge_p=edge_p,
        group_p=group_p,
        edge_components_obs=edge_components_obs,
        edge_estimable=edge_estimable,
        group_estimable=group_estimable,
        W=W,
        y=y,
        covariance_components={str(k): float(v) for k, v in cv.items()},
        group_partial=group_partial,
        gated_snp=gated,
        feature_cache=feats,
        covariate_block=C,
        covariate_metadata=cov_meta,
    )
    return scores, expanded


def _omnib_components_over_Y(Wh, Yw, Cw, gsx, gsy):
    """Return the three omniB component p-values (component x phenotype).

    Keeping this matrix available is essential for interpretation: an omniB
    discovery is evidence from the *combined* test and must not automatically
    be described as a burden-product interaction.  The smallest component is
    descriptive only; inference remains on the predeclared omniB p-value.
    """
    bx, p1x, PX = gsx
    by, p1y, PY = gsy
    comps = []
    for ax, ay in ((bx, by), (p1x, p1y), (PX, PY)):
        Xred = np.column_stack([Cw, Wh @ ax, Wh @ ay])
        cross = (ax[:, :, None] * ay[:, None, :]).reshape(ax.shape[0], -1)
        comps.append(_batch_nested_f(Yw, Xred, Wh @ cross))
    return np.vstack(comps)


def _omnib_pair_over_Y(Wh, Yw, Cw, gsx, gsy):
    """omniB = ACAT(minor-burden, PC1xPC1, kernel-Hadamard) for one pair over all Yw columns.

    ``gsx``/``gsy`` are (minor_burden[n,1], pc1[n,1], pc_block[n,n_pc]) for the two genes, built once
    (genotype-only). The reduced model of each component is its two constituent main effects plus the
    fixed-effect block ``Cw`` (whitened), so every component is tested CONDITIONAL on its main
    effects, and all three are encoding-invariant."""
    P = _omnib_components_over_Y(Wh, Yw, Cw, gsx, gsy)
    return np.array([acat(P[:, j]) for j in range(P.shape[1])])


def _omnib_component_record(pvals) -> dict:
    """JSON-safe descriptive component record for one observed omniB unit."""
    p = np.asarray(pvals, float)
    finite = np.isfinite(p)
    driver = None
    driver_p = None
    if finite.any():
        idx = int(np.nanargmin(np.where(finite, p, np.nan)))
        driver = OMNIB_COMPONENT_NAMES[idx]
        driver_p = float(p[idx])
    return {
        "component_p": {
            name: (float(value) if np.isfinite(value) else None)
            for name, value in zip(OMNIB_COMPONENT_NAMES, p, strict=True)
        },
        "smallest_component": driver,
        "smallest_component_p": driver_p,
    }


def _omnib_component_summary(component_p: np.ndarray) -> dict:
    """Compact all-units component audit (the full values belong in the ranking TSV)."""
    P = np.asarray(component_p, float)
    counts = {name: 0 for name in OMNIB_COMPONENT_NAMES}
    for row in P:
        finite = np.isfinite(row)
        if finite.any():
            idx = int(np.nanargmin(np.where(finite, row, np.nan)))
            counts[OMNIB_COMPONENT_NAMES[idx]] += 1
    return {
        "names": list(OMNIB_COMPONENT_NAMES),
        "smallest_component_counts": counts,
        "n_units": int(P.shape[0]),
        "n_component_estimable": {
            name: int(np.isfinite(P[:, i]).sum())
            for i, name in enumerate(OMNIB_COMPONENT_NAMES)
        },
        "interpretation": (
            "The smallest component is descriptive, not a separately calibrated discovery. "
            "An omniB-significant unit is an omnibus interaction; call it a burden-product "
            "interaction only when the minor_burden component is itself the prespecified target "
            "and is supported at its declared threshold."
        ),
    }


OmniBFamilyScores = _family_score.OmniBFamilyScores  # noqa: F811
_score_omnib_family = _family_score.score_omnib_family
_omnib_components_over_Y = _family_score.omnib_components_over_Y
_bootstrap_minp_calibration = _family_score.bootstrap_minp_calibration
run_group_scan_omnib = _family_score.run_group_scan_omnib


def run_pair_scan_omnib(
    subdata: dict[str, SubgenomeData],
    pairs: list,
    y_raw: np.ndarray,
    sample_idx: np.ndarray,
    *,
    cap: int = 150,
    n_pc: int = 3,
    transform: str = "INT",
    bootstrap_B: int = 2000,
    bootstrap_seed: int = 2026,
    n_jobs: int = 8,
    pair_subs: tuple = None,
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
    burden_maf: float = 0.01,
    min_snp: int = 3,
    covariates: dict = None,
    tail_thresholds: tuple = (1e-2, 1e-3, 1e-4, 1e-5),
    inferential: bool = True,           # False => sensitivity run, emits no rejection set
    full_dump_path: str = None,         # full omniB ranking + all three component p-values
    burden_dump_path: str = None,       # per-sample minor burdens for top-ranked pairs
    top_k_burden: int = 5,
    primary_multiplicity: str = "bonferroni",
) -> InteractResult:
    """Encoding-invariant primary interaction scan (omniB) with kinship-preserving bootstrap.

    Per pair omniB = ACAT(minor-allele burden product, PC1xPC1 product, low-rank kernel-Hadamard),
    each strictly invariant to a REF/ALT swap. The whitener is built ONCE from the subgenome GRMs;
    the parametric bootstrap draws null phenotypes ``y* = C.beta_hat + V^{1/2}z`` (kinship exact) and
    the SAME whitener rescans every pair for each ``y*`` — never refitting REML on a bootstrap
    phenotype (that is the permutation flaw). Experiment-wide FWER is the empirical p of the observed
    minimum omniB against the bootstrap min-p distribution; the aggregate tail-excess compares the
    observed count of small omniB p-values to its bootstrap null at each threshold."""
    subs = list(subdata.keys())
    sx, sy = pair_subs
    n_t = sample_idx.size
    family = MasterGroupFamily(
        subgenomes=(sx, sy),
        group_ids=tuple(f"pair_{i}" for i in range(len(pairs))),
        genes=tuple(tuple(pair) for pair in pairs),
    )
    scores, expanded = _score_omnib_family(
        subdata, family, y_raw, sample_idx, cap=cap, n_pc=n_pc,
        transform=transform, bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed, n_jobs=n_jobs,
        grm_method=grm_method, maf_min=maf_min, burden_maf=burden_maf,
        min_snp=min_snp, covariates=covariates)
    selected = np.flatnonzero(scores.edge_estimable)
    kept = [
        (expanded.edges[i].gene_x, expanded.edges[i].gene_y)
        for i in selected
    ]
    G = len(kept)
    if G < 1:
        raise ValueError(
            "no homoeolog pairs retained: each copy must be present with "
            f">= {min_snp} SNPs passing burden MAF >= {burden_maf}")
    P = scores.edge_p[selected]
    component_obs = scores.edge_components_obs[selected]
    W = scores.W
    y = scores.y
    cv = scores.covariance_components
    C = scores.covariate_block
    cov_meta = scores.covariate_metadata
    feats = scores.feature_cache
    gated = scores.gated_snp

    p_obs = P[:, 0]
    finite = np.isfinite(p_obs)
    if not finite.any():
        raise ValueError("no estimable pairs (every omniB component was non-estimable)")
    minp_obs = float(np.nanmin(p_obs))
    bonf = 0.05 / G
    order = [int(i) for i in np.argsort(np.where(finite, p_obs, np.inf)) if finite[i]]
    primary_multiplicity = str(primary_multiplicity).lower()
    if primary_multiplicity not in {"bonferroni", "bootstrap_minp"}:
        raise ValueError(
            "pairwise omniB primary_multiplicity must be bonferroni or bootstrap_minp")
    minp_adjusted = np.full(G, np.nan)

    def _hit(i):
        return (dict(
            pair=kept[i], p=float(p_obs[i]),
            p_adjusted_bonferroni=float(min(p_obs[i] * G, 1.0)),
            p_adjusted_bootstrap_minp=(
                float(minp_adjusted[i]) if np.isfinite(minp_adjusted[i]) else None),
        ) | _omnib_component_record(component_obs[i]))

    analytic_indices = [i for i in order if p_obs[i] < bonf]
    p_acat_obs = acat(p_obs)

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
        if primary_multiplicity == "bootstrap_minp":
            minp_boot_rejected = cal["rejected"] if inferential else None
            formal_indices = (
                [int(family_indices[i]) for i in cal["rejected_local"]]
                if inferential else None)
        bootstrap_fwer = {
            key: value for key, value in cal.items()
            if key not in {"rejected_local", "adjusted_p_local"}
        } | {
            "inferential": bool(
                inferential and primary_multiplicity == "bootstrap_minp"),
            "rejected": minp_boot_rejected,
            "n_rejected": (
                len(formal_indices) if formal_indices is not None else None),
            "sig": (
                [_hit(i) for i in formal_indices]
                if formal_indices is not None else None),
            "note": (
                "This bootstrap min-P object is the sole calibrated discovery "
                "layer. Bonferroni fields are a descriptive analytic screen."
                if primary_multiplicity == "bootstrap_minp" else
                "Bootstrap min-P is diagnostic because Bonferroni was the "
                "predeclared discovery layer."
            ),
        }
        tail_excess = {}
        null_family = P[finite, 1:]
        for threshold in tail_thresholds:
            obs_ct = int((p_obs[finite] < threshold).sum())
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
    elif primary_multiplicity == "bootstrap_minp" and inferential:
        raise ValueError(
            "formal pairwise omniB bootstrap_minp requires bootstrap_B >= 1")

    if primary_multiplicity == "bonferroni" and inferential:
        formal_indices = analytic_indices
    analytic_sig = [_hit(i) for i in analytic_indices]
    sig = ([_hit(i) for i in formal_indices]
           if formal_indices is not None else None)
    top = [_hit(i) for i in order[:5]]

    if full_dump_path or burden_dump_path:
        BX = np.column_stack([feats[(sx, gx)][0][:, 0] for gx, _ in kept])
        BY = np.column_stack([feats[(sy, gy)][0][:, 0] for _, gy in kept])
        pmx, dmx = marginal_pvals(W, y, BX, C=C, return_diag=True)
        pmy, dmy = marginal_pvals(W, y, BY, C=C, return_diag=True)
        if dmx["failures"] or dmy["failures"]:
            raise ValueError(
                "single-minor-burden marginal test failed on a design-valid gene; "
                "the omniB interaction-vs-marginal contrast would silently read NA")
        coords = [(_gene_coord(subdata[sx], gated[(sx, gx)]),
                   _gene_coord(subdata[sy], gated[(sy, gy)]))
                  for gx, gy in kept]

    if full_dump_path:
        nx = np.asarray([gated[(sx, gx)].size for gx, _ in kept], int)
        ny = np.asarray([gated[(sy, gy)].size for _, gy in kept], int)
        sort_key = np.where(finite, p_obs, np.inf)
        _, rank_of, tie_of = _rank_with_ties(sort_key, kept)
        nl = _neglog10(p_obs)
        nlmx, nlmy = _neglog10(pmx), _neglog10(pmy)
        cbin = _decile_bin(nx + ny)
        formal_set = set(formal_indices or [])
        analytic_set = set(analytic_indices)
        rows = []
        for i in np.argsort(sort_key, kind="stable"):
            record = _omnib_component_record(component_obs[i])
            cp = record["component_p"]
            rows.append([
                int(rank_of[i]), kept[i][0], kept[i][1], sx, sy,
                _tsv_p(p_obs[i]), _tsv_p(minp_adjusted[i]), _tsv_p(nl[i]),
                _tsv_p(cp["minor_burden"]), _tsv_p(cp["pc1"]),
                _tsv_p(cp["kernel_hadamard"]),
                record["smallest_component"] or "NA",
                _tsv_p(record["smallest_component_p"]),
                int(nx[i]), int(ny[i]), int(nx[i] + ny[i]), int(cbin[i]),
                int(tie_of[i]), int(not finite[i]),
                (int(i in formal_set) if inferential else "NA"),
                int(i in analytic_set),
                coords[i][0][0], coords[i][0][1],
                coords[i][1][0], coords[i][1][1],
                _tsv_p(pmx[i]), _tsv_p(pmy[i]),
                _tsv_p(nlmx[i]), _tsv_p(nlmy[i]),
            ])
        _write_ranking_tsv(
            full_dump_path,
            ["rank", f"gene_{sx}", f"gene_{sy}", "sub_x", "sub_y",
             "p_interaction", "p_adjusted_bootstrap_minp", "neglog10p",
             "p_minor_burden", "p_pc1",
             "p_kernel_hadamard", "smallest_component", "smallest_component_p",
             f"n_snp_{sx}", f"n_snp_{sy}", "n_snp_pair",
             "callable_snp_decile", "tie_group", "p_unestimable", "primary_sig",
             "analytic_screen_sig",
             "chrom_x", "pos_x", "chrom_y", "pos_y",
             "p_marginal_x", "p_marginal_y",
             "neglog10p_marginal_x", "neglog10p_marginal_y"],
            rows)

    if burden_dump_path:
        if C is None:
            resid_y = y - float(np.mean(y))
        else:
            b_c, *_ = np.linalg.lstsq(C, y, rcond=None)
            resid_y = y - C @ b_c
        rows = []
        for kr, i in enumerate(order[:max(int(top_k_burden), 1)]):
            gx, gy = kept[i]
            for sample_row in range(n_t):
                rows.append([
                    kr, gx, gy, sx, sy, sample_row,
                    repr(float(BX[sample_row, i])),
                    repr(float(BY[sample_row, i])),
                    repr(float(y[sample_row])), repr(float(resid_y[sample_row])),
                ])
        _write_ranking_tsv(
            burden_dump_path,
            ["pair_rank", "gene_x", "gene_y", "sub_x", "sub_y", "sample_row",
             "minor_burden_x", "minor_burden_y", "phenotype", "resid"],
            rows)

    return InteractResult(
        trait="", transform=transform, n=int(n_t), G=int(G),
        pair_acat=float(p_acat_obs), pair_acat_emp=float("nan"), min_p=minp_obs,
        lambda_gc_obs=float(lambda_gc(p_obs[finite])), lambda_gc_perm_median=float("nan"),
        bonferroni_alpha=float(bonf), n_sig=(len(sig) if sig is not None else None), sig=sig,
        top=top,
        sigma_hat={s: float(cv.get(s, 0.0)) for s in subs} | {"e": float(cv.get("e", 0.0))},
        weighted=None, covariates=cov_meta,
        n_planned=int(G), n_valid=int(finite.sum()), n_unestimable=int((~finite).sum()),
        statistic="omniB", calibration_method=("bootstrap" if bootstrap_B else "none"),
        bootstrap_B=int(bootstrap_B), bootstrap_seed=int(bootstrap_seed),
        minp_boot_emp=minp_boot_emp, minp_boot_threshold=minp_boot_threshold,
        minp_boot_rejected=minp_boot_rejected,
        tail_excess=tail_excess,
        component_diagnostics=_omnib_component_summary(component_obs),
        analytic_screen_n=len(analytic_sig), analytic_screen_sig=analytic_sig,
        model_diagnostics={"bootstrap_fwer": bootstrap_fwer})


def run_clique_scan_omnib(
    subdata: dict[str, SubgenomeData],
    groups: list,
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
    tail_thresholds: tuple = (1e-2, 1e-3, 1e-4, 1e-5),
    inferential: bool = True,           # False => sensitivity run, emits no rejection set
    full_dump_path: str = None,         # full group ranking + pair/component decomposition
) -> InteractResult:
    """n-subgenome homoeolog-clique omniB scan (triad = n=3 special case; wheat/JA/8x).

    Each group's p is the ACAT of the omniB of its C(n,2) constituent subgenome pairs, so every copy
    pair is tested with the encoding-invariant primary and the group omnibus inherits invariance.
    Same kinship-preserving bootstrap as :func:`run_pair_scan_omnib`: whiten once, rescan every group
    for each null phenotype, and calibrate the experiment-wide FWER and tail-excess on group p."""
    subs = list(subdata.keys())
    n_t = sample_idx.size
    pair_defs = list(itertools.combinations(subs, 2))
    family = MasterGroupFamily(
        subgenomes=tuple(subs),
        group_ids=tuple(f"group_{i}" for i in range(len(groups))),
        genes=tuple(tuple(group) for group in groups),
    )
    scores, expanded = _score_omnib_family(
        subdata, family, y_raw, sample_idx, cap=cap, n_pc=n_pc,
        transform=transform, bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed, n_jobs=n_jobs,
        grm_method=grm_method, maf_min=maf_min, burden_maf=burden_maf,
        min_snp=min_snp, covariates=covariates)
    selected = np.flatnonzero(scores.group_estimable & ~scores.group_partial)
    kept = [family.genes[i] for i in selected]
    G = len(kept)
    if G < 1:
        raise ValueError("no homoeolog groups retained (need all copies present with >= min_snp SNPs)")
    P = scores.group_p[selected]
    pair_obs = np.empty((G, len(pair_defs)))
    component_obs = np.empty((G, len(pair_defs), len(OMNIB_COMPONENT_NAMES)))
    for local, family_index in enumerate(selected):
        edge_idx = np.asarray(expanded.group_edge_indices[family_index], int)
        pair_obs[local] = scores.edge_p[edge_idx, 0]
        component_obs[local] = scores.edge_components_obs[edge_idx]
    cv = scores.covariance_components
    cov_meta = scores.covariate_metadata
    gated_by_gene = scores.gated_snp

    p_obs = P[:, 0]
    finite = np.isfinite(p_obs)
    if not finite.any():
        raise ValueError("no estimable pairs (every omniB component was non-estimable)")
    minp_obs = float(np.nanmin(p_obs))
    bonf = 0.05 / G
    order = [int(i) for i in np.argsort(np.where(finite, p_obs, np.inf)) if finite[i]]
    pair_labels = [f"{sa}{sb}" for sa, sb in pair_defs]

    def _hit(i):
        pairwise = {}
        for pi, label in enumerate(pair_labels):
            pairwise[label] = (
                {"p_omnib": float(pair_obs[i, pi])}
                | _omnib_component_record(component_obs[i, pi]))
        flat = component_obs[i].reshape(-1)
        finite_flat = np.isfinite(flat)
        smallest_pair = smallest_component = None
        smallest_p = None
        if finite_flat.any():
            idx = int(np.nanargmin(np.where(finite_flat, flat, np.nan)))
            pi, ci = np.unravel_index(idx, component_obs[i].shape)
            smallest_pair = pair_labels[pi]
            smallest_component = OMNIB_COMPONENT_NAMES[ci]
            smallest_p = float(flat[idx])
        return {
            "pair": kept[i],
            "p": float(p_obs[i]),
            "p_adjusted_bonferroni": float(min(p_obs[i] * G, 1.0)),
            "pairwise": pairwise,
            "smallest_pair": smallest_pair,
            "smallest_component": smallest_component,
            "smallest_component_p": smallest_p,
        }

    sig = ([_hit(i) for i in order if p_obs[i] < bonf]
           if inferential else None)
    top = [_hit(i) for i in order[:5]]

    if full_dump_path:
        nsnp = {
            s: np.asarray([
                gated_by_gene[(s, dict(zip(subs, group, strict=True))[s])].size
                for group in kept
            ], int)
            for s in subs
        }
        coords = {
            s: [
                _gene_coord(
                    subdata[s],
                    gated_by_gene[(s, dict(zip(subs, group, strict=True))[s])])
                for group in kept
            ]
            for s in subs
        }
        sort_key = np.where(finite, p_obs, np.inf)
        _, rank_of, tie_of = _rank_with_ties(sort_key, kept)
        rows = []
        for i in np.argsort(sort_key, kind="stable"):
            h = _hit(i)
            row = [
                int(rank_of[i]), *kept[i], _tsv_p(p_obs[i]),
                _tsv_p(_neglog10(np.array([p_obs[i]]))[0]),
                h["smallest_pair"] or "NA",
                h["smallest_component"] or "NA",
                _tsv_p(h["smallest_component_p"]),
                *[int(nsnp[s][i]) for s in subs],
                int(sum(nsnp[s][i] for s in subs)),
                int(tie_of[i]), int(not finite[i]),
                int(finite[i] and inferential and p_obs[i] < bonf)
                if inferential else "NA",
                *[value for s in subs for value in coords[s][i]],
            ]
            for label in pair_labels:
                ph = h["pairwise"][label]
                row.extend([
                    _tsv_p(ph["p_omnib"]),
                    _tsv_p(ph["component_p"]["minor_burden"]),
                    _tsv_p(ph["component_p"]["pc1"]),
                    _tsv_p(ph["component_p"]["kernel_hadamard"]),
                ])
            rows.append(row)
        header = (
            ["rank", *[f"gene_{s}" for s in subs], "p_interaction", "neglog10p",
             "smallest_pair", "smallest_component", "smallest_component_p",
             *[f"n_snp_{s}" for s in subs], "n_snp_group",
             "tie_group", "p_unestimable", "primary_sig"]
            + [value for s in subs for value in (f"chrom_{s}", f"pos_{s}")]
            + [value
               for label in pair_labels
               for value in (f"p_omnib_{label}", f"p_minor_burden_{label}",
                             f"p_pc1_{label}", f"p_kernel_hadamard_{label}")]
        )
        _write_ranking_tsv(full_dump_path, header, rows)

    minp_boot_emp = minp_boot_threshold = None
    tail_excess = None
    if bootstrap_B and bootstrap_B > 0:
        null_min = np.nanmin(P[:, 1:], axis=0)
        minp_boot_emp = float((1 + int((null_min <= minp_obs).sum())) / (null_min.size + 1))
        minp_boot_threshold = float(np.quantile(null_min, 0.05))
        tail_excess = {}
        for t in tail_thresholds:
            obs_ct = int((p_obs[finite] < t).sum())
            null_ct = (P[:, 1:] < t).sum(0).astype(float)
            tail_excess[f"n_below_{t:g}"] = dict(
                observed=obs_ct, null_mean=float(null_ct.mean()),
                null_q95=float(np.quantile(null_ct, 0.95)),
                empirical_p=float((1 + int((null_ct >= obs_ct).sum())) / (null_ct.size + 1)))

    return InteractResult(
        trait="", transform=transform, n=int(n_t), G=int(G),
        pair_acat=float(acat(p_obs)), pair_acat_emp=float("nan"), min_p=minp_obs,
        lambda_gc_obs=float(lambda_gc(p_obs[finite])), lambda_gc_perm_median=float("nan"),
        bonferroni_alpha=float(bonf), n_sig=(len(sig) if sig is not None else None), sig=sig,
        top=top,
        sigma_hat={s: float(cv.get(s, 0.0)) for s in subs} | {"e": float(cv.get("e", 0.0))},
        weighted=None, covariates=cov_meta,
        n_planned=int(G), n_valid=int(finite.sum()), n_unestimable=int((~finite).sum()),
        statistic="omniB", calibration_method=("bootstrap" if bootstrap_B else "none"),
        bootstrap_B=int(bootstrap_B), bootstrap_seed=int(bootstrap_seed),
        minp_boot_emp=minp_boot_emp, minp_boot_threshold=minp_boot_threshold,
        tail_excess=tail_excess,
        component_diagnostics={
            "pair_labels": pair_labels,
            "per_pair": {
                label: _omnib_component_summary(component_obs[:, pi, :])
                for pi, label in enumerate(pair_labels)
            },
            "interpretation": (
                "The group p-value is ACAT across pairwise omniB tests. Pair and component "
                "decomposition is descriptive localization, not an additional rejection family."
            ),
        })


def run_clique_scan(
    subdata: dict[str, SubgenomeData],
    triads: list[tuple],                # list of homoeolog groups, each aligned to subs order
    y_raw: np.ndarray,
    sample_idx: np.ndarray,
    *,
    cap: int = 150,
    transform: str = "INT",
    perm_B: int = 2000,
    n_jobs: int = 8,
    seed: int = 7,
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
    min_snp: int = 1,
    triad_weights: dict = None,         # {group_tuple: w}  y-INDEPENDENT prior (HEB/DL), frozen
    covariates: dict = None,            # {n_pcs:int, extra:(n_t,q)} fixed effects; None=legacy
    dominance_adjust: bool = False,     # add per-gene b^2 covariates to every pairwise interaction
    primary_weighting: str = "unweighted",   # which weighting spends alpha
    primary_multiplicity: str = "bonferroni",  # only bonferroni is defined for the group omnibus
    inferential: bool = True,           # False => sensitivity run, emits no rejection set
    full_dump_path: str = None,         # if set, write FULL per-group ranking TSV (else top-N only)
) -> dict:
    """Generic n-subgenome homoeolog-clique burden-product scan: full {K_s} whitening, all
    C(n,2) within-group pairwise interactions, and a group-level ACAT omnibus. Handles any
    ``len(subs) >= 2`` (n=2 dyad, 3 triad, 4 quartet, …); triad is the n=3 special case. Requires
    every homoeolog of a group retained (dense panels). ``covariates`` enter the whitener mean
    model and every pairwise GLS; ``None`` = intercept-only. Permutation is Freedman-Lane. Result
    keys keep the ``triad_*`` names for backward compatibility (they are the group omnibus)."""
    from joblib import Parallel, delayed

    if primary_weighting not in ("unweighted", "weighted"):
        raise ValueError("primary_weighting must be 'unweighted' or 'weighted'")
    if primary_multiplicity != "bonferroni":
        raise ValueError("run_clique_scan supports primary_multiplicity='bonferroni' only")
    bonf_is_primary = bool(inferential)
    rng = np.random.default_rng(seed)
    subs = list(subdata.keys())            # n subgenomes in config order
    n_t = sample_idx.size
    kernels = {s: _build_grm(subdata[s], sample_idx, grm_method, maf_min) for s in subs}
    cov_meta = dict(policy="none")
    C = None
    if covariates:
        C, cov_meta = build_covariate_block(kernels, n_t, n_pcs=int(covariates.get("n_pcs", 0)),
                                            extra=covariates.get("extra"))

    # keep triads with all three homoeologs retained; build aligned burden matrices
    cols = {s: [] for s in subs}
    nsnp = {s: [] for s in subs}                     # callable SNP count per gene, per subgenome
    kept = []
    for group in triads:
        gmap = dict(zip(subs, group, strict=True))   # subgenome -> gene id for this group
        if all(gmap[s] in subdata[s].gene_snp
               and np.asarray(subdata[s].gene_snp[gmap[s]]).size >= min_snp
               for s in subs):
            for s in subs:
                cols[s].append(block_burden_capped(subdata[s].X, subdata[s].gene_snp[gmap[s]],
                                                    cap, rng)[sample_idx])
                nsnp[s].append(int(np.asarray(subdata[s].gene_snp[gmap[s]]).size))
            kept.append(tuple(group))
    G = len(kept)
    if G < 1:
        raise ValueError(
            "no homoeolog group had every subgenome copy retained with "
            f">= {min_snp} SNPs; there is no hypothesis to test")
    Bd = {s: scols_safe(np.column_stack(cols[s])) for s in subs}

    # y-independent per-triad prior weights (HEB/DL), normalized to sum G (missing -> 1)
    w = None
    if triad_weights:
        w = _normalize_weights([triad_weights.get(t, 1.0) for t in kept], G, "triad weights")
    if primary_weighting == "weighted" and w is None:
        raise ValueError("primary_weighting='weighted' requires triad_weights; otherwise the "
                         "unweighted rejections are suppressed and no primary procedure runs")
    weights_checked = False

    y = rank_int(y_raw) if transform == "INT" else y_raw.astype(float)
    Wh, cv = whiten_multi(kernels, y, X=C, seed=42)

    pairwise_defs = list(itertools.combinations(subs, 2))   # all C(n,2) within-group pairs
    tags = [f"{sx}{sy}" for sx, sy in pairwise_defs]
    pw_p, pw_est, pw_excl = {}, {}, {}
    n_late_total = 0
    for sx, sy in pairwise_defs:
        tag = f"{sx}{sy}"
        # RAW-design estimability: the whitener is fitted to y, so deciding membership after
        # whitening would let the tested family move with the phenotype
        pw_est[tag], pw_excl[tag] = pairwise_design_mask(Bd[sx], Bd[sy], C=C,
                                                         dominance_adjust=dominance_adjust)
        pw_p[tag] = pairwise_pvals(Wh, y, Bd[sx], Bd[sy], C=C,
                                   dominance_adjust=dominance_adjust, fixed_mask=pw_est[tag])
        late = int((pw_est[tag] & ~np.isfinite(pw_p[tag])).sum())
        n_late_total += late
        if late:
            raise ValueError(f"contrast {tag}: {late} groups estimable on the raw design lost "
                             "their statistic after whitening")
    if not any(pw_est[t].any() for t in tags):
        raise ValueError("no estimable group in any contrast")
    if inferential and primary_weighting == "weighted" and not weights_checked:
        any_est = np.logical_or.reduce([pw_est[t] for t in tags])
        if not (any_est & (w > 0)).any():
            raise ValueError("every estimable group has weight zero: no hypothesis receives any "
                             "alpha under weighted primary")
        weights_checked = True
    triad_acat = np.array([acat([pw_p[t][i] for t in tags]) for i in range(G)])
    triad_omnibus = acat(triad_acat)
    group_est = np.isfinite(triad_acat)                     # a group is testable if any contrast is

    if full_dump_path:
        # Full genome-wide per-triad ranking by ascending triad-ACAT (descriptive). gene_len is
        # emitted as NA (engine has no coordinates) and joined downstream from the GFF.
        pmat = np.column_stack([pw_p[t] for t in tags])      # (G, C(n,2)) per-pairwise p
        pmat_key = np.where(np.isfinite(pmat), pmat, np.inf)  # private sort key; pmat itself is kept
        # unestimable rows stay in the dump but sort last and print NA rather than a fabricated 1.0
        sort_key = np.where(group_est, triad_acat, np.inf)
        min_pair = np.where(np.isfinite(pmat_key).any(1), pmat_key.min(1), np.nan)
        min_tag = np.where(np.isfinite(min_pair), np.array(tags)[pmat_key.argmin(1)], "NA")
        bonf_ac = 0.05 / G
        _, rank_of, tie_of = _rank_with_ties(sort_key, kept)
        nl = _neglog10(triad_acat)
        ns = {s: np.asarray(nsnp[s]) for s in subs}
        ns_sum = sum(ns[s] for s in subs)
        cbin = _decile_bin(ns_sum)
        na_len = ["NA"] * len(subs)
        rows = []
        for i in np.argsort(sort_key, kind="stable"):
            row = [int(rank_of[i])]
            row += [kept[i][j] for j in range(len(subs))]                 # gene_<s>
            row += [_tsv_p(pmat[i, j]) for j in range(len(tags))]         # p_<tag>
            row += [_tsv_p(triad_acat[i]), _tsv_p(nl[i]),
                    _tsv_p(min_pair[i]), str(min_tag[i])]
            row += [int(ns[s][i]) for s in subs]                          # n_snp_<s>
            g_alpha = (bonf_ac * w[i] if (w is not None and primary_weighting == "weighted")
                       else bonf_ac)
            row += [int(ns_sum[i]), int(cbin[i]), int(tie_of[i]),
                    (int(group_est[i] and triad_acat[i] < g_alpha)
                     if (inferential and bonf_is_primary) else "NA")]
            row += na_len                                                 # gene_len_<s>
            rows.append(row)
        # n=3 keeps the legacy column name `n_snp_triad` for byte-compat with existing
        # triad rankings; n!=3 uses the generic `n_snp_group`.
        n_snp_total_col = "n_snp_triad" if len(subs) == 3 else "n_snp_group"
        header = (["rank"] + [f"gene_{s}" for s in subs]
                  + [f"p_{t}" for t in tags]
                  + ["p_acat", "neglog10p_acat", "min_pair_p", "min_pair_tag"]
                  + [f"n_snp_{s}" for s in subs]
                  + [n_snp_total_col, "callable_snp_decile", "tie_group", "primary_sig_acat"]
                  + [f"gene_len_{s}" for s in subs])
        _write_ranking_tsv(full_dump_path, header, rows)

    # Multiplicity families. A group contributes K = C(s,2) pairwise tests, so a pairwise p that was
    # selected across contrasts is a hypothesis of the whole G*K family; 0.05/G only controls the G
    # tests inside ONE contrast, and the authoritative fields below therefore carry the G*K family.
    # The per-group ACAT omnibus is one test per group (0.05/G) and each contrast also emits one
    # omnibus ACAT over its G groups, which is a third family of size K.
    n_contrasts = len(pairwise_defs)
    bonf_pairwise_family = 0.05 / (G * n_contrasts)
    bonf_group = 0.05 / G
    bonf_contrast = 0.05 / n_contrasts
    n_valid_pairwise = int(sum(int(pw_est[t].sum()) for t in tags))
    families = dict(
        pairwise_all=dict(hypothesis_unit="group_x_subgenome_contrast", alpha=0.05,
                          method="bonferroni", contrasts=list(tags),
                          n_contrasts=int(n_contrasts), n_groups=int(G),
                          n_tests=int(G * n_contrasts), n_planned=int(G * n_contrasts),
                          n_valid=n_valid_pairwise,
                          n_unestimable=int(G * n_contrasts - n_valid_pairwise),
                          per_test_alpha=(float(bonf_pairwise_family)
                                          if primary_weighting == "unweighted" else None),
                          per_test_alpha_rule=("0.05/(G*n_contrasts)"
                                               if primary_weighting == "unweighted"
                                               else "0.05*w_i/(G*n_contrasts)"),
                          method_detail=("bonferroni" if primary_weighting == "unweighted"
                                         else "weighted_bonferroni")),
        group_omnibus=dict(hypothesis_unit="group", alpha=0.05, method="bonferroni",
                           n_tests=int(G), n_planned=int(G), n_valid=int(group_est.sum()),
                           n_unestimable=int((~group_est).sum()),
                           per_test_alpha=(float(bonf_group) if primary_weighting == "unweighted"
                                           else None),
                           per_test_alpha_rule=("0.05/G" if primary_weighting == "unweighted"
                                                else "0.05*w_i/G")),
        contrast_omnibus=dict(hypothesis_unit="subgenome_contrast_omnibus", alpha=0.05,
                              method="bonferroni", n_tests=int(n_contrasts),
                              n_planned=int(n_contrasts), per_test_alpha=float(bonf_contrast)))
    inference_plan = dict(
        primary_family=("group_omnibus" if inferential else None),
        inferential=bool(inferential), pairwise_role="gated_follow_up_localization",
        gatekeeping=("a pairwise or contrast rejection is emitted ONLY for a group that already "
                     "rejected in the primary group_omnibus family, so the two cannot be unioned "
                     "into an uncontrolled claim; contrast-level ACAT emits no rejection decision"),
        primary_weighting=primary_weighting,
        weighted_role=("primary" if primary_weighting == "weighted" else
                       "exploratory_no_rejections_emitted"),
        note="a pairwise p selected across the contrasts of its group is a pairwise_all hypothesis; "
             "judging it at the group_omnibus threshold does not control the family. The weighted "
             "and unweighted procedures each spend the FULL alpha over the SAME hypotheses, so "
             "taking discoveries from whichever one rejects is a union of two alpha-level "
             "procedures and is NOT controlled at alpha; exactly one must be predeclared")

    # group omnibus (the declared primary family) reported as an inference, not just as metadata
    # under weighted primary the ranking score is the WEIGHTED adjusted p, whose minimum can sit at
    # a different group than the smallest raw p
    if primary_weighting == "weighted":
        g_score = np.where(group_est & (w > 0), triad_acat * G / np.where(w > 0, w, 1.0), np.inf)
    else:
        g_score = np.where(group_est, triad_acat * G, np.inf)
    g_order = [int(i) for i in np.argsort(g_score) if np.isfinite(g_score[i])]
    # the primary family must actually be decided under whichever weighting was predeclared
    if primary_weighting == "unweighted":
        gated = {i for i in g_order if bonf_is_primary and triad_acat[i] < bonf_group}
        g_rej = [dict(triad=kept[i], p=float(triad_acat[i]),
                      p_adjusted_bonferroni=float(min(g_score[i], 1.0)))
                 for i in g_order if i in gated]
    else:
        gated = {i for i in g_order
                 if bonf_is_primary and w[i] > 0 and triad_acat[i] < bonf_group * w[i]}
        g_rej = [dict(triad=kept[i], p=float(triad_acat[i]), weight=float(w[i]),
                      p_adjusted_bonferroni=float(min(g_score[i], 1.0)))
                 for i in g_order if i in gated]
    # a group whose ACAT combined fewer than K contrasts is still a valid test, but the reader must
    # be able to see that it is a partial omnibus
    all_est = np.logical_and.reduce([pw_est[t] for t in tags])
    group_omnibus = dict(
        bonferroni_family_id="group_omnibus", weighting=primary_weighting,
        bonferroni_alpha=(float(bonf_group) if primary_weighting == "unweighted" else None),
        per_test_alpha_rule=("0.05/G" if primary_weighting == "unweighted"
                             else "0.05*w_i/G (weights normalised to sum G)"),
        n_planned=int(G), n_valid=int(group_est.sum()), n_unestimable=int((~group_est).sum()),
        n_partial=int((group_est & ~all_est).sum()), k_expected=int(n_contrasts),
        min_p=float(triad_acat[g_order[0]]) if g_order else float("nan"),
        min_p_adjusted_bonferroni=(float(min(g_score[g_order[0]], 1.0)) if g_order
                                   else float("nan")),
        n_sig=(len(g_rej) if bonf_is_primary else None),
        sig=(g_rej if bonf_is_primary else None),
        top=[dict(triad=kept[i], p=float(triad_acat[i]),
                  p_adjusted_bonferroni=float(min(g_score[i], 1.0))) for i in g_order[:5]])

    # `gated` holds the groups that survived the primary family; a pairwise or contrast rejection
    # is a FOLLOW-UP inside them, not an independent full-alpha family a reader could union with it
    pw_res = {}
    for tag in tags:
        pv = pw_p[tag]
        est = pw_est[tag]
        nv = int(est.sum())
        order = [int(i) for i in np.argsort(np.where(est, pv, np.inf)) if est[i]]
        p_adj = np.minimum(pv * G * n_contrasts, 1.0)
        rej = ([dict(triad=kept[i], p=float(pv[i]), p_adjusted_bonferroni=float(p_adj[i]))
                for i in order if i in gated and pv[i] < bonf_pairwise_family]
               if (bonf_is_primary and primary_weighting == "unweighted") else None)
        loc = [dict(triad=kept[i], p=float(pv[i])) for i in order if pv[i] < bonf_group]
        ac = acat(pv)
        pw_res[tag] = dict(
            G=int(G), acat=float(ac),
            acat_adjusted_bonferroni=(float(min(ac * n_contrasts, 1.0)) if np.isfinite(ac)
                                      else float("nan")),
            acat_family_id="contrast_omnibus",
            acat_rejected_familywise=None,      # gated: the primary family is group_omnibus
            min_p=float(pv[order[0]]) if order else float("nan"),
            lambda_gc_obs=float(lambda_gc(pv)),
            n_planned=int(G), n_valid=nv, n_unestimable=int(G - nv),
            n_nonestimable=int(G - nv),                      # deprecated alias of n_unestimable
            unestimable=[dict(triad=kept[i], reason=r) for i, r in sorted(pw_excl[tag].items())],
            bonferroni_family_id="pairwise_all",
            min_p_adjusted_bonferroni=(float(p_adj[order[0]]) if order else float("nan")),
            bonferroni_alpha=float(bonf_pairwise_family),
            bonferroni_alpha_basis="0.05/(G*n_contrasts)",
            n_sig=(len(rej) if rej is not None else None), sig=rej,
            n_rejected_familywise=(len(rej) if rej is not None else None),
            rejected_familywise=rej,
            exploratory_within_contrast=dict(
                role="exploratory_only", alpha=0.05, method="bonferroni", n_tests=int(G),
                per_test_alpha=float(bonf_group), n_below_per_contrast_alpha=len(loc),
                valid_for_cross_contrast_selection=False,
                note=f"descriptive count of contrast-{tag} p-values below 0.05/G, retained so that "
                     "results from engines through 1.0.2 remain traceable. NO identifier list is "
                     "emitted: a second selectable rejection set beside the primary one would make "
                     "the pair of procedures a union that is not controlled at alpha"),
            top=[dict(triad=kept[i], p=float(pv[i])) for i in order[:5]])
        if w is not None:
            # weighted Bonferroni over the whole pairwise family: sum_i alpha_i <= alpha requires the
            # weights to be normalised across all G*K tests, not within one contrast
            wsig = ([dict(triad=kept[i], p=float(pv[i]), weight=float(w[i]))
                     for i in order
                     if i in gated and w[i] > 0 and pv[i] < bonf_pairwise_family * w[i]]
                    if (bonf_is_primary and primary_weighting == "weighted") else None)
            wac = acat_weighted(pv, w)
            pw_res[tag]["weighted"] = dict(
                role=("primary" if primary_weighting == "weighted" else
                      "exploratory_no_rejections_emitted"),
                acat_weighted=float(wac),
                acat_weighted_adjusted_bonferroni=(float(min(wac * n_contrasts, 1.0))
                                                   if np.isfinite(wac) else float("nan")),
                acat_rejected_familywise=None,  # gated: the primary family is group_omnibus
                acat_family_id="contrast_omnibus", bonferroni_family_id="pairwise_all",
                n_rejected_familywise=(len(wsig) if wsig is not None else None),
                bonferroni_n_sig=(len(wsig) if wsig is not None else None), sig=wsig,
                per_test_alpha_basis="0.05/(G*n_contrasts) * w_i",
                note="controls FWER only if this weighted procedure was predeclared as THE primary "
                     "one; reporting it alongside the unweighted rejections and taking either is a "
                     "union of two alpha-level procedures")

    n_ac_valid = int(sum(1 for t in tags if np.isfinite(pw_res[t]["acat"])))
    families["contrast_omnibus"]["n_valid"] = n_ac_valid
    families["contrast_omnibus"]["n_unestimable"] = int(n_contrasts - n_ac_valid)

    # Freedman-Lane permutation (reduces to y-shuffle when C is intercept-only)
    if C is None:
        fl_fit, fl_resid = None, None
    else:
        b_fl, *_ = np.linalg.lstsq(C, y, rcond=None)
        fl_fit = C @ b_fl
        fl_resid = y - fl_fit

    def _perm(seed_i):
        r = np.random.default_rng(seed_i)
        perm = r.permutation(n_t)
        ys = y[perm] if C is None else (fl_fit + fl_resid[perm])
        try:
            Whp, _ = whiten_multi(kernels, ys, X=C, seed=seed_i % 100000)
            pp = {}
            for sx, sy in pairwise_defs:
                t = f"{sx}{sy}"
                pp[t] = pairwise_pvals(Whp, ys, Bd[sx], Bd[sy], C=C,
                                       dominance_adjust=dominance_adjust, fixed_mask=pw_est[t])
                # a replicate whose mask-true test yields no statistic is counted as MAXIMALLY
                # extreme, not dropped: those failures are degenerate fits (t -> inf), i.e. the very
                # tail of the null, so deleting them would bias the calibration anticonservatively
                if not np.isfinite(pp[t][pw_est[t]]).all():
                    return 0.0, {t2: float("nan") for t2 in tags}, 1
        except np.linalg.LinAlgError:
            # the null model degenerated for this replicate; any OTHER exception is a bug or a data
            # error and must not be laundered into null-tail evidence
            return 0.0, {t2: float("nan") for t2 in tags}, 1
        tacat = np.array([acat([pp[t][i] for t in tags]) for i in range(G)])
        return acat(tacat), {t: lambda_gc(pp[t]) for t in tags}, 0

    triad_emp = float("nan")
    lam_perm = {}
    perm_status = dict(status="not_run", B_requested=int(perm_B or 0), n_used=0, n_degenerate=0,
                       note="no resampling was run; empirical p fields are null, not 1.0")
    if perm_B and perm_B > 0:
        res = Parallel(n_jobs=n_jobs)(delayed(_perm)(900000 + i) for i in range(perm_B))
        n_deg = int(sum(r[2] for r in res))
        perm_status = dict(
            status="completed" if n_deg == 0 else "completed_with_degenerate_replicates",
            B_requested=int(perm_B), n_used=len(res), n_degenerate=n_deg,
            note=None if n_deg == 0 else
            f"{n_deg} replicate(s) had a design-valid test with no statistic and were counted as "
            "maximally extreme (conservative); a large count means the null model is degenerate")
        if res:
            om = np.array([r[0] for r in res])
            if np.isfinite(triad_omnibus):
                triad_emp = float((1 + int((om <= triad_omnibus).sum())) / (len(om) + 1))
            lam_perm = {}
            for t in tags:
                lv = np.array([r[1][t] for r in res])
                lam_perm[t] = float(np.nanmedian(lv)) if np.isfinite(lv).any() else float("nan")

    estimability = dict(ESTIMABILITY_POLICY, decided_on="raw_design",
                        n_planned=int(G * n_contrasts), n_valid=n_valid_pairwise,
                        n_unestimable=int(G * n_contrasts - n_valid_pairwise),
                        n_late_fail=n_late_total,
                        by_contrast={t: int((~pw_est[t]).sum()) for t in tags})
    out = dict(result_schema_version=2, transform=transform, n=int(n_t), G=int(G),
               families=families, inference_plan=inference_plan, permutation=perm_status,
               estimability=estimability, group_omnibus=group_omnibus,
               triad_acat_omnibus=float(triad_omnibus), triad_acat_omnibus_emp=triad_emp,
               pairwise=pw_res, lambda_gc_perm_median=lam_perm, covariates=cov_meta,
               sigma_hat={s: float(cv.get(s, 0.0)) for s in subs} | {"e": float(cv.get("e", 0.0))})
    if w is not None:
        n_cov = int(sum(t in triad_weights for t in kept))
        out["weighted"] = dict(
            triad_acat_omnibus_weighted=float(acat_weighted(triad_acat, w)),
            audit=dict(n_triads=int(G), n_covered=n_cov, n_missing_default1=int(G - n_cov),
                       weight_min=float(w.min()), weight_mean=float(w.mean()), weight_max=float(w.max())),
            role=("primary" if primary_weighting == "weighted" else
                  "exploratory_no_rejections_emitted"),
            note=("per-triad y-independent prior (HEB/DL) frozen pre-association; weighted "
                  "Bonferroni controls per-pairwise FWER only when predeclared as the single "
                  "primary procedure, weighted ACAT is the prior-weighted omnibus."))
    return out


# Backward-compat alias: the hexaploid triad scan is the n=3 special case of the generic clique scan.
run_triad_scan = run_clique_scan


def _ols_rss(Xd: np.ndarray, yv: np.ndarray):
    """OLS fit; return (beta, residual-sum-of-squares)."""
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    resid = yv - Xd @ beta
    return beta, float(resid @ resid)


def pair_conditional_diagnostics(Wh: np.ndarray, y: np.ndarray, bx: np.ndarray, by: np.ndarray,
                                 C: np.ndarray = None) -> dict:
    """Conditional-on-marginals sanity panel for one homoeolog pair in the whitened GLS
    ``y_w ~ C + bX + bY + bX*bY`` (the exact design the scan tests). ``bx``/``by`` are the
    column-standardized gene burdens; ``C`` is the fixed-effect covariate block including the
    intercept (default ``None`` => intercept-only). Returns:
      - the interaction Wald p/t/beta;
      - projection-based collinearity in whitened space: R2_int|marg = 1 - SSE(cI~C0+cX+cY)/
        SSE(cI~C0) and VIF_int = 1/(1-R2) (C0 = Wh@C; partial/projection R2, not TSS-around-mean);
      - a nested-model F panel: interaction | marginals (with the F==t^2 check), joint main effects,
        and the total pair model;
      - single-gene burden-GLS marginal p (each burden alone under the same covariance + C).
    All quantities use the same whitener/burdens/covariates as the production scan."""
    n = int(y.shape[0])
    C0 = (Wh @ np.ones(n)).reshape(-1, 1) if C is None else Wh @ np.asarray(C, float).reshape(n, -1)
    p_c = C0.shape[1]                                   # covariate-block width (1 = intercept only)
    cX = Wh @ bx
    cY = Wh @ by
    cI = Wh @ (bx * by)
    yw = Wh @ y

    Xfull = np.column_stack([C0, cX, cY, cI])           # interaction at index p_c+2
    Xmain = np.column_stack([C0, cX, cY])
    Xnull = C0
    j_int = p_c + 2
    bfull, rss_full = _ols_rss(Xfull, yw)
    bmain, rss_main = _ols_rss(Xmain, yw)
    _, rss_null = _ols_rss(Xnull, yw)

    # rank-aware df (== n-(p_c+3) when full rank; guards collinear covariates)
    rank_full = int(np.linalg.matrix_rank(Xfull))
    rank_main = int(np.linalg.matrix_rank(Xmain))
    estimable = bool(rank_full == p_c + 3)
    df = n - rank_full
    s2 = rss_full / df
    se_I = np.sqrt(max(s2 * np.linalg.pinv(Xfull.T @ Xfull)[j_int, j_int], 1e-30))
    t_I = float(bfull[j_int] / se_I)
    p_int = float(2.0 * stats.t.sf(abs(t_I), df))

    # partial R2 / VIF of the interaction column vs the covariate+marginal columns
    _, sse0 = _ols_rss(Xnull, cI)                       # cI ~ C0
    _, sse1 = _ols_rss(Xmain, cI)                       # cI ~ C0 + cX + cY
    r2_im = float(1.0 - sse1 / sse0) if sse0 > 0 else float("nan")
    vif = float(1.0 / (1.0 - r2_im)) if np.isfinite(r2_im) and r2_im < 1 else float("inf")

    def _F(rss_r, rss_f, df_num, df_den):
        F = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
        return float(F), float(stats.f.sf(F, df_num, df_den))

    df_main = n - rank_main
    F_int, pF_int = _F(rss_main, rss_full, 1, df)        # interaction | marginals (== Wald)
    F_main, pF_main = _F(rss_null, rss_main, 2, df_main)  # joint main effects
    F_pair, pF_pair = _F(rss_null, rss_full, 3, df)      # total pair model

    # single-gene burden-GLS marginal p (each burden alone, same whitener + covariates)
    def _marg_p(c):
        Xm = np.column_stack([C0, c])
        bm, rss_m = _ols_rss(Xm, yw)
        dfm = n - (p_c + 1)
        se = np.sqrt(max((rss_m / dfm) * np.linalg.pinv(Xm.T @ Xm)[p_c, p_c], 1e-30))
        return float(2.0 * stats.t.sf(abs(bm[p_c] / se), dfm))

    # joint main-effect coefficient p (each in presence of the other + covariates)
    se_main = np.sqrt(np.maximum((rss_main / df_main) * np.diag(np.linalg.pinv(Xmain.T @ Xmain)), 1e-30))
    p_mainX = float(2.0 * stats.t.sf(abs(bmain[p_c] / se_main[p_c]), df_main))
    p_mainY = float(2.0 * stats.t.sf(abs(bmain[p_c + 1] / se_main[p_c + 1]), df_main))

    # Frisch-Waugh-Lovell check: the interaction t on the part of cI orthogonal to the marginals
    # (+covariates) equals the full-model interaction t.
    b_proj, _ = _ols_rss(Xmain, cI)
    cI_resid = cI - Xmain @ b_proj
    Xres = np.column_stack([C0, cX, cY, cI_resid])
    bres, rss_res = _ols_rss(Xres, yw)
    se_res = np.sqrt(max((rss_res / df) * np.linalg.pinv(Xres.T @ Xres)[j_int, j_int], 1e-30))
    t_resid = float(bres[j_int] / se_res)

    return dict(
        n=n, n_covariates=p_c, design_rank=rank_full, interaction_estimable=estimable,
        interaction_p_wald=p_int, interaction_t=t_I,
        interaction_beta=float(bfull[j_int]), vif_int=vif, r2_int_given_marginals=r2_im,
        residualized_interaction=dict(t=t_resid, absdiff_vs_full_t=float(abs(t_resid - t_I))),
        corr_int_bX=float(np.corrcoef(cI, cX)[0, 1]), corr_int_bY=float(np.corrcoef(cI, cY)[0, 1]),
        nested_interaction_given_marginals=dict(
            F=F_int, p=pF_int, df=[1, df],
            check_F_eq_t2=dict(t2=float(t_I ** 2), absdiff=float(abs(F_int - t_I ** 2)),
                               p_wald=p_int, p_F_minus_p_wald=float(abs(pF_int - p_int)))),
        nested_joint_main_effects=dict(F=F_main, p=pF_main, df=[2, df_main]),
        nested_total_pair_model=dict(F=F_pair, p=pF_pair, df=[3, df]),
        single_gene_marginal=dict(p_X_alone=_marg_p(cX), p_Y_alone=_marg_p(cY),
                                  p_X_joint=p_mainX, p_Y_joint=p_mainY))


def run_multitrait_pair_scan(
    subdata: dict[str, SubgenomeData],
    pairs: list[tuple],
    y_by_trait: dict[str, np.ndarray],  # trait -> y_raw on complete-case samples (order = trait_set)
    sample_idx: np.ndarray,
    *,
    trait_set: FrozenTraitSet,          # frozen, ordered; keys of y_by_trait MUST match
    inferential: bool = True,           # False => sensitivity run, emits no rejection set
    cap: int = 150,
    transform: str = "INT",
    perm_B: int = 2000,
    n_jobs: int = 8,
    seed: int = 7,
    pair_subs: tuple = None,
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
    min_snp: int = 1,
    dominance_adjust: bool = False,     # add per-gene b^2 covariates to every pairwise interaction
) -> dict:
    """Multi-trait (pleiotropy) pairwise scan: ACAT-across-traits.

    For each homoeolog pair, the per-trait whitened interaction p-values over a predeclared, frozen
    trait set are combined with ACAT into a single per-pair pleiotropy p. Multiplicity is over G
    pairs only (each pair yields exactly one combined p), not G x T. ACAT is dependence-robust so
    the combination stays calibrated under cross-trait correlation, but is not the optimal combiner
    for dense weak same-direction effects.

    Calibration uses a shared permutation: one row permutation per replicate applied to every
    trait's phenotype vector, preserving cross-trait correlation while breaking the
    genotype-phenotype link. The whitener depends on y (REML), so each trait is re-whitened in
    each permutation. ``audit_min_trait_*`` fields are reported for transparency only."""
    from joblib import Parallel, delayed

    if pair_subs is None:
        raise ValueError("pair_subs=(sx, sy) is required for the pairwise multi-trait scan")
    rng = np.random.default_rng(seed)
    subs = list(subdata.keys())
    sx, sy = pair_subs
    n_t = sample_idx.size
    traits = list(trait_set.traits)
    if list(y_by_trait.keys()) != traits:
        raise ValueError("y_by_trait key order must match the frozen trait set "
                         f"{traits}; got {list(y_by_trait.keys())}")
    for t in traits:
        yt = np.asarray(y_by_trait[t], float)
        if yt.shape[0] != n_t:
            raise ValueError(f"trait '{t}' has {yt.shape[0]} values; expected {n_t} "
                             "(complete-case sample count)")
        if not np.all(np.isfinite(yt)):
            raise ValueError(f"trait '{t}' contains non-finite values; pass complete-case "
                             "(NA-filtered across the whole trait set) phenotypes")

    # genotype-only objects built once (trait-independent)
    kernels = {s: _build_grm(subdata[s], sample_idx, grm_method, maf_min) for s in subs}
    bx_cols, by_cols, kept_pairs = [], [], []
    for gx, gy in pairs:
        if (gx in subdata[sx].gene_snp and gy in subdata[sy].gene_snp
                and np.asarray(subdata[sx].gene_snp[gx]).size >= min_snp
                and np.asarray(subdata[sy].gene_snp[gy]).size >= min_snp):
            bx_cols.append(block_burden_capped(subdata[sx].X, subdata[sx].gene_snp[gx], cap, rng)[sample_idx])
            by_cols.append(block_burden_capped(subdata[sy].X, subdata[sy].gene_snp[gy], cap, rng)[sample_idx])
            kept_pairs.append((gx, gy))
    G = len(kept_pairs)
    if G < 1:
        raise ValueError("no homoeolog pairs retained (none present in both subgenomes); "
                         "check pairs table vs snp_to_gene gene IDs")
    BX = scols_safe(np.column_stack(bx_cols))
    BY = scols_safe(np.column_stack(by_cols))

    # per-trait observed transform + whitened per-pair interaction p (whitener depends on y).
    # Observed whitening uses a deterministic per-trait seed; REML is 3-start so the optimum is
    # stable regardless of seed.
    y_t = {t: (rank_int(np.asarray(y_by_trait[t], float)) if transform == "INT"
               else np.asarray(y_by_trait[t], float)) for t in traits}
    P = np.empty((G, len(traits)))
    cv_by_trait = {}
    # one RAW-design mask shared by every trait and every permutation: the whitener is refitted per
    # trait, so a mask read off the whitened design would make the family depend on the phenotypes
    design_mask, design_excl = pairwise_design_mask(BX, BY, dominance_adjust=dominance_adjust)
    if not design_mask.any():
        raise ValueError("no estimable pair in the multi-trait scan")
    M = np.repeat(design_mask[:, None], len(traits), axis=1)
    for j, t in enumerate(traits):
        Wh, cv = whiten_multi(kernels, y_t[t], seed=42 + j)
        pv = pairwise_pvals(Wh, y_t[t], BX, BY, dominance_adjust=dominance_adjust,
                            fixed_mask=design_mask)
        late = int((design_mask & ~np.isfinite(pv)).sum())
        if late:
            raise ValueError(f"trait '{t}': {late} pairs estimable on the raw design lost their "
                             "statistic after whitening")
        P[:, j] = pv                                    # non-estimable stays NaN; ACAT drops it
        cv_by_trait[t] = {s: float(cv.get(s, 0.0)) for s in subs} | {"e": float(cv.get("e", 0.0))}

    Pf = np.where(np.isfinite(P), P, np.inf)
    has_any = np.isfinite(P).any(1)
    pleio_p = np.array([acat(P[i]) for i in range(G)])
    audit_min_p = np.where(has_any, Pf.min(1), np.nan)
    audit_min_trait = [traits[int(j)] if has_any[i] else None for i, j in enumerate(Pf.argmin(1))]

    pleio_omnibus = acat(pleio_p)
    finite_pleio = np.isfinite(pleio_p)
    if not finite_pleio.any():
        raise ValueError("no pair has a defined pleiotropy statistic; an empirical p computed "
                         "against an undefined observed value would report the smallest possible p")
    minp_obs = float(pleio_p[finite_pleio].min()) if finite_pleio.any() else float("nan")
    lam_obs = lambda_gc(pleio_p)
    bonf = 0.05 / G                                     # multiplicity over G pairs only
    order = np.argsort(np.where(finite_pleio, pleio_p, np.inf))
    order_valid = [int(i) for i in order if finite_pleio[i]]
    sig = ([dict(pair=kept_pairs[int(i)], pleio_p=float(pleio_p[i]),
                 per_trait_p={t: float(P[i, j]) for j, t in enumerate(traits)})
            for i in order_valid if pleio_p[i] < bonf] if inferential else None)
    top = [dict(pair=kept_pairs[int(i)], pleio_p=float(pleio_p[i]),
                audit_min_trait_p=float(audit_min_p[i]), audit_min_trait_name=audit_min_trait[int(i)],
                per_trait_p={t: float(P[i, j]) for j, t in enumerate(traits)})
           for i in order_valid[:5]]

    # shared-permutation calibration (same row permutation across all traits)
    def _perm(seed_i):
        r = np.random.default_rng(seed_i)
        perm = r.permutation(n_t)
        cols = []
        for j, t in enumerate(traits):
            ys = y_t[t][perm]
            # the trait x pair family is frozen to the observed design mask; a replicate that loses
            # or gains a component would be testing a different family
            try:
                Whp, _ = whiten_multi(kernels, ys, seed=seed_i % 100000)
                pp = pairwise_pvals(Whp, ys, BX, BY, dominance_adjust=dominance_adjust,
                                    fixed_mask=M[:, j])
            except np.linalg.LinAlgError:
                # only a degenerate null model may become a conservative extreme replicate; any
                # other exception is a bug or a data error and must propagate
                return 0.0, 0.0, 1
            if not np.isfinite(pp[M[:, j]]).all():
                return 0.0, 0.0, 1                      # degenerate => maximally extreme, not dropped
            cols.append(pp)
        Pp = np.column_stack(cols)
        pleio_perm = np.array([acat(Pp[i]) for i in range(G)])
        if not np.isfinite(pleio_perm).any():
            return 0.0, 0.0, 1
        return acat(pleio_perm), float(np.nanmin(pleio_perm)), 0

    pleio_emp = float("nan")
    minp_emp = float("nan")
    perm_status = dict(status="not_run", B_requested=int(perm_B or 0), n_used=0, n_degenerate=0)
    if perm_B and perm_B > 0:
        res = Parallel(n_jobs=n_jobs)(delayed(_perm)(900000 + i) for i in range(perm_B))
        n_deg = int(sum(r[2] for r in res))
        perm_status = dict(
            status="completed" if n_deg == 0 else "completed_with_degenerate_replicates",
            B_requested=int(perm_B), n_used=len(res), n_degenerate=n_deg)
        if res:
            om = np.array([r[0] for r in res])
            mp = np.array([r[1] for r in res])
            pleio_emp = float((1 + int((om <= pleio_omnibus).sum())) / (len(om) + 1))
            minp_emp = float((1 + int((mp <= minp_obs).sum())) / (len(mp) + 1))

    return dict(
        transform=transform, n=int(n_t), G=int(G), traits=list(traits),
        trait_set_digest=trait_set.digest, single_trait=bool(len(traits) == 1),
        pleio_acat_omnibus=float(pleio_omnibus), pleio_acat_omnibus_emp=pleio_emp,
        n_planned=int(G), n_valid=int(finite_pleio.sum()),
        n_unestimable=int((~finite_pleio).sum()),
        n_partial=int((finite_pleio & ~M.all(1)).sum()), k_expected=len(traits),
        permutation=perm_status,
        estimability=dict(ESTIMABILITY_POLICY, decided_on="raw_design",
                          n_planned=int(G), n_valid=int(design_mask.sum()),
                          n_unestimable=int((~design_mask).sum()),
                          excluded=[dict(pair=kept_pairs[i], reason=r)
                                    for i, r in sorted(design_excl.items())]),
        min_p=minp_obs, min_p_emp=minp_emp, lambda_gc_obs=float(lam_obs),
        bonferroni_alpha=float(bonf), n_sig=(len(sig) if sig is not None else None), sig=sig,
        top=top, inferential=bool(inferential),
        sigma_hat_by_trait=cv_by_trait,
        note=("DEGENERATE single-trait set: pleiotropy ACAT reduces to the single-trait scan. "
              if len(traits) == 1 else "")
        + ("multiplicity is over G pairs (one ACAT-combined pleiotropy p per pair) for ONE "
              "frozen multi-trait family; audit_min_trait_* are reported for transparency and are "
              "NOT inferential. ACAT is dependence-robust (valid under cross-trait correlation) but "
              "is not the optimal combiner for dense weak same-direction effects, so the "
              "discover-more claim is conditional on signal being shared across the frozen traits, "
              "relative to G x T single-trait Bonferroni."))


_bootstrap_minp_calibration = _family_score.bootstrap_minp_calibration


def run_triad3_scan(*args, **kwargs):
    """Dispatch the opt-in exact-three-copy estimator without normalizing it."""
    from .triad3 import run_triad3_scan as _run

    return _run(*args, **kwargs)


def _load_pairs(path: str, subs: list[str]):
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    cols = [f"gene_{s}" for s in subs]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"pairs table missing columns {missing}; has {list(df.columns)}")
    return [tuple(r) for r in df[cols].itertuples(index=False, name=None)]


def _load_pair_weights(path: str, subs: list[str]) -> dict:
    """Load y-independent pair priors -> {(gene_s0, gene_s1): weight}. Columns: gene_<S> per
    subgenome + 'weight'. Weights must be computed without the phenotype and frozen before the scan."""
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    cols = [f"gene_{s}" for s in subs]
    if "weight" not in df.columns or any(c not in df.columns for c in cols):
        raise ValueError(f"weights table needs columns {cols + ['weight']}; has {list(df.columns)}")
    return {tuple(r[:-1]): float(r[-1])
            for r in df[cols + ["weight"]].itertuples(index=False, name=None)}


def validate_interact_config(cfg: dict) -> None:
    """Validate the interaction schema without opening large genotype matrices."""
    cfg = normalize_interact_config(cfg)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("interact"), dict):
        raise SystemExit("ERR: interaction config needs a top-level `interact` mapping")
    ic = cfg["interact"]
    subs = ic.get("subgenomes")
    if not isinstance(subs, list) or not subs:
        raise SystemExit("ERR: interact.subgenomes must be a non-empty list")
    if len(set(subs)) != len(subs):
        raise SystemExit(f"ERR: interact.subgenomes has duplicates: {subs}")
    mode = str(ic.get("mode", "pairwise")).lower()
    expected_n = {"pairwise": 2, "triad": 3}
    statistic_requested = str(ic.get("statistic", "omniB")).lower()
    if statistic_requested.replace("-", "").replace("_", "") in {
        "fourway", "4way",
    }:
        raise SystemExit(
            "ERR: HomoeoGWAS never fits a direct four-way interaction; "
            "use the pair-edge group omniB (subset_order=2), which combines "
            "the six supported pair interactions without a fourth-order term")
    if mode == "group":
        if statistic_requested != "omnib":
            raise SystemExit(
                "ERR: interact.mode=group is currently defined only for statistic=omniB")
        if len(subs) < 2:
            raise SystemExit("ERR: interact mode=group needs at least two subgenomes")
        hypothesis_unit = str(ic.get("hypothesis_unit", "")).lower()
        if hypothesis_unit not in {"edge", "group"}:
            raise SystemExit("ERR: interact.hypothesis_unit must be edge or group")
        if ic.get("subset_order") != 2:
            raise SystemExit(
                "ERR: interact.subset_order=2 is required for hypothesis_unit=edge/group")
        family_scope = str(ic.get("family_scope", "primary_only")).lower()
        if family_scope not in {"primary_only", "joint"}:
            raise SystemExit(
                "ERR: interact.family_scope must be primary_only or joint")
        grm = ic.get("grm", {})
        if not isinstance(grm, dict):
            raise SystemExit("ERR: interact.grm must be a mapping")
        if grm.get("method") != "grm_from_X":
            raise SystemExit(
                "ERR: canonical group omniB requires "
                "interact.grm.method=grm_from_X; remove the explicit value "
                "to use the canonical default")
        if grm.get("maf_min") != 0.01:
            raise SystemExit(
                "ERR: canonical group omniB requires "
                "interact.grm.maf_min=0.01; remove the explicit value to use "
                "the canonical default")
        if grm.get("scope") != "all_subgenomes":
            raise SystemExit(
                "ERR: canonical group omniB requires "
                "interact.grm.scope=all_subgenomes so every pair edge shares "
                "one null model")
        burden = ic.get("burden", {})
        if not isinstance(burden, dict):
            raise SystemExit("ERR: interact.burden must be a mapping")
        if burden.get("maf_min") != 0.01:
            raise SystemExit(
                "ERR: canonical group omniB requires "
                "interact.burden.maf_min=0.01; remove the explicit value to "
                "use the canonical default")
        feature_seed = burden.get("feature_seed")
        if feature_seed is not None and (
            isinstance(feature_seed, bool)
            or not isinstance(feature_seed, int)
            or feature_seed < 0
        ):
            raise SystemExit(
                "ERR: interact.burden.feature_seed must be an integer >= 0")
    elif mode not in expected_n:
        raise SystemExit(
            "ERR: interact.mode must be group, pairwise (2 subgenomes), or triad "
            "(3 subgenomes); for 4+ use canonical group mode over 2-/3-subgenome subsets")
    if mode in expected_n and len(subs) != expected_n[mode]:
        raise SystemExit(
            f"ERR: interact mode={mode} needs exactly {expected_n[mode]} "
            f"subgenomes; got {subs}")
    for key in ("genotype", "snp_to_gene"):
        mapping = ic.get(key)
        if not isinstance(mapping, dict):
            raise SystemExit(f"ERR: interact.{key} must map subgenome to path")
        missing = [s for s in subs if not mapping.get(s)]
        if missing:
            raise SystemExit(
                f"ERR: interact.{key} missing paths for subgenomes {missing}")
    for key in ("phenotype", "sample_col"):
        if not ic.get(key):
            raise SystemExit(f"ERR: interact.{key} is required")
    if not ic.get("trait") and not ic.get("multi_trait"):
        raise SystemExit("ERR: interact needs `trait` or a non-empty `multi_trait` list")
    if ic.get("trait") and ic.get("multi_trait"):
        raise SystemExit("ERR: set only one of interact.trait and interact.multi_trait")
    if ic.get("multi_trait") is not None:
        traits = ic["multi_trait"]
        if not isinstance(traits, list) or not traits or len(set(traits)) != len(traits):
            raise SystemExit(
                "ERR: interact.multi_trait must be a non-empty list of unique trait names")
        if mode != "pairwise":
            raise SystemExit("ERR: interact.multi_trait is supported only in pairwise mode")
    statistic = statistic_requested
    if statistic not in {"omnib", "burden", "triad3"}:
        raise SystemExit(
            "ERR: interact.statistic must be omniB, burden, or experimental triad3; "
            f"got {ic.get('statistic')!r}")
    if statistic == "triad3" and mode != "triad":
        raise SystemExit(
            "ERR: interact.statistic=triad3 requires mode=triad and exactly "
            "three subgenomes")
    primary_transform = str(ic.get("primary_transform", "INT")).upper()
    if primary_transform not in {"INT", "RAW"}:
        raise SystemExit(
            "ERR: interact.primary_transform must be INT or raw; "
            f"got {ic.get('primary_transform')!r}")
    if primary_transform == "RAW" and (
        (mode == "group" and statistic == "omnib") or statistic == "triad3"
    ):
        raise SystemExit(
            "ERR: interact.primary_transform must be INT for canonical group omniB "
            "and triad3; raw is sensitivity-only")
    primary_weighting = str(ic.get("primary_weighting", "unweighted")).lower()
    if primary_weighting not in {"unweighted", "weighted"}:
        raise SystemExit(
            "ERR: interact.primary_weighting must be unweighted or weighted; "
            f"got {ic.get('primary_weighting')!r}")
    if primary_weighting == "weighted" and not ic.get("weights"):
        raise SystemExit(
            "ERR: interact.primary_weighting=weighted requires interact.weights")
    if primary_weighting == "weighted" and statistic in {"omnib", "triad3"}:
        raise SystemExit(
            f"ERR: weighted primary inference is not implemented for statistic={statistic}")
    if primary_weighting == "weighted" and ic.get("multi_trait"):
        raise SystemExit(
            "ERR: weighted primary inference is not implemented for multi_trait")
    if statistic in {"omnib", "triad3"} and ic.get("weights"):
        raise SystemExit(
            f"ERR: interact.weights is not used by statistic={statistic}; use burden or "
            "remove the weights file")
    default_multiplicity = (
        "bootstrap_minp"
        if statistic == "triad3" or (statistic == "omnib" and mode == "group")
        else "bonferroni")
    primary_multiplicity = str(
        ic.get("primary_multiplicity", default_multiplicity)
    ).lower()
    if primary_multiplicity not in {
        "bonferroni", "permutation_minp", "bootstrap_minp"
    }:
        raise SystemExit(
            "ERR: interact.primary_multiplicity must be bonferroni, "
            "bootstrap_minp, or "
            f"permutation_minp; got {ic.get('primary_multiplicity')!r}")
    if statistic == "triad3" and primary_multiplicity != "bootstrap_minp":
        raise SystemExit(
            "ERR: statistic=triad3 requires "
            "interact.primary_multiplicity=bootstrap_minp; analytic "
            "Bonferroni is descriptive only")
    if (
        mode == "group"
        and statistic == "omnib"
        and primary_multiplicity != "bootstrap_minp"
    ):
        raise SystemExit(
            "ERR: interact.mode=group statistic=omniB requires "
            "primary_multiplicity=bootstrap_minp; the formal group API always "
            "uses one bootstrap min-P family")
    bootstrap_minp_supported = (
        statistic == "triad3"
        or (statistic == "omnib" and mode in {"pairwise", "group"})
    )
    if primary_multiplicity == "bootstrap_minp" and not bootstrap_minp_supported:
        raise SystemExit(
            "ERR: interact.primary_multiplicity=bootstrap_minp is defined for "
            "statistic=triad3 or group/pairwise statistic=omniB")
    if primary_multiplicity == "permutation_minp":
        if statistic != "burden" or mode != "pairwise" or ic.get("multi_trait"):
            raise SystemExit(
                "ERR: permutation_minp is defined only for a single-trait "
                "pairwise burden scan")
        if primary_weighting == "weighted":
            raise SystemExit(
                "ERR: weighted permutation_minp is not implemented")
    calibration = ic.get("calibration", {})
    if not isinstance(calibration, dict):
        raise SystemExit("ERR: interact.calibration must be a mapping")
    calibration_method = str(calibration.get(
        "method", "bootstrap" if statistic in {"omnib", "triad3"} else "permutation"
    )).lower()
    qa_only = calibration.get("qa_only", False)
    if not isinstance(qa_only, bool):
        raise SystemExit(
            "ERR: interact.calibration.qa_only must be true or false")
    if (statistic, calibration_method) not in {
        ("omnib", "bootstrap"),
        ("triad3", "bootstrap"),
        ("burden", "permutation"),
    }:
        raise SystemExit(
            "ERR: supported statistic/calibration pairs are "
            "omniB+bootstrap, triad3+bootstrap, and burden+permutation; got "
            f"{statistic}+{calibration_method}")
    checkpoint = calibration.get("checkpoint")
    if checkpoint is not None:
        if not isinstance(checkpoint, dict):
            raise SystemExit(
                "ERR: interact.calibration.checkpoint must be a mapping")
        checkpoint_enabled = checkpoint.get("enabled", False)
        if not isinstance(checkpoint_enabled, bool):
            raise SystemExit(
                "ERR: interact.calibration.checkpoint.enabled must be true or false")
        block_size = checkpoint.get("block_size", 25)
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size < 1
        ):
            raise SystemExit(
                "ERR: interact.calibration.checkpoint.block_size must be an "
                f"integer >= 1; got {block_size!r}")
        if checkpoint_enabled:
            if mode != "group" or statistic != "omnib":
                raise SystemExit(
                    "ERR: bootstrap checkpointing is supported only by the "
                    "canonical group omniB path")
            checkpoint_root = checkpoint.get("root")
            if not isinstance(checkpoint_root, str) or not checkpoint_root.strip():
                raise SystemExit(
                    "ERR: interact.calibration.checkpoint.root must be a "
                    "non-empty path string")
    replicate_key = "B" if statistic in {"omnib", "triad3"} else "perm_B"
    replicate_value = calibration.get(
        replicate_key,
        calibration.get("perm_B" if replicate_key == "B" else "B", 2000),
    )
    if (
        isinstance(replicate_value, bool)
        or not isinstance(replicate_value, int)
        or replicate_value < 0
    ):
        raise SystemExit(
            f"ERR: interact.calibration.{replicate_key} must be an integer >= 0; "
            f"got {replicate_value!r}")
    if (
        statistic == "triad3"
        or (statistic == "omnib" and mode in {"pairwise", "group"}
            and primary_multiplicity == "bootstrap_minp")
        or (statistic == "burden" and mode == "pairwise"
            and primary_multiplicity == "permutation_minp")
    ):
        required_b = (
            19 if qa_only else (
                TRIAD3_FORMAL_BOOTSTRAP_MIN_B
                if statistic == "triad3"
                else (PAIRWISE_OMNIB_FORMAL_BOOTSTRAP_MIN_B
                      if statistic == "omnib"
                      else PAIRWISE_BURDEN_FORMAL_PERMUTATION_MIN_B)))
        if replicate_value < required_b:
            purpose = "QA-only" if qa_only else "formal inferential"
            analysis = (
                "triad3" if statistic == "triad3" else
                ("pairwise omniB" if statistic == "omnib"
                 else "pairwise burden permutation-minP"))
            raise SystemExit(
                f"ERR: {purpose} {analysis} requires "
                f"interact.calibration.{replicate_key} >= {required_b}")
    table_key = ("groups" if mode == "group"
                 else ("pairs" if mode == "pairwise" else "triads"))
    if not ic.get(table_key):
        raise SystemExit(f"ERR: interact mode={mode} requires interact.{table_key}")
    burden = ic.get("burden", {})
    if not isinstance(burden, dict):
        raise SystemExit("ERR: interact.burden must be a mapping")
    if statistic == "triad3" and bool(burden.get("dominance_adjust", False)):
        raise SystemExit(
            "ERR: interact.burden.dominance_adjust is not implemented for "
            "statistic=triad3; remove it or set it to false")
    min_snp = burden.get("min_snp", 2)
    if not isinstance(min_snp, int) or min_snp < 1:
        raise SystemExit(
            f"ERR: interact.burden.min_snp must be an integer >= 1, got {min_snp!r}")
    cap = burden.get("cap", 150)
    if not isinstance(cap, int) or cap < 1:
        raise SystemExit(
            f"ERR: interact.burden.cap must be an integer >= 1, got {cap!r}")
    burden_maf = burden.get("maf_min", 0.01)
    if not isinstance(burden_maf, (int, float)) or not 0 <= float(burden_maf) <= 0.5:
        raise SystemExit(
            "ERR: interact.burden.maf_min must be a number in [0, 0.5], "
            f"got {burden_maf!r}")


def preflight_interact(cfg: dict, *, master_family=None) -> list[str]:
    """Check interaction paths, mapping provenance, table columns, and samples."""
    from bed_reader import open_bed

    from .io import plink_path, read_delimited

    cfg = normalize_interact_config(cfg)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("interact"), dict):
        return ["interaction config needs a top-level `interact` mapping"]
    ic = cfg["interact"]
    subs = list(ic["subgenomes"])
    mode = str(ic.get("mode", "pairwise")).lower()
    problems: list[str] = []
    sample_orders: dict[str, list[str]] = {}
    for s in subs:
        prefix = ic["genotype"][s]
        missing = [
            str(plink_path(prefix, ext))
            for ext in (".bed", ".bim", ".fam")
            if not plink_path(prefix, ext).exists()
        ]
        if missing:
            problems.append(
                f"genotype {s} is missing PLINK files: {missing}")
            continue
        npz_path = Path(ic["snp_to_gene"][s])
        if not npz_path.exists():
            problems.append(f"snp_to_gene {s} missing: {npz_path}")
            continue
        try:
            _validate_snp_mapping(prefix, str(npz_path), expected_subgenome=s)
            with open_bed(str(plink_path(prefix, ".bed"))) as bed:
                sample_orders[s] = [str(v) for v in np.asarray(bed.iid)]
                n_variants = int(bed.sid_count)
            if len(set(sample_orders[s])) != len(sample_orders[s]):
                problems.append(f"genotype {s} has duplicate IID values")
            z = np.load(npz_path, allow_pickle=True)
            if int(np.asarray(z["n_variants"]).item()) != n_variants:
                problems.append(
                    f"snp_to_gene {s} records {int(np.asarray(z['n_variants']).item())} "
                    f"variants but BED has {n_variants}")
        except Exception as exc:  # noqa: BLE001 - aggregate actionable preflight errors
            problems.append(f"snp_to_gene {s} invalid: {exc}")
    if sample_orders:
        first = next((s for s in subs if s in sample_orders), None)
        if first:
            for s in subs:
                if s in sample_orders and sample_orders[s] != sample_orders[first]:
                    problems.append(
                        f"sample order mismatch between genotype {first} and {s}")

    table_key = ("groups" if mode == "group"
                 else ("pairs" if mode == "pairwise" else "triads"))
    table_value = ic.get(table_key)
    if not table_value:
        problems.append(f"interact.{table_key} is required")
        return problems
    table = Path(table_value)
    if mode == "group" and master_family is not None:
        if tuple(master_family.subgenomes) != tuple(subs):
            problems.append(
                "groups table invalid: preloaded master-family subgenomes "
                f"{list(master_family.subgenomes)} do not match {subs}")
    elif not table.exists():
        problems.append(f"{table_key} table missing: {table}")
    else:
        try:
            if mode == "group":
                load_master_group_family(table, subs)
            else:
                _load_pairs(str(table), subs)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{table_key} table invalid: {exc}")
    weights = ic.get("weights")
    if weights:
        weights_path = Path(weights)
        if not weights_path.exists():
            problems.append(f"weights table missing: {weights_path}")
        else:
            try:
                _load_pair_weights(str(weights_path), subs)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"weights table invalid: {exc}")

    phenotype = Path(ic["phenotype"])
    if not phenotype.exists():
        problems.append(f"phenotype missing: {phenotype}")
    else:
        sample_col = ic["sample_col"]
        traits = ([ic["trait"]] if ic.get("trait") else list(ic["multi_trait"]))
        try:
            ph = read_delimited(
                phenotype, dtype={sample_col: "string"})
            missing_cols = [c for c in [sample_col, *traits] if c not in ph.columns]
            if missing_cols:
                problems.append(
                    f"phenotype missing columns {missing_cols}; has {list(ph.columns)}")
            elif sample_orders:
                ids = set(ph[sample_col].dropna().astype(str))
                first = next((s for s in subs if s in sample_orders), None)
                overlap = len(ids.intersection(sample_orders[first])) if first else 0
                if overlap < 10:
                    problems.append(
                        f"only {overlap} samples overlap genotype and phenotype; need >= 10")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"phenotype invalid: {exc}")
    return problems


def _build_cov_arg(cov_cfg, valid: list):
    """Parse the ``interact.covariates`` config into the ``covariates`` dict {n_pcs, extra} that the
    scan functions accept. Returns ``(None, "none")`` when absent.

    Config::

        covariates:
          n_pcs: 10                 # genotype PCs from the combined subgenome GRM (y-independent)
          file: covars.tsv          # optional y-independent covariate table (sample_col + columns)
          sample_col: sample
          columns: [env, batch]     # optional subset; default = all non-sample columns

    The covariate file must be y-independent and frozen before association. Extra covariates are
    aligned to the complete-case ``valid`` sample order."""
    if not cov_cfg:
        return None, "none"
    n_pcs = int(cov_cfg.get("n_pcs", 0))
    extra = None
    label = f"n_pcs={n_pcs}"
    if cov_cfg.get("file"):
        from .io import read_delimited

        cov_sample_col = cov_cfg.get("sample_col", "sample")
        cf = read_delimited(
            cov_cfg["file"], dtype={cov_sample_col: "string"})
        cf = cf.loc[cf[cov_sample_col].notna()].copy()
        cf[cov_sample_col] = cf[cov_sample_col].astype(str)
        if cf[cov_sample_col].duplicated().any():
            dup = cf.loc[cf[cov_sample_col].duplicated(), cov_sample_col].tolist()[:3]
            raise ValueError(
                f"covariate file has duplicate sample IDs (e.g. {dup})")
        cf = cf.set_index(cov_sample_col)
        cols = cov_cfg.get("columns") or [c for c in cf.columns]
        missing_s = [s for s in valid if s not in cf.index]
        if missing_s:
            raise ValueError(f"covariate file missing {len(missing_s)} samples (e.g. {missing_s[:3]})")
        extra = cf.loc[valid, cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(extra)):
            raise ValueError("covariate file contains non-finite values for the complete-case samples")
        label += f"+file[{','.join(map(str, cols))}]"
    if n_pcs <= 0 and extra is None:
        return None, "none"
    return dict(n_pcs=n_pcs, extra=extra), label


def _run_multitrait(args, ic, subs, out_dir, subdata, samples, ph, t0) -> int:
    """Multi-trait (pleiotropy) CLI path: complete-case across a frozen trait set, ACAT-across-traits.

    The trait set is frozen before any association; missing traits raise rather than silently drop,
    so multiplicity is honestly over G pairs."""
    import hashlib

    import pandas as pd

    trait_set = FrozenTraitSet.from_list(ic["multi_trait"])
    tlist = list(trait_set.traits)
    missing_tr = [t for t in tlist if t not in ph.columns]
    if missing_tr:
        print(f"ERROR: interact multi_trait: traits not in phenotype {missing_tr} "
              f"(available {list(ph.columns)[:20]}...)")
        return 1

    # complete-case: rows non-NA across all frozen traits (one shared sample set)
    valid = [s for s in samples if s in ph.index and bool(ph.loc[s, tlist].notna().all())]
    if len(valid) < 10:
        print(f"ERROR: interact multi_trait: only {len(valid)} complete-case samples across "
              f"{len(tlist)} traits (need >=10)")
        return 1
    sample_idx = np.array([samples.index(s) for s in valid])
    y_by_trait = {t: np.array([float(ph.loc[s, t]) for s in valid]) for t in tlist}
    per_trait_present = {t: int(sum((s in ph.index) and pd.notna(ph.loc[s, t]) for s in samples))
                         for t in tlist}

    pairs = _load_pairs(ic["pairs"], subs)
    burden = ic.get("burden", {})
    cap = int(burden.get("cap", 150))
    min_snp = int(burden.get("min_snp", 2))
    dominance_adjust = bool(burden.get("dominance_adjust", False))
    perm_B = int(ic.get("calibration", {}).get("perm_B", 2000))
    grm_cfg = ic.get("grm", {})
    grm_method = grm_cfg.get("method", "compute_grm_maf")
    maf_min = float(grm_cfg.get("maf_min", 0.01))
    n_jobs = int(args.n_jobs)
    primary_transform = str(ic.get("primary_transform", "INT")).upper()
    if primary_transform not in ("INT", "RAW"):
        print(f"ERROR: interact.primary_transform must be 'INT' or 'raw'; got "
              f"'{primary_transform}'.")
        return 2

    print(f"  multi_trait n_complete_case={len(valid)} traits={len(tlist)} digest={trait_set.digest} "
          f"pairs(raw)={len(pairs)} ({time.time()-t0:.1f}s)", flush=True)
    results = {}
    for transform in ("INT", "raw"):
        r = run_multitrait_pair_scan(subdata, pairs, y_by_trait, sample_idx, trait_set=trait_set,
                                     cap=cap, transform=transform,
                                     inferential=(transform.upper() == primary_transform),
                                     perm_B=(perm_B if transform.upper() == primary_transform
                                             else 0), n_jobs=n_jobs,
                                     pair_subs=(subs[0], subs[1]), grm_method=grm_method,
                                     maf_min=maf_min, min_snp=min_snp,
                                     dominance_adjust=dominance_adjust)
        results[transform] = r
        print(f"  [{transform}] G={r['G']} pleio_ACAT_omnibus={r['pleio_acat_omnibus']:.3g} "
              f"emp={r['pleio_acat_omnibus_emp']} minP={r['min_p']:.3g} λ_obs={r['lambda_gc_obs']:.3f} "
              f"nsig(α={r['bonferroni_alpha']:.1e})={r['n_sig']}", flush=True)
        for h in (r["sig"] or []):
            print(f"      HIT {h['pair']} pleio_p={h['pleio_p']:.3g}")

    from . import __version__
    cc_hash = hashlib.sha256("\x00".join(valid).encode()).hexdigest()[:16]
    pairs_path = ic["pairs"]
    pairs_sha = (hashlib.sha256(Path(pairs_path).read_bytes()).hexdigest()[:16]
                 if Path(pairs_path).exists() else None)
    provenance = dict(
        version=__version__, mode="pairwise", multi_trait=True, transform="INT(primary)+raw(sens)",
        grm_method=grm_method, maf_min=maf_min, burden_cap=cap,
        burden_min_snp=min_snp, dominance_adjust=dominance_adjust,
        perm_B=perm_B,
        covariate_policy="none: subgenome-stratified GRMs only (no PCs/covariates)",
        trait_set=tlist, trait_set_digest=trait_set.digest, n_traits=len(tlist),
        primary_transform=primary_transform,
        transform_firewall=("both transforms are scanned but only the primary one emits "
                            "rejections; the other is a sensitivity analysis"),
        n_complete_case=len(valid), complete_case_sha256=cc_hash, complete_case_sample_order=valid,
        per_trait_present=per_trait_present, n_pairs_raw=len(pairs),
        pairs_source=pairs_path, pairs_sha256=pairs_sha,
        psd_floor=1e-6 if grm_method == "grm_from_X" else None, config_path=str(args.config),
        firewall=("trait set is predeclared and FROZEN before association (digest recorded); it must "
                  "NOT be chosen by per-trait GWAS significance, and covariate/PC count must be fixed "
                  "a priori by variance explained. Multiplicity is over G pairs only because each pair "
                  "yields exactly one ACAT-combined p across the frozen trait family. No trait "
                  "selection occurs inside the scan; missing traits raise rather than silently drop."))
    payload = dict(tool="homoeogwas", command="interact", mode="pairwise", multi_trait=True,
                   subgenomes=subs, traits=tlist, provenance=provenance, results=results)
    fp = out_dir / f"interact_multitrait_{trait_set.digest}.json"
    fp.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False))
    print(f"homoeogwas interact (multi-trait) -> {fp} ({time.time()-t0:.1f}s)")
    return 0


def _validate_canonical_parallel_runtime(
        *, n_jobs: int, libraries: list[dict] | None = None) -> None:
    """Refuse process parallelism after an oversized native pool initialized."""
    if int(n_jobs) <= 1:
        return
    if libraries is None:
        from threadpoolctl import threadpool_info

        libraries = threadpool_info()
    offenders = [
        library for library in libraries
        if library.get("user_api") in {"blas", "openmp"}
        and int(library.get("num_threads") or 0) != 1
    ]
    if not offenders:
        return
    detail = ", ".join(
        f"{library.get('prefix') or library.get('internal_api') or 'numeric-library'}="
        f"{library.get('num_threads')}"
        for library in offenders
    )
    raise RuntimeError(
        "native numeric thread pools are already oversubscribed "
        f"({detail}); launch through the installed `homoeogwas interact` console "
        "command so OPENBLAS/OMP/MKL/NUMEXPR thread limits are applied before "
        "NumPy/SciPy import, or use --n-jobs 1"
    )


def cmd_interact(args) -> int:
    import yaml

    t0 = time.time()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    try:
        verified_launch = verify_formal_launch(args.config, cfg)
    except FormalLaunchError as exc:
        print(f"ERROR: formal launch identity check failed: {exc}")
        return 1
    cfg = normalize_interact_config(cfg)
    validate_interact_config(cfg)
    ic = cfg["interact"]
    subs = list(ic["subgenomes"])
    mode = str(ic.get("mode", "pairwise")).lower()
    if mode == "group" and str(ic.get("statistic", "burden")) == "omniB":
        try:
            _validate_canonical_parallel_runtime(n_jobs=args.n_jobs)
        except RuntimeError as exc:
            print(f"ERROR: interaction parallel runtime invalid: {exc}")
            return 1
    master_family = None
    if mode == "group":
        try:
            master_family = load_master_group_family(ic["groups"], subs)
        except Exception as exc:  # noqa: BLE001 - user-facing input repair
            print(f"ERROR: groups table invalid: {exc}")
            return 1
    problems = preflight_interact(cfg, master_family=master_family)
    if problems:
        print(f"ERROR: interaction input preflight found {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    group_modes = ("group", "triad")
    out_dir = Path(args.out_dir or cfg.get("outputs", {}).get("out_dir", "results/interact"))
    out_dir.mkdir(parents=True, exist_ok=True)

    trait_label = ic.get("trait") if ic.get("multi_trait") is None else f"multi:{ic['multi_trait']}"
    print(f"=== homoeogwas interact (mode={mode}) — subgenomes={subs} trait={trait_label} ===",
          flush=True)
    # preflight_interact has already verified every mapping fingerprint; avoid
    # hashing very large BIMs a second time while loading dosage matrices.
    subdata = {
        s: _load_subgenome(
            ic["genotype"][s], ic["snp_to_gene"][s], verify_mapping=False)
        for s in subs
    }
    samples = subdata[subs[0]].samples
    for s in subs[1:]:
        if subdata[s].samples != samples:
            print(f"ERROR: interact: sample order mismatch between {subs[0]} and {s}")
            return 1

    from .io import read_delimited

    sample_col = ic.get("sample_col", "sample")
    traits = ([ic["trait"]] if ic.get("trait") else list(ic["multi_trait"]))
    ph_raw = read_delimited(
        ic["phenotype"], dtype={sample_col: "string"})
    ph_raw = ph_raw.loc[ph_raw[sample_col].notna()].copy()
    ph_raw[sample_col] = ph_raw[sample_col].astype(str)
    for trait_name in traits:
        ph_raw[trait_name] = pd.to_numeric(ph_raw[trait_name], errors="coerce")
    # Repeated-site phenotype rows are averaged exactly as in the fit path.
    ph = ph_raw.groupby(sample_col, sort=False)[traits].mean()

    if ic.get("multi_trait") is not None:
        if mode != "pairwise":
            print(f"ERROR: interact: multi_trait is pairwise-only; mode={mode} not supported (future).")
            return 1
        return _run_multitrait(args, ic, subs, out_dir, subdata, samples, ph, t0)

    trait = ic["trait"]
    valid = [s for s in samples if s in ph.index and pd.notna(ph.loc[s, trait])]
    if len(valid) < 10:
        print(f"ERROR: interact: only {len(valid)} samples overlap genotype and "
              "non-missing phenotype for {trait!r}; need >= 10")
        return 1
    sample_idx = np.array([samples.index(s) for s in valid])
    y_raw = np.array([float(ph.loc[s, trait]) for s in valid])

    burden = ic.get("burden", {})
    cap = int(burden.get("cap", 150))
    min_snp = int(burden.get("min_snp", 2))
    burden_maf = float(burden.get("maf_min", 0.01))
    dominance_adjust = bool(burden.get("dominance_adjust", False))
    calib = ic.get("calibration", {})
    # Production default = encoding-invariant omniB + kinship-preserving parametric bootstrap;
    # the legacy REF-burden product + Freedman-Lane permutation stays available as explicit opt-in.
    statistic = str(ic.get("statistic", "omniB")).lower()
    primary_weighting = str(ic.get("primary_weighting", "unweighted")).lower()
    if primary_weighting not in ("unweighted", "weighted"):
        print(f"ERROR: interact.primary_weighting must be 'unweighted' or 'weighted'; "
              f"got '{primary_weighting}'.")
        return 2
    if primary_weighting == "weighted" and statistic in {"omnib", "triad3"}:
        print(f"ERROR: primary_weighting='weighted' is not implemented for statistic={statistic}; "
              "use statistic=burden or primary_weighting=unweighted.")
        return 2
    if primary_weighting == "weighted" and ic.get("multi_trait"):
        print("ERROR: primary_weighting='weighted' is not implemented for multi-trait scans; "
              "the pleiotropy ACAT has no weighted counterpart yet.")
        return 2
    primary_transform = str(ic.get("primary_transform", "INT")).upper()
    if primary_transform not in ("INT", "RAW"):
        print(f"ERROR: interact.primary_transform must be 'INT' or 'raw'; got "
              f"'{primary_transform}'.")
        return 2
    default_multiplicity = (
        "bootstrap_minp"
        if statistic == "triad3" or (mode == "group" and statistic == "omnib")
        else "bonferroni")
    primary_multiplicity = str(
        ic.get("primary_multiplicity", default_multiplicity)).lower()
    if primary_multiplicity not in (
            "bonferroni", "permutation_minp", "bootstrap_minp"):
        print("ERROR: interact.primary_multiplicity must be 'bonferroni', "
              f"'bootstrap_minp', or 'permutation_minp'; got '{primary_multiplicity}'.")
        return 2
    if statistic == "triad3" and primary_multiplicity != "bootstrap_minp":
        print("ERROR: statistic=triad3 requires primary_multiplicity='bootstrap_minp'; "
              "analytic Bonferroni is descriptive only.")
        return 2
    if primary_multiplicity == "permutation_minp" and (statistic != "burden" or mode in group_modes
                                                       or ic.get("multi_trait")):
        print("ERROR: primary_multiplicity='permutation_minp' is only defined for the "
              "single-trait two-subgenome burden pair scan.")
        return 2
    if primary_multiplicity == "permutation_minp" and primary_weighting == "weighted":
        print("ERROR: weighted permutation min-P is not implemented; the permuted statistic would "
              "have to be the weighted one. Use primary_multiplicity=bonferroni.")
        return 2
    if statistic == "triad3" and mode != "triad":
        print("ERROR: statistic=triad3 requires mode=triad and exactly three subgenomes.")
        return 2
    calib_method = str(calib.get(
        "method", "bootstrap" if statistic in {"omnib", "triad3"} else "permutation"
    )).lower()
    if (statistic, calib_method) not in (
            ("omnib", "bootstrap"), ("triad3", "bootstrap"),
            ("burden", "permutation")):
        print(f"ERROR: interact: statistic=omniB or triad3 requires "
              f"calibration.method=bootstrap; statistic=burden requires permutation; got "
              f"statistic={statistic}, calibration.method={calib_method}.")
        return 1
    perm_B = int(calib.get("perm_B", calib.get("B", 2000)))
    boot_B = int(calib.get("B", calib.get("perm_B", 2000)))
    boot_seed = int(calib.get("seed", 2026))
    configured_feature_seed = burden.get("feature_seed")
    feature_seed = (
        None
        if configured_feature_seed is None else int(configured_feature_seed)
    )
    calibration_qa_only = bool(calib.get("qa_only", False))
    checkpoint_cfg = calib.get("checkpoint") or {}
    checkpoint_dir = (
        checkpoint_cfg.get("root")
        if checkpoint_cfg.get("enabled", False) else None)
    checkpoint_block_size = int(checkpoint_cfg.get("block_size", 25))
    n_pc = int(burden.get("n_pc", 3))
    grm_cfg = ic.get("grm", {})
    grm_method = grm_cfg.get(
        "method",
        "grm_from_X" if mode == "group" and statistic == "omnib"
        else "compute_grm_maf")
    maf_min = float(grm_cfg.get("maf_min", 0.01))
    n_jobs = int(args.n_jobs)

    # optional fixed-effect covariates (genotype PCs and/or a y-independent covariate file)
    cov_arg, cov_label = _build_cov_arg(ic.get("covariates"), valid)
    if cov_arg:
        print(f"  covariates: {cov_label}", flush=True)

    # opt-in full genome-wide ranking dump (outputs.full_ranking: true)
    # Canonical formal families always serialize their complete ranking.  It is
    # part of the inferential record, not an optional plotting convenience.
    dump_on = (
        mode == "group" and statistic == "omnib"
    ) or bool(cfg.get("outputs", {}).get("full_ranking", False))

    def _dump_path(transform):
        return str(out_dir / f"interact_{trait}_ranking_{mode}_{transform}.tsv") if dump_on else None

    def _burden_path(transform):
        return (str(out_dir / f"interact_{trait}_topburdens_{transform}.tsv")
                if dump_on else None)

    canonical_family_provenance = {}
    canonical_parallel_execution = {}
    if mode == "group" and statistic == "omnib":
        family = master_family
        hypothesis_unit = str(ic["hypothesis_unit"]).lower()
        family_scope = str(ic.get("family_scope", "primary_only")).lower()
        print(
            f"  n={len(valid)} groups(raw)={len(family.group_ids)} "
            f"(n_sub={len(subs)}) ({time.time()-t0:.1f}s)",
            flush=True,
        )
        r = run_group_scan_omnib(
            subdata, family, y_raw, sample_idx,
            hypothesis_unit=hypothesis_unit,
            family_scope=family_scope,
            cap=cap,
            n_pc=n_pc,
            transform="INT",
            bootstrap_B=boot_B,
            bootstrap_seed=boot_seed,
            feature_seed=feature_seed,
            n_jobs=n_jobs,
            grm_method=grm_method,
            maf_min=maf_min,
            burden_maf=burden_maf,
            min_snp=min_snp,
            covariates=cov_arg,
            full_dump_path=_dump_path("INT"),
            inferential=not calibration_qa_only,
            checkpoint_dir=checkpoint_dir,
            checkpoint_block_size=checkpoint_block_size,
            checkpoint_manifest_context=(
                verified_launch.checkpoint_context
                if verified_launch is not None else None),
        )
        r.trait = trait
        results = {"INT": r.__dict__}
        canonical_family_provenance = dict(
            (r.model_diagnostics or {}).get("family_provenance") or {})
        canonical_parallel_execution = dict(
            (r.model_diagnostics or {}).get("parallel_execution") or {})
        required_provenance = {
            "group_family_sha256", "edge_family_sha256",
            "n_groups_raw", "n_unique_edges",
        }
        missing_provenance = sorted(
            required_provenance - canonical_family_provenance.keys())
        if missing_provenance:
            raise RuntimeError(
                "canonical group scanner omitted required family provenance: "
                + ", ".join(missing_provenance))
        qa_diagnostics = (
            ((r.model_diagnostics or {}).get("bootstrap_fwer") or {})
            .get("qa_diagnostics") or {})
        authority = (
            "QA_bootstrap(no formal rejections, diagnostic_minP_emp="
            f"{qa_diagnostics.get('empirical_p')})"
            if calibration_qa_only else
            f"formal_bootFWER(nsig={r.n_sig})")
        print(
            f"  [INT] G={r.G} statistic=omniB "
            f"primary={hypothesis_unit} minP={r.min_p:.3g} {authority}",
            flush=True,
        )
        if canonical_parallel_execution:
            print(
                "  [parallel] "
                f"requested={canonical_parallel_execution.get('requested_jobs')} "
                f"effective={canonical_parallel_execution.get('effective_jobs')} "
                f"backend={canonical_parallel_execution.get('backend')} "
                f"inner_threads={canonical_parallel_execution.get('inner_threads')}",
                flush=True,
            )
        for hit in (r.sig or []):
            print(
                f"      BOOTSTRAP-FWER HIT {hit['hypothesis_id']} "
                f"p={hit['p_interaction']:.3g}")
        n_units = len(family.group_ids)
    elif mode in group_modes:
        # gene_<S> columns -> n-tuple per homoeolog group; key `groups` (generic) or legacy `triads`
        group_file = ic.get("groups") or ic.get("triads")
        if not group_file:
            print(f"ERROR: interact mode={mode} needs a homoeolog-group TSV via "
                  f"`interact.groups` (or legacy `interact.triads` for 3 subgenomes).")
            return 1
        triads = _load_pairs(group_file, subs)
        triad_weights = _load_pair_weights(ic.get("weights"), subs) if ic.get("weights") else None
        print(f"  n={len(valid)} groups(raw)={len(triads)} (n_sub={len(subs)})"
              + (f" weights={len(triad_weights)}" if triad_weights else "")
              + f" ({time.time()-t0:.1f}s)", flush=True)
        results = {}

        def _g3(v):  # None-safe ".3g" — omnibus fields are None when G=0 / raw transform
            return f"{v:.3g}" if isinstance(v, (int, float)) else str(v)

        if statistic == "omnib":
            for transform in ("INT", "raw"):
                r = run_clique_scan_omnib(
                    subdata, triads, y_raw, sample_idx, cap=cap, n_pc=n_pc, transform=transform,
                    inferential=(transform.upper() == primary_transform),
                    bootstrap_B=(boot_B if transform == "INT" else 0), bootstrap_seed=boot_seed,
                    n_jobs=n_jobs, grm_method=grm_method, maf_min=maf_min,
                    burden_maf=burden_maf, min_snp=min_snp, covariates=cov_arg,
                    full_dump_path=_dump_path(transform))
                r.trait = trait
                results[transform] = r.__dict__
                te = r.tail_excess.get("n_below_0.001") if r.tail_excess else None
                print(f"  [{transform}] G={r.G} statistic=omniB group-omnibus min_p={r.min_p:.3g} "
                      f"nsig(Bonf α={r.bonferroni_alpha:.1e})={r.n_sig} "
                      f"bootFWER(minP_emp={r.minp_boot_emp})"
                      + (f" tail(p<1e-3 obs={te['observed']} null={te['null_mean']:.1f} "
                         f"emp_p={te['empirical_p']:.3g})" if te else ""), flush=True)
                for h in (r.sig or []):
                    print(f"      HIT {h['pair']} p={h['p']:.3g}")
        elif statistic == "triad3":
            for transform in ("INT", "raw"):
                r = run_triad3_scan(
                    subdata, triads, y_raw, sample_idx, cap=cap,
                    transform=transform,
                    inferential=(
                        transform.upper() == primary_transform
                        and not calibration_qa_only),
                    bootstrap_B=(boot_B if transform.upper() == primary_transform
                                 else 0),
                    bootstrap_seed=boot_seed, n_jobs=n_jobs,
                    grm_method=grm_method, maf_min=maf_min,
                    burden_maf=burden_maf, min_snp=min_snp,
                    covariates=cov_arg,
                    full_dump_path=_dump_path(transform))
                r.trait = trait
                results[transform] = r.__dict__
                degenerate_b = int(
                    ((r.model_diagnostics or {}).get("bootstrap_fwer") or {})
                    .get("n_degenerate_replicates", 0))
                print(
                    f"  [{transform}] G={r.G} statistic=triad3 "
                    f"conditional {'×'.join(subs)} minP={r.min_p:.3g} "
                    f"λ_obs={r.lambda_gc_obs:.3f} "
                    f"analytic_screen(Bonf α={r.bonferroni_alpha:.1e})="
                    f"{r.analytic_screen_n} "
                    + (
                        f"QA_bootFWER(no formal rejections, "
                        f"minP_emp={r.minp_boot_emp})"
                        if calibration_qa_only else
                        f"formal_bootFWER(nsig={r.n_sig}, "
                        f"minP_emp={r.minp_boot_emp})"
                    )
                    + (f" degenerate_bootstrap={degenerate_b}/{r.bootstrap_B}"
                       if degenerate_b else ""),
                    flush=True)
                for h in (r.sig or []):
                    print(
                        f"      BOOTSTRAP-FWER HIT {h['triad']} p={h['p']:.3g} "
                        f"residual_ratio={h['target_residual_ratio']:.3g}")
                if not r.sig:
                    for h in (r.analytic_screen_sig or [])[:5]:
                        print(
                            f"      ANALYTIC SCREEN ONLY {h['triad']} "
                            f"p={h['p']:.3g} "
                            f"residual_ratio={h['target_residual_ratio']:.3g}")
        else:
            for transform in ("INT", "raw"):
                r = run_clique_scan(subdata, triads, y_raw, sample_idx, cap=cap, transform=transform,
                                    dominance_adjust=dominance_adjust,
                                    inferential=(transform.upper() == primary_transform),
                                    primary_weighting=primary_weighting,
                                    perm_B=(perm_B if transform.upper() == primary_transform
                                            else 0), n_jobs=n_jobs,
                                    grm_method=grm_method, maf_min=maf_min,
                                    min_snp=min_snp, triad_weights=triad_weights,
                                    covariates=cov_arg, full_dump_path=_dump_path(transform))
                results[transform] = r
                print(f"  [{transform}] G={r.get('G')} "
                      f"triad_ACAT_omnibus={_g3(r.get('triad_acat_omnibus'))} "
                      f"emp={r.get('triad_acat_omnibus_emp')}"
                      + (f" | weighted_omnibus={_g3(r['weighted']['triad_acat_omnibus_weighted'])}"
                         if r.get("weighted") else ""), flush=True)
                go = r.get("group_omnibus")
                if go:
                    a = go.get("bonferroni_alpha")
                    astr = f"{a:.1e}" if isinstance(a, float) else go.get("per_test_alpha_rule")
                    print(f"    [primary: group omnibus] minP={_g3(go['min_p'])} "
                          f"adjP={_g3(go['min_p_adjusted_bonferroni'])} "
                          f"nsig(α={astr})={go['n_sig']} "
                          f"valid={go['n_valid']}/{go['n_planned']}")
                    for h in (go["sig"] or [])[:5]:
                        print(f"        GROUP HIT {h['triad']} p={h['p']:.3g} "
                              f"adjP={h['p_adjusted_bonferroni']:.3g}")
                for tag, pw in r.get("pairwise", {}).items():
                    wstr = (f" | wACAT={pw['weighted']['acat_weighted']:.3g} "
                            f"wnsig={pw['weighted']['bonferroni_n_sig']}" if pw.get("weighted")
                            else "")
                    print(f"    {tag}: ACAT={_g3(pw['acat'])} minP={_g3(pw['min_p'])} "
                          f"λ_obs={pw['lambda_gc_obs']:.3f} "
                          f"nsig(α={pw['bonferroni_alpha']:.1e})={pw['n_sig']}{wstr}")  # noqa: E501
                    for h in (pw["sig"] or []):
                        print(f"        HIT {h['triad']} p={h['p']:.3g}")
        n_units = len(triads)
    else:
        pairs = _load_pairs(ic["pairs"], subs)
        pair_weights = _load_pair_weights(ic.get("weights"), subs) if ic.get("weights") else None
        print(f"  n={len(valid)} pairs(raw)={len(pairs)}"
              + (f" weights={len(pair_weights)}" if pair_weights else "")
              + f" ({time.time()-t0:.1f}s)", flush=True)
        results = {}
        for transform in ("INT", "raw"):
            if statistic == "omnib":
                # encoding-invariant primary: omniB + kinship-preserving parametric bootstrap
                r = run_pair_scan_omnib(
                    subdata, pairs, y_raw, sample_idx, cap=cap, n_pc=n_pc, transform=transform,
                    inferential=(
                        transform.upper() == primary_transform
                        and not calibration_qa_only),
                    bootstrap_B=(boot_B if transform == "INT" else 0), bootstrap_seed=boot_seed,
                    n_jobs=n_jobs, pair_subs=(subs[0], subs[1]), grm_method=grm_method,
                    maf_min=maf_min, burden_maf=burden_maf,
                    min_snp=min_snp, covariates=cov_arg,
                    full_dump_path=_dump_path(transform),
                    burden_dump_path=_burden_path(transform),
                    primary_multiplicity=primary_multiplicity)
                r.trait = trait
                results[transform] = r.__dict__
                te = r.tail_excess.get("n_below_0.001") if r.tail_excess else None
                decision_label = (
                    "QA_bootFWER(no formal rejections)"
                    if calibration_qa_only else
                    ("formal_bootFWER" if primary_multiplicity == "bootstrap_minp"
                     else "Bonferroni"))
                print(f"  [{transform}] G={r.G} statistic=omniB minP={r.min_p:.3g} "
                      f"λ_obs={r.lambda_gc_obs:.3f} nsig({decision_label})={r.n_sig} "
                      f"bootFWER(minP_emp={r.minp_boot_emp} thr05="
                      f"{r.minp_boot_threshold if r.minp_boot_threshold is None else f'{r.minp_boot_threshold:.2g}'})"
                      + (f" tail(p<1e-3 obs={te['observed']} null={te['null_mean']:.1f} "
                         f"emp_p={te['empirical_p']:.3g})" if te else ""), flush=True)
                for h in (r.sig or []):
                    print(f"      HIT {h['pair']} p={h['p']:.3g}")
                continue
            r = run_pair_scan(subdata, pairs, y_raw, sample_idx, cap=cap, transform=transform,
                              inferential=(transform.upper() == primary_transform
                                           and not calibration_qa_only),
                              perm_B=(perm_B if transform.upper() == primary_transform
                                      else 0), n_jobs=n_jobs,
                              pair_subs=(subs[0], subs[1]), grm_method=grm_method, maf_min=maf_min,
                              min_snp=min_snp,
                              pair_weights=pair_weights, covariates=cov_arg,
                              dominance_adjust=dominance_adjust,
                              primary_weighting=primary_weighting,
                              primary_multiplicity=primary_multiplicity,
                              full_dump_path=_dump_path(transform),
                              burden_dump_path=_burden_path(transform))
            r.trait = trait
            results[transform] = r.__dict__
            decision_label = (
                "QA_permFWER(no formal rejections)" if calibration_qa_only else
                ("formal_permFWER" if primary_multiplicity == "permutation_minp"
                 else "Bonferroni"))
            print(f"  [{transform}] G={r.G} pair_ACAT={r.pair_acat:.3g} emp={r.pair_acat_emp} "
                  f"minP={r.min_p:.3g} λ_obs={r.lambda_gc_obs:.3f} λ_perm={r.lambda_gc_perm_median} "
                  f"nsig({decision_label})={r.n_sig} "
                  f"permFWER(minP_emp={r.minp_perm_emp} thr05={r.minp_perm_threshold:.2g})", flush=True)
            for h in (r.sig or []):
                print(f"      HIT {h['pair']} p={h['p']:.3g}")
            if r.weighted:
                print(f"      [weighted] ACAT={r.weighted['acat_weighted']:.3g} "
                      f"nsig={r.weighted['bonferroni_n_sig']}", flush=True)
                for h in (r.weighted["sig"] or []):
                    print(f"        WHIT {h['pair']} p={h['p']:.3g} w={h['weight']:.2f}")
        n_units = len(pairs)

    from . import __version__
    weights_path = ic.get("weights")
    weights_sha = None
    if weights_path and Path(weights_path).exists():
        weights_sha = hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()[:16]
    provenance = dict(version=__version__, mode=mode, grm_method=grm_method, maf_min=maf_min,
                      burden_cap=cap, burden_min_snp=min_snp,
                      burden_maf_min=burden_maf, dominance_adjust=dominance_adjust,
                      perm_B=perm_B, n_samples=len(valid), n_units_raw=n_units,
                      psd_floor=1e-6 if grm_method == "grm_from_X" else None,
                      covariate_policy=(cov_label if cov_arg else
                                        "none: subgenome-stratified GRMs only (no PCs/covariates)"),
                      covariates_detail=results["INT"].get("covariates"),
                      covariate_permutation=("freedman-lane (residualize on C, permute residuals; "
                                             "reduces to y-shuffle when intercept-only)"
                                             if cov_arg else "y-shuffle"),
                      covariate_firewall=("covariates (genotype PCs / environment) must be "
                                          "y-independent and the PC count fixed a priori; PCs enter "
                                          "both the null-LMM mean model and the per-pair GLS design"),
                      weights_source=weights_path, weights_sha256=weights_sha,
                      weights_firewall="weights must be y-independent (DL/HEB) and frozen "
                                       "pre-association; not enforced by the tool",
                      primary_weighting=primary_weighting,
                      primary_multiplicity=primary_multiplicity,
                      primary_transform=primary_transform,
                      statistic=("omniB" if statistic == "omnib" else statistic),
                      calibration_method=calib_method,
                      calibration_qa_only=calibration_qa_only,
                      transform_firewall=(
                          "canonical group omniB emits INT formal inference only; "
                          "RAW is not emitted and cannot spend alpha"
                          if mode == "group" and statistic == "omnib" else
                          "both transforms are scanned but only the primary one "
                          "emits rejections; the other is a sensitivity analysis "
                          "whose rejection fields are null"),
                      weighting_firewall="rejection fields are emitted only for the predeclared "
                                         "procedure; the weighted and unweighted Bonferroni tests "
                                         "each spend the full alpha over the same hypotheses",
                      full_ranking=dump_on,
                      full_ranking_note=(
                          "Full per-unit ranking TSV is descriptive (every callable unit). "
                          "For omniB it includes minor-burden, PC1 and kernel-Hadamard component "
                          "p-values; the smallest component localizes evidence but is not a "
                          "separately calibrated discovery. Inference remains on the predeclared "
                          "primary statistic and multiplicity procedure."),
                      config_path=str(args.config),
                      config_sha256=hashlib.sha256(
                          Path(args.config).read_bytes()).hexdigest())
    if mode == "group" and statistic == "omnib":
        provenance.update({
            "mode": "group",
            "hypothesis_unit": str(ic["hypothesis_unit"]).lower(),
            "subset_order": 2,
            "family_scope": str(
                ic.get("family_scope", "primary_only")).lower(),
            "grm_scope": "all_subgenomes",
            "parallel_execution": canonical_parallel_execution,
            **canonical_family_provenance,
        })
    payload = dict(tool="homoeogwas", command="interact", mode=mode, subgenomes=subs, trait=trait,
                   provenance=provenance, results=results)
    fp = out_dir / f"interact_{trait}.json"
    fp.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False))
    print(f"homoeogwas interact -> {fp} ({time.time()-t0:.1f}s)")
    # best-effort: auto-generate the distinctive interaction figures into the run
    # dir (like `fit`). R is optional and this never fails the stats run; opt out
    # with outputs.plots: false.
    if cfg.get("outputs", {}).get("plots", True):
        try:
            from .cli import _autoplot_interact_figures
            _autoplot_interact_figures(out_dir)
        except Exception as exc:
            print(f"[interact] figure auto-generation skipped: {exc}")
    return 0


def add_interact_subparser(sub) -> None:
    ap = sub.add_parser(
        "interact",
        help=("gene-resolution homoeolog interaction scan from a YAML config "
              "(omniB default; experimental triad3 and legacy burden opt-ins)"))
    ap.add_argument("-c", "--config", required=True, help="YAML run-config path")
    ap.add_argument("-o", "--out-dir", default=None, help="override outputs.out_dir")
    ap.add_argument("--n-jobs", type=int, default=8,
                    help=("worker processes for canonical group omniB; "
                          "legacy bootstrap/permutation engines may differ"))
