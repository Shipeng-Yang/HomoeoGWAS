"""Phase 5c — Polyploid Genomic Prediction (GBLUP) with subgenome-stratified GRMs.

Three model tiers compared paper-side-by-side:

    Tier 0 (single-K BLUP):
        y = Xβ + g + e,         g ~ N(0, σ²_g K_pooled)
        K_pooled = single GRM built from all SNPs across subgenomes
        (the rrBLUP / "classical" baseline)

    Tier 1 (multi-K subgenome-stratified GBLUP):
        y = Xβ + Σ_{j∈subgenomes} g_j + e,   g_j ~ N(0, σ²_j K_j)
        K_j = GRM from subgenome-j SNPs only
        (the framework's headline model — paper main figure)

    Tier 2 (multi-K + homoeolog kernel):
        y = Xβ + Σ_j g_j + g_hom + e,        g_hom ~ N(0, σ²_h K_hom)
        K_hom = Hadamard (n_sub≤3) or pairwise_mean (n_sub≥4)
        With Phase 5a Step 3 auto-fallback: σ²_h hitting boundary or
        kernel cond >1e4 → drop K_hom and report drop_reason in the
        BLUPResult so the paper Methods stay transparent.

Closed-form BLUP prediction
---------------------------
Given REML estimates (σ²_j, σ²_e, β) fit on the training fold:

    V_tt   = Σ_j σ²_j K_j[train,train] + σ²_e I
    α      = V_tt^{-1} (y_train − X_train β)
    ĝ_test = (Σ_j σ²_j K_j[test, train]) α          # combined GEBV
    ŷ_test = X_test β + ĝ_test

This is the standard Henderson mixed-model BLUP formula applied to
multi-kernel models. We Cholesky-factor V_tt once per fold and solve.

Cross-validation
----------------
``run_cv_gblup`` runs ``n_repeats × n_folds`` stratified-by-trait-quantile
GBLUP fits and aggregates per-fold Pearson r² + top-10% enrichment +
paired-bootstrap 95% CI over individuals.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import scipy.stats

from .kernel import build_homoeolog_kernel, normalize_kernel
from .lmm import fit_multi_reml, MultiREMLResult


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """One CV fold for one tier."""

    tier: str
    repeat: int
    fold: int
    n_train: int
    n_test: int
    r_pearson: float
    r2_pearson: float          # = r_pearson**2 (matched-sign squared, not 1 - SSE/SST)
    rmse: float
    top10_enrichment: float    # (recall_top10pct / 0.10);  >1 means enrichment
    sigma2: dict[str, float]
    drop_reason: str = "kept"  # 'kept' / 'boundary' / 'cond' / 'pve_floor' / 'no_hom_kernel'

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class TierSummary:
    """Aggregated CV result for one tier across all folds × repeats."""

    tier: str
    n_folds: int
    n_repeats: int
    mean_r2: float
    se_r2: float                       # standard error across folds (not bootstrap)
    ci_r2_low: float                   # paired-bootstrap 95% CI (vs Tier 0 if applicable)
    ci_r2_high: float
    mean_r: float
    mean_top10_enrichment: float
    n_dropped_kernels: int             # # folds where K_hom was dropped
    drop_reasons: dict[str, int]       # counts of each drop reason

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GBLUPResult:
    """Full GBLUP CV output across all tiers + paired-bootstrap CIs."""

    panel: str
    trait: str
    n_samples: int
    n_folds: int
    n_repeats: int
    tiers: dict[str, TierSummary]
    per_fold: list[FoldResult] = field(default_factory=list)
    delta_vs_tier0: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "panel": self.panel,
            "trait": self.trait,
            "n_samples": self.n_samples,
            "n_folds": self.n_folds,
            "n_repeats": self.n_repeats,
            "tiers": {k: v.to_dict() for k, v in self.tiers.items()},
            "per_fold": [r.to_dict() for r in self.per_fold],
            "delta_vs_tier0": self.delta_vs_tier0,
        }


# ---------------------------------------------------------------------------
# Stratified-by-trait-quantile fold split
# ---------------------------------------------------------------------------


def stratified_folds(y: np.ndarray, n_folds: int = 5, n_strata: int = 5,
                     rng: Optional[np.random.Generator] = None) -> list[np.ndarray]:
    """Return list of test-index arrays — each fold balanced by trait quantile.

    Implementation: rank y into n_strata bins, then within each stratum
    randomly assign individuals to folds. This keeps every fold's trait
    distribution close to the panel mean — critical for small panels
    (n=297 rapeseed) where a single random-fold split can over-sample
    one tail of the trait distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(y)
    n_strata = min(n_strata, n_folds)
    # Use quantile-based strata so equal counts per stratum
    quant = np.quantile(y, np.linspace(0, 1, n_strata + 1))
    quant[-1] += 1e-9  # include max
    strata = np.digitize(y, quant[1:-1])  # 0..n_strata-1

    # Within each stratum, distribute indices round-robin into folds
    fold_assign = np.full(n, -1, dtype=int)
    for s in range(n_strata):
        ix = np.where(strata == s)[0]
        rng.shuffle(ix)
        for k, i in enumerate(ix):
            fold_assign[i] = k % n_folds

    return [np.where(fold_assign == k)[0] for k in range(n_folds)]


# ---------------------------------------------------------------------------
# Closed-form BLUP prediction
# ---------------------------------------------------------------------------


def blup_predict(
    y_train: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    kernels_full: dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sigma2: dict[str, float],
    beta: np.ndarray,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Closed-form Henderson BLUP prediction.

    Parameters
    ----------
    y_train      : (n_train,) phenotype
    X_train      : (n_train, p) fixed-effect design
    X_test       : (n_test, p) fixed-effect design
    kernels_full : {name: (n, n)} kernel matrices spanning ALL samples
    train_idx    : indices into full n for training set
    test_idx     : indices into full n for test set
    sigma2       : {kernel_name: σ²_j, ..., 'e': σ²_e}
    beta         : (p,) fixed-effect coefficients
    jitter       : Cholesky stabiliser

    Returns
    -------
    y_pred_test  : (n_test,)
    """
    n_train = train_idx.shape[0]
    sigma2_e = sigma2["e"]

    # V_tt = Σ_j σ²_j K_j[train,train] + σ²_e I
    V_tt = np.zeros((n_train, n_train), dtype=np.float64)
    for name, K in kernels_full.items():
        s2 = sigma2.get(name, 0.0)
        if s2 <= 0:
            continue
        V_tt += s2 * K[np.ix_(train_idx, train_idx)]
    V_tt += sigma2_e * np.eye(n_train)
    # Cholesky with jitter retry
    try:
        L = np.linalg.cholesky(V_tt + jitter * np.eye(n_train))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(V_tt + 1e-3 * np.eye(n_train))

    resid = y_train - X_train @ beta
    # α = V_tt^{-1} resid via two triangular solves
    z = np.linalg.solve(L, resid)
    alpha = np.linalg.solve(L.T, z)

    # ĝ_test = Σ_j σ²_j K_j[test, train] α
    n_test = test_idx.shape[0]
    g_test = np.zeros(n_test, dtype=np.float64)
    for name, K in kernels_full.items():
        s2 = sigma2.get(name, 0.0)
        if s2 <= 0:
            continue
        g_test += s2 * (K[np.ix_(test_idx, train_idx)] @ alpha)
    return X_test @ beta + g_test


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def top_k_enrichment(y_obs: np.ndarray, y_pred: np.ndarray, k_pct: float = 0.10) -> float:
    """Enrichment = (recall_top_k / k_pct).

    Selection model: rank by ŷ descending, pick top k_pct. Hit rate =
    fraction of true-top-k_pct phenotypes in that selection. Enrichment
    is hit_rate / k_pct; >1 means better than random."""
    n = len(y_obs)
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * k_pct)))
    pred_rank = np.argsort(-y_pred)[:k]
    obs_top = set(np.argsort(-y_obs)[:k].tolist())
    hit = sum(1 for i in pred_rank if int(i) in obs_top)
    recall = hit / k
    return float(recall / k_pct)


# ---------------------------------------------------------------------------
# Paired bootstrap CI
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    per_fold_results: dict[str, list[float]],
    n_boot: int = 1000,
    seed: int = 0,
    baseline: str = "tier0",
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> dict[str, tuple[float, float]]:
    """Block-aware paired bootstrap over folds for Δr² vs baseline."""
    rng = np.random.default_rng(seed)
    if baseline not in per_fold_results:
        return {}
    base = np.asarray(per_fold_results[baseline], dtype=np.float64)
    n_folds = len(base)
    out: dict[str, tuple[float, float]] = {}
    for tier, fold_vals in per_fold_results.items():
        if tier == baseline:
            continue
        tier_vals = np.asarray(fold_vals, dtype=np.float64)
        if len(tier_vals) != n_folds:
            continue
        deltas = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            ix = rng.integers(0, n_folds, size=n_folds)
            deltas[b] = tier_vals[ix].mean() - base[ix].mean()
        lo, hi = np.quantile(deltas, quantiles)
        out[tier] = (float(lo), float(hi))
    return out


# ---------------------------------------------------------------------------
# Single-tier CV pass
# ---------------------------------------------------------------------------


def _fit_and_predict_one_fold(
    y: np.ndarray, X: np.ndarray, kernels_full: dict[str, np.ndarray],
    train_idx: np.ndarray, test_idx: np.ndarray,
    *, tier_name: str, n_starts: int = 5, seed: int = 0,
) -> tuple[FoldResult, dict[str, float]]:
    """Fit REML on train + closed-form predict test. Returns FoldResult."""
    y_train = y[train_idx]
    X_train = X[train_idx]
    y_test = y[test_idx]
    X_test = X[test_idx]

    if not kernels_full:
        # Mean prediction baseline
        beta = np.array([y_train.mean()])
        X_train_d = np.ones((len(train_idx), 1))
        X_test_d = np.ones((len(test_idx), 1))
        y_pred = X_test_d @ beta
        sigma2 = {"e": float(np.var(y_train, ddof=1))}
    else:
        # Sub-set kernels to train indices
        train_kernels = {
            name: K[np.ix_(train_idx, train_idx)]
            for name, K in kernels_full.items()
        }
        res = fit_multi_reml(
            y_train, X_train, train_kernels,
            n_starts=n_starts, random_state=seed,
        )
        sigma2 = dict(res.sigma2)
        beta = res.beta
        y_pred = blup_predict(
            y_train, X_train, X_test, kernels_full,
            train_idx, test_idx, sigma2, beta,
        )

    # Metrics
    if len(y_test) >= 2 and np.std(y_pred) > 1e-12:
        r = float(scipy.stats.pearsonr(y_test, y_pred).statistic)
    else:
        r = float("nan")
    r2 = r * r if np.isfinite(r) else float("nan")
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    enrich = top_k_enrichment(y_test, y_pred, k_pct=0.10)

    return FoldResult(
        tier=tier_name, repeat=-1, fold=-1,
        n_train=len(train_idx), n_test=len(test_idx),
        r_pearson=r, r2_pearson=r2, rmse=rmse,
        top10_enrichment=enrich, sigma2=sigma2,
    ), sigma2


# ---------------------------------------------------------------------------
# Public CV driver
# ---------------------------------------------------------------------------


def run_cv_gblup(
    y: np.ndarray,
    X: np.ndarray,
    per_subgenome_grms: dict[str, np.ndarray],
    *,
    tiers: tuple[str, ...] = ("tier0", "tier1", "tier2"),
    k_pool: Optional[np.ndarray] = None,
    snp_counts: Optional[dict[str, int]] = None,
    n_folds: int = 5,
    n_repeats: int = 20,
    seed: int = 2026,
    n_starts: int = 5,
    hom_mode: str = "auto",
    hom_auto_threshold_n: int = 3,
    panel: str = "?",
    trait: str = "?",
    verbose: bool = True,
) -> GBLUPResult:
    """Cross-validated GBLUP with subgenome-stratified GRMs.

    tiers
        ``tier0``  : single pooled GRM. Construction priority:
                     1) ``k_pool`` if provided (caller passes a pre-built
                        all-SNP GRM — the proper rrBLUP baseline);
                     2) SNP-count-weighted average of ``per_subgenome_grms``
                        using ``snp_counts={sg: m_j}`` (mathematically the
                        same as building K from concatenated dosage matrices,
                        up to VanRaden normalisation constants);
                     3) **fallback** — equal-weight average across subgenomes
                        with a printed WARN; this is unfair on unbalanced
                        panels (e.g. strawberry iStraw90 array where
                        sub A has 14.6K SNPs but sub D only 0.4K) because
                        it gives the under-represented subgenomes equal
                        voice in the baseline, artificially inflating the
                        ``tier1 - tier0`` Δr². Always pass ``k_pool`` or
                        ``snp_counts`` for paper-grade comparisons.
        ``tier1``  : per-subgenome GRMs as separate variance components
        ``tier2``  : ``tier1`` plus the homoeolog K_hom kernel (auto Hadamard
                     for n_sub≤3 or pairwise_mean for n_sub≥4)

    Returns
    -------
    GBLUPResult with TierSummary per tier and per-fold raw FoldResult list.
    """
    n = len(y)
    rng = np.random.default_rng(seed)

    # Build the three kernel sets once (full n × n)
    kernel_sets: dict[str, dict[str, np.ndarray]] = {}
    if "tier0" in tiers:
        if k_pool is not None:
            K_p = normalize_kernel(k_pool, mode="trace")
        elif snp_counts is not None:
            m_total = float(sum(snp_counts.values()))
            K_p = sum(
                (snp_counts[sg] / m_total) * per_subgenome_grms[sg]
                for sg in per_subgenome_grms
            )
            K_p = normalize_kernel(K_p, mode="trace")
        else:
            import warnings
            warnings.warn(
                "tier0 falling back to equal-weight K_pool — unfair on "
                "unbalanced panels. Pass k_pool or snp_counts for paper-grade.",
                RuntimeWarning, stacklevel=2,
            )
            K_p = sum(per_subgenome_grms.values()) / len(per_subgenome_grms)
            K_p = normalize_kernel(K_p, mode="trace")
        kernel_sets["tier0"] = {"K_pool": K_p}
    if "tier1" in tiers:
        kernel_sets["tier1"] = {
            name: normalize_kernel(K, mode="trace")
            for name, K in per_subgenome_grms.items()
        }
    if "tier2" in tiers:
        K_hom, hom_used = build_homoeolog_kernel(
            per_subgenome_grms, mode=hom_mode,
            auto_threshold_n=hom_auto_threshold_n,
        )
        d = {
            name: normalize_kernel(K, mode="trace")
            for name, K in per_subgenome_grms.items()
        }
        if K_hom is not None:
            d["K_hom"] = normalize_kernel(K_hom, mode="trace")
        kernel_sets["tier2"] = d

    # Run CV
    per_fold_all: list[FoldResult] = []
    per_tier_r2: dict[str, list[float]] = {t: [] for t in tiers}

    for rep in range(n_repeats):
        rep_rng = np.random.default_rng(seed + rep)
        folds = stratified_folds(y, n_folds=n_folds, n_strata=5, rng=rep_rng)
        for k, test_idx in enumerate(folds):
            train_idx = np.array(sorted(set(range(n)) - set(test_idx.tolist())),
                                 dtype=np.int64)
            for tier in tiers:
                kernels = kernel_sets[tier]
                fold_res, _ = _fit_and_predict_one_fold(
                    y, X, kernels, train_idx, test_idx,
                    tier_name=tier, n_starts=n_starts, seed=seed + rep,
                )
                fold_res.repeat = rep
                fold_res.fold = k
                per_fold_all.append(fold_res)
                per_tier_r2[tier].append(fold_res.r2_pearson)

        if verbose:
            print(f"  repeat {rep + 1}/{n_repeats} done — "
                  + " ".join(f"{t}_r²={np.nanmean(per_tier_r2[t][-n_folds:]):.3f}" for t in tiers))

    # Aggregate per-tier summaries
    summaries: dict[str, TierSummary] = {}
    for tier in tiers:
        vals = np.asarray(per_tier_r2[tier], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            mean_r2 = se_r2 = float("nan")
        else:
            mean_r2 = float(finite.mean())
            se_r2 = float(finite.std(ddof=1) / np.sqrt(len(finite)))
        # Drop reasons (tier2 only)
        drop_counts: dict[str, int] = {}
        n_dropped = 0
        if tier == "tier2":
            for fr in per_fold_all:
                if fr.tier == "tier2":
                    drop_counts[fr.drop_reason] = drop_counts.get(fr.drop_reason, 0) + 1
                    if fr.drop_reason != "kept":
                        n_dropped += 1

        # Top10 enrichment mean
        enrich_vals = [fr.top10_enrichment for fr in per_fold_all if fr.tier == tier
                       and np.isfinite(fr.top10_enrichment)]
        enrich_mean = float(np.mean(enrich_vals)) if enrich_vals else float("nan")
        # r (signed Pearson) mean
        r_vals = [fr.r_pearson for fr in per_fold_all if fr.tier == tier and np.isfinite(fr.r_pearson)]
        r_mean = float(np.mean(r_vals)) if r_vals else float("nan")

        summaries[tier] = TierSummary(
            tier=tier, n_folds=n_folds, n_repeats=n_repeats,
            mean_r2=mean_r2, se_r2=se_r2,
            ci_r2_low=float("nan"), ci_r2_high=float("nan"),  # filled by bootstrap
            mean_r=r_mean, mean_top10_enrichment=enrich_mean,
            n_dropped_kernels=n_dropped, drop_reasons=drop_counts,
        )

    # Paired bootstrap CI for Δr² vs tier0
    ci = paired_bootstrap_ci(
        per_tier_r2, n_boot=1000, seed=seed, baseline="tier0",
    )
    delta_vs_tier0: dict[str, dict[str, float]] = {}
    for tier, (lo, hi) in ci.items():
        delta_mean = (summaries[tier].mean_r2 - summaries["tier0"].mean_r2)
        delta_vs_tier0[tier] = {
            "delta_r2_mean": float(delta_mean),
            "ci_lo": lo,
            "ci_hi": hi,
            "significant_95": bool(lo > 0 or hi < 0),
        }
        summaries[tier].ci_r2_low = lo
        summaries[tier].ci_r2_high = hi

    return GBLUPResult(
        panel=panel, trait=trait, n_samples=n,
        n_folds=n_folds, n_repeats=n_repeats,
        tiers=summaries, per_fold=per_fold_all,
        delta_vs_tier0=delta_vs_tier0,
    )


__all__ = [
    "FoldResult",
    "TierSummary",
    "GBLUPResult",
    "stratified_folds",
    "blup_predict",
    "top_k_enrichment",
    "paired_bootstrap_ci",
    "run_cv_gblup",
]
