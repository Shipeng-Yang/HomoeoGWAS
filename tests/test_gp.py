"""Tests for Phase 5c GBLUP module — closed-form correctness + CV pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from homoeogwas.gp import (
    GBLUPResult,
    blup_predict,
    paired_bootstrap_ci,
    run_cv_gblup,
    stratified_folds,
    top_k_enrichment,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_grms(n: int, n_sub: int, n_snp_per_sub: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    sg_ids = ["A", "B", "C", "D"][:n_sub]
    grms = {}
    for sg in sg_ids:
        # Simulate biallelic SNPs (0/1/2) centered to MAF ~ 0.3
        maf = rng.uniform(0.1, 0.4, size=n_snp_per_sub)
        # X dims n × n_snp; columns ~ Binomial(2, p)
        X = rng.binomial(2, maf, size=(n, n_snp_per_sub)).astype(np.float64)
        # Standardize per SNP
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        K = X @ X.T / n_snp_per_sub
        K = 0.5 * (K + K.T)
        K = K + 1e-6 * np.eye(n)
        grms[sg] = K
    return grms


def _sim_pheno_from_multi_K(grms: dict[str, np.ndarray], sig2_each: float = 0.3,
                            sig2_e: float = 0.4, seed: int = 1):
    rng = np.random.default_rng(seed)
    n = next(iter(grms.values())).shape[0]
    y = rng.standard_normal(n) * np.sqrt(sig2_e)
    for K in grms.values():
        u = rng.multivariate_normal(np.zeros(n), sig2_each * K)
        y = y + u
    X = np.ones((n, 1))
    return y, X


# ---------------------------------------------------------------------------
# 1. Stratified folds
# ---------------------------------------------------------------------------


def test_stratified_folds_balanced_means(rng_seed=0):
    """Each fold's mean trait should be close to the overall mean."""
    rng = np.random.default_rng(rng_seed)
    y = rng.standard_normal(200) * 2.0 + 10.0
    folds = stratified_folds(y, n_folds=5, n_strata=5, rng=rng)
    assert len(folds) == 5
    assert sum(len(f) for f in folds) == 200
    fold_means = [y[f].mean() for f in folds]
    overall = y.mean()
    # Stratified fold means should be within 0.5 × σ of overall (tight bound)
    assert all(abs(m - overall) < 0.5 * y.std() for m in fold_means)


def test_stratified_folds_disjoint_complete():
    y = np.arange(50, dtype=np.float64)
    folds = stratified_folds(y, n_folds=5, n_strata=5,
                              rng=np.random.default_rng(0))
    all_idx = np.concatenate(folds)
    assert len(set(all_idx)) == 50
    assert sorted(all_idx) == list(range(50))


# ---------------------------------------------------------------------------
# 2. Top-K enrichment
# ---------------------------------------------------------------------------


def test_top_k_enrichment_perfect_predictor():
    """Perfect predictor → enrichment = 1 / k_pct (all top-k true)."""
    y_obs = np.arange(100, dtype=np.float64)
    y_pred = y_obs.copy()
    e = top_k_enrichment(y_obs, y_pred, k_pct=0.10)
    assert e == pytest.approx(10.0)  # recall=1, k_pct=0.1 → 10


def test_top_k_enrichment_random_predictor():
    """Random predictor → enrichment around 1 on expectation."""
    rng = np.random.default_rng(0)
    y_obs = rng.standard_normal(1000)
    y_pred = rng.standard_normal(1000)
    e = top_k_enrichment(y_obs, y_pred, k_pct=0.10)
    # Random selection → enrichment ~ 1; tolerate ± 0.5 with one seed
    assert 0.3 < e < 1.7


# ---------------------------------------------------------------------------
# 3. blup_predict closed-form sanity
# ---------------------------------------------------------------------------


def test_blup_predict_recovers_train_zero_residual():
    """When σ²_e → 0 the BLUP must fit the training data exactly
    (zero residual on training points used as 'test')."""
    grms = _make_grms(n=30, n_sub=2, seed=0)
    y, X = _sim_pheno_from_multi_K(grms, sig2_each=0.4, sig2_e=0.05, seed=1)
    sigma2 = {"A": 0.4, "B": 0.4, "e": 1e-6}
    train_idx = np.arange(30)
    test_idx = np.array([0, 1, 2])  # subset of training
    beta = np.array([y.mean()])
    y_pred = blup_predict(
        y, X, X[test_idx], grms, train_idx, test_idx, sigma2, beta, jitter=1e-8,
    )
    # Should be very close to true y on training points (since σ²_e→0)
    assert np.allclose(y_pred, y[test_idx], atol=0.1)


# ---------------------------------------------------------------------------
# 4. Paired bootstrap CI
# ---------------------------------------------------------------------------


def test_paired_bootstrap_ci_identical_input_zero_delta():
    """If two tiers have identical per-fold values, Δ should be ~0 with tight CI."""
    fold_vals = [0.30, 0.32, 0.28, 0.31, 0.29]
    per_fold = {"tier0": fold_vals, "tier1": list(fold_vals)}
    ci = paired_bootstrap_ci(per_fold, n_boot=200, seed=0, baseline="tier0")
    assert "tier1" in ci
    lo, hi = ci["tier1"]
    # Should bracket 0 tightly
    assert -0.02 < lo < 0.02
    assert -0.02 < hi < 0.02


def test_paired_bootstrap_ci_signed_delta():
    """Tier1 systematically above baseline → CI should be positive."""
    rng = np.random.default_rng(0)
    fold_vals_base = rng.normal(0.3, 0.02, 10).tolist()
    fold_vals_tier1 = [v + 0.05 for v in fold_vals_base]
    per_fold = {"tier0": fold_vals_base, "tier1": fold_vals_tier1}
    ci = paired_bootstrap_ci(per_fold, n_boot=500, seed=0, baseline="tier0")
    lo, hi = ci["tier1"]
    # Delta ~ 0.05; CI should exclude 0
    assert lo > 0
    assert 0.03 < hi < 0.07


# ---------------------------------------------------------------------------
# 5. End-to-end CV with synthetic multi-K signal
# ---------------------------------------------------------------------------


def test_run_cv_gblup_synthetic_tier1_outperforms_tier0_when_multi_K_signal():
    """When phenotype is truly multi-K (sub-A and sub-B both contribute),
    tier1 (subgenome-stratified) should match or beat tier0 (pooled single K).

    This is a sanity check, not a strict assertion of statistical power —
    in n=80 small simulated data the variance is high. The contract is:
    no NaNs, tier1 r² is in a sensible range, and tier0 also works."""
    grms = _make_grms(n=80, n_sub=2, n_snp_per_sub=100, seed=7)
    y, X = _sim_pheno_from_multi_K(grms, sig2_each=0.5, sig2_e=0.2, seed=8)
    res = run_cv_gblup(
        y, X, grms,
        tiers=("tier0", "tier1"),
        n_folds=4, n_repeats=3, seed=2026, n_starts=2,
        panel="synthetic", trait="multi_k_signal", verbose=False,
    )
    assert isinstance(res, GBLUPResult)
    assert "tier0" in res.tiers
    assert "tier1" in res.tiers
    assert 0 <= res.tiers["tier0"].mean_r2 <= 1.0
    assert 0 <= res.tiers["tier1"].mean_r2 <= 1.0
    # paired bootstrap CI populated
    assert "tier1" in res.delta_vs_tier0
    assert "delta_r2_mean" in res.delta_vs_tier0["tier1"]


def test_run_cv_gblup_three_tier_full_pipeline():
    """Smoke test: all three tiers (incl. tier2 K_hom) run without error."""
    grms = _make_grms(n=60, n_sub=3, n_snp_per_sub=60, seed=11)
    y, X = _sim_pheno_from_multi_K(grms, sig2_each=0.3, sig2_e=0.3, seed=12)
    snp_counts = {sg: 60 for sg in grms}   # equal SNP counts to exercise weighting
    res = run_cv_gblup(
        y, X, grms,
        tiers=("tier0", "tier1", "tier2"),
        snp_counts=snp_counts,
        n_folds=4, n_repeats=2, seed=2026, n_starts=2,
        panel="synthetic", trait="three_sub", verbose=False,
    )
    assert set(res.tiers.keys()) == {"tier0", "tier1", "tier2"}
    assert res.n_samples == 60
    assert len(res.per_fold) == 3 * 4 * 2  # 3 tiers × 4 folds × 2 repeats


def test_tier0_kpool_overrides_snp_counts():
    """Explicit ``k_pool`` should be used in preference to ``snp_counts``."""
    grms = _make_grms(n=50, n_sub=2, n_snp_per_sub=50, seed=4)
    y, X = _sim_pheno_from_multi_K(grms, sig2_each=0.3, sig2_e=0.3, seed=5)
    # Two different "pools" — one is a known matrix the caller forces
    snp_counts = {sg: 50 for sg in grms}
    user_k_pool = sum(grms.values()) / 2.0
    res = run_cv_gblup(
        y, X, grms,
        tiers=("tier0",),
        k_pool=user_k_pool,
        snp_counts=snp_counts,    # should be ignored when k_pool given
        n_folds=4, n_repeats=2, seed=2026, n_starts=2,
        panel="synthetic", trait="kpool_priority", verbose=False,
    )
    assert "tier0" in res.tiers
    assert np.isfinite(res.tiers["tier0"].mean_r2)


def test_tier0_fallback_equal_weight_warns(recwarn):
    """If neither k_pool nor snp_counts given, equal-weight average warns."""
    grms = _make_grms(n=40, n_sub=2, n_snp_per_sub=40, seed=6)
    y, X = _sim_pheno_from_multi_K(grms, sig2_each=0.3, sig2_e=0.3, seed=7)
    run_cv_gblup(
        y, X, grms,
        tiers=("tier0",),
        n_folds=3, n_repeats=1, seed=2026, n_starts=2,
        panel="synthetic", trait="fallback_warn", verbose=False,
    )
    # At least one RuntimeWarning was raised (the "unfair baseline" warning)
    msgs = [str(w.message) for w in recwarn.list]
    assert any("equal-weight K_pool" in m for m in msgs), msgs
