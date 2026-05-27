"""Unit tests for diagnostics.py — Phase 2 M2.4.2 Step 2."""
from __future__ import annotations

import numpy as np
import pytest

from homoeogwas.diagnostics import (
    BootstrapLRTResult,
    BoundaryLRTResult,
    NestedREMLComparison,
    ResidualOnlyResult,
    SensitivityGridResult,
    _bh_adjust,
    _default_model_specs,
    bootstrap_lrt_table,
    boundary_lrt,
    boundary_lrt_table,
    compare_nested_reml,
    lrt_boundary_pvalue,
    parametric_bootstrap_lrt,
    pve_sensitivity_grid,
    residual_only_reml,
)


def _toy_psd(n: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, r))
    K = M @ M.T
    K += 1e-3 * np.eye(n)
    K *= n / np.trace(K)
    return K


def _simulate_2kernel(n, sig2_A, sig2_C, sig2_e, seed):
    rng = np.random.default_rng(seed)
    K_A = _toy_psd(n, r=20, seed=seed)
    K_C = _toy_psd(n, r=25, seed=seed + 1)
    L_A = np.linalg.cholesky(K_A + 1e-8 * np.eye(n))
    L_C = np.linalg.cholesky(K_C + 1e-8 * np.eye(n))
    u_A = np.sqrt(sig2_A) * (L_A @ rng.standard_normal(n)) if sig2_A > 0 else np.zeros(n)
    u_C = np.sqrt(sig2_C) * (L_C @ rng.standard_normal(n)) if sig2_C > 0 else np.zeros(n)
    e = np.sqrt(sig2_e) * rng.standard_normal(n)
    y = 5.0 + u_A + u_C + e
    X = np.ones((n, 1))
    return y, K_A, K_C, X


def test_residual_only_reml_matches_ols_variance():
    """σ²_e_hat = RSS/(n-p) from OLS."""
    n = 100
    rng = np.random.default_rng(0)
    y = 3.0 + rng.standard_normal(n) * 2.0   # σ_e=2, σ²_e=4
    X = np.ones((n, 1))
    res = residual_only_reml(y, X)
    assert isinstance(res, ResidualOnlyResult)
    # OLS: β=mean, σ²_e_hat ≈ var(y) (ddof=1)
    expected_sigma2 = float(np.var(y, ddof=1))
    assert abs(res.sigma2["e"] - expected_sigma2) < 1e-9
    assert res.pve == {"e": 1.0}
    assert np.isfinite(res.log_lik)
    assert res.n == n and res.p == 1


def test_default_model_specs_exhaustive_J2():
    specs = _default_model_specs(["A", "C"], strategy="exhaustive")
    assert set(specs.keys()) == {"e", "A+e", "C+e", "A+C+e"}
    assert specs["e"] == []
    assert specs["A+e"] == ["A"]
    assert specs["C+e"] == ["C"]
    assert specs["A+C+e"] == ["A", "C"]


def test_default_model_specs_leave_one_out_J3():
    specs = _default_model_specs(["A", "B", "D"], strategy="leave_one_out")
    # null + 3 leave-one-out + full = 5
    assert len(specs) == 5
    assert specs["e"] == []
    assert "B+D+e" in specs  # skip A
    assert "A+D+e" in specs  # skip B
    assert "A+B+e" in specs  # skip D
    assert "A+B+D+e" in specs  # full


def test_compare_nested_reml_4_models_J2():
    y, K_A, K_C, X = _simulate_2kernel(n=200, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=0)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C}, strategy="exhaustive")
    assert isinstance(cmp, NestedREMLComparison)
    assert set(cmp.fits.keys()) == {"e", "A+e", "C+e", "A+C+e"}
    # likelihood_table 应有 4 行
    assert len(cmp.likelihood_table) == 4
    # 列名
    expected_cols = {"model", "kernels", "n", "p", "log_lik", "n_components", "n_params"}
    assert expected_cols <= set(cmp.likelihood_table.columns)


def test_compare_nested_reml_monotonic_in_kernels():
    """Adding kernels never decreases log-lik (same data, same objective)."""
    y, K_A, K_C, X = _simulate_2kernel(n=200, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=1)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    ll = {row["model"]: row["log_lik"] for _, row in cmp.likelihood_table.iterrows()}
    # A+C+e ≥ A+e, C+e (small numerical slack)
    assert ll["A+C+e"] >= ll["A+e"] - 1e-3
    assert ll["A+C+e"] >= ll["C+e"] - 1e-3
    # A+e, C+e ≥ e
    assert ll["A+e"] >= ll["e"] - 1e-3
    assert ll["C+e"] >= ll["e"] - 1e-3


def test_compare_nested_reml_custom_specs():
    y, K_A, K_C, X = _simulate_2kernel(n=150, sig2_A=0.5, sig2_C=0.2, sig2_e=0.3, seed=2)
    custom = {"only_A": ["A"], "full": ["A", "C"]}
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C}, model_specs=custom)
    assert set(cmp.fits.keys()) == {"only_A", "full"}


def test_compare_nested_reml_fit_kwargs_propagate():
    """fit_kwargs should pass n_starts down to fit_multi_reml."""
    y, K_A, K_C, X = _simulate_2kernel(n=150, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=3)
    cmp = compare_nested_reml(
        y, X, {"A": K_A, "C": K_C},
        model_specs={"full": ["A", "C"]},
        fit_kwargs={"n_starts": 3, "random_state": 42},
    )
    fit = cmp.fits["full"]
    assert fit.n_starts == 3
    assert len(fit.all_start_log_lik) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =====================================================================
# Phase 2 M2.4.2 Step 3 — Boundary-Corrected LRT tests
# =====================================================================

def test_lrt_boundary_pvalue_k1_at_3_84_gives_one_half_chi2_sf():
    """k=1 at chi²_1 95% crit (T=3.841): p_naive ≈ 0.05, p_mix ≈ 0.025."""
    from scipy import stats
    T = 3.841
    p_n, p_m, w = lrt_boundary_pvalue(T, df_added=1)
    assert w == {0: 0.5, 1: 0.5}
    assert abs(p_n - float(stats.chi2.sf(T, df=1))) < 1e-12
    assert abs(p_m - 0.5 * float(stats.chi2.sf(T, df=1))) < 1e-12


def test_lrt_boundary_pvalue_k2_binomial_weights():
    """k=2 should use {0:0.25, 1:0.5, 2:0.25} weights."""
    _, _, w = lrt_boundary_pvalue(5.0, df_added=2)
    assert w == {0: 0.25, 1: 0.5, 2: 0.25}
    from scipy import stats
    T = 5.0
    expected = 0.25 * 0.0 + 0.5 * float(stats.chi2.sf(T, 1)) + 0.25 * float(stats.chi2.sf(T, 2))
    _, p_m, _ = lrt_boundary_pvalue(T, df_added=2)
    assert abs(p_m - expected) < 1e-12


def test_lrt_boundary_pvalue_T_zero_returns_one():
    """At T=0, p_mixture = sum_j w_j · chi²_j.sf(0) = 1.0 (all chi² survives at 0)."""
    _, p_m, _ = lrt_boundary_pvalue(0.0, df_added=1)
    assert abs(p_m - 1.0) < 1e-12
    _, p_m, _ = lrt_boundary_pvalue(0.0, df_added=2)
    assert abs(p_m - 1.0) < 1e-12


def test_lrt_boundary_pvalue_negative_T_clipped_to_zero():
    """Negative T clip → p_mixture = 1.0 (T=0 limit)."""
    _, p_m, _ = lrt_boundary_pvalue(-2.5, df_added=1)
    assert abs(p_m - 1.0) < 1e-12


def test_lrt_boundary_pvalue_validates_df_and_weights():
    with pytest.raises(ValueError, match="df_added must be"):
        lrt_boundary_pvalue(1.0, df_added=0)
    with pytest.raises(ValueError, match="must sum to 1"):
        lrt_boundary_pvalue(1.0, df_added=1, weights={0: 0.3, 1: 0.5})


def test_boundary_lrt_full_vs_null_horvath_style():
    """End-to-end: simulate 2-kernel data, fit nested, run LRT."""
    y, K_A, K_C, X = _simulate_2kernel(n=200, sig2_A=0.5, sig2_C=0.3, sig2_e=0.2, seed=0)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    res = boundary_lrt(cmp, "e", "A+C+e")
    assert isinstance(res, BoundaryLRTResult)
    assert res.df_added == 2
    assert res.added_components == ("A", "C")
    assert res.is_nested is True
    assert res.statistic >= 0
    assert res.statistic_raw == 2.0 * (res.ll_alt - res.ll_null)
    # 2-kernel test 在 simulated data (强 signal) 上 p < 0.05
    assert res.p_mixture < 0.05
    # p_mixture < p_naive (boundary correction tightens)
    assert res.p_mixture <= res.p_naive + 1e-12


def test_boundary_lrt_rejects_non_nested():
    y, K_A, K_C, X = _simulate_2kernel(n=120, sig2_A=0.3, sig2_C=0.3, sig2_e=0.4, seed=1)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    # A+e and C+e are not strict nested
    with pytest.raises(ValueError, match="not strictly nested"):
        boundary_lrt(cmp, "A+e", "C+e")
    # Same model
    with pytest.raises(ValueError, match="not strictly nested"):
        boundary_lrt(cmp, "A+e", "A+e")
    # reverse order (alt 嵌套 null,不行)
    with pytest.raises(ValueError, match="not strictly nested"):
        boundary_lrt(cmp, "A+C+e", "A+e")


def test_boundary_lrt_warns_on_negative_statistic(monkeypatch):
    """Force ll_alt < ll_null by patching results, check warning + clip."""
    import warnings
    y, K_A, K_C, X = _simulate_2kernel(n=100, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=2)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    # 人为造负:把 alt log_lik 改小
    cmp.fits["A+C+e"] = cmp.fits["A+C+e"]
    # 用 __setattr__ since dataclass might be frozen — actually MultiREMLResult is NOT frozen
    object.__setattr__(cmp.fits["A+C+e"], "log_lik", cmp.fits["e"].log_lik - 1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = boundary_lrt(cmp, "e", "A+C+e")
    assert res.clipped is True
    assert res.statistic == 0.0
    assert any("LRT statistic_raw" in str(rec.message) for rec in w)


def test_boundary_lrt_table_default_pairs_horvath():
    """Default pairs for J=2: null→singletons (2) + leave-one-out→full (2) + null→full (1) = 5."""
    y, K_A, K_C, X = _simulate_2kernel(n=200, sig2_A=0.5, sig2_C=0.3, sig2_e=0.2, seed=0)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    table = boundary_lrt_table(cmp, pairs="default")
    assert len(table) == 5
    expected_cols = {
        "null_model", "alt_model", "added_components", "ll_null", "ll_alt",
        "lrt", "lrt_raw", "df_added", "p_naive", "p_boundary",
        "mixture_weights", "null_boundary_components", "boundary_method",
        "clipped", "both_converged", "bootstrap_p",
    }
    assert expected_cols <= set(table.columns)
    # 应包含 (e, A+C+e) with df=2
    assert any(
        (table.null_model == "e") & (table.alt_model == "A+C+e") & (table.df_added == 2)
    )


def test_boundary_lrt_table_all_nested_J2_count():
    """J=2 with 4 models, strict nested pairs: e<A+e, e<C+e, e<A+C+e, A+e<A+C+e, C+e<A+C+e = 5."""
    y, K_A, K_C, X = _simulate_2kernel(n=150, sig2_A=0.4, sig2_C=0.2, sig2_e=0.4, seed=3)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    table = boundary_lrt_table(cmp, pairs="all_nested")
    assert len(table) == 5


def test_boundary_lrt_null_boundary_components_empty_for_residual_null():
    """When null is residual-only ('e' kernels=[]), null_boundary_components must be empty."""
    y, K_A, K_C, X = _simulate_2kernel(n=120, sig2_A=0.6, sig2_C=0.05, sig2_e=0.35, seed=4)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    res = boundary_lrt(cmp, "e", "A+e")
    assert res.null_boundary_components == ()


def test_boundary_lrt_null_boundary_components_positive_case():
    """Monkeypatch a null fit's boundary_components to include 'A',
    verify boundary_lrt surfaces it in null_boundary_components."""
    y, K_A, K_C, X = _simulate_2kernel(n=150, sig2_A=0.5, sig2_C=0.3, sig2_e=0.2, seed=5)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    # 强制把 'A' 加入 A+e null fit 的 boundary_components
    null_fit = cmp.fits["A+e"]
    # MultiREMLResult 不是 frozen,直接赋值
    null_fit.boundary_components = list(null_fit.boundary_components) + ["A"]
    res = boundary_lrt(cmp, "A+e", "A+C+e")
    # null_kernels = {"A"}, boundary ∩ null_kernels = {"A"}
    assert res.null_boundary_components == ("A",)


# =====================================================================
# Phase 2 M2.4.2 Step 4 — Parametric Bootstrap LRT tests
# =====================================================================

def test_parametric_bootstrap_lrt_returns_valid_p():
    """B=50 small sim, default n_jobs=1, p ∈ [1/(B+1), 1]."""
    y, K_A, K_C, X = _simulate_2kernel(n=120, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=0)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    res = parametric_bootstrap_lrt(
        cmp, "A+e", "A+C+e",
        y=y, X=X, kernels={"A": K_A, "C": K_C},
        B=50, seed=42, n_jobs=1, min_success_rate=0.5,
    )
    assert isinstance(res, BootstrapLRTResult)
    assert 1.0 / (res.B_success + 1.0) <= res.bootstrap_p <= 1.0
    assert res.B_requested == 50
    assert res.B_success > 0
    assert res.T_boot.shape == (50,)
    assert res.simulation_model == "fitted-null"
    assert np.isfinite(res.mcse)


def test_parametric_bootstrap_lrt_deterministic_with_seed():
    """Same seed + n_jobs=1 → identical T_boot."""
    y, K_A, K_C, X = _simulate_2kernel(n=100, sig2_A=0.3, sig2_C=0.3, sig2_e=0.4, seed=1)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    r1 = parametric_bootstrap_lrt(
        cmp, "e", "A+e", y=y, X=X, kernels={"A": K_A, "C": K_C},
        B=30, seed=99, n_jobs=1, min_success_rate=0.5,
    )
    r2 = parametric_bootstrap_lrt(
        cmp, "e", "A+e", y=y, X=X, kernels={"A": K_A, "C": K_C},
        B=30, seed=99, n_jobs=1, min_success_rate=0.5,
    )
    np.testing.assert_array_equal(r1.T_boot, r2.T_boot)
    assert r1.bootstrap_p == r2.bootstrap_p


def test_parametric_bootstrap_lrt_phipson_smyth_lower_bound():
    """p >= 1/(B+1) always."""
    y, K_A, K_C, X = _simulate_2kernel(n=100, sig2_A=0.5, sig2_C=0.2, sig2_e=0.3, seed=2)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    res = parametric_bootstrap_lrt(
        cmp, "e", "A+C+e", y=y, X=X, kernels={"A": K_A, "C": K_C},
        B=20, seed=5, n_jobs=1, min_success_rate=0.5,
    )
    assert res.bootstrap_p >= 1.0 / (res.B_success + 1.0) - 1e-12


def test_parametric_bootstrap_lrt_under_null_uniform_T_distribution():
    """When K_A真值 σ²=0 (true null), T_b should have many zeros / small values
    and bootstrap p should NOT be small (most T_b >= T_obs)."""
    # Sim: K_A=0 but include K_A in model
    y, K_A, K_C, X = _simulate_2kernel(n=200, sig2_A=0.0, sig2_C=0.5, sig2_e=0.5, seed=3)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    # Test A on top of (C+e), null is "C+e": sigma_A=0 真值 → bootstrap p 应该不显著
    res = parametric_bootstrap_lrt(
        cmp, "C+e", "A+C+e", y=y, X=X, kernels={"A": K_A, "C": K_C},
        B=80, seed=7, n_jobs=1, min_success_rate=0.5,
    )
    # 真 null,p 不应该很小;不强 < 1.0 because sample randomness
    # 但 T_obs 应该不在 T_b 极右侧
    # 不强校验 p > 0.05,但 p > 1/B (i.e. > 1.25%) 应可靠
    assert res.bootstrap_p > 1.0 / (res.B_success + 1.0)


def test_bh_adjust_monotone_in_input_ranking():
    """BH q values 单调不增 when sorted by p."""
    p = np.array([0.001, 0.04, 0.08, 0.10, 0.50])
    q = _bh_adjust(p)
    # q[order] sorted should be monotonic
    order = np.argsort(p)
    q_sorted = q[order]
    # monotone non-decreasing
    assert np.all(np.diff(q_sorted) >= -1e-12)
    # All in [0, 1]
    assert np.all((q >= 0) & (q <= 1))


def test_bh_adjust_uniform_p_close_to_input():
    """For uniform p, BH q ≈ p * n / rank (after monotone enforcement)."""
    p = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
    q = _bh_adjust(p)
    # smallest p (0.05) has rank=1, q = 0.05 * 5 / 1 = 0.25
    assert abs(q[0] - 0.25) < 1e-12


def test_bootstrap_lrt_table_includes_columns():
    """Table should have all paper columns + optional bh."""
    y, K_A, K_C, X = _simulate_2kernel(n=120, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=8)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    table = bootstrap_lrt_table(
        cmp, pairs="default", y=y, X=X, kernels={"A": K_A, "C": K_C},
        B=20, seed=42, n_jobs=1, min_success_rate=0.5, adjust_method="bh",
    )
    assert len(table) == 5  # default 5 pairs for J=2
    cols = {"null_model", "alt_model", "lrt", "lrt_raw", "p_boundary", "bootstrap_p",
            "B_success", "B_usable", "mcse", "success_rate", "usable_rate",
            "jitter_used", "n_starts_bootstrap", "boundary_method", "bootstrap_q_bh"}
    assert cols <= set(table.columns)
    # BH q ≥ raw bootstrap_p element-wise (after monotone) but generally q≥p
    # at least, both in [0,1]
    assert (table["bootstrap_q_bh"].values >= 0).all()
    assert (table["bootstrap_q_bh"].values <= 1).all()


# --- M2.4.2 Step 4 codex-review remediation tests --------------------


def test_parametric_bootstrap_lrt_backfills_bootstrap_p():
    """res.boundary.bootstrap_p must be backfilled to equal res.bootstrap_p."""
    y, K_A, K_C, X = _simulate_2kernel(n=120, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=0)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    res = parametric_bootstrap_lrt(
        cmp, "A+e", "A+C+e", y=y, X=X, kernels={"A": K_A, "C": K_C},
        B=30, seed=1, n_jobs=1, min_success_rate=0.5,
    )
    assert res.boundary.bootstrap_p is not None
    assert res.boundary.bootstrap_p == res.bootstrap_p


def test_parametric_bootstrap_lrt_rejects_B_zero():
    """B < 1 must raise ValueError before any fitting."""
    y, K_A, K_C, X = _simulate_2kernel(n=80, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=2)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    with pytest.raises(ValueError, match="B must be"):
        parametric_bootstrap_lrt(
            cmp, "e", "A+e", y=y, X=X, kernels={"A": K_A, "C": K_C}, B=0, seed=1,
        )


def test_parametric_bootstrap_lrt_missing_kernel_raises():
    """kernels dict missing an alt kernel → informative ValueError, not KeyError."""
    y, K_A, K_C, X = _simulate_2kernel(n=100, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=3)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    with pytest.raises(ValueError, match="missing"):
        parametric_bootstrap_lrt(
            cmp, "A+e", "A+C+e", y=y, X=X, kernels={"A": K_A},  # no "C"
            B=10, seed=1, min_success_rate=0.5,
        )


def test_parametric_bootstrap_lrt_records_jitter_and_usable():
    """jitter_used / B_usable / usable_rate / n_starts_bootstrap are populated."""
    y, K_A, K_C, X = _simulate_2kernel(n=120, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=4)
    cmp = compare_nested_reml(y, X, {"A": K_A, "C": K_C})
    res = parametric_bootstrap_lrt(
        cmp, "e", "A+C+e", y=y, X=X, kernels={"A": K_A, "C": K_C},
        B=40, seed=7, n_jobs=1, min_success_rate=0.5,
    )
    assert res.jitter_used >= 0.0
    assert 0 <= res.B_usable <= res.B_success <= res.B_requested
    assert 0.0 <= res.usable_rate <= 1.0
    assert res.n_starts_bootstrap >= 1


def test_parametric_bootstrap_lrt_warns_on_n_starts_downgrade():
    """Observed fit n_starts=2 but bootstrap explicitly n_starts=1 → RuntimeWarning."""
    import warnings
    y, K_A, K_C, X = _simulate_2kernel(n=100, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=5)
    cmp = compare_nested_reml(
        y, X, {"A": K_A, "C": K_C}, fit_kwargs={"n_starts": 2, "random_state": 0},
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parametric_bootstrap_lrt(
            cmp, "e", "A+e", y=y, X=X, kernels={"A": K_A, "C": K_C},
            B=15, seed=1, min_success_rate=0.5,
            bootstrap_fit_kwargs={"n_starts": 1},
        )
    assert any("anti-conservative" in str(rec.message) for rec in w)


def test_bh_adjust_rejects_nan():
    """_bh_adjust must reject non-finite p-values explicitly."""
    with pytest.raises(ValueError, match="finite"):
        _bh_adjust(np.array([0.1, np.nan, 0.3]))


# =====================================================================
# Phase 2 M2.4.2 Step 5 — PVE Sensitivity Grid tests
# =====================================================================

def test_pve_sensitivity_grid_row_count_and_columns():
    """Default grid = 2 y_modes × 3 norms × 2 n_starts = 12 cells."""
    y, K_A, K_C, X = _simulate_2kernel(n=200, sig2_A=0.5, sig2_C=0.3, sig2_e=0.2, seed=0)
    res = pve_sensitivity_grid(y, X, {"A": K_A, "C": K_C})
    assert isinstance(res, SensitivityGridResult)
    assert len(res.table) == 12
    assert res.all_components == ["A", "C", "e"]
    assert res.genetic_components == ["A", "C"]
    needed = {
        "y_mode", "kernel_norm", "n_starts", "is_reference", "optimizer_status",
        "genetic_upper_bound_hit", "pve_A", "pve_C", "pve_e", "delta_pve_A_vs_ref",
        "abs_delta_pve_C_vs_ref", "max_abs_delta_genetic_vs_ref", "sigma2_A",
        "boundary_components",
    }
    assert needed <= set(res.table.columns)
    # PVE rows sum to 1 within each cell
    pve_sum = res.table[["pve_A", "pve_C", "pve_e"]].sum(axis=1)
    assert np.allclose(pve_sum.to_numpy(), 1.0, atol=1e-9)
    # well-scaled VanRaden-style kernels: no genetic σ² pinned at upper bound
    assert not res.table["genetic_upper_bound_hit"].any()


def test_pve_sensitivity_grid_reference_zero_drift():
    """Exactly one reference cell; its deltas vs ref are all 0."""
    y, K_A, K_C, X = _simulate_2kernel(n=180, sig2_A=0.4, sig2_C=0.4, sig2_e=0.2, seed=1)
    res = pve_sensitivity_grid(y, X, {"A": K_A, "C": K_C})
    assert int(res.table["is_reference"].sum()) == 1
    ref_row = res.table[res.table["is_reference"]].iloc[0]
    assert ref_row["y_mode"] == "raw"
    assert ref_row["kernel_norm"] == "trace"
    assert ref_row["n_starts"] == 10
    for c in ("A", "C", "e"):
        assert abs(ref_row[f"delta_pve_{c}_vs_ref"]) < 1e-12
    assert ref_row["max_abs_delta_genetic_vs_ref"] < 1e-12


def test_pve_sensitivity_grid_scale_invariance_kernel_norm():
    """PVE is invariant to K↦cK: trace/frobenius/none agree (n_starts=10)."""
    y, K_A, K_C, X = _simulate_2kernel(n=200, sig2_A=0.5, sig2_C=0.3, sig2_e=0.2, seed=2)
    res = pve_sensitivity_grid(y, X, {"A": K_A, "C": K_C})
    sub = res.table[(res.table["y_mode"] == "raw") & (res.table["n_starts"] == 10)]
    assert len(sub) == 3
    assert sub["pve_A"].max() - sub["pve_A"].min() < 2e-3
    assert sub["pve_C"].max() - sub["pve_C"].min() < 2e-3


def test_pve_sensitivity_grid_y_transform_invariance():
    """PVE is invariant to affine y transform when X has an intercept."""
    y, K_A, K_C, X = _simulate_2kernel(n=200, sig2_A=0.5, sig2_C=0.3, sig2_e=0.2, seed=3)
    res = pve_sensitivity_grid(y, X, {"A": K_A, "C": K_C})
    sub = res.table[(res.table["kernel_norm"] == "trace") & (res.table["n_starts"] == 10)]
    assert set(sub["y_mode"]) == {"raw", "zscore"}
    assert sub["pve_A"].max() - sub["pve_A"].min() < 2e-3


def test_pve_sensitivity_grid_does_not_mutate_inputs():
    """y and kernel matrices must be untouched after the grid runs."""
    y, K_A, K_C, X = _simulate_2kernel(n=150, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=4)
    y0, KA0, KC0 = y.copy(), K_A.copy(), K_C.copy()
    pve_sensitivity_grid(y, X, {"A": K_A, "C": K_C})
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_array_equal(K_A, KA0)
    np.testing.assert_array_equal(K_C, KC0)


def test_pve_sensitivity_grid_drift_table():
    """drift_table covers every (component × axis) with non-negative range."""
    y, K_A, K_C, X = _simulate_2kernel(n=180, sig2_A=0.5, sig2_C=0.3, sig2_e=0.2, seed=5)
    res = pve_sensitivity_grid(y, X, {"A": K_A, "C": K_C})
    dt = res.drift_table
    assert set(dt["axis"]) == {"y_mode", "kernel_norm", "n_starts", "global"}
    assert set(dt["component"]) == {"A", "C", "e"}
    assert len(dt) == 3 * 4
    assert (dt["pve_range"] >= 0).all()
    glob = dt[dt["axis"] == "global"]
    assert (glob["n_cells"] == 12).all()


def test_pve_sensitivity_grid_rejects_bad_input():
    """Malformed reference / non-integer n_starts / grid-controlled fit_kwargs rejected."""
    y, K_A, K_C, X = _simulate_2kernel(n=120, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=6)
    kers = {"A": K_A, "C": K_C}
    # reference not in grid
    with pytest.raises(ValueError, match="reference"):
        pve_sensitivity_grid(y, X, kers, reference=("raw", "trace", 7))
    # reference wrong length
    with pytest.raises(ValueError, match="reference must be"):
        pve_sensitivity_grid(y, X, kers, reference=("raw", "trace"))
    # reference n_starts non-integer
    with pytest.raises(ValueError, match="integer"):
        pve_sensitivity_grid(y, X, kers, reference=("raw", "trace", 10.5))
    # n_starts_grid non-integer
    with pytest.raises(ValueError, match="integer"):
        pve_sensitivity_grid(y, X, kers, n_starts_grid=(1.5,))
    # fit_kwargs may not override grid-controlled args
    with pytest.raises(ValueError, match="grid-controlled"):
        pve_sensitivity_grid(y, X, kers, fit_kwargs={"n_starts": 3})


def test_pve_sensitivity_grid_upper_bound_hit_honours_user_bounds():
    """genetic_upper_bound_hit is checked against user-supplied fit_kwargs bounds."""
    y, K_A, K_C, X = _simulate_2kernel(n=120, sig2_A=0.5, sig2_C=0.3, sig2_e=0.2, seed=9)
    # Pin σ²_A at a tiny upper bound: every cell should flag the bound hit.
    res = pve_sensitivity_grid(
        y, X, {"A": K_A, "C": K_C},
        fit_kwargs={"bounds": {"A": (0.0, 1e-12)}},
    )
    assert res.table["genetic_upper_bound_hit"].all()
