"""Unit tests for calibration.py — Phase 2 M2.4.3 null-simulation calibration."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homoeogwas.calibration import (
    NullCalibrationResult,
    NullSimulationScenario,
    _chibar_quantile,
    _chibar_sf,
    _chibar_weights,
    _covariance_from_sigma2,
    _wilson_ci,
    run_null_lrt_calibration,
    scenario_from_reduced_fit,
    summarize_type1,
)
from homoeogwas.diagnostics import lrt_boundary_pvalue


def _toy_psd(n: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, r))
    K = M @ M.T
    K += 1e-3 * np.eye(n)
    K *= n / np.trace(K)
    return K


# ---------------------------------------------------------------------
# chi-bar helpers
# ---------------------------------------------------------------------

def test_chibar_weights_binomial():
    assert _chibar_weights(1) == {0: 0.5, 1: 0.5}
    assert _chibar_weights(2) == {0: 0.25, 1: 0.5, 2: 0.25}
    with pytest.raises(ValueError):
        _chibar_weights(0)


def test_chibar_quantile_sf_inverse():
    """quantile(q) then sf() must round-trip to 1-q for q above the atom."""
    for k in (1, 2):
        w = _chibar_weights(k)
        for q in (0.90, 0.95, 0.99):
            t = _chibar_quantile(q, w)
            assert t > 0
            assert abs(_chibar_sf(t, w) - (1.0 - q)) < 1e-8
    # q at/below the atom w_0 -> quantile 0
    assert _chibar_quantile(0.5, _chibar_weights(1)) == 0.0
    assert _chibar_quantile(0.2, _chibar_weights(1)) == 0.0


def test_chibar_quantile_matches_lrt_boundary_pvalue():
    """T = chi-bar q-quantile fed to lrt_boundary_pvalue gives p_mixture = 1-q."""
    for k in (1, 2):
        w = _chibar_weights(k)
        for q in (0.92, 0.97, 0.995):
            t = _chibar_quantile(q, w)
            _, p_mix, _ = lrt_boundary_pvalue(t, df_added=k)
            assert abs(p_mix - (1.0 - q)) < 1e-7


def test_wilson_ci():
    lo, hi = _wilson_ci(50, 1000)
    assert lo < 0.05 < hi
    assert 0.0 <= lo < hi <= 1.0
    # empty -> nan
    lo0, hi0 = _wilson_ci(0, 0)
    assert np.isnan(lo0) and np.isnan(hi0)
    # zero rejections -> lo ~ 0 (Wilson lower bound, modulo float noise)
    lo1, hi1 = _wilson_ci(0, 200)
    assert 0.0 <= lo1 < 1e-9 and hi1 > 0.0


# ---------------------------------------------------------------------
# summarize_type1
# ---------------------------------------------------------------------

def test_summarize_type1_exact_counts():
    """Hand-checked rejection counts, including a p=1 atom."""
    p = np.array([0.001, 0.02, 0.04, 0.20, 0.5, 1.0, 1.0, 1.0, 0.009, 0.30])
    tbl = pd.DataFrame({"p_mixture": p})
    out = summarize_type1(tbl, alphas=(0.05, 0.01), methods=("p_mixture",))
    r05 = out[(out.method == "p_mixture") & (out.alpha == 0.05)].iloc[0]
    r01 = out[(out.method == "p_mixture") & (out.alpha == 0.01)].iloc[0]
    # p <= 0.05 : 0.001, 0.02, 0.04, 0.009  -> 4
    assert r05["n_reject"] == 4
    assert r05["n_used"] == 10
    assert abs(r05["type1_rate"] - 0.4) < 1e-12
    # p <= 0.01 : 0.001, 0.009 -> 2
    assert r01["n_reject"] == 2


def test_summarize_type1_handles_nan():
    """NaN p-values are dropped from the denominator."""
    p = np.array([0.01, np.nan, 0.5, np.nan, 0.02])
    tbl = pd.DataFrame({"p_naive": p})
    out = summarize_type1(tbl, alphas=(0.05,), methods=("p_naive",))
    row = out.iloc[0]
    assert row["n_used"] == 3
    assert row["n_reject"] == 2


# ---------------------------------------------------------------------
# covariance
# ---------------------------------------------------------------------

def test_covariance_from_sigma2():
    n = 40
    K_A = _toy_psd(n, 12, seed=0)
    K_C = _toy_psd(n, 12, seed=1)
    sig = {"A": 2.0, "C": 0.0, "e": 0.5}
    V = _covariance_from_sigma2(sig, {"A": K_A, "C": K_C}, n)
    # C contributes nothing (σ²_C = 0)
    expected = 0.5 * np.eye(n) + 2.0 * (0.5 * (K_A + K_A.T))
    assert np.allclose(V, 0.5 * (expected + expected.T), atol=1e-8)
    assert np.allclose(V, V.T)


# ---------------------------------------------------------------------
# scenario construction
# ---------------------------------------------------------------------

def test_scenario_from_reduced_fit_pins_tested_to_zero():
    """S2: fit A+e on real y; C (tested) must be pinned to 0."""
    n = 160
    rng = np.random.default_rng(3)
    K_A = _toy_psd(n, 20, seed=3)
    K_C = _toy_psd(n, 22, seed=4)
    L_A = np.linalg.cholesky(K_A + 1e-8 * np.eye(n))
    y = 5.0 + np.sqrt(0.6) * (L_A @ rng.standard_normal(n)) + np.sqrt(0.4) * rng.standard_normal(n)
    X = np.ones((n, 1))
    scen = scenario_from_reduced_fit(
        "S2", y, X, {"A": K_A, "C": K_C},
        null_model="A+e", alt_model="A+C+e", n_starts=1, random_state=0,
    )
    assert scen.tested_components == ("C",)
    assert scen.df_added == 1
    assert scen.sigma2_true["C"] == 0.0
    assert scen.sigma2_true["A"] > 0.0
    assert scen.sigma2_true["e"] > 0.0
    assert scen.model_specs == {"A+e": ["A"], "A+C+e": ["A", "C"]}
    assert scen.beta.shape == (1,)


def test_scenario_from_reduced_fit_global_null():
    """S1: null is residual-only 'e'; both A and C pinned to 0."""
    n = 120
    rng = np.random.default_rng(7)
    K_A = _toy_psd(n, 15, seed=7)
    K_C = _toy_psd(n, 15, seed=8)
    y = 2.0 + rng.standard_normal(n)
    X = np.ones((n, 1))
    scen = scenario_from_reduced_fit(
        "S1", y, X, {"A": K_A, "C": K_C},
        null_model="e", alt_model="A+C+e",
    )
    assert scen.df_added == 2
    assert scen.tested_components == ("A", "C")
    assert scen.sigma2_true["A"] == 0.0 and scen.sigma2_true["C"] == 0.0
    assert scen.sigma2_true["e"] > 0.0


def test_scenario_from_reduced_fit_rejects_non_nested():
    n = 80
    K_A = _toy_psd(n, 10, seed=0)
    K_C = _toy_psd(n, 10, seed=1)
    y = np.random.default_rng(0).standard_normal(n)
    X = np.ones((n, 1))
    with pytest.raises(ValueError, match="nested"):
        scenario_from_reduced_fit("bad", y, X, {"A": K_A, "C": K_C},
                                  null_model="A+e", alt_model="C+e")


# ---------------------------------------------------------------------
# run_null_lrt_calibration
# ---------------------------------------------------------------------

def _toy_scenario_S2():
    return NullSimulationScenario(
        name="S2_smoke", null_model="A+e", alt_model="A+C+e",
        model_specs={"A+e": ["A"], "A+C+e": ["A", "C"]},
        sigma2_true={"A": 1.0, "C": 0.0, "e": 1.0},
        beta=np.array([0.0]), df_added=1, tested_components=("C",),
    )


def test_run_null_lrt_calibration_smoke():
    n = 90
    K_A = _toy_psd(n, 14, seed=11)
    K_C = _toy_psd(n, 16, seed=12)
    X = np.ones((n, 1))
    res = run_null_lrt_calibration(
        X, {"A": K_A, "C": K_C}, _toy_scenario_S2(),
        n_sim=12, seed=2026, fit_kwargs={"n_starts": 1}, n_jobs=1,
    )
    assert isinstance(res, NullCalibrationResult)
    assert res.n_sim_requested == 12
    assert res.n_sim_success == 12
    assert len(res.replicate_table) == 12
    assert {"p_naive", "p_mixture", "lrt", "e_at_boundary",
            "null_boundary_components", "alt_boundary_components"} <= set(res.replicate_table.columns)
    assert set(res.type1_table["method"]) == {"p_naive", "p_mixture"}
    # p-values in [0,1]
    for col in ("p_naive", "p_mixture"):
        v = res.replicate_table[col].to_numpy()
        assert np.all((v >= 0) & (v <= 1))
    assert len(res.tail_inflation) == 2
    assert res.df_added == 1
    assert 0.0 <= res.e_boundary_rate <= 1.0


def test_run_null_lrt_calibration_deterministic():
    n = 80
    K_A = _toy_psd(n, 12, seed=21)
    K_C = _toy_psd(n, 12, seed=22)
    X = np.ones((n, 1))
    kw = dict(n_sim=10, seed=99, fit_kwargs={"n_starts": 1}, n_jobs=1)
    r1 = run_null_lrt_calibration(X, {"A": K_A, "C": K_C}, _toy_scenario_S2(), **kw)
    r2 = run_null_lrt_calibration(X, {"A": K_A, "C": K_C}, _toy_scenario_S2(), **kw)
    np.testing.assert_array_equal(
        r1.replicate_table["p_mixture"].to_numpy(),
        r2.replicate_table["p_mixture"].to_numpy(),
    )


def test_run_null_lrt_calibration_njobs_invariant():
    n = 80
    K_A = _toy_psd(n, 12, seed=31)
    K_C = _toy_psd(n, 12, seed=32)
    X = np.ones((n, 1))
    kw = dict(n_sim=10, seed=7, fit_kwargs={"n_starts": 1})
    r1 = run_null_lrt_calibration(X, {"A": K_A, "C": K_C}, _toy_scenario_S2(), n_jobs=1, **kw)
    r2 = run_null_lrt_calibration(X, {"A": K_A, "C": K_C}, _toy_scenario_S2(), n_jobs=2, **kw)
    np.testing.assert_allclose(
        r1.replicate_table["lrt"].to_numpy(),
        r2.replicate_table["lrt"].to_numpy(),
        rtol=0, atol=0,
    )


def test_run_null_lrt_calibration_rejects_non_null_scenario():
    """A scenario whose tested component has σ²!=0 is not a null scenario."""
    n = 70
    K_A = _toy_psd(n, 10, seed=41)
    K_C = _toy_psd(n, 10, seed=42)
    X = np.ones((n, 1))
    bad = NullSimulationScenario(
        name="not_null", null_model="A+e", alt_model="A+C+e",
        model_specs={"A+e": ["A"], "A+C+e": ["A", "C"]},
        sigma2_true={"A": 1.0, "C": 0.7, "e": 1.0},  # C != 0 !
        beta=np.array([0.0]), df_added=1, tested_components=("C",),
    )
    with pytest.raises(ValueError, match="not a null scenario"):
        run_null_lrt_calibration(X, {"A": K_A, "C": K_C}, bad, n_sim=4, seed=0)


def test_run_null_lrt_calibration_rejects_random_state_in_fit_kwargs():
    n = 60
    K_A = _toy_psd(n, 8, seed=51)
    K_C = _toy_psd(n, 8, seed=52)
    X = np.ones((n, 1))
    with pytest.raises(ValueError, match="random_state"):
        run_null_lrt_calibration(
            X, {"A": K_A, "C": K_C}, _toy_scenario_S2(),
            n_sim=4, seed=0, fit_kwargs={"random_state": 1},
        )


# ---------------------------------------------------------------------
# known-good statistical calibration (no LMM fitting)
# ---------------------------------------------------------------------

def test_known_good_chibar_calibration():
    """If T is drawn from the exact chi-bar mixture, p_mixture from
    lrt_boundary_pvalue must reject at ~the nominal rate (calibrated)."""
    from scipy import stats
    rng = np.random.default_rng(2026)
    k = 1
    w = _chibar_weights(k)
    n_sim = 40000
    # draw component index j ~ {0: .5, 1: .5}; T = chi2_j (chi2_0 == 0)
    j = rng.random(n_sim) >= w[0]            # True -> j=1
    T = np.where(j, stats.chi2.rvs(df=1, size=n_sim, random_state=rng), 0.0)
    p_mix = np.array([lrt_boundary_pvalue(t, df_added=k)[1] for t in T])
    tbl = pd.DataFrame({"p_mixture": p_mix})
    out = summarize_type1(tbl, alphas=(0.10, 0.05, 0.01), methods=("p_mixture",))
    for _, row in out.iterrows():
        # well-calibrated: empirical rate within Wilson CI of nominal
        assert row["ci_lo"] <= row["alpha"] <= row["ci_hi"], (
            f"alpha={row['alpha']} not in CI [{row['ci_lo']:.4f},{row['ci_hi']:.4f}]"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
