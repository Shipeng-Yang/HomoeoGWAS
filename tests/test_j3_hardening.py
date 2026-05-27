"""J=3 hardening tests — Phase 2 M2.4.4 (Hadamard kernel inclusion).

The M2.4.1-M2.4.3 engine (fit_multi_reml / compare_nested_reml / boundary_lrt /
bootstrap_lrt_table / pve_sensitivity_grid) is written J-agnostic but was only
exercised at J=2. M2.4.4 adds a third kernel (the homoeolog Hadamard kernel),
so these tests lock down the J=3 behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from homoeogwas import fit_multi_reml, hadamard_kernel, normalize_kernel
from homoeogwas.diagnostics import (
    bootstrap_lrt_table,
    boundary_lrt_table,
    compare_nested_reml,
    pve_sensitivity_grid,
)


def _toy_psd(n: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, r))
    K = M @ M.T
    K += 1e-3 * np.eye(n)
    K *= n / np.trace(K)
    return K


def _toy_3kernel(n: int, seed: int):
    """y simulated from A+C+e; hom = trace-norm Hadamard(K_A, K_C) (truly 0)."""
    rng = np.random.default_rng(seed)
    K_A = _toy_psd(n, 20, seed=seed)
    K_C = _toy_psd(n, 24, seed=seed + 1)
    K_hom = normalize_kernel(hadamard_kernel({"A": K_A, "C": K_C}), mode="trace")
    L_A = np.linalg.cholesky(K_A + 1e-8 * np.eye(n))
    L_C = np.linalg.cholesky(K_C + 1e-8 * np.eye(n))
    y = (5.0
         + np.sqrt(0.45) * (L_A @ rng.standard_normal(n))
         + np.sqrt(0.35) * (L_C @ rng.standard_normal(n))
         + np.sqrt(0.20) * rng.standard_normal(n))
    X = np.ones((n, 1))
    return y, X, {"A": K_A, "C": K_C, "hom": K_hom}


def test_fit_multi_reml_j3_runs():
    """3-kernel REML: 4 variance components, PVE sums to 1, full diagnostics."""
    y, X, kernels = _toy_3kernel(n=200, seed=0)
    res = fit_multi_reml(y, X, kernels, n_starts=5, random_state=1)
    assert set(res.sigma2.keys()) == {"A", "C", "hom", "e"}
    assert set(res.pve.keys()) == {"A", "C", "hom", "e"}
    assert abs(sum(res.pve.values()) - 1.0) < 1e-9
    # pairwise kernel correlation is a 3x3 dict
    assert set(res.kernel_corr.keys()) == {"A", "C", "hom"}
    assert np.isfinite(res.kernel_design_cond)
    assert res.n_starts == 5


def test_compare_nested_reml_j3_exhaustive_8_models():
    """J=3 exhaustive => 2^3 = 8 models (e + 3 singletons + 3 pairs + full)."""
    y, X, kernels = _toy_3kernel(n=160, seed=1)
    cmp = compare_nested_reml(y, X, kernels, strategy="exhaustive")
    assert set(cmp.fits.keys()) == {
        "e", "A+e", "C+e", "hom+e", "A+C+e", "A+hom+e", "C+hom+e", "A+C+hom+e",
    }
    assert len(cmp.likelihood_table) == 8


def test_boundary_lrt_table_j3_default_7_contrasts():
    """Default pairs at J=3: 3 singletons + 3 leave-one-out + null->full = 7."""
    y, X, kernels = _toy_3kernel(n=160, seed=2)
    cmp = compare_nested_reml(y, X, kernels, strategy="exhaustive")
    tbl = boundary_lrt_table(cmp, "default")
    assert len(tbl) == 7
    # headline contrast present
    hl = tbl[(tbl.null_model == "A+C+e") & (tbl.alt_model == "A+C+hom+e")]
    assert len(hl) == 1
    assert hl.iloc[0]["df_added"] == 1


def test_boundary_lrt_table_j3_all_nested_19():
    """Every strict-nested pair among the 8 J=3 models = 19."""
    y, X, kernels = _toy_3kernel(n=140, seed=3)
    cmp = compare_nested_reml(y, X, kernels, strategy="exhaustive")
    tbl = boundary_lrt_table(cmp, "all_nested")
    assert len(tbl) == 19


def test_bootstrap_lrt_table_j3_headline():
    """Bootstrap the explicit headline contrast at J=3."""
    y, X, kernels = _toy_3kernel(n=130, seed=4)
    cmp = compare_nested_reml(y, X, kernels, strategy="exhaustive")
    tbl = bootstrap_lrt_table(
        cmp, pairs=[("A+C+e", "A+C+hom+e")],
        y=y, X=X, kernels=kernels,
        B=20, seed=42, n_jobs=1, min_success_rate=0.5,
    )
    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert row["null_model"] == "A+C+e" and row["alt_model"] == "A+C+hom+e"
    assert 0.0 <= row["bootstrap_p"] <= 1.0
    assert row["B_usable"] > 0


def test_pve_sensitivity_grid_j3():
    """Sensitivity grid at J=3: pve_hom column, 4 components in drift_table."""
    y, X, kernels = _toy_3kernel(n=160, seed=5)
    res = pve_sensitivity_grid(y, X, kernels)
    assert len(res.table) == 12
    assert {"pve_A", "pve_C", "pve_hom", "pve_e"} <= set(res.table.columns)
    assert res.all_components == ["A", "C", "hom", "e"]
    # drift_table: 4 components x 4 axes
    assert set(res.drift_table["component"]) == {"A", "C", "hom", "e"}
    assert len(res.drift_table) == 4 * 4
    pve_sum = res.table[["pve_A", "pve_C", "pve_hom", "pve_e"]].sum(axis=1)
    assert np.allclose(pve_sum.to_numpy(), 1.0, atol=1e-9)


def test_hadamard_normalize_equivalence():
    """trace-norm(Hadamard(raw)) == trace-norm(Hadamard(trace-norm inputs)).

    Hadamard of c1·A and c2·C scales the product by c1·c2; trace-normalisation
    cancels any positive scalar, so the canonical hom kernel is the same either
    way. M2.4.4 builds hom from raw GRMs; this confirms that is well-defined.
    """
    n = 80
    G_A = _toy_psd(n, 15, seed=10) * 7.3       # arbitrary scales
    G_C = _toy_psd(n, 15, seed=11) * 0.21
    hom_from_raw = normalize_kernel(hadamard_kernel({"A": G_A, "C": G_C}), mode="trace")
    A_n = normalize_kernel(G_A, mode="trace")
    C_n = normalize_kernel(G_C, mode="trace")
    hom_from_norm = normalize_kernel(hadamard_kernel({"A": A_n, "C": C_n}), mode="trace")
    np.testing.assert_allclose(hom_from_raw, hom_from_norm, rtol=1e-10, atol=1e-10)


def test_fit_multi_reml_j3_near_collinear_does_not_crash():
    """hom = Hadamard(A,C) is correlated with A/C; the 3-kernel fit must still
    converge and report a finite design condition number + boundary flags."""
    y, X, kernels = _toy_3kernel(n=150, seed=6)
    res = fit_multi_reml(y, X, kernels, n_starts=10, random_state=0)
    assert np.isfinite(res.log_lik)
    assert np.isfinite(res.kernel_design_cond)
    assert all(np.isfinite(v) and v >= 0 for v in res.sigma2.values())
    # boundary_components is a list (hom may legitimately sit at 0)
    assert isinstance(res.boundary_components, list)


# --- J=4 (wheat A/B/D + hom, M2.4.5) ---------------------------------

def _toy_4kernel(n: int, seed: int):
    """y from A+B+D+e; hom = trace-norm Hadamard(A,B,D). Models the wheat
    A/B/D additive + homoeolog Hadamard 4-kernel set used by M2.4.5."""
    rng = np.random.default_rng(seed)
    K_A = _toy_psd(n, 20, seed=seed)
    K_B = _toy_psd(n, 22, seed=seed + 1)
    K_D = _toy_psd(n, 24, seed=seed + 2)
    K_hom = normalize_kernel(
        hadamard_kernel({"A": K_A, "B": K_B, "D": K_D}), mode="trace")
    parts = []
    for K, s in ((K_A, 0.30), (K_B, 0.25), (K_D, 0.25)):
        L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
        parts.append(np.sqrt(s) * (L @ rng.standard_normal(n)))
    y = 5.0 + sum(parts) + np.sqrt(0.20) * rng.standard_normal(n)
    return y, np.ones((n, 1)), {"A": K_A, "B": K_B, "D": K_D, "hom": K_hom}


def test_compare_nested_reml_j4_exhaustive_16_models():
    """J=4 exhaustive => 2^4 = 16 models; wheat headline contrast present."""
    y, X, kernels = _toy_4kernel(n=170, seed=20)
    cmp = compare_nested_reml(y, X, kernels, strategy="exhaustive")
    assert len(cmp.fits) == 16
    assert "A+B+D+hom+e" in cmp.fits
    tbl = boundary_lrt_table(cmp, "default")
    hl = tbl[(tbl.null_model == "A+B+D+e") & (tbl.alt_model == "A+B+D+hom+e")]
    assert len(hl) == 1
    assert hl.iloc[0]["df_added"] == 1


def test_fit_multi_reml_j4_runs():
    """4-kernel REML: 5 variance components, PVE sums to 1."""
    y, X, kernels = _toy_4kernel(n=160, seed=21)
    res = fit_multi_reml(y, X, kernels, n_starts=8, random_state=1)
    assert set(res.pve.keys()) == {"A", "B", "D", "hom", "e"}
    assert abs(sum(res.pve.values()) - 1.0) < 1e-9
    assert np.isfinite(res.kernel_design_cond)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
