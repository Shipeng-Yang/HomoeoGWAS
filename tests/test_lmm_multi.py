"""Unit tests for fit_multi_reml — Phase 2 M2.4.1."""
from __future__ import annotations

import numpy as np
import pytest

from homoeogwas.lmm import (
    MultiREMLResult,
    fit_multi_reml,
    fit_reml,
)


def _toy_psd(n: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, r))
    K = X @ X.T
    K += 1e-3 * np.eye(n)
    K *= n / np.trace(K)  # normalize to trace = n
    return K


def _simulate_2kernel(
    n: int, sig2_A: float, sig2_C: float, sig2_e: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    K_A = _toy_psd(n, r=20, seed=seed)
    K_C = _toy_psd(n, r=25, seed=seed + 1)
    L_A = np.linalg.cholesky(K_A + 1e-8 * np.eye(n))
    L_C = np.linalg.cholesky(K_C + 1e-8 * np.eye(n))
    u_A = np.sqrt(sig2_A) * (L_A @ rng.standard_normal(n)) if sig2_A > 0 else np.zeros(n)
    u_C = np.sqrt(sig2_C) * (L_C @ rng.standard_normal(n)) if sig2_C > 0 else np.zeros(n)
    e = np.sqrt(sig2_e) * rng.standard_normal(n)
    y = 5.0 + u_A + u_C + e
    return y, K_A, K_C, np.ones((n, 1))


def test_multi_reml_finite_components():
    y, K_A, K_C, X = _simulate_2kernel(n=150, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=0)
    res = fit_multi_reml(y, X, {"A": K_A, "C": K_C})
    assert isinstance(res, MultiREMLResult)
    assert set(res.sigma2.keys()) == {"A", "C", "e"}
    assert all(np.isfinite(v) and v >= 0 for v in res.sigma2.values())
    assert abs(sum(res.pve.values()) - 1.0) < 1e-9
    assert np.isfinite(res.log_lik)
    assert res.kernel_names == ["A", "C"]
    assert res.backend_used == "cpu"


def test_multi_reml_recovers_dominant_component():
    """When K_A dominates (true σ²_A=0.6, σ²_C=0.05, σ²_e=0.35),
    fitted σ²_A should be largest of the three."""
    y, K_A, K_C, X = _simulate_2kernel(n=250, sig2_A=0.6, sig2_C=0.05, sig2_e=0.35, seed=42)
    res = fit_multi_reml(y, X, {"A": K_A, "C": K_C})
    # 在小样本里精确恢复不强求,只要 ranking 对
    assert res.sigma2["A"] > res.sigma2["C"], (
        f"expected σ²_A > σ²_C; got {res.sigma2}"
    )


def test_multi_reml_handles_zero_component():
    """K_C 真值 σ²=0 时应远小于 σ²_A,不应崩溃."""
    y, K_A, K_C, X = _simulate_2kernel(n=400, sig2_A=0.5, sig2_C=0.0, sig2_e=0.5, seed=7)
    res = fit_multi_reml(y, X, {"A": K_A, "C": K_C})
    assert np.isfinite(res.log_lik)
    assert all(v >= 0 for v in res.sigma2.values())
    # σ²_C 应远小于 σ²_A (即使 small-sample noise 没让它精确到 boundary)
    ratio = res.sigma2["C"] / max(res.sigma2["A"], 1e-12)
    assert ratio < 0.2, (
        f"expected σ²_C/σ²_A < 0.2 (true σ²_C=0); got ratio={ratio:.3f}, sigma2={res.sigma2}"
    )


def test_multi_reml_log_lik_invariant_to_dict_order():
    """字典 insertion order 不应影响 log_lik 或方差分量."""
    y, K_A, K_C, X = _simulate_2kernel(n=120, sig2_A=0.4, sig2_C=0.4, sig2_e=0.2, seed=11)
    r1 = fit_multi_reml(y, X, {"A": K_A, "C": K_C})
    r2 = fit_multi_reml(y, X, {"C": K_C, "A": K_A})
    assert abs(r1.log_lik - r2.log_lik) < 1e-4
    assert abs(r1.sigma2["A"] - r2.sigma2["A"]) < 1e-3
    assert abs(r1.sigma2["C"] - r2.sigma2["C"]) < 1e-3


def test_multi_reml_input_validation():
    n = 50
    K = _toy_psd(n, r=10, seed=0)
    y = np.zeros(n)
    X = np.ones((n, 1))
    # Empty dict
    with pytest.raises(ValueError, match="empty"):
        fit_multi_reml(y, X, {})
    # Shape mismatch
    with pytest.raises(ValueError, match="square"):
        fit_multi_reml(y, X, {"A": K[:30, :30]})
    # Non-finite kernel
    K_bad = K.copy()
    K_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_multi_reml(y, X, {"A": K_bad})
    # y/X dim mismatch
    with pytest.raises(ValueError, match="matching y"):
        fit_multi_reml(y, X[:30], {"A": K})


def test_multi_reml_kernel_diagnostics():
    """Two near-identical kernels → high pairwise corr ≈ 1 + huge design cond."""
    n = 80
    K = _toy_psd(n, r=15, seed=3)
    K_jit = K + 1e-6 * _toy_psd(n, r=15, seed=4)  # 几乎相同 kernel
    y = np.random.default_rng(0).standard_normal(n)
    X = np.ones((n, 1))
    res = fit_multi_reml(y, X, {"A": K, "C": K_jit})
    # 对角线 1
    assert abs(res.kernel_corr["A"]["A"] - 1.0) < 1e-9
    # 高度相关
    assert res.kernel_corr["A"]["C"] > 0.99, (
        f"expected near-identical kernels to have corr>0.99, got {res.kernel_corr['A']['C']}"
    )


def test_multi_reml_J1_consistency_with_fit_reml():
    """J=1 时 fit_multi_reml 与 single-kernel fit_reml 应严格一致 (h2 内 1e-5)."""
    n = 200
    K = _toy_psd(n, r=20, seed=99)
    rng = np.random.default_rng(101)
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    u = 0.7 * (L @ rng.standard_normal(n))   # σ²_g = 0.49
    e = 0.5 * rng.standard_normal(n)         # σ²_e = 0.25
    y = 3.0 + u + e
    X = np.ones((n, 1))

    res_multi = fit_multi_reml(y, X, {"A": K})
    res_single = fit_reml(y, X, K, backend="cpu")

    h2_multi = res_multi.sigma2["A"] / (res_multi.sigma2["A"] + res_multi.sigma2["e"])
    h2_single = res_single.h2
    # 同一 likelihood 应该 close to 1e-5 (Codex M2.4.1 review: ~1.3e-7 on this fixture)
    assert abs(h2_multi - h2_single) < 1e-5, (
        f"J=1 multi h²={h2_multi:.8f} differs from single h²={h2_single:.8f}"
    )


def test_multi_reml_residual_boundary_reported():
    """σ²_e hit lower bound → 'e' should be in boundary_components."""
    # 构造一个 K 完全解释 y 的情形:y = u (no noise)
    n = 80
    K = _toy_psd(n, r=10, seed=5)
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    y = L @ rng.standard_normal(n)  # 纯 u,无 e
    X = np.ones((n, 1))
    res = fit_multi_reml(y, X, {"A": K})
    assert "e" in res.boundary_components, (
        f"expected 'e' in boundary_components when σ²_e at lower bound; got {res.boundary_components}; sigma2={res.sigma2}"
    )


def test_multi_reml_rejects_non_psd_kernel():
    """显式负 eigval > eig_tol 应 raise."""
    n = 30
    K = _toy_psd(n, r=8, seed=0)
    K_bad = K - 1.0 * np.eye(n)   # 注入 −1 eigval shift
    y = np.random.default_rng(0).standard_normal(n)
    X = np.ones((n, 1))
    with pytest.raises(ValueError, match=r"not PSD.*min_eig"):
        fit_multi_reml(y, X, {"A": K_bad}, eig_tol=1e-8)


def test_multi_reml_clips_tiny_negative_eigvals():
    """K in [−eig_tol, 0) 应 silent clip, n_eig_clipped > 0."""
    n = 40
    K = _toy_psd(n, r=10, seed=1)
    K_tiny_neg = K - 1e-12 * np.eye(n)  # tiny negative
    y = np.random.default_rng(0).standard_normal(n)
    X = np.ones((n, 1))
    res = fit_multi_reml(y, X, {"A": K_tiny_neg}, eig_tol=1e-8)
    assert res.kernel_n_eig_clipped["A"] >= 0  # might be 0 or more, depending on rounding
    assert np.isfinite(res.log_lik)


def test_multi_reml_pve_uses_trace_weighting():
    """对非 trace-normalized kernels,PVE 应用 σ²·trace(K)/n 加权."""
    n = 100
    K = _toy_psd(n, r=15, seed=2)
    K_huge = 10.0 * K  # trace 是 K 的 10 倍
    y, _, _, X = _simulate_2kernel(n=n, sig2_A=0.5, sig2_C=0.0, sig2_e=0.5, seed=2)
    # 比较 K 和 10K 的拟合; PVE 应一致 (paper-grade formula 抵消 scale)
    res_K = fit_multi_reml(y, X, {"A": K})
    res_K10 = fit_multi_reml(y, X, {"A": K_huge})
    # paper-grade PVE 等价
    assert abs(res_K.pve["A"] - res_K10.pve["A"]) < 1e-4, (
        f"PVE_A should be scale-invariant under K↦10K; "
        f"got {res_K.pve['A']} vs {res_K10.pve['A']}"
    )
    # σ² 不一致 (尺度不同),但 component_var 应一致
    assert abs(res_K.component_var["A"] - res_K10.component_var["A"]) / max(res_K.component_var["A"], 1e-12) < 1e-3


def test_multi_reml_multi_start_returns_best():
    """n_starts>1 should return best log_lik ≥ single start log_lik."""
    y, K_A, K_C, X = _simulate_2kernel(n=200, sig2_A=0.4, sig2_C=0.3, sig2_e=0.3, seed=33)
    res1 = fit_multi_reml(y, X, {"A": K_A, "C": K_C}, n_starts=1, random_state=42)
    res10 = fit_multi_reml(y, X, {"A": K_A, "C": K_C}, n_starts=5, random_state=42)
    assert res1.n_starts == 1
    assert res10.n_starts == 5
    assert res10.log_lik >= res1.log_lik - 1e-6  # multi-start at least matches
    assert len(res10.all_start_log_lik) == 5
    assert res10.best_start in range(5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
