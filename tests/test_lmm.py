"""Unit tests for src/homoeogwas/lmm.py — Phase 2 M2.3."""
from __future__ import annotations

import numpy as np
import pytest

from homoeogwas.lmm import REMLResult, fit_reml


def _toy_psd(n: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, r))
    K = X @ X.T
    K += 1e-3 * np.eye(n)
    return K


def _simulate(n: int, h2_true: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """y = X β + u + e with known σ_g² and σ_e² implied by h²."""
    rng = np.random.default_rng(seed)
    K = _toy_psd(n, r=30, seed=seed)
    # Normalize trace to n so σ_g² 直接对 h² 解读
    K *= n / np.trace(K)
    sigma_g2 = h2_true
    sigma_e2 = 1.0 - h2_true
    # u from K: sample u = L z, where K = L L^T
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    u = np.sqrt(sigma_g2) * (L @ rng.standard_normal(n))
    e = np.sqrt(sigma_e2) * rng.standard_normal(n)
    beta_true = 5.0
    X = np.ones((n, 1))
    y = X @ np.array([beta_true]) + u + e
    return y, X, K, sigma_g2, sigma_e2


def test_reml_returns_finite_components():
    y, X, K, _, _ = _simulate(n=80, h2_true=0.5, seed=0)
    res = fit_reml(y, X, K, backend="cpu")
    assert isinstance(res, REMLResult)
    assert np.isfinite(res.sigma_g2) and res.sigma_g2 >= 0
    assert np.isfinite(res.sigma_e2) and res.sigma_e2 >= 0
    assert 0.0 <= res.h2 <= 1.0
    assert np.isfinite(res.log_lik)
    assert res.backend_used == "cpu"


def test_reml_recovers_h2_on_synthetic_data():
    """With n=200 + decent kernel, profiled REML should recover h² within 0.1."""
    y, X, K, sigma_g2_true, sigma_e2_true = _simulate(n=200, h2_true=0.6, seed=42)
    res = fit_reml(y, X, K, backend="cpu")
    # h² 估计应在真值 ±0.15 内 (有限样本 noise)
    assert abs(res.h2 - 0.6) < 0.2, f"h² estimate {res.h2} far from true 0.6"


def test_reml_clips_tiny_negative_eigenvalues():
    """Small negative eigenvalues from roundoff get clipped, larger ones raise."""
    n = 30
    K = _toy_psd(n, r=10, seed=0)
    # 注入 tiny 负值 (within eig_tol)
    K_bad_tiny = K - 1e-12 * np.eye(n)
    y = np.random.default_rng(0).standard_normal(n)
    X = np.ones((n, 1))
    res = fit_reml(y, X, K_bad_tiny, backend="cpu", eig_tol=1e-8)
    assert np.isfinite(res.h2)
    # 注入 large 负值
    K_bad_big = K - 0.5 * np.eye(n)
    with pytest.raises(ValueError, match="not be PSD|< -eig_tol"):
        fit_reml(y, X, K_bad_big, backend="cpu", eig_tol=1e-8)


def test_reml_input_validation():
    n = 20
    K = _toy_psd(n, r=5, seed=0)
    y = np.zeros(n)
    X = np.ones((n, 1))
    # y wrong dim
    with pytest.raises(ValueError, match=r"y must be 1D"):
        fit_reml(y[:, None], X, K)
    # X wrong sample count
    with pytest.raises(ValueError, match="sample dimensions mismatch"):
        fit_reml(y, X[:5], K)
    # K not square
    with pytest.raises(ValueError, match=r"K must be square"):
        fit_reml(y, X, np.zeros((n, n + 1)))
    # NaN in y
    y_bad = y.copy()
    y_bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_reml(y_bad, X, K)


def test_reml_backend_gpu_missing_cupy_raises():
    """If cupy unavailable, backend='gpu' must raise; backend='auto' falls back."""
    n = 20
    K = _toy_psd(n, r=5, seed=0)
    y = np.random.default_rng(0).standard_normal(n)
    X = np.ones((n, 1))
    try:
        import cupy  # noqa: F401
        # If CuPy present, auto picks gpu
        res_auto = fit_reml(y, X, K, backend="auto")
        assert res_auto.backend_used == "gpu"
        res_gpu = fit_reml(y, X, K, backend="gpu")
        assert res_gpu.backend_used == "gpu"
        # CPU/GPU 一致性
        res_cpu = fit_reml(y, X, K, backend="cpu")
        assert abs(res_cpu.log_lik - res_gpu.log_lik) < 1e-4
        assert abs(res_cpu.h2 - res_gpu.h2) < 1e-4
    except ImportError:
        res_auto = fit_reml(y, X, K, backend="auto")
        assert res_auto.backend_used == "cpu"
        with pytest.raises(RuntimeError, match="CuPy not importable"):
            fit_reml(y, X, K, backend="gpu")


def test_reml_invalid_backend_string():
    n = 20
    K = _toy_psd(n, r=5, seed=0)
    y = np.zeros(n)
    X = np.ones((n, 1))
    with pytest.raises(ValueError, match="invalid backend"):
        fit_reml(y, X, K, backend="bogus")  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
