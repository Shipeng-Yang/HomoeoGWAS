"""Unit tests for src/homoeogwas/kernel.py — M2.2 scaffold acceptance."""
from __future__ import annotations

import numpy as np
import pytest

from homoeogwas.kernel import hadamard_kernel, normalize_kernel, sum_kernel


def _toy_psd(n: int, r: int, seed: int) -> np.ndarray:
    """Build a deterministic PSD matrix via X X.T + small jitter."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, r))
    K = X @ X.T
    K += 1e-3 * np.eye(n)
    return K


def test_hadamard_kernel_is_elementwise():
    """Hadamard product is exact elementwise multiplication."""
    K_A = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]])
    K_C = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 1.5], [0.0, 1.5, 4.0]])
    K_hom = hadamard_kernel({"A": K_A, "C": K_C})
    assert K_hom.shape == (3, 3)
    assert K_hom.dtype == np.float64
    for i in range(3):
        for j in range(3):
            assert K_hom[i, j] == K_A[i, j] * K_C[i, j]
    # Also check whole-array
    np.testing.assert_allclose(K_hom, K_A * K_C)
    # Inputs not mutated
    assert np.array_equal(K_A, np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]]))


def test_hadamard_kernel_preserves_psd_for_psd_inputs():
    """Schur product theorem: Hadamard of PSD is PSD."""
    K_A = _toy_psd(n=50, r=20, seed=0)
    K_C = _toy_psd(n=50, r=15, seed=1)
    K_hom = hadamard_kernel({"A": K_A, "C": K_C})
    # symmetric
    assert np.allclose(K_hom, K_hom.T, atol=1e-10)
    # PSD up to float tolerance
    eigs = np.linalg.eigvalsh(K_hom)
    assert eigs.min() > -1e-6, f"Hadamard PSD violated, min eig={eigs.min()}"


def test_sum_kernel_weighted():
    """Weighted sum: 0.25*K_A + 2.0*K_C."""
    K_A = np.array([[1.0, 0.2], [0.2, 1.0]])
    K_C = np.array([[2.0, 0.5], [0.5, 2.0]])
    K = sum_kernel({"A": K_A, "C": K_C}, weights={"A": 0.25, "C": 2.0})
    np.testing.assert_allclose(K, 0.25 * K_A + 2.0 * K_C)
    # Default weights = 1.0
    K_unw = sum_kernel({"A": K_A, "C": K_C})
    np.testing.assert_allclose(K_unw, K_A + K_C)
    # Missing weight key defaults to 1.0
    K_part = sum_kernel({"A": K_A, "C": K_C}, weights={"A": 0.5})
    np.testing.assert_allclose(K_part, 0.5 * K_A + 1.0 * K_C)


def test_normalize_kernel_trace_sets_trace_to_n():
    """After mode='trace', trace(K') == n."""
    n = 30
    K = _toy_psd(n=n, r=10, seed=42)
    K_norm = normalize_kernel(K, mode="trace")
    assert K_norm.shape == (n, n)
    assert np.trace(K_norm) == pytest.approx(n, rel=1e-12, abs=1e-9)
    # Frobenius mode
    K_fnorm = normalize_kernel(K, mode="frobenius")
    assert np.linalg.norm(K_fnorm, "fro") == pytest.approx(n, rel=1e-12, abs=1e-9)


# ---- Negative tests: locked-down error semantics ----

def test_kernel_input_validation_empty_dict():
    with pytest.raises(ValueError, match="empty"):
        hadamard_kernel({})
    with pytest.raises(ValueError, match="empty"):
        sum_kernel({})


def test_kernel_input_validation_shape_mismatch():
    K_A = np.eye(3)
    K_C = np.eye(4)
    with pytest.raises(ValueError, match="shape"):
        hadamard_kernel({"A": K_A, "C": K_C})


def test_kernel_input_validation_nonfinite():
    K_bad = np.array([[1.0, np.nan], [0.0, 1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        hadamard_kernel({"A": K_bad})


def test_normalize_kernel_zero_trace_raises():
    K_zero = np.zeros((5, 5))
    with pytest.raises(ValueError, match="denominator"):
        normalize_kernel(K_zero, mode="trace")


def test_normalize_kernel_invalid_mode():
    K = np.eye(3)
    with pytest.raises(ValueError, match="invalid mode"):
        normalize_kernel(K, mode="bogus")  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
