"""Composite kernels for the HomoeoGWAS multi-subgenome LMM.

Two constructions over per-subgenome GRMs:

- ``sum_kernel``:      K_sum = Σ_s w_s · G_s
- ``hadamard_kernel``: K_hom = G_{s1} ⊙ G_{s2} ⊙ … (elementwise)

K_hom encodes homoeolog co-association: two samples score high only if
jointly related in all subgenomes. K_sum is the conventional additive null
(subgenomes contribute independently). By the Schur product theorem the
elementwise product of PSD matrices is PSD, so K_hom stays a valid kernel.

``normalize_kernel`` rescales a kernel to trace == n (or Frobenius == n) so
variance-component magnitudes are comparable across kernels in REML.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


def _validate_kernel_dict(grms: dict[str, np.ndarray]) -> tuple[int, list[str]]:
    """Sanity-check a {subgenome -> matrix} dict; return (n, ordered_keys)."""
    if not grms:
        raise ValueError("kernel input dict is empty")
    keys = list(grms.keys())
    shape0 = None
    for k in keys:
        M = grms[k]
        if not isinstance(M, np.ndarray):
            raise TypeError(f"grms[{k!r}] is {type(M).__name__}, expected np.ndarray")
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"grms[{k!r}] not square 2D: shape={M.shape}")
        if shape0 is None:
            shape0 = M.shape
        elif M.shape != shape0:
            raise ValueError(
                f"grms[{k!r}] shape {M.shape} differs from first key shape {shape0}"
            )
        if not np.all(np.isfinite(M)):
            raise ValueError(f"grms[{k!r}] contains non-finite values")
    return shape0[0], keys


def hadamard_kernel(grms: dict[str, np.ndarray]) -> np.ndarray:
    """Elementwise (Hadamard) product of per-subgenome GRMs.

    K_hom[i,j] = ∏_s G_s[i,j]

    Args:
        grms: {subgenome -> GRM (n,n)}. ≥1 subgenome required; all matrices
            must be 2D square float, same n, finite. Inputs are not mutated.

    Returns:
        K_hom: float64 (n,n). PSD if all inputs are PSD (Schur product theorem).
    """
    n, keys = _validate_kernel_dict(grms)
    K = np.asarray(grms[keys[0]], dtype=np.float64).copy()
    for k in keys[1:]:
        K *= np.asarray(grms[k], dtype=np.float64)
    return K


def pairwise_mean_kernel(grms: dict[str, np.ndarray]) -> np.ndarray:
    """Mean of pairwise Hadamard products: K_pair = (1/C(n,2)) Σ_{i<j} G_i ⊙ G_j.

    For ≥4 subgenomes the full Hadamard product goes numerically sparse
    (off-diagonals decay multiplicatively) and fits poorly in REML. The
    pairwise mean keeps a usable dynamic range, preserves the cross-subgenome
    epistasis signal, and stays PSD (each pair-product PSD by Schur, sum PSD).

    Args:
        grms: {subgenome -> GRM (n,n)}, ≥2 subgenomes; same shape / finiteness
            constraints as ``hadamard_kernel``.

    Returns:
        K_pair: float64 (n,n), PSD.
    """
    n, keys = _validate_kernel_dict(grms)
    n_keys = len(keys)
    if n_keys < 2:
        raise ValueError(
            f"pairwise_mean_kernel requires ≥2 subgenomes, got {n_keys}")
    K = np.zeros((n, n), dtype=np.float64)
    n_pairs = 0
    for i in range(n_keys):
        Gi = np.asarray(grms[keys[i]], dtype=np.float64)
        for j in range(i + 1, n_keys):
            Gj = np.asarray(grms[keys[j]], dtype=np.float64)
            K += Gi * Gj
            n_pairs += 1
    K *= (1.0 / n_pairs)
    return K


def build_homoeolog_kernel(
    grms: dict[str, np.ndarray],
    *,
    mode: Literal["auto", "hadamard", "pairwise_mean", "none"] = "auto",
    auto_threshold_n: int = 3,
) -> tuple[np.ndarray | None, str]:
    """Construct the homoeolog co-association kernel from per-subgenome GRMs.

    Returns ``(K_hom, mode_used)``. When ``mode_used == "none"`` (single
    subgenome or ``mode="none"``) the kernel is ``None`` and the caller should
    drop the homoeolog random effect; the additive-only LMM is then the null.

    Auto decision:
      - 1 subgenome → ``(None, "none")`` (nothing to multiply)
      - 2..``auto_threshold_n`` (default 3) → full Hadamard
      - >``auto_threshold_n`` → pairwise-mean (≥4-way Hadamard too sparse)

    Explicit modes bypass the decision so Methods can report a single
    deterministic mode while auto stays numerically safe for new species.
    """
    if mode == "none":
        return None, "none"
    n_sub = len(grms)
    if n_sub <= 1:
        return None, "none"
    if mode == "hadamard":
        return hadamard_kernel(grms), "hadamard"
    if mode == "pairwise_mean":
        return pairwise_mean_kernel(grms), "pairwise_mean"
    if n_sub <= auto_threshold_n:
        return hadamard_kernel(grms), "hadamard"
    return pairwise_mean_kernel(grms), "pairwise_mean"


def sum_kernel(
    grms: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Weighted sum of per-subgenome GRMs.

    K_sum = Σ_s w_s · G_s,  default w_s = 1.0 for all s.

    Args:
        grms: {subgenome -> GRM (n,n)}, same constraints as hadamard_kernel.
        weights: optional {subgenome -> float}; missing keys default to 1.0.

    Returns:
        K_sum: float64 (n,n).
    """
    n, keys = _validate_kernel_dict(grms)
    K = np.zeros((n, n), dtype=np.float64)
    for k in keys:
        w = 1.0 if weights is None else float(weights.get(k, 1.0))
        K += w * np.asarray(grms[k], dtype=np.float64)
    return K


def normalize_kernel(
    K: np.ndarray,
    mode: Literal["trace", "frobenius"] = "trace",
    tol: float = 1e-12,
) -> np.ndarray:
    """Scale a kernel for comparable magnitude across LMM variance components.

    - mode="trace":     K' = K · n / trace(K)         => trace(K') == n
    - mode="frobenius": K' = K · n / ||K||_F          => ||K'||_F == n

    Args:
        K: (n,n) finite 2D square.
        mode: scaling mode.
        tol: minimum |denominator|; below raises ValueError to avoid div-by-zero
            and silent inflation of numerical noise.

    Returns:
        K': new float64 array; input not mutated.
    """
    if not isinstance(K, np.ndarray):
        raise TypeError(f"K is {type(K).__name__}, expected np.ndarray")
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] == 0:
        raise ValueError(f"K must be non-empty square 2D, got shape={K.shape}")
    if not np.all(np.isfinite(K)):
        raise ValueError("K contains non-finite values")

    n = K.shape[0]
    Kf = K.astype(np.float64, copy=True)
    if mode == "trace":
        denom = float(np.trace(Kf))
    elif mode == "frobenius":
        denom = float(np.linalg.norm(Kf, "fro"))
    else:
        raise ValueError(f"invalid mode={mode!r}, expected 'trace' or 'frobenius'")

    if not np.isfinite(denom) or abs(denom) < tol:
        raise ValueError(
            f"normalization denominator (mode={mode}) too small or non-finite: "
            f"denom={denom}, tol={tol}"
        )
    return Kf * (n / denom)
