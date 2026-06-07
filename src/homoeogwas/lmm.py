"""Linear mixed model with single random-effect kernel (HomoeoGWAS M2.3).

Model
-----
    y  =  X β  +  u  +  e
    u  ~  N(0, σ_g²  K)
    e  ~  N(0, σ_e²  I)

Parameterise in δ = σ_e² / σ_g². Eigendecompose K = U Λ Uᵀ once;
profile out β and σ_g² so that REML log-likelihood depends on δ only:

    L_REML(δ)  =  −½ [ (n−p) log(σ_g²(δ))
                       + Σ log(λ_i + δ)
                       + log |Xᵀ (K+δI)⁻¹ X|  ]                (+ const)

Minimise neg-log-REML over log_delta ∈ log_delta_bounds with
``scipy.optimize.minimize_scalar`` (bounded Brent).

Backends
--------
- ``cpu`` — NumPy / SciPy (default).
- ``gpu`` — CuPy if importable; raises if requested but unavailable.
- ``auto`` — try CuPy, fall back to CPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


@dataclass
class REMLResult:
    """Profiled-REML fit result for a single-kernel LMM."""
    sigma_g2: float
    sigma_e2: float
    h2: float
    beta: np.ndarray         # (p,)
    log_lik: float           # REML log-lik at optimum (up to constants)
    log_delta: float         # log(σ_e²/σ_g²)
    n: int
    p: int
    backend_used: str
    min_eig: float
    n_eig_clipped: int       # number of eigenvalues clipped to 0 from (−eig_tol, 0)
    optimizer_status: bool
    log_delta_bounds: tuple[float, float]
    boundary_hit: str | None = None     # "lower" / "upper" / None, set when |log_δ−bound|<1e-3

    def to_dict(self) -> dict:
        d = asdict(self)
        d["beta"] = self.beta.tolist()
        return d


def _validate_inputs(y: np.ndarray, X: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError(f"y must be 1D ndarray, got shape={getattr(y,'shape',None)}")
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(f"X must be 2D ndarray, got shape={getattr(X,'shape',None)}")
    if not isinstance(K, np.ndarray) or K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square 2D ndarray, got shape={getattr(K,'shape',None)}")
    n = K.shape[0]
    if y.shape[0] != n or X.shape[0] != n:
        raise ValueError(
            f"sample dimensions mismatch: y={y.shape}, X={X.shape}, K={K.shape}"
        )
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(X)) and np.all(np.isfinite(K))):
        raise ValueError("y / X / K must be all finite")
    p = X.shape[1]
    if p >= n:
        raise ValueError(f"design rank issue: p={p} >= n={n}")
    return n, p


def _eigendecomp_cpu(K_sym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy symmetric eigendecomposition. Returns (lam, U) ascending."""
    return np.linalg.eigh(K_sym)


def _eigendecomp_gpu(K_sym: np.ndarray):
    """CuPy symmetric eigendecomposition; returns numpy arrays."""
    import cupy as cp  # type: ignore
    K_gpu = cp.asarray(K_sym)
    lam_g, U_g = cp.linalg.eigh(K_gpu)
    return cp.asnumpy(lam_g), cp.asnumpy(U_g)


def _resolve_backend(backend: str) -> tuple[str, callable]:
    """Pick eigendecomp function. Returns (backend_used, fn)."""
    if backend == "cpu":
        return "cpu", _eigendecomp_cpu
    if backend == "gpu":
        try:
            import cupy  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"backend='gpu' but CuPy not importable: {e}"
            ) from None
        return "gpu", _eigendecomp_gpu
    if backend == "auto":
        try:
            import cupy  # noqa: F401
            return "gpu", _eigendecomp_gpu
        except ImportError:
            return "cpu", _eigendecomp_cpu
    raise ValueError(f"invalid backend={backend!r}; expected 'cpu' / 'gpu' / 'auto'")


def _neg_log_reml(
    log_delta: float,
    lam: np.ndarray,
    y_t: np.ndarray,
    X_t: np.ndarray,
    n: int,
    p: int,
) -> float:
    """Profiled negative REML log-likelihood at log_delta."""
    delta = float(np.exp(log_delta))
    d = lam + delta                          # (n,)
    if np.any(d <= 0):
        return np.inf
    inv_d = 1.0 / d
    # X_t^T D^{-1} X_t   (p, p)
    XtDinvX = (X_t.T * inv_d) @ X_t
    XtDinvy = (X_t.T * inv_d) @ y_t          # (p,)
    try:
        L = np.linalg.cholesky(XtDinvX)
    except np.linalg.LinAlgError:
        return np.inf
    # β̂ = (X^T D^{-1} X)^{-1} X^T D^{-1} y
    z = np.linalg.solve(L, XtDinvy)
    beta = np.linalg.solve(L.T, z)
    r = y_t - X_t @ beta
    # σ_g²(δ) profile = (1/(n−p)) Σ r²/d
    rss = float(np.sum(r * r * inv_d))
    if rss <= 0:
        return np.inf
    sigma_g2 = rss / (n - p)
    log_det_K_plus_dI = float(np.sum(np.log(d)))
    log_det_XtDinvX = 2.0 * float(np.sum(np.log(np.diag(L))))
    # Up to additive constants in δ:
    ll = -0.5 * (
        (n - p) * np.log(sigma_g2)
        + log_det_K_plus_dI
        + log_det_XtDinvX
    )
    return float(-ll)


def fit_reml(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    backend: Literal["cpu", "gpu", "auto"] = "auto",
    eig_tol: float = 1e-8,
    log_delta_bounds: tuple[float, float] = (-10.0, 10.0),
) -> REMLResult:
    """Fit single-kernel LMM via profiled REML.

    Args:
        y: phenotype (n,) float.
        X: fixed-effect design (n, p), typically [1] for intercept-only.
        K: kernel (n, n), symmetric, PSD up to ``eig_tol``.
        backend: 'cpu' / 'gpu' / 'auto'.
        eig_tol: eigenvalues in [−eig_tol, 0) clipped to 0; lower (more negative) raises.
        log_delta_bounds: search range for log(σ_e²/σ_g²); default (−10, 10) covers
            δ ∈ [4.5e−5, 22026] which spans h² ∈ (~0, ~1).

    Returns:
        REMLResult.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    K = np.ascontiguousarray(K, dtype=np.float64)
    n, p = _validate_inputs(y, X, K)

    # Symmetrize then eigendecompose
    K_sym = 0.5 * (K + K.T)
    backend_used, eig_fn = _resolve_backend(backend)
    lam, U = eig_fn(K_sym)
    min_eig = float(lam.min())
    # PSD enforcement
    n_clipped = int(((lam < 0) & (lam >= -eig_tol)).sum())
    if (lam < -eig_tol).any():
        raise ValueError(
            f"K has eigenvalues < -eig_tol ({eig_tol}); "
            f"min eig = {min_eig:.6g}, n_below_tol={int((lam < -eig_tol).sum())}. "
            "K may not be PSD; symmetrize or use a better-conditioned kernel."
        )
    lam = np.clip(lam, 0.0, None)

    # Project to eigenbasis
    y_t = U.T @ y
    X_t = U.T @ X

    # Bounded profile minimisation
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(
        _neg_log_reml,
        args=(lam, y_t, X_t, n, p),
        bounds=log_delta_bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )

    # Recompute components at optimum
    log_delta_hat = float(res.x)
    delta_hat = float(np.exp(log_delta_hat))
    d = lam + delta_hat
    inv_d = 1.0 / d
    XtDinvX = (X_t.T * inv_d) @ X_t
    XtDinvy = (X_t.T * inv_d) @ y_t
    L = np.linalg.cholesky(XtDinvX)
    z = np.linalg.solve(L, XtDinvy)
    beta = np.linalg.solve(L.T, z)
    r = y_t - X_t @ beta
    rss = float(np.sum(r * r * inv_d))
    sigma_g2 = rss / (n - p)
    sigma_e2 = delta_hat * sigma_g2
    h2 = sigma_g2 / (sigma_g2 + sigma_e2)
    log_lik = float(-res.fun)

    lo, hi = log_delta_bounds
    boundary: str | None = None
    if abs(log_delta_hat - lo) < 1e-3:
        boundary = "lower"
    elif abs(log_delta_hat - hi) < 1e-3:
        boundary = "upper"

    return REMLResult(
        sigma_g2=float(sigma_g2),
        sigma_e2=float(sigma_e2),
        h2=float(h2),
        beta=beta,
        log_lik=log_lik,
        log_delta=log_delta_hat,
        n=n, p=p,
        backend_used=backend_used,
        min_eig=min_eig,
        n_eig_clipped=n_clipped,
        optimizer_status=bool(res.success),
        log_delta_bounds=tuple(log_delta_bounds),
        boundary_hit=boundary,
    )


# Multi-kernel REML (J >= 1 random-effect kernels)
#
#     y = X β + Σ_j u_j + e,    u_j ~ N(0, σ²_j K_j),    e ~ N(0, σ²_e I)
#
# Box-constrained L-BFGS-B over raw σ² (no exp(θ) reparam, because the σ² = 0
# boundary must be reachable for the Davies / Stram-Lee 50:50 chi² mixture LRT).
# n is small (a few hundred to ~1000), so the O(n³) cho_factor/cho_solve per
# likelihood eval is cheap. Three diagnostics distinguish "no signal in
# subgenome j" from "K_j is collinear with the others": PVE_j boundary flag,
# pairwise vec(K_j) correlations, and the design condition number.


@dataclass
class MultiREMLResult:
    """Multi-kernel REML fit result."""
    sigma2: dict[str, float]           # {kernel_name: σ², …, "e": σ²_e}
    pve: dict[str, float]              # paper-grade PVE = σ²_j·trace(K_j)/n  Σ
    component_var: dict[str, float]    # σ²_j·trace(K_j)/n per genetic + σ²_e
    beta: np.ndarray                   # (p,)
    log_lik: float                     # REML log-lik (no additive const)
    n: int
    p: int
    kernel_names: list[str]            # ordered (genetic only)
    optimizer_status: bool
    optimizer_message: str
    n_iter: int
    boundary_components: list[str]     # names of components hitting σ² → boundary (may include "e")
    kernel_corr: dict[str, dict[str, float]]  # pairwise Pearson, off-diag
    kernel_design_cond: float          # cond(N²×(J+1) matrix of vec(K_j))
    kernel_min_eig: dict[str, float]   # min eigval of each symmetrized K
    kernel_n_eig_clipped: dict[str, int]   # # eigs in (-eig_tol, 0) clipped to 0
    n_starts: int                      # number of optimizer starts attempted
    best_start: int                    # which start (0-indexed) gave the best log_lik
    all_start_log_lik: list[float]     # log_lik per start
    backend_used: str                  # "cpu"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["beta"] = self.beta.tolist()
        return d


def _build_V(sigma2: np.ndarray, K_list: list[np.ndarray], n: int) -> np.ndarray:
    """V = Σ σ²_j K_j + σ²_e I."""
    V = sigma2[-1] * np.eye(n, dtype=np.float64)
    for j, K in enumerate(K_list):
        V = V + sigma2[j] * K
    return V


def _neg_log_reml_multi(
    sigma2: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    K_list: list[np.ndarray],
    n: int,
    p: int,
    jitter_var: float,
) -> float:
    """Direct REML neg log-likelihood:
       L = -0.5 (log|V| + log|X'V⁻¹X| + (y-Xβ̂)' V⁻¹ (y-Xβ̂))   (+ const)
    """
    from scipy.linalg import cho_factor, cho_solve

    V = _build_V(sigma2, K_list, n)
    try:
        c, lower = cho_factor(V, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        try:
            c, lower = cho_factor(V + jitter_var * np.eye(n), lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf

    log_det_V = 2.0 * float(np.sum(np.log(np.diag(c))))
    if not np.isfinite(log_det_V):
        return np.inf

    Vinv_X = cho_solve((c, lower), X, check_finite=False)
    XtVinvX = X.T @ Vinv_X
    try:
        c2, lower2 = cho_factor(XtVinvX, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf
    log_det_XtVinvX = 2.0 * float(np.sum(np.log(np.diag(c2))))

    Vinv_y = cho_solve((c, lower), y, check_finite=False)
    XtVinvy = X.T @ Vinv_y
    beta = cho_solve((c2, lower2), XtVinvy, check_finite=False)
    r = y - X @ beta
    Vinv_r = cho_solve((c, lower), r, check_finite=False)
    quad = float(r @ Vinv_r)
    if not np.isfinite(quad):
        return np.inf

    ll = -0.5 * (log_det_V + log_det_XtVinvX + quad)
    return float(-ll)


def _compute_kernel_diagnostics(
    kernels: dict[str, np.ndarray],
    n: int,
) -> tuple[dict[str, dict[str, float]], float]:
    """Pairwise Pearson corr of vec(K_j) (upper triangle, no diag) +
    condition number of design matrix [vec(K_1), …, vec(K_J), vec(I)]."""
    names = list(kernels.keys())
    iu = np.triu_indices(n, k=1)
    vecs = {}
    for name in names:
        vecs[name] = kernels[name][iu].astype(np.float64)
    # pairwise correlations
    corr: dict[str, dict[str, float]] = {name: {} for name in names}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                corr[ni][nj] = 1.0
            else:
                c = float(np.corrcoef(vecs[ni], vecs[nj])[0, 1])
                corr[ni][nj] = c
    # design matrix cond: stack flattened K_j + vec(I) (all, including diag)
    design_cols = [kernels[name].reshape(-1) for name in names]
    design_cols.append(np.eye(n, dtype=np.float64).reshape(-1))
    design = np.column_stack(design_cols)
    cond = float(np.linalg.cond(design))
    return corr, cond


def fit_multi_reml(
    y: np.ndarray,
    X: np.ndarray,
    kernels: dict[str, np.ndarray],
    *,
    init: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    boundary_eps: float = 1e-3,
    eig_tol: float = 1e-8,
    jitter: float = 1e-8,
    maxiter: int = 500,
    ftol: float = 1e-8,
    gtol: float = 1e-6,
    n_starts: int = 1,
    random_state: int | None = None,
    start_scale: tuple[float, float] = (1e-3, 1e1),
) -> MultiREMLResult:
    """Fit y = X β + Σ_j u_j + e with multiple GRM-like kernels via L-BFGS-B
    over raw σ² with box bounds.

    Args:
        y, X, kernels, init, bounds: as in fit_reml / above.
        boundary_eps: a component is flagged in ``boundary_components`` when
            its PVE < boundary_eps (PVE = σ²·trace(K)/n, normalized); for "e"
            also when σ²_e ≤ lower_bound·(1+1e-6). Scale-invariant under K↦cK.
        eig_tol: each kernel must have min eigval ≥ −eig_tol after symmetrize;
            eigs in (−eig_tol, 0) are clipped to 0, below raises.
        jitter: V + jitter·var(y)·I on Cholesky retry.
        n_starts: number of optimizer starts (≥ 1); start 0 is deterministic,
            the rest draw σ²_j log-uniformly over var(y)·start_scale, clipped
            into bounds.
        random_state: seed for multi-start sampling.
        start_scale: (lo, hi) bounds for the log-uniform draw, scaled by var(y).

    Returns:
        MultiREMLResult, with PVE = σ²_j·trace(K_j)/n normalized.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}")
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X must be (n,p) matching y; got X={X.shape} y={y.shape}")
    n = y.shape[0]
    p = X.shape[1]
    if p >= n:
        raise ValueError(f"design rank issue: p={p} >= n={n}")

    if not kernels:
        raise ValueError("kernels dict is empty")
    kernel_names = list(kernels.keys())
    J = len(kernel_names)
    K_list: list[np.ndarray] = []
    kernel_min_eig: dict[str, float] = {}
    kernel_n_eig_clipped: dict[str, int] = {}
    for name in kernel_names:
        K = kernels[name]
        if not isinstance(K, np.ndarray) or K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] != n:
            raise ValueError(
                f"kernels[{name!r}] must be square (n,n) with n={n}; got {getattr(K, 'shape', None)}"
            )
        if not np.all(np.isfinite(K)):
            raise ValueError(f"kernels[{name!r}] contains non-finite values")
        K_sym = 0.5 * (K + K.T).astype(np.float64)
        eigs = np.linalg.eigvalsh(K_sym)
        min_eig = float(eigs.min())
        kernel_min_eig[name] = min_eig
        n_clipped = int(((eigs < 0) & (eigs >= -eig_tol)).sum())
        kernel_n_eig_clipped[name] = n_clipped
        if (eigs < -eig_tol).any():
            raise ValueError(
                f"kernels[{name!r}] not PSD: min_eig={min_eig:.6g} < -eig_tol={eig_tol}; "
                f"n_below_tol={int((eigs < -eig_tol).sum())}"
            )
        if n_clipped > 0:
            # rebuild K from the clipped eigenvalues
            evals, evecs = np.linalg.eigh(K_sym)
            evals = np.clip(evals, 0.0, None)
            K_sym = (evecs * evals) @ evecs.T
        K_list.append(K_sym)

    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(X))):
        raise ValueError("y / X must be all finite")

    if n_starts < 1:
        raise ValueError(f"n_starts must be ≥ 1, got {n_starts}")
    if not (start_scale[0] > 0 and start_scale[1] > 0 and start_scale[0] < start_scale[1]):
        raise ValueError(f"start_scale must be (lo, hi) with 0 < lo < hi, got {start_scale}")

    var_y = float(np.var(y, ddof=1))
    if var_y <= 0:
        raise ValueError(f"var(y)={var_y} non-positive")
    jitter_var = jitter * var_y

    def _default_init() -> np.ndarray:
        a = np.empty(J + 1, dtype=np.float64)
        a[:J] = 0.3 * var_y / J
        a[J] = 0.7 * var_y
        return a

    init_arr_default = _default_init()
    if init is not None:
        for j, name in enumerate(kernel_names):
            init_arr_default[j] = float(init.get(name, 0.3 * var_y / J))
        init_arr_default[J] = float(init.get("e", 0.7 * var_y))

    if bounds is None:
        bounds_list = [(0.0, var_y * 1e4)] * J + [(var_y * 1e-8, var_y * 1e4)]
    else:
        bounds_list = []
        for name in kernel_names:
            lo, hi = bounds.get(name, (0.0, var_y * 1e4))
            bounds_list.append((float(lo), float(hi)))
        lo_e, hi_e = bounds.get("e", (var_y * 1e-8, var_y * 1e4))
        bounds_list.append((float(lo_e), float(hi_e)))

    rng = np.random.default_rng(random_state)
    starts = [init_arr_default]
    for _ in range(max(0, n_starts - 1)):
        u = rng.uniform(np.log(start_scale[0]), np.log(start_scale[1]), size=J + 1)
        scaled = np.exp(u) * var_y
        for k in range(J + 1):
            lo_k, hi_k = bounds_list[k]
            scaled[k] = float(np.clip(scaled[k], lo_k, hi_k))
        starts.append(scaled)

    from scipy.optimize import minimize
    all_start_log_lik: list[float] = []
    best_res = None
    best_log_lik = -np.inf
    best_start_idx = 0
    for s_idx, init_arr in enumerate(starts):
        res = minimize(
            _neg_log_reml_multi,
            init_arr,
            args=(y, X, K_list, n, p, jitter_var),
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
        )
        ll = float(-res.fun) if np.isfinite(res.fun) else -np.inf
        all_start_log_lik.append(ll)
        if ll > best_log_lik:
            best_log_lik = ll
            best_res = res
            best_start_idx = s_idx
    assert best_res is not None
    sigma2_hat = best_res.x.astype(np.float64)
    log_lik = float(-best_res.fun)

    # final β at the best-fit V
    V = _build_V(sigma2_hat, K_list, n)
    from scipy.linalg import cho_factor, cho_solve
    try:
        c, lo = cho_factor(V, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        c, lo = cho_factor(V + jitter_var * np.eye(n), lower=True, check_finite=False)
    Vinv_X = cho_solve((c, lo), X, check_finite=False)
    XtVinvX = X.T @ Vinv_X
    Vinv_y = cho_solve((c, lo), y, check_finite=False)
    beta = np.linalg.solve(XtVinvX, X.T @ Vinv_y)

    # component-wise PVE = σ²·trace(K)/n, normalized
    sigma2_dict = {kernel_names[j]: float(sigma2_hat[j]) for j in range(J)}
    sigma2_dict["e"] = float(sigma2_hat[J])
    component_var = {kernel_names[j]: float(sigma2_hat[j] * np.trace(K_list[j]) / n)
                     for j in range(J)}
    component_var["e"] = float(sigma2_hat[J])  # I has trace n; σ²_e·n/n = σ²_e
    total_var = sum(component_var.values())
    pve = {k: (v / total_var if total_var > 0 else 0.0) for k, v in component_var.items()}

    # boundary detection: scale-invariant PVE threshold + lower-bound hit
    boundary_components: list[str] = []
    for k_name in sigma2_dict.keys():
        pve_k = pve[k_name]
        if k_name == "e":
            lo_e = bounds_list[-1][0]
            if sigma2_dict[k_name] <= lo_e * (1 + 1e-6) or pve_k < boundary_eps:
                boundary_components.append(k_name)
        else:
            if pve_k < boundary_eps:
                boundary_components.append(k_name)

    kernel_corr, kernel_design_cond = _compute_kernel_diagnostics(
        {name: K for name, K in zip(kernel_names, K_list, strict=True)}, n
    )

    return MultiREMLResult(
        sigma2=sigma2_dict,
        pve=pve,
        component_var=component_var,
        beta=beta,
        log_lik=log_lik,
        n=n, p=p,
        kernel_names=kernel_names,
        optimizer_status=bool(best_res.success),
        optimizer_message=str(best_res.message),
        n_iter=int(best_res.nit),
        boundary_components=boundary_components,
        kernel_corr=kernel_corr,
        kernel_design_cond=kernel_design_cond,
        kernel_min_eig=kernel_min_eig,
        kernel_n_eig_clipped=kernel_n_eig_clipped,
        n_starts=len(starts),
        best_start=best_start_idx,
        all_start_log_lik=all_start_log_lik,
        backend_used="cpu",
    )


# Auto-fallback wrapper around fit_multi_reml.
#
# The Hadamard K_hom often gives no detectable variance component (PVE_hom ≈ 0,
# σ²_hom at the lower bound), e.g. with modest sample size, ≥4 subgenomes (the
# tensor goes sparse), or an ill-conditioned [vec(K_j), vec(I)] design. When
# that happens the additive-only model is the right headline.
#
# This wrapper builds K_hom via kernel.build_homoeolog_kernel, fits the full
# model, and refits without K_hom when σ²_hom hits its lower bound,
# kernel_design_cond > cond_threshold, or PVE_hom < pve_floor. The returned
# result keeps both fits plus a reason string.


@dataclass
class MultiREMLAutoResult:
    """Wrapper around two MultiREMLResult: full (with K_hom) and additive
    (without). ``preferred`` points at the one the framework recommends
    using downstream (typically additive when the fallback triggered)."""

    full: MultiREMLResult | None        # None if hom kernel was 'none' (n_sub<=1)
    additive: MultiREMLResult
    homoeolog_mode: str                    # 'hadamard' / 'pairwise_mean' / 'none'
    homoeolog_dropped: bool                # True → fallback to additive
    drop_reason: str                       # 'no_hom_kernel' / 'boundary' / 'cond' / 'pve_floor' / 'kept'
    cond_threshold: float
    pve_floor: float
    preferred: str                         # 'full' or 'additive'

    def to_dict(self) -> dict:
        return {
            "full": self.full.to_dict() if self.full else None,
            "additive": self.additive.to_dict(),
            "homoeolog_mode": self.homoeolog_mode,
            "homoeolog_dropped": self.homoeolog_dropped,
            "drop_reason": self.drop_reason,
            "cond_threshold": self.cond_threshold,
            "pve_floor": self.pve_floor,
            "preferred": self.preferred,
        }


def fit_with_homoeolog_fallback(
    y: np.ndarray,
    X: np.ndarray,
    per_subgenome_grms: dict[str, np.ndarray],
    *,
    hom_mode: str = "auto",
    cond_threshold: float = 1e4,
    pve_floor: float = 1e-3,
    auto_threshold_n: int = 3,
    hom_kernel_name: str = "K_hom",
    **fit_kwargs,
) -> MultiREMLAutoResult:
    """Fit subgenome-stratified LMM with automatic Hadamard fallback.

    Builds K_hom from per-subgenome GRMs (Hadamard for ≤``auto_threshold_n``
    subgenomes, pairwise-mean otherwise) and fits the full multi-kernel
    REML. If K_hom turns out to be ill-conditioned, redundant, or
    contributes no variance, refits without it and returns the additive
    model as the preferred result.

    The additive fit always runs, so the caller has a baseline regardless
    of fallback outcome.

    Parameters
    ----------
    y, X : phenotype + fixed-effect design
    per_subgenome_grms : ``{subgenome_id -> GRM (n,n)}``
    hom_mode : ``auto`` (default) / ``hadamard`` / ``pairwise_mean`` / ``none``
    cond_threshold : fallback if kernel_design_cond exceeds this (default 1e4)
    pve_floor : fallback if PVE_hom < this (default 1e-3)
    auto_threshold_n : ``≤n`` subgenomes → Hadamard; ``>n`` → pairwise-mean
    hom_kernel_name : name to register K_hom under in the kernels dict
    fit_kwargs : forwarded to ``fit_multi_reml`` (n_starts, bounds, …)

    Returns
    -------
    MultiREMLAutoResult
    """
    from .kernel import build_homoeolog_kernel

    K_hom, hom_mode_used = build_homoeolog_kernel(
        per_subgenome_grms, mode=hom_mode, auto_threshold_n=auto_threshold_n,
    )

    additive_kernels = dict(per_subgenome_grms)
    additive_result = fit_multi_reml(y, X, additive_kernels, **fit_kwargs)

    if K_hom is None:
        return MultiREMLAutoResult(
            full=None,
            additive=additive_result,
            homoeolog_mode=hom_mode_used,
            homoeolog_dropped=True,
            drop_reason="no_hom_kernel",
            cond_threshold=cond_threshold,
            pve_floor=pve_floor,
            preferred="additive",
        )

    full_kernels = dict(per_subgenome_grms)
    full_kernels[hom_kernel_name] = K_hom
    full_result = fit_multi_reml(y, X, full_kernels, **fit_kwargs)

    pve_hom = full_result.pve.get(hom_kernel_name, 0.0)
    drop_reason = "kept"
    if hom_kernel_name in full_result.boundary_components:
        drop_reason = "boundary"
    elif full_result.kernel_design_cond > cond_threshold:
        drop_reason = "cond"
    elif pve_hom < pve_floor:
        drop_reason = "pve_floor"

    dropped = drop_reason != "kept"
    preferred = "additive" if dropped else "full"
    return MultiREMLAutoResult(
        full=full_result,
        additive=additive_result,
        homoeolog_mode=hom_mode_used,
        homoeolog_dropped=dropped,
        drop_reason=drop_reason,
        cond_threshold=cond_threshold,
        pve_floor=pve_floor,
        preferred=preferred,
    )
