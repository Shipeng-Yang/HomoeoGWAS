"""Diagnostics for multi-kernel REML (Phase 2 M2.4.2).

Public functions:
- ``compare_nested_reml(y, X, kernels, model_specs)``: fit nested models with
  the *same* direct REML objective so log-likelihoods are LRT-comparable.
- ``residual_only_reml(y, X)``: closed-form REML for σ²_e-only null model.

Result types:
- ``NestedREMLComparison``: container with per-model fits + likelihood table.
- ``ResidualOnlyResult``: lightweight result for J=0 model.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field, replace
from itertools import combinations, product
from typing import Literal

import numpy as np
import pandas as pd

from .kernel import normalize_kernel
from .lmm import MultiREMLResult, fit_multi_reml


@dataclass
class ResidualOnlyResult:
    """REML fit of y = X β + e (no random-effect kernel)."""
    sigma2: dict[str, float]            # {"e": σ²_e}
    pve: dict[str, float]               # {"e": 1.0}
    component_var: dict[str, float]
    beta: np.ndarray
    log_lik: float
    n: int
    p: int
    optimizer_status: bool = True
    optimizer_message: str = "closed-form REML (no random effects)"
    n_iter: int = 0
    boundary_components: list[str] = field(default_factory=list)
    kernel_names: list[str] = field(default_factory=list)
    backend_used: str = "cpu"


def residual_only_reml(y: np.ndarray, X: np.ndarray) -> ResidualOnlyResult:
    """Closed-form REML for y = X β + e, e ~ N(0, σ²_e I).

    REML log-likelihood (direct, same constants as fit_multi_reml):
        L = -0.5 [ log|V| + log|X'V⁻¹X| + (y-Xβ̂)' V⁻¹ (y-Xβ̂) ]
    With V = σ²_e I, this is
        L = -0.5 [ n log σ²_e + log|X'X / σ²_e| + RSS / σ²_e ]
          = -0.5 [ (n-p) log σ²_e + log|X'X| + RSS / σ²_e ]
    REML estimator: σ²_e_hat = RSS / (n-p).
    At optimum L = -0.5 [(n-p) log σ²_e_hat + log|X'X| + (n-p)].
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}")
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X must be (n,p) matching y; got X={X.shape}, y={y.shape}")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(X))):
        raise ValueError("y / X must be all finite")
    n = y.shape[0]
    p = X.shape[1]
    if p >= n:
        raise ValueError(f"design rank issue: p={p} >= n={n}")
    XtX = X.T @ X
    sign, logabs = np.linalg.slogdet(XtX)
    if sign <= 0 or not np.isfinite(logabs):
        raise ValueError(f"X'X is singular or non-positive (slogdet sign={sign}); X may be rank-deficient")
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    r = y - X @ beta
    rss = float(r @ r)
    sigma2_e_hat = rss / (n - p)
    log_lik = -0.5 * ((n - p) * np.log(sigma2_e_hat) + logabs + (n - p))
    return ResidualOnlyResult(
        sigma2={"e": float(sigma2_e_hat)},
        pve={"e": 1.0},
        component_var={"e": float(sigma2_e_hat)},
        beta=beta,
        log_lik=float(log_lik),
        n=n, p=p,
    )


@dataclass
class NestedREMLComparison:
    """Nested-model REML comparison."""
    fits: dict[str, MultiREMLResult | ResidualOnlyResult]
    model_specs: dict[str, list[str]]   # {model_name: [kernel_names]}
    likelihood_table: pd.DataFrame      # cols: model, kernels, n, p, log_lik, n_components, n_params, sigma2_json, pve_json, boundary_components, optimizer_status, n_iter
    kernels_used: list[str]


def _default_model_specs(
    kernel_names: list[str],
    strategy: Literal["exhaustive", "leave_one_out", "singletons"] = "exhaustive",
) -> dict[str, list[str]]:
    """Build a default nested model spec dict.

    - 'exhaustive': all 2^J subsets + residual-only (2^J+1 models). Good for J ≤ 3.
    - 'leave_one_out': null + J leave-one-out + full (J+2 models).
    - 'singletons': null + per-kernel singletons + full (J+2 models).
    """
    specs: dict[str, list[str]] = {"e": []}
    J = len(kernel_names)
    if strategy == "exhaustive":
        for r in range(1, J + 1):
            for combo in combinations(kernel_names, r):
                name = "+".join(combo) + "+e"
                specs[name] = list(combo)
    elif strategy == "leave_one_out":
        for skip in kernel_names:
            kept = [k for k in kernel_names if k != skip]
            if not kept:
                continue
            name = "+".join(kept) + "+e"
            specs[name] = kept
        name = "+".join(kernel_names) + "+e"
        specs[name] = list(kernel_names)
    elif strategy == "singletons":
        for name in kernel_names:
            specs[f"{name}+e"] = [name]
        full = "+".join(kernel_names) + "+e"
        specs[full] = list(kernel_names)
    else:
        raise ValueError(f"unknown strategy={strategy!r}")
    return specs


def compare_nested_reml(
    y: np.ndarray,
    X: np.ndarray,
    kernels: dict[str, np.ndarray],
    *,
    model_specs: dict[str, list[str]] | None = None,
    strategy: Literal["exhaustive", "leave_one_out", "singletons"] = "exhaustive",
    fit_kwargs: dict | None = None,
) -> NestedREMLComparison:
    """Fit each model in ``model_specs`` with the same direct REML objective.

    Args:
        y, X: as in fit_multi_reml.
        kernels: master dict of all kernel matrices (only those referenced in
            model_specs are used).
        model_specs: {model_name: [kernel_names]}.
            Empty list = residual-only (uses closed-form REML).
            Default: auto-generate via ``strategy``.
        strategy: ignored if model_specs given.
        fit_kwargs: extra kwargs passed to ``fit_multi_reml`` (e.g. n_starts).

    Returns:
        NestedREMLComparison.
    """
    fit_kwargs = dict(fit_kwargs or {})
    kernel_names = list(kernels.keys())
    if model_specs is None:
        model_specs = _default_model_specs(kernel_names, strategy=strategy)

    fits: dict[str, MultiREMLResult | ResidualOnlyResult] = {}
    rows: list[dict] = []
    for model_name, k_names in model_specs.items():
        if len(k_names) == 0:
            res = residual_only_reml(y, X)
        else:
            sub_kernels = {k: kernels[k] for k in k_names}
            res = fit_multi_reml(y, X, sub_kernels, **fit_kwargs)
        fits[model_name] = res

        # Track number of free parameters: J genetic + 1 residual + p fixed
        n_genetic = len(k_names)
        n_params = n_genetic + 1 + X.shape[1]

        rows.append({
            "model": model_name,
            "kernels": ",".join(k_names) if k_names else "",
            "n": res.n,
            "p": res.p,
            "log_lik": res.log_lik,
            "n_components": n_genetic + 1,
            "n_params": n_params,
            "sigma2_json": json.dumps(res.sigma2),
            "pve_json": json.dumps(res.pve),
            "boundary_components": ",".join(res.boundary_components) if res.boundary_components else "",
            "optimizer_status": getattr(res, "optimizer_status", True),
            "n_iter": getattr(res, "n_iter", 0),
        })

    likelihood_table = pd.DataFrame(rows).sort_values("log_lik", ascending=False).reset_index(drop=True)
    return NestedREMLComparison(
        fits=fits,
        model_specs=model_specs,
        likelihood_table=likelihood_table,
        kernels_used=sorted({k for ks in model_specs.values() for k in ks}),
    )


# =====================================================================
# Phase 2 M2.4.2 Step 3 — Boundary-Corrected LRT
# =====================================================================
#
# When testing H0: σ²_j = 0 (variance component on boundary of parameter
# space), the asymptotic LRT statistic does NOT follow chi²_k. Instead
# it follows a chi-square mixture (Self-Liang 1987; Stram-Lee 1994):
#
#     T ~ Σ_j w_j chi²_j     where w_j = C(k, j) · 2^(-k)
#
# For k=1 boundary component: 50:50 mixture of chi²_0 (point mass) and chi²_1.
# For k=2: 25:50:25 over chi²_0/1/2.
# The binomial weights assume k INDEPENDENT components (diagonal Fisher info)
# — correlated kernels (e.g. K_A K_C in cultivar panels) make these
# approximate; Step 4 parametric bootstrap p-value is the ground truth.


@dataclass(frozen=True)
class BoundaryLRTResult:
    """Boundary-corrected LRT for nested REML comparison."""
    null_model: str
    alt_model: str
    ll_null: float
    ll_alt: float
    statistic: float                       # max(0, statistic_raw)
    statistic_raw: float                   # 2·(ll_alt − ll_null), possibly negative
    df_added: int
    p_naive: float                         # asymptotic chi²_df
    p_mixture: float                       # boundary-corrected mixture
    mixture_weights: dict[int, float]      # {j: w_j}
    added_components: tuple[str, ...]
    null_boundary_components: tuple[str, ...]  # boundary fits in NULL model — caveat for LRT
    is_nested: bool
    clipped: bool                          # True if statistic_raw < 0 was clipped
    both_converged: bool
    boundary_method: str = "self-liang-stram-lee-binomial"
    bootstrap_p: float | None = None       # filled by Step 4 if available


def lrt_boundary_pvalue(
    statistic: float,
    df_added: int,
    *,
    weights: dict[int, float] | None = None,
) -> tuple[float, float, dict[int, float]]:
    """Compute (p_naive, p_mixture, weights) for a boundary-corrected LRT.

    Args:
        statistic: T = max(0, 2(ll_alt − ll_null)). If non-positive, returns
            p_mixture = 1.0 (inclusive upper tail at the atom T=0).
        df_added: number of variance components tested at boundary (k ≥ 1).
        weights: optional override; default binomial {j: C(k,j)/2^k for j=0..k}.

    Returns:
        (p_naive, p_mixture, weights).
    """
    from math import comb
    from scipy import stats

    if df_added < 1:
        raise ValueError(f"df_added must be ≥ 1, got {df_added}")
    if weights is None:
        denom = 2.0 ** df_added
        weights = {j: comb(df_added, j) / denom for j in range(df_added + 1)}
    else:
        # Validate user weights
        w_sum = sum(weights.values())
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {w_sum}")
        for j in weights:
            if not (0 <= j <= df_added):
                raise ValueError(f"weight index {j} out of [0, {df_added}]")

    T = max(0.0, float(statistic))
    p_naive = float(stats.chi2.sf(T, df=df_added))
    p_mix = 0.0
    for j, w in weights.items():
        if j == 0:
            p_mix += w * (1.0 if T <= 0 else 0.0)
        else:
            p_mix += w * float(stats.chi2.sf(T, df=j))
    # Clamp to [0,1] for floating noise
    p_naive = float(np.clip(p_naive, 0.0, 1.0))
    p_mix = float(np.clip(p_mix, 0.0, 1.0))
    return p_naive, p_mix, dict(weights)


def _extract_added_components(
    comparison: NestedREMLComparison, null_model: str, alt_model: str
) -> tuple[tuple[str, ...], bool]:
    """Return (sorted tuple of components in alt-not-null, is_strict_nested)."""
    if null_model not in comparison.model_specs:
        raise KeyError(f"null_model {null_model!r} not in comparison.fits")
    if alt_model not in comparison.model_specs:
        raise KeyError(f"alt_model {alt_model!r} not in comparison.fits")
    null_k = set(comparison.model_specs[null_model])
    alt_k = set(comparison.model_specs[alt_model])
    is_strict = null_k < alt_k     # strict subset
    added = tuple(sorted(alt_k - null_k))
    return added, is_strict


def boundary_lrt(
    comparison: NestedREMLComparison,
    null_model: str,
    alt_model: str,
    *,
    lrt_negative_tol: float = 1e-8,
) -> BoundaryLRTResult:
    """One boundary-corrected LRT between two nested fits.

    Raises ValueError if (null, alt) is not strict nested (alt_kernels ⊋ null_kernels).
    """
    import warnings

    added, is_nested = _extract_added_components(comparison, null_model, alt_model)
    if not is_nested:
        raise ValueError(
            f"{null_model!r} is not strictly nested in {alt_model!r}; "
            f"need alt_kernels ⊋ null_kernels. null_kernels={comparison.model_specs[null_model]}, "
            f"alt_kernels={comparison.model_specs[alt_model]}."
        )
    df_added = len(added)
    if df_added < 1:
        raise ValueError(f"df_added must be ≥ 1, got {df_added}")

    null_fit = comparison.fits[null_model]
    alt_fit = comparison.fits[alt_model]
    ll_null = float(null_fit.log_lik)
    ll_alt = float(alt_fit.log_lik)
    T_raw = 2.0 * (ll_alt - ll_null)
    clipped = T_raw < -lrt_negative_tol
    if clipped:
        warnings.warn(
            f"LRT statistic_raw={T_raw:.4g} < -{lrt_negative_tol} for "
            f"{null_model} vs {alt_model}; clipping to 0. "
            "Indicates optimizer noise or inconsistent fits.",
            RuntimeWarning,
            stacklevel=2,
        )
    T = max(0.0, T_raw)

    p_naive, p_mix, weights = lrt_boundary_pvalue(T, df_added)

    null_boundary = tuple(sorted(set(getattr(null_fit, "boundary_components", [])) & set(comparison.model_specs[null_model])))

    return BoundaryLRTResult(
        null_model=null_model,
        alt_model=alt_model,
        ll_null=ll_null,
        ll_alt=ll_alt,
        statistic=float(T),
        statistic_raw=float(T_raw),
        df_added=df_added,
        p_naive=p_naive,
        p_mixture=p_mix,
        mixture_weights=weights,
        added_components=added,
        null_boundary_components=null_boundary,
        is_nested=True,
        clipped=clipped,
        both_converged=bool(
            getattr(null_fit, "optimizer_status", True)
            and getattr(alt_fit, "optimizer_status", True)
        ),
    )


def _default_pairs(comparison: NestedREMLComparison) -> list[tuple[str, str]]:
    """Default LRT pair set for paper tables:
      1. each-singleton vs null ('e' → 'A+e', 'e' → 'C+e', ...)
      2. full vs each leave-one-out
      3. full vs null
    Pairs that don't exist in comparison.model_specs are skipped.
    """
    specs = comparison.model_specs
    full_models = [name for name, ks in specs.items() if len(ks) == max(len(v) for v in specs.values())]
    null_models = [name for name, ks in specs.items() if len(ks) == 0]
    null = null_models[0] if null_models else None
    full = full_models[0] if full_models else None
    pairs: list[tuple[str, str]] = []

    # 1. null → each-singleton
    if null is not None:
        for name, ks in specs.items():
            if len(ks) == 1:
                pairs.append((null, name))

    # 2. each-leave-one-out → full
    if full is not None:
        full_set = set(specs[full])
        for name, ks in specs.items():
            if len(ks) == len(full_set) - 1 and set(ks) < full_set:
                pairs.append((name, full))

    # 3. null → full
    if null is not None and full is not None and null != full:
        pairs.append((null, full))

    # De-dup while preserving order
    seen: set[tuple[str, str]] = set()
    dedup: list[tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def boundary_lrt_table(
    comparison: NestedREMLComparison,
    pairs: list[tuple[str, str]] | Literal["default", "all_nested"] = "default",
    *,
    lrt_negative_tol: float = 1e-8,
) -> pd.DataFrame:
    """Compute boundary-corrected LRTs for multiple (null, alt) pairs.

    Args:
        comparison: NestedREMLComparison from compare_nested_reml().
        pairs: 'default' (paper-style 5 contrasts) / 'all_nested' (every strict
            nested pair) / explicit list of (null_model, alt_model).
        lrt_negative_tol: clip T_raw below this and warn.

    Returns:
        pandas DataFrame with one row per LRT.
    """
    if pairs == "default":
        pair_list = _default_pairs(comparison)
    elif pairs == "all_nested":
        names = list(comparison.model_specs.keys())
        pair_list = []
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i == j:
                    continue
                _, is_nested = _extract_added_components(comparison, ni, nj)
                if is_nested:
                    pair_list.append((ni, nj))
    elif isinstance(pairs, list):
        pair_list = list(pairs)
    else:
        raise ValueError(f"unknown pairs={pairs!r}")

    rows: list[dict] = []
    for null_m, alt_m in pair_list:
        result = boundary_lrt(comparison, null_m, alt_m, lrt_negative_tol=lrt_negative_tol)
        rows.append({
            "null_model": result.null_model,
            "alt_model": result.alt_model,
            "added_components": ",".join(result.added_components),
            "ll_null": result.ll_null,
            "ll_alt": result.ll_alt,
            "lrt": result.statistic,
            "lrt_raw": result.statistic_raw,
            "df_added": result.df_added,
            "p_naive": result.p_naive,
            "p_boundary": result.p_mixture,
            "mixture_weights": ";".join(f"{j}:{w:.4g}" for j, w in result.mixture_weights.items()),
            "null_boundary_components": ",".join(result.null_boundary_components),
            "boundary_method": result.boundary_method,
            "clipped": result.clipped,
            "both_converged": result.both_converged,
            "bootstrap_p": result.bootstrap_p,
        })
    return pd.DataFrame(rows)


# =====================================================================
# Phase 2 M2.4.2 Step 4 — Parametric Bootstrap for LRT
# =====================================================================
#
# Simulate y_b under fitted null (β̂_null, V̂_null = Σ σ²_j K_j + σ²_e I),
# refit null + alt with same direct REML, T_b = max(0, 2(ll_alt_b - ll_null_b)),
# Phipson-Smyth p = (1 + #{T_b ≥ T_obs}) / (n_usable + 1).
#
# Used when correlated kernels make binomial chi-bar mixture weights only
# asymptotic; bootstrap is finite-sample ground truth.
#
# Calibration safeguards (so T_obs and T_b are the SAME numerical process):
#  - bootstrap refits inherit the observed fits' n_starts by default; a
#    smaller n_starts is allowed but warns (anti-conservative risk).
#  - only replicates whose null AND alt refits converge enter the p-value
#    (require_converged=True); non-converged finite T_b can underestimate
#    the alt optimum.


@dataclass(frozen=True)
class BootstrapLRTResult:
    """Parametric bootstrap LRT result. Wraps observed BoundaryLRTResult."""
    boundary: BoundaryLRTResult      # observed LRT; .bootstrap_p is backfilled
    bootstrap_p: float
    B_requested: int
    B_success: int                   # finite T_b count
    B_usable: int                    # replicates entering the p-value
    seed: int | None
    T_boot: np.ndarray
    converged: np.ndarray            # (B,) bool: both null & alt optimizer.success
    clipped: np.ndarray              # (B,) bool: T_raw < 0
    success: np.ndarray              # (B,) bool: finite T_b (no exception)
    converged_rate: float
    clip_rate: float
    success_rate: float              # B_success / B
    usable_rate: float               # B_usable / B
    quantiles: dict[str, float]      # {"q05","q50","q95","q99"} of T_boot[usable]
    mcse: float                      # Monte Carlo standard error of bootstrap_p
    jitter_used: float = 0.0         # Cholesky jitter on V̂_null (0 = exact V̂_null)
    n_starts_bootstrap: int = 1      # n_starts used to refit each replicate
    simulation_model: str = "fitted-null"


def _psd_project(K: np.ndarray) -> np.ndarray:
    """Symmetrize K and clip negative eigenvalues to 0.

    Mirrors fit_multi_reml's internal PSD treatment so the bootstrap
    simulation covariance matches the kernel the fitted likelihood used.
    """
    A = np.asarray(K, dtype=np.float64)
    K_sym = 0.5 * (A + A.T)
    evals = np.linalg.eigvalsh(K_sym)
    if float(evals.min()) < 0.0:
        ev, evec = np.linalg.eigh(K_sym)
        ev = np.clip(ev, 0.0, None)
        K_sym = (evec * ev) @ evec.T
        K_sym = 0.5 * (K_sym + K_sym.T)
    return K_sym


def _covariance_from_fit(fit, kernel_names: list[str], kernels: dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct V = σ²_e I + Σ σ²_j K_j from a fitted result.

    Each K_j is symmetrized and PSD-projected (mirrors fit_multi_reml) so the
    simulated covariance matches the kernel used by the fitted likelihood.
    """
    n = fit.n
    V = float(fit.sigma2["e"]) * np.eye(n, dtype=np.float64)
    for k in kernel_names:
        V = V + float(fit.sigma2[k]) * _psd_project(kernels[k])
    return 0.5 * (V + V.T)


def _cholesky_with_jitter(
    V: np.ndarray,
    mean_diag: float,
    ladder: tuple[float, ...] = (0.0, 1e-10, 1e-8, 1e-6),
) -> tuple[np.ndarray, float]:
    """Cholesky with progressive jitter ladder; returns (L, jitter_used)."""
    n = V.shape[0]
    last_err: Exception | None = None
    for jitter in ladder:
        V_try = V + jitter * mean_diag * np.eye(n) if jitter > 0 else V
        try:
            L = np.linalg.cholesky(V_try)
            return L, jitter
        except np.linalg.LinAlgError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Cholesky failed even with max jitter {ladder[-1]}·mean_diag={ladder[-1]*mean_diag:.4g}: {last_err}"
    )


def _fit_one_bootstrap_pair(
    y_b: np.ndarray,
    X: np.ndarray,
    null_kernels: dict[str, np.ndarray],
    alt_kernels: dict[str, np.ndarray],
    fit_kwargs: dict,
) -> tuple[float, float, bool, bool, str | None]:
    """Refit null + alt on y_b. Returns (T_clipped, T_raw, null_ok, alt_ok, err)."""
    try:
        if len(null_kernels) == 0:
            null_b = residual_only_reml(y_b, X)
            null_ok = True
        else:
            null_b = fit_multi_reml(y_b, X, null_kernels, **fit_kwargs)
            null_ok = bool(null_b.optimizer_status)
        alt_b = fit_multi_reml(y_b, X, alt_kernels, **fit_kwargs)
        alt_ok = bool(alt_b.optimizer_status)
        T_raw = 2.0 * (alt_b.log_lik - null_b.log_lik)
        T_clipped = max(0.0, T_raw)
        return float(T_clipped), float(T_raw), null_ok, alt_ok, None
    except Exception as e:
        return float("nan"), float("nan"), False, False, str(e)


def _bh_adjust(p_array: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values. Monotone-enforced from sorted p."""
    p_arr = np.asarray(p_array, dtype=np.float64)
    n = len(p_arr)
    if n == 0:
        return p_arr.copy()
    if not np.all(np.isfinite(p_arr)):
        raise ValueError("_bh_adjust requires all-finite p-values; filter NaN/inf first")
    order = np.argsort(p_arr)
    p_sorted = p_arr[order]
    ranks = np.arange(1, n + 1)
    q_sorted = p_sorted * n / ranks
    # Enforce monotone (q_(i) ≤ q_(i+1)): take cum-min from right
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(n, dtype=np.float64)
    q[order] = q_sorted
    return q


def parametric_bootstrap_lrt(
    comparison: NestedREMLComparison,
    null_model: str,
    alt_model: str,
    *,
    y: np.ndarray,
    X: np.ndarray,
    kernels: dict[str, np.ndarray],
    B: int = 200,
    seed: int | None = None,
    bootstrap_fit_kwargs: dict | None = None,
    observed_lrt_negative_tol: float = 1e-8,
    n_jobs: int = 1,
    min_success_rate: float = 0.95,
    require_converged: bool = True,
    on_failure: Literal["raise", "warn"] = "raise",
) -> BootstrapLRTResult:
    """Parametric bootstrap LRT (simulate under fitted null).

    Simulates y_b ~ N(X β̂_null, V̂_null), refits null + alt with the *same*
    direct REML objective, and forms T_b = max(0, 2(ll_alt_b − ll_null_b)).
    p = (1 + #{T_b ≥ T_obs}) / (n_usable + 1)   (Phipson-Smyth).

    For T_obs and T_b to be the *same numerical process*, the bootstrap refits
    inherit the observed fits' ``n_starts`` by default (override via
    ``bootstrap_fit_kwargs``); a smaller n_starts is allowed but warns, as it
    can underestimate the alt optimum and make ``bootstrap_p`` anti-conservative.

    Args:
        comparison: NestedREMLComparison from compare_nested_reml.
        null_model, alt_model: model names from comparison.fits; must be
            strict-nested (alt_kernels ⊋ null_kernels).
        y, X, kernels: original data + design + kernel matrices used for fitting
            (must match `comparison.fits` shape).
        B: number of bootstrap replicates (acceptance: 200; paper: 1000+).
        seed: master seed; np.random.SeedSequence spawns 2 independent child
            states per replicate (noise draw + refit multi-start), so results
            are deterministic regardless of n_jobs even when n_starts > 1.
        bootstrap_fit_kwargs: passed to fit_multi_reml for each replicate;
            ``n_starts`` defaults to the observed alt fit's n_starts.
        n_jobs: 1 = serial; >1 = joblib parallel (order-preserving).
        min_success_rate: raise/warn if B_usable/B < this (default 0.95).
        require_converged: if True (default), only replicates whose null *and*
            alt refits report optimizer convergence enter the p-value; finite
            but non-converged T_b can systematically underestimate the alt
            optimum. Set False to use all finite T_b (approximate).
        on_failure: "raise" or "warn" when usable rate violated.

    Returns:
        BootstrapLRTResult; ``.boundary.bootstrap_p`` is backfilled.
    """
    import warnings

    if B < 1:
        raise ValueError(f"B must be >= 1, got {B}")

    obs = boundary_lrt(comparison, null_model, alt_model, lrt_negative_tol=observed_lrt_negative_tol)
    T_obs = obs.statistic

    null_fit = comparison.fits[null_model]
    alt_fit = comparison.fits[alt_model]
    null_kernel_names = comparison.model_specs[null_model]
    alt_kernel_names = comparison.model_specs[alt_model]
    n = null_fit.n

    # Resolve required kernels with an explicit error (not a bare KeyError).
    needed = set(null_kernel_names) | set(alt_kernel_names)
    missing = sorted(k for k in needed if k not in kernels)
    if missing:
        raise ValueError(
            f"kernels dict missing {missing} required by models "
            f"{null_model!r}/{alt_model!r}; have {sorted(kernels.keys())}"
        )

    # n_starts inheritance: keep T_obs and T_b the same numerical process.
    observed_n_starts = int(getattr(alt_fit, "n_starts", 1) or 1)
    if bootstrap_fit_kwargs is None:
        fit_kwargs = {"n_starts": observed_n_starts}
    else:
        fit_kwargs = dict(bootstrap_fit_kwargs)
        fit_kwargs.setdefault("n_starts", observed_n_starts)
    n_starts_bootstrap = int(fit_kwargs.get("n_starts", 1))
    if n_starts_bootstrap < observed_n_starts:
        warnings.warn(
            f"bootstrap n_starts={n_starts_bootstrap} < observed fit "
            f"n_starts={observed_n_starts}: bootstrap T_b may underestimate the "
            "alt optimum and make bootstrap_p anti-conservative (throughput "
            "mode, not paper-grade calibration).",
            RuntimeWarning, stacklevel=2,
        )

    # Cholesky of V̂_null (reuse across all B reps)
    if len(null_kernel_names) == 0:
        sig2_e = float(null_fit.sigma2["e"])
        L_V = np.sqrt(max(sig2_e, 1e-30)) * np.eye(n, dtype=np.float64)
        jitter_used = 0.0
    else:
        V = _covariance_from_fit(null_fit, null_kernel_names, kernels)
        mean_diag = float(np.trace(V) / n)
        L_V, jitter_used = _cholesky_with_jitter(V, mean_diag)
    if jitter_used > 0:
        warnings.warn(
            f"V̂_null needed Cholesky jitter {jitter_used:.3g}·mean_diag; "
            "simulated covariance is V̂_null + jitter·mean_diag·I, marginally "
            "inflated vs the fitted null.",
            RuntimeWarning, stacklevel=2,
        )
    beta_null = np.asarray(null_fit.beta, dtype=np.float64)

    # Two independent child states per replicate (noise draw + refit RNG), so
    # determinism holds regardless of n_jobs and n_starts.
    ss = np.random.SeedSequence(seed)
    seed_pairs: list[tuple[int, int]] = []
    for child in ss.spawn(B):
        st = child.generate_state(2, dtype=np.uint32)
        seed_pairs.append((int(st[0]), int(st[1])))

    null_k_dict = {k: kernels[k] for k in null_kernel_names}
    alt_k_dict = {k: kernels[k] for k in alt_kernel_names}

    def _one(seed_z: int, seed_fit: int):
        rng = np.random.default_rng(seed_z)
        z = rng.standard_normal(n)
        y_b = X @ beta_null + L_V @ z
        rep_kwargs = {**fit_kwargs, "random_state": seed_fit}
        return _fit_one_bootstrap_pair(y_b, X, null_k_dict, alt_k_dict, rep_kwargs)

    if n_jobs == 1:
        results = [_one(sz, sf) for sz, sf in seed_pairs]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError as e:
            raise RuntimeError(
                f"joblib needed for n_jobs={n_jobs} but not importable. pip install joblib"
            ) from e
        results = Parallel(n_jobs=n_jobs)(delayed(_one)(sz, sf) for sz, sf in seed_pairs)

    T_boot = np.array([r[0] for r in results], dtype=np.float64)
    T_raw = np.array([r[1] for r in results], dtype=np.float64)
    converged = np.array([(r[2] and r[3]) for r in results], dtype=bool)
    clipped = np.where(np.isfinite(T_raw), T_raw < 0, False)
    success = np.isfinite(T_boot)
    B_success = int(success.sum())
    success_rate = B_success / B
    converged_rate = float(converged.sum() / B)
    clip_rate = float(clipped.sum() / B)

    # Replicates entering the p-value.
    usable = (success & converged) if require_converged else success
    B_usable = int(usable.sum())
    usable_rate = B_usable / B

    if usable_rate < min_success_rate:
        msg = (
            f"bootstrap usable rate {usable_rate:.3f} below threshold "
            f"{min_success_rate}; B_usable={B_usable}/{B} "
            f"(finite={B_success}, converged={int(converged.sum())}, "
            f"require_converged={require_converged}). "
            "Inspect convergence / Cholesky failures."
        )
        if on_failure == "raise":
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    T_used = T_boot[usable]
    if len(T_used) == 0:
        bootstrap_p = float("nan")
        mcse = float("nan")
        quantiles = {q: float("nan") for q in ("q05", "q50", "q95", "q99")}
    else:
        n_used = len(T_used)
        bootstrap_p = float((1.0 + np.sum(T_used >= T_obs)) / (n_used + 1.0))
        # MCSE of the Phipson-Smyth estimate p̂=(K+1)/(n+1), K~Binomial(n,p):
        #   Var(p̂) = n·p(1−p)/(n+1)²  →  plug in p̂.
        mcse = float(
            np.sqrt(max(n_used * bootstrap_p * (1.0 - bootstrap_p), 0.0)) / (n_used + 1.0)
        )
        quantiles = {
            "q05": float(np.quantile(T_used, 0.05)),
            "q50": float(np.quantile(T_used, 0.50)),
            "q95": float(np.quantile(T_used, 0.95)),
            "q99": float(np.quantile(T_used, 0.99)),
        }

    obs_with_p = replace(obs, bootstrap_p=bootstrap_p)

    return BootstrapLRTResult(
        boundary=obs_with_p,
        bootstrap_p=bootstrap_p,
        B_requested=B,
        B_success=B_success,
        B_usable=B_usable,
        seed=seed,
        T_boot=T_boot,
        converged=converged,
        clipped=clipped,
        success=success,
        converged_rate=converged_rate,
        clip_rate=clip_rate,
        success_rate=success_rate,
        usable_rate=usable_rate,
        quantiles=quantiles,
        mcse=mcse,
        jitter_used=float(jitter_used),
        n_starts_bootstrap=n_starts_bootstrap,
        simulation_model="fitted-null",
    )


def bootstrap_lrt_table(
    comparison: NestedREMLComparison,
    pairs: list[tuple[str, str]] | Literal["default", "all_nested"] = "default",
    *,
    y: np.ndarray,
    X: np.ndarray,
    kernels: dict[str, np.ndarray],
    B: int = 200,
    seed: int | None = None,
    bootstrap_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    min_success_rate: float = 0.95,
    require_converged: bool = True,
    on_failure: Literal["raise", "warn"] = "raise",
    adjust_method: Literal["bh"] | None = None,
) -> pd.DataFrame:
    """Compute parametric bootstrap LRT for multiple (null, alt) pairs."""
    # Resolve pairs same as boundary_lrt_table
    if pairs == "default":
        pair_list = _default_pairs(comparison)
    elif pairs == "all_nested":
        names = list(comparison.model_specs.keys())
        pair_list = []
        for ni in names:
            for nj in names:
                if ni == nj:
                    continue
                _, nested = _extract_added_components(comparison, ni, nj)
                if nested:
                    pair_list.append((ni, nj))
    elif isinstance(pairs, list):
        pair_list = list(pairs)
    else:
        raise ValueError(f"unknown pairs={pairs!r}")

    rows: list[dict] = []
    for null_m, alt_m in pair_list:
        res = parametric_bootstrap_lrt(
            comparison, null_m, alt_m,
            y=y, X=X, kernels=kernels,
            B=B, seed=seed,
            bootstrap_fit_kwargs=bootstrap_fit_kwargs,
            n_jobs=n_jobs,
            min_success_rate=min_success_rate,
            require_converged=require_converged,
            on_failure=on_failure,
        )
        b = res.boundary
        rows.append({
            "null_model": b.null_model,
            "alt_model": b.alt_model,
            "added_components": ",".join(b.added_components),
            "ll_null": b.ll_null,
            "ll_alt": b.ll_alt,
            "lrt": b.statistic,
            "lrt_raw": b.statistic_raw,
            "df_added": b.df_added,
            "p_naive": b.p_naive,
            "p_boundary": b.p_mixture,
            "mixture_weights": ";".join(f"{j}:{w:.4g}" for j, w in b.mixture_weights.items()),
            "bootstrap_p": res.bootstrap_p,
            "B_requested": res.B_requested,
            "B_success": res.B_success,
            "B_usable": res.B_usable,
            "mcse": res.mcse,
            "T_boot_q95": res.quantiles["q95"],
            "success_rate": res.success_rate,
            "converged_rate": res.converged_rate,
            "usable_rate": res.usable_rate,
            "clip_rate": res.clip_rate,
            "jitter_used": res.jitter_used,
            "n_starts_bootstrap": res.n_starts_bootstrap,
            "clipped": b.clipped,
            "both_converged": b.both_converged,
            "boundary_method": b.boundary_method,
            "null_boundary_components": ",".join(b.null_boundary_components),
        })
    table = pd.DataFrame(rows)
    if adjust_method == "bh" and len(table) > 0:
        p = table["bootstrap_p"].to_numpy(dtype=np.float64)
        q = np.full(len(p), np.nan, dtype=np.float64)
        finite = np.isfinite(p)
        if finite.any():
            q[finite] = _bh_adjust(p[finite])
        table["bootstrap_q_bh"] = q
    return table


# =====================================================================
# Phase 2 M2.4.2 Step 5 — PVE Sensitivity Grid
# =====================================================================
#
# Re-fit the full multi-kernel REML over a grid of preprocessing /
# optimizer choices and report how the PVE decomposition drifts:
#   - y transform:   raw | zscore
#   - kernel norm:   trace | frobenius | none
#   - n_starts:      1 | 10
#
# The paper-grade PVE (σ²_j·trace(K_j)/n  normalised) is analytically
# invariant to K↦cK and to affine y transforms (given an intercept in X);
# this grid demonstrates that empirically. A non-trivial drift on the
# kernel-norm or y axis therefore signals an *optimizer* problem (local
# optimum / unconverged), not statistical instability — which is why each
# cell keeps optimizer_status / n_iter / boundary_components, and n_starts
# is itself a grid axis (n_starts=10 is the robust anchor). LRT / bootstrap
# are deliberately NOT rerun per cell: Step 3/4 already cover inference, and
# bootstrap Monte-Carlo noise would blur the sensitivity claim.


_Y_MODES = ("raw", "zscore")
_KERNEL_NORMS = ("trace", "frobenius", "none")


def _transform_y(y: np.ndarray, mode: str) -> np.ndarray:
    """Apply a y preprocessing transform. Input is never mutated."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    if mode == "raw":
        return y.copy()
    if mode == "zscore":
        sd = float(np.std(y, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"cannot z-score y: std(y, ddof=1)={sd}")
        return (y - float(np.mean(y))) / sd
    raise ValueError(f"unknown y mode={mode!r}; expected one of {_Y_MODES}")


def _apply_kernel_norm(K: np.ndarray, mode: str) -> np.ndarray:
    """Apply a kernel normalization. Input is never mutated."""
    if mode in ("trace", "frobenius"):
        return normalize_kernel(K, mode=mode)
    if mode == "none":
        return np.array(K, dtype=np.float64, copy=True)
    raise ValueError(f"unknown kernel norm={mode!r}; expected one of {_KERNEL_NORMS}")


def _max_offdiag_corr(kernel_corr: dict[str, dict[str, float]]) -> float:
    """Largest absolute off-diagonal kernel correlation (0.0 if single kernel)."""
    vals = [
        abs(float(c))
        for ni, row in kernel_corr.items()
        for nj, c in row.items()
        if ni != nj
    ]
    return float(max(vals)) if vals else 0.0


def _as_int(v: object, what: str) -> int:
    """Coerce v to int, rejecting bools and non-integer-valued floats."""
    if isinstance(v, bool):
        raise ValueError(f"{what} must be an integer, got bool {v!r}")
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return int(v)
    raise ValueError(f"{what} must be an integer, got {v!r}")


@dataclass(frozen=True)
class SensitivityGridResult:
    """PVE sensitivity grid over (y transform × kernel norm × n_starts)."""
    table: pd.DataFrame              # one row per grid cell
    drift_table: pd.DataFrame        # per (component × axis) PVE drift attribution
    reference_cell: dict[str, object]  # {"y_mode","kernel_norm","n_starts"}
    genetic_components: list[str]    # kernel names (PVE_e excluded)
    all_components: list[str]        # kernel names + ["e"]


def pve_sensitivity_grid(
    y: np.ndarray,
    X: np.ndarray,
    kernels: dict[str, np.ndarray],
    *,
    y_modes: tuple[str, ...] = _Y_MODES,
    kernel_norms: tuple[str, ...] = _KERNEL_NORMS,
    n_starts_grid: tuple[int, ...] = (1, 10),
    reference: tuple[str, str, int] = ("raw", "trace", 10),
    random_state: int | None = 42,
    fit_kwargs: dict | None = None,
) -> SensitivityGridResult:
    """Re-fit the full multi-kernel REML over a preprocessing/optimizer grid.

    Demonstrates that the PVE decomposition (the headline subgenome split) is
    robust to arbitrary input-scale choices. PVE is analytically invariant to
    K↦cK and to affine y transforms (given an intercept in X); a non-trivial
    drift therefore flags an optimizer issue, cross-checked via the per-cell
    ``optimizer_status`` column (and the n_starts axis, n_starts=10 = anchor).

    Args:
        y: phenotype (n,) — RAW values; transforms are applied internally.
        X: fixed-effect design (n, p); should contain an intercept column for
            the z-score invariance argument to hold.
        kernels: {name: K} — RAW base kernels (un-normalized); the grid applies
            each normalization itself. Passing pre-normalized kernels makes the
            "none" cell meaningless. Inputs are not mutated.
        y_modes / kernel_norms / n_starts_grid: grid axes.
        reference: (y_mode, kernel_norm, n_starts) cell that deltas are measured
            against; must lie in the grid.
        random_state: passed to every fit_multi_reml (fixed across cells so the
            grid carries no Monte-Carlo noise).
        fit_kwargs: extra kwargs for fit_multi_reml (e.g. maxiter); must NOT
            include n_starts / random_state (grid-controlled). Each cell also
            gets scale-aware ``init`` / ``bounds`` (rescaled by n/trace(K_j) so
            the σ²_j optimum stays reachable for any kernel norm); passing
            explicit ``init`` / ``bounds`` here overrides that.

    Returns:
        SensitivityGridResult with ``table`` (one row per cell) and
        ``drift_table`` (per component × axis PVE range). Each row carries
        ``optimizer_status`` and ``genetic_upper_bound_hit`` so a numerically
        unreliable cell is never mistaken for genuine PVE instability.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}")
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X must be (n,p) matching y; got X={X.shape}, y={y.shape}")
    if not kernels:
        raise ValueError("kernels dict is empty")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(X))):
        raise ValueError("y / X must be all finite")

    y_modes = tuple(y_modes)
    kernel_norms = tuple(kernel_norms)
    n_starts_grid = tuple(_as_int(s, "n_starts") for s in n_starts_grid)
    if not (y_modes and kernel_norms and n_starts_grid):
        raise ValueError("y_modes / kernel_norms / n_starts_grid must be non-empty")
    for s in n_starts_grid:
        if s < 1:
            raise ValueError(f"n_starts values must be >= 1, got {s}")
    for m in y_modes:
        if m not in _Y_MODES:
            raise ValueError(f"unknown y mode {m!r}; expected subset of {_Y_MODES}")
    for m in kernel_norms:
        if m not in _KERNEL_NORMS:
            raise ValueError(f"unknown kernel norm {m!r}; expected subset of {_KERNEL_NORMS}")

    reference = tuple(reference)
    if len(reference) != 3:
        raise ValueError(
            f"reference must be (y_mode, kernel_norm, n_starts), got {reference!r}"
        )
    ref_y, ref_norm = reference[0], reference[1]
    ref_starts = _as_int(reference[2], "reference n_starts")
    if ref_y not in y_modes or ref_norm not in kernel_norms or ref_starts not in n_starts_grid:
        raise ValueError(
            f"reference {reference} not in grid (y_modes={y_modes}, "
            f"kernel_norms={kernel_norms}, n_starts_grid={n_starts_grid})"
        )

    fk = dict(fit_kwargs or {})
    for forbidden in ("n_starts", "random_state"):
        if forbidden in fk:
            raise ValueError(f"fit_kwargs must not set {forbidden!r}; it is grid-controlled")

    kernel_names = list(kernels.keys())
    all_components = kernel_names + ["e"]

    # ---- run the grid -------------------------------------------------
    rows: list[dict] = []
    cell_pve: dict[tuple[str, str, int], dict[str, float]] = {}
    for y_mode, k_norm, n_starts in product(y_modes, kernel_norms, n_starts_grid):
        y_c = _transform_y(y, y_mode)
        K_dict = {name: _apply_kernel_norm(kernels[name], k_norm) for name in kernel_names}

        # Scale-aware init / bounds: fit_multi_reml's var(y)-based defaults live
        # in raw-σ² space, but the data-equivalent scale is σ²_j·trace(K_j)/n.
        # Rescaling by n/trace(K_j) keeps the σ²_j optimum reachable regardless
        # of kernel norm (esp. the un-normalized "none" cell with extreme scale).
        # For trace-norm kernels (trace == n) this is the identity, so trace
        # cells stay bit-identical to fit_multi_reml defaults. Explicit
        # fit_kwargs (init / bounds / ...) override these.
        var_yc = float(np.var(y_c, ddof=1))
        init_cell: dict[str, float] = {"e": 0.7 * var_yc}
        bounds_cell: dict[str, tuple[float, float]] = {"e": (var_yc * 1e-8, var_yc * 1e4)}
        for name in kernel_names:
            tr = float(np.trace(K_dict[name]))
            scale = (len(y_c) / tr) if tr > 0 else 1.0
            init_cell[name] = 0.3 * var_yc / len(kernel_names) * scale
            bounds_cell[name] = (0.0, var_yc * 1e4 * scale)
        cell_fk = {"init": init_cell, "bounds": bounds_cell, **fk}

        res = fit_multi_reml(
            y_c, X, K_dict, n_starts=n_starts, random_state=random_state, **cell_fk,
        )
        cell_pve[(y_mode, k_norm, n_starts)] = dict(res.pve)
        # Genetic σ²_j pinned at its upper bound ⇒ fit unreliable; surface it so
        # a pinned-fit drift is never misread as instability. Check against the
        # bounds actually passed to fit_multi_reml (a user fit_kwargs["bounds"]
        # wins), falling back to fit_multi_reml's own default for missing keys.
        eff_bounds = cell_fk["bounds"]
        ub_hit = False
        for name in kernel_names:
            if isinstance(eff_bounds, dict) and name in eff_bounds:
                ub = float(eff_bounds[name][1])
            else:
                ub = var_yc * 1e4
            if float(res.sigma2[name]) >= ub * (1.0 - 1e-6):
                ub_hit = True
                break
        row = {
            "y_mode": y_mode,
            "kernel_norm": k_norm,
            "n_starts": n_starts,
            "is_reference": (
                y_mode == ref_y and k_norm == ref_norm and n_starts == ref_starts
            ),
            "n": res.n,
            "p": res.p,
            "log_lik": res.log_lik,
            "optimizer_status": bool(res.optimizer_status),
            "genetic_upper_bound_hit": bool(ub_hit),
            "n_iter": res.n_iter,
            "best_start": res.best_start,
            "boundary_components": ",".join(res.boundary_components),
            "kernel_corr_max_offdiag": _max_offdiag_corr(res.kernel_corr),
            "kernel_design_cond": res.kernel_design_cond,
        }
        for comp in all_components:
            row[f"sigma2_{comp}"] = float(res.sigma2[comp])
            row[f"pve_{comp}"] = float(res.pve[comp])
        rows.append(row)

    # ---- deltas vs reference (PVE only; log_lik is NOT comparable across
    #      y transforms because the y scale shifts the likelihood) --------
    ref_pve = cell_pve[(ref_y, ref_norm, ref_starts)]
    for row in rows:
        max_abs_genetic = 0.0
        for comp in all_components:
            d = row[f"pve_{comp}"] - ref_pve[comp]
            row[f"delta_pve_{comp}_vs_ref"] = d
            row[f"abs_delta_pve_{comp}_vs_ref"] = abs(d)
            if comp != "e":
                max_abs_genetic = max(max_abs_genetic, abs(d))
        row["max_abs_delta_genetic_vs_ref"] = max_abs_genetic

    table = pd.DataFrame(rows)

    # ---- drift attribution -------------------------------------------
    # Isolate each axis by holding the other two at the reference level;
    # 'global' ranges over the whole grid.
    drift_rows: list[dict] = []
    for comp in all_components:
        col = f"pve_{comp}"
        axis_masks = {
            "y_mode": (table["kernel_norm"] == ref_norm) & (table["n_starts"] == ref_starts),
            "kernel_norm": (table["y_mode"] == ref_y) & (table["n_starts"] == ref_starts),
            "n_starts": (table["y_mode"] == ref_y) & (table["kernel_norm"] == ref_norm),
            "global": pd.Series(True, index=table.index),
        }
        for axis, mask in axis_masks.items():
            vals = table.loc[mask, col]
            pmin, pmax = float(vals.min()), float(vals.max())
            drift_rows.append({
                "component": comp,
                "is_genetic": comp != "e",
                "axis": axis,
                "n_cells": int(mask.sum()),
                "pve_min": pmin,
                "pve_max": pmax,
                "pve_range": pmax - pmin,
            })
    drift_table = pd.DataFrame(drift_rows)

    return SensitivityGridResult(
        table=table,
        drift_table=drift_table,
        reference_cell={"y_mode": ref_y, "kernel_norm": ref_norm, "n_starts": ref_starts},
        genetic_components=list(kernel_names),
        all_components=list(all_components),
    )
