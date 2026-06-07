"""Null-simulation calibration for the multi-kernel variance-component LRT.

Checks that the boundary-corrected LRT (and, optionally, the parametric
bootstrap LRT) controls type-I error under H0, i.e. a variance component
truly equal to 0.

For a ``NullSimulationScenario`` (which fixes the true variance components,
with the tested component(s) pinned to 0), simulate ``n_sim`` phenotypes

    y_b = X β  +  L z,    V_true = σ²_e I + Σ_j σ²_j K_j,    L Lᵀ = V_true

refit the nested pair with the same direct REML objective, and record the
LRT statistic plus three p-values: asymptotic χ²_df (``p_naive``), the
boundary-corrected Stram-Lee binomial chi-bar (``p_mixture``), and the
optional Phipson-Smyth parametric bootstrap (``bootstrap_p``).

Calibration is the empirical type-I rate at nominal α with a Wilson CI.
The boundary p-value has a point mass at 1 (when T=0) and is not
continuous-Uniform, so KS / standard λ_GC do not apply: lower-tail rejection
``P(p ≤ α)`` is the check, and a chi-bar tail-inflation ratio replaces λ_GC.
Per-SNP QQ / λ_GC are out of scope; this calibrates the variance-component
LRT only.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import (
    _cholesky_with_jitter,
    _psd_project,
    boundary_lrt,
    compare_nested_reml,
    parametric_bootstrap_lrt,
    residual_only_reml,
)
from .lmm import fit_multi_reml

_Z95 = 1.959963984540054  # standard normal 97.5% quantile


# chi-bar-square helpers (Self-Liang / Stram-Lee binomial mixture)


def _chibar_weights(df_added: int) -> dict[int, float]:
    """Binomial chi-bar weights {j: C(k,j)/2^k} for k = df_added boundary comps."""
    if df_added < 1:
        raise ValueError(f"df_added must be >= 1, got {df_added}")
    denom = 2.0 ** df_added
    return {j: comb(df_added, j) / denom for j in range(df_added + 1)}


def _chibar_sf(t: float, weights: dict[int, float]) -> float:
    """Survival function P(T > t) of the chi-bar mixture (atom χ²_0 at 0)."""
    from scipy import stats
    t = float(t)
    s = 0.0
    for j, w in weights.items():
        if j == 0:
            s += w * (1.0 if t <= 0.0 else 0.0)
        else:
            s += w * float(stats.chi2.sf(t, df=j))
    return s


def _chibar_quantile(q: float, weights: dict[int, float]) -> float:
    """q-quantile of the chi-bar mixture: smallest t with CDF(t) >= q.

    CDF(0) = w_0 (the atom), so q <= w_0 -> quantile 0.
    """
    from scipy import stats
    from scipy.optimize import brentq

    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1), got {q}")
    w0 = weights.get(0, 0.0)
    if q <= w0:
        return 0.0
    target_sf = 1.0 - q                       # want SF(t) == target_sf
    k = max(weights)
    # Upper bracket: SF is bounded above by chi2.sf(t,k); grow until SF < target.
    t_hi = float(stats.chi2.isf(target_sf, df=k)) + 1.0
    for _ in range(60):
        if _chibar_sf(t_hi, weights) < target_sf:
            break
        t_hi *= 2.0
    else:
        raise RuntimeError(f"could not bracket chi-bar quantile for q={q}")
    return float(brentq(lambda t: _chibar_sf(t, weights) - target_sf, 1e-12, t_hi))


def _wilson_ci(n_reject: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = n_reject / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


# Scenario + result types


@dataclass(frozen=True)
class NullSimulationScenario:
    """A null-hypothesis simulation scenario for one nested LRT contrast.

    The tested component(s) (alt-not-null kernels) must have sigma2_true == 0;
    that is what makes it a null scenario.
    """
    name: str
    null_model: str
    alt_model: str
    model_specs: dict[str, list[str]]   # {null_model: [...], alt_model: [...]}
    sigma2_true: dict[str, float]       # {kernel: σ², "e": σ²_e}; tested comps = 0
    beta: np.ndarray                    # (p,) fixed mean effect
    df_added: int
    tested_components: tuple[str, ...]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NullCalibrationResult:
    """Result of a null-simulation calibration run."""
    scenario: NullSimulationScenario
    n_sim_requested: int
    n_sim_success: int
    seed: int | None
    replicate_table: pd.DataFrame   # one row per simulation
    type1_table: pd.DataFrame       # method × alpha empirical type-I + Wilson CI
    tail_inflation: pd.DataFrame    # chi-bar tail-inflation ratios (q95/q99)
    df_added: int
    converged_rate: float
    clip_rate: float
    e_boundary_rate: float          # fraction of replicates with σ²_e at boundary
    jitter_used: float
    bootstrap_B: int | None


# Scenario construction


def _kernels_of_model(model_name: str) -> list[str]:
    """Parse a default model name ('A+C+e' -> ['A','C'], 'e' -> []).

    Tolerates surrounding whitespace; rejects empty / duplicate kernel tokens.
    """
    toks = [t.strip() for t in str(model_name).split("+")]
    if any(t == "" for t in toks):
        raise ValueError(f"model name {model_name!r} has an empty '+'-token")
    kernels = [t for t in toks if t != "e"]
    if len(set(kernels)) != len(kernels):
        raise ValueError(f"model name {model_name!r} has duplicate kernel tokens")
    return kernels


def scenario_from_reduced_fit(
    name: str,
    y: np.ndarray,
    X: np.ndarray,
    kernels: dict[str, np.ndarray],
    *,
    null_model: str,
    alt_model: str,
    n_starts: int = 10,
    random_state: int | None = None,
    description: str = "",
) -> NullSimulationScenario:
    """Build a null scenario by fitting the reduced (null) model on real y.

    The null model is fit on the observed phenotype to get realistic σ² / β;
    the tested component(s) are pinned to 0. Using the reduced fit rather than
    the full model keeps the tested component's real signal out of the null
    simulation.

    Args:
        name: scenario label.
        y, X: real phenotype + design used to estimate the nuisance σ² / β.
        kernels: master {name: K} dict (canonical, e.g. trace-normalized).
        null_model, alt_model: default-style names ('e', 'A+e', 'A+C+e'); must
            be strict-nested.
        n_starts, random_state: passed to fit_multi_reml for the reduced fit.
    """
    null_k = _kernels_of_model(null_model)
    alt_k = _kernels_of_model(alt_model)
    if not set(null_k) < set(alt_k):
        raise ValueError(
            f"{null_model!r} not strictly nested in {alt_model!r} "
            f"(null kernels {null_k}, alt kernels {alt_k})"
        )
    for k in alt_k:
        if k not in kernels:
            raise ValueError(f"kernel {k!r} (from {alt_model!r}) not in kernels dict")
    tested = tuple(sorted(set(alt_k) - set(null_k)))

    # Fit the reduced (null) model on real y.
    if len(null_k) == 0:
        red = residual_only_reml(y, X)
    else:
        red = fit_multi_reml(
            y, X, {k: kernels[k] for k in null_k},
            n_starts=n_starts, random_state=random_state,
        )
    sigma2_true = {"e": float(red.sigma2["e"])}
    for k in alt_k:
        sigma2_true[k] = float(red.sigma2[k]) if k in null_k else 0.0

    # Flag a degenerate null: if the reduced fit pushed σ²_e to its boundary,
    # the simulation is a near-zero-residual null and the Stram-Lee analytic
    # mixture is only approximate.
    red_bc = sorted(getattr(red, "boundary_components", []) or [])
    residual_pve = float(red.pve.get("e", float("nan")))
    residual_boundary = "e" in red_bc
    if residual_boundary:
        warnings.warn(
            f"scenario {name!r}: reduced model {null_model!r} fit has residual "
            f"σ²_e at its boundary (boundary_components={red_bc}, "
            f"residual_pve={residual_pve:.3g}); the null simulation is a "
            "near-zero-residual / residual-boundary null. Empirical type-I from "
            "this scenario stays valid (the same pipeline is simulated and "
            "refit), but the Stram-Lee analytic mixture is only approximate here.",
            RuntimeWarning, stacklevel=2,
        )

    return NullSimulationScenario(
        name=name,
        null_model=null_model,
        alt_model=alt_model,
        model_specs={null_model: null_k, alt_model: alt_k},
        sigma2_true=sigma2_true,
        beta=np.asarray(red.beta, dtype=np.float64).copy(),
        df_added=len(tested),
        tested_components=tested,
        description=description,
        metadata={
            "reduced_fit_log_lik": float(red.log_lik),
            "reduced_model": null_model,
            "reduced_fit_boundary_components": red_bc,
            "reduced_fit_residual_pve": residual_pve,
            "residual_boundary": residual_boundary,
        },
    )


# Simulation


def _covariance_from_sigma2(
    sigma2: dict[str, float], kernels: dict[str, np.ndarray], n: int
) -> np.ndarray:
    """V = σ²_e I + Σ_j σ²_j K_j from explicit true variance components.

    Each K_j is PSD-projected (mirrors fit_multi_reml).
    """
    V = float(sigma2["e"]) * np.eye(n, dtype=np.float64)
    for name, K in kernels.items():
        s = float(sigma2.get(name, 0.0))
        if s != 0.0:
            V = V + s * _psd_project(K)
    return 0.5 * (V + V.T)


def summarize_type1(
    replicate_table: pd.DataFrame,
    alphas: tuple[float, ...],
    methods: tuple[str, ...],
) -> pd.DataFrame:
    """Empirical type-I rate + Wilson CI for each (method, α).

    The lower-tail rule ``P(p ≤ α)`` is exact even though the boundary
    p-values are not continuous-Uniform (atom at p=1).
    """
    rows: list[dict] = []
    for method in methods:
        if method not in replicate_table.columns:
            continue
        p = replicate_table[method].to_numpy(dtype=np.float64)
        finite = np.isfinite(p)
        p_used = p[finite]
        n_used = int(p_used.size)
        for alpha in alphas:
            n_rej = int(np.sum(p_used <= alpha))
            rate = (n_rej / n_used) if n_used else float("nan")
            lo, hi = _wilson_ci(n_rej, n_used)
            if n_used == 0:
                verdict = "no_data"
            elif lo > alpha:
                verdict = "anti-conservative"
            elif hi < alpha:
                verdict = "conservative"
            else:
                verdict = "calibrated"
            rows.append({
                "method": method,
                "alpha": alpha,
                "n_used": n_used,
                "n_reject": n_rej,
                "type1_rate": rate,
                "ci_lo": lo,
                "ci_hi": hi,
                "nominal_in_ci": bool(n_used and lo <= alpha <= hi),
                "verdict": verdict,
            })
    return pd.DataFrame(rows)


def run_null_lrt_calibration(
    X: np.ndarray,
    kernels: dict[str, np.ndarray],
    scenario: NullSimulationScenario,
    *,
    n_sim: int = 1000,
    alphas: tuple[float, ...] = (0.10, 0.05, 0.01),
    seed: int | None = None,
    fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    bootstrap_B: int | None = None,
    bootstrap_fit_kwargs: dict | None = None,
    require_converged: bool = True,
) -> NullCalibrationResult:
    """Run a null-simulation calibration for one nested LRT contrast.

    Args:
        X: fixed-effect design (n, p).
        kernels: master {name: K} dict; must contain every kernel referenced
            by scenario.model_specs.
        scenario: NullSimulationScenario (tested component(s) pinned to 0).
        n_sim: number of null replicates.
        alphas: nominal type-I levels to report.
        seed: master seed; SeedSequence spawns independent (y, fit, bootstrap)
            states per replicate, so results are n_jobs-invariant.
        fit_kwargs: passed to compare_nested_reml (e.g. n_starts); must not set
            random_state (per-replicate controlled).
        n_jobs: 1 = serial; >1 = joblib parallel.
        bootstrap_B: if set, also run a parametric bootstrap LRT per replicate
            (expensive: n_sim × B refits) and calibrate bootstrap_p.
        bootstrap_fit_kwargs: passed to parametric_bootstrap_lrt.
        require_converged: forwarded to parametric_bootstrap_lrt.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    n = X.shape[0]
    if n_sim < 1:
        raise ValueError(f"n_sim must be >= 1, got {n_sim}")

    alt_k = scenario.model_specs[scenario.alt_model]
    if scenario.null_model not in scenario.model_specs:
        raise ValueError(
            f"scenario {scenario.name!r} null_model {scenario.null_model!r} "
            f"not present in model_specs"
        )
    for k in alt_k:
        if k not in kernels:
            raise ValueError(f"kernels dict missing {k!r} required by scenario")
        K = kernels[k]
        if not (isinstance(K, np.ndarray) and K.shape == (n, n)):
            raise ValueError(f"kernels[{k!r}] must be ({n},{n}), got {getattr(K,'shape',None)}")
    # Tested components must be truly absent for a null scenario.
    for t in scenario.tested_components:
        if float(scenario.sigma2_true.get(t, 0.0)) != 0.0:
            raise ValueError(
                f"scenario {scenario.name!r} is not a null scenario: tested "
                f"component {t!r} has sigma2_true={scenario.sigma2_true[t]} != 0"
            )
    for key in ("e", *alt_k):
        if float(scenario.sigma2_true.get(key, 0.0)) < 0.0:
            raise ValueError(f"sigma2_true[{key!r}] is negative")

    fk = dict(fit_kwargs or {})
    if "random_state" in fk:
        raise ValueError("fit_kwargs must not set random_state (per-replicate controlled)")

    sub_kernels = {k: np.ascontiguousarray(kernels[k], dtype=np.float64) for k in alt_k}
    beta = np.asarray(scenario.beta, dtype=np.float64)
    if beta.shape != (X.shape[1],):
        raise ValueError(f"scenario.beta shape {beta.shape} != (p={X.shape[1]},)")

    # True covariance + Cholesky, reused across all replicates.
    V_true = _covariance_from_sigma2(scenario.sigma2_true, sub_kernels, n)
    mean_diag = float(np.trace(V_true) / n)
    L_V, jitter_used = _cholesky_with_jitter(V_true, mean_diag)
    if jitter_used > 0:
        warnings.warn(
            f"calibration scenario {scenario.name!r}: V_true needed Cholesky "
            f"jitter {jitter_used:.3g}·mean_diag (near-singular true covariance, "
            "e.g. a near-zero-residual null).",
            RuntimeWarning, stacklevel=2,
        )
    Xb = X @ beta

    # Independent per-replicate seeds (y draw, refit RNG, bootstrap RNG).
    ss = np.random.SeedSequence(seed)
    seed_rows: list[tuple[int, int, int]] = []
    for child in ss.spawn(n_sim):
        st = child.generate_state(3, dtype=np.uint32)
        seed_rows.append((int(st[0]), int(st[1]), int(st[2])))

    model_specs = scenario.model_specs
    null_model, alt_model = scenario.null_model, scenario.alt_model

    def _one(sim_id: int, seed_y: int, seed_fit: int, seed_boot: int) -> dict:
        try:
            rng = np.random.default_rng(seed_y)
            y_b = Xb + L_V @ rng.standard_normal(n)
            cmp = compare_nested_reml(
                y_b, X, sub_kernels, model_specs=model_specs,
                fit_kwargs={**fk, "random_state": seed_fit},
            )
            blrt = boundary_lrt(cmp, null_model, alt_model)
            nbc = sorted(getattr(cmp.fits[null_model], "boundary_components", []) or [])
            abc = sorted(getattr(cmp.fits[alt_model], "boundary_components", []) or [])
            row = {
                "sim_id": sim_id,
                "success": True,
                "lrt": float(blrt.statistic),
                "lrt_raw": float(blrt.statistic_raw),
                "df_added": int(blrt.df_added),
                "p_naive": float(blrt.p_naive),
                "p_mixture": float(blrt.p_mixture),
                "clipped": bool(blrt.clipped),
                "both_converged": bool(blrt.both_converged),
                "null_boundary_components": ",".join(nbc),
                "alt_boundary_components": ",".join(abc),
                "e_at_boundary": ("e" in nbc) or ("e" in abc),
                "error": "",
            }
            if bootstrap_B is not None:
                bres = parametric_bootstrap_lrt(
                    cmp, null_model, alt_model,
                    y=y_b, X=X, kernels=sub_kernels,
                    B=bootstrap_B, seed=seed_boot, n_jobs=1,
                    bootstrap_fit_kwargs=bootstrap_fit_kwargs,
                    require_converged=require_converged,
                    on_failure="warn",
                )
                row["bootstrap_p"] = float(bres.bootstrap_p)
            return row
        except Exception as e:  # noqa: BLE001 one bad replicate must not kill the run
            row = {
                "sim_id": sim_id, "success": False,
                "lrt": float("nan"), "lrt_raw": float("nan"), "df_added": scenario.df_added,
                "p_naive": float("nan"), "p_mixture": float("nan"),
                "clipped": False, "both_converged": False,
                "null_boundary_components": "", "alt_boundary_components": "",
                "e_at_boundary": False, "error": str(e),
            }
            if bootstrap_B is not None:
                row["bootstrap_p"] = float("nan")
            return row

    tasks = [(i, sy, sf, sb) for i, (sy, sf, sb) in enumerate(seed_rows)]
    if n_jobs == 1:
        results = [_one(*t) for t in tasks]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError as e:
            raise RuntimeError(f"joblib needed for n_jobs={n_jobs}: {e}") from e
        results = Parallel(n_jobs=n_jobs)(delayed(_one)(*t) for t in tasks)

    replicate_table = pd.DataFrame(results).sort_values("sim_id").reset_index(drop=True)
    success_mask = replicate_table["success"].to_numpy(dtype=bool)
    n_success = int(success_mask.sum())
    if n_success < n_sim:
        warnings.warn(
            f"calibration scenario {scenario.name!r}: {n_sim - n_success}/{n_sim} "
            "replicates failed (see replicate_table['error']).",
            RuntimeWarning, stacklevel=2,
        )

    methods = ["p_naive", "p_mixture"]
    if bootstrap_B is not None:
        methods.append("bootstrap_p")
    type1_table = summarize_type1(replicate_table, tuple(alphas), tuple(methods))
    type1_table.insert(0, "scenario", scenario.name)

    # chi-bar tail inflation (replaces λ_GC, which is ill-defined here)
    weights = _chibar_weights(scenario.df_added)
    T_ok = replicate_table.loc[success_mask, "lrt"].to_numpy(dtype=np.float64)
    tail_rows: list[dict] = []
    for q in (0.95, 0.99):
        if T_ok.size:
            emp = float(np.quantile(T_ok, q))
        else:
            emp = float("nan")
        theo = _chibar_quantile(q, weights)
        tail_rows.append({
            "scenario": scenario.name,
            "quantile": q,
            "empirical_lrt": emp,
            "theoretical_chibar": theo,
            "inflation_ratio": (emp / theo) if theo > 0 else float("nan"),
        })
    tail_inflation = pd.DataFrame(tail_rows)

    conv = replicate_table.loc[success_mask, "both_converged"].to_numpy(dtype=bool)
    clip = replicate_table.loc[success_mask, "clipped"].to_numpy(dtype=bool)
    e_bnd = replicate_table.loc[success_mask, "e_at_boundary"].to_numpy(dtype=bool)

    return NullCalibrationResult(
        scenario=scenario,
        n_sim_requested=n_sim,
        n_sim_success=n_success,
        seed=seed,
        replicate_table=replicate_table,
        type1_table=type1_table,
        tail_inflation=tail_inflation,
        df_added=scenario.df_added,
        converged_rate=float(conv.mean()) if conv.size else 0.0,
        clip_rate=float(clip.mean()) if clip.size else 0.0,
        e_boundary_rate=float(e_bnd.mean()) if e_bnd.size else 0.0,
        jitter_used=float(jitter_used),
        bootstrap_B=bootstrap_B,
    )
