"""K_hom Tier 1 defensive analyses (charter §2.1, 2026-05-24).

This module hosts the three Tier 1 defensive paths from the K_hom revival
dual-plan. Tier 1 results are *defensive*: they characterize K_hom honestly
but **CANNOT promote K_hom back to a main paper claim** (charter §2.1
hard gate; only Tier 2 with synteny-aware local K_hom or epistasis-rich
trait re-test can do that).

Tier 1 paths
------------
* **Path F — ablation**: recall@K of paper-reported known QTL anchors under
  per-SNP scan *without* vs *with* K_hom in the kinship stack. Quantifies
  "K_hom is power-useful but not variance-identifiable" framing on real data.

* **Path A — boundary-aware variance test**: Self-Liang (1987) mixture χ²
  for a single variance component at the boundary. Replaces the naive REML
  σ²_h estimate (pegged at 0 in 4/4 Phase 5c panels) with an honest
  likelihood-ratio test that respects the boundary.

* **Path H' — spike-in simulation**: inject controlled digenic epistasis
  between two subgenomes and measure the detection rate of K_hom (σ²_h
  significantly > 0) as a function of sample size and true epistasis
  variance fraction. Shows whether Phase 5c's universal null is intrinsic
  to the K_hom test or only to the panel-trait conditions we tested.

Citation backbone
-----------------
* Self & Liang 1987, JASA 82:605 — asymptotic distribution of LRT for
  parameters at a boundary; mixture 0.5 χ²₀ + 0.5 χ²₁ for one
  variance component.
* Crainiceanu & Ruppert 2004, JRSS-B 66:165 — exact finite-sample
  reference for spline/RE variance components (future enhancement; we
  ship Self-Liang asymptotic by default).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path F — recall@K ablation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecallAtK:
    """One (k, recall) point with sample size for downstream CI calc."""

    k: int
    n_known: int
    n_hit: int

    @property
    def recall(self) -> float:
        return self.n_hit / self.n_known if self.n_known else float("nan")


def recall_at_k(
    sumstats: pd.DataFrame,
    known_qtl: pd.DataFrame,
    *,
    k_grid: Sequence[int] = (10, 50, 100, 500, 1000, 5000),
    window_bp: int = 500_000,
    sumstats_chrom_col: str = "chrom",
    sumstats_pos_col: str = "pos",
    sumstats_p_col: str = "p",
    known_chrom_col: str = "logical_chrom",
    known_pos_col: str = "pos",
) -> list[RecallAtK]:
    """For each k in k_grid, count how many distinct known QTL anchors are
    within ``window_bp`` of *any* SNP in the top-k of ``sumstats`` (sorted
    by p). Returns one ``RecallAtK`` per k.

    The recall is computed per *anchor* (a known QTL is "hit" if at least
    one top-k SNP is within window_bp on the same chrom) — not per top-k
    SNP. This matches paper convention.

    Notes
    -----
    Designed to operate on the ``sumstats_<trait>_loco.tsv`` files already
    written by Phase 3 M3.1 (`scripts/phase3/run_m3_1_*.py`). Pass the
    *same* known_qtl tsv to compare across panels — the recall numerator
    is panel-specific (set of nearby top-k SNPs) but the denominator is
    a fixed anchor set per panel.
    """
    if not k_grid:
        raise ValueError("k_grid is empty")
    if sumstats_p_col not in sumstats.columns:
        raise KeyError(f"sumstats lacks p column {sumstats_p_col!r}")
    if known_chrom_col not in known_qtl.columns or known_pos_col not in known_qtl.columns:
        raise KeyError(
            f"known_qtl lacks {known_chrom_col!r}/{known_pos_col!r} columns")

    ss = sumstats.copy()
    ss = ss.dropna(subset=[sumstats_p_col])
    # Codex Q2 polish: cast chrom to str (silent mismatch on "1" vs 1) +
    # ensure pos is numeric and not NaN.
    ss[sumstats_chrom_col] = ss[sumstats_chrom_col].astype(str)
    ss[sumstats_pos_col] = pd.to_numeric(ss[sumstats_pos_col], errors="coerce")
    ss = ss.dropna(subset=[sumstats_pos_col])
    ss = ss.sort_values(sumstats_p_col, ascending=True).reset_index(drop=True)

    kn = known_qtl.copy()
    kn[known_chrom_col] = kn[known_chrom_col].astype(str)
    kn[known_pos_col] = pd.to_numeric(kn[known_pos_col], errors="coerce")
    kn = kn.dropna(subset=[known_pos_col]).reset_index(drop=True)
    # Codex Q2 polish: dedup distinct anchors on (chrom, pos) — docstring
    # claimed "distinct" but code did not deduplicate.
    kn = kn.drop_duplicates(subset=[known_chrom_col, known_pos_col])\
           .reset_index(drop=True)
    n_known = len(kn)

    results: list[RecallAtK] = []
    for k in k_grid:
        topk = ss.head(k)
        topk_by_chrom = {c: g for c, g in topk.groupby(sumstats_chrom_col)}
        n_hit = 0
        for _, row in kn.iterrows():
            chrom = row[known_chrom_col]
            qpos = float(row[known_pos_col])
            block = topk_by_chrom.get(chrom)
            if block is None or len(block) == 0:
                continue
            distance = np.abs(block[sumstats_pos_col].astype(float) - qpos)
            if (distance <= window_bp).any():
                n_hit += 1
        results.append(RecallAtK(k=int(k), n_known=n_known, n_hit=int(n_hit)))
    return results


def ablation_table(
    sumstats_without_khom: pd.DataFrame,
    sumstats_with_khom: pd.DataFrame,
    known_qtl: pd.DataFrame,
    *,
    k_grid: Sequence[int] = (10, 50, 100, 500, 1000, 5000),
    window_bp: int = 500_000,
    **recall_kwargs,
) -> pd.DataFrame:
    """Side-by-side recall@K table for the K_hom ablation.

    Output columns:
        k, n_known, recall_without_khom, recall_with_khom, delta_recall

    delta_recall > 0 means K_hom helped on that recall point. A
    consistently positive delta across k_grid is the "power-useful but
    not variance-identifiable" evidence (charter §2.1 Tier 1 F).
    """
    a = recall_at_k(sumstats_without_khom, known_qtl,
                    k_grid=k_grid, window_bp=window_bp, **recall_kwargs)
    b = recall_at_k(sumstats_with_khom, known_qtl,
                    k_grid=k_grid, window_bp=window_bp, **recall_kwargs)
    rows = []
    for ai, bi in zip(a, b, strict=True):
        assert ai.k == bi.k and ai.n_known == bi.n_known
        rows.append({
            "k": ai.k,
            "n_known": ai.n_known,
            "recall_without_khom": ai.recall,
            "recall_with_khom": bi.recall,
            "delta_recall": bi.recall - ai.recall,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Path A — Self-Liang boundary-aware LRT
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryLRT:
    """Self-Liang (1987) mixture χ² LRT for one variance component at zero.

    For testing H0: σ² = 0 vs H1: σ² > 0 with a single VC at the boundary,
    the LRT statistic 2(ll_alt - ll_null) is distributed as
    0.5 · δ_0 + 0.5 · χ²₁ under H0 (mixture point-mass at 0 + chi-sq 1 df).

    Attributes
    ----------
    lr_stat : float
        2 (ll_alt - ll_null); clipped at 0 (negative values mean numerical
        noise around the boundary OR optimization failure — see
        ``ll_decrease_warn``).
    p_value : float
        Mixture p = 0.5 · 1{lr_stat > 0} · P(χ²₁ > lr_stat)
        i.e. half the naive χ²₁ p-value, zero if lr_stat == 0.
    boundary : bool
        Backward-compat alias of ``lr_collapsed``. True if lr_stat == 0
        (REML pegged σ²_alt at the boundary → no evidence to reject H0).
        Codex Q1 polish: name was historically ambiguous; both names retained.
    lr_collapsed : bool
        Synonym of ``boundary`` (Codex Q1 polish: more explicit name —
        "LR ratio collapsed to zero" rather than just "test is a boundary test").
    ll_decrease_warn : bool
        Codex Q1 polish: True if ``ll_alt < ll_null`` BEFORE clipping. A
        properly fitted alt model with one extra VC ought to have at least
        the same likelihood as the null (REML is monotone in alt-set), so
        a decrease usually signals optimization failure (e.g. REML hit a
        local minimum). Inspect the alt fit when this is True.
    """

    lr_stat: float
    p_value: float
    boundary: bool
    lr_collapsed: bool = False     # Codex Q1: synonym of boundary
    ll_decrease_warn: bool = False # Codex Q1: True if ll_alt < ll_null pre-clip


def self_liang_lrt(
    ll_null: float,
    ll_alt: float,
    *,
    n_vc_at_boundary: int = 1,
    tol: float = 1e-8,
) -> BoundaryLRT:
    """Self-Liang (1987) mixture LRT for ``n_vc_at_boundary == 1``.

    For ``n_vc_at_boundary > 1`` (e.g. testing K_hom *and* a second
    optional kernel together), the asymptotic distribution is a more
    elaborate mixture (Stram & Lee 1994); we raise NotImplementedError
    rather than silently return a wrong p-value. The single-VC case is
    what Tier 1 needs (K_hom alone, in addition to fixed K_pool / per-sub).

    Parameters
    ----------
    ll_null, ll_alt : float
        REML log-likelihoods under null (no K_hom) and alternative
        (K_pool + K_hom or per-sub + K_hom). Pass the values from
        ``fit_multi_reml(..., kernels=[...])`` runs — they are stored on
        ``MultiREMLResult`` from lmm.py.
    """
    if n_vc_at_boundary != 1:
        raise NotImplementedError(
            f"Self-Liang mixture for {n_vc_at_boundary} VCs at boundary requires "
            "Stram-Lee 1994 generalization; this scaffold only ships single-VC.")

    from scipy.stats import chi2

    raw = 2.0 * (ll_alt - ll_null)
    ll_decrease_warn = raw < -tol     # Codex Q1 polish: surface unexpected decrease
    lr = max(0.0, raw)
    if lr <= tol:
        return BoundaryLRT(
            lr_stat=0.0, p_value=1.0,
            boundary=True, lr_collapsed=True,
            ll_decrease_warn=ll_decrease_warn,
        )
    p_naive = float(chi2.sf(lr, df=1))
    p_mixture = 0.5 * p_naive
    return BoundaryLRT(
        lr_stat=float(lr), p_value=p_mixture,
        boundary=False, lr_collapsed=False,
        ll_decrease_warn=ll_decrease_warn,
    )


# ---------------------------------------------------------------------------
# Path H' — spike-in epistasis detection power
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpikeInScenario:
    """One spike-in cell: (n, h2_add, h2_epi)."""

    n: int
    h2_add: float
    h2_epi: float
    seed: int = 0

    def __post_init__(self):
        if not 0.0 <= self.h2_add < 1.0:
            raise ValueError(f"h2_add out of [0,1): {self.h2_add}")
        if not 0.0 <= self.h2_epi < 1.0:
            raise ValueError(f"h2_epi out of [0,1): {self.h2_epi}")
        if self.h2_add + self.h2_epi >= 1.0:
            raise ValueError(
                f"h2_add + h2_epi must be < 1: got {self.h2_add + self.h2_epi}")


@dataclass
class SpikeInResult:
    """Outcome of a single spike-in REML fit.

    Use ``.detected`` to summarize Tier 1 H' power: fraction of replicates
    where the boundary-aware LRT rejects H0: σ²_h = 0 at the given alpha.

    The ``backend`` field MUST be set to "PLACEHOLDER" while
    ``_run_one_replicate`` returns the placeholder REML output
    (see module docstring TODO). Production wire-up of ``fit_multi_reml``
    will switch it to "reml". Downstream code should refuse to publish
    PLACEHOLDER rows without an explicit override.
    """

    scenario: SpikeInScenario
    sigma2_h_est: float
    lrt_p: float
    lrt_lr_stat: float
    backend: Literal["PLACEHOLDER", "reml"] = "PLACEHOLDER"
    paper_value: bool = False     # set True only when backend == "reml"
    detected: bool = field(init=False)

    def __post_init__(self):
        self.detected = (self.lrt_p < 0.05) and (self.sigma2_h_est > 0)
        if self.backend == "reml" and not self.paper_value:
            # gentle correction; production should set both explicitly
            self.paper_value = True


def _assert_trace_normalized(K: np.ndarray, name: str, tol: float = 0.05) -> None:
    """Codex review Q3 fix: silent normalization drift would make h2_add/h2_epi
    misleadingly different from the requested variance ratios. Hard-fail rather
    than let a TSV ship with the wrong h2 interpretation."""
    n = K.shape[0]
    tr_per_n = float(np.trace(K)) / n
    if abs(tr_per_n - 1.0) > tol:
        raise ValueError(
            f"{name}: trace(K)/n = {tr_per_n:.4f}, expected ≈ 1.0 within ±{tol}. "
            "Spike-in scenarios assume trace-normalized kernels (use "
            "homoeogwas.kernel.normalize_kernel(K, mode='trace') first), "
            "otherwise h2_add/h2_epi do NOT equal the requested variance ratios."
        )


def simulate_additive_plus_epistasis(
    K_pool: np.ndarray,
    K_hom: np.ndarray,
    scenario: SpikeInScenario,
    *,
    skip_normalization_check: bool = False,
) -> np.ndarray:
    """Draw a single phenotype y ~ N(0, h2_add·K_pool + h2_epi·K_hom + (1-h2_add-h2_epi)·I).

    Uses Cholesky/eigendecomposition factors of K_pool / K_hom to draw
    from the GRM-induced random effect distributions. **Both kernels must be
    trace-normalized** (trace(K)/n ≈ 1, tol 5%) so that h2_add / h2_epi
    equal the requested variance ratios — see ``_assert_trace_normalized``.

    Pass ``skip_normalization_check=True`` only for known-degenerate unit
    tests (e.g. tiny random kernels used to exercise plumbing).
    """
    from .sim import kernel_factor

    n = K_pool.shape[0]
    if K_hom.shape != (n, n):
        raise ValueError(f"K_pool {K_pool.shape} and K_hom {K_hom.shape} shape mismatch")
    if not skip_normalization_check:
        _assert_trace_normalized(K_pool, "K_pool")
        _assert_trace_normalized(K_hom, "K_hom")
    rng = np.random.default_rng(scenario.seed)
    L_pool = kernel_factor(K_pool)
    L_hom = kernel_factor(K_hom)
    u_add = L_pool @ rng.standard_normal(L_pool.shape[1])
    u_epi = L_hom @ rng.standard_normal(L_hom.shape[1])
    eps = rng.standard_normal(n)
    h2_e = 1.0 - scenario.h2_add - scenario.h2_epi
    y = (np.sqrt(scenario.h2_add) * u_add
         + np.sqrt(scenario.h2_epi) * u_epi
         + np.sqrt(h2_e) * eps)
    # Center (mean-removed phenotype is the LMM convention).
    return y - y.mean()


def spike_in_power_grid(
    K_pool: np.ndarray,
    K_hom: np.ndarray,
    n_grid: Sequence[int],
    h2_epi_grid: Sequence[float],
    *,
    h2_add: float = 0.30,
    n_replicates: int = 50,
    base_seed: int = 2026,
    progress: bool = False,
    skip_normalization_check: bool = False,
) -> pd.DataFrame:
    """For each (n, h2_epi), simulate ``n_replicates`` phenotypes and
    measure the fraction where Self-Liang LRT for K_hom rejects H0.

    Output columns:
        n, h2_add, h2_epi, n_replicates, n_detected, power, mean_sigma2_h, mean_p

    Cost note
    ---------
    Each replicate requires two ``fit_multi_reml`` calls (null + alt)
    plus one ``kernel_factor`` on K_hom. For 50 replicates × 5 n_grid ×
    4 h2_epi_grid that is 1000 REML fits. Default ``n_replicates=50``
    is enough for a first-pass power curve; bump to 200 for paper.

    This function imports ``fit_multi_reml`` lazily so the module remains
    importable for pure helper testing without the full LMM dependency.
    """
    from .lmm import fit_multi_reml  # noqa: F401 — TODO: wire after impl audit

    rows = []
    n_max = K_pool.shape[0]
    iterator = []
    for n in n_grid:
        if n > n_max:
            raise ValueError(f"n={n} exceeds K_pool n={n_max}")
        for h2_epi in h2_epi_grid:
            iterator.append((int(n), float(h2_epi)))

    for n, h2_epi in iterator:
        if progress:
            print(f"[spike] n={n} h2_epi={h2_epi}", flush=True)
        scenarios = [
            SpikeInScenario(n=n, h2_add=h2_add, h2_epi=h2_epi,
                            seed=base_seed + 1000 * n + int(h2_epi * 1000) + r)
            for r in range(n_replicates)
        ]
        results = [_run_one_replicate(K_pool, K_hom, sc,
                                       skip_normalization_check=skip_normalization_check)
                   for sc in scenarios]
        det = sum(r.detected for r in results)
        # Codex Q4 fix: surface backend + paper_value at the row level so
        # downstream readers cannot mistake PLACEHOLDER rows for paper-grade.
        backends = {r.backend for r in results}
        paper_value = all(r.paper_value for r in results)
        rows.append({
            "n": n,
            "h2_add": h2_add,
            "h2_epi": h2_epi,
            "n_replicates": n_replicates,
            "n_detected": det,
            "power": det / n_replicates,
            "mean_sigma2_h": float(np.mean([r.sigma2_h_est for r in results])),
            "mean_p": float(np.mean([r.lrt_p for r in results])),
            "backend": "+".join(sorted(backends)),
            "paper_value": paper_value,
        })
    return pd.DataFrame(rows)


def _run_one_replicate(
    K_pool: np.ndarray,
    K_hom: np.ndarray,
    scenario: SpikeInScenario,
    *,
    skip_normalization_check: bool = False,
) -> SpikeInResult:
    """One spike-in replicate: simulate y, fit null & alt, run LRT.

    TODO(production): wire the actual ``fit_multi_reml`` call when ready.
    For now we return a placeholder consistent with the API. This stub
    lets driver scripts import-and-run a smoke pass without doing REML
    on 50 replicates × 20 cells × 1000s (CPU-heavy, scaffolding only).

    The returned ``SpikeInResult.backend`` is "PLACEHOLDER" and
    ``paper_value`` is False — downstream readers and the spike_in_power_grid
    aggregator surface this in the output TSV so the rows can never be
    silently mistaken for paper-grade results (Codex review Q4 fix).
    """
    # Subset kernels to first n samples
    n = scenario.n
    K_p = K_pool[:n, :n]
    K_h = K_hom[:n, :n]
    # Call retained for its scenario/normalization validation side-effects;
    # the placeholder REML below does not consume the simulated phenotype.
    simulate_additive_plus_epistasis(K_p, K_h, scenario,
                                      skip_normalization_check=skip_normalization_check)
    # PLACEHOLDER REML output — replace with fit_multi_reml in production
    # Self-Liang p drawn from a chi-squared mixture under H0 + simple
    # signal-leak model proportional to h2_epi for sanity in tests.
    rng = np.random.default_rng(scenario.seed + 999)
    sigma2_h = max(0.0, scenario.h2_epi + 0.05 * rng.standard_normal())
    lr_stat = max(0.0, 4.0 * (scenario.h2_epi * np.sqrt(n / 500.0)) ** 2
                  + rng.chisquare(1) * 0.5)
    from scipy.stats import chi2
    p = 0.5 * float(chi2.sf(lr_stat, df=1)) if lr_stat > 0 else 1.0
    return SpikeInResult(
        scenario=scenario, sigma2_h_est=float(sigma2_h),
        lrt_p=float(p), lrt_lr_stat=float(lr_stat),
        backend="PLACEHOLDER", paper_value=False,
    )


__all__ = [
    "RecallAtK",
    "recall_at_k",
    "ablation_table",
    "BoundaryLRT",
    "self_liang_lrt",
    "SpikeInScenario",
    "SpikeInResult",
    "simulate_additive_plus_epistasis",
    "spike_in_power_grid",
]
