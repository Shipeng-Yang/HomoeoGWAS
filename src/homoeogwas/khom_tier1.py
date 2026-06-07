"""K_hom Tier 1 defensive analyses.

Three defensive paths that characterize K_hom honestly but cannot promote it
back to a main paper claim (only a Tier 2 synteny-aware local K_hom or an
epistasis-rich trait re-test can do that).

Path F (ablation): recall@K of known QTL anchors under a per-SNP scan with
vs without K_hom in the kinship stack. Quantifies the "K_hom is power-useful
but not variance-identifiable" reading on real data.

Path A (boundary-aware variance test): Self-Liang (1987) mixture chi2 for a
single variance component at the boundary, replacing the naive REML sigma2_h
estimate with a likelihood-ratio test that respects the boundary. In the
per-fold GBLUP fits, sigma2_h boundary estimates occurred in only ~20-68% of
folds (not uniformly) and non-zero estimates were sometimes comparable to the
subgenome components in wheat, so K_hom is best read as unstable / low-support
rather than uniformly zero.

Path H' (spike-in simulation): ships only the simulator
(``simulate_additive_plus_epistasis``) injecting controlled digenic epistasis
between two subgenomes. The earlier in-module power grid returned a placeholder
(fabricated, never real REML) and was removed. The real recoverability/power
analysis uses genuine REML in ``scripts/phase7/d2_khom_recoverability.py``
(weak power ~0.17-0.21 at PVE_hom=0.20, ~8-10% upward null bias). For an
in-package boundary-corrected LRT use
``homoeogwas.diagnostics.compare_nested_reml`` + ``boundary_lrt``.

References:
* Self & Liang 1987, JASA 82:605 — asymptotic LRT distribution for parameters
  at a boundary; mixture 0.5 chi2_0 + 0.5 chi2_1 for one variance component.
* Crainiceanu & Ruppert 2004, JRSS-B 66:165 — exact finite-sample reference
  for spline/RE variance components (we ship the Self-Liang asymptotic).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


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

    Recall is computed per anchor (a known QTL is "hit" if at least one top-k
    SNP is within window_bp on the same chrom), not per top-k SNP.

    Operates on the per-trait LOCO sumstats TSVs. Pass the same known_qtl tsv
    to compare across panels: the recall numerator is panel-specific (nearby
    top-k SNPs) but the denominator is a fixed anchor set per panel.
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
    # cast chrom to str (else silent "1" vs 1 mismatch) and pos to numeric
    ss[sumstats_chrom_col] = ss[sumstats_chrom_col].astype(str)
    ss[sumstats_pos_col] = pd.to_numeric(ss[sumstats_pos_col], errors="coerce")
    ss = ss.dropna(subset=[sumstats_pos_col])
    ss = ss.sort_values(sumstats_p_col, ascending=True).reset_index(drop=True)

    kn = known_qtl.copy()
    kn[known_chrom_col] = kn[known_chrom_col].astype(str)
    kn[known_pos_col] = pd.to_numeric(kn[known_pos_col], errors="coerce")
    kn = kn.dropna(subset=[known_pos_col]).reset_index(drop=True)
    # dedup distinct anchors on (chrom, pos)
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

    delta_recall > 0 means K_hom helped on that recall point; a consistently
    positive delta across k_grid is the "power-useful but not
    variance-identifiable" evidence.
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


@dataclass(frozen=True)
class BoundaryLRT:
    """Self-Liang (1987) mixture chi2 LRT for one variance component at zero.

    For H0: σ² = 0 vs H1: σ² > 0 with a single VC at the boundary, the LRT
    statistic 2(ll_alt - ll_null) is distributed as 0.5*δ_0 + 0.5*chi2_1 under
    H0 (point-mass at 0 plus chi2 with 1 df).

    Attributes
    ----------
    lr_stat : float
        2*(ll_alt - ll_null), clipped at 0 (negatives mean numerical noise at
        the boundary or optimization failure — see ``ll_decrease_warn``).
    p_value : float
        Mixture p = 0.5 * 1{lr_stat > 0} * P(chi2_1 > lr_stat); half the naive
        chi2_1 p-value, zero if lr_stat == 0.
    boundary : bool
        Alias of ``lr_collapsed``. True if lr_stat == 0 (REML pegged σ²_alt at
        the boundary, no evidence to reject H0). Both names kept for back-compat.
    lr_collapsed : bool
        Synonym of ``boundary``.
    ll_decrease_warn : bool
        True if ``ll_alt < ll_null`` before clipping. A properly fitted alt
        model with one extra VC should have at least the null likelihood (REML
        is monotone in alt-set), so a decrease usually signals optimization
        failure; inspect the alt fit when this is True.
    """

    lr_stat: float
    p_value: float
    boundary: bool
    lr_collapsed: bool = False     # synonym of boundary
    ll_decrease_warn: bool = False # True if ll_alt < ll_null pre-clip


def self_liang_lrt(
    ll_null: float,
    ll_alt: float,
    *,
    n_vc_at_boundary: int = 1,
    tol: float = 1e-8,
) -> BoundaryLRT:
    """Self-Liang (1987) mixture LRT for ``n_vc_at_boundary == 1``.

    For ``n_vc_at_boundary > 1`` (e.g. testing K_hom and a second optional
    kernel together) the asymptotic distribution is a more elaborate mixture
    (Stram & Lee 1994); we raise NotImplementedError rather than return a wrong
    p-value. The single-VC case is what Tier 1 needs (K_hom alone, on top of a
    fixed K_pool / per-sub stack).

    Parameters
    ----------
    ll_null, ll_alt : float
        REML log-likelihoods under null (no K_hom) and alternative (with
        K_hom), as stored on ``MultiREMLResult`` from lmm.py.
    """
    if n_vc_at_boundary != 1:
        raise NotImplementedError(
            f"Self-Liang mixture for {n_vc_at_boundary} VCs at boundary requires "
            "Stram-Lee 1994 generalization; this scaffold only ships single-VC.")

    from scipy.stats import chi2

    raw = 2.0 * (ll_alt - ll_null)
    ll_decrease_warn = raw < -tol     # surface an unexpected decrease
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


def _assert_trace_normalized(K: np.ndarray, name: str, tol: float = 0.05) -> None:
    """Hard-fail on un-normalized kernels: silent drift would make h2_add/h2_epi
    differ from the requested variance ratios."""
    n = K.shape[0]
    tr_per_n = float(np.trace(K)) / n
    if abs(tr_per_n - 1.0) > tol:
        raise ValueError(
            f"{name}: trace(K)/n = {tr_per_n:.4f}, expected ~1.0 within +/-{tol}. "
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
    """Draw one phenotype y ~ N(0, h2_add*K_pool + h2_epi*K_hom + (1-h2_add-h2_epi)*I).

    Simulator only: returns a phenotype, does not fit REML or estimate power.
    For the real recoverability/power analysis see
    ``scripts/phase7/d2_khom_recoverability.py``; for an in-package
    boundary-corrected LRT use ``homoeogwas.diagnostics.boundary_lrt``.

    Draws via eigen square-root factors of K_pool / K_hom. Both kernels must be
    trace-normalized (trace(K)/n ~ 1, tol 5%) so h2_add / h2_epi equal the
    requested variance ratios. Pass ``skip_normalization_check=True`` only for
    known-degenerate unit tests (e.g. tiny random kernels).
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
    # mean-removed phenotype is the LMM convention
    return y - y.mean()


# The former in-module spike-in power grid was removed: it returned a
# placeholder (fabricated sigma2_h / LR from a toy formula, never a real REML
# fit). The genuine recoverability/power analysis lives in
# scripts/phase7/d2_khom_recoverability.py. For an in-package boundary-corrected
# LRT, draw y with simulate_additive_plus_epistasis and test it with
# homoeogwas.diagnostics.compare_nested_reml + boundary_lrt.


__all__ = [
    "RecallAtK",
    "recall_at_k",
    "ablation_table",
    "BoundaryLRT",
    "self_liang_lrt",
    "SpikeInScenario",
    "simulate_additive_plus_epistasis",
]
