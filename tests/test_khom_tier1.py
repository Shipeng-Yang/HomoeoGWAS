"""Minimal pytest for khom_tier1 scaffolding (charter §2.1 Tier 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from homoeogwas.khom_tier1 import (
    BoundaryLRT,
    SpikeInScenario,
    ablation_table,
    recall_at_k,
    self_liang_lrt,
    simulate_additive_plus_epistasis,
)

# ---------------------------------------------------------------------------
# Path F — recall@K
# ---------------------------------------------------------------------------

def test_recall_at_k_hits_anchor_in_window():
    ss = pd.DataFrame({
        "chrom": ["1A", "1A", "2B", "2B", "3D"],
        "pos":   [100_000, 200_000, 1_000_000, 5_000_000, 10_000_000],
        "p":     [1e-10, 1e-5, 1e-9, 0.5, 1e-7],
    })
    qtl = pd.DataFrame({
        "logical_chrom": ["1A", "2B", "5A"],
        "pos":           [150_000, 1_200_000, 1_000_000],
    })
    res = recall_at_k(ss, qtl, k_grid=(2, 5), window_bp=500_000)
    assert len(res) == 2
    # top-2 = (1A:100k p=1e-10, 2B:1M p=1e-9) — both hit anchors at (1A:150k, 2B:1.2M)
    assert res[0].k == 2 and res[0].n_known == 3 and res[0].n_hit == 2
    # top-5 still no hit on 5A (no SNPs on chr5A in sumstats)
    assert res[1].k == 5 and res[1].n_hit == 2


def test_recall_at_k_empty_sumstats_chromosome():
    ss = pd.DataFrame({"chrom": ["1A"], "pos": [10], "p": [1e-3]})
    qtl = pd.DataFrame({"logical_chrom": ["7D"], "pos": [1000]})
    res = recall_at_k(ss, qtl, k_grid=(1,), window_bp=1000)
    assert res[0].n_hit == 0


def test_ablation_table_delta_correct():
    ss_without = pd.DataFrame({
        "chrom": ["1A", "1A", "2B"],
        "pos":   [100, 200, 300],
        "p":     [1e-9, 1e-6, 1e-3],  # top-1 = 1A:100
    })
    ss_with = pd.DataFrame({
        "chrom": ["1A", "1A", "2B"],
        "pos":   [100, 200, 300],
        "p":     [1e-3, 1e-9, 1e-6],  # top-1 = 1A:200 (different)
    })
    qtl = pd.DataFrame({
        "logical_chrom": ["1A", "2B"],
        "pos":           [205, 305],
    })
    tab = ablation_table(ss_without, ss_with, qtl,
                         k_grid=(1,), window_bp=50)
    assert "delta_recall" in tab.columns
    # without@1 hits no anchor (1A:100 not within 50bp of 205);
    # with@1 hits 1A:205 (within 5bp of 200) → delta_recall = 0.5
    row = tab.iloc[0]
    assert row["recall_without_khom"] == 0.0
    assert row["recall_with_khom"] == 0.5
    assert row["delta_recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Path A — Self-Liang mixture LRT
# ---------------------------------------------------------------------------

def test_self_liang_boundary_returns_p_one():
    res = self_liang_lrt(ll_null=-100.0, ll_alt=-100.0)
    assert isinstance(res, BoundaryLRT)
    assert res.boundary is True
    assert res.lr_stat == 0.0
    assert res.p_value == 1.0


def test_self_liang_strong_signal_rejects():
    # 2(ll_alt - ll_null) = 20 → χ²₁ p ~ 7.7e-6, mixture p ~ 3.8e-6
    res = self_liang_lrt(ll_null=-100.0, ll_alt=-90.0)
    assert res.boundary is False
    assert res.lr_stat == pytest.approx(20.0)
    assert res.p_value < 1e-5


def test_self_liang_halves_naive_chisq_p():
    """Mixture p should be exactly half the naive χ²₁ p (for lr > 0)."""
    from scipy.stats import chi2
    lr = 5.0
    res = self_liang_lrt(ll_null=0.0, ll_alt=lr / 2.0)
    assert res.p_value == pytest.approx(0.5 * float(chi2.sf(lr, df=1)))


def test_self_liang_two_vc_raises():
    with pytest.raises(NotImplementedError):
        self_liang_lrt(ll_null=-100, ll_alt=-95, n_vc_at_boundary=2)


# ---------------------------------------------------------------------------
# Path H' — spike-in scenario validation + smoke
# ---------------------------------------------------------------------------

def test_scenario_rejects_negative_h2():
    with pytest.raises(ValueError, match="h2_add"):
        SpikeInScenario(n=100, h2_add=-0.1, h2_epi=0.2)


def test_scenario_rejects_total_h2_ge_1():
    with pytest.raises(ValueError, match="h2_add \\+ h2_epi"):
        SpikeInScenario(n=100, h2_add=0.6, h2_epi=0.5)


def test_simulate_additive_plus_epistasis_shape_and_centering():
    rng = np.random.default_rng(0)
    n = 30
    A = rng.standard_normal((n, 10))
    K_pool = (A @ A.T) / 10
    B = rng.standard_normal((n, 10))
    K_hom = (B @ B.T) / 10
    sc = SpikeInScenario(n=n, h2_add=0.3, h2_epi=0.2, seed=1)
    # Random kernels are not trace-normalized; pass the bypass flag
    y = simulate_additive_plus_epistasis(K_pool, K_hom, sc,
                                          skip_normalization_check=True)
    assert y.shape == (n,)
    assert abs(y.mean()) < 1e-10  # centered


def test_simulate_rejects_unnormalized_by_default():
    """Unnormalized kernels raise without the bypass flag."""
    n = 30
    # Trace will be ≫ n → fails normalization check
    K_pool = np.eye(n) * 10.0
    K_hom = np.eye(n) * 10.0
    sc = SpikeInScenario(n=n, h2_add=0.3, h2_epi=0.2, seed=1)
    with pytest.raises(ValueError, match="trace.*K.*expected"):
        simulate_additive_plus_epistasis(K_pool, K_hom, sc)


def test_placeholder_spike_in_grid_removed():
    """The fabricated spike-in power grid (PLACEHOLDER REML) was removed (2026-06-02).
    Guard that the fake public symbols are gone so no caller can resurrect fake numbers;
    the simulator (simulate_additive_plus_epistasis) stays. Real power = Phase 7
    scripts/phase7/d2_khom_recoverability.py + diagnostics.boundary_lrt."""
    import homoeogwas.khom_tier1 as kt

    assert not hasattr(kt, "spike_in_power_grid")
    assert not hasattr(kt, "_run_one_replicate")
    assert not hasattr(kt, "SpikeInResult")
    assert "spike_in_power_grid" not in kt.__all__
    assert "SpikeInResult" not in kt.__all__
    # the real simulator is retained
    assert hasattr(kt, "simulate_additive_plus_epistasis")
