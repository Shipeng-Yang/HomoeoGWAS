"""Unit tests for sim.py — Phase 2 M2.6 simulation benchmark utilities.

All tests use toy arrays — no Horvath data, no GEMMA/regenie binaries.
"""
from __future__ import annotations

import numpy as np
import pytest

from homoeogwas import (
    normalize_kernel,  # noqa: E402
    sim,  # noqa: E402
)

# ---------------------------------------------------------------------
# toy fixtures
# ---------------------------------------------------------------------

def _toy_kernel(n: int, m: int, seed: int) -> np.ndarray:
    """A trace-normalized PSD GRM-like kernel from random standardized geno."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, m))
    Z = (Z - Z.mean(0)) / Z.std(0)
    K = Z @ Z.T / m
    return normalize_kernel(K, "trace")


def _toy_dosage(n: int, m: int, seed: int, maf: float = 0.3) -> np.ndarray:
    """Random {0,1,2} hard-call dosage."""
    rng = np.random.default_rng(seed)
    return rng.binomial(2, maf, size=(n, m)).astype(np.float64)


# ---------------------------------------------------------------------
# SimArm
# ---------------------------------------------------------------------

def test_default_arms_heritability():
    arms = sim.default_arms()
    assert set(arms) == {"S1_C_dominant", "S2_A_dominant", "S3_pooled",
                         "S4_balanced", "S5_null", "S6_hadamard"}
    for arm in arms.values():
        assert -1e-9 <= arm.h_e <= 1.0 + 1e-9
    # S1: qtl 0.18+0.18 + poly 0.09+0.33 -> h_e 0.22
    assert abs(arms["S1_C_dominant"].h_e - 0.22) < 1e-9
    assert arms["S5_null"].h_qtl_A == 0.0 and arms["S5_null"].h_qtl_C == 0.0


def test_simarm_validation():
    with pytest.raises(ValueError):
        sim.SimArm("bad", "stratified", h_qtl_A=0.7, h_qtl_C=0.7)   # sum > 1
    with pytest.raises(ValueError):
        sim.SimArm("bad", "weird", h_qtl_A=0.1, h_qtl_C=0.1)        # bad kind
    with pytest.raises(ValueError):
        sim.SimArm("bad", "null", h_qtl_A=0.1, h_qtl_C=0.0)         # null w/ qtl
    with pytest.raises(ValueError):
        sim.SimArm("bad", "pooled", h_qtl_A=0.04, h_qtl_C=0.04,
                   h_poly_A=0.1)                                    # pooled w/ poly_A
    with pytest.raises(ValueError):
        sim.SimArm("bad", "stratified", h_qtl_A=0.04, h_qtl_C=0.04,
                   h_poly_hom=0.1)                                  # hom on non-hadamard


# ---------------------------------------------------------------------
# standardize_dosage
# ---------------------------------------------------------------------

def test_standardize_dosage_unit_variance():
    G = _toy_dosage(150, 40, seed=1)
    Z, keep, maf, cr = sim.standardize_dosage(G, maf_min=0.05)
    assert Z.shape == G.shape
    # kept columns are unit variance, zero mean
    kept = np.where(keep)[0]
    assert kept.size > 30
    np.testing.assert_allclose(Z[:, kept].mean(0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Z[:, kept].std(0), 1.0, atol=1e-9)
    assert np.all((maf[keep] >= 0.05) & (maf[keep] <= 0.5))
    assert np.all(cr == 1.0)


def test_standardize_dosage_monomorphic_and_missing():
    G = _toy_dosage(80, 6, seed=2)
    G[:, 0] = 1.0                       # monomorphic -> dropped, all-zero column
    G[:5, 1] = np.nan                   # missing -> imputed
    Z, keep, maf, cr = sim.standardize_dosage(G, maf_min=0.01,
                                              call_rate_min=0.0)
    assert not keep[0]
    np.testing.assert_allclose(Z[:, 0], 0.0)
    assert np.all(np.isfinite(Z))       # imputation removed NaN
    assert cr[1] == pytest.approx(75 / 80)


def test_standardize_dosage_call_rate_filter():
    G = _toy_dosage(100, 4, seed=3)
    G[:60, 0] = np.nan                  # call rate 0.4
    _Z, keep, _maf, cr = sim.standardize_dosage(G, maf_min=0.01,
                                                call_rate_min=0.9)
    assert cr[0] == pytest.approx(0.4)
    assert not keep[0]


# ---------------------------------------------------------------------
# select_causal_snps
# ---------------------------------------------------------------------

def test_select_causal_snps_count_and_separation():
    rng = np.random.default_rng(10)
    m = 200
    chrom = np.array(["1"] * 100 + ["2"] * 100, dtype=object)
    pos = np.concatenate([np.arange(100) * 100_000, np.arange(100) * 100_000])
    eligible = np.arange(m)
    idx = sim.select_causal_snps(rng, eligible, chrom, pos, q=6,
                                 min_sep_bp=1_000_000)
    assert idx.size == 6
    assert np.all(np.diff(idx) > 0)     # sorted
    # within-chromosome separation respected
    for a in idx:
        for b in idx:
            if a != b and chrom[a] == chrom[b]:
                assert abs(int(pos[a]) - int(pos[b])) >= 1_000_000


def test_select_causal_snps_deterministic():
    chrom = np.array(["1"] * 300, dtype=object)
    pos = np.arange(300) * 50_000
    el = np.arange(300)
    a = sim.select_causal_snps(np.random.default_rng(7), el, chrom, pos, 5,
                               min_sep_bp=200_000)
    b = sim.select_causal_snps(np.random.default_rng(7), el, chrom, pos, 5,
                               min_sep_bp=200_000)
    np.testing.assert_array_equal(a, b)


def test_select_causal_snps_zero_and_too_few():
    rng = np.random.default_rng(1)
    chrom = np.array(["1"] * 10, dtype=object)
    pos = np.arange(10) * 100_000
    assert sim.select_causal_snps(rng, np.arange(10), chrom, pos, 0).size == 0
    with pytest.raises(ValueError):
        sim.select_causal_snps(rng, np.arange(3), chrom, pos, 5)


# ---------------------------------------------------------------------
# kernel_factor
# ---------------------------------------------------------------------

def test_kernel_factor_reconstructs():
    K = _toy_kernel(60, 120, seed=5)
    L = sim.kernel_factor(K)
    np.testing.assert_allclose(L @ L.T, K, atol=1e-8)


def test_kernel_factor_rejects_non_psd():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])    # eigenvalue -1
    with pytest.raises(ValueError):
        sim.kernel_factor(bad)


# ---------------------------------------------------------------------
# simulate_phenotype
# ---------------------------------------------------------------------

def _sim_setup(n=220, seed=11):
    K_A = _toy_kernel(n, 200, seed)
    K_C = _toy_kernel(n, 200, seed + 1)
    K_sum = normalize_kernel(K_A + K_C, "trace")
    from homoeogwas import hadamard_kernel
    K_hom = normalize_kernel(hadamard_kernel({"A": K_A, "C": K_C}), "trace")
    factors = {"A": sim.kernel_factor(K_A), "C": sim.kernel_factor(K_C),
               "pooled": sim.kernel_factor(K_sum),
               "hom": sim.kernel_factor(K_hom)}
    rng = np.random.default_rng(seed + 2)
    q = 5
    Z_A = rng.standard_normal((n, q))
    Z_A = (Z_A - Z_A.mean(0)) / Z_A.std(0)
    Z_C = rng.standard_normal((n, q))
    Z_C = (Z_C - Z_C.mean(0)) / Z_C.std(0)
    causal = sim.CausalSet(
        causal_id=np.arange(2 * q), subgenome=np.array(["A"] * q + ["C"] * q),
        snp_id=np.array([f"s{i}" for i in range(2 * q)], dtype=object),
        chrom=np.array(["1"] * 2 * q, dtype=object),
        pos=np.arange(2 * q) * 10_000, marker_index=np.arange(2 * q),
        effect=np.zeros(2 * q))
    return factors, Z_A, Z_C, causal


def test_simulate_phenotype_realized_variances_stratified():
    factors, Z_A, Z_C, causal = _sim_setup()
    arm = sim.default_arms()["S1_C_dominant"]
    ph = sim.simulate_phenotype(np.random.default_rng(0), arm, causal,
                                Z_A, Z_C, factors)
    rv = ph.realized_var
    # each component is rescaled to its exact target sample variance
    assert rv["var_qtl_A"] == pytest.approx(arm.h_qtl_A, abs=1e-9)
    assert rv["var_qtl_C"] == pytest.approx(arm.h_qtl_C, abs=1e-9)
    assert rv["var_poly_A"] == pytest.approx(arm.h_poly_A, abs=1e-9)
    assert rv["var_poly_C"] == pytest.approx(arm.h_poly_C, abs=1e-9)
    assert rv["var_poly_pooled"] == 0.0 and rv["var_poly_hom"] == 0.0
    assert rv["var_residual"] == pytest.approx(arm.h_e, abs=1e-9)
    # realized PVE ~ target (deviates by finite-sample cross-covariance)
    assert rv["pve_qtl_A"] == pytest.approx(arm.h_qtl_A, abs=0.10)
    assert rv["pve_poly_C"] == pytest.approx(arm.h_poly_C, abs=0.10)
    # phenotype is centered and standardized to unit variance
    assert ph.y.shape == (Z_A.shape[0],)
    assert abs(ph.y.mean()) < 1e-9
    assert np.var(ph.y, ddof=1) == pytest.approx(1.0, abs=1e-9)


def test_simulate_phenotype_pooled_and_null_and_hadamard():
    factors, Z_A, Z_C, causal = _sim_setup()
    arms = sim.default_arms()
    ph_p = sim.simulate_phenotype(np.random.default_rng(1), arms["S3_pooled"],
                                  causal, Z_A, Z_C, factors)
    assert ph_p.realized_var["var_poly_pooled"] == pytest.approx(
        arms["S3_pooled"].h_poly_pooled, abs=1e-9)
    assert ph_p.realized_var["var_poly_A"] == 0.0
    ph_n = sim.simulate_phenotype(np.random.default_rng(2), arms["S5_null"],
                                  causal, Z_A, Z_C, factors)
    assert ph_n.realized_var["var_qtl_A"] == 0.0
    assert ph_n.realized_var["var_qtl_C"] == 0.0
    ph_h = sim.simulate_phenotype(np.random.default_rng(3), arms["S6_hadamard"],
                                  causal, Z_A, Z_C, factors)
    assert ph_h.realized_var["var_poly_hom"] == pytest.approx(
        arms["S6_hadamard"].h_poly_hom, abs=1e-9)


def test_draw_qtl_equal_magnitude_effects():
    """Causal effects are equal-magnitude (random sign) for a fair power test."""
    factors, Z_A, Z_C, causal = _sim_setup()
    arm = sim.default_arms()["S1_C_dominant"]
    ph = sim.simulate_phenotype(np.random.default_rng(0), arm, causal,
                                Z_A, Z_C, factors)
    eff = ph.causal.effect
    # all A effects share one magnitude, all C effects share another
    q = Z_A.shape[1]
    assert np.allclose(np.abs(eff[:q]), np.abs(eff[0]))
    assert np.allclose(np.abs(eff[q:]), np.abs(eff[q]))
    assert set(np.sign(eff).tolist()) <= {-1.0, 1.0}


def test_simulate_phenotype_deterministic():
    factors, Z_A, Z_C, causal = _sim_setup()
    arm = sim.default_arms()["S2_A_dominant"]
    y1 = sim.simulate_phenotype(np.random.default_rng(42), arm, causal,
                                Z_A, Z_C, factors).y
    y2 = sim.simulate_phenotype(np.random.default_rng(42), arm, causal,
                                Z_A, Z_C, factors).y
    np.testing.assert_array_equal(y1, y2)


def test_simulate_phenotype_causal_signal_is_recoverable():
    """An (almost) QTL-only arm: causal SNPs explain ~all variance."""
    factors, Z_A, Z_C, causal = _sim_setup(n=400)
    arm = sim.SimArm("qtl_only", "stratified", h_qtl_A=0.45, h_qtl_C=0.45)
    ph = sim.simulate_phenotype(np.random.default_rng(9), arm, causal,
                                Z_A, Z_C, factors)
    Zc = np.hstack([Z_A, Z_C])
    # OLS of y on the causal SNPs
    coef, *_ = np.linalg.lstsq(Zc, ph.y, rcond=None)
    resid = ph.y - Zc @ coef
    r2 = 1.0 - np.var(resid) / np.var(ph.y)
    assert r2 > 0.85                          # causal SNPs carry the signal


# ---------------------------------------------------------------------
# LD & truth windows
# ---------------------------------------------------------------------

def test_ld_r2_self_and_independent():
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((300, 5))
    Z = (Z - Z.mean(0)) / Z.std(0)
    r2 = sim.ld_r2(Z[:, 0], Z)
    assert r2[0] == pytest.approx(1.0, abs=1e-9)          # self
    assert np.all(r2[1:] < 0.1)                           # independent cols


def test_build_truth_windows_physical_and_ld():
    n = 250
    rng = np.random.default_rng(6)
    Z_A = rng.standard_normal((n, 10))
    Z_A = (Z_A - Z_A.mean(0)) / Z_A.std(0)
    # make marker 3 a near-copy of the causal marker 0 (high LD)
    Z_A[:, 3] = Z_A[:, 0] + 0.05 * rng.standard_normal(n)
    Z_A[:, 3] = (Z_A[:, 3] - Z_A[:, 3].mean()) / Z_A[:, 3].std()
    snp = np.array([f"a{i}" for i in range(10)], dtype=object)
    chrom = np.array(["1"] * 10, dtype=object)
    # a3 sits outside the 500 kb physical window but inside the 2 Mb LD search
    pos = np.array([0, 100_000, 700_000, 1_500_000, 1_800_000,
                    20_000_000, 21_000_000, 22_000_000, 23_000_000,
                    24_000_000])
    causal = sim.CausalSet(
        causal_id=np.array([0]), subgenome=np.array(["A"], dtype=object),
        snp_id=np.array(["a0"], dtype=object), chrom=np.array(["1"], dtype=object),
        pos=np.array([0]), marker_index=np.array([0]), effect=np.zeros(1))
    geno_meta = {"A": {"snp_id": snp, "chrom": chrom, "pos": pos}}
    truth = sim.build_truth_windows(causal, geno_meta, {"A": Z_A},
                                    window_bp=500_000, r2_min=0.20)
    sub, tagged = truth[0]
    assert sub == "A"
    assert "a0" in tagged              # self
    assert "a1" in tagged              # within 500 kb (100k)
    assert "a2" not in tagged          # 700 kb > 500 kb, independent
    assert "a3" in tagged              # high LD, within the 2 Mb LD search
    assert "a5" not in tagged          # far + independent


def test_build_truth_windows_physical_boundary():
    snp = np.array(["a0", "a1", "a2"], dtype=object)
    chrom = np.array(["1", "1", "1"], dtype=object)
    pos = np.array([1_000_000, 1_400_000, 1_600_000])
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((100, 3))
    Z = (Z - Z.mean(0)) / Z.std(0)
    causal = sim.CausalSet(
        causal_id=np.array([0]), subgenome=np.array(["A"], dtype=object),
        snp_id=np.array(["a0"], dtype=object), chrom=np.array(["1"], dtype=object),
        pos=np.array([1_000_000]), marker_index=np.array([0]),
        effect=np.zeros(1))
    truth = sim.build_truth_windows(
        causal, {"A": {"snp_id": snp, "chrom": chrom, "pos": pos}},
        {"A": Z}, window_bp=500_000, r2_min=0.99)
    _sub, tagged = truth[0]
    assert "a1" in tagged              # 400 kb <= 500 kb
    assert "a2" not in tagged          # 600 kb > 500 kb, low LD


# ---------------------------------------------------------------------
# collapse_loci & evaluate_threshold
# ---------------------------------------------------------------------

def test_collapse_loci_grouping():
    sub = np.array(["A", "A", "A", "A", "C"], dtype=object)
    chrom = np.array(["1", "1", "1", "2", "1"], dtype=object)
    pos = np.array([100, 200_000, 9_000_000, 100, 100])
    p = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    loci = sim._collapse_loci(sub, chrom, pos, p, collapse_window_bp=500_000)
    # (A,1,100)+(A,1,200000) merge; (A,1,9e6) alone; (A,2) alone; (C,1) alone
    assert len(loci) == 4
    sizes = sorted(len(x) for x in loci)
    assert sizes == [1, 1, 1, 2]


def _toy_sumstats(rows):
    """rows: list of (subgenome, snp_id, chrom, pos, p)."""
    return {
        "subgenome": np.array([r[0] for r in rows], dtype=object),
        "snp_id": np.array([r[1] for r in rows], dtype=object),
        "chrom": np.array([r[2] for r in rows], dtype=object),
        "pos": np.array([r[3] for r in rows], dtype=np.int64),
        "p": np.array([r[4] for r in rows], dtype=np.float64),
    }


def test_evaluate_threshold_hand_computed():
    truth = {0: ("A", frozenset({"snpA1", "snpA2"})),
             1: ("C", frozenset({"snpC1"}))}
    ss = _toy_sumstats([
        ("A", "snpA1", "1", 100, 1e-9),          # tags causal 0  -> TP
        ("A", "snpA9", "1", 5_000_000, 1e-9),    # tags nothing   -> FP
        ("C", "snpC1", "1", 100, 1e-9),          # tags causal 1  -> TP
        ("C", "snpX", "2", 100, 0.5),            # not significant
    ])
    ev = sim.evaluate_threshold(ss, truth, threshold=1e-5,
                                collapse_window_bp=500_000)
    assert ev.n_sig == 3
    assert ev.n_loci == 3
    assert ev.n_tp_loci == 2
    assert ev.n_fp_loci == 1
    assert ev.n_detected_causal == 2
    assert ev.power == pytest.approx(1.0)
    assert ev.fdp == pytest.approx(1 / 3)


def test_evaluate_threshold_nothing_significant():
    truth = {0: ("A", frozenset({"snpA1"}))}
    ss = _toy_sumstats([("A", "snpA1", "1", 100, 0.4)])
    ev = sim.evaluate_threshold(ss, truth, threshold=1e-5)
    assert ev.n_sig == 0
    assert ev.power == 0.0 and ev.fdp == 0.0


def test_evaluate_threshold_null_arm():
    """No causal SNPs -> power is NaN, fdp counts every discovery locus."""
    truth: dict = {}
    ss = _toy_sumstats([
        ("A", "s1", "1", 100, 1e-9),
        ("C", "s2", "1", 100, 1e-9),
    ])
    ev = sim.evaluate_threshold(ss, truth, threshold=1e-5)
    assert ev.n_causal == 0
    assert np.isnan(ev.power)
    assert ev.n_fp_loci == 2
    assert ev.fdp == pytest.approx(1.0)


# ---------------------------------------------------------------------
# BH adjust / threshold
# ---------------------------------------------------------------------

def test_bh_adjust_known_values():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = sim.bh_adjust(p)
    # BH q_(i) = p_(i) * n / i, monotone from the right
    expected = np.minimum.accumulate(
        (p * 5 / np.arange(1, 6))[::-1])[::-1]
    np.testing.assert_allclose(q, expected)
    assert np.all(np.diff(q) >= -1e-12)            # monotone


def test_bh_adjust_handles_nan():
    p = np.array([0.001, np.nan, 0.5])
    q = sim.bh_adjust(p)
    assert np.isnan(q[1])
    assert np.isfinite(q[0]) and np.isfinite(q[2])


def test_bh_threshold():
    # 100 p-values: 5 tiny, 95 uniform-ish large
    p = np.concatenate([np.full(5, 1e-6), np.linspace(0.2, 1.0, 95)])
    thr = sim.bh_threshold(p, q_level=0.05)
    assert thr >= 1e-6
    # everything <= thr should be the 5 signals
    assert np.sum(p <= thr) >= 5
    # no signal at all -> threshold 0
    assert sim.bh_threshold(np.linspace(0.5, 1.0, 50), 0.05) == 0.0


# ---------------------------------------------------------------------
# power_at_fdr / partial_auc
# ---------------------------------------------------------------------

def test_power_at_fdr():
    mp = np.array([0.9, 0.7, 0.5, 0.3])
    mf = np.array([0.40, 0.20, 0.04, 0.01])
    # thresholds with FDR <= 0.05: indices 2,3 -> powers 0.5,0.3 -> max 0.5
    assert sim.power_at_fdr(mp, mf, 0.05) == pytest.approx(0.5)
    # none qualify
    assert sim.power_at_fdr(mp, np.full(4, 0.9), 0.05) == 0.0


def test_partial_auc_monotone_envelope():
    # near-perfect method: power 1 from FDR~0 -> pAUC ~ 1 (the (0,0) anchor
    # costs a negligible sliver below the smallest attainable FDR)
    mp = np.ones(10)
    mf = np.linspace(0.0, 0.2, 10)
    assert sim.partial_auc(mp, mf, fdr_cap=0.10) > 0.99
    # zero-power method -> pAUC 0
    assert sim.partial_auc(np.zeros(10), mf, 0.10) == pytest.approx(0.0, abs=1e-6)
    # the (0,0) anchor: power credited only above the smallest attainable FDR
    mp2 = np.array([1.0, 1.0])
    mf2 = np.array([0.10, 0.20])      # nothing reaches FDR < 0.10
    assert sim.partial_auc(mp2, mf2, fdr_cap=0.10) == pytest.approx(0.5, abs=0.02)


def test_evaluate_threshold_duplicate_locus_counts_as_fp():
    """Two discovery loci tagging the SAME causal: 1 TP, 1 duplicate (FP)."""
    truth = {0: ("A", frozenset({"mL", "mR"}))}
    ss = _toy_sumstats([
        ("A", "mL", "1", 100_000, 1e-9),     # locus 1 -> claims causal 0 (TP)
        ("A", "mR", "1", 900_000, 1e-8),     # locus 2 -> causal 0 already (dup)
    ])
    ev = sim.evaluate_threshold(ss, truth, threshold=1e-5,
                                collapse_window_bp=500_000)
    assert ev.n_loci == 2                    # 800 kb apart -> not collapsed
    assert ev.n_tp_loci == 1
    assert ev.n_duplicate_loci == 1
    assert ev.n_fp_loci == 1                 # duplicate counts toward FP
    assert ev.power == pytest.approx(1.0)
    assert ev.fdp == pytest.approx(0.5)


# ---------------------------------------------------------------------
# paired_bootstrap / holm
# ---------------------------------------------------------------------

def test_paired_bootstrap_clear_win():
    a = np.full(30, 0.8)
    b = np.full(30, 0.5)
    res = sim.paired_bootstrap(a, b, rng=np.random.default_rng(0), n_boot=2000)
    assert res["delta"] == pytest.approx(0.3)
    assert res["p_value"] < 0.001            # always positive -> tiny p
    assert res["ci_lo"] == pytest.approx(0.3) and res["ci_hi"] == pytest.approx(0.3)


def test_paired_bootstrap_no_difference():
    a = np.linspace(0.4, 0.6, 25)
    res = sim.paired_bootstrap(a, a.copy(), rng=np.random.default_rng(1),
                               n_boot=2000)
    assert res["delta"] == pytest.approx(0.0)
    assert res["p_value"] > 0.5              # no evidence a > b


def test_paired_bootstrap_deterministic():
    a = np.random.default_rng(2).random(20)
    b = np.random.default_rng(3).random(20)
    r1 = sim.paired_bootstrap(a, b, rng=np.random.default_rng(5), n_boot=1000)
    r2 = sim.paired_bootstrap(a, b, rng=np.random.default_rng(5), n_boot=1000)
    assert r1 == r2


def test_paired_bootstrap_shape_mismatch():
    with pytest.raises(ValueError):
        sim.paired_bootstrap(np.zeros(5), np.zeros(6),
                             rng=np.random.default_rng(0))


def test_holm_correct():
    raw = {"gemma": 0.01, "regenie": 0.04}
    adj = sim.holm_correct(raw)
    # sorted: gemma 0.01*2=0.02 ; regenie 0.04*1=0.04
    assert adj["gemma"] == pytest.approx(0.02)
    assert adj["regenie"] == pytest.approx(0.04)
    # monotone: the larger raw p never gets a smaller adjusted p
    assert adj["regenie"] >= adj["gemma"]


def test_power_fdr_curve_length():
    truth = {0: ("A", frozenset({"s1"}))}
    ss = _toy_sumstats([("A", "s1", "1", 100, 1e-9),
                        ("A", "s2", "1", 9_000_000, 0.5)])
    thr = np.logspace(-2, -8, 10)
    curve = sim.power_fdr_curve(ss, truth, thr)
    assert len(curve) == 10
    assert all(isinstance(e, sim.ThresholdEval) for e in curve)
