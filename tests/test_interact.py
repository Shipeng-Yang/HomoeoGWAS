"""Tests for homoeogwas.interact — homoeolog-pair / triad burden-product engine.

Synthetic genotypes (random dosage -> near-identity GRM) exercise the statistical core:
an injected interaction is detected in the correct subgenome-pair and isolated from the
others; a pure-noise null produces no Bonferroni hits. These guard the engine before the
DL-weighting and multi-trait extensions are layered on.
"""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

import homoeogwas.interact as I
from homoeogwas import workflow
from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges
from homoeogwas.interact import (
    FrozenTraitSet,
    SubgenomeData,
    acat,
    acat_weighted,
    block_burden_capped,
    gene_pc1_matrix,
    gene_pc_scores,
    kernel_interaction_pvals,
    omnibus_pvals,
    pair_conditional_diagnostics,
    pairwise_pvals,
    run_clique_scan,
    run_clique_scan_omnib,
    run_multitrait_pair_scan,
    run_pair_scan,
    run_pair_scan_omnib,
    run_triad3_scan,  # noqa: F401 - exercised by legacy-route tests outside Task 5
    run_triad_scan,
    threeway_design_mask,  # noqa: F401 - exercised by legacy-route tests outside Task 5
    threeway_pvals,  # noqa: F401 - exercised by legacy-route tests outside Task 5
)
from homoeogwas.interaction_config import normalize_interact_config
from homoeogwas.omnib_family import OmniBFamilyScores

N, G, SPG = 300, 60, 8  # samples, genes/triads, snps per gene


def _make_sub(rng, n=N, g=G, spg=SPG):
    X = rng.integers(0, 3, size=(n, g * spg)).astype(float)
    gene_snp = {f"g{i}": np.arange(i * spg, (i + 1) * spg) for i in range(g)}
    return SubgenomeData(X=X, gene_snp=gene_snp, samples=[f"s{j}" for j in range(n)], chunk=None)


def _std(v):
    return (v - v.mean()) / v.std()


def test_acat_combines_pvalues():
    # a single tiny p drives the ACAT combination toward significance
    assert acat(np.array([1e-8, 0.5, 0.5])) < 0.05
    # all-null p stay null
    assert acat(np.array([0.5, 0.5, 0.5])) > 0.2


def test_acat_extreme_small_p_robust_and_bit_exact():
    # a 1e-200 dominating p must give a finite near-zero combined p (plain tan saturates near pi/2)
    a = acat(np.array([1e-200, 0.5, 0.5, 0.5]))
    assert np.isfinite(a) and 0.0 < a < 1e-2
    # monotone: a tinier dominating p yields a smaller combined p
    assert acat(np.array([1e-200, 0.5])) < acat(np.array([1e-8, 0.5]))
    # normal-range values are bit-exact vs the plain Cauchy formula (small-p branch not triggered)
    p = np.array([0.01, 0.2, 0.5, 0.8])
    t = float(np.mean(np.tan((0.5 - p) * np.pi)))
    assert acat(p) == float(0.5 - np.arctan(t) / np.pi)


def _ld_block(rng, n, m):
    """Dosage block (0/1/2) with LD so PCs are non-trivial."""
    lat = rng.standard_normal((n, 3)) @ rng.standard_normal((3, m))
    out = np.zeros((n, m))
    for j in range(m):
        p = rng.uniform(0.15, 0.5)
        out[:, j] = ((lat[:, j] > np.quantile(lat[:, j], 1 - p)).astype(float)
                     + (lat[:, j] > np.quantile(lat[:, j], 1 - p / 2)).astype(float))
    return out


def test_gene_pc_scores_invariant_to_ref_alt_flip():
    # per-SNP REF/ALT swap (x -> 2 - x) sends z -> -z; PC scores must be numerically identical.
    rng = np.random.default_rng(3)
    X = _ld_block(rng, 200, 14)
    idx = np.arange(14)
    s0 = gene_pc_scores(X, idx, 150, rng, n_pc=3)
    flip = rng.random(14) < 0.5
    Xf = X.copy()
    Xf[:, flip] = 2 - Xf[:, flip]
    s1 = gene_pc_scores(Xf, idx, 150, rng, n_pc=3)
    assert np.allclose(s0, s1, atol=1e-9)
    # the burden is NOT invariant under a partial flip (the estimand the reviewer flagged)
    b0 = block_burden_capped(X, idx, 150, rng)
    b1 = block_burden_capped(Xf, idx, 150, rng)
    assert not np.allclose(b0, b1, atol=1e-6)


def _separated_spectrum_block(rng, n=90, m=7):
    """Finite block whose leading singular directions are unambiguous across drivers."""
    left, _ = np.linalg.qr(rng.normal(size=(n, m)))
    right, _ = np.linalg.qr(rng.normal(size=(m, m)))
    singular_values = np.geomspace(12.0, 0.25, num=m)
    return left @ np.diag(singular_values) @ right.T


def test_pc_scores_uses_gesvd_fallback_with_frozen_options(monkeypatch):
    import scipy.linalg

    rng = np.random.default_rng(301)
    Z = _separated_spectrum_block(rng)
    expected = I.pc_scores_std(Z, n_pc=3)
    scipy_svd = scipy.linalg.svd
    calls = []

    def _fail_gesdd(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic gesdd non-convergence")

    def _record_gesvd(a, **kwargs):
        calls.append((a, kwargs))
        return scipy_svd(a, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", _fail_gesdd)
    monkeypatch.setattr(scipy.linalg, "svd", _record_gesvd)
    observed = I.pc_scores_std(Z, n_pc=3)

    assert len(calls) == 1
    assert calls[0][0] is Z
    assert calls[0][1] == {
        "full_matrices": False,
        "check_finite": True,
        "lapack_driver": "gesvd",
    }
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(observed.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(observed.std(axis=0), 1.0, atol=1e-14)


def test_pc_scores_gesvd_fallback_handles_finite_rank_deficiency(monkeypatch):
    rng = np.random.default_rng(302)
    a, b = rng.normal(size=(2, 80))
    Z = np.column_stack([a, b, a + b, 2.0 * a, np.zeros_like(a)])
    expected = I.pc_scores_std(Z, n_pc=5)

    def _fail_gesdd(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic gesdd non-convergence")

    monkeypatch.setattr(np.linalg, "svd", _fail_gesdd)
    observed = I.pc_scores_std(Z, n_pc=5)

    assert observed.shape == (80, 2)
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(observed.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(observed.std(axis=0), 1.0, atol=1e-14)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_pc_scores_rejects_nonfinite_before_either_svd(monkeypatch, bad_value):
    import scipy.linalg

    Z = np.ones((12, 3), dtype=float)
    Z[4, 1] = bad_value

    def _forbidden(*args, **kwargs):
        raise AssertionError("an SVD driver was called for non-finite input")

    monkeypatch.setattr(np.linalg, "svd", _forbidden)
    monkeypatch.setattr(scipy.linalg, "svd", _forbidden)
    with pytest.raises(ValueError, match="standardized gene block contains non-finite"):
        I.pc_scores_std(Z, n_pc=2)


def test_pc_scores_normal_path_never_calls_scipy_svd(monkeypatch):
    import scipy.linalg

    Z = _separated_spectrum_block(np.random.default_rng(303))

    def _forbidden(*args, **kwargs):
        raise AssertionError("SciPy SVD must remain exceptional-path-only")

    monkeypatch.setattr(scipy.linalg, "svd", _forbidden)
    observed = I.pc_scores_std(Z, n_pc=3)
    assert observed.shape == (90, 3)


def test_pc_scores_does_not_mask_non_linalg_primary_error(monkeypatch):
    import scipy.linalg

    Z = _separated_spectrum_block(np.random.default_rng(304))

    def _programming_error(*args, **kwargs):
        raise TypeError("synthetic programming error")

    def _forbidden(*args, **kwargs):
        raise AssertionError("SciPy SVD must not handle non-LinAlgError failures")

    monkeypatch.setattr(np.linalg, "svd", _programming_error)
    monkeypatch.setattr(scipy.linalg, "svd", _forbidden)
    with pytest.raises(TypeError, match="synthetic programming error"):
        I.pc_scores_std(Z, n_pc=3)


def test_pc_scores_both_svd_drivers_fail_informatively(monkeypatch):
    import scipy.linalg

    Z = np.ones((25, 4), dtype=float)
    gesvd_error = np.linalg.LinAlgError("synthetic gesvd non-convergence")

    def _fail_gesdd(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic gesdd non-convergence")

    def _fail_gesvd(*args, **kwargs):
        raise gesvd_error

    def _forbid_matrix_rank(*args, **kwargs):
        raise AssertionError("failure diagnostics must not run a third SVD")

    monkeypatch.setattr(np.linalg, "svd", _fail_gesdd)
    monkeypatch.setattr(np.linalg, "matrix_rank", _forbid_matrix_rank)
    monkeypatch.setattr(scipy.linalg, "svd", _fail_gesvd)
    with pytest.raises(
            ValueError,
            match=r"gene-block PCA failed with both gesdd and gesvd \(shape=\(25, 4\)\)") as err:
        I.pc_scores_std(Z, n_pc=3)
    assert err.value.__cause__ is gesvd_error


def test_pc_and_kernel_interaction_invariant_burden_not():
    # inject a coherent burden-product signal, then flip half of each gene's SNPs: the burden-product
    # p moves, but PC1xPC1 and the low-rank kernel-interaction p are invariant to numerical precision.
    rng = np.random.default_rng(5)
    n = 300
    X, D = _ld_block(rng, n, 12), _ld_block(rng, n, 10)
    ix, iD = np.arange(12), np.arange(10)
    bx = _std(block_burden_capped(X, ix, 150, rng))
    bd = _std(block_burden_capped(D, iD, 150, rng))
    y = rng.standard_normal(n) + 1.2 * _std(bx * bd)
    Wh = np.eye(n)

    def three(Xi, Di):
        pb = pairwise_pvals(Wh, y, block_burden_capped(Xi, ix, 150, rng).reshape(n, 1),
                            block_burden_capped(Di, iD, 150, rng).reshape(n, 1))[0]
        ppc = pairwise_pvals(Wh, y, gene_pc1_matrix(Xi, ["g"], {"g": ix}, 150, rng),
                             gene_pc1_matrix(Di, ["g"], {"g": iD}, 150, rng))[0]
        pk = kernel_interaction_pvals(Wh, y, [gene_pc_scores(Xi, ix, 150, rng, 3)],
                                      [gene_pc_scores(Di, iD, 150, rng, 3)])[0]
        return pb, ppc, pk

    b0 = three(X, D)
    fx, fd = rng.random(12) < 0.5, rng.random(10) < 0.5
    Xf, Df = X.copy(), D.copy()
    Xf[:, fx] = 2 - Xf[:, fx]
    Df[:, fd] = 2 - Df[:, fd]
    b1 = three(Xf, Df)
    assert abs(b1[1] - b0[1]) < 1e-9 and abs(b1[2] - b0[2]) < 1e-9   # PC & kernel invariant
    assert abs(b1[0] - b0[0]) > 1e-6                                 # burden-product moves
    # the omnibus stays significant even when the flipped burden collapses
    o1 = omnibus_pvals(np.array([b1[0]]), np.array([b1[1]]), np.array([b1[2]]))[0]
    assert o1 < 1e-3


def test_null_replicates_preserve_kinship_yshuffle_does_not():
    # A raw y-shuffle destroys the kinship covariance: refitting REML on the shuffled phenotype
    # drives the genetic variance components to ~0, so the permuted scans run effectively
    # unwhitened. The bootstrap / whitened-residual nulls must REPRODUCE the genetic variance.
    from homoeogwas.interact import grm_from_X, null_replicates, whiten_multi

    rng = np.random.default_rng(0)
    n, m = 150, 400
    pop = np.repeat(np.arange(3), n // 3)
    freq = rng.uniform(0.1, 0.9, (3, m))
    X = np.array([rng.binomial(2, freq[pop[i]]) for i in range(n)], float)
    XA, XD = X[:, :200], X[:, 200:]
    kernels = {"A": grm_from_X(XA), "D": grm_from_X(XD)}
    gA = _std(XA @ rng.standard_normal(200))
    gD = _std(XD @ rng.standard_normal(200))
    y = _std(1.5 * gA + 1.5 * gD + rng.standard_normal(n))
    _, cv_obs = whiten_multi(kernels, y, seed=1)
    g_obs = cv_obs.get("A", 0.0) + cv_obs.get("D", 0.0)
    assert g_obs > 0.3                                   # the observed phenotype IS kinship-structured

    def _gvar(method):
        reps, _, _ = null_replicates(kernels, y, B=3, method=method, seed=5)
        cvs = [whiten_multi(kernels, r, seed=1)[1] for r in reps]
        return np.mean([c.get("A", 0.0) + c.get("D", 0.0) for c in cvs])

    assert _gvar("yshuffle") < 0.15 * g_obs              # kinship destroyed
    assert _gvar("bootstrap") > 0.5 * g_obs              # kinship reproduced
    assert _gvar("whitened") > 0.5 * g_obs               # kinship reproduced


def test_minor_allele_burden_strictly_invariant():
    # minor-allele-coded burden is fixed by frequency, not REF/ALT -> a swap leaves it identical,
    # while the default (REF-coded) burden changes under a partial flip.
    rng = np.random.default_rng(4)
    X = _ld_block(rng, 200, 13)
    # append an exact freq=0.5 double-tie SNP (equal homozygotes) — the case that broke a float rule
    tie = np.array(([2] * 50 + [0] * 50 + [1] * 100), float)
    X = np.column_stack([X, tie])
    idx = np.arange(14)
    b0 = block_burden_capped(X, idx, 150, rng, minor=True)
    for f in (0.25, 0.5, 1.0):
        fl = rng.random(14) < f
        Xf = X.copy()
        Xf[:, fl] = 2 - Xf[:, fl]
        assert np.allclose(b0, block_burden_capped(Xf, idx, 150, rng, minor=True), atol=1e-12)
    Xh = X.copy()
    Xh[:, :7] = 2 - Xh[:, :7]
    assert not np.allclose(block_burden_capped(X, idx, 150, rng),
                           block_burden_capped(Xh, idx, 150, rng), atol=1e-6)


def test_omnibus_drops_nonestimable_component():
    # a non-estimable component (NaN or the p==1 sentinel) must be DROPPED, not clipped to 1-eps
    # (which injects a huge negative Cauchy term and would mask a real signal).
    pb = np.array([1e-6, 0.4])
    ppc = np.array([np.nan, 0.5])          # NaN component
    pk = np.array([1.0, 0.6])              # p==1 sentinel
    o = omnibus_pvals(pb, ppc, pk)
    # pair 0: only the 1e-6 burden survives -> omnibus stays significant despite NaN + p==1
    assert o[0] < 1e-3
    # equivalent to ACAT of the burden alone for pair 0
    assert abs(o[0] - acat(np.array([1e-6]))) < 1e-12
    with pytest.raises(ValueError):
        omnibus_pvals(np.array([0.1, 0.2]), np.array([0.3]))   # unequal length


def test_kernel_interaction_nonestimable_is_nan():
    # a monomorphic / rank-0 gene block yields no estimable interaction -> NaN (not p=1)
    rng = np.random.default_rng(9)
    n = 120
    y = rng.standard_normal(n)
    good = gene_pc_scores(_ld_block(rng, n, 8), np.arange(8), 150, rng, 3)
    mono = np.zeros((n, 1))                 # degenerate block -> zero PC column
    pk = kernel_interaction_pvals(np.eye(n), y, [good], [mono])
    assert np.isnan(pk[0])


def test_triad_detects_and_isolates_bd_interaction():
    rng = np.random.default_rng(0)
    subs = ["A", "B", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    triads = [(f"g{i}", f"g{i}", f"g{i}") for i in range(G)]
    sidx = np.arange(N)
    hit = 17
    bB = block_burden_capped(subdata["B"].X, subdata["B"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.8 * (_std(bB) * _std(bD))

    r = run_triad_scan(subdata, triads, y, sidx, cap=150, transform="INT", perm_B=0,
                       grm_method="grm_from_X")
    assert r["G"] == G
    # B-D pairwise detects the injected triad at Bonferroni
    bd = r["pairwise"]["BD"]
    assert bd["n_sig"] >= 1
    assert any(h["triad"][0] == f"g{hit}" for h in bd["sig"])
    # A-B and A-D do NOT pick it up (interaction is BD-specific)
    assert r["pairwise"]["AB"]["n_sig"] == 0
    assert r["pairwise"]["AD"]["n_sig"] == 0
    # triad-level omnibus is significant
    assert r["triad_acat_omnibus"] < 0.05


def test_triad_null_no_false_hits():
    rng = np.random.default_rng(1)
    subs = ["A", "B", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    triads = [(f"g{i}", f"g{i}", f"g{i}") for i in range(G)]
    y = rng.standard_normal(N)  # pure noise

    r = run_triad_scan(subdata, triads, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                       grm_method="grm_from_X")
    # hard calibration check: a pure-noise null yields no Bonferroni hits in any pairwise
    for tag in ("AB", "AD", "BD"):
        assert r["pairwise"][tag]["n_sig"] == 0
    # lambda_gc on only G=60 null p-values is high-variance; require it merely not wildly
    # inflated (mean across the 3 pairwise stays in a sane band)
    mean_lambda = np.mean([r["pairwise"][t]["lambda_gc_obs"] for t in ("AB", "AD", "BD")])
    assert 0.3 < mean_lambda < 2.2


def test_clique_n4_quartet_detects_and_isolates_cd_and_dumps_schema(tmp_path):
    # generic n-subgenome path: a true quartet (4 subgenomes) has C(4,2)=6 within-group
    # pairwise interactions; inject a C-D interaction and confirm it is detected, isolated,
    # and that the ranking dump has the generalized (4 gene / 6 p / 4 n_snp) columns.
    rng = np.random.default_rng(4)
    subs = ["A", "B", "C", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    quartets = [(f"g{i}", f"g{i}", f"g{i}", f"g{i}") for i in range(G)]
    hit = 23
    bC = block_burden_capped(subdata["C"].X, subdata["C"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.8 * (_std(bC) * _std(bD))

    dump = tmp_path / "rank_clique_INT.tsv"
    r = run_clique_scan(subdata, quartets, y, np.arange(N), cap=150, transform="INT",
                        perm_B=0, grm_method="grm_from_X", full_dump_path=str(dump))
    assert r["G"] == G
    # all six within-quartet pairs are scanned
    assert set(r["pairwise"]) == {"AB", "AC", "AD", "BC", "BD", "CD"}
    # C-D detects the injected interaction at Bonferroni...
    assert r["pairwise"]["CD"]["n_sig"] >= 1
    assert any(h["triad"][2] == f"g{hit}" for h in r["pairwise"]["CD"]["sig"])
    # ...and the injected pair dominates: CD has the smallest min-p of all six pairs
    # (a strict "0 hits in the other 5 pairs" is too tight at 6xG null tests).
    minp = {t: r["pairwise"][t]["min_p"] for t in r["pairwise"]}
    assert minp["CD"] == min(minp.values())
    assert minp["CD"] < 0.05 / G
    assert r["triad_acat_omnibus"] < 0.05
    # generalized ranking schema scales with n: 4 gene cols, 6 pair cols, 4 n_snp cols
    header = dump.read_text().splitlines()[0].split("\t")
    assert [c for c in header if c.startswith("gene_") and not c.startswith("gene_len")] == \
        ["gene_A", "gene_B", "gene_C", "gene_D"]
    assert [c for c in header if c.startswith("p_") and c != "p_acat"] == \
        ["p_AB", "p_AC", "p_AD", "p_BC", "p_BD", "p_CD"]
    assert [c for c in header if c.startswith("n_snp_") and c != "n_snp_group"] == \
        ["n_snp_A", "n_snp_B", "n_snp_C", "n_snp_D"]
    assert "n_snp_group" in header


def test_clique_n3_matches_triad_alias():
    # run_triad_scan is now an alias of run_clique_scan; n=3 behaviour is unchanged
    assert run_triad_scan is run_clique_scan
    rng = np.random.default_rng(7)
    subs = ["A", "B", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    triads = [(f"g{i}", f"g{i}", f"g{i}") for i in range(G)]
    r = run_clique_scan(subdata, triads, rng.standard_normal(N), np.arange(N),
                        cap=150, transform="INT", perm_B=0, grm_method="grm_from_X")
    assert r["G"] == G
    assert set(r["pairwise"]) == {"AB", "AD", "BD"}


def test_dominance_adjust_off_matches_legacy_ols():
    # With Wh=I and C=None the test reduces to OLS on [1,zA,zB,zA*zB]; dominance_adjust=False must
    # reproduce that legacy design exactly (backward compatibility for existing results).
    rng = np.random.default_rng(0)
    n = 200
    zA = _std(rng.standard_normal(n))
    zB = _std(0.5 * zA + rng.standard_normal(n))
    y = rng.standard_normal(n)
    p_off = pairwise_pvals(np.eye(n), y, zA.reshape(n, 1), zB.reshape(n, 1),
                           C=None, dominance_adjust=False)[0]
    # hand-rolled OLS interaction t-test on [1,zA,zB,zA*zB]
    X = np.column_stack([np.ones(n), zA, zB, zA * zB])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - 4
    se = np.sqrt((resid @ resid) / df * np.linalg.inv(X.T @ X)[3, 3])
    from scipy.stats import t as _t
    p_ref = float(2 * _t.sf(abs(beta[3] / se), df))
    assert abs(p_off - p_ref) < 1e-9


def test_dominance_adjust_controls_collinearity_dominance_leak():
    # Homoeolog collinearity + an unmodelled per-copy dominance (z^2) main effect, NO interaction:
    # the product term is collinear with z^2 as r->1, so the legacy test leaks; conditioning on the
    # squared burdens (dominance_adjust=True) must restore near-nominal type-I.
    rng = np.random.default_rng(1)
    n, reps, r = 400, 80, 0.85
    eye = np.eye(n)
    off, on = [], []
    for _ in range(reps):
        zA = _std(rng.standard_normal(n))
        zB = _std(r * zA + np.sqrt(1 - r * r) * rng.standard_normal(n))
        y = 1.3 * (zA ** 2 - 1) + 1.3 * (zB ** 2 - 1) + rng.standard_normal(n)  # dominance, no zA*zB
        bx, by = zA.reshape(n, 1), zB.reshape(n, 1)
        off.append(pairwise_pvals(eye, y, bx, by, dominance_adjust=False)[0] < 0.05)
        on.append(pairwise_pvals(eye, y, bx, by, dominance_adjust=True)[0] < 0.05)
    assert np.mean(off) > 0.30          # legacy design leaks badly under collinearity x dominance
    assert np.mean(on) < 0.15           # dominance-adjusted design controlled near nominal 0.05


def test_dominance_adjust_threads_through_pair_scan():
    # the switch reaches the production scan and changes the per-pair p on a dominance-leak gene,
    # while default-off leaves run_pair_scan behaviour unchanged.
    rng = np.random.default_rng(2)
    subs = ["A", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    hit = 11
    bA = block_burden_capped(subdata["A"].X, subdata["A"].gene_snp[f"g{hit}"], 150, rng)
    # make D's hit gene burden collinear-ish proxy via A so the product aliases the dominance term
    y = 1.5 * (_std(bA) ** 2 - 1) + rng.standard_normal(N)
    r_off = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                          pair_subs=("A", "D"), grm_method="grm_from_X", dominance_adjust=False)
    r_on = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                         pair_subs=("A", "D"), grm_method="grm_from_X", dominance_adjust=True)
    assert r_off.G == r_on.G
    assert r_off.min_p != r_on.min_p   # the dominance-conditioned design changes the result


def test_pairwise_2sub_detects_interaction():
    rng = np.random.default_rng(2)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    hit = 30
    bA = block_burden_capped(subdata["A"].X, subdata["A"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.8 * (_std(bA) * _std(bD))

    r = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                      pair_subs=("A", "D"), grm_method="grm_from_X")
    assert r.G == G
    assert r.n_sig >= 1
    assert any(h["pair"][0] == f"g{hit}" for h in r.sig)


def test_acat_weighted_upweights_the_signal():
    # one true signal (tiny p) among nulls; up-weighting it sharpens significance,
    # down-weighting it dulls it (weights are the y-independent prior)
    p = np.array([1e-4, 0.5, 0.5, 0.5])
    a_equal = acat_weighted(p, np.ones(4))
    a_up = acat_weighted(p, np.array([10.0, 1, 1, 1]))
    a_down = acat_weighted(p, np.array([0.1, 1, 1, 1]))
    assert a_up < a_equal < a_down


def test_weighted_scan_prior_helps_true_pair():
    rng = np.random.default_rng(3)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    hit = 30
    bA = block_burden_capped(subdata["A"].X, subdata["A"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 0.7 * (_std(bA) * _std(bD))
    weights = {(f"g{i}", f"g{i}"): (5.0 if i == hit else 1.0) for i in range(G)}

    r = run_pair_scan(subdata, pairs, y, np.arange(N), transform="INT", perm_B=0,
                      pair_subs=("A", "D"), grm_method="grm_from_X", pair_weights=weights,
                      primary_weighting="weighted")
    assert r.weighted is not None
    # weighted ACAT is at least as significant as unweighted (true signal up-weighted)
    assert r.weighted["acat_weighted"] <= r.pair_acat + 1e-12
    # the up-weighted true pair is among the weighted Bonferroni hits
    assert any(h["pair"][0] == f"g{hit}" for h in r.weighted["sig"])
    # exclusivity: the procedure that was NOT predeclared emits no rejections
    assert r.sig is None and r.weighted["role"] == "primary"
    r_unw = run_pair_scan(subdata, pairs, y, np.arange(N), transform="INT", perm_B=0,
                          pair_subs=("A", "D"), grm_method="grm_from_X", pair_weights=weights)
    # a suppressed procedure reports null, never an empty list that reads as "found nothing"
    assert r_unw.weighted["sig"] is None and r_unw.weighted["bonferroni_n_sig"] is None
    assert r_unw.weighted["role"] == "exploratory_no_rejections_emitted"
    assert r_unw.inference_plan["primary_weighting"] == "unweighted"


def test_weighted_null_no_type1_inflation():
    rng = np.random.default_rng(7)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    y = rng.standard_normal(N)  # pure noise
    # random y-INDEPENDENT weights -> weighted Bonferroni still controls FWER
    weights = {(f"g{i}", f"g{i}"): float(rng.uniform(0.2, 5.0)) for i in range(G)}
    r = run_pair_scan(subdata, pairs, y, np.arange(N), transform="INT", perm_B=0,
                      pair_subs=("A", "D"), grm_method="grm_from_X", pair_weights=weights)
    assert r.n_sig == 0                                  # unweighted is primary here
    assert r.weighted["bonferroni_n_sig"] is None        # weighted is suppressed, not "zero hits"
    rw = run_pair_scan(subdata, pairs, y, np.arange(N), transform="INT", perm_B=0,
                       pair_subs=("A", "D"), grm_method="grm_from_X", pair_weights=weights,
                       primary_weighting="weighted")
    assert rw.weighted["bonferroni_n_sig"] == 0          # evaluated, and finds nothing under the null
    assert rw.n_sig is None


# ----------------------------------------------------------------------------- multi-trait (#4)


def test_frozen_trait_set_contract():
    # empty / duplicate rejected; order preserved (NOT sorted); digest is deterministic
    with pytest.raises(ValueError):
        FrozenTraitSet.from_list([])
    with pytest.raises(ValueError):
        FrozenTraitSet.from_list(["t1", "t1", "t2"])
    ts = FrozenTraitSet.from_list(["b", "a", "c"])
    assert ts.traits == ("b", "a", "c")
    assert ts.digest == FrozenTraitSet.from_list(["b", "a", "c"]).digest
    assert ts.digest != FrozenTraitSet.from_list(["a", "b", "c"]).digest


def test_multitrait_detects_pleiotropic_pair_and_contract():
    rng = np.random.default_rng(11)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    hit = 25
    bA = block_burden_capped(subdata["A"].X, subdata["A"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    inter = _std(bA) * _std(bD)
    traits = ["t1", "t2", "t3"]
    ts = FrozenTraitSet.from_list(traits)
    # the same interaction is shared (pleiotropic) across the frozen trait set
    y_by_trait = {t: rng.standard_normal(N) + 1.0 * inter for t in traits}

    r = run_multitrait_pair_scan(subdata, pairs, y_by_trait, np.arange(N), trait_set=ts,
                                 transform="INT", perm_B=0, pair_subs=("A", "D"),
                                 grm_method="grm_from_X")
    assert r["G"] == G
    assert r["traits"] == traits                         # frozen order preserved in output
    assert r["bonferroni_alpha"] == pytest.approx(0.05 / G)  # multiplicity over G, NOT G*T
    assert r["n_sig"] >= 1
    assert any(h["pair"][0] == f"g{hit}" for h in r["sig"])
    # pleio_p of each reported pair == ACAT of its per-trait p's (audit equality)
    h = r["sig"][0]
    assert h["pleio_p"] == pytest.approx(acat(np.array([h["per_trait_p"][t] for t in traits])))


def test_multitrait_correlated_null_no_false_hits():
    rng = np.random.default_rng(12)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    traits = ["a", "b", "c"]
    ts = FrozenTraitSet.from_list(traits)
    # correlated null: shared latent across traits, NO genotype effect -> ACAT must stay calibrated
    shared = rng.standard_normal(N)
    y_by_trait = {t: 0.7 * shared + rng.standard_normal(N) for t in traits}

    r = run_multitrait_pair_scan(subdata, pairs, y_by_trait, np.arange(N), trait_set=ts,
                                 transform="INT", perm_B=0, pair_subs=("A", "D"),
                                 grm_method="grm_from_X")
    assert r["n_sig"] == 0
    assert 0.3 < r["lambda_gc_obs"] < 2.5


def test_multitrait_rejects_misordered_trait_keys():
    rng = np.random.default_rng(13)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    ts = FrozenTraitSet.from_list(["x", "y"])
    # y_by_trait keys in a DIFFERENT order than the frozen set -> must raise (no silent realign)
    y_by_trait = {"y": rng.standard_normal(N), "x": rng.standard_normal(N)}
    with pytest.raises(ValueError):
        run_multitrait_pair_scan(subdata, pairs, y_by_trait, np.arange(N), trait_set=ts,
                                 transform="INT", perm_B=0, pair_subs=("A", "D"),
                                 grm_method="grm_from_X")


def test_multitrait_edge_case_contracts():
    rng = np.random.default_rng(14)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    ts = FrozenTraitSet.from_list(["x", "y"])
    base = {"x": rng.standard_normal(N), "y": rng.standard_normal(N)}

    # pair_subs=None must raise
    with pytest.raises(ValueError):
        run_multitrait_pair_scan(subdata, pairs, base, np.arange(N), trait_set=ts,
                                 perm_B=0, pair_subs=None, grm_method="grm_from_X")
    # non-finite trait values must raise (direct-call firewall mirrors CLI complete-case)
    bad = {"x": base["x"].copy(), "y": base["y"].copy()}
    bad["y"][0] = np.nan
    with pytest.raises(ValueError):
        run_multitrait_pair_scan(subdata, pairs, bad, np.arange(N), trait_set=ts,
                                 perm_B=0, pair_subs=("A", "D"), grm_method="grm_from_X")
    # no retained pairs (gene IDs absent in both subgenomes) must raise, not divide-by-zero
    with pytest.raises(ValueError):
        run_multitrait_pair_scan(subdata, [("zzz", "zzz")], base, np.arange(N), trait_set=ts,
                                 perm_B=0, pair_subs=("A", "D"), grm_method="grm_from_X")


def test_multitrait_single_trait_flagged_degenerate():
    rng = np.random.default_rng(15)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    ts = FrozenTraitSet.from_list(["solo"])
    r = run_multitrait_pair_scan(subdata, pairs, {"solo": rng.standard_normal(N)}, np.arange(N),
                                 trait_set=ts, perm_B=0, pair_subs=("A", "D"),
                                 grm_method="grm_from_X")
    assert r["single_trait"] is True
    assert "DEGENERATE" in r["note"]


# ----------------------------------------------------------------------------- conditional sanity (P0-1)


def test_pair_conditional_diagnostics_math():
    rng = np.random.default_rng(21)
    n = 400
    bx = _std(rng.standard_normal(n))
    by = _std(rng.standard_normal(n))
    # inject the interaction ORTHOGONALIZED to [1, bx, by] -> the injected signal is, in-sample,
    # exactly the "pair-only" part (carried by neither single burden). This is the precise
    # operationalization of the claim the panel must back.
    inter = bx * by
    M = np.column_stack([np.ones(n), bx, by])
    inter_perp = _std(inter - M @ np.linalg.lstsq(M, inter, rcond=None)[0])
    y = rng.standard_normal(n) + 1.4 * inter_perp
    d = pair_conditional_diagnostics(np.eye(n), y, bx, by)
    # (a) exact identities: nested-F for 1 added param == Wald t^2, and the two p's coincide
    chk = d["nested_interaction_given_marginals"]["check_F_eq_t2"]
    assert chk["absdiff"] < 1e-6
    assert chk["p_F_minus_p_wald"] < 1e-9
    assert abs(d["nested_interaction_given_marginals"]["p"] - d["interaction_p_wald"]) < 1e-9
    # (b) VIF identity VIF == 1/(1-R2); near-orthogonal product -> low collinearity
    r2 = d["r2_int_given_marginals"]
    assert abs(d["vif_int"] - 1.0 / (1.0 - r2)) < 1e-9
    assert d["vif_int"] < 1.5
    # (b2) full-rank design + Frisch-Waugh-Lovell: interaction t on the marginal-orthogonal part
    # equals the full-model interaction t
    assert d["design_rank"] == 4
    assert d["residualized_interaction"]["absdiff_vs_full_t"] < 1e-6
    # (c) pair-only: the orthogonalized interaction is detected, each single-gene marginal is
    # vastly weaker than the interaction (relative, so robust to finite-sample noise)
    assert d["interaction_p_wald"] < 1e-3
    assert d["single_gene_marginal"]["p_X_alone"] > 100 * d["interaction_p_wald"]
    assert d["single_gene_marginal"]["p_Y_alone"] > 100 * d["interaction_p_wald"]


def test_pair_conditional_diagnostics_detects_collinearity():
    # positive, skewed burdens -> the product bx*by IS partly a linear combo of the marginals
    # (unlike symmetric-Gaussian burdens whose product is orthogonal to mains). Guards that the
    # projection-based R2/VIF actually picks up real collinearity.
    rng = np.random.default_rng(22)
    n = 300
    bx = rng.random(n)                                 # uniform(0,1): positive, mean ~0.5
    by = bx + 0.1 * rng.random(n)                       # correlated positive -> bx*by ~ bx^2
    y = rng.standard_normal(n)
    d = pair_conditional_diagnostics(np.eye(n), y, bx, by)
    assert d["r2_int_given_marginals"] > 0.2
    assert d["vif_int"] > 1.25


# --------------------------------------------------------------------------- covariate / PC support


def test_pairwise_pvals_covariate_none_is_bit_exact():
    # default (C=None) must equal an explicit intercept-only covariate block, exactly
    rng = np.random.default_rng(40)
    n, g = 200, 50
    Wh = np.eye(n)
    y = rng.standard_normal(n)
    BX = rng.standard_normal((n, g))
    BY = rng.standard_normal((n, g))
    from homoeogwas.interact import pairwise_pvals
    p_none = pairwise_pvals(Wh, y, BX, BY)
    p_ones = pairwise_pvals(Wh, y, BX, BY, C=np.ones((n, 1)))
    assert np.max(np.abs(p_none - p_ones)) == 0.0


def test_freedman_lane_reduces_to_yshuffle_for_intercept():
    # y* = C beta_hat + (y - C beta_hat)[perm] with C=intercept equals y[perm] (to machine eps)
    rng = np.random.default_rng(41)
    n = 200
    y = rng.standard_normal(n)
    C = np.ones((n, 1))
    b, *_ = np.linalg.lstsq(C, y, rcond=None)
    fit = C @ b
    resid = y - fit
    perm = rng.permutation(n)
    assert np.max(np.abs((fit + resid[perm]) - y[perm])) < 1e-12


def test_covariate_pc_path_runs_and_keeps_hit():
    # an injected A-D interaction is still detected after adding genotype PCs as fixed effects,
    # and the covariate policy is recorded with the right PC count.
    rng = np.random.default_rng(42)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    hit = 30
    bA = block_burden_capped(subdata["A"].X, subdata["A"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.8 * (_std(bA) * _std(bD))
    r = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                      pair_subs=("A", "D"), grm_method="grm_from_X", covariates={"n_pcs": 5})
    assert r.covariates["n_pcs"] == 5
    assert r.covariates["n_cols"] == 6                  # intercept + 5 PCs
    assert r.n_sig >= 1
    assert any(h["pair"][0] == f"g{hit}" for h in r.sig)


def test_covariate_default_none_path_unchanged():
    # passing covariates=None reproduces the no-covariate scan exactly (same p-values)
    rng = np.random.default_rng(43)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    y = rng.standard_normal(N)
    r0 = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                       pair_subs=("A", "D"), grm_method="grm_from_X")
    r1 = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                       pair_subs=("A", "D"), grm_method="grm_from_X", covariates=None)
    assert r0.pair_acat == r1.pair_acat and r0.min_p == r1.min_p
    assert r0.covariates["policy"] == "none"


def test_conditional_diagnostics_covariate_consistency():
    # C=None equals explicit intercept; PC block bumps n_covariates and keeps F == t^2
    rng = np.random.default_rng(44)
    n = 300
    bx = rng.standard_normal(n)
    by = rng.standard_normal(n)
    y = rng.standard_normal(n) + 1.5 * _std(bx) * _std(by)
    d_none = pair_conditional_diagnostics(np.eye(n), y, bx, by)
    d_ones = pair_conditional_diagnostics(np.eye(n), y, bx, by, C=np.ones((n, 1)))
    assert abs(d_none["interaction_p_wald"] - d_ones["interaction_p_wald"]) == 0.0
    assert d_none["n_covariates"] == 1
    C = np.column_stack([np.ones(n), rng.standard_normal((n, 4))])  # intercept + 4 covariates
    d_pc = pair_conditional_diagnostics(np.eye(n), y, bx, by, C=C)
    assert d_pc["n_covariates"] == 5
    chk = d_pc["nested_interaction_given_marginals"]["check_F_eq_t2"]
    assert chk["absdiff"] < 1e-6 and d_pc["nested_interaction_given_marginals"]["df"] == [1, n - 8]


def _read_tsv(path):
    rows = [ln.rstrip("\n").split("\t") for ln in path.read_text().splitlines()]
    return rows[0], rows[1:]


def test_pair_full_dump_complete_ordered_and_genelen_na(tmp_path):
    # the FULL pair-ranking dump has one row per callable pair, is ascending-p ordered, and emits
    # gene_len columns as literal NA (the engine has no coordinates -> no fabrication).
    rng = np.random.default_rng(7)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    hit = 30
    bA = block_burden_capped(subdata["A"].X, subdata["A"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.8 * (_std(bA) * _std(bD))
    dp = tmp_path / "rank.tsv"
    r = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                      pair_subs=("A", "D"), grm_method="grm_from_X", full_dump_path=str(dp))
    head, body = _read_tsv(dp)
    assert len(body) == r.G == G                              # every callable pair present
    assert head[0] == "rank" and "gene_len_pair_sum" in head
    pcol = head.index("p_interaction")
    ps = [float(row[pcol]) for row in body]
    assert ps == sorted(ps)                                  # ascending-p ordered
    assert body[0][1] == f"g{hit}"                           # top row is the injected hit
    # gene_len is NA (not fabricated); n_snp_pair equals the two per-gene counts summed
    glen = [head.index(c) for c in head if c.startswith("gene_len")]
    assert all(body[0][c] == "NA" for c in glen)
    nx, ny, ns = head.index("n_snp_A"), head.index("n_snp_D"), head.index("n_snp_pair")
    assert int(body[0][nx]) == int(body[0][ny]) == SPG and int(body[0][ns]) == 2 * SPG


def test_pair_dump_has_marginal_and_burden_exports(tmp_path):
    # the enriched dump carries single-gene marginal p (for the "invisible to
    # single-locus" contrast) + coords, and the top-K burden export is written.
    rng = np.random.default_rng(11)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    hit = 12
    bA = block_burden_capped(subdata["A"].X, subdata["A"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.8 * (_std(bA) * _std(bD))
    dp = tmp_path / "rank.tsv"
    bp = tmp_path / "burden.tsv"
    run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT",
                  perm_B=0, pair_subs=("A", "D"), grm_method="grm_from_X",
                  full_dump_path=str(dp), burden_dump_path=str(bp), top_k_burden=2)
    head, body = _read_tsv(dp)
    for c in ("chrom_x", "pos_x", "chrom_y", "pos_y", "p_marginal_x",
              "p_marginal_y", "neglog10p_marginal_x", "neglog10p_marginal_y"):
        assert c in head
    # marginal p of each gene alone is a valid probability
    pmx = head.index("p_marginal_x")
    vals = [float(r[pmx]) for r in body]
    assert all(0.0 <= v <= 1.0 for v in vals)
    # burden export: top_k_burden pairs x N samples, with the expected columns
    bhead, bbody = _read_tsv(bp)
    assert bhead[:6] == ["pair_rank", "gene_x", "gene_y", "sub_x", "sub_y",
                         "sample_row"]
    assert {"burden_x", "burden_y", "phenotype", "resid"} <= set(bhead)
    assert len(bbody) == 2 * N
    assert {int(r[0]) for r in bbody} == {0, 1}


def test_marginal_pvals_basic():
    from homoeogwas.interact import marginal_pvals
    rng = np.random.default_rng(3)
    n = 200
    Wh = np.eye(n)                         # identity whitener -> plain OLS t-test
    B = rng.standard_normal((n, 2))
    y = 0.8 * B[:, 0] + rng.standard_normal(n)   # col0 strongly associated
    pv = marginal_pvals(Wh, y, B)
    assert pv.shape == (2,)
    assert np.all((pv >= 0) & (pv <= 1))
    assert pv[0] < pv[1]                   # the associated gene has the smaller p


def test_triad_full_dump_acat_ordered_and_pairwise_columns(tmp_path):
    # the FULL triad dump is ascending-ACAT ordered with all 3 pairwise p columns + NA gene_len.
    rng = np.random.default_rng(0)
    subs = ["A", "B", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    triads = [(f"g{i}", f"g{i}", f"g{i}") for i in range(G)]
    hit = 17
    bB = block_burden_capped(subdata["B"].X, subdata["B"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.8 * (_std(bB) * _std(bD))
    dp = tmp_path / "triad_rank.tsv"
    r = run_triad_scan(subdata, triads, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                       grm_method="grm_from_X", full_dump_path=str(dp))
    head, body = _read_tsv(dp)
    assert len(body) == r["G"] == G
    for c in ("p_AB", "p_AD", "p_BD", "p_acat", "min_pair_p", "min_pair_tag"):
        assert c in head
    acol = head.index("p_acat")
    acs = [float(row[acol]) for row in body]
    assert acs == sorted(acs)
    assert body[0][1] == f"g{hit}"                           # injected triad ranks first by ACAT
    assert body[0][head.index("min_pair_tag")] == "BD"       # BD is the driving pairwise
    glen = [head.index(c) for c in head if c.startswith("gene_len")]
    assert all(body[0][c] == "NA" for c in glen)


def test_full_dump_off_writes_nothing(tmp_path):
    # default (no dump path) must not create any TSV (legacy bit-exact behaviour preserved)
    rng = np.random.default_rng(9)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    y = rng.standard_normal(N)
    run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                  pair_subs=("A", "D"), grm_method="grm_from_X")
    assert list(tmp_path.iterdir()) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --- omniB production path (encoding-invariant primary + kinship-preserving bootstrap) ---------

def _make_sub_maf(rng, n=N, g=G, spg=SPG):
    """Synthetic genotypes with every SNP common (MAF>=0.01 gate is a no-op) so the omniB
    production path (which gates burden SNPs at MAF>=0.01) retains all genes."""
    X = rng.integers(0, 3, size=(n, g * spg)).astype(float)
    gene_snp = {f"g{i}": np.arange(i * spg, (i + 1) * spg) for i in range(g)}
    return SubgenomeData(X=X, gene_snp=gene_snp, samples=[f"s{j}" for j in range(n)], chunk=None)


def test_omnib_pair_scan_detects_and_isolates_interaction():
    rng = np.random.default_rng(11)
    subdata = {"A": _make_sub_maf(rng), "D": _make_sub_maf(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    hit = 30
    bA = block_burden_capped(subdata["A"].X, subdata["A"].gene_snp[f"g{hit}"], 150, rng, minor=True)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng, minor=True)
    y = rng.standard_normal(N) + 3.0 * (_std(bA) * _std(bD))
    r = run_pair_scan_omnib(subdata, pairs, y, np.arange(N), cap=150, n_pc=3, bootstrap_B=0,
                            n_jobs=1, pair_subs=("A", "D"), grm_method="grm_from_X")
    assert r.statistic == "omniB"
    # omniB is a 3-way ACAT (minor-burden + PC1 + kernel), so a single-component injected signal is
    # diluted vs a lone burden test; assert the injected pair is the genome-wide strongest, not that
    # it clears the small-sample Bonferroni bar.
    assert tuple(r.top[0]["pair"])[0] == f"g{hit}"
    assert r.min_p < 1e-2


def test_omnib_min_snp_gate_honours_configured_threshold():
    rng = np.random.default_rng(111)
    n = 50
    subdata = {
        s: SubgenomeData(
            X=rng.integers(0, 3, size=(n, 2)).astype(float),
            gene_snp={"g": np.array([0, 1])},
            samples=[f"s{i}" for i in range(n)],
            chunk=None)
        for s in ("A", "D")
    }
    y = rng.normal(size=n)
    result = run_pair_scan_omnib(
        subdata, [("g", "g")], y, np.arange(n), bootstrap_B=0, n_jobs=1,
        pair_subs=("A", "D"), grm_method="grm_from_X", min_snp=2)
    assert result.G == 1
    with pytest.raises(ValueError, match="no homoeolog pairs retained"):
        run_pair_scan_omnib(
            subdata, [("g", "g")], y, np.arange(n), bootstrap_B=0, n_jobs=1,
            pair_subs=("A", "D"), grm_method="grm_from_X", min_snp=3)


def test_interact_schema_refuses_four_way_cli_config():
    cfg = {
        "interact": {
            "mode": "clique",
            "subgenomes": ["A", "B", "C", "D"],
        }
    }
    with pytest.raises(SystemExit, match="2-/3-subgenome subsets"):
        I.validate_interact_config(cfg)


def _canonical_group_config(subgenomes=("A", "B", "C", "D"), **updates):
    interact = {
        "mode": "group",
        "subgenomes": list(subgenomes),
        "groups": "groups.tsv",
        "statistic": "omniB",
        "hypothesis_unit": "group",
        "subset_order": 2,
        "family_scope": "primary_only",
        "primary_transform": "INT",
        "primary_multiplicity": "bootstrap_minp",
        "genotype": {sub: sub.lower() for sub in subgenomes},
        "snp_to_gene": {sub: f"n{sub.lower()}" for sub in subgenomes},
        "phenotype": "p.tsv",
        "sample_col": "sample",
        "trait": "trait",
        "burden": {"cap": 150, "min_snp": 3, "maf_min": 0.01},
        "grm": {
            "method": "grm_from_X", "maf_min": 0.01,
            "scope": "all_subgenomes",
        },
        "calibration": {"method": "bootstrap", "B": 2000, "seed": 2026},
    }
    interact.update(updates)
    return {"interact": interact}


@pytest.mark.parametrize("statistic", ["fourway", "4way", "four-way", "4_way"])
def test_group_mode_accepts_four_copies_but_refuses_fourway_statistic(statistic):
    cfg = _canonical_group_config()
    I.validate_interact_config(cfg)
    cfg["interact"]["statistic"] = statistic
    with pytest.raises(SystemExit, match="never fits a direct four-way"):
        I.validate_interact_config(cfg)


def test_group_checkpoint_config_is_strict_and_canonical_only(tmp_path):
    cfg = _canonical_group_config()
    cfg["interact"]["calibration"]["checkpoint"] = {
        "enabled": True,
        "root": str(tmp_path / "checkpoint"),
        "block_size": 25,
    }
    I.validate_interact_config(cfg)

    for bad in (True, 0, 1.5, "25"):
        invalid = copy.deepcopy(cfg)
        invalid["interact"]["calibration"]["checkpoint"]["block_size"] = bad
        with pytest.raises(SystemExit, match="checkpoint.block_size"):
            I.validate_interact_config(invalid)
    invalid = copy.deepcopy(cfg)
    invalid["interact"]["calibration"]["checkpoint"]["root"] = ""
    with pytest.raises(SystemExit, match="checkpoint.root"):
        I.validate_interact_config(invalid)
    invalid = copy.deepcopy(cfg)
    invalid["interact"]["calibration"]["checkpoint"]["enabled"] = "yes"
    with pytest.raises(SystemExit, match="checkpoint.enabled"):
        I.validate_interact_config(invalid)

    pair = _canonical_group_config(subgenomes=("A", "D"))
    pair["interact"].update({
        "mode": "pairwise",
        "pairs": pair["interact"].pop("groups"),
        "statistic": "burden",
        "primary_multiplicity": "bonferroni",
        "calibration": {"method": "permutation", "perm_B": 2000},
    })
    pair["interact"]["calibration"]["checkpoint"] = {
        "enabled": True, "root": str(tmp_path / "pair"), "block_size": 25}
    with pytest.raises(SystemExit, match="canonical group omniB"):
        I.validate_interact_config(pair)


@pytest.mark.parametrize(
    ("legacy_mode", "subgenomes", "table_key", "hypothesis_unit"),
    [
        ("pairwise", ("A", "D"), "pairs", "edge"),
        ("triad", ("A", "B", "D"), "triads", "group"),
    ],
)
def test_legacy_and_canonical_omnib_configs_normalize_identically(
        legacy_mode, subgenomes, table_key, hypothesis_unit):
    base = {
        "subgenomes": list(subgenomes),
        "statistic": "omniB",
        "genotype": {sub: sub.lower() for sub in subgenomes},
        "snp_to_gene": {sub: f"n{sub.lower()}" for sub in subgenomes},
        "phenotype": "p.tsv",
        "sample_col": "sample",
        "trait": "trait",
    }
    legacy = {
        "interact": base | {"mode": legacy_mode, table_key: "groups.tsv"}}
    canonical = {"interact": base | {
        "mode": "group", "groups": "groups.tsv",
        "hypothesis_unit": hypothesis_unit, "subset_order": 2,
        "family_scope": "primary_only",
    }}
    assert normalize_interact_config(legacy) == normalize_interact_config(canonical)


def test_cmd_interact_routes_quartet_to_one_group_fwer_family(
        monkeypatch, tmp_path):
    subs = ("A", "B", "C", "D")
    groups = tmp_path / "groups.tsv"
    groups.write_text(
        "group_id\tgene_A\tgene_B\tgene_C\tgene_D\n"
        "q1\ta1\tb1\tc1\td1\n",
        encoding="utf-8",
    )
    phenotype = tmp_path / "phenotype.tsv"
    phenotype.write_text(
        "sample\ttrait\n" + "".join(
            f"s{i}\t{i / 10}\n" for i in range(12)),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    cfg = workflow.build_interact_config(
        subgenomes=subs,
        bed_prefixes={sub: sub.lower() for sub in subs},
        snp_to_gene={sub: f"n{sub.lower()}" for sub in subs},
        phenotype=str(phenotype), sample_col="sample", trait="trait",
        out_dir=str(out_dir), groups=str(groups), perm_b=2000,
    )
    cfg["outputs"] = {
        "out_dir": str(out_dir), "full_ranking": True, "plots": False}
    checkpoint_root = out_dir / "checkpoint"
    cfg["interact"]["calibration"]["checkpoint"] = {
        "enabled": True, "root": str(checkpoint_root), "block_size": 25}
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps(cfg), encoding="utf-8")

    samples = [f"s{i}" for i in range(12)]
    monkeypatch.setattr(
        I,
        "verify_formal_launch",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_context={"raw_config_sha256": "a" * 64}),
    )
    monkeypatch.setattr(I, "preflight_interact", lambda _cfg, **_kwargs: [])
    monkeypatch.setattr(
        I, "_load_subgenome",
        lambda *_args, **_kwargs: SimpleNamespace(samples=samples),
    )
    calls = []

    def _fake_group_scan(_subdata, family, _y, _sample_idx, **kwargs):
        calls.append((family, kwargs))
        return SimpleNamespace(
            trait="", G=1, min_p=0.2, lambda_gc_obs=1.0,
            bonferroni_alpha=0.05, n_sig=0, sig=[], top=[], covariates=None,
            model_diagnostics={
                "bootstrap_fwer": {"family_id": "group"},
                "family_provenance": {
                    "group_family_sha256": "g" * 64,
                    "edge_family_sha256": "e" * 64,
                    "n_groups_raw": 1,
                    "n_unique_edges": 6,
                },
            },
        )

    monkeypatch.setattr(I, "run_group_scan_omnib", _fake_group_scan)
    rc = I.cmd_interact(SimpleNamespace(
        config=str(config), out_dir=None, n_jobs=4))

    assert rc == 0
    assert len(calls) == 1
    family, kwargs = calls[0]
    assert family.subgenomes == subs
    assert family.group_ids == ("q1",)
    assert kwargs["hypothesis_unit"] == "group"
    assert kwargs["family_scope"] == "primary_only"
    assert kwargs["bootstrap_B"] == 2000
    assert kwargs["checkpoint_dir"] == str(checkpoint_root)
    assert kwargs["checkpoint_block_size"] == 25
    payload = json.loads((out_dir / "interact_trait.json").read_text())
    provenance = payload["provenance"]
    assert provenance | {
        "mode": "group",
        "hypothesis_unit": "group",
        "subset_order": 2,
        "family_scope": "primary_only",
        "group_family_sha256": "g" * 64,
        "edge_family_sha256": "e" * 64,
        "n_groups_raw": 1,
        "n_unique_edges": 6,
        "grm_scope": "all_subgenomes",
    } == provenance
    assert list(payload["results"]) == ["INT"]
    assert payload["provenance"]["config_sha256"] == hashlib.sha256(
        config.read_bytes()).hexdigest()
    assert payload["results"]["INT"]["model_diagnostics"][
        "bootstrap_fwer"]["family_id"] == "group"


def test_quartet_formal_group_has_six_edges_and_no_fourth_order_fields(
        monkeypatch, tmp_path):
    subs = ("A", "B", "C", "D")
    family = MasterGroupFamily(
        subgenomes=subs,
        group_ids=("quartet_1",),
        genes=(("g0", "g0", "g0", "g0"),),
    )
    expanded = expand_pair_edges(family)
    assert len(expanded.edges) == 6
    edge_p = np.array([
        [0.10 + index * 0.01, 0.2, 0.3, 0.4]
        for index in range(6)
    ])
    group_p = np.array([[
        acat(edge_p[:, column]) for column in range(edge_p.shape[1])
    ]])
    scores = OmniBFamilyScores(
        edge_p=edge_p,
        group_p=group_p,
        edge_components_obs=np.tile(
            np.array([[0.1, 0.2, 0.3]]), (6, 1)),
        edge_estimable=np.ones(6, bool),
        group_estimable=np.ones(1, bool),
        W=np.eye(4), y=np.arange(4.0),
        covariance_components={
            "A": 0.15, "B": 0.15, "C": 0.15, "D": 0.15, "e": 0.4},
    )
    monkeypatch.setattr(
        I._family_score, "score_omnib_family",
        lambda *_args, **_kwargs: (scores, expanded),
    )
    ranking = tmp_path / "quartet.tsv"
    result = I.run_group_scan_omnib(
        {}, family, np.arange(4.0), np.arange(4),
        hypothesis_unit="group", family_scope="primary_only",
        transform="INT", bootstrap_B=3, full_dump_path=str(ranking),
    )

    assert result.G == 1
    assert len(result.top) == 1
    assert len(result.top[0]["edge_localization"]) == 6
    provenance = result.model_diagnostics["family_provenance"]
    assert provenance["n_groups_raw"] == 1
    assert provenance["n_unique_edges"] == 6
    assert len(provenance["group_family_sha256"]) == 64
    assert len(provenance["edge_family_sha256"]) == 64
    serialized = json.dumps(I._json_safe(result.__dict__))
    header = ranking.read_text().splitlines()[0]
    for forbidden in ("p_fourway", "A:B:C:D", "fourth_order"):
        assert forbidden not in serialized
        assert forbidden not in header


def test_cmd_interact_triad3_bypasses_group_normalization_and_route(
        monkeypatch, tmp_path):
    subs = ("A", "B", "D")
    triads = tmp_path / "triads.tsv"
    triads.write_text(
        "gene_A\tgene_B\tgene_D\na1\tb1\td1\n", encoding="utf-8")
    phenotype = tmp_path / "phenotype.tsv"
    phenotype.write_text(
        "sample\ttrait\n" + "".join(
            f"s{i}\t{i / 10}\n" for i in range(12)),
        encoding="utf-8",
    )
    cfg = {
        "interact": {
            "mode": "triad", "subgenomes": list(subs),
            "triads": str(triads), "statistic": "triad3",
            "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
            "genotype": {sub: sub.lower() for sub in subs},
            "snp_to_gene": {sub: f"n{sub.lower()}" for sub in subs},
            "phenotype": str(phenotype), "sample_col": "sample",
            "trait": "trait",
            "calibration": {"method": "bootstrap", "B": 2000, "seed": 2026},
        },
        "outputs": {
            "out_dir": str(tmp_path / "out"), "plots": False,
        },
    }
    config = tmp_path / "triad3.yaml"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    samples = [f"s{i}" for i in range(12)]
    monkeypatch.setattr(I, "preflight_interact", lambda _cfg, **_kwargs: [])
    monkeypatch.setattr(
        I, "_load_subgenome",
        lambda *_args, **_kwargs: SimpleNamespace(samples=samples),
    )
    monkeypatch.setattr(
        I, "run_group_scan_omnib",
        lambda *_args, **_kwargs: pytest.fail("triad3 entered group omniB"),
    )
    calls = []

    def _fake_triad3(_subdata, groups, _y, _sample_idx, **kwargs):
        calls.append((groups, kwargs))
        return SimpleNamespace(
            trait="", G=1, min_p=0.3, lambda_gc_obs=1.0,
            bonferroni_alpha=0.05, analytic_screen_n=0,
            analytic_screen_sig=[], n_sig=0 if kwargs["inferential"] else None,
            sig=[] if kwargs["inferential"] else None,
            bootstrap_B=kwargs["bootstrap_B"], covariates=None,
            model_diagnostics={"bootstrap_fwer": {
                "n_degenerate_replicates": 0}},
            minp_boot_emp=0.5,
        )

    monkeypatch.setattr(I, "run_triad3_scan", _fake_triad3)
    assert I.cmd_interact(SimpleNamespace(
        config=str(config), out_dir=None, n_jobs=3)) == 0
    assert len(calls) == 2
    assert all(groups == [("a1", "b1", "d1")] for groups, _ in calls)
    assert calls[0][1]["inferential"] is True
    assert calls[1][1]["inferential"] is False
    payload = json.loads(
        (tmp_path / "out" / "interact_trait.json").read_text())
    assert payload["mode"] == "triad"
    assert payload["provenance"]["statistic"] == "triad3"


def test_triad3_dispatch_executes_exact_three_copy_estimator():
    rng = np.random.default_rng(15000)
    n, group_count = 64, 2
    subdata = {
        sub: _make_sub_maf(rng, n=n, g=group_count, spg=5)
        for sub in ("A", "B", "D")
    }
    triads = [tuple(f"g{i}" for _ in subdata) for i in range(group_count)]
    result = I.run_triad3_scan(
        subdata, triads, rng.normal(size=n), np.arange(n),
        transform="INT", bootstrap_B=0, inferential=False,
        n_jobs=1, grm_method="grm_from_X", min_snp=3,
    )
    assert result.statistic == "triad3"
    assert result.G == group_count
    assert result.n_sig is None
    assert result.model_diagnostics["tested_term"] == "A:B:D"


def _minimal_interact_config(**updates):
    interact = {
        "mode": "pairwise",
        "subgenomes": ["A", "D"],
        "genotype": {"A": "a", "D": "d"},
        "snp_to_gene": {"A": "a.npz", "D": "d.npz"},
        "pairs": "pairs.tsv",
        "phenotype": "pheno.tsv",
        "sample_col": "sample",
        "trait": "trait",
    }
    interact.update(updates)
    return {"interact": interact}


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"statistic": "unknown"}, "statistic must be"),
        (
            {"statistic": "burden", "calibration": {"method": "bootstrap"}},
            "supported statistic/calibration",
        ),
        (
            {"statistic": "omniB", "weights": "weights.tsv"},
            "weights is not used",
        ),
        (
            {"statistic": "burden", "primary_weighting": "weighted"},
            "requires interact.weights",
        ),
        (
            {"statistic": "burden", "calibration": {"perm_B": -1}},
            "must be an integer >= 0",
        ),
    ],
)
def test_interact_schema_rejects_inference_config_mismatches(updates, message):
    with pytest.raises(SystemExit, match=message):
        I.validate_interact_config(_minimal_interact_config(**updates))


def test_interact_schema_accepts_explicit_frozen_burden_config():
    cfg = _minimal_interact_config(
        statistic="burden",
        primary_transform="INT",
        primary_weighting="unweighted",
        calibration={"method": "permutation", "perm_B": 500},
    )
    I.validate_interact_config(cfg)


@pytest.mark.parametrize("perm_B", [0, 18, 998])
def test_interact_schema_rejects_underresolved_formal_burden_permutation_minp(perm_B):
    cfg = _minimal_interact_config(
        statistic="burden",
        primary_multiplicity="permutation_minp",
        calibration={"method": "permutation", "perm_B": perm_B},
    )
    with pytest.raises(SystemExit, match="requires interact.calibration.perm_B >= 999"):
        I.validate_interact_config(cfg)


def test_interact_schema_accepts_burden_permutation_minp_qa_only():
    cfg = _minimal_interact_config(
        statistic="burden",
        primary_multiplicity="permutation_minp",
        calibration={"method": "permutation", "perm_B": 19, "qa_only": True},
    )
    I.validate_interact_config(cfg)


def test_interact_schema_accepts_experimental_triad3_config():
    cfg = {
        "interact": {
            "mode": "triad",
            "subgenomes": ["A", "B", "D"],
            "statistic": "triad3",
            "genotype": {"A": "a", "B": "b", "D": "d"},
            "snp_to_gene": {
                "A": "a.npz", "B": "b.npz", "D": "d.npz"},
            "triads": "triads.tsv",
            "phenotype": "pheno.tsv",
            "sample_col": "sample",
            "trait": "trait",
            "calibration": {"method": "bootstrap", "B": 999},
        }
    }
    I.validate_interact_config(cfg)


@pytest.mark.parametrize("B", [0, 18, 998])
def test_interact_schema_rejects_underresolved_triad3_bootstrap(B):
    cfg = {
        "interact": {
            "mode": "triad",
            "subgenomes": ["A", "B", "C"],
            "statistic": "triad3",
            "genotype": {"A": "a", "B": "b", "C": "c"},
            "snp_to_gene": {"A": "a.npz", "B": "b.npz", "C": "c.npz"},
            "triads": "triads.tsv",
            "phenotype": "pheno.tsv",
            "sample_col": "sample",
            "trait": "trait",
            "calibration": {"method": "bootstrap", "B": B},
        }
    }
    with pytest.raises(SystemExit, match="requires interact.calibration.B >= 999"):
        I.validate_interact_config(cfg)


def test_interact_schema_accepts_explicit_triad3_qa_only():
    cfg = {
        "interact": {
            "mode": "triad",
            "subgenomes": ["A", "B", "C"],
            "statistic": "triad3",
            "genotype": {"A": "a", "B": "b", "C": "c"},
            "snp_to_gene": {"A": "a.npz", "B": "b.npz", "C": "c.npz"},
            "triads": "triads.tsv",
            "phenotype": "pheno.tsv",
            "sample_col": "sample",
            "trait": "trait",
            "calibration": {
                "method": "bootstrap", "B": 19, "qa_only": True},
        }
    }
    I.validate_interact_config(cfg)


def test_interact_schema_rejects_triad3_dominance_adjust():
    cfg = {
        "interact": {
            "mode": "triad",
            "subgenomes": ["A", "B", "C"],
            "statistic": "triad3",
            "genotype": {"A": "a", "B": "b", "C": "c"},
            "snp_to_gene": {"A": "a.npz", "B": "b.npz", "C": "c.npz"},
            "triads": "triads.tsv",
            "phenotype": "pheno.tsv",
            "sample_col": "sample",
            "trait": "trait",
            "burden": {"dominance_adjust": True},
            "calibration": {"method": "bootstrap", "B": 999},
        }
    }
    with pytest.raises(SystemExit, match="dominance_adjust is not implemented"):
        I.validate_interact_config(cfg)


def test_interact_schema_rejects_triad3_outside_triad_mode():
    with pytest.raises(SystemExit, match="triad3 requires mode=triad"):
        I.validate_interact_config(
            _minimal_interact_config(statistic="triad3"))


def test_omnib_strictly_invariant_to_ref_alt_recoding():
    # recode every SNP dosage x -> 2 - x; omniB per pair must be unchanged (the paper's #1 claim)
    rng = np.random.default_rng(12)
    subdata = {"A": _make_sub_maf(rng), "D": _make_sub_maf(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    bA = block_burden_capped(subdata["A"].X, subdata["A"].gene_snp["g10"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp["g10"], 150, rng)
    y = rng.standard_normal(N) + 1.5 * (_std(bA) * _std(bD))
    r0 = run_pair_scan_omnib(subdata, pairs, y, np.arange(N), cap=150, bootstrap_B=0, n_jobs=1,
                             pair_subs=("A", "D"), grm_method="grm_from_X")
    flip = {s: SubgenomeData(X=2.0 - d.X, gene_snp=d.gene_snp, samples=d.samples, chunk=None)
            for s, d in subdata.items()}
    r1 = run_pair_scan_omnib(flip, pairs, y, np.arange(N), cap=150, bootstrap_B=0, n_jobs=1,
                             pair_subs=("A", "D"), grm_method="grm_from_X")
    p0 = {tuple(h["pair"]): h["p"] for h in r0.top}
    p1 = {tuple(h["pair"]): h["p"] for h in r1.top}
    # minor-burden and PC1 are exactly invariant; the low-rank kernel component drifts by a tiny
    # amount from tied-singular-value rotation on recoding, so omniB is invariant to numerical
    # tolerance (the honest "strictly invariant up to float/SVD tolerance" claim), not bit-exact.
    assert r0.min_p == pytest.approx(r1.min_p, rel=1e-5)
    for k in p0:
        assert p0[k] == pytest.approx(p1[k], rel=1e-5)


def test_omnib_bootstrap_reports_fwer_and_tail_excess():
    rng = np.random.default_rng(13)
    subdata = {"A": _make_sub_maf(rng), "D": _make_sub_maf(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    y = rng.standard_normal(N)                                   # pure null
    r = run_pair_scan_omnib(subdata, pairs, y, np.arange(N), cap=150, bootstrap_B=50,
                            bootstrap_seed=1, n_jobs=1, pair_subs=("A", "D"),
                            grm_method="grm_from_X")
    assert r.calibration_method == "bootstrap"
    assert r.bootstrap_B == 50
    assert 0.0 < r.minp_boot_emp <= 1.0                          # experiment-wide FWER is a valid p
    assert "n_below_0.01" in r.tail_excess
    assert 0.0 < r.tail_excess["n_below_0.01"]["empirical_p"] <= 1.0


def test_omnib_bootstrap_whitens_once_never_refits_null():
    # the critical pitfall guard: the kinship-preserving bootstrap must fit the null LMM / build the
    # whitener EXACTLY once (never refit REML on a bootstrap phenotype, which would collapse the
    # variance components as the legacy permutation does).
    import homoeogwas.interact as I

    rng = np.random.default_rng(14)
    subdata = {"A": _make_sub_maf(rng), "D": _make_sub_maf(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    y = rng.standard_normal(N)
    calls = {"n": 0}
    orig = I.null_lmm_fit

    def _counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    I.null_lmm_fit = _counting
    try:
        run_pair_scan_omnib(subdata, pairs, y, np.arange(N), cap=150, bootstrap_B=40, n_jobs=1,
                            pair_subs=("A", "D"), grm_method="grm_from_X")
    finally:
        I.null_lmm_fit = orig
    # one fit for the scan whitener + one inside null_replicates for the bootstrap null: both
    # one-time, neither scales with B (B=40 here), which is the guarantee that matters.
    assert calls["n"] <= 2


def test_omnib_clique_triad_is_acat_of_pairwise_omnib():
    rng = np.random.default_rng(15)
    subdata = {s: _make_sub_maf(rng) for s in ("A", "B", "D")}
    groups = [(f"g{i}", f"g{i}", f"g{i}") for i in range(G)]
    hit = 20
    bB = block_burden_capped(subdata["B"].X, subdata["B"].gene_snp[f"g{hit}"], 150, rng, minor=True)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng, minor=True)
    y = rng.standard_normal(N) + 3.0 * (_std(bB) * _std(bD))     # inject on the B-D pair (minor burden)
    r = run_clique_scan_omnib(subdata, groups, y, np.arange(N), cap=150, bootstrap_B=0, n_jobs=1,
                              grm_method="grm_from_X")
    assert r.statistic == "omniB"
    # group p = ACAT of the 3 pairwise omniBs (only B-D carries signal) -> further diluted; assert
    # the injected group is the strongest.
    assert tuple(r.top[0]["pair"])[0] == f"g{hit}"


# --- multiplicity / estimability regression suite (engine v2) -------------------------------------

def _basis(n, k):
    e = np.zeros(n)
    e[k] = 1.0
    return e


def test_fwl_full_rank_reproduces_frozen_wheat_statistic():
    # orthonormal construction with the wheat scan geometry (n=827, 3 nuisance columns, df=823):
    # the frozen headline t must map to the frozen headline p through the new estimator.
    n, t_target = 827, 4.748395874321968
    Z = np.column_stack([_basis(n, k) for k in range(3)])
    x = _basis(n, 3)
    y = (t_target / np.sqrt(823)) * x + _basis(n, 4)
    p, why, ok = I._coef_pval_fwl(Z, x, y)
    assert why is None and ok
    assert abs(p / 2.418077841738446e-06 - 1.0) < 1e-9
    # and it agrees with a plain OLS t-test on the assembled design
    X = np.column_stack([Z, x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt((resid @ resid) / 823 * np.linalg.inv(X.T @ X)[3, 3])
    p_ref = float(2 * stats.t.sf(abs(beta[3] / se), 823))
    assert abs(np.log10(p / p_ref)) < 1e-9


def test_fwl_explicit_residual_survives_cancellation():
    # on a near-perfect fit, rss as y'y - (U'y)'(U'y) cancels catastrophically and goes NEGATIVE;
    # clamping that to zero turned a perfectly valid model into NaN. The explicit residual does not.
    n, found = 400, 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        zA = _std(rng.standard_normal(n))
        zB = _std(rng.standard_normal(n))
        X = np.column_stack([np.ones(n), zA, zB, zA * zB])
        y = X @ np.array([1.0, 2.0, -1.5, 0.7]) + 1e-8 * rng.standard_normal(n)
        d = np.sqrt(np.einsum("ij,ij->j", X, X))
        U, _s, _Vt = np.linalg.svd(X / d, full_matrices=False)
        Uy = U.T @ y
        rss_subtractive = float(y @ y) - float(Uy @ Uy)
        rss_direct = float((y - U @ Uy) @ (y - U @ Uy))
        if rss_subtractive > 0:
            continue
        found += 1
        assert rss_direct > 0
        p, why, ok = I._coef_pval_fwl(X[:, :3], X[:, 3], y)
        assert why is None and ok and np.isfinite(p)   # explicit residual keeps the model estimable
    assert found > 0, "no cancellation case found; the construction stopped exercising the bug"


def test_fwl_duplicate_nuisance_keeps_estimable_target():
    # two identical nuisance columns: the full design is rank deficient but the target is not, so
    # gating on the whole design's condition number would discard a valid test
    rng = np.random.default_rng(1)
    n = 400
    z = _std(rng.standard_normal(n))
    x = _std(rng.standard_normal(n))
    y = rng.standard_normal(n) + 0.25 * x
    Z = np.column_stack([np.ones(n), z, z])
    p, why, ok = I._coef_pval_fwl(Z, x, y)
    assert why is None and ok and np.isfinite(p)
    Xc = np.column_stack([np.ones(n), z, x])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    se = np.sqrt((resid @ resid) / (n - 3) * np.linalg.inv(Xc.T @ Xc)[2, 2])
    p_ref = float(2 * stats.t.sf(abs(beta[2] / se), n - 3))
    assert abs(np.log10(p / p_ref)) < 1e-9


def test_fwl_target_inside_nuisance_span_is_not_estimable():
    rng = np.random.default_rng(2)
    n = 200
    z = _std(rng.standard_normal(n))
    Z = np.column_stack([np.ones(n), z])
    p, why, ok = I._coef_pval_fwl(Z, 2.0 * z, rng.standard_normal(n))
    assert np.isnan(p) and why == "target_nonestimable" and not ok


def test_pairwise_reports_mask_and_never_returns_p_one_for_degenerate():
    # gene 0 has an interaction column inside the span of [1, bX, bY]; gene 1 is ordinary
    rng = np.random.default_rng(3)
    n = 120
    bx = np.column_stack([_std(rng.standard_normal(n)), _std(rng.standard_normal(n))])
    by = bx.copy()
    by[:, 1] = _std(rng.standard_normal(n))
    bx[:, 0] = np.repeat([0.0, 1.0], n // 2)             # binary => bX*bY == bX when bY == bX
    by[:, 0] = bx[:, 0]
    y = rng.standard_normal(n)
    pv, diag = pairwise_pvals(np.eye(n), y, bx, by, return_diag=True)
    assert np.isnan(pv[0]) and not diag["estimable"][0]
    assert diag["exclusions"][0] == "target_nonestimable"
    assert np.isfinite(pv[1]) and diag["n_planned"] == 2


def test_top_lists_never_contain_nonfinite_p():
    pv = np.array([np.nan, 0.3, np.nan, 0.1])
    est = np.isfinite(pv)
    order = [int(i) for i in np.argsort(np.where(est, pv, np.inf)) if est[i]]
    top = [float(pv[i]) for i in order[:5]]
    assert top == [0.1, 0.3]


def test_json_strict_round_trip_maps_nonfinite_to_null():
    payload = dict(a=float("nan"), b=[np.float64("inf"), -np.inf, np.int64(3), np.bool_(True)],
                   c=dict(d=(np.nan, 0.5)), e=np.array([1.0, np.nan]))
    txt = json.dumps(I._json_safe(payload), allow_nan=False)
    assert "NaN" not in txt and "Infinity" not in txt
    back = json.loads(txt)
    assert back["a"] is None and back["b"][:2] == [None, None]
    assert back["b"][2] == 3 and back["b"][3] is True
    assert back["c"]["d"] == [None, 0.5] and back["e"] == [1.0, None]


def _fake_clique_scan(monkeypatch, pmap, **kw):
    """Run run_clique_scan with the per-contrast p-values replaced by fixed vectors."""
    def _fake(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        key = (BX.shape[1], float(BX[0, 0]), float(BY[0, 0]))
        pv = np.asarray(pmap[key], float)
        if return_diag:
            return pv, dict(n_planned=pv.size, estimable=np.isfinite(pv), exclusions={}, failures={})
        return pv
    monkeypatch.setattr(I, "pairwise_pvals", _fake)
    rng = np.random.default_rng(0)
    subs = kw.pop("subs", ["A", "B", "D"])
    g = kw.pop("g", 10)
    subdata = {s: _make_sub(rng, g=g) for s in subs}
    groups = [(f"g{i}",) * len(subs) for i in range(g)]
    return run_clique_scan(subdata, groups, rng.standard_normal(N), np.arange(N),
                           cap=150, transform="INT", perm_B=0, grm_method="grm_from_X", **kw)


def test_p_between_local_and_global_threshold_is_not_globally_rejected(monkeypatch):
    # G=10, K=3 => global 0.05/30 = 1.667e-3, within-contrast 0.05/10 = 5e-3.
    # p = 3e-3 sits strictly between: it must be exploratory-only, never a top-level hit.
    g, k = 10, 3
    base = np.full(g, 0.5)
    ab = base.copy()
    ab[2] = 3e-3
    pv_by_call = [ab, base.copy(), base.copy()]
    calls = {"i": 0}

    def _fake(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        pv = pv_by_call[calls["i"] % len(pv_by_call)]
        calls["i"] += 1
        if return_diag:
            return pv, dict(n_planned=pv.size, estimable=np.isfinite(pv), exclusions={}, failures={})
        return pv

    monkeypatch.setattr(I, "pairwise_pvals", _fake)
    rng = np.random.default_rng(0)
    subdata = {s: _make_sub(rng, g=g) for s in ("A", "B", "D")}
    groups = [(f"g{i}",) * 3 for i in range(g)]
    r = run_clique_scan(subdata, groups, rng.standard_normal(N), np.arange(N), cap=150,
                        transform="INT", perm_B=0, grm_method="grm_from_X")
    ab_res = r["pairwise"]["AB"]
    assert ab_res["bonferroni_alpha"] == pytest.approx(0.05 / (g * k))
    assert ab_res["n_sig"] == 0 and ab_res["sig"] == []
    exp = ab_res["exploratory_within_contrast"]
    assert exp["n_below_per_contrast_alpha"] == 1
    assert exp["per_test_alpha"] == pytest.approx(0.05 / g)
    # the exploratory block must NOT expose a second selectable rejection set
    assert "sig" not in exp and "n_sig" not in exp
    assert ab_res["min_p_adjusted_bonferroni"] == pytest.approx(3e-3 * g * k)
    assert r["families"]["pairwise_all"]["n_tests"] == g * k


def test_contrast_omnibus_is_corrected_across_k(monkeypatch):
    g = 10
    strong = np.full(g, 0.5)
    strong[0] = 1e-4
    calls = {"i": 0}
    seq = [strong, np.full(g, 0.5), np.full(g, 0.5)]

    def _fake(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        pv = seq[calls["i"] % 3]
        calls["i"] += 1
        if return_diag:
            return pv, dict(n_planned=pv.size, estimable=np.isfinite(pv), exclusions={}, failures={})
        return pv

    monkeypatch.setattr(I, "pairwise_pvals", _fake)
    rng = np.random.default_rng(0)
    subdata = {s: _make_sub(rng, g=g) for s in ("A", "B", "D")}
    groups = [(f"g{i}",) * 3 for i in range(g)]
    r = run_clique_scan(subdata, groups, rng.standard_normal(N), np.arange(N), cap=150,
                        transform="INT", perm_B=0, grm_method="grm_from_X")
    for tag in ("AB", "AD", "BD"):
        pw = r["pairwise"][tag]
        assert pw["acat_adjusted_bonferroni"] == pytest.approx(min(pw["acat"] * 3, 1.0))
        assert pw["acat_family_id"] == "contrast_omnibus"
    assert r["families"]["contrast_omnibus"]["per_test_alpha"] == pytest.approx(0.05 / 3)


def test_group_omnibus_is_reported_as_an_inference(monkeypatch):
    g = 10
    hit = np.full(g, 0.5)
    hit[4] = 1e-6
    calls = {"i": 0}

    def _fake(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        pv = hit if calls["i"] % 3 == 0 else np.full(g, 0.5)
        calls["i"] += 1
        if return_diag:
            return pv, dict(n_planned=pv.size, estimable=np.isfinite(pv), exclusions={}, failures={})
        return pv

    monkeypatch.setattr(I, "pairwise_pvals", _fake)
    rng = np.random.default_rng(0)
    subdata = {s: _make_sub(rng, g=g) for s in ("A", "B", "D")}
    groups = [(f"g{i}",) * 3 for i in range(g)]
    r = run_clique_scan(subdata, groups, rng.standard_normal(N), np.arange(N), cap=150,
                        transform="INT", perm_B=0, grm_method="grm_from_X")
    go = r["group_omnibus"]
    assert go["bonferroni_alpha"] == pytest.approx(0.05 / g)
    assert go["n_sig"] == 1 and tuple(go["sig"][0]["triad"])[0] == "g4"
    assert go["min_p_adjusted_bonferroni"] == pytest.approx(min(go["min_p"] * g, 1.0))
    assert r["inference_plan"]["primary_family"] == "group_omnibus"


def test_k1_pair_scan_threshold_and_family_unchanged():
    rng = np.random.default_rng(5)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    r = run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=0, pair_subs=("A", "D"), grm_method="grm_from_X")
    assert r.bonferroni_alpha == pytest.approx(0.05 / r.G)
    assert r.n_planned == r.G and r.n_valid + r.n_unestimable == r.G
    assert r.estimability["target_estimability_rtol"] == pytest.approx(1.4901161193847656e-08)


def test_weight_rescaling_leaves_results_invariant():
    rng = np.random.default_rng(6)
    subs = ["A", "B", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    triads = [(f"g{i}",) * 3 for i in range(G)]
    y = rng.standard_normal(N)
    wraw = {t: 0.5 + (i % 4) for i, t in enumerate(triads)}
    out = []
    for scale in (1.0, 100.0):
        out.append(run_clique_scan(subdata, triads, y, np.arange(N), cap=150, transform="INT",
                                   perm_B=0, grm_method="grm_from_X",
                                   triad_weights={t: v * scale for t, v in wraw.items()}))
    for tag in ("AB", "AD", "BD"):
        a, b = out[0]["pairwise"][tag]["weighted"], out[1]["pairwise"][tag]["weighted"]
        assert a["acat_weighted"] == pytest.approx(b["acat_weighted"], rel=1e-12)
        assert a["n_rejected_familywise"] == b["n_rejected_familywise"]
    assert out[0]["weighted"]["audit"]["weight_mean"] == pytest.approx(1.0)


def test_invalid_weights_raise_instead_of_being_rewritten():
    rng = np.random.default_rng(7)
    subdata = {s: _make_sub(rng) for s in ("A", "B", "D")}
    triads = [(f"g{i}",) * 3 for i in range(G)]
    bad = {t: 1.0 for t in triads}
    bad[triads[0]] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        run_clique_scan(subdata, triads, rng.standard_normal(N), np.arange(N), cap=150,
                        transform="INT", perm_B=0, grm_method="grm_from_X", triad_weights=bad)


def test_permutation_family_is_frozen_to_the_observed_mask(monkeypatch):
    seen = {"masks": []}
    orig = I.pairwise_pvals

    def _spy(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        seen["masks"].append(None if fixed_mask is None else np.asarray(fixed_mask).copy())
        return orig(Wh, y, BX, BY, C=C, dominance_adjust=dominance_adjust,
                    fixed_mask=fixed_mask, return_diag=return_diag)

    monkeypatch.setattr(I, "pairwise_pvals", _spy)
    rng = np.random.default_rng(8)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150, transform="INT",
                  perm_B=3, n_jobs=1, pair_subs=("A", "D"), grm_method="grm_from_X")
    # the observed scan and all three replicates use ONE raw-design mask
    assert len(seen["masks"]) == 4
    assert all(m is not None for m in seen["masks"])
    assert all(np.array_equal(m, seen["masks"][0]) for m in seen["masks"])


def test_multitrait_missing_component_is_not_folded_to_one(monkeypatch):
    calls = {"i": 0}

    def _fake(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        pv = np.full(BX.shape[1], 0.5)
        pv[0] = 0.02 if calls["i"] == 0 else np.nan     # trait 2 loses pair 0
        calls["i"] += 1
        if fixed_mask is not None:
            pv = np.where(fixed_mask, pv, np.nan)
        if return_diag:
            return pv, dict(n_planned=pv.size, estimable=np.isfinite(pv), exclusions={}, failures={})
        return pv

    monkeypatch.setattr(I, "pairwise_pvals", _fake)
    # pair 1 is genuinely non-estimable on the RAW design, so it never enters the family
    orig_mask = I.pairwise_design_mask

    def _mask(BX, BY, C=None, dominance_adjust=False):
        m, e = orig_mask(BX, BY, C=C, dominance_adjust=dominance_adjust)
        m = m.copy()
        m[1] = False
        e = dict(e)
        e[1] = "target_nonestimable"
        return m, e

    monkeypatch.setattr(I, "pairwise_design_mask", _mask)
    rng = np.random.default_rng(9)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    ts = FrozenTraitSet.from_list(["t1", "t2"])
    with pytest.raises(ValueError, match="lost their statistic after whitening"):
        run_multitrait_pair_scan(
            subdata, pairs, {"t1": rng.standard_normal(N), "t2": rng.standard_normal(N)},
            np.arange(N), trait_set=ts, cap=150, transform="INT", perm_B=0, pair_subs=("A", "D"),
            grm_method="grm_from_X")


def test_estimability_reasons_are_all_response_independent():
    # the mask is only outcome-independent if every reason that can clear it is decided without y
    assert set(I.DESIGN_EXCLUSION_REASONS) == {
        "nonfinite_design", "zero_target", "target_nonestimable", "insufficient_df"}
    # a failed decomposition means estimability was NOT DETERMINED, so it must not silently retire
    # a hypothesis, and a response-dependent failure is an analysis error, not an exclusion
    for bad in ("zero_residual_variance", "nonfinite_statistic", "svd_failed"):
        assert bad not in I.DESIGN_EXCLUSION_REASONS


def test_perfect_fit_underflows_to_zero_rather_than_failing():
    # a p that underflows is a real tail probability, not a missing test: it must stay numeric
    n = 60
    Z = np.ones((n, 1))
    x = _std(np.random.default_rng(21).standard_normal(n))
    p, why, ok = I._coef_pval_fwl(Z, x, 3.0 * x + np.ones(n))
    assert why is None and ok and p == 0.0 and not np.isnan(p)


def test_scan_raises_rather_than_shrinking_the_family_on_a_response_failure(monkeypatch):
    def _fake(Zw, xw, yw, *, target_rtol=I.TARGET_ESTIMABILITY_RTOL):
        return float("nan"), "zero_residual_variance", True

    monkeypatch.setattr(I, "_coef_pval_fwl", _fake)
    rng = np.random.default_rng(22)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    with pytest.raises(ValueError, match="lost their statistic after whitening"):
        run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=0, pair_subs=("A", "D"), grm_method="grm_from_X")


def test_degenerate_permutation_replicate_counts_as_maximally_extreme(monkeypatch):
    # a replicate that loses a mask-true statistic must be kept and counted as extreme; deleting it
    # would remove exactly the tail of the null and shrink the empirical p
    rng = np.random.default_rng(23)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    orig = I.pairwise_pvals
    state = {"n": 0}

    def _spy(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        out = orig(Wh, y, BX, BY, C=C, dominance_adjust=dominance_adjust,
                   fixed_mask=fixed_mask, return_diag=return_diag)
        if not return_diag:                       # permutation call (observed asks for diagnostics)
            state["n"] += 1
            if state["n"] <= 2:                   # break two replicates
                out = out.copy()
                out[0] = np.nan
        return out

    monkeypatch.setattr(I, "pairwise_pvals", _spy)
    r = run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=6, n_jobs=1, pair_subs=("A", "D"),
                      grm_method="grm_from_X")
    assert r.permutation["n_used"] == 6           # nothing deleted
    assert r.permutation["n_degenerate"] == 2
    assert r.permutation["status"] == "completed_with_degenerate_replicates"
    # the two degenerate replicates enter the null at min-p = 0, so they always count as <= observed
    assert r.minp_perm_emp >= 3 / 7


def test_weight_normalisation_survives_overflow_prone_magnitudes():
    big = [1e300, 2e300, 3e300]
    w = I._normalize_weights(big, 3, "test weights")
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(3.0)
    small = I._normalize_weights([1.0, 2.0, 3.0], 3, "test weights")
    assert np.allclose(w, small, rtol=1e-12)
    with pytest.raises(ValueError, match="all zero"):
        I._normalize_weights([0.0, 0.0], 2, "test weights")


def test_weighted_block_is_marked_exploratory_and_splits_family_ids():
    rng = np.random.default_rng(24)
    subs = ["A", "B", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    triads = [(f"g{i}",) * 3 for i in range(G)]
    r = run_clique_scan(subdata, triads, rng.standard_normal(N), np.arange(N), cap=150,
                        transform="INT", perm_B=0, grm_method="grm_from_X",
                        triad_weights={t: 1.0 + (i % 3) for i, t in enumerate(triads)})
    assert r["inference_plan"]["primary_weighting"] == "unweighted"
    wb = r["pairwise"]["BD"]["weighted"]
    assert wb["role"] == "exploratory_no_rejections_emitted"
    assert wb["sig"] is None and wb["n_rejected_familywise"] is None
    assert wb["acat_family_id"] == "contrast_omnibus"
    assert wb["bonferroni_family_id"] == "pairwise_all"
    assert "family_id" not in wb


def test_multitrait_permutation_family_is_frozen(monkeypatch):
    seen = []
    orig = I.pairwise_pvals

    def _spy(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        seen.append(None if fixed_mask is None else fixed_mask.copy())
        return orig(Wh, y, BX, BY, C=C, dominance_adjust=dominance_adjust,
                    fixed_mask=fixed_mask, return_diag=return_diag)

    monkeypatch.setattr(I, "pairwise_pvals", _spy)
    rng = np.random.default_rng(25)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    ts = FrozenTraitSet.from_list(["t1", "t2"])
    r = run_multitrait_pair_scan(
        subdata, pairs, {"t1": rng.standard_normal(N), "t2": rng.standard_normal(N)},
        np.arange(N), trait_set=ts, cap=150, transform="INT", perm_B=2, n_jobs=1,
        pair_subs=("A", "D"), grm_method="grm_from_X")
    # one raw-design mask is shared by both observed traits and all four permutation calls
    assert len(seen) == 6 and all(m is not None for m in seen)
    assert all(np.array_equal(m, seen[0]) for m in seen)
    assert r["permutation"]["n_degenerate"] == 0
    assert r["estimability"]["decided_on"] == "raw_design"


def test_permutation_cutoff_is_an_exact_order_statistic():
    # an interpolated quantile can disagree with the plus-one empirical p; the order statistic
    # cannot. With B replicates the cutoff is the k-th smallest, k = floor(alpha*(B+1)).
    rng = np.random.default_rng(31)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    r = run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=39, n_jobs=1, pair_subs=("A", "D"),
                      grm_method="grm_from_X")
    B = r.permutation["n_used"]
    k = int(np.floor(0.05 * (B + 1)))
    assert k == 2 and B == 39
    # the two criteria must agree by construction
    reject_by_cutoff = r.min_p < r.minp_perm_threshold
    reject_by_emp = r.minp_perm_emp <= 0.05
    assert reject_by_cutoff == reject_by_emp


def test_zero_estimable_pairs_raises_instead_of_reporting_the_smallest_p(monkeypatch):
    # with no testable hypothesis, "mp <= nan" is all-false and the empirical p would come out at
    # 1/(B+1) -- the most significant value possible
    monkeypatch.setattr(I, "pairwise_design_mask",
                        lambda BX, BY, C=None, dominance_adjust=False:
                        (np.zeros(BX.shape[1], bool), {i: "target_nonestimable"
                                                       for i in range(BX.shape[1])}))
    rng = np.random.default_rng(32)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    with pytest.raises(ValueError, match="no estimable pair"):
        run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=0, pair_subs=("A", "D"), grm_method="grm_from_X")


def test_design_mask_is_computed_on_the_raw_design():
    # the mask must not move when the phenotype (and hence the fitted whitener) changes
    rng = np.random.default_rng(33)
    n, g = 200, 5
    BX = np.column_stack([_std(rng.standard_normal(n)) for _ in range(g)])
    BY = BX.copy()
    BY[:, 2] = BX[:, 2]                                # duplicate -> product is still estimable
    m1, e1 = I.pairwise_design_mask(BX, BY)
    m2, e2 = I.pairwise_design_mask(BX, BY)
    assert np.array_equal(m1, m2) and e1 == e2
    # a target that IS in the nuisance span is excluded, with a design-only reason
    BXc = BX.copy()
    BYc = np.ones((n, g))                              # product == BX column => in span([1, bX])
    mc, ec = I.pairwise_design_mask(BXc, BYc)
    assert not mc.any()
    assert set(ec.values()) <= set(I.DESIGN_EXCLUSION_REASONS)


def test_non_degenerate_exception_is_not_laundered_into_null_evidence(monkeypatch):
    # only a genuine numerical degeneracy may become the extreme tuple; a programming error must
    # surface rather than silently count as a maximally extreme null replicate
    rng = np.random.default_rng(34)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    orig = I.pairwise_pvals

    def _boom(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        if not return_diag:
            raise RuntimeError("injected bug")
        return orig(Wh, y, BX, BY, C=C, dominance_adjust=dominance_adjust,
                    fixed_mask=fixed_mask, return_diag=return_diag)

    monkeypatch.setattr(I, "pairwise_pvals", _boom)
    with pytest.raises(RuntimeError, match="injected bug"):
        run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=2, n_jobs=1, pair_subs=("A", "D"),
                      grm_method="grm_from_X")


def test_weighted_primary_without_weights_is_rejected():
    # suppressing the unweighted rejections while no weighted procedure can run would leave the
    # scan with no primary analysis at all
    rng = np.random.default_rng(41)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    with pytest.raises(ValueError, match="requires pair_weights"):
        run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=0, pair_subs=("A", "D"), grm_method="grm_from_X",
                      primary_weighting="weighted")
    subs3 = {s: _make_sub(rng) for s in ("A", "B", "D")}
    triads = [(f"g{i}",) * 3 for i in range(G)]
    with pytest.raises(ValueError, match="requires triad_weights"):
        run_clique_scan(subs3, triads, rng.standard_normal(N), np.arange(N), cap=150,
                        transform="INT", perm_B=0, grm_method="grm_from_X",
                        primary_weighting="weighted")


def test_weighted_primary_decides_the_group_omnibus_too():
    # the declared primary family must actually be evaluated under the declared weighting
    rng = np.random.default_rng(42)
    subs = ["A", "B", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    triads = [(f"g{i}",) * 3 for i in range(G)]
    hit = 17
    bB = block_burden_capped(subdata["B"].X, subdata["B"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.8 * (_std(bB) * _std(bD))
    w = {t: (5.0 if i == hit else 1.0) for i, t in enumerate(triads)}
    r = run_clique_scan(subdata, triads, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                        grm_method="grm_from_X", triad_weights=w, primary_weighting="weighted")
    go = r["group_omnibus"]
    assert go["weighting"] == "weighted"
    assert go["n_sig"] >= 1 and all("weight" in h for h in go["sig"])
    assert any(tuple(h["triad"])[0] == f"g{hit}" for h in go["sig"])


def test_permutation_threshold_publishes_its_comparator():
    rng = np.random.default_rng(43)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    r = run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=39, n_jobs=1, pair_subs=("A", "D"),
                      grm_method="grm_from_X")
    assert r.minp_perm_threshold_comparator == "strict_less_than"
    # Bonferroni is primary by default, so min-P emits no rejection decision
    assert r.minp_perm_rejected is None
    r2 = run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                       transform="INT", perm_B=39, n_jobs=1, pair_subs=("A", "D"),
                       grm_method="grm_from_X", primary_multiplicity="permutation_minp")
    assert r2.sig is None and r2.n_sig is None       # the non-primary procedure emits nothing
    assert r2.minp_perm_rejected == (r2.minp_perm_emp <= 0.05)
    assert r2.minp_perm_rejected == (r2.min_p < r2.minp_perm_threshold)


def test_weighted_primary_headline_uses_the_weighted_adjusted_p():
    # the reported group-omnibus minimum must be the minimum of the PRIMARY score; ranking by raw p
    # can headline a different group than the one the weighted rule actually rejects
    rng = np.random.default_rng(44)
    subs = ["A", "B", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    triads = [(f"g{i}",) * 3 for i in range(G)]
    hit = 11
    bB = block_burden_capped(subdata["B"].X, subdata["B"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.6 * (_std(bB) * _std(bD))
    w = {t: (20.0 if i == hit else 1.0) for i, t in enumerate(triads)}
    r = run_clique_scan(subdata, triads, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                        grm_method="grm_from_X", triad_weights=w, primary_weighting="weighted")
    go = r["group_omnibus"]
    assert go["bonferroni_alpha"] is None                    # a single cutoff does not exist
    assert go["per_test_alpha_rule"].startswith("0.05*w_i/G")
    # the headline adjusted p is p*G/w, so an up-weighted group reports a SMALLER adjusted p than
    # the unweighted rule would give it
    top = go["top"][0]
    assert top["p_adjusted_bonferroni"] <= min(top["p"] * r["G"], 1.0) + 1e-12
    assert all(h["p_adjusted_bonferroni"] <= 1.0 for h in go["top"])
    assert go["min_p_adjusted_bonferroni"] == pytest.approx(top["p_adjusted_bonferroni"])


def test_pairwise_rejections_are_gated_on_the_primary_group_family(monkeypatch):
    # a pairwise p below 0.05/(G*K) whose GROUP omnibus does not reject must not surface as a
    # rejection: otherwise the group and pairwise families could be unioned at full alpha
    g = 10
    strong = np.full(g, 0.5)
    strong[3] = 1e-9          # one contrast is extreme, but ACAT over 3 contrasts dilutes it
    calls = {"i": 0}

    def _fake(Wh, y, BX, BY, C=None, dominance_adjust=False, *, fixed_mask=None, return_diag=False):
        pv = strong if calls["i"] % 3 == 0 else np.full(g, 0.5)
        calls["i"] += 1
        if return_diag:
            return pv, dict(n_planned=pv.size, estimable=np.isfinite(pv), exclusions={}, failures={})
        return pv

    monkeypatch.setattr(I, "pairwise_pvals", _fake)
    monkeypatch.setattr(I, "pairwise_design_mask",
                        lambda BX, BY, C=None, dominance_adjust=False:
                        (np.ones(BX.shape[1], bool), {}))
    rng = np.random.default_rng(51)
    subdata = {s: _make_sub(rng, g=g) for s in ("A", "B", "D")}
    groups = [(f"g{i}",) * 3 for i in range(g)]
    r = run_clique_scan(subdata, groups, rng.standard_normal(N), np.arange(N), cap=150,
                        transform="INT", perm_B=0, grm_method="grm_from_X")
    go, ab = r["group_omnibus"], r["pairwise"]["AB"]
    gated_ids = {tuple(h["triad"]) for h in go["sig"]}
    assert all(tuple(h["triad"]) in gated_ids for h in ab["sig"])
    assert r["inference_plan"]["pairwise_role"] == "gated_follow_up_localization"
    assert ab["acat_rejected_familywise"] is None


def test_sensitivity_transform_emits_no_rejection_set():
    # both transforms are scanned, but only the predeclared one may spend alpha; otherwise taking
    # hits from either is an uncontrolled union over the same biological units
    rng = np.random.default_rng(61)
    subs = ["A", "B", "D"]
    subdata = {s: _make_sub(rng) for s in subs}
    triads = [(f"g{i}",) * 3 for i in range(G)]
    hit = 17
    bB = block_burden_capped(subdata["B"].X, subdata["B"].gene_snp[f"g{hit}"], 150, rng)
    bD = block_burden_capped(subdata["D"].X, subdata["D"].gene_snp[f"g{hit}"], 150, rng)
    y = rng.standard_normal(N) + 1.8 * (_std(bB) * _std(bD))
    prim = run_clique_scan(subdata, triads, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                           grm_method="grm_from_X")
    sens = run_clique_scan(subdata, triads, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                           grm_method="grm_from_X", inferential=False)
    assert prim["group_omnibus"]["n_sig"] >= 1
    assert sens["group_omnibus"]["n_sig"] is None and sens["group_omnibus"]["sig"] is None
    assert sens["pairwise"]["BD"]["sig"] is None and sens["pairwise"]["BD"]["n_sig"] is None
    assert sens["inference_plan"]["inferential"] is False
    assert sens["inference_plan"]["primary_family"] is None
    # descriptive statistics are unchanged: only the rejection layer is withheld
    assert sens["pairwise"]["BD"]["min_p"] == prim["pairwise"]["BD"]["min_p"]
    assert sens["group_omnibus"]["min_p"] == prim["group_omnibus"]["min_p"]


def test_pair_sensitivity_run_withholds_rejections_only():
    rng = np.random.default_rng(62)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    y = rng.standard_normal(N)
    a = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                      pair_subs=("A", "D"), grm_method="grm_from_X")
    b = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                      pair_subs=("A", "D"), grm_method="grm_from_X", inferential=False)
    assert a.sig is not None and b.sig is None and b.n_sig is None
    assert a.min_p == b.min_p and a.pair_acat == b.pair_acat


def test_permutation_primary_requires_enough_replicates():
    rng = np.random.default_rng(63)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    with pytest.raises(ValueError, match="perm_B >= 19"):
        run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=5, n_jobs=1, pair_subs=("A", "D"),
                      grm_method="grm_from_X", primary_multiplicity="permutation_minp")
    with pytest.raises(ValueError, match="weighted permutation min-P"):
        run_pair_scan(subdata, pairs, rng.standard_normal(N), np.arange(N), cap=150,
                      transform="INT", perm_B=39, n_jobs=1, pair_subs=("A", "D"),
                      grm_method="grm_from_X", primary_multiplicity="permutation_minp",
                      primary_weighting="weighted",
                      pair_weights={(f"g{i}", f"g{i}"): 1.0 for i in range(G)})


def test_sensitivity_run_is_never_aborted_by_inference_only_guards():
    # a sensitivity run makes no claim, so guards that exist to protect a CLAIM must not fire on it
    rng = np.random.default_rng(71)
    subdata = {s: _make_sub(rng) for s in ("A", "D")}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    y = rng.standard_normal(N)
    zero_w = {(f"g{i}", f"g{i}"): 0.0 for i in range(G)}
    zero_w[("g0", "g0")] = 1.0
    # weighted primary with usable weights is fine either way; the point is that the
    # inference-only guards are skipped when inferential=False
    r = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                      pair_subs=("A", "D"), grm_method="grm_from_X", pair_weights=zero_w,
                      primary_weighting="weighted", inferential=False)
    assert r.sig is None and r.weighted["sig"] is None
    # permutation-primary with too few replicates would raise for a claim, but not for sensitivity
    r2 = run_pair_scan(subdata, pairs, y, np.arange(N), cap=150, transform="INT", perm_B=0,
                       pair_subs=("A", "D"), grm_method="grm_from_X",
                       primary_multiplicity="permutation_minp", inferential=False)
    assert r2.minp_perm_rejected is None and r2.sig is None


def test_null_replicates_by_index_are_order_independent_and_not_retransformed():
    from homoeogwas.resampling_checkpoint import replicate_seed

    n = 6
    kernels = {"A": np.eye(n)}
    y = np.linspace(7.0, 12.0, n)
    C = np.ones((n, 1))
    W = np.eye(n)
    V = np.diag(np.linspace(0.2, 0.7, n))
    beta = np.array([10.0])
    null_fit = (W, V, beta, {"A": 0.0, "e": 1.0})

    values, returned_W, returned_cv = I.null_replicates_by_index(
        kernels, y, C, indices=[4, 1], base_seed=2026,
        null_fit=null_fit)
    reversed_values, _, _ = I.null_replicates_by_index(
        kernels, y, C, indices=[1, 4], base_seed=2026,
        null_fit=null_fit)

    np.testing.assert_array_equal(values[0], reversed_values[1])
    np.testing.assert_array_equal(values[1], reversed_values[0])
    root = np.diag(np.sqrt(np.diag(V)))
    expected = 10.0 + root @ np.random.default_rng(
        replicate_seed(2026, 4)).standard_normal(n)
    np.testing.assert_array_equal(values[0], expected)
    assert values[0].mean() > 5.0  # still on the analysis scale, not rank-INT
    np.testing.assert_array_equal(returned_W, W)
    assert returned_cv == {"A": 0.0, "e": 1.0}

    with pytest.raises(ValueError, match="non-negative"):
        I.null_replicates_by_index(
            kernels, y, C, indices=[-1], base_seed=2026,
            null_fit=null_fit)
