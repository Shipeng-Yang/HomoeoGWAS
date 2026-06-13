"""Tests for homoeogwas.interact — homoeolog-pair / triad burden-product engine.

Synthetic genotypes (random dosage -> near-identity GRM) exercise the statistical core:
an injected interaction is detected in the correct subgenome-pair and isolated from the
others; a pure-noise null produces no Bonferroni hits. These guard the engine before the
DL-weighting and multi-trait extensions are layered on.
"""
import numpy as np
import pytest

from homoeogwas.interact import (
    FrozenTraitSet,
    SubgenomeData,
    acat,
    acat_weighted,
    block_burden_capped,
    pair_conditional_diagnostics,
    run_multitrait_pair_scan,
    run_pair_scan,
    run_triad_scan,
)

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
                      pair_subs=("A", "D"), grm_method="grm_from_X", pair_weights=weights)
    assert r.weighted is not None
    # weighted ACAT is at least as significant as unweighted (true signal up-weighted)
    assert r.weighted["acat_weighted"] <= r.pair_acat + 1e-12
    # the up-weighted true pair is among the weighted Bonferroni hits
    assert any(h["pair"][0] == f"g{hit}" for h in r.weighted["sig"])


def test_weighted_null_no_type1_inflation():
    rng = np.random.default_rng(7)
    subdata = {"A": _make_sub(rng), "D": _make_sub(rng)}
    pairs = [(f"g{i}", f"g{i}") for i in range(G)]
    y = rng.standard_normal(N)  # pure noise
    # random y-INDEPENDENT weights -> weighted Bonferroni still controls FWER
    weights = {(f"g{i}", f"g{i}"): float(rng.uniform(0.2, 5.0)) for i in range(G)}
    r = run_pair_scan(subdata, pairs, y, np.arange(N), transform="INT", perm_B=0,
                      pair_subs=("A", "D"), grm_method="grm_from_X", pair_weights=weights)
    assert r.n_sig == 0
    assert r.weighted["bonferroni_n_sig"] == 0


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
