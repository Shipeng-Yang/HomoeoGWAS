"""Unit tests for LOCO (Phase 3 M3.1) — per-chrom GRM parts + LOCO scan.

Covers the engine surface added in src/homoeogwas/{grm,scan}.py:
  • compute_grm_parts / compute_loco_grm_parts / loco_grm_from_parts
  • LOCOContext / build_loco_scan_contexts
  • scan_snps_loco / scan_bed_stream_loco
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from homoeogwas.grm import (
    GRMPart,
    compute_grm,
    compute_grm_parts,
    compute_loco_grm_parts,
    loco_grm_from_parts,
)
from homoeogwas.io import GenoChunk
from homoeogwas.kernel import hadamard_kernel, normalize_kernel
from homoeogwas.scan import (
    LOCOContext,
    _chrom_runs,
    build_loco_scan_contexts,
    build_scan_context,
    scan_bed_stream_loco,
    scan_snps,
    scan_snps_loco,
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _toy_multi_chrom_chunk(
    n: int = 60, m_per_chrom: int = 80,
    chroms: tuple[str, ...] = ("c1", "c2", "c3", "c4"),
    seed: int = 0, maf_lo: float = 0.05,
) -> GenoChunk:
    """Toy genotype chunk with multiple chrom blocks."""
    rng = np.random.default_rng(seed)
    m = m_per_chrom * len(chroms)
    p = rng.uniform(maf_lo, 0.5, size=m).astype(np.float32)
    X = rng.binomial(2, p[np.newaxis, :].repeat(n, axis=0)).astype(np.float32)
    chrom_arr = np.array(
        [c for c in chroms for _ in range(m_per_chrom)], dtype=object)
    pos_arr = np.array(
        [j for _ in chroms for j in range(1, m_per_chrom + 1)], dtype=np.int64)
    return GenoChunk(
        samples=np.array([f"s{i:03d}" for i in range(n)], dtype=object),
        variant_ids=np.array([f"v{j:04d}" for j in range(m)], dtype=object),
        chrom=chrom_arr,
        pos=pos_arr,
        dosage=X,
    )


def _toy_psd(n: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, r))
    K = M @ M.T + 1e-3 * np.eye(n)
    return K * n / np.trace(K)


def _toy_loco_setup(
    n: int = 100, m_per_chrom: int = 60,
    chroms: tuple[str, ...] = ("cA", "cB", "cC"), seed: int = 7,
):
    """Build a small LOCO scenario: kernels_by_chrom + sigma2 + y/X + geno."""
    chunk = _toy_multi_chrom_chunk(
        n=n, m_per_chrom=m_per_chrom, chroms=chroms, seed=seed)
    global_part, parts = compute_loco_grm_parts(chunk, maf_min=0.01)
    # kernels_by_chrom: each chrom -> {"g": normalised LOCO GRM}
    kernels_by_chrom: dict[str, dict[str, np.ndarray]] = {}
    for c in chroms:
        K_raw, _info = loco_grm_from_parts(global_part, parts, c)
        kernels_by_chrom[c] = {"g": normalize_kernel(K_raw, mode="trace")}
    # global REML σ² (synthetic — not actually fit; LOCO doesn't refit)
    sigma2 = {"g": 0.6, "e": 0.4}
    # simulate y under the global GRM (kernels are similar across chrom-out)
    rng = np.random.default_rng(seed + 1000)
    K_global = normalize_kernel(global_part.grm, mode="trace")
    V_global = sigma2["g"] * K_global + sigma2["e"] * np.eye(n)
    L = np.linalg.cholesky(V_global + 1e-8 * np.eye(n))
    y = 3.0 + L @ rng.standard_normal(n)
    X = np.ones((n, 1))
    return (y, X, kernels_by_chrom, sigma2, chunk, parts, global_part,
            K_global)


# ---------------------------------------------------------------------
# grm.py LOCO building blocks
# ---------------------------------------------------------------------


def test_compute_grm_parts_matches_compute_grm():
    """``compute_grm_parts(...).grm`` reproduces ``compute_grm(...)`` exactly."""
    chunk = _toy_multi_chrom_chunk(n=40, m_per_chrom=50, seed=1)
    part = compute_grm_parts(chunk, maf_min=0.01)
    assert isinstance(part, GRMPart)
    G_parts = part.grm
    G_old, _ = compute_grm(chunk, maf_min=0.01)
    np.testing.assert_allclose(G_parts, G_old, rtol=1e-12, atol=1e-12)


def test_loco_parts_sum_to_global_numerator_and_denom():
    """Sum of per-chrom parts equals the global part (numerator + denom)."""
    chunk = _toy_multi_chrom_chunk(n=50, m_per_chrom=70, seed=2)
    global_part, parts = compute_loco_grm_parts(chunk, maf_min=0.01)
    sum_num = sum(p.numerator for p in parts.values())
    sum_denom = sum(p.denominator for p in parts.values())
    sum_used = sum(p.n_variants_used for p in parts.values())
    np.testing.assert_allclose(sum_num, global_part.numerator,
                               rtol=1e-12, atol=1e-10)
    assert abs(sum_denom - global_part.denominator) < 1e-10 * global_part.denominator
    assert sum_used == global_part.n_variants_used


def test_loco_grm_subtraction_equals_recompute():
    """``loco_grm_from_parts`` ≡ compute_grm on the genotype with chrom c removed."""
    chunk = _toy_multi_chrom_chunk(
        n=60, m_per_chrom=80, chroms=("cA", "cB", "cC", "cD"), seed=3)
    chrom_arr = np.asarray(chunk.chrom).astype(str)
    global_part, parts = compute_loco_grm_parts(chunk, maf_min=0.01)
    for c in ("cA", "cB", "cC", "cD"):
        K_minus, info = loco_grm_from_parts(global_part, parts, c)
        # recompute by physical subsetting
        keep_mask = chrom_arr != c
        sub_chunk = GenoChunk(
            samples=chunk.samples,
            variant_ids=chunk.variant_ids[keep_mask],
            chrom=chunk.chrom[keep_mask],
            pos=chunk.pos[keep_mask],
            dosage=chunk.dosage[:, keep_mask],
        )
        G_recompute, info_rc = compute_grm(sub_chunk, maf_min=0.01)
        # MAF filter applied identically; result should match to float-eps
        np.testing.assert_allclose(K_minus, G_recompute, rtol=1e-10, atol=1e-10)
        # info fields: retained = global - chrom-c; removed = chrom-c
        assert info["n_variants_retained"] == (
            global_part.n_variants_used - parts[c].n_variants_used)
        assert info["n_variants_removed"] == parts[c].n_variants_used
        assert info["n_variants_global"] == global_part.n_variants_used
        # 4 balanced chroms => retained ~ 75%, removed ~ 25%, no low-risk flag
        assert 0.7 < info["denom_retained_fraction"] < 0.8
        assert 0.2 < info["denom_removed_fraction"] < 0.3
        assert info["retained_fraction_low_risk"] is False


def test_loco_grm_denominator_guard_raises():
    """If one chrom dominates the GRM, leave-one-out should fail-fast."""
    # synthetic: chrom 'big' has 1000 variants, chrom 'tiny' has 5
    chunk_big = _toy_multi_chrom_chunk(
        n=40, m_per_chrom=1000, chroms=("big",), seed=10)
    chunk_tiny = _toy_multi_chrom_chunk(
        n=40, m_per_chrom=5, chroms=("tiny",), seed=11)
    merged = GenoChunk(
        samples=chunk_big.samples,
        variant_ids=np.concatenate([chunk_big.variant_ids, chunk_tiny.variant_ids]),
        chrom=np.concatenate([chunk_big.chrom, chunk_tiny.chrom]),
        pos=np.concatenate([chunk_big.pos, chunk_tiny.pos]),
        dosage=np.concatenate([chunk_big.dosage, chunk_tiny.dosage], axis=1),
    )
    global_part, parts = compute_loco_grm_parts(merged, maf_min=0.01)
    # Leaving out 'big' retains <1% of denominator → must raise at floor 0.1
    with pytest.raises(ValueError, match="dominates"):
        loco_grm_from_parts(global_part, parts, "big",
                            min_denominator_fraction=0.1)
    # Leaving out 'tiny' is fine — and surfaces retained_fraction_low_risk=False
    K_minus_tiny, info_tiny = loco_grm_from_parts(
        global_part, parts, "tiny", min_denominator_fraction=0.1)
    assert K_minus_tiny.shape == (40, 40)
    assert info_tiny["denom_retained_fraction"] > 0.99   # big retains everything
    assert info_tiny["retained_fraction_low_risk"] is False


def test_loco_low_retained_fraction_warning():
    """retained_fraction < warn_retained_fraction surfaces the flag (no raise)."""
    chunk_big = _toy_multi_chrom_chunk(
        n=30, m_per_chrom=500, chroms=("big",), seed=12)
    chunk_small = _toy_multi_chrom_chunk(
        n=30, m_per_chrom=3, chroms=("small",), seed=13)
    merged = GenoChunk(
        samples=chunk_big.samples,
        variant_ids=np.concatenate([chunk_big.variant_ids, chunk_small.variant_ids]),
        chrom=np.concatenate([chunk_big.chrom, chunk_small.chrom]),
        pos=np.concatenate([chunk_big.pos, chunk_small.pos]),
        dosage=np.concatenate([chunk_big.dosage, chunk_small.dosage], axis=1),
    )
    global_part, parts = compute_loco_grm_parts(merged, maf_min=0.01)
    # Leave-out 'big': only ~0.6% denom retained → above 1e-6 floor but
    # below 1% warn threshold → no raise, flag = True
    K_minus, info = loco_grm_from_parts(
        global_part, parts, "big",
        min_denominator_fraction=1.0e-6,
        warn_retained_fraction=0.01,
    )
    assert K_minus.shape == (30, 30)
    assert info["denom_retained_fraction"] < 0.01
    assert info["retained_fraction_low_risk"] is True


def test_loco_grm_unknown_chrom_raises():
    chunk = _toy_multi_chrom_chunk(n=30, m_per_chrom=40, seed=4)
    global_part, parts = compute_loco_grm_parts(chunk, maf_min=0.01)
    with pytest.raises(KeyError, match="not in parts"):
        loco_grm_from_parts(global_part, parts, "no_such_chrom")


def test_compute_loco_grm_parts_checksum_passes_with_per_variant_maf():
    """The build-time checksum is satisfied by any per-variant MAF filter."""
    chunk = _toy_multi_chrom_chunk(
        n=80, m_per_chrom=120, chroms=("c1", "c2", "c3"), seed=5)
    global_part, parts = compute_loco_grm_parts(chunk, maf_min=0.01)
    assert global_part.n_variants_used > 0
    assert all(p.n_variants_used >= 0 for p in parts.values())


# ---------------------------------------------------------------------
# Hadamard with LOCO factors
# ---------------------------------------------------------------------


def test_loco_hadamard_only_replaces_affected_factor():
    """K_hom(-c) = K_{sub(c)}(-c) ⊙ K_{j!=sub(c)}(global)."""
    chunk_A = _toy_multi_chrom_chunk(
        n=40, m_per_chrom=80, chroms=("a1", "a2", "a3"), seed=21)
    chunk_B = _toy_multi_chrom_chunk(
        n=40, m_per_chrom=80, chroms=("b1", "b2"), seed=22)
    # match samples (same n, same labels)
    chunk_B = GenoChunk(
        samples=chunk_A.samples, variant_ids=chunk_B.variant_ids,
        chrom=chunk_B.chrom, pos=chunk_B.pos, dosage=chunk_B.dosage)
    gA_glob, gA_parts = compute_loco_grm_parts(chunk_A, maf_min=0.01)
    gB_glob, gB_parts = compute_loco_grm_parts(chunk_B, maf_min=0.01)
    K_A_global = gA_glob.grm
    K_B_global = gB_glob.grm
    # LOCO chrom 'a2' (belongs to subgenome A):
    #   K_A(-a2) = sub LOCO; K_B(-a2) = K_B_global
    K_A_minus_a2, _ = loco_grm_from_parts(gA_glob, gA_parts, "a2")
    # Manual Hadamard
    K_hom_minus_a2_expected = K_A_minus_a2 * K_B_global
    # via hadamard_kernel
    K_hom_minus_a2 = hadamard_kernel({"A": K_A_minus_a2, "B": K_B_global})
    np.testing.assert_allclose(
        K_hom_minus_a2, K_hom_minus_a2_expected, rtol=1e-12, atol=1e-12)
    # LOCO chrom 'b1' (belongs to subgenome B):
    K_B_minus_b1, _ = loco_grm_from_parts(gB_glob, gB_parts, "b1")
    K_hom_minus_b1 = hadamard_kernel({"A": K_A_global, "B": K_B_minus_b1})
    np.testing.assert_allclose(
        K_hom_minus_b1, K_A_global * K_B_minus_b1, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------
# build_loco_scan_contexts
# ---------------------------------------------------------------------


def test_build_loco_scan_contexts_basic():
    y, X, kernels_by_chrom, sigma2, _, _, _, _ = _toy_loco_setup()
    sample_ids = np.array([f"s{i:03d}" for i in range(len(y))], dtype=object)
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=sample_ids)
    assert isinstance(loco_ctx, LOCOContext)
    assert loco_ctx.n == len(y)
    assert loco_ctx.p == X.shape[1]
    assert loco_ctx.kernel_names == ["g"]
    assert loco_ctx.chroms == ["cA", "cB", "cC"]
    assert loco_ctx.sigma2 == {"g": 0.6, "e": 0.4}
    # Each per-chrom P annihilates X
    for _c, ctx_c in loco_ctx.contexts.items():
        assert np.allclose(ctx_c.P @ X, 0.0, atol=1e-8)
        assert np.allclose(ctx_c.P, ctx_c.P.T, atol=1e-10)


def test_build_loco_scan_contexts_kernel_name_mismatch():
    y, X, kernels_by_chrom, sigma2, *_ = _toy_loco_setup()
    # Inject a different kernel name into one chrom
    bad = {c: dict(d) for c, d in kernels_by_chrom.items()}
    bad["cB"] = {"other": bad["cB"]["g"]}
    with pytest.raises(ValueError, match="kernel_names mismatch"):
        build_loco_scan_contexts(y, X, bad, sigma2)


def test_loco_context_collapses_to_emmax_when_kernels_identical():
    """If kernels_by_chrom are all the same, LOCO scan ≡ EMMAX scan."""
    rng = np.random.default_rng(99)
    n, m_per = 80, 40
    chunk = _toy_multi_chrom_chunk(
        n=n, m_per_chrom=m_per, chroms=("k1", "k2"), seed=99)
    global_part, _ = compute_loco_grm_parts(chunk, maf_min=0.01)
    K_global = normalize_kernel(global_part.grm, mode="trace")
    sigma2 = {"g": 0.5, "e": 0.5}
    sample_ids = chunk.samples
    V = sigma2["g"] * K_global + sigma2["e"] * np.eye(n)
    L = np.linalg.cholesky(V + 1e-8 * np.eye(n))
    y = L @ rng.standard_normal(n)
    X = np.ones((n, 1))
    # LOCO with the SAME global kernel for every chrom
    kernels_by_chrom = {c: {"g": K_global} for c in ("k1", "k2")}
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=sample_ids)
    emmax_ctx = build_scan_context(
        y, X, {"g": K_global}, sigma2, sample_ids=sample_ids)
    loco_res = scan_snps_loco(loco_ctx, chunk, backend="cpu")
    emmax_res = scan_snps(emmax_ctx, chunk, backend="cpu")
    np.testing.assert_allclose(loco_res.chi2, emmax_res.chi2,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(loco_res.beta, emmax_res.beta,
                               rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------
# _chrom_runs
# ---------------------------------------------------------------------


def test_chrom_runs_sorted():
    arr = np.array(["c1"] * 3 + ["c2"] * 2 + ["c3"] * 4, dtype=object)
    runs = _chrom_runs(arr)
    assert runs == [("c1", 0, 3), ("c2", 3, 5), ("c3", 5, 9)]


def test_chrom_runs_unsorted_emits_multiple_runs_per_chrom():
    arr = np.array(["c1", "c1", "c2", "c1", "c3"], dtype=object)
    runs = _chrom_runs(arr)
    assert runs == [("c1", 0, 2), ("c2", 2, 3), ("c1", 3, 4), ("c3", 4, 5)]


def test_chrom_runs_empty():
    assert _chrom_runs(np.array([], dtype=object)) == []


# ---------------------------------------------------------------------
# scan_snps_loco
# ---------------------------------------------------------------------


def test_scan_snps_loco_matches_explicit_per_chrom_scan():
    """LOCO scan on a multi-chrom chunk ≡ per-chrom build_scan_context+scan."""
    y, X, kernels_by_chrom, sigma2, chunk, *_ = _toy_loco_setup(seed=33)
    sample_ids = chunk.samples
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=sample_ids)
    loco_res = scan_snps_loco(loco_ctx, chunk, backend="cpu")
    # Reference: run scan_snps per chrom with its own ScanContext
    chrom_arr = np.asarray(chunk.chrom).astype(str)
    ref_chi2 = np.full(chunk.dosage.shape[1], np.nan)
    for c, ctx_c in loco_ctx.contexts.items():
        mask = chrom_arr == c
        sub_chunk = GenoChunk(
            samples=chunk.samples,
            variant_ids=chunk.variant_ids[mask],
            chrom=chunk.chrom[mask],
            pos=chunk.pos[mask],
            dosage=chunk.dosage[:, mask],
        )
        sub_res = scan_snps(ctx_c, sub_chunk, backend="cpu")
        # Align by snp_id
        for sid, chi2 in zip(sub_res.snp_id, sub_res.chi2, strict=True):
            idx = np.where(chunk.variant_ids == sid)[0][0]
            ref_chi2[idx] = chi2
    # Align loco_res to the original chunk variant order
    loco_chi2 = np.full(chunk.dosage.shape[1], np.nan)
    for sid, chi2 in zip(loco_res.snp_id, loco_res.chi2, strict=True):
        idx = np.where(chunk.variant_ids == sid)[0][0]
        loco_chi2[idx] = chi2
    ok = np.isfinite(ref_chi2) & np.isfinite(loco_chi2)
    assert ok.sum() > 0
    np.testing.assert_allclose(loco_chi2[ok], ref_chi2[ok],
                               rtol=1e-10, atol=1e-10)


def test_scan_snps_loco_single_snp_matches_explicit_gls_wald():
    """Per-chrom score χ² ≡ explicit GLS Wald with V_(-c) for a single SNP."""
    y, X, kernels_by_chrom, sigma2, chunk, *_ = _toy_loco_setup(seed=44)
    sample_ids = chunk.samples
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=sample_ids)
    chrom_arr = np.asarray(chunk.chrom).astype(str)

    def _gls_wald(y, X, V, g_col):
        Vinv = np.linalg.inv(V)
        Z = np.column_stack([X, g_col])
        cov = np.linalg.inv(Z.T @ Vinv @ Z)
        theta = cov @ (Z.T @ Vinv @ y)
        gamma = theta[-1]
        var_gamma = cov[-1, -1]
        return gamma * gamma / var_gamma

    # Pick 5 random SNPs across chroms
    rng = np.random.default_rng(0)
    picks = rng.choice(chunk.dosage.shape[1], size=5, replace=False)
    loco_res = scan_snps_loco(loco_ctx, chunk, backend="cpu")
    snp_to_chi2 = dict(zip(loco_res.snp_id, loco_res.chi2, strict=True))
    for j in picks:
        c = chrom_arr[j]
        K_c = kernels_by_chrom[c]["g"]
        V_c = sigma2["g"] * K_c + sigma2["e"] * np.eye(chunk.dosage.shape[0])
        g_col = np.asarray(chunk.dosage[:, j], dtype=np.float64)
        # Mean-impute (matches _impute_filter_batch on no-missing data)
        finite = np.isfinite(g_col)
        if not finite.all():
            g_col[~finite] = g_col[finite].mean()
        chi2_ref = _gls_wald(y, X, V_c, g_col)
        chi2_loco = snp_to_chi2.get(chunk.variant_ids[j])
        if chi2_loco is None:
            continue
        # tight: the LOCO score χ² and Wald χ² agree to ~1e-8 in float64
        assert abs(chi2_loco - chi2_ref) / max(abs(chi2_ref), 1e-9) < 1e-7


def test_scan_snps_loco_does_not_degrade_cis_chi2():
    """LOCO does NOT degrade the χ² at a planted cis-effect (regression test).

    Codex review note: this is the floor — it does not by itself prove the
    LOCO advantage. See test_scan_snps_loco_advantage_under_concentrated_*
    for a stronger fixture that exercises the proximal-contamination escape.
    """
    rng = np.random.default_rng(2026)
    n, m_per = 200, 80
    chroms = ("c1", "c2", "c3", "c4")
    chunk = _toy_multi_chrom_chunk(
        n=n, m_per_chrom=m_per, chroms=chroms, seed=2026, maf_lo=0.1)
    global_part, parts = compute_loco_grm_parts(chunk, maf_min=0.01)
    K_global = normalize_kernel(global_part.grm, mode="trace")
    kernels_by_chrom = {}
    for c in chroms:
        K_raw, _ = loco_grm_from_parts(global_part, parts, c)
        kernels_by_chrom[c] = {"g": normalize_kernel(K_raw, mode="trace")}
    sigma2 = {"g": 0.3, "e": 0.4}
    V = sigma2["g"] * K_global + sigma2["e"] * np.eye(n)
    L = np.linalg.cholesky(V + 1e-8 * np.eye(n))
    u_plus_e = L @ rng.standard_normal(n)
    chrom_arr = np.asarray(chunk.chrom).astype(str)
    causal_idx = np.where(chrom_arr == "c1")[0][m_per // 2]
    g_causal = np.asarray(chunk.dosage[:, causal_idx], dtype=np.float64)
    g_causal = (g_causal - g_causal.mean()) / max(g_causal.std(), 1e-9)
    y = 2.0 + 0.8 * g_causal + u_plus_e
    X = np.ones((n, 1))
    emmax_ctx = build_scan_context(
        y, X, {"g": K_global}, sigma2, sample_ids=chunk.samples)
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=chunk.samples)
    emmax_res = scan_snps(emmax_ctx, chunk, backend="cpu")
    loco_res = scan_snps_loco(loco_ctx, chunk, backend="cpu")
    sid = chunk.variant_ids[causal_idx]
    emmax_chi2 = float(emmax_res.chi2[emmax_res.snp_id == sid][0])
    loco_chi2 = float(loco_res.chi2[loco_res.snp_id == sid][0])
    assert loco_chi2 >= emmax_chi2 * 0.99    # no degradation (1% float slack)
    assert loco_chi2 > 4.0                   # at least nominally significant


def test_scan_snps_loco_advantage_under_concentrated_polygenic():
    """Strong proximal contamination → LOCO must beat EMMAX by a clear margin.

    Fixture: the polygenic signal is concentrated on a single chrom (15 random
    SNPs on chrom c1, each contributing equal-weight to y). Then we plant
    an additional cis-effect SNP on c1. EMMAX's global GRM absorbs the
    c1-polygenic mass into u, deflating the cis-SNP score; LOCO drops chrom
    c1 from V_(-c1) and recovers it. Asserts LOCO χ² ≥ EMMAX χ² × 1.03 on
    the causal SNP (3 % advantage), matching the Horvath2020 end-to-end
    finding (median ratio 1.027 across the top-100 EMMAX hits).
    """
    rng = np.random.default_rng(20260523)
    n, m_per = 250, 100
    chroms = ("c1", "c2", "c3", "c4")
    chunk = _toy_multi_chrom_chunk(
        n=n, m_per_chrom=m_per, chroms=chroms, seed=20260523, maf_lo=0.15)
    chrom_arr = np.asarray(chunk.chrom).astype(str)
    c1_idx = np.where(chrom_arr == "c1")[0]

    # 15 random c1 SNPs each contribute β=0.25 to y -> concentrated polygenic
    # signal on chrom c1 (EMMAX kernel will absorb it into u)
    poly_idx = rng.choice(c1_idx, size=15, replace=False)
    Y_poly = np.zeros(n, dtype=np.float64)
    for i in poly_idx:
        g = np.asarray(chunk.dosage[:, i], dtype=np.float64)
        g = (g - g.mean()) / max(g.std(), 1e-9)
        Y_poly += 0.25 * g
    # add one independent cis-effect SNP (not among the poly set)
    causal_idx = int(np.setdiff1d(c1_idx, poly_idx)[m_per // 3])
    g_causal = np.asarray(chunk.dosage[:, causal_idx], dtype=np.float64)
    g_causal = (g_causal - g_causal.mean()) / max(g_causal.std(), 1e-9)
    y = 3.0 + 0.7 * g_causal + Y_poly + 0.3 * rng.standard_normal(n)
    X = np.ones((n, 1))

    # build kernels
    global_part, parts = compute_loco_grm_parts(chunk, maf_min=0.01)
    K_global = normalize_kernel(global_part.grm, mode="trace")
    kernels_by_chrom = {}
    for c in chroms:
        K_raw, _ = loco_grm_from_parts(global_part, parts, c)
        kernels_by_chrom[c] = {"g": normalize_kernel(K_raw, mode="trace")}
    # Use approximate true σ² (would be REML-fit in CLI; for the test we set
    # the same σ² in both EMMAX and LOCO so the only difference is K vs K_(-c))
    sigma2 = {"g": 0.6, "e": 0.4}

    emmax_ctx = build_scan_context(
        y, X, {"g": K_global}, sigma2, sample_ids=chunk.samples)
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=chunk.samples)
    emmax_res = scan_snps(emmax_ctx, chunk, backend="cpu")
    loco_res = scan_snps_loco(loco_ctx, chunk, backend="cpu")
    sid = chunk.variant_ids[causal_idx]
    emmax_chi2 = float(emmax_res.chi2[emmax_res.snp_id == sid][0])
    loco_chi2 = float(loco_res.chi2[loco_res.snp_id == sid][0])
    # Quantitative gate: LOCO recovers at least 3 % χ² over EMMAX, both
    # comfortably significant (well above χ²₁ 5e-8 threshold of ~30).
    assert loco_chi2 > emmax_chi2 * 1.03, (
        f"LOCO did not beat EMMAX as expected — EMMAX={emmax_chi2:.2f}, "
        f"LOCO={loco_chi2:.2f}, ratio={loco_chi2/emmax_chi2:.4f}")
    assert loco_chi2 > 10.0                  # nominal cis-detection


def test_scan_snps_loco_rejects_unknown_chrom():
    y, X, kernels_by_chrom, sigma2, chunk, *_ = _toy_loco_setup()
    # Build a chunk with a chrom not in kernels_by_chrom
    bad_chunk = GenoChunk(
        samples=chunk.samples,
        variant_ids=np.array(["xv1", "xv2"], dtype=object),
        chrom=np.array(["zzz_not_in_loco", "zzz_not_in_loco"], dtype=object),
        pos=np.array([1, 2], dtype=np.int64),
        dosage=np.ones((len(chunk.samples), 2), dtype=np.float32),
    )
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=chunk.samples)
    with pytest.raises(KeyError, match="not in LOCOContext"):
        scan_snps_loco(loco_ctx, bad_chunk, backend="cpu")


def test_scan_snps_loco_sample_alignment():
    """Genotype rows are re-ordered to match LOCOContext sample_ids."""
    y, X, kernels_by_chrom, sigma2, chunk, *_ = _toy_loco_setup(seed=55)
    # Reverse genotype row order — LOCOContext sample_ids stays in original order
    rev_chunk = GenoChunk(
        samples=chunk.samples[::-1],
        variant_ids=chunk.variant_ids,
        chrom=chunk.chrom,
        pos=chunk.pos,
        dosage=chunk.dosage[::-1, :].copy(),
    )
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=chunk.samples)
    res_normal = scan_snps_loco(loco_ctx, chunk, backend="cpu")
    res_reordered = scan_snps_loco(loco_ctx, rev_chunk, backend="cpu")
    # Same SNPs → same χ² regardless of input row ordering
    s_to_chi_n = dict(zip(res_normal.snp_id, res_normal.chi2, strict=True))
    s_to_chi_r = dict(zip(res_reordered.snp_id, res_reordered.chi2, strict=True))
    common = set(s_to_chi_n) & set(s_to_chi_r)
    assert len(common) > 0
    for s in common:
        np.testing.assert_allclose(s_to_chi_n[s], s_to_chi_r[s],
                                   rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------
# scan_bed_stream_loco
# ---------------------------------------------------------------------


def test_scan_bed_stream_loco_matches_in_memory(tmp_path):
    """Streaming LOCO over a BED ≡ in-memory LOCO scan."""
    from bed_reader import to_bed
    y, X, kernels_by_chrom, sigma2, chunk, *_ = _toy_loco_setup(seed=66)
    sample_ids = chunk.samples
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=sample_ids)
    # Write the chunk to a BED for streaming
    bed_path = tmp_path / "loco_test.bed"
    to_bed(
        str(bed_path),
        np.where(np.isnan(chunk.dosage), -127, chunk.dosage).astype(np.int8),
        properties={
            "iid": [str(s) for s in chunk.samples],
            "sid": [str(v) for v in chunk.variant_ids],
            "chromosome": [str(c) for c in chunk.chrom],
            "bp_position": chunk.pos.astype(np.int32),
        },
        count_A1=True,
    )
    out_path = tmp_path / "stream_loco.tsv.gz"
    summary = scan_bed_stream_loco(
        loco_ctx, tmp_path / "loco_test", out_path,
        backend="cpu", chunk_size=50, subgenome="test", gzip_out=True)
    assert summary.n_chunks >= 1
    # Compare with in-memory
    res_mem = scan_snps_loco(loco_ctx, chunk, backend="cpu")
    stream_df = pd.read_csv(out_path, sep="\t")
    # Align by snp_id
    mem_map = dict(zip(res_mem.snp_id, res_mem.chi2, strict=True))
    stream_map = dict(zip(stream_df["snp_id"], stream_df["chi2"], strict=True))
    common = set(mem_map) & set(stream_map)
    assert len(common) > 0
    for s in common:
        np.testing.assert_allclose(mem_map[s], stream_map[s],
                                   rtol=1e-10, atol=1e-10)


def test_scan_bed_stream_loco_chunk_crosses_chrom_boundary(tmp_path):
    """chunk_size that straddles chrom boundary -> per-chrom P is honoured."""
    from bed_reader import to_bed
    y, X, kernels_by_chrom, sigma2, chunk, *_ = _toy_loco_setup(seed=77)
    sample_ids = chunk.samples
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=sample_ids)
    bed_path = tmp_path / "loco_xchrom.bed"
    to_bed(
        str(bed_path),
        np.where(np.isnan(chunk.dosage), -127, chunk.dosage).astype(np.int8),
        properties={
            "iid": [str(s) for s in chunk.samples],
            "sid": [str(v) for v in chunk.variant_ids],
            "chromosome": [str(c) for c in chunk.chrom],
            "bp_position": chunk.pos.astype(np.int32),
        },
        count_A1=True,
    )
    out_path = tmp_path / "stream_xchrom.tsv.gz"
    # chunk_size deliberately straddles m_per_chrom (60) — multiple chrom per chunk
    summary = scan_bed_stream_loco(
        loco_ctx, tmp_path / "loco_xchrom", out_path,
        backend="cpu", chunk_size=70, subgenome="test", gzip_out=True)
    assert summary.n_chunks >= 2
    res_mem = scan_snps_loco(loco_ctx, chunk, backend="cpu")
    stream_df = pd.read_csv(out_path, sep="\t")
    mem_map = dict(zip(res_mem.snp_id, res_mem.chi2, strict=True))
    stream_map = dict(zip(stream_df["snp_id"], stream_df["chi2"], strict=True))
    common = set(mem_map) & set(stream_map)
    assert len(common) > 0
    for s in common:
        np.testing.assert_allclose(mem_map[s], stream_map[s],
                                   rtol=1e-10, atol=1e-10)


def test_scan_bed_stream_loco_requires_sample_ids():
    """Streaming LOCO must know sample_ids for alignment."""
    y, X, kernels_by_chrom, sigma2, *_ = _toy_loco_setup()
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=None)
    with pytest.raises(ValueError, match="sample_ids is required"):
        scan_bed_stream_loco(
            loco_ctx, "/tmp/nope", "/tmp/nope.tsv",
            backend="cpu", chunk_size=10)


def test_scan_bed_stream_loco_rejects_unknown_chrom(tmp_path):
    """Streaming LOCO must fail-fast on a BED chunk containing an unknown chrom.

    Mirror of test_scan_snps_loco_rejects_unknown_chrom but for the streaming
    code path (Codex MINOR_FIX #5).
    """
    from bed_reader import to_bed
    y, X, kernels_by_chrom, sigma2, chunk, *_ = _toy_loco_setup(seed=88)
    sample_ids = chunk.samples
    loco_ctx = build_loco_scan_contexts(
        y, X, kernels_by_chrom, sigma2, sample_ids=sample_ids)

    # build a BED that includes chrom 'cA' (known) AND 'zzz_unknown'
    n = chunk.dosage.shape[0]
    extra_dos = np.ones((n, 3), dtype=np.float32)
    mixed_dos = np.concatenate([chunk.dosage[:, :5], extra_dos], axis=1)
    mixed_chrom = np.array(
        list(chunk.chrom[:5]) + ["zzz_unknown"] * 3, dtype=object)
    mixed_pos = np.concatenate([chunk.pos[:5], np.array([1, 2, 3])])
    mixed_ids = np.array(
        list(chunk.variant_ids[:5]) + ["xv1", "xv2", "xv3"], dtype=object)
    bed_path = tmp_path / "loco_unknown.bed"
    to_bed(
        str(bed_path),
        np.where(np.isnan(mixed_dos), -127, mixed_dos).astype(np.int8),
        properties={
            "iid": [str(s) for s in chunk.samples],
            "sid": [str(v) for v in mixed_ids],
            "chromosome": [str(c) for c in mixed_chrom],
            "bp_position": mixed_pos.astype(np.int32),
        },
        count_A1=True,
    )
    with pytest.raises(KeyError, match="unknown chrom"):
        scan_bed_stream_loco(
            loco_ctx, tmp_path / "loco_unknown",
            tmp_path / "out.tsv.gz",
            backend="cpu", chunk_size=20, subgenome="test", gzip_out=True)
