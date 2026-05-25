"""Unit tests for scan.py — Phase 2 M2.5 per-SNP association scan."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homoeogwas.io import GenoChunk
from homoeogwas.scan import (
    ScanContext,
    ScanResult,
    build_scan_context,
    lambda_gc,
    scan_snps,
)


def _toy_psd(n: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, r))
    K = M @ M.T
    K += 1e-3 * np.eye(n)
    return K * n / np.trace(K)


def _toy_geno(n: int, m: int, seed: int) -> GenoChunk:
    """Random {0,1,2} dosage with reasonable MAF."""
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.15, 0.5, size=m)
    dosage = np.zeros((n, m), dtype=np.float32)
    for j in range(m):
        dosage[:, j] = rng.binomial(2, af[j], size=n).astype(np.float32)
    return GenoChunk(
        samples=np.array([f"s{i}" for i in range(n)], dtype=object),
        variant_ids=np.array([f"snp{j}" for j in range(m)], dtype=object),
        chrom=np.array(["1"] * m, dtype=object),
        pos=np.arange(1, m + 1, dtype=np.int64),
        dosage=dosage,
    )


def _toy_setup(n=120, m=60, seed=0):
    """Return (y, X, kernels, sigma2, geno, V)."""
    rng = np.random.default_rng(seed)
    K = _toy_psd(n, 30, seed=seed)
    sigma2 = {"g": 0.6, "e": 0.4}
    V = sigma2["g"] * K + sigma2["e"] * np.eye(n)
    L = np.linalg.cholesky(V)
    y = 3.0 + L @ rng.standard_normal(n)
    X = np.ones((n, 1))
    geno = _toy_geno(n, m, seed=seed + 100)
    return y, X, {"g": K}, sigma2, geno, V


# ---------------------------------------------------------------------
# build_scan_context
# ---------------------------------------------------------------------

def test_build_scan_context_P_symmetric_and_annihilates_X():
    y, X, kernels, sigma2, _, _ = _toy_setup()
    ctx = build_scan_context(y, X, kernels, sigma2)
    assert isinstance(ctx, ScanContext)
    assert np.allclose(ctx.P, ctx.P.T, atol=1e-10)
    assert np.allclose(ctx.P @ X, 0.0, atol=1e-8)


def test_build_scan_context_Py():
    y, X, kernels, sigma2, _, _ = _toy_setup()
    ctx = build_scan_context(y, X, kernels, sigma2)
    np.testing.assert_allclose(ctx.Py, ctx.P @ y, rtol=0, atol=1e-10)


def test_build_scan_context_rejects_bad_sigma2():
    y, X, kernels, sigma2, _, _ = _toy_setup()
    with pytest.raises(ValueError, match="missing component"):
        build_scan_context(y, X, kernels, {"g": 0.6})          # no "e"
    with pytest.raises(ValueError, match="finite"):
        build_scan_context(y, X, kernels, {"g": -1.0, "e": 0.4})


# ---------------------------------------------------------------------
# core score test vs explicit GLS
# ---------------------------------------------------------------------

def _explicit_gls_wald(y, X, V, g):
    """Wald test of the marker coefficient in the GLS of [X | g]."""
    Vinv = np.linalg.inv(V)
    Z = np.column_stack([X, g])
    cov = np.linalg.inv(Z.T @ Vinv @ Z)
    theta = cov @ (Z.T @ Vinv @ y)
    gamma = theta[-1]
    var_gamma = cov[-1, -1]
    return gamma, np.sqrt(var_gamma), gamma * gamma / var_gamma


def test_scan_single_snp_matches_explicit_gls():
    """scan beta/se/chi2 must equal the explicit fixed-V GLS Wald test."""
    y, X, kernels, sigma2, geno, V = _toy_setup()
    ctx = build_scan_context(y, X, kernels, sigma2)
    res = scan_snps(ctx, geno, backend="cpu", maf_min=0.0, call_rate_min=0.0)
    # check every kept SNP against the explicit GLS
    for k in range(res.n_kept):
        j = int(np.where(geno.variant_ids == res.snp_id[k])[0][0])
        g = geno.dosage[:, j].astype(np.float64)
        gamma, se, chi2 = _explicit_gls_wald(y, X, V, g)
        assert abs(res.beta[k] - gamma) < 1e-7
        assert abs(res.se[k] - se) < 1e-7
        assert abs(res.chi2[k] - chi2) < 1e-6


def test_scan_translation_invariance():
    """The score statistic is invariant to g -> g + c·1 (P annihilates the
    intercept). Tested on the batched core, bypassing the MAF QC which
    legitimately assumes {0,1,2} allele-count coding."""
    from homoeogwas.scan import _scan_batch_cpu
    y, X, kernels, sigma2, geno, _ = _toy_setup()
    ctx = build_scan_context(y, X, kernels, sigma2)
    G = geno.dosage.astype(np.float64)
    U0, I0 = _scan_batch_cpu(ctx.P, ctx.Py, G)
    U1, I1 = _scan_batch_cpu(ctx.P, ctx.Py, G + 5.0)
    np.testing.assert_allclose(U0, U1, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(I0, I1, rtol=1e-8, atol=1e-8)


def test_scan_snp_scaling():
    """g -> c·g: U scales by c, I by c² ⇒ chi2 invariant, beta & se scale 1/c."""
    from homoeogwas.scan import _scan_batch_cpu
    y, X, kernels, sigma2, geno, _ = _toy_setup(m=20)
    ctx = build_scan_context(y, X, kernels, sigma2)
    G = geno.dosage.astype(np.float64)
    c = 3.0
    U0, I0 = _scan_batch_cpu(ctx.P, ctx.Py, G)
    Uc, Ic = _scan_batch_cpu(ctx.P, ctx.Py, G * c)
    np.testing.assert_allclose(Uc, c * U0, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(Ic, c * c * I0, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(Uc**2 / Ic, U0**2 / I0, rtol=1e-7, atol=1e-8)


# ---------------------------------------------------------------------
# missing values & QC filtering
# ---------------------------------------------------------------------

def test_scan_missing_mean_imputation():
    """A SNP with scattered missing calls is mean-imputed and still scanned."""
    y, X, kernels, sigma2, geno, _ = _toy_setup(m=10)
    d = geno.dosage.astype(np.float64).copy()
    d[:8, 3] = np.nan                              # 8 missing in SNP 3
    geno2 = GenoChunk(geno.samples, geno.variant_ids, geno.chrom, geno.pos, d)
    ctx = build_scan_context(y, X, kernels, sigma2)
    res = scan_snps(ctx, geno2, backend="cpu", maf_min=0.0, call_rate_min=0.5)
    k = int(np.where(res.snp_id == "snp3")[0][0])
    assert res.call_rate[k] == pytest.approx((geno.dosage.shape[0] - 8) / geno.dosage.shape[0])
    assert np.isfinite(res.chi2[k])


def test_scan_filters_monomorphic_and_low_callrate():
    """Monomorphic and low-call-rate SNPs are dropped and tallied."""
    y, X, kernels, sigma2, geno, _ = _toy_setup(m=12)
    d = geno.dosage.astype(np.float64).copy()
    d[:, 0] = 0.0                                  # monomorphic
    d[:, 1] = np.nan                               # all missing
    d[3:, 2] = np.nan                              # very low call rate
    geno2 = GenoChunk(geno.samples, geno.variant_ids, geno.chrom, geno.pos, d)
    ctx = build_scan_context(y, X, kernels, sigma2)
    res = scan_snps(ctx, geno2, backend="cpu", maf_min=0.01, call_rate_min=0.9)
    kept = set(res.snp_id)
    assert "snp0" not in kept and "snp1" not in kept and "snp2" not in kept
    assert sum(res.filter_counts.values()) >= 3
    assert res.n_input == 12


# ---------------------------------------------------------------------
# batching, backends, misc
# ---------------------------------------------------------------------

def test_scan_cpu_batch_equals_single_batch():
    """Small batch_size must give identical results to one big batch."""
    y, X, kernels, sigma2, geno, _ = _toy_setup(m=80)
    ctx = build_scan_context(y, X, kernels, sigma2)
    r_big = scan_snps(ctx, geno, backend="cpu", batch_size=10_000,
                      maf_min=0.0, call_rate_min=0.0)
    r_small = scan_snps(ctx, geno, backend="cpu", batch_size=7,
                        maf_min=0.0, call_rate_min=0.0)
    np.testing.assert_array_equal(r_big.snp_id, r_small.snp_id)
    np.testing.assert_allclose(r_big.chi2, r_small.chi2, rtol=1e-10, atol=1e-12)


def test_scan_gpu_matches_cpu():
    """GPU backend must agree with the CPU reference (skip if no torch/CUDA)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    y, X, kernels, sigma2, geno, _ = _toy_setup(m=200)
    ctx = build_scan_context(y, X, kernels, sigma2)
    r_cpu = scan_snps(ctx, geno, backend="cpu", maf_min=0.0, call_rate_min=0.0)
    r_gpu = scan_snps(ctx, geno, backend="gpu", maf_min=0.0, call_rate_min=0.0)
    np.testing.assert_array_equal(r_cpu.snp_id, r_gpu.snp_id)
    np.testing.assert_allclose(r_cpu.chi2, r_gpu.chi2, rtol=1e-8, atol=1e-8)
    assert r_gpu.backend_used == "gpu"


def test_scan_sample_alignment():
    """Genotypes in a shuffled sample order are realigned via sample_ids."""
    y, X, kernels, sigma2, geno, V = _toy_setup(m=15)
    sample_ids = geno.samples.copy()
    ctx = build_scan_context(y, X, kernels, sigma2, sample_ids=sample_ids)
    r_ref = scan_snps(ctx, geno, backend="cpu", maf_min=0.0, call_rate_min=0.0)
    # shuffle genotype rows + sample labels together
    rng = np.random.default_rng(7)
    perm = rng.permutation(geno.dosage.shape[0])
    shuffled = GenoChunk(
        samples=geno.samples[perm], variant_ids=geno.variant_ids,
        chrom=geno.chrom, pos=geno.pos, dosage=geno.dosage[perm, :],
    )
    r_shuf = scan_snps(ctx, shuffled, backend="cpu", maf_min=0.0, call_rate_min=0.0)
    np.testing.assert_allclose(np.sort(r_ref.chi2), np.sort(r_shuf.chi2),
                               rtol=1e-9, atol=1e-9)


def test_scan_p_in_unit_interval():
    y, X, kernels, sigma2, geno, _ = _toy_setup(m=60)
    ctx = build_scan_context(y, X, kernels, sigma2)
    res = scan_snps(ctx, geno, backend="cpu", maf_min=0.0, call_rate_min=0.0)
    assert np.all((res.p >= 0.0) & (res.p <= 1.0))
    assert np.all(res.chi2 >= 0.0)


def test_lambda_gc():
    """λ_GC = median(χ²)/0.4549364; for χ²₁ draws it is ~1."""
    rng = np.random.default_rng(0)
    chi2_null = rng.chisquare(df=1, size=50_000)
    assert abs(lambda_gc(chi2_null) - 1.0) < 0.05
    # exact: median == the χ²₁ median -> λ == 1
    assert abs(lambda_gc(np.array([0.4549364231195724])) - 1.0) < 1e-9


def test_scan_result_lambda_gc_property():
    y, X, kernels, sigma2, geno, _ = _toy_setup(m=40)
    ctx = build_scan_context(y, X, kernels, sigma2)
    res = scan_snps(ctx, geno, backend="cpu", maf_min=0.0, call_rate_min=0.0)
    assert np.isfinite(res.lambda_gc)
    assert isinstance(res, ScanResult)


# ---------------------------------------------------------------------
# M2.5-v2a — streaming BED scan
# ---------------------------------------------------------------------

def _write_toy_bed(path_prefix, geno):
    """Write a toy GenoChunk to a PLINK1 BED via bed_reader.to_bed."""
    from bed_reader import to_bed
    to_bed(
        str(path_prefix) + ".bed",
        geno.dosage.astype(np.float32),
        properties={
            "iid": list(np.asarray(geno.samples, dtype=str)),
            "sid": list(np.asarray(geno.variant_ids, dtype=str)),
            "chromosome": [str(c) for c in np.asarray(geno.chrom)],
            "bp_position": list(np.asarray(geno.pos, dtype=np.int64)),
        },
    )


def test_scan_bed_stream_matches_in_memory(tmp_path):
    """Streaming BED scan must equal in-memory scan_snps on the same data."""
    from homoeogwas.scan import scan_bed_stream
    y, X, kernels, sigma2, geno, _ = _toy_setup(n=100, m=130)
    ctx = build_scan_context(y, X, kernels, sigma2, sample_ids=geno.samples)
    bed_prefix = tmp_path / "toy"
    _write_toy_bed(bed_prefix, geno)

    ref = scan_snps(ctx, geno, backend="cpu", maf_min=0.0, call_rate_min=0.0)
    out = tmp_path / "stream.tsv.gz"
    summ = scan_bed_stream(ctx, bed_prefix, out, backend="cpu", chunk_size=37,
                           maf_min=0.0, call_rate_min=0.0, subgenome="X",
                           progress_every=0)
    assert summ.n_kept == ref.n_kept
    assert summ.backend_used == "cpu"
    st = pd.read_csv(out, sep="\t").set_index("snp_id")
    assert set(st["subgenome"]) == {"X"}
    for k in range(ref.n_kept):
        sid = ref.snp_id[k]
        # chi2/p are allele-coding invariant; the streaming math must match exactly
        assert abs(st.loc[sid, "chi2"] - ref.chi2[k]) < 1e-7
        assert abs(st.loc[sid, "p"] - ref.p[k]) < 1e-9


def test_scan_bed_stream_chunk_size_invariant(tmp_path):
    """Streaming results must not depend on chunk_size."""
    from homoeogwas.scan import scan_bed_stream
    y, X, kernels, sigma2, geno, _ = _toy_setup(n=90, m=145)
    ctx = build_scan_context(y, X, kernels, sigma2, sample_ids=geno.samples)
    bed_prefix = tmp_path / "toy"
    _write_toy_bed(bed_prefix, geno)
    o1, o2 = tmp_path / "s1.tsv.gz", tmp_path / "s2.tsv.gz"
    scan_bed_stream(ctx, bed_prefix, o1, backend="cpu", chunk_size=16,
                    maf_min=0.0, call_rate_min=0.0, progress_every=0)
    scan_bed_stream(ctx, bed_prefix, o2, backend="cpu", chunk_size=10_000,
                    maf_min=0.0, call_rate_min=0.0, progress_every=0)
    d1 = pd.read_csv(o1, sep="\t").set_index("snp_id").sort_index()
    d2 = pd.read_csv(o2, sep="\t").set_index("snp_id").sort_index()
    np.testing.assert_array_equal(d1.index, d2.index)
    np.testing.assert_allclose(d1["chi2"], d2["chi2"], rtol=1e-10, atol=1e-12)


def test_scan_bed_stream_plain_and_gzip(tmp_path):
    """gzip vs plain output give identical rows; header written once."""
    from homoeogwas.scan import scan_bed_stream
    y, X, kernels, sigma2, geno, _ = _toy_setup(n=80, m=60)
    ctx = build_scan_context(y, X, kernels, sigma2, sample_ids=geno.samples)
    bed_prefix = tmp_path / "toy"
    _write_toy_bed(bed_prefix, geno)
    o_gz, o_txt = tmp_path / "o.tsv.gz", tmp_path / "o.tsv"
    scan_bed_stream(ctx, bed_prefix, o_gz, backend="cpu", chunk_size=25,
                    maf_min=0.0, call_rate_min=0.0, gzip_out=True, progress_every=0)
    scan_bed_stream(ctx, bed_prefix, o_txt, backend="cpu", chunk_size=25,
                    maf_min=0.0, call_rate_min=0.0, gzip_out=False, progress_every=0)
    dg = pd.read_csv(o_gz, sep="\t")
    dt = pd.read_csv(o_txt, sep="\t")
    assert list(dg.columns) == list(dt.columns)
    assert len(dg) == len(dt)
    # exactly one header line in the plain file
    with open(o_txt) as fh:
        assert sum(1 for ln in fh if ln.startswith("snp_id\t")) == 1


def test_build_scan_context_rejects_duplicate_sample_ids():
    """Duplicate sample_ids would misalign genotypes to P/Py — must be rejected."""
    y, X, kernels, sigma2, geno, _ = _toy_setup(n=60, m=20)
    dup = np.asarray(geno.samples, dtype=object).copy()
    dup[1] = dup[0]                                    # inject a duplicate IID
    with pytest.raises(ValueError, match="unique"):
        build_scan_context(y, X, kernels, sigma2, sample_ids=dup)


def test_scan_bed_stream_requires_sample_ids(tmp_path):
    """Streaming scan needs context.sample_ids for genotype alignment."""
    from homoeogwas.scan import scan_bed_stream
    y, X, kernels, sigma2, geno, _ = _toy_setup(n=70, m=40)
    ctx = build_scan_context(y, X, kernels, sigma2)        # no sample_ids
    bed_prefix = tmp_path / "toy"
    _write_toy_bed(bed_prefix, geno)
    with pytest.raises(ValueError, match="sample_ids"):
        scan_bed_stream(ctx, bed_prefix, tmp_path / "o.tsv.gz", backend="cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
