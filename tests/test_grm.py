"""Unit tests for src/homoeogwas/grm.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from homoeogwas.grm import compute_grm, compute_grm_panel
from homoeogwas.io import GenoChunk


def _toy_chunk(n: int = 50, m: int = 1000, seed: int = 0, maf_lo: float = 0.05) -> GenoChunk:
    """Generate a deterministic toy biallelic dosage matrix at HWE."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(maf_lo, 0.5, size=m).astype(np.float32)
    # dosage X[i,j] ~ Binomial(2, p[j])
    X = rng.binomial(2, p[np.newaxis, :].repeat(n, axis=0)).astype(np.float32)
    # add 1% missing
    miss = rng.random((n, m)) < 0.01
    X[miss] = np.nan
    return GenoChunk(
        samples=np.array([f"s{i:03d}" for i in range(n)], dtype=object),
        variant_ids=np.array([f"v{j:04d}" for j in range(m)], dtype=object),
        chrom=np.array(["1"] * m, dtype=object),
        pos=np.arange(m, dtype=np.int64),
        dosage=X,
    )


def test_grm_shape_and_psd():
    """G should be (n,n), symmetric, positive semi-definite."""
    chunk = _toy_chunk(n=60, m=500)
    G, info = compute_grm(chunk)
    assert G.shape == (60, 60)
    assert np.allclose(G, G.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(G)
    assert eigvals.min() > -1e-6, f"GRM not PSD, min eigval={eigvals.min()}"


def test_grm_trace_approx_n():
    """trace(G) ≈ n for VanRaden GRM at HWE (within ~5%)."""
    n = 100
    chunk = _toy_chunk(n=n, m=5000, seed=42)
    G, info = compute_grm(chunk)
    # trace ~ n, allowing ~5% slack from finite m and missing imputation
    rel_err = abs(info["trace"] - n) / n
    assert rel_err < 0.05, f"trace(G)={info['trace']:.2f}, expected ~{n}, rel_err={rel_err:.3f}"


def test_grm_maf_filter():
    """MAF filter should drop monomorphic variants."""
    chunk = _toy_chunk(n=50, m=200, maf_lo=0.1)
    # Force the first 10 variants to be monomorphic (all zero)
    X = chunk.dosage.copy()
    X[:, :10] = 0
    chunk2 = GenoChunk(
        samples=chunk.samples, variant_ids=chunk.variant_ids,
        chrom=chunk.chrom, pos=chunk.pos, dosage=X,
    )
    _, info = compute_grm(chunk2, maf_min=0.01)
    assert info["n_variants_dropped_maf"] >= 10


def test_grm_handles_missing():
    """NaN dosage should be imputed (mean), not propagate."""
    chunk = _toy_chunk(n=30, m=100)
    G, info = compute_grm(chunk)
    assert np.all(np.isfinite(G)), "GRM has NaN — missing not imputed"


def test_grm_zero_denominator_raises():
    """Monomorphic variants kept by maf_min=0 should raise, not return NaN GRM."""
    chunk = GenoChunk(
        samples=np.array(["s1", "s2", "s3", "s4"], dtype=object),
        variant_ids=np.array(["v1", "v2", "v3"], dtype=object),
        chrom=np.array(["1", "1", "1"], dtype=object),
        pos=np.array([1, 2, 3], dtype=np.int64),
        dosage=np.zeros((4, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match=r"non-positive denominator|no informative variants"):
        compute_grm(chunk, maf_min=0.0)


def test_compute_grm_panel_duplicate_iid_raises(tmp_path, monkeypatch):
    """Duplicate IIDs must not be silently collapsed by sample_to_idx."""
    sub_dir = tmp_path / "A"
    sub_dir.mkdir()
    (sub_dir / "all.bed").touch()

    chunk = GenoChunk(
        samples=np.array(["dup", "dup"], dtype=object),
        variant_ids=np.array(["v1", "v2"], dtype=object),
        chrom=np.array(["1", "1"], dtype=object),
        pos=np.array([1, 2], dtype=np.int64),
        dosage=np.array([[0, 1], [2, 1]], dtype=np.float32),
    )

    def fake_load_bed_hardcall(prefix):
        assert Path(prefix) == sub_dir / "all"
        return chunk

    monkeypatch.setattr("homoeogwas.io.load_bed_hardcall", fake_load_bed_hardcall)
    with pytest.raises(ValueError, match=r"chunk\.samples must be unique.*A.*dup"):
        compute_grm_panel(tmp_path, ["A"], maf_min=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
