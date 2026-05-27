"""Genotype I/O for HomoeoGWAS.

最小实现:从 PLINK1 .bed 读 hard-call allele-count 矩阵
({0,1,2}, NaN = missing),用 bed-reader 后端。

Conventions:
- shape (n_samples, n_variants), float32
- missing -> NaN
- dosage 字段当前只承载 BED hard calls,不是 imputed/DS dosage
- 不做 imputation,后续步骤(GRM 计算)显式处理 NaN
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GenoChunk:
    """Genotype chunk.

    The ``dosage`` field is kept as the common GRM input name, but the built-in
    BED reader only populates hard-call values in {0,1,2,NaN}.
    """
    samples: np.ndarray   # shape (n,), dtype=object (str IIDs)
    variant_ids: np.ndarray  # shape (m,), dtype=object
    chrom: np.ndarray     # shape (m,), dtype=object
    pos: np.ndarray       # shape (m,), dtype=int64
    dosage: np.ndarray    # shape (n, m), dtype=float32, values {0,1,2,NaN}


def load_bed_hardcall(bed_prefix: str | Path) -> GenoChunk:
    """Read a PLINK1 .bed/.bim/.fam prefix into a GenoChunk.

    Args:
        bed_prefix: path prefix; expects <prefix>.bed, .bim, .fam.

    Returns:
        GenoChunk with shape (n_samples, n_variants), hard-call dosage in
        {0,1,2} or NaN. Direct .pgen or imputed VCF DS dosage readers are not
        wired in for v0.1.
    """
    from bed_reader import open_bed

    prefix = Path(bed_prefix)
    bed_path = prefix.with_suffix(".bed")
    if bed_path.exists():
        path = bed_path
    else:
        raise FileNotFoundError(
            f"Expected {bed_path} (run `plink2 --pfile {prefix} --make-bed --out {prefix}` "
            "to produce .bed first; direct .pgen/DS dosage readers are not in v0.1)"
        )

    with open_bed(str(path), count_A1=True) as bed:
        # bed_reader returns (n_samples, n_variants) with values in {0,1,2, NaN}
        dosage = bed.read(dtype="float32")
        samples = np.asarray(bed.iid, dtype=object)
        variant_ids = np.asarray(bed.sid, dtype=object)
        chrom = np.asarray(bed.chromosome, dtype=object)
        pos = np.asarray(bed.bp_position, dtype=np.int64)

    return GenoChunk(
        samples=samples,
        variant_ids=variant_ids,
        chrom=chrom,
        pos=pos,
        dosage=dosage,
    )


def iter_bed_chunks(bed_prefix: str | Path, *, chunk_size: int = 200_000):
    """Stream a PLINK1 BED in variant chunks (for files larger than RAM).

    Reads ``chunk_size`` variants at a time via bed_reader variant slicing, so
    a multi-GB / tens-of-millions-of-SNP BED can be scanned without ever
    materializing the whole genotype matrix. Variant metadata (id/chrom/pos)
    is read once from the .bim; the full sample axis is loaded per chunk.

    Args:
        bed_prefix: path prefix; expects <prefix>.bed/.bim/.fam.
        chunk_size: variants per yielded chunk.

    Yields:
        (variant_start, GenoChunk) — variant_start is the 0-based index of the
        chunk's first variant in the full BED.
    """
    from bed_reader import open_bed

    prefix = Path(bed_prefix)
    bed_path = prefix.with_suffix(".bed")
    if not bed_path.exists():
        raise FileNotFoundError(f"Expected {bed_path} (.bed/.bim/.fam prefix)")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    with open_bed(str(bed_path), count_A1=True) as bed:
        n_var = int(bed.sid_count)
        samples = np.asarray(bed.iid, dtype=object)
        sid = np.asarray(bed.sid, dtype=object)
        chrom = np.asarray(bed.chromosome, dtype=object)
        pos = np.asarray(bed.bp_position, dtype=np.int64)
        for start in range(0, n_var, chunk_size):
            end = min(start + chunk_size, n_var)
            dosage = bed.read(index=(slice(None), slice(start, end)), dtype="float32")
            yield start, GenoChunk(
                samples=samples,
                variant_ids=sid[start:end],
                chrom=chrom[start:end],
                pos=pos[start:end],
                dosage=dosage,
            )


def vcf_to_bed(vcf_gz: str | Path, out_prefix: str | Path, threads: int = 8) -> Path:
    """Convert VCF.gz to plink1.9 bed/bim/fam using plink2.

    Idempotent: skips if <out_prefix>.bed already exists.

    Args:
        vcf_gz: input .vcf.gz path
        out_prefix: output bed/bim/fam prefix
        threads: passed to plink2 --threads

    Returns:
        Path to <out_prefix>.bed
    """
    import shutil
    import subprocess

    out = Path(out_prefix)
    bed = out.with_suffix(".bed")
    if bed.exists():
        return bed
    out.parent.mkdir(parents=True, exist_ok=True)
    # 解析 plink2 绝对路径(subprocess 不一定继承 PATH)
    plink2 = shutil.which("plink2") or "/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin/plink2"
    if not Path(plink2).exists():
        raise FileNotFoundError(f"plink2 not found (looked up via PATH and fallback {plink2})")
    cmd = [
        plink2, "--vcf", str(vcf_gz), "--make-bed",
        "--out", str(out), "--threads", str(threads),
    ]
    subprocess.run(cmd, check=True)
    return bed
