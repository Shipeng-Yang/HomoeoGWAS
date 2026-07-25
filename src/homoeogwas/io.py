"""Genotype I/O for HomoeoGWAS.

Reads PLINK1 .bed hard-call allele-count matrices ({0,1,2}, NaN = missing) via
the bed-reader backend.

Conventions:
- shape (n_samples, n_variants), float32
- missing -> NaN
- the dosage field carries BED hard calls only, not imputed/DS dosage
- no imputation here; downstream GRM code handles NaN explicitly
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def plink_path(prefix: str | Path, extension: str) -> Path:
    """Return a PLINK component path without treating dots in the prefix as suffixes.

    ``Path.with_suffix(".bed")`` turns a valid prefix such as ``panel.v1`` into
    ``panel.bed``.  PLINK prefixes are opaque strings, so extensions must be
    appended instead.  Passing an already-complete component path is accepted.
    """
    if extension not in {".bed", ".bim", ".fam"}:
        raise ValueError(f"unsupported PLINK extension {extension!r}")
    text = str(prefix)
    return Path(text if text.endswith(extension) else text + extension)


def read_delimited(path: str | Path, **kwargs):
    """Read a TSV or CSV, preserving caller-supplied pandas options.

    Known extensions use a deterministic delimiter; unfamiliar extensions are
    sniffed by pandas' Python engine.  This keeps the documented TSV/CSV
    phenotype contract consistent across fit, interaction, and workflow checks.
    """
    import pandas as pd

    suffix = Path(path).suffix.lower()
    if "sep" not in kwargs:
        if suffix == ".csv":
            kwargs["sep"] = ","
        elif suffix in {".tsv", ".tab", ".txt"}:
            kwargs["sep"] = "\t"
        else:
            kwargs["sep"] = None
            kwargs.setdefault("engine", "python")
    return pd.read_csv(path, **kwargs)


def plink_bim_sha256(prefix: str | Path) -> str:
    """Return the SHA-256 digest of the exact BIM that defines BED column order."""
    path = plink_path(prefix, ".bim")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    bed_path = plink_path(prefix, ".bed")
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
    bed_path = plink_path(prefix, ".bed")
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
    bed = plink_path(out, ".bed")
    if bed.exists():
        return bed
    out.parent.mkdir(parents=True, exist_ok=True)
    plink2 = shutil.which("plink2")
    if plink2 is None:
        raise FileNotFoundError("plink2 not found on PATH")
    cmd = [
        plink2, "--vcf", str(vcf_gz), "--make-bed",
        "--out", str(out), "--threads", str(threads),
    ]
    subprocess.run(cmd, check=True)
    return bed
