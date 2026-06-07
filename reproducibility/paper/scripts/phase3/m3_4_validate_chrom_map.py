#!/usr/bin/env python3
"""M3.4 Step 0 — validate chrom map TSV against pvar/bim + fasta lengths.

Hard gate: for each panel_chrom in the map, max(POS) across the panel's
pvar/bim must be ≤ fasta_chrom length. A mismatch means the map is wrong
(species/chromosome confusion) and we abort before any downstream M3.4
work.

Also unzips the .fna.gz to a flat .fa next to it on first call (idempotent;
pysam.FastaFile needs uncompressed FASTA for samtools-style indexing).
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def ensure_fasta_unzipped(fna_gz: Path) -> Path:
    """Decompress .fna.gz to .fa (idempotent). Return uncompressed path."""
    out = fna_gz.with_suffix("")               # strip .gz → .fna
    if out.suffix == ".fna":
        out = out.with_suffix(".fa")
    if out.exists() and out.stat().st_size > 0:
        return out
    print(f"  decompressing {fna_gz} → {out}", flush=True)
    with gzip.open(fna_gz, "rb") as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=64 << 20)
    return out


def main():
    ap = argparse.ArgumentParser(description="M3.4 validate chrom_map TSV")
    ap.add_argument("--chrom-map", required=True,
                    help="TSV with panel_chrom / fasta_chrom / subgenome columns")
    ap.add_argument("--bim-or-pvar", action="append", required=True,
                    help="path to a .bim or .pvar (one per subgenome); "
                         "may be repeated")
    ap.add_argument("--fasta-gz", required=True,
                    help="path to .fna.gz (will be unzipped to .fa next to it)")
    args = ap.parse_args()

    chrom_map = pd.read_csv(args.chrom_map, sep="\t")
    print(f"=== {args.chrom_map}: {len(chrom_map)} entries ===")

    fa_gz = Path(args.fasta_gz)
    if not fa_gz.exists():
        sys.exit(f"ERR: fasta not found: {fa_gz}")
    fa_path = ensure_fasta_unzipped(fa_gz)
    import pysam
    fa = pysam.FastaFile(str(fa_path))         # auto-creates .fai
    fasta_chrom_len: dict[str, int] = dict(zip(fa.references, fa.lengths, strict=True))

    # Collect pvar/bim max pos per panel_chrom
    panel_max_pos: dict[str, int] = {}
    for bp in args.bim_or_pvar:
        bp_path = Path(bp)
        if not bp_path.exists():
            sys.exit(f"ERR: bim/pvar not found: {bp_path}")
        # detect format: pvar has ## header lines, bim doesn't
        with open(bp_path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                break
        if bp_path.suffix == ".pvar":
            df = pd.read_csv(bp_path, sep="\t", comment="#", header=None,
                              usecols=[0, 1], names=["chrom", "pos"])
        else:                                   # bim
            df = pd.read_csv(bp_path, sep="\t", header=None,
                              names=["chrom","snp","cm","pos","a1","a2"],
                              usecols=["chrom", "pos"])
        for chrom, sub in df.groupby("chrom"):
            chrom = str(chrom)
            mx = int(sub["pos"].max())
            panel_max_pos[chrom] = max(panel_max_pos.get(chrom, 0), mx)
        print(f"  parsed {bp_path.name}: {len(df)} rows, chroms={df['chrom'].nunique()}")

    # Validate each map row
    print("\n=== validation ===")
    errors: list[str] = []
    rows: list[dict] = []
    for _, row in chrom_map.iterrows():
        pc = str(row["panel_chrom"])
        fc = str(row["fasta_chrom"])
        max_pos = panel_max_pos.get(pc)
        fa_len = fasta_chrom_len.get(fc)
        if fa_len is None:
            errors.append(f"{pc}: fasta {fc} not in fasta index")
            status = "FASTA_MISSING"
        elif max_pos is None:
            status = "NO_VARIANTS"
        elif max_pos > fa_len:
            errors.append(f"{pc}: max_pos {max_pos:,} > fasta {fc} length {fa_len:,}")
            status = "OVERFLOW"
        else:
            status = "OK"
        rows.append({
            "panel_chrom": pc, "fasta_chrom": fc,
            "panel_max_pos": max_pos, "fasta_chrom_length": fa_len,
            "status": status,
        })
    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    print(f"\nstatus counts: {out_df['status'].value_counts().to_dict()}")
    if errors:
        sys.exit(f"\nERR: {len(errors)} validation errors:\n  "
                 + "\n  ".join(errors))
    print("chrom_map validation PASS")


if __name__ == "__main__":
    main()
