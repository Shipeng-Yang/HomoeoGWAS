#!/usr/bin/env python3
"""Phase C wheat AABBDD Step 1 (pilot 1A-1B-1D): GFF -> gene-region SNP extraction
via plink2 -> per-gene burden + position-homology triad construction.

wheat-specific challenges:
  - 86.5M SNP total (A 34M / B 43M / D 9M) -> CANNOT memory-map; use plink2 --extract
    range to pull only gene-body±flank SNPs into small per-subgenome beds.
  - CHROM naming: pvar = 'chr1A'/'chr1B'/'chr1D' (chr prefix); GFF = '1A'/'1B'/'1D'.
  - triad = position-homology: genes ranked by chrom position within 1A/1B/1D, then
    aligned by relative rank (1A/1B/1D are highly collinear). gene_id (TraesCS1A..)
    only validates subgenome+chrom, NOT used for pairing (numbering is per-chrom
    independent, NOT triad-matched). FRAMING: this is synteny/position-homology, a
    documented wheat approximation, NOT curated ortholog (MCScanX/Compara = later).
  - 3 subgenomes -> triad-organized 3 pairwise (A-B, A-D, B-D) downstream.
Pilot scope: chromosomes 1A, 1B, 1D only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
GFF = ROOT / "data/reference/wheat/annotation/Triticum_aestivum.IWGSC.57.gff3.gz"
PLINK2 = Path.home() / ".local/share/mamba/envs/polygwas-cpu/bin/plink2"
OUT = ROOT / "results/phase7/bio_wheat"
WORK = Path("/mnt/nvme/wheat_pilot")
SUB_BED = {s: ROOT / f"data/processed/wheat/{s}/all" for s in ("A", "B", "D")}

_GENE_ID_RE = re.compile(r"gene_id=([^;]+)")
# pilot chromosomes: GFF name (no prefix) -> subgenome
PILOT_CHROMS = {"1A": "A", "1B": "B", "1D": "D"}


def _parse_gff_pilot(flank_bp):
    """Return per-subgenome list of (gene_id, gff_chrom, start, end, strand,
    flank_start, flank_end) for protein_coding genes on pilot chroms."""
    by_sub = {"A": [], "B": [], "D": []}
    with gzip.open(GFF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene" or p[0] not in PILOT_CHROMS:
                continue
            if "biotype=protein_coding" not in p[8]:
                continue
            m = _GENE_ID_RE.search(p[8])
            if not m:
                continue
            sub = PILOT_CHROMS[p[0]]
            gs, ge = int(p[3]), int(p[4])
            by_sub[sub].append((m.group(1), p[0], gs, ge, p[6],
                                max(1, gs - flank_bp), ge + flank_bp))
    return by_sub


def _write_range_file(genes, path, mode):
    """plink2 --extract range file: CHROM(chr-prefixed) START END NAME.
    mode 'body' uses gene start/end; 'flank' uses flank_start/flank_end."""
    with open(path, "w") as f:
        for gid, gff_chrom, gs, ge, strand, fs, fe in genes:
            chrom = f"chr{gff_chrom}"
            if mode == "body":
                f.write(f"{chrom}\t{gs}\t{ge}\t{gid}\n")
            else:
                f.write(f"{chrom}\t{fs}\t{fe}\t{gid}\n")


def _plink2_extract(sub, chrom_chr, range_file, out_prefix):
    """Extract gene-region SNPs for one subgenome+chrom into a small bed."""
    cmd = [str(PLINK2), "--pfile", str(SUB_BED[sub]), "--chr", chrom_chr,
           "--extract", "range", str(range_file), "--make-bed", "--out", str(out_prefix),
           "--threads", "16", "--memory", "60000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  PLINK2 ERROR ({sub}):\n{r.stderr[-800:]}"); return False
    return True


def _map_snps_to_genes(bim_path, genes, mode):
    """Map extracted SNPs (bim) to genes by position overlap. Returns
    gene_id -> list of bim-row indices."""
    import bisect
    # genes: (gid, gff_chrom, gs, ge, strand, fs, fe); use mode window
    intervals = []
    for gid, gff_chrom, gs, ge, strand, fs, fe in genes:
        ws, we = (gs, ge) if mode == "body" else (fs, fe)
        intervals.append((gid, ws, we))
    intervals.sort(key=lambda x: x[1])
    starts = [iv[1] for iv in intervals]
    gene2snp = {iv[0]: [] for iv in intervals}
    # read bim
    pos = []
    with open(bim_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            pos.append(int(p[3]))
    for si, ps in enumerate(pos):
        j = bisect.bisect_right(starts, ps) - 1
        steps = 0
        while j >= 0 and steps < 3000:
            gid, ws, we = intervals[j]
            if ps > we and ps - ws > 2_000_000:
                break
            if ws <= ps <= we:
                gene2snp[gid].append(si)
            j -= 1; steps += 1
    return gene2snp, len(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flank-bp", type=int, default=2000)
    ap.add_argument("--min-snp", type=int, default=3)
    ap.add_argument("--mode", choices=["body", "flank"], default="flank",
                    help="window for SNP-to-gene (pilot default flank to match sparse coverage)")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase C wheat pilot ETL (1A-1B-1D, mode={args.mode}, flank={args.flank_bp}) ===")

    genes_by_sub = _parse_gff_pilot(args.flank_bp)
    for s in ("A", "B", "D"):
        print(f"  GFF 1{s}: {len(genes_by_sub[s])} protein_coding genes")

    sys.path.insert(0, str(ROOT / "scripts/phase7"))
    from homoeogwas.io import load_bed_hardcall

    etl = {}
    for sub in ("A", "B", "D"):
        chrom_chr = f"chr1{sub}"
        rf = WORK / f"range_1{sub}_{args.mode}.txt"
        _write_range_file(genes_by_sub[sub], rf, args.mode)
        outp = WORK / f"pilot_1{sub}_{args.mode}"
        print(f"  plink2 extracting {chrom_chr} ({len(genes_by_sub[sub])} gene ranges)...")
        if not _plink2_extract(sub, chrom_chr, rf, outp):
            print(f"  ABORT: plink2 failed for {sub}"); return
        bed = load_bed_hardcall(outp)
        X = np.asarray(bed.dosage, dtype=np.float64)  # n_samp x n_snp
        samples = list(np.asarray(bed.samples))
        gene2snp, n_snp = _map_snps_to_genes(f"{outp}.bim", genes_by_sub[sub], args.mode)
        retained = {g: idx for g, idx in gene2snp.items() if len(idx) >= args.min_snp}
        snp_counts = np.array([len(v) for v in retained.values()])
        print(f"    {chrom_chr}: extracted {n_snp} SNP, {X.shape[0]} samples; "
              f"genes with ≥{args.min_snp} SNP = {len(retained)}/{len(genes_by_sub[sub])} "
              f"(SNP/gene p50={int(np.percentile(snp_counts,50)) if len(snp_counts) else 0} "
              f"p90={int(np.percentile(snp_counts,90)) if len(snp_counts) else 0} "
              f"max={int(snp_counts.max()) if len(snp_counts) else 0})")
        # store
        gene_meta = {}
        for gid, gff_chrom, gs, ge, strand, fs, fe in genes_by_sub[sub]:
            gene_meta[gid] = dict(chrom=gff_chrom, start=gs, end=ge, strand=strand)
        etl[sub] = dict(X=X, samples=samples, gene2snp=retained, gene_meta=gene_meta,
                        n_snp_extracted=n_snp)
        # save per-subgenome npz
        keys = list(retained.keys())
        # sort genes by position for position-homology ranking later
        keys.sort(key=lambda g: gene_meta[g]["start"])
        arrs = np.array([np.array(retained[g], int) for g in keys], dtype=object)
        starts = np.array([gene_meta[g]["start"] for g in keys])
        np.savez(out / f"pilot_1{sub}_{args.mode}.npz",
                 gene_ids=np.array(keys), snp_idx=arrs, starts=starts,
                 samples=np.array(samples), allow_pickle=True)
        # also persist the dosage as .npy (small after extraction)
        np.save(out / f"pilot_1{sub}_{args.mode}_X.npy", X)

    # summary
    summ = dict(mode=args.mode, flank_bp=args.flank_bp, min_snp=args.min_snp,
                pilot_chroms=list(PILOT_CHROMS),
                per_sub={s: dict(n_genes_gff=len(genes_by_sub[s]),
                                 n_retained=len(etl[s]["gene2snp"]),
                                 n_snp_extracted=etl[s]["n_snp_extracted"],
                                 n_samples=len(etl[s]["samples"])) for s in ("A", "B", "D")})
    (out / "etl_summary_pilot.json").write_text(json.dumps(summ, indent=2))
    print(f"  ETL SUMMARY: {summ['per_sub']}")
    print(f"  DONE -> {out}")


if __name__ == "__main__":
    main()
