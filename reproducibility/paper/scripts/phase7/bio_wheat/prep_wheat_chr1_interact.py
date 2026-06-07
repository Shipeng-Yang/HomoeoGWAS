#!/usr/bin/env python3
"""Prepare wheat chr1 data in the `homoeogwas interact` (mode=triad) input format, to
validate the hexaploid triad CLI path on REAL hexaploid data (reproduce the chr1 B-D hit).

Produces, under results/phase7/interact_validate/wheat_chr1/:
  {A,B,D}/all.{bed,bim,fam}    chr1 per-subgenome PLINK1 bed (plink2 --chr extract)
  snp_to_gene_{A,B,D}.npz      gene_ids + snp_idx (flank +-2kb windows) into that bed
  triads_chr1.tsv              gene_A/gene_B/gene_D from HCTriads chr1 1:1:1
"""
from __future__ import annotations

import bisect
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
PLINK2 = Path.home() / ".local/share/mamba/envs/polygwas-cpu/bin/plink2"
SUB_PFILE = {s: ROOT / f"data/processed/wheat/{s}/all" for s in ("A", "B", "D")}
HCTRIADS = ROOT / "data/raw/expression/wheat/HCTriads.csv"
OUT = ROOT / "results/phase7/interact_validate/wheat_chr1"
FLANK = 2000

_s4 = importlib.util.spec_from_file_location(
    "w4", str(ROOT / "scripts/phase7/bio_wheat/04w_wheat_genomewide.py"))
_w4 = importlib.util.module_from_spec(_s4); _s4.loader.exec_module(_w4)


def _v11(g):
    return g.replace("01G", "02G", 1) if "01G" in g else g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ("A", "B", "D"):
        outp = OUT / s / "all"
        outp.parent.mkdir(parents=True, exist_ok=True)
        # genes on chr1 of this subgenome (flank windows), via 04w GFF parse
        genes = [g for g in _w4._parse_gff(s, FLANK) if g[1] == f"1{s}"]  # (gid,chrom,gs,ge,fs,fe)
        # extract ONLY gene-region (flank) SNPs (matches 03w/04w gene-region GRM; small bed)
        rf = OUT / f"range_{s}.txt"
        with open(rf, "w") as f:
            for gid, chrom, gs, ge, fs, fe in genes:
                f.write(f"chr1{s}\t{fs}\t{fe}\t{gid}\n")
        print(f"[{s}] plink2 --extract range (chr1 gene flank windows) -> bed ...", flush=True)
        cmd = [str(PLINK2), "--pfile", str(SUB_PFILE[s]), "--extract", "range", str(rf),
               "--make-bed", "--out", str(outp), "--threads", "32", "--memory", "200000"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"plink2 {s}: {r.stderr[-500:]}")
        # bed bim: positions in column order
        pos = []
        with open(f"{outp}.bim") as f:
            for line in f:
                pos.append(int(line.rstrip("\n").split("\t")[3]))
        pos = np.array(pos)
        order = np.argsort(pos)
        pos_sorted = pos[order]
        gc = sorted([(g[0], g[4], g[5]) for g in genes], key=lambda x: x[1])  # (gid, fs, fe)
        starts = [g[1] for g in gc]
        gene2snp = {}
        for j, ps in enumerate(pos_sorted):
            k = bisect.bisect_right(starts, ps) - 1
            steps = 0
            while k >= 0 and steps < 3000:
                gid, ws, we = gc[k]
                if ps > we and ps - ws > 2_000_000:
                    break
                if ws <= ps <= we:
                    gene2snp.setdefault(gid, []).append(int(order[j]))
                k -= 1
                steps += 1
        retained = {g: idx for g, idx in gene2snp.items() if len(idx) >= 3}
        gene_ids = list(retained.keys())
        snp_idx = np.array([np.array(retained[g], int) for g in gene_ids], dtype=object)
        np.savez(OUT / f"snp_to_gene_{s}.npz", gene_ids=np.array(gene_ids),
                 snp_idx=snp_idx, allow_pickle=True)
        print(f"  {s}: {len(pos)} chr1 SNPs, {len(gene_ids)} genes retained (>=3 SNP flank)")

    # triads: HCTriads chr1 1:1:1, v1.1 ids
    hc = pd.read_csv(HCTRIADS)
    hc = hc[hc["cardinality_abs"] == "1:1:1"].copy()
    for c in ("A", "B", "D"):
        hc[c] = hc[c].map(_v11)
    chr1 = hc[hc["A"].str.contains("TraesCS1A")]
    chr1[["A", "B", "D"]].rename(columns={"A": "gene_A", "B": "gene_B", "D": "gene_D"}).to_csv(
        OUT / "triads_chr1.tsv", sep="\t", index=False)
    print(f"triads_chr1.tsv: {len(chr1)} chr1 1:1:1 triads")
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
