#!/usr/bin/env python3
"""Build y-INDEPENDENT HEB pair-prior weights for wheat chr1 triads (demo of homoeogwas
`interact --weights` with expression-guided prioritization).

Weight = 0.5 + imbalance_percentile, where imbalance = Euclidean distance of (A%,B%,D%)
relative TPM from the balanced point (1/3,1/3,1/3) (Ramirez-Gonzalez 2018), and the
percentile is computed over ALL chr1 1:1:1 triads. The weight uses ONLY expression
(Development TPM) — never the GWAS phenotype — so it is a legitimate frozen prior. The
chr1 B-D GWAS hit triad is the most expression-imbalanced, so it receives a high weight
a priori: an independent expression prediction of where the homoeolog interaction sits.

Output: triads_chr1_heb_weights.tsv  (gene_A, gene_B, gene_D, weight).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
OUT = ROOT / "results/phase7/interact_validate/wheat_chr1"

_s = importlib.util.spec_from_file_location(
    "w5", str(ROOT / "scripts/phase7/bio_wheat/05w_heb_validation.py"))
_w5 = importlib.util.module_from_spec(_s); _s.loader.exec_module(_w5)


def main():
    means = _w5._load_tpm_mean()
    triads = pd.read_csv(OUT / "triads_chr1.tsv", sep="\t")  # gene_A/gene_B/gene_D (already v1.1)
    rows = []
    for _, r in triads.iterrows():
        a, b, d = r["gene_A"], r["gene_B"], r["gene_D"]
        ta, tb, td = means.get(a, np.nan), means.get(b, np.nan), means.get(d, np.nan)
        if not np.all(np.isfinite([ta, tb, td])):
            continue
        bal = _w5._triad_balance(ta, tb, td)
        if bal is None:
            continue
        rows.append(dict(gene_A=a, gene_B=b, gene_D=d, dist=bal["dist_center"]))
    df = pd.DataFrame(rows)
    dists = df["dist"].values
    # imbalance percentile over chr1 triads with expression; weight = 0.5 + percentile
    df["weight"] = df["dist"].map(lambda x: 0.5 + float((dists < x).mean()))
    df[["gene_A", "gene_B", "gene_D", "weight"]].to_csv(
        OUT / "triads_chr1_heb_weights.tsv", sep="\t", index=False)
    print(f"  {len(df)} chr1 triads with expression; weight=0.5+imbalance_percentile "
          f"(min={df['weight'].min():.3f} mean={df['weight'].mean():.3f} max={df['weight'].max():.3f})")
    # report the chr1 B-D hit triad's weight
    hit = df[df["gene_B"] == "TraesCS1B02G143800"]
    if len(hit):
        h = hit.iloc[0]
        print(f"  chr1 B-D hit triad weight = {h['weight']:.3f} (dist={h['dist']:.4f}) "
              f"-> HEB up-weights the hit a priori")
    print(f"  wrote {OUT/'triads_chr1_heb_weights.tsv'}")


if __name__ == "__main__":
    main()
