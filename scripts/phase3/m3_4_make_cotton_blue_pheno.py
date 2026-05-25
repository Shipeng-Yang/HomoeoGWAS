#!/usr/bin/env python3
"""M3.4 Step 1 — build cotton BLUE-across-years phenotype TSV.

Inputs:  data/processed/cotton/pheno_clean.tsv  (26 traits = 13 × {14,15})
Outputs: data/processed/cotton/pheno_m3_4_blue.tsv
         data/processed/cotton/pheno_m3_4_blue_summary.json

Strategy:  for each trait family {fiber_length, fiber_strength, lint_percentage, ...},
  blue = mean(trait_14, trait_15) wherever both years are non-null;
         or single-year value where the other is null (preserve n).
  Records per-trait per-year correlation + non-null counts for QC.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
# Trait families = column name with year suffix stripped
TRAIT_FAMILIES = [
    "fiber_length", "fiber_strength", "micronaire", "elongation",
    "length_uniformity", "maturity", "spinning_consistency",
    "boll_weight", "lint_percentage", "seed_index", "lint_index",
    "fiber_weight_per_boll", "fiber_density",
]


def main():
    ap = argparse.ArgumentParser(description="M3.4 cotton BLUE pheno")
    ap.add_argument("--in", dest="in_path",
                    default=str(ROOT / "data/processed/cotton/pheno_clean.tsv"))
    ap.add_argument("--out", default=str(
        ROOT / "data/processed/cotton/pheno_m3_4_blue.tsv"))
    args = ap.parse_args()
    df = pd.read_csv(args.in_path, sep="\t")
    print(f"=== cotton BLUE pheno ===  {args.in_path}  rows={len(df)}")
    out = pd.DataFrame({"sample": df["sample"]})
    summary: dict = {"n_samples": int(len(df)), "traits": {}}
    for fam in TRAIT_FAMILIES:
        c14, c15 = f"{fam}_14", f"{fam}_15"
        if c14 not in df.columns or c15 not in df.columns:
            print(f"  WARN: trait family {fam} missing year columns")
            continue
        x14 = df[c14].to_numpy(dtype=np.float64)
        x15 = df[c15].to_numpy(dtype=np.float64)
        both = np.isfinite(x14) & np.isfinite(x15)
        only14 = np.isfinite(x14) & ~np.isfinite(x15)
        only15 = ~np.isfinite(x14) & np.isfinite(x15)
        blue = np.full(len(df), np.nan)
        blue[both] = (x14[both] + x15[both]) / 2.0
        blue[only14] = x14[only14]
        blue[only15] = x15[only15]
        col = f"{fam}_BLUE"
        out[col] = blue
        if both.sum() >= 5:
            year_corr = float(np.corrcoef(x14[both], x15[both])[0, 1])
        else:
            year_corr = float("nan")
        summary["traits"][col] = {
            "n_blue_nonnull": int(np.isfinite(blue).sum()),
            "n_both_years": int(both.sum()),
            "n_only_14": int(only14.sum()),
            "n_only_15": int(only15.sum()),
            "year_corr_14_vs_15": year_corr,
            "mean": float(np.nanmean(blue)),
            "std": float(np.nanstd(blue)),
            "min": float(np.nanmin(blue)),
            "max": float(np.nanmax(blue)),
        }
        print(f"  {col}: n_nonnull={summary['traits'][col]['n_blue_nonnull']:>4} "
              f"both={both.sum():>4} only14={only14.sum():>3} only15={only15.sum():>3} "
              f"corr14v15={year_corr:.3f}")
    out.to_csv(args.out, sep="\t", index=False)
    print(f"\nwrote {args.out}  ({len(out.columns)-1} BLUE traits)")
    summary_path = Path(args.out).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
