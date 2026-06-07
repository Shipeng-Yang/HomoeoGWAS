#!/usr/bin/env python3
"""Phase D: wheat HEB / triad-balance validation of the GWAS hits.

Question: do the wheat homoeolog-pair GWAS hits sit in triads whose A/B/D copies are
EXPRESSION-IMBALANCED (homoeolog expression bias, HEB), rather than balanced? If the
statistically-inferred interaction hits are enriched for expression-imbalanced triads,
that gives a biological mechanism (dosage imbalance) backing the "antagonistic /
imbalance" statistical signal.

Method (Ramirez-Gonzalez 2018 framework):
  - per triad, mean TPM of A, B, D copies across Development samples
  - normalize to relative fractions (A%, B%, D%) summing to 1
  - distance to the balanced point (1/3,1/3,1/3); triads far from center = imbalanced
  - categorize: balanced vs dominant(one high) vs suppressed(one low) by nearest ideal vertex
  - compare our 2 GWAS-hit triads + the broader hit set vs genome-wide background
Gene IDs: Development TPM uses v1.1 TraesCS...02G... = same as our GWAS. HCTriads is
v1.0 (01G) -> map 01G->02G.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
TPM = ROOT / "data/raw/expression/wheat/tpm/Development_tpm.tsv.gz"
HCTRIADS = ROOT / "data/raw/expression/wheat/HCTriads.csv"
OUT = ROOT / "results/phase7/bio_wheat"

# our two genome-wide INT hits (A,B,D triad members, v1.1 IDs)
HITS = {
    "chr1_BD": dict(A="TraesCS1A02G125400", B="TraesCS1B02G143800", D="TraesCS1D02G128400",
                    pair="B-D", p=2.42e-6, vif=1.13),
    "chr5_AD": dict(A="TraesCS5A02G169500", B="TraesCS5B02G166300", D="TraesCS5D02G173900",
                    pair="A-D", p=3.21e-7, vif=2.28),
}


def _v10_to_v11(g):
    return g.replace("01G", "02G", 1) if "01G" in g else g


def _load_tpm_mean():
    """Return dict gene_id -> mean TPM across Development samples."""
    means = {}
    with gzip.open(TPM, "rt") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            gid = parts[0]
            vals = np.array([float(x) for x in parts[1:] if x != ""], dtype=float)
            means[gid] = float(vals.mean()) if vals.size else 0.0
    return means


def _triad_balance(a, b, d):
    """Relative fractions + distance-to-center + 7-category (Ramirez-Gonzalez)."""
    tot = a + b + d
    if tot <= 1e-9:
        return None
    fa, fb, fd = a / tot, b / tot, d / tot
    center = np.array([1 / 3, 1 / 3, 1 / 3])
    frac = np.array([fa, fb, fd])
    dist = float(np.linalg.norm(frac - center))
    # nearest ideal vertex among 7: balanced, A/B/D-dominant (vertex), A/B/D-suppressed (edge mid)
    ideals = {
        "balanced": np.array([1 / 3, 1 / 3, 1 / 3]),
        "A.dominant": np.array([1, 0, 0]), "B.dominant": np.array([0, 1, 0]),
        "D.dominant": np.array([0, 0, 1]),
        "A.suppressed": np.array([0, 1 / 2, 1 / 2]), "B.suppressed": np.array([1 / 2, 0, 1 / 2]),
        "D.suppressed": np.array([1 / 2, 1 / 2, 0]),
    }
    cat = min(ideals, key=lambda k: np.linalg.norm(frac - ideals[k]))
    return dict(fa=fa, fb=fb, fd=fd, dist_center=dist, category=cat)


def main():
    print("=== wheat HEB / triad-balance validation ===")
    means = _load_tpm_mean()
    print(f"  loaded TPM means for {len(means)} genes")

    hc = pd.read_csv(HCTRIADS)
    hc = hc[hc["cardinality_abs"] == "1:1:1"].copy()
    rows = []
    for _, r in hc.iterrows():
        a = _v10_to_v11(str(r["A"])); b = _v10_to_v11(str(r["B"])); d = _v10_to_v11(str(r["D"]))
        ta, tb, td = means.get(a, np.nan), means.get(b, np.nan), means.get(d, np.nan)
        if not np.all(np.isfinite([ta, tb, td])):
            continue
        bal = _triad_balance(ta, tb, td)
        if bal is None:
            continue
        rows.append(dict(A=a, B=b, D=d, tpmA=ta, tpmB=tb, tpmD=td, **bal))
    df = pd.DataFrame(rows)
    print(f"  {len(df)} background 1:1:1 triads with expression")

    # background distribution
    bg_dist = df["dist_center"].values
    bg_cat = df["category"].value_counts(normalize=True).to_dict()
    print(f"  background category fractions: { {k: round(v,3) for k,v in bg_cat.items()} }")
    print(f"  background dist_center: median={np.median(bg_dist):.4f} "
          f"p75={np.percentile(bg_dist,75):.4f} p90={np.percentile(bg_dist,90):.4f}")

    # our hits
    print("\n  === GWAS hit triads ===")
    hit_out = {}
    for name, h in HITS.items():
        a, b, d = h["A"], h["B"], h["D"]
        ta, tb, td = means.get(a, np.nan), means.get(b, np.nan), means.get(d, np.nan)
        bal = _triad_balance(ta, tb, td) if np.all(np.isfinite([ta, tb, td])) else None
        if bal is None:
            print(f"  {name}: NO expression data"); continue
        pct = float((bg_dist < bal["dist_center"]).mean())  # percentile of imbalance
        hit_out[name] = dict(pair=h["pair"], p=h["p"], vif=h["vif"],
                             tpmA=ta, tpmB=tb, tpmD=td, **bal, imbalance_percentile=pct)
        print(f"  {name} ({h['pair']}, GWAS p={h['p']:.2g}):")
        print(f"     TPM A={ta:.2f} B={tb:.2f} D={td:.2f} -> fracs A={bal['fa']:.2f} B={bal['fb']:.2f} D={bal['fd']:.2f}")
        print(f"     category={bal['category']}  dist_center={bal['dist_center']:.4f}  "
              f"more imbalanced than {pct*100:.1f}% of background triads")

    out = dict(study="Development", n_background=len(df),
               background_category_fractions=bg_cat,
               background_dist_quantiles=dict(median=float(np.median(bg_dist)),
                                              p75=float(np.percentile(bg_dist, 75)),
                                              p90=float(np.percentile(bg_dist, 90))),
               hits=hit_out,
               note=("wheat HEB validation: do GWAS-hit triads show homoeolog expression "
                     "imbalance? dist_center = Euclidean distance of (A%,B%,D%) from balanced "
                     "(1/3,1/3,1/3); higher = more imbalanced. imbalance_percentile = fraction "
                     "of background triads LESS imbalanced than the hit. category per "
                     "Ramirez-Gonzalez 2018 (balanced/dominant/suppressed). If hits land high "
                     "on imbalance, the statistical interaction has a dosage-imbalance "
                     "biological correlate."))
    (OUT / "wheat_HEB_validation.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  wrote {OUT/'wheat_HEB_validation.json'}")


if __name__ == "__main__":
    main()
