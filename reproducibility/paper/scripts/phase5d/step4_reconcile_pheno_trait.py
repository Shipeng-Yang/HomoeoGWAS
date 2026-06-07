"""Phase 5d Step 4 — reconcile pheno samples to VCF + trait alias.

Two reconciliations:

1) Sample diff: pheno 738 vs VCF 737. The extra pheno entry is "Sd" which
   has full env data but is not in the VCF (paper phenotyped but failed
   genotype QC). Drop it from pheno_clean.tsv.

2) Trait alias: paper File S3 GWAS results use "FROST_DAYS" while paper
   File S1 Table C (and our pheno_clean.tsv) uses the short code "FD".
   Rename the qtl column to "FD" and record the original paper trait
   name in a new `paper_trait_name` column so the original mapping is
   preserved.

Backups: pheno_clean.tsv.prev_step4_bak, known_qtl_oat.tsv.prev_step4_bak
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

PROJ = Path("/mnt/7302share/fast_ysp/U7_GWAS")
PHENO = PROJ / "data/processed/oat/pheno_clean.tsv"
QTL = PROJ / "data/reference/oat/known_qtl_oat.tsv"

# --- 1) Drop "Sd" from pheno ---
shutil.copy(PHENO, str(PHENO) + ".prev_step4_bak")
ph = pd.read_csv(PHENO, sep="\t")
n_before = len(ph)
ph = ph[ph["sample_id"] != "Sd"].reset_index(drop=True)
ph.to_csv(PHENO, sep="\t", index=False)
print(f"[pheno] {n_before} → {len(ph)} samples (dropped 'Sd'); wrote {PHENO}")

# --- 2) Fix qtl trait alias FROST_DAYS → FD ---
shutil.copy(QTL, str(QTL) + ".prev_step4_bak")
qt = pd.read_csv(QTL, sep="\t")
qt["paper_trait_name"] = qt["trait"]   # preserve paper name
n_frost = (qt["trait"] == "FROST_DAYS").sum()
qt.loc[qt["trait"] == "FROST_DAYS", "trait"] = "FD"
qt.to_csv(QTL, sep="\t", index=False)
print(f"[qtl] renamed {n_frost} 'FROST_DAYS' → 'FD'; preserved in paper_trait_name col; wrote {QTL}")

# --- 3) Verify ---
print("\n[verify] new pheno trait set vs qtl trait set:")
pheno_traits = set(ph.columns) - {"sample_id"}
qtl_traits = set(qt["trait"].dropna().unique())
left = qtl_traits - pheno_traits
right = pheno_traits - qtl_traits
print(f"  pheno traits ({len(pheno_traits)}): {sorted(pheno_traits)}")
print(f"  qtl traits ({len(qtl_traits)}):   {sorted(qtl_traits)}")
print(f"  in qtl not pheno (should be 0): {sorted(left)}")
print(f"  in pheno not qtl (no anchor): {sorted(right)}")
