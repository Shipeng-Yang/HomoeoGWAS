#!/usr/bin/env python3
"""M3.4 v2 — fix cotton SNP-id ('.' in BIM and sumstats → 'chrom:pos').

The hebau cotton plink2 import created BED/BIM with snp_id='.' (no IDs in
the source VCF). This blocks:
  - M3.3 v2 prepare (drop_duplicates collapses all SNPs to one row)
  - plink2 --ld-snp lookup (cannot reference '.')

Fix:
  1. rewrite BIM giving each SNP id = '<chrom>:<pos>'
  2. rewrite LOCO sumstats giving snp_id = '<chrom>:<pos>'

Idempotent: detects whether IDs are already non-empty and skips.
"""
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
COTTON_BED_ROOT = ROOT / "data/processed/cotton"
LOCO_ROOT = ROOT / "results/phase3/m3_1_loco_v2/cotton_hebau"
TRAITS = ["fiber_length_BLUE", "lint_percentage_BLUE"]
SUBGENOMES = ["A", "D"]


def fix_bim(sg: str):
    bim_path = COTTON_BED_ROOT / sg / "all.bim"
    backup = COTTON_BED_ROOT / sg / "all.bim.preid.bak"
    df = pd.read_csv(bim_path, sep="\t", header=None,
                     names=["chrom","snp_id","cm","pos","a1","a2"],
                     dtype={"snp_id": str, "chrom": str})
    # Codex review fix: skip only when ALL ids are non-dot. Mixed states
    # (a few '.' lines among real IDs) must trigger rewrite, not skip.
    non_dot = df["snp_id"] != "."
    if non_dot.all() and len(df) > 0:
        sample = df["snp_id"].iloc[:3].tolist()
        print(f"  {sg}: BIM already has IDs (sample={sample}, all non-dot), skipping")
        return
    if not backup.exists():
        shutil.copy2(bim_path, backup)
    df["snp_id"] = df["chrom"].astype(str) + ":" + df["pos"].astype(str)
    # Assert uniqueness: chrom:pos must be unique within a subgenome BIM
    n_dup = int(df["snp_id"].duplicated().sum())
    assert n_dup == 0, f"{sg}: {n_dup} duplicate chrom:pos snp_ids — BIM has multi-allelic sites?"
    df.to_csv(bim_path, sep="\t", index=False, header=False,
              columns=["chrom","snp_id","cm","pos","a1","a2"])
    print(f"  {sg}: BIM rewritten ({len(df)} rows, uniq) → all.bim (backup {backup.name})")


def fix_sumstats(trait: str):
    p = LOCO_ROOT / trait / f"sumstats_{trait}_loco.tsv"
    backup = p.with_suffix(".preid.bak")
    df = pd.read_csv(p, sep="\t", dtype={"snp_id": str, "chrom": str})
    # Same idempotence fix as BIM: skip only when ALL ids are non-dot
    if (df["snp_id"] != ".").all() and len(df) > 0:
        sample = df["snp_id"].iloc[:3].tolist()
        print(f"  {trait}: sumstats already has IDs (sample={sample}, all non-dot), skipping")
        return
    if not backup.exists():
        shutil.copy2(p, backup)
    df["snp_id"] = df["chrom"].astype(str) + ":" + df["pos"].astype(str)
    df.to_csv(p, sep="\t", index=False)
    print(f"  {trait}: sumstats rewritten ({len(df)} rows) → {p.name} (backup {backup.name})")


def main():
    print("=== cotton snp_id fix (BIM + LOCO sumstats) ===")
    for sg in SUBGENOMES:
        fix_bim(sg)
    for tr in TRAITS:
        fix_sumstats(tr)
    print("DONE")


if __name__ == "__main__":
    main()
