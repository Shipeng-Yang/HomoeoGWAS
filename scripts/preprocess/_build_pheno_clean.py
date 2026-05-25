#!/usr/bin/env python3
"""Build pheno_clean.tsv for cotton / rapeseed Horvath / wheat Watkins panels.

输出:
- data/processed/cotton/pheno_clean.tsv             宽表: sample + 13 trait × 2 年
- data/processed/rapeseed/horvath/pheno_clean.tsv   宽表: sample + 11 pheno 列
- data/processed/wheat/pheno_clean.tsv              宽表: sample + days_to_emerg_<env> × 4 + days_to_emerg (mean)

数据契约:
- sample 列在每个 pheno_clean.tsv 中**保证唯一**;若上游有重复(如 Horvath ars403
  在原始 xlsx 中有两行 site 测量),按 sample 取数值均值合并。
- wheat join key 用 StoreCode (WATDE0xxx), 对齐 VCF sample id;
  现代品种(无 StoreCode)不在 wheat pheno_clean.tsv 中.
"""
from __future__ import annotations
import argparse
import gzip
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
COTTON_PHENO_ZIP = ROOT / "data/raw/cotton/phenotype/phenotypic-data.zip"
WHEAT_PHENO_XLSX = (
    ROOT
    / "data/raw/wheat/phenotype/Natural_Populations"
    / "Watkins_Collection_WGIN_WISP_DFW_watseq_phenotype_data_JIC.xlsx"
)
WHEAT_VCF_FOR_IDS = (
    ROOT
    / "data/raw/wheat/vcf"
    / "chr1B.SNP.Missing-unphasing.ID.ann.finalSID.allele2_retain.hard_retain.InbreedingCoeff_retain.missing_retain.maf_retain.vcf.gz"
)


def load_vcf_samples(vcf_gz: Path) -> list[str]:
    """Read VCF header, return sample ID list."""
    with gzip.open(vcf_gz, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
            if not line.startswith("#"):
                break
    return []


def _extract_cotton_pheno() -> Path:
    out_dir = ROOT / "data/raw/cotton/phenotype/extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / "phenotypic_data.xlsx"
    if not out_xlsx.exists():
        with zipfile.ZipFile(COTTON_PHENO_ZIP) as z:
            for name in z.namelist():
                if name.endswith(".xlsx"):
                    with z.open(name) as src, open(out_xlsx, "wb") as dst:
                        dst.write(src.read())
                    break
    return out_xlsx


def _dedup_by_mean(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df[key].is_unique:
        return df
    num_cols = [c for c in df.columns if c != key and pd.api.types.is_numeric_dtype(df[c])]
    return df.groupby(key, as_index=False)[num_cols].mean()


# ---------------- cotton ----------------
def build_cotton() -> None:
    pheno_xlsx = _extract_cotton_pheno()
    out_tsv = ROOT / "data/processed/cotton/pheno_clean.tsv"
    vcf = ROOT / "data/processed/cotton/A/all.vcf.gz"

    vcf_samples = set(load_vcf_samples(vcf))
    SHEET_TRAIT = {
        "1-FL": "fiber_length", "2-FS": "fiber_strength", "3-M": "micronaire",
        "4-E": "elongation", "5-LU": "length_uniformity", "6-MAT": "maturity",
        "7-SCI": "spinning_consistency", "8-BW": "boll_weight",
        "9-LP": "lint_percentage", "10-SI": "seed_index", "11-LI": "lint_index",
        "12-FWPB": "fiber_weight_per_boll", "13-FD": "fiber_density",
    }

    merged = None
    for sh, trait in SHEET_TRAIT.items():
        df = pd.read_excel(pheno_xlsx, sheet_name=sh)
        df.columns = ["sample"] + [f"{trait}_{c.split('-')[1]}" for c in df.columns[1:]]
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = _dedup_by_mean(df, "sample")
        merged = df if merged is None else merged.merge(df, on="sample", how="outer")

    merged = merged[merged["sample"].isin(vcf_samples)]
    merged = _dedup_by_mean(merged, "sample")
    assert merged["sample"].is_unique, "cotton pheno_clean still has duplicate samples"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_tsv, sep="\t", index=False, na_rep="NA")
    print(
        f"cotton: wrote {out_tsv} shape={merged.shape}, "
        f"unique-samples-in-VCF={merged['sample'].nunique()}/{len(vcf_samples)}"
    )


# ---------------- rapeseed horvath ----------------
def build_rapeseed_horvath() -> None:
    pheno_xlsx = ROOT / "data/raw/rapeseed/Horvath2020_Zenodo4302088/pheno_lines_phenotypes.xlsx"
    out_tsv = ROOT / "data/processed/rapeseed/horvath/pheno_clean.tsv"
    vcf = ROOT / "data/processed/rapeseed/horvath/A/all.vcf.gz"

    vcf_samples_lc = set(s.lower() for s in load_vcf_samples(vcf))
    df = pd.read_excel(pheno_xlsx, sheet_name=0)

    df.rename(columns={
        "ARS#": "sample",
        "WINTER SURVIVAL": "winter_survival",
        "GROWTH HABIT": "growth_habit",
        "50% BLOOM": "bloom_50pct",
        "Fall vigor (10-8-15)": "fall_vigor",
        "Bloom": "bloom",
        "ave bloom date": "ave_bloom_date",
        "Stem elongation (12-9-15)": "stem_elongation",
        "Growth Habit (3-11-16)": "growth_habit_2",
        "ave growth habit": "ave_growth_habit",
        "Fall stand": "fall_stand",
        "Plant height": "plant_height",
    }, inplace=True)

    df["sample"] = df["sample"].astype(str).str.lower()
    df = df[df["sample"].isin(vcf_samples_lc)]

    for c in df.columns:
        if c not in ("sample", "NAME", "SOURCE", "Bloom notes"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    drop_cols = [c for c in ("Unnamed: 6", "NAME", "SOURCE", "Bloom notes") if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = _dedup_by_mean(df, "sample")
    assert df["sample"].is_unique
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False, na_rep="NA")
    print(
        f"rapeseed horvath: wrote {out_tsv} shape={df.shape}, "
        f"unique-samples-in-VCF={df['sample'].nunique()}/{len(vcf_samples_lc)}"
    )


# ---------------- wheat watkins ----------------
WHEAT_SHEETS = [
    ("WGIN_Watkins_JIC_CFLN06", "CFLN06"),
    ("WISP_Watkins_JIC_CFLN10", "CFLN10"),
    ("DFW_Watkins_JI_CFLN14",   "CFLN14"),
    ("DFW_Watkins_JI_CFLN20",   "CFLN20"),
]


def _wheat_extract_dto(df: pd.DataFrame, env_tag: str) -> pd.DataFrame:
    """从一个 Watkins phenotype sheet 找 StoreCode + Hd_dto_day(s)-* 列."""
    cols = list(df.columns)
    storecode_col = next((c for c in cols if str(c).strip().lower() == "storecode"), None)
    dto_col = next(
        (c for c in cols if re.match(r"^Hd_dto_days?-.*", str(c), re.IGNORECASE)),
        None,
    )
    if storecode_col is None or dto_col is None:
        raise ValueError(f"missing StoreCode or Hd_dto column in env={env_tag}: {cols}")
    out = pd.DataFrame({
        "sample": df[storecode_col].astype(str).str.strip(),
        f"days_to_emerg_{env_tag}": pd.to_numeric(df[dto_col], errors="coerce"),
    })
    out = out[out["sample"].str.match(r"^WATDE\d{4}$", na=False)]
    return _dedup_by_mean(out, "sample")


def build_wheat_watkins(sample_vcf: Path | None = None) -> None:
    out_tsv = ROOT / "data/processed/wheat/pheno_clean.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    # 取 VCF sample 列表(从 split 输出 或 fallback 到 raw chr1B)
    vcf_for_ids = sample_vcf
    if vcf_for_ids is None:
        split_vcf = ROOT / "data/processed/wheat/A/all.vcf.gz"
        vcf_for_ids = split_vcf if split_vcf.exists() else WHEAT_VCF_FOR_IDS
    vcf_samples = load_vcf_samples(vcf_for_ids)
    print(f"wheat: using {vcf_for_ids} for sample list (n={len(vcf_samples)})")

    # 用 StoreCode (WATDE0xxx) 作 join key
    watde_in_vcf = {s for s in vcf_samples if re.match(r"^WATDE\d{4}$", s)}
    print(f"wheat: WATDE samples in VCF = {len(watde_in_vcf)}; non-WATDE = {len(vcf_samples)-len(watde_in_vcf)}")

    merged: pd.DataFrame | None = None
    for sheet, env in WHEAT_SHEETS:
        df_raw = pd.read_excel(WHEAT_PHENO_XLSX, sheet_name=sheet, header=0)
        env_df = _wheat_extract_dto(df_raw, env)
        env_df = env_df[env_df["sample"].isin(watde_in_vcf)]
        print(f"  sheet {sheet}: {len(env_df)} WATDE rows with {env} dto")
        merged = env_df if merged is None else merged.merge(env_df, on="sample", how="outer")

    if merged is None or merged.empty:
        raise RuntimeError("wheat: no rows matched StoreCode∩VCF; check input")

    env_cols = [c for c in merged.columns if c.startswith("days_to_emerg_")]
    merged["days_to_emerg"] = merged[env_cols].mean(axis=1)
    merged = _dedup_by_mean(merged, "sample")
    assert merged["sample"].is_unique

    merged.to_csv(out_tsv, sep="\t", index=False, na_rep="NA")
    print(
        f"wheat: wrote {out_tsv} shape={merged.shape}, "
        f"unique-samples-in-VCF={merged['sample'].nunique()}/{len(watde_in_vcf)}  "
        f"(VCF total samples={len(vcf_samples)}, non-WATDE excluded)"
    )


def main():
    p = argparse.ArgumentParser(description="Build pheno_clean.tsv per panel.")
    p.add_argument("--panel", choices=["cotton", "rapeseed", "wheat", "all"], default="all")
    args = p.parse_args()

    if args.panel in ("cotton", "all"):
        build_cotton()
    if args.panel in ("rapeseed", "all"):
        build_rapeseed_horvath()
    if args.panel in ("wheat", "all"):
        build_wheat_watkins()


if __name__ == "__main__":
    main()
