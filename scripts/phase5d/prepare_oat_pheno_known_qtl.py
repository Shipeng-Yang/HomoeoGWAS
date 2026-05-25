"""Parse Rahman 2025 OLD panel File_S1 (Table C = 36 environmental phenotypes)
and File_S3 (GWAS results + candidate gene) into project-standard TSV.

Phase 5d Step 2-3 dependency. Run after Step 1 downloads finish.

Outputs:
  data/processed/oat/pheno_clean.tsv         — sample_id + 36 trait cols (739 rows)
  data/reference/oat/known_qtl_oat.tsv       — anchor SNPs for M3.2 self-replication gate
  data/reference/oat/candidate_genes_oat.tsv — 192 candidate genes (Arabidopsis homolog)
  data/reference/oat/chrom_map_oat.tsv       — panel_chrom (1A_0/1A_1) → fasta_chrom (TBD after ref unzip) → subgenome (A/C/D) → logical_chrom (1A) for LOCO grouping
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import sys

BASE = Path("/mnt/nvme/oat_raw/figshare")
OUT_PHENO = Path("/mnt/7302share/fast_ysp/U7_GWAS/data/processed/oat/pheno_clean.tsv")
OUT_QTL = Path("/mnt/7302share/fast_ysp/U7_GWAS/data/reference/oat/known_qtl_oat.tsv")
OUT_CAND = Path("/mnt/7302share/fast_ysp/U7_GWAS/data/reference/oat/candidate_genes_oat.tsv")
OUT_CHROMMAP = Path("/mnt/7302share/fast_ysp/U7_GWAS/data/reference/oat/chrom_map_oat.tsv")

# ---------------------------------------------------------------------------
# 1) Pheno from File_S1 Table C
# ---------------------------------------------------------------------------
s1 = pd.ExcelFile(BASE / "File_S1.xlsx")
# Table C: row 0 is the section title, row 1 is the column header
df = s1.parse("Table C", header=None)
header_row_idx = 1
header = df.iloc[header_row_idx].tolist()
data = df.iloc[header_row_idx + 1:].reset_index(drop=True)
data.columns = header
data = data.rename(columns={"Individual": "sample_id"})
pheno_cols = [c for c in data.columns if c != "sample_id"]
print(f"[pheno] {len(data)} samples × {len(pheno_cols)} traits")
print(f"[pheno] traits: {pheno_cols}")
# Coerce numerics
for c in pheno_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
OUT_PHENO.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUT_PHENO, sep="\t", index=False)
print(f"[pheno] wrote {OUT_PHENO}")

# ---------------------------------------------------------------------------
# 2) Known QTL from File_S3 GWAS_results
# ---------------------------------------------------------------------------
s3 = pd.ExcelFile(BASE / "File_S3.xlsx")
gwas_raw = s3.parse("GWAS_results", header=None)
g_header = gwas_raw.iloc[1].tolist()
g_data = gwas_raw.iloc[2:].reset_index(drop=True)
g_data.columns = g_header
g_data = g_data.rename(columns={
    "Trait": "trait", "Chr": "logical_chrom", "SNP": "snp_id",
    "Position": "pos", "Minor allele": "minor_allele",
    "Major allele": "major_allele", "Allele frequency": "maf",
    "Effect": "effect", "p": "p_value", "-log10(p)": "neg_log10_p",
})
for c in ("pos", "maf", "effect", "p_value", "neg_log10_p"):
    g_data[c] = pd.to_numeric(g_data[c], errors="coerce")
# subgenome from logical_chrom (3D → D, 7A → A, 5C → C)
g_data["subgenome"] = g_data["logical_chrom"].astype(str).str.extract(r"(\d)([ACD])")[1]
g_data["qtl_name"] = g_data["trait"].astype(str) + "_" + g_data["snp_id"].astype(str)
g_data = g_data[[
    "qtl_name", "trait", "logical_chrom", "subgenome", "snp_id", "pos",
    "minor_allele", "major_allele", "maf", "effect", "p_value", "neg_log10_p",
]]
OUT_QTL.parent.mkdir(parents=True, exist_ok=True)
g_data.to_csv(OUT_QTL, sep="\t", index=False)
print(f"[qtl] {len(g_data)} known QTL sentinels → {OUT_QTL}")
print(f"[qtl] traits represented: {sorted(g_data['trait'].dropna().unique())}")

# ---------------------------------------------------------------------------
# 3) Candidate genes from File_S3 Candidate gene
# ---------------------------------------------------------------------------
cand_raw = s3.parse("Candidate gene", header=None)
c_header = cand_raw.iloc[1].tolist()
c_data = cand_raw.iloc[2:].reset_index(drop=True)
c_data.columns = c_header
# Normalize column names
c_data = c_data.rename(columns={
    "chr": "logical_chrom", "rs": "snp_id", "ps": "pos", "Trait": "trait",
    "Gene name": "gene_name", "Gene position": "gene_position",
    "Length": "gene_length", "Gene distance (kb)": "gene_distance_kb",
    "Description": "description", "GO biological": "go_biological",
    "Arabidopsis homolog": "arabidopsis_homolog", "E-value": "e_value",
    "Additonal function": "additional_function",
})
for c in ("pos", "gene_distance_kb", "e_value"):
    c_data[c] = pd.to_numeric(c_data[c], errors="coerce")
c_data["subgenome"] = c_data["logical_chrom"].astype(str).str.extract(r"(\d)([ACD])")[1]
OUT_CAND.parent.mkdir(parents=True, exist_ok=True)
c_data.to_csv(OUT_CAND, sep="\t", index=False)
print(f"[cand] {len(c_data)} candidate genes → {OUT_CAND}")

# ---------------------------------------------------------------------------
# 4) chrom_map: panel_chrom (1A_0/1A_1) → logical_chrom (1A) → subgenome (A/C/D)
#    fasta_chrom column LEFT BLANK — fill after OT3098 v2 unzip + fasta header inspect
# ---------------------------------------------------------------------------
panel_chroms = [f"{i}{s}_{seg}" for i in range(1, 8) for s in "ACD" for seg in (0, 1)]
panel_chroms.append("UN")
rows = []
for pc in panel_chroms:
    if pc == "UN":
        rows.append({
            "panel_chrom": "UN", "logical_chrom": "UN",
            "subgenome": "UN", "fasta_chrom": "",  # TBD
            "segment": "UN",
        })
    else:
        # 1A_0 → logical 1A, sub A, seg 0
        lc = pc.split("_")[0]
        sub = lc[-1]
        seg = pc.split("_")[1]
        rows.append({
            "panel_chrom": pc, "logical_chrom": lc, "subgenome": sub,
            "fasta_chrom": "",  # TBD after fasta header inspect
            "segment": seg,
        })
cm = pd.DataFrame(rows)
OUT_CHROMMAP.parent.mkdir(parents=True, exist_ok=True)
cm.to_csv(OUT_CHROMMAP, sep="\t", index=False)
print(f"[chrom_map] {len(cm)} entries (42 segments + UN) → {OUT_CHROMMAP}")
print(f"[chrom_map] NOTE: fasta_chrom column is BLANK — fill after OT3098 v2 unzip")
print()
print("===== DONE =====")
