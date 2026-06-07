#!/bin/bash
# scripts/phase5d/step6b_rename_chroms_and_rerun.sh
# Collapse segment-coded chroms (1A_0/1A_1) into
# 21 logical chroms (1A) so LOCO truly leaves out one whole chrom (BOLT-LOCO
# semantics), not one of two segments. MVP via bcftools annotate --rename-chrs;
# avoids touching framework code.
#
# After this script, we have:
#   /mnt/nvme/oat_raw/figshare/oat_old.logical_chrom.vcf.gz   (new VCF, 1A/1C/1D names)
#   data/processed/avena_sativa_logical/{A,C,D}/all.*        (new split outputs)
#   results/phase5d/m3_1_loco_v2/oat_old_rahman2025/BIO6/*   (rerun)
#
# Original 42-segment outputs preserved at:
#   /mnt/nvme/oat_raw/figshare/nsgc.gdv.gln...vcf.gz
#   data/processed/avena_sativa/{A,C,D}/all.*
#   results/phase5d/m3_1_loco/oat_old_rahman2025/

set -uo pipefail
export PATH=/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin:$PATH

PROJ=/mnt/7302share/fast_ysp/U7_GWAS
SRC_VCF=/mnt/nvme/oat_raw/figshare/nsgc.gdv.gln.cohort.clean.maf01maxmiss75diploid.beagle_sorted_filt.gt.vcf.gz
DST_VCF=/mnt/nvme/oat_raw/figshare/oat_old.logical_chrom.vcf.gz
LOG=$PROJ/scripts/phase5d/step6b_rename_chroms_and_rerun.log
RENAME_MAP=$PROJ/scripts/phase5d/_rename_chrs_oat.tsv

exec > >(tee -a "$LOG") 2>&1
echo "=== Step 6b started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- 1) Build rename map: panel segment → logical chrom ---
echo "[1/6] building rename map (1A_0/1A_1 → 1A, etc.)..."
python <<PY
rows = []
for i in range(1, 8):
    for s in "ACD":
        for seg in (0, 1):
            rows.append(f"{i}{s}_{seg}\t{i}{s}")
# UN stays UN
rows.append("UN\tUN")
with open("$RENAME_MAP", "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"   {len(rows)} rename entries written to $RENAME_MAP")
PY
echo "   sample:"; head -3 "$RENAME_MAP"; echo "   ..."; tail -3 "$RENAME_MAP"

# --- 2) bcftools annotate --rename-chrs (two-step, avoid BCF integer-id race) ---
# Piping annotate→sort via -Ou BCF fails with "Bad BCF record at X: Invalid
# CONTIG id 1" because BCF carries integer contig ids that don't get rewritten
# when name changes mid-stream. Two-step intermediate VCF.gz is safe.
echo "[2/6] bcftools annotate --rename-chrs (two-step VCF, avoid BCF id race)..."
if [[ -s "$DST_VCF" ]]; then
    echo "   EXISTS skip: $DST_VCF"
else
    INTER_VCF=/tmp/oat_old.renamed.unsorted.vcf.gz
    bcftools annotate --rename-chrs "$RENAME_MAP" --threads 8 \
        -Oz -o "$INTER_VCF" "$SRC_VCF"
    echo "   annotate DONE: $(stat -c%s "$INTER_VCF") B"
    bcftools sort "$INTER_VCF" -Oz -o "$DST_VCF" --max-mem 4G -T /tmp/bcfsort_$$
    echo "   sort DONE: $(stat -c%s "$DST_VCF") B"
    rm -f "$INTER_VCF"
fi

# --- 3) csi index ---
if [[ -s "${DST_VCF}.csi" ]]; then
    echo "[3/6] csi EXISTS skip"
else
    echo "[3/6] csi indexing..."
    bcftools index --csi --threads 8 "$DST_VCF"
fi

# --- 4) Build a new species YAML pointing at logical-chrom VCF ---
LOGICAL_YAML=$PROJ/configs/species/avena_sativa_logical.yaml
echo "[4/6] generating species YAML $LOGICAL_YAML..."
python <<PY
import yaml
from pathlib import Path
src = yaml.safe_load(open("$PROJ/configs/species/avena_sativa.yaml"))
src["id"] = "avena_sativa_logical"
src["common_name"] = src["common_name"] + " — logical-chrom LOCO variant"
# Replace chrom segments with logical chroms
for sg in src["subgenomes"]:
    new_chroms = sorted({c.split("_")[0] for c in sg["chroms"]})
    sg["chroms"] = new_chroms
src["geno"]["vcf"] = "$DST_VCF"
# chrom_map_oat_logical.tsv: panel_chrom = logical_chrom now
src["chrom_map"] = "data/reference/oat/chrom_map_oat_logical.tsv"
src["provenance"]["notes"] = (
    "Logical-chrom variant for Step 6b A/B comparison. VCF segments "
    "(1A_0/1A_1 etc.) collapsed to 21 logical chroms via bcftools annotate "
    "--rename-chrs so LOCO leaves out one whole chrom (BOLT-LOCO semantics), "
    "not one of two segments. Paper Methods footnote will document this. "
    "Underlying SNP positions unchanged."
)
yaml.safe_dump(src, open("$LOGICAL_YAML", "w"), default_flow_style=False, sort_keys=False)
print(f"   wrote $LOGICAL_YAML")
print(f"   subgenomes: { {sg['id']: len(sg['chroms']) for sg in src['subgenomes']} } (expect A/C/D = 7/7/7)")
PY

# --- 4b) Build logical chrom_map (21 logical chroms + UN) ---
LOGICAL_CHROM_MAP=$PROJ/data/reference/oat/chrom_map_oat_logical.tsv
python <<PY
import pandas as pd
rows = []
for i in range(1, 8):
    for s in "ACD":
        rows.append({"panel_chrom": f"{i}{s}", "logical_chrom": f"{i}{s}",
                     "subgenome": s, "fasta_chrom": f"{i}{s}"})
rows.append({"panel_chrom": "UN", "logical_chrom": "UN",
             "subgenome": "UN", "fasta_chrom": ""})
pd.DataFrame(rows).to_csv("$LOGICAL_CHROM_MAP", sep="\t", index=False)
print(f"   wrote $LOGICAL_CHROM_MAP — 22 rows (21 logical chroms + UN)")
PY

# --- 5) homoeogwas split with new YAML ---
echo "[5/6] homoeogwas split (logical-chrom)..."
LOGICAL_OUT=$PROJ/data/processed/avena_sativa_logical
if [[ -d "$LOGICAL_OUT/A" ]]; then
    echo "   EXISTS skip"
else
    homoeogwas split --species-yaml "$LOGICAL_YAML" --threads 8 --keep-vcf
fi

# --- 6) Generate logical fit YAML + rerun BIO6 LOCO ---
LOGICAL_FIT=$PROJ/configs/runs/fit_oat_old_BIO6_loco_logical.yaml
echo "[6/6] generating logical fit YAML + rerun BIO6 LOCO..."
python <<PY
import yaml
src = yaml.safe_load(open("$PROJ/configs/runs/fit_oat_old_BIO6_loco.yaml"))
src["genotype"]["scan_bed_prefix_template"] = "data/processed/avena_sativa_logical/{subgenome}/all"
src["genotype"]["grm"]["bed_prefix_template"] = "data/processed/avena_sativa_logical/{subgenome}/all"
src["outputs"]["out_dir"] = "results/phase5d/m3_1_loco_v2/oat_old_rahman2025"
yaml.safe_dump(src, open("$LOGICAL_FIT", "w"), default_flow_style=False, sort_keys=False)
print(f"   wrote $LOGICAL_FIT")
PY

mkdir -p $PROJ/results/phase5d/m3_1_loco_v2/oat_old_rahman2025
cd $PROJ
homoeogwas fit --config "$LOGICAL_FIT" --backend cpu

echo "=== Step 6b FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- 7) A/B compare: λ_GC and top hits ---
python <<PY
import pandas as pd
import json

a_dir = "$PROJ/results/phase5d/m3_1_loco/oat_old_rahman2025"
b_dir = "$PROJ/results/phase5d/m3_1_loco_v2/oat_old_rahman2025"
a_sum = json.load(open(f"{a_dir}/summary_BIO6.json"))
b_sum = json.load(open(f"{b_dir}/summary_BIO6.json"))
print("\n=== A/B comparison ===")
print(f"  A (42-segment LOCO):  λ_GC = {a_sum['lambda_gc']:.4f}, n_sig 5e-8 = {a_sum.get('n_sig_5e8', '?')}, runtime = {a_sum.get('runtime_s', '?'):.1f}s")
print(f"  B (21-logical LOCO):  λ_GC = {b_sum['lambda_gc']:.4f}, n_sig 5e-8 = {b_sum.get('n_sig_5e8', '?')}, runtime = {b_sum.get('runtime_s', '?'):.1f}s")
print(f"  Δλ_GC = {b_sum['lambda_gc'] - a_sum['lambda_gc']:+.4f}")

a_ss = pd.read_csv(f"{a_dir}/sumstats_BIO6_loco.tsv", sep="\t").nsmallest(10, "p")
b_ss = pd.read_csv(f"{b_dir}/sumstats_BIO6_loco.tsv", sep="\t").nsmallest(10, "p")
print("\n  A top 5:")
for _, r in a_ss.head(5).iterrows():
    print(f"    {r['chrom']}:{int(r['pos'])} p={r['p']:.2e}")
print("  B top 5:")
for _, r in b_ss.head(5).iterrows():
    print(f"    {r['chrom']}:{int(r['pos'])} p={r['p']:.2e}")
PY
