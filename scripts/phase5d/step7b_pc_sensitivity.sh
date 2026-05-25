#!/bin/bash
# scripts/phase5d/step7b_pc_sensitivity.sh
# Codex review #2 Q1 NEXT_STEP — PC1-5 sensitivity for 3 traits.
# We don't yet have covariates support in cli.py / lmm.py, so we residualize
# the trait by PC1-5 OLS first then re-run fit on the residualized phenotype.
# This is equivalent to including PC1-5 as fixed effects in the LMM
# (REML invariance under linear projections of fixed effects).

set -uo pipefail
export PATH=/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin:$PATH

PROJ=/mnt/7302share/fast_ysp/U7_GWAS
LOG=$PROJ/scripts/phase5d/step7b_pc_sensitivity.log
exec > >(tee -a "$LOG") 2>&1
echo "=== Step 7b PC sensitivity started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- 1) Merge per-sub bed files into one bed for PCA (plink2 --pca needs merged) ---
MERGED_PREFIX=$PROJ/data/processed/avena_sativa_logical/merged
if [[ ! -s "${MERGED_PREFIX}.bed" ]]; then
    echo "[1/5] plink2 import logical-chrom VCF → merged bed..."
    # plink2 v2.0.0-a.6.9 does not yet support non-concatenating --pmerge-list
    # (cross-sample-set merge); shortcut by importing the full logical-chrom
    # VCF directly. Variant IDs are forced to chrom:pos via --set-all-var-ids
    # so cross-sub duplicate rsid '7A:136473558' (from per-sub bims) is moot.
    cd $PROJ
    plink2 --vcf /mnt/nvme/oat_raw/figshare/oat_old.logical_chrom.vcf.gz \
           --set-all-var-ids '@:#:\$r:\$a' --new-id-max-allele-len 200 missing \
           --max-alleles 2 --snps-only \
           --maf 0.01 --geno 0.1 \
           --make-bed --out "$MERGED_PREFIX" \
           --memory 16000 --threads 8
    echo "[1/5] merged DONE: $(wc -l < ${MERGED_PREFIX}.bim) SNPs × $(wc -l < ${MERGED_PREFIX}.fam) samples"
else
    echo "[1/5] merged EXISTS skip"
fi

# --- 2) PCA on merged bed ---
PCA_PREFIX=$PROJ/data/processed/avena_sativa_logical/pca
if [[ ! -s "${PCA_PREFIX}.eigenvec" ]]; then
    echo "[2/5] plink2 --pca on merged bed..."
    plink2 --bfile "$MERGED_PREFIX" --pca 5 --out "$PCA_PREFIX" --memory 16000 --threads 8
    echo "[2/5] PCA DONE"
else
    echo "[2/5] PCA EXISTS skip"
fi
echo "PCA eigenvec head:"
head -3 "${PCA_PREFIX}.eigenvec"

# --- 3) Residualize 3 traits by PC1-5 ---
echo "[3/5] residualize 3 traits by PC1-5 (OLS)..."
python <<PY
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression

pheno = pd.read_csv("$PROJ/data/processed/oat/pheno_clean.tsv", sep="\t")
pca = pd.read_csv("${PCA_PREFIX}.eigenvec", sep="\t")
# plink2 PC file: IID, PC1, ..., PC5
print(f"   pheno {pheno.shape}, pca {pca.shape}")
print(f"   pca cols: {list(pca.columns)}")
m = pheno.merge(pca, left_on="sample_id", right_on="IID", how="inner")
print(f"   merged: {len(m)} samples (expect 737)")
pc_cols = [c for c in pca.columns if c.startswith("PC")]
print(f"   PCs: {pc_cols}")
X = m[pc_cols].values
out_rows = m[["sample_id"]].copy()
for trait in ("BIO6", "SOC", "BIO12"):
    y = m[trait].values
    mask = ~np.isnan(y)
    lr = LinearRegression().fit(X[mask], y[mask])
    resid = y - lr.predict(X)
    out_rows[trait + "_pcadj"] = resid
out_rows.to_csv("$PROJ/data/processed/oat/pheno_pcadj.tsv", sep="\t", index=False)
print(f"   wrote $PROJ/data/processed/oat/pheno_pcadj.tsv with cols {list(out_rows.columns)}")
PY

# --- 4) Generate 3 _pcadj fit YAMLs ---
echo "[4/5] generate _pcadj fit YAMLs..."
python <<PY
import yaml
for trait in ("BIO6", "SOC", "BIO12"):
    src = "$PROJ/configs/runs/fit_oat_old_" + trait + "_loco_logical.yaml"
    cfg = yaml.safe_load(open(src))
    cfg["phenotype"]["path"] = "data/processed/oat/pheno_pcadj.tsv"
    cfg["phenotype"]["trait"] = trait + "_pcadj"
    cfg["outputs"]["out_dir"] = "results/phase5d/m3_1_loco_v2_pcadj/oat_old_rahman2025/" + trait
    cfg["outputs"]["prefix"] = trait + "_pcadj"
    dst = "$PROJ/configs/runs/fit_oat_old_" + trait + "_loco_logical_pcadj.yaml"
    yaml.safe_dump(cfg, open(dst, "w"), default_flow_style=False, sort_keys=False)
    print(f"   wrote {dst}")
PY

# --- 5) Sequential fit ---
echo "[5/5] sequential fit (3 trait pcadj)..."
cd $PROJ
for trait in BIO6 SOC BIO12; do
    echo "=== fit ${trait}_pcadj $(date -u +%H:%M:%S) ==="
    homoeogwas fit --config configs/runs/fit_oat_old_${trait}_loco_logical_pcadj.yaml --backend cpu
done

echo "=== Step 7b PC sensitivity FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- 6) A/B compare ---
python <<PY
import pandas as pd, json
base = "$PROJ/results/phase5d"
qtl = pd.read_csv("$PROJ/data/reference/oat/known_qtl_oat.tsv", sep="\t")

print()
print(f"{'trait':<8} {'λ_GC raw':<10} {'λ_GC pcadj':<12} {'sig5e8 raw':<12} {'sig5e8 pcadj':<14} {'top raw':<26} {'top pcadj':<26}")
for trait in ("BIO6","SOC","BIO12"):
    raw_d = f"{base}/m3_1_loco_v2/oat_old_rahman2025/{trait}"
    adj_d = f"{base}/m3_1_loco_v2_pcadj/oat_old_rahman2025/{trait}"
    raw_s = json.load(open(f"{raw_d}/summary_{trait}.json"))
    adj_s = json.load(open(f"{adj_d}/summary_{trait}_pcadj.json"))
    raw_lam = raw_s.get("lambda_gc"); raw_lam = raw_lam.get("overall", float("nan")) if isinstance(raw_lam, dict) else raw_lam
    adj_lam = adj_s.get("lambda_gc"); adj_lam = adj_lam.get("overall", float("nan")) if isinstance(adj_lam, dict) else adj_lam
    raw_ss = pd.read_csv(f"{raw_d}/sumstats_{trait}_loco.tsv", sep="\t")
    adj_ss = pd.read_csv(f"{adj_d}/sumstats_{trait}_pcadj_loco.tsv", sep="\t")
    raw_sig8 = (raw_ss["p"]<5e-8).sum(); adj_sig8 = (adj_ss["p"]<5e-8).sum()
    raw_top = raw_ss.nsmallest(1,"p").iloc[0]; adj_top = adj_ss.nsmallest(1,"p").iloc[0]
    raw_s2 = f"{raw_top['chrom']}:{int(raw_top['pos'])} p={raw_top['p']:.1e}"
    adj_s2 = f"{adj_top['chrom']}:{int(adj_top['pos'])} p={adj_top['p']:.1e}"
    print(f"{trait:<8} {raw_lam:<10.4f} {adj_lam:<12.4f} {raw_sig8:<12} {adj_sig8:<14} {raw_s2:<26} {adj_s2:<26}")
PY
