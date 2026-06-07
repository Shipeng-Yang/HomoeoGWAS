#!/bin/bash
# regenie step 2: single-variant assoc + gene-burden (SKATO/ACATO) for one cotton subgenome.
# Estimands: single-marker marginal GWAS + single-gene region burden.
set -uo pipefail
sub=$1
ROOT=/mnt/7302share/fast_ysp/U7_GWAS
B=~/.local/share/mamba/envs/baseline_ext/bin
EB=$ROOT/results/phase7/ext_baseline
cd $EB/regenie/$sub

# --- single-variant association (whole-subgenome -> proper genome-wide threshold) ---
$B/regenie --step 2 --bed $ROOT/data/processed/cotton/${sub}_int/all \
  --phenoFile $EB/pheno_regenie.tsv --phenoColList fiber_length_BLUE,length_uniformity_BLUE \
  --qt --pred cotton_${sub}_step1_pred.list --bsize 400 \
  --out ${sub}_assoc >/tmp/regenie_${sub}_assoc.log 2>&1
echo "[$sub] single-variant done: $(ls ${sub}_assoc_*.regenie 2>/dev/null | wc -l) files"

# --- gene-burden (single-gene aggregation; common-variant inclusive aaf<=0.5) ---
$B/regenie --step 2 --bed $ROOT/data/processed/cotton/${sub}_int/all \
  --phenoFile $EB/pheno_regenie.tsv --phenoColList fiber_length_BLUE,length_uniformity_BLUE \
  --qt --pred cotton_${sub}_step1_pred.list --bsize 400 \
  --anno-file $EB/regenie/${sub}_anno.txt --set-list $EB/regenie/${sub}_setlist.txt \
  --mask-def $EB/regenie/${sub}_mask.txt --aaf-bins 0.5 --build-mask sum --vc-tests skato,acato \
  --out ${sub}_burden >/tmp/regenie_${sub}_burden.log 2>&1
echo "[$sub] burden done: $(ls ${sub}_burden_*.regenie 2>/dev/null | wc -l) files"
grep -iE "error" /tmp/regenie_${sub}_assoc.log /tmp/regenie_${sub}_burden.log | head -3
