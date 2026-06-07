#!/usr/bin/env bash
# wheat_split_subgenome.sh
# Merge + QC the 21 IWGSC v1.0 chromosomes (chr1A..chr7D) into the three A/B/D subgenomes.
# Output: data/processed/wheat/{A,B,D}/all.{vcf.gz,pgen,pvar,psam}
#
# Usage:
#   bash scripts/preprocess/wheat_split_subgenome.sh
# Optional env vars:
#   THREADS=16  MAF=0.01  GENO=0.1  HWE=1e-6  ROOT=/mnt/7302share/fast_ysp/U7_GWAS

set -euo pipefail

ROOT=${ROOT:-/mnt/7302share/fast_ysp/U7_GWAS}
RAW="$ROOT/data/raw/wheat/vcf"
OUT="$ROOT/data/processed/wheat"
LOG="$ROOT/logs/preprocess"
THREADS=${THREADS:-16}
MAF=${MAF:-0.01}
GENO=${GENO:-0.1}
# wheat Watkins is an inbred landrace + cultivar collection; a plain HWE test rejects ~95% of SNPs by default.
# HWE is disabled by default (HWE=""); override explicitly, e.g. HWE=1e-50.
HWE=${HWE:-}

mkdir -p "$OUT" "$LOG"

SUFFIX='.SNP.Missing-unphasing.ID.ann.finalSID.allele2_retain.hard_retain.InbreedingCoeff_retain.missing_retain.maf_retain.vcf.gz'

run_subgenome() {
  local SUB=$1
  local DST="$OUT/$SUB"
  local TMP="$DST/tmp"
  mkdir -p "$DST" "$TMP"

  echo "[$(date)] ===== wheat subgenome $SUB =====" | tee -a "$LOG/wheat.log"

  local FILES=()
  for i in 1 2 3 4 5 6 7; do
    local F="$RAW/chr${i}${SUB}${SUFFIX}"
    [ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }
    # wheat chromosomes > 512Mb exceed the tbi limit, use csi
    [ -f "$F.tbi" ] || [ -f "$F.csi" ] || tabix --csi -p vcf "$F"
    FILES+=("$F")
  done

  local MERGED="$TMP/merged.vcf.gz"
  if [ ! -s "$MERGED" ]; then
    echo "[$(date)] bcftools concat -> $MERGED" | tee -a "$LOG/wheat.log"
    bcftools concat --threads "$THREADS" -Oz -o "$MERGED" "${FILES[@]}"
    tabix --csi -p vcf "$MERGED"
  else
    echo "[$(date)] reuse existing $MERGED" | tee -a "$LOG/wheat.log"
  fi

  local HWE_FLAG=""
  if [ -n "$HWE" ]; then HWE_FLAG="--hwe $HWE"; fi
  echo "[$(date)] plink2 filter+import (MAF>=$MAF, geno<=$GENO, HWE=${HWE:-disabled}, bi-allelic SNP)" | tee -a "$LOG/wheat.log"
  plink2 --vcf "$MERGED" \
    --maf "$MAF" --geno "$GENO" $HWE_FLAG \
    --max-alleles 2 --snps-only \
    --threads "$THREADS" \
    --make-pgen --out "$DST/all" 2>&1 | tee -a "$LOG/wheat.log"

  echo "[$(date)] plink2 export filtered VCF" | tee -a "$LOG/wheat.log"
  plink2 --pfile "$DST/all" \
    --export vcf-4.2 bgz id-paste=iid \
    --threads "$THREADS" \
    --out "$DST/all" 2>&1 | tee -a "$LOG/wheat.log"
  tabix -f --csi -p vcf "$DST/all.vcf.gz"

  local N=$(bcftools view -H "$DST/all.vcf.gz" | wc -l)
  echo "[$(date)] wheat $SUB done. final SNPs = $N" | tee -a "$LOG/wheat.log"
  # Cleanup: keep merged.vcf.gz by default for debugging; on the order of 100GB, manually rm -rf $TMP once verified OK
}

for SUB in A B D; do
  run_subgenome "$SUB"
done

echo "wheat split DONE."
