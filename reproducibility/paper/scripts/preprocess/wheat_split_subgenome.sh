#!/usr/bin/env bash
# wheat_split_subgenome.sh
# 把 21 个 IWGSC v1.0 染色体 (chr1A..chr7D) 按 A/B/D 三亚基因组分别合并 + QC。
# 输出: data/processed/wheat/{A,B,D}/all.{vcf.gz,pgen,pvar,psam}
#
# 用法:
#   bash scripts/preprocess/wheat_split_subgenome.sh
# 可选 env vars:
#   THREADS=16  MAF=0.01  GENO=0.1  HWE=1e-6  ROOT=/mnt/7302share/fast_ysp/U7_GWAS

set -euo pipefail

ROOT=${ROOT:-/mnt/7302share/fast_ysp/U7_GWAS}
RAW="$ROOT/data/raw/wheat/vcf"
OUT="$ROOT/data/processed/wheat"
LOG="$ROOT/logs/preprocess"
THREADS=${THREADS:-16}
MAF=${MAF:-0.01}
GENO=${GENO:-0.1}
# wheat Watkins 是 inbred landrace + cultivar 集合, 普通 HWE 测试默认 reject ~95% SNP.
# 默认禁用 HWE (HWE=""); 可显式覆盖 e.g. HWE=1e-50.
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
    # wheat 染色体 > 512Mb 超 tbi 限制, 用 csi
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
  # 释放: 默认保留 merged.vcf.gz 方便排错; 大约 100GB 量级, 验证 OK 后手动 rm -rf $TMP
}

for SUB in A B D; do
  run_subgenome "$SUB"
done

echo "wheat split DONE."
