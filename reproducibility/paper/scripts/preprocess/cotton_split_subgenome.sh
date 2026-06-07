#!/usr/bin/env bash
# cotton_split_subgenome.sh
# The hebau panel raw data is CSV (one file per chromosome, A01..A13, D01..D13).
# Flow: unzip -> CSV->VCF (per chr) -> bcftools concat (A/D) -> plink2 QC -> output pgen + filtered vcf.
# Output: data/processed/cotton/{A,D}/all.{vcf.gz,pgen,pvar,psam}
#
# Usage:
#   bash scripts/preprocess/cotton_split_subgenome.sh
# Optional env: THREADS=8 MAF=0.01 GENO=0.1 HWE=1e-6 ROOT=...

set -euo pipefail

ROOT=${ROOT:-/mnt/7302share/fast_ysp/U7_GWAS}
SRC_ZIP="$ROOT/data/raw/cotton/snp/genotype-data.zip"
OUT="$ROOT/data/processed/cotton"
LOG="$ROOT/logs/preprocess"
HELPER="$ROOT/scripts/preprocess/_cotton_csv_to_vcf.py"
THREADS=${THREADS:-8}
MAF=${MAF:-0.01}
GENO=${GENO:-0.1}
HWE=${HWE:-1e-6}

mkdir -p "$OUT" "$LOG"
STAGE="$OUT/stage"
mkdir -p "$STAGE"

# 1) unzip CSV (~280MB zip -> ~3GB csv)
if [ ! -f "$STAGE/genotype-data/chr_A01.csv.gz" ]; then
  echo "[$(date)] unzip $SRC_ZIP" | tee -a "$LOG/cotton.log"
  unzip -o "$SRC_ZIP" -d "$STAGE" 2>&1 | tail -5 | tee -a "$LOG/cotton.log"
fi

# 2) CSV -> VCF per chromosome
ALL_CHROMS=(A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 A11 A12 A13 \
            D01 D02 D03 D04 D05 D06 D07 D08 D09 D10 D11 D12 D13)
for CHR in "${ALL_CHROMS[@]}"; do
  CSV="$STAGE/genotype-data/chr_${CHR}.csv.gz"
  VCF="$STAGE/chr_${CHR}.vcf.gz"
  [ -f "$CSV" ] || { echo "MISSING $CSV" >&2; exit 1; }
  if [ ! -s "$VCF" ]; then
    echo "[$(date)] csv->vcf $CHR" | tee -a "$LOG/cotton.log"
    python "$HELPER" "$CSV" "$VCF" "$CHR"
    tabix -f -p vcf "$VCF"
  fi
done

# 3) concat + QC per subgenome
run_subgenome() {
  local SUB=$1
  local DST="$OUT/$SUB"
  mkdir -p "$DST"

  echo "[$(date)] ===== cotton subgenome $SUB =====" | tee -a "$LOG/cotton.log"
  local FILES=()
  for n in 01 02 03 04 05 06 07 08 09 10 11 12 13; do
    FILES+=("$STAGE/chr_${SUB}${n}.vcf.gz")
  done

  local MERGED="$DST/merged.vcf.gz"
  if [ ! -s "$MERGED" ]; then
    bcftools concat --threads "$THREADS" -Oz -o "$MERGED" "${FILES[@]}"
    tabix -p vcf "$MERGED"
  fi

  plink2 --vcf "$MERGED" \
    --maf "$MAF" --geno "$GENO" --hwe "$HWE" \
    --max-alleles 2 --snps-only \
    --threads "$THREADS" \
    --make-pgen --out "$DST/all" 2>&1 | tee -a "$LOG/cotton.log"

  plink2 --pfile "$DST/all" \
    --export vcf-4.2 bgz id-paste=iid \
    --threads "$THREADS" \
    --out "$DST/all" 2>&1 | tee -a "$LOG/cotton.log"
  tabix -f -p vcf "$DST/all.vcf.gz"

  local N=$(bcftools view -H "$DST/all.vcf.gz" | wc -l)
  echo "[$(date)] cotton $SUB done. final SNPs = $N" | tee -a "$LOG/cotton.log"
}

for SUB in A D; do
  run_subgenome "$SUB"
done

echo "cotton split DONE."
