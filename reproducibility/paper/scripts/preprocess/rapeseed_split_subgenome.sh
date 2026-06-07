#!/usr/bin/env bash
# rapeseed_split_subgenome.sh
# One script for both panels:
#   - Horvath2020 (CHROM = chrA01..chrA10 / chrC01..chrC09, *_random dropped by default)
#   - Hu2022      (CHROM = NC_027757.2..NC_027775.2; NC_757..766 -> A, NC_767..775 -> C)
# Output:
#   data/processed/rapeseed/horvath/{A,C}/all.{vcf.gz,pgen,pvar,psam}
#   data/processed/rapeseed/hu2022/{A,C}/all.{vcf.gz,pgen,pvar,psam}
#
# Usage:
#   bash scripts/preprocess/rapeseed_split_subgenome.sh                  # run both panels
#   PANEL=horvath bash scripts/preprocess/rapeseed_split_subgenome.sh   # only Horvath2020
#   PANEL=hu2022  bash scripts/preprocess/rapeseed_split_subgenome.sh
#   KEEP_RANDOM=1 PANEL=horvath ...                                      # keep *_random as unanchored

set -euo pipefail

ROOT=${ROOT:-/mnt/7302share/fast_ysp/U7_GWAS}
OUT="$ROOT/data/processed/rapeseed"
LOG="$ROOT/logs/preprocess"
THREADS=${THREADS:-8}
MAF=${MAF:-0.01}
GENO=${GENO:-0.1}
HWE=${HWE:-1e-6}
PANEL=${PANEL:-both}
KEEP_RANDOM=${KEEP_RANDOM:-0}

mkdir -p "$OUT" "$LOG"

# $1 input vcf; $2 panel name; $3 sub (A/C); $4...$N regions
split_one() {
  local IN_VCF=$1; local NAME=$2; local SUB=$3; shift 3
  local DST="$OUT/$NAME/$SUB"
  mkdir -p "$DST"
  local REGIONS
  REGIONS=$(IFS=','; echo "$*")

  echo "[$(date)] ===== rapeseed $NAME / $SUB (KEEP_RANDOM=$KEEP_RANDOM) =====" | tee -a "$LOG/rapeseed.log"
  # The raw.vcf.gz cache key must include a hash of the region list, otherwise toggling
  # KEEP_RANDOM would silently reuse the old file
  local REGIONS_HASH
  REGIONS_HASH=$(echo -n "$REGIONS" | md5sum | cut -c1-8)
  local SUBSET="$DST/raw.${REGIONS_HASH}.vcf.gz"
  local SUBSET_LINK="$DST/raw.vcf.gz"
  if [ ! -s "$SUBSET" ]; then
    echo "[$(date)] bcftools view -r $REGIONS  -> raw.${REGIONS_HASH}.vcf.gz" | tee -a "$LOG/rapeseed.log"
    bcftools view --threads "$THREADS" -r "$REGIONS" -Oz -o "$SUBSET" "$IN_VCF"
    tabix -p vcf "$SUBSET"
  fi
  # Update the raw.vcf.gz symlink -> current region set
  ln -sf "$(basename "$SUBSET")" "$SUBSET_LINK"
  ln -sf "$(basename "$SUBSET").tbi" "$SUBSET_LINK.tbi"

  plink2 --vcf "$SUBSET" \
    --maf "$MAF" --geno "$GENO" --hwe "$HWE" \
    --max-alleles 2 --snps-only \
    --threads "$THREADS" \
    --make-pgen --out "$DST/all" 2>&1 | tee -a "$LOG/rapeseed.log"

  plink2 --pfile "$DST/all" \
    --export vcf-4.2 bgz id-paste=iid \
    --threads "$THREADS" \
    --out "$DST/all" 2>&1 | tee -a "$LOG/rapeseed.log"
  tabix -f -p vcf "$DST/all.vcf.gz"

  local N=$(bcftools view -H "$DST/all.vcf.gz" | wc -l)
  echo "[$(date)] rapeseed $NAME/$SUB done. final SNPs = $N" | tee -a "$LOG/rapeseed.log"
}

run_horvath() {
  local IN="$ROOT/data/raw/rapeseed/Horvath2020_Zenodo4302088/imputed.vcf.gz"
  [ -f "$IN" ] || { echo "MISSING $IN" >&2; exit 1; }
  [ -f "$IN.tbi" ] || tabix -p vcf "$IN"

  local A_CHR=(chrA01 chrA02 chrA03 chrA04 chrA05 chrA06 chrA07 chrA08 chrA09 chrA10)
  local C_CHR=(chrC01 chrC02 chrC03 chrC04 chrC05 chrC06 chrC07 chrC08 chrC09)
  if [ "$KEEP_RANDOM" = "1" ]; then
    A_CHR+=(chrA01_random chrA02_random chrA03_random chrA04_random chrA05_random \
            chrA06_random chrA07_random chrA08_random chrA09_random chrA10_random chrAnn_random)
    C_CHR+=(chrC01_random chrC02_random chrC03_random chrC04_random chrC05_random \
            chrC06_random chrC07_random chrC08_random chrC09_random chrCnn_random)
    # chrUnn_random is of unknown nature, not assigned to any subgenome for now
  fi
  split_one "$IN" horvath A "${A_CHR[@]}"
  split_one "$IN" horvath C "${C_CHR[@]}"
}

run_hu2022() {
  local IN="$ROOT/data/raw/rapeseed/vcf/all_snp.vcf.gz"
  [ -f "$IN" ] || { echo "MISSING $IN" >&2; exit 1; }
  [ -f "$IN.tbi" ] || tabix -p vcf "$IN"

  # A subgenome (Bra_napus_v2.0 NCBI naming): chrA01..A10
  local A_CHR=(NC_027757.2 NC_027758.2 NC_027759.2 NC_027760.2 NC_027761.2 \
               NC_027762.2 NC_027763.2 NC_027764.2 NC_027765.2 NC_027766.2)
  # C subgenome: chrC01..C09
  local C_CHR=(NC_027767.2 NC_027768.2 NC_027769.2 NC_027770.2 NC_027771.2 \
               NC_027772.2 NC_027773.2 NC_027774.2 NC_027775.2)
  split_one "$IN" hu2022 A "${A_CHR[@]}"
  split_one "$IN" hu2022 C "${C_CHR[@]}"
}

case "$PANEL" in
  both)    run_horvath; run_hu2022 ;;
  horvath) run_horvath ;;
  hu2022)  run_hu2022 ;;
  *) echo "unknown PANEL=$PANEL (use both|horvath|hu2022)" >&2; exit 2 ;;
esac

echo "rapeseed split DONE (panel=$PANEL)."
