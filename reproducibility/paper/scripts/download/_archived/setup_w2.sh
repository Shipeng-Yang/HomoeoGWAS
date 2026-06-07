#!/usr/bin/env bash
# =============================================================================
# U7_GWAS — W2 setup script: large GWAS panel downloads
# Target: Synology NAS at /volume2/U7_GWAS
# Expected total: ~431 GB
#   - Cotton G. hirsutum 1,081 SNPs + phenotypes  (~288 MB, minutes)
#   - B. napus 2,311-accession merged VCF on BnIR (~770 MB, minutes)
#     [contains the Wu 2019 991 accessions as a subset; subset by sample ID later]
#   - Watkins wheat 1,051-accession VCFs (ENA mirror)  (~430 GB, hours-days)
# Resumable: aria2c -c on every file; safe to Ctrl-C and re-run.
# Tmux/screen recommended for the wheat phase.
# Usage:
#   bash setup_w2.sh           # default = all
#   bash setup_w2.sh cotton    # cotton only
#   bash setup_w2.sh napus     # B. napus only
#   bash setup_w2.sh wheat     # Watkins only
# URLs verified 2026-05-13.
# =============================================================================
set -euo pipefail
export LC_ALL=C

ROOT="/volume2/U7_GWAS"
# -j 1 = at most 1 file at a time per aria2c invocation (kind to ENA / BnIR).
# Each file still uses 16 connections for chunked download.
ARIA_OPTS=(-x 16 -s 16 -j 1 -c
           --auto-file-renaming=false
           --allow-overwrite=false
           --console-log-level=warn
           --summary-interval=60
           --retry-wait=10
           --max-tries=5)

WHAT="${1:-all}"  # all | wheat | napus | cotton

# ---------- Tool checks ----------
command -v aria2c >/dev/null || { echo "ERROR: aria2c not installed"; exit 1; }
command -v wget   >/dev/null || { echo "ERROR: wget not installed"; exit 1; }
command -v curl   >/dev/null || { echo "ERROR: curl not installed"; exit 1; }

# ---------- Disk check (portable: works on Synology busybox + GNU) ----------
FREE_GB=$(df -k "$ROOT" 2>/dev/null | tail -1 | awk '{printf "%d", $4/1024/1024}')
NEED_GB=500
if [ "${FREE_GB:-0}" -lt "$NEED_GB" ]; then
  echo "WARNING: $ROOT has ${FREE_GB:-unknown} GB free; Watkins alone needs ~430 GB."
  echo "         Recommend ≥${NEED_GB} GB free before running 'wheat' or 'all'."
  if [ "$WHAT" = "wheat" ] || [ "$WHAT" = "all" ]; then
    printf "Continue anyway? [y/N] "
    read -r confirm
    [ "$confirm" = "y" ] || { echo "Aborted."; exit 1; }
  fi
fi

# ---------- Directory layout ----------
mkdir -p "$ROOT"/data/raw/wheat/{vcf,phenotype}
mkdir -p "$ROOT"/data/raw/rapeseed/{vcf,phenotype}
mkdir -p "$ROOT"/data/raw/cotton/{snp,phenotype}
mkdir -p "$ROOT"/logs

LOG="$ROOT/logs/setup_w2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " U7_GWAS W2 setup — $(date)"
echo " mode = $WHAT, ROOT = $ROOT"
echo " LOG  = $LOG"
echo " Free = ${FREE_GB} GB"
echo "============================================================"

# =============================================================================
# Cotton — easiest, run first
# =============================================================================
download_cotton() {
  echo
  echo "===== [1] Cotton G. hirsutum 1,081 accessions (Hebei Ag U) ====="
  echo "       ~288 MB — minutes"
  echo "       NOTE: portal is plain HTTP (no TLS). This is normal for the host."

  aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/cotton/snp" \
    "http://cotton.hebau.edu.cn/wenjian/genotype-data.zip"

  aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/cotton/phenotype" \
    "http://cotton.hebau.edu.cn/wenjian/phenotypic-data.zip" \
    "http://cotton.hebau.edu.cn/wenjian/Gene-expression-data.zip"

  echo "  cotton done — sizes:"
  du -sh "$ROOT"/data/raw/cotton/*/ 2>/dev/null
}

# =============================================================================
# B. napus — small VCF set + Springer phenotype xlsx
# =============================================================================
download_napus() {
  echo
  echo "===== [2] B. napus 2,311-accession merged VCF (BnIR / HZAU) ====="
  echo "       ~770 MB across 19 chromosomes (A01-A10 + C01-C09)"
  echo "       NOTE: this is the BnIR 2,311 panel. The Wu 2019 991 accessions"
  echo "             are a subset — subset later with bcftools -S 991_ids.txt."

  BNIR="https://yanglab.hzau.edu.cn/static/bnir/assets/variation_data"
  URLS=()
  for chr in A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 \
             C01 C02 C03 C04 C05 C06 C07 C08 C09; do
    URLS+=("$BNIR/bna2311_ZS11_SNP.${chr}.vcf.gz")
  done
  aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/rapeseed/vcf" "${URLS[@]}"

  echo "  pulling Wu 2019 supplementary phenotype table ..."
  aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/rapeseed/phenotype" \
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09134-9/MediaObjects/41467_2019_9134_MOESM5_ESM.xlsx"

  echo "  rapeseed done — sizes:"
  du -sh "$ROOT"/data/raw/rapeseed/*/ 2>/dev/null
}

# =============================================================================
# Watkins wheat — the long-tail
# =============================================================================
download_wheat() {
  echo
  echo "===== [3] Watkins wheat 1,051-accession VCFs (ENA mirror) ====="
  echo "       ~430 GB across 21 chromosomes — HOURS to DAYS depending on link"
  echo "       Ctrl-C is safe; aria2c -c resumes on next run."
  echo "       Recommend running inside tmux / screen."

  ENA="https://ftp.sra.ebi.ac.uk/vol1/analysis/ERZ221/ERZ22157275"
  # Filename suffix is identical across chromosomes (filtered SNP VCFs, Apr 2024 re-call).
  SFX="SNP.Missing-unphasing.ID.ann.finalSID.allele2_retain.hard_retain.InbreedingCoeff_retain.missing_retain.maf_retain.vcf.gz"

  URLS=()
  for chr in 1A 1B 1D 2A 2B 2D 3A 3B 3D 4A 4B 4D 5A 5B 5D 6A 6B 6D 7A 7B 7D; do
    URLS+=("$ENA/chr${chr}.${SFX}")
  done
  aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/wheat/vcf" "${URLS[@]}"

  echo "  pulling Watkins phenotype directory (Earlham webdav) ..."
  # Recursive wget — small files only (pdf/csv/tsv/xlsx/txt).
  # || true: phenotype mirror is bonus, do not fail the whole script if it 404s.
  wget -r -np -nH --cut-dirs=5 -nv \
       -P "$ROOT/data/raw/wheat/phenotype" \
       -R "index.html*,*.tmp" \
       -A "*.csv,*.tsv,*.xlsx,*.txt,*.pdf" \
       --no-check-certificate \
       "https://opendata.earlham.ac.uk/wheat/under_license/toronto/WatSeq_2023-09-15_landrace_modern_Variation_Data/Watseq_phenotype_data/" \
       || echo "  (phenotype mirror returned non-zero; continuing — files may still have been pulled)"

  echo "  wheat done — sizes:"
  du -sh "$ROOT"/data/raw/wheat/*/ 2>/dev/null
}

# =============================================================================
# Dispatch
# =============================================================================
case "$WHAT" in
  cotton) download_cotton ;;
  napus)  download_napus ;;
  wheat)  download_wheat ;;
  all)
    download_cotton
    download_napus
    download_wheat
    ;;
  *)
    echo "Usage: $0 [all|cotton|napus|wheat]"
    exit 1
    ;;
esac

# =============================================================================
# Summary
# =============================================================================
echo
echo "============================================================"
echo " W2 setup complete — $(date)"
echo "============================================================"
du -sh "$ROOT"/data/raw/* 2>/dev/null | sort -k2
echo
echo "Quick sanity check:"
echo "  bcftools view -h $ROOT/data/raw/wheat/vcf/chr1A.${SFX:-...vcf.gz} | tail -5"
echo "  bcftools view -h $ROOT/data/raw/rapeseed/vcf/bna2311_ZS11_SNP.A01.vcf.gz | tail -5"
echo "  unzip -l $ROOT/data/raw/cotton/snp/genotype-data.zip | head -20"
echo
echo "Next:"
echo "  1. Subset B. napus 991: bcftools view -S 991_ids.txt --threads 8 ..."
echo "  2. Unzip cotton zips into a working dir"
echo "  3. Start W3 — phenotype harmonization to data/processed/"
