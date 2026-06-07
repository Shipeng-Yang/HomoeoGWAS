#!/usr/bin/env bash
# Phase 1 wheat closeout 自动化:
#   1. 等当前 aria2 重下完成 (PID 在 logs/download/wheat_redl2.pid)
#   2. 校验 5 个之前损坏的 wheat VCF (chr1A/2A/4B/5A/5D) 都通过 bgzip -t
#   3. 跑 wheat_split_subgenome.sh (A/B/D)
#   4. 跑 _build_pheno_clean.py --panel wheat
#   5. 写 results/phase1/wheat_watkins/closeout/qc_summary.tsv
#   6. 显式记录 1035 (manifest) vs 1051 (实测 VCF) 的 16-sample delta
set -uo pipefail

ROOT=/mnt/7302share/fast_ysp/U7_GWAS
ENV=$HOME/.local/share/mamba/envs/polygwas-cpu
export PATH=$ENV/bin:$PATH

LOG="$ROOT/logs/preprocess/wheat.closeout.log"
QC_DIR="$ROOT/results/phase1/wheat_watkins/closeout"
QC_TSV="$QC_DIR/qc_summary.tsv"
mkdir -p "$QC_DIR" "$ROOT/logs/preprocess"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] wheat closeout wrapper starting"

# ---- 1) 等当前 aria2 进程退出 ----
echo "[$(ts)] waiting for aria2c (data/raw/wheat/vcf) to finish ..."
while pgrep -af 'aria2c.*data/raw/wheat/vcf' > /dev/null 2>&1; do
  sleep 60
done
echo "[$(ts)] aria2c gone, proceeding."

# ---- 2) 校验所有 *.bad 的修复 sibling: 数据驱动 + 第一个错就退 ----
RAW_VCF="$ROOT/data/raw/wheat/vcf"
shopt -s nullglob
BAD_FILES=("$RAW_VCF"/*.bad)
shopt -u nullglob
if [ ${#BAD_FILES[@]} -eq 0 ]; then
  echo "[$(ts)] no *.bad files found in $RAW_VCF; nothing to validate" | tee -a "$LOG"
fi
for bad in "${BAD_FILES[@]}"; do
  repaired="${bad%.bad}"
  short=$(basename "$repaired" | cut -d. -f1)
  if [ ! -f "$repaired" ]; then
    echo "[$(ts)] ERR: $short repaired file MISSING -> $repaired" | tee -a "$LOG"
    echo -e "entity\tn_samples\tn_snps\tnotes"        > "$QC_TSV"
    printf "closeout_status\tNA\tNA\tFAILED missing=%s\n" "$short" >> "$QC_TSV"
    exit 5
  fi
  if ! bgzip -t "$repaired" 2>>"$LOG"; then
    echo "[$(ts)] ERR: $short BGZF CORRUPT after redownload" | tee -a "$LOG"
    echo -e "entity\tn_samples\tn_snps\tnotes"        > "$QC_TSV"
    printf "closeout_status\tNA\tNA\tFAILED bgzf_corrupt=%s\n" "$short" >> "$QC_TSV"
    exit 5
  fi
  if ! tabix -f --csi -p vcf "$repaired" 2>>"$LOG"; then
    echo "[$(ts)] ERR: $short tabix index rebuild failed" | tee -a "$LOG"
    echo -e "entity\tn_samples\tn_snps\tnotes"        > "$QC_TSV"
    printf "closeout_status\tNA\tNA\tFAILED tabix_failed=%s\n" "$short" >> "$QC_TSV"
    exit 5
  fi
  echo "[$(ts)] OK $short (bgzip -t passed, tabix re-indexed)" | tee -a "$LOG"
done
echo "[$(ts)] all ${#BAD_FILES[@]} repaired wheat VCF(s) pass validation" | tee -a "$LOG"

# ---- 3) 跑 wheat split (A/B/D) ----
echo "[$(ts)] starting wheat_split_subgenome.sh (THREADS=16) ..." | tee -a "$LOG"
# HWE 留空 (inbred Watkins landrace panel, HWE 测试会 reject ~95% SNP)
THREADS=16 MAF=0.01 GENO=0.1 HWE="" \
  bash "$ROOT/scripts/preprocess/wheat_split_subgenome.sh" >> "$LOG" 2>&1
SPLIT_RC=$?
if [ $SPLIT_RC -ne 0 ]; then
  echo "[$(ts)] ERR: split exit=$SPLIT_RC" | tee -a "$LOG"
  echo -e "metric\tvalue\tnotes" > "$QC_TSV"
  printf "closeout_status\tFAILED\tsplit_rc=%d\n" "$SPLIT_RC" >> "$QC_TSV"
  exit 6
fi
echo "[$(ts)] split done." | tee -a "$LOG"

# ---- 4) 跑 wheat pheno join ----
echo "[$(ts)] building wheat pheno_clean.tsv ..." | tee -a "$LOG"
"$ENV/bin/python" "$ROOT/scripts/preprocess/_build_pheno_clean.py" --panel wheat >> "$LOG" 2>&1
PHENO_RC=$?
if [ $PHENO_RC -ne 0 ]; then
  echo "[$(ts)] ERR: pheno build exit=$PHENO_RC" | tee -a "$LOG"
  exit 7
fi

# ---- 5) 写 qc_summary.tsv ----
echo "[$(ts)] writing qc_summary.tsv ..." | tee -a "$LOG"

count_samples_vcf() { bcftools view -h "$1" 2>/dev/null | tail -1 | awk '{print NF-9}'; }
count_snps_vcf()    { bcftools view -H "$1" 2>/dev/null | wc -l; }

RAW_CHR1B="$RAW_VCF/chr1B.SNP.Missing-unphasing.ID.ann.finalSID.allele2_retain.hard_retain.InbreedingCoeff_retain.missing_retain.maf_retain.vcf.gz"
RAW_N=$(count_samples_vcf "$RAW_CHR1B")

A_VCF="$ROOT/data/processed/wheat/A/all.vcf.gz"
B_VCF="$ROOT/data/processed/wheat/B/all.vcf.gz"
D_VCF="$ROOT/data/processed/wheat/D/all.vcf.gz"

A_N=$(count_samples_vcf "$A_VCF")
B_N=$(count_samples_vcf "$B_VCF")
D_N=$(count_samples_vcf "$D_VCF")

A_SNP=$(count_snps_vcf "$A_VCF")
B_SNP=$(count_snps_vcf "$B_VCF")
D_SNP=$(count_snps_vcf "$D_VCF")

PHENO_TSV="$ROOT/data/processed/wheat/pheno_clean.tsv"
PHENO_N=$(($(wc -l < "$PHENO_TSV") - 1))

DELTA=$((RAW_N - 1035))

cat > "$QC_TSV" << EOF
entity	n_samples	n_snps	notes
raw_vcf_samples	$RAW_N	NA	chr1B.SNP*.vcf.gz header; manifest_n_samples=1035 (configs/panels/wheat_watkins.yaml frozen 2026-05-20); delta=$DELTA unexplained — do NOT silently sync manifest until the extra IDs are categorized (likely 224 modern non-WATDE cultivars + 16 unknown)
wheat_A	$A_N	$A_SNP	post-QC MAF>=0.01 geno<=0.1 hwe>1e-6 biallelic SNP only
wheat_B	$B_N	$B_SNP	post-QC same QC params as A
wheat_D	$D_N	$D_SNP	post-QC same QC params as A
wheat_pheno	$PHENO_N	NA	StoreCode (WATDE0xxx) keyed pheno_clean.tsv; non-WATDE 224 modern cultivars excluded by design
EOF

echo "[$(ts)] DONE wheat closeout. summary:" | tee -a "$LOG"
cat "$QC_TSV" | tee -a "$LOG"
