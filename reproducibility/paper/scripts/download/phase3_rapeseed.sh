#!/usr/bin/env bash
# =============================================================================
# U7_GWAS — Phase 3: RAPESEED / B. napus (allotetraploid AACC)
#
# Auto-downloads (all from HZAU BnIR / Springer — reachable from mainland China):
#   - BnIR 2,311-accession merged SNP VCF   ~770 MB across 19 chromosomes
#   - Wu 2019 supplementary phenotype        43 KB (Springer)
#
# The ZS11 REFERENCE genome is a MANUAL browser step — see:
#   scripts/download/phase3_rapeseed_reference.md
#
# Total auto: ~770 MB; minutes-level.
# Target: Synology NAS at /volume2/U7_GWAS
# Usage:   bash phase3_rapeseed.sh
# URLs verified 2026-05-17.
#
# =====  WHY NO NCBI REFERENCE STEP HERE  (correctness note)  =================
#   The earlier draft pulled Darmor-bzh v2.0 from NCBI (GCF_000686985.2).
#   That was wrong on two counts:
#     1. The BnIR panel files are  bna2311_ZS11_SNP.*  — every position is
#        called against the **ZS11** assembly, NOT Darmor-bzh. Pairing
#        ZS11-coordinate VCFs with a Darmor reference silently corrupts every
#        downstream position/allele/annotation lookup.
#     2. ftp.ncbi.nlm.nih.gov is throttled/blocked from mainland China, so the
#        step exhausted its retries and `set -e` aborted the WHOLE script
#        before the BnIR VCFs ran. That is why "rapeseed wouldn't download"
#        even though A01 had worked on the first pass.
#   Fix: drop NCBI. Get the matching ZS11 reference from BnIR's own genome
#   page (China host, same coordinates + Bna gene IDs as the VCFs).
#
# NOTE on 991 vs 2,311:
#   No public 991-only VCF exists (Wu 2019 deposited only 2.26 TB raw fastq).
#   The BnIR 2,311 panel is a superset that INCLUDES the Wu 2019 991. Subset:
#     bcftools view -S 991_sample_ids.txt --threads 8 ...
#   The 991 IDs are in Wu 2019 Supp Table 5 (41467_2019_9134_MOESM5_ESM.xlsx).
# =============================================================================
set -euo pipefail
export LC_ALL=C

ROOT="/volume2/U7_GWAS"
ARIA_OPTS=(-Z -x 16 -s 16 -j 1 -c
           --auto-file-renaming=false
           --allow-overwrite=false
           --console-log-level=warn
           --summary-interval=30
           --retry-wait=10
           --max-tries=5)
# -Z = force-sequential: treat each URL as a SEPARATE file (not mirrors of one).

# ---------- Tool checks ----------
command -v aria2c >/dev/null || { echo "ERROR: aria2c not installed"; exit 1; }

# ---------- Disk check ----------
FREE_GB=$(df -k "$ROOT" 2>/dev/null | tail -1 | awk '{printf "%d", $4/1024/1024}')
if [ "${FREE_GB:-0}" -lt 5 ]; then
  echo "WARNING: $ROOT has only ${FREE_GB:-unknown} GB free; ≥5 GB recommended."
fi

# ---------- Directories ----------
mkdir -p "$ROOT"/data/reference/rapeseed
mkdir -p "$ROOT"/data/raw/rapeseed/{vcf,phenotype}
mkdir -p "$ROOT"/logs

LOG="$ROOT/logs/phase3_rapeseed_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo " U7_GWAS Phase 3 — RAPESEED — $(date)"
echo " ROOT=$ROOT, free=${FREE_GB} GB, LOG=$LOG"
echo "============================================================"

# =============================================================================
# Step 1 — BnIR 2,311-accession merged SNP VCF — ~770 MB across 19 chromosomes
#          HZAU, China host (verified reachable: A01 already downloaded once).
#          Variants are ZS11-anchored — pair with the ZS11 ref (see the .md).
# =============================================================================
echo
echo "===== [1/2] BnIR 2,311-accession merged SNP VCF (HZAU, ZS11-anchored) ====="

BNIR="https://yanglab.hzau.edu.cn/static/bnir/assets/variation_data"
URLS=()
for chr in A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 \
           C01 C02 C03 C04 C05 C06 C07 C08 C09; do
  URLS+=("$BNIR/bna2311_ZS11_SNP.${chr}.vcf.gz")
done
aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/rapeseed/vcf" "${URLS[@]}"

# =============================================================================
# Step 2 — Wu 2019 supplementary phenotype (Springer Nature) — 43 KB
# =============================================================================
echo
echo "===== [2/2] Wu 2019 phenotype + 991 sample ID table (Springer Supp 5) ====="

aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/rapeseed/phenotype" \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09134-9/MediaObjects/41467_2019_9134_MOESM5_ESM.xlsx"

# =============================================================================
# Summary
# =============================================================================
echo
echo "============================================================"
echo " Phase 3 (rapeseed) auto-download complete — $(date)"
echo "============================================================"
du -sh "$ROOT"/data/raw/rapeseed/* 2>/dev/null
N=$(ls -1 "$ROOT"/data/raw/rapeseed/vcf/bna2311_ZS11_SNP.*.vcf.gz 2>/dev/null | wc -l)
echo "  VCF chromosomes present: $N / 19"
echo
echo ">>> STILL MANUAL: the ZS11 reference genome (browser download)."
echo "    Follow  scripts/download/phase3_rapeseed_reference.md"
echo "    Page:   https://yanglab.hzau.edu.cn/BnIR/genome_data"
echo
echo "Sanity check:"
echo "  zcat $ROOT/data/raw/rapeseed/vcf/bna2311_ZS11_SNP.A01.vcf.gz | head -40"
echo "  ls -lh $ROOT/data/raw/rapeseed/phenotype/"
echo
echo "Next:"
echo "  - Get ZS11 reference (see .md), then scripts/preprocess/subset_napus_991.sh (TBW)"
