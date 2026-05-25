#!/usr/bin/env bash
# =============================================================================
# U7_GWAS — Phase 2: COTTON (allotetraploid AADD)
# Downloads:
#   - Gossypium hirsutum TM-1 NAU v1.1 reference   ~2 GB (NCBI)
#   - 1,081-accession SNP zip (Hebei Ag U lab)     287 MB
#   - Phenotype zip + Gene-expression zip          ~520 KB
# Total: ~3 GB; minutes-level.
# Target: Synology NAS at /volume2/U7_GWAS
# Usage:   bash phase2_cotton.sh
# URLs verified 2026-05-13.
#
# NOTE on TM-1 reference:
#   We use NCBI-mirrored NAU TM-1 v1.1 (Zhang 2015 Nat Biotechnol) because it
#   is the most widely cited Upland cotton reference and is freely accessible.
#   Newer assemblies (HAU 2019, UTX 2020) live only on CottonGen and require
#   registration. Substitute later if a specific GWAS panel demands it.
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
if [ "${FREE_GB:-0}" -lt 10 ]; then
  echo "WARNING: $ROOT has only ${FREE_GB:-unknown} GB free; ≥10 GB recommended."
fi

# ---------- Directories ----------
mkdir -p "$ROOT"/data/reference/cotton
mkdir -p "$ROOT"/data/raw/cotton/{snp,phenotype}
mkdir -p "$ROOT"/logs

LOG="$ROOT/logs/phase2_cotton_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo " U7_GWAS Phase 2 — COTTON — $(date)"
echo " ROOT=$ROOT, free=${FREE_GB} GB, LOG=$LOG"
echo "============================================================"

# =============================================================================
# Step 1 — TM-1 NAU v1.1 reference (NCBI GCA_000987745.1) — ~2 GB
# =============================================================================
echo
echo "===== [1/3] Reference genome (NCBI RefSeq TM-1 NAU v1.1, GCF_000987745.1) ====="
# Use the RefSeq (GCF) mirror — same assembly content as GCA but with full
# gene annotation (gff). GCA path lacked the gff (404).

aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/reference/cotton" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/987/745/GCF_000987745.1_ASM98774v1/GCF_000987745.1_ASM98774v1_genomic.fna.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/987/745/GCF_000987745.1_ASM98774v1/GCF_000987745.1_ASM98774v1_genomic.gff.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/987/745/GCF_000987745.1_ASM98774v1/md5checksums.txt"

# =============================================================================
# Step 2 — 1,081-accession SNP zip (Hebei Ag U lab portal) — 287 MB
# =============================================================================
echo
echo "===== [2/3] G. hirsutum 1,081-accession SNP data (Hebei Ag U) ====="
echo "  NOTE: portal is plain HTTP (no TLS). This is the host's normal setup."

aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/cotton/snp" \
  "http://cotton.hebau.edu.cn/wenjian/genotype-data.zip"

# =============================================================================
# Step 3 — Phenotype + gene-expression zips — <1 MB
# =============================================================================
echo
echo "===== [3/3] Phenotype + gene-expression zips ====="

aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/raw/cotton/phenotype" \
  "http://cotton.hebau.edu.cn/wenjian/phenotypic-data.zip" \
  "http://cotton.hebau.edu.cn/wenjian/Gene-expression-data.zip"

# =============================================================================
# Summary
# =============================================================================
echo
echo "============================================================"
echo " Phase 2 (cotton) complete — $(date)"
echo "============================================================"
du -sh "$ROOT"/data/reference/cotton "$ROOT"/data/raw/cotton/* 2>/dev/null
echo
echo "Sanity check:"
echo "  unzip -l $ROOT/data/raw/cotton/snp/genotype-data.zip | head -20"
echo "  unzip -l $ROOT/data/raw/cotton/phenotype/phenotypic-data.zip"
echo
echo "Next:"
echo "  - Run phase3_rapeseed.sh or phase1_wheat.sh"
echo "  - scripts/preprocess/cotton_unzip_and_convert.sh (TBW)"
