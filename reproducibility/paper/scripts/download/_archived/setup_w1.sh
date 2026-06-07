#!/usr/bin/env bash
# =============================================================================
# U7_GWAS — W1 setup script
# Downloads P0 reference genomes + DL prior models + baseline tool repos.
# Target machine: Synology NAS, working dir /volume2/U7_GWAS
# Expected total: ~13 GB
# Resumable: aria2c -c is set; re-run after any interruption.
# Usage:    bash setup_w1.sh
# Verified: 2026-05-13 (URLs confirmed via HEAD)
# =============================================================================
set -euo pipefail

# ---------- Config ----------
ROOT="/volume2/U7_GWAS"
ARIA_OPTS=(-x 16 -s 16 -c
           --auto-file-renaming=false
           --allow-overwrite=false
           --console-log-level=warn
           --summary-interval=30)

# ---------- Tool checks ----------
command -v aria2c >/dev/null || { echo "ERROR: aria2c not installed"; exit 1; }
command -v git    >/dev/null || { echo "ERROR: git not installed";    exit 1; }
command -v curl   >/dev/null || { echo "ERROR: curl not installed";   exit 1; }

# ---------- Directory layout ----------
mkdir -p "$ROOT"/data/reference/{wheat,cotton,rapeseed,strawberry,models}
mkdir -p "$ROOT"/benchmarks "$ROOT"/logs

# ---------- Logging ----------
LOG="$ROOT/logs/setup_w1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo " U7_GWAS W1 setup — $(date)"
echo " ROOT = $ROOT"
echo " LOG  = $LOG"
echo "============================================================"

# =============================================================================
# Part 1 — Reference genomes (NCBI FTP via HTTPS)
# =============================================================================
echo
echo "[1/3] Reference genomes ..."

# --- Wheat: IWGSC RefSeq v2.1 (Chinese Spring) — confirmed 4.33 GB ---
echo "  · Wheat IWGSC RefSeq v2.1 (~4.3 GB)"
aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/reference/wheat" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/294/505/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1_genomic.fna.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/294/505/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1_genomic.gff.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/294/505/GCF_018294505.1_IWGSC_CS_RefSeq_v2.1/md5checksums.txt"

# --- Cotton: TM-1 NAU v1.1 (Zhang 2015 Nat Biotechnol) ---
# NOTE: HAU 2019 / UTX 2020 newer TM-1 assemblies are only on CottonGen and
#       require account registration. NAU v1.1 is NCBI-mirrored and is the
#       most widely cited Upland cotton reference. Can upgrade later.
echo "  · Cotton TM-1 NAU v1.1"
aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/reference/cotton" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/987/745/GCA_000987745.1_ASM98774v1/GCA_000987745.1_ASM98774v1_genomic.fna.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/987/745/GCA_000987745.1_ASM98774v1/GCA_000987745.1_ASM98774v1_genomic.gff.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/987/745/GCA_000987745.1_ASM98774v1/md5checksums.txt"

# --- Rapeseed: B. napus Darmor-bzh v2.0 ---
# NOTE: ZS11 (HZAU 2017) is only on BnIR (requires registration). Darmor-bzh
#       is the NCBI-mirrored, widely used B. napus reference. Substitute ZS11
#       later if a specific GWAS panel requires it.
echo "  · Rapeseed B. napus Darmor-bzh v2.0"
aria2c "${ARIA_OPTS[@]}" -d "$ROOT/data/reference/rapeseed" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/686/985/GCA_000686985.2_Bra_napus_v2.0/GCA_000686985.2_Bra_napus_v2.0_genomic.fna.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/686/985/GCA_000686985.2_Bra_napus_v2.0/GCA_000686985.2_Bra_napus_v2.0_genomic.gff.gz" \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/686/985/GCA_000686985.2_Bra_napus_v2.0/md5checksums.txt"

# --- Strawberry: SKIPPED in W1 ---
# Royal Royce FaRR1 (Hardigan 2024) is hosted only on GDR (rosaceae.org)
# behind a Ft. Lauderdale agreement; no NCBI mirror found.
# Strawberry is P2 (v2 stress test) in docs/01_project_plan.md.
# When ready, visit https://www.rosaceae.org/Analysis/12335030 manually,
# accept the agreement, then aria2c into $ROOT/data/reference/strawberry/
echo "  · Strawberry: SKIPPED (P2, deferred to v2; see docs/01_project_plan.md)"

# =============================================================================
# Part 2 — DL prior models (HuggingFace)
# =============================================================================
echo
echo "[2/3] DL prior models ..."

SKIP_HF=""
if ! command -v huggingface-cli >/dev/null 2>&1; then
  if command -v pip3 >/dev/null 2>&1; then
    echo "  Installing huggingface_hub via pip3 ..."
    pip3 install --user --quiet 'huggingface_hub[cli]'
  elif command -v pip >/dev/null 2>&1; then
    echo "  Installing huggingface_hub via pip ..."
    pip install --user --quiet 'huggingface_hub[cli]'
  else
    echo "  WARNING: pip not available — skipping DL model downloads."
    echo "           Install Python3 from Synology DSM Package Center, then re-run,"
    echo "           OR pull models on another machine and rsync to"
    echo "           $ROOT/data/reference/models/"
    SKIP_HF=1
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ -z "$SKIP_HF" ]; then
  # If you are in mainland China and HuggingFace is slow, uncomment:
  # export HF_ENDPOINT=https://hf-mirror.com

  echo "  · PlantCaduceus L32 (~2 GB)"
  huggingface-cli download kuleshov-group/PlantCaduceus_l32 \
    --local-dir "$ROOT/data/reference/models/plantcaduceus" \
    --local-dir-use-symlinks False

  echo "  · AgroNT 1B (~4 GB)"
  huggingface-cli download InstaDeepAI/agro-nucleotide-transformer-1b \
    --local-dir "$ROOT/data/reference/models/agront" \
    --local-dir-use-symlinks False
fi

# =============================================================================
# Part 3 — Baseline tool repos (git clone, shallow)
# =============================================================================
echo
echo "[3/3] Baseline tool source repos ..."

REPOS=(
  "NodeGWAS|https://github.com/zhangyixing3/NodeGWAS"
  "GWASpoly|https://github.com/jendelman/GWASpoly"
  "networkGWAS|https://github.com/BorgwardtLab/networkGWAS"
  "REGENIE|https://github.com/rgcgithub/regenie"
  "SAIGE|https://github.com/saigegit/SAIGE"
  "PanGenie|https://github.com/eblerjana/pangenie"
  "STAAR|https://github.com/xihaoli/STAAR"
  "DeepRVAT|https://github.com/PMBio/deeprvat"
)

for entry in "${REPOS[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  target="$ROOT/benchmarks/$name"
  if [ -d "$target/.git" ]; then
    echo "  · $name: already cloned, skip"
  else
    echo "  · clone $name"
    git clone --depth 1 "$url" "$target"
  fi
done

# =============================================================================
# Summary
# =============================================================================
echo
echo "============================================================"
echo " Done — $(date)"
echo "============================================================"
echo
echo "Per-tier sizes:"
du -sh "$ROOT"/data/reference/* "$ROOT"/benchmarks/* 2>/dev/null | sort -k2
echo
echo "MD5 verification (optional):"
echo "  cd $ROOT/data/reference/wheat   && grep _genomic.fna.gz md5checksums.txt | md5sum -c -"
echo "  cd $ROOT/data/reference/cotton  && grep _genomic.fna.gz md5checksums.txt | md5sum -c -"
echo "  cd $ROOT/data/reference/rapeseed && grep _genomic.fna.gz md5checksums.txt | md5sum -c -"
echo
echo "Next steps:"
echo "  1. Inspect any aria2c warnings in $LOG"
echo "  2. Start W2 panel downloads (Watkins wheat, B. napus 991, cotton 1081)"
echo "  3. Build conda env: envs/polygwas.yaml (TBW)"
