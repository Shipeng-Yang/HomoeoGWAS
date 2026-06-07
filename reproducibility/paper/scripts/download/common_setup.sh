#!/usr/bin/env bash
# =============================================================================
# U7_GWAS — Common setup (run once, before any phase script)
# Downloads:
#   - DL prior models (PlantCaduceus + AgroNT)        ~6 GB via hf CLI
#   - 8 baseline tool source repos                    ~1 GB (git OR tarball)
# Target: Synology NAS at /volume2/U7_GWAS
# Usage:   bash common_setup.sh
# Resumable: hf + git/tarball all skip already-present items.
# URLs verified 2026-05-13.
# =============================================================================
set -euo pipefail
export LC_ALL=C
export PATH="$HOME/.local/bin:$PATH"   # for pip --user installs (hf, etc.)

ROOT="/volume2/U7_GWAS"

ARIA_OPTS=(-x 16 -s 16 -c
           --auto-file-renaming=false
           --allow-overwrite=true
           --console-log-level=warn
           --summary-interval=30
           --retry-wait=10
           --max-tries=5)

# ---------- Tool checks ----------
command -v curl   >/dev/null || { echo "ERROR: curl not installed";   exit 1; }
command -v aria2c >/dev/null || { echo "ERROR: aria2c not installed"; exit 1; }
command -v tar    >/dev/null || { echo "ERROR: tar not installed";    exit 1; }

HAS_GIT=0
if command -v git >/dev/null 2>&1; then HAS_GIT=1; fi

# ---------- Directories ----------
mkdir -p "$ROOT"/data/reference/models
mkdir -p "$ROOT"/benchmarks
mkdir -p "$ROOT"/logs

LOG="$ROOT/logs/common_setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo " U7_GWAS common setup — $(date)"
echo " ROOT=$ROOT, LOG=$LOG"
echo " git available = $([ $HAS_GIT -eq 1 ] && echo yes || echo 'no (using tarball fallback)')"
echo "============================================================"

# =============================================================================
# Part 1 — DL prior models (HuggingFace via `hf` CLI)
# =============================================================================
echo
echo "===== [1/2] DL prior models ====="

# Use the huggingface_hub Python API directly — the `hf` / `huggingface-cli`
# CLIs on Synology DSM Python crash because the stdlib `venv` module is
# missing from the DSM Python package. The API does not depend on venv.

PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "  WARNING: no python interpreter found — skipping DL models."
  echo "  Install Python3 via Synology DSM Package Center and re-run."
else
  # Ensure huggingface_hub is installed
  if ! "$PY" -c "import huggingface_hub" 2>/dev/null; then
    if command -v pip3 >/dev/null 2>&1; then
      pip3 install --user --quiet --upgrade huggingface_hub
    elif command -v pip >/dev/null 2>&1; then
      pip install --user --quiet --upgrade huggingface_hub
    else
      echo "  WARNING: pip missing and huggingface_hub not importable; skipping."
      PY=""
    fi
  fi
fi

if [ -n "$PY" ]; then
  # Mainland-China mirror (default ON because huggingface.co is often blocked).
  # If you are outside China, comment this out to use the canonical host.
  : "${HF_ENDPOINT:=https://hf-mirror.com}"
  export HF_ENDPOINT

  echo "  using Python API ($PY) for HuggingFace downloads"
  echo "  HF_ENDPOINT = $HF_ENDPOINT"

  for spec in \
    "PlantCaduceus L32|kuleshov-group/PlantCaduceus_l32|$ROOT/data/reference/models/plantcaduceus" \
    "AgroNT 1B|InstaDeepAI/agro-nucleotide-transformer-1b|$ROOT/data/reference/models/agront"
  do
    label="${spec%%|*}"
    rest="${spec#*|}"
    repo_id="${rest%%|*}"
    local_dir="${rest##*|}"
    echo "  · $label"
    "$PY" - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$repo_id",
    local_dir="$local_dir",
    max_workers=8,
)
PYEOF
  done
else
  echo "  DL model step skipped (no Python available)."
fi

# =============================================================================
# Part 2 — Baseline tool source repos (git clone OR tarball fallback)
# =============================================================================
echo
echo "===== [2/2] Baseline tool source repos ====="

REPOS=(
  "NodeGWAS|zhangyixing3/NodeGWAS"
  "GWASpoly|jendelman/GWASpoly"
  "networkGWAS|BorgwardtLab/networkGWAS"
  "REGENIE|rgcgithub/regenie"
  "SAIGE|saigegit/SAIGE"
  "PanGenie|eblerjana/pangenie"
  "STAAR|xihaoli/STAAR"
  "DeepRVAT|PMBio/deeprvat"
)

fetch_repo() {
  local name=$1 owner_repo=$2
  local target="$ROOT/benchmarks/$name"

  # Already populated?
  if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
    echo "  · $name: already present, skip"
    return
  fi
  mkdir -p "$target"

  if [ "$HAS_GIT" -eq 1 ]; then
    echo "  · git clone $name"
    git clone --depth 1 "https://github.com/${owner_repo}.git" "$target"
  else
    echo "  · download $name (GitHub tarball)"
    local tarball="$target/_src.tar.gz"
    aria2c "${ARIA_OPTS[@]}" -d "$target" --out="_src.tar.gz" \
      "https://api.github.com/repos/${owner_repo}/tarball"
    tar -xzf "$tarball" --strip-components=1 -C "$target"
    rm -f "$tarball"
  fi
}

for entry in "${REPOS[@]}"; do
  name="${entry%%|*}"
  owner_repo="${entry##*|}"
  fetch_repo "$name" "$owner_repo"
done

# =============================================================================
# Summary
# =============================================================================
echo
echo "============================================================"
echo " Common setup done — $(date)"
echo "============================================================"
du -sh "$ROOT"/data/reference/models/* "$ROOT"/benchmarks/* 2>/dev/null | sort -k2
echo
echo "Next: run a phase script for the species you want first:"
echo "  bash phase2_cotton.sh    (~3 GB, minutes)"
echo "  bash phase3_rapeseed.sh  (~2.3 GB, minutes)"
echo "  bash phase1_wheat.sh     (~435 GB, run inside screen/tmux)"
