#!/usr/bin/env bash
# Clone the eight baseline GWAS / rare-variant / pangenome tools that HomoeoGWAS
# benchmarks against. They are NOT vendored in this repo (size + heterogeneous
# licenses). Pinned to upstream default branches; see each tool's own LICENSE.
#
# Usage:  bash scripts/reproduce_baselines.sh [target_dir]
# Default target_dir = benchmarks/
set -euo pipefail

TARGET="${1:-benchmarks}"
mkdir -p "$TARGET"
cd "$TARGET"

declare -A REPOS=(
  [NodeGWAS]="https://github.com/zhangyixing3/NodeGWAS"
  [GWASpoly]="https://github.com/jendelman/GWASpoly"
  [networkGWAS]="https://github.com/BorgwardtLab/networkGWAS"
  [regenie]="https://github.com/rgcgithub/regenie"
  [SAIGE]="https://github.com/saigegit/SAIGE"
  [pangenie]="https://github.com/eblerjana/pangenie"
  [STAAR]="https://github.com/xihaoli/STAAR"
  [deeprvat]="https://github.com/PMBio/deeprvat"
)

for name in "${!REPOS[@]}"; do
  url="${REPOS[$name]}"
  if [ -d "$name/.git" ]; then
    echo "[skip] $name already cloned"
  else
    echo "[clone] $name <- $url"
    git clone --depth 1 "$url" "$name"
  fi
done

echo "Done. Eight baseline tools available under $TARGET/"
echo "Build/install each per its own README (regenie/SAIGE/NodeGWAS need compilation)."
