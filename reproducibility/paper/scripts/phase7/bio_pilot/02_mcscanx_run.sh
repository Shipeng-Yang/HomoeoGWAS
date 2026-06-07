#!/bin/bash
# Phase 7 bio-pilot Step 2: diamond blastp self + MCScanX on retained A01-D01 PEP set.
# Run-fast: small subset (332 genes), should complete in seconds.
set -euo pipefail

PROJ=/mnt/7302share/fast_ysp/U7_GWAS
IN=$PROJ/results/phase7/bio_pilot
WORK=/mnt/nvme/cotton_hbau/mcscanx_pilot
DIAMOND=~/.local/share/mamba/envs/polygwas-cpu/bin/diamond
MCSCANX=/mnt/lldata/sz/shared_tools/MCScanX/MCScanX

mkdir -p "$WORK"

# MCScanX expects PREFIX.gff and PREFIX.blast in the SAME dir, run as: MCScanX PREFIX
PREFIX=$WORK/pilot
cp "$IN/mcscanx.gff" "$PREFIX.gff"

echo "=== diamond makedb ==="
$DIAMOND makedb --in "$IN/retained_pep.fa" -d "$WORK/pilot_db" --threads 16 2>&1 | tail -3

echo "=== diamond blastp self vs self (top 10 hits/query, e<=1e-5) ==="
$DIAMOND blastp -q "$IN/retained_pep.fa" -d "$WORK/pilot_db" \
  -o "$PREFIX.blast" --outfmt 6 -k 10 -e 1e-5 --threads 16 2>&1 | tail -3

echo "  blast hits: $(wc -l < $PREFIX.blast)"

echo "=== MCScanX (default m=5 max_gaps, s=5 min_anchors) ==="
# MCScanX writes outputs as PREFIX.collinearity etc. in the same dir
$MCSCANX "$PREFIX" 2>&1 | tail -8

echo "=== outputs ==="
ls -la "$WORK"/pilot.* 2>/dev/null

echo "=== collinearity head ==="
head -30 "$PREFIX.collinearity" 2>/dev/null | tail -25

echo "=== alignment count + cross-subgenome pair count ==="
grep -c "^## Alignment" "$PREFIX.collinearity" 2>/dev/null || echo 0
# pair lines look like "  0-  0:	aa01_GhM_A01G... 	dd01_GhM_D01G... 	1e-X"
# count A-D inter-subgenome pairs
grep -cE "^[ 0-9-]+:[[:space:]]+(aa\S+\s+dd|dd\S+\s+aa)" "$PREFIX.collinearity" 2>/dev/null || echo 0

echo "=== DONE ==="
