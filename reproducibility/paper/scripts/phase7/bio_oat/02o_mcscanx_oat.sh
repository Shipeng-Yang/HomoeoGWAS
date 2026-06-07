#!/bin/bash
# OAT Step 2b: diamond self-blast (full proteome) + MCScanX -> genome-wide collinearity.
# Backbone = ALL genes (full_pep.fa / full_mcscanx.gff from 02o_build_backbone.py).
set -euo pipefail

PROJ=/mnt/7302share/fast_ysp/U7_GWAS
IN=$PROJ/results/phase7/bio_oat
WORK=/mnt/nvme/oat_raw/mcscanx
DIAMOND=/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin/diamond
MCSCANX=/mnt/lldata/sz/shared_tools/MCScanX/MCScanX

mkdir -p "$WORK"
PREFIX=$WORK/oat
cp "$IN/full_mcscanx.gff" "$PREFIX.gff"

echo "=== diamond makedb (full PEP, $(grep -c '^>' "$IN/full_pep.fa") seqs) ==="
$DIAMOND makedb --in "$IN/full_pep.fa" -d "$WORK/oat_db" --threads 32 2>&1 | tail -1

echo "=== diamond blastp self vs self (top 20/query, e<=1e-5) ==="
$DIAMOND blastp -q "$IN/full_pep.fa" -d "$WORK/oat_db" \
  -o "$PREFIX.blast" --outfmt 6 -k 20 -e 1e-5 --threads 32 2>&1 | tail -2
echo "  blast hits: $(wc -l < "$PREFIX.blast")"

echo "=== MCScanX (defaults: m=5 min_anchors) ==="
$MCSCANX "$PREFIX" 2>&1 | tail -5

ALN=$(grep -c "^## Alignment" "$PREFIX.collinearity" 2>/dev/null || echo 0)
echo "=== collinearity: $ALN alignments -> $PREFIX.collinearity ==="
ls -la "$PREFIX".collinearity 2>/dev/null
echo "DONE"
