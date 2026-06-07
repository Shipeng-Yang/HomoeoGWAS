#!/bin/bash
# Phase B rapeseed Step 2: diamond + MCScanX on union 4409 PEP backbone.
# Labels: aaXX (A subgenome) vs ccXX (C subgenome) for inter-subgenome A-C detection.
set -euo pipefail

PROJ=/mnt/7302share/fast_ysp/U7_GWAS
IN=$PROJ/results/phase7/bio_rapeseed
WORK=/mnt/nvme/rapeseed_mcscanx
DIAMOND=~/.local/share/mamba/envs/polygwas-cpu/bin/diamond
MCSCANX=/mnt/lldata/sz/shared_tools/MCScanX/MCScanX

mkdir -p "$WORK"
PREFIX=$WORK/union
cp "$IN/union_mcscanx.gff" "$PREFIX.gff"

echo "=== diamond makedb (rapeseed union PEP, $(grep -c '^>' $IN/union_pep.fa) seqs) ==="
$DIAMOND makedb --in "$IN/union_pep.fa" -d "$WORK/union_db" --threads 32 2>&1 | tail -2

echo "=== diamond blastp self vs self (top 20/query, e≤1e-5) ==="
$DIAMOND blastp -q "$IN/union_pep.fa" -d "$WORK/union_db" \
  -o "$PREFIX.blast" --outfmt 6 -k 20 -e 1e-5 --threads 32 2>&1 | tail -3
echo "  blast hits: $(wc -l < $PREFIX.blast)"

echo "=== MCScanX ==="
$MCSCANX "$PREFIX" 2>&1 | tail -5

echo "=== collinearity summary ==="
ALN_COUNT=$(grep -c "^## Alignment" "$PREFIX.collinearity" 2>/dev/null || echo 0)
PAIR_COUNT=$(grep -cE "^[ ]*[0-9]+- *[0-9]+:" "$PREFIX.collinearity" 2>/dev/null || echo 0)
echo "  alignments: $ALN_COUNT,  total pair lines: $PAIR_COUNT"
INTER_ALN=$(grep -cE "^## Alignment.*(aa\S*&cc|cc\S*&aa)" "$PREFIX.collinearity" 2>/dev/null || echo 0)
echo "  A-C inter-subgenome alignments: $INTER_ALN"
echo "=== DONE ==="
