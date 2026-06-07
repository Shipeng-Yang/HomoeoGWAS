#!/bin/bash
# Phase A Step 2: diamond blastp self vs self + MCScanX on the UNION retained PEP
# backbone (Codex Phase A: shared backbone across body / ±2kb modes -> later mode
# differences are pure SNP-assignment differences, not synteny redefinition).
# Top-20 hits/query (Codex recommendation, up from pilot's top-10); MCScanX defaults
# m=5 min_anchors, max_gaps=25 (more permissive than pilot's default 5 — Codex notes
# MCScanX max_gaps semantics depend on whether gaps are counted in all GFF genes or
# only retained PEP; 25 matches MCScanX standard usage).
set -euo pipefail

PROJ=/mnt/7302share/fast_ysp/U7_GWAS
IN=$PROJ/results/phase7/bio_full
WORK=/mnt/nvme/cotton_hbau/mcscanx_full
DIAMOND=~/.local/share/mamba/envs/polygwas-cpu/bin/diamond
MCSCANX=/mnt/lldata/sz/shared_tools/MCScanX/MCScanX

mkdir -p "$WORK"
PREFIX=$WORK/union
cp "$IN/union_mcscanx.gff" "$PREFIX.gff"

echo "=== diamond makedb (union PEP, $(grep -c '^>' $IN/union_pep.fa) seqs) ==="
$DIAMOND makedb --in "$IN/union_pep.fa" -d "$WORK/union_db" --threads 32 2>&1 | tail -2

echo "=== diamond blastp self vs self (top 20/query, e≤1e-5) ==="
$DIAMOND blastp -q "$IN/union_pep.fa" -d "$WORK/union_db" \
  -o "$PREFIX.blast" --outfmt 6 -k 20 -e 1e-5 --threads 32 2>&1 | tail -3
echo "  blast hits: $(wc -l < $PREFIX.blast)"

echo "=== MCScanX (m=5 min_anchors, default gaps) ==="
$MCSCANX "$PREFIX" 2>&1 | tail -5

echo "=== outputs ==="
ls -la "$WORK"/union.* 2>/dev/null | head

echo "=== collinearity summary ==="
ALN_COUNT=$(grep -c "^## Alignment" "$PREFIX.collinearity" 2>/dev/null || echo 0)
PAIR_COUNT=$(grep -cE "^[ ]*[0-9]+- *[0-9]+:" "$PREFIX.collinearity" 2>/dev/null || echo 0)
echo "  alignments: $ALN_COUNT,  total pair lines: $PAIR_COUNT"

# count inter-subgenome (A-D) blocks via the alignment header line pattern "aa..&dd.." or "dd..&aa.."
INTER_ALN=$(grep -cE "^## Alignment.*(aa\S*&dd|dd\S*&aa)" "$PREFIX.collinearity" 2>/dev/null || echo 0)
echo "  A-D inter-subgenome alignments: $INTER_ALN"

echo "=== DONE ==="
