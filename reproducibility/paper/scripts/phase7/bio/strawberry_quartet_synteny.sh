#!/bin/bash
# Strawberry 8n quartet (Step 1b): DIAMOND blastp self-vs-self + MCScanX synteny.
# Reuses the cotton 02b backbone (top-20 hits/query, e<=1e-5, MCScanX m=5).
# Inputs from strawberry_quartet_prep.py: rep_pep.fa (1 rep/gene) + camarosa.gff
# (MCScanX 4-col, tag=<subgenome A-D><group 1-7>). Outputs union.collinearity for
# the downstream best-per-subgenome INTERSECT synteny quartet assembly.
set -euo pipefail

PROJ=/mnt/7302share/fast_ysp/U7_GWAS
IN=$PROJ/results/phase7/bio_strawberry
WORK=/mnt/nvme/strawberry_camarosa/mcscanx
# NOTE: env diamond 2.2.0 deadlocks at "Loading sequences" (threading bug); use
# the working static 2.1.11. Camarosa proteins use '.' as stop -> stripped in prep.
DIAMOND=/mnt/lldata/sz/shared_tools/diamond_2.1.11
MCSCANX=/mnt/lldata/sz/shared_tools/MCScanX/MCScanX

mkdir -p "$WORK"
PREFIX=$WORK/union
cp "$IN/camarosa.gff" "$PREFIX.gff"

echo "=== diamond makedb (rep PEP, $(grep -c '^>' "$IN/rep_pep.fa") genes) ==="
$DIAMOND makedb --in "$IN/rep_pep.fa" -d "$WORK/db" --threads 32 2>&1 | tail -2

echo "=== diamond blastp self vs self (top 20/query, e<=1e-5) ==="
$DIAMOND blastp -q "$IN/rep_pep.fa" -d "$WORK/db" \
  -o "$PREFIX.blast" --outfmt 6 -k 20 -e 1e-5 --threads 32 2>&1 | tail -3
echo "  blast hits: $(wc -l < "$PREFIX.blast")"

echo "=== MCScanX (m=5 min_anchors, default gaps) ==="
$MCSCANX "$PREFIX" 2>&1 | tail -5

echo "=== collinearity summary ==="
ALN=$(grep -c "^## Alignment" "$PREFIX.collinearity" 2>/dev/null || echo 0)
echo "  alignments: $ALN"
# cross-subgenome within-group block = header connects <X>n & <Y>n with same digit n, diff letter
cp "$PREFIX.blast" "$IN/union.blast"
cp "$PREFIX.collinearity" "$IN/union.collinearity"
cp "$PREFIX.gff" "$IN/union.mcscanx.gff"
echo "=== copied blast/collinearity/gff -> $IN ==="
echo "STRAWBERRY_SYNTENY_DONE"
