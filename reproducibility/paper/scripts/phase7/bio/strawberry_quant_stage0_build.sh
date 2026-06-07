#!/bin/bash
# Stage 0 build: Camarosa transcripts + salmon decoy index + hisat2 index.
# Runs once the straw_quant env + genome.fa.gz are in place. Gene IDs stay in
# Camarosa space (FxaC_..g) so they map directly onto the 6,355 quartets.
set -euo pipefail

PROJ=/mnt/7302share/fast_ysp/U7_GWAS
REF=$PROJ/data/raw/strawberry/camarosa
WORK=/mnt/nvme/strawberry_quant
ENV=$HOME/.local/share/mamba/envs/straw_quant/bin
mkdir -p "$WORK"

echo "=== decompress genome + gff ==="
[ -f "$WORK/genome.fa" ] || zcat "$REF/genome.fa.gz" > "$WORK/genome.fa"
[ -f "$WORK/camarosa.gff" ] || zcat "$REF/gff.gz" > "$WORK/camarosa.gff"
echo "  genome seqs: $(grep -c '^>' "$WORK/genome.fa")"
echo "  genome chrom sample: $(grep '^>' "$WORK/genome.fa" | head -3 | tr '\n' ' ')"

echo "=== gffread: extract transcripts (mRNA, spliced) ==="
"$ENV/gffread" -w "$WORK/transcripts.fa" -g "$WORK/genome.fa" "$WORK/camarosa.gff"
echo "  transcripts: $(grep -c '^>' "$WORK/transcripts.fa")"

echo "=== build salmon decoy index (transcripts + genome decoy) ==="
grep '^>' "$WORK/genome.fa" | sed 's/^>//; s/[[:space:]].*//' > "$WORK/decoys.txt"
cat "$WORK/transcripts.fa" "$WORK/genome.fa" > "$WORK/gentrome.fa"
"$ENV/salmon" index -t "$WORK/gentrome.fa" -d "$WORK/decoys.txt" \
  -i "$WORK/salmon_index" -k 31 -p 64 2>&1 | tail -4
rm -f "$WORK/gentrome.fa"

echo "=== build hisat2 genome index (unique-map confirmatory track) ==="
hisat2-build -p 64 "$WORK/genome.fa" "$WORK/hisat2_index/camarosa" 2>&1 | tail -3 || \
  { mkdir -p "$WORK/hisat2_index"; hisat2-build -p 64 "$WORK/genome.fa" "$WORK/hisat2_index/camarosa" 2>&1 | tail -3; }

echo "STAGE0_BUILD_DONE"
