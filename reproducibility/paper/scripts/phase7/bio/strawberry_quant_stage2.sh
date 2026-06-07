#!/bin/bash
# Stage 2: per-sample quantification (runs once .sra/fastq are downloaded).
# For each run in sample_manifest.tsv: get FASTQ (fasterq-dump if only .sra),
# salmon quant (selective alignment, decoy-aware, GC/seq-bias, bootstraps),
# then delete the transient FASTQ to stay disk-frugal. SINGLE-end run handled.
#
# Usage: bash strawberry_quant_stage2.sh [N_PARALLEL]
#   expects .sra in  data/raw/strawberry/rnaseq_PRJNA787565/sra/<RUN>.sra
#        OR fastq in  data/raw/strawberry/rnaseq_PRJNA787565/fastq/<RUN>{_1,_2,}.fastq.gz
set -uo pipefail
PROJ=/mnt/7302share/fast_ysp/U7_GWAS
ENV=$HOME/.local/share/mamba/envs/straw_quant/bin
IDX=/mnt/nvme/strawberry_quant/salmon_index
DATA=$PROJ/data/raw/strawberry/rnaseq_PRJNA787565
SRA=$DATA/sra
FQ=$DATA/fastq
TMP=/mnt/nvme/strawberry_quant/fq_tmp
QUANT=$PROJ/results/phase7/bio_strawberry/salmon_quant
MANIFEST=$DATA/sample_manifest.tsv
NP=${1:-6}            # samples in flight (each salmon uses THREADS)
THREADS=16
mkdir -p "$TMP" "$QUANT"

quant_one() {
  run=$1 layout=$2
  out="$QUANT/$run"
  [ -f "$out/quant.sf" ] && { echo "[skip] $run done"; return 0; }
  local r1="" r2="" single="" made_fq=0
  # prefer ENA fastq if present, else extract from .sra
  if [ -f "$FQ/${run}_1.fastq.gz" ]; then r1="$FQ/${run}_1.fastq.gz"; r2="$FQ/${run}_2.fastq.gz"
  elif [ -f "$FQ/${run}.fastq.gz" ]; then single="$FQ/${run}.fastq.gz"
  elif [ -f "$SRA/${run}.sra" ] || [ -f "$SRA/$run/${run}.sra" ]; then
    sra=$([ -f "$SRA/${run}.sra" ] && echo "$SRA/${run}.sra" || echo "$SRA/$run/${run}.sra")
    "$ENV/fasterq-dump" "$sra" --split-files -e "$THREADS" -t "$TMP" -O "$TMP" >/dev/null 2>&1
    made_fq=1
    if [ -f "$TMP/${run}_1.fastq" ]; then r1="$TMP/${run}_1.fastq"; r2="$TMP/${run}_2.fastq"
    else single="$TMP/${run}.fastq"; fi
  else echo "[MISS] $run: no fastq/sra"; return 1; fi

  if [ -n "$single" ]; then
    "$ENV/salmon" quant -i "$IDX" -l A -r "$single" -p "$THREADS" \
      --gcBias --seqBias --numBootstraps 30 --validateMappings -o "$out" >/dev/null 2>&1
  else
    "$ENV/salmon" quant -i "$IDX" -l A -1 "$r1" -2 "$r2" -p "$THREADS" \
      --gcBias --seqBias --numBootstraps 30 --validateMappings -o "$out" >/dev/null 2>&1
  fi
  rc=$?
  [ "$made_fq" = 1 ] && rm -f "$TMP/${run}"*.fastq
  [ $rc -eq 0 ] && echo "[ok] $run ($(awk 'NR==2{print $5}' "$out/lib_format_counts.json" 2>/dev/null))" || echo "[FAIL] $run rc=$rc"
}
export -f quant_one
export QUANT ENV IDX SRA FQ TMP THREADS

# drive from manifest (col1=run, col6=layout); skip header
tail -n +2 "$MANIFEST" | awk -F'\t' '{print $1"\t"$6}' | \
  xargs -P "$NP" -I{} bash -c 'set -- {}; quant_one $1 $2'
echo "STAGE2_QUANT_DONE: $(ls "$QUANT"/*/quant.sf 2>/dev/null | wc -l) samples quantified"
