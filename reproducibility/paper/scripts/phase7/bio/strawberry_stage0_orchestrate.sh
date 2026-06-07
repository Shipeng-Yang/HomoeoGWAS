#!/bin/bash
# Wait for straw_quant env to finish installing, then run Stage 0 build +
# per-quartet mappability. Self-contained; meant to run in background.
set -uo pipefail
PROJ=/mnt/7302share/fast_ysp/U7_GWAS
ENV=$HOME/.local/share/mamba/envs/straw_quant/bin
PY=$HOME/.local/share/mamba/envs/polygwas-cpu/bin/python

echo "[orch] waiting for straw_quant tools..."
for i in $(seq 1 240); do   # up to ~60 min
  if [ -x "$ENV/salmon" ] && [ -x "$ENV/fasterq-dump" ] && [ -x "$ENV/gffread" ] \
     && [ -x "$ENV/featureCounts" ] && [ -x "$ENV/samtools" ]; then
    echo "[orch] tools ready at iter $i"; break
  fi
  sleep 15
done
if ! [ -x "$ENV/salmon" ]; then echo "[orch] ABORT: env not ready"; exit 1; fi
"$ENV/salmon" --version; "$ENV/gffread" --version 2>&1 | head -1

echo "[orch] === Stage 0 build ==="
bash "$PROJ/scripts/phase7/bio/strawberry_quant_stage0_build.sh"

echo "[orch] === mappability ==="
$PY "$PROJ/scripts/phase7/bio/strawberry_quartet_mappability.py" /mnt/nvme/strawberry_quant/transcripts.fa

echo "STAGE0_ORCHESTRATE_DONE"
