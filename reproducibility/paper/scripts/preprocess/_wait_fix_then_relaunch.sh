#!/usr/bin/env bash
# Wait for the bcftools upgrade + rapeseed reblock to finish, then restart wheat / rapeseed split.
# Cotton is already running, leave it.
set -uo pipefail

ROOT="/mnt/7302share/fast_ysp/U7_GWAS"
ENV="$HOME/.local/share/mamba/envs/polygwas-cpu"
UP_PID=1886476
RB_PID=1886519

cd "$ROOT"
echo "[$(date)] waiting for bcftools upgrade (PID=$UP_PID) ..."
while kill -0 "$UP_PID" 2>/dev/null; do sleep 5; done
echo "[$(date)] upgrade done."

echo "[$(date)] waiting for rapeseed reblock (PID=$RB_PID) ..."
while kill -0 "$RB_PID" 2>/dev/null; do sleep 5; done
echo "[$(date)] reblock done."

# Verify
BCFV=$("$ENV/bin/bcftools" --version | head -1 | awk '{print $2}')
echo "[$(date)] bcftools version: $BCFV"
RAPE_VCF="$ROOT/data/raw/rapeseed/Horvath2020_Zenodo4302088/imputed.vcf.gz"
if [ ! -f "${RAPE_VCF}.tbi" ]; then
  echo "ERR: rapeseed .tbi missing after reblock" >&2
  exit 5
fi
"$ENV/bin/htsfile" "$RAPE_VCF" | grep -q BGZF || { echo "ERR: rapeseed not BGZF" >&2; exit 6; }
echo "[$(date)] rapeseed imputed.vcf.gz is BGZF, .tbi exists. OK."

# Restart wheat
setsid bash -c "PATH=$ENV/bin:\$PATH; cd $ROOT; exec bash scripts/preprocess/wheat_split_subgenome.sh" \
  > logs/preprocess/wheat.run.log 2>&1 &
echo $! > logs/preprocess/wheat.pid
echo "[$(date)] wheat launched PID=$(cat logs/preprocess/wheat.pid)"

# Restart rapeseed
setsid bash -c "PATH=$ENV/bin:\$PATH; cd $ROOT; exec bash scripts/preprocess/rapeseed_split_subgenome.sh" \
  > logs/preprocess/rapeseed.run.log 2>&1 &
echo $! > logs/preprocess/rapeseed.pid
echo "[$(date)] rapeseed launched PID=$(cat logs/preprocess/rapeseed.pid)"

sleep 3
for s in wheat rapeseed; do
  pid=$(cat logs/preprocess/$s.pid)
  if kill -0 "$pid" 2>/dev/null; then echo "  $s alive (PID=$pid)"; else echo "  $s DEAD"; tail -5 logs/preprocess/$s.run.log; fi
done
echo "[$(date)] relaunch wrapper done."
