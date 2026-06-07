#!/bin/bash
# OAT overnight orchestrator: ETL -> backbone -> MCScanX -> triads -> deploy(flank primary
# + body sensitivity) -> rollup. Failure-tolerant (continues on step error, preserves
# partial results), writes a STATUS file + timestamped log for morning review.
PROJ=/mnt/7302share/fast_ysp/U7_GWAS
PY=/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin/python
OUT=$PROJ/results/phase7/bio_oat
STATUS=$OUT/OVERNIGHT_STATUS.txt
cd "$PROJ"
: > "$STATUS"

step() {  # step "name" cmd...
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name" | tee -a "$STATUS"
  local t0=$(date +%s)
  if "$@"; then
    echo "[$(date '+%F %T')] OK    $name ($(($(date +%s)-t0))s)" | tee -a "$STATUS"
  else
    echo "[$(date '+%F %T')] FAIL  $name (exit $?) -- continuing" | tee -a "$STATUS"
  fi
}

echo "===== OAT OVERNIGHT RUN start $(date '+%F %T') =====" | tee -a "$STATUS"

step "01_ETL"            $PY scripts/phase7/bio_oat/01o_oat_etl.py
step "02a_backbone"      $PY scripts/phase7/bio_oat/02o_build_backbone.py
step "02b_mcscanx"       bash scripts/phase7/bio_oat/02o_mcscanx_oat.sh
step "03_triads"         $PY scripts/phase7/bio_oat/03o_triad_build.py
step "04a_deploy_flank_PRIMARY" $PY scripts/phase7/bio_oat/04o_oat_deploy.py --mode flank --perm-B 500 --n-jobs 48
step "04b_deploy_body_SENS"     $PY scripts/phase7/bio_oat/04o_oat_deploy.py --mode body  --perm-B 0   --n-jobs 48
step "05_rollup"         $PY scripts/phase7/bio_oat/05o_rollup.py

echo "===== OAT OVERNIGHT RUN done $(date '+%F %T') =====" | tee -a "$STATUS"
echo "morning review: cat $OUT/OVERNIGHT_STATUS.txt ; cat $OUT/oat_master_summary.json" | tee -a "$STATUS"
