#!/usr/bin/env bash
# Wait for the aria2 download to /mnt/nvme to finish -> verify integrity with bgzip -t on the local SSD
# -> mv the passing files to NFS data/raw/wheat/vcf/ -> rebuild tabix -> trigger the closeout wrapper.
# Purpose: bypass the byte-level NFS large-file write race.
set -euo pipefail

ROOT=/mnt/7302share/fast_ysp/U7_GWAS
ENV=$HOME/.local/share/mamba/envs/polygwas-cpu
export PATH=$ENV/bin:$PATH
LOG="$ROOT/logs/preprocess/wheat.nvme_repair.log"
NVME=/mnt/nvme/wheat_repair
NFS_VCF="$ROOT/data/raw/wheat/vcf"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] waiting for aria2c (NVMe) to finish ..." | tee -a "$LOG"
while pgrep -af 'aria2c.*/mnt/nvme/wheat_repair' > /dev/null 2>&1; do
  sleep 60
done
echo "[$(ts)] aria2c gone, validating on /mnt/nvme ..." | tee -a "$LOG"

# Validate + move to NFS
shopt -s nullglob
NVME_FILES=("$NVME"/*.vcf.gz)
shopt -u nullglob
if [ ${#NVME_FILES[@]} -eq 0 ]; then
  echo "[$(ts)] ERR: no .vcf.gz in $NVME" | tee -a "$LOG"; exit 5
fi

for f in "${NVME_FILES[@]}"; do
  short=$(basename "$f" | cut -d. -f1)
  echo "[$(ts)] validate $short on NVMe ..." | tee -a "$LOG"
  if ! bgzip -t "$f" 2>>"$LOG"; then
    echo "[$(ts)] ERR: $short BGZF CORRUPT on NVMe (data source level)" | tee -a "$LOG"
    exit 6
  fi
  echo "[$(ts)] OK $short ($(stat -c%s "$f") bytes)" | tee -a "$LOG"
done
echo "[$(ts)] all NVMe files validated, moving to NFS ..." | tee -a "$LOG"

for f in "${NVME_FILES[@]}"; do
  short=$(basename "$f" | cut -d. -f1)
  dst="$NFS_VCF/$(basename "$f")"
  echo "[$(ts)] move $short to NFS ..." | tee -a "$LOG"
  # rm old NFS copy if exists
  [ -f "$dst" ] && rm -f "$dst" "${dst}.tbi"
  # mv of a large file across mounts = cp + unlink; use explicit cp then unlink for control
  cp "$f" "$dst"
  # Re-validate with bgzip -t immediately after writing to NFS
  if ! bgzip -t "$dst" 2>>"$LOG"; then
    echo "[$(ts)] ERR: NFS copy of $short corrupted during write — NFS write race confirmed" | tee -a "$LOG"
    exit 7
  fi
  if ! tabix -f --csi -p vcf "$dst" 2>>"$LOG"; then
    echo "[$(ts)] ERR: tabix failed for $short on NFS" | tee -a "$LOG"
    exit 8
  fi
  echo "[$(ts)] OK $short on NFS + tabix" | tee -a "$LOG"
  # Only delete the NVMe copy on success (safety)
  rm -f "$f"
done

echo "[$(ts)] all repaired files now on NFS with valid bgzf + tabix. closeout can proceed." | tee -a "$LOG"
echo "[$(ts)] auto-launching closeout wrapper ..." | tee -a "$LOG"
closeout_log="$ROOT/logs/preprocess/wheat.closeout.log"
closeout_pid_file="$ROOT/logs/preprocess/wheat.closeout.pid"
nohup bash "$ROOT/scripts/preprocess/_wait_wheat_redownload_then_closeout.sh" \
  > "$closeout_log" 2>&1 &
closeout_pid=$!
if ! printf '%s\n' "$closeout_pid" > "$closeout_pid_file"; then
  echo "[$(ts)] ERR: failed to write closeout PID file $closeout_pid_file" | tee -a "$LOG"
  kill "$closeout_pid" 2>/dev/null || true
  exit 9
fi
echo "[$(ts)] closeout PID=$closeout_pid" | tee -a "$LOG"
