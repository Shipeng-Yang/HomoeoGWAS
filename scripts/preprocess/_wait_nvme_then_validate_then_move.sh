#!/usr/bin/env bash
# 等 aria2 下载到 /mnt/nvme 完成 → bgzip -t 在本地 SSD 验证完整性
# → 通过的 mv 到 NFS data/raw/wheat/vcf/  → 重建 tabix → 触发 closeout wrapper.
# 设计目的: 绕过 NFS 大文件写入字节级 race.
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

# 验证 + 移到 NFS
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
  # 大文件 mv across mount = cp + unlink. 用 cp 然后 unlink 显式控制
  cp "$f" "$dst"
  # 写到 NFS 后立即再 bgzip -t 验证
  if ! bgzip -t "$dst" 2>>"$LOG"; then
    echo "[$(ts)] ERR: NFS copy of $short corrupted during write — NFS write race confirmed" | tee -a "$LOG"
    exit 7
  fi
  if ! tabix -f --csi -p vcf "$dst" 2>>"$LOG"; then
    echo "[$(ts)] ERR: tabix failed for $short on NFS" | tee -a "$LOG"
    exit 8
  fi
  echo "[$(ts)] OK $short on NFS + tabix" | tee -a "$LOG"
  # 成功才删 NVMe 副本(保险)
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
