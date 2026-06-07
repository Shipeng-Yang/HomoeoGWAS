#!/usr/bin/env bash
# 等 polygwas-cpu env 装完, 然后启动三个 split 脚本到背景, wrapper 即退出.
set -uo pipefail

ROOT="/mnt/7302share/fast_ysp/U7_GWAS"
cd "$ROOT"

CPU_PID=$(awk -F= '{print $NF}' logs/envs/cpu.pid 2>/dev/null | tr -d ' ')
if [ -z "$CPU_PID" ]; then
  echo "ERR: no CPU env install PID found"; exit 2
fi

echo "[$(date)] waiting for conda env create (PID=$CPU_PID) ..."
# 轮询 PID 是否还活
while kill -0 "$CPU_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date)] env install process exited."

# 验证 polygwas-cpu 真的装出来了
if ! conda env list | awk '{print $1}' | grep -qx polygwas-cpu; then
  echo "ERR: polygwas-cpu env NOT created. last 50 lines of log:" >&2
  tail -50 logs/envs/cpu.log >&2
  exit 3
fi

source /home/yys05/miniconda3/etc/profile.d/conda.sh
conda activate polygwas-cpu

# sanity: 主要工具就位
for t in bcftools tabix plink2 python; do
  if ! command -v $t >/dev/null; then echo "ERR: $t missing in env" >&2; exit 4; fi
done
echo "[$(date)] env OK. launching split scripts."

mkdir -p logs/preprocess

setsid bash -c '
  source /home/yys05/miniconda3/etc/profile.d/conda.sh
  conda activate polygwas-cpu
  cd '"$ROOT"'
  exec bash scripts/preprocess/wheat_split_subgenome.sh
' > logs/preprocess/wheat.run.log 2>&1 &
echo $! > logs/preprocess/wheat.pid

setsid bash -c '
  source /home/yys05/miniconda3/etc/profile.d/conda.sh
  conda activate polygwas-cpu
  cd '"$ROOT"'
  exec bash scripts/preprocess/cotton_split_subgenome.sh
' > logs/preprocess/cotton.run.log 2>&1 &
echo $! > logs/preprocess/cotton.pid

setsid bash -c '
  source /home/yys05/miniconda3/etc/profile.d/conda.sh
  conda activate polygwas-cpu
  cd '"$ROOT"'
  exec bash scripts/preprocess/rapeseed_split_subgenome.sh
' > logs/preprocess/rapeseed.run.log 2>&1 &
echo $! > logs/preprocess/rapeseed.pid

sleep 2
echo "[$(date)] launched:"
for s in wheat cotton rapeseed; do
  pid=$(cat logs/preprocess/$s.pid)
  if kill -0 "$pid" 2>/dev/null; then
    echo "  $s PID=$pid OK"
  else
    echo "  $s PID=$pid DEAD (check log)"
  fi
done
echo "[$(date)] wrapper done."
