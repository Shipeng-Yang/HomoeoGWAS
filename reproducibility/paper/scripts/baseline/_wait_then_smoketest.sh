#!/usr/bin/env bash
# 等 pip + gemma/regenie 装好, 然后跑 Horvath2020 烟雾测试.
set -uo pipefail

ROOT=/mnt/7302share/fast_ysp/U7_GWAS
ENV=$HOME/.local/share/mamba/envs/polygwas-cpu

echo "[$(date)] waiting for pip + gemma/regenie install ..."

# 等 pip 装好(matplotlib)
while ! "$ENV/bin/python" -c 'import matplotlib' 2>/dev/null; do sleep 5; done
echo "[$(date)] pip matplotlib OK"

# 等 conda 装好 gemma + regenie
while [ ! -x "$ENV/bin/gemma" ] || [ ! -x "$ENV/bin/regenie" ]; do sleep 5; done
echo "[$(date)] gemma + regenie OK"

# 实际验证
"$ENV/bin/gemma" 2>&1 | head -1 || true
"$ENV/bin/regenie" --version 2>&1 | head -1 || true
"$ENV/bin/python" -c 'import matplotlib, scipy, sklearn, seaborn, statsmodels; print("py pkgs OK")'

# 跑 Step E
echo "[$(date)] launching Step E smoke test ..."
bash "$ROOT/scripts/baseline/run_horvath_smoketest.sh"
echo "[$(date)] Step E wrapper done."
