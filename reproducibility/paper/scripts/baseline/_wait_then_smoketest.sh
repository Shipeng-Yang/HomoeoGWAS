#!/usr/bin/env bash
# Wait for pip + gemma/regenie to finish installing, then run the Horvath2020 smoke test.
set -uo pipefail

ROOT=/mnt/7302share/fast_ysp/U7_GWAS
ENV=$HOME/.local/share/mamba/envs/polygwas-cpu

echo "[$(date)] waiting for pip + gemma/regenie install ..."

# wait for pip to install matplotlib
while ! "$ENV/bin/python" -c 'import matplotlib' 2>/dev/null; do sleep 5; done
echo "[$(date)] pip matplotlib OK"

# wait for conda to install gemma + regenie
while [ ! -x "$ENV/bin/gemma" ] || [ ! -x "$ENV/bin/regenie" ]; do sleep 5; done
echo "[$(date)] gemma + regenie OK"

# actual verification
"$ENV/bin/gemma" 2>&1 | head -1 || true
"$ENV/bin/regenie" --version 2>&1 | head -1 || true
"$ENV/bin/python" -c 'import matplotlib, scipy, sklearn, seaborn, statsmodels; print("py pkgs OK")'

# run Step E
echo "[$(date)] launching Step E smoke test ..."
bash "$ROOT/scripts/baseline/run_horvath_smoketest.sh"
echo "[$(date)] Step E wrapper done."
