#!/usr/bin/env bash
# FL CPU Training — runs federated learning simulation on CPU.
# Uses a small synthetic model for CI; pass a real config for full runs.
#
# Usage: bash scripts/run_fl_cpu.sh [--config configs/fl/fedavg_cpu.yaml]
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:-configs/fl/fedavg_cpu.yaml}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="runs/fl_cpu_${TS}"

echo "===== FL CPU Training ====="
echo "Config: ${CONFIG}"
echo "Output: ${OUT}"
echo ""
echo "NOTE: CPU FL runs use a synthetic model by default."
echo "      For real model training, set backend in the config."
echo ""

python -m tpd_fl.fl.server \
    --config "${CONFIG}" \
    --output-dir "${OUT}" \
    --seed 42

echo ""
echo "===== FL Training complete ====="
echo "History:  ${OUT}/history.json"
echo "Config:   ${OUT}/config.json"
