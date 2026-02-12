#!/usr/bin/env bash
# CPU Evaluation — generates S1-S3 benchmarks, runs baselines B0-B5.
# Produces metrics.json + figures + table in runs/eval_cpu_<timestamp>/
#
# Usage: bash scripts/run_cpu_eval.sh [--config configs/eval/cpu_small.yaml]
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:-configs/eval/cpu_small.yaml}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="runs/eval_cpu_${TS}"

echo "===== CPU Evaluation ====="
echo "Config: ${CONFIG}"
echo "Output: ${OUT}"
echo ""
echo "NOTE: CPU runs may take several minutes depending on sample count."
echo ""

python -m tpd_fl.eval.run_eval \
    --config "${CONFIG}" \
    --output-dir "${OUT}"

echo ""
echo "===== Evaluation complete ====="
echo "Results:  ${OUT}/metrics.json"
echo "Figures:  ${OUT}/figures/"
echo "Table:    ${OUT}/table.csv"
