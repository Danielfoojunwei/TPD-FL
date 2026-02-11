#!/usr/bin/env bash
# Run full evaluation suite.
# Usage: bash scripts/run_eval.sh [config_path]
set -euo pipefail

CONFIG="${1:-configs/eval/main.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="runs/eval_${TIMESTAMP}"

echo "=== TPD+FL Evaluation Suite ==="
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

cd "$(dirname "$0")/.."

python -m tpd_fl.eval.run_eval \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}"

echo "=== Done. Results in ${OUTPUT_DIR} ==="
echo "Figures: ${OUTPUT_DIR}/figures/"
echo "Metrics: ${OUTPUT_DIR}/metrics.json"
