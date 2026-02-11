#!/usr/bin/env bash
# Run FL training with TPD.
# Usage: bash scripts/run_fl.sh [config_path]
set -euo pipefail

CONFIG="${1:-configs/fl/fedavg.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="runs/fl_${TIMESTAMP}"

echo "=== Federated Learning with TPD ==="
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

cd "$(dirname "$0")/.."

python -m tpd_fl.fl.server \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}"

echo "=== Done. Artifacts in ${OUTPUT_DIR} ==="
