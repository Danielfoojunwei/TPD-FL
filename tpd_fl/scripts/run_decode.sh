#!/usr/bin/env bash
# Run TPD diffusion decode loop.
# Usage: bash scripts/run_decode.sh [config_path]
set -euo pipefail

CONFIG="${1:-configs/decode/tpd.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="runs/decode_${TIMESTAMP}"

echo "=== TPD Diffusion Decode ==="
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

cd "$(dirname "$0")/.."

python -m tpd_fl.diffusion.decode_loop \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed 42

echo "=== Done. Artifacts in ${OUTPUT_DIR} ==="
