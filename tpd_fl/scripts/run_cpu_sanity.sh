#!/usr/bin/env bash
# CPU Sanity Check — runs one prompt in each mode: B0, B3, B4, B5.
# Uses synthetic backend by default (no model download needed).
# To use LLaDA 8B: pass --backend llada8b
#
# Usage: bash scripts/run_cpu_sanity.sh [--backend llada8b]
set -euo pipefail

cd "$(dirname "$0")/.."
BACKEND="${1:-synthetic}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="runs/sanity_${TS}"
TEXT="Please contact John Smith at john.smith@company.com or call (555) 123-4567. His SSN is 123-45-6789."

echo "===== CPU Sanity Check ====="
echo "Backend: ${BACKEND}"
echo "Output:  ${OUT}"
echo ""

for MODE in tpd_off tpd_projection tpd_projection_schedule tpd_full; do
    echo "--- Running ${MODE} ---"
    python -m tpd_fl.diffusion.decode_loop \
        --config "configs/decode/${MODE}.yaml" \
        --backend "${BACKEND}" \
        --device cpu \
        --output-dir "${OUT}/${MODE}" \
        --seed 42 2>&1 | head -20
    echo ""
done

echo "===== Sanity check complete. Results in ${OUT} ====="
