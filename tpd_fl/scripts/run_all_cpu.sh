#!/usr/bin/env bash
# Full CPU Pipeline — sanity check + evaluation + FL training.
# Runs everything on CPU with synthetic backends for CI validation.
#
# Usage: bash scripts/run_all_cpu.sh
set -euo pipefail

cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M%S)
BASE_OUT="runs/all_cpu_${TS}"
mkdir -p "${BASE_OUT}"

echo "============================================"
echo "  TPD+FL Full CPU Pipeline"
echo "  Timestamp: ${TS}"
echo "  Output:    ${BASE_OUT}/"
echo "============================================"
echo ""

# ── Step 1: Sanity checks ──
echo "═══ Step 1/4: Sanity Checks ═══"
echo ""

echo "→ Running unit tests..."
python -m pytest tpd_fl/tests/ -x -q --tb=short 2>&1 | tee "${BASE_OUT}/test_output.txt"
echo ""

echo "→ Running decode loop sanity check..."
python -m tpd_fl.diffusion.decode_loop \
    --steps 16 \
    --output-dir "${BASE_OUT}/sanity_decode" \
    --seed 42
echo ""

# ── Step 2: Benchmark generation ──
echo "═══ Step 2/4: Benchmark Generation ═══"
echo ""

python -c "
from tpd_fl.eval.benchgen import BenchmarkGenerator
import os, json

gen = BenchmarkGenerator()
suites = gen.generate_all(num_s1=20, num_s2=10, num_s3=10, seed=42)
out = '${BASE_OUT}/benchmarks'
os.makedirs(out, exist_ok=True)
for name, samples in suites.items():
    gen.save_jsonl(samples, f'{out}/{name.lower()}.jsonl')
    print(f'  {name}: {len(samples)} samples saved')

total = sum(len(s) for s in suites.values())
print(f'  Total: {total} benchmark samples')
"
echo ""

# ── Step 3: Evaluation ──
echo "═══ Step 3/4: Evaluation (B0-B5) ═══"
echo ""

python -m tpd_fl.eval.run_eval \
    --output-dir "${BASE_OUT}/eval" \
    --experiment-name "cpu_eval" \
    --baselines B0 B1 B2 B3 B4 B5 \
    --num-s1 10 --num-s2 5 --num-s3 5 \
    --steps 32 \
    --seed 42 \
    --no-plots
echo ""

# ── Step 4: FL Training ──
echo "═══ Step 4/4: FL Training ═══"
echo ""

python -m tpd_fl.fl.server \
    --output-dir "${BASE_OUT}/fl" \
    --seed 42
echo ""

# ── Summary ──
echo "============================================"
echo "  All CPU pipeline steps complete!"
echo "  Results: ${BASE_OUT}/"
echo ""
echo "  Artifacts:"
echo "    Tests:      ${BASE_OUT}/test_output.txt"
echo "    Benchmarks: ${BASE_OUT}/benchmarks/"
echo "    Eval:       ${BASE_OUT}/eval/"
echo "    FL:         ${BASE_OUT}/fl/"
echo "============================================"
