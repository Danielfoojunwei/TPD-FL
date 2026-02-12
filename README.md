# TPD-FL: Typed Privacy Diffusion + Federated Learning

A research-grade implementation of **Typed Privacy Diffusion (TPD)** —
typed operational semantics over diffusion language model decoding —
combined with **Federated Learning (FL)** of LoRA adapters.

TPD provides **hard, formally proven guarantees** that sensitive token
classes can never be emitted during diffusion decoding, regardless of
model weights, adapter updates, or adversarial inputs.

## CPU-First Design

This codebase is designed to run **end-to-end on CPU-only machines**.
All proofs, evaluations, and CI pipelines work on CPU by default.

| Tier | Model | Device | Status |
|------|-------|--------|--------|
| **1 (default)** | LLaDA 8B (non-MoE) | CPU | Correctness, proofs, evaluation |
| 2 (optional) | LLaDA2.1-mini 16B MoE | GPU | Scaling experiments |
| CI | Synthetic backend | CPU | Unit tests, no model weights |

CPU dtype auto-detection (`bf16` if AVX-512 BF16 / AMX supported, else `fp32`)
is built in — see `model/backend_base.py:detect_cpu_bf16_support()`.

## Key Contributions

1. **TPD Core**: Span typing, allowed-set projection, schedule-driven
   mask phases, deterministic verifier gate, and monotone repair.
2. **FL Integration**: FedAvg/FedAdam over LoRA adapters with provable
   non-emission guarantees preserved across arbitrary adapter updates.
3. **Formal Proofs**: Type preservation, hard non-emission, edit closure,
   and verifier-lifted global safety (see `tpd_fl/proofs/`).
4. **Evaluation Suite**: 8 baselines (B0-B7), S1-S3 benchmark suites,
   leakage/utility metrics, and publication-ready plots.

## Repository Structure

```
tpd_fl/
  model/                # Model backends (CPU-first)
    backend_base.py     # DiffusionBackend ABC + SyntheticBackend
    backend_hf_llada.py # LLaDA 8B HuggingFace backend (Tier 1, CPU)
    backend_hf_llada2.py# LLaDA2.1-mini HF backend (Tier 2, GPU)
  tpd/                  # TPD core module
    typing.py           # Span typer tau
    allowed_sets.py     # A(type) vocabulary masks
    schedule.py         # Policy-driven mask schedule (draft/safe/reveal)
    projection.py       # Logits projection Pi_{A(tau_i)}
    verifier.py         # Deterministic verifier gate Okpi
    repair.py           # Resample / edit repair
    diagnostics.py      # Z_i allowed-mass measurement
  diffusion/            # Diffusion decode
    decode_loop.py      # M2T decode loop with TPD hooks A-E
  fl/                   # Federated Learning
    lora.py             # LoRA adapters
    step_adapters.py    # Per-diffusion-step adapters
    client.py           # FL client training loop
    server.py           # FL aggregator
    protocols.py        # FedAvg, FedAdam, secure-agg stub
    datasets.py         # Non-IID partitioning + synthetic PII data
  eval/                 # Evaluation suite
    benchgen.py         # S1-S3 benchmark generation
    leakage.py          # Regex leakage metrics
    leakage_semantic.py # Semantic leakage detection
    utility.py          # Exact-match, ROUGE, fluency
    speed.py            # Throughput / timing
    baselines.py        # B0-B7 baseline implementations
    plots.py            # Publication-ready matplotlib figures
    run_eval.py         # Main evaluation runner
  proofs/               # Formal proof package
    tpd_semantics.tex   # Definitions, theorems, proofs (LaTeX)
    mapping.md          # Theorem-to-code mapping
  configs/              # YAML configuration files
    model/*.yaml        # Model backend configs
    decode/*.yaml       # Decode loop configs
    fl/*.yaml           # FL training configs
    eval/*.yaml         # Evaluation configs
  scripts/              # Shell scripts
    run_cpu_sanity.sh   # Quick sanity check on CPU
    run_cpu_eval.sh     # Full CPU evaluation
    run_fl_cpu.sh       # FL training on CPU
    run_all_cpu.sh      # Complete CPU pipeline
  tests/                # Pytest test suite (~210 tests)
```

## Quick Start

### Installation

```bash
# Python 3.10+ required
pip install torch pyyaml matplotlib pytest
# Optional (for real model backends):
pip install transformers accelerate
```

### Run Tests

```bash
pytest tpd_fl/tests/ -q
```

### CPU Sanity Check

```bash
bash tpd_fl/scripts/run_cpu_sanity.sh
```

### Reproduce Results

#### 1. TPD Decode (synthetic backend, CPU)

```bash
python -m tpd_fl.diffusion.decode_loop \
    --steps 64 \
    --output-dir runs/decode_demo \
    --seed 42
```

#### 2. TPD Decode (LLaDA 8B, CPU)

```bash
python -m tpd_fl.diffusion.decode_loop \
    --config tpd_fl/configs/decode/tpd_full.yaml \
    --backend llada8b \
    --device cpu \
    --output-dir runs/decode_llada \
    --seed 42
```

#### 3. Federated Learning (CPU)

```bash
python -m tpd_fl.fl.server \
    --config tpd_fl/configs/fl/fedavg_cpu.yaml \
    --output-dir runs/fl_demo
```

#### 4. Full Evaluation (B0-B5, CPU)

```bash
python -m tpd_fl.eval.run_eval \
    --config tpd_fl/configs/eval/cpu_small.yaml \
    --output-dir runs/eval_full
```

#### 5. Complete CPU Pipeline

```bash
bash tpd_fl/scripts/run_all_cpu.sh
```

Each command writes to `runs/<name>/` with:
- `config.json` — config snapshot
- `metrics.json` — aggregate metrics
- `table.csv` — summary table
- `figures/` — plots (eval only)

## Baselines

| Baseline | Description | Projection | Schedule | Verifier | Repair | FL |
|----------|-------------|:----------:|:--------:|:--------:|:------:|:--:|
| B0 | Unprotected | | | | | |
| B1 | Post-hoc redaction | | | | | |
| B2 | AR logit masking | | | | | |
| B3 | TPD projection only | X | | | | |
| B4 | TPD + schedule | X | X | | | |
| B5 | TPD full | X | X | X | X | |
| B6 | FL only (no TPD) | | | | | X |
| B7 | TPD + FL | X | X | X | X | X |

B6 demonstrates that FL alone does **not** solve output privacy.

## Benchmark Suites

| Suite | Description |
|-------|-------------|
| S1 | PII redaction: text with embedded EMAIL, PHONE, SSN, CC, ID |
| S2 | Adversarial extraction: prompts designed to extract PII |
| S3 | Derived summaries: text where model must summarise without leaking |

## Hard Guarantees

TPD provides **provable** safety properties (see `tpd_fl/proofs/tpd_semantics.tex`):

| Theorem | Guarantee |
|---------|-----------|
| Type Preservation | After projected decode, all tokens satisfy type constraints |
| Hard Non-Emission | P(forbidden token) = 0 after projection — not low, **zero** |
| Edit Closure | Repair operations never introduce forbidden tokens |
| Schedule Compliance | Phase-driven position restrictions are enforced |
| Verifier Safety | Projection + verifier implies full policy satisfaction |
| FL Adapter Safety | Arbitrary adapter weights cannot break non-emission |

The key insight: projection operates on logits **after** model computation,
making it independent of model parameters. Any adapter (including
adversarially trained ones) produces logits that are then projected.

See `tpd_fl/proofs/mapping.md` for a detailed theorem-to-code mapping.

## Testing

The test suite covers ~210 tests:

| Category | File | Tests |
|----------|------|-------|
| Projection guarantee | `test_projection.py` | 40 |
| Schedule enforcement | `test_schedule.py` | 32 |
| Edit closure / repair | `test_edit_closure.py` | 20 |
| Verifier coverage | `test_okpi.py` | 17 |
| FL invariance | `test_fl_invariance.py` | ~20 |
| Property-based | `test_invariants_property.py` | 79 |

```bash
# Run all tests
pytest tpd_fl/tests/ -v

# Run specific test category
pytest tpd_fl/tests/test_projection.py -v
pytest tpd_fl/tests/test_fl_invariance.py -v
```

## Dependencies

Required:
- Python >= 3.10
- PyTorch >= 2.0
- PyYAML >= 6.0
- matplotlib >= 3.7

Optional (for real model backends):
- transformers >= 4.40 (for LLaDA 8B / LLaDA2.1-mini)
- accelerate >= 0.30 (for model loading)

## License

MIT
