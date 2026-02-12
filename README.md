# TPD-FL: Typed Privacy Diffusion + Federated Learning

A research-grade implementation of **Typed Privacy Diffusion (TPD)** —
typed operational semantics over diffusion language model decoding —
combined with **Federated Learning (FL)** of LoRA adapters.

TPD provides **hard, formally proven guarantees** that sensitive token
classes can never be emitted during diffusion decoding, regardless of
model weights, adapter updates, or adversarial inputs.

## Empirical Results

Evaluated on **bert-base-uncased** (110M params, real open weights) using
the Carlini et al. (2021) context-completion extraction setup: 20 samples
across three benchmark suites (S1: PII redaction, S2: adversarial extraction,
S3: derived summaries).

| Baseline | Forbidden% | Forbid | Leak% | R-1 | R-2 | R-L | BLEU | D-1 | D-2 | s/samp | Hard% |
|----------|-----------|--------|-------|-----|-----|-----|------|-----|-----|--------|-------|
| B0 (Unprotected) | **17.5%** | 48/275 | 0.0% | 0.473 | 0.339 | 0.471 | 0.205 | 0.88 | 0.99 | 3.57s | 100% |
| B1 (Post-hoc) | **17.5%** | 48/275 | 0.0% | 0.473 | 0.339 | 0.471 | 0.205 | 0.88 | 0.99 | 3.54s | 100% |
| B3 (TPD proj) | **0.0%** | 0/275 | 0.0% | 0.475 | 0.334 | 0.473 | 0.201 | 0.89 | 0.99 | 3.58s | 100% |
| B4 (proj+sched) | **0.0%** | 0/275 | 0.0% | 0.480 | 0.336 | 0.475 | 0.199 | 0.90 | 0.99 | 1.74s | 100% |
| B5 (Full TPD) | **0.0%** | 0/275 | 0.0% | 0.480 | 0.336 | 0.475 | 0.199 | 0.90 | 0.99 | 1.77s | 100% |

**Key findings:**

- **Forbidden%** = fraction of tokens at sensitive positions from the blocked
  set (digits, `@`, PII-indicative patterns). This is the core privacy metric.
- B0/B1 emit **17.5%** forbidden tokens at sensitive positions — the model
  freely predicts PII-shaped content when given surrounding context.
- B1 (post-hoc regex redaction) provides **no improvement** at the token level;
  regex scrubbing operates on text, not logits.
- B3-B5 achieve **0.0%** forbidden tokens — the projection hard guarantee
  (Theorem 2) holds empirically across all 275 sensitive positions.
- **Utility is preserved**: ROUGE-1 ~0.47-0.48 across all baselines;
  projection does not degrade text quality.
- B4 is **2x faster** than B0/B3 due to schedule-driven DRAFT-phase skipping.

Per-suite forbidden token rates:

| Baseline | S1 (PII redaction) | S2 (Adversarial) | S3 (Summaries) |
|----------|--------------------|-------------------|----------------|
| B0 | 21.0% | 17.8% | 20.8% |
| B1 | 21.0% | 17.8% | 20.8% |
| B3 | 0.0% | 0.0% | 0.0% |
| B4 | 0.0% | 0.0% | 0.0% |
| B5 | 0.0% | 0.0% | 0.0% |

### Reproduce

```bash
pip install transformers nltk
python -m tpd_fl.eval.empirical_eval --output-dir runs/empirical --steps 64 --seed 42
```

Results are saved to `runs/empirical/` with `metrics.json`, `table.csv`,
and `per_sample_results.json`.

## CPU-First Design

This codebase is designed to run **end-to-end on CPU-only machines**.
All proofs, evaluations, and CI pipelines work on CPU by default.

| Tier | Model | Device | Status |
|------|-------|--------|--------|
| **1 (default)** | BERT-base-uncased (110M) | CPU | Empirical evaluation |
| 2 (optional) | LLaDA 8B (non-MoE) | CPU | Full diffusion decode |
| 3 (optional) | LLaDA2.1-mini 16B MoE | GPU | Scaling experiments |
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
4. **Empirical Evaluation**: Real model (bert-base-uncased, 110M params),
   canonical metrics (ROUGE, BLEU, forbidden token rate), Carlini et al.
   (2021) extraction setup, 20-sample benchmark across S1-S3 suites.

## Repository Structure

```
tpd_fl/
  model/                  # Model backends (CPU-first)
    backend_base.py       # DiffusionBackend ABC + SyntheticBackend
    backend_hf_bert.py    # BERT-base-uncased MLM backend (empirical eval)
    backend_hf_llada.py   # LLaDA 8B HuggingFace backend (Tier 2, CPU)
    backend_hf_llada2.py  # LLaDA2.1-mini HF backend (Tier 3, GPU)
  tpd/                    # TPD core module
    typing.py             # Span typer tau
    allowed_sets.py       # A(type) vocabulary masks
    schedule.py           # Policy-driven mask schedule (draft/safe/reveal)
    projection.py         # Logits projection Pi_{A(tau_i)}
    verifier.py           # Deterministic verifier gate Okpi
    repair.py             # Resample / edit repair
    diagnostics.py        # Z_i allowed-mass measurement
  diffusion/              # Diffusion decode
    decode_loop.py        # M2T decode loop with TPD hooks A-E
  fl/                     # Federated Learning
    lora.py               # LoRA adapters
    step_adapters.py      # Per-diffusion-step adapters
    client.py             # FL client training loop
    server.py             # FL aggregator
    protocols.py          # FedAvg, FedAdam, secure-agg stub
    datasets.py           # Non-IID partitioning + synthetic PII data
  eval/                   # Evaluation suite
    empirical_eval.py     # Real-model empirical evaluation (BERT-base)
    metrics_real.py       # Canonical NLP metrics (ROUGE, BLEU, PII detection)
    benchgen.py           # S1-S3 benchmark generation
    leakage.py            # Regex leakage metrics
    leakage_semantic.py   # Semantic leakage detection
    utility.py            # Exact-match, ROUGE, fluency
    speed.py              # Throughput / timing
    baselines.py          # B0-B7 baseline implementations
    plots.py              # Publication-ready matplotlib figures
    run_eval.py           # Main evaluation runner
  proofs/                 # Formal proof package
    tpd_semantics.tex     # Definitions, theorems, proofs (LaTeX)
    mapping.md            # Theorem-to-code mapping
  configs/                # YAML configuration files
    model/*.yaml          # Model backend configs
    decode/*.yaml         # Decode loop configs
    fl/*.yaml             # FL training configs
    eval/*.yaml           # Evaluation configs
  scripts/                # Shell scripts
    run_cpu_sanity.sh     # Quick sanity check on CPU
    run_cpu_eval.sh       # Full CPU evaluation
    run_fl_cpu.sh         # FL training on CPU
    run_all_cpu.sh        # Complete CPU pipeline
  tests/                  # Pytest test suite (~210 tests)
```

## Quick Start

### Installation

```bash
# Python 3.10+ required
pip install torch pyyaml matplotlib pytest

# For empirical evaluation (real model):
pip install transformers nltk

# Optional (for LLaDA backends):
pip install accelerate
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

#### 4. Empirical Evaluation (BERT-base, CPU)

```bash
python -m tpd_fl.eval.empirical_eval \
    --output-dir runs/empirical \
    --steps 64 \
    --seed 42
```

Downloads bert-base-uncased (110M params) and runs B0-B5 baselines with
canonical metrics (ROUGE, BLEU, forbidden token rate).

#### 5. Synthetic Evaluation (B0-B5, CPU)

```bash
python -m tpd_fl.eval.run_eval \
    --config tpd_fl/configs/eval/cpu_small.yaml \
    --output-dir runs/eval_full
```

#### 6. Complete CPU Pipeline

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

For empirical evaluation:
- transformers >= 4.40 (for BERT-base-uncased / LLaDA backends)
- nltk >= 3.8 (for ROUGE / BLEU metrics)

Optional:
- accelerate >= 0.30 (for large model loading)

## License

MIT
