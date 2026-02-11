# TPD-FL: Typed Privacy Diffusion + Federated Learning

A research-grade implementation of **Typed Privacy Diffusion (TPD)** —
typed operational semantics over diffusion language model decoding —
combined with **Federated Learning (FL)** of LoRA adapters.

TPD provides **hard, formally proven guarantees** that sensitive token
classes can never be emitted during diffusion decoding, regardless of
model weights, adapter updates, or adversarial inputs.

## Key Contributions

1. **TPD Core**: Span typing, allowed-set projection, schedule-driven
   mask phases, deterministic verifier gate, and monotone repair.
2. **FL Integration**: FedAvg/FedAdam over LoRA adapters with provable
   non-emission guarantees preserved across arbitrary adapter updates.
3. **Formal Proofs**: Type preservation, hard non-emission, edit closure,
   and verifier-lifted global safety (see `tpd_fl/proofs/`).
4. **Evaluation Suite**: 8 baselines, leakage/utility/fluency/efficiency
   metrics, and publication-ready plots.

## Repository Structure

```
tpd_fl/
  tpd/                  # TPD core module
    typing.py           # Span typer tau
    allowed_sets.py     # A(type) vocabulary masks
    schedule.py         # Policy-driven mask schedule (draft/safe/reveal)
    projection.py       # Logits projection
    verifier.py         # Deterministic verifier gate
    repair.py           # Resample / edit repair
    diagnostics.py      # Z_i allowed-mass measurement
  diffusion/            # Diffusion LLM abstraction
    model_adapter.py    # DiffusionModel ABC + synthetic + HF backends
    decode_loop.py      # M2T decode loop with TPD hooks
  fl/                   # Federated Learning
    lora.py             # LoRA adapters
    step_adapters.py    # Per-diffusion-step adapters
    client.py           # FL client training loop
    server.py           # FL aggregator
    protocols.py        # FedAvg, FedAdam, secure-agg stub
    datasets.py         # Non-IID partitioning + synthetic PII data
  eval/                 # Evaluation suite
    leakage.py          # Regex + semantic leakage metrics
    utility.py          # Exact-match, ROUGE, fluency
    speed.py            # Throughput / timing
    baselines.py        # B0-B7 baseline implementations
    plots.py            # Publication-ready matplotlib figures
    run_eval.py         # Main evaluation runner
  proofs/               # Formal proof package
    tpd_semantics.tex   # Definitions, theorems, proofs (LaTeX)
    README.md           # Assumption-code mapping
  configs/              # YAML configuration files
    decode/*.yaml
    fl/*.yaml
    eval/*.yaml
  scripts/              # Shell scripts
  tests/                # Pytest test suite
```

## Quick Start

### Installation

```bash
# Python 3.10+ required
pip install -e "tpd_fl/[dev]"
# or manually:
pip install torch pyyaml matplotlib pytest
```

### Run Tests

```bash
pytest tpd_fl/tests/ -q
```

### Reproduce Results

#### 1. TPD Decode (synthetic backend)

```bash
python -m tpd_fl.diffusion.decode_loop \
    --config tpd_fl/configs/decode/tpd.yaml \
    --output-dir runs/decode_demo \
    --seed 42
```

#### 2. Unprotected Baseline

```bash
python -m tpd_fl.diffusion.decode_loop \
    --config tpd_fl/configs/decode/no_tpd.yaml \
    --output-dir runs/decode_no_tpd \
    --seed 42
```

#### 3. Federated Learning

```bash
python -m tpd_fl.fl.server \
    --config tpd_fl/configs/fl/fedavg.yaml \
    --output-dir runs/fl_demo
```

#### 4. Full Evaluation (all baselines + plots)

```bash
python -m tpd_fl.eval.run_eval \
    --config tpd_fl/configs/eval/main.yaml \
    --output-dir runs/eval_full
```

Each command writes to `runs/<name>/` with:
- `config.json` -- config snapshot
- `logs.jsonl` -- per-step diagnostics
- `metrics.json` -- aggregate metrics
- `figures/` -- plots (eval only)

## Hard Guarantees

TPD provides **provable** safety properties (see `tpd_fl/proofs/tpd_semantics.tex`):

| Theorem | Guarantee |
|---|---|
| Type Preservation | After projected decode, all tokens satisfy type constraints |
| Hard Non-Emission | P(forbidden token) = 0 after projection -- not low, **zero** |
| Edit Closure | Repair operations never introduce forbidden tokens |
| Verifier Safety | Projection + verifier implies full policy satisfaction |
| FL Adapter Safety | Arbitrary adapter weights cannot break non-emission |

The key insight: projection operates on logits **after** model computation,
making it independent of model parameters. Any adapter (including
adversarially trained ones) produces logits that are then projected.

## Testing

The test suite covers:

- **Unit tests**: projection correctness, schedule phase enforcement,
  verifier detection
- **Property tests**: randomised logits with forced forbidden maxima --
  sampling still never produces forbidden tokens
- **FL regression**: malicious adapter perturbations + projection gives
  hard leakage of zero

```bash
pytest tpd_fl/tests/test_projection.py -v
pytest tpd_fl/tests/test_schedule.py -v
pytest tpd_fl/tests/test_edit_closure.py -v
pytest tpd_fl/tests/test_invariants_property.py -v
pytest tpd_fl/tests/test_okpi.py -v
```

## Configuration

All experiments are configured via YAML files in `tpd_fl/configs/`.
See individual files for documented parameters.

## Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- PyYAML >= 6.0
- matplotlib >= 3.7

Optional:
- transformers (for HF model backend)
- accelerate (for HF model backend)

## License

MIT
