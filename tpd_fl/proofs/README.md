# Formal Proofs Package — TPD Semantics

This directory contains the formal proof document for Typed Privacy
Diffusion (TPD).

## Files

- `tpd_semantics.tex` — Full LaTeX document with definitions, transition
  rules, theorems, and proofs.

## Assumptions

The proofs rest on the following assumptions, each of which maps to a
testable code invariant:

| Assumption | Code invariant | Test |
|---|---|---|
| A1: Projection sets forbidden logits to −∞ | `project_logits()` uses `masked_fill` with `finfo.min` | `test_projection.py::test_sens_forbidden_tokens_neg_inf` |
| A2: `softmax(−∞) = 0` exactly | IEEE 754 float underflow | `test_projection.py::test_softmax_forbidden_probability_zero` |
| A3: `multinomial(0) = impossible` | PyTorch `multinomial` never samples zero-probability events | `test_projection.py::test_sampling_never_produces_forbidden` |
| A4: Schedule blocks sensitive positions in DRAFT | `apply_schedule()` intersects with allowed positions | `test_schedule.py::test_draft_never_updates_sensitive` |
| A5: Repair is monotone in sensitivity | Repair uses projection; never introduces forbidden tokens | `test_edit_closure.py::test_resample_repair_preserves_types` |
| A6: Projection is independent of model parameters | Projection operates on logits post-computation | `test_invariants_property.py::test_malicious_adapter_perturbation` |

## Theorem–Code Correspondence

| Theorem | Core guarantee | Primary code path |
|---|---|---|
| Thm 1: Type Preservation | After projected M2T, all tokens satisfy type constraints | `projection.py:project_logits` → `model_adapter.py:sample_tokens` |
| Thm 2: Hard Non-Emission | P(forbidden token) = 0 after projection | `projection.py:project_logits` sets logits to −∞ |
| Thm 3: Closure Under Editing | T2T edits preserve type constraints | `repair.py:RepairEngine` applies projection on edit |
| Thm 4: Verifier-Lifted Safety | If Thm 1 + Okπ pass, output satisfies π | `verifier.py:Verifier.check` + `decode_loop.py` |
| Cor: FL Adapter Safety | Arbitrary adapter params cannot break non-emission | Projection applied after logit computation regardless of adapters |

## Building the PDF

```bash
cd proofs/
pdflatex tpd_semantics.tex
pdflatex tpd_semantics.tex   # second pass for references
```
