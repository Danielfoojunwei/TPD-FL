# Theorem-to-Code Mapping

This document maps each formal theorem in `tpd_semantics.tex` to the
Python code that implements the corresponding guarantee.

## Notation

| Symbol | Code equivalent |
|--------|----------------|
| `tau_i` | `SpanType` enum in `tpd/typing.py` |
| `A(tau_i)` | `allowed_masks[stype]` in `tpd/allowed_sets.py` |
| `Pi_{A(tau_i)}` | `project_logits()` in `tpd/projection.py` |
| `Okpi(y)` | `Verifier.check()` in `tpd/verifier.py` |
| `f_edit` | `RepairEngine.repair()` in `tpd/repair.py` |
| `Z_i` | `compute_allowed_mass()` in `tpd/diagnostics.py` |
| `sigma(tau_i,t)` | `MaskSchedule.apply_schedule()` in `tpd/schedule.py` |
| `Delta_c` | `FLClient.train()` return value in `fl/client.py` |
| `theta_G` | `FLServer._global_state` in `fl/server.py` |
| `DiffusionBackend` | `model/backend_base.py` abstract class |

---

## Theorem 1: Type Preservation

**Statement** (tex §3.1): For every position *i*, if `type(i) = tau_i`
at the start of the decode loop, then `type(i) = tau_i` at every
subsequent step.

**Code**:
- `tpd/typing.py:SpanTyper.type_text()` — assigns types once at
  initialisation.
- `diffusion/decode_loop.py:DiffusionDecodeLoop.run()` — `pos_type`
  is set once (line ~205) and **never mutated** during the loop.

**Test**: `tests/test_schedule.py::TestScheduleTypePreservation` —
verifies pos_type stability across 64 steps.

---

## Theorem 2: Hard Non-Emission Guarantee

**Statement** (tex §3.2): After projection, for every position *i*
with `type(i) in SENSITIVE_TYPES` and every token *v* not in `A(tau_i)`:

    softmax(logits)[i, v] = 0

**Code**:
- `tpd/projection.py:project_logits()` — sets `logits[i, v] = -inf`
  for all forbidden `v`.  This is an in-place operation; the key
  correctness property is that direct indexing (not fancy indexing
  with copies) is used.
- `tpd/projection.py:ProjectionEngine.verify_hard_guarantee()` —
  runtime assertion that checks sampled token IDs against allowed sets.

**Test**: `tests/test_projection.py` — 40 tests including:
- `TestProjectLogits::test_sens_forbidden_tokens_neg_inf`
- `TestProjectLogits::test_sampling_never_produces_forbidden`
- `TestProjectLogits::test_adversarial_logits_blocked`
- `TestProjectionPropertyBased::test_random_logits_forbidden_never_sampled`
  (20 parametrised seeds)
- `TestProjectionPropertyBased::test_softmax_forbidden_probability_zero`
  (10 parametrised seeds)

---

## Theorem 3: Closure Under Editing (Repair Monotonicity)

**Statement** (tex §3.3): If the repair function `f_edit` replaces
token at position *i* with a new token, the new token is drawn from
`A(tau_i)`.  Therefore the hard guarantee is preserved after repair.

**Code**:
- `tpd/repair.py:RepairEngine.repair()` — both RESAMPLE and EDIT
  modes apply `project_logits()` before sampling replacement tokens.
- `tpd/repair.py:RepairEngine._resample()` — masks positions, obtains
  logits, projects, then samples.
- `diffusion/decode_loop.py` (lines ~282-297) — repair is invoked
  only on violating positions, and post-repair tokens are verified.

**Test**: `tests/test_edit_closure.py` — 20 tests verifying:
- Repair never introduces forbidden tokens.
- Repaired text passes verifier checks.
- Multiple repair rounds are monotone.

---

## Theorem 4: Schedule Phase Compliance

**Statement** (tex §3.4): During DRAFT phase only PUB positions are
updated; during SAFE phase all positions may update but SENS/REG are
constrained; during REVEAL only PUB and explicit allowlist.

**Code**:
- `tpd/schedule.py:MaskSchedule.apply_schedule()` — intersects
  proposed update mask with phase-allowed positions.
- `tpd/schedule.py:MaskSchedule.phase()` — returns current phase
  (DRAFT/SAFE/REVEAL) based on normalised progress.

**Test**: `tests/test_schedule.py` — 32 tests including phase
boundary tests and position restriction verification.

---

## Theorem 5: Verifier-Lifted Global Safety

**Statement** (tex §3.5): If the verifier `Okpi` passes on the final
output *y*, then *y* contains no forbidden PII patterns and all
sensitive positions hold tokens from their allowed sets.

**Code**:
- `tpd/verifier.py:Verifier.check()` — runs regex patterns,
  structural checks, and semantic checks.
- `diffusion/decode_loop.py` (lines ~308-309) — final verifier
  check after loop completion.

**Test**: `tests/test_okpi.py` — 17 tests for verifier coverage.

---

## Theorem 6: FL Adapter Independence

**Statement** (tex §4.1): Federated learning adapter updates modify
model parameters `theta` (and therefore logits), but TPD projection
`Pi_{A(tau_i)}` is applied independently of `theta` at decode time.
Therefore:

    forall Delta theta: Pi_{A(tau_i)}(f_theta+Delta(x))[i, v] = -inf
                        for all v not in A(tau_i)

**Code**:
- `fl/lora.py:LoRALinear` — modifies the weight matrix via low-rank
  adaptation: `W' = W + alpha/rank * B @ A`.
- `tpd/projection.py:project_logits()` — applied to logits
  regardless of their source (original model or LoRA-adapted model).
- `diffusion/decode_loop.py` (lines ~258-264) — projection is
  applied after `forward_logits()`, which includes any LoRA effect.

**Test**: `tests/test_fl_invariance.py` — tests including:
- `TestFLProjectionInvariance::test_projection_after_lora_perturbation`
  (10 seeds) — verifies projection blocks forbidden tokens after
  arbitrary LoRA weight perturbation.
- `TestFedAvgPreservesProjection::test_fedavg_adversarial_deltas` —
  verifies projection after aggregating adversarial client deltas.
- `TestFLServerIntegration::test_server_round_preserves_projection` —
  end-to-end FL training + projection check.

---

## Corollary: Z_i Allowed Mass

**Statement** (tex §3.6): For sensitive position *i*, the allowed mass
`Z_i = sum_{v in A(tau_i)} softmax(logits)[i, v]` satisfies
`0 < Z_i <= 1`.  After projection, `Z_i = 1` (all probability mass
is on allowed tokens).

**Code**:
- `tpd/diagnostics.py:compute_allowed_mass()` — computes Z_i per
  position, both before and after projection.
- `tpd/diagnostics.py:compute_z_stats()` — aggregates Z statistics
  by type.
- `diffusion/decode_loop.py` (lines ~250-256) — Z_i is logged at
  each decode step.

**Test**: `tests/test_invariants_property.py` — property-based tests
for Z_i bounds.

---

## Backend Architecture

All model backends implement `DiffusionBackend` (abstract class in
`model/backend_base.py`).  The projection is applied identically
regardless of which backend produces the logits:

| Backend | Model | Device | Code |
|---------|-------|--------|------|
| `SyntheticBackend` | Random logits | CPU | `model/backend_base.py` |
| `HFLLaDABackend` | LLaDA 8B | CPU | `model/backend_hf_llada.py` |
| `HFLLaDA2Backend` | LLaDA2.1-mini | GPU | `model/backend_hf_llada2.py` |

The decode loop (`diffusion/decode_loop.py:DiffusionDecodeLoop`)
accepts any `DiffusionBackend` and applies all TPD hooks identically.

---

## Test Coverage Summary

| Theorem | Test file | # Tests |
|---------|-----------|---------|
| T1 Type Preservation | `test_schedule.py` | 32 |
| T2 Hard Non-Emission | `test_projection.py` | 40 |
| T3 Closure Under Editing | `test_edit_closure.py` | 20 |
| T5 Verifier Safety | `test_okpi.py` | 17 |
| T6 FL Independence | `test_fl_invariance.py` | ~20 |
| Properties | `test_invariants_property.py` | 79 |
| **Total** | | **~210** |
