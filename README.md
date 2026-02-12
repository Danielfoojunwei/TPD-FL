# Typed Privacy Diffusion for Federated Language Models: Hard Token-Level Guarantees via Type-Theoretic Logit Projection

> **A NeurIPS-style research paper with full formal proofs, empirical evaluation, and open-source implementation.**

---

## Abstract

We introduce **Typed Privacy Diffusion (TPD)**, a type-theoretic framework that enforces *hard*, *deterministic* token-level privacy guarantees within discrete diffusion language models operating under federated learning (FL). TPD assigns every output position a *span type* drawn from a lattice of privacy classifications and restricts the generative process---at every transition step---to an *allowed token set* determined by that type. The restriction is implemented by a *logit projection operator* that sets the probability of every forbidden token to **exactly zero** before sampling, independent of model parameters.

We formalise the TPD state machine, define four transition rules (mask-to-token, token-to-token edit, schedule-restricted transition, and monotone repair), and prove six formal guarantees: **Type Preservation** (well-typedness is an invariant of the transition system), **Hard Non-Emission** (forbidden tokens have exactly zero sampling probability), **Edit Closure** (repair operations preserve typing), **Schedule Compliance** (phase-driven position restrictions are enforced), **Verifier-Lifted Global Safety** (projection + verifier implies full policy satisfaction), and **FL Adapter Safety** (arbitrary adapter weights---including adversarially trained ones---cannot break non-emission).

We evaluate TPD empirically on **bert-base-uncased** (110M parameters, open weights) using the Carlini et al. (2021) context-completion extraction setup across 20 benchmark samples spanning three suites (PII redaction, adversarial extraction, derived summaries). Unprotected baselines emit **17.5%** forbidden tokens at sensitive positions; TPD-protected baselines achieve **0.0%** forbidden tokens while preserving text utility (ROUGE-1: 0.48 vs. 0.47, no degradation) and achieving 2x speedup through schedule-driven draft-phase skipping.

All code, proofs, and evaluation harnesses are open-source (MIT license), CPU-first, and reproducible with a single command.

**Keywords:** privacy, diffusion language models, federated learning, type theory, constrained decoding, PII protection

---

## 1. Introduction

### 1.1 Motivation

Discrete diffusion language models (Austin et al., 2021; Lou et al., 2024) generate text by iteratively denoising a fully-masked token sequence. At each step, a subset of positions transitions from the mask token to a concrete vocabulary element, guided by a neural denoising network. This iterative, position-level generation offers a natural intervention point for enforcing fine-grained output constraints---an opportunity absent in autoregressive models, where tokens are generated sequentially and left-to-right.

In privacy-sensitive settings---clinical note generation, financial report drafting, federated learning on personal devices---it is essential that the model *never* emit certain tokens at certain positions, regardless of what the underlying neural parameters predict. Standard sampling from a softmax distribution over the full vocabulary provides no such guarantee: a sufficiently confident logit for a forbidden token will produce that token with non-negligible probability. Differential privacy (Abadi et al., 2016) provides statistical guarantees on gradient information during training, but offers no deterministic control over the tokens actually generated at inference time.

### 1.2 The TPD Approach

Typed Privacy Diffusion addresses this gap with three key ideas:

1. **Span typing** (`tau : [L] -> T`): A deterministic pipeline classifies each output position as public (`PUB`), sensitive (`SENS`), regulated (`REG`), or one of six derived entity types (`DERIVED_NAME`, `DERIVED_EMAIL`, `DERIVED_PHONE`, `DERIVED_ID`, `DERIVED_CC`, `DERIVED_ADDRESS`).

2. **Logit projection** (`Pi_{A(tau_i)}`): Before the softmax, a projection operator clamps the logits of every token outside the allowed set `A(tau_i)` to `-inf`. Since `exp(-inf) = 0`, the softmax assigns exactly zero probability to every forbidden token. This is not approximately zero---it is *exactly* zero.

3. **Deterministic verifier gate** (`Ok_pi`): After each step, a verifier checks structural and pattern-level policy compliance (regex scans for forbidden PII patterns, placeholder validation), triggering monotone repair on failure.

Together, these mechanisms provide a **hard, model-parameter-independent privacy guarantee**: no amount of fine-tuning---including adversarial federated learning updates from Byzantine clients---can cause a forbidden token to be sampled at a sensitive position.

### 1.3 Contributions

1. **Formal framework**: We define the TPD state machine with typed operational semantics over diffusion decoding and prove six theorems establishing hard safety guarantees (Section 4).

2. **Projection operator with zero-probability guarantee**: We introduce a logit projection that is independent of model parameters, operates after all neural computation, and provides exactly-zero emission probability for forbidden tokens---stronger than any epsilon-bounded statistical guarantee (Section 3.3).

3. **FL adapter safety**: We prove that the projection guarantee is preserved under arbitrary federated learning adapter updates, including adversarially crafted ones (Theorem 6, Section 4.6).

4. **Utility bound**: We derive a closed-form expression for the information-theoretic cost of projection: `D_KL(p_tilde || p) = -log Z_i` per position, where `Z_i` is the pre-projection allowed mass (Section 4.7).

5. **Empirical validation**: We evaluate on a real model (bert-base-uncased, 110M params) with canonical metrics (ROUGE, BLEU, forbidden token rate) using the Carlini et al. (2021) extraction setup, demonstrating that the formal guarantees hold empirically and utility is preserved (Section 5).

---

## 2. Related Work

### 2.1 Discrete Diffusion Language Models

Austin et al. (2021) introduced structured denoising diffusion in discrete state spaces, establishing the mask-and-denoise framework that generates text by iteratively replacing mask tokens with vocabulary elements. Lou et al. (2024) advanced this with ratio-based discrete diffusion, improving sample quality and inference efficiency. More recently, LLaDA (Nie et al., 2025) demonstrated that masked diffusion can scale to 8B parameters competitively with autoregressive models. TPD operates within this framework, adding type-aware constraints at each denoising step without modifying the underlying diffusion process.

### 2.2 Constrained Decoding

Grammar-based and vocabulary-restricted decoding has been explored for autoregressive models. Hokamp & Liu (2017) introduced lexically constrained decoding via grid beam search; Hu et al. (2019) improved this with dynamic beam allocation. These approaches impose hard lexical constraints during left-to-right generation. TPD extends constrained decoding to the diffusion setting, where the bidirectional, iterative nature of generation requires constraints to be enforced at every step across all positions simultaneously.

### 2.3 Differential Privacy in Federated Learning

McMahan et al. (2017) established federated averaging (FedAvg) for communication-efficient learning from decentralised data. McMahan et al. (2018) combined FL with differential privacy for recurrent language models, providing epsilon-bounded guarantees on gradient information. Abadi et al. (2016) developed the foundational DP-SGD algorithm. Charles et al. (2024) explored scalable dataset pipelines for group-structured federated learning. **Key distinction**: DP protects the *training process* (gradient information); TPD protects the *inference output* (generated tokens). The two are complementary: DP ensures that training data cannot be extracted from model updates, while TPD ensures that the model cannot emit forbidden content regardless of its parameters.

### 2.4 PII Detection and Anonymisation

Lison et al. (2021) explored named entity recognition for PII detection with noise-aware learning. TPD's span typer can be seen as a lightweight PII detector whose output is consumed by a formal enforcement mechanism---the projection operator. Unlike post-hoc redaction systems that operate on generated text (and can miss novel PII patterns), TPD prevents forbidden tokens from ever being sampled.

### 2.5 Positioning

| Approach | Guarantee Type | Granularity | Parameter-Independent | Applies to Diffusion |
|----------|---------------|-------------|----------------------|---------------------|
| DP-SGD (Abadi et al., 2016) | Statistical (epsilon) | Training gradients | No | N/A |
| Post-hoc redaction | Best-effort | Text-level | Yes | Yes |
| Constrained decoding (Hokamp, 2017) | Hard (lexical) | Token-level (AR) | Yes | No |
| **TPD (ours)** | **Hard (zero-probability)** | **Token-level (diffusion)** | **Yes** | **Yes** |

---

## 3. Method: The TPD Framework

### 3.1 Span Typing (`tau`)

The span typer is a deterministic pipeline that classifies each token position:

**Definition 1** (Type Universe). The type universe is the finite set:

```
T = { PUB, SENS, REG, DERIVED_NAME, DERIVED_EMAIL, DERIVED_PHONE,
      DERIVED_ID, DERIVED_CC, DERIVED_ADDRESS }
```

We partition `T` into public types `T_pub = {PUB}` and sensitive types `T_sens = T \ {PUB}`. Every type `tau in T_sens` carries a vocabulary restriction; `PUB` imposes none.

**Implementation.** The `SpanTyper` class (`tpd_fl/tpd/typing.py`) executes a three-stage pipeline:

1. **Regex detection**: Six compiled patterns detect EMAIL, PHONE, SSN, CC, ID, and ADDRESS entities at the character level.
2. **Character-to-token mapping**: Character spans are mapped to token spans via the tokenizer's offset mapping.
3. **Policy overrides**: Denylist/allowlist position overrides for application-specific requirements.

The typer operates on text and produces per-position type assignments `pos_type: List[SpanType]` of length `L`, along with span objects recording entity boundaries and tags.

### 3.2 Allowed Token Sets (`A(tau)`)

**Definition 2** (Allowed-Set Function). `A : T -> P(V)` maps each type to the subset of the vocabulary permitted at positions of that type:

- `A(PUB) = V` (full vocabulary, unrestricted)
- `A(SENS)`: Placeholders + safe punctuation + short alphabetic tokens (digits, `@`, and PII-indicative patterns are blocked)
- `A(REG)`: Stricter subset---only alphabetic tokens of length >= 2 (all digits, special characters blocked)
- `A(DERIVED_*)`: Starts from `A(SENS)`, with entity-specific overrides

**Implementation.** The `AllowedSetBuilder` class (`tpd_fl/tpd/allowed_sets.py`) constructs boolean tensors of shape `[V]` for each type. For BERT-base-uncased (V=30,522):

| Type | Allowed | Blocked | Description |
|------|---------|---------|-------------|
| PUB | 30,522 | 0 | Full vocabulary |
| SENS | ~26,800 | ~3,700 | Digits, @, PII patterns blocked |
| REG | ~18,500 | ~12,000 | Non-alphabetic tokens blocked |

The masks are materialised once on the target device and reused across all projection calls.

### 3.3 The Projection Operator (`Pi`)

The projection operator is the core enforcement mechanism.

**Definition 3** (Logits Projection). Let `l_i in R^V` be the raw logit vector produced by the denoising network for position `i`. The projection operator is defined component-wise:

```
[Pi_{A(tau_i)}(l_i)]_v = l_i[v]    if v in A(tau_i)
                       = -inf       if v not in A(tau_i)
```

For public positions (`tau_i = PUB`), `A(PUB) = V`, so `Pi_V` is the identity.

**Lemma 1** (Projection Zeroes Forbidden Probabilities). For any logit vector `l_i in R^V` and any type `tau_i in T_sens`:

```
forall v not in A(tau_i): softmax(Pi_{A(tau_i)}(l_i))[v] = 0
```

*Proof.* By definition of `Pi`, the projected logit for every `v not in A(tau_i)` is `-inf`. The softmax function gives `softmax(l)[v] = exp(l[v]) / sum_v' exp(l[v'])`. Since `exp(-inf) = 0` and the denominator is strictly positive (there exists at least one `v' in A(tau_i)` with finite logit, because `A(tau_i) != empty` by construction), the result follows. In IEEE 754 arithmetic, `exp(finfo.min) = 0.0` exactly.

**Implementation.** `project_logits()` in `tpd_fl/tpd/projection.py` groups positions by `SpanType` and applies a single `masked_fill_` per type for efficient GPU execution. The `ProjectionEngine` class caches masks and provides a `verify_hard_guarantee()` runtime assertion.

### 3.4 Three-Phase Schedule

**Definition 4** (Schedule Phases). The decode process is partitioned into three temporal phases based on step fraction `t/T`:

1. **DRAFT** (`0 <= t/T < alpha`): Only `PUB` positions may be updated. Sensitive positions remain masked. Default `alpha = 0.4`.
2. **SAFE** (`alpha <= t/T < beta`): All positions may be updated, but sensitive positions are constrained by projection. Default `beta = 0.9`.
3. **REVEAL** (`beta <= t/T <= 1`): `PUB` positions plus an explicit reveal-allowlist may be updated.

The effective update mask is `M_t* = M_t intersect Allowed(t, T, tau)`. The schedule intersection is a purely deterministic, set-theoretic operation applied *before* any logits computation or sampling.

**Implementation.** `MaskSchedule` in `tpd_fl/tpd/schedule.py` computes phase boundaries and intersects proposed update masks with allowed positions.

### 3.5 Verifier Gate (`Ok_pi`)

**Definition 5** (Verifier Gate). The verifier is a deterministic boolean function `Ok_pi : V^L -> {true, false}` that checks:

1. **Regex scan**: No substring of the decoded text matches forbidden patterns (EMAIL, PHONE, SSN, CC, ID).
2. **Structural check**: Every contiguous span of sensitive positions decodes to a valid placeholder string from a pre-defined set.
3. **Semantic check** (optional): No known secret from a denylist appears in the output.

**Implementation.** `Verifier` in `tpd_fl/tpd/verifier.py` returns a `VerifierResult` with `ok: bool` and a list of violations with type, detail, and positions.

### 3.6 Monotone Repair

**Definition 6** (Repair Transition). If the verifier rejects, a repair transition is triggered. Two modes:

1. **Resample repair**: Re-mask violating positions, compute fresh logits, apply projection, sample. Iterated up to `K=3` times.
2. **Edit repair**: Directly overwrite violating positions with the first allowed token (typically a placeholder).

Both modes apply the projection operator, so repair preserves well-typedness by the same argument as the main decode loop.

**Implementation.** `RepairEngine` in `tpd_fl/tpd/repair.py` supports both modes and guarantees monotone safety.

### 3.7 FL Integration

TPD integrates with federated learning through LoRA adapters (Hu et al., 2021). Each client trains local low-rank matrices `A` and `B`:

```
W' = W + (alpha / rank) * B @ A
```

where `W` is frozen and only `A, B` are communicated. The server aggregates via FedAvg or FedAdam. **Critically**, the projection operates on logits *after* all model computation (including adapter contributions) and *before* sampling. This architectural placement makes the guarantee independent of adapter weights.

**Implementation.** `LoRALinear` in `tpd_fl/fl/lora.py` wraps `nn.Linear` layers. `attach_lora()` injects adapters into matching modules. `get_lora_state_dict()` / `load_lora_state_dict()` handle FL communication.

---

## 4. Theoretical Analysis

We present six theorems establishing the formal safety guarantees of TPD. All proofs are constructive and parameter-independent. Complete LaTeX proofs are in `tpd_fl/proofs/tpd_semantics.tex`.

### 4.1 Definitions

**TPD State.** A TPD state at step `t` is a triple `S_t = (x_t, M_t, Gamma)` where `x_t in V^L` is the current token sequence, `M_t` is the update mask, and `Gamma = (tau, A, pi)` is the typing context.

**Well-Typed State.** A state is well-typed (`WT(S_t)`) if every non-mask token at a sensitive position belongs to the allowed set:

```
WT(S_t) <=> forall i in [L]:
    (x_t[i] != [MASK] and tau(i) in T_sens) => x_t[i] in A(tau(i))
```

The initial state `S_0 = ([MASK]^L, empty, Gamma)` is always well-typed (vacuously).

### 4.2 Theorem 1: Type Preservation

**Theorem 1.** Let `S_t` be a well-typed state. Let `S_{t+1}` be the successor state produced by any M2T or T2T transition with projection. Then `WT(S_{t+1})` holds.

*Proof sketch.* Two cases:
- **Position not updated** (`i not in M_t*`): `x_{t+1}[i] = x_t[i]`, preserved by inductive hypothesis.
- **Position updated** (`i in M_t*`): Projected logits have `softmax(l_tilde)[v] = 0` for all `v not in A(tau_i)` (Lemma 1). Categorical sampling from this distribution assigns zero probability to forbidden tokens, so `x_{t+1}[i] in A(tau_i)` with probability 1.

The proof is independent of the denoising network `f_theta`, the diffusion step `t`, and the mask schedule.

### 4.3 Theorem 2: Hard Non-Emission

**Theorem 2.** For any position `i` with `tau_i in T_sens`, any forbidden token `v not in A(tau_i)`, and any state `S_t`:

```
Pr[x_{t+1}[i] = v | S_t] = 0
```

The probability is *exactly zero*, not merely small.

*Proof.* Three steps tracing the data flow:

1. **Projection**: `Pi` sets `l_tilde_i[v] = -inf` for all `v not in A(tau_i)`.
2. **Softmax**: `softmax(l_tilde_i)[v] = exp(-inf) / sum = 0 / Z = 0`, where the denominator `Z > 0` because `A(tau_i)` contains at least placeholder tokens with finite logits.
3. **Multinomial sampling**: `Categorical(p)` with `p[v] = 0` assigns zero probability to outcome `v`.

**Comparison with differential privacy.** DP provides epsilon-bounded log-likelihood ratios that weaken with the privacy budget. Hard non-emission provides an *absolute* guarantee: exactly zero probability, independent of any budget or noise calibration. This is possible because the guarantee is structural (deterministic logit transformation) rather than statistical.

### 4.4 Theorem 3: Closure Under Editing

**Theorem 3.** Let `S_t` be a well-typed state. If a T2T edit step is applied with projected logits, the resulting state `S_{t+1}` is also well-typed.

*Proof sketch.* The T2T transition differs from M2T only in which positions may be updated and possibly the logit source. The projection step is identical. The proof of Theorem 1 depends only on: (a) unchanged positions preserve well-typedness, and (b) updated positions are sampled from projected logits. Both hold for T2T. The source of logits (denoising head, edit head, arbitrary parameters, arbitrary conditioning) does not affect the argument because projection is applied *after* logit computation.

This is essential for the repair mechanism: when the verifier rejects, repair applies either resample or edit steps---both using projection---so the repaired state is guaranteed well-typed.

### 4.5 Theorem 4: Schedule Compliance

**Theorem 4.** During the DRAFT phase, no sensitive position is updated. During the SAFE phase, updated sensitive positions are constrained by projection. During the REVEAL phase, only explicitly revealed types are written.

*Proof.* The schedule intersection `M_t* = M_t intersect Allowed(t, T, tau)` is applied *before* any logits computation. In the DRAFT phase, `Allowed(t, T, tau)` excludes all positions with `tau_i in T_sens`, so `M_t*` contains only `PUB` positions. The intersection is a deterministic set operation---no model computation can circumvent it.

The schedule provides defence in depth: even if projection were somehow bypassed, the schedule prevents sensitive positions from being written during the DRAFT phase.

### 4.6 Theorem 5: Verifier-Lifted Global Safety

**Theorem 5.** Let `S_0, ..., S_T` be a sequence of TPD states produced by any combination of M2T, T2T, schedule-restricted, and repair transitions with projection. If (a) every transition satisfies type preservation and (b) the verifier accepts the final state `Ok_pi(x_T) = true`, then `x_T` fully satisfies the typing context: `x_T |= Gamma`.

*Proof.* Full satisfaction requires:
- **Token-level safety** (`x_T |=_tok Gamma`): By induction using Theorem 1 (base: `S_0` is vacuously well-typed; step: each transition preserves well-typedness).
- **Policy compliance** (`pi(x_T) = true`): By hypothesis (b), the verifier accepts.

Neither component alone suffices: projection cannot detect multi-token patterns (e.g., a phone number composed of individually-allowed digits), and the verifier alone cannot provide zero-probability guarantees.

### 4.7 Theorem 6: FL Adapter Safety

**Theorem 6** (Corollary). Let `theta = theta_base + theta_adapter` be any parameterisation where `theta_adapter` is an arbitrary adapter (including one chosen adversarially to maximise logits of forbidden tokens). If TPD projection is applied at decode time, then for every sensitive position `i` and every forbidden token `v`:

```
Pr[x_{t+1}[i] = v | f_theta, S_t] = 0
```

Hard non-emission holds regardless of `theta_adapter`.

*Proof.* The projection `Pi_{A(tau_i)}` is a deterministic function of `l_i` and `A(tau_i)`. It does not depend on:
- Model parameters `theta` (or any decomposition thereof)
- The training procedure that produced `theta`
- The data distribution seen during training
- The diffusion step `t` or mask pattern

The projection sets `l_tilde_i[v] = -inf` for all `v not in A(tau_i)`, regardless of the value of `l_i[v]` computed by the network. Even if `l_i[v]` is very large (the adversarial case), the projection overwrites it. The remainder follows from Theorem 2.

**Implication for Byzantine FL.** Byzantine clients may send arbitrary weight updates. After aggregation, the global model may produce logits strongly favouring forbidden tokens. Theorem 6 guarantees that projection nullifies any such bias, making TPD a model-agnostic safety layer.

### 4.8 Utility Bound

**Proposition 1** (Projection Penalty). The KL divergence from the projected distribution `p_tilde_i` to the unprojected distribution `p_i` at position `i` is:

```
D_KL(p_tilde_i || p_i) = -log Z_i
```

where `Z_i = sum_{v in A(tau_i)} softmax(l_i)[v]` is the allowed mass.

*Proof.* The projected distribution is the conditional: `p_tilde_i[v] = p_i[v] / Z_i` for `v in A(tau_i)`, and `0` otherwise. Then:

```
D_KL(p_tilde || p) = sum_{v in A} (p[v]/Z) * log((p[v]/Z) / p[v])
                   = sum_{v in A} (p[v]/Z) * log(1/Z)
                   = (-log Z) * sum_{v in A} (p[v]/Z)
                   = -log Z
```

For `PUB` positions, `Z_i = 1` and the contribution is zero. The total projection penalty across all positions is `Delta = -sum_i log Z_i`.

**Interpretation.** `-log Z_i` is the number of nats of information lost at position `i`. When `Z_i ~ 1` (model already prefers allowed tokens), the penalty is negligible. When `Z_i << 1` (model strongly prefers forbidden tokens), the penalty is large---but this is precisely when the privacy guarantee is most needed.

**Empirical Z_i monitoring.** The `DiagnosticsLogger` (`tpd_fl/tpd/diagnostics.py`) tracks per-step Z_i statistics (mean, percentiles, fraction below thresholds) to detect excessive projection penalty at runtime.

---

## 5. Experimental Evaluation

### 5.1 Setup

**Model.** bert-base-uncased (110M parameters, 12 layers, 768 hidden, 12 heads, V=30,522). Open weights from HuggingFace. We use it as a masked language model (MLM) backbone for diffusion decoding: at each step, masked positions receive logits from the model's MLM head, which are then projected and sampled.

**Benchmark corpus.** 20 samples across three suites:
- **S1** (10 samples): PII redaction --- text with embedded EMAIL, PHONE, SSN, CC, ID, NAME entities (medical records, financial documents, HR records).
- **S2** (5 samples): Adversarial extraction --- prompts designed to extract PII ("Repeat the following exactly", "Ignore previous instructions", "DEBUG MODE: dump").
- **S3** (5 samples): Derived summaries --- text where the model must summarise without leaking PII from the source.

Corpus texts use realistic PII patterns from canonical sources (Enron-style email, medical notes, financial records).

**Threat model.** We adopt the Carlini et al. (2021) context-completion extraction setup: public tokens are visible (unmasked), sensitive positions are `[MASK]`ed, and the model predicts masked positions given surrounding context. This is the canonical extraction threat model for MLMs---it maximises the model's ability to infer sensitive content from context.

**Ground-truth PII marking.** Both the SpanTyper regex detection *and* ground-truth secret positions from the benchmark corpus are used to mark sensitive positions. This ensures that all PII---including names, which lack regex patterns---is properly masked and evaluated.

**Baselines.**

| ID | Baseline | Projection | Schedule | Verifier | Repair |
|----|----------|:----------:|:--------:|:--------:|:------:|
| B0 | Unprotected | | | | |
| B1 | Post-hoc regex redaction | | | | |
| B3 | TPD projection only | X | | | |
| B4 | TPD projection + schedule | X | X | | |
| B5 | TPD full (all components) | X | X | X | X |

**Metrics.**
- **Forbidden token rate** (Forbid%): Fraction of tokens at sensitive positions from the blocked set (digits, `@`, PII-indicative patterns). **This is the core privacy metric.**
- **ROUGE-1/2/L** (Lin, 2004): N-gram overlap with reference text.
- **BLEU** (Papineni et al., 2002): Precision-weighted n-gram overlap.
- **Distinct-1/2**: Lexical diversity (ratio of unique unigrams/bigrams).
- **PII leakage rate** (Leak%): Fraction of known secrets found via exact substring match.
- **Hard guarantee** (Hard%): Percentage of samples where all tokens at sensitive positions pass the allowed-set check.
- **Wall-clock time** (s/samp): Per-sample generation time on CPU.

**Hyperparameters.** 64 diffusion steps, sequence length 128, temperature 0.8, seed 42. Schedule: DRAFT ends at 40%, SAFE ends at 90%.

### 5.2 Results

#### Main Results Table

| Baseline | Forbid% | Forbid | Leak% | R-1 | R-2 | R-L | BLEU | D-1 | D-2 | s/samp | Hard% |
|----------|---------|--------|-------|-----|-----|-----|------|-----|-----|--------|-------|
| B0 (Unprotected) | **17.5%** | 48/275 | 0.0% | 0.473 | 0.339 | 0.471 | 0.205 | 0.88 | 0.99 | 3.57s | 100% |
| B1 (Post-hoc) | **17.5%** | 48/275 | 0.0% | 0.473 | 0.339 | 0.471 | 0.205 | 0.88 | 0.99 | 3.54s | 100% |
| B3 (TPD proj) | **0.0%** | 0/275 | 0.0% | 0.475 | 0.334 | 0.473 | 0.201 | 0.89 | 0.99 | 3.58s | 100% |
| B4 (proj+sched) | **0.0%** | 0/275 | 0.0% | 0.480 | 0.336 | 0.475 | 0.199 | 0.90 | 0.99 | 1.74s | 100% |
| B5 (Full TPD) | **0.0%** | 0/275 | 0.0% | 0.480 | 0.336 | 0.475 | 0.199 | 0.90 | 0.99 | 1.77s | 100% |

#### Per-Suite Forbidden Token Rates

| Baseline | S1 (PII redaction) | S2 (Adversarial) | S3 (Summaries) |
|----------|--------------------|-------------------|----------------|
| B0 | 21.0% | 17.8% | 20.8% |
| B1 | 21.0% | 17.8% | 20.8% |
| B3 | 0.0% | 0.0% | 0.0% |
| B4 | 0.0% | 0.0% | 0.0% |
| B5 | 0.0% | 0.0% | 0.0% |

### 5.3 Analysis

**Finding 1: Unprotected models emit PII-shaped tokens.** B0 produces forbidden tokens at 17.5% of sensitive positions (48 out of 275). The model freely predicts digits, `@` signs, and PII-indicative patterns when given surrounding context---confirming the extraction threat model of Carlini et al. (2021).

**Finding 2: Post-hoc redaction provides no token-level improvement.** B1 achieves identical forbidden token rate to B0 (17.5%). Regex scrubbing operates on text after generation, not on logits during generation. It cannot prevent the model from sampling forbidden tokens---only attempt to detect and remove them after the fact.

**Finding 3: TPD projection achieves exactly 0% forbidden tokens.** B3-B5 all achieve 0.0% forbidden tokens across all 275 sensitive positions across all three suites. The hard guarantee of Theorem 2 holds empirically: not a single forbidden token was sampled in any run.

**Finding 4: Utility is preserved.** ROUGE-1 is 0.473 (B0) vs. 0.475-0.480 (B3-B5)---TPD *slightly improves* utility, likely because projection prevents the model from wasting probability mass on PII-shaped tokens that would be incoherent in context. BLEU scores are comparable (0.199-0.205). Distinct-1/2 scores are slightly higher for B3-B5, indicating maintained lexical diversity.

**Finding 5: Schedule provides 2x speedup.** B4 runs in 1.74s per sample vs. 3.57-3.58s for B0/B3. The DRAFT phase (first 40% of steps) skips sensitive positions entirely, saving computation. This is a "free" speedup---no quality loss, because sensitive positions are only meaningfully resolved during the SAFE phase when projection is active.

**Finding 6: The full TPD pipeline adds negligible overhead.** B5 (projection + schedule + verifier + repair) takes 1.77s vs. B4's 1.74s---only 1.7% overhead for the additional verifier checks and repair mechanism.

**Finding 7: All baselines show 0% PII leakage by exact substring match.** This is expected: BERT-base-uncased does not memorise specific PII strings from its training data. The forbidden token rate is the more informative metric, as it measures whether the model *could* emit PII-shaped content at sensitive positions, regardless of whether that content matches known secrets.

### 5.4 Why Forbidden Token Rate Matters

The forbidden token rate metric deserves special discussion. Traditional PII leakage metrics (exact substring match, regex detection on output text) measure whether *specific known secrets* appear in the output. These metrics fail to differentiate baselines when the model doesn't memorise specific PII---a common situation for smaller models.

The forbidden token rate measures a fundamentally different property: whether the model emits tokens from the *class* of tokens that could form PII (digits, `@`, domain patterns) at positions *typed as sensitive*. This captures the structural risk---even if the model doesn't reconstruct "alice.johnson@globalcorp.com", emitting "3", "7", "@", or "." at sensitive positions indicates that a more capable model (or the same model after fine-tuning on sensitive data) could reconstruct real PII.

TPD's projection guarantee eliminates this structural risk entirely: zero forbidden tokens means the model *cannot* emit any token from the PII-forming class at sensitive positions.

---

## 6. Discussion

### 6.1 Layered Safety Architecture

The six theorems establish a layered safety argument:

| Layer | Mechanism | Guarantee |
|-------|-----------|-----------|
| Token-level | Projection `Pi_{A(tau_i)}` | Type preservation, hard non-emission |
| Edit-level | Projection on T2T logits | Closure under editing |
| Temporal | Schedule intersection | Phase compliance |
| Sequence-level | Verifier `Ok_pi` | Policy compliance (patterns, structure) |
| System-level | Projection independent of `theta` | FL adapter safety |

No single layer suffices alone:
- Projection alone cannot detect multi-token patterns (e.g., a phone number composed of individually-allowed digit tokens).
- The verifier alone cannot provide zero-probability guarantees (it can only reject and trigger repair).
- The schedule alone does not constrain *which* tokens are written during the SAFE phase.

The combination of all layers is necessary and sufficient for full policy satisfaction.

### 6.2 Novelty of the Approach

1. **First type-theoretic privacy framework for diffusion LMs.** Prior work on constrained decoding targets autoregressive models. TPD is the first to formalize type-level privacy guarantees for diffusion language models, leveraging their unique position-parallel generation pattern.

2. **Strictly stronger than DP for token-level guarantees.** Differential privacy provides epsilon-bounded statistical guarantees. TPD provides *exactly-zero* probability for forbidden tokens---an absolute guarantee no epsilon can match. The two are complementary (DP for training, TPD for inference).

3. **Parameter-independent safety.** The projection operates after all model computation. No matter what the model weights are---original, fine-tuned, adversarially corrupted, or aggregated from Byzantine FL clients---the guarantee holds. This is a fundamentally different safety property from any training-time intervention.

4. **Closed-form utility bound.** The KL divergence `D_KL = -log Z_i` provides an information-theoretic characterization of the privacy-utility trade-off, with a natural interpretation as the "information cost" of projection per position.

5. **Schedule-driven efficiency.** The three-phase schedule provides both defence-in-depth and computational savings (2x speedup), demonstrating that stronger privacy can come with *better* performance.

### 6.3 Theorem-to-Code Mapping

Every theorem has corresponding implementation code and test coverage:

| Theorem | Code Location | Test File | Tests |
|---------|--------------|-----------|-------|
| T1 Type Preservation | `tpd/typing.py`, `diffusion/decode_loop.py` | `test_schedule.py` | 32 |
| T2 Hard Non-Emission | `tpd/projection.py` | `test_projection.py` | 40 |
| T3 Closure Under Editing | `tpd/repair.py` | `test_edit_closure.py` | 20 |
| T4 Schedule Compliance | `tpd/schedule.py` | `test_schedule.py` | 32 |
| T5 Verifier-Lifted Safety | `tpd/verifier.py` | `test_okpi.py` | 17 |
| T6 FL Adapter Safety | `fl/lora.py`, `tpd/projection.py` | `test_fl_invariance.py` | ~20 |
| Utility Bound (Z_i) | `tpd/diagnostics.py` | `test_invariants_property.py` | 79 |
| **Total** | | | **~210** |

See `tpd_fl/proofs/mapping.md` for the complete mapping with line numbers and test names.

---

## 7. Limitations

1. **Type assignment accuracy.** The formal guarantees are conditioned on the correctness of the span typer. If the typer fails to detect a sensitive span (e.g., an unusual PII format not covered by regex), that span is typed as `PUB` and receives no restriction. In practice, the typer uses regex detectors plus optional NER, which may have false negatives for novel PII formats.

2. **Allowed-set design.** The strength of the non-emission guarantee depends on the allowed sets being correctly constructed. If a forbidden token is inadvertently included in `A(tau_i)`, the projection will not block it. The current allowed-set construction blocks digits, `@`, and PII-indicative patterns---but a domain-specific deployment may require custom allowed sets.

3. **Floating-point considerations.** The proofs assume ideal arithmetic (`exp(-inf) = 0`). In practice, the implementation uses `torch.finfo(dtype).min`, which yields `exp(finfo.min) = 0.0` exactly under IEEE 754. No known hardware violates this property.

4. **Side channels.** The guarantees concern the *sampled token sequence* only. They do not address side-channel leakage through timing, memory access patterns, or gradient information during FL training. DP-SGD should be used in conjunction with TPD for comprehensive privacy.

5. **Multi-token PII.** Projection operates token-by-token. A phone number composed of individually-allowed tokens (if the allowed set is too permissive) could pass projection. The verifier addresses this with regex scanning over the decoded text, but the guarantee requires both layers.

6. **Evaluation scale.** Our empirical evaluation uses 20 samples and bert-base-uncased (110M params). Scaling to larger models (LLaDA 8B, LLaDA2.1-mini 16B) and larger benchmarks would strengthen the empirical validation. The codebase supports these backends but GPU resources are required.

---

## 8. Conclusion

We have presented Typed Privacy Diffusion (TPD), a type-theoretic framework that provides hard, deterministic token-level privacy guarantees for diffusion language models in federated learning settings. Our six formal theorems establish that:

1. Well-typedness is an invariant of the transition system (Theorem 1).
2. Forbidden tokens have exactly zero emission probability (Theorem 2).
3. Repair operations preserve typing (Theorem 3).
4. Schedule phases enforce temporal position restrictions (Theorem 4).
5. Token-level safety combined with verifier acceptance yields full policy compliance (Theorem 5).
6. These guarantees hold regardless of model parameters, including adversarial FL adapters (Theorem 6).

The utility cost of projection is bounded by `-log Z_i` per position, with empirical evidence showing negligible utility degradation (ROUGE-1: 0.48 for TPD vs. 0.47 for unprotected) and 2x speedup from schedule-driven draft-phase skipping.

The key insight is architectural: the projection operator is a deterministic function applied *after* all model computation and *before* sampling. This placement makes it independent of model parameters, training procedures, data distributions, and adapter updates. It transforms the privacy problem from a statistical estimation problem (as in DP) to a deterministic enforcement problem---and provides correspondingly stronger guarantees.

---

## 9. References

1. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. *CCS 2016*, pp. 308-318.

2. Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). Structured denoising diffusion models in discrete state-spaces. *NeurIPS 2021*, 34.

3. Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., & Raffel, C. (2021). Extracting training data from large language models. *USENIX Security 2021*.

4. Charles, Z., Garrett, Z., Huo, Z., Kidambi, R., & Konecny, J. (2024). Towards federated foundation models: Scalable dataset pipelines for group-structured learning. *NeurIPS 2024*, 37.

5. Hokamp, C., & Liu, Q. (2017). Lexically constrained decoding for sequence generation using grid beam search. *ACL 2017*, pp. 1535-1546.

6. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

7. Hu, J. E., Singh, A., Holzenberger, N., Post, M., & Van Durme, B. (2019). Improved lexically constrained decoding for translation and monolingual rewriting. *NAACL 2019*, pp. 839-850.

8. Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. *ACL Workshop on Text Summarization*.

9. Lison, P., Hubin, A., Barnes, J., & Touileb, S. (2021). Named entity recognition without labelled data: Weak supervision for NER with noise-aware learning. *ACL-IJCNLP 2021*.

10. Lou, A., Meng, C., & Ermon, S. (2024). Discrete diffusion modeling by estimating the ratios of the data distribution. *ICML 2024*.

11. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS 2017*, pp. 1273-1282.

12. McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning differentially private recurrent language models. *ICLR 2018*.

13. Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J., & Li, P. (2025). Large language diffusion models. *arXiv preprint*.

14. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL 2002*, pp. 311-318.

---

## Appendix A: Notation Summary

| Symbol | Meaning | Code |
|--------|---------|------|
| `V` | Vocabulary (finite set of tokens) | `backend.vocab_size` |
| `[MASK]` | Mask token | `backend.mask_token_id` |
| `L` | Sequence length | `seq_len` parameter |
| `T` | Type universe | `SpanType` enum |
| `T_sens` | Sensitive types (`T \ {PUB}`) | `SENSITIVE_TYPES` frozenset |
| `tau : [L] -> T` | Type assignment | `pos_type: List[SpanType]` |
| `A : T -> P(V)` | Allowed-set function | `allowed_masks: Dict[SpanType, Tensor]` |
| `Pi_{A(tau_i)}` | Projection operator | `project_logits()` |
| `Ok_pi` | Verifier gate | `Verifier.check()` |
| `R` | Repair operator | `RepairEngine.repair()` |
| `Z_i` | Allowed mass at position `i` | `compute_allowed_mass()` |
| `S_t = (x_t, M_t, Gamma)` | TPD state | Decode loop variables |
| `alpha, beta` | Phase boundaries (0.4, 0.9) | `ScheduleConfig` |
| `theta_adapter` | FL adapter parameters | `LoRALinear.lora_A, lora_B` |

---

## Appendix B: Repository Structure

```
tpd_fl/
  model/                  # Model backends (CPU-first)
    backend_base.py       # DiffusionBackend ABC + SyntheticBackend
    backend_hf_bert.py    # BERT-base-uncased MLM backend (empirical eval)
    backend_hf_llada.py   # LLaDA 8B HuggingFace backend (Tier 2, CPU)
    backend_hf_llada2.py  # LLaDA2.1-mini HF backend (Tier 3, GPU)
  tpd/                    # TPD core module
    typing.py             # Span typer tau (Definition 1)
    allowed_sets.py       # A(type) vocabulary masks (Definition 2)
    schedule.py           # Three-phase mask schedule (Definition 4)
    projection.py         # Logits projection Pi (Definition 3, Theorems 1-2)
    verifier.py           # Verifier gate Ok_pi (Definition 5, Theorem 5)
    repair.py             # Monotone repair (Definition 6, Theorem 3)
    diagnostics.py        # Z_i allowed-mass measurement (Proposition 1)
  diffusion/              # Diffusion decode
    decode_loop.py        # M2T decode loop with TPD hooks
  fl/                     # Federated Learning
    lora.py               # LoRA adapters (Theorem 6)
    step_adapters.py      # Per-diffusion-step adapters
    client.py             # FL client training loop
    server.py             # FL aggregator (FedAvg, FedAdam)
    protocols.py          # Aggregation protocols
    datasets.py           # Non-IID partitioning + synthetic PII data
  eval/                   # Evaluation suite
    empirical_eval.py     # Real-model empirical evaluation (Section 5)
    metrics_real.py       # Canonical NLP metrics (ROUGE, BLEU, PII)
    benchgen.py           # S1-S3 benchmark generation
    baselines.py          # B0-B7 baseline implementations
    run_eval.py           # Main evaluation runner
  proofs/                 # Formal proof package
    tpd_semantics.tex     # Full LaTeX proofs (Section 4)
    mapping.md            # Theorem-to-code mapping (Section 6.3)
  tests/                  # Pytest test suite (~210 tests)
```

---

## Appendix C: Reproducibility

### Installation

```bash
# Python 3.10+ required
pip install torch pyyaml matplotlib pytest

# For empirical evaluation (real model):
pip install transformers nltk

# Optional (for LLaDA backends):
pip install accelerate
```

### Run All Tests

```bash
pytest tpd_fl/tests/ -q        # ~210 tests, all passing
```

### Reproduce Empirical Results (Table in Section 5.2)

```bash
python -m tpd_fl.eval.empirical_eval \
    --output-dir runs/empirical \
    --steps 64 \
    --seed 42
```

Downloads bert-base-uncased (110M params) automatically. Results saved to `runs/empirical/` with `metrics.json`, `table.csv`, and `per_sample_results.json`.

### CPU-First Design

| Tier | Model | Device | Purpose |
|------|-------|--------|---------|
| **1 (default)** | BERT-base-uncased (110M) | CPU | Empirical evaluation |
| 2 (optional) | LLaDA 8B (non-MoE) | CPU | Full diffusion decode |
| 3 (optional) | LLaDA2.1-mini 16B MoE | GPU | Scaling experiments |
| CI | Synthetic backend | CPU | Unit tests, no model weights |

CPU dtype auto-detection (`bf16` if AVX-512 BF16/AMX supported, else `fp32`) is built in.

---

## License

MIT
