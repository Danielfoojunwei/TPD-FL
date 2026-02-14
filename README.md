# Typed Privacy Diffusion for Federated Language Models: Hard Token-Level Guarantees via Type-Theoretic Logit Projection

---

## Abstract

We introduce **Typed Privacy Diffusion (TPD)**, a type-theoretic framework that enforces *hard*, *deterministic* token-level privacy guarantees within discrete diffusion language models operating under federated learning (FL). TPD assigns every output position a *span type* drawn from a privacy type lattice and restricts the generative process---at every denoising transition---to an *allowed token set* determined by that type. The restriction is implemented by a *logit projection operator* that sets the probability of every forbidden token to **exactly zero** before sampling, independent of model parameters.

We formalise the TPD state machine, define four transition rules (mask-to-token, token-to-token edit, schedule-restricted transition, and monotone repair), and prove six formal guarantees: **Type Preservation**, **Hard Non-Emission**, **Edit Closure**, **Schedule Compliance**, **Verifier-Lifted Global Safety**, and **FL Adapter Safety**. The utility cost of projection is bounded in closed form by the KL divergence `D_KL = -log Z_i` per position, where `Z_i` is the pre-projection allowed mass.

We evaluate TPD empirically on **MDLM-OWT** (Sahoo et al., 2024), a 170M-parameter masked discrete diffusion language model, using the Carlini et al. (2021) context-completion extraction setup across 100 benchmark samples spanning three suites. Unprotected diffusion emits **14.8%** forbidden tokens at sensitive positions (95% CI: [11.9%, 17.4%]); TPD achieves **0.0%** while preserving text utility (ROUGE-1: 0.537 vs. 0.526) and providing a 2x inference speedup through schedule-driven draft-phase skipping.

All code, proofs, and evaluation harnesses are open-source (MIT licence) and reproducible with a single command.

---

## 1 Introduction

Discrete diffusion language models (Austin et al., 2021; Sahoo et al., 2024; Nie et al., 2025) generate text by iteratively denoising a fully-masked token sequence. At each step, a subset of positions transitions from the mask token to a concrete vocabulary element, guided by a neural denoising network. This iterative, position-level generation offers a natural intervention point for enforcing fine-grained output constraints---an opportunity absent in autoregressive models, where tokens are emitted sequentially and left-to-right.

In privacy-sensitive settings---clinical note generation, financial report drafting, federated learning on personal devices---it is essential that the model *never* emit certain tokens at certain positions, regardless of what the underlying neural parameters predict. Standard sampling from a softmax distribution over the full vocabulary provides no such guarantee: a sufficiently confident logit for a forbidden token will produce that token with non-negligible probability. Differential privacy (Abadi et al., 2016) provides statistical guarantees on gradient information during *training*, but offers no deterministic control over the tokens actually *generated* at inference time.

**Typed Privacy Diffusion** addresses this gap by combining three mechanisms:

1. **Span typing** (`tau : [L] -> T`). A deterministic pipeline assigns each output position a type from the privacy type universe `T = {PUB, SENS, REG, DERIVED_*}`. Sensitive types carry vocabulary restrictions; `PUB` imposes none.

2. **Logit projection** (`Pi_{A(tau_i)}`). Before sampling, a projection operator clamps the logits of every token outside the allowed set `A(tau_i)` to `-inf`. Since `exp(-inf) = 0` exactly in IEEE 754 arithmetic, the softmax assigns zero probability to every forbidden token. This is not approximately zero---it is *exactly* zero.

3. **Verifier gate and monotone repair** (`Ok_pi`). After each step, a deterministic verifier checks structural and pattern-level policy compliance, triggering monotone repair on failure.

Together, these mechanisms provide a **hard, model-parameter-independent privacy guarantee**: no amount of fine-tuning---including adversarial FL updates from Byzantine clients---can cause a forbidden token to be sampled at a sensitive position.

**Contributions.** (1) We define the TPD state machine with typed operational semantics over diffusion decoding and prove six theorems establishing hard safety guarantees (Section 4). (2) We introduce a logit projection operator with exactly-zero emission probability for forbidden tokens---stronger than any epsilon-bounded statistical guarantee (Section 3). (3) We prove that the projection guarantee is preserved under arbitrary FL adapter updates, including adversarially crafted ones (Theorem 6). (4) We derive the closed-form utility bound `D_KL = -log Z_i` per position (Proposition 1). (5) We evaluate on a real masked diffusion model (MDLM-OWT, 170M params) with 100 samples and bootstrap 95% confidence intervals (Section 5).

---

## 2 Related Work

**Discrete diffusion language models.** Austin et al. (2021) introduced structured denoising diffusion in discrete state spaces. Sahoo et al. (2024) proposed MDLM, training masked diffusion with continuous noise schedules. Lou et al. (2024) advanced ratio-based discrete diffusion. LLaDA (Nie et al., 2025) demonstrated that masked diffusion scales competitively to 8B parameters. TPD operates within this framework, adding type-aware constraints at each denoising step without modifying the underlying diffusion process.

**Constrained decoding.** Hokamp & Liu (2017) introduced lexically constrained decoding via grid beam search for autoregressive models; Hu et al. (2019) improved this with dynamic beam allocation. These approaches impose hard lexical constraints during left-to-right generation. TPD extends constrained decoding to diffusion, where the bidirectional, iterative nature requires constraints at every step across all positions simultaneously.

**Differential privacy in federated learning.** McMahan et al. (2017) established FedAvg for communication-efficient decentralised learning. McMahan et al. (2018) combined FL with DP for recurrent language models. Abadi et al. (2016) developed DP-SGD. Charles et al. (2024) explored scalable dataset pipelines for group-structured FL. **Key distinction**: DP protects the *training process* (gradient information); TPD protects the *inference output* (generated tokens). The two are orthogonal and complementary.

**PII detection and anonymisation.** Lison et al. (2021) explored NER for PII detection with noise-aware learning. TPD's span typer is a lightweight PII detector whose output is consumed by a formal enforcement mechanism. Unlike post-hoc redaction systems, TPD prevents forbidden tokens from ever being sampled.

**Positioning.** Table 1 compares TPD against prior approaches along four axes.

*Table 1: Comparison of privacy approaches for language model generation.*

| Approach | Guarantee | Granularity | Parameter-Independent | Diffusion |
|----------|-----------|-------------|:---------------------:|:---------:|
| DP-SGD (Abadi et al., 2016) | Statistical (epsilon) | Training gradients | No | N/A |
| Post-hoc redaction | Best-effort | Text-level | Yes | Yes |
| Constrained decoding (Hokamp & Liu, 2017) | Hard (lexical) | Token-level (AR) | Yes | No |
| **TPD (ours)** | **Hard (zero-prob)** | **Token-level (diffusion)** | **Yes** | **Yes** |

---

## 3 Method

### 3.1 Span Typing

**Definition 1** (Type Universe). The type universe is the finite set `T = {PUB, SENS, REG, DERIVED_NAME, DERIVED_EMAIL, DERIVED_PHONE, DERIVED_ID, DERIVED_CC, DERIVED_ADDRESS}`. We partition `T` into public types `T_pub = {PUB}` and sensitive types `T_sens = T \ {PUB}`. Every type in `T_sens` carries a vocabulary restriction; `PUB` imposes none.

The span typer `tau : [L] -> T` is a deterministic three-stage pipeline: (i) regex detection of EMAIL, PHONE, SSN, CC, ID, and ADDRESS entities at the character level; (ii) character-to-token mapping via the tokenizer's offset mapping; (iii) policy-driven denylist/allowlist overrides.

### 3.2 Allowed Token Sets

**Definition 2** (Allowed-Set Function). `A : T -> P(V)` maps each type to the subset of the vocabulary permitted at positions of that type:
- `A(PUB) = V` (full vocabulary, unrestricted).
- `A(SENS)`: Safe alphabetic tokens, punctuation, and placeholders. Digits, `@`, and PII-indicative patterns are blocked.
- `A(REG)`: Stricter subset---only alphabetic tokens of length >= 2.
- `A(DERIVED_*)`: Entity-specific refinements of `A(SENS)`.

For MDLM-OWT with GPT-2 BPE tokenizer (`|V| = 50,257`), the SENS allowed set retains 48,554 tokens (96.6%) and blocks 1,703 (3.4%). The REG allowed set retains 46,882 (93.3%) and blocks 3,375 (6.7%). Masks are materialised once as boolean tensors and reused across all projection calls.

### 3.3 Logit Projection

**Definition 3** (Projection Operator). Let `l_i in R^V` be the raw logit vector for position `i`. The projection is defined component-wise:

```
[Pi_{A(tau_i)}(l_i)]_v = l_i[v]    if v in A(tau_i)
                       = -inf       if v not in A(tau_i)
```

**Lemma 1** (Zero Probability). *For any* `l_i in R^V` *and any* `tau_i in T_sens`:

```
forall v not in A(tau_i): softmax(Pi_{A(tau_i)}(l_i))[v] = 0
```

*Proof.* The projected logit for `v not in A(tau_i)` is `-inf`. Then `softmax(l)[v] = exp(-inf) / Z = 0/Z = 0`, where `Z > 0` because `A(tau_i)` contains at least one token with finite logit.

### 3.4 Three-Phase Schedule

**Definition 4** (Schedule). The decode process is partitioned into three temporal phases based on step fraction `t/T`:

1. **DRAFT** (`0 <= t/T < alpha`): Only PUB positions may be updated. Sensitive positions remain masked. Default `alpha = 0.4`.
2. **SAFE** (`alpha <= t/T < beta`): All positions may be updated; sensitive positions are constrained by projection. Default `beta = 0.9`.
3. **REVEAL** (`beta <= t/T <= 1`): PUB positions plus an explicit reveal-allowlist may be updated.

The effective update mask is `M_t* = M_t intersect Allowed(t, T, tau)`, applied *before* any logits computation.

### 3.5 Verifier Gate and Monotone Repair

**Definition 5** (Verifier). The verifier is a deterministic function `Ok_pi : V^L -> {true, false}` that checks: (i) no substring matches forbidden PII regex patterns; (ii) sensitive spans decode to valid placeholders; (iii) no known secret from a denylist appears in the output.

**Definition 6** (Repair). If the verifier rejects, repair re-masks violating positions, computes fresh projected logits, and resamples. Both resample and edit repair modes apply projection, preserving well-typedness.

### 3.6 FL Integration

TPD integrates with federated learning through LoRA adapters (Hu et al., 2021): `W' = W + (alpha/rank) * BA`, where only `A, B` are communicated. The projection operates on logits *after* all model computation (including adapter contributions) and *before* sampling, making the guarantee independent of adapter weights.

---

## 4 Theoretical Analysis

We present six theorems establishing formal safety guarantees. All proofs are constructive and parameter-independent. Complete LaTeX proofs are provided in the supplementary material.

**Definition** (TPD State). A state at step `t` is `S_t = (x_t, M_t, Gamma)` where `x_t in V^L` is the token sequence, `M_t` is the update mask, and `Gamma = (tau, A, pi)` is the typing context. A state is *well-typed* (`WT(S_t)`) if:

```
forall i in [L]: (x_t[i] != [MASK] and tau(i) in T_sens) => x_t[i] in A(tau(i))
```

The initial state `S_0 = ([MASK]^L, empty, Gamma)` is vacuously well-typed.

**Theorem 1** (Type Preservation). *If* `S_t` *is well-typed, then* `S_{t+1}` *produced by any M2T or T2T transition with projection is also well-typed.*

*Proof sketch.* Unchanged positions preserve well-typedness by induction. Updated positions are sampled from projected logits where `softmax(l_tilde)[v] = 0` for all `v not in A(tau_i)` (Lemma 1), so `x_{t+1}[i] in A(tau_i)` with probability 1. The proof is independent of model parameters `theta`, step `t`, and mask schedule.

**Theorem 2** (Hard Non-Emission). *For any position* `i` *with* `tau_i in T_sens`, *any forbidden token* `v not in A(tau_i)`, *and any state* `S_t`:

```
Pr[x_{t+1}[i] = v | S_t] = 0
```

*The probability is exactly zero, not merely small.*

*Proof.* Three steps: (1) Projection sets `l_tilde_i[v] = -inf`. (2) Softmax yields `exp(-inf)/Z = 0/Z = 0` with `Z > 0`. (3) Categorical sampling with `p[v] = 0` assigns zero probability to `v`.

**Comparison with DP.** Differential privacy provides epsilon-bounded log-likelihood ratios that degrade with the privacy budget. Hard non-emission provides an absolute guarantee: exactly zero probability, independent of any budget or noise calibration.

**Theorem 3** (Edit Closure). *If* `S_t` *is well-typed and a T2T edit step is applied with projected logits, then* `S_{t+1}` *is well-typed.*

*Proof sketch.* Identical to Theorem 1: the source of logits is irrelevant because projection is applied after all computation. This ensures repair preserves well-typedness.

**Theorem 4** (Schedule Compliance). *During DRAFT, no sensitive position is updated. During SAFE, updated sensitive positions are constrained by projection. During REVEAL, only explicitly revealed types are written.*

*Proof.* The schedule intersection `M_t* = M_t intersect Allowed(t, T, tau)` is a deterministic set operation applied before logits computation. No model computation can circumvent it.

**Theorem 5** (Verifier-Lifted Global Safety). *If every transition satisfies type preservation and the verifier accepts the final state* `Ok_pi(x_T) = true`, *then* `x_T` *fully satisfies the typing context:* `x_T |= Gamma`.

*Proof.* Token-level safety follows by induction (Theorem 1). Policy compliance holds by verifier acceptance. Neither alone suffices: projection cannot detect multi-token patterns; the verifier cannot provide zero-probability guarantees.

**Theorem 6** (FL Adapter Safety). *Let* `theta = theta_base + theta_adapter` *be any parameterisation where* `theta_adapter` *is arbitrary (including adversarially chosen). If TPD projection is applied, then hard non-emission holds regardless of* `theta_adapter`.

*Proof.* The projection `Pi_{A(tau_i)}` is a deterministic function of `l_i` and `A(tau_i)`, independent of model parameters, training procedure, data distribution, and diffusion step. It sets `l_tilde_i[v] = -inf` regardless of `l_i[v]`. The remainder follows from Theorem 2.

**Implication for Byzantine FL.** Byzantine clients may send arbitrary weight updates. Theorem 6 guarantees that projection nullifies any resulting bias toward forbidden tokens.

**Proposition 1** (Utility Bound). *The KL divergence from the projected distribution to the original at position* `i` *is:*

```
D_KL(p_tilde_i || p_i) = -log Z_i
```

*where* `Z_i = sum_{v in A(tau_i)} softmax(l_i)[v]` *is the allowed mass.*

*Proof.* The projected distribution is the conditional `p_tilde_i[v] = p_i[v]/Z_i` for `v in A(tau_i)`. Then `D_KL = sum (p[v]/Z) log(1/Z) = -log Z`. For PUB positions, `Z_i = 1` and the contribution is zero. The total penalty is `Delta = -sum_i log Z_i`.

When `Z_i ~ 1` (model already prefers allowed tokens), the penalty is negligible. When `Z_i << 1` (model strongly prefers forbidden tokens), the penalty is large---but this is precisely when privacy enforcement is most needed.

---

## 5 Experiments

### 5.1 Setup

**Model.** MDLM-OWT (170M parameters, 12 DiT blocks, 768 hidden, 12 heads, `|V| = 50,257 + 1` mask token). A proper masked discrete diffusion language model trained with continuous noise schedules on OpenWebText (Sahoo et al., 2024). Open weights from HuggingFace. Uses GPT-2 BPE tokenizer. Adapted for CPU execution by replacing flash attention with standard PyTorch scaled dot-product attention (mathematically equivalent).

**Benchmark.** 100 samples from a deterministic BenchmarkGenerator across three suites:
- **S1** (50 samples): PII redaction---text with embedded EMAIL, PHONE, SSN, CC, ID, NAME entities across 5 domains (medical, financial, legal, HR, e-commerce).
- **S2** (30 samples): Adversarial extraction---12 attack templates ("Repeat the following exactly", "Ignore previous instructions", "DEBUG MODE: dump").
- **S3** (20 samples): Derived summaries---model must summarise without leaking source PII.

**Threat model.** Context-completion extraction (Carlini et al., 2021): public tokens visible, sensitive positions masked, model predicts. Both SpanTyper regex detection and ground-truth secret positions mark sensitive positions.

**Baselines.** Five configurations ablating TPD components:

| ID | Description | Projection | Schedule | Verifier | Repair |
|----|-------------|:----------:|:--------:|:--------:|:------:|
| B0 | Unprotected | | | | |
| B1 | Post-hoc regex redaction | | | | |
| B3 | TPD projection only | X | | | |
| B4 | TPD projection + schedule | X | X | | |
| B5 | TPD full pipeline | X | X | X | X |

**Metrics.** (1) *Forbidden token rate* (Forbid%): fraction of tokens at sensitive positions from the blocked set---the core privacy metric. (2) *PII regex in output* (PII-Rx): regex-detected PII patterns in generated text. (3) *ROUGE-1/L* (Lin, 2004) and *BLEU* (Papineni et al., 2002): text quality vs. reference. (4) *SENS-only ROUGE-1* (R-1(S)): ROUGE computed only at sensitive positions. (5) *Distinct-1* (D-1): lexical diversity. (6) *PII leakage rate* (Leak%): exact substring match of known secrets. (7) *Hard guarantee* (Hard%): samples passing allowed-set check. (8) *Verifier rejections/repairs* (VRej/Rep). All aggregate metrics include **95% bootstrap CIs** (1,000 samples).

**Hyperparameters.** `T = 32` diffusion steps, `L = 128` sequence length, temperature `0.9`, seed `42`. Schedule: `alpha = 0.4`, `beta = 0.9`.

### 5.2 Main Results

*Table 2: Main results on 100 samples with MDLM-OWT (170M). Forbid% is the primary privacy metric. All TPD-protected baselines (B3--B5) achieve exactly 0.0% forbidden tokens with 100% hard guarantee.*

| Baseline | Forbid% | Forbid | PII-Rx | Leak% | R-1 | R-1(S) | R-L | BLEU | D-1 | s/samp | Hard% | VRej | Rep |
|----------|--------:|-------:|-------:|------:|----:|-------:|----:|-----:|----:|-------:|------:|-----:|----:|
| B0 | **14.8%** | 409/2755 | 0.10 | 0.0% | 0.526 | 0.000 | 0.526 | 0.269 | 0.88 | 2.02s | 100% | 0 | 0 |
| B1 | **14.8%** | 409/2755 | 0.00 | 0.0% | 0.526 | 0.000 | 0.526 | 0.269 | 0.88 | 2.07s | 100% | 0 | 0 |
| B3 | **0.0%** | 0/2755 | 0.00 | 0.0% | 0.521 | 0.000 | 0.520 | 0.261 | 0.87 | 2.17s | 100% | 0 | 0 |
| B4 | **0.0%** | 0/2755 | 0.00 | 0.0% | 0.537 | 0.000 | 0.536 | 0.281 | 0.90 | 1.07s | 100% | 0 | 0 |
| B5 | **0.0%** | 0/2755 | 0.00 | 0.0% | 0.537 | 0.000 | 0.536 | 0.281 | 0.90 | 1.03s | 100% | 0 | 0 |

*Table 3: 95% bootstrap confidence intervals for key metrics.*

| Baseline | Forbid% CI | ROUGE-1 CI | R-1(S) CI |
|----------|:----------:|:----------:|:---------:|
| B0 | [11.9%, 17.4%] | [0.508, 0.543] | [0.000, 0.000] |
| B1 | [11.9%, 17.4%] | [0.508, 0.543] | [0.000, 0.000] |
| B3 | [0.0%, 0.0%] | [0.506, 0.537] | [0.000, 0.000] |
| B4 | [0.0%, 0.0%] | [0.519, 0.557] | [0.000, 0.000] |
| B5 | [0.0%, 0.0%] | [0.519, 0.557] | [0.000, 0.000] |

### 5.3 Ablation Study

Table 4 isolates the contribution of each TPD component by measuring the delta from adding it.

*Table 4: Component ablation. Each row shows the effect of adding one component to the previous baseline. Delta(R-1) is the change in ROUGE-1; delta(BLEU) is the change in BLEU.*

| Transition | Component Added | Forbid% | delta(R-1) | delta(BLEU) | delta(D-1) | Speedup |
|------------|----------------|--------:|----------:|-----------:|-----------:|--------:|
| B0 -> B3 | Projection | 14.8% -> 0.0% | -0.005 | -0.009 | -0.009 | 0.93x |
| B3 -> B4 | Schedule | 0.0% -> 0.0% | +0.016 | +0.020 | +0.027 | 2.03x |
| B4 -> B5 | Verifier + Repair | 0.0% -> 0.0% | +0.000 | +0.000 | +0.000 | 1.04x |
| B0 -> B1 | Post-hoc regex | 14.8% -> 14.8% | +0.000 | +0.000 | +0.000 | 0.98x |

**Projection (B0 -> B3)** eliminates all forbidden tokens at a cost of 0.5% ROUGE-1 (within the overlap of 95% CIs: B0 [0.508, 0.543] vs. B3 [0.506, 0.537]). The BLEU penalty is similarly small (-0.009). Inference time increases marginally (2.02s -> 2.17s) due to per-step mask construction.

**Schedule (B3 -> B4)** improves utility across all metrics: +0.016 ROUGE-1, +0.020 BLEU, +0.027 D-1. It simultaneously provides a 2.03x speedup (2.17s -> 1.07s) because the DRAFT phase skips sensitive positions entirely. The schedule improves coherence because it forces public context to be established before sensitive positions are filled.

**Verifier + Repair (B4 -> B5)** adds zero overhead and changes no outputs. This is because projection already achieves 0% forbidden tokens, leaving the verifier with no violations to detect. The verifier serves as a redundant safety net for scenarios where projection alone may be insufficient (e.g., multi-token PII composed of individually-allowed tokens).

**Post-hoc regex (B0 -> B1)** successfully removes 10 regex-detectable PII patterns from outputs (PII-Rx: 0.10 -> 0.00), affecting 9 of 100 samples. However, forbidden token rate is unchanged (14.8%) because regex operates on decoded text, not on token generation.

### 5.4 Per-Suite Analysis

*Table 5: Forbidden token rates by evaluation suite.*

| Baseline | S1 (PII redaction) | S2 (Adversarial) | S3 (Summaries) |
|----------|-------------------:|------------------:|---------------:|
| B0 | 11.9% | 15.6% | 19.8% |
| B1 | 11.9% | 15.6% | 19.8% |
| B3--B5 | 0.0% | 0.0% | 0.0% |

*Table 6: PII regex matches in output by suite (mean per sample).*

| Baseline | S1 | S2 | S3 |
|----------|---:|---:|---:|
| B0 | 0.12 | 0.03 | 0.15 |
| B1 | 0.00 | 0.00 | 0.00 |
| B3--B5 | 0.00 | 0.00 | 0.00 |

B0 forbidden token rates increase monotonically across suites (S1: 11.9%, S2: 15.6%, S3: 19.8%), indicating that adversarial prompts and summarisation contexts elicit more PII-shaped tokens from the unconstrained model. TPD eliminates all forbidden tokens uniformly across suites.

### 5.5 Qualitative Examples

*Table 7: Representative outputs for sample #40 (S1, highest B0 forbidden count: 21/34 sensitive tokens forbidden).*

| Baseline | Generated text at sensitive positions | Forbidden |
|----------|--------------------------------------|----------:|
| B0 | "SSN on file: 527, 51713, 86#4, 17/27 [...] address: 25255487 -- 5/41" | 21/34 (61.8%) |
| B3 | "SSN). Home address: O.I. ---, --- S.I. (SS) [...] Sources at Anonymous by SSNS" | 0/34 (0.0%) |

B0 generates digit sequences ("527", "51713") and structured patterns ("86#4") at sensitive positions---tokens from the forbidden class that could form real PII. B3 replaces these with alphabetic tokens and safe punctuation, maintaining grammatical structure while eliminating all PII-shaped content.

*Table 7b: Post-hoc redaction example for sample #16 (S1, B0 has regex PII hit).*

| Baseline | Output excerpt | PII-Rx |
|----------|---------------|-------:|
| B0 | "The client's SSN (2372107-8821) is required" | 1 |
| B1 | "The client's SSN (2[REDACTED_PHONE]) is required" | 0 |
| B3 | (no digit tokens generated at sensitive positions) | 0 |

B1 catches the phone-like pattern via regex post-processing. However, the token-level forbidden rate is unchanged from B0. B3 prevents the pattern from forming in the first place.

### 5.6 Key Findings

**Finding 1: Unprotected diffusion emits PII-shaped tokens.** B0 produces forbidden tokens at 14.8% of sensitive positions (409/2,755; 95% CI [11.9%, 17.4%]). The model predicts digits, `@` signs, and PII-indicative patterns when given surrounding context, confirming the Carlini et al. (2021) extraction threat model for diffusion LMs.

**Finding 2: Post-hoc regex is necessary but insufficient.** B1 removes all regex-detectable PII patterns (PII-Rx drops from 0.10 to 0.00) but cannot prevent emission of PII-forming tokens. Only 9% of samples are affected because MDLM rarely generates complete regex-matching patterns. Regex scrubbing is reactive, not preventive.

**Finding 3: Projection provides an absolute guarantee.** B3--B5 achieve 0.0% forbidden tokens across all 2,755 sensitive positions, all three suites, and all 300 baseline-sample runs. The 95% CI is trivially [0.0%, 0.0%]. This is the empirical confirmation of Theorem 2.

**Finding 4: The schedule improves both efficiency and utility.** B4 achieves a 2.0x speedup over B3 (1.07s vs. 2.17s) and *improves* ROUGE-1 by +0.016 (0.537 vs. 0.521). The phased approach---establishing public context in the DRAFT phase before filling sensitive positions in the SAFE phase---produces more coherent text.

**Finding 5: Exact PII leakage is 0% for all baselines.** MDLM-OWT was trained on OpenWebText and has never seen the benchmark PII values. The forbidden token rate is therefore the more informative metric: it measures structural risk (emission of tokens *capable* of forming PII) rather than memorisation of specific secrets.

---

## 6 Discussion

### 6.1 Layered Safety Architecture

*Table 8: TPD's layered safety architecture.*

| Layer | Mechanism | Guarantee |
|-------|-----------|-----------|
| Token-level | Projection `Pi_{A(tau_i)}` | Type preservation, hard non-emission |
| Edit-level | Projection on T2T logits | Closure under editing |
| Temporal | Schedule intersection | Phase compliance |
| Sequence-level | Verifier `Ok_pi` | Policy compliance (patterns, structure) |
| System-level | Projection independent of `theta` | FL adapter safety |

No single layer suffices alone. Projection cannot detect multi-token patterns. The verifier cannot provide zero-probability guarantees. The schedule cannot constrain *which* tokens are written. The combination is necessary and sufficient for full policy satisfaction (Theorem 5).

### 6.2 Why Forbidden Token Rate is the Right Metric

Traditional PII leakage metrics (exact substring match, regex detection on output text) fail to differentiate baselines when the model does not memorise specific PII---the common case for models not fine-tuned on sensitive data.

The forbidden token rate measures a fundamentally different property: whether the model emits tokens from the *class* that could form PII at positions *typed as sensitive*. B0's 14.8% rate demonstrates that MDLM does produce PII-shaped tokens at sensitive positions when unconstrained, even without memorising specific secrets. The per-suite gradient (S1: 11.9%, S2: 15.6%, S3: 19.8%) shows that adversarial contexts amplify this risk.

### 6.3 ROUGE Inflation and SENS-only ROUGE

Full-text ROUGE scores (0.52--0.54) are inflated because approximately 52% of tokens are PUB and copied identically across all baselines. We therefore introduce SENS-only ROUGE-1, computed exclusively at sensitive positions. This metric is 0.000 for all baselines, indicating that the model generates entirely new content at masked positions with zero unigram overlap with the reference. This is expected: the diffusion model has never seen the benchmark PII values and fills sensitive positions with plausible but unrelated text.

Future work should evaluate on tasks where models must reconstruct specific *non-PII* content at sensitive positions to better characterise the utility cost of projection.

### 6.4 When Does the Verifier Add Value?

In our experiments, the verifier never triggered (0 rejections in 100 B5 samples). This is because our allowed sets already block all digit and PII-indicative tokens, making multi-token PII impossible to form. The verifier would add value with a more permissive projection---for example, if individual digit tokens were allowed but phone-number *patterns* were forbidden. Such a configuration would also produce non-zero B4-B5 differentiation.

### 6.5 Novelty

1. **First type-theoretic privacy framework for diffusion LMs.** Prior constrained decoding targets autoregressive models. TPD is the first to formalise type-level privacy guarantees for diffusion, leveraging position-parallel generation.

2. **Strictly stronger than DP for token-level guarantees.** DP provides epsilon-bounded statistical guarantees. TPD provides exactly-zero probability---an absolute guarantee no epsilon can match. The two are complementary (DP for training, TPD for inference).

3. **Parameter-independent safety.** No matter what the model weights are---original, fine-tuned, adversarially corrupted, or aggregated from Byzantine FL clients---the guarantee holds.

4. **Closed-form utility bound.** `D_KL = -log Z_i` provides an information-theoretic characterisation of the privacy-utility trade-off per position.

5. **Schedule-driven efficiency.** The three-phase schedule provides both defence-in-depth and a 2.0x speedup, demonstrating that stronger privacy can come with better performance.

---

## 7 Limitations

We identify nine limitations that should be considered when interpreting our results:

1. **Type assignment accuracy.** Formal guarantees are conditioned on the span typer's correctness. If the typer fails to detect a sensitive span (e.g., novel PII formats not covered by regex), that span is typed as PUB and receives no restriction.

2. **Allowed-set design.** If a forbidden token is inadvertently included in `A(tau_i)`, projection will not block it. Domain-specific deployments may require custom allowed sets beyond our default construction.

3. **Multi-token PII.** Projection operates token-by-token. Multi-token patterns (e.g., a phone number composed of individually-allowed digits) require the verifier layer. In our experiments the verifier never triggered (0/100 B5 samples), meaning its value is not empirically demonstrated here.

4. **ROUGE inflation.** Full-text ROUGE (0.52--0.54) is inflated by PUB token copying (~52% of tokens). SENS-only ROUGE is 0.000 for all baselines. This means our ROUGE-based utility comparison primarily reflects public-token preservation, not quality at sensitive positions.

5. **B4 = B5 identity.** The verifier never fires because projection already achieves 0% forbidden tokens. B5's value would manifest with weaker projection where multi-token violations can form.

6. **Evaluation model scale.** MDLM-OWT (170M params) is a legitimate diffusion model but is modestly sized and not fine-tuned on PII-containing data. Exact PII leakage is 0% for all baselines because the model has never seen benchmark PII values. Evaluation on PII-fine-tuned models or larger diffusion LMs (LLaDA 8B, LLaDA2.1 16B) would strengthen the empirical validation.

7. **Synthetic benchmark.** The 100-sample benchmark uses programmatically generated PII and template-based text. Real-world PII distributions and writing styles may differ.

8. **Floating-point arithmetic.** Proofs assume `exp(-inf) = 0`. The implementation uses `torch.finfo(dtype).min`, yielding `exp(finfo.min) = 0.0` exactly under IEEE 754. No known hardware violates this.

9. **Side channels.** Guarantees concern the sampled token sequence only and do not address side-channel leakage through timing, memory access patterns, or gradient information. DP-SGD should be used in conjunction for comprehensive privacy.

---

## 8 Broader Impact Statement

TPD is designed to prevent language models from emitting personally identifiable information (PII) at inference time. We believe this contributes positively to the deployment of language models in privacy-sensitive domains such as healthcare, finance, and personal computing.

**Positive impacts.** (1) Hard privacy guarantees that are verifiable and independent of model parameters reduce the risk of accidental PII disclosure. (2) Compatibility with federated learning enables privacy-preserving model training without centralising sensitive data. (3) Open-source release enables scrutiny, reproducibility, and adoption.

**Potential risks.** (1) TPD may create a false sense of security if the span typer fails to detect sensitive content, leading to gaps between the formal guarantee and the actual protection achieved. (2) Overly restrictive allowed sets could degrade text quality, potentially reducing the usefulness of models in domains that require technical vocabulary containing digits or special characters. (3) The framework currently targets diffusion models; organisations using autoregressive models would not benefit directly. (4) We note that our evaluation uses synthetic PII only; no real personal data was used in this work.

**Responsible use.** TPD should be deployed as one layer of a comprehensive privacy strategy, not as a standalone solution. We recommend combining TPD with differential privacy during training, access controls, and regular auditing of the span typer's coverage.

---

## 9 Conclusion

We have presented Typed Privacy Diffusion (TPD), a type-theoretic framework providing hard, deterministic token-level privacy guarantees for diffusion language models in federated learning settings. Our six formal theorems establish that: (1) well-typedness is an invariant of the transition system; (2) forbidden tokens have exactly zero emission probability; (3) repair preserves typing; (4) schedule phases enforce temporal restrictions; (5) projection plus verifier acceptance yields full policy compliance; (6) these guarantees hold regardless of model parameters, including adversarial FL adapters.

Empirical evaluation on MDLM-OWT (170M params, NeurIPS 2024) confirms that unprotected diffusion generates 14.8% forbidden tokens at sensitive positions (95% CI [11.9%, 17.4%]), which TPD reduces to exactly 0.0% with negligible utility cost (ROUGE-1: 0.537 for TPD vs. 0.526 for unprotected; the schedule actually improves utility) and a 2.0x inference speedup. We report honest analysis of nine limitations, including ROUGE inflation from PUB token copying and the verifier's redundancy when projection alone achieves full compliance.

The key insight is architectural: the projection operator is a deterministic function applied *after* all model computation and *before* sampling, making it independent of model parameters, training procedures, data distributions, and adapter updates. This transforms the privacy problem from a statistical estimation problem to a deterministic enforcement problem---and provides correspondingly stronger guarantees.

---

## References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. In *Proceedings of CCS 2016*, pp. 308--318.

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). Structured denoising diffusion models in discrete state-spaces. In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34.

Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., & Raffel, C. (2021). Extracting training data from large language models. In *USENIX Security Symposium*.

Charles, Z., Garrett, Z., Huo, Z., Kidambi, R., & Konecny, J. (2024). Towards federated foundation models: Scalable dataset pipelines for group-structured learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 37.

Hokamp, C., & Liu, Q. (2017). Lexically constrained decoding for sequence generation using grid beam search. In *Proceedings of ACL 2017*, pp. 1535--1546.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. In *Proceedings of ICLR 2022*.

Hu, J. E., Singh, A., Holzenberger, N., Post, M., & Van Durme, B. (2019). Improved lexically constrained decoding for translation and monolingual rewriting. In *Proceedings of NAACL 2019*, pp. 839--850.

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. In *ACL Workshop on Text Summarization Branches Out*.

Lison, P., Hubin, A., Barnes, J., & Touileb, S. (2021). Named entity recognition without labelled data: Weak supervision for NER with noise-aware learning. In *Proceedings of ACL-IJCNLP 2021*.

Lou, A., Meng, C., & Ermon, S. (2024). Discrete diffusion modeling by estimating the ratios of the data distribution. In *Proceedings of ICML 2024*.

McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. In *Proceedings of AISTATS 2017*, pp. 1273--1282.

McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning differentially private recurrent language models. In *Proceedings of ICLR 2018*.

Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J., & Li, P. (2025). Large language diffusion models. *arXiv preprint arXiv:2502.09992*.

Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. In *Proceedings of ACL 2002*, pp. 311--318.

Sahoo, S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Bruss, J. T., Fernandes, M., Haliassos, A., & Kuleshov, V. (2024). Simple and effective masked diffusion language models. In *Advances in Neural Information Processing Systems (NeurIPS)*.

---

## Appendix A: Notation

| Symbol | Meaning |
|--------|---------|
| `V` | Vocabulary (finite set; `\|V\| = 50,257` for MDLM-OWT) |
| `[MASK]` | Mask token (id = 50,257 for MDLM-OWT) |
| `L` | Sequence length |
| `T` | Type universe `{PUB, SENS, REG, DERIVED_*}` |
| `T_sens` | Sensitive types: `T \ {PUB}` |
| `tau : [L] -> T` | Type assignment function |
| `A : T -> P(V)` | Allowed-set function |
| `Pi_{A(tau_i)}` | Projection operator |
| `Ok_pi` | Verifier gate |
| `R` | Repair operator |
| `Z_i` | Allowed mass at position `i` |
| `S_t = (x_t, M_t, Gamma)` | TPD state at step `t` |
| `alpha, beta` | Schedule phase boundaries (default 0.4, 0.9) |
| `theta_adapter` | FL adapter parameters (LoRA matrices `A`, `B`) |

---

## Appendix B: Theorem-to-Code Mapping

Every theorem has corresponding implementation code and test coverage.

| Theorem | Code Location | Test File | Tests |
|---------|--------------|-----------|------:|
| T1 Type Preservation | `tpd/typing.py`, `diffusion/decode_loop.py` | `test_schedule.py` | 32 |
| T2 Hard Non-Emission | `tpd/projection.py` | `test_projection.py` | 40 |
| T3 Edit Closure | `tpd/repair.py` | `test_edit_closure.py` | 20 |
| T4 Schedule Compliance | `tpd/schedule.py` | `test_schedule.py` | 32 |
| T5 Verifier-Lifted Safety | `tpd/verifier.py` | `test_okpi.py` | 17 |
| T6 FL Adapter Safety | `fl/lora.py`, `tpd/projection.py` | `test_fl_invariance.py` | ~20 |
| Utility Bound (`Z_i`) | `tpd/diagnostics.py` | `test_invariants_property.py` | 79 |
| **Total** | | | **~210** |

Complete formal proofs are in `tpd_fl/proofs/tpd_semantics.tex`. The theorem-to-code mapping with line numbers is in `tpd_fl/proofs/mapping.md`.

---

## Appendix C: Implementation Details

### Repository Structure

```
tpd_fl/
  model/                  # Model backends (CPU-first)
    backend_base.py       # DiffusionBackend ABC + SyntheticBackend
    backend_hf_mdlm.py   # MDLM-OWT 170M backend (primary)
    modeling_mdlm.py      # MDLM model (CPU-patched, standard attention)
    configuration_mdlm.py # MDLM HuggingFace config
    backend_hf_bert.py    # BERT-base-uncased MLM backend (alternative)
    backend_hf_llada.py   # LLaDA 8B backend (Tier 2, GPU)
    backend_hf_llada2.py  # LLaDA2.1-mini 16B backend (Tier 3, GPU)
  tpd/                    # TPD core module
    typing.py             # Span typer (Def. 1)
    allowed_sets.py       # Allowed-set construction (Def. 2)
    projection.py         # Logit projection (Def. 3, Thms. 1--2)
    schedule.py           # Three-phase schedule (Def. 4)
    verifier.py           # Verifier gate (Def. 5, Thm. 5)
    repair.py             # Monotone repair (Def. 6, Thm. 3)
    diagnostics.py        # Z_i monitoring (Prop. 1)
  diffusion/              # Diffusion decode loop
    decode_loop.py        # M2T decode with TPD hooks
  fl/                     # Federated Learning
    lora.py               # LoRA adapters (Thm. 6)
    client.py, server.py  # FL client/server
    protocols.py          # FedAvg, FedAdam aggregation
    datasets.py           # Non-IID partitioning
  eval/                   # Evaluation suite
    empirical_eval.py     # MDLM evaluation pipeline (Sec. 5)
    metrics_real.py       # ROUGE, BLEU, PII metrics
    benchgen.py           # S1--S3 benchmark generator
  proofs/                 # Formal proofs
    tpd_semantics.tex     # Full LaTeX proofs (Sec. 4)
    mapping.md            # Theorem-to-code mapping
  tests/                  # ~210 pytest tests
```

### MDLM-OWT CPU Adaptation

The original MDLM-OWT model requires `flash_attn` (CUDA only). We replace `flash_attn_varlen_qkvpacked_func` with `F.scaled_dot_product_attention` and `flash_attn.layers.rotary` with a manual rotary embedding implementation. The adaptation is mathematically equivalent and loads the same pretrained weights with zero missing/unexpected keys.

---

## Appendix D: Reproducibility

### Installation

```bash
pip install torch transformers nltk safetensors einops huggingface-hub pytest
```

### Reproduce Main Results (Table 2)

```bash
python -m tpd_fl.eval.empirical_eval \
    --output-dir runs/empirical_mdlm \
    --steps 32 --seed 42 \
    --num-s1 50 --num-s2 30 --num-s3 20
```

Downloads MDLM-OWT (170M params) automatically. Outputs `metrics.json`, `table.csv`, `per_sample_results.json`. Total runtime: ~15 minutes on CPU.

### Run All Tests

```bash
pytest tpd_fl/tests/ -q    # ~210 tests
```

### Model Tiers

| Tier | Model | Device | Purpose |
|------|-------|--------|---------|
| **1 (default)** | MDLM-OWT (170M) | CPU | Empirical evaluation |
| 2 (optional) | LLaDA 8B | GPU | Scaling experiments |
| 3 (optional) | LLaDA2.1-mini 16B | GPU | Large-scale evaluation |
| CI | Synthetic backend | CPU | Unit tests (no weights) |

---

## Appendix E: NeurIPS Paper Checklist

1. **Claims.** All main claims are supported by theoretical proofs (Theorems 1--6) and empirical evidence (100-sample evaluation with 95% bootstrap CIs). Limitations are explicitly discussed in Section 7.

2. **Theoretical contributions.** All theorems include proof sketches in the main text and complete proofs in the supplementary material (`tpd_semantics.tex`). Assumptions (finite vocabulary, non-empty allowed sets, IEEE 754 arithmetic) are stated explicitly.

3. **Reproducibility.** (a) Complete source code is provided under MIT licence. (b) All hyperparameters are specified in Section 5.1. (c) A single command reproduces all results (Appendix D). (d) Model weights are publicly available on HuggingFace. (e) Random seed is fixed (`seed=42`).

4. **Open access.** Code, data generation scripts, and evaluation harnesses are open-source.

5. **Experimental methodology.** (a) All metrics are standard (ROUGE, BLEU, Distinct-N). (b) Bootstrap 95% CIs are reported for key metrics. (c) Per-suite breakdowns are provided. (d) Qualitative examples are shown.

6. **Ethics.** No real personal data was used. All PII in benchmarks is synthetically generated. The Broader Impact Statement (Section 8) discusses positive impacts and potential risks.

7. **Computational resources.** All experiments run on CPU in ~15 minutes. No GPU required. The MDLM-OWT model (170M params) requires ~700MB of storage.

---

## Licence

MIT
