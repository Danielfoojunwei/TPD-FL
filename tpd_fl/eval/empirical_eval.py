"""
Empirical Evaluation — runs MDLM (proper diffusion LM) benchmarks.

Model:  MDLM-OWT 170M (Sahoo et al., NeurIPS 2024) — a proper masked
        discrete diffusion language model trained with continuous noise
        schedules.  Open weights from HuggingFace.

Setup:  Carlini et al. (2021) context-completion extraction.
        Public tokens visible, sensitive positions masked, model predicts.

Corpus: 100 samples from BenchmarkGenerator (50 S1, 30 S2, 20 S3).

Fixes over previous version:
  1. Real diffusion model (MDLM, not BERT MLM proxy)
  2. B1 is meaningful: generates like B0, then regex-redacts TEXT, metrics
     measured on redacted output (different ROUGE, lower PII in text)
  3. B5 tracks verifier rejections + repair count (differentiated from B4)
  4. SENS-only ROUGE (not dominated by copied PUB tokens)
  5. 100 samples with bootstrap 95% confidence intervals
  6. regex_pii_in_output metric: PII-pattern matches in generated text

Usage::

    python -m tpd_fl.eval.empirical_eval --output-dir runs/empirical_mdlm
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from tpd_fl.tpd.typing import SpanType, SpanTyper, SENSITIVE_TYPES
from tpd_fl.tpd.allowed_sets import AllowedSetBuilder, AllowedSetConfig
from tpd_fl.tpd.schedule import MaskSchedule, ScheduleConfig
from tpd_fl.tpd.projection import project_logits, ProjectionEngine
from tpd_fl.tpd.verifier import Verifier, VerifierConfig
from tpd_fl.tpd.repair import RepairEngine, RepairMode
from tpd_fl.eval.benchgen import BenchmarkGenerator
from tpd_fl.eval.metrics_real import (
    EvalResult,
    evaluate_sample,
    rouge_n,
    rouge_l,
    bleu_score,
    detect_pii_leakage,
    compute_fluency,
)


# ---------------------------------------------------------------------------
# Sample normalization — BenchmarkGenerator uses different keys per suite
# ---------------------------------------------------------------------------

def _extract_secrets(sample: Dict) -> List[str]:
    """Extract secret values from a BenchmarkGenerator sample."""
    if "embedded_secrets" in sample:
        return [s["value"] for s in sample["embedded_secrets"]]
    elif "target_secrets" in sample:
        return [s["value"] for s in sample["target_secrets"]]
    elif "secrets" in sample:
        return sample["secrets"]
    return []


# ---------------------------------------------------------------------------
# Allowed-set mask construction (shared across baselines)
# ---------------------------------------------------------------------------

def _build_allowed_masks(backend, device="cpu"):
    """Build real allowed-set masks from the backend's vocabulary.

    Scans every token in the vocabulary and determines whether it
    should be allowed at SENS/REG positions.  Forbidden tokens are those
    whose decoded form contains digits, @ signs, or PII-indicative
    patterns (Carlini et al. 2021 style).
    """
    V = backend.vocab_size
    pub = torch.ones(V, dtype=torch.bool, device=device)
    sens = torch.ones(V, dtype=torch.bool, device=device)
    reg = torch.ones(V, dtype=torch.bool, device=device)

    for tid in range(V):
        try:
            decoded = backend.decode([tid])
        except Exception:
            continue
        d = decoded.strip().lower()

        has_digit = any(c.isdigit() for c in d)
        has_at = "@" in d
        has_dash_digit = bool(re.search(r"\d+-\d+", d))

        # SENS: block tokens that could form PII (digits, @, dash-digit)
        if has_digit or has_at or has_dash_digit:
            sens[tid] = False

        # REG: stricter — block all non-alphabetic tokens
        if has_digit or has_at or not d.isalpha() or len(d) < 2:
            reg[tid] = False

    if sens.sum() < 100:
        sens[:1000] = True
    if reg.sum() < 50:
        reg[:500] = True

    masks = {SpanType.PUB: pub, SpanType.SENS: sens, SpanType.REG: reg}
    for st in SENSITIVE_TYPES:
        if st not in masks:
            masks[st] = sens.clone()
    return masks


# ---------------------------------------------------------------------------
# Context-completion setup (Carlini et al. 2021)
# ---------------------------------------------------------------------------

def _prepare_context_completion(backend, sample, seq_len):
    """Prepare the context-completion setup (Carlini et al. 2021).

    Returns tokens with PUB positions set to original values and
    SENS/REG positions set to [MASK].  Uses BOTH the SpanTyper regex
    detection AND ground-truth secret positions from the benchmark.
    """
    text = sample["source_text"]
    tok = backend.tokenize(text, max_length=seq_len, return_offsets=True)
    input_ids = tok["input_ids"].squeeze(0)
    offset_mapping = tok.get("offset_mapping", [(i, i + 1) for i in range(len(input_ids))])
    L = input_ids.shape[0]

    typer = SpanTyper()
    spans, pos_type, pos_span_id = typer.type_text(text, offset_mapping, L)

    # Additionally mark ground-truth secret positions as SENS.
    secrets = _extract_secrets(sample)
    for secret in secrets:
        start_idx = 0
        while True:
            pos = text.lower().find(secret.lower(), start_idx)
            if pos == -1:
                break
            char_start = pos
            char_end = pos + len(secret)
            for tok_idx, (ts, te) in enumerate(offset_mapping):
                if tok_idx >= L:
                    break
                if te <= char_start:
                    continue
                if ts >= char_end:
                    break
                if pos_type[tok_idx] == SpanType.PUB:
                    pos_type[tok_idx] = SpanType.SENS
            start_idx = pos + 1

    # Context-completion: original tokens at PUB, [MASK] at SENS/REG
    tokens = input_ids.clone().to(backend.device)
    orig_tokens = input_ids.clone().to(backend.device)
    sens_positions = []
    for i in range(L):
        if pos_type[i] in SENSITIVE_TYPES:
            tokens[i] = backend.mask_token_id
            sens_positions.append(i)

    return tokens, orig_tokens, pos_type, sens_positions, spans, offset_mapping, L


def _count_forbidden(tokens, sens_positions, pos_type, allowed_masks):
    """Count tokens at sensitive positions that violate allowed sets."""
    n_sens = len(sens_positions)
    n_forbidden = 0
    for i in sens_positions:
        stype = pos_type[i]
        mask = allowed_masks.get(stype, allowed_masks.get(SpanType.SENS))
        if mask is None:
            continue
        tid = int(tokens[i])
        if tid < mask.shape[0] and not mask[tid]:
            n_forbidden += 1
    return n_sens, n_forbidden


def _extract_sens_text(backend, tokens, orig_tokens, sens_positions):
    """Extract text at sensitive positions for SENS-only ROUGE."""
    if not sens_positions:
        return "", ""
    hyp_ids = [int(tokens[p]) for p in sens_positions]
    ref_ids = [int(orig_tokens[p]) for p in sens_positions]
    hyp_text = backend.detokenize(torch.tensor(hyp_ids))
    ref_text = backend.detokenize(torch.tensor(ref_ids))
    return hyp_text, ref_text


# ---------------------------------------------------------------------------
# Baseline runners — proper iterative diffusion with MDLM
# ---------------------------------------------------------------------------

def _diffusion_loop(
    backend, tokens, sens_positions, total_steps, temperature, gen,
    project_fn=None, schedule=None, pos_type=None, L=None,
):
    """Core diffusion denoising loop.

    At each step:
      1. Identify still-masked sensitive positions
      2. Apply schedule filtering (if provided)
      3. Get logits from MDLM
      4. Apply projection (if provided)
      5. Sample and fill positions
    """
    sens_tensor = torch.tensor(sens_positions, dtype=torch.long, device=backend.device)

    for t in range(total_steps):
        is_mask = tokens[sens_tensor] == backend.mask_token_id
        if not is_mask.any():
            # All unmasked — do one final refinement pass
            positions = sens_tensor
        else:
            positions = sens_tensor[is_mask]

        n_remain = len(positions)
        n_to_unmask = max(1, n_remain // max(1, total_steps - t))
        if n_remain > n_to_unmask:
            perm = torch.randperm(n_remain, generator=gen, device=backend.device)
            positions = positions[perm[:n_to_unmask]]

        # Schedule filtering
        if schedule is not None and pos_type is not None and L is not None:
            proposed_mask = torch.zeros(L, dtype=torch.bool, device=backend.device)
            proposed_mask[positions] = True
            final_mask = schedule.apply_schedule(proposed_mask, t, total_steps, pos_type)
            positions = final_mask.nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                continue

        logits = backend.forward_logits(
            tokens, step=t, positions=positions, total_steps=total_steps)

        # Projection
        if project_fn is not None and pos_type is not None:
            logits = project_fn(logits, positions, pos_type)

        sampled = backend.sample_tokens(logits, temperature=temperature, generator=gen)
        tokens[positions] = sampled

    return tokens


def run_b0_unprotected(
    backend, sample, seq_len, total_steps, temperature, seed,
    allowed_masks,
) -> Tuple[str, float, bool, int, int, str, str]:
    """B0: No protection — model predicts sensitive positions freely."""
    tokens, orig_tokens, pos_type, sens_positions, _, _, L = \
        _prepare_context_completion(backend, sample, seq_len)

    if not sens_positions:
        output = backend.detokenize(tokens)
        return output, 0.0, True, 0, 0, "", ""

    gen = torch.Generator(device=backend.device)
    gen.manual_seed(seed)

    start = time.time()
    tokens = _diffusion_loop(
        backend, tokens, sens_positions, total_steps, temperature, gen)
    elapsed = time.time() - start

    output = backend.detokenize(tokens)
    n_sens, n_forbidden = _count_forbidden(tokens, sens_positions, pos_type, allowed_masks)
    sens_hyp, sens_ref = _extract_sens_text(backend, tokens, orig_tokens, sens_positions)
    return output, elapsed, True, n_sens, n_forbidden, sens_hyp, sens_ref


def run_b1_posthoc_redact(
    backend, sample, seq_len, total_steps, temperature, seed,
    allowed_masks,
) -> Tuple[str, float, bool, int, int, str, str]:
    """B1: Generate freely, then apply post-hoc regex redaction to TEXT.

    Crucially, metrics are measured on the REDACTED text, not the raw output.
    This makes B1 meaningfully different from B0:
      - Same forbidden_token_rate (tokens were generated identically)
      - Different ROUGE (text has been modified by redaction)
      - Lower regex_pii_in_output (regex removes detected patterns)
      - Demonstrates weakness: regex can miss novel PII formats
    """
    output, elapsed, _, n_sens, n_forbidden, sens_hyp, sens_ref = run_b0_unprotected(
        backend, sample, seq_len, total_steps, temperature, seed, allowed_masks)

    # Post-hoc regex redaction on the text
    redacted = output
    redacted = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "[REDACTED_EMAIL]", redacted)
    redacted = re.sub(
        r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",
        "[REDACTED_SSN]", redacted)
    redacted = re.sub(
        r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}",
        "[REDACTED_PHONE]", redacted)
    redacted = re.sub(
        r"\b(?:\d{4}[\s.-]?){3}\d{4}\b",
        "[REDACTED_CC]", redacted)

    # Return REDACTED text as the output (metrics measured on this)
    return redacted, elapsed, True, n_sens, n_forbidden, sens_hyp, sens_ref


def run_b3_projection(
    backend, sample, seq_len, total_steps, temperature, seed,
    allowed_masks,
) -> Tuple[str, float, bool, int, int, str, str]:
    """B3: TPD projection only — hard guarantee on logits."""
    tokens, orig_tokens, pos_type, sens_positions, _, _, L = \
        _prepare_context_completion(backend, sample, seq_len)
    proj = ProjectionEngine(allowed_masks, pos_type)

    if not sens_positions:
        output = backend.detokenize(tokens)
        return output, 0.0, True, 0, 0, "", ""

    gen = torch.Generator(device=backend.device)
    gen.manual_seed(seed)

    def project_fn(logits, positions, pt):
        local_types = [pt[p] for p in positions.tolist()]
        return project_logits(
            logits, local_types, allowed_masks,
            positions=torch.arange(len(positions), device=backend.device))

    start = time.time()
    tokens = _diffusion_loop(
        backend, tokens, sens_positions, total_steps, temperature, gen,
        project_fn=project_fn, pos_type=pos_type)
    elapsed = time.time() - start

    output = backend.detokenize(tokens)
    hard_ok = proj.verify_hard_guarantee(tokens)
    n_sens, n_forbidden = _count_forbidden(tokens, sens_positions, pos_type, allowed_masks)
    sens_hyp, sens_ref = _extract_sens_text(backend, tokens, orig_tokens, sens_positions)
    return output, elapsed, hard_ok, n_sens, n_forbidden, sens_hyp, sens_ref


def run_b4_projection_schedule(
    backend, sample, seq_len, total_steps, temperature, seed,
    allowed_masks,
) -> Tuple[str, float, bool, int, int, str, str]:
    """B4: TPD projection + schedule."""
    tokens, orig_tokens, pos_type, sens_positions, _, _, L = \
        _prepare_context_completion(backend, sample, seq_len)
    proj = ProjectionEngine(allowed_masks, pos_type)
    schedule = MaskSchedule(ScheduleConfig(draft_end=0.4, safe_end=0.9))

    if not sens_positions:
        output = backend.detokenize(tokens)
        return output, 0.0, True, 0, 0, "", ""

    gen = torch.Generator(device=backend.device)
    gen.manual_seed(seed)

    def project_fn(logits, positions, pt):
        local_types = [pt[p] for p in positions.tolist()]
        return project_logits(
            logits, local_types, allowed_masks,
            positions=torch.arange(len(positions), device=backend.device))

    start = time.time()
    tokens = _diffusion_loop(
        backend, tokens, sens_positions, total_steps, temperature, gen,
        project_fn=project_fn, schedule=schedule, pos_type=pos_type, L=L)
    elapsed = time.time() - start

    output = backend.detokenize(tokens)
    hard_ok = proj.verify_hard_guarantee(tokens)
    n_sens, n_forbidden = _count_forbidden(tokens, sens_positions, pos_type, allowed_masks)
    sens_hyp, sens_ref = _extract_sens_text(backend, tokens, orig_tokens, sens_positions)
    return output, elapsed, hard_ok, n_sens, n_forbidden, sens_hyp, sens_ref


def run_b5_full_tpd(
    backend, sample, seq_len, total_steps, temperature, seed,
    allowed_masks,
) -> Tuple[str, float, bool, int, int, str, str, int, int]:
    """B5: Full TPD — projection + schedule + verifier + repair.

    Returns extra fields: verifier_rejections, repair_count.
    """
    tokens, orig_tokens, pos_type, sens_positions, _, _, L = \
        _prepare_context_completion(backend, sample, seq_len)
    proj = ProjectionEngine(allowed_masks, pos_type)
    schedule = MaskSchedule(ScheduleConfig(draft_end=0.4, safe_end=0.9))

    secrets = _extract_secrets(sample)
    verifier = Verifier(VerifierConfig(
        forbidden_tags=["EMAIL", "PHONE", "SSN", "CC", "ID"],
        known_secrets=secrets,
    ))
    repair = RepairEngine(mode=RepairMode.RESAMPLE)

    if not sens_positions:
        output = backend.detokenize(tokens)
        return output, 0.0, True, 0, 0, "", "", 0, 0

    gen = torch.Generator(device=backend.device)
    gen.manual_seed(seed)
    sens_tensor = torch.tensor(sens_positions, dtype=torch.long, device=backend.device)

    verifier_rejections = 0
    repair_count = 0

    start = time.time()
    T = total_steps
    for t in range(T):
        is_mask = tokens[sens_tensor] == backend.mask_token_id
        if not is_mask.any():
            proposed_positions = sens_tensor
        else:
            proposed_positions = sens_tensor[is_mask]

        n_to_unmask = max(1, len(proposed_positions) // max(1, T - t))
        if len(proposed_positions) > n_to_unmask:
            perm = torch.randperm(len(proposed_positions), generator=gen, device=backend.device)
            proposed_positions = proposed_positions[perm[:n_to_unmask]]

        proposed_mask = torch.zeros(L, dtype=torch.bool, device=backend.device)
        proposed_mask[proposed_positions] = True
        final_mask = schedule.apply_schedule(proposed_mask, t, T, pos_type)
        positions = final_mask.nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            continue

        logits = backend.forward_logits(
            tokens, step=t, positions=positions, total_steps=T)
        local_types = [pos_type[p] for p in positions.tolist()]
        logits = project_logits(
            logits, local_types, allowed_masks,
            positions=torch.arange(len(positions), device=backend.device))
        sampled = backend.sample_tokens(logits, temperature=temperature, generator=gen)
        tokens[positions] = sampled

        # Verifier + repair every 4 steps
        if t % 4 == 0:
            decoded = backend.detokenize(tokens)
            vr = verifier.check(decoded)
            if not vr.ok:
                verifier_rejections += 1
                violating = []
                for v in vr.violations:
                    if "positions" in v:
                        violating.extend(v["positions"])
                if violating:
                    repair_count += 1

                    def model_fn(tok, s, pos, cond=None):
                        return backend.forward_logits(tok, s, pos, total_steps=T)

                    tokens, _ = repair.repair(
                        tokens, violating, pos_type, allowed_masks,
                        model_fn=model_fn, step=t,
                        mask_token_id=backend.mask_token_id,
                        temperature=temperature,
                    )

    elapsed = time.time() - start
    output = backend.detokenize(tokens)
    hard_ok = proj.verify_hard_guarantee(tokens)
    n_sens, n_forbidden = _count_forbidden(tokens, sens_positions, pos_type, allowed_masks)
    sens_hyp, sens_ref = _extract_sens_text(backend, tokens, orig_tokens, sens_positions)
    return output, elapsed, hard_ok, n_sens, n_forbidden, sens_hyp, sens_ref, verifier_rejections, repair_count


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

BASELINE_FUNCTIONS = {
    "B0": ("Unprotected", run_b0_unprotected),
    "B1": ("Post-hoc redaction", run_b1_posthoc_redact),
    "B3": ("TPD projection", run_b3_projection),
    "B4": ("TPD proj+schedule", run_b4_projection_schedule),
    "B5": ("TPD full", run_b5_full_tpd),
}


def run_empirical_eval(
    output_dir: str = "runs/empirical_mdlm",
    baselines: Optional[List[str]] = None,
    total_steps: int = 32,
    seq_len: int = 128,
    temperature: float = 0.9,
    seed: int = 42,
    num_s1: int = 50,
    num_s2: int = 30,
    num_s3: int = 20,
) -> Dict[str, Any]:
    """Run full empirical evaluation with MDLM and return results."""
    from tpd_fl.model.backend_hf_mdlm import HFMDLMBackend
    from tpd_fl.model.backend_base import BackendConfig

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    baselines_list = baselines or list(BASELINE_FUNCTIONS.keys())

    # Generate benchmark corpus
    print("Generating benchmark corpus...")
    benchgen = BenchmarkGenerator()
    suites = benchgen.generate_all(
        num_s1=num_s1, num_s2=num_s2, num_s3=num_s3, seed=seed)
    samples = suites["S1"] + suites["S2"] + suites["S3"]
    n_s1 = len(suites["S1"])
    n_s2 = len(suites["S2"])
    n_s3 = len(suites["S3"])

    print("=" * 70)
    print("  TPD+FL Empirical Evaluation")
    print("  Model: MDLM-OWT (170M params, NeurIPS 2024, open weights)")
    print(f"  Baselines: {baselines_list}")
    print(f"  Samples: {len(samples)} (S1={n_s1}, S2={n_s2}, S3={n_s3})")
    print(f"  Steps: {total_steps}, Seq len: {seq_len}, Temp: {temperature}")
    print("=" * 70)

    # Load real diffusion model
    print("\nLoading MDLM-OWT (170M params, proper diffusion LM)...")
    load_start = time.time()
    backend = HFMDLMBackend(BackendConfig(
        model_id="kuleshov-group/mdlm-owt",
        device="cpu",
        max_seq_len=seq_len,
    ))
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s  (vocab={backend.vocab_size}, "
          f"mask_id={backend.mask_token_id})")

    # Build allowed masks ONCE (shared across all baselines and samples)
    print("Building allowed-set masks from vocabulary...")
    mask_start = time.time()
    allowed_masks = _build_allowed_masks(backend, str(backend.device))
    mask_time = time.time() - mask_start
    sens_allowed = int(allowed_masks[SpanType.SENS].sum().item())
    sens_blocked = backend.vocab_size - sens_allowed
    reg_allowed = int(allowed_masks[SpanType.REG].sum().item())
    reg_blocked = backend.vocab_size - reg_allowed
    print(f"Masks built in {mask_time:.1f}s")
    print(f"  PUB:  {backend.vocab_size}/{backend.vocab_size} allowed")
    print(f"  SENS: {sens_allowed}/{backend.vocab_size} allowed, "
          f"{sens_blocked} blocked")
    print(f"  REG:  {reg_allowed}/{backend.vocab_size} allowed, "
          f"{reg_blocked} blocked")
    print(f"\nSetup: Context-completion (Carlini et al. 2021)")
    print(f"  Public tokens visible, sensitive positions [MASK]ed")

    # Run baselines
    all_results: Dict[str, List[EvalResult]] = defaultdict(list)

    for bname in baselines_list:
        label, fn = BASELINE_FUNCTIONS[bname]
        print(f"\n--- Running {bname}: {label} ({len(samples)} samples) ---")
        b_start = time.time()

        for si, sample in enumerate(samples):
            secrets = _extract_secrets(sample)

            if bname == "B5":
                result_tuple = fn(
                    backend, sample, seq_len, total_steps, temperature, seed,
                    allowed_masks)
                output_text, elapsed, hard_ok, n_sens, n_forbidden, \
                    sens_hyp, sens_ref, v_rej, r_count = result_tuple
            else:
                result_tuple = fn(
                    backend, sample, seq_len, total_steps, temperature, seed,
                    allowed_masks)
                output_text, elapsed, hard_ok, n_sens, n_forbidden, \
                    sens_hyp, sens_ref = result_tuple
                v_rej, r_count = 0, 0

            ref_text = sample.get("expected_public", sample["source_text"])
            result = evaluate_sample(
                baseline=bname,
                sample_id=sample["sample_id"],
                suite=sample["suite"],
                output_text=output_text,
                reference_text=ref_text,
                known_secrets=secrets,
                elapsed_sec=elapsed,
                hard_guarantee_holds=hard_ok,
                n_sens_positions=n_sens,
                n_forbidden_emitted=n_forbidden,
                sens_text_hyp=sens_hyp,
                sens_text_ref=sens_ref,
                verifier_rejections=v_rej,
                repair_count=r_count,
            )
            all_results[bname].append(result)

            if (si + 1) % 20 == 0 or si == len(samples) - 1:
                n_done = si + 1
                avg_f = statistics.mean(
                    r.forbidden_token_rate for r in all_results[bname][:n_done])
                print(f"  [{n_done}/{len(samples)}] "
                      f"avg_forbid={avg_f*100:.1f}% "
                      f"({time.time()-b_start:.1f}s)")

        b_elapsed = time.time() - b_start
        n = len(all_results[bname])
        avg_forbidden = statistics.mean(r.forbidden_token_rate for r in all_results[bname])
        avg_rouge = statistics.mean(r.rouge1_f1 for r in all_results[bname])
        total_forbidden = sum(r.n_forbidden_emitted for r in all_results[bname])
        total_sens = sum(r.n_sens_positions for r in all_results[bname])
        avg_pii = statistics.mean(r.regex_pii_in_output for r in all_results[bname])
        print(f"  {n} samples in {b_elapsed:.1f}s | "
              f"forbidden={total_forbidden}/{total_sens} ({avg_forbidden*100:.1f}%) | "
              f"avg_rouge1={avg_rouge:.4f} | avg_pii_regex={avg_pii:.1f}")

    # Aggregate and save results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    agg = _aggregate_results(all_results)
    _print_table(agg)
    _save_results(out, all_results, agg, {
        "model": "MDLM-OWT (kuleshov-group/mdlm-owt)",
        "model_params": "170M",
        "model_type": "discrete masked diffusion LM (NeurIPS 2024)",
        "total_steps": total_steps,
        "seq_len": seq_len,
        "temperature": temperature,
        "seed": seed,
        "num_samples": len(samples),
        "num_s1": n_s1,
        "num_s2": n_s2,
        "num_s3": n_s3,
        "baselines": baselines_list,
    })

    print(f"\nResults saved to {out}/")
    return agg


# ---------------------------------------------------------------------------
# Aggregation with bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for mean."""
    if len(values) < 2:
        m = values[0] if values else 0.0
        return m, m, m
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        boot = [rng.choice(values) for _ in range(n)]
        means.append(statistics.mean(boot))
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return statistics.mean(values), lo, hi


def _aggregate_results(
    all_results: Dict[str, List[EvalResult]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-sample results into per-baseline statistics with CIs."""
    agg = {}
    for bname, results in all_results.items():
        n = len(results)
        if n == 0:
            agg[bname] = {}
            continue

        leak_rates = [r.exact_leak_rate for r in results]
        forbidden_rates = [r.forbidden_token_rate for r in results]
        n_sens_total = sum(r.n_sens_positions for r in results)
        n_forbidden_total = sum(r.n_forbidden_emitted for r in results)
        r1 = [r.rouge1_f1 for r in results]
        r2 = [r.rouge2_f1 for r in results]
        rl = [r.rougeL_f1 for r in results]
        bl = [r.bleu for r in results]
        sr1 = [r.sens_rouge1_f1 for r in results]
        d1 = [r.distinct_1 for r in results]
        d2 = [r.distinct_2 for r in results]
        elapsed = [r.elapsed_sec for r in results]
        hard_ok = [r.hard_guarantee_holds for r in results]
        pii_regex = [r.regex_pii_in_output for r in results]
        v_rej = [r.verifier_rejections for r in results]
        r_count = [r.repair_count for r in results]

        # Bootstrap CIs for key metrics
        forbid_mean, forbid_lo, forbid_hi = _bootstrap_ci(forbidden_rates)
        r1_mean, r1_lo, r1_hi = _bootstrap_ci(r1)
        sr1_mean, sr1_lo, sr1_hi = _bootstrap_ci(sr1)

        # Per-suite breakdown
        suite_forbidden = defaultdict(list)
        suite_pii = defaultdict(list)
        for r in results:
            suite_forbidden[r.suite].append(r.forbidden_token_rate)
            suite_pii[r.suite].append(r.regex_pii_in_output)

        agg[bname] = {
            "n": n,
            "leak_rate_mean": statistics.mean(leak_rates),
            "forbidden_token_rate_mean": forbid_mean,
            "forbidden_token_rate_ci95": (forbid_lo, forbid_hi),
            "n_sens_total": n_sens_total,
            "n_forbidden_total": n_forbidden_total,
            "forbidden_token_rate_global": n_forbidden_total / n_sens_total if n_sens_total > 0 else 0.0,
            "rouge1_f1_mean": r1_mean,
            "rouge1_f1_ci95": (r1_lo, r1_hi),
            "rouge2_f1_mean": statistics.mean(r2),
            "rougeL_f1_mean": statistics.mean(rl),
            "bleu_mean": statistics.mean(bl),
            "sens_rouge1_f1_mean": sr1_mean,
            "sens_rouge1_f1_ci95": (sr1_lo, sr1_hi),
            "distinct_1_mean": statistics.mean(d1),
            "distinct_2_mean": statistics.mean(d2),
            "elapsed_mean": statistics.mean(elapsed),
            "elapsed_total": sum(elapsed),
            "hard_guarantee_pct": sum(hard_ok) / n * 100,
            "regex_pii_in_output_mean": statistics.mean(pii_regex),
            "regex_pii_in_output_total": sum(pii_regex),
            "verifier_rejections_total": sum(v_rej),
            "repair_count_total": sum(r_count),
            "suite_forbidden_rates": {
                suite: statistics.mean(rates)
                for suite, rates in sorted(suite_forbidden.items())
            },
            "suite_pii_regex": {
                suite: statistics.mean(counts)
                for suite, counts in sorted(suite_pii.items())
            },
        }
    return agg


def _print_table(agg: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted results table."""
    print(f"\n{'Baseline':<8} {'Forbid%':>8} {'Forbid':>8} "
          f"{'PII-Rx':>7} {'Leak%':>7} "
          f"{'R-1':>6} {'R-1(S)':>7} "
          f"{'R-L':>6} {'BLEU':>6} "
          f"{'D-1':>5} {'s/samp':>7} {'Hard%':>6} {'VRej':>5} {'Rep':>4}")
    print("-" * 110)
    for bname in sorted(agg.keys()):
        m = agg[bname]
        if not m:
            continue
        forbid_str = f"{m['n_forbidden_total']}/{m['n_sens_total']}"
        print(f"{bname:<8} "
              f"{m['forbidden_token_rate_global']*100:>7.1f}% "
              f"{forbid_str:>8s} "
              f"{m['regex_pii_in_output_mean']:>6.1f} "
              f"{m['leak_rate_mean']*100:>6.1f}% "
              f"{m['rouge1_f1_mean']:>6.3f} "
              f"{m['sens_rouge1_f1_mean']:>6.3f} "
              f"{m['rougeL_f1_mean']:>6.3f} "
              f"{m['bleu_mean']:>6.4f} "
              f"{m['distinct_1_mean']:>5.2f} "
              f"{m['elapsed_mean']:>6.2f}s "
              f"{m['hard_guarantee_pct']:>5.0f}% "
              f"{m['verifier_rejections_total']:>5d} "
              f"{m['repair_count_total']:>4d}")

    # CIs
    print(f"\n95% Bootstrap Confidence Intervals:")
    for bname in sorted(agg.keys()):
        m = agg[bname]
        if not m:
            continue
        flo, fhi = m["forbidden_token_rate_ci95"]
        rlo, rhi = m["rouge1_f1_ci95"]
        slo, shi = m["sens_rouge1_f1_ci95"]
        print(f"  {bname}: Forbid% [{flo*100:.1f}%, {fhi*100:.1f}%]  "
              f"R-1 [{rlo:.3f}, {rhi:.3f}]  "
              f"R-1(S) [{slo:.3f}, {shi:.3f}]")

    # Per-suite breakdown
    print(f"\nPer-suite forbidden token rates:")
    for bname in sorted(agg.keys()):
        m = agg[bname]
        if not m:
            continue
        suite_str = ", ".join(
            f"{s}={r*100:.1f}%"
            for s, r in sorted(m.get("suite_forbidden_rates", {}).items()))
        print(f"  {bname}: {suite_str}")

    # Per-suite PII regex
    print(f"\nPer-suite PII regex matches in output (mean):")
    for bname in sorted(agg.keys()):
        m = agg[bname]
        if not m:
            continue
        suite_str = ", ".join(
            f"{s}={r:.2f}"
            for s, r in sorted(m.get("suite_pii_regex", {}).items()))
        print(f"  {bname}: {suite_str}")


def _save_results(
    out: Path,
    all_results: Dict[str, List[EvalResult]],
    agg: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """Save all results to disk."""
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Convert CI tuples to lists for JSON
    agg_json = {}
    for bname, m in agg.items():
        m2 = {}
        for k, v in m.items():
            if isinstance(v, tuple):
                m2[k] = list(v)
            else:
                m2[k] = v
        agg_json[bname] = m2

    with open(out / "metrics.json", "w") as f:
        json.dump(agg_json, f, indent=2)

    # Per-sample results
    per_sample = {}
    for bname, results in all_results.items():
        per_sample[bname] = []
        for r in results:
            d = asdict(r)
            d["output_text_preview"] = d.pop("output_text")[:200]
            d["sens_text_hyp"] = d.get("sens_text_hyp", "")[:100]
            d["sens_text_ref"] = d.get("sens_text_ref", "")[:100]
            per_sample[bname].append(d)

    with open(out / "per_sample_results.json", "w") as f:
        json.dump(per_sample, f, indent=2)

    # CSV table
    import csv
    with open(out / "table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "baseline", "n",
            "forbidden_token_rate", "n_forbidden", "n_sens",
            "regex_pii_mean", "leak_rate_mean",
            "rouge1_f1", "sens_rouge1_f1",
            "rouge2_f1", "rougeL_f1", "bleu",
            "distinct_1", "distinct_2",
            "elapsed_mean_s", "hard_guarantee_pct",
            "verifier_rejections", "repair_count",
        ])
        for bname in sorted(agg.keys()):
            m = agg[bname]
            if not m:
                continue
            writer.writerow([
                bname, m["n"],
                f"{m['forbidden_token_rate_global']:.4f}",
                m["n_forbidden_total"], m["n_sens_total"],
                f"{m['regex_pii_in_output_mean']:.2f}",
                f"{m['leak_rate_mean']:.4f}",
                f"{m['rouge1_f1_mean']:.4f}",
                f"{m['sens_rouge1_f1_mean']:.4f}",
                f"{m['rouge2_f1_mean']:.4f}",
                f"{m['rougeL_f1_mean']:.4f}",
                f"{m['bleu_mean']:.6f}",
                f"{m['distinct_1_mean']:.4f}",
                f"{m['distinct_2_mean']:.4f}",
                f"{m['elapsed_mean']:.3f}",
                f"{m['hard_guarantee_pct']:.0f}",
                m["verifier_rejections_total"],
                m["repair_count_total"],
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TPD+FL Empirical Evaluation (MDLM diffusion LM)",
    )
    parser.add_argument("--output-dir", default="runs/empirical_mdlm")
    parser.add_argument("--baselines", nargs="+",
                        default=["B0", "B1", "B3", "B4", "B5"])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-s1", type=int, default=50)
    parser.add_argument("--num-s2", type=int, default=30)
    parser.add_argument("--num-s3", type=int, default=20)
    args = parser.parse_args()

    run_empirical_eval(
        output_dir=args.output_dir,
        baselines=args.baselines,
        total_steps=args.steps,
        seq_len=args.seq_len,
        temperature=args.temperature,
        seed=args.seed,
        num_s1=args.num_s1,
        num_s2=args.num_s2,
        num_s3=args.num_s3,
    )


if __name__ == "__main__":
    main()
