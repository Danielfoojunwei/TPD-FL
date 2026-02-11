"""
Baseline Implementations — B0 through B7 for the TPD+FL ablation study.

Each baseline function takes a model, input text, and configuration, and
returns a dict with ``output_text`` and ``metrics``.  The baselines form
a progression from no protection (B0) to full TPD+FL with typed privacy
diffusion and federated adapter (B7):

  B0  Unprotected         — no privacy mechanism at all.
  B1  Post-hoc redaction  — generate freely, regex-replace after.
  B2  AR logit mask       — autoregressive rolling regex mask (approximate).
  B3  TPD projection only — apply Π but no schedule or repair.
  B4  TPD + schedule      — projection with 3-phase mask schedule.
  B5  TPD + schedule + repair — full TPD pipeline.
  B6  TPD + FL            — add federated adapter (type-agnostic).
  B7  TPD + FL + typed    — full system with typed FL adapter.

The :class:`BaselineRunner` orchestrates all baselines and collects
results for evaluation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from tpd_fl.tpd.typing import SpanType, SpanTyper, SENSITIVE_TYPES
from tpd_fl.tpd.allowed_sets import AllowedSetBuilder, AllowedSetConfig
from tpd_fl.tpd.schedule import MaskSchedule, ScheduleConfig, build_random_block_mask
from tpd_fl.tpd.projection import ProjectionEngine, project_logits
from tpd_fl.tpd.verifier import Verifier, VerifierConfig
from tpd_fl.tpd.repair import RepairEngine, RepairMode
from tpd_fl.tpd.diagnostics import compute_allowed_mass, compute_z_stats

from tpd_fl.eval.leakage import LeakageEvaluator, STANDARD_PATTERNS
from tpd_fl.eval.utility import UtilityEvaluator
from tpd_fl.eval.speed import SpeedTracker


# ---------------------------------------------------------------------------
# Standard redaction patterns for B1 (post-hoc)
# ---------------------------------------------------------------------------

_REDACTION_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[CC]"),
    (re.compile(r"\b[A-Z]{1,3}\d{6,10}\b"), "[ID]"),
]


# ---------------------------------------------------------------------------
# Baseline configuration
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    """Configuration shared across all baselines."""
    # Diffusion parameters
    total_steps: int = 64
    seq_len: int = 128
    tokens_per_step_frac: float = 0.15
    temperature: float = 1.0
    seed: int = 42

    # Schedule parameters (for B4+)
    schedule_draft_end: float = 0.4
    schedule_safe_end: float = 0.9

    # Repair parameters (for B5+)
    repair_mode: str = "resample"
    repair_max_iters: int = 3

    # Verifier
    verifier_forbidden_tags: List[str] = field(
        default_factory=lambda: ["EMAIL", "PHONE", "SSN", "CC", "ID"]
    )
    known_secrets: List[str] = field(default_factory=list)

    # FL adapter parameters (for B6+)
    fl_enabled: bool = False
    fl_typed: bool = False


# ---------------------------------------------------------------------------
# Helper: simple tokenizer for baselines (reuse from decode_loop)
# ---------------------------------------------------------------------------

class _SimpleTokenizer:
    """Minimal character-level tokenizer for baseline runs."""

    def __init__(self, vocab_size: int = 32000):
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [min(ord(c), self._vocab_size - 1) for c in text]

    def decode(self, ids) -> str:
        chars = []
        for i in ids:
            i = int(i)
            if 32 <= i < 127:
                chars.append(chr(i))
            else:
                chars.append(f"[{i}]")
        return "".join(chars)

    def __len__(self) -> int:
        return self._vocab_size


# ---------------------------------------------------------------------------
# Helper: run a basic diffusion decode loop
# ---------------------------------------------------------------------------

def _run_decode_loop(
    model,
    text: str,
    config: BaselineConfig,
    tokenizer=None,
    projection_engine: Optional[ProjectionEngine] = None,
    schedule: Optional[MaskSchedule] = None,
    pos_type: Optional[List[SpanType]] = None,
    allowed_masks: Optional[Dict[SpanType, torch.Tensor]] = None,
    verifier: Optional[Verifier] = None,
    repair_engine: Optional[RepairEngine] = None,
    fl_adapter=None,
    fl_typed: bool = False,
    tracker: Optional[SpeedTracker] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Core decode loop shared by baselines B0 and B3-B7.

    Returns (output_text, metrics_dict).
    """
    if tokenizer is None:
        tokenizer = _SimpleTokenizer(model.vocab_size)

    # Tokenise
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = max(len(token_ids), config.seq_len)
    if len(token_ids) < seq_len:
        token_ids = token_ids + [model.mask_token_id] * (seq_len - len(token_ids))
    else:
        token_ids = token_ids[:seq_len]

    # Type assignment (if not provided)
    if pos_type is None:
        typer = SpanTyper()
        offset_mapping = [(i, min(i + 1, len(text))) for i in range(len(token_ids))]
        _spans, pos_type, _span_ids = typer.type_text(text, offset_mapping, seq_len)

    # Build allowed masks if needed
    if allowed_masks is None and projection_engine is not None:
        builder = AllowedSetBuilder(tokenizer, AllowedSetConfig(), device="cpu")
        allowed_masks = builder.build()

    # Initialise: all masked
    tokens = torch.full(
        (seq_len,), model.mask_token_id, dtype=torch.long, device="cpu",
    )
    T = config.total_steps
    tokens_per_step = max(1, int(seq_len * config.tokens_per_step_frac))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    total_updated = 0
    total_repaired = 0

    if tracker:
        tracker.start_run()

    start_time = time.perf_counter()

    for t in range(T):
        step_start = time.perf_counter()

        masked = (tokens == model.mask_token_id)
        num_masked = masked.sum().item()
        if num_masked == 0:
            break

        # Propose positions
        num_to_update = min(tokens_per_step, num_masked)
        masked_indices = masked.nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(masked_indices), generator=generator, device="cpu")
        proposed_indices = masked_indices[perm[:num_to_update]]

        proposed_mask = torch.zeros(seq_len, dtype=torch.bool, device="cpu")
        proposed_mask[proposed_indices] = True

        # Schedule hook
        if schedule is not None:
            final_mask = schedule.apply_schedule(proposed_mask, t, T, pos_type)
        else:
            final_mask = proposed_mask

        update_positions = final_mask.nonzero(as_tuple=True)[0]
        if len(update_positions) == 0:
            if tracker:
                tracker.record_step(t, 0)
            continue

        # Get logits
        logits = model.forward_logits(tokens, t, update_positions, conditioning=None)

        # FL adapter hook (modifies logits before projection)
        if fl_adapter is not None:
            if fl_typed and hasattr(fl_adapter, "adapt_logits_typed"):
                local_types = [pos_type[p] for p in update_positions.tolist()]
                logits = fl_adapter.adapt_logits_typed(logits, local_types, t, T)
            elif hasattr(fl_adapter, "adapt_logits"):
                logits = fl_adapter.adapt_logits(logits, t, T)

        # Projection hook
        if projection_engine is not None:
            local_pos_type = [pos_type[p] for p in update_positions.tolist()]
            logits = project_logits(
                logits,
                local_pos_type,
                projection_engine.allowed_masks,
                positions=torch.arange(len(update_positions), device="cpu"),
            )

        # Sample
        probs = torch.softmax(logits / max(config.temperature, 1e-8), dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        tokens[update_positions] = sampled
        total_updated += len(update_positions)

        # Verifier + repair
        step_repaired = 0
        if verifier is not None:
            decoded = tokenizer.decode(tokens.tolist())
            vresult = verifier.check(
                decoded,
                token_ids=tokens.tolist(),
                pos_type=pos_type,
                tokenizer=tokenizer,
            )
            if not vresult.ok and repair_engine is not None:
                violating = []
                for v in vresult.violations:
                    if "positions" in v:
                        violating.extend(v["positions"])
                if violating:
                    def model_fn(tok, s, pos, cond=None):
                        return model.forward_logits(tok, s, pos, cond)
                    tokens, _iters = repair_engine.repair(
                        tokens, violating, pos_type,
                        allowed_masks if allowed_masks is not None else {},
                        model_fn=model_fn, step=t,
                        mask_token_id=model.mask_token_id,
                        temperature=config.temperature,
                    )
                    step_repaired = len(violating)
                    total_repaired += step_repaired

        if tracker:
            tracker.record_step(t, len(update_positions))

    if tracker:
        tracker.end_run()

    elapsed = time.perf_counter() - start_time
    output_text = tokenizer.decode(tokens.tolist())

    metrics: Dict[str, Any] = {
        "elapsed_sec": elapsed,
        "total_steps": T,
        "tokens_updated": total_updated,
        "tokens_repaired": total_repaired,
        "throughput_tok_per_sec": total_updated / max(elapsed, 1e-6),
    }

    if tracker:
        metrics["speed_summary"] = tracker.summary()

    return output_text, metrics


# ---------------------------------------------------------------------------
# B0: Unprotected baseline
# ---------------------------------------------------------------------------

def B0_unprotected(
    model,
    text: str,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """B0 — No protection.  Standard diffusion decode with no privacy hooks.

    The model generates freely with no projection, schedule, verifier,
    or repair.  This baseline measures the *maximum utility* achievable
    and serves as the upper bound for leakage.

    Parameters
    ----------
    model : DiffusionModel
        The diffusion language model.
    text : str
        Input text (may contain PII).
    config : BaselineConfig, optional.

    Returns
    -------
    dict with ``output_text`` and ``metrics``.
    """
    cfg = config or BaselineConfig()
    tracker = SpeedTracker()
    output_text, metrics = _run_decode_loop(
        model, text, cfg, tracker=tracker,
    )
    return {"output_text": output_text, "metrics": metrics}


# ---------------------------------------------------------------------------
# B1: Post-hoc redaction
# ---------------------------------------------------------------------------

def B1_posthoc_redaction(
    output_text: str,
) -> Dict[str, Any]:
    """B1 — Post-hoc regex redaction.

    Takes already-generated text and applies regex-based find-and-replace
    to remove detectable PII patterns.  This is a common "easy" baseline
    but cannot catch PII that does not match the patterns.

    Parameters
    ----------
    output_text : str
        The raw model output (e.g., from B0).

    Returns
    -------
    dict with ``output_text`` (redacted) and ``metrics``.
    """
    start = time.perf_counter()
    redacted = output_text
    total_redactions = 0

    for pat, replacement in _REDACTION_MAP:
        matches = pat.findall(redacted)
        total_redactions += len(matches)
        redacted = pat.sub(replacement, redacted)

    elapsed = time.perf_counter() - start

    return {
        "output_text": redacted,
        "metrics": {
            "elapsed_sec": elapsed,
            "total_redactions": total_redactions,
            "method": "posthoc_regex",
        },
    }


# ---------------------------------------------------------------------------
# B2: AR logit mask (approximate)
# ---------------------------------------------------------------------------

def B2_ar_logit_mask(
    model,
    text: str,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """B2 — Autoregressive-style rolling regex logit mask.

    Approximates what an AR model would do with a logit mask that
    prevents generation of PII patterns.  In the diffusion setting,
    this is approximated by checking the partial output at each step
    and masking logits that would extend a forbidden pattern.

    This is a simplified approximation: at each step, after sampling,
    we check if the output now contains any forbidden pattern and
    re-mask those positions for the next step.  Unlike TPD, there is
    no formal guarantee.

    Parameters
    ----------
    model : DiffusionModel
    text : str
    config : BaselineConfig, optional.

    Returns
    -------
    dict with ``output_text`` and ``metrics``.
    """
    cfg = config or BaselineConfig()
    tokenizer = _SimpleTokenizer(model.vocab_size)

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = max(len(token_ids), cfg.seq_len)
    if len(token_ids) < seq_len:
        token_ids = token_ids + [model.mask_token_id] * (seq_len - len(token_ids))
    else:
        token_ids = token_ids[:seq_len]

    tokens = torch.full(
        (seq_len,), model.mask_token_id, dtype=torch.long, device="cpu",
    )
    T = cfg.total_steps
    tokens_per_step = max(1, int(seq_len * cfg.tokens_per_step_frac))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed)

    tracker = SpeedTracker()
    tracker.start_run()
    total_updated = 0
    total_remasked = 0
    start_time = time.perf_counter()

    # Compile forbidden patterns
    forbidden_pats = [
        (tag, re.compile(pat_str))
        for tag, pat_str in [
            ("EMAIL", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
            ("PHONE", r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
            ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
            ("CC", r"\b(?:\d[ -]*?){13,19}\b"),
            ("ID", r"\b[A-Z]{1,3}\d{6,10}\b"),
        ]
    ]

    for t in range(T):
        masked = (tokens == model.mask_token_id)
        num_masked = masked.sum().item()
        if num_masked == 0:
            break

        num_to_update = min(tokens_per_step, num_masked)
        masked_indices = masked.nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(masked_indices), generator=generator, device="cpu")
        update_positions = masked_indices[perm[:num_to_update]]

        if len(update_positions) == 0:
            tracker.record_step(t, 0)
            continue

        # Get logits and sample
        logits = model.forward_logits(tokens, t, update_positions, conditioning=None)
        probs = torch.softmax(logits / max(cfg.temperature, 1e-8), dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        tokens[update_positions] = sampled
        total_updated += len(update_positions)

        # AR-style check: after writing, check if any forbidden pattern
        # appeared and re-mask those positions
        decoded = tokenizer.decode(tokens.tolist())
        for _tag, pat in forbidden_pats:
            for m in pat.finditer(decoded):
                # Re-mask character positions that form the match
                for ci in range(m.start(), min(m.end(), seq_len)):
                    if tokens[ci] != model.mask_token_id:
                        tokens[ci] = model.mask_token_id
                        total_remasked += 1

        tracker.record_step(t, len(update_positions))

    tracker.end_run()
    elapsed = time.perf_counter() - start_time
    output_text = tokenizer.decode(tokens.tolist())

    return {
        "output_text": output_text,
        "metrics": {
            "elapsed_sec": elapsed,
            "total_steps": T,
            "tokens_updated": total_updated,
            "tokens_remasked": total_remasked,
            "throughput_tok_per_sec": total_updated / max(elapsed, 1e-6),
            "method": "ar_logit_mask",
            "speed_summary": tracker.summary(),
        },
    }


# ---------------------------------------------------------------------------
# B3: TPD projection only
# ---------------------------------------------------------------------------

def B3_tpd_projection_only(
    model,
    text: str,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """B3 — TPD projection only.  No schedule, no verifier, no repair.

    Applies the support-restriction projection at every step for all
    positions, but does not use the 3-phase schedule or the verifier
    gate.  Tests the effect of projection alone.

    Parameters
    ----------
    model : DiffusionModel
    text : str
    config : BaselineConfig, optional.

    Returns
    -------
    dict with ``output_text`` and ``metrics``.
    """
    cfg = config or BaselineConfig()
    tokenizer = _SimpleTokenizer(model.vocab_size)
    builder = AllowedSetBuilder(tokenizer, AllowedSetConfig(), device="cpu")
    allowed_masks = builder.build()

    typer = SpanTyper()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = max(len(token_ids), cfg.seq_len)
    offset_mapping = [(i, min(i + 1, len(text))) for i in range(seq_len)]
    _spans, pos_type, _span_ids = typer.type_text(text, offset_mapping, seq_len)

    proj_engine = ProjectionEngine(allowed_masks, pos_type)
    tracker = SpeedTracker()

    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        tokenizer=tokenizer,
        projection_engine=proj_engine,
        pos_type=pos_type,
        allowed_masks=allowed_masks,
        tracker=tracker,
    )
    metrics["method"] = "tpd_projection_only"
    return {"output_text": output_text, "metrics": metrics}


# ---------------------------------------------------------------------------
# B4: TPD projection + schedule
# ---------------------------------------------------------------------------

def B4_tpd_projection_schedule(
    model,
    text: str,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """B4 — TPD with projection and 3-phase mask schedule.

    Adds the DRAFT/SAFE/REVEAL schedule on top of projection.
    No verifier or repair.

    Parameters
    ----------
    model : DiffusionModel
    text : str
    config : BaselineConfig, optional.

    Returns
    -------
    dict with ``output_text`` and ``metrics``.
    """
    cfg = config or BaselineConfig()
    tokenizer = _SimpleTokenizer(model.vocab_size)
    builder = AllowedSetBuilder(tokenizer, AllowedSetConfig(), device="cpu")
    allowed_masks = builder.build()

    typer = SpanTyper()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = max(len(token_ids), cfg.seq_len)
    offset_mapping = [(i, min(i + 1, len(text))) for i in range(seq_len)]
    _spans, pos_type, _span_ids = typer.type_text(text, offset_mapping, seq_len)

    proj_engine = ProjectionEngine(allowed_masks, pos_type)
    schedule = MaskSchedule(ScheduleConfig(
        draft_end=cfg.schedule_draft_end,
        safe_end=cfg.schedule_safe_end,
    ))
    tracker = SpeedTracker()

    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        tokenizer=tokenizer,
        projection_engine=proj_engine,
        schedule=schedule,
        pos_type=pos_type,
        allowed_masks=allowed_masks,
        tracker=tracker,
    )
    metrics["method"] = "tpd_projection_schedule"
    return {"output_text": output_text, "metrics": metrics}


# ---------------------------------------------------------------------------
# B5: TPD projection + schedule + repair
# ---------------------------------------------------------------------------

def B5_tpd_schedule_repair(
    model,
    text: str,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """B5 — Full TPD pipeline: projection + schedule + verifier + repair.

    This is the complete TPD system without the FL adapter.

    Parameters
    ----------
    model : DiffusionModel
    text : str
    config : BaselineConfig, optional.

    Returns
    -------
    dict with ``output_text`` and ``metrics``.
    """
    cfg = config or BaselineConfig()
    tokenizer = _SimpleTokenizer(model.vocab_size)
    builder = AllowedSetBuilder(tokenizer, AllowedSetConfig(), device="cpu")
    allowed_masks = builder.build()

    typer = SpanTyper()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = max(len(token_ids), cfg.seq_len)
    offset_mapping = [(i, min(i + 1, len(text))) for i in range(seq_len)]
    _spans, pos_type, _span_ids = typer.type_text(text, offset_mapping, seq_len)

    proj_engine = ProjectionEngine(allowed_masks, pos_type)
    schedule = MaskSchedule(ScheduleConfig(
        draft_end=cfg.schedule_draft_end,
        safe_end=cfg.schedule_safe_end,
    ))
    verifier = Verifier(VerifierConfig(
        forbidden_tags=cfg.verifier_forbidden_tags,
        known_secrets=cfg.known_secrets,
    ))
    repair_mode = RepairMode.RESAMPLE if cfg.repair_mode == "resample" else RepairMode.EDIT
    repair_engine = RepairEngine(
        mode=repair_mode,
        max_repair_iters=cfg.repair_max_iters,
    )
    tracker = SpeedTracker()

    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        tokenizer=tokenizer,
        projection_engine=proj_engine,
        schedule=schedule,
        pos_type=pos_type,
        allowed_masks=allowed_masks,
        verifier=verifier,
        repair_engine=repair_engine,
        tracker=tracker,
    )
    metrics["method"] = "tpd_schedule_repair"
    return {"output_text": output_text, "metrics": metrics}


# ---------------------------------------------------------------------------
# B6: TPD + FL (type-agnostic adapter)
# ---------------------------------------------------------------------------

def B6_tpd_fl(
    model,
    text: str,
    config: Optional[BaselineConfig] = None,
    fl_adapter=None,
) -> Dict[str, Any]:
    """B6 — TPD + federated adapter (type-agnostic).

    Adds a federated learning adapter that modifies logits before
    projection.  The adapter is type-agnostic: it applies the same
    transformation regardless of the position's SpanType.

    Parameters
    ----------
    model : DiffusionModel
    text : str
    config : BaselineConfig, optional.
    fl_adapter : object with ``adapt_logits(logits, t, T)`` method.

    Returns
    -------
    dict with ``output_text`` and ``metrics``.
    """
    cfg = config or BaselineConfig()
    tokenizer = _SimpleTokenizer(model.vocab_size)
    builder = AllowedSetBuilder(tokenizer, AllowedSetConfig(), device="cpu")
    allowed_masks = builder.build()

    typer = SpanTyper()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = max(len(token_ids), cfg.seq_len)
    offset_mapping = [(i, min(i + 1, len(text))) for i in range(seq_len)]
    _spans, pos_type, _span_ids = typer.type_text(text, offset_mapping, seq_len)

    proj_engine = ProjectionEngine(allowed_masks, pos_type)
    schedule = MaskSchedule(ScheduleConfig(
        draft_end=cfg.schedule_draft_end,
        safe_end=cfg.schedule_safe_end,
    ))
    verifier = Verifier(VerifierConfig(
        forbidden_tags=cfg.verifier_forbidden_tags,
        known_secrets=cfg.known_secrets,
    ))
    repair_engine = RepairEngine(
        mode=RepairMode.RESAMPLE if cfg.repair_mode == "resample" else RepairMode.EDIT,
        max_repair_iters=cfg.repair_max_iters,
    )
    tracker = SpeedTracker()

    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        tokenizer=tokenizer,
        projection_engine=proj_engine,
        schedule=schedule,
        pos_type=pos_type,
        allowed_masks=allowed_masks,
        verifier=verifier,
        repair_engine=repair_engine,
        fl_adapter=fl_adapter,
        fl_typed=False,
        tracker=tracker,
    )
    metrics["method"] = "tpd_fl"
    return {"output_text": output_text, "metrics": metrics}


# ---------------------------------------------------------------------------
# B7: TPD + FL + typed
# ---------------------------------------------------------------------------

def B7_tpd_fl_typed(
    model,
    text: str,
    config: Optional[BaselineConfig] = None,
    fl_adapter=None,
) -> Dict[str, Any]:
    """B7 — Full TPD+FL system with typed federated adapter.

    The typed adapter uses per-SpanType logit adjustments, enabling
    finer-grained control (e.g., different behaviour for EMAIL vs PHONE
    sensitive positions).

    Parameters
    ----------
    model : DiffusionModel
    text : str
    config : BaselineConfig, optional.
    fl_adapter : object with ``adapt_logits_typed(logits, types, t, T)`` method.

    Returns
    -------
    dict with ``output_text`` and ``metrics``.
    """
    cfg = config or BaselineConfig()
    tokenizer = _SimpleTokenizer(model.vocab_size)
    builder = AllowedSetBuilder(tokenizer, AllowedSetConfig(), device="cpu")
    allowed_masks = builder.build()

    typer = SpanTyper()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = max(len(token_ids), cfg.seq_len)
    offset_mapping = [(i, min(i + 1, len(text))) for i in range(seq_len)]
    _spans, pos_type, _span_ids = typer.type_text(text, offset_mapping, seq_len)

    proj_engine = ProjectionEngine(allowed_masks, pos_type)
    schedule = MaskSchedule(ScheduleConfig(
        draft_end=cfg.schedule_draft_end,
        safe_end=cfg.schedule_safe_end,
    ))
    verifier = Verifier(VerifierConfig(
        forbidden_tags=cfg.verifier_forbidden_tags,
        known_secrets=cfg.known_secrets,
    ))
    repair_engine = RepairEngine(
        mode=RepairMode.RESAMPLE if cfg.repair_mode == "resample" else RepairMode.EDIT,
        max_repair_iters=cfg.repair_max_iters,
    )
    tracker = SpeedTracker()

    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        tokenizer=tokenizer,
        projection_engine=proj_engine,
        schedule=schedule,
        pos_type=pos_type,
        allowed_masks=allowed_masks,
        verifier=verifier,
        repair_engine=repair_engine,
        fl_adapter=fl_adapter,
        fl_typed=True,
        tracker=tracker,
    )
    metrics["method"] = "tpd_fl_typed"
    return {"output_text": output_text, "metrics": metrics}


# ---------------------------------------------------------------------------
# BaselineRunner
# ---------------------------------------------------------------------------

class BaselineRunner:
    """Orchestrates all baselines and collects results.

    Parameters
    ----------
    model : DiffusionModel
        The diffusion model to use for all baselines.
    config : BaselineConfig, optional
        Shared configuration.
    fl_adapter : optional FL adapter for B6/B7.
    leakage_evaluator : optional LeakageEvaluator instance.
    utility_evaluator : optional UtilityEvaluator instance.
    reference_text : optional ground-truth reference for utility metrics.
    reference_secrets : optional list of secrets for leakage evaluation.
    """

    def __init__(
        self,
        model,
        config: Optional[BaselineConfig] = None,
        fl_adapter=None,
        leakage_evaluator: Optional[LeakageEvaluator] = None,
        utility_evaluator: Optional[UtilityEvaluator] = None,
        reference_text: Optional[str] = None,
        reference_secrets: Optional[List[str]] = None,
    ):
        self.model = model
        self.config = config or BaselineConfig()
        self.fl_adapter = fl_adapter
        self.leakage_eval = leakage_evaluator or LeakageEvaluator()
        self.utility_eval = utility_evaluator or UtilityEvaluator()
        self.reference_text = reference_text or ""
        self.reference_secrets = reference_secrets or []

    def run_all(
        self,
        text: str,
        baselines: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all (or selected) baselines on the given text.

        Parameters
        ----------
        text : str
            Input text containing PII.
        baselines : list of baseline names, optional.
            If given, only run these baselines.  Names are "B0" through "B7".

        Returns
        -------
        dict mapping baseline name -> result dict with keys:
            output_text, metrics, leakage, utility.
        """
        all_baselines = {
            "B0": self._run_b0,
            "B1": self._run_b1,
            "B2": self._run_b2,
            "B3": self._run_b3,
            "B4": self._run_b4,
            "B5": self._run_b5,
            "B6": self._run_b6,
            "B7": self._run_b7,
        }

        to_run = baselines or list(all_baselines.keys())
        results: Dict[str, Dict[str, Any]] = {}

        # B0 needed for B1
        b0_output = None

        for name in to_run:
            if name not in all_baselines:
                continue

            if name == "B1":
                # B1 needs B0 output
                if b0_output is None:
                    b0_result = self._run_b0(text)
                    b0_output = b0_result["output_text"]
                result = all_baselines[name](text, b0_output=b0_output)
            else:
                result = all_baselines[name](text)

            if name == "B0":
                b0_output = result["output_text"]

            # Evaluate leakage and utility
            result["leakage"] = self.leakage_eval.evaluate(
                result["output_text"], self.reference_secrets,
            )
            result["utility"] = self.utility_eval.evaluate(
                result["output_text"], self.reference_text,
            )
            results[name] = result

        return results

    # ------------------------------------------------------------------
    # Per-baseline runners
    # ------------------------------------------------------------------

    def _run_b0(self, text: str, **kwargs) -> Dict[str, Any]:
        return B0_unprotected(self.model, text, self.config)

    def _run_b1(self, text: str, b0_output: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if b0_output is None:
            b0_result = B0_unprotected(self.model, text, self.config)
            b0_output = b0_result["output_text"]
        return B1_posthoc_redaction(b0_output)

    def _run_b2(self, text: str, **kwargs) -> Dict[str, Any]:
        return B2_ar_logit_mask(self.model, text, self.config)

    def _run_b3(self, text: str, **kwargs) -> Dict[str, Any]:
        return B3_tpd_projection_only(self.model, text, self.config)

    def _run_b4(self, text: str, **kwargs) -> Dict[str, Any]:
        return B4_tpd_projection_schedule(self.model, text, self.config)

    def _run_b5(self, text: str, **kwargs) -> Dict[str, Any]:
        return B5_tpd_schedule_repair(self.model, text, self.config)

    def _run_b6(self, text: str, **kwargs) -> Dict[str, Any]:
        return B6_tpd_fl(self.model, text, self.config, self.fl_adapter)

    def _run_b7(self, text: str, **kwargs) -> Dict[str, Any]:
        return B7_tpd_fl_typed(self.model, text, self.config, self.fl_adapter)
