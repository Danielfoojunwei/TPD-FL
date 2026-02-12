"""
Baseline Implementations — B0 through B7 for the TPD+FL ablation study.

Each baseline function takes a model backend, a benchmark sample (dict from
:mod:`tpd_fl.eval.benchgen`), and a configuration object, and returns a
:class:`BaselineResult` with the output text, timing, and placeholders for
leakage/utility metrics (filled in by the evaluator).

Baseline progression:

  B0  Unprotected         — no privacy mechanism at all.
  B1  Post-hoc redaction  — generate freely, regex-replace after.
  B2  AR logit mask       — autoregressive rolling regex mask (approximate).
  B3  TPD projection only — apply projection but no schedule or repair.
  B4  TPD + schedule      — projection with 3-phase mask schedule.
  B5  TPD full            — projection + schedule + verifier + repair.
  B6  FL only             — FL adapter with NO TPD.  Shows FL alone does not
                            solve output privacy.
  B7  TPD + FL            — full system with TPD and FL adapters.

The :class:`BaselineRunner` orchestrates selected baselines across all
benchmark samples and collects results.

All code is CPU-practical and works with short outputs.
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

# Optional presidio-backed redaction for B1 (fallback to regex if unavailable)
_PRESIDIO_AVAILABLE = False
try:
    from presidio_analyzer import AnalyzerEngine as _AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine as _AnonymizerEngine
    _PRESIDIO_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# BaselineResult
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Result container returned by every baseline function.

    The ``output_text`` and ``elapsed_sec`` fields are filled in by the
    baseline itself.  The leakage and utility fields are placeholders
    that the evaluator fills in after running leakage and utility checks.

    Attributes
    ----------
    name : str
        Baseline identifier (e.g. "B0_unprotected").
    output_text : str
        The text produced by the baseline.
    elapsed_sec : float
        Wall-clock time for the baseline run in seconds.
    hard_leakage_count : int
        Number of hard (regex) leakage matches.  Filled by evaluator.
    hard_leakage_rate : float
        Fraction of secrets leaked (regex).  Filled by evaluator.
    semantic_leakage : bool
        Whether any known secret was found via substring matching.
        Filled by evaluator.
    utility_score : float
        Composite utility score.  Filled by evaluator.
    """

    name: str
    output_text: str
    elapsed_sec: float
    hard_leakage_count: int = 0
    hard_leakage_rate: float = 0.0
    semantic_leakage: bool = False
    utility_score: float = 0.0


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
    """Minimal character-level tokenizer for baseline runs.

    Maps each character to its ``ord()`` value (clamped to vocab_size).
    This is sufficient for CPU-practical evaluation with synthetic backends.
    """

    def __init__(self, vocab_size: int = 32000):
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to a list of integer IDs."""
        return [min(ord(c), self._vocab_size - 1) for c in text]

    def decode(self, ids) -> str:
        """Decode a list of integer IDs back to text."""
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
# Helper: extract input text from a benchmark sample
# ---------------------------------------------------------------------------

def _get_input_text(sample: Dict[str, Any]) -> str:
    """Extract the input text from a benchmark sample dict.

    Looks for ``source_text`` first (used in S1/S2/S3 samples), then
    falls back to ``prompt``, then to the sample itself if it is a string.
    """
    if isinstance(sample, str):
        return sample
    return sample.get("source_text", sample.get("prompt", ""))


def _get_known_secrets(sample: Dict[str, Any]) -> List[str]:
    """Extract the list of known secret values from a benchmark sample.

    Handles both S1 ``embedded_secrets`` and S2 ``target_secrets`` formats.
    """
    secrets: List[str] = []
    for entry in sample.get("embedded_secrets", []):
        if isinstance(entry, dict) and "value" in entry:
            secrets.append(entry["value"])
        elif isinstance(entry, str):
            secrets.append(entry)
    for entry in sample.get("target_secrets", []):
        if isinstance(entry, dict) and "value" in entry:
            secrets.append(entry["value"])
        elif isinstance(entry, str):
            secrets.append(entry)
    return secrets


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

    This mirrors the logic in :class:`tpd_fl.diffusion.decode_loop.DiffusionDecodeLoop`
    but accepts per-component overrides so that each baseline can enable
    or disable individual hooks.

    Parameters
    ----------
    model : DiffusionModel
        The diffusion language model backend.
    text : str
        Input text (may contain PII).
    config : BaselineConfig
        Shared configuration.
    tokenizer : optional tokenizer (defaults to _SimpleTokenizer).
    projection_engine : optional ProjectionEngine for logit projection.
    schedule : optional MaskSchedule for phase-gated updates.
    pos_type : optional pre-computed per-position type list.
    allowed_masks : optional pre-built allowed-set masks.
    verifier : optional Verifier for post-step checking.
    repair_engine : optional RepairEngine for fixing violations.
    fl_adapter : optional FL adapter with ``adapt_logits`` method.
    fl_typed : bool — if True, use ``adapt_logits_typed`` on the adapter.
    tracker : optional SpeedTracker for timing.

    Returns
    -------
    (output_text, metrics_dict)
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
# Helper: build TPD components from config and text
# ---------------------------------------------------------------------------

def _build_tpd_components(
    model,
    text: str,
    config: BaselineConfig,
    enable_schedule: bool = False,
    enable_verifier: bool = False,
    enable_repair: bool = False,
) -> Dict[str, Any]:
    """Build reusable TPD components for baselines B3-B7.

    Returns a dict with keys: tokenizer, proj_engine, pos_type,
    allowed_masks, schedule, verifier, repair_engine, tracker.
    """
    tokenizer = _SimpleTokenizer(model.vocab_size)
    builder = AllowedSetBuilder(tokenizer, AllowedSetConfig(), device="cpu")
    allowed_masks = builder.build()

    typer = SpanTyper()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = max(len(token_ids), config.seq_len)
    offset_mapping = [(i, min(i + 1, len(text))) for i in range(seq_len)]
    _spans, pos_type, _span_ids = typer.type_text(text, offset_mapping, seq_len)

    proj_engine = ProjectionEngine(allowed_masks, pos_type)

    schedule = None
    if enable_schedule:
        schedule = MaskSchedule(ScheduleConfig(
            draft_end=config.schedule_draft_end,
            safe_end=config.schedule_safe_end,
        ))

    verifier = None
    if enable_verifier:
        verifier = Verifier(VerifierConfig(
            forbidden_tags=config.verifier_forbidden_tags,
            known_secrets=config.known_secrets,
        ))

    repair_engine = None
    if enable_repair:
        repair_mode = (
            RepairMode.RESAMPLE
            if config.repair_mode == "resample"
            else RepairMode.EDIT
        )
        repair_engine = RepairEngine(
            mode=repair_mode,
            max_repair_iters=config.repair_max_iters,
        )

    return {
        "tokenizer": tokenizer,
        "proj_engine": proj_engine,
        "pos_type": pos_type,
        "allowed_masks": allowed_masks,
        "schedule": schedule,
        "verifier": verifier,
        "repair_engine": repair_engine,
        "tracker": SpeedTracker(),
    }


# ---------------------------------------------------------------------------
# B0: Unprotected baseline
# ---------------------------------------------------------------------------

def B0_unprotected(
    model,
    sample: Dict[str, Any],
    config: Optional[BaselineConfig] = None,
) -> BaselineResult:
    """B0 -- No protection.  Standard diffusion decode with no privacy hooks.

    The model generates freely with no projection, schedule, verifier,
    or repair.  This baseline measures the *maximum utility* achievable
    and serves as the upper bound for leakage.

    Parameters
    ----------
    model : DiffusionModel
        The diffusion language model backend.
    sample : dict
        A benchmark sample from :mod:`tpd_fl.eval.benchgen`.
    config : BaselineConfig, optional.

    Returns
    -------
    BaselineResult
    """
    cfg = config or BaselineConfig()
    text = _get_input_text(sample)
    tracker = SpeedTracker()

    start = time.perf_counter()
    output_text, metrics = _run_decode_loop(model, text, cfg, tracker=tracker)
    elapsed = time.perf_counter() - start

    return BaselineResult(
        name="B0_unprotected",
        output_text=output_text,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# B1: Post-hoc redaction
# ---------------------------------------------------------------------------

def _apply_presidio_redaction(text: str) -> Tuple[str, int]:
    """Apply presidio-based redaction if available.

    Returns (redacted_text, num_redactions).
    """
    analyzer = _AnalyzerEngine()
    anonymizer = _AnonymizerEngine()
    results = analyzer.analyze(text=text, language="en")
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, len(results)


def _apply_regex_redaction(text: str) -> Tuple[str, int]:
    """Apply regex-based redaction as a fallback.

    Returns (redacted_text, num_redactions).
    """
    redacted = text
    total_redactions = 0
    for pat, replacement in _REDACTION_MAP:
        matches = pat.findall(redacted)
        total_redactions += len(matches)
        redacted = pat.sub(replacement, redacted)
    return redacted, total_redactions


def B1_posthoc_redaction(
    model,
    sample: Dict[str, Any],
    config: Optional[BaselineConfig] = None,
) -> BaselineResult:
    """B1 -- Post-hoc regex redaction.

    Runs B0 (unprotected decode) first, then applies regex-based
    find-and-replace to remove detectable PII patterns from the output.
    Tries to use the ``presidio`` library for more comprehensive NER-based
    redaction; falls back to simple regex replacement if presidio is not
    installed.

    Parameters
    ----------
    model : DiffusionModel
        The diffusion language model backend.
    sample : dict
        A benchmark sample.
    config : BaselineConfig, optional.

    Returns
    -------
    BaselineResult
    """
    cfg = config or BaselineConfig()
    text = _get_input_text(sample)

    start = time.perf_counter()

    # Step 1: generate unprotected output
    raw_output, _metrics = _run_decode_loop(model, text, cfg)

    # Step 2: apply redaction
    if _PRESIDIO_AVAILABLE:
        try:
            redacted, num_redactions = _apply_presidio_redaction(raw_output)
        except Exception:
            # Fall back to regex if presidio fails at runtime
            redacted, num_redactions = _apply_regex_redaction(raw_output)
    else:
        redacted, num_redactions = _apply_regex_redaction(raw_output)

    elapsed = time.perf_counter() - start

    return BaselineResult(
        name="B1_posthoc_redaction",
        output_text=redacted,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# B2: AR logit mask (approximate)
# ---------------------------------------------------------------------------

def B2_ar_logit_mask(
    model,
    sample: Dict[str, Any],
    config: Optional[BaselineConfig] = None,
) -> BaselineResult:
    """B2 -- Autoregressive-style rolling regex logit mask.

    Approximates what an AR model would do with a logit mask that
    prevents generation of PII patterns.  In the diffusion setting this
    is approximated by checking the partial output at each step and
    re-masking positions that form part of a forbidden pattern.

    For each position, if recently decoded text matches a PII-starting
    pattern, the tokens at those positions are reset to the mask token
    so the model must try again.  Unlike TPD, there is no formal
    guarantee that PII will never appear.

    Parameters
    ----------
    model : DiffusionModel
    sample : dict
    config : BaselineConfig, optional.

    Returns
    -------
    BaselineResult
    """
    cfg = config or BaselineConfig()
    text = _get_input_text(sample)
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
                for ci in range(m.start(), min(m.end(), seq_len)):
                    if tokens[ci] != model.mask_token_id:
                        tokens[ci] = model.mask_token_id
                        total_remasked += 1

        tracker.record_step(t, len(update_positions))

    tracker.end_run()
    elapsed = time.perf_counter() - start_time
    output_text = tokenizer.decode(tokens.tolist())

    return BaselineResult(
        name="B2_ar_logit_mask",
        output_text=output_text,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# B3: TPD projection only
# ---------------------------------------------------------------------------

def B3_tpd_projection(
    model,
    sample: Dict[str, Any],
    config: Optional[BaselineConfig] = None,
) -> BaselineResult:
    """B3 -- TPD projection ONLY.  No schedule, no verifier, no repair.

    Applies the support-restriction projection at every step for all
    positions, but does not use the 3-phase schedule or the verifier
    gate.  Tests the effect of projection alone.

    Parameters
    ----------
    model : DiffusionModel
    sample : dict
    config : BaselineConfig, optional.

    Returns
    -------
    BaselineResult
    """
    cfg = config or BaselineConfig()
    text = _get_input_text(sample)
    components = _build_tpd_components(
        model, text, cfg,
        enable_schedule=False,
        enable_verifier=False,
        enable_repair=False,
    )

    start = time.perf_counter()
    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        tokenizer=components["tokenizer"],
        projection_engine=components["proj_engine"],
        pos_type=components["pos_type"],
        allowed_masks=components["allowed_masks"],
        tracker=components["tracker"],
    )
    elapsed = time.perf_counter() - start

    return BaselineResult(
        name="B3_tpd_projection",
        output_text=output_text,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# B4: TPD projection + schedule
# ---------------------------------------------------------------------------

def B4_tpd_projection_schedule(
    model,
    sample: Dict[str, Any],
    config: Optional[BaselineConfig] = None,
) -> BaselineResult:
    """B4 -- TPD with projection and 3-phase mask schedule.

    Adds the DRAFT / SAFE / REVEAL schedule on top of projection.
    Sensitive positions are only updated during the SAFE phase, when
    the projection engine constrains their output to the allowed set.
    No verifier or repair.

    Parameters
    ----------
    model : DiffusionModel
    sample : dict
    config : BaselineConfig, optional.

    Returns
    -------
    BaselineResult
    """
    cfg = config or BaselineConfig()
    text = _get_input_text(sample)
    components = _build_tpd_components(
        model, text, cfg,
        enable_schedule=True,
        enable_verifier=False,
        enable_repair=False,
    )

    start = time.perf_counter()
    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        tokenizer=components["tokenizer"],
        projection_engine=components["proj_engine"],
        schedule=components["schedule"],
        pos_type=components["pos_type"],
        allowed_masks=components["allowed_masks"],
        tracker=components["tracker"],
    )
    elapsed = time.perf_counter() - start

    return BaselineResult(
        name="B4_tpd_projection_schedule",
        output_text=output_text,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# B5: TPD full (projection + schedule + verifier + repair)
# ---------------------------------------------------------------------------

def B5_tpd_full(
    model,
    sample: Dict[str, Any],
    config: Optional[BaselineConfig] = None,
) -> BaselineResult:
    """B5 -- Full TPD pipeline: projection + schedule + verifier + repair.

    This is the complete TPD system without the FL adapter.  The verifier
    checks for forbidden patterns after each step.  If violations are
    detected, the repair engine re-masks and re-samples the violating
    positions under projection constraints.

    Parameters
    ----------
    model : DiffusionModel
    sample : dict
    config : BaselineConfig, optional.

    Returns
    -------
    BaselineResult
    """
    cfg = config or BaselineConfig()

    # Inject known secrets from the sample into the verifier config
    secrets = _get_known_secrets(sample)
    if secrets:
        cfg = BaselineConfig(
            total_steps=cfg.total_steps,
            seq_len=cfg.seq_len,
            tokens_per_step_frac=cfg.tokens_per_step_frac,
            temperature=cfg.temperature,
            seed=cfg.seed,
            schedule_draft_end=cfg.schedule_draft_end,
            schedule_safe_end=cfg.schedule_safe_end,
            repair_mode=cfg.repair_mode,
            repair_max_iters=cfg.repair_max_iters,
            verifier_forbidden_tags=list(cfg.verifier_forbidden_tags),
            known_secrets=secrets,
            fl_enabled=cfg.fl_enabled,
            fl_typed=cfg.fl_typed,
        )

    text = _get_input_text(sample)
    components = _build_tpd_components(
        model, text, cfg,
        enable_schedule=True,
        enable_verifier=True,
        enable_repair=True,
    )

    start = time.perf_counter()
    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        tokenizer=components["tokenizer"],
        projection_engine=components["proj_engine"],
        schedule=components["schedule"],
        pos_type=components["pos_type"],
        allowed_masks=components["allowed_masks"],
        verifier=components["verifier"],
        repair_engine=components["repair_engine"],
        tracker=components["tracker"],
    )
    elapsed = time.perf_counter() - start

    return BaselineResult(
        name="B5_tpd_full",
        output_text=output_text,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# B6: FL only (no TPD)
# ---------------------------------------------------------------------------

def B6_fl_only(
    model,
    sample: Dict[str, Any],
    config: Optional[BaselineConfig] = None,
    fl_adapter=None,
) -> BaselineResult:
    """B6 -- FL adapter applied, but NO TPD.

    Demonstrates that federated learning adapters alone do not solve
    output privacy.  The FL adapter modifies logits but without
    projection, schedule, or verifier there is no hard privacy guarantee.

    Parameters
    ----------
    model : DiffusionModel
    sample : dict
    config : BaselineConfig, optional.
    fl_adapter : object with ``adapt_logits(logits, t, T)`` method.

    Returns
    -------
    BaselineResult
    """
    cfg = config or BaselineConfig()
    text = _get_input_text(sample)
    tracker = SpeedTracker()

    start = time.perf_counter()
    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        fl_adapter=fl_adapter,
        fl_typed=False,
        tracker=tracker,
    )
    elapsed = time.perf_counter() - start

    return BaselineResult(
        name="B6_fl_only",
        output_text=output_text,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# B7: TPD + FL (full system)
# ---------------------------------------------------------------------------

def B7_tpd_fl(
    model,
    sample: Dict[str, Any],
    config: Optional[BaselineConfig] = None,
    fl_adapter=None,
) -> BaselineResult:
    """B7 -- TPD + FL adapters.  Full system.

    Combines the complete TPD pipeline (projection + schedule + verifier
    + repair) with a federated learning adapter.  The FL adapter modifies
    logits before projection, and the projection enforces the hard
    privacy guarantee.  If the adapter exposes ``adapt_logits_typed``,
    the typed variant is used for per-SpanType logit adjustments.

    Parameters
    ----------
    model : DiffusionModel
    sample : dict
    config : BaselineConfig, optional.
    fl_adapter : object with ``adapt_logits(logits, t, T)`` or
        ``adapt_logits_typed(logits, types, t, T)`` method.

    Returns
    -------
    BaselineResult
    """
    cfg = config or BaselineConfig()

    # Inject known secrets from the sample into the verifier config
    secrets = _get_known_secrets(sample)
    if secrets:
        cfg = BaselineConfig(
            total_steps=cfg.total_steps,
            seq_len=cfg.seq_len,
            tokens_per_step_frac=cfg.tokens_per_step_frac,
            temperature=cfg.temperature,
            seed=cfg.seed,
            schedule_draft_end=cfg.schedule_draft_end,
            schedule_safe_end=cfg.schedule_safe_end,
            repair_mode=cfg.repair_mode,
            repair_max_iters=cfg.repair_max_iters,
            verifier_forbidden_tags=list(cfg.verifier_forbidden_tags),
            known_secrets=secrets,
            fl_enabled=cfg.fl_enabled,
            fl_typed=cfg.fl_typed,
        )

    text = _get_input_text(sample)
    # Determine if the adapter supports typed logits
    use_typed = (
        fl_adapter is not None
        and hasattr(fl_adapter, "adapt_logits_typed")
    )
    components = _build_tpd_components(
        model, text, cfg,
        enable_schedule=True,
        enable_verifier=True,
        enable_repair=True,
    )

    start = time.perf_counter()
    output_text, metrics = _run_decode_loop(
        model, text, cfg,
        tokenizer=components["tokenizer"],
        projection_engine=components["proj_engine"],
        schedule=components["schedule"],
        pos_type=components["pos_type"],
        allowed_masks=components["allowed_masks"],
        verifier=components["verifier"],
        repair_engine=components["repair_engine"],
        fl_adapter=fl_adapter,
        fl_typed=use_typed,
        tracker=components["tracker"],
    )
    elapsed = time.perf_counter() - start

    return BaselineResult(
        name="B7_tpd_fl",
        output_text=output_text,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Backward-compatible aliases (for existing __init__.py imports)
# ---------------------------------------------------------------------------

def B3_tpd_projection_only(
    model, text: str, config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper — delegates to B3_tpd_projection.

    Accepts a raw text string instead of a sample dict and returns a
    dict with ``output_text`` and ``metrics`` keys for compatibility
    with callers that use the old interface.
    """
    sample = {"source_text": text}
    result = B3_tpd_projection(model, sample, config)
    return {"output_text": result.output_text, "metrics": {"elapsed_sec": result.elapsed_sec}}


def B5_tpd_schedule_repair(
    model, text: str, config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper — delegates to B5_tpd_full."""
    sample = {"source_text": text}
    result = B5_tpd_full(model, sample, config)
    return {"output_text": result.output_text, "metrics": {"elapsed_sec": result.elapsed_sec}}


def B6_tpd_fl(
    model, text: str, config: Optional[BaselineConfig] = None,
    fl_adapter=None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper — delegates to B6_fl_only."""
    sample = {"source_text": text}
    result = B6_fl_only(model, sample, config, fl_adapter)
    return {"output_text": result.output_text, "metrics": {"elapsed_sec": result.elapsed_sec}}


def B7_tpd_fl_typed(
    model, text: str, config: Optional[BaselineConfig] = None,
    fl_adapter=None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper — delegates to B7_tpd_fl."""
    sample = {"source_text": text}
    result = B7_tpd_fl(model, sample, config, fl_adapter)
    return {"output_text": result.output_text, "metrics": {"elapsed_sec": result.elapsed_sec}}


# ---------------------------------------------------------------------------
# BaselineRunner
# ---------------------------------------------------------------------------

# Registry mapping baseline names to (function, requires_fl_adapter) pairs
_BASELINE_REGISTRY: Dict[str, Tuple[Callable, bool]] = {
    "B0": (B0_unprotected, False),
    "B1": (B1_posthoc_redaction, False),
    "B2": (B2_ar_logit_mask, False),
    "B3": (B3_tpd_projection, False),
    "B4": (B4_tpd_projection_schedule, False),
    "B5": (B5_tpd_full, False),
    "B6": (B6_fl_only, True),
    "B7": (B7_tpd_fl, True),
}


class BaselineRunner:
    """Orchestrates all baselines across benchmark samples and collects results.

    The runner iterates over a list of benchmark samples (from
    :class:`~tpd_fl.eval.benchgen.BenchmarkGenerator`), runs each
    selected baseline on every sample, and optionally evaluates leakage
    and utility metrics on each result.

    Parameters
    ----------
    model : DiffusionModel
        The diffusion model backend to use for all baselines.
    config : BaselineConfig, optional
        Shared configuration for all baselines.
    fl_adapter : optional
        FL adapter for B6 and B7.  Must expose ``adapt_logits(logits, t, T)``
        and optionally ``adapt_logits_typed(logits, types, t, T)``.
    leakage_evaluator : LeakageEvaluator, optional
        Evaluator for leakage metrics.  If not given, a default instance
        using :data:`STANDARD_PATTERNS` is created.
    utility_evaluator : UtilityEvaluator, optional
        Evaluator for utility metrics.

    Example
    -------
    ::

        from tpd_fl.eval.benchgen import BenchmarkGenerator
        from tpd_fl.eval.baselines import BaselineRunner, BaselineConfig
        from tpd_fl.diffusion.model_adapter import SyntheticDiffusionModel

        model = SyntheticDiffusionModel()
        gen = BenchmarkGenerator()
        samples = gen.generate_s1(num_samples=10)

        runner = BaselineRunner(model)
        results = runner.run(samples, baseline_names=["B0", "B3", "B5"])
        for name, result_list in results.items():
            print(f"{name}: {len(result_list)} results")
    """

    def __init__(
        self,
        model,
        config: Optional[BaselineConfig] = None,
        fl_adapter=None,
        leakage_evaluator: Optional[LeakageEvaluator] = None,
        utility_evaluator: Optional[UtilityEvaluator] = None,
    ):
        self.model = model
        self.config = config or BaselineConfig()
        self.fl_adapter = fl_adapter
        self.leakage_eval = leakage_evaluator or LeakageEvaluator()
        self.utility_eval = utility_evaluator or UtilityEvaluator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        samples: List[Dict[str, Any]],
        baseline_names: Optional[List[str]] = None,
    ) -> Dict[str, List[BaselineResult]]:
        """Run selected baselines on all samples and collect results.

        Parameters
        ----------
        samples : list of dict
            Benchmark samples from :mod:`tpd_fl.eval.benchgen`.
        baseline_names : list of str, optional
            Which baselines to run.  Defaults to all (B0-B7).

        Returns
        -------
        Dict mapping baseline name -> list of BaselineResult (one per
        sample).  Leakage and utility fields are filled in.
        """
        names = baseline_names or list(_BASELINE_REGISTRY.keys())
        results: Dict[str, List[BaselineResult]] = {n: [] for n in names}

        for sample in samples:
            known_secrets = _get_known_secrets(sample)
            reference_text = sample.get("source_text", "")

            for name in names:
                if name not in _BASELINE_REGISTRY:
                    continue

                fn, needs_fl = _BASELINE_REGISTRY[name]

                # Call the baseline function
                if needs_fl:
                    result = fn(
                        self.model, sample, self.config,
                        fl_adapter=self.fl_adapter,
                    )
                else:
                    result = fn(self.model, sample, self.config)

                # Evaluate leakage
                leakage = self.leakage_eval.evaluate(
                    result.output_text, known_secrets,
                )
                result.hard_leakage_count = leakage["hard_leakage_count"]
                result.hard_leakage_rate = leakage["hard_leakage_rate"]
                result.semantic_leakage = leakage["semantic_leakage_detected"]

                # Evaluate utility
                utility = self.utility_eval.evaluate(
                    result.output_text, reference_text,
                )
                result.utility_score = utility.get(
                    "exact_match_public",
                    utility.get("rouge", {}).get("rouge1", {}).get("f1", 0.0),
                )

                results[name].append(result)

        return results

    def run_single(
        self,
        sample: Dict[str, Any],
        baseline_names: Optional[List[str]] = None,
    ) -> Dict[str, BaselineResult]:
        """Run selected baselines on a single sample.

        Convenience wrapper around :meth:`run` for single-sample usage.

        Parameters
        ----------
        sample : dict
            A single benchmark sample.
        baseline_names : list of str, optional.

        Returns
        -------
        Dict mapping baseline name -> BaselineResult.
        """
        batch_results = self.run([sample], baseline_names=baseline_names)
        return {name: results[0] for name, results in batch_results.items() if results}

    # ------------------------------------------------------------------
    # Legacy interface (for backward compatibility with run_eval.py)
    # ------------------------------------------------------------------

    def run_all(
        self,
        text: str,
        baselines: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all (or selected) baselines on raw text.

        This method preserves backward compatibility with the original
        ``BaselineRunner.run_all`` interface used by ``run_eval.py``.

        Parameters
        ----------
        text : str
            Input text containing PII.
        baselines : list of baseline names, optional.

        Returns
        -------
        dict mapping baseline name -> dict with ``output_text``, ``metrics``,
        ``leakage``, ``utility`` keys.
        """
        sample = {"source_text": text}
        names = baselines or list(_BASELINE_REGISTRY.keys())
        results: Dict[str, Dict[str, Any]] = {}

        for name in names:
            if name not in _BASELINE_REGISTRY:
                continue

            fn, needs_fl = _BASELINE_REGISTRY[name]
            if needs_fl:
                br = fn(self.model, sample, self.config, fl_adapter=self.fl_adapter)
            else:
                br = fn(self.model, sample, self.config)

            leakage = self.leakage_eval.evaluate(br.output_text, [])
            utility = self.utility_eval.evaluate(br.output_text, text)

            results[name] = {
                "output_text": br.output_text,
                "metrics": {"elapsed_sec": br.elapsed_sec},
                "leakage": leakage,
                "utility": utility,
            }

        return results


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_baselines(
    backend,
    samples: List[Dict[str, Any]],
    baseline_names: Optional[List[str]] = None,
    config: Optional[BaselineConfig] = None,
    fl_adapter=None,
) -> Dict[str, List[BaselineResult]]:
    """Run baselines on all samples and return collected results.

    This is a module-level convenience function that creates a
    :class:`BaselineRunner` and delegates to :meth:`BaselineRunner.run`.

    Parameters
    ----------
    backend : DiffusionModel
        The diffusion model backend.
    samples : list of dict
        Benchmark samples from :mod:`tpd_fl.eval.benchgen`.
    baseline_names : list of str, optional
        Which baselines to run (defaults to all B0-B7).
    config : BaselineConfig, optional.
    fl_adapter : optional FL adapter for B6/B7.

    Returns
    -------
    Dict mapping baseline name -> list of BaselineResult.
    """
    runner = BaselineRunner(
        model=backend,
        config=config,
        fl_adapter=fl_adapter,
    )
    return runner.run(samples, baseline_names=baseline_names)
