"""
Diffusion Decode Loop with TPD hooks.

This module implements the full M2T (mask-to-token) decode loop for
diffusion language models, with integrated TPD hooks:

  Hook A — Schedule:   Intersects proposed update mask with phase-allowed positions.
  Hook B — Projection: Applies Π_{A(τ_i)} to logits before sampling.
  Hook C — Verifier:   Runs Okπ after each step (or periodically).
  Hook D — Repair:     Triggers repair if verifier rejects.
  Hook E — Diagnostics: Logs Z_i statistics per step.

The loop is backend-agnostic: it uses the DiffusionModel abstraction
and can run with synthetic, HF, or LLaDA backends.

CLI entry point::

    python -m tpd_fl.diffusion.decode_loop --config configs/decode/tpd.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from tpd_fl.tpd.typing import SpanType, SpanTyper, SENSITIVE_TYPES
from tpd_fl.tpd.allowed_sets import AllowedSetBuilder, AllowedSetConfig
from tpd_fl.tpd.schedule import MaskSchedule, ScheduleConfig, build_random_block_mask
from tpd_fl.tpd.projection import ProjectionEngine, project_logits
from tpd_fl.tpd.verifier import Verifier, VerifierConfig
from tpd_fl.tpd.repair import RepairEngine, RepairMode
from tpd_fl.tpd.diagnostics import (
    DiagnosticsLogger,
    compute_allowed_mass,
    compute_z_stats,
)
from tpd_fl.diffusion.model_adapter import (
    DiffusionModel,
    SyntheticDiffusionModel,
    SamplerConfig,
)


@dataclass
class DecodeConfig:
    """Full configuration for a TPD-enabled diffusion decode run."""
    # Model
    model_backend: str = "synthetic"     # "synthetic" | "hf" | "llada"
    model_name_or_path: str = ""
    vocab_size: int = 32000
    mask_token_id: int = 0
    synthetic_mode: str = "uniform"

    # Diffusion
    total_steps: int = 64
    seq_len: int = 128
    tokens_per_step_frac: float = 0.15   # fraction of masked positions to update per step

    # Sampling
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 42

    # TPD Schedule
    schedule_draft_end: float = 0.4
    schedule_safe_end: float = 0.9
    schedule_selection: str = "random"

    # TPD Projection — enabled by default
    projection_enabled: bool = True

    # TPD Verifier
    verifier_enabled: bool = True
    verifier_check_every: int = 1       # check every N steps (1 = every step)
    verifier_forbidden_tags: List[str] = field(
        default_factory=lambda: ["EMAIL", "PHONE", "SSN", "CC", "ID"]
    )
    known_secrets: List[str] = field(default_factory=list)

    # TPD Repair
    repair_enabled: bool = True
    repair_mode: str = "resample"       # "resample" | "edit"
    repair_max_iters: int = 3

    # Diagnostics
    diagnostics_enabled: bool = True
    output_dir: str = "runs/decode"

    # Input
    input_text: str = ""
    input_file: str = ""

    # Typing
    typing_use_ner: bool = False
    denylist_positions: List[int] = field(default_factory=list)


def load_decode_config(path: str) -> DecodeConfig:
    """Load config from YAML file, returning a DecodeConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return DecodeConfig(**{k: v for k, v in raw.items() if k in DecodeConfig.__dataclass_fields__})


class SimpleTokenizer:
    """Minimal character-level tokenizer for synthetic testing.

    Maps characters to integer IDs. Used when no real tokenizer
    is available (synthetic backend).
    """

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


class DiffusionDecodeLoop:
    """Full diffusion decode loop with TPD hooks.

    Usage::

        loop = DiffusionDecodeLoop(config)
        result = loop.run(input_text="Some text with user@email.com")
    """

    def __init__(
        self,
        config: Optional[DecodeConfig] = None,
        model: Optional[DiffusionModel] = None,
        tokenizer=None,
    ):
        self.config = config or DecodeConfig()
        self.device = "cpu"

        # Model
        if model is not None:
            self.model = model
        else:
            self.model = self._build_model()

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SimpleTokenizer(self.config.vocab_size)

        # TPD components
        self.typer = SpanTyper(
            use_ner=self.config.typing_use_ner,
            denylist_positions=self.config.denylist_positions,
        )
        self.allowed_builder = AllowedSetBuilder(
            self.tokenizer,
            AllowedSetConfig(),
            device=self.device,
        )
        self.schedule = MaskSchedule(ScheduleConfig(
            draft_end=self.config.schedule_draft_end,
            safe_end=self.config.schedule_safe_end,
            selection=self.config.schedule_selection,
        ))
        self.verifier = Verifier(VerifierConfig(
            forbidden_tags=self.config.verifier_forbidden_tags,
            known_secrets=self.config.known_secrets,
        )) if self.config.verifier_enabled else None

        self.repair_engine = RepairEngine(
            mode=RepairMode.RESAMPLE if self.config.repair_mode == "resample" else RepairMode.EDIT,
            max_repair_iters=self.config.repair_max_iters,
        ) if self.config.repair_enabled else None

        # Sampler config
        self.sampler_config = SamplerConfig(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            seed=self.config.seed,
        )

        # Generator for reproducibility
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.config.seed)

    def _build_model(self) -> DiffusionModel:
        if self.config.model_backend == "synthetic":
            return SyntheticDiffusionModel(
                vocab_size=self.config.vocab_size,
                mask_token_id_val=self.config.mask_token_id,
                mode=self.config.synthetic_mode,
                device=self.device,
                seed=self.config.seed,
            )
        elif self.config.model_backend == "hf":
            # Lazy import for optional dependency
            from tpd_fl.diffusion.model_adapter import HFDiffusionModel
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
            mdl = AutoModelForMaskedLM.from_pretrained(self.config.model_name_or_path)
            return HFDiffusionModel(mdl, tok, device=self.device)
        else:
            raise ValueError(f"Unknown backend: {self.config.model_backend}")

    def run(
        self,
        input_text: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Run the full decode loop.

        Parameters
        ----------
        input_text : the source text (may contain PII to be redacted).
        output_dir : directory for logs and artifacts.

        Returns
        -------
        dict with keys: output_text, metrics, diagnostics_summary
        """
        text = input_text or self.config.input_text
        out_dir = output_dir or self.config.output_dir

        # Set up output
        run_dir = Path(out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        diag_logger = DiagnosticsLogger(str(run_dir)) if self.config.diagnostics_enabled else None

        # Tokenize
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        seq_len = max(len(token_ids), self.config.seq_len)
        # Pad or truncate
        if len(token_ids) < seq_len:
            token_ids = token_ids + [self.model.mask_token_id] * (seq_len - len(token_ids))
        else:
            token_ids = token_ids[:seq_len]

        # Build offset mapping for typing
        offset_mapping = self._build_offset_mapping(text, token_ids)

        # Run typer
        spans, pos_type, pos_span_id = self.typer.type_text(text, offset_mapping, seq_len)

        # Build allowed masks
        allowed_masks = self.allowed_builder.build()

        # Build projection engine
        proj_engine = ProjectionEngine(allowed_masks, pos_type)

        # Initialize tokens: all masked
        tokens = torch.full(
            (seq_len,), self.model.mask_token_id,
            dtype=torch.long, device=self.device,
        )

        T = self.config.total_steps
        tokens_per_step = max(1, int(seq_len * self.config.tokens_per_step_frac))
        start_time = time.time()
        total_updated = 0
        total_repaired = 0

        for t in range(T):
            # Determine which positions are still masked
            masked = (tokens == self.model.mask_token_id)
            num_masked = masked.sum().item()
            if num_masked == 0:
                break

            # Propose update mask (random selection among masked)
            num_to_update = min(tokens_per_step, num_masked)
            masked_indices = masked.nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(masked_indices), generator=self._generator, device=self.device)
            proposed_indices = masked_indices[perm[:num_to_update]]

            proposed_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
            proposed_mask[proposed_indices] = True

            # Hook A — Schedule
            final_mask = self.schedule.apply_schedule(proposed_mask, t, T, pos_type)
            update_positions = final_mask.nonzero(as_tuple=True)[0]

            if len(update_positions) == 0:
                continue

            # Get logits
            logits = self.model.forward_logits(
                tokens, t, update_positions, conditioning=None,
            )

            # Hook E — Diagnostics (pre-projection)
            if diag_logger and self.config.diagnostics_enabled:
                full_logits = self.model.forward_logits_full(tokens, t)
                z_values = compute_allowed_mass(full_logits, pos_type, allowed_masks)
                z_stats = compute_z_stats(z_values, pos_type)
                phase = self.schedule.phase(t, T).name

            # Hook B — Projection
            if self.config.projection_enabled:
                # Build local pos_type for the positions being updated
                local_pos_type = [pos_type[p] for p in update_positions.tolist()]
                logits = project_logits(
                    logits,
                    local_pos_type,
                    allowed_masks,
                    positions=torch.arange(len(update_positions), device=self.device),
                )

            # Sample tokens
            sampled = self.model.sample_tokens(logits, self.sampler_config)

            # Write sampled tokens
            tokens[update_positions] = sampled
            total_updated += len(update_positions)

            # Hook C — Verifier
            step_repaired = 0
            if self.verifier and (t % self.config.verifier_check_every == 0):
                decoded = self.tokenizer.decode(tokens.tolist())
                vresult = self.verifier.check(
                    decoded,
                    token_ids=tokens.tolist(),
                    pos_type=pos_type,
                    tokenizer=self.tokenizer,
                )

                # Hook D — Repair
                if not vresult.ok and self.repair_engine:
                    violating = []
                    for v in vresult.violations:
                        if "positions" in v:
                            violating.extend(v["positions"])
                    if violating:
                        def model_fn(tok, s, pos, cond=None):
                            return self.model.forward_logits(tok, s, pos, cond)

                        tokens, iters = self.repair_engine.repair(
                            tokens, violating, pos_type, allowed_masks,
                            model_fn=model_fn, step=t,
                            mask_token_id=self.model.mask_token_id,
                            temperature=self.config.temperature,
                        )
                        step_repaired = len(violating)
                        total_repaired += step_repaired

            # Log diagnostics
            if diag_logger and self.config.diagnostics_enabled:
                diag_logger.log_step(
                    step=t,
                    total_steps=T,
                    phase=phase,
                    z_stats=z_stats,
                    num_updated=len(update_positions),
                    num_repaired=step_repaired,
                )

        # Final decode
        output_text = self.tokenizer.decode(tokens.tolist())
        elapsed = time.time() - start_time

        # Final verifier check
        final_check = None
        if self.verifier:
            final_check = self.verifier.check(
                output_text,
                token_ids=tokens.tolist(),
                pos_type=pos_type,
                tokenizer=self.tokenizer,
            )

        # Hard guarantee check
        hard_guarantee = proj_engine.verify_hard_guarantee(tokens)

        metrics = {
            "output_text": output_text,
            "seq_len": seq_len,
            "total_steps": T,
            "tokens_updated": total_updated,
            "tokens_repaired": total_repaired,
            "elapsed_sec": elapsed,
            "throughput_tok_per_sec": total_updated / max(elapsed, 1e-6),
            "verifier_pass": final_check.ok if final_check else None,
            "verifier_violations": len(final_check.violations) if final_check else 0,
            "hard_guarantee_holds": hard_guarantee,
            "num_sensitive_positions": sum(1 for pt in pos_type if pt in SENSITIVE_TYPES),
            "num_spans_detected": len(spans),
        }

        if diag_logger:
            diag_logger.log_final(metrics)
            metrics["diagnostics_summary"] = diag_logger.summary()

        # Save artifacts
        self._save_artifacts(run_dir, metrics, spans, pos_type)

        return metrics

    def _build_offset_mapping(
        self, text: str, token_ids: List[int],
    ) -> List[Tuple[int, int]]:
        """Build a simple character offset mapping.

        For real tokenizers, use ``return_offsets_mapping=True``.
        For the simple tokenizer, each character maps to one token.
        """
        offsets = []
        for i, _ in enumerate(token_ids):
            if i < len(text):
                offsets.append((i, i + 1))
            else:
                offsets.append((len(text), len(text)))
        return offsets

    def _save_artifacts(
        self,
        run_dir: Path,
        metrics: Dict,
        spans,
        pos_type: List[SpanType],
    ):
        """Save config snapshot, metrics, and span info."""
        # Config snapshot
        with open(run_dir / "config.json", "w") as f:
            cfg_dict = {k: v for k, v in self.config.__dict__.items()}
            json.dump(cfg_dict, f, indent=2, default=str)

        # Metrics
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                serializable_metrics[k] = v
            elif isinstance(v, dict):
                serializable_metrics[k] = v
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        # Spans
        span_data = [
            {"start": s.start, "end": s.end, "type": s.type.name, "entity_tag": s.entity_tag}
            for s in spans
        ]
        with open(run_dir / "spans.json", "w") as f:
            json.dump(span_data, f, indent=2)


def main():
    """CLI entry point for diffusion decode with TPD."""
    parser = argparse.ArgumentParser(description="TPD Diffusion Decode Loop")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--input-text", type=str, default=None, help="Input text")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=None, help="Total diffusion steps")
    args = parser.parse_args()

    if args.config:
        config = load_decode_config(args.config)
    else:
        config = DecodeConfig()

    if args.input_text:
        config.input_text = args.input_text
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.steps is not None:
        config.total_steps = args.steps

    loop = DiffusionDecodeLoop(config)
    result = loop.run()

    print(json.dumps({k: v for k, v in result.items() if k != "output_text"}, indent=2))
    print(f"\nOutput text:\n{result['output_text']}")


if __name__ == "__main__":
    main()
