"""
Diffusion Decode Loop with TPD hooks — backend-agnostic, CPU-first.

Implements the full M2T (mask-to-token) decode loop with integrated TPD hooks:

  Hook A — Schedule:    Intersects proposed update mask with phase-allowed positions.
  Hook B — Projection:  Applies support-restriction Π_{A(τ_i)} to logits.
  Hook C — Verifier:    Runs Okπ after each step (or periodically).
  Hook D — Repair:      Triggers repair if verifier rejects.
  Hook E — Diagnostics: Logs Z_i statistics per step.

The loop uses the :class:`DiffusionBackend` abstraction from
``tpd_fl.model.backend_base`` and works identically on CPU and GPU.

CLI entry point::

    python -m tpd_fl.diffusion.decode_loop --config configs/decode/tpd_full.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from tpd_fl.tpd.typing import SpanType, SpanTyper, SENSITIVE_TYPES
from tpd_fl.tpd.allowed_sets import AllowedSetBuilder, AllowedSetConfig
from tpd_fl.tpd.schedule import MaskSchedule, ScheduleConfig
from tpd_fl.tpd.projection import ProjectionEngine, project_logits
from tpd_fl.tpd.verifier import Verifier, VerifierConfig
from tpd_fl.tpd.repair import RepairEngine, RepairMode
from tpd_fl.tpd.diagnostics import (
    DiagnosticsLogger,
    compute_allowed_mass,
    compute_z_stats,
)


@dataclass
class DecodeConfig:
    """Full configuration for a TPD-enabled diffusion decode run."""
    # Backend selection
    backend: str = "synthetic"          # "synthetic" | "llada8b" | "llada2"
    model_id: str = ""
    device: str = "cpu"
    dtype: str = "auto"                 # "auto" | "bf16" | "fp32"

    # Diffusion
    total_steps: int = 64
    seq_len: int = 128
    tokens_per_step_frac: float = 0.15

    # Sampling
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 42

    # TPD Schedule
    schedule_enabled: bool = True
    schedule_draft_end: float = 0.4
    schedule_safe_end: float = 0.9

    # TPD Projection
    projection_enabled: bool = True

    # TPD Verifier
    verifier_enabled: bool = True
    verifier_check_every: int = 4
    verifier_forbidden_tags: List[str] = field(
        default_factory=lambda: ["EMAIL", "PHONE", "SSN", "CC", "ID"]
    )
    known_secrets: List[str] = field(default_factory=list)

    # TPD Repair
    repair_enabled: bool = True
    repair_mode: str = "resample"

    # Diagnostics
    diagnostics_enabled: bool = True
    output_dir: str = "runs/decode"

    # Input
    input_text: str = ""

    # Typing
    typing_use_ner: bool = False


def load_decode_config(path: str) -> DecodeConfig:
    """Load DecodeConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    fields = {f.name for f in DecodeConfig.__dataclass_fields__.values()}
    return DecodeConfig(**{k: v for k, v in raw.items() if k in fields})


def _build_backend(cfg: DecodeConfig):
    """Instantiate the appropriate DiffusionBackend."""
    if cfg.backend == "synthetic":
        from tpd_fl.model.backend_base import SyntheticBackend
        return SyntheticBackend(device_str=cfg.device, seed=cfg.seed)
    elif cfg.backend == "llada8b":
        from tpd_fl.model.backend_hf_llada import HFLLaDABackend
        from tpd_fl.model.backend_base import BackendConfig
        bcfg = BackendConfig(
            model_id=cfg.model_id or "GSAI-ML/LLaDA-8B-Instruct",
            device=cfg.device, dtype=cfg.dtype,
            max_seq_len=cfg.seq_len,
            diffusion_steps=cfg.total_steps,
        )
        return HFLLaDABackend(bcfg)
    elif cfg.backend == "llada2":
        from tpd_fl.model.backend_hf_llada2 import HFLLaDA2Backend
        from tpd_fl.model.backend_base import BackendConfig
        bcfg = BackendConfig(
            model_id=cfg.model_id or "inclusionAI/LLaDA2.1-mini",
            device=cfg.device, dtype=cfg.dtype,
            max_seq_len=cfg.seq_len,
            diffusion_steps=cfg.total_steps,
        )
        return HFLLaDA2Backend(bcfg)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")


class DiffusionDecodeLoop:
    """Full diffusion decode loop with TPD hooks.

    Usage::

        loop = DiffusionDecodeLoop(config, backend=my_backend)
        result = loop.run(input_text="Contact john@acme.com for details.")
    """

    def __init__(
        self,
        config: Optional[DecodeConfig] = None,
        backend=None,
    ):
        self.config = config or DecodeConfig()
        self.backend = backend or _build_backend(self.config)
        self.device = str(self.backend.device)

        # TPD components
        self.typer = SpanTyper(use_ner=self.config.typing_use_ner)
        self.allowed_builder = AllowedSetBuilder(
            self.backend,  # acts as tokenizer (has encode/decode)
            AllowedSetConfig(),
            device=self.device,
        )
        self.schedule = MaskSchedule(ScheduleConfig(
            draft_end=self.config.schedule_draft_end,
            safe_end=self.config.schedule_safe_end,
        )) if self.config.schedule_enabled else None

        self.verifier = Verifier(VerifierConfig(
            forbidden_tags=self.config.verifier_forbidden_tags,
            known_secrets=self.config.known_secrets,
        )) if self.config.verifier_enabled else None

        self.repair_engine = RepairEngine(
            mode=RepairMode.RESAMPLE if self.config.repair_mode == "resample" else RepairMode.EDIT,
        ) if self.config.repair_enabled else None

        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(self.config.seed)

    def run(
        self,
        input_text: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Run the full decode loop and return metrics.

        Parameters
        ----------
        input_text : source text (may contain PII to redact).
        output_dir : directory for logs and artifacts.

        Returns
        -------
        dict with keys: output_text, metrics, diagnostics_summary.
        """
        text = input_text or self.config.input_text or "Hello world."
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        diag = DiagnosticsLogger(str(out_dir)) if self.config.diagnostics_enabled else None
        backend = self.backend

        # ── Tokenize ──
        tok_result = backend.tokenize(text, max_length=self.config.seq_len, return_offsets=True)
        input_ids = tok_result["input_ids"].squeeze(0)  # [L]
        offset_mapping = tok_result.get("offset_mapping", [(i, i+1) for i in range(len(input_ids))])
        seq_len = input_ids.shape[0]

        # ── Span typing ──
        spans, pos_type, pos_span_id = self.typer.type_text(text, offset_mapping, seq_len)

        # ── Allowed masks ──
        allowed_masks = self._build_allowed_masks(backend, pos_type)

        # ── Projection engine ──
        proj = ProjectionEngine(allowed_masks, pos_type)

        # ── Init tokens: all masked ──
        tokens = torch.full((seq_len,), backend.mask_token_id, dtype=torch.long, device=self.device)

        T = self.config.total_steps
        per_step = max(1, int(seq_len * self.config.tokens_per_step_frac))
        start = time.time()
        total_updated = 0
        total_repaired = 0

        for t in range(T):
            masked = (tokens == backend.mask_token_id)
            n_masked = masked.sum().item()
            if n_masked == 0:
                break

            # ── Propose positions ──
            masked_idx = masked.nonzero(as_tuple=True)[0]
            k = min(per_step, n_masked)
            perm = torch.randperm(len(masked_idx), generator=self._gen, device=self.device)
            proposed = masked_idx[perm[:k]]

            proposed_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
            proposed_mask[proposed] = True

            # ── Hook A: Schedule ──
            if self.schedule:
                final_mask = self.schedule.apply_schedule(proposed_mask, t, T, pos_type)
            else:
                final_mask = proposed_mask
            positions = final_mask.nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                continue

            # ── Forward logits ──
            logits = backend.forward_logits(tokens, t, positions)

            # ── Hook E: Diagnostics (pre-projection) ──
            if diag:
                try:
                    full_logits = backend.forward_logits_full(tokens, t)
                    z_vals = compute_allowed_mass(full_logits, pos_type, allowed_masks)
                    z_stats = compute_z_stats(z_vals, pos_type)
                except Exception:
                    z_stats = None

            # ── Hook B: Projection ──
            if self.config.projection_enabled:
                local_types = [pos_type[p] for p in positions.tolist()]
                logits = project_logits(
                    logits, local_types, allowed_masks,
                    positions=torch.arange(len(positions), device=self.device),
                )

            # ── Sample ──
            sampled = backend.sample_tokens(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                generator=self._gen,
            )
            tokens[positions] = sampled
            total_updated += len(positions)

            # ── Hook C + D: Verifier + Repair ──
            step_repaired = 0
            if self.verifier and (t % max(self.config.verifier_check_every, 1) == 0):
                decoded = backend.detokenize(tokens)
                vr = self.verifier.check(decoded)
                if not vr.ok and self.repair_engine:
                    violating = []
                    for v in vr.violations:
                        if "positions" in v:
                            violating.extend(v["positions"])
                    if violating:
                        def model_fn(tok, s, pos, cond=None):
                            return backend.forward_logits(tok, s, pos, cond)
                        tokens, _ = self.repair_engine.repair(
                            tokens, violating, pos_type, allowed_masks,
                            model_fn=model_fn, step=t,
                            mask_token_id=backend.mask_token_id,
                            temperature=self.config.temperature,
                        )
                        step_repaired = len(violating)
                        total_repaired += step_repaired

            # ── Log step ──
            if diag and z_stats:
                phase = self.schedule.phase(t, T).name if self.schedule else "NONE"
                diag.log_step(t, T, phase, z_stats, len(positions), step_repaired)

        # ── Final ──
        output_text = backend.detokenize(tokens)
        elapsed = time.time() - start
        hard_ok = proj.verify_hard_guarantee(tokens)
        final_v = self.verifier.check(output_text) if self.verifier else None

        metrics = {
            "output_text": output_text,
            "seq_len": seq_len,
            "total_steps": T,
            "tokens_updated": total_updated,
            "tokens_repaired": total_repaired,
            "elapsed_sec": round(elapsed, 3),
            "throughput_tok_per_sec": round(total_updated / max(elapsed, 1e-6), 1),
            "hard_guarantee_holds": hard_ok,
            "verifier_pass": final_v.ok if final_v else None,
            "verifier_violations": len(final_v.violations) if final_v else 0,
            "num_sensitive_positions": sum(1 for p in pos_type if p in SENSITIVE_TYPES),
            "num_spans_detected": len(spans),
        }
        if diag:
            diag.log_final(metrics)
            metrics["diagnostics_summary"] = diag.summary()

        # Save artifacts
        _save(out_dir, self.config, metrics, spans)
        return metrics

    def _build_allowed_masks(self, backend, pos_type):
        """Build allowed-set masks for the tokenizer.

        For real tokenizers we use AllowedSetBuilder; for synthetic
        we create simple masks based on the vocab size.
        """
        from tpd_fl.model.backend_base import SyntheticBackend
        if isinstance(backend, SyntheticBackend):
            V = backend.vocab_size
            pub = torch.ones(V, dtype=torch.bool, device=self.device)
            sens = torch.zeros(V, dtype=torch.bool, device=self.device)
            # Allow first half of vocab for SENS
            sens[:V // 2] = True
            reg = torch.zeros(V, dtype=torch.bool, device=self.device)
            reg[:V // 4] = True
            masks = {SpanType.PUB: pub, SpanType.SENS: sens, SpanType.REG: reg}
            for st in SENSITIVE_TYPES:
                if st not in masks:
                    masks[st] = sens.clone()
            return masks
        # Real tokenizer path
        return self.allowed_builder.build()


def _save(out_dir: Path, config, metrics, spans):
    """Persist config snapshot, metrics, and span info."""
    with open(out_dir / "config.json", "w") as f:
        json.dump({k: v for k, v in config.__dict__.items()}, f, indent=2, default=str)
    ser = {k: v for k, v in metrics.items() if isinstance(v, (str, int, float, bool, type(None), dict))}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(ser, f, indent=2)
    with open(out_dir / "spans.json", "w") as f:
        json.dump([{"start": s.start, "end": s.end, "type": s.type.name, "tag": s.entity_tag} for s in spans], f, indent=2)


# ───────── CLI ─────────

def main():
    parser = argparse.ArgumentParser(description="TPD Diffusion Decode (CPU-first)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-text", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    args = parser.parse_args()

    cfg = load_decode_config(args.config) if args.config else DecodeConfig()
    if args.input_text:
        cfg.input_text = args.input_text
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.steps is not None:
        cfg.total_steps = args.steps
    if args.backend:
        cfg.backend = args.backend
    if args.device:
        cfg.device = args.device
    if args.dtype:
        cfg.dtype = args.dtype

    loop = DiffusionDecodeLoop(cfg)
    result = loop.run()
    print(json.dumps({k: v for k, v in result.items() if k != "output_text"}, indent=2))
    print(f"\nOutput:\n{result['output_text'][:500]}")


if __name__ == "__main__":
    main()
