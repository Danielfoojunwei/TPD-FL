"""
Diagnostics — allowed-mass Z_i measurement and logging.

For each position *i* with type τ_i, the allowed mass is:

    Z_i = Σ_{v ∈ A(τ_i)} softmax(logits_i)[v]

Z_i = 1.0 for PUB positions (entire vocabulary allowed).
Z_i < 1.0 for SENS/REG positions, quantifying the projection penalty.

This module computes Z_i statistics and persists per-run JSONL logs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES


@dataclass
class ZStats:
    """Summary statistics for allowed-mass distribution."""
    count: int = 0
    mean: float = 0.0
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    below_01: int = 0    # count where Z_i < 0.01
    below_05: int = 0    # count where Z_i < 0.05
    below_10: int = 0    # count where Z_i < 0.10

    def to_dict(self) -> Dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "min": self.min_val,
            "max": self.max_val,
            "below_01": self.below_01,
            "below_05": self.below_05,
            "below_10": self.below_10,
        }


def compute_allowed_mass(
    logits: torch.Tensor,
    pos_type: List[SpanType],
    allowed_masks: Dict[SpanType, torch.Tensor],
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Z_i for each position.

    Parameters
    ----------
    logits : Tensor[L, V] — raw (pre-projection) logits.
    pos_type : per-position type list.
    allowed_masks : per-type boolean masks [V].
    positions : optional subset of positions to compute.

    Returns
    -------
    z_values : Tensor[L] with Z_i per position (1.0 for PUB).
    """
    L, V = logits.shape
    probs = torch.softmax(logits, dim=-1)
    z_values = torch.ones(L, device=logits.device)

    idx_list = positions.tolist() if positions is not None else list(range(L))

    for idx in idx_list:
        stype = pos_type[idx]
        if stype not in SENSITIVE_TYPES:
            continue
        allow = allowed_masks.get(stype, allowed_masks.get(SpanType.SENS))
        if allow is None:
            continue
        z_values[idx] = probs[idx][allow].sum()

    return z_values


def compute_z_stats(z_values: torch.Tensor, pos_type: List[SpanType]) -> ZStats:
    """Compute summary statistics over sensitive positions only."""
    sensitive_mask = torch.tensor(
        [pt in SENSITIVE_TYPES for pt in pos_type],
        dtype=torch.bool,
        device=z_values.device,
    )
    z_sens = z_values[sensitive_mask]

    if z_sens.numel() == 0:
        return ZStats()

    z_cpu = z_sens.float().cpu()
    return ZStats(
        count=int(z_cpu.numel()),
        mean=float(z_cpu.mean()),
        p10=float(z_cpu.quantile(0.1)),
        p50=float(z_cpu.quantile(0.5)),
        p90=float(z_cpu.quantile(0.9)),
        min_val=float(z_cpu.min()),
        max_val=float(z_cpu.max()),
        below_01=int((z_cpu < 0.01).sum()),
        below_05=int((z_cpu < 0.05).sum()),
        below_10=int((z_cpu < 0.10).sum()),
    )


class DiagnosticsLogger:
    """Per-run JSONL logger for Z_i diagnostics and step metadata."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self._records: List[Dict] = []
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = self.log_dir / "diagnostics.jsonl"
        else:
            self._log_path = None

    def log_step(
        self,
        step: int,
        total_steps: int,
        phase: str,
        z_stats: ZStats,
        num_updated: int = 0,
        num_repaired: int = 0,
        extra: Optional[Dict] = None,
    ):
        """Log a single diffusion step."""
        record = {
            "timestamp": time.time(),
            "step": step,
            "total_steps": total_steps,
            "phase": phase,
            "z_stats": z_stats.to_dict(),
            "num_updated": num_updated,
            "num_repaired": num_repaired,
        }
        if extra:
            record.update(extra)
        self._records.append(record)

        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def log_final(
        self,
        metrics: Dict,
    ):
        """Log final decode result metrics."""
        record = {
            "timestamp": time.time(),
            "event": "final",
            **metrics,
        }
        self._records.append(record)
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    @property
    def records(self) -> List[Dict]:
        return list(self._records)

    def summary(self) -> Dict:
        """Return aggregate summary across all logged steps."""
        if not self._records:
            return {}
        step_records = [r for r in self._records if "step" in r and "event" not in r]
        if not step_records:
            return {}
        z_means = [r["z_stats"]["mean"] for r in step_records if r["z_stats"]["count"] > 0]
        return {
            "num_steps_logged": len(step_records),
            "avg_z_mean": sum(z_means) / len(z_means) if z_means else 0.0,
            "total_updated": sum(r["num_updated"] for r in step_records),
            "total_repaired": sum(r["num_repaired"] for r in step_records),
        }
