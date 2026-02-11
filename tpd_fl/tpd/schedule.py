"""
Policy-driven mask schedule for diffusion decoding.

Defines three phases that govern which positions may be updated at each
diffusion step:

  DRAFT  — only PUB positions are updatable.
  SAFE   — PUB + SENS/REG are updatable, but SENS/REG are constrained
            by their A(type) masks (enforced by the projection engine).
  REVEAL — only PUB + explicit reveal-allowlist (off by default).

The schedule intersects the proposed update mask M_t (from the diffusion
sampler) with the set of positions allowed by the current phase, ensuring
that sensitive positions are never written outside of constrained phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set

import torch

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES


class SchedulePhase(Enum):
    DRAFT = auto()
    SAFE = auto()
    REVEAL = auto()


@dataclass
class ScheduleConfig:
    """Configure the phase boundaries (as fractions of total steps T)."""
    # Fraction of steps [0, draft_end) → DRAFT phase
    draft_end: float = 0.4
    # Fraction of steps [draft_end, safe_end) → SAFE phase
    safe_end: float = 0.9
    # Fraction of steps [safe_end, 1.0] → REVEAL phase
    # (REVEAL is off by default — acts like SAFE unless reveal_types set)
    reveal_types: List[SpanType] = field(default_factory=list)
    # Selection strategy for M_t within allowed positions
    selection: str = "random"   # "random" | "entropy"


class MaskSchedule:
    """Implements the three-phase mask schedule.

    Given the current step ``t`` out of ``T`` total steps, and the
    per-position type assignments ``pos_type``, the schedule computes
    which positions are eligible for update.
    """

    def __init__(self, config: Optional[ScheduleConfig] = None):
        self.config = config or ScheduleConfig()
        self._reveal_set: FrozenSet[SpanType] = frozenset(self.config.reveal_types)

    def phase(self, t: int, T: int) -> SchedulePhase:
        """Return the phase for step *t* given *T* total steps."""
        frac = t / max(T, 1)
        if frac < self.config.draft_end:
            return SchedulePhase.DRAFT
        elif frac < self.config.safe_end:
            return SchedulePhase.SAFE
        else:
            return SchedulePhase.REVEAL

    def allowed_positions(
        self,
        t: int,
        T: int,
        pos_type: List[SpanType],
        device: str = "cpu",
    ) -> torch.Tensor:
        """Return a boolean mask [L] of positions allowed for update.

        This mask is intersected with the sampler's proposed M_t before
        any tokens are written.
        """
        L = len(pos_type)
        phase = self.phase(t, T)
        allowed = torch.zeros(L, dtype=torch.bool, device=device)

        if phase == SchedulePhase.DRAFT:
            # Only public positions
            for i, pt in enumerate(pos_type):
                if pt == SpanType.PUB:
                    allowed[i] = True

        elif phase == SchedulePhase.SAFE:
            # All positions — but SENS/REG will be constrained by projection
            allowed.fill_(True)

        elif phase == SchedulePhase.REVEAL:
            # PUB always; SENS/REG only if in reveal set
            for i, pt in enumerate(pos_type):
                if pt == SpanType.PUB:
                    allowed[i] = True
                elif pt in self._reveal_set:
                    allowed[i] = True
                elif pt not in SENSITIVE_TYPES:
                    allowed[i] = True

        return allowed

    def apply_schedule(
        self,
        proposed_mask: torch.Tensor,
        t: int,
        T: int,
        pos_type: List[SpanType],
    ) -> torch.Tensor:
        """Intersect proposed update mask with schedule-allowed positions.

        Parameters
        ----------
        proposed_mask : Tensor[L], bool
            The sampler's proposed positions to update at this step.
        t, T : int
            Current step and total number of steps.
        pos_type : List[SpanType]
            Per-position type assignments.

        Returns
        -------
        Tensor[L], bool — the final update mask.
        """
        device = proposed_mask.device
        allowed = self.allowed_positions(t, T, pos_type, device=str(device))
        return proposed_mask & allowed


def build_random_block_mask(
    seq_len: int,
    num_to_update: int,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build a random block-selection mask for ``num_to_update`` positions."""
    num_to_update = min(num_to_update, seq_len)
    perm = torch.randperm(seq_len, generator=generator, device=device)
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[perm[:num_to_update]] = True
    return mask
