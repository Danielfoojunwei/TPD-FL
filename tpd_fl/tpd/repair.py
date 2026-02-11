"""
Repair Engine — monotone-safe repair for verifier violations.

Two modes:
1. **Resample-only**: re-mask violating spans and redo a projected
   denoise step.  The projection guarantees the re-sampled tokens are
   in A(type).
2. **Edit-repair**: run a projected T2T edit step restricted to
   placeholder-only output for violating spans.

Both modes are *monotone in sensitivity*: they never introduce tokens
from a forbidden class that was not already present.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES
from tpd_fl.tpd.projection import project_logits


class RepairMode(Enum):
    RESAMPLE = auto()
    EDIT = auto()


class RepairEngine:
    """Repair violating positions after a verifier rejection.

    The engine receives the violating positions, the current token
    sequence, and either re-samples those positions (RESAMPLE mode)
    or runs an edit step (EDIT mode).  In both cases, the projection
    engine is applied to ensure monotone safety.
    """

    def __init__(
        self,
        mode: RepairMode = RepairMode.RESAMPLE,
        max_repair_iters: int = 3,
        placeholder_token_id: Optional[int] = None,
    ):
        self.mode = mode
        self.max_repair_iters = max_repair_iters
        self.placeholder_token_id = placeholder_token_id

    def repair(
        self,
        tokens: torch.Tensor,
        violating_positions: List[int],
        pos_type: List[SpanType],
        allowed_masks: Dict[SpanType, torch.Tensor],
        model_fn=None,
        step: int = 0,
        conditioning: Optional[torch.Tensor] = None,
        mask_token_id: int = 0,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, int]:
        """Repair violating positions.

        Parameters
        ----------
        tokens : Tensor[L] — current token sequence.
        violating_positions : positions that violate the policy.
        pos_type : per-position type assignments.
        allowed_masks : per-type allowed-set boolean masks [V].
        model_fn : callable(tokens, step, positions) -> logits [K, V]
            (only needed for RESAMPLE mode).
        step : current diffusion step.
        conditioning : optional conditioning tensor.
        mask_token_id : ID of the [MASK] token for re-masking.
        temperature : sampling temperature.

        Returns
        -------
        (repaired_tokens, num_iterations)
        """
        if self.mode == RepairMode.RESAMPLE:
            return self._resample_repair(
                tokens, violating_positions, pos_type, allowed_masks,
                model_fn, step, conditioning, mask_token_id, temperature,
            )
        elif self.mode == RepairMode.EDIT:
            return self._edit_repair(
                tokens, violating_positions, pos_type, allowed_masks,
            )
        else:
            raise ValueError(f"Unknown repair mode: {self.mode}")

    def _resample_repair(
        self,
        tokens: torch.Tensor,
        violating_positions: List[int],
        pos_type: List[SpanType],
        allowed_masks: Dict[SpanType, torch.Tensor],
        model_fn,
        step: int,
        conditioning: Optional[torch.Tensor],
        mask_token_id: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, int]:
        """Re-mask violating spans and redo projected denoise."""
        tokens = tokens.clone()
        remaining = list(violating_positions)

        for iteration in range(self.max_repair_iters):
            if not remaining:
                break

            # Re-mask violating positions
            pos_tensor = torch.tensor(remaining, dtype=torch.long, device=tokens.device)
            tokens[pos_tensor] = mask_token_id

            # Get logits from model
            if model_fn is not None:
                logits = model_fn(tokens, step, pos_tensor, conditioning)
                # logits shape: [K, V] where K = len(remaining)
            else:
                # Fallback: uniform logits (for testing without a model)
                V = max(m.shape[0] for m in allowed_masks.values())
                logits = torch.zeros(len(remaining), V, device=tokens.device)

            # Build pos_type subset for the violating positions
            sub_pos_type = [pos_type[p] for p in remaining]

            # Project
            logits = project_logits(
                logits,
                sub_pos_type,
                allowed_masks,
                positions=torch.arange(len(remaining), device=logits.device),
            )

            # Sample
            probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Write back
            tokens[pos_tensor] = sampled

            # Check which positions are still violating
            new_remaining = []
            for i, pos in enumerate(remaining):
                stype = pos_type[pos]
                if stype in SENSITIVE_TYPES:
                    allow = allowed_masks.get(stype, allowed_masks.get(SpanType.SENS))
                    if allow is not None and not allow[int(tokens[pos])]:
                        new_remaining.append(pos)
            remaining = new_remaining

        return tokens, iteration + 1

    def _edit_repair(
        self,
        tokens: torch.Tensor,
        violating_positions: List[int],
        pos_type: List[SpanType],
        allowed_masks: Dict[SpanType, torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Replace violating positions with placeholder tokens directly."""
        tokens = tokens.clone()

        if self.placeholder_token_id is not None:
            ph_id = self.placeholder_token_id
        else:
            # Find the first allowed token for each position
            ph_id = None

        for pos in violating_positions:
            stype = pos_type[pos]
            if stype not in SENSITIVE_TYPES:
                continue
            allow = allowed_masks.get(stype, allowed_masks.get(SpanType.SENS))
            if allow is None:
                continue

            if ph_id is not None and allow[ph_id]:
                tokens[pos] = ph_id
            else:
                # Pick the first allowed token
                allowed_ids = allow.nonzero(as_tuple=True)[0]
                if len(allowed_ids) > 0:
                    tokens[pos] = allowed_ids[0]

        return tokens, 1
