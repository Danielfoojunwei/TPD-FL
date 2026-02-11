"""
Logits Projection — support restriction Π_{A(τ_i)}.

The projection engine sets logits for forbidden tokens to −∞ (or
``finfo.min`` for numerical safety).  This is the **hard guarantee**
at the core of TPD: after projection, sampling from the resulting
distribution *cannot* produce a forbidden token.

The implementation groups positions by SpanType so that a single
``masked_fill_`` call handles all positions of each type, yielding
efficient GPU execution.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES


def project_logits(
    logits: torch.Tensor,
    pos_type: List[SpanType],
    allowed_masks: Dict[SpanType, torch.Tensor],
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply support-restriction projection Π in-place.

    Parameters
    ----------
    logits : Tensor of shape [L, V] or [B, L, V]
        Raw logits from the diffusion model.  Modified **in-place** and
        returned.
    pos_type : List[SpanType] of length L
        Per-position type assignment from the span typer.
    allowed_masks : Dict[SpanType, Tensor[V]]
        Boolean allowed-set masks built by :class:`AllowedSetBuilder`.
    positions : Optional Tensor[K]
        If given, only project these positions (used when the decode loop
        only updates a subset of positions at each step).

    Returns
    -------
    logits : same tensor, modified in-place with forbidden entries set
        to ``finfo.min``.

    Guarantee
    ---------
    For every position *i* with type τ_i ∈ SENSITIVE_TYPES and every
    token *v* ∉ A(τ_i):  ``softmax(logits)[i, v] == 0`` after
    projection (up to floating-point flush).
    """
    batched = logits.dim() == 3
    if batched:
        B, L, V = logits.shape
    else:
        L, V = logits.shape
        B = None

    neg_inf = torch.finfo(logits.dtype).min

    # Group positions by type for efficient masked_fill
    type_to_indices: Dict[SpanType, List[int]] = defaultdict(list)

    if positions is not None:
        idx_list = positions.tolist()
    else:
        idx_list = list(range(L))

    for idx in idx_list:
        stype = pos_type[idx]
        if stype in SENSITIVE_TYPES:
            type_to_indices[stype].append(idx)

    for stype, indices in type_to_indices.items():
        if not indices:
            continue
        allow = allowed_masks.get(stype)
        if allow is None:
            # Fall back to SENS mask
            allow = allowed_masks.get(SpanType.SENS)
        if allow is None:
            continue

        # forbidden = ~allow  → boolean [V]
        forbidden = ~allow

        if batched:
            for idx in indices:
                logits[:, idx, forbidden] = neg_inf
        else:
            for idx in indices:
                logits[idx, forbidden] = neg_inf

    return logits


class ProjectionEngine:
    """Stateful projection engine caching allowed-masks for repeated use.

    Wraps :func:`project_logits` with a cached mask dictionary and
    provides convenience methods for the decode loop.
    """

    def __init__(
        self,
        allowed_masks: Dict[SpanType, torch.Tensor],
        pos_type: List[SpanType],
    ):
        self.allowed_masks = allowed_masks
        self.pos_type = pos_type

    def project(
        self,
        logits: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project logits in-place and return them."""
        return project_logits(
            logits, self.pos_type, self.allowed_masks, positions
        )

    def verify_hard_guarantee(
        self,
        token_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> bool:
        """Check that sampled token_ids respect allowed sets.

        Returns True if all sampled tokens at sensitive positions
        are in the allowed set.  This is a *runtime assertion* that
        should always hold after projection.
        """
        L = token_ids.shape[-1]
        idx_list = positions.tolist() if positions is not None else list(range(L))

        for idx in idx_list:
            stype = self.pos_type[idx]
            if stype not in SENSITIVE_TYPES:
                continue
            allow = self.allowed_masks.get(stype, self.allowed_masks.get(SpanType.SENS))
            if allow is None:
                continue
            tid = int(token_ids[idx]) if token_ids.dim() == 1 else int(token_ids[..., idx])
            if tid < allow.shape[0] and not allow[tid]:
                return False
        return True
