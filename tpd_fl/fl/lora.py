"""
Low-Rank Adaptation (LoRA) for Diffusion LLMs.

Implements LoRA as described in Hu et al. (2021), adapted for the
TPD+FL privacy-preserving federated learning pipeline.  LoRA
decomposes weight updates into low-rank matrices A and B:

    W' = W + (alpha / rank) * B @ A

where A is initialised from N(0, sigma^2) and B from zeros, so that
the initial perturbation is zero.  Only A and B are trained; the
base model weights W remain frozen.

Key operations:
  - attach_lora:  inject LoRA layers into an existing model
  - detach_lora:  remove LoRA layers and restore originals
  - merge_lora:   fold LoRA weights into base weights (for inference)
  - unmerge_lora: reverse a previous merge
  - get_lora_state_dict / load_lora_state_dict: serialisation helpers

This module depends only on torch and the Python standard library.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Configuration for LoRA injection.

    Attributes
    ----------
    rank : int
        Rank of the low-rank decomposition.  Typical values: 4, 8, 16.
    alpha : float
        Scaling factor.  The effective scaling applied to the low-rank
        update is ``alpha / rank``.
    dropout : float
        Dropout probability applied to the input before the low-rank
        path.  Set to 0.0 to disable.
    target_modules : List[str]
        List of regex patterns or exact substrings matched against
        fully-qualified module names (e.g. ``["q_proj", "v_proj"]``).
        Only ``nn.Linear`` modules whose name matches at least one
        pattern will receive LoRA adapters.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
    )


# ---------------------------------------------------------------------------
# LoRA Linear Layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """A single LoRA adapter wrapping an existing ``nn.Linear`` layer.

    The forward pass computes::

        out = original_linear(x) + scale * dropout(x) @ A^T @ B^T

    where ``scale = alpha / rank``.

    Parameters
    ----------
    original : nn.Linear
        The frozen linear layer being adapted.
    rank : int
        Low-rank dimension.
    alpha : float
        LoRA scaling factor.
    dropout : float
        Dropout probability on the LoRA input path.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # A: (rank, in_features) — initialised with Kaiming-uniform
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: (out_features, rank) — initialised to zero so initial delta is 0
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Optional dropout on the LoRA input path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Track whether the LoRA delta has been merged into base weights
        self._merged = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the adapted linear output.

        If the LoRA weights have been merged into the base layer, this
        simply calls the original linear (which already contains the
        merged weights).  Otherwise it adds the low-rank residual.
        """
        base_out = self.original(x)
        if self._merged:
            return base_out

        # Low-rank path: x @ A^T => (*, rank) => @ B^T => (*, out_features)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scale * lora_out

    # ------------------------------------------------------------------
    # Merge / unmerge
    # ------------------------------------------------------------------

    def merge(self) -> None:
        """Fold the LoRA delta into the base weight for efficient inference."""
        if self._merged:
            return
        with torch.no_grad():
            # delta = scale * B @ A  shape: (out_features, in_features)
            delta = self.scale * (self.lora_B @ self.lora_A)
            self.original.weight.add_(delta)
        self._merged = True

    def unmerge(self) -> None:
        """Reverse a previous merge, restoring the original base weight."""
        if not self._merged:
            return
        with torch.no_grad():
            delta = self.scale * (self.lora_B @ self.lora_A)
            self.original.weight.sub_(delta)
        self._merged = False

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"in={self.original.in_features}, "
            f"out={self.original.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"merged={self._merged}"
        )


# ---------------------------------------------------------------------------
# Attach / Detach helpers
# ---------------------------------------------------------------------------

def _matches_target(name: str, target_modules: List[str]) -> bool:
    """Return True if *name* matches any pattern in *target_modules*.

    Patterns are treated as regex if they contain regex meta-characters;
    otherwise they are matched as plain substrings.
    """
    for pattern in target_modules:
        # Try regex first
        try:
            if re.search(pattern, name):
                return True
        except re.error:
            # Fall back to substring match
            if pattern in name:
                return True
    return False


def attach_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> Dict[str, LoRALinear]:
    """Inject LoRA adapters into *model* for all matching ``nn.Linear`` layers.

    This function:
    1. Freezes **all** base-model parameters.
    2. For each ``nn.Linear`` whose fully-qualified name matches
       ``config.target_modules``, creates a :class:`LoRALinear` wrapper
       and replaces the original module in the model tree.
    3. Returns a mapping ``{qualified_name: LoRALinear}`` for later use
       by :func:`detach_lora`, :func:`get_lora_state_dict`, etc.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.  Base parameters are frozen in-place.
    config : LoRAConfig
        LoRA hyper-parameters and target module patterns.

    Returns
    -------
    Dict[str, LoRALinear]
        Mapping from fully-qualified module name to the injected
        LoRA wrapper.
    """
    # Step 1: freeze all existing parameters
    for param in model.parameters():
        param.requires_grad = False

    lora_modules: Dict[str, LoRALinear] = {}

    # Step 2: find matching nn.Linear modules and replace them
    # We need to walk the module tree and replace children in their parent.
    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_target(name, config.target_modules):
            continue

        # Build the LoRA wrapper
        lora_layer = LoRALinear(
            original=module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )

        # Replace in the parent module
        _replace_module(model, name, lora_layer)
        lora_modules[name] = lora_layer

    return lora_modules


def detach_lora(
    model: nn.Module,
    lora_modules: Dict[str, LoRALinear],
) -> None:
    """Remove LoRA wrappers, restoring the original ``nn.Linear`` layers.

    If any LoRA module has been merged, it is unmerged first so that
    the base weights are restored to their pre-LoRA state.

    Parameters
    ----------
    model : nn.Module
        The model that was previously adapted with :func:`attach_lora`.
    lora_modules : Dict[str, LoRALinear]
        The mapping returned by :func:`attach_lora`.
    """
    for name, lora_layer in lora_modules.items():
        # Unmerge if needed so base weights are clean
        lora_layer.unmerge()
        # Restore the original nn.Linear
        _replace_module(model, name, lora_layer.original)


def _replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace a submodule in *root* identified by a dot-separated path.

    For example, ``_replace_module(model, "encoder.layer.0.attn.q_proj", lora)``
    sets ``model.encoder.layer[0].attn.q_proj = lora``.
    """
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module  # type: ignore[index]
    else:
        setattr(parent, last, new_module)


# ---------------------------------------------------------------------------
# State dict helpers
# ---------------------------------------------------------------------------

def get_lora_state_dict(
    lora_modules: Dict[str, LoRALinear],
) -> Dict[str, torch.Tensor]:
    """Extract LoRA parameters as a flat state dict.

    Keys have the form ``"<module_name>.lora_A"`` and
    ``"<module_name>.lora_B"``.

    Parameters
    ----------
    lora_modules : Dict[str, LoRALinear]
        Mapping returned by :func:`attach_lora`.

    Returns
    -------
    Dict[str, Tensor]
        State dict containing only the LoRA A and B matrices.
    """
    state: Dict[str, torch.Tensor] = {}
    for name, lora_layer in lora_modules.items():
        state[f"{name}.lora_A"] = lora_layer.lora_A.detach().clone()
        state[f"{name}.lora_B"] = lora_layer.lora_B.detach().clone()
    return state


def load_lora_state_dict(
    lora_modules: Dict[str, LoRALinear],
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Load LoRA parameters from a flat state dict.

    Parameters
    ----------
    lora_modules : Dict[str, LoRALinear]
        Mapping returned by :func:`attach_lora`.
    state_dict : Dict[str, Tensor]
        State dict as produced by :func:`get_lora_state_dict`.
    strict : bool
        If True, raise ``KeyError`` when expected keys are missing.
    """
    for name, lora_layer in lora_modules.items():
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"

        if a_key in state_dict:
            lora_layer.lora_A.data.copy_(state_dict[a_key])
        elif strict:
            raise KeyError(f"Missing key in state_dict: {a_key}")

        if b_key in state_dict:
            lora_layer.lora_B.data.copy_(state_dict[b_key])
        elif strict:
            raise KeyError(f"Missing key in state_dict: {b_key}")


# ---------------------------------------------------------------------------
# Merge / Unmerge (batch)
# ---------------------------------------------------------------------------

def merge_lora(lora_modules: Dict[str, LoRALinear]) -> None:
    """Merge all LoRA deltas into the base model weights.

    After merging, forward passes through the model use only the
    (modified) base weights — no additional LoRA computation.  This
    is useful for efficient inference after training.
    """
    for lora_layer in lora_modules.values():
        lora_layer.merge()


def unmerge_lora(lora_modules: Dict[str, LoRALinear]) -> None:
    """Reverse :func:`merge_lora`, restoring original base weights.

    After unmerging, the LoRA adapters are once again applied as a
    separate low-rank residual during forward passes.
    """
    for lora_layer in lora_modules.values():
        lora_layer.unmerge()
