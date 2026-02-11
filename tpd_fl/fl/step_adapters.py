"""
Per-Diffusion-Step LoRA Adapter Bank.

In diffusion language models the denoising process progresses through
multiple steps, each with different noise characteristics.  Early steps
(high noise) require broad corrections while late steps (low noise)
refine details.  This module manages a bank of independent LoRA adapters
indexed by step range, allowing different adapter weights for different
phases of the diffusion process.

The adapter bank partitions the total diffusion steps ``[0, T)`` into
contiguous, non-overlapping ranges.  At each step, exactly one adapter
is active.  This provides:

- **Phase-specialised adaptation**: early-step adapters can learn coarse
  privacy-aware patterns while late-step adapters handle fine-grained
  token selection.
- **Federated granularity**: clients can send per-range deltas, enabling
  the server to aggregate at the phase level.

Usage::

    from tpd_fl.fl.step_adapters import StepAdapterBank, StepAdapterConfig
    from tpd_fl.fl.lora import LoRAConfig

    config = StepAdapterConfig(num_ranges=4, lora=LoRAConfig(rank=8))
    bank = StepAdapterBank(model, config)
    adapter = bank.get_adapter(step=10, total_steps=64)
    # adapter is a dict {name: LoRALinear} for the active range

This module depends only on torch, the Python standard library, and
:mod:`tpd_fl.fl.lora`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tpd_fl.fl.lora import (
    LoRAConfig,
    LoRALinear,
    attach_lora,
    detach_lora,
    get_lora_state_dict,
    load_lora_state_dict,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StepAdapterConfig:
    """Configuration for the step-based adapter bank.

    Attributes
    ----------
    num_ranges : int
        Number of step ranges.  The total diffusion steps are divided
        into ``num_ranges`` contiguous intervals of roughly equal size.
    lora : LoRAConfig
        LoRA hyper-parameters shared by all adapters in the bank.
        Each range gets an independent copy of LoRA weights.
    custom_boundaries : Optional[List[float]]
        If provided, a list of ``num_ranges - 1`` fractions in (0, 1)
        defining the boundaries between ranges.  For example,
        ``[0.25, 0.5, 0.75]`` for 4 equal-sized ranges.  When ``None``,
        boundaries are computed as uniform splits.
    """

    num_ranges: int = 4
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    custom_boundaries: Optional[List[float]] = None

    def get_boundaries(self) -> List[float]:
        """Return the range boundaries as fractions of total steps.

        Returns a list of ``num_ranges - 1`` values in (0, 1).
        """
        if self.custom_boundaries is not None:
            if len(self.custom_boundaries) != self.num_ranges - 1:
                raise ValueError(
                    f"custom_boundaries must have {self.num_ranges - 1} "
                    f"elements, got {len(self.custom_boundaries)}"
                )
            return list(self.custom_boundaries)
        # Uniform split
        return [i / self.num_ranges for i in range(1, self.num_ranges)]


# ---------------------------------------------------------------------------
# Adapter Bank
# ---------------------------------------------------------------------------

class StepAdapterBank:
    """Manages a bank of LoRA adapters indexed by diffusion step range.

    The bank holds ``num_ranges`` independent sets of LoRA parameters.
    Only one set is "active" (injected into the model) at any time.
    Switching adapters is performed lazily: calling :meth:`get_adapter`
    with a step in a different range triggers a swap.

    Parameters
    ----------
    model : nn.Module
        The base model to adapt.  The bank will attach/detach LoRA
        modules as needed.
    config : StepAdapterConfig
        Bank configuration.

    Notes
    -----
    - The base model parameters are frozen once during initialisation.
    - Adapter state dicts are stored on CPU when not active, and moved
      to the model device when activated.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[StepAdapterConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or StepAdapterConfig()
        self._boundaries = self.config.get_boundaries()

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Create the first adapter set to discover module names
        initial_modules = attach_lora(model, self.config.lora)
        self._module_names: List[str] = list(initial_modules.keys())

        # Store the initial (zero-init) state dict as the template
        template_state = get_lora_state_dict(initial_modules)

        # Detach the initial adapter so we start clean
        detach_lora(model, initial_modules)

        # Create per-range state dicts (independent copies)
        self._range_states: List[Dict[str, torch.Tensor]] = []
        for _ in range(self.config.num_ranges):
            self._range_states.append(
                {k: v.clone() for k, v in template_state.items()}
            )

        # Track currently active range (-1 = none active)
        self._active_range: int = -1
        self._active_modules: Optional[Dict[str, LoRALinear]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_ranges(self) -> int:
        """Number of step ranges in the bank."""
        return self.config.num_ranges

    @property
    def active_range(self) -> int:
        """Index of the currently active adapter range, or -1 if none."""
        return self._active_range

    @property
    def active_modules(self) -> Optional[Dict[str, LoRALinear]]:
        """The currently active LoRA modules, or None."""
        return self._active_modules

    def step_to_range(self, step: int, total_steps: int) -> int:
        """Map a diffusion step to a range index.

        Parameters
        ----------
        step : int
            Current diffusion step (0-indexed).
        total_steps : int
            Total number of diffusion steps.

        Returns
        -------
        int
            Range index in ``[0, num_ranges)``.
        """
        if total_steps <= 0:
            return 0
        frac = step / total_steps
        for i, boundary in enumerate(self._boundaries):
            if frac < boundary:
                return i
        return self.config.num_ranges - 1

    def get_adapter(
        self,
        step: int,
        total_steps: int,
    ) -> Dict[str, LoRALinear]:
        """Return the LoRA modules for the adapter covering *step*.

        If the requested range differs from the currently active one,
        this method saves the active adapter's state, detaches it,
        attaches the target adapter, and loads its state.

        Parameters
        ----------
        step : int
            Current diffusion step.
        total_steps : int
            Total number of diffusion steps.

        Returns
        -------
        Dict[str, LoRALinear]
            Mapping of module names to active LoRA layers.
        """
        range_idx = self.step_to_range(step, total_steps)

        if range_idx == self._active_range and self._active_modules is not None:
            return self._active_modules

        # Save current adapter state if one is active
        self._save_active()

        # Detach current adapter
        self._detach_active()

        # Attach new adapter
        new_modules = attach_lora(self.model, self.config.lora)
        load_lora_state_dict(new_modules, self._range_states[range_idx])

        self._active_range = range_idx
        self._active_modules = new_modules
        return new_modules

    def save_all_states(self) -> List[Dict[str, torch.Tensor]]:
        """Return a copy of all per-range state dicts.

        Ensures the currently active adapter's latest weights are
        included.

        Returns
        -------
        List[Dict[str, Tensor]]
            One state dict per range.
        """
        self._save_active()
        return [
            {k: v.clone() for k, v in sd.items()}
            for sd in self._range_states
        ]

    def load_all_states(
        self,
        states: List[Dict[str, torch.Tensor]],
    ) -> None:
        """Load per-range state dicts, replacing existing weights.

        Parameters
        ----------
        states : List[Dict[str, Tensor]]
            One state dict per range.
        """
        if len(states) != self.config.num_ranges:
            raise ValueError(
                f"Expected {self.config.num_ranges} state dicts, "
                f"got {len(states)}"
            )
        for i, sd in enumerate(states):
            self._range_states[i] = {k: v.clone() for k, v in sd.items()}

        # Reload active adapter if one is attached
        if self._active_modules is not None and self._active_range >= 0:
            load_lora_state_dict(
                self._active_modules,
                self._range_states[self._active_range],
            )

    def deactivate(self) -> None:
        """Detach the currently active adapter from the model."""
        self._save_active()
        self._detach_active()

    def get_range_state(self, range_idx: int) -> Dict[str, torch.Tensor]:
        """Return a copy of the state dict for a specific range.

        Parameters
        ----------
        range_idx : int
            Range index in ``[0, num_ranges)``.

        Returns
        -------
        Dict[str, Tensor]
        """
        if range_idx < 0 or range_idx >= self.config.num_ranges:
            raise IndexError(
                f"range_idx {range_idx} out of bounds "
                f"[0, {self.config.num_ranges})"
            )
        # Make sure active state is up-to-date
        if range_idx == self._active_range and self._active_modules is not None:
            self._save_active()
        return {k: v.clone() for k, v in self._range_states[range_idx].items()}

    def set_range_state(
        self,
        range_idx: int,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Set the state dict for a specific range.

        Parameters
        ----------
        range_idx : int
            Range index.
        state_dict : Dict[str, Tensor]
            New state dict.
        """
        if range_idx < 0 or range_idx >= self.config.num_ranges:
            raise IndexError(
                f"range_idx {range_idx} out of bounds "
                f"[0, {self.config.num_ranges})"
            )
        self._range_states[range_idx] = {k: v.clone() for k, v in state_dict.items()}
        # Reload if this range is currently active
        if range_idx == self._active_range and self._active_modules is not None:
            load_lora_state_dict(self._active_modules, self._range_states[range_idx])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_active(self) -> None:
        """Persist the active adapter's weights to ``_range_states``."""
        if self._active_modules is not None and self._active_range >= 0:
            self._range_states[self._active_range] = get_lora_state_dict(
                self._active_modules
            )

    def _detach_active(self) -> None:
        """Detach the currently active adapter from the model."""
        if self._active_modules is not None:
            detach_lora(self.model, self._active_modules)
            self._active_modules = None
            self._active_range = -1
