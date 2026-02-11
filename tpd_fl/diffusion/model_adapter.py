"""
Diffusion Model Adapter — clean abstraction for diffusion LLMs.

Provides a unified interface for:
  - forward_logits: get logits for positions at a given diffusion step
  - sample_tokens: sample from (projected) logits
  - edit_logits: (optional) T2T edit logits

Two concrete implementations:
  - SyntheticDiffusionModel: deterministic/random logits for testing
  - HFDiffusionModel: wrapper for HuggingFace-compatible models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SamplerConfig:
    """Configuration for token sampling."""
    temperature: float = 1.0
    top_k: int = 0          # 0 = no top-k
    top_p: float = 1.0      # 1.0 = no nucleus sampling
    seed: Optional[int] = None


class DiffusionModel(ABC):
    """Abstract base class for diffusion text models."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        ...

    @abstractmethod
    def forward_logits(
        self,
        tokens: torch.Tensor,
        step: int,
        positions_to_update: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute denoising logits for given positions.

        Parameters
        ----------
        tokens : Tensor[L] or [B, L] — current token sequence with masks.
        step : int — current diffusion step.
        positions_to_update : Tensor[K] — indices of positions to denoise.
        conditioning : optional conditioning (e.g., encoder hidden states).

        Returns
        -------
        logits : Tensor[K, V] or [B, K, V] — logits over vocabulary
            for each position to update.
        """
        ...

    @abstractmethod
    def forward_logits_full(
        self,
        tokens: torch.Tensor,
        step: int,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits for ALL positions (used for diagnostics).

        Returns
        -------
        logits : Tensor[L, V] or [B, L, V]
        """
        ...

    def sample_tokens(
        self,
        logits: torch.Tensor,
        config: Optional[SamplerConfig] = None,
    ) -> torch.Tensor:
        """Sample token IDs from (projected) logits.

        Parameters
        ----------
        logits : Tensor[K, V] or [B, K, V] — projected logits.
        config : sampling configuration.

        Returns
        -------
        token_ids : Tensor[K] or [B, K]
        """
        cfg = config or SamplerConfig()
        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(cfg.seed)

        # Apply temperature
        scaled = logits / max(cfg.temperature, 1e-8)

        # Apply top-k
        if cfg.top_k > 0:
            topk_vals, _ = scaled.topk(cfg.top_k, dim=-1)
            min_topk = topk_vals[..., -1:]
            scaled = scaled.masked_fill(scaled < min_topk, torch.finfo(scaled.dtype).min)

        # Apply top-p (nucleus)
        if cfg.top_p < 1.0:
            sorted_logits, sorted_indices = scaled.sort(dim=-1, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative_probs - sorted_logits.softmax(dim=-1) >= cfg.top_p
            sorted_logits[remove] = torch.finfo(scaled.dtype).min
            # Unsort
            scaled = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(scaled, dim=-1)

        if probs.dim() == 3:
            B, K, V = probs.shape
            ids = torch.stack([
                torch.multinomial(probs[b], 1, generator=generator).squeeze(-1)
                for b in range(B)
            ])
        else:
            ids = torch.multinomial(probs, 1, generator=generator).squeeze(-1)

        return ids

    def edit_logits(
        self,
        tokens: torch.Tensor,
        step: int,
        positions_to_edit: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Optional T2T edit logits. Returns None if not supported."""
        return None


class SyntheticDiffusionModel(DiffusionModel):
    """Synthetic diffusion model for testing and pipeline validation.

    Generates logits based on configurable patterns:
    - 'uniform': uniform logits over vocabulary
    - 'peaked': strong preference for a fixed set of token IDs
    - 'adversarial': high logits on *forbidden* tokens (for testing projection)

    This enables full end-to-end pipeline testing without a real model.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        mask_token_id_val: int = 0,
        mode: str = "uniform",
        peak_token_ids: Optional[list] = None,
        peak_strength: float = 10.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        self._vocab_size = vocab_size
        self._mask_token_id = mask_token_id_val
        self.mode = mode
        self.peak_token_ids = peak_token_ids or [1, 2, 3, 4, 5]
        self.peak_strength = peak_strength
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    def forward_logits(
        self,
        tokens: torch.Tensor,
        step: int,
        positions_to_update: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K = positions_to_update.shape[0]
        V = self._vocab_size
        return self._make_logits(K, V)

    def forward_logits_full(
        self,
        tokens: torch.Tensor,
        step: int,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        L = tokens.shape[-1]
        V = self._vocab_size
        return self._make_logits(L, V)

    def _make_logits(self, K: int, V: int) -> torch.Tensor:
        if self.mode == "uniform":
            return torch.zeros(K, V, device=self.device)
        elif self.mode == "peaked":
            logits = torch.zeros(K, V, device=self.device)
            for tid in self.peak_token_ids:
                if tid < V:
                    logits[:, tid] = self.peak_strength
            return logits
        elif self.mode == "adversarial":
            # Generate logits that strongly prefer specific tokens
            # (the test will set these to forbidden tokens)
            logits = torch.randn(K, V, device=self.device, generator=self._generator)
            for tid in self.peak_token_ids:
                if tid < V:
                    logits[:, tid] = self.peak_strength
            return logits
        else:
            return torch.zeros(K, V, device=self.device)


class HFDiffusionModel(DiffusionModel):
    """Wrapper for HuggingFace-compatible diffusion language models.

    Expects a model with a forward method that accepts input_ids and
    returns logits.  The diffusion step is encoded via the mask pattern
    in the input tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        mask_token_id_val: Optional[int] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self._mask_token_id = (
            mask_token_id_val
            if mask_token_id_val is not None
            else getattr(tokenizer, "mask_token_id", 0)
        )
        self._vocab_size = (
            tokenizer.vocab_size
            if hasattr(tokenizer, "vocab_size")
            else len(tokenizer)
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    def forward_logits(
        self,
        tokens: torch.Tensor,
        step: int,
        positions_to_update: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            input_ids = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
            input_ids = input_ids.to(self._device)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            # Select positions
            if logits.dim() == 3:
                logits = logits[0]  # remove batch dim
            return logits[positions_to_update]

    def forward_logits_full(
        self,
        tokens: torch.Tensor,
        step: int,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            input_ids = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
            input_ids = input_ids.to(self._device)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if logits.dim() == 3:
                logits = logits[0]
            return logits
