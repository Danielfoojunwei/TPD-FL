"""
Abstract base class for diffusion model backends.

Every backend must expose:
  - tokenize / detokenize
  - forward_logits (M2T denoising)
  - optional forward_edit_logits (T2T editing)
  - vocab_size, mask_token_id, special_token_ids

The interface is device-agnostic: a CPU backend and a GPU backend
implement the same contract.
"""

from __future__ import annotations

import platform
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# CPU dtype auto-detection
# ---------------------------------------------------------------------------

def detect_cpu_bf16_support() -> bool:
    """Return True if the CPU supports efficient bfloat16 operations.

    Checks for AVX-512 BF16 or AMX-BF16 instruction sets via /proc/cpuinfo
    on Linux.  Falls back to False on other platforms or if detection fails.
    """
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read().lower()
            # AVX-512 BF16 or AMX-BF16
            return "avx512_bf16" in cpuinfo or "amx_bf16" in cpuinfo
    except Exception:
        pass
    return False


def resolve_dtype(requested: str = "auto", device: str = "cpu") -> torch.dtype:
    """Resolve the dtype to use.

    Parameters
    ----------
    requested : str
        One of ``"auto"``, ``"bf16"``, ``"fp32"``, ``"fp16"``.
    device : str
        Target device (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    torch.dtype
    """
    if requested == "fp32":
        return torch.float32
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    # auto
    if "cuda" in device:
        return torch.bfloat16  # safe default for modern GPUs
    if detect_cpu_bf16_support():
        return torch.bfloat16
    return torch.float32


# ---------------------------------------------------------------------------
# Backend abstract interface
# ---------------------------------------------------------------------------

@dataclass
class BackendConfig:
    """Shared configuration for all model backends."""
    model_id: str = ""
    device: str = "cpu"
    dtype: str = "auto"                # "auto" | "bf16" | "fp32" | "fp16"
    max_seq_len: int = 512
    trust_remote_code: bool = True
    # Diffusion parameters
    diffusion_steps: int = 64          # total T
    mask_token_id: Optional[int] = None
    # Download / cache
    cache_dir: Optional[str] = None


class DiffusionBackend(ABC):
    """Unified interface for diffusion language model backends.

    Both CPU-first (LLaDA 8B) and GPU (LLaDA2.1-mini) backends
    implement this contract so the TPD decode loop is backend-agnostic.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size of the underlying tokenizer/model."""
        ...

    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        """Token ID used for masked (unknown) positions in diffusion."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device the model is on."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Compute dtype."""
        ...

    @abstractmethod
    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None,
        return_offsets: bool = False,
    ) -> Dict:
        """Tokenize text.

        Returns
        -------
        dict with keys:
          - ``"input_ids"``: Tensor[1, L]
          - ``"attention_mask"``: Tensor[1, L]
          - ``"offset_mapping"``: List[Tuple[int,int]] (if return_offsets)
        """
        ...

    @abstractmethod
    def detokenize(self, token_ids) -> str:
        """Convert token IDs back to text.

        Parameters
        ----------
        token_ids : Tensor[L] or List[int]
        """
        ...

    @abstractmethod
    def forward_logits(
        self,
        tokens: torch.Tensor,
        step: int,
        positions: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute denoising logits for the specified positions.

        Parameters
        ----------
        tokens : Tensor[L]  — current token sequence (with masks).
        step : int           — current diffusion step.
        positions : Tensor[K] — indices of positions to denoise.
        conditioning : optional conditioning tensor.

        Returns
        -------
        logits : Tensor[K, V] — logits for each queried position.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience encode/decode (compatible with AllowedSetBuilder)
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to a list of token IDs.

        Default implementation delegates to :meth:`tokenize`.
        Subclasses backed by a real HF tokenizer should override this
        to call the tokenizer's native ``encode`` for efficiency.
        """
        result = self.tokenize(text)
        return result["input_ids"].squeeze(0).tolist()

    def decode(self, ids) -> str:
        """Decode a list of token IDs to text.

        Default implementation delegates to :meth:`detokenize`.
        """
        return self.detokenize(ids)

    def forward_logits_full(
        self,
        tokens: torch.Tensor,
        step: int,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits for ALL positions (used for diagnostics).

        Default implementation calls forward_logits with all positions.
        """
        L = tokens.shape[-1]
        return self.forward_logits(
            tokens, step, torch.arange(L, device=self.device), conditioning
        )

    def forward_edit_logits(
        self,
        tokens: torch.Tensor,
        step: int,
        positions: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Optional T2T edit logits.  Returns None if not supported."""
        return None

    def sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample token IDs from (projected) logits.

        Parameters
        ----------
        logits : Tensor[K, V]

        Returns
        -------
        token_ids : Tensor[K]
        """
        scaled = logits / max(temperature, 1e-8)

        if top_k > 0:
            topk_vals, _ = scaled.topk(min(top_k, scaled.size(-1)), dim=-1)
            scaled = scaled.masked_fill(scaled < topk_vals[..., -1:], torch.finfo(scaled.dtype).min)

        if top_p < 1.0:
            sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cum_probs - sorted_logits.softmax(dim=-1) >= top_p
            sorted_logits[remove] = torch.finfo(scaled.dtype).min
            scaled = torch.zeros_like(scaled).scatter_(-1, sorted_idx, sorted_logits)

        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs.float(), 1, generator=generator).squeeze(-1)


# ---------------------------------------------------------------------------
# Lightweight synthetic backend (for tests / CI without model weights)
# ---------------------------------------------------------------------------

class SyntheticBackend(DiffusionBackend):
    """Synthetic backend for unit tests and CI.

    Generates random or peaked logits so the full pipeline can run
    without downloading model weights.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        mask_token_id_val: int = 126336,
        mode: str = "uniform",
        peak_ids: Optional[List[int]] = None,
        peak_strength: float = 10.0,
        device_str: str = "cpu",
        seed: int = 42,
    ):
        self._vocab_size = vocab_size
        self._mask_token_id = mask_token_id_val
        self._device = torch.device(device_str)
        self._dtype = torch.float32
        self.mode = mode
        self.peak_ids = peak_ids or []
        self.peak_strength = peak_strength
        self._gen = torch.Generator(device=self._device)
        self._gen.manual_seed(seed)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def tokenize(self, text, max_length=None, return_offsets=False):
        ids = [min(ord(c), self._vocab_size - 1) for c in text]
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        t = torch.tensor([ids], dtype=torch.long, device=self._device)
        result = {"input_ids": t, "attention_mask": torch.ones_like(t)}
        if return_offsets:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(ids))]
        return result

    def detokenize(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(chr(i) if 32 <= i < 127 else f"[{i}]" for i in token_ids)

    def forward_logits(self, tokens, step, positions, conditioning=None):
        K = positions.shape[0]
        V = self._vocab_size
        if self.mode == "uniform":
            return torch.zeros(K, V, device=self._device)
        elif self.mode == "peaked":
            logits = torch.zeros(K, V, device=self._device)
            for tid in self.peak_ids:
                if tid < V:
                    logits[:, tid] = self.peak_strength
            return logits
        elif self.mode == "adversarial":
            logits = torch.randn(K, V, device=self._device, generator=self._gen)
            for tid in self.peak_ids:
                if tid < V:
                    logits[:, tid] = self.peak_strength
            return logits
        return torch.zeros(K, V, device=self._device)
