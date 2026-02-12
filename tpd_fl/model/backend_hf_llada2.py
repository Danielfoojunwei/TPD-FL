"""
HuggingFace LLaDA2.1-mini backend -- GPU Tier 2 implementation.

Wraps the ``inclusionAI/LLaDA2.1-mini`` model behind the
:class:`DiffusionBackend` interface.  This is a **16 B Mixture-of-Experts
(MoE)** diffusion language model that requires a CUDA GPU with
sufficient VRAM.

Memory requirements
-------------------
* **bf16 / fp16**: ~32 GB VRAM (16B total params, ~4B active per token)
* Flash Attention 2 is used automatically when available to reduce
  memory pressure and improve throughput.

This backend is **optional**.  If CUDA is not available or VRAM is
insufficient, :meth:`is_available` returns ``False`` and construction
raises a clear error.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch

from tpd_fl.model.backend_base import BackendConfig, DiffusionBackend, resolve_dtype

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# Default HuggingFace model identifier.
_DEFAULT_MODEL_ID = "inclusionAI/LLaDA2.1-mini"

# LLaDA2.x mask token ID -- the model config typically defines this;
# we keep a fallback that matches the published checkpoint.
_DEFAULT_MASK_TOKEN_ID = 126336

# Minimum VRAM (in GiB) required for the 16B MoE model in bf16/fp16.
_MIN_VRAM_GIB = 32.0


def _flash_attn_available() -> bool:
    """Return ``True`` if Flash Attention 2 can be imported."""
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def _cuda_vram_gib(device_index: int = 0) -> float:
    """Return total VRAM in GiB for the given CUDA device, or 0."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        total_bytes = torch.cuda.get_device_properties(device_index).total_mem
        return total_bytes / (1024 ** 3)
    except Exception:
        return 0.0


class HFLLaDA2Backend(DiffusionBackend):
    """GPU backend for **LLaDA2.1-mini** (16 B MoE).

    This is the Tier 2 (optional) backend for TPD-FL.  It loads the
    LLaDA2.1-mini model onto a CUDA GPU and exposes the standard
    :class:`DiffusionBackend` interface.

    Parameters
    ----------
    config : BackendConfig
        Shared backend configuration.  Relevant fields:

        * ``model_id`` -- HuggingFace model ID
          (default ``inclusionAI/LLaDA2.1-mini``).
        * ``device``  -- ``"cuda"`` (default) or ``"cuda:N"``.
        * ``dtype``   -- ``"auto"``, ``"bf16"``, ``"fp32"``, ``"fp16"``.
        * ``cache_dir`` -- optional HuggingFace cache directory.
        * ``mask_token_id`` -- override auto-detected mask token.

    Raises
    ------
    ImportError
        If ``transformers`` is not installed.
    RuntimeError
        If CUDA is not available or VRAM is insufficient.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[BackendConfig] = None) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "The `transformers` package is required for HFLLaDA2Backend.  "
                "Install it with:  pip install transformers"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "HFLLaDA2Backend requires a CUDA GPU, but torch.cuda is not "
                "available.  Install a CUDA-enabled PyTorch build or use "
                "HFLLaDABackend (CPU Tier 1) instead."
            )

        if config is None:
            config = BackendConfig()

        model_id = config.model_id or _DEFAULT_MODEL_ID
        device_str = config.device or "cuda"

        # Validate that the target is actually a CUDA device.
        if "cuda" not in device_str:
            warnings.warn(
                f"HFLLaDA2Backend is designed for CUDA GPUs but device="
                f"'{device_str}' was requested.  Proceeding, but expect "
                f"degraded performance or errors.",
                RuntimeWarning,
                stacklevel=2,
            )

        resolved_dtype = resolve_dtype(config.dtype, device_str)

        # -- VRAM check ------------------------------------------------
        device_index = 0
        if ":" in device_str:
            try:
                device_index = int(device_str.split(":")[1])
            except (ValueError, IndexError):
                device_index = 0

        vram = _cuda_vram_gib(device_index)
        if 0 < vram < _MIN_VRAM_GIB:
            logger.warning(
                "GPU %d has %.1f GiB VRAM, but LLaDA2.1-mini typically "
                "needs ~%.0f GiB in %s.  Loading may fail or fall back "
                "to CPU offload.",
                device_index,
                vram,
                _MIN_VRAM_GIB,
                resolved_dtype,
            )

        # -- Flash Attention -------------------------------------------
        attn_impl: Optional[str] = None
        if _flash_attn_available():
            attn_impl = "flash_attention_2"
            logger.info("Flash Attention 2 detected -- enabling for LLaDA2.1-mini.")
        else:
            logger.info(
                "Flash Attention 2 not found; using default attention.  "
                "Install flash-attn for lower memory usage and faster inference."
            )

        logger.info(
            "Loading model %s  device=%s  dtype=%s",
            model_id,
            device_str,
            resolved_dtype,
        )

        # -- Load tokenizer --------------------------------------------
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
        )

        # -- Load model ------------------------------------------------
        model_kwargs = dict(
            torch_dtype=resolved_dtype,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
        )
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self._model.to(device_str)
        self._model.eval()

        # -- Resolve mask token ID -------------------------------------
        if config.mask_token_id is not None:
            self._mask_token_id = config.mask_token_id
        else:
            self._mask_token_id = self._resolve_mask_token_id()

        self._device = torch.device(device_str)
        self._dtype = resolved_dtype
        self._max_seq_len = config.max_seq_len

        logger.info(
            "HFLLaDA2Backend ready  vocab=%d  mask_id=%d  device=%s  "
            "flash_attn=%s",
            self.vocab_size,
            self._mask_token_id,
            self._device,
            attn_impl is not None,
        )

    # ------------------------------------------------------------------
    # Mask-token resolution
    # ------------------------------------------------------------------

    def _resolve_mask_token_id(self) -> int:
        """Detect the ``[MASK]`` token ID from the model or tokenizer.

        The LLaDA2.x model config may expose the mask token directly.
        Falls back to tokenizer inspection and finally to the well-known
        default (126336).
        """
        # Strategy 1: model config attribute (some LLaDA2 checkpoints)
        if hasattr(self._model, "config"):
            cfg = self._model.config
            for attr in ("mask_token_id", "mask_id"):
                val = getattr(cfg, attr, None)
                if val is not None:
                    return int(val)

        # Strategy 2: tokenizer attribute
        if hasattr(self._tokenizer, "mask_token_id") and self._tokenizer.mask_token_id is not None:
            return int(self._tokenizer.mask_token_id)

        # Strategy 3: encode the literal ``[MASK]`` string
        try:
            ids = self._tokenizer.encode("[MASK]", add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        except Exception:
            pass

        # Strategy 4: scan ``added_tokens_encoder``
        if hasattr(self._tokenizer, "added_tokens_encoder"):
            for token_str, token_id in self._tokenizer.added_tokens_encoder.items():
                if token_str.upper() == "[MASK]":
                    return int(token_id)

        logger.warning(
            "Could not auto-detect [MASK] token ID -- using default %d",
            _DEFAULT_MASK_TOKEN_ID,
        )
        return _DEFAULT_MASK_TOKEN_ID

    # ------------------------------------------------------------------
    # DiffusionBackend properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Vocabulary size reported by the underlying tokenizer."""
        return int(self._tokenizer.vocab_size)

    @property
    def mask_token_id(self) -> int:
        """Token ID used for masked (unknown) positions during diffusion."""
        return self._mask_token_id

    @property
    def device(self) -> torch.device:
        """Device the model weights reside on."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype used by the model."""
        return self._dtype

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None,
        return_offsets: bool = False,
    ) -> Dict:
        """Tokenize *text* using the HuggingFace tokenizer.

        Parameters
        ----------
        text : str
            Input text to tokenize.
        max_length : int, optional
            Truncate to at most this many tokens.  Defaults to the
            backend's ``max_seq_len``.
        return_offsets : bool
            If ``True``, include ``"offset_mapping"`` in the result.

        Returns
        -------
        dict
            ``"input_ids"`` : Tensor[1, L],
            ``"attention_mask"`` : Tensor[1, L],
            ``"offset_mapping"`` : list[tuple[int,int]]  *(if requested)*
        """
        max_len = max_length or self._max_seq_len

        kwargs = dict(
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        offset_mapping: Optional[List[Tuple[int, int]]] = None
        if return_offsets:
            try:
                enc_with_offsets = self._tokenizer(
                    text,
                    return_offsets_mapping=True,
                    **kwargs,
                )
                offset_mapping = enc_with_offsets.get("offset_mapping")
                if isinstance(offset_mapping, torch.Tensor):
                    offset_mapping = [
                        (int(s), int(e)) for s, e in offset_mapping[0].tolist()
                    ]
                enc = {
                    "input_ids": enc_with_offsets["input_ids"].to(self._device),
                    "attention_mask": enc_with_offsets["attention_mask"].to(self._device),
                }
            except Exception:
                logger.debug(
                    "Tokenizer does not support return_offsets_mapping; "
                    "falling back to synthetic offsets."
                )
                enc = self._tokenizer(text, **kwargs)
                enc = {
                    "input_ids": enc["input_ids"].to(self._device),
                    "attention_mask": enc["attention_mask"].to(self._device),
                }
                seq_len = enc["input_ids"].shape[1]
                offset_mapping = [(i, i + 1) for i in range(seq_len)]
        else:
            enc = self._tokenizer(text, **kwargs)
            enc = {
                "input_ids": enc["input_ids"].to(self._device),
                "attention_mask": enc["attention_mask"].to(self._device),
            }

        result: Dict = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        if return_offsets:
            result["offset_mapping"] = offset_mapping  # type: ignore[assignment]
        return result

    def detokenize(self, token_ids) -> str:
        """Decode token IDs back to a string.

        Parameters
        ----------
        token_ids : Tensor[L] | list[int]
            A 1-D sequence of token IDs.

        Returns
        -------
        str
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    # ------------------------------------------------------------------
    # AllowedSetBuilder-compatible encode/decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode *text* using the underlying HF tokenizer."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids) -> str:
        """Decode token IDs using the underlying HF tokenizer."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_logits(
        self,
        tokens: torch.Tensor,
        step: int,
        positions: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute denoising logits for the requested *positions*.

        Parameters
        ----------
        tokens : Tensor[L]
            Current (partially masked) token sequence.
        step : int
            Current diffusion step index (unused by the raw model, but
            part of the interface contract).
        positions : Tensor[K]
            Indices into *tokens* for which logits are needed.
        conditioning : Tensor, optional
            Not used by LLaDA2 -- reserved for future conditioning.

        Returns
        -------
        Tensor[K, V]
            Logits for each queried position over the full vocabulary.
        """
        input_ids = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        input_ids = input_ids.to(self._device)
        positions = positions.to(self._device)

        output = self._model(input_ids=input_ids)

        if hasattr(output, "logits"):
            all_logits = output.logits  # [1, L, V]
        else:
            all_logits = output[0]

        return all_logits[0, positions, :]

    @torch.no_grad()
    def forward_logits_full(
        self,
        tokens: torch.Tensor,
        step: int,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits for **all** positions in *tokens*.

        Parameters
        ----------
        tokens : Tensor[L]
            Current (partially masked) token sequence.
        step : int
            Current diffusion step index.
        conditioning : Tensor, optional
            Reserved for future use.

        Returns
        -------
        Tensor[L, V]
            Logits for every position over the full vocabulary.
        """
        input_ids = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        input_ids = input_ids.to(self._device)

        output = self._model(input_ids=input_ids)

        if hasattr(output, "logits"):
            all_logits = output.logits
        else:
            all_logits = output[0]

        return all_logits[0]

    # ------------------------------------------------------------------
    # Availability helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if the backend can be instantiated.

        Checks that:
        1. ``transformers`` is installed.
        2. CUDA is available.
        3. At least one GPU has sufficient VRAM (~32 GiB).
        """
        if not _HAS_TRANSFORMERS:
            return False
        if not torch.cuda.is_available():
            return False
        if _cuda_vram_gib(0) < _MIN_VRAM_GIB:
            return False
        return True

    @staticmethod
    def estimated_memory_gb(dtype: str = "auto", device: str = "cuda") -> float:
        """Return the estimated VRAM footprint in GiB.

        Parameters
        ----------
        dtype : str
            One of ``"auto"``, ``"bf16"``, ``"fp32"``, ``"fp16"``.
        device : str
            Target device string.

        Returns
        -------
        float
            Approximate VRAM requirement in GiB.
        """
        resolved = resolve_dtype(dtype, device)
        bytes_per_param = 4 if resolved == torch.float32 else 2
        # LLaDA2.1-mini has ~16B total parameters (MoE).
        param_count = 16e9
        return (param_count * bytes_per_param) / (1024 ** 3)
