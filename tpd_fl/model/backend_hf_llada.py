"""
HuggingFace LLaDA 8B backend -- CPU-first Tier 1 implementation.

Wraps the ``GSAI-ML/LLaDA-8B-Instruct`` model behind the
:class:`DiffusionBackend` interface so the TPD decode loop can run
on commodity hardware without a GPU.

Memory requirements
-------------------
* **fp32**: ~32 GB RAM (8B params x 4 bytes)
* **bf16**: ~16 GB RAM (8B params x 2 bytes)

CPU inference is functional but slow -- expect minutes per diffusion
run on a modern desktop.  Use bf16 on CPUs with AVX-512 BF16 / AMX
support for a ~2x speedup.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch

from tpd_fl.model.backend_base import BackendConfig, DiffusionBackend, resolve_dtype

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports -- allow the rest of the library to load even when
# ``transformers`` is not installed.
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# Default HuggingFace model identifier.
_DEFAULT_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

# LLaDA uses token ID 126336 for ``[MASK]``.
_DEFAULT_MASK_TOKEN_ID = 126336


class HFLLaDABackend(DiffusionBackend):
    """CPU-first backend for **LLaDA 8B Instruct**.

    This is the Tier 1 (default) backend for TPD-FL.  It loads the full
    LLaDA 8B model onto the CPU (or an available CUDA device when
    explicitly requested) and exposes the standard
    :class:`DiffusionBackend` interface for the TPD decode loop.

    Parameters
    ----------
    config : BackendConfig
        Shared backend configuration.  The following fields are
        particularly relevant:

        * ``model_id`` -- HuggingFace model ID
          (default ``GSAI-ML/LLaDA-8B-Instruct``).
        * ``device``  -- ``"cpu"`` (default) or ``"cuda"``.
        * ``dtype``   -- ``"auto"``, ``"bf16"``, ``"fp32"``, or ``"fp16"``.
        * ``cache_dir`` -- optional HuggingFace cache directory.
        * ``mask_token_id`` -- override auto-detected mask token.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[BackendConfig] = None) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "The `transformers` package is required for HFLLaDABackend.  "
                "Install it with:  pip install transformers"
            )

        if config is None:
            config = BackendConfig()

        model_id = config.model_id or _DEFAULT_MODEL_ID
        device_str = config.device or "cpu"
        resolved_dtype = resolve_dtype(config.dtype, device_str)

        # -- Warn about CPU performance --------------------------------
        if "cpu" in device_str:
            logger.warning(
                "Loading LLaDA 8B on CPU (dtype=%s).  Inference will be "
                "slow -- expect several minutes per diffusion run.  "
                "Use a bf16-capable CPU (AVX-512 BF16 / AMX) or switch "
                "to a GPU backend for interactive speeds.",
                resolved_dtype,
            )
            mem_gb = 16.0 if resolved_dtype in (torch.bfloat16, torch.float16) else 32.0
            logger.info(
                "Estimated CPU memory requirement for LLaDA 8B: ~%.0f GB",
                mem_gb,
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
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=resolved_dtype,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
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
            "HFLLaDABackend ready  vocab=%d  mask_id=%d  device=%s",
            self.vocab_size,
            self._mask_token_id,
            self._device,
        )

    # ------------------------------------------------------------------
    # Mask-token resolution
    # ------------------------------------------------------------------

    def _resolve_mask_token_id(self) -> int:
        """Detect the ``[MASK]`` token ID from the tokenizer.

        Falls back to the well-known default (126336) used by
        ``GSAI-ML/LLaDA-8B-Instruct``.
        """
        # Strategy 1: tokenizer attribute
        if hasattr(self._tokenizer, "mask_token_id") and self._tokenizer.mask_token_id is not None:
            return int(self._tokenizer.mask_token_id)

        # Strategy 2: encode the literal ``[MASK]`` string
        try:
            ids = self._tokenizer.encode("[MASK]", add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        except Exception:
            pass

        # Strategy 3: check ``added_tokens_encoder``
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

        # Build base kwargs.
        kwargs = dict(
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        # Attempt to retrieve byte-offset mapping.
        offset_mapping: Optional[List[Tuple[int, int]]] = None
        if return_offsets:
            try:
                enc_with_offsets = self._tokenizer(
                    text,
                    return_offsets_mapping=True,
                    **kwargs,
                )
                offset_mapping = enc_with_offsets.get("offset_mapping")
                # HF returns a tensor for offset_mapping -- convert
                if isinstance(offset_mapping, torch.Tensor):
                    offset_mapping = [
                        (int(s), int(e)) for s, e in offset_mapping[0].tolist()
                    ]
                enc = {
                    "input_ids": enc_with_offsets["input_ids"].to(self._device),
                    "attention_mask": enc_with_offsets["attention_mask"].to(self._device),
                }
            except Exception:
                # Some tokenizers do not support offset mapping -- fall
                # back to a plain encode and synthesise offsets.
                logger.debug(
                    "Tokenizer does not support return_offsets_mapping; "
                    "falling back to synthetic offsets."
                )
                enc = self._tokenizer(text, **kwargs)
                enc = {
                    "input_ids": enc["input_ids"].to(self._device),
                    "attention_mask": enc["attention_mask"].to(self._device),
                }
                # Synthesise character-level offsets as a rough fallback.
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
            Not used by LLaDA -- reserved for future conditioning
            mechanisms.

        Returns
        -------
        Tensor[K, V]
            Logits for each queried position over the full vocabulary.
        """
        # Ensure 2-D input: [1, L]
        input_ids = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        input_ids = input_ids.to(self._device)
        positions = positions.to(self._device)

        output = self._model(input_ids=input_ids)

        # The model may return a ModelOutput with ``.logits`` or a raw
        # tuple whose first element is the logits tensor.
        if hasattr(output, "logits"):
            all_logits = output.logits  # [1, L, V]
        else:
            all_logits = output[0]

        # Extract the requested positions -> [K, V]
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

        # [1, L, V] -> [L, V]
        return all_logits[0]

    # ------------------------------------------------------------------
    # Availability helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if the backend can be instantiated.

        Checks that the ``transformers`` library is installed.  Does
        **not** verify that model weights are cached locally.
        """
        return _HAS_TRANSFORMERS

    @staticmethod
    def estimated_memory_gb(dtype: str = "auto", device: str = "cpu") -> float:
        """Return the estimated memory footprint in GiB.

        Parameters
        ----------
        dtype : str
            One of ``"auto"``, ``"bf16"``, ``"fp32"``, ``"fp16"``.
        device : str
            Target device string.

        Returns
        -------
        float
            Approximate memory requirement in GiB.
        """
        resolved = resolve_dtype(dtype, device)
        bytes_per_param = 4 if resolved == torch.float32 else 2
        param_count = 8e9  # 8 billion parameters
        return (param_count * bytes_per_param) / (1024 ** 3)
