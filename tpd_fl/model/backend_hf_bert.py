"""
BERT-base Masked LM backend for TPD empirical evaluation.

Uses ``bert-base-uncased`` (110M params, open weights) as a real
diffusion-compatible backbone.  BERT naturally performs masked token
prediction — the same operation as a diffusion LM's denoising step —
making it a legitimate backbone for evaluating TPD's projection
guarantee on real model logits.

This backend downloads real weights from HuggingFace and runs real
forward passes on CPU.  No mocks, no fakes, no simulations.

Reference:
    Devlin et al., "BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding", NAACL 2019.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from tpd_fl.model.backend_base import DiffusionBackend, BackendConfig


class HFBertMLMBackend(DiffusionBackend):
    """HuggingFace BERT-base-uncased backend for empirical TPD evaluation.

    Uses the real pretrained BERT masked LM to produce logits for
    masked positions.  The model is loaded with real weights and runs
    real forward passes — no synthetic data.

    Parameters
    ----------
    config : BackendConfig, optional
        Backend configuration.  Defaults to bert-base-uncased on CPU.
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        cfg = config or BackendConfig(model_id="bert-base-uncased")
        model_id = cfg.model_id or "bert-base-uncased"

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForMaskedLM.from_pretrained(model_id)
        self._model.eval()

        self._device = torch.device(cfg.device)
        self._dtype = torch.float32  # BERT runs fine in fp32 on CPU
        self._model.to(self._device)

        self._mask_token_id = self._tokenizer.mask_token_id
        self._vocab_size = self._tokenizer.vocab_size
        self._max_seq_len = min(cfg.max_seq_len, 512)

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

    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None,
        return_offsets: bool = False,
    ) -> Dict:
        max_len = max_length or self._max_seq_len

        enc = self._tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=return_offsets,
        )

        result: Dict = {
            "input_ids": enc["input_ids"].to(self._device),
            "attention_mask": enc["attention_mask"].to(self._device),
        }
        if return_offsets and "offset_mapping" in enc:
            result["offset_mapping"] = enc["offset_mapping"][0].tolist()
        elif return_offsets:
            L = result["input_ids"].shape[1]
            result["offset_mapping"] = [(i, i + 1) for i in range(L)]
        return result

    def detokenize(self, token_ids) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    @torch.no_grad()
    def forward_logits(
        self,
        tokens: torch.Tensor,
        step: int,
        positions: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute real BERT MLM logits for the given positions.

        Runs a real forward pass through BERT-base-uncased with
        110M parameters and real pretrained weights.
        """
        input_ids = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        input_ids = input_ids.to(self._device)
        positions = positions.to(self._device)

        output = self._model(input_ids=input_ids)
        all_logits = output.logits  # [1, L, V]
        return all_logits[0, positions, :]  # [K, V]

    @torch.no_grad()
    def forward_logits_full(
        self,
        tokens: torch.Tensor,
        step: int,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        input_ids = input_ids.to(self._device)
        output = self._model(input_ids=input_ids)
        return output.logits[0]  # [L, V]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False
