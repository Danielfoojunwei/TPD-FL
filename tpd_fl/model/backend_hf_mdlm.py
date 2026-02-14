"""
MDLM (Masked Diffusion Language Model) backend for TPD empirical evaluation.

Uses ``kuleshov-group/mdlm-owt`` (170M params, NeurIPS 2024, open weights)
as a proper discrete diffusion language model backbone.

Unlike BERT (which is trained with 15% random masking), MDLM is trained
with a continuous noise schedule covering *all* mask ratios from 0% to 100%.
This makes it a legitimate diffusion LM that is well-calibrated for
iterative denoising from fully masked to fully unmasked.

Architecture: DiT (Diffusion Transformer) with 12 blocks, 768 hidden dim,
12 heads, rotary embeddings, adaptive layer norm (adaLN).

Tokenizer: GPT-2 BPE (50,257 tokens) + 1 mask token (id=50,257).

Reference:
    Sahoo et al., "Simple and Effective Masked Diffusion Language Models",
    NeurIPS 2024. https://arxiv.org/abs/2406.07524
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from tpd_fl.model.backend_base import DiffusionBackend, BackendConfig


class HFMDLMBackend(DiffusionBackend):
    """HuggingFace MDLM backend — proper discrete diffusion LM.

    Loads MDLM-OWT (170M params) trained on OpenWebText with proper
    masked diffusion training (continuous noise schedule).  Runs on CPU
    using our patched modeling code (flash_attn → standard PyTorch attention).

    Parameters
    ----------
    config : BackendConfig, optional
        Backend configuration.
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        from tpd_fl.model.configuration_mdlm import MDLMConfig
        from tpd_fl.model.modeling_mdlm import MDLM

        cfg = config or BackendConfig(model_id="kuleshov-group/mdlm-owt")
        model_id = cfg.model_id or "kuleshov-group/mdlm-owt"

        # Load tokenizer (GPT-2 BPE)
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # GPT-2 has no pad token; use EOS
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model config
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            cfg_dict = json.load(f)

        mdlm_config = MDLMConfig(
            vocab_size=cfg_dict["vocab_size"],
            model_length=cfg_dict["model_length"],
            hidden_dim=cfg_dict["hidden_dim"],
            cond_dim=cfg_dict["cond_dim"],
            n_blocks=cfg_dict["n_blocks"],
            n_heads=cfg_dict["n_heads"],
            dropout=cfg_dict.get("dropout", 0.1),
            time_conditioning=cfg_dict.get("time_conditioning", False),
        )

        # Build model and load weights
        model = MDLM(mdlm_config)
        weights_path = hf_hub_download(model_id, "model.safetensors")
        state_dict = safetensors.torch.load_file(weights_path)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self._model = model
        self._config = mdlm_config
        self._device = torch.device(cfg.device)
        self._dtype = torch.float32  # CPU
        self._model.to(self._device)

        # MDLM mask token = vocab_size - 1 = 50257
        self._mask_token_id = mdlm_config.vocab_size - 1
        self._vocab_size = mdlm_config.vocab_size  # 50258 (includes mask)
        self._real_vocab_size = self._tokenizer.vocab_size  # 50257 (GPT-2)
        self._max_seq_len = min(cfg.max_seq_len, mdlm_config.model_length)

    @property
    def vocab_size(self) -> int:
        return self._real_vocab_size  # 50257 (real tokens, excluding mask)

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id  # 50257

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
        # Filter out mask tokens before decoding
        real_ids = [t for t in token_ids if t < self._real_vocab_size]
        return self._tokenizer.decode(real_ids, skip_special_tokens=True)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        real_ids = [t for t in ids if t < self._real_vocab_size]
        return self._tokenizer.decode(real_ids, skip_special_tokens=False)

    @torch.no_grad()
    def forward_logits(
        self,
        tokens: torch.Tensor,
        step: int,
        positions: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        total_steps: int = 64,
    ) -> torch.Tensor:
        """Compute MDLM denoising logits for the given positions.

        Runs a real forward pass through MDLM-OWT (170M params).
        The sigma (noise level) is derived from the step and total_steps.
        """
        input_ids = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        input_ids = input_ids.to(self._device)
        positions = positions.to(self._device)

        # Compute sigma (noise level) for this step
        # MDLM uses sigma = 1 for fully masked, sigma -> 0 for fully unmasked
        sigma = 1.0 - step / max(total_steps, 1)
        timesteps = torch.tensor([sigma], device=self._device)

        output = self._model(
            input_ids=input_ids,
            timesteps=timesteps,
            return_dict=True,
        )
        # Logits: [1, L, V_full] where V_full includes mask token
        all_logits = output.logits[0]  # [L, V_full]

        # Return logits for requested positions, truncated to real vocab
        return all_logits[positions, :self._real_vocab_size]  # [K, V]

    @torch.no_grad()
    def forward_logits_full(
        self,
        tokens: torch.Tensor,
        step: int,
        conditioning: Optional[torch.Tensor] = None,
        total_steps: int = 64,
    ) -> torch.Tensor:
        """Compute logits for ALL positions."""
        input_ids = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        input_ids = input_ids.to(self._device)

        sigma = 1.0 - step / max(total_steps, 1)
        timesteps = torch.tensor([sigma], device=self._device)

        output = self._model(
            input_ids=input_ids,
            timesteps=timesteps,
            return_dict=True,
        )
        return output.logits[0, :, :self._real_vocab_size]  # [L, V]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import transformers
            import safetensors
            import einops
            return True
        except ImportError:
            return False
