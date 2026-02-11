"""
Federated Learning Client — local training loop for TPD+FL.

Each FL client holds a local copy of the diffusion LLM with LoRA
adapters attached.  The client trains on its private data partition
using a diffusion denoising objective:

  1. **Random masking**: for each sample, randomly replace a fraction
     of token positions with the mask token.
  2. **Denoising loss**: the model predicts the original tokens at
     masked positions; cross-entropy loss is computed.
  3. **Typed training** (optional): for positions typed as SENS or REG,
     the target is replaced with a designated placeholder token so the
     model learns to *never* predict the original sensitive value.

After ``local_epochs`` of training the client computes *adapter deltas*
(the difference between the trained and starting LoRA weights) and
returns them to the server for aggregation.

Usage::

    from tpd_fl.fl.client import FLClient, FLClientConfig

    config = FLClientConfig(local_epochs=3, lr=1e-4)
    client = FLClient(model, tokenizer, lora_modules, config)
    deltas = client.train(dataset)

This module depends on torch and the Python standard library only.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tpd_fl.fl.lora import (
    LoRALinear,
    get_lora_state_dict,
    load_lora_state_dict,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FLClientConfig:
    """Configuration for federated learning client training.

    Attributes
    ----------
    local_epochs : int
        Number of local training epochs per FL round.
    lr : float
        Learning rate for the local AdamW optimiser.
    batch_size : int
        Mini-batch size for local training.
    typed_training : bool
        If True, sensitive/regulated positions are trained to predict
        placeholder tokens rather than the original values, reinforcing
        the TPD privacy objective.
    mask_fraction : float
        Fraction of tokens to mask per training sample for the
        diffusion denoising objective.
    placeholder_token_id : int
        Token ID used as the target for sensitive positions when
        ``typed_training`` is enabled.  Should be the tokenizer's
        ``[REDACTED]`` or ``[MASK]`` token.
    mask_token_id : int
        Token ID used to replace masked positions in the input.
    max_grad_norm : float
        Maximum gradient norm for gradient clipping.
    weight_decay : float
        AdamW weight decay.
    seed : Optional[int]
        Random seed for reproducibility.  If None, training is
        non-deterministic.
    """

    local_epochs: int = 3
    lr: float = 1e-4
    batch_size: int = 4
    typed_training: bool = True
    mask_fraction: float = 0.15
    placeholder_token_id: int = 0
    mask_token_id: int = 0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# FL Client
# ---------------------------------------------------------------------------

class FLClient:
    """Federated learning client for TPD+FL diffusion LLM training.

    The client performs local training on private data using a
    diffusion denoising objective with optional typed privacy training.

    Parameters
    ----------
    model : nn.Module
        The diffusion LLM with LoRA adapters already attached.
        Only LoRA parameters (requires_grad=True) are trained.
    tokenizer
        Tokenizer compatible with the model.  Must support ``encode``
        and ``decode`` methods, plus ``vocab_size`` or ``__len__``.
    lora_modules : Dict[str, LoRALinear]
        Mapping of LoRA module names to layers, as returned by
        :func:`tpd_fl.fl.lora.attach_lora`.
    config : FLClientConfig
        Training configuration.
    position_types : Optional[List[List[str]]]
        Per-sample, per-position type labels (e.g., "PUB", "SENS",
        "REG").  Used when ``typed_training`` is True.  If None,
        all positions are treated as PUB.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        lora_modules: Dict[str, LoRALinear],
        config: Optional[FLClientConfig] = None,
        position_types: Optional[List[List[str]]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lora_modules = lora_modules
        self.config = config or FLClientConfig()
        self.position_types = position_types

        # Determine device from model parameters
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        # Set seed if provided
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            random.seed(self.config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Run local training and return adapter weight deltas.

        Parameters
        ----------
        dataset : List[Dict[str, Any]]
            Local training data.  Each element is a dict with at least:
              - ``"input_ids"`` : List[int] — tokenised sequence
            Optional keys:
              - ``"position_types"`` : List[str] — per-position type labels
              - ``"attention_mask"`` : List[int] — 1 for real tokens, 0 for padding

        Returns
        -------
        Dict[str, Tensor]
            Adapter deltas: the element-wise difference between the
            trained LoRA state and the starting state.  The server
            aggregates these deltas across clients.
        """
        # Snapshot the starting state
        start_state = get_lora_state_dict(self.lora_modules)

        # Build optimiser over LoRA parameters only
        lora_params = []
        for lora_layer in self.lora_modules.values():
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.model.train()

        for epoch in range(self.config.local_epochs):
            # Shuffle dataset each epoch
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[
                    batch_start : batch_start + self.config.batch_size
                ]
                batch = [dataset[i] for i in batch_indices]

                loss = self._compute_batch_loss(batch)

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    lora_params, self.config.max_grad_norm
                )

                optimizer.step()

        # Compute deltas
        end_state = get_lora_state_dict(self.lora_modules)
        deltas: Dict[str, torch.Tensor] = {}
        for key in start_state:
            deltas[key] = end_state[key] - start_state[key]

        return deltas

    # ------------------------------------------------------------------
    # Internal: loss computation
    # ------------------------------------------------------------------

    def _compute_batch_loss(
        self,
        batch: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Compute the diffusion denoising loss for a mini-batch.

        For each sample:
        1. Randomly select ``mask_fraction`` of positions to mask.
        2. Replace those positions with ``mask_token_id`` in the input.
        3. Forward pass through the model.
        4. Compute cross-entropy loss at masked positions against targets.
        5. If ``typed_training``, override targets for SENS/REG positions
           with ``placeholder_token_id``.

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            Mini-batch of samples.

        Returns
        -------
        Tensor
            Scalar loss averaged over all masked positions in the batch.
        """
        # Collate batch into tensors
        input_ids_list = [
            torch.tensor(sample["input_ids"], dtype=torch.long)
            for sample in batch
        ]

        # Pad to max length in batch
        max_len = max(ids.size(0) for ids in input_ids_list)
        padded_ids = torch.full(
            (len(batch), max_len),
            self.config.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            len(batch), max_len, dtype=torch.long, device=self.device
        )

        for i, ids in enumerate(input_ids_list):
            length = ids.size(0)
            padded_ids[i, :length] = ids.to(self.device)
            if "attention_mask" in batch[i]:
                am = torch.tensor(
                    batch[i]["attention_mask"], dtype=torch.long
                )
                attention_mask[i, : am.size(0)] = am.to(self.device)
            else:
                attention_mask[i, :length] = 1

        # Build targets (original token ids, before masking)
        targets = padded_ids.clone()

        # Apply typed training: override targets at SENS/REG positions
        if self.config.typed_training:
            for i, sample in enumerate(batch):
                ptypes = sample.get("position_types", None)
                if ptypes is None:
                    continue
                for j, ptype in enumerate(ptypes):
                    if j >= max_len:
                        break
                    if ptype in ("SENS", "REG", "DERIVED_NAME",
                                 "DERIVED_EMAIL", "DERIVED_PHONE",
                                 "DERIVED_ID", "DERIVED_CC",
                                 "DERIVED_ADDRESS"):
                        targets[i, j] = self.config.placeholder_token_id

        # Random masking
        mask_prob = torch.full(
            (len(batch), max_len),
            self.config.mask_fraction,
            device=self.device,
        )
        # Do not mask padding
        mask_prob = mask_prob * attention_mask.float()

        mask = torch.bernoulli(mask_prob).bool()
        # Ensure at least one position is masked per sample
        for i in range(len(batch)):
            if not mask[i].any():
                valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    rand_idx = valid_positions[
                        torch.randint(len(valid_positions), (1,))
                    ]
                    mask[i, rand_idx] = True

        # Create masked input
        masked_input = padded_ids.clone()
        masked_input[mask] = self.config.mask_token_id

        # Forward pass
        outputs = self.model(input_ids=masked_input, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # logits shape: (batch_size, seq_len, vocab_size)
        # Compute loss only at masked positions
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)  # (B*L, V)
        targets_flat = targets.view(-1)             # (B*L,)
        mask_flat = mask.view(-1)                   # (B*L,)

        # Select only masked positions
        masked_logits = logits_flat[mask_flat]       # (M, V)
        masked_targets = targets_flat[mask_flat]     # (M,)

        if masked_logits.size(0) == 0:
            # No masked positions: return zero loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = F.cross_entropy(masked_logits, masked_targets)
        return loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_num_trainable_params(self) -> int:
        """Return the total number of trainable (LoRA) parameters."""
        total = 0
        for lora_layer in self.lora_modules.values():
            total += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
        return total

    def get_current_state(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the current LoRA state dict."""
        return get_lora_state_dict(self.lora_modules)

    def set_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a LoRA state dict into the client's adapters.

        Parameters
        ----------
        state_dict : Dict[str, Tensor]
            LoRA state dict, as produced by :func:`get_lora_state_dict`
            or received from the FL server.
        """
        load_lora_state_dict(self.lora_modules, state_dict)
