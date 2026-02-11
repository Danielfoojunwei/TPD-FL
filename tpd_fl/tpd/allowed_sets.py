"""
Allowed Token Sets A(type) — per-type boolean vocabulary masks.

For each SpanType we build a boolean tensor of shape [V] where
V = vocabulary size.  ``True`` means the token is permitted for
positions of that type.

Mask semantics
--------------
- PUB:        all True  (unrestricted)
- SENS:       placeholders + limited safe tokens + punctuation
- REG:        strict subset — placeholders only + minimal punctuation
- DERIVED_*:  configurable "template vocabulary" subsets

The masks are materialised on the target device (GPU) once and reused
across all projection calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

import torch

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES


# Default placeholder tokens (string forms) that are always allowed for
# sensitive positions.  These are merged with tokenizer-specific IDs.
DEFAULT_PLACEHOLDERS = [
    "[REDACTED]", "[NAME]", "[EMAIL]", "[PHONE]", "[ID]", "[ADDRESS]",
    "[CC]", "[MASKED]", "[PRIVATE]", "<REDACTED>", "<NAME>", "<EMAIL>",
    "<PHONE>", "<ID>", "<ADDRESS>", "<CC>", "<MASKED>", "<PRIVATE>",
    "REDACTED", "NAME_REDACTED", "EMAIL_REDACTED", "***",
]

# ASCII punctuation that is generally safe to emit even in sensitive spans
_SAFE_PUNCTUATION = set(" .,;:!?'-\"()/\\@#$%^&*_+=[]{}|<>~`\n\t")


@dataclass
class AllowedSetConfig:
    """Configuration for allowed-set construction."""
    placeholder_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_PLACEHOLDERS))
    # Extra token strings always allowed for SENS
    safe_tokens_sens: List[str] = field(default_factory=list)
    # Extra token strings always allowed for REG
    safe_tokens_reg: List[str] = field(default_factory=list)
    # Per-entity overrides: entity_tag -> list of allowed token strings
    entity_overrides: Dict[str, List[str]] = field(default_factory=dict)
    # Whether to include single-character ASCII punctuation in SENS/REG
    include_punctuation: bool = True
    # Maximum token length (in characters) for "safe" tokens in SENS
    max_safe_token_len: int = 6


class AllowedSetBuilder:
    """Builds and caches boolean allowed-set masks on the target device.

    Usage::

        builder = AllowedSetBuilder(tokenizer, config, device="cuda")
        masks = builder.build()          # Dict[SpanType, Tensor[V]]
        mask = masks[SpanType.SENS]      # boolean [V]
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[AllowedSetConfig] = None,
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.config = config or AllowedSetConfig()
        self.device = torch.device(device)
        self._vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
        self._masks: Optional[Dict[SpanType, torch.Tensor]] = None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def build(self) -> Dict[SpanType, torch.Tensor]:
        """Build all masks. Cached after first call."""
        if self._masks is not None:
            return self._masks

        V = self._vocab_size
        cfg = self.config

        # -- token id sets --
        placeholder_ids = self._encode_strings(cfg.placeholder_tokens)
        safe_sens_ids = self._encode_strings(cfg.safe_tokens_sens)
        safe_reg_ids = self._encode_strings(cfg.safe_tokens_reg)

        punct_ids: Set[int] = set()
        if cfg.include_punctuation:
            punct_ids = self._single_char_token_ids(_SAFE_PUNCTUATION, max_len=cfg.max_safe_token_len)

        # PUB — everything allowed
        pub_mask = torch.ones(V, dtype=torch.bool, device=self.device)

        # SENS — placeholders + safe + punctuation
        sens_allowed = placeholder_ids | safe_sens_ids | punct_ids
        sens_mask = torch.zeros(V, dtype=torch.bool, device=self.device)
        if sens_allowed:
            sens_mask[torch.tensor(sorted(sens_allowed), dtype=torch.long, device=self.device)] = True

        # REG — stricter: placeholders + reg-safe + minimal punctuation
        reg_allowed = placeholder_ids | safe_reg_ids | punct_ids
        reg_mask = torch.zeros(V, dtype=torch.bool, device=self.device)
        if reg_allowed:
            reg_mask[torch.tensor(sorted(reg_allowed), dtype=torch.long, device=self.device)] = True

        masks: Dict[SpanType, torch.Tensor] = {
            SpanType.PUB: pub_mask,
            SpanType.SENS: sens_mask,
            SpanType.REG: reg_mask,
        }

        # DERIVED_* — start from SENS mask, apply entity overrides
        for stype in SENSITIVE_TYPES:
            if stype in masks:
                continue
            entity_tag = stype.name.replace("DERIVED_", "")
            override_strs = cfg.entity_overrides.get(entity_tag, [])
            if override_strs:
                extra_ids = self._encode_strings(override_strs)
                m = sens_mask.clone()
                if extra_ids:
                    m[torch.tensor(sorted(extra_ids), dtype=torch.long, device=self.device)] = True
                masks[stype] = m
            else:
                masks[stype] = sens_mask.clone()

        self._masks = masks
        return masks

    def get_mask(self, span_type: SpanType) -> torch.Tensor:
        """Return the allowed mask for a given SpanType."""
        masks = self.build()
        return masks.get(span_type, masks[SpanType.SENS])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_strings(self, strings: List[str]) -> Set[int]:
        """Encode a list of strings to token IDs (union of all sub-tokens)."""
        ids: Set[int] = set()
        for s in strings:
            encoded = self.tokenizer.encode(s, add_special_tokens=False)
            if isinstance(encoded, list):
                ids.update(encoded)
            else:
                ids.add(int(encoded))
        return ids

    def _single_char_token_ids(self, chars: Set[str], max_len: int = 6) -> Set[int]:
        """Find token IDs whose decoded form is a single safe character
        or a short string composed only of safe characters."""
        ids: Set[int] = set()
        # Iterate vocab — this is O(V) but runs once
        for tid in range(self._vocab_size):
            try:
                decoded = self.tokenizer.decode([tid])
            except Exception:
                continue
            stripped = decoded.strip()
            if 0 < len(stripped) <= max_len and all(c in chars for c in stripped):
                ids.add(tid)
        return ids
