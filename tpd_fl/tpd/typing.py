"""
Span Typer τ — deterministic pipeline for classifying output positions.

Assigns each token position a SpanType ∈ {PUB, SENS, REG, DERIVED_*}
and a span identifier so that the projection engine can apply the
correct allowed-set mask per position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple


class SpanType(Enum):
    """Privacy classification of a token span."""
    PUB = auto()        # Public — unrestricted vocabulary
    SENS = auto()       # Sensitive PII (e.g., name, email, phone)
    REG = auto()        # Regulated data (e.g., SSN, credit card)
    DERIVED_NAME = auto()
    DERIVED_EMAIL = auto()
    DERIVED_PHONE = auto()
    DERIVED_ID = auto()
    DERIVED_CC = auto()
    DERIVED_ADDRESS = auto()


# Mapping from detected entity tag to SpanType
_ENTITY_TO_TYPE: Dict[str, SpanType] = {
    "EMAIL": SpanType.DERIVED_EMAIL,
    "PHONE": SpanType.DERIVED_PHONE,
    "SSN": SpanType.REG,
    "CC": SpanType.DERIVED_CC,
    "ID": SpanType.DERIVED_ID,
    "NAME": SpanType.DERIVED_NAME,
    "ADDRESS": SpanType.DERIVED_ADDRESS,
}

# Sensitive types that receive restricted vocabulary
SENSITIVE_TYPES = frozenset({
    SpanType.SENS,
    SpanType.REG,
    SpanType.DERIVED_NAME,
    SpanType.DERIVED_EMAIL,
    SpanType.DERIVED_PHONE,
    SpanType.DERIVED_ID,
    SpanType.DERIVED_CC,
    SpanType.DERIVED_ADDRESS,
})


@dataclass(frozen=True)
class Span:
    """A typed span in the token sequence."""
    start: int          # inclusive token index
    end: int            # exclusive token index
    type: SpanType
    entity_tag: str = ""  # e.g. "EMAIL", "PHONE"
    meta: Dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start


# ---------- regex detectors ----------

# Patterns operate on the *text* (pre-tokenization).  After detection we
# map character spans → token spans via an offset table.

_PATTERNS: List[Tuple[str, str]] = [
    ("EMAIL", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ("PHONE", r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("CC", r"\b(?:\d[ -]*?){13,19}\b"),
    ("ID", r"\b[A-Z]{1,3}\d{6,10}\b"),
    # Simple address heuristic: number + street word
    ("ADDRESS", r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:St|Ave|Blvd|Rd|Dr|Ln|Ct|Way|Pl)\b"),
]

_COMPILED = [(tag, re.compile(pat)) for tag, pat in _PATTERNS]


def _char_to_token_spans(
    char_spans: List[Tuple[int, int, str]],
    offset_mapping: List[Tuple[int, int]],
) -> List[Tuple[int, int, str]]:
    """Convert character-level spans to token-level spans via offset_mapping.

    offset_mapping: list of (char_start, char_end) per token.
    Returns list of (tok_start_inclusive, tok_end_exclusive, entity_tag).
    """
    token_spans = []
    for cs, ce, tag in char_spans:
        tok_start: Optional[int] = None
        tok_end: Optional[int] = None
        for idx, (ts, te) in enumerate(offset_mapping):
            if te <= cs:
                continue
            if ts >= ce:
                break
            if tok_start is None:
                tok_start = idx
            tok_end = idx + 1
        if tok_start is not None and tok_end is not None:
            token_spans.append((tok_start, tok_end, tag))
    return token_spans


class SpanTyper:
    """Deterministic span typing pipeline.

    Steps:
    1. Regex detectors for EMAIL / PHONE / SSN / CC / ID / ADDRESS.
    2. (Optional) lightweight NER — disabled by default.
    3. Policy overrides (denylist / allowlist positions).

    The typer operates on *text* and an offset_mapping that maps tokens
    back to characters (as produced by HuggingFace tokenizers with
    ``return_offsets_mapping=True``).
    """

    def __init__(
        self,
        use_ner: bool = False,
        denylist_positions: Optional[List[int]] = None,
        allowlist_positions: Optional[List[int]] = None,
        extra_patterns: Optional[List[Tuple[str, str]]] = None,
    ):
        self.use_ner = use_ner
        self.denylist_positions = set(denylist_positions or [])
        self.allowlist_positions = set(allowlist_positions or [])
        self._patterns = list(_COMPILED)
        if extra_patterns:
            for tag, pat in extra_patterns:
                self._patterns.append((tag, re.compile(pat)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def type_text(
        self,
        text: str,
        offset_mapping: List[Tuple[int, int]],
        seq_len: Optional[int] = None,
    ) -> Tuple[List[Span], List[SpanType], List[int]]:
        """Run the full typing pipeline.

        Returns
        -------
        spans : List[Span]
            Detected typed spans (sorted by start position).
        pos_type : List[SpanType]
            Per-token type assignment (length ``seq_len``).
        pos_span_id : List[int]
            Per-token span identifier (−1 for PUB positions).
        """
        L = seq_len or len(offset_mapping)

        # Step 1 — regex detection (character level)
        char_spans = self._detect_regex(text)

        # Step 2 — optional NER (placeholder)
        if self.use_ner:
            char_spans.extend(self._detect_ner(text))

        # Deduplicate overlapping character spans (longest wins)
        char_spans = self._deduplicate(char_spans)

        # Map to token level
        tok_spans = _char_to_token_spans(char_spans, offset_mapping)

        # Build Span objects
        spans: List[Span] = []
        for sid, (ts, te, tag) in enumerate(tok_spans):
            stype = _ENTITY_TO_TYPE.get(tag, SpanType.SENS)
            spans.append(Span(start=ts, end=te, type=stype, entity_tag=tag))

        # Initialise per-position arrays
        pos_type: List[SpanType] = [SpanType.PUB] * L
        pos_span_id: List[int] = [-1] * L

        for sid, span in enumerate(spans):
            for i in range(span.start, min(span.end, L)):
                pos_type[i] = span.type
                pos_span_id[i] = sid

        # Step 3 — policy overrides
        for i in self.denylist_positions:
            if 0 <= i < L:
                if pos_type[i] == SpanType.PUB:
                    pos_type[i] = SpanType.SENS
        for i in self.allowlist_positions:
            if 0 <= i < L:
                pos_type[i] = SpanType.PUB
                pos_span_id[i] = -1

        return spans, pos_type, pos_span_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_regex(self, text: str) -> List[Tuple[int, int, str]]:
        results: List[Tuple[int, int, str]] = []
        for tag, pattern in self._patterns:
            for m in pattern.finditer(text):
                results.append((m.start(), m.end(), tag))
        return results

    def _detect_ner(self, text: str) -> List[Tuple[int, int, str]]:
        """Placeholder for optional NER integration."""
        return []

    @staticmethod
    def _deduplicate(
        spans: List[Tuple[int, int, str]],
    ) -> List[Tuple[int, int, str]]:
        """Remove overlapping spans — longest span wins on conflict."""
        if not spans:
            return spans
        # Sort by start, then by descending length
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        deduped: List[Tuple[int, int, str]] = []
        last_end = -1
        for s, e, tag in spans:
            if s >= last_end:
                deduped.append((s, e, tag))
                last_end = e
        return deduped
