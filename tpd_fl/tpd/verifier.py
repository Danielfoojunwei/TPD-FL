"""
Verifier Gate Okπ — deterministic post-step safety check.

After each diffusion step (or at the end of decoding), the verifier
inspects the current token sequence and decides whether it satisfies
the privacy policy π.

Checks (all deterministic):
1. Regex scan for forbidden patterns (emails, phones, SSNs, etc.)
   anywhere in the decoded text.
2. Structural check: sensitive spans must contain only placeholder
   tokens if the policy requires full redaction.
3. Optional semantic proxy: check output against a known secret list
   (for evaluation / red-teaming only).

If the verifier rejects, the decode loop triggers a repair action
(resample or edit-repair).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES


# Reuse the same pattern bank as the typer (consistency)
_FORBIDDEN_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("PHONE", re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CC", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("ID", re.compile(r"\b[A-Z]{1,3}\d{6,10}\b")),
]


@dataclass
class VerifierConfig:
    """Configuration for the verifier gate."""
    # Which pattern tags are forbidden in the output
    forbidden_tags: List[str] = field(
        default_factory=lambda: ["EMAIL", "PHONE", "SSN", "CC", "ID"]
    )
    # Known secrets for semantic-proxy checking (test/eval only)
    known_secrets: List[str] = field(default_factory=list)
    # Placeholder strings that are acceptable in sensitive spans
    valid_placeholders: List[str] = field(
        default_factory=lambda: [
            "[REDACTED]", "[NAME]", "[EMAIL]", "[PHONE]", "[ID]",
            "[ADDRESS]", "[CC]", "[MASKED]", "[PRIVATE]",
            "<REDACTED>", "<NAME>", "<EMAIL>", "<PHONE>", "<ID>",
            "<ADDRESS>", "<CC>", "<MASKED>", "<PRIVATE>",
            "REDACTED", "***",
        ]
    )
    # Whether to run the structural placeholder check
    check_placeholders: bool = True
    # Whether to run the semantic proxy check
    check_semantic: bool = False
    # Extra forbidden regex patterns
    extra_forbidden: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class VerifierResult:
    """Result of a verifier check."""
    ok: bool
    violations: List[Dict]  # list of {type, detail, positions?}


class Verifier:
    """Deterministic verifier gate Okπ."""

    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or VerifierConfig()
        self._forbidden_tags = set(self.config.forbidden_tags)
        self._patterns = [
            (tag, pat) for tag, pat in _FORBIDDEN_PATTERNS
            if tag in self._forbidden_tags
        ]
        # Add extra patterns
        for tag, pat_str in self.config.extra_forbidden:
            self._patterns.append((tag, re.compile(pat_str)))
        self._known_set: FrozenSet[str] = frozenset(
            s.lower() for s in self.config.known_secrets
        )

    def check(
        self,
        text: str,
        token_ids: Optional[List[int]] = None,
        pos_type: Optional[List[SpanType]] = None,
        tokenizer=None,
    ) -> VerifierResult:
        """Run all configured checks.

        Parameters
        ----------
        text : str
            The decoded output text.
        token_ids : optional list of token IDs (for structural check).
        pos_type : optional per-token type list (for structural check).
        tokenizer : optional tokenizer (for structural check).

        Returns
        -------
        VerifierResult with ``ok=True`` if all checks pass.
        """
        violations: List[Dict] = []

        # 1. Regex scan for forbidden patterns
        violations.extend(self._check_regex(text))

        # 2. Structural placeholder check
        if (
            self.config.check_placeholders
            and token_ids is not None
            and pos_type is not None
            and tokenizer is not None
        ):
            violations.extend(
                self._check_structural(token_ids, pos_type, tokenizer)
            )

        # 3. Semantic proxy
        if self.config.check_semantic and self._known_set:
            violations.extend(self._check_semantic(text))

        return VerifierResult(ok=len(violations) == 0, violations=violations)

    # ------------------------------------------------------------------
    # Check implementations
    # ------------------------------------------------------------------

    def _check_regex(self, text: str) -> List[Dict]:
        violations = []
        for tag, pat in self._patterns:
            for m in pat.finditer(text):
                violations.append({
                    "type": "regex",
                    "tag": tag,
                    "match": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                })
        return violations

    def _check_structural(
        self,
        token_ids: List[int],
        pos_type: List[SpanType],
        tokenizer,
    ) -> List[Dict]:
        """Check that sensitive positions contain valid placeholders."""
        violations = []
        # Collect contiguous sensitive spans
        i = 0
        while i < len(pos_type):
            if pos_type[i] in SENSITIVE_TYPES:
                start = i
                while i < len(pos_type) and pos_type[i] in SENSITIVE_TYPES:
                    i += 1
                span_text = tokenizer.decode(token_ids[start:i]).strip()
                if not self._is_valid_placeholder(span_text):
                    violations.append({
                        "type": "structural",
                        "detail": f"sensitive span [{start}:{i}] contains non-placeholder text",
                        "span_text": span_text,
                        "positions": list(range(start, i)),
                    })
            else:
                i += 1
        return violations

    def _check_semantic(self, text: str) -> List[Dict]:
        """Check if any known secrets appear in output."""
        violations = []
        text_lower = text.lower()
        for secret in self._known_set:
            if secret in text_lower:
                violations.append({
                    "type": "semantic",
                    "detail": f"known secret found in output",
                    "secret_prefix": secret[:10] + "..." if len(secret) > 10 else secret,
                })
        return violations

    def _is_valid_placeholder(self, text: str) -> bool:
        """Check if text consists only of valid placeholder strings or whitespace."""
        t = text.strip()
        if not t:
            return True
        for ph in self.config.valid_placeholders:
            t = t.replace(ph, "")
        # After removing all placeholders, only whitespace/punctuation should remain
        return all(c in " \t\n.,;:!?" for c in t.strip())
