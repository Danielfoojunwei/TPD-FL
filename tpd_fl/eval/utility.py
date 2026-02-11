"""
Utility Metrics — measure output quality on public (non-sensitive) content.

All metrics operate on the **public portion** of the output: positions that
are not marked as sensitive.  This ensures that utility is evaluated only on
content that the model is expected to reproduce faithfully (the non-PII
parts), while sensitive spans are expected to contain placeholders.

Metrics provided:

  - **exact_match_public**: token-level exact match accuracy on public positions.
  - **rouge_public**: ROUGE-1, ROUGE-2, ROUGE-L on public-position text.
    Implemented from scratch (no external dependency).
  - **fluency_metrics**: surface-level fluency indicators (length, repetition,
    placeholder count and correctness).
  - **UtilityEvaluator**: class combining all metrics with a single ``evaluate`` call.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Valid placeholder strings (consistent with tpd_fl.tpd.verifier)
# ---------------------------------------------------------------------------

VALID_PLACEHOLDERS: List[str] = [
    "[REDACTED]", "[NAME]", "[EMAIL]", "[PHONE]", "[ID]",
    "[ADDRESS]", "[CC]", "[MASKED]", "[PRIVATE]",
    "<REDACTED>", "<NAME>", "<EMAIL>", "<PHONE>", "<ID>",
    "<ADDRESS>", "<CC>", "<MASKED>", "<PRIVATE>",
    "REDACTED", "***",
]


# ---------------------------------------------------------------------------
# Exact match on public positions
# ---------------------------------------------------------------------------

def exact_match_public(
    output: str,
    reference: str,
    sensitive_positions: Optional[Set[int]] = None,
) -> float:
    """Compute token-level exact match accuracy on public positions.

    Tokens are obtained by whitespace splitting.  Positions listed in
    ``sensitive_positions`` are excluded from the comparison.

    Parameters
    ----------
    output : str
        Model output text.
    reference : str
        Ground-truth reference text.
    sensitive_positions : set of int, optional
        Token indices (0-based) that correspond to sensitive content.
        These positions are skipped in the comparison.

    Returns
    -------
    float
        Fraction of matching public tokens in [0, 1].
        Returns 1.0 when there are no public positions to compare.
    """
    sens = sensitive_positions or set()
    out_tokens = output.split()
    ref_tokens = reference.split()

    # Align by position (up to the shorter sequence)
    max_len = max(len(out_tokens), len(ref_tokens))
    if max_len == 0:
        return 1.0

    matches = 0
    total = 0
    for i in range(max_len):
        if i in sens:
            continue
        total += 1
        out_tok = out_tokens[i] if i < len(out_tokens) else ""
        ref_tok = ref_tokens[i] if i < len(ref_tokens) else ""
        if out_tok == ref_tok:
            matches += 1

    return matches / total if total > 0 else 1.0


# ---------------------------------------------------------------------------
# ROUGE implementation (no external dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenization for ROUGE computation."""
    return text.lower().split()


def _ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-gram counts from a token list."""
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def _rouge_n_score(
    hypothesis_tokens: List[str],
    reference_tokens: List[str],
    n: int,
) -> Dict[str, float]:
    """Compute ROUGE-N precision, recall, and F1.

    Parameters
    ----------
    hypothesis_tokens : tokenised model output.
    reference_tokens : tokenised reference.
    n : n-gram order (1 for unigrams, 2 for bigrams).

    Returns
    -------
    dict with ``precision``, ``recall``, ``f1``.
    """
    hyp_ngrams = _ngrams(hypothesis_tokens, n)
    ref_ngrams = _ngrams(reference_tokens, n)

    overlap = sum((hyp_ngrams & ref_ngrams).values())
    hyp_total = sum(hyp_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    precision = overlap / hyp_total if hyp_total > 0 else 0.0
    recall = overlap / ref_total if ref_total > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute the length of the longest common subsequence.

    Uses O(min(|x|, |y|)) space DP.
    """
    if len(x) < len(y):
        x, y = y, x
    m, n = len(x), len(y)
    if n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _rouge_l_score(
    hypothesis_tokens: List[str],
    reference_tokens: List[str],
) -> Dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 via LCS."""
    lcs = _lcs_length(hypothesis_tokens, reference_tokens)
    hyp_len = len(hypothesis_tokens)
    ref_len = len(reference_tokens)

    precision = lcs / hyp_len if hyp_len > 0 else 0.0
    recall = lcs / ref_len if ref_len > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _extract_public_text(
    text: str,
    sensitive_positions: Optional[Set[int]] = None,
) -> str:
    """Extract only the tokens at public positions."""
    if not sensitive_positions:
        return text
    tokens = text.split()
    public_tokens = [
        tok for i, tok in enumerate(tokens) if i not in sensitive_positions
    ]
    return " ".join(public_tokens)


def rouge_public(
    output: str,
    reference: str,
    sensitive_positions: Optional[Set[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L on public-position text.

    Parameters
    ----------
    output : str
        Model output text.
    reference : str
        Ground-truth reference text.
    sensitive_positions : set of int, optional
        Token indices to exclude from the ROUGE computation.

    Returns
    -------
    dict with keys ``rouge1``, ``rouge2``, ``rougeL``, each containing
    ``precision``, ``recall``, ``f1``.
    """
    out_public = _extract_public_text(output, sensitive_positions)
    ref_public = _extract_public_text(reference, sensitive_positions)

    hyp_tokens = _tokenize(out_public)
    ref_tokens = _tokenize(ref_public)

    return {
        "rouge1": _rouge_n_score(hyp_tokens, ref_tokens, 1),
        "rouge2": _rouge_n_score(hyp_tokens, ref_tokens, 2),
        "rougeL": _rouge_l_score(hyp_tokens, ref_tokens),
    }


# ---------------------------------------------------------------------------
# Fluency metrics
# ---------------------------------------------------------------------------

def fluency_metrics(
    text: str,
    valid_placeholders: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute surface-level fluency indicators.

    Parameters
    ----------
    text : str
        Model output text.
    valid_placeholders : list of str, optional
        Strings that count as valid placeholders.  Defaults to
        :data:`VALID_PLACEHOLDERS`.

    Returns
    -------
    dict with keys:
        length : int
            Total number of whitespace-delimited tokens.
        char_length : int
            Total number of characters.
        repetition_ratio : float
            Fraction of consecutive duplicate token pairs.
            High values indicate degenerate repetition.
        placeholder_count : int
            Number of valid placeholder tokens found in the text.
        placeholder_correctness : float
            Fraction of detected placeholder-like tokens (tokens containing
            brackets or asterisks) that are in the valid set.  Measures
            whether the model produces well-formed placeholders.
    """
    phs = valid_placeholders if valid_placeholders is not None else VALID_PLACEHOLDERS
    tokens = text.split()
    n = len(tokens)

    # Repetition ratio: consecutive duplicate pairs
    if n <= 1:
        rep_ratio = 0.0
    else:
        duplicates = sum(
            1 for i in range(1, n) if tokens[i] == tokens[i - 1]
        )
        rep_ratio = duplicates / (n - 1)

    # Placeholder detection
    # A token is "placeholder-like" if it contains [ ] or < > or ***
    placeholder_like_count = 0
    valid_placeholder_count = 0

    # We check for valid placeholders as substrings in the text
    ph_count = 0
    for ph in phs:
        ph_count += text.count(ph)

    # Check placeholder-like tokens
    for tok in tokens:
        is_placeholder_like = (
            (tok.startswith("[") and tok.endswith("]"))
            or (tok.startswith("<") and tok.endswith(">"))
            or tok == "***"
            or tok == "REDACTED"
        )
        if is_placeholder_like:
            placeholder_like_count += 1
            if tok in phs:
                valid_placeholder_count += 1

    ph_correctness = (
        valid_placeholder_count / placeholder_like_count
        if placeholder_like_count > 0
        else 1.0
    )

    return {
        "length": n,
        "char_length": len(text),
        "repetition_ratio": rep_ratio,
        "placeholder_count": ph_count,
        "placeholder_correctness": ph_correctness,
    }


# ---------------------------------------------------------------------------
# UtilityEvaluator
# ---------------------------------------------------------------------------

class UtilityEvaluator:
    """Combined utility evaluator for TPD+FL outputs.

    Wraps :func:`exact_match_public`, :func:`rouge_public`, and
    :func:`fluency_metrics` into a single evaluation call.

    Parameters
    ----------
    valid_placeholders : list of str, optional
        Override the default valid placeholder strings.
    """

    def __init__(
        self,
        valid_placeholders: Optional[List[str]] = None,
    ):
        self._valid_placeholders = valid_placeholders or VALID_PLACEHOLDERS

    def evaluate(
        self,
        output_text: str,
        reference_text: str,
        sensitive_positions: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate utility of a single output against a reference.

        Parameters
        ----------
        output_text : str
            Model output.
        reference_text : str
            Ground-truth reference.
        sensitive_positions : set of int, optional
            Token indices corresponding to sensitive content.

        Returns
        -------
        dict with keys:
            exact_match_public : float
            rouge : dict (rouge1, rouge2, rougeL)
            fluency : dict (length, repetition_ratio, placeholder_count,
                            placeholder_correctness)
        """
        em = exact_match_public(output_text, reference_text, sensitive_positions)
        rouge = rouge_public(output_text, reference_text, sensitive_positions)
        fluency = fluency_metrics(output_text, self._valid_placeholders)

        return {
            "exact_match_public": em,
            "rouge": rouge,
            "fluency": fluency,
        }

    def evaluate_batch(
        self,
        outputs: List[str],
        references: List[str],
        sensitive_positions_list: Optional[List[Set[int]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a batch of outputs and return aggregated metrics.

        Parameters
        ----------
        outputs : list of model output texts.
        references : list of ground-truth reference texts.
        sensitive_positions_list : parallel list of sensitive position sets.

        Returns
        -------
        dict with per-sample results and aggregate statistics.
        """
        per_sample: List[Dict[str, Any]] = []
        for i in range(len(outputs)):
            sens = (
                sensitive_positions_list[i]
                if sensitive_positions_list and i < len(sensitive_positions_list)
                else None
            )
            ref = references[i] if i < len(references) else ""
            per_sample.append(self.evaluate(outputs[i], ref, sens))

        # Aggregate
        em_scores = [r["exact_match_public"] for r in per_sample]
        rouge1_f1 = [r["rouge"]["rouge1"]["f1"] for r in per_sample]
        rouge2_f1 = [r["rouge"]["rouge2"]["f1"] for r in per_sample]
        rougeL_f1 = [r["rouge"]["rougeL"]["f1"] for r in per_sample]
        rep_ratios = [r["fluency"]["repetition_ratio"] for r in per_sample]

        n = len(per_sample) if per_sample else 1

        return {
            "num_samples": len(outputs),
            "mean_exact_match_public": sum(em_scores) / n,
            "mean_rouge1_f1": sum(rouge1_f1) / n,
            "mean_rouge2_f1": sum(rouge2_f1) / n,
            "mean_rougeL_f1": sum(rougeL_f1) / n,
            "mean_repetition_ratio": sum(rep_ratios) / n,
            "per_sample": per_sample,
        }
