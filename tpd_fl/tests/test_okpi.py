"""
Tests for the Verifier gate Okπ.

Verifies:
1. Regex detection catches forbidden patterns (emails, phones, SSNs, etc.)
2. Clean text passes verification.
3. Structural checks detect non-placeholder content in sensitive spans.
4. Semantic proxy catches known secrets.
5. Multiple simultaneous violations are all reported.
"""

import pytest

from tpd_fl.tpd.typing import SpanType
from tpd_fl.tpd.verifier import Verifier, VerifierConfig


class TestRegexDetection:
    """Verify regex-based forbidden pattern detection."""

    @pytest.fixture
    def verifier(self):
        return Verifier(VerifierConfig(
            forbidden_tags=["EMAIL", "PHONE", "SSN", "CC", "ID"],
        ))

    def test_email_detected(self, verifier):
        result = verifier.check("Send to alice@example.com please.")
        assert not result.ok
        assert any(v["tag"] == "EMAIL" for v in result.violations)

    def test_phone_detected(self, verifier):
        result = verifier.check("Call me at (555) 123-4567.")
        assert not result.ok
        assert any(v["tag"] == "PHONE" for v in result.violations)

    def test_ssn_detected(self, verifier):
        result = verifier.check("SSN: 123-45-6789")
        assert not result.ok
        assert any(v["tag"] == "SSN" for v in result.violations)

    def test_clean_text_passes(self, verifier):
        result = verifier.check("This is a normal sentence with [REDACTED] data.")
        assert result.ok
        assert len(result.violations) == 0

    def test_placeholder_text_passes(self, verifier):
        result = verifier.check(
            "The patient [NAME] can be reached at [EMAIL] or [PHONE]."
        )
        assert result.ok

    def test_multiple_violations(self, verifier):
        text = "Email: test@foo.com, Phone: 555-123-4567, SSN: 987-65-4321"
        result = verifier.check(text)
        assert not result.ok
        tags = {v["tag"] for v in result.violations}
        assert "EMAIL" in tags
        assert "PHONE" in tags
        assert "SSN" in tags


class TestSelectiveForbiddenTags:
    """Verify that only configured tags trigger violations."""

    def test_email_only(self):
        v = Verifier(VerifierConfig(forbidden_tags=["EMAIL"]))
        # Email triggers
        assert not v.check("Contact user@test.com").ok
        # Phone does not trigger
        assert v.check("Call 555-123-4567").ok

    def test_ssn_only(self):
        v = Verifier(VerifierConfig(forbidden_tags=["SSN"]))
        assert not v.check("SSN: 111-22-3333").ok
        assert v.check("Email: a@b.com").ok

    def test_no_tags_passes_everything(self):
        v = Verifier(VerifierConfig(forbidden_tags=[]))
        assert v.check("a@b.com 555-123-4567 111-22-3333").ok


class TestSemanticProxy:
    """Test semantic-proxy known-secret detection."""

    def test_known_secret_detected(self):
        v = Verifier(VerifierConfig(
            forbidden_tags=[],
            check_semantic=True,
            known_secrets=["alice johnson", "project-phoenix"],
        ))
        result = v.check("The report mentions Alice Johnson and Project-Phoenix.")
        assert not result.ok
        assert len(result.violations) >= 2

    def test_partial_match(self):
        v = Verifier(VerifierConfig(
            forbidden_tags=[],
            check_semantic=True,
            known_secrets=["alice"],
        ))
        result = v.check("alice was here")
        assert not result.ok

    def test_case_insensitive(self):
        v = Verifier(VerifierConfig(
            forbidden_tags=[],
            check_semantic=True,
            known_secrets=["SECRET"],
        ))
        result = v.check("the secret is out")
        assert not result.ok

    def test_no_match_passes(self):
        v = Verifier(VerifierConfig(
            forbidden_tags=[],
            check_semantic=True,
            known_secrets=["xyzzy"],
        ))
        result = v.check("nothing special here")
        assert result.ok


class TestExtraForbiddenPatterns:
    """Test custom forbidden regex patterns."""

    def test_custom_pattern(self):
        v = Verifier(VerifierConfig(
            forbidden_tags=[],
            extra_forbidden=[("CUSTOM", r"PROJ-\d{4}")],
        ))
        result = v.check("The code is PROJ-1234.")
        assert not result.ok
        assert result.violations[0]["tag"] == "CUSTOM"

    def test_custom_plus_standard(self):
        v = Verifier(VerifierConfig(
            forbidden_tags=["EMAIL"],
            extra_forbidden=[("BADGE", r"BADGE-[A-Z]{2}\d{3}")],
        ))
        result = v.check("Contact a@b.com with BADGE-AB123.")
        assert not result.ok
        tags = {v_item["tag"] for v_item in result.violations}
        assert "EMAIL" in tags
        assert "BADGE" in tags


class TestStructuralCheck:
    """Test structural placeholder verification."""

    def _make_token_ids_and_types(self, text_tokens, sens_range):
        """Helper to build token_ids and pos_type for structural check."""
        L = len(text_tokens)
        pos_type = [SpanType.PUB] * L
        for i in sens_range:
            pos_type[i] = SpanType.SENS
        return text_tokens, pos_type

    def test_valid_placeholder_passes(self):
        """Sensitive span containing only valid placeholders should pass."""

        class MockTokenizer:
            def decode(self, ids):
                mapping = {1: "[REDACTED]", 2: " ", 3: "hello", 4: "world"}
                return "".join(mapping.get(i, f"[{i}]") for i in ids)

        v = Verifier(VerifierConfig(
            forbidden_tags=[],
            check_placeholders=True,
        ))
        # tokens: [3, 4, 1, 1, 3]  pos_type: PUB PUB SENS SENS PUB
        token_ids = [3, 4, 1, 1, 3]
        pos_type = [SpanType.PUB, SpanType.PUB, SpanType.SENS, SpanType.SENS, SpanType.PUB]

        result = v.check(
            "hello world [REDACTED][REDACTED] hello",
            token_ids=token_ids,
            pos_type=pos_type,
            tokenizer=MockTokenizer(),
        )
        assert result.ok

    def test_invalid_content_in_sensitive_span(self):
        """Sensitive span with non-placeholder text should fail."""

        class MockTokenizer:
            def decode(self, ids):
                mapping = {1: "John", 2: " ", 3: "Smith", 4: "hello"}
                return "".join(mapping.get(i, f"[{i}]") for i in ids)

        v = Verifier(VerifierConfig(
            forbidden_tags=[],
            check_placeholders=True,
        ))
        token_ids = [4, 1, 3, 4]
        pos_type = [SpanType.PUB, SpanType.SENS, SpanType.SENS, SpanType.PUB]

        result = v.check(
            "hello John Smith hello",
            token_ids=token_ids,
            pos_type=pos_type,
            tokenizer=MockTokenizer(),
        )
        assert not result.ok
        assert any(v_item["type"] == "structural" for v_item in result.violations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
