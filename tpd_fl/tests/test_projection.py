"""
Tests for TPD logits projection — the core hard guarantee.

These tests verify that:
1. Projection never allows forbidden token IDs for SENS/REG positions.
2. PUB positions remain unrestricted after projection.
3. Sampling from projected logits never produces forbidden tokens.
4. Adversarial logits (maximally favouring forbidden tokens) are still
   correctly blocked.
5. The guarantee holds for batched inputs.
"""

import pytest
import torch

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES
from tpd_fl.tpd.projection import project_logits, ProjectionEngine


# --------------- fixtures ---------------

VOCAB_SIZE = 100
FORBIDDEN_IDS = list(range(50, 100))  # tokens 50–99 are forbidden for SENS
ALLOWED_IDS = list(range(0, 50))       # tokens 0–49 are allowed for SENS


def _make_masks(device="cpu"):
    """Build test allowed-set masks."""
    pub_mask = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)

    sens_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    sens_mask[torch.tensor(ALLOWED_IDS, dtype=torch.long)] = True

    reg_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    reg_mask[torch.tensor(ALLOWED_IDS[:10], dtype=torch.long)] = True  # even stricter

    return {
        SpanType.PUB: pub_mask,
        SpanType.SENS: sens_mask,
        SpanType.REG: reg_mask,
        SpanType.DERIVED_EMAIL: sens_mask.clone(),
        SpanType.DERIVED_PHONE: sens_mask.clone(),
    }


def _make_pos_type(L=20):
    """Create a mixed type sequence: first 10 PUB, next 5 SENS, next 5 REG."""
    return (
        [SpanType.PUB] * 10
        + [SpanType.SENS] * 5
        + [SpanType.REG] * 5
    )


# --------------- unit tests ---------------


class TestProjectLogits:
    """Unit tests for project_logits."""

    def test_pub_positions_unrestricted(self):
        """PUB positions should keep all logits unchanged."""
        L, V = 20, VOCAB_SIZE
        logits = torch.randn(L, V)
        original = logits.clone()
        masks = _make_masks()
        pos_type = _make_pos_type(L)

        project_logits(logits, pos_type, masks)

        # PUB positions (0–9) should be unchanged
        assert torch.allclose(logits[:10], original[:10])

    def test_sens_forbidden_tokens_neg_inf(self):
        """SENS positions must have -inf for forbidden tokens."""
        L, V = 20, VOCAB_SIZE
        logits = torch.randn(L, V)
        masks = _make_masks()
        pos_type = _make_pos_type(L)

        project_logits(logits, pos_type, masks)

        neg_inf = torch.finfo(logits.dtype).min

        # SENS positions (10–14): forbidden tokens (50–99) should be -inf
        for i in range(10, 15):
            for v in FORBIDDEN_IDS:
                assert logits[i, v] == neg_inf, f"pos={i}, token={v}"

    def test_reg_forbidden_tokens_neg_inf(self):
        """REG positions must have -inf for all non-allowed tokens."""
        L, V = 20, VOCAB_SIZE
        logits = torch.randn(L, V)
        masks = _make_masks()
        pos_type = _make_pos_type(L)

        project_logits(logits, pos_type, masks)

        neg_inf = torch.finfo(logits.dtype).min

        # REG positions (15–19): only tokens 0–9 allowed
        for i in range(15, 20):
            for v in range(10, VOCAB_SIZE):
                assert logits[i, v] == neg_inf, f"pos={i}, token={v}"

    def test_allowed_tokens_preserved(self):
        """Allowed tokens should keep their original logit values."""
        L, V = 20, VOCAB_SIZE
        logits = torch.randn(L, V)
        original = logits.clone()
        masks = _make_masks()
        pos_type = _make_pos_type(L)

        project_logits(logits, pos_type, masks)

        # SENS positions: allowed tokens (0–49) should be unchanged
        for i in range(10, 15):
            for v in ALLOWED_IDS:
                assert logits[i, v] == original[i, v], f"pos={i}, token={v}"

    def test_sampling_never_produces_forbidden(self):
        """After projection, sampling must never produce forbidden tokens."""
        L, V = 20, VOCAB_SIZE
        masks = _make_masks()
        pos_type = _make_pos_type(L)

        for trial in range(50):
            logits = torch.randn(L, V)
            # Make forbidden tokens the most attractive
            logits[:, FORBIDDEN_IDS] += 100.0

            project_logits(logits, pos_type, masks)
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)

            for i in range(10, 15):  # SENS
                assert int(sampled[i]) in ALLOWED_IDS, (
                    f"Trial {trial}: SENS pos {i} sampled forbidden token {int(sampled[i])}"
                )
            for i in range(15, 20):  # REG
                assert int(sampled[i]) in ALLOWED_IDS[:10], (
                    f"Trial {trial}: REG pos {i} sampled forbidden token {int(sampled[i])}"
                )

    def test_adversarial_logits_blocked(self):
        """Even with extreme logits favouring forbidden tokens, projection blocks them."""
        L, V = 20, VOCAB_SIZE
        masks = _make_masks()
        pos_type = _make_pos_type(L)

        # Adversarial: set forbidden logits to +1000, allowed to -1000
        logits = torch.full((L, V), -1000.0)
        logits[:, FORBIDDEN_IDS] = 1000.0

        project_logits(logits, pos_type, masks)

        # Verify softmax gives zero prob to forbidden tokens
        probs = torch.softmax(logits, dim=-1)
        for i in range(10, 20):
            for v in FORBIDDEN_IDS:
                assert probs[i, v] == 0.0, f"pos={i}, token={v}, prob={probs[i, v]}"

    def test_positions_subset(self):
        """Projection with a position subset should only modify those positions."""
        L, V = 20, VOCAB_SIZE
        logits = torch.randn(L, V)
        original = logits.clone()
        masks = _make_masks()
        pos_type = _make_pos_type(L)

        # Only project positions 10, 11 (SENS)
        positions = torch.tensor([10, 11])
        project_logits(logits, pos_type, masks, positions=positions)

        neg_inf = torch.finfo(logits.dtype).min

        # These should be modified
        for i in [10, 11]:
            for v in FORBIDDEN_IDS:
                assert logits[i, v] == neg_inf

        # These should be unchanged (not in positions subset)
        for i in [12, 13, 14, 15, 16, 17, 18, 19]:
            assert torch.allclose(logits[i], original[i])


class TestProjectionEngine:
    """Tests for the ProjectionEngine wrapper."""

    def test_verify_hard_guarantee_passes(self):
        """verify_hard_guarantee returns True for valid tokens."""
        masks = _make_masks()
        pos_type = _make_pos_type(20)
        engine = ProjectionEngine(masks, pos_type)

        # Set all tokens to allowed values
        tokens = torch.zeros(20, dtype=torch.long)
        # PUB: anything is fine (0 is fine)
        # SENS: token 5 is in ALLOWED_IDS
        tokens[10:15] = 5
        # REG: token 3 is in ALLOWED_IDS[:10]
        tokens[15:20] = 3

        assert engine.verify_hard_guarantee(tokens) is True

    def test_verify_hard_guarantee_fails(self):
        """verify_hard_guarantee returns False for forbidden tokens."""
        masks = _make_masks()
        pos_type = _make_pos_type(20)
        engine = ProjectionEngine(masks, pos_type)

        tokens = torch.zeros(20, dtype=torch.long)
        tokens[12] = 75  # forbidden for SENS

        assert engine.verify_hard_guarantee(tokens) is False

    def test_project_modifies_logits(self):
        """Engine.project applies projection correctly."""
        masks = _make_masks()
        pos_type = _make_pos_type(20)
        engine = ProjectionEngine(masks, pos_type)

        logits = torch.randn(20, VOCAB_SIZE)
        engine.project(logits)

        neg_inf = torch.finfo(logits.dtype).min
        for i in range(10, 15):
            for v in FORBIDDEN_IDS:
                assert logits[i, v] == neg_inf


class TestProjectionPropertyBased:
    """Property-based / randomized tests for projection correctness."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_logits_forbidden_never_sampled(self, seed):
        """With random logits and forced forbidden maxima, projection blocks them."""
        torch.manual_seed(seed)
        L, V = 30, VOCAB_SIZE
        masks = _make_masks()
        pos_type = (
            [SpanType.PUB] * 10
            + [SpanType.SENS] * 10
            + [SpanType.REG] * 10
        )

        logits = torch.randn(L, V)
        # Force forbidden tokens to be the maximum at sensitive positions
        for i in range(10, 30):
            max_forbidden = torch.randint(50, 100, (1,)).item()
            logits[i, max_forbidden] = 1000.0

        project_logits(logits, pos_type, masks)
        probs = torch.softmax(logits, dim=-1)

        # Sample many times
        for _ in range(100):
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            for i in range(10, 20):  # SENS
                assert int(sampled[i]) < 50, f"SENS pos {i}: sampled {int(sampled[i])}"
            for i in range(20, 30):  # REG
                assert int(sampled[i]) < 10, f"REG pos {i}: sampled {int(sampled[i])}"

    @pytest.mark.parametrize("seed", range(10))
    def test_softmax_forbidden_probability_zero(self, seed):
        """After projection, softmax probability of forbidden tokens must be exactly 0."""
        torch.manual_seed(seed)
        L, V = 15, VOCAB_SIZE
        masks = _make_masks()
        pos_type = [SpanType.PUB] * 5 + [SpanType.SENS] * 5 + [SpanType.REG] * 5

        logits = torch.randn(L, V) * 10  # wide spread
        project_logits(logits, pos_type, masks)
        probs = torch.softmax(logits, dim=-1)

        for i in range(5, 10):  # SENS
            assert probs[i, 50:].sum() == 0.0
        for i in range(10, 15):  # REG
            assert probs[i, 10:].sum() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
