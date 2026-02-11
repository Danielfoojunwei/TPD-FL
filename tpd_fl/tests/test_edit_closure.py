"""
Tests for Theorem 3: Closure under editing (T2T).

These tests verify that repair/edit operations are monotone in
sensitivity — they never introduce tokens from a forbidden class
that was not already present.

Invariant: for any edit operation on a sequence that satisfies the
type constraints, the resulting sequence also satisfies the type
constraints.
"""

import pytest
import torch

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES
from tpd_fl.tpd.projection import project_logits, ProjectionEngine
from tpd_fl.tpd.repair import RepairEngine, RepairMode


VOCAB_SIZE = 100
ALLOWED_IDS = list(range(0, 30))
FORBIDDEN_IDS = list(range(30, 100))


def _make_masks(device="cpu"):
    pub_mask = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
    sens_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    sens_mask[torch.tensor(ALLOWED_IDS, dtype=torch.long)] = True
    return {
        SpanType.PUB: pub_mask,
        SpanType.SENS: sens_mask,
        SpanType.REG: sens_mask.clone(),
    }


def _make_valid_sequence(L=20):
    """Create a sequence that satisfies type constraints."""
    pos_type = [SpanType.PUB] * 10 + [SpanType.SENS] * 5 + [SpanType.REG] * 5
    tokens = torch.zeros(L, dtype=torch.long)
    # PUB: any token is fine
    tokens[:10] = torch.randint(0, VOCAB_SIZE, (10,))
    # SENS: must be in ALLOWED_IDS
    tokens[10:15] = torch.tensor([5, 10, 15, 20, 25])
    # REG: must be in ALLOWED_IDS
    tokens[15:20] = torch.tensor([0, 1, 2, 3, 4])
    return tokens, pos_type


class TestEditClosure:
    """Verify that edit/repair operations preserve type constraints."""

    def test_resample_repair_preserves_types(self):
        """After resample repair, all positions still satisfy type constraints."""
        masks = _make_masks()
        tokens, pos_type = _make_valid_sequence()

        # Intentionally violate some positions
        tokens[12] = 50  # forbidden for SENS
        tokens[17] = 80  # forbidden for REG
        violating = [12, 17]

        engine = RepairEngine(mode=RepairMode.RESAMPLE, max_repair_iters=5)
        repaired, iters = engine.repair(
            tokens, violating, pos_type, masks,
            model_fn=None,  # uses fallback uniform logits
            mask_token_id=0,
        )

        # Check all sensitive positions are in allowed set
        for i in range(10, 20):
            tid = int(repaired[i])
            assert tid in ALLOWED_IDS, (
                f"Position {i} (type {pos_type[i]}) has token {tid} "
                f"which is not in ALLOWED_IDS after repair"
            )

    def test_edit_repair_preserves_types(self):
        """After edit repair, all positions still satisfy type constraints."""
        masks = _make_masks()
        tokens, pos_type = _make_valid_sequence()

        tokens[11] = 60  # forbidden
        tokens[16] = 90  # forbidden
        violating = [11, 16]

        engine = RepairEngine(mode=RepairMode.EDIT, max_repair_iters=3)
        repaired, iters = engine.repair(
            tokens, violating, pos_type, masks,
        )

        for i in range(10, 20):
            tid = int(repaired[i])
            assert tid in ALLOWED_IDS, (
                f"Position {i} has forbidden token {tid} after edit repair"
            )

    def test_repair_does_not_modify_pub_positions(self):
        """Repair should not change PUB positions."""
        masks = _make_masks()
        tokens, pos_type = _make_valid_sequence()
        original_pub = tokens[:10].clone()

        tokens[12] = 75
        violating = [12]

        engine = RepairEngine(mode=RepairMode.RESAMPLE)
        repaired, _ = engine.repair(
            tokens, violating, pos_type, masks,
            model_fn=None, mask_token_id=0,
        )

        assert torch.equal(repaired[:10], original_pub)

    def test_repair_non_violating_positions_unchanged(self):
        """Positions not in the violating set should remain unchanged."""
        masks = _make_masks()
        tokens, pos_type = _make_valid_sequence()

        # Only position 13 violates
        tokens[13] = 55
        violating = [13]

        engine = RepairEngine(mode=RepairMode.EDIT)
        repaired, _ = engine.repair(tokens, violating, pos_type, masks)

        # Positions 10, 11, 12, 14 should be unchanged
        for i in [10, 11, 12, 14]:
            assert repaired[i] == tokens[i]

    @pytest.mark.parametrize("seed", range(15))
    def test_repeated_repair_monotone(self, seed):
        """Multiple repair rounds are monotone: never introduce new violations."""
        torch.manual_seed(seed)
        masks = _make_masks()
        L = 30
        pos_type = [SpanType.PUB] * 10 + [SpanType.SENS] * 10 + [SpanType.REG] * 10

        # Start with random violations
        tokens = torch.randint(0, VOCAB_SIZE, (L,))
        # Fix PUB (anything ok)
        # Find violating SENS/REG positions
        violating = []
        for i in range(10, L):
            if int(tokens[i]) not in ALLOWED_IDS:
                violating.append(i)

        engine = RepairEngine(mode=RepairMode.RESAMPLE, max_repair_iters=10)
        repaired, iters = engine.repair(
            tokens, violating, pos_type, masks,
            model_fn=None, mask_token_id=0,
        )

        # After repair, no violations should remain
        for i in range(10, L):
            tid = int(repaired[i])
            assert tid in ALLOWED_IDS, (
                f"Seed {seed}: pos {i} still has forbidden token {tid} after repair"
            )

    def test_projection_then_sample_closure(self):
        """project -> sample -> check is a closed operation on type constraints."""
        masks = _make_masks()
        L = 20
        pos_type = [SpanType.PUB] * 10 + [SpanType.SENS] * 5 + [SpanType.REG] * 5

        engine = ProjectionEngine(masks, pos_type)

        for _ in range(50):
            logits = torch.randn(L, VOCAB_SIZE) * 5
            engine.project(logits)
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)

            # Verify type constraints
            assert engine.verify_hard_guarantee(sampled), (
                "Closure violation: sample after projection violates type constraints"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
