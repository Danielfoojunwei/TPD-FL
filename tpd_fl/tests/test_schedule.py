"""
Tests for TPD mask schedule.

Verifies:
1. Phase boundaries are correct.
2. DRAFT phase never allows SENS/REG positions.
3. SAFE phase allows all positions (with projection handling constraints).
4. REVEAL phase restricts to PUB + reveal_types.
5. Schedule intersection with proposed mask works correctly.
"""

import pytest
import torch

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES
from tpd_fl.tpd.schedule import (
    MaskSchedule,
    ScheduleConfig,
    SchedulePhase,
    build_random_block_mask,
)


def _make_pos_type(L=20):
    """PUB*8, SENS*6, REG*6."""
    return [SpanType.PUB] * 8 + [SpanType.SENS] * 6 + [SpanType.REG] * 6


class TestPhaseBoundaries:
    """Verify phase transitions at configured boundaries."""

    def test_default_boundaries(self):
        sched = MaskSchedule(ScheduleConfig(draft_end=0.4, safe_end=0.9))
        T = 100
        assert sched.phase(0, T) == SchedulePhase.DRAFT
        assert sched.phase(10, T) == SchedulePhase.DRAFT
        assert sched.phase(39, T) == SchedulePhase.DRAFT
        assert sched.phase(40, T) == SchedulePhase.SAFE
        assert sched.phase(60, T) == SchedulePhase.SAFE
        assert sched.phase(89, T) == SchedulePhase.SAFE
        assert sched.phase(90, T) == SchedulePhase.REVEAL
        assert sched.phase(99, T) == SchedulePhase.REVEAL

    def test_custom_boundaries(self):
        sched = MaskSchedule(ScheduleConfig(draft_end=0.2, safe_end=0.8))
        T = 50
        assert sched.phase(0, T) == SchedulePhase.DRAFT
        assert sched.phase(9, T) == SchedulePhase.DRAFT
        assert sched.phase(10, T) == SchedulePhase.SAFE
        assert sched.phase(39, T) == SchedulePhase.SAFE
        assert sched.phase(40, T) == SchedulePhase.REVEAL

    def test_single_step(self):
        sched = MaskSchedule(ScheduleConfig(draft_end=0.4, safe_end=0.9))
        # T=1: step 0 is frac=0.0 -> DRAFT
        assert sched.phase(0, 1) == SchedulePhase.DRAFT


class TestDraftPhase:
    """DRAFT phase: only PUB positions updatable."""

    def test_draft_only_pub(self):
        sched = MaskSchedule(ScheduleConfig(draft_end=0.5, safe_end=0.9))
        pos_type = _make_pos_type(20)
        T = 100
        t = 10  # well within DRAFT

        allowed = sched.allowed_positions(t, T, pos_type)

        # PUB positions (0–7) should be True
        for i in range(8):
            assert allowed[i].item() is True, f"PUB pos {i} should be allowed in DRAFT"

        # SENS and REG (8–19) should be False
        for i in range(8, 20):
            assert allowed[i].item() is False, f"SENS/REG pos {i} should not be allowed in DRAFT"

    def test_draft_schedule_intersection(self):
        """apply_schedule in DRAFT blocks SENS/REG even if proposed."""
        sched = MaskSchedule(ScheduleConfig(draft_end=0.5, safe_end=0.9))
        pos_type = _make_pos_type(20)
        T = 100
        t = 5

        # Propose updating everything
        proposed = torch.ones(20, dtype=torch.bool)
        result = sched.apply_schedule(proposed, t, T, pos_type)

        assert result[:8].all()       # PUB allowed
        assert not result[8:].any()   # SENS/REG blocked


class TestSafePhase:
    """SAFE phase: all positions allowed (projection handles constraints)."""

    def test_safe_allows_all(self):
        sched = MaskSchedule(ScheduleConfig(draft_end=0.3, safe_end=0.9))
        pos_type = _make_pos_type(20)
        T = 100
        t = 50  # SAFE phase

        allowed = sched.allowed_positions(t, T, pos_type)
        assert allowed.all()

    def test_safe_schedule_intersection(self):
        sched = MaskSchedule(ScheduleConfig(draft_end=0.3, safe_end=0.9))
        pos_type = _make_pos_type(20)
        T = 100
        t = 50

        # Propose a subset
        proposed = torch.zeros(20, dtype=torch.bool)
        proposed[5] = True
        proposed[12] = True  # SENS
        proposed[18] = True  # REG

        result = sched.apply_schedule(proposed, t, T, pos_type)
        assert result[5].item() is True
        assert result[12].item() is True
        assert result[18].item() is True
        assert result.sum() == 3


class TestRevealPhase:
    """REVEAL phase: PUB + reveal_types only."""

    def test_reveal_pub_only_by_default(self):
        sched = MaskSchedule(ScheduleConfig(
            draft_end=0.3, safe_end=0.8, reveal_types=[]
        ))
        pos_type = _make_pos_type(20)
        T = 100
        t = 95  # REVEAL

        allowed = sched.allowed_positions(t, T, pos_type)
        for i in range(8):
            assert allowed[i].item() is True
        for i in range(8, 20):
            assert allowed[i].item() is False

    def test_reveal_with_sens_allowed(self):
        sched = MaskSchedule(ScheduleConfig(
            draft_end=0.3, safe_end=0.8, reveal_types=[SpanType.SENS]
        ))
        pos_type = _make_pos_type(20)
        T = 100
        t = 95

        allowed = sched.allowed_positions(t, T, pos_type)
        # PUB
        assert allowed[:8].all()
        # SENS — allowed in reveal
        assert allowed[8:14].all()
        # REG — not in reveal_types
        assert not allowed[14:].any()


class TestBuildRandomBlockMask:
    """Tests for the random block mask builder."""

    def test_correct_count(self):
        mask = build_random_block_mask(50, 10)
        assert mask.sum() == 10

    def test_max_clamp(self):
        mask = build_random_block_mask(5, 100)
        assert mask.sum() == 5  # clamped to seq_len

    def test_deterministic_with_generator(self):
        g1 = torch.Generator()
        g1.manual_seed(42)
        m1 = build_random_block_mask(50, 10, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(42)
        m2 = build_random_block_mask(50, 10, generator=g2)

        assert torch.equal(m1, m2)


class TestSchedulePropertyBased:
    """Randomized property tests for schedule invariants."""

    @pytest.mark.parametrize("seed", range(20))
    def test_draft_never_updates_sensitive(self, seed):
        """No matter the proposed mask, DRAFT never permits SENS/REG."""
        torch.manual_seed(seed)
        L = 50
        pos_type = (
            [SpanType.PUB] * 20
            + [SpanType.SENS] * 15
            + [SpanType.REG] * 15
        )
        sched = MaskSchedule(ScheduleConfig(draft_end=0.5, safe_end=0.9))
        T = 100

        for t in range(0, 50):  # all DRAFT steps
            proposed = torch.rand(L) > 0.5
            result = sched.apply_schedule(proposed, t, T, pos_type)
            # No SENS/REG position should be True
            for i in range(20, L):
                assert result[i].item() is False, (
                    f"Seed {seed}, step {t}: pos {i} (type {pos_type[i]}) "
                    f"was allowed in DRAFT"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
