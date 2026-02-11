"""
Property-based / randomized tests for TPD invariants.

Tests correspond to the formal theorems in proofs/tpd_semantics.tex:
  - Theorem 1: Type preservation — projection preserves type assignments
  - Theorem 2: Hard non-emission — forbidden tokens have zero probability
  - Theorem 3: Closure under editing — covered in test_edit_closure.py
  - Theorem 4: Verifier-lifted global safety

Also includes FL regression test:
  - Malicious adapter updates do not change hard leakage under TPD.
"""

import pytest
import torch

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES
from tpd_fl.tpd.projection import project_logits, ProjectionEngine
from tpd_fl.tpd.schedule import MaskSchedule, ScheduleConfig
from tpd_fl.tpd.verifier import Verifier, VerifierConfig


VOCAB_SIZE = 200
ALLOWED_SENS = list(range(0, 50))
ALLOWED_REG = list(range(0, 20))
FORBIDDEN_SENS = list(range(50, 200))
FORBIDDEN_REG = list(range(20, 200))


def _build_masks(device="cpu"):
    pub = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
    sens = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    sens[torch.tensor(ALLOWED_SENS)] = True
    reg = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    reg[torch.tensor(ALLOWED_REG)] = True
    return {SpanType.PUB: pub, SpanType.SENS: sens, SpanType.REG: reg}


# ---- Theorem 1: Type Preservation ----

class TestTypePreservation:
    """After projection, the type assignment is still valid:
    for all i, tokens[i] ∈ A(τ_i)."""

    @pytest.mark.parametrize("seed", range(25))
    def test_type_preservation_random(self, seed):
        torch.manual_seed(seed)
        L = 40
        pos_type = (
            [SpanType.PUB] * 15
            + [SpanType.SENS] * 15
            + [SpanType.REG] * 10
        )
        masks = _build_masks()

        logits = torch.randn(L, VOCAB_SIZE) * 10
        project_logits(logits, pos_type, masks)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)

        engine = ProjectionEngine(masks, pos_type)
        assert engine.verify_hard_guarantee(sampled), (
            f"Type preservation violated at seed {seed}"
        )


# ---- Theorem 2: Hard Non-Emission ----

class TestHardNonEmission:
    """For any position i with τ_i ∈ SENSITIVE_TYPES and any v ∉ A(τ_i):
    P(v | projected_logits) = 0."""

    @pytest.mark.parametrize("seed", range(25))
    def test_zero_probability_forbidden(self, seed):
        torch.manual_seed(seed)
        L = 30
        pos_type = [SpanType.PUB] * 10 + [SpanType.SENS] * 10 + [SpanType.REG] * 10
        masks = _build_masks()

        logits = torch.randn(L, VOCAB_SIZE) * 20  # high variance
        project_logits(logits, pos_type, masks)
        probs = torch.softmax(logits, dim=-1)

        for i in range(10, 20):  # SENS
            mass_forbidden = probs[i, 50:].sum().item()
            assert mass_forbidden == 0.0, (
                f"SENS pos {i}: forbidden mass = {mass_forbidden}"
            )

        for i in range(20, 30):  # REG
            mass_forbidden = probs[i, 20:].sum().item()
            assert mass_forbidden == 0.0, (
                f"REG pos {i}: forbidden mass = {mass_forbidden}"
            )

    def test_extreme_logits_still_zero(self):
        """Even with logits = +inf for forbidden tokens."""
        L = 10
        pos_type = [SpanType.SENS] * L
        masks = _build_masks()

        logits = torch.full((L, VOCAB_SIZE), -100.0)
        logits[:, 50:] = 100.0  # strongly favour forbidden

        project_logits(logits, pos_type, masks)
        probs = torch.softmax(logits, dim=-1)

        for i in range(L):
            assert probs[i, 50:].sum() == 0.0


# ---- Theorem 4: Verifier-Lifted Global Safety ----

class TestVerifierGlobalSafety:
    """If projection holds and verifier passes, the output satisfies π."""

    def test_clean_text_passes_verifier(self):
        verifier = Verifier(VerifierConfig(
            forbidden_tags=["EMAIL", "PHONE", "SSN"],
        ))
        result = verifier.check("This is a [REDACTED] document about [NAME].")
        assert result.ok

    def test_leaked_email_fails_verifier(self):
        verifier = Verifier(VerifierConfig(
            forbidden_tags=["EMAIL"],
        ))
        result = verifier.check("Contact me at john@example.com for details.")
        assert not result.ok
        assert any(v["tag"] == "EMAIL" for v in result.violations)

    def test_leaked_ssn_fails_verifier(self):
        verifier = Verifier(VerifierConfig(
            forbidden_tags=["SSN"],
        ))
        result = verifier.check("My SSN is 123-45-6789.")
        assert not result.ok

    def test_known_secrets_caught(self):
        verifier = Verifier(VerifierConfig(
            forbidden_tags=[],
            check_semantic=True,
            known_secrets=["john doe", "secret-project-x"],
        ))
        result = verifier.check("The person is John Doe and works on Secret-Project-X.")
        assert not result.ok
        assert len(result.violations) >= 2


# ---- FL Regression: Malicious Adapters Cannot Break Non-Emission ----

class TestFLCannotBreakNonEmission:
    """Demonstrate that regardless of FL adapter updates, projection
    ensures hard non-emission.

    Simulation: apply random perturbations to logits (simulating
    arbitrary adapter effects) and show projection still holds.
    """

    @pytest.mark.parametrize("seed", range(20))
    def test_malicious_adapter_perturbation(self, seed):
        """Random large perturbations to logits (simulating malicious adapter)
        cannot break the hard guarantee after projection."""
        torch.manual_seed(seed)
        L = 20
        pos_type = [SpanType.PUB] * 5 + [SpanType.SENS] * 10 + [SpanType.REG] * 5
        masks = _build_masks()

        # Base logits
        base_logits = torch.randn(L, VOCAB_SIZE)

        # "Malicious adapter": large perturbation favouring forbidden tokens
        adapter_delta = torch.zeros(L, VOCAB_SIZE)
        for i in range(5, 15):  # SENS positions
            # Strongly boost forbidden tokens
            adapter_delta[i, 50:] = torch.rand(150) * 100
        for i in range(15, 20):  # REG positions
            adapter_delta[i, 20:] = torch.rand(180) * 100

        perturbed = base_logits + adapter_delta

        # Apply projection
        project_logits(perturbed, pos_type, masks)
        probs = torch.softmax(perturbed, dim=-1)

        # Sample many times
        for _ in range(200):
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            for i in range(5, 15):
                assert int(sampled[i]) < 50, (
                    f"Seed {seed}: SENS pos {i} sampled {int(sampled[i])} "
                    f"despite malicious adapter"
                )
            for i in range(15, 20):
                assert int(sampled[i]) < 20, (
                    f"Seed {seed}: REG pos {i} sampled {int(sampled[i])} "
                    f"despite malicious adapter"
                )

    def test_gradient_attack_blocked(self):
        """Gradient-based attack on logits: maximize probability of forbidden tokens.
        Projection must still block them."""
        L = 10
        pos_type = [SpanType.SENS] * L
        masks = _build_masks()

        # Simulate gradient attack: set forbidden logits to maximum float value
        logits = torch.full((L, VOCAB_SIZE), 0.0)
        logits[:, 50:] = 1e6  # "attack" logits

        project_logits(logits, pos_type, masks)
        probs = torch.softmax(logits, dim=-1)

        # All forbidden probability must be zero
        assert probs[:, 50:].sum() == 0.0

    @pytest.mark.parametrize("num_rounds", [1, 5, 10])
    def test_iterated_adapter_updates(self, num_rounds):
        """Multiple rounds of adapter updates (FL rounds) cannot accumulate
        to break the guarantee."""
        L = 15
        pos_type = [SpanType.SENS] * L
        masks = _build_masks()

        logits = torch.randn(L, VOCAB_SIZE)
        for round_idx in range(num_rounds):
            # Simulate adapter update
            delta = torch.randn(L, VOCAB_SIZE) * (round_idx + 1)
            logits = logits + delta

            # Projection after each round
            proj_logits = logits.clone()
            project_logits(proj_logits, pos_type, masks)
            probs = torch.softmax(proj_logits, dim=-1)

            assert probs[:, 50:].sum() == 0.0, (
                f"Round {round_idx}: forbidden mass leaked"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
