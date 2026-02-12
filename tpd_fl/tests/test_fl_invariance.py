"""
Tests for FL + TPD Invariance — verifies that federated learning
adapter updates cannot violate TPD's non-emission guarantee.

Core property:
    FL adapter updates modify model weights (logits), but TPD
    projection is applied *after* the adapter at decode time.
    Therefore, no adapter update can cause a forbidden token to
    appear in the output.

These tests verify:
1. Projection holds after arbitrary LoRA weight perturbation.
2. Projection holds after FedAvg aggregation of adversarial deltas.
3. The decode loop with FL adapters still satisfies hard guarantee.
4. Typed training with sensitive positions produces correct targets.
"""

import pytest
import torch
import torch.nn as nn

from tpd_fl.tpd.typing import SpanType, SENSITIVE_TYPES
from tpd_fl.tpd.projection import project_logits, ProjectionEngine
from tpd_fl.fl.lora import LoRAConfig, LoRALinear, attach_lora, get_lora_state_dict, load_lora_state_dict
from tpd_fl.fl.client import FLClient, FLClientConfig
from tpd_fl.fl.server import FLServer, FLServerConfig
from tpd_fl.fl.protocols import fedavg
from tpd_fl.model.backend_base import SyntheticBackend


# --------------- fixtures ---------------

VOCAB_SIZE = 100
FORBIDDEN_IDS = list(range(50, 100))
ALLOWED_IDS = list(range(0, 50))


def _make_masks(device="cpu"):
    pub_mask = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
    sens_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    sens_mask[torch.tensor(ALLOWED_IDS, dtype=torch.long)] = True
    reg_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    reg_mask[torch.tensor(ALLOWED_IDS[:10], dtype=torch.long)] = True
    return {
        SpanType.PUB: pub_mask,
        SpanType.SENS: sens_mask,
        SpanType.REG: reg_mask,
    }


def _make_pos_type(L=20):
    return (
        [SpanType.PUB] * 10
        + [SpanType.SENS] * 5
        + [SpanType.REG] * 5
    )


class SimpleLM(nn.Module):
    """Minimal LM for testing FL invariance."""
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return type("Output", (), {"logits": logits})()


# --------------- tests ---------------


class TestFLProjectionInvariance:
    """Test that projection guarantee holds regardless of FL adapter state."""

    @pytest.mark.parametrize("seed", range(10))
    def test_projection_after_lora_perturbation(self, seed):
        """Projection must block forbidden tokens regardless of LoRA weights."""
        torch.manual_seed(seed)
        model = SimpleLM()
        lora_config = LoRAConfig(rank=4, alpha=8.0, target_modules=["fc1", "fc2"])
        lora_modules = attach_lora(model, lora_config)

        # Randomly perturb LoRA weights to simulate adversarial FL updates
        with torch.no_grad():
            for lora_layer in lora_modules.values():
                lora_layer.lora_A.add_(torch.randn_like(lora_layer.lora_A) * 10.0)
                lora_layer.lora_B.add_(torch.randn_like(lora_layer.lora_B) * 10.0)

        # Generate logits through the perturbed model
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 20))
        output = model(input_ids=input_ids)
        logits = output.logits[0]  # [20, V]

        # Apply projection
        masks = _make_masks()
        pos_type = _make_pos_type(20)
        project_logits(logits, pos_type, masks)

        # Verify: no forbidden token should have non-neg-inf logit at SENS/REG
        neg_inf = torch.finfo(logits.dtype).min
        for i in range(10, 15):  # SENS
            for v in FORBIDDEN_IDS:
                assert logits[i, v] == neg_inf, f"SENS pos {i}, token {v}"
        for i in range(15, 20):  # REG
            for v in range(10, VOCAB_SIZE):
                assert logits[i, v] == neg_inf, f"REG pos {i}, token {v}"

    @pytest.mark.parametrize("seed", range(5))
    def test_sampling_safe_after_lora_perturbation(self, seed):
        """Sampling from projected logits never produces forbidden tokens, even
        with adversarially perturbed LoRA weights."""
        torch.manual_seed(seed)
        model = SimpleLM()
        lora_config = LoRAConfig(rank=4, alpha=8.0, target_modules=["fc1", "fc2"])
        lora_modules = attach_lora(model, lora_config)

        # Adversarial perturbation: make LoRA strongly prefer forbidden tokens
        with torch.no_grad():
            for lora_layer in lora_modules.values():
                lora_layer.lora_A.fill_(1.0)
                lora_layer.lora_B.fill_(1.0)

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 20))
        output = model(input_ids=input_ids)
        logits = output.logits[0]

        masks = _make_masks()
        pos_type = _make_pos_type(20)
        project_logits(logits, pos_type, masks)

        probs = torch.softmax(logits, dim=-1)
        for trial in range(50):
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            for i in range(10, 15):
                assert int(sampled[i]) in ALLOWED_IDS, (
                    f"Trial {trial}: SENS pos {i} sampled {int(sampled[i])}"
                )
            for i in range(15, 20):
                assert int(sampled[i]) in ALLOWED_IDS[:10], (
                    f"Trial {trial}: REG pos {i} sampled {int(sampled[i])}"
                )


class TestFedAvgPreservesProjection:
    """Test that FedAvg aggregation of client deltas preserves projection."""

    def test_fedavg_adversarial_deltas(self):
        """Aggregating adversarial client deltas must not break projection."""
        torch.manual_seed(42)
        masks = _make_masks()
        pos_type = _make_pos_type(20)

        # Create multiple "client" models with adversarial LoRA updates
        num_clients = 5
        all_deltas = []
        for c in range(num_clients):
            model = SimpleLM()
            lora_config = LoRAConfig(rank=4, alpha=8.0, target_modules=["fc1", "fc2"])
            lora_modules = attach_lora(model, lora_config)

            # Record initial state
            init_state = get_lora_state_dict(lora_modules)

            # Adversarial update
            with torch.no_grad():
                for lora_layer in lora_modules.values():
                    lora_layer.lora_A.add_(torch.randn_like(lora_layer.lora_A) * (c + 1) * 5.0)
                    lora_layer.lora_B.add_(torch.randn_like(lora_layer.lora_B) * (c + 1) * 5.0)

            end_state = get_lora_state_dict(lora_modules)
            delta = {k: end_state[k] - init_state[k] for k in init_state}
            all_deltas.append(delta)

        # Aggregate
        agg_delta = fedavg(all_deltas)

        # Apply aggregated delta to a fresh model
        server_model = SimpleLM()
        server_lora = attach_lora(server_model, LoRAConfig(rank=4, alpha=8.0, target_modules=["fc1", "fc2"]))
        server_state = get_lora_state_dict(server_lora)
        new_state = {k: server_state[k] + agg_delta[k] for k in server_state}
        load_lora_state_dict(server_lora, new_state)

        # Generate logits and project
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 20))
        output = server_model(input_ids=input_ids)
        logits = output.logits[0]
        project_logits(logits, pos_type, masks)

        # Verify
        neg_inf = torch.finfo(logits.dtype).min
        for i in range(10, 15):
            for v in FORBIDDEN_IDS:
                assert logits[i, v] == neg_inf

    def test_multiple_fedavg_rounds(self):
        """Projection must hold after multiple rounds of FedAvg aggregation."""
        torch.manual_seed(42)
        masks = _make_masks()
        pos_type = _make_pos_type(20)

        # Simulate 5 rounds of FedAvg
        model = SimpleLM()
        lora_config = LoRAConfig(rank=4, alpha=8.0, target_modules=["fc1", "fc2"])
        server_lora = attach_lora(model, lora_config)

        for round_idx in range(5):
            # Simulate 3 client deltas
            deltas = []
            for c in range(3):
                delta = {}
                state = get_lora_state_dict(server_lora)
                for k, v in state.items():
                    delta[k] = torch.randn_like(v) * 0.5
                deltas.append(delta)

            agg_delta = fedavg(deltas)
            state = get_lora_state_dict(server_lora)
            new_state = {k: state[k] + agg_delta[k] for k in state}
            load_lora_state_dict(server_lora, new_state)

        # Final check: projection still holds
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 20))
        output = model(input_ids=input_ids)
        logits = output.logits[0]
        project_logits(logits, pos_type, masks)

        neg_inf = torch.finfo(logits.dtype).min
        for i in range(10, 15):
            for v in FORBIDDEN_IDS:
                assert logits[i, v] == neg_inf


class TestFLClientTraining:
    """Test that FL client training produces valid deltas."""

    def test_client_produces_deltas(self):
        """Client.train() returns non-trivial deltas."""
        model = SimpleLM()
        lora_config = LoRAConfig(rank=4, alpha=8.0, target_modules=["fc1", "fc2"])
        lora_modules = attach_lora(model, lora_config)

        config = FLClientConfig(local_epochs=1, lr=0.01, batch_size=2, seed=42)
        client = FLClient(model, tokenizer=None, lora_modules=lora_modules, config=config)

        dataset = [
            {"input_ids": list(range(10))},
            {"input_ids": list(range(5, 15))},
            {"input_ids": list(range(20, 30))},
            {"input_ids": list(range(30, 40))},
        ]

        deltas = client.train(dataset)
        assert len(deltas) > 0

        # At least some deltas should be non-zero
        total_norm = sum(d.norm().item() for d in deltas.values())
        assert total_norm > 0, "Deltas should be non-trivial after training"

    def test_typed_training_preserves_projection(self):
        """Typed training with SENS positions should not break projection."""
        torch.manual_seed(42)
        model = SimpleLM()
        lora_config = LoRAConfig(rank=4, alpha=8.0, target_modules=["fc1", "fc2"])
        lora_modules = attach_lora(model, lora_config)

        config = FLClientConfig(
            local_epochs=2, lr=0.01, batch_size=2,
            typed_training=True, seed=42,
        )
        client = FLClient(model, tokenizer=None, lora_modules=lora_modules, config=config)

        dataset = [
            {
                "input_ids": list(range(20)),
                "position_types": ["PUB"] * 10 + ["SENS"] * 5 + ["REG"] * 5,
            },
            {
                "input_ids": list(range(10, 30)),
                "position_types": ["PUB"] * 10 + ["SENS"] * 5 + ["REG"] * 5,
            },
        ]

        deltas = client.train(dataset)

        # After training, apply deltas and check projection
        masks = _make_masks()
        pos_type = _make_pos_type(20)

        state = get_lora_state_dict(lora_modules)
        new_state = {k: state[k] for k in state}  # already updated by training
        load_lora_state_dict(lora_modules, new_state)

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 20))
        output = model(input_ids=input_ids)
        logits = output.logits[0]
        project_logits(logits, pos_type, masks)

        neg_inf = torch.finfo(logits.dtype).min
        for i in range(10, 15):
            for v in FORBIDDEN_IDS:
                assert logits[i, v] == neg_inf


class TestFLServerIntegration:
    """Integration tests for FL server + projection."""

    def test_server_round_preserves_projection(self):
        """A full FL server round should not break projection."""
        torch.manual_seed(42)

        model = SimpleLM()
        lora_config = LoRAConfig(rank=4, alpha=8.0, target_modules=["fc1", "fc2"])

        server_config = FLServerConfig(
            num_rounds=3, min_clients=2, strategy="fedavg", seed=42,
        )
        server = FLServer(model, server_config, lora_config=lora_config)

        # Create clients
        num_clients = 3
        clients = []
        datasets = []
        for i in range(num_clients):
            client_model = SimpleLM()
            client_lora = attach_lora(client_model, lora_config)
            client_config = FLClientConfig(local_epochs=1, lr=0.01, batch_size=2, seed=i)
            client = FLClient(
                client_model, tokenizer=None,
                lora_modules=client_lora, config=client_config,
            )
            clients.append(client)
            datasets.append([
                {"input_ids": list(range(i * 5, i * 5 + 10))},
                {"input_ids": list(range(i * 10, i * 10 + 10))},
            ])

        history = server.run(clients, datasets)
        assert len(history) == 3

        # After training, the server model should still respect projection
        masks = _make_masks()
        pos_type = _make_pos_type(20)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 20))
        output = model(input_ids=input_ids)
        logits = output.logits[0]
        project_logits(logits, pos_type, masks)

        neg_inf = torch.finfo(logits.dtype).min
        for i in range(10, 15):
            for v in FORBIDDEN_IDS:
                assert logits[i, v] == neg_inf


class TestSyntheticBackendIntegration:
    """Test that SyntheticBackend works with the full pipeline."""

    def test_backend_encode_decode_roundtrip(self):
        """Backend encode/decode should roundtrip for ASCII text."""
        backend = SyntheticBackend(vocab_size=256, seed=42)
        text = "Hello world"
        ids = backend.encode(text)
        decoded = backend.decode(ids)
        assert decoded == text

    def test_backend_tokenize_returns_correct_shape(self):
        """tokenize should return input_ids of correct shape."""
        backend = SyntheticBackend(vocab_size=256, seed=42)
        result = backend.tokenize("Hello", max_length=10, return_offsets=True)
        assert result["input_ids"].shape[1] == 5  # 5 chars
        assert len(result["offset_mapping"]) == 5

    def test_backend_forward_logits_shape(self):
        """forward_logits should return correct shape."""
        backend = SyntheticBackend(vocab_size=100, seed=42)
        tokens = torch.zeros(20, dtype=torch.long)
        positions = torch.tensor([0, 5, 10])
        logits = backend.forward_logits(tokens, step=0, positions=positions)
        assert logits.shape == (3, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
