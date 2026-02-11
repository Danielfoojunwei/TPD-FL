"""
Federated Learning Server — aggregation and round management for TPD+FL.

The FL server coordinates federated training across multiple clients:

1. **Broadcast**: send the current global adapter state to all clients.
2. **Local training**: each client trains on its private data partition
   and returns adapter deltas (handled by :class:`tpd_fl.fl.client.FLClient`).
3. **Aggregate**: combine client deltas into a global update using
   the configured strategy (FedAvg, FedAdam, or SecureAgg).
4. **Apply**: update the global adapter state with the aggregated delta.

The server maintains training history (per-round metrics) and supports
both synchronous simulation and (future) asynchronous operation.

Usage::

    from tpd_fl.fl.server import FLServer, FLServerConfig

    server = FLServer(model, FLServerConfig(num_rounds=50))
    history = server.run(clients, datasets)

This module depends on torch, the Python standard library, and
:mod:`tpd_fl.fl.protocols`.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from tpd_fl.fl.lora import (
    LoRAConfig,
    LoRALinear,
    attach_lora,
    get_lora_state_dict,
    load_lora_state_dict,
)
from tpd_fl.fl.protocols import (
    FedAdamServerState,
    fedavg,
    fedadam,
    SecureAggStub,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FLServerConfig:
    """Configuration for the federated learning server.

    Attributes
    ----------
    num_rounds : int
        Total number of federated training rounds.
    min_clients : int
        Minimum number of clients required per round.  If fewer
        clients are available, the round is skipped.
    strategy : str
        Aggregation strategy.  One of:
        - ``"fedavg"`` — Federated Averaging (default).
        - ``"fedadam"`` — Federated Adam with server-side momentum.
        - ``"secureagg"`` — Secure aggregation stub.
    fedadam_lr : float
        Server-side learning rate for FedAdam.
    fedadam_betas : Tuple[float, float]
        Momentum coefficients for FedAdam.
    fedadam_tau : float
        Numerical stability constant for FedAdam.
    client_fraction : float
        Fraction of clients to sample per round.  1.0 means all
        clients participate every round.
    log_every : int
        Log metrics every N rounds.
    seed : Optional[int]
        Random seed for client sampling reproducibility.
    """

    num_rounds: int = 50
    min_clients: int = 2
    strategy: str = "fedavg"
    fedadam_lr: float = 1e-3
    fedadam_betas: Tuple[float, float] = (0.9, 0.99)
    fedadam_tau: float = 1e-3
    client_fraction: float = 1.0
    log_every: int = 1
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# FL Server
# ---------------------------------------------------------------------------

class FLServer:
    """Federated Learning server for TPD+FL.

    Manages the global model state (LoRA adapters) and orchestrates
    the training rounds across participating clients.

    Parameters
    ----------
    model : nn.Module
        The global diffusion LLM.  LoRA adapters will be attached
        if not already present.
    config : FLServerConfig
        Server configuration.
    lora_config : Optional[LoRAConfig]
        LoRA configuration for adapter injection.  If ``None``, a
        default ``LoRAConfig()`` is used.
    lora_modules : Optional[Dict[str, LoRALinear]]
        Pre-attached LoRA modules.  If ``None``, the server will
        attach them using ``lora_config``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[FLServerConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        lora_modules: Optional[Dict[str, LoRALinear]] = None,
    ) -> None:
        self.model = model
        self.config = config or FLServerConfig()
        self.lora_config = lora_config or LoRAConfig()

        # Attach LoRA if not provided
        if lora_modules is not None:
            self.lora_modules = lora_modules
        else:
            self.lora_modules = attach_lora(model, self.lora_config)

        # Global adapter state
        self._global_state = get_lora_state_dict(self.lora_modules)

        # Server-side optimizer state (for FedAdam)
        self._fedadam_state: Optional[FedAdamServerState] = None

        # Training history
        self._history: List[Dict[str, Any]] = []

        # Random generator for client sampling
        self._rng = torch.Generator()
        if self.config.seed is not None:
            self._rng.manual_seed(self.config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def global_state(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the current global adapter state dict."""
        return {k: v.clone() for k, v in self._global_state.items()}

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return the training history (list of per-round metric dicts)."""
        return list(self._history)

    def broadcast(
        self,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the global adapter state for distribution to clients.

        In a real distributed setting this would send the state over
        the network.  Here it returns a copy.

        Parameters
        ----------
        global_state : Optional[Dict[str, Tensor]]
            If provided, broadcast this state instead of the server's
            internal state.  Useful for external orchestration.

        Returns
        -------
        Dict[str, Tensor]
            Copy of the global state dict.
        """
        state = global_state if global_state is not None else self._global_state
        return {k: v.clone() for k, v in state.items()}

    def aggregate_round(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate a set of client deltas into a global update.

        Dispatches to the configured aggregation strategy.

        Parameters
        ----------
        client_deltas : List[Dict[str, Tensor]]
            Per-client adapter deltas from local training.
        client_weights : Optional[List[float]]
            Per-client weights (e.g., dataset sizes).

        Returns
        -------
        Dict[str, Tensor]
            The new global adapter state dict (not a delta, but the
            full updated state).
        """
        if len(client_deltas) < self.config.min_clients:
            logger.warning(
                "Received %d deltas, need at least %d. Skipping round.",
                len(client_deltas),
                self.config.min_clients,
            )
            return self.global_state

        strategy = self.config.strategy.lower()

        if strategy == "fedavg":
            aggregated_delta = fedavg(client_deltas, client_weights)

        elif strategy == "fedadam":
            aggregated_delta, self._fedadam_state = fedadam(
                client_deltas,
                server_state=self._fedadam_state,
                lr=self.config.fedadam_lr,
                betas=self.config.fedadam_betas,
                tau=self.config.fedadam_tau,
                client_weights=client_weights,
            )

        elif strategy == "secureagg":
            stub = SecureAggStub(
                num_clients=len(client_deltas),
                threshold=self.config.min_clients,
            )
            for i, delta in enumerate(client_deltas):
                stub.submit(client_id=i, delta=delta)
            weight_dict = None
            if client_weights is not None:
                weight_dict = {i: w for i, w in enumerate(client_weights)}
            aggregated_delta = stub.aggregate(client_weights=weight_dict)

        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

        # Apply delta to global state
        new_state: Dict[str, torch.Tensor] = {}
        for key in self._global_state:
            if key in aggregated_delta:
                new_state[key] = self._global_state[key] + aggregated_delta[key]
            else:
                new_state[key] = self._global_state[key].clone()

        self._global_state = new_state

        # Update the model's LoRA modules with the new global state
        load_lora_state_dict(self.lora_modules, self._global_state)

        return self.global_state

    def run(
        self,
        clients: List[Any],
        datasets: List[List[Dict[str, Any]]],
        client_weights: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the full federated training loop.

        This is the main entry point for simulation.  For each round:

        1. Sample a subset of clients.
        2. Broadcast the global state.
        3. Each sampled client trains locally.
        4. Aggregate client deltas.
        5. Record metrics.

        Parameters
        ----------
        clients : List[FLClient]
            List of FL clients.  Each must have a ``.train(dataset)``
            method that returns adapter deltas, and a ``.set_state(sd)``
            method for receiving broadcast state.
        datasets : List[List[Dict]]
            Per-client datasets.  ``datasets[i]`` is the training data
            for ``clients[i]``.
        client_weights : Optional[List[float]]
            Static weights for each client (e.g., dataset sizes).  If
            ``None``, weights are inferred from dataset sizes.

        Returns
        -------
        List[Dict[str, Any]]
            Training history: one dict per round with metrics.
        """
        num_clients = len(clients)
        if len(datasets) != num_clients:
            raise ValueError(
                f"Number of datasets ({len(datasets)}) must match "
                f"number of clients ({num_clients})"
            )

        # Infer weights from dataset sizes if not provided
        if client_weights is None:
            total_samples = sum(len(ds) for ds in datasets)
            if total_samples > 0:
                client_weights = [len(ds) / total_samples for ds in datasets]
            else:
                client_weights = [1.0 / num_clients] * num_clients

        # Number of clients to sample per round
        num_sample = max(
            self.config.min_clients,
            int(self.config.client_fraction * num_clients),
        )
        num_sample = min(num_sample, num_clients)

        self._history = []

        for round_idx in range(self.config.num_rounds):
            round_start = time.time()

            # Sample clients
            perm = torch.randperm(num_clients, generator=self._rng)
            sampled_indices = perm[:num_sample].tolist()

            # Broadcast global state to sampled clients
            global_state = self.broadcast()
            for idx in sampled_indices:
                clients[idx].set_state(global_state)

            # Local training
            round_deltas: List[Dict[str, torch.Tensor]] = []
            round_weights: List[float] = []

            for idx in sampled_indices:
                delta = clients[idx].train(datasets[idx])
                round_deltas.append(delta)
                round_weights.append(client_weights[idx])

            # Aggregate
            new_state = self.aggregate_round(round_deltas, round_weights)

            round_elapsed = time.time() - round_start

            # Compute metrics
            delta_norms = []
            for delta in round_deltas:
                norm = sum(
                    v.float().norm().item() ** 2 for v in delta.values()
                ) ** 0.5
                delta_norms.append(norm)

            round_metrics = {
                "round": round_idx,
                "num_clients_sampled": len(sampled_indices),
                "client_indices": sampled_indices,
                "elapsed_sec": round_elapsed,
                "mean_delta_norm": (
                    sum(delta_norms) / len(delta_norms)
                    if delta_norms
                    else 0.0
                ),
                "max_delta_norm": max(delta_norms) if delta_norms else 0.0,
                "min_delta_norm": min(delta_norms) if delta_norms else 0.0,
            }

            self._history.append(round_metrics)

            if round_idx % self.config.log_every == 0:
                logger.info(
                    "Round %d/%d | clients=%d | mean_delta_norm=%.6f | "
                    "elapsed=%.2fs",
                    round_idx + 1,
                    self.config.num_rounds,
                    len(sampled_indices),
                    round_metrics["mean_delta_norm"],
                    round_elapsed,
                )

        return self._history

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_global_state(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Manually set the global adapter state.

        Parameters
        ----------
        state_dict : Dict[str, Tensor]
            New global state dict.
        """
        self._global_state = {k: v.clone() for k, v in state_dict.items()}
        load_lora_state_dict(self.lora_modules, self._global_state)

    def reset(self) -> None:
        """Reset the server state (optimizer and history)."""
        self._fedadam_state = None
        self._history = []
        # Reset LoRA to zero
        for key in self._global_state:
            self._global_state[key].zero_()
        load_lora_state_dict(self.lora_modules, self._global_state)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    """CLI entry point for FL server simulation.

    Usage::

        python -m tpd_fl.fl.server --config configs/fl/fedavg.yaml
    """
    import argparse
    import json
    from pathlib import Path

    import yaml

    from tpd_fl.fl.client import FLClient, FLClientConfig
    from tpd_fl.fl.datasets import SyntheticPIIDataset, partition_non_iid_domain

    parser = argparse.ArgumentParser(description="TPD+FL Server Simulation")
    parser.add_argument("--config", type=str, default=None, help="YAML config")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    seed = cfg.get("seed", args.seed)
    torch.manual_seed(seed)

    num_clients = cfg.get("num_clients", 5)
    num_rounds = cfg.get("num_rounds", 10)
    strategy = cfg.get("strategy", "fedavg")
    lora_rank = cfg.get("lora_rank", 8)
    lora_alpha = cfg.get("lora_alpha", 16.0)
    local_epochs = cfg.get("local_epochs", 3)
    lr = cfg.get("learning_rate", 0.001)
    batch_size = cfg.get("batch_size", 4)
    output_dir = args.output_dir or cfg.get("output_dir", "runs/fl")

    # Build a simple model for simulation
    # Use small vocab for CLI demo (char-level data fits in 256)
    vocab_size = min(cfg.get("vocab_size", 32000), 512)
    embed_dim = 32

    class SimpleLM(nn.Module):
        """Minimal LM for FL simulation."""
        def __init__(self):
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

    model = SimpleLM()

    # LoRA config — target all nn.Linear layers in the model
    lora_config = LoRAConfig(rank=lora_rank, alpha=lora_alpha, target_modules=["fc1", "fc2"])

    # Create server config
    server_config = FLServerConfig()
    server_config.num_rounds = num_rounds
    server_config.min_clients = min(num_clients, 3)
    server_config.strategy = strategy
    server_config.seed = seed

    server = FLServer(model, server_config, lora_config=lora_config)

    # Create synthetic dataset and generate samples
    dataset_gen = SyntheticPIIDataset()
    dataset_gen.num_samples = num_clients * 10
    dataset_gen.seed = seed
    dataset_gen.max_length = 64
    samples = dataset_gen.generate()

    partitions = partition_non_iid_domain(
        samples,
        num_clients=num_clients,
        domain_skew=cfg.get("domain_skew", 0.8),
        seed=seed,
    )

    # Create clients
    client_config = FLClientConfig()
    client_config.local_epochs = local_epochs
    client_config.lr = lr
    client_config.batch_size = batch_size

    # Build clients from fresh model copies (before LoRA on server model)
    clients = []
    for i in range(num_clients):
        client_model = SimpleLM()
        client_lora = attach_lora(client_model, lora_config)
        client = FLClient(
            model=client_model,
            tokenizer=None,
            lora_modules=client_lora,
            config=client_config,
        )
        clients.append(client)

    # Run FL
    print(f"Starting FL simulation: {num_rounds} rounds, {num_clients} clients, {strategy}")
    history = server.run(clients, partitions)

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\nFL simulation complete. {len(history)} rounds logged.")
    print(f"Artifacts saved to {output_dir}")

    # Print summary
    if history:
        last = history[-1]
        print(f"Last round metrics: {json.dumps(last, indent=2, default=str)}")


if __name__ == "__main__":
    main()
