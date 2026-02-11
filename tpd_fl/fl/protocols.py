"""
Federated Aggregation Protocols for TPD+FL.

Implements aggregation strategies that combine LoRA adapter deltas
from multiple FL clients into a single global update:

- **FedAvg** (McMahan et al., 2017): weighted average of client
  deltas, where weights are typically proportional to local dataset
  size.

- **FedAdam** (Reddi et al., 2021): server-side adaptive optimisation
  that applies Adam-style momentum and variance tracking to the
  aggregated pseudo-gradient.

- **SecureAggStub**: placeholder for secure aggregation protocols
  (e.g., Bonawitz et al., 2017).  In production this would integrate
  with an MPC or HE backend; here it validates the interface.

All functions operate on state dicts (``Dict[str, Tensor]``) so they
are agnostic to the LoRA layer structure.

Usage::

    from tpd_fl.fl.protocols import fedavg, fedadam

    global_delta = fedavg(client_deltas, client_weights)
    global_delta, new_state = fedadam(
        client_deltas, server_state, lr=1e-3
    )

This module depends on torch and the Python standard library only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def fedavg(
    client_deltas: List[Dict[str, torch.Tensor]],
    client_weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Federated Averaging: weighted mean of client adapter deltas.

    Parameters
    ----------
    client_deltas : List[Dict[str, Tensor]]
        Per-client adapter deltas.  Each dict maps parameter names
        (e.g. ``"layer.q_proj.lora_A"``) to delta tensors.
    client_weights : Optional[List[float]]
        Non-negative weights for each client.  Typically proportional
        to local dataset sizes.  If ``None``, uniform weighting is
        used.

    Returns
    -------
    Dict[str, Tensor]
        Aggregated delta: the weighted average across clients.

    Raises
    ------
    ValueError
        If ``client_deltas`` is empty or weights have wrong length.
    """
    if not client_deltas:
        raise ValueError("client_deltas must be non-empty")

    num_clients = len(client_deltas)

    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        if len(client_weights) != num_clients:
            raise ValueError(
                f"client_weights length ({len(client_weights)}) must match "
                f"number of clients ({num_clients})"
            )
        # Normalise weights to sum to 1
        total_weight = sum(client_weights)
        if total_weight <= 0:
            raise ValueError("Total client weight must be positive")
        client_weights = [w / total_weight for w in client_weights]

    # Aggregate
    aggregated: Dict[str, torch.Tensor] = {}
    keys = client_deltas[0].keys()

    for key in keys:
        weighted_sum = torch.zeros_like(client_deltas[0][key])
        for i, delta in enumerate(client_deltas):
            if key not in delta:
                raise KeyError(
                    f"Key '{key}' missing from client {i} delta"
                )
            weighted_sum.add_(delta[key], alpha=client_weights[i])
        aggregated[key] = weighted_sum

    return aggregated


# ---------------------------------------------------------------------------
# FedAdam
# ---------------------------------------------------------------------------

@dataclass
class FedAdamServerState:
    """Server-side optimizer state for FedAdam.

    Maintains first and second moment estimates (like Adam) across
    FL rounds, enabling adaptive server-side learning.

    Attributes
    ----------
    m : Dict[str, Tensor]
        First moment (mean of gradients).
    v : Dict[str, Tensor]
        Second moment (mean of squared gradients).
    round_num : int
        Current FL round number, used for bias correction.
    """

    m: Dict[str, torch.Tensor] = field(default_factory=dict)
    v: Dict[str, torch.Tensor] = field(default_factory=dict)
    round_num: int = 0


def fedadam(
    client_deltas: List[Dict[str, torch.Tensor]],
    server_state: Optional[FedAdamServerState] = None,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.99),
    tau: float = 1e-3,
    client_weights: Optional[List[float]] = None,
) -> Tuple[Dict[str, torch.Tensor], FedAdamServerState]:
    """Federated Adam: server-side adaptive optimisation of aggregated deltas.

    Applies Adam-style momentum and variance tracking to the
    pseudo-gradient obtained by averaging client deltas.  This
    improves convergence in heterogeneous data settings common in
    privacy-sensitive FL deployments.

    Parameters
    ----------
    client_deltas : List[Dict[str, Tensor]]
        Per-client adapter deltas.
    server_state : Optional[FedAdamServerState]
        Server optimizer state from the previous round.  If ``None``,
        a fresh state is initialised.
    lr : float
        Server-side learning rate.
    betas : Tuple[float, float]
        Coefficients for first and second moment estimates.
        ``betas[0]`` controls momentum decay; ``betas[1]`` controls
        variance decay.
    tau : float
        Numerical stability constant added to the denominator (akin
        to Adam's epsilon but typically larger for FL).
    client_weights : Optional[List[float]]
        Client weighting for the initial averaging step.

    Returns
    -------
    Tuple[Dict[str, Tensor], FedAdamServerState]
        - Aggregated global update (to be applied to the global model).
        - Updated server optimizer state for the next round.

    Algorithm
    ---------
    1. Compute the pseudo-gradient: ``delta = FedAvg(client_deltas, weights)``
    2. Update moments:
       - ``m = beta1 * m + (1 - beta1) * delta``
       - ``v = beta2 * v + (1 - beta2) * delta^2``
    3. Bias-correct: ``m_hat``, ``v_hat``
    4. Global update: ``lr * m_hat / (sqrt(v_hat) + tau)``
    """
    # Step 1: compute pseudo-gradient via FedAvg
    pseudo_grad = fedavg(client_deltas, client_weights)

    # Initialise server state if needed
    if server_state is None:
        server_state = FedAdamServerState()

    beta1, beta2 = betas
    server_state.round_num += 1
    t = server_state.round_num

    aggregated: Dict[str, torch.Tensor] = {}

    for key, delta in pseudo_grad.items():
        # Initialise moment buffers on first round
        if key not in server_state.m:
            server_state.m[key] = torch.zeros_like(delta)
        if key not in server_state.v:
            server_state.v[key] = torch.zeros_like(delta)

        # Update biased first moment estimate
        server_state.m[key].mul_(beta1).add_(delta, alpha=1.0 - beta1)

        # Update biased second raw moment estimate
        server_state.v[key].mul_(beta2).addcmul_(
            delta, delta, value=1.0 - beta2
        )

        # Bias correction
        m_hat = server_state.m[key] / (1.0 - beta1 ** t)
        v_hat = server_state.v[key] / (1.0 - beta2 ** t)

        # Compute update
        aggregated[key] = lr * m_hat / (v_hat.sqrt() + tau)

    return aggregated, server_state


# ---------------------------------------------------------------------------
# Secure Aggregation Stub
# ---------------------------------------------------------------------------

class SecureAggStub:
    """Placeholder for secure aggregation protocols.

    In a production deployment, this class would implement or delegate
    to a secure multi-party computation (MPC) protocol such as:

    - **Bonawitz et al. (2017)**: practical secure aggregation for
      federated learning using secret sharing and key agreement.
    - **Homomorphic encryption**: Paillier or CKKS-based aggregation
      where individual client updates are never seen in the clear.

    This stub validates the interface and performs plain-text aggregation
    internally, serving as a drop-in replacement during development and
    testing.

    Usage::

        agg = SecureAggStub(num_clients=10)
        agg.submit(client_id=0, delta=delta_0)
        agg.submit(client_id=1, delta=delta_1)
        ...
        result = agg.aggregate()
    """

    def __init__(
        self,
        num_clients: int,
        threshold: Optional[int] = None,
    ) -> None:
        """Initialise the secure aggregation stub.

        Parameters
        ----------
        num_clients : int
            Expected number of participating clients.
        threshold : Optional[int]
            Minimum number of clients required to reconstruct the
            aggregate.  Defaults to ``num_clients`` (all must submit).
        """
        self.num_clients = num_clients
        self.threshold = threshold if threshold is not None else num_clients
        self._submissions: Dict[int, Dict[str, torch.Tensor]] = {}
        self._finalized = False

    @property
    def num_submissions(self) -> int:
        """Number of client submissions received so far."""
        return len(self._submissions)

    @property
    def ready(self) -> bool:
        """Whether enough submissions have been received to aggregate."""
        return self.num_submissions >= self.threshold

    def submit(
        self,
        client_id: int,
        delta: Dict[str, torch.Tensor],
    ) -> None:
        """Submit a client's adapter delta for aggregation.

        Parameters
        ----------
        client_id : int
            Unique client identifier.
        delta : Dict[str, Tensor]
            The client's adapter delta.

        Raises
        ------
        RuntimeError
            If aggregation has already been finalized or if the
            client has already submitted.
        """
        if self._finalized:
            raise RuntimeError("Aggregation already finalized")
        if client_id in self._submissions:
            raise RuntimeError(
                f"Client {client_id} has already submitted"
            )
        # In a real implementation, the delta would be encrypted or
        # secret-shared before storage
        self._submissions[client_id] = {
            k: v.clone() for k, v in delta.items()
        }

    def aggregate(
        self,
        client_weights: Optional[Dict[int, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Perform the aggregation and return the global delta.

        In a real secure aggregation protocol, this would reconstruct
        the sum from secret shares or decrypt homomorphically encrypted
        updates.  This stub performs plain-text weighted averaging.

        Parameters
        ----------
        client_weights : Optional[Dict[int, float]]
            Mapping from client_id to weight.  If ``None``, uniform
            weights are used.

        Returns
        -------
        Dict[str, Tensor]
            The aggregated global delta.

        Raises
        ------
        RuntimeError
            If insufficient submissions have been received.
        """
        if not self.ready:
            raise RuntimeError(
                f"Need at least {self.threshold} submissions, "
                f"have {self.num_submissions}"
            )

        deltas = list(self._submissions.values())
        ids = list(self._submissions.keys())

        if client_weights is not None:
            weights = [client_weights.get(cid, 1.0) for cid in ids]
        else:
            weights = None

        result = fedavg(deltas, weights)
        self._finalized = True
        return result

    def reset(self) -> None:
        """Reset the aggregator for a new round."""
        self._submissions.clear()
        self._finalized = False
