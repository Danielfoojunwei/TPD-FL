"""
Federated Learning (FL) Module for TPD+FL.

Implements privacy-preserving federated learning with LoRA adapters
for diffusion language models.  The module provides:

- **LoRA**: Low-Rank Adaptation layers for parameter-efficient fine-tuning.
- **Step Adapters**: Per-diffusion-step adapter banks for phase-specialised
  adaptation.
- **FL Client**: Local training loop with diffusion denoising loss and
  optional typed privacy training.
- **FL Server**: Aggregation and round management for federated training.
- **Protocols**: Aggregation strategies (FedAvg, FedAdam, Secure Aggregation).
- **Datasets**: Synthetic PII dataset generation and non-IID partitioning.

Key property: FL adapter updates cannot violate TPD's non-emission
guarantee because projection is applied at decode time, independently
of model parameters.
"""

from tpd_fl.fl.lora import (
    LoRAConfig,
    LoRALinear,
    attach_lora,
    detach_lora,
    get_lora_state_dict,
    load_lora_state_dict,
    merge_lora,
    unmerge_lora,
)
from tpd_fl.fl.step_adapters import (
    StepAdapterConfig,
    StepAdapterBank,
)
from tpd_fl.fl.client import (
    FLClientConfig,
    FLClient,
)
from tpd_fl.fl.protocols import (
    fedavg,
    fedadam,
    FedAdamServerState,
    SecureAggStub,
)
from tpd_fl.fl.datasets import (
    SyntheticPIIDataset,
    partition_iid,
    partition_non_iid_domain,
    partition_non_iid_sensitivity,
)


def __getattr__(name):
    """Lazy import for server to avoid -m runpy warning."""
    if name == "FLServer":
        from tpd_fl.fl.server import FLServer
        return FLServer
    if name == "FLServerConfig":
        from tpd_fl.fl.server import FLServerConfig
        return FLServerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LoRAConfig", "LoRALinear", "attach_lora", "detach_lora",
    "get_lora_state_dict", "load_lora_state_dict", "merge_lora", "unmerge_lora",
    "StepAdapterConfig", "StepAdapterBank",
    "FLClientConfig", "FLClient",
    "FLServerConfig", "FLServer",
    "fedavg", "fedadam", "FedAdamServerState", "SecureAggStub",
    "SyntheticPIIDataset", "partition_iid",
    "partition_non_iid_domain", "partition_non_iid_sensitivity",
]
