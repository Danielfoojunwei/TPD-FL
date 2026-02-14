"""
Model backends for TPD-FL.

Provides a unified DiffusionBackend interface with concrete implementations:
  - HFLLaDABackend:  LLaDA 8B on CPU (Tier 1, default)
  - HFLLaDA2Backend: LLaDA2.1-mini 16B MoE on GPU (Tier 2, optional)
"""
