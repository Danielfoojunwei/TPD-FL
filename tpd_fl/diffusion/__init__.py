"""
Diffusion LLM module — model abstraction and decode loop.

Supports:
  Backend A: Synthetic / lightweight diffusion text model (for testing).
  Backend B: HuggingFace-compatible diffusion LLM wrapper.
  Backend C: LLaDA/LLaDA2.x hooks (when available).

Note: decode_loop is imported lazily to avoid circular import issues
when running as ``python -m tpd_fl.diffusion.decode_loop``.
"""

from tpd_fl.diffusion.model_adapter import DiffusionModel, SyntheticDiffusionModel


def __getattr__(name):
    if name == "DiffusionDecodeLoop":
        from tpd_fl.diffusion.decode_loop import DiffusionDecodeLoop
        return DiffusionDecodeLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DiffusionModel", "SyntheticDiffusionModel", "DiffusionDecodeLoop"]
