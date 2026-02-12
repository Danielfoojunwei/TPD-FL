"""
Diffusion LLM module — decode loop with TPD hooks.

Imports are lazy to avoid issues with ``python -m`` execution.
"""


def __getattr__(name):
    if name == "DiffusionDecodeLoop":
        from tpd_fl.diffusion.decode_loop import DiffusionDecodeLoop
        return DiffusionDecodeLoop
    if name == "DecodeConfig":
        from tpd_fl.diffusion.decode_loop import DecodeConfig
        return DecodeConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DiffusionDecodeLoop", "DecodeConfig"]
