"""
TPD (Typed Privacy Diffusion) Core Module.

Implements typed operational semantics over diffusion decoding:
- Span typing τ over output positions
- Allowed token sets A(type)
- Policy-driven mask schedule (draft/safe/reveal phases)
- Logits projection (support restriction)
- Verifier gate Okπ
- Repair logic
- Diagnostics (Z_i measurement)
"""

from tpd_fl.tpd.typing import SpanType, Span, SpanTyper
from tpd_fl.tpd.allowed_sets import AllowedSetBuilder
from tpd_fl.tpd.schedule import SchedulePhase, MaskSchedule
from tpd_fl.tpd.projection import project_logits, ProjectionEngine
from tpd_fl.tpd.verifier import Verifier
from tpd_fl.tpd.repair import RepairEngine
from tpd_fl.tpd.diagnostics import DiagnosticsLogger

__all__ = [
    "SpanType", "Span", "SpanTyper",
    "AllowedSetBuilder",
    "SchedulePhase", "MaskSchedule",
    "project_logits", "ProjectionEngine",
    "Verifier",
    "RepairEngine",
    "DiagnosticsLogger",
]
