"""
TPD+FL Evaluation Suite.

Provides comprehensive evaluation tools for the Typed Privacy Diffusion +
Federated Learning system:

- **Leakage metrics**: regex-based and semantic leakage detection.
- **Utility metrics**: exact match, ROUGE, fluency on public content.
- **Speed metrics**: per-step timing and throughput tracking.
- **Baselines**: B0 (unprotected) through B7 (full TPD+FL typed).
- **Plots**: matplotlib-based publication-ready figures.
- **Runner**: CLI-driven evaluation pipeline.
"""

from tpd_fl.eval.leakage import (
    regex_leakage_count,
    regex_leakage_rate,
    quasi_identifier_check,
    LeakageEvaluator,
    STANDARD_PATTERNS,
)
from tpd_fl.eval.utility import (
    exact_match_public,
    rouge_public,
    fluency_metrics,
    UtilityEvaluator,
)
from tpd_fl.eval.speed import SpeedTracker
from tpd_fl.eval.baselines import (
    BaselineConfig,
    BaselineRunner,
    B0_unprotected,
    B1_posthoc_redaction,
    B2_ar_logit_mask,
    B3_tpd_projection_only,
    B4_tpd_projection_schedule,
    B5_tpd_schedule_repair,
    B6_tpd_fl,
    B7_tpd_fl_typed,
)
from tpd_fl.eval.plots import (
    plot_leakage_bar,
    plot_utility_vs_leakage,
    plot_z_distribution,
    plot_runtime_vs_utility,
    plot_fl_convergence,
)


def __getattr__(name):
    """Lazy import for run_eval to avoid -m runpy warning."""
    if name in ("EvalConfig", "EvalRunner", "load_eval_config"):
        import tpd_fl.eval.run_eval as _run_eval
        return getattr(_run_eval, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "regex_leakage_count", "regex_leakage_rate", "quasi_identifier_check",
    "LeakageEvaluator", "STANDARD_PATTERNS",
    "exact_match_public", "rouge_public", "fluency_metrics", "UtilityEvaluator",
    "SpeedTracker",
    "BaselineConfig", "BaselineRunner",
    "B0_unprotected", "B1_posthoc_redaction", "B2_ar_logit_mask",
    "B3_tpd_projection_only", "B4_tpd_projection_schedule",
    "B5_tpd_schedule_repair", "B6_tpd_fl", "B7_tpd_fl_typed",
    "plot_leakage_bar", "plot_utility_vs_leakage", "plot_z_distribution",
    "plot_runtime_vs_utility", "plot_fl_convergence",
    "EvalConfig", "EvalRunner", "load_eval_config",
]
