"""
Plotting utilities for TPD+FL evaluation results.

All plots use **matplotlib only** (no seaborn) and save to the provided
file path.  The style is clean and publication-ready: white background,
labelled axes, legends where appropriate, and minimal chart junk.

Functions:

  - :func:`plot_leakage_bar` — bar chart of leakage rate across baselines.
  - :func:`plot_utility_vs_leakage` — scatter of utility vs. leakage.
  - :func:`plot_z_distribution` — histogram of Z_i allowed-mass values.
  - :func:`plot_runtime_vs_utility` — scatter of runtime vs. utility.
  - :func:`plot_fl_convergence` — FL training convergence curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

_BASELINE_COLOURS = {
    "B0": "#e74c3c",   # red — unprotected
    "B1": "#e67e22",   # orange
    "B2": "#f1c40f",   # yellow
    "B3": "#2ecc71",   # green
    "B4": "#1abc9c",   # teal
    "B5": "#3498db",   # blue
    "B6": "#9b59b6",   # purple
    "B7": "#2c3e50",   # dark blue-grey — full system
}

_BASELINE_LABELS = {
    "B0": "B0: Unprotected",
    "B1": "B1: Post-hoc redact",
    "B2": "B2: AR logit mask",
    "B3": "B3: TPD proj.",
    "B4": "B4: TPD proj.+sched.",
    "B5": "B5: TPD full",
    "B6": "B6: TPD+FL",
    "B7": "B7: TPD+FL typed",
}


def _apply_clean_style(ax: plt.Axes) -> None:
    """Apply a clean, publication-ready style to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)


def _ensure_parent_dir(path: str) -> None:
    """Create parent directories for the save path if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _get_colour(name: str) -> str:
    """Return colour for a baseline name, with fallback."""
    return _BASELINE_COLOURS.get(name, "#7f8c8d")


def _get_label(name: str) -> str:
    """Return display label for a baseline name, with fallback."""
    return _BASELINE_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# Plot 1: Leakage bar chart
# ---------------------------------------------------------------------------

def plot_leakage_bar(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
    metric_key: str = "hard_leakage_rate",
    title: str = "Hard Leakage Rate by Baseline",
    figsize: Tuple[float, float] = (10, 5),
    dpi: int = 150,
) -> None:
    """Bar chart of leakage rate across baselines.

    Parameters
    ----------
    results : dict
        Mapping from baseline name (e.g., "B0") to a result dict.
        Each result must contain a ``leakage`` sub-dict with the
        ``metric_key`` field.
    save_path : str
        File path for the saved figure (e.g., "plots/leakage.png").
    metric_key : str
        Which leakage metric to plot (default ``hard_leakage_rate``).
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution.
    """
    _ensure_parent_dir(save_path)

    names = sorted(results.keys())
    values = []
    colours = []
    labels = []
    for name in names:
        r = results[name]
        leak = r.get("leakage", {})
        values.append(leak.get(metric_key, 0.0))
        colours.append(_get_colour(name))
        labels.append(_get_label(name))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars = ax.bar(range(len(names)), values, color=colours, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Leakage Rate", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylim(0, max(max(values) * 1.15, 0.05) if values else 1.0)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Utility vs. Leakage scatter
# ---------------------------------------------------------------------------

def plot_utility_vs_leakage(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
    utility_key: str = "exact_match_public",
    leakage_key: str = "hard_leakage_rate",
    title: str = "Utility vs. Leakage Trade-off",
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> None:
    """Scatter plot of utility (y) vs. leakage (x) per baseline.

    The ideal system is in the top-left corner (high utility, low leakage).

    Parameters
    ----------
    results : dict
        Baseline name -> result dict with ``utility`` and ``leakage`` sub-dicts.
    save_path : str
    utility_key : str
        Key in the ``utility`` sub-dict (default ``exact_match_public``).
    leakage_key : str
        Key in the ``leakage`` sub-dict (default ``hard_leakage_rate``).
    title, figsize, dpi : plot parameters.
    """
    _ensure_parent_dir(save_path)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for name in sorted(results.keys()):
        r = results[name]
        leak_val = r.get("leakage", {}).get(leakage_key, 0.0)
        util_val = r.get("utility", {}).get(utility_key, 0.0)
        colour = _get_colour(name)
        label = _get_label(name)

        ax.scatter(
            leak_val, util_val,
            c=colour, s=120, edgecolors="white", linewidths=1.0,
            zorder=3, label=label,
        )
        # Annotate point
        ax.annotate(
            name,
            (leak_val, util_val),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color=colour,
        )

    ax.set_xlabel("Leakage Rate", fontsize=11)
    ax.set_ylabel("Utility (Exact Match Public)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)

    # Draw ideal region indicator
    ax.axvline(x=0, color="#cccccc", linewidth=0.5, linestyle="--", zorder=1)
    ax.axhline(y=1, color="#cccccc", linewidth=0.5, linestyle="--", zorder=1)

    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Z_i distribution histogram
# ---------------------------------------------------------------------------

def plot_z_distribution(
    z_values: Union[List[float], "torch.Tensor"],
    save_path: str,
    title: str = "Distribution of Allowed Mass $Z_i$",
    bins: int = 50,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> None:
    """Histogram of Z_i values (allowed mass at sensitive positions).

    Parameters
    ----------
    z_values : list of float or Tensor
        The Z_i values for sensitive positions.
    save_path : str
    title : str
    bins : int
        Number of histogram bins.
    figsize, dpi : plot parameters.
    """
    _ensure_parent_dir(save_path)

    # Convert tensor to list if needed
    if hasattr(z_values, "tolist"):
        vals = z_values.tolist()
    else:
        vals = list(z_values)

    if not vals:
        # Nothing to plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No Z_i values to plot", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999999")
        ax.set_title(title, fontsize=13, pad=12)
        _apply_clean_style(ax)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.hist(
        vals, bins=bins, range=(0, 1),
        color="#3498db", edgecolor="white", linewidth=0.5,
        alpha=0.85, density=True,
    )

    # Summary statistics
    n = len(vals)
    mean_z = sum(vals) / n
    sorted_vals = sorted(vals)
    median_z = sorted_vals[n // 2]
    below_01 = sum(1 for v in vals if v < 0.01)
    below_05 = sum(1 for v in vals if v < 0.05)

    stats_text = (
        f"n = {n}\n"
        f"mean = {mean_z:.4f}\n"
        f"median = {median_z:.4f}\n"
        f"Z < 0.01: {below_01} ({100*below_01/n:.1f}%)\n"
        f"Z < 0.05: {below_05} ({100*below_05/n:.1f}%)"
    )
    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    # Vertical lines for thresholds
    ax.axvline(x=0.01, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.7, label="Z=0.01")
    ax.axvline(x=0.05, color="#e67e22", linewidth=1, linestyle="--", alpha=0.7, label="Z=0.05")
    ax.axvline(x=mean_z, color="#2c3e50", linewidth=1.5, linestyle="-", alpha=0.8, label=f"mean={mean_z:.3f}")

    ax.set_xlabel("$Z_i$ (Allowed Mass)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)

    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Runtime vs. Utility scatter
# ---------------------------------------------------------------------------

def plot_runtime_vs_utility(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
    utility_key: str = "exact_match_public",
    title: str = "Runtime vs. Utility",
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> None:
    """Scatter plot of runtime (x) vs. utility (y) per baseline.

    Parameters
    ----------
    results : dict
        Baseline name -> result dict.  The ``metrics`` sub-dict must
        contain ``elapsed_sec``.  The ``utility`` sub-dict must contain
        the ``utility_key`` metric.
    save_path : str
    utility_key : str
    title, figsize, dpi : plot parameters.
    """
    _ensure_parent_dir(save_path)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for name in sorted(results.keys()):
        r = results[name]
        runtime = r.get("metrics", {}).get("elapsed_sec", 0.0)
        util_val = r.get("utility", {}).get(utility_key, 0.0)
        colour = _get_colour(name)
        label = _get_label(name)

        ax.scatter(
            runtime, util_val,
            c=colour, s=120, edgecolors="white", linewidths=1.0,
            zorder=3, label=label,
        )
        ax.annotate(
            name,
            (runtime, util_val),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color=colour,
        )

    ax.set_xlabel("Runtime (seconds)", fontsize=11)
    ax.set_ylabel("Utility (Exact Match Public)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)

    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: FL convergence
# ---------------------------------------------------------------------------

def plot_fl_convergence(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Federated Learning Convergence",
    figsize: Tuple[float, float] = (9, 5),
    dpi: int = 150,
) -> None:
    """Line plot of FL training convergence metrics over rounds.

    Parameters
    ----------
    history : dict
        Maps metric name (e.g., "loss", "leakage_rate", "utility") to
        a list of per-round values.  The x-axis is the round index.
    save_path : str
    title, figsize, dpi : plot parameters.
    """
    _ensure_parent_dir(save_path)

    if not history:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No convergence data to plot", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999999")
        ax.set_title(title, fontsize=13, pad=12)
        _apply_clean_style(ax)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return

    # Determine number of metrics for subplot layout
    metric_names = sorted(history.keys())
    n_metrics = len(metric_names)

    if n_metrics <= 1:
        fig, axes_arr = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        axes_list = [axes_arr]
    elif n_metrics <= 3:
        fig, axes_arr = plt.subplots(1, n_metrics, figsize=(figsize[0], figsize[1]), dpi=dpi)
        axes_list = list(axes_arr) if n_metrics > 1 else [axes_arr]
    else:
        nrows = (n_metrics + 1) // 2
        fig, axes_arr = plt.subplots(nrows, 2, figsize=(figsize[0], figsize[1] * nrows / 2), dpi=dpi)
        axes_list = list(axes_arr.flat)

    # Colour palette for different metrics
    line_colours = ["#2c3e50", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22"]

    for idx, metric_name in enumerate(metric_names):
        if idx >= len(axes_list):
            break
        ax = axes_list[idx]
        vals = history[metric_name]
        rounds = list(range(1, len(vals) + 1))
        colour = line_colours[idx % len(line_colours)]

        ax.plot(rounds, vals, color=colour, linewidth=1.5, marker="o", markersize=3)
        ax.set_xlabel("Round", fontsize=10)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=10)
        ax.set_title(metric_name.replace("_", " ").title(), fontsize=11)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        _apply_clean_style(ax)

    # Hide unused axes
    for idx in range(n_metrics, len(axes_list)):
        axes_list[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
