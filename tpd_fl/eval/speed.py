"""
Speed / Performance Metrics — wall-clock timing and throughput tracking.

Provides the :class:`SpeedTracker` class for instrumenting decode runs.
The tracker records per-step timing and token counts, then produces
aggregate statistics (total time, throughput, average tokens per step).

Usage::

    tracker = SpeedTracker()
    tracker.start_run()
    for step in range(T):
        # ... decode step ...
        tracker.record_step(step, tokens_updated=k)
    tracker.end_run()
    print(tracker.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    """Timing record for a single decode step."""
    step: int
    tokens_updated: int
    wall_time_sec: float
    cumulative_tokens: int
    cumulative_time_sec: float


class SpeedTracker:
    """Performance tracker for diffusion decode runs.

    Records per-step timing and token-update counts.  Produces a
    summary dictionary with aggregate throughput statistics.

    The tracker supports two usage patterns:

    1. **Manual timing** — call :meth:`start_run`, then
       :meth:`record_step` after each step, then :meth:`end_run`.
       Each ``record_step`` call records the wall-clock delta since
       the previous call.

    2. **External timing** — call ``record_step`` with an explicit
       ``elapsed`` parameter if you manage timing yourself.

    Example
    -------
    ::

        tracker = SpeedTracker()
        tracker.start_run()
        for step in range(total_steps):
            # ... do work ...
            tracker.record_step(step, tokens_updated=num_tokens)
        tracker.end_run()
        stats = tracker.summary()
    """

    def __init__(self) -> None:
        self._steps: List[StepRecord] = []
        self._run_start: Optional[float] = None
        self._run_end: Optional[float] = None
        self._last_step_time: Optional[float] = None
        self._cumulative_tokens: int = 0
        self._cumulative_time: float = 0.0

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """Mark the beginning of a decode run."""
        self._steps = []
        self._cumulative_tokens = 0
        self._cumulative_time = 0.0
        self._run_start = time.perf_counter()
        self._last_step_time = self._run_start
        self._run_end = None

    def end_run(self) -> None:
        """Mark the end of a decode run."""
        self._run_end = time.perf_counter()

    # ------------------------------------------------------------------
    # Step recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        tokens_updated: int,
        elapsed: Optional[float] = None,
    ) -> None:
        """Record metrics for a single decode step.

        Parameters
        ----------
        step : int
            The step index.
        tokens_updated : int
            Number of tokens updated at this step.
        elapsed : float, optional
            Wall-clock time for this step in seconds.  If not provided,
            the tracker computes the delta since the previous
            ``record_step`` (or ``start_run``).
        """
        now = time.perf_counter()
        if elapsed is not None:
            wall = elapsed
        elif self._last_step_time is not None:
            wall = now - self._last_step_time
        else:
            wall = 0.0

        self._last_step_time = now
        self._cumulative_tokens += tokens_updated
        self._cumulative_time += wall

        self._steps.append(StepRecord(
            step=step,
            tokens_updated=tokens_updated,
            wall_time_sec=wall,
            cumulative_tokens=self._cumulative_tokens,
            cumulative_time_sec=self._cumulative_time,
        ))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Produce aggregate performance statistics.

        Returns
        -------
        dict with keys:
            total_time : float
                Total wall-clock time from start_run to end_run (seconds).
                Falls back to cumulative step time if end_run was not called.
            steps : int
                Number of recorded steps.
            total_tokens_updated : int
                Sum of tokens updated across all steps.
            throughput : float
                Tokens per second (total_tokens / total_time).
            avg_tokens_per_step : float
                Mean tokens updated per step.
            avg_step_time : float
                Mean wall-clock time per step (seconds).
            min_step_time : float
                Fastest step time.
            max_step_time : float
                Slowest step time.
            step_times : list of float
                Per-step wall-clock times.
        """
        n_steps = len(self._steps)
        total_tokens = self._cumulative_tokens

        # Total time
        if self._run_start is not None and self._run_end is not None:
            total_time = self._run_end - self._run_start
        else:
            total_time = self._cumulative_time

        # Per-step times
        step_times = [s.wall_time_sec for s in self._steps]

        throughput = total_tokens / total_time if total_time > 0 else 0.0
        avg_tokens = total_tokens / n_steps if n_steps > 0 else 0.0
        avg_time = total_time / n_steps if n_steps > 0 else 0.0
        min_time = min(step_times) if step_times else 0.0
        max_time = max(step_times) if step_times else 0.0

        return {
            "total_time": total_time,
            "steps": n_steps,
            "total_tokens_updated": total_tokens,
            "throughput": throughput,
            "avg_tokens_per_step": avg_tokens,
            "avg_step_time": avg_time,
            "min_step_time": min_time,
            "max_step_time": max_time,
            "step_times": step_times,
        }

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    @property
    def steps(self) -> List[StepRecord]:
        """Return the list of recorded step records."""
        return list(self._steps)

    @property
    def total_time(self) -> float:
        """Total wall-clock time (start_run to end_run or cumulative)."""
        if self._run_start is not None and self._run_end is not None:
            return self._run_end - self._run_start
        return self._cumulative_time

    @property
    def total_tokens(self) -> int:
        """Total tokens updated across all steps."""
        return self._cumulative_tokens

    def reset(self) -> None:
        """Reset the tracker for a new run."""
        self._steps = []
        self._run_start = None
        self._run_end = None
        self._last_step_time = None
        self._cumulative_tokens = 0
        self._cumulative_time = 0.0
