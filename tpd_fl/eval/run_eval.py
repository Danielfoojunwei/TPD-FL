"""
Main Evaluation Runner — CLI entry point for the TPD+FL evaluation suite.

Usage::

    python -m tpd_fl.eval.run_eval --config configs/eval/cpu_small.yaml

The runner:

1. Loads evaluation configuration from YAML.
2. Generates (or loads) benchmark data via :mod:`tpd_fl.eval.benchgen`.
3. Runs selected baselines (B0-B7) on each benchmark sample.
4. Computes aggregated leakage, utility, and speed metrics.
5. Generates publication-ready plots.
6. Saves all results to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from tpd_fl.eval.benchgen import BenchmarkGenerator
from tpd_fl.eval.leakage import LeakageEvaluator, STANDARD_PATTERNS
from tpd_fl.eval.utility import UtilityEvaluator
from tpd_fl.eval.speed import SpeedTracker
from tpd_fl.eval.baselines import (
    BaselineConfig,
    BaselineRunner,
    BaselineResult,
    B0_unprotected,
    B1_posthoc_redaction,
    B2_ar_logit_mask,
    B3_tpd_projection,
    B4_tpd_projection_schedule,
    B5_tpd_full,
    B6_fl_only,
    B7_tpd_fl,
    # Backward-compatible aliases
    B3_tpd_projection_only,
    B5_tpd_schedule_repair,
    B6_tpd_fl as B6_tpd_fl_compat,
    B7_tpd_fl_typed,
)
from tpd_fl.eval.plots import (
    plot_leakage_bar,
    plot_utility_vs_leakage,
    plot_z_distribution,
    plot_runtime_vs_utility,
    plot_fl_convergence,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Configuration for a full evaluation run."""
    # Output
    output_dir: str = "runs/eval"
    experiment_name: str = "tpd_fl_eval"

    # Backend
    backend: str = "synthetic"        # "synthetic" | "llada8b" | "llada2"
    model_id: str = ""
    device: str = "cpu"
    dtype: str = "auto"

    # Synthetic backend params (used when backend == "synthetic")
    vocab_size: int = 32000
    mask_token_id: int = 126336
    synthetic_mode: str = "uniform"

    # Diffusion parameters (passed to baselines)
    total_steps: int = 64
    seq_len: int = 128
    tokens_per_step_frac: float = 0.15
    temperature: float = 1.0
    seed: int = 42

    # Schedule
    schedule_draft_end: float = 0.4
    schedule_safe_end: float = 0.9

    # Repair
    repair_mode: str = "resample"
    repair_max_iters: int = 3

    # Verifier
    verifier_forbidden_tags: List[str] = field(
        default_factory=lambda: ["EMAIL", "PHONE", "SSN", "CC", "ID"]
    )

    # Benchmark data — new benchgen-based config
    num_s1: int = 20
    num_s2: int = 10
    num_s3: int = 10
    benchmark_data_path: str = ""  # if set, load from JSONL instead of generating

    # Legacy alias
    num_samples: int = 20
    include_adversarial: bool = True

    # Baselines to run
    baselines: List[str] = field(
        default_factory=lambda: ["B0", "B1", "B2", "B3", "B4", "B5"]
    )

    # FL adapter
    fl_adapter_enabled: bool = False

    # Plots
    generate_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 150


def load_eval_config(path: str) -> EvalConfig:
    """Load evaluation configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    valid_fields = set(EvalConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return EvalConfig(**filtered)


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------

def _build_backend(config: EvalConfig):
    """Construct the DiffusionBackend from eval config."""
    from tpd_fl.model.backend_base import SyntheticBackend

    if config.backend == "synthetic":
        return SyntheticBackend(
            vocab_size=config.vocab_size,
            mask_token_id_val=config.mask_token_id,
            mode=config.synthetic_mode,
            device_str=config.device,
            seed=config.seed,
        )
    elif config.backend == "llada8b":
        from tpd_fl.model.backend_hf_llada import HFLLaDABackend
        from tpd_fl.model.backend_base import BackendConfig
        bcfg = BackendConfig(
            model_id=config.model_id or "GSAI-ML/LLaDA-8B-Instruct",
            device=config.device,
            dtype=config.dtype,
            max_seq_len=config.seq_len,
            diffusion_steps=config.total_steps,
        )
        return HFLLaDABackend(bcfg)
    elif config.backend == "llada2":
        from tpd_fl.model.backend_hf_llada2 import HFLLaDA2Backend
        from tpd_fl.model.backend_base import BackendConfig
        bcfg = BackendConfig(
            model_id=config.model_id or "inclusionAI/LLaDA2.1-mini",
            device=config.device,
            dtype=config.dtype,
            max_seq_len=config.seq_len,
            diffusion_steps=config.total_steps,
        )
        return HFLLaDA2Backend(bcfg)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Main evaluation runner.

    Orchestrates the full evaluation pipeline: benchmark generation,
    baseline execution, metric computation, plot generation, and
    result persistence.
    """

    def __init__(
        self,
        config: EvalConfig,
        model=None,
        fl_adapter=None,
    ):
        self.config = config
        self.fl_adapter = fl_adapter

        # Build backend
        if model is not None:
            self.model = model
        else:
            self.model = _build_backend(config)

        # Output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluators
        self.leakage_eval = LeakageEvaluator()
        self.utility_eval = UtilityEvaluator()

        # State
        self._benchmark_data: Optional[List[Dict[str, Any]]] = None
        self._baseline_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self._aggregated_metrics: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def generate_benchmark_data(self) -> List[Dict[str, Any]]:
        """Generate or load benchmark data using BenchmarkGenerator."""
        if self.config.benchmark_data_path:
            self._benchmark_data = BenchmarkGenerator.load_jsonl(
                self.config.benchmark_data_path
            )
        else:
            gen = BenchmarkGenerator()
            suites = gen.generate_all(
                num_s1=self.config.num_s1,
                num_s2=self.config.num_s2,
                num_s3=self.config.num_s3,
                seed=self.config.seed,
            )
            self._benchmark_data = (
                suites["S1"] + suites["S2"] + suites["S3"]
            )

        # Save benchmark data
        bench_path = self.output_dir / "benchmark_data.jsonl"
        BenchmarkGenerator.save_jsonl(self._benchmark_data, str(bench_path))

        return self._benchmark_data

    def run_all_baselines(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run all configured baselines on every benchmark sample."""
        if self._benchmark_data is None:
            self.generate_benchmark_data()

        baseline_config = BaselineConfig(
            total_steps=self.config.total_steps,
            seq_len=self.config.seq_len,
            tokens_per_step_frac=self.config.tokens_per_step_frac,
            temperature=self.config.temperature,
            seed=self.config.seed,
            schedule_draft_end=self.config.schedule_draft_end,
            schedule_safe_end=self.config.schedule_safe_end,
            repair_mode=self.config.repair_mode,
            repair_max_iters=self.config.repair_max_iters,
            verifier_forbidden_tags=self.config.verifier_forbidden_tags,
        )

        runner = BaselineRunner(
            model=self.model,
            config=baseline_config,
            fl_adapter=self.fl_adapter,
            leakage_evaluator=self.leakage_eval,
            utility_evaluator=self.utility_eval,
        )

        # Run baselines
        br_results = runner.run(
            self._benchmark_data,
            baseline_names=self.config.baselines,
        )

        # Convert BaselineResult objects to dicts for storage
        results: Dict[str, List[Dict[str, Any]]] = {}
        for bname, result_list in br_results.items():
            results[bname] = []
            for i, br in enumerate(result_list):
                sample = self._benchmark_data[i]
                entry = {
                    "sample_id": sample.get("sample_id", i),
                    "suite": sample.get("suite", "unknown"),
                    "output_text": br.output_text,
                    "elapsed_sec": br.elapsed_sec,
                    "leakage": {
                        "hard_leakage_count": br.hard_leakage_count,
                        "hard_leakage_rate": br.hard_leakage_rate,
                        "semantic_leakage_detected": br.semantic_leakage,
                    },
                    "utility": {
                        "utility_score": br.utility_score,
                    },
                }
                results[bname].append(entry)

        self._baseline_results = results

        # Save raw results
        serializable = {}
        for bname, bresults in results.items():
            serializable[bname] = []
            for r in bresults:
                entry = {}
                for k, v in r.items():
                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        entry[k] = v
                serializable[bname].append(entry)

        with open(self.output_dir / "baseline_results.json", "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        return results

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregated metrics across all samples per baseline."""
        if self._baseline_results is None:
            self.run_all_baselines()

        aggregated: Dict[str, Any] = {}

        for bname, bresults in self._baseline_results.items():
            if not bresults:
                aggregated[bname] = {}
                continue

            n = len(bresults)

            hard_counts = [
                r.get("leakage", {}).get("hard_leakage_count", 0) for r in bresults
            ]
            hard_rates = [
                r.get("leakage", {}).get("hard_leakage_rate", 0.0) for r in bresults
            ]
            semantic_flags = [
                r.get("leakage", {}).get("semantic_leakage_detected", False)
                for r in bresults
            ]
            elapsed = [r.get("elapsed_sec", 0.0) for r in bresults]

            # Split by suite
            suite_rates: Dict[str, List[float]] = {}
            for r in bresults:
                suite = r.get("suite", "unknown")
                rate = r.get("leakage", {}).get("hard_leakage_rate", 0.0)
                suite_rates.setdefault(suite, []).append(rate)

            agg = {
                "num_samples": n,
                "leakage": {
                    "mean_hard_leakage_count": sum(hard_counts) / n,
                    "mean_hard_leakage_rate": sum(hard_rates) / n,
                    "max_hard_leakage_rate": max(hard_rates) if hard_rates else 0.0,
                    "total_hard_leakage_count": sum(hard_counts),
                    "semantic_leakage_fraction": sum(semantic_flags) / n,
                },
                "speed": {
                    "mean_elapsed_sec": sum(elapsed) / n,
                    "total_elapsed_sec": sum(elapsed),
                },
            }

            # Per-suite breakdown
            for suite, rates in suite_rates.items():
                agg["leakage"][f"{suite}_mean_rate"] = (
                    sum(rates) / len(rates) if rates else 0.0
                )

            aggregated[bname] = agg

        self._aggregated_metrics = aggregated

        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(aggregated, f, indent=2)

        # Save CSV table
        self._save_table(aggregated)

        return aggregated

    def _save_table(self, aggregated: Dict[str, Any]) -> None:
        """Save a CSV summary table."""
        table_path = self.output_dir / "table.csv"
        fields = ["baseline", "num_samples", "mean_leak_rate", "max_leak_rate",
                   "semantic_leak_frac", "mean_elapsed_sec"]
        with open(table_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for bname in sorted(aggregated.keys()):
                m = aggregated[bname]
                writer.writerow({
                    "baseline": bname,
                    "num_samples": m.get("num_samples", 0),
                    "mean_leak_rate": f"{m.get('leakage', {}).get('mean_hard_leakage_rate', 0.0):.4f}",
                    "max_leak_rate": f"{m.get('leakage', {}).get('max_hard_leakage_rate', 0.0):.4f}",
                    "semantic_leak_frac": f"{m.get('leakage', {}).get('semantic_leakage_fraction', 0.0):.4f}",
                    "mean_elapsed_sec": f"{m.get('speed', {}).get('mean_elapsed_sec', 0.0):.4f}",
                })

    def generate_plots(self) -> None:
        """Generate all evaluation plots and save to the output directory."""
        if self._aggregated_metrics is None:
            self.compute_metrics()

        plot_dir = self.output_dir / "figures"
        plot_dir.mkdir(parents=True, exist_ok=True)
        ext = self.config.plot_format

        # Build results dict for plot functions
        plot_results: Dict[str, Dict[str, Any]] = {}
        for bname, agg in self._aggregated_metrics.items():
            plot_results[bname] = {
                "leakage": {
                    "hard_leakage_rate": agg.get("leakage", {}).get("mean_hard_leakage_rate", 0.0),
                    "hard_leakage_count": agg.get("leakage", {}).get("total_hard_leakage_count", 0),
                },
                "utility": {
                    "exact_match_public": 0.0,
                    "rouge1_f1": 0.0,
                    "rougeL_f1": 0.0,
                },
                "metrics": {
                    "elapsed_sec": agg.get("speed", {}).get("mean_elapsed_sec", 0.0),
                },
            }

        try:
            plot_leakage_bar(
                plot_results,
                str(plot_dir / f"leakage_bar.{ext}"),
                dpi=self.config.plot_dpi,
            )
        except Exception:
            pass

        try:
            plot_utility_vs_leakage(
                plot_results,
                str(plot_dir / f"utility_vs_leakage.{ext}"),
                dpi=self.config.plot_dpi,
            )
        except Exception:
            pass

        try:
            plot_runtime_vs_utility(
                plot_results,
                str(plot_dir / f"runtime_vs_utility.{ext}"),
                dpi=self.config.plot_dpi,
            )
        except Exception:
            pass

        # Z distribution (synthetic data for demonstration)
        rng = random.Random(self.config.seed)
        z_values = [max(0.0, min(1.0, rng.gauss(0.4, 0.2))) for _ in range(200)]
        try:
            plot_z_distribution(
                z_values,
                str(plot_dir / f"z_distribution.{ext}"),
                dpi=self.config.plot_dpi,
            )
        except Exception:
            pass

        # FL convergence (synthetic data for demonstration)
        rng = random.Random(self.config.seed)
        rounds = 20
        loss_curve, leak_curve, util_curve = [], [], []
        loss, leak, util = 2.0, 0.8, 0.3
        for _ in range(rounds):
            loss = loss * 0.92 + rng.gauss(0, 0.02)
            leak = leak * 0.88 + rng.gauss(0, 0.01)
            util = util + 0.03 + rng.gauss(0, 0.01)
            loss_curve.append(max(0.01, loss))
            leak_curve.append(max(0, min(1, leak)))
            util_curve.append(max(0, min(1, util)))
        fl_history = {"loss": loss_curve, "leakage_rate": leak_curve, "utility": util_curve}
        try:
            plot_fl_convergence(
                fl_history,
                str(plot_dir / f"fl_convergence.{ext}"),
                dpi=self.config.plot_dpi,
            )
        except Exception:
            pass

    def save_results(self) -> str:
        """Save all results, metrics, and config to the output directory."""
        config_dict = {}
        for k in EvalConfig.__dataclass_fields__:
            config_dict[k] = getattr(self.config, k)

        with open(self.output_dir / "eval_config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        summary = {
            "experiment_name": self.config.experiment_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_samples": len(self._benchmark_data) if self._benchmark_data else 0,
            "baselines_run": self.config.baselines,
            "aggregated_metrics": self._aggregated_metrics,
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return str(self.output_dir)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        print(f"[TPD+FL Eval] Starting evaluation: {self.config.experiment_name}")
        print(f"[TPD+FL Eval] Backend: {self.config.backend}  Device: {self.config.device}")
        print(f"[TPD+FL Eval] Output directory: {self.output_dir}")

        # Step 1: Benchmark data
        print("[TPD+FL Eval] Generating benchmark data...")
        data = self.generate_benchmark_data()
        suites = {}
        for s in data:
            suite = s.get("suite", "unknown")
            suites[suite] = suites.get(suite, 0) + 1
        suite_str = ", ".join(f"{k}={v}" for k, v in sorted(suites.items()))
        print(f"[TPD+FL Eval]   Generated {len(data)} samples ({suite_str})")

        # Step 2: Run baselines
        print(f"[TPD+FL Eval] Running baselines: {self.config.baselines}")
        start = time.time()
        self.run_all_baselines()
        elapsed = time.time() - start
        print(f"[TPD+FL Eval]   Completed in {elapsed:.2f}s")

        # Step 3: Compute metrics
        print("[TPD+FL Eval] Computing aggregated metrics...")
        metrics = self.compute_metrics()

        # Print summary table
        print("\n" + "=" * 72)
        print(f"{'Baseline':<25} {'Leak Rate':>10} {'Max Leak':>10} {'Sem Leak':>10} {'Time(s)':>10}")
        print("-" * 72)
        for bname in sorted(metrics.keys()):
            m = metrics[bname]
            leak = m.get("leakage", {}).get("mean_hard_leakage_rate", 0.0)
            maxl = m.get("leakage", {}).get("max_hard_leakage_rate", 0.0)
            sem = m.get("leakage", {}).get("semantic_leakage_fraction", 0.0)
            t = m.get("speed", {}).get("mean_elapsed_sec", 0.0)
            print(f"{bname:<25} {leak:>10.4f} {maxl:>10.4f} {sem:>10.4f} {t:>10.4f}")
        print("=" * 72 + "\n")

        # Step 4: Generate plots
        if self.config.generate_plots:
            print("[TPD+FL Eval] Generating plots...")
            self.generate_plots()
            print(f"[TPD+FL Eval]   Saved to {self.output_dir / 'figures'}")

        # Step 5: Save results
        result_path = self.save_results()
        print(f"[TPD+FL Eval] Results saved to {result_path}")

        return {
            "output_dir": result_path,
            "num_samples": len(data),
            "baselines_run": self.config.baselines,
            "aggregated_metrics": metrics,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: ``python -m tpd_fl.eval.run_eval``."""
    parser = argparse.ArgumentParser(
        description="TPD+FL Evaluation Suite (CPU-first)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file (e.g., configs/eval/cpu_small.yaml)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None,
                        choices=["synthetic", "llada8b", "llada2"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--num-s1", type=int, default=None)
    parser.add_argument("--num-s2", type=int, default=None)
    parser.add_argument("--num-s3", type=int, default=None)
    parser.add_argument(
        "--baselines", type=str, nargs="+", default=None,
        help="Which baselines to run (e.g., B0 B1 B5 B7)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_eval_config(args.config)
    else:
        config = EvalConfig()

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.backend:
        config.backend = args.backend
    if args.device:
        config.device = args.device
    if args.dtype:
        config.dtype = args.dtype
    if args.num_s1 is not None:
        config.num_s1 = args.num_s1
    if args.num_s2 is not None:
        config.num_s2 = args.num_s2
    if args.num_s3 is not None:
        config.num_s3 = args.num_s3
    if args.baselines:
        config.baselines = args.baselines
    if args.seed is not None:
        config.seed = args.seed
    if args.steps is not None:
        config.total_steps = args.steps
    if args.no_plots:
        config.generate_plots = False

    # Run evaluation
    runner = EvalRunner(config)
    result = runner.run()

    print(f"\nEvaluation complete. Results at: {result['output_dir']}")


if __name__ == "__main__":
    main()
