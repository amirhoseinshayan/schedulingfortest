from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from _bootstrap import PROJECT_ROOT  # noqa: F401
from config import (
    ALLOW_MIXED_GANGS,
    DATASET_PATH,
    EVALUATION_ALPHA,
    EVALUATION_RUNS,
    EVAL_DIR,
    MAX_GANG_SIZE,
    MODEL_PATH,
    NUM_A12_CORES,
    NUM_A7_CORES,
    QOS_FACTOR,
    REWARD_MODE,
    REWARD_PROPOSAL,
    REWARD_WEIGHTS,
    RUNTIME_MODEL,
    SEED,
)
from legacy_gat.env import DagSchedulingEnv
from policy import MaskedNoGNNActorCriticPolicy
from legacy_gat.utils import run_policy_episode


EPS = 1e-12


def build_env(seed: int = SEED) -> DagSchedulingEnv:
    return DagSchedulingEnv(
        csv_path=DATASET_PATH,
        row_index=0,
        num_a7_cores=NUM_A7_CORES,
        num_a12_cores=NUM_A12_CORES,
        max_gang_size=MAX_GANG_SIZE,
        allow_mixed_gangs=ALLOW_MIXED_GANGS,
        runtime_model=RUNTIME_MODEL,
        reward_mode=REWARD_MODE,
        reward_weights=REWARD_WEIGHTS,
        invalid_action_penalty=2.0,
        sample_dags=False,
        seed=seed,
        qos_factor=QOS_FACTOR,
        reward_proposal=REWARD_PROPOSAL,
    )


def percent_improvement(reference_value: float, candidate_value: float) -> float:
    reference_value = float(reference_value)
    candidate_value = float(candidate_value)
    if abs(reference_value) < EPS:
        return float("nan")
    return ((reference_value - candidate_value) / reference_value) * 100.0


def tradeoff_ratio(time_improvement: float, energy_improvement: float) -> float:
    if np.isnan(time_improvement) or np.isnan(energy_improvement):
        return float("nan")
    denominator = abs(float(energy_improvement))
    if denominator < EPS:
        if abs(float(time_improvement)) < EPS:
            return 0.0
        return float("inf") if time_improvement > 0 else float("-inf")
    return float(time_improvement) / denominator


def metrics_to_row(
    method: str,
    evaluation_run: int,
    metrics: Dict[str, float],
    total_reward: float,
) -> Dict[str, object]:
    return {
        "dataset": DATASET_PATH.name,
        "row_index": 0,
        "method": method,
        "evaluation_run": int(evaluation_run),
        "makespan": float(metrics["makespan"]),
        "total_energy": float(metrics["total_energy"]),
        "total_energy_raw": float(metrics["total_energy_raw"]),
        "total_tardiness": float(metrics["total_tardiness"]),
        "average_qos": float(metrics["average_qos"]),
        "minimum_qos": float(metrics["minimum_qos"]),
        "maximum_qos": float(metrics["maximum_qos"]),
        "qos_std": float(metrics["qos_std"]),
        "on_time_task_ratio": float(metrics["on_time_task_ratio"]),
        "zero_qos_task_ratio": float(metrics["zero_qos_task_ratio"]),
        "gang_usage_ratio": float(metrics["gang_usage_ratio"]),
        "mean_scheduled_width": float(metrics["mean_scheduled_width"]),
        "mean_a12_fraction": float(metrics["mean_a12_fraction"]),
        "physical_core_utilization": float(metrics["physical_core_utilization"]),
        "skipped_oversized_count": float(metrics["skipped_oversized_count"]),
        "total_reward": float(total_reward),
    }


def append_composition_rows(
    destination: List[Dict[str, object]],
    method: str,
    evaluation_run: int,
    step_rows: List[dict],
) -> None:
    for step_index, row in enumerate(step_rows):
        destination.append(
            {
                "method": method,
                "evaluation_run": int(evaluation_run),
                "step": int(step_index),
                "task": int(row["task"]),
                "gang_width": int(row["gang_width"]),
                "a7_count": int(row["a7_count"]),
                "a12_count": int(row["a12_count"]),
                "start": float(row["start"]),
                "finish": float(row["finish"]),
                "energy": float(row["energy"]),
                "qos": float(row["qos"]),
                "selected_core_names": "|".join(row["selected_core_names"]),
            }
        )


def evaluate(
    model_path: Path,
    evaluation_runs: int,
    evaluation_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Gang dataset not found: {DATASET_PATH}. Run python run_legacy_gat.py generate first."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"PPO Gang model not found: {model_path}")

    load_env = build_env(seed=SEED)
    print("\n=== WITHOUT-GNN GANG EVALUATION ===")
    print(f"Dataset           : {DATASET_PATH.name}")
    print(f"Model             : {model_path}")
    print(f"Hardware          : {NUM_A7_CORES} A7 + {NUM_A12_CORES} A12")
    print(f"Action dimension  : {load_env.action_space.n}")
    print(f"Runtime model     : {RUNTIME_MODEL}")
    print(f"Reward mode       : {REWARD_MODE}")
    print(f"PPO eval runs     : {evaluation_runs}")
    print(f"PPO eval alpha    : {evaluation_alpha}\n")

    model = PPO.load(
        str(model_path),
        env=load_env,
        custom_objects={"policy_class": MaskedNoGNNActorCriticPolicy},
    )
    model.policy.set_training_mode(False)

    rows: List[Dict[str, object]] = []
    composition_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(SEED)

    # Fair Gang-aware heuristic baseline.
    heft_env = build_env(seed=SEED)
    try:
        _, _, _, metrics, total_reward, steps = run_policy_episode(
            heft_env,
            "HEFT_GANG",
            rng=rng,
        )
        rows.append(metrics_to_row("HEFT_GANG", 0, metrics, total_reward))
        append_composition_rows(composition_rows, "HEFT_GANG", 0, steps)
    finally:
        heft_env.close()

    # Stochastic PPO evaluation with the advisor-requested minimum alpha.
    for evaluation_run in range(1, evaluation_runs + 1):
        ppo_env = build_env(seed=SEED + evaluation_run)
        try:
            _, _, _, metrics, total_reward, steps = run_policy_episode(
                ppo_env,
                "PPO_GANG",
                rng=rng,
                model=model,
                ppo_deterministic=False,
                ppo_exploration_alpha=evaluation_alpha,
            )
            rows.append(
                metrics_to_row(
                    "PPO_NO_GNN",
                    evaluation_run,
                    metrics,
                    total_reward,
                )
            )
            append_composition_rows(
                composition_rows,
                "PPO_NO_GNN",
                evaluation_run,
                steps,
            )
        finally:
            ppo_env.close()

    load_env.close()
    return pd.DataFrame(rows), pd.DataFrame(composition_rows)


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby("method", as_index=False)
        .agg(
            evaluation_count=("makespan", "count"),
            makespan_mean=("makespan", "mean"),
            makespan_std=("makespan", "std"),
            total_energy_mean=("total_energy", "mean"),
            total_energy_std=("total_energy", "std"),
            total_tardiness_mean=("total_tardiness", "mean"),
            average_qos_mean=("average_qos", "mean"),
            average_qos_std=("average_qos", "std"),
            on_time_task_ratio_mean=("on_time_task_ratio", "mean"),
            zero_qos_task_ratio_mean=("zero_qos_task_ratio", "mean"),
            gang_usage_ratio_mean=("gang_usage_ratio", "mean"),
            mean_scheduled_width_mean=("mean_scheduled_width", "mean"),
            mean_a12_fraction_mean=("mean_a12_fraction", "mean"),
            physical_core_utilization_mean=("physical_core_utilization", "mean"),
            total_reward_mean=("total_reward", "mean"),
        )
        .fillna(0.0)
    )


def build_improvement_report(detail: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = detail[detail["method"] == "HEFT_GANG"]
    ppo_rows = detail[detail["method"] == "PPO_NO_GNN"]
    if baseline_rows.empty or ppo_rows.empty:
        return pd.DataFrame()

    baseline = baseline_rows.iloc[0]
    rows: List[Dict[str, float]] = []
    for _, ppo in ppo_rows.iterrows():
        time_improvement = percent_improvement(
            baseline["makespan"],
            ppo["makespan"],
        )
        energy_improvement = percent_improvement(
            baseline["total_energy"],
            ppo["total_energy"],
        )
        rows.append(
            {
                "evaluation_run": int(ppo["evaluation_run"]),
                "heft_gang_makespan": float(baseline["makespan"]),
                "ppo_gang_makespan": float(ppo["makespan"]),
                "time_improvement_percent": float(time_improvement),
                "heft_gang_total_energy": float(baseline["total_energy"]),
                "ppo_gang_total_energy": float(ppo["total_energy"]),
                "energy_improvement_percent": float(energy_improvement),
                "time_energy_tradeoff_ratio": float(
                    tradeoff_ratio(time_improvement, energy_improvement)
                ),
                "heft_gang_average_qos": float(baseline["average_qos"]),
                "ppo_gang_average_qos": float(ppo["average_qos"]),
                "qos_difference": float(
                    ppo["average_qos"] - baseline["average_qos"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_improvement_summary(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "time_improvement_percent_mean": float(
                    report["time_improvement_percent"].mean()
                ),
                "time_improvement_percent_std": float(
                    report["time_improvement_percent"].std(ddof=0)
                ),
                "energy_improvement_percent_mean": float(
                    report["energy_improvement_percent"].mean()
                ),
                "energy_improvement_percent_std": float(
                    report["energy_improvement_percent"].std(ddof=0)
                ),
                "time_energy_tradeoff_ratio_mean": float(
                    report["time_energy_tradeoff_ratio"].replace(
                        [np.inf, -np.inf], np.nan
                    ).mean()
                ),
                "qos_difference_mean": float(report["qos_difference"].mean()),
            }
        ]
    )


def _bar_plot(
    methods: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    output_path: Path,
    dynamic_qos_axis: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(methods, values)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", alpha=0.5)

    if dynamic_qos_axis and values:
        minimum = min(values)
        maximum = max(values)
        if abs(maximum - minimum) < 1e-9:
            padding = max(0.002, 0.20 * max(abs(maximum), 0.01))
        else:
            padding = max(0.002, 0.20 * (maximum - minimum))
        lower = max(0.0, minimum - padding)
        upper = min(1.0, maximum + padding)
        if upper <= lower:
            upper = min(1.0, lower + 0.01)
        axis.set_ylim(lower, upper)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    methods = summary["method"].tolist()
    _bar_plot(
        methods,
        summary["makespan_mean"].tolist(),
        "Mean Makespan",
        "PPO-without-GNN+Gang vs HEFT-Gang: Makespan",
        output_dir / "makespan_comparison.png",
    )
    _bar_plot(
        methods,
        summary["total_energy_mean"].tolist(),
        "Mean Normalized Energy",
        "PPO-without-GNN+Gang vs HEFT-Gang: Energy",
        output_dir / "energy_comparison.png",
    )
    _bar_plot(
        methods,
        summary["average_qos_mean"].tolist(),
        "Mean QoS",
        "PPO-without-GNN+Gang vs HEFT-Gang: Average QoS",
        output_dir / "qos_comparison.png",
        dynamic_qos_axis=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO WITHOUT GNN Gang scheduler.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--eval-runs", type=int, default=EVALUATION_RUNS)
    parser.add_argument("--eval-alpha", type=float, default=EVALUATION_ALPHA)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args()

    if args.eval_runs <= 0:
        raise ValueError("--eval-runs must be positive.")
    if not 0.0 <= args.eval_alpha <= 1.0:
        raise ValueError("--eval-alpha must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail, compositions = evaluate(
        model_path=args.model,
        evaluation_runs=args.eval_runs,
        evaluation_alpha=args.eval_alpha,
    )
    summary = build_summary(detail)
    improvements = build_improvement_report(detail)
    improvement_summary = build_improvement_summary(improvements)

    detail_path = args.output_dir / "comparison_detail.csv"
    summary_path = args.output_dir / "comparison_summary.csv"
    improvement_path = args.output_dir / "comparison_improvements.csv"
    improvement_summary_path = args.output_dir / "comparison_improvements_summary.csv"
    qos_path = args.output_dir / "qos_report.csv"
    composition_path = args.output_dir / "gang_composition_detail.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    improvements.to_csv(improvement_path, index=False)
    improvement_summary.to_csv(improvement_summary_path, index=False)
    detail[
        [
            "method",
            "evaluation_run",
            "average_qos",
            "minimum_qos",
            "maximum_qos",
            "qos_std",
            "on_time_task_ratio",
            "zero_qos_task_ratio",
        ]
    ].to_csv(qos_path, index=False)
    compositions.to_csv(composition_path, index=False)
    save_plots(summary, args.output_dir)

    print("\n=== Detailed results ===")
    print(detail.to_string(index=False))
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\n=== PPO_NO_GNN improvement vs HEFT_GANG ===")
    print(improvement_summary.to_string(index=False))
    print(f"\nSaved evaluation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
