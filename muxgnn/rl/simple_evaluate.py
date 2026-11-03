from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .config import MUX_ARTIFACT_DIR, MUX_SEED
from .simple_env import SimpleMuxGNNSmokeEnv
from .simple_heft import HEFTResult, result_detail_frame, run_heft_evaluation


SIMPLE_MODEL_PATH = MUX_ARTIFACT_DIR / "simple_v2_smoke" / "models" / "ppo_muxgnn_simple_v2.zip"
SIMPLE_EVAL_DIR = MUX_ARTIFACT_DIR / "simple_v2_smoke" / "evaluation"
SIMPLE_EVAL_PATH = SIMPLE_EVAL_DIR / "evaluation_summary.json"
PPO_EVAL_PATH = SIMPLE_EVAL_DIR / "ppo_evaluation_summary.json"
HEFT_EVAL_PATH = SIMPLE_EVAL_DIR / "heft_evaluation_summary.json"
COMPARISON_CSV_PATH = SIMPLE_EVAL_DIR / "comparison_summary.csv"
IMPROVEMENT_JSON_PATH = SIMPLE_EVAL_DIR / "comparison_improvements.json"
PPO_DETAIL_PATH = SIMPLE_EVAL_DIR / "ppo_schedule_detail.csv"
HEFT_DETAIL_PATH = SIMPLE_EVAL_DIR / "heft_schedule_detail.csv"


_EPS = 1e-12


def _percent_gain_higher_is_better(reference: float, candidate: float) -> float:
    reference = float(reference)
    candidate = float(candidate)
    if abs(reference) < _EPS:
        return float("nan")
    return float((candidate - reference) / abs(reference) * 100.0)


def _percent_reduction_lower_is_better(reference: float, candidate: float) -> float:
    reference = float(reference)
    candidate = float(candidate)
    if abs(reference) < _EPS:
        return float("nan")
    return float((reference - candidate) / abs(reference) * 100.0)


def _resource_label(env: SimpleMuxGNNSmokeEnv, resource_slot: int) -> tuple[str, int]:
    if resource_slot < env.num_vehicles:
        return "vehicle", int(resource_slot)
    return "edge_server", int(resource_slot - env.num_vehicles)


def _ppo_task_detail(
    env: SimpleMuxGNNSmokeEnv,
    assignment_by_task: Dict[int, tuple[int, int]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for task_id in range(env.num_tasks):
        resource_slot, processor_index = assignment_by_task[task_id]
        resource_type, resource_id = _resource_label(env, resource_slot)
        predecessors = env._predecessors[task_id]
        predecessor_ready_time = (
            max(float(env._task_finish[pred]) for pred in predecessors)
            if predecessors
            else 0.0
        )
        rows.append(
            {
                "task_id": int(task_id),
                "predecessor_ready_time": float(predecessor_ready_time),
                "resource_slot": int(resource_slot),
                "resource_type": resource_type,
                "resource_id": int(resource_id),
                "processor_index": int(processor_index),
                "processor": "A7" if processor_index == 0 else "A12",
                "gang_width": int(env._gang_size[task_id]),
                "start": float(env._task_start[task_id]),
                "finish": float(env._task_finish[task_id]),
                "duration": float(
                    env._candidate_duration[task_id, resource_slot, processor_index]
                ),
                "deadline": float(env._deadline[task_id]),
                "energy": float(env._task_energy[task_id]),
                "normalized_energy": float(env._task_energy_norm[task_id]),
                "qos": float(env._task_qos[task_id]),
                "reward": float(env._task_reward[task_id]),
            }
        )
    return pd.DataFrame(rows).sort_values("task_id")


def _run_ppo_evaluation(
    model_path: Path,
    seed: int,
    device: str,
) -> tuple[Dict[str, float | int | bool | str], pd.DataFrame]:
    if not model_path.exists():
        raise FileNotFoundError(f"Simple v2 PPO model not found: {model_path}")

    env = SimpleMuxGNNSmokeEnv(seed=seed)
    model = PPO.load(str(model_path), env=env, device=device)
    model.policy.set_training_mode(False)
    if hasattr(model.policy, "set_exploration_alpha"):
        model.policy.set_exploration_alpha(0.0)

    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    final_info: Dict = {}
    actions = 0
    assignment_by_task: Dict[int, tuple[int, int]] = {}

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        action_int = int(action)
        task_id, resource_slot, processor_index = env.decode_action(action_int)
        if task_id in assignment_by_task:
            raise RuntimeError(f"PPO attempted to schedule task {task_id} more than once.")
        assignment_by_task[task_id] = (resource_slot, processor_index)

        observation, _, terminated, truncated, final_info = env.step(action_int)
        actions += 1
        if actions > env.num_tasks + 5:
            raise RuntimeError("Simple v2 deterministic evaluation exceeded expected task actions.")

    if actions != env.num_tasks or len(assignment_by_task) != env.num_tasks:
        raise RuntimeError(
            "Deterministic PPO evaluation did not produce exactly one action per task."
        )

    payload: Dict[str, float | int | bool | str] = {
        "method": "PPO_MUXGNN_V2",
        "dataset_version": str(getattr(env.graph, "dataset_version", "legacy_native_simple_magnet_v2")),
        "deterministic": True,
        "exploration_alpha": 0.0,
        "seed": int(seed),
        "actions": int(actions),
        "scheduled_tasks": int(final_info["scheduled_tasks"]),
        "completed_tasks": int(final_info["completed_tasks"]),
        "average_qos": float(final_info["average_qos"]),
        "mean_normalized_energy": float(final_info["mean_normalized_energy"]),
        "total_energy": float(final_info["total_energy"]),
        "makespan": float(final_info["makespan"]),
        "on_time_ratio": float(final_info["on_time_ratio"]),
        "episode_reward": float(final_info["episode_reward"]),
        "max_global_energy": float(final_info["max_global_energy"]),
    }
    detail = _ppo_task_detail(env, assignment_by_task)
    env.close()
    return payload, detail


def _validate_fairness(
    ppo: Dict[str, float | int | bool | str],
    heft: HEFTResult,
) -> None:
    heft_metrics = heft.metrics
    if str(ppo["dataset_version"]) != str(heft_metrics["dataset_version"]):
        raise AssertionError("PPO and HEFT did not evaluate the same dataset version.")
    if int(ppo["scheduled_tasks"]) != int(heft_metrics["scheduled_tasks"]):
        raise AssertionError("PPO and HEFT scheduled different task counts.")
    if int(ppo["completed_tasks"]) != int(heft_metrics["completed_tasks"]):
        raise AssertionError("PPO and HEFT completed different task counts.")
    if abs(float(ppo["max_global_energy"]) - float(heft_metrics["max_global_energy"])) > 1e-7:
        raise AssertionError(
            "PPO and HEFT used different global energy-normalization maxima."
        )


def _comparison_frame(
    ppo: Dict[str, float | int | bool | str],
    heft: Dict[str, float | int | bool | str],
) -> pd.DataFrame:
    metric_names = (
        "average_qos",
        "mean_normalized_energy",
        "total_energy",
        "makespan",
        "on_time_ratio",
        "episode_reward",
    )
    return pd.DataFrame(
        [
            {"method": "HEFT_GANG_V2", **{name: float(heft[name]) for name in metric_names}},
            {"method": "PPO_MUXGNN_V2", **{name: float(ppo[name]) for name in metric_names}},
        ]
    )


def _improvement_payload(
    ppo: Dict[str, float | int | bool | str],
    heft: Dict[str, float | int | bool | str],
) -> Dict[str, float]:
    return {
        "qos_gain_percent": _percent_gain_higher_is_better(
            float(heft["average_qos"]), float(ppo["average_qos"])
        ),
        "mean_normalized_energy_reduction_percent": _percent_reduction_lower_is_better(
            float(heft["mean_normalized_energy"]),
            float(ppo["mean_normalized_energy"]),
        ),
        "total_energy_reduction_percent": _percent_reduction_lower_is_better(
            float(heft["total_energy"]), float(ppo["total_energy"])
        ),
        "makespan_reduction_percent": _percent_reduction_lower_is_better(
            float(heft["makespan"]), float(ppo["makespan"])
        ),
        "on_time_ratio_gain_percentage_points": float(
            (float(ppo["on_time_ratio"]) - float(heft["on_time_ratio"])) * 100.0
        ),
        "episode_reward_gain_percent": _percent_gain_higher_is_better(
            float(heft["episode_reward"]), float(ppo["episode_reward"])
        ),
    }


def _save_bar_plot(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    methods = frame["method"].tolist()
    values = frame[metric].astype(float).tolist()
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(methods, values)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_comparison_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    _save_bar_plot(
        frame,
        "average_qos",
        "Average QoS (higher is better)",
        "Simple v2: HEFT-Gang vs PPO + muxGNN — QoS",
        output_dir / "qos_comparison.png",
    )
    _save_bar_plot(
        frame,
        "mean_normalized_energy",
        "Mean normalized energy (lower is better)",
        "Simple v2: HEFT-Gang vs PPO + muxGNN — Energy",
        output_dir / "energy_comparison.png",
    )
    _save_bar_plot(
        frame,
        "makespan",
        "Makespan (lower is better)",
        "Simple v2: HEFT-Gang vs PPO + muxGNN — Makespan",
        output_dir / "makespan_comparison.png",
    )
    _save_bar_plot(
        frame,
        "on_time_ratio",
        "On-time ratio (higher is better)",
        "Simple v2: HEFT-Gang vs PPO + muxGNN — On-time ratio",
        output_dir / "on_time_ratio_comparison.png",
    )


def run_evaluation(
    model_path: Path = SIMPLE_MODEL_PATH,
    seed: int = MUX_SEED,
    device: str = "auto",
) -> Dict[str, object]:
    """Evaluate independent HEFT and PPO+muxGNN on exactly the same v2 graph."""

    SIMPLE_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # The two methods are executed independently.  HEFT never sees the PPO model,
    # logits, rewards, or online action sequence; PPO never sees HEFT ranks/schedule.
    ppo_metrics, ppo_detail = _run_ppo_evaluation(
        model_path=model_path,
        seed=seed,
        device=device,
    )
    heft_result = run_heft_evaluation(output_dir=None)
    heft_metrics = dict(heft_result.metrics)

    _validate_fairness(ppo_metrics, heft_result)

    comparison = _comparison_frame(ppo_metrics, heft_metrics)
    improvements = _improvement_payload(ppo_metrics, heft_metrics)

    PPO_EVAL_PATH.write_text(json.dumps(ppo_metrics, indent=2), encoding="utf-8")
    HEFT_EVAL_PATH.write_text(json.dumps(heft_metrics, indent=2), encoding="utf-8")
    ppo_detail.to_csv(PPO_DETAIL_PATH, index=False)
    result_detail_frame(heft_result).to_csv(HEFT_DETAIL_PATH, index=False)
    comparison.to_csv(COMPARISON_CSV_PATH, index=False)
    IMPROVEMENT_JSON_PATH.write_text(
        json.dumps(improvements, indent=2), encoding="utf-8"
    )
    _save_comparison_plots(comparison, SIMPLE_EVAL_DIR)

    payload: Dict[str, object] = {
        "stage": "legacy_native_v2_fair_heft_vs_ppo_muxgnn_smoke_comparison",
        "paper_grade_result": False,
        "dataset_version": str(ppo_metrics["dataset_version"]),
        "same_dataset": True,
        "same_compute_energy_deadline_semantics": True,
        "same_energy_normalization_maximum": True,
        "algorithmic_independence": {
            "heft_uses_ppo_model": False,
            "heft_uses_ppo_environment_actions": False,
            "ppo_uses_heft_rank_or_schedule": False,
            "shared_input": "same saved Legacy-native v2 HeteroData",
            "heft_decision_rule": "standard upward-rank + minimum EFT with Gang core calendars",
            "communication_cost": "zero because v2 intentionally has no network model",
        },
        "HEFT_GANG_V2": heft_metrics,
        "PPO_MUXGNN_V2": ppo_metrics,
        "improvements_ppo_vs_heft": improvements,
        "note": (
            "This is a fair single-graph v2 smoke comparison. It validates the comparison "
            "pipeline but is not yet multi-DAG/multi-scenario paper-grade evidence."
        ),
    }
    SIMPLE_EVAL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n============================================================")
    print("Simple v2 fair comparison: HEFT-Gang vs PPO + muxGNN")
    print("============================================================")
    print(f"{'Metric':28s} {'HEFT-Gang':>16s} {'PPO+muxGNN':>16s}")
    print("-" * 64)
    for metric in (
        "average_qos",
        "mean_normalized_energy",
        "total_energy",
        "makespan",
        "on_time_ratio",
        "episode_reward",
    ):
        print(
            f"{metric:28s} {float(heft_metrics[metric]):16.6f} "
            f"{float(ppo_metrics[metric]):16.6f}"
        )
    print("\nPPO improvement relative to HEFT:")
    for key, value in improvements.items():
        print(f"  {key:44s}: {value:.6f}")
    print(f"\nSaved: {SIMPLE_EVAL_PATH}")
    print(f"Saved: {COMPARISON_CSV_PATH}")
    print(f"Saved: {PPO_DETAIL_PATH}")
    print(f"Saved: {HEFT_DETAIL_PATH}")
    print("Important: this is still a single-graph v2 smoke comparison, not final paper evidence.")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fair deterministic evaluation of independent HEFT-Gang and simple v2 "
            "PPO+muxGNN on exactly the same Legacy-native v2 graph."
        )
    )
    parser.add_argument("--model", type=Path, default=SIMPLE_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=MUX_SEED)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    run_evaluation(model_path=args.model, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
