from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

# Safe for Windows, SSH, CI, and headless servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .adapter import RELATION_NAMES
from .config import (
    MUX_EVAL_DIR,
    MUX_EVALUATION_ALPHA,
    MUX_EVALUATION_RUNS,
    MUX_MAX_RESOURCES,
    MUX_NUM_PROCESSOR_TYPES,
    MUX_PROCESSOR_TYPES,
    MUX_SCHEDULE_ACTIONS,
    MUX_MODEL_PATH,
    MUX_SEED,
)
from .env import MuxGNNEventDrivenEnv


EPS = 1e-12

METHOD_HEFT = "HEFT_GANG"
METHOD_PPO = "PPO_MUXGNN"


# ============================================================
# Generic helpers
# ============================================================


def percent_improvement(
    reference_value: float,
    candidate_value: float,
) -> float:
    """Positive percentage means the candidate is lower/better."""

    reference_value = float(reference_value)
    candidate_value = float(candidate_value)

    if abs(reference_value) < EPS:
        return float("nan")

    return (
        (reference_value - candidate_value)
        / reference_value
        * 100.0
    )



def tradeoff_ratio(
    time_improvement: float,
    energy_improvement: float,
) -> float:
    """Legacy-compatible makespan/energy improvement ratio."""

    if (
        np.isnan(time_improvement)
        or np.isnan(energy_improvement)
    ):
        return float("nan")

    denominator = abs(
        float(energy_improvement)
    )

    if denominator < EPS:
        if abs(float(time_improvement)) < EPS:
            return 0.0

        return (
            float("inf")
            if time_improvement > 0.0
            else float("-inf")
        )

    return float(
        time_improvement
        / denominator
    )



def _decode_action(
    action: int,
) -> Tuple[str, int, int, str]:
    """Decode the joint schedule/migration action layout."""

    action = int(action)
    if action < MUX_SCHEDULE_ACTIONS:
        action_type = "schedule"
        local = action
    else:
        action_type = "migration"
        local = action - MUX_SCHEDULE_ACTIONS

    pair_index = local // MUX_NUM_PROCESSOR_TYPES
    processor_index = local % MUX_NUM_PROCESSOR_TYPES
    task_id = pair_index // MUX_MAX_RESOURCES
    resource_slot = pair_index % MUX_MAX_RESOURCES

    return (
        action_type,
        int(task_id),
        int(resource_slot),
        str(MUX_PROCESSOR_TYPES[processor_index]),
    )



def _valid_actions(
    obs: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return valid discrete task-resource-processor actions."""

    mask = np.asarray(
        obs["action_mask"],
        dtype=bool,
    )

    return np.flatnonzero(mask)


# ============================================================
# HEFT-Gang baseline adapted to the new event-driven environment
# ============================================================


def _average_task_durations(
    env: MuxGNNEventDrivenEnv,
) -> np.ndarray:
    """
    Build the same style of computation-cost estimate used by the old
    HEFT-Gang baseline, but for homogeneous A7/A12 gangs.

    The ranking is static and independent of PPO. Resource selection itself
    remains dynamic and is evaluated from the current environment state.
    """

    durations = np.zeros(
        env.num_tasks,
        dtype=np.float64,
    )

    for task_id in range(
        env.num_tasks
    ):
        profile = env.profiles[
            task_id
        ]

        gang_size = max(
            int(profile["gang_size"]),
            1,
        )

        a7_duration = (
            float(profile["a7_time"])
            / gang_size
        )

        a12_duration = (
            float(profile["a12_time"])
            / gang_size
        )

        durations[task_id] = (
            0.5
            * (
                a7_duration
                + a12_duration
            )
        )

    return durations



def _compute_upward_ranks(
    graph: nx.DiGraph,
    average_durations: np.ndarray,
) -> np.ndarray:
    """Compute HEFT-like upward ranks over the DAG."""

    rank = np.zeros(
        graph.number_of_nodes(),
        dtype=np.float64,
    )

    for node in reversed(
        list(
            nx.topological_sort(
                graph
            )
        )
    ):
        successors = list(
            graph.successors(
                node
            )
        )

        if not successors:
            rank[node] = float(
                average_durations[
                    node
                ]
            )
        else:
            rank[node] = float(
                average_durations[
                    node
                ]
            ) + max(
                rank[successor]
                for successor
                in successors
            )

    return rank



def _choose_heft_action(
    env: MuxGNNEventDrivenEnv,
    obs: Dict[str, np.ndarray],
    upward_rank: np.ndarray,
) -> int:
    """
    Event-driven HEFT-Gang baseline.

    HEFT keeps its original non-migrating semantics. At runtime migration
    windows it explicitly chooses KEEP; for ready tasks it chooses the highest
    upward-rank task and the earliest-finish resource/processor action.
    """

    valid = _valid_actions(obs)
    if valid.size == 0:
        raise RuntimeError(
            "HEFT_GANG found no valid action in a non-terminal state."
        )

    schedule_valid = valid[valid < MUX_SCHEDULE_ACTIONS]

    # Migration-only decision window: preserve the non-migrating HEFT baseline.
    if schedule_valid.size == 0:
        for action in valid:
            action_type, task_id, resource_slot, core_type = _decode_action(
                int(action)
            )
            if action_type != "migration":
                continue
            if not (0 <= task_id < env.num_tasks):
                continue
            if (
                resource_slot == int(env.task_resource_slot[task_id])
                and core_type == env.task_core_type[task_id]
            ):
                return int(action)

        raise RuntimeError(
            "HEFT_GANG reached a migration window without a valid KEEP action."
        )

    actions_by_task: Dict[int, List[int]] = {}
    for action in schedule_valid:
        action_type, task_id, _, _ = _decode_action(int(action))
        if action_type != "schedule":
            continue
        if not (0 <= task_id < env.num_tasks):
            continue
        actions_by_task.setdefault(task_id, []).append(int(action))

    if not actions_by_task:
        raise RuntimeError(
            "HEFT_GANG found no real ready task after padding removal."
        )

    selected_task = max(
        actions_by_task.keys(),
        key=lambda task_id: (
            float(upward_rank[task_id]),
            -float(env.profiles[task_id]["deadline"]),
            -int(task_id),
        ),
    )

    candidates: List[Tuple[float, float, int]] = []
    for action in actions_by_task[selected_task]:
        _, task_id, resource_slot, core_type = _decode_action(action)
        choice = env._execution_choice(  # noqa: SLF001
            task_id,
            resource_slot,
            core_type,
        )
        candidates.append(
            (
                float(choice.finish_time),
                float(choice.energy),
                int(action),
            )
        )

    if not candidates:
        raise RuntimeError(
            f"HEFT_GANG found no resource for task {selected_task}."
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return int(candidates[0][2])


# ============================================================
# muxGNN relation attention
# ============================================================


def _mean_relation_attention(
    model: PPO,
    obs: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return mean mux coupling weights over valid graph nodes."""

    tensor_obs, _ = (
        model.policy.obs_to_tensor(
            obs
        )
    )

    attention = (
        model.policy.relation_attention(
            tensor_obs
        )
    )

    attention = (
        attention[0]
        .detach()
        .cpu()
        .numpy()
    )

    node_mask = np.asarray(
        obs["node_mask"],
        dtype=bool,
    )

    relation_presence = np.asarray(
        obs["relation_presence"],
        dtype=bool,
    )

    present_any = (
        relation_presence.any(
            axis=0
        )
    )

    valid_nodes = (
        node_mask
        & present_any
    )

    if not np.any(
        valid_nodes
    ):
        return np.zeros(
            len(RELATION_NAMES),
            dtype=np.float64,
        )

    return attention[
        valid_nodes
    ].mean(
        axis=0
    )


# ============================================================
# Evaluation metrics
# ============================================================


def _max_reference_dag_energy(
    env: MuxGNNEventDrivenEnv,
) -> float:
    """
    Build a scenario-specific upper reference energy for legacy-style
    normalized DAG energy reporting.

    This value is for evaluation/reporting only. It does not alter training,
    rewards, observations, or the graph.
    """

    resources = [
        *env.state.vehicles,
        *env.state.edge_servers,
    ]

    per_task_max: List[float] = []

    for task_id in range(
        env.num_tasks
    ):
        profile = env.profiles[
            task_id
        ]

        gang_size = int(
            profile["gang_size"]
        )

        task_energies: List[float] = []

        for resource in resources:
            compute_scale = max(
                float(
                    resource.compute_scale
                ),
                EPS,
            )

            power_scale = float(
                resource.power_scale
            )

            if (
                int(resource.total_a7)
                >= gang_size
            ):
                exec_time = (
                    float(
                        profile["a7_time"]
                    )
                    / max(
                        gang_size
                        * compute_scale,
                        EPS,
                    )
                )

                task_energies.append(
                    exec_time
                    * gang_size
                    * float(
                        profile["a7_power"]
                    )
                    * power_scale
                )

            if (
                int(resource.total_a12)
                >= gang_size
            ):
                exec_time = (
                    float(
                        profile["a12_time"]
                    )
                    / max(
                        gang_size
                        * compute_scale,
                        EPS,
                    )
                )

                task_energies.append(
                    exec_time
                    * gang_size
                    * float(
                        profile["a12_power"]
                    )
                    * power_scale
                )

        if not task_energies:
            raise RuntimeError(
                f"Task {task_id} has no valid reference-energy resource."
            )

        per_task_max.append(
            max(task_energies)
        )

    return max(
        float(
            np.sum(
                per_task_max
            )
        ),
        EPS,
    )



def _physical_core_utilization(
    env: MuxGNNEventDrivenEnv,
    makespan: float,
) -> float:
    """
    Compute occupied gang core-time divided by schedulable core capacity-time.

    Background load is already represented by resource capacity_a7/capacity_a12,
    so this denominator uses the capacity actually available to this scenario.
    """

    makespan = float(
        makespan
    )

    if makespan <= 0.0:
        return 0.0

    occupied_core_time = 0.0

    for task_id in range(
        env.num_tasks
    ):
        start = float(
            env.task_start_time[
                task_id
            ]
        )

        finish = float(
            env.task_finish_time[
                task_id
            ]
        )

        if (
            not np.isfinite(start)
            or not np.isfinite(finish)
        ):
            continue

        width = int(
            env.profiles[
                task_id
            ]["gang_size"]
        )

        occupied_core_time += (
            width
            * max(
                finish - start,
                0.0,
            )
        )

    schedulable_cores = sum(
        int(resource.capacity_a7)
        + int(resource.capacity_a12)
        for resource
        in [
            *env.state.vehicles,
            *env.state.edge_servers,
        ]
    )

    capacity_time = (
        max(
            schedulable_cores,
            1,
        )
        * makespan
    )

    return float(
        occupied_core_time
        / max(
            capacity_time,
            EPS,
        )
    )



def _metrics_from_env(
    env: MuxGNNEventDrivenEnv,
) -> Dict[str, float]:
    """Build legacy-compatible and new metrics from one completed episode."""

    base_metrics = (
        env.get_metrics()
    )

    finish = np.asarray(
        env.task_finish_time[
            : env.num_tasks
        ],
        dtype=np.float64,
    )

    qos = np.asarray(
        env.task_qos[
            : env.num_tasks
        ],
        dtype=np.float64,
    )

    deadlines = np.asarray(
        [
            float(
                profile["deadline"]
            )
            for profile
            in env.profiles[
                : env.num_tasks
            ]
        ],
        dtype=np.float64,
    )

    gang_widths = np.asarray(
        [
            int(
                profile["gang_size"]
            )
            for profile
            in env.profiles[
                : env.num_tasks
            ]
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(finish)
    ):
        raise AssertionError(
            "Evaluation metrics requested before all task finish times exist."
        )

    tardiness = np.maximum(
        finish - deadlines,
        0.0,
    )

    makespan = float(
        base_metrics["makespan"]
    )

    total_energy_raw = float(
        base_metrics["total_energy"]
    )

    max_vf_dag_energy = (
        _max_reference_dag_energy(
            env
        )
    )

    normalized_energy = (
        total_energy_raw
        / max_vf_dag_energy
    )

    a12_fractions = np.asarray(
        [
            1.0
            if env.task_core_type[
                task_id
            ] == "a12"
            else 0.0
            for task_id
            in range(
                env.num_tasks
            )
        ],
        dtype=np.float64,
    )

    execution_modes: Counter[str] = (
        Counter()
    )

    for task_id in range(
        env.num_tasks
    ):
        slot = int(
            env.task_resource_slot[
                task_id
            ]
        )

        if slot < 0:
            continue

        source_vehicle = int(
            env.current_graph[
                "task"
            ].source_vehicle[
                task_id
            ].item()
        )

        if slot < len(
            env.state.vehicles
        ):
            mode = (
                "local"
                if slot == source_vehicle
                else "v2v"
            )
        else:
            mode = "v2i"

        execution_modes[
            mode
        ] += 1

    total_tasks = max(
        env.num_tasks,
        1,
    )

    return {
        "makespan": makespan,
        "total_energy": float(
            normalized_energy
        ),
        "total_energy_raw": (
            total_energy_raw
        ),
        "max_vf_dag_energy": float(
            max_vf_dag_energy
        ),
        "total_tardiness": float(
            np.sum(
                tardiness
            )
        ),
        "average_qos": float(
            np.mean(qos)
        ),
        "minimum_qos": float(
            np.min(qos)
        ),
        "maximum_qos": float(
            np.max(qos)
        ),
        "qos_std": float(
            np.std(qos)
        ),
        "on_time_task_ratio": float(
            np.mean(
                np.isclose(
                    qos,
                    1.0,
                    atol=1e-8,
                )
            )
        ),
        "zero_qos_task_ratio": float(
            np.mean(
                qos <= 1e-8
            )
        ),
        "deadline_miss_ratio": float(
            base_metrics[
                "deadline_miss_ratio"
            ]
        ),
        "gang_usage_ratio": float(
            np.mean(
                gang_widths > 1.0
            )
        ),
        "mean_scheduled_width": float(
            np.mean(
                gang_widths
            )
        ),
        "mean_a12_fraction": float(
            np.mean(
                a12_fractions
            )
        ),
        "physical_core_utilization": (
            _physical_core_utilization(
                env,
                makespan,
            )
        ),
        "skipped_oversized_count": 0.0,
        "simulation_time": float(
            base_metrics[
                "simulation_time"
            ]
        ),
        "completed_tasks": float(
            base_metrics[
                "completed_tasks"
            ]
        ),
        "local_task_ratio": float(
            execution_modes["local"]
            / total_tasks
        ),
        "v2v_task_ratio": float(
            execution_modes["v2v"]
            / total_tasks
        ),
        "v2i_task_ratio": float(
            execution_modes["v2i"]
            / total_tasks
        ),
        "total_migrations": float(base_metrics["total_migrations"]),
        "migrated_task_ratio": float(base_metrics["migrated_task_ratio"]),
        "migration_energy": float(base_metrics["migration_energy"]),
        "migration_overhead_time": float(
            base_metrics["migration_overhead_time"]
        ),
    }



def _metrics_to_row(
    method: str,
    evaluation_run: int,
    scenario: str,
    scenario_seed: int,
    metrics: Dict[str, float],
    total_reward: float,
) -> Dict[str, object]:
    """Create one row compatible with the legacy comparison report."""

    return {
        "dataset": "event_driven_heterogeneous_MAGNET",
        "row_index": 0,
        "scenario": scenario,
        "scenario_seed": int(
            scenario_seed
        ),
        "method": method,
        "evaluation_run": int(
            evaluation_run
        ),
        "makespan": float(
            metrics["makespan"]
        ),
        "total_energy": float(
            metrics["total_energy"]
        ),
        "total_energy_raw": float(
            metrics["total_energy_raw"]
        ),
        "max_vf_dag_energy": float(
            metrics["max_vf_dag_energy"]
        ),
        "total_tardiness": float(
            metrics["total_tardiness"]
        ),
        "average_qos": float(
            metrics["average_qos"]
        ),
        "minimum_qos": float(
            metrics["minimum_qos"]
        ),
        "maximum_qos": float(
            metrics["maximum_qos"]
        ),
        "qos_std": float(
            metrics["qos_std"]
        ),
        "on_time_task_ratio": float(
            metrics[
                "on_time_task_ratio"
            ]
        ),
        "zero_qos_task_ratio": float(
            metrics[
                "zero_qos_task_ratio"
            ]
        ),
        "deadline_miss_ratio": float(
            metrics[
                "deadline_miss_ratio"
            ]
        ),
        "gang_usage_ratio": float(
            metrics[
                "gang_usage_ratio"
            ]
        ),
        "mean_scheduled_width": float(
            metrics[
                "mean_scheduled_width"
            ]
        ),
        "mean_a12_fraction": float(
            metrics[
                "mean_a12_fraction"
            ]
        ),
        "physical_core_utilization": float(
            metrics[
                "physical_core_utilization"
            ]
        ),
        "skipped_oversized_count": float(
            metrics[
                "skipped_oversized_count"
            ]
        ),
        "simulation_time": float(
            metrics[
                "simulation_time"
            ]
        ),
        "local_task_ratio": float(
            metrics[
                "local_task_ratio"
            ]
        ),
        "v2v_task_ratio": float(
            metrics[
                "v2v_task_ratio"
            ]
        ),
        "v2i_task_ratio": float(
            metrics[
                "v2i_task_ratio"
            ]
        ),
        "total_migrations": float(metrics["total_migrations"]),
        "migrated_task_ratio": float(metrics["migrated_task_ratio"]),
        "migration_energy": float(metrics["migration_energy"]),
        "migration_overhead_time": float(metrics["migration_overhead_time"]),
        "total_reward": float(
            total_reward
        ),
    }


# ============================================================
# Episode execution
# ============================================================


def _append_step_row(
    destination: List[
        Dict[str, object]
    ],
    method: str,
    evaluation_run: int,
    scenario: str,
    step_index: int,
    action: int,
    reward: float,
    env: MuxGNNEventDrivenEnv,
    choice,
) -> None:
    """Store a legacy-style gang composition row plus offloading details."""

    task_id = int(
        choice.task_id
    )

    gang_width = int(
        choice.cores
    )

    a7_count = (
        gang_width
        if choice.core_type == "a7"
        else 0
    )

    a12_count = (
        gang_width
        if choice.core_type == "a12"
        else 0
    )

    start_time = float(
        env.task_start_time[
            task_id
        ]
    )

    finish_time = float(
        env.task_finish_time[
            task_id
        ]
    )

    deadline = float(
        env.profiles[
            task_id
        ]["deadline"]
    )

    qos = float(
        env.task_qos[
            task_id
        ]
    )

    tardiness = max(
        finish_time
        - deadline,
        0.0,
    )

    resource_label = (
        f"V{choice.resource_id}"
        if choice.resource_type == "vehicle"
        else f"E{choice.resource_id}"
    )

    selected_core_names = (
        f"{resource_label}:"
        f"{choice.core_type.upper()}"
        f"x{gang_width}"
    )

    destination.append(
        {
            "method": method,
            "evaluation_run": int(
                evaluation_run
            ),
            "scenario": scenario,
            "step": int(
                step_index
            ),
            "action": int(action),
            "reward": float(reward),
            "task": task_id,
            "gang_width": gang_width,
            "a7_count": a7_count,
            "a12_count": a12_count,
            "execution_mode": str(
                choice.execution_mode
            ),
            "resource_type": str(
                choice.resource_type
            ),
            "resource_id": int(
                choice.resource_id
            ),
            "resource_slot": int(
                choice.resource_slot
            ),
            "core_type": str(
                choice.core_type
            ),
            "start": start_time,
            "finish": finish_time,
            "deadline": deadline,
            "exec_time": float(
                choice.exec_time
            ),
            "queue_delay": float(
                choice.queue_delay
            ),
            "duration": float(
                choice.duration
            ),
            "energy": float(
                choice.energy
            ),
            "tardiness": float(
                tardiness
            ),
            "qos": qos,
            "selected_core_names": (
                selected_core_names
            ),
        }
    )



def _run_episode(
    env: MuxGNNEventDrivenEnv,
    method: str,
    evaluation_run: int,
    model: Optional[PPO] = None,
    exploration_alpha: float = 0.0,
    deterministic_ppo: bool = False,
) -> Tuple[
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """Run one complete schedule with HEFT-Gang or PPO+MUXGNN."""

    obs, reset_info = (
        env.reset()
    )

    scenario = str(
        reset_info["scenario"]
    )

    scenario_seed = int(
        reset_info[
            "scenario_seed"
        ]
    )

    upward_rank = None

    if method == METHOD_HEFT:
        upward_rank = (
            _compute_upward_ranks(
                env.dag,
                _average_task_durations(
                    env
                ),
            )
        )

    if (
        method == METHOD_PPO
        and model is not None
    ):
        model.policy.set_exploration_alpha(
            float(
                exploration_alpha
            )
        )
        model.policy.set_training_mode(
            False
        )

    terminated = False
    truncated = False
    total_reward = 0.0
    step_index = 0

    composition_rows: List[
        Dict[str, object]
    ] = []

    attention_rows: List[
        Dict[str, object]
    ] = []

    while not (
        terminated
        or truncated
    ):
        if method == METHOD_HEFT:
            assert upward_rank is not None

            action = (
                _choose_heft_action(
                    env,
                    obs,
                    upward_rank,
                )
            )

        elif method == METHOD_PPO:
            if model is None:
                raise ValueError(
                    "model is required for PPO_MUXGNN evaluation."
                )

            relation_attention = (
                _mean_relation_attention(
                    model,
                    obs,
                )
            )

            attention_row: Dict[
                str,
                object,
            ] = {
                "method": method,
                "evaluation_run": int(
                    evaluation_run
                ),
                "scenario": scenario,
                "decision_step": int(
                    step_index
                ),
                "simulation_time": float(
                    env.state.simulation_time
                ),
            }

            for relation_name, weight in zip(
                RELATION_NAMES,
                relation_attention,
            ):
                attention_row[
                    relation_name
                ] = float(weight)

            attention_rows.append(
                attention_row
            )

            predicted_action, _ = (
                model.predict(
                    obs,
                    deterministic=(
                        deterministic_ppo
                    ),
                )
            )

            action = int(
                np.asarray(
                    predicted_action
                ).item()
            )

        else:
            raise ValueError(
                f"Unknown method: {method}"
            )

        action_type, task_id, resource_slot, core_type = (
            _decode_action(action)
        )

        # Preview only initial scheduling actions for the legacy composition
        # table. Migration actions are reported through episode migration metrics.
        choice = None
        if action_type == "schedule":
            choice = env._execution_choice(  # noqa: SLF001
                task_id,
                resource_slot,
                core_type,
            )

        (
            next_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        if info.get(
            "invalid_action",
            False,
        ):
            raise RuntimeError(
                f"{method} produced invalid action {action}."
            )

        total_reward += float(
            reward
        )

        if choice is not None:
            _append_step_row(
                destination=composition_rows,
                method=method,
                evaluation_run=evaluation_run,
                scenario=scenario,
                step_index=step_index,
                action=action,
                reward=float(reward),
                env=env,
                choice=choice,
            )

        obs = next_obs
        step_index += 1

    env.validate_schedule()

    metrics = _metrics_from_env(
        env
    )

    detail_row = _metrics_to_row(
        method=method,
        evaluation_run=(
            evaluation_run
        ),
        scenario=scenario,
        scenario_seed=(
            scenario_seed
        ),
        metrics=metrics,
        total_reward=(
            total_reward
        ),
    )

    return (
        detail_row,
        composition_rows,
        attention_rows,
    )


# ============================================================
# Full paired evaluation
# ============================================================


def evaluate_methods(
    model_path: Path,
    evaluation_runs: int,
    evaluation_alpha: float,
    seed: int,
    device: str,
    deterministic_ppo: bool,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate HEFT-Gang and PPO+MUXGNN on the same ordered test scenarios.
    """

    if evaluation_runs <= 0:
        raise ValueError(
            "evaluation_runs must be positive."
        )

    if not (
        0.0
        <= evaluation_alpha
        <= 1.0
    ):
        raise ValueError(
            "evaluation_alpha must be in [0, 1]."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"PPO + muxGNN model not found: {model_path}\n"
            "Run:\n"
            "python -m muxgnn.rl.train"
        )

    # Separate environments with deterministic split cursors guarantee that
    # run k uses the same scenario for both methods.
    heft_env = MuxGNNEventDrivenEnv(
        split="test",
        random_scenarios=False,
        seed=seed,
    )

    ppo_env = MuxGNNEventDrivenEnv(
        split="test",
        random_scenarios=False,
        seed=seed,
    )

    model = PPO.load(
        str(model_path),
        env=ppo_env,
        device=device,
    )

    detail_rows: List[
        Dict[str, object]
    ] = []

    composition_rows: List[
        Dict[str, object]
    ] = []

    attention_rows: List[
        Dict[str, object]
    ] = []

    print(
        "\n=== PPO + muxGNN vs HEFT-Gang ==="
    )
    print(
        f"Model            : {model_path}"
    )
    print(
        f"Evaluation runs  : {evaluation_runs}"
    )
    print(
        f"PPO alpha        : {evaluation_alpha}"
    )
    print(
        f"PPO deterministic: {deterministic_ppo}"
    )

    for evaluation_run in range(
        1,
        evaluation_runs + 1,
    ):
        (
            heft_detail,
            heft_composition,
            _,
        ) = _run_episode(
            env=heft_env,
            method=METHOD_HEFT,
            evaluation_run=(
                evaluation_run
            ),
        )

        (
            ppo_detail,
            ppo_composition,
            ppo_attention,
        ) = _run_episode(
            env=ppo_env,
            method=METHOD_PPO,
            evaluation_run=(
                evaluation_run
            ),
            model=model,
            exploration_alpha=(
                evaluation_alpha
            ),
            deterministic_ppo=(
                deterministic_ppo
            ),
        )

        if (
            heft_detail["scenario"]
            != ppo_detail["scenario"]
        ):
            raise AssertionError(
                "Fair comparison failed: HEFT and PPO used different scenarios."
            )

        detail_rows.extend(
            [
                heft_detail,
                ppo_detail,
            ]
        )

        composition_rows.extend(
            heft_composition
        )

        composition_rows.extend(
            ppo_composition
        )

        attention_rows.extend(
            ppo_attention
        )

        print(
            f"run={evaluation_run:02d} "
            f"scenario={ppo_detail['scenario']} | "
            f"HEFT: makespan={heft_detail['makespan']:.4f}s, "
            f"energy={heft_detail['total_energy']:.6f}, "
            f"qos={heft_detail['average_qos']:.4f} | "
            f"PPO+MUX: makespan={ppo_detail['makespan']:.4f}s, "
            f"energy={ppo_detail['total_energy']:.6f}, "
            f"qos={ppo_detail['average_qos']:.4f}, "
            f"migrations={int(ppo_detail['total_migrations'])}"
        )

    heft_env.close()
    ppo_env.close()

    return (
        pd.DataFrame(
            detail_rows
        ),
        pd.DataFrame(
            composition_rows
        ),
        pd.DataFrame(
            attention_rows
        ),
    )


# ============================================================
# Reports
# ============================================================


def build_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the same core statistics as the legacy project."""

    return (
        detail.groupby(
            "method",
            as_index=False,
        )
        .agg(
            evaluation_count=(
                "makespan",
                "count",
            ),
            makespan_mean=(
                "makespan",
                "mean",
            ),
            makespan_std=(
                "makespan",
                "std",
            ),
            total_energy_mean=(
                "total_energy",
                "mean",
            ),
            total_energy_std=(
                "total_energy",
                "std",
            ),
            total_energy_raw_mean=(
                "total_energy_raw",
                "mean",
            ),
            total_energy_raw_std=(
                "total_energy_raw",
                "std",
            ),
            total_tardiness_mean=(
                "total_tardiness",
                "mean",
            ),
            average_qos_mean=(
                "average_qos",
                "mean",
            ),
            average_qos_std=(
                "average_qos",
                "std",
            ),
            on_time_task_ratio_mean=(
                "on_time_task_ratio",
                "mean",
            ),
            zero_qos_task_ratio_mean=(
                "zero_qos_task_ratio",
                "mean",
            ),
            deadline_miss_ratio_mean=(
                "deadline_miss_ratio",
                "mean",
            ),
            gang_usage_ratio_mean=(
                "gang_usage_ratio",
                "mean",
            ),
            mean_scheduled_width_mean=(
                "mean_scheduled_width",
                "mean",
            ),
            mean_a12_fraction_mean=(
                "mean_a12_fraction",
                "mean",
            ),
            physical_core_utilization_mean=(
                "physical_core_utilization",
                "mean",
            ),
            local_task_ratio_mean=(
                "local_task_ratio",
                "mean",
            ),
            v2v_task_ratio_mean=(
                "v2v_task_ratio",
                "mean",
            ),
            v2i_task_ratio_mean=(
                "v2i_task_ratio",
                "mean",
            ),
            total_migrations_mean=(
                "total_migrations",
                "mean",
            ),
            migrated_task_ratio_mean=(
                "migrated_task_ratio",
                "mean",
            ),
            migration_energy_mean=(
                "migration_energy",
                "mean",
            ),
            migration_overhead_time_mean=(
                "migration_overhead_time",
                "mean",
            ),
            total_reward_mean=(
                "total_reward",
                "mean",
            ),
        )
        .fillna(0.0)
    )



def build_improvement_report(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """Create run-by-run PPO improvement versus paired HEFT baseline."""

    baseline = detail[
        detail["method"]
        == METHOD_HEFT
    ].copy()

    ppo = detail[
        detail["method"]
        == METHOD_PPO
    ].copy()

    if (
        baseline.empty
        or ppo.empty
    ):
        return pd.DataFrame()

    paired = baseline.merge(
        ppo,
        on=[
            "evaluation_run",
            "scenario",
            "scenario_seed",
        ],
        suffixes=(
            "_heft",
            "_ppo",
        ),
        validate="one_to_one",
    )

    rows: List[
        Dict[str, object]
    ] = []

    for _, row in (
        paired.iterrows()
    ):
        time_improvement = (
            percent_improvement(
                row["makespan_heft"],
                row["makespan_ppo"],
            )
        )

        energy_improvement = (
            percent_improvement(
                row[
                    "total_energy_heft"
                ],
                row[
                    "total_energy_ppo"
                ],
            )
        )

        raw_energy_improvement = (
            percent_improvement(
                row[
                    "total_energy_raw_heft"
                ],
                row[
                    "total_energy_raw_ppo"
                ],
            )
        )

        rows.append(
            {
                "evaluation_run": int(
                    row[
                        "evaluation_run"
                    ]
                ),
                "scenario": str(
                    row["scenario"]
                ),
                "scenario_seed": int(
                    row[
                        "scenario_seed"
                    ]
                ),
                "heft_gang_makespan": float(
                    row[
                        "makespan_heft"
                    ]
                ),
                "ppo_muxgnn_makespan": float(
                    row[
                        "makespan_ppo"
                    ]
                ),
                "time_improvement_percent": float(
                    time_improvement
                ),
                "heft_gang_total_energy": float(
                    row[
                        "total_energy_heft"
                    ]
                ),
                "ppo_muxgnn_total_energy": float(
                    row[
                        "total_energy_ppo"
                    ]
                ),
                "energy_improvement_percent": float(
                    energy_improvement
                ),
                "heft_gang_total_energy_raw": float(
                    row[
                        "total_energy_raw_heft"
                    ]
                ),
                "ppo_muxgnn_total_energy_raw": float(
                    row[
                        "total_energy_raw_ppo"
                    ]
                ),
                "raw_energy_improvement_percent": float(
                    raw_energy_improvement
                ),
                "time_energy_tradeoff_ratio": float(
                    tradeoff_ratio(
                        time_improvement,
                        energy_improvement,
                    )
                ),
                "heft_gang_average_qos": float(
                    row[
                        "average_qos_heft"
                    ]
                ),
                "ppo_muxgnn_average_qos": float(
                    row[
                        "average_qos_ppo"
                    ]
                ),
                "qos_difference": float(
                    row[
                        "average_qos_ppo"
                    ]
                    - row[
                        "average_qos_heft"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )



def build_improvement_summary(
    report: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate run-by-run improvement statistics."""

    if report.empty:
        return pd.DataFrame()

    finite_tradeoff = (
        report[
            "time_energy_tradeoff_ratio"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    return pd.DataFrame(
        [
            {
                "time_improvement_percent_mean": float(
                    report[
                        "time_improvement_percent"
                    ].mean()
                ),
                "time_improvement_percent_std": float(
                    report[
                        "time_improvement_percent"
                    ].std(
                        ddof=0
                    )
                ),
                "energy_improvement_percent_mean": float(
                    report[
                        "energy_improvement_percent"
                    ].mean()
                ),
                "energy_improvement_percent_std": float(
                    report[
                        "energy_improvement_percent"
                    ].std(
                        ddof=0
                    )
                ),
                "raw_energy_improvement_percent_mean": float(
                    report[
                        "raw_energy_improvement_percent"
                    ].mean()
                ),
                "raw_energy_improvement_percent_std": float(
                    report[
                        "raw_energy_improvement_percent"
                    ].std(
                        ddof=0
                    )
                ),
                "time_energy_tradeoff_ratio_mean": float(
                    finite_tradeoff.mean()
                ),
                "qos_difference_mean": float(
                    report[
                        "qos_difference"
                    ].mean()
                ),
            }
        ]
    )



def build_execution_mode_summary(
    composition: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize local/V2V/V2I and A7/A12 selections."""

    if composition.empty:
        return pd.DataFrame()

    grouped = (
        composition.groupby(
            [
                "method",
                "execution_mode",
                "core_type",
            ],
            as_index=False,
        )
        .agg(
            task_count=(
                "task",
                "count",
            ),
            mean_energy=(
                "energy",
                "mean",
            ),
            mean_duration=(
                "duration",
                "mean",
            ),
            mean_qos=(
                "qos",
                "mean",
            ),
        )
    )

    totals = (
        composition.groupby(
            "method"
        )[
            "task"
        ]
        .count()
        .to_dict()
    )

    grouped[
        "task_ratio"
    ] = grouped.apply(
        lambda row: (
            float(
                row["task_count"]
            )
            / max(
                int(
                    totals[
                        row["method"]
                    ]
                ),
                1,
            )
        ),
        axis=1,
    )

    return grouped



def build_attention_summary(
    attention: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate learned mux coupling weights over evaluation decisions."""

    if attention.empty:
        return pd.DataFrame()

    rows: List[
        Dict[str, object]
    ] = []

    for relation_name in (
        RELATION_NAMES
    ):
        values = np.asarray(
            attention[
                relation_name
            ],
            dtype=np.float64,
        )

        rows.append(
            {
                "relation": (
                    relation_name
                ),
                "mean_attention": float(
                    values.mean()
                ),
                "std_attention": float(
                    values.std()
                ),
                "min_attention": float(
                    values.min()
                ),
                "max_attention": float(
                    values.max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Plots
# ============================================================


def _bar_plot(
    methods: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    output_path: Path,
    dynamic_qos_axis: bool = False,
) -> None:
    """Legacy-style comparison bar plot."""

    figure, axis = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    bars = axis.bar(
        methods,
        values,
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    if (
        dynamic_qos_axis
        and values
    ):
        minimum = min(values)
        maximum = max(values)

        if abs(
            maximum - minimum
        ) < 1e-9:
            padding = max(
                0.002,
                0.20
                * max(
                    abs(maximum),
                    0.01,
                ),
            )
        else:
            padding = max(
                0.002,
                0.20
                * (
                    maximum
                    - minimum
                ),
            )

        lower = max(
            0.0,
            minimum - padding,
        )

        upper = min(
            1.0,
            maximum + padding,
        )

        if upper <= lower:
            upper = min(
                1.0,
                lower + 0.01,
            )

        axis.set_ylim(
            lower,
            upper,
        )

    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            value,
            (
                f"{value:.4f}"
                if abs(value) < 10
                else f"{value:.2f}"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        figure
    )



def save_plots(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create the three legacy comparison plots."""

    methods = summary[
        "method"
    ].tolist()

    _bar_plot(
        methods,
        summary[
            "makespan_mean"
        ].tolist(),
        "Mean Makespan (s)",
        "PPO+muxGNN vs HEFT-Gang: Makespan",
        output_dir
        / "makespan_comparison.png",
    )

    _bar_plot(
        methods,
        summary[
            "total_energy_mean"
        ].tolist(),
        "Mean Normalized Energy",
        "PPO+muxGNN vs HEFT-Gang: Energy",
        output_dir
        / "energy_comparison.png",
    )

    _bar_plot(
        methods,
        summary[
            "average_qos_mean"
        ].tolist(),
        "Mean QoS",
        "PPO+muxGNN vs HEFT-Gang: Average QoS",
        output_dir
        / "qos_comparison.png",
        dynamic_qos_axis=True,
    )


# ============================================================
# Save all artifacts
# ============================================================


def _save_json_summary(
    output_path: Path,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    improvement_summary: pd.DataFrame,
    attention_summary: pd.DataFrame,
) -> None:
    """Save one compact machine-readable overview."""

    payload: Dict[
        str,
        object,
    ] = {
        "methods": [
            METHOD_HEFT,
            METHOD_PPO,
        ],
        "evaluation_rows": int(
            len(detail)
        ),
        "summary": (
            summary.to_dict(
                orient="records"
            )
        ),
        "improvement_summary": (
            improvement_summary.to_dict(
                orient="records"
            )
        ),
        "mux_attention_summary": (
            attention_summary.to_dict(
                orient="records"
            )
        ),
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )



def run_benchmark(
    model_path: Path = MUX_MODEL_PATH,
    evaluation_runs: int = MUX_EVALUATION_RUNS,
    evaluation_alpha: float = MUX_EVALUATION_ALPHA,
    output_dir: Path = MUX_EVAL_DIR,
    seed: int = MUX_SEED,
    device: str = "auto",
    deterministic_ppo: bool = False,
) -> Dict[str, Path]:
    """Run the complete legacy-compatible evaluation and save all outputs."""

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        detail,
        composition,
        attention,
    ) = evaluate_methods(
        model_path=Path(
            model_path
        ),
        evaluation_runs=int(
            evaluation_runs
        ),
        evaluation_alpha=float(
            evaluation_alpha
        ),
        seed=int(seed),
        device=device,
        deterministic_ppo=bool(
            deterministic_ppo
        ),
    )

    summary = build_summary(
        detail
    )

    improvements = (
        build_improvement_report(
            detail
        )
    )

    improvement_summary = (
        build_improvement_summary(
            improvements
        )
    )

    execution_mode_summary = (
        build_execution_mode_summary(
            composition
        )
    )

    attention_summary = (
        build_attention_summary(
            attention
        )
    )

    paths = {
        "detail": (
            output_dir
            / "comparison_detail.csv"
        ),
        "summary": (
            output_dir
            / "comparison_summary.csv"
        ),
        "improvements": (
            output_dir
            / "comparison_improvements.csv"
        ),
        "improvement_summary": (
            output_dir
            / "comparison_improvements_summary.csv"
        ),
        "qos": (
            output_dir
            / "qos_report.csv"
        ),
        "composition": (
            output_dir
            / "gang_composition_detail.csv"
        ),
        "execution_modes": (
            output_dir
            / "execution_mode_summary.csv"
        ),
        "attention": (
            output_dir
            / "mux_attention_detail.csv"
        ),
        "attention_summary": (
            output_dir
            / "mux_attention_summary.csv"
        ),
        "json_summary": (
            output_dir
            / "evaluation_summary.json"
        ),
        "makespan_plot": (
            output_dir
            / "makespan_comparison.png"
        ),
        "energy_plot": (
            output_dir
            / "energy_comparison.png"
        ),
        "qos_plot": (
            output_dir
            / "qos_comparison.png"
        ),
    }

    detail.to_csv(
        paths["detail"],
        index=False,
    )

    summary.to_csv(
        paths["summary"],
        index=False,
    )

    improvements.to_csv(
        paths["improvements"],
        index=False,
    )

    improvement_summary.to_csv(
        paths[
            "improvement_summary"
        ],
        index=False,
    )

    detail[
        [
            "scenario",
            "scenario_seed",
            "method",
            "evaluation_run",
            "average_qos",
            "minimum_qos",
            "maximum_qos",
            "qos_std",
            "on_time_task_ratio",
            "zero_qos_task_ratio",
            "deadline_miss_ratio",
        ]
    ].to_csv(
        paths["qos"],
        index=False,
    )

    composition.to_csv(
        paths["composition"],
        index=False,
    )

    execution_mode_summary.to_csv(
        paths["execution_modes"],
        index=False,
    )

    attention.to_csv(
        paths["attention"],
        index=False,
    )

    attention_summary.to_csv(
        paths[
            "attention_summary"
        ],
        index=False,
    )

    save_plots(
        summary,
        output_dir,
    )

    _save_json_summary(
        output_path=(
            paths[
                "json_summary"
            ]
        ),
        detail=detail,
        summary=summary,
        improvement_summary=(
            improvement_summary
        ),
        attention_summary=(
            attention_summary
        ),
    )

    print(
        "\n=== Detailed results ==="
    )
    print(
        detail.to_string(
            index=False
        )
    )

    print(
        "\n=== Summary ==="
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n=== PPO_MUXGNN improvement vs HEFT_GANG ==="
    )

    if improvement_summary.empty:
        print(
            "No paired improvement rows."
        )
    else:
        print(
            improvement_summary.to_string(
                index=False
            )
        )

    print(
        f"\nSaved evaluation outputs to: {output_dir}"
    )

    return paths


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PPO+muxGNN against an event-driven HEFT-Gang baseline "
            "and reproduce the legacy comparison outputs."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=MUX_MODEL_PATH,
    )

    parser.add_argument(
        "--eval-runs",
        type=int,
        default=(
            MUX_EVALUATION_RUNS
        ),
    )

    parser.add_argument(
        "--eval-alpha",
        type=float,
        default=(
            MUX_EVALUATION_ALPHA
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MUX_EVAL_DIR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=MUX_SEED,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Use deterministic PPO actions. "
            "By default evaluation remains stochastic, as in the legacy workflow."
        ),
    )

    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        evaluation_runs=(
            args.eval_runs
        ),
        evaluation_alpha=(
            args.eval_alpha
        ),
        output_dir=(
            args.output_dir
        ),
        seed=args.seed,
        device=args.device,
        deterministic_ppo=(
            args.deterministic
        ),
    )


if __name__ == "__main__":
    main()
