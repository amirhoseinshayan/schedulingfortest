from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

from .v4_settings import (
    MILLISECONDS_PER_SECOND,
    V4_BENCHMARK_PROFILE_PATH,
    V4_DEADLINE_SLACK_FACTOR,
    V4_LEGACY_CANONICAL_REFERENCE_RAW_COUNT,
    V4_LEGACY_DAG_PATH,
    V4_RELEASE_TIME_SEC,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_single_dag_row() -> Tuple[List[Dict[str, object]], List[Tuple[int, int]]]:
    with V4_LEGACY_DAG_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"v4 expects exactly one Legacy DAG row; found {len(rows)}")

    row = rows[0]
    characteristics = json.loads(row["characteristics"])
    edges = [(int(u), int(v)) for u, v in json.loads(row["edges"])]
    if int(row["num_subtasks"]) != len(characteristics):
        raise ValueError("Legacy num_subtasks does not match characteristics length")
    return characteristics, edges


def _read_benchmark_profiles() -> Dict[str, Dict[str, str]]:
    with V4_BENCHMARK_PROFILE_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        name = str(row["benchmark_name"])
        if name in result:
            raise ValueError(f"Duplicate benchmark profile: {name}")
        result[name] = row
    return result


def _topological_layers(dag: nx.DiGraph) -> Tuple[List[List[int]], Dict[int, int]]:
    layers = [list(layer) for layer in nx.topological_generations(dag)]
    level_by_task: Dict[int, int] = {}
    for level, layer in enumerate(layers):
        for task_id in layer:
            level_by_task[int(task_id)] = int(level)
    return layers, level_by_task


def load_v4_physical_task_graph() -> Dict[str, object]:
    """Load the exact 60-task DAG with finalized physical-time semantics.

    Timing contract:
      * raw A7/A12 execution values (historically named *_sample_count) are ms,
        per supervisor clarification;
      * all scheduling time stored by v4 is seconds;
      * all tasks have external release_time_sec = 0 because this is a single
        design-time DAG whose complete task set is known at t=0;
      * dependency readiness is separate from external release;
      * task deadlines are DAG-aware, absolute, and independent of HEFT/PPO.
    """

    characteristics, dependencies = _read_single_dag_row()
    benchmark_rows = _read_benchmark_profiles()
    num_tasks = len(characteristics)
    if num_tasks != 60:
        raise ValueError(f"v4 expects the validated 60-task DAG; found {num_tasks}")

    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_tasks))
    dag.add_edges_from(dependencies)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("v4 source graph is not a DAG")
    if not nx.is_weakly_connected(dag):
        raise ValueError("v4 source graph must remain weakly connected")

    layers, level_by_task = _topological_layers(dag)
    roots = [int(n) for n in dag.nodes if dag.in_degree(n) == 0]
    sinks = [int(n) for n in dag.nodes if dag.out_degree(n) == 0]

    profiles: List[Dict[str, object]] = []
    for task_id, task in enumerate(characteristics):
        benchmark_name = str(task.get("benchmark_name", ""))
        if benchmark_name not in benchmark_rows:
            raise ValueError(f"Missing benchmark profile for task {task_id}: {benchmark_name}")
        raw = benchmark_rows[benchmark_name]

        # Supervisor-confirmed interpretation: these raw numeric values are ms.
        a7_ms = float(raw["a7_sample_count"])
        a12_ms = float(raw["a12_sample_count"])
        if a7_ms <= 0.0 or a12_ms <= 0.0:
            raise ValueError(f"Non-positive execution time for task {task_id}")
        a7_sec = a7_ms / MILLISECONDS_PER_SECOND
        a12_sec = a12_ms / MILLISECONDS_PER_SECOND

        a7_power_w = float(raw["a7_avg_total_power"])
        a12_power_w = float(raw["a12_avg_total_power"])
        if a7_power_w <= 0.0 or a12_power_w <= 0.0:
            raise ValueError(f"Non-positive power for task {task_id}")

        gang_size = int(task["m_i"])
        if gang_size < 1:
            raise ValueError(f"Invalid gang size for task {task_id}")

        # Supervisor-confirmed gang semantics: the measured A7/A12 execution
        # time already represents the task duration while it occupies its required
        # m_i cores.  m_i is therefore a simultaneous core requirement only; it
        # does not rescale runtime.  Energy keeps the supplied average total-power
        # interpretation: E[J] = T[s] * P[W], with no m_i multiplier/divisor.
        a7_energy_j = a7_sec * a7_power_w
        a12_energy_j = a12_sec * a12_power_w

        legacy_deadline_canonical = float(task["implicit_deadline"])
        legacy_deadline_physical_sec = (
            legacy_deadline_canonical
            * V4_LEGACY_CANONICAL_REFERENCE_RAW_COUNT
            / MILLISECONDS_PER_SECOND
        )

        profiles.append({
            "task_id": int(task_id),
            "benchmark_name": benchmark_name,
            "gang_size": gang_size,
            "release_time_sec": float(V4_RELEASE_TIME_SEC),
            "a7_execution_time_ms": a7_ms,
            "a12_execution_time_ms": a12_ms,
            "a7_time_sec": a7_sec,
            "a12_time_sec": a12_sec,
            # Compatibility aliases retained for the existing v4 schema.
            # Under the supervisor-confirmed semantics these equal the measured
            # task durations; there is no T/m_i speedup.
            "a7_gang_time_sec": a7_sec,
            "a12_gang_time_sec": a12_sec,
            "a7_power_w": a7_power_w,
            "a12_power_w": a12_power_w,
            "a7_energy_j": a7_energy_j,
            "a12_energy_j": a12_energy_j,
            "period_placeholder": float(task.get("period", 1.0)),
            "legacy_deadline_canonical": legacy_deadline_canonical,
            "legacy_deadline_physical_sec": legacy_deadline_physical_sec,
            "topological_layer": int(level_by_task[task_id]),
        })

    # Independent DAG-aware deadline model. No HEFT/PPO schedule is used.
    earliest_ready_sec = [0.0 for _ in range(num_tasks)]
    earliest_finish_sec = [0.0 for _ in range(num_tasks)]
    fastest_gang_time_sec = [0.0 for _ in range(num_tasks)]
    for task_id in nx.topological_sort(dag):
        task_id = int(task_id)
        preds = list(dag.predecessors(task_id))
        ready = max((earliest_finish_sec[int(p)] for p in preds), default=0.0)
        # Supervisor-defined deadline weight: the measured A7/A12 task time is
        # already interpreted as the duration for the task's required gang m_i.
        # Therefore m_i constrains core occupancy, but it does not divide w_i.
        fastest = min(
            float(profiles[task_id]["a7_time_sec"]),
            float(profiles[task_id]["a12_time_sec"]),
        )
        earliest_ready_sec[task_id] = ready
        fastest_gang_time_sec[task_id] = fastest
        earliest_finish_sec[task_id] = ready + fastest

    critical_path_sec = max(earliest_finish_sec[int(s)] for s in sinks)
    workflow_deadline_sec = V4_RELEASE_TIME_SEC + V4_DEADLINE_SLACK_FACTOR * critical_path_sec
    if critical_path_sec <= 0.0:
        raise ValueError("Invalid critical path duration")

    for task_id, profile in enumerate(profiles):
        progress = earliest_finish_sec[task_id] / critical_path_sec
        deadline_sec = V4_RELEASE_TIME_SEC + progress * (workflow_deadline_sec - V4_RELEASE_TIME_SEC)
        profile["fastest_gang_time_sec"] = float(fastest_gang_time_sec[task_id])
        profile["dependency_earliest_ready_sec"] = float(earliest_ready_sec[task_id])
        profile["dependency_earliest_finish_sec"] = float(earliest_finish_sec[task_id])
        profile["deadline_sec"] = float(deadline_sec)
        profile["deadline_slack_from_dependency_earliest_finish_sec"] = float(
            deadline_sec - earliest_finish_sec[task_id]
        )
        if deadline_sec + 1e-15 < earliest_finish_sec[task_id]:
            raise AssertionError("v4 deadline precedes dependency-constrained earliest finish")

    return {
        "dag": dag,
        "dependencies": dependencies,
        "profiles": profiles,
        "layers": layers,
        "level_by_task": level_by_task,
        "roots": roots,
        "sinks": sinks,
        "critical_path_sec": float(critical_path_sec),
        "workflow_deadline_sec": float(workflow_deadline_sec),
        "deadline_slack_factor": float(V4_DEADLINE_SLACK_FACTOR),
        "source_hashes": {
            "legacy_dag_sha256": sha256(V4_LEGACY_DAG_PATH),
            "benchmark_profiles_sha256": sha256(V4_BENCHMARK_PROFILE_PATH),
        },
    }
