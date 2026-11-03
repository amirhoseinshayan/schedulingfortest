from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from typing import Dict, List, Tuple

import torch

from .v4_builder import build_v4_snapshot, compute_v4_normalization, v4_dataset_metadata
from .v4_settings import (
    V4_BASE_SEED,
    V4_BEHAVIOR_POLICIES,
    V4_DEADLINE_AUDIT_CSV_PATH,
    V4_DEPENDENCY_CSV_PATH,
    V4_MANIFEST_PATH,
    V4_METADATA_PATH,
    V4_NUM_TEST_SCENARIOS,
    V4_NUM_TRAIN_SCENARIOS,
    V4_NUM_VAL_SCENARIOS,
    V4_OUTPUT_DIR,
    V4_README_PATH,
    V4_SOURCE_TASKS_CSV_PATH,
    V4_SUMMARY_CSV_PATH,
    V4_TOTAL_SCENARIOS,
)
from .v4_simulation import LegacyNativeV4ScenarioSimulator
from .v4_source import load_v4_physical_task_graph


def _scenario_name(scenario_id: int) -> str:
    return f"scenario_{scenario_id:04d}"


def _split(scenario_id: int) -> str:
    train_end = V4_NUM_TRAIN_SCENARIOS
    val_end = train_end + V4_NUM_VAL_SCENARIOS
    if scenario_id < train_end:
        return "train"
    if scenario_id < val_end:
        return "val"
    return "test"


def _behavior_policy(scenario_id: int) -> str:
    return V4_BEHAVIOR_POLICIES[scenario_id % len(V4_BEHAVIOR_POLICIES)]


def _write_source_tables() -> None:
    source = load_v4_physical_task_graph()
    profiles = list(source["profiles"])

    task_columns = [
        "task_id", "benchmark_name", "topological_layer", "gang_size",
        "release_time_sec", "period_placeholder",
        "a7_execution_time_ms", "a12_execution_time_ms",
        "a7_time_sec", "a12_time_sec",
        "a7_gang_time_sec", "a12_gang_time_sec",
        "a7_power_w", "a12_power_w", "a7_energy_j", "a12_energy_j",
        "dependency_earliest_ready_sec", "dependency_earliest_finish_sec",
        "deadline_sec", "deadline_slack_from_dependency_earliest_finish_sec",
        "legacy_deadline_canonical", "legacy_deadline_physical_sec",
    ]
    with V4_SOURCE_TASKS_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_columns)
        writer.writeheader()
        for p in profiles:
            writer.writerow({k: p[k] for k in task_columns})

    with V4_DEPENDENCY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_task", "target_task"])
        writer.writeheader()
        for u, v in source["dependencies"]:
            writer.writerow({"source_task": int(u), "target_task": int(v)})

    deadline_columns = [
        "task_id", "benchmark_name", "dependency_earliest_ready_sec",
        "fastest_gang_time_sec", "dependency_earliest_finish_sec", "deadline_sec",
        "deadline_slack_from_dependency_earliest_finish_sec",
        "legacy_deadline_physical_sec",
    ]
    with V4_DEADLINE_AUDIT_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=deadline_columns)
        writer.writeheader()
        for p in profiles:
            writer.writerow({k: p[k] for k in deadline_columns})


def _write_trajectory_csv(path, rows: List[Dict[str, object]]) -> None:
    columns = [
        "decision_epoch_index", "simulation_time_sec", "trigger_event",
        "started_task", "completed_tasks", "newly_ready_tasks",
        "num_ready_tasks", "num_running_tasks", "num_completed_tasks",
        "num_scheduled_tasks", "num_local_edges", "num_v2v_edges",
        "num_v2i_edges", "num_contention_edges", "snapshot_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("completed_tasks", "newly_ready_tasks"):
                out[key] = json.dumps(out.get(key, []), separators=(",", ":"))
            writer.writerow({key: out.get(key) for key in columns})


def _save_scenario(
    split: str,
    scenario_id: int,
    scenario_seed: int,
    behavior_policy: str,
    normalization: Dict[str, float],
) -> Dict[str, object]:
    scenario_name = _scenario_name(scenario_id)
    scenario_dir = V4_OUTPUT_DIR / split / scenario_name
    snapshots_dir = scenario_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    simulator = LegacyNativeV4ScenarioSimulator(
        scenario_id=scenario_id,
        seed=scenario_seed,
        behavior_policy=behavior_policy,
    )

    rows: List[Dict[str, object]] = []
    event_counts: Counter[str] = Counter()
    for epoch in simulator.iter_decision_epochs():
        context = epoch.to_metadata()
        graph, graph_meta = build_v4_snapshot(
            state=simulator.state,
            decision_epoch_index=epoch.index,
            event_context=context,
            split=split,
            normalization=normalization,
        )
        snapshot_name = f"snapshot_{epoch.index:04d}.pt"
        torch.save(graph, snapshots_dir / snapshot_name)
        event_counts[epoch.trigger_event] += 1
        rows.append({
            **context,
            "num_ready_tasks": graph_meta["num_ready_tasks"],
            "num_running_tasks": graph_meta["num_running_tasks"],
            "num_completed_tasks": graph_meta["num_completed_tasks"],
            "num_scheduled_tasks": graph_meta["num_scheduled_tasks"],
            "num_local_edges": graph_meta["num_local_edges"],
            "num_v2v_edges": graph_meta["num_v2v_edges"],
            "num_v2i_edges": graph_meta["num_v2i_edges"],
            "num_contention_edges": graph_meta["num_contention_edges"],
            "snapshot_file": snapshot_name,
        })

    if not rows:
        raise RuntimeError(f"{scenario_name} produced no snapshots")
    if int(rows[-1]["num_completed_tasks"]) != simulator.num_tasks:
        raise RuntimeError(f"{scenario_name} did not complete all tasks")

    started = [int(r["started_task"]) for r in rows if r.get("started_task") is not None]
    if len(started) != simulator.num_tasks or len(set(started)) != simulator.num_tasks:
        raise RuntimeError(f"{scenario_name} did not start every task exactly once")

    scenario_meta: Dict[str, object] = {
        "scenario_id": int(scenario_id),
        "scenario_name": scenario_name,
        "split": split,
        "seed": int(scenario_seed),
        "behavior_policy": behavior_policy,
        "snapshot_mode": "event_driven_decision_epochs",
        "time_unit": "second",
        "num_snapshots": len(rows),
        "final_simulation_time_sec": float(simulator.state.simulation_time_sec),
        "final_completed_tasks": int(rows[-1]["num_completed_tasks"]),
        "event_counts": dict(event_counts),
        "scenario_capacity_a7": simulator.state.scenario_capacity_a7.astype(int).tolist(),
        "scenario_capacity_a12": simulator.state.scenario_capacity_a12.astype(int).tolist(),
        "background_reserved_a7": (
            simulator.state.total_a7 - simulator.state.scenario_capacity_a7
        ).astype(int).tolist(),
        "background_reserved_a12": (
            simulator.state.total_a12 - simulator.state.scenario_capacity_a12
        ).astype(int).tolist(),
        "trajectory_csv": "trajectory.csv",
        "snapshots_dir": "snapshots",
    }
    (scenario_dir / "scenario.json").write_text(
        json.dumps(scenario_meta, indent=2), encoding="utf-8"
    )
    _write_trajectory_csv(scenario_dir / "trajectory.csv", rows)
    return scenario_meta


def _write_readme(total_snapshots: int) -> None:
    content = f"""# Legacy-native Dynamic MAGNET v4 — Physical Time

This is the finalized compute-scheduling dataset stage for the next muxGNN/PPO and HEFT integration.

## What changed from v3

- Original raw A7/A12 execution values are interpreted as **milliseconds**, based on supervisor confirmation.
- All scheduling times are stored in **seconds**: `T_sec = T_ms / 1000`.
- Power is in watts and energy is recomputed in joules: `E_J = T_sec * P_W`. The measured task time is used directly; `m_i` only reserves simultaneous cores and never divides execution time.
- This is one design-time DAG: every task has `release_time_sec = 0`.
- Release time and dependency-ready time are separate. Roots are ready at 0; non-roots become ready at `max(predecessor_finish_time_sec)`.
- `period=1` is preserved only as a source placeholder used historically to make utilization numerically equal execution time. It is **not** interpreted as 1 second, an arrival period, or a release time.
- Final task deadlines are scheduler-independent and DAG-aware. The workflow deadline is `1.15 * critical_path`, and each task receives a proportional absolute subdeadline (`D_i = 1.15 * dependency_earliest_finish_i`) using `w_i=min(T_A7_i,T_A12_i)` with no division by `m_i`. The old HEFT-like deadline is retained only in the audit CSV.

## Dataset shape

- 60 tasks, 126 DAG dependency edges, same validated layered DAG as v2/v3.
- Node types: Task, Vehicle, EdgeServer.
- Relations: `precedes`, `local_execute_on`, `v2v_offload_to`, `v2i_offload_to`, `contends_with`.
- 24 event-driven scenarios: 18 train / 3 validation / 3 test.
- Total snapshots for this deterministic design after generation: {total_snapshots}.
- Split unit is the full scenario; snapshots from one scenario never cross splits.

## Dynamic state

Task ready/running/completed flags, ready time, free A7/A12 cores, candidate feasibility, and contention are updated only by valid task start/completion events. Candidate topology remains stable under temporary core scarcity; feasibility is an edge feature.

## Intentionally not invented

No mobility trace, coordinates, distance, bandwidth, data size, transmission delay, contact duration, queue delay, or migration delay is synthesized here. Those require their own defensible source/model and can now be added later in seconds without a unit mismatch.

## Build and validate

```powershell
python -m muxgnn.graph.generate_v4_dataset
python -m muxgnn.graph.validate_v4_dataset
```

The generator deletes/rebuilds **only** `data/muxgnn/legacy_native_dynamic_magnet_v4_physical_time`. Existing v2/v3 datasets, muxGNN/PPO code, reward, HEFT, and evaluation files are untouched.
"""
    V4_README_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    if V4_OUTPUT_DIR.exists():
        print(f"Removing old v4 dataset only: {V4_OUTPUT_DIR}")
        shutil.rmtree(V4_OUTPUT_DIR)
    V4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normalization = compute_v4_normalization()
    metadata = v4_dataset_metadata(normalization)
    metadata["scenario_design"] = {
        "total_scenarios": V4_TOTAL_SCENARIOS,
        "train_scenarios": V4_NUM_TRAIN_SCENARIOS,
        "val_scenarios": V4_NUM_VAL_SCENARIOS,
        "test_scenarios": V4_NUM_TEST_SCENARIOS,
        "base_seed": V4_BASE_SEED,
        "behavior_policies": list(V4_BEHAVIOR_POLICIES),
        "policy_assignment": "scenario_id modulo number of behavior policies",
        "split_unit": "scenario",
        "snapshot_leakage_across_splits": False,
    }
    V4_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_source_tables()

    split_scenarios: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    records: List[Dict[str, object]] = []
    for scenario_id in range(V4_TOTAL_SCENARIOS):
        split = _split(scenario_id)
        seed = V4_BASE_SEED + scenario_id
        policy = _behavior_policy(scenario_id)
        print(f"Building {split}/{_scenario_name(scenario_id)} seed={seed} policy={policy}")
        record = _save_scenario(split, scenario_id, seed, policy, normalization)
        split_scenarios[split].append(str(record["scenario_name"]))
        records.append(record)

    total_snapshots = sum(int(r["num_snapshots"]) for r in records)
    manifest = {
        "dataset_version": "legacy_native_dynamic_magnet_v4_physical_time",
        "time_unit": "second",
        "split_unit": "scenario",
        "splits": split_scenarios,
        "scenario_count": len(records),
        "total_snapshots": total_snapshots,
        "scenarios": {
            str(r["scenario_name"]): {k: v for k, v in r.items() if k != "event_counts"}
            for r in records
        },
    }
    V4_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    columns = [
        "scenario_id", "scenario_name", "split", "seed", "behavior_policy",
        "num_snapshots", "final_simulation_time_sec", "final_completed_tasks",
    ]
    with V4_SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in columns})

    _write_readme(total_snapshots)
    print("\nLegacy-native Dynamic MAGNET v4 physical-time dataset generated.")
    print(f"Output          : {V4_OUTPUT_DIR}")
    print(f"Scenarios       : {len(records)}")
    print(f"Total snapshots : {total_snapshots}")


if __name__ == "__main__":
    main()
