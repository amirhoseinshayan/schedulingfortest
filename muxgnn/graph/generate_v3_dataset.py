from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .v3_builder import (
    build_v3_snapshot,
    compute_v3_normalization,
    v3_dataset_metadata,
)
from .v3_settings import (
    V3_BASE_SEED,
    V3_BEHAVIOR_POLICIES,
    V3_MANIFEST_PATH,
    V3_METADATA_PATH,
    V3_NUM_TEST_SCENARIOS,
    V3_NUM_TRAIN_SCENARIOS,
    V3_NUM_VAL_SCENARIOS,
    V3_OUTPUT_DIR,
    V3_README_PATH,
    V3_SUMMARY_CSV_PATH,
    V3_TOTAL_SCENARIOS,
)
from .v3_simulation import LegacyNativeV3ScenarioSimulator


def _scenario_name(scenario_id: int) -> str:
    return f"scenario_{scenario_id:04d}"


def _split_and_local_id(scenario_id: int) -> Tuple[str, int]:
    train_end = V3_NUM_TRAIN_SCENARIOS
    val_end = train_end + V3_NUM_VAL_SCENARIOS
    if scenario_id < train_end:
        return "train", scenario_id
    if scenario_id < val_end:
        return "val", scenario_id - train_end
    return "test", scenario_id - val_end


def _behavior_policy(scenario_id: int) -> str:
    return V3_BEHAVIOR_POLICIES[scenario_id % len(V3_BEHAVIOR_POLICIES)]


def _write_trajectory_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    columns = [
        "decision_epoch_index",
        "simulation_time",
        "trigger_event",
        "started_task",
        "completed_tasks",
        "released_tasks",
        "num_ready_tasks",
        "num_running_tasks",
        "num_completed_tasks",
        "num_scheduled_tasks",
        "num_local_edges",
        "num_v2v_edges",
        "num_v2i_edges",
        "num_contention_edges",
        "snapshot_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for key in ("completed_tasks", "released_tasks"):
                serialized[key] = json.dumps(serialized.get(key, []), separators=(",", ":"))
            writer.writerow({key: serialized.get(key) for key in columns})


def _save_scenario(
    split: str,
    scenario_id: int,
    scenario_seed: int,
    behavior_policy: str,
    normalization: Dict[str, float],
) -> Dict[str, object]:
    scenario_name = _scenario_name(scenario_id)
    scenario_dir = V3_OUTPUT_DIR / split / scenario_name
    snapshots_dir = scenario_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    simulator = LegacyNativeV3ScenarioSimulator(
        scenario_id=scenario_id,
        seed=scenario_seed,
        behavior_policy=behavior_policy,
    )

    trajectory_rows: List[Dict[str, object]] = []
    event_counts: Counter[str] = Counter()

    for epoch in simulator.iter_decision_epochs():
        context = epoch.to_metadata()
        graph, graph_meta = build_v3_snapshot(
            state=simulator.state,
            decision_epoch_index=epoch.index,
            event_context=context,
            split=split,
            normalization=normalization,
        )

        snapshot_name = f"snapshot_{epoch.index:04d}.pt"
        snapshot_path = snapshots_dir / snapshot_name
        torch.save(graph, snapshot_path)

        event_counts[epoch.trigger_event] += 1
        trajectory_rows.append({
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

    if not trajectory_rows:
        raise RuntimeError(f"{scenario_name} produced no snapshots.")

    final_completed = int(trajectory_rows[-1]["num_completed_tasks"])
    if final_completed != simulator.num_tasks:
        raise RuntimeError(
            f"{scenario_name} ended with {final_completed}/{simulator.num_tasks} completed tasks."
        )

    started_tasks = [
        int(row["started_task"])
        for row in trajectory_rows
        if row.get("started_task") is not None
    ]
    if len(started_tasks) != simulator.num_tasks or len(set(started_tasks)) != simulator.num_tasks:
        raise RuntimeError(f"{scenario_name} did not start every task exactly once.")

    scenario_metadata: Dict[str, object] = {
        "scenario_id": int(scenario_id),
        "scenario_name": scenario_name,
        "split": split,
        "seed": int(scenario_seed),
        "behavior_policy": behavior_policy,
        "snapshot_mode": "event_driven_decision_epochs",
        "num_snapshots": len(trajectory_rows),
        "final_simulation_time": float(simulator.state.simulation_time),
        "final_completed_tasks": final_completed,
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
        json.dumps(scenario_metadata, indent=2), encoding="utf-8"
    )
    _write_trajectory_csv(scenario_dir / "trajectory.csv", trajectory_rows)
    return scenario_metadata


def _write_readme(total_snapshots: int) -> None:
    content = (
        f"# Legacy-native Dynamic MAGNET v3\n\n"
        f"This dataset extends the validated v2 heterogeneous graph into event-driven, "
        f"multi-snapshot trajectories while intentionally keeping mobility/network physics out.\n\n"
        f"## Size\n\n"
        f"- Scenarios: {V3_TOTAL_SCENARIOS}\n"
        f"- Train / Val / Test: {V3_NUM_TRAIN_SCENARIOS} / {V3_NUM_VAL_SCENARIOS} / {V3_NUM_TEST_SCENARIOS}\n"
        f"- Total snapshots in this build: {total_snapshots}\n"
        f"- DAG: the same validated 60-task Legacy DAG used by v2\n"
        f"- Split unit: complete scenario (no snapshot leakage across splits)\n\n"
        f"## Dynamic state\n\n"
        f"Task ready/running/completed flags, free A7/A12 cores, candidate feasibility, "
        f"and contention edges change only through valid task start/completion events.\n\n"
        f"Candidate relation topology is stable: relation existence means static reachability + "
        f"hardware capability; temporary core scarcity is represented by feasibility features, "
        f"not by deleting the relation.\n\n"
        f"## Intentionally absent\n\n"
        f"No DAGBench, mobility, distance, link rate, transmission delay, contact duration, "
        f"queue delay, or migration is used in v3. Time remains canonical relative simulation time, "
        f"not claimed physical seconds.\n\n"
        f"## Rebuild / validate\n\n"
        f"```bash\npython -m muxgnn.graph.generate_v3_dataset\n"
        f"python -m muxgnn.graph.validate_v3_dataset\n```\n"
    )
    V3_README_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    if V3_OUTPUT_DIR.exists():
        print(f"Removing old v3 dataset: {V3_OUTPUT_DIR}")
        shutil.rmtree(V3_OUTPUT_DIR)
    V3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normalization = compute_v3_normalization()
    metadata = v3_dataset_metadata(normalization)
    metadata["scenario_design"] = {
        "total_scenarios": V3_TOTAL_SCENARIOS,
        "train_scenarios": V3_NUM_TRAIN_SCENARIOS,
        "val_scenarios": V3_NUM_VAL_SCENARIOS,
        "test_scenarios": V3_NUM_TEST_SCENARIOS,
        "base_seed": V3_BASE_SEED,
        "behavior_policies": list(V3_BEHAVIOR_POLICIES),
        "policy_assignment": "scenario_id modulo number of behavior policies",
        "split_unit": "scenario",
        "snapshot_leakage_across_splits": False,
    }
    V3_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    split_scenarios: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    scenario_records: List[Dict[str, object]] = []

    for scenario_id in range(V3_TOTAL_SCENARIOS):
        split, _ = _split_and_local_id(scenario_id)
        scenario_seed = V3_BASE_SEED + scenario_id
        policy = _behavior_policy(scenario_id)
        print(
            f"Building {split}/{_scenario_name(scenario_id)} "
            f"seed={scenario_seed} policy={policy}"
        )
        record = _save_scenario(
            split=split,
            scenario_id=scenario_id,
            scenario_seed=scenario_seed,
            behavior_policy=policy,
            normalization=normalization,
        )
        split_scenarios[split].append(str(record["scenario_name"]))
        scenario_records.append(record)

    total_snapshots = sum(int(record["num_snapshots"]) for record in scenario_records)
    manifest = {
        "dataset_version": "legacy_native_dynamic_magnet_v3",
        "split_unit": "scenario",
        "splits": split_scenarios,
        "scenario_count": len(scenario_records),
        "total_snapshots": total_snapshots,
        "scenarios": {
            str(record["scenario_name"]): {
                key: value
                for key, value in record.items()
                if key not in {"event_counts"}
            }
            for record in scenario_records
        },
    }
    V3_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with V3_SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "scenario_id",
            "scenario_name",
            "split",
            "seed",
            "behavior_policy",
            "num_snapshots",
            "final_simulation_time",
            "final_completed_tasks",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in scenario_records:
            writer.writerow({key: record[key] for key in columns})

    _write_readme(total_snapshots)

    print("\nLegacy-native Dynamic MAGNET v3 generated.")
    print(f"Output          : {V3_OUTPUT_DIR}")
    print(f"Scenarios       : {len(scenario_records)}")
    print(f"Total snapshots : {total_snapshots}")
    print(
        "Split scenarios  : "
        f"train={len(split_scenarios['train'])}, "
        f"val={len(split_scenarios['val'])}, "
        f"test={len(split_scenarios['test'])}"
    )


if __name__ == "__main__":
    main()
