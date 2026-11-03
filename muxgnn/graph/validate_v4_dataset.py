from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from .magnet_relation_layers import (
    DAG_DEPENDENCY_RELATION,
    GANG_CONTENTION_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2I_OFFLOADING_RELATION,
    V2V_OFFLOADING_RELATION,
)
from .settings import (
    EDGE_A12_CORES,
    EDGE_A7_CORES,
    NUM_EDGE_SERVERS,
    NUM_VEHICLES,
    SOURCE_VEHICLE_ID,
    VEHICLE_A12_CORES,
    VEHICLE_A7_CORES,
    V2V_REACHABLE_VEHICLE_IDS,
)
from .v4_builder import compute_v4_normalization
from .v4_settings import (
    V4_BEHAVIOR_POLICIES,
    V4_MANIFEST_PATH,
    V4_METADATA_PATH,
    V4_NUM_TEST_SCENARIOS,
    V4_NUM_TRAIN_SCENARIOS,
    V4_NUM_VAL_SCENARIOS,
    V4_OUTPUT_DIR,
    V4_RELEASE_TIME_SEC,
    V4_TOTAL_SCENARIOS,
)
from .v4_source import load_v4_physical_task_graph


EPS = 1e-6
ALLOWED_EVENTS = {
    "initial_ready_tasks",
    "task_started",
    "task_completion",
    "task_completion_and_ready",
}


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pairs(edge_index) -> List[Tuple[int, int]]:
    arr = _np(edge_index).astype(int)
    if arr.size == 0:
        return []
    return [(int(u), int(v)) for u, v in arr.T.tolist()]


def _expected_contention(graph) -> Set[Tuple[int, int]]:
    ready = _np(graph["task"].is_ready).astype(bool)
    gang = _np(graph["task"].gang_size).astype(int)
    free_a7 = np.concatenate([
        _np(graph["vehicle"].available_a7_cores),
        _np(graph["edge_server"].available_a7_cores),
    ]).astype(int)
    free_a12 = np.concatenate([
        _np(graph["vehicle"].available_a12_cores),
        _np(graph["edge_server"].available_a12_cores),
    ]).astype(int)
    total_a7 = np.asarray([VEHICLE_A7_CORES] * NUM_VEHICLES + [EDGE_A7_CORES] * NUM_EDGE_SERVERS)
    total_a12 = np.asarray([VEHICLE_A12_CORES] * NUM_VEHICLES + [EDGE_A12_CORES] * NUM_EDGE_SERVERS)

    task_resources: Dict[int, Set[int]] = {i: set() for i in range(len(gang))}
    for relation, offset in (
        (LOCAL_EXECUTION_RELATION, 0),
        (V2V_OFFLOADING_RELATION, 0),
        (V2I_OFFLOADING_RELATION, NUM_VEHICLES),
    ):
        for t, r in _pairs(graph[relation].edge_index):
            task_resources[t].add(r + offset)

    result: Set[Tuple[int, int]] = set()
    ready_ids = [int(x) for x in np.flatnonzero(ready).tolist()]
    for pos, a in enumerate(ready_ids):
        for b in ready_ids[pos + 1:]:
            blocking = 0
            for slot in task_resources[a].intersection(task_resources[b]):
                for free, total in ((free_a7[slot], total_a7[slot]), (free_a12[slot], total_a12[slot])):
                    if total < gang[a] or total < gang[b]:
                        continue
                    if free >= gang[a] and free >= gang[b] and gang[a] + gang[b] > free:
                        blocking += 1
            if blocking:
                result.add((a, b))
                result.add((b, a))
    return result


def _validate_snapshot(
    graph,
    path: Path,
    source_dependencies: Set[Tuple[int, int]],
    predecessors: List[List[int]],
    source_profiles: List[Dict[str, object]],
    expected_candidate_pairs: Dict[Tuple[str, str, str], Set[Tuple[int, int]]],
    normalization: Dict[str, float],
) -> Dict[str, float | int]:
    if str(getattr(graph, "time_unit")) != "second":
        raise AssertionError(f"{path}: time unit is not second")
    if str(getattr(graph, "energy_unit")) != "joule":
        raise AssertionError(f"{path}: energy unit is not joule")

    if set(_pairs(graph[DAG_DEPENDENCY_RELATION].edge_index)) != source_dependencies:
        raise AssertionError(f"{path}: dependency topology changed")

    for relation, expected in expected_candidate_pairs.items():
        if set(_pairs(graph[relation].edge_index)) != expected:
            raise AssertionError(f"{path}: candidate topology mismatch for {relation}")

    ready = _np(graph["task"].is_ready).astype(bool)
    running = _np(graph["task"].is_running).astype(bool)
    completed = _np(graph["task"].is_completed).astype(bool)
    if np.any(ready & running) or np.any(ready & completed) or np.any(running & completed):
        raise AssertionError(f"{path}: task state flags overlap")

    release = _np(graph["task"].release_time_sec).astype(float)
    ready_time = _np(graph["task"].ready_time_sec).astype(float)
    start = _np(graph["task"].start_time_sec).astype(float)
    finish = _np(graph["task"].finish_time_sec).astype(float)
    deadline = _np(graph["task"].deadline_sec).astype(float)
    a7_ms = _np(graph["task"].exec_time_a7_ms).astype(float)
    a12_ms = _np(graph["task"].exec_time_a12_ms).astype(float)
    a7_sec = _np(graph["task"].exec_time_a7_sec).astype(float)
    a12_sec = _np(graph["task"].exec_time_a12_sec).astype(float)
    gang_exec_a7 = _np(graph["task"].gang_exec_time_a7_sec).astype(float)
    gang_exec_a12 = _np(graph["task"].gang_exec_time_a12_sec).astype(float)
    gang = _np(graph["task"].gang_size).astype(int)

    if not np.allclose(release, V4_RELEASE_TIME_SEC, atol=EPS):
        raise AssertionError(f"{path}: external release time must be zero for all tasks")
    if not np.allclose(a7_sec, a7_ms / 1000.0, atol=1e-6):
        raise AssertionError(f"{path}: A7 ms->s conversion mismatch")
    if not np.allclose(a12_sec, a12_ms / 1000.0, atol=1e-6):
        raise AssertionError(f"{path}: A12 ms->s conversion mismatch")
    if not np.allclose(gang_exec_a7, a7_sec, atol=1e-6):
        raise AssertionError(f"{path}: A7 runtime was incorrectly rescaled by m_i")
    if not np.allclose(gang_exec_a12, a12_sec, atol=1e-6):
        raise AssertionError(f"{path}: A12 runtime was incorrectly rescaled by m_i")

    expected_deadline = np.asarray([float(p["deadline_sec"]) for p in source_profiles])
    if not np.allclose(deadline, expected_deadline, atol=1e-5):
        raise AssertionError(f"{path}: deadline mismatch")

    # Ready time is dependency availability, not external release.
    for task_id in range(len(source_profiles)):
        preds = predecessors[task_id]
        has_become_ready = bool(ready[task_id] or running[task_id] or completed[task_id] or start[task_id] >= 0.0)
        if not preds:
            if abs(ready_time[task_id]) > EPS:
                raise AssertionError(f"{path}: root ready time must be zero")
        elif has_become_ready:
            pred_finish = [finish[p] for p in preds]
            if any(x < 0.0 for x in pred_finish):
                raise AssertionError(f"{path}: task became ready before all predecessors had finish times")
            expected_ready = max(max(pred_finish), release[task_id])
            if abs(ready_time[task_id] - expected_ready) > 1e-5:
                raise AssertionError(f"{path}: dependency-ready time mismatch for task {task_id}")
        else:
            if ready_time[task_id] >= 0.0:
                raise AssertionError(f"{path}: not-yet-ready task has a ready timestamp")

        if start[task_id] >= 0.0:
            if start[task_id] + EPS < max(release[task_id], ready_time[task_id]):
                raise AssertionError(f"{path}: task starts before release/ready time")
            if finish[task_id] <= start[task_id]:
                raise AssertionError(f"{path}: task finish is not after start")

    total_a7 = np.asarray([VEHICLE_A7_CORES] * NUM_VEHICLES + [EDGE_A7_CORES] * NUM_EDGE_SERVERS)
    total_a12 = np.asarray([VEHICLE_A12_CORES] * NUM_VEHICLES + [EDGE_A12_CORES] * NUM_EDGE_SERVERS)
    capacity_a7 = np.concatenate([
        _np(graph["vehicle"].scenario_capacity_a7),
        _np(graph["edge_server"].scenario_capacity_a7),
    ]).astype(int)
    capacity_a12 = np.concatenate([
        _np(graph["vehicle"].scenario_capacity_a12),
        _np(graph["edge_server"].scenario_capacity_a12),
    ]).astype(int)
    free_a7 = np.concatenate([
        _np(graph["vehicle"].available_a7_cores),
        _np(graph["edge_server"].available_a7_cores),
    ]).astype(int)
    free_a12 = np.concatenate([
        _np(graph["vehicle"].available_a12_cores),
        _np(graph["edge_server"].available_a12_cores),
    ]).astype(int)
    if np.any(capacity_a7 > total_a7) or np.any(capacity_a12 > total_a12):
        raise AssertionError(f"{path}: scenario capacity exceeds hardware")
    if np.any(free_a7 < 0) or np.any(free_a12 < 0):
        raise AssertionError(f"{path}: negative free cores")
    if np.any(free_a7 > capacity_a7) or np.any(free_a12 > capacity_a12):
        raise AssertionError(f"{path}: free cores exceed scenario capacity")

    assigned_resource = _np(graph["task"].assigned_resource_slot).astype(int)
    assigned_processor = _np(graph["task"].assigned_processor_index).astype(int)
    assigned_cores = _np(graph["task"].assigned_cores).astype(int)
    expected_free_a7 = capacity_a7.copy()
    expected_free_a12 = capacity_a12.copy()
    for task_id in np.flatnonzero(running).tolist():
        slot = int(assigned_resource[task_id])
        proc = int(assigned_processor[task_id])
        cores = int(assigned_cores[task_id])
        if slot < 0 or proc not in (0, 1) or cores != gang[task_id]:
            raise AssertionError(f"{path}: invalid running assignment")
        if proc == 0:
            expected_free_a7[slot] -= cores
        else:
            expected_free_a12[slot] -= cores
    if not np.array_equal(expected_free_a7, free_a7) or not np.array_equal(expected_free_a12, free_a12):
        raise AssertionError(f"{path}: core conservation failed")

    time_scale = float(normalization["candidate_execution_time_scale_sec"])
    energy_scale = float(normalization["candidate_energy_scale_j"])
    for relation, offset in (
        (LOCAL_EXECUTION_RELATION, 0),
        (V2V_OFFLOADING_RELATION, 0),
        (V2I_OFFLOADING_RELATION, NUM_VEHICLES),
    ):
        edge_index = _np(graph[relation].edge_index).astype(int)
        raw = _np(graph[relation].edge_attr_raw).astype(float)
        norm = _np(graph[relation].edge_attr).astype(float)
        if np.any(raw[:, 0:4] <= 0.0):
            raise AssertionError(f"{path}:{relation} non-positive candidate cost")
        if not np.allclose(norm[:, 0:2], raw[:, 0:2] / time_scale, atol=1e-5):
            raise AssertionError(f"{path}:{relation} time normalization mismatch")
        if not np.allclose(norm[:, 2:4], raw[:, 2:4] / energy_scale, atol=1e-5):
            raise AssertionError(f"{path}:{relation} energy normalization mismatch")

        task_ids = edge_index[0]
        resource_slots = edge_index[1] + int(offset)
        widths = gang[task_ids]
        # Candidate runtime must be the measured task duration itself.  m_i is
        # used below only for capacity/feasibility, never to divide runtime.
        if not np.allclose(raw[:, 0], a7_sec[task_ids], atol=1e-6):
            raise AssertionError(f"{path}:{relation} A7 runtime incorrectly depends on m_i")
        if not np.allclose(raw[:, 1], a12_sec[task_ids], atol=1e-6):
            raise AssertionError(f"{path}:{relation} A12 runtime incorrectly depends on m_i")
        expected_ratio_a7 = free_a7[resource_slots] / np.maximum(total_a7[resource_slots], 1)
        expected_ratio_a12 = free_a12[resource_slots] / np.maximum(total_a12[resource_slots], 1)
        expected_feasible_a7 = (free_a7[resource_slots] >= widths) & (total_a7[resource_slots] >= widths)
        expected_feasible_a12 = (free_a12[resource_slots] >= widths) & (total_a12[resource_slots] >= widths)
        if not np.allclose(raw[:, 4], expected_ratio_a7, atol=EPS):
            raise AssertionError(f"{path}:{relation} A7 availability mismatch")
        if not np.allclose(raw[:, 5], expected_ratio_a12, atol=EPS):
            raise AssertionError(f"{path}:{relation} A12 availability mismatch")
        if not np.array_equal(raw[:, 6] > 0.5, expected_feasible_a7):
            raise AssertionError(f"{path}:{relation} A7 feasibility mismatch")
        if not np.array_equal(raw[:, 7] > 0.5, expected_feasible_a12):
            raise AssertionError(f"{path}:{relation} A12 feasibility mismatch")

    actual_contention = set(_pairs(graph[GANG_CONTENTION_RELATION].edge_index))
    if actual_contention != _expected_contention(graph):
        raise AssertionError(f"{path}: contention relation mismatch")

    return {
        "simulation_time_sec": float(_np(graph.simulation_time_sec)[0]),
        "ready": int(np.sum(ready)),
        "running": int(np.sum(running)),
        "completed": int(np.sum(completed)),
        "scheduled": int(np.sum(start >= 0.0)),
    }


def main() -> None:
    if not V4_MANIFEST_PATH.exists() or not V4_METADATA_PATH.exists():
        raise FileNotFoundError(
            "v4 dataset not found. Run first:\npython -m muxgnn.graph.generate_v4_dataset"
        )
    manifest = json.loads(V4_MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(V4_METADATA_PATH.read_text(encoding="utf-8"))
    normalization = compute_v4_normalization()

    if manifest.get("scenario_count") != V4_TOTAL_SCENARIOS:
        raise AssertionError("v4 scenario count mismatch")
    if manifest.get("time_unit") != "second":
        raise AssertionError("v4 manifest time unit mismatch")
    if metadata["physical_units"]["stored_scheduling_time"] != "second":
        raise AssertionError("v4 metadata physical-time contract mismatch")

    expected_split_sizes = {
        "train": V4_NUM_TRAIN_SCENARIOS,
        "val": V4_NUM_VAL_SCENARIOS,
        "test": V4_NUM_TEST_SCENARIOS,
    }
    seen: Set[str] = set()
    for split, count in expected_split_sizes.items():
        names = list(manifest["splits"][split])
        if len(names) != count:
            raise AssertionError(f"v4 {split} scenario count mismatch")
        if seen.intersection(names):
            raise AssertionError("v4 scenario leakage across splits")
        seen.update(names)

    source = load_v4_physical_task_graph()
    profiles = list(source["profiles"])
    source_dependencies = {(int(u), int(v)) for u, v in source["dependencies"]}
    predecessors: List[List[int]] = [[] for _ in range(len(profiles))]
    for u, v in source_dependencies:
        predecessors[v].append(u)

    reachable = {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}
    expected_candidate_pairs: Dict[Tuple[str, str, str], Set[Tuple[int, int]]] = {
        LOCAL_EXECUTION_RELATION: {(t, SOURCE_VEHICLE_ID) for t in range(len(profiles))},
        V2V_OFFLOADING_RELATION: {(t, v) for t in range(len(profiles)) for v in reachable},
        V2I_OFFLOADING_RELATION: {(t, 0) for t in range(len(profiles))},
    }

    total_snapshots = 0
    split_snapshot_counts = {"train": 0, "val": 0, "test": 0}
    policy_counts: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        for scenario_name in manifest["splits"][split]:
            scenario_dir = V4_OUTPUT_DIR / split / scenario_name
            scenario_meta = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
            policy = str(scenario_meta["behavior_policy"])
            if policy not in V4_BEHAVIOR_POLICIES:
                raise AssertionError(f"{scenario_name}: invalid behavior policy")
            policy_counts[policy] += 1

            paths = sorted((scenario_dir / "snapshots").glob("snapshot_*.pt"))
            if len(paths) != int(scenario_meta["num_snapshots"]):
                raise AssertionError(f"{scenario_name}: snapshot count mismatch")
            if not paths:
                raise AssertionError(f"{scenario_name}: no snapshots")

            previous_time = -1.0
            previous_scheduled = -1
            previous_completed = -1
            started: Set[int] = set()
            for index, path in enumerate(paths):
                graph = _load(path)
                if int(getattr(graph, "decision_epoch_index")) != index:
                    raise AssertionError(f"{path}: decision epoch index mismatch")
                if str(getattr(graph, "dataset_split")) != split:
                    raise AssertionError(f"{path}: split mismatch")
                if str(getattr(graph, "trigger_event")) not in ALLOWED_EVENTS:
                    raise AssertionError(f"{path}: invalid event")

                stats = _validate_snapshot(
                    graph, path, source_dependencies, predecessors, profiles,
                    expected_candidate_pairs, normalization,
                )
                if float(stats["simulation_time_sec"]) + EPS < previous_time:
                    raise AssertionError(f"{scenario_name}: time decreased")
                if int(stats["scheduled"]) < previous_scheduled:
                    raise AssertionError(f"{scenario_name}: scheduled count decreased")
                if int(stats["completed"]) < previous_completed:
                    raise AssertionError(f"{scenario_name}: completed count decreased")
                previous_time = float(stats["simulation_time_sec"])
                previous_scheduled = int(stats["scheduled"])
                previous_completed = int(stats["completed"])

                assigned = _np(graph["task"].assigned_resource_slot).astype(int)
                current_started = set(np.flatnonzero(assigned >= 0).tolist())
                if str(getattr(graph, "trigger_event")) == "task_started":
                    new = current_started - started
                    if len(new) != 1:
                        raise AssertionError(f"{path}: start event must add exactly one task")
                started = current_started

            final = _load(paths[-1])
            if int(np.sum(_np(final["task"].is_completed))) != len(profiles):
                raise AssertionError(f"{scenario_name}: final snapshot not complete")
            if len(started) != len(profiles):
                raise AssertionError(f"{scenario_name}: not every task started exactly once")

            total_snapshots += len(paths)
            split_snapshot_counts[split] += len(paths)

    if total_snapshots != int(manifest["total_snapshots"]):
        raise AssertionError("v4 total snapshot count mismatch")

    print("v4 physical-time dataset validation: PASS")
    print(f"Scenarios           : {V4_TOTAL_SCENARIOS}")
    print(f"Total snapshots     : {total_snapshots}")
    print(f"Split snapshots     : {split_snapshot_counts}")
    print(f"Behavior policies   : {dict(policy_counts)}")
    print("Time unit           : second")
    print("Release model       : all tasks release at 0; dependency-ready time separate")
    print("Deadline model      : scheduler-independent DAG-aware subdeadlines")
    print("HEFT/PPO files      : untouched")


if __name__ == "__main__":
    main()
