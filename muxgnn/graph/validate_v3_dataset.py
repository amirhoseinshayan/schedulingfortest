from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

from .builder import (
    CANDIDATE_EDGE_FEATURE_NAMES,
    RESOURCE_FEATURE_NAMES,
    TASK_FEATURE_NAMES,
    load_legacy_native_task_graph,
)
from .magnet_relation_layers import (
    DAG_DEPENDENCY_RELATION,
    GANG_CONTENTION_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2I_OFFLOADING_RELATION,
    V2V_OFFLOADING_RELATION,
)
from .settings import MAX_GANG_SIZE, NUM_EDGE_SERVERS, NUM_VEHICLES, SOURCE_VEHICLE_ID, V2V_REACHABLE_VEHICLE_IDS
from .v3_builder import V3_CONTENTION_EDGE_FEATURE_NAMES, compute_v3_normalization
from .v3_settings import (
    V3_BEHAVIOR_POLICIES,
    V3_MANIFEST_PATH,
    V3_METADATA_PATH,
    V3_NUM_TEST_SCENARIOS,
    V3_NUM_TRAIN_SCENARIOS,
    V3_NUM_VAL_SCENARIOS,
    V3_OUTPUT_DIR,
    V3_TOTAL_SCENARIOS,
)


EXPECTED_NODE_TYPES = {"task", "vehicle", "edge_server"}
EXPECTED_EDGE_TYPES = {
    DAG_DEPENDENCY_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2V_OFFLOADING_RELATION,
    V2I_OFFLOADING_RELATION,
    GANG_CONTENTION_RELATION,
}
ALLOWED_EVENTS = {
    "initial_task_release",
    "task_started",
    "task_completion",
    "task_completion_and_release",
}
EPS = 1e-6


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if tensor.numel() and not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf.")


def _pairs(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    if edge_index.numel() == 0:
        return []
    return [(int(a), int(b)) for a, b in edge_index.t().tolist()]


def _expected_contention(graph) -> Set[Tuple[int, int]]:
    ready = _tensor_numpy(graph["task"].is_ready).astype(bool)
    gang = _tensor_numpy(graph["task"].gang_size).astype(int)

    num_resources = NUM_VEHICLES + NUM_EDGE_SERVERS
    free_a7 = np.concatenate([
        _tensor_numpy(graph["vehicle"].available_a7_cores),
        _tensor_numpy(graph["edge_server"].available_a7_cores),
    ]).astype(int)
    free_a12 = np.concatenate([
        _tensor_numpy(graph["vehicle"].available_a12_cores),
        _tensor_numpy(graph["edge_server"].available_a12_cores),
    ]).astype(int)
    total_a7 = np.concatenate([
        _tensor_numpy(graph["vehicle"].total_a7_cores),
        _tensor_numpy(graph["edge_server"].total_a7_cores),
    ]).astype(int)
    total_a12 = np.concatenate([
        _tensor_numpy(graph["vehicle"].total_a12_cores),
        _tensor_numpy(graph["edge_server"].total_a12_cores),
    ]).astype(int)

    task_resources: Dict[int, Set[int]] = {i: set() for i in range(graph["task"].num_nodes)}
    for relation, edge_offset in (
        (LOCAL_EXECUTION_RELATION, 0),
        (V2V_OFFLOADING_RELATION, 0),
        (V2I_OFFLOADING_RELATION, NUM_VEHICLES),
    ):
        for task_id, resource_local in _pairs(graph[relation].edge_index):
            task_resources[task_id].add(edge_offset + resource_local)

    expected: Set[Tuple[int, int]] = set()
    ready_ids = np.flatnonzero(ready).tolist()
    for pos, task_a in enumerate(ready_ids):
        for task_b in ready_ids[pos + 1 :]:
            blocking = 0
            for resource_slot in task_resources[task_a].intersection(task_resources[task_b]):
                for free, total in (
                    (free_a7[resource_slot], total_a7[resource_slot]),
                    (free_a12[resource_slot], total_a12[resource_slot]),
                ):
                    if total < gang[task_a] or total < gang[task_b]:
                        continue
                    if free >= gang[task_a] and free >= gang[task_b] and gang[task_a] + gang[task_b] > free:
                        blocking += 1
            if blocking:
                expected.add((int(task_a), int(task_b)))
                expected.add((int(task_b), int(task_a)))
    return expected


def _validate_snapshot(
    graph,
    path: Path,
    source_dependencies: Set[Tuple[int, int]],
    predecessors: Sequence[Sequence[int]],
    expected_candidate_pairs: Dict[Tuple[str, str, str], Set[Tuple[int, int]]],
    normalization: Dict[str, float],
) -> Dict[str, object]:
    if set(graph.node_types) != EXPECTED_NODE_TYPES:
        raise AssertionError(f"{path}: unexpected node types {graph.node_types}")
    if set(graph.edge_types) != EXPECTED_EDGE_TYPES:
        raise AssertionError(f"{path}: unexpected relation set {graph.edge_types}")

    num_tasks = int(graph["task"].num_nodes)
    if num_tasks != 60:
        raise AssertionError(f"{path}: expected 60 tasks, got {num_tasks}")
    if int(graph["vehicle"].num_nodes) != NUM_VEHICLES:
        raise AssertionError(f"{path}: vehicle count mismatch")
    if int(graph["edge_server"].num_nodes) != NUM_EDGE_SERVERS:
        raise AssertionError(f"{path}: edge-server count mismatch")

    if int(graph["task"].x.shape[1]) != len(TASK_FEATURE_NAMES):
        raise AssertionError(f"{path}: task feature dimension mismatch")
    if int(graph["vehicle"].x.shape[1]) != len(RESOURCE_FEATURE_NAMES):
        raise AssertionError(f"{path}: vehicle feature dimension mismatch")
    if int(graph["edge_server"].x.shape[1]) != len(RESOURCE_FEATURE_NAMES):
        raise AssertionError(f"{path}: edge-server feature dimension mismatch")

    expected_dims = {
        DAG_DEPENDENCY_RELATION: 0,
        LOCAL_EXECUTION_RELATION: len(CANDIDATE_EDGE_FEATURE_NAMES),
        V2V_OFFLOADING_RELATION: len(CANDIDATE_EDGE_FEATURE_NAMES),
        V2I_OFFLOADING_RELATION: len(CANDIDATE_EDGE_FEATURE_NAMES),
        GANG_CONTENTION_RELATION: len(V3_CONTENTION_EDGE_FEATURE_NAMES),
    }
    for node_type in graph.node_types:
        _check_finite(f"{path}:{node_type}.x", graph[node_type].x)
    for relation, width in expected_dims.items():
        store = graph[relation]
        if int(store.edge_attr.shape[1]) != width:
            raise AssertionError(f"{path}:{relation} edge feature width mismatch")
        if int(store.edge_attr.shape[0]) != int(store.edge_index.shape[1]):
            raise AssertionError(f"{path}:{relation} edge attr count mismatch")
        _check_finite(f"{path}:{relation}.edge_attr", store.edge_attr)

    dependencies = set(_pairs(graph[DAG_DEPENDENCY_RELATION].edge_index))
    if dependencies != source_dependencies:
        raise AssertionError(f"{path}: dependency topology changed")

    # v3 candidate topology is stable and no longer pruned by temporary core scarcity.
    for relation, expected_pairs in expected_candidate_pairs.items():
        actual_pairs = set(_pairs(graph[relation].edge_index))
        if actual_pairs != expected_pairs:
            raise AssertionError(f"{path}: candidate topology changed for {relation}")

    ready = _tensor_numpy(graph["task"].is_ready).astype(bool)
    running = _tensor_numpy(graph["task"].is_running).astype(bool)
    completed = _tensor_numpy(graph["task"].is_completed).astype(bool)
    if np.any(ready & running) or np.any(ready & completed) or np.any(running & completed):
        raise AssertionError(f"{path}: task state flags are not mutually exclusive")

    x = _tensor_numpy(graph["task"].x)
    for name, values in (
        ("is_ready", ready),
        ("is_running", running),
        ("is_completed", completed),
    ):
        index = TASK_FEATURE_NAMES.index(name)
        if not np.array_equal(x[:, index] > 0.5, values):
            raise AssertionError(f"{path}: task.x {name} disagrees with task attribute")

    active = ready | running | completed
    for task_id in np.flatnonzero(active).tolist():
        if predecessors[task_id] and not all(completed[p] for p in predecessors[task_id]):
            raise AssertionError(f"{path}: active task {task_id} violates DAG precedence")

    start = _tensor_numpy(graph["task"].start_time).astype(float)
    finish = _tensor_numpy(graph["task"].finish_time).astype(float)
    assigned_resource = _tensor_numpy(graph["task"].assigned_resource_slot).astype(int)
    assigned_processor = _tensor_numpy(graph["task"].assigned_processor_index).astype(int)
    assigned_cores = _tensor_numpy(graph["task"].assigned_cores).astype(int)
    scheduled = start >= 0.0
    if np.any(scheduled != (running | completed)):
        raise AssertionError(f"{path}: scheduled tasks must be exactly running or completed")
    if np.any(~scheduled & ((finish >= 0.0) | (assigned_resource >= 0) | (assigned_processor >= 0) | (assigned_cores > 0))):
        raise AssertionError(f"{path}: unscheduled task carries assignment metadata")
    if np.any(scheduled & (finish <= start)):
        raise AssertionError(f"{path}: scheduled task has invalid start/finish")

    total_a7 = np.concatenate([
        _tensor_numpy(graph["vehicle"].total_a7_cores),
        _tensor_numpy(graph["edge_server"].total_a7_cores),
    ]).astype(int)
    total_a12 = np.concatenate([
        _tensor_numpy(graph["vehicle"].total_a12_cores),
        _tensor_numpy(graph["edge_server"].total_a12_cores),
    ]).astype(int)
    capacity_a7 = np.concatenate([
        _tensor_numpy(graph["vehicle"].scenario_capacity_a7),
        _tensor_numpy(graph["edge_server"].scenario_capacity_a7),
    ]).astype(int)
    capacity_a12 = np.concatenate([
        _tensor_numpy(graph["vehicle"].scenario_capacity_a12),
        _tensor_numpy(graph["edge_server"].scenario_capacity_a12),
    ]).astype(int)
    free_a7 = np.concatenate([
        _tensor_numpy(graph["vehicle"].available_a7_cores),
        _tensor_numpy(graph["edge_server"].available_a7_cores),
    ]).astype(int)
    free_a12 = np.concatenate([
        _tensor_numpy(graph["vehicle"].available_a12_cores),
        _tensor_numpy(graph["edge_server"].available_a12_cores),
    ]).astype(int)

    if np.any(capacity_a7 > total_a7) or np.any(capacity_a12 > total_a12):
        raise AssertionError(f"{path}: scenario capacity exceeds total hardware")
    if np.any(free_a7 < 0) or np.any(free_a12 < 0):
        raise AssertionError(f"{path}: negative free cores")
    if np.any(free_a7 > capacity_a7) or np.any(free_a12 > capacity_a12):
        raise AssertionError(f"{path}: free cores exceed scenario capacity")

    expected_free_a7 = capacity_a7.copy()
    expected_free_a12 = capacity_a12.copy()
    for task_id in np.flatnonzero(running).tolist():
        slot = int(assigned_resource[task_id])
        proc = int(assigned_processor[task_id])
        cores = int(assigned_cores[task_id])
        if slot < 0 or slot >= len(expected_free_a7) or proc not in (0, 1) or cores <= 0:
            raise AssertionError(f"{path}: invalid running-task assignment")
        if proc == 0:
            expected_free_a7[slot] -= cores
        else:
            expected_free_a12[slot] -= cores
    if not np.array_equal(expected_free_a7, free_a7) or not np.array_equal(expected_free_a12, free_a12):
        raise AssertionError(f"{path}: resource conservation check failed")

    # Candidate availability/feasibility must exactly reflect current free cores.
    candidate_time_scale = normalization["candidate_execution_time_scale"]
    candidate_energy_scale = normalization["candidate_energy_scale"]
    gang_size = _tensor_numpy(graph["task"].gang_size).astype(int)
    for relation, offset in (
        (LOCAL_EXECUTION_RELATION, 0),
        (V2V_OFFLOADING_RELATION, 0),
        (V2I_OFFLOADING_RELATION, NUM_VEHICLES),
    ):
        edge_index = _tensor_numpy(graph[relation].edge_index).astype(int)
        attr = _tensor_numpy(graph[relation].edge_attr).astype(float)
        raw = _tensor_numpy(graph[relation].edge_attr_raw).astype(float)
        if np.any(raw[:, 0:4] <= 0.0):
            raise AssertionError(f"{path}:{relation} candidate cost must stay positive even when temporarily infeasible")
        if not np.allclose(attr[:, 0:2], raw[:, 0:2] / candidate_time_scale, atol=1e-5):
            raise AssertionError(f"{path}:{relation} time normalization mismatch")
        if not np.allclose(attr[:, 2:4], raw[:, 2:4] / candidate_energy_scale, atol=1e-5):
            raise AssertionError(f"{path}:{relation} energy normalization mismatch")

        task_ids = edge_index[0]
        resource_slots = edge_index[1] + int(offset)
        widths = gang_size[task_ids]
        expected_a7_ratio = free_a7[resource_slots] / np.maximum(total_a7[resource_slots], 1)
        expected_a12_ratio = free_a12[resource_slots] / np.maximum(total_a12[resource_slots], 1)
        expected_a7_feasible = (free_a7[resource_slots] >= widths) & (total_a7[resource_slots] >= widths)
        expected_a12_feasible = (free_a12[resource_slots] >= widths) & (total_a12[resource_slots] >= widths)

        if not np.allclose(raw[:, 4], expected_a7_ratio, atol=EPS):
            raise AssertionError(f"{path}:{relation} A7 availability ratio mismatch")
        if not np.allclose(raw[:, 5], expected_a12_ratio, atol=EPS):
            raise AssertionError(f"{path}:{relation} A12 availability ratio mismatch")
        if not np.array_equal(raw[:, 6] > 0.5, expected_a7_feasible):
            raise AssertionError(f"{path}:{relation} A7 feasibility mismatch")
        if not np.array_equal(raw[:, 7] > 0.5, expected_a12_feasible):
            raise AssertionError(f"{path}:{relation} A12 feasibility mismatch")

    actual_contention = set(_pairs(graph[GANG_CONTENTION_RELATION].edge_index))
    expected_contention = _expected_contention(graph)
    if actual_contention != expected_contention:
        raise AssertionError(f"{path}: dynamic contention relation mismatch")
    for u, v in actual_contention:
        if (v, u) not in actual_contention:
            raise AssertionError(f"{path}: contention relation is not symmetric")
        if not ready[u] or not ready[v]:
            raise AssertionError(f"{path}: contention edge connects non-ready task")

    return {
        "simulation_time": float(_tensor_numpy(graph.simulation_time)[0]),
        "ready": int(np.sum(ready)),
        "running": int(np.sum(running)),
        "completed": int(np.sum(completed)),
        "scheduled": int(np.sum(scheduled)),
        "contention_edges": len(actual_contention),
    }


def main() -> None:
    if not V3_MANIFEST_PATH.exists() or not V3_METADATA_PATH.exists():
        raise FileNotFoundError(
            "v3 dataset not found. Run first:\npython -m muxgnn.graph.generate_v3_dataset"
        )

    manifest = json.loads(V3_MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(V3_METADATA_PATH.read_text(encoding="utf-8"))
    normalization = compute_v3_normalization()

    if manifest.get("scenario_count") != V3_TOTAL_SCENARIOS:
        raise AssertionError("v3 manifest scenario count mismatch")
    expected_split_sizes = {
        "train": V3_NUM_TRAIN_SCENARIOS,
        "val": V3_NUM_VAL_SCENARIOS,
        "test": V3_NUM_TEST_SCENARIOS,
    }
    seen_scenarios: Set[str] = set()
    for split, expected_size in expected_split_sizes.items():
        names = list(manifest["splits"][split])
        if len(names) != expected_size:
            raise AssertionError(f"v3 {split} scenario count mismatch")
        if seen_scenarios.intersection(names):
            raise AssertionError("scenario leakage across v3 splits")
        seen_scenarios.update(names)

    source = load_legacy_native_task_graph()
    source_dependencies = {(int(u), int(v)) for u, v in source["dependencies"]}
    predecessors: List[List[int]] = [[] for _ in range(len(source["profiles"]))]
    for u, v in source_dependencies:
        predecessors[v].append(u)

    reachable_v2v = {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}
    expected_candidate_pairs: Dict[Tuple[str, str, str], Set[Tuple[int, int]]] = {
        LOCAL_EXECUTION_RELATION: {(task_id, SOURCE_VEHICLE_ID) for task_id in range(60)},
        V2V_OFFLOADING_RELATION: {(task_id, vehicle_id) for task_id in range(60) for vehicle_id in reachable_v2v},
        V2I_OFFLOADING_RELATION: {(task_id, 0) for task_id in range(60)},
    }

    total_snapshots = 0
    policy_counts: Counter[str] = Counter()
    split_snapshot_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split in ("train", "val", "test"):
        for scenario_name in manifest["splits"][split]:
            scenario_dir = V3_OUTPUT_DIR / split / scenario_name
            scenario_meta = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
            policy = str(scenario_meta["behavior_policy"])
            if policy not in V3_BEHAVIOR_POLICIES:
                raise AssertionError(f"{scenario_name}: invalid behavior policy")
            policy_counts[policy] += 1

            snapshot_paths = sorted((scenario_dir / "snapshots").glob("snapshot_*.pt"))
            if len(snapshot_paths) != int(scenario_meta["num_snapshots"]):
                raise AssertionError(f"{scenario_name}: snapshot count mismatch")
            if not snapshot_paths:
                raise AssertionError(f"{scenario_name}: no snapshots")

            previous_time = -1.0
            previous_scheduled = -1
            previous_completed = -1
            started_tasks: List[int] = []

            for snapshot_index, snapshot_path in enumerate(snapshot_paths):
                graph = _load(snapshot_path)
                if int(getattr(graph, "decision_epoch_index")) != snapshot_index:
                    raise AssertionError(f"{snapshot_path}: decision epoch index mismatch")
                if str(getattr(graph, "dataset_split")) != split:
                    raise AssertionError(f"{snapshot_path}: split attribute mismatch")
                trigger = str(getattr(graph, "trigger_event"))
                if trigger not in ALLOWED_EVENTS:
                    raise AssertionError(f"{snapshot_path}: invalid trigger event {trigger}")

                stats = _validate_snapshot(
                    graph,
                    snapshot_path,
                    source_dependencies,
                    predecessors,
                    expected_candidate_pairs,
                    normalization,
                )
                if stats["simulation_time"] + EPS < previous_time:
                    raise AssertionError(f"{scenario_name}: simulation time decreased")
                if stats["scheduled"] < previous_scheduled:
                    raise AssertionError(f"{scenario_name}: scheduled count decreased")
                if stats["completed"] < previous_completed:
                    raise AssertionError(f"{scenario_name}: completed count decreased")
                previous_time = float(stats["simulation_time"])
                previous_scheduled = int(stats["scheduled"])
                previous_completed = int(stats["completed"])

                assigned = _tensor_numpy(graph["task"].assigned_resource_slot).astype(int)
                if trigger == "task_started":
                    current_started = np.flatnonzero(assigned >= 0).tolist()
                    new_starts = [task for task in current_started if task not in started_tasks]
                    if len(new_starts) != 1:
                        raise AssertionError(f"{snapshot_path}: start event must add exactly one task")
                    started_tasks.extend(new_starts)

            first = _load(snapshot_paths[0])
            if int(np.sum(_tensor_numpy(first["task"].is_running))) != 0 or int(np.sum(_tensor_numpy(first["task"].is_completed))) != 0:
                raise AssertionError(f"{scenario_name}: invalid initial snapshot")
            last = _load(snapshot_paths[-1])
            if int(np.sum(_tensor_numpy(last["task"].is_completed))) != 60:
                raise AssertionError(f"{scenario_name}: final snapshot does not complete all tasks")
            if len(started_tasks) != 60 or len(set(started_tasks)) != 60:
                raise AssertionError(f"{scenario_name}: every task must start exactly once")

            total_snapshots += len(snapshot_paths)
            split_snapshot_counts[split] += len(snapshot_paths)

    if int(manifest["total_snapshots"]) != total_snapshots:
        raise AssertionError("v3 manifest total snapshot count mismatch")

    # With 18/3/3 scenarios and modulo-3 policy assignment, each policy should
    # appear eight times overall and once in each validation/test split.
    expected_per_policy = V3_TOTAL_SCENARIOS // len(V3_BEHAVIOR_POLICIES)
    for policy in V3_BEHAVIOR_POLICIES:
        if policy_counts[policy] != expected_per_policy:
            raise AssertionError(f"v3 policy distribution mismatch for {policy}")

    if metadata["dataset_version"] != "legacy_native_dynamic_magnet_v3":
        raise AssertionError("v3 metadata version mismatch")

    print("Legacy-native Dynamic MAGNET v3 validation: PASSED")
    print(f"Scenarios       : {V3_TOTAL_SCENARIOS}")
    print(f"Total snapshots : {total_snapshots}")
    print(
        "Snapshots/split : "
        f"train={split_snapshot_counts['train']}, "
        f"val={split_snapshot_counts['val']}, "
        f"test={split_snapshot_counts['test']}"
    )
    print(f"Behavior policies: {dict(policy_counts)}")
    print("No scenario-level train/val/test leakage detected.")


if __name__ == "__main__":
    main()
