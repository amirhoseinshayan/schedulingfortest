from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from .builder import (
    CANDIDATE_EDGE_FEATURE_NAMES,
    DEPENDENCY_EDGE_FEATURE_NAMES,
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
from .settings import (
    EDGE_A12_CORES,
    EDGE_A7_CORES,
    LEGACY_DATASET_PATH,
    LEGACY_DATASET_PATH,
    MAX_GANG_SIZE,
    NUM_EDGE_SERVERS,
    NUM_VEHICLES,
    SOURCE_VEHICLE_ID,
    VEHICLE_A12_CORES,
    VEHICLE_A7_CORES,
    V2V_REACHABLE_VEHICLE_IDS,
)
from .v3_settings import V3_DATASET_VERSION
from .v3_simulation import PROCESSOR_A12, PROCESSOR_A7, V3ScenarioState


V3_CONTENTION_EDGE_FEATURE_NAMES: Tuple[str, ...] = (
    "source_gang_ratio",
    "target_gang_ratio",
    "combined_gang_ratio",
    "blocking_shared_pool_ratio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edge_index(pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _safe_scale(value: float) -> float:
    value = float(value)
    return value if abs(value) > 1e-12 else 1.0


def _candidate_relation_exists(
    task_id: int,
    resource_slot: int,
    gang_size: int,
) -> bool:
    del task_id
    if resource_slot < NUM_VEHICLES:
        if resource_slot == SOURCE_VEHICLE_ID:
            reachable = True
        else:
            reachable = resource_slot in {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}
        if not reachable:
            return False
        return bool(VEHICLE_A7_CORES >= gang_size or VEHICLE_A12_CORES >= gang_size)

    edge_id = resource_slot - NUM_VEHICLES
    if edge_id < 0 or edge_id >= NUM_EDGE_SERVERS:
        return False
    return bool(EDGE_A7_CORES >= gang_size or EDGE_A12_CORES >= gang_size)


def _candidate_raw_row(
    profile: Dict[str, object],
    width: int,
    free_a7: int,
    free_a12: int,
    total_a7: int,
    total_a12: int,
) -> List[float]:
    # Cost is a static property of the task-resource-processor pairing in v3.
    # Current availability changes only the ratio/feasibility channels.
    exec_time_a7 = float(profile["a7_time"]) / float(width)
    exec_time_a12 = float(profile["a12_time"]) / float(width)
    exec_energy_a7 = exec_time_a7 * float(width) * float(profile["a7_power"])
    exec_energy_a12 = exec_time_a12 * float(width) * float(profile["a12_power"])

    return [
        exec_time_a7,
        exec_time_a12,
        exec_energy_a7,
        exec_energy_a12,
        float(free_a7) / max(float(total_a7), 1.0),
        float(free_a12) / max(float(total_a12), 1.0),
        float(free_a7 >= width and total_a7 >= width),
        float(free_a12 >= width and total_a12 >= width),
    ]


def compute_v3_normalization() -> Dict[str, float]:
    source = load_legacy_native_task_graph()
    profiles = list(source["profiles"])

    deadline_scale = max(float(p["deadline"]) for p in profiles)
    task_execution_time_scale = max(
        max(float(p["a7_time"]), float(p["a12_time"])) for p in profiles
    )
    task_energy_scale = max(
        max(float(p["a7_energy"]), float(p["a12_energy"])) for p in profiles
    )

    candidate_times: List[float] = []
    candidate_energies: List[float] = []
    for task_id, profile in enumerate(profiles):
        width = int(profile["gang_size"])
        for resource_slot in range(NUM_VEHICLES + NUM_EDGE_SERVERS):
            if not _candidate_relation_exists(task_id, resource_slot, width):
                continue
            row = _candidate_raw_row(
                profile,
                width,
                total_a7=(VEHICLE_A7_CORES if resource_slot < NUM_VEHICLES else EDGE_A7_CORES),
                total_a12=(VEHICLE_A12_CORES if resource_slot < NUM_VEHICLES else EDGE_A12_CORES),
                free_a7=(VEHICLE_A7_CORES if resource_slot < NUM_VEHICLES else EDGE_A7_CORES),
                free_a12=(VEHICLE_A12_CORES if resource_slot < NUM_VEHICLES else EDGE_A12_CORES),
            )
            candidate_times.extend(row[0:2])
            candidate_energies.extend(row[2:4])

    return {
        "gang_size_scale": float(MAX_GANG_SIZE),
        "deadline_scale": float(deadline_scale),
        "task_execution_time_scale": float(task_execution_time_scale),
        "task_energy_scale": float(task_energy_scale),
        "resource_core_scale": float(max(EDGE_A7_CORES, EDGE_A12_CORES)),
        "candidate_execution_time_scale": float(max(candidate_times)),
        "candidate_energy_scale": float(max(candidate_energies)),
    }


def _normalize_candidate_rows(
    rows: Sequence[Sequence[float]],
    normalization: Dict[str, float],
) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32).reshape(-1, len(CANDIDATE_EDGE_FEATURE_NAMES))
    if array.size == 0:
        return array
    array[:, 0:2] /= _safe_scale(normalization["candidate_execution_time_scale"])
    array[:, 2:4] /= _safe_scale(normalization["candidate_energy_scale"])
    return array


def _static_candidate_pairs(
    profiles: Sequence[Dict[str, object]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    local: List[Tuple[int, int]] = []
    v2v: List[Tuple[int, int]] = []
    v2i: List[Tuple[int, int]] = []
    reachable = {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}

    for task_id, profile in enumerate(profiles):
        width = int(profile["gang_size"])
        if _candidate_relation_exists(task_id, SOURCE_VEHICLE_ID, width):
            local.append((task_id, SOURCE_VEHICLE_ID))
        for vehicle_id in sorted(reachable):
            if _candidate_relation_exists(task_id, vehicle_id, width):
                v2v.append((task_id, vehicle_id))
        for edge_id in range(NUM_EDGE_SERVERS):
            resource_slot = NUM_VEHICLES + edge_id
            if _candidate_relation_exists(task_id, resource_slot, width):
                v2i.append((task_id, edge_id))
    return local, v2v, v2i


def _contention_edges(
    state: V3ScenarioState,
    profiles: Sequence[Dict[str, object]],
    local_pairs: Sequence[Tuple[int, int]],
    v2v_pairs: Sequence[Tuple[int, int]],
    v2i_pairs: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    # Build static task -> resource-slot candidate sets.
    task_resources: Dict[int, List[int]] = {i: [] for i in range(len(profiles))}
    for task_id, vehicle_id in list(local_pairs) + list(v2v_pairs):
        task_resources[int(task_id)].append(int(vehicle_id))
    for task_id, edge_id in v2i_pairs:
        task_resources[int(task_id)].append(NUM_VEHICLES + int(edge_id))

    ready = [int(x) for x in np.flatnonzero(state.ready).tolist()]
    pairs: List[Tuple[int, int]] = []
    attrs: List[List[float]] = []

    for pos, task_a in enumerate(ready):
        gang_a = int(profiles[task_a]["gang_size"])
        resources_a = set(task_resources[task_a])
        for task_b in ready[pos + 1 :]:
            gang_b = int(profiles[task_b]["gang_size"])
            shared = sorted(resources_a.intersection(task_resources[task_b]))
            feasible_shared_pools = 0
            blocking_pools = 0

            for resource_slot in shared:
                for processor_index in (PROCESSOR_A7, PROCESSOR_A12):
                    free = int(
                        state.free_a7[resource_slot]
                        if processor_index == PROCESSOR_A7
                        else state.free_a12[resource_slot]
                    )
                    total = int(
                        state.total_a7[resource_slot]
                        if processor_index == PROCESSOR_A7
                        else state.total_a12[resource_slot]
                    )
                    if total < gang_a or total < gang_b:
                        continue
                    if free >= gang_a and free >= gang_b:
                        feasible_shared_pools += 1
                        if gang_a + gang_b > free:
                            blocking_pools += 1

            if blocking_pools <= 0:
                continue

            ratio = blocking_pools / max(feasible_shared_pools, 1)
            combined = min((gang_a + gang_b) / (2.0 * MAX_GANG_SIZE), 1.0)

            pairs.append((task_a, task_b))
            attrs.append([
                gang_a / MAX_GANG_SIZE,
                gang_b / MAX_GANG_SIZE,
                combined,
                ratio,
            ])
            pairs.append((task_b, task_a))
            attrs.append([
                gang_b / MAX_GANG_SIZE,
                gang_a / MAX_GANG_SIZE,
                combined,
                ratio,
            ])

    return pairs, np.asarray(attrs, dtype=np.float32).reshape(-1, len(V3_CONTENTION_EDGE_FEATURE_NAMES))


def build_v3_snapshot(
    state: V3ScenarioState,
    decision_epoch_index: int,
    event_context: Dict[str, object],
    split: str,
    normalization: Dict[str, float] | None = None,
) -> Tuple[HeteroData, Dict[str, object]]:
    source = load_legacy_native_task_graph()
    dag: nx.DiGraph = source["dag"]
    profiles: List[Dict[str, object]] = list(source["profiles"])
    dependencies = [(int(u), int(v)) for u, v in source["dependencies"]]

    if normalization is None:
        normalization = compute_v3_normalization()

    if len(profiles) != len(state.ready):
        raise ValueError("v3 state/task-profile size mismatch.")

    local_pairs, v2v_pairs, v2i_pairs = _static_candidate_pairs(profiles)

    task_rows: List[List[float]] = []
    for task_id, profile in enumerate(profiles):
        task_rows.append([
            int(profile["gang_size"]) / _safe_scale(normalization["gang_size_scale"]),
            float(profile["deadline"]) / _safe_scale(normalization["deadline_scale"]),
            float(profile["a7_time"]) / _safe_scale(normalization["task_execution_time_scale"]),
            float(profile["a12_time"]) / _safe_scale(normalization["task_execution_time_scale"]),
            float(profile["a7_energy"]) / _safe_scale(normalization["task_energy_scale"]),
            float(profile["a12_energy"]) / _safe_scale(normalization["task_energy_scale"]),
            float(state.ready[task_id]),
            float(state.running[task_id]),
            float(state.completed[task_id]),
        ])

    resource_core_scale = _safe_scale(normalization["resource_core_scale"])
    vehicle_rows: List[List[float]] = []
    for vehicle_id in range(NUM_VEHICLES):
        vehicle_rows.append([
            VEHICLE_A7_CORES / resource_core_scale,
            VEHICLE_A12_CORES / resource_core_scale,
            float(state.free_a7[vehicle_id]) / VEHICLE_A7_CORES,
            float(state.free_a12[vehicle_id]) / VEHICLE_A12_CORES,
        ])

    edge_rows: List[List[float]] = []
    for edge_id in range(NUM_EDGE_SERVERS):
        slot = NUM_VEHICLES + edge_id
        edge_rows.append([
            EDGE_A7_CORES / resource_core_scale,
            EDGE_A12_CORES / resource_core_scale,
            float(state.free_a7[slot]) / EDGE_A7_CORES,
            float(state.free_a12[slot]) / EDGE_A12_CORES,
        ])

    local_raw: List[List[float]] = []
    for task_id, vehicle_id in local_pairs:
        profile = profiles[task_id]
        local_raw.append(_candidate_raw_row(
            profile,
            int(profile["gang_size"]),
            int(state.free_a7[vehicle_id]),
            int(state.free_a12[vehicle_id]),
            VEHICLE_A7_CORES,
            VEHICLE_A12_CORES,
        ))

    v2v_raw: List[List[float]] = []
    for task_id, vehicle_id in v2v_pairs:
        profile = profiles[task_id]
        v2v_raw.append(_candidate_raw_row(
            profile,
            int(profile["gang_size"]),
            int(state.free_a7[vehicle_id]),
            int(state.free_a12[vehicle_id]),
            VEHICLE_A7_CORES,
            VEHICLE_A12_CORES,
        ))

    v2i_raw: List[List[float]] = []
    for task_id, edge_id in v2i_pairs:
        slot = NUM_VEHICLES + edge_id
        profile = profiles[task_id]
        v2i_raw.append(_candidate_raw_row(
            profile,
            int(profile["gang_size"]),
            int(state.free_a7[slot]),
            int(state.free_a12[slot]),
            EDGE_A7_CORES,
            EDGE_A12_CORES,
        ))

    local_attr = _normalize_candidate_rows(local_raw, normalization)
    v2v_attr = _normalize_candidate_rows(v2v_raw, normalization)
    v2i_attr = _normalize_candidate_rows(v2i_raw, normalization)
    contention_pairs, contention_attr = _contention_edges(
        state, profiles, local_pairs, v2v_pairs, v2i_pairs
    )

    data = HeteroData()
    data.dataset_version = V3_DATASET_VERSION
    data.dataset_split = str(split)
    data.scenario_id = int(state.scenario_id)
    data.scenario_seed = int(state.seed)
    data.behavior_policy = str(state.behavior_policy)
    data.decision_epoch_index = int(decision_epoch_index)
    data.simulation_time = torch.tensor([float(state.simulation_time)], dtype=torch.float32)
    data.trigger_event = str(event_context.get("trigger_event", "unknown"))

    data["task"].x = torch.tensor(task_rows, dtype=torch.float32)
    data["task"].task_id = torch.arange(len(profiles), dtype=torch.long)
    data["task"].source_vehicle = torch.full((len(profiles),), SOURCE_VEHICLE_ID, dtype=torch.long)
    data["task"].gang_size = torch.tensor([int(p["gang_size"]) for p in profiles], dtype=torch.long)
    data["task"].deadline = torch.tensor([float(p["deadline"]) for p in profiles], dtype=torch.float32)
    data["task"].exec_time_a7 = torch.tensor([float(p["a7_time"]) for p in profiles], dtype=torch.float32)
    data["task"].exec_time_a12 = torch.tensor([float(p["a12_time"]) for p in profiles], dtype=torch.float32)
    data["task"].energy_a7 = torch.tensor([float(p["a7_energy"]) for p in profiles], dtype=torch.float32)
    data["task"].energy_a12 = torch.tensor([float(p["a12_energy"]) for p in profiles], dtype=torch.float32)
    data["task"].topological_layer = torch.tensor([int(p["topological_layer"]) for p in profiles], dtype=torch.long)
    data["task"].is_ready = torch.from_numpy(state.ready.astype(np.int64))
    data["task"].is_running = torch.from_numpy(state.running.astype(np.int64))
    data["task"].is_completed = torch.from_numpy(state.completed.astype(np.int64))
    data["task"].assigned_resource_slot = torch.from_numpy(state.assigned_resource_slot.copy())
    data["task"].assigned_processor_index = torch.from_numpy(state.assigned_processor_index.copy())
    data["task"].assigned_cores = torch.from_numpy(state.assigned_cores.copy())
    data["task"].start_time = torch.from_numpy(state.task_start_time.astype(np.float32))
    data["task"].finish_time = torch.from_numpy(state.task_finish_time.astype(np.float32))

    data["vehicle"].x = torch.tensor(vehicle_rows, dtype=torch.float32)
    data["vehicle"].node_id = torch.arange(NUM_VEHICLES, dtype=torch.long)
    data["vehicle"].total_a7_cores = torch.full((NUM_VEHICLES,), VEHICLE_A7_CORES, dtype=torch.long)
    data["vehicle"].total_a12_cores = torch.full((NUM_VEHICLES,), VEHICLE_A12_CORES, dtype=torch.long)
    data["vehicle"].scenario_capacity_a7 = torch.from_numpy(state.scenario_capacity_a7[:NUM_VEHICLES].copy())
    data["vehicle"].scenario_capacity_a12 = torch.from_numpy(state.scenario_capacity_a12[:NUM_VEHICLES].copy())
    data["vehicle"].available_a7_cores = torch.from_numpy(state.free_a7[:NUM_VEHICLES].copy())
    data["vehicle"].available_a12_cores = torch.from_numpy(state.free_a12[:NUM_VEHICLES].copy())

    edge_slice = slice(NUM_VEHICLES, NUM_VEHICLES + NUM_EDGE_SERVERS)
    data["edge_server"].x = torch.tensor(edge_rows, dtype=torch.float32)
    data["edge_server"].node_id = torch.arange(NUM_EDGE_SERVERS, dtype=torch.long)
    data["edge_server"].total_a7_cores = torch.full((NUM_EDGE_SERVERS,), EDGE_A7_CORES, dtype=torch.long)
    data["edge_server"].total_a12_cores = torch.full((NUM_EDGE_SERVERS,), EDGE_A12_CORES, dtype=torch.long)
    data["edge_server"].scenario_capacity_a7 = torch.from_numpy(state.scenario_capacity_a7[edge_slice].copy())
    data["edge_server"].scenario_capacity_a12 = torch.from_numpy(state.scenario_capacity_a12[edge_slice].copy())
    data["edge_server"].available_a7_cores = torch.from_numpy(state.free_a7[edge_slice].copy())
    data["edge_server"].available_a12_cores = torch.from_numpy(state.free_a12[edge_slice].copy())

    data[DAG_DEPENDENCY_RELATION].edge_index = _edge_index(dependencies)
    data[DAG_DEPENDENCY_RELATION].edge_attr = torch.empty((len(dependencies), 0), dtype=torch.float32)

    data[LOCAL_EXECUTION_RELATION].edge_index = _edge_index(local_pairs)
    data[LOCAL_EXECUTION_RELATION].edge_attr = torch.from_numpy(local_attr)
    data[LOCAL_EXECUTION_RELATION].edge_attr_raw = torch.tensor(local_raw, dtype=torch.float32)

    data[V2V_OFFLOADING_RELATION].edge_index = _edge_index(v2v_pairs)
    data[V2V_OFFLOADING_RELATION].edge_attr = torch.from_numpy(v2v_attr)
    data[V2V_OFFLOADING_RELATION].edge_attr_raw = torch.tensor(v2v_raw, dtype=torch.float32)

    data[V2I_OFFLOADING_RELATION].edge_index = _edge_index(v2i_pairs)
    data[V2I_OFFLOADING_RELATION].edge_attr = torch.from_numpy(v2i_attr)
    data[V2I_OFFLOADING_RELATION].edge_attr_raw = torch.tensor(v2i_raw, dtype=torch.float32)

    data[GANG_CONTENTION_RELATION].edge_index = _edge_index(contention_pairs)
    data[GANG_CONTENTION_RELATION].edge_attr = torch.from_numpy(contention_attr)

    metadata: Dict[str, object] = {
        "dataset_version": V3_DATASET_VERSION,
        "split": str(split),
        "scenario_id": int(state.scenario_id),
        "scenario_seed": int(state.seed),
        "behavior_policy": str(state.behavior_policy),
        "decision_epoch_index": int(decision_epoch_index),
        "simulation_time": float(state.simulation_time),
        "trigger_event": str(event_context.get("trigger_event", "unknown")),
        "num_ready_tasks": int(np.sum(state.ready)),
        "num_running_tasks": int(np.sum(state.running)),
        "num_completed_tasks": int(np.sum(state.completed)),
        "num_scheduled_tasks": int(np.sum(state.task_start_time >= 0.0)),
        "num_dependency_edges": len(dependencies),
        "num_local_edges": len(local_pairs),
        "num_v2v_edges": len(v2v_pairs),
        "num_v2i_edges": len(v2i_pairs),
        "num_contention_edges": len(contention_pairs),
        "dag_is_acyclic": bool(nx.is_directed_acyclic_graph(dag)),
    }
    return data, metadata


def v3_dataset_metadata(normalization: Dict[str, float] | None = None) -> Dict[str, object]:
    source = load_legacy_native_task_graph()
    profiles = list(source["profiles"])
    dependencies = list(source["dependencies"])
    if normalization is None:
        normalization = compute_v3_normalization()

    local_pairs, v2v_pairs, v2i_pairs = _static_candidate_pairs(profiles)
    return {
        "dataset_name": "Legacy-native Dynamic MAGNET v3",
        "dataset_version": V3_DATASET_VERSION,
        "purpose": "event-driven multi-snapshot, multi-scenario dataset for the next muxGNN stage",
        "source": {
            "legacy_dataset": str(LEGACY_DATASET_PATH.relative_to(LEGACY_DATASET_PATH.parents[2])),
            "legacy_dataset_sha256": _sha256(LEGACY_DATASET_PATH),
            "dagbench_used": False,
            "same_validated_legacy_dag_as_v2": True,
        },
        "graph": {
            "node_types": ["task", "vehicle", "edge_server"],
            "num_tasks": len(profiles),
            "num_vehicles": NUM_VEHICLES,
            "num_edge_servers": NUM_EDGE_SERVERS,
            "num_dependency_edges": len(dependencies),
            "static_candidate_edge_counts": {
                "local_execute_on": len(local_pairs),
                "v2v_offload_to": len(v2v_pairs),
                "v2i_offload_to": len(v2i_pairs),
            },
            "dynamic_relation": "contends_with",
        },
        "dynamic_state": {
            "task": ["is_ready", "is_running", "is_completed"],
            "resource": ["available_a7_ratio", "available_a12_ratio"],
            "candidate": ["available_ratio_a7", "available_ratio_a12", "feasible_a7", "feasible_a12"],
            "contention": "recomputed from currently-ready tasks and currently-free shared processor pools",
        },
        "candidate_topology_semantics": {
            "relation_existence": "static hardware capacity + controlled v2 reachability",
            "current_core_scarcity_does_not_remove_relation": True,
            "current_feasibility_is_an_edge_feature": True,
        },
        "feature_names": {
            "task_node": list(TASK_FEATURE_NAMES),
            "vehicle_node": list(RESOURCE_FEATURE_NAMES),
            "edge_server_node": list(RESOURCE_FEATURE_NAMES),
            "dependency_edge": list(DEPENDENCY_EDGE_FEATURE_NAMES),
            "candidate_edge": list(CANDIDATE_EDGE_FEATURE_NAMES),
            "contention_edge": list(V3_CONTENTION_EDGE_FEATURE_NAMES),
        },
        "normalization": {
            **{key: float(value) for key, value in normalization.items()},
            "scope": "fixed once from the static Legacy-native task/resource model and reused for every v3 scenario/snapshot",
            "per_snapshot_refit": False,
        },
        "time_semantics": {
            "unit": "canonical relative simulation time inherited from Legacy timing",
            "physical_seconds_claimed": False,
            "network_seconds_mixed_with_canonical_time": False,
        },
        "provenance": {
            "task_profiles_and_dag": "validated Legacy dataset",
            "scenario_resource_capacity": "controlled seeded synthetic background reservation",
            "runtime_resource_changes": "task start/completion only",
            "trajectory_actions": "independent non-learning behavior policies for state coverage",
        },
        "intentionally_omitted_v3": [
            "DAGBench",
            "mobility",
            "vehicle position",
            "distance",
            "V2V/V2I link rate",
            "transmission delay",
            "contact duration",
            "queue delay",
            "migration",
            "physical wall-clock seconds",
            "multi-DAG generalization",
        ],
        "limitations": [
            "All v3 scenarios share one validated 60-task DAG; this stage improves dynamic-state coverage, not topology generalization.",
            "Deadlines are still inherited from the current Legacy dataset and are not yet redesigned as an independent MAGNET deadline model.",
            "Scenario resource reservations are controlled synthetic states and are not measured vehicular background loads.",
        ],
    }
