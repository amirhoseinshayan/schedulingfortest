from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..graph.magnet_relation_layers import (
    DAG_DEPENDENCY_RELATION,
    GANG_CONTENTION_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2I_OFFLOADING_RELATION,
    V2V_OFFLOADING_RELATION,
)
from ..mobility3.settings import (
    EDGE_A12_CORES,
    EDGE_A7_CORES,
    MAX_GANG_SIZE,
    NUM_EDGE_SERVERS,
    NUM_VEHICLES,
    SOURCE_VEHICLE_ID,
    VEHICLE_A12_CORES,
    VEHICLE_A7_CORES,
    V2V_REACHABLE_VEHICLE_IDS,
)
from ..graph.v4_settings import (
    RAW_EXECUTION_TIME_UNIT,
    TIME_UNIT,
    V4_BENCHMARK_PROFILE_PATH,
    V4_DEADLINE_MODEL,
    V4_DEADLINE_SLACK_FACTOR,
    V4_LEGACY_DAG_PATH,
    V4_RELEASE_TIME_SEC,
)
from ..graph.v4_simulation import PROCESSOR_A12, PROCESSOR_A7, V4ScenarioState
from ..graph.v4_source import sha256
from .scaled_source import load_scaled_task_graph
from ..mobility3.settings import DATASET_VERSION as V4_DATASET_VERSION


V4_TASK_FEATURE_NAMES: Tuple[str, ...] = (
    "gang_size",
    "deadline_sec",
    "exec_time_a7_sec",
    "exec_time_a12_sec",
    "energy_a7_j",
    "energy_a12_j",
    "ready_time_sec",
    "is_ready",
    "is_running",
    "is_completed",
)

V4_RESOURCE_FEATURE_NAMES: Tuple[str, ...] = (
    "total_a7_cores",
    "total_a12_cores",
    "available_a7_ratio",
    "available_a12_ratio",
)

V4_CANDIDATE_EDGE_FEATURE_NAMES: Tuple[str, ...] = (
    "exec_time_a7_sec",
    "exec_time_a12_sec",
    "exec_energy_a7_j",
    "exec_energy_a12_j",
    "available_ratio_a7",
    "available_ratio_a12",
    "feasible_a7",
    "feasible_a12",
)

V4_CONTENTION_EDGE_FEATURE_NAMES: Tuple[str, ...] = (
    "source_gang_ratio",
    "target_gang_ratio",
    "combined_gang_ratio",
    "blocking_shared_pool_ratio",
)


def _safe_scale(value: float) -> float:
    value = float(value)
    return value if abs(value) > 1e-15 else 1.0


def _edge_index(pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _candidate_relation_exists(gang_size: int, resource_slot: int) -> bool:
    if resource_slot < NUM_VEHICLES:
        vehicle_id = int(resource_slot)
        reachable = (
            vehicle_id == SOURCE_VEHICLE_ID
            or vehicle_id in {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}
        )
        if not reachable:
            return False
        return bool(VEHICLE_A7_CORES >= gang_size or VEHICLE_A12_CORES >= gang_size)
    edge_id = resource_slot - NUM_VEHICLES
    if edge_id < 0 or edge_id >= NUM_EDGE_SERVERS:
        return False
    return bool(EDGE_A7_CORES >= gang_size or EDGE_A12_CORES >= gang_size)


def _candidate_raw_row(
    profile: Dict[str, object],
    free_a7: int,
    free_a12: int,
    total_a7: int,
    total_a12: int,
) -> List[float]:
    width = int(profile["gang_size"])
    # m_i controls only simultaneous core occupancy/feasibility.  The supplied
    # A7/A12 time is already the duration of the task on its required gang.
    exec_time_a7_sec = float(profile["a7_time_sec"])
    exec_time_a12_sec = float(profile["a12_time_sec"])
    exec_energy_a7_j = float(profile["a7_energy_j"])
    exec_energy_a12_j = float(profile["a12_energy_j"])
    return [
        exec_time_a7_sec,
        exec_time_a12_sec,
        exec_energy_a7_j,
        exec_energy_a12_j,
        float(free_a7) / max(float(total_a7), 1.0),
        float(free_a12) / max(float(total_a12), 1.0),
        float(free_a7 >= width and total_a7 >= width),
        float(free_a12 >= width and total_a12 >= width),
    ]


def _static_candidate_pairs(
    profiles: Sequence[Dict[str, object]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    local: List[Tuple[int, int]] = []
    v2v: List[Tuple[int, int]] = []
    v2i: List[Tuple[int, int]] = []
    reachable = {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}
    for task_id, profile in enumerate(profiles):
        width = int(profile["gang_size"])
        if _candidate_relation_exists(width, SOURCE_VEHICLE_ID):
            local.append((task_id, SOURCE_VEHICLE_ID))
        for vehicle_id in sorted(reachable):
            if _candidate_relation_exists(width, vehicle_id):
                v2v.append((task_id, vehicle_id))
        for edge_id in range(NUM_EDGE_SERVERS):
            slot = NUM_VEHICLES + edge_id
            if _candidate_relation_exists(width, slot):
                v2i.append((task_id, edge_id))
    return local, v2v, v2i


def compute_v4_normalization() -> Dict[str, float]:
    source = load_scaled_task_graph()
    profiles = list(source["profiles"])
    deadline_scale = max(float(p["deadline_sec"]) for p in profiles)
    task_execution_time_scale = max(
        max(float(p["a7_time_sec"]), float(p["a12_time_sec"])) for p in profiles
    )
    task_energy_scale = max(
        max(float(p["a7_energy_j"]), float(p["a12_energy_j"])) for p in profiles
    )
    ready_time_scale = max(float(source["workflow_deadline_sec"]), deadline_scale)

    candidate_times: List[float] = []
    candidate_energies: List[float] = []
    for profile in profiles:
        width = int(profile["gang_size"])
        for slot in range(NUM_VEHICLES + NUM_EDGE_SERVERS):
            if not _candidate_relation_exists(width, slot):
                continue
            total_a7 = VEHICLE_A7_CORES if slot < NUM_VEHICLES else EDGE_A7_CORES
            total_a12 = VEHICLE_A12_CORES if slot < NUM_VEHICLES else EDGE_A12_CORES
            row = _candidate_raw_row(profile, total_a7, total_a12, total_a7, total_a12)
            candidate_times.extend(row[0:2])
            candidate_energies.extend(row[2:4])

    return {
        "gang_size_scale": float(MAX_GANG_SIZE),
        "deadline_scale_sec": float(deadline_scale),
        "task_execution_time_scale_sec": float(task_execution_time_scale),
        "task_energy_scale_j": float(task_energy_scale),
        "ready_time_scale_sec": float(ready_time_scale),
        "resource_core_scale": float(max(EDGE_A7_CORES, EDGE_A12_CORES)),
        "candidate_execution_time_scale_sec": float(max(candidate_times)),
        "candidate_energy_scale_j": float(max(candidate_energies)),
    }


def _normalize_candidate_rows(
    rows: Sequence[Sequence[float]], normalization: Dict[str, float]
) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32).reshape(-1, len(V4_CANDIDATE_EDGE_FEATURE_NAMES))
    if array.size == 0:
        return array
    array[:, 0:2] /= _safe_scale(normalization["candidate_execution_time_scale_sec"])
    array[:, 2:4] /= _safe_scale(normalization["candidate_energy_scale_j"])
    return array


def _contention_edges(
    state: V4ScenarioState,
    profiles: Sequence[Dict[str, object]],
    local_pairs: Sequence[Tuple[int, int]],
    v2v_pairs: Sequence[Tuple[int, int]],
    v2i_pairs: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
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
            attrs.append([gang_a / MAX_GANG_SIZE, gang_b / MAX_GANG_SIZE, combined, ratio])
            pairs.append((task_b, task_a))
            attrs.append([gang_b / MAX_GANG_SIZE, gang_a / MAX_GANG_SIZE, combined, ratio])
    return pairs, np.asarray(attrs, dtype=np.float32).reshape(-1, len(V4_CONTENTION_EDGE_FEATURE_NAMES))


def build_v4_snapshot(
    state: V4ScenarioState,
    decision_epoch_index: int,
    event_context: Dict[str, object],
    split: str,
    normalization: Dict[str, float] | None = None,
    workload: Dict[str, object] | None = None,
) -> Tuple[HeteroData, Dict[str, object]]:
    source = workload if workload is not None else load_scaled_task_graph()
    dag: nx.DiGraph = source["dag"]
    profiles: List[Dict[str, object]] = list(source["profiles"])
    dependencies = [(int(u), int(v)) for u, v in source["dependencies"]]
    if normalization is None:
        normalization = compute_v4_normalization()

    local_pairs, v2v_pairs, v2i_pairs = _static_candidate_pairs(profiles)
    # Dynamic reachability: use only the physical state at this event timestamp.
    v2v_pairs = [(t,v) for t,v in v2v_pairs if v in state.v2v_connected]
    v2i_pairs = [(t,e) for t,e in v2i_pairs if e in state.edge_connected]

    task_rows: List[List[float]] = []
    for task_id, p in enumerate(profiles):
        ready_time = max(float(state.ready_time_sec[task_id]), 0.0)
        task_rows.append([
            int(p["gang_size"]) / _safe_scale(normalization["gang_size_scale"]),
            float(p["deadline_sec"]) / _safe_scale(normalization["deadline_scale_sec"]),
            float(p["a7_time_sec"]) / _safe_scale(normalization["task_execution_time_scale_sec"]),
            float(p["a12_time_sec"]) / _safe_scale(normalization["task_execution_time_scale_sec"]),
            float(p["a7_energy_j"]) / _safe_scale(normalization["task_energy_scale_j"]),
            float(p["a12_energy_j"]) / _safe_scale(normalization["task_energy_scale_j"]),
            ready_time / _safe_scale(normalization["ready_time_scale_sec"]),
            float(state.ready[task_id]),
            float(state.running[task_id]),
            float(state.completed[task_id]),
        ])

    resource_core_scale = _safe_scale(normalization["resource_core_scale"])
    vehicle_rows = [[
        VEHICLE_A7_CORES / resource_core_scale,
        VEHICLE_A12_CORES / resource_core_scale,
        float(state.free_a7[v]) / VEHICLE_A7_CORES,
        float(state.free_a12[v]) / VEHICLE_A12_CORES,
    ] for v in range(NUM_VEHICLES)]
    edge_rows = []
    for e in range(NUM_EDGE_SERVERS):
        slot = NUM_VEHICLES + e
        edge_rows.append([
            EDGE_A7_CORES / resource_core_scale,
            EDGE_A12_CORES / resource_core_scale,
            float(state.free_a7[slot]) / EDGE_A7_CORES,
            float(state.free_a12[slot]) / EDGE_A12_CORES,
        ])

    def rows_for(pairs: Sequence[Tuple[int, int]], edge_offset: int = 0) -> List[List[float]]:
        rows: List[List[float]] = []
        for task_id, local_id in pairs:
            slot = int(local_id) + int(edge_offset)
            total_a7 = VEHICLE_A7_CORES if slot < NUM_VEHICLES else EDGE_A7_CORES
            total_a12 = VEHICLE_A12_CORES if slot < NUM_VEHICLES else EDGE_A12_CORES
            rows.append(_candidate_raw_row(
                profiles[int(task_id)], int(state.free_a7[slot]), int(state.free_a12[slot]),
                total_a7, total_a12,
            ))
        return rows

    local_raw = rows_for(local_pairs)
    v2v_raw = rows_for(v2v_pairs)
    v2i_raw = rows_for(v2i_pairs, NUM_VEHICLES)
    contention_pairs, contention_attr = _contention_edges(
        state, profiles, local_pairs, v2v_pairs, v2i_pairs
    )

    data = HeteroData()
    data.dataset_version = V4_DATASET_VERSION
    data.dataset_split = str(split)
    data.scenario_id = int(state.scenario_id)
    data.scenario_seed = int(state.seed)
    data.dag_seed = int(source.get('dag_seed', -1))
    data.dag_fingerprint = str(source.get('dag_fingerprint', 'original_v4'))
    data.behavior_policy = str(state.behavior_policy)
    data.decision_epoch_index = int(decision_epoch_index)
    data.simulation_time_sec = torch.tensor([float(state.simulation_time_sec)], dtype=torch.float32)
    data.sumo_time_sec = torch.tensor([float(state.physical_time_sec)],dtype=torch.float32)
    # One existing global time channel tells PPO how much of the 600 s episode
    # has elapsed, without adding any feature dimension or changing relations.
    episode_start = float(event_context.get('episode_start_sec', state.physical_time_sec - state.simulation_time_sec))
    episode_duration = float(event_context.get('episode_duration_sec', 600.))
    data.episode_elapsed_fraction = float(np.clip((state.physical_time_sec-episode_start)/episode_duration, 0.,1.))
    data.dag_index = int(event_context.get('dag_index', 0))
    data.time_unit = TIME_UNIT
    data.energy_unit = "joule"
    data.time_scale_factor = float(source["time_scale_factor"])
    data.power_unit = "watt"
    data.trigger_event = str(event_context.get("trigger_event", "unknown"))
    data.link_events = getattr(state, "last_link_events", [])

    data["task"].x = torch.tensor(task_rows, dtype=torch.float32)
    data["task"].task_id = torch.arange(len(profiles), dtype=torch.long)
    data["task"].source_vehicle = torch.full((len(profiles),), SOURCE_VEHICLE_ID, dtype=torch.long)
    data["task"].gang_size = torch.tensor([int(p["gang_size"]) for p in profiles], dtype=torch.long)
    data["task"].release_time_sec = torch.tensor([float(p["release_time_sec"]) for p in profiles], dtype=torch.float32)
    data["task"].ready_time_sec = torch.from_numpy(state.ready_time_sec.astype(np.float32))
    data["task"].deadline_sec = torch.tensor([float(p["deadline_sec"]) for p in profiles], dtype=torch.float32)
    data["task"].reference_v4_exec_time_a7_ms = torch.tensor([float(p["reference_v4_a7_execution_time_ms"]) for p in profiles], dtype=torch.float32)
    data["task"].reference_v4_exec_time_a12_ms = torch.tensor([float(p["reference_v4_a12_execution_time_ms"]) for p in profiles], dtype=torch.float32)
    data["task"].reference_v4_deadline_sec = torch.tensor([float(p["reference_v4_deadline_sec"]) for p in profiles], dtype=torch.float32)
    data["task"].exec_time_a7_ms = torch.tensor([float(p["a7_execution_time_ms"]) for p in profiles], dtype=torch.float32)
    data["task"].exec_time_a12_ms = torch.tensor([float(p["a12_execution_time_ms"]) for p in profiles], dtype=torch.float32)
    data["task"].exec_time_a7_sec = torch.tensor([float(p["a7_time_sec"]) for p in profiles], dtype=torch.float32)
    data["task"].exec_time_a12_sec = torch.tensor([float(p["a12_time_sec"]) for p in profiles], dtype=torch.float32)
    data["task"].gang_exec_time_a7_sec = torch.tensor([float(p["a7_gang_time_sec"]) for p in profiles], dtype=torch.float32)
    data["task"].gang_exec_time_a12_sec = torch.tensor([float(p["a12_gang_time_sec"]) for p in profiles], dtype=torch.float32)
    data["task"].power_a7_w = torch.tensor([float(p["a7_power_w"]) for p in profiles], dtype=torch.float32)
    data["task"].power_a12_w = torch.tensor([float(p["a12_power_w"]) for p in profiles], dtype=torch.float32)
    data["task"].energy_a7_j = torch.tensor([float(p["a7_energy_j"]) for p in profiles], dtype=torch.float32)
    data["task"].energy_a12_j = torch.tensor([float(p["a12_energy_j"]) for p in profiles], dtype=torch.float32)
    data["task"].period_placeholder = torch.tensor([float(p["period_placeholder"]) for p in profiles], dtype=torch.float32)
    data["task"].topological_layer = torch.tensor([int(p["topological_layer"]) for p in profiles], dtype=torch.long)
    data["task"].is_ready = torch.from_numpy(state.ready.astype(np.int64))
    data["task"].is_running = torch.from_numpy(state.running.astype(np.int64))
    data["task"].is_completed = torch.from_numpy(state.completed.astype(np.int64))
    data["task"].assigned_resource_slot = torch.from_numpy(state.assigned_resource_slot.copy())
    data["task"].assigned_processor_index = torch.from_numpy(state.assigned_processor_index.copy())
    data["task"].assigned_cores = torch.from_numpy(state.assigned_cores.copy())
    data["task"].start_time_sec = torch.from_numpy(state.task_start_time_sec.astype(np.float32))
    data["task"].finish_time_sec = torch.from_numpy(state.task_finish_time_sec.astype(np.float32))

    data["vehicle"].x = torch.tensor(vehicle_rows, dtype=torch.float32)
    data["vehicle"].node_id = torch.arange(NUM_VEHICLES, dtype=torch.long)
    data["vehicle"].position_xy = torch.tensor([
        [state.mobility["vehicles"][f"V{i}"]["x"],state.mobility["vehicles"][f"V{i}"]["y"]]
        for i in range(NUM_VEHICLES)],dtype=torch.float32)
    data["vehicle"].speed_mps = torch.tensor([
        state.mobility["vehicles"][f"V{i}"]["speed"] for i in range(NUM_VEHICLES)],dtype=torch.float32)
    data["vehicle"].total_a7_cores = torch.full((NUM_VEHICLES,), VEHICLE_A7_CORES, dtype=torch.long)
    data["vehicle"].total_a12_cores = torch.full((NUM_VEHICLES,), VEHICLE_A12_CORES, dtype=torch.long)
    data["vehicle"].scenario_capacity_a7 = torch.from_numpy(state.scenario_capacity_a7[:NUM_VEHICLES].copy())
    data["vehicle"].scenario_capacity_a12 = torch.from_numpy(state.scenario_capacity_a12[:NUM_VEHICLES].copy())
    data["vehicle"].available_a7_cores = torch.from_numpy(state.free_a7[:NUM_VEHICLES].copy())
    data["vehicle"].available_a12_cores = torch.from_numpy(state.free_a12[:NUM_VEHICLES].copy())

    edge_slice = slice(NUM_VEHICLES, NUM_VEHICLES + NUM_EDGE_SERVERS)
    data["edge_server"].x = torch.tensor(edge_rows, dtype=torch.float32)
    data["edge_server"].node_id = torch.arange(NUM_EDGE_SERVERS, dtype=torch.long)
    data["edge_server"].position_xy = torch.tensor([
        [float(s["x"]),float(s["y"])] for s in state._mobility_servers],dtype=torch.float32)
    data["edge_server"].total_a7_cores = torch.full((NUM_EDGE_SERVERS,), EDGE_A7_CORES, dtype=torch.long)
    data["edge_server"].total_a12_cores = torch.full((NUM_EDGE_SERVERS,), EDGE_A12_CORES, dtype=torch.long)
    data["edge_server"].scenario_capacity_a7 = torch.from_numpy(state.scenario_capacity_a7[edge_slice].copy())
    data["edge_server"].scenario_capacity_a12 = torch.from_numpy(state.scenario_capacity_a12[edge_slice].copy())
    data["edge_server"].available_a7_cores = torch.from_numpy(state.free_a7[edge_slice].copy())
    data["edge_server"].available_a12_cores = torch.from_numpy(state.free_a12[edge_slice].copy())

    data[DAG_DEPENDENCY_RELATION].edge_index = _edge_index(dependencies)
    data[DAG_DEPENDENCY_RELATION].edge_attr = torch.empty((len(dependencies), 0), dtype=torch.float32)

    for relation, pairs, raw in (
        (LOCAL_EXECUTION_RELATION, local_pairs, local_raw),
        (V2V_OFFLOADING_RELATION, v2v_pairs, v2v_raw),
        (V2I_OFFLOADING_RELATION, v2i_pairs, v2i_raw),
    ):
        data[relation].edge_index = _edge_index(pairs)
        data[relation].edge_attr_raw = torch.tensor(raw, dtype=torch.float32)
        data[relation].edge_attr = torch.from_numpy(_normalize_candidate_rows(raw, normalization))

    data[GANG_CONTENTION_RELATION].edge_index = _edge_index(contention_pairs)
    data[GANG_CONTENTION_RELATION].edge_attr = torch.from_numpy(contention_attr)

    metadata = {
        "dataset_version": V4_DATASET_VERSION,
        "runtime_deadline_energy_scale": float(source["time_scale_factor"]),
        "split": str(split),
        "scenario_id": int(state.scenario_id),
        "scenario_seed": int(state.seed),
        "dag_seed": int(source.get('dag_seed', -1)),
        "dag_fingerprint": str(source.get('dag_fingerprint', 'original_v4')),
        "behavior_policy": str(state.behavior_policy),
        "decision_epoch_index": int(decision_epoch_index),
        "simulation_time_sec": float(state.simulation_time_sec),
        "trigger_event": str(event_context.get("trigger_event", "unknown")),
        "num_ready_tasks": int(np.sum(state.ready)),
        "num_running_tasks": int(np.sum(state.running)),
        "num_completed_tasks": int(np.sum(state.completed)),
        "num_scheduled_tasks": int(np.sum(state.task_start_time_sec >= 0.0)),
        "num_dependency_edges": len(dependencies),
        "num_local_edges": len(local_pairs),
        "num_v2v_edges": len(v2v_pairs),
        "num_v2i_edges": len(v2i_pairs),
        "sumo_time_sec": float(state.physical_time_sec),
        "link_events": getattr(state, "last_link_events", []),
        "vehicle_positions": {f"V{i}": [state.mobility["vehicles"][f"V{i}"]["x"],state.mobility["vehicles"][f"V{i}"]["y"]] for i in range(NUM_VEHICLES)},
        "reachable_vehicles": sorted(state.v2v_connected),
        "reachable_edge_servers": sorted(state.edge_connected),
        "num_contention_edges": len(contention_pairs),
        "dag_is_acyclic": bool(nx.is_directed_acyclic_graph(dag)),
    }
    return data, metadata


def v4_dataset_metadata(normalization: Dict[str, float] | None = None) -> Dict[str, object]:
    source = load_scaled_task_graph()
    profiles = list(source["profiles"])
    dependencies = list(source["dependencies"])
    if normalization is None:
        normalization = compute_v4_normalization()
    local_pairs, v2v_pairs, v2i_pairs = _static_candidate_pairs(profiles)
    return {
        "dataset_name": "Derived v4 Mobility3 5x Physical Duration Scenario",
        "dataset_version": V4_DATASET_VERSION,
        "runtime_deadline_energy_scale": float(source["time_scale_factor"]),
        "purpose": "finalized physical-time, event-driven multi-snapshot dataset for fair muxGNN/PPO and HEFT integration",
        "source": {
            "legacy_dag_file": str(V4_LEGACY_DAG_PATH),
            "benchmark_profile_file": str(V4_BENCHMARK_PROFILE_PATH),
            "legacy_dag_sha256": sha256(V4_LEGACY_DAG_PATH),
            "benchmark_profiles_sha256": sha256(V4_BENCHMARK_PROFILE_PATH),
            "same_validated_60_task_126_edge_dag_as_v2_v3": True,
            "dagbench_used": False,
        },
        "physical_units": {
            "raw_execution_time": RAW_EXECUTION_TIME_UNIT,
            "stored_scheduling_time": TIME_UNIT,
            "power": "watt",
            "energy": "joule",
            "conversion": "derived_duration_sec = reference_v4_execution_time_ms / 1000 * time_scale_factor",
            "time_scale_factor": float(source["time_scale_factor"]),
            "raw_benchmark_files_unchanged": True,
        },
        "period_semantics": {
            "source_value": 1,
            "physical_period_claimed": False,
            "meaning": "placeholder chosen so utilization = execution_time / period numerically equals execution_time",
            "used_as_arrival_period": False,
            "used_as_release_time": False,
        },
        "release_ready_semantics": {
            "workload_model": "single DAG known completely at design time",
            "external_release_time_sec": float(V4_RELEASE_TIME_SEC),
            "all_tasks_release_at_zero": True,
            "root_ready_time_sec": 0.0,
            "non_root_ready_time_formula": "max(release_time_sec, max(predecessor_finish_time_sec))",
            "release_time_and_ready_time_are_distinct": True,
        },
        "deadline_model": {
            "type": V4_DEADLINE_MODEL,
            "independent_of_heft_and_ppo": True,
            "slack_factor": float(V4_DEADLINE_SLACK_FACTOR),
            "fastest_task_runtime": "w_i = min(A7_execution_time_sec, A12_execution_time_sec); no division by m_i",
            "dependency_earliest_finish": "EF_i = max(EF_pred) + fastest_task_runtime_i",
            "critical_path_sec": float(source["critical_path_sec"]),
            "workflow_deadline_sec": float(source["workflow_deadline_sec"]),
            "reference_v4_workflow_deadline_sec": float(source["reference_v4_workflow_deadline_sec"]),
            "task_deadline_formula": "D_i = release + (EF_i/CP) * (D_workflow-release) = 1.15 * EF_i",
            "legacy_heft_like_deadline_retained_only_for_audit": True,
        },
        "graph": {
            "node_types": ["task", "vehicle", "edge_server"],
            "num_tasks": len(profiles),
            "num_vehicles": NUM_VEHICLES,
            "num_edge_servers": NUM_EDGE_SERVERS,
            "num_dependency_edges": len(dependencies),
            "num_topological_layers": len(source["layers"]),
            "static_candidate_edge_counts": {
                "local_execute_on": len(local_pairs),
                "v2v_offload_to": len(v2v_pairs),
                "v2i_offload_to": len(v2i_pairs),
            },
            "dynamic_relation": "contends_with",
        },
        "feature_names": {
            "task_node": list(V4_TASK_FEATURE_NAMES),
            "vehicle_node": list(V4_RESOURCE_FEATURE_NAMES),
            "edge_server_node": list(V4_RESOURCE_FEATURE_NAMES),
            "dependency_edge": [],
            "candidate_edge": list(V4_CANDIDATE_EDGE_FEATURE_NAMES),
            "contention_edge": list(V4_CONTENTION_EDGE_FEATURE_NAMES),
        },
        "normalization": {
            **{k: float(v) for k, v in normalization.items()},
            "per_snapshot_refit": False,
            "label_or_outcome_statistics_used": False,
            "scope": "fixed source-task/resource scales reused across all scenarios and snapshots",
        },
        "compute_model": {
            "processor_types": ["A7", "A12"],
            "gang_size_range": [1, MAX_GANG_SIZE],
            "mixed_gang": False,
            "runtime_model": "measured task duration: Runtime_A7=T_A7 and Runtime_A12=T_A12; m_i only reserves simultaneous cores",
            "energy_model": "E_derived=T_derived*P_avg_total (reference average power held constant); no m_i rescaling",
            "time_scaling_is_synthetic": True,
            "time_scale_factor": float(source["time_scale_factor"]),
        },
        "provenance": {
            "execution_time_unit": "supervisor-confirmed millisecond interpretation of original raw execution values",
            "task_profiles_and_dag": "existing validated project artifacts",
            "resource_scenario_capacity": "controlled seeded synthetic background reservations",
            "trajectory_actions": "non-learning policies only for offline state coverage",
        },
        "intentionally_not_invented": [
            "mobility trace", "vehicle coordinates", "physical distance", "link bandwidth",
            "task data size", "V2V/V2I transmission delay", "contact duration", "queue delay",
            "migration delay",
        ],
        "limitations": [
            "All scenarios share one 60-task DAG, so this split tests dynamic resource-state generalization rather than unseen-DAG topology generalization.",
            "Scenario background reservations are controlled synthetic states, not measured vehicular background loads.",
            "The deadline model is synthetic but scheduler-independent and explicitly documented.",
        ],
    }
