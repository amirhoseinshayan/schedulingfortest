from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from legacy_gat.create_dag import CreateDAG

from .magnet_relation_layers import (
    DAG_DEPENDENCY_RELATION,
    GANG_CONTENTION_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2I_OFFLOADING_RELATION,
    V2V_OFFLOADING_RELATION,
)
from .settings import (
    DATASET_VERSION,
    EDGE_A12_CORES,
    EDGE_A7_CORES,
    LEGACY_DATASET_PATH,
    LEGACY_ROW_INDEX,
    MAX_GANG_SIZE,
    NUM_EDGE_SERVERS,
    NUM_VEHICLES,
    SEED,
    SOURCE_VEHICLE_ID,
    VEHICLE_A12_CORES,
    VEHICLE_A7_CORES,
    VEHICLE_AVAILABLE_A7,
    VEHICLE_AVAILABLE_A12,
    V2V_REACHABLE_VEHICLE_IDS,
)


# ============================================================
# Feature schemas for Legacy-native Simple MAGNET v2
# ============================================================

TASK_FEATURE_NAMES: Tuple[str, ...] = (
    "gang_size",
    "deadline",
    "exec_time_a7",
    "exec_time_a12",
    "energy_a7",
    "energy_a12",
    "is_ready",
    "is_running",
    "is_completed",
)

RESOURCE_FEATURE_NAMES: Tuple[str, ...] = (
    "total_a7_cores",
    "total_a12_cores",
    "available_a7_ratio",
    "available_a12_ratio",
)

# Dependency is topology-only in v2.  No synthetic data-size/edge weight is used.
DEPENDENCY_EDGE_FEATURE_NAMES: Tuple[str, ...] = tuple()

# All task-resource relations use the same compute-only semantics in v2.
CANDIDATE_EDGE_FEATURE_NAMES: Tuple[str, ...] = (
    "exec_time_a7",
    "exec_time_a12",
    "exec_energy_a7",
    "exec_energy_a12",
    "available_ratio_a7",
    "available_ratio_a12",
    "feasible_a7",
    "feasible_a12",
)

CONTENTION_EDGE_FEATURE_NAMES: Tuple[str, ...] = (
    "source_gang_ratio",
    "target_gang_ratio",
    "combined_gang_ratio",
    "vehicle_pool_contention",
)


# ============================================================
# Helpers
# ============================================================


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


def _empty_edge_attr(num_edges: int) -> torch.Tensor:
    """Return an explicit zero-width feature tensor for topology-only edges."""
    return torch.empty((int(num_edges), 0), dtype=torch.float32)


def _safe_scale(value: float) -> float:
    value = float(value)
    return value if abs(value) > 1e-8 else 1.0


def _normalize(values: Sequence[float], scale: float) -> List[float]:
    denominator = _safe_scale(scale)
    return [float(value) / denominator for value in values]


def _topological_layers(dag: nx.DiGraph) -> Tuple[List[List[int]], Dict[int, int]]:
    layers = [list(layer) for layer in nx.topological_generations(dag)]
    level_by_task: Dict[int, int] = {}
    for level, layer in enumerate(layers):
        for task_id in layer:
            level_by_task[int(task_id)] = int(level)
    return layers, level_by_task


# ============================================================
# Legacy-native task source
# ============================================================


def load_legacy_native_task_graph() -> Dict[str, object]:
    """
    Load the exact validated Legacy DAG and A7/A12 task profiles.

    No DAGBench workflow, DAGBench cost, DAGBench data-size, or topology fusion
    is used in this dataset path.
    """

    legacy = CreateDAG(LEGACY_DATASET_PATH, LEGACY_ROW_INDEX)
    dag = legacy.graph.copy()

    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Legacy source graph is not a DAG.")

    if legacy.num_subtasks != 60:
        raise ValueError(
            f"Simple MAGNET v2 expects the validated 60-task Legacy DAG; "
            f"found {legacy.num_subtasks}."
        )

    dependencies = [(int(u), int(v)) for u, v in legacy.edges]
    layers, level_by_task = _topological_layers(dag)

    profiles: List[Dict[str, float | int | str]] = []
    for task_id in range(legacy.num_subtasks):
        node = legacy.graph.nodes[task_id]
        profiles.append(
            {
                "task_id": task_id,
                "benchmark_name": str(node.get("benchmark_name", f"task_{task_id}")),
                "gang_size": int(legacy.m_i_list[task_id]),
                "deadline": float(legacy.deadlines[task_id]),
                "a7_time": float(legacy.a7_times[task_id]),
                "a12_time": float(legacy.a12_times[task_id]),
                "a7_power": float(legacy.a7_powers[task_id]),
                "a12_power": float(legacy.a12_powers[task_id]),
                "a7_energy": float(legacy.a7_energy[task_id]),
                "a12_energy": float(legacy.a12_energy[task_id]),
                "period": float(legacy.periods[task_id]),
                "topological_layer": int(level_by_task[task_id]),
            }
        )

    roots = [int(node) for node in dag.nodes if dag.in_degree(node) == 0]
    sinks = [int(node) for node in dag.nodes if dag.out_degree(node) == 0]

    return {
        "legacy": legacy,
        "dag": dag,
        "dependencies": dependencies,
        "profiles": profiles,
        "layers": layers,
        "level_by_task": level_by_task,
        "roots": roots,
        "sinks": sinks,
    }


# Backward-compatible helper retained for old code that only asks for profiles.
def load_project_task_profiles(num_tasks: int | None = None) -> List[Dict[str, object]]:
    source = load_legacy_native_task_graph()
    profiles = list(source["profiles"])
    if num_tasks is None:
        return profiles
    if int(num_tasks) > len(profiles):
        raise ValueError(
            f"Legacy dataset has {len(profiles)} task profiles, requested {num_tasks}."
        )
    return profiles[: int(num_tasks)]


# ============================================================
# Simple compute-resource model
# ============================================================


def generate_resources(seed: int = SEED) -> Tuple[List[Dict[str, float | int]], List[Dict[str, float | int]]]:
    """
    Create deterministic but heterogeneous resources for the simple v2 graph.

    v2 changes only the initial free-core state: the six vehicles now expose
    different A7/A12 availability while total hardware, compute_scale, and
    power_scale stay fixed.  There is still no mobility, queue delay, random
    background process, compute jitter, or power jitter.
    """

    del seed  # v2 resource state is controlled and reproducible, not random.

    if len(VEHICLE_AVAILABLE_A7) != NUM_VEHICLES:
        raise ValueError("VEHICLE_AVAILABLE_A7 must define one value per vehicle.")
    if len(VEHICLE_AVAILABLE_A12) != NUM_VEHICLES:
        raise ValueError("VEHICLE_AVAILABLE_A12 must define one value per vehicle.")

    vehicles: List[Dict[str, float | int]] = []
    for vehicle_id in range(NUM_VEHICLES):
        available_a7 = int(VEHICLE_AVAILABLE_A7[vehicle_id])
        available_a12 = int(VEHICLE_AVAILABLE_A12[vehicle_id])
        if not 0 <= available_a7 <= VEHICLE_A7_CORES:
            raise ValueError(f"Invalid A7 availability for vehicle {vehicle_id}.")
        if not 0 <= available_a12 <= VEHICLE_A12_CORES:
            raise ValueError(f"Invalid A12 availability for vehicle {vehicle_id}.")

        vehicles.append(
            {
                "id": vehicle_id,
                "a7_cores": VEHICLE_A7_CORES,
                "a12_cores": VEHICLE_A12_CORES,
                "available_a7": available_a7,
                "available_a12": available_a12,
                "compute_scale": 1.0,
                "power_scale": 1.0,
            }
        )

    edge_servers: List[Dict[str, float | int]] = []
    for edge_id in range(NUM_EDGE_SERVERS):
        edge_servers.append(
            {
                "id": edge_id,
                "a7_cores": EDGE_A7_CORES,
                "a12_cores": EDGE_A12_CORES,
                "available_a7": EDGE_A7_CORES,
                "available_a12": EDGE_A12_CORES,
                "compute_scale": 1.0,
                "power_scale": 1.0,
            }
        )

    return vehicles, edge_servers


def _candidate_features_raw(
    profile: Dict[str, float | int | str],
    resource: Dict[str, float | int],
) -> List[float]:
    """Compute relation features with the Legacy perfect-linear-speedup model."""

    width = int(profile["gang_size"])
    if not 1 <= width <= MAX_GANG_SIZE:
        raise ValueError(f"Invalid gang size: {width}")

    compute_scale = float(resource["compute_scale"])
    power_scale = float(resource["power_scale"])

    available_a7 = int(resource["available_a7"])
    available_a12 = int(resource["available_a12"])

    feasible_a7 = float(available_a7 >= width)
    feasible_a12 = float(available_a12 >= width)

    exec_time_a7 = 0.0
    exec_time_a12 = 0.0
    exec_energy_a7 = 0.0
    exec_energy_a12 = 0.0

    if feasible_a7:
        exec_time_a7 = float(profile["a7_time"]) / (width * compute_scale)
        exec_energy_a7 = (
            exec_time_a7 * width * float(profile["a7_power"]) * power_scale
        )

    if feasible_a12:
        exec_time_a12 = float(profile["a12_time"]) / (width * compute_scale)
        exec_energy_a12 = (
            exec_time_a12 * width * float(profile["a12_power"]) * power_scale
        )

    return [
        exec_time_a7,
        exec_time_a12,
        exec_energy_a7,
        exec_energy_a12,
        available_a7 / max(int(resource["a7_cores"]), 1),
        available_a12 / max(int(resource["a12_cores"]), 1),
        feasible_a7,
        feasible_a12,
    ]


def _resource_is_feasible(
    profile: Dict[str, float | int | str],
    resource: Dict[str, float | int],
) -> bool:
    """True when at least one homogeneous processor pool can fit the gang."""

    width = int(profile["gang_size"])
    return (
        int(resource["available_a7"]) >= width
        or int(resource["available_a12"]) >= width
    )


def _normalize_candidate_rows(
    raw_rows: np.ndarray,
    time_scale: float,
    energy_scale: float,
) -> np.ndarray:
    rows = np.asarray(raw_rows, dtype=np.float32).copy()
    if rows.size == 0:
        return rows.reshape(0, len(CANDIDATE_EDGE_FEATURE_NAMES))
    rows[:, 0:2] /= _safe_scale(time_scale)
    rows[:, 2:4] /= _safe_scale(energy_scale)
    return rows


# ============================================================
# Five-relation graph construction
# ============================================================


def _build_contention_edges(
    roots: Sequence[int],
    profiles: Sequence[Dict[str, float | int | str]],
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Build potential homogeneous-pool contention among initially ready tasks.

    v2 intentionally keeps the same initial contention rule as v1 so this stage
    changes only resource heterogeneity, V2V sparsity, and candidate feasibility.
    A root-task pair is linked if both gangs would not fit simultaneously in one
    6-core homogeneous vehicle pool.  Resource-specific contention is deferred
    to the later dynamic-snapshot stage.
    """

    pairs: List[Tuple[int, int]] = []
    rows: List[List[float]] = []

    for first_pos, task_a in enumerate(roots):
        gang_a = int(profiles[task_a]["gang_size"])
        for task_b in roots[first_pos + 1 :]:
            gang_b = int(profiles[task_b]["gang_size"])
            if gang_a + gang_b <= VEHICLE_A7_CORES:
                continue

            combined = min((gang_a + gang_b) / (2.0 * MAX_GANG_SIZE), 1.0)

            pairs.append((int(task_a), int(task_b)))
            rows.append(
                [
                    gang_a / MAX_GANG_SIZE,
                    gang_b / MAX_GANG_SIZE,
                    combined,
                    1.0,
                ]
            )

            pairs.append((int(task_b), int(task_a)))
            rows.append(
                [
                    gang_b / MAX_GANG_SIZE,
                    gang_a / MAX_GANG_SIZE,
                    combined,
                    1.0,
                ]
            )

    return pairs, np.asarray(rows, dtype=np.float32).reshape(
        -1, len(CONTENTION_EDGE_FEATURE_NAMES)
    )


def build_hetero_graph(seed: int = SEED) -> Tuple[HeteroData, Dict[str, object]]:
    """Build Legacy-native Simple MAGNET Heterogeneous Graph v2."""

    source = load_legacy_native_task_graph()
    dag: nx.DiGraph = source["dag"]  # type: ignore[assignment]
    dependencies: List[Tuple[int, int]] = source["dependencies"]  # type: ignore[assignment]
    profiles: List[Dict[str, float | int | str]] = source["profiles"]  # type: ignore[assignment]
    roots: List[int] = source["roots"]  # type: ignore[assignment]
    sinks: List[int] = source["sinks"]  # type: ignore[assignment]
    layers: List[List[int]] = source["layers"]  # type: ignore[assignment]

    num_tasks = len(profiles)
    vehicles, edge_servers = generate_resources(seed)

    # --------------------------------------------------------
    # Fixed normalization scales (recorded in metadata)
    # --------------------------------------------------------

    deadline_scale = max(float(profile["deadline"]) for profile in profiles)
    execution_time_scale = max(
        max(float(profile["a7_time"]), float(profile["a12_time"]))
        for profile in profiles
    )
    task_energy_scale = max(
        max(float(profile["a7_energy"]), float(profile["a12_energy"]))
        for profile in profiles
    )
    resource_core_scale = float(max(EDGE_A7_CORES, EDGE_A12_CORES))

    root_set = set(roots)

    task_rows: List[List[float]] = []
    for task_id, profile in enumerate(profiles):
        task_rows.append(
            [
                int(profile["gang_size"]) / MAX_GANG_SIZE,
                float(profile["deadline"]) / _safe_scale(deadline_scale),
                float(profile["a7_time"]) / _safe_scale(execution_time_scale),
                float(profile["a12_time"]) / _safe_scale(execution_time_scale),
                float(profile["a7_energy"]) / _safe_scale(task_energy_scale),
                float(profile["a12_energy"]) / _safe_scale(task_energy_scale),
                float(task_id in root_set),
                0.0,
                0.0,
            ]
        )

    resource_rows_vehicle = [
        [
            float(resource["a7_cores"]) / resource_core_scale,
            float(resource["a12_cores"]) / resource_core_scale,
            float(resource["available_a7"]) / max(float(resource["a7_cores"]), 1.0),
            float(resource["available_a12"]) / max(float(resource["a12_cores"]), 1.0),
        ]
        for resource in vehicles
    ]

    resource_rows_edge = [
        [
            float(resource["a7_cores"]) / resource_core_scale,
            float(resource["a12_cores"]) / resource_core_scale,
            float(resource["available_a7"]) / max(float(resource["a7_cores"]), 1.0),
            float(resource["available_a12"]) / max(float(resource["a12_cores"]), 1.0),
        ]
        for resource in edge_servers
    ]

    # --------------------------------------------------------
    # Candidate relations: sparse + feasibility-pruned compute candidates in v2
    # --------------------------------------------------------

    local_pairs: List[Tuple[int, int]] = []
    v2v_pairs: List[Tuple[int, int]] = []
    v2i_pairs: List[Tuple[int, int]] = []

    local_raw: List[List[float]] = []
    v2v_raw: List[List[float]] = []
    v2i_raw: List[List[float]] = []

    source_vehicle = vehicles[SOURCE_VEHICLE_ID]
    reachable_v2v_ids = set(int(v) for v in V2V_REACHABLE_VEHICLE_IDS)

    if SOURCE_VEHICLE_ID in reachable_v2v_ids:
        raise ValueError("The source vehicle must not appear in V2V reachability.")
    if any(vehicle_id < 0 or vehicle_id >= NUM_VEHICLES for vehicle_id in reachable_v2v_ids):
        raise ValueError("V2V_REACHABLE_VEHICLE_IDS contains an invalid vehicle id.")

    for task_id, profile in enumerate(profiles):
        # Local relation: include only if at least one homogeneous pool can fit
        # the gang.  With the controlled v2 source state this remains 60/60.
        if _resource_is_feasible(profile, source_vehicle):
            local_pairs.append((task_id, SOURCE_VEHICLE_ID))
            local_raw.append(_candidate_features_raw(profile, source_vehicle))

        # V2V relation: relation existence now means BOTH statically reachable
        # from source Vehicle 0 AND computationally feasible for this gang.
        for vehicle in vehicles:
            vehicle_id = int(vehicle["id"])
            if vehicle_id not in reachable_v2v_ids:
                continue
            if not _resource_is_feasible(profile, vehicle):
                continue
            v2v_pairs.append((task_id, vehicle_id))
            v2v_raw.append(_candidate_features_raw(profile, vehicle))

        # V2I relation follows the same feasibility-pruning rule.  The current
        # 12+12-core edge server can fit all m_i<=6 tasks, so 60 edges remain.
        for edge_server in edge_servers:
            if not _resource_is_feasible(profile, edge_server):
                continue
            v2i_pairs.append((task_id, int(edge_server["id"])))
            v2i_raw.append(_candidate_features_raw(profile, edge_server))

    all_candidate_raw = np.asarray(local_raw + v2v_raw + v2i_raw, dtype=np.float32)
    candidate_time_scale = float(np.max(all_candidate_raw[:, 0:2]))
    candidate_energy_scale = float(np.max(all_candidate_raw[:, 2:4]))

    local_attr = _normalize_candidate_rows(
        np.asarray(local_raw, dtype=np.float32),
        candidate_time_scale,
        candidate_energy_scale,
    )
    v2v_attr = _normalize_candidate_rows(
        np.asarray(v2v_raw, dtype=np.float32),
        candidate_time_scale,
        candidate_energy_scale,
    )
    v2i_attr = _normalize_candidate_rows(
        np.asarray(v2i_raw, dtype=np.float32),
        candidate_time_scale,
        candidate_energy_scale,
    )

    contention_pairs, contention_attr = _build_contention_edges(roots, profiles)

    # --------------------------------------------------------
    # PyG HeteroData
    # --------------------------------------------------------

    data = HeteroData()
    data.dataset_version = DATASET_VERSION
    data.simulation_time = torch.tensor([0.0], dtype=torch.float32)

    data["task"].x = torch.tensor(task_rows, dtype=torch.float32)
    data["task"].task_id = torch.arange(num_tasks, dtype=torch.long)
    data["task"].source_vehicle = torch.full(
        (num_tasks,), SOURCE_VEHICLE_ID, dtype=torch.long
    )
    data["task"].gang_size = torch.tensor(
        [int(profile["gang_size"]) for profile in profiles], dtype=torch.long
    )
    data["task"].deadline = torch.tensor(
        [float(profile["deadline"]) for profile in profiles], dtype=torch.float32
    )
    data["task"].exec_time_a7 = torch.tensor(
        [float(profile["a7_time"]) for profile in profiles], dtype=torch.float32
    )
    data["task"].exec_time_a12 = torch.tensor(
        [float(profile["a12_time"]) for profile in profiles], dtype=torch.float32
    )
    data["task"].energy_a7 = torch.tensor(
        [float(profile["a7_energy"]) for profile in profiles], dtype=torch.float32
    )
    data["task"].energy_a12 = torch.tensor(
        [float(profile["a12_energy"]) for profile in profiles], dtype=torch.float32
    )
    data["task"].topological_layer = torch.tensor(
        [int(profile["topological_layer"]) for profile in profiles], dtype=torch.long
    )
    data["task"].is_ready = torch.tensor(
        [int(task_id in root_set) for task_id in range(num_tasks)], dtype=torch.long
    )
    data["task"].is_running = torch.zeros(num_tasks, dtype=torch.long)
    data["task"].is_completed = torch.zeros(num_tasks, dtype=torch.long)

    data["vehicle"].x = torch.tensor(resource_rows_vehicle, dtype=torch.float32)
    data["vehicle"].node_id = torch.arange(len(vehicles), dtype=torch.long)
    data["vehicle"].total_a7_cores = torch.tensor(
        [int(resource["a7_cores"]) for resource in vehicles], dtype=torch.long
    )
    data["vehicle"].total_a12_cores = torch.tensor(
        [int(resource["a12_cores"]) for resource in vehicles], dtype=torch.long
    )
    data["vehicle"].available_a7_cores = torch.tensor(
        [int(resource["available_a7"]) for resource in vehicles], dtype=torch.long
    )
    data["vehicle"].available_a12_cores = torch.tensor(
        [int(resource["available_a12"]) for resource in vehicles], dtype=torch.long
    )

    data["edge_server"].x = torch.tensor(resource_rows_edge, dtype=torch.float32)
    data["edge_server"].node_id = torch.arange(len(edge_servers), dtype=torch.long)
    data["edge_server"].total_a7_cores = torch.tensor(
        [int(resource["a7_cores"]) for resource in edge_servers], dtype=torch.long
    )
    data["edge_server"].total_a12_cores = torch.tensor(
        [int(resource["a12_cores"]) for resource in edge_servers], dtype=torch.long
    )
    data["edge_server"].available_a7_cores = torch.tensor(
        [int(resource["available_a7"]) for resource in edge_servers], dtype=torch.long
    )
    data["edge_server"].available_a12_cores = torch.tensor(
        [int(resource["available_a12"]) for resource in edge_servers], dtype=torch.long
    )

    data[DAG_DEPENDENCY_RELATION].edge_index = _edge_index(dependencies)
    data[DAG_DEPENDENCY_RELATION].edge_attr = _empty_edge_attr(len(dependencies))

    data[LOCAL_EXECUTION_RELATION].edge_index = _edge_index(local_pairs)
    data[LOCAL_EXECUTION_RELATION].edge_attr = torch.from_numpy(local_attr)
    data[LOCAL_EXECUTION_RELATION].edge_attr_raw = torch.tensor(
        local_raw, dtype=torch.float32
    )

    data[V2V_OFFLOADING_RELATION].edge_index = _edge_index(v2v_pairs)
    data[V2V_OFFLOADING_RELATION].edge_attr = torch.from_numpy(v2v_attr)
    data[V2V_OFFLOADING_RELATION].edge_attr_raw = torch.tensor(
        v2v_raw, dtype=torch.float32
    )

    data[V2I_OFFLOADING_RELATION].edge_index = _edge_index(v2i_pairs)
    data[V2I_OFFLOADING_RELATION].edge_attr = torch.from_numpy(v2i_attr)
    data[V2I_OFFLOADING_RELATION].edge_attr_raw = torch.tensor(
        v2i_raw, dtype=torch.float32
    )

    data[GANG_CONTENTION_RELATION].edge_index = _edge_index(contention_pairs)
    data[GANG_CONTENTION_RELATION].edge_attr = torch.from_numpy(contention_attr)

    # --------------------------------------------------------
    # Metadata / provenance
    # --------------------------------------------------------

    level_jumps = [
        int(profiles[v]["topological_layer"]) - int(profiles[u]["topological_layer"])
        for u, v in dependencies
    ]

    metadata: Dict[str, object] = {
        "dataset_name": "Legacy-native Simple MAGNET Heterogeneous Graph v2",
        "dataset_version": DATASET_VERSION,
        "purpose": "sparse, resource-heterogeneous, auditable base heterogeneous graph for muxGNN",
        "seed": int(seed),
        "source": {
            "legacy_dataset": str(LEGACY_DATASET_PATH.relative_to(LEGACY_DATASET_PATH.parents[2])),
            "legacy_row_index": int(LEGACY_ROW_INDEX),
            "legacy_dataset_sha256": _sha256(LEGACY_DATASET_PATH),
            "dagbench_used": False,
            "dagbench_topology_used": False,
            "external_workflow_fusion": False,
        },
        "graph": {
            "node_types": ["task", "vehicle", "edge_server"],
            "num_tasks": num_tasks,
            "num_vehicles": len(vehicles),
            "num_edge_servers": len(edge_servers),
            "total_nodes": num_tasks + len(vehicles) + len(edge_servers),
            "num_topological_layers": len(layers),
            "topological_layer_sizes": [len(layer) for layer in layers],
            "num_roots": len(roots),
            "root_task_ids": roots,
            "num_sinks": len(sinks),
            "sink_task_ids": sinks,
            "num_dependency_edges": len(dependencies),
            "max_layer_jump": max(level_jumps) if level_jumps else 0,
            "weakly_connected": bool(nx.is_weakly_connected(dag)),
            "relation_edge_counts": {
                "precedes": len(dependencies),
                "local_execute_on": len(local_pairs),
                "v2v_offload_to": len(v2v_pairs),
                "v2i_offload_to": len(v2i_pairs),
                "contends_with": len(contention_pairs),
            },
            "candidate_policy": {
                "v2v_reachability_pruned": True,
                "infeasible_task_resource_edges_removed": True,
                "candidate_edge_exists_only_if_any_processor_feasible": True,
            },
        },
        "resource_model": {
            "source_vehicle_id": SOURCE_VEHICLE_ID,
            "vehicle_count": NUM_VEHICLES,
            "vehicle_a7_cores": VEHICLE_A7_CORES,
            "vehicle_a12_cores": VEHICLE_A12_CORES,
            "edge_server_count": NUM_EDGE_SERVERS,
            "edge_a7_cores": EDGE_A7_CORES,
            "edge_a12_cores": EDGE_A12_CORES,
            "initial_available_cores_equal_total": False,
            "vehicle_available_a7": list(VEHICLE_AVAILABLE_A7),
            "vehicle_available_a12": list(VEHICLE_AVAILABLE_A12),
            "controlled_v2v_reachable_vehicle_ids": list(V2V_REACHABLE_VEHICLE_IDS),
            "reachability_semantics": "controlled static candidate subset; no distance/radio physics claimed",
            "compute_scale": 1.0,
            "power_scale": 1.0,
            "background_load": False,
        },
        "task_state": {
            "initial_ready_rule": "DAG roots only",
            "initial_running": 0,
            "initial_completed": 0,
        },
        "execution_model": {
            "runtime": "legacy perfect_linear_speedup",
            "formula": "exec_time(resource,processor)=legacy_time(processor)/(gang_size*compute_scale)",
            "energy_formula": "exec_energy=exec_time*gang_size*legacy_power*power_scale",
            "time_semantics": "canonical relative simulation time inherited from validated Legacy dataset",
            "physical_wall_clock_seconds_claimed": False,
        },
        "relations": {
            "precedes": {
                "meaning": "exact Legacy DAG precedence",
                "edge_features": [],
            },
            "local_execute_on": {
                "meaning": "task to source vehicle 0",
                "edge_features": list(CANDIDATE_EDGE_FEATURE_NAMES),
            },
            "v2v_offload_to": {
                "meaning": "candidate to a controlled reachable non-source vehicle, kept only when at least one processor pool can fit the gang; no wireless physics in v2",
                "edge_features": list(CANDIDATE_EDGE_FEATURE_NAMES),
            },
            "v2i_offload_to": {
                "meaning": "candidate to edge server, kept only when at least one processor pool can fit the gang; no wireless physics in v2",
                "edge_features": list(CANDIDATE_EDGE_FEATURE_NAMES),
            },
            "contends_with": {
                "meaning": "initially-ready task pair whose gang widths exceed one 6-core homogeneous vehicle pool",
                "edge_features": list(CONTENTION_EDGE_FEATURE_NAMES),
                "symmetric": True,
            },
        },
        "feature_names": {
            "task_node": list(TASK_FEATURE_NAMES),
            "vehicle_node": list(RESOURCE_FEATURE_NAMES),
            "edge_server_node": list(RESOURCE_FEATURE_NAMES),
            "dependency_edge": list(DEPENDENCY_EDGE_FEATURE_NAMES),
            "local_execution_edge": list(CANDIDATE_EDGE_FEATURE_NAMES),
            "v2v_offloading_edge": list(CANDIDATE_EDGE_FEATURE_NAMES),
            "v2i_offloading_edge": list(CANDIDATE_EDGE_FEATURE_NAMES),
            "contention_edge": list(CONTENTION_EDGE_FEATURE_NAMES),
        },
        "normalization": {
            "gang_size_scale": MAX_GANG_SIZE,
            "deadline_scale": deadline_scale,
            "task_execution_time_scale": execution_time_scale,
            "task_energy_scale": task_energy_scale,
            "resource_core_scale": resource_core_scale,
            "candidate_execution_time_scale": candidate_time_scale,
            "candidate_energy_scale": candidate_energy_scale,
            "binary_features": "kept as 0/1",
            "fit_scope": "single validated Legacy graph v2; scales are saved explicitly for reproducibility",
        },
        "provenance": {
            "task_gang_size": "Legacy dataset",
            "task_deadline": "Legacy DAG-aware derived deadline",
            "task_execution_time": "Legacy measurement-derived canonical timing",
            "task_energy": "Legacy canonical execution time multiplied by measured/derived power",
            "dependency_topology": "Legacy validated 60-task DAG",
            "resource_capacity": "controlled simulation assumption",
            "resource_availability": "controlled heterogeneous initial simulation state",
            "v2v_reachability": "controlled static candidate subset",
            "candidate_time_energy": "derived from task profile + gang width + resource capacity model",
            "contention": "derived from initial ready state + gang widths",
        },
        "intentionally_omitted_v2": [
            "DAGBench",
            "DAGBench cost",
            "DAGBench data_size",
            "mobility",
            "vehicle position",
            "V2V/V2I distance",
            "link rate",
            "transmission delay",
            "contact duration",
            "queue delay",
            "random background load",
            "migration",
        ],
        "benchmark_names": [str(profile["benchmark_name"]) for profile in profiles],
    }

    return data, metadata
