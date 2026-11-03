from __future__ import annotations

import json

import networkx as nx
import torch

from .builder import (
    CANDIDATE_EDGE_FEATURE_NAMES,
    CONTENTION_EDGE_FEATURE_NAMES,
    TASK_FEATURE_NAMES,
    RESOURCE_FEATURE_NAMES,
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
    DATASET_VERSION,
    GRAPH_PATH,
    METADATA_PATH,
    NUM_EDGE_SERVERS,
    NUM_VEHICLES,
    SOURCE_VEHICLE_ID,
    VEHICLE_A7_CORES,
    VEHICLE_AVAILABLE_A7,
    VEHICLE_AVAILABLE_A12,
    V2V_REACHABLE_VEHICLE_IDS,
)

EXPECTED_NODE_TYPES = {"task", "vehicle", "edge_server"}
EXPECTED_EDGE_TYPES = {
    DAG_DEPENDENCY_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2V_OFFLOADING_RELATION,
    V2I_OFFLOADING_RELATION,
    GANG_CONTENTION_RELATION,
}


def _load_graph():
    try:
        return torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(GRAPH_PATH, map_location="cpu")


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if tensor.numel() and not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf.")


def _check_edge_bounds(graph, relation) -> None:
    source_type, _, target_type = relation
    edge_index = graph[relation].edge_index
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise AssertionError(f"{relation}: invalid edge_index shape.")
    if edge_index.numel() == 0:
        return
    if int(edge_index[0].min()) < 0 or int(edge_index[1].min()) < 0:
        raise AssertionError(f"{relation}: negative edge index.")
    if int(edge_index[0].max()) >= graph[source_type].num_nodes:
        raise AssertionError(f"{relation}: source index out of range.")
    if int(edge_index[1].max()) >= graph[target_type].num_nodes:
        raise AssertionError(f"{relation}: target index out of range.")


def _pairs(graph, relation):
    return {
        (int(source), int(target))
        for source, target in graph[relation].edge_index.t().tolist()
    }


def main() -> None:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            "Graph not found. Run: python -m muxgnn.graph.generate"
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError("Dataset metadata is missing.")

    graph = _load_graph()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    source = load_legacy_native_task_graph()

    if metadata.get("dataset_version") != DATASET_VERSION:
        raise AssertionError("Unexpected dataset version.")
    if metadata["source"].get("dagbench_used") is not False:
        raise AssertionError("DAGBench must not be used by this dataset.")

    if set(graph.node_types) != EXPECTED_NODE_TYPES:
        raise AssertionError(f"Unexpected node types: {graph.node_types}")
    if set(graph.edge_types) != EXPECTED_EDGE_TYPES:
        raise AssertionError(f"Unexpected relation layers: {graph.edge_types}")

    if int(graph["task"].num_nodes) != 60:
        raise AssertionError("Expected exactly 60 Legacy tasks.")
    if int(graph["vehicle"].num_nodes) != NUM_VEHICLES:
        raise AssertionError("Unexpected vehicle count.")
    if int(graph["edge_server"].num_nodes) != NUM_EDGE_SERVERS:
        raise AssertionError("Unexpected edge-server count.")

    if int(graph["task"].x.shape[1]) != len(TASK_FEATURE_NAMES):
        raise AssertionError("Task feature dimension mismatch.")
    if int(graph["vehicle"].x.shape[1]) != len(RESOURCE_FEATURE_NAMES):
        raise AssertionError("Vehicle feature dimension mismatch.")
    if int(graph["edge_server"].x.shape[1]) != len(RESOURCE_FEATURE_NAMES):
        raise AssertionError("Edge-server feature dimension mismatch.")

    for node_type in graph.node_types:
        _check_finite(f"{node_type}.x", graph[node_type].x)
        if graph[node_type].x.numel():
            if float(graph[node_type].x.min()) < -1e-6:
                raise AssertionError(f"{node_type}.x has negative normalized features.")
            if float(graph[node_type].x.max()) > 1.0 + 1e-6:
                raise AssertionError(f"{node_type}.x exceeds the expected [0,1] range.")

    for relation in graph.edge_types:
        _check_edge_bounds(graph, relation)
        _check_finite(f"{relation}.edge_attr", graph[relation].edge_attr)
        if graph[relation].edge_attr.shape[0] != graph[relation].num_edges:
            raise AssertionError(f"{relation}: edge_attr row count mismatch.")

    # Dependency topology must be exactly the validated Legacy DAG.
    expected_dependencies = set(source["dependencies"])
    observed_dependencies = _pairs(graph, DAG_DEPENDENCY_RELATION)
    if observed_dependencies != expected_dependencies:
        raise AssertionError("Dependency relation does not match the Legacy DAG.")
    if graph[DAG_DEPENDENCY_RELATION].edge_attr.shape[1] != 0:
        raise AssertionError("Dependency edges must have no features in v2.")

    dag = nx.DiGraph()
    dag.add_nodes_from(range(60))
    dag.add_edges_from(observed_dependencies)
    if not nx.is_directed_acyclic_graph(dag):
        raise AssertionError("Dependency relation is cyclic.")
    layers = list(nx.topological_generations(dag))
    if [len(layer) for layer in layers] != [8, 8, 8, 8, 7, 7, 7, 7]:
        raise AssertionError("Unexpected Legacy topological-layer structure.")

    level = {}
    for layer_id, layer in enumerate(layers):
        for task_id in layer:
            level[int(task_id)] = layer_id
    if any(level[v] - level[u] != 1 for u, v in observed_dependencies):
        raise AssertionError("Every dependency must connect adjacent layers only.")

    roots = {int(node) for node in dag.nodes if dag.in_degree(node) == 0}
    ready = {
        task_id
        for task_id in range(60)
        if int(graph["task"].is_ready[task_id]) == 1
    }
    if ready != roots:
        raise AssertionError("Initial ready state must contain exactly DAG roots.")
    if int(graph["task"].is_running.sum()) != 0:
        raise AssertionError("No task may be running in the initial v2 snapshot.")
    if int(graph["task"].is_completed.sum()) != 0:
        raise AssertionError("No task may be completed in the initial v2 snapshot.")

    # Resource states must match the controlled heterogeneous v2 contract.
    observed_a7 = tuple(int(v) for v in graph["vehicle"].available_a7_cores.tolist())
    observed_a12 = tuple(int(v) for v in graph["vehicle"].available_a12_cores.tolist())
    if observed_a7 != tuple(VEHICLE_AVAILABLE_A7):
        raise AssertionError("Vehicle A7 availability does not match the v2 contract.")
    if observed_a12 != tuple(VEHICLE_AVAILABLE_A12):
        raise AssertionError("Vehicle A12 availability does not match the v2 contract.")

    feasible_a7_idx = CANDIDATE_EDGE_FEATURE_NAMES.index("feasible_a7")
    feasible_a12_idx = CANDIDATE_EDGE_FEATURE_NAMES.index("feasible_a12")

    def _assert_edge_is_feasible(relation) -> None:
        attrs = graph[relation].edge_attr
        if attrs.numel() == 0:
            return
        any_feasible = (attrs[:, feasible_a7_idx] > 0.5) | (attrs[:, feasible_a12_idx] > 0.5)
        if not bool(torch.all(any_feasible)):
            raise AssertionError(f"{relation}: edge exists although both processor pools are infeasible.")

    # Local: one source-vehicle edge per task because Vehicle 0 is fully free.
    local = graph[LOCAL_EXECUTION_RELATION]
    if int(local.num_edges) != 60:
        raise AssertionError("Expected 60 local execution edges in v2.")
    if local.edge_attr.shape[1] != len(CANDIDATE_EDGE_FEATURE_NAMES):
        raise AssertionError("Local candidate feature dimension mismatch.")
    for task_id, vehicle_id in local.edge_index.t().tolist():
        if int(vehicle_id) != SOURCE_VEHICLE_ID:
            raise AssertionError("Local edge targets a non-source vehicle.")
    _assert_edge_is_feasible(LOCAL_EXECUTION_RELATION)

    # V2V: only controlled reachable vehicles are eligible, and an edge exists
    # only if A7 or A12 can fit the task gang on that vehicle.
    v2v = graph[V2V_OFFLOADING_RELATION]
    if v2v.edge_attr.shape[1] != len(CANDIDATE_EDGE_FEATURE_NAMES):
        raise AssertionError("V2V candidate feature dimension mismatch.")

    reachable = set(int(v) for v in V2V_REACHABLE_VEHICLE_IDS)
    observed_v2v = _pairs(graph, V2V_OFFLOADING_RELATION)
    expected_v2v = set()
    for task_id in range(60):
        width = int(graph["task"].gang_size[task_id])
        for vehicle_id in reachable:
            feasible = (
                int(VEHICLE_AVAILABLE_A7[vehicle_id]) >= width
                or int(VEHICLE_AVAILABLE_A12[vehicle_id]) >= width
            )
            if feasible:
                expected_v2v.add((task_id, vehicle_id))

    if observed_v2v != expected_v2v:
        raise AssertionError("V2V edges do not match v2 reachability + feasibility pruning.")
    for _, vehicle_id in observed_v2v:
        if vehicle_id == SOURCE_VEHICLE_ID or vehicle_id not in reachable:
            raise AssertionError("V2V relation contains an unreachable/invalid vehicle.")
    _assert_edge_is_feasible(V2V_OFFLOADING_RELATION)

    # V2I: same feasibility-pruning rule.  With 12+12 available cores every
    # current Legacy task is feasible, so the expected count remains 60.
    v2i = graph[V2I_OFFLOADING_RELATION]
    if v2i.edge_attr.shape[1] != len(CANDIDATE_EDGE_FEATURE_NAMES):
        raise AssertionError("V2I candidate feature dimension mismatch.")
    if int(v2i.num_edges) != 60 * NUM_EDGE_SERVERS:
        raise AssertionError("Expected every m_i<=6 task to remain V2I-feasible in v2.")
    _assert_edge_is_feasible(V2I_OFFLOADING_RELATION)

    # Contention: symmetric, root-only, and exactly when gang_a + gang_b > 6.
    contention = graph[GANG_CONTENTION_RELATION]
    if contention.edge_attr.shape[1] != len(CONTENTION_EDGE_FEATURE_NAMES):
        raise AssertionError("Contention feature dimension mismatch.")
    contention_pairs = _pairs(graph, GANG_CONTENTION_RELATION)
    for source_task, target_task in contention_pairs:
        if source_task not in roots or target_task not in roots:
            raise AssertionError("Contention v2 must involve initially-ready roots only.")
        if (target_task, source_task) not in contention_pairs:
            raise AssertionError("Contention relation must be symmetric.")
        gang_a = int(graph["task"].gang_size[source_task])
        gang_b = int(graph["task"].gang_size[target_task])
        if gang_a + gang_b <= VEHICLE_A7_CORES:
            raise AssertionError("Invalid contention edge for co-fitting gangs.")

    expected_contention = set()
    roots_sorted = sorted(roots)
    for i, task_a in enumerate(roots_sorted):
        for task_b in roots_sorted[i + 1 :]:
            if int(graph["task"].gang_size[task_a]) + int(graph["task"].gang_size[task_b]) > VEHICLE_A7_CORES:
                expected_contention.add((task_a, task_b))
                expected_contention.add((task_b, task_a))
    if contention_pairs != expected_contention:
        raise AssertionError("Contention relation does not match the unchanged v2 base rule.")

    print("============================================")
    print("Legacy-native Simple MAGNET v2 validation PASSED")
    print("============================================")
    print(f"Tasks           : {graph['task'].num_nodes}")
    print(f"Vehicles        : {graph['vehicle'].num_nodes}")
    print(f"Edge servers    : {graph['edge_server'].num_nodes}")
    print(f"DAG dependency  : {graph[DAG_DEPENDENCY_RELATION].num_edges}")
    print(f"Local execution : {graph[LOCAL_EXECUTION_RELATION].num_edges}")
    print(f"V2V offloading  : {graph[V2V_OFFLOADING_RELATION].num_edges}")
    print(f"V2I offloading  : {graph[V2I_OFFLOADING_RELATION].num_edges}")
    print(f"Gang contention : {graph[GANG_CONTENTION_RELATION].num_edges}")
    print("DAGBench used   : False")


if __name__ == "__main__":
    main()
