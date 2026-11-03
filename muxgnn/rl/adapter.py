from __future__ import annotations

from typing import Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..graph.builder import (
    CANDIDATE_EDGE_FEATURE_NAMES,
    RESOURCE_FEATURE_NAMES,
    TASK_FEATURE_NAMES,
)
from ..graph.magnet_relation_layers import (
    DAG_DEPENDENCY_RELATION,
    GANG_CONTENTION_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2I_OFFLOADING_RELATION,
    V2V_OFFLOADING_RELATION,
)
# v4's task nodes have a 10th feature (ready_time_sec) compared with v2's
# 9 features. Switch *only* for the explicit three-edge mobility mode.
# No relation, message passing depth, hidden width, or reward is changed.
import os
if os.environ.get("MUX_MOBILITY3", "0") == "1":
    from ..mobility3.builder import (
        V4_TASK_FEATURE_NAMES as TASK_FEATURE_NAMES,
        V4_RESOURCE_FEATURE_NAMES as RESOURCE_FEATURE_NAMES,
        V4_CANDIDATE_EDGE_FEATURE_NAMES as CANDIDATE_EDGE_FEATURE_NAMES,
    )

from .config import (
    MUX_EDGE_FEATURE_DIM,
    MUX_GLOBAL_FEATURE_DIM,
    MUX_MAX_EDGE_SERVERS,
    MUX_MAX_NODES,
    MUX_MAX_RESOURCES,
    MUX_MAX_TASKS,
    MUX_MAX_VEHICLES,
    MUX_NUM_PROCESSOR_TYPES,
    MUX_NUM_RELATIONS,
    MUX_PAIR_FEATURE_DIM,
)


# ============================================================
# Legacy-native v2 muxGNN observation contract
# ============================================================

RELATION_NAMES: Tuple[str, ...] = (
    "dependency",
    "local",
    "v2v",
    "v2i",
    "contention",
)

RELATION_KEYS: Tuple[Tuple[str, str, str], ...] = (
    DAG_DEPENDENCY_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2V_OFFLOADING_RELATION,
    V2I_OFFLOADING_RELATION,
    GANG_CONTENTION_RELATION,
)

RELATION_ID = {
    relation: index
    for index, relation in enumerate(RELATION_KEYS)
}

# A task-resource pair is represented only by the semantics that actually exist
# in Legacy-native Simple MAGNET v2.  There are no placeholder communication
# features in the Smoke-Train Bridge.
PAIR_FEATURE_NAMES: Tuple[str, ...] = (
    "is_local",
    "is_v2v",
    "is_v2i",
    *CANDIDATE_EDGE_FEATURE_NAMES,
)

GLOBAL_FEATURE_NAMES: Tuple[str, ...] = (
    "simulation_time_norm",
    "ready_ratio",
    "running_ratio",
    "completed_ratio",
    "v2v_density",
    "v2i_density",
    "contention_density",
    "candidate_density",
)

if len(PAIR_FEATURE_NAMES) != MUX_PAIR_FEATURE_DIM:
    raise RuntimeError(
        "MUX_PAIR_FEATURE_DIM does not match Legacy-native v2 pair features: "
        f"expected {len(PAIR_FEATURE_NAMES)}, configured {MUX_PAIR_FEATURE_DIM}."
    )

if len(GLOBAL_FEATURE_NAMES) != MUX_GLOBAL_FEATURE_DIM:
    raise RuntimeError(
        "MUX_GLOBAL_FEATURE_DIM does not match the v2 global feature contract."
    )

# The widest stored v2 relation has 8 features.  One final channel is reserved
# for the derived message-flow direction used only inside the dense muxGNN view.
if MUX_EDGE_FEATURE_DIM < len(CANDIDATE_EDGE_FEATURE_NAMES) + 1:
    raise RuntimeError(
        "MUX_EDGE_FEATURE_DIM is too small for v2 candidate features + direction."
    )


class HeteroMuxAdapter:
    """Convert the Legacy-native Simple MAGNET v2 HeteroData graph to dense tensors.

    This adapter is the explicit v2 -> muxGNN Smoke-Train Bridge.  It deliberately
    contains no DAGBench, mobility, physical-network, queue-delay, or migration
    assumptions.  The stored HeteroData graph is never modified.
    """

    def __init__(
        self,
        max_tasks: int = MUX_MAX_TASKS,
        max_vehicles: int = MUX_MAX_VEHICLES,
        max_edge_servers: int = MUX_MAX_EDGE_SERVERS,
    ) -> None:
        self.max_tasks = int(max_tasks)
        self.max_vehicles = int(max_vehicles)
        self.max_edge_servers = int(max_edge_servers)
        self.max_resources = self.max_vehicles + self.max_edge_servers
        self.num_processor_types = MUX_NUM_PROCESSOR_TYPES
        self.max_nodes = self.max_tasks + self.max_resources

        if self.max_tasks != MUX_MAX_TASKS:
            raise ValueError("Use the fixed MUX_MAX_TASKS contract.")
        if self.max_vehicles != MUX_MAX_VEHICLES:
            raise ValueError("Use the fixed MUX_MAX_VEHICLES contract.")
        if self.max_edge_servers != MUX_MAX_EDGE_SERVERS:
            raise ValueError("Use the fixed MUX_MAX_EDGE_SERVERS contract.")
        if self.max_nodes != MUX_MAX_NODES:
            raise RuntimeError("Internal muxGNN node-count contract mismatch.")

        self.task_feature_dim = len(TASK_FEATURE_NAMES)
        self.vehicle_feature_dim = len(RESOURCE_FEATURE_NAMES)
        self.edge_server_feature_dim = len(RESOURCE_FEATURE_NAMES)

        self.ready_feature_index = TASK_FEATURE_NAMES.index("is_ready")
        self.running_feature_index = TASK_FEATURE_NAMES.index("is_running")
        self.completed_feature_index = TASK_FEATURE_NAMES.index("is_completed")

        self.feasible_a7_pair_index = PAIR_FEATURE_NAMES.index("feasible_a7")
        self.feasible_a12_pair_index = PAIR_FEATURE_NAMES.index("feasible_a12")

        self.observation_space = gym.spaces.Dict(
            {
                "task_features": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.max_tasks, self.task_feature_dim),
                    dtype=np.float32,
                ),
                "vehicle_features": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.max_vehicles, self.vehicle_feature_dim),
                    dtype=np.float32,
                ),
                "edge_server_features": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.max_edge_servers, self.edge_server_feature_dim),
                    dtype=np.float32,
                ),
                "pair_features": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.max_tasks, self.max_resources, MUX_PAIR_FEATURE_DIM),
                    dtype=np.float32,
                ),
                # relation_adjacency[r, target, source] = 1 means source sends a
                # message to target in relation r.
                "relation_adjacency": gym.spaces.MultiBinary(
                    (MUX_NUM_RELATIONS, self.max_nodes, self.max_nodes)
                ),
                "relation_edge_features": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(
                        MUX_NUM_RELATIONS,
                        self.max_nodes,
                        self.max_nodes,
                        MUX_EDGE_FEATURE_DIM,
                    ),
                    dtype=np.float32,
                ),
                "relation_presence": gym.spaces.MultiBinary(
                    (MUX_NUM_RELATIONS, self.max_nodes)
                ),
                "task_mask": gym.spaces.MultiBinary(self.max_tasks),
                "vehicle_mask": gym.spaces.MultiBinary(self.max_vehicles),
                "edge_server_mask": gym.spaces.MultiBinary(self.max_edge_servers),
                "resource_mask": gym.spaces.MultiBinary(self.max_resources),
                "node_mask": gym.spaces.MultiBinary(self.max_nodes),
                "global_features": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(MUX_GLOBAL_FEATURE_DIM,),
                    dtype=np.float32,
                ),
                # The current bridge only needs this mask to verify that the
                # v2 graph exposes valid task-resource-processor candidates.
                "action_mask": gym.spaces.MultiBinary(
                    self.max_tasks
                    * self.max_resources
                    * self.num_processor_types
                ),
            }
        )

    # ========================================================
    # Public conversion
    # ========================================================

    def to_observation(self, data: HeteroData) -> Dict[str, np.ndarray]:
        """Convert one Legacy-native v2 graph to a fixed muxGNN observation."""

        self._validate_graph_contract(data)

        num_tasks = int(data["task"].num_nodes)
        num_vehicles = int(data["vehicle"].num_nodes)
        num_edge_servers = int(data["edge_server"].num_nodes)

        task_features = np.zeros(
            (self.max_tasks, self.task_feature_dim), dtype=np.float32
        )
        vehicle_features = np.zeros(
            (self.max_vehicles, self.vehicle_feature_dim), dtype=np.float32
        )
        edge_server_features = np.zeros(
            (self.max_edge_servers, self.edge_server_feature_dim), dtype=np.float32
        )

        task_features[:num_tasks] = self._to_numpy(data["task"].x)
        vehicle_features[:num_vehicles] = self._to_numpy(data["vehicle"].x)
        edge_server_features[:num_edge_servers] = self._to_numpy(
            data["edge_server"].x
        )

        task_mask = np.zeros(self.max_tasks, dtype=np.int8)
        task_mask[:num_tasks] = 1

        vehicle_mask = np.zeros(self.max_vehicles, dtype=np.int8)
        vehicle_mask[:num_vehicles] = 1

        edge_server_mask = np.zeros(self.max_edge_servers, dtype=np.int8)
        edge_server_mask[:num_edge_servers] = 1

        resource_mask = np.concatenate(
            [vehicle_mask, edge_server_mask]
        ).astype(np.int8)
        node_mask = np.concatenate(
            [task_mask, vehicle_mask, edge_server_mask]
        ).astype(np.int8)

        pair_features = np.zeros(
            (self.max_tasks, self.max_resources, MUX_PAIR_FEATURE_DIM),
            dtype=np.float32,
        )
        self._fill_pair_features(data, pair_features)

        adjacency = np.zeros(
            (MUX_NUM_RELATIONS, self.max_nodes, self.max_nodes), dtype=np.int8
        )
        edge_features = np.zeros(
            (
                MUX_NUM_RELATIONS,
                self.max_nodes,
                self.max_nodes,
                MUX_EDGE_FEATURE_DIM,
            ),
            dtype=np.float32,
        )
        presence = np.zeros(
            (MUX_NUM_RELATIONS, self.max_nodes), dtype=np.int8
        )

        self._fill_relation_tensors(
            data=data,
            adjacency=adjacency,
            edge_features=edge_features,
            presence=presence,
        )

        action_mask = self._build_action_mask(
            task_features=task_features,
            pair_features=pair_features,
            task_mask=task_mask,
            resource_mask=resource_mask,
        )

        global_features = self._build_global_features(
            data=data,
            task_features=task_features,
            num_tasks=num_tasks,
            num_vehicles=num_vehicles,
            num_edge_servers=num_edge_servers,
        )

        observation = {
            "task_features": task_features,
            "vehicle_features": vehicle_features,
            "edge_server_features": edge_server_features,
            "pair_features": pair_features,
            "relation_adjacency": adjacency,
            "relation_edge_features": edge_features,
            "relation_presence": presence,
            "task_mask": task_mask,
            "vehicle_mask": vehicle_mask,
            "edge_server_mask": edge_server_mask,
            "resource_mask": resource_mask,
            "node_mask": node_mask,
            "global_features": global_features,
            "action_mask": action_mask,
        }

        if not self.observation_space.contains(observation):
            raise AssertionError(
                "Converted Legacy-native v2 muxGNN observation is outside the fixed Gym space."
            )

        return observation

    # ========================================================
    # Graph validation
    # ========================================================

    def _validate_graph_contract(self, data: HeteroData) -> None:
        expected_node_types = {"task", "vehicle", "edge_server"}
        expected_edge_types = set(RELATION_KEYS)

        if set(data.node_types) != expected_node_types:
            raise ValueError(
                f"Expected node types {expected_node_types}, got {data.node_types}."
            )

        if set(data.edge_types) != expected_edge_types:
            raise ValueError(
                "v2 muxGNN adapter requires exactly the five MAGNET relations. "
                f"Got: {data.edge_types}"
            )

        if int(data["task"].num_nodes) > self.max_tasks:
            raise ValueError(
                f"Graph has {data['task'].num_nodes} tasks but max_tasks={self.max_tasks}."
            )
        if int(data["vehicle"].num_nodes) > self.max_vehicles:
            raise ValueError("Graph vehicle count exceeds muxGNN padding contract.")
        if int(data["edge_server"].num_nodes) > self.max_edge_servers:
            raise ValueError("Graph edge-server count exceeds muxGNN padding contract.")

        if int(data["task"].x.shape[1]) != self.task_feature_dim:
            raise ValueError(
                "Task feature dimension does not match Legacy-native v2 TASK_FEATURE_NAMES."
            )
        if int(data["vehicle"].x.shape[1]) != self.vehicle_feature_dim:
            raise ValueError(
                "Vehicle feature dimension does not match v2 RESOURCE_FEATURE_NAMES."
            )
        if int(data["edge_server"].x.shape[1]) != self.edge_server_feature_dim:
            raise ValueError(
                "Edge-server feature dimension does not match v2 RESOURCE_FEATURE_NAMES."
            )

        # Enforce the v2 edge-feature contract explicitly.
        expected_dims = {
            DAG_DEPENDENCY_RELATION: 0,
            LOCAL_EXECUTION_RELATION: len(CANDIDATE_EDGE_FEATURE_NAMES),
            V2V_OFFLOADING_RELATION: len(CANDIDATE_EDGE_FEATURE_NAMES),
            V2I_OFFLOADING_RELATION: len(CANDIDATE_EDGE_FEATURE_NAMES),
            GANG_CONTENTION_RELATION: 4,
        }
        for relation, expected_dim in expected_dims.items():
            edge_attr = data[relation].edge_attr
            actual_dim = int(edge_attr.shape[1])
            if actual_dim != expected_dim:
                raise ValueError(
                    f"{relation} edge feature dimension mismatch: "
                    f"expected {expected_dim}, got {actual_dim}."
                )

    # ========================================================
    # Pair features / action mask
    # ========================================================

    def _fill_pair_features(
        self,
        data: HeteroData,
        output: np.ndarray,
    ) -> None:
        candidate_dim = len(CANDIDATE_EDGE_FEATURE_NAMES)

        relation_specs = (
            (LOCAL_EXECUTION_RELATION, 0, False),
            (V2V_OFFLOADING_RELATION, 1, False),
            (V2I_OFFLOADING_RELATION, 2, True),
        )

        for relation, mode_index, is_edge_server in relation_specs:
            edge_index = self._to_numpy(data[relation].edge_index)
            edge_attr = self._to_numpy(data[relation].edge_attr)

            for edge_id in range(edge_index.shape[1]):
                task_id = int(edge_index[0, edge_id])
                target_local = int(edge_index[1, edge_id])
                resource_slot = (
                    self.max_vehicles + target_local
                    if is_edge_server
                    else target_local
                )

                output[task_id, resource_slot, mode_index] = 1.0
                output[
                    task_id,
                    resource_slot,
                    3 : 3 + candidate_dim,
                ] = edge_attr[edge_id]

    def _build_action_mask(
        self,
        task_features: np.ndarray,
        pair_features: np.ndarray,
        task_mask: np.ndarray,
        resource_mask: np.ndarray,
    ) -> np.ndarray:
        mask = np.zeros(
            self.max_tasks
            * self.max_resources
            * self.num_processor_types,
            dtype=np.int8,
        )

        for task_id in range(self.max_tasks):
            if task_mask[task_id] == 0:
                continue

            ready = task_features[task_id, self.ready_feature_index] > 0.5
            running = task_features[task_id, self.running_feature_index] > 0.5
            completed = task_features[task_id, self.completed_feature_index] > 0.5

            if not ready or running or completed:
                continue

            for resource_slot in range(self.max_resources):
                if resource_mask[resource_slot] == 0:
                    continue

                pair = pair_features[task_id, resource_slot]
                if not bool(np.any(pair[:3] > 0.5)):
                    continue

                processor_feasible = (
                    pair[self.feasible_a7_pair_index] > 0.5,
                    pair[self.feasible_a12_pair_index] > 0.5,
                )

                for processor_index, feasible in enumerate(processor_feasible):
                    if not feasible:
                        continue
                    pair_index = task_id * self.max_resources + resource_slot
                    action = pair_index * self.num_processor_types + processor_index
                    mask[action] = 1

        return mask

    # ========================================================
    # Dense relation tensors
    # ========================================================

    def _fill_relation_tensors(
        self,
        data: HeteroData,
        adjacency: np.ndarray,
        edge_features: np.ndarray,
        presence: np.ndarray,
    ) -> None:
        # Every real Task participates in the dependency view so roots/leaves keep
        # a dependency-specific representation via the relation encoder self loop.
        num_tasks = int(data["task"].num_nodes)
        for task_id in range(num_tasks):
            presence[RELATION_ID[DAG_DEPENDENCY_RELATION], self._global_task_node(task_id)] = 1

        for relation in RELATION_KEYS:
            relation_id = RELATION_ID[relation]
            edge_index = self._to_numpy(data[relation].edge_index)
            edge_attr = self._to_numpy(data[relation].edge_attr)

            for edge_id in range(edge_index.shape[1]):
                source_local = int(edge_index[0, edge_id])
                target_local = int(edge_index[1, edge_id])
                source_global = self._global_node_id(relation[0], source_local)
                target_global = self._global_node_id(relation[2], target_local)
                attributes = edge_attr[edge_id]

                self._add_dense_edge(
                    relation_id=relation_id,
                    source_node=source_global,
                    target_node=target_global,
                    attributes=attributes,
                    direction=1.0,
                    adjacency=adjacency,
                    edge_features=edge_features,
                    presence=presence,
                )

                # The stored graph remains directional.  Reverse message flow is
                # derived only in the dense encoder view for dependency and
                # Task-Resource relations so both endpoint types can update.
                if relation in {
                    DAG_DEPENDENCY_RELATION,
                    LOCAL_EXECUTION_RELATION,
                    V2V_OFFLOADING_RELATION,
                    V2I_OFFLOADING_RELATION,
                }:
                    self._add_dense_edge(
                        relation_id=relation_id,
                        source_node=target_global,
                        target_node=source_global,
                        attributes=attributes,
                        direction=-1.0,
                        adjacency=adjacency,
                        edge_features=edge_features,
                        presence=presence,
                    )

    def _add_dense_edge(
        self,
        relation_id: int,
        source_node: int,
        target_node: int,
        attributes: Sequence[float] | np.ndarray,
        direction: float,
        adjacency: np.ndarray,
        edge_features: np.ndarray,
        presence: np.ndarray,
    ) -> None:
        attributes_array = np.asarray(attributes, dtype=np.float32).reshape(-1)

        if len(attributes_array) + 1 > MUX_EDGE_FEATURE_DIM:
            raise ValueError(
                f"Relation edge has {len(attributes_array)} features, but v2 muxGNN "
                f"supports at most {MUX_EDGE_FEATURE_DIM - 1} before direction."
            )

        vector = np.zeros(MUX_EDGE_FEATURE_DIM, dtype=np.float32)
        vector[: len(attributes_array)] = attributes_array
        vector[-1] = float(direction)

        adjacency[relation_id, target_node, source_node] = 1
        edge_features[relation_id, target_node, source_node] = vector
        presence[relation_id, source_node] = 1
        presence[relation_id, target_node] = 1

    # ========================================================
    # Global features
    # ========================================================

    def _build_global_features(
        self,
        data: HeteroData,
        task_features: np.ndarray,
        num_tasks: int,
        num_vehicles: int,
        num_edge_servers: int,
    ) -> np.ndarray:
        denominator = max(num_tasks, 1)
        real_tasks = task_features[:num_tasks]

        ready_ratio = float(
            np.sum(real_tasks[:, self.ready_feature_index] > 0.5) / denominator
        )
        running_ratio = float(
            np.sum(real_tasks[:, self.running_feature_index] > 0.5) / denominator
        )
        completed_ratio = float(
            np.sum(real_tasks[:, self.completed_feature_index] > 0.5) / denominator
        )

        # v2 is a static initial snapshot.  Keep an explicit time channel at zero
        # so the same observation contract can later be extended by dynamic v3.
        simulation_time_norm = float(getattr(data, "episode_elapsed_fraction", 0.0))

        local_edges = int(data[LOCAL_EXECUTION_RELATION].num_edges)
        v2v_edges = int(data[V2V_OFFLOADING_RELATION].num_edges)
        v2i_edges = int(data[V2I_OFFLOADING_RELATION].num_edges)
        contention_edges = int(data[GANG_CONTENTION_RELATION].num_edges)

        possible_v2v = max(num_tasks * max(num_vehicles - 1, 1), 1)
        possible_v2i = max(num_tasks * max(num_edge_servers, 1), 1)
        possible_contention = max(num_tasks * max(num_tasks - 1, 1), 1)
        possible_candidates = max(
            num_tasks * max(num_vehicles + num_edge_servers, 1), 1
        )

        v2v_density = min(v2v_edges / possible_v2v, 1.0)
        v2i_density = min(v2i_edges / possible_v2i, 1.0)
        contention_density = min(contention_edges / possible_contention, 1.0)
        candidate_density = min(
            (local_edges + v2v_edges + v2i_edges) / possible_candidates,
            1.0,
        )

        return np.asarray(
            [
                simulation_time_norm,
                ready_ratio,
                running_ratio,
                completed_ratio,
                v2v_density,
                v2i_density,
                contention_density,
                candidate_density,
            ],
            dtype=np.float32,
        )

    # ========================================================
    # Node mapping / utility
    # ========================================================

    def _global_task_node(self, task_id: int) -> int:
        return int(task_id)

    def _global_vehicle_node(self, vehicle_id: int) -> int:
        return self.max_tasks + int(vehicle_id)

    def _global_edge_server_node(self, edge_server_id: int) -> int:
        return self.max_tasks + self.max_vehicles + int(edge_server_id)

    def _global_node_id(self, node_type: str, local_id: int) -> int:
        if node_type == "task":
            return self._global_task_node(local_id)
        if node_type == "vehicle":
            return self._global_vehicle_node(local_id)
        if node_type == "edge_server":
            return self._global_edge_server_node(local_id)
        raise ValueError(f"Unsupported node type: {node_type}")

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()
