from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from ..graph.loader import LegacyNativeV2Dataset
from ..graph.magnet_relation_layers import GANG_CONTENTION_RELATION
from .adapter import HeteroMuxAdapter, PAIR_FEATURE_NAMES, RELATION_ID
from .config import (
    MUX_INVALID_ACTION_PENALTY,
    MUX_MAX_RESOURCES,
    MUX_MAX_TASKS,
    MUX_NUM_PROCESSOR_TYPES,
    MUX_QOS_FACTOR,
    MUX_REWARD_PROPOSAL,
    MUX_SCHEDULE_ACTIONS,
)


_EPS = 1e-9


@dataclass(order=True)
class _CompletionEvent:
    finish_time: float
    sequence: int
    task_id: int = field(compare=False)
    resource_slot: int = field(compare=False)
    processor_index: int = field(compare=False)
    cores: int = field(compare=False)


class SimpleMuxGNNSmokeEnv(gym.Env):
    """Minimal event-driven PPO environment for the static Legacy-native v2 graph.

    This environment exists only to verify the complete v2 -> frozen muxGNN -> PPO
    training path before dynamic scenarios, mobility, network delay, queueing, and
    migration are introduced.

    The stored v2 graph is immutable. Runtime state is represented in the dense
    observation produced from that graph:
      * task ready/running/completed flags change online;
      * currently free A7/A12 cores change online;
      * task-resource feasibility/action masks change online;
      * contention among currently-ready tasks is recomputed online;
      * DAG topology and static candidate reachability remain exactly v2.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42) -> None:
        super().__init__()

        dataset = LegacyNativeV2Dataset()
        if len(dataset) != 1:
            raise RuntimeError(
                "Simple muxGNN PPO smoke training expects exactly one Legacy-native v2 graph."
            )

        self.graph = dataset[0]
        self.adapter = HeteroMuxAdapter()
        self.observation_space = self.adapter.observation_space
        self.action_space = gym.spaces.Discrete(MUX_SCHEDULE_ACTIONS)

        self.num_tasks = int(self.graph["task"].num_nodes)
        self.num_vehicles = int(self.graph["vehicle"].num_nodes)
        self.num_edge_servers = int(self.graph["edge_server"].num_nodes)
        self.num_resources = self.num_vehicles + self.num_edge_servers

        if self.num_tasks > MUX_MAX_TASKS:
            raise ValueError("v2 graph exceeds MUX_MAX_TASKS.")
        if self.num_resources > MUX_MAX_RESOURCES:
            raise ValueError("v2 graph exceeds MUX_MAX_RESOURCES.")

        self._template = self.adapter.to_observation(self.graph)
        self._pair_static = self._template["pair_features"].copy()
        self._relation_static = self._template["relation_adjacency"].copy()
        self._relation_edge_static = self._template["relation_edge_features"].copy()
        self._relation_presence_static = self._template["relation_presence"].copy()

        self._ready_index = self.adapter.ready_feature_index
        self._running_index = self.adapter.running_feature_index
        self._completed_index = self.adapter.completed_feature_index
        self._feasible_a7_index = self.adapter.feasible_a7_pair_index
        self._feasible_a12_index = self.adapter.feasible_a12_pair_index
        self._avail_a7_pair_index = PAIR_FEATURE_NAMES.index("available_ratio_a7")
        self._avail_a12_pair_index = PAIR_FEATURE_NAMES.index("available_ratio_a12")

        self._gang_size = self.graph["task"].gang_size.detach().cpu().numpy().astype(np.int64)
        self._deadline = self.graph["task"].deadline.detach().cpu().numpy().astype(np.float64)

        dependency_index = self.graph[("task", "precedes", "task")].edge_index
        dep = dependency_index.detach().cpu().numpy().astype(np.int64)
        self._predecessors: List[List[int]] = [[] for _ in range(self.num_tasks)]
        self._successors: List[List[int]] = [[] for _ in range(self.num_tasks)]
        for source, target in zip(dep[0].tolist(), dep[1].tolist()):
            self._predecessors[target].append(source)
            self._successors[source].append(target)

        self._resource_total_a7 = np.concatenate(
            [
                self.graph["vehicle"].total_a7_cores.detach().cpu().numpy(),
                self.graph["edge_server"].total_a7_cores.detach().cpu().numpy(),
            ]
        ).astype(np.int64)
        self._resource_total_a12 = np.concatenate(
            [
                self.graph["vehicle"].total_a12_cores.detach().cpu().numpy(),
                self.graph["edge_server"].total_a12_cores.detach().cpu().numpy(),
            ]
        ).astype(np.int64)
        self._resource_capacity_a7 = np.concatenate(
            [
                self.graph["vehicle"].available_a7_cores.detach().cpu().numpy(),
                self.graph["edge_server"].available_a7_cores.detach().cpu().numpy(),
            ]
        ).astype(np.int64)
        self._resource_capacity_a12 = np.concatenate(
            [
                self.graph["vehicle"].available_a12_cores.detach().cpu().numpy(),
                self.graph["edge_server"].available_a12_cores.detach().cpu().numpy(),
            ]
        ).astype(np.int64)

        self._candidate_duration = np.zeros(
            (self.num_tasks, self.num_resources, MUX_NUM_PROCESSOR_TYPES),
            dtype=np.float64,
        )
        self._candidate_energy = np.zeros_like(self._candidate_duration)
        self._candidate_static_feasible = np.zeros_like(
            self._candidate_duration, dtype=bool
        )
        self._build_raw_candidate_lookup()

        positive_energies = self._candidate_energy[self._candidate_static_feasible]
        if positive_energies.size == 0 or not np.all(np.isfinite(positive_energies)):
            raise RuntimeError("v2 graph does not contain valid raw task-resource energies.")
        self.max_global_energy = float(np.max(positive_energies))
        if self.max_global_energy <= 0.0:
            raise RuntimeError("Global maximum candidate energy must be positive.")

        self._time_scale = max(float(MUX_QOS_FACTOR * np.max(self._deadline)), 1.0)
        self._seed_value = int(seed)
        self._rng = np.random.default_rng(self._seed_value)

        # Runtime fields are initialized by reset().
        self.current_time = 0.0
        self._free_a7 = self._resource_capacity_a7.copy()
        self._free_a12 = self._resource_capacity_a12.copy()
        self._ready = np.zeros(self.num_tasks, dtype=bool)
        self._running = np.zeros(self.num_tasks, dtype=bool)
        self._completed = np.zeros(self.num_tasks, dtype=bool)
        self._events: List[_CompletionEvent] = []
        self._event_sequence = 0
        self._task_start = np.full(self.num_tasks, np.nan, dtype=np.float64)
        self._task_finish = np.full(self.num_tasks, np.nan, dtype=np.float64)
        self._task_qos = np.zeros(self.num_tasks, dtype=np.float64)
        self._task_energy = np.zeros(self.num_tasks, dtype=np.float64)
        self._task_energy_norm = np.zeros(self.num_tasks, dtype=np.float64)
        self._task_reward = np.zeros(self.num_tasks, dtype=np.float64)
        self._scheduled_count = 0

    # ------------------------------------------------------------------
    # Static candidate lookup
    # ------------------------------------------------------------------

    def _build_raw_candidate_lookup(self) -> None:
        relation_specs: Tuple[Tuple[Tuple[str, str, str], bool], ...] = (
            (("task", "local_execute_on", "vehicle"), False),
            (("task", "v2v_offload_to", "vehicle"), False),
            (("task", "v2i_offload_to", "edge_server"), True),
        )

        for relation, is_edge in relation_specs:
            store = self.graph[relation]
            edge_index = store.edge_index.detach().cpu().numpy().astype(np.int64)
            raw = store.edge_attr_raw.detach().cpu().numpy().astype(np.float64)

            for edge_id in range(edge_index.shape[1]):
                task_id = int(edge_index[0, edge_id])
                local_resource_id = int(edge_index[1, edge_id])
                resource_slot = (
                    self.num_vehicles + local_resource_id
                    if is_edge
                    else local_resource_id
                )
                row = raw[edge_id]

                self._candidate_duration[task_id, resource_slot, 0] = float(row[0])
                self._candidate_duration[task_id, resource_slot, 1] = float(row[1])
                self._candidate_energy[task_id, resource_slot, 0] = float(row[2])
                self._candidate_energy[task_id, resource_slot, 1] = float(row[3])
                self._candidate_static_feasible[task_id, resource_slot, 0] = bool(
                    row[6] > 0.5 and row[0] > 0.0 and row[2] > 0.0
                )
                self._candidate_static_feasible[task_id, resource_slot, 1] = bool(
                    row[7] > 0.5 and row[1] > 0.0 and row[3] > 0.0
                )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_value = int(seed)
            self._rng = np.random.default_rng(self._seed_value)

        self.current_time = 0.0
        self._free_a7 = self._resource_capacity_a7.copy()
        self._free_a12 = self._resource_capacity_a12.copy()
        self._ready[:] = False
        self._running[:] = False
        self._completed[:] = False
        self._events.clear()
        self._event_sequence = 0
        self._task_start[:] = np.nan
        self._task_finish[:] = np.nan
        self._task_qos[:] = 0.0
        self._task_energy[:] = 0.0
        self._task_energy_norm[:] = 0.0
        self._task_reward[:] = 0.0
        self._scheduled_count = 0

        for task_id in range(self.num_tasks):
            if not self._predecessors[task_id]:
                self._ready[task_id] = True

        observation = self._observation()
        return observation, self._info()

    def step(self, action: int):
        action = int(action)
        mask = self._build_runtime_action_mask()

        if action < 0 or action >= MUX_SCHEDULE_ACTIONS or mask[action] == 0:
            return (
                self._observation(),
                -float(MUX_INVALID_ACTION_PENALTY),
                False,
                False,
                {**self._info(), "invalid_action": True},
            )

        task_id, resource_slot, processor_index = self.decode_action(action)
        cores = int(self._gang_size[task_id])
        duration = float(
            self._candidate_duration[task_id, resource_slot, processor_index]
        )
        energy = float(
            self._candidate_energy[task_id, resource_slot, processor_index]
        )

        if processor_index == 0:
            if self._free_a7[resource_slot] < cores:
                raise RuntimeError("Action mask admitted an infeasible A7 allocation.")
            self._free_a7[resource_slot] -= cores
        else:
            if self._free_a12[resource_slot] < cores:
                raise RuntimeError("Action mask admitted an infeasible A12 allocation.")
            self._free_a12[resource_slot] -= cores

        start_time = float(self.current_time)
        finish_time = start_time + duration

        self._ready[task_id] = False
        self._running[task_id] = True
        self._task_start[task_id] = start_time
        self._task_finish[task_id] = finish_time

        self._event_sequence += 1
        heapq.heappush(
            self._events,
            _CompletionEvent(
                finish_time=finish_time,
                sequence=self._event_sequence,
                task_id=task_id,
                resource_slot=resource_slot,
                processor_index=processor_index,
                cores=cores,
            ),
        )

        qos = self._qos(finish_time, float(self._deadline[task_id]))
        normalized_energy = float(energy / self.max_global_energy)
        normalized_energy = float(np.clip(normalized_energy, 0.0, 1.0))

        if MUX_REWARD_PROPOSAL != "B":
            raise RuntimeError(
                "Simple v2 PPO smoke environment expects the final exponential reward proposal B."
            )
        reward = float(qos / math.exp(normalized_energy))

        self._task_qos[task_id] = qos
        self._task_energy[task_id] = energy
        self._task_energy_norm[task_id] = normalized_energy
        self._task_reward[task_id] = reward
        self._scheduled_count += 1

        self._advance_until_action_or_done()
        terminated = bool(np.all(self._completed))

        return self._observation(), reward, terminated, False, self._info()

    # ------------------------------------------------------------------
    # Runtime progression
    # ------------------------------------------------------------------

    def _advance_until_action_or_done(self) -> None:
        while True:
            if np.all(self._completed):
                return
            if np.any(self._build_runtime_action_mask()):
                return
            if not self._events:
                pending = np.where(~self._completed)[0].tolist()
                raise RuntimeError(
                    "Simple v2 environment deadlocked with no valid action and no running task. "
                    f"Pending tasks: {pending}"
                )

            next_time = float(self._events[0].finish_time)
            self.current_time = next_time

            completed_now: List[int] = []
            while self._events and abs(self._events[0].finish_time - next_time) <= _EPS:
                event = heapq.heappop(self._events)
                task_id = int(event.task_id)
                if not self._running[task_id]:
                    continue

                if event.processor_index == 0:
                    self._free_a7[event.resource_slot] += int(event.cores)
                    if self._free_a7[event.resource_slot] > self._resource_capacity_a7[event.resource_slot]:
                        raise RuntimeError("A7 core release exceeded configured v2 capacity.")
                else:
                    self._free_a12[event.resource_slot] += int(event.cores)
                    if self._free_a12[event.resource_slot] > self._resource_capacity_a12[event.resource_slot]:
                        raise RuntimeError("A12 core release exceeded configured v2 capacity.")

                self._running[task_id] = False
                self._completed[task_id] = True
                completed_now.append(task_id)

            for task_id in completed_now:
                for successor in self._successors[task_id]:
                    if self._completed[successor] or self._running[successor]:
                        continue
                    if all(self._completed[pred] for pred in self._predecessors[successor]):
                        self._ready[successor] = True

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self) -> Dict[str, np.ndarray]:
        obs = {key: value.copy() for key, value in self._template.items()}

        # Dynamic task state.
        obs["task_features"][: self.num_tasks, self._ready_index] = self._ready.astype(np.float32)
        obs["task_features"][: self.num_tasks, self._running_index] = self._running.astype(np.float32)
        obs["task_features"][: self.num_tasks, self._completed_index] = self._completed.astype(np.float32)

        # Dynamic resource availability ratios. Resource feature indices 2/3 are
        # available A7/A12 ratios by the v2 schema.
        for resource_slot in range(self.num_resources):
            a7_ratio = float(
                self._free_a7[resource_slot]
                / max(int(self._resource_total_a7[resource_slot]), 1)
            )
            a12_ratio = float(
                self._free_a12[resource_slot]
                / max(int(self._resource_total_a12[resource_slot]), 1)
            )

            if resource_slot < self.num_vehicles:
                obs["vehicle_features"][resource_slot, 2] = a7_ratio
                obs["vehicle_features"][resource_slot, 3] = a12_ratio
            else:
                edge_id = resource_slot - self.num_vehicles
                obs["edge_server_features"][edge_id, 2] = a7_ratio
                obs["edge_server_features"][edge_id, 3] = a12_ratio

        # Candidate pair semantics stay static except for current availability and
        # processor feasibility, which are runtime state.
        obs["pair_features"][:] = self._pair_static
        for task_id in range(self.num_tasks):
            width = int(self._gang_size[task_id])
            for resource_slot in range(self.num_resources):
                pair = obs["pair_features"][task_id, resource_slot]
                if not np.any(pair[:3] > 0.5):
                    continue

                pair[self._avail_a7_pair_index] = float(
                    self._free_a7[resource_slot]
                    / max(int(self._resource_total_a7[resource_slot]), 1)
                )
                pair[self._avail_a12_pair_index] = float(
                    self._free_a12[resource_slot]
                    / max(int(self._resource_total_a12[resource_slot]), 1)
                )
                pair[self._feasible_a7_index] = float(
                    self._candidate_static_feasible[task_id, resource_slot, 0]
                    and self._free_a7[resource_slot] >= width
                )
                pair[self._feasible_a12_index] = float(
                    self._candidate_static_feasible[task_id, resource_slot, 1]
                    and self._free_a12[resource_slot] >= width
                )

        # Keep v2 topology/candidate relations fixed, but update the contention
        # view so it describes currently-ready gangs instead of only the roots.
        obs["relation_adjacency"][:] = self._relation_static
        obs["relation_edge_features"][:] = self._relation_edge_static
        obs["relation_presence"][:] = self._relation_presence_static
        self._update_dynamic_contention(obs)

        obs["action_mask"] = self._build_runtime_action_mask()
        obs["global_features"] = self._global_features(obs)

        if not self.observation_space.contains(obs):
            raise AssertionError("Simple v2 runtime observation left the fixed muxGNN Gym space.")
        return obs

    def _build_runtime_action_mask(self) -> np.ndarray:
        mask = np.zeros(MUX_SCHEDULE_ACTIONS, dtype=np.int8)
        for task_id in range(self.num_tasks):
            if not self._ready[task_id] or self._running[task_id] or self._completed[task_id]:
                continue
            width = int(self._gang_size[task_id])
            for resource_slot in range(self.num_resources):
                for processor_index in range(MUX_NUM_PROCESSOR_TYPES):
                    if not self._candidate_static_feasible[task_id, resource_slot, processor_index]:
                        continue
                    free = (
                        self._free_a7[resource_slot]
                        if processor_index == 0
                        else self._free_a12[resource_slot]
                    )
                    if free < width:
                        continue
                    action = self.encode_action(task_id, resource_slot, processor_index)
                    mask[action] = 1
        return mask

    def _update_dynamic_contention(self, obs: Dict[str, np.ndarray]) -> None:
        relation_id = RELATION_ID[GANG_CONTENTION_RELATION]
        obs["relation_adjacency"][relation_id, :, :] = 0
        obs["relation_edge_features"][relation_id, :, :, :] = 0.0
        obs["relation_presence"][relation_id, :] = 0

        ready_ids = np.where(self._ready)[0].tolist()
        for pos, task_a in enumerate(ready_ids):
            gang_a = int(self._gang_size[task_a])
            for task_b in ready_ids[pos + 1 :]:
                gang_b = int(self._gang_size[task_b])
                if gang_a + gang_b <= 6:
                    continue
                combined = min((gang_a + gang_b) / 12.0, 1.0)
                self._add_contention_direction(
                    obs, relation_id, task_a, task_b, gang_a, gang_b, combined
                )
                self._add_contention_direction(
                    obs, relation_id, task_b, task_a, gang_b, gang_a, combined
                )

    @staticmethod
    def _add_contention_direction(
        obs: Dict[str, np.ndarray],
        relation_id: int,
        source: int,
        target: int,
        source_gang: int,
        target_gang: int,
        combined: float,
    ) -> None:
        obs["relation_adjacency"][relation_id, target, source] = 1
        vector = np.zeros(obs["relation_edge_features"].shape[-1], dtype=np.float32)
        vector[:4] = np.asarray(
            [source_gang / 6.0, target_gang / 6.0, combined, 1.0],
            dtype=np.float32,
        )
        vector[-1] = 1.0
        obs["relation_edge_features"][relation_id, target, source] = vector
        obs["relation_presence"][relation_id, source] = 1
        obs["relation_presence"][relation_id, target] = 1

    def _global_features(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        ready_ratio = float(np.mean(self._ready))
        running_ratio = float(np.mean(self._running))
        completed_ratio = float(np.mean(self._completed))
        simulation_time_norm = float(min(self.current_time / self._time_scale, 1.0))

        relation_id = RELATION_ID[GANG_CONTENTION_RELATION]
        contention_edges = int(np.sum(obs["relation_adjacency"][relation_id]))
        possible_contention = max(self.num_tasks * (self.num_tasks - 1), 1)
        contention_density = min(contention_edges / possible_contention, 1.0)

        pair = obs["pair_features"][: self.num_tasks, : self.num_resources]
        v2v_pairs = int(np.sum(pair[:, : self.num_vehicles, 1] > 0.5))
        v2i_pairs = int(np.sum(pair[:, self.num_vehicles :, 2] > 0.5))
        possible_v2v = max(self.num_tasks * max(self.num_vehicles - 1, 1), 1)
        possible_v2i = max(self.num_tasks * max(self.num_edge_servers, 1), 1)

        action_mask = self._build_runtime_action_mask().reshape(
            MUX_MAX_TASKS, MUX_MAX_RESOURCES, MUX_NUM_PROCESSOR_TYPES
        )
        dynamic_candidate_pairs = int(np.sum(np.any(action_mask > 0, axis=2)))
        possible_candidates = max(self.num_tasks * self.num_resources, 1)

        return np.asarray(
            [
                simulation_time_norm,
                ready_ratio,
                running_ratio,
                completed_ratio,
                min(v2v_pairs / possible_v2v, 1.0),
                min(v2i_pairs / possible_v2i, 1.0),
                contention_density,
                min(dynamic_candidate_pairs / possible_candidates, 1.0),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Metrics / utilities
    # ------------------------------------------------------------------

    @staticmethod
    def encode_action(task_id: int, resource_slot: int, processor_index: int) -> int:
        pair_index = int(task_id) * MUX_MAX_RESOURCES + int(resource_slot)
        return pair_index * MUX_NUM_PROCESSOR_TYPES + int(processor_index)

    @staticmethod
    def decode_action(action: int) -> Tuple[int, int, int]:
        processor_index = int(action) % MUX_NUM_PROCESSOR_TYPES
        pair_index = int(action) // MUX_NUM_PROCESSOR_TYPES
        task_id = pair_index // MUX_MAX_RESOURCES
        resource_slot = pair_index % MUX_MAX_RESOURCES
        return int(task_id), int(resource_slot), int(processor_index)

    @staticmethod
    def _qos(finish_time: float, deadline: float) -> float:
        if deadline <= 0.0:
            return 0.0
        if finish_time <= deadline:
            return 1.0
        upper = MUX_QOS_FACTOR * deadline
        if finish_time >= upper:
            return 0.0
        return float((upper - finish_time) / max(upper - deadline, _EPS))

    def _info(self) -> Dict[str, float | int | bool]:
        scheduled = max(int(self._scheduled_count), 1)
        finished_values = self._task_finish[np.isfinite(self._task_finish)]
        makespan = float(np.max(finished_values)) if finished_values.size else 0.0
        actual_scheduled = int(self._scheduled_count)

        info: Dict[str, float | int | bool] = {
            "simulation_time": float(self.current_time),
            "scheduled_tasks": actual_scheduled,
            "completed_tasks": int(np.sum(self._completed)),
            "running_tasks": int(np.sum(self._running)),
            "ready_tasks": int(np.sum(self._ready)),
            "makespan": makespan,
            "total_energy": float(np.sum(self._task_energy)),
            "mean_normalized_energy": float(
                np.sum(self._task_energy_norm) / scheduled
            ),
            "average_qos": float(np.sum(self._task_qos) / scheduled),
            "on_time_ratio": float(
                np.sum(
                    np.isfinite(self._task_finish)
                    & (self._task_finish <= self._deadline + _EPS)
                )
                / scheduled
            ),
            "episode_reward": float(np.sum(self._task_reward)),
            "max_global_energy": float(self.max_global_energy),
            "paper_grade_evaluation": False,
        }
        return info

    def close(self) -> None:
        return None
