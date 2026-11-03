from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from .create_dag import CreateDAG


class DagSchedulingEnv(gym.Env):
    """
    Gang-aware heterogeneous DAG scheduling environment.

    Gang semantics
    --------------
    m_i is the exact number of physical cores that task i must reserve
    simultaneously. PPO does NOT choose physical core identities. It chooses:

        task_id + how many of the m_i cores are A12

    Action encoding in Gang mode:

        action = task_id * num_action_slots + slot
        slot = number_of_A12_cores
        number_of_A7_cores = m_i - slot

    With max_gang_size=6, num_action_slots=7 (slots 0..6). Invalid compositions
    are removed by action_mask.

    The environment then chooses concrete physical cores deterministically by
    finding the earliest common free interval and reserves exactly m_i cores
    with the same start and finish time.

    Runtime model
    -------------
    The professor-supplied benchmark data gives single-core A7/A12 execution
    time and power but not measured T_i(m_i). Therefore the default Gang runtime
    is an explicit idealized assumption:

        service_rate = nA7 / T_A7 + nA12 / T_A12
        T_gang = 1 / service_rate
        E_gang = T_gang * (nA7*P_A7 + nA12*P_A12)

    Reward
    ------
    Advisor-requested exponential normalized-energy reward:

        normalized_energy = actual_gang_energy / max_node_energy
        reward = QoS / exp(normalized_energy)
               = QoS * exp(-normalized_energy)

    ``max_node_energy`` is computed once from the loaded DAG as the largest
    feasible Gang energy among all tasks and all valid A7/A12 compositions.
    Therefore every per-action energy used by the reward is normalized by the
    same dataset-level maximum before the exponential energy penalty is applied.

    Optional ablation:
        reward = wE * (QoS / exp(normalized_energy))
                 - wM * delta_makespan / rank_scale
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        row_index: int = 0,
        csv_paths: Optional[Sequence[Union[str, Path]]] = None,
        dag_records: Optional[Sequence[Tuple[Union[str, Path], int]]] = None,
        num_a7_cores: int = 6,
        num_a12_cores: int = 6,
        max_gang_size: int = 6,
        allow_mixed_gangs: bool = True,
        runtime_model: str = "perfect_linear_speedup",
        reward_mode: str = "qos_energy_only",
        reward_weights: Tuple[float, float] = (0.008, 45.0),
        invalid_action_penalty: float = 2.0,
        max_tasks: Optional[int] = None,
        max_edges: Optional[int] = None,
        sample_dags: bool = False,
        seed: Optional[int] = None,
        qos_factor: float = 1.20,
        reward_proposal: str = "A",
    ) -> None:
        super().__init__()

        self.num_a7_cores = int(num_a7_cores)
        self.num_a12_cores = int(num_a12_cores)
        self.max_gang_size = int(max_gang_size)
        self.allow_mixed_gangs = bool(allow_mixed_gangs)
        self.runtime_model = str(runtime_model).lower()
        self.reward_mode = str(reward_mode).lower()
        self.wE, self.wM = map(float, reward_weights)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.sample_dags = bool(sample_dags)
        self.qos_factor = float(qos_factor)
        self.reward_proposal = str(reward_proposal).upper()
        self.np_random = np.random.default_rng(seed)

        if self.num_a7_cores < 0 or self.num_a12_cores < 0:
            raise ValueError("Physical core counts must be non-negative.")
        if self.num_a7_cores + self.num_a12_cores <= 0:
            raise ValueError("At least one physical core is required.")
        if self.max_gang_size <= 0:
            raise ValueError("max_gang_size must be positive.")
        if self.qos_factor <= 1.0:
            raise ValueError("qos_factor must be > 1.0.")
        if self.runtime_model not in {
            "perfect_linear_speedup",
            "dataset_time_is_gang_time",
        }:
            raise ValueError(
                "runtime_model must be 'perfect_linear_speedup' or "
                "'dataset_time_is_gang_time'."
            )
        if self.reward_mode not in {
            "qos_energy_only",
            "qos_energy_makespan",
        }:
            raise ValueError(
                "reward_mode must be 'qos_energy_only' or 'qos_energy_makespan'."
            )
        if self.reward_proposal not in {"A", "B"}:
            raise ValueError("reward_proposal must be 'A' or 'B'.")

        if dag_records is not None:
            self.dag_records = [(str(path), int(index)) for path, index in dag_records]
        elif csv_paths is not None:
            self.dag_records = CreateDAG.list_records(csv_paths)
        elif csv_path is not None:
            self.dag_records = [(str(csv_path), int(row_index))]
        else:
            raise ValueError("Provide csv_path, csv_paths, or dag_records.")

        sizes: List[int] = []
        edge_sizes: List[int] = []
        for path, index in self.dag_records:
            record = CreateDAG.read_record(path, index)
            sizes.append(record.num_subtasks)
            edge_sizes.append(len(record.edges))

        self.max_tasks = int(max_tasks or max(sizes))
        self.max_edges = int(max_edges or max(max(edge_sizes), 1))
        if self.max_tasks < max(sizes):
            raise ValueError("max_tasks is smaller than at least one DAG.")
        if self.max_edges < max(edge_sizes):
            raise ValueError("max_edges is smaller than at least one DAG edge count.")

        self.num_action_slots = self.max_gang_size + 1
        self.node_feature_dim = 12
        self.core_feature_dim = 4
        self.global_feature_dim = 8

        self.action_space = spaces.Discrete(self.max_tasks * self.num_action_slots)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.max_tasks, self.node_feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    0,
                    self.max_tasks - 1,
                    shape=(2, self.max_edges),
                    dtype=np.int64,
                ),
                "edge_mask": spaces.MultiBinary(self.max_edges),
                "node_mask": spaces.MultiBinary(self.max_tasks),
                # Two aggregate resource pools: row 0=A7, row 1=A12.
                "core_features": spaces.Box(
                    0.0,
                    np.inf,
                    shape=(2, self.core_feature_dim),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    0.0,
                    np.inf,
                    shape=(self.global_feature_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(
                    self.max_tasks * self.num_action_slots
                ),
            }
        )

        # DAG/profile state, initialized by reset().
        self.dag: CreateDAG
        self.G: nx.DiGraph
        self.num_tasks = 0
        self.edges: List[Tuple[int, int]] = []
        self.current_record = self.dag_records[0]

        self.exec_times = np.ones((1, 2), dtype=np.float32)  # col0=A7, col1=A12
        self.power_mat = np.ones((1, 2), dtype=np.float32)
        self.single_core_energy = np.ones((1, 2), dtype=np.float32)
        self.deadlines = np.ones(1, dtype=np.float32)
        self.periods = np.ones(1, dtype=np.float32)
        self.m_i = np.ones(1, dtype=np.int32)
        self.rank_u_raw = np.ones(1, dtype=np.float32)

        self.time_scale = 1.0
        self.energy_scale = 1.0
        self.period_scale = 1.0
        self.rank_scale = 1.0
        self.max_global_gang_energy = 1.0
        self.max_node_energy = 1.0
        self.max_vf_dag_energy = 1.0

        # Physical cores: first all A7, then all A12.
        self.physical_core_types: List[int] = []
        self.physical_core_names: List[str] = []
        self.a7_core_ids: List[int] = []
        self.a12_core_ids: List[int] = []
        self._build_physical_core_map()

        # Episode state.
        self.done_mask: np.ndarray
        self.ready_mask: np.ndarray
        self.assigned: Dict[int, List[int]]
        self.start: Dict[int, float]
        self.finish: Dict[int, float]
        self.core_intervals: List[List[Tuple[float, float, int]]]
        self.total_energy = 0.0
        self.total_tardiness = 0.0
        self.makespan = 0.0
        self.steps = 0
        self.qos_values: List[float] = []
        self.gang_widths: List[int] = []
        self.a12_fractions: List[float] = []

        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Static setup and task profiles
    # ------------------------------------------------------------------
    def _build_physical_core_map(self) -> None:
        self.physical_core_types = []
        self.physical_core_names = []
        self.a7_core_ids = []
        self.a12_core_ids = []

        for index in range(self.num_a7_cores):
            core_id = len(self.physical_core_types)
            self.physical_core_types.append(0)
            self.physical_core_names.append(f"A7-{index}")
            self.a7_core_ids.append(core_id)

        for index in range(self.num_a12_cores):
            core_id = len(self.physical_core_types)
            self.physical_core_types.append(1)
            self.physical_core_names.append(f"A12-{index}")
            self.a12_core_ids.append(core_id)

    def _load_dag(self, record: Tuple[str, int]) -> None:
        self.current_record = (str(record[0]), int(record[1]))
        self.dag = CreateDAG(self.current_record[0], self.current_record[1])
        self.G = self.dag.graph
        self.edges = list(self.G.edges())
        self.num_tasks = int(self.G.number_of_nodes())

        self.exec_times = np.stack(
            [
                np.asarray(self.dag.a7_times, dtype=np.float32),
                np.asarray(self.dag.a12_times, dtype=np.float32),
            ],
            axis=1,
        )
        self.power_mat = np.stack(
            [
                np.asarray(self.dag.a7_powers, dtype=np.float32),
                np.asarray(self.dag.a12_powers, dtype=np.float32),
            ],
            axis=1,
        )
        self.single_core_energy = self.exec_times * self.power_mat
        self.deadlines = np.asarray(self.dag.deadlines, dtype=np.float32)
        self.periods = np.asarray(self.dag.periods, dtype=np.float32)
        self.m_i = np.asarray(self.dag.m_i_list, dtype=np.int32)

        oversized = np.where(self.m_i > self.max_gang_size)[0].tolist()
        if oversized:
            raise ValueError(
                f"Tasks {oversized} have m_i > max_gang_size={self.max_gang_size}."
            )
        total_capacity = self.num_a7_cores + self.num_a12_cores
        too_wide_for_hardware = np.where(self.m_i > total_capacity)[0].tolist()
        if too_wide_for_hardware:
            raise ValueError(
                f"Tasks {too_wide_for_hardware} require more than the "
                f"{total_capacity} physical cores."
            )

        for task in range(self.num_tasks):
            if not self._valid_compositions_for_width(int(self.m_i[task])):
                raise ValueError(
                    f"Task {task} with m_i={int(self.m_i[task])} has no valid "
                    "A7/A12 gang composition for the configured hardware."
                )

        avg_duration = np.asarray(
            [self._mean_composition_duration(task) for task in range(self.num_tasks)],
            dtype=np.float32,
        )
        self.rank_u_raw = self._compute_upward_ranks(self.G, avg_duration).astype(
            np.float32
        )

        max_deadline = float(np.max(self.deadlines)) if self.deadlines.size else 1.0
        max_single_time = float(np.max(self.exec_times)) if self.exec_times.size else 1.0
        self.time_scale = max(
            max_deadline,
            self.num_tasks * max_single_time,
            float(np.sum(avg_duration)),
            1e-6,
        )
        self.period_scale = max(float(np.max(self.periods)), 1e-6)
        self.rank_scale = max(float(np.max(self.rank_u_raw)), 1e-6)

        per_task_max_energy: List[float] = []
        all_gang_energies: List[float] = []
        for task in range(self.num_tasks):
            energies = []
            for a7_count, a12_count in self._valid_compositions_for_width(
                int(self.m_i[task])
            ):
                _, energy = self._gang_duration_energy(task, a7_count, a12_count)
                energies.append(float(energy))
                all_gang_energies.append(float(energy))
            per_task_max_energy.append(max(energies))

        # The reward uses one global maximum across all task nodes and all
        # feasible Gang compositions. Per-task energy is divided by this value
        # before QoS is divided by the resulting normalized energy.
        self.max_node_energy = max(max(per_task_max_energy), 1e-6)
        self.max_global_gang_energy = self.max_node_energy  # compatibility alias
        self.max_vf_dag_energy = max(float(np.sum(per_task_max_energy)), 1e-6)
        self.energy_scale = self.max_node_energy

    def _valid_compositions_for_width(self, width: int) -> List[Tuple[int, int]]:
        compositions: List[Tuple[int, int]] = []
        if width <= 0 or width > self.max_gang_size:
            return compositions

        for a12_count in range(width + 1):
            a7_count = width - a12_count
            if a7_count > self.num_a7_cores or a12_count > self.num_a12_cores:
                continue
            if not self.allow_mixed_gangs and a7_count > 0 and a12_count > 0:
                continue
            compositions.append((a7_count, a12_count))
        return compositions

    def valid_slots_for_task(self, task: int) -> List[int]:
        width = int(self.m_i[task])
        return [a12_count for _, a12_count in self._valid_compositions_for_width(width)]

    def _gang_duration_energy(
        self,
        task: int,
        a7_count: int,
        a12_count: int,
    ) -> Tuple[float, float]:
        if a7_count < 0 or a12_count < 0 or a7_count + a12_count <= 0:
            raise ValueError("Gang composition must contain at least one core.")

        time_a7 = float(self.exec_times[task, 0])
        time_a12 = float(self.exec_times[task, 1])
        power_a7 = float(self.power_mat[task, 0])
        power_a12 = float(self.power_mat[task, 1])

        if self.runtime_model == "perfect_linear_speedup":
            service_rate = 0.0
            if a7_count:
                service_rate += a7_count / max(time_a7, 1e-12)
            if a12_count:
                service_rate += a12_count / max(time_a12, 1e-12)
            duration = 1.0 / max(service_rate, 1e-12)
        else:  # dataset_time_is_gang_time
            participating_times: List[float] = []
            if a7_count:
                participating_times.append(time_a7)
            if a12_count:
                participating_times.append(time_a12)
            duration = max(participating_times)

        total_power = a7_count * power_a7 + a12_count * power_a12
        energy = duration * total_power
        return float(duration), float(energy)

    def _mean_composition_duration(self, task: int) -> float:
        durations = [
            self._gang_duration_energy(task, a7_count, a12_count)[0]
            for a7_count, a12_count in self._valid_compositions_for_width(
                int(self.m_i[task])
            )
        ]
        if not durations:
            raise ValueError(f"Task {task} has no valid Gang composition.")
        return float(np.mean(durations))

    @staticmethod
    def _compute_upward_ranks(
        graph: nx.DiGraph,
        average_durations: np.ndarray,
    ) -> np.ndarray:
        rank = np.zeros(graph.number_of_nodes(), dtype=float)
        for node in reversed(list(nx.topological_sort(graph))):
            successors = list(graph.successors(node))
            if not successors:
                rank[node] = float(average_durations[node])
            else:
                rank[node] = float(average_durations[node]) + max(
                    rank[successor] for successor in successors
                )
        return rank

    # ------------------------------------------------------------------
    # Gang calendar / placement
    # ------------------------------------------------------------------
    def ready_time_of(self, task: int) -> float:
        predecessors = list(self.G.predecessors(task))
        if not predecessors:
            return 0.0
        return float(max(self.finish[pred] for pred in predecessors))

    def _core_is_free(
        self,
        core_id: int,
        start: float,
        finish: float,
    ) -> bool:
        for existing_start, existing_finish, _ in self.core_intervals[core_id]:
            if start < existing_finish and finish > existing_start:
                return False
        return True

    def _free_core_ids(
        self,
        pool_ids: Sequence[int],
        start: float,
        finish: float,
    ) -> List[int]:
        return [
            core_id
            for core_id in pool_ids
            if self._core_is_free(core_id, start, finish)
        ]

    def estimate_gang_start_finish(
        self,
        task: int,
        a7_count: int,
        a12_count: int,
    ) -> Tuple[float, float, List[int], float]:
        width = int(self.m_i[task])
        if a7_count + a12_count != width:
            raise ValueError(
                f"Task {task} needs m_i={width}, received "
                f"A7={a7_count}, A12={a12_count}."
            )
        if (a7_count, a12_count) not in self._valid_compositions_for_width(width):
            raise ValueError(f"Infeasible Gang composition: {(a7_count, a12_count)}")

        duration, energy = self._gang_duration_energy(task, a7_count, a12_count)
        ready_time = self.ready_time_of(task)

        candidate_times = {float(ready_time)}
        for core_intervals in self.core_intervals:
            for _, interval_finish, _ in core_intervals:
                if interval_finish >= ready_time:
                    candidate_times.add(float(interval_finish))

        for candidate_start in sorted(candidate_times):
            candidate_finish = candidate_start + duration
            free_a7 = self._free_core_ids(
                self.a7_core_ids, candidate_start, candidate_finish
            )
            free_a12 = self._free_core_ids(
                self.a12_core_ids, candidate_start, candidate_finish
            )
            if len(free_a7) >= a7_count and len(free_a12) >= a12_count:
                selected = sorted(free_a7[:a7_count] + free_a12[:a12_count])
                return (
                    float(candidate_start),
                    float(candidate_finish),
                    selected,
                    float(energy),
                )

        # All existing reservations are finite. Starting after the latest end is
        # always feasible for a statically feasible composition.
        latest_finish = ready_time
        for core_intervals in self.core_intervals:
            for _, interval_finish, _ in core_intervals:
                latest_finish = max(latest_finish, interval_finish)
        candidate_start = float(latest_finish)
        candidate_finish = candidate_start + duration
        free_a7 = self._free_core_ids(self.a7_core_ids, candidate_start, candidate_finish)
        free_a12 = self._free_core_ids(
            self.a12_core_ids, candidate_start, candidate_finish
        )
        if len(free_a7) < a7_count or len(free_a12) < a12_count:
            raise RuntimeError("Could not find a common free interval for a feasible gang.")
        selected = sorted(free_a7[:a7_count] + free_a12[:a12_count])
        return candidate_start, float(candidate_finish), selected, float(energy)

    def best_allocation_for_task(self, task: int) -> int:
        """Return the A12-count slot with earliest finish; tie-break on energy."""
        best_slot = -1
        best_key = (float("inf"), float("inf"), float("inf"))
        width = int(self.m_i[task])
        for slot in self.valid_slots_for_task(task):
            a12_count = int(slot)
            a7_count = width - a12_count
            start, finish, _, energy = self.estimate_gang_start_finish(
                task, a7_count, a12_count
            )
            key = (finish, energy, start)
            if key < best_key:
                best_key = key
                best_slot = slot
        if best_slot < 0:
            raise RuntimeError(f"Task {task} has no valid Gang allocation.")
        return int(best_slot)

    # ------------------------------------------------------------------
    # Observation and masking
    # ------------------------------------------------------------------
    def _update_ready_mask(self) -> None:
        self.ready_mask[:] = False
        for task in range(self.num_tasks):
            if self.done_mask[task]:
                continue
            self.ready_mask[task] = all(
                self.done_mask[pred] for pred in self.G.predecessors(task)
            )

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(
            self.max_tasks * self.num_action_slots,
            dtype=np.int8,
        )
        for task in range(self.num_tasks):
            if not self.ready_mask[task] or self.done_mask[task]:
                continue
            for slot in self.valid_slots_for_task(task):
                action = task * self.num_action_slots + slot
                mask[action] = 1
        return mask

    def _node_features(self) -> np.ndarray:
        features = np.zeros(
            (self.max_tasks, self.node_feature_dim),
            dtype=np.float32,
        )
        for task in range(self.num_tasks):
            features[task] = np.asarray(
                [
                    self.periods[task] / self.period_scale,
                    self.deadlines[task] / self.time_scale,
                    float(self.m_i[task]) / max(self.max_gang_size, 1),
                    self.exec_times[task, 0] / self.time_scale,
                    self.exec_times[task, 1] / self.time_scale,
                    self.single_core_energy[task, 0] / max(self.energy_scale, 1e-6),
                    self.single_core_energy[task, 1] / max(self.energy_scale, 1e-6),
                    self.rank_u_raw[task] / self.rank_scale,
                    float(self.done_mask[task]),
                    float(self.ready_mask[task]),
                    float(self.start.get(task, 0.0)) / self.time_scale,
                    float(self.finish.get(task, 0.0)) / self.time_scale,
                ],
                dtype=np.float32,
            )
        return features

    def _edge_index_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        edge_mask = np.zeros(self.max_edges, dtype=np.int8)
        for index, (source, target) in enumerate(self.edges[: self.max_edges]):
            edge_index[0, index] = int(source)
            edge_index[1, index] = int(target)
            edge_mask[index] = 1
        return edge_index, edge_mask

    def _core_available_time(self, core_id: int) -> float:
        if not self.core_intervals[core_id]:
            return 0.0
        return float(max(interval[1] for interval in self.core_intervals[core_id]))

    def _aggregate_pool_features(
        self,
        processor_type: int,
        pool_ids: Sequence[int],
    ) -> np.ndarray:
        if not pool_ids:
            return np.asarray([float(processor_type), 0.0, 0.0, 0.0], dtype=np.float32)
        available_times = [self._core_available_time(core_id) for core_id in pool_ids]
        reservation_count = sum(len(self.core_intervals[core_id]) for core_id in pool_ids)
        return np.asarray(
            [
                float(processor_type),
                len(pool_ids) / max(self.max_gang_size, 1),
                float(np.mean(available_times)) / self.time_scale,
                reservation_count / max(self.num_tasks * len(pool_ids), 1),
            ],
            dtype=np.float32,
        )

    def _normalized_dag_energy(self) -> float:
        return float(self.total_energy) / max(float(self.max_vf_dag_energy), 1e-6)

    def _gang_usage_ratio(self) -> float:
        if not self.gang_widths:
            return 0.0
        return float(np.mean(np.asarray(self.gang_widths) > 1))

    def _mean_scheduled_width(self) -> float:
        if not self.gang_widths:
            return 0.0
        return float(np.mean(self.gang_widths))

    def _obs(self) -> Dict[str, np.ndarray]:
        self._update_ready_mask()
        edge_index, edge_mask = self._edge_index_and_mask()
        node_mask = np.zeros(self.max_tasks, dtype=np.int8)
        node_mask[: self.num_tasks] = 1

        core_features = np.stack(
            [
                self._aggregate_pool_features(0, self.a7_core_ids),
                self._aggregate_pool_features(1, self.a12_core_ids),
            ],
            axis=0,
        ).astype(np.float32)

        global_features = np.asarray(
            [
                self.steps / max(self.num_tasks, 1),
                float(np.sum(self.done_mask[: self.num_tasks])) / max(self.num_tasks, 1),
                self._normalized_dag_energy(),
                self.total_tardiness / self.time_scale,
                self.makespan / self.time_scale,
                float(np.sum(self.ready_mask[: self.num_tasks])) / max(self.num_tasks, 1),
                self._gang_usage_ratio(),
                self._mean_scheduled_width() / max(self.max_gang_size, 1),
            ],
            dtype=np.float32,
        )

        return {
            "node_features": self._node_features(),
            "edge_index": edge_index,
            "edge_mask": edge_mask,
            "node_mask": node_mask,
            "core_features": core_features,
            "global_features": global_features,
            "action_mask": self._action_mask(),
        }

    # ------------------------------------------------------------------
    # QoS, metrics and Gym API
    # ------------------------------------------------------------------
    def _calculate_qos(self, finish_time: float, deadline: float) -> float:
        if deadline <= 0.0:
            return 0.0
        if finish_time <= deadline:
            return 1.0
        upper = self.qos_factor * deadline
        if finish_time <= upper:
            value = (upper - finish_time) / ((self.qos_factor - 1.0) * deadline)
            return float(np.clip(value, 0.0, 1.0))
        return 0.0

    def _qos_metrics(self) -> Dict[str, float]:
        if not self.qos_values:
            return {
                "average_qos": 0.0,
                "minimum_qos": 0.0,
                "maximum_qos": 0.0,
                "qos_std": 0.0,
                "on_time_task_ratio": 0.0,
                "zero_qos_task_ratio": 0.0,
            }
        values = np.asarray(self.qos_values, dtype=np.float64)
        return {
            "average_qos": float(np.mean(values)),
            "minimum_qos": float(np.min(values)),
            "maximum_qos": float(np.max(values)),
            "qos_std": float(np.std(values)),
            "on_time_task_ratio": float(np.mean(np.isclose(values, 1.0, atol=1e-8))),
            "zero_qos_task_ratio": float(np.mean(values <= 1e-8)),
        }

    def _physical_core_utilization(self) -> float:
        if self.makespan <= 0.0:
            return 0.0
        occupied = 0.0
        for intervals in self.core_intervals:
            occupied += sum(finish - start for start, finish, _ in intervals)
        capacity = len(self.core_intervals) * self.makespan
        return float(occupied / max(capacity, 1e-12))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if self.sample_dags and len(self.dag_records) > 1:
            record_index = int(self.np_random.integers(0, len(self.dag_records)))
            record = self.dag_records[record_index]
        else:
            record = self.dag_records[0]

        self._load_dag(record)
        self.done_mask = np.zeros(self.max_tasks, dtype=bool)
        self.ready_mask = np.zeros(self.max_tasks, dtype=bool)
        self.assigned = {}
        self.start = {}
        self.finish = {}
        self.core_intervals = [
            [] for _ in range(self.num_a7_cores + self.num_a12_cores)
        ]
        self.total_energy = 0.0
        self.total_tardiness = 0.0
        self.makespan = 0.0
        self.steps = 0
        self.qos_values = []
        self.gang_widths = []
        self.a12_fractions = []

        observation = self._obs()
        return observation, {
            "csv_path": self.current_record[0],
            "row_index": self.current_record[1],
            "num_a7_cores": self.num_a7_cores,
            "num_a12_cores": self.num_a12_cores,
            "max_gang_size": self.max_gang_size,
            "runtime_model": self.runtime_model,
            "reward_mode": self.reward_mode,
        }

    def step(self, action: int):
        action = int(action)
        task = action // self.num_action_slots
        slot = action % self.num_action_slots

        current_mask = self._action_mask()
        valid = bool(
            0 <= action < len(current_mask)
            and current_mask[action] == 1
            and 0 <= task < self.num_tasks
            and not self.done_mask[task]
        )
        if not valid:
            observation = self._obs()
            return (
                observation,
                -self.invalid_action_penalty,
                False,
                False,
                {
                    "invalid_action": True,
                    "action": action,
                    "task": task,
                    "slot": slot,
                },
            )

        width = int(self.m_i[task])
        a12_count = int(slot)
        a7_count = width - a12_count

        previous_makespan = self.makespan
        previous_energy = self.total_energy

        start, finish, selected_cores, energy = self.estimate_gang_start_finish(
            task,
            a7_count,
            a12_count,
        )

        for core_id in selected_cores:
            self.core_intervals[core_id].append((start, finish, task))
            self.core_intervals[core_id].sort(key=lambda interval: interval[0])

        deadline = float(self.deadlines[task])
        tardiness = max(0.0, finish - deadline)
        qos = self._calculate_qos(finish, deadline)

        self.assigned[task] = list(selected_cores)
        self.start[task] = float(start)
        self.finish[task] = float(finish)
        self.done_mask[task] = True
        self.steps += 1
        self.total_energy += float(energy)
        self.total_tardiness += float(tardiness)
        self.makespan = max(self.makespan, float(finish))
        self.qos_values.append(float(qos))
        self.gang_widths.append(width)
        self.a12_fractions.append(a12_count / max(width, 1))

        # Normalize the energy of this exact node/action by the maximum feasible
        # node energy in the whole loaded DAG, then apply the advisor-requested
        # exponential energy penalty:
        #     reward = QoS / exp(normalized_energy)
        #            = QoS * exp(-normalized_energy)
        # No lambda coefficient and no additional P95 reward scaling are used.
        normalized_energy = float(energy) / max(float(self.max_node_energy), 1e-8)
        normalized_energy = max(normalized_energy, 1e-8)
        qos_energy_reward = float(qos) / float(np.exp(normalized_energy))
        delta_makespan = self.makespan - previous_makespan

        if self.reward_mode == "qos_energy_only":
            reward = qos_energy_reward
            makespan_penalty = 0.0
        else:
            makespan_penalty = delta_makespan / max(self.rank_scale, 1e-6)
            reward = self.wE * qos_energy_reward - self.wM * makespan_penalty

        reward = float(reward)
        terminated = bool(np.all(self.done_mask[: self.num_tasks]))
        observation = self._obs()
        qos_metrics = self._qos_metrics()

        info = {
            "invalid_action": False,
            "action": action,
            "task": task,
            "slot": slot,
            "a7_count": a7_count,
            "a12_count": a12_count,
            "gang_width": width,
            "selected_core_ids": list(selected_cores),
            "selected_core_names": [self.physical_core_names[i] for i in selected_cores],
            "start": float(start),
            "finish": float(finish),
            "deadline": deadline,
            "energy": float(energy),
            "normalized_energy": float(normalized_energy),
            "max_node_energy": float(self.max_node_energy),
            "tardiness": float(tardiness),
            "qos": float(qos),
            "qos_energy_ratio": float(qos_energy_reward),
            "base_qos_energy_reward": float(qos_energy_reward),
            "raw_qos_energy_reward": float(qos_energy_reward),
            "scaled_qos_energy_reward": float(qos_energy_reward),
            "reward_scale": 1.0,
            "reward_scale_percentile": 0.0,
            "makespan_penalty": float(makespan_penalty),
            "delta_energy": float(self.total_energy - previous_energy),
            "delta_makespan": float(delta_makespan),
            "makespan": float(self.makespan),
            "total_energy": self._normalized_dag_energy(),
            "total_energy_raw": float(self.total_energy),
            "total_tardiness": float(self.total_tardiness),
            "average_qos": qos_metrics["average_qos"],
            "gang_usage_ratio": self._gang_usage_ratio(),
            "mean_scheduled_width": self._mean_scheduled_width(),
            "reward_mode": self.reward_mode,
            "runtime_model": self.runtime_model,
        }
        return observation, reward, terminated, False, info

    def get_schedule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assigned = np.full(
            (self.num_tasks, self.max_gang_size),
            -1,
            dtype=np.int32,
        )
        start = np.zeros(self.num_tasks, dtype=np.float32)
        finish = np.zeros(self.num_tasks, dtype=np.float32)

        for task in range(self.num_tasks):
            core_ids = self.assigned.get(task, [])
            assigned[task, : len(core_ids)] = np.asarray(core_ids, dtype=np.int32)
            start[task] = float(self.start.get(task, 0.0))
            finish[task] = float(self.finish.get(task, 0.0))
        return assigned, start, finish

    def get_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "makespan": float(self.makespan),
            "total_tardiness": float(self.total_tardiness),
            "total_energy": self._normalized_dag_energy(),
            "total_energy_raw": float(self.total_energy),
            "max_vf_dag_energy": float(self.max_vf_dag_energy),
            "gang_usage_ratio": self._gang_usage_ratio(),
            "mean_scheduled_width": self._mean_scheduled_width(),
            "mean_a12_fraction": (
                float(np.mean(self.a12_fractions)) if self.a12_fractions else 0.0
            ),
            "physical_core_utilization": self._physical_core_utilization(),
            "skipped_oversized_count": 0.0,
        }
        metrics.update(self._qos_metrics())
        return metrics

    def validate_schedule(self) -> None:
        """Raise AssertionError if a completed schedule violates a Gang invariant."""
        if not np.all(self.done_mask[: self.num_tasks]):
            raise AssertionError("Schedule is incomplete.")

        for task in range(self.num_tasks):
            core_ids = self.assigned.get(task, [])
            if len(core_ids) != int(self.m_i[task]):
                raise AssertionError(
                    f"Task {task}: allocated {len(core_ids)} cores, expected m_i={int(self.m_i[task])}."
                )
            for predecessor in self.G.predecessors(task):
                if self.start[task] + 1e-9 < self.finish[predecessor]:
                    raise AssertionError(
                        f"Precedence violated: {predecessor} -> {task}."
                    )

        for core_id, intervals in enumerate(self.core_intervals):
            ordered = sorted(intervals, key=lambda item: item[0])
            for previous, current in zip(ordered, ordered[1:]):
                if current[0] < previous[1] - 1e-9:
                    raise AssertionError(
                        f"Physical core overlap on {self.physical_core_names[core_id]}: "
                        f"{previous} vs {current}"
                    )

        if abs(self.makespan - max(self.finish.values())) > 1e-6:
            raise AssertionError("Makespan is inconsistent with task finish times.")

    def render(self) -> None:
        metrics = self.get_metrics()
        print(
            f"DAG={Path(self.current_record[0]).name}:{self.current_record[1]} "
            f"steps={self.steps}/{self.num_tasks} "
            f"makespan={metrics['makespan']:.3f} "
            f"energy={metrics['total_energy']:.6f} "
            f"avg_qos={metrics['average_qos']:.6f} "
            f"mean_width={metrics['mean_scheduled_width']:.3f} "
            f"util={metrics['physical_core_utilization']:.3f}"
        )
