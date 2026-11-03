from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..graph.loader import LegacyNativeV2Dataset
from .config import MUX_ARTIFACT_DIR, MUX_NUM_PROCESSOR_TYPES, MUX_QOS_FACTOR


_EPS = 1e-9
_PROCESSOR_NAMES = ("A7", "A12")


@dataclass(frozen=True)
class HEFTProblem:
    """Immutable scheduling problem extracted from the exact Legacy-native v2 graph.

    The HEFT baseline intentionally depends only on the dataset graph.  It does not
    import, query, or call the PPO policy/environment.  This keeps the two methods
    algorithmically independent while giving them exactly the same DAG, resource
    capacities, task-resource candidates, execution times, energies, gang widths,
    and deadlines.
    """

    dataset_version: str
    num_tasks: int
    num_vehicles: int
    num_edge_servers: int
    num_resources: int
    gang_size: np.ndarray
    deadline: np.ndarray
    predecessors: Tuple[Tuple[int, ...], ...]
    successors: Tuple[Tuple[int, ...], ...]
    resource_capacity_a7: np.ndarray
    resource_capacity_a12: np.ndarray
    candidate_duration: np.ndarray
    candidate_energy: np.ndarray
    candidate_feasible: np.ndarray
    max_global_energy: float


@dataclass(frozen=True)
class HEFTAssignment:
    selection_order: int
    task_id: int
    upward_rank: float
    predecessor_ready_time: float
    resource_slot: int
    resource_type: str
    resource_id: int
    processor_index: int
    processor: str
    gang_width: int
    start: float
    finish: float
    duration: float
    deadline: float
    energy: float
    normalized_energy: float
    qos: float
    reward: float


@dataclass(frozen=True)
class HEFTResult:
    metrics: Dict[str, float | int | bool | str]
    assignments: Tuple[HEFTAssignment, ...]
    upward_ranks: np.ndarray


@dataclass(frozen=True)
class _Reservation:
    start: float
    finish: float
    cores: int
    task_id: int


class _GangPoolCalendar:
    """Insertion-based calendar for one homogeneous processor pool.

    A reservation consumes ``cores`` identical cores over [start, finish).  The
    calendar finds the earliest interval that can fit a Gang task while allowing
    overlap whenever the total concurrent core demand does not exceed the pool
    capacity.  This is stronger and more faithful than a single "next-free" time
    because HEFT may insert work into genuine gaps in an existing schedule.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        if self.capacity < 0:
            raise ValueError("Pool capacity cannot be negative.")
        self.reservations: List[_Reservation] = []

    @staticmethod
    def _overlaps(reservation: _Reservation, start: float, finish: float) -> bool:
        return reservation.start < finish - _EPS and reservation.finish > start + _EPS

    def _interval_is_feasible(self, start: float, duration: float, cores: int) -> bool:
        finish = float(start + duration)
        if cores <= 0 or cores > self.capacity:
            return False
        if duration <= 0.0:
            return False

        overlapping = [
            r for r in self.reservations if self._overlaps(r, start, finish)
        ]
        if not overlapping:
            return True

        boundaries = {float(start), float(finish)}
        for reservation in overlapping:
            boundaries.add(max(float(start), float(reservation.start)))
            boundaries.add(min(float(finish), float(reservation.finish)))
        ordered = sorted(boundaries)

        for left, right in zip(ordered[:-1], ordered[1:]):
            if right - left <= _EPS:
                continue
            probe = (left + right) / 2.0
            used = sum(
                int(r.cores)
                for r in overlapping
                if r.start <= probe + _EPS and probe < r.finish - _EPS
            )
            if used + cores > self.capacity:
                return False
        return True

    def earliest_start(self, release_time: float, duration: float, cores: int) -> float:
        release_time = max(float(release_time), 0.0)
        duration = float(duration)
        cores = int(cores)
        if cores > self.capacity or cores <= 0:
            return float("inf")
        if duration <= 0.0 or not np.isfinite(duration):
            return float("inf")

        # For a fixed set of reservations, the earliest feasible insertion can
        # begin at the release time or immediately after an existing finish.
        candidates = {release_time}
        for reservation in self.reservations:
            if reservation.finish >= release_time - _EPS:
                candidates.add(max(release_time, float(reservation.finish)))

        for candidate in sorted(candidates):
            if self._interval_is_feasible(candidate, duration, cores):
                return float(candidate)

        # The latest reservation finish is always a safe final candidate when
        # the task width fits this pool.
        tail = max(
            [release_time] + [float(r.finish) for r in self.reservations],
        )
        if self._interval_is_feasible(tail, duration, cores):
            return float(tail)
        raise RuntimeError("Gang calendar could not find a feasible insertion slot.")

    def reserve(self, start: float, duration: float, cores: int, task_id: int) -> None:
        start = float(start)
        duration = float(duration)
        cores = int(cores)
        finish = start + duration
        if not self._interval_is_feasible(start, duration, cores):
            raise RuntimeError(
                f"Attempted to overbook a {self.capacity}-core pool for task {task_id}."
            )
        self.reservations.append(
            _Reservation(start=start, finish=finish, cores=cores, task_id=int(task_id))
        )
        self.reservations.sort(key=lambda r: (r.start, r.finish, r.task_id))

    def validate(self) -> None:
        if not self.reservations:
            return
        boundaries = sorted(
            {
                point
                for reservation in self.reservations
                for point in (reservation.start, reservation.finish)
            }
        )
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right - left <= _EPS:
                continue
            probe = (left + right) / 2.0
            used = sum(
                reservation.cores
                for reservation in self.reservations
                if reservation.start <= probe + _EPS
                and probe < reservation.finish - _EPS
            )
            if used > self.capacity:
                raise AssertionError(
                    f"HEFT schedule overbooked pool capacity: {used} > {self.capacity}."
                )


def _to_numpy(tensor, dtype) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype)


def _topological_order(
    predecessors: Sequence[Sequence[int]],
    successors: Sequence[Sequence[int]],
) -> List[int]:
    indegree = [len(preds) for preds in predecessors]
    ready = sorted(i for i, degree in enumerate(indegree) if degree == 0)
    order: List[int] = []

    while ready:
        node = ready.pop(0)
        order.append(node)
        for successor in sorted(successors[node]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()

    if len(order) != len(predecessors):
        raise ValueError("Legacy-native v2 precedence relation is not a DAG.")
    return order


def load_problem_from_v2_graph() -> HEFTProblem:
    """Load exactly the same saved v2 HeteroData consumed by PPO/muxGNN."""

    dataset = LegacyNativeV2Dataset()
    if len(dataset) != 1:
        raise RuntimeError("HEFT v2 comparison expects exactly one saved v2 graph.")
    graph = dataset[0]

    num_tasks = int(graph["task"].num_nodes)
    num_vehicles = int(graph["vehicle"].num_nodes)
    num_edge_servers = int(graph["edge_server"].num_nodes)
    num_resources = num_vehicles + num_edge_servers

    gang_size = _to_numpy(graph["task"].gang_size, np.int64)
    deadline = _to_numpy(graph["task"].deadline, np.float64)

    dependency = _to_numpy(
        graph[("task", "precedes", "task")].edge_index,
        np.int64,
    )
    predecessor_lists: List[List[int]] = [[] for _ in range(num_tasks)]
    successor_lists: List[List[int]] = [[] for _ in range(num_tasks)]
    for source, target in zip(dependency[0].tolist(), dependency[1].tolist()):
        predecessor_lists[int(target)].append(int(source))
        successor_lists[int(source)].append(int(target))
    for values in predecessor_lists:
        values.sort()
    for values in successor_lists:
        values.sort()

    resource_capacity_a7 = np.concatenate(
        [
            _to_numpy(graph["vehicle"].available_a7_cores, np.int64),
            _to_numpy(graph["edge_server"].available_a7_cores, np.int64),
        ]
    )
    resource_capacity_a12 = np.concatenate(
        [
            _to_numpy(graph["vehicle"].available_a12_cores, np.int64),
            _to_numpy(graph["edge_server"].available_a12_cores, np.int64),
        ]
    )

    candidate_duration = np.zeros(
        (num_tasks, num_resources, MUX_NUM_PROCESSOR_TYPES), dtype=np.float64
    )
    candidate_energy = np.zeros_like(candidate_duration)
    candidate_feasible = np.zeros_like(candidate_duration, dtype=bool)

    relation_specs: Tuple[Tuple[Tuple[str, str, str], bool], ...] = (
        (("task", "local_execute_on", "vehicle"), False),
        (("task", "v2v_offload_to", "vehicle"), False),
        (("task", "v2i_offload_to", "edge_server"), True),
    )
    for relation, is_edge_server in relation_specs:
        store = graph[relation]
        edge_index = _to_numpy(store.edge_index, np.int64)
        raw = _to_numpy(store.edge_attr_raw, np.float64)
        for edge_id in range(edge_index.shape[1]):
            task_id = int(edge_index[0, edge_id])
            local_resource_id = int(edge_index[1, edge_id])
            resource_slot = (
                num_vehicles + local_resource_id
                if is_edge_server
                else local_resource_id
            )
            row = raw[edge_id]
            candidate_duration[task_id, resource_slot, 0] = float(row[0])
            candidate_duration[task_id, resource_slot, 1] = float(row[1])
            candidate_energy[task_id, resource_slot, 0] = float(row[2])
            candidate_energy[task_id, resource_slot, 1] = float(row[3])
            candidate_feasible[task_id, resource_slot, 0] = bool(
                row[6] > 0.5 and row[0] > 0.0 and row[2] > 0.0
            )
            candidate_feasible[task_id, resource_slot, 1] = bool(
                row[7] > 0.5 and row[1] > 0.0 and row[3] > 0.0
            )

    for task_id in range(num_tasks):
        if not np.any(candidate_feasible[task_id]):
            raise RuntimeError(f"Task {task_id} has no feasible HEFT candidate in v2.")

    positive_energies = candidate_energy[candidate_feasible]
    if positive_energies.size == 0 or not np.all(np.isfinite(positive_energies)):
        raise RuntimeError("v2 graph has no finite positive candidate energies for HEFT.")
    max_global_energy = float(np.max(positive_energies))
    if max_global_energy <= 0.0:
        raise RuntimeError("HEFT energy normalization requires a positive global maximum.")

    problem = HEFTProblem(
        dataset_version=str(getattr(graph, "dataset_version", "legacy_native_simple_magnet_v2")),
        num_tasks=num_tasks,
        num_vehicles=num_vehicles,
        num_edge_servers=num_edge_servers,
        num_resources=num_resources,
        gang_size=gang_size,
        deadline=deadline,
        predecessors=tuple(tuple(v) for v in predecessor_lists),
        successors=tuple(tuple(v) for v in successor_lists),
        resource_capacity_a7=resource_capacity_a7,
        resource_capacity_a12=resource_capacity_a12,
        candidate_duration=candidate_duration,
        candidate_energy=candidate_energy,
        candidate_feasible=candidate_feasible,
        max_global_energy=max_global_energy,
    )

    # Acyclicity is checked independently rather than trusting the dataset metadata.
    _topological_order(problem.predecessors, problem.successors)
    return problem


class IndependentGangHEFT:
    """Independent Gang-aware HEFT baseline for the Legacy-native v2 problem.

    Algorithm:
      1. Compute standard HEFT upward ranks using the average execution cost over
         every statically valid (resource, processor) candidate. Communication
         cost is zero because v2 intentionally has no network/transfer model.
      2. Visit tasks in descending upward-rank order.
      3. For each candidate, compute predecessor release time plus the earliest
         insertion slot in the corresponding homogeneous Gang core calendar.
      4. Select minimum EFT. Exact ties are resolved deterministically by lower
         task energy, earlier EST, resource slot, then processor index.

    Deadlines, QoS, and the PPO reward are *not* used to make HEFT decisions.
    They are calculated only after the schedule is built, keeping the baseline
    independent of the learned objective.
    """

    def __init__(self, problem: HEFTProblem | None = None) -> None:
        self.problem = problem if problem is not None else load_problem_from_v2_graph()
        self._topological = _topological_order(
            self.problem.predecessors, self.problem.successors
        )
        self._ranks = self._compute_upward_ranks()

    def _compute_upward_ranks(self) -> np.ndarray:
        average_compute = np.zeros(self.problem.num_tasks, dtype=np.float64)
        for task_id in range(self.problem.num_tasks):
            values = self.problem.candidate_duration[task_id][
                self.problem.candidate_feasible[task_id]
            ]
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise RuntimeError(f"Task {task_id} has invalid HEFT computation costs.")
            average_compute[task_id] = float(np.mean(values))

        rank = np.zeros(self.problem.num_tasks, dtype=np.float64)
        for task_id in reversed(self._topological):
            if self.problem.successors[task_id]:
                # v2 has intentionally no communication cost, so c_ij = 0.
                downstream = max(rank[succ] for succ in self.problem.successors[task_id])
            else:
                downstream = 0.0
            rank[task_id] = float(average_compute[task_id] + downstream)
        return rank

    @staticmethod
    def _qos(finish_time: float, deadline: float) -> float:
        if deadline <= 0.0:
            return 0.0
        if finish_time <= deadline + _EPS:
            return 1.0
        upper = float(MUX_QOS_FACTOR * deadline)
        if finish_time >= upper - _EPS:
            return 0.0
        return float((upper - finish_time) / max(upper - deadline, _EPS))

    def schedule(self) -> HEFTResult:
        problem = self.problem
        calendars: Dict[Tuple[int, int], _GangPoolCalendar] = {}
        for resource_slot in range(problem.num_resources):
            calendars[(resource_slot, 0)] = _GangPoolCalendar(
                int(problem.resource_capacity_a7[resource_slot])
            )
            calendars[(resource_slot, 1)] = _GangPoolCalendar(
                int(problem.resource_capacity_a12[resource_slot])
            )

        finish_times = np.full(problem.num_tasks, np.nan, dtype=np.float64)
        assignments: List[HEFTAssignment] = []

        order = sorted(
            range(problem.num_tasks),
            key=lambda task_id: (-float(self._ranks[task_id]), int(task_id)),
        )

        for selection_order, task_id in enumerate(order):
            predecessors = problem.predecessors[task_id]
            if predecessors:
                if not all(np.isfinite(finish_times[pred]) for pred in predecessors):
                    raise RuntimeError(
                        "Upward-rank order violated precedence; HEFT cannot place task "
                        f"{task_id} before all predecessors are scheduled."
                    )
                release_time = max(float(finish_times[pred]) for pred in predecessors)
            else:
                release_time = 0.0

            gang_width = int(problem.gang_size[task_id])
            candidates: List[Tuple[float, float, float, int, int, float, float]] = []

            for resource_slot in range(problem.num_resources):
                for processor_index in range(MUX_NUM_PROCESSOR_TYPES):
                    if not problem.candidate_feasible[
                        task_id, resource_slot, processor_index
                    ]:
                        continue
                    duration = float(
                        problem.candidate_duration[task_id, resource_slot, processor_index]
                    )
                    energy = float(
                        problem.candidate_energy[task_id, resource_slot, processor_index]
                    )
                    calendar = calendars[(resource_slot, processor_index)]
                    start = calendar.earliest_start(
                        release_time=release_time,
                        duration=duration,
                        cores=gang_width,
                    )
                    if not np.isfinite(start):
                        continue
                    finish = float(start + duration)
                    candidates.append(
                        (
                            finish,
                            energy,
                            start,
                            int(resource_slot),
                            int(processor_index),
                            duration,
                            energy,
                        )
                    )

            if not candidates:
                raise RuntimeError(f"HEFT found no feasible placement for task {task_id}.")

            # Primary HEFT objective: earliest finish. Remaining fields are
            # deterministic tie-breaks only; energy never overrides a better EFT.
            best = min(candidates, key=lambda row: row[:5])
            finish, _energy_key, start, resource_slot, processor_index, duration, energy = best

            calendars[(resource_slot, processor_index)].reserve(
                start=start,
                duration=duration,
                cores=gang_width,
                task_id=task_id,
            )
            finish_times[task_id] = float(finish)

            deadline = float(problem.deadline[task_id])
            qos = self._qos(finish, deadline)
            normalized_energy = float(
                np.clip(energy / problem.max_global_energy, 0.0, 1.0)
            )
            reward = float(qos / math.exp(normalized_energy))

            if resource_slot < problem.num_vehicles:
                resource_type = "vehicle"
                resource_id = int(resource_slot)
            else:
                resource_type = "edge_server"
                resource_id = int(resource_slot - problem.num_vehicles)

            assignments.append(
                HEFTAssignment(
                    selection_order=int(selection_order),
                    task_id=int(task_id),
                    upward_rank=float(self._ranks[task_id]),
                    predecessor_ready_time=float(release_time),
                    resource_slot=int(resource_slot),
                    resource_type=resource_type,
                    resource_id=resource_id,
                    processor_index=int(processor_index),
                    processor=_PROCESSOR_NAMES[processor_index],
                    gang_width=gang_width,
                    start=float(start),
                    finish=float(finish),
                    duration=float(duration),
                    deadline=deadline,
                    energy=float(energy),
                    normalized_energy=normalized_energy,
                    qos=float(qos),
                    reward=reward,
                )
            )

        result = HEFTResult(
            metrics=self._build_metrics(assignments),
            assignments=tuple(assignments),
            upward_ranks=self._ranks.copy(),
        )
        self.validate(result, calendars)
        return result

    def _build_metrics(
        self, assignments: Sequence[HEFTAssignment]
    ) -> Dict[str, float | int | bool | str]:
        if len(assignments) != self.problem.num_tasks:
            raise RuntimeError("HEFT metric calculation requires one assignment per task.")
        finish = np.asarray([row.finish for row in assignments], dtype=np.float64)
        qos = np.asarray([row.qos for row in assignments], dtype=np.float64)
        energy = np.asarray([row.energy for row in assignments], dtype=np.float64)
        energy_norm = np.asarray(
            [row.normalized_energy for row in assignments], dtype=np.float64
        )
        reward = np.asarray([row.reward for row in assignments], dtype=np.float64)
        on_time = np.asarray(
            [row.finish <= row.deadline + _EPS for row in assignments], dtype=np.float64
        )
        return {
            "method": "HEFT_GANG_V2",
            "dataset_version": self.problem.dataset_version,
            "deterministic": True,
            "scheduled_tasks": int(len(assignments)),
            "completed_tasks": int(len(assignments)),
            "average_qos": float(np.mean(qos)),
            "mean_normalized_energy": float(np.mean(energy_norm)),
            "total_energy": float(np.sum(energy)),
            "makespan": float(np.max(finish)),
            "on_time_ratio": float(np.mean(on_time)),
            "episode_reward": float(np.sum(reward)),
            "max_global_energy": float(self.problem.max_global_energy),
            "communication_cost_model": "zero_in_v2",
            "decision_objective": "standard_HEFT_minimum_EFT",
        }

    def validate(
        self,
        result: HEFTResult,
        calendars: Dict[Tuple[int, int], _GangPoolCalendar] | None = None,
    ) -> None:
        problem = self.problem
        if len(result.assignments) != problem.num_tasks:
            raise AssertionError("HEFT did not schedule every task exactly once.")

        by_task = {row.task_id: row for row in result.assignments}
        if set(by_task) != set(range(problem.num_tasks)):
            raise AssertionError("HEFT task IDs are missing or duplicated.")

        for task_id, row in by_task.items():
            if row.processor_index not in range(MUX_NUM_PROCESSOR_TYPES):
                raise AssertionError("HEFT selected an invalid processor index.")
            if not problem.candidate_feasible[
                task_id, row.resource_slot, row.processor_index
            ]:
                raise AssertionError(
                    f"HEFT selected a non-candidate/infeasible placement for task {task_id}."
                )
            expected_duration = float(
                problem.candidate_duration[task_id, row.resource_slot, row.processor_index]
            )
            expected_energy = float(
                problem.candidate_energy[task_id, row.resource_slot, row.processor_index]
            )
            if abs(row.duration - expected_duration) > 1e-7:
                raise AssertionError("HEFT duration does not match v2 edge data.")
            if abs(row.energy - expected_energy) > 1e-7:
                raise AssertionError("HEFT energy does not match v2 edge data.")
            if abs(row.finish - (row.start + row.duration)) > 1e-7:
                raise AssertionError("HEFT finish time is inconsistent with start + duration.")
            for predecessor in problem.predecessors[task_id]:
                pred_finish = by_task[predecessor].finish
                if row.start + _EPS < pred_finish:
                    raise AssertionError(
                        f"HEFT precedence violation: task {task_id} starts before predecessor "
                        f"{predecessor} finishes."
                    )

        if calendars is None:
            calendars = {}
            for resource_slot in range(problem.num_resources):
                calendars[(resource_slot, 0)] = _GangPoolCalendar(
                    int(problem.resource_capacity_a7[resource_slot])
                )
                calendars[(resource_slot, 1)] = _GangPoolCalendar(
                    int(problem.resource_capacity_a12[resource_slot])
                )
            for row in result.assignments:
                calendars[(row.resource_slot, row.processor_index)].reserve(
                    row.start, row.duration, row.gang_width, row.task_id
                )

        for calendar in calendars.values():
            calendar.validate()

        numeric_metric_keys = (
            "average_qos",
            "mean_normalized_energy",
            "total_energy",
            "makespan",
            "on_time_ratio",
            "episode_reward",
            "max_global_energy",
        )
        for key in numeric_metric_keys:
            value = float(result.metrics[key])
            if not np.isfinite(value):
                raise AssertionError(f"HEFT metric {key} is not finite.")


def result_detail_frame(result: HEFTResult) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in result.assignments]).sort_values(
        "task_id"
    )


def run_heft_evaluation(
    output_dir: Path | None = None,
) -> HEFTResult:
    scheduler = IndependentGangHEFT()
    result = scheduler.schedule()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "heft_evaluation_summary.json"
        detail_path = output_dir / "heft_schedule_detail.csv"
        summary_path.write_text(
            json.dumps(result.metrics, indent=2), encoding="utf-8"
        )
        result_detail_frame(result).to_csv(detail_path, index=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independent Gang-aware HEFT baseline on the exact Legacy-native v2 "
            "dataset used by simple muxGNN+PPO."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MUX_ARTIFACT_DIR / "simple_v2_smoke" / "evaluation",
    )
    args = parser.parse_args()

    result = run_heft_evaluation(output_dir=args.output_dir)
    print("\n============================================")
    print("Independent HEFT-Gang v2 evaluation")
    print("============================================")
    for key in (
        "scheduled_tasks",
        "completed_tasks",
        "average_qos",
        "mean_normalized_energy",
        "total_energy",
        "makespan",
        "on_time_ratio",
        "episode_reward",
    ):
        print(f"{key:24s}: {result.metrics[key]}")
    print(f"Saved: {args.output_dir / 'heft_evaluation_summary.json'}")
    print(f"Saved: {args.output_dir / 'heft_schedule_detail.csv'}")


if __name__ == "__main__":
    main()
