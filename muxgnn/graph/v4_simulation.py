from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .settings import (
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
from .v4_settings import (
    V4_BEHAVIOR_POLICIES,
    V4_EDGE_MIN_FREE_CORES,
    V4_EVENT_TIME_EPSILON_SEC,
    V4_MAX_DECISION_EPOCHS,
    V4_MIN_POSITIVE_DURATION_SEC,
    V4_OTHER_VEHICLE_MIN_FREE_CORES,
    V4_REACHABLE_V2V_MIN_FREE_CORES,
    V4_SOURCE_MIN_FREE_CORES,
)
from .v4_source import load_v4_physical_task_graph


PROCESSOR_A7 = 0
PROCESSOR_A12 = 1


@dataclass(order=True)
class V4CompletionEvent:
    finish_time_sec: float
    sequence: int
    task_id: int = field(compare=False)
    resource_slot: int = field(compare=False)
    processor_index: int = field(compare=False)
    cores: int = field(compare=False)


@dataclass(frozen=True)
class V4ReferenceAction:
    task_id: int
    resource_slot: int
    processor_index: int
    cores: int
    duration_sec: float
    energy_j: float


@dataclass(frozen=True)
class V4DecisionEpoch:
    index: int
    simulation_time_sec: float
    trigger_event: str
    affected_tasks: Tuple[int, ...] = ()
    started_task: Optional[int] = None
    completed_tasks: Tuple[int, ...] = ()
    newly_ready_tasks: Tuple[int, ...] = ()

    def to_metadata(self) -> Dict[str, object]:
        return {
            "decision_epoch_index": int(self.index),
            "simulation_time_sec": float(self.simulation_time_sec),
            "trigger_event": self.trigger_event,
            "affected_tasks": [int(x) for x in self.affected_tasks],
            "started_task": None if self.started_task is None else int(self.started_task),
            "completed_tasks": [int(x) for x in self.completed_tasks],
            "newly_ready_tasks": [int(x) for x in self.newly_ready_tasks],
        }


@dataclass
class V4ScenarioState:
    scenario_id: int
    seed: int
    behavior_policy: str
    simulation_time_sec: float

    ready: np.ndarray
    running: np.ndarray
    completed: np.ndarray

    total_a7: np.ndarray
    total_a12: np.ndarray
    scenario_capacity_a7: np.ndarray
    scenario_capacity_a12: np.ndarray
    free_a7: np.ndarray
    free_a12: np.ndarray

    release_time_sec: np.ndarray
    ready_time_sec: np.ndarray
    task_start_time_sec: np.ndarray
    task_finish_time_sec: np.ndarray
    assigned_resource_slot: np.ndarray
    assigned_processor_index: np.ndarray
    assigned_cores: np.ndarray


class LegacyNativeV4ScenarioSimulator:
    """Generate a valid event-driven physical-time trajectory.

    This simulator is intentionally independent of PPO and of the HEFT baseline.
    It uses simple non-learning behavior policies only to create diverse state
    trajectories for offline graph pretraining.
    """

    def __init__(self, scenario_id: int, seed: int, behavior_policy: str) -> None:
        if behavior_policy not in V4_BEHAVIOR_POLICIES:
            raise ValueError(f"Unsupported v4 behavior policy: {behavior_policy}")

        self.scenario_id = int(scenario_id)
        self.seed = int(seed)
        self.behavior_policy = str(behavior_policy)
        self.rng = np.random.default_rng(self.seed)

        source = load_v4_physical_task_graph()
        self.dag = source["dag"]
        self.profiles = list(source["profiles"])
        self.num_tasks = len(self.profiles)
        self.num_vehicles = NUM_VEHICLES
        self.num_edge_servers = NUM_EDGE_SERVERS
        self.num_resources = self.num_vehicles + self.num_edge_servers

        self.predecessors: List[List[int]] = [[] for _ in range(self.num_tasks)]
        self.successors: List[List[int]] = [[] for _ in range(self.num_tasks)]
        for u, v in source["dependencies"]:
            self.predecessors[int(v)].append(int(u))
            self.successors[int(u)].append(int(v))

        self.gang_size = np.asarray(
            [int(p["gang_size"]) for p in self.profiles], dtype=np.int64
        )
        self.deadline_sec = np.asarray(
            [float(p["deadline_sec"]) for p in self.profiles], dtype=np.float64
        )
        self.release_time_sec = np.asarray(
            [float(p["release_time_sec"]) for p in self.profiles], dtype=np.float64
        )

        self.total_a7 = np.asarray(
            [VEHICLE_A7_CORES] * NUM_VEHICLES + [EDGE_A7_CORES] * NUM_EDGE_SERVERS,
            dtype=np.int64,
        )
        self.total_a12 = np.asarray(
            [VEHICLE_A12_CORES] * NUM_VEHICLES + [EDGE_A12_CORES] * NUM_EDGE_SERVERS,
            dtype=np.int64,
        )
        cap_a7, cap_a12 = self._sample_scenario_capacities()

        ready = np.zeros(self.num_tasks, dtype=bool)
        ready_time_sec = np.full(self.num_tasks, -1.0, dtype=np.float64)
        for task_id in range(self.num_tasks):
            if not self.predecessors[task_id]:
                ready[task_id] = True
                ready_time_sec[task_id] = self.release_time_sec[task_id]

        self.state = V4ScenarioState(
            scenario_id=self.scenario_id,
            seed=self.seed,
            behavior_policy=self.behavior_policy,
            simulation_time_sec=0.0,
            ready=ready,
            running=np.zeros(self.num_tasks, dtype=bool),
            completed=np.zeros(self.num_tasks, dtype=bool),
            total_a7=self.total_a7.copy(),
            total_a12=self.total_a12.copy(),
            scenario_capacity_a7=cap_a7.copy(),
            scenario_capacity_a12=cap_a12.copy(),
            free_a7=cap_a7.copy(),
            free_a12=cap_a12.copy(),
            release_time_sec=self.release_time_sec.copy(),
            ready_time_sec=ready_time_sec,
            task_start_time_sec=np.full(self.num_tasks, -1.0, dtype=np.float64),
            task_finish_time_sec=np.full(self.num_tasks, -1.0, dtype=np.float64),
            assigned_resource_slot=np.full(self.num_tasks, -1, dtype=np.int64),
            assigned_processor_index=np.full(self.num_tasks, -1, dtype=np.int64),
            assigned_cores=np.zeros(self.num_tasks, dtype=np.int64),
        )

        self._events: List[V4CompletionEvent] = []
        self._event_sequence = 0
        self._decision_epoch = 0

    def _sample_scenario_capacities(self) -> Tuple[np.ndarray, np.ndarray]:
        a7: List[int] = []
        a12: List[int] = []
        reachable = {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}
        for vehicle_id in range(NUM_VEHICLES):
            if vehicle_id == SOURCE_VEHICLE_ID:
                minimum = V4_SOURCE_MIN_FREE_CORES
            elif vehicle_id in reachable:
                minimum = V4_REACHABLE_V2V_MIN_FREE_CORES
            else:
                minimum = V4_OTHER_VEHICLE_MIN_FREE_CORES
            a7.append(int(self.rng.integers(minimum, VEHICLE_A7_CORES + 1)))
            a12.append(int(self.rng.integers(minimum, VEHICLE_A12_CORES + 1)))
        for _ in range(NUM_EDGE_SERVERS):
            a7.append(int(self.rng.integers(V4_EDGE_MIN_FREE_CORES, EDGE_A7_CORES + 1)))
            a12.append(int(self.rng.integers(V4_EDGE_MIN_FREE_CORES, EDGE_A12_CORES + 1)))

        result_a7 = np.asarray(a7, dtype=np.int64)
        result_a12 = np.asarray(a12, dtype=np.int64)
        if np.any(result_a7 < 0) or np.any(result_a7 > self.total_a7):
            raise RuntimeError("Invalid sampled A7 scenario capacity")
        if np.any(result_a12 < 0) or np.any(result_a12 > self.total_a12):
            raise RuntimeError("Invalid sampled A12 scenario capacity")

        edge_slot = NUM_VEHICLES
        if result_a7[edge_slot] < MAX_GANG_SIZE and result_a12[edge_slot] < MAX_GANG_SIZE:
            raise RuntimeError("v4 edge-server fallback must fit maximum gang size")
        return result_a7, result_a12

    def _resource_relation_exists(self, task_id: int, resource_slot: int) -> bool:
        width = int(self.gang_size[task_id])
        if resource_slot < NUM_VEHICLES:
            vehicle_id = int(resource_slot)
            reachable = (
                vehicle_id == SOURCE_VEHICLE_ID
                or vehicle_id in {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}
            )
            if not reachable:
                return False
        else:
            edge_id = resource_slot - NUM_VEHICLES
            if edge_id < 0 or edge_id >= NUM_EDGE_SERVERS:
                return False
        return bool(
            self.total_a7[resource_slot] >= width
            or self.total_a12[resource_slot] >= width
        )

    def _duration_energy(
        self, task_id: int, resource_slot: int, processor_index: int
    ) -> Tuple[float, float]:
        del resource_slot
        profile = self.profiles[task_id]
        width = int(self.gang_size[task_id])
        if processor_index == PROCESSOR_A7:
            base_time_sec = float(profile["a7_time_sec"])
            power_w = float(profile["a7_power_w"])
        elif processor_index == PROCESSOR_A12:
            base_time_sec = float(profile["a12_time_sec"])
            power_w = float(profile["a12_power_w"])
        else:
            raise ValueError(processor_index)

        # Supervisor-confirmed semantics: m_i is only the number of simultaneous
        # cores reserved by the task.  It does not divide the measured runtime.
        duration_sec = base_time_sec
        energy_j = duration_sec * power_w
        if not np.isfinite(duration_sec) or duration_sec <= V4_MIN_POSITIVE_DURATION_SEC:
            raise RuntimeError("v4 candidate duration must be finite and positive")
        if not np.isfinite(energy_j) or energy_j <= 0.0:
            raise RuntimeError("v4 candidate energy must be finite and positive")
        return float(duration_sec), float(energy_j)

    def _valid_actions(self) -> List[V4ReferenceAction]:
        actions: List[V4ReferenceAction] = []
        for task_id in np.flatnonzero(self.state.ready).tolist():
            if self.state.running[task_id] or self.state.completed[task_id]:
                continue
            if self.state.release_time_sec[task_id] > self.state.simulation_time_sec + V4_EVENT_TIME_EPSILON_SEC:
                continue
            width = int(self.gang_size[task_id])
            for resource_slot in range(self.num_resources):
                if not self._resource_relation_exists(task_id, resource_slot):
                    continue
                for processor_index in (PROCESSOR_A7, PROCESSOR_A12):
                    free = int(
                        self.state.free_a7[resource_slot]
                        if processor_index == PROCESSOR_A7
                        else self.state.free_a12[resource_slot]
                    )
                    total = int(
                        self.total_a7[resource_slot]
                        if processor_index == PROCESSOR_A7
                        else self.total_a12[resource_slot]
                    )
                    if total < width or free < width:
                        continue
                    duration_sec, energy_j = self._duration_energy(
                        task_id, resource_slot, processor_index
                    )
                    actions.append(V4ReferenceAction(
                        task_id=int(task_id),
                        resource_slot=int(resource_slot),
                        processor_index=int(processor_index),
                        cores=width,
                        duration_sec=duration_sec,
                        energy_j=energy_j,
                    ))
        return actions

    def _select_action(self) -> Optional[V4ReferenceAction]:
        actions = self._valid_actions()
        if not actions:
            return None
        now = float(self.state.simulation_time_sec)

        def tail(a: V4ReferenceAction) -> Tuple[int, int, int]:
            return (int(a.task_id), int(a.resource_slot), int(a.processor_index))

        if self.behavior_policy == "earliest_finish":
            return min(actions, key=lambda a: (
                now + a.duration_sec, a.energy_j, self.deadline_sec[a.task_id], *tail(a)
            ))
        if self.behavior_policy == "edf":
            return min(actions, key=lambda a: (
                self.deadline_sec[a.task_id], now + a.duration_sec, a.energy_j, *tail(a)
            ))
        if self.behavior_policy == "lowest_energy":
            return min(actions, key=lambda a: (
                a.energy_j, now + a.duration_sec, self.deadline_sec[a.task_id], *tail(a)
            ))
        raise RuntimeError(self.behavior_policy)

    def iter_decision_epochs(self) -> Iterator[V4DecisionEpoch]:
        roots = tuple(int(x) for x in np.flatnonzero(self.state.ready).tolist())
        yield self._make_epoch(
            "initial_ready_tasks", affected_tasks=roots, newly_ready_tasks=roots
        )

        while not bool(np.all(self.state.completed)):
            if self._decision_epoch >= V4_MAX_DECISION_EPOCHS:
                raise RuntimeError("v4 maximum decision epochs exceeded")

            action = self._select_action()
            if action is not None:
                self._start_task(action)
                yield self._make_epoch(
                    "task_started", affected_tasks=(action.task_id,), started_task=action.task_id
                )
                continue

            if not self._events:
                pending = np.flatnonzero(~self.state.completed).tolist()
                raise RuntimeError(
                    "v4 trajectory deadlocked with no valid action and no running task. "
                    f"Pending tasks: {pending}"
                )

            completed = self._process_next_completion_batch()
            newly_ready = self._mark_newly_ready_tasks()
            trigger = "task_completion_and_ready" if newly_ready else "task_completion"
            affected = tuple(sorted(set(completed) | set(newly_ready)))
            yield self._make_epoch(
                trigger,
                affected_tasks=affected,
                completed_tasks=completed,
                newly_ready_tasks=newly_ready,
            )

    def _make_epoch(
        self,
        trigger_event: str,
        affected_tasks: Sequence[int] = (),
        started_task: Optional[int] = None,
        completed_tasks: Sequence[int] = (),
        newly_ready_tasks: Sequence[int] = (),
    ) -> V4DecisionEpoch:
        epoch = V4DecisionEpoch(
            index=self._decision_epoch,
            simulation_time_sec=float(self.state.simulation_time_sec),
            trigger_event=str(trigger_event),
            affected_tasks=tuple(int(x) for x in affected_tasks),
            started_task=None if started_task is None else int(started_task),
            completed_tasks=tuple(int(x) for x in completed_tasks),
            newly_ready_tasks=tuple(int(x) for x in newly_ready_tasks),
        )
        self._decision_epoch += 1
        return epoch

    def _start_task(self, action: V4ReferenceAction) -> None:
        task_id = int(action.task_id)
        resource_slot = int(action.resource_slot)
        cores = int(action.cores)
        processor = int(action.processor_index)
        if not self.state.ready[task_id] or self.state.running[task_id] or self.state.completed[task_id]:
            raise RuntimeError("v4 dispatcher attempted invalid task state")

        if processor == PROCESSOR_A7:
            if self.state.free_a7[resource_slot] < cores:
                raise RuntimeError("v4 A7 over-allocation")
            self.state.free_a7[resource_slot] -= cores
        else:
            if self.state.free_a12[resource_slot] < cores:
                raise RuntimeError("v4 A12 over-allocation")
            self.state.free_a12[resource_slot] -= cores

        start_sec = float(self.state.simulation_time_sec)
        if start_sec + V4_EVENT_TIME_EPSILON_SEC < self.state.ready_time_sec[task_id]:
            raise RuntimeError("v4 task started before dependency-ready time")
        if start_sec + V4_EVENT_TIME_EPSILON_SEC < self.state.release_time_sec[task_id]:
            raise RuntimeError("v4 task started before external release time")
        finish_sec = start_sec + float(action.duration_sec)

        self.state.ready[task_id] = False
        self.state.running[task_id] = True
        self.state.task_start_time_sec[task_id] = start_sec
        self.state.task_finish_time_sec[task_id] = finish_sec
        self.state.assigned_resource_slot[task_id] = resource_slot
        self.state.assigned_processor_index[task_id] = processor
        self.state.assigned_cores[task_id] = cores

        self._event_sequence += 1
        heapq.heappush(self._events, V4CompletionEvent(
            finish_time_sec=finish_sec,
            sequence=self._event_sequence,
            task_id=task_id,
            resource_slot=resource_slot,
            processor_index=processor,
            cores=cores,
        ))

    def _process_next_completion_batch(self) -> Tuple[int, ...]:
        next_time_sec = float(self._events[0].finish_time_sec)
        self.state.simulation_time_sec = next_time_sec
        completed: List[int] = []
        while self._events and abs(self._events[0].finish_time_sec - next_time_sec) <= V4_EVENT_TIME_EPSILON_SEC:
            event = heapq.heappop(self._events)
            task_id = int(event.task_id)
            if not self.state.running[task_id]:
                continue
            if event.processor_index == PROCESSOR_A7:
                self.state.free_a7[event.resource_slot] += int(event.cores)
                if self.state.free_a7[event.resource_slot] > self.state.scenario_capacity_a7[event.resource_slot]:
                    raise RuntimeError("v4 A7 release exceeded scenario capacity")
            else:
                self.state.free_a12[event.resource_slot] += int(event.cores)
                if self.state.free_a12[event.resource_slot] > self.state.scenario_capacity_a12[event.resource_slot]:
                    raise RuntimeError("v4 A12 release exceeded scenario capacity")
            self.state.running[task_id] = False
            self.state.completed[task_id] = True
            completed.append(task_id)
        return tuple(sorted(completed))

    def _mark_newly_ready_tasks(self) -> Tuple[int, ...]:
        newly_ready: List[int] = []
        now = float(self.state.simulation_time_sec)
        for task_id in range(self.num_tasks):
            if self.state.ready[task_id] or self.state.running[task_id] or self.state.completed[task_id]:
                continue
            if self.state.task_start_time_sec[task_id] >= 0.0:
                continue
            preds = self.predecessors[task_id]
            if preds and all(self.state.completed[p] for p in preds):
                dependency_ready = max(self.state.task_finish_time_sec[p] for p in preds)
                actual_ready = max(float(self.state.release_time_sec[task_id]), float(dependency_ready))
                if actual_ready > now + V4_EVENT_TIME_EPSILON_SEC:
                    continue
                self.state.ready[task_id] = True
                self.state.ready_time_sec[task_id] = actual_ready
                newly_ready.append(task_id)
        return tuple(newly_ready)
