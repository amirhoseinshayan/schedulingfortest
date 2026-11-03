from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .builder import load_legacy_native_task_graph
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
from .v3_settings import (
    V3_BEHAVIOR_POLICIES,
    V3_EDGE_MIN_FREE_CORES,
    V3_EVENT_TIME_EPSILON,
    V3_MAX_DECISION_EPOCHS,
    V3_MIN_POSITIVE_DURATION,
    V3_OTHER_VEHICLE_MIN_FREE_CORES,
    V3_REACHABLE_V2V_MIN_FREE_CORES,
    V3_SOURCE_MIN_FREE_CORES,
)


PROCESSOR_A7 = 0
PROCESSOR_A12 = 1


@dataclass(order=True)
class V3CompletionEvent:
    finish_time: float
    sequence: int
    task_id: int = field(compare=False)
    resource_slot: int = field(compare=False)
    processor_index: int = field(compare=False)
    cores: int = field(compare=False)


@dataclass(frozen=True)
class V3ReferenceAction:
    task_id: int
    resource_slot: int
    processor_index: int
    cores: int
    duration: float
    energy: float


@dataclass(frozen=True)
class V3DecisionEpoch:
    index: int
    simulation_time: float
    trigger_event: str
    affected_tasks: Tuple[int, ...] = ()
    started_task: Optional[int] = None
    completed_tasks: Tuple[int, ...] = ()
    released_tasks: Tuple[int, ...] = ()

    def to_metadata(self) -> Dict[str, object]:
        return {
            "decision_epoch_index": int(self.index),
            "simulation_time": float(self.simulation_time),
            "trigger_event": self.trigger_event,
            "affected_tasks": [int(x) for x in self.affected_tasks],
            "started_task": None if self.started_task is None else int(self.started_task),
            "completed_tasks": [int(x) for x in self.completed_tasks],
            "released_tasks": [int(x) for x in self.released_tasks],
        }


@dataclass
class V3ScenarioState:
    scenario_id: int
    seed: int
    behavior_policy: str
    simulation_time: float

    ready: np.ndarray
    running: np.ndarray
    completed: np.ndarray

    total_a7: np.ndarray
    total_a12: np.ndarray
    scenario_capacity_a7: np.ndarray
    scenario_capacity_a12: np.ndarray
    free_a7: np.ndarray
    free_a12: np.ndarray

    task_start_time: np.ndarray
    task_finish_time: np.ndarray
    assigned_resource_slot: np.ndarray
    assigned_processor_index: np.ndarray
    assigned_cores: np.ndarray


class LegacyNativeV3ScenarioSimulator:
    """Generate one valid event-driven trajectory for the v3 graph dataset.

    The simulator intentionally contains no PPO, no HEFT upward-rank logic, no
    mobility, no communication delay, no queueing, and no migration.  It only
    creates physically consistent compute-state transitions on the exact Legacy
    DAG.  A small fixed set of non-learning behavior policies is used to provide
    diverse but reproducible trajectories for self-supervised muxGNN data.
    """

    def __init__(
        self,
        scenario_id: int,
        seed: int,
        behavior_policy: str,
    ) -> None:
        if behavior_policy not in V3_BEHAVIOR_POLICIES:
            raise ValueError(f"Unsupported v3 behavior policy: {behavior_policy}")

        self.scenario_id = int(scenario_id)
        self.seed = int(seed)
        self.behavior_policy = str(behavior_policy)
        self.rng = np.random.default_rng(self.seed)

        source = load_legacy_native_task_graph()
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
        self.deadline = np.asarray(
            [float(p["deadline"]) for p in self.profiles], dtype=np.float64
        )

        self.total_a7 = np.asarray(
            [VEHICLE_A7_CORES] * NUM_VEHICLES + [EDGE_A7_CORES] * NUM_EDGE_SERVERS,
            dtype=np.int64,
        )
        self.total_a12 = np.asarray(
            [VEHICLE_A12_CORES] * NUM_VEHICLES + [EDGE_A12_CORES] * NUM_EDGE_SERVERS,
            dtype=np.int64,
        )

        scenario_capacity_a7, scenario_capacity_a12 = self._sample_scenario_capacities()

        self.state = V3ScenarioState(
            scenario_id=self.scenario_id,
            seed=self.seed,
            behavior_policy=self.behavior_policy,
            simulation_time=0.0,
            ready=np.zeros(self.num_tasks, dtype=bool),
            running=np.zeros(self.num_tasks, dtype=bool),
            completed=np.zeros(self.num_tasks, dtype=bool),
            total_a7=self.total_a7.copy(),
            total_a12=self.total_a12.copy(),
            scenario_capacity_a7=scenario_capacity_a7.copy(),
            scenario_capacity_a12=scenario_capacity_a12.copy(),
            free_a7=scenario_capacity_a7.copy(),
            free_a12=scenario_capacity_a12.copy(),
            task_start_time=np.full(self.num_tasks, -1.0, dtype=np.float64),
            task_finish_time=np.full(self.num_tasks, -1.0, dtype=np.float64),
            assigned_resource_slot=np.full(self.num_tasks, -1, dtype=np.int64),
            assigned_processor_index=np.full(self.num_tasks, -1, dtype=np.int64),
            assigned_cores=np.zeros(self.num_tasks, dtype=np.int64),
        )

        for task_id in range(self.num_tasks):
            if not self.predecessors[task_id]:
                self.state.ready[task_id] = True

        self._events: List[V3CompletionEvent] = []
        self._event_sequence = 0
        self._decision_epoch = 0

    # ------------------------------------------------------------------
    # Scenario initialization
    # ------------------------------------------------------------------

    def _sample_scenario_capacities(self) -> Tuple[np.ndarray, np.ndarray]:
        a7: List[int] = []
        a12: List[int] = []
        reachable = {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}

        for vehicle_id in range(NUM_VEHICLES):
            if vehicle_id == SOURCE_VEHICLE_ID:
                minimum = V3_SOURCE_MIN_FREE_CORES
            elif vehicle_id in reachable:
                minimum = V3_REACHABLE_V2V_MIN_FREE_CORES
            else:
                minimum = V3_OTHER_VEHICLE_MIN_FREE_CORES

            a7.append(int(self.rng.integers(minimum, VEHICLE_A7_CORES + 1)))
            a12.append(int(self.rng.integers(minimum, VEHICLE_A12_CORES + 1)))

        for _ in range(NUM_EDGE_SERVERS):
            a7.append(int(self.rng.integers(V3_EDGE_MIN_FREE_CORES, EDGE_A7_CORES + 1)))
            a12.append(int(self.rng.integers(V3_EDGE_MIN_FREE_CORES, EDGE_A12_CORES + 1)))

        result_a7 = np.asarray(a7, dtype=np.int64)
        result_a12 = np.asarray(a12, dtype=np.int64)

        if np.any(result_a7 < 0) or np.any(result_a7 > self.total_a7):
            raise RuntimeError("Invalid sampled A7 scenario capacity.")
        if np.any(result_a12 < 0) or np.any(result_a12 > self.total_a12):
            raise RuntimeError("Invalid sampled A12 scenario capacity.")

        # The edge server is deliberately guaranteed to fit every m_i<=6 task,
        # which prevents scenario deadlock while still allowing vehicle scarcity.
        edge_slot = NUM_VEHICLES
        if result_a7[edge_slot] < MAX_GANG_SIZE and result_a12[edge_slot] < MAX_GANG_SIZE:
            raise RuntimeError("v3 edge-server fallback must fit the maximum gang size.")

        return result_a7, result_a12

    # ------------------------------------------------------------------
    # Static candidate semantics
    # ------------------------------------------------------------------

    def _resource_relation_exists(self, task_id: int, resource_slot: int) -> bool:
        width = int(self.gang_size[task_id])
        if resource_slot < NUM_VEHICLES:
            vehicle_id = int(resource_slot)
            if vehicle_id == SOURCE_VEHICLE_ID:
                reachable = True
            else:
                reachable = vehicle_id in {int(x) for x in V2V_REACHABLE_VEHICLE_IDS}
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
        self,
        task_id: int,
        resource_slot: int,
        processor_index: int,
    ) -> Tuple[float, float]:
        del resource_slot  # compute_scale/power_scale are intentionally 1.0 in v3.
        profile = self.profiles[task_id]
        width = int(self.gang_size[task_id])
        if processor_index == PROCESSOR_A7:
            base_time = float(profile["a7_time"])
            power = float(profile["a7_power"])
        elif processor_index == PROCESSOR_A12:
            base_time = float(profile["a12_time"])
            power = float(profile["a12_power"])
        else:
            raise ValueError(processor_index)

        duration = base_time / float(width)
        energy = duration * float(width) * power
        if not np.isfinite(duration) or duration <= V3_MIN_POSITIVE_DURATION:
            raise RuntimeError("v3 candidate duration must be finite and positive.")
        if not np.isfinite(energy) or energy <= 0.0:
            raise RuntimeError("v3 candidate energy must be finite and positive.")
        return float(duration), float(energy)

    def _valid_actions(self) -> List[V3ReferenceAction]:
        actions: List[V3ReferenceAction] = []
        for task_id in np.flatnonzero(self.state.ready).tolist():
            if self.state.running[task_id] or self.state.completed[task_id]:
                continue
            width = int(self.gang_size[task_id])
            for resource_slot in range(self.num_resources):
                if not self._resource_relation_exists(task_id, resource_slot):
                    continue
                for processor_index in (PROCESSOR_A7, PROCESSOR_A12):
                    free = (
                        int(self.state.free_a7[resource_slot])
                        if processor_index == PROCESSOR_A7
                        else int(self.state.free_a12[resource_slot])
                    )
                    total = (
                        int(self.total_a7[resource_slot])
                        if processor_index == PROCESSOR_A7
                        else int(self.total_a12[resource_slot])
                    )
                    if total < width or free < width:
                        continue
                    duration, energy = self._duration_energy(
                        task_id, resource_slot, processor_index
                    )
                    actions.append(
                        V3ReferenceAction(
                            task_id=int(task_id),
                            resource_slot=int(resource_slot),
                            processor_index=int(processor_index),
                            cores=width,
                            duration=duration,
                            energy=energy,
                        )
                    )
        return actions

    def _select_action(self) -> Optional[V3ReferenceAction]:
        actions = self._valid_actions()
        if not actions:
            return None

        now = float(self.state.simulation_time)

        def common_tail(action: V3ReferenceAction) -> Tuple[int, int, int]:
            return (
                int(action.task_id),
                int(action.resource_slot),
                int(action.processor_index),
            )

        if self.behavior_policy == "earliest_finish":
            return min(
                actions,
                key=lambda a: (
                    now + a.duration,
                    a.energy,
                    self.deadline[a.task_id],
                    *common_tail(a),
                ),
            )
        if self.behavior_policy == "edf":
            return min(
                actions,
                key=lambda a: (
                    self.deadline[a.task_id],
                    now + a.duration,
                    a.energy,
                    *common_tail(a),
                ),
            )
        if self.behavior_policy == "lowest_energy":
            return min(
                actions,
                key=lambda a: (
                    a.energy,
                    now + a.duration,
                    self.deadline[a.task_id],
                    *common_tail(a),
                ),
            )
        raise RuntimeError(self.behavior_policy)

    # ------------------------------------------------------------------
    # Event-driven progression
    # ------------------------------------------------------------------

    def iter_decision_epochs(self) -> Iterator[V3DecisionEpoch]:
        roots = tuple(int(x) for x in np.flatnonzero(self.state.ready).tolist())
        yield self._make_epoch(
            "initial_task_release",
            affected_tasks=roots,
            released_tasks=roots,
        )

        while not bool(np.all(self.state.completed)):
            if self._decision_epoch >= V3_MAX_DECISION_EPOCHS:
                raise RuntimeError("v3 maximum decision epochs exceeded.")

            action = self._select_action()
            if action is not None:
                self._start_task(action)
                yield self._make_epoch(
                    "task_started",
                    affected_tasks=(action.task_id,),
                    started_task=action.task_id,
                )
                continue

            if not self._events:
                pending = np.flatnonzero(~self.state.completed).tolist()
                raise RuntimeError(
                    "v3 event-driven trajectory deadlocked with no valid action "
                    f"and no running task. Pending tasks: {pending}"
                )

            completed = self._process_next_completion_batch()
            released = self._release_newly_ready_tasks()
            trigger = (
                "task_completion_and_release" if released else "task_completion"
            )
            affected = tuple(sorted(set(completed) | set(released)))
            yield self._make_epoch(
                trigger,
                affected_tasks=affected,
                completed_tasks=completed,
                released_tasks=released,
            )

    def _make_epoch(
        self,
        trigger_event: str,
        affected_tasks: Sequence[int] = (),
        started_task: Optional[int] = None,
        completed_tasks: Sequence[int] = (),
        released_tasks: Sequence[int] = (),
    ) -> V3DecisionEpoch:
        epoch = V3DecisionEpoch(
            index=self._decision_epoch,
            simulation_time=float(self.state.simulation_time),
            trigger_event=str(trigger_event),
            affected_tasks=tuple(int(x) for x in affected_tasks),
            started_task=None if started_task is None else int(started_task),
            completed_tasks=tuple(int(x) for x in completed_tasks),
            released_tasks=tuple(int(x) for x in released_tasks),
        )
        self._decision_epoch += 1
        return epoch

    def _start_task(self, action: V3ReferenceAction) -> None:
        task_id = int(action.task_id)
        resource_slot = int(action.resource_slot)
        cores = int(action.cores)
        processor = int(action.processor_index)

        if not self.state.ready[task_id] or self.state.running[task_id] or self.state.completed[task_id]:
            raise RuntimeError("v3 dispatcher attempted to start an invalid task state.")

        if processor == PROCESSOR_A7:
            if self.state.free_a7[resource_slot] < cores:
                raise RuntimeError("v3 A7 over-allocation.")
            self.state.free_a7[resource_slot] -= cores
        else:
            if self.state.free_a12[resource_slot] < cores:
                raise RuntimeError("v3 A12 over-allocation.")
            self.state.free_a12[resource_slot] -= cores

        start = float(self.state.simulation_time)
        finish = start + float(action.duration)

        self.state.ready[task_id] = False
        self.state.running[task_id] = True
        self.state.task_start_time[task_id] = start
        self.state.task_finish_time[task_id] = finish
        self.state.assigned_resource_slot[task_id] = resource_slot
        self.state.assigned_processor_index[task_id] = processor
        self.state.assigned_cores[task_id] = cores

        self._event_sequence += 1
        heapq.heappush(
            self._events,
            V3CompletionEvent(
                finish_time=finish,
                sequence=self._event_sequence,
                task_id=task_id,
                resource_slot=resource_slot,
                processor_index=processor,
                cores=cores,
            ),
        )

    def _process_next_completion_batch(self) -> Tuple[int, ...]:
        next_time = float(self._events[0].finish_time)
        self.state.simulation_time = next_time
        completed: List[int] = []

        while self._events and abs(self._events[0].finish_time - next_time) <= V3_EVENT_TIME_EPSILON:
            event = heapq.heappop(self._events)
            task_id = int(event.task_id)
            if not self.state.running[task_id]:
                continue

            if event.processor_index == PROCESSOR_A7:
                self.state.free_a7[event.resource_slot] += int(event.cores)
                if self.state.free_a7[event.resource_slot] > self.state.scenario_capacity_a7[event.resource_slot]:
                    raise RuntimeError("v3 A7 release exceeded scenario capacity.")
            else:
                self.state.free_a12[event.resource_slot] += int(event.cores)
                if self.state.free_a12[event.resource_slot] > self.state.scenario_capacity_a12[event.resource_slot]:
                    raise RuntimeError("v3 A12 release exceeded scenario capacity.")

            self.state.running[task_id] = False
            self.state.completed[task_id] = True
            completed.append(task_id)

        return tuple(sorted(completed))

    def _release_newly_ready_tasks(self) -> Tuple[int, ...]:
        released: List[int] = []
        for task_id in range(self.num_tasks):
            if self.state.ready[task_id] or self.state.running[task_id] or self.state.completed[task_id]:
                continue
            if self.state.task_start_time[task_id] >= 0.0:
                continue
            predecessors = self.predecessors[task_id]
            if predecessors and all(self.state.completed[pred] for pred in predecessors):
                self.state.ready[task_id] = True
                released.append(task_id)
        return tuple(released)
