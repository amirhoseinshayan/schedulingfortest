from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .builder import (
    load_dagbench_task_graph,
    load_project_task_profiles,
)

from .dynamic_builder import (
    DynamicScenarioState,
    EdgeServerState,
    VehicleState,
    advance_scenario_time,
    initialize_scenario_state,
    refresh_resource_queue_delays,
)

from .dynamic_settings import (
    EVENT_TIME_EPSILON_SEC,
    MAX_DECISION_EPOCHS,
    MIN_TASK_DURATION_SEC,
)

from .settings import (
    DAGBENCH_WORKFLOW_ID,
    SOURCE_VEHICLE_ID,
    V2I_RANGE_M,
    V2V_RANGE_M,
)


# ============================================================
# Event and action records
# ============================================================


@dataclass(order=True)
class SimulationEvent:
    """Future event stored in the discrete-event queue."""

    time: float
    sequence: int

    event_type: str = field(compare=False)
    task_id: int = field(compare=False)

    resource_type: str = field(compare=False)
    resource_id: int = field(compare=False)

    core_type: str = field(compare=False)
    cores: int = field(compare=False)


@dataclass(frozen=True)
class ReferenceAction:
    """Temporary deterministic action used only to advance dataset simulation."""

    task_id: int
    execution_mode: str

    resource_type: str
    resource_id: int

    core_type: str
    cores: int

    duration: float


@dataclass(frozen=True)
class DecisionEpoch:
    """Metadata for one PPO decision epoch / saved snapshot."""

    index: int
    simulation_time: float
    trigger_event: str

    affected_tasks: Tuple[int, ...] = ()
    started_task: Optional[int] = None
    completed_tasks: Tuple[int, ...] = ()
    released_tasks: Tuple[int, ...] = ()

    def to_metadata(self) -> Dict[str, object]:
        """Convert event information to JSON-safe metadata."""

        return {
            "decision_epoch_index": self.index,
            "simulation_time": self.simulation_time,
            "trigger_event": self.trigger_event,
            "affected_tasks": list(self.affected_tasks),
            "started_task": self.started_task,
            "completed_tasks": list(self.completed_tasks),
            "released_tasks": list(self.released_tasks),
        }


# ============================================================
# Event-driven scenario simulator
# ============================================================


class EventDrivenScenarioSimulator:
    """
    Discrete-event simulator for dynamic MAGNET graph generation.

    Snapshots are produced only at decision-relevant state changes:
    initial task release, task start/resource reservation, and task
    completion together with any newly released successor tasks.

    The deterministic reference dispatcher is NOT the final PPO policy.
    It exists only so Phase-2 dataset generation can advance the system
    until PPO is integrated later.
    """

    def __init__(
        self,
        seed: int,
        workflow_id: str = DAGBENCH_WORKFLOW_ID,
    ) -> None:

        self.seed = seed
        self.workflow_id = workflow_id

        dagbench_data = load_dagbench_task_graph(
            workflow_id
        )

        self.dag = dagbench_data["dag"]

        self.num_tasks = int(
            self.dag.number_of_nodes()
        )

        self.profiles = load_project_task_profiles(
            self.num_tasks
        )

        self.state = initialize_scenario_state(
            seed=seed,
            workflow_id=workflow_id,
        )

        self._event_queue: List[
            SimulationEvent
        ] = []

        self._event_sequence = 0
        self._decision_epoch = 0

    # ========================================================
    # Public API
    # ========================================================

    def iter_decision_epochs(
        self,
    ) -> Iterator[DecisionEpoch]:
        """Yield event-driven decision epochs until all tasks finish."""

        initial_released = tuple(
            int(task_id)
            for task_id in np.flatnonzero(
                self.state.task_ready > 0.5
            )
        )

        yield self._make_epoch(
            trigger_event="initial_task_release",
            affected_tasks=initial_released,
            released_tasks=initial_released,
        )

        while not self._all_tasks_completed():

            if (
                self._decision_epoch
                >= MAX_DECISION_EPOCHS
            ):
                raise RuntimeError(
                    "Maximum decision epochs exceeded. "
                    "Possible event-loop deadlock."
                )

            # If a ready task can start now, reserve resources.
            # This changes the state and therefore creates a new
            # PPO decision epoch at the same simulation time.
            action = self._select_reference_action()

            if action is not None:

                self._start_task(
                    action
                )

                yield self._make_epoch(
                    trigger_event="task_started",
                    affected_tasks=(
                        action.task_id,
                    ),
                    started_task=(
                        action.task_id
                    ),
                )

                continue

            # No task can be dispatched now.
            # Jump directly to the next event instead of using fixed dt.
            if not self._event_queue:

                ready_tasks = [
                    int(task_id)
                    for task_id
                    in np.flatnonzero(
                        self.state.task_ready
                        > 0.5
                    )
                ]

                raise RuntimeError(
                    "Event-driven simulation deadlocked. "
                    f"Ready tasks: {ready_tasks}"
                )

            completed_tasks = (
                self._process_next_completion_batch()
            )

            released_tasks = (
                self._release_newly_ready_tasks()
            )

            affected = tuple(
                sorted(
                    set(completed_tasks)
                    | set(released_tasks)
                )
            )

            if released_tasks:
                trigger = (
                    "task_completion_and_release"
                )
            else:
                trigger = "task_completion"

            yield self._make_epoch(
                trigger_event=trigger,
                affected_tasks=affected,
                completed_tasks=(
                    completed_tasks
                ),
                released_tasks=(
                    released_tasks
                ),
            )

    # ========================================================
    # Decision epoch
    # ========================================================

    def _make_epoch(
        self,
        trigger_event: str,
        affected_tasks: Sequence[int] = (),
        started_task: Optional[int] = None,
        completed_tasks: Sequence[int] = (),
        released_tasks: Sequence[int] = (),
    ) -> DecisionEpoch:

        epoch = DecisionEpoch(
            index=self._decision_epoch,

            simulation_time=float(
                self.state.simulation_time
            ),

            trigger_event=trigger_event,

            affected_tasks=tuple(
                int(task_id)
                for task_id
                in affected_tasks
            ),

            started_task=(
                None
                if started_task is None
                else int(started_task)
            ),

            completed_tasks=tuple(
                int(task_id)
                for task_id
                in completed_tasks
            ),

            released_tasks=tuple(
                int(task_id)
                for task_id
                in released_tasks
            ),
        )

        self._decision_epoch += 1

        return epoch

    # ========================================================
    # Temporary reference dispatcher
    # ========================================================

    def _select_reference_action(
        self,
    ) -> Optional[ReferenceAction]:
        """
        Select a deterministic feasible action.

        This is NOT PPO. It is only used to advance Phase-2
        event-driven scenario generation.
        """

        ready_tasks = [
            int(task_id)
            for task_id
            in np.flatnonzero(
                self.state.task_ready
                > 0.5
            )
        ]

        if not ready_tasks:
            return None

        # Earliest deadline first.
        ready_tasks.sort(
            key=lambda task_id: (
                float(
                    self.profiles[
                        task_id
                    ]["deadline"]
                ),
                task_id,
            )
        )

        for task_id in ready_tasks:

            candidates = (
                self._candidate_actions_for_task(
                    task_id
                )
            )

            if not candidates:
                continue

            mode_priority = {
                "local": 0,
                "v2v": 1,
                "v2i": 2,
            }

            candidates.sort(
                key=lambda action: (
                    action.duration,

                    mode_priority[
                        action.execution_mode
                    ],

                    action.resource_id,

                    0
                    if action.core_type == "a7"
                    else 1,
                )
            )

            return candidates[0]

        return None

    def _candidate_actions_for_task(
        self,
        task_id: int,
    ) -> List[ReferenceAction]:

        profile = self.profiles[
            task_id
        ]

        gang_size = int(
            profile["gang_size"]
        )

        source_vehicle = (
            self.state.vehicles[
                SOURCE_VEHICLE_ID
            ]
        )

        candidates: List[
            ReferenceAction
        ] = []

        # ----------------------------------------------------
        # Local source vehicle
        # ----------------------------------------------------

        candidates.extend(
            self._resource_actions(
                task_id=task_id,
                execution_mode="local",
                resource_type="vehicle",
                resource_id=(
                    source_vehicle.vehicle_id
                ),
                resource=source_vehicle,
                gang_size=gang_size,
                profile=profile,
            )
        )

        # ----------------------------------------------------
        # V2V neighbors
        # ----------------------------------------------------

        for vehicle in (
            self.state.vehicles
        ):

            if (
                vehicle.vehicle_id
                == SOURCE_VEHICLE_ID
            ):
                continue

            distance = self._distance(
                source_vehicle.x,
                source_vehicle.y,
                vehicle.x,
                vehicle.y,
            )

            if distance > V2V_RANGE_M:
                continue

            candidates.extend(
                self._resource_actions(
                    task_id=task_id,
                    execution_mode="v2v",
                    resource_type="vehicle",
                    resource_id=(
                        vehicle.vehicle_id
                    ),
                    resource=vehicle,
                    gang_size=gang_size,
                    profile=profile,
                )
            )

        # ----------------------------------------------------
        # V2I edge servers
        # ----------------------------------------------------

        for edge in (
            self.state.edge_servers
        ):

            distance = self._distance(
                source_vehicle.x,
                source_vehicle.y,
                edge.x,
                edge.y,
            )

            if distance > V2I_RANGE_M:
                continue

            candidates.extend(
                self._resource_actions(
                    task_id=task_id,
                    execution_mode="v2i",
                    resource_type=(
                        "edge_server"
                    ),
                    resource_id=edge.edge_id,
                    resource=edge,
                    gang_size=gang_size,
                    profile=profile,
                )
            )

        return candidates

    def _resource_actions(
        self,
        task_id: int,
        execution_mode: str,
        resource_type: str,
        resource_id: int,
        resource: VehicleState | EdgeServerState,
        gang_size: int,
        profile: Dict[
            str,
            float | int | str,
        ],
    ) -> List[ReferenceAction]:

        actions: List[
            ReferenceAction
        ] = []

        if (
            resource.available_a7
            >= gang_size
        ):

            duration = (
                float(
                    resource.queue_delay
                )
                + float(
                    profile["a7_time"]
                )
                / max(
                    gang_size
                    * resource.compute_scale,
                    1e-8,
                )
            )

            actions.append(
                ReferenceAction(
                    task_id=task_id,
                    execution_mode=(
                        execution_mode
                    ),
                    resource_type=(
                        resource_type
                    ),
                    resource_id=resource_id,
                    core_type="a7",
                    cores=gang_size,
                    duration=max(
                        duration,
                        MIN_TASK_DURATION_SEC,
                    ),
                )
            )

        if (
            resource.available_a12
            >= gang_size
        ):

            duration = (
                float(
                    resource.queue_delay
                )
                + float(
                    profile["a12_time"]
                )
                / max(
                    gang_size
                    * resource.compute_scale,
                    1e-8,
                )
            )

            actions.append(
                ReferenceAction(
                    task_id=task_id,
                    execution_mode=(
                        execution_mode
                    ),
                    resource_type=(
                        resource_type
                    ),
                    resource_id=resource_id,
                    core_type="a12",
                    cores=gang_size,
                    duration=max(
                        duration,
                        MIN_TASK_DURATION_SEC,
                    ),
                )
            )

        return actions

    # ========================================================
    # Task start
    # ========================================================

    def _start_task(
        self,
        action: ReferenceAction,
    ) -> None:

        task_id = action.task_id

        if not (
            self.state.task_ready[
                task_id
            ]
            > 0.5
        ):
            raise RuntimeError(
                f"Task {task_id} is not ready."
            )

        resource = self._get_resource(
            action.resource_type,
            action.resource_id,
        )

        self._reserve_cores(
            resource=resource,
            core_type=action.core_type,
            cores=action.cores,
        )

        self.state.task_ready[
            task_id
        ] = 0.0

        self.state.task_running[
            task_id
        ] = 1.0

        finish_time = (
            self.state.simulation_time
            + action.duration
        )

        heapq.heappush(
            self._event_queue,

            SimulationEvent(
                time=float(
                    finish_time
                ),

                sequence=(
                    self._next_event_sequence()
                ),

                event_type=(
                    "task_completion"
                ),

                task_id=task_id,

                resource_type=(
                    action.resource_type
                ),

                resource_id=(
                    action.resource_id
                ),

                core_type=(
                    action.core_type
                ),

                cores=action.cores,
            ),
        )

    # ========================================================
    # Task completion
    # ========================================================

    def _process_next_completion_batch(
        self,
    ) -> Tuple[int, ...]:

        if not self._event_queue:
            raise RuntimeError(
                "No event available to process."
            )

        next_time = float(
            self._event_queue[0].time
        )

        delta = (
            next_time
            - self.state.simulation_time
        )

        if (
            delta
            < -EVENT_TIME_EPSILON_SEC
        ):
            raise RuntimeError(
                "Event queue moved backward in time."
            )

        if delta > 0.0:

            advance_scenario_time(
                self.state,
                delta,
            )

        # Keep the exact event timestamp.
        self.state.simulation_time = (
            next_time
        )

        completed: List[int] = []

        # Process simultaneous completion events together.
        while self._event_queue:

            event = self._event_queue[0]

            if (
                abs(
                    event.time
                    - next_time
                )
                > EVENT_TIME_EPSILON_SEC
            ):
                break

            event = heapq.heappop(
                self._event_queue
            )

            resource = (
                self._get_resource(
                    event.resource_type,
                    event.resource_id,
                )
            )

            self._release_cores(
                resource=resource,
                core_type=event.core_type,
                cores=event.cores,
            )

            task_id = event.task_id

            self.state.task_running[
                task_id
            ] = 0.0

            self.state.task_completed[
                task_id
            ] = 1.0

            completed.append(
                task_id
            )

        refresh_resource_queue_delays(
            self.state
        )

        return tuple(
            sorted(completed)
        )

    # ========================================================
    # Task release
    # ========================================================

    def _release_newly_ready_tasks(
        self,
    ) -> Tuple[int, ...]:

        released: List[int] = []

        for task_id in range(
            self.num_tasks
        ):

            if (
                self.state.task_released[
                    task_id
                ]
                > 0.5
            ):
                continue

            predecessors = list(
                self.dag.predecessors(
                    task_id
                )
            )

            if not predecessors:
                continue

            if all(
                self.state.task_completed[
                    predecessor
                ]
                > 0.5

                for predecessor
                in predecessors
            ):

                self.state.task_released[
                    task_id
                ] = 1.0

                self.state.task_ready[
                    task_id
                ] = 1.0

                released.append(
                    task_id
                )

        return tuple(
            sorted(released)
        )

    # ========================================================
    # Resource reservation
    # ========================================================

    def _get_resource(
        self,
        resource_type: str,
        resource_id: int,
    ) -> VehicleState | EdgeServerState:

        if resource_type == "vehicle":

            return self.state.vehicles[
                resource_id
            ]

        if (
            resource_type
            == "edge_server"
        ):

            return (
                self.state.edge_servers[
                    resource_id
                ]
            )

        raise ValueError(
            f"Unknown resource type: "
            f"{resource_type}"
        )

    def _reserve_cores(
        self,
        resource: VehicleState | EdgeServerState,
        core_type: str,
        cores: int,
    ) -> None:

        if core_type == "a7":

            if (
                resource.available_a7
                < cores
            ):
                raise RuntimeError(
                    "Insufficient A7 cores."
                )

            resource.available_a7 -= (
                cores
            )

        elif core_type == "a12":

            if (
                resource.available_a12
                < cores
            ):
                raise RuntimeError(
                    "Insufficient A12 cores."
                )

            resource.available_a12 -= (
                cores
            )

        else:
            raise ValueError(
                f"Unknown core type: "
                f"{core_type}"
            )

        refresh_resource_queue_delays(
            self.state
        )

    def _release_cores(
        self,
        resource: VehicleState | EdgeServerState,
        core_type: str,
        cores: int,
    ) -> None:

        if core_type == "a7":

            resource.available_a7 = min(
                resource.capacity_a7,

                resource.available_a7
                + cores,
            )

        elif core_type == "a12":

            resource.available_a12 = min(
                resource.capacity_a12,

                resource.available_a12
                + cores,
            )

        else:
            raise ValueError(
                f"Unknown core type: "
                f"{core_type}"
            )

    # ========================================================
    # Utility
    # ========================================================

    def _all_tasks_completed(
        self,
    ) -> bool:

        return bool(
            np.all(
                self.state.task_completed
                > 0.5
            )
        )

    def _next_event_sequence(
        self,
    ) -> int:

        value = self._event_sequence

        self._event_sequence += 1

        return value

    @staticmethod
    def _distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:

        return float(
            math.hypot(
                x1 - x2,
                y1 - y2,
            )
        )