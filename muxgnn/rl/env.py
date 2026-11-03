from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..graph.builder import (
    load_dagbench_task_graph,
    load_project_task_profiles,
)
from ..graph.dynamic_builder import (
    DynamicScenarioState,
    EdgeServerState,
    VehicleState,
    advance_scenario_time,
    build_dynamic_hetero_graph,
    initialize_scenario_state,
    refresh_resource_queue_delays,
)
from ..graph.dynamic_settings import (
    CONTACT_DURATION_NORM_SEC,
    DYNAMIC_OUTPUT_DIR,
    MANIFEST_PATH,
)
from ..graph.settings import (
    BASE_V2I_RATE_MBPS,
    BASE_V2V_RATE_MBPS,
    DAGBENCH_WORKFLOW_ID,
    V2I_RANGE_M,
    V2V_RANGE_M,
)
from .adapter import HeteroMuxAdapter
from .config import (
    MUX_INVALID_ACTION_PENALTY,
    MUX_MAX_EDGE_SERVERS,
    MUX_MAX_RESOURCES,
    MUX_MAX_TASKS,
    MUX_MAX_VEHICLES,
    MUX_MIGRATION_ACTIONS,
    MUX_MIGRATION_CHECK_MIN_INTERVAL_SEC,
    MUX_MIGRATION_CHECK_REMAINING_RATIO,
    MUX_MIGRATION_CONTACT_SAFETY_FACTOR,
    MUX_MIGRATION_CONTROL_POWER_W,
    MUX_MIGRATION_COOLDOWN_SEC,
    MUX_MIGRATION_FEATURE_DIM,
    MUX_MIGRATION_MAX_PER_TASK,
    MUX_MIGRATION_MIN_REMAINING_FRACTION,
    MUX_MIGRATION_MIN_RUNTIME_SEC,
    MUX_MIGRATION_PREEMPT_SEC,
    MUX_MIGRATION_RESUME_SEC,
    MUX_MIGRATION_SERIALIZE_SEC_PER_MB,
    MUX_MIGRATION_STATE_BASE_MB,
    MUX_MIGRATION_STATE_COST_MB,
    MUX_MIGRATION_STATE_NORM_MB,
    MUX_MIGRATION_STATE_PER_CORE_MB,
    MUX_MIGRATION_TRANSFER_DELAY_NORM_SEC,
    MUX_MIGRATION_TX_ENERGY_J_PER_MB,
    MUX_NUM_PROCESSOR_TYPES,
    MUX_PROCESSOR_TYPES,
    MUX_QOS_FACTOR,
    MUX_REWARD_PROPOSAL,
    MUX_SCHEDULE_ACTIONS,
    MUX_TOTAL_ACTIONS,
)
from .migration import MIGRATION_FEATURE_NAMES


_EPS = 1e-9


@dataclass(order=True)
class _CompletionEvent:
    """Versioned task-completion event; stale events are ignored after migration."""

    time: float
    sequence: int

    task_id: int = field(compare=False)
    resource_type: str = field(compare=False)
    resource_id: int = field(compare=False)
    core_type: str = field(compare=False)
    cores: int = field(compare=False)
    generation: int = field(compare=False)


@dataclass(frozen=True)
class _ExecutionChoice:
    task_id: int
    resource_slot: int
    resource_type: str
    resource_id: int
    execution_mode: str
    core_type: str
    cores: int
    exec_time: float
    queue_delay: float
    duration: float
    energy: float
    finish_time: float


@dataclass(frozen=True)
class _MigrationChoice:
    task_id: int
    source_slot: int
    target_slot: int
    source_core_type: str
    target_core_type: str
    target_resource_type: str
    target_resource_id: int
    cores: int
    is_keep: bool
    progress: float
    remaining_fraction: float
    state_size_mb: float
    link_rate_mbps: float
    contact_duration: float
    preempt_time: float
    transfer_time: float
    resume_time: float
    queue_delay: float
    remaining_exec_time: float
    remaining_compute_energy: float
    migration_energy: float
    duration: float
    finish_time: float
    old_duration: float
    old_remaining_energy: float
    old_finish_time: float


class _ScenarioSeedPool:
    """Read deterministic scenario seeds from the generated dynamic split."""

    def __init__(
        self,
        split: str,
        randomize: bool,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                "Dynamic dataset manifest not found. Run:\n"
                "python -m muxgnn.graph.generate_dynamic_dataset"
            )

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        if manifest.get("snapshot_mode") != "event_driven":
            raise ValueError(
                "The dynamic dataset is not the event-driven version. Run:\n"
                "python -m muxgnn.graph.generate_dynamic_dataset"
            )

        scenario_names = manifest.get("splits", {}).get(split, [])
        if not scenario_names:
            raise ValueError(f"No scenarios found in split {split!r}.")

        self.split = split
        self.randomize = bool(randomize)
        self.records: List[Tuple[str, int]] = []
        self.cursor = 0

        for scenario_name in scenario_names:
            metadata_path = (
                DYNAMIC_OUTPUT_DIR
                / split
                / str(scenario_name)
                / "scenario.json"
            )
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Scenario metadata not found: {metadata_path}"
                )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.records.append(
                (
                    str(metadata["scenario_name"]),
                    int(metadata["seed"]),
                )
            )

    def sample(self, rng: np.random.Generator) -> Tuple[str, int]:
        if self.randomize:
            index = int(rng.integers(0, len(self.records)))
        else:
            index = self.cursor % len(self.records)
            self.cursor += 1
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)


class MuxGNNEventDrivenEnv(gym.Env):
    """
    Event-driven PPO environment with adaptive runtime task migration.

    Scheduling action:
        ready task x resource x processor(A7/A12)

    Migration action:
        running task x destination resource x destination processor(A7/A12)

    A migration window is opened periodically and after task-completion events.
    Each eligible running task receives a KEEP action plus feasible migration
    actions. This prevents forced migration while allowing PPO to learn when a
    runtime move is useful. The stored graph schema and offline muxGNN encoder
    remain unchanged; runtime migration context is supplied to a Transformer
    head through migration_features.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        split: str = "train",
        random_scenarios: bool = True,
        workflow_id: str = DAGBENCH_WORKFLOW_ID,
        invalid_action_penalty: float = MUX_INVALID_ACTION_PENALTY,
        qos_factor: float = MUX_QOS_FACTOR,
        reward_proposal: str = MUX_REWARD_PROPOSAL,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.split = split
        self.workflow_id = workflow_id
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.qos_factor = float(qos_factor)
        self.reward_proposal = str(reward_proposal).upper()

        if self.qos_factor <= 1.0:
            raise ValueError("qos_factor must be > 1.0")
        if self.reward_proposal not in {"A", "B"}:
            raise ValueError("reward_proposal must be 'A' or 'B'.")

        self.adapter = HeteroMuxAdapter()

        # Keep the stored-graph adapter unchanged and augment only the online
        # observation contract with runtime migration tensors.
        observation_spaces = dict(self.adapter.observation_space.spaces)
        observation_spaces["migration_features"] = spaces.Box(
            -np.inf,
            np.inf,
            shape=(
                MUX_MAX_TASKS,
                MUX_MAX_RESOURCES,
                MUX_NUM_PROCESSOR_TYPES,
                MUX_MIGRATION_FEATURE_DIM,
            ),
            dtype=np.float32,
        )
        observation_spaces["migration_candidate_mask"] = spaces.MultiBinary(
            (
                MUX_MAX_TASKS,
                MUX_MAX_RESOURCES,
                MUX_NUM_PROCESSOR_TYPES,
            )
        )
        observation_spaces["migration_task_mask"] = spaces.MultiBinary(
            MUX_MAX_TASKS
        )
        observation_spaces["action_mask"] = spaces.MultiBinary(MUX_TOTAL_ACTIONS)
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Discrete(MUX_TOTAL_ACTIONS)

        self.seed_pool = _ScenarioSeedPool(
            split=split,
            randomize=random_scenarios,
        )
        self.np_random = np.random.default_rng(seed)

        dagbench_data = load_dagbench_task_graph(workflow_id)
        self.dag = dagbench_data["dag"]
        self.num_tasks = int(self.dag.number_of_nodes())
        self.profiles = load_project_task_profiles(self.num_tasks)
        self.task_costs = np.asarray(
            dagbench_data.get("task_costs", [1.0] * self.num_tasks),
            dtype=np.float64,
        )

        if self.num_tasks > MUX_MAX_TASKS:
            raise ValueError(
                f"Current workflow has {self.num_tasks} tasks but "
                f"MUX_MAX_TASKS={MUX_MAX_TASKS}."
            )

        cost_scale = max(float(np.max(np.abs(self.task_costs))), 1e-8)
        cost_norm = np.abs(self.task_costs) / cost_scale
        self.task_state_size_mb = np.zeros(MUX_MAX_TASKS, dtype=np.float64)
        for task_id in range(self.num_tasks):
            self.task_state_size_mb[task_id] = (
                MUX_MIGRATION_STATE_BASE_MB
                + MUX_MIGRATION_STATE_PER_CORE_MB
                * int(self.profiles[task_id]["gang_size"])
                + MUX_MIGRATION_STATE_COST_MB * float(cost_norm[task_id])
            )

        self.state: DynamicScenarioState
        self.scenario_name = ""
        self.scenario_seed = 0

        self._event_queue: List[_CompletionEvent] = []
        self._event_sequence = 0
        self._decision_epoch = 0
        self._next_migration_check_time: Optional[float] = None
        self._migration_pending_tasks: Set[int] = set()
        self.task_next_migration_check_time = np.full(
            MUX_MAX_TASKS, np.inf, dtype=np.float64
        )

        self.task_start_time = np.full(MUX_MAX_TASKS, np.nan, dtype=np.float64)
        self.task_finish_time = np.full(MUX_MAX_TASKS, np.nan, dtype=np.float64)
        self.task_energy = np.zeros(MUX_MAX_TASKS, dtype=np.float64)
        self.task_qos = np.zeros(MUX_MAX_TASKS, dtype=np.float64)
        self.task_resource_slot = np.full(MUX_MAX_TASKS, -1, dtype=np.int64)
        self.task_core_type: List[Optional[str]] = [None] * MUX_MAX_TASKS

        # Runtime progress model for preemption/resume.
        self.task_remaining_fraction = np.ones(MUX_MAX_TASKS, dtype=np.float64)
        self.task_compute_start_time = np.full(
            MUX_MAX_TASKS, np.nan, dtype=np.float64
        )
        self.task_full_exec_time = np.full(MUX_MAX_TASKS, np.nan, dtype=np.float64)
        self.task_power_rate = np.zeros(MUX_MAX_TASKS, dtype=np.float64)
        self.task_generation = np.zeros(MUX_MAX_TASKS, dtype=np.int64)
        self.task_migration_count = np.zeros(MUX_MAX_TASKS, dtype=np.int64)
        self.task_last_migration_time = np.full(
            MUX_MAX_TASKS, -np.inf, dtype=np.float64
        )
        self.task_migration_energy = np.zeros(MUX_MAX_TASKS, dtype=np.float64)

        self.total_energy = 0.0
        self.total_migration_energy = 0.0
        self.total_migration_overhead = 0.0
        self.max_global_gang_energy = 1.0
        self.deadline_misses = 0
        self.steps = 0
        self.migration_history: List[Dict[str, object]] = []

        self.current_graph = None
        self.current_observation: Dict[str, np.ndarray]

    # ========================================================
    # Gym reset
    # ========================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.scenario_name, self.scenario_seed = self.seed_pool.sample(
            self.np_random
        )
        self.state = initialize_scenario_state(
            seed=self.scenario_seed,
            workflow_id=self.workflow_id,
        )
        # Reward normalization follows the legacy Dag_Env definition, adapted
        # to the concrete resources available in this dynamic scenario.
        self.max_global_gang_energy = self._compute_max_global_gang_energy()

        self._event_queue = []
        self._event_sequence = 0
        self._decision_epoch = 0
        self._next_migration_check_time = None
        self._migration_pending_tasks = set()
        self.task_next_migration_check_time.fill(np.inf)

        self.task_start_time.fill(np.nan)
        self.task_finish_time.fill(np.nan)
        self.task_energy.fill(0.0)
        self.task_qos.fill(0.0)
        self.task_resource_slot.fill(-1)
        self.task_core_type = [None] * MUX_MAX_TASKS
        self.task_remaining_fraction.fill(1.0)
        self.task_compute_start_time.fill(np.nan)
        self.task_full_exec_time.fill(np.nan)
        self.task_power_rate.fill(0.0)
        self.task_generation.fill(0)
        self.task_migration_count.fill(0)
        self.task_last_migration_time.fill(-np.inf)
        self.task_migration_energy.fill(0.0)

        self.total_energy = 0.0
        self.total_migration_energy = 0.0
        self.total_migration_overhead = 0.0
        self.deadline_misses = 0
        self.steps = 0
        self.migration_history = []

        released_tasks = tuple(
            int(task_id)
            for task_id in np.flatnonzero(self.state.task_ready > 0.5)
        )
        self.current_observation = self._rebuild_observation(
            trigger_event="initial_task_release",
            affected_tasks=released_tasks,
            released_tasks=released_tasks,
        )

        if not np.any(self.current_observation["action_mask"]):
            raise RuntimeError("Initial graph contains no valid PPO action.")

        info = {
            "scenario": self.scenario_name,
            "scenario_seed": self.scenario_seed,
            "split": self.split,
            "tasks": self.num_tasks,
            "vehicles": len(self.state.vehicles),
            "edge_servers": len(self.state.edge_servers),
            "simulation_time": float(self.state.simulation_time),
            "reward_proposal": self.reward_proposal,
            "max_global_gang_energy": float(self.max_global_gang_energy),
        }
        return self.current_observation, info

    # ========================================================
    # Gym step
    # ========================================================

    def step(self, action: int):
        action = int(action)

        if not self._is_action_valid(action):
            return (
                self.current_observation,
                -self.invalid_action_penalty,
                False,
                False,
                {
                    "invalid_action": True,
                    "action": action,
                    "scenario": self.scenario_name,
                },
            )

        action_time = float(self.state.simulation_time)
        action_type, task_id, resource_slot, core_type = self._decode_action(action)
        action_info: Dict[str, object]

        if action_type == "schedule":
            choice = self._execution_choice(task_id, resource_slot, core_type)
            self._start_task(choice)
            reward = self._reward_for_choice(choice)
            action_info = {
                "action_type": "schedule",
                "task_id": task_id,
                "resource_slot": resource_slot,
                "resource_type": choice.resource_type,
                "resource_id": choice.resource_id,
                "execution_mode": choice.execution_mode,
                "core_type": choice.core_type,
                "required_cores": choice.cores,
                "exec_time": choice.exec_time,
                "queue_delay": choice.queue_delay,
                "duration": choice.duration,
                "energy": choice.energy,
                "estimated_finish_time": choice.finish_time,
            }
            trigger_event = "task_started"
            started_task: Optional[int] = task_id

        else:
            migration = self._migration_choice(
                task_id,
                resource_slot,
                core_type,
                require_pending=True,
            )
            if migration is None:
                raise RuntimeError(
                    "Migration action passed the mask but no valid migration plan exists."
                )

            reward = self._reward_for_migration(migration)
            if migration.is_keep:
                self._migration_pending_tasks.discard(task_id)
                trigger_event = "migration_keep"
            else:
                self._apply_migration(migration)
                trigger_event = "task_migrated"

            started_task = None
            action_info = {
                "action_type": "migration_keep" if migration.is_keep else "migration",
                "task_id": task_id,
                "source_resource_slot": migration.source_slot,
                "target_resource_slot": migration.target_slot,
                "source_core_type": migration.source_core_type,
                "target_core_type": migration.target_core_type,
                "target_resource_type": migration.target_resource_type,
                "target_resource_id": migration.target_resource_id,
                "required_cores": migration.cores,
                "progress": migration.progress,
                "remaining_fraction": migration.remaining_fraction,
                "state_size_mb": migration.state_size_mb,
                "link_rate_mbps": migration.link_rate_mbps,
                "contact_duration": migration.contact_duration,
                "transfer_time": migration.transfer_time,
                "migration_energy": migration.migration_energy,
                "remaining_exec_time": migration.remaining_exec_time,
                "duration": migration.duration,
                "estimated_finish_time": migration.finish_time,
            }

        self.steps += 1

        completed_tasks: List[int] = []
        released_tasks: List[int] = []

        self._decision_epoch += 1
        observation = self._rebuild_observation(
            trigger_event=trigger_event,
            affected_tasks=(task_id,),
            started_task=started_task,
        )
        terminated = self._all_tasks_completed()

        # Advance to a completion or migration-check event only when PPO has no
        # immediate scheduling/migration action left at the current timestamp.
        while not terminated and not np.any(observation["action_mask"]):
            just_completed, just_released, runtime_trigger = (
                self._advance_to_next_runtime_event()
            )
            completed_tasks.extend(just_completed)
            released_tasks.extend(just_released)

            affected = tuple(sorted(set(just_completed) | set(just_released)))
            self._decision_epoch += 1
            observation = self._rebuild_observation(
                trigger_event=runtime_trigger,
                affected_tasks=affected,
                completed_tasks=just_completed,
                released_tasks=just_released,
            )
            terminated = self._all_tasks_completed()

        self.current_observation = observation

        info = {
            "invalid_action": False,
            "scenario": self.scenario_name,
            "scenario_seed": self.scenario_seed,
            "action_time": action_time,
            "simulation_time": float(self.state.simulation_time),
            "completed_tasks_since_action": completed_tasks,
            "released_tasks_since_action": released_tasks,
            **action_info,
            **self.get_metrics(),
        }
        return observation, reward, terminated, False, info

    # ========================================================
    # Graph -> online observation
    # ========================================================

    def _rebuild_observation(
        self,
        trigger_event: str,
        affected_tasks: Sequence[int] = (),
        started_task: Optional[int] = None,
        completed_tasks: Sequence[int] = (),
        released_tasks: Sequence[int] = (),
    ) -> Dict[str, np.ndarray]:
        graph, _ = build_dynamic_hetero_graph(
            state=self.state,
            workflow_id=self.workflow_id,
            decision_epoch_index=self._decision_epoch,
            event_context={
                "trigger_event": trigger_event,
                "affected_tasks": list(affected_tasks),
                "started_task": started_task,
                "completed_tasks": list(completed_tasks),
                "released_tasks": list(released_tasks),
                "migration_pending_tasks": sorted(self._migration_pending_tasks),
            },
        )

        self.current_graph = graph
        base_observation = self.adapter.to_observation(graph)
        schedule_mask = np.asarray(
            base_observation.pop("action_mask"), dtype=np.int8
        )

        migration_features, migration_mask, migration_task_mask = (
            self._build_migration_tensors()
        )
        action_mask = np.concatenate([schedule_mask, migration_mask]).astype(
            np.int8
        )

        observation = {
            **base_observation,
            "migration_features": migration_features,
            "migration_candidate_mask": migration_mask.reshape(
                MUX_MAX_TASKS,
                MUX_MAX_RESOURCES,
                MUX_NUM_PROCESSOR_TYPES,
            ),
            "migration_task_mask": migration_task_mask,
            "action_mask": action_mask,
        }

        if not self.observation_space.contains(observation):
            raise AssertionError(
                "Online migration observation is outside the fixed Gym space."
            )
        return observation

    # ========================================================
    # Action semantics
    # ========================================================

    @property
    def schedule_action_count(self) -> int:
        return MUX_SCHEDULE_ACTIONS

    @property
    def migration_action_offset(self) -> int:
        return MUX_SCHEDULE_ACTIONS

    def _encode_action(
        self,
        action_type: str,
        task_id: int,
        resource_slot: int,
        core_type: str,
    ) -> int:
        if core_type not in MUX_PROCESSOR_TYPES:
            raise ValueError(f"Unknown processor type: {core_type}")
        processor_index = MUX_PROCESSOR_TYPES.index(core_type)
        local = (
            (int(task_id) * MUX_MAX_RESOURCES + int(resource_slot))
            * MUX_NUM_PROCESSOR_TYPES
            + processor_index
        )
        if action_type == "schedule":
            return int(local)
        if action_type == "migration":
            return int(MUX_SCHEDULE_ACTIONS + local)
        raise ValueError(f"Unknown action type: {action_type}")

    def _decode_action(self, action: int) -> Tuple[str, int, int, str]:
        if not 0 <= action < MUX_TOTAL_ACTIONS:
            raise ValueError(f"Action out of range: {action}")

        if action < MUX_SCHEDULE_ACTIONS:
            action_type = "schedule"
            local = action
        else:
            action_type = "migration"
            local = action - MUX_SCHEDULE_ACTIONS

        pair_index = local // MUX_NUM_PROCESSOR_TYPES
        processor_index = local % MUX_NUM_PROCESSOR_TYPES
        task_id = pair_index // MUX_MAX_RESOURCES
        resource_slot = pair_index % MUX_MAX_RESOURCES
        core_type = MUX_PROCESSOR_TYPES[processor_index]
        return action_type, int(task_id), int(resource_slot), str(core_type)

    def _is_action_valid(self, action: int) -> bool:
        if action < 0 or action >= self.action_space.n:
            return False
        return bool(self.current_observation["action_mask"][action] == 1)

    def migration_keep_action(self, task_id: int) -> int:
        """Return the explicit KEEP action for one pending running task."""
        source = int(self.task_resource_slot[task_id])
        core_type = self.task_core_type[task_id]
        if source < 0 or core_type is None:
            raise ValueError(f"Task {task_id} is not running on a resource.")
        return self._encode_action("migration", task_id, source, core_type)

    def _execution_choice(
        self,
        task_id: int,
        resource_slot: int,
        core_type: str,
    ) -> _ExecutionChoice:
        """Build the exact initial execution selected by PPO."""

        if core_type not in MUX_PROCESSOR_TYPES:
            raise ValueError(f"Unknown processor type: {core_type}")

        profile = self.profiles[task_id]
        gang_size = int(profile["gang_size"])
        source_vehicle = int(
            self.current_graph["task"].source_vehicle[task_id].item()
        )
        resource_type, resource_id, resource = self._resource_from_slot(
            resource_slot
        )

        if resource_type == "vehicle" and resource_id == source_vehicle:
            execution_mode = "local"
        elif resource_type == "vehicle":
            execution_mode = "v2v"
        else:
            execution_mode = "v2i"

        available = (
            int(resource.available_a7)
            if core_type == "a7"
            else int(resource.available_a12)
        )
        if available < gang_size:
            raise RuntimeError(
                f"PPO selected {core_type.upper()} but only {available} cores "
                f"are available for gang_size={gang_size}."
            )

        base_time = float(profile[f"{core_type}_time"])
        base_power = float(profile[f"{core_type}_power"])
        exec_time = base_time / max(
            gang_size * float(resource.compute_scale),
            1e-8,
        )
        queue_delay = float(resource.queue_delay)
        duration = queue_delay + exec_time
        energy = (
            exec_time
            * gang_size
            * base_power
            * float(resource.power_scale)
        )
        finish_time = float(self.state.simulation_time) + duration

        return _ExecutionChoice(
            task_id=task_id,
            resource_slot=resource_slot,
            resource_type=resource_type,
            resource_id=resource_id,
            execution_mode=execution_mode,
            core_type=core_type,
            cores=gang_size,
            exec_time=float(exec_time),
            queue_delay=queue_delay,
            duration=float(duration),
            energy=float(energy),
            finish_time=float(finish_time),
        )

    # ========================================================
    # Runtime migration planning
    # ========================================================

    def _current_remaining_fraction(
        self,
        task_id: int,
        at_time: Optional[float] = None,
    ) -> float:
        if self.state.task_completed[task_id] > 0.5:
            return 0.0

        remaining_at_plan = float(self.task_remaining_fraction[task_id])
        compute_start = float(self.task_compute_start_time[task_id])
        full_exec_time = float(self.task_full_exec_time[task_id])

        if not np.isfinite(compute_start) or not np.isfinite(full_exec_time):
            return remaining_at_plan

        now = float(self.state.simulation_time if at_time is None else at_time)
        if now <= compute_start:
            return remaining_at_plan

        elapsed_compute = max(now - compute_start, 0.0)
        consumed_fraction = elapsed_compute / max(full_exec_time, 1e-8)
        return float(np.clip(remaining_at_plan - consumed_fraction, 0.0, 1.0))

    def _migration_choice(
        self,
        task_id: int,
        target_slot: int,
        target_core_type: str,
        require_pending: bool = False,
    ) -> Optional[_MigrationChoice]:
        if not 0 <= task_id < self.num_tasks:
            return None
        if self.state.task_running[task_id] <= 0.5:
            return None
        if require_pending and task_id not in self._migration_pending_tasks:
            return None
        if target_core_type not in MUX_PROCESSOR_TYPES:
            return None

        source_slot = int(self.task_resource_slot[task_id])
        source_core_type = self.task_core_type[task_id]
        if source_slot < 0 or source_core_type is None:
            return None

        now = float(self.state.simulation_time)
        remaining_fraction = self._current_remaining_fraction(task_id, now)
        progress = float(np.clip(1.0 - remaining_fraction, 0.0, 1.0))
        profile = self.profiles[task_id]
        gang_size = int(profile["gang_size"])
        state_size_mb = float(self.task_state_size_mb[task_id])

        old_finish = float(self.task_finish_time[task_id])
        old_duration = max(old_finish - now, 0.0)
        old_compute_start = max(float(self.task_compute_start_time[task_id]), now)
        old_compute_seconds = max(old_finish - old_compute_start, 0.0)
        old_remaining_energy = old_compute_seconds * float(
            self.task_power_rate[task_id]
        )

        # Same resource + same processor is the explicit no-migration action.
        if target_slot == source_slot and target_core_type == source_core_type:
            resource_type, resource_id, resource = self._resource_from_slot(
                source_slot
            )
            return _MigrationChoice(
                task_id=task_id,
                source_slot=source_slot,
                target_slot=source_slot,
                source_core_type=source_core_type,
                target_core_type=source_core_type,
                target_resource_type=resource_type,
                target_resource_id=resource_id,
                cores=gang_size,
                is_keep=True,
                progress=progress,
                remaining_fraction=remaining_fraction,
                state_size_mb=state_size_mb,
                link_rate_mbps=0.0,
                contact_duration=0.0,
                preempt_time=0.0,
                transfer_time=0.0,
                resume_time=0.0,
                queue_delay=0.0,
                remaining_exec_time=old_compute_seconds,
                remaining_compute_energy=old_remaining_energy,
                migration_energy=0.0,
                duration=old_duration,
                finish_time=old_finish,
                old_duration=old_duration,
                old_remaining_energy=old_remaining_energy,
                old_finish_time=old_finish,
            )

        # In-place A7<->A12 switching is not called migration here. A runtime
        # migration must change the physical resource.
        if target_slot == source_slot:
            return None

        if int(self.task_migration_count[task_id]) >= MUX_MIGRATION_MAX_PER_TASK:
            return None
        if remaining_fraction < MUX_MIGRATION_MIN_REMAINING_FRACTION:
            return None
        compute_start = float(self.task_compute_start_time[task_id])
        if (
            not np.isfinite(compute_start)
            or now - compute_start < MUX_MIGRATION_MIN_RUNTIME_SEC
        ):
            return None
        if (
            now - float(self.task_last_migration_time[task_id])
            < MUX_MIGRATION_COOLDOWN_SEC
        ):
            return None

        target_type, target_id, target = self._resource_from_slot(target_slot)
        available = (
            int(target.available_a7)
            if target_core_type == "a7"
            else int(target.available_a12)
        )
        if available < gang_size:
            return None

        reachable, rate_mbps, contact_duration = self._migration_link(
            source_slot,
            target_slot,
        )
        if not reachable:
            return None

        preempt_time = (
            MUX_MIGRATION_PREEMPT_SEC
            + MUX_MIGRATION_SERIALIZE_SEC_PER_MB * state_size_mb
        )
        transfer_time = state_size_mb * 8.0 / max(rate_mbps, 1e-8)
        resume_time = MUX_MIGRATION_RESUME_SEC
        link_needed = (preempt_time + transfer_time + resume_time)
        if contact_duration + _EPS < (
            MUX_MIGRATION_CONTACT_SAFETY_FACTOR * link_needed
        ):
            return None

        full_exec_time = float(profile[f"{target_core_type}_time"]) / max(
            gang_size * float(target.compute_scale),
            1e-8,
        )
        remaining_exec_time = remaining_fraction * full_exec_time
        queue_delay = float(target.queue_delay)
        duration = (
            preempt_time
            + transfer_time
            + resume_time
            + queue_delay
            + remaining_exec_time
        )
        finish_time = now + duration

        base_power = float(profile[f"{target_core_type}_power"])
        target_power_rate = (
            gang_size * base_power * float(target.power_scale)
        )
        remaining_compute_energy = remaining_exec_time * target_power_rate
        migration_energy = (
            state_size_mb * MUX_MIGRATION_TX_ENERGY_J_PER_MB
            + (preempt_time + resume_time) * MUX_MIGRATION_CONTROL_POWER_W
        )

        return _MigrationChoice(
            task_id=task_id,
            source_slot=source_slot,
            target_slot=target_slot,
            source_core_type=source_core_type,
            target_core_type=target_core_type,
            target_resource_type=target_type,
            target_resource_id=target_id,
            cores=gang_size,
            is_keep=False,
            progress=progress,
            remaining_fraction=remaining_fraction,
            state_size_mb=state_size_mb,
            link_rate_mbps=float(rate_mbps),
            contact_duration=float(contact_duration),
            preempt_time=float(preempt_time),
            transfer_time=float(transfer_time),
            resume_time=float(resume_time),
            queue_delay=queue_delay,
            remaining_exec_time=float(remaining_exec_time),
            remaining_compute_energy=float(remaining_compute_energy),
            migration_energy=float(migration_energy),
            duration=float(duration),
            finish_time=float(finish_time),
            old_duration=float(old_duration),
            old_remaining_energy=float(old_remaining_energy),
            old_finish_time=float(old_finish),
        )

    def _migration_link(
        self,
        source_slot: int,
        target_slot: int,
    ) -> Tuple[bool, float, float]:
        source_type, _, source = self._resource_from_slot(source_slot)
        target_type, _, target = self._resource_from_slot(target_slot)

        if source_type == "edge_server" and target_type == "edge_server":
            return True, float(BASE_V2I_RATE_MBPS), float(CONTACT_DURATION_NORM_SEC)

        source_x = float(source.x)
        source_y = float(source.y)
        target_x = float(target.x)
        target_y = float(target.y)
        distance = float(math.hypot(source_x - target_x, source_y - target_y))

        if source_type == "vehicle" and target_type == "vehicle":
            range_m = float(V2V_RANGE_M)
            base_rate = float(BASE_V2V_RATE_MBPS)
            source_velocity = float(source.speed) * float(source.direction)
            target_velocity = float(target.speed) * float(target.direction)
        else:
            range_m = float(V2I_RANGE_M)
            base_rate = float(BASE_V2I_RATE_MBPS)
            if source_type == "vehicle":
                source_velocity = float(source.speed) * float(source.direction)
                target_velocity = 0.0
            else:
                source_velocity = 0.0
                target_velocity = float(target.speed) * float(target.direction)

        if distance > range_m:
            return False, 0.0, 0.0

        ratio = distance / max(range_m, 1e-8)
        rate_mbps = base_rate / (1.0 + ratio * ratio)

        remaining_distance = max(range_m - distance, 0.0)
        relative_speed = abs(source_velocity - target_velocity)
        if relative_speed < 0.25:
            contact_duration = float(CONTACT_DURATION_NORM_SEC)
        else:
            contact_duration = remaining_distance / relative_speed

        return True, float(rate_mbps), float(contact_duration)

    def _build_migration_tensors(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.zeros(
            (
                MUX_MAX_TASKS,
                MUX_MAX_RESOURCES,
                MUX_NUM_PROCESSOR_TYPES,
                MUX_MIGRATION_FEATURE_DIM,
            ),
            dtype=np.float32,
        )
        mask = np.zeros(MUX_MIGRATION_ACTIONS, dtype=np.int8)
        task_mask = np.zeros(MUX_MAX_TASKS, dtype=np.int8)

        for task_id in sorted(self._migration_pending_tasks):
            if self.state.task_running[task_id] <= 0.5:
                continue

            task_has_action = False
            for resource_slot in range(MUX_MAX_RESOURCES):
                try:
                    self._resource_from_slot(resource_slot)
                except IndexError:
                    continue

                for processor_index, core_type in enumerate(MUX_PROCESSOR_TYPES):
                    choice = self._migration_choice(
                        task_id,
                        resource_slot,
                        core_type,
                        require_pending=False,
                    )
                    if choice is None:
                        continue

                    features[
                        task_id,
                        resource_slot,
                        processor_index,
                    ] = self._migration_feature_vector(choice)

                    local = (
                        (task_id * MUX_MAX_RESOURCES + resource_slot)
                        * MUX_NUM_PROCESSOR_TYPES
                        + processor_index
                    )
                    mask[local] = 1
                    task_has_action = True

            if task_has_action:
                task_mask[task_id] = 1

        return features, mask, task_mask

    def _migration_feature_vector(
        self,
        choice: _MigrationChoice,
    ) -> np.ndarray:
        task_id = choice.task_id
        profile = self.profiles[task_id]
        deadline = max(float(profile["deadline"]), 1e-8)
        base_energy = self._base_energy(task_id)
        _, _, target = self._resource_from_slot(choice.target_slot)

        if choice.target_core_type == "a7":
            available = int(target.available_a7)
            capacity = int(target.capacity_a7)
        else:
            available = int(target.available_a12)
            capacity = int(target.capacity_a12)

        if choice.is_keep:
            # The task's own reserved cores belong to the current allocation.
            available += choice.cores

        values = np.asarray(
            [
                choice.progress,
                choice.remaining_fraction,
                float(self.task_migration_count[task_id])
                / max(MUX_MIGRATION_MAX_PER_TASK, 1),
                min(choice.state_size_mb / MUX_MIGRATION_STATE_NORM_MB, 2.0),
                choice.source_slot / max(MUX_MAX_RESOURCES - 1, 1),
                float(choice.source_core_type == "a12"),
                choice.target_slot / max(MUX_MAX_RESOURCES - 1, 1),
                float(choice.target_core_type == "a12"),
                float(choice.is_keep),
                float(np.clip(available / max(capacity, 1), 0.0, 1.0)),
                min(choice.queue_delay / deadline, 2.0),
                float(target.compute_scale),
                float(target.power_scale),
                1.0,
                min(
                    choice.link_rate_mbps
                    / max(BASE_V2I_RATE_MBPS, BASE_V2V_RATE_MBPS, 1e-8),
                    1.0,
                ),
                min(
                    choice.transfer_time / MUX_MIGRATION_TRANSFER_DELAY_NORM_SEC,
                    2.0,
                ),
                min(
                    choice.contact_duration / max(CONTACT_DURATION_NORM_SEC, 1e-8),
                    2.0,
                ),
                min(choice.remaining_exec_time / deadline, 2.0),
                min(choice.finish_time / deadline, 2.0),
                min(choice.migration_energy / base_energy, 2.0),
            ],
            dtype=np.float32,
        )

        if len(values) != len(MIGRATION_FEATURE_NAMES):
            raise RuntimeError("Migration feature-vector contract mismatch.")
        return values

    def _task_has_real_migration(self, task_id: int) -> bool:
        source_slot = int(self.task_resource_slot[task_id])
        source_core = self.task_core_type[task_id]
        if source_slot < 0 or source_core is None:
            return False

        for target_slot in range(MUX_MAX_RESOURCES):
            if target_slot == source_slot:
                continue
            try:
                self._resource_from_slot(target_slot)
            except IndexError:
                continue
            for core_type in MUX_PROCESSOR_TYPES:
                if self._migration_choice(
                    task_id,
                    target_slot,
                    core_type,
                    require_pending=False,
                ) is not None:
                    return True
        return False

    def _open_migration_window(
        self,
        task_ids: Optional[Sequence[int]] = None,
    ) -> None:
        # Periodic checkpoints only reconsider the tasks whose adaptive timers
        # fired. Completion events may still reconsider all running tasks.
        candidates = (
            range(self.num_tasks)
            if task_ids is None
            else (int(task_id) for task_id in task_ids)
        )
        for task_id in candidates:
            if not 0 <= task_id < self.num_tasks:
                continue
            if self.state.task_running[task_id] <= 0.5:
                continue
            if self._task_has_real_migration(task_id):
                self._migration_pending_tasks.add(task_id)

    # ========================================================
    # Event-driven execution and progress accounting
    # ========================================================

    def _reserve_cores(
        self,
        resource_slot: int,
        core_type: str,
        cores: int,
    ) -> None:
        _, _, resource = self._resource_from_slot(resource_slot)
        if core_type == "a7":
            if resource.available_a7 < cores:
                raise RuntimeError("Insufficient A7 cores.")
            resource.available_a7 -= cores
        else:
            if resource.available_a12 < cores:
                raise RuntimeError("Insufficient A12 cores.")
            resource.available_a12 -= cores

    def _release_cores(
        self,
        resource_slot: int,
        core_type: str,
        cores: int,
    ) -> None:
        _, _, resource = self._resource_from_slot(resource_slot)
        if core_type == "a7":
            resource.available_a7 = min(
                int(resource.capacity_a7),
                int(resource.available_a7) + int(cores),
            )
        else:
            resource.available_a12 = min(
                int(resource.capacity_a12),
                int(resource.available_a12) + int(cores),
            )

    def _start_task(self, choice: _ExecutionChoice) -> None:
        task_id = choice.task_id
        if self.state.task_ready[task_id] <= 0.5:
            raise RuntimeError(f"Task {task_id} is not ready.")

        self._reserve_cores(choice.resource_slot, choice.core_type, choice.cores)
        self.state.task_ready[task_id] = 0.0
        self.state.task_running[task_id] = 1.0
        refresh_resource_queue_delays(self.state)

        now = float(self.state.simulation_time)
        self.task_start_time[task_id] = now
        self.task_energy[task_id] = 0.0
        self.task_resource_slot[task_id] = choice.resource_slot
        self.task_core_type[task_id] = choice.core_type
        self.task_remaining_fraction[task_id] = 1.0
        self.task_compute_start_time[task_id] = now + choice.queue_delay
        self.task_full_exec_time[task_id] = choice.exec_time

        _, _, resource = self._resource_from_slot(choice.resource_slot)
        self.task_power_rate[task_id] = (
            choice.cores
            * float(self.profiles[task_id][f"{choice.core_type}_power"])
            * float(resource.power_scale)
        )

        self.task_generation[task_id] += 1
        self._update_finish_prediction(task_id, choice.finish_time)
        self._push_completion(task_id)
        self._schedule_next_migration_check_for_task(task_id, now)
        self._ensure_next_migration_check()

    def _apply_migration(self, choice: _MigrationChoice) -> None:
        task_id = choice.task_id
        if choice.is_keep:
            raise ValueError("_apply_migration requires a real migration choice.")

        source_slot = int(self.task_resource_slot[task_id])
        source_core = self.task_core_type[task_id]
        if source_slot != choice.source_slot or source_core != choice.source_core_type:
            raise RuntimeError("Task source changed before migration was applied.")

        self._release_cores(source_slot, choice.source_core_type, choice.cores)
        self._reserve_cores(choice.target_slot, choice.target_core_type, choice.cores)
        refresh_resource_queue_delays(self.state)

        now = float(self.state.simulation_time)
        self.task_remaining_fraction[task_id] = choice.remaining_fraction
        self.task_resource_slot[task_id] = choice.target_slot
        self.task_core_type[task_id] = choice.target_core_type
        self.task_compute_start_time[task_id] = (
            now
            + choice.preempt_time
            + choice.transfer_time
            + choice.resume_time
            + choice.queue_delay
        )

        if choice.remaining_fraction > _EPS:
            self.task_full_exec_time[task_id] = (
                choice.remaining_exec_time / choice.remaining_fraction
            )
        else:
            self.task_full_exec_time[task_id] = 0.0

        _, _, target = self._resource_from_slot(choice.target_slot)
        self.task_power_rate[task_id] = (
            choice.cores
            * float(self.profiles[task_id][f"{choice.target_core_type}_power"])
            * float(target.power_scale)
        )

        self.task_migration_count[task_id] += 1
        self.task_last_migration_time[task_id] = now
        self.task_migration_energy[task_id] += choice.migration_energy
        self.total_migration_energy += choice.migration_energy
        self.total_migration_overhead += (
            choice.preempt_time
            + choice.transfer_time
            + choice.resume_time
        )
        self.task_energy[task_id] += choice.migration_energy
        self.total_energy += choice.migration_energy

        self.task_generation[task_id] += 1
        self._update_finish_prediction(task_id, choice.finish_time)
        self._push_completion(task_id)
        self._migration_pending_tasks.discard(task_id)
        self._schedule_next_migration_check_for_task(task_id, now)

        self.migration_history.append(
            {
                "task_id": int(task_id),
                "time": now,
                "source_slot": int(choice.source_slot),
                "target_slot": int(choice.target_slot),
                "source_core_type": str(choice.source_core_type),
                "target_core_type": str(choice.target_core_type),
                "progress": float(choice.progress),
                "state_size_mb": float(choice.state_size_mb),
                "transfer_time": float(choice.transfer_time),
                "migration_energy": float(choice.migration_energy),
                "new_finish_time": float(choice.finish_time),
            }
        )
        self._ensure_next_migration_check()

    def _push_completion(self, task_id: int) -> None:
        resource_slot = int(self.task_resource_slot[task_id])
        core_type = self.task_core_type[task_id]
        if resource_slot < 0 or core_type is None:
            raise RuntimeError("Cannot create completion event without allocation.")

        resource_type, resource_id, _ = self._resource_from_slot(resource_slot)
        heapq.heappush(
            self._event_queue,
            _CompletionEvent(
                time=float(self.task_finish_time[task_id]),
                sequence=self._next_event_sequence(),
                task_id=task_id,
                resource_type=resource_type,
                resource_id=resource_id,
                core_type=core_type,
                cores=int(self.profiles[task_id]["gang_size"]),
                generation=int(self.task_generation[task_id]),
            ),
        )

    def _event_is_current(self, event: _CompletionEvent) -> bool:
        return bool(
            self.state.task_running[event.task_id] > 0.5
            and int(event.generation) == int(self.task_generation[event.task_id])
            and abs(float(event.time) - float(self.task_finish_time[event.task_id]))
            <= 1e-6
        )

    def _discard_stale_events(self) -> None:
        while self._event_queue and not self._event_is_current(self._event_queue[0]):
            heapq.heappop(self._event_queue)

    def _next_completion_time(self) -> Optional[float]:
        self._discard_stale_events()
        if not self._event_queue:
            return None
        return float(self._event_queue[0].time)

    def _accrue_running_energy_until(self, target_time: float) -> None:
        start_time = float(self.state.simulation_time)
        if target_time <= start_time + _EPS:
            return

        for task_id in range(self.num_tasks):
            if self.state.task_running[task_id] <= 0.5:
                continue

            compute_start = float(self.task_compute_start_time[task_id])
            finish = float(self.task_finish_time[task_id])
            if not np.isfinite(compute_start) or not np.isfinite(finish):
                continue

            active_start = max(start_time, compute_start)
            active_end = min(target_time, finish)
            if active_end <= active_start:
                continue

            energy = (
                (active_end - active_start)
                * float(self.task_power_rate[task_id])
            )
            self.task_energy[task_id] += energy
            self.total_energy += energy

    def _advance_time_to(self, target_time: float) -> None:
        now = float(self.state.simulation_time)
        delta = float(target_time) - now
        if delta < -1e-8:
            raise RuntimeError("Runtime event queue moved backward in time.")
        if delta <= 0.0:
            return

        self._accrue_running_energy_until(float(target_time))
        advance_scenario_time(self.state, delta)
        self.state.simulation_time = float(target_time)

    def _process_completions_at_current_time(self) -> Tuple[int, ...]:
        now = float(self.state.simulation_time)
        completed: List[int] = []
        self._discard_stale_events()

        while self._event_queue:
            event = self._event_queue[0]
            if float(event.time) > now + 1e-7:
                break
            event = heapq.heappop(self._event_queue)
            if not self._event_is_current(event):
                continue

            task_id = int(event.task_id)
            resource_slot = int(self.task_resource_slot[task_id])
            self._release_cores(resource_slot, event.core_type, event.cores)

            self.state.task_running[task_id] = 0.0
            self.state.task_completed[task_id] = 1.0
            self.task_remaining_fraction[task_id] = 0.0
            self.task_power_rate[task_id] = 0.0
            self.task_next_migration_check_time[task_id] = np.inf
            self._migration_pending_tasks.discard(task_id)
            completed.append(task_id)

        if completed:
            refresh_resource_queue_delays(self.state)
        return tuple(sorted(completed))

    def _schedule_next_migration_check_for_task(
        self,
        task_id: int,
        now: Optional[float] = None,
    ) -> None:
        """Schedule one duration-adaptive checkpoint for a running task."""
        if not 0 <= task_id < self.num_tasks:
            return
        if self.state.task_running[task_id] <= 0.5:
            self.task_next_migration_check_time[task_id] = np.inf
            return

        current_time = float(
            self.state.simulation_time if now is None else now
        )
        finish_time = float(self.task_finish_time[task_id])
        compute_start = float(self.task_compute_start_time[task_id])
        remaining_fraction = self._current_remaining_fraction(
            task_id, current_time
        )

        if (
            not np.isfinite(finish_time)
            or finish_time <= current_time + _EPS
            or remaining_fraction < MUX_MIGRATION_MIN_REMAINING_FRACTION
        ):
            self.task_next_migration_check_time[task_id] = np.inf
            return

        remaining_runtime = max(finish_time - current_time, 0.0)
        interval = max(
            MUX_MIGRATION_CHECK_MIN_INTERVAL_SEC,
            MUX_MIGRATION_CHECK_REMAINING_RATIO * remaining_runtime,
        )
        candidate_time = current_time + interval

        # Do not spend a migration decision before useful compute has run.
        if np.isfinite(compute_start):
            candidate_time = max(
                candidate_time,
                compute_start + MUX_MIGRATION_MIN_RUNTIME_SEC,
            )

        last_migration = float(self.task_last_migration_time[task_id])
        if np.isfinite(last_migration):
            candidate_time = max(
                candidate_time,
                last_migration + MUX_MIGRATION_COOLDOWN_SEC,
            )

        # If the checkpoint would occur at/after completion, no periodic check
        # is useful. Completion events can still open an immediate window.
        if candidate_time >= finish_time - _EPS:
            self.task_next_migration_check_time[task_id] = np.inf
        else:
            self.task_next_migration_check_time[task_id] = float(candidate_time)

    def _ensure_next_migration_check(self) -> None:
        running = self.state.task_running[: self.num_tasks] > 0.5
        if not np.any(running):
            self._next_migration_check_time = None
            return

        timers = self.task_next_migration_check_time[: self.num_tasks]
        finite = running & np.isfinite(timers)
        if not np.any(finite):
            self._next_migration_check_time = None
            return

        self._next_migration_check_time = float(np.min(timers[finite]))

    def _advance_migration_clock_after_check(
        self,
        now: float,
    ) -> Tuple[int, ...]:
        due_tasks: List[int] = []
        for task_id in range(self.num_tasks):
            if self.state.task_running[task_id] <= 0.5:
                continue
            check_time = float(self.task_next_migration_check_time[task_id])
            if np.isfinite(check_time) and check_time <= now + _EPS:
                due_tasks.append(task_id)
                self._schedule_next_migration_check_for_task(task_id, now)

        self._ensure_next_migration_check()
        return tuple(sorted(due_tasks))

    def _advance_to_next_runtime_event(
        self,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], str]:
        self._ensure_next_migration_check()
        completion_time = self._next_completion_time()
        migration_check_time = self._next_migration_check_time

        candidates = [
            value
            for value in (completion_time, migration_check_time)
            if value is not None
        ]
        if not candidates:
            raise RuntimeError(
                "No future completion or migration-check event exists before termination."
            )

        next_time = min(candidates)
        self._advance_time_to(next_time)

        periodic_check = bool(
            migration_check_time is not None
            and migration_check_time <= next_time + _EPS
        )
        due_migration_tasks: Tuple[int, ...] = ()
        if periodic_check:
            due_migration_tasks = self._advance_migration_clock_after_check(
                next_time
            )

        completed = self._process_completions_at_current_time()
        released = self._release_newly_ready_tasks()

        # Adaptive periodic checks reconsider only due tasks. A completion can
        # change resource availability, so it may reconsider all running tasks.
        if periodic_check:
            self._open_migration_window(due_migration_tasks)
        if completed:
            self._open_migration_window()

        self._ensure_next_migration_check()

        if completed and released:
            trigger = "task_completion_release_migration_check"
        elif completed:
            trigger = "task_completion_migration_check"
        elif periodic_check:
            trigger = "runtime_migration_check"
        else:
            trigger = "runtime_event"

        return completed, released, trigger

    def _release_newly_ready_tasks(self) -> Tuple[int, ...]:
        released: List[int] = []
        for task_id in range(self.num_tasks):
            if self.state.task_released[task_id] > 0.5:
                continue

            predecessors = list(self.dag.predecessors(task_id))
            if not predecessors:
                continue

            if all(
                self.state.task_completed[pred] > 0.5
                for pred in predecessors
            ):
                self.state.task_released[task_id] = 1.0
                self.state.task_ready[task_id] = 1.0
                released.append(task_id)
        return tuple(sorted(released))

    # ========================================================
    # Resource mapping
    # ========================================================

    def _resource_from_slot(
        self,
        resource_slot: int,
    ) -> Tuple[str, int, VehicleState | EdgeServerState]:
        if 0 <= resource_slot < MUX_MAX_VEHICLES:
            resource_id = resource_slot
            if resource_id >= len(self.state.vehicles):
                raise IndexError("Vehicle slot is padding, not a real resource.")
            return "vehicle", resource_id, self.state.vehicles[resource_id]

        edge_slot = resource_slot - MUX_MAX_VEHICLES
        if 0 <= edge_slot < MUX_MAX_EDGE_SERVERS:
            if edge_slot >= len(self.state.edge_servers):
                raise IndexError("Edge-server slot is padding, not a real resource.")
            return "edge_server", edge_slot, self.state.edge_servers[edge_slot]

        raise IndexError(f"Invalid resource slot: {resource_slot}")

    # ========================================================
    # Reward / metrics
    # ========================================================

    def _base_energy(self, task_id: int) -> float:
        # Keep the existing migration-feature normalization unchanged.
        profile = self.profiles[task_id]
        return max(
            float(profile["a7_time"]) * float(profile["a7_power"]),
            float(profile["a12_time"]) * float(profile["a12_power"]),
            1e-8,
        )

    def _compute_max_global_gang_energy(self) -> float:
        """Return the largest feasible full-task energy in this scenario.

        This is the dynamic-resource counterpart of Dag_Env's
        max_global_gang_energy. Only the resource/processor combinations that
        can host the task gang are included.
        """

        energies: List[float] = []
        resources = [*self.state.vehicles, *self.state.edge_servers]

        for task_id in range(self.num_tasks):
            profile = self.profiles[task_id]
            gang_size = int(profile["gang_size"])

            for resource in resources:
                for core_type in MUX_PROCESSOR_TYPES:
                    capacity = (
                        int(resource.capacity_a7)
                        if core_type == "a7"
                        else int(resource.capacity_a12)
                    )
                    if capacity < gang_size:
                        continue

                    base_time = float(profile[f"{core_type}_time"])
                    base_power = float(profile[f"{core_type}_power"])
                    exec_time = base_time / max(
                        gang_size * float(resource.compute_scale),
                        1e-8,
                    )
                    energy = (
                        exec_time
                        * gang_size
                        * base_power
                        * float(resource.power_scale)
                    )
                    energies.append(float(energy))

        if not energies:
            raise RuntimeError(
                "Could not build max_global_gang_energy: no feasible "
                "task-resource-processor allocation exists."
            )

        return max(max(energies), 1e-6)

    def _qos_energy_reward(
        self,
        task_id: int,
        energy: float,
        finish_time: float,
    ) -> Tuple[float, float, float]:
        """Apply the same QoS x energy reward used by legacy Dag_Env."""

        deadline = float(self.profiles[task_id]["deadline"])
        qos = self._qos(float(finish_time), deadline)
        actual_energy = max(float(energy), 1e-8)

        if self.reward_proposal == "A":
            energy_term = self.max_global_gang_energy / actual_energy
        else:
            energy_term = np.exp(
                -actual_energy / self.max_global_gang_energy
            )

        base_reward = float(qos) * float(energy_term)
        reward = float(np.clip(base_reward, -10.0, 10.0))
        return reward, float(qos), float(energy_term)

    def _reward_for_choice(self, choice: _ExecutionChoice) -> float:
        # Scheduling receives the exact advisor-requested QoS-energy reward.
        reward, _, _ = self._qos_energy_reward(
            choice.task_id,
            choice.energy,
            choice.finish_time,
        )
        return reward

    def _reward_for_migration(self, choice: _MigrationChoice) -> float:
        """Use the same reward objective without double-counting migrations.

        The initial scheduling action already received the task-plan reward. A
        migration therefore receives only the change in that same QoS-energy
        reward. Across multiple migrations this telescopes to the reward of the
        final runtime plan. KEEP naturally receives zero.
        """

        task_id = choice.task_id
        accrued_energy = float(self.task_energy[task_id])

        old_total_energy = accrued_energy + float(choice.old_remaining_energy)
        new_total_energy = (
            accrued_energy
            + float(choice.migration_energy)
            + float(choice.remaining_compute_energy)
        )

        old_reward, _, _ = self._qos_energy_reward(
            task_id,
            old_total_energy,
            choice.old_finish_time,
        )
        new_reward, _, _ = self._qos_energy_reward(
            task_id,
            new_total_energy,
            choice.finish_time,
        )

        return float(np.clip(new_reward - old_reward, -10.0, 10.0))

    def _update_finish_prediction(self, task_id: int, finish_time: float) -> None:
        deadline = float(self.profiles[task_id]["deadline"])
        old_finish = float(self.task_finish_time[task_id])

        old_miss = bool(np.isfinite(old_finish) and old_finish > deadline)
        new_miss = bool(float(finish_time) > deadline)
        if old_miss and not new_miss:
            self.deadline_misses = max(self.deadline_misses - 1, 0)
        elif new_miss and not old_miss:
            self.deadline_misses += 1

        self.task_finish_time[task_id] = float(finish_time)
        self.task_qos[task_id] = self._qos(float(finish_time), deadline)

    def _qos(self, finish_time: float, deadline: float) -> float:
        if deadline <= 0.0:
            return 0.0
        if finish_time <= deadline:
            return 1.0

        upper = self.qos_factor * deadline
        if finish_time <= upper:
            return float(
                np.clip(
                    (upper - finish_time)
                    / ((self.qos_factor - 1.0) * deadline),
                    0.0,
                    1.0,
                )
            )
        return 0.0

    def get_metrics(self) -> Dict[str, float]:
        completed_mask = self.state.task_completed[: self.num_tasks] > 0.5
        completed_count = int(np.sum(completed_mask))

        finished = self.task_finish_time[: self.num_tasks]
        finite_finished = finished[np.isfinite(finished)]
        makespan = float(np.max(finite_finished)) if finite_finished.size else 0.0

        qos_values = self.task_qos[: self.num_tasks]
        active_qos = qos_values[np.isfinite(finished)]
        migrated_tasks = int(
            np.sum(self.task_migration_count[: self.num_tasks] > 0)
        )

        return {
            "completed_tasks": float(completed_count),
            "total_tasks": float(self.num_tasks),
            "total_energy": float(self.total_energy),
            "makespan": makespan,
            "deadline_miss_ratio": float(
                self.deadline_misses / max(self.num_tasks, 1)
            ),
            "mean_qos": float(np.mean(active_qos)) if active_qos.size else 0.0,
            "simulation_time": float(self.state.simulation_time),
            "total_migrations": float(
                np.sum(self.task_migration_count[: self.num_tasks])
            ),
            "migrated_task_ratio": float(
                migrated_tasks / max(self.num_tasks, 1)
            ),
            "migration_energy": float(self.total_migration_energy),
            "migration_overhead_time": float(self.total_migration_overhead),
        }

    def validate_schedule(self) -> None:
        if not self._all_tasks_completed():
            raise AssertionError("Not every task completed.")

        for task_id in range(self.num_tasks):
            start = float(self.task_start_time[task_id])
            finish = float(self.task_finish_time[task_id])
            if not np.isfinite(start) or not np.isfinite(finish):
                raise AssertionError(f"Task {task_id} has invalid timing.")
            if finish < start:
                raise AssertionError(f"Task {task_id} finishes before it starts.")

            for predecessor in self.dag.predecessors(task_id):
                pred_finish = float(self.task_finish_time[predecessor])
                if start + 1e-9 < pred_finish:
                    raise AssertionError(
                        f"Task {task_id} violates predecessor {predecessor}."
                    )

        for resource in [*self.state.vehicles, *self.state.edge_servers]:
            if not 0 <= int(resource.available_a7) <= int(resource.capacity_a7):
                raise AssertionError("A7 core accounting is inconsistent.")
            if not 0 <= int(resource.available_a12) <= int(resource.capacity_a12):
                raise AssertionError("A12 core accounting is inconsistent.")

    def render(self) -> None:
        metrics = self.get_metrics()
        print(
            f"scenario={self.scenario_name} "
            f"completed={int(metrics['completed_tasks'])}/{self.num_tasks} "
            f"t={metrics['simulation_time']:.4f}s "
            f"energy={metrics['total_energy']:.4f} "
            f"makespan={metrics['makespan']:.4f}s "
            f"QoS={metrics['mean_qos']:.4f} "
            f"migrations={int(metrics['total_migrations'])}"
        )

    # ========================================================
    # Utility
    # ========================================================

    def _all_tasks_completed(self) -> bool:
        return bool(
            np.all(self.state.task_completed[: self.num_tasks] > 0.5)
        )

    def _next_event_sequence(self) -> int:
        value = self._event_sequence
        self._event_sequence += 1
        return value
