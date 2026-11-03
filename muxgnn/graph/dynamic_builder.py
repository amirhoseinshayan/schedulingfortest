from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from torch_geometric.data import HeteroData

from .builder import (
    _edge_index,
    _link_rate,
    _max_normalize,
    _tensor_2d,
    load_dagbench_task_graph,
    load_project_task_profiles,
)

from .dynamic_settings import (
    CONTACT_DURATION_NORM_SEC,
    MAX_EDGE_QUEUE_DELAY_SEC,
    MAX_RESOURCE_LOAD,
    MAX_VEHICLE_QUEUE_DELAY_SEC,
    MAX_VEHICLE_SPEED_MPS,
    MIN_RESOURCE_LOAD,
    MIN_VEHICLE_SPEED_MPS,
    TX_DELAY_NORM_SEC,
)

from .magnet_relation_layers import (
    apply_magnet_relation_layers,
)

from .settings import (
    BASE_V2I_RATE_MBPS,
    BASE_V2V_RATE_MBPS,
    DAGBENCH_WORKFLOW_ID,
    EDGE_A12_CORES,
    EDGE_A7_CORES,
    MAX_GANG_SIZE,
    NUM_EDGE_SERVERS,
    NUM_VEHICLES,
    ROAD_LENGTH_M,
    SOURCE_VEHICLE_ID,
    V2I_RANGE_M,
    V2V_RANGE_M,
    VEHICLE_A12_CORES,
    VEHICLE_A7_CORES,
)


# ============================================================
# Dynamic feature schemas
# ============================================================

TASK_DYNAMIC_FEATURE_NAMES = [
    "dagbench_cost",
    "gang_size",
    "deadline",
    "is_released",
    "is_ready",
    "is_running",
    "is_completed",
]

RESOURCE_NODE_FEATURE_NAMES = [
    "total_a7_cores",
    "total_a12_cores",
    "compute_scale",
    "power_scale",
]

PRECEDENCE_FEATURE_NAMES = [
    "data_size",
]

CANDIDATE_DYNAMIC_FEATURE_NAMES = [
    "exec_time_a7",
    "exec_time_a12",
    "exec_energy_a7",
    "exec_energy_a12",
    "available_ratio_a7",
    "available_ratio_a12",
    "queue_delay",
    "estimated_finish_a7",
    "estimated_finish_a12",
    "feasible_a7",
    "feasible_a12",
    "is_local",
]

COMMUNICATION_FEATURE_NAMES = [
    "distance",
    "link_rate",
    "tx_delay_per_mb",
    "contact_duration",
    "reachable",
]


# ============================================================
# Scenario state
# ============================================================


@dataclass
class VehicleState:
    """Dynamic vehicle state."""

    vehicle_id: int

    x: float
    y: float

    speed: float
    direction: float

    total_a7: int
    total_a12: int

    # Maximum cores usable by MAGNET after fixed background load.
    capacity_a7: int
    capacity_a12: int

    # Current free cores.
    available_a7: int
    available_a12: int

    compute_scale: float
    power_scale: float

    queue_delay: float


@dataclass
class EdgeServerState:
    """Dynamic edge-server state."""

    edge_id: int

    x: float
    y: float

    total_a7: int
    total_a12: int

    capacity_a7: int
    capacity_a12: int

    available_a7: int
    available_a12: int

    compute_scale: float
    power_scale: float

    queue_delay: float


@dataclass
class DynamicScenarioState:
    """Complete event-driven scenario state."""

    vehicles: List[VehicleState]
    edge_servers: List[EdgeServerState]

    simulation_time: float
    seed: int

    task_released: np.ndarray
    task_ready: np.ndarray
    task_running: np.ndarray
    task_completed: np.ndarray


# ============================================================
# Resource helpers
# ============================================================


def _queue_delay_from_state(
    total_a7: int,
    total_a12: int,
    available_a7: int,
    available_a12: int,
    max_delay_sec: float,
) -> float:
    """Estimate queue pressure from current utilization."""

    total = max(
        total_a7 + total_a12,
        1,
    )

    free = max(
        available_a7
        + available_a12,
        0,
    )

    utilization = float(
        np.clip(
            1.0 - free / total,
            0.0,
            1.0,
        )
    )

    return (
        utilization
        * max_delay_sec
    )


def refresh_resource_queue_delays(
    state: DynamicScenarioState,
) -> None:
    """Refresh queue-delay features after allocation changes."""

    for vehicle in state.vehicles:

        vehicle.queue_delay = (
            _queue_delay_from_state(
                total_a7=(
                    vehicle.total_a7
                ),
                total_a12=(
                    vehicle.total_a12
                ),
                available_a7=(
                    vehicle.available_a7
                ),
                available_a12=(
                    vehicle.available_a12
                ),
                max_delay_sec=(
                    MAX_VEHICLE_QUEUE_DELAY_SEC
                ),
            )
        )

    for edge in state.edge_servers:

        edge.queue_delay = (
            _queue_delay_from_state(
                total_a7=edge.total_a7,
                total_a12=edge.total_a12,
                available_a7=(
                    edge.available_a7
                ),
                available_a12=(
                    edge.available_a12
                ),
                max_delay_sec=(
                    MAX_EDGE_QUEUE_DELAY_SEC
                ),
            )
        )


def _initial_capacity(
    total_cores: int,
    rng: np.random.Generator,
    force_full: bool,
) -> int:
    """Sample fixed background capacity."""

    if force_full:
        return total_cores

    load = float(
        rng.uniform(
            MIN_RESOURCE_LOAD,
            MAX_RESOURCE_LOAD,
        )
    )

    return max(
        1,
        int(
            round(
                total_cores
                * (1.0 - load)
            )
        ),
    )


# ============================================================
# Scenario initialization
# ============================================================


def initialize_scenario_state(
    seed: int,
    workflow_id: str
    = DAGBENCH_WORKFLOW_ID,
) -> DynamicScenarioState:
    """Create a reproducible event-driven initial scenario."""

    rng = np.random.default_rng(
        seed
    )

    dagbench_data = (
        load_dagbench_task_graph(
            workflow_id
        )
    )

    dag: nx.DiGraph = (
        dagbench_data["dag"]
    )

    num_tasks = (
        dag.number_of_nodes()
    )

    # --------------------------------------------------------
    # Vehicles
    # --------------------------------------------------------

    base_positions = np.linspace(
        70.0,
        ROAD_LENGTH_M - 70.0,
        NUM_VEHICLES,
    )

    jitter = rng.uniform(
        -40.0,
        40.0,
        size=NUM_VEHICLES,
    )

    positions = np.clip(
        base_positions + jitter,
        0.0,
        ROAD_LENGTH_M,
    )

    vehicles: List[
        VehicleState
    ] = []

    for vehicle_id in range(
        NUM_VEHICLES
    ):

        speed = float(
            rng.uniform(
                MIN_VEHICLE_SPEED_MPS,
                MAX_VEHICLE_SPEED_MPS,
            )
        )

        direction = float(
            rng.choice(
                [-1.0, 1.0]
            )
        )

        compute_scale = float(
            rng.uniform(
                0.85,
                1.15,
            )
        )

        power_scale = float(
            rng.uniform(
                0.90,
                1.10,
            )
        )

        # Source vehicle always has a valid local fallback.
        force_full = (
            vehicle_id
            == SOURCE_VEHICLE_ID
        )

        capacity_a7 = (
            _initial_capacity(
                VEHICLE_A7_CORES,
                rng,
                force_full,
            )
        )

        capacity_a12 = (
            _initial_capacity(
                VEHICLE_A12_CORES,
                rng,
                force_full,
            )
        )

        vehicles.append(
            VehicleState(
                vehicle_id=vehicle_id,

                x=float(
                    positions[
                        vehicle_id
                    ]
                ),

                y=0.0,

                speed=speed,

                direction=direction,

                total_a7=(
                    VEHICLE_A7_CORES
                ),

                total_a12=(
                    VEHICLE_A12_CORES
                ),

                capacity_a7=(
                    capacity_a7
                ),

                capacity_a12=(
                    capacity_a12
                ),

                available_a7=(
                    capacity_a7
                ),

                available_a12=(
                    capacity_a12
                ),

                compute_scale=(
                    compute_scale
                ),

                power_scale=(
                    power_scale
                ),

                queue_delay=0.0,
            )
        )

    # --------------------------------------------------------
    # Edge servers
    # --------------------------------------------------------

    edge_servers: List[
        EdgeServerState
    ] = []

    for edge_id in range(
        NUM_EDGE_SERVERS
    ):

        x = (
            ROAD_LENGTH_M
            * (edge_id + 1)
            / (
                NUM_EDGE_SERVERS + 1
            )
        )

        capacity_a7 = (
            _initial_capacity(
                EDGE_A7_CORES,
                rng,
                False,
            )
        )

        capacity_a12 = (
            _initial_capacity(
                EDGE_A12_CORES,
                rng,
                False,
            )
        )

        edge_servers.append(
            EdgeServerState(
                edge_id=edge_id,

                x=float(x),
                y=0.0,

                total_a7=(
                    EDGE_A7_CORES
                ),

                total_a12=(
                    EDGE_A12_CORES
                ),

                capacity_a7=(
                    capacity_a7
                ),

                capacity_a12=(
                    capacity_a12
                ),

                available_a7=(
                    capacity_a7
                ),

                available_a12=(
                    capacity_a12
                ),

                compute_scale=1.75,
                power_scale=1.15,

                queue_delay=0.0,
            )
        )

    # --------------------------------------------------------
    # Event-driven task state
    # --------------------------------------------------------

    released = np.zeros(
        num_tasks,
        dtype=np.float32,
    )

    ready = np.zeros(
        num_tasks,
        dtype=np.float32,
    )

    running = np.zeros(
        num_tasks,
        dtype=np.float32,
    )

    completed = np.zeros(
        num_tasks,
        dtype=np.float32,
    )

    # Only DAG roots are released at t=0.
    for task_id in dag.nodes:

        if (
            dag.in_degree(
                task_id
            )
            == 0
        ):

            released[
                task_id
            ] = 1.0

            ready[
                task_id
            ] = 1.0

    state = DynamicScenarioState(
        vehicles=vehicles,
        edge_servers=edge_servers,

        simulation_time=0.0,
        seed=seed,

        task_released=released,
        task_ready=ready,
        task_running=running,
        task_completed=completed,
    )

    refresh_resource_queue_delays(
        state
    )

    return state


# ============================================================
# Event-driven mobility
# ============================================================


def _advance_vehicle_position(
    vehicle: VehicleState,
    delta_time_sec: float,
) -> None:
    """Advance one vehicle for an arbitrary event-to-event interval."""

    if delta_time_sec <= 0.0:
        return

    if ROAD_LENGTH_M <= 0.0:
        return

    road = float(
        ROAD_LENGTH_M
    )

    # Reflected 1-D motion on [0, road].
    if vehicle.direction >= 0.0:

        phase = vehicle.x

    else:

        phase = (
            2.0 * road
            - vehicle.x
        )

    phase = (
        phase
        + vehicle.speed
        * delta_time_sec
    ) % (
        2.0 * road
    )

    if phase <= road:

        vehicle.x = float(
            phase
        )

        vehicle.direction = 1.0

    else:

        vehicle.x = float(
            2.0 * road
            - phase
        )

        vehicle.direction = -1.0


def advance_scenario_time(
    state: DynamicScenarioState,
    delta_time_sec: float,
) -> DynamicScenarioState:
    """Advance physical state directly to the next event time."""

    if delta_time_sec < 0.0:

        raise ValueError(
            "delta_time_sec cannot "
            "be negative."
        )

    if delta_time_sec == 0.0:
        return state

    for vehicle in (
        state.vehicles
    ):

        _advance_vehicle_position(
            vehicle,
            delta_time_sec,
        )

    state.simulation_time += float(
        delta_time_sec
    )

    return state


# ============================================================
# Resource conversion helpers
# ============================================================


def _vehicle_to_dict(
    vehicle: VehicleState,
) -> Dict[
    str,
    float | int,
]:

    return {
        "id":
            vehicle.vehicle_id,

        "x":
            vehicle.x,

        "y":
            vehicle.y,

        "total_a7":
            vehicle.total_a7,

        "total_a12":
            vehicle.total_a12,

        "available_a7":
            vehicle.available_a7,

        "available_a12":
            vehicle.available_a12,

        "compute_scale":
            vehicle.compute_scale,

        "power_scale":
            vehicle.power_scale,

        "queue_delay":
            vehicle.queue_delay,
    }


def _edge_to_dict(
    edge: EdgeServerState,
) -> Dict[
    str,
    float | int,
]:

    return {
        "id":
            edge.edge_id,

        "x":
            edge.x,

        "y":
            edge.y,

        "total_a7":
            edge.total_a7,

        "total_a12":
            edge.total_a12,

        "available_a7":
            edge.available_a7,

        "available_a12":
            edge.available_a12,

        "compute_scale":
            edge.compute_scale,

        "power_scale":
            edge.power_scale,

        "queue_delay":
            edge.queue_delay,
    }


# ============================================================
# Dynamic task-resource features
# ============================================================


def _candidate_edge_features(
    profile: Dict[
        str,
        float | int | str,
    ],

    resource: Dict[
        str,
        float | int,
    ],

    is_local: bool,
) -> List[float]:

    width = int(
        profile[
            "gang_size"
        ]
    )

    if (
        width < 1
        or width > MAX_GANG_SIZE
    ):

        raise ValueError(
            f"Invalid gang size: "
            f"{width}"
        )

    available_a7 = int(
        resource[
            "available_a7"
        ]
    )

    available_a12 = int(
        resource[
            "available_a12"
        ]
    )

    total_a7 = int(
        resource[
            "total_a7"
        ]
    )

    total_a12 = int(
        resource[
            "total_a12"
        ]
    )

    compute_scale = float(
        resource[
            "compute_scale"
        ]
    )

    power_scale = float(
        resource[
            "power_scale"
        ]
    )

    queue_delay = float(
        resource[
            "queue_delay"
        ]
    )

    feasible_a7 = float(
        available_a7
        >= width
    )

    feasible_a12 = float(
        available_a12
        >= width
    )

    exec_time_a7 = 0.0
    exec_time_a12 = 0.0

    energy_a7 = 0.0
    energy_a12 = 0.0

    if feasible_a7:

        exec_time_a7 = (
            float(
                profile[
                    "a7_time"
                ]
            )
            / max(
                width
                * compute_scale,
                1e-8,
            )
        )

        energy_a7 = (
            exec_time_a7
            * width
            * float(
                profile[
                    "a7_power"
                ]
            )
            * power_scale
        )

    if feasible_a12:

        exec_time_a12 = (
            float(
                profile[
                    "a12_time"
                ]
            )
            / max(
                width
                * compute_scale,
                1e-8,
            )
        )

        energy_a12 = (
            exec_time_a12
            * width
            * float(
                profile[
                    "a12_power"
                ]
            )
            * power_scale
        )

    available_ratio_a7 = (
        available_a7
        / max(
            total_a7,
            1,
        )
    )

    available_ratio_a12 = (
        available_a12
        / max(
            total_a12,
            1,
        )
    )

    estimated_finish_a7 = (
        queue_delay
        + exec_time_a7

        if feasible_a7

        else 0.0
    )

    estimated_finish_a12 = (
        queue_delay
        + exec_time_a12

        if feasible_a12

        else 0.0
    )

    return [
        exec_time_a7,
        exec_time_a12,

        energy_a7,
        energy_a12,

        available_ratio_a7,
        available_ratio_a12,

        queue_delay,

        estimated_finish_a7,
        estimated_finish_a12,

        feasible_a7,
        feasible_a12,

        float(is_local),
    ]


def _normalize_candidate_features(
    rows: np.ndarray,
) -> np.ndarray:

    rows = np.asarray(
        rows,
        dtype=np.float32,
    ).copy()

    if rows.size == 0:

        return rows.reshape(
            0,
            len(
                CANDIDATE_DYNAMIC_FEATURE_NAMES
            ),
        )

    time_columns = [
        0,
        1,
        6,
        7,
        8,
    ]

    time_scale = max(
        float(
            np.max(
                rows[
                    :,
                    time_columns,
                ]
            )
        ),
        1e-8,
    )

    rows[
        :,
        time_columns,
    ] /= time_scale

    energy_columns = [
        2,
        3,
    ]

    energy_scale = max(
        float(
            np.max(
                rows[
                    :,
                    energy_columns,
                ]
            )
        ),
        1e-8,
    )

    rows[
        :,
        energy_columns,
    ] /= energy_scale

    return rows.astype(
        np.float32
    )


# ============================================================
# Communication helpers
# ============================================================


def _state_distance(
    a_x: float,
    a_y: float,
    b_x: float,
    b_y: float,
) -> float:

    return float(
        math.hypot(
            a_x - b_x,
            a_y - b_y,
        )
    )


def _contact_duration(
    distance_m: float,
    range_m: float,

    speed_a: float,
    direction_a: float,

    speed_b: float,
    direction_b: float,
) -> float:

    remaining_distance = max(
        range_m
        - distance_m,
        0.0,
    )

    velocity_a = (
        speed_a
        * direction_a
    )

    velocity_b = (
        speed_b
        * direction_b
    )

    relative_speed = abs(
        velocity_a
        - velocity_b
    )

    if relative_speed < 0.25:

        return (
            CONTACT_DURATION_NORM_SEC
        )

    return float(
        remaining_distance
        / relative_speed
    )


def _communication_attr(
    distance_m: float,
    range_m: float,

    rate_mbps: float,
    base_rate_mbps: float,

    contact_duration_sec: float,
) -> List[float]:

    distance_norm = min(
        distance_m
        / max(
            range_m,
            1e-8,
        ),
        1.0,
    )

    rate_norm = min(
        rate_mbps
        / max(
            base_rate_mbps,
            1e-8,
        ),
        1.0,
    )

    # 1 MB = 8 megabits.
    tx_delay_per_mb = (
        8.0
        / max(
            rate_mbps,
            1e-8,
        )
    )

    tx_delay_norm = min(
        tx_delay_per_mb
        / TX_DELAY_NORM_SEC,
        1.0,
    )

    contact_norm = min(
        contact_duration_sec
        / CONTACT_DURATION_NORM_SEC,
        1.0,
    )

    return [
        distance_norm,
        rate_norm,
        tx_delay_norm,
        contact_norm,
        1.0,
    ]


# ============================================================
# Event-driven heterogeneous graph
# ============================================================


def build_dynamic_hetero_graph(
    state: DynamicScenarioState,

    workflow_id: str
    = DAGBENCH_WORKFLOW_ID,

    decision_epoch_index: int = 0,

    event_context: Optional[
        Dict[
            str,
            object,
        ]
    ] = None,
) -> Tuple[
    HeteroData,
    Dict[
        str,
        object,
    ],
]:

    dagbench_data = (
        load_dagbench_task_graph(
            workflow_id
        )
    )

    task_names = list(
        dagbench_data[
            "task_names"
        ]
    )

    task_costs = np.asarray(
        dagbench_data[
            "task_costs"
        ],
        dtype=np.float32,
    )

    dependencies = list(
        dagbench_data[
            "dependencies"
        ]
    )

    num_tasks = len(
        task_names
    )

    profiles = (
        load_project_task_profiles(
            num_tasks
        )
    )

    state_arrays = [
        state.task_released,
        state.task_ready,
        state.task_running,
        state.task_completed,
    ]

    if any(
        len(array)
        != num_tasks

        for array
        in state_arrays
    ):

        raise ValueError(
            "Task-state arrays do not "
            "match DAG task count."
        )

    # ========================================================
    # Task features
    # ========================================================

    static_task_rows = np.asarray(
        [
            [
                task_costs[
                    task_id
                ],

                float(
                    profiles[
                        task_id
                    ][
                        "gang_size"
                    ]
                ),

                float(
                    profiles[
                        task_id
                    ][
                        "deadline"
                    ]
                ),
            ]

            for task_id
            in range(
                num_tasks
            )
        ],
        dtype=np.float32,
    )

    static_task_rows = (
        _max_normalize(
            static_task_rows
        )
    )

    task_x = np.concatenate(
        [
            static_task_rows,

            state.task_released[
                :,
                None,
            ],

            state.task_ready[
                :,
                None,
            ],

            state.task_running[
                :,
                None,
            ],

            state.task_completed[
                :,
                None,
            ],
        ],
        axis=1,
    ).astype(
        np.float32
    )

    # ========================================================
    # Resource node features
    # ========================================================

    vehicle_rows = np.asarray(
        [
            [
                float(
                    vehicle.total_a7
                ),

                float(
                    vehicle.total_a12
                ),

                float(
                    vehicle.compute_scale
                ),

                float(
                    vehicle.power_scale
                ),
            ]

            for vehicle
            in state.vehicles
        ],
        dtype=np.float32,
    )

    edge_rows = np.asarray(
        [
            [
                float(
                    edge.total_a7
                ),

                float(
                    edge.total_a12
                ),

                float(
                    edge.compute_scale
                ),

                float(
                    edge.power_scale
                ),
            ]

            for edge
            in state.edge_servers
        ],
        dtype=np.float32,
    )

    resource_scale = np.max(
        np.vstack(
            [
                vehicle_rows,
                edge_rows,
            ]
        ),
        axis=0,
        keepdims=True,
    )

    resource_scale[
        resource_scale < 1e-8
    ] = 1.0

    vehicle_x = (
        vehicle_rows
        / resource_scale
    ).astype(
        np.float32
    )

    edge_x = (
        edge_rows
        / resource_scale
    ).astype(
        np.float32
    )

    # ========================================================
    # Intermediate HeteroData
    # ========================================================

    data = HeteroData()

    data[
        "task"
    ].x = torch.from_numpy(
        task_x
    )

    data[
        "task"
    ].task_id = torch.arange(
        num_tasks,
        dtype=torch.long,
    )

    data[
        "task"
    ].source_vehicle = torch.full(
        (
            num_tasks,
        ),
        SOURCE_VEHICLE_ID,
        dtype=torch.long,
    )

    data[
        "task"
    ].gang_size = torch.tensor(
        [
            int(
                profile[
                    "gang_size"
                ]
            )

            for profile
            in profiles
        ],
        dtype=torch.long,
    )

    # Vehicle nodes

    data[
        "vehicle"
    ].x = torch.from_numpy(
        vehicle_x
    )

    data[
        "vehicle"
    ].node_id = torch.arange(
        len(
            state.vehicles
        ),
        dtype=torch.long,
    )

    data[
        "vehicle"
    ].pos = torch.tensor(
        [
            [
                vehicle.x,
                vehicle.y,
            ]

            for vehicle
            in state.vehicles
        ],
        dtype=torch.float32,
    )

    data[
        "vehicle"
    ].speed = torch.tensor(
        [
            vehicle.speed

            for vehicle
            in state.vehicles
        ],
        dtype=torch.float32,
    )

    data[
        "vehicle"
    ].direction = torch.tensor(
        [
            vehicle.direction

            for vehicle
            in state.vehicles
        ],
        dtype=torch.float32,
    )

    # Edge-server nodes

    data[
        "edge_server"
    ].x = torch.from_numpy(
        edge_x
    )

    data[
        "edge_server"
    ].node_id = torch.arange(
        len(
            state.edge_servers
        ),
        dtype=torch.long,
    )

    data[
        "edge_server"
    ].pos = torch.tensor(
        [
            [
                edge.x,
                edge.y,
            ]

            for edge
            in state.edge_servers
        ],
        dtype=torch.float32,
    )

    # ========================================================
    # DAG dependency
    # ========================================================

    precedence_pairs = [
        (
            int(
                dependency[
                    "source"
                ]
            ),

            int(
                dependency[
                    "target"
                ]
            ),
        )

        for dependency
        in dependencies
    ]

    precedence_raw = np.asarray(
        [
            [
                float(
                    dependency[
                        "data_size"
                    ]
                )
            ]

            for dependency
            in dependencies
        ],
        dtype=np.float32,
    )

    precedence_attr = (
        _max_normalize(
            precedence_raw
        )
    )

    data[
        (
            "task",
            "precedes",
            "task",
        )
    ].edge_index = (
        _edge_index(
            precedence_pairs
        )
    )

    data[
        (
            "task",
            "precedes",
            "task",
        )
    ].edge_attr = (
        torch.from_numpy(
            precedence_attr
        )
    )

    # ========================================================
    # Task -> Vehicle intermediate candidates
    # ========================================================

    task_vehicle_pairs: List[
        Tuple[int, int]
    ] = []

    task_vehicle_raw: List[
        List[float]
    ] = []

    for task_id, profile in (
        enumerate(
            profiles
        )
    ):

        for vehicle in (
            state.vehicles
        ):

            task_vehicle_pairs.append(
                (
                    task_id,
                    vehicle.vehicle_id,
                )
            )

            task_vehicle_raw.append(
                _candidate_edge_features(
                    profile=profile,

                    resource=(
                        _vehicle_to_dict(
                            vehicle
                        )
                    ),

                    is_local=(
                        vehicle.vehicle_id
                        == SOURCE_VEHICLE_ID
                    ),
                )
            )

    # ========================================================
    # Task -> Edge intermediate candidates
    # ========================================================

    task_edge_pairs: List[
        Tuple[int, int]
    ] = []

    task_edge_raw: List[
        List[float]
    ] = []

    for task_id, profile in (
        enumerate(
            profiles
        )
    ):

        for edge in (
            state.edge_servers
        ):

            task_edge_pairs.append(
                (
                    task_id,
                    edge.edge_id,
                )
            )

            task_edge_raw.append(
                _candidate_edge_features(
                    profile=profile,

                    resource=(
                        _edge_to_dict(
                            edge
                        )
                    ),

                    is_local=False,
                )
            )

    all_candidate_raw = np.asarray(
        task_vehicle_raw
        + task_edge_raw,
        dtype=np.float32,
    )

    all_candidate_norm = (
        _normalize_candidate_features(
            all_candidate_raw
        )
    )

    num_task_vehicle = len(
        task_vehicle_raw
    )

    task_vehicle_attr = (
        all_candidate_norm[
            :num_task_vehicle
        ]
    )

    task_edge_attr = (
        all_candidate_norm[
            num_task_vehicle:
        ]
    )

    data[
        (
            "task",
            "candidate_on",
            "vehicle",
        )
    ].edge_index = (
        _edge_index(
            task_vehicle_pairs
        )
    )

    data[
        (
            "task",
            "candidate_on",
            "vehicle",
        )
    ].edge_attr = (
        torch.from_numpy(
            task_vehicle_attr
        )
    )

    data[
        (
            "task",
            "candidate_on",
            "edge_server",
        )
    ].edge_index = (
        _edge_index(
            task_edge_pairs
        )
    )

    data[
        (
            "task",
            "candidate_on",
            "edge_server",
        )
    ].edge_attr = (
        torch.from_numpy(
            task_edge_attr
        )
    )

    # ========================================================
    # V2V communication
    # ========================================================

    v2v_pairs: List[
        Tuple[int, int]
    ] = []

    v2v_attr: List[
        List[float]
    ] = []

    vehicles = (
        state.vehicles
    )

    for first_vehicle in range(
        len(vehicles)
    ):

        for second_vehicle in range(
            first_vehicle + 1,
            len(vehicles),
        ):

            vehicle_a = (
                vehicles[
                    first_vehicle
                ]
            )

            vehicle_b = (
                vehicles[
                    second_vehicle
                ]
            )

            distance_m = (
                _state_distance(
                    vehicle_a.x,
                    vehicle_a.y,

                    vehicle_b.x,
                    vehicle_b.y,
                )
            )

            if (
                distance_m
                > V2V_RANGE_M
            ):
                continue

            rate = _link_rate(
                distance_m=distance_m,

                range_m=V2V_RANGE_M,

                base_rate_mbps=(
                    BASE_V2V_RATE_MBPS
                ),
            )

            contact = (
                _contact_duration(
                    distance_m=(
                        distance_m
                    ),

                    range_m=(
                        V2V_RANGE_M
                    ),

                    speed_a=(
                        vehicle_a.speed
                    ),

                    direction_a=(
                        vehicle_a.direction
                    ),

                    speed_b=(
                        vehicle_b.speed
                    ),

                    direction_b=(
                        vehicle_b.direction
                    ),
                )
            )

            attributes = (
                _communication_attr(
                    distance_m=(
                        distance_m
                    ),

                    range_m=(
                        V2V_RANGE_M
                    ),

                    rate_mbps=rate,

                    base_rate_mbps=(
                        BASE_V2V_RATE_MBPS
                    ),

                    contact_duration_sec=(
                        contact
                    ),
                )
            )

            v2v_pairs.extend(
                [
                    (
                        first_vehicle,
                        second_vehicle,
                    ),

                    (
                        second_vehicle,
                        first_vehicle,
                    ),
                ]
            )

            v2v_attr.extend(
                [
                    attributes,
                    attributes,
                ]
            )

    data[
        (
            "vehicle",
            "distance_to",
            "vehicle",
        )
    ].edge_index = (
        _edge_index(
            v2v_pairs
        )
    )

    data[
        (
            "vehicle",
            "distance_to",
            "vehicle",
        )
    ].edge_attr = (
        _tensor_2d(
            v2v_attr,

            len(
                COMMUNICATION_FEATURE_NAMES
            ),
        )
    )

    # ========================================================
    # V2I communication
    # ========================================================

    v2i_pairs: List[
        Tuple[int, int]
    ] = []

    v2i_attr: List[
        List[float]
    ] = []

    for vehicle in vehicles:

        for edge in (
            state.edge_servers
        ):

            distance_m = (
                _state_distance(
                    vehicle.x,
                    vehicle.y,

                    edge.x,
                    edge.y,
                )
            )

            if (
                distance_m
                > V2I_RANGE_M
            ):
                continue

            rate = _link_rate(
                distance_m=distance_m,

                range_m=V2I_RANGE_M,

                base_rate_mbps=(
                    BASE_V2I_RATE_MBPS
                ),
            )

            contact = (
                _contact_duration(
                    distance_m=(
                        distance_m
                    ),

                    range_m=(
                        V2I_RANGE_M
                    ),

                    speed_a=(
                        vehicle.speed
                    ),

                    direction_a=(
                        vehicle.direction
                    ),

                    speed_b=0.0,

                    direction_b=0.0,
                )
            )

            attributes = (
                _communication_attr(
                    distance_m=(
                        distance_m
                    ),

                    range_m=(
                        V2I_RANGE_M
                    ),

                    rate_mbps=rate,

                    base_rate_mbps=(
                        BASE_V2I_RATE_MBPS
                    ),

                    contact_duration_sec=(
                        contact
                    ),
                )
            )

            v2i_pairs.append(
                (
                    vehicle.vehicle_id,
                    edge.edge_id,
                )
            )

            v2i_attr.append(
                attributes
            )

    data[
        (
            "vehicle",
            "distance_to",
            "edge_server",
        )
    ].edge_index = (
        _edge_index(
            v2i_pairs
        )
    )

    data[
        (
            "vehicle",
            "distance_to",
            "edge_server",
        )
    ].edge_attr = (
        _tensor_2d(
            v2i_attr,

            len(
                COMMUNICATION_FEATURE_NAMES
            ),
        )
    )

    # ========================================================
    # Event metadata
    # ========================================================

    context = dict(
        event_context
        or {}
    )

    trigger_event = str(
        context.get(
            "trigger_event",
            "manual_snapshot",
        )
    )

    metadata: Dict[
        str,
        object,
    ] = {

        "workflow_id":
            workflow_id,

        "scenario_seed":
            state.seed,

        "decision_epoch_index":
            decision_epoch_index,

        "simulation_time":
            float(
                state.simulation_time
            ),

        "trigger_event":
            trigger_event,

        "affected_tasks":
            list(
                context.get(
                    "affected_tasks",
                    [],
                )
            ),

        "started_task":
            context.get(
                "started_task"
            ),

        "completed_tasks":
            list(
                context.get(
                    "completed_tasks",
                    [],
                )
            ),

        "released_tasks":
            list(
                context.get(
                    "released_tasks",
                    [],
                )
            ),

        "num_tasks":
            num_tasks,

        "num_vehicles":
            len(
                state.vehicles
            ),

        "num_edge_servers":
            len(
                state.edge_servers
            ),

        "num_released_tasks":
            int(
                state.task_released.sum()
            ),

        "num_ready_tasks":
            int(
                state.task_ready.sum()
            ),

        "num_running_tasks":
            int(
                state.task_running.sum()
            ),

        "num_completed_tasks":
            int(
                state.task_completed.sum()
            ),

        "feature_names": {

            "task_node":
                TASK_DYNAMIC_FEATURE_NAMES,

            "vehicle_node":
                RESOURCE_NODE_FEATURE_NAMES,

            "edge_server_node":
                RESOURCE_NODE_FEATURE_NAMES,

            "precedence_edge":
                PRECEDENCE_FEATURE_NAMES,

            "candidate_edge":
                CANDIDATE_DYNAMIC_FEATURE_NAMES,

            "communication_edge":
                COMMUNICATION_FEATURE_NAMES,
        },
    }

    # ========================================================
    # Convert to final five MAGNET layers
    # ========================================================

    data, metadata = (
        apply_magnet_relation_layers(
            source_data=data,

            metadata=metadata,

            task_feature_names=(
                TASK_DYNAMIC_FEATURE_NAMES
            ),

            candidate_feature_names=(
                CANDIDATE_DYNAMIC_FEATURE_NAMES
            ),

            communication_feature_names=(
                COMMUNICATION_FEATURE_NAMES
            ),

            dynamic=True,
        )
    )

    # Store event information directly in the .pt graph too.
    data.simulation_time = torch.tensor(
        [
            state.simulation_time
        ],
        dtype=torch.float64,
    )

    data.decision_epoch_index = (
        torch.tensor(
            [
                decision_epoch_index
            ],
            dtype=torch.long,
        )
    )

    data.trigger_event = (
        trigger_event
    )

    return (
        data,
        metadata,
    )