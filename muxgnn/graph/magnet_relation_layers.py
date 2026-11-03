from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
import torch
from torch_geometric.data import HeteroData

from .settings import (
    EDGE_A12_CORES,
    EDGE_A7_CORES,
    MAX_GANG_SIZE,
    NUM_EDGE_SERVERS,
    NUM_VEHICLES,
    VEHICLE_A12_CORES,
    VEHICLE_A7_CORES,
)


# ============================================================
# Final MAGNET relation layers
# ============================================================

DAG_DEPENDENCY_RELATION = (
    "task",
    "precedes",
    "task",
)

LOCAL_EXECUTION_RELATION = (
    "task",
    "local_execute_on",
    "vehicle",
)

V2V_OFFLOADING_RELATION = (
    "task",
    "v2v_offload_to",
    "vehicle",
)

V2I_OFFLOADING_RELATION = (
    "task",
    "v2i_offload_to",
    "edge_server",
)

GANG_CONTENTION_RELATION = (
    "task",
    "contends_with",
    "task",
)


# ============================================================
# Intermediate relations created by the graph builders
# ============================================================

INTERMEDIATE_TASK_VEHICLE_RELATION = (
    "task",
    "candidate_on",
    "vehicle",
)

INTERMEDIATE_TASK_EDGE_RELATION = (
    "task",
    "candidate_on",
    "edge_server",
)

INTERMEDIATE_V2V_RELATION = (
    "vehicle",
    "distance_to",
    "vehicle",
)

INTERMEDIATE_V2I_RELATION = (
    "vehicle",
    "distance_to",
    "edge_server",
)


# ============================================================
# Gang contention feature schema
# ============================================================

CONTENTION_FEATURE_NAMES = [
    "source_gang_ratio",
    "target_gang_ratio",
    "combined_gang_ratio",
    "shared_resource_ratio",
    "blocking_ratio",
    "pair_infeasible",
]


# ============================================================
# Generic helpers
# ============================================================


def _make_edge_index(
    pairs: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Convert edge pairs to PyG edge_index."""

    if not pairs:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    return (
        torch.tensor(
            pairs,
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )


def _stack_edge_attributes(
    rows: List[torch.Tensor],
    width: int,
) -> torch.Tensor:
    """Stack edge feature rows."""

    if not rows:
        return torch.empty(
            (0, width),
            dtype=torch.float32,
        )

    return torch.stack(
        rows,
        dim=0,
    ).to(
        dtype=torch.float32,
    )


def _copy_node_stores(
    source: HeteroData,
    target: HeteroData,
) -> None:
    """Copy all node attributes into the final graph."""

    for node_type in source.node_types:

        for key, value in source[node_type].items():

            if torch.is_tensor(value):

                target[node_type][key] = (
                    value.clone()
                )

            else:

                target[node_type][key] = (
                    copy.deepcopy(value)
                )


# ============================================================
# Communication lookup
# ============================================================


def _build_communication_lookup(
    data: HeteroData,
    relation: Tuple[str, str, str],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Create a lookup for communication edge features."""

    lookup: Dict[
        Tuple[int, int],
        torch.Tensor,
    ] = {}

    if relation not in data.edge_types:
        return lookup

    edge_index = data[
        relation
    ].edge_index

    edge_attr = data[
        relation
    ].edge_attr

    for edge_id in range(
        edge_index.shape[1]
    ):

        source = int(
            edge_index[
                0,
                edge_id,
            ]
        )

        target = int(
            edge_index[
                1,
                edge_id,
            ]
        )

        lookup[
            (
                source,
                target,
            )
        ] = edge_attr[
            edge_id
        ].clone()

    return lookup


# ============================================================
# Resource availability
# ============================================================


def _extract_available_cores(
    source_data: HeteroData,
    candidate_feature_names: Sequence[str],
    dynamic: bool,
) -> Dict[
    Tuple[str, int],
    Tuple[int, int],
]:
    """Recover current A7/A12 core availability."""

    availability: Dict[
        Tuple[str, int],
        Tuple[int, int],
    ] = {}

    # Static graph uses full configured capacity.
    if not dynamic:

        for vehicle_id in range(
            source_data[
                "vehicle"
            ].num_nodes
        ):

            availability[
                (
                    "vehicle",
                    vehicle_id,
                )
            ] = (
                VEHICLE_A7_CORES,
                VEHICLE_A12_CORES,
            )

        for edge_id in range(
            source_data[
                "edge_server"
            ].num_nodes
        ):

            availability[
                (
                    "edge_server",
                    edge_id,
                )
            ] = (
                EDGE_A7_CORES,
                EDGE_A12_CORES,
            )

        return availability

    # Dynamic graph keeps the availability ratios
    # on task-resource relations.
    a7_ratio_index = (
        candidate_feature_names.index(
            "available_ratio_a7"
        )
    )

    a12_ratio_index = (
        candidate_feature_names.index(
            "available_ratio_a12"
        )
    )

    # --------------------------------------------------------
    # Vehicles
    # --------------------------------------------------------

    vehicle_index = source_data[
        INTERMEDIATE_TASK_VEHICLE_RELATION
    ].edge_index

    vehicle_attr = source_data[
        INTERMEDIATE_TASK_VEHICLE_RELATION
    ].edge_attr

    seen_vehicles: Set[int] = set()

    for edge_id in range(
        vehicle_index.shape[1]
    ):

        vehicle_id = int(
            vehicle_index[
                1,
                edge_id,
            ]
        )

        if vehicle_id in seen_vehicles:
            continue

        a7_ratio = float(
            vehicle_attr[
                edge_id,
                a7_ratio_index,
            ]
        )

        a12_ratio = float(
            vehicle_attr[
                edge_id,
                a12_ratio_index,
            ]
        )

        availability[
            (
                "vehicle",
                vehicle_id,
            )
        ] = (
            int(
                round(
                    a7_ratio
                    * VEHICLE_A7_CORES
                )
            ),
            int(
                round(
                    a12_ratio
                    * VEHICLE_A12_CORES
                )
            ),
        )

        seen_vehicles.add(
            vehicle_id
        )

    # --------------------------------------------------------
    # Edge servers
    # --------------------------------------------------------

    edge_index = source_data[
        INTERMEDIATE_TASK_EDGE_RELATION
    ].edge_index

    edge_attr = source_data[
        INTERMEDIATE_TASK_EDGE_RELATION
    ].edge_attr

    seen_servers: Set[int] = set()

    for relation_id in range(
        edge_index.shape[1]
    ):

        server_id = int(
            edge_index[
                1,
                relation_id,
            ]
        )

        if server_id in seen_servers:
            continue

        a7_ratio = float(
            edge_attr[
                relation_id,
                a7_ratio_index,
            ]
        )

        a12_ratio = float(
            edge_attr[
                relation_id,
                a12_ratio_index,
            ]
        )

        availability[
            (
                "edge_server",
                server_id,
            )
        ] = (
            int(
                round(
                    a7_ratio
                    * EDGE_A7_CORES
                )
            ),
            int(
                round(
                    a12_ratio
                    * EDGE_A12_CORES
                )
            ),
        )

        seen_servers.add(
            server_id
        )

    return availability


# ============================================================
# Gang resource contention
# ============================================================


def _can_two_gangs_run_together(
    gang_a: int,
    gang_b: int,
    available_a7: int,
    available_a12: int,
) -> bool:
    """Check simultaneous homogeneous-core gang allocation."""

    # Both gangs use A7 cores.
    if (
        gang_a + gang_b
        <= available_a7
    ):
        return True

    # Both gangs use A12 cores.
    if (
        gang_a + gang_b
        <= available_a12
    ):
        return True

    # Gang A uses A7 and Gang B uses A12.
    if (
        gang_a <= available_a7
        and gang_b <= available_a12
    ):
        return True

    # Gang A uses A12 and Gang B uses A7.
    if (
        gang_a <= available_a12
        and gang_b <= available_a7
    ):
        return True

    return False


def _extract_active_tasks(
    data: HeteroData,
    task_feature_names: Sequence[str],
) -> List[int]:
    """Return tasks that may participate in contention."""

    num_tasks = data[
        "task"
    ].num_nodes

    # Static graph has no runtime ready/completed state.
    if (
        "is_ready"
        not in task_feature_names
    ):

        return list(
            range(num_tasks)
        )

    ready_index = (
        task_feature_names.index(
            "is_ready"
        )
    )

    completed_index = (
        task_feature_names.index(
            "is_completed"
        )
    )

    task_x = data[
        "task"
    ].x

    active_tasks: List[int] = []

    for task_id in range(
        num_tasks
    ):

        ready = float(
            task_x[
                task_id,
                ready_index,
            ]
        )

        completed = float(
            task_x[
                task_id,
                completed_index,
            ]
        )

        if (
            ready > 0.5
            and completed < 0.5
        ):

            active_tasks.append(
                task_id
            )

    return active_tasks


def _build_task_resource_map(
    final_data: HeteroData,
) -> List[
    Set[
        Tuple[str, int]
    ]
]:
    """Find resources accessible to every task."""

    num_tasks = final_data[
        "task"
    ].num_nodes

    task_resources: List[
        Set[
            Tuple[str, int]
        ]
    ] = [
        set()
        for _ in range(
            num_tasks
        )
    ]

    # --------------------------------------------------------
    # Local vehicle
    # --------------------------------------------------------

    local_index = final_data[
        LOCAL_EXECUTION_RELATION
    ].edge_index

    for edge_id in range(
        local_index.shape[1]
    ):

        task_id = int(
            local_index[
                0,
                edge_id,
            ]
        )

        vehicle_id = int(
            local_index[
                1,
                edge_id,
            ]
        )

        task_resources[
            task_id
        ].add(
            (
                "vehicle",
                vehicle_id,
            )
        )

    # --------------------------------------------------------
    # Neighbor vehicles
    # --------------------------------------------------------

    v2v_index = final_data[
        V2V_OFFLOADING_RELATION
    ].edge_index

    for edge_id in range(
        v2v_index.shape[1]
    ):

        task_id = int(
            v2v_index[
                0,
                edge_id,
            ]
        )

        vehicle_id = int(
            v2v_index[
                1,
                edge_id,
            ]
        )

        task_resources[
            task_id
        ].add(
            (
                "vehicle",
                vehicle_id,
            )
        )

    # --------------------------------------------------------
    # Edge servers
    # --------------------------------------------------------

    v2i_index = final_data[
        V2I_OFFLOADING_RELATION
    ].edge_index

    for edge_id in range(
        v2i_index.shape[1]
    ):

        task_id = int(
            v2i_index[
                0,
                edge_id,
            ]
        )

        server_id = int(
            v2i_index[
                1,
                edge_id,
            ]
        )

        task_resources[
            task_id
        ].add(
            (
                "edge_server",
                server_id,
            )
        )

    return task_resources


def _build_gang_contention_layer(
    source_data: HeteroData,
    final_data: HeteroData,
    task_feature_names: Sequence[str],
    candidate_feature_names: Sequence[str],
    dynamic: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Build Task-to-Task gang resource contention edges."""

    num_tasks = final_data[
        "task"
    ].num_nodes

    gang_sizes = final_data[
        "task"
    ].gang_size.to(
        dtype=torch.long
    )

    availability = (
        _extract_available_cores(
            source_data=
                source_data,

            candidate_feature_names=
                candidate_feature_names,

            dynamic=
                dynamic,
        )
    )

    task_resources = (
        _build_task_resource_map(
            final_data
        )
    )

    # --------------------------------------------------------
    # Dependency reachability
    # --------------------------------------------------------

    dag = nx.DiGraph()

    dag.add_nodes_from(
        range(num_tasks)
    )

    dependency_index = final_data[
        DAG_DEPENDENCY_RELATION
    ].edge_index

    dag.add_edges_from(
        dependency_index
        .t()
        .tolist()
    )

    descendants = {
        task_id:
            nx.descendants(
                dag,
                task_id,
            )

        for task_id in range(
            num_tasks
        )
    }

    active_tasks = (
        _extract_active_tasks(
            final_data,
            task_feature_names,
        )
    )

    contention_pairs: List[
        Tuple[int, int]
    ] = []

    contention_rows: List[
        torch.Tensor
    ] = []

    max_resource_count = max(
        NUM_VEHICLES
        + NUM_EDGE_SERVERS,
        1,
    )

    # --------------------------------------------------------
    # Pairwise contention
    # --------------------------------------------------------

    for first_index in range(
        len(active_tasks)
    ):

        task_a = active_tasks[
            first_index
        ]

        for second_index in range(
            first_index + 1,
            len(active_tasks),
        ):

            task_b = active_tasks[
                second_index
            ]

            # Tasks with a dependency path cannot execute
            # concurrently and are not resource competitors.
            if (
                task_b
                in descendants[
                    task_a
                ]
                or
                task_a
                in descendants[
                    task_b
                ]
            ):

                continue

            shared_resources = (
                task_resources[
                    task_a
                ]
                &
                task_resources[
                    task_b
                ]
            )

            if not shared_resources:
                continue

            gang_a = int(
                gang_sizes[
                    task_a
                ]
            )

            gang_b = int(
                gang_sizes[
                    task_b
                ]
            )

            blocked_resources = 0

            for resource in (
                shared_resources
            ):

                if resource not in (
                    availability
                ):

                    continue

                (
                    available_a7,
                    available_a12,
                ) = availability[
                    resource
                ]

                can_coexist = (
                    _can_two_gangs_run_together(
                        gang_a=
                            gang_a,

                        gang_b=
                            gang_b,

                        available_a7=
                            available_a7,

                        available_a12=
                            available_a12,
                    )
                )

                if not can_coexist:
                    blocked_resources += 1

            shared_count = len(
                shared_resources
            )

            blocking_ratio = (
                blocked_resources
                / max(
                    shared_count,
                    1,
                )
            )

            pair_infeasible = float(
                blocked_resources
                == shared_count
            )

            shared_resource_ratio = (
                shared_count
                / max_resource_count
            )

            source_gang_ratio = (
                gang_a
                / MAX_GANG_SIZE
            )

            target_gang_ratio = (
                gang_b
                / MAX_GANG_SIZE
            )

            combined_gang_ratio = min(
                (
                    gang_a
                    + gang_b
                )
                / (
                    2
                    * MAX_GANG_SIZE
                ),
                1.0,
            )

            forward_attr = torch.tensor(
                [
                    source_gang_ratio,
                    target_gang_ratio,
                    combined_gang_ratio,
                    shared_resource_ratio,
                    blocking_ratio,
                    pair_infeasible,
                ],
                dtype=torch.float32,
            )

            reverse_attr = torch.tensor(
                [
                    target_gang_ratio,
                    source_gang_ratio,
                    combined_gang_ratio,
                    shared_resource_ratio,
                    blocking_ratio,
                    pair_infeasible,
                ],
                dtype=torch.float32,
            )

            # Contention is conceptually symmetric.
            contention_pairs.append(
                (
                    task_a,
                    task_b,
                )
            )

            contention_rows.append(
                forward_attr
            )

            contention_pairs.append(
                (
                    task_b,
                    task_a,
                )
            )

            contention_rows.append(
                reverse_attr
            )

    return (
        _make_edge_index(
            contention_pairs
        ),
        _stack_edge_attributes(
            contention_rows,
            len(
                CONTENTION_FEATURE_NAMES
            ),
        ),
    )


# ============================================================
# Final MAGNET graph conversion
# ============================================================


def apply_magnet_relation_layers(
    source_data: HeteroData,
    metadata: Dict[str, object],
    task_feature_names: Sequence[str],
    candidate_feature_names: Sequence[str],
    communication_feature_names: Sequence[str],
    dynamic: bool,
) -> Tuple[
    HeteroData,
    Dict[str, object],
]:
    """Convert the intermediate graph to the five MAGNET layers."""

    if (
        not candidate_feature_names
        or candidate_feature_names[-1]
        != "is_local"
    ):

        raise ValueError(
            "The final candidate feature "
            "must be 'is_local'."
        )

    final_data = HeteroData()

    _copy_node_stores(
        source_data,
        final_data,
    )

    # ========================================================
    # 1. DAG Dependency Layer
    # ========================================================

    final_data[
        DAG_DEPENDENCY_RELATION
    ].edge_index = (
        source_data[
            DAG_DEPENDENCY_RELATION
        ].edge_index.clone()
    )

    final_data[
        DAG_DEPENDENCY_RELATION
    ].edge_attr = (
        source_data[
            DAG_DEPENDENCY_RELATION
        ].edge_attr.clone()
    )

    # ========================================================
    # Communication lookups
    # ========================================================

    v2v_lookup = (
        _build_communication_lookup(
            source_data,
            INTERMEDIATE_V2V_RELATION,
        )
    )

    v2i_lookup = (
        _build_communication_lookup(
            source_data,
            INTERMEDIATE_V2I_RELATION,
        )
    )

    # is_local is only needed to split Local and V2V.
    relation_candidate_names = list(
        candidate_feature_names[
            :-1
        ]
    )

    candidate_width = len(
        relation_candidate_names
    )

    communication_width = len(
        communication_feature_names
    )

    # ========================================================
    # 2. Local Execution Layer
    # 3. V2V Offloading Layer
    # ========================================================

    task_vehicle_index = source_data[
        INTERMEDIATE_TASK_VEHICLE_RELATION
    ].edge_index

    task_vehicle_attr = source_data[
        INTERMEDIATE_TASK_VEHICLE_RELATION
    ].edge_attr

    source_vehicle = source_data[
        "task"
    ].source_vehicle.to(
        dtype=torch.long
    )

    local_pairs: List[
        Tuple[int, int]
    ] = []

    local_rows: List[
        torch.Tensor
    ] = []

    v2v_pairs: List[
        Tuple[int, int]
    ] = []

    v2v_rows: List[
        torch.Tensor
    ] = []

    for edge_id in range(
        task_vehicle_index.shape[1]
    ):

        task_id = int(
            task_vehicle_index[
                0,
                edge_id,
            ]
        )

        vehicle_id = int(
            task_vehicle_index[
                1,
                edge_id,
            ]
        )

        origin_vehicle = int(
            source_vehicle[
                task_id
            ]
        )

        candidate_features = (
            task_vehicle_attr[
                edge_id,
                :-1,
            ].clone()
        )

        # ----------------------------------------------------
        # Local execution
        # ----------------------------------------------------

        if vehicle_id == origin_vehicle:

            local_pairs.append(
                (
                    task_id,
                    vehicle_id,
                )
            )

            local_rows.append(
                candidate_features
            )

            continue

        # ----------------------------------------------------
        # V2V offloading
        # ----------------------------------------------------

        communication_key = (
            origin_vehicle,
            vehicle_id,
        )

        # No communication edge means the vehicle
        # is currently not a valid V2V neighbor.
        if (
            communication_key
            not in v2v_lookup
        ):

            continue

        combined_features = torch.cat(
            [
                candidate_features,
                v2v_lookup[
                    communication_key
                ],
            ],
            dim=0,
        )

        v2v_pairs.append(
            (
                task_id,
                vehicle_id,
            )
        )

        v2v_rows.append(
            combined_features
        )

    final_data[
        LOCAL_EXECUTION_RELATION
    ].edge_index = (
        _make_edge_index(
            local_pairs
        )
    )

    final_data[
        LOCAL_EXECUTION_RELATION
    ].edge_attr = (
        _stack_edge_attributes(
            local_rows,
            candidate_width,
        )
    )

    final_data[
        V2V_OFFLOADING_RELATION
    ].edge_index = (
        _make_edge_index(
            v2v_pairs
        )
    )

    final_data[
        V2V_OFFLOADING_RELATION
    ].edge_attr = (
        _stack_edge_attributes(
            v2v_rows,
            candidate_width
            + communication_width,
        )
    )

    # ========================================================
    # 4. V2I / Edge Offloading Layer
    # ========================================================

    task_edge_index = source_data[
        INTERMEDIATE_TASK_EDGE_RELATION
    ].edge_index

    task_edge_attr = source_data[
        INTERMEDIATE_TASK_EDGE_RELATION
    ].edge_attr

    v2i_pairs: List[
        Tuple[int, int]
    ] = []

    v2i_rows: List[
        torch.Tensor
    ] = []

    for edge_id in range(
        task_edge_index.shape[1]
    ):

        task_id = int(
            task_edge_index[
                0,
                edge_id,
            ]
        )

        server_id = int(
            task_edge_index[
                1,
                edge_id,
            ]
        )

        origin_vehicle = int(
            source_vehicle[
                task_id
            ]
        )

        communication_key = (
            origin_vehicle,
            server_id,
        )

        # No V2I link means the edge server is
        # not currently reachable.
        if (
            communication_key
            not in v2i_lookup
        ):

            continue

        candidate_features = (
            task_edge_attr[
                edge_id,
                :-1,
            ].clone()
        )

        combined_features = torch.cat(
            [
                candidate_features,
                v2i_lookup[
                    communication_key
                ],
            ],
            dim=0,
        )

        v2i_pairs.append(
            (
                task_id,
                server_id,
            )
        )

        v2i_rows.append(
            combined_features
        )

    final_data[
        V2I_OFFLOADING_RELATION
    ].edge_index = (
        _make_edge_index(
            v2i_pairs
        )
    )

    final_data[
        V2I_OFFLOADING_RELATION
    ].edge_attr = (
        _stack_edge_attributes(
            v2i_rows,
            candidate_width
            + communication_width,
        )
    )

    # ========================================================
    # 5. Gang Resource-Contention Layer
    # ========================================================

    (
        contention_index,
        contention_attr,
    ) = _build_gang_contention_layer(
        source_data=
            source_data,

        final_data=
            final_data,

        task_feature_names=
            task_feature_names,

        candidate_feature_names=
            candidate_feature_names,

        dynamic=
            dynamic,
    )

    final_data[
        GANG_CONTENTION_RELATION
    ].edge_index = (
        contention_index
    )

    final_data[
        GANG_CONTENTION_RELATION
    ].edge_attr = (
        contention_attr
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = copy.deepcopy(
        metadata
    )

    feature_names = (
        metadata.setdefault(
            "feature_names",
            {},
        )
    )

    # Remove names belonging to intermediate relations.
    feature_names.pop(
        "precedence_edge",
        None,
    )

    feature_names.pop(
        "candidate_edge",
        None,
    )

    feature_names.pop(
        "distance_edge",
        None,
    )

    feature_names.pop(
        "communication_edge",
        None,
    )

    feature_names[
        "dag_dependency_edge"
    ] = [
        "data_size"
    ]

    feature_names[
        "local_execution_edge"
    ] = list(
        relation_candidate_names
    )

    feature_names[
        "v2v_offloading_edge"
    ] = (
        list(
            relation_candidate_names
        )
        + list(
            communication_feature_names
        )
    )

    feature_names[
        "v2i_offloading_edge"
    ] = (
        list(
            relation_candidate_names
        )
        + list(
            communication_feature_names
        )
    )

    feature_names[
        "gang_contention_edge"
    ] = list(
        CONTENTION_FEATURE_NAMES
    )

    metadata[
        "relation_layers"
    ] = [
        "DAG Dependency Layer",
        "Local Execution Layer",
        "V2V Offloading Layer",
        "V2I/Edge Offloading Layer",
        "Gang Resource-Contention Layer",
    ]

    metadata[
        "relations"
    ] = [
        list(relation)
        for relation
        in final_data.edge_types
    ]

    return (
        final_data,
        metadata,
    )