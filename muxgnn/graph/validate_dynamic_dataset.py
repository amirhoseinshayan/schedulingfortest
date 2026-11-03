from __future__ import annotations

import json

from pathlib import Path

import networkx as nx
import torch

from .dynamic_builder import (
    CANDIDATE_DYNAMIC_FEATURE_NAMES,
    COMMUNICATION_FEATURE_NAMES,
    TASK_DYNAMIC_FEATURE_NAMES,
)

from .dynamic_settings import (
    DYNAMIC_OUTPUT_DIR,
    MANIFEST_PATH,
    NUM_TEST_SCENARIOS,
    NUM_TRAIN_SCENARIOS,
    NUM_VAL_SCENARIOS,
)

from .magnet_relation_layers import (
    CONTENTION_FEATURE_NAMES,
    DAG_DEPENDENCY_RELATION,
    GANG_CONTENTION_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2I_OFFLOADING_RELATION,
    V2V_OFFLOADING_RELATION,
)


EXPECTED_NODE_TYPES = {
    "task",
    "vehicle",
    "edge_server",
}


EXPECTED_EDGE_TYPES = {
    DAG_DEPENDENCY_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2V_OFFLOADING_RELATION,
    V2I_OFFLOADING_RELATION,
    GANG_CONTENTION_RELATION,
}


ALLOWED_TRIGGER_EVENTS = {
    "initial_task_release",
    "task_started",
    "task_completion",
    "task_completion_and_release",
}


def _load_graph(
    path: Path,
):

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )


def _check_finite(
    name: str,
    tensor: torch.Tensor,
) -> None:

    if not torch.isfinite(
        tensor
    ).all():

        raise AssertionError(
            f"{name} contains "
            f"NaN or Inf."
        )


def _check_edge_bounds(
    graph,
    relation,
) -> None:

    (
        source_type,
        _,
        target_type,
    ) = relation

    edge_index = (
        graph[
            relation
        ].edge_index
    )

    if (
        edge_index.ndim != 2
        or edge_index.shape[0]
        != 2
    ):

        raise AssertionError(
            f"{relation}: invalid "
            f"edge_index shape."
        )

    if edge_index.numel() == 0:
        return

    if int(
        edge_index[
            0
        ].min()
    ) < 0:

        raise AssertionError(
            f"{relation}: negative "
            f"source index."
        )

    if int(
        edge_index[
            1
        ].min()
    ) < 0:

        raise AssertionError(
            f"{relation}: negative "
            f"target index."
        )

    if int(
        edge_index[
            0
        ].max()
    ) >= graph[
        source_type
    ].num_nodes:

        raise AssertionError(
            f"{relation}: source "
            f"index out of range."
        )

    if int(
        edge_index[
            1
        ].max()
    ) >= graph[
        target_type
    ].num_nodes:

        raise AssertionError(
            f"{relation}: target "
            f"index out of range."
        )


def _check_contention_symmetry(
    graph,
    path: Path,
) -> None:

    edge_index = (
        graph[
            GANG_CONTENTION_RELATION
        ].edge_index
    )

    pairs = {
        (
            int(source),
            int(target),
        )

        for source, target
        in edge_index
        .t()
        .tolist()
    }

    for source, target in pairs:

        if (
            target,
            source,
        ) not in pairs:

            raise AssertionError(
                f"{path}: gang "
                f"contention layer "
                f"is not symmetric."
            )


def _task_state_tensor(
    graph,
) -> torch.Tensor:

    indexes = [
        TASK_DYNAMIC_FEATURE_NAMES.index(
            "is_released"
        ),

        TASK_DYNAMIC_FEATURE_NAMES.index(
            "is_ready"
        ),

        TASK_DYNAMIC_FEATURE_NAMES.index(
            "is_running"
        ),

        TASK_DYNAMIC_FEATURE_NAMES.index(
            "is_completed"
        ),
    ]

    return (
        graph[
            "task"
        ].x[
            :,
            indexes,
        ]
    )


def _validate_graph(
    graph,
    path: Path,
) -> None:

    if (
        set(
            graph.node_types
        )
        != EXPECTED_NODE_TYPES
    ):

        raise AssertionError(
            f"{path}: unexpected "
            f"node types."
        )

    if (
        set(
            graph.edge_types
        )
        != EXPECTED_EDGE_TYPES
    ):

        raise AssertionError(
            f"{path}: expected "
            f"exactly five MAGNET "
            f"relation layers.\n"
            f"Found: "
            f"{graph.edge_types}"
        )

    num_tasks = (
        graph[
            "task"
        ].num_nodes
    )

    if (
        graph[
            "task"
        ].x.shape[1]
        != len(
            TASK_DYNAMIC_FEATURE_NAMES
        )
    ):

        raise AssertionError(
            f"{path}: invalid task "
            f"feature dimension."
        )

    # ========================================================
    # Edge feature dimensions
    # ========================================================

    local_width = (
        len(
            CANDIDATE_DYNAMIC_FEATURE_NAMES
        )
        - 1
    )

    if (
        graph[
            LOCAL_EXECUTION_RELATION
        ].edge_attr.shape[1]
        != local_width
    ):

        raise AssertionError(
            f"{path}: invalid "
            f"Local Execution "
            f"feature dimension."
        )

    offload_width = (
        local_width
        + len(
            COMMUNICATION_FEATURE_NAMES
        )
    )

    for relation in [
        V2V_OFFLOADING_RELATION,
        V2I_OFFLOADING_RELATION,
    ]:

        if (
            graph[
                relation
            ].edge_attr.shape[1]
            != offload_width
        ):

            raise AssertionError(
                f"{path}: invalid "
                f"offloading feature "
                f"dimension for "
                f"{relation}."
            )

    if (
        graph[
            GANG_CONTENTION_RELATION
        ].edge_attr.shape[1]
        != len(
            CONTENTION_FEATURE_NAMES
        )
    ):

        raise AssertionError(
            f"{path}: invalid "
            f"Gang Resource-Contention "
            f"feature dimension."
        )

    # ========================================================
    # Finite values
    # ========================================================

    for node_type in (
        graph.node_types
    ):

        _check_finite(
            f"{path} "
            f"{node_type}.x",

            graph[
                node_type
            ].x,
        )

    for relation in (
        graph.edge_types
    ):

        _check_edge_bounds(
            graph,
            relation,
        )

        edge_attr = (
            graph[
                relation
            ].edge_attr
        )

        _check_finite(
            f"{path} "
            f"{relation}.edge_attr",

            edge_attr,
        )

        if (
            edge_attr.shape[0]
            != graph[
                relation
            ].num_edges
        ):

            raise AssertionError(
                f"{path}: edge "
                f"attribute count "
                f"mismatch."
            )

    # ========================================================
    # Local execution
    # ========================================================

    if (
        graph[
            LOCAL_EXECUTION_RELATION
        ].num_edges
        != num_tasks
    ):

        raise AssertionError(
            f"{path}: every task "
            f"must have exactly "
            f"one Local Execution edge."
        )

    local_index = (
        graph[
            LOCAL_EXECUTION_RELATION
        ].edge_index
    )

    source_vehicle = (
        graph[
            "task"
        ].source_vehicle
    )

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

        if (
            vehicle_id
            != int(
                source_vehicle[
                    task_id
                ]
            )
        ):

            raise AssertionError(
                f"{path}: Local "
                f"Execution edge "
                f"does not target "
                f"the source vehicle."
            )

    # ========================================================
    # V2V must exclude local source vehicle
    # ========================================================

    v2v_index = (
        graph[
            V2V_OFFLOADING_RELATION
        ].edge_index
    )

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

        if (
            vehicle_id
            == int(
                source_vehicle[
                    task_id
                ]
            )
        ):

            raise AssertionError(
                f"{path}: V2V "
                f"Offloading Layer "
                f"contains the "
                f"source vehicle."
            )

    # ========================================================
    # DAG validation
    # ========================================================

    precedence = (
        graph[
            DAG_DEPENDENCY_RELATION
        ].edge_index
    )

    dag = nx.DiGraph()

    dag.add_nodes_from(
        range(
            num_tasks
        )
    )

    dag.add_edges_from(
        precedence
        .t()
        .tolist()
    )

    if not (
        nx.is_directed_acyclic_graph(
            dag
        )
    ):

        raise AssertionError(
            f"{path}: DAG "
            f"Dependency Layer "
            f"contains a cycle."
        )

    # ========================================================
    # Event-driven state consistency
    # ========================================================

    released_index = (
        TASK_DYNAMIC_FEATURE_NAMES.index(
            "is_released"
        )
    )

    ready_index = (
        TASK_DYNAMIC_FEATURE_NAMES.index(
            "is_ready"
        )
    )

    running_index = (
        TASK_DYNAMIC_FEATURE_NAMES.index(
            "is_running"
        )
    )

    completed_index = (
        TASK_DYNAMIC_FEATURE_NAMES.index(
            "is_completed"
        )
    )

    task_x = (
        graph[
            "task"
        ].x
    )

    released = (
        task_x[
            :,
            released_index,
        ]
        > 0.5
    )

    ready = (
        task_x[
            :,
            ready_index,
        ]
        > 0.5
    )

    running = (
        task_x[
            :,
            running_index,
        ]
        > 0.5
    )

    completed = (
        task_x[
            :,
            completed_index,
        ]
        > 0.5
    )

    if torch.any(
        ready
        & running
    ):

        raise AssertionError(
            f"{path}: task cannot "
            f"be ready and running."
        )

    if torch.any(
        ready
        & completed
    ):

        raise AssertionError(
            f"{path}: task cannot "
            f"be ready and completed."
        )

    if torch.any(
        running
        & completed
    ):

        raise AssertionError(
            f"{path}: task cannot "
            f"be running and completed."
        )

    if torch.any(
        (
            ready
            | running
            | completed
        )
        &
        (
            ~released
        )
    ):

        raise AssertionError(
            f"{path}: active or "
            f"completed task must "
            f"already be released."
        )

    _check_contention_symmetry(
        graph,
        path,
    )


def main() -> None:

    if not (
        MANIFEST_PATH.exists()
    ):

        raise FileNotFoundError(
            "Dynamic dataset not found.\n"
            "Run:\n"
            "python -m "
            "muxgnn.graph."
            "generate_dynamic_dataset"
        )

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get(
            "snapshot_mode"
        )
        != "event_driven"
    ):

        raise AssertionError(
            "Manifest is not "
            "event-driven."
        )

    expected_counts = {

        "train":
            NUM_TRAIN_SCENARIOS,

        "val":
            NUM_VAL_SCENARIOS,

        "test":
            NUM_TEST_SCENARIOS,
    }

    total_graphs = 0
    strict_time_advance_seen = False

    for split, expected_count in (
        expected_counts.items()
    ):

        scenarios = (
            manifest[
                "splits"
            ][
                split
            ]
        )

        if (
            len(scenarios)
            != expected_count
        ):

            raise AssertionError(
                f"{split}: expected "
                f"{expected_count} "
                f"scenarios, got "
                f"{len(scenarios)}."
            )

        for scenario_name in (
            scenarios
        ):

            scenario_dir = (
                DYNAMIC_OUTPUT_DIR
                / split
                / scenario_name
            )

            metadata_path = (
                scenario_dir
                / "scenario.json"
            )

            if not (
                metadata_path.exists()
            ):

                raise AssertionError(
                    f"Missing "
                    f"{metadata_path}"
                )

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                metadata.get(
                    "snapshot_mode"
                )
                != "event_driven"
            ):

                raise AssertionError(
                    f"{scenario_name}: "
                    f"snapshot mode is "
                    f"not event-driven."
                )

            num_snapshots = int(
                metadata[
                    "num_snapshots"
                ]
            )

            if num_snapshots < 1:

                raise AssertionError(
                    f"{scenario_name}: "
                    f"no snapshots."
                )

            snapshots_dir = (
                scenario_dir
                / "snapshots"
            )

            snapshot_files = sorted(
                snapshots_dir.glob(
                    "snapshot_*.pt"
                )
            )

            if (
                len(snapshot_files)
                != num_snapshots
            ):

                raise AssertionError(
                    f"{scenario_name}: "
                    f"metadata and "
                    f"snapshot file "
                    f"count differ."
                )

            snapshot_rows = (
                metadata[
                    "snapshots"
                ]
            )

            if (
                len(snapshot_rows)
                != num_snapshots
            ):

                raise AssertionError(
                    f"{scenario_name}: "
                    f"snapshot metadata "
                    f"count mismatch."
                )

            previous_time = None

            previous_task_state = (
                None
            )

            previous_positions = (
                None
            )

            reference_precedence = (
                None
            )

            movement_detected = (
                False
            )

            for expected_index, (
                snapshot_path,
                snapshot_row,
            ) in enumerate(
                zip(
                    snapshot_files,
                    snapshot_rows,
                )
            ):

                if (
                    int(
                        snapshot_row[
                            "decision_epoch_index"
                        ]
                    )
                    != expected_index
                ):

                    raise AssertionError(
                        f"{scenario_name}: "
                        f"non-sequential "
                        f"decision epoch."
                    )

                trigger_event = str(
                    snapshot_row[
                        "trigger_event"
                    ]
                )

                if (
                    trigger_event
                    not in ALLOWED_TRIGGER_EVENTS
                ):

                    raise AssertionError(
                        f"{scenario_name}: "
                        f"unsupported event "
                        f"{trigger_event!r}."
                    )

                simulation_time = float(
                    snapshot_row[
                        "simulation_time"
                    ]
                )

                if (
                    previous_time
                    is not None
                ):

                    if (
                        simulation_time
                        < previous_time
                        - 1e-9
                    ):

                        raise AssertionError(
                            f"{scenario_name}: "
                            f"simulation time "
                            f"moved backward."
                        )

                    if (
                        simulation_time
                        > previous_time
                        + 1e-9
                    ):

                        strict_time_advance_seen = (
                            True
                        )

                graph = _load_graph(
                    snapshot_path
                )

                _validate_graph(
                    graph,
                    snapshot_path,
                )

                # Every saved snapshot must represent a new task state.
                current_task_state = (
                    _task_state_tensor(
                        graph
                    ).clone()
                )

                if (
                    previous_task_state
                    is not None
                    and torch.equal(
                        current_task_state,
                        previous_task_state,
                    )
                ):

                    raise AssertionError(
                        f"{scenario_name}: "
                        f"consecutive "
                        f"snapshots have "
                        f"identical task state."
                    )

                previous_task_state = (
                    current_task_state
                )

                # DAG must remain static.
                current_precedence = (
                    graph[
                        DAG_DEPENDENCY_RELATION
                    ].edge_index
                )

                if (
                    reference_precedence
                    is None
                ):

                    reference_precedence = (
                        current_precedence.clone()
                    )

                elif not torch.equal(
                    current_precedence,
                    reference_precedence,
                ):

                    raise AssertionError(
                        f"{scenario_name}: "
                        f"DAG dependency "
                        f"layer changed."
                    )

                positions = (
                    graph[
                        "vehicle"
                    ].pos
                )

                if (
                    previous_positions
                    is not None
                    and not torch.equal(
                        positions,
                        previous_positions,
                    )
                ):

                    movement_detected = (
                        True
                    )

                previous_positions = (
                    positions.clone()
                )

                previous_time = (
                    simulation_time
                )

                total_graphs += 1

            # Initial event must be release at t=0.
            first_row = (
                snapshot_rows[0]
            )

            if (
                first_row[
                    "trigger_event"
                ]
                != "initial_task_release"
            ):

                raise AssertionError(
                    f"{scenario_name}: "
                    f"first snapshot is "
                    f"not initial release."
                )

            if abs(
                float(
                    first_row[
                        "simulation_time"
                    ]
                )
            ) > 1e-9:

                raise AssertionError(
                    f"{scenario_name}: "
                    f"initial snapshot "
                    f"must be at t=0."
                )

            # Final snapshot must have completed DAG.
            final_graph = (
                _load_graph(
                    snapshot_files[-1]
                )
            )

            completed_index = (
                TASK_DYNAMIC_FEATURE_NAMES.index(
                    "is_completed"
                )
            )

            running_index = (
                TASK_DYNAMIC_FEATURE_NAMES.index(
                    "is_running"
                )
            )

            completed_count = int(
                (
                    final_graph[
                        "task"
                    ].x[
                        :,
                        completed_index,
                    ]
                    > 0.5
                ).sum()
            )

            if (
                completed_count
                != final_graph[
                    "task"
                ].num_nodes
            ):

                raise AssertionError(
                    f"{scenario_name}: "
                    f"final snapshot "
                    f"does not contain "
                    f"a completed DAG."
                )

            if torch.any(
                final_graph[
                    "task"
                ].x[
                    :,
                    running_index,
                ]
                > 0.5
            ):

                raise AssertionError(
                    f"{scenario_name}: "
                    f"final snapshot "
                    f"still has "
                    f"running tasks."
                )

            if (
                float(
                    metadata[
                        "final_simulation_time"
                    ]
                )
                > 0.0
                and not movement_detected
            ):

                raise AssertionError(
                    f"{scenario_name}: "
                    f"time advanced "
                    f"but no vehicle "
                    f"movement occurred."
                )

    if not (
        strict_time_advance_seen
    ):

        raise AssertionError(
            "No event caused "
            "simulation time to advance."
        )

    if (
        int(
            manifest[
                "total_snapshots"
            ]
        )
        != total_graphs
    ):

        raise AssertionError(
            "Manifest total snapshot "
            "count is incorrect."
        )

    print(
        "================================="
    )

    print(
        "Event-driven dynamic dataset "
        "validation PASSED"
    )

    print(
        "================================="
    )

    print(
        f"Total graphs: "
        f"{total_graphs}"
    )

    print(
        f"Train scenarios: "
        f"{NUM_TRAIN_SCENARIOS}"
    )

    print(
        f"Validation scenarios: "
        f"{NUM_VAL_SCENARIOS}"
    )

    print(
        f"Test scenarios: "
        f"{NUM_TEST_SCENARIOS}"
    )

    print(
        "Snapshots/scenario: "
        "variable (event-driven)"
    )


if __name__ == "__main__":
    main()