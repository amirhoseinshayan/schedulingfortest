from __future__ import annotations

import json
import shutil

from collections import Counter

from typing import Dict, List

import torch

from .dynamic_builder import (
    build_dynamic_hetero_graph,
)

from .dynamic_settings import (
    BASE_SEED,
    DYNAMIC_OUTPUT_DIR,
    MANIFEST_PATH,
    NUM_TEST_SCENARIOS,
    NUM_TRAIN_SCENARIOS,
    NUM_VAL_SCENARIOS,
)

from .event_driven_simulation import (
    EventDrivenScenarioSimulator,
)

from .magnet_relation_layers import (
    GANG_CONTENTION_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2I_OFFLOADING_RELATION,
    V2V_OFFLOADING_RELATION,
)


def _scenario_name(
    scenario_id: int,
) -> str:

    return (
        f"scenario_"
        f"{scenario_id:04d}"
    )


def _save_scenario(
    split: str,
    scenario_id: int,
    scenario_seed: int,
) -> Dict[
    str,
    object,
]:

    scenario_name = (
        _scenario_name(
            scenario_id
        )
    )

    scenario_dir = (
        DYNAMIC_OUTPUT_DIR
        / split
        / scenario_name
    )

    snapshots_dir = (
        scenario_dir
        / "snapshots"
    )

    snapshots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    simulator = (
        EventDrivenScenarioSimulator(
            seed=scenario_seed
        )
    )

    snapshot_metadata: List[
        Dict[
            str,
            object,
        ]
    ] = []

    event_counts: Counter[
        str
    ] = Counter()

    for epoch in (
        simulator.iter_decision_epochs()
    ):

        graph, metadata = (
            build_dynamic_hetero_graph(
                state=(
                    simulator.state
                ),

                decision_epoch_index=(
                    epoch.index
                ),

                event_context=(
                    epoch.to_metadata()
                ),
            )
        )

        snapshot_path = (
            snapshots_dir
            / (
                f"snapshot_"
                f"{epoch.index:04d}"
                f".pt"
            )
        )

        torch.save(
            graph,
            snapshot_path,
        )

        event_counts[
            epoch.trigger_event
        ] += 1

        snapshot_metadata.append(
            {
                "decision_epoch_index":
                    epoch.index,

                "simulation_time":
                    epoch.simulation_time,

                "trigger_event":
                    epoch.trigger_event,

                "affected_tasks":
                    list(
                        epoch.affected_tasks
                    ),

                "started_task":
                    epoch.started_task,

                "completed_tasks":
                    list(
                        epoch.completed_tasks
                    ),

                "released_tasks":
                    list(
                        epoch.released_tasks
                    ),

                "file":
                    snapshot_path.name,

                "num_released_tasks":
                    metadata[
                        "num_released_tasks"
                    ],

                "num_ready_tasks":
                    metadata[
                        "num_ready_tasks"
                    ],

                "num_running_tasks":
                    metadata[
                        "num_running_tasks"
                    ],

                "num_completed_tasks":
                    metadata[
                        "num_completed_tasks"
                    ],

                "num_local_edges":
                    graph[
                        LOCAL_EXECUTION_RELATION
                    ].num_edges,

                "num_v2v_offload_edges":
                    graph[
                        V2V_OFFLOADING_RELATION
                    ].num_edges,

                "num_v2i_offload_edges":
                    graph[
                        V2I_OFFLOADING_RELATION
                    ].num_edges,

                "num_contention_edges":
                    graph[
                        GANG_CONTENTION_RELATION
                    ].num_edges,
            }
        )

    if not snapshot_metadata:

        raise RuntimeError(
            f"{scenario_name} "
            f"produced no snapshots."
        )

    scenario_metadata: Dict[
        str,
        object,
    ] = {

        "scenario_id":
            scenario_id,

        "scenario_name":
            scenario_name,

        "split":
            split,

        "seed":
            scenario_seed,

        "snapshot_mode":
            "event_driven",

        "num_snapshots":
            len(
                snapshot_metadata
            ),

        "final_simulation_time":
            float(
                simulator.state.simulation_time
            ),

        "event_counts":
            dict(
                event_counts
            ),

        "snapshots":
            snapshot_metadata,
    }

    (
        scenario_dir
        / "scenario.json"
    ).write_text(
        json.dumps(
            scenario_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        scenario_metadata
    )


def main() -> None:

    if (
        DYNAMIC_OUTPUT_DIR.exists()
    ):

        print(
            "Removing old dynamic "
            "dataset..."
        )

        shutil.rmtree(
            DYNAMIC_OUTPUT_DIR
        )

    DYNAMIC_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest: Dict[
        str,
        object,
    ] = {

        "version":
            3,

        "description":
            (
                "Event-driven five-layer "
                "MAGNET heterogeneous "
                "graph dataset"
            ),

        "snapshot_mode":
            "event_driven",

        "decision_epoch_rule":
            (
                "A snapshot is saved only "
                "when a decision-relevant "
                "simulation event changes "
                "task/resource state."
            ),

        "relation_layers": [
            "DAG Dependency Layer",
            "Local Execution Layer",
            "V2V Offloading Layer",
            "V2I/Edge Offloading Layer",
            "Gang Resource-Contention Layer",
        ],

        "splits": {
            "train": [],
            "val": [],
            "test": [],
        },

        "split_snapshot_counts": {
            "train": 0,
            "val": 0,
            "test": 0,
        },
    }

    global_id = 0
    total_snapshots = 0

    split_definitions = [
        (
            "train",
            NUM_TRAIN_SCENARIOS,
        ),

        (
            "val",
            NUM_VAL_SCENARIOS,
        ),

        (
            "test",
            NUM_TEST_SCENARIOS,
        ),
    ]

    for split, count in (
        split_definitions
    ):

        print(
            f"\n=== Generating "
            f"{split} split ==="
        )

        for _ in range(
            count
        ):

            scenario_seed = (
                BASE_SEED
                + global_id
                * 1009
            )

            metadata = (
                _save_scenario(
                    split=split,

                    scenario_id=(
                        global_id
                    ),

                    scenario_seed=(
                        scenario_seed
                    ),
                )
            )

            manifest[
                "splits"
            ][
                split
            ].append(
                metadata[
                    "scenario_name"
                ]
            )

            scenario_snapshots = int(
                metadata[
                    "num_snapshots"
                ]
            )

            manifest[
                "split_snapshot_counts"
            ][
                split
            ] += (
                scenario_snapshots
            )

            total_snapshots += (
                scenario_snapshots
            )

            print(
                f"Created "
                f"{metadata['scenario_name']} "
                f"(seed={scenario_seed}, "
                f"snapshots="
                f"{scenario_snapshots}, "
                f"final_t="
                f"{metadata['final_simulation_time']:.6f}s)"
            )

            global_id += 1

    manifest[
        "total_scenarios"
    ] = global_id

    manifest[
        "total_snapshots"
    ] = total_snapshots

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n================================="
    )

    print(
        "Event-driven dynamic dataset "
        "generation completed"
    )

    print(
        "================================="
    )

    print(
        f"Output: "
        f"{DYNAMIC_OUTPUT_DIR}"
    )

    print(
        f"Scenarios: "
        f"{global_id}"
    )

    print(
        f"Snapshots: "
        f"{total_snapshots}"
    )

    print(
        "Snapshots/scenario: variable "
        "(event-driven)"
    )


if __name__ == "__main__":
    main()