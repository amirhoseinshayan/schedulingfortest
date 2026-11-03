from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from .builder import (
    CANDIDATE_EDGE_FEATURE_NAMES,
    CONTENTION_EDGE_FEATURE_NAMES,
    build_hetero_graph,
)
from .magnet_relation_layers import (
    DAG_DEPENDENCY_RELATION,
    GANG_CONTENTION_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2I_OFFLOADING_RELATION,
    V2V_OFFLOADING_RELATION,
)
from .settings import (
    GRAPH_PATH,
    METADATA_PATH,
    OUTPUT_DIR,
    README_PATH,
    TABLE_DIR,
)
from .static_graph_visualization import save_static_graph_visualizations




def _open_visualization(image_path: Path) -> None:
    """Open the generated relation-layer image in the OS default image viewer.

    Dataset generation must also remain safe on headless servers, so failure to
    open a GUI viewer never makes generation fail.  On Windows (the primary
    development environment) this opens the PNG automatically after `prepare`.
    """

    resolved = image_path.resolve()

    try:
        if sys.platform.startswith("win"):
            os.startfile(str(resolved))  # type: ignore[attr-defined]
            return

        if sys.platform == "darwin":
            subprocess.Popen(
                ["open", str(resolved)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            subprocess.Popen(
                ["xdg-open", str(resolved)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        print("Image display skipped: no desktop display was detected.")

    except Exception as exc:
        # The PNG has already been generated successfully.  GUI opening is a
        # convenience only and must never invalidate the dataset build.
        print(f"Image display skipped: {exc}")


def _to_list(tensor: torch.Tensor):
    return tensor.detach().cpu().tolist()


def _save_node_tables(graph, metadata) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    num_tasks = int(graph["task"].num_nodes)
    task_rows = []
    benchmark_names = metadata["benchmark_names"]
    for task_id in range(num_tasks):
        task_rows.append(
            {
                "task_id": task_id,
                "benchmark_name": benchmark_names[task_id],
                "topological_layer": int(graph["task"].topological_layer[task_id]),
                "gang_size": int(graph["task"].gang_size[task_id]),
                "deadline": float(graph["task"].deadline[task_id]),
                "exec_time_a7": float(graph["task"].exec_time_a7[task_id]),
                "exec_time_a12": float(graph["task"].exec_time_a12[task_id]),
                "energy_a7": float(graph["task"].energy_a7[task_id]),
                "energy_a12": float(graph["task"].energy_a12[task_id]),
                "source_vehicle": int(graph["task"].source_vehicle[task_id]),
                "is_ready": int(graph["task"].is_ready[task_id]),
                "is_running": int(graph["task"].is_running[task_id]),
                "is_completed": int(graph["task"].is_completed[task_id]),
            }
        )
    pd.DataFrame(task_rows).to_csv(TABLE_DIR / "task_nodes.csv", index=False)

    vehicle_rows = []
    for node_id in range(int(graph["vehicle"].num_nodes)):
        vehicle_rows.append(
            {
                "vehicle_id": node_id,
                "total_a7_cores": int(graph["vehicle"].total_a7_cores[node_id]),
                "total_a12_cores": int(graph["vehicle"].total_a12_cores[node_id]),
                "available_a7_cores": int(graph["vehicle"].available_a7_cores[node_id]),
                "available_a12_cores": int(graph["vehicle"].available_a12_cores[node_id]),
                "is_source_vehicle": int(node_id == metadata["resource_model"]["source_vehicle_id"]),
            }
        )
    pd.DataFrame(vehicle_rows).to_csv(TABLE_DIR / "vehicle_nodes.csv", index=False)

    edge_rows = []
    for node_id in range(int(graph["edge_server"].num_nodes)):
        edge_rows.append(
            {
                "edge_server_id": node_id,
                "total_a7_cores": int(graph["edge_server"].total_a7_cores[node_id]),
                "total_a12_cores": int(graph["edge_server"].total_a12_cores[node_id]),
                "available_a7_cores": int(graph["edge_server"].available_a7_cores[node_id]),
                "available_a12_cores": int(graph["edge_server"].available_a12_cores[node_id]),
            }
        )
    pd.DataFrame(edge_rows).to_csv(TABLE_DIR / "edge_server_nodes.csv", index=False)


def _relation_frame(graph, relation, feature_names, use_raw=False):
    edge_index = graph[relation].edge_index.detach().cpu()
    rows = []

    if use_raw:
        attrs = graph[relation].edge_attr_raw.detach().cpu()
    else:
        attrs = graph[relation].edge_attr.detach().cpu()

    for edge_id in range(edge_index.shape[1]):
        row = {
            "source": int(edge_index[0, edge_id]),
            "target": int(edge_index[1, edge_id]),
        }
        for feature_id, name in enumerate(feature_names):
            row[name] = float(attrs[edge_id, feature_id])
        rows.append(row)

    return pd.DataFrame(rows)


def _save_edge_tables(graph) -> None:
    dep = graph[DAG_DEPENDENCY_RELATION].edge_index.detach().cpu()
    pd.DataFrame(
        {
            "source_task": _to_list(dep[0]),
            "target_task": _to_list(dep[1]),
        }
    ).to_csv(TABLE_DIR / "dependency_edges.csv", index=False)

    _relation_frame(
        graph,
        LOCAL_EXECUTION_RELATION,
        CANDIDATE_EDGE_FEATURE_NAMES,
        use_raw=True,
    ).rename(columns={"source": "task_id", "target": "vehicle_id"}).to_csv(
        TABLE_DIR / "local_execution_edges.csv", index=False
    )

    _relation_frame(
        graph,
        V2V_OFFLOADING_RELATION,
        CANDIDATE_EDGE_FEATURE_NAMES,
        use_raw=True,
    ).rename(columns={"source": "task_id", "target": "vehicle_id"}).to_csv(
        TABLE_DIR / "v2v_offloading_edges.csv", index=False
    )

    _relation_frame(
        graph,
        V2I_OFFLOADING_RELATION,
        CANDIDATE_EDGE_FEATURE_NAMES,
        use_raw=True,
    ).rename(columns={"source": "task_id", "target": "edge_server_id"}).to_csv(
        TABLE_DIR / "v2i_offloading_edges.csv", index=False
    )

    _relation_frame(
        graph,
        GANG_CONTENTION_RELATION,
        CONTENTION_EDGE_FEATURE_NAMES,
        use_raw=False,
    ).rename(columns={"source": "source_task", "target": "target_task"}).to_csv(
        TABLE_DIR / "contention_edges.csv", index=False
    )


def _write_dataset_readme(metadata) -> None:
    counts = metadata["graph"]["relation_edge_counts"]
    text = f"""# Legacy-native Simple MAGNET Heterogeneous Graph v2

This folder is the new base muxGNN dataset.  It is built only from the validated
Legacy 60-task DAG and its A7/A12/Gang/Deadline profiles.  DAGBench is not used.

## Graph schema

- Task nodes: {metadata['graph']['num_tasks']}
- Vehicle nodes: {metadata['graph']['num_vehicles']}
- Edge-server nodes: {metadata['graph']['num_edge_servers']}
- DAG precedence edges: {counts['precedes']}
- Local task->vehicle edges: {counts['local_execute_on']}
- V2V compute-candidate edges: {counts['v2v_offload_to']}
- V2I compute-candidate edges: {counts['v2i_offload_to']}
- Initial gang-contention edges: {counts['contends_with']}

## Important v2 boundary

This v2 graph is still intentionally compute-only.  It does not claim a physical
vehicular-network model.  Mobility, distance, link rate, transmission delay,
contact duration, queue delay, background load, and migration are intentionally
omitted and will be added in later versions after this graph contract is tested.

The Task->Task precedence relation has no edge features.  Task-resource edges
contain only compute/gang feasibility features derived from the validated Legacy
profiles and controlled resource states.  Vehicle availability is heterogeneous,
V2V candidates are limited to the controlled reachable subset, and a task-resource
edge is omitted when neither A7 nor A12 can fit the gang.

## Rebuild

From `scheduling_utils`:

```bash
python -m muxgnn.graph.generate
python -m muxgnn.graph.validate
```

The human-readable tables in `tables/` are audit artifacts.  `hetero_graph.pt`
is the PyTorch-Geometric derivative used by later muxGNN stages.
"""
    README_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    """Generate Legacy-native Simple MAGNET Heterogeneous Graph v2."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graph, metadata = build_hetero_graph()

    torch.save(graph, GRAPH_PATH)

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _save_node_tables(graph, metadata)
    _save_edge_tables(graph)
    _write_dataset_readme(metadata)

    full_graph_image, layered_graph_image = save_static_graph_visualizations(graph)

    print("============================================")
    print("Legacy-native Simple MAGNET v2 generated")
    print("============================================")
    print(f"Graph    : {GRAPH_PATH}")
    print(f"Metadata : {METADATA_PATH}")
    print(f"Tables   : {TABLE_DIR}")
    print(f"Full view: {full_graph_image}")
    print(f"Layers   : {layered_graph_image}")

    # The relation-layer figure is the primary human-inspection view: it shows
    # all five heterogeneous relation layers plus the graph schema in one PNG.
    # Open it automatically on desktop systems after every dataset build.
    _open_visualization(layered_graph_image)

    print("\nNode types:")
    for node_type in graph.node_types:
        print(f"  {node_type}: {graph[node_type].num_nodes}")

    print("\nRelation layers:")
    for relation in graph.edge_types:
        print(f"  {relation}: {graph[relation].num_edges}")


if __name__ == "__main__":
    main()
