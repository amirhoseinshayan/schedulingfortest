from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

# Headless-safe backend for servers and SSH environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from torch_geometric.data import HeteroData

from .magnet_relation_layers import (
    DAG_DEPENDENCY_RELATION,
    GANG_CONTENTION_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2I_OFFLOADING_RELATION,
    V2V_OFFLOADING_RELATION,
)

from .settings import (
    GRAPH_PATH,
    OUTPUT_DIR,
)


# ============================================================
# Output paths
# ============================================================

FULL_GRAPH_IMAGE_PATH = (
    OUTPUT_DIR
    / "hetero_graph_full.png"
)

LAYERED_GRAPH_IMAGE_PATH = (
    OUTPUT_DIR
    / "hetero_graph_layers.png"
)


# ============================================================
# Visualization styles
# ============================================================

NODE_STYLES = {
    "task": {
        "color": "#4C78A8",
        "shape": "o",
        "size": 360,
        "label": "Task",
    },

    "vehicle": {
        "color": "#F58518",
        "shape": "s",
        "size": 700,
        "label": "Vehicle",
    },

    "edge_server": {
        "color": "#54A24B",
        "shape": "D",
        "size": 850,
        "label": "Edge Server",
    },
}


RELATION_STYLES = {
    DAG_DEPENDENCY_RELATION: {
        "color": "#2F4B7C",
        "width": 1.25,
        "alpha": 0.65,
        "style": "solid",
        "label": "DAG Dependency",
        "arrows": True,
    },

    LOCAL_EXECUTION_RELATION: {
        "color": "#2CA02C",
        "width": 1.0,
        "alpha": 0.30,
        "style": "solid",
        "label": "Local Execution",
        "arrows": True,
    },

    V2V_OFFLOADING_RELATION: {
        "color": "#FF8C00",
        "width": 0.9,
        "alpha": 0.22,
        "style": "dashed",
        "label": "V2V Offloading",
        "arrows": True,
    },

    V2I_OFFLOADING_RELATION: {
        "color": "#9467BD",
        "width": 1.0,
        "alpha": 0.28,
        "style": "dashdot",
        "label": "V2I/Edge Offloading",
        "arrows": True,
    },

    GANG_CONTENTION_RELATION: {
        "color": "#D62728",
        "width": 0.65,
        "alpha": 0.13,
        "style": "dotted",
        "label": "Gang Resource-Contention",
        "arrows": False,
    },
}


LAYER_ORDER = [
    DAG_DEPENDENCY_RELATION,
    LOCAL_EXECUTION_RELATION,
    V2V_OFFLOADING_RELATION,
    V2I_OFFLOADING_RELATION,
    GANG_CONTENTION_RELATION,
]


# ============================================================
# Node helpers
# ============================================================


def _node_key(
    node_type: str,
    node_id: int,
) -> str:
    """Create a unique visualization node key."""

    return (
        f"{node_type}:"
        f"{node_id}"
    )


def _node_label(
    node_type: str,
    node_id: int,
) -> str:
    """Create a short human-readable node label."""

    if node_type == "task":
        return f"T{node_id}"

    if node_type == "vehicle":
        return f"V{node_id}"

    if node_type == "edge_server":
        return f"E{node_id}"

    return (
        f"{node_type}_"
        f"{node_id}"
    )


def _all_node_keys(
    data: HeteroData,
) -> List[str]:
    """Return every node in the heterogeneous graph."""

    nodes: List[str] = []

    for node_type in (
        data.node_types
    ):

        for node_id in range(
            data[
                node_type
            ].num_nodes
        ):

            nodes.append(
                _node_key(
                    node_type,
                    node_id,
                )
            )

    return nodes


# ============================================================
# Edge helpers
# ============================================================


def _relation_edges(
    data: HeteroData,
    relation: Tuple[
        str,
        str,
        str,
    ],
) -> List[
    Tuple[str, str]
]:
    """Convert one PyG relation to visualization edges."""

    if relation not in (
        data.edge_types
    ):

        return []

    (
        source_type,
        _,
        target_type,
    ) = relation

    edge_index = (
        data[
            relation
        ].edge_index
    )

    edges: List[
        Tuple[str, str]
    ] = []

    for edge_id in range(
        edge_index.shape[1]
    ):

        source_id = int(
            edge_index[
                0,
                edge_id,
            ]
        )

        target_id = int(
            edge_index[
                1,
                edge_id,
            ]
        )

        edges.append(
            (
                _node_key(
                    source_type,
                    source_id,
                ),

                _node_key(
                    target_type,
                    target_id,
                ),
            )
        )

    return edges


def _relation_count(
    data: HeteroData,
    relation: Tuple[
        str,
        str,
        str,
    ],
) -> int:
    """Return edge count for one relation."""

    if relation not in (
        data.edge_types
    ):
        return 0

    return int(
        data[
            relation
        ].edge_index.shape[1]
    )


# ============================================================
# Layout
# ============================================================


def _task_dag_layout(
    data: HeteroData,
) -> Tuple[
    Dict[str, Tuple[float, float]],
    float,
]:
    """Create a deterministic topological task layout."""

    num_tasks = (
        data[
            "task"
        ].num_nodes
    )

    dag = nx.DiGraph()

    dag.add_nodes_from(
        range(
            num_tasks
        )
    )

    dependency_index = (
        data[
            DAG_DEPENDENCY_RELATION
        ].edge_index
    )

    dag.add_edges_from(
        dependency_index
        .t()
        .cpu()
        .tolist()
    )

    if not (
        nx.is_directed_acyclic_graph(
            dag
        )
    ):

        raise ValueError(
            "DAG Dependency Layer "
            "must be acyclic before "
            "visualization."
        )

    generations = list(
        nx.topological_generations(
            dag
        )
    )

    positions: Dict[
        str,
        Tuple[float, float],
    ] = {}

    max_depth = max(
        len(generations) - 1,
        1,
    )

    max_generation_size = max(
        (
            len(generation)
            for generation
            in generations
        ),
        default=1,
    )

    for depth, generation in (
        enumerate(
            generations
        )
    ):

        generation_nodes = sorted(
            list(
                generation
            )
        )

        count = len(
            generation_nodes
        )

        if count == 1:

            y_positions = [
                0.0
            ]

        else:

            vertical_span = max(
                8.0,
                float(
                    max_generation_size
                )
                * 1.1,
            )

            y_positions = (
                np.linspace(
                    vertical_span
                    / 2.0,
                    -vertical_span
                    / 2.0,
                    count,
                )
                .tolist()
            )

        for task_id, y_pos in zip(
            generation_nodes,
            y_positions,
        ):

            positions[
                _node_key(
                    "task",
                    int(task_id),
                )
            ] = (
                float(depth),
                float(y_pos),
            )

    return (
        positions,
        float(max_depth),
    )


def _build_global_layout(
    data: HeteroData,
) -> Dict[
    str,
    Tuple[float, float]
]:
    """Create one consistent layout for all five layers."""

    positions, max_depth = (
        _task_dag_layout(
            data
        )
    )

    task_y_values = [
        y
        for key, (_, y)
        in positions.items()
        if key.startswith(
            "task:"
        )
    ]

    if task_y_values:

        minimum_task_y = min(
            task_y_values
        )

    else:

        minimum_task_y = 0.0

    vehicle_y = (
        minimum_task_y
        - 4.5
    )

    edge_server_y = (
        vehicle_y
        - 4.0
    )

    # --------------------------------------------------------
    # Vehicle positions
    # --------------------------------------------------------

    num_vehicles = (
        data[
            "vehicle"
        ].num_nodes
    )

    if num_vehicles == 1:

        vehicle_x_positions = [
            max_depth / 2.0
        ]

    else:

        vehicle_x_positions = (
            np.linspace(
                0.0,
                max_depth,
                num_vehicles,
            )
            .tolist()
        )

    for vehicle_id, x_pos in (
        enumerate(
            vehicle_x_positions
        )
    ):

        positions[
            _node_key(
                "vehicle",
                vehicle_id,
            )
        ] = (
            float(x_pos),
            float(vehicle_y),
        )

    # --------------------------------------------------------
    # Edge server positions
    # --------------------------------------------------------

    num_edge_servers = (
        data[
            "edge_server"
        ].num_nodes
    )

    if num_edge_servers == 1:

        edge_x_positions = [
            max_depth / 2.0
        ]

    else:

        edge_x_positions = (
            np.linspace(
                0.0,
                max_depth,
                num_edge_servers,
            )
            .tolist()
        )

    for server_id, x_pos in (
        enumerate(
            edge_x_positions
        )
    ):

        positions[
            _node_key(
                "edge_server",
                server_id,
            )
        ] = (
            float(x_pos),
            float(edge_server_y),
        )

    return positions


# ============================================================
# Drawing primitives
# ============================================================


def _build_networkx_graph(
    data: HeteroData,
) -> nx.MultiDiGraph:
    """Create a NetworkX projection for drawing."""

    graph = nx.MultiDiGraph()

    for node_type in (
        data.node_types
    ):

        for node_id in range(
            data[
                node_type
            ].num_nodes
        ):

            graph.add_node(
                _node_key(
                    node_type,
                    node_id,
                ),

                node_type=
                    node_type,

                node_id=
                    node_id,
            )

    for relation in (
        LAYER_ORDER
    ):

        for source, target in (
            _relation_edges(
                data,
                relation,
            )
        ):

            graph.add_edge(
                source,
                target,
                relation=
                    relation,
            )

    return graph


def _draw_nodes(
    ax,
    graph: nx.MultiDiGraph,
    positions: Dict[
        str,
        Tuple[float, float]
    ],
    node_types: Iterable[str],
    node_size_scale: float = 1.0,
) -> None:
    """Draw heterogeneous node types."""

    requested_types = set(
        node_types
    )

    for node_type, style in (
        NODE_STYLES.items()
    ):

        if (
            node_type
            not in requested_types
        ):
            continue

        nodes = [
            node
            for node, attrs
            in graph.nodes(
                data=True
            )
            if (
                attrs[
                    "node_type"
                ]
                == node_type
            )
        ]

        if not nodes:
            continue

        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=
                nodes,
            node_color=
                style[
                    "color"
                ],
            node_shape=
                style[
                    "shape"
                ],
            node_size=
                style[
                    "size"
                ]
                * node_size_scale,
            edgecolors=
                "white",
            linewidths=
                0.8,
            alpha=
                0.95,
            ax=
                ax,
        )


def _draw_labels(
    ax,
    graph: nx.MultiDiGraph,
    positions: Dict[
        str,
        Tuple[float, float]
    ],
    node_types: Iterable[str],
    font_size: float,
) -> None:
    """Draw compact node labels."""

    requested_types = set(
        node_types
    )

    labels: Dict[
        str,
        str,
    ] = {}

    for node, attrs in (
        graph.nodes(
            data=True
        )
    ):

        node_type = attrs[
            "node_type"
        ]

        if (
            node_type
            not in requested_types
        ):
            continue

        labels[
            node
        ] = _node_label(
            node_type,
            int(
                attrs[
                    "node_id"
                ]
            ),
        )

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=
            labels,
        font_size=
            font_size,
        font_color=
            "#111111",
        font_weight=
            "normal",
        ax=
            ax,
    )


def _draw_relation(
    ax,
    graph: nx.MultiDiGraph,
    data: HeteroData,
    positions: Dict[
        str,
        Tuple[float, float]
    ],
    relation: Tuple[
        str,
        str,
        str,
    ],
    emphasize: bool = False,
) -> None:
    """Draw one MAGNET relation layer."""

    edges = _relation_edges(
        data,
        relation,
    )

    if not edges:
        return

    style = (
        RELATION_STYLES[
            relation
        ]
    )

    width = float(
        style[
            "width"
        ]
    )

    alpha = float(
        style[
            "alpha"
        ]
    )

    if emphasize:

        width *= 1.5

        alpha = min(
            alpha * 2.2,
            0.85,
        )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=
            edges,
        edge_color=
            style[
                "color"
            ],
        width=
            width,
        alpha=
            alpha,
        style=
            style[
                "style"
            ],
        arrows=
            bool(
                style[
                    "arrows"
                ]
            ),
        arrowsize=
            8
            if not emphasize
            else 11,
        arrowstyle=
            "-|>",
        connectionstyle=
            "arc3,rad=0.02",
        min_source_margin=
            4,
        min_target_margin=
            4,
        ax=
            ax,
    )


def _hide_axes(
    ax,
) -> None:
    """Apply a clean academic figure style."""

    ax.set_xticks(
        []
    )

    ax.set_yticks(
        []
    )

    for spine in (
        ax.spines.values()
    ):

        spine.set_visible(
            False
        )

    ax.set_facecolor(
        "white"
    )


# ============================================================
# Full graph visualization
# ============================================================


def save_full_graph_visualization(
    data: HeteroData,
    output_path: Path
    = FULL_GRAPH_IMAGE_PATH,
) -> Path:
    """Save one figure containing the complete graph."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph = (
        _build_networkx_graph(
            data
        )
    )

    positions = (
        _build_global_layout(
            data
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            24,
            16,
        )
    )

    # Draw weaker/dense relations first.
    draw_order = [
        GANG_CONTENTION_RELATION,
        V2V_OFFLOADING_RELATION,
        V2I_OFFLOADING_RELATION,
        LOCAL_EXECUTION_RELATION,
        DAG_DEPENDENCY_RELATION,
    ]

    for relation in (
        draw_order
    ):

        _draw_relation(
            ax=
                ax,
            graph=
                graph,
            data=
                data,
            positions=
                positions,
            relation=
                relation,
            emphasize=False,
        )

    _draw_nodes(
        ax=
            ax,
        graph=
            graph,
        positions=
            positions,
        node_types=[
            "task",
            "vehicle",
            "edge_server",
        ],
        node_size_scale=
            1.0,
    )

    _draw_labels(
        ax=
            ax,
        graph=
            graph,
        positions=
            positions,
        node_types=[
            "task",
            "vehicle",
            "edge_server",
        ],
        font_size=
            6.5,
    )

    total_edges = sum(
        _relation_count(
            data,
            relation,
        )

        for relation
        in LAYER_ORDER
    )

    ax.set_title(
        (
            "MAGNET Static Heterogeneous Graph\n"
            f"{data['task'].num_nodes} Tasks | "
            f"{data['vehicle'].num_nodes} Vehicles | "
            f"{data['edge_server'].num_nodes} Edge Server(s) | "
            f"{total_edges} Relation Edges"
        ),
        fontsize=
            17,
        fontweight=
            "bold",
        pad=
            18,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_items = [
        Patch(
            facecolor=
                NODE_STYLES[
                    "task"
                ][
                    "color"
                ],
            edgecolor=
                "white",
            label=
                "Task",
        ),

        Patch(
            facecolor=
                NODE_STYLES[
                    "vehicle"
                ][
                    "color"
                ],
            edgecolor=
                "white",
            label=
                "Vehicle",
        ),

        Patch(
            facecolor=
                NODE_STYLES[
                    "edge_server"
                ][
                    "color"
                ],
            edgecolor=
                "white",
            label=
                "Edge Server",
        ),
    ]

    for relation in (
        LAYER_ORDER
    ):

        style = (
            RELATION_STYLES[
                relation
            ]
        )

        legend_items.append(
            Line2D(
                [0],
                [0],
                color=
                    style[
                        "color"
                    ],
                linewidth=
                    2.0,
                linestyle=
                    style[
                        "style"
                    ],
                label=(
                    f"{style['label']} "
                    f"({_relation_count(data, relation)} edges)"
                ),
            )
        )

    ax.legend(
        handles=
            legend_items,
        loc=
            "upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
        fontsize=
            9,
        frameon=
            True,
        title=
            "Nodes and Relation Layers",
        title_fontsize=
            10,
    )

    _hide_axes(
        ax
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=
            300,
        bbox_inches=
            "tight",
        facecolor=
            "white",
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# Five-layer visualization
# ============================================================


def _layer_node_types(
    relation: Tuple[
        str,
        str,
        str,
    ],
) -> List[str]:
    """Return relevant node types for one relation."""

    if (
        relation
        == DAG_DEPENDENCY_RELATION
    ):

        return [
            "task",
        ]

    if (
        relation
        == GANG_CONTENTION_RELATION
    ):

        return [
            "task",
        ]

    if (
        relation
        == LOCAL_EXECUTION_RELATION
    ):

        return [
            "task",
            "vehicle",
        ]

    if (
        relation
        == V2V_OFFLOADING_RELATION
    ):

        return [
            "task",
            "vehicle",
        ]

    if (
        relation
        == V2I_OFFLOADING_RELATION
    ):

        return [
            "task",
            "edge_server",
        ]

    return [
        "task",
        "vehicle",
        "edge_server",
    ]


def save_layered_graph_visualization(
    data: HeteroData,
    output_path: Path
    = LAYERED_GRAPH_IMAGE_PATH,
) -> Path:
    """Save five separate relation views in one image."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph = (
        _build_networkx_graph(
            data
        )
    )

    positions = (
        _build_global_layout(
            data
        )
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(
            24,
            25,
        ),
    )

    axes_flat = list(
        axes.flatten()
    )

    layer_titles = {
        DAG_DEPENDENCY_RELATION:
            (
                "1. DAG Dependency Layer\n"
                "Task precedence and data dependencies"
            ),

        LOCAL_EXECUTION_RELATION:
            (
                "2. Local Execution Layer\n"
                "Gang tasks connected to their source vehicle"
            ),

        V2V_OFFLOADING_RELATION:
            (
                "3. V2V Offloading Layer\n"
                "Gang tasks connected to reachable neighboring vehicles"
            ),

        V2I_OFFLOADING_RELATION:
            (
                "4. V2I / Edge Offloading Layer\n"
                "Gang tasks connected to reachable edge servers"
            ),

        GANG_CONTENTION_RELATION:
            (
                "5. Gang Resource-Contention Layer\n"
                "Competition between simultaneously executable gang tasks"
            ),
    }

    for ax, relation in zip(
        axes_flat,
        LAYER_ORDER,
    ):

        node_types = (
            _layer_node_types(
                relation
            )
        )

        _draw_relation(
            ax=
                ax,
            graph=
                graph,
            data=
                data,
            positions=
                positions,
            relation=
                relation,
            emphasize=
                True,
        )

        _draw_nodes(
            ax=
                ax,
            graph=
                graph,
            positions=
                positions,
            node_types=
                node_types,
            node_size_scale=
                0.82,
        )

        _draw_labels(
            ax=
                ax,
            graph=
                graph,
            positions=
                positions,
            node_types=
                node_types,
            font_size=
                5.5,
        )

        edge_count = (
            _relation_count(
                data,
                relation,
            )
        )

        ax.set_title(
            (
                f"{layer_titles[relation]}\n"
                f"Edges: {edge_count}"
            ),
            fontsize=
                12,
            fontweight=
                "bold",
            pad=
                10,
        )

        _hide_axes(
            ax
        )

    # --------------------------------------------------------
    # Final panel: schema summary
    # --------------------------------------------------------

    summary_ax = (
        axes_flat[-1]
    )

    summary_ax.axis(
        "off"
    )

    summary_lines = [
        "MAGNET Heterogeneous Graph Schema",
        "",
        (
            f"Task nodes: "
            f"{data['task'].num_nodes}"
        ),
        (
            f"Vehicle nodes: "
            f"{data['vehicle'].num_nodes}"
        ),
        (
            f"Edge-server nodes: "
            f"{data['edge_server'].num_nodes}"
        ),
        "",
        "Relation layers:",
    ]

    for index, relation in (
        enumerate(
            LAYER_ORDER,
            start=1,
        )
    ):

        summary_lines.append(
            (
                f"{index}. "
                f"{RELATION_STYLES[relation]['label']}"
                f" — "
                f"{_relation_count(data, relation)} edges"
            )
        )

    summary_lines.extend(
        [
            "",
            "Node labels:",
            "T = Task",
            "V = Vehicle",
            "E = Edge Server",
            "",
            (
                "This figure is generated directly "
                "from the saved static HeteroData graph."
            ),
        ]
    )

    summary_ax.text(
        0.05,
        0.95,
        "\n".join(
            summary_lines
        ),
        va=
            "top",
        ha=
            "left",
        fontsize=
            13,
        linespacing=
            1.5,
        transform=
            summary_ax.transAxes,
    )

    fig.suptitle(
        (
            "MAGNET Static Heterogeneous Graph — "
            "Relation-Layer Visualization"
        ),
        fontsize=
            19,
        fontweight=
            "bold",
        y=
            0.995,
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.985,
        ]
    )

    fig.savefig(
        output_path,
        dpi=
            300,
        bbox_inches=
            "tight",
        facecolor=
            "white",
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# Public API
# ============================================================


def save_static_graph_visualizations(
    data: HeteroData,
) -> Tuple[
    Path,
    Path,
]:
    """Generate both static graph visualizations."""

    full_path = (
        save_full_graph_visualization(
            data
        )
    )

    layered_path = (
        save_layered_graph_visualization(
            data
        )
    )

    return (
        full_path,
        layered_path,
    )


# ============================================================
# Standalone execution
# ============================================================


def _load_saved_static_graph() -> HeteroData:
    """Load the previously generated static graph."""

    if not GRAPH_PATH.exists():

        raise FileNotFoundError(
            "Static graph not found.\n"
            "Run:\n"
            "python -m "
            "muxgnn.graph.generate"
        )

    try:

        return torch.load(
            GRAPH_PATH,
            map_location=
                "cpu",
            weights_only=
                False,
        )

    except TypeError:

        return torch.load(
            GRAPH_PATH,
            map_location=
                "cpu",
        )


def main() -> None:
    """Generate visualizations from the saved static graph."""

    data = (
        _load_saved_static_graph()
    )

    (
        full_path,
        layered_path,
    ) = (
        save_static_graph_visualizations(
            data
        )
    )

    print(
        "================================="
    )

    print(
        "Static graph visualization "
        "completed"
    )

    print(
        "================================="
    )

    print(
        f"Full graph : "
        f"{full_path}"
    )

    print(
        f"Layer view : "
        f"{layered_path}"
    )


if __name__ == "__main__":
    main()