from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def _topological_levels(graph: nx.DiGraph) -> Dict[int, int]:
    levels: Dict[int, int] = {}
    for node in nx.topological_sort(graph):
        levels[int(node)] = max((levels[int(pred)] + 1 for pred in graph.predecessors(node)), default=0)
    return levels


def _max_vf_row(task: dict) -> dict:
    return max(task["v_f_levels"], key=lambda item: float(item["frequency_proc1"]))


def build_dag_detail_rows(
    dag_index: int,
    graph: nx.DiGraph,
    characteristics: Sequence[dict],
    schedule,
    edge_prob: float,
) -> list[dict]:
    """Flatten one generated DAG into an audit-friendly one-row-per-task table."""
    levels = _topological_levels(graph)
    topo = list(nx.topological_sort(graph))
    topo_index = {int(node): i for i, node in enumerate(topo)}
    rows: list[dict] = []

    for task_id, task in enumerate(characteristics):
        a7_count, a12_count = schedule.allocation[task_id]
        core_ids = list(schedule.cores[task_id])
        start = float(schedule.start[task_id])
        finish = float(schedule.finish[task_id])
        deadline = float(task["implicit_deadline"])
        row = {
            "dag_index": int(dag_index),
            "task_id": int(task_id),
            "topological_order": int(topo_index[task_id]),
            "topological_level": int(levels[task_id]),
            "benchmark_name": task["benchmark_name"],
            "period": float(task["period"]),
            "m_i": int(task["m_i"]),
            "implicit_deadline_sec": deadline,
            "predecessors": "[" + ",".join(str(int(x)) for x in sorted(graph.predecessors(task_id))) + "]",
            "successors": "[" + ",".join(str(int(x)) for x in sorted(graph.successors(task_id))) + "]",
            "in_degree": int(graph.in_degree(task_id)),
            "out_degree": int(graph.out_degree(task_id)),
            "reference_start_sec": start,
            "reference_finish_sec": finish,
            "reference_duration_sec": finish - start,
            "deadline_slack_after_reference_sec": deadline - finish,
            "reference_a7_core_count": int(a7_count),
            "reference_a12_core_count": int(a12_count),
            "reference_core_ids": "[" + ",".join(str(int(x)) for x in core_ids) + "]",
            "num_subtasks": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
            "edge_probability": float(edge_prob),
            "reference_makespan_sec": float(schedule.makespan),
        }

        # Keep every DVFS attribute from the dataset visible as ordinary CSV columns.
        for vf in sorted(task["v_f_levels"], key=lambda item: int(item["level"])):
            level = int(vf["level"])
            prefix = f"vf{level}"
            row.update(
                {
                    f"{prefix}_a12_voltage": float(vf["voltage_proc1"]),
                    f"{prefix}_a12_frequency": float(vf["frequency_proc1"]),
                    f"{prefix}_a12_execution_time_sec": float(vf["utilization_proc1_vf"]),
                    f"{prefix}_a12_avg_total_power": float(vf["avg_total_power_proc1"]),
                    f"{prefix}_a7_voltage": float(vf["voltage_proc2"]),
                    f"{prefix}_a7_frequency": float(vf["frequency_proc2"]),
                    f"{prefix}_a7_execution_time_sec": float(vf["utilization_proc2_vf"]),
                    f"{prefix}_a7_avg_total_power": float(vf["avg_total_power_proc2"]),
                }
            )
        rows.append(row)
    return rows


def write_dag_detail_csv(
    output_path: Path,
    dag_index: int,
    graph: nx.DiGraph,
    characteristics: Sequence[dict],
    schedule,
    edge_prob: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dag_detail_rows(dag_index, graph, characteristics, schedule, edge_prob)
    pd.DataFrame(rows).sort_values("task_id").to_csv(output_path, index=False)


def write_dag_image(
    output_path: Path,
    dag_index: int,
    graph: nx.DiGraph,
    characteristics: Sequence[dict],
    schedule,
    edge_prob: float,
) -> None:
    """Render a high-resolution, zoomable DAG inspection PNG without Graphviz.

    The graph itself shows all directed dependencies.  Each node box includes the
    most important scheduling values; the companion CSV contains every DVFS field.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    levels = _topological_levels(graph)
    layers: Dict[int, list[int]] = {}
    for node, level in levels.items():
        layers.setdefault(level, []).append(node)
    for nodes in layers.values():
        nodes.sort()

    max_layer = max(layers, default=0)
    max_nodes_in_layer = max((len(nodes) for nodes in layers.values()), default=1)
    x_step = 3.6
    y_step = 2.9
    pos: Dict[int, tuple[float, float]] = {}
    for level in range(max_layer + 1):
        nodes = layers.get(level, [])
        center = (len(nodes) - 1) / 2.0
        for idx, node in enumerate(nodes):
            pos[node] = (level * x_step, (center - idx) * y_step)

    fig_width = max(40.0, (max_layer + 1) * 3.3)
    fig_height = max(26.0, max_nodes_in_layer * 4.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=8,
        width=0.45,
        alpha=0.22,
        min_source_margin=6,
        min_target_margin=6,
    )

    for task_id, task in enumerate(characteristics):
        max_vf = _max_vf_row(task)
        a7_count, a12_count = schedule.allocation[task_id]
        label = (
            f"T{task_id:02d} | {task['benchmark_name']}\n"
            f"m_i={int(task['m_i'])} | deadline={float(task['implicit_deadline']):.2f}s\n"
            f"REF {float(schedule.start[task_id]):.2f} -> {float(schedule.finish[task_id]):.2f}s\n"
            f"alloc={a7_count}xA7 + {a12_count}xA12 | cores={','.join(map(str, schedule.cores[task_id]))}\n"
            f"A12(maxVF): t={float(max_vf['utilization_proc1_vf']):.2f}s, P={float(max_vf['avg_total_power_proc1']):.4f}\n"
            f"A7(maxVF):  t={float(max_vf['utilization_proc2_vf']):.2f}s, P={float(max_vf['avg_total_power_proc2']):.4f}\n"
            f"in={graph.in_degree(task_id)} | out={graph.out_degree(task_id)} | level={levels[task_id]}"
        )
        x, y = pos[task_id]
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=5.4,
            bbox={"boxstyle": "round,pad=0.28", "alpha": 0.96},
            zorder=3,
        )

    ax.set_title(
        f"Generated Legacy GAT/PPO DAG {dag_index} | "
        f"tasks={graph.number_of_nodes()} | edges={graph.number_of_edges()} | "
        f"p={edge_prob:.6f} | reference makespan={schedule.makespan:.2f}s\n"
        "Directed arrows are precedence constraints. Node labels show max-VF A7/A12 profiles, "
        "gang width, absolute deadline, and reference schedule.",
        fontsize=12,
        pad=18,
    )
    ax.set_axis_off()
    ax.margins(0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_dag_inspection_artifacts(
    output_dir: Path,
    dag_index: int,
    graph: nx.DiGraph,
    characteristics: Sequence[dict],
    schedule,
    edge_prob: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"dag_{dag_index:04d}_full_details.csv"
    png_path = output_dir / f"dag_{dag_index:04d}_detailed.png"
    write_dag_detail_csv(csv_path, dag_index, graph, characteristics, schedule, edge_prob)
    write_dag_image(png_path, dag_index, graph, characteristics, schedule, edge_prob)
    project_root = output_dir.parents[2]
    return {
        "details_csv": str(csv_path.relative_to(project_root)),
        "detailed_png": str(png_path.relative_to(project_root)),
    }
