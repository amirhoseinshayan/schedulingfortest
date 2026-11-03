from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
for path in (PROJECT_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from legacy_gat.config import (
    ALLOW_MIXED_GANGS,
    NUM_A12_CORES,
    NUM_A7_CORES,
    RUNTIME_MODEL,
)

from gatppo_config import (
    DEADLINE_FACTOR,
    EDGE_PROB_RANGES,
    LEGACY_METADATA_JSON,
    LEGACY_OUTPUT_CSV,
    LEGACY_PROFILE_REPORT_CSV,
    LEGACY_DAG_INSPECTION_DIR,
    M_I_BINOMIAL_WEIGHTS,
    M_I_DISTRIBUTION_NAME,
    M_I_VALUES,
    NUM_DAGS,
    NUM_SUBTASKS,
    NUM_TOPOLOGICAL_LAYERS,
    RAW_RESULTS_ROOT,
    SEED,
    VF_LEVELS_A12,
    VF_LEVELS_A7,
    WORKLOAD_GROUP,
)
from gatppo_raw_profiles import (
    calibrate_execution_times,
    extract_processor_data,
    find_processor_folder,
    pair_processor_data,
    profile_report_rows,
)
from gatppo_dag_artifacts import write_dag_inspection_artifacts


def convert_np(obj):
    if isinstance(obj, dict):
        return {key: convert_np(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_np(value) for value in obj]
    if isinstance(obj, tuple):
        return [convert_np(value) for value in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def update_task_vf_attributes(vf_level: dict, max_voltage: float, task_data: dict) -> Tuple[float, float]:
    """Preserve the existing project's voltage-scaling assumption unchanged."""
    rho = float(vf_level["voltage"]) / float(max_voltage)
    execution_time = float(task_data["execution_time_max_vf"]) / max(rho, 1e-12)
    power = (rho**3) * float(task_data["avg_total_power_max_vf"])
    return float(execution_time), float(power)


def build_vf_levels(a7_task: dict, a12_task: dict) -> List[dict]:
    levels: List[dict] = []
    max_voltage_a12 = max(float(level["voltage"]) for level in VF_LEVELS_A12)
    max_voltage_a7 = max(float(level["voltage"]) for level in VF_LEVELS_A7)
    for level_a12, level_a7 in zip(VF_LEVELS_A12, VF_LEVELS_A7):
        time_a12, power_a12 = update_task_vf_attributes(level_a12, max_voltage_a12, a12_task)
        time_a7, power_a7 = update_task_vf_attributes(level_a7, max_voltage_a7, a7_task)
        levels.append(
            {
                "level": int(level_a12["level"]),
                "voltage_proc1": float(level_a12["voltage"]),
                "frequency_proc1": float(level_a12["frequency"]),
                "utilization_proc1_vf": time_a12,
                "avg_total_power_proc1": power_a12,
                "voltage_proc2": float(level_a7["voltage"]),
                "frequency_proc2": float(level_a7["frequency"]),
                "utilization_proc2_vf": time_a7,
                "avg_total_power_proc2": power_a7,
            }
        )
    return levels


def build_m_i_binomial_population(num_subtasks: int, rng: random.Random) -> List[int]:
    """Return finite-DAG gang widths following a symmetric Binomial(5, 0.5).

    The mathematical probabilities are proportional to (1, 5, 10, 10, 5, 1)
    for m_i=1..6.  Independent random draws can accidentally omit rare classes
    (especially m_i=1 or 6) in a 60-task DAG, so we first convert the target
    probabilities to integer quotas with the largest-remainder method and then
    shuffle the assignments.  This preserves the intended distribution while
    guaranteeing finite-sample coverage whenever the DAG is large enough.
    """
    if num_subtasks <= 0:
        raise ValueError("num_subtasks must be positive")
    if len(M_I_VALUES) != len(M_I_BINOMIAL_WEIGHTS):
        raise ValueError("M_I_VALUES and M_I_BINOMIAL_WEIGHTS length mismatch")

    weights = np.asarray(M_I_BINOMIAL_WEIGHTS, dtype=float)
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("M_I_BINOMIAL_WEIGHTS must be non-negative with positive sum")

    target = weights / weights.sum() * int(num_subtasks)
    counts = np.floor(target).astype(int)
    remaining = int(num_subtasks - counts.sum())

    # Largest-remainder apportionment.  For exact ties, prefer the more extreme
    # symmetric classes first so both rare tails remain represented for N=60.
    center = (len(M_I_VALUES) - 1) / 2.0
    order = sorted(
        range(len(M_I_VALUES)),
        key=lambda i: (target[i] - counts[i], abs(i - center)),
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1

    population: List[int] = []
    for value, count in zip(M_I_VALUES, counts.tolist()):
        population.extend([int(value)] * int(count))
    if len(population) != num_subtasks:
        raise AssertionError("m_i population size mismatch")

    rng.shuffle(population)
    return population


def generate_node_characteristics(num_subtasks: int, paired_data: Sequence[dict], rng: random.Random) -> List[dict]:
    if num_subtasks > len(paired_data):
        raise ValueError(
            f"num_subtasks={num_subtasks} exceeds the {len(paired_data)} matched profiles."
        )
    selected = rng.sample(list(paired_data), k=num_subtasks)
    m_i_population = build_m_i_binomial_population(num_subtasks, rng)
    result: List[dict] = []
    for pair, m_i in zip(selected, m_i_population):
        result.append(
            {
                "benchmark_name": pair["benchmark_name"],
                "period": 1,
                # Filled only after a feasible DAG-aware reference schedule exists.
                "implicit_deadline": None,
                "m_i": int(m_i),
                "v_f_levels": build_vf_levels(pair["A7"], pair["A12"]),
            }
        )
    return result


def _balanced_topological_layers(num_subtasks: int, num_layers: int) -> List[List[int]]:
    """Partition task ids into deterministic, nearly equal topological layers.

    Every layer is non-empty. Keeping ids contiguous by layer makes the generated
    DAG easier to audit visually while task/profile assignment remains shuffled by
    the existing seeded generator.
    """
    if num_subtasks <= 0:
        raise ValueError("num_subtasks must be positive")
    if num_layers < 2:
        raise ValueError("num_layers must be at least 2")
    if num_layers > num_subtasks:
        raise ValueError("num_layers cannot exceed num_subtasks")

    base, remainder = divmod(num_subtasks, num_layers)
    sizes = [base + (1 if layer < remainder else 0) for layer in range(num_layers)]
    layers: List[List[int]] = []
    start = 0
    for size in sizes:
        layers.append(list(range(start, start + size)))
        start += size
    return layers


def _topological_level_map(graph: nx.DiGraph) -> Dict[int, int]:
    """Return Kahn/topological-generation levels for an already validated DAG."""
    levels: Dict[int, int] = {}
    for level, generation in enumerate(nx.topological_generations(graph)):
        for node in generation:
            levels[int(node)] = int(level)
    return levels


def generate_dag(num_subtasks: int, edge_prob: float, np_rng: np.random.Generator) -> nx.DiGraph:
    """Generate a sparse layered DAG with no long-range skip edges.

    Construction contract:
      * nodes are partitioned into balanced topological layers;
      * an edge may only connect layer k to layer k+1;
      * every non-source task has >=1 predecessor in the previous layer;
      * every non-sink task has >=1 successor in the next layer;
      * extra adjacent-layer edges are sampled with ``edge_prob``;
      * a final topological-sort/generation audit rejects cycles or layer jumps.

    The mandatory edges keep the graph structurally connected across all layers,
    while the low probability controls density without creating near-complete
    bipartite blocks.
    """
    if not 0.0 <= edge_prob <= 1.0:
        raise ValueError("edge_prob must be in [0, 1]")

    layers = _balanced_topological_layers(num_subtasks, NUM_TOPOLOGICAL_LAYERS)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_subtasks))

    for layer_index in range(len(layers) - 1):
        current = layers[layer_index]
        nxt = layers[layer_index + 1]

        # First guarantee that every node in the next layer has a predecessor.
        shuffled_current = list(current)
        np_rng.shuffle(shuffled_current)
        for idx, target in enumerate(nxt):
            source = shuffled_current[idx % len(shuffled_current)]
            dag.add_edge(int(source), int(target))

        # Also guarantee that every current-layer node participates in at least
        # one dependency leading into the next layer.
        shuffled_next = list(nxt)
        np_rng.shuffle(shuffled_next)
        for idx, source in enumerate(current):
            target = shuffled_next[idx % len(shuffled_next)]
            dag.add_edge(int(source), int(target))

        # Add only sparse extra edges between these two adjacent layers.
        for source in current:
            for target in nxt:
                if dag.has_edge(source, target):
                    continue
                if float(np_rng.random()) < edge_prob:
                    dag.add_edge(int(source), int(target))

    # Explicit DAG/topological validation requested by the supervisor.
    if not nx.is_directed_acyclic_graph(dag):
        raise AssertionError("Generated graph must be acyclic")
    topo_order = list(nx.topological_sort(dag))
    if len(topo_order) != num_subtasks:
        raise AssertionError("Topological sort did not cover every task")

    actual_levels = _topological_level_map(dag)
    expected_levels = {node: level for level, nodes in enumerate(layers) for node in nodes}
    if actual_levels != expected_levels:
        raise AssertionError(
            "Generated topological levels do not match the intended layered construction"
        )

    bad_edges = [
        (int(u), int(v), actual_levels[int(u)], actual_levels[int(v)])
        for u, v in dag.edges()
        if actual_levels[int(v)] - actual_levels[int(u)] != 1
    ]
    if bad_edges:
        raise AssertionError(
            "DAG contains a non-adjacent layer edge; examples=" + repr(bad_edges[:10])
        )

    if not nx.is_weakly_connected(dag):
        raise AssertionError("Layered DAG must be weakly connected")
    return dag


def _max_vf_profile(task: dict, processor: str) -> Tuple[float, float]:
    proc = processor.upper()
    if proc == "A12":
        freq_key, time_key, power_key = (
            "frequency_proc1",
            "utilization_proc1_vf",
            "avg_total_power_proc1",
        )
    elif proc == "A7":
        freq_key, time_key, power_key = (
            "frequency_proc2",
            "utilization_proc2_vf",
            "avg_total_power_proc2",
        )
    else:
        raise ValueError(processor)
    level = max(task["v_f_levels"], key=lambda item: float(item[freq_key]))
    return float(level[time_key]), float(level[power_key])


def _valid_compositions(width: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for a12_count in range(width + 1):
        a7_count = width - a12_count
        if a7_count > NUM_A7_CORES or a12_count > NUM_A12_CORES:
            continue
        if not ALLOW_MIXED_GANGS and a7_count > 0 and a12_count > 0:
            continue
        out.append((a7_count, a12_count))
    if not out:
        raise ValueError(f"No valid gang composition for m_i={width}")
    return out


def _duration_energy(task: dict, a7_count: int, a12_count: int) -> Tuple[float, float]:
    time_a7, power_a7 = _max_vf_profile(task, "A7")
    time_a12, power_a12 = _max_vf_profile(task, "A12")
    if RUNTIME_MODEL == "perfect_linear_speedup":
        rate = 0.0
        if a7_count:
            rate += a7_count / max(time_a7, 1e-12)
        if a12_count:
            rate += a12_count / max(time_a12, 1e-12)
        duration = 1.0 / max(rate, 1e-12)
    elif RUNTIME_MODEL == "dataset_time_is_gang_time":
        participating = []
        if a7_count:
            participating.append(time_a7)
        if a12_count:
            participating.append(time_a12)
        duration = max(participating)
    else:
        raise ValueError(f"Unsupported RUNTIME_MODEL={RUNTIME_MODEL}")
    energy = duration * (a7_count * power_a7 + a12_count * power_a12)
    return float(duration), float(energy)


def _upward_ranks(graph: nx.DiGraph, characteristics: Sequence[dict]) -> np.ndarray:
    mean_duration = np.zeros(graph.number_of_nodes(), dtype=float)
    for task in graph.nodes:
        width = int(characteristics[task]["m_i"])
        durations = [_duration_energy(characteristics[task], *comp)[0] for comp in _valid_compositions(width)]
        mean_duration[task] = float(np.mean(durations))
    rank = np.zeros(graph.number_of_nodes(), dtype=float)
    for node in reversed(list(nx.topological_sort(graph))):
        succ = list(graph.successors(node))
        rank[node] = mean_duration[node] + (max(rank[s] for s in succ) if succ else 0.0)
    return rank


def _core_free(intervals: List[Tuple[float, float, int]], start: float, finish: float) -> bool:
    return all(not (start < e and finish > s) for s, e, _ in intervals)


def _estimate(
    task_id: int,
    task: dict,
    a7_count: int,
    a12_count: int,
    ready_time: float,
    calendars: List[List[Tuple[float, float, int]]],
) -> Tuple[float, float, List[int], float]:
    duration, energy = _duration_energy(task, a7_count, a12_count)
    a7_ids = list(range(NUM_A7_CORES))
    a12_ids = list(range(NUM_A7_CORES, NUM_A7_CORES + NUM_A12_CORES))
    candidate_times = {float(ready_time)}
    for calendar in calendars:
        for _, end, _ in calendar:
            if end >= ready_time:
                candidate_times.add(float(end))

    for start in sorted(candidate_times):
        finish = start + duration
        free_a7 = [cid for cid in a7_ids if _core_free(calendars[cid], start, finish)]
        free_a12 = [cid for cid in a12_ids if _core_free(calendars[cid], start, finish)]
        if len(free_a7) >= a7_count and len(free_a12) >= a12_count:
            selected = free_a7[:a7_count] + free_a12[:a12_count]
            return float(start), float(finish), selected, float(energy)

    latest = max(
        [ready_time]
        + [end for calendar in calendars for _, end, _ in calendar]
    )
    finish = latest + duration
    free_a7 = [cid for cid in a7_ids if _core_free(calendars[cid], latest, finish)]
    free_a12 = [cid for cid in a12_ids if _core_free(calendars[cid], latest, finish)]
    if len(free_a7) < a7_count or len(free_a12) < a12_count:
        raise RuntimeError("Reference scheduler could not find a feasible gang interval")
    return float(latest), float(finish), free_a7[:a7_count] + free_a12[:a12_count], float(energy)


@dataclass
class ReferenceSchedule:
    start: Dict[int, float]
    finish: Dict[int, float]
    allocation: Dict[int, Tuple[int, int]]
    cores: Dict[int, Tuple[int, ...]]
    makespan: float


def build_reference_schedule(graph: nx.DiGraph, characteristics: Sequence[dict]) -> ReferenceSchedule:
    """Build a deterministic HEFT-like feasible gang schedule.

    This schedule is used only to anchor absolute deadlines.  It respects the
    same precedence, core counts, gang width, mixed-gang flag, and runtime model
    used by the legacy environment.
    """
    ranks = _upward_ranks(graph, characteristics)
    calendars: List[List[Tuple[float, float, int]]] = [
        [] for _ in range(NUM_A7_CORES + NUM_A12_CORES)
    ]
    start: Dict[int, float] = {}
    finish: Dict[int, float] = {}
    allocation: Dict[int, Tuple[int, int]] = {}
    cores_by_task: Dict[int, Tuple[int, ...]] = {}
    unscheduled = set(graph.nodes)

    while unscheduled:
        ready = [
            task for task in unscheduled
            if all(pred in finish for pred in graph.predecessors(task))
        ]
        if not ready:
            raise RuntimeError("Reference scheduling deadlock")
        task_id = max(ready, key=lambda t: (float(ranks[t]), -int(t)))
        ready_time = max([finish[p] for p in graph.predecessors(task_id)] or [0.0])
        width = int(characteristics[task_id]["m_i"])

        best = None
        for a7_count, a12_count in _valid_compositions(width):
            candidate = _estimate(
                task_id,
                characteristics[task_id],
                a7_count,
                a12_count,
                ready_time,
                calendars,
            )
            c_start, c_finish, cores, energy = candidate
            key = (c_finish, energy, c_start, a7_count, a12_count)
            if best is None or key < best[0]:
                best = (key, candidate, (a7_count, a12_count))
        assert best is not None
        _, (task_start, task_finish, cores, _), comp = best
        for core_id in cores:
            calendars[core_id].append((task_start, task_finish, task_id))
            calendars[core_id].sort(key=lambda x: x[0])
        start[task_id] = task_start
        finish[task_id] = task_finish
        allocation[task_id] = comp
        cores_by_task[task_id] = tuple(int(core_id) for core_id in cores)
        unscheduled.remove(task_id)

    makespan = max(finish.values(), default=0.0)
    return ReferenceSchedule(
        start=start,
        finish=finish,
        allocation=allocation,
        cores=cores_by_task,
        makespan=makespan,
    )


def assign_dag_aware_deadlines(characteristics: List[dict], schedule: ReferenceSchedule, factor: float) -> None:
    for task_id, task in enumerate(characteristics):
        task["implicit_deadline"] = float(schedule.finish[task_id] * factor)


def validate_generated_dag(graph: nx.DiGraph, characteristics: Sequence[dict], schedule: ReferenceSchedule) -> dict:
    if len(characteristics) != graph.number_of_nodes():
        raise AssertionError("Task/graph size mismatch")
    if not nx.is_directed_acyclic_graph(graph):
        raise AssertionError("Graph is not a DAG")

    misses = []
    precedence_deadline_violations = []
    for task_id, task in enumerate(characteristics):
        deadline = float(task["implicit_deadline"])
        if schedule.finish[task_id] > deadline + 1e-9:
            misses.append(task_id)
    for u, v in graph.edges:
        if schedule.finish[u] > schedule.start[v] + 1e-9:
            raise AssertionError(f"Reference schedule violates precedence edge {(u, v)}")
        if float(characteristics[u]["implicit_deadline"]) >= float(characteristics[v]["implicit_deadline"]):
            precedence_deadline_violations.append((u, v))
    
    if precedence_deadline_violations:
        raise AssertionError(
            "Successor deadline must be later than predecessor deadline: "
            f"{precedence_deadline_violations[:10]}"
        )
    return {
        "reference_deadline_misses": len(misses),
        "reference_makespan_sec": float(schedule.makespan),
        "min_deadline_sec": float(min(float(t["implicit_deadline"]) for t in characteristics)),
        "max_deadline_sec": float(max(float(t["implicit_deadline"]) for t in characteristics)),
    }


def generate_dataset(
    num_dags: int,
    num_subtasks: int,
    edge_prob_ranges: Sequence[Tuple[float, float]],
    paired_data: Sequence[dict],
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    dataset: List[dict] = []
    diagnostics: List[dict] = []

    for dag_index in range(num_dags):
        low, high = edge_prob_ranges[int(np_rng.integers(0, len(edge_prob_ranges)))]
        edge_prob = float(np_rng.uniform(low, high))
        graph = generate_dag(num_subtasks, edge_prob, np_rng)
        characteristics = generate_node_characteristics(num_subtasks, paired_data, py_rng)
        reference = build_reference_schedule(graph, characteristics)
        assign_dag_aware_deadlines(characteristics, reference, DEADLINE_FACTOR)
        diag = validate_generated_dag(graph, characteristics, reference)
        m_i_counts = {str(value): 0 for value in M_I_VALUES}
        for task in characteristics:
            m_i_counts[str(int(task["m_i"]))] += 1
        topo_levels = _topological_level_map(graph)
        max_layer_jump = max(
            (topo_levels[int(v)] - topo_levels[int(u)] for u, v in graph.edges()),
            default=0,
        )
        possible_directed_edges = max(num_subtasks * (num_subtasks - 1), 1)
        diag.update(
            {
                "dag_index": dag_index,
                "num_subtasks": num_subtasks,
                "num_edges": graph.number_of_edges(),
                "edge_prob": edge_prob,
                "num_topological_layers": 1 + max(topo_levels.values(), default=0),
                "max_layer_jump": int(max_layer_jump),
                "directed_edge_density": graph.number_of_edges() / float(possible_directed_edges),
                "weakly_connected": bool(nx.is_weakly_connected(graph)),
                "m_i_counts": m_i_counts,
                "m_i_percentages": {
                    key: 100.0 * count / float(num_subtasks)
                    for key, count in m_i_counts.items()
                },
            }
        )
        inspection = write_dag_inspection_artifacts(
            LEGACY_DAG_INSPECTION_DIR,
            dag_index,
            graph,
            characteristics,
            reference,
            edge_prob,
        )
        diag["inspection_artifacts"] = inspection
        diagnostics.append(diag)
        dataset.append(
            {
                "num_subtasks": int(num_subtasks),
                "edge_prob": edge_prob,
                "characteristics": characteristics,
                "edges": [(int(u), int(v)) for u, v in graph.edges()],
            }
        )
    return dataset, diagnostics


def _write_outputs(dataset: Sequence[dict], paired_data: Sequence[dict], metadata: dict) -> None:
    LEGACY_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "num_subtasks": data["num_subtasks"],
            "edge_prob": data["edge_prob"],
            "characteristics": json.dumps(convert_np(data["characteristics"])),
            "edges": json.dumps(convert_np(data["edges"])),
        }
        for data in dataset
    ]
    # IMPORTANT: exact legacy schema is preserved.
    pd.DataFrame(rows, columns=["num_subtasks", "edge_prob", "characteristics", "edges"]).to_csv(
        LEGACY_OUTPUT_CSV, index=False
    )
    pd.DataFrame(profile_report_rows(list(paired_data))).to_csv(
        LEGACY_PROFILE_REPORT_CSV, index=False
    )
    LEGACY_METADATA_JSON.write_text(
        json.dumps(convert_np(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> List[dict]:
    parser = argparse.ArgumentParser(description="Generate corrected legacy GAT dataset")
    parser.add_argument(
        "--sample-interval-sec",
        type=float,
        default=None,
        help=(
            "Verified physical acquisition interval in seconds/sample. Omit this option "
            "to use the default reproducible canonical relative-time calibration. "
            "Never supply a guessed value."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.sample_interval_sec is not None and args.sample_interval_sec <= 0.0:
        parser.error("--sample-interval-sec must be > 0 when supplied")

    a7_folder = find_processor_folder(RAW_RESULTS_ROOT, WORKLOAD_GROUP, "A7")
    a12_folder = find_processor_folder(RAW_RESULTS_ROOT, WORKLOAD_GROUP, "A12")
    a7_data = extract_processor_data(a7_folder)
    a12_data = extract_processor_data(a12_folder)
    paired_raw = pair_processor_data(a7_data, a12_data)
    paired_data, timing_calibration = calibrate_execution_times(
        paired_raw, physical_sample_interval_sec=args.sample_interval_sec
    )

    dataset, diagnostics = generate_dataset(
        NUM_DAGS,
        NUM_SUBTASKS,
        EDGE_PROB_RANGES,
        paired_data,
        SEED,
    )

    ratios = [
        pair["A7"]["execution_time_max_vf"] / pair["A12"]["execution_time_max_vf"]
        for pair in paired_data
    ]
    metadata = {
        "generator": "generate_data/gat + ppo/legacy_gat_generator.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_schema_preserved": ["num_subtasks", "edge_prob", "characteristics", "edges"],
        "raw_results_workload_group": WORKLOAD_GROUP,
        "raw_timing": {
            "raw_index_semantics": (
                "# is validated as contiguous sample counter 1..N and is never interpreted "
                "directly as elapsed seconds"
            ),
            **timing_calibration,
            "training_scale_note": (
                "One common time scale is used for every task and both processors. "
                "Therefore A7/A12 timing ratios and all DAG scheduling relationships are "
                "preserved. The legacy environment subsequently normalizes time/energy "
                "features before they enter GAT/PPO."
            ),
        },
        "dag_topology": {
            "type": "sparse_balanced_adjacent_topological_layers",
            "num_layers": NUM_TOPOLOGICAL_LAYERS,
            "edge_rule": "only layer_k -> layer_k_plus_1",
            "edge_probability_ranges": [list(item) for item in EDGE_PROB_RANGES],
            "mandatory_connectivity": (
                "every non-source task has a predecessor in the previous layer and "
                "every non-sink task has a successor in the next layer"
            ),
            "validation": (
                "networkx DAG check + topological_sort + topological_generations + "
                "max layer jump == 1 + weak connectivity"
            ),
        },
        "deadline_method": {
            "type": "feasible_reference_schedule_absolute_deadline",
            "formula": "deadline_i = deadline_factor * reference_finish_i",
            "deadline_factor": DEADLINE_FACTOR,
            "reference_policy": "HEFT-like upward-rank ready-task order + earliest-finish gang placement",
            "runtime_model": RUNTIME_MODEL,
            "num_a7_cores": NUM_A7_CORES,
            "num_a12_cores": NUM_A12_CORES,
            "allow_mixed_gangs": ALLOW_MIXED_GANGS,
        },
        "m_i_distribution": {
            "name": M_I_DISTRIBUTION_NAME,
            "values": list(M_I_VALUES),
            "binomial_weights": list(M_I_BINOMIAL_WEIGHTS),
            "exact_probabilities_percent": [
                100.0 * float(weight) / float(sum(M_I_BINOMIAL_WEIGHTS))
                for weight in M_I_BINOMIAL_WEIGHTS
            ],
            "finite_dag_sampling": (
                "largest-remainder integer quotas followed by seeded shuffle; "
                "avoids accidental omission of rare m_i classes"
            ),
        },
        "reproducibility": {"seed": SEED, "num_dags": NUM_DAGS, "num_subtasks": NUM_SUBTASKS},
        "profile_checks": {
            "matched_a7_a12_profiles": len(paired_data),
            "a12_faster_than_a7_profiles": sum(
                pair["A12"]["execution_time_max_vf"] < pair["A7"]["execution_time_max_vf"]
                for pair in paired_data
            ),
            "a7_to_a12_time_ratio_min": float(min(ratios)),
            "a7_to_a12_time_ratio_median": float(np.median(ratios)),
            "a7_to_a12_time_ratio_max": float(max(ratios)),
        },
        "dag_diagnostics": diagnostics,
    }
    _write_outputs(dataset, paired_data, metadata)

    print(f"A7 folder               : {a7_folder}")
    print(f"A12 folder              : {a12_folder}")
    print(f"Matched benchmark pairs : {len(paired_data)}")
    print(f"Timing mode             : {timing_calibration['mode']}")
    if timing_calibration["mode"] == "canonical_relative_time":
        print(
            "Timing reference        : median A12 sample count "
            f"= {timing_calibration['reference_sample_count']:.9g} -> "
            "1 canonical simulation second"
        )
        print("Physical seconds claimed : NO")
    else:
        print(
            "Physical sample interval: "
            f"{timing_calibration['seconds_per_sample']:.9g} sec/sample"
        )
        print("Physical seconds claimed : YES (explicit interval supplied)")
    for diag in diagnostics:
        print(
            f"DAG {diag['dag_index']}: tasks={diag['num_subtasks']} "
            f"edges={diag['num_edges']} layers={diag['num_topological_layers']} "
            f"max_layer_jump={diag['max_layer_jump']} "
            f"density={diag['directed_edge_density']:.6f} "
            f"reference_makespan={diag['reference_makespan_sec']:.6f}s "
            f"reference_deadline_misses={diag['reference_deadline_misses']} "
            f"m_i_counts={diag['m_i_counts']}"
        )
    print(f"Dataset saved           : {LEGACY_OUTPUT_CSV}")
    print(f"Profile report saved    : {LEGACY_PROFILE_REPORT_CSV}")
    print(f"Generation metadata     : {LEGACY_METADATA_JSON}")
    print(f"DAG inspection folder   : {LEGACY_DAG_INSPECTION_DIR}")
    for diag in diagnostics:
        artifacts = diag.get("inspection_artifacts", {})
        if artifacts:
            print(f"  DAG details CSV       : {artifacts.get('details_csv')}")
            print(f"  DAG detailed PNG      : {artifacts.get('detailed_png')}")
    return dataset


if __name__ == "__main__":
    main()
