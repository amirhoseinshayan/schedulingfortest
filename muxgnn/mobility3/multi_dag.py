"""Seven different, validated DAGs per episode; no changes to measured task profiles.

Each DAG changes only precedence (126 edges) and independently recomputes
scheduler-independent CP/deadlines using the existing v4 1.15x rule. Seeds are
split-disjoint; no run-time/energy re-sampling. Source v4 remains immutable.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import networkx as nx
import numpy as np

from .scaled_source import load_scaled_task_graph

DAGS_PER_EPISODE = 7
EPISODE_DURATION_SEC = 600.0
BASE_SLACK_FACTOR = 1.15
SPLIT_BASES = {'train': 1_000_000, 'val': 2_000_000_000, 'test': 4_000_000_000}


def dag_seed(split: str, episode_index: int, dag_index: int) -> int:
    if split not in SPLIT_BASES or episode_index < 0 or not 0 <= dag_index < DAGS_PER_EPISODE:
        raise ValueError('Invalid split/episode/dag index')
    return SPLIT_BASES[split] + episode_index * DAGS_PER_EPISODE + dag_index


def make_workload(seed: int) -> dict:
    """Generate a connected, acyclic 60-node, 126-edge, 8-generation graph.

    A random topological ordering and random layer widths give different root
    counts and dependency readiness. Mandatory predecessor edges for all nodes
    after layer zero ensure genuine eight generations. Root-to-next-layer
    connections make the underlying graph connected. Other edges are uniformly
    selected from remaining permissible forward-layer pairs.
    """
    source = load_scaled_task_graph()
    n = len(source['profiles'])
    required_edges = len(source['dependencies'])
    required_layers = len(source['layers'])
    if (n, required_edges, required_layers) != (60, 126, 8):
        raise ValueError('Unexpected validated v4 DAG dimensions')
    rng = np.random.default_rng(int(seed))
    order = list(map(int, rng.permutation(n)))
    # Nondegenerate root count (2..8) and at least two nodes in every layer.
    root_count = int(rng.integers(2, 9))
    widths = [root_count] + [2] * (required_layers - 1)
    for _ in range(n - sum(widths)):
        widths[int(rng.integers(1, required_layers))] += 1
    layers, start = [], 0
    for width in widths:
        layers.append(order[start:start + width]); start += width
    layer_by_task = {task: layer for layer, ids in enumerate(layers) for task in ids}
    edges = set()
    # All layer-zero roots have a directed route to the same layer-one anchor.
    for root in layers[0]:
        edges.add((root, layers[1][0]))
    for layer in range(1, required_layers):
        for task in layers[layer]:
            if layer == 1 and task == layers[1][0]:
                continue
            edges.add((int(rng.choice(layers[layer - 1])), task))
    candidates = [(u, v) for layer in range(required_layers) for later in range(layer + 1, required_layers)
                  for u in layers[layer] for v in layers[later] if (u, v) not in edges]
    rng.shuffle(candidates)
    edges.update(candidates[:required_edges - len(edges)])
    dag = nx.DiGraph(); dag.add_nodes_from(range(n)); dag.add_edges_from(sorted(edges))
    derived_layers = [list(layer) for layer in nx.topological_generations(dag)]
    if not nx.is_directed_acyclic_graph(dag) or not nx.is_weakly_connected(dag):
        raise AssertionError('Invalid random DAG')
    if len(edges) != required_edges or len(derived_layers) != required_layers:
        raise AssertionError('DAG size/layers mismatch')
    profiles = deepcopy(source['profiles'])
    ef = np.zeros(n, dtype=float)
    er = np.zeros(n, dtype=float)
    for task in nx.topological_sort(dag):
        preds = list(dag.predecessors(task))
        er[task] = max([ef[p] for p in preds], default=0.)
        p = profiles[task]
        fastest = min(float(p['a7_time_sec']), float(p['a12_time_sec']))
        ef[task] = er[task] + fastest
    cp = float(max(ef))
    workflow_deadline = BASE_SLACK_FACTOR * cp
    for task, p in enumerate(profiles):
        p['topological_layer'] = layer_by_task[task]
        p['dependency_earliest_ready_sec'] = float(er[task])
        p['dependency_earliest_finish_sec'] = float(ef[task])
        p['fastest_gang_time_sec'] = float(ef[task] - er[task])
        p['deadline_sec'] = float(BASE_SLACK_FACTOR * ef[task])
        p['deadline_slack_from_dependency_earliest_finish_sec'] = float((BASE_SLACK_FACTOR - 1.) * ef[task])
        # Reference v4 measurements and power are unchanged. Only dependencies
        # and their derived time/deadline annotations differ.
    canonical_edges = sorted((int(u), int(v)) for u, v in edges)
    fingerprint = hashlib.sha256(json.dumps(canonical_edges).encode()).hexdigest()
    return {**source, 'dag':dag, 'profiles':profiles, 'dependencies':canonical_edges,
            'layers':derived_layers, 'level_by_task':layer_by_task,
            'roots':sorted(int(t) for t in dag.nodes if dag.in_degree(t)==0),
            'sinks':sorted(int(t) for t in dag.nodes if dag.out_degree(t)==0),
            'critical_path_sec':cp,'workflow_deadline_sec':workflow_deadline,
            'dag_seed':int(seed),'dag_fingerprint':fingerprint}


def episode_workloads(split: str, episode_index: int):
    graphs = [make_workload(dag_seed(split, episode_index, i)) for i in range(DAGS_PER_EPISODE)]
    hashes = [g['dag_fingerprint'] for g in graphs]
    if len(set(hashes)) != DAGS_PER_EPISODE:
        raise AssertionError('Duplicate DAG in an episode')
    return graphs


def dag_summary(g: dict) -> dict:
    return {'dag_seed':g['dag_seed'], 'fingerprint':g['dag_fingerprint'],
            'num_tasks':len(g['profiles']), 'num_dependencies':len(g['dependencies']),
            'num_layers':len(g['layers']), 'num_roots':len(g['roots']),
            'critical_path_sec':g['critical_path_sec'], 'deadline_sec':g['workflow_deadline_sec']}
