# Legacy-native Simple MAGNET Heterogeneous Graph v2

This folder is the new base muxGNN dataset.  It is built only from the validated
Legacy 60-task DAG and its A7/A12/Gang/Deadline profiles.  DAGBench is not used.

## Graph schema

- Task nodes: 60
- Vehicle nodes: 6
- Edge-server nodes: 1
- DAG precedence edges: 126
- Local task->vehicle edges: 60
- V2V compute-candidate edges: 109
- V2I compute-candidate edges: 60
- Initial gang-contention edges: 30

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
