# Legacy-native Dynamic MAGNET v4 — Physical Time

This is the finalized compute-scheduling dataset stage for the next muxGNN/PPO and HEFT integration.

## What changed from v3

- Original raw A7/A12 execution values are interpreted as **milliseconds**, based on supervisor confirmation.
- All scheduling times are stored in **seconds**: `T_sec = T_ms / 1000`.
- Power is in watts and energy is recomputed in joules: `E_J = T_sec * P_W`. The measured task time is used directly; `m_i` only reserves simultaneous cores and never divides execution time.
- This is one design-time DAG: every task has `release_time_sec = 0`.
- Release time and dependency-ready time are separate. Roots are ready at 0; non-roots become ready at `max(predecessor_finish_time_sec)`.
- `period=1` is preserved only as a source placeholder used historically to make utilization numerically equal execution time. It is **not** interpreted as 1 second, an arrival period, or a release time.
- Final task deadlines are scheduler-independent and DAG-aware. The workflow deadline is `1.15 * critical_path`, and each task receives a proportional absolute subdeadline (`D_i = 1.15 * dependency_earliest_finish_i`) using `w_i=min(T_A7_i,T_A12_i)` with no division by `m_i`. The old HEFT-like deadline is retained only in the audit CSV.

## Dataset shape

- 60 tasks, 126 DAG dependency edges, same validated layered DAG as v2/v3.
- Node types: Task, Vehicle, EdgeServer.
- Relations: `precedes`, `local_execute_on`, `v2v_offload_to`, `v2i_offload_to`, `contends_with`.
- 24 event-driven scenarios: 18 train / 3 validation / 3 test.
- Total snapshots for this deterministic design after generation: 2902.
- Split unit is the full scenario; snapshots from one scenario never cross splits.

## Dynamic state

Task ready/running/completed flags, ready time, free A7/A12 cores, candidate feasibility, and contention are updated only by valid task start/completion events. Candidate topology remains stable under temporary core scarcity; feasibility is an edge feature.

## Intentionally not invented

No mobility trace, coordinates, distance, bandwidth, data size, transmission delay, contact duration, queue delay, or migration delay is synthesized here. Those require their own defensible source/model and can now be added later in seconds without a unit mismatch.

## Build and validate

```powershell
python -m muxgnn.graph.generate_v4_dataset
python -m muxgnn.graph.validate_v4_dataset
```

The generator deletes/rebuilds **only** `data/muxgnn/legacy_native_dynamic_magnet_v4_physical_time`. Existing v2/v3 datasets, muxGNN/PPO code, reward, HEFT, and evaluation files are untouched.
