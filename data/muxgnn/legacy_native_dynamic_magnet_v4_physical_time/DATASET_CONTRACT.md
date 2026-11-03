# Dataset contract — MAGNET v4 physical time

This v4 dataset is an isolated addition. Existing v2/v3 dataset code, muxGNN/PPO code, reward, HEFT baseline, and evaluation code are unchanged.

## Confirmed timing assumptions

1. The original A7/A12 raw execution values are interpreted as **milliseconds** (supervisor confirmation).
2. v4 stores all scheduling times in **seconds** using `T_sec = T_ms / 1000`.
3. The source `period=1` is a mathematical placeholder used so `utilization = execution_time / period` numerically equals execution time. It is not treated as 1 s, an arrival interval, or a task release period.
4. The workload is one DAG known completely at design time. Therefore every task has external `release_time_sec = 0`.
5. A task is executable only when it is dependency-ready. Roots have `ready_time_sec=0`; non-roots use `ready_time_sec=max(predecessor_finish_time_sec)` because external release is zero.

## Deadline contract

The old 60-task Legacy deadline was derived from a HEFT-like reference schedule. That creates an avoidable baseline-coupling issue for a future HEFT-vs-PPO comparison. v4 therefore does **not** use that value as the learning/evaluation deadline.

v4 uses a scheduler-independent DAG-aware deadline model:

- `w_i = min(T_A7_i, T_A12_i)` (no division by `m_i`; `m_i` is the simultaneous core requirement)
- The existing audit-column name `fastest_gang_time_sec` is retained for schema compatibility; for deadline construction its value is this `w_i`.
- `EF_i = max(EF_pred) + w_i`, roots use predecessor term 0
- `CP = max(EF_sink)`
- `D_workflow = 1.15 * CP`
- `D_i = (EF_i/CP) * D_workflow = 1.15 * EF_i`

The factor 1.15 preserves the original 15% slack concept while removing dependence on HEFT or PPO. The previous Legacy deadline is retained only in `deadline_audit.csv` for traceability.

## Runtime, gang, and energy contract

The supervisor-confirmed interpretation is:

- `Runtime_A7_i = T_A7_i`
- `Runtime_A12_i = T_A12_i`
- `m_i` is only the number of simultaneous cores that must be available and reserved for task `i`.
- There is **no** `T/m_i` execution-time scaling anywhere in v4.

The legacy field names `a7_gang_time_sec`, `a12_gang_time_sec`, `gang_exec_time_a7_sec`, and `gang_exec_time_a12_sec` are retained only for schema compatibility; their values equal the corresponding measured task execution times.

Power is stored in watts using the supplied average-total-power profile. Physical energy is recomputed in joules as:

`E = T_sec * P_watt`.

`m_i` does not multiply or divide runtime or energy in this dataset; it affects only capacity, feasibility, reservation, and contention.

## Graph contract

Node types: `task`, `vehicle`, `edge_server`.

Relations:

- `(task, precedes, task)` — topology only; no invented edge feature.
- `(task, local_execute_on, vehicle)`.
- `(task, v2v_offload_to, vehicle)`.
- `(task, v2i_offload_to, edge_server)`.
- `(task, contends_with, task)` — dynamic.

Temporary core scarcity does not delete a structurally valid candidate relation. It changes availability/feasibility features.

## Scenario split

24 scenarios: 18 train / 3 validation / 3 test. Split is at full-scenario level, so snapshots from one scenario cannot leak across splits. The three non-learning trajectory policies (`earliest_finish`, `edf`, `lowest_energy`) are used only to create state coverage; they are not PPO and not the HEFT evaluation baseline.

## Intentionally absent

No mobility trace, coordinates, physical distance, bandwidth, task data size, V2V/V2I transmission delay, contact duration, queue delay, or migration delay is fabricated. The physical time base now makes it dimensionally valid to add those later when a defensible source/model is selected.

## Serialization note

The ZIP ships source tables, metadata, scenario trajectories, generator, and validators. Pre-serialized PyG `.pt` snapshots are intentionally not bundled because the build environment used to prepare the ZIP does not include `torch-geometric`. In the project environment, generate the real PyG snapshots with:

```powershell
python -m muxgnn.graph.generate_v4_dataset
python -m muxgnn.graph.validate_v4_dataset
```
