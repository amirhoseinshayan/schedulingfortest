# Dataset generation workspace

This folder keeps the two dataset-generation routes next to each other:

```text
generate_data/
├── gat + ppo/    # Phase 1: legacy GAT + PPO dataset generator (implemented)
└── muxgnn/       # Phase 2: muxGNN + PPO generator (reserved for next phase)
```

## Phase 1 — GAT + PPO

All implementation files used to build the Phase-1 dataset are now under:

```text
generate_data/gat + ppo/
```

The old project command is intentionally unchanged:

```powershell
.\make.bat generate
```

`legacy_gat/generate_dataset.py` is only a compatibility entry point; it delegates to the generator above so the training/evaluation project structure does not change.

### Outputs

The legacy training dataset keeps exactly the same schema and path:

```text
data/legacy/dag_dataset_gang_a7_a12.csv
```

Audit/provenance outputs:

```text
data/legacy/benchmark_profiles_a7_a12.csv
data/legacy/dataset_generation_metadata.json
```

DAG inspection outputs are generated automatically on every dataset build:

```text
data/legacy/dag_inspection/dag_0000_full_details.csv
data/legacy/dag_inspection/dag_0000_detailed.png
```

The detailed CSV has one row per task and contains DAG connectivity, topological order/level, benchmark, gang width, deadline, reference schedule, core allocation, and every A7/A12 DVFS field. The PNG is a high-resolution layered rendering of the full directed DAG and shows key scheduling values inside every node.

### Correct execution-time handling

The raw CSV column `#` is validated only as a contiguous sample counter (`1..N`). It is never treated directly as timestamp seconds. The archive contains no acquisition-period metadata, so the default generator does **not** invent a physical sampling interval.

By default it uses a reproducible canonical relative-time calibration:

```text
reference_samples = median(A12 sample counts across the matched 60 profiles)
execution_time_canonical_sec = sample_count / reference_samples
```

For the supplied workload group, `reference_samples = 43`, so 43 raw samples define `1.0 canonical simulation second`. This is a common scale applied to every A7/A12 profile, preserving all timing ratios and scheduling relationships while making no unsupported wall-clock claim. The legacy schema is unchanged; metadata explicitly marks these values as canonical simulation seconds.

If the real acquisition interval later becomes known and verified, physical seconds can be generated explicitly:

```powershell
python "generate_data/gat + ppo/legacy_gat_generator.py" --sample-interval-sec 0.01
```

Only use that option with a documented measurement interval.

### DAG-aware attainable deadlines

A deterministic feasible reference Gang schedule is built first. It respects DAG precedence, `m_i`, A7/A12 core counts, the mixed-gang setting, and the same runtime model used by the legacy environment. Ready tasks are selected by an HEFT-like upward rank and each task is placed on the valid gang composition with the earliest feasible finish.

For task `i`:

```text
ready_i = max(reference_finish[p] for p in predecessors(i))
reference_finish_i = earliest feasible finish after ready_i
absolute_deadline_i = 1.15 * reference_finish_i
```

The generator rejects a DAG if the reference schedule misses any generated deadline. Thus the deadline is an absolute DAG-aware deadline rather than `1.15 * isolated_task_runtime` measured from time zero.

## Phase 2 — muxGNN + PPO

`generate_data/muxgnn/` remains reserved for the next phase. It will reuse the same corrected raw A7/A12 semantics established in Phase 1 rather than reintroducing the old row-index timing bug.
