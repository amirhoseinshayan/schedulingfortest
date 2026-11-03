# GAT + PPO dataset generator

This folder contains the complete implementation for building and auditing the Phase-1 legacy GAT/PPO dataset.

Files:

- `legacy_gat_generator.py` — orchestrates raw-profile loading, DAG generation, reference scheduling, deadline assignment, validation, and output writing.
- `gatppo_config.py` — dataset generation constants and output paths.
- `gatppo_raw_profiles.py` — validates raw A7/A12 CSVs and applies the canonical relative-time calibration (or an explicitly supplied verified physical sampling interval).
- `gatppo_dag_artifacts.py` — writes the one-row-per-task detailed CSV and the high-resolution DAG PNG.

Recommended command from the project root remains:

```powershell
.\make.bat generate
```

Generated inspection files:

```text
data/legacy/dag_inspection/dag_0000_full_details.csv
data/legacy/dag_inspection/dag_0000_detailed.png
```

## Timing calibration

The raw traces contain sample counters but no timestamps. The default generator therefore uses the median A12 sample count as a deterministic canonical reference instead of assuming `1 sample = 1 second`. For the current 60 matched profiles, `43 samples = 1 canonical simulation second`. This preserves every A7/A12 timing ratio and keeps the existing dataset schema unchanged. Physical wall-clock seconds are generated only when a verified `--sample-interval-sec` is explicitly provided.

## Gang-width (`m_i`) distribution

The Phase-1 generator uses a shifted symmetric `Binomial(5, 0.5)` distribution
for `m_i in {1,2,3,4,5,6}`.  The exact probabilities are:

- `m_i=1`: 3.125%
- `m_i=2`: 15.625%
- `m_i=3`: 31.25%
- `m_i=4`: 31.25%
- `m_i=5`: 15.625%
- `m_i=6`: 3.125%

For a finite 60-task DAG, independent random draws could omit the rare classes.
The generator therefore converts these probabilities to integer quotas using
the largest-remainder method and then shuffles the assignments with the project
seed.  For 60 tasks this gives counts `[2, 9, 19, 19, 9, 2]`, i.e. roughly
`[3.33%, 15%, 31.67%, 31.67%, 15%, 3.33%]`.
