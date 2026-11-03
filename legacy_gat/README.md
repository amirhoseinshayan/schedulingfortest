# PPO+GAT Gang Scheduling Integration

## Confirmed Gang meaning

`m_i` is the exact number of physical cores that task `i` must reserve simultaneously.
Every selected core receives the same task start and finish interval.

## Default hardware

- 6 physical A7 cores
- 6 physical A12 cores
- `max_gang_size = 6`
- mixed A7/A12 gangs enabled

The hardware is configurable in `config.py`.

## Important runtime assumption

The professor-supplied benchmark data contains single-core A7/A12 time and power but does not contain measured runtime versus gang width.
The default code therefore uses an explicitly idealized model:

```text
service_rate = nA7 / T_A7 + nA12 / T_A12
T_gang       = 1 / service_rate
E_gang       = T_gang * (nA7*P_A7 + nA12*P_A12)
```

Do not present this as measured multicore speedup. Replace it with measured `T_i(nA7,nA12)` data if such measurements become available.

## Reward used by default

Advisor-requested experiment:

```text
reward = QoS * energy_term
```

Makespan is still calculated and reported but is not part of the default reward.

## Prepare the dataset

1. Place the extracted benchmark folders under `data/legacy/raw_results/`.
2. The generator uses workload group `60`, because that group contains 60 matched A7/A12 benchmark files.
3. Run the existing command:

```powershell
.\make.bat generate
```

The command is backward-compatible but now delegates to `generate_data/gat + ppo/legacy_gat_generator.py`. The output schema remains unchanged:

```text
data/legacy/dag_dataset_gang_a7_a12.csv
```

Two dataset-generation corrections are applied before training:

- raw `#` values are validated only as sample indices; because the archive contains no timestamp metadata, the default generator converts them to a documented canonical relative-time scale (median A12 sample count = 1 canonical simulation second) instead of assuming `1 sample = 1 physical second`;
- deadlines are absolute DAG-aware deadlines anchored to a feasible reference Gang schedule, rather than isolated task runtimes measured from `t=0`.

Generation provenance is written to:

```text
data/legacy/benchmark_profiles_a7_a12.csv
data/legacy/dataset_generation_metadata.json
```

See `generate_data/README.md` for the timing calibration contract. Supply `--sample-interval-sec` only if the real acquisition period is known and verified. Every generation also writes `data/legacy/dag_inspection/dag_0000_full_details.csv` and `dag_0000_detailed.png` for direct inspection of the generated DAG.

## Validate before training

```powershell
.\make.bat validate
```

This runs Gym/SB3 environment checks and a complete `HEFT_GANG` schedule, then verifies:

- every task is scheduled exactly once;
- allocated core count equals `m_i`;
- all gang cores have the same interval;
- no physical core reservations overlap;
- all DAG precedence constraints hold;
- QoS is within `[0,1]`.

## Recommended first run

Do not start with 200k immediately. First:

```powershell
.\make.bat pilot
```

This trains for 20,000 timesteps.
Inspect TensorBoard:

```powershell
.\make.bat tensorboard
```

Check especially:

```text
exploration/alpha
gang/usage_rate
gang/mean_width
gang/mean_a12_fraction
train/approx_kl
train/clip_fraction
train/explained_variance
```

## Full training

After the pilot passes:

```powershell
.\make.bat clean
.\make.bat train
```

`clean` removes only `artifacts/legacy_gat`.

## Evaluation

```powershell
.\make.bat evaluate
```

The default comparison is fair for Gang semantics:

```text
HEFT_GANG vs PPO_GANG
```

PPO is evaluated 20 stochastic times with `alpha = 0.01`.
Outputs are stored in:

```text
artifacts/legacy_gat/evaluation/
```

including:

```text
comparison_detail.csv
comparison_summary.csv
comparison_improvements.csv
comparison_improvements_summary.csv
qos_report.csv
gang_composition_detail.csv
makespan_comparison.png
energy_comparison.png
qos_comparison.png
```

## Action representation

The actor does not enumerate concrete core subsets.
For task `i`:

```text
action = task_id * 7 + slot
slot = A12 core count
A7 core count = m_i - slot
```

Example for `m_i = 4`:

```text
slot 0 -> 4 A7 + 0 A12
slot 1 -> 3 A7 + 1 A12
slot 2 -> 2 A7 + 2 A12
slot 3 -> 1 A7 + 3 A12
slot 4 -> 0 A7 + 4 A12
```

Slots 5 and 6 are masked for this task.

## Old model compatibility

The old model cannot be reused because the action space changes from approximately:

```text
Task x 2 processor choices
```

to:

```text
Task x 7 Gang-composition choices
```

Train the Gang model from scratch and keep old results separately.
