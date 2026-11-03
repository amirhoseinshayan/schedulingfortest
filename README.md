# Heterogeneous DAG Scheduling: PPO+GAT and muxGNN

This repository contains two scheduling paths built around the same A7/A12 task profiles:

1. **Legacy PPO + GAT gang scheduler** — schedules DAG tasks on heterogeneous A7/A12 cores.
2. **Dynamic muxGNN + PPO scheduler** — schedules DAG tasks across vehicles and an edge server using a five-relation heterogeneous graph.

The two execution routes are physically separated and exposed through:

- `run_legacy_gat.py` -> `legacy_gat/`
- `run_muxgnn.py` -> `muxgnn/`

Shared project data is centralized under `data/`, while runtime outputs are written under `artifacts/`.

## Project layout

```text
scheduling_utils/
├── legacy_gat/          # legacy PPO + GAT implementation
├── muxgnn/
│   ├── graph/           # graph/dataset construction and validation
│   └── rl/              # offline muxGNN pretraining + PPO
├── data/
│   ├── legacy/          # raw A7/A12 results + legacy CSV dataset
│   └── muxgnn/          # static and dynamic graph data
├── artifacts/           # generated at runtime
├── run_legacy_gat.py
└── run_muxgnn.py
```

## 1. Installation

```bash
pip install -r requirements.txt
```

Main dependencies include PyTorch, PyTorch Geometric, Gymnasium, Stable-Baselines3, NetworkX, and DAGBench.

---

## 2. Legacy PPO + GAT route

Main flow:

```text
data/legacy/raw_results/
  -> legacy_gat/generate_dataset.py (compatibility entry point)
     -> generate_data/gat + ppo/legacy_gat_generator.py (audited implementation)
  -> data/legacy/dag_dataset_gang_a7_a12.csv
  -> legacy_gat/env.py
  -> legacy_gat/policy.py
  -> legacy_gat/train.py
  -> legacy_gat/evaluate.py
```

Convenience commands:

```bash
python run_legacy_gat.py generate
python run_legacy_gat.py validate
python run_legacy_gat.py train --timesteps 200000
python run_legacy_gat.py evaluate
```

Or run the complete legacy route:

```bash
python run_legacy_gat.py all --timesteps 200000
```

The legacy scheduler uses gang scheduling on A7/A12 cores, action masking, PPO, and a directed GAT-based policy. HEFT is used as the evaluation baseline.

Generation also creates inspection artifacts automatically:

```text
data/legacy/dag_inspection/dag_0000_full_details.csv
data/legacy/dag_inspection/dag_0000_detailed.png
```

The CSV exposes one row per task with topology, deadlines, reference schedule/core allocation and all A7/A12 DVFS fields; the PNG renders the complete directed DAG with key values inside each node.

---

## 3. Dynamic muxGNN route

### Graph and dataset

The muxGNN route uses a DAGBench Cholesky workflow together with the existing project task profiles. Dynamic event-driven scenarios contain three node types:

- `task`
- `vehicle`
- `edge_server`

Five graph relations are built:

1. task dependency
2. local execution
3. V2V offloading
4. V2I offloading
5. gang/resource contention

The dynamic snapshots are stored under:

```text
data/muxgnn/dynamic/
  train/
  val/
  test/
  manifest.json
```

The default dataset contains 18 training scenarios, 3 validation scenarios, and 3 test scenarios.

### Offline graph training (important)

The graph encoder is **not trained end-to-end with PPO anymore**.

The training process is now explicitly two-stage:

```text
Saved dynamic graph snapshots
        |
        v
Offline muxGNN encoder pretraining
(masked node-feature reconstruction
 + relation reconstruction)
        |
        v
muxgnn_encoder.pt
        |
        v
Load + freeze encoder
        |
        v
PPO trains scheduling heads + Critic + migration Transformer
```

During PPO training:

- muxGNN encoder parameters have `requires_grad=False`;
- the encoder stays in `eval()` mode;
- encoder forward passes run under `torch.no_grad()`;
- encoder parameters are excluded from the PPO optimizer;
- the scheduling head, Critic, pair encoder, and migration Transformer remain trainable.

This removes PPO backpropagation through the graph-processing path.

### Prepare/validate the graph dataset

If the included dynamic dataset is already present, this step can be skipped.

```bash
python run_muxgnn.py prepare
```

Equivalent modules:

```bash
python -m muxgnn.graph.generate
python -m muxgnn.graph.validate
python -m muxgnn.graph.generate_dynamic_dataset
python -m muxgnn.graph.validate_dynamic_dataset
```

### Stage 1 — offline muxGNN pretraining

```bash
python run_muxgnn.py pretrain --epochs 10 --batch-size 4
```

Equivalent direct command:

```bash
python -m muxgnn.rl.offline_pretrain --epochs 10 --batch-size 4
```

Outputs:

```text
artifacts/muxgnn/offline_encoder/
  muxgnn_encoder.pt
  pretraining_config.json
```

### Stage 2 — PPO training with frozen muxGNN + runtime migration

```bash
python run_muxgnn.py train --timesteps 200000 --skip-evaluation
```

The online policy now has two masked action families:

```text
Schedule:  (ready task, resource, A7/A12)
Migration: (running task, target resource, A7/A12)
```

PPO therefore chooses the processor type directly. Runtime migration windows are
opened periodically and after task-completion events. For each eligible running
task the action set contains an explicit KEEP choice plus feasible migrations.
A trainable Transformer compares the destination alternatives using the frozen
muxGNN task/resource embeddings together with runtime migration features such as
progress, remaining work, checkpoint size, link rate, contact duration, target
load, transfer delay, and migration energy.

The graph schema and offline muxGNN checkpoint remain unchanged; migration-only
state is added to the online PPO observation. Because the source data has no
checkpoint/state-size field, checkpoint size is generated deterministically from
task cost and gang width using the constants in `muxgnn/rl/config.py`.

Migration runtime assumptions are explicit and configurable: all running tasks are
treated as preemptible, migration state is checkpointed and transferred over the
current V2V/V2I link, transfer feasibility requires sufficient contact duration,
and the destination reserves the required homogeneous gang while transfer/resume
is in progress. Preemption, serialization, transfer, resume delay, migration
energy, cooldown, minimum executed runtime, and maximum migrations per task are
all configured in `muxgnn/rl/config.py`.

Main PPO output:

```text
artifacts/muxgnn/models/ppo_muxgnn.zip
```

TensorBoard logs:

```text
artifacts/muxgnn/logs/tensorboard/
```

### Run both training stages in sequence

```bash
python run_muxgnn.py all --epochs 10 --timesteps 200000 --skip-evaluation
```

### Evaluation

```bash
python run_muxgnn.py evaluate
```

or:

```bash
python -m muxgnn.rl.evaluate
```

Evaluation compares PPO+muxGNN against the project HEFT baseline and writes CSV/JSON summaries and plots under:

```text
artifacts/muxgnn/evaluation/
```

---

## 4. muxGNN data flow

```text
DynamicScenarioState
        |
        v
build_dynamic_hetero_graph()
        |
        v
5-relation PyG HeteroData
        |
        v
HeteroMuxAdapter
        |
        v
fixed-size dense tensors
        |
        v
Frozen offline-pretrained muxGNN encoder
        |
        +--------------------+
        |                    |
        v                    v
 task/resource embeddings   global embedding
        |                    |
        +---------+----------+
                  |
                  v
            PPO Actor/Critic
                  |
          +-------+--------+
          |                |
          v                v
 schedule head      migration Transformer
(task,res,A7/A12)   (running task,target,A7/A12)
          |                |
          +-------+--------+
                  |
                  v
        event-driven environment
```

Key implementation files:

```text
muxgnn/graph/dynamic_builder.py        dynamic graph construction
muxgnn/graph/event_driven_simulation.py dataset-generation simulator
muxgnn/rl/adapter.py         HeteroData -> fixed PPO tensors
muxgnn/rl/model.py           muxGNN encoder + Actor/Critic
muxgnn/rl/offline_pretrain.py offline graph-encoder training
muxgnn/rl/policy.py          Stable-Baselines3 policy integration
muxgnn/rl/env.py             PPO scheduling environment
muxgnn/rl/train.py           frozen-encoder PPO training
muxgnn/rl/evaluate.py        HEFT vs PPO+muxGNN evaluation
```

## 5. Recommended overnight run

With the dynamic dataset already generated:

```bash
python run_muxgnn.py all --epochs 10 --timesteps 200000 --skip-evaluation
```

After training:

```bash
python run_muxgnn.py evaluate
```


## SUMO 6 vehicles / 3 Edge servers
See [RUN_SUMO_3EDGE_FA.md](RUN_SUMO_3EDGE_FA.md). Existing v4 data is preserved.


### SUMO ×5 reproducible variant (31 Oct 2026)
See `RUN_SUMO_3EDGE_FA.md` for seeded random depart positions (`42`), 0.05 s SUMO steps, and the separately derived 5x runtime/deadline/energy scenario. Original v4 inputs and existing muxGNN depth/reward are preserved.
