# PPO Gang Scheduler — Without GNN Ablation

This folder is a controlled ablation of `legacy_gat` requested for comparison.

## Fair-comparison contract

The following are **shared unchanged** with `legacy_gat`:

- exact dataset file: `data/legacy/dag_dataset_gang_a7_a12.csv`
- exact `legacy_gat.env.DagSchedulingEnv`
- DAG, deadlines, A7/A12 profiles and Gang `m_i`
- action space and action mask
- runtime/Gang model
- Reward and fixed reward scaling
- QoS definition
- physical hardware (6 A7 + 6 A12)
- PPO hyperparameters
- seed
- exploration-alpha schedule
- evaluation defaults
- HEFT_GANG baseline and rollout helper

The **only intended model difference** is graph processing:

- `legacy_gat`: two GAT/message-passing blocks consume `edge_index`/`edge_mask`.
- `without-gnn`: two independent per-node residual MLP blocks are used instead.
  `edge_index` and `edge_mask` remain in the observation for environment/API parity,
  but the policy deliberately ignores them.

After node encoding, the no-GNN route keeps the same mean+max pooling idea,
resource/global encoders, allocation features, actor head, critic head, action
masking and alpha exploration mechanism.

## Commands (Windows)

From `scheduling_utils`:

```powershell
.\without-gnn\make.bat validate
.\without-gnn\make.bat pilot
.\without-gnn\make.bat train
.\without-gnn\make.bat evaluate
```

The full comparison training uses the same `200000` timesteps as GAT.
Artifacts are isolated in `artifacts/without_gnn` so GAT results are never
overwritten.

## Interpretation

- If without-GNN performs similarly badly on QoS, the evidence points away from
  GAT and toward observation/action/reward/environment design.
- If without-GNN is materially better, inspect GAT message passing/pooling.
- If GAT is materially better, graph structure is contributing useful signal.
