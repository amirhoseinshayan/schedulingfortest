# Legacy-native Dynamic MAGNET v3

This directory is reserved for the v3 event-driven multi-snapshot dataset.
The ZIP intentionally does not ship pre-serialized `.pt` snapshots; generate
them with the real `torch-geometric` installation in the project environment.

Build:

```bash
python -m muxgnn.graph.generate_v3_dataset
```

Validate:

```bash
python -m muxgnn.graph.validate_v3_dataset
```

The generator replaces only this v3 directory. It leaves the v2 dataset,
PPO/muxGNN training code, reward, HEFT baseline, and evaluation pipeline
unchanged.

Expected design after generation:

- 24 scenario trajectories: 18 train / 3 validation / 3 test.
- 121 event-driven snapshots per scenario, 2904 snapshots total for the current
  deterministic source DAG and behavior-policy set.
- The exact validated 60-task / 126-edge Legacy DAG is reused.
- Task ready/running/completed state, free A7/A12 cores, candidate feasibility,
  and contention are dynamic.
- Candidate relation topology remains stable and is based on static reachability
  plus hardware capability; temporary core shortage changes feasibility features
  instead of deleting the relation.
- No DAGBench, mobility, physical network delay, queue delay, or migration is
  introduced in v3.
- Time remains canonical relative simulation time; physical seconds are not
  claimed.
