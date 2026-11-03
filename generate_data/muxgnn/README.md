# muxGNN generator — phase 2

This directory is reserved for the muxGNN + PPO dataset generator.

Phase 1 is deliberately completed first under the sibling folder:

```text
generate_data/gat + ppo/
```

When Phase 2 is implemented, it should reuse the corrected raw A7/A12 timing/provenance semantics from Phase 1 rather than reading the raw `#` sample counter as seconds.
