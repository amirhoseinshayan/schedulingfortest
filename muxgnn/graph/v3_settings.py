from __future__ import annotations

from .settings import PROJECT_ROOT


# ============================================================
# Legacy-native Dynamic MAGNET v3 output
# ============================================================

V3_DATASET_VERSION = "legacy_native_dynamic_magnet_v3"
V3_OUTPUT_DIR = PROJECT_ROOT / "data" / "muxgnn" / V3_DATASET_VERSION
V3_MANIFEST_PATH = V3_OUTPUT_DIR / "manifest.json"
V3_METADATA_PATH = V3_OUTPUT_DIR / "dataset_metadata.json"
V3_README_PATH = V3_OUTPUT_DIR / "README.md"
V3_SUMMARY_CSV_PATH = V3_OUTPUT_DIR / "dataset_summary.csv"


# ============================================================
# Scenario split
# ============================================================

# Split is performed at scenario level.  Snapshots from one trajectory never
# appear in more than one split.
V3_NUM_TRAIN_SCENARIOS = 18
V3_NUM_VAL_SCENARIOS = 3
V3_NUM_TEST_SCENARIOS = 3
V3_TOTAL_SCENARIOS = (
    V3_NUM_TRAIN_SCENARIOS
    + V3_NUM_VAL_SCENARIOS
    + V3_NUM_TEST_SCENARIOS
)

V3_BASE_SEED = 3000

# Independent, non-learning behavior policies are used only to generate valid
# state trajectories for self-supervised graph pretraining.  They are NOT PPO
# and they are NOT the HEFT evaluation baseline.
V3_BEHAVIOR_POLICIES = (
    "earliest_finish",
    "edf",
    "lowest_energy",
)


# ============================================================
# Controlled scenario diversity
# ============================================================

# These values define fixed scenario-level background reservations.  They are
# synthetic simulation assumptions, sampled once per scenario and then kept
# fixed.  Runtime changes after that are caused only by task start/completion.
# Total hardware remains 6xA7 + 6xA12 per vehicle and 12xA7 + 12xA12 at the
# edge server, as defined by settings.py.
V3_SOURCE_MIN_FREE_CORES = 4
V3_REACHABLE_V2V_MIN_FREE_CORES = 2
V3_OTHER_VEHICLE_MIN_FREE_CORES = 1
V3_EDGE_MIN_FREE_CORES = 8


# ============================================================
# Event-driven simulation safeguards
# ============================================================

V3_EVENT_TIME_EPSILON = 1e-9
V3_MAX_DECISION_EPOCHS = 1000
V3_MIN_POSITIVE_DURATION = 1e-9
