from __future__ import annotations

from .settings import PROJECT_ROOT


# ============================================================
# Finalized physical-time dynamic MAGNET dataset (v4)
# ============================================================

V4_DATASET_VERSION = "legacy_native_dynamic_magnet_v4_physical_time"
V4_OUTPUT_DIR = PROJECT_ROOT / "data" / "muxgnn" / V4_DATASET_VERSION
V4_MANIFEST_PATH = V4_OUTPUT_DIR / "manifest.json"
V4_METADATA_PATH = V4_OUTPUT_DIR / "dataset_metadata.json"
V4_README_PATH = V4_OUTPUT_DIR / "README.md"
V4_SUMMARY_CSV_PATH = V4_OUTPUT_DIR / "dataset_summary.csv"
V4_SOURCE_TASKS_CSV_PATH = V4_OUTPUT_DIR / "source_task_profiles.csv"
V4_DEPENDENCY_CSV_PATH = V4_OUTPUT_DIR / "dependency_edges.csv"
V4_DEADLINE_AUDIT_CSV_PATH = V4_OUTPUT_DIR / "deadline_audit.csv"

# The old benchmark profile report still names these columns "sample_count".
# The supervisor has clarified that those raw numeric execution values are in ms.
V4_BENCHMARK_PROFILE_PATH = PROJECT_ROOT / "data" / "legacy" / "benchmark_profiles_a7_a12.csv"
V4_LEGACY_DAG_PATH = PROJECT_ROOT / "data" / "legacy" / "dag_dataset_gang_a7_a12.csv"

RAW_EXECUTION_TIME_UNIT = "millisecond"
TIME_UNIT = "second"
MILLISECONDS_PER_SECOND = 1000.0

# Single-DAG design-time assumption confirmed by the supervisor:
# the complete task set is known at t=0, so all external release times are 0.
V4_RELEASE_TIME_SEC = 0.0

# Deadline model for the final dataset is intentionally independent of HEFT/PPO.
# We preserve the original 15% slack concept, but apply it to DAG-aware earliest
# completion progress with w_i=min(T_A7_i,T_A12_i), without division by m_i.
V4_DEADLINE_SLACK_FACTOR = 1.15
V4_DEADLINE_MODEL = "critical_path_proportional_absolute_subdeadlines"

# The old 60-task Legacy dataset used a canonical scaling of raw_count / 43.
# This constant is used only to convert the old legacy deadline into physical
# seconds for an audit column; it is NOT used by the new v4 deadline model.
V4_LEGACY_CANONICAL_REFERENCE_RAW_COUNT = 43.0


# ============================================================
# Scenario split (same controlled v3 design; no split leakage)
# ============================================================

V4_NUM_TRAIN_SCENARIOS = 18
V4_NUM_VAL_SCENARIOS = 3
V4_NUM_TEST_SCENARIOS = 3
V4_TOTAL_SCENARIOS = (
    V4_NUM_TRAIN_SCENARIOS
    + V4_NUM_VAL_SCENARIOS
    + V4_NUM_TEST_SCENARIOS
)
V4_BASE_SEED = 4000

# These are trajectory-generation policies only. They are not PPO and not the
# HEFT evaluation baseline.
V4_BEHAVIOR_POLICIES = (
    "earliest_finish",
    "edf",
    "lowest_energy",
)


# ============================================================
# Controlled resource-state diversity
# ============================================================

V4_SOURCE_MIN_FREE_CORES = 4
V4_REACHABLE_V2V_MIN_FREE_CORES = 2
V4_OTHER_VEHICLE_MIN_FREE_CORES = 1
V4_EDGE_MIN_FREE_CORES = 8


# ============================================================
# Event-driven safeguards
# ============================================================

V4_EVENT_TIME_EPSILON_SEC = 1e-12
V4_MAX_DECISION_EPOCHS = 1000
V4_MIN_POSITIVE_DURATION_SEC = 1e-12
