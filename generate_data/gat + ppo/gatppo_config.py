from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATA_DIR = PROJECT_ROOT / "data" / "legacy"
RAW_RESULTS_ROOT = LEGACY_DATA_DIR / "raw_results"
LEGACY_OUTPUT_CSV = LEGACY_DATA_DIR / "dag_dataset_gang_a7_a12.csv"
LEGACY_METADATA_JSON = LEGACY_DATA_DIR / "dataset_generation_metadata.json"
LEGACY_PROFILE_REPORT_CSV = LEGACY_DATA_DIR / "benchmark_profiles_a7_a12.csv"
LEGACY_DAG_INSPECTION_DIR = LEGACY_DATA_DIR / "dag_inspection"

# Keep the existing dataset shape/selection so the training/evaluation code does
# not need to change.
WORKLOAD_GROUP = "60"
NUM_DAGS = 1
NUM_SUBTASKS = 60
# Sparse layered-DAG topology requested for the GAT/PPO dataset.
# With 60 tasks, eight balanced topological layers keep the graph readable
# while still allowing parallelism. Edges are allowed ONLY from layer k to
# layer k+1; there are no skip connections to deeper layers.
NUM_TOPOLOGICAL_LAYERS = 8
EDGE_PROB_RANGES = (
    (0.06, 0.10),
)
SEED = 42

# Raw-result timing contract.
# The CSVs contain a contiguous sample index (#), not timestamps.  Because the
# archive has no acquisition-period metadata, the default generator MUST NOT
# claim that one row equals a physical second.  It uses a deterministic
# canonical relative time base derived from the matched A12 traces instead.
# A real wall-clock sampling interval can still be supplied explicitly through
# --sample-interval-sec if verified measurement metadata becomes available.
TIMING_MODE_DEFAULT = "canonical_relative_time"

# Preserve the previous slack factor, but apply it to a feasible DAG-aware
# reference completion time instead of to each task's isolated runtime at t=0.
DEADLINE_FACTOR = 0.9

M_I_VALUES = (1, 2, 3, 4, 5, 6)
M_I_BINOMIAL_WEIGHTS = (1, 5, 10, 10, 5, 1)
# Symmetric Binomial(5, 0.5), shifted from k=0..5 to m_i=1..6.
# Exact probabilities: 3.125%, 15.625%, 31.25%, 31.25%, 15.625%, 3.125%.
# For a finite DAG we convert these probabilities to integer quotas with the
# largest-remainder method, then shuffle the resulting m_i assignments.
M_I_DISTRIBUTION_NAME = "shifted_symmetric_binomial_n5_p0.5"

# VF levels are intentionally unchanged from the existing implementation.
VF_LEVELS_A12 = [
    {"level": 1, "frequency": 0.8, "voltage": 0.85},
    {"level": 2, "frequency": 1.0, "voltage": 0.90},
    {"level": 3, "frequency": 1.2, "voltage": 0.95},
    {"level": 4, "frequency": 1.4, "voltage": 1.05},
    {"level": 5, "frequency": 1.6, "voltage": 1.10},
    {"level": 6, "frequency": 1.8, "voltage": 1.15},
]

VF_LEVELS_A7 = [
    {"level": 1, "frequency": 0.5, "voltage": 0.85},
    {"level": 2, "frequency": 0.7, "voltage": 0.90},
    {"level": 3, "frequency": 0.9, "voltage": 0.95},
    {"level": 4, "frequency": 1.1, "voltage": 1.05},
    {"level": 5, "frequency": 1.3, "voltage": 1.10},
    {"level": 6, "frequency": 1.4, "voltage": 1.15},
]
