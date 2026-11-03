"""Versioned changes; old graph.settings and v4 files are NOT modified."""
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
NUM_VEHICLES=16
SOURCE_VEHICLE_ID=0
VEHICLE_A7_CORES=VEHICLE_A12_CORES=6
EDGE_A7_CORES=EDGE_A12_CORES=12
MAX_GANG_SIZE=6
NUM_EDGE_SERVERS = 4
V2V_REACHABLE_VEHICLE_IDS = tuple(range(1, NUM_VEHICLES))
DATASET_VERSION = "v4_mobility_16vehicle_4edge_x5_seed42_step005_7dag_v2"
DATASET_DIR = PROJECT_ROOT / 'data' / 'muxgnn' / DATASET_VERSION
SUMO_DIR = PROJECT_ROOT / 'sumo_three_edge'
V2V_RANGE_M = 150.0
V2I_RANGE_M = 300.0

# Derived 5x duration scenario. Reference v4 source files remain byte-identical.
TIME_SCALE_FACTOR = 5.0
SUMO_STEP_LENGTH_SEC = 0.05
SUMO_RANDOM_SEED = 42
# Conservative trace margin; workflows that cannot finish still fail explicitly.
MIN_TRACE_TAIL_SEC = 150.0
