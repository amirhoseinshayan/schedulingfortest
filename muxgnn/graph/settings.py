from pathlib import Path

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LEGACY_DATASET_PATH = (
    PROJECT_ROOT / "data" / "legacy" / "dag_dataset_gang_a7_a12.csv"
)

LEGACY_ROW_INDEX = 0

# The new muxGNN dataset is intentionally isolated from the old
# DAGBench/dynamic prototype.  The CSV tables are the human-readable
# source artifacts; hetero_graph.pt is the PyG derivative generated locally.
DATASET_VERSION = "legacy_native_simple_magnet_v2"
OUTPUT_DIR = PROJECT_ROOT / "data" / "muxgnn" / DATASET_VERSION
TABLE_DIR = OUTPUT_DIR / "tables"

GRAPH_PATH = OUTPUT_DIR / "hetero_graph.pt"
METADATA_PATH = OUTPUT_DIR / "dataset_metadata.json"
README_PATH = OUTPUT_DIR / "README.md"


# ============================================================
# Simple-v2 controlled resource model
# ============================================================

NUM_VEHICLES = 6
NUM_EDGE_SERVERS = 1
SOURCE_VEHICLE_ID = 0

VEHICLE_A7_CORES = 6
VEHICLE_A12_CORES = 6

# v2 intentionally gives the six vehicles different initial free-core states.
# Total hardware remains 6xA7 + 6xA12 per vehicle; only availability differs.
# The source vehicle remains fully available so every Legacy task has a local
# fallback.  These are controlled simulation states, not measured loads.
VEHICLE_AVAILABLE_A7 = (6, 4, 3, 5, 2, 6)
VEHICLE_AVAILABLE_A12 = (6, 2, 6, 4, 3, 1)

# v2 keeps network physics out of the dataset, but removes the maximum-connectivity
# assumption.  Since every task originates at Vehicle 0, only these non-source
# vehicles are declared statically reachable V2V candidates in this base graph.
# Relation existence encodes this controlled reachability; no distance/rate is
# claimed yet.
V2V_REACHABLE_VEHICLE_IDS = (1, 2)

EDGE_A7_CORES = 12
EDGE_A12_CORES = 12

MAX_GANG_SIZE = 6
SEED = 42

# Kept only for compatibility with dormant old dynamic modules.  They are
# NOT used by Legacy-native Simple MAGNET Heterogeneous Graph v2.
ROAD_LENGTH_M = 1000.0
V2V_RANGE_M = 350.0
V2I_RANGE_M = 700.0
BASE_V2V_RATE_MBPS = 600.0
BASE_V2I_RATE_MBPS = 1000.0
