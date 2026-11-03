from .settings import PROJECT_ROOT


# ============================================================
# Output
# ============================================================

DYNAMIC_OUTPUT_DIR = PROJECT_ROOT / "data" / "muxgnn" / "dynamic"
MANIFEST_PATH = DYNAMIC_OUTPUT_DIR / "manifest.json"


# ============================================================
# Dataset size
# ============================================================

NUM_TRAIN_SCENARIOS = 18
NUM_VAL_SCENARIOS = 3
NUM_TEST_SCENARIOS = 3


# ============================================================
# Reproducibility
# ============================================================

BASE_SEED = 1000


# ============================================================
# Event-driven simulation
# ============================================================

# Floating-point tolerance for simultaneous events.
EVENT_TIME_EPSILON_SEC = 1e-9

# Safety guard against accidental infinite event loops.
MAX_DECISION_EPOCHS = 10000

# Minimum positive task duration used by the reference dispatcher.
MIN_TASK_DURATION_SEC = 1e-6


# ============================================================
# Mobility
# ============================================================

MIN_VEHICLE_SPEED_MPS = 8.0
MAX_VEHICLE_SPEED_MPS = 25.0


# ============================================================
# Background resource load
# ============================================================

# Background load is sampled once when a scenario starts.
# It is NOT changed periodically. Runtime resource changes then happen
# only through event-driven task start/completion operations.
MIN_RESOURCE_LOAD = 0.05
MAX_RESOURCE_LOAD = 0.75

MAX_VEHICLE_QUEUE_DELAY_SEC = 2.0
MAX_EDGE_QUEUE_DELAY_SEC = 4.0


# ============================================================
# Communication normalization
# ============================================================

CONTACT_DURATION_NORM_SEC = 60.0
TX_DELAY_NORM_SEC = 0.1