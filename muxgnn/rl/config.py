from __future__ import annotations

from pathlib import Path
import os

from ..graph.settings import PROJECT_ROOT, NUM_VEHICLES, NUM_EDGE_SERVERS


# ============================================================
# Fixed padded dimensions used by SB3
# ============================================================

# Legacy-native v2 contains 60 tasks.  A small padding margin keeps the
# observation contract fixed for the current muxGNN smoke-train bridge.
MUX_MAX_TASKS = 64
MUX_MAX_VEHICLES = 16 if os.environ.get("MUX_MOBILITY3", "0") == "1" else NUM_VEHICLES
MUX_MAX_EDGE_SERVERS = 4 if os.environ.get("MUX_MOBILITY3", "0") == "1" else NUM_EDGE_SERVERS
MUX_MAX_RESOURCES = MUX_MAX_VEHICLES + MUX_MAX_EDGE_SERVERS
MUX_MAX_NODES = MUX_MAX_TASKS + MUX_MAX_RESOURCES

# PPO now chooses the processor type as part of each scheduling action.
MUX_PROCESSOR_TYPES = ("a7", "a12")
MUX_NUM_PROCESSOR_TYPES = len(MUX_PROCESSOR_TYPES)

# Joint runtime action layout:
#   [0, MUX_SCHEDULE_ACTIONS) -> schedule a ready task
#   [MUX_SCHEDULE_ACTIONS, MUX_TOTAL_ACTIONS) -> keep/migrate a running task
MUX_SCHEDULE_ACTIONS = (
    MUX_MAX_TASKS * MUX_MAX_RESOURCES * MUX_NUM_PROCESSOR_TYPES
)
MUX_MIGRATION_ACTIONS = MUX_SCHEDULE_ACTIONS
MUX_TOTAL_ACTIONS = MUX_SCHEDULE_ACTIONS + MUX_MIGRATION_ACTIONS


# ============================================================
# Relation / feature dimensions
# ============================================================

MUX_NUM_RELATIONS = 5

# Legacy-native v2 candidate relations contain 8 compute/resource features.
# One extra channel is reserved for internal message-flow direction.
MUX_EDGE_FEATURE_DIM = 9

# Pair features in the v2 smoke-train bridge:
#   3 execution-mode flags (local / V2V / V2I)
#   8 compute/resource candidate features
# No communication placeholders are kept in this stage.
MUX_PAIR_FEATURE_DIM = 11

MUX_GLOBAL_FEATURE_DIM = 8


# ============================================================
# muxGNN architecture
# ============================================================

MUX_HIDDEN_DIM = 96
MUX_HEADS = 4
MUX_LAYERS_PER_RELATION = 2
MUX_PAIR_HIDDEN_DIM = 48
MUX_GLOBAL_HIDDEN_DIM = 48
MUX_ACTOR_HIDDEN_DIM = 128
MUX_CRITIC_HIDDEN_DIM = 128
MUX_DROPOUT = 0.10

# Runtime migration head. The frozen muxGNN embeddings are reused as token
# context; only this lightweight Transformer and PPO heads are trainable.
MUX_MIGRATION_FEATURE_DIM = 20
MUX_MIGRATION_HIDDEN_DIM = 128
MUX_MIGRATION_HEADS = 4
MUX_MIGRATION_LAYERS = 2
MUX_MIGRATION_FF_DIM = 256


# ============================================================
# PPO
# ============================================================

MUX_SEED = 42
MUX_TOTAL_TIMESTEPS = 20_000
MUX_LEARNING_RATE = 3e-4
MUX_N_STEPS = 512
MUX_BATCH_SIZE = 128
MUX_N_EPOCHS = 6
MUX_GAMMA = 0.99
MUX_GAE_LAMBDA = 0.95
MUX_CLIP_RANGE = 0.20
MUX_ENT_COEF = 0.02
MUX_VF_COEF = 0.50
MUX_MAX_GRAD_NORM = 0.50


# ============================================================
# Reward
# ============================================================

# Match the final advisor-requested reward used by Legacy GAT + PPO:
#   normalized_energy = actual_task_energy / max_global_gang_energy
#   reward = QoS / exp(normalized_energy)
#          = QoS * exp(-normalized_energy)
# No lambda coefficient and no additional reward scaling are used.
MUX_REWARD_PROPOSAL = "B"
MUX_INVALID_ACTION_PENALTY = 2.0
MUX_QOS_FACTOR = 1.20


# ============================================================
# Runtime task migration
# ============================================================

# Migration is considered at adaptive runtime checkpoints and after completion
# events. The next periodic checkpoint is placed halfway through the currently
# predicted remaining runtime, with a minimum spacing to avoid decision storms.
MUX_MIGRATION_CHECK_REMAINING_RATIO = 0.50
MUX_MIGRATION_CHECK_MIN_INTERVAL_SEC = 5.0
MUX_MIGRATION_MIN_RUNTIME_SEC = 2.0
MUX_MIGRATION_COOLDOWN_SEC = 5.0
MUX_MIGRATION_MAX_PER_TASK = 3
MUX_MIGRATION_MIN_REMAINING_FRACTION = 0.05

# The source dataset has no checkpoint/state-size field. We therefore use a
# deterministic synthetic state model derived from task cost and gang width.
MUX_MIGRATION_STATE_BASE_MB = 8.0
MUX_MIGRATION_STATE_PER_CORE_MB = 2.0
MUX_MIGRATION_STATE_COST_MB = 16.0
MUX_MIGRATION_STATE_NORM_MB = 64.0

# Runtime checkpoint/restore and transfer model.
MUX_MIGRATION_PREEMPT_SEC = 0.02
MUX_MIGRATION_RESUME_SEC = 0.02
MUX_MIGRATION_SERIALIZE_SEC_PER_MB = 0.0005
MUX_MIGRATION_TX_ENERGY_J_PER_MB = 0.03
MUX_MIGRATION_CONTROL_POWER_W = 1.0
MUX_MIGRATION_CONTACT_SAFETY_FACTOR = 1.10
MUX_MIGRATION_TRANSFER_DELAY_NORM_SEC = 5.0


# ============================================================
# Global feature scaling
# ============================================================

MUX_SIMULATION_TIME_SCALE_SEC = 60.0


# ============================================================
# Evaluation
# ============================================================

# Kept close to the legacy evaluation workflow.
MUX_EVALUATION_RUNS = 20
MUX_EVALUATION_ALPHA = 0.01


# ============================================================
# Artifacts
# ============================================================

MUX_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "muxgnn"
MUX_MODEL_DIR = MUX_ARTIFACT_DIR / "models"
MUX_LOG_DIR = MUX_ARTIFACT_DIR / "logs"
MUX_TENSORBOARD_DIR = MUX_LOG_DIR / "tensorboard"
MUX_EVAL_DIR = MUX_ARTIFACT_DIR / "evaluation"

# The graph encoder is trained once, offline, from the saved dynamic graph
# snapshots. PPO only consumes the frozen encoder afterwards.
MUX_ENCODER_DIR = MUX_ARTIFACT_DIR / "offline_encoder"
MUX_ENCODER_CHECKPOINT_PATH = MUX_ENCODER_DIR / "muxgnn_encoder.pt"
MUX_ENCODER_PRETRAIN_CONFIG_PATH = MUX_ENCODER_DIR / "pretraining_config.json"

MUX_MODEL_PATH = MUX_MODEL_DIR / "ppo_muxgnn.zip"
MUX_TRAINING_CONFIG_PATH = MUX_ARTIFACT_DIR / "training_config.json"
