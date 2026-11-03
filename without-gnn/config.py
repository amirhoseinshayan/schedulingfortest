from __future__ import annotations

from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401
import legacy_gat.config as _base

# Re-export every experiment constant from legacy_gat so the ablation uses the
# exact same dataset, hardware, reward, QoS, PPO, exploration and evaluation
# settings. Only artifact/model output paths are separated to avoid collisions.
for _name in dir(_base):
    if _name.isupper():
        globals()[_name] = getattr(_base, _name)

PACKAGE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "without_gnn"
MODEL_DIR = ARTIFACT_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = ARTIFACT_DIR / "logs"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
EVAL_DIR = ARTIFACT_DIR / "evaluation"
MODEL_PATH = MODEL_DIR / "ppo_no_gnn_gang.zip"
