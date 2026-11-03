from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import List

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from _bootstrap import PROJECT_ROOT  # noqa: F401
from config import (
    ALLOW_MIXED_GANGS,
    ARTIFACT_DIR,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_FREQUENCY,
    CLIP_RANGE,
    DATASET_PATH,
    ENT_COEF,
    EXPLORATION_ALPHA_END,
    EXPLORATION_ALPHA_START,
    GAE_LAMBDA,
    GAMMA,
    LEARNING_RATE,
    LOG_DIR,
    MAX_GANG_SIZE,
    MAX_GRAD_NORM,
    MODEL_PATH,
    N_EPOCHS,
    N_STEPS,
    NUM_A12_CORES,
    NUM_A7_CORES,
    QOS_FACTOR,
    REWARD_MODE,
    REWARD_PROPOSAL,
    REWARD_WEIGHTS,
    RUNTIME_MODEL,
    SEED,
    TENSORBOARD_DIR,
    TOTAL_TIMESTEPS,
    VF_COEF,
)
from legacy_gat.env import DagSchedulingEnv
from policy import MaskedNoGNNActorCriticPolicy


class ExplorationAndGangStatsCallback(BaseCallback):
    """Same linear alpha schedule and Gang logging as legacy_gat.train."""

    def __init__(
        self,
        alpha_start: float,
        alpha_end: float,
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.total_training_timesteps = max(int(total_timesteps), 1)
        self.widths: List[float] = []
        self.a12_fractions: List[float] = []
        if not 0.0 <= self.alpha_end <= self.alpha_start <= 1.0:
            raise ValueError("Expected 0 <= alpha_end <= alpha_start <= 1.")

    def _current_alpha(self) -> float:
        progress = min(
            max(self.num_timesteps / self.total_training_timesteps, 0.0), 1.0
        )
        return self.alpha_start + progress * (self.alpha_end - self.alpha_start)

    def _apply_alpha(self) -> None:
        alpha = self._current_alpha()
        self.model.policy.set_exploration_alpha(alpha)
        self.logger.record("exploration/alpha", alpha)

    def _on_training_start(self) -> None:
        self._apply_alpha()

    def _on_rollout_start(self) -> None:
        self._apply_alpha()
        self.widths = []
        self.a12_fractions = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("invalid_action", False):
                continue
            width = info.get("gang_width")
            if width is not None:
                self.widths.append(float(width))
                a12_count = float(info.get("a12_count", 0.0))
                self.a12_fractions.append(a12_count / max(float(width), 1.0))
        return True

    def _on_rollout_end(self) -> None:
        if self.widths:
            widths = np.asarray(self.widths, dtype=np.float64)
            self.logger.record("gang/usage_rate", float(np.mean(widths > 1.0)))
            self.logger.record("gang/mean_width", float(np.mean(widths)))
            self.logger.record("gang/max_width", float(np.max(widths)))
        if self.a12_fractions:
            self.logger.record(
                "gang/mean_a12_fraction", float(np.mean(self.a12_fractions))
            )
        self.logger.record("gang/skipped_oversized", 0.0)


def progress_bar_dependencies_available() -> bool:
    try:
        import rich  # noqa: F401
        from tqdm.rich import tqdm as _rich_tqdm  # noqa: F401
    except ImportError:
        return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def make_env(seed: int = SEED) -> DagSchedulingEnv:
    # Intentionally identical to legacy_gat.train.make_env().
    return DagSchedulingEnv(
        csv_path=DATASET_PATH,
        row_index=0,
        num_a7_cores=NUM_A7_CORES,
        num_a12_cores=NUM_A12_CORES,
        max_gang_size=MAX_GANG_SIZE,
        allow_mixed_gangs=ALLOW_MIXED_GANGS,
        runtime_model=RUNTIME_MODEL,
        reward_mode=REWARD_MODE,
        reward_weights=REWARD_WEIGHTS,
        invalid_action_penalty=2.0,
        sample_dags=False,
        seed=seed,
        qos_factor=QOS_FACTOR,
        reward_proposal=REWARD_PROPOSAL,
    )


def save_run_config(env: DagSchedulingEnv, total_timesteps: int, seed: int) -> Path:
    observation_shapes = {
        key: list(space.shape) if getattr(space, "shape", None) is not None else None
        for key, space in env.observation_space.spaces.items()
    }
    config = {
        "ablation": "WITHOUT_GNN",
        "only_intended_difference": "GAT/message passing replaced by per-node MLP blocks; edge_index/edge_mask ignored by policy",
        "dataset": str(DATASET_PATH),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "row_index": 0,
        "task_count": int(env.num_tasks),
        "edge_count": int(len(env.edges)),
        "max_m_i": int(np.max(env.m_i)),
        "num_a7_cores": NUM_A7_CORES,
        "num_a12_cores": NUM_A12_CORES,
        "max_gang_size": MAX_GANG_SIZE,
        "allow_mixed_gangs": ALLOW_MIXED_GANGS,
        "runtime_model": RUNTIME_MODEL,
        "reward_mode": REWARD_MODE,
        "reward_proposal": REWARD_PROPOSAL,
        "reward_weights": list(REWARD_WEIGHTS),
        "qos_factor": QOS_FACTOR,
        "action_dimension": int(env.action_space.n),
        "num_action_slots": int(env.num_action_slots),
        "observation_shapes": observation_shapes,
        "ppo": {
            "total_timesteps": int(total_timesteps),
            "learning_rate": LEARNING_RATE,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "alpha_schedule": {
            "start": EXPLORATION_ALPHA_START,
            "end": EXPLORATION_ALPHA_END,
            "mode": "linear_at_rollout_boundaries",
        },
        "no_gnn_policy": {
            "node_hidden_dim": 64,
            "node_mlp_layers": 2,
            "core_hidden_dim": 32,
            "global_hidden_dim": 32,
            "allocation_hidden_dim": 32,
            "context_hidden_dim": 64,
            "graph_pool_dim": 128,
            "actor_hidden_dim": 128,
            "critic_hidden_dim": 128,
            "dropout": 0.10,
            "uses_edge_index": False,
            "uses_edge_mask": False,
        },
        "seed": int(seed),
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    }
    path = ARTIFACT_DIR / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO Gang scheduler WITHOUT GNN (controlled ablation)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Training steps. Use 20000 for pilot, then 200000 for full comparison.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.timesteps <= 0 or args.timesteps > 200_000:
        raise ValueError("--timesteps must be in 1..200000.")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Gang dataset not found: {DATASET_PATH}\n"
            "Generate/validate the shared dataset first with the legacy_gat pipeline."
        )

    for directory in [
        ARTIFACT_DIR,
        MODEL_PATH.parent,
        CHECKPOINT_DIR,
        LOG_DIR,
        TENSORBOARD_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    env = make_env(seed=args.seed)
    print("\n=== PPO WITHOUT-GNN GANG TRAINING ===")
    print(f"Dataset              : {DATASET_PATH.name}")
    print(f"Tasks / edges        : {env.num_tasks} / {len(env.edges)}")
    print(f"max(m_i)             : {int(np.max(env.m_i))}")
    print(f"Physical hardware    : {NUM_A7_CORES} A7 + {NUM_A12_CORES} A12")
    print(f"Action slots/task    : {env.num_action_slots}")
    print(f"Action dimension     : {env.action_space.n}")
    print(f"Runtime model        : {RUNTIME_MODEL}")
    print(f"Reward mode          : {REWARD_MODE}")
    print(f"QoS factor           : {QOS_FACTOR}")
    print("Graph message passing: DISABLED")
    print(
        f"Exploration alpha    : {EXPLORATION_ALPHA_START} -> {EXPLORATION_ALPHA_END}"
    )
    print(f"Total timesteps      : {args.timesteps}\n")

    check_env(env, warn=True, skip_render_check=True)
    config_path = save_run_config(env, args.timesteps, args.seed)
    print(f"Run config saved     : {config_path}\n")

    monitored_env = Monitor(env, filename=str(LOG_DIR / "monitor.csv"))
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_no_gnn_gang",
        verbose=1,
    )
    exploration_callback = ExplorationAndGangStatsCallback(
        alpha_start=EXPLORATION_ALPHA_START,
        alpha_end=EXPLORATION_ALPHA_END,
        total_timesteps=args.timesteps,
        verbose=0,
    )
    callbacks = CallbackList([exploration_callback, checkpoint_callback])

    model = PPO(
        policy=MaskedNoGNNActorCriticPolicy,
        env=monitored_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        seed=args.seed,
        device="cpu",
        tensorboard_log=str(TENSORBOARD_DIR),
        policy_kwargs={
            "node_hidden_dim": 64,
            "node_mlp_layers": 2,
            "core_hidden_dim": 32,
            "global_hidden_dim": 32,
            "allocation_hidden_dim": 32,
            "context_hidden_dim": 64,
            "graph_pool_dim": 128,
            "actor_hidden_dim": 128,
            "critic_hidden_dim": 128,
            "dropout": 0.10,
            "exploration_alpha": EXPLORATION_ALPHA_START,
        },
    )

    use_progress_bar = progress_bar_dependencies_available()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=use_progress_bar,
        )
        model.policy.set_exploration_alpha(EXPLORATION_ALPHA_END)
        model.save(str(MODEL_PATH))
        print("\n=== Training completed ===")
        print(f"Saved model: {MODEL_PATH}")
    finally:
        monitored_env.close()


if __name__ == "__main__":
    main()
