from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .config import (
    MUX_ACTOR_HIDDEN_DIM,
    MUX_BATCH_SIZE,
    MUX_CLIP_RANGE,
    MUX_CRITIC_HIDDEN_DIM,
    MUX_DROPOUT,
    MUX_ENCODER_CHECKPOINT_PATH,
    MUX_ENT_COEF,
    MUX_GAE_LAMBDA,
    MUX_GAMMA,
    MUX_GLOBAL_HIDDEN_DIM,
    MUX_HEADS,
    MUX_HIDDEN_DIM,
    MUX_LAYERS_PER_RELATION,
    MUX_LEARNING_RATE,
    MUX_MAX_GRAD_NORM,
    MUX_N_EPOCHS,
    MUX_N_STEPS,
    MUX_PAIR_HIDDEN_DIM,
    MUX_SEED,
    MUX_VF_COEF,
    MUX_ARTIFACT_DIR,
)
from .simple_env import SimpleMuxGNNSmokeEnv
from .simple_policy import SimpleMaskedMuxPolicy


SIMPLE_ARTIFACT_DIR = MUX_ARTIFACT_DIR / "simple_v2_smoke"
SIMPLE_MODEL_DIR = SIMPLE_ARTIFACT_DIR / "models"
SIMPLE_MODEL_PATH = SIMPLE_MODEL_DIR / "ppo_muxgnn_simple_v2.zip"
SIMPLE_TENSORBOARD_DIR = SIMPLE_ARTIFACT_DIR / "logs" / "tensorboard"
SIMPLE_CONFIG_PATH = SIMPLE_ARTIFACT_DIR / "training_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end PPO smoke training on the Legacy-native v2 graph with a "
            "frozen offline-pretrained muxGNN encoder. No mobility/network/migration."
        )
    )
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=MUX_SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--encoder-checkpoint", type=Path, default=MUX_ENCODER_CHECKPOINT_PATH
    )
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--exploration-alpha", type=float, default=0.0)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")
    if not 0.0 <= args.exploration_alpha <= 1.0:
        raise ValueError("--exploration-alpha must be in [0, 1].")
    if not args.encoder_checkpoint.exists():
        raise FileNotFoundError(
            "Offline muxGNN encoder checkpoint is required first.\n"
            f"Expected: {args.encoder_checkpoint}\n"
            "Run: python run_muxgnn.py pretrain --epochs 10 --batch-size 1"
        )

    raw_env = SimpleMuxGNNSmokeEnv(seed=args.seed)
    env = Monitor(raw_env)

    SIMPLE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SIMPLE_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    policy_kwargs = {
        "hidden_dim": MUX_HIDDEN_DIM,
        "heads": MUX_HEADS,
        "layers_per_relation": MUX_LAYERS_PER_RELATION,
        "pair_hidden_dim": MUX_PAIR_HIDDEN_DIM,
        "global_hidden_dim": MUX_GLOBAL_HIDDEN_DIM,
        "actor_hidden_dim": MUX_ACTOR_HIDDEN_DIM,
        "critic_hidden_dim": MUX_CRITIC_HIDDEN_DIM,
        "dropout": MUX_DROPOUT,
        "exploration_alpha": float(args.exploration_alpha),
        "freeze_graph_encoder": True,
    }

    model = PPO(
        policy=SimpleMaskedMuxPolicy,
        env=env,
        learning_rate=MUX_LEARNING_RATE,
        n_steps=MUX_N_STEPS,
        batch_size=MUX_BATCH_SIZE,
        n_epochs=MUX_N_EPOCHS,
        gamma=MUX_GAMMA,
        gae_lambda=MUX_GAE_LAMBDA,
        clip_range=MUX_CLIP_RANGE,
        ent_coef=MUX_ENT_COEF,
        vf_coef=MUX_VF_COEF,
        max_grad_norm=MUX_MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(SIMPLE_TENSORBOARD_DIR),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    model.policy.load_pretrained_encoder(args.encoder_checkpoint)

    encoder_parameters = list(model.policy.scheduler_net.encoder.parameters())
    if any(parameter.requires_grad for parameter in encoder_parameters):
        raise RuntimeError("muxGNN encoder must be frozen during simple PPO smoke training.")
    optimizer_ids = {
        id(parameter)
        for group in model.policy.optimizer.param_groups
        for parameter in group["params"]
    }
    if any(id(parameter) in optimizer_ids for parameter in encoder_parameters):
        raise RuntimeError("Frozen muxGNN encoder unexpectedly appears in PPO optimizer.")

    config = {
        "stage": "legacy_native_v2_simple_end_to_end_ppo_smoke",
        "paper_grade_result": False,
        "purpose": "functional end-to-end validation only",
        "dataset": "legacy_native_simple_magnet_v2",
        "static_graph_count": 1,
        "seed": int(args.seed),
        "timesteps": int(args.timesteps),
        "device": args.device,
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "encoder_frozen": True,
        "action": "ready_task_x_static_candidate_resource_x_{A7,A12}",
        "runtime": "minimal event-driven compute-only scheduling",
        "reward": "QoS / exp(normalized_energy)",
        "intentionally_absent": [
            "DAGBench",
            "mobility",
            "network delay",
            "queue delay",
            "migration",
            "multi-DAG train/val/test generalization",
        ],
        "ppo": {
            "learning_rate": MUX_LEARNING_RATE,
            "n_steps": MUX_N_STEPS,
            "batch_size": MUX_BATCH_SIZE,
            "n_epochs": MUX_N_EPOCHS,
            "gamma": MUX_GAMMA,
            "gae_lambda": MUX_GAE_LAMBDA,
            "clip_range": MUX_CLIP_RANGE,
            "ent_coef": MUX_ENT_COEF,
            "vf_coef": MUX_VF_COEF,
            "max_grad_norm": MUX_MAX_GRAD_NORM,
            "exploration_alpha": float(args.exploration_alpha),
        },
    }
    SIMPLE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SIMPLE_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    callback = CheckpointCallback(
        save_freq=max(int(args.checkpoint_freq), 1),
        save_path=str(SIMPLE_MODEL_DIR / "checkpoints"),
        name_prefix="ppo_muxgnn_simple_v2",
    )

    print("============================================")
    print("Simple v2 muxGNN + PPO end-to-end smoke train")
    print("============================================")
    print(f"Timesteps : {args.timesteps}")
    print(f"Encoder   : {args.encoder_checkpoint} (frozen)")
    print(f"Model     : {SIMPLE_MODEL_PATH}")
    print("Scope     : one static Legacy-native graph; compute-only runtime")
    print("Paper-grade generalization: NO")

    model.learn(total_timesteps=int(args.timesteps), callback=callback, progress_bar=True)
    model.save(str(SIMPLE_MODEL_PATH))
    env.close()

    print("\nTraining completed")
    print(f"Saved model: {SIMPLE_MODEL_PATH}")
    print(f"Config     : {SIMPLE_CONFIG_PATH}")

    if not args.skip_evaluation:
        from .simple_evaluate import run_evaluation

        run_evaluation(
            model_path=SIMPLE_MODEL_PATH,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
