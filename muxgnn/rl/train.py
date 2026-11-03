from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .config import (
    MUX_ACTOR_HIDDEN_DIM,
    MUX_ARTIFACT_DIR,
    MUX_BATCH_SIZE,
    MUX_CLIP_RANGE,
    MUX_CRITIC_HIDDEN_DIM,
    MUX_DROPOUT,
    MUX_ENCODER_CHECKPOINT_PATH,
    MUX_ENT_COEF,
    MUX_EVALUATION_ALPHA,
    MUX_EVALUATION_RUNS,
    MUX_GAE_LAMBDA,
    MUX_GAMMA,
    MUX_GLOBAL_HIDDEN_DIM,
    MUX_HEADS,
    MUX_HIDDEN_DIM,
    MUX_LAYERS_PER_RELATION,
    MUX_LEARNING_RATE,
    MUX_MAX_GRAD_NORM,
    MUX_MIGRATION_FF_DIM,
    MUX_MIGRATION_HEADS,
    MUX_MIGRATION_HIDDEN_DIM,
    MUX_MIGRATION_LAYERS,
    MUX_MODEL_DIR,
    MUX_MODEL_PATH,
    MUX_N_EPOCHS,
    MUX_N_STEPS,
    MUX_PAIR_HIDDEN_DIM,
    MUX_SEED,
    MUX_TENSORBOARD_DIR,
    MUX_TOTAL_TIMESTEPS,
    MUX_TRAINING_CONFIG_PATH,
    MUX_VF_COEF,
)
from .env import MuxGNNEventDrivenEnv
from .policy import MaskedMuxActorCriticPolicy



def build_env(
    seed: int,
) -> Monitor:
    """Build the training environment."""

    return Monitor(
        MuxGNNEventDrivenEnv(
            split="train",
            random_scenarios=True,
            seed=seed,
        )
    )



def save_training_config(
    env: MuxGNNEventDrivenEnv,
    timesteps: int,
    seed: int,
    device: str,
    encoder_checkpoint: Path,
) -> None:
    """Save the exact model/training contract used for this run."""

    payload = {
        "pipeline": (
            "offline_pretrained_frozen_muxgnn_to_ppo"
        ),
        "seed": int(seed),
        "timesteps": int(
            timesteps
        ),
        "device": device,
        "action_space": (
            "schedule(task_x_resource_x_processor) + "
            "runtime_migration(running_task_x_target_resource_x_processor)"
        ),
        "processor_type_resolution": "ppo_selected_A7_or_A12",
        "runtime_migration": {
            "enabled": True,
            "decision_model": "ppo_trainable_transformer_head",
            "graph_encoder": "offline_pretrained_frozen_muxgnn",
            "keep_action": True,
            "checkpoint_state_size": "deterministic_synthetic_from_task_cost_and_gang",
        },
        "graph_encoder_training": {
            "mode": "offline_pretrained_then_frozen",
            "checkpoint": str(encoder_checkpoint),
            "ppo_backprop_through_encoder": False,
        },
        "graph_node_types": [
            "task",
            "vehicle",
            "edge_server",
        ],
        "relation_layers": [
            "dependency",
            "local",
            "v2v",
            "v2i",
            "contention",
        ],
        "muxgnn": {
            "hidden_dim": (
                MUX_HIDDEN_DIM
            ),
            "heads": (
                MUX_HEADS
            ),
            "layers_per_relation": (
                MUX_LAYERS_PER_RELATION
            ),
            "pair_hidden_dim": (
                MUX_PAIR_HIDDEN_DIM
            ),
            "global_hidden_dim": (
                MUX_GLOBAL_HIDDEN_DIM
            ),
            "actor_hidden_dim": (
                MUX_ACTOR_HIDDEN_DIM
            ),
            "critic_hidden_dim": (
                MUX_CRITIC_HIDDEN_DIM
            ),
            "dropout": (
                MUX_DROPOUT
            ),
            "migration_transformer_hidden_dim": MUX_MIGRATION_HIDDEN_DIM,
            "migration_transformer_heads": MUX_MIGRATION_HEADS,
            "migration_transformer_layers": MUX_MIGRATION_LAYERS,
            "migration_transformer_ff_dim": MUX_MIGRATION_FF_DIM,
        },
        "ppo": {
            "learning_rate": (
                MUX_LEARNING_RATE
            ),
            "n_steps": (
                MUX_N_STEPS
            ),
            "batch_size": (
                MUX_BATCH_SIZE
            ),
            "n_epochs": (
                MUX_N_EPOCHS
            ),
            "gamma": (
                MUX_GAMMA
            ),
            "gae_lambda": (
                MUX_GAE_LAMBDA
            ),
            "clip_range": (
                MUX_CLIP_RANGE
            ),
            "ent_coef": (
                MUX_ENT_COEF
            ),
            "vf_coef": (
                MUX_VF_COEF
            ),
            "max_grad_norm": (
                MUX_MAX_GRAD_NORM
            ),
        },
        "observation_space": str(
            env.observation_space
        ),
        "action_dimension": int(
            env.action_space.n
        ),
    }

    MUX_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MUX_TRAINING_CONFIG_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO with muxGNN on the event-driven MAGNET graph."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=(
            MUX_TOTAL_TIMESTEPS
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=MUX_SEED,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=MUX_ENCODER_CHECKPOINT_PATH,
        help=(
            "Offline-pretrained muxGNN encoder checkpoint. "
            "Create it first with: python -m muxgnn.rl.offline_pretrain"
        ),
    )

    parser.add_argument(
        "--exploration-alpha",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--eval-runs",
        type=int,
        default=(
            MUX_EVALUATION_RUNS
        ),
        help=(
            "Number of paired HEFT/PPO evaluation runs after training."
        ),
    )

    parser.add_argument(
        "--eval-alpha",
        type=float,
        default=(
            MUX_EVALUATION_ALPHA
        ),
        help=(
            "Exploration alpha used by PPO during post-training evaluation."
        ),
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help=(
            "Save the model without automatically creating comparison outputs."
        ),
    )

    args = parser.parse_args()

    if args.timesteps <= 0:
        raise ValueError(
            "--timesteps must be positive."
        )

    if args.eval_runs <= 0:
        raise ValueError(
            "--eval-runs must be positive."
        )

    if not (
        0.0
        <= args.exploration_alpha
        <= 1.0
    ):
        raise ValueError(
            "--exploration-alpha must be in [0, 1]."
        )

    if not (
        0.0
        <= args.eval_alpha
        <= 1.0
    ):
        raise ValueError(
            "--eval-alpha must be in [0, 1]."
        )

    encoder_checkpoint = Path(args.encoder_checkpoint)
    if not encoder_checkpoint.exists():
        raise FileNotFoundError(
            "Offline muxGNN encoder checkpoint is required before PPO training.\n"
            f"Expected: {encoder_checkpoint}\n"
            "Run first:\n"
            "python -m muxgnn.rl.offline_pretrain"
        )

    MUX_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MUX_TENSORBOARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_env = MuxGNNEventDrivenEnv(
        split="train",
        random_scenarios=True,
        seed=args.seed,
    )

    save_training_config(
        env=raw_env,
        timesteps=(
            args.timesteps
        ),
        seed=args.seed,
        device=args.device,
        encoder_checkpoint=encoder_checkpoint,
    )

    env = Monitor(
        raw_env
    )

    policy_kwargs = {
        "hidden_dim": (
            MUX_HIDDEN_DIM
        ),
        "heads": (
            MUX_HEADS
        ),
        "layers_per_relation": (
            MUX_LAYERS_PER_RELATION
        ),
        "pair_hidden_dim": (
            MUX_PAIR_HIDDEN_DIM
        ),
        "global_hidden_dim": (
            MUX_GLOBAL_HIDDEN_DIM
        ),
        "actor_hidden_dim": (
            MUX_ACTOR_HIDDEN_DIM
        ),
        "critic_hidden_dim": (
            MUX_CRITIC_HIDDEN_DIM
        ),
        "dropout": (
            MUX_DROPOUT
        ),
        "migration_hidden_dim": MUX_MIGRATION_HIDDEN_DIM,
        "migration_heads": MUX_MIGRATION_HEADS,
        "migration_layers": MUX_MIGRATION_LAYERS,
        "migration_ff_dim": MUX_MIGRATION_FF_DIM,
        "exploration_alpha": float(
            args.exploration_alpha
        ),
        "freeze_graph_encoder": True,
    }

    model = PPO(
        policy=(
            MaskedMuxActorCriticPolicy
        ),
        env=env,
        learning_rate=(
            MUX_LEARNING_RATE
        ),
        n_steps=(
            MUX_N_STEPS
        ),
        batch_size=(
            MUX_BATCH_SIZE
        ),
        n_epochs=(
            MUX_N_EPOCHS
        ),
        gamma=MUX_GAMMA,
        gae_lambda=(
            MUX_GAE_LAMBDA
        ),
        clip_range=(
            MUX_CLIP_RANGE
        ),
        ent_coef=(
            MUX_ENT_COEF
        ),
        vf_coef=(
            MUX_VF_COEF
        ),
        max_grad_norm=(
            MUX_MAX_GRAD_NORM
        ),
        policy_kwargs=(
            policy_kwargs
        ),
        tensorboard_log=str(
            MUX_TENSORBOARD_DIR
        ),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    # Stage 2 starts here: import the graph representation learned offline.
    # The policy constructor has already removed encoder parameters from the
    # PPO optimizer; load_pretrained_encoder() copies the learned weights and
    # keeps the complete graph-processing path frozen/eval/no-grad.
    model.policy.load_pretrained_encoder(
        encoder_checkpoint
    )

    encoder_parameters = list(
        model.policy.scheduler_net.encoder.parameters()
    )
    if any(parameter.requires_grad for parameter in encoder_parameters):
        raise RuntimeError(
            "muxGNN encoder must remain frozen during PPO training."
        )

    optimizer_parameter_ids = {
        id(parameter)
        for group in model.policy.optimizer.param_groups
        for parameter in group["params"]
    }
    if any(id(parameter) in optimizer_parameter_ids for parameter in encoder_parameters):
        raise RuntimeError(
            "Frozen muxGNN encoder unexpectedly appears in the PPO optimizer."
        )

    # Migration-specific modules must stay trainable inside PPO.
    migration_modules = (
        model.policy.scheduler_net.migration_input_projection,
        model.policy.scheduler_net.migration_processor_embedding,
        model.policy.scheduler_net.migration_transformer,
        model.policy.scheduler_net.migration_actor,
    )
    migration_parameters = [
        parameter
        for module in migration_modules
        for parameter in module.parameters()
    ]
    if not migration_parameters or not all(
        parameter.requires_grad for parameter in migration_parameters
    ):
        raise RuntimeError(
            "Runtime migration Transformer must remain trainable during PPO."
        )
    if not all(
        id(parameter) in optimizer_parameter_ids
        for parameter in migration_parameters
    ):
        raise RuntimeError(
            "Runtime migration Transformer is missing from the PPO optimizer."
        )

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=max(
                int(
                    args.checkpoint_freq
                ),
                1,
            ),
            save_path=str(
                MUX_MODEL_DIR
                / "checkpoints"
            ),
            name_prefix=(
                "ppo_muxgnn"
            ),
        )
    )

    print(
        "================================="
    )
    print(
        "Training PPO with frozen muxGNN + runtime migration Transformer"
    )
    print(
        "================================="
    )
    print(
        f"Timesteps : {args.timesteps}"
    )
    print(
        f"Model path: {MUX_MODEL_PATH}"
    )
    print(
        f"Encoder   : {encoder_checkpoint} (frozen)"
    )

    model.learn(
        total_timesteps=int(
            args.timesteps
        ),
        callback=(
            checkpoint_callback
        ),
        progress_bar=True,
    )

    model.save(
        str(
            MUX_MODEL_PATH
        )
    )

    env.close()

    print(
        "\n================================="
    )
    print(
        "Training completed"
    )
    print(
        "================================="
    )
    print(
        f"Saved model: {MUX_MODEL_PATH}"
    )
    print(
        f"Config     : {MUX_TRAINING_CONFIG_PATH}"
    )

    # --------------------------------------------------------
    # Automatic legacy-compatible evaluation outputs
    # --------------------------------------------------------

    if not args.skip_evaluation:
        print(
            "\nStarting post-training comparison evaluation..."
        )

        # Lazy import keeps training startup simple and avoids loading
        # matplotlib/pandas unless evaluation is actually requested.
        from .evaluate import run_benchmark

        run_benchmark(
            model_path=(
                MUX_MODEL_PATH
            ),
            evaluation_runs=(
                args.eval_runs
            ),
            evaluation_alpha=(
                args.eval_alpha
            ),
            seed=args.seed,
            device=args.device,
            deterministic_ppo=False,
        )


if __name__ == "__main__":
    main()
