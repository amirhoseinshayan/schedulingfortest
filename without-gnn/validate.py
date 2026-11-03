from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

from _bootstrap import PROJECT_ROOT  # noqa: F401
from config import DATASET_PATH, LEARNING_RATE, SEED
from legacy_gat.utils import run_policy_episode
from policy import MaskedNoGNNActorCriticPolicy
from train import make_env


def _obs_to_tensors(obs: dict) -> dict[str, torch.Tensor]:
    boolean_keys = {"edge_mask", "node_mask", "action_mask"}
    result: dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        array = np.asarray(value)
        if key == "edge_index":
            tensor = torch.as_tensor(array, dtype=torch.long)
        elif key in boolean_keys:
            tensor = torch.as_tensor(array, dtype=torch.bool)
        else:
            tensor = torch.as_tensor(array, dtype=torch.float32)
        result[key] = tensor.unsqueeze(0)
    return result


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    env = make_env(seed=SEED)
    try:
        print("=== Shared environment checks ===")
        print(f"tasks               = {env.num_tasks}")
        print(f"edges               = {len(env.edges)}")
        print(f"max m_i             = {int(np.max(env.m_i))}")
        print(f"physical cores      = {env.num_a7_cores} A7 + {env.num_a12_cores} A12")
        print(f"action dimension    = {env.action_space.n}")
        print(f"observation space   = {env.observation_space}")
        check_env(env, warn=True, skip_render_check=True)
        print("check_env: PASS")

        print("\n=== HEFT_GANG sanity schedule (same environment) ===")
        _, _, _, metrics, total_reward, steps = run_policy_episode(
            env, "HEFT_GANG", rng=np.random.default_rng(SEED)
        )
        env.validate_schedule()
        print("schedule invariants: PASS")
        print(f"scheduled tasks      = {len(steps)}")
        print(f"makespan             = {metrics['makespan']:.6f}")
        print(f"normalized energy    = {metrics['total_energy']:.6f}")
        print(f"average QoS          = {metrics['average_qos']:.6f}")
        print(f"episode reward       = {total_reward:.6f}")

        print("\n=== No-GNN policy checks ===")
        obs, _ = env.reset(seed=SEED)
        policy = MaskedNoGNNActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: LEARNING_RATE,
            node_hidden_dim=64,
            node_mlp_layers=2,
            core_hidden_dim=32,
            global_hidden_dim=32,
            allocation_hidden_dim=32,
            context_hidden_dim=64,
            graph_pool_dim=128,
            actor_hidden_dim=128,
            critic_hidden_dim=128,
            dropout=0.10,
            exploration_alpha=0.0,
        )
        policy.set_training_mode(False)
        tensors_a = _obs_to_tensors(obs)
        tensors_b = {key: value.clone() for key, value in tensors_a.items()}
        # Deliberately destroy graph connectivity. A true no-GNN policy must be
        # exactly invariant because edge_index/edge_mask are not consumed.
        tensors_b["edge_index"].zero_()
        tensors_b["edge_mask"].zero_()
        with torch.no_grad():
            logits_a, values_a = policy.scheduler_net(tensors_a)
            logits_b, values_b = policy.scheduler_net(tensors_b)
        if not torch.equal(logits_a, logits_b) or not torch.equal(values_a, values_b):
            raise AssertionError("No-GNN policy unexpectedly depends on graph edges.")
        print("edge invariance      : PASS (edge_index/edge_mask are ignored)")
        print(f"policy logits shape : {tuple(logits_a.shape)}")
        print(f"value shape         : {tuple(values_a.shape)}")
        print("WITHOUT-GNN validation: PASS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
