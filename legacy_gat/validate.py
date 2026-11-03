from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from .config import DATASET_PATH, SEED
from .utils import run_policy_episode
from .train import make_env


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Run python run_legacy_gat.py generate first."
        )

    env = make_env(seed=SEED)
    try:
        print("=== Structural checks ===")
        print(f"tasks               = {env.num_tasks}")
        print(f"edges               = {len(env.edges)}")
        print(f"max m_i             = {int(np.max(env.m_i))}")
        print(f"physical cores      = {env.num_a7_cores} A7 + {env.num_a12_cores} A12")
        print(f"action dimension    = {env.action_space.n}")
        print(f"observation space   = {env.observation_space}")
        check_env(env, warn=True, skip_render_check=True)
        print("check_env: PASS")

        print("\n=== HEFT_GANG sanity schedule ===")
        _, _, _, metrics, total_reward, steps = run_policy_episode(
            env,
            "HEFT_GANG",
            rng=np.random.default_rng(SEED),
        )
        env.validate_schedule()
        print("schedule invariants: PASS")
        print(f"scheduled tasks      = {len(steps)}")
        print(f"makespan             = {metrics['makespan']:.6f}")
        print(f"normalized energy    = {metrics['total_energy']:.6f}")
        print(f"average QoS          = {metrics['average_qos']:.6f}")
        print(f"on-time ratio        = {metrics['on_time_task_ratio']:.6f}")
        print(f"zero-QoS ratio       = {metrics['zero_qos_task_ratio']:.6f}")
        print(f"mean gang width      = {metrics['mean_scheduled_width']:.6f}")
        print(f"gang usage ratio     = {metrics['gang_usage_ratio']:.6f}")
        print(f"core utilization     = {metrics['physical_core_utilization']:.6f}")
        print(f"episode reward       = {total_reward:.6f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
