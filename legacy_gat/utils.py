from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def encode_action(task: int, slot: int, num_action_slots: int) -> int:
    return int(task * num_action_slots + slot)


def decode_action(action: int, num_action_slots: int) -> Tuple[int, int]:
    return int(action // num_action_slots), int(action % num_action_slots)


def valid_actions_from_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    mask = np.asarray(obs["action_mask"], dtype=bool)
    return np.flatnonzero(mask)


def _ready_tasks_from_obs(env, obs: Dict[str, np.ndarray]) -> List[int]:
    valid_actions = valid_actions_from_obs(obs)
    tasks = {
        decode_action(int(action), env.num_action_slots)[0]
        for action in valid_actions
    }
    return sorted(task for task in tasks if 0 <= task < env.num_tasks)


def choose_action_random(
    env,
    obs: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> int:
    valid_actions = valid_actions_from_obs(obs)
    if valid_actions.size == 0:
        raise RuntimeError("No valid action exists in a non-terminal state.")
    return int(rng.choice(valid_actions))


def choose_action_edf_gang(env, obs: Dict[str, np.ndarray]) -> int:
    ready = _ready_tasks_from_obs(env, obs)
    if not ready:
        raise RuntimeError("EDF_GANG found no ready task.")
    task = min(ready, key=lambda t: (float(env.deadlines[t]), int(t)))
    slot = env.best_allocation_for_task(task)
    return encode_action(task, slot, env.num_action_slots)


def choose_action_heft_gang(env, obs: Dict[str, np.ndarray]) -> int:
    """
    Gang-aware HEFT-like baseline:
      1) choose highest upward-rank ready task,
      2) choose Gang composition with earliest finish,
         tie-breaking on lower Gang energy.
    """
    ready = _ready_tasks_from_obs(env, obs)
    if not ready:
        raise RuntimeError("HEFT_GANG found no ready task.")

    task = max(
        ready,
        key=lambda t: (
            float(env.rank_u_raw[t]),
            -float(env.deadlines[t]),
            -int(t),
        ),
    )
    slot = env.best_allocation_for_task(task)
    return encode_action(task, slot, env.num_action_slots)


def choose_action_from_sb3_model(
    model,
    obs: Dict[str, np.ndarray],
    deterministic: bool = True,
    exploration_alpha: Optional[float] = None,
) -> int:
    if exploration_alpha is not None:
        model.policy.set_exploration_alpha(float(exploration_alpha))

    model.policy.set_training_mode(False)
    device = model.policy.device
    boolean_keys = {"edge_mask", "node_mask", "action_mask"}
    observation_tensors: Dict[str, torch.Tensor] = {}

    for key, value in obs.items():
        array = np.asarray(value)
        if key == "edge_index":
            tensor = torch.as_tensor(array, dtype=torch.long, device=device)
        elif key in boolean_keys:
            tensor = torch.as_tensor(array, dtype=torch.bool, device=device)
        else:
            tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
        observation_tensors[key] = tensor.unsqueeze(0)

    with torch.no_grad():
        actions, _, _ = model.policy.forward(
            observation_tensors,
            deterministic=deterministic,
        )
    return int(actions.reshape(-1)[0].item())


def run_policy_episode(
    env,
    policy_name: str,
    rng: Optional[np.random.Generator] = None,
    model=None,
    ppo_deterministic: bool = True,
    ppo_exploration_alpha: Optional[float] = None,
):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    step_rows: List[dict] = []

    normalized_name = policy_name.upper()
    while not (terminated or truncated):
        if normalized_name == "RANDOM_GANG":
            if rng is None:
                rng = np.random.default_rng()
            action = choose_action_random(env, obs, rng)
        elif normalized_name == "EDF_GANG":
            action = choose_action_edf_gang(env, obs)
        elif normalized_name in {"HEFT_GANG", "HEFT-GANG"}:
            action = choose_action_heft_gang(env, obs)
        elif normalized_name in {"PPO_GANG", "PPO+GAT+GANG", "PPO_GAT_GANG"}:
            if model is None:
                raise ValueError("model is required for PPO_GANG evaluation.")
            action = choose_action_from_sb3_model(
                model,
                obs,
                deterministic=ppo_deterministic,
                exploration_alpha=ppo_exploration_alpha,
            )
        else:
            raise ValueError(f"Unknown policy_name: {policy_name}")

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)

        if not step_info.get("invalid_action", False):
            step_rows.append(
                {
                    "action": int(action),
                    "reward": float(reward),
                    **step_info,
                }
            )
        obs = next_obs

    env.validate_schedule()
    assigned, start, finish = env.get_schedule()
    metrics = env.get_metrics()
    return assigned, start, finish, metrics, total_reward, step_rows
