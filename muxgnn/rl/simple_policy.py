from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from .config import MUX_NUM_PROCESSOR_TYPES, MUX_SCHEDULE_ACTIONS
from .model import MuxGNNEncoder


class _UnusedSimpleMuxExtractor(BaseFeaturesExtractor):
    """SB3 constructor shim; the custom muxGNN network consumes the Dict directly."""

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        global_features = observations["global_features"].float()
        if global_features.ndim == 1:
            global_features = global_features.unsqueeze(0)
        return global_features[:, :1]


class SimpleMuxActorCriticNetwork(nn.Module):
    """Frozen muxGNN encoder + scheduling-only Actor/Critic for the v2 smoke stage."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        hidden_dim: int = 96,
        heads: int = 4,
        layers_per_relation: int = 2,
        pair_hidden_dim: int = 48,
        global_hidden_dim: int = 48,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("SimpleMuxActorCriticNetwork requires Discrete actions.")
        if int(action_space.n) != MUX_SCHEDULE_ACTIONS:
            raise ValueError(
                f"Simple v2 PPO expects {MUX_SCHEDULE_ACTIONS} scheduling actions, "
                f"got {action_space.n}."
            )

        self.max_tasks = int(observation_space["task_features"].shape[0])
        self.max_vehicles = int(observation_space["vehicle_features"].shape[0])
        self.max_edge_servers = int(observation_space["edge_server_features"].shape[0])
        self.max_resources = self.max_vehicles + self.max_edge_servers
        self.num_processor_types = MUX_NUM_PROCESSOR_TYPES

        pair_feature_dim = int(observation_space["pair_features"].shape[-1])

        self.encoder = MuxGNNEncoder(
            observation_space=observation_space,
            hidden_dim=hidden_dim,
            heads=heads,
            layers_per_relation=layers_per_relation,
            global_hidden_dim=global_hidden_dim,
            dropout=dropout,
        )
        self.encoder_frozen = False

        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_feature_dim, pair_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(pair_hidden_dim),
        )

        schedule_input_dim = 2 * hidden_dim + pair_hidden_dim + global_hidden_dim
        self.actor = nn.Sequential(
            nn.Linear(schedule_input_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, self.num_processor_types),
        )

        pooled_dim = 6 * hidden_dim + global_hidden_dim
        self.critic = nn.Sequential(
            nn.Linear(pooled_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1),
        )
        self._initialize()

    def freeze_encoder(self) -> None:
        self.encoder_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def _encode(self, observations: Dict[str, torch.Tensor]):
        if self.encoder_frozen:
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder(observations)
        return self.encoder(observations)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    @staticmethod
    def _ensure_batch(tensor: torch.Tensor, expected_without_batch: int) -> torch.Tensor:
        return tensor.unsqueeze(0) if tensor.ndim == expected_without_batch else tensor

    @staticmethod
    def _masked_mean_max(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        mask_f = mask.unsqueeze(-1).to(embeddings.dtype)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (embeddings * mask_f).sum(dim=1) / count
        very_negative = torch.finfo(embeddings.dtype).min
        max_values = embeddings.masked_fill(~mask.unsqueeze(-1), very_negative).max(dim=1).values
        max_values = torch.where(
            mask.any(dim=1, keepdim=True), max_values, torch.zeros_like(max_values)
        )
        return torch.cat([mean, max_values], dim=-1)

    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            task_embeddings,
            vehicle_embeddings,
            edge_server_embeddings,
            global_embedding,
            _,
            _,
        ) = self._encode(observations)

        pair_features = self._ensure_batch(observations["pair_features"], 3).float()
        task_mask = self._ensure_batch(observations["task_mask"], 1).bool()
        vehicle_mask = self._ensure_batch(observations["vehicle_mask"], 1).bool()
        edge_server_mask = self._ensure_batch(observations["edge_server_mask"], 1).bool()

        resource_embeddings = torch.cat([vehicle_embeddings, edge_server_embeddings], dim=1)
        pair_embeddings = self.pair_encoder(pair_features)
        batch_size = task_embeddings.shape[0]

        task_expanded = task_embeddings.unsqueeze(2).expand(
            batch_size, self.max_tasks, self.max_resources, task_embeddings.shape[-1]
        )
        resource_expanded = resource_embeddings.unsqueeze(1).expand(
            batch_size, self.max_tasks, self.max_resources, resource_embeddings.shape[-1]
        )
        global_expanded = global_embedding[:, None, None, :].expand(
            batch_size, self.max_tasks, self.max_resources, global_embedding.shape[-1]
        )

        schedule_input = torch.cat(
            [task_expanded, resource_expanded, pair_embeddings, global_expanded], dim=-1
        )
        logits = self.actor(schedule_input).reshape(batch_size, -1)

        task_pool = self._masked_mean_max(task_embeddings, task_mask)
        vehicle_pool = self._masked_mean_max(vehicle_embeddings, vehicle_mask)
        edge_pool = self._masked_mean_max(edge_server_embeddings, edge_server_mask)
        value_input = torch.cat([task_pool, vehicle_pool, edge_pool, global_embedding], dim=-1)
        values = self.critic(value_input)
        return logits, values

    @torch.no_grad()
    def relation_attention(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, _, _, _, weights, _ = self._encode(observations)
        return weights


class SimpleMaskedMuxPolicy(ActorCriticPolicy):
    """Scheduling-only masked PPO policy used exclusively by the v2 smoke stage."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        *args: Any,
        hidden_dim: int = 96,
        heads: int = 4,
        layers_per_relation: int = 2,
        pair_hidden_dim: int = 48,
        global_hidden_dim: int = 48,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        dropout: float = 0.10,
        exploration_alpha: float = 0.0,
        freeze_graph_encoder: bool = True,
        **kwargs: Any,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.layers_per_relation = int(layers_per_relation)
        self.pair_hidden_dim = int(pair_hidden_dim)
        self.global_hidden_dim = int(global_hidden_dim)
        self.actor_hidden_dim = int(actor_hidden_dim)
        self.critic_hidden_dim = int(critic_hidden_dim)
        self.dropout = float(dropout)
        self.freeze_graph_encoder = bool(freeze_graph_encoder)
        self.exploration_alpha = 0.0
        self.set_exploration_alpha(exploration_alpha)

        kwargs["features_extractor_class"] = _UnusedSimpleMuxExtractor
        kwargs["features_extractor_kwargs"] = {}
        kwargs["net_arch"] = []
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args,
            **kwargs,
        )

    def set_exploration_alpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("exploration_alpha must be in [0, 1].")
        self.exploration_alpha = alpha

    def _build(self, lr_schedule: Schedule) -> None:
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise TypeError("SimpleMaskedMuxPolicy requires Dict observations.")
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError("SimpleMaskedMuxPolicy requires Discrete actions.")

        self.scheduler_net = SimpleMuxActorCriticNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            layers_per_relation=self.layers_per_relation,
            pair_hidden_dim=self.pair_hidden_dim,
            global_hidden_dim=self.global_hidden_dim,
            actor_hidden_dim=self.actor_hidden_dim,
            critic_hidden_dim=self.critic_hidden_dim,
            dropout=self.dropout,
        )
        if self.freeze_graph_encoder:
            self.scheduler_net.freeze_encoder()

        self.optimizer = self.optimizer_class(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def load_pretrained_encoder(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Offline muxGNN encoder checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "encoder_state_dict" not in checkpoint:
            raise ValueError("Invalid offline encoder checkpoint: missing encoder_state_dict.")
        self.scheduler_net.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        self.scheduler_net.freeze_encoder()

    @staticmethod
    def _safe_action_mask(logits: torch.Tensor, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        action_mask = observations.get("action_mask")
        if action_mask is None:
            return torch.ones_like(logits, dtype=torch.bool)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        action_mask = action_mask.bool()
        has_valid = action_mask.any(dim=1, keepdim=True)
        return torch.where(has_valid, action_mask, torch.ones_like(action_mask))

    def _distribution_and_value(
        self, observations: Dict[str, torch.Tensor]
    ) -> Tuple[Distribution, torch.Tensor]:
        logits, values = self.scheduler_net(observations)
        safe_mask = self._safe_action_mask(logits, observations)
        masked_logits = logits.masked_fill(~safe_mask, -1e9)
        distribution = self.action_dist.proba_distribution(action_logits=masked_logits)

        alpha = float(self.exploration_alpha)
        if alpha > 0.0:
            base_prob = distribution.distribution.probs
            valid_float = safe_mask.to(base_prob.dtype)
            uniform_valid = valid_float / valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            mixed = (1.0 - alpha) * base_prob + alpha * uniform_valid
            mixed = mixed * valid_float
            mixed = mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-8)
            distribution = self.action_dist.proba_distribution(
                action_logits=torch.log(mixed.clamp_min(1e-8))
            )
        return distribution, values

    def forward(
        self, observations: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_value(observations)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self, observations: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_value(observations)
        return values, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, observations: Dict[str, torch.Tensor]) -> Distribution:
        distribution, _ = self._distribution_and_value(observations)
        return distribution

    def predict_values(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, values = self.scheduler_net(observations)
        return values

    @torch.no_grad()
    def relation_attention(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.scheduler_net.relation_attention(observations)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "hidden_dim": self.hidden_dim,
                "heads": self.heads,
                "layers_per_relation": self.layers_per_relation,
                "pair_hidden_dim": self.pair_hidden_dim,
                "global_hidden_dim": self.global_hidden_dim,
                "actor_hidden_dim": self.actor_hidden_dim,
                "critic_hidden_dim": self.critic_hidden_dim,
                "dropout": self.dropout,
                "exploration_alpha": self.exploration_alpha,
                "freeze_graph_encoder": self.freeze_graph_encoder,
            }
        )
        return data
