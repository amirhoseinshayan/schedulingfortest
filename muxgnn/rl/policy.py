from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import torch
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from .config import (
    MUX_MIGRATION_FF_DIM,
    MUX_MIGRATION_HEADS,
    MUX_MIGRATION_HIDDEN_DIM,
    MUX_MIGRATION_LAYERS,
)
from .model import MuxActorCriticNetwork


class _UnusedMuxExtractor(BaseFeaturesExtractor):
    """SB3 constructor shim; muxGNN processing happens in scheduler_net."""

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        global_features = observations["global_features"].float()
        if global_features.ndim == 1:
            global_features = global_features.unsqueeze(0)
        return global_features[:, :1]


class MaskedMuxActorCriticPolicy(ActorCriticPolicy):
    """PPO policy with masked scheduling and Transformer migration actions."""

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
        migration_hidden_dim: int = MUX_MIGRATION_HIDDEN_DIM,
        migration_heads: int = MUX_MIGRATION_HEADS,
        migration_layers: int = MUX_MIGRATION_LAYERS,
        migration_ff_dim: int = MUX_MIGRATION_FF_DIM,
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
        self.migration_hidden_dim = int(migration_hidden_dim)
        self.migration_heads = int(migration_heads)
        self.migration_layers = int(migration_layers)
        self.migration_ff_dim = int(migration_ff_dim)
        self.freeze_graph_encoder = bool(freeze_graph_encoder)
        self.exploration_alpha = 0.0
        self.set_exploration_alpha(exploration_alpha)

        kwargs["features_extractor_class"] = _UnusedMuxExtractor
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
            raise TypeError("MaskedMuxActorCriticPolicy requires Dict observations.")
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError("MaskedMuxActorCriticPolicy requires Discrete actions.")

        self.scheduler_net = MuxActorCriticNetwork(
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
            migration_hidden_dim=self.migration_hidden_dim,
            migration_heads=self.migration_heads,
            migration_layers=self.migration_layers,
            migration_ff_dim=self.migration_ff_dim,
        )

        if self.freeze_graph_encoder:
            self.scheduler_net.freeze_encoder()

        self.optimizer = self.optimizer_class(
            [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            ],
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def load_pretrained_encoder(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        """Load the offline muxGNN encoder checkpoint and keep it frozen."""

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Offline muxGNN encoder checkpoint not found: {path}"
            )

        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                path,
                map_location="cpu",
            )

        if not isinstance(checkpoint, dict) or "encoder_state_dict" not in checkpoint:
            raise ValueError(
                "Invalid offline encoder checkpoint: expected 'encoder_state_dict'."
            )

        self.scheduler_net.encoder.load_state_dict(
            checkpoint["encoder_state_dict"],
            strict=True,
        )
        self.scheduler_net.freeze_encoder()

    @staticmethod
    def _safe_action_mask(
        logits: torch.Tensor,
        observations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action_mask = observations.get("action_mask")

        if action_mask is None:
            return torch.ones_like(logits, dtype=torch.bool)

        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)

        action_mask = action_mask.bool()
        has_valid = action_mask.any(dim=1, keepdim=True)

        # Terminal observations can contain no valid actions. SB3 still needs
        # a numerically safe distribution object for value/log-prob plumbing.
        return torch.where(
            has_valid,
            action_mask,
            torch.ones_like(action_mask),
        )

    def _distribution_and_value(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[Distribution, torch.Tensor]:
        logits, values = self.scheduler_net(observations)
        safe_mask = self._safe_action_mask(logits, observations)

        masked_logits = logits.masked_fill(~safe_mask, -1e9)
        distribution = self.action_dist.proba_distribution(
            action_logits=masked_logits
        )

        alpha = float(self.exploration_alpha)
        if alpha > 0.0:
            base_prob = distribution.distribution.probs
            valid_float = safe_mask.to(base_prob.dtype)
            uniform_valid = valid_float / valid_float.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)

            mixed = (
                (1.0 - alpha) * base_prob
                + alpha * uniform_valid
            )
            mixed = mixed * valid_float
            mixed = mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-8)

            distribution = self.action_dist.proba_distribution(
                action_logits=torch.log(mixed.clamp_min(1e-8))
            )

        return distribution, values

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_value(observations)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_value(observations)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Distribution:
        distribution, _ = self._distribution_and_value(observations)
        return distribution

    def predict_values(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        _, values = self.scheduler_net(observations)
        return values

    @torch.no_grad()
    def relation_attention(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
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
                "migration_hidden_dim": self.migration_hidden_dim,
                "migration_heads": self.migration_heads,
                "migration_layers": self.migration_layers,
                "migration_ff_dim": self.migration_ff_dim,
                "exploration_alpha": self.exploration_alpha,
                "freeze_graph_encoder": self.freeze_graph_encoder,
            }
        )
        return data
