from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn

from .config import (
    MUX_MIGRATION_ACTIONS,
    MUX_MIGRATION_FF_DIM,
    MUX_MIGRATION_HEADS,
    MUX_MIGRATION_HIDDEN_DIM,
    MUX_MIGRATION_LAYERS,
    MUX_NUM_PROCESSOR_TYPES,
    MUX_NUM_RELATIONS,
    MUX_SCHEDULE_ACTIONS,
)


class DenseRelationAttentionLayer(nn.Module):
    """Multi-head attention for one derived dense relation view."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        edge_feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0 or heads <= 0 or hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim must be positive and divisible by heads."
            )

        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.edge_bias = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, heads),
        )

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        relation_presence: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = node_embeddings.shape

        active_nodes = relation_presence.bool() & node_mask.bool()

        eye = torch.eye(
            num_nodes,
            dtype=torch.bool,
            device=node_embeddings.device,
        )

        self_loops = eye.unsqueeze(0) & active_nodes.unsqueeze(1)
        relation_adj = adjacency.bool() | self_loops

        q = (
            self.q_proj(node_embeddings)
            .view(batch_size, num_nodes, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(node_embeddings)
            .view(batch_size, num_nodes, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(node_embeddings)
            .view(batch_size, num_nodes, self.heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        edge_bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)
        scores = scores + edge_bias

        attention_mask = relation_adj.unsqueeze(1)
        masked_scores = scores.masked_fill(~attention_mask, -1e9)

        weights = torch.softmax(masked_scores, dim=-1)
        weights = weights * attention_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = self.dropout(weights)

        messages = torch.einsum("bhij,bhjd->bhid", weights, v)
        messages = (
            messages.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_nodes, hidden_dim)
        )

        messages = self.dropout(self.out_proj(messages))
        output = self.norm(node_embeddings + messages)

        return output * active_nodes.unsqueeze(-1).to(output.dtype)


class RelationEncoder(nn.Module):
    """Stack relation-specific attention layers."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        edge_feature_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if layers <= 0:
            raise ValueError("layers must be positive.")

        self.layers = nn.ModuleList(
            [
                DenseRelationAttentionLayer(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    edge_feature_dim=edge_feature_dim,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        base_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        relation_presence: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = base_embeddings

        for layer in self.layers:
            hidden = layer(
                node_embeddings=hidden,
                adjacency=adjacency,
                edge_features=edge_features,
                relation_presence=relation_presence,
                node_mask=node_mask,
            )

        return hidden


class MultiplexCouplingAttention(nn.Module):
    """Learn per-node attention over the five MAGNET relation views."""

    def __init__(
        self,
        hidden_dim: int,
        global_hidden_dim: int,
        num_relations: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_relations = int(num_relations)

        self.relation_embedding = nn.Embedding(
            num_relations,
            hidden_dim,
        )

        self.base_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.global_projection = nn.Linear(
            global_hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        base_embeddings: torch.Tensor,
        relation_views: torch.Tensor,
        relation_presence: torch.Tensor,
        global_embedding: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, num_relations, hidden_dim = relation_views.shape

        relation_ids = torch.arange(
            num_relations,
            device=relation_views.device,
            dtype=torch.long,
        )

        relation_bias = self.relation_embedding(relation_ids).view(
            1,
            1,
            num_relations,
            hidden_dim,
        )

        base_query = self.base_projection(base_embeddings).unsqueeze(2)
        global_query = self.global_projection(global_embedding)[:, None, None, :]

        score_input = (
            relation_views
            + relation_bias
            + base_query
            + global_query
        )

        logits = self.score(score_input).squeeze(-1)

        coupling_mask = relation_presence.transpose(1, 2).bool()
        coupling_mask = coupling_mask & node_mask.unsqueeze(-1).bool()

        masked_logits = logits.masked_fill(~coupling_mask, -1e9)
        weights = torch.softmax(masked_logits, dim=-1)
        weights = weights * coupling_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        fused = torch.sum(
            weights.unsqueeze(-1) * relation_views,
            dim=2,
        )

        fused = self.dropout(self.output_projection(fused))
        fused = self.norm(base_embeddings + fused)
        fused = fused * node_mask.unsqueeze(-1).to(fused.dtype)

        return fused, weights


class MuxGNNEncoder(nn.Module):
    """
    Five-relation muxGNN encoder for the current three-node-type graph.

    Node types are encoded independently:
      Task, Vehicle, Edge Server.
    They are then placed into one padded supra-node tensor for relation-specific
    message passing and multiplex coupling attention.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        hidden_dim: int = 96,
        heads: int = 4,
        layers_per_relation: int = 2,
        global_hidden_dim: int = 48,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("MuxGNNEncoder requires Dict observations.")

        task_shape = observation_space["task_features"].shape
        vehicle_shape = observation_space["vehicle_features"].shape
        edge_server_shape = observation_space["edge_server_features"].shape
        relation_edge_shape = observation_space["relation_edge_features"].shape
        global_shape = observation_space["global_features"].shape

        self.max_tasks = int(task_shape[0])
        self.max_vehicles = int(vehicle_shape[0])
        self.max_edge_servers = int(edge_server_shape[0])
        self.max_resources = self.max_vehicles + self.max_edge_servers
        self.max_nodes = self.max_tasks + self.max_resources

        self.task_feature_dim = int(task_shape[1])
        self.vehicle_feature_dim = int(vehicle_shape[1])
        self.edge_server_feature_dim = int(edge_server_shape[1])
        self.edge_feature_dim = int(relation_edge_shape[-1])
        self.global_feature_dim = int(global_shape[0])
        self.hidden_dim = int(hidden_dim)
        self.global_hidden_dim = int(global_hidden_dim)

        self.task_encoder = nn.Sequential(
            nn.Linear(self.task_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.vehicle_encoder = nn.Sequential(
            nn.Linear(self.vehicle_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_server_encoder = nn.Sequential(
            nn.Linear(self.edge_server_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_dim, global_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(global_hidden_dim),
        )

        self.relation_encoders = nn.ModuleList(
            [
                RelationEncoder(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    edge_feature_dim=self.edge_feature_dim,
                    layers=layers_per_relation,
                    dropout=dropout,
                )
                for _ in range(MUX_NUM_RELATIONS)
            ]
        )

        self.coupling_attention = MultiplexCouplingAttention(
            hidden_dim=hidden_dim,
            global_hidden_dim=global_hidden_dim,
            num_relations=MUX_NUM_RELATIONS,
            dropout=dropout,
        )

    @staticmethod
    def _ensure_batch(
        tensor: torch.Tensor,
        expected_without_batch: int,
    ) -> torch.Tensor:
        if tensor.ndim == expected_without_batch:
            return tensor.unsqueeze(0)
        return tensor

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        task_features = self._ensure_batch(
            observations["task_features"],
            2,
        ).float()

        vehicle_features = self._ensure_batch(
            observations["vehicle_features"],
            2,
        ).float()

        edge_server_features = self._ensure_batch(
            observations["edge_server_features"],
            2,
        ).float()

        relation_adjacency = self._ensure_batch(
            observations["relation_adjacency"],
            3,
        ).bool()

        relation_edge_features = self._ensure_batch(
            observations["relation_edge_features"],
            4,
        ).float()

        relation_presence = self._ensure_batch(
            observations["relation_presence"],
            2,
        ).bool()

        task_mask = self._ensure_batch(
            observations["task_mask"],
            1,
        ).bool()

        vehicle_mask = self._ensure_batch(
            observations["vehicle_mask"],
            1,
        ).bool()

        edge_server_mask = self._ensure_batch(
            observations["edge_server_mask"],
            1,
        ).bool()

        node_mask = self._ensure_batch(
            observations["node_mask"],
            1,
        ).bool()

        global_features = self._ensure_batch(
            observations["global_features"],
            1,
        ).float()

        task_embeddings = self.task_encoder(task_features)
        vehicle_embeddings = self.vehicle_encoder(vehicle_features)
        edge_server_embeddings = self.edge_server_encoder(edge_server_features)

        base_embeddings = torch.cat(
            [
                task_embeddings,
                vehicle_embeddings,
                edge_server_embeddings,
            ],
            dim=1,
        )

        base_embeddings = base_embeddings * node_mask.unsqueeze(-1).to(
            base_embeddings.dtype
        )

        global_embedding = self.global_encoder(global_features)

        relation_views = []

        for relation_id, encoder in enumerate(self.relation_encoders):
            view = encoder(
                base_embeddings=base_embeddings,
                adjacency=relation_adjacency[:, relation_id],
                edge_features=relation_edge_features[:, relation_id],
                relation_presence=relation_presence[:, relation_id],
                node_mask=node_mask,
            )
            relation_views.append(view)

        stacked_views = torch.stack(relation_views, dim=2)

        fused_nodes, coupling_weights = self.coupling_attention(
            base_embeddings=base_embeddings,
            relation_views=stacked_views,
            relation_presence=relation_presence,
            global_embedding=global_embedding,
            node_mask=node_mask,
        )

        task_end = self.max_tasks
        vehicle_end = task_end + self.max_vehicles

        fused_tasks = fused_nodes[:, :task_end]
        fused_vehicles = fused_nodes[:, task_end:vehicle_end]
        fused_edge_servers = fused_nodes[:, vehicle_end:]

        return (
            fused_tasks,
            fused_vehicles,
            fused_edge_servers,
            global_embedding,
            coupling_weights,
            node_mask,
        )


class MuxActorCriticNetwork(nn.Module):
    """
    muxGNN encoder plus two PPO action heads.

    - Initial scheduling uses the existing task-resource-processor MLP.
    - Runtime migration uses a Transformer over the 14 destination alternatives
      (7 resources x A7/A12) of each task. The Transformer reuses frozen muxGNN
      task/resource embeddings and adds runtime migration features.
    """

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
        migration_hidden_dim: int = MUX_MIGRATION_HIDDEN_DIM,
        migration_heads: int = MUX_MIGRATION_HEADS,
        migration_layers: int = MUX_MIGRATION_LAYERS,
        migration_ff_dim: int = MUX_MIGRATION_FF_DIM,
    ) -> None:
        super().__init__()

        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("MuxActorCriticNetwork requires a Discrete action space.")

        self.max_tasks = int(observation_space["task_features"].shape[0])
        self.max_vehicles = int(observation_space["vehicle_features"].shape[0])
        self.max_edge_servers = int(
            observation_space["edge_server_features"].shape[0]
        )
        self.max_resources = self.max_vehicles + self.max_edge_servers
        self.num_processor_types = MUX_NUM_PROCESSOR_TYPES

        expected_schedule_actions = (
            self.max_tasks * self.max_resources * self.num_processor_types
        )
        if expected_schedule_actions != MUX_SCHEDULE_ACTIONS:
            raise ValueError("Runtime dimensions do not match MUX_SCHEDULE_ACTIONS.")

        expected_actions = expected_schedule_actions + MUX_MIGRATION_ACTIONS
        if int(action_space.n) != expected_actions:
            raise ValueError(
                f"Expected {expected_actions} schedule+migration actions, "
                f"got {action_space.n}."
            )

        pair_feature_dim = int(observation_space["pair_features"].shape[-1])
        migration_feature_dim = int(
            observation_space["migration_features"].shape[-1]
        )

        self.encoder = MuxGNNEncoder(
            observation_space=observation_space,
            hidden_dim=hidden_dim,
            heads=heads,
            layers_per_relation=layers_per_relation,
            global_hidden_dim=global_hidden_dim,
            dropout=dropout,
        )

        # Offline muxGNN remains frozen during PPO training.
        self.encoder_frozen = False

        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_feature_dim, pair_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(pair_hidden_dim),
        )

        schedule_input_dim = (
            2 * hidden_dim + pair_hidden_dim + global_hidden_dim
        )
        self.actor = nn.Sequential(
            nn.Linear(schedule_input_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            # One logit for A7 and one for A12 per task-resource pair.
            nn.Linear(actor_hidden_dim, self.num_processor_types),
        )

        # Migration tokens combine the dynamic graph embedding with state that
        # only exists at runtime (progress, current placement, transfer cost...).
        migration_input_dim = (
            2 * hidden_dim + global_hidden_dim + migration_feature_dim
        )
        self.migration_input_projection = nn.Sequential(
            nn.Linear(migration_input_dim, migration_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(migration_hidden_dim),
        )
        self.migration_processor_embedding = nn.Embedding(
            self.num_processor_types,
            migration_hidden_dim,
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=migration_hidden_dim,
            nhead=migration_heads,
            dim_feedforward=migration_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.migration_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=migration_layers,
            enable_nested_tensor=False,
        )
        self.migration_actor = nn.Sequential(
            nn.LayerNorm(migration_hidden_dim),
            nn.Linear(migration_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, 1),
        )

        # mean + max for each node type plus the active migration context.
        pooled_dim = (
            6 * hidden_dim
            + global_hidden_dim
            + migration_hidden_dim
        )
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
        """Freeze the offline-pretrained graph encoder for PPO training."""
        self.encoder_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        """Keep the frozen graph encoder in eval mode while PPO heads train."""
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
                nn.init.orthogonal_(
                    module.weight,
                    gain=nn.init.calculate_gain("relu"),
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.migration_actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.migration_actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    @staticmethod
    def _ensure_batch(
        tensor: torch.Tensor,
        expected_without_batch: int,
    ) -> torch.Tensor:
        if tensor.ndim == expected_without_batch:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _masked_mean_max(
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.bool()
        mask_f = mask.unsqueeze(-1).to(embeddings.dtype)

        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (embeddings * mask_f).sum(dim=1) / count

        very_negative = torch.finfo(embeddings.dtype).min
        max_values = embeddings.masked_fill(
            ~mask.unsqueeze(-1),
            very_negative,
        ).max(dim=1).values

        has_any = mask.any(dim=1, keepdim=True)
        max_values = torch.where(
            has_any,
            max_values,
            torch.zeros_like(max_values),
        )
        return torch.cat([mean, max_values], dim=-1)

    def _migration_logits(
        self,
        task_embeddings: torch.Tensor,
        resource_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        migration_features = self._ensure_batch(
            observations["migration_features"],
            4,
        ).float()
        resource_mask = self._ensure_batch(
            observations["resource_mask"],
            1,
        ).bool()

        batch_size = task_embeddings.shape[0]
        hidden_dim = task_embeddings.shape[-1]
        global_dim = global_embedding.shape[-1]

        task_expanded = task_embeddings[:, :, None, None, :].expand(
            batch_size,
            self.max_tasks,
            self.max_resources,
            self.num_processor_types,
            hidden_dim,
        )
        resource_expanded = resource_embeddings[:, None, :, None, :].expand(
            batch_size,
            self.max_tasks,
            self.max_resources,
            self.num_processor_types,
            hidden_dim,
        )
        global_expanded = global_embedding[:, None, None, None, :].expand(
            batch_size,
            self.max_tasks,
            self.max_resources,
            self.num_processor_types,
            global_dim,
        )

        token_input = torch.cat(
            [
                task_expanded,
                resource_expanded,
                migration_features,
                global_expanded,
            ],
            dim=-1,
        )
        tokens = self.migration_input_projection(token_input)

        processor_ids = torch.arange(
            self.num_processor_types,
            device=tokens.device,
            dtype=torch.long,
        )
        processor_embedding = self.migration_processor_embedding(
            processor_ids
        ).view(1, 1, 1, self.num_processor_types, -1)
        tokens = tokens + processor_embedding

        token_count = self.max_resources * self.num_processor_types
        tokens = tokens.reshape(
            batch_size * self.max_tasks,
            token_count,
            -1,
        )

        candidate_mask = self._ensure_batch(
            observations["migration_candidate_mask"],
            3,
        ).bool()

        # Attend only over currently feasible KEEP/migration alternatives.
        # Tasks outside a migration window have no valid migration tokens, so
        # one harmless fallback token is unmasked only to keep Transformer
        # numerics finite; its logit is still removed by the PPO action mask.
        valid_tokens = candidate_mask & resource_mask[:, None, :, None]
        valid_tokens = valid_tokens.reshape(
            batch_size * self.max_tasks,
            token_count,
        )
        has_candidate = valid_tokens.any(dim=1)
        safe_valid_tokens = valid_tokens.clone()
        if torch.any(~has_candidate):
            safe_valid_tokens[~has_candidate, 0] = True

        key_padding_mask = ~safe_valid_tokens

        contextual = self.migration_transformer(
            tokens,
            src_key_padding_mask=key_padding_mask,
        )
        logits = self.migration_actor(contextual).squeeze(-1)

        # Give the critic the same runtime migration context seen by the actor.
        # Only currently exposed KEEP/migration candidates contribute.
        contextual_grid = contextual.reshape(
            batch_size,
            self.max_tasks,
            self.max_resources,
            self.num_processor_types,
            -1,
        )
        candidate_mask_f = candidate_mask.unsqueeze(-1).to(contextual_grid.dtype)
        candidate_count = candidate_mask_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
        migration_summary = (
            contextual_grid * candidate_mask_f
        ).sum(dim=(1, 2, 3)) / candidate_count

        return logits.reshape(batch_size, -1), migration_summary

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            task_embeddings,
            vehicle_embeddings,
            edge_server_embeddings,
            global_embedding,
            _,
            _,
        ) = self._encode(observations)

        pair_features = self._ensure_batch(
            observations["pair_features"],
            3,
        ).float()
        task_mask = self._ensure_batch(
            observations["task_mask"],
            1,
        ).bool()
        vehicle_mask = self._ensure_batch(
            observations["vehicle_mask"],
            1,
        ).bool()
        edge_server_mask = self._ensure_batch(
            observations["edge_server_mask"],
            1,
        ).bool()

        resource_embeddings = torch.cat(
            [vehicle_embeddings, edge_server_embeddings],
            dim=1,
        )
        pair_embeddings = self.pair_encoder(pair_features)
        batch_size = task_embeddings.shape[0]

        task_expanded = task_embeddings.unsqueeze(2).expand(
            batch_size,
            self.max_tasks,
            self.max_resources,
            task_embeddings.shape[-1],
        )
        resource_expanded = resource_embeddings.unsqueeze(1).expand(
            batch_size,
            self.max_tasks,
            self.max_resources,
            resource_embeddings.shape[-1],
        )
        global_expanded = global_embedding[:, None, None, :].expand(
            batch_size,
            self.max_tasks,
            self.max_resources,
            global_embedding.shape[-1],
        )

        schedule_input = torch.cat(
            [
                task_expanded,
                resource_expanded,
                pair_embeddings,
                global_expanded,
            ],
            dim=-1,
        )
        schedule_logits = self.actor(schedule_input).reshape(batch_size, -1)

        migration_logits, migration_summary = self._migration_logits(
            task_embeddings=task_embeddings,
            resource_embeddings=resource_embeddings,
            global_embedding=global_embedding,
            observations=observations,
        )

        logits = torch.cat(
            [schedule_logits, migration_logits],
            dim=1,
        )

        task_pool = self._masked_mean_max(task_embeddings, task_mask)
        vehicle_pool = self._masked_mean_max(vehicle_embeddings, vehicle_mask)
        edge_server_pool = self._masked_mean_max(
            edge_server_embeddings,
            edge_server_mask,
        )
        value_input = torch.cat(
            [
                task_pool,
                vehicle_pool,
                edge_server_pool,
                global_embedding,
                migration_summary,
            ],
            dim=-1,
        )
        values = self.critic(value_input)
        return logits, values

    @torch.no_grad()
    def relation_attention(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return mux coupling weights with shape [B, N, 5]."""
        _, _, _, _, coupling_weights, _ = self._encode(observations)
        return coupling_weights
