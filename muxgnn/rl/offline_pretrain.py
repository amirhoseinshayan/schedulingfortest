from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph.loader import LegacyNativeV2Dataset
from ..graph.settings import DATASET_VERSION, GRAPH_PATH, METADATA_PATH
from .adapter import HeteroMuxAdapter
from .config import (
    MUX_DROPOUT,
    MUX_ENCODER_CHECKPOINT_PATH,
    MUX_ENCODER_DIR,
    MUX_ENCODER_PRETRAIN_CONFIG_PATH,
    MUX_GLOBAL_HIDDEN_DIM,
    MUX_HEADS,
    MUX_HIDDEN_DIM,
    MUX_LAYERS_PER_RELATION,
    MUX_NUM_RELATIONS,
    MUX_SEED,
)
from .model import MuxGNNEncoder


DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MASK_RATIO = 0.15
DEFAULT_NODE_LOSS_WEIGHT = 0.25
DEFAULT_RELATION_LOSS_WEIGHT = 1.0


class OfflineMuxGNNPretrainer(nn.Module):
    """Self-supervised wrapper used only for offline muxGNN pretraining.

    The encoder sees corrupted node features and the original multiplex graph.
    It is trained to:
      1) reconstruct the masked node features; and
      2) reconstruct each of the five relation adjacencies.

    Only ``encoder.state_dict()`` is exported for PPO. The temporary decoders
    below are discarded after pretraining.
    """

    def __init__(
        self,
        observation_space,
        hidden_dim: int,
        heads: int,
        layers_per_relation: int,
        global_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.encoder = MuxGNNEncoder(
            observation_space=observation_space,
            hidden_dim=hidden_dim,
            heads=heads,
            layers_per_relation=layers_per_relation,
            global_hidden_dim=global_hidden_dim,
            dropout=dropout,
        )

        task_dim = int(observation_space["task_features"].shape[-1])
        vehicle_dim = int(observation_space["vehicle_features"].shape[-1])
        edge_dim = int(observation_space["edge_server_features"].shape[-1])

        self.task_decoder = nn.Linear(hidden_dim, task_dim)
        self.vehicle_decoder = nn.Linear(hidden_dim, vehicle_dim)
        self.edge_decoder = nn.Linear(hidden_dim, edge_dim)

        # Directional relation reconstruction: target/source projections are
        # separate so the decoder can represent asymmetric relations.
        self.relation_source = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(MUX_NUM_RELATIONS)]
        )
        self.relation_target = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(MUX_NUM_RELATIONS)]
        )
        self.hidden_dim = int(hidden_dim)

    @staticmethod
    def _ensure_one_masked(mask: torch.Tensor, ratio: float) -> torch.Tensor:
        selected = mask.bool() & (torch.rand_like(mask.float()) < float(ratio))

        # Each batch item should contribute at least one reconstruction target
        # for every real node type that exists.
        for batch_id in range(mask.shape[0]):
            if mask[batch_id].any() and not selected[batch_id].any():
                first = torch.nonzero(mask[batch_id], as_tuple=False)[0, 0]
                selected[batch_id, first] = True

        return selected

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not mask.any():
            return prediction.sum() * 0.0

        expanded = mask.unsqueeze(-1).expand_as(prediction)
        return F.mse_loss(prediction[expanded], target[expanded])

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        mask_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        original_task = observations["task_features"].float()
        original_vehicle = observations["vehicle_features"].float()
        original_edge = observations["edge_server_features"].float()

        task_mask = observations["task_mask"].bool()
        vehicle_mask = observations["vehicle_mask"].bool()
        edge_mask = observations["edge_server_mask"].bool()
        node_mask = observations["node_mask"].bool()

        task_reconstruction_mask = self._ensure_one_masked(task_mask, mask_ratio)
        vehicle_reconstruction_mask = self._ensure_one_masked(vehicle_mask, mask_ratio)
        edge_reconstruction_mask = self._ensure_one_masked(edge_mask, mask_ratio)

        corrupted = dict(observations)
        corrupted["task_features"] = original_task.clone()
        corrupted["vehicle_features"] = original_vehicle.clone()
        corrupted["edge_server_features"] = original_edge.clone()

        corrupted["task_features"][task_reconstruction_mask] = 0.0
        corrupted["vehicle_features"][vehicle_reconstruction_mask] = 0.0
        corrupted["edge_server_features"][edge_reconstruction_mask] = 0.0

        (
            task_embeddings,
            vehicle_embeddings,
            edge_embeddings,
            _,
            _,
            _,
        ) = self.encoder(corrupted)

        task_prediction = self.task_decoder(task_embeddings)
        vehicle_prediction = self.vehicle_decoder(vehicle_embeddings)
        edge_prediction = self.edge_decoder(edge_embeddings)

        task_loss = self._masked_mse(
            task_prediction,
            original_task,
            task_reconstruction_mask,
        )
        vehicle_loss = self._masked_mse(
            vehicle_prediction,
            original_vehicle,
            vehicle_reconstruction_mask,
        )
        edge_loss = self._masked_mse(
            edge_prediction,
            original_edge,
            edge_reconstruction_mask,
        )
        node_loss = (task_loss + vehicle_loss + edge_loss) / 3.0

        all_nodes = torch.cat(
            [task_embeddings, vehicle_embeddings, edge_embeddings],
            dim=1,
        )
        adjacency = observations["relation_adjacency"].float()

        real_pair_mask = (
            node_mask.unsqueeze(1).unsqueeze(-1)
            & node_mask.unsqueeze(1).unsqueeze(-2)
        )

        relation_losses: List[torch.Tensor] = []
        scale = math.sqrt(float(self.hidden_dim))

        for relation_id in range(MUX_NUM_RELATIONS):
            source = self.relation_source[relation_id](all_nodes)
            target = self.relation_target[relation_id](all_nodes)

            # [B, target, source], matching adapter.relation_adjacency.
            logits = torch.einsum("bth,bsh->bts", target, source) / scale
            labels = adjacency[:, relation_id]
            valid = real_pair_mask[:, 0]

            relation_logits = logits[valid]
            relation_labels = labels[valid]

            positives = relation_labels.sum()
            negatives = relation_labels.numel() - positives
            pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 50.0)

            relation_losses.append(
                F.binary_cross_entropy_with_logits(
                    relation_logits,
                    relation_labels,
                    pos_weight=pos_weight,
                )
            )

        relation_loss = torch.stack(relation_losses).mean()

        return {
            "node_loss": node_loss,
            "relation_loss": relation_loss,
        }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _tensorize_batch(
    observations: Iterable[Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    observation_list = list(observations)
    keys = observation_list[0].keys()

    return {
        key: torch.as_tensor(
            np.stack([obs[key] for obs in observation_list], axis=0),
            device=device,
        )
        for key in keys
    }


def _iter_batches(indices: List[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _load_batch(
    dataset: LegacyNativeV2Dataset,
    adapter: HeteroMuxAdapter,
    indices: List[int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    observations = [adapter.to_observation(dataset[index]) for index in indices]
    return _tensorize_batch(observations, device)


def _run_epoch(
    model: OfflineMuxGNNPretrainer,
    dataset: LegacyNativeV2Dataset,
    adapter: HeteroMuxAdapter,
    indices: List[int],
    device: torch.device,
    batch_size: int,
    mask_ratio: float,
    node_loss_weight: float,
    relation_loss_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_node = 0.0
    total_relation = 0.0
    total_samples = 0

    for batch_indices in _iter_batches(indices, batch_size):
        batch = _load_batch(dataset, adapter, batch_indices, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            losses = model(batch, mask_ratio=mask_ratio)
            loss = (
                float(node_loss_weight) * losses["node_loss"]
                + float(relation_loss_weight) * losses["relation_loss"]
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        count = len(batch_indices)
        total_samples += count
        total_loss += float(loss.detach().cpu()) * count
        total_node += float(losses["node_loss"].detach().cpu()) * count
        total_relation += float(losses["relation_loss"].detach().cpu()) * count

    denominator = max(total_samples, 1)
    return {
        "loss": total_loss / denominator,
        "node_loss": total_node / denominator,
        "relation_loss": total_relation / denominator,
    }


def _limit_indices(length: int, maximum: int | None) -> List[int]:
    indices = list(range(length))
    if maximum is not None:
        indices = indices[: max(int(maximum), 0)]
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-train bridge for the Legacy-native v2 muxGNN encoder. "
            "This stage verifies adapter/encoder/loss/backward/checkpoint on the single static v2 graph."
        )
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--mask-ratio", type=float, default=DEFAULT_MASK_RATIO)
    parser.add_argument("--node-loss-weight", type=float, default=DEFAULT_NODE_LOSS_WEIGHT)
    parser.add_argument(
        "--relation-loss-weight",
        type=float,
        default=DEFAULT_RELATION_LOSS_WEIGHT,
    )
    parser.add_argument("--seed", type=int, default=MUX_SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=MUX_ENCODER_CHECKPOINT_PATH,
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if not 0.0 < args.mask_ratio <= 1.0:
        raise ValueError("--mask-ratio must be in (0, 1].")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _resolve_device(args.device)
    adapter = HeteroMuxAdapter()
    # Smoke-Train Bridge intentionally uses the same single v2 graph for the
    # optimization step and a deterministic corruption-based smoke evaluation.
    # This is NOT an independent validation split and is never reported as
    # paper-grade generalization evidence.
    train_dataset = LegacyNativeV2Dataset()
    val_dataset = LegacyNativeV2Dataset()

    train_indices = [0]
    val_indices = [0]

    model = OfflineMuxGNNPretrainer(
        observation_space=adapter.observation_space,
        hidden_dim=MUX_HIDDEN_DIM,
        heads=MUX_HEADS,
        layers_per_relation=MUX_LAYERS_PER_RELATION,
        global_hidden_dim=MUX_GLOBAL_HIDDEN_DIM,
        dropout=MUX_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=1e-5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    MUX_ENCODER_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []

    print("========================================")
    print("muxGNN v2 Smoke-Train Bridge pretraining")
    print("========================================")
    print(f"Dataset version : {DATASET_VERSION}")
    print(f"Static graphs   : {len(train_indices)}")
    print("Validation note : same graph, deterministic corruption only")
    print(f"Device          : {device}")
    print(f"Checkpoint      : {args.output}")

    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(train_indices)

        train_metrics = _run_epoch(
            model=model,
            dataset=train_dataset,
            adapter=adapter,
            indices=train_indices,
            device=device,
            batch_size=int(args.batch_size),
            mask_ratio=float(args.mask_ratio),
            node_loss_weight=float(args.node_loss_weight),
            relation_loss_weight=float(args.relation_loss_weight),
            optimizer=optimizer,
        )

        # Fixed seed makes the validation corruption reproducible per epoch.
        torch.manual_seed(args.seed + 10_000)
        val_metrics = _run_epoch(
            model=model,
            dataset=val_dataset,
            adapter=adapter,
            indices=val_indices,
            device=device,
            batch_size=int(args.batch_size),
            mask_ratio=float(args.mask_ratio),
            node_loss_weight=float(args.node_loss_weight),
            relation_loss_weight=float(args.relation_loss_weight),
            optimizer=None,
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
        )

        print(
            f"epoch={epoch:03d} "
            f"train={train_metrics['loss']:.6f} "
            f"val={val_metrics['loss']:.6f} "
            f"node={val_metrics['node_loss']:.6f} "
            f"relation={val_metrics['relation_loss']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "version": 2,
                    "training_mode": "legacy_native_v2_single_graph_smoke_pretrain",
                    "paper_grade_validation": False,
                    "dataset_version": DATASET_VERSION,
                    "graph_path": str(GRAPH_PATH),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "architecture": {
                        "hidden_dim": MUX_HIDDEN_DIM,
                        "heads": MUX_HEADS,
                        "layers_per_relation": MUX_LAYERS_PER_RELATION,
                        "global_hidden_dim": MUX_GLOBAL_HIDDEN_DIM,
                        "dropout": MUX_DROPOUT,
                    },
                    "best_validation_loss": float(best_val),
                    "epoch": int(epoch),
                },
                args.output,
            )

    dataset_metadata = {}
    if METADATA_PATH.exists():
        dataset_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    config_payload = {
        "mode": "legacy_native_v2_muxgnn_smoke_train_bridge",
        "paper_grade_validation": False,
        "validation_semantics": "same static graph with deterministic masking; smoke test only",
        "dataset_version": DATASET_VERSION,
        "graph_path": str(GRAPH_PATH),
        "dataset_source_sha256": dataset_metadata.get("source", {}).get("legacy_dataset_sha256"),
        "ppo_encoder_policy": "checkpoint is structurally loadable/freezeable; PPO integration is a later stage",
        "checkpoint": str(args.output),
        "seed": int(args.seed),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "mask_ratio": float(args.mask_ratio),
        "node_loss_weight": float(args.node_loss_weight),
        "relation_loss_weight": float(args.relation_loss_weight),
        "train_graphs": len(train_indices),
        "smoke_eval_graphs": len(val_indices),
        "best_validation_loss": float(best_val),
        "history": history,
    }

    MUX_ENCODER_PRETRAIN_CONFIG_PATH.write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    print("\n========================================")
    print("muxGNN v2 Smoke-Train Bridge PASSED")
    print("========================================")
    print(f"Saved encoder : {args.output}")
    print(f"Config/history: {MUX_ENCODER_PRETRAIN_CONFIG_PATH}")
    print("Important: this checkpoint comes from ONE static graph and is for smoke validation, not final paper training.")


if __name__ == "__main__":
    main()
