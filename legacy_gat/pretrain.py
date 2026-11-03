from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ALLOW_MIXED_GANGS,
    DATASET_PATH,
    GAT_ATTENTION_DROPOUT,
    GAT_DROPOUT,
    GAT_HEADS,
    GAT_HIDDEN_DIM,
    GAT_LAYERS,
    GAT_PRETRAIN_EPOCHS,
    GAT_PRETRAIN_LR,
    MAX_GANG_SIZE,
    NUM_A12_CORES,
    NUM_A7_CORES,
    PRETRAINED_GAT_PATH,
    QOS_FACTOR,
    REWARD_MODE,
    REWARD_PROPOSAL,
    REWARD_WEIGHTS,
    RUNTIME_MODEL,
    SEED,
    STATIC_NODE_FEATURE_DIM,
)
from .env import DagSchedulingEnv
from .policy import StaticGATEncoder


PRETRAIN_OBJECTIVE = "full_pair_bce_only"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_env(seed: int) -> DagSchedulingEnv:
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


def _checkpoint_metadata(seed: int, max_tasks: int) -> Dict[str, object]:
    return {
        "dataset_sha256": sha256_file(DATASET_PATH),
        "seed": int(seed),
        "max_tasks": int(max_tasks),
        "static_feature_dim": STATIC_NODE_FEATURE_DIM,
        "hidden_dim": GAT_HIDDEN_DIM,
        "heads": GAT_HEADS,
        "layers": GAT_LAYERS,
        "attention_dropout": GAT_ATTENTION_DROPOUT,
        "feature_dropout": GAT_DROPOUT,
        "objective": PRETRAIN_OBJECTIVE,
        "epochs": GAT_PRETRAIN_EPOCHS,
        "learning_rate": GAT_PRETRAIN_LR,
    }


def checkpoint_matches_current_dataset(path: Path, seed: int, max_tasks: int) -> bool:
    if not path.exists():
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata", {})
    except Exception:
        return False
    expected = _checkpoint_metadata(seed=seed, max_tasks=max_tasks)
    keys = (
        "dataset_sha256",
        "seed",
        "max_tasks",
        "static_feature_dim",
        "hidden_dim",
        "heads",
        "layers",
        "objective",
    )
    return all(metadata.get(key) == expected[key] for key in keys)


def _all_ordered_node_pairs(real_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return every directed i->j pair over real nodes, excluding self-pairs."""
    src = real_nodes.repeat_interleave(real_nodes.numel())
    dst = real_nodes.repeat(real_nodes.numel())
    keep = src != dst
    return src[keep], dst[keep]


def _edge_labels(
    pair_src: torch.Tensor,
    pair_dst: torch.Tensor,
    edges: list[tuple[int, int]],
) -> torch.Tensor:
    """Binary adjacency labels for every ordered real-node pair."""
    edge_set = {(int(source), int(target)) for source, target in edges}
    labels = [
        1.0 if (int(source), int(target)) in edge_set else 0.0
        for source, target in zip(pair_src.tolist(), pair_dst.tolist())
    ]
    return torch.tensor(labels, dtype=torch.float32)


def _pair_representation(
    embeddings: torch.Tensor,
    pair_src: torch.Tensor,
    pair_dst: torch.Tensor,
) -> torch.Tensor:
    """Directed pair representation used by the structural edge classifier."""
    src = embeddings[0, pair_src, :]
    dst = embeddings[0, pair_dst, :]
    return torch.cat(
        [
            src,
            dst,
            torch.abs(dst - src),
            src * dst,
        ],
        dim=-1,
    )


def pretrain_offline_gat(
    seed: int = SEED,
    force: bool = False,
    epochs: int = GAT_PRETRAIN_EPOCHS,
) -> Path:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Gang dataset not found: {DATASET_PATH}. "
            "Run python run_legacy_gat.py generate first."
        )

    env = _make_env(seed)
    try:
        obs, _ = env.reset(seed=seed)
        max_tasks = int(env.max_tasks)

        if (
            not force
            and epochs == GAT_PRETRAIN_EPOCHS
            and checkpoint_matches_current_dataset(
                PRETRAINED_GAT_PATH,
                seed=seed,
                max_tasks=max_tasks,
            )
        ):
            print(f"Offline GAT checkpoint already valid: {PRETRAINED_GAT_PATH}")
            return PRETRAINED_GAT_PATH

        np.random.seed(seed)
        torch.manual_seed(seed)

        static_features = torch.as_tensor(
            obs["node_features"][:, :STATIC_NODE_FEATURE_DIM],
            dtype=torch.float32,
        ).unsqueeze(0)
        edge_index = torch.as_tensor(obs["edge_index"], dtype=torch.long).unsqueeze(0)
        edge_mask = torch.as_tensor(obs["edge_mask"], dtype=torch.bool).unsqueeze(0)
        node_mask = torch.as_tensor(obs["node_mask"], dtype=torch.bool).unsqueeze(0)

        encoder = StaticGATEncoder(
            max_tasks=max_tasks,
            static_feature_dim=STATIC_NODE_FEATURE_DIM,
            hidden_dim=GAT_HIDDEN_DIM,
            heads=GAT_HEADS,
            layers=GAT_LAYERS,
            attention_dropout=GAT_ATTENTION_DROPOUT,
            feature_dropout=GAT_DROPOUT,
        )

        pair_dim = 4 * GAT_HIDDEN_DIM
        edge_classifier = nn.Sequential(
            nn.Linear(pair_dim, GAT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(GAT_HIDDEN_DIM, 1),
        )
        optimizer = torch.optim.Adam(
            list(encoder.parameters())
            + list(edge_classifier.parameters()),
            lr=GAT_PRETRAIN_LR,
        )

        # torch.flatnonzero does not exist in supported PyTorch builds. This is
        # the equivalent, version-safe operation and has identical semantics.
        real_nodes = torch.nonzero(node_mask[0], as_tuple=False).flatten()
        if real_nodes.numel() < 2:
            raise RuntimeError("Offline GAT pretraining needs at least two real nodes.")

        pair_src, pair_dst = _all_ordered_node_pairs(real_nodes)
        edge_labels = _edge_labels(pair_src, pair_dst, env.edges)
        positive_count = float(edge_labels.sum().item())
        negative_count = float(edge_labels.numel() - positive_count)
        if positive_count <= 0.0:
            raise RuntimeError("Offline GAT pretraining needs at least one DAG edge.")
        # Full-pair BCE is highly imbalanced for a sparse DAG. Weighting only the
        # positive class keeps the requested complete BCE objective from being
        # dominated by the many non-edge pairs.
        pos_weight = torch.tensor(
            negative_count / positive_count,
            dtype=torch.float32,
        )

        effective_epochs = int(epochs)
        if effective_epochs <= 0:
            raise ValueError("epochs must be positive.")

        print("\n=== OFFLINE GAT PRETRAINING ===")
        print(f"Dataset          : {DATASET_PATH.name}")
        print(f"Tasks / edges    : {env.num_tasks} / {len(env.edges)}")
        print(f"Static features  : {STATIC_NODE_FEATURE_DIM}")
        print(f"Ordered pairs    : {edge_labels.numel()}")
        print(f"Positive edges   : {int(positive_count)}")
        print(f"Negative pairs   : {int(negative_count)}")
        print(f"Epochs           : {effective_epochs}")
        print("Objective        : full-pair BCE only\n")

        last_loss = float("nan")
        last_bce = float("nan")

        for epoch in range(1, effective_epochs + 1):
            encoder.train()
            edge_classifier.train()
            optimizer.zero_grad(set_to_none=True)

            embeddings = encoder(
                static_features,
                edge_index,
                edge_mask,
                node_mask,
            )
            pair_repr = _pair_representation(
                embeddings,
                pair_src,
                pair_dst,
            )

            edge_logits = edge_classifier(pair_repr).squeeze(-1)
            bce_loss = F.binary_cross_entropy_with_logits(
                edge_logits,
                edge_labels,
                pos_weight=pos_weight,
            )

            # The legacy DAG has no measured/native edge attributes. Therefore
            # offline GAT pretraining uses only structural supervision: predict
            # whether every ordered node pair is a real directed DAG edge.
            loss = bce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                + list(edge_classifier.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            last_loss = float(loss.detach().cpu().item())
            last_bce = float(bce_loss.detach().cpu().item())
            if epoch == 1 or epoch % 100 == 0 or epoch == effective_epochs:
                with torch.no_grad():
                    predicted_positive_all = edge_logits >= 0.0
                    tp = int(((edge_labels > 0.5) & predicted_positive_all).sum().item())
                    fp = int(((edge_labels <= 0.5) & predicted_positive_all).sum().item())
                    fn = int(((edge_labels > 0.5) & ~predicted_positive_all).sum().item())
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                print(
                    f"epoch={epoch:04d} "
                    f"loss={last_loss:.6f} "
                    f"bce={last_bce:.6f} "
                    f"tp={tp} "
                    f"precision={precision:.4f} "
                    f"recall={recall:.4f}"
                )

        encoder.eval()
        PRETRAINED_GAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        metadata = _checkpoint_metadata(seed=seed, max_tasks=max_tasks)
        metadata["epochs"] = effective_epochs
        metadata["final_loss"] = last_loss
        metadata["final_bce_loss"] = last_bce
        metadata["positive_pairs"] = int(positive_count)
        metadata["negative_pairs"] = int(negative_count)
        torch.save(
            {
                "encoder_state_dict": encoder.state_dict(),
                "metadata": metadata,
            },
            PRETRAINED_GAT_PATH,
        )
        print(f"\nSaved frozen-GAT source checkpoint: {PRETRAINED_GAT_PATH}")
        return PRETRAINED_GAT_PATH
    finally:
        env.close()


def ensure_offline_gat_checkpoint(seed: int = SEED) -> Path:
    """Create/rebuild the offline GAT checkpoint if it is missing or stale."""
    env = _make_env(seed)
    try:
        max_tasks = int(env.max_tasks)
    finally:
        env.close()

    if checkpoint_matches_current_dataset(
        PRETRAINED_GAT_PATH,
        seed=seed,
        max_tasks=max_tasks,
    ):
        return PRETRAINED_GAT_PATH
    return pretrain_offline_gat(seed=seed, force=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline-pretrain the legacy GAT encoder before PPO training."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=GAT_PRETRAIN_EPOCHS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pretrain_offline_gat(
        seed=args.seed,
        force=args.force,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
