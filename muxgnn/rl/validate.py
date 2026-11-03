from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from ..graph.loader import LegacyNativeV2Dataset
from ..graph.settings import DATASET_VERSION
from .adapter import (
    GLOBAL_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    HeteroMuxAdapter,
    RELATION_NAMES,
)
from .model import MuxGNNEncoder
from .offline_pretrain import OfflineMuxGNNPretrainer


def _tensorize(obs: dict) -> dict:
    return {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in obs.items()
    }


def _validate_adapter() -> tuple[HeteroMuxAdapter, dict]:
    dataset = LegacyNativeV2Dataset()
    graph = dataset[0]
    adapter = HeteroMuxAdapter()
    obs = adapter.to_observation(graph)

    if not adapter.observation_space.contains(obs):
        raise AssertionError("Adapter output is outside the declared Gym space.")

    if len(RELATION_NAMES) != 5:
        raise AssertionError("Smoke bridge requires exactly five muxGNN relations.")
    if len(PAIR_FEATURE_NAMES) != obs["pair_features"].shape[-1]:
        raise AssertionError("Pair feature schema mismatch.")
    if len(GLOBAL_FEATURE_NAMES) != obs["global_features"].shape[-1]:
        raise AssertionError("Global feature schema mismatch.")

    if int(obs["task_mask"].sum()) != int(graph["task"].num_nodes):
        raise AssertionError("Task padding mask mismatch.")
    if int(obs["vehicle_mask"].sum()) != int(graph["vehicle"].num_nodes):
        raise AssertionError("Vehicle padding mask mismatch.")
    if int(obs["edge_server_mask"].sum()) != int(graph["edge_server"].num_nodes):
        raise AssertionError("Edge-server padding mask mismatch.")
    if int(obs["action_mask"].sum()) <= 0:
        raise AssertionError("v2 initial graph exposes no valid scheduling action.")

    for key, value in obs.items():
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise AssertionError(f"Observation field {key} contains NaN/Inf.")

    print("[OK] Legacy-native v2 graph loads through the static dataset loader")
    print("[OK] v2 HeteroData -> fixed dense muxGNN adapter")
    print(f"[OK] Pair schema: {len(PAIR_FEATURE_NAMES)} features")
    print(f"[OK] Global schema: {len(GLOBAL_FEATURE_NAMES)} features")
    print(f"[OK] Valid initial schedule actions: {int(obs['action_mask'].sum())}")

    return adapter, obs


def _validate_encoder_forward(adapter: HeteroMuxAdapter, obs: dict) -> None:
    encoder = MuxGNNEncoder(
        observation_space=adapter.observation_space,
        hidden_dim=32,
        heads=4,
        layers_per_relation=1,
        global_hidden_dim=16,
        dropout=0.0,
    )
    tensors = _tensorize(obs)

    with torch.no_grad():
        (
            task_embeddings,
            vehicle_embeddings,
            edge_embeddings,
            global_embedding,
            attention,
            node_mask,
        ) = encoder(tensors)

    if task_embeddings.shape != (1, adapter.max_tasks, 32):
        raise AssertionError(f"Unexpected task embedding shape: {task_embeddings.shape}")
    if vehicle_embeddings.shape != (1, adapter.max_vehicles, 32):
        raise AssertionError(
            f"Unexpected vehicle embedding shape: {vehicle_embeddings.shape}"
        )
    if edge_embeddings.shape != (1, adapter.max_edge_servers, 32):
        raise AssertionError(
            f"Unexpected edge-server embedding shape: {edge_embeddings.shape}"
        )
    if global_embedding.shape != (1, 16):
        raise AssertionError(f"Unexpected global embedding shape: {global_embedding.shape}")
    if attention.shape != (1, adapter.max_nodes, 5):
        raise AssertionError(f"Unexpected coupling-attention shape: {attention.shape}")

    for name, tensor in {
        "task_embeddings": task_embeddings,
        "vehicle_embeddings": vehicle_embeddings,
        "edge_embeddings": edge_embeddings,
        "global_embedding": global_embedding,
        "coupling_attention": attention,
    }.items():
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} contains NaN/Inf.")

    relation_presence = torch.as_tensor(obs["relation_presence"]).bool()
    present_any = relation_presence.any(dim=0)
    real_nodes = torch.as_tensor(obs["node_mask"]).bool()
    check_nodes = real_nodes & present_any
    attention_sums = attention[0].sum(dim=-1)
    if not torch.allclose(
        attention_sums[check_nodes],
        torch.ones_like(attention_sums[check_nodes]),
        atol=1e-5,
    ):
        raise AssertionError("Coupling attention does not sum to 1 on active nodes.")

    if not torch.equal(node_mask[0].cpu(), real_nodes):
        raise AssertionError("Encoder node mask changed unexpectedly.")

    print("[OK] muxGNN encoder forward pass")
    print("[OK] Five relation-specific views + coupling attention")
    print("[OK] Embeddings/attention are finite and correctly shaped")


def _validate_loss_backward_and_checkpoint(
    adapter: HeteroMuxAdapter,
    obs: dict,
) -> None:
    torch.manual_seed(123)

    model = OfflineMuxGNNPretrainer(
        observation_space=adapter.observation_space,
        hidden_dim=32,
        heads=4,
        layers_per_relation=1,
        global_hidden_dim=16,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tensors = _tensorize(obs)

    tracked_parameter = next(
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    )
    before = tracked_parameter.detach().clone()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model(tensors, mask_ratio=0.15)
    total_loss = 0.25 * losses["node_loss"] + losses["relation_loss"]

    if not torch.isfinite(total_loss):
        raise AssertionError("Smoke pretraining loss is NaN/Inf.")

    total_loss.backward()

    finite_gradients = []
    nonzero_gradient_seen = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        finite_gradients.append(bool(torch.isfinite(parameter.grad).all()))
        if bool(torch.any(parameter.grad != 0)):
            nonzero_gradient_seen = True

    if not finite_gradients or not all(finite_gradients):
        raise AssertionError("Smoke backward pass produced missing/non-finite gradients.")
    if not nonzero_gradient_seen:
        raise AssertionError("Smoke backward pass produced only zero gradients.")

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    after = tracked_parameter.detach()
    if torch.equal(before, after):
        raise AssertionError("Optimizer step did not update the muxGNN encoder.")

    # Verify that the encoder state can be serialized and loaded strictly into a
    # fresh encoder with the same v2 observation contract.
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "muxgnn_v2_smoke_encoder.pt"
        torch.save(
            {
                "dataset_version": DATASET_VERSION,
                "training_mode": "v2_smoke_bridge",
                "encoder_state_dict": model.encoder.state_dict(),
            },
            checkpoint_path,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        restored = MuxGNNEncoder(
            observation_space=adapter.observation_space,
            hidden_dim=32,
            heads=4,
            layers_per_relation=1,
            global_hidden_dim=16,
            dropout=0.0,
        )
        restored.load_state_dict(checkpoint["encoder_state_dict"], strict=True)

    print(
        "[OK] Smoke loss finite: "
        f"total={float(total_loss.detach()):.6f}, "
        f"node={float(losses['node_loss'].detach()):.6f}, "
        f"relation={float(losses['relation_loss'].detach()):.6f}"
    )
    print("[OK] Backward pass + non-zero finite gradients")
    print("[OK] Optimizer updates muxGNN encoder parameters")
    print("[OK] Encoder checkpoint strict save/load")


def main() -> None:
    adapter, obs = _validate_adapter()
    _validate_encoder_forward(adapter, obs)
    _validate_loss_backward_and_checkpoint(adapter, obs)

    print("\n========================================")
    print("v2 -> muxGNN Smoke-Train Bridge PASSED")
    print("========================================")
    print("Scope: static v2 graph only; no PPO/environment/dynamic-v3 validation yet.")


if __name__ == "__main__":
    main()
