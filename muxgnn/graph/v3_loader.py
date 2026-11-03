from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch

from .v3_settings import V3_MANIFEST_PATH, V3_OUTPUT_DIR


class LegacyNativeV3DynamicDataset:
    """Scenario-split loader for Legacy-native Dynamic MAGNET v3 snapshots."""

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(self, split: str = "train") -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid v3 split: {split}")
        if not V3_MANIFEST_PATH.exists():
            raise FileNotFoundError(
                "v3 dynamic dataset does not exist. Run:\n"
                "python -m muxgnn.graph.generate_v3_dataset"
            )

        self.split = split
        self.manifest: Dict[str, object] = json.loads(
            V3_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        scenario_names = list(self.manifest["splits"][split])  # type: ignore[index]

        self.samples: List[Tuple[str, int, Path]] = []
        for scenario_name in scenario_names:
            snapshots_dir = V3_OUTPUT_DIR / split / str(scenario_name) / "snapshots"
            paths = sorted(snapshots_dir.glob("snapshot_*.pt"))
            if not paths:
                raise FileNotFoundError(f"No snapshots found for {scenario_name}.")
            for snapshot_index, path in enumerate(paths):
                self.samples.append((str(scenario_name), int(snapshot_index), path))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def __getitem__(self, index: int):
        return self._load(self.samples[index][2])

    def get_with_info(self, index: int):
        scenario_name, snapshot_index, path = self.samples[index]
        graph = self._load(path)
        return graph, {
            "split": self.split,
            "scenario": scenario_name,
            "snapshot_index": snapshot_index,
            "path": str(path),
        }

    def __iter__(self) -> Iterator:
        for index in range(len(self)):
            yield self[index]
