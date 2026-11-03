from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch

from .v4_settings import V4_MANIFEST_PATH, V4_OUTPUT_DIR


class LegacyNativeV4PhysicalDataset:
    """Scenario-split loader for v4 physical-time HeteroData snapshots."""

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(self, split: str = "train") -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid v4 split: {split}")
        if not V4_MANIFEST_PATH.exists():
            raise FileNotFoundError(
                "v4 dataset not generated. Run:\npython -m muxgnn.graph.generate_v4_dataset"
            )
        self.split = split
        self.manifest: Dict[str, object] = json.loads(
            V4_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        names = list(self.manifest["splits"][split])  # type: ignore[index]
        self.samples: List[Tuple[str, int, Path]] = []
        for name in names:
            snapshot_dir = V4_OUTPUT_DIR / split / str(name) / "snapshots"
            paths = sorted(snapshot_dir.glob("snapshot_*.pt"))
            if not paths:
                raise FileNotFoundError(f"No v4 snapshots found for {name}")
            for index, path in enumerate(paths):
                self.samples.append((str(name), int(index), path))

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
        scenario, snapshot_index, path = self.samples[index]
        return self._load(path), {
            "split": self.split,
            "scenario": scenario,
            "snapshot_index": snapshot_index,
            "path": str(path),
        }

    def __iter__(self) -> Iterator:
        for index in range(len(self)):
            yield self[index]
