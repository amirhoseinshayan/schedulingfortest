from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Tuple

import torch

from .dynamic_settings import (
    DYNAMIC_OUTPUT_DIR,
    MANIFEST_PATH,
)


class DynamicHeteroDataset:
    """Simple snapshot dataset loader."""

    VALID_SPLITS = {
        "train",
        "val",
        "test",
    }


    def __init__(
        self,
        split: str = "train",
    ) -> None:

        if (
            split
            not in self.VALID_SPLITS
        ):

            raise ValueError(
                f"Invalid split: "
                f"{split}"
            )


        if not (
            MANIFEST_PATH.exists()
        ):

            raise FileNotFoundError(
                "Dynamic dataset "
                "does not exist."
            )


        self.split = split


        manifest = json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )


        scenario_names = (
            manifest[
                "splits"
            ][
                split
            ]
        )


        self.samples: List[
            Tuple[
                str,
                int,
                Path,
            ]
        ] = []


        for scenario_name in (
            scenario_names
        ):

            snapshots_dir = (
                DYNAMIC_OUTPUT_DIR
                / split
                / scenario_name
                / "snapshots"
            )


            snapshot_paths = sorted(
                snapshots_dir.glob(
                    "snapshot_*.pt"
                )
            )


            for (
                snapshot_index,
                snapshot_path,
            ) in enumerate(
                snapshot_paths
            ):

                self.samples.append(
                    (
                        scenario_name,

                        snapshot_index,

                        snapshot_path,
                    )
                )


    def __len__(
        self,
    ) -> int:

        return len(
            self.samples
        )


    @staticmethod
    def _load(
        path: Path,
    ):

        try:

            return torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:

            return torch.load(
                path,
                map_location="cpu",
            )


    def __getitem__(
        self,
        index: int,
    ):

        (
            scenario_name,
            snapshot_index,
            path,
        ) = self.samples[
            index
        ]


        graph = self._load(
            path
        )


        return graph


    def get_with_info(
        self,
        index: int,
    ):

        (
            scenario_name,
            snapshot_index,
            path,
        ) = self.samples[
            index
        ]


        graph = self._load(
            path
        )


        info = {

            "scenario":
                scenario_name,

            "snapshot_index":
                snapshot_index,

            "path":
                str(path),
        }


        return (
            graph,
            info,
        )


    def __iter__(
        self,
    ) -> Iterator:

        for index in range(
            len(self)
        ):

            yield self[
                index
            ]

class LegacyNativeV2Dataset:
    """Single-graph loader for the v2 -> muxGNN Smoke-Train Bridge.

    This is intentionally a one-sample dataset.  It is only for verifying the
    adapter, muxGNN forward pass, self-supervised loss, backward pass, and
    checkpoint serialization before dynamic v3 snapshots are introduced.
    """

    def __init__(self) -> None:
        from .settings import DATASET_VERSION, GRAPH_PATH

        self.graph_path = GRAPH_PATH
        self.dataset_version = DATASET_VERSION

        if not self.graph_path.exists():
            raise FileNotFoundError(
                "Legacy-native v2 graph does not exist. Run first:\n"
                "python run_muxgnn.py prepare"
            )

        self._graph = self._load(self.graph_path)

        graph_version = getattr(self._graph, "dataset_version", None)
        if graph_version is not None and str(graph_version) != self.dataset_version:
            raise ValueError(
                f"Expected dataset_version={self.dataset_version}, got {graph_version}."
            )

    def __len__(self) -> int:
        return 1

    @staticmethod
    def _load(path: Path):
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            return torch.load(
                path,
                map_location="cpu",
            )

    def __getitem__(self, index: int):
        if index not in (0, -1):
            raise IndexError(index)
        return self._graph

    def get_with_info(self, index: int):
        graph = self[index]
        return graph, {
            "dataset_version": self.dataset_version,
            "sample_type": "single_static_v2_graph",
            "path": str(self.graph_path),
        }

    def __iter__(self) -> Iterator:
        yield self._graph

