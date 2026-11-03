"""Backward-compatible entry point for legacy GAT/PPO dataset generation.

The audited implementation lives under ``generate_data/gat + ppo``.  This file
intentionally remains so existing commands (``make.bat generate`` and
``run_legacy_gat.py generate``) do not change.
"""
from __future__ import annotations

import runpy
from pathlib import Path
from typing import Sequence

_GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "generate_data"
    / "gat + ppo"
    / "legacy_gat_generator.py"
)


def main(argv: Sequence[str] | None = None):
    namespace = runpy.run_path(str(_GENERATOR_PATH), run_name="gat_ppo_dataset_generator")
    return namespace["main"](argv)


if __name__ == "__main__":
    main()
