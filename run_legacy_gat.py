from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run_module(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route 1: legacy PPO + GAT gang-scheduling pipeline."
    )
    parser.add_argument(
        "command",
        choices=("generate", "validate", "pretrain", "train", "evaluate", "all"),
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.command in {"generate", "all"}:
        _run_module("legacy_gat.generate_dataset")
    if args.command in {"validate", "all"}:
        _run_module("legacy_gat.validate")
    if args.command in {"pretrain", "all"}:
        _run_module("legacy_gat.pretrain", "--seed", str(args.seed))
    if args.command in {"train", "all"}:
        _run_module(
            "legacy_gat.train",
            "--timesteps",
            str(args.timesteps),
            "--seed",
            str(args.seed),
        )
    if args.command in {"evaluate", "all"}:
        _run_module("legacy_gat.evaluate")


if __name__ == "__main__":
    main()
