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
        description=(
            "Route 2: MAGNET/muxGNN pipeline. Current stage: Legacy-native v2 graph, "
            "offline muxGNN smoke pretraining, then simple end-to-end PPO smoke training."
        )
    )
    parser.add_argument(
        "command",
        choices=("prepare", "validate", "pretrain", "train", "evaluate", "all"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    if args.command == "prepare":
        _run_module("muxgnn.graph.generate")
        _run_module("muxgnn.graph.validate")
        return

    if args.command == "validate":
        _run_module("muxgnn.rl.validate")
        return

    if args.command in {"pretrain", "all"}:
        _run_module(
            "muxgnn.rl.offline_pretrain",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        )

    if args.command in {"train", "all"}:
        train_args = [
            "--timesteps",
            str(args.timesteps),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if args.skip_evaluation:
            train_args.append("--skip-evaluation")
        _run_module("muxgnn.rl.simple_train", *train_args)

    if args.command == "evaluate":
        _run_module(
            "muxgnn.rl.simple_evaluate",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        )


if __name__ == "__main__":
    main()
