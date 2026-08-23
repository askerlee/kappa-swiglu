"""Print residual-lambda statistics from one or more model checkpoints.

Example:
    python scripts/lambda_stats.py path/to/model_000100.pt path/to/model_000200.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


LAMBDA_KEYS = ("resid_lambdas", "x0_lambdas")


def load_state_dict(checkpoint_path: Path) -> Mapping:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint {checkpoint_path} did not load as a mapping; "
            f"got {type(checkpoint).__name__}"
        )
    for wrapper_key in ("model", "state_dict"):
        wrapped = checkpoint.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            return wrapped
    return checkpoint


def calculate_statistics(value: torch.Tensor) -> dict[str, float]:
    if not value.is_floating_point():
        raise TypeError(f"expected a floating-point tensor, got {value.dtype}")
    if value.numel() == 0:
        raise ValueError("cannot calculate statistics for an empty tensor")

    values = value.detach().reshape(-1).to(dtype=torch.float64)
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "max": values.max().item(),
        "min": values.min().item(),
        "abs_max": values.abs().max().item(),
    }


def checkpoint_statistics(checkpoint_path: Path) -> dict[str, dict[str, float]]:
    state_dict = load_state_dict(checkpoint_path)
    statistics = {}
    for key in LAMBDA_KEYS:
        if key not in state_dict:
            raise KeyError(f"Checkpoint {checkpoint_path} is missing {key!r}")
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise TypeError(
                f"Checkpoint {checkpoint_path} entry {key!r} is not a tensor; "
                f"got {type(value).__name__}"
            )
        statistics[key] = calculate_statistics(value)
    return statistics


def print_statistics(checkpoint_path: Path, statistics: Mapping[str, Mapping[str, float]]) -> None:
    print(f"Checkpoint: {checkpoint_path}")
    for key in LAMBDA_KEYS:
        stats = statistics[key]
        print(
            f"  {key}: mean={stats['mean']:.8g} std={stats['std']:.8g} "
            f"max={stats['max']:.8g} min={stats['min']:.8g} "
            f"abs_max={stats['abs_max']:.8g}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate resid_lambdas and x0_lambdas statistics in model checkpoints"
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="paths to model checkpoint files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for index, checkpoint_path in enumerate(args.checkpoints):
        if index:
            print()
        print_statistics(checkpoint_path, checkpoint_statistics(checkpoint_path))


if __name__ == "__main__":
    main()