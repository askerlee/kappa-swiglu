"""Print lambda statistics from one or more model checkpoints.

Example:
    python scripts/lambda_stats.py path/to/model_000100.pt path/to/model_000200.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


LAMBDA_KEYS = ("resid_lambdas", "x0_lambdas", "ut_source_lambdas")


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


def load_lambdas(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    state_dict = load_state_dict(checkpoint_path)
    lambdas = {}
    for key in LAMBDA_KEYS:
        if key not in state_dict:
            raise KeyError(f"Checkpoint {checkpoint_path} is missing {key!r}")
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise TypeError(
                f"Checkpoint {checkpoint_path} entry {key!r} is not a tensor; "
                f"got {type(value).__name__}"
            )
        lambdas[key] = value
    return lambdas


def checkpoint_statistics(checkpoint_path: Path) -> dict[str, dict[str, float]]:
    return {
        key: calculate_statistics(value)
        for key, value in load_lambdas(checkpoint_path).items()
    }


def format_values(values) -> str:
    if isinstance(values, list):
        return "[" + ", ".join(format_values(value) for value in values) + "]"
    return f"{values:.2f}"


def print_statistics(
    checkpoint_path: Path,
    statistics: Mapping[str, Mapping[str, float]],
    lambdas: Mapping[str, torch.Tensor],
) -> None:
    print(f"Checkpoint: {checkpoint_path}")
    for key in LAMBDA_KEYS:
        stats = statistics[key]
        print(
            f"  {key}: mean={stats['mean']:.2f} std={stats['std']:.2f} "
            f"max={stats['max']:.2f} min={stats['min']:.2f} "
            f"abs_max={stats['abs_max']:.2f}"
        )
    print(f"resid {format_values(lambdas['resid_lambdas'].detach().float().tolist())}")
    print(f"x0    {format_values(lambdas['x0_lambdas'].detach().float().tolist())}")
    print(f"source {format_values(lambdas['ut_source_lambdas'].detach().float().tolist())}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate scalar-mixing lambda statistics in model checkpoints"
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
        lambdas = load_lambdas(checkpoint_path)
        statistics = {
            key: calculate_statistics(value)
            for key, value in lambdas.items()
        }
        print_statistics(checkpoint_path, statistics, lambdas)


if __name__ == "__main__":
    main()