"""Print scalar statistics from one or more model checkpoints.

Example:
    python scripts/scalar_stats.py path/to/model_000100.pt path/to/model_000200.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


LAMBDA_KEYS = ("resid_lambdas", "x0_lambdas", "ut_source_lambdas")
KAPPA_NAMES = ("kappa_scale", "kappa_bias")


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


def load_scalars(checkpoint_path: Path) -> dict[str, torch.Tensor]:
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

    resid_lambdas = lambdas["resid_lambdas"]
    total_ut_steps = resid_lambdas.shape[0] if resid_lambdas.ndim == 2 else 1

    scalars = dict(lambdas)
    for name in KAPPA_NAMES:
        values = [
            value
            for key, value in state_dict.items()
            if key == name or key == f"global_{name}" or key.endswith(f".{name}")
        ]
        if not values:
            continue
        if not all(torch.is_tensor(value) for value in values):
            raise TypeError(
                f"Checkpoint {checkpoint_path} has a non-tensor {name!r} entry"
            )
        if total_ut_steps > 1:
            invalid_shapes = [
                tuple(value.shape)
                for value in values
                if value.ndim < 2 or value.shape[0] != total_ut_steps
            ]
            if invalid_shapes:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} has {name} tensors whose leading "
                    f"dimension does not match total_ut_steps={total_ut_steps}: "
                    f"{invalid_shapes}"
                )
            scalars[name] = torch.cat(
                [value.detach().reshape(total_ut_steps, -1) for value in values], dim=1
            )
        else:
            scalars[name] = torch.cat([value.detach().reshape(-1) for value in values])
    return scalars


def statistics_for_scalar(value: torch.Tensor):
    if value.ndim == 2 and value.shape[0] > 1:
        return [calculate_statistics(step) for step in value]
    return calculate_statistics(value)


def checkpoint_statistics(checkpoint_path: Path) -> dict:
    return {
        key: statistics_for_scalar(value) if key in KAPPA_NAMES else calculate_statistics(value)
        for key, value in load_scalars(checkpoint_path).items()
    }


def format_values(values) -> str:
    if isinstance(values, list):
        return "[" + ", ".join(format_values(value) for value in values) + "]"
    return f"{values:.2f}"


def print_statistics(
    checkpoint_path: Path,
    statistics: Mapping,
    scalars: Mapping[str, torch.Tensor],
) -> None:
    print(f"Checkpoint: {checkpoint_path}")
    for key, scalar_stats in statistics.items():
        if key == "ut_source_lambdas":
            continue
        step_stats = scalar_stats if isinstance(scalar_stats, list) else [scalar_stats]
        for step, stats in enumerate(step_stats):
            step_label = f" step={step}" if isinstance(scalar_stats, list) else ""
            print(
                f"  {key}{step_label}: mean={stats['mean']:.2f} "
                f"std={stats['std']:.2f} max={stats['max']:.2f} "
                f"min={stats['min']:.2f} abs_max={stats['abs_max']:.2f}"
            )
    print(f"resid {format_values(scalars['resid_lambdas'].detach().float().tolist())}")
    print(f"x0    {format_values(scalars['x0_lambdas'].detach().float().tolist())}")
    print(f"source {format_values(scalars['ut_source_lambdas'].detach().float().tolist())}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate lambda and kappa scalar statistics in model checkpoints"
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
        scalars = load_scalars(checkpoint_path)
        statistics = {
            key: statistics_for_scalar(value) if key in KAPPA_NAMES else calculate_statistics(value)
            for key, value in scalars.items()
        }
        print_statistics(checkpoint_path, statistics, scalars)


if __name__ == "__main__":
    main()