"""Print scalar statistics from one or more model checkpoints.

Example:
    python scripts/scalar_stats.py path/to/model_000100.pt path/to/model_000200.pt
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path

import torch


LAMBDA_KEYS = ("resid_lambdas", "x0_lambdas", "ut_source_lambdas")
KAPPA_NAMES = ("kappa_scale", "kappa_bias")
KAPPA_LAYER_PATTERN = re.compile(
    r"transformer\.h\.(?P<layer>\d+)\.mlp(?:\.experts)?\.(?P<name>kappa_(?:bias|scale))"
)


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


def calculate_kappa_correlation(bias: torch.Tensor, scale: torch.Tensor) -> float:
    if bias.shape != scale.shape:
        raise ValueError(
            f"kappa_bias and kappa_scale shapes must match: "
            f"{tuple(bias.shape)} != {tuple(scale.shape)}"
        )
    calculate_statistics(bias)
    calculate_statistics(scale)
    paired_values = torch.stack([
        bias.detach().reshape(-1).to(dtype=torch.float64),
        scale.detach().reshape(-1).to(dtype=torch.float64),
    ])
    if paired_values.size(1) < 2 or (paired_values.std(dim=1, unbiased=False) == 0).any():
        return float("nan")
    return torch.corrcoef(paired_values)[0, 1].item()


def calculate_kappa_ratio_mean(bias: torch.Tensor, scale: torch.Tensor) -> float:
    if bias.shape != scale.shape:
        raise ValueError(
            f"kappa_bias and kappa_scale shapes must match: "
            f"{tuple(bias.shape)} != {tuple(scale.shape)}"
        )
    calculate_statistics(bias)
    calculate_statistics(scale)
    bias_values = bias.detach().reshape(-1).to(dtype=torch.float64)
    scale_values = scale.detach().reshape(-1).to(dtype=torch.float64)
    nonzero_scale = scale_values.ne(0)
    if not nonzero_scale.any():
        return float("nan")
    return (bias_values[nonzero_scale] / scale_values[nonzero_scale]).mean().item()


def calculate_kappa_shared_alpha(bias: torch.Tensor, scale: torch.Tensor) -> float:
    if bias.shape != scale.shape:
        raise ValueError(
            f"kappa_bias and kappa_scale shapes must match: "
            f"{tuple(bias.shape)} != {tuple(scale.shape)}"
        )
    calculate_statistics(bias)
    calculate_statistics(scale)
    bias_values = bias.detach().to(dtype=torch.float64)
    scale_values = scale.detach().to(dtype=torch.float64)
    scale_squared_sum = scale_values.square().sum()
    if scale_squared_sum == 0:
        return float("nan")
    return ((scale_values * bias_values).sum() / scale_squared_sum).item()


def calculate_kappa_residual_abs_mean(
    bias: torch.Tensor, scale: torch.Tensor, ratio_mean: float
) -> float:
    return calculate_kappa_residual_statistics(bias, scale, ratio_mean)["abs_mean"]


def calculate_kappa_residual_statistics(
    bias: torch.Tensor, scale: torch.Tensor, ratio_mean: float
) -> dict[str, float]:
    if bias.shape != scale.shape:
        raise ValueError(
            f"kappa_bias and kappa_scale shapes must match: "
            f"{tuple(bias.shape)} != {tuple(scale.shape)}"
        )
    calculate_statistics(bias)
    calculate_statistics(scale)
    bias_values = bias.detach().to(dtype=torch.float64)
    scale_values = scale.detach().to(dtype=torch.float64)
    residual = bias_values - ratio_mean * scale_values
    abs_mean = residual.abs().mean().item()
    bias_abs_mean = bias_values.abs().mean().item()
    return {
        "abs_mean": abs_mean,
        "std": residual.std(unbiased=False).item(),
        "relative_abs_mean": abs_mean / bias_abs_mean if bias_abs_mean != 0 else float("nan"),
    }


def calculate_kappa_dimension_stds(value: torch.Tensor) -> dict[str, list[float]]:
    if value.ndim != 3:
        return {}
    values = value.detach().to(dtype=torch.float64)
    return {
        "std_dim2_mean": values.std(dim=1, unbiased=False).mean(dim=1).tolist(),
        "std_dim3_mean": values.std(dim=2, unbiased=False).mean(dim=1).tolist(),
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


def load_kappa_tensors(checkpoint_path: Path):
    """Load individual kappa tensors without discarding layer identity."""
    state_dict = load_state_dict(checkpoint_path)
    resid_lambdas = state_dict.get("resid_lambdas")
    if not torch.is_tensor(resid_lambdas):
        raise KeyError(f"Checkpoint {checkpoint_path} is missing resid_lambdas")
    total_ut_steps = resid_lambdas.shape[0] if resid_lambdas.ndim == 2 else 1

    tensors = []
    for key, value in state_dict.items():
        match = KAPPA_LAYER_PATTERN.fullmatch(key)
        if match is not None:
            name = match.group("name")
            layer = int(match.group("layer"))
        elif key in KAPPA_NAMES or key in {f"global_{name}" for name in KAPPA_NAMES}:
            name = key.removeprefix("global_")
            layer = None
        else:
            continue
        if not torch.is_tensor(value):
            raise TypeError(
                f"Checkpoint {checkpoint_path} entry {key!r} is not a tensor; "
                f"got {type(value).__name__}"
            )
        if total_ut_steps > 1 and (
            value.ndim < 2 or value.shape[0] != total_ut_steps
        ):
            raise ValueError(
                f"Checkpoint {checkpoint_path} entry {key!r} has shape "
                f"{tuple(value.shape)}, whose leading dimension does not match "
                f"total_ut_steps={total_ut_steps}"
            )
        tensors.append((name, layer, key, value))
    tensors.sort(
        key=lambda item: (item[0], -1 if item[1] is None else item[1], item[2])
    )
    return total_ut_steps, tensors


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


def print_per_layer_kappa_statistics(checkpoint_path: Path) -> None:
    total_ut_steps, tensors = load_kappa_tensors(checkpoint_path)
    if not tensors:
        return
    print("Per-layer kappa statistics:")
    for name, layer, key, value in tensors:
        layer_label = "global" if layer is None else str(layer)
        pass_values = value if total_ut_steps > 1 or value.ndim == 3 else value.unsqueeze(0)
        dimension_stds = calculate_kappa_dimension_stds(value)
        for pass_idx, pass_value in enumerate(pass_values):
            stats = calculate_statistics(pass_value)
            dimension_label = "".join(
                f" {label}={stds[pass_idx]:.2f}"
                for label, stds in dimension_stds.items()
            )
            print(
                f"  {name} layer={layer_label} pass={pass_idx}: "
                f"mean={stats['mean']:.2f} std={stats['std']:.2f} "
                f"max={stats['max']:.2f} min={stats['min']:.2f} "
                f"abs_max={stats['abs_max']:.2f}{dimension_label} key={key}"
            )
    tensors_by_key = {key: value for _, _, key, value in tensors}
    paired_passes = {}
    for name, layer, key, bias in tensors:
        if name != "kappa_bias":
            continue
        scale = tensors_by_key.get(key.removesuffix("kappa_bias") + "kappa_scale")
        if scale is None:
            continue
        if bias.shape != scale.shape:
            raise ValueError(f"Mismatched kappa_bias/kappa_scale shapes for {key}")
        layer_label = "global" if layer is None else str(layer)
        bias_passes = bias if total_ut_steps > 1 or bias.ndim == 3 else bias.unsqueeze(0)
        scale_passes = scale if total_ut_steps > 1 or scale.ndim == 3 else scale.unsqueeze(0)
        shared_alpha = calculate_kappa_shared_alpha(bias, scale)
        for pass_idx, (pass_bias, pass_scale) in enumerate(zip(bias_passes, scale_passes)):
            correlation = calculate_kappa_correlation(pass_bias, pass_scale)
            ratio_mean = calculate_kappa_ratio_mean(pass_bias, pass_scale)
            residual_stats = calculate_kappa_residual_statistics(
                pass_bias, pass_scale, ratio_mean
            )
            shared_residual_stats = calculate_kappa_residual_statistics(
                pass_bias, pass_scale, shared_alpha
            )
            print(
                f"  kappa_bias/kappa_scale layer={layer_label} pass={pass_idx}: "
                f"pearson={correlation:.4f} ratio_mean={ratio_mean:.4f} "
                f"residual_abs_mean={residual_stats['abs_mean']:.4f} "
                f"residual_std={residual_stats['std']:.4f} "
                f"shared_alpha={shared_alpha:.4f} "
                f"shared_residual_abs_mean={shared_residual_stats['abs_mean']:.4f} "
                f"shared_residual_std={shared_residual_stats['std']:.4f} "
                f"shared_residual_rel_mae={shared_residual_stats['relative_abs_mean']:.4f}"
            )
            paired_passes.setdefault(pass_idx, []).append((pass_bias, pass_scale))
    for pass_idx, pairs in sorted(paired_passes.items()):
        bias = torch.cat([pair[0].reshape(-1) for pair in pairs])
        scale = torch.cat([pair[1].reshape(-1) for pair in pairs])
        correlation = calculate_kappa_correlation(bias, scale)
        ratio_mean = calculate_kappa_ratio_mean(bias, scale)
        residual_stats = calculate_kappa_residual_statistics(bias, scale, ratio_mean)
        print(
            f"  kappa_bias/kappa_scale overall pass={pass_idx}: "
            f"pearson={correlation:.4f} ratio_mean={ratio_mean:.4f} "
            f"residual_abs_mean={residual_stats['abs_mean']:.4f} "
            f"residual_std={residual_stats['std']:.4f}"
        )


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
        print_per_layer_kappa_statistics(checkpoint_path)


if __name__ == "__main__":
    main()