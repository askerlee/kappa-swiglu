"""Scan a model checkpoint for unusual floating-point tensor values.

Example:
    python -m scripts.scan_checkpoint_weights path/to/model_004762.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


def load_checkpoint_state(path: Path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )


def normalize_state_dict(state, checkpoint_path: Path):
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint {checkpoint_path} did not load as a mapping; got {type(state).__name__}")

    if "model" in state and isinstance(state["model"], Mapping):
        return dict(state["model"].items())
    if "state_dict" in state and isinstance(state["state_dict"], Mapping):
        return dict(state["state_dict"].items())
    return dict(state.items())


def is_scannable_tensor(value) -> bool:
    return torch.is_tensor(value) and value.is_floating_point()


def summarize_tensor(key, value, abs_threshold: float, z_threshold: float):
    flat = value.detach().reshape(-1)
    total_count = int(flat.numel())
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    nonfinite_count = total_count - finite_count

    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    if nonfinite_count > 0:
        nan_count = int(torch.isnan(flat).sum().item())
        posinf_count = int(torch.isposinf(flat).sum().item())
        neginf_count = int(torch.isneginf(flat).sum().item())

    summary = {
        "key": key,
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "numel": total_count,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "abs_max": None,
        "abs_over_threshold": 0,
        "max_abs_z": None,
        "z_over_threshold": 0,
        "flagged_reasons": [],
    }

    if nonfinite_count > 0:
        summary["flagged_reasons"].append("nonfinite")

    if finite_count == 0:
        return summary

    finite_values = flat[finite_mask].to(dtype=torch.float32)
    mean = float(finite_values.mean().item())
    std = float(finite_values.std(unbiased=False).item()) if finite_count > 1 else 0.0
    min_value = float(finite_values.min().item())
    max_value = float(finite_values.max().item())
    abs_values = finite_values.abs()
    abs_max = float(abs_values.max().item())
    abs_over_threshold = int((abs_values > abs_threshold).sum().item())

    max_abs_z = None
    z_over_threshold = 0
    if std > 0.0:
        z_scores = (finite_values - mean).abs() / std
        max_abs_z = float(z_scores.max().item())
        z_over_threshold = int((z_scores > z_threshold).sum().item())

    summary.update(
        {
            "min": min_value,
            "max": max_value,
            "mean": mean,
            "std": std,
            "abs_max": abs_max,
            "abs_over_threshold": abs_over_threshold,
            "max_abs_z": max_abs_z,
            "z_over_threshold": z_over_threshold,
        }
    )

    if abs_over_threshold > 0:
        summary["flagged_reasons"].append("abs")
    if z_over_threshold > 0:
        summary["flagged_reasons"].append("zscore")

    return summary


def scan_checkpoint(state_dict, abs_threshold: float, z_threshold: float):
    scanned = []
    skipped = []

    total_nonfinite = 0
    global_abs_max = 0.0
    for key in sorted(state_dict):
        value = state_dict[key]
        if not is_scannable_tensor(value):
            skipped.append((key, type(value).__name__ if not torch.is_tensor(value) else str(value.dtype)))
            continue

        summary = summarize_tensor(key, value, abs_threshold=abs_threshold, z_threshold=z_threshold)
        scanned.append(summary)
        total_nonfinite += summary["nonfinite_count"]
        if summary["abs_max"] is not None:
            global_abs_max = max(global_abs_max, summary["abs_max"])

    flagged = [item for item in scanned if item["flagged_reasons"]]
    flagged.sort(
        key=lambda item: (
            item["nonfinite_count"],
            item["abs_over_threshold"],
            item["z_over_threshold"],
            item["abs_max"] if item["abs_max"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    return {
        "scanned": scanned,
        "flagged": flagged,
        "skipped": skipped,
        "total_nonfinite": total_nonfinite,
        "global_abs_max": global_abs_max,
    }


def format_float(value):
    if value is None:
        return "n/a"
    return f"{value:.6e}"


def print_report(checkpoint_path: Path, summary, abs_threshold: float, z_threshold: float, limit: int, show_all: bool):
    scanned = summary["scanned"]
    flagged = summary["flagged"]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Scanned floating tensors: {len(scanned):,}")
    print(f"Flagged tensors: {len(flagged):,}")
    print(f"Skipped non-floating or non-tensor entries: {len(summary['skipped']):,}")
    print(f"Total non-finite entries: {summary['total_nonfinite']:,}")
    print(f"Global absolute max: {summary['global_abs_max']:.6e}")
    print(f"Absolute threshold: {abs_threshold:.6e}")
    print(f"Z-score threshold: {z_threshold:.6e}")

    items_to_print = scanned if show_all else flagged[:limit]
    if not items_to_print:
        print("\nNo unusual floating-point values were found with the current thresholds.")
        return

    heading = "All scanned tensors:" if show_all else f"Top {min(limit, len(flagged))} flagged tensors:"
    print(f"\n{heading}")
    for item in items_to_print:
        reasons = ",".join(item["flagged_reasons"]) if item["flagged_reasons"] else "ok"
        print(
            f"- {item['key']}: shape={item['shape']} dtype={item['dtype']} reasons={reasons} "
            f"nonfinite={item['nonfinite_count']} abs_over={item['abs_over_threshold']} z_over={item['z_over_threshold']}"
        )
        print(
            f"    min={format_float(item['min'])} max={format_float(item['max'])} "
            f"mean={format_float(item['mean'])} std={format_float(item['std'])} "
            f"abs_max={format_float(item['abs_max'])} max_abs_z={format_float(item['max_abs_z'])}"
        )
        if item["nan_count"] or item["posinf_count"] or item["neginf_count"]:
            print(
                f"    nan={item['nan_count']} +inf={item['posinf_count']} -inf={item['neginf_count']}"
            )


def build_parser():
    parser = argparse.ArgumentParser(description="Scan a model checkpoint for unusual floating-point tensor values")
    parser.add_argument("checkpoint", type=Path, help="path to the model checkpoint to scan")
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=1e3,
        help="flag tensors that contain values whose absolute magnitude exceeds this threshold",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=8.0,
        help="flag tensors that contain values whose absolute z-score exceeds this threshold",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="print up to this many flagged tensors",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="print stats for every scanned floating tensor, not just flagged ones",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.abs_threshold <= 0:
        parser.error("--abs-threshold must be > 0")
    if args.z_threshold <= 0:
        parser.error("--z-threshold must be > 0")
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    state = normalize_state_dict(load_checkpoint_state(args.checkpoint), args.checkpoint)
    summary = scan_checkpoint(state, abs_threshold=args.abs_threshold, z_threshold=args.z_threshold)
    print_report(
        args.checkpoint,
        summary,
        abs_threshold=args.abs_threshold,
        z_threshold=args.z_threshold,
        limit=args.limit,
        show_all=args.show_all,
    )


if __name__ == "__main__":
    main()