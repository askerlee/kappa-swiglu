"""Compute weight similarity between two model checkpoint state_dict files.

Example:
    python -m scripts.compute_weight_similarity path/to/model_004762.pt path/to/model_004886.pt
"""

from __future__ import annotations

import argparse
import math
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
    return dict(state.items())


def is_comparable_tensor(value) -> bool:
    return torch.is_tensor(value) and value.is_floating_point()


def cosine_similarity_from_stats(dot: float, norm_a_sq: float, norm_b_sq: float) -> float:
    if norm_a_sq == 0.0 and norm_b_sq == 0.0:
        return 1.0
    if norm_a_sq == 0.0 or norm_b_sq == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a_sq * norm_b_sq)


def summarize_shared_tensors(state_a, state_b):
    keys_a = set(state_a)
    keys_b = set(state_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shared_keys = sorted(keys_a & keys_b)

    shape_mismatches = []
    non_float_keys = []
    compared = []

    global_dot = 0.0
    global_norm_a_sq = 0.0
    global_norm_b_sq = 0.0
    total_numel = 0

    for key in shared_keys:
        value_a = state_a[key]
        value_b = state_b[key]

        if not torch.is_tensor(value_a) or not torch.is_tensor(value_b):
            non_float_keys.append((key, type(value_a).__name__, type(value_b).__name__))
            continue

        if tuple(value_a.shape) != tuple(value_b.shape):
            shape_mismatches.append((key, tuple(value_a.shape), tuple(value_b.shape)))
            continue

        if not is_comparable_tensor(value_a) or not is_comparable_tensor(value_b):
            non_float_keys.append((key, str(getattr(value_a, "dtype", type(value_a).__name__)), str(getattr(value_b, "dtype", type(value_b).__name__))))
            continue

        flat_a = value_a.detach().reshape(-1).to(dtype=torch.float32)
        flat_b = value_b.detach().reshape(-1).to(dtype=torch.float32)
        dot = float(torch.dot(flat_a, flat_b))
        norm_a_sq = float(torch.dot(flat_a, flat_a))
        norm_b_sq = float(torch.dot(flat_b, flat_b))
        l2_distance = float(torch.linalg.vector_norm(flat_a - flat_b))
        mean_abs_diff = float((flat_a - flat_b).abs().mean())
        max_abs_diff = float((flat_a - flat_b).abs().max())
        similarity = cosine_similarity_from_stats(dot, norm_a_sq, norm_b_sq)

        compared.append(
            {
                "key": key,
                "shape": tuple(value_a.shape),
                "numel": int(flat_a.numel()),
                "cosine_similarity": similarity,
                "l2_distance": l2_distance,
                "mean_abs_diff": mean_abs_diff,
                "max_abs_diff": max_abs_diff,
            }
        )

        global_dot += dot
        global_norm_a_sq += norm_a_sq
        global_norm_b_sq += norm_b_sq
        total_numel += int(flat_a.numel())

    if global_norm_a_sq == 0.0 and global_norm_b_sq == 0.0:
        global_cosine = 1.0
    elif global_norm_a_sq == 0.0 or global_norm_b_sq == 0.0:
        global_cosine = 0.0
    else:
        global_cosine = global_dot / math.sqrt(global_norm_a_sq * global_norm_b_sq)

    return {
        "only_a": only_a,
        "only_b": only_b,
        "shape_mismatches": shape_mismatches,
        "non_float_keys": non_float_keys,
        "compared": compared,
        "global_cosine": global_cosine,
        "total_numel": total_numel,
        "shared_keys": shared_keys,
    }


def print_report(path_a: Path, path_b: Path, summary, show_bottom: int, show_top: int):
    compared = summary["compared"]
    compared_sorted = sorted(compared, key=lambda item: item["cosine_similarity"])

    print(f"Checkpoint A: {path_a}")
    print(f"Checkpoint B: {path_b}")
    print(f"Keys in A: {len(summary['only_a']) + len(summary['shared_keys']):,}")
    print(f"Keys in B: {len(summary['only_b']) + len(summary['shared_keys']):,}")
    print(f"Shared keys: {len(summary['shared_keys']):,}")
    print(f"Compared floating tensors: {len(compared):,}")
    print(f"Only in A: {len(summary['only_a']):,}")
    print(f"Only in B: {len(summary['only_b']):,}")
    print(f"Shape mismatches: {len(summary['shape_mismatches']):,}")
    print(f"Skipped non-floating or non-tensor keys: {len(summary['non_float_keys']):,}")
    print(f"Aggregate cosine similarity: {summary['global_cosine']:.8f}")
    print(f"Total compared parameters: {summary['total_numel']:,}")

    if summary["only_a"]:
        print("\nKeys only in A:")
        for key in summary["only_a"]:
            print(f"  {key}")

    if summary["only_b"]:
        print("\nKeys only in B:")
        for key in summary["only_b"]:
            print(f"  {key}")

    if summary["shape_mismatches"]:
        print("\nShared keys with shape mismatches:")
        for key, shape_a, shape_b in summary["shape_mismatches"]:
            print(f"  {key}: A{shape_a} vs B{shape_b}")

    if summary["non_float_keys"]:
        print("\nSkipped shared keys that are not floating tensors in both checkpoints:")
        for key, type_a, type_b in summary["non_float_keys"]:
            print(f"  {key}: A={type_a} B={type_b}")

    if show_bottom > 0 and compared_sorted:
        print(f"\nLowest {min(show_bottom, len(compared_sorted))} tensor cosine similarities:")
        for item in compared_sorted[:show_bottom]:
            print(
                f"  {item['key']}: cos={item['cosine_similarity']:.8f} "
                f"mean_abs_diff={item['mean_abs_diff']:.8e} "
                f"max_abs_diff={item['max_abs_diff']:.8e} "
                f"l2={item['l2_distance']:.8e} shape={item['shape']}"
            )

    if show_top > 0 and compared_sorted:
        print(f"\nHighest {min(show_top, len(compared_sorted))} tensor cosine similarities:")
        for item in reversed(compared_sorted[-show_top:]):
            print(
                f"  {item['key']}: cos={item['cosine_similarity']:.8f} "
                f"mean_abs_diff={item['mean_abs_diff']:.8e} "
                f"max_abs_diff={item['max_abs_diff']:.8e} "
                f"l2={item['l2_distance']:.8e} shape={item['shape']}"
            )


def build_parser():
    parser = argparse.ArgumentParser(description="Compute weight similarity between two model checkpoint state_dict files")
    parser.add_argument("checkpoint_a", type=Path, help="path to the first model_*.pt checkpoint")
    parser.add_argument("checkpoint_b", type=Path, help="path to the second model_*.pt checkpoint")
    parser.add_argument(
        "--show-bottom",
        type=int,
        default=20,
        help="print this many tensors with the lowest cosine similarity",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=5,
        help="print this many tensors with the highest cosine similarity",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    state_a = normalize_state_dict(load_checkpoint_state(args.checkpoint_a), args.checkpoint_a)
    state_b = normalize_state_dict(load_checkpoint_state(args.checkpoint_b), args.checkpoint_b)
    summary = summarize_shared_tensors(state_a, state_b)
    print_report(args.checkpoint_a, args.checkpoint_b, summary, args.show_bottom, args.show_top)


if __name__ == "__main__":
    main()