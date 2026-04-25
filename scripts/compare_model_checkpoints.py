"""Compare two saved model checkpoint state_dict files.

Example:
    python -m scripts.compare_model_checkpoints path/to/model_004762.pt path/to/model_004886.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch

try:
    from torch._subclasses.fake_tensor import FakeTensorMode
except ImportError:  # pragma: no cover - older torch fallback
    FakeTensorMode = None


def load_checkpoint_state(path: Path, use_fake_tensors: bool = True):
    load_kwargs = {
        "map_location": "cpu",
        "weights_only": True,
        "mmap": True,
    }
    if use_fake_tensors and FakeTensorMode is not None:
        with FakeTensorMode():
            return torch.load(path, **load_kwargs)
    return torch.load(path, **load_kwargs)


def normalize_state_dict(state, checkpoint_path: Path):
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint {checkpoint_path} did not load as a mapping; got {type(state).__name__}")
    return dict(state.items())


def tensor_metadata(value):
    if not torch.is_tensor(value):
        return {
            "kind": type(value).__name__,
            "repr": repr(value),
        }

    shape = tuple(value.shape)
    numel = value.numel()
    element_size = value.element_size()
    return {
        "kind": "tensor",
        "shape": shape,
        "dtype": str(value.dtype),
        "stride": tuple(value.stride()),
        "requires_grad": bool(value.requires_grad),
        "is_contiguous": bool(value.is_contiguous()),
        "numel": int(numel),
        "element_size": int(element_size),
        "tensor_bytes": int(numel * element_size),
    }


def format_metadata(meta):
    if meta.get("kind") != "tensor":
        return f"kind={meta['kind']} value={meta['repr']}"
    return (
        f"shape={meta['shape']} dtype={meta['dtype']} stride={meta['stride']} "
        f"contiguous={meta['is_contiguous']} requires_grad={meta['requires_grad']} "
        f"numel={meta['numel']} tensor_bytes={meta['tensor_bytes']}"
    )


def compare_checkpoints(path_a: Path, path_b: Path, show_equal_limit: int = 0):
    state_a = normalize_state_dict(load_checkpoint_state(path_a), path_a)
    state_b = normalize_state_dict(load_checkpoint_state(path_b), path_b)

    keys_a = set(state_a)
    keys_b = set(state_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shared_keys = sorted(keys_a & keys_b)

    metadata_mismatches = []
    equal_metadata_keys = []
    total_tensor_bytes_a = 0
    total_tensor_bytes_b = 0

    for key in shared_keys:
        meta_a = tensor_metadata(state_a[key])
        meta_b = tensor_metadata(state_b[key])
        total_tensor_bytes_a += int(meta_a.get("tensor_bytes", 0))
        total_tensor_bytes_b += int(meta_b.get("tensor_bytes", 0))
        if meta_a == meta_b:
            equal_metadata_keys.append(key)
            continue
        metadata_mismatches.append((key, meta_a, meta_b))

    print(f"Checkpoint A: {path_a}")
    print(f"Checkpoint B: {path_b}")
    print(f"Keys in A: {len(keys_a):,}")
    print(f"Keys in B: {len(keys_b):,}")
    print(f"Shared keys: {len(shared_keys):,}")
    print(f"Only in A: {len(only_a):,}")
    print(f"Only in B: {len(only_b):,}")
    print(f"Shared keys with metadata mismatches: {len(metadata_mismatches):,}")
    print(f"Summed tensor bytes in shared keys, A: {total_tensor_bytes_a:,}")
    print(f"Summed tensor bytes in shared keys, B: {total_tensor_bytes_b:,}")

    if only_a:
        print("\nKeys only in A:")
        for key in only_a:
            print(f"  {key}")

    if only_b:
        print("\nKeys only in B:")
        for key in only_b:
            print(f"  {key}")

    if metadata_mismatches:
        print("\nShared keys with different metadata:")
        for key, meta_a, meta_b in metadata_mismatches:
            print(f"- {key}")
            print(f"    A: {format_metadata(meta_a)}")
            print(f"    B: {format_metadata(meta_b)}")
    else:
        print("\nAll shared keys have matching metadata.")

    if show_equal_limit > 0 and equal_metadata_keys:
        print(f"\nFirst {min(show_equal_limit, len(equal_metadata_keys))} shared keys with matching metadata:")
        for key in equal_metadata_keys[:show_equal_limit]:
            print(f"  {key}")


def build_parser():
    parser = argparse.ArgumentParser(description="Compare two saved model checkpoint state_dict files")
    parser.add_argument("checkpoint_a", type=Path, help="path to the first model_*.pt checkpoint")
    parser.add_argument("checkpoint_b", type=Path, help="path to the second model_*.pt checkpoint")
    parser.add_argument(
        "--show-equal-limit",
        type=int,
        default=0,
        help="print up to this many keys whose metadata matches exactly",
    )
    parser.add_argument(
        "--no-fake-tensors",
        action="store_true",
        help="load real tensors instead of using FakeTensorMode",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    global FakeTensorMode
    if args.no_fake_tensors:
        FakeTensorMode = None
    compare_checkpoints(args.checkpoint_a, args.checkpoint_b, show_equal_limit=args.show_equal_limit)


if __name__ == "__main__":
    main()