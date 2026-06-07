"""Rewrite legacy checkpoint metadata/state-dict keys after repo-wide renames.

Examples:
    python -m scripts.migrate_checkpoint_renames output/base_checkpoints/d8 --dry-run
    python -m scripts.migrate_checkpoint_renames output/base_checkpoints/d8
    python -m scripts.migrate_checkpoint_renames output/base_checkpoints/d8/meta_000100.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch


# Exact renames that are safe for checkpoint metadata and payloads everywhere,
# including model_config and state-dict-adjacent metadata.
ALWAYS_EXACT_KEY_RENAMES = {
    "use_gate_proj_bias": "use_kappa_swiglu",
    "use_exp_kappa_bias": "use_kappa_swiglu",
    "kappa_bias_input": "kappa_input",
    "kappa_bias_input_constant": "kappa_input_constant",
    "log_implicit_kappa_bias": "log_implicit_gate_proj_bias",
    "compute_gate_proj_slope_magnitude_losses": "compute_kappa_slope_magnitude_losses",
    "gate_proj_slope_l2_loss": "kappa_bias_l2_loss",
    "implicit_kappa_bias_top5p_mean": "implicit_gate_proj_bias_top5p_mean",
    "implicit_kappa_bias_bottom5p_mean": "implicit_gate_proj_bias_bottom5p_mean",
}

# These renames reflect CLI/user-config surface changes. They are intentionally
# skipped under model_config because the current runtime still expects the older
# internal GPTConfig field names there.
NON_MODEL_CONFIG_EXACT_KEY_RENAMES = {
    "constant_kappa_bias_dense_layers": "constant_kappa_dense_layers",
    "global_kappa_bias_granularity": "global_kappa_granularity",
    "kappa_bias_start_layer": "kappa_start_layer",
    "kappa_bias_lr_max_scale": "kappa_lr_max_scale",
    "kappa_bias_lr_final_scale": "kappa_lr_final_scale",
    "kappa_bias_delay_start_min_iterations": "kappa_delay_start_min_iterations",
    "kappa_bias_delay_start_iteration_frac": "kappa_delay_start_iteration_frac",
    "kappa_bias_lr_warmup_iterations": "kappa_lr_warmup_iterations",
    "kappa_bias_l2_loss_weight": "kappa_l2_loss_weight",
    "kappa_bias_ema_rms_reg": "kappa_ema_rms_reg",
    "kappa_bias_l2_ema_beta": "kappa_l2_ema_beta",
    "kappa_bias_l2_ema_anchor_start": "kappa_l2_ema_anchor_start",
    "kappa_bias_l2_ema_anchor_end": "kappa_l2_ema_anchor_end",
    "kappa_bias_l2_ema_floor_frac": "kappa_l2_ema_floor_frac",
    "kappa_bias_l2_loss_anneal_iterations": "kappa_l2_loss_anneal_iterations",
    "kappa_bias_l2_loss_stage1_frac": "kappa_l2_loss_stage1_frac",
    "kappa_bias_l2_loss_final_frac": "kappa_l2_loss_final_frac",
}

# Ordered substring rewrites for legacy parameter names, metric keys, and other
# string payloads embedded inside torch checkpoint files.
ORDERED_TEXT_RENAMES = (
    ("compute_gate_proj_slope_magnitude_losses", "compute_kappa_slope_magnitude_losses"),
    ("implicit_kappa_bias_top5p_mean", "implicit_gate_proj_bias_top5p_mean"),
    ("implicit_kappa_bias_bottom5p_mean", "implicit_gate_proj_bias_bottom5p_mean"),
    ("gate_proj_slope_l2_loss", "kappa_bias_l2_loss"),
    ("gate_proj_bias_scale", "kappa_scale"),
    ("use_gate_proj_bias", "use_kappa_swiglu"),
    ("implicit_kappa_", "implicit_gate_proj_"),
    ("gate_proj_bias", "kappa_bias"),
    ("gate_slope", "kappa_slope"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy checkpoint/meta key names after code renames."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Checkpoint directory, checkpoint file, or meta json file to migrate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report pending rewrites without modifying files.",
    )
    return parser.parse_args()


def _path_contains_component(path: str, component: str) -> bool:
    if not path:
        return False
    for part in path.replace("[", ".[").split("."):
        if not part or part.startswith("["):
            continue
        if part == component:
            return True
    return False


def _rename_text(text: str, path: str) -> str:
    in_model_config = _path_contains_component(path, "model_config")

    renamed = ALWAYS_EXACT_KEY_RENAMES.get(text, text)
    if not in_model_config:
        renamed = NON_MODEL_CONFIG_EXACT_KEY_RENAMES.get(renamed, renamed)

    for old, new in ORDERED_TEXT_RENAMES:
        renamed = renamed.replace(old, new)

    renamed = ALWAYS_EXACT_KEY_RENAMES.get(renamed, renamed)
    if not in_model_config:
        renamed = NON_MODEL_CONFIG_EXACT_KEY_RENAMES.get(renamed, renamed)

    return renamed


def _rename_mapping_keys(mapping: dict[str, Any], changes: list[str], file_path: Path, path: str) -> dict[str, Any]:
    renamed: dict[str, Any] = {}
    for key, value in mapping.items():
        new_key = _rename_text(key, path)
        child_path = f"{path}.{new_key}" if path else new_key
        new_value = rename_legacy_keys(value, changes, file_path, child_path)
        if new_key != key:
            changes.append(f"{file_path}: {path or '<root>'}.{key} -> {new_key}".replace("<root>.", ""))
        if new_key in renamed and renamed[new_key] != new_value:
            raise ValueError(
                f"Conflicting values while renaming {file_path}: key {new_key!r} already exists at {path or '<root>'}"
            )
        renamed[new_key] = new_value
    return renamed


def rename_legacy_keys(value: Any, changes: list[str], file_path: Path, path: str = "") -> Any:
    if isinstance(value, dict):
        return _rename_mapping_keys(value, changes, file_path, path)
    if isinstance(value, list):
        renamed_items = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            renamed_items.append(rename_legacy_keys(item, changes, file_path, child_path))
        return renamed_items
    if isinstance(value, str):
        renamed = _rename_text(value, path)
        if renamed != value:
            changes.append(f"{file_path}: {path} value {value!r} -> {renamed!r}")
        return renamed
    return value


def _find_paired_model_file(meta_path: Path) -> Path | None:
    match = re.match(r"meta_(\d+)\.json$", meta_path.name)
    if match is None:
        return None
    candidate = meta_path.with_name(f"model_{match.group(1)}.pt")
    if candidate.is_file():
        return candidate
    return None


def _infer_model_config_updates_from_model_data(model_data: dict[str, Any]) -> dict[str, Any]:
    layer_pattern = re.compile(r"^transformer\.h\.(\d+)\.")

    kappa_layers: set[int] = set()
    dense_kappa_layers: set[int] = set()
    has_kappa_bias_ema_rms_reg = False

    for key in model_data:
        if not isinstance(key, str):
            continue
        layer_match = layer_pattern.match(key)
        layer_idx = int(layer_match.group(1)) if layer_match is not None else None

        if ".kappa_bias" in key or ".kappa_scale" in key:
            if layer_idx is not None:
                kappa_layers.add(layer_idx)
            if ".mlp.kappa_bias" in key or ".mlp.kappa_scale" in key:
                if ".mlp.experts." not in key and layer_idx is not None:
                    dense_kappa_layers.add(layer_idx)

        if "kappa_bias_ema_rms_reg_keeper." in key or "kappa_scale_ema_rms_reg_keeper." in key:
            has_kappa_bias_ema_rms_reg = True
            if layer_idx is not None:
                kappa_layers.add(layer_idx)
            if ".mlp.experts." not in key and layer_idx is not None:
                dense_kappa_layers.add(layer_idx)

    updates: dict[str, Any] = {}
    if kappa_layers:
        updates["use_kappa_swiglu"] = True
        updates["kappa_bias_start_layer"] = min(kappa_layers)
    if dense_kappa_layers:
        updates["constant_kappa_bias_dense_layers"] = True
    if has_kappa_bias_ema_rms_reg:
        updates["kappa_bias_ema_rms_reg"] = True
    return updates


def _reconcile_model_config_from_model_file(
    payload: Any,
    meta_path: Path,
    dry_run: bool,
    changes: list[str],
) -> Any:
    if not isinstance(payload, dict):
        return payload
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        return payload

    model_path = _find_paired_model_file(meta_path)
    if model_path is None:
        return payload

    model_data = torch.load(model_path, map_location="cpu")
    if not isinstance(model_data, dict):
        return payload

    inferred_updates = _infer_model_config_updates_from_model_data(model_data)
    if not inferred_updates:
        return payload

    reconciled_payload = payload
    for key, inferred_value in inferred_updates.items():
        existing_value = model_config.get(key)
        if existing_value == inferred_value:
            continue
        model_config[key] = inferred_value
        changes.append(
            f"{meta_path}: model_config.{key} {existing_value!r} -> {inferred_value!r} (inferred from {model_path.name})"
        )

    return reconciled_payload


def atomic_write_json(path: Path, payload: Any) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def atomic_write_torch(path: Path, payload: Any) -> None:
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def process_json_file(path: Path, dry_run: bool) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    changes: list[str] = []
    renamed_payload = rename_legacy_keys(payload, changes, path)
    renamed_payload = _reconcile_model_config_from_model_file(renamed_payload, path, dry_run, changes)
    if changes and not dry_run:
        atomic_write_json(path, renamed_payload)
    return changes


def process_torch_file(path: Path, dry_run: bool) -> list[str]:
    payload = torch.load(path, map_location="cpu")
    changes: list[str] = []
    renamed_payload = rename_legacy_keys(payload, changes, path)
    if changes and not dry_run:
        atomic_write_torch(path, renamed_payload)
    return changes


def iter_target_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files: list[Path] = []
    files.extend(sorted(path.glob("meta_*.json")))
    files.extend(sorted(path.glob("model_*.pt")))
    files.extend(sorted(path.glob("optimizer_*.pt")))
    return files


def main() -> None:
    args = parse_args()
    seen: set[Path] = set()
    targets: list[Path] = []
    for raw_path in args.paths:
        for target in iter_target_files(Path(raw_path)):
            resolved = target.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            targets.append(target)

    if not targets:
        raise FileNotFoundError("No checkpoint files matched the provided paths.")

    changed_files = 0
    total_changes = 0
    for target in targets:
        if target.suffix == ".json":
            changes = process_json_file(target, dry_run=args.dry_run)
        elif target.suffix == ".pt":
            changes = process_torch_file(target, dry_run=args.dry_run)
        else:
            continue

        if not changes:
            continue

        changed_files += 1
        total_changes += len(changes)
        print(f"[{target}] {len(changes)} change(s)")
        for change in changes:
            print(f"  {change}")

    mode = "Would rewrite" if args.dry_run else "Rewrote"
    print(f"{mode} {changed_files} file(s) with {total_changes} total change(s).")


if __name__ == "__main__":
    main()