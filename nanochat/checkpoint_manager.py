"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import copy
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, get_moe_layer_indices
from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")
    if "use_aux_free_load_balancing" not in model_config_kwargs:
        model_config_kwargs["use_aux_free_load_balancing"] = False
    if "aux_free_load_balancing_bias_update_speed" not in model_config_kwargs:
        model_config_kwargs["aux_free_load_balancing_bias_update_speed"] = 1e-3
    if "num_moe_layers" not in model_config_kwargs:
        model_config_kwargs["num_moe_layers"] = -1


def _infer_use_qwen3_dense_mlp(model_data, model_config_kwargs):
    """Infer whether dense layers use the new gated Qwen3 MLP or the legacy ReLU^2 MLP."""
    if "use_qwen3_dense_mlp" in model_config_kwargs:
        return

    temp_config = GPTConfig(**model_config_kwargs)
    moe_layer_indices = set(get_moe_layer_indices(temp_config))
    missing_dense_gate_proj = False
    for layer_idx in range(temp_config.n_layer):
        if layer_idx in moe_layer_indices:
            continue
        gate_proj_key = f"transformer.h.{layer_idx}.mlp.gate_proj.weight"
        if gate_proj_key not in model_data:
            missing_dense_gate_proj = True
            break

    model_config_kwargs["use_qwen3_dense_mlp"] = not missing_dense_gate_proj
    if missing_dense_gate_proj:
        log0("Patching missing use_qwen3_dense_mlp in model config to False")


def _infer_exp_gate_proj_bias(model_data, model_config_kwargs):
    """Infer expert gate-projection bias config for checkpoints with sparse metadata."""
    if "use_exp_gate_proj_bias" in model_config_kwargs:
        return

    gate_proj_bias_layers = []
    gate_proj_bias_pattern = re.compile(r"^transformer\.h\.(\d+)\.mlp\.experts\.gate_proj_bias$")
    for key in model_data:
        match = gate_proj_bias_pattern.match(key)
        if match is not None:
            gate_proj_bias_layers.append(int(match.group(1)))

    use_exp_gate_proj_bias = bool(gate_proj_bias_layers)
    model_config_kwargs["use_exp_gate_proj_bias"] = use_exp_gate_proj_bias
    if not use_exp_gate_proj_bias:
        return

    inferred_start_layer = min(gate_proj_bias_layers)
    model_config_kwargs.setdefault("gate_proj_bias_start_layer", inferred_start_layer)
    log0(
        "Patching missing expert gate_proj_bias config in model config to "
        f"enabled from layer {model_config_kwargs['gate_proj_bias_start_layer']}"
    )


def _override_disabled_exp_gate_proj_bias_with_zeros(model_data, model_kwargs):
    """Keep checkpoint gate_proj_bias params loadable when a caller overrides them off."""
    if model_kwargs.get("use_exp_gate_proj_bias") is not False:
        return model_kwargs

    gate_proj_bias_pattern = re.compile(r"^transformer\.h\.\d+\.mlp\.experts\.gate_proj_bias$")
    gate_proj_bias_keys = [key for key in model_data if gate_proj_bias_pattern.match(key)]
    if not gate_proj_bias_keys:
        return model_kwargs

    model_kwargs = dict(model_kwargs)
    model_kwargs.pop("use_exp_gate_proj_bias", None)
    for key in gate_proj_bias_keys:
        model_data[key] = torch.zeros_like(model_data[key])
    log0(
        "Preserving checkpoint expert gate_proj_bias parameters for loading and zeroing them "
        "because use_exp_gate_proj_bias was overridden to False"
    )
    return model_kwargs

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")
    if model_config.n_exp > 1:
        for layer_idx in get_moe_layer_indices(model_config):
            gate_proj_key = f"transformer.h.{layer_idx}.mlp.experts.gate_proj"
            gate_proj_a_key = f"transformer.h.{layer_idx}.mlp.experts.gate_proj_a"
            gate_proj_b_key = f"transformer.h.{layer_idx}.mlp.experts.gate_proj_b"
            gate_proj = model_data.get(gate_proj_key)
            if gate_proj is not None and gate_proj.ndim != 3:
                raise ValueError(
                    f"Expected {gate_proj_key} to be a 3D dense tensor, got shape {tuple(gate_proj.shape)}"
                )
            model_data.pop(gate_proj_a_key, None)
            model_data.pop(gate_proj_b_key, None)
            expert_bias_key = f"transformer.h.{layer_idx}.mlp.router.expert_bias"
            if expert_bias_key not in model_data:
                model_data[expert_bias_key] = torch.zeros(model_config.n_exp, dtype=torch.float32)
                log0(f"Patching missing {expert_bias_key} in model data to zeros")


def _optimizer_shard_path(checkpoint_dir, step, rank):
    return os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")


def _parse_checkpoint_filename(filename):
    match = re.match(r"model_(\d+)\.pt$", filename)
    if match is not None:
        return int(match.group(1)), "model"

    match = re.match(r"meta_(\d+)\.json$", filename)
    if match is not None:
        return int(match.group(1)), "meta"

    match = re.match(r"optim_(\d+)_rank(\d+)\.pt$", filename)
    if match is not None:
        return int(match.group(1)), f"optim_rank{int(match.group(2))}"

    return None, None


def _checkpoint_step_from_filename(filename):
    step, _ = _parse_checkpoint_filename(filename)
    return step


def _checkpoint_files_for_step(checkpoint_dir, step):
    checkpoint_files = {}
    if not os.path.isdir(checkpoint_dir):
        return checkpoint_files

    for entry in os.scandir(checkpoint_dir):
        if not entry.is_file():
            continue
        entry_step, role = _parse_checkpoint_filename(entry.name)
        if entry_step != step or role is None:
            continue
        checkpoint_files[role] = entry.path

    return checkpoint_files


def _older_checkpoint_steps(checkpoint_dir, step):
    if not os.path.isdir(checkpoint_dir):
        return []

    older_steps = set()
    for entry in os.scandir(checkpoint_dir):
        if not entry.is_file():
            continue
        entry_step = _checkpoint_step_from_filename(entry.name)
        if entry_step is not None and entry_step < step:
            older_steps.add(entry_step)

    return sorted(older_steps, reverse=True)


def _checkpoint_file_size_tolerance(role, reference_size):
    if role == "model":
        return 0
    return max(16, min(4096, reference_size // 100))


def _expected_checkpoint_roles(expected_optimizer_ranks=None):
    expected_roles = {"model"}
    if expected_optimizer_ranks is not None:
        expected_roles.update(f"optim_rank{rank}" for rank in expected_optimizer_ranks)
    return expected_roles


def _find_comparison_checkpoint_files(checkpoint_dir, step, expected_roles):
    for older_step in _older_checkpoint_steps(checkpoint_dir, step):
        candidate_files = _checkpoint_files_for_step(checkpoint_dir, older_step)
        if expected_roles.issubset(candidate_files):
            return older_step, candidate_files
    return None, None


def find_optimizer_shard_ranks(checkpoint_dir, step):
    shard_pattern = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank*.pt")
    ranks = []
    for shard_path in glob.glob(shard_pattern):
        match = re.search(r"_rank(\d+)\.pt$", os.path.basename(shard_path))
        if match is not None:
            ranks.append(int(match.group(1)))
    return sorted(ranks)


def inspect_optimizer_shards(checkpoint_dir, step, saved_world_size=None):
    available_ranks = find_optimizer_shard_ranks(checkpoint_dir, step)
    detected_world_size = len(available_ranks)
    if saved_world_size is None:
        saved_world_size = detected_world_size

    expected_ranks = list(range(saved_world_size)) if saved_world_size > 0 else []
    missing_ranks = [saved_rank for saved_rank in expected_ranks if saved_rank not in available_ranks]

    return {
        "available_ranks": available_ranks,
        "detected_world_size": detected_world_size,
        "saved_world_size": saved_world_size,
        "expected_ranks": expected_ranks,
        "missing_ranks": missing_ranks,
    }


def _clone_optimizer_state_value(value):
    if torch.is_tensor(value):
        return value.clone()
    return copy.deepcopy(value)


def _require_complete_shard_entries(shard_entries, description):
    present_entries = [entry for entry in shard_entries if entry is not None]
    if not present_entries:
        return None
    if len(present_entries) != len(shard_entries):
        missing_ranks = [idx for idx, entry in enumerate(shard_entries) if entry is None]
        raise ValueError(f"Incomplete optimizer state for {description}; missing shards {missing_ranks}")
    return present_entries


def _reshard_adamw_state(shard_entries, param, rank, current_world_size):
    if param.numel() < 1024:
        return {key: _clone_optimizer_state_value(value) for key, value in shard_entries[0].items()}

    if param.shape[0] % current_world_size != 0:
        raise ValueError(
            "AdamW optimizer state reshard requires shape[0] divisible by current world size. "
            f"Got shape[0]={param.shape[0]} and world size={current_world_size}."
        )

    rank_size = param.shape[0] // current_world_size
    start = rank * rank_size
    end = start + rank_size
    local_state = {}
    for key, value in shard_entries[0].items():
        if torch.is_tensor(value) and value.ndim > 0:
            full_value = torch.cat([entry[key] for entry in shard_entries], dim=0)
            if full_value.shape[0] != param.shape[0]:
                raise ValueError(
                    f"AdamW state shape mismatch for key '{key}': "
                    f"reconstructed dim0={full_value.shape[0]}, expected={param.shape[0]}"
                )
            local_state[key] = full_value[start:end].clone()
        else:
            local_state[key] = _clone_optimizer_state_value(value)
    return local_state


def _reshard_muon_state(shard_entries, num_params, rank, current_world_size):
    chunk_size = (num_params + current_world_size - 1) // current_world_size
    start = rank * chunk_size
    end = min(start + chunk_size, num_params)
    local_state = {}

    for key, value in shard_entries[0].items():
        if torch.is_tensor(value) and value.ndim > 0:
            full_value = torch.cat([entry[key] for entry in shard_entries], dim=0)
            if full_value.shape[0] < num_params:
                raise ValueError(
                    f"Muon state shape mismatch for key '{key}': "
                    f"reconstructed dim0={full_value.shape[0]}, expected at least {num_params}"
                )
            full_value = full_value[:num_params]
            local_value = value.new_zeros((chunk_size, *value.shape[1:]))
            if start < num_params:
                local_value[:end - start].copy_(full_value[start:end])
            local_state[key] = local_value
        else:
            local_state[key] = _clone_optimizer_state_value(value)
    return local_state


def reshard_optimizer_state_dict(shard_state_dicts, optimizer, rank=0, saved_world_size=1, current_world_size=1):
    if not shard_state_dicts:
        raise ValueError("No optimizer state shards provided for resharding")
    if current_world_size <= 0:
        raise ValueError(f"Current optimizer world size must be positive, got {current_world_size}")
    if not (0 <= rank < current_world_size):
        raise ValueError(f"Optimizer rank {rank} is out of bounds for world size {current_world_size}")

    saved_param_groups = shard_state_dicts[0]["param_groups"]
    current_param_groups = optimizer.param_groups
    if len(saved_param_groups) != len(current_param_groups):
        raise ValueError(
            "Optimizer param group count mismatch between checkpoint and current optimizer: "
            f"{len(saved_param_groups)} != {len(current_param_groups)}"
        )

    resharded_state = {}
    for group_idx, (saved_group, current_group) in enumerate(zip(saved_param_groups, current_param_groups)):
        saved_param_ids = saved_group.get("params", [])
        current_params = current_group.get("params", [])
        if len(saved_param_ids) != len(current_params):
            raise ValueError(
                f"Optimizer param count mismatch in group {group_idx}: "
                f"{len(saved_param_ids)} != {len(current_params)}"
            )

        saved_kind = saved_group.get("kind")
        current_kind = current_group.get("kind")
        if saved_kind != current_kind:
            raise ValueError(
                f"Optimizer group kind mismatch in group {group_idx}: "
                f"checkpoint={saved_kind}, current={current_kind}"
            )

        if saved_kind == "adamw":
            for param_id, param in zip(saved_param_ids, current_params):
                shard_entries = _require_complete_shard_entries(
                    [shard_state_dict["state"].get(param_id) for shard_state_dict in shard_state_dicts],
                    f"AdamW parameter {param_id}",
                )
                if shard_entries is None:
                    continue
                resharded_state[param_id] = _reshard_adamw_state(shard_entries, param, rank, current_world_size)
        elif saved_kind == "muon":
            if not saved_param_ids:
                continue
            state_param_id = saved_param_ids[0]
            shard_entries = _require_complete_shard_entries(
                [shard_state_dict["state"].get(state_param_id) for shard_state_dict in shard_state_dicts],
                f"Muon group {group_idx}",
            )
            if shard_entries is None:
                continue
            resharded_state[state_param_id] = _reshard_muon_state(
                shard_entries,
                len(current_params),
                rank,
                current_world_size,
            )
        else:
            raise ValueError(f"Unsupported optimizer kind '{saved_kind}' in checkpoint group {group_idx}")

    return {
        "state": resharded_state,
        "param_groups": copy.deepcopy(saved_param_groups),
    }


def load_optimizer_state_dict(checkpoint_dir, step, optimizer, device, rank=0, current_world_size=1, saved_world_size=None):
    shard_info = inspect_optimizer_shards(checkpoint_dir, step, saved_world_size=saved_world_size)
    available_ranks = shard_info["available_ranks"]
    saved_world_size = shard_info["saved_world_size"]
    if saved_world_size <= 0:
        raise FileNotFoundError(f"No optimizer checkpoint shards found for step {step} in {checkpoint_dir}")

    expected_ranks = shard_info["expected_ranks"]
    missing_ranks = shard_info["missing_ranks"]
    if missing_ranks:
        raise FileNotFoundError(
            f"Missing optimizer checkpoint shards for step {step}: expected ranks {expected_ranks}, "
            f"found {available_ranks}"
        )

    if current_world_size == saved_world_size:
        return torch.load(_optimizer_shard_path(checkpoint_dir, step, rank), map_location=device)

    shard_state_dicts = [
        torch.load(_optimizer_shard_path(checkpoint_dir, step, saved_rank), map_location=device)
        for saved_rank in expected_ranks
    ]
    return reshard_optimizer_state_dict(
        shard_state_dicts,
        optimizer,
        rank=rank,
        saved_world_size=saved_world_size,
        current_world_size=current_world_size,
    )

# the sharding being handled is optimizer-state sharding, not model-weight sharding. 
# Rank 0 saves one full model checkpoint, while every rank saves its own optimizer 
# shard as optim_<step>_rank<rank>.pt. 
# It's data-parallel training with a custom ZeRO-2-style optimizer/update sharding scheme, 
# not FSDP or tensor/pipeline/expert parallelism.
def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = _optimizer_shard_path(checkpoint_dir, step, rank)
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")


def snapshot_checkpoint_file_sizes(checkpoint_dir, step, expected_optimizer_ranks=None):
    expected_roles = _expected_checkpoint_roles(expected_optimizer_ranks)
    comparison_step, comparison_files = _find_comparison_checkpoint_files(
        checkpoint_dir,
        step,
        expected_roles,
    )

    if comparison_files is None:
        logger.warning(
            "Skipping checkpoint file size validation for step %06d; no previous checkpoint with matching file layout was found.",
            step,
        )
        return None, None

    return comparison_step, {
        role: os.path.getsize(comparison_files[role])
        for role in sorted(expected_roles)
    }


def validate_checkpoint_file_sizes(
    checkpoint_dir,
    step,
    expected_optimizer_ranks=None,
    comparison_step=None,
    reference_file_sizes=None,
):
    expected_roles = _expected_checkpoint_roles(expected_optimizer_ranks)

    current_files = _checkpoint_files_for_step(checkpoint_dir, step)
    missing_roles = sorted(expected_roles.difference(current_files))
    if missing_roles:
        raise ValueError(
            f"Checkpoint step {step:06d} is missing expected files for size validation: "
            f"{', '.join(missing_roles)}"
        )

    if reference_file_sizes is None:
        comparison_step, reference_file_sizes = snapshot_checkpoint_file_sizes(
            checkpoint_dir,
            step,
            expected_optimizer_ranks=expected_optimizer_ranks,
        )
        if reference_file_sizes is None:
            return None

    missing_reference_roles = sorted(expected_roles.difference(reference_file_sizes))
    if missing_reference_roles:
        raise ValueError(
            "Checkpoint file size reference is missing expected roles: "
            f"{', '.join(missing_reference_roles)}"
        )

    comparison_desc = (
        f"step {comparison_step:06d}"
        if comparison_step is not None
        else "the provided reference sizes"
    )

    mismatches = []
    for role in sorted(expected_roles):
        current_path = current_files[role]
        current_size = os.path.getsize(current_path)
        comparison_size = reference_file_sizes[role]
        allowed_delta = _checkpoint_file_size_tolerance(role, comparison_size)
        if abs(current_size - comparison_size) > allowed_delta:
            mismatches.append(
                f"{role}: current={current_size} bytes ({os.path.basename(current_path)}), "
                f"previous={comparison_size} bytes, "
                f"allowed_delta={allowed_delta} bytes"
            )

    if mismatches:
        message = (
            f"Checkpoint file size validation failed for step {step:06d} against {comparison_desc}: "
            + "; ".join(mismatches)
        )
        logger.warning(message)
        raise ValueError(message)

    logger.info(
        "Validated checkpoint file sizes for step %06d against %s",
        step,
        comparison_desc,
    )
    return comparison_step


def delete_checkpoint_step(checkpoint_dir, step):
    if not os.path.isdir(checkpoint_dir):
        return []

    deleted_paths = []
    for entry in os.scandir(checkpoint_dir):
        if not entry.is_file():
            continue
        checkpoint_step = _checkpoint_step_from_filename(entry.name)
        if checkpoint_step != step:
            continue
        try:
            os.remove(entry.path)
        except FileNotFoundError:
            continue
        deleted_paths.append(entry.path)

    if deleted_paths:
        logger.warning(
            "Deleted %d checkpoint file(s) for failed step %06d",
            len(deleted_paths),
            step,
        )

    return sorted(deleted_paths)


def delete_old_checkpoints(checkpoint_dir, step, keep_steps=None):
    if not os.path.isdir(checkpoint_dir):
        return []

    keep_steps_set = set()
    if keep_steps is not None:
        for keep_step in keep_steps:
            if keep_step is None:
                continue
            keep_steps_set.add(int(keep_step))

    deleted_paths = []
    deleted_steps = set()
    for entry in os.scandir(checkpoint_dir):
        if not entry.is_file():
            continue
        checkpoint_step = _checkpoint_step_from_filename(entry.name)
        if checkpoint_step is None or checkpoint_step >= step or checkpoint_step in keep_steps_set:
            continue
        try:
            os.remove(entry.path)
        except FileNotFoundError:
            continue
        deleted_paths.append(entry.path)
        deleted_steps.add(checkpoint_step)

    if deleted_paths:
        logger.info(
            "Deleted %d checkpoint file(s) older than step %06d (steps: %s)",
            len(deleted_paths),
            step,
            ", ".join(f"{deleted_step:06d}" for deleted_step in sorted(deleted_steps)),
        )

    return sorted(deleted_paths)

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = _optimizer_shard_path(checkpoint_dir, step, rank)
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase, **kwargs):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    kwargs = _override_disabled_exp_gate_proj_bias_with_zeros(model_data, kwargs)
    model_config_kwargs = meta_data["model_config"]
    # Override model config with any kwargs provided whose values are not None
    model_config_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
    _patch_missing_config_keys(model_config_kwargs)
    _infer_use_qwen3_dense_mlp(model_data, model_config_kwargs)
    _infer_exp_gate_proj_bias(model_data, model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None, **kwargs):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase, **kwargs)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
