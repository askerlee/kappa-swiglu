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
from nanochat.gpt import GPT
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


def find_optimizer_shard_ranks(checkpoint_dir, step):
    shard_pattern = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank*.pt")
    ranks = []
    for shard_path in glob.glob(shard_pattern):
        match = re.search(r"_rank(\d+)\.pt$", os.path.basename(shard_path))
        if match is not None:
            ranks.append(int(match.group(1)))
    return sorted(ranks)


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
    available_ranks = find_optimizer_shard_ranks(checkpoint_dir, step)
    detected_world_size = len(available_ranks)
    if saved_world_size is None:
        saved_world_size = detected_world_size
    if saved_world_size <= 0:
        raise FileNotFoundError(f"No optimizer checkpoint shards found for step {step} in {checkpoint_dir}")

    expected_ranks = list(range(saved_world_size))
    missing_ranks = [saved_rank for saved_rank in expected_ranks if saved_rank not in available_ranks]
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


def validate_checkpoint_file_sizes(checkpoint_dir, step, expected_optimizer_ranks=None):
    expected_roles = {"model"}
    if expected_optimizer_ranks is not None:
        expected_roles.update(f"optim_rank{rank}" for rank in expected_optimizer_ranks)

    current_files = _checkpoint_files_for_step(checkpoint_dir, step)
    missing_roles = sorted(expected_roles.difference(current_files))
    if missing_roles:
        raise ValueError(
            f"Checkpoint step {step:06d} is missing expected files for size validation: "
            f"{', '.join(missing_roles)}"
        )

    comparison_step = None
    comparison_files = None
    for older_step in _older_checkpoint_steps(checkpoint_dir, step):
        candidate_files = _checkpoint_files_for_step(checkpoint_dir, older_step)
        if expected_roles.issubset(candidate_files):
            comparison_step = older_step
            comparison_files = candidate_files
            break

    if comparison_files is None:
        logger.warning(
            "Skipping checkpoint file size validation for step %06d; no previous checkpoint with matching file layout was found.",
            step,
        )
        return None

    mismatches = []
    for role in sorted(expected_roles):
        current_path = current_files[role]
        comparison_path = comparison_files[role]
        current_size = os.path.getsize(current_path)
        comparison_size = os.path.getsize(comparison_path)
        allowed_delta = _checkpoint_file_size_tolerance(role, comparison_size)
        if abs(current_size - comparison_size) > allowed_delta:
            mismatches.append(
                f"{role}: current={current_size} bytes ({os.path.basename(current_path)}), "
                f"previous={comparison_size} bytes ({os.path.basename(comparison_path)}), "
                f"allowed_delta={allowed_delta} bytes"
            )

    if mismatches:
        logger.warning(
            f"Checkpoint file size validation failed for step {step:06d} against step {comparison_step:06d}: "
            + "; ".join(mismatches)
        )

    logger.info(
        "Validated checkpoint file sizes for step %06d against step %06d",
        step,
        comparison_step,
    )
    return comparison_step


def delete_old_checkpoints(checkpoint_dir, step):
    if not os.path.isdir(checkpoint_dir):
        return []

    deleted_paths = []
    deleted_steps = set()
    for entry in os.scandir(checkpoint_dir):
        if not entry.is_file():
            continue
        checkpoint_step = _checkpoint_step_from_filename(entry.name)
        if checkpoint_step is None or checkpoint_step >= step:
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
    model_config_kwargs = meta_data["model_config"]
    # Override model config with any kwargs provided whose values are not None
    model_config_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
    _patch_missing_config_keys(model_config_kwargs)
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
