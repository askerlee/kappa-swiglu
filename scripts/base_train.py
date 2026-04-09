"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
import shlex
import subprocess
import sys
from contextlib import nullcontext, contextmanager
import re

import wandb
import torch

from nanochat.gpt import GPT
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import delete_old_checkpoints, save_checkpoint, load_checkpoint, inspect_optimizer_shards, load_optimizer_state_dict, snapshot_checkpoint_file_sizes, validate_checkpoint_file_sizes
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FLASH_ATTN, FLASH_ATTN_BACKEND
from scripts.base_eval import evaluate_core
from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.manager import MANAGER

# print_banner()
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_was_explicitly_set(argv, option_name):
    return any(token == option_name or token.startswith(f"{option_name}=") for token in argv)

def parse_milestones_arg(milestones_arg):
    if not milestones_arg:
        return []
    milestones = []
    for raw in milestones_arg.split(','):
        token = raw.strip()
        if not token:
            continue
        try:
            milestone = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid milestone '{token}'. Milestones must be integers."
            ) from exc
        if milestone < 0:
            raise argparse.ArgumentTypeError(
                f"Invalid milestone '{token}'. Milestones must be >= 0."
            )
        milestones.append(milestone)
    return sorted(set(milestones))


def strip_and_override_runtime_args(argv, remaining_milestones, resume_from_step):
    cleaned_argv = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--milestones":
            skip_next = True
            continue
        if token == "--resume-from-step":
            skip_next = True
            continue
        if token.startswith("--milestones="):
            continue
        if token.startswith("--resume-from-step="):
            continue
        cleaned_argv.append(token)

    if remaining_milestones:
        cleaned_argv.extend(["--milestones", ",".join(str(m) for m in remaining_milestones)])
    cleaned_argv.extend(["--resume-from-step", str(resume_from_step)])

    return cleaned_argv


def build_self_command_with_milestones(remaining_milestones, resume_from_step):
    # Slurm path: if extra submit-time args were captured in base_train.sbatch,
    # relaunch via sbatch so the new training run gets a fresh allocation.
    base_train_extra_args = os.environ.get("BASE_TRAIN_EXTRA_ARGS")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if slurm_job_id and base_train_extra_args is not None:
        try:
            slurm_extra_argv = shlex.split(base_train_extra_args)
        except ValueError:
            slurm_extra_argv = []
        slurm_extra_argv = strip_and_override_runtime_args(
            slurm_extra_argv,
            remaining_milestones,
            resume_from_step,
        )
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sbatch_script = os.path.join(repo_root, "base_train.sbatch")
        cmd_parts = ["sbatch", sbatch_script, *slurm_extra_argv]
        return " ".join(shlex.quote(part) for part in cmd_parts)
    else:
        return None


def infer_last_completed_core_eval_step(checkpoint_dir, current_step, core_metric_every):
    if core_metric_every <= 0 or not os.path.isdir(checkpoint_dir):
        return None

    last_core_eval_step = None
    for entry in os.scandir(checkpoint_dir):
        if not entry.is_file():
            continue
        match = re.match(r"meta_(\d+)\.json$", entry.name)
        if match is None:
            continue
        candidate_step = int(match.group(1))
        if candidate_step >= current_step:
            continue
        if candidate_step == 0 or candidate_step % core_metric_every != 0:
            continue
        model_path = os.path.join(checkpoint_dir, f"model_{candidate_step:06d}.pt")
        if not os.path.isfile(model_path):
            continue
        if last_core_eval_step is None or candidate_step > last_core_eval_step:
            last_core_eval_step = candidate_step

    return last_core_eval_step
    
# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--seed", type=int, default=26, help="random seed for initialization")
parser.add_argument("--mockup-mode", type=str2bool, nargs='?', const=True, default=False, help="skip actual training/eval/sample compute and only advance step counter")
# FP8 training
parser.add_argument("--fp8", type=str2bool, nargs='?', const=True, default=False, help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=8, help="depth of the Transformer model")
parser.add_argument("--moe-start-layer", type=int, default=2, help="first layer index of MoE layers")
parser.add_argument("--n-exp", type=int, default=64, help="number of experts per MoE layer")
parser.add_argument("--moe-top-k", type=int, default=2, help="top-k of the MoE routing")
parser.add_argument("--use-aux-free-load-balancing", type=str2bool, nargs='?', const=True, default=False, help="enable DeepSeekV3 auxiliary-loss-free load balancing instead of the Switch auxiliary router loss")
parser.add_argument("--aux-loss-weight", type=float, default=0.001, help="weight for the Switch-style router auxiliary load-balancing loss")
parser.add_argument("--use-full-router-probs-for-aux-loss", type=str2bool, nargs='?', const=True, default=True, help="compute router auxiliary load-balancing loss from a full softmax over all experts instead of sparse top-k probabilities")
parser.add_argument("--router-ortho-loss-weight", type=float, default=0.0001, help="weight for router orthogonality loss")
parser.add_argument("--router-ortho-loss-anneal-iterations", type=int, default=-1, help="Total anneal iterations for the router ortho loss")
parser.add_argument("--router-ortho-loss-floor-frac", type=float, default=0, help="fraction of the base router ortho loss weight to keep after annealing completes")
parser.add_argument("--use-router-ortho-blockwise", type=str2bool, nargs='?', const=True, default=False, help="enable blockwise on/off schedule for router-ortho loss")
parser.add_argument("--router-ortho-block-size", type=int, default=100, help="block size (in optimizer steps) for blockwise router-ortho loss gating")
parser.add_argument("--router-ortho-on-prob", type=float, default=0.8, help="probability a router-ortho block is active; set to 1.0 to disable blockwise gating")
parser.add_argument("--router-ortho-blockwise-scale-preserve", type=str2bool, nargs='?', const=True, default=True, help="when a router-ortho block is active, scale by 1/on_prob to preserve expected loss weight")
parser.add_argument("--router-ortho-neg-corr-weight", type=float, default=1, help="weight for negative correlations in router-ortho loss.")
parser.add_argument("--experts-gate-output-loss-weight", type=float, default=0.00001, help="weight for expert gate z loss")
# use_experts_ortho_loss is False by default. So this weight has no effect.
parser.add_argument("--experts-ortho-loss-weight", type=float, default=0.01, help="weight for experts orthogonality loss")
parser.add_argument("--router-z-loss-weight", type=float, default=0.00001, help="weight for router z loss")
parser.add_argument("--router-z-loss-input-grad-scale", type=float, default=0.1, help="scaling factor for gradients to router input when computing router z loss. Setting this to a value < 1.0 can help stabilize training by preventing large z-loss gradients from destabilizing the router input representations.")
# How to set --router-wg-grad-scale? Maybe it should be set proportional to --moe-top-k,
# since --moe-top-k determines how dilluted the router wg gradients are across experts?
# If --moe-top-k == 4, then each row of router wg weight receives gradients scaled down by 1/4
# (the softmax weights) on average, so we suggest scaling wg grad by 
# 4 (actual moe_top_k) / 2 (default moe_top_k) * 2.0 (default router_wg_grad_scale) = 4.
parser.add_argument("--router-wg-grad-scale", type=float, default=1.0, help="scaling factor for gradients to router w_g weights only. This does not affect gradients flowing back into router inputs.")
parser.add_argument("--router-wg-grad-scale-anneal-iterations", type=int, default=-1, 
                    help="anneal router w_g grad scale over this many iterations (-1 disables)")
parser.add_argument("--router-wg-grad-scale-anneal-target", type=float, default=1.0,
                    help="final router w_g grad scale reached after annealing completes")
parser.add_argument("--use-router-wg-dyn-grad-scale", type=str2bool, nargs='?', const=True, default=False, 
                    help="whether to use dynamic gradient scaling for router w_g weights")
parser.add_argument("--use-experts-dyn-grad-scale", type=str2bool, nargs='?', const=True, default=False,
                    help="whether to apply the derived router grad scaling to expert weights")
parser.add_argument("--use-cumulative-dyn-grad-scale", type=str2bool, nargs='?', const=True, default=False,
                    help="whether to use moving-average smoothing for dynamic router/expert grad scales")
parser.add_argument("--dyn-grad-scale-ma-window-size", type=int, default=128,
                    help="number of recent steps used by moving-average smoothing for dynamic router/expert grad scales")
parser.add_argument("--z-loss-demean-logits", type=str2bool, nargs='?', const=True, default=True, help="use logits-demeaned router z loss")
parser.add_argument("--z-loss-penalize-mean-logits", type=str2bool, nargs='?', const=True, default=True, help="penalize mean logits in router z loss")
parser.add_argument("--aspect-ratio", type=int, default=96, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="LLLL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--compile", type=str2bool, nargs='?', const=True, default=True, help="use torch.compile to speed up training (may cause instability, use with caution)")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.05, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.01, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--muon-match-rms-adamw", type=str2bool, nargs='?', const=True, default=True, help="use Kimi Muon LR scaling: 0.2*sqrt(max(out,in))")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--lr-scheduler-skip-iters", type=int, default=0, help="number of initial iterations to skip for LR scheduling (to allow for redoing warmup when resuming from a later point in training)")
parser.add_argument("--lr-base-scale", type=float, default=1.0, help="base scale for learning rate")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--delete-old-ckpts", type=str2bool, nargs='?', const=True, default=True, help="after saving a checkpoint, delete all older checkpoints based on step number")
parser.add_argument("--delete-old-ckpts-before-save", action="store_true", help="delete old checkpoints before saving the new checkpoint; keeps file-size validation by snapshotting the previous checkpoint sizes first")
parser.add_argument("--milestones", type=str, default="", help="comma-separated iteration milestones; when a checkpoint save crosses a milestone, spawn this script again with that milestone removed")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
parser.add_argument("--wandb-api-key-file", type=str, default=None, help="Weights & Biases API key file (optional). If provided, sets WANDB_API_KEY for this run")
parser.add_argument("--log-grad-stats", action="store_true", help="log gradient statistics for MoE layers")
parser.add_argument("--log-interval", type=int, default=20, help="interval (in steps) for logging grad stats")

args = parser.parse_args()
if args.router_ortho_block_size <= 0:
    raise ValueError("--router-ortho-block-size must be > 0")
if not (0.0 < args.router_ortho_on_prob <= 1.0):
    raise ValueError("--router-ortho-on-prob must be in (0, 1]")
if args.use_aux_free_load_balancing:
    print("Disabling auxiliary router loss because --use-aux-free-load-balancing is enabled.")
if args.moe_top_k == 1 and not args.use_aux_free_load_balancing and not args.use_full_router_probs_for_aux_loss:
    print("Forcing --use-full-router-probs-for-aux-loss=True because --moe-top-k=1.")
    args.use_full_router_probs_for_aux_loss = True
user_config = vars(args).copy()  # for logging
milestones = parse_milestones_arg(args.milestones)
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# ddp is just a boolean meaning “this run was launched in distributed mode,” 
# not “the model is wrapped in PyTorch DistributedDataParallel.”
# The model is only assigned to orig_model and optionally passed to torch.compile; 
# it is never wrapped in DistributedDataParallel(...).
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type, seed=args.seed)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.mockup_mode or args.model_tag is None or not master_process
ckpt_prefix2 = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
if args.resume_from_step != -1:
    mat = re.search(r"(\d+)$", str(args.resume_from_step).rstrip('/'))
    if mat:
        ckpt_prefix2 += f"-resume{mat.group(1)}"

wandb_run_name = ckpt_prefix2 + '-' + time.strftime('%Y-%m-%d %H:%M:%S')

if args.wandb_api_key_file:
    with open(args.wandb_api_key_file, "r") as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()

wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nano-moe", name=wandb_run_name, config=user_config)
# logging
if not use_dummy_wandb:
    wandb.define_metric("tokens_seen")
    wandb.define_metric("train/*", step_metric="tokens_seen")
    wandb.define_metric("val/*", step_metric="tokens_seen")

# Flash Attention status
if HAS_FLASH_ATTN:
    backend_label = {
        "fa3": "Flash Attention 3",
        "fa4": "Flash Attention 4",
    }.get(FLASH_ATTN_BACKEND, "Flash Attention")
    print0(f"✓ Using {backend_label} backend.")
else:
    print0("!" * 80)
    print0("WARNING: No Flash Attention backend available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without Flash Attention")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio    # 8 * 128 = 1024 for depth=8
    # (1024 + 128 - 1) // 128 = 8; 8 * 128 = 1024
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    # 1024 // 128 = 8 heads
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, moe_start_layer=args.moe_start_layer,
        n_exp=args.n_exp, moe_top_k=args.moe_top_k,
        use_aux_loss=not args.use_aux_free_load_balancing,
        use_aux_free_load_balancing=args.use_aux_free_load_balancing,
        aux_loss_weight=args.aux_loss_weight,
        use_full_router_probs_for_aux_loss=args.use_full_router_probs_for_aux_loss,
        router_ortho_loss_weight=args.router_ortho_loss_weight,
        router_ortho_neg_corr_weight=args.router_ortho_neg_corr_weight,
        # this is the alpha in the paper that scales down gradients to expert gate projection weights during router orthogonality loss computation.
        experts_gate_output_loss_weight=args.experts_gate_output_loss_weight,
        experts_ortho_loss_weight=args.experts_ortho_loss_weight,
        router_z_loss_weight=args.router_z_loss_weight,
        router_z_loss_input_grad_scale=args.router_z_loss_input_grad_scale,
        router_wg_grad_scale=args.router_wg_grad_scale,
        use_router_wg_dyn_grad_scale=args.use_router_wg_dyn_grad_scale,
        use_experts_dyn_grad_scale=args.use_experts_dyn_grad_scale,
        use_cumulative_dyn_grad_scale=args.use_cumulative_dyn_grad_scale,
        dyn_grad_scale_ma_window_size=args.dyn_grad_scale_ma_window_size,
        z_loss_demean_logits=args.z_loss_demean_logits,
        z_loss_penalize_mean_logits=args.z_loss_penalize_mean_logits,
        n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta


def set_router_wg_grad_scale(model, router_wg_grad_scale):
    router_wg_grad_scale = float(router_wg_grad_scale)
    model.config.router_wg_grad_scale = router_wg_grad_scale
    for layer in model.transformer.h:
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "router", None)
        if router is not None:
            if hasattr(router, "set_router_wg_grad_scale"):
                router.set_router_wg_grad_scale(router_wg_grad_scale)
            else:
                router.router_wg_grad_scale = router_wg_grad_scale

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = vars(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized
set_router_wg_grad_scale(model, args.router_wg_grad_scale)

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
load_optimizer_state = False
saved_optimizer_world_size = 0
if resuming:
    print0(f"Resuming optimization from {checkpoint_dir} step {args.resume_from_step}")
    skip_optimizer_reason = None
    model_data, _, meta_data = load_checkpoint(
        checkpoint_dir,
        args.resume_from_step,
        device,
        load_optimizer=False,
    )
    optimizer_shard_info = inspect_optimizer_shards(
        checkpoint_dir,
        args.resume_from_step,
        saved_world_size=meta_data.get("optimizer_world_size"),
    )
    saved_optimizer_world_size = optimizer_shard_info["saved_world_size"]
    load_optimizer_state = saved_optimizer_world_size > 0 and not optimizer_shard_info["missing_ranks"]
    if saved_optimizer_world_size <= 0:
        skip_optimizer_reason = "No optimizer checkpoint shard found; resuming with fresh optimizer state."
    elif not load_optimizer_state:
        skip_optimizer_reason = (
            "Optimizer checkpoint shards are incomplete for the resume step; "
            f"expected ranks {optimizer_shard_info['expected_ranks']}, found {optimizer_shard_info['available_ranks']}. "
            "Resuming with fresh optimizer state."
        )
    elif saved_optimizer_world_size != ddp_world_size:
        print0(
            "Resharding optimizer state from checkpoint world size "
            f"{saved_optimizer_world_size} to current world size {ddp_world_size}."
        )
    if skip_optimizer_reason is not None:
        print0(skip_optimizer_reason)
    model.load_state_dict(model_data, strict=True, assign=True)
    set_router_wg_grad_scale(model, args.router_wg_grad_scale)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: only convert layers with dimensions divisible by 16 (FP8 hardware requirement)
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            # FP8 requires both in_features and out_features divisible by 16
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers, skipped {num_skipped} (dims not divisible by 16)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if args.compile:
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Determine the optimization horizon based on the model size
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params

# Get the parameter counts of the model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
active_param_count = model.get_num_active_params(args.n_exp, args.moe_top_k)
print0(f"Active parameters: {active_param_count:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Scaling params: transformer matrices + lm_head (gives cleanest scaling laws, see dev/LOG.md Jan 27, 2026)
get_scaling_params = lambda m: m.num_scaling_params()['transformer_matrices'] + m.num_scaling_params()['lm_head']
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

# Auto-compute optimal batch size based on Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
total_batch_size = args.total_batch_size
if total_batch_size == -1:
    d12_ref = build_model_meta(12) # d12 is where the optimal batch size was measured to be 2**19 tokens
    d12_num_scaling_params = get_scaling_params(d12_ref)
    D_REF = args.target_param_data_ratio * d12_num_scaling_params
    B_REF = 2**19
    batch_size_ratio = target_tokens / D_REF
    total_batch_size = 2 ** round(math.log2(B_REF * batch_size_ratio ** 0.383)) # also clamp to power of 2
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks

if total_batch_size % world_tokens_per_fwdbwd != 0:
    if args.total_batch_size == -1:
        # Auto batch size might not be divisible by world_tokens_per_fwdbwd.
        rounded = round(total_batch_size / world_tokens_per_fwdbwd) * world_tokens_per_fwdbwd
        if rounded == 0:
            rounded = world_tokens_per_fwdbwd
        print0(
            "Auto-computed total_batch_size isn't divisible by world_tokens_per_fwdbwd; "
            f"adjusting from {total_batch_size:,} to {rounded:,}."
        )
        total_batch_size = rounded
    else:
        raise ValueError(
            "total_batch_size must be a multiple of world_tokens_per_fwdbwd. "
            f"Got total_batch_size={total_batch_size:,}, world_tokens_per_fwdbwd={world_tokens_per_fwdbwd:,}. "
            "Adjust --total-batch-size, --device-batch-size, --max-seq-len, or DDP world size."
        )
    
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
batch_lr_scale = 1.0
reference_batch_size = 2**19
batch_ratio = total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard
    # Muon: sqrt scaling is an assumption - not fully studied, but it's a second-order-ish optimizer
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {reference_batch_size:,})")

# Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2 (or equivalently, \propto 1/depth^2 due to constant aspect ratio)
weight_decay_scaled = args.weight_decay * (12 / args.depth)**2
if args.depth != 12:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# After setup_optimizer(), one shouldn't change grad scale settings.
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    muon_match_rms_adamw=args.muon_match_rms_adamw,
)

if resuming and load_optimizer_state:
    optimizer_state_dict = load_optimizer_state_dict(
        checkpoint_dir,
        args.resume_from_step,
        optimizer,
        device,
        rank=ddp_rank,
        current_world_size=ddp_world_size,
        saved_world_size=saved_optimizer_world_size,
    )
    optimizer.load_state_dict(optimizer_state_dict)
    del optimizer_state_dict

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it, num_iterations, warmup_ratio, warmdown_ratio, 
                      final_lr_frac, lr_scheduler_skip_iters=0, lr_base_scale=1.0):
    it = max(0, it - lr_scheduler_skip_iters) # allow skipping the LR scheduler for the first N iterations (useful for redoing warmup when resuming from a later point in training)
    num_iterations = max(1, num_iterations - lr_scheduler_skip_iters) # avoid division by zero or negative iterations
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return lr_base_scale * (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return lr_base_scale * 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return lr_base_scale * (progress * 1.0 + (1 - progress) * final_lr_frac)

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
def get_weight_decay(it, weight_decay_scaled, num_iterations):
    return weight_decay_scaled * (1 - it / num_iterations)

def get_router_ortho_loss_weight(base_weight, it, num_anneal_iterations, floor_frac=0.01):
    if num_anneal_iterations <= 0:
        return base_weight
    progress = min(max(it, 0), num_anneal_iterations) / num_anneal_iterations
    # Anneal router_ortho_loss_weight from base_weight to a small floor over the anneal horizon.
    return base_weight * (floor_frac + (1.0 - floor_frac) * (1.0 - progress))

def get_router_wg_grad_scale(base_scale, target_scale, it, num_anneal_iterations):
    if num_anneal_iterations <= 0 or base_scale == target_scale:
        return base_scale
    anneal_progress = min(max(it, 0), num_anneal_iterations) / num_anneal_iterations
    return target_scale + (base_scale - target_scale) * (1.0 - anneal_progress)

# Hard-coded warmup before enabling blockwise router-ortho gating.
ROUTER_ORTHO_BLOCKWISE_WARMUP_STEPS = 1000

def get_router_ortho_blockwise_gate(it, block_size, on_prob, seed):
    """Deterministic per-block Bernoulli gate shared across ranks."""
    if on_prob >= 1.0:
        return 1.0, 1.0, it // block_size
    block_idx = it // block_size
    # Deterministic 64-bit mix to produce a stable pseudo-random value in [0, 1).
    state = (int(seed) ^ (block_idx * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 30)
    state = (state * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 27)
    state = (state * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 31)
    u = state / float(1 << 64)
    gate = 1.0 if u < on_prob else 0.0
    return gate, u, block_idx

def accumulate_step_losses(step_losses, micro_losses):
    """Accumulate detached per-microstep losses for step-level logging."""
    if step_losses is None:
        step_losses = {}

    for key, value in micro_losses.items():
        if value is None:
            step_losses.setdefault(key, None)
            continue

        if torch.is_tensor(value):
            detached_value = value.detach()
            if key not in step_losses or step_losses[key] is None:
                step_losses[key] = detached_value.clone()
            else:
                step_losses[key].add_(detached_value)
        else:
            if key not in step_losses or step_losses[key] is None:
                step_losses[key] = value
            else:
                step_losses[key] += value

    return step_losses

def average_step_losses(step_losses, grad_accum_steps):
    """Average accumulated losses across microsteps."""
    averaged_losses = {}
    for key, value in step_losses.items():
        if value is None:
            averaged_losses[key] = None
        elif torch.is_tensor(value):
            averaged_losses[key] = value / grad_accum_steps
        else:
            averaged_losses[key] = value / grad_accum_steps
    return averaged_losses

def collect_grad_stats(model, losses, moe_start_layer, n_layer):
    router_grad_norms = []
    router_grad_self_alignments = []
    router_weight_exp_alignments = []
    exp_gate_grad_norms = []
    expert_utilities = losses.get('expert_utilities', None)
    selected_scores = losses.get('selected_scores', None)
    router_wg_grad_dyn_scales = MANAGER.aggregate("router_wg_grad_dyn_scales")
    MANAGER.reset("router_wg_grad_dyn_scales")
    num_moe_layers = sum(
        1 for i in range(moe_start_layer, n_layer)
        if hasattr(model.transformer.h[i].mlp, 'experts')
    )
    if router_wg_grad_dyn_scales is not None and num_moe_layers > 0:
        if router_wg_grad_dyn_scales.shape[0] % num_moe_layers == 0:
            router_wg_grad_dyn_scales = router_wg_grad_dyn_scales.view(
                -1, num_moe_layers, router_wg_grad_dyn_scales.shape[1]
            ).mean(dim=0)
            # router_wg_grad_dyn_scales are collected during backward, where the layer order
            # is from higher to lower. Therefore we flip the tensor to be from lower to higher,
            # same as other quantities.
            router_wg_grad_dyn_scales = router_wg_grad_dyn_scales.flip(0)
        losses['router_wg_grad_dyn_scales'] = router_wg_grad_dyn_scales.detach()

    for i in range(moe_start_layer, n_layer):
        layer = model.transformer.h[i]
        if hasattr(layer.mlp, 'experts'):
            # [n_exp, hidden_size]
            router_gate_grad = layer.mlp.router.w_g.weight.grad
            router_grad_norm = router_gate_grad.norm(dim=1)
            router_grad_norms.append(router_grad_norm)
            losses[f'router_grad_norm_{i}'] = router_grad_norm.mean().item()
            exp_gate_grad = layer.mlp.experts.gate_proj.grad
            exp_gate_grad_norm = exp_gate_grad.norm(dim=(1,2))
            exp_gate_grad_norms.append(exp_gate_grad_norm)
            losses[f'exp_gate_grad_norm_{i}'] = exp_gate_grad_norm.mean().item()

            # Compute router grad - router weight alignment
            # Compute router expert - gate weight alignment
            with torch.inference_mode():
                router_weight = layer.mlp.router.w_g.weight  # [n_exp, hidden_size]
                exp_gate_mean_weight = layer.mlp.experts.gate_proj.mean(dim=2)  # [n_exp, hidden_size]
                # Compute the cosine similarity between router weights and router weight grads.
                # With SGD: Δw = -lr * ∇w. Since w·Δw = -lr*(w·∇w),
                # -(w·∇w) is positive when the update has a component along w (tends to increase ||w||),
                # and negative when it moves against w (tends to decrease ||w||). 
                rg_rw_alignment = -(router_gate_grad * router_weight).sum(dim=1) / (
                    router_weight.norm(dim=1) * router_gate_grad.norm(dim=1) + 1e-10
                )  # [n_exp]
                router_grad_self_alignments.append(rg_rw_alignment)
                mean_rg_rw_alignment = rg_rw_alignment.mean().item()
                losses[f'router_grad_self_alignment_{i}'] = mean_rg_rw_alignment

                # No negative sign here since these are weights, not gradients.
                rw_ew_alignment = (exp_gate_mean_weight * router_weight).sum(dim=1) / \
                        (router_weight.norm(dim=1) * (exp_gate_mean_weight.norm(dim=1) + 1e-10)) # [n_exp]
                router_weight_exp_alignments.append(rw_ew_alignment)
                mean_rw_ew_alignment = rw_ew_alignment.mean().item()
                losses[f'router_weight_exp_alignment_{i}'] = mean_rw_ew_alignment

                if expert_utilities is not None:
                    # expert_utilities: Tensor of shape (num_moe_layers, n_exp)
                    exp_utilities = expert_utilities[i - moe_start_layer]  # [n_exp]
                    half_experts = exp_utilities.shape[0] // 2
                    top_indices    = torch.topk(exp_utilities, k=half_experts, largest=True).indices
                    bottom_indices = torch.topk(exp_utilities, k=half_experts, largest=False).indices

                    top_rg_rw_alignment    = rg_rw_alignment[top_indices].mean().item()
                    bottom_rg_rw_alignment = rg_rw_alignment[bottom_indices].mean().item()
                    losses[f'router_grad_self_alignment_top_{i}']    = top_rg_rw_alignment
                    losses[f'router_grad_self_alignment_bottom_{i}'] = bottom_rg_rw_alignment

                    top_rw_ew_alignment    = rw_ew_alignment[top_indices].mean().item()
                    bottom_rw_ew_alignment = rw_ew_alignment[bottom_indices].mean().item()
                    losses[f'router_weight_exp_alignment_top_{i}']    = top_rw_ew_alignment
                    losses[f'router_weight_exp_alignment_bottom_{i}'] = bottom_rw_ew_alignment

                    top_router_grad_norm    = router_grad_norm[top_indices].mean().item()
                    bottom_router_grad_norm = router_grad_norm[bottom_indices].mean().item()
                    losses[f'router_grad_norm_top_{i}']    = top_router_grad_norm
                    losses[f'router_grad_norm_bottom_{i}'] = bottom_router_grad_norm

                    if selected_scores is not None:
                        # selected_scores: Tensor of shape (num_moe_layers, n_exp)
                        layer_selected_scores = selected_scores[i - moe_start_layer]  # [n_exp]
                        top_selected_scores    = layer_selected_scores[top_indices].mean().item()
                        bottom_selected_scores = layer_selected_scores[bottom_indices].mean().item()
                        losses[f'selected_scores_top_{i}']    = top_selected_scores
                        losses[f'selected_scores_bottom_{i}'] = bottom_selected_scores

                    if router_wg_grad_dyn_scales is not None and \
                       router_wg_grad_dyn_scales.shape[0] == expert_utilities.shape[0]:
                        layer_router_wg_grad_dyn_scale = router_wg_grad_dyn_scales[i - moe_start_layer]
                        top_router_wg_grad_dyn_scale = layer_router_wg_grad_dyn_scale[top_indices].mean().item()
                        bottom_router_wg_grad_dyn_scale = layer_router_wg_grad_dyn_scale[bottom_indices].mean().item()
                        losses[f'router_wg_grad_dyn_scale_top_{i}'] = top_router_wg_grad_dyn_scale
                        losses[f'router_wg_grad_dyn_scale_bottom_{i}'] = bottom_router_wg_grad_dyn_scale

    router_grad_norms = torch.stack(router_grad_norms, dim=0) if router_grad_norms else None
    losses['router_grad_norms'] = router_grad_norms
    router_grad_self_alignments = torch.stack(router_grad_self_alignments, dim=0) if router_grad_self_alignments else None
    losses['router_grad_self_alignments'] = router_grad_self_alignments
    router_weight_exp_alignments = torch.stack(router_weight_exp_alignments, dim=0) if router_weight_exp_alignments else None
    losses['router_weight_exp_alignments'] = router_weight_exp_alignments
    exp_gate_grad_norms = torch.stack(exp_gate_grad_norms, dim=0) if exp_gate_grad_norms else None
    losses['exp_gate_grad_norms'] = exp_gate_grad_norms

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    last_core_eval_step = None
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    last_core_eval_step = loop_state.get("last_core_eval_step")
    if last_core_eval_step is not None:
        last_core_eval_step = int(last_core_eval_step)
    else:
        last_core_eval_step = infer_last_completed_core_eval_step(
            checkpoint_dir,
            step,
            args.core_metric_every,
        )
        if last_core_eval_step is not None:
            print0(
                f"Recovered last completed CORE evaluation checkpoint from checkpoint directory: "
                f"step {last_core_eval_step:06d}."
            )

pending_milestones = [m for m in milestones if m > step]
if milestones:
    print0(f"Milestones configured: {milestones}")
if milestones and not pending_milestones:
    print0(f"All milestones are <= current step ({step}); no milestone spawn will be triggered.")
if args.mockup_mode:
    print0("Mockup mode enabled: skipping training/eval/sample compute and only advancing steps.")

terminate_after_checkpoint = False
core_results = {}

# -----------------------------------------------------------------------------
# Training loop
while True:
    is_last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    tokens_seen = total_batch_size * step
    flops_so_far = num_flops_per_token * tokens_seen
    router_ortho_num_anneal_iterations = num_iterations
    if args.router_ortho_loss_anneal_iterations > 0:
        router_ortho_num_anneal_iterations = min(
            router_ortho_num_anneal_iterations,
            args.router_ortho_loss_anneal_iterations,
        )
    router_ortho_loss_weight = get_router_ortho_loss_weight(
        args.router_ortho_loss_weight,
        step,
        router_ortho_num_anneal_iterations,
        floor_frac=args.router_ortho_loss_floor_frac,
    )
    if args.use_router_ortho_blockwise and step >= ROUTER_ORTHO_BLOCKWISE_WARMUP_STEPS:
        router_ortho_gate, router_ortho_gate_u, router_ortho_block_idx = get_router_ortho_blockwise_gate(
            step,
            args.router_ortho_block_size,
            args.router_ortho_on_prob,
            args.seed,
        )
    else:
        router_ortho_gate, router_ortho_gate_u, router_ortho_block_idx = 1.0, 1.0, step // args.router_ortho_block_size

    router_ortho_gate_scale = 1.0
    if args.use_router_ortho_blockwise and router_ortho_gate > 0.0 and args.router_ortho_blockwise_scale_preserve and args.router_ortho_on_prob < 1.0:
        router_ortho_gate_scale = 1.0 / args.router_ortho_on_prob
    router_ortho_effective_loss_weight = router_ortho_loss_weight * router_ortho_gate * router_ortho_gate_scale

    router_wg_grad_scale = get_router_wg_grad_scale(
        args.router_wg_grad_scale,
        args.router_wg_grad_scale_anneal_target,
        step,
        args.router_wg_grad_scale_anneal_iterations,
    )
    set_router_wg_grad_scale(orig_model, router_wg_grad_scale)

    # once in a while: evaluate the val bpb (all ranks participate)
    if (not args.mockup_mode) and args.eval_every > 0 and (is_last_step or (step > 0 and step % args.eval_every == 0)):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            # val_bpb: Compute summed loss over targets, but normalize by the number of bytes 
            # of the target text, not tokens.
            val_bpb, ntp_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "tokens_seen": tokens_seen,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
            "val/loss": ntp_loss,
        }, step=step)
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if is_last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        expected_optimizer_ranks = range(ddp_world_size)
        delete_old_ckpts_failed = False
        delete_old_ckpts_error = ""
        comparison_step = None
        reference_file_sizes = None
        keep_checkpoint_steps = [last_core_eval_step]
        if args.delete_old_ckpts and args.delete_old_ckpts_before_save and master_process:
            try:
                comparison_step, reference_file_sizes = snapshot_checkpoint_file_sizes(
                    checkpoint_dir,
                    step,
                    expected_optimizer_ranks=expected_optimizer_ranks,
                )
                delete_old_checkpoints(checkpoint_dir, step, keep_steps=keep_checkpoint_steps)
            except ValueError as exc:
                delete_old_ckpts_failed = True
                delete_old_ckpts_error = str(exc)
                print0(delete_old_ckpts_error)
                
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "optimizer_world_size": ddp_world_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "last_core_eval_step": last_core_eval_step,
                },
            },
            rank=ddp_rank,
        )
        if ddp:
            torch.distributed.barrier()
        if args.delete_old_ckpts and master_process and not delete_old_ckpts_failed:
            try:
                if args.delete_old_ckpts_before_save:
                    if comparison_step is None:
                        print0(
                            f"Skipped checkpoint file size validation at step {step}: "
                            "no prior checkpoint with matching file layout was found for file-size validation."
                        )
                    else:
                        validate_checkpoint_file_sizes(
                            checkpoint_dir,
                            step,
                            expected_optimizer_ranks=expected_optimizer_ranks,
                            comparison_step=comparison_step,
                            reference_file_sizes=reference_file_sizes,
                        )
                else:
                    comparison_step = validate_checkpoint_file_sizes(
                        checkpoint_dir,
                        step,
                        expected_optimizer_ranks=expected_optimizer_ranks,
                    )
                    if comparison_step is None:
                        print0(
                            f"Skipping old checkpoint deletion at step {step}: "
                            "no prior checkpoint with matching file layout was found for file-size validation."
                        )
                    else:
                        delete_old_checkpoints(checkpoint_dir, step, keep_steps=keep_checkpoint_steps)
            except ValueError as exc:
                delete_old_ckpts_failed = True
                delete_old_ckpts_error = str(exc)
                print0(delete_old_ckpts_error)
        if ddp:
            delete_status = torch.tensor(
                [1 if delete_old_ckpts_failed else 0],
                device=device,
                dtype=torch.int32,
            )
            torch.distributed.broadcast(delete_status, src=0)
            delete_old_ckpts_failed = bool(delete_status.item())
        if delete_old_ckpts_failed:
            if master_process:
                raise ValueError(delete_old_ckpts_error)
            raise RuntimeError(
                f"Checkpoint file size validation failed on rank 0 at step {step}. See rank 0 logs for details."
            )
        if master_process and pending_milestones:
            hit_milestones = [m for m in pending_milestones if step >= m]
            if hit_milestones:
                print0(f"Milestone(s) hit at step {step}: {hit_milestones}")
                pending_milestones = [m for m in pending_milestones if m > step]
                relaunch_cmd = build_self_command_with_milestones(pending_milestones, step)
                if relaunch_cmd is not None:
                    subprocess.Popen(relaunch_cmd, shell=True, start_new_session=True)
                    terminate_after_checkpoint = True
                    print0(
                        f"Milestone hit at step {step}: {hit_milestones}. "
                        f"Spawned self command with remaining milestones {pending_milestones} and --resume-from-step {step}: {relaunch_cmd}"
                    )

        if ddp:
            terminate_tensor = torch.tensor(
                [1 if terminate_after_checkpoint else 0],
                device=device,
                dtype=torch.int32,
            )
            torch.distributed.broadcast(terminate_tensor, src=0)
            terminate_after_checkpoint = bool(terminate_tensor.item())

    # If a milestone-triggered relaunch was spawned, stop immediately after
    # checkpoint save/broadcast to avoid doing expensive eval/sample work.
    if terminate_after_checkpoint and not is_last_step:
        print0(f"Stopping current run after milestone-triggered relaunch at step {step}.")
        break

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results

    if (not args.mockup_mode) and args.core_metric_every > 0 and (is_last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            # for the final evaluation at the end of training, run on the full set of tasks instead of a subset            
            max_per_task = args.core_metric_max_per_task if not is_last_step else -1 
            core_results = evaluate_core(orig_model, tokenizer, device, max_per_task=max_per_task)
        core_metric = core_results["core_metric"]
        print0(f"Step {step:05d} | CORE metric: {core_metric:.4f}")
        print0(f"Step {step:05d} | CORE metric (no boolq): {core_results['core_metric_no_boolq']:.4f}")
        wandb_run.log({
            "step": step,
            "tokens_seen": tokens_seen,
            "total_training_flops": flops_so_far,
            "core_metric": core_metric,
            "core_metric_no_boolq": core_results["core_metric_no_boolq"],
            "centered_results": core_results["centered_results"],
        }, step=step)
        last_core_eval_step = step
        model.train()

        # For the final evaluation at the end of training, write CSV output
        if is_last_step and ddp_rank == 0:
            model_slug = f"{output_dirname}_base_{step:06d}"
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
                f.write(f"{'CORE (no boolq)':<35}, {'':<10}, {core_results['core_metric_no_boolq']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")
            print0(f"CORE metric (no boolq): {core_results['core_metric_no_boolq']:.4f}")

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if (not args.mockup_mode) and args.sample_every > 0 and master_process and (is_last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if is_last_step or terminate_after_checkpoint:
        if terminate_after_checkpoint and not is_last_step:
            print0(f"Stopping current run after milestone-triggered relaunch at step {step}.")
        break

    MANAGER.collect_load_balancing_stats = args.log_grad_stats and (step % args.log_interval == 0)
    MANAGER.collect_backward_stats = False

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    if args.mockup_mode:
        lrm = get_lr_multiplier(step, num_iterations, args.warmup_ratio, args.warmdown_ratio, 
                                args.final_lr_frac, lr_scheduler_skip_iters=args.lr_scheduler_skip_iters, 
                                lr_base_scale=args.lr_base_scale)
        losses = {
            'ntp_loss': 0.0,
            'aux_loss': 0.0,
            'gated_aux_loss': 0.0,
            'router_z_loss': 0.0,
            'router_ortho_loss': 0.0,
            'experts_ortho_loss': 0.0,
            'experts_gate_output_loss': 0.0,
            'projs_diversity_loss': 0.0,
            'drop_rate_per_ks': None,
        }
        train_loss_f = 0.0
        dt = 1.0
    else:
        synchronize()
        t0 = time.time()
        step_losses = None
        for micro_step in range(grad_accum_steps):
            MANAGER.collect_backward_stats = (
                MANAGER.collect_load_balancing_stats and micro_step == grad_accum_steps - 1
            )
            with autocast_ctx:
                loss, micro_losses = model(x, y)
            step_losses = accumulate_step_losses(step_losses, micro_losses)
            # Most values in losses are detached and for logging only, but router_ortho_loss is not.
            router_ortho_loss = micro_losses['router_ortho_loss'] 
            loss = loss + router_ortho_effective_loss_weight * router_ortho_loss
            
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            loss.backward()
            MANAGER.collect_backward_stats = False
            x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward

        losses = average_step_losses(step_losses, grad_accum_steps)

        if MANAGER.collect_load_balancing_stats:
            collect_grad_stats(model, losses, args.moe_start_layer, args.depth)
        
        # step the optimizer
        lrm = get_lr_multiplier(step, num_iterations, args.warmup_ratio, args.warmdown_ratio, 
                                args.final_lr_frac, lr_scheduler_skip_iters=args.lr_scheduler_skip_iters, 
                                lr_base_scale=args.lr_base_scale)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step, weight_decay_scaled, num_iterations)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        orig_model.update_aux_free_load_balancing()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        train_loss_f = losses['ntp_loss'].item() # .item() is a CPU-GPU sync point
        synchronize()
        t1 = time.time()
        dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    # We don't do EMA on other types of losses (e.g. router ortho loss). Just the main NTP loss.
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % args.log_interval == 0:
        log_data = {
            "step": step,
            "tokens_seen": tokens_seen,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss_step":              debiased_smooth_loss,
            "train/aux_loss_step":          losses['aux_loss'],
            "train/gated_aux_loss_step":   losses['gated_aux_loss'],
            "train/router_z_loss_step":     losses['router_z_loss'],
            "train/router_ortho_loss_step": losses['router_ortho_loss'].detach().item(),
            "train/router_ortho_loss_weight": router_ortho_loss_weight,
            "train/experts_ortho_loss_step": losses['experts_ortho_loss'],
            "train/experts_gate_output_loss_step": losses['experts_gate_output_loss'],
            "train/projs_diversity_loss_step": losses['projs_diversity_loss'],
            "lrm": lrm,
            "router_wg_grad_scale": router_wg_grad_scale,
            "dt": dt,
            "tok_per_sec": tok_per_sec,
            "mfu": mfu,
            "epoch": epoch,
        }
        drop_rates = losses['drop_rate_per_ks']
        if drop_rates is not None:
            if len(drop_rates) >= 1:
                log_data["inspect/drop_rate_0_step"] = drop_rates[0]
            if len(drop_rates) >= 2:
                log_data["inspect/drop_rate_1_step"] = drop_rates[1]
        expert_utilities = losses['expert_utilities']
        for i in range(args.moe_start_layer, args.depth):
            if expert_utilities is not None:
                layer_expert_utilities = expert_utilities[i - args.moe_start_layer]
                log_data.update({f"inspect/expert_utility_min_{i}": layer_expert_utilities.min().item()})
                log_data.update({f"inspect/expert_utility_max_{i}": layer_expert_utilities.max().item()})
                log_data.update({f"inspect/expert_utility_mean_{i}": layer_expert_utilities.mean().item()})
            if f'router_grad_norm_top_{i}' in losses:
                log_data.update({f"inspect/router_grad_norm_top_{i}": losses[f'router_grad_norm_top_{i}']})
            if f'router_grad_norm_bottom_{i}' in losses:
                log_data.update({f"inspect/router_grad_norm_bottom_{i}": losses[f'router_grad_norm_bottom_{i}']})
            if f'router_grad_self_alignment_top_{i}' in losses:
                log_data.update({f"inspect/router_grad_self_alignment_top_{i}": losses[f'router_grad_self_alignment_top_{i}']})
            if f'router_grad_self_alignment_bottom_{i}' in losses:
                log_data.update({f"inspect/router_grad_self_alignment_bottom_{i}": losses[f'router_grad_self_alignment_bottom_{i}']})
            if f'router_weight_exp_alignment_top_{i}' in losses:
                log_data.update({f"inspect/router_weight_exp_alignment_top_{i}": losses[f'router_weight_exp_alignment_top_{i}']})
            if f'router_weight_exp_alignment_bottom_{i}' in losses:
                log_data.update({f"inspect/router_weight_exp_alignment_bottom_{i}": losses[f'router_weight_exp_alignment_bottom_{i}']})
            if f'selected_scores_top_{i}' in losses:
                log_data.update({f"inspect/selected_scores_top_{i}": losses[f'selected_scores_top_{i}']})
            if f'selected_scores_bottom_{i}' in losses:
                log_data.update({f"inspect/selected_scores_bottom_{i}": losses[f'selected_scores_bottom_{i}']})
            if f'router_wg_grad_dyn_scale_top_{i}' in losses:
                log_data.update({f"inspect/router_wg_grad_dyn_scale_top_{i}": losses[f'router_wg_grad_dyn_scale_top_{i}']})
            if f'router_wg_grad_dyn_scale_bottom_{i}' in losses:
                log_data.update({f"inspect/router_wg_grad_dyn_scale_bottom_{i}": losses[f'router_wg_grad_dyn_scale_bottom_{i}']})
                        
        wandb_run.log(log_data, step=step)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": core_results.get("core_metric", None),
        "CORE metric estimate (no boolq)": core_results.get("core_metric_no_boolq", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
