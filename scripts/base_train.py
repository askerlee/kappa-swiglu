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
import sys
import signal
import shlex
from contextlib import nullcontext, contextmanager
import re

import wandb
import torch

from nanochat.gpt import GPT, get_moe_layer_indices
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import delete_checkpoint_step, delete_old_checkpoints, save_checkpoint, load_checkpoint, inspect_optimizer_shards, load_optimizer_state_dict, snapshot_checkpoint_file_sizes, validate_checkpoint_file_sizes
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FLASH_ATTN, FLASH_ATTN_BACKEND, ALLOW_FA4_TRAINING
from scripts.base_eval import evaluate_core
from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.manager import MANAGER
torch.set_printoptions(sci_mode=False)

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


shutdown_requested = False
shutdown_signal_name = None


def handle_shutdown_signal(signum, frame):
    del frame
    global shutdown_requested
    global shutdown_signal_name
    shutdown_requested = True
    try:
        shutdown_signal_name = signal.Signals(signum).name
    except ValueError:
        shutdown_signal_name = f"signal {signum}"


def build_chat_sft_exec_argv(
    python_executable,
    model_tag,
    model_step,
    extra_args_text="",
):
    import shlex

    argv = [
        python_executable,
        "-m",
        "scripts.chat_sft",
        "--model-tag",
        model_tag,
        "--model-step",
        str(model_step),
    ]
    if extra_args_text:
        argv.extend(shlex.split(extra_args_text))
    return argv


def pick_free_tcp_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def prepare_chat_sft_rendezvous(ddp, ddp_rank, device):
    import torch

    if not ddp:
        return None

    chat_sft_master_port = 0
    if ddp_rank == 0:
        chat_sft_master_port = pick_free_tcp_port()
    port_tensor = torch.tensor([chat_sft_master_port], device=device, dtype=torch.int64)
    torch.distributed.broadcast(port_tensor, src=0)
    chat_sft_master_port = int(port_tensor.item())
    os.environ["MASTER_PORT"] = str(chat_sft_master_port)
    torch.distributed.barrier()
    return chat_sft_master_port


def sanitize_chat_sft_rendezvous_env():
    # chat_sft reuses the existing torchrun workers via exec(), but it needs to
    # form a fresh TCPStore on the new MASTER_PORT instead of reusing the
    # torchelastic agent store semantics from the previous job.
    # base_train.py was creating a fresh MASTER_PORT for the exec into chat SFT, 
    # but it was still leaving TORCHELASTIC_USE_AGENT_STORE in the environment. 
    # Under torchrun that makes the new process-group init treat the rendezvous 
    # like an agent-managed store.
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    
# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
DEFAULT_SEED = 26
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed for initialization")
parser.add_argument("--mockup-mode", type=str2bool, nargs='?', const=True, default=False, help="skip actual training/eval/sample compute and only advance step counter")
# FP8 training
parser.add_argument("--fp8", type=str2bool, nargs='?', const=True, default=False, help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=8, help="depth of the Transformer model")
parser.add_argument("--moe-start-layer", type=int, default=2, help="first layer index of MoE layers")
parser.add_argument("--num-moe-layers", type=int, default=-1, help="number of MoE layers to instantiate from --moe-start-layer onward (-1 = all eligible layers)")
parser.add_argument("--n-exp", type=int, default=64, help="number of experts per MoE layer")
parser.add_argument("--moe-top-k", type=int, default=2, help="top-k of the MoE routing")
parser.add_argument("--use-aux-free-load-balancing", type=str2bool, nargs='?', const=True, default=False, help="enable DeepSeekV3 auxiliary-loss-free load balancing instead of the Switch auxiliary router loss")
parser.add_argument("--aux-loss-weight", type=float, default=1e-3, help="final weight for the Switch-style router auxiliary load-balancing loss after the initial 500-step anneal")
parser.add_argument("--aux-loss-weight-init-scale", type=float, default=2.0, help="initial aux loss weight scale factor; the anneal starts from --aux-loss-weight * this value")
parser.add_argument("--aux-loss-weight-init-anneal-iterations", type=int, default=500, help="number of iterations used to anneal aux loss weight from --aux-loss-weight * --aux-loss-weight-init-scale down to --aux-loss-weight")
# router ortho loss is around 10 (if the loss is enabled). So * weight = 1e-4.
parser.add_argument("--router-ortho-loss-weight", type=float, default=0, help="weight for router orthogonality loss")
parser.add_argument("--router-ortho-loss-warmup-iterations", type=int, default=500, help="number of iterations to linearly ramp router ortho loss weight from 0 up to --router-ortho-loss-weight before annealing")
parser.add_argument("--router-ortho-loss-anneal-iterations", type=int, default=-1, help="Total anneal iterations for the router ortho loss")
parser.add_argument("--router-ortho-loss-floor-frac", type=float, default=0, help="fraction of the base router ortho loss weight to keep after annealing completes")
parser.add_argument("--use-router-ortho-blockwise", type=str2bool, nargs='?', const=True, default=True, 
                    help="Enable blockwise on/off schedule for router-ortho loss to counter the memory effect of Muon")
parser.add_argument("--router-ortho-block-size", type=int, default=100, help="block size (in optimizer steps) for blockwise router-ortho loss gating")
parser.add_argument("--router-ortho-on-prob", type=float, default=0.8, help="probability a router-ortho block is active; set to 1.0 to disable blockwise gating")
parser.add_argument("--router-ortho-neg-corr-weight", type=float, default=1, help="weight for negative correlations in router-ortho loss.")
parser.add_argument("--use-exp-gate-proj-bias", type=str2bool, nargs='?', const=True, default=False,
                    help="add a learnable bias to Qwen3 expert gate activations after gate_proj and SiLU")
parser.add_argument("--exp-gate-proj-bias-mode", type=str, default="rank1_residual", choices=["full", "rank1", "rank1_residual"],
                    help="parameterization for expert gate_proj_bias: full matrix, rank-1 expert/intermediate factors, or rank-1 plus dense residual")
parser.add_argument("--gate-proj-bias-start-layer", type=int, default=None,
                    help="first transformer layer index where MoE gate_proj_bias is enabled (default: when omitted and MoE is enabled, use min(moe_start_layer + 2, depth//2, 5))")
parser.add_argument("--gate-proj-bias-lr-max-scale", type=float, default=0.5,
                    help="peak LR scale factor for gate_proj_bias params after warming from 0 before annealing to --gate-proj-bias-lr-final-scale")
parser.add_argument("--gate-proj-bias-lr-final-scale", type=float, default=0.01,
                    help="final LR scale factor for gate_proj_bias params after warming from 0 to 1")
parser.add_argument("--gate-proj-bias-delay-start-iterations", type=int, default=200,
                    help="number of initial iterations to keep gate_proj_bias LR at 0 before warmup and annealing")
parser.add_argument("--gate-proj-bias-lr-warmup-iterations", type=int, default=1000,
                    help="number of iterations to linearly ramp gate_proj_bias LR scale from 0 to --gate-proj-bias-lr-max-scale before annealing to --gate-proj-bias-lr-final-scale")
# L2 and hinge losses on gate_proj_bias.
# The L2 loss is typically 0.01~0.04. The hinge loss is typically 0.0001~0.0002.
# Therefore the hinge loss is ~ 1/100 of the L2 loss.
# Since both are quadratic, the grad scale is roughly 10:1.
# So we set the default weight of the hinge loss to 10x the L2 loss weight 
# to make the grad scale of the hinge loss comparable to that of the L2 loss. 
parser.add_argument("--gate-proj-bias-l2-loss-weight", type=float, default=1e-2, help="weight for MoE gate_proj_bias L2 loss")
parser.add_argument("--gate-proj-bias-residual-l2-loss-weight", type=float, default=0.0, help="weight for the dense residual part of MoE gate_proj_bias when using rank1_residual")
parser.add_argument("--gate-proj-bias-l2-loss-anneal-iterations", type=int, default=-1, help="iterations for stage-1 anneal of the MoE (2D) gate_proj_bias L2 loss from 1.0 to --gate-proj-bias-l2-loss-stage1-frac (-1 = use half total training iterations)")
parser.add_argument("--gate-proj-bias-l2-loss-stage1-frac", type=float, default=0.1, help="fraction of the MoE (2D) gate_proj_bias L2 base weight to reach at the end of stage 1 (1 = no stage-1 annealing)")
parser.add_argument("--gate-proj-bias-l2-loss-final-frac", type=float, default=0.02, help="fraction of the MoE (2D) gate_proj_bias L2 base weight to reach at the end of training during stage 2")
parser.add_argument("--gate-proj-bias-shift-abs-mean-half-slope-start", type=float, default=0.1,
                    help="lower threshold a for the normalized gate-proj-bias band loss; below this there is no penalty, and <= 0 disables the loss")
parser.add_argument("--gate-proj-bias-shift-abs-mean-full-slope-start", type=float, default=0.13,
                    help="upper threshold b for the normalized gate-proj-bias band loss; slope is half-strength between a and b and full-strength above b")
parser.add_argument("--gate-proj-bias-abs-mean-loss-weight-scale", type=float, default=0.05,
                    help="scale factor applied to the L1 loss weight to get the MoE gate_proj_bias abs-mean hinge loss weight")
parser.add_argument("--bilinear-mlp-moe", type=str2bool, nargs='?', const=True, default=False,
                    help="disable the SiLU gate in Qwen3-style MoE MLPs only, using raw bilinear gating in expert layers")
# router-z-loss is around 200. So * weight ~ 0.002.
parser.add_argument("--router-z-loss-weight", type=float, default=1e-5, help="weight for router z loss")
parser.add_argument("--router-z-loss-input-grad-scale", type=float, default=0.1, help="scaling factor for gradients to router input when computing router z loss. Setting this to a value < 1.0 can help stabilize training by preventing large z-loss gradients from destabilizing the router input representations.")
parser.add_argument("--z-loss-demean-logits", type=str2bool, nargs='?', const=True, default=True, help="use logits-demeaned router z loss")
parser.add_argument("--z-loss-penalize-mean-logits", type=str2bool, nargs='?', const=True, default=True, help="penalize mean logits in router z loss")
parser.add_argument("--aspect-ratio", type=int, default=96, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="LLLL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
parser.add_argument("--use-moe-adjusted-scaling-params", type=str2bool, nargs='?', const=True, default=True,
                    help="use MoE-adjusted scaling params instead of raw scaling params when --target-param-data-ratio determines target tokens")
# Optimization
parser.add_argument("--compile", type=str2bool, nargs='?', const=True, default=True, help="use torch.compile to speed up training (may cause instability, use with caution)")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument(
    "--total-batch-size",
    type=int,
    default=-1,
    help=(
        "total batch size in tokens. Must currently be divisible by "
        "--device-batch-size * --max-seq-len * DDP world size because each "
        "micro-step uses a fixed-shape batch and padded rows would still "
        "affect auxiliary MoE losses. Decent numbers are e.g. 524288. "
        "(-1 = auto-compute optimal)"
    ),
)
parser.add_argument("--max-auto-grad-accum-steps", type=int, default=64, help="cap gradient accumulation steps when --total-batch-size=-1 (-1 = disable cap)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.05, help="cautious weight decay for Transformer layer weights in the Muon optimizer")
parser.add_argument("--matrix-lr", type=float, default=0.01, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--matrix-optimizer", type=str, default="muon", choices=["muon", "aurora"], help="matrix optimizer for 2D parameters")
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
parser.add_argument("--core-metric-every", type=int, default=1000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=1000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--save-optimizer-state", type=str2bool, nargs='?', const=True, default=True, help="save optimizer shards alongside model checkpoints")
parser.add_argument("--delete-old-ckpts", type=str2bool, nargs='?', const=True, default=True, help="after saving a checkpoint, delete all older checkpoints based on step number")
parser.add_argument("--delete-old-ckpts-before-save", action="store_true", help="delete old checkpoints before saving the new checkpoint; keeps file-size validation by snapshotting the previous checkpoint sizes first")
parser.add_argument("--continue-to-chat-sft", action="store_true", help="after a successful base training run, exec scripts.chat_sft from the final base checkpoint; when launched under torchrun, each existing worker continues in place with the same world size")
parser.add_argument("--continue-to-chat-sft-args", type=str, default="", help="extra CLI args forwarded to scripts.chat_sft when --continue-to-chat-sft is set")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
parser.add_argument("--wandb-api-key-file", type=str, default=None, help="Weights & Biases API key file (optional). If provided, sets WANDB_API_KEY for this run")
parser.add_argument("--log-grad-stats", action="store_true", help="log gradient statistics for MoE layers")
parser.add_argument("--log-interval", type=int, default=20, help="interval (in steps) for logging grad stats")
parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False)

args = parser.parse_args()
gate_proj_bias_l2_loss_weight_was_specified = arg_was_explicitly_set(
    sys.argv[1:],
    '--gate-proj-bias-l2-loss-weight',
)
gate_proj_bias_residual_l2_loss_weight_was_specified = arg_was_explicitly_set(
    sys.argv[1:],
    '--gate-proj-bias-residual-l2-loss-weight',
)
gate_proj_bias_abs_mean_loss_weight_scale_was_specified = arg_was_explicitly_set(
    sys.argv[1:],
    '--gate-proj-bias-abs-mean-loss-weight-scale',
)
if args.model_tag is not None and arg_was_explicitly_set(sys.argv[1:], '--seed'):
    args.model_tag = f"{args.model_tag}-s{args.seed}"
if args.debug:
    args.compile = False

if args.router_ortho_block_size <= 0:
    raise ValueError("--router-ortho-block-size must be > 0")
if not (0.0 < args.router_ortho_on_prob <= 1.0):
    raise ValueError("--router-ortho-on-prob must be in (0, 1]")
if args.router_ortho_loss_warmup_iterations < 0:
    raise ValueError("--router-ortho-loss-warmup-iterations must be >= 0")
if args.gate_proj_bias_delay_start_iterations < 0:
    raise ValueError("--gate-proj-bias-delay-start-iterations must be >= 0")
if args.gate_proj_bias_shift_abs_mean_half_slope_start > 0:
    args.gate_proj_bias_shift_abs_mean_half_slope_start = float(args.gate_proj_bias_shift_abs_mean_half_slope_start)
else:
    args.gate_proj_bias_shift_abs_mean_half_slope_start = None
if args.gate_proj_bias_shift_abs_mean_full_slope_start > 0:
    args.gate_proj_bias_shift_abs_mean_full_slope_start = float(args.gate_proj_bias_shift_abs_mean_full_slope_start)
else:
    args.gate_proj_bias_shift_abs_mean_full_slope_start = None
if (
    args.gate_proj_bias_shift_abs_mean_half_slope_start is not None
    and args.gate_proj_bias_shift_abs_mean_full_slope_start is not None
    and args.gate_proj_bias_shift_abs_mean_full_slope_start <= args.gate_proj_bias_shift_abs_mean_half_slope_start
):
    raise ValueError(
        "--gate-proj-bias-shift-abs-mean-full-slope-start must be > --gate-proj-bias-shift-abs-mean-half-slope-start"
    )
if args.gate_proj_bias_abs_mean_loss_weight_scale < 0:
    raise ValueError("--gate-proj-bias-abs-mean-loss-weight-scale must be >= 0")
if args.aux_loss_weight_init_scale <= 0.0:
    raise ValueError("--aux-loss-weight-init-scale must be > 0")
if args.aux_loss_weight_init_anneal_iterations < 0:
    raise ValueError("--aux-loss-weight--init-anneal-iterations must be >= 0")
if not (0.0 <= args.gate_proj_bias_l2_loss_final_frac <= args.gate_proj_bias_l2_loss_stage1_frac <= 1.0):
    raise ValueError(
        "--gate-proj-bias-l2-loss-final-frac and --gate-proj-bias-l2-loss-stage1-frac must satisfy "
        "0 <= final_frac <= stage1_frac <= 1"
    )
if (
    args.exp_gate_proj_bias_mode in {"rank1", "rank1_residual"}
    and not gate_proj_bias_l2_loss_weight_was_specified
    and args.gate_proj_bias_l2_loss_weight == parser.get_default("gate_proj_bias_l2_loss_weight")
):
    args.gate_proj_bias_l2_loss_weight = 5e-3
if (
    args.exp_gate_proj_bias_mode in {"rank1", "rank1_residual"}
    and not gate_proj_bias_abs_mean_loss_weight_scale_was_specified
    and args.gate_proj_bias_abs_mean_loss_weight_scale == parser.get_default("gate_proj_bias_abs_mean_loss_weight_scale")
):
    args.gate_proj_bias_abs_mean_loss_weight_scale = 0
if (
    args.exp_gate_proj_bias_mode == "rank1_residual"
    and not gate_proj_bias_residual_l2_loss_weight_was_specified
    and args.gate_proj_bias_residual_l2_loss_weight == parser.get_default("gate_proj_bias_residual_l2_loss_weight")
):
    # Use the same L2 loss weight for the residual part as for the rank-1 part by default.
    # gate_proj_bias_l2_loss is computed on the full materialized bias, and 
    # for rank1_residual that materialized bias is rank1 + residual. 
    # Then gate_proj_bias_residual_l2_loss adds a second penalty on just the residual matrix.
    # i.e., the residual is penalized more strongly than the rank-1 part, 
    # which encourages the model to be approximately rank-1 and only use the residual 
    # for smaller corrections.
    args.gate_proj_bias_residual_l2_loss_weight = args.gate_proj_bias_l2_loss_weight * 2
    
# num_moe_layers: 
# -1 (default): all layers from moe_start_layer
# 0: no moe layers, i.e., a dense model
# N: N moe layers from moe_start_layer
if args.num_moe_layers < -1:
    raise ValueError("--num-moe-layers must be >= -1")
elif args.num_moe_layers == 0:
    args.router_ortho_loss_weight = 0
    print("Setting router orthogonality loss weight to 0 because --num-moe-layers=0")
effective_moe_layer_count = len(get_moe_layer_indices(argparse.Namespace(
    n_exp=args.n_exp,
    num_moe_layers=args.num_moe_layers,
    moe_start_layer=args.moe_start_layer,
    moe_layer_stride=1,
    n_layer=args.depth,
)))
if args.use_moe_adjusted_scaling_params and effective_moe_layer_count < args.depth / 5:
    args.use_moe_adjusted_scaling_params = False
    print(
        "Disabling --use-moe-adjusted-scaling-params because the effective number of MoE layers "
        f"({effective_moe_layer_count}) is less than one fifth of depth ({args.depth / 5:.2f})."
    )
if args.gate_proj_bias_start_layer is None:
    if args.num_moe_layers != 0:
        # If depth = 4, start layer = 2; if depth = 6, start layer = 3;
        # If depth = 8, start layer = 4; 
        # if depth >= 10, start layer = 5 (capped to avoid missing too many moe layers 
        # and reducing the benefit of gate_proj_bias).
        # moe_start_layer + 2: at most skip the first 2 moe layers, 
        # to avoid missing too many moe layers.
        # If depth = 10 and moe_start_layer = 2, then bias starts at layer 4 instead of 5.
        args.gate_proj_bias_start_layer = min(args.moe_start_layer + 2, args.depth // 2, 5)
    else:
        args.gate_proj_bias_start_layer = 0
if args.gate_proj_bias_start_layer < 0:
    raise ValueError("--gate-proj-bias-start-layer must be >= 0")
if args.max_auto_grad_accum_steps != -1 and args.max_auto_grad_accum_steps < 1:
    raise ValueError("--max-auto-grad-accum-steps must be >= 1 or -1 to disable the cap")
if args.use_aux_free_load_balancing:
    print("Disabling auxiliary router loss because --use-aux-free-load-balancing is enabled.")

user_config = vars(args).copy()  # for logging
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
    wandb.define_metric("step")
    wandb.define_metric("tokens_seen")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("val/*", step_metric="step")

# Flash Attention status
if HAS_FLASH_ATTN:
    backend_label = {
        "fa3": "Flash Attention 3",
        "fa4": "Flash Attention 4",
    }.get(FLASH_ATTN_BACKEND, "Flash Attention")
    if FLASH_ATTN_BACKEND == "fa4" and not ALLOW_FA4_TRAINING:
        print0(f"✓ {backend_label} is available, but training defaults to PyTorch SDPA to avoid unrecoverable FA4 backward OOMs.")
        print0("  Set NANOCHAT_ALLOW_FA4_TRAINING=1 to opt back into FA4 training.")
    else:
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
        num_moe_layers=args.num_moe_layers,
        n_exp=args.n_exp, moe_top_k=args.moe_top_k,
        use_aux_loss=not args.use_aux_free_load_balancing,
        use_aux_free_load_balancing=args.use_aux_free_load_balancing,
        aux_loss_weight=args.aux_loss_weight,
        router_ortho_loss_weight=args.router_ortho_loss_weight,
        router_ortho_neg_corr_weight=args.router_ortho_neg_corr_weight,
        use_exp_gate_proj_bias=args.use_exp_gate_proj_bias,
        exp_gate_proj_bias_mode=args.exp_gate_proj_bias_mode,
        exp_gate_proj_bias_input="top_logits",
        gate_proj_bias_start_layer=args.gate_proj_bias_start_layer,
        gate_proj_bias_l2_loss_weight=args.gate_proj_bias_l2_loss_weight,
        gate_proj_bias_residual_l2_loss_weight=args.gate_proj_bias_residual_l2_loss_weight,
        gate_proj_bias_shift_abs_mean_half_slope_start=args.gate_proj_bias_shift_abs_mean_half_slope_start,
        gate_proj_bias_shift_abs_mean_full_slope_start=args.gate_proj_bias_shift_abs_mean_full_slope_start,
        bilinear_mlp_moe=args.bilinear_mlp_moe,
        router_z_loss_weight=args.router_z_loss_weight,
        router_z_loss_input_grad_scale=args.router_z_loss_input_grad_scale,
        z_loss_demean_logits=args.z_loss_demean_logits,
        z_loss_penalize_mean_logits=args.z_loss_penalize_mean_logits,
        n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
        debug=args.debug
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
moe_layer_indices = get_moe_layer_indices(model_config)
model_config_kwargs = vars(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

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
router_ortho_loss_name = "router_ortho_loss"
router_ortho_sub_loss_names = ("router_ortho_loss_gate_proj",)

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
num_scaling_params = param_counts['transformer_matrices'] + param_counts['lm_head']
moe_adjusted_scaling_params = model.get_moe_adjusted_scaling_params(args.n_exp, args.moe_top_k)
print0(f"MoE-adjusted scaling parameters: {int(moe_adjusted_scaling_params):,}")
target_scaling_params = moe_adjusted_scaling_params if args.use_moe_adjusted_scaling_params else num_scaling_params
target_scaling_params_label = "MoE-adjusted scaling params" if args.use_moe_adjusted_scaling_params else "scaling params"
print0(f"Using {target_scaling_params_label} for --target-param-data-ratio: {int(target_scaling_params):,}")
target_tokens = int(args.target_param_data_ratio * target_scaling_params)
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks

# Auto-compute optimal batch size based on Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
total_batch_size = args.total_batch_size
if total_batch_size == -1:
    d12_ref = build_model_meta(12) # d12 is where the optimal batch size was measured to be 2**19 tokens
    d12_moe_adjusted_scaling_params = d12_ref.get_moe_adjusted_scaling_params(args.n_exp, args.moe_top_k)
    d12_target_scaling_params = d12_moe_adjusted_scaling_params if args.use_moe_adjusted_scaling_params else (
        d12_ref.num_scaling_params()['transformer_matrices'] + d12_ref.num_scaling_params()['lm_head']
    )
    D_REF = args.target_param_data_ratio * d12_target_scaling_params
    B_REF = 2**19
    batch_size_ratio = target_tokens / D_REF
    total_batch_size = 2 ** round(math.log2(B_REF * batch_size_ratio ** 0.383)) # also clamp to power of 2
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")
    if args.max_auto_grad_accum_steps != -1:
        max_auto_total_batch_size = world_tokens_per_fwdbwd * args.max_auto_grad_accum_steps
        if total_batch_size > max_auto_total_batch_size:
            print0(
                "Auto-computed total_batch_size would require too many gradient accumulation steps; "
                f"capping from {total_batch_size:,} to {max_auto_total_batch_size:,} "
                f"to respect --max-auto-grad-accum-steps={args.max_auto_grad_accum_steps}."
            )
            total_batch_size = max_auto_total_batch_size

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
print0(f"Tokens : {target_scaling_params_label} ratio: {total_batch_size * num_iterations / target_scaling_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
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
            "total_batch_size must be a multiple of world_tokens_per_fwdbwd "
            "(= --device-batch-size * --max-seq-len * DDP world size). "
            f"Got total_batch_size={total_batch_size:,}, world_tokens_per_fwdbwd={world_tokens_per_fwdbwd:,}. "
            "This script currently uses fixed-shape micro-batches, and simply padding the "
            "remainder would change auxiliary/router losses instead of only masking the LM loss. "
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
    print0(
        f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} "
        f"for depth {args.depth}"
    )

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# After setup_optimizer(), one shouldn't change parameter-group LR scaling settings.
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    matrix_optimizer=args.matrix_optimizer,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    muon_match_rms_adamw=args.muon_match_rms_adamw,
    gate_proj_bias_lr_final_scale=args.gate_proj_bias_lr_final_scale,
    gate_proj_bias_lr_max_scale=args.gate_proj_bias_lr_max_scale,
    gate_proj_bias_delay_start_iterations=args.gate_proj_bias_delay_start_iterations,
    gate_proj_bias_lr_warmup_iterations=args.gate_proj_bias_lr_warmup_iterations,
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
def get_weight_decay(base_weight_decay, it, num_iterations):
    return base_weight_decay * (1 - it / num_iterations)

def get_router_ortho_loss_weight(base_weight, it, num_warmup_iterations=0, num_anneal_iterations=-1, floor_frac=0.01):
    warmup_weight = base_weight
    if num_warmup_iterations > 0:
        warmup_progress = min(max(it, 0), num_warmup_iterations) / num_warmup_iterations
        warmup_weight = base_weight * warmup_progress

    if num_anneal_iterations <= 0:
        return warmup_weight

    anneal_step = max(it - num_warmup_iterations, 0)
    anneal_progress = min(anneal_step, num_anneal_iterations) / num_anneal_iterations
    # Warm up router_ortho_loss_weight to base_weight, then anneal it to a small floor.
    anneal_multiplier = floor_frac + (1.0 - floor_frac) * (1.0 - anneal_progress)
    return warmup_weight * anneal_multiplier


def get_annealed_loss_weight(base_weight, it, num_anneal_iterations=500, final_weight=1e-3):
    if num_anneal_iterations <= 0 or base_weight <= final_weight:
        return final_weight if base_weight <= final_weight else base_weight

    anneal_progress = min(max(it, 0), num_anneal_iterations) / num_anneal_iterations
    return base_weight + (final_weight - base_weight) * anneal_progress


def get_two_stage_annealed_loss_weight(base_weight, it, total_iterations, stage1_iterations=-1, stage1_floor_frac=0.1, final_floor_frac=0.01, nolearn_iterations=0):
    total_iterations = max(int(total_iterations), 1)
    effective_nolearn_iterations = min(max(int(nolearn_iterations), 0), total_iterations)
    if it < effective_nolearn_iterations:
        return 0.0

    effective_total_iterations = max(total_iterations - effective_nolearn_iterations, 1)
    effective_it = min(max(int(it) - effective_nolearn_iterations, 0), effective_total_iterations)
    if stage1_iterations <= 0:
        stage1_iterations = max((effective_total_iterations + 1) // 2, 1)
    stage1_iterations = min(max(int(stage1_iterations), 0), effective_total_iterations)

    if stage1_iterations > 0 and effective_it <= stage1_iterations:
        stage1_progress = min(max(effective_it, 0), stage1_iterations) / stage1_iterations
        stage1_multiplier = stage1_floor_frac + (1.0 - stage1_floor_frac) * (1.0 - stage1_progress)
        return base_weight * stage1_multiplier

    if stage1_iterations >= effective_total_iterations:
        return base_weight * stage1_floor_frac

    stage2_iterations = effective_total_iterations - stage1_iterations
    stage2_step = effective_it - stage1_iterations
    stage2_progress = min(max(stage2_step, 0), stage2_iterations) / stage2_iterations
    stage2_multiplier = final_floor_frac + (stage1_floor_frac - final_floor_frac) * (1.0 - stage2_progress)
    return base_weight * stage2_multiplier


def get_linear_lr_scale(it, num_iterations, end_scale=1.0, max_scale=1.0, warmup_iterations=1000, nolearn_iterations=0):
    num_iterations = max(0, num_iterations)
    effective_nolearn_iterations = min(max(0, nolearn_iterations), num_iterations)
    effective_warmup_iterations = min(max(0, warmup_iterations), max(0, num_iterations - effective_nolearn_iterations))
    it = min(max(it, 0), num_iterations)

    if it < effective_nolearn_iterations:
        return 0.0

    warmup_step = it - effective_nolearn_iterations
    if effective_warmup_iterations > 0 and warmup_step < effective_warmup_iterations:
        return max_scale * warmup_step / effective_warmup_iterations

    remaining_iterations = num_iterations - effective_nolearn_iterations - effective_warmup_iterations
    if remaining_iterations <= 0:
        return max_scale

    decay_progress = min(max(warmup_step - effective_warmup_iterations, 0), remaining_iterations) / remaining_iterations
    return max_scale + (end_scale - max_scale) * decay_progress


def get_gate_proj_bias_lr_scale(optimizer, step, num_iterations):
    for group in optimizer.param_groups:
        if group.get("name") == "gate_proj_bias" and group.get("kind") == "adamw":
            return get_linear_lr_scale(
                step,
                num_iterations,
                end_scale=group.get("lr_scale_end", 1.0),
                max_scale=group.get("lr_scale_max", 1.0),
                nolearn_iterations=group.get("lr_scale_nolearn_iterations", 0),
                warmup_iterations=group.get("lr_scale_warmup_iterations", 1000),
            )
    return 1.0

def scalar_loss_to_item(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return float(value)

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

def get_dense_gate_proj_bias_stat_layer_indices(model):
    config = model.config
    if int(getattr(config, 'num_moe_layers', -1)) != 0:
        return []
    stride = max(1, int(getattr(config, 'moe_layer_stride', 1)))
    start_layer = max(0, int(getattr(config, 'moe_start_layer', 0)))
    return [
        layer_idx
        for layer_idx in range(start_layer, len(model.transformer.h))
        if (layer_idx + 1) % stride == 0
    ][:2]

def snapshot_exp_gate_implicit_bias_signs(model, moe_layer_indices):
    sign_snapshots = {}
    with torch.inference_mode():
        for layer_idx in moe_layer_indices:
            layer = model.transformer.h[layer_idx]
            experts = getattr(layer.mlp, 'experts', None)
            if experts is None:
                continue
            router_weight = layer.mlp.router.w_g.weight.float()  # [n_exp, d_model]
            exp_gate_weight = experts.gate_proj.float()  # [n_exp, d_model, intermediate_size]
            normalized_router_weight = torch.nn.functional.normalize(router_weight, dim=1, eps=1e-12)
            normalized_exp_gate_weight = torch.nn.functional.normalize(exp_gate_weight, dim=1, eps=1e-12)
            implicit_bias = (normalized_exp_gate_weight * normalized_router_weight.unsqueeze(2)).sum(dim=1)
            sign_snapshots[layer_idx] = torch.sign(implicit_bias).to(device='cpu', dtype=torch.int8)
    return sign_snapshots

def collect_exp_gate_implicit_bias_flip_rates(model, moe_layer_indices, previous_sign_snapshots, losses):
    current_sign_snapshots = snapshot_exp_gate_implicit_bias_signs(model, moe_layer_indices)
    for layer_idx, current_signs in current_sign_snapshots.items():
        previous_signs = previous_sign_snapshots.get(layer_idx)
        if previous_signs is None or previous_signs.shape != current_signs.shape:
            continue
        losses[f'exp_gate_implicit_bias_flip_rate_{layer_idx}'] = current_signs.ne(previous_signs).float().mean().item()
    return current_sign_snapshots

def collect_weight_grad_stats(model, losses, moe_layer_indices):
    # weight: [n_exp, n_rows, row_dim]
    # returns: [n_exp, n_rows], the ratio of the mean component to the overall norm 
    # for each row. Higher means more of the row is aligned with the mean direction.
    def compute_row_mean_component_ratio(weight):
        weight = weight.float()
        row_dim = weight.shape[2]
        row_means = weight.mean(dim=2)
        row_mean_component_norm = row_means.abs() * (row_dim ** 0.5)
        row_norm = weight.norm(dim=2).clamp_min(1e-12)
        return row_mean_component_norm / row_norm

    router_grad_norms = []
    router_row_norms = []
    router_grad_self_alignments = []
    router_weight_exp_gate_alignments = []
    gate_proj_row_mean_component_ratios = []
    exp_gate_grad_norms = []
    expert_utilities = losses.get('expert_utilities', None)
    selected_scores = losses.get('selected_scores', None)
    moe_layer_to_stats_idx = {layer_idx: stats_idx for stats_idx, layer_idx in enumerate(moe_layer_indices)}

    for i in moe_layer_indices:
        layer = model.transformer.h[i]
        if hasattr(layer.mlp, 'experts'):
            # [n_exp, hidden_size]
            router_gate_grad = layer.mlp.router.w_g.weight.grad
            router_grad_norm = router_gate_grad.norm(dim=1)
            router_grad_norms.append(router_grad_norm)
            losses[f'router_grad_norm_{i}'] = router_grad_norm.mean().item()
            exp_gate_grad = layer.mlp.experts.gate_proj.grad
            exp_gate_grad_norm = None if exp_gate_grad is None else torch.linalg.vector_norm(
                exp_gate_grad,
                dim=tuple(range(1, exp_gate_grad.ndim)),
            )
            if exp_gate_grad_norm is not None:
                exp_gate_grad_norms.append(exp_gate_grad_norm)
                losses[f'exp_gate_grad_norm_{i}'] = exp_gate_grad_norm.mean().item()

            # Compute router grad - router weight alignment.
            # Compute router weight alignment against expert projections.
            with torch.inference_mode():
                router_weight = layer.mlp.router.w_g.weight  # [n_exp, hidden_size]
                router_row_norm = router_weight.norm(dim=1)
                router_row_norms.append(router_row_norm)
                losses[f'router_row_norm_{i}'] = router_row_norm.mean().item()
                experts = layer.mlp.experts
                exp_gate_weight = experts.gate_proj
                gate_proj_row_mean_component_ratio = compute_row_mean_component_ratio(exp_gate_weight).mean(dim=1)
                gate_proj_row_mean_component_ratios.append(gate_proj_row_mean_component_ratio)
                losses[f'gate_proj_row_mean_component_ratio_{i}'] = gate_proj_row_mean_component_ratio.mean().item()
                if experts.use_gate_proj_bias:
                    exp_gate_proj_bias = experts._materialize_gate_proj_bias()
                    losses[f'gate_proj_bias_mean_{i}'] = exp_gate_proj_bias.mean().float().item()
                    losses[f'gate_proj_bias_abs_mean_{i}'] = exp_gate_proj_bias.abs().mean().float().item()
                exp_gate_mean_weight = exp_gate_weight.mean(dim=2)  # [n_exp, hidden_size]
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
                router_weight_exp_gate_alignments.append(rw_ew_alignment)
                mean_rw_ew_alignment = rw_ew_alignment.mean().item()
                losses[f'router_weight_exp_gate_alignment_{i}'] = mean_rw_ew_alignment

                if expert_utilities is not None:
                    # expert_utilities: Tensor of shape (num_moe_layers, n_exp)
                    exp_utilities = expert_utilities[moe_layer_to_stats_idx[i]]  # [n_exp]
                    half_experts = exp_utilities.shape[0] // 2
                    top_indices    = torch.topk(exp_utilities, k=half_experts, largest=True).indices
                    bottom_indices = torch.topk(exp_utilities, k=half_experts, largest=False).indices

                    if experts.use_gate_proj_bias:
                        reduce_dims = tuple(range(1, exp_gate_proj_bias.ndim))
                        exp_gate_proj_bias_mean = exp_gate_proj_bias.float().mean(dim=reduce_dims)
                        exp_gate_proj_bias_abs_mean = exp_gate_proj_bias.abs().float().mean(dim=reduce_dims)
                        losses[f'gate_proj_bias_mean_top_{i}'] = exp_gate_proj_bias_mean[top_indices].mean().item()
                        losses[f'gate_proj_bias_mean_bottom_{i}'] = exp_gate_proj_bias_mean[bottom_indices].mean().item()
                        losses[f'gate_proj_bias_abs_mean_top_{i}'] = exp_gate_proj_bias_abs_mean[top_indices].mean().item()
                        losses[f'gate_proj_bias_abs_mean_bottom_{i}'] = exp_gate_proj_bias_abs_mean[bottom_indices].mean().item()

                    top_rg_rw_alignment    = rg_rw_alignment[top_indices].mean().item()
                    bottom_rg_rw_alignment = rg_rw_alignment[bottom_indices].mean().item()
                    losses[f'router_grad_self_alignment_top_{i}']    = top_rg_rw_alignment
                    losses[f'router_grad_self_alignment_bottom_{i}'] = bottom_rg_rw_alignment

                    top_rw_ew_alignment    = rw_ew_alignment[top_indices].mean().item()
                    bottom_rw_ew_alignment = rw_ew_alignment[bottom_indices].mean().item()
                    losses[f'router_weight_exp_gate_alignment_top_{i}']    = top_rw_ew_alignment
                    losses[f'router_weight_exp_gate_alignment_bottom_{i}'] = bottom_rw_ew_alignment

                    top_router_grad_norm    = router_grad_norm[top_indices].mean().item()
                    bottom_router_grad_norm = router_grad_norm[bottom_indices].mean().item()
                    losses[f'router_grad_norm_top_{i}']    = top_router_grad_norm
                    losses[f'router_grad_norm_bottom_{i}'] = bottom_router_grad_norm

                    top_router_row_norm = router_row_norm[top_indices].mean().item()
                    bottom_router_row_norm = router_row_norm[bottom_indices].mean().item()
                    losses[f'router_row_norm_top_{i}'] = top_router_row_norm
                    losses[f'router_row_norm_bottom_{i}'] = bottom_router_row_norm

                    if selected_scores is not None:
                        # selected_scores: Tensor of shape (num_moe_layers, n_exp)
                        layer_selected_scores = selected_scores[moe_layer_to_stats_idx[i]]  # [n_exp]
                        top_selected_scores    = layer_selected_scores[top_indices].mean().item()
                        bottom_selected_scores = layer_selected_scores[bottom_indices].mean().item()
                        losses[f'selected_scores_top_{i}']    = top_selected_scores
                        losses[f'selected_scores_bottom_{i}'] = bottom_selected_scores

    for i in get_dense_gate_proj_bias_stat_layer_indices(model):
        layer = model.transformer.h[i]
        mlp = getattr(layer, 'mlp', None)
        if hasattr(mlp, 'experts'):
            continue
        gate_proj_weight = getattr(mlp.gate_proj, 'weight', None)
        if gate_proj_weight is not None:
            dense_gate_proj_weight = gate_proj_weight.transpose(0, 1).unsqueeze(0)
            gate_proj_row_mean_component_ratio = compute_row_mean_component_ratio(dense_gate_proj_weight)
            losses[f'gate_proj_row_mean_component_ratio_{i}'] = gate_proj_row_mean_component_ratio.mean().item()
        gate_proj_bias = getattr(mlp, 'gate_proj_bias', None)
        if gate_proj_bias is not None:
            losses[f'gate_proj_bias_mean_{i}'] = gate_proj_bias.mean().float().item()
            losses[f'gate_proj_bias_abs_mean_{i}'] = gate_proj_bias.abs().mean().float().item()

    router_grad_norms = torch.stack(router_grad_norms, dim=0) if router_grad_norms else None
    losses['router_grad_norms'] = router_grad_norms
    router_row_norms = torch.stack(router_row_norms, dim=0) if router_row_norms else None
    losses['router_row_norms'] = router_row_norms
    router_grad_self_alignments = torch.stack(router_grad_self_alignments, dim=0) if router_grad_self_alignments else None
    losses['router_grad_self_alignments'] = router_grad_self_alignments
    router_weight_exp_gate_alignments = torch.stack(router_weight_exp_gate_alignments, dim=0) if router_weight_exp_gate_alignments else None
    losses['router_weight_exp_gate_alignments'] = router_weight_exp_gate_alignments
    gate_proj_row_mean_component_ratios = torch.stack(gate_proj_row_mean_component_ratios, dim=0) if gate_proj_row_mean_component_ratios else None
    losses['gate_proj_row_mean_component_ratios'] = gate_proj_row_mean_component_ratios
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

if args.mockup_mode:
    print0("Mockup mode enabled: skipping training/eval/sample compute and only advancing steps.")

core_results = {}
prev_exp_gate_implicit_bias_signs = {}

signal.signal(signal.SIGTERM, handle_shutdown_signal)
signal.signal(signal.SIGINT, handle_shutdown_signal)

# -----------------------------------------------------------------------------
# Training loop
while True:
    is_last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    should_terminate_after_checkpoint = shutdown_requested and not is_last_step
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
        num_warmup_iterations=args.router_ortho_loss_warmup_iterations,
        num_anneal_iterations=router_ortho_num_anneal_iterations,
        floor_frac=args.router_ortho_loss_floor_frac,
    )
    aux_loss_weight = get_annealed_loss_weight(
        args.aux_loss_weight * args.aux_loss_weight_init_scale,
        step,
        num_anneal_iterations=args.aux_loss_weight_init_anneal_iterations,
        final_weight=args.aux_loss_weight,
    )
    gate_proj_bias_l2_stage1_iterations = args.gate_proj_bias_l2_loss_anneal_iterations
    gate_proj_bias_l2_loss_weight = get_two_stage_annealed_loss_weight(
        args.gate_proj_bias_l2_loss_weight,
        step,
        total_iterations=num_iterations,
        stage1_iterations=gate_proj_bias_l2_stage1_iterations,
        stage1_floor_frac=args.gate_proj_bias_l2_loss_stage1_frac,
        final_floor_frac=args.gate_proj_bias_l2_loss_final_frac,
        nolearn_iterations=args.gate_proj_bias_delay_start_iterations,
    )
    gate_proj_bias_residual_l2_loss_weight = get_two_stage_annealed_loss_weight(
        args.gate_proj_bias_residual_l2_loss_weight,
        step,
        total_iterations=num_iterations,
        stage1_iterations=gate_proj_bias_l2_stage1_iterations,
        stage1_floor_frac=args.gate_proj_bias_l2_loss_stage1_frac,
        final_floor_frac=args.gate_proj_bias_l2_loss_final_frac,
        nolearn_iterations=args.gate_proj_bias_delay_start_iterations,
    )
    gate_proj_bias_shift_abs_mean_loss_weight = (
        gate_proj_bias_l2_loss_weight * args.gate_proj_bias_abs_mean_loss_weight_scale
    )
    router_ortho_blockwise_active = False
    if args.use_router_ortho_blockwise and step >= ROUTER_ORTHO_BLOCKWISE_WARMUP_STEPS:
        router_ortho_blockwise_active = True
    if router_ortho_blockwise_active:
        # router_ortho_gate_random_u is the underlying random sample u in [0, 1),
        # It is the random score that gets compared against on_prob to decide the gate:
        # if u < on_prob, then router_ortho_is_on = 1.0, otherwise router_ortho_is_on = 0.0.
        router_ortho_is_on, router_ortho_gate_random_u, router_ortho_block_idx = get_router_ortho_blockwise_gate(
            step,
            args.router_ortho_block_size,
            args.router_ortho_on_prob,
            args.seed,
        )
    else:
        router_ortho_is_on, router_ortho_gate_random_u, router_ortho_block_idx = 1.0, 1.0, step // args.router_ortho_block_size

    router_ortho_loss_on_scale = 1.0
    # If blockwise router-ortho is active but the gate is only on for a fraction of the blocks, 
    # we scale up the loss when it's on to maintain a stable loss scale and avoid underflow issues.
    if router_ortho_blockwise_active and router_ortho_is_on > 0.0 and args.router_ortho_on_prob < 1.0:
        router_ortho_loss_on_scale = 1.0 / args.router_ortho_on_prob
    router_ortho_effective_loss_weight = router_ortho_loss_weight * router_ortho_is_on * router_ortho_loss_on_scale

    # once in a while: evaluate the val bpb (all ranks participate)
    if (not should_terminate_after_checkpoint) and (not args.mockup_mode) and args.eval_every > 0 and (is_last_step or (step > 0 and step % args.eval_every == 0)):
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
    if should_terminate_after_checkpoint and master_process:
        signal_label = shutdown_signal_name or "shutdown signal"
        print0(f"{signal_label} received; saving checkpoint at step {step:06d} before exit.")

    if should_terminate_after_checkpoint or is_last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        expected_optimizer_ranks = range(ddp_world_size) if args.save_optimizer_state else None
        checkpoint_save_failed = False
        checkpoint_save_error = ""
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
            optimizer.state_dict() if args.save_optimizer_state else None, # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "optimizer_world_size": ddp_world_size if args.save_optimizer_state else 0,
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
                checkpoint_save_failed = True
                checkpoint_save_error = str(exc)
                print0(
                    f"{checkpoint_save_error} Removing checkpoint files for step {step:06d} and continuing training."
                )
        if ddp:
            checkpoint_save_status = torch.tensor(
                [1 if checkpoint_save_failed else 0],
                device=device,
                dtype=torch.int32,
            )
            torch.distributed.broadcast(checkpoint_save_status, src=0)
            checkpoint_save_failed = bool(checkpoint_save_status.item())
        if checkpoint_save_failed:
            delete_checkpoint_step(checkpoint_dir, step)
            if ddp:
                torch.distributed.barrier()
        if delete_old_ckpts_failed:
            if master_process:
                raise ValueError(delete_old_ckpts_error)
            raise RuntimeError(
                f"Checkpoint deletion failed on rank 0 at step {step}. See rank 0 logs for details."
            )
        if should_terminate_after_checkpoint:
            break

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results

    if (not should_terminate_after_checkpoint) and (not args.mockup_mode) and args.core_metric_every > 0 and (is_last_step or (step > 0 and step % args.core_metric_every == 0)):
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
    if (not should_terminate_after_checkpoint) and (not args.mockup_mode) and args.sample_every > 0 and master_process and (is_last_step or (step > 0 and step % args.sample_every == 0)):
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
    if is_last_step:
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
            'router_z_loss': 0.0,
            router_ortho_loss_name: 0.0,
            'router_ortho_loss_gate_proj': 0.0,
            'gate_proj_bias_l2_loss': 0.0,
            'gate_proj_bias_shift_abs_mean': 0.0,
            'gate_proj_bias_shift_abs_mean_loss': 0.0,
            'drop_rate_per_ks': None,
        }
        train_loss_f = 0.0
        dt = 1.0
    else:
        synchronize()
        t0 = time.time()
        step_losses = None
        gate_proj_bias_lr_scale = get_gate_proj_bias_lr_scale(optimizer, step, num_iterations)
        orig_model.set_router_confidence_gate_bias_grad_scale(0.25 * gate_proj_bias_lr_scale)
        orig_model.config.aux_loss_weight = aux_loss_weight
        if model is not orig_model and hasattr(model, "config"):
            model.config.aux_loss_weight = aux_loss_weight
        for micro_step in range(grad_accum_steps):
            MANAGER.collect_backward_stats = (
                MANAGER.collect_load_balancing_stats and micro_step == grad_accum_steps - 1
            )
            with autocast_ctx:
                loss, micro_losses = model(x, y)
            step_losses = accumulate_step_losses(step_losses, micro_losses)
            # Most values in losses are detached and for logging only, but router_ortho_loss is not.
            router_ortho_loss = micro_losses.get("router_ortho_loss_gate_proj")
            if router_ortho_loss is None:
                router_ortho_loss = 0.0
            loss = loss + router_ortho_effective_loss_weight * router_ortho_loss
            gate_proj_bias_l2_loss = micro_losses.get("gate_proj_bias_l2_loss")
            if gate_proj_bias_l2_loss is None:
                gate_proj_bias_l2_loss = 0.0
            loss = loss + gate_proj_bias_l2_loss_weight * gate_proj_bias_l2_loss
            gate_proj_bias_residual_l2_loss = micro_losses.get("gate_proj_bias_residual_l2_loss")
            if gate_proj_bias_residual_l2_loss is None:
                gate_proj_bias_residual_l2_loss = 0.0
            loss = loss + gate_proj_bias_residual_l2_loss_weight * gate_proj_bias_residual_l2_loss
            gate_proj_bias_shift_abs_mean_loss = micro_losses.get("gate_proj_bias_shift_abs_mean_loss")
            if gate_proj_bias_shift_abs_mean_loss is None:
                gate_proj_bias_shift_abs_mean_loss = 0.0
            loss = loss + gate_proj_bias_shift_abs_mean_loss_weight * gate_proj_bias_shift_abs_mean_loss
            
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            loss.backward()
            MANAGER.collect_backward_stats = False
            x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward

        losses = average_step_losses(step_losses, grad_accum_steps)
        losses[router_ortho_loss_name] = losses.get("router_ortho_loss_gate_proj")
        if losses[router_ortho_loss_name] is None:
            losses[router_ortho_loss_name] = 0.0

        if MANAGER.collect_load_balancing_stats:
            collect_weight_grad_stats(model, losses, moe_layer_indices)
        
        # step the optimizer
        lrm = get_lr_multiplier(step, num_iterations, args.warmup_ratio, args.warmdown_ratio, 
                                args.final_lr_frac, lr_scheduler_skip_iters=args.lr_scheduler_skip_iters, 
                                lr_base_scale=args.lr_base_scale)
        muon_momentum = get_muon_momentum(step)
        for group in optimizer.param_groups:
                if group.get("name") in {"gate_proj_bias", "gate_proj_bias_residual"} and group['kind'] == 'adamw':
                    group["lr"] = group.get("base_lr", group["initial_lr"]) * lrm * gate_proj_bias_lr_scale
                else:
                    group["lr"] = group["initial_lr"] * lrm
                if group['kind'] == 'muon':
                    group["momentum"] = muon_momentum
                group["weight_decay"] = get_weight_decay(group["initial_weight_decay"], step, num_iterations)
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
        prev_exp_gate_implicit_bias_signs = collect_exp_gate_implicit_bias_flip_rates(
            orig_model,
            moe_layer_indices,
            prev_exp_gate_implicit_bias_signs,
            losses,
        )
        log_data = {
            "step": step,
            "tokens_seen": tokens_seen,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss_step":              debiased_smooth_loss,
            "train/aux_loss_step":          losses['aux_loss'],
            "train/router_z_loss_step":     losses['router_z_loss'],
            "train/gate_proj_bias_l2_loss_step": losses['gate_proj_bias_l2_loss'],
            "train/gate_proj_bias_residual_l2_loss_step": losses['gate_proj_bias_residual_l2_loss'],
            "train/gate_proj_bias_shift_abs_mean_step": losses['gate_proj_bias_shift_abs_mean'],
            "train/gate_proj_bias_shift_abs_mean_normalized_step": losses['gate_proj_bias_shift_abs_mean_normalized'],
            "train/gate_proj_bias_shift_abs_mean_loss_step": losses['gate_proj_bias_shift_abs_mean_loss'],
            "train/gate_proj_bias_lr_scale": gate_proj_bias_lr_scale,
            "lrm": lrm,
            "dt": dt,
            "tok_per_sec": tok_per_sec,
            "mfu": mfu,
            "epoch": epoch,
        }
        log_data[f"train/{router_ortho_loss_name}_step"] = scalar_loss_to_item(losses[router_ortho_loss_name])
        log_data["train/aux_loss_weight"] = aux_loss_weight
        log_data[f"train/{router_ortho_loss_name}_weight"] = router_ortho_loss_weight
        log_data["train/gate_proj_bias_l2_loss_weight"] = gate_proj_bias_l2_loss_weight
        log_data["train/gate_proj_bias_residual_l2_loss_weight"] = gate_proj_bias_residual_l2_loss_weight
        log_data["train/gate_proj_bias_shift_abs_mean_loss_weight"] = gate_proj_bias_shift_abs_mean_loss_weight
        for sub_loss_name in router_ortho_sub_loss_names:
            sub_loss = losses.get(sub_loss_name)
            if sub_loss is not None:
                log_data[f"train/{sub_loss_name}_step"] = scalar_loss_to_item(sub_loss)
        drop_rates = losses['drop_rate_per_ks']
        if drop_rates is not None:
            if len(drop_rates) >= 1:
                log_data["inspect/drop_rate_0_step"] = drop_rates[0]
            if len(drop_rates) >= 2:
                log_data["inspect/drop_rate_1_step"] = drop_rates[1]
        expert_utilities = losses['expert_utilities']
        moe_layer_to_stats_idx = {layer_idx: stats_idx for stats_idx, layer_idx in enumerate(moe_layer_indices)}
        for i in moe_layer_indices:
            if expert_utilities is not None:
                layer_expert_utilities = expert_utilities[moe_layer_to_stats_idx[i]]
                log_data.update({f"inspect/expert_utility_min_{i}": layer_expert_utilities.min().item()})
                log_data.update({f"inspect/expert_utility_mean_{i}": layer_expert_utilities.mean().item()})
            if f'router_row_norm_{i}' in losses:
                log_data.update({f"inspect/router_row_norm_{i}": losses[f'router_row_norm_{i}']})
            if f'gate_proj_row_mean_component_ratio_{i}' in losses:
                log_data.update({f"inspect/gate_proj_row_mean_component_ratio_{i}": losses[f'gate_proj_row_mean_component_ratio_{i}']})
            if f'gate_proj_bias_mean_{i}' in losses:
                log_data.update({f"inspect/gate_proj_bias_mean_{i}": losses[f'gate_proj_bias_mean_{i}']})
            if f'gate_proj_bias_abs_mean_{i}' in losses:
                log_data.update({f"inspect/gate_proj_bias_abs_mean_{i}": losses[f'gate_proj_bias_abs_mean_{i}']})
            if f'gate_proj_bias_mean_top_{i}' in losses:
                log_data.update({f"inspect/gate_proj_bias_mean_top_{i}": losses[f'gate_proj_bias_mean_top_{i}']})
            if f'gate_proj_bias_mean_bottom_{i}' in losses:
                log_data.update({f"inspect/gate_proj_bias_mean_bottom_{i}": losses[f'gate_proj_bias_mean_bottom_{i}']})
            if f'gate_proj_bias_abs_mean_top_{i}' in losses:
                log_data.update({f"inspect/gate_proj_bias_abs_mean_top_{i}": losses[f'gate_proj_bias_abs_mean_top_{i}']})
            if f'gate_proj_bias_abs_mean_bottom_{i}' in losses:
                log_data.update({f"inspect/gate_proj_bias_abs_mean_bottom_{i}": losses[f'gate_proj_bias_abs_mean_bottom_{i}']})
            if f'gate_proj_bias_shift_abs_mean_{i}' in losses:
                log_data.update({f"inspect/gate_proj_bias_shift_abs_mean_{i}": losses[f'gate_proj_bias_shift_abs_mean_{i}']})
            if f'gate_proj_bias_shift_abs_mean_normalized_{i}' in losses:
                log_data.update({f"inspect/gate_proj_bias_shift_abs_mean_normalized_{i}": losses[f'gate_proj_bias_shift_abs_mean_normalized_{i}']})
            if f'exp_gate_implicit_bias_flip_rate_{i}' in losses:
                log_data.update({f"inspect/exp_gate_implicit_bias_flip_rate_{i}": losses[f'exp_gate_implicit_bias_flip_rate_{i}']})
            if f'mean_abs_gate_{i}' in losses:
                log_data.update({f"inspect/mean_abs_gate_{i}": losses[f'mean_abs_gate_{i}']})
            if f'active_frac_gate_{i}' in losses:
                log_data.update({f"inspect/active_frac_gate_{i}": losses[f'active_frac_gate_{i}']})
            if f'topk_share_gate_{i}' in losses:
                log_data.update({f"inspect/topk_share_gate_{i}": losses[f'topk_share_gate_{i}']})
            if f'entropy_gate_{i}' in losses:
                log_data.update({f"inspect/entropy_gate_{i}": losses[f'entropy_gate_{i}']})
            if f'router_weight_exp_gate_alignment_{i}' in losses:
                log_data.update({f"inspect/router_weight_exp_gate_alignment_{i}": losses[f'router_weight_exp_gate_alignment_{i}']})
            if f'router_grad_norm_top_{i}' in losses:
                log_data.update({f"inspect/router_grad_norm_top_{i}": losses[f'router_grad_norm_top_{i}']})
            if f'router_grad_norm_bottom_{i}' in losses:
                log_data.update({f"inspect/router_grad_norm_bottom_{i}": losses[f'router_grad_norm_bottom_{i}']})
            if f'router_row_norm_top_{i}' in losses:
                log_data.update({f"inspect/router_row_norm_top_{i}": losses[f'router_row_norm_top_{i}']})
            if f'router_row_norm_bottom_{i}' in losses:
                log_data.update({f"inspect/router_row_norm_bottom_{i}": losses[f'router_row_norm_bottom_{i}']})
            if f'router_grad_self_alignment_top_{i}' in losses:
                log_data.update({f"inspect/router_grad_self_alignment_top_{i}": losses[f'router_grad_self_alignment_top_{i}']})
            if f'router_grad_self_alignment_bottom_{i}' in losses:
                log_data.update({f"inspect/router_grad_self_alignment_bottom_{i}": losses[f'router_grad_self_alignment_bottom_{i}']})
            if f'router_weight_exp_gate_alignment_top_{i}' in losses:
                log_data.update({f"inspect/router_weight_exp_gate_alignment_top_{i}": losses[f'router_weight_exp_gate_alignment_top_{i}']})
            if f'router_weight_exp_gate_alignment_bottom_{i}' in losses:
                log_data.update({f"inspect/router_weight_exp_gate_alignment_bottom_{i}": losses[f'router_weight_exp_gate_alignment_bottom_{i}']})
            if f'selected_scores_top_{i}' in losses:
                log_data.update({f"inspect/selected_scores_top_{i}": losses[f'selected_scores_top_{i}']})
            if f'selected_scores_bottom_{i}' in losses:
                log_data.update({f"inspect/selected_scores_bottom_{i}": losses[f'selected_scores_bottom_{i}']})

        for i in get_dense_gate_proj_bias_stat_layer_indices(orig_model):
            if f'gate_proj_row_mean_component_ratio_{i}' in losses:
                log_data.update({f"inspect/gate_proj_row_mean_component_ratio_{i}": losses[f'gate_proj_row_mean_component_ratio_{i}']})
                        
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
        f"Tokens : {target_scaling_params_label} ratio": total_batch_size * num_iterations / target_scaling_params,
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
should_continue_to_chat_sft = args.continue_to_chat_sft and step == num_iterations
chat_sft_master_port = None
if should_continue_to_chat_sft:
    chat_sft_master_port = prepare_chat_sft_rendezvous(ddp, ddp_rank, device)
wandb_run.finish() # wandb run finish
compute_cleanup()

if should_continue_to_chat_sft:
    chat_sft_argv = build_chat_sft_exec_argv(
        sys.executable,
        output_dirname,
        step,
        args.continue_to_chat_sft_args,
    )
    sanitize_chat_sft_rendezvous_env()
    if chat_sft_master_port is not None:
        print0(f"Prepared fresh chat_sft rendezvous port: {chat_sft_master_port}")
    print0(f"Continuing into chat_sft: {shlex.join(chat_sft_argv)}")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp(chat_sft_argv[0], chat_sft_argv)
