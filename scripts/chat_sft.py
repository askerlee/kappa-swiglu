"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16

chat_sft.py inherits a lot of configs from the base model, for example
aux_loss_weight from the checkpoint config. The model returns raw aux_loss,
and this script applies the inherited aux_loss_weight in its training loop.
"""

import argparse
import math
import os
import sys

allocator_conf = os.environ.get(
    "PYTORCH_ALLOC_CONF",
    os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
)
os.environ["PYTORCH_ALLOC_CONF"] = allocator_conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = allocator_conf

from tasks.arc import ARC
import time, re
import wandb
import torch
from contextlib import nullcontext
from nanochat.common import cast_model_parameters, compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.checkpoint_manager import load_model
from nanochat.gpt import get_moe_layer_indices
from nanochat.manager import MANAGER
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval, compute_chatcore_metric, ALL_CHAT_EVAL_TASKS
import torch.distributed as dist

# TaskMixture shuffles the datasets at initialization
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.tulu3 import Tulu3SFTMixture, Tulu3SFTPersonaIF
from tasks.ultradata import UltraDataSFTIF
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

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

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=("float32", "bfloat16"),
                    help="autocast compute dtype")
parser.add_argument("--parameter-dtype", type=str, default="bfloat16", choices=("reference", "float32", "bfloat16"),
                    help="parameter storage: reference keeps token/value embeddings in BF16 on CUDA and other parameters in FP32")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-save-tag", type=str, default=None, help="extra model tag to append to the saved folder")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--loop", dest="total_ut_steps", type=int, default=None, help="override the checkpoint Universal Transformer loop count")
parser.add_argument("--ut-everypass-ntp", dest="ut_everypass_ntp", type=str2bool, nargs='?', const=True, default=False,
                    help="override whether NTP loss is computed at every UT pass or only the final pass")
parser.add_argument("--ut-detach", dest="ut_detach", type=str2bool, nargs='?', const=True, default=False,
                    help="detach the hidden state between UT passes (truncated BPTT); requires --ut-everypass-ntp and cuts cross-pass gradient flow")
parser.add_argument("--loss-recompute-backward", dest="loss_recompute_backward", type=str2bool, nargs='?', const=True, default=True,
                    help="override recomputing lm_head loss chunks during backward to reduce retained vocab-logit memory")
parser.add_argument("--activation-checkpointing", dest="activation_checkpointing", type=str2bool, nargs='?', const=True, default=None,
                    help="override activation checkpointing for each full transformer-stack pass")
parser.add_argument("--activation-offload", dest="activation_offload", type=str2bool, nargs='?', const=True, default=None,
                    help="override storing transformer activations in pinned CPU memory")
parser.add_argument("--use-kappa-swiglu", type=str2bool, nargs='?', const=True, default=None,
                    help="enable kappa SwiGLU when the base model does not already enable it")
parser.add_argument("--constant-kappa-dense-layers", dest="constant_kappa_dense_layers", type=str2bool, nargs='?', const=True, default=None,
                    help="override constant kappa bias for dense layers (default: inherit from base model)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
parser.add_argument("--train-mixture-repeats", type=int, default=4, help="expand the train mixture by N repeats; "
                    "tulu3 is not repeated; procedural tasks use fresh index ranges and SmolTalk grows its slice accordingly (default: 4)")
parser.add_argument("--global-packing-buffer-size", type=int, default=200,
                    help="total best-fit packing lookahead divided across all DDP ranks (default: 200)")
parser.add_argument("--use-tulu3-sft-mixture", type=str2bool, nargs='?', const=True, default=True, help="include allenai/tulu-3-sft-mixture in the SFT train mixture")
parser.add_argument("--tulu3-english-only", type=str2bool, nargs='?', const=True, default=False, help="filter Tulu 3 conversations to English only")
parser.add_argument("--use-ultradata-sft-if", type=str2bool, nargs='?', const=True, default=True, help="include CJK-filtered openbmb/UltraData-SFT-2605 IF/no_think data in the SFT train mixture")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
# Since the SFT dataset is much smaller, we use a smaller learning rate for the residual and input-embedding scalars, 
# but the matrix parameters have the same LR.
parser.add_argument("--scalar-lr", type=float, default=0.05, help="learning rate for x0_lambdas (resid_lambdas use 0.1x)")
parser.add_argument("--matrix-lr", type=float, default=0.01, help="learning rate for matrix parameters")
parser.add_argument("--matrix-optimizer", type=str, default="aurora", choices=["muon", "muonh", "aurora"], help="matrix optimizer for 2D parameters")
parser.add_argument("--lr-base-scale", type=float, default=0.2, help="base scaling factor for all types of learning rates, relative to the LR used during base model pretraining")
parser.add_argument("--warmup-ratio", type=float, default=0.01, help="ratio of SFT progress used for linear LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.2, help="ratio of SFT progress used for linear LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as a fraction of the peak SFT LR")
parser.add_argument("--kappa-lr-max-scale",
                    dest="kappa_lr_max_scale", type=float, default=0.01,
                    help="peak LR scale factor for kappa_bias params after warming from 0 before annealing to --kappa-lr-final-scale")
parser.add_argument("--kappa-lr-final-scale",
                    dest="kappa_lr_final_scale", type=float, default=0.005,
                    help="final LR scale factor for kappa_bias params after warming from 0 to --kappa-lr-max-scale")
parser.add_argument("--kappa-bias-delay-start-min-iterations", "--kappa-bias-delay-start-iterations",
                    dest="kappa_bias_delay_start_min_iterations", type=int, default=100,
                    help="number of initial iterations to keep kappa_bias LR at 0 before warmup and annealing")
parser.add_argument("--kappa-bias-lr-warmup-iterations", type=int, default=100,
                    help="number of iterations to linearly ramp kappa_bias LR scale from 0 to --kappa-lr-max-scale before annealing to --kappa-lr-final-scale")
parser.add_argument(
    "--kappa-l2-loss-weight",
    dest="kappa_l2_loss_weight",
    type=float,
    default=0,
    help="L2 weight on kappa_bias values",
)
parser.add_argument("--kappa-scale-l2-loss-weight-scale", type=float, default=0.2,
                    help="multiplier applied to --kappa-l2-loss-weight when weighting kappa_scale L2 loss")
parser.add_argument("--kappa-params-l2-anchor", type=str, choices=("initial", "zero"), default="initial",
                    help="anchor expert kappa bias and scale L2 either around their loaded initial values or around 0")
parser.add_argument("--muon-match-rms-adamw", type=str2bool, nargs='?', const=True, default=True, help="use Kimi Muon LR scaling: 0.2*sqrt(max(out,in))")
parser.add_argument("--weight-decay", type=float, default=0.005, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--router-z-loss-weight", type=float, default=-1, help="weight for router z loss")
parser.add_argument("--router-wg-delta", action="store_true", help="train a full additive delta for each MoE router")
parser.add_argument("--router-wg-delta-l2-loss-weight", type=float, default=0.001,
                    help="L2 weight on the additive router delta")
parser.add_argument("--use-aux-free-load-balancing", type=str2bool, nargs='?', const=True, default=None, help="enable DeepSeekV3 auxiliary-loss-free load balancing instead of the Switch auxiliary router loss (default: inherit from saved config of base model)")

# Evaluation
parser.add_argument("--eval-every", type=int, default=150, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--chat-eval-task-name", type=str, default=None, help="chat eval task name(s); default = all tasks. Use | to split multiple tasks.")
parser.add_argument("--chat-eval-temperature", type=float, default=0.0, help="temperature for generative chat eval")
parser.add_argument("--chat-eval-max-new-tokens", type=int, default=512, help="max new tokens for generative chat eval")
parser.add_argument("--chat-eval-num-samples", type=int, default=1, help="number of samples for generative chat eval")
parser.add_argument("--chat-eval-top-k", type=int, default=50, help="top-k for generative chat eval")
parser.add_argument("--chat-eval-batch-size", type=int, default=8, help="batch size for categorical chat eval")
parser.add_argument("--chat-eval-max-problems", type=int, default=None, help="max problems per chat eval task")
parser.add_argument("--eval-only", action="store_true", help="load the checkpoint, run evaluation, and exit without training")
# Output
parser.add_argument("--dry-run", action="store_true", help="log to wandb but skip checkpoints/report")
parser.add_argument("--wandb-api-key-file", type=str, default=None, help="Weights & Biases API key file (optional). If provided, sets WANDB_API_KEY for this run")
parser.add_argument("--log-grad-stats", type=str2bool, nargs='?', const=True, default=True, help="log gradient statistics for MoE layers")
parser.add_argument("--log-interval", type=int, default=10, help="interval (in steps) for logging train and grad stats")

args = parser.parse_args()
if args.activation_checkpointing and args.activation_offload:
    raise ValueError("--activation-checkpointing and --activation-offload are mutually exclusive")
if args.activation_checkpointing and args.log_grad_stats:
    print("Disabling --log-grad-stats because it bypasses activation checkpointing on logging steps.")
    args.log_grad_stats = False
if args.ut_detach and not args.ut_everypass_ntp:
    print0("--ut-detach requires --ut-everypass-ntp; disabling --ut-detach because intermediate passes need their own NTP loss.")
    args.ut_detach = False
if args.train_mixture_repeats < 1:
    raise ValueError("--train-mixture-repeats must be >= 1")
if args.global_packing_buffer_size < 1:
    raise ValueError("--global-packing-buffer-size must be >= 1")
if not (0.0 <= args.warmup_ratio <= 1.0):
    raise ValueError("--warmup-ratio must satisfy 0 <= ratio <= 1")
if not (0.0 <= args.warmdown_ratio <= 1.0):
    raise ValueError("--warmdown-ratio must satisfy 0 <= ratio <= 1")
if args.warmup_ratio + args.warmdown_ratio > 1.0:
    raise ValueError("--warmup-ratio + --warmdown-ratio must be <= 1")
if not (0.0 <= args.final_lr_frac <= 1.0):
    raise ValueError("--final-lr-frac must satisfy 0 <= fraction <= 1")
if args.kappa_bias_delay_start_min_iterations < 0:
    raise ValueError("--kappa-bias-delay-start-min-iterations must be >= 0")
if args.kappa_bias_lr_warmup_iterations < 0:
    raise ValueError("--kappa-bias-lr-warmup-iterations must be >= 0")
user_config = vars(args).copy()
matrix_optimizer_was_specified = arg_was_explicitly_set(sys.argv[1:], '--matrix-optimizer')
router_z_loss_weight_was_specified = arg_was_explicitly_set(sys.argv[1:], '--router-z-loss-weight')


def drop_none_log_values(log_data):
    return {key: value for key, value in log_data.items() if value is not None}


def get_interval_throughput(total_batch_size, num_flops_per_token, gpu_peak_flops, ddp_world_size, interval_steps, interval_time):
    if interval_steps <= 0 or interval_time <= 0:
        raise ValueError("Throughput interval must contain at least one step with positive elapsed time")

    average_dt = interval_time / interval_steps
    tok_per_sec = int(total_batch_size * interval_steps / interval_time)
    flops_per_sec = num_flops_per_token * total_batch_size * interval_steps / interval_time
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    return average_dt, tok_per_sec, mfu

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
parameter_dtype = torch.bfloat16 if args.parameter_dtype == 'bfloat16' else torch.float32
embedding_dtype = (
    torch.bfloat16
    if args.parameter_dtype == 'reference' and device_type == 'cuda'
    else parameter_dtype
)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
if args.wandb_api_key_file:
    with open(args.wandb_api_key_file, "r") as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()

use_dummy_wandb = args.model_tag is None or not master_process
ckpt_prefix2 = args.model_tag
if args.model_step != -1:
    mat = re.search(r"(\d+)$", str(args.model_step).rstrip('/'))
    if mat:
        ckpt_prefix2 += f"-{mat.group(1)}"

if args.model_save_tag:
    ckpt_prefix2 = ckpt_prefix2 + '-' + args.model_save_tag

# Load the model and tokenizer
# NOTE: the optim state of the base model is not loaded here.
refresh_kappa_param_references = args.kappa_params_l2_anchor == "initial"
print0(f"expert kappa params L2 anchor: {args.kappa_params_l2_anchor}")
sft_checkpoint_source = "sft" if args.eval_only else "base"
use_kappa_swiglu = True if args.use_kappa_swiglu is None else args.use_kappa_swiglu
model, tokenizer, meta = load_model(
    sft_checkpoint_source,
    device,
    phase="train",
    model_tag=args.model_tag,
    step=args.model_step,
    total_ut_steps=args.total_ut_steps,
    ut_everypass_ntp=args.ut_everypass_ntp,
    ut_detach=args.ut_detach,
    loss_recompute_backward=args.loss_recompute_backward,
    activation_checkpointing=args.activation_checkpointing,
    activation_offload=args.activation_offload,
    use_kappa_swiglu=use_kappa_swiglu,
    constant_kappa_bias_dense_layers=args.constant_kappa_dense_layers,
    refresh_kappa_param_references=refresh_kappa_param_references,
)
checkpoint_used_kappa_swiglu = bool(
    meta.get("model_config", {}).get("use_kappa_swiglu", False)
)
if args.use_kappa_swiglu is True and not checkpoint_used_kappa_swiglu:
    ckpt_prefix2 += "-kappa"

wandb_run_name = ckpt_prefix2 + '-' + time.strftime('%Y-%m-%d %H:%M:%S')
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nano-moe-sft", name=wandb_run_name, config=user_config)
if not use_dummy_wandb:
    wandb.define_metric("step")
    wandb.define_metric("tokens_seen")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("val/*", step_metric="step")
    wandb.define_metric("chat_eval/*", step_metric="step")
    wandb.define_metric("inspect/*", step_metric="step")

args.router_wg_delta = args.router_wg_delta or bool(
    getattr(model.config, "router_wg_delta", False)
)
if args.router_wg_delta:
    model.setup_router_wg_delta()
user_config["router_wg_delta"] = args.router_wg_delta
args.total_ut_steps = model.config.total_ut_steps
args.ut_everypass_ntp = model.config.ut_everypass_ntp
args.ut_detach = model.config.ut_detach
if args.total_ut_steps > 1:
    print0(f"Loops = {args.total_ut_steps}")
loaded_checkpoint_step = int(meta.get("step", 0))
user_config["kappa_l2_loss_weight"] = args.kappa_l2_loss_weight
if not use_dummy_wandb:
    wandb_run.config.update(
        {
            "kappa_l2_loss_weight": args.kappa_l2_loss_weight,
        },
        allow_val_change=True,
    )
    
print0(
    "kappa_scale_l2_loss_weight_scale: "
    f"{args.kappa_scale_l2_loss_weight_scale}"
)

user_config["kappa_scale_l2_loss_weight_scale"] = args.kappa_scale_l2_loss_weight_scale
if not use_dummy_wandb:
    wandb_run.config.update(
        {
            "kappa_scale_l2_loss_weight_scale": args.kappa_scale_l2_loss_weight_scale,
        },
        allow_val_change=True,
    )
if args.use_aux_free_load_balancing is None:
    args.use_aux_free_load_balancing = bool(
        getattr(model.config, "use_aux_free_load_balancing", False)
    )
    print0(
        "Inherited use_aux_free_load_balancing: "
        f"{args.use_aux_free_load_balancing}"
    )
else:
    print0(
        "Specified use_aux_free_load_balancing: "
        f"{args.use_aux_free_load_balancing}"
    )
model.set_aux_free_load_balancing(args.use_aux_free_load_balancing)
user_config["use_aux_free_load_balancing"] = args.use_aux_free_load_balancing
if not use_dummy_wandb:
    wandb_run.config.update(
        {
            "use_aux_free_load_balancing": args.use_aux_free_load_balancing,
        },
        allow_val_change=True,
    )
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and args.device_batch_size > pretrain_batch_size:
    print0(f"FOOTGUN WARNING: base model training used device_batch_size {pretrain_batch_size}, did you pass in a good --device-batch-size to this script?")
if matrix_optimizer_was_specified:
    print0(f"Specified matrix_optimizer: {args.matrix_optimizer}")
else:
    args.matrix_optimizer = meta.get("user_config", {}).get("matrix_optimizer", "muon")
    print0(f"Inherited matrix_optimizer: {args.matrix_optimizer}")
if args.matrix_optimizer == "muonh":
    args.router_wg_delta_l2_loss_weight = 0.0
user_config["matrix_optimizer"] = args.matrix_optimizer
user_config["router_wg_delta_l2_loss_weight"] = args.router_wg_delta_l2_loss_weight
if not use_dummy_wandb:
    wandb_run.config.update(
        {
            "matrix_optimizer": args.matrix_optimizer,
            "router_wg_delta": args.router_wg_delta,
            "router_wg_delta_l2_loss_weight": args.router_wg_delta_l2_loss_weight,
        },
        allow_val_change=True,
    )
if router_z_loss_weight_was_specified:
    model.config.router_z_loss_weight = args.router_z_loss_weight
    print0(f"Specified router_z_loss_weight: {args.router_z_loss_weight}")
else:
    args.router_z_loss_weight = model.config.router_z_loss_weight
    print0(f"Inherited router_z_loss_weight: {args.router_z_loss_weight}")
user_config["router_z_loss_weight"] = args.router_z_loss_weight
if not use_dummy_wandb:
    wandb_run.config.update({"router_z_loss_weight": args.router_z_loss_weight}, allow_val_change=True)
aux_loss_weight = float(getattr(model.config, "aux_loss_weight", 0.0))
print0(f"Inherited aux_loss_weight: {aux_loss_weight}")
user_config["aux_loss_weight"] = aux_loss_weight
if not use_dummy_wandb:
    wandb_run.config.update({"aux_loss_weight": aux_loss_weight}, allow_val_change=True)
kappa_scale_l2_loss_weight = args.kappa_l2_loss_weight * args.kappa_scale_l2_loss_weight_scale

# If the model has not initialized kappa swiglu params, this has no effect.
# Therefore, if the model is trained without enabling kappa swiglu, this call has no effect.
model.set_kappa_swiglu_enabled(True)
cast_model_parameters(model, parameter_dtype, embedding_dtype=embedding_dtype)
print0(
    f"Compute dtype: {ptdtype}; parameter storage mode: {args.parameter_dtype} "
    f"(default={parameter_dtype}, embeddings={embedding_dtype})"
)
orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
moe_layer_indices = get_moe_layer_indices(model.config)
    
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
total_batch_size = args.total_batch_size
world_tokens_per_row = args.max_seq_len * ddp_world_size
if total_batch_size % world_tokens_per_row != 0:
    raise ValueError(
        "total_batch_size must be a multiple of --max-seq-len * DDP world size "
        "so the partial final micro-batch retains the same integer number of rows on every rank. "
        f"Got total_batch_size={total_batch_size:,}, row granularity={world_tokens_per_row:,}."
    )
grad_accum_steps = math.ceil(total_batch_size / world_tokens_per_fwdbwd)
last_device_batch_size = (
    total_batch_size - (grad_accum_steps - 1) * world_tokens_per_fwdbwd
) // world_tokens_per_row
last_micro_weight = last_device_batch_size / args.device_batch_size
grad_accum_normalizer = (grad_accum_steps - 1) + last_micro_weight
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(
    f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps} "
    f"(last device batch size: {last_device_batch_size})"
)
token_bytes = get_token_bytes(device=device)

# Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2
# (or equivalently, \propto 1/depth^2 due to constant aspect ratio)
weight_decay_scaled = args.weight_decay * (12 / depth)**2
if depth != 12:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {depth}")

# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# After setup_optimizer(), one shouldn't change grad scale settings.
optimizer = None
if not args.eval_only:
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        scalar_lr=args.scalar_lr,
        matrix_lr=args.matrix_lr,
        matrix_optimizer=args.matrix_optimizer,
        weight_decay=weight_decay_scaled,
        muon_match_rms_adamw=args.muon_match_rms_adamw,
        kappa_lr_final_scale=args.kappa_lr_final_scale,
        kappa_lr_max_scale=args.kappa_lr_max_scale,
        kappa_bias_delay_start_iterations=args.kappa_bias_delay_start_min_iterations,
        kappa_bias_lr_warmup_iterations=args.kappa_bias_lr_warmup_iterations,
    )
    # Override the initial learning rate as a fraction of the base learning rate
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

# SFT data mixture and DataLoader
base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
smoltalk_rows_per_repeat = 50000
simple_spelling_rows_per_repeat = 200000
spellingbee_rows_per_repeat = 80000

# TaskMixture shuffles the datasets at initialization.
train_tasks = [
    SmolTalk(split="train", stop=smoltalk_rows_per_repeat * args.train_mixture_repeats), # grow the capped SmolTalk slice instead of replaying the same subset
    MMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems drawn from ARC, MC_TEST, OBQA, RACE
    ARC(subset="ARC-Easy", split="train"),
    ARC(subset="ARC-Challenge", split="train"),
    GSM8K(subset="main", split="train"), # 8K rows teaching simple math and (calculator) tool use
    GSM8K(subset="main", split="train"), # 2 epochs of GSM8K
    CustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
    CustomJSON(filepath=identity_conversations_filepath), # let's do 2 epochs of these
]
if args.use_tulu3_sft_mixture:
    # allenai/tulu-3-sft-mixture: 939,344 samples
    train_tasks.append(Tulu3SFTMixture(split="train", english_only=args.tulu3_english_only))
    # Adds a second exposure to the 30k focused Persona-IF examples.
    train_tasks.append(Tulu3SFTPersonaIF(split="train", english_only=args.tulu3_english_only))
if args.use_ultradata_sft_if:
    # openbmb/UltraData-SFT-2605 IF/no_think: 199,991 samples before CJK filtering
    train_tasks.append(UltraDataSFTIF())

for repeat_idx in range(args.train_mixture_repeats):
    simple_spelling_start = repeat_idx * simple_spelling_rows_per_repeat
    spellingbee_start = repeat_idx * spellingbee_rows_per_repeat
    train_tasks.extend([
        SimpleSpelling(
            size=simple_spelling_start + simple_spelling_rows_per_repeat,
            split="train",
            start=simple_spelling_start,
            stop=simple_spelling_start + simple_spelling_rows_per_repeat,
        ), # use a fresh procedural slice each repeat instead of duplicating the same examples
        SpellingBee(
            size=spellingbee_start + spellingbee_rows_per_repeat,
            split="train",
            response_style="mixed",
            start=spellingbee_start,
            stop=spellingbee_start + spellingbee_rows_per_repeat,
        ), # mix direct answers with tool-verified ones over fresh seeds each repeat
        SpellingBee(
            size=spellingbee_start + spellingbee_rows_per_repeat,
            split="train",
            response_style="direct",
            start=spellingbee_start,
            stop=spellingbee_start + spellingbee_rows_per_repeat,
        ), # extra direct-answer supervision for spelling/counting over fresh seeds each repeat
    ])
train_dataset = TaskMixture(train_tasks)
val_dataset = TaskMixture([
    SmolTalk(split="test"), # 24K rows in test set
    MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
    GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
]) # total: 24K + 14K + 1.32K ~= 39K rows
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = args.eval_only # eval-only goes straight to the existing final evaluation path
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
current_epoch = 1 # track epoch for logging
train_seen_conversations = 0 # consumed + skipped conversations in train split
train_skipped_conversations = 0 # conversations skipped for being malformed, exceeding row_capacity, or lacking supervised targets

def get_rank_packing_buffer_size(global_buffer_size, world_size, rank):
    if global_buffer_size < world_size:
        raise ValueError("global packing buffer size must be at least the DDP world size")
    buffer_size, remainder = divmod(global_buffer_size, world_size)
    return buffer_size + int(rank < remainder)

rank_packing_buffer_size = get_rank_packing_buffer_size(
    args.global_packing_buffer_size,
    ddp_world_size,
    ddp_rank,
)
print0(
    f"Global packing buffer: {args.global_packing_buffer_size} conversations "
    f"({rank_packing_buffer_size} on rank 0)"
)

def sft_data_generator_bos_bestfit(split, buffer_size=rank_packing_buffer_size):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Targets are supervised only on assistant tokens. Padding positions are masked
    with -1 (ignore_index for cross-entropy).
    """
    global last_step, approx_progress, current_epoch, train_seen_conversations, train_skipped_conversations
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: list of (token_ids, supervision_mask) tuples
    conv_buffer = []
    cursor = ddp_rank  # Each rank processes different conversations (for fetching)
    consumed = ddp_rank  # Track actual consumption separately from buffering
    skipped_conversations = 0
    epoch = 1
    it = 0  # iteration counter

    def refill_buffer():
        nonlocal cursor, epoch, skipped_conversations
        while len(conv_buffer) < buffer_size:
            try:
                conversation = dataset[cursor]
                ids, mask = tokenizer.render_conversation(conversation, max_tokens=None)
                # NOTE: in the call above, max_tokens=None, this means:
                # Full render, then fit-check, instead of truncating to fit.
                has_supervised_targets = any(mask[1:]) if len(mask) > 1 else False
                if len(ids) <= row_capacity and has_supervised_targets:
                    conv_buffer.append((ids, mask))
                else:
                    skipped_conversations += ddp_world_size
            except (AssertionError, KeyError, TypeError, ValueError):
                skipped_conversations += ddp_world_size
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
                # Note: last_step is now triggered based on consumption, not fetching

    while True:
        rows = []
        row_masks = []
        row_valid_masks = []
        for _ in range(args.device_batch_size):
            row = []
            row_mask = []
            row_valid_mask = []
            while len(row) < row_capacity:
                # Ensure buffer has conversations
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    row_mask.extend(conv_mask)
                    row_valid_mask.extend([1] * len(conv))
                    consumed += ddp_world_size  # Track actual consumption
                else:
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    # Pad with BOS tokens
                    # NOTE: If not masked with row_valid_mask, the padded BOS tokens will 
                    # hurt routing, by hogging on a particular expert. 
                    # So we mask them out in row_valid_mask, and they do not take up any routing capacity.
                    row.extend([bos_token] * remaining)  
                    row_mask.extend([0] * remaining)
                    row_valid_mask.extend([0] * remaining)
                    break  # Row is now full (with padding)

            rows.append(row[:row_capacity])
            row_masks.append(row_mask[:row_capacity])
            row_valid_masks.append(row_valid_mask[:row_capacity])

        # Stopping condition to respect num_iterations, if given
        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        if split == "train":
            current_epoch = epoch
            train_seen_conversations = consumed + skipped_conversations
            train_skipped_conversations = skipped_conversations
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = (consumed + skipped_conversations) / dataset_size
            # Trigger last_step when we've consumed enough (instead of when cursor wraps)
            if consumed + skipped_conversations >= dataset_size:
                last_step = True

        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        mask_tensor = torch.tensor(row_masks, dtype=torch.bool, pin_memory=use_cuda)
        valid_mask_tensor = torch.tensor(row_valid_masks, dtype=torch.bool, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        valid_token_mask = valid_mask_tensor[:, :-1].to(device=device, dtype=torch.bool, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)
        target_mask = mask_tensor[:, 1:].to(device=device, dtype=torch.bool, non_blocking=use_cuda)

        # Skip fully unsupervised batches so training/eval never runs only on padding/user tokens.
        if not target_mask.any():
            continue

        # Supervise only assistant tokens; user, BOS, and padding tokens are ignored.
        targets[~target_mask] = -1

        yield inputs, targets, valid_token_mask

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0 # will go from 0 to 1 over the course of the epoch
chat_eval_task_names = ALL_CHAT_EVAL_TASKS if args.chat_eval_task_name is None else args.chat_eval_task_name.split('|')

# Learning rate scheduler
def get_lr_multiplier(progress, lr_base_scale=1.0, warmup_ratio=0.01,
                      warmdown_ratio=0.2, final_lr_frac=0.05):
    progress = min(max(progress, 0.0), 1.0)
    if warmup_ratio > 0.0 and progress < warmup_ratio:
        return lr_base_scale * progress / warmup_ratio
    if warmdown_ratio > 0.0 and progress > 1.0 - warmdown_ratio:
        decay_progress = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
        return lr_base_scale * (1.0 + decay_progress * (final_lr_frac - 1.0))
    return lr_base_scale

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

def get_kappa_bias_schedule_total_iterations(step, progress):
    if args.num_iterations > 0:
        return args.num_iterations
    if progress > 0.0:
        return max(step + 1, math.ceil((step + 1) / progress))
    return step + 1


def get_kappa_bias_lr_scale(optimizer, step, num_iterations):
    for group in optimizer.param_groups:
        if group.get("name") == "kappa_params" and group.get("kind") == "adamw":
            return get_linear_lr_scale(
                step,
                num_iterations,
                end_scale=group.get("lr_scale_end", 1.0),
                max_scale=group.get("lr_scale_max", 1.0),
                nolearn_iterations=group.get("lr_scale_nolearn_iterations", 0),
                warmup_iterations=group.get("lr_scale_warmup_iterations", 1000),
            )
    return 1.0

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
def get_weight_decay(progress, weight_decay_scaled):
    progress = min(max(progress, 0.0), 1.0)
    return weight_decay_scaled * (1 - progress)

def scalar_loss_to_item(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return float(value)

def gradient_correlation(first_grad, second_grad):
    if first_grad is None or second_grad is None or first_grad.shape != second_grad.shape:
        return None
    first_centered = first_grad.detach().reshape(-1).float()
    second_centered = second_grad.detach().reshape(-1).float()
    if first_centered.numel() < 2:
        return None
    first_centered = first_centered - first_centered.mean()
    second_centered = second_centered - second_centered.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if denominator.item() == 0 or not denominator.isfinite().item():
        return None
    correlation = first_centered.dot(second_centered) / denominator
    return correlation.item() if correlation.isfinite().item() else None

def collect_weight_grad_stats(model, losses, moe_layer_indices):
    def mean_by_sign(values, reduce_dims, sign):
        values = values.float()
        if sign == 'positive':
            mask = values > 0
        elif sign == 'negative':
            mask = values < 0
        else:
            raise ValueError(f"Unsupported sign selector: {sign}")
        counts = mask.sum(dim=reduce_dims)
        sums = values.masked_fill(~mask, 0).sum(dim=reduce_dims)
        means = sums / counts.clamp_min(1)
        return means.masked_fill(counts == 0, float('nan'))

    def finite_mean_item(values):
        finite_values = values[torch.isfinite(values)]
        if finite_values.numel() == 0:
            return None
        return finite_values.mean().item()

    router_grad_norms = []
    router_row_norms = []
    router_grad_self_alignments = []
    router_weight_exp_gate_alignments = []
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

            if layer.mlp.experts.use_kappa_scale:
                kappa_scale = layer.mlp.experts._get_kappa_scale_parameter()
                kappa_bias = layer.mlp.experts._get_kappa_bias_parameter()
                kappa_grad_correlation = gradient_correlation(
                    None if kappa_scale is None else kappa_scale.grad,
                    None if kappa_bias is None else kappa_bias.grad,
                )
                if kappa_grad_correlation is not None:
                    losses[f'kappa_grad_correlation_{i}'] = kappa_grad_correlation

            # Compute router grad - router weight alignment.
            # Compute router weight alignment against expert projections.
            with torch.inference_mode():
                router_weight = layer.mlp.router.effective_w_g_weight()  # [n_exp, hidden_size]
                router_row_norm = router_weight.norm(dim=1)
                router_row_norms.append(router_row_norm)
                losses[f'router_row_norm_{i}'] = router_row_norm.mean().item()
                exp_gate_weight = layer.mlp.experts.gate_proj
                if layer.mlp.experts.use_kappa_swiglu:
                    exp_kappa_bias = layer.mlp.experts._materialize_kappa_bias()
                    losses[f'kappa_bias_mean_{i}'] = exp_kappa_bias.mean().float().item()
                    losses[f'kappa_bias_abs_mean_{i}'] = exp_kappa_bias.abs().mean().float().item()
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

                    if layer.mlp.experts.use_kappa_swiglu:
                        reduce_dims = tuple(range(1, exp_kappa_bias.ndim))
                        exp_kappa_bias_mean = exp_kappa_bias.float().mean(dim=reduce_dims)
                        exp_kappa_bias_abs_mean = exp_kappa_bias.abs().float().mean(dim=reduce_dims)
                        exp_kappa_bias_positive_mean = mean_by_sign(exp_kappa_bias, reduce_dims, sign='positive')
                        exp_kappa_bias_negative_mean = mean_by_sign(exp_kappa_bias, reduce_dims, sign='negative')
                        losses[f'kappa_bias_mean_top_{i}'] = exp_kappa_bias_mean[top_indices].mean().item()
                        losses[f'kappa_bias_mean_bottom_{i}'] = exp_kappa_bias_mean[bottom_indices].mean().item()
                        losses[f'kappa_bias_abs_mean_top_{i}'] = exp_kappa_bias_abs_mean[top_indices].mean().item()
                        losses[f'kappa_bias_abs_mean_bottom_{i}'] = exp_kappa_bias_abs_mean[bottom_indices].mean().item()
                        losses[f'kappa_bias_positive_mean_top_{i}'] = finite_mean_item(exp_kappa_bias_positive_mean[top_indices])
                        losses[f'kappa_bias_positive_mean_bottom_{i}'] = finite_mean_item(exp_kappa_bias_positive_mean[bottom_indices])
                        losses[f'kappa_bias_negative_mean_top_{i}'] = finite_mean_item(exp_kappa_bias_negative_mean[top_indices])
                        losses[f'kappa_bias_negative_mean_bottom_{i}'] = finite_mean_item(exp_kappa_bias_negative_mean[bottom_indices])

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

    router_grad_norms = torch.stack(router_grad_norms, dim=0).detach().cpu() if router_grad_norms else None
    losses['router_grad_norms'] = router_grad_norms
    router_row_norms = torch.stack(router_row_norms, dim=0).detach().cpu() if router_row_norms else None
    losses['router_row_norms'] = router_row_norms
    router_grad_self_alignments = torch.stack(router_grad_self_alignments, dim=0).detach().cpu() if router_grad_self_alignments else None
    losses['router_grad_self_alignments'] = router_grad_self_alignments
    router_weight_exp_gate_alignments = torch.stack(router_weight_exp_gate_alignments, dim=0).detach().cpu() if router_weight_exp_gate_alignments else None
    losses['router_weight_exp_gate_alignments'] = router_weight_exp_gate_alignments
    exp_gate_grad_norms = torch.stack(exp_gate_grad_norms, dim=0).detach().cpu() if exp_gate_grad_norms else None
    losses['exp_gate_grad_norms'] = exp_gate_grad_norms

# -----------------------------------------------------------------------------
# Training loop
x, y, valid_token_mask = (
    (None, None, None) if args.eval_only else next(train_loader)
) # skip train prefetch when evaluating only
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
latest_chat_eval_results = None
latest_chat_eval_step = None
throughput_interval_steps = 0
throughput_interval_time = 0.0
step = 0
while True:
    if args.eval_only and step == 0 and master_process:
        print0("Running in eval-only mode; skipping training and checkpoint save.")
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    kappa_bias_schedule_total_iterations = get_kappa_bias_schedule_total_iterations(
        step,
        max(progress, approx_progress),
    )
    orig_model.set_kappa_bias_ema_rms_reg_total_iterations(kappa_bias_schedule_total_iterations)
    orig_model.set_training_step(loaded_checkpoint_step + step)
    orig_model.set_kappa_bias_ema_rms_reg_step(step)
    moe_kappa_slope_max_scale = getattr(orig_model.config, "moe_kappa_slope_max_scale", 3.0)
    dense_kappa_slope_max_scale = getattr(orig_model.config, "dense_kappa_slope_max_scale", 2.0)
    orig_model.set_kappa_slope_max_scales(
        moe_kappa_slope_max_scale=moe_kappa_slope_max_scale,
        dense_kappa_slope_max_scale=dense_kappa_slope_max_scale,
    )

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or (args.eval_every > 0 and step > 0 and step % args.eval_every == 0):
        model.eval()
        orig_model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb, ntp_loss = evaluate_bpb(orig_model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(drop_none_log_values({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        }), step=step)
        model.train()
        orig_model.train()
        MANAGER.reset_all()

    # save checkpoint at the end of the run before the expensive final chat eval
    if master_process and last_step and not args.dry_run and not args.eval_only:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # e.g. d12
        if args.model_save_tag:
            output_dirname += f"-{args.model_save_tag}"
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        model_config_kwargs = orig_model.config.__dict__.copy()

        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            # No need to save the optimizer stats, as currently our chat sft models are used one-off.
            None, # optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
            }
        )

    if last_step:
        model.eval()
        orig_model.eval()
        engine = Engine(orig_model, tokenizer)
        chat_eval_step = loaded_checkpoint_step if args.eval_only else step
        chat_eval_results = {}
        with autocast_ctx:
            for task_name in chat_eval_task_names:
                acc = run_chat_eval(
                    task_name,
                    orig_model,
                    tokenizer,
                    engine,
                    batch_size=args.chat_eval_batch_size,
                    num_samples=args.chat_eval_num_samples,
                    max_new_tokens=args.chat_eval_max_new_tokens,
                    temperature=args.chat_eval_temperature,
                    top_k=args.chat_eval_top_k,
                    max_problems=args.chat_eval_max_problems,
                )
                chat_eval_results[task_name] = acc
                print0(f"{task_name} accuracy: {100 * acc:.2f}%")
        chatcore_metric_dict = compute_chatcore_metric(chat_eval_results)
        latest_chat_eval_results = dict(chat_eval_results)
        latest_chat_eval_results.update(chatcore_metric_dict)
        latest_chat_eval_step = chat_eval_step
        wandb_log_data = {
            "step": chat_eval_step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
        }
        for task_name, acc in chat_eval_results.items():
            wandb_log_data[f"chat_eval/{task_name}"] = acc
        for metric_name, metric_value in chatcore_metric_dict.items():
            print0(f"{metric_name}: {metric_value:.4f}")
        if "ChatCORE metric" in chatcore_metric_dict:
            wandb_log_data["chat_eval/ChatCORE"] = chatcore_metric_dict["ChatCORE metric"]
        if "ChatCORE metric (without SpellingBee)" in chatcore_metric_dict:
            wandb_log_data["chat_eval/ChatCORE_without_SpellingBee"] = chatcore_metric_dict["ChatCORE metric (without SpellingBee)"]
        wandb_run.log(drop_none_log_values(wandb_log_data), step=chat_eval_step)
        if ddp_rank == 0:
            model_slug = ckpt_prefix2 + f"_chat_{chat_eval_step:06d}"
            output_csv_path = os.path.join(base_dir, "chat_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}\n")
                for task_name, acc in chat_eval_results.items():
                    f.write(f"{task_name:<35}, {acc:<10.6f}\n")
                for metric_name, metric_value in chatcore_metric_dict.items():
                    f.write(f"{metric_name:<35}, {metric_value:<10.6f}\n")
            if not use_dummy_wandb:
                artifact = wandb.Artifact(
                    name=f"{model_slug}-chat-eval",
                    type="chat-eval-results",
                    metadata={
                        "model_slug": model_slug,
                        "step": chat_eval_step,
                        "task_names": list(chat_eval_results.keys()),
                        "max_problems": args.chat_eval_max_problems,
                        "batch_size": args.chat_eval_batch_size,
                        "num_samples": args.chat_eval_num_samples,
                        "max_new_tokens": args.chat_eval_max_new_tokens,
                        "temperature": args.chat_eval_temperature,
                        "top_k": args.chat_eval_top_k,
                        **chatcore_metric_dict,
                    },
                )
                artifact.add_file(output_csv_path, name=os.path.basename(output_csv_path))
                wandb_run.log_artifact(artifact)
            print0(f"\nChat eval results written to: {output_csv_path}")
        model.train()
        orig_model.train()

    if last_step:
        break

    should_log_this_step = ((step + 1) % args.log_interval == 0)
    MANAGER.collect_load_balancing_stats = args.log_grad_stats and should_log_this_step
    MANAGER.collect_backward_stats = MANAGER.collect_load_balancing_stats

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    kappa_bias_lr_scale = get_kappa_bias_lr_scale(
        optimizer,
        step,
        kappa_bias_schedule_total_iterations,
    )
    orig_model.set_router_confidence_gate_bias_grad_scale(0.25 * kappa_bias_lr_scale)
    step_train_loss = 0.0
    step_sft_padding_tokens = 0
    step_sft_token_positions = 0
    for micro_step in range(grad_accum_steps):
        micro_weight = last_micro_weight if micro_step == grad_accum_steps - 1 else 1.0
        micro_x = x[:last_device_batch_size] if micro_step == grad_accum_steps - 1 else x
        micro_y = y[:last_device_batch_size] if micro_step == grad_accum_steps - 1 else y
        micro_valid_token_mask = (
            valid_token_mask[:last_device_batch_size]
            if micro_step == grad_accum_steps - 1
            else valid_token_mask
        )
        step_sft_padding_tokens = (
            step_sft_padding_tokens + (~micro_valid_token_mask).sum()
        )
        step_sft_token_positions += micro_valid_token_mask.numel()
        with autocast_ctx:
            loss, losses = model(
                micro_x, micro_y, valid_token_mask=micro_valid_token_mask
            )
        step_train_loss = step_train_loss + losses['ntp_loss'].detach() * micro_weight
        aux_loss = losses.get("aux_loss")
        if aux_loss is None:
            aux_loss = 0.0
        loss = loss + aux_loss_weight * aux_loss
        if args.router_wg_delta:
            router_wg_delta_l2_loss = losses["router_wg_delta_l2_loss"]
            loss = loss + args.router_wg_delta_l2_loss_weight * router_wg_delta_l2_loss
        kappa_bias_l2_loss = losses.get("kappa_bias_l2_loss")
        if kappa_bias_l2_loss is None:
            kappa_bias_l2_loss = 0.0
        kappa_scale_l2_loss = losses.get("kappa_scale_l2_loss")
        if kappa_scale_l2_loss is None:
            kappa_scale_l2_loss = 0.0
        loss = loss + args.kappa_l2_loss_weight * kappa_bias_l2_loss
        loss = loss + kappa_scale_l2_loss_weight * kappa_scale_l2_loss

        loss = loss * micro_weight / grad_accum_normalizer # normalize by retained sequence rows
        loss.backward()
        x, y, valid_token_mask = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        progress = max(progress, approx_progress) # only increase progress monotonically

    train_loss = step_train_loss / grad_accum_normalizer

    if MANAGER.collect_load_balancing_stats:
        collect_weight_grad_stats(model, losses, moe_layer_indices)

    # step the optimizer
    lrm = get_lr_multiplier(
        progress,
        args.lr_base_scale,
        args.warmup_ratio,
        args.warmdown_ratio,
        args.final_lr_frac,
    )
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress, weight_decay_scaled)
    for group in optimizer.param_groups:
        if group.get("name") == "kappa_params" and group.get("kind") == "adamw":
            group["lr"] = group.get("base_lr", group["initial_lr"]) * lrm * kappa_bias_lr_scale
        else:
            group["lr"] = group["initial_lr"] * lrm
        if group['kind'] in ('muon', 'muonh'):
            group["momentum"] = muon_momentum
        if group['kind'] == 'muon':
            group["weight_decay"] = muon_weight_decay
    orig_model.update_aux_free_load_balancing()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    discard_fraction = train_skipped_conversations / max(train_seen_conversations, 1)
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    throughput_interval_steps += 1
    throughput_interval_time += dt
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(
        f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
        f"dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | "
        f"discarded: {train_skipped_conversations}/{train_seen_conversations} ({100 * discard_fraction:.2f}%) | "
        f"total time: {total_training_time/60:.2f}m"
    )
    if step % args.log_interval == 0:
        logged_dt, logged_tok_per_sec, logged_mfu = get_interval_throughput(
            args.total_batch_size,
            num_flops_per_token,
            989e12,
            ddp_world_size,
            throughput_interval_steps,
            throughput_interval_time,
        )
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/aux_loss_step":          losses['aux_loss'],
            "train/router_z_loss_step":     losses['router_z_loss'],
            "train/router_wg_delta_l2_loss_step": scalar_loss_to_item(losses['router_wg_delta_l2_loss']),
            "train/kappa_bias_l2_loss_step": scalar_loss_to_item(losses['kappa_bias_l2_loss']),
            "train/kappa_scale_l2_loss_step": scalar_loss_to_item(losses['kappa_scale_l2_loss']),
            "train/kappa_slope_scale_abs_mean_step": scalar_loss_to_item(losses['kappa_slope_scale_abs_mean'].mean()),
            "train/kappa_slope_scale_abs_top5p_mean_step": scalar_loss_to_item(losses['kappa_slope_scale_abs_top5p_mean'].mean()),
            "train/kappa_slope_scale_abs_bottom5p_mean_step": scalar_loss_to_item(losses['kappa_slope_scale_abs_bottom5p_mean'].mean()),
            "train/kappa_slope_scale_abs_mean_normalized_step": scalar_loss_to_item(losses['kappa_slope_scale_abs_mean_normalized'].mean()),
            "train/aux_loss_weight": aux_loss_weight,
            "train/router_wg_delta_l2_loss_weight": args.router_wg_delta_l2_loss_weight,
            "train/kappa_bias_l2_loss_weight": args.kappa_l2_loss_weight,
            "train/kappa_scale_l2_loss_weight": kappa_scale_l2_loss_weight,
            "train/kappa_bias_lr_scale": kappa_bias_lr_scale,
            "train/lrm": lrm,
            "train/dt": logged_dt,
            "train/tok_per_sec": logged_tok_per_sec,
            "train/mfu": logged_mfu,
            "train/epoch": current_epoch,
            "train/seen_conversations": train_seen_conversations,
            "train/skipped_conversations": train_skipped_conversations,
            "train/skipped_fraction": discard_fraction,
            "train/sft_padding_fraction": scalar_loss_to_item(
                step_sft_padding_tokens / step_sft_token_positions
            ),
        }
        drop_rates = losses['drop_rate_per_ks']
        if drop_rates is not None:
            if drop_rates.shape[1] >= 1:
                log_data["inspect/drop_rate_0_step"] = drop_rates[:, 0].mean()
            if drop_rates.shape[1] >= 2:
                log_data["inspect/drop_rate_1_step"] = drop_rates[:, 1].mean()
                
            for stats_idx, layer_idx in enumerate(moe_layer_indices):
                if stats_idx >= drop_rates.shape[0] or drop_rates.shape[1] < 2:
                    continue
                log_data[f"inspect/drop_rate_1_step_{layer_idx}"] = scalar_loss_to_item(
                    drop_rates[stats_idx, 1]
                )        
        no_expert_rates = losses['no_expert_rates']
        if no_expert_rates is not None:
            log_data["inspect/no_expert_rate_step"] = no_expert_rates.mean()
            for stats_idx, layer_idx in enumerate(moe_layer_indices):
                if stats_idx >= no_expert_rates.shape[0]:
                    continue
                log_data[f"inspect/no_expert_rate_step_{layer_idx}"] = scalar_loss_to_item(
                    no_expert_rates[stats_idx]
                )
        expert_utilities = losses['expert_utilities']
        moe_layer_to_stats_idx = {layer_idx: stats_idx for stats_idx, layer_idx in enumerate(moe_layer_indices)}
        for i in moe_layer_indices:
            if expert_utilities is not None:
                layer_expert_utilities = expert_utilities[moe_layer_to_stats_idx[i]]
                log_data[f"inspect/expert_utility_min_{i}"] = layer_expert_utilities.min().item()
                log_data[f"inspect/expert_utility_mean_{i}"] = layer_expert_utilities.mean().item()
            if f'router_row_norm_{i}' in losses:
                log_data[f"inspect/router_row_norm_{i}"] = losses[f'router_row_norm_{i}']
            if f'kappa_bias_mean_{i}' in losses:
                log_data[f"inspect/kappa_bias_mean_{i}"] = losses[f'kappa_bias_mean_{i}']
            if f'kappa_bias_abs_mean_{i}' in losses:
                log_data[f"inspect/kappa_bias_abs_mean_{i}"] = losses[f'kappa_bias_abs_mean_{i}']
            if f'kappa_grad_correlation_{i}' in losses:
                log_data[f"inspect/kappa_grad_correlation_{i}"] = losses[f'kappa_grad_correlation_{i}']
            if f'kappa_bias_mean_top_{i}' in losses:
                log_data[f"inspect/kappa_bias_mean_top_{i}"] = losses[f'kappa_bias_mean_top_{i}']
            if f'kappa_bias_mean_bottom_{i}' in losses:
                log_data[f"inspect/kappa_bias_mean_bottom_{i}"] = losses[f'kappa_bias_mean_bottom_{i}']
            if f'kappa_bias_abs_mean_top_{i}' in losses:
                log_data[f"inspect/kappa_bias_abs_mean_top_{i}"] = losses[f'kappa_bias_abs_mean_top_{i}']
            if f'kappa_bias_abs_mean_bottom_{i}' in losses:
                log_data[f"inspect/kappa_bias_abs_mean_bottom_{i}"] = losses[f'kappa_bias_abs_mean_bottom_{i}']
            if f'kappa_bias_positive_mean_top_{i}' in losses:
                log_data[f"inspect/kappa_bias_positive_mean_top_{i}"] = losses[f'kappa_bias_positive_mean_top_{i}']
            if f'kappa_bias_positive_mean_bottom_{i}' in losses:
                log_data[f"inspect/kappa_bias_positive_mean_bottom_{i}"] = losses[f'kappa_bias_positive_mean_bottom_{i}']
            if f'kappa_bias_negative_mean_top_{i}' in losses:
                log_data[f"inspect/kappa_bias_negative_mean_top_{i}"] = losses[f'kappa_bias_negative_mean_top_{i}']
            if f'kappa_bias_negative_mean_bottom_{i}' in losses:
                log_data[f"inspect/kappa_bias_negative_mean_bottom_{i}"] = losses[f'kappa_bias_negative_mean_bottom_{i}']
            if f'kappa_slope_scale_abs_mean_{i}' in losses:
                log_data[f"inspect/kappa_slope_scale_abs_mean_{i}"] = losses[f'kappa_slope_scale_abs_mean_{i}']
            if f'kappa_slope_scale_abs_top5p_mean_{i}' in losses:
                log_data[f"inspect/kappa_slope_scale_abs_top5p_mean_{i}"] = losses[f'kappa_slope_scale_abs_top5p_mean_{i}']
            if f'kappa_slope_scale_abs_bottom5p_mean_{i}' in losses:
                log_data[f"inspect/kappa_slope_scale_abs_bottom5p_mean_{i}"] = losses[f'kappa_slope_scale_abs_bottom5p_mean_{i}']
            if f'mean_abs_gate_{i}' in losses:
                log_data[f"inspect/mean_abs_gate_{i}"] = losses[f'mean_abs_gate_{i}']
            if f'active_frac_gate_{i}' in losses:
                log_data[f"inspect/active_frac_gate_{i}"] = losses[f'active_frac_gate_{i}']
            if f'topk_share_gate_{i}' in losses:
                log_data[f"inspect/topk_share_gate_{i}"] = losses[f'topk_share_gate_{i}']
            if f'entropy_gate_{i}' in losses:
                log_data[f"inspect/entropy_gate_{i}"] = losses[f'entropy_gate_{i}']
            if f'router_grad_norm_top_{i}' in losses:
                log_data[f"inspect/router_grad_norm_top_{i}"] = losses[f'router_grad_norm_top_{i}']
            if f'router_grad_norm_bottom_{i}' in losses:
                log_data[f"inspect/router_grad_norm_bottom_{i}"] = losses[f'router_grad_norm_bottom_{i}']
            if f'router_row_norm_top_{i}' in losses:
                log_data[f"inspect/router_row_norm_top_{i}"] = losses[f'router_row_norm_top_{i}']
            if f'router_row_norm_bottom_{i}' in losses:
                log_data[f"inspect/router_row_norm_bottom_{i}"] = losses[f'router_row_norm_bottom_{i}']
            if f'router_grad_self_alignment_top_{i}' in losses:
                log_data[f"inspect/router_grad_self_alignment_top_{i}"] = losses[f'router_grad_self_alignment_top_{i}']
            if f'router_grad_self_alignment_bottom_{i}' in losses:
                log_data[f"inspect/router_grad_self_alignment_bottom_{i}"] = losses[f'router_grad_self_alignment_bottom_{i}']
            if f'router_weight_exp_gate_alignment_{i}' in losses:
                log_data[f"inspect/router_weight_exp_gate_alignment_{i}"] = losses[f'router_weight_exp_gate_alignment_{i}']
            if f'router_weight_exp_gate_alignment_top_{i}' in losses:
                log_data[f"inspect/router_weight_exp_gate_alignment_top_{i}"] = losses[f'router_weight_exp_gate_alignment_top_{i}']
            if f'router_weight_exp_gate_alignment_bottom_{i}' in losses:
                log_data[f"inspect/router_weight_exp_gate_alignment_bottom_{i}"] = losses[f'router_weight_exp_gate_alignment_bottom_{i}']
            if f'selected_scores_top_{i}' in losses:
                log_data[f"inspect/selected_scores_top_{i}"] = losses[f'selected_scores_top_{i}']
            if f'selected_scores_bottom_{i}' in losses:
                log_data[f"inspect/selected_scores_bottom_{i}"] = losses[f'selected_scores_bottom_{i}']
        wandb_run.log(drop_none_log_values(log_data), step=step)
        throughput_interval_steps = 0
        throughput_interval_time = 0.0

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")
print0(
    f"Skipped train conversations: {train_skipped_conversations}/{train_seen_conversations} "
    f"({100 * train_skipped_conversations / max(train_seen_conversations, 1):.2f}%)"
)

# Log to report
if not args.dry_run:
    from nanochat.report import get_report
    get_report().log(section="SFT", data=[
        user_config, # CLI args
        { # stats about the training setup
            "Number of iterations": step,
            "DDP world size": ddp_world_size,
        },
        { # stats about training outcomes
            "Minimum validation bpb": min_val_bpb,
        }
    ])
    if latest_chat_eval_results is not None:
        get_report().log(section="Chat evaluation sft", data=[
            {
                "step": latest_chat_eval_step,
                "task_names": chat_eval_task_names,
                "max_problems": args.chat_eval_max_problems,
                "batch_size": args.chat_eval_batch_size,
                "num_samples": args.chat_eval_num_samples,
                "max_new_tokens": args.chat_eval_max_new_tokens,
                "temperature": args.chat_eval_temperature,
                "top_k": args.chat_eval_top_k,
            },
            latest_chat_eval_results,
        ])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
