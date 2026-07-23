"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288

    # Override MoE expert eval capacity at inference time
    python -m scripts.base_eval --model-tag d24 --eval-capacity=2.0
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task, evaluate_task_detailed
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace loading utilities

class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
BOOLQ_YES_RATE_PRIOR = 0.62


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def normalize_boolq_answer(text):
    """Map a BoolQ choice string to True for yes and False for no."""
    normalized = text.strip().lower().rstrip(".:")
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    raise ValueError(f"Unsupported BoolQ answer label: {text!r}")


def compute_boolq_margins(details, data):
    """Compute per-example margins logp_yes - logp_no and gold labels."""
    margins = []
    for detail in details:
        item = data[detail['index']]
        choice_logps = detail.get('choice_logps')
        if choice_logps is None:
            raise ValueError("BoolQ margins require per-choice log probabilities.")

        yes_idx = None
        no_idx = None
        for idx, choice in enumerate(item['choices']):
            if normalize_boolq_answer(choice):
                yes_idx = idx
            else:
                no_idx = idx

        if yes_idx is None or no_idx is None:
            raise ValueError("Each BoolQ example must contain both a yes and a no choice.")

        margins.append({
            'margin': choice_logps[yes_idx] - choice_logps[no_idx],
            'gold_is_yes': normalize_boolq_answer(item['choices'][detail['gold_idx']]),
        })

    return margins


def compute_prior_matching_tau(margins, target_yes_rate):
    """Choose tau so the predicted yes rate matches the requested prior as closely as possible."""
    if not 0.0 <= target_yes_rate <= 1.0:
        raise ValueError(f"target_yes_rate must be in [0, 1], got {target_yes_rate!r}")

    sorted_margins = sorted(entry['margin'] for entry in margins)
    num_examples = len(sorted_margins)
    num_yes_predictions = int(round(target_yes_rate * num_examples))

    if num_yes_predictions <= 0:
        return sorted_margins[-1]
    if num_yes_predictions >= num_examples:
        return sorted_margins[0] - 1e-12

    lower = sorted_margins[num_examples - num_yes_predictions - 1]
    upper = sorted_margins[num_examples - num_yes_predictions]
    return 0.5 * (lower + upper)


def compute_calibrated_boolq_accuracy(details, data, tau):
    """Compute BoolQ accuracy after applying the calibrated decision rule margin > tau."""
    margins = compute_boolq_margins(details, data)
    num_correct = sum((entry['margin'] > tau) == entry['gold_is_yes'] for entry in margins)
    return num_correct / len(margins)


def compute_predicted_yes_rate(margins, tau):
    """Compute the fraction of BoolQ examples predicted as yes for a given tau."""
    return sum(entry['margin'] > tau for entry in margins) / len(margins)


def evaluate_core(model, tokenizer, device, max_per_task=-1, boolq_tau_mode='manual', boolq_target_yes_rate=BOOLQ_YES_RATE_PRIOR):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, core_metric, and core_metric_no_boolq.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        boolq_tau = None
        boolq_predicted_yes_rate = None
        if label.lower() == 'boolq' and boolq_tau_mode == 'prior-match':
            detailed = evaluate_task_detailed(model, tokenizer, data, device, task_meta)
            margins = compute_boolq_margins(detailed['details'], data)
            boolq_tau = compute_prior_matching_tau(margins, boolq_target_yes_rate)
            boolq_predicted_yes_rate = compute_predicted_yes_rate(margins, boolq_tau)
            accuracy = compute_calibrated_boolq_accuracy(detailed['details'], data, boolq_tau)
        else:
            accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        summary = f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f}"
        if boolq_tau is not None:
            summary = (
                f"{summary} | tau_mode: {boolq_tau_mode} | tau: {boolq_tau:.6f} "
                f"| predicted_yes_rate: {boolq_predicted_yes_rate:.4f}"
            )
        print0(f"{summary} | time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    centered_results_no_boolq = {
        label: value for label, value in centered_results.items() if label.lower() != "boolq"
    }
    if centered_results_no_boolq:
        core_metric_no_boolq = sum(centered_results_no_boolq.values()) / len(centered_results_no_boolq)
    else:
        core_metric_no_boolq = core_metric
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "core_metric_no_boolq": core_metric_no_boolq,
    }
    return out

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('-i', '--source', type=str, default='base', required=True, 
                        help="Source of the model: base|sft|rl")
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--loop', dest='total_ut_steps', type=int, default=None, help='Override the checkpoint Universal Transformer loop count')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument(
        '--boolq-tau-mode',
        type=str,
        choices=('manual', 'prior-match'),
        default='manual',
        help='BoolQ only: manual keeps the default argmin-loss decision rule, prior-match calibrates tau to match --boolq-target-yes-rate',
    )
    parser.add_argument(
        '--boolq-target-yes-rate',
        type=float,
        default=BOOLQ_YES_RATE_PRIOR,
        help='BoolQ only: target predicted yes rate used when --boolq-tau-mode=prior-match',
    )
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--eval-capacity', type=float, default=None, help='Override MoE eval capacity for nanochat checkpoints')
    parser.add_argument(
        '--use-kappa-swiglu',
        action=argparse.BooleanOptionalAction,
        dest='use_kappa_swiglu',
        default=None,
        help='Override the checkpoint config for expert kappa_bias on nanochat checkpoints',
    )
    parser.add_argument('--kappa-bias-fill-value', type=float, default=None,
                        help='Override all expert kappa_bias tensors in the loaded checkpoint with this constant value')
    parser.add_argument('--kappa-scale-fill-value', type=float, default=None,
                        help='Override all kappa_scale tensors in the loaded checkpoint with this constant value')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
        if args.eval_capacity is not None:
            print0("Ignoring --eval-capacity for HuggingFace models")
    else:
        model, tokenizer, meta = load_model(
            args.source,
            device,
            phase="eval",
            model_tag=args.model_tag,
            step=args.step,
            total_ut_steps=args.total_ut_steps,
            eval_capacity=args.eval_capacity,
            use_kappa_swiglu=args.use_kappa_swiglu,
            kappa_bias_fill_value=args.kappa_bias_fill_value,
            kappa_scale_fill_value=args.kappa_scale_fill_value,
        )
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model (step {meta['step']})"
        model_slug = f"{args.model_tag}_base_{meta['step']:06d}"
        if args.eval_capacity is not None:
            model_name = f"{model_name}, eval_capacity={args.eval_capacity:g}"
            model_slug = f"{model_slug}_ecap{args.eval_capacity:g}"
            print0(f"Overriding eval_capacity to {args.eval_capacity:g}")
        if args.use_kappa_swiglu is not None:
            model_name = f"{model_name}, use_kappa_swiglu={args.use_kappa_swiglu}"
            model_slug = f"{model_slug}_swiglu{int(args.use_kappa_swiglu)}"
            print0(f"Overriding use_kappa_swiglu to {args.use_kappa_swiglu}")
        if args.kappa_bias_fill_value is not None:
            model_name = f"{model_name}, kappa_bias_fill_value={args.kappa_bias_fill_value:g}"
            model_slug = f"{model_slug}_gpbias{args.kappa_bias_fill_value:g}"
            print0(f"Overriding expert kappa_bias to {args.kappa_bias_fill_value:g}")
        if args.kappa_scale_fill_value is not None:
            model_name = f"{model_name}, kappa_scale_fill_value={args.kappa_scale_fill_value:g}"
            model_slug = f"{model_slug}_gpscale{args.kappa_scale_fill_value:g}"
            print0(f"Overriding kappa_scale to {args.kappa_scale_fill_value:g}")

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes and not is_hf_model:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                with autocast_ctx:
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            with autocast_ctx:
                uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif 'sample' in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            with autocast_ctx:
                bpb, ntp_loss = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        with autocast_ctx:
            core_results = evaluate_core(
                model,
                tokenizer,
                device,
                max_per_task=args.max_per_task,
                boolq_tau_mode=args.boolq_tau_mode,
                boolq_target_yes_rate=args.boolq_target_yes_rate,
            )

        # Write CSV output
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_name = model_slug
            if args.max_per_task > 0:
                output_name = f"{output_name}_maxpt{args.max_per_task}"
            output_csv_path = os.path.join(base_dir, "base_eval", f"{output_name}.csv")
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

    # --- Log to report ---
    from nanochat.report import get_report
    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data[0]["CORE metric (no boolq)"] = core_results["core_metric_no_boolq"]
        report_data.append(core_results["centered_results"])

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
