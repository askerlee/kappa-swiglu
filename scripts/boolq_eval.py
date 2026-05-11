"""Evaluate BoolQ only and report the confusion matrix."""
import argparse
import json
import os
import random
from contextlib import nullcontext

import torch
import yaml

from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    download_file_with_lock,
    get_base_dir,
    print0,
)
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task_detailed


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


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
            reduction=loss_reduction,
        )
        return loss


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    import shutil
    import tempfile
    import zipfile

    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def load_hf_model(hf_path, device):
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


def normalize_boolq_answer(text):
    """Map a BoolQ choice string to True for yes and False for no."""
    normalized = text.strip().lower().rstrip(".:")
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    raise ValueError(f"Unsupported BoolQ answer label: {text!r}")


def compute_boolq_confusion_counts(details, data, tau=0.0):
    """Treat Yes as the positive class and compute TP/TN/FP/FN using margin > tau."""
    counts = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for entry in compute_boolq_margins(details, data):
        pred_is_yes = entry['margin'] > tau
        gold_is_yes = entry['gold_is_yes']
        if pred_is_yes and gold_is_yes:
            counts['tp'] += 1
        elif pred_is_yes and not gold_is_yes:
            counts['fp'] += 1
        elif not pred_is_yes and gold_is_yes:
            counts['fn'] += 1
        else:
            counts['tn'] += 1
    return counts


def compute_average_boolq_margin(details, data):
    """Compute the average per-example margin logp_yes - logp_no."""
    margins = compute_boolq_margins(details, data)
    return sum(entry['margin'] for entry in margins) / len(margins)


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


def compute_class_conditional_boolq_margin_means(details, data):
    """Compute mean margins separately for gold-yes and gold-no examples."""
    margins = compute_boolq_margins(details, data)
    yes_margins = [entry['margin'] for entry in margins if entry['gold_is_yes']]
    no_margins = [entry['margin'] for entry in margins if not entry['gold_is_yes']]
    return {
        'mean_margin_yes_examples': sum(yes_margins) / len(yes_margins),
        'mean_margin_no_examples': sum(no_margins) / len(no_margins),
    }


def compute_calibrated_boolq_accuracy(details, data, tau=0.0):
    """Compute accuracy after applying the calibrated decision rule margin > tau."""
    margins = compute_boolq_margins(details, data)
    num_correct = sum((entry['margin'] > tau) == entry['gold_is_yes'] for entry in margins)
    return num_correct / len(margins)


def load_boolq_data(max_examples):
    """Load BoolQ task metadata and examples from the eval bundle."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    boolq_task = None
    for task in config['icl_tasks']:
        if task['label'].lower() == 'boolq':
            boolq_task = task
            break
    if boolq_task is None:
        raise ValueError("BoolQ task not found in CORE config.")

    task_meta = {
        'task_type': boolq_task['icl_task_type'],
        'dataset_uri': boolq_task['dataset_uri'],
        'num_fewshot': boolq_task['num_fewshot'][0],
        'continuation_delimiter': boolq_task.get('continuation_delimiter', ' '),
    }
    data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(data)
    if max_examples > 0:
        data = data[:max_examples]
    return data, task_meta


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate BoolQ only and report TP/TN/FP/FN")
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('-i', '--source', type=str, default='base', help='Source of the model: base|sft|rl')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-examples', type=int, default=-1, help='Max BoolQ examples to evaluate (-1 = all)')
    parser.add_argument('--tau', type=float, default=0.0, help='Predict yes when margin logp_yes - logp_no is greater than tau')
    parser.add_argument('--eval-capacity', type=float, default=None, help='Override MoE eval capacity for nanochat checkpoints')
    parser.add_argument(
        '--use-exp-gate-proj-bias',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Override the checkpoint config for expert gate_proj_bias on nanochat checkpoints',
    )
    parser.add_argument(
        '--exp-gate-proj-bias-mode',
        type=str,
        default=None,
        choices=['full', 'rank1'],
        help='Override expert gate_proj_bias parameterization for nanochat checkpoints',
    )
    parser.add_argument(
        '--exp-gate-proj-bias-fill-value',
        type=float,
        default=None,
        help='Override all expert gate_proj_bias tensors in the loaded checkpoint with this constant value',
    )
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()

    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        model_name = args.hf_path
        if args.eval_capacity is not None:
            print0('Ignoring --eval-capacity for HuggingFace models')
    else:
        model, tokenizer, meta = load_model(
            args.source,
            device,
            phase='eval',
            model_tag=args.model_tag,
            step=args.step,
            eval_capacity=args.eval_capacity,
            use_exp_gate_proj_bias=args.use_exp_gate_proj_bias,
            exp_gate_proj_bias_mode=args.exp_gate_proj_bias_mode,
            exp_gate_proj_bias_fill_value=args.exp_gate_proj_bias_fill_value,
        )
        model_name = f"{args.source}_model (step {meta['step']})"
        if args.eval_capacity is not None:
            model_name = f"{model_name}, eval_capacity={args.eval_capacity:g}"
        if args.exp_gate_proj_bias_mode is not None:
            model_name = f"{model_name}, exp_gate_proj_bias_mode={args.exp_gate_proj_bias_mode}"
        if args.exp_gate_proj_bias_fill_value is not None:
            model_name = f"{model_name}, exp_gate_proj_bias_fill_value={args.exp_gate_proj_bias_fill_value:g}"

    data, task_meta = load_boolq_data(args.max_examples)
    print0(f"Evaluating model on BoolQ: {model_name}")
    print0(f"Examples: {len(data)} | {task_meta['num_fewshot']}-shot")

    with autocast_ctx:
        results = evaluate_task_detailed(model, tokenizer, data, device, task_meta)

    confusion = compute_boolq_confusion_counts(results['details'], data, tau=args.tau)
    calibrated_accuracy = compute_calibrated_boolq_accuracy(results['details'], data, tau=args.tau)
    average_margin = compute_average_boolq_margin(results['details'], data)
    class_conditional_margins = compute_class_conditional_boolq_margin_means(results['details'], data)
    total = sum(confusion.values())
    if ddp_rank == 0:
        print(f"Original accuracy: {results['accuracy']:.4f}")
        print(f"Calibrated accuracy: {calibrated_accuracy:.4f}")
        print(f"tau: {args.tau:.4f}")
        print(f"Average margin (logp_yes - logp_no): {average_margin:.4f}")
        print(f"mean_margin_yes_examples: {class_conditional_margins['mean_margin_yes_examples']:.4f}")
        print(f"mean_margin_no_examples: {class_conditional_margins['mean_margin_no_examples']:.4f}")
        print(f"TP: {confusion['tp']}")
        print(f"TN: {confusion['tn']}")
        print(f"FP: {confusion['fp']}")
        print(f"FN: {confusion['fn']}")
        print(f"Total: {total}")

    compute_cleanup()


if __name__ == '__main__':
    main()