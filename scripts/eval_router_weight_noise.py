"""Compare clean and perturbed top-k expert choices on the same validation tokens."""

import argparse
import math
from contextlib import ExitStack, nullcontext
from functools import partial
from unittest.mock import patch

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import COMPUTE_DTYPE, autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import get_moe_layer_indices, perturb_router_weight


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('base', 'sft', 'rl'), default='base')
    parser.add_argument('--model-tag', default=None, help='checkpoint directory name')
    parser.add_argument('--step', type=int, default=None, help='checkpoint step (default: latest)')
    parser.add_argument('--device-type', choices=('cuda', 'cpu', 'mps'), default=None)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--sequence-len', type=int, default=None, help='default: min(512, model context length)')
    parser.add_argument('--device-batch-size', type=int, default=1)
    parser.add_argument('--batches', type=int, default=16, help='number of distinct batches')
    parser.add_argument('--trials', type=int, default=4, help='independent noisy forwards per batch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--router-weight-noise-prob', type=float, default=0.1,
                        help='probability of perturbing a router on each invocation (1 always perturbs)')
    parser.add_argument('--router-weight-noise-std', type=float, default=0.5,
                        help='Gaussian noise std before dividing by sqrt(hidden size)')
    args = parser.parse_args()
    if not 0.0 <= args.router_weight_noise_prob <= 1.0:
        parser.error('--router-weight-noise-prob must be in [0, 1]')
    if not math.isfinite(args.router_weight_noise_std) or args.router_weight_noise_std < 0.0:
        parser.error('--router-weight-noise-std must be finite and >= 0')
    if min(args.device_batch_size, args.batches, args.trials) < 1:
        parser.error('--device-batch-size, --batches, and --trials must be positive')
    if args.sequence_len is not None and args.sequence_len < 1:
        parser.error('--sequence-len must be positive')
    return args


def record_top_k(choices, module, inputs, output):
    choices.append(output[3])


def capture_routing(model, routers, tokens, probability=0.0, std=0.0):
    choices = [[] for _ in routers]
    with ExitStack() as stack:
        for router, layer_choices in zip(routers, choices, strict=True):
            stack.enter_context(router.register_forward_hook(partial(record_top_k, layer_choices)))
            if probability > 0.0 and std > 0.0:
                stack.enter_context(patch.object(
                    router, 'sample_weight', partial(perturb_router_weight, router.w_g.weight, probability, std),
                ))
        model(tokens)
    return choices


def routing_change_counts(clean, noisy):
    if clean.shape != noisy.shape:
        raise ValueError('Clean and noisy routing must have the same shape')
    replaced = ~(clean.unsqueeze(-1) == noisy.unsqueeze(-2)).any(dim=-1)
    return torch.stack((replaced.sum(), replaced.any(dim=-1).sum()))


@torch.inference_mode()
def measure_routing_changes(model, routers, batches, args, device_type):
    device = routers[0].w_g.weight.device
    changes = torch.zeros((len(routers), 2), dtype=torch.int64, device=device)
    totals = torch.zeros_like(changes)
    autocast = torch.amp.autocast(device_type, dtype=COMPUTE_DTYPE) if device_type == 'cuda' else nullcontext()
    with autocast:
        for _ in range(args.batches):
            tokens, _ = next(batches)
            clean = capture_routing(model, routers, tokens)
            for _ in range(args.trials):
                noisy = capture_routing(model, routers, tokens, args.router_weight_noise_prob, args.router_weight_noise_std)
                for layer_idx, (clean_passes, noisy_passes) in enumerate(zip(clean, noisy, strict=True)):
                    for clean_choices, noisy_choices in zip(clean_passes, noisy_passes, strict=True):
                        changes[layer_idx] += routing_change_counts(clean_choices, noisy_choices)
            for layer_idx, clean_passes in enumerate(clean):
                totals[layer_idx, 0] += args.trials * sum(choices.numel() for choices in clean_passes)
                totals[layer_idx, 1] += args.trials * sum(choices.shape[0] for choices in clean_passes)
    return changes, totals


def print_result(label, changed, total):
    replaced, changed_tokens = changed
    assignments, token_visits = total
    print0(f'{label:<9} {100 * replaced / assignments:7.2f}% ({replaced:,}/{assignments:,}) replaced | '
           f'{100 * changed_tokens / token_visits:7.2f}% token visits with at least one change')


def main():
    args = parse_args()
    device_type = args.device_type or autodetect_device_type()
    ddp, rank, _, world_size, device = compute_init(device_type, seed=args.seed)
    try:
        model, tokenizer, meta = load_model(args.source, device, phase='eval', model_tag=args.model_tag, step=args.step)
        layer_indices = get_moe_layer_indices(model.config)
        if not layer_indices:
            raise ValueError('The checkpoint has no MoE routers')
        routers = [model.transformer.h[layer_idx].mlp.router for layer_idx in layer_indices]
        sequence_len = args.sequence_len or min(512, model.config.sequence_len)
        if sequence_len > model.config.sequence_len:
            raise ValueError(f'--sequence-len exceeds the model context length ({model.config.sequence_len})')
        batches = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, sequence_len, split=args.split, device=device,
        )
        torch.manual_seed(args.seed + rank)
        changes, totals = measure_routing_changes(model, routers, batches, args, device_type)
        if ddp:
            torch.distributed.all_reduce(changes)
            torch.distributed.all_reduce(totals)
        changes, totals = changes.cpu(), totals.cpu()
        input_tokens = args.batches * args.device_batch_size * sequence_len * world_size
        print0(f'Checkpoint step {meta["step"]} | {args.split}: {input_tokens:,} input tokens | {args.trials} trials')
        print0(f'Router weight noise: probability={args.router_weight_noise_prob:g}, std={args.router_weight_noise_std:g}')
        print0('Pre-capacity top-k changes, ignoring order and aggregating all UT passes:')
        for layer_idx, changed, total in zip(layer_indices, changes.tolist(), totals.tolist(), strict=True):
            print_result(f'Layer {layer_idx}', changed, total)
        print_result('Overall', changes.sum(dim=0).tolist(), totals.sum(dim=0).tolist())
    finally:
        compute_cleanup()


if __name__ == '__main__':
    main()