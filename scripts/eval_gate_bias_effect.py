"""Evaluate expert gate bias effects over a fixed token budget.

This mirrors the validation-style token loop used in scripts/base_train.py,
but instead of reporting BPB it measures how the expert gate bias term changes
the post-SiLU gate activation across all valid dispatched expert slots.

Example:

    python -m scripts.eval_gate_bias_effect --model-tag d24
    torchrun --nproc_per_node=8 -m scripts.eval_gate_bias_effect --model-tag d24
"""

import argparse
from contextlib import nullcontext
from types import MethodType

import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import MOELayer


class ExpertGateBiasDeltaAccumulator:
    def __init__(self, bias_sign: int):
        if bias_sign not in (-1, 1):
            raise ValueError(f"bias_sign must be -1 or 1, got {bias_sign}")
        self.bias_sign = bias_sign
        self.sum_delta = None
        self.sum_abs_delta = None
        self.sum_sq_delta = None
        self.positive_count = None
        self.total_count = None

    def _lazy_init(self, device):
        if self.sum_delta is not None:
            return
        self.sum_delta = torch.zeros((), device=device, dtype=torch.float64)
        self.sum_abs_delta = torch.zeros((), device=device, dtype=torch.float64)
        self.sum_sq_delta = torch.zeros((), device=device, dtype=torch.float64)
        self.positive_count = torch.zeros((), device=device, dtype=torch.float64)
        self.total_count = torch.zeros((), device=device, dtype=torch.float64)

    @torch.inference_mode()
    def observe(self, layer: MOELayer, expert_inputs: torch.Tensor, expert_slot_mask: torch.Tensor):
        experts = layer.experts
        if getattr(experts, "gate_proj_bias", None) is None:
            return

        self._lazy_init(expert_inputs.device)

        gate_base = torch.bmm(expert_inputs, experts.gate_proj)
        router_confidence = experts._compute_router_confidence_gate_scale(
            expert_inputs,
            layer.router,
            grad_scale=experts.router_confidence_gate_bias_grad_scale,
        )
        if router_confidence is None:
            return

        bias_term = router_confidence.unsqueeze(-1) * experts.gate_proj_bias.unsqueeze(1)
        delta_gate = experts.act_fn(gate_base + self.bias_sign * bias_term) - experts.act_fn(gate_base)
        delta_gate = delta_gate.float()

        slot_mask = expert_slot_mask.unsqueeze(-1)
        slot_mask_f = slot_mask.to(dtype=delta_gate.dtype)
        total_count = expert_slot_mask.sum(dtype=torch.float64) * delta_gate.shape[-1]

        self.sum_delta.add_((delta_gate * slot_mask_f).sum(dtype=torch.float64))
        self.sum_abs_delta.add_((delta_gate.abs() * slot_mask_f).sum(dtype=torch.float64))
        self.sum_sq_delta.add_((delta_gate.square() * slot_mask_f).sum(dtype=torch.float64))
        self.positive_count.add_(torch.logical_and(delta_gate > 0, slot_mask).sum(dtype=torch.float64))
        self.total_count.add_(total_count)

    def reduce(self):
        if self.sum_delta is None:
            raise RuntimeError("No gate-bias statistics were collected")
        if dist.is_initialized():
            for tensor in (
                self.sum_delta,
                self.sum_abs_delta,
                self.sum_sq_delta,
                self.positive_count,
                self.total_count,
            ):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def summary(self):
        total_count = self.total_count.item()
        if total_count == 0:
            raise RuntimeError("Collected zero valid gate activations; no summary can be computed")
        mean_delta = self.sum_delta.item() / total_count
        mean_abs_delta = self.sum_abs_delta.item() / total_count
        rms_delta = (self.sum_sq_delta.item() / total_count) ** 0.5
        fraction_positive = self.positive_count.item() / total_count
        return {
            "mean(delta_gate)": mean_delta,
            "mean(abs(delta_gate))": mean_abs_delta,
            "rms(delta_gate)": rms_delta,
            "fraction_positive": fraction_positive,
            "count": total_count,
        }


def install_gate_bias_instrumentation(model, accumulator: ExpertGateBiasDeltaAccumulator):
    instrumented_layers = 0

    def instrumented_forward(self, x: torch.Tensor):
        B, T, C = x.size()

        expert_mask, router_probs, top_k_indices, rank = self.router(x)
        del expert_mask

        x_flat = x.view(B * T, C)
        exp_capacity = self.router.get_capacity(B * T)
        flat_top_k_indices = top_k_indices.view(-1)
        flat_rank = rank.view(-1)
        flat_token_indices = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)

        expert_inputs = torch.zeros(
            self.n_exp,
            exp_capacity,
            x_flat.size(1),
            dtype=x_flat.dtype,
            device=x_flat.device,
        )
        self._build_expert_inputs(
            x_flat,
            flat_rank,
            exp_capacity,
            flat_token_indices,
            flat_top_k_indices,
            expert_inputs,
        )

        valid_mask = flat_rank < exp_capacity
        expert_slot_mask = torch.zeros(
            self.n_exp,
            exp_capacity,
            dtype=torch.bool,
            device=x.device,
        )
        if valid_mask.any():
            expert_slot_mask[
                flat_top_k_indices[valid_mask],
                flat_rank[valid_mask],
            ] = True
            if self.use_qwen3_moe_mlp and getattr(self.experts, "gate_proj_bias", None) is not None:
                accumulator.observe(self, expert_inputs, expert_slot_mask)

        expert_outputs = self.experts(expert_inputs)
        output_flat = self._combine_expert_outputs(
            x_flat,
            expert_outputs,
            flat_rank,
            exp_capacity,
            flat_token_indices,
            flat_top_k_indices,
            router_probs,
            rank,
        )
        return output_flat.view(B, T, C)

    for module in model.modules():
        if not isinstance(module, MOELayer):
            continue
        if not getattr(module, "use_qwen3_moe_mlp", False):
            continue
        if getattr(module.experts, "gate_proj_bias", None) is None:
            continue
        module.forward = MethodType(instrumented_forward, module)
        instrumented_layers += 1

    return instrumented_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate expert gate bias activation effects")
    parser.add_argument("--source", type=str, default="base", choices=["base", "sft", "rl"], help="checkpoint family to load")
    parser.add_argument("--model-tag", type=str, default=None, help="checkpoint directory tag")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to load (default: latest)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="data split to evaluate")
    parser.add_argument("--eval-tokens", type=int, default=40 * 524288, help="target token budget, matching base_train.py by default")
    parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
    parser.add_argument("--eval-capacity", type=float, default=None, help="override MoE eval capacity")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument(
        "--bias-sign",
        type=str,
        default="implementation",
        choices=["implementation", "plus"],
        help=(
            "how to apply the bias term inside delta computation: "
            "'implementation' uses g_base - a*s to match nanochat's current forward, "
            "'plus' uses g_base + a*s to match the algebra written in the request"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bias_sign = -1 if args.bias_sign == "implementation" else 1

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    del ddp, ddp_local_rank
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
        eval_capacity=args.eval_capacity,
    )
    model.eval()

    accumulator = ExpertGateBiasDeltaAccumulator(bias_sign=bias_sign)
    instrumented_layers = install_gate_bias_instrumentation(model, accumulator)
    if instrumented_layers == 0:
        raise RuntimeError("No MoE expert layers with gate_proj_bias were found in the loaded model")

    sequence_len = meta["model_config"]["sequence_len"]
    tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
    eval_steps = args.eval_tokens // tokens_per_step
    if eval_steps <= 0:
        raise ValueError(
            f"eval_tokens={args.eval_tokens} is too small for one step with tokens_per_step={tokens_per_step}"
        )
    actual_tokens = eval_steps * tokens_per_step

    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        sequence_len,
        split=args.split,
        device=device,
    )

    print0(f"Evaluating gate bias effect for model step {meta['step']:06d}")
    print0(f"Split: {args.split} | requested tokens: {args.eval_tokens:,} | actual tokens: {actual_tokens:,}")
    print0(f"Instrumented MoE layers: {instrumented_layers} | bias sign mode: {args.bias_sign}")

    batch_iter = iter(loader)
    with torch.inference_mode():
        for step_idx in range(eval_steps):
            x, y = next(batch_iter)
            with autocast_ctx:
                model(x, y, loss_reduction="none")
            if step_idx == 0 or (step_idx + 1) % 50 == 0 or step_idx + 1 == eval_steps:
                print0(f"Processed {step_idx + 1}/{eval_steps} eval steps")

    accumulator.reduce()
    summary = accumulator.summary()
    print0("Gate bias activation delta summary:")
    print0(f"mean(delta_gate): {summary['mean(delta_gate)']:.8e}")
    print0(f"mean(abs(delta_gate)): {summary['mean(abs(delta_gate))']:.8e}")
    print0(f"rms(delta_gate): {summary['rms(delta_gate)']:.8e}")
    print0(f"fraction_positive: {summary['fraction_positive']:.8e}")
    print0(f"count: {int(summary['count'])}")

    compute_cleanup()


if __name__ == "__main__":
    main()