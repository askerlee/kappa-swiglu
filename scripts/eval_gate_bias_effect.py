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


TOP_FRACTION = 0.1
TOP_QUANTILE = 1.0 - TOP_FRACTION
QUANTILE_SAMPLE_SIZE = 131072


class StreamingValueStats:
    def __init__(self):
        self.sum_value = None
        self.sum_abs_value = None
        self.sum_sq_value = None
        self.positive_count = None
        self.count = None

    def _lazy_init(self, device):
        if self.sum_value is not None:
            return
        self.sum_value = torch.zeros((), device=device, dtype=torch.float64)
        self.sum_abs_value = torch.zeros((), device=device, dtype=torch.float64)
        self.sum_sq_value = torch.zeros((), device=device, dtype=torch.float64)
        self.positive_count = torch.zeros((), device=device, dtype=torch.float64)
        self.count = torch.zeros((), device=device, dtype=torch.float64)

    @torch.inference_mode()
    def observe(self, values: torch.Tensor, mask: torch.Tensor):
        self._lazy_init(values.device)
        expanded_mask = torch.broadcast_to(mask, values.shape)
        masked_values = values[expanded_mask]
        if masked_values.numel() == 0:
            return
        self.sum_value.add_(masked_values.sum(dtype=torch.float64))
        self.sum_abs_value.add_(masked_values.abs().sum(dtype=torch.float64))
        self.sum_sq_value.add_(masked_values.square().sum(dtype=torch.float64))
        self.positive_count.add_((masked_values > 0).sum(dtype=torch.float64))
        self.count.add_(masked_values.numel())

    def reduce(self):
        if self.sum_value is None:
            return
        if dist.is_initialized():
            for tensor in (
                self.sum_value,
                self.sum_abs_value,
                self.sum_sq_value,
                self.positive_count,
                self.count,
            ):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def summary(self, name: str):
        count = 0.0 if self.count is None else self.count.item()
        if count == 0:
            return {
                f"mean({name})": float("nan"),
                f"mean(abs({name}))": float("nan"),
                f"rms({name})": float("nan"),
                f"fraction_positive_{name}": float("nan"),
                f"count_{name}": 0.0,
            }
        return {
            f"mean({name})": self.sum_value.item() / count,
            f"mean(abs({name}))": self.sum_abs_value.item() / count,
            f"rms({name})": (self.sum_sq_value.item() / count) ** 0.5,
            f"fraction_positive_{name}": self.positive_count.item() / count,
            f"count_{name}": count,
        }


class PrioritySampleQuantile:
    def __init__(self, sample_size: int):
        self.sample_size = int(sample_size)
        self.samples = None
        self.priorities = None

    @torch.inference_mode()
    def observe(self, values: torch.Tensor, mask: torch.Tensor):
        if self.sample_size <= 0:
            return
        expanded_mask = torch.broadcast_to(mask, values.shape)
        masked_values = values[expanded_mask]
        if masked_values.numel() == 0:
            return
        masked_values = masked_values.detach().float().reshape(-1).cpu()
        priorities = torch.rand(masked_values.numel(), dtype=torch.float32)
        if self.samples is None:
            self.samples = masked_values
            self.priorities = priorities
        else:
            self.samples = torch.cat((self.samples, masked_values), dim=0)
            self.priorities = torch.cat((self.priorities, priorities), dim=0)
        if self.samples.numel() > self.sample_size:
            keep = torch.topk(self.priorities, k=self.sample_size, largest=True, sorted=False).indices
            self.samples = self.samples[keep]
            self.priorities = self.priorities[keep]

    def get_samples(self):
        if self.samples is None:
            return torch.empty(0, dtype=torch.float32)
        return self.samples


def compute_global_quantile(local_samples: torch.Tensor, quantile: float):
    samples = local_samples
    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, samples)
        sample_parts = [part for part in gathered if isinstance(part, torch.Tensor) and part.numel() > 0]
        samples = torch.cat(sample_parts, dim=0) if sample_parts else torch.empty(0, dtype=torch.float32)
    if samples.numel() == 0:
        return float("nan")
    return torch.quantile(samples, quantile).item()


class GateBiasStatsCollector:
    def __init__(self, bias_sign: int):
        if bias_sign not in (-1, 1):
            raise ValueError(f"bias_sign must be -1 or 1, got {bias_sign}")
        self.bias_sign = bias_sign
    @torch.inference_mode()
    def initialize_stats(self, delta_threshold, relative_threshold):
        self.delta_threshold = delta_threshold
        self.relative_threshold = relative_threshold
        self.delta_stats = StreamingValueStats()
        self.relative_stats = StreamingValueStats()
        self.top_delta_stats = StreamingValueStats() if delta_threshold is not None else None
        self.top_relative_stats = StreamingValueStats() if relative_threshold is not None else None
        self.delta_sampler = None
        self.relative_sampler = None

    def enable_sampling(self, sample_size: int):
        self.delta_sampler = PrioritySampleQuantile(sample_size)
        self.relative_sampler = PrioritySampleQuantile(sample_size)

    def observe(self, layer: MOELayer, expert_inputs: torch.Tensor, expert_slot_mask: torch.Tensor):
        experts = layer.experts
        if getattr(experts, "gate_proj_bias", None) is None:
            return

        gate_base = torch.bmm(expert_inputs, experts.gate_proj)
        router_confidence = experts._compute_router_confidence_gate_scale(
            expert_inputs,
            layer.router,
            grad_scale=experts.router_confidence_gate_bias_grad_scale,
        )
        if router_confidence is None:
            return

        bias_term = router_confidence.unsqueeze(-1) * experts.gate_proj_bias.unsqueeze(1)
        gate_base_acts = experts.act_fn(gate_base)
        delta_gate = experts.act_fn(gate_base + self.bias_sign * bias_term) - gate_base_acts
        delta_gate = delta_gate.float()
        gate_base_acts = gate_base_acts.float()

        slot_mask = expert_slot_mask.unsqueeze(-1)
        self.delta_stats.observe(delta_gate, slot_mask)
        if self.delta_sampler is not None:
            self.delta_sampler.observe(delta_gate, slot_mask)
        if self.top_delta_stats is not None:
            top_delta_mask = torch.logical_and(slot_mask, delta_gate >= self.delta_threshold)
            self.top_delta_stats.observe(delta_gate, top_delta_mask)

        relative_delta = delta_gate / gate_base_acts
        finite_relative_mask = torch.logical_and(slot_mask, torch.isfinite(relative_delta))
        self.relative_stats.observe(relative_delta, finite_relative_mask)
        if self.relative_sampler is not None:
            self.relative_sampler.observe(relative_delta, finite_relative_mask)
        if self.top_relative_stats is not None:
            top_relative_mask = torch.logical_and(finite_relative_mask, relative_delta >= self.relative_threshold)
            self.top_relative_stats.observe(relative_delta, top_relative_mask)

    def reduce(self):
        self.delta_stats.reduce()
        self.relative_stats.reduce()
        if self.top_delta_stats is not None:
            self.top_delta_stats.reduce()
        if self.top_relative_stats is not None:
            self.top_relative_stats.reduce()

    def summary(self):
        delta_summary = self.delta_stats.summary("delta_gate")
        if delta_summary["count_delta_gate"] == 0:
            raise RuntimeError("Collected zero valid gate activations; no summary can be computed")
        summary = {
            "mean(delta_gate)": delta_summary["mean(delta_gate)"],
            "mean(abs(delta_gate))": delta_summary["mean(abs(delta_gate))"],
            "rms(delta_gate)": delta_summary["rms(delta_gate)"],
            "fraction_positive": delta_summary["fraction_positive_delta_gate"],
            "count": delta_summary["count_delta_gate"],
        }
        relative_summary = self.relative_stats.summary("delta_gate / silu(g_base)")
        summary.update({
            "mean(delta_gate / silu(g_base))": relative_summary["mean(delta_gate / silu(g_base))"],
            "mean(abs(delta_gate / silu(g_base)))": relative_summary["mean(abs(delta_gate / silu(g_base)))"],
            "rms(delta_gate / silu(g_base))": relative_summary["rms(delta_gate / silu(g_base))"],
            "fraction_positive_relative": relative_summary["fraction_positive_delta_gate / silu(g_base)"],
            "relative_count": relative_summary["count_delta_gate / silu(g_base)"],
        })
        if self.top_delta_stats is not None:
            top_delta_summary = self.top_delta_stats.summary("top10_delta_gate")
            summary.update({
                "mean(top10_delta_gate)": top_delta_summary["mean(top10_delta_gate)"],
                "mean(abs(top10_delta_gate))": top_delta_summary["mean(abs(top10_delta_gate))"],
                "rms(top10_delta_gate)": top_delta_summary["rms(top10_delta_gate)"],
                "fraction_positive_top10_delta_gate": top_delta_summary["fraction_positive_top10_delta_gate"],
                "count_top10_delta_gate": top_delta_summary["count_top10_delta_gate"],
            })
        if self.top_relative_stats is not None:
            top_relative_summary = self.top_relative_stats.summary("top10_delta_gate / silu(g_base)")
            summary.update({
                "mean(top10_delta_gate / silu(g_base))": top_relative_summary["mean(top10_delta_gate / silu(g_base))"],
                "mean(abs(top10_delta_gate / silu(g_base)))": top_relative_summary["mean(abs(top10_delta_gate / silu(g_base)))"],
                "rms(top10_delta_gate / silu(g_base))": top_relative_summary["rms(top10_delta_gate / silu(g_base))"],
                "fraction_positive_top10_relative": top_relative_summary["fraction_positive_top10_delta_gate / silu(g_base)"],
                "count_top10_relative": top_relative_summary["count_top10_delta_gate / silu(g_base)"],
            })
        return summary


class GateBiasObserverHost:
    def __init__(self):
        self.collector = None

    def set_collector(self, collector: GateBiasStatsCollector):
        self.collector = collector

    @torch._dynamo.disable
    @torch.inference_mode()
    def observe(self, layer: MOELayer, expert_inputs: torch.Tensor, expert_slot_mask: torch.Tensor):
        if self.collector is None:
            raise RuntimeError("Gate bias observer host was called without an active collector")
        self.collector.observe(layer, expert_inputs, expert_slot_mask)


def install_gate_bias_instrumentation(model, observer_host: GateBiasObserverHost):
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
                observer_host.observe(self, expert_inputs, expert_slot_mask)

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


def build_loader(tokenizer, device_batch_size, sequence_len, split, device):
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        device_batch_size,
        sequence_len,
        split=split,
        device=device,
    )


def run_eval_pass(model, loader, eval_steps, autocast_ctx, pass_name):
    batch_iter = iter(loader)
    with torch.inference_mode():
        for step_idx in range(eval_steps):
            x, y = next(batch_iter)
            with autocast_ctx:
                model(x, y, loss_reduction="none")
            if step_idx == 0 or (step_idx + 1) % 50 == 0 or step_idx + 1 == eval_steps:
                print0(f"{pass_name}: processed {step_idx + 1}/{eval_steps} eval steps")


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
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the instrumented model for faster inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bias_sign = -1

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

    observer_host = GateBiasObserverHost()
    instrumented_layers = install_gate_bias_instrumentation(model, observer_host)
    if instrumented_layers == 0:
        raise RuntimeError("No MoE expert layers with gate_proj_bias were found in the loaded model")
    if args.compile:
        model = torch.compile(model, dynamic=False)

    sequence_len = meta["model_config"]["sequence_len"]
    tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
    eval_steps = args.eval_tokens // tokens_per_step
    if eval_steps <= 0:
        raise ValueError(
            f"eval_tokens={args.eval_tokens} is too small for one step with tokens_per_step={tokens_per_step}"
        )
    actual_tokens = eval_steps * tokens_per_step

    print0(f"Evaluating gate bias effect for model step {meta['step']:06d}")
    print0(f"Split: {args.split} | requested tokens: {args.eval_tokens:,} | actual tokens: {actual_tokens:,}")
    print0(
        f"Instrumented MoE layers: {instrumented_layers} | bias sign: negative | compile: {args.compile}"
    )

    sampling_collector = GateBiasStatsCollector(bias_sign=bias_sign)
    sampling_collector.initialize_stats(None, None)
    sampling_collector.enable_sampling(QUANTILE_SAMPLE_SIZE)
    observer_host.set_collector(sampling_collector)
    sampling_loader = build_loader(tokenizer, args.device_batch_size, sequence_len, args.split, device)
    run_eval_pass(model, sampling_loader, eval_steps, autocast_ctx, pass_name="Sampling pass")

    delta_threshold = compute_global_quantile(sampling_collector.delta_sampler.get_samples(), TOP_QUANTILE)
    relative_threshold = compute_global_quantile(sampling_collector.relative_sampler.get_samples(), TOP_QUANTILE)
    print0(f"Top {TOP_FRACTION:.0%} delta_gate threshold: {delta_threshold:.5e}")
    print0(f"Top {TOP_FRACTION:.0%} relative delta_gate threshold: {relative_threshold:.5e}")

    summary_collector = GateBiasStatsCollector(bias_sign=bias_sign)
    summary_collector.initialize_stats(delta_threshold, relative_threshold)
    observer_host.set_collector(summary_collector)
    summary_loader = build_loader(tokenizer, args.device_batch_size, sequence_len, args.split, device)
    run_eval_pass(model, summary_loader, eval_steps, autocast_ctx, pass_name="Summary pass")

    summary_collector.reduce()
    summary = summary_collector.summary()
    print0("Gate bias activation delta summary:")
    print0(f"mean(delta_gate): {summary['mean(delta_gate)']:.5e}")
    print0(f"mean(abs(delta_gate)): {summary['mean(abs(delta_gate))']:.5e}")
    print0(f"rms(delta_gate): {summary['rms(delta_gate)']:.5e}")
    print0(f"fraction_positive: {summary['fraction_positive']:.5e}")
    print0(f"count: {int(summary['count'])}")
    print0(f"mean(delta_gate / silu(g_base)): {summary['mean(delta_gate / silu(g_base))']:.5e}")
    print0(f"mean(abs(delta_gate / silu(g_base))): {summary['mean(abs(delta_gate / silu(g_base)))']:.5e}")
    print0(f"rms(delta_gate / silu(g_base)): {summary['rms(delta_gate / silu(g_base))']:.5e}")
    print0(f"fraction_positive_relative: {summary['fraction_positive_relative']:.5e}")
    print0(f"relative_count: {int(summary['relative_count'])}")
    print0(f"mean(top10_delta_gate): {summary['mean(top10_delta_gate)']:.5e}")
    print0(f"mean(abs(top10_delta_gate)): {summary['mean(abs(top10_delta_gate))']:.5e}")
    print0(f"rms(top10_delta_gate): {summary['rms(top10_delta_gate)']:.5e}")
    print0(f"fraction_positive_top10_delta_gate: {summary['fraction_positive_top10_delta_gate']:.5e}")
    print0(f"count_top10_delta_gate: {int(summary['count_top10_delta_gate'])}")
    print0(f"mean(top10_delta_gate / silu(g_base)): {summary['mean(top10_delta_gate / silu(g_base))']:.5e}")
    print0(f"mean(abs(top10_delta_gate / silu(g_base))): {summary['mean(abs(top10_delta_gate / silu(g_base)))']:.5e}")
    print0(f"rms(top10_delta_gate / silu(g_base)): {summary['rms(top10_delta_gate / silu(g_base))']:.5e}")
    print0(f"fraction_positive_top10_relative: {summary['fraction_positive_top10_relative']:.5e}")
    print0(f"count_top10_relative: {int(summary['count_top10_relative'])}")

    compute_cleanup()


if __name__ == "__main__":
    main()