"""Evaluate expert kappa slope effects over a fixed token budget.

This mirrors the validation-style token loop used in scripts/base_train.py,
but instead of reporting BPB it measures how kappa_bias and kappa_scale change
the slope-scaled SiLU gate activation across all valid dispatched expert slots.

Example:

    python -m scripts.eval_gate_bias_effect --model-tag d24
    torchrun --nproc_per_node=8 -m scripts.eval_gate_bias_effect --model-tag d24
"""

import argparse
from contextlib import nullcontext
import math
import time
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.common import COMPUTE_DTYPE, autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import MOELayer


TOP10_FRACTION = 0.1
TOP10_QUANTILE = 1.0 - TOP10_FRACTION
TOP2_FRACTION = 0.02
TOP2_QUANTILE = 1.0 - TOP2_FRACTION
QUANTILE_SAMPLE_SIZE = 131072
QUANTILE_SLOT_SAMPLE_COUNT = 8
RELATIVE_DENOM_EPS = 1e-6


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


@torch.inference_mode()
def compute_routing_metrics(top_k_indices: torch.Tensor, all_router_probs: torch.Tensor, n_exp: int):
    assignment_counts = torch.bincount(top_k_indices.reshape(-1), minlength=n_exp).to(torch.float64)
    router_entropy_sum = -torch.xlogy(all_router_probs, all_router_probs).sum(dtype=torch.float64)
    token_count = torch.tensor(all_router_probs.size(0), device=all_router_probs.device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(assignment_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(router_entropy_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    total_assignments = assignment_counts.sum()
    if total_assignments.item() == 0 or token_count.item() == 0:
        raise RuntimeError("Cannot compute routing metrics without tokens and expert assignments")
    utilization = assignment_counts / total_assignments
    if n_exp > 1:
        normalized_load_entropy = -torch.xlogy(utilization, utilization).sum() / math.log(n_exp)
    else:
        normalized_load_entropy = utilization.new_ones(())
    load_cv = utilization.std(unbiased=False) / utilization.mean()
    router_probability_entropy = router_entropy_sum / token_count
    return utilization, normalized_load_entropy, load_cv, router_probability_entropy


class GateBiasStatsCollector:
    def __init__(self, collect_activation_stats=True):
        self.collect_activation_stats = collect_activation_stats

    @torch.inference_mode()
    def initialize_stats(
        self,
        top10_delta_threshold,
        top10_relative_threshold,
        top2_delta_threshold=None,
        top2_relative_threshold=None,
    ):
        self.top10_delta_threshold = top10_delta_threshold
        self.top10_relative_threshold = top10_relative_threshold
        self.top2_delta_threshold = top2_delta_threshold
        self.top2_relative_threshold = top2_relative_threshold
        self.delta_stats = StreamingValueStats()
        self.relative_stats = StreamingValueStats()
        self.top10_delta_stats = StreamingValueStats() if top10_delta_threshold is not None else None
        self.top10_relative_stats = StreamingValueStats() if top10_relative_threshold is not None else None
        self.top2_delta_stats = StreamingValueStats() if top2_delta_threshold is not None else None
        self.top2_relative_stats = StreamingValueStats() if top2_relative_threshold is not None else None
        self.normalized_load_entropy_stats = StreamingValueStats()
        self.load_cv_stats = StreamingValueStats()
        self.router_probability_entropy_stats = StreamingValueStats()
        self.expert_utilization_sum = None
        self.routing_observation_count = None
        self.routing_stats_by_layer = {}
        self.delta_sampler = None
        self.relative_sampler = None

    def enable_sampling(self, sample_size: int):
        self.delta_sampler = PrioritySampleQuantile(sample_size)
        self.relative_sampler = PrioritySampleQuantile(sample_size)

    @torch.inference_mode()
    def _sample_valid_slot_values(self, values: torch.Tensor, expert_slot_mask: torch.Tensor):
        valid_slots = torch.nonzero(expert_slot_mask, as_tuple=False)
        if valid_slots.numel() == 0:
            return None
        if valid_slots.size(0) > QUANTILE_SLOT_SAMPLE_COUNT:
            sampled_idx = torch.randint(
                valid_slots.size(0),
                (QUANTILE_SLOT_SAMPLE_COUNT,),
                device=valid_slots.device,
            )
            valid_slots = valid_slots[sampled_idx]
        sampled_values = values[valid_slots[:, 0], valid_slots[:, 1]]
        return sampled_values

    def observe(
        self,
        layer: MOELayer,
        expert_inputs: torch.Tensor,
        expert_router_scores: torch.Tensor,
        expert_slot_mask: torch.Tensor,
        top_k_indices: torch.Tensor,
        all_router_probs: torch.Tensor,
    ):
        experts = layer.experts
        if self.collect_activation_stats and not getattr(experts, "use_kappa_swiglu", False):
            return

        utilization, normalized_load_entropy, load_cv, router_probability_entropy = compute_routing_metrics(
            top_k_indices,
            all_router_probs,
            layer.n_exp,
        )
        scalar_mask = torch.ones((), device=utilization.device, dtype=torch.bool)
        self.normalized_load_entropy_stats.observe(normalized_load_entropy, scalar_mask)
        self.load_cv_stats.observe(load_cv, scalar_mask)
        self.router_probability_entropy_stats.observe(router_probability_entropy, scalar_mask)
        if self.expert_utilization_sum is None:
            self.expert_utilization_sum = utilization.clone()
            self.routing_observation_count = utilization.new_ones(())
        else:
            self.expert_utilization_sum.add_(utilization)
            self.routing_observation_count.add_(1)
        layer_idx = experts.layer_idx
        if layer_idx not in self.routing_stats_by_layer:
            self.routing_stats_by_layer[layer_idx] = {
                "normalized_load_entropy": StreamingValueStats(),
                "load_cv": StreamingValueStats(),
                "router_probability_entropy": StreamingValueStats(),
                "expert_utilization_sum": utilization.clone(),
                "observation_count": utilization.new_ones(()),
            }
        else:
            layer_stats = self.routing_stats_by_layer[layer_idx]
            layer_stats["expert_utilization_sum"].add_(utilization)
            layer_stats["observation_count"].add_(1)
        layer_stats = self.routing_stats_by_layer[layer_idx]
        layer_stats["normalized_load_entropy"].observe(normalized_load_entropy, scalar_mask)
        layer_stats["load_cv"].observe(load_cv, scalar_mask)
        layer_stats["router_probability_entropy"].observe(router_probability_entropy, scalar_mask)

        if not self.collect_activation_stats:
            return

        gate_base = torch.bmm(expert_inputs, experts.gate_proj)
        kappa_bias = experts._materialize_kappa_bias_for_eval(
            gate_base.dtype,
            gate_base.device,
        )
        kappa_scale = experts._materialize_kappa_scale_for_eval(
            gate_base.dtype,
            gate_base.device,
        )
        router_score = expert_router_scores.unsqueeze(-1)
        if experts.kappa_input in {"top_logits", "router_probs"}:
            slope_input = kappa_bias.unsqueeze(1) + router_score * kappa_scale.unsqueeze(1)
        else:
            slope_input = router_score * kappa_bias.unsqueeze(1)
        max_scale = experts.kappa_slope_max_scale.to(device=gate_base.device, dtype=gate_base.dtype)
        slope_scale = torch.exp(torch.log(max_scale) * torch.tanh(slope_input))
        gate_base_acts = experts._apply_gate_activation(gate_base)
        gate_biased_acts = gate_base * torch.sigmoid(gate_base * slope_scale)
        delta_gate = gate_biased_acts - gate_base_acts
        delta_gate = delta_gate.float()
        gate_base_acts = gate_base_acts.float()

        slot_mask = expert_slot_mask.unsqueeze(-1)
        self.delta_stats.observe(delta_gate, slot_mask)
        if self.delta_sampler is not None:
            sampled_delta = self._sample_valid_slot_values(delta_gate, expert_slot_mask)
            if sampled_delta is not None:
                sampled_delta_mask = torch.ones_like(sampled_delta, dtype=torch.bool)
                self.delta_sampler.observe(sampled_delta, sampled_delta_mask)
        threshold_mask = None
        if self.top10_delta_stats is not None:
            threshold_mask = torch.logical_and(slot_mask, delta_gate >= self.top10_delta_threshold)
            self.top10_delta_stats.observe(delta_gate, threshold_mask)
        if self.top2_delta_stats is not None:
            threshold_mask = torch.logical_and(slot_mask, delta_gate >= self.top2_delta_threshold)
            self.top2_delta_stats.observe(delta_gate, threshold_mask)
        del threshold_mask

        safe_gate_base_acts = torch.where(
            gate_base_acts >= 0,
            gate_base_acts.clamp_min(RELATIVE_DENOM_EPS),
            gate_base_acts.clamp_max(-RELATIVE_DENOM_EPS),
        )
        # relative_delta = 0.1 means the bias increased that gate activation by roughly 10%, 
        # while -0.1 means it decreased it by roughly 10%.
        relative_delta = delta_gate / safe_gate_base_acts
        relative_mask = torch.logical_and(slot_mask, torch.isfinite(relative_delta))
        self.relative_stats.observe(relative_delta, relative_mask)
        # relative_sampler: This is reservoir-style sampling: it approximates a uniform sample 
        # without storing every activation. The collected sample is then used to
        # estimate the global 90th- and 98th-percentile thresholds
        if self.relative_sampler is not None:
            sampled_relative = self._sample_valid_slot_values(relative_delta, expert_slot_mask)
            if sampled_relative is not None:
                sampled_relative = sampled_relative[torch.isfinite(sampled_relative)]
                if sampled_relative.numel() > 0:
                    sampled_relative_mask = torch.ones_like(sampled_relative, dtype=torch.bool)
                    self.relative_sampler.observe(sampled_relative, sampled_relative_mask)
        threshold_mask = None
        if self.top10_relative_stats is not None:
            threshold_mask = torch.logical_and(relative_mask, relative_delta >= self.top10_relative_threshold)
            self.top10_relative_stats.observe(relative_delta, threshold_mask)
        if self.top2_relative_stats is not None:
            threshold_mask = torch.logical_and(relative_mask, relative_delta >= self.top2_relative_threshold)
            self.top2_relative_stats.observe(relative_delta, threshold_mask)
        del threshold_mask

    def reduce(self):
        if self.collect_activation_stats:
            self.delta_stats.reduce()
            self.relative_stats.reduce()
            if self.top10_delta_stats is not None:
                self.top10_delta_stats.reduce()
            if self.top10_relative_stats is not None:
                self.top10_relative_stats.reduce()
            if self.top2_delta_stats is not None:
                self.top2_delta_stats.reduce()
            if self.top2_relative_stats is not None:
                self.top2_relative_stats.reduce()
        self.normalized_load_entropy_stats.reduce()
        self.load_cv_stats.reduce()
        self.router_probability_entropy_stats.reduce()
        if dist.is_initialized() and self.expert_utilization_sum is not None:
            dist.all_reduce(self.expert_utilization_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.routing_observation_count, op=dist.ReduceOp.SUM)
        for layer_stats in self.routing_stats_by_layer.values():
            layer_stats["normalized_load_entropy"].reduce()
            layer_stats["load_cv"].reduce()
            layer_stats["router_probability_entropy"].reduce()
            if dist.is_initialized():
                dist.all_reduce(layer_stats["expert_utilization_sum"], op=dist.ReduceOp.SUM)
                dist.all_reduce(layer_stats["observation_count"], op=dist.ReduceOp.SUM)

    def summary(self):
        if self.expert_utilization_sum is None:
            raise RuntimeError("Collected zero routing observations; no summary can be computed")
        summary = {}
        load_entropy_summary = self.normalized_load_entropy_stats.summary("normalized_load_entropy")
        load_cv_summary = self.load_cv_stats.summary("load_cv")
        router_entropy_summary = self.router_probability_entropy_stats.summary("router_probability_entropy")
        summary.update({
            "mean_normalized_load_entropy": load_entropy_summary["mean(normalized_load_entropy)"],
            "mean_load_cv": load_cv_summary["mean(load_cv)"],
            "mean_router_probability_entropy": router_entropy_summary["mean(router_probability_entropy)"],
            "mean_expert_assignment_fractions": (
                self.expert_utilization_sum / self.routing_observation_count
            ).tolist(),
        })
        routing_by_layer = {}
        for layer_idx, layer_stats in self.routing_stats_by_layer.items():
            layer_load_entropy = layer_stats["normalized_load_entropy"].summary("normalized_load_entropy")
            layer_load_cv = layer_stats["load_cv"].summary("load_cv")
            layer_router_entropy = layer_stats["router_probability_entropy"].summary("router_probability_entropy")
            routing_by_layer[layer_idx] = {
                "mean_normalized_load_entropy": layer_load_entropy["mean(normalized_load_entropy)"],
                "mean_load_cv": layer_load_cv["mean(load_cv)"],
                "mean_router_probability_entropy": layer_router_entropy["mean(router_probability_entropy)"],
                "mean_expert_assignment_fractions": (
                    layer_stats["expert_utilization_sum"] / layer_stats["observation_count"]
                ).tolist(),
            }
        summary["routing_by_layer"] = routing_by_layer
        if not self.collect_activation_stats:
            return summary

        delta_summary = self.delta_stats.summary("delta_gate")
        if delta_summary["count_delta_gate"] == 0:
            raise RuntimeError("Collected zero valid gate activations; no summary can be computed")
        summary.update({
            "mean(delta_gate)": delta_summary["mean(delta_gate)"],
            "mean(abs(delta_gate))": delta_summary["mean(abs(delta_gate))"],
            "rms(delta_gate)": delta_summary["rms(delta_gate)"],
            "fraction_positive": delta_summary["fraction_positive_delta_gate"],
            "count": delta_summary["count_delta_gate"],
        })
        relative_summary = self.relative_stats.summary("delta_gate / silu(g_base)")
        summary.update({
            "mean(delta_gate / silu(g_base))": relative_summary["mean(delta_gate / silu(g_base))"],
            "mean(abs(delta_gate / silu(g_base)))": relative_summary["mean(abs(delta_gate / silu(g_base)))"],
            "rms(delta_gate / silu(g_base))": relative_summary["rms(delta_gate / silu(g_base))"],
            "fraction_positive_relative": relative_summary["fraction_positive_delta_gate / silu(g_base)"],
            "relative_count": relative_summary["count_delta_gate / silu(g_base)"],
        })
        if self.top10_delta_stats is not None:
            top_delta_summary = self.top10_delta_stats.summary("top10_delta_gate")
            summary.update({
                "mean(top10_delta_gate)": top_delta_summary["mean(top10_delta_gate)"],
                "mean(abs(top10_delta_gate))": top_delta_summary["mean(abs(top10_delta_gate))"],
                "rms(top10_delta_gate)": top_delta_summary["rms(top10_delta_gate)"],
                "fraction_positive_top10_delta_gate": top_delta_summary["fraction_positive_top10_delta_gate"],
                "count_top10_delta_gate": top_delta_summary["count_top10_delta_gate"],
            })
        if self.top10_relative_stats is not None:
            top_relative_summary = self.top10_relative_stats.summary("top10_delta_gate / silu(g_base)")
            summary.update({
                "mean(top10_delta_gate / silu(g_base))": top_relative_summary["mean(top10_delta_gate / silu(g_base))"],
                "mean(abs(top10_delta_gate / silu(g_base)))": top_relative_summary["mean(abs(top10_delta_gate / silu(g_base)))"],
                "rms(top10_delta_gate / silu(g_base))": top_relative_summary["rms(top10_delta_gate / silu(g_base))"],
                "fraction_positive_top10_relative": top_relative_summary["fraction_positive_top10_delta_gate / silu(g_base)"],
                "count_top10_relative": top_relative_summary["count_top10_delta_gate / silu(g_base)"],
            })
        if self.top2_delta_stats is not None:
            top2_delta_summary = self.top2_delta_stats.summary("top2_delta_gate")
            summary.update({
                "mean(top2_delta_gate)": top2_delta_summary["mean(top2_delta_gate)"],
                "mean(abs(top2_delta_gate))": top2_delta_summary["mean(abs(top2_delta_gate))"],
                "rms(top2_delta_gate)": top2_delta_summary["rms(top2_delta_gate)"],
                "fraction_positive_top2_delta_gate": top2_delta_summary["fraction_positive_top2_delta_gate"],
                "count_top2_delta_gate": top2_delta_summary["count_top2_delta_gate"],
            })
        if self.top2_relative_stats is not None:
            top2_relative_summary = self.top2_relative_stats.summary("top2_delta_gate / silu(g_base)")
            summary.update({
                "mean(top2_delta_gate / silu(g_base))": top2_relative_summary["mean(top2_delta_gate / silu(g_base))"],
                "mean(abs(top2_delta_gate / silu(g_base)))": top2_relative_summary["mean(abs(top2_delta_gate / silu(g_base)))"],
                "rms(top2_delta_gate / silu(g_base))": top2_relative_summary["rms(top2_delta_gate / silu(g_base))"],
                "fraction_positive_top2_relative": top2_relative_summary["fraction_positive_top2_delta_gate / silu(g_base)"],
                "count_top2_relative": top2_relative_summary["count_top2_delta_gate / silu(g_base)"],
            })
        return summary


class GateBiasObserverHost:
    def __init__(self):
        self.collector = None

    def set_collector(self, collector: GateBiasStatsCollector):
        self.collector = collector

    @torch._dynamo.disable
    @torch.inference_mode()
    def observe(
        self,
        layer: MOELayer,
        expert_inputs: torch.Tensor,
        expert_router_scores: torch.Tensor,
        expert_slot_mask: torch.Tensor,
        top_k_indices: torch.Tensor,
        all_router_probs: torch.Tensor,
    ):
        if self.collector is None:
            raise RuntimeError("Gate bias observer host was called without an active collector")
        self.collector.observe(
            layer,
            expert_inputs,
            expert_router_scores,
            expert_slot_mask,
            top_k_indices,
            all_router_probs,
        )


def install_gate_bias_instrumentation(model, observer_host: GateBiasObserverHost, routing_only=False):
    instrumented_layers = 0

    def instrumented_forward(self, x: torch.Tensor):
        B, T, C = x.size()

        expert_mask, router_probs, top_k_scores, top_k_indices, rank = self.router(x)
        del expert_mask

        x_flat = x.view(B * T, C)
        all_router_probs = F.softmax(F.linear(x_flat.float(), self.router.w_g.weight.float()), dim=-1)
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
        selected_gate_confidence = self._select_gate_confidence(
            top_k_scores,
            router_probs,
            x_flat=x_flat,
            top_k_indices=top_k_indices,
        )
        expert_router_scores = torch.zeros(
            self.n_exp,
            exp_capacity,
            dtype=selected_gate_confidence.dtype,
            device=selected_gate_confidence.device,
        )
        self._build_expert_inputs(
            x_flat,
            flat_rank,
            exp_capacity,
            flat_token_indices,
            flat_top_k_indices,
            selected_gate_confidence.view(-1),
            expert_inputs,
            expert_router_scores,
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
        if self.use_qwen3_moe_mlp and (
            routing_only or getattr(self.experts, "use_kappa_swiglu", False)
        ):
            observer_host.observe(
                self,
                expert_inputs,
                expert_router_scores,
                expert_slot_mask,
                top_k_indices,
                all_router_probs,
            )

        expert_outputs = self.experts(
            expert_inputs,
            selected_router_scores=expert_router_scores,
            router_weight=self.router.w_g.weight,
        )
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
        if not routing_only and not getattr(module.experts, "use_kappa_swiglu", False):
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
    last_log_time = time.perf_counter()
    with torch.inference_mode():
        for step_idx in range(eval_steps):
            x, y = next(batch_iter)
            with autocast_ctx:
                model(x, y, loss_reduction="none")
            if step_idx == 0 or (step_idx + 1) % 50 == 0 or step_idx + 1 == eval_steps:
                now = time.perf_counter()
                elapsed_since_last_log = now - last_log_time
                print0(
                    f"{pass_name}: processed {step_idx + 1}/{eval_steps} eval steps "
                    f"({elapsed_since_last_log:.2f}s)"
                )
                last_log_time = now


def print_routing_summary(summary):
    print0("Routing summary:")
    print0(f"mean_normalized_load_entropy: {summary['mean_normalized_load_entropy']:.4f}")
    print0(f"mean_load_cv: {summary['mean_load_cv']:.4f}")
    print0(f"mean_router_probability_entropy: {summary['mean_router_probability_entropy']:.4f}")
    print0(
        "mean_expert_assignment_fractions: "
        + ", ".join(f"{fraction:.4f}" for fraction in summary["mean_expert_assignment_fractions"])
    )
    print0("Routing summary by MoE layer:")
    for layer_idx, layer_summary in sorted(summary["routing_by_layer"].items()):
        fractions = ", ".join(
            f"{fraction:.4f}" for fraction in layer_summary["mean_expert_assignment_fractions"]
        )
        print0(
            f"layer {layer_idx}: normalized_load_entropy={layer_summary['mean_normalized_load_entropy']:.4f} "
            f"load_cv={layer_summary['mean_load_cv']:.4f} "
            f"router_probability_entropy={layer_summary['mean_router_probability_entropy']:.4f} "
            f"expert_assignment_fractions=[{fractions}]"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate expert gate bias activation effects")
    parser.add_argument("--source", type=str, default="base", choices=["base", "sft", "rl"], help="checkpoint family to load")
    parser.add_argument("--model-tag", type=str, default=None, help="checkpoint directory tag")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to load (default: latest)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="data split to evaluate")
    parser.add_argument("--eval-tokens", type=int, default=40 * 524288, help="target token budget, matching base_train.py by default")
    parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
    parser.add_argument("--eval-capacity", type=float, default=None, help="override MoE eval capacity")
    parser.add_argument(
        "--kappa-bias-fill-value",
        type=float,
        default=None,
        help="override all expert kappa_bias tensors in the loaded checkpoint with this constant value",
    )
    parser.add_argument(
        "--kappa-scale-fill-value",
        type=float,
        default=None,
        help="override all kappa_scale tensors in the loaded checkpoint with this constant value",
    )
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the instrumented model for faster inference",
    )
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="run one pass and report routing statistics without activation quantiles or deltas",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    del ddp, ddp_local_rank
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=COMPUTE_DTYPE) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
        eval_capacity=args.eval_capacity,
        kappa_bias_fill_value=args.kappa_bias_fill_value,
        kappa_scale_fill_value=args.kappa_scale_fill_value,
    )
    model.eval()

    observer_host = GateBiasObserverHost()
    instrumented_layers = install_gate_bias_instrumentation(model, observer_host, routing_only=args.routing_only)
    if instrumented_layers == 0:
        if args.routing_only:
            raise RuntimeError("No Qwen3 MoE layers were found in the loaded model")
        raise RuntimeError("No MoE expert layers with kappa_bias were found in the loaded model")
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
    print0(f"Instrumented MoE layers: {instrumented_layers} | compile: {args.compile}")
    if device_type == "cuda":
        print0(f"CUDA compute dtype: {COMPUTE_DTYPE}")

    if args.routing_only:
        routing_collector = GateBiasStatsCollector(collect_activation_stats=False)
        routing_collector.initialize_stats(None, None, None, None)
        observer_host.set_collector(routing_collector)
        routing_loader = build_loader(tokenizer, args.device_batch_size, sequence_len, args.split, device)
        run_eval_pass(model, routing_loader, eval_steps, autocast_ctx, pass_name="Routing pass")
        routing_collector.reduce()
        print_routing_summary(routing_collector.summary())
        compute_cleanup()
        return

    # Two-pass approach: first pass collects quantile samples to 
    # determine cutoff thresholds, second pass collects stats and summaries based on 
    # those cutoff thresholds
    sampling_collector = GateBiasStatsCollector()
    sampling_collector.initialize_stats(None, None, None, None)
    sampling_collector.enable_sampling(QUANTILE_SAMPLE_SIZE)
    observer_host.set_collector(sampling_collector)
    sampling_loader = build_loader(tokenizer, args.device_batch_size, sequence_len, args.split, device)
    run_eval_pass(model, sampling_loader, eval_steps, autocast_ctx, pass_name="Sampling pass")

    top10_delta_threshold = compute_global_quantile(sampling_collector.delta_sampler.get_samples(), TOP10_QUANTILE)
    top10_relative_threshold = compute_global_quantile(sampling_collector.relative_sampler.get_samples(), TOP10_QUANTILE)
    top2_delta_threshold = compute_global_quantile(sampling_collector.delta_sampler.get_samples(), TOP2_QUANTILE)
    top2_relative_threshold = compute_global_quantile(sampling_collector.relative_sampler.get_samples(), TOP2_QUANTILE)
    print0(f"Top {TOP10_FRACTION:.0%} delta_gate threshold: {top10_delta_threshold:.3e}")
    print0(f"Top {TOP10_FRACTION:.0%} relative delta_gate threshold: {top10_relative_threshold:.3e}")
    print0(f"Top {TOP2_FRACTION:.0%} delta_gate threshold: {top2_delta_threshold:.3e}")
    print0(f"Top {TOP2_FRACTION:.0%} relative delta_gate threshold: {top2_relative_threshold:.3e}")

    summary_collector = GateBiasStatsCollector()
    summary_collector.initialize_stats(
        top10_delta_threshold,
        top10_relative_threshold,
        top2_delta_threshold,
        top2_relative_threshold,
    )
    observer_host.set_collector(summary_collector)
    summary_loader = build_loader(tokenizer, args.device_batch_size, sequence_len, args.split, device)
    run_eval_pass(model, summary_loader, eval_steps, autocast_ctx, pass_name="Summary pass")

    summary_collector.reduce()
    summary = summary_collector.summary()
    print_routing_summary(summary)
    
    print0()

    print0("Gate bias activation delta summary:")
    # count is not a token count. It is the number of valid scalar gate activations included in delta_gate statistics
    # It's a huge number, for example 1546188226560.
    print0(f"Gate activation count: {int(summary['count'])}")
    print0(f"mean(delta_gate): {summary['mean(delta_gate)']:.3e}")
    print0(f"mean(abs(delta_gate)): {summary['mean(abs(delta_gate))']:.3e}")
    print0(f"rms(delta_gate): {summary['rms(delta_gate)']:.3e}")
    print0(f"fraction_positive: {summary['fraction_positive']:.3e}")
    print0(f"mean(delta_gate / silu(g_base)): {summary['mean(delta_gate / silu(g_base))']:.3e}")
    print0(f"mean(abs(delta_gate / silu(g_base))): {summary['mean(abs(delta_gate / silu(g_base)))']:.3e}")
    print0(f"rms(delta_gate / silu(g_base)): {summary['rms(delta_gate / silu(g_base))']:.3e}")
    print0(f"fraction_positive_relative: {summary['fraction_positive_relative']:.3e}")
    print0(f"relative_count: {int(summary['relative_count'])}")
    print0(f"mean(top10_delta_gate): {summary['mean(top10_delta_gate)']:.3e}")
    print0(f"mean(abs(top10_delta_gate)): {summary['mean(abs(top10_delta_gate))']:.3e}")
    print0(f"rms(top10_delta_gate): {summary['rms(top10_delta_gate)']:.3e}")
    print0(f"fraction_positive_top10_delta_gate: {summary['fraction_positive_top10_delta_gate']:.3e}")
    print0(f"count_top10_delta_gate: {int(summary['count_top10_delta_gate'])}")
    print0(f"mean(top10_delta_gate / silu(g_base)): {summary['mean(top10_delta_gate / silu(g_base))']:.3e}")
    print0(f"mean(abs(top10_delta_gate / silu(g_base))): {summary['mean(abs(top10_delta_gate / silu(g_base)))']:.3e}")
    print0(f"rms(top10_delta_gate / silu(g_base)): {summary['rms(top10_delta_gate / silu(g_base))']:.3e}")
    print0(f"fraction_positive_top10_relative: {summary['fraction_positive_top10_relative']:.3e}")
    print0(f"count_top10_relative: {int(summary['count_top10_relative'])}")
    print0(f"mean(top2_delta_gate): {summary['mean(top2_delta_gate)']:.3e}")
    print0(f"mean(abs(top2_delta_gate)): {summary['mean(abs(top2_delta_gate))']:.3e}")
    print0(f"rms(top2_delta_gate): {summary['rms(top2_delta_gate)']:.3e}")
    print0(f"fraction_positive_top2_delta_gate: {summary['fraction_positive_top2_delta_gate']:.3e}")
    print0(f"count_top2_delta_gate: {int(summary['count_top2_delta_gate'])}")
    print0(f"mean(top2_delta_gate / silu(g_base)): {summary['mean(top2_delta_gate / silu(g_base))']:.3e}")
    print0(f"mean(abs(top2_delta_gate / silu(g_base))): {summary['mean(abs(top2_delta_gate / silu(g_base)))']:.3e}")
    print0(f"rms(top2_delta_gate / silu(g_base)): {summary['rms(top2_delta_gate / silu(g_base))']:.3e}")
    print0(f"fraction_positive_top2_relative: {summary['fraction_positive_top2_relative']:.3e}")
    print0(f"count_top2_relative: {int(summary['count_top2_relative'])}")

    compute_cleanup()


if __name__ == "__main__":
    main()