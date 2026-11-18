"""Measure token-wise MLP latent norms with or without trained kappa SwiGLU.

Run with the same checkpoint and validation data used for base evaluation:
    python -m scripts.check_kappa_latent_norms --model-tag d24 --eval-tokens 65536
    torchrun --nproc_per_node=8 -m scripts.check_kappa_latent_norms --model-tag d24

CV (population std / mean) measures relative norm spread; smaller is more uniform.
Local pairs hold each layer's input and routing fixed. Full-model pairs also
include upstream activation and routing changes. Expert outputs are measured
before routing weights and aggregation; padding slots are excluded. This is an
inference ablation of one trained checkpoint, not a comparison of training runs.
Non-kappa layers are included. Checkpoints without kappa run one baseline pass.
By default, input/output refer to the MLP boundaries. --norm-location swiglu
measures input [gate_proj(x), up_proj(x)] and output activation(gate) * up,
before the down-projection (after output RMSNorm when configured). The input
norm combines both projection branches.
"""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_model
from nanochat.common import COMPUTE_DTYPE, autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.gpt import Qwen3MLP, Qwen3MLPExperts, norm
from scripts.eval_kappa_effect import build_loader


class ScalarStats:
    def __init__(self):
        self.count = 0
        self.nonfinite = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = None
        self.maximum = None

    def observe(self, values):
        values = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        finite = torch.isfinite(values)
        self.nonfinite += int((~finite).sum())
        values = values[finite]
        if values.numel() == 0:
            return
        variance, mean = torch.var_mean(values, unbiased=False)
        other = ScalarStats()
        other.count = values.numel()
        other.mean = mean.item()
        other.m2 = variance.item() * other.count
        other.minimum = values.min().item()
        other.maximum = values.max().item()
        self.merge(other)

    def merge(self, other):
        self.nonfinite += other.nonfinite
        if other.count == 0:
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total
        self.minimum = other.minimum if self.minimum is None else min(self.minimum, other.minimum)
        self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)

    def summary(self):
        std = math.sqrt(max(0.0, self.m2 / self.count)) if self.count else None
        return dict(
            count=self.count,
            nonfinite=self.nonfinite,
            mean=self.mean if self.count else None,
            std=std,
            cv=std / self.mean if self.count and self.mean > 0 else None,
            min=self.minimum,
            max=self.maximum,
        )


class LatentNormProbe:
    def __init__(self, per_expert=False, norm_location="mlp"):
        if norm_location not in {"mlp", "swiglu"}:
            raise ValueError(f"Unsupported norm location: {norm_location!r}")
        self.per_expert = per_expert
        self.norm_location = norm_location
        self.mode = "enabled"
        self.stats = {}
        self.handles = []
        self.layer_kappa_enabled = {}
        self.activations = {}
        self.patched_methods = []

    def capture_projection(self, owner, branch, output):
        values = self.activations.setdefault(owner, {})
        values[branch] = output.detach().float().square().sum(dim=-1)

    def install_swiglu_capture(self, module):
        if isinstance(module, Qwen3MLPExperts):
            if module.bilinear_mlp_moe:
                raise ValueError("SwiGLU norms require SwiGLU experts, not bilinear_mlp_moe")
            for method_name in ("_apply_gate_activation", "_apply_kappa_slope_scaled_activation"):
                original = getattr(module, method_name)
                previous = module.__dict__.get(method_name)
                self.patched_methods.append((module, method_name, previous))

                def capture(current, gate_raw, *args, original=original, **kwargs):
                    activated = original(gate_raw, *args, **kwargs)
                    self.capture_projection(current, "gate", gate_raw)
                    self.activations[current]["activated_gate"] = activated.detach()
                    return activated

                setattr(module, method_name, MethodType(capture, module))
        else:
            def gate_hook(current, args, output):
                self.capture_projection(module, "gate", output)

            def up_hook(current, args, output):
                self.capture_projection(module, "up", output)

            def product_hook(current, args):
                self.activations.setdefault(module, {})["product"] = args[0].detach()

            self.handles.append(module.gate_proj.register_forward_hook(gate_hook))
            self.handles.append(module.c_fc.register_forward_hook(up_hook))
            self.handles.append(module.c_proj.register_forward_pre_hook(product_hook))

    def measurement(self, module, inputs, output):
        if self.norm_location == "mlp":
            return inputs.float().norm(dim=-1), output
        values = self.activations.pop(module)
        if isinstance(module, Qwen3MLPExperts):
            up = torch.bmm(inputs, module.c_fc)
            values["up"] = up.float().square().sum(dim=-1)
            output = values["activated_gate"] * up
            if module.swiglu_norm_output:
                output = norm(output)
        else:
            output = values["product"]
        return (values["gate"] + values["up"]).sqrt(), output

    def observe(self, name, metric, values):
        key = f"{name}/{metric}"
        self.stats.setdefault(key, ScalarStats()).observe(values)

    @torch.inference_mode()
    def hook(self, name, module, args, kwargs, output):
        inputs = args[0]
        is_expert = isinstance(module, Qwen3MLPExperts)
        mask_name = "valid_score_mask" if is_expert else "valid_token_mask"
        mask = kwargs.get(mask_name)
        if mask is None:
            if is_expert:
                raise RuntimeError("Expert validity mask is required to exclude dispatch padding")
            mask = torch.ones(inputs.shape[:-1], dtype=torch.bool, device=inputs.device)
        mask = torch.broadcast_to(mask.bool(), inputs.shape[:-1])
        input_norm, output = self.measurement(module, inputs, output)
        output_norm = output.float().norm(dim=-1)
        groups = [(name, mask)]
        if is_expert and self.per_expert:
            for expert_idx in range(inputs.size(0)):
                expert_mask = torch.zeros_like(mask)
                expert_mask[expert_idx] = mask[expert_idx]
                groups.append((f"{name}.expert{expert_idx}", expert_mask))
        for group, group_mask in groups:
            self.layer_kappa_enabled[group] = self.layer_kappa_enabled.get(name, module.kappa_swiglu_enabled)
            self.observe(group, f"{self.mode}/input_norm", input_norm[group_mask])
            self.observe(group, f"{self.mode}/output_norm", output_norm[group_mask])
            nonzero_input = group_mask & (input_norm > 0)
            self.observe(group, f"{self.mode}/gain", output_norm[nonzero_input] / input_norm[nonzero_input])
        if self.mode != "enabled" or not module.kappa_swiglu_enabled:
            return
        previous = module.kappa_swiglu_enabled
        try:
            module.set_kappa_swiglu_enabled(False)
            baseline = module.forward(*args, **kwargs)
            _, baseline = self.measurement(module, inputs, baseline)
        finally:
            module.set_kappa_swiglu_enabled(previous)
        baseline_norm = baseline.float().norm(dim=-1)
        valid_direction = (output_norm > 0) & (baseline_norm > 0)
        tiny = torch.finfo(torch.float32).tiny
        cosine = torch.nn.functional.cosine_similarity(output.float(), baseline.float(), dim=-1, eps=tiny)
        for group, group_mask in groups:
            self.observe(group, "local_disabled/output_norm", baseline_norm[group_mask])
            nonzero_input = group_mask & (input_norm > 0)
            self.observe(group, "local_enabled/gain", output_norm[nonzero_input] / input_norm[nonzero_input])
            self.observe(group, "local_disabled/gain", baseline_norm[nonzero_input] / input_norm[nonzero_input])
            self.observe(group, "local/output_cosine", cosine[group_mask & valid_direction])
            self.observe(group, "local/zero_direction_pairs", (~valid_direction[group_mask]).float())

    def install(self, model):
        names = []
        for name, module in model.named_modules():
            if not isinstance(module, (Qwen3MLP, Qwen3MLPExperts)):
                continue
            if self.norm_location == "swiglu":
                self.install_swiglu_capture(module)
            self.layer_kappa_enabled[name] = module.kappa_swiglu_enabled
            def callback(current, args, kwargs, output, name=name):
                self.hook(name, current, args, kwargs, output)
            self.handles.append(module.register_forward_hook(callback, with_kwargs=True))
            names.append(name)
        return names

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, method_name, previous in reversed(self.patched_methods):
            if previous is None:
                delattr(module, method_name)
            else:
                setattr(module, method_name, previous)
        self.patched_methods.clear()
        self.activations.clear()

    def summaries(self):
        if dist.is_initialized():
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, self.stats)
            merged = {}
            for part in parts:
                for key, stats in part.items():
                    merged.setdefault(key, ScalarStats()).merge(stats)
        else:
            merged = self.stats
        return {key: stats.summary() for key, stats in sorted(merged.items())}


def format_number(value):
    return "n/a" if value is None else f"{value:.5g}"


def summarize_experts(names, summaries):
    grouped = {}
    for key, stats in summaries.items():
        group, metric = key.split("/", 1)
        layer, separator, expert_idx = group.rpartition(".expert")
        if separator and expert_idx.isdigit() and layer in names:
            grouped.setdefault(f"{layer}/{metric}", []).append(stats)
    result = {}
    for key, experts in sorted(grouped.items()):
        aggregate = {
            "expert_count": len(experts),
            "observed_experts": sum(expert["count"] > 0 for expert in experts),
        }
        for field in ("mean", "std", "cv"):
            values = [expert[field] for expert in experts if expert[field] is not None]
            stats = ScalarStats()
            stats.observe(torch.tensor(values, dtype=torch.float64))
            aggregate[field] = stats.summary()
        result[key] = aggregate
    return result


def print_comparisons(names, summaries, expert_summaries=None, norm_location="mlp"):
    expert_summaries = expert_summaries or {}

    def comparison_stats(key):
        if key in expert_summaries:
            return {field: expert_summaries[key][field]["mean"] for field in ("mean", "std", "cv")}
        return summaries[key]

    print0("CV: lower means more uniform across tokens/valid expert assignments.")
    if norm_location == "swiglu":
        print0("SwiGLU norms: input=[gate_proj(x), up_proj(x)], output=activation(gate) * up before down-projection (after optional RMSNorm).")
    else:
        print0("Norms are at the MLP input and final projected output, not internal SwiGLU boundaries.")
    print0("MoE: each column is mean +/- population std across observed experts (equal weight).")
    print0("The std/CV columns summarize within-expert token variation; empty experts are excluded.")
    print0("Layer | mode | metric | mean | std | CV")
    for name in names:
        modes = ("baseline",) if f"{name}/baseline/input_norm" in summaries else ("disabled", "enabled")
        for mode in modes:
            for metric in ("input_norm", "output_norm", "gain"):
                key = f"{name}/{mode}/{metric}"
                if key in expert_summaries:
                    stats = expert_summaries[key]
                    columns = [
                        f"{format_number(stats[field]['mean'])} +/- {format_number(stats[field]['std'])}"
                        for field in ("mean", "std", "cv")
                    ]
                    print0(
                        f"{name} | {mode} | {metric} | {' | '.join(columns)} | "
                        f"experts {stats['observed_experts']}/{stats['expert_count']}"
                    )
                else:
                    stats = summaries[key]
                    print0(
                        f"{name} | {mode} | {metric} | {format_number(stats['mean'])} | "
                        f"{format_number(stats['std'])} | {format_number(stats['cv'])}"
                    )
        if modes == ("baseline",):
            continue
        has_local_kappa = f"{name}/local_disabled/output_norm" in summaries
        comparisons = [
            ("full input", "disabled/input_norm", "enabled/input_norm"),
            ("full output", "disabled/output_norm", "enabled/output_norm"),
        ]
        if has_local_kappa:
            comparisons = [
                ("local output", "local_disabled/output_norm", "enabled/output_norm"),
                ("local gain", "local_disabled/gain", "local_enabled/gain"),
            ] + comparisons
        else:
            print0(f"  {name}: no local kappa; full-model differences reflect upstream changes only.")
        for label, off_metric, on_metric in comparisons:
            off = comparison_stats(f"{name}/{off_metric}")
            on = comparison_stats(f"{name}/{on_metric}")
            change = 100 * (on["cv"] / off["cv"] - 1) if off["cv"] and on["cv"] is not None else None
            print0(
                f"  {label}: mean off -> on {format_number(off['mean'])} -> {format_number(on['mean'])}; "
                f"CV {format_number(off['cv'])} -> {format_number(on['cv'])}; change (%) {format_number(change)}"
            )
        if has_local_kappa:
            key = f"{name}/local/output_cosine"
            if key in expert_summaries:
                cosine = expert_summaries[key]["mean"]
                print0(f"  Local output cosine: expert mean +/- std={format_number(cosine['mean'])} +/- {format_number(cosine['std'])}")
            else:
                cosine = summaries[key]
                print0(f"  Local output cosine: mean={format_number(cosine['mean'])}, min={format_number(cosine['min'])}")
    print0("Enabled/disabled refer to model-wide kappa. Local comparisons hold inputs/routes fixed.")
    print0("A mean-norm change alone is rescaling, not uniformization (CV is scale invariant).")
    print0("Cosine below 1 means kappa also changes direction, not just output norm.")
    print0("These diagnostics cannot establish that norm uniformization is kappa's only role.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["base", "sft", "rl"], default="base")
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--eval-tokens", type=int, default=524288)
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--eval-capacity", type=float, default=None)
    parser.add_argument("--device-type", choices=["", "cuda", "cpu", "mps"], default="")
    parser.add_argument("--norm-location", choices=["mlp", "swiglu"], default="mlp", help="measure MLP boundaries (default) or concatenated gate/up inputs and gated-product output of SwiGLU")
    parser.add_argument("--per-expert", action="store_true", help="retain individual expert statistics in JSON; console always summarizes each layer")
    parser.add_argument("--output", type=Path, default=None, help="write full statistics as JSON on rank 0")
    args = parser.parse_args()
    if args.eval_tokens <= 0 or args.device_batch_size <= 0:
        parser.error("--eval-tokens and --device-batch-size must be positive")
    return args


def main():
    args = parse_args()
    device_type = args.device_type or autodetect_device_type()
    _, rank, _, world_size, device = compute_init(device_type)
    probe = LatentNormProbe(per_expert=True, norm_location=args.norm_location)
    states = []
    try:
        model, tokenizer, meta = load_model(
            args.source, device, phase="eval", model_tag=args.model_tag,
            step=args.step, eval_capacity=args.eval_capacity,
        )
        model.eval()
        states = [(module, module.kappa_swiglu_enabled) for module in model.modules()
                  if isinstance(module, (Qwen3MLP, Qwen3MLPExperts))]
        names = probe.install(model)
        if not names:
            raise RuntimeError("Checkpoint has no supported Qwen3 dense or expert MLP layers")
        has_kappa = any(enabled for _, enabled in states)
        modes = ("enabled", "disabled") if has_kappa else ("baseline",)
        sequence_len = meta["model_config"]["sequence_len"]
        tokens_per_step = args.device_batch_size * sequence_len * world_size
        steps = max(1, math.ceil(args.eval_tokens / tokens_per_step))
        print0(f"Checkpoint step: {meta['step']} | tokens per mode: {steps * tokens_per_step:,}")
        if has_kappa:
            print0("Eager inference; each batch is evaluated enabled then disabled, without reloading data.")
        else:
            print0("No enabled kappa: one baseline pass, with no synthetic on/off comparison.")
        loader = iter(build_loader(tokenizer, args.device_batch_size, sequence_len, args.split, device))
        autocast = torch.amp.autocast(device_type="cuda", dtype=COMPUTE_DTYPE) if device_type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            for step in range(steps):
                inputs, targets = next(loader)
                for mode in modes:
                    probe.mode = mode
                    for module, originally_enabled in states:
                        module.set_kappa_swiglu_enabled(originally_enabled and mode == "enabled")
                    _, losses = model(inputs, targets)
                    probe.observe("model", f"{mode}/nll", losses["ntp_loss"])
                if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
                    print0(f"Processed {step + 1}/{steps} batches ({', '.join(modes)})")
        summaries = probe.summaries()
        expert_summaries = summarize_experts(names, summaries)
        print_comparisons(names, summaries, expert_summaries, args.norm_location)
        for mode in modes:
            print0(f"{mode} validation NLL: {format_number(summaries[f'model/{mode}/nll']['mean'])}")
        nonfinite = sum(stats["nonfinite"] for stats in summaries.values())
        if nonfinite:
            print0(f"WARNING: {nonfinite} nonfinite observations excluded; inspect JSON counts.")
        if args.output is not None and rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_summaries = summaries
            layer_kappa_enabled = probe.layer_kappa_enabled
            if not args.per_expert:
                output_summaries = {key: stats for key, stats in summaries.items() if key.split("/", 1)[0] in names or key.startswith("model/")}
                layer_kappa_enabled = {name: enabled for name, enabled in layer_kappa_enabled.items() if name in names}
            report = dict(
                source=args.source, model_tag=args.model_tag, step=meta["step"], split=args.split,
                tokens_per_mode=steps * tokens_per_step, world_size=world_size, modes=modes,
                layer_kappa_enabled=layer_kappa_enabled, statistics=output_summaries,
                expert_summary=expert_summaries,
                norm_location=args.norm_location,
                norm_definition=(
                    {"input": "concat(gate_proj(x), up_proj(x))", "output": "activation(gate_proj(x)) * up_proj(x), after RMSNorm if configured"}
                    if args.norm_location == "swiglu" else {"input": "MLP input", "output": "MLP output after down-projection"}
                ),
            )
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print0(f"Wrote {args.output}")
    finally:
        probe.remove()
        for module, enabled in states:
            module.set_kappa_swiglu_enabled(enabled)
        compute_cleanup()


if __name__ == "__main__":
    main()