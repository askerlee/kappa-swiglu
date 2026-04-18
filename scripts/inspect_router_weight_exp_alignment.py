"""Static checkpoint analysis for router-to-expert weight alignments.

This script computes the same per-expert cosine similarity used in
scripts/base_train.py for each MoE layer in a checkpoint, along with
the corresponding raw dot product scores:

    rw_gate_alignment = cos(router.w_g.weight, experts.gate_proj.mean(dim=2))
    rw_cfc_alignment = cos(router.w_g.weight, experts.c_fc.mean(dim=2))
    rw_gate_dot = router.w_g.weight · experts.gate_proj.mean(dim=2)
    rw_cfc_dot = router.w_g.weight · experts.c_fc.mean(dim=2)

Examples:

    python scripts/inspect_router_weight_exp_alignment.py --source base --model-tag d24
    python scripts/inspect_router_weight_exp_alignment.py --checkpoint-dir /path/to/ckpt --step 12000 --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import torch

from nanochat.checkpoint_manager import find_largest_model, find_last_step, load_checkpoint
from nanochat.common import get_base_dir


SOURCE_TO_DIRNAME = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}


def log_progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cosine and dot-product router-weight alignments against expert gate_proj and c_fc for every MoE layer in a checkpoint."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=sorted(SOURCE_TO_DIRNAME),
        default="base",
        help="Checkpoint source when --checkpoint-dir is not provided.",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Model tag inside the source checkpoint directory. Defaults to the largest available model.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Explicit checkpoint directory containing model_<step>.pt and meta_<step>.json.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to analyze. Defaults to the latest step in the checkpoint directory.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write the full analysis as JSON.",
    )
    parser.add_argument(
        "--print-expert-alignments",
        action="store_true",
        help="Print the per-expert alignment vector for each MoE layer.",
    )
    return parser.parse_args()


def resolve_checkpoint_dir(source: str, model_tag: str | None, checkpoint_dir: str | None) -> tuple[str, str | None]:
    if checkpoint_dir is not None:
        return os.path.abspath(checkpoint_dir), model_tag

    checkpoints_root = os.path.join(get_base_dir(), SOURCE_TO_DIRNAME[source])
    resolved_model_tag = model_tag if model_tag is not None else find_largest_model(checkpoints_root)
    return os.path.join(checkpoints_root, resolved_model_tag), resolved_model_tag


def normalize_state_dict_keys(model_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("_orig_mod."): value for key, value in model_data.items()}


def find_moe_layers(model_data: dict[str, torch.Tensor]) -> list[int]:
    pattern = re.compile(r"^transformer\.h\.(\d+)\.mlp\.router\.w_g\.weight$")
    layer_indices = []
    for key in model_data:
        match = pattern.match(key)
        if match is not None:
            layer_indices.append(int(match.group(1)))
    return sorted(layer_indices)


def compute_router_weight_exp_alignment(
    model_data: dict[str, torch.Tensor],
    layer_idx: int,
    expert_weight_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    router_key = f"transformer.h.{layer_idx}.mlp.router.w_g.weight"
    expert_weight_key = f"transformer.h.{layer_idx}.mlp.experts.{expert_weight_name}"

    if router_key not in model_data:
        raise KeyError(f"Missing router weights for layer {layer_idx}: {router_key}")
    if expert_weight_key not in model_data:
        raise KeyError(f"Missing expert {expert_weight_name} weights for layer {layer_idx}: {expert_weight_key}")

    router_weight = model_data[router_key].float()
    expert_weight = model_data[expert_weight_key].float()
    if router_weight.ndim != 2:
        raise ValueError(
            f"Expected router weights with shape [n_exp, hidden_size] at layer {layer_idx}, got {tuple(router_weight.shape)}"
        )
    if expert_weight.ndim != 3:
        raise ValueError(
            f"Expected expert {expert_weight_name} weights with shape [n_exp, hidden_size, intermediate_size] at layer {layer_idx}, got {tuple(expert_weight.shape)}"
        )

    expert_mean_weight = expert_weight.mean(dim=2)
    if router_weight.shape != expert_mean_weight.shape:
        raise ValueError(
            f"Shape mismatch at layer {layer_idx}: router {tuple(router_weight.shape)} vs mean {expert_weight_name} {tuple(expert_mean_weight.shape)}"
        )

    dot_products = (expert_mean_weight * router_weight).sum(dim=1)
    denominator = router_weight.norm(dim=1) * expert_mean_weight.norm(dim=1)
    cosine_alignments = dot_products / (denominator + 1e-10)
    return cosine_alignments, dot_products


def summarize_values(alignments: torch.Tensor) -> dict[str, Any]:
    return {
        "n_experts": int(alignments.numel()),
        "mean": float(alignments.mean().item()),
        "abs-mean": float(alignments.abs().mean().item()),
        "std": float(alignments.std(unbiased=False).item()),
        "min": float(alignments.min().item()),
        "max": float(alignments.max().item()),
        "alignments": [float(value) for value in alignments.tolist()],
    }


def compute_alignment_correlation(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.shape != rhs.shape:
        raise ValueError(f"Correlation requires matching shapes, got {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    if lhs.numel() < 2:
        return float("nan")

    lhs_centered = lhs - lhs.mean()
    rhs_centered = rhs - rhs.mean()
    denominator = lhs_centered.norm() * rhs_centered.norm()
    if denominator.item() <= 1e-12:
        return float("nan")
    return float((lhs_centered * rhs_centered).sum().item() / denominator.item())


def summarize_layer(
    layer_idx: int,
    gate_alignments: torch.Tensor,
    cfc_alignments: torch.Tensor,
    gate_dot_products: torch.Tensor,
    cfc_dot_products: torch.Tensor,
) -> dict[str, Any]:
    gate_summary = summarize_values(gate_alignments)
    cfc_summary = summarize_values(cfc_alignments)
    gate_dot_summary = summarize_values(gate_dot_products)
    cfc_dot_summary = summarize_values(cfc_dot_products)
    if gate_summary["n_experts"] != cfc_summary["n_experts"]:
        raise ValueError(
            f"Expert count mismatch at layer {layer_idx}: gate_proj={gate_summary['n_experts']} c_fc={cfc_summary['n_experts']}"
        )
    return {
        "layer": layer_idx,
        "n_experts": gate_summary["n_experts"],
        "gate_cfc_correlation": compute_alignment_correlation(gate_alignments, cfc_alignments),
        "gate_cfc_dot_correlation": compute_alignment_correlation(gate_dot_products, cfc_dot_products),
        "gate_proj": gate_summary,
        "c_fc": cfc_summary,
        "gate_proj_dot": gate_dot_summary,
        "c_fc_dot": cfc_dot_summary,
    }


def summarize_overall(alignments: torch.Tensor) -> dict[str, Any]:
    summary = summarize_values(alignments)
    return {
        "mean": summary["mean"],
        "abs-mean": summary["abs-mean"],
        "std": summary["std"],
        "min": summary["min"],
        "max": summary["max"],
        "n_values": summary["n_experts"],
    }


def print_summary(result: dict[str, Any], print_expert_alignments: bool) -> None:
    print(f"checkpoint_dir: {result['checkpoint_dir']}")
    if result["model_tag"] is not None:
        print(f"model_tag: {result['model_tag']}")
    print(f"step: {result['step']}")
    print(f"num_moe_layers: {len(result['layers'])}")
    print()
    print("gate_proj vs c_fc correlation")
    print(f"{'layer':>5}  {'cos-corr':>10}  {'dot-corr':>10}")
    for layer_result in result["layers"]:
        print(
            f"{layer_result['layer']:5d}  "
            f"{layer_result['gate_cfc_correlation']:10.6f}  "
            f"{layer_result['gate_cfc_dot_correlation']:10.6f}"
        )

    print()
    print("gate_proj cosine alignment")
    print(f"{'layer':>5}  {'n_exp':>5}  {'mean':>10}  {'abs-mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    for layer_result in result["layers"]:
        gate_result = layer_result["gate_proj"]
        print(
            f"{layer_result['layer']:5d}  "
            f"{layer_result['n_experts']:5d}  "
            f"{gate_result['mean']:10.6f}  "
            f"{gate_result['abs-mean']:10.6f}  "
            f"{gate_result['std']:10.6f}  "
            f"{gate_result['min']:10.6f}  "
            f"{gate_result['max']:10.6f}"
        )
        if print_expert_alignments:
            values = ", ".join(f"{value:.6f}" for value in gate_result["alignments"])
            print(f"  gate_proj experts[{layer_result['layer']}]: [{values}]")

    print()
    print("c_fc cosine alignment")
    print(f"{'layer':>5}  {'n_exp':>5}  {'mean':>10}  {'abs-mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    for layer_result in result["layers"]:
        cfc_result = layer_result["c_fc"]
        print(
            f"{layer_result['layer']:5d}  "
            f"{layer_result['n_experts']:5d}  "
            f"{cfc_result['mean']:10.6f}  "
            f"{cfc_result['abs-mean']:10.6f}  "
            f"{cfc_result['std']:10.6f}  "
            f"{cfc_result['min']:10.6f}  "
            f"{cfc_result['max']:10.6f}"
        )
        if print_expert_alignments:
            values = ", ".join(f"{value:.6f}" for value in cfc_result["alignments"])
            print(f"  c_fc experts[{layer_result['layer']}]: [{values}]")

    print()
    print("gate_proj dot product alignment")
    print(f"{'layer':>5}  {'n_exp':>5}  {'mean':>12}  {'abs-mean':>12}  {'std':>12}  {'min':>12}  {'max':>12}")
    for layer_result in result["layers"]:
        gate_result = layer_result["gate_proj_dot"]
        print(
            f"{layer_result['layer']:5d}  "
            f"{layer_result['n_experts']:5d}  "
            f"{gate_result['mean']:12.6f}  "
            f"{gate_result['abs-mean']:12.6f}  "
            f"{gate_result['std']:12.6f}  "
            f"{gate_result['min']:12.6f}  "
            f"{gate_result['max']:12.6f}"
        )
        if print_expert_alignments:
            values = ", ".join(f"{value:.6f}" for value in gate_result["alignments"])
            print(f"  gate_proj_dot experts[{layer_result['layer']}]: [{values}]")

    print()
    print("c_fc dot product alignment")
    print(f"{'layer':>5}  {'n_exp':>5}  {'mean':>12}  {'abs-mean':>12}  {'std':>12}  {'min':>12}  {'max':>12}")
    for layer_result in result["layers"]:
        cfc_result = layer_result["c_fc_dot"]
        print(
            f"{layer_result['layer']:5d}  "
            f"{layer_result['n_experts']:5d}  "
            f"{cfc_result['mean']:12.6f}  "
            f"{cfc_result['abs-mean']:12.6f}  "
            f"{cfc_result['std']:12.6f}  "
            f"{cfc_result['min']:12.6f}  "
            f"{cfc_result['max']:12.6f}"
        )
        if print_expert_alignments:
            values = ", ".join(f"{value:.6f}" for value in cfc_result["alignments"])
            print(f"  c_fc_dot experts[{layer_result['layer']}]: [{values}]")

    gate_overall = result["overall"]["gate_proj"]
    cfc_overall = result["overall"]["c_fc"]
    gate_dot_overall = result["overall"]["gate_proj_dot"]
    cfc_dot_overall = result["overall"]["c_fc_dot"]
    print()
    print(
        "overall gate_proj_vs_c_fc correlation: "
        f"cos={result['overall']['gate_cfc_correlation']:.6f}, "
        f"dot={result['overall']['gate_cfc_dot_correlation']:.6f}"
    )
    print(
        "overall gate_proj: "
        f"mean={gate_overall['mean']:.6f}, abs-mean={gate_overall['abs-mean']:.6f}, std={gate_overall['std']:.6f}, min={gate_overall['min']:.6f}, max={gate_overall['max']:.6f}, "
        f"n_values={gate_overall['n_values']}"
    )
    print(
        "overall c_fc: "
        f"mean={cfc_overall['mean']:.6f}, abs-mean={cfc_overall['abs-mean']:.6f}, std={cfc_overall['std']:.6f}, min={cfc_overall['min']:.6f}, max={cfc_overall['max']:.6f}, "
        f"n_values={cfc_overall['n_values']}"
    )
    print(
        "overall gate_proj_dot: "
        f"mean={gate_dot_overall['mean']:.6f}, abs-mean={gate_dot_overall['abs-mean']:.6f}, std={gate_dot_overall['std']:.6f}, min={gate_dot_overall['min']:.6f}, max={gate_dot_overall['max']:.6f}, "
        f"n_values={gate_dot_overall['n_values']}"
    )
    print(
        "overall c_fc_dot: "
        f"mean={cfc_dot_overall['mean']:.6f}, abs-mean={cfc_dot_overall['abs-mean']:.6f}, std={cfc_dot_overall['std']:.6f}, min={cfc_dot_overall['min']:.6f}, max={cfc_dot_overall['max']:.6f}, "
        f"n_values={cfc_dot_overall['n_values']}"
    )


def main() -> None:
    args = parse_args()
    log_progress("parsed command line arguments")

    checkpoint_dir, resolved_model_tag = resolve_checkpoint_dir(args.source, args.model_tag, args.checkpoint_dir)
    step = args.step if args.step is not None else find_last_step(checkpoint_dir)
    log_progress(f"resolved checkpoint directory: {checkpoint_dir}")
    log_progress(f"resolved checkpoint step: {step}")

    log_progress("loading checkpoint tensors on CPU")
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device=torch.device("cpu"), load_optimizer=False)
    log_progress("finished loading checkpoint tensors")

    model_data = normalize_state_dict_keys(model_data)
    log_progress("normalized state dict keys")

    moe_layers = find_moe_layers(model_data)
    if not moe_layers:
        raise ValueError(f"No MoE router weights found in checkpoint: {checkpoint_dir} step {step}")
    log_progress(f"found {len(moe_layers)} MoE layers")

    with torch.inference_mode():
        layer_results = []
        all_gate_alignments = []
        all_cfc_alignments = []
        all_gate_dot_products = []
        all_cfc_dot_products = []
        for layer_idx in moe_layers:
            log_progress(f"computing alignments for layer {layer_idx}")
            gate_alignments, gate_dot_products = compute_router_weight_exp_alignment(model_data, layer_idx, "gate_proj")
            cfc_alignments, cfc_dot_products = compute_router_weight_exp_alignment(model_data, layer_idx, "c_fc")
            layer_results.append(summarize_layer(layer_idx, gate_alignments, cfc_alignments, gate_dot_products, cfc_dot_products))
            all_gate_alignments.append(gate_alignments)
            all_cfc_alignments.append(cfc_alignments)
            all_gate_dot_products.append(gate_dot_products)
            all_cfc_dot_products.append(cfc_dot_products)
            log_progress(f"finished layer {layer_idx}")

        log_progress("concatenating per-layer alignments")
        overall_gate_alignments = torch.cat(all_gate_alignments, dim=0)
        overall_cfc_alignments = torch.cat(all_cfc_alignments, dim=0)
        overall_gate_dot_products = torch.cat(all_gate_dot_products, dim=0)
        overall_cfc_dot_products = torch.cat(all_cfc_dot_products, dim=0)
        log_progress("finished overall alignment aggregation")

    result = {
        "checkpoint_dir": checkpoint_dir,
        "model_tag": resolved_model_tag,
        "source": args.source,
        "step": step,
        "model_config": meta_data.get("model_config", {}),
        "layers": layer_results,
        "overall": {
            "gate_cfc_correlation": compute_alignment_correlation(overall_gate_alignments, overall_cfc_alignments),
            "gate_cfc_dot_correlation": compute_alignment_correlation(overall_gate_dot_products, overall_cfc_dot_products),
            "gate_proj": summarize_overall(overall_gate_alignments),
            "c_fc": summarize_overall(overall_cfc_alignments),
            "gate_proj_dot": summarize_overall(overall_gate_dot_products),
            "c_fc_dot": summarize_overall(overall_cfc_dot_products),
        },
    }
    log_progress("built result summary")

    print_summary(result, print_expert_alignments=args.print_expert_alignments)
    log_progress("printed summary")

    if args.json_out is not None:
        json_out = os.path.abspath(args.json_out)
        log_progress(f"writing JSON output to {json_out}")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print()
        print(f"wrote_json: {json_out}")
        log_progress("finished writing JSON output")


if __name__ == "__main__":
    main()