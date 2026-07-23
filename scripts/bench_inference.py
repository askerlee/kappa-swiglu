"""
Benchmark inference throughput on the Engine generation path.

Examples:

python -m scripts.bench_inference --source base --max-tokens 128
python -m scripts.bench_inference --source sft --prompt "Write a haiku about GPUs" --runs 5
python -m scripts.bench_inference --source rl --num-samples 8 --max-tokens 64
python -m scripts.bench_inference --source sft --max-seconds 30 --num-samples 16
"""

import argparse
import statistics
import time
from contextlib import nullcontext

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.engine import Engine, KVCache, sample_next_token


def maybe_synchronize(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()


def resolve_max_tokens(args, prompt_tokens_count, sequence_len):
    remaining_context = max(sequence_len - prompt_tokens_count, 0)
    if args.max_tokens is None:
        return remaining_context
    return min(args.max_tokens, remaining_context)


def refill_decode_slot(kv_cache_decode, kv_cache_prefill, logits, prefill_logits, row_idx):
    prefill_pos = kv_cache_prefill.get_pos()
    kv_cache_decode.k_cache[:, row_idx, :prefill_pos, :, :] = kv_cache_prefill.k_cache[:, 0, :prefill_pos, :, :]
    kv_cache_decode.v_cache[:, row_idx, :prefill_pos, :, :] = kv_cache_prefill.v_cache[:, 0, :prefill_pos, :, :]
    kv_cache_decode.cache_seqlens[row_idx] = prefill_pos
    logits[row_idx] = prefill_logits[0]


def build_prompt_tokens(tokenizer, source, prompt):
    bos = tokenizer.get_bos_token_id()
    if source == "base":
        return tokenizer.encode(prompt, prepend=bos)

    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    prompt_tokens = [bos, user_start]
    prompt_tokens.extend(tokenizer.encode(prompt))
    prompt_tokens.extend([user_end, assistant_start])
    return prompt_tokens


@torch.inference_mode()
def run_one(engine, prompt_tokens, generate_kwargs, autocast_ctx, device_type):
    model = engine.model
    tokenizer = engine.tokenizer
    device = model.get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rng = None
    if generate_kwargs["temperature"] > 0:
        rng = torch.Generator(device=device)
        rng.manual_seed(generate_kwargs["seed"])

    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()
    m = model.config
    kv_model_kwargs = {
        "num_heads": m.n_kv_head,
        "head_dim": m.n_embd // m.n_head,
        "num_layers": m.n_layer * getattr(m, "total_ut_steps", 1),
    }

    kv_cache_prefill = KVCache(
        batch_size=1,
        seq_len=len(prompt_tokens),
        device=device,
        dtype=dtype,
        **kv_model_kwargs,
    )
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    maybe_synchronize(device_type)
    t0 = time.perf_counter()
    with autocast_ctx:
        prefill_logits = model.forward(ids, kv_cache=kv_cache_prefill)
    maybe_synchronize(device_type)
    prompt_time = time.perf_counter() - t0

    prefill_logits = prefill_logits[:, -1, :]
    logits = prefill_logits.expand(generate_kwargs["num_samples"], -1).clone()
    kv_length_hint = len(prompt_tokens) + generate_kwargs["max_tokens"]
    kv_cache_decode = KVCache(
        batch_size=generate_kwargs["num_samples"],
        seq_len=kv_length_hint,
        device=device,
        dtype=dtype,
        **kv_model_kwargs,
    )
    kv_cache_decode.prefill(kv_cache_prefill)

    decode_tokens = 0
    completed = [False] * generate_kwargs["num_samples"]
    row_decode_steps = [0] * generate_kwargs["num_samples"]
    launched_samples = generate_kwargs["num_samples"]
    max_seconds = generate_kwargs["max_seconds"]
    sustain_batch = generate_kwargs["sustain_batch"] and max_seconds is not None
    maybe_synchronize(device_type)
    t1 = time.perf_counter()
    with autocast_ctx:
        while True:
            if all(completed):
                break
            if max_seconds is not None:
                maybe_synchronize(device_type)
                if time.perf_counter() - t1 >= max_seconds:
                    break
            next_ids = sample_next_token(
                logits,
                rng,
                temperature=generate_kwargs["temperature"],
                top_k=generate_kwargs["top_k"],
            )
            token_column = next_ids[:, 0].tolist()
            for row_idx, token in enumerate(token_column):
                if completed[row_idx]:
                    continue
                row_decode_steps[row_idx] += 1
                if token == assistant_end or token == bos:
                    completed[row_idx] = True
                else:
                    decode_tokens += 1
                if row_decode_steps[row_idx] >= generate_kwargs["max_tokens"]:
                    completed[row_idx] = True
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
            if sustain_batch:
                for row_idx, is_completed in enumerate(completed):
                    if not is_completed:
                        continue
                    refill_decode_slot(kv_cache_decode, kv_cache_prefill, logits, prefill_logits, row_idx)
                    completed[row_idx] = False
                    row_decode_steps[row_idx] = 0
                    launched_samples += 1
    maybe_synchronize(device_type)
    decode_time = time.perf_counter() - t1

    prompt_tokens_count = len(prompt_tokens)
    total_tokens = prompt_tokens_count + decode_tokens
    return {
        "prompt_tokens": prompt_tokens_count,
        "decode_tokens": decode_tokens,
        "total_tokens": total_tokens,
        "launched_samples": launched_samples,
        "prompt_time": prompt_time,
        "decode_time": decode_time,
        "total_time": prompt_time + decode_time,
        "decode_tok_sec": decode_tokens / decode_time if decode_time > 0 else 0.0,
        "end_to_end_tok_sec": total_tokens / (prompt_time + decode_time) if (prompt_time + decode_time) > 0 else 0.0,
        "prompt_tok_sec": prompt_tokens_count / prompt_time if prompt_time > 0 else 0.0,
    }


def format_stats(label, values):
    if len(values) == 1:
        return f"{label}: {values[0]:,.2f}"
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    return f"{label}: mean={mean:,.2f} stdev={stdev:,.2f} min={min(values):,.2f} max={max(values):,.2f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark nanochat inference throughput")
    parser.add_argument("-i", "--source", type=str, default="sft", choices=["base", "sft", "rl"], help="Source of the model: base|sft|rl")
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    parser.add_argument("-p", "--prompt", type=str, default="Write one sentence about the moon.", help="Prompt text to prefill")
    parser.add_argument("-m", "--max-tokens", type=int, default=None, help="Maximum number of decode tokens per sample. Default: use all remaining context")
    parser.add_argument("--max-seconds", type=float, default=20, help="Maximum decode time per run in seconds")
    parser.add_argument("-n", "--num-samples", type=int, default=1, help="Number of parallel samples to decode")
    parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs to discard")
    parser.add_argument("--runs", type=int, default=3, help="Number of measured runs")
    parser.add_argument("--sustain-batch", action=argparse.BooleanOptionalAction, default=True, help="When using max-seconds, refill finished rows to keep the decode batch full")
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type: cuda|cpu|mps. empty => autodetect")
    parser.add_argument("-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="Autocast dtype on CUDA")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    model, tokenizer, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    model.eval()
    engine = Engine(model, tokenizer)
    prompt_tokens = build_prompt_tokens(tokenizer, args.source, args.prompt)
    max_tokens = resolve_max_tokens(args, len(prompt_tokens), model.config.sequence_len)
    generate_kwargs = {
        "num_samples": args.num_samples,
        "max_tokens": max_tokens,
        "max_seconds": args.max_seconds,
        "sustain_batch": args.sustain_batch,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
    }

    print0("\nInference benchmark")
    print0("-" * 80)
    print0(f"source: {args.source}")
    print0(f"device: {device}")
    print0(f"dtype: {args.dtype}")
    print0(f"prompt tokens: {len(prompt_tokens)}")
    print0(f"max decode tokens/sample: {max_tokens}")
    print0(f"max decode seconds/run: {args.max_seconds}")
    print0(f"sustain batch: {args.sustain_batch and args.max_seconds is not None}")
    print0(f"num samples: {args.num_samples}")
    print0(f"warmup runs: {args.warmup_runs}")
    print0(f"measured runs: {args.runs}")

    for _ in range(args.warmup_runs):
        run_one(engine, prompt_tokens, generate_kwargs, autocast_ctx, device_type)

    results = [run_one(engine, prompt_tokens, generate_kwargs, autocast_ctx, device_type) for _ in range(args.runs)]
    prompt_times = [result["prompt_time"] for result in results]
    decode_times = [result["decode_time"] for result in results]
    total_times = [result["total_time"] for result in results]
    prompt_tok_secs = [result["prompt_tok_sec"] for result in results]
    decode_tok_secs = [result["decode_tok_sec"] for result in results]
    end_to_end_tok_secs = [result["end_to_end_tok_sec"] for result in results]
    decode_tokens = [result["decode_tokens"] for result in results]
    launched_samples = [result["launched_samples"] for result in results]

    print0("\nResults")
    print0("-" * 80)
    print0(format_stats("prompt time (s)", prompt_times))
    print0(format_stats("decode time (s)", decode_times))
    print0(format_stats("total time (s)", total_times))
    print0(format_stats("prompt tok/s", prompt_tok_secs))
    print0(format_stats("decode tok/s", decode_tok_secs))
    print0(format_stats("end-to-end tok/s", end_to_end_tok_secs))
    print0(f"decode tokens per run: {decode_tokens}")
    print0(f"launched samples per run: {launched_samples}")


if __name__ == "__main__":
    main()