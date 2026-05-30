"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import random
import os
import time

from jinja2 import Template
import torch
import torch.distributed as dist


_MC_TEMPLATE = Template("""
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip())

_SCHEMA_TEMPLATE = Template("""
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip())

_LM_TEMPLATE = Template("""
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip())


def _core_profile_enabled():
    return os.environ.get("NANOCHAT_CORE_EVAL_PROFILE", "").lower() not in {"", "0", "false", "no"}


def _cuda_sync_if_needed(device, enabled):
    if not enabled:
        return
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def _core_tokenizer_threads():
    override = os.environ.get("NANOCHAT_CORE_TOKENIZER_THREADS")
    if override is not None:
        return max(1, int(override))
    world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    return 1 if world_size > 1 else 8

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [_MC_TEMPLATE.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [_SCHEMA_TEMPLATE.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = _LM_TEMPLATE.render(include_continuation=False, **context)
    prompt_with = _LM_TEMPLATE.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id(), num_threads=_core_tokenizer_threads())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id(), num_threads=_core_tokenizer_threads())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id(), num_threads=_core_tokenizer_threads())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.inference_mode()
def forward_model(model, input_ids, device=None, profile=None):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    profile_enabled = profile is not None
    batch_size, seq_len = input_ids.size()

    _cuda_sync_if_needed(device, profile_enabled)
    start_time = time.perf_counter() if profile_enabled else None
    outputs = model(input_ids)
    _cuda_sync_if_needed(device, profile_enabled)
    if profile_enabled:
        profile['model_forward'] += time.perf_counter() - start_time

    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)

    # Calculate cross entropy at all positions
    _cuda_sync_if_needed(device, profile_enabled)
    start_time = time.perf_counter() if profile_enabled else None
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    _cuda_sync_if_needed(device, profile_enabled)
    if profile_enabled:
        profile['loss'] += time.perf_counter() - start_time

    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')

    # Get the argmax predictions at each position
    _cuda_sync_if_needed(device, profile_enabled)
    start_time = time.perf_counter() if profile_enabled else None
    predictions = outputs.argmax(dim=-1)
    _cuda_sync_if_needed(device, profile_enabled)
    if profile_enabled:
        profile['argmax'] += time.perf_counter() - start_time
    return losses, predictions


@torch.inference_mode()
def evaluate_example_details(idx, model, tokenizer, data, device, task_meta, profile=None):
    """Evaluate a single example and return prediction details."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']
    profile_enabled = profile is not None

    # Sample few-shot examples (excluding current item)
    start_time = time.perf_counter() if profile_enabled else None
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]
    if profile_enabled:
        profile['fewshot'] += time.perf_counter() - start_time

    # Render prompts and batch sequences based on task type
    start_time = time.perf_counter() if profile_enabled else None
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # In these cases, we have to truncate sequences to max length and adjust the indices
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # take the last max_tokens tokens
                new_start_idxs.append(s - num_to_crop) # shift the indices down
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t) # keep unchanged
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs
    if profile_enabled:
        profile['prepare'] += time.perf_counter() - start_time

    # Stack up all the sequences into a batch
    start_time = time.perf_counter() if profile_enabled else None
    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)
    if profile_enabled:
        profile['stack'] += time.perf_counter() - start_time

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    start_time = time.perf_counter() if profile_enabled else None
    losses, predictions = forward_model(model, input_ids, device=device, profile=profile)
    if profile_enabled:
        profile['forward'] += time.perf_counter() - start_time

    # See if the losses/predictions come out correctly
    start_time = time.perf_counter() if profile_enabled else None
    if task_type == 'language_modeling':
        # language modeling task is currently always batch size 1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
        pred_idx = -1
        gold_idx = -1
        choice_logps = None
    elif task_type in ['multiple_choice', 'schema']:
        # For MC/schema: find the option with lowest average loss
        choice_logps = [
            -losses[i, si-1:ei-1].sum().item()
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))
        ]
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        gold_idx = int(item['gold'])
        is_correct = pred_idx == gold_idx
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    if profile_enabled:
        profile['score'] += time.perf_counter() - start_time

    return {
        'is_correct': bool(is_correct),
        'pred_idx': pred_idx,
        'gold_idx': gold_idx,
        'choice_logps': choice_logps,
    }


@torch.inference_mode()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, return True if correct, False otherwise"""
    result = evaluate_example_details(idx, model, tokenizer, data, device, task_meta)
    return result['is_correct']


def evaluate_task_detailed(model, tokenizer, data, device, task_meta):
    """Evaluate a task and return accuracy plus per-example prediction details."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    profile_enabled = _core_profile_enabled()
    if profile_enabled and rank == 0:
        local_examples = len(range(rank, len(data), world_size))
        requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if requested_world_size > 1 and world_size == 1:
            print(
                f"[core_eval] torchrun requested WORLD_SIZE={requested_world_size}, "
                f"but dist is not initialized; CORE is running single-rank over {len(data)} examples"
            )
        else:
            print(
                f"[core_eval] rank {rank}/{world_size} evaluating {local_examples}/{len(data)} examples"
            )
    local_details = []
    local_profile = {
        'fewshot': 0.0,
        'prepare': 0.0,
        'stack': 0.0,
        'model_forward': 0.0,
        'loss': 0.0,
        'argmax': 0.0,
        'forward': 0.0,
        'score': 0.0,
    }
    local_example_count = 0

    for idx in range(rank, len(data), world_size):
        result = evaluate_example_details(
            idx,
            model,
            tokenizer,
            data,
            device,
            task_meta,
            profile=local_profile if profile_enabled else None,
        )
        local_example_count += 1
        local_details.append({
            'index': idx,
            'is_correct': bool(result['is_correct']),
            'pred_idx': int(result['pred_idx']),
            'gold_idx': int(result['gold_idx']),
            'choice_logps': result['choice_logps'],
        })

    gather_start = time.perf_counter() if profile_enabled else None
    if world_size > 1:
        gathered_details = [None] * world_size
        dist.all_gather_object(gathered_details, local_details)
        details = [detail for rank_details in gathered_details for detail in rank_details]
    else:
        details = local_details
    gather_time = time.perf_counter() - gather_start if profile_enabled else 0.0

    details.sort(key=lambda detail: detail['index'])
    accuracy = sum(float(detail['is_correct']) for detail in details) / len(data)

    if profile_enabled:
        local_total = (
            local_profile['fewshot']
            + local_profile['prepare']
            + local_profile['stack']
            + local_profile['forward']
            + local_profile['score']
        )
        if world_size > 1:
            metrics_sum = torch.tensor([
                float(local_example_count),
                local_profile['fewshot'],
                local_profile['prepare'],
                local_profile['stack'],
                local_profile['model_forward'],
                local_profile['loss'],
                local_profile['argmax'],
                local_profile['forward'],
                local_profile['score'],
                gather_time,
                local_total,
            ], dtype=torch.float64, device=device)
            metrics_max = metrics_sum.clone()
            dist.all_reduce(metrics_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(metrics_max, op=dist.ReduceOp.MAX)
            metrics_sum = metrics_sum.cpu()
            metrics_max = metrics_max.cpu()
        else:
            metrics_sum = torch.tensor([
                float(local_example_count),
                local_profile['fewshot'],
                local_profile['prepare'],
                local_profile['stack'],
                local_profile['model_forward'],
                local_profile['loss'],
                local_profile['argmax'],
                local_profile['forward'],
                local_profile['score'],
                gather_time,
                local_total,
            ], dtype=torch.float64)
            metrics_max = metrics_sum.clone()

        if rank == 0:
            total_examples = max(int(metrics_sum[0].item()), 1)
            mean_examples_per_rank = total_examples / world_size
            labels = ['fewshot', 'prepare', 'stack', 'model_forward', 'loss', 'argmax', 'forward', 'score', 'gather']
            sum_values = metrics_sum.tolist()[1:10]
            max_values = metrics_max.tolist()[1:10]
            pieces = []
            for label, sum_value, max_value in zip(labels, sum_values, max_values):
                per_example_ms = 1000.0 * sum_value / total_examples
                max_rank_s = max_value
                pieces.append(f"{label}={per_example_ms:.2f}ms/ex (max-rank {max_rank_s:.2f}s)")
            mean_rank_total = metrics_sum[10].item() / world_size
            max_rank_total = metrics_max[10].item()
            print(
                "[core_eval_profile] "
                f"mean_examples_per_rank={mean_examples_per_rank:.1f}; "
                + "; ".join(pieces)
                + f"; total_compute_mean={mean_rank_total:.2f}s; total_compute_max={max_rank_total:.2f}s"
            )

    return {
        'accuracy': accuracy,
        'details': details,
    }


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    """
    return evaluate_task_detailed(model, tokenizer, data, device, task_meta)['accuracy']
