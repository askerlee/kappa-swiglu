"""
Check the integrity of the training data used by scripts/base_train_mix.py.

This covers two data sources:
1. The base pretraining corpus: FineWeb-Edu parquet shards on disk (nanochat/dataset.py).
2. The chat-SFT data mixed into training: the local identity_conversations.jsonl file,
   and (optionally, since they require network access / HF downloads) the various
   HuggingFace-backed Task datasets (SmolTalk, MMLU, ARC, GSM8K, Tulu3, UltraData, ...).

Usage:
    python -m scripts.check_data_integrity
    python -m scripts.check_data_integrity --check-sft
    python -m scripts.check_data_integrity --sample-docs 20
"""

import os
import sys
import json
import argparse

import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.dataset import DATA_DIR, MAX_SHARD, index_to_filename, list_parquet_files

parser = argparse.ArgumentParser(description="Check integrity of base_train_mix.py training data")
parser.add_argument("--data-dir", type=str, default=None, help="Override the base pretraining data dir (default: nanochat/dataset.py DATA_DIR)")
parser.add_argument("--sample-docs", type=int, default=5, help="Number of documents to spot-check per parquet file for malformed/empty text")
parser.add_argument("--check-sft", action="store_true", help="Also instantiate the HF-backed SFT task datasets (requires network access, can be slow)")
parser.add_argument("--identity-filepath", type=str, default=None, help="Override the identity_conversations.jsonl path")
args = parser.parse_args()

errors = []
warnings = []

def error(msg):
    errors.append(msg)
    print(f"[ERROR] {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"[WARN] {msg}")

def ok(msg):
    print(f"[OK] {msg}")

# -----------------------------------------------------------------------------
# 1. Base pretraining data: parquet shards

def check_base_pretraining_data(data_dir, sample_docs):
    print("\n=== Checking base pretraining data (parquet shards) ===")
    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        error(f"No parquet files found in {data_dir}. Did you run `python -m nanochat.dataset`?")
        return

    # Figure out which shard indices are present, to report gaps.
    expected_names = {index_to_filename(i): i for i in range(MAX_SHARD + 1)}
    present_indices = []
    unrecognized = []
    for path in parquet_paths:
        name = os.path.basename(path)
        if name in expected_names:
            present_indices.append(expected_names[name])
        else:
            unrecognized.append(name)
    if unrecognized:
        warn(f"{len(unrecognized)} parquet file(s) do not match the expected shard_XXXXX.parquet naming: {unrecognized[:5]}{'...' if len(unrecognized) > 5 else ''}")
    if present_indices:
        missing = sorted(set(range(min(present_indices), max(present_indices) + 1)) - set(present_indices))
        if missing:
            warn(f"{len(missing)} shard(s) missing within the downloaded range (e.g. {missing[:10]}{'...' if len(missing) > 10 else ''})")

    total_rows = 0
    total_bytes = 0
    for path in parquet_paths:
        name = os.path.basename(path)
        try:
            size = os.path.getsize(path)
            if size == 0:
                error(f"{name}: file is empty (0 bytes)")
                continue
            pf = pq.ParquetFile(path)
        except Exception as e:
            error(f"{name}: failed to open as parquet ({e})")
            continue

        total_bytes += size
        schema_names = pf.schema.names
        if "text" not in schema_names:
            error(f"{name}: missing required 'text' column (columns present: {schema_names})")
            continue

        num_rows = pf.metadata.num_rows
        total_rows += num_rows
        if num_rows == 0:
            warn(f"{name}: file has 0 rows")

        # Spot-check a handful of documents in the first row group for corruption / empty text.
        if sample_docs > 0 and pf.num_row_groups > 0:
            try:
                rg = pf.read_row_group(0)
                texts = rg.column("text").to_pylist()[:sample_docs]
                for i, text in enumerate(texts):
                    if text is None:
                        error(f"{name}: document {i} in row group 0 has null text")
                    elif not isinstance(text, str):
                        error(f"{name}: document {i} in row group 0 has non-string text ({type(text)})")
                    elif len(text.strip()) == 0:
                        warn(f"{name}: document {i} in row group 0 has empty/whitespace-only text")
            except Exception as e:
                error(f"{name}: failed to read row group 0 ({e})")

    ok(f"Found {len(parquet_paths)} parquet file(s), {total_rows:,} total rows, {total_bytes / 1e9:.2f} GB on disk")
    if len(parquet_paths) < 2:
        warn("Fewer than 2 parquet files found; base_train_mix.py needs at least 1 train file + 1 val file (last shard is used for val)")


# -----------------------------------------------------------------------------
# 2. Identity conversations JSONL (used via CustomJSON in the chat-SFT mixture)

def check_identity_conversations(filepath):
    print("\n=== Checking identity_conversations.jsonl ===")
    if not os.path.exists(filepath):
        warn(f"{filepath} does not exist. Download it with:\n"
             f"  curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
        return

    num_conversations = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                messages = json.loads(line)
            except json.JSONDecodeError as e:
                error(f"identity_conversations.jsonl:{line_num}: invalid JSON ({e})")
                continue
            if not isinstance(messages, list) or len(messages) < 2:
                error(f"identity_conversations.jsonl:{line_num}: expected a list of >= 2 messages, got {messages!r}")
                continue
            valid = True
            for i, message in enumerate(messages):
                expected_role = "user" if i % 2 == 0 else "assistant"
                if not isinstance(message, dict) or "role" not in message or "content" not in message:
                    error(f"identity_conversations.jsonl:{line_num}: message {i} missing 'role'/'content'")
                    valid = False
                    break
                if message["role"] != expected_role:
                    error(f"identity_conversations.jsonl:{line_num}: message {i} has role '{message['role']}', expected '{expected_role}'")
                    valid = False
                    break
                if not isinstance(message["content"], str):
                    error(f"identity_conversations.jsonl:{line_num}: message {i} content is not a string")
                    valid = False
                    break
            if valid:
                num_conversations += 1

    ok(f"identity_conversations.jsonl: {num_conversations} valid conversation(s)")


# -----------------------------------------------------------------------------
# 3. (Optional) HF-backed chat-SFT task datasets

def check_language_filter(task, name, is_english_fn, sample_size=200):
    """ Re-runs the task's own English-only filter predicate on a sample of its (already filtered) rows. """
    import random
    n = len(task.ds)
    if n == 0:
        return
    rng = random.Random(0)
    indices = rng.sample(range(n), min(sample_size, n))
    failures = [idx for idx in indices if not is_english_fn(task.ds[idx])]
    ratio = len(failures) / len(indices)
    if ratio > 0.05:
        warn(f"{name}: english_only filter check failed for {len(failures)}/{len(indices)} sampled rows ({ratio:.0%}) — filtering may be broken")
    else:
        ok(f"{name}: english_only filter verified on {len(indices)} sampled rows ({len(failures)} flagged)")


def check_sft_tasks():
    print("\n=== Checking HF-backed SFT task datasets (this may download data) ===")
    from tasks.arc import ARC
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk
    from tasks.spellingbee import SimpleSpelling, SpellingBee
    from tasks.tulu3 import Tulu3SFTMixture, Tulu3SFTPersonaIF, has_only_english_messages
    from tasks.ultradata import UltraDataSFTIF, has_only_english_user_questions

    # Tasks that filter their rows down to English-only content, and the predicate
    # used to do so (reused here to verify the filtering actually took effect).
    language_filters = {
        "Tulu3SFTMixture": has_only_english_messages,
        "Tulu3SFTPersonaIF": has_only_english_messages,
        "UltraDataSFTIF": has_only_english_user_questions,
    }

    task_builders = {
        "SmolTalk(train)": lambda: SmolTalk(split="train"),
        "MMLU(auxiliary_train)": lambda: MMLU(subset="auxiliary_train", split="train"),
        "ARC-Easy(train)": lambda: ARC(subset="ARC-Easy", split="train"),
        "ARC-Challenge(train)": lambda: ARC(subset="ARC-Challenge", split="train"),
        "GSM8K(main,train)": lambda: GSM8K(subset="main", split="train"),
        "Tulu3SFTMixture": lambda: Tulu3SFTMixture(split="train", english_only=True),
        "Tulu3SFTPersonaIF": lambda: Tulu3SFTPersonaIF(split="train", english_only=True),
        "UltraDataSFTIF": lambda: UltraDataSFTIF(),
        "SimpleSpelling": lambda: SimpleSpelling(size=1000, split="train", start=0, stop=1000),
        "SpellingBee": lambda: SpellingBee(size=1000, split="train", response_style="mixed", start=0, stop=1000),
    }

    for name, build in task_builders.items():
        try:
            task = build()
            n = len(task)
            if n == 0:
                warn(f"{name}: dataset has 0 examples")
                continue
            # Spot-check a couple of examples to make sure they're well-formed.
            for idx in (0, n // 2, n - 1):
                task[idx]
            ok(f"{name}: {n:,} examples, sample checks passed")
            # HF `datasets` fingerprints .filter()/.map() calls and caches the result as an
            # Arrow file on disk; report the cache file so it's clear whether we hit that
            # cache (same one base_train_mix.py would use) or freshly recomputed the filter.
            ds = getattr(task, "ds", None)
            cache_files = getattr(ds, "cache_files", None) if ds is not None else None
            if cache_files:
                ok(f"{name}: reading from on-disk HF datasets cache ({cache_files[0]['filename']})")
            if name in language_filters:
                check_language_filter(task, name, language_filters[name])
        except Exception as e:
            error(f"{name}: failed to load/validate ({e})")


# -----------------------------------------------------------------------------

base_dir = get_base_dir()
data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
identity_filepath = args.identity_filepath if args.identity_filepath is not None else os.path.join(base_dir, "identity_conversations.jsonl")

check_base_pretraining_data(data_dir, args.sample_docs)
check_identity_conversations(identity_filepath)
if args.check_sft:
    check_sft_tasks()

print(f"\n=== Summary: {len(errors)} error(s), {len(warnings)} warning(s) ===")
sys.exit(1 if errors else 0)
