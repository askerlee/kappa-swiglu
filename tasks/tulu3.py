"""
Tulu 3 SFT mixture by Allen AI.
https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
"""

import os
import unicodedata

from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect
from tasks.common import Task, normalize_conversation


DetectorFactory.seed = 42
MAX_NON_LATIN_LETTER_RATIO = 0.1


def has_only_english_messages(row):
    messages = row.get("messages", [])
    contents = [
        message.get("content", "").strip()
        for message in messages
        if isinstance(message.get("content"), str)
    ]
    if not contents or len(contents) != len(messages) or not all(contents):
        return False

    conversation = "\n".join(contents)
    letters = [character for character in conversation if character.isalpha()]
    non_latin_letters = sum(
        "LATIN" not in unicodedata.name(character, "") for character in letters
    )
    if letters and non_latin_letters / len(letters) > MAX_NON_LATIN_LETTER_RATIO:
        return False
    try:
        return detect(conversation) == "en"
    except LangDetectException:
        return False


def get_filter_num_proc():
    allocated_cpus = None
    for variable in ("SLURM_CPUS_PER_TASK", "PBS_NCPUS", "NCPUS", "PBS_NP"):
        if variable in os.environ:
            allocated_cpus = int(os.environ[variable])
            break
    if allocated_cpus is None and hasattr(os, "sched_getaffinity"):
        allocated_cpus = len(os.sched_getaffinity(0))
    if allocated_cpus is None:
        allocated_cpus = os.cpu_count() or 1
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return max(1, allocated_cpus // local_world_size)


def load_tulu_dataset(path, split, english_only):
    dataset = load_dataset(path, split=split)
    if english_only:
        filter_num_proc = get_filter_num_proc()
        filter_kwargs = {"desc": "Filtering non-English Tulu conversations"}
        if filter_num_proc > 1:
            filter_kwargs["num_proc"] = filter_num_proc
        dataset = dataset.filter(has_only_english_messages, **filter_kwargs)
    return dataset.shuffle(seed=42)


class Tulu3SFTMixture(Task):
    """ Tulu 3 SFT mixture. train is 939K rows. """

    def __init__(self, split="train", english_only=False, **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "Tulu3SFTMixture split must be train"
        self.ds = load_tulu_dataset(
            "allenai/tulu-3-sft-mixture",
            split,
            english_only,
        )
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        assert len(messages) >= 2, "Tulu3SFTMixture messages must have at least 2 messages"
        for message in messages:
            assert "role" in message, "Message missing 'role' field"
            assert "content" in message, "Message missing 'content' field"
            assert isinstance(message["content"], str), "Content must be a string"
        return {"messages": normalize_conversation(messages)}


class Tulu3SFTPersonaIF(Task):
    """Focused 29,980-example Persona-IF subset already present in the full mixture."""

    def __init__(self, split="train", english_only=False, **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "Tulu3SFTPersonaIF split must be train"
        self.ds = load_tulu_dataset(
            "allenai/tulu-3-sft-personas-instruction-following",
            split,
            english_only,
        )
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.ds[index]["messages"]
        assert len(messages) >= 2, "Tulu3SFTPersonaIF messages must have at least 2 messages"
        for message in messages:
            assert "role" in message, "Message missing 'role' field"
            assert "content" in message, "Message missing 'content' field"
            assert isinstance(message["content"], str), "Content must be a string"
        return {"messages": normalize_conversation(messages)}