"""
UltraData-SFT-2605 by OpenBMB.
https://huggingface.co/datasets/openbmb/UltraData-SFT-2605
"""

import re

from datasets import load_dataset

from tasks.common import Task, normalize_conversation
from tasks.tulu3 import get_filter_num_proc


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def print_row_count(label: str, ds) -> None:
    print(f"{label}: {len(ds)} rows")


def is_non_cjk_question(text: str, max_cjk_ratio: float = 0.1) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True

    cjk_count = len(_CJK_RE.findall(text))
    return cjk_count / len(letters) <= max_cjk_ratio

def has_no_cjk_user_questions(row):
    user_questions = [
        message.get("content")
        for message in row.get("messages", [])
        if message.get("role") == "user"
    ]
    return bool(user_questions) and all(is_non_cjk_question(question) for question in user_questions)


class UltraDataSFTIF(Task):
    """IF/no_think conversations from UltraData-SFT-2605 without CJK questions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raw_ds = load_dataset(
            "openbmb/UltraData-SFT-2605",
            "IF",
            split="no_think",
            token=True,
        )
        print_row_count("UltraDataSFTIF rows before filtering", raw_ds)
        filter_kwargs = {"desc": "Filtering CJK UltraData conversations"}
        filter_num_proc = get_filter_num_proc()
        if filter_num_proc > 1:
            filter_kwargs["num_proc"] = filter_num_proc
        filtered_ds = raw_ds.filter(has_no_cjk_user_questions, **filter_kwargs)
        print_row_count("UltraDataSFTIF rows after filtering", filtered_ds)
        self.ds = filtered_ds.shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.ds[index]["messages"]
        assert len(messages) >= 2, "UltraDataSFTIF messages must have at least 2 messages"
        for message in messages:
            assert message["role"] in ("system", "user", "assistant"), "Unexpected message role"
            assert isinstance(message["content"], str), "Content must be a string"
        return {"messages": normalize_conversation(messages)}