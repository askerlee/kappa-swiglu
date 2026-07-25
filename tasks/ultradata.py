"""
UltraData-SFT-2605 by OpenBMB.
https://huggingface.co/datasets/openbmb/UltraData-SFT-2605
"""

import re

from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect

from tasks.common import Task


DetectorFactory.seed = 42
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def is_english_question(text: str, max_cjk_ratio: float = 0.1) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False

    cjk_count = len(_CJK_RE.findall(text))
    if cjk_count / len(letters) > max_cjk_ratio:
        return False

    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def has_only_english_user_questions(row):
    user_questions = [
        message.get("content")
        for message in row.get("messages", [])
        if message.get("role") == "user"
    ]
    return bool(user_questions) and all(is_english_question(question) for question in user_questions)


class UltraDataSFTIF(Task):
    """English-only IF/no_think conversations from UltraData-SFT-2605."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset(
            "openbmb/UltraData-SFT-2605",
            "IF",
            split="no_think",
        ).filter(has_only_english_user_questions).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.ds[index]["messages"]
        assert len(messages) >= 2, "UltraDataSFTIF messages must have at least 2 messages"
        for message in messages:
            assert message["role"] in ("system", "user", "assistant"), "Unexpected message role"
            assert isinstance(message["content"], str), "Content must be a string"
        return {"messages": messages}