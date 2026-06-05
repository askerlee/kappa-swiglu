"""
Tulu 3 SFT mixture by Allen AI.
https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
"""

from datasets import load_dataset
from tasks.common import Task


class Tulu3SFTMixture(Task):
    """ Tulu 3 SFT mixture. train is 939K rows. """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "Tulu3SFTMixture split must be train"
        self.ds = load_dataset("allenai/tulu-3-sft-mixture", split=split).shuffle(seed=42)
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
        return {"messages": messages}