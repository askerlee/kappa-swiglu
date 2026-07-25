from datasets import Dataset

from tasks import tulu3


def test_tulu3_persona_if_loads_focused_training_dataset(monkeypatch):
    messages = [
        {"role": "user", "content": "Answer using exactly three sentences."},
        {"role": "assistant", "content": "First. Second. Third."},
    ]
    load_calls = []

    def fake_load_dataset(path, split):
        load_calls.append((path, split))
        return Dataset.from_dict({"messages": [messages]})

    monkeypatch.setattr(tulu3, "load_dataset", fake_load_dataset)

    task = tulu3.Tulu3SFTPersonaIF(split="train")

    assert load_calls == [("allenai/tulu-3-sft-personas-instruction-following", "train")]
    assert len(task) == 1
    assert task[0] == {"messages": messages}