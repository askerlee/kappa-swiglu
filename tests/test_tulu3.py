import ast
from pathlib import Path

from datasets import Dataset

from tasks import tulu3


ROOT = Path(__file__).resolve().parents[1]


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


def test_tulu3_english_filter_checks_full_conversation(monkeypatch):
    english_messages = [
        {"role": "user", "content": "Analyze the meaning of this song."},
        {"role": "assistant", "content": "The song describes memory and loss."},
    ]
    telugu_messages = [
        {"role": "user", "content": "ఈ పాటను విశ్లేషించండి."},
        {"role": "assistant", "content": "ఈ పాట జీవితం గురించి చెబుతుంది."},
    ]
    mixed_messages = [
        {"role": "user", "content": "Write a detailed analysis of this song and its themes."},
        {"role": "assistant", "content": "ఈ పాట జీవితం గురించి చెబుతుంది."},
    ]

    def fake_load_dataset(path, split):
        return Dataset.from_dict({"messages": [english_messages, telugu_messages, mixed_messages]})

    monkeypatch.setattr(tulu3, "load_dataset", fake_load_dataset)

    task = tulu3.Tulu3SFTMixture(split="train", english_only=True)

    assert len(task) == 1
    assert task[0] == {"messages": english_messages}


def test_training_scripts_enable_english_only_tulu_data():
    for script_name in ("base_train_mix.py", "chat_sft.py"):
        tree = ast.parse((ROOT / "scripts" / script_name).read_text())
        tulu_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"Tulu3SFTMixture", "Tulu3SFTPersonaIF"}
        ]

        assert len(tulu_calls) == 2
        for call in tulu_calls:
            english_only = next(
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "english_only"
            )
            assert isinstance(english_only, ast.Constant)
            assert english_only.value is True