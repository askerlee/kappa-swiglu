import ast
from pathlib import Path

from datasets import Dataset

from tasks import ultradata


CHAT_SFT = Path(__file__).resolve().parents[1] / "scripts" / "chat_sft.py"


def test_has_no_cjk_user_questions_filters_only_cjk_questions():
    assert ultradata.has_no_cjk_user_questions({
        "messages": [
            {"role": "user", "content": "Write a short poem about the ocean."},
            {"role": "assistant", "content": "Waves fold into the shore."},
        ]
    })
    assert not ultradata.has_no_cjk_user_questions({
        "messages": [
            {"role": "user", "content": "请写一首关于海洋的短诗。"},
            {"role": "assistant", "content": "海浪轻拍岸边。"},
        ]
    })
    assert ultradata.has_no_cjk_user_questions({
        "messages": [
            {"role": "user", "content": "Write a short poem about the ocean."},
            {"role": "assistant", "content": "Here it is."},
            {"role": "user", "content": "Ahora hazlo más corto."},
            {"role": "assistant", "content": "Waves rest."},
        ]
    })
    assert ultradata.has_no_cjk_user_questions({
        "messages": [
            {"role": "user", "content": "تابع"},
            {"role": "assistant", "content": "Continuing."},
        ]
    })
    assert ultradata.has_no_cjk_user_questions({
        "messages": [
            {"role": "user", "content": "```python\nprint(42)\n```"},
            {"role": "assistant", "content": "42"},
        ]
    })


def test_ultradata_sft_if_loads_expected_slice_and_filters_before_shuffle(monkeypatch):
    rows = {
        "uid": ["english", "chinese"],
        "messages": [
            [
                {"role": "user", "content": "List three uses for a paper clip."},
                {"role": "assistant", "content": "Bookmark, zipper pull, and reset pin."},
            ],
            [
                {"role": "user", "content": "列出回形针的三种用途。"},
                {"role": "assistant", "content": "书签、拉链头和复位针。"},
            ],
        ],
    }
    load_calls = []

    def fake_load_dataset(path, config, split, token):
        load_calls.append((path, config, split, token))
        return Dataset.from_dict(rows)

    monkeypatch.setattr(ultradata, "load_dataset", fake_load_dataset)

    task = ultradata.UltraDataSFTIF()

    assert load_calls == [("openbmb/UltraData-SFT-2605", "IF", "no_think", True)]
    assert len(task) == 1
    assert task[0]["messages"][0]["content"] == "List three uses for a paper clip."


def test_chat_sft_ultradata_flag_defaults_true_and_guards_dataset_construction():
    tree = ast.parse(CHAT_SFT.read_text(), filename=str(CHAT_SFT))
    flag_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--use-ultradata-sft-if"
    )
    defaults = {
        keyword.arg: keyword.value.value
        for keyword in flag_call.keywords
        if isinstance(keyword.value, ast.Constant)
    }
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "use_ultradata_sft_if"
    )

    assert defaults["default"] is True
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "UltraDataSFTIF"
        for statement in guard.body
        for node in ast.walk(statement)
    )