from datasets import Dataset

from tasks import ultradata


def test_has_only_english_user_questions_filters_non_english_questions():
    assert ultradata.has_only_english_user_questions({
        "messages": [
            {"role": "user", "content": "Write a short poem about the ocean."},
            {"role": "assistant", "content": "Waves fold into the shore."},
        ]
    })
    assert not ultradata.has_only_english_user_questions({
        "messages": [
            {"role": "user", "content": "请写一首关于海洋的短诗。"},
            {"role": "assistant", "content": "海浪轻拍岸边。"},
        ]
    })
    assert not ultradata.has_only_english_user_questions({
        "messages": [
            {"role": "user", "content": "Write a short poem about the ocean."},
            {"role": "assistant", "content": "Here it is."},
            {"role": "user", "content": "Ahora hazlo más corto."},
            {"role": "assistant", "content": "Waves rest."},
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

    def fake_load_dataset(path, config, split):
        load_calls.append((path, config, split))
        return Dataset.from_dict(rows)

    monkeypatch.setattr(ultradata, "load_dataset", fake_load_dataset)

    task = ultradata.UltraDataSFTIF()

    assert load_calls == [("openbmb/UltraData-SFT-2605", "IF", "no_think")]
    assert len(task) == 1
    assert task[0]["messages"][0]["content"] == "List three uses for a paper clip."