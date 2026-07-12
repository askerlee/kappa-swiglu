import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanochat.chatcore import ALL_CHAT_EVAL_TASKS, CHATCORE_TASKS
from tasks.ifeval import IFEval, _check_instruction


class Dataset(list):
    def shuffle(self, seed):
        assert seed == 42
        return self


def test_ifeval_is_a_default_chat_eval_but_not_a_chatcore_component():
    assert "IFEval" in ALL_CHAT_EVAL_TASKS
    assert "IFEval" not in CHATCORE_TASKS


def test_ifeval_converts_dataset_row_and_requires_all_instructions(monkeypatch):
    rows = Dataset([{
        "prompt": "Respond with a sentence.",
        "instruction_id_list": ["punctuation:no_comma", "startend:quotation"],
        "kwargs": [{}, {}],
    }])
    monkeypatch.setattr("tasks.ifeval.load_dataset", lambda *args, **kwargs: rows)
    task = IFEval()

    conversation = task[0]

    assert conversation["messages"] == [
        {"role": "user", "content": "Respond with a sentence."},
        {"role": "assistant", "content": ""},
    ]
    assert task.evaluate(conversation, '"A complete response."')
    assert not task.evaluate(conversation, '"This response, has a comma."')


def test_ifeval_repeat_prompt_uses_the_instruction_argument():
    assert _check_instruction(
        "combination:repeat_prompt",
        {"prompt_to_repeat": "Repeat exactly this"},
        "Unrelated user prompt",
        "Repeat exactly this, then answer.",
    )