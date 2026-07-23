import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRESH_MODEL_SCRIPTS = (
    ROOT / "scripts" / "base_train.py",
    ROOT / "scripts" / "base_train_mix.py",
)
CHECKPOINT_MODEL_SCRIPTS = (
    ROOT / "scripts" / "base_eval.py",
    ROOT / "scripts" / "chat_sft.py",
    ROOT / "scripts" / "chat_eval.py",
    ROOT / "scripts" / "boolq_eval.py",
)


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_calls(tree, function_name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]


def _has_args_total_ut_steps_keyword(call):
    return any(
        keyword.arg == "total_ut_steps"
        and isinstance(keyword.value, ast.Attribute)
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "args"
        and keyword.value.attr == "total_ut_steps"
        for keyword in call.keywords
    )


def _assert_loop_argument(tree, expected_default):
    loop_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(
            isinstance(arg, ast.Constant) and arg.value == "--loop"
            for arg in node.args
        )
    ]
    assert len(loop_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in loop_calls[0].keywords}
    assert ast.literal_eval(keywords["dest"]) == "total_ut_steps"
    assert ast.literal_eval(keywords["default"]) == expected_default


def test_fresh_model_scripts_wire_loop_into_gpt_config():
    for path in FRESH_MODEL_SCRIPTS:
        tree = _parse(path)
        _assert_loop_argument(tree, expected_default=1)
        assert any(
            _has_args_total_ut_steps_keyword(call)
            for call in _find_calls(tree, "GPTConfig")
        ), path


def test_checkpoint_scripts_wire_loop_into_load_model():
    for path in CHECKPOINT_MODEL_SCRIPTS:
        tree = _parse(path)
        _assert_loop_argument(tree, expected_default=None)
        assert any(
            _has_args_total_ut_steps_keyword(call)
            for call in _find_calls(tree, "load_model")
        ), path