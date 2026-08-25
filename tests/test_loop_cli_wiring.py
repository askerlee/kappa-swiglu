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
ALL_LOOP_SCRIPTS = FRESH_MODEL_SCRIPTS + CHECKPOINT_MODEL_SCRIPTS
EVERYPASS_NTP_SCRIPTS = FRESH_MODEL_SCRIPTS + (
    ROOT / "scripts" / "chat_sft.py",
)
UT_DETACH_SCRIPTS = EVERYPASS_NTP_SCRIPTS


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


def test_fresh_model_scripts_wire_ut_source_and_destination():
    for path in FRESH_MODEL_SCRIPTS:
        tree = _parse(path)
        source = path.read_text(encoding="utf-8")

        assert 'parser.add_argument("--ut-source", type=int, default=None' in source
        assert 'parser.add_argument("--ut-destination", type=int, default=None' in source
        assert "ut_edge_offset = max(1, args.depth // 6)" in source
        assert "args.ut_source = -ut_edge_offset" in source
        assert "args.ut_destination = ut_edge_offset" in source
        config_calls = _find_calls(tree, "GPTConfig")
        assert any(
            {keyword.arg for keyword in call.keywords} >= {"ut_source", "ut_destination"}
            for call in config_calls
        ), path


def test_checkpoint_scripts_wire_loop_into_load_model():
    for path in CHECKPOINT_MODEL_SCRIPTS:
        tree = _parse(path)
        _assert_loop_argument(tree, expected_default=None)
        assert any(
            _has_args_total_ut_steps_keyword(call)
            for call in _find_calls(tree, "load_model")
        ), path


def test_loop_scripts_print_nondefault_loop_count():
    for path in ALL_LOOP_SCRIPTS:
        source = path.read_text(encoding="utf-8")
        assert "if args.total_ut_steps > 1:" in source, path
        assert 'print0(f"Loops = {args.total_ut_steps}")' in source, path


def test_requested_scripts_wire_ut_everypass_ntp():
    for path in EVERYPASS_NTP_SCRIPTS:
        tree = _parse(path)
        argument_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and any(
                isinstance(arg, ast.Constant) and arg.value == "--ut-everypass-ntp"
                for arg in node.args
            )
        ]
        assert len(argument_calls) == 1, path
        expected_default = False if path.name == "chat_sft.py" else True
        keywords = {keyword.arg: keyword.value for keyword in argument_calls[0].keywords}
        assert ast.literal_eval(keywords["dest"]) == "ut_everypass_ntp"
        assert ast.literal_eval(keywords["default"]) is expected_default

        target_function = "load_model" if path.name == "chat_sft.py" else "GPTConfig"
        assert any(
            any(
                keyword.arg == "ut_everypass_ntp"
                and isinstance(keyword.value, ast.Attribute)
                and isinstance(keyword.value.value, ast.Name)
                and keyword.value.value.id == "args"
                and keyword.value.attr == "ut_everypass_ntp"
                for keyword in call.keywords
            )
            for call in _find_calls(tree, target_function)
        ), path


def test_training_scripts_explicitly_disable_ut_detach_by_default():
    for path in UT_DETACH_SCRIPTS:
        tree = _parse(path)
        argument_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and any(
                isinstance(arg, ast.Constant) and arg.value == "--ut-detach"
                for arg in node.args
            )
        ]
        assert len(argument_calls) == 1, path
        keywords = {keyword.arg: keyword.value for keyword in argument_calls[0].keywords}
        assert ast.literal_eval(keywords["dest"]) == "ut_detach"
        assert ast.literal_eval(keywords["default"]) is False

        target_function = "load_model" if path.name == "chat_sft.py" else "GPTConfig"
        assert any(
            any(
                keyword.arg == "ut_detach"
                and isinstance(keyword.value, ast.Attribute)
                and isinstance(keyword.value.value, ast.Name)
                and keyword.value.value.id == "args"
                and keyword.value.attr == "ut_detach"
                for keyword in call.keywords
            )
            for call in _find_calls(tree, target_function)
        ), path