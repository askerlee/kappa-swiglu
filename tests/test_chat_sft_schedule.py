import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAT_SFT = ROOT / "scripts" / "chat_sft.py"


def load_function_from_script(function_name):
    source = CHAT_SFT.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(CHAT_SFT))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(function_module)
            namespace = {}
            exec(compile(function_module, filename=str(CHAT_SFT), mode="exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name} not found in {CHAT_SFT}")


def test_exp_gate_bias_lr_schedule_linearly_decays_from_default_start_to_end():
    schedule = load_function_from_script("get_exp_gate_bias_lr_scale")

    assert abs(schedule(0.0) - 0.1) < 1e-12
    assert abs(schedule(0.5) - 0.055) < 1e-12
    assert abs(schedule(1.0) - 0.01) < 1e-12


def test_exp_gate_bias_lr_schedule_clamps_progress_and_supports_overrides():
    schedule = load_function_from_script("get_exp_gate_bias_lr_scale")

    assert abs(schedule(-1.0, start_scale=0.2, end_scale=0.05) - 0.2) < 1e-12
    assert abs(schedule(2.0, start_scale=0.2, end_scale=0.05) - 0.05) < 1e-12


def test_chat_eval_task_names_default_to_all_tasks():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'chat_eval_task_names = ALL_CHAT_EVAL_TASKS if args.chat_eval_task_name is None else args.chat_eval_task_name.split(\'|\')' in source


def test_chat_eval_runs_only_on_last_step():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "if last_step:\n        model.eval()\n        engine = Engine(orig_model, tokenizer)" in source
    assert "chat_eval_every" not in source


def test_final_checkpoint_is_saved_before_final_chat_eval():
    source = CHAT_SFT.read_text(encoding="utf-8")

    save_index = source.index("    # save checkpoint at the end of the run before the expensive final chat eval")
    chat_eval_index = source.index("    if last_step:\n        model.eval()\n        engine = Engine(orig_model, tokenizer)")

    assert save_index < chat_eval_index


def test_gate_proj_bias_l2_anchor_cli_defaults_to_initial_and_wires_load_behavior():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--exp-gate-proj-bias-l2-anchor", type=str, choices=("initial", "zero"), default="initial"' in source
    assert 'refresh_gate_proj_bias_references = args.exp_gate_proj_bias_l2_anchor == "initial"' in source
    assert 'refresh_gate_proj_bias_references=refresh_gate_proj_bias_references' in source