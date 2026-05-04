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