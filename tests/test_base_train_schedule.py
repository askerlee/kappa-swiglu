import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = ROOT / "scripts" / "base_train.py"


def load_function_from_script(function_name):
    source = BASE_TRAIN.read_text()
    module = ast.parse(source, filename=str(BASE_TRAIN))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(function_module, filename=str(BASE_TRAIN), mode="exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name} not found in {BASE_TRAIN}")


def test_get_annealed_loss_weight_reaches_zero_at_anneal_limit():
    get_annealed_loss_weight = load_function_from_script("get_annealed_loss_weight")

    assert get_annealed_loss_weight(0.25, 5, num_anneal_iterations=5, floor_frac=0.0) == 0.0


def test_gate_proj_bias_l2_default_schedule_uses_half_run_and_zero_floor():
    source = BASE_TRAIN.read_text()

    assert 'parser.add_argument("--gate-proj-bias-l2-loss-floor-frac", type=float, default=0.0' in source
    assert 'gate_proj_bias_l2_num_anneal_iterations = max(math.ceil(num_iterations / 2), 1)' in source