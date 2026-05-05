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


def test_get_annealed_loss_weight_respects_final_fraction():
    get_annealed_loss_weight = load_function_from_script("get_annealed_loss_weight")

    assert get_annealed_loss_weight(0.25, 0, num_anneal_iterations=10, floor_frac=0.2) == 0.25
    assert abs(get_annealed_loss_weight(0.25, 5, num_anneal_iterations=10, floor_frac=0.2) - 0.15) < 1e-12
    assert get_annealed_loss_weight(0.25, 10, num_anneal_iterations=10, floor_frac=0.2) == 0.05


def test_gate_proj_bias_l2_two_stage_schedule_uses_half_run_then_decays_to_final_floor():
    get_two_stage_annealed_loss_weight = load_function_from_script("get_two_stage_annealed_loss_weight")

    assert get_two_stage_annealed_loss_weight(1.0, 0, total_iterations=10) == 1.0
    assert get_two_stage_annealed_loss_weight(1.0, 5, total_iterations=10) == 0.1
    assert abs(get_two_stage_annealed_loss_weight(1.0, 7, total_iterations=10) - 0.064) < 1e-12
    assert get_two_stage_annealed_loss_weight(1.0, 10, total_iterations=10) == 0.01


def test_gate_proj_bias_l2_default_schedule_uses_half_run_and_two_stage_floors():
    source = BASE_TRAIN.read_text()

    assert 'parser.add_argument("--aux-loss-weight-final-frac", type=float, default=0.1' in source
    assert 'orig_model.config.aux_loss_weight = aux_loss_weight' in source
    assert 'log_data["train/aux_loss_weight"] = aux_loss_weight' in source
    assert 'parser.add_argument("--gate-proj-bias-l2-loss-stage1-frac", "--gate-proj-bias-l2-loss-floor-frac", dest="gate_proj_bias_l2_loss_stage1_frac", type=float, default=0.1' in source
    assert '--gate-proj-bias-l2-loss-final-frac", type=float, default=0.02' in source
    assert 'stage1_iterations = max((effective_total_iterations + 1) // 2, 1)' in source