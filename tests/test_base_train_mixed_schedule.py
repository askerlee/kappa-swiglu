import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN_MIX = ROOT / "scripts" / "base_train_mix.py"


def load_function_from_script(function_name):
    source = BASE_TRAIN_MIX.read_text()
    module = ast.parse(source, filename=str(BASE_TRAIN_MIX))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(function_module, filename=str(BASE_TRAIN_MIX), mode="exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name} not found in {BASE_TRAIN_MIX}")


def test_should_use_chat_sft_step_runs_only_on_positive_multiples():
    should_use_chat_sft_step = load_function_from_script("should_use_chat_sft_step")

    assert should_use_chat_sft_step(0, 10) is False
    assert should_use_chat_sft_step(9, 10) is False
    assert should_use_chat_sft_step(10, 10) is True
    assert should_use_chat_sft_step(20, 10) is True
    assert should_use_chat_sft_step(10, -1) is False


def test_mixed_script_persists_separate_chat_sft_loader_state():
    source = BASE_TRAIN_MIX.read_text()

    assert '"chat_sft_dataloader_state_dict": chat_sft_dataloader_state_dict' in source
    assert 'if is_chat_sft_step:' in source
    assert 'checkpoint_dir = os.path.join(base_dir, "base_mixed_checkpoints", output_dirname)' in source


def test_mixed_script_logs_chat_sft_loss_separately_from_base_loss():
    source = BASE_TRAIN_MIX.read_text()

    assert 'log_data["train/chat_sft_ntp_loss_step"] = scalar_loss_to_item(losses[\'ntp_loss\'])' in source
    assert 'log_data["train/loss_step"] = debiased_smooth_loss' in source