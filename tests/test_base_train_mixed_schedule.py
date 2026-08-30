import ast
from pathlib import Path
from types import SimpleNamespace


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


def test_get_task_mixture_source_resolves_shuffled_index():
    get_task_mixture_source = load_function_from_script("get_task_mixture_source")
    dataset = SimpleNamespace(
        index_map=[(1, 7), (0, 3)],
        tasks=[SimpleNamespace(), []],
    )

    assert get_task_mixture_source(dataset, 0) == {
        "mixture_index": 0,
        "task_index": 1,
        "task_name": "list",
        "local_index": 7,
    }


def test_chat_sft_steps_keep_base_train_capacity():
    source = BASE_TRAIN_MIX.read_text()

    assert "chat_sft_train_capacity" not in source
    assert "--chat-sft-train-capacity" not in source
    assert "set_train_capacity" not in source


def test_router_wg_delta_updates_only_on_mixed_chat_sft_steps():
    source = BASE_TRAIN_MIX.read_text()

    assert 'parser.add_argument("--router-wg-delta", action="store_true"' in source
    assert "model.setup_router_wg_delta()" in source
    assert "orig_model.enable_router_wg_delta(is_chat_sft_step)" in source
    assert 'group.get("name") == "router_wg_delta" and not is_chat_sft_step' in source
    assert 'group.get("name") == "router_wg_base" and is_chat_sft_step' not in source
    assert 'group["initial_lr"] * lrm if is_chat_sft_step else 0.0' in source
    assert 'group["lr"] = group["initial_lr"] * lrm' in source
    assert 'args.router_wg_delta = bool(getattr(model.config, "router_wg_delta", False))' in source
    assert "if is_chat_sft_step:" in source
    assert 'loss = loss + args.router_wg_delta_l2_loss_weight * router_wg_delta_l2_loss' in source
    assert '"train/router_wg_delta_l2_loss_step"' in source


def test_kappa_swiglu_can_run_only_on_mixed_chat_sft_steps():
    source = BASE_TRAIN_MIX.read_text()

    assert 'parser.add_argument("--use-kappa-swiglu-sft-only"' in source
    assert "if args.use_kappa_swiglu_sft_only:" in source
    assert "args.use_kappa_swiglu = True" in source
    assert "orig_model.set_kappa_swiglu_enabled(" in source
    assert "is_chat_sft_step if args.use_kappa_swiglu_sft_only else True" in source


def test_get_compile_rebuild_plan_rebuilds_before_resuming_training():
    get_compile_rebuild_plan = load_function_from_script("get_compile_rebuild_plan")

    assert get_compile_rebuild_plan(False, False, False, False) == (False, False)
    assert get_compile_rebuild_plan(True, True, False, False) == (True, False)
    assert get_compile_rebuild_plan(True, False, True, False) == (False, True)
    assert get_compile_rebuild_plan(True, False, True, True) == (False, False)


def test_chat_sft_continuation_inherits_shape_without_changing_total_batch_size():
    build_chat_sft_exec_argv = load_function_from_script("build_chat_sft_exec_argv")

    argv = build_chat_sft_exec_argv(
        "/usr/bin/python3",
        "d8-mixed",
        120,
        24,
        2048,
    )

    assert argv[-4:] == [
        "--device-batch-size",
        "24",
        "--max-seq-len",
        "2048",
    ]
    assert "--total-batch-size" not in argv


def test_mixed_interval_throughput_averages_all_steps_since_previous_log():
    get_interval_throughput = load_function_from_script("get_interval_throughput")

    average_dt, tok_per_sec, mfu = get_interval_throughput(
        total_batch_size=1_000,
        num_flops_per_token=2_000,
        gpu_peak_flops=10_000_000,
        ddp_world_size=2,
        interval_steps=4,
        interval_time=2.0,
    )

    assert average_dt == 0.5
    assert tok_per_sec == 2_000
    assert mfu == 20.0


def test_mixed_logged_throughput_uses_interval_values_and_resets_window():
    source = BASE_TRAIN_MIX.read_text()

    assert '"tok_per_sec": logged_tok_per_sec' in source
    assert '"mfu": logged_mfu' in source
    assert '"dt": logged_dt' in source
    assert "throughput_interval_steps += 1" in source
    assert "throughput_interval_steps = 0" in source
    assert "throughput_interval_time = 0.0" in source


def test_mixed_script_persists_separate_chat_sft_loader_state():
    source = BASE_TRAIN_MIX.read_text()

    assert '"chat_sft_dataloader_state_dict": chat_sft_dataloader_state_dict' in source
    assert 'if is_chat_sft_step:' in source
    assert 'checkpoint_dir = os.path.join(base_dir, "base_mixed_checkpoints", output_dirname)' in source


def test_mixed_script_logs_chat_sft_loss_separately_from_base_loss():
    source = BASE_TRAIN_MIX.read_text()

    assert 'log_data["train/chat_sft_ntp_loss_step"] = scalar_loss_to_item(losses[\'ntp_loss\'])' in source
    assert 'log_data["train/loss_step"] = debiased_smooth_loss' in source


def test_mixed_script_adds_focused_chat_sft_datasets():
    tree = ast.parse(BASE_TRAIN_MIX.read_text(), filename=str(BASE_TRAIN_MIX))
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
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_chat_sft_train_dataset"
    )
    guarded_calls = {
        node.test.id: {
            call.func.id
            for statement in node.body
            for call in ast.walk(statement)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        for node in builder.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
    }
    builder_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_chat_sft_train_dataset"
    )
    call_keywords = {keyword.arg: keyword.value for keyword in builder_call.keywords}

    assert defaults["default"] is True
    assert {"Tulu3SFTMixture", "Tulu3SFTPersonaIF"} <= guarded_calls["use_tulu3_sft_mixture"]
    assert "UltraDataSFTIF" in guarded_calls["use_ultradata_sft_if"]
    assert isinstance(call_keywords["tulu3_english_only"], ast.Attribute)
    assert call_keywords["tulu3_english_only"].attr == "tulu3_english_only"
    assert isinstance(call_keywords["use_ultradata_sft_if"], ast.Attribute)
    assert call_keywords["use_ultradata_sft_if"].attr == "use_ultradata_sft_if"