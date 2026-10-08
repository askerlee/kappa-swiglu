import ast
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN_MIX = ROOT / "scripts" / "base_train_mix.py"


@pytest.mark.parametrize("script_name", ("base_train.py", "base_train_mix.py"))
def test_training_script_rejects_removed_training_options(script_name):
    script_path = ROOT / "scripts" / script_name
    source = script_path.read_text()
    tree = ast.parse(source, filename=str(script_path))
    parser_nodes = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets)
        ):
            break
        if isinstance(node, ast.FunctionDef) and node.name == "str2bool":
            parser_nodes.append(node)
        elif isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id in {"parser", "DEFAULT_SEED", "AUX_LOSS_WEIGHT_DEFAULT"}
                for target in node.targets
            ):
                parser_nodes.append(node)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "parser"
        ):
            parser_nodes.append(node)
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), str(script_path), "exec"), namespace)
    parser = namespace["parser"]
    parser.parse_args([])
    removed_options = [
        "--mockup-mode", "--fp8", "--fp8-recipe",
        "--kappa-ema-rms-reg", "--kappa-l2-ema-beta",
        "--kappa-l2-ema-anchor-start", "--kappa-l2-ema-anchor-end",
        "--kappa-l2-ema-floor-frac",
    ]
    for option in removed_options:
        assert option not in parser.format_help()
        with pytest.raises(SystemExit) as error:
            parser.parse_args([option])
        assert error.value.code == 2
    assert "mockup" not in source.lower()
    assert "fp8" not in source.lower()


def load_function_from_script(function_name, script_path=BASE_TRAIN_MIX):
    source = script_path.read_text()
    module = ast.parse(source, filename=str(script_path))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace = {"torch": torch}
            exec(compile(function_module, filename=str(script_path), mode="exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name} not found in {script_path}")


def test_cached_independent_kappa_statistics():
    from nanochat.configuration_nanomoe_gpt import GPTConfig
    from nanochat.gpt import GPT

    model = GPT(GPTConfig(
        n_layer=1, n_head=2, n_embd=32, n_exp=2, vocab_size=64,
        sequence_len=8, moe_start_layer=0, use_kappa_swiglu=True,
        independent_kappa_router=True,
    ))
    model.init_weights()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    experts = model.transformer.h[0].mlp.experts
    with torch.no_grad():
        experts.kappa_bias[0, 0].fill_(2.0)
        experts.kappa_bias[0, 1].fill_(-6.0)
    logits = torch.tensor([[0.0, 2.0, 99.0], [-3.0, 99.0, 99.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    for script_name in ('base_train_mix.py', 'base_train.py', 'chat_sft.py'):
        experts._materialize_kappa_scale(0, selected_gate_scores=logits, valid_score_mask=mask)
        collect_stats = load_function_from_script('collect_weight_grad_stats', ROOT / 'scripts' / script_name)
        collect_stats.__globals__.update({
            'math': math,
            'get_dense_kappa_bias_stat_layer_indices': lambda model: [],
        })
        losses = {'expert_utilities': torch.tensor([[0.8, 0.2]])}
        collect_stats(model, losses, [0])
        if script_name != 'chat_sft.py':
            assert math.isclose(losses['kappa_scale_mean_0'], -1.0 / 3.0, rel_tol=1e-6)
            assert math.isclose(losses['kappa_scale_abs_mean_0'], 5.0 / 3.0, rel_tol=1e-6)
            assert losses['kappa_scale_mean_top_0'] == 1.0
            assert losses['kappa_scale_mean_bottom_0'] == -3.0
        assert losses['kappa_bias_mean_0'] == -2.0
        assert losses['kappa_bias_mean_top_0'] == 2.0
        assert losses['kappa_bias_mean_bottom_0'] == -6.0
        experts._cached_kappa_scale = None
        losses = {'expert_utilities': torch.tensor([[0.8, 0.2]])}
        collect_stats(model, losses, [0])
        assert 'kappa_scale_mean_0' not in losses
        assert losses['kappa_bias_mean_0'] == -2.0


def test_should_use_chat_sft_step_runs_only_on_positive_multiples():
    should_use_chat_sft_step = load_function_from_script("should_use_chat_sft_step")

    assert should_use_chat_sft_step(0, 10) is False
    assert should_use_chat_sft_step(9, 10) is False
    assert should_use_chat_sft_step(10, 10) is True
    assert should_use_chat_sft_step(20, 10) is True
    assert should_use_chat_sft_step(10, -1) is False


def test_kappa_router_sft_l2_logging_retains_latest_sft_value():
    tree = ast.parse(BASE_TRAIN_MIX.read_text())
    cache_name = "latest_kappa_router_sft_l2_loss"
    initialization = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == cache_name for target in node.targets)
    )
    training_loop = next(node for node in tree.body if isinstance(node, ast.While))
    cache_update = next(
        node for node in training_loop.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name) and node.test.id == "is_chat_sft_step"
        and any(
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == cache_name for target in statement.targets)
            for statement in node.body
        )
    )
    log_value = next(
        value for node in ast.walk(training_loop) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "train/kappa_router_sft_l2_loss_step"
    )
    namespace = {}
    exec(compile(ast.Module(body=[initialization], type_ignores=[]), str(BASE_TRAIN_MIX), "exec"), namespace)
    update_code = compile(ast.Module(body=[cache_update], type_ignores=[]), str(BASE_TRAIN_MIX), "exec")
    log_code = compile(ast.Expression(body=log_value), str(BASE_TRAIN_MIX), "eval")
    for is_sft, current_value, expected_value in (
        (False, 0.0, 0.0),
        (True, 0.25, 0.25),
        (False, 0.0, 0.25),
        (True, 0.5, 0.5),
        (False, 0.0, 0.5),
        (True, 0.0, 0.0),
    ):
        current_loss = torch.tensor(current_value, requires_grad=True)
        namespace.update(is_chat_sft_step=is_sft, losses={"kappa_router_sft_l2_loss": current_loss})
        exec(update_code, namespace)
        logged_value = eval(log_code, namespace)
        assert float(logged_value) == expected_value
        assert float(current_loss.detach()) == current_value
        if torch.is_tensor(logged_value):
            assert not logged_value.requires_grad
            assert logged_value.data_ptr() != current_loss.data_ptr()


def test_kappa_delay_freezes_lr_and_delays_slope_scale_warmup():
    get_kappa_slope_max_scale = load_function_from_script("get_kappa_slope_max_scale")
    get_kappa_slope_max_scale.__globals__["math"] = math
    get_kappa_bias_lr_scale = load_function_from_script("get_kappa_bias_lr_scale")
    get_kappa_bias_lr_scale.__globals__["get_linear_lr_scale"] = load_function_from_script(
        "get_linear_lr_scale"
    )
    optimizer = SimpleNamespace(param_groups=[{
        "name": "kappa_params",
        "kind": "adamw",
        "kappa_param_delay_start_iterations": 20,
        "lr_scale_warmup_iterations": 10,
    }])

    assert get_kappa_bias_lr_scale(optimizer, 10, 100) == 0.0
    assert get_kappa_bias_lr_scale(optimizer, 19, 100) == 0.0
    assert get_kappa_bias_lr_scale(optimizer, 25, 100) == 0.5
    for target in (3.0, 2.0):
        for step in (0, 5, 19, 20):
            assert get_kappa_slope_max_scale(target, step, 100, delay_iterations=20) == 1.0
        assert get_kappa_slope_max_scale(target, 25, 100, delay_iterations=20) == (1.0 + target) / 2
        assert get_kappa_slope_max_scale(target, 30, 100, delay_iterations=20) == target

    tree = ast.parse(BASE_TRAIN_MIX.read_text())
    slope_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_kappa_slope_max_scale"
    ]
    assert len(slope_calls) == 2
    assert all(
        any(
            keyword.arg == "delay_iterations"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "kappa_param_delay_start_iterations"
            for keyword in call.keywords
        )
        for call in slope_calls
    )


def test_zero_kappa_lr_preserves_parameters_and_accumulates_adamw_moments():
    from nanochat.optim import adamw_step_fused

    parameter = torch.tensor([1.0, 2.0])
    original_parameter = parameter.clone()
    gradient = torch.tensor([2.0, -3.0])
    exp_avg = torch.zeros_like(parameter)
    exp_avg_sq = torch.zeros_like(parameter)
    for step in (1, 2):
        adamw_step_fused.__wrapped__(
            parameter, gradient, exp_avg, exp_avg_sq,
            torch.tensor(float(step)), torch.tensor(0.0),
            torch.tensor(0.8), torch.tensor(0.95),
            torch.tensor(1e-10), torch.tensor(0.1),
        )

    torch.testing.assert_close(parameter, original_parameter, rtol=0, atol=0)
    torch.testing.assert_close(exp_avg, gradient * (1 - 0.8 ** 2))
    torch.testing.assert_close(exp_avg_sq, gradient.square() * (1 - 0.95 ** 2))


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


def test_kappa_swiglu_sft_only_defaults_and_overrides():
    import argparse

    tree = ast.parse(BASE_TRAIN_MIX.read_text())
    options = ("--use-kappa-swiglu", "--use-kappa-swiglu-sft-only", "--separate-base-sft-kappa")
    parser = argparse.ArgumentParser()
    declarations = [
        node for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "add_argument"
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value in options
    ]
    namespace = {"parser": parser, "str2bool": load_function_from_script("str2bool")}
    namespace["str2bool"].__globals__["argparse"] = argparse
    exec(compile(ast.Module(body=declarations, type_ignores=[]),
                 filename=str(BASE_TRAIN_MIX), mode="exec"), namespace)
    normalization = [
        node for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
        and node.test.attr in ("separate_base_sft_kappa", "use_kappa_swiglu_sft_only")
    ]
    for argv, expected in (
        ([], (True, False)),
        (["--use-kappa-swiglu-sft-only"], (True, True)),
        (["--use-kappa-swiglu-sft-only=false"], (True, False)),
        (["--separate-base-sft-kappa=false"], (True, True)),
        (["--separate-base-sft-kappa=false", "--use-kappa-swiglu-sft-only=false"], (False, False)),
    ):
        args = parser.parse_args(argv)
        namespace.update(args=args, sys=SimpleNamespace(argv=["script", *argv]),
                         arg_was_explicitly_set=load_function_from_script("arg_was_explicitly_set"))
        exec(compile(ast.Module(body=normalization, type_ignores=[]),
                     filename=str(BASE_TRAIN_MIX), mode="exec"), namespace)
        assert (args.use_kappa_swiglu, args.use_kappa_swiglu_sft_only) == expected


def test_separate_kappa_routes_mixed_training_and_base_evaluation():
    source = BASE_TRAIN_MIX.read_text()
    assert 'parser.add_argument("--separate-base-sft-kappa"' in source
    assert 'separate_base_sft_kappa=args.separate_base_sft_kappa' in source
    assert 'orig_model.set_kappa_training_phase(is_chat_sft_step)' in source
    assert 'orig_model.set_kappa_training_phase(False)' in source
    assert 'group["active_kappa_slot"] = int(is_chat_sft_step)' in source


def test_base_inference_disables_sft_only_kappa_and_restores_training_phase():
    tree = ast.parse(BASE_TRAIN_MIX.read_text())
    training_loop = next(node for node in tree.body if isinstance(node, ast.While))
    for evaluation_name in ("evaluate_bpb", "evaluate_core", "generate_batch"):
        evaluation_block = next(
            node for node in training_loop.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and (
                    isinstance(child.func, ast.Name) and child.func.id == evaluation_name
                    or isinstance(child.func, ast.Attribute) and child.func.attr == evaluation_name
                )
                for child in ast.walk(node)
            )
        )
        if evaluation_name == "generate_batch":
            evaluation_block = next(
                node for node in evaluation_block.body
                if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "master_process"
            )
        transitions = [
            node for node in evaluation_block.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr in (
                "set_kappa_training_phase", "set_kappa_swiglu_enabled",
            )
        ]
        for enabled in (False, True):
            for sft_only in (False, True):
                for step in (2500, 2750, 5500, 8250):
                    events = []
                    model = SimpleNamespace(
                        kappa_swiglu_enabled=enabled,
                        set_kappa_training_phase=lambda phase: events.append(("phase", phase)),
                        set_kappa_swiglu_enabled=lambda value: events.append(("enabled", value)),
                    )
                    namespace = {
                        "orig_model": model,
                        "args": SimpleNamespace(use_kappa_swiglu=enabled, use_kappa_swiglu_sft_only=sft_only),
                        "kappa_swiglu_training_enabled": enabled and (not sft_only or step % 11 == 0),
                        "is_chat_sft_step": step % 11 == 0,
                    }
                    exec(compile(ast.Module(body=transitions, type_ignores=[]),
                                 filename=str(BASE_TRAIN_MIX), mode="exec"), namespace)
                    assert events == [
                        ("phase", False), ("enabled", enabled and not sft_only),
                        ("phase", step % 11 == 0),
                        ("enabled", enabled and (not sft_only or step % 11 == 0)),
                    ]


def test_kappa_logging_uses_current_step_metrics():
    source = BASE_TRAIN_MIX.read_text()

    assert "last_chat_sft_kappa_metrics" not in source
    assert "overlay_last_chat_sft_kappa_metrics" not in source


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
        "muonh",
    )

    assert argv[-6:] == [
        "--device-batch-size",
        "24",
        "--max-seq-len",
        "2048",
        "--matrix-optimizer",
        "muonh",
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