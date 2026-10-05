import ast
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = ROOT / "scripts" / "base_train.py"
BASE_TRAIN_MIX = ROOT / "scripts" / "base_train_mix.py"


def load_function_from_script(function_name, script=BASE_TRAIN):
    source = script.read_text()
    module = ast.parse(source, filename=str(script))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(function_module, filename=str(BASE_TRAIN), mode="exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name} not found in {BASE_TRAIN}")


def test_kappa_router_lr_delay_is_applied_in_all_training_scripts():
    for script in (BASE_TRAIN, BASE_TRAIN_MIX, ROOT / "scripts" / "chat_sft.py"):
        module = ast.parse(script.read_text(), filename=str(script))
        router_branch = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Constant) and child.value == "kappa_router"
                for child in ast.walk(node.test)
            )
        )
        branch_module = ast.Module(body=[router_branch], type_ignores=[])
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        parameter.grad = torch.ones_like(parameter)
        group = {
            "name": "kappa_router", "initial_lr": 0.01,
            "kappa_param_delay_start_iterations": 20, "params": [parameter],
        }
        for step, expected_lr in ((0, 0.0), (19, 0.0), (20, 0.005), (21, 0.005)):
            exec(compile(branch_module, filename=str(script), mode="exec"), {
                "group": group, "step": step, "lrm": 0.5,
            })
            assert group["lr"] == expected_lr
            assert parameter.grad is not None


def test_kappa_bias_from_scale_overrides_only_implicit_default_l2_weight():
    cases = [
        (False, [], 0.01, 0.01),
        (True, [], 0.01, 0.002),
        (True, ["--kappa-l2-loss-weight", "0.01"], 0.01, 0.01),
        (True, ["--kappa-l2-loss-weight=0.01"], 0.01, 0.01),
        (True, ["--kappa-l2-loss-weight", "0.02"], 0.02, 0.02),
        (True, ["--kappa-l2-loss-weight=0"], 0.0, 0.0),
    ]
    for script in (BASE_TRAIN, BASE_TRAIN_MIX):
        module = ast.parse(script.read_text(), filename=str(script))
        weight_arg = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
            and node.args and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--kappa-l2-loss-weight"
        )
        default = next(keyword.value for keyword in weight_arg.keywords if keyword.arg == "default")
        assert ast.literal_eval(default) == 0.01
        adjustment = next(
            node for node in module.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Attribute) and child.attr == "kappa_bias_from_scale"
                for child in ast.walk(node.test)
            )
        )
        adjustment_module = ast.Module(body=[adjustment], type_ignores=[])
        for independent_router in (False, True):
            for enabled, argv, weight, expected in cases:
                args = SimpleNamespace(
                    kappa_bias_from_scale=enabled,
                    independent_kappa_router=independent_router,
                    kappa_l2_loss_weight=weight,
                )
                namespace = {
                    "args": args,
                    "sys": SimpleNamespace(argv=[str(script), *argv]),
                    "arg_was_explicitly_set": load_function_from_script("arg_was_explicitly_set", script),
                }
                exec(compile(adjustment_module, filename=str(script), mode="exec"), namespace)
                assert args.kappa_l2_loss_weight == (0.001 if independent_router and not argv else expected)


def test_independent_kappa_router_bias_l2_weight_is_ten_times_base_weight():
    for script in (BASE_TRAIN, BASE_TRAIN_MIX):
        module = ast.parse(script.read_text(), filename=str(script))
        base_assignment = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "get_two_stage_annealed_loss_weight"
            and any(isinstance(target, ast.Name) and target.id in {"kappa_l2_loss_weight", "kappa_bias_l2_loss_weight"} for target in node.targets)
        )
        scale_assignment = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "kappa_scale_l2_loss_weight" for target in node.targets)
        )
        adjustment = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute) and node.test.attr == "independent_kappa_router"
            and any(isinstance(child, ast.Name) and child.id == "kappa_bias_l2_loss_weight" for child in ast.walk(node))
        )
        assert base_assignment.lineno < scale_assignment.lineno < adjustment.lineno
        weight_module = ast.Module(body=[base_assignment, scale_assignment, adjustment], type_ignores=[])
        for independent_router in (False, True):
            for base_weight in (0.0, 0.001, 0.02):
                for step in (0, 50, 100):
                    args = SimpleNamespace(
                        independent_kappa_router=independent_router,
                        kappa_l2_loss_weight=base_weight,
                        kappa_scale_l2_loss_weight_scale=2.0,
                        kappa_l2_loss_stage1_frac=0.5,
                        kappa_l2_loss_final_frac=0.1,
                    )
                    anneal = load_function_from_script("get_two_stage_annealed_loss_weight", script)
                    namespace = {
                        "args": args, "step": step, "num_iterations": 100,
                        "kappa_l2_stage1_iterations": 50,
                        "get_two_stage_annealed_loss_weight": anneal,
                    }
                    exec(compile(weight_module, filename=str(script), mode="exec"), namespace)
                    scheduled_weight = anneal(base_weight, step, 100, 50, 0.5, 0.1)
                    assert namespace["kappa_bias_l2_loss_weight"] == scheduled_weight * (10 if independent_router else 1)
                    assert namespace["kappa_scale_l2_loss_weight"] == scheduled_weight * 2.0


def test_base_train_separates_compute_and_parameter_storage_dtypes():
    for script in (BASE_TRAIN, BASE_TRAIN_MIX):
        source = script.read_text()

        dtype_arg_index = source.index('parser.add_argument("--dtype"')
        parameter_dtype_arg_index = source.index('parser.add_argument("--parameter-dtype"')
        dtype_resolution_index = source.index('ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16')
        parameter_dtype_resolution_index = source.index(
            'parameter_dtype = torch.bfloat16 if args.parameter_dtype == "bfloat16" else torch.float32'
        )
        autocast_index = source.index('dtype=ptdtype')
        cast_index = source.index("cast_model_parameters(model, parameter_dtype, embedding_dtype=embedding_dtype)")
        compile_index = source.index("model = build_training_model(orig_model, args.compile)", cast_index)
        optimizer_index = source.index("optimizer = model.setup_optimizer(", compile_index)

        assert 'default="reference", choices=("reference", "float32", "bfloat16")' in source[parameter_dtype_arg_index:parameter_dtype_arg_index + 180]
        assert source.count("cast_model_parameters(model, parameter_dtype, embedding_dtype=embedding_dtype)") == 1
        assert dtype_arg_index < parameter_dtype_arg_index < dtype_resolution_index
        assert dtype_resolution_index < parameter_dtype_resolution_index < autocast_index < cast_index
        assert cast_index < compile_index < optimizer_index


def test_base_train_scalar_lr_defaults_to_x0_learning_rate():
    for script in (BASE_TRAIN, BASE_TRAIN_MIX):
        source = script.read_text()

        assert 'parser.add_argument("--scalar-lr", type=float, default=0.05' in source
        assert "scalar_lr=args.scalar_lr * batch_lr_scale" in source


def test_base_train_does_not_monitor_kappa_gradient_correlation():
    for script in (BASE_TRAIN, BASE_TRAIN_MIX):
        source = script.read_text()

        assert "kappa_grad_correlation" not in source
        assert "def gradient_correlation(" not in source


def test_resolve_loss_chunk_tokens_limits_compiled_logits_to_32_mib():
    auto_args = SimpleNamespace(loss_chunk_tokens=-1, compile=True)
    explicit_args = SimpleNamespace(loss_chunk_tokens=256, compile=True)

    for script in (BASE_TRAIN, BASE_TRAIN_MIX):
        resolve_loss_chunk_tokens = load_function_from_script("resolve_loss_chunk_tokens", script)
        assert resolve_loss_chunk_tokens(auto_args, ddp_world_size=1, vocab_size=32768) == (512, True)
        assert resolve_loss_chunk_tokens(auto_args, ddp_world_size=2, vocab_size=32768) == (512, True)
        assert resolve_loss_chunk_tokens(explicit_args, ddp_world_size=2, vocab_size=32768) == (256, False)


def test_get_annealed_loss_weight_drops_to_floor_in_first_500_steps_then_stays_there():
    get_annealed_loss_weight = load_function_from_script("get_annealed_loss_weight")

    assert get_annealed_loss_weight(0.002, 0, final_weight=0.001) == 0.002
    assert abs(get_annealed_loss_weight(0.002, 250, final_weight=0.001) - 0.0015) < 1e-12
    assert get_annealed_loss_weight(0.002, 500, final_weight=0.001) == 0.001
    assert get_annealed_loss_weight(0.002, 900, final_weight=0.001) == 0.001


def test_interval_throughput_averages_all_steps_since_previous_log():
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


def test_resume_lr_warmup_scales_then_restores_absolute_schedule():
    for script in (BASE_TRAIN, BASE_TRAIN_MIX):
        get_resume_lr_warmup_scale = load_function_from_script(
            "get_resume_lr_warmup_scale",
            script,
        )

        assert get_resume_lr_warmup_scale(4000, -1, 50) == 1.0
        assert get_resume_lr_warmup_scale(4000, 4000, 0) == 1.0
        assert get_resume_lr_warmup_scale(4000, 4000, 50) == 0.02
        assert get_resume_lr_warmup_scale(4024, 4000, 50) == 0.5
        assert get_resume_lr_warmup_scale(4049, 4000, 50) == 1.0
        assert get_resume_lr_warmup_scale(4050, 4000, 50) == 1.0

        source = script.read_text()
        assert 'parser.add_argument("--resume-lr-warmup-steps"' in source
        assert source.count("lrm *= get_resume_lr_warmup_scale(") == 1


def test_resume_lr_warmup_freezes_kappa_until_warmup_finishes():
    for script in (BASE_TRAIN, BASE_TRAIN_MIX):
        get_resume_kappa_lr_scale = load_function_from_script(
            "get_resume_kappa_lr_scale",
            script,
        )

        assert get_resume_kappa_lr_scale(4000, -1, 50) == 1.0
        assert get_resume_kappa_lr_scale(4000, 4000, 0) == 1.0
        assert get_resume_kappa_lr_scale(4000, 4000, 50) == 0.0
        assert get_resume_kappa_lr_scale(4049, 4000, 50) == 0.0
        assert get_resume_kappa_lr_scale(4050, 4000, 50) == 1.0

        source = script.read_text()
        assert "* kappa_bias_lr_scale * resume_kappa_lr_scale" in source
        assert 'if resume_kappa_lr_scale == 0.0:' in source
        assert 'for param in group["params"]:' in source
        assert "param.grad = None" in source


def test_logged_throughput_uses_interval_values_and_resets_window():
    source = BASE_TRAIN.read_text()

    assert '"tok_per_sec": logged_tok_per_sec' in source
    assert '"mfu": logged_mfu' in source
    assert '"dt": logged_dt' in source
    assert "throughput_interval_steps += 1" in source
    assert "throughput_interval_steps = 0" in source
    assert "throughput_interval_time = 0.0" in source


def test_kappa_bias_l2_two_stage_schedule_uses_half_run_then_decays_to_final_floor():
    get_two_stage_annealed_loss_weight = load_function_from_script("get_two_stage_annealed_loss_weight")

    assert get_two_stage_annealed_loss_weight(1.0, 0, total_iterations=10) == 1.0
    assert get_two_stage_annealed_loss_weight(1.0, 5, total_iterations=10) == 0.1
    assert abs(get_two_stage_annealed_loss_weight(1.0, 7, total_iterations=10) - 0.064) < 1e-12
    assert get_two_stage_annealed_loss_weight(1.0, 10, total_iterations=10) == 0.01


def test_kappa_bias_l2_two_stage_schedule_can_increase_during_stage_2():
    get_two_stage_annealed_loss_weight = load_function_from_script("get_two_stage_annealed_loss_weight")

    assert get_two_stage_annealed_loss_weight(
        1.0,
        5,
        total_iterations=10,
        stage1_floor_frac=0.1,
        final_floor_frac=0.4,
    ) == 0.1
    assert abs(
        get_two_stage_annealed_loss_weight(
            1.0,
            7,
            total_iterations=10,
            stage1_floor_frac=0.1,
            final_floor_frac=0.4,
        ) - 0.22
    ) < 1e-12
    assert get_two_stage_annealed_loss_weight(
        1.0,
        10,
        total_iterations=10,
        stage1_floor_frac=0.1,
        final_floor_frac=0.4,
    ) == 0.4


def test_kappa_slope_max_scale_anneals_from_one_to_target_during_initial_fraction():
    get_kappa_slope_max_scale = load_function_from_script("get_kappa_slope_max_scale")

    assert get_kappa_slope_max_scale(3.0, 0, total_iterations=100, warmup_iteration_frac=0.1) == 1.0
    assert get_kappa_slope_max_scale(3.0, 5, total_iterations=100, warmup_iteration_frac=0.1) == 2.0
    assert get_kappa_slope_max_scale(3.0, 10, total_iterations=100, warmup_iteration_frac=0.1) == 3.0
    assert get_kappa_slope_max_scale(3.0, 50, total_iterations=100, warmup_iteration_frac=0.1) == 3.0


def test_kappa_slope_max_scale_stays_at_one_during_delay_then_anneals():
    get_kappa_slope_max_scale = load_function_from_script("get_kappa_slope_max_scale")

    assert get_kappa_slope_max_scale(3.0, 0, total_iterations=100, warmup_iteration_frac=0.1, delay_iterations=20) == 1.0
    assert get_kappa_slope_max_scale(3.0, 19, total_iterations=100, warmup_iteration_frac=0.1, delay_iterations=20) == 1.0
    assert get_kappa_slope_max_scale(3.0, 20, total_iterations=100, warmup_iteration_frac=0.1, delay_iterations=20) == 1.0
    assert get_kappa_slope_max_scale(3.0, 25, total_iterations=100, warmup_iteration_frac=0.1, delay_iterations=20) == 2.0
    assert get_kappa_slope_max_scale(3.0, 30, total_iterations=100, warmup_iteration_frac=0.1, delay_iterations=20) == 3.0


def test_build_chat_sft_exec_argv_pins_final_checkpoint_and_splits_extra_args():
    build_chat_sft_exec_argv = load_function_from_script("build_chat_sft_exec_argv")

    argv = build_chat_sft_exec_argv(
        "/usr/bin/python3",
        "d8",
        120,
        16,
        2048,
        "--device-batch-size 8 --model-save-tag after-base",
    )

    assert argv == [
        "/usr/bin/python3",
        "-m",
        "scripts.chat_sft",
        "--log-grad-stats",
        "--model-tag",
        "d8",
        "--model-step",
        "120",
        "--device-batch-size",
        "16",
        "--max-seq-len",
        "2048",
        "--device-batch-size",
        "8",
        "--model-save-tag",
        "after-base",
    ]


def test_pick_free_tcp_port_returns_valid_port_number():
    pick_free_tcp_port = load_function_from_script("pick_free_tcp_port")

    port = pick_free_tcp_port()

    assert isinstance(port, int)
    assert 0 < port < 65536


def test_get_compile_rebuild_plan_defers_one_time_rebuild_until_after_eager_step():
    get_compile_rebuild_plan = load_function_from_script("get_compile_rebuild_plan")

    assert get_compile_rebuild_plan(False, False, False, False) == (False, False)
    assert get_compile_rebuild_plan(True, True, False, False) == (True, False)
    assert get_compile_rebuild_plan(True, False, True, False) == (False, True)
    assert get_compile_rebuild_plan(True, False, True, True) == (False, False)


def test_kappa_bias_l2_default_schedule_uses_half_run_and_two_stage_floors():
    source = BASE_TRAIN.read_text()

    assert 'parser.add_argument("--aux-loss-weight", type=float, default=1e-3' in source
    assert 'parser.add_argument("--aux-loss-weight-init-scale", type=float, default=2.0' in source
    assert 'parser.add_argument("--aux-loss-weight-init-anneal-iterations", type=int, default=500' in source
    assert 'orig_model.config.aux_loss_weight = aux_loss_weight' in source
    assert 'log_data["train/aux_loss_weight"] = aux_loss_weight' in source
    assert 'args.aux_loss_weight * args.aux_loss_weight_init_scale' in source
    assert 'num_anneal_iterations=args.aux_loss_weight_init_anneal_iterations' in source
    assert 'final_weight=args.aux_loss_weight' in source
    assert '--use-kappa-swiglu-as-lr-scaler' not in source
    assert 'parser.add_argument("--kappa-l2-loss-stage1-frac", dest="kappa_l2_loss_stage1_frac", type=float, default=0.1' in source
    assert '--kappa-l2-loss-final-frac", dest="kappa_l2_loss_final_frac", type=float, default=0.02' in source
    assert 'stage1_iterations = max((effective_total_iterations + 1) // 2, 1)' in source
    assert 'parser.add_argument("--continue-to-chat-sft", action="store_true"' in source
    assert 'parser.add_argument("--continue-to-chat-sft-args", type=str, default=""' in source
    assert 'should_continue_to_chat_sft = args.continue_to_chat_sft and step == num_iterations' in source
    assert 'chat_sft_master_port = prepare_chat_sft_rendezvous(ddp, ddp_rank, device)' in source
    assert 'os.environ["MASTER_PORT"] = str(chat_sft_master_port)' in source
    assert 'torch.distributed.broadcast(port_tensor, src=0)' in source
    assert 'os.execvp(chat_sft_argv[0], chat_sft_argv)' in source


def test_base_train_removes_kappa_ema_rms_reg_and_preserves_ordinary_l2():
    from inspect import signature
    from nanochat.configuration_nanomoe_gpt import GPTConfig

    removed_fields = (
        "kappa_bias_ema_rms_reg",
        "kappa_bias_l2_ema_beta",
        "kappa_bias_l2_ema_anchor_start",
        "kappa_bias_l2_ema_anchor_end",
        "kappa_bias_l2_ema_floor_frac",
    )
    for config in (GPTConfig(), GPTConfig(**dict.fromkeys(removed_fields, True))):
        for name in removed_fields:
            assert name not in signature(GPTConfig).parameters
            assert not hasattr(config, name)
        assert config.kappa_bias_l2_loss_weight == 0.0
    for script in (BASE_TRAIN, BASE_TRAIN_MIX):
        source = script.read_text()
        assert "ema_rms" not in source
        assert "kappa_l2_ema" not in source
        assert "--kappa-l2-ema" not in source
        assert "loss = loss + kappa_bias_l2_loss_weight * kappa_bias_l2_loss" in source
        assert "loss = loss + kappa_scale_l2_loss_weight * kappa_scale_l2_loss" in source


def test_kappa_input_logit_norm_exponent_cli_is_wired_into_config():
    source = BASE_TRAIN.read_text()

    assert '--normalize-top-logits' not in source
    assert 'parser.add_argument("--kappa-input-logit-norm-exponent", dest="kappa_input_logit_norm_exponent", type=float, default=0.5,' in source
    assert 'kappa_input_logit_norm_exponent=args.kappa_input_logit_norm_exponent' in source


def test_loss_recompute_backward_cli_is_wired_into_config():
    source = BASE_TRAIN.read_text()

    assert 'parser.add_argument("--loss-recompute-backward", dest="loss_recompute_backward", type=str2bool, nargs=' in source
    assert 'loss_recompute_backward=args.loss_recompute_backward' in source


def test_kappa_slope_max_scale_anneal_cli_is_wired_into_step_updates():
    source = BASE_TRAIN.read_text()

    assert '"--kappa-slope-max-scale-warmup-iteration-frac"' in source
    assert '"--kappa-slope-max-scale-annealing-iteration-frac"' not in source
    assert 'dest="kappa_slope_max_scale_warmup_iteration_frac", type=float, default=0.1' in source
    assert 'def get_kappa_slope_max_scale(target_max_scale, it, total_iterations, warmup_iteration_frac=0.1, delay_iterations=0):' in source
    assert 'moe_kappa_slope_max_scale = get_kappa_slope_max_scale(' in source
    assert 'dense_kappa_slope_max_scale = get_kappa_slope_max_scale(' in source
    assert 'warmup_iteration_frac=args.kappa_slope_max_scale_warmup_iteration_frac' in source
    assert 'delay_iterations=kappa_param_delay_start_iterations' in source
    assert 'orig_model.set_kappa_slope_max_scales(' in source
    assert 'log_data["train/moe_kappa_slope_max_scale"] = moe_kappa_slope_max_scale' in source
    assert 'log_data["train/dense_kappa_slope_max_scale"] = dense_kappa_slope_max_scale' in source


def test_nonfinite_grad_debug_guard_is_wired_before_optimizer_step():
    source = BASE_TRAIN.read_text()

    assert 'def find_first_nonfinite_grad(model):' in source
    assert 'def summarize_loss_snapshot(loss, micro_losses):' in source
    assert 'abort_on_nonfinite_grad = args.debug or env_flag_is_true("NANOCHAT_ABORT_ON_NONFINITE_GRAD")' in source
    assert 'grad_issue = find_first_nonfinite_grad(orig_model)' in source
    assert 'Non-finite gradient detected before optimizer.step' in source
    assert 'loss_snapshot = summarize_loss_snapshot(loss, micro_losses)' in source