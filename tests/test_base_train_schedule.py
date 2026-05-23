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


def test_get_annealed_loss_weight_drops_to_floor_in_first_500_steps_then_stays_there():
    get_annealed_loss_weight = load_function_from_script("get_annealed_loss_weight")

    assert get_annealed_loss_weight(0.002, 0, final_weight=0.001) == 0.002
    assert abs(get_annealed_loss_weight(0.002, 250, final_weight=0.001) - 0.0015) < 1e-12
    assert get_annealed_loss_weight(0.002, 500, final_weight=0.001) == 0.001
    assert get_annealed_loss_weight(0.002, 900, final_weight=0.001) == 0.001


def test_gate_proj_bias_l2_two_stage_schedule_uses_half_run_then_decays_to_final_floor():
    get_two_stage_annealed_loss_weight = load_function_from_script("get_two_stage_annealed_loss_weight")

    assert get_two_stage_annealed_loss_weight(1.0, 0, total_iterations=10) == 1.0
    assert get_two_stage_annealed_loss_weight(1.0, 5, total_iterations=10) == 0.1
    assert abs(get_two_stage_annealed_loss_weight(1.0, 7, total_iterations=10) - 0.064) < 1e-12
    assert get_two_stage_annealed_loss_weight(1.0, 10, total_iterations=10) == 0.01


def test_gate_proj_bias_l2_two_stage_schedule_can_increase_during_stage_2():
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


def test_build_chat_sft_exec_argv_pins_final_checkpoint_and_splits_extra_args():
    build_chat_sft_exec_argv = load_function_from_script("build_chat_sft_exec_argv")

    argv = build_chat_sft_exec_argv(
        "/usr/bin/python3",
        "d8",
        120,
        "--device-batch-size 8 --model-save-tag after-base",
    )

    assert argv == [
        "/usr/bin/python3",
        "-m",
        "scripts.chat_sft",
        "--model-tag",
        "d8",
        "--model-step",
        "120",
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


def test_gate_proj_bias_l2_default_schedule_uses_half_run_and_two_stage_floors():
    source = BASE_TRAIN.read_text()

    assert 'parser.add_argument("--aux-loss-weight", type=float, default=1e-3' in source
    assert 'parser.add_argument("--aux-loss-weight-init-scale", type=float, default=2.0' in source
    assert 'parser.add_argument("--aux-loss-weight-init-anneal-iterations", type=int, default=500' in source
    assert 'orig_model.config.aux_loss_weight = aux_loss_weight' in source
    assert 'log_data["train/aux_loss_weight"] = aux_loss_weight' in source
    assert 'args.aux_loss_weight * args.aux_loss_weight_init_scale' in source
    assert 'num_anneal_iterations=args.aux_loss_weight_init_anneal_iterations' in source
    assert 'final_weight=args.aux_loss_weight' in source
    assert '--use-gate-proj-bias-as-lr-scaler' not in source
    assert 'parser.add_argument("--gate-proj-bias-l2-target", type=str, default="zero", choices=["zero", "ema"]' in source
    assert 'gate_proj_bias_l2_target=args.gate_proj_bias_l2_target' in source
    assert 'orig_model.set_gate_proj_bias_l2_target_step(step)' in source
    assert 'parser.add_argument("--gate-proj-bias-l2-loss-stage1-frac", type=float, default=0.1' in source
    assert '--gate-proj-bias-l2-loss-final-frac", type=float, default=0.02' in source
    assert 'stage1_iterations = max((effective_total_iterations + 1) // 2, 1)' in source
    assert 'parser.add_argument("--continue-to-chat-sft", action="store_true"' in source
    assert 'parser.add_argument("--continue-to-chat-sft-args", type=str, default=""' in source
    assert 'should_continue_to_chat_sft = args.continue_to_chat_sft and step == num_iterations' in source
    assert 'chat_sft_master_port = prepare_chat_sft_rendezvous(ddp, ddp_rank, device)' in source
    assert 'os.environ["MASTER_PORT"] = str(chat_sft_master_port)' in source
    assert 'torch.distributed.broadcast(port_tensor, src=0)' in source
    assert 'os.execvp(chat_sft_argv[0], chat_sft_argv)' in source


def test_gate_proj_bias_l2_ema_target_cli_is_wired_into_config_and_step_updates():
    source = BASE_TRAIN.read_text()

    assert 'parser.add_argument("--gate-proj-bias-l2-target", type=str, default="zero", choices=["zero", "ema"]' in source
    assert 'parser.add_argument("--gate-proj-bias-l2-ema-beta", type=float, default=0.99' in source
    assert 'parser.add_argument("--gate-proj-bias-l2-ema-anchor-start", type=float, default=0.4' in source
    assert 'parser.add_argument("--gate-proj-bias-l2-ema-anchor-end", type=float, default=0.8' in source
    assert 'parser.add_argument("--gate-proj-bias-l2-ema-floor-frac", type=float, default=0.8' in source
    assert 'gate_proj_bias_l2_target=args.gate_proj_bias_l2_target' in source
    assert 'gate_proj_bias_l2_ema_beta=args.gate_proj_bias_l2_ema_beta' in source
    assert 'gate_proj_bias_l2_ema_anchor_start=args.gate_proj_bias_l2_ema_anchor_start' in source
    assert 'gate_proj_bias_l2_ema_anchor_end=args.gate_proj_bias_l2_ema_anchor_end' in source
    assert 'gate_proj_bias_l2_ema_floor_frac=args.gate_proj_bias_l2_ema_floor_frac' in source
    assert 'orig_model.set_gate_proj_bias_l2_target_total_iterations(num_iterations)' in source
    assert 'orig_model.set_gate_proj_bias_l2_target_step(step)' in source


def test_nonfinite_grad_debug_guard_is_wired_before_optimizer_step():
    source = BASE_TRAIN.read_text()

    assert 'def find_first_nonfinite_grad(model):' in source
    assert 'def summarize_loss_snapshot(loss, micro_losses):' in source
    assert 'abort_on_nonfinite_grad = args.debug or env_flag_is_true("NANOCHAT_ABORT_ON_NONFINITE_GRAD")' in source
    assert 'grad_issue = find_first_nonfinite_grad(orig_model)' in source
    assert 'Non-finite gradient detected before optimizer.step' in source
    assert 'loss_snapshot = summarize_loss_snapshot(loss, micro_losses)' in source