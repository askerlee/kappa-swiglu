import ast
from pathlib import Path

import torch

from nanochat.common import cast_model_parameters


ROOT = Path(__file__).resolve().parents[1]
CHAT_SFT = ROOT / "scripts" / "chat_sft.py"


def load_function_from_script(function_name):
    source = CHAT_SFT.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(CHAT_SFT))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(function_module, filename=str(CHAT_SFT), mode="exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name} not found in {CHAT_SFT}")


def test_chat_sft_casts_floating_parameters_without_casting_buffers():
    module = torch.nn.Linear(4, 3)
    module.register_buffer("stats", torch.ones(2, dtype=torch.float32))

    cast_model_parameters(module, torch.bfloat16)

    assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())
    assert module.stats.dtype == torch.float32


def test_reference_parameter_storage_keeps_only_embeddings_in_bfloat16():
    module = torch.nn.Module()
    module.transformer = torch.nn.Module()
    module.transformer.wte = torch.nn.Embedding(8, 4)
    module.value_embeds = torch.nn.ModuleDict({"0": torch.nn.Embedding(8, 2)})
    module.projection = torch.nn.Linear(4, 3)

    cast_model_parameters(module, torch.float32, embedding_dtype=torch.bfloat16)

    assert module.transformer.wte.weight.dtype == torch.bfloat16
    assert module.value_embeds["0"].weight.dtype == torch.bfloat16
    assert module.projection.weight.dtype == torch.float32


def test_chat_sft_casts_parameters_before_compile_and_optimizer_setup():
    source = CHAT_SFT.read_text(encoding="utf-8")

    parameter_dtype_arg_index = source.index('parser.add_argument("--parameter-dtype"')
    cast_index = source.index("cast_model_parameters(model, parameter_dtype, embedding_dtype=embedding_dtype)")
    compile_index = source.index("model = torch.compile(model, dynamic=False)")
    optimizer_index = source.index("optimizer = model.setup_optimizer(")

    assert 'default="reference", choices=("reference", "float32", "bfloat16")' in source[parameter_dtype_arg_index:parameter_dtype_arg_index + 180]
    assert cast_index < compile_index < optimizer_index


def test_chat_sft_inherits_checkpoint_train_capacity():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--train-capacity"' not in source
    assert "args.train_capacity" not in source
    assert "model.set_train_capacity" not in source


def test_chat_sft_scalar_lr_defaults_to_005_and_is_wired_to_optimizer():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--scalar-lr", type=float, default=0.05' in source
    assert "scalar_lr=args.scalar_lr" in source


def test_router_wg_delta_cli_applies_during_training_and_eval():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "if args.router_wg_delta:" in source
    assert "if args.router_wg_delta and not args.eval_only:" not in source
    assert "model.setup_router_wg_delta()" in source
    assert '--router-wg-delta-l2-loss-weight' in source
    assert 'loss = loss + args.router_wg_delta_l2_loss_weight * router_wg_delta_l2_loss' in source
    assert 'group.get("name") == "router_wg_base"' not in source


def test_chat_sft_interval_throughput_averages_all_steps_since_previous_log():
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


def test_chat_sft_logged_throughput_uses_interval_values_and_resets_window():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert '"train/tok_per_sec": logged_tok_per_sec' in source
    assert '"train/mfu": logged_mfu' in source
    assert '"train/dt": logged_dt' in source
    assert "throughput_interval_steps += 1" in source
    assert "throughput_interval_steps = 0" in source
    assert "throughput_interval_time = 0.0" in source


def test_loss_recompute_backward_cli_is_wired_into_loaded_model_config():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--loss-recompute-backward", dest="loss_recompute_backward", type=str2bool, nargs=' in source
    assert "loss_recompute_backward=args.loss_recompute_backward" in source


def test_kappa_bias_lr_schedule_uses_total_iterations_helper_and_cli_scales():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "def get_kappa_bias_lr_scale(optimizer, step, num_iterations):" in source
    assert 'if group.get("name") == "kappa_params" and group.get("kind") == "adamw":' in source
    assert 'end_scale=group.get("lr_scale_end", 1.0)' in source
    assert 'max_scale=group.get("lr_scale_max", 1.0)' in source


def test_kappa_bias_lr_schedule_wires_delay_and_warmup_cli_args():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'nolearn_iterations=group.get("lr_scale_nolearn_iterations", 0)' in source
    assert 'warmup_iterations=group.get("lr_scale_warmup_iterations", 1000)' in source


def test_chat_sft_uses_schedule_total_iterations_when_applying_kappa_bias_lr_scale():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "kappa_bias_schedule_total_iterations = get_kappa_bias_schedule_total_iterations(" in source
    assert 'kappa_bias_lr_scale = get_kappa_bias_lr_scale(' in source
    assert '        optimizer,' in source
    assert '        kappa_bias_schedule_total_iterations,' in source


def test_chat_sft_inherits_kappa_slope_max_scale_without_sft_warmup():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "def get_kappa_slope_max_scale" not in source
    assert 'moe_kappa_slope_max_scale = getattr(orig_model.config, "moe_kappa_slope_max_scale", 3.0)' in source
    assert 'dense_kappa_slope_max_scale = getattr(orig_model.config, "dense_kappa_slope_max_scale", 2.0)' in source
    assert 'orig_model.set_kappa_slope_max_scales(' in source


def test_chat_eval_task_names_default_to_all_tasks():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'chat_eval_task_names = ALL_CHAT_EVAL_TASKS if args.chat_eval_task_name is None else args.chat_eval_task_name.split(\'|\')' in source


def test_chat_eval_runs_only_on_last_step():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "if last_step:\n        model.eval()\n        orig_model.eval()\n        engine = Engine(orig_model, tokenizer)" in source
    assert "chat_eval_every" not in source


def test_final_checkpoint_is_saved_before_final_chat_eval():
    source = CHAT_SFT.read_text(encoding="utf-8")

    save_index = source.index("    # save checkpoint at the end of the run before the expensive final chat eval")
    chat_eval_index = source.index("    if last_step:\n        model.eval()\n        orig_model.eval()\n        engine = Engine(orig_model, tokenizer)")

    assert save_index < chat_eval_index


def test_kappa_bias_l2_anchor_cli_defaults_to_initial_and_wires_load_behavior():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--kappa-bias-l2-anchor", type=str, choices=("initial", "zero"), default="zero"' in source
    assert '--use-kappa-swiglu-as-lr-scaler' not in source
    assert 'refresh_kappa_bias_references = args.kappa_bias_l2_anchor == "initial"' in source
    assert 'refresh_kappa_bias_references=refresh_kappa_bias_references' in source


def test_matrix_optimizer_inherits_from_base_checkpoint_unless_explicitly_set():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "matrix_optimizer_was_specified = arg_was_explicitly_set(sys.argv[1:], '--matrix-optimizer')" in source
    assert 'args.matrix_optimizer = meta.get("user_config", {}).get("matrix_optimizer", "muon")' in source
    assert 'print0(f"Inherited matrix_optimizer: {args.matrix_optimizer}")' in source
    assert 'print0(f"Specified matrix_optimizer: {args.matrix_optimizer}")' in source