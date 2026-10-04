import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanochat.common import cast_model_parameters


ROOT = Path(__file__).resolve().parents[1]
CHAT_SFT = ROOT / "scripts" / "chat_sft.py"
CHECKPOINT_MANAGER = ROOT / "nanochat" / "checkpoint_manager.py"


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


@pytest.mark.parametrize('script_name', ['base_train', 'base_train_mix', 'chat_sft'])
def test_independent_kappa_router_cli_wires_model_config(script_name):
    source = (ROOT / 'scripts' / f'{script_name}.py').read_text(encoding='utf-8')
    assert '"--independent-kappa-router"' in source
    assert 'independent_kappa_router=args.independent_kappa_router,' in source
    module = ast.parse(source)
    option = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'add_argument' and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == '--independent-kappa-router'
    )
    default = next(keyword.value.value for keyword in option.keywords if keyword.arg == 'default')
    assert default is (None if script_name == 'chat_sft' else False)


@pytest.mark.parametrize("independent_router", [False, True])
@pytest.mark.parametrize("base_weight", [0.0, 0.001, 0.02])
def test_chat_sft_independent_kappa_bias_l2_weight(independent_router, base_weight):
    module = ast.parse(CHAT_SFT.read_text(), filename=str(CHAT_SFT))
    weights = ast.Module(body=[
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {"kappa_bias_l2_loss_weight", "kappa_scale_l2_loss_weight"}
            for target in node.targets
        )
    ], type_ignores=[])
    namespace = {
        "args": SimpleNamespace(
            independent_kappa_router=None, kappa_l2_loss_weight=base_weight,
            kappa_scale_l2_loss_weight_scale=0.2,
        ),
        "model": SimpleNamespace(config=SimpleNamespace(independent_kappa_router=independent_router)),
    }
    exec(compile(weights, filename=str(CHAT_SFT), mode="exec"), namespace)
    assert namespace["kappa_bias_l2_loss_weight"] == base_weight * (10 if independent_router else 1)
    assert namespace["kappa_scale_l2_loss_weight"] == base_weight * 0.2
    source = CHAT_SFT.read_text()
    assert 'loss = loss + kappa_bias_l2_loss_weight * kappa_bias_l2_loss' in source
    assert '"train/kappa_bias_l2_loss_weight": kappa_bias_l2_loss_weight' in source


def test_chat_sft_keeps_sensitive_parameters_in_fp32_without_casting_buffers():
    module = torch.nn.Linear(4, 3)
    module.register_buffer("stats", torch.ones(2, dtype=torch.float32))

    cast_model_parameters(module, torch.bfloat16)

    assert module.weight.dtype == torch.bfloat16
    assert module.bias.dtype == torch.float32
    assert module.stats.dtype == torch.float32


def test_chat_sft_keeps_scalar_vector_and_router_parameters_in_fp32():
    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(4, 3, bias=False)

    module = torch.nn.Module()
    module.projection = torch.nn.Linear(4, 3, bias=False)
    module.norm = torch.nn.LayerNorm(4)
    module.kappa = torch.nn.Parameter(torch.tensor(1.0))
    module.kappa_per_gate = torch.nn.Parameter(torch.ones(2, 3, 4))
    module.gate_proj_bias = torch.nn.Parameter(torch.ones(3, 4))
    module.router = Router()

    cast_model_parameters(module, torch.bfloat16)

    assert module.projection.weight.dtype == torch.bfloat16
    assert module.norm.weight.dtype == torch.float32
    assert module.norm.bias.dtype == torch.float32
    assert module.kappa.dtype == torch.float32
    assert module.kappa_per_gate.dtype == torch.float32
    assert module.gate_proj_bias.dtype == torch.float32
    assert module.router.gate.weight.dtype == torch.float32


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


def test_chat_sft_enables_kappa_swiglu_for_all_iterations():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--use-kappa-swiglu", type=str2bool, nargs=\'?\', const=True, default=None' in source
    assert 'parser.add_argument("--constant-kappa-dense-layers", dest="constant_kappa_dense_layers", type=str2bool, nargs=\'?\', const=True, default=None' in source
    assert "use_kappa_swiglu = args.use_kappa_swiglu" in source
    assert "use_kappa_swiglu=use_kappa_swiglu" in source
    assert "constant_kappa_bias_dense_layers=args.constant_kappa_dense_layers" in source
    enable_index = source.index("model.set_kappa_swiglu_enabled(True)")
    compile_index = source.index("model = torch.compile(model, dynamic=False)")
    assert enable_index < compile_index


def test_chat_sft_kappa_override_inherits_when_omitted():
    source = CHAT_SFT.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(CHAT_SFT))
    assignment = next(
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "use_kappa_swiglu" for target in node.targets)
    )
    value = compile(ast.Expression(assignment.value), filename=str(CHAT_SFT), mode="eval")

    for cli_value in (None, True, False):
        args = type("Args", (), {"use_kappa_swiglu": cli_value})()
        override = eval(value, {"args": args})
        for checkpoint_value in (True, False):
            effective_value = checkpoint_value if override is None else override
            assert effective_value == (checkpoint_value if cli_value is None else cli_value)


def test_chat_sft_marks_signature_when_kappa_overrides_checkpoint():
    source = CHAT_SFT.read_text(encoding="utf-8")
    checkpoint_manager_source = CHECKPOINT_MANAGER.read_text(encoding="utf-8")

    load_index = source.index("model, tokenizer, meta = load_model(")
    checkpoint_config_index = source.index(
        'meta.get("model_config", {}).get("use_kappa_swiglu", False)',
        load_index,
    )
    signature_index = source.index('ckpt_prefix2 += "-kappa"', checkpoint_config_index)
    wandb_name_index = source.index("wandb_run_name = ckpt_prefix2", signature_index)

    assert "if args.use_kappa_swiglu is True and not checkpoint_used_kappa_swiglu:" in source
    assert load_index < checkpoint_config_index < signature_index < wandb_name_index
    assert 'model_config_kwargs = meta_data["model_config"].copy()' in checkpoint_manager_source


def test_chat_sft_uses_10x_kappa_lr_scales_when_enabling_kappa_for_checkpoint():
    source = CHAT_SFT.read_text(encoding="utf-8")

    checkpoint_config_index = source.index(
        'meta.get("model_config", {}).get("use_kappa_swiglu", False)'
    )
    scale_condition_index = source.index(
        "if use_kappa_swiglu and not checkpoint_used_kappa_swiglu:",
        checkpoint_config_index,
    )
    max_scale_index = source.index("args.kappa_lr_max_scale *= 10", scale_condition_index)
    final_scale_index = source.index("args.kappa_lr_final_scale *= 10", scale_condition_index)
    optimizer_index = source.index("optimizer = model.setup_optimizer(", final_scale_index)

    assert checkpoint_config_index < scale_condition_index < max_scale_index < optimizer_index
    assert checkpoint_config_index < scale_condition_index < final_scale_index < optimizer_index
    assert 'user_config["kappa_lr_max_scale"] = args.kappa_lr_max_scale' in source
    assert 'user_config["kappa_lr_final_scale"] = args.kappa_lr_final_scale' in source


def test_chat_sft_defaults_kappa_l2_when_enabling_kappa_for_checkpoint():
    source = CHAT_SFT.read_text(encoding="utf-8")

    explicit_check_index = source.index(
        "kappa_l2_loss_weight_was_specified = "
        "arg_was_explicitly_set(sys.argv[1:], '--kappa-l2-loss-weight')"
    )
    enable_condition_index = source.index(
        "if use_kappa_swiglu and not checkpoint_used_kappa_swiglu:"
    )
    omitted_condition_index = source.index(
        "if not kappa_l2_loss_weight_was_specified:",
        enable_condition_index,
    )
    default_index = source.index(
        "args.kappa_l2_loss_weight = 0.01",
        omitted_condition_index,
    )
    wandb_index = source.index("wandb_run = DummyWandb()", default_index)

    assert explicit_check_index < enable_condition_index < omitted_condition_index
    assert omitted_condition_index < default_index < wandb_index
    assert 'user_config["kappa_l2_loss_weight"] = args.kappa_l2_loss_weight' in source[
        default_index:wandb_index
    ]


def test_chat_sft_inherits_checkpoint_train_capacity():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--train-capacity"' not in source
    assert "args.train_capacity" not in source
    assert "model.set_train_capacity" not in source


def test_chat_sft_scalar_lr_defaults_to_005_and_is_wired_to_optimizer():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--scalar-lr", type=float, default=0.05' in source
    assert "scalar_lr=args.scalar_lr" in source


def test_chat_sft_global_lr_schedule_defaults_and_wiring():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--warmup-ratio", type=float, default=0.01' in source
    assert 'parser.add_argument("--warmdown-ratio", type=float, default=0.2' in source
    assert 'parser.add_argument("--final-lr-frac", type=float, default=0.05' in source
    assert "        args.warmup_ratio," in source
    assert "        args.warmdown_ratio," in source
    assert "        args.final_lr_frac," in source


def test_chat_sft_global_lr_schedule_warms_holds_and_decays_to_floor():
    get_lr_multiplier = load_function_from_script("get_lr_multiplier")

    assert get_lr_multiplier(0.0, 0.2, 0.01, 0.2, 0.05) == pytest.approx(0.0)
    assert get_lr_multiplier(0.005, 0.2, 0.01, 0.2, 0.05) == pytest.approx(0.1)
    assert get_lr_multiplier(0.01, 0.2, 0.01, 0.2, 0.05) == pytest.approx(0.2)
    assert get_lr_multiplier(0.8, 0.2, 0.01, 0.2, 0.05) == pytest.approx(0.2)
    assert get_lr_multiplier(0.9, 0.2, 0.01, 0.2, 0.05) == pytest.approx(0.105)
    assert get_lr_multiplier(1.0, 0.2, 0.01, 0.2, 0.05) == pytest.approx(0.01)


def test_gradient_correlation_pairs_matching_kappa_gate_gradients():
    gradient_correlation = load_function_from_script("gradient_correlation")

    scale_grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bias_grad = torch.tensor([[2.0, 4.0], [6.0, 8.0]])

    assert gradient_correlation(scale_grad, bias_grad) == 1.0
    assert gradient_correlation(scale_grad, -bias_grad) == -1.0
    assert gradient_correlation(scale_grad, torch.ones_like(scale_grad)) is None


def test_chat_sft_logs_kappa_gradient_correlation_per_layer():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "losses[f'kappa_grad_correlation_{i}'] = kappa_grad_correlation" in source
    assert 'log_data[f"inspect/kappa_grad_correlation_{i}"]' in source


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


def test_chat_sft_divides_global_packing_buffer_across_ranks():
    get_rank_packing_buffer_size = load_function_from_script("get_rank_packing_buffer_size")

    assert get_rank_packing_buffer_size(200, 1, 0) == 200
    assert [get_rank_packing_buffer_size(200, 2, rank) for rank in range(2)] == [100, 100]
    assert sum(get_rank_packing_buffer_size(200, 3, rank) for rank in range(3)) == 200
    with pytest.raises(ValueError, match="at least the DDP world size"):
        get_rank_packing_buffer_size(1, 2, 0)


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

    assert 'nolearn_iterations=group.get("kappa_param_delay_start_iterations", 0)' in source
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


def test_kappa_params_l2_anchor_cli_defaults_to_zero_and_wires_load_behavior():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--kappa-params-l2-anchor", type=str, choices=("initial", "zero"), default="zero"' in source
    assert '--use-kappa-swiglu-as-lr-scaler' not in source
    assert 'refresh_kappa_param_references = args.kappa_params_l2_anchor == "initial"' in source
    assert 'refresh_kappa_param_references=refresh_kappa_param_references' in source


def test_matrix_optimizer_inherits_from_base_checkpoint_unless_explicitly_set():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "matrix_optimizer_was_specified = arg_was_explicitly_set(sys.argv[1:], '--matrix-optimizer')" in source
    assert 'args.matrix_optimizer = meta.get("user_config", {}).get("matrix_optimizer", "muon")' in source
    assert 'print0(f"Inherited matrix_optimizer: {args.matrix_optimizer}")' in source
    assert 'print0(f"Specified matrix_optimizer: {args.matrix_optimizer}")' in source