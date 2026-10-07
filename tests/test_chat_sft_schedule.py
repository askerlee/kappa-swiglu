import argparse
import ast
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanochat.common import cast_model_parameters


ROOT = Path(__file__).resolve().parents[1]
CHAT_SFT = ROOT / "scripts" / "chat_sft.py"
CHECKPOINT_MANAGER = ROOT / "nanochat" / "checkpoint_manager.py"


@pytest.mark.parametrize("script_name", ["chat_sft", "chat_eval"])
@pytest.mark.parametrize("run_as_main", [True, False])
@pytest.mark.parametrize("offline,existing", [(True, None), (True, "0"), (False, None), (False, "1")])
def test_chat_hf_offline_before_task_imports(monkeypatch, script_name, run_as_main, offline, existing):
    script_path = ROOT / "scripts" / f"{script_name}.py"
    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        if existing is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, existing)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "test")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "test")
    monkeypatch.setattr(sys, "argv", [str(script_path), "--device-type", "cpu"] + (["--hf-offline"] if offline else []))
    module = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    prefix = []
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.module == "tasks.arc":
            break
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
            break
        prefix.append(node)
    namespace = {"__name__": "__main__" if run_as_main else f"scripts.{script_name}"}
    exec(compile(ast.Module(body=prefix, type_ignores=[]), filename=str(script_path), mode="exec"), namespace)
    enabled = offline and (run_as_main or script_name == "chat_sft")
    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        assert os.environ.get(variable) == ("1" if enabled else existing)

    parser_assignment = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets)
    )
    exec(compile(ast.Module(body=[parser_assignment], type_ignores=[]), filename=str(script_path), mode="exec"), namespace)
    parser = namespace["parser"]
    assert parser.parse_args(["--hf-offline"] if offline else []).hf_offline is offline
    assert "--hf-offline" in parser.format_help()


def load_function_from_script(function_name):
    source = CHAT_SFT.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(CHAT_SFT))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace = {"torch": torch}
            exec(compile(function_module, filename=str(CHAT_SFT), mode="exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name} not found in {CHAT_SFT}")


@pytest.mark.parametrize("coeff", [0.0, 0.25, 0.5, 1.0])
def test_blend_sft_kappa_params_preserves_base_and_shared_params(coeff):
    model = torch.nn.Module()
    model.config = SimpleNamespace(separate_base_sft_kappa=True)
    model.global_kappa_bias = torch.nn.Parameter(torch.tensor([[2.0], [6.0]]))
    model.global_kappa_scale = torch.nn.Parameter(torch.tensor([[1.0], [3.0]]))
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.Module()
    model.mlp.experts.kappa_bias = torch.nn.Parameter(torch.arange(12.0).reshape(2, 2, 3))
    model.mlp.experts.kappa_scale = torch.nn.Parameter(torch.arange(12.0).reshape(2, 2, 3) + 1)
    model.mlp.experts.kappa_bias_alpha = torch.nn.Parameter(torch.tensor(2.0))
    model.mlp.kappa_router = torch.nn.Linear(3, 4)
    model.mlp.router = torch.nn.Linear(3, 4)
    original = {name: param.detach().clone() for name, param in model.named_parameters()}

    count = load_function_from_script("blend_sft_kappa_params")(model, coeff)

    assert count == (0 if coeff == 0.0 else 6)
    for name, param in model.named_parameters():
        if name.endswith("kappa_bias_alpha") or ".router." in name:
            torch.testing.assert_close(param, original[name])
            continue
        slots = param.reshape(2, -1)
        old_slots = original[name].reshape(2, -1)
        torch.testing.assert_close(slots[0], old_slots[0])
        torch.testing.assert_close(slots[1], old_slots[1].lerp(old_slots[0], coeff))


@pytest.mark.parametrize("coeff", [-0.1, 1.1, float("nan"), float("inf")])
def test_blend_sft_kappa_params_rejects_invalid_coeff(coeff):
    with pytest.raises(ValueError, match="0 <= coefficient <= 1"):
        load_function_from_script("blend_sft_kappa_params")(None, coeff)


def test_blend_sft_kappa_params_requires_separate_slots_only_when_enabled():
    model = SimpleNamespace(config=SimpleNamespace(separate_base_sft_kappa=False))
    blend = load_function_from_script("blend_sft_kappa_params")
    assert blend(model, 0.0) == 0
    with pytest.raises(ValueError, match="requires separate base/SFT"):
        blend(model, 0.5)


def test_kappa_blend_cli_and_initial_anchor_order():
    source = CHAT_SFT.read_text()
    module = ast.parse(source)
    option = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--kappa-blend-coeff"
    )
    parser = argparse.ArgumentParser()
    option_module = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=option)], type_ignores=[]))
    exec(compile(option_module, str(CHAT_SFT), "exec"), {"parser": parser})
    assert parser.parse_args([]).kappa_blend_coeff == 0.0
    assert parser.parse_args(["--kappa-blend-coeff", "0.5"]).kappa_blend_coeff == 0.5
    load_position = source.index("model, tokenizer, meta = load_model(")
    blend_position = source.index("blended_kappa_params = blend_sft_kappa_params(")
    anchor_position = source.index("model.refresh_kappa_param_references()", blend_position)
    assert load_position < blend_position < anchor_position < source.index("optimizer = model.setup_optimizer(")


@pytest.mark.parametrize("rank,upload", [(0, True), (0, False), (1, True)])
def test_chat_eval_csv_and_wandb_artifact(tmp_path, monkeypatch, rank, upload):
    script_path = ROOT / "scripts" / "chat_eval.py"
    module = ast.parse(script_path.read_text(encoding="utf-8"))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "save_chat_eval_results")
    logged = {}

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            logged["finished"] = True

        def log(self, data, step):
            logged["metrics"] = data
            logged["step"] = step

        def log_artifact(self, artifact):
            logged["artifact"] = artifact

    class FakeArtifact:
        def __init__(self, **kwargs):
            self.options = kwargs

        def add_file(self, path, name):
            self.file = (path, name)

    def fake_init(**kwargs):
        logged["init"] = kwargs
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init, Artifact=FakeArtifact))
    namespace = {"os": os, "csv": csv, "get_base_dir": lambda: str(tmp_path),
                 "get_dist_info": lambda: (rank != 0, rank, rank, 2), "print0": lambda *args: None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script_path), "exec"), namespace)
    args = SimpleNamespace(model_tag="d8", source="sft", step=42, run=None if upload else "dummy",
                           wandb_project="nano-moe-sft", wandb_api_key_file=None,
                           max_problems=10, batch_size=8, num_samples=1, max_new_tokens=512,
                           temperature=0.0, top_k=50, total_ut_steps=2)
    metrics = {"ChatCORE metric": 0.25, "ChatCORE metric (without SpellingBee)": 0.3}
    path = namespace["save_chat_eval_results"](args, {"ARC-Easy": 0.5}, metrics)
    if rank != 0:
        assert path is None
        assert not list(tmp_path.iterdir())
        assert not logged
        return
    with open(path, newline="", encoding="utf-8") as result_file:
        rows = list(csv.reader(result_file))
    assert rows == [["Task", "Accuracy"], ["ARC-Easy", "0.500000"],
                    ["ChatCORE metric", "0.250000"],
                    ["ChatCORE metric (without SpellingBee)", "0.300000"]]
    if upload:
        assert logged["init"]["project"] == "nano-moe-sft"
        assert logged["artifact"].options["type"] == "chat-eval-results"
        assert logged["artifact"].options["metadata"]["task_names"] == ["ARC-Easy"]
        assert logged["artifact"].file == (path, os.path.basename(path))
        assert logged["metrics"]["chat_eval/ChatCORE_without_SpellingBee"] == 0.3
        assert logged["step"] == 42
        assert logged["finished"]
    else:
        assert not logged


@pytest.mark.parametrize('script_name', ['base_train', 'base_train_mix', 'chat_sft'])
def test_disable_kappa_bias_cli_wires_model_config(script_name):
    source = (ROOT / 'scripts' / f'{script_name}.py').read_text(encoding='utf-8')
    assert 'disable_kappa_bias=args.disable_kappa_bias,' in source
    module = ast.parse(source)
    option = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'add_argument' and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == '--disable-kappa-bias'
    )
    default = next(keyword.value.value for keyword in option.keywords if keyword.arg == 'default')
    assert default is (None if script_name == 'chat_sft' else False)


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


def test_chat_sft_does_not_monitor_kappa_gradient_correlation():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "kappa_grad_correlation" not in source
    assert "def gradient_correlation(" not in source


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

    assert 'def get_kappa_lr_scale(optimizer, step, num_iterations, group_name="kappa_params"):' in source
    assert 'if group_name == "kappa_params" and group.get("kind") == "adamw":' in source
    assert 'end_scale=group.get("lr_scale_end", 1.0)' in source
    assert 'max_scale=group.get("lr_scale_max", 1.0)' in source


def test_kappa_bias_lr_schedule_wires_delay_and_warmup_cli_args():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert 'nolearn_iterations=group.get("kappa_param_delay_start_iterations", 0)' in source
    assert 'warmup_iterations=group.get("lr_scale_warmup_iterations", 1000)' in source


@pytest.mark.parametrize("warmup_iterations", [None, 0, 37, -1])
@pytest.mark.parametrize("option,field,optimizer_keyword", [
    ("--kappa-lr-warmup-iterations", "kappa_lr_warmup_iterations", "kappa_lr_warmup_iterations"),
    ("--kappa-delay-start-min-iterations", "kappa_delay_start_min_iterations", "kappa_param_delay_start_iterations"),
    ("--kappa-delay-start-iterations", "kappa_delay_start_min_iterations", "kappa_param_delay_start_iterations"),
])
def test_chat_sft_kappa_warmup_parser_validation_and_optimizer(monkeypatch, warmup_iterations, option, field, optimizer_keyword):
    module = ast.parse(CHAT_SFT.read_text(encoding="utf-8"), filename=str(CHAT_SFT))
    parser_nodes = []
    collecting = False
    for node in module.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "parser" in names:
                collecting = True
            if "user_config" in names:
                break
        if collecting:
            parser_nodes.append(node)
    str2bool_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "str2bool"
    )
    namespace = {"argparse": argparse, "offline_parser": argparse.ArgumentParser(add_help=False),
                 "print0": lambda *args: None}
    argv = [str(CHAT_SFT)]
    if warmup_iterations is not None:
        argv += [option, str(warmup_iterations)]
    monkeypatch.setattr(sys, "argv", argv)
    code = compile(ast.Module(body=[str2bool_node, *parser_nodes], type_ignores=[]), str(CHAT_SFT), "exec")
    if warmup_iterations == -1:
        with pytest.raises(ValueError, match=f"--{field.replace('_', '-')} must be >= 0"):
            exec(code, namespace)
        return
    exec(code, namespace)
    expected = 100 if warmup_iterations is None else warmup_iterations
    assert getattr(namespace["args"], field) == expected
    optimizer_call = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setup_optimizer"
    )
    warmup_value = next(
        keyword.value for keyword in optimizer_call.keywords if keyword.arg == optimizer_keyword
    )
    assert eval(compile(ast.Expression(warmup_value), str(CHAT_SFT), "eval"), namespace) == expected


def test_chat_sft_uses_schedule_total_iterations_when_applying_kappa_bias_lr_scale():
    source = CHAT_SFT.read_text(encoding="utf-8")

    assert "kappa_bias_schedule_total_iterations = get_kappa_bias_schedule_total_iterations(" in source
    assert 'kappa_bias_lr_scale = get_kappa_lr_scale(' in source
    assert 'kappa_router_lr_scale = get_kappa_lr_scale(' in source
    assert '        optimizer,' in source
    assert '        kappa_bias_schedule_total_iterations,' in source


@pytest.mark.parametrize("kind", ["muon", "muonh", "aurora", "adamw"])
@pytest.mark.parametrize("delay,warmup,step,expected_scale", [
    (100, 100, 99, 0.0),
    (100, 100, 100, 0.0),
    (100, 100, 150, 0.5),
    (100, 100, 200, 1.0),
    (100, 100, 500, 1.0),
    (0, 100, 0, 0.0),
    (0, 100, 50, 0.5),
    (100, 0, 100, 1.0),
])
def test_chat_sft_kappa_router_lr_warms_up(kind, delay, warmup, step, expected_scale):
    module = ast.parse(CHAT_SFT.read_text(encoding="utf-8"), filename=str(CHAT_SFT))
    router_branch = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "group.get('name') == 'kappa_router'"
    )
    group = {
        "name": "kappa_router", "kind": kind, "initial_lr": 0.01,
        "kappa_param_delay_start_iterations": delay,
    }
    get_kappa_lr_scale = load_function_from_script("get_kappa_lr_scale")
    get_kappa_lr_scale.__globals__.update({
        "args": SimpleNamespace(kappa_lr_warmup_iterations=warmup),
        "get_linear_lr_scale": load_function_from_script("get_linear_lr_scale"),
    })
    router_scale = get_kappa_lr_scale(
        SimpleNamespace(param_groups=[group]), step, 1000, group_name="kappa_router"
    )
    assert router_scale == pytest.approx(expected_scale)
    namespace = {
        "group": group,
        "lrm": 0.2,
        "kappa_router_lr_scale": router_scale,
    }
    exec(compile(ast.Module(body=router_branch.body, type_ignores=[]), str(CHAT_SFT), "exec"), namespace)
    assert group["lr"] == pytest.approx(0.01 * 0.2 * expected_scale)


@pytest.mark.parametrize("step,expected_scale", [(99, 0.0), (150, 0.005), (200, 0.01), (1000, 0.005)])
def test_chat_sft_shared_kappa_schedule_preserves_bias_scales(step, expected_scale):
    get_kappa_lr_scale = load_function_from_script("get_kappa_lr_scale")
    get_kappa_lr_scale.__globals__["get_linear_lr_scale"] = load_function_from_script("get_linear_lr_scale")
    group = {
        "name": "kappa_params", "kind": "adamw",
        "lr_scale_max": 0.01, "lr_scale_end": 0.005,
        "kappa_param_delay_start_iterations": 100,
        "lr_scale_warmup_iterations": 100,
    }
    optimizer = SimpleNamespace(param_groups=[group])
    assert get_kappa_lr_scale(optimizer, step, 1000) == pytest.approx(expected_scale)
    assert get_kappa_lr_scale(optimizer, step, 1000, group_name="kappa_router") == 1.0


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