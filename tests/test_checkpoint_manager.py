import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanochat.checkpoint_manager import _infer_kappa_bias, _infer_use_qwen3_dense_mlp, _migrate_optimizer_param_group_names, _override_kappa_bias_values, _override_kappa_scale_values, _patch_missing_config_keys, _patch_missing_keys, delete_old_checkpoints, inspect_optimizer_shards, load_optimizer_state_dict, reshard_optimizer_state_dict, save_checkpoint, snapshot_checkpoint_file_sizes, validate_checkpoint_file_sizes
from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat import checkpoint_manager


def make_optimizer(param_groups):
    return SimpleNamespace(param_groups=param_groups)


@pytest.mark.parametrize('total_ut_steps', [1, 3])
@pytest.mark.parametrize('granularity', ['per-gate', 'per-expert', 'per-layer', 'global'])
def test_task_kappa_checkpoint_keeps_two_slots_when_loop_count_changes(total_ut_steps, granularity):
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=32, n_exp=2,
        moe_start_layer=1, total_ut_steps=total_ut_steps,
        use_kappa_swiglu=True, constant_kappa_bias_dense_layers=True,
        separate_base_sft_kappa=True, global_kappa_bias_granularity=granularity,
    )
    model_data = {}
    _patch_missing_keys(model_data, config)
    kappa_keys = [key for key in model_data if 'kappa_' in key]
    assert kappa_keys
    for key in kappa_keys:
        assert model_data[key].shape[0] == 2
        model_data[key][1].fill_(0.7)
    _patch_missing_keys(model_data, config)
    for key in kappa_keys:
        assert model_data[key].shape[0] == 2
        torch.testing.assert_close(model_data[key][1], torch.full_like(model_data[key][1], 0.7))
    assert model_data['resid_lambdas'].shape[0] == total_ut_steps


@pytest.mark.parametrize('granularity', ['per-gate', 'per-expert', 'per-layer', 'global'])
@pytest.mark.parametrize('dense_kappa', [False, True])
def test_kappa_bias_from_scale_checkpoint_round_trip(granularity, dense_kappa):
    from nanochat.gpt import GPT

    config = GPTConfig(
        n_layer=3, n_head=2, n_embd=16, vocab_size=32, n_exp=2,
        moe_start_layer=1, total_ut_steps=3, use_kappa_swiglu=True,
        kappa_bias_from_scale=True, constant_kappa_bias_dense_layers=dense_kappa,
        separate_base_sft_kappa=True, global_kappa_bias_granularity=granularity,
    )
    model = GPT(config)
    model.init_weights()
    for layer_idx in (1, 2):
        experts = model.transformer.h[layer_idx].mlp.experts
        assert experts.kappa_bias_alpha.item() == 1.0
        with torch.no_grad():
            experts.kappa_bias_alpha.fill_(float(layer_idx))
            experts._get_kappa_scale_parameter().fill_(3.0)
    model_data = model.state_dict()
    _patch_missing_keys(model_data, config)
    restored = GPT(GPTConfig(**vars(config)))
    restored.init_weights()
    restored.load_state_dict(model_data)
    for layer_idx in (1, 2):
        experts = restored.transformer.h[layer_idx].mlp.experts
        torch.testing.assert_close(
            experts._materialize_kappa_bias(1),
            torch.full((2, 64), 3.0 * layer_idx),
        )
    missing_data = {}
    _patch_missing_keys(missing_data, config)
    assert "transformer.h.1.mlp.experts.kappa_bias" not in missing_data
    assert missing_data["transformer.h.1.mlp.experts.kappa_bias_alpha"].ndim == 0
    assert missing_data["transformer.h.1.mlp.experts.kappa_bias_alpha"].item() == 1.0
    if granularity == 'global' and not dense_kappa:
        assert 'global_kappa_bias' not in missing_data


def test_task_kappa_optimizer_state_stays_replicated_when_world_size_changes():
    param = torch.nn.Parameter(torch.zeros(2, 1024))
    optimizer = make_optimizer([
        {'kind': 'adamw', 'params': [param], 'active_kappa_slot': 0},
    ])
    groups = [{'kind': 'adamw', 'params': [0], 'active_kappa_slot': 0}]
    shard = make_adamw_shard(groups, 0, torch.ones_like(param), torch.ones_like(param))
    shard['state'][0]['slot_steps'] = [5, 2]
    result = reshard_optimizer_state_dict(
        [shard, copy.deepcopy(shard)], optimizer, rank=3,
        saved_world_size=2, current_world_size=4,
    )
    assert result['state'][0]['slot_steps'] == [5, 2]
    torch.testing.assert_close(result['state'][0]['exp_avg'], torch.ones_like(param))


@pytest.mark.parametrize('source,expected_slot', [('base', 0), ('sft', 1), ('rl', 1)])
def test_load_model_selects_kappa_task_from_checkpoint_source(monkeypatch, source, expected_slot):
    selected_phases = []
    model = SimpleNamespace(set_kappa_training_phase=selected_phases.append)
    monkeypatch.setattr(checkpoint_manager, 'get_base_dir', lambda: '/unused')
    monkeypatch.setattr(checkpoint_manager, 'load_model_from_dir', lambda *args, **kwargs: (model, None, {}))
    checkpoint_manager.load_model(source)
    assert selected_phases == [bool(expected_slot)]


def make_adamw_shard(param_groups, param_id, exp_avg, exp_avg_sq, step=7):
    return {
        "state": {
            param_id: {
                "step": step,
                "exp_avg": exp_avg.clone(),
                "exp_avg_sq": exp_avg_sq.clone(),
            }
        },
        "param_groups": copy.deepcopy(param_groups),
    }


def make_row_tensor(start_row, rows, cols):
    row_values = torch.arange(start_row, start_row + rows, dtype=torch.float32)
    return row_values.unsqueeze(1).expand(rows, cols).clone()


def test_migrate_optimizer_param_group_names_renames_legacy_kappa_group():
    state_dict = {"param_groups": [{"name": "kappa_bias"}, {"name": "embedding"}]}

    result = _migrate_optimizer_param_group_names(state_dict)

    assert [group["name"] for group in result["param_groups"]] == ["kappa_params", "embedding"]


def write_sized_file(path, size):
    path.write_bytes(b"x" * size)


def test_reshard_optimizer_state_dict_preserves_small_adamw_replica():
    param = torch.nn.Parameter(torch.zeros(8, 8))
    optimizer = make_optimizer([
        {"kind": "adamw", "params": [param], "lr": 1e-3}
    ])
    saved_param_groups = [{"kind": "adamw", "params": [0], "lr": 1e-3}]
    exp_avg = make_row_tensor(0, 8, 8)
    exp_avg_sq = make_row_tensor(100, 8, 8)
    shard_state_dicts = [
        make_adamw_shard(saved_param_groups, 0, exp_avg, exp_avg_sq),
        make_adamw_shard(saved_param_groups, 0, exp_avg, exp_avg_sq),
    ]

    state_dict = reshard_optimizer_state_dict(
        shard_state_dicts,
        optimizer,
        rank=3,
        saved_world_size=2,
        current_world_size=4,
    )

    loaded_state = state_dict["state"][0]
    assert torch.equal(loaded_state["exp_avg"], exp_avg)
    assert torch.equal(loaded_state["exp_avg_sq"], exp_avg_sq)
    assert loaded_state["step"] == 7


def test_reshard_optimizer_state_dict_expands_legacy_ut_scalar_moments():
    param = torch.nn.Parameter(torch.zeros(3, 2))
    optimizer = make_optimizer([
        {"kind": "adamw", "params": [param], "lr": 1e-3}
    ])
    saved_param_groups = [{"kind": "adamw", "params": [0], "lr": 1e-3}]
    exp_avg = torch.tensor([0.1, 0.2])
    exp_avg_sq = torch.tensor([0.3, 0.4])
    shard_state_dicts = [
        make_adamw_shard(saved_param_groups, 0, exp_avg, exp_avg_sq),
    ]

    state_dict = reshard_optimizer_state_dict(shard_state_dicts, optimizer)

    loaded_state = state_dict["state"][0]
    torch.testing.assert_close(loaded_state["exp_avg"], exp_avg.repeat(3, 1))
    torch.testing.assert_close(loaded_state["exp_avg_sq"], exp_avg_sq.repeat(3, 1))
    assert loaded_state["step"] == 7


def test_reshard_optimizer_state_dict_expands_legacy_kappa_moments():
    param = torch.nn.Parameter(torch.zeros(3, 2, 4))
    optimizer = make_optimizer([{
        "kind": "adamw",
        "params": [param],
        "debug_param_names": ["transformer.h.0.mlp.experts.kappa_bias"],
        "lr": 1e-3,
    }])
    saved_param_groups = [{"kind": "adamw", "params": [0], "lr": 1e-3}]
    exp_avg = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    exp_avg_sq = exp_avg + 10.0

    state_dict = reshard_optimizer_state_dict([
        make_adamw_shard(saved_param_groups, 0, exp_avg, exp_avg_sq),
    ], optimizer)

    loaded_state = state_dict["state"][0]
    torch.testing.assert_close(loaded_state["exp_avg"], exp_avg.repeat(3, 1, 1))
    torch.testing.assert_close(loaded_state["exp_avg_sq"], exp_avg_sq.repeat(3, 1, 1))


def test_reshard_optimizer_state_dict_expands_sharded_legacy_kappa_moments():
    param = torch.nn.Parameter(torch.zeros(2, 2, 300))
    optimizer = make_optimizer([{
        "kind": "adamw",
        "params": [param],
        "debug_param_names": ["transformer.h.0.mlp.experts.kappa_bias"],
        "lr": 1e-3,
    }])
    saved_param_groups = [{"kind": "adamw", "params": [0], "lr": 1e-3}]
    exp_avg = make_row_tensor(0, 2, 300)
    exp_avg_sq = exp_avg + 10.0
    shards = [
        make_adamw_shard(saved_param_groups, 0, exp_avg[:1], exp_avg_sq[:1]),
        make_adamw_shard(saved_param_groups, 0, exp_avg[1:], exp_avg_sq[1:]),
    ]

    state_dict = reshard_optimizer_state_dict(
        shards, optimizer, saved_world_size=2, current_world_size=1
    )

    loaded_state = state_dict["state"][0]
    torch.testing.assert_close(loaded_state["exp_avg"], exp_avg.repeat(2, 1, 1))
    torch.testing.assert_close(loaded_state["exp_avg_sq"], exp_avg_sq.repeat(2, 1, 1))


def test_reshard_optimizer_state_dict_converts_legacy_ut_source_lambda_moments():
    param = torch.nn.Parameter(torch.zeros(3))
    optimizer = make_optimizer([{
        "kind": "adamw",
        "params": [param],
        "debug_param_names": ["ut_source_lambdas"],
        "ut_destination": 2,
        "lr": 1e-3,
    }])
    saved_param_groups = [{"kind": "adamw", "params": [0], "lr": 1e-3}]
    exp_avg = torch.tensor([
        [0.0, 0.1, 0.2, 0.3],
        [1.0, 1.1, 1.2, 1.3],
        [2.0, 2.1, 2.2, 2.3],
    ])
    exp_avg_sq = exp_avg + 10.0
    shard_state_dicts = [
        make_adamw_shard(saved_param_groups, 0, exp_avg, exp_avg_sq),
    ]

    state_dict = reshard_optimizer_state_dict(shard_state_dicts, optimizer)

    loaded_state = state_dict["state"][0]
    torch.testing.assert_close(loaded_state["exp_avg"], exp_avg[:, 2])
    torch.testing.assert_close(loaded_state["exp_avg_sq"], exp_avg_sq[:, 2])
    assert loaded_state["step"] == 7


def test_reshard_optimizer_state_dict_reshards_muon_group():
    params = [torch.nn.Parameter(torch.zeros(2, 2)) for _ in range(5)]
    optimizer = make_optimizer([
        {"kind": "muon", "params": params, "lr": 1e-2, "momentum": 0.95}
    ])
    saved_param_groups = [{"kind": "muon", "params": [0, 1, 2, 3, 4], "lr": 1e-2, "momentum": 0.95}]

    full_momentum = torch.stack([torch.full((2, 2), float(idx)) for idx in range(5)])
    full_second = torch.stack([torch.full((2, 1), float(10 + idx)) for idx in range(5)])
    shard_state_dicts = [
        {
            "state": {
                0: {
                    "momentum_buffer": full_momentum[:3].clone(),
                    "second_momentum_buffer": full_second[:3].clone(),
                }
            },
            "param_groups": copy.deepcopy(saved_param_groups),
        },
        {
            "state": {
                0: {
                    "momentum_buffer": torch.cat([full_momentum[3:].clone(), torch.zeros(1, 2, 2)], dim=0),
                    "second_momentum_buffer": torch.cat([full_second[3:].clone(), torch.zeros(1, 2, 1)], dim=0),
                }
            },
            "param_groups": copy.deepcopy(saved_param_groups),
        },
    ]

    state_dict = reshard_optimizer_state_dict(
        shard_state_dicts,
        optimizer,
        rank=2,
        saved_world_size=2,
        current_world_size=4,
    )

    loaded_state = state_dict["state"][0]
    expected_momentum = torch.stack([full_momentum[4], torch.zeros(2, 2)], dim=0)
    expected_second = torch.stack([full_second[4], torch.zeros(2, 1)], dim=0)
    assert torch.equal(loaded_state["momentum_buffer"], expected_momentum)
    assert torch.equal(loaded_state["second_momentum_buffer"], expected_second)


def test_reshard_optimizer_state_dict_reshards_muonh_norms():
    params = [torch.nn.Parameter(torch.zeros(2, 2)) for _ in range(2)]
    optimizer = make_optimizer([
        {"kind": "muonh", "params": params, "lr": 1e-2, "momentum": 0.95}
    ])
    saved_groups = [{"kind": "muonh", "params": [0, 1], "lr": 1e-2, "momentum": 0.95}]
    p_norm = torch.tensor([[[2.0]], [[3.0]]])
    shard = {
        "state": {0: {"p_norm": p_norm.clone()}},
        "param_groups": copy.deepcopy(saved_groups),
    }

    state_dict = reshard_optimizer_state_dict([shard], optimizer)

    assert torch.equal(state_dict["state"][0]["p_norm"], p_norm)


def test_reshard_optimizer_state_dict_reshards_aurora_group():
    params = [torch.nn.Parameter(torch.zeros(2, 2)) for _ in range(5)]
    optimizer = make_optimizer([
        {"kind": "aurora", "params": params, "lr": 1e-2, "momentum": 0.95}
    ])
    saved_param_groups = [{"kind": "aurora", "params": [0, 1, 2, 3, 4], "lr": 1e-2, "momentum": 0.95}]

    full_momentum = torch.stack([torch.full((2, 2), float(idx)) for idx in range(5)])
    shard_state_dicts = [
        {
            "state": {
                0: {
                    "momentum_buffer": full_momentum[:3].clone(),
                }
            },
            "param_groups": copy.deepcopy(saved_param_groups),
        },
        {
            "state": {
                0: {
                    "momentum_buffer": torch.cat([full_momentum[3:].clone(), torch.zeros(1, 2, 2)], dim=0),
                }
            },
            "param_groups": copy.deepcopy(saved_param_groups),
        },
    ]

    state_dict = reshard_optimizer_state_dict(
        shard_state_dicts,
        optimizer,
        rank=2,
        saved_world_size=2,
        current_world_size=4,
    )

    loaded_state = state_dict["state"][0]
    expected_momentum = torch.stack([full_momentum[4], torch.zeros(2, 2)], dim=0)
    assert torch.equal(loaded_state["momentum_buffer"], expected_momentum)


def test_load_optimizer_state_dict_reshards_without_current_rank_file(tmp_path):
    step = 12
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    param = torch.nn.Parameter(torch.zeros(64, 32))
    optimizer = make_optimizer([
        {"kind": "adamw", "params": [param], "lr": 1e-3}
    ])
    saved_param_groups = [{"kind": "adamw", "params": [0], "lr": 1e-3}]

    shard0 = make_adamw_shard(
        saved_param_groups,
        0,
        make_row_tensor(0, 32, 32),
        make_row_tensor(1000, 32, 32),
    )
    shard1 = make_adamw_shard(
        saved_param_groups,
        0,
        make_row_tensor(32, 32, 32),
        make_row_tensor(1032, 32, 32),
    )
    torch.save(shard0, checkpoint_dir / f"optim_{step:06d}_rank0.pt")
    torch.save(shard1, checkpoint_dir / f"optim_{step:06d}_rank1.pt")

    state_dict = load_optimizer_state_dict(
        str(checkpoint_dir),
        step,
        optimizer,
        device="cpu",
        rank=3,
        current_world_size=4,
        saved_world_size=2,
    )

    loaded_state = state_dict["state"][0]
    assert loaded_state["exp_avg"].shape == (16, 32)
    assert loaded_state["exp_avg_sq"].shape == (16, 32)
    assert torch.equal(loaded_state["exp_avg"], make_row_tensor(48, 16, 32))
    assert torch.equal(loaded_state["exp_avg_sq"], make_row_tensor(1048, 16, 32))


def test_load_optimizer_state_dict_reshards_when_world_size_shrinks(tmp_path):
    step = 34
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    param = torch.nn.Parameter(torch.zeros(64, 32))
    optimizer = make_optimizer([
        {"kind": "adamw", "params": [param], "lr": 1e-3}
    ])
    saved_param_groups = [{"kind": "adamw", "params": [0], "lr": 1e-3}]

    for saved_rank in range(4):
        shard = make_adamw_shard(
            saved_param_groups,
            0,
            make_row_tensor(saved_rank * 16, 16, 32),
            make_row_tensor(2000 + saved_rank * 16, 16, 32),
        )
        torch.save(shard, checkpoint_dir / f"optim_{step:06d}_rank{saved_rank}.pt")

    state_dict = load_optimizer_state_dict(
        str(checkpoint_dir),
        step,
        optimizer,
        device="cpu",
        rank=1,
        current_world_size=2,
        saved_world_size=4,
    )

    loaded_state = state_dict["state"][0]
    assert loaded_state["exp_avg"].shape == (32, 32)
    assert loaded_state["exp_avg_sq"].shape == (32, 32)
    assert torch.equal(loaded_state["exp_avg"], make_row_tensor(32, 32, 32))
    assert torch.equal(loaded_state["exp_avg_sq"], make_row_tensor(2032, 32, 32))


def test_inspect_optimizer_shards_reports_missing_expected_ranks(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    shard_info = inspect_optimizer_shards(str(checkpoint_dir), 53334, saved_world_size=2)

    assert shard_info["saved_world_size"] == 2
    assert shard_info["expected_ranks"] == [0, 1]
    assert shard_info["available_ranks"] == []
    assert shard_info["missing_ranks"] == [0, 1]


def test_inspect_optimizer_shards_infers_world_size_from_available_files(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    torch.save({"state": {}, "param_groups": []}, checkpoint_dir / "optim_000012_rank0.pt")

    shard_info = inspect_optimizer_shards(str(checkpoint_dir), 12)

    assert shard_info["saved_world_size"] == 1
    assert shard_info["expected_ranks"] == [0]
    assert shard_info["available_ranks"] == [0]
    assert shard_info["missing_ranks"] == []


def test_infer_use_qwen3_dense_mlp_disables_gated_dense_mlp_for_legacy_checkpoints():
    model_config_kwargs = {
        "n_layer": 4,
        "n_exp": 2,
        "moe_start_layer": 2,
        "moe_layer_stride": 1,
        "num_moe_layers": -1,
    }
    model_data = {
        "transformer.h.0.mlp.c_fc.weight": torch.zeros(32, 8),
        "transformer.h.0.mlp.c_proj.weight": torch.zeros(8, 32),
        "transformer.h.1.mlp.c_fc.weight": torch.zeros(32, 8),
        "transformer.h.1.mlp.c_proj.weight": torch.zeros(8, 32),
    }

    _infer_use_qwen3_dense_mlp(model_data, model_config_kwargs)

    assert model_config_kwargs["use_qwen3_dense_mlp"] is False


def test_infer_use_qwen3_dense_mlp_keeps_gated_dense_mlp_when_gate_proj_exists():
    model_config_kwargs = {
        "n_layer": 2,
        "n_exp": 1,
    }
    model_data = {
        "transformer.h.0.mlp.gate_proj.weight": torch.zeros(32, 8),
        "transformer.h.1.mlp.gate_proj.weight": torch.zeros(32, 8),
    }

    _infer_use_qwen3_dense_mlp(model_data, model_config_kwargs)

    assert model_config_kwargs["use_qwen3_dense_mlp"] is True


def test_override_disabled_kappa_bias_keeps_loadable_zero_bias_tensors():
    model_data = {
        "transformer.h.0.mlp.experts.kappa_bias": torch.randn(4, 8),
        "transformer.h.1.mlp.experts.kappa_bias": torch.randn(4, 8),
        "transformer.h.0.mlp.experts.kappa_scale": torch.randn(4, 8),
        "transformer.h.1.mlp.kappa_scale": torch.randn(8),
        "global_kappa_scale": torch.randn(1),
        "transformer.h.1.mlp.experts.gate_proj": torch.randn(4, 8, 16),
    }
    model_kwargs = {"use_kappa_swiglu": False, "eval_capacity": 1.5}

    sanitized_kwargs = _override_kappa_bias_values(model_data, model_kwargs)
    sanitized_kwargs = _override_kappa_scale_values(model_data, sanitized_kwargs)

    assert "use_kappa_swiglu" not in sanitized_kwargs
    assert sanitized_kwargs["eval_capacity"] == 1.5
    assert torch.count_nonzero(model_data["transformer.h.0.mlp.experts.kappa_bias"]) == 0
    assert torch.count_nonzero(model_data["transformer.h.1.mlp.experts.kappa_bias"]) == 0
    assert torch.count_nonzero(model_data["transformer.h.0.mlp.experts.kappa_scale"]) == 0
    assert torch.count_nonzero(model_data["transformer.h.1.mlp.kappa_scale"]) == 0
    assert torch.count_nonzero(model_data["global_kappa_scale"]) == 0

def test_override_kappa_bias_fill_value_sets_constant_bias_tensors():
    model_data = {
        "transformer.h.0.mlp.experts.kappa_bias": torch.randn(4, 8),
        "transformer.h.1.mlp.experts.kappa_bias": torch.randn(4, 8),
    }
    model_kwargs = {"kappa_bias_fill_value": 0.4, "eval_capacity": 1.5}

    sanitized_kwargs = _override_kappa_bias_values(model_data, model_kwargs)

    assert "kappa_bias_fill_value" not in sanitized_kwargs
    assert sanitized_kwargs["eval_capacity"] == 1.5
    assert torch.all(model_data["transformer.h.0.mlp.experts.kappa_bias"] == 0.4)
    assert torch.all(model_data["transformer.h.1.mlp.experts.kappa_bias"] == 0.4)


def test_override_kappa_scale_fill_value_sets_constant_scale_tensors():
    model_data = {
        "transformer.h.0.mlp.experts.kappa_scale": torch.randn(4, 8),
        "transformer.h.1.mlp.kappa_scale": torch.randn(8),
        "global_kappa_scale": torch.randn(1),
    }
    model_kwargs = {"kappa_scale_fill_value": 0.25, "eval_capacity": 1.5}

    sanitized_kwargs = _override_kappa_scale_values(model_data, model_kwargs)

    assert "kappa_scale_fill_value" not in sanitized_kwargs
    assert sanitized_kwargs["eval_capacity"] == 1.5
    assert torch.all(model_data["transformer.h.0.mlp.experts.kappa_scale"] == 0.25)
    assert torch.all(model_data["transformer.h.1.mlp.kappa_scale"] == 0.25)
    assert torch.all(model_data["global_kappa_scale"] == 0.25)


def test_infer_kappa_bias_detects_rank1_residual_checkpoint_layout():
    model_config_kwargs = {
        "n_layer": 2,
        "n_exp": 2,
    }
    model_data = {
        "transformer.h.1.mlp.experts.kappa_bias_expert": torch.ones(2),
        "transformer.h.1.mlp.experts.kappa_bias_intermediate": torch.zeros(16),
        "transformer.h.1.mlp.experts.kappa_bias_residual": torch.zeros(2, 16),
    }

    _infer_kappa_bias(model_data, model_config_kwargs)

    assert model_config_kwargs["use_kappa_swiglu"] is True
    assert model_config_kwargs["kappa_bias_start_layer"] == 1


def test_patch_missing_keys_initializes_newly_enabled_kappa_parameters_to_zero():
    model_data = {}
    config = GPTConfig(
        n_layer=2,
        n_exp=2,
        n_embd=4,
        moe_start_layer=1,
        use_kappa_swiglu=True,
        constant_kappa_bias_dense_layers=True,
        total_ut_steps=2,
    )

    _patch_missing_keys(model_data, config)

    dense_bias = model_data["transformer.h.0.mlp.kappa_bias"]
    expert_bias = model_data["transformer.h.1.mlp.experts.kappa_bias"]
    expert_scale = model_data["transformer.h.1.mlp.experts.kappa_scale"]
    assert dense_bias.shape == (2, 16)
    assert expert_bias.shape == (2, 2, 16)
    assert expert_scale.shape == (2, 2, 16)
    assert torch.count_nonzero(dense_bias) == 0
    assert torch.count_nonzero(expert_bias) == 0
    assert torch.count_nonzero(expert_scale) == 0


def test_patch_missing_keys_uses_loaded_checkpoint_device_for_new_kappa_parameters():
    model_data = {"checkpoint_weight": torch.empty((), device="meta")}
    config = GPTConfig(
        n_layer=2,
        n_exp=2,
        n_embd=4,
        moe_start_layer=1,
        use_kappa_swiglu=True,
        constant_kappa_bias_dense_layers=True,
        total_ut_steps=2,
    )

    _patch_missing_keys(model_data, config)

    patched_keys = (
        "transformer.h.0.mlp.kappa_bias",
        "transformer.h.1.mlp.experts.kappa_bias",
        "transformer.h.1.mlp.experts.kappa_scale",
    )
    assert all(model_data[key].device.type == "meta" for key in patched_keys)


def test_override_kappa_bias_fill_value_keeps_rank1_residual_checkpoint_loadable():
    fill_value = 0.4
    model_data = {
        "transformer.h.0.mlp.experts.gate_proj": torch.randn(2, 4, 16),
        "transformer.h.0.mlp.experts.kappa_bias_expert": torch.randn(2),
        "transformer.h.0.mlp.experts.kappa_bias_intermediate": torch.randn(16),
        "transformer.h.0.mlp.experts.kappa_bias_residual": torch.randn(2, 16),
    }
    model_kwargs = {
        "kappa_bias_fill_value": fill_value,
    }

    sanitized_kwargs = _override_kappa_bias_values(model_data, model_kwargs)
    model_config_kwargs = {
        "n_layer": 1,
        "moe_start_layer": 0,
        "moe_layer_stride": 1,
        "n_exp": 2,
        "n_embd": 4,
    }
    model_config_kwargs.update(sanitized_kwargs)
    _infer_kappa_bias(model_data, model_config_kwargs)
    config = GPTConfig(**model_config_kwargs)

    _patch_missing_keys(model_data, config)

    torch.testing.assert_close(
        model_data["transformer.h.0.mlp.experts.kappa_bias"],
        torch.full((config.total_ut_steps, 2, 16), fill_value),
    )


def test_patch_missing_keys_resizes_kappa_parameters_for_ut_passes():
    config = GPTConfig(
        n_layer=2,
        moe_start_layer=1,
        num_moe_layers=1,
        n_exp=2,
        n_embd=4,
        total_ut_steps=3,
        use_kappa_swiglu=True,
    )
    dense_bias = torch.arange(16, dtype=torch.float32)
    moe_bias = torch.arange(64, dtype=torch.float32).reshape(2, 2, 16)
    moe_scale = moe_bias + 100.0
    model_data = {
        "transformer.h.0.mlp.kappa_bias": dense_bias,
        "transformer.h.1.mlp.experts.kappa_bias": moe_bias,
        "transformer.h.1.mlp.experts.kappa_scale": moe_scale,
        "global_kappa_bias": torch.tensor([0.25]),
        "global_kappa_scale": torch.tensor([0.5]),
    }

    _patch_missing_keys(model_data, config)

    torch.testing.assert_close(
        model_data["transformer.h.0.mlp.kappa_bias"], dense_bias.repeat(3, 1)
    )
    torch.testing.assert_close(
        model_data["transformer.h.1.mlp.experts.kappa_bias"],
        torch.cat((moe_bias, moe_bias[-1:])),
    )
    torch.testing.assert_close(
        model_data["transformer.h.1.mlp.experts.kappa_scale"],
        torch.cat((moe_scale, moe_scale[-1:])),
    )
    torch.testing.assert_close(
        model_data["global_kappa_bias"], torch.full((3, 1), 0.25)
    )
    torch.testing.assert_close(
        model_data["global_kappa_scale"], torch.full((3, 1), 0.5)
    )


def test_patch_missing_keys_creates_per_ut_layer_scalars():
    config = GPTConfig(n_layer=2, total_ut_steps=3)
    model_data = {}

    _patch_missing_keys(model_data, config)

    torch.testing.assert_close(model_data["resid_lambdas"], torch.ones(3, 2))
    torch.testing.assert_close(model_data["x0_lambdas"], torch.zeros(3, 2))
    torch.testing.assert_close(
        model_data["ut_source_lambdas"],
        torch.tensor([0.0, 1.0, 1.0]),
    )


def test_patch_missing_keys_expands_legacy_layer_scalars_across_ut_steps():
    config = GPTConfig(n_layer=2, total_ut_steps=3)
    resid_lambdas = torch.tensor([0.8, 0.9])
    x0_lambdas = torch.tensor([0.1, 0.2])
    model_data = {
        "resid_lambdas": resid_lambdas,
        "x0_lambdas": x0_lambdas,
    }

    _patch_missing_keys(model_data, config)

    torch.testing.assert_close(model_data["resid_lambdas"], resid_lambdas.repeat(3, 1))
    torch.testing.assert_close(model_data["x0_lambdas"], x0_lambdas.repeat(3, 1))


def test_patch_missing_keys_extends_per_ut_layer_scalars_with_last_step():
    config = GPTConfig(n_layer=2, total_ut_steps=3)
    resid_lambdas = torch.tensor([[0.7, 0.8], [0.9, 1.0]])
    x0_lambdas = torch.tensor([[0.0, 0.1], [0.2, 0.3]])
    model_data = {
        "resid_lambdas": resid_lambdas,
        "x0_lambdas": x0_lambdas,
    }

    _patch_missing_keys(model_data, config)

    torch.testing.assert_close(
        model_data["resid_lambdas"], torch.cat((resid_lambdas, resid_lambdas[-1:]))
    )
    torch.testing.assert_close(
        model_data["x0_lambdas"], torch.cat((x0_lambdas, x0_lambdas[-1:]))
    )


def test_patch_missing_keys_converts_and_extends_ut_source_lambdas():
    config = GPTConfig(n_layer=3, total_ut_steps=3, ut_destination=1)
    ut_source_lambdas = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.8, 0.2],
    ])
    model_data = {"ut_source_lambdas": ut_source_lambdas}

    _patch_missing_keys(model_data, config)

    torch.testing.assert_close(
        model_data["ut_source_lambdas"],
        torch.tensor([0.0, 0.8, 0.8]),
    )


def test_patch_missing_keys_converts_full_kappa_bias_to_rank1_factors():
    config = GPTConfig(
        n_layer=1,
        moe_start_layer=0,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
    )
    full_bias = torch.randn(2, 16)
    model_data = {
        "transformer.h.0.mlp.experts.gate_proj": torch.randn(2, 4, 16),
        "transformer.h.0.mlp.experts.kappa_bias": full_bias.clone(),
    }

    _patch_missing_keys(model_data, config)

    assert "transformer.h.0.mlp.experts.kappa_bias" not in model_data
    assert "transformer.h.0.mlp.experts.kappa_bias_expert" in model_data
    assert "transformer.h.0.mlp.experts.kappa_bias_intermediate" in model_data
    reconstructed = (
        model_data["transformer.h.0.mlp.experts.kappa_bias_expert"].unsqueeze(1)
        * model_data["transformer.h.0.mlp.experts.kappa_bias_intermediate"].unsqueeze(0)
    )
    assert reconstructed.shape == full_bias.shape


def test_patch_missing_keys_converts_full_kappa_bias_to_rank1_residual_factors():
    config = GPTConfig(
        n_layer=1,
        moe_start_layer=0,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
    )
    full_bias = torch.randn(2, 16)
    model_data = {
        "transformer.h.0.mlp.experts.gate_proj": torch.randn(2, 4, 16),
        "transformer.h.0.mlp.experts.kappa_bias": full_bias.clone(),
    }

    _patch_missing_keys(model_data, config)

    assert "transformer.h.0.mlp.experts.kappa_bias" not in model_data
    assert "transformer.h.0.mlp.experts.kappa_bias_expert" in model_data
    assert "transformer.h.0.mlp.experts.kappa_bias_intermediate" in model_data
    assert "transformer.h.0.mlp.experts.kappa_bias_residual" in model_data
    reconstructed = (
        model_data["transformer.h.0.mlp.experts.kappa_bias_expert"].unsqueeze(1)
        * model_data["transformer.h.0.mlp.experts.kappa_bias_intermediate"].unsqueeze(0)
        + model_data["transformer.h.0.mlp.experts.kappa_bias_residual"]
    )
    torch.testing.assert_close(reconstructed, full_bias)


def test_delete_old_checkpoints_removes_all_older_steps(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    for filename in (
        "model_000010.pt",
        "meta_000010.json",
        "optim_000010_rank0.pt",
        "optim_000010_rank3.pt",
        "model_000015.pt",
        "meta_000015.json",
        "optim_000015_rank1.pt",
        "model_000020.pt",
        "meta_000020.json",
        "optim_000020_rank0.pt",
        "notes.txt",
    ):
        (checkpoint_dir / filename).write_text("x", encoding="utf-8")

    deleted_paths = delete_old_checkpoints(str(checkpoint_dir), 20)

    assert {Path(path).name for path in deleted_paths} == {
        "model_000010.pt",
        "meta_000010.json",
        "optim_000010_rank0.pt",
        "optim_000010_rank3.pt",
        "model_000015.pt",
        "meta_000015.json",
        "optim_000015_rank1.pt",
    }
    assert not (checkpoint_dir / "model_000010.pt").exists()
    assert not (checkpoint_dir / "optim_000015_rank1.pt").exists()
    assert (checkpoint_dir / "model_000020.pt").exists()
    assert (checkpoint_dir / "meta_000020.json").exists()
    assert (checkpoint_dir / "optim_000020_rank0.pt").exists()
    assert (checkpoint_dir / "notes.txt").exists()


def test_save_checkpoint_skips_optimizer_shard_when_optimizer_data_is_none(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"

    save_checkpoint(
        str(checkpoint_dir),
        20,
        {"weight": torch.ones(1)},
        None,
        {"optimizer_world_size": 0},
        rank=0,
    )

    assert (checkpoint_dir / "model_000020.pt").exists()
    assert (checkpoint_dir / "meta_000020.json").exists()
    assert not (checkpoint_dir / "optim_000020_rank0.pt").exists()


def test_checkpoint_file_sizes_skip_changed_model_layout(tmp_path):
    save_checkpoint(str(tmp_path), 10, {"weight": torch.ones(32)}, None, {})
    model_data = {"weight": torch.ones(8)}
    assert snapshot_checkpoint_file_sizes(str(tmp_path), 20, model_data=model_data) == (None, None)

    save_checkpoint(str(tmp_path), 20, model_data, None, {})
    assert validate_checkpoint_file_sizes(str(tmp_path), 20) is None

    save_checkpoint(str(tmp_path), 30, model_data, None, {})
    assert validate_checkpoint_file_sizes(str(tmp_path), 30) == 20
    model_path = tmp_path / "model_000030.pt"
    model_path.write_bytes(model_path.read_bytes()[:-64])
    with pytest.raises(ValueError, match="validation failed"):
        validate_checkpoint_file_sizes(str(tmp_path), 30)


def test_checkpoint_file_sizes_skip_legacy_reference_before_predelete(tmp_path):
    model_data = {"weight": torch.ones(8)}
    save_checkpoint(str(tmp_path), 10, model_data, None, {})
    (tmp_path / "meta_000010.json").write_text("{}")

    assert snapshot_checkpoint_file_sizes(str(tmp_path), 20, model_data=model_data) == (None, None)
    delete_old_checkpoints(str(tmp_path), 20)
    save_checkpoint(str(tmp_path), 20, model_data, None, {})
    assert validate_checkpoint_file_sizes(str(tmp_path), 20) is None


def test_checkpoint_file_sizes_snapshot_matching_signature_before_predelete(tmp_path):
    model_data = {"weight": torch.ones(8)}
    save_checkpoint(str(tmp_path), 10, model_data, None, {})
    comparison_step, reference_file_sizes = snapshot_checkpoint_file_sizes(
        str(tmp_path), 20, model_data=model_data,
    )
    assert comparison_step == 10
    delete_old_checkpoints(str(tmp_path), 20)
    save_checkpoint(str(tmp_path), 20, model_data, None, {})
    assert validate_checkpoint_file_sizes(
        str(tmp_path), 20, comparison_step=comparison_step,
        reference_file_sizes=reference_file_sizes,
    ) == 10


def test_validate_checkpoint_file_sizes_matches_previous_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    write_sized_file(checkpoint_dir / "model_000010.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000010.json", 120)
    write_sized_file(checkpoint_dir / "optim_000010_rank0.pt", 180)
    write_sized_file(checkpoint_dir / "optim_000010_rank1.pt", 180)

    write_sized_file(checkpoint_dir / "model_000020.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000020.json", 132)
    write_sized_file(checkpoint_dir / "optim_000020_rank0.pt", 192)
    write_sized_file(checkpoint_dir / "optim_000020_rank1.pt", 180)

    comparison_step = validate_checkpoint_file_sizes(
        str(checkpoint_dir),
        20,
        expected_optimizer_ranks=[0, 1],
    )

    assert comparison_step == 10


def test_validate_checkpoint_file_sizes_ignores_meta_size_changes(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    write_sized_file(checkpoint_dir / "model_000010.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000010.json", 120)
    write_sized_file(checkpoint_dir / "optim_000010_rank0.pt", 180)

    write_sized_file(checkpoint_dir / "model_000020.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000020.json", 4096)
    write_sized_file(checkpoint_dir / "optim_000020_rank0.pt", 180)

    comparison_step = validate_checkpoint_file_sizes(
        str(checkpoint_dir),
        20,
        expected_optimizer_ranks=[0],
    )

    assert comparison_step == 10


def test_validate_checkpoint_file_sizes_handles_model_only_checkpoints(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    write_sized_file(checkpoint_dir / "model_000010.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000010.json", 120)

    write_sized_file(checkpoint_dir / "model_000020.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000020.json", 132)

    comparison_step = validate_checkpoint_file_sizes(
        str(checkpoint_dir),
        20,
        expected_optimizer_ranks=None,
    )

    assert comparison_step == 10


def test_validate_checkpoint_file_sizes_raises_when_current_files_are_missing(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    write_sized_file(checkpoint_dir / "model_000010.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000010.json", 120)
    write_sized_file(checkpoint_dir / "optim_000010_rank0.pt", 180)
    write_sized_file(checkpoint_dir / "optim_000010_rank1.pt", 180)

    write_sized_file(checkpoint_dir / "model_000020.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000020.json", 120)
    write_sized_file(checkpoint_dir / "optim_000020_rank0.pt", 180)

    with pytest.raises(ValueError, match="missing expected files"):
        validate_checkpoint_file_sizes(
            str(checkpoint_dir),
            20,
            expected_optimizer_ranks=[0, 1],
        )


def test_validate_checkpoint_file_sizes_returns_none_without_matching_layout(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    write_sized_file(checkpoint_dir / "model_000010.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000010.json", 120)
    write_sized_file(checkpoint_dir / "optim_000010_rank0.pt", 180)

    write_sized_file(checkpoint_dir / "model_000020.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000020.json", 120)
    write_sized_file(checkpoint_dir / "optim_000020_rank0.pt", 180)
    write_sized_file(checkpoint_dir / "optim_000020_rank1.pt", 180)

    comparison_step = validate_checkpoint_file_sizes(
        str(checkpoint_dir),
        20,
        expected_optimizer_ranks=[0, 1],
    )

    assert comparison_step is None


def test_validate_checkpoint_file_sizes_with_snapshot_after_predelete(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    write_sized_file(checkpoint_dir / "model_000010.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000010.json", 120)
    write_sized_file(checkpoint_dir / "optim_000010_rank0.pt", 180)
    write_sized_file(checkpoint_dir / "optim_000010_rank1.pt", 180)

    comparison_step, reference_file_sizes = snapshot_checkpoint_file_sizes(
        str(checkpoint_dir),
        20,
        expected_optimizer_ranks=[0, 1],
    )
    delete_old_checkpoints(str(checkpoint_dir), 20)

    write_sized_file(checkpoint_dir / "model_000020.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000020.json", 132)
    write_sized_file(checkpoint_dir / "optim_000020_rank0.pt", 192)
    write_sized_file(checkpoint_dir / "optim_000020_rank1.pt", 180)

    validated_step = validate_checkpoint_file_sizes(
        str(checkpoint_dir),
        20,
        expected_optimizer_ranks=[0, 1],
        comparison_step=comparison_step,
        reference_file_sizes=reference_file_sizes,
    )

    assert validated_step == 10


def test_delete_old_checkpoints_can_run_without_validation_snapshot(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    write_sized_file(checkpoint_dir / "model_000010.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000010.json", 120)
    write_sized_file(checkpoint_dir / "optim_000010_rank0.pt", 180)

    comparison_step, reference_file_sizes = snapshot_checkpoint_file_sizes(
        str(checkpoint_dir),
        20,
        expected_optimizer_ranks=[0, 1],
    )

    assert comparison_step is None
    assert reference_file_sizes is None

    deleted_paths = delete_old_checkpoints(str(checkpoint_dir), 20)

    assert {Path(path).name for path in deleted_paths} == {
        "model_000010.pt",
        "meta_000010.json",
        "optim_000010_rank0.pt",
    }
    assert not (checkpoint_dir / "model_000010.pt").exists()


def test_validate_checkpoint_file_sizes_raises_on_large_size_mismatch(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    write_sized_file(checkpoint_dir / "model_000010.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000010.json", 120)
    write_sized_file(checkpoint_dir / "optim_000010_rank0.pt", 180)

    write_sized_file(checkpoint_dir / "model_000020.pt", 256)
    write_sized_file(checkpoint_dir / "meta_000020.json", 120)
    write_sized_file(checkpoint_dir / "optim_000020_rank0.pt", 240)

    with pytest.raises(ValueError, match="validation failed"):
        validate_checkpoint_file_sizes(
            str(checkpoint_dir),
            20,
            expected_optimizer_ranks=[0],
        )


def test_patch_missing_keys_removes_legacy_gate_proj_factors():
    config = GPTConfig(
        n_layer=1,
        moe_start_layer=0,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=4,
    )

    dense_gate_proj = torch.randn(2, 4, 16)
    model_data = {
        "transformer.h.0.mlp.experts.gate_proj": dense_gate_proj,
        "transformer.h.0.mlp.experts.gate_proj_a": torch.randn(2, 4, 2),
        "transformer.h.0.mlp.experts.gate_proj_b": torch.randn(2, 2, 16),
    }

    _patch_missing_keys(model_data, config)

    assert torch.equal(model_data["transformer.h.0.mlp.experts.gate_proj"], dense_gate_proj)
    assert "transformer.h.0.mlp.experts.gate_proj_a" not in model_data
    assert "transformer.h.0.mlp.experts.gate_proj_b" not in model_data