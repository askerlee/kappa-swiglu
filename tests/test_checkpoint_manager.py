import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanochat.checkpoint_manager import _patch_missing_keys, delete_old_checkpoints, inspect_optimizer_shards, load_optimizer_state_dict, reshard_optimizer_state_dict, snapshot_checkpoint_file_sizes, validate_checkpoint_file_sizes
from nanochat.configuration_nanomoe_gpt import GPTConfig


def make_optimizer(param_groups):
    return SimpleNamespace(param_groups=param_groups)


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


def test_patch_missing_keys_factorizes_dense_gate_proj_when_rank_enabled():
    config = GPTConfig(
        n_layer=1,
        moe_start_layer=0,
        stride=1,
        n_exp=2,
        n_embd=4,
        exp_gate_proj_rank=2,
        exp_gate_proj_m=3,
    )

    gate_proj_a = torch.randn(2, 4, 2, 3)
    gate_proj_b = torch.randn(2, 2, 16, 3)
    dense_gate_proj = torch.einsum('ehrm,erim->ehim', gate_proj_a, gate_proj_b)
    model_data = {
        "transformer.h.0.mlp.experts.gate_proj": dense_gate_proj.clone(),
    }

    _patch_missing_keys(model_data, config)

    assert "transformer.h.0.mlp.experts.gate_proj" not in model_data
    assert "transformer.h.0.mlp.experts.gate_proj_a" in model_data
    assert "transformer.h.0.mlp.experts.gate_proj_b" in model_data

    reconstructed = torch.einsum(
        'ehrm,erim->ehim',
        model_data["transformer.h.0.mlp.experts.gate_proj_a"],
        model_data["transformer.h.0.mlp.experts.gate_proj_b"],
    )
    torch.testing.assert_close(reconstructed, dense_gate_proj, atol=1e-5, rtol=1e-5)