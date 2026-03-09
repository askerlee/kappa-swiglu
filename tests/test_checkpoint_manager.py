import copy
from types import SimpleNamespace

import torch

from nanochat.checkpoint_manager import load_optimizer_state_dict, reshard_optimizer_state_dict


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