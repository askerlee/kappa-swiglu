import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "scalar_stats.py"
SPEC = importlib.util.spec_from_file_location("scalar_stats", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_checkpoint(
    path: Path,
    resid_lambdas: list[float],
    x0_lambdas: list[float],
    ut_source_lambdas: list[float],
) -> None:
    torch.save(
        {
            "model": {
                "resid_lambdas": torch.tensor(resid_lambdas),
                "x0_lambdas": torch.tensor(x0_lambdas),
                "ut_source_lambdas": torch.tensor(ut_source_lambdas),
            }
        },
        path,
    )


def test_checkpoint_statistics_and_multi_checkpoint_cli(tmp_path: Path):
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    write_checkpoint(first_path, [-2.0, 0.0, 4.0], [-1.0, 1.0], [0.0, 0.5, 1.0])
    write_checkpoint(second_path, [1.0, 1.0], [0.0, 3.0], [-0.5, 0.5])

    statistics = MODULE.checkpoint_statistics(first_path)

    assert statistics["resid_lambdas"] == {
        "mean": 2 / 3,
        "std": (56 / 9) ** 0.5,
        "max": 4.0,
        "min": -2.0,
        "abs_max": 4.0,
    }
    assert statistics["x0_lambdas"] == {
        "mean": 0.0,
        "std": 1.0,
        "max": 1.0,
        "min": -1.0,
        "abs_max": 1.0,
    }
    assert statistics["ut_source_lambdas"] == {
        "mean": 0.5,
        "std": (1 / 6) ** 0.5,
        "max": 1.0,
        "min": 0.0,
        "abs_max": 1.0,
    }

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(first_path), str(second_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.count("Checkpoint:") == 2
    assert result.stdout.count("resid_lambdas:") == 2
    assert result.stdout.count("x0_lambdas:") == 2
    assert result.stdout.count("ut_source_lambdas:") == 2
    assert "mean=0.67 std=2.49 max=4.00 min=-2.00 abs_max=4.00" in result.stdout
    assert "resid [-2.00, 0.00, 4.00]" in result.stdout
    assert "x0    [-1.00, 1.00]" in result.stdout
    assert "source [0.00, 0.50, 1.00]" in result.stdout


def test_kappa_dimension_stds_and_correlations(tmp_path: Path, capsys):
    bias = torch.tensor([[[0.0, 2.0], [4.0, 6.0]], [[0.0, 4.0], [8.0, 12.0]]])
    scale = -2 * bias + 1
    assert MODULE.calculate_kappa_dimension_stds(bias) == {
        "std_dim2_mean": [2.0, 4.0],
        "std_dim3_mean": [1.0, 2.0],
    }
    assert MODULE.calculate_kappa_dimension_stds(scale) == {
        "std_dim2_mean": [4.0, 8.0],
        "std_dim3_mean": [2.0, 4.0],
    }
    assert MODULE.calculate_kappa_dimension_stds(bias[0]) == {}
    assert math.isclose(MODULE.calculate_kappa_correlation(bias, scale), -1.0)
    assert math.isnan(MODULE.calculate_kappa_correlation(bias, torch.ones_like(bias)))
    assert math.isnan(MODULE.calculate_kappa_correlation(torch.ones(1), torch.ones(1)))

    path = tmp_path / "kappa.pt"
    torch.save({
        "resid_lambdas": torch.ones(2, 2),
        "transformer.h.1.mlp.experts.kappa_scale": scale,
        "transformer.h.0.mlp.experts.kappa_bias": bias + 100,
        "transformer.h.1.mlp.experts.kappa_bias": bias,
    }, path)
    MODULE.print_per_layer_kappa_statistics(path)
    output = capsys.readouterr().out
    assert "std_dim2_mean=2.00 std_dim3_mean=1.00" in output
    assert "std_dim2_mean=8.00 std_dim3_mean=4.00" in output
    assert "kappa_bias/kappa_scale layer=1 pass=0: pearson=-1.0000" in output
    assert "kappa_bias/kappa_scale layer=1 pass=1: pearson=-1.0000" in output
    assert "kappa_bias/kappa_scale overall pass=0: pearson=-1.0000" in output
    assert "kappa_bias/kappa_scale layer=0" not in output


def test_kappa_single_pass_and_global_pairs(tmp_path: Path, capsys):
    path = tmp_path / "single_pass.pt"
    bias = torch.tensor([[[0.0, 2.0], [4.0, 6.0]]])
    torch.save({
        "resid_lambdas": torch.ones(1, 1),
        "transformer.h.0.mlp.experts.kappa_bias": bias,
        "transformer.h.0.mlp.experts.kappa_scale": bias * 2,
    }, path)
    MODULE.print_per_layer_kappa_statistics(path)
    output = capsys.readouterr().out
    assert "std_dim2_mean=2.00 std_dim3_mean=1.00" in output
    assert "kappa_bias/kappa_scale layer=0 pass=0: pearson=1.0000" in output

    torch.save({
        "resid_lambdas": torch.ones(2, 1),
        "global_kappa_bias": torch.tensor([[1.0], [2.0]]),
        "global_kappa_scale": torch.tensor([[3.0], [4.0]]),
    }, path)
    MODULE.print_per_layer_kappa_statistics(path)
    output = capsys.readouterr().out
    assert "kappa_bias/kappa_scale layer=global pass=0: pearson=nan" in output
    assert "kappa_bias/kappa_scale layer=global pass=1: pearson=nan" in output
    assert "std_dim2_mean" not in output


def test_kappa_correlation_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes must match"):
        MODULE.calculate_kappa_correlation(torch.ones(2, 3), torch.ones(3, 2))


def test_kappa_ratio_mean(tmp_path: Path, capsys):
    assert MODULE.calculate_kappa_ratio_mean(
        torch.tensor([2.0, -6.0, 100.0]), torch.tensor([2.0, 3.0, 0.0])
    ) == -0.5
    assert math.isnan(MODULE.calculate_kappa_ratio_mean(torch.ones(2), torch.zeros(2)))
    with pytest.raises(ValueError, match="shapes must match"):
        MODULE.calculate_kappa_ratio_mean(torch.ones(2, 3), torch.ones(3, 2))

    path = tmp_path / "ratios.pt"
    torch.save({
        "resid_lambdas": torch.ones(2, 2),
        "transformer.h.0.mlp.experts.kappa_bias": torch.tensor([[2.0, 12.0], [1.0, 3.0]]),
        "transformer.h.0.mlp.experts.kappa_scale": torch.tensor([[2.0, 4.0], [0.0, 0.0]]),
        "transformer.h.1.mlp.experts.kappa_bias": torch.tensor([[16.0], [-6.0]]),
        "transformer.h.1.mlp.experts.kappa_scale": torch.tensor([[2.0], [3.0]]),
    }, path)
    MODULE.print_per_layer_kappa_statistics(path)
    output = capsys.readouterr().out
    assert "layer=0 pass=0: pearson=1.0000 ratio_mean=2.0000 residual_abs_mean=3.0000 residual_std=3.0000" in output
    assert "layer=0 pass=1: pearson=nan ratio_mean=nan residual_abs_mean=nan residual_std=nan" in output
    assert "layer=1 pass=0: pearson=nan ratio_mean=8.0000 residual_abs_mean=0.0000 residual_std=0.0000" in output
    overall_lines = [line for line in output.splitlines() if "overall pass=" in line]
    assert "overall pass=0:" in overall_lines[0]
    assert "ratio_mean=4.0000" in overall_lines[0]
    assert "residual_abs_mean=6.0000" in overall_lines[0]
    assert "residual_std=6.1824" in overall_lines[0]
    assert "overall pass=1:" in overall_lines[1]
    assert "ratio_mean=-2.0000" in overall_lines[1]
    assert "residual_abs_mean=1.3333" in overall_lines[1]
    assert "residual_std=1.2472" in overall_lines[1]


@pytest.mark.parametrize(
    "bias, scale, expected, expected_std",
    [
        ([2.0, -4.0, 0.0], [1.0, -2.0, 0.0], 0.0, 0.0),
        ([2.0, 12.0], [2.0, 4.0], 3.0, 3.0),
        ([2.0, -6.0, 100.0], [2.0, 3.0, 0.0], 215 / 6,
         math.sqrt(10029.25 / 3 - (98.5 / 3) ** 2)),
        ([1.0, 2.0], [0.0, 0.0], float("nan"), float("nan")),
        ([6.0], [2.0], 0.0, 0.0),
    ],
)
def test_kappa_residual_abs_mean(bias, scale, expected, expected_std):
    bias = torch.tensor(bias)
    scale = torch.tensor(scale)
    ratio_mean = MODULE.calculate_kappa_ratio_mean(bias, scale)
    residual = MODULE.calculate_kappa_residual_abs_mean(bias, scale, ratio_mean)
    statistics = MODULE.calculate_kappa_residual_statistics(bias, scale, ratio_mean)
    if math.isnan(expected):
        assert math.isnan(residual)
        assert math.isnan(statistics["abs_mean"])
        assert math.isnan(statistics["std"])
    else:
        assert residual == pytest.approx(expected)
        assert statistics["abs_mean"] == pytest.approx(expected)
        assert statistics["std"] == pytest.approx(expected_std)


def test_kappa_residual_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes must match"):
        MODULE.calculate_kappa_residual_abs_mean(torch.ones(2, 3), torch.ones(3, 2), 1.0)


def test_kappa_shared_alpha_across_passes(tmp_path: Path, capsys):
    bias = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    scale = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    assert MODULE.calculate_kappa_shared_alpha(bias, scale) == 2.0
    assert MODULE.calculate_kappa_shared_alpha(
        torch.tensor([2.0, 12.0]), torch.tensor([2.0, 4.0])
    ) == pytest.approx(2.6)

    path = tmp_path / "shared_alpha.pt"
    torch.save({
        "resid_lambdas": torch.ones(2, 2),
        "transformer.h.0.mlp.experts.kappa_bias": bias,
        "transformer.h.0.mlp.experts.kappa_scale": scale,
        "transformer.h.1.mlp.experts.kappa_bias": 4 * scale,
        "transformer.h.1.mlp.experts.kappa_scale": scale,
    }, path)
    MODULE.print_per_layer_kappa_statistics(path)
    output = capsys.readouterr().out
    layer_lines = [line for line in output.splitlines() if "kappa_bias/kappa_scale layer=" in line]
    assert len(layer_lines) == 4
    assert "shared_alpha=2.0000 shared_residual_abs_mean=1.5000 shared_residual_std=0.5000 shared_residual_rel_mae=1.0000" in layer_lines[0]
    assert "shared_alpha=2.0000 shared_residual_abs_mean=1.5000 shared_residual_std=0.5000 shared_residual_rel_mae=0.3333" in layer_lines[1]
    for line in layer_lines[2:]:
        assert "shared_alpha=4.0000 shared_residual_abs_mean=0.0000 shared_residual_std=0.0000 shared_residual_rel_mae=0.0000" in line


def test_kappa_shared_alpha_edge_cases():
    assert math.isnan(MODULE.calculate_kappa_shared_alpha(torch.ones(2), torch.zeros(2)))
    assert MODULE.calculate_kappa_shared_alpha(
        torch.tensor([-2.0, 100.0]), torch.tensor([1.0, 0.0])
    ) == -2.0
    with pytest.raises(ValueError, match="shapes must match"):
        MODULE.calculate_kappa_shared_alpha(torch.ones(2, 3), torch.ones(3, 2))
    statistics = MODULE.calculate_kappa_residual_statistics(torch.zeros(2), torch.ones(2), 0.0)
    assert statistics["abs_mean"] == 0.0
    assert math.isnan(statistics["relative_abs_mean"])