import importlib.util
import subprocess
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "lambda_stats.py"
SPEC = importlib.util.spec_from_file_location("lambda_stats", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_checkpoint(path: Path, resid_lambdas: list[float], x0_lambdas: list[float]) -> None:
    torch.save(
        {
            "model": {
                "resid_lambdas": torch.tensor(resid_lambdas),
                "x0_lambdas": torch.tensor(x0_lambdas),
            }
        },
        path,
    )


def test_checkpoint_statistics_and_multi_checkpoint_cli(tmp_path: Path):
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    write_checkpoint(first_path, [-2.0, 0.0, 4.0], [-1.0, 1.0])
    write_checkpoint(second_path, [1.0, 1.0], [0.0, 3.0])

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

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(first_path), str(second_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.count("Checkpoint:") == 2
    assert result.stdout.count("resid_lambdas:") == 2
    assert result.stdout.count("x0_lambdas:") == 2
    assert "mean=0.66666667 std=2.4944383 max=4 min=-2 abs_max=4" in result.stdout
    assert "resid [-2.0, 0.0, 4.0]" in result.stdout
    assert "x0    [-1.0, 1.0]" in result.stdout