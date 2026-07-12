import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_result_significance.py"
SPEC = importlib.util.spec_from_file_location("compare_result_significance", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_result_csv(path: Path, rows):
    path.write_text(
        "\n".join(["Task, Accuracy, Centered", *rows]) + "\n",
        encoding="utf-8",
    )


def write_list_file(path: Path, csv_paths):
    path.write_text("\n".join(str(csv_path) for csv_path in csv_paths) + "\n", encoding="utf-8")


def test_load_list_file_expands_seed_placeholder_and_csv_suffix(tmp_path: Path):
    list_path = tmp_path / "inputs.txt"
    list_path.write_text(
        "\n".join(
            [
                str(tmp_path / "aaa_seed_bbb"),
                str(tmp_path / "already_s99_named.csv"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    expanded = MODULE.load_list_file(list_path)

    assert expanded == [
        (tmp_path / "aaa_s24_bbb.csv").resolve(),
        (tmp_path / "aaa_s26_bbb.csv").resolve(),
        (tmp_path / "aaa_s28_bbb.csv").resolve(),
        (tmp_path / "already_s99_named.csv").resolve(),
    ]


def test_compare_lists_builds_expected_seed_and_benchmark_differences(tmp_path: Path):
    left_csv_1 = tmp_path / "left_1.csv"
    right_csv_1 = tmp_path / "right_1.csv"
    left_csv_2 = tmp_path / "left_2.csv"
    right_csv_2 = tmp_path / "right_2.csv"

    write_result_csv(
        left_csv_1,
        [
            "arc_easy, 0.100000, 0.100000",
            "boolq, 0.200000, 0.200000",
            "CORE, , 0.150000",
            "CORE (no boolq), , 0.100000",
        ],
    )
    write_result_csv(
        right_csv_1,
        [
            "arc_easy, 0.100000, 0.300000",
            "boolq, 0.200000, 0.400000",
            "CORE, , 0.350000",
            "CORE (no boolq), , 0.300000",
        ],
    )
    write_result_csv(
        left_csv_2,
        [
            "arc_easy, 0.100000, 0.500000",
            "boolq, 0.200000, 0.600000",
            "CORE, , 0.550000",
            "CORE (no boolq), , 0.500000",
        ],
    )
    write_result_csv(
        right_csv_2,
        [
            "arc_easy, 0.100000, 0.700000",
            "boolq, 0.200000, 0.800000",
            "CORE, , 0.750000",
            "CORE (no boolq), , 0.700000",
        ],
    )

    left_list = tmp_path / "left.txt"
    right_list = tmp_path / "right.txt"
    write_list_file(left_list, [left_csv_1, left_csv_2])
    write_list_file(right_list, [right_csv_1, right_csv_2])

    pair_diffs = MODULE.compare_lists(
        MODULE.load_list_file(left_list),
        MODULE.load_list_file(right_list),
    )

    assert len(pair_diffs) == 2
    assert pair_diffs[0].core_diff == pytest.approx(0.2)
    assert pair_diffs[1].core_diff == pytest.approx(0.2)
    assert pair_diffs[0].task_diffs == pytest.approx({"arc_easy": 0.2, "boolq": 0.2})
    assert pair_diffs[1].task_diffs == pytest.approx({"arc_easy": 0.2, "boolq": 0.2})

    benchmark_avg_diffs = MODULE.compute_benchmark_average_diffs(pair_diffs)
    assert benchmark_avg_diffs == pytest.approx({"arc_easy": 0.2, "boolq": 0.2})

    analyses = MODULE.run_analyses(pair_diffs)

    seed_summary = analyses["seed_paired_core"]
    assert seed_summary["count"] == 2
    assert seed_summary["wins"] == 2
    assert seed_summary["losses"] == 0
    assert seed_summary["ties"] == 0
    assert seed_summary["mean"] == pytest.approx(0.2)
    assert seed_summary["median"] == pytest.approx(0.2)
    assert seed_summary["sign_pvalue"] == pytest.approx(0.5)
    assert seed_summary["wilcoxon_stat"] == pytest.approx(0.0)
    assert seed_summary["wilcoxon_pvalue"] == pytest.approx(0.5)

    benchmark_summary = analyses["benchmark_paired"]
    assert benchmark_summary["count"] == 2
    assert benchmark_summary["wins"] == 2
    assert benchmark_summary["losses"] == 0
    assert benchmark_summary["ties"] == 0
    assert benchmark_summary["mean"] == pytest.approx(0.2)
    assert benchmark_summary["median"] == pytest.approx(0.2)
    assert benchmark_summary["sign_pvalue"] == pytest.approx(0.5)
    assert benchmark_summary["wilcoxon_stat"] == pytest.approx(0.0)
    assert benchmark_summary["wilcoxon_pvalue"] == pytest.approx(0.5)

    benchmark_no_boolq_summary = analyses["benchmark_paired_no_boolq"]
    assert benchmark_no_boolq_summary["count"] == 1
    assert benchmark_no_boolq_summary["wins"] == 1
    assert benchmark_no_boolq_summary["losses"] == 0
    assert benchmark_no_boolq_summary["ties"] == 0
    assert benchmark_no_boolq_summary["mean"] == pytest.approx(0.2)
    assert benchmark_no_boolq_summary["median"] == pytest.approx(0.2)
    assert benchmark_no_boolq_summary["sign_pvalue"] == pytest.approx(1.0)
    assert benchmark_no_boolq_summary["wilcoxon_stat"] == pytest.approx(0.0)
    assert benchmark_no_boolq_summary["wilcoxon_pvalue"] == pytest.approx(1.0)


def test_compare_lists_rejects_task_mismatch(tmp_path: Path):
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    write_result_csv(left_csv, ["arc_easy, 0.100000, 0.100000"])
    write_result_csv(right_csv, ["piqa, 0.200000, 0.200000"])

    with pytest.raises(ValueError, match="Task mismatch"):
        MODULE.build_pair_diff(left_csv, right_csv)