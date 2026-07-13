import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_chat_sft_significance.py"
SPEC = importlib.util.spec_from_file_location("compare_chat_sft_significance", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_result_csv(path: Path, rows):
    path.write_text(
        "\n".join(["Task, Accuracy", *rows]) + "\n",
        encoding="utf-8",
    )


def write_list_file(path: Path, entries):
    path.write_text("\n".join(str(entry) for entry in entries) + "\n", encoding="utf-8")


def test_load_list_file_expands_seed_placeholder_and_csv_suffix(tmp_path: Path):
    list_path = tmp_path / "inputs.txt"
    list_path.write_text(str(tmp_path / "aaa_seed_bbb") + "\n", encoding="utf-8")

    expanded = MODULE.load_list_file(list_path)

    assert expanded == [
        (tmp_path / "aaa_s24_bbb.csv").resolve(),
        (tmp_path / "aaa_s26_bbb.csv").resolve(),
        (tmp_path / "aaa_s28_bbb.csv").resolve(),
    ]


def test_chatcore_significance_and_ifeval_paired_differences(tmp_path: Path):
    left_csv_1 = tmp_path / "left_1.csv"
    right_csv_1 = tmp_path / "right_1.csv"
    left_csv_2 = tmp_path / "left_2.csv"
    right_csv_2 = tmp_path / "right_2.csv"

    write_result_csv(
        left_csv_1,
        [
            "ARC-Easy, 0.10",
            "ARC-Challenge, 0.20",
            "MMLU, 0.30",
            "GSM8K, 0.40",
            "HumanEval, 0.50",
            "SpellingBee, 0.60",
            "IFEval, 0.277264",
            "ChatCORE metric, 0.35",
            "ChatCORE metric (without SpellingBee), 0.30",
        ],
    )
    write_result_csv(
        right_csv_1,
        [
            "ARC-Easy, 0.20",
            "ARC-Challenge, 0.30",
            "MMLU, 0.40",
            "GSM8K, 0.50",
            "HumanEval, 0.60",
            "SpellingBee, 0.70",
            "IFEval, 0.300000",
            "ChatCORE metric, 0.45",
            "ChatCORE metric (without SpellingBee), 0.40",
        ],
    )
    write_result_csv(
        left_csv_2,
        [
            "ARC-Easy, 0.15",
            "ARC-Challenge, 0.25",
            "MMLU, 0.35",
            "GSM8K, 0.45",
            "HumanEval, 0.55",
            "SpellingBee, 0.65",
            "IFEval, 0.277264",
            "ChatCORE metric, 0.40",
            "ChatCORE metric (without SpellingBee), 0.35",
        ],
    )
    write_result_csv(
        right_csv_2,
        [
            "ARC-Easy, 0.25",
            "ARC-Challenge, 0.35",
            "MMLU, 0.45",
            "GSM8K, 0.55",
            "HumanEval, 0.65",
            "SpellingBee, 0.75",
            "IFEval, 0.300000",
            "ChatCORE metric, 0.50",
            "ChatCORE metric (without SpellingBee), 0.45",
        ],
    )

    pair_results = MODULE.compare_lists(
        [left_csv_1, left_csv_2],
        [right_csv_1, right_csv_2],
    )

    assert len(pair_results) == 2
    assert pair_results[0].chatcore_diffs == pytest.approx({task: 0.1 for task in MODULE.CHATCORE_TASKS})
    assert pair_results[1].chatcore_diffs == pytest.approx({task: 0.1 for task in MODULE.CHATCORE_TASKS})
    assert pair_results[0].ifeval_diff == pytest.approx(0.022736)
    assert pair_results[1].ifeval_diff == pytest.approx(0.022736)

    analyses = MODULE.run_analyses(pair_results)
    assert analyses["chatcore_average_diffs"] == pytest.approx({task: 0.1 for task in MODULE.CHATCORE_TASKS})

    chatcore_summary = analyses["chatcore_benchmark_paired"]
    assert chatcore_summary["count"] == 6
    assert chatcore_summary["wins"] == 6
    assert chatcore_summary["losses"] == 0
    assert chatcore_summary["ties"] == 0
    assert chatcore_summary["mean"] == pytest.approx(0.1)
    assert chatcore_summary["median"] == pytest.approx(0.1)
    assert chatcore_summary["sign_pvalue"] == pytest.approx(0.03125)
    assert chatcore_summary["wilcoxon_stat"] == pytest.approx(0.0)
    assert chatcore_summary["wilcoxon_pvalue"] == pytest.approx(0.03125)

    combined_seed_summary = analyses["chatcore_ifeval_average_seed_paired"]
    assert combined_seed_summary["count"] == 2
    assert combined_seed_summary["wins"] == 2
    assert combined_seed_summary["losses"] == 0
    assert combined_seed_summary["ties"] == 0
    assert combined_seed_summary["mean"] == pytest.approx(0.061368)
    assert combined_seed_summary["median"] == pytest.approx(0.061368)
    assert combined_seed_summary["sign_pvalue"] == pytest.approx(0.5)
    assert combined_seed_summary["wilcoxon_stat"] == pytest.approx(0.0)
    assert combined_seed_summary["wilcoxon_pvalue"] == pytest.approx(0.5)

    ifeval_summary = analyses["ifeval_seed_paired"]
    assert ifeval_summary["count"] == 2
    assert ifeval_summary["wins"] == 2
    assert ifeval_summary["losses"] == 0
    assert ifeval_summary["ties"] == 0
    assert ifeval_summary["mean"] == pytest.approx(0.022736)
    assert ifeval_summary["median"] == pytest.approx(0.022736)
    assert ifeval_summary["sign_pvalue"] == pytest.approx(0.5)
    assert ifeval_summary["wilcoxon_stat"] == pytest.approx(0.0)
    assert ifeval_summary["wilcoxon_pvalue"] == pytest.approx(0.5)


def test_task_config_robustness_averages_seeds_before_counting_cells(tmp_path: Path):
    pair_result_groups = []
    for config_index in range(4):
        group = []
        for seed_index in range(3):
            group.append(
                MODULE.PairResult(
                    left_path=tmp_path / f"left_{config_index}_{seed_index}.csv",
                    right_path=tmp_path / f"right_{config_index}_{seed_index}.csv",
                    chatcore_diffs={task: 0.01 * (config_index + 1) for task in MODULE.CHATCORE_TASKS},
                    ifeval_diff=0.01,
                )
            )
        pair_result_groups.append(group)

    pair_results = [pair_result for group in pair_result_groups for pair_result in group]
    analyses = MODULE.run_analyses(pair_results, pair_result_groups=pair_result_groups)

    robustness_summary = analyses["task_config_robustness"]
    assert robustness_summary["count"] == 24
    assert robustness_summary["wins"] == 24
    assert robustness_summary["losses"] == 0
    assert robustness_summary["ties"] == 0
    assert robustness_summary["mean"] == pytest.approx(0.025)
    assert robustness_summary["median"] == pytest.approx(0.025)