import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import binomtest, wilcoxon


ACCURACY_COLUMN = "Accuracy"
CHATCORE_TASKS = (
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
    "SpellingBee",
)
CHATCORE_METRIC_TASK = "ChatCORE metric"
CHATCORE_METRIC_WITHOUT_SPELLINGBEE_TASK = "ChatCORE metric (without SpellingBee)"
SUMMARY_TASKS = {
    CHATCORE_METRIC_TASK.lower(),
    CHATCORE_METRIC_WITHOUT_SPELLINGBEE_TASK.lower(),
}
IFEVAL_TASK = "IFEval"
SEEDS = (24, 26, 28)


@dataclass(frozen=True)
class PairResult:
    left_path: Path
    right_path: Path
    chatcore_diffs: dict[str, float]
    ifeval_diff: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load two lists of chat-SFT result CSV files, compute paired ChatCORE task "
            "differences, and report two-sided sign and Wilcoxon signed-rank tests. "
            "Differences are reported as right - left."
        )
    )
    parser.add_argument("left_list", help="Text file listing the first set of CSV paths, one per line.")
    parser.add_argument("right_list", help="Text file listing the second set of CSV paths, one per line.")
    parser.add_argument(
        "--no-per-file",
        action="store_true",
        help="Suppress the per-file breakdown and print only the pooled summary.",
    )
    return parser.parse_args()


def normalize_row(row):
    return {
        (key or "").strip(): (value or "").strip()
        for key, value in row.items()
    }


def expand_list_entry(raw_path: str):
    normalized = raw_path if raw_path.endswith(".csv") else f"{raw_path}.csv"
    if "seed" not in normalized:
        return [Path(normalized).expanduser().resolve()]

    expanded_paths = []
    for seed in SEEDS:
        expanded_paths.append(Path(normalized.replace("seed", f"s{seed}")).expanduser().resolve())
    return expanded_paths


def load_list_file(path: Path):
    groups = load_list_groups(path)
    return [entry for group in groups for entry in group]


def load_list_groups(path: Path):
    groups = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        groups.append(expand_list_entry(stripped))
    if not groups:
        raise ValueError(f"List file contains no CSV paths: {path}")
    return groups


def load_accuracy_map(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        fieldnames = [(name or "").strip() for name in (reader.fieldnames or [])]
        if ACCURACY_COLUMN not in fieldnames:
            raise ValueError(f"Column '{ACCURACY_COLUMN}' not found in {csv_path}")

        metrics = {}
        summary_metrics = {}
        for raw_row in reader:
            row = normalize_row(raw_row)
            task = row.get("Task", "")
            if not task:
                continue

            raw_value = row.get(ACCURACY_COLUMN, "")
            if raw_value == "":
                continue

            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in column '{ACCURACY_COLUMN}' for task '{task}' in {csv_path}: {raw_value}"
                ) from exc

            if task.lower() in SUMMARY_TASKS:
                summary_metrics[task] = value
                continue

            metrics[task] = value

    if not metrics:
        raise ValueError(f"No numeric task values found in column '{ACCURACY_COLUMN}' for {csv_path}")
    return metrics, summary_metrics


def build_pair_result(left_path: Path, right_path: Path):
    left_metrics, _ = load_accuracy_map(left_path)
    right_metrics, _ = load_accuracy_map(right_path)

    left_tasks = set(left_metrics)
    right_tasks = set(right_metrics)
    if left_tasks != right_tasks:
        missing_left = sorted(right_tasks - left_tasks)
        missing_right = sorted(left_tasks - right_tasks)
        raise ValueError(
            f"Task mismatch between {left_path} and {right_path}. "
            f"Missing from left: {missing_left or 'none'}. Missing from right: {missing_right or 'none'}."
        )

    missing_chatcore = [task for task in CHATCORE_TASKS if task not in left_metrics or task not in right_metrics]
    if missing_chatcore:
        raise ValueError(f"Missing ChatCORE task(s) {missing_chatcore} in {left_path} or {right_path}")
    if IFEVAL_TASK not in left_metrics or IFEVAL_TASK not in right_metrics:
        raise ValueError(f"Missing '{IFEVAL_TASK}' task in {left_path} or {right_path}")

    chatcore_diffs = {task: right_metrics[task] - left_metrics[task] for task in CHATCORE_TASKS}
    return PairResult(
        left_path=left_path,
        right_path=right_path,
        chatcore_diffs=chatcore_diffs,
        ifeval_diff=right_metrics[IFEVAL_TASK] - left_metrics[IFEVAL_TASK],
    )


def compare_lists(left_paths: list[Path], right_paths: list[Path]):
    if len(left_paths) != len(right_paths):
        raise ValueError(
            f"List lengths differ: {len(left_paths)} paths in left list, {len(right_paths)} in right list"
        )

    return [build_pair_result(left_path, right_path) for left_path, right_path in zip(left_paths, right_paths)]


def compare_list_groups(left_groups: list[list[Path]], right_groups: list[list[Path]]):
    if len(left_groups) != len(right_groups):
        raise ValueError(
            f"Configuration counts differ: {len(left_groups)} in left list, {len(right_groups)} in right list"
        )
    return [compare_lists(left_paths, right_paths) for left_paths, right_paths in zip(left_groups, right_groups)]


def compute_chatcore_average_diffs(pair_results: list[PairResult]):
    if not pair_results:
        raise ValueError("At least one paired result is required")

    first_tasks = set(pair_results[0].chatcore_diffs)
    for pair_result in pair_results[1:]:
        if set(pair_result.chatcore_diffs) != first_tasks:
            raise ValueError("ChatCORE task sets differ across matched file pairs")

    averaged_diffs = {}
    for task in CHATCORE_TASKS:
        averaged_diffs[task] = sum(pair_result.chatcore_diffs[task] for pair_result in pair_results) / len(pair_results)
    return averaged_diffs


def compute_seed_paired_chatcore_average_diffs(pair_results: list[PairResult]):
    if not pair_results:
        raise ValueError("At least one paired result is required")

    return [
        sum(pair_result.chatcore_diffs[task] for task in CHATCORE_TASKS) / len(CHATCORE_TASKS)
        for pair_result in pair_results
    ]


def compute_seed_paired_combined_average_diffs(pair_results: list[PairResult]):
    chatcore_diffs = compute_seed_paired_chatcore_average_diffs(pair_results)
    return [
        (chatcore_diff + pair_result.ifeval_diff) / 2.0
        for chatcore_diff, pair_result in zip(chatcore_diffs, pair_results)
    ]


def compute_task_config_diffs(pair_result_groups: list[list[PairResult]]):
    task_config_diffs = []
    for pair_results in pair_result_groups:
        task_config_diffs.extend(compute_chatcore_average_diffs(pair_results).values())
    return task_config_diffs


def summarize_diffs(diffs: list[float]):
    if not diffs:
        raise ValueError("At least one paired difference is required")

    wins = sum(diff > 0 for diff in diffs)
    losses = sum(diff < 0 for diff in diffs)
    ties = len(diffs) - wins - losses
    sorted_diffs = sorted(diffs)
    mid = len(sorted_diffs) // 2
    if len(sorted_diffs) % 2 == 1:
        median = sorted_diffs[mid]
    else:
        median = (sorted_diffs[mid - 1] + sorted_diffs[mid]) / 2.0

    nonzero = wins + losses
    sign_pvalue = None
    wilcoxon_stat = None
    wilcoxon_pvalue = None
    if nonzero:
        sign_pvalue = binomtest(wins, n=nonzero, p=0.5, alternative="two-sided").pvalue
        wilcoxon_result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
        wilcoxon_stat = float(wilcoxon_result.statistic)
        wilcoxon_pvalue = float(wilcoxon_result.pvalue)

    return {
        "count": len(diffs),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean": sum(diffs) / len(diffs),
        "median": median,
        "sign_pvalue": sign_pvalue,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_pvalue,
    }


def format_float(value):
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def summarize_direction(summary):
    if summary["wins"] == 0 and summary["losses"] == 0:
        return "no non-zero differences"
    if summary["mean"] > 0:
        return "right > left"
    if summary["mean"] < 0:
        return "left > right"
    return "no average difference"


def print_summary(label: str, summary):
    print(label)
    print(f"  interpretation={summarize_direction(summary)}")
    print(f"  count={summary['count']}")
    print(f"  wins={summary['wins']} losses={summary['losses']} ties={summary['ties']}")
    print(f"  mean_diff={summary['mean']:.6f}")
    print(f"  median_diff={summary['median']:.6f}")
    print(f"  sign_test_pvalue={format_float(summary['sign_pvalue'])}")
    print(f"  wilcoxon_statistic={format_float(summary['wilcoxon_stat'])}")
    print(f"  wilcoxon_pvalue={format_float(summary['wilcoxon_pvalue'])}")


def run_analyses(pair_results: list[PairResult], pair_result_groups: list[list[PairResult]] | None = None):
    chatcore_avg_diffs = compute_chatcore_average_diffs(pair_results)
    combined_seed_paired_diffs = compute_seed_paired_combined_average_diffs(pair_results)
    ifeval_diffs = [pair_result.ifeval_diff for pair_result in pair_results]

    analyses = {
        "chatcore_benchmark_paired": summarize_diffs(list(chatcore_avg_diffs.values())),
        "chatcore_average_diffs": chatcore_avg_diffs,
        "chatcore_ifeval_average_seed_paired": summarize_diffs(combined_seed_paired_diffs),
        "ifeval_seed_paired": summarize_diffs(ifeval_diffs),
    }
    if pair_result_groups is not None:
        analyses["task_config_robustness"] = summarize_diffs(compute_task_config_diffs(pair_result_groups))
    return analyses


def main():
    args = parse_args()
    left_list = Path(args.left_list).expanduser().resolve()
    right_list = Path(args.right_list).expanduser().resolve()
    left_groups = load_list_groups(left_list)
    right_groups = load_list_groups(right_list)
    pair_result_groups = compare_list_groups(left_groups, right_groups)
    pair_results = [pair_result for group in pair_result_groups for pair_result in group]
    analyses = run_analyses(pair_results, pair_result_groups=pair_result_groups)

    print(
        f"Compared column '{ACCURACY_COLUMN}' as right - left across {len(pair_results)} matched file pair(s)."
    )
    print_summary("ChatCORE benchmark-paired", analyses["chatcore_benchmark_paired"])
    print_summary(
        "Average ChatCORE and IFEval score seed-paired",
        analyses["chatcore_ifeval_average_seed_paired"],
    )
    print("Task-configuration robustness (supplementary; cells are correlated, so p-values are descriptive)")
    print_summary("  6-task task-configuration cells", analyses["task_config_robustness"])
    print_summary("IFEval seed-paired", analyses["ifeval_seed_paired"])

    if args.no_per_file:
        return

    for pair_result in pair_results:
        label = f"{pair_result.left_path.name} vs {pair_result.right_path.name}"
        print_summary(label, summarize_diffs(list(pair_result.chatcore_diffs.values())))


if __name__ == "__main__":
    main()