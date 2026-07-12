import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import binomtest, wilcoxon


METRIC_COLUMN = "Centered"
CORE_TASK = "CORE"
CORE_NO_BOOLQ_TASK = "CORE (no boolq)"
SUMMARY_TASKS = {CORE_TASK.lower(), CORE_NO_BOOLQ_TASK.lower()}
SEEDS = (24, 26, 28)


@dataclass(frozen=True)
class PairDiff:
    left_path: Path
    right_path: Path
    core_diff: float
    task_diffs: dict[str, float]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load two lists of result CSV files, compute paired metric differences, "
            "and run two-sided sign and Wilcoxon signed-rank tests. Differences are "
            "reported as right - left."
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


def load_list_file(path: Path):
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.extend(expand_list_entry(stripped))
    if not entries:
        raise ValueError(f"List file contains no CSV paths: {path}")
    return entries


def expand_list_entry(raw_path: str):
    normalized = raw_path if raw_path.endswith(".csv") else f"{raw_path}.csv"
    if "seed" not in normalized:
        return [Path(normalized).expanduser().resolve()]

    expanded_paths = []
    for seed in SEEDS:
        expanded_paths.append(Path(normalized.replace("seed", f"s{seed}")).expanduser().resolve())
    return expanded_paths


def load_metric_map(csv_path: Path, column: str):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        fieldnames = [(name or "").strip() for name in (reader.fieldnames or [])]
        if column not in fieldnames:
            raise ValueError(f"Column '{column}' not found in {csv_path}")

        metrics = {}
        summary_metrics = {}
        for raw_row in reader:
            row = normalize_row(raw_row)
            task = row.get("Task", "")
            if not task:
                continue

            raw_value = row.get(column, "")
            if raw_value == "":
                continue

            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in column '{column}' for task '{task}' in {csv_path}: {raw_value}"
                ) from exc

            if task.lower() in SUMMARY_TASKS:
                summary_metrics[task] = value
                continue

            metrics[task] = value

    if not metrics:
        raise ValueError(f"No numeric task values found in column '{column}' for {csv_path}")
    return metrics, summary_metrics


def build_pair_diff(left_path: Path, right_path: Path):
    left_metrics, left_summary = load_metric_map(left_path, METRIC_COLUMN)
    right_metrics, right_summary = load_metric_map(right_path, METRIC_COLUMN)

    left_tasks = list(left_metrics)
    right_tasks = list(right_metrics)
    if set(left_tasks) != set(right_tasks):
        missing_left = sorted(set(right_tasks) - set(left_tasks))
        missing_right = sorted(set(left_tasks) - set(right_tasks))
        raise ValueError(
            f"Task mismatch between {left_path} and {right_path}. "
            f"Missing from left: {missing_left or 'none'}. Missing from right: {missing_right or 'none'}."
        )

    if CORE_TASK not in left_summary or CORE_TASK not in right_summary:
        raise ValueError(f"Missing '{CORE_TASK}' row in {left_path} or {right_path}")

    task_diffs = {task: right_metrics[task] - left_metrics[task] for task in left_tasks}
    return PairDiff(
        left_path=left_path,
        right_path=right_path,
        core_diff=right_summary[CORE_TASK] - left_summary[CORE_TASK],
        task_diffs=task_diffs,
    )


def compute_benchmark_average_diffs(pair_diffs: list[PairDiff], exclude_tasks: set[str] | None = None):
    if not pair_diffs:
        raise ValueError("At least one paired result is required")

    exclude_tasks = {task.lower() for task in (exclude_tasks or set())}
    first_tasks = set(pair_diffs[0].task_diffs)
    for pair_diff in pair_diffs[1:]:
        if set(pair_diff.task_diffs) != first_tasks:
            raise ValueError("Task sets differ across matched file pairs")

    averaged_diffs = {}
    for task in sorted(first_tasks):
        if task.lower() in exclude_tasks:
            continue
        averaged_diffs[task] = sum(pair_diff.task_diffs[task] for pair_diff in pair_diffs) / len(pair_diffs)
    return averaged_diffs


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


def compare_lists(left_paths: list[Path], right_paths: list[Path]):
    if len(left_paths) != len(right_paths):
        raise ValueError(
            f"List lengths differ: {len(left_paths)} paths in left list, {len(right_paths)} in right list"
        )

    pair_diffs = []
    for left_path, right_path in zip(left_paths, right_paths):
        pair_diffs.append(build_pair_diff(left_path, right_path))
    return pair_diffs


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


def run_analyses(pair_diffs: list[PairDiff]):
    seed_paired_core_diffs = [pair_diff.core_diff for pair_diff in pair_diffs]
    benchmark_avg_diffs = compute_benchmark_average_diffs(pair_diffs)
    benchmark_avg_diffs_no_boolq = compute_benchmark_average_diffs(pair_diffs, exclude_tasks={"boolq"})

    return {
        "seed_paired_core": summarize_diffs(seed_paired_core_diffs),
        "benchmark_paired": summarize_diffs(list(benchmark_avg_diffs.values())),
        "benchmark_paired_no_boolq": summarize_diffs(list(benchmark_avg_diffs_no_boolq.values())),
    }


def main():
    args = parse_args()
    left_list = Path(args.left_list).expanduser().resolve()
    right_list = Path(args.right_list).expanduser().resolve()
    left_paths = load_list_file(left_list)
    right_paths = load_list_file(right_list)
    pair_diffs = compare_lists(left_paths, right_paths)

    analyses = run_analyses(pair_diffs)

    print(f"Compared column '{METRIC_COLUMN}' as right - left across {len(pair_diffs)} matched file pair(s).")
    print_summary("Seed-paired CORE", analyses["seed_paired_core"])
    print_summary("Benchmark-paired", analyses["benchmark_paired"])
    print_summary("Benchmark-paired (no BoolQ)", analyses["benchmark_paired_no_boolq"])

    if args.no_per_file:
        return

    for pair_diff in pair_diffs:
        label = f"{pair_diff.left_path.name} vs {pair_diff.right_path.name}"
        print_summary(label, summarize_diffs(list(pair_diff.task_diffs.values())))


if __name__ == "__main__":
    main()