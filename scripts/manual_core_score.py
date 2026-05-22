import argparse
import csv
import sys


def _normalize_row(row):
    return {
        (key or "").strip(): (value or "").strip()
        for key, value in row.items()
    }


def load_rows(handle):
    reader = csv.DictReader(handle, skipinitialspace=True)
    rows = [_normalize_row(row) for row in reader]
    if not rows:
        raise ValueError("CSV contains no data rows")
    return rows


def compute_scores(rows):
    centered_scores = []
    centered_scores_no_boolq = []

    for row in rows:
        task = row["Task"].strip().lower()
        centered = float(row["Centered"])
        centered_scores.append(centered)
        if task != "boolq":
            centered_scores_no_boolq.append(centered)

    if not centered_scores:
        raise ValueError("No centered scores found")

    core = sum(centered_scores) / len(centered_scores)
    core_no_boolq = sum(centered_scores_no_boolq) / len(centered_scores_no_boolq)
    return core, core_no_boolq


def main():
    parser = argparse.ArgumentParser(
        description="Compute CORE and CORE (no boolq) from a CSV of centered task scores."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Path to CSV with Task and Centered columns. Reads stdin when omitted.",
    )
    args = parser.parse_args()

    if args.csv_path:
        with open(args.csv_path, "r", encoding="utf-8", newline="") as handle:
            rows = load_rows(handle)
    else:
        rows = load_rows(sys.stdin)

    core, core_no_boolq = compute_scores(rows)
    print(f"CORE: {core:.6f}")
    print(f"CORE (no boolq): {core_no_boolq:.6f}")


if __name__ == "__main__":
    main()