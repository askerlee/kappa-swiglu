import argparse
import csv
import re
from pathlib import Path


BASE_RESULT_PATTERN = re.compile(r"^(?P<method>.+)-s\d+_base_[^/]+\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan a directory for *_base_*.csv result files, group files from the same "
            "method across seeds, and write averaged CSVs."
        )
    )
    parser.add_argument("directory", help="Directory containing result CSV files.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories recursively.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum number of files required to emit an averaged CSV.",
    )
    parser.add_argument(
        "--suffix",
        default="_base_avg.csv",
        help="Suffix used for averaged output files.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where averaged CSV files will be written. Defaults to the current directory.",
    )
    return parser.parse_args()


def iter_candidate_files(directory: Path, recursive: bool):
    if recursive:
        yield from directory.rglob("*.csv")
        return
    yield from directory.glob("*.csv")


def collect_method_files(directory: Path, recursive: bool = False):
    groups = {}
    for path in sorted(iter_candidate_files(directory, recursive)):
        match = BASE_RESULT_PATTERN.match(path.name)
        if not match:
            continue
        method = match.group("method")
        groups.setdefault(method, []).append(path)
    return groups


def load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        rows = [[cell.strip() for cell in row] for row in reader]
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    width = len(rows[0])
    for index, row in enumerate(rows[1:], start=2):
        if len(row) != width:
            raise ValueError(
                f"Row {index} in {path} has {len(row)} columns; expected {width}"
            )
    return rows[0], rows[1:]


def _parse_float(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def average_group(paths):
    first_header, first_rows = load_csv_rows(paths[0])
    averaged_rows = []

    loaded = [(first_header, first_rows)]
    for path in paths[1:]:
        header, rows = load_csv_rows(path)
        if header != first_header:
            raise ValueError(f"Header mismatch between {paths[0]} and {path}")
        if len(rows) != len(first_rows):
            raise ValueError(f"Row count mismatch between {paths[0]} and {path}")
        loaded.append((header, rows))

    for row_index in range(len(first_rows)):
        row_values = [rows[row_index] for _, rows in loaded]
        label = row_values[0][0]
        if any(row[0] != label for row in row_values[1:]):
            raise ValueError(
                f"Row label mismatch at data row {row_index + 1} for group rooted at {paths[0]}"
            )

        averaged_row = [label]
        for column_index in range(1, len(first_header)):
            values = [row[column_index] for row in row_values]
            if all(value == "" for value in values):
                averaged_row.append("")
                continue

            numeric_values = [_parse_float(value) for value in values]
            if all(value is not None for value in numeric_values):
                averaged = sum(numeric_values) / len(numeric_values)
                averaged_row.append(f"{averaged:.6f}")
                continue

            if len(set(values)) == 1:
                averaged_row.append(values[0])
                continue

            raise ValueError(
                "Non-numeric column values do not match for "
                f"row '{label}' column '{first_header[column_index]}'"
            )

        averaged_rows.append(averaged_row)

    return first_header, averaged_rows


def format_table(header, rows):
    widths = [len(cell) for cell in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    lines = [", ".join(f"{cell:<{widths[index]}}" for index, cell in enumerate(header))]
    for row in rows:
        lines.append(
            ", ".join(f"{cell:<{widths[index]}}" for index, cell in enumerate(row))
        )
    return "\n".join(lines) + "\n"


def write_average_files(output_dir: Path, groups, min_count: int = 2, suffix: str = "_base_avg.csv"):
    written_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, paths in sorted(groups.items()):
        if len(paths) < min_count:
            continue
        header, rows = average_group(paths)
        output_path = output_dir / f"{method}{suffix}"
        output_path.write_text(format_table(header, rows), encoding="utf-8")
        written_paths.append(output_path)
    return written_paths


def main():
    args = parse_args()
    directory = Path(args.directory).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    groups = collect_method_files(directory, recursive=args.recursive)
    written_paths = write_average_files(
        output_dir=output_dir,
        groups=groups,
        min_count=args.min_count,
        suffix=args.suffix,
    )
    if not written_paths:
        print("No averaged CSV files were written.")
        return

    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()