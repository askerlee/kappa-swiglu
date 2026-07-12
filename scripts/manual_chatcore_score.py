import argparse
import sys

from nanochat.chatcore import compute_chatcore_metric, parse_accuracy_report


def read_input_text(input_file):
    if input_file is None:
        return sys.stdin.read()
    with open(input_file, "r", encoding="utf-8") as handle:
        return handle.read()


def main():
    parser = argparse.ArgumentParser(
        description="Compute ChatCORE metrics from pasted task accuracy lines."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Optional text file containing lines like 'ARC-Easy accuracy: 51.22%'",
    )
    args = parser.parse_args()

    results = parse_accuracy_report(read_input_text(args.input_file))
    if not results:
        raise SystemExit("No accuracy lines found in input.")

    metrics = compute_chatcore_metric(results)
    if not metrics:
        raise SystemExit("Could not compute any ChatCORE metrics from the provided tasks.")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()