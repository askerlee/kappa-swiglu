"""Preview or delete the oldest runs or largest run files in a W&B project.

Usage: python -m scripts.delete_wandb_runs ENTITY/PROJECT COUNT [--execute]
    python -m scripts.delete_wandb_runs ENTITY/PROJECT MODE COUNT [--execute]
Modes: oldest-runs (default), largest-files
"""

import argparse
import heapq
from itertools import islice


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("count must be a positive integer")
    return number


def largest_files(api, project, count):
    candidates = (
        (file.size, run.id, file.name, run, file)
        for run in api.runs(project)
        for file in run.files()
        if file.size is not None
    )
    return heapq.nlargest(count, candidates, key=lambda item: (item[0], item[1], item[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="W&B project path in ENTITY/PROJECT form")
    parser.add_argument("mode_or_count", help="count, or mode when followed by count")
    parser.add_argument("count", nargs="?", type=positive_int, help="number of runs or files to select")
    parser.add_argument("--execute", action="store_true", help="delete selected items (default: dry run)")
    args = parser.parse_args(argv)
    if args.count is None:
        mode = "oldest-runs"
        try:
            args.count = positive_int(args.mode_or_count)
        except (ValueError, argparse.ArgumentTypeError):
            parser.error("count must be a positive integer")
    else:
        mode = args.mode_or_count
        if mode not in ("oldest-runs", "largest-files"):
            parser.error("mode must be oldest-runs or largest-files")
    if len(args.project.split("/")) != 2 or not all(args.project.split("/")):
        parser.error("project must be in ENTITY/PROJECT form")

    import wandb

    api = wandb.Api()
    if mode == "oldest-runs":
        runs = list(islice(api.runs(args.project, order="+created_at"), args.count))
        print(f"Selected {len(runs)} oldest run(s) in {args.project}:")
        for run in runs:
            print(f"  {run.created_at}  {run.id}  {run.name}")
        selected = [(run.id, run) for run in runs]
    else:
        files = largest_files(api, args.project, args.count)
        print(f"Selected {len(files)} largest run file(s) in {args.project}:")
        for size, run_id, name, run, _ in files:
            print(f"  {size} bytes  {run.created_at}  {run_id}  {run.name}  {name}")
        selected = [(f"{run_id}/{name}", file) for _, run_id, name, _, file in files]

    if not args.execute:
        print("Dry run. Pass --execute to delete these items.")
        return

    for label, item in selected:
        item.delete()
        print(f"Deleted {label}")


if __name__ == "__main__":
    main()