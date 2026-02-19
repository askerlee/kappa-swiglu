#!/usr/bin/env fish
# keep_latest_ckpts.fish
#
# Recursively keeps only the latest checkpoint triplet per directory:
#   meta_<step>.json
#   model_<step>.pt
#   optim_<step>_rank0.pt
#
# Usage:
#   ./keep_latest_ckpts.fish /path/to/root
#   ./keep_latest_ckpts.fish /path/to/root --dry-run
#   ./keep_latest_ckpts.fish /path/to/root --keep 3

function usage
    echo "Usage: (basename (status filename)) <root_dir> [--dry-run|-n] [--keep|-k M]"
    exit 2
end

function process_ckpt_dir
    set -l ckpt_dir $argv[1]
    set -l dry_run $argv[2]
    set -l keep_count $argv[3]

    # Collect steps from meta_*.json (treat meta as the authoritative index)
    set -l meta_files
    for f in "$ckpt_dir"/meta_*.json
        if test -f "$f"
            set meta_files $meta_files $f
        end
    end

    if test (count $meta_files) -eq 0
        return 0
    end

    # Extract step numbers
    set -l steps

    for f in $meta_files
        set -l base (basename -- $f)
        # base like: meta_107771.json -> step=107771
        set -l step (string replace -r '^meta_(\d+)\.json$' '$1' -- $base)

        if string match -qr '^\d+$' -- $step
            set steps $steps $step
        end
    end

    if test (count $steps) -eq 0
        echo "Skipping $ckpt_dir: could not parse any step numbers"
        return 0
    end

    # Unique + sort descending (latest first)
    set -l sorted_steps (printf '%s\n' $steps | sort -nu -r)

    # Keep latest N steps
    set -l keep_steps
    set -l i 1
    for step in $sorted_steps
        if test $i -le $keep_count
            set keep_steps $keep_steps $step
        end
        set i (math $i + 1)
    end

    echo "Processing $ckpt_dir (keeping latest $keep_count step(s): "(string join ', ' $keep_steps)")"

    # Delete all other steps' triplets
    for step in $sorted_steps
        if contains -- $step $keep_steps
            continue
        end

        set -l meta "$ckpt_dir/meta_$step.json"
        set -l model "$ckpt_dir/model_$step.pt"
        set -l optim "$ckpt_dir/optim_"$step"_rank0.pt"

        for p in $meta $model $optim
            if test -e "$p"
                echo "rm -f -- $p"
                if test $dry_run -eq 0
                    rm -f -- "$p"
                end
            end
        end
    end
end

set -l dry_run 0
set -l keep_count 1
set -l root_dir ""

set -l i 1
while test $i -le (count $argv)
    set -l arg $argv[$i]

    switch $arg
        case --dry-run -n
            set dry_run 1
        case --keep -k
            set i (math $i + 1)
            if test $i -gt (count $argv)
                echo "Missing value for $arg"
                usage
            end
            set keep_count $argv[$i]
        case --keep='*'
            set keep_count (string replace -r '^--keep=' '' -- $arg)
        case '*'
            if test -z "$root_dir"
                set root_dir $arg
            else
                echo "Unknown extra argument: $arg"
                usage
            end
    end

    set i (math $i + 1)
end

if not string match -qr '^[1-9][0-9]*$' -- $keep_count
    echo "Invalid keep count: $keep_count (expected a positive integer)"
    usage
end

if test -z "$root_dir"
    usage
end

if not test -d "$root_dir"
    echo "Not a directory: $root_dir"
    exit 1
end

echo "Searching recursively under: $root_dir"

# Find checkpoint directories by locating meta_*.json files recursively.
set -l ckpt_dirs
for meta in (find "$root_dir" -type f -name 'meta_*.json' 2>/dev/null)
    set ckpt_dirs $ckpt_dirs (dirname -- "$meta")
end

# De-duplicate directories while preserving order.
set -l unique_ckpt_dirs
for dir in $ckpt_dirs
    if not contains -- "$dir" $unique_ckpt_dirs
        set unique_ckpt_dirs $unique_ckpt_dirs "$dir"
    end
end

if test (count $unique_ckpt_dirs) -eq 0
    echo "No checkpoint directories found (no meta_*.json files)."
    exit 0
end

for ckpt_dir in $unique_ckpt_dirs
    process_ckpt_dir "$ckpt_dir" "$dry_run" "$keep_count"
end

echo
if test $dry_run -eq 1
    echo "(dry-run) No files were deleted."
else
    echo "Cleanup complete."
end
