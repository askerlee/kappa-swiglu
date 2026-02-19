#!/usr/bin/env fish
# keep_latest_ckpts.fish
#
# Keeps only the latest checkpoint triplet:
#   meta_<step>.json
#   model_<step>.pt
#   optim_<step>_rank0.pt
#
# Usage:
#   ./keep_latest_triplet_ckpt.fish /path/to/ckpts
#   ./keep_latest_triplet_ckpt.fish /path/to/ckpts --dry-run

function usage
    echo "Usage: (basename (status filename)) <ckpt_dir> [--dry-run|-n]"
    exit 2
end

set -l dry_run 0
set -l ckpt_dir ""

for arg in $argv
    switch $arg
        case --dry-run -n
            set dry_run 1
        case '*'
            if test -z "$ckpt_dir"
                set ckpt_dir $arg
            else
                echo "Unknown extra argument: $arg"
                usage
            end
    end
end

if test -z "$ckpt_dir"
    usage
end

if not test -d "$ckpt_dir"
    echo "Not a directory: $ckpt_dir"
    exit 1
end

# Collect steps from meta_*.json (treat meta as the authoritative index)
set -l meta_files
for f in "$ckpt_dir"/meta_*.json
    if test -f "$f"
        set meta_files $meta_files $f
    end
end

if test (count $meta_files) -eq 0
    echo "No meta_*.json found in $ckpt_dir"
    exit 0
end

# Extract step numbers and find max
set -l max_step -1
set -l steps

for f in $meta_files
    set -l base (basename -- $f)
    # base like: meta_107771.json  -> step=107771
    set -l step (string replace -r '^meta_(\d+)\.json$' '$1' -- $base)

    if string match -qr '^\d+$' -- $step
        set steps $steps $step
        if test $step -gt $max_step
            set max_step $step
        end
    end
end

if test $max_step -lt 0
    echo "Could not parse any step numbers from meta_*.json"
    exit 1
end

echo "Keeping latest step: $max_step"
echo

# Delete all other steps' triplets
# Note: we only iterate over steps that have meta_*.json.
for step in $steps
    if test $step -eq $max_step
        continue
    end

    set -l meta "$ckpt_dir/meta_$step.json"
    set -l model "$ckpt_dir/model_$step.pt"
    set -l optim "$ckpt_dir/optim_{$step}_rank0.pt"
    # fish doesn't expand {} inside quotes, so build it explicitly:
    set optim "$ckpt_dir/optim_"$step"_rank0.pt"

    for p in $meta $model $optim
        if test -e "$p"
            echo "rm -f -- $p"
            if test $dry_run -eq 0
                rm -f -- $p
            end
        else
            # Optional: uncomment if you want to see missing pieces
            # echo "missing: $p"
        end
    end
end

echo
if test $dry_run -eq 1
    echo "(dry-run) No files were deleted."
end
