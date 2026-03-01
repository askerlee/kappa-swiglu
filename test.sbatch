#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 18:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Make this bash script immediately fail on errors.
# set -e
# Exit immediately if any command returns a non-zero status.
# set -u
# Treat using an unset variable as an error (and exit).
# set -o pipefail
# If you use a pipeline like a | b | c, the pipeline’s exit code becomes the first non-zero exit in the pipeline (instead of just c’s). This prevents hidden failures.
# Together they make scripts less “silently broken,” especially on clusters where you want the job to stop at the first real error.
set -euo pipefail

# Ensure we run from the submit directory (good practice for relative paths)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Job-local temp under the run directory
JOB_TMP="$(pwd)/tmp/${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"

# Force temp files (incl. wandb service socket) off /tmp
export TMPDIR="$JOB_TMP"
export TEMP="$JOB_TMP"
export TMP="$JOB_TMP"

# Put wandb files under the same job-local temp
export WANDB_DIR="$JOB_TMP/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_DATA_DIR="$WANDB_DIR/data"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

export PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR=output
export BASE_TRAIN_EXTRA_ARGS="$@"
echo "Test Running with extra args: $BASE_TRAIN_EXTRA_ARGS"
