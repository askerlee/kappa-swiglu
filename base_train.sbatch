#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 18:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
export WANDB_DIR="wandb/$SLURM_JOB_ID"
export WANDB_CACHE_DIR="$WANDB_DIR/wandb_cache"
export WANDB_DATA_DIR="$WANDB_DIR/wandb_data"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

export PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR=output
export BASE_TRAIN_EXTRA_ARGS="$@"
python -m scripts.base_train --save-every 8889 --model-tag "$SLURM_JOB_ID" "$@"
