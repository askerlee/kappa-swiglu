#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 18:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

export PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR=output
export BASE_TRAIN_EXTRA_ARGS="$@"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train --save-every 17778 --model-tag "$SLURM_JOB_ID" "$@"
