#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 18:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

export PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR=output
torchrun --standalone --nproc_per_node=2 -m scripts.base_train --save-every 8889 --model-tag "$SLURM_JOB_ID" "$@"
