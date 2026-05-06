#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 03:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
export PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR=output
python -m scripts.chat_sft --model-tag "$SLURM_JOB_ID" "$@"
