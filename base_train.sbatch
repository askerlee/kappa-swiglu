#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 18:00:00
#SBATCH --signal=USR1@300
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

export PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR=output
python -m scripts.base_train --save-every 2000 --wandb-api-key-file ~/.wandb_key_sh --continue-to-chat-sft --model-tag "$SLURM_JOB_ID" "$@"
