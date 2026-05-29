# nanochat experiment README

This repository contains a local training workflow for running a kappa-SwiGLU base-training on nanochat. The commands below set up the cache layout, install the extra dependency, prepare the dataset shards, log in to Weights & Biases, and launch three 2-GPU training jobs in tmux.

For the broader upstream project documentation, see `nanochat-README.md`.

## Prerequisites

- Python 3.10 or newer
- `git`, `tmux`, and `fish`
- A machine with at least 6 visible GPUs
- A writable `/DataDrive`
- A Weights & Biases account for experiment tracking

The launch loop below uses fish shell syntax. Run it with `fish`, or translate it to your preferred shell.

## 1. Prepare the cache directory

Create the shared data directory and point nanochat's cache at it:

```bash
mkdir /DataDrive/nanochat-data && ln -s /DataDrive/nanochat-data ~/.cache/nanochat
```

## 2. Install Python build tools and kappa-swiglu

Upgrade the packaging toolchain:

```bash
python -m pip install -U pip setuptools wheel build packaging
```

Clone and install the kappa-SwiGLU dependency in editable mode:

```bash
git clone https://github.com/askerlee/kappa-swiglu
cd kappa-swiglu && pip install -e .
```

Return to the root of this repository before continuing.

## 3. Copy the tokenizer assets into the nanochat cache

```bash
mkdir -p ~/.cache/nanochat/ && cp -r tokenizer/ ~/.cache/nanochat/
```

## 4. Download and prepare dataset shards

This command fetches the shard ranges `1-300` and `1750-`:

```bash
python -m nanochat.dataset --shards 1-300,1750-
```

## 5. Authenticate with Weights & Biases

```bash
wandb login
```

## 6. Launch the three training runs

The following fish script launches three detached tmux sessions. Each run uses two GPUs, a different seed, and writes pane output to a matching log file.

```fish
set seeds 24 26 28
set devices "0,1" "2,3" "4,5"

for i in 1 2 3
    set seed $seeds[$i]
    set cuda $devices[$i]
    set sess exp64-d8-kappa-lin-au-s$seed

    tmux new -d -s $sess "CUDA_VISIBLE_DEVICES=$cuda torchrun --standalone --nproc_per_node=2 -m scripts.base_train --delete-old-ckpts-before-save --model-tag exp64-d8-kappa-lin-au --n-exp 64 --depth 8 --device-batch-size 32 --use-kappa-swiglu --constant-kappa-dense-layers --kappa-ema-rms-reg --seed $seed"

    tmux pipe-pane -t $sess:0.0 -o "cat >> $sess.log"
end
```

Each run starts a tmux session with one of these names:

- `exp64-d8-kappa-lin-au-s24`
- `exp64-d8-kappa-lin-au-s26`
- `exp64-d8-kappa-lin-au-s28`

## 7. Monitor or attach to a run

List running tmux sessions:

```bash
tmux ls
```

Attach to one session:

```bash
tmux attach -t exp64-d8-kappa-lin-au-s24
```

Watch the corresponding log file without attaching:

```bash
tail -f exp64-d8-kappa-lin-au-s24.log
```

## Notes

- The training command assumes this repository is the current working directory.
- `--delete-old-ckpts-before-save` removes earlier checkpoints before writing new ones, which keeps disk usage under control.
- `--rebuild-compile-after-first-eval-only` avoids paying a full compile rebuild after every later CORE/sample pass.
- If a cold compile on changed code is still too slow, add `--compile false` to fall back to eager execution.
- If `/DataDrive/nanochat-data` or `~/.cache/nanochat` already exists, adjust the setup command accordingly instead of rerunning it blindly.