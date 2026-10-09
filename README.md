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

    tmux new -d -s $sess "CUDA_VISIBLE_DEVICES=$cuda torchrun --standalone --nproc_per_node=2 -m scripts.base_train --delete-old-ckpts-before-save --model-tag exp64-d8-kappa-lin-au --n-exp 64 --depth 8 --device-batch-size 32 --use-kappa-swiglu --constant-kappa-dense-layers --seed $seed"

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

## Separate Base/SFT Kappa

Add `--separate-base-sft-kappa` to `python -m scripts.base_train_mix` to use
two kappa bias/scale slots: slot 0 for base steps and slot 1 for SFT steps.
All UT passes share the selected task slot. The option enables kappa on both
sources unless `--use-kappa-swiglu-sft-only true` is explicitly supplied.
With separate slots disabled, SFT-only activation defaults to true; pass
`--use-kappa-swiglu-sft-only false` to use kappa on both sources. SFT-only mode
disables kappa during base validation, CORE, and base-prompt sampling.

Base validation and CORE use the base slot in this mode. Standalone chat SFT
and models loaded from SFT/RL checkpoints use the SFT slot. Each slot has
independent AdamW moments and update counts; inactive slots remain unchanged.
The mode is stored in checkpoint model configuration. Start a fresh run when
switching an existing mixed-training recipe to the new parameter layout.

## Independent Kappa Router

Add `--use-kappa-swiglu --independent-kappa-router` to base or mixed training
to predict token-dependent kappa scales with a separate bias-free token-to-expert
projection. The original router still selects experts and weights their
outputs. The predictor gathers raw outputs for those selected experts and
uses `kappa_bias + predicted_kappa_scale` as the activation conditioning.
There is no additional learned `kappa_scale` multiplier, softmax, or logit
normalization in this mode; `top_logits` and `router_probs` behave identically.
Diagnostics use detached predictor logits cached from the latest forward,
excluding padding and unused expert slots. Constant conditioning is unsupported.

Predictor weights receive full gradients. Only the predictor's gradient path
back to the input latent is multiplied by 0.1; expert and routing paths are
unchanged. The predictor uses the MoE matrix optimizer and matrix learning
rate, not the kappa bias/scale learning-rate schedule.

With `--separate-base-sft-kappa`, one predictor projection stores two weight
sets as stacked rows, reshaped to `(2, n_exp, n_embd)`. Base steps select
slot 0 and SFT steps select slot 1; all UT passes use the selected phase.
Each slot has independent matrix-optimizer state, and inactive weights and
state remain unchanged. Without this option, the predictor remains shared.

Chat SFT inherits this mode from the checkpoint. Add
`--independent-kappa-router` to enable it on an existing kappa checkpoint;
missing predictor weights are copied from its router weights,
preserving deterministic initial conditioning. In separate-phase mode,
missing or older single-set predictor weights are copied into both slots.
Fresh pretraining initializes the predictor uniformly like
other input projections. Predictor weights and the mode are saved for evaluation.
Legacy learned scale tensors are ignored when loading independent mode.
Existing independent checkpoints therefore load but do not preserve their old outputs.

## Notes

- The training command assumes this repository is the current working directory.
- `--delete-old-ckpts-before-save` removes earlier checkpoints before writing new ones, which keeps disk usage under control.
- `--rebuild-compile-after-first-eval-only` avoids paying a full compile rebuild after every later CORE/sample pass.
- If a cold compile on changed code is still too slow, add `--compile false` to fall back to eager execution.
- If `/DataDrive/nanochat-data` or `~/.cache/nanochat` already exists, adjust the setup command accordingly instead of rerunning it blindly.