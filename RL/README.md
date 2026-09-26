# Sokoban Experiments

## Installation

Requires Python 3.10 or 3.11, `uv`, and an NVIDIA GPU with CUDA 12.
Run from this directory:

```bash
uv sync
```

## Example Runs

Sokoban small-budget example (`depth=1`, `num_experts=1`,
`expert_hidden_dim=128`):

```bash
uv run python -u drc/train.py \
  --expert_type stacked_lstm \
  --depth 1 \
  --num_experts 1 \
  --expert_hidden_dim 128
```

## Fixed-Compute Sweeps

For a sweep, override `--depth`, `--num_experts`, and `--expert_hidden_dim`.

For ConvGLU architecture, use `--expert_type rec_conv_glu`.
Add `--reset_hidden_every_step` for the non-recurrent option.

