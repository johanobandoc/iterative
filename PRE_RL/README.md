# PRE RL Experiments

This directory contains the JAX reinforcement-learning implementation for the
Sokoban and Craftax PRE experiments. 

## Model Overview

`drc` implements `PRENet`, a policy/value network whose recurrent core is a
set of parallel recurrent experts. Each expert reads the encoded observation,
the shared residual state, and its own recurrent memory. Expert outputs are
merged with a fixed `sum / sqrt(num_experts)` aggregation rule into the next
shared residual state.

The main computation axes are:

| Name | CLI argument | Meaning |
|---|---|---|
| Within-step depth | `--depth` | Number of serial recurrent transformations inside one environment step. |
| Number of experts | `--num_experts` | Number of recurrent experts evaluated in parallel at each depth level. |
| Expert width | `--expert_hidden_dim` | Hidden dimension of each recurrent expert. |

Sokoban uses convolutional LSTM experts (`--expert_type stacked_lstm`) over a
grid embedding. Craftax symbolic observations use dense LSTM experts
(`--expert_type stacked_dense_lstm`) over an MLP-GLU encoder.

## Install

This project uses `uv` for dependency locking and environment creation. Install
`uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Create a CPU environment:

```bash
uv sync
```

On CUDA 12 GPU machines, install the CUDA JAX extra:

```bash
uv sync --extra cuda12
```

Run commands from this directory through the environment:

```bash
uv run python -u drc/train.py --help
```

On CUDA machines, include the same extra when running commands:

```bash
uv run --extra cuda12 python -u drc/train.py --help
```

The examples below use the portable CPU form. On CUDA jobs, insert `--extra cuda12`
after `uv run`.

For exact reproduction from `uv.lock`, add `--frozen` to `uv sync` and `uv run`.

## Example Runs

Sokoban small-budget example (`depth=1`, `num_experts=1`,
`expert_hidden_dim=128`):

```bash
uv run python -u drc/train.py \
  --env_id Sokoban-v0 \
  --total_timesteps 50000000 \
  --obs_encoder embedding_rms \
  --expert_type stacked_lstm \
  --prehead_aggregation attn \
  --head_type glu \
  --head_hidden_dim 256 \
  --head_num_experts 1 \
  --learning_rate 4e-4 \
  --num_envs 32 \
  --num_steps 20 \
  --agent PRENet \
  --ticks 1 \
  --depth 1 \
  --num_experts 1 \
  --expert_hidden_dim 128 \
  --encoder_hidden_dim 32 \
  --seed 1 \
  --wandb_project_name pre-rl \
  --wandb-tags sokoban fixed_compute small_budget \
  --track
```

Craftax small-budget example (`depth=1`, `num_experts=1`,
`expert_hidden_dim=128`):

```bash
uv run python -u drc/train.py \
  --env_id Craftax-Symbolic-v1 \
  --total_timesteps 1000000000 \
  --obs_encoder mlp_glu \
  --prehead_aggregation enc_cell_flatten \
  --expert_type stacked_dense_lstm \
  --head_type none \
  --learning_rate 2e-4 \
  --num_envs 256 \
  --num_steps 20 \
  --agent PRENet \
  --ticks 1 \
  --depth 1 \
  --num_experts 1 \
  --expert_hidden_dim 128 \
  --encoder_hidden_dim 256 \
  --gamma 0.99 \
  --gae_lambda 0.8 \
  --norm_adv \
  --seed 1 \
  --wandb_project_name pre-rl \
  --wandb-tags craftax symbolic fixed_compute small_budget \
  --track
```

Use a distinct `--exp_name` and W&B tag set for each row when running a sweep.

## Fixed-Compute Sweeps

The paper's fixed-compute sweeps vary only `depth`, `num_experts`, and
`expert_hidden_dim` while keeping the domain-specific flags fixed.

Sokoban fixed flags:

```text
--env_id Sokoban-v0
--obs_encoder embedding_rms
--expert_type stacked_lstm
--prehead_aggregation attn
--head_type glu
--head_hidden_dim 256
--head_num_experts 1
--encoder_hidden_dim 32
--num_envs 32
--num_steps 20
--ticks 1
--total_timesteps 50000000
--learning_rate 4e-4
```

Craftax fixed flags:

```text
--env_id Craftax-Symbolic-v1
--obs_encoder mlp_glu
--expert_type stacked_dense_lstm
--prehead_aggregation enc_cell_flatten
--head_type none
--encoder_hidden_dim 256
--num_envs 256
--num_steps 20
--ticks 1
--total_timesteps 1000000000
--learning_rate 2e-4
--gamma 0.99
--gae_lambda 0.8
--norm_adv
```

For Sokoban, the recurrent work proxy is:

```text
depth * num_experts * expert_hidden_dim * (32 + 2 * expert_hidden_dim)
```

For Craftax, the recurrent work proxy is:

```text
depth * num_experts * expert_hidden_dim * (256 + 2 * expert_hidden_dim)
```

Run one job for each table row and each seed.

Sokoban, small budget:

| `expert_hidden_dim` | `depth` | `num_experts` |
|---:|---:|---:|
| 128 | 1 | 1 |
| 64 | 1 | 4 |
| 64 | 2 | 2 |
| 64 | 4 | 1 |
| 32 | 1 | 12 |
| 32 | 2 | 6 |
| 32 | 3 | 4 |
| 32 | 4 | 3 |
| 32 | 8 | 2 |
| 16 | 1 | 36 |
| 16 | 2 | 18 |
| 16 | 3 | 12 |
| 16 | 4 | 9 |
| 16 | 8 | 5 |

Craftax, small budget:

| `expert_hidden_dim` | `depth` | `num_experts` |
|---:|---:|---:|
| 128 | 1 | 1 |
| 64 | 1 | 3 |
| 64 | 3 | 1 |
| 32 | 1 | 6 |
| 32 | 2 | 3 |
| 32 | 3 | 2 |
| 16 | 1 | 14 |
| 16 | 2 | 7 |
| 16 | 3 | 5 |
| 16 | 4 | 4 |
| 16 | 8 | 2 |

Sokoban, medium budget:

| `expert_hidden_dim` | `depth` | `num_experts` |
|---:|---:|---:|
| 256 | 1 | 1 |
| 128 | 1 | 4 |
| 128 | 2 | 2 |
| 128 | 3 | 1 |
| 128 | 4 | 1 |
| 64 | 1 | 14 |
| 64 | 2 | 7 |
| 64 | 3 | 5 |
| 64 | 4 | 3 |
| 64 | 8 | 2 |
| 32 | 1 | 45 |
| 32 | 2 | 23 |
| 32 | 3 | 15 |
| 32 | 4 | 11 |
| 32 | 8 | 6 |
| 16 | 1 | 136 |
| 16 | 2 | 68 |
| 16 | 3 | 45 |
| 16 | 4 | 34 |
| 16 | 8 | 17 |

Craftax, medium budget:

| `expert_hidden_dim` | `depth` | `num_experts` |
|---:|---:|---:|
| 256 | 1 | 1 |
| 128 | 1 | 3 |
| 128 | 3 | 1 |
| 64 | 1 | 8 |
| 64 | 2 | 4 |
| 64 | 3 | 3 |
| 64 | 4 | 2 |
| 64 | 8 | 1 |
| 32 | 1 | 19 |
| 32 | 2 | 10 |
| 32 | 3 | 6 |
| 32 | 4 | 5 |

Sokoban, large budget:

| `expert_hidden_dim` | `depth` | `num_experts` |
|---:|---:|---:|
| 1024 | 1 | 1 |
| 512 | 1 | 4 |
| 512 | 2 | 2 |
| 512 | 4 | 1 |
| 256 | 1 | 15 |
| 256 | 2 | 8 |
| 256 | 3 | 5 |
| 256 | 4 | 4 |
| 256 | 8 | 2 |
| 128 | 1 | 58 |
| 128 | 2 | 29 |
| 128 | 3 | 19 |
| 128 | 4 | 14 |
| 128 | 8 | 7 |
| 64 | 1 | 208 |
| 64 | 2 | 104 |
| 64 | 3 | 69 |
| 64 | 4 | 52 |

Craftax, large budget:

| `expert_hidden_dim` | `depth` | `num_experts` |
|---:|---:|---:|
| 1024 | 1 | 1 |
| 512 | 1 | 4 |
| 512 | 2 | 2 |
| 512 | 4 | 1 |
| 256 | 1 | 12 |
| 256 | 2 | 6 |
| 256 | 3 | 4 |
| 256 | 4 | 3 |
| 128 | 1 | 36 |
| 128 | 2 | 18 |
| 128 | 3 | 12 |
| 128 | 4 | 9 |
| 64 | 1 | 96 |
| 64 | 2 | 48 |
| 64 | 3 | 32 |
| 64 | 4 | 24 |

## Entrypoints

- `drc/train.py`: Tyro CLI for training PRE actor-critic agents.
- `drc/nets.py`: `PRENet` policy/value network.
- `drc/recurrent_moe.py`: recurrent expert core.
- `drc/trainer.py`: A2C/GAE training loop.
