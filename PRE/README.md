# PRE FineWeb Experiments

This repository contains JAX implementations for NanoGPT-style baselines and
PRE recurrent-transformer FineWeb experiments.
The standard NanoGPT-style implementation is included as a reference baseline
for comparison.

The codebase is heavily based on
[nanoGPTJAX](https://github.com/AakashKumarNain/nanoGPTJAX).

## Model Overview

The `recnanogpt` package contains the PRE recurrent transformer used for the
FineWeb experiments. It replaces the sequential transformer stack with parallel
recurrent experts executed with `vmap`, while keeping the token embedding and LM
head outside the recurrent core.

In the recurrent model, experts share one residual stream and one K/V memory
cache across recurrent steps. Each expert proposes a K/V update at each tick,
expert outputs are aggregated into the next shared residual stream, and the
merged K/V candidate is written to memory on the final tick for each token.
Serial depth stacks independent shared-K/V recurrent cores, each with its own
K/V cache.

Training uses equal-size streamed token chunks and carries recurrent state across
batches. Recurrent state is reset when a new shard starts, and gradients are
truncated across streamed microbatch boundaries.

## Install

Install `uv`, then create the environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

For NVIDIA GPUs with CUDA 12:

```bash
uv sync --extra jaxgpu
```

Run commands through `uv`:

```bash
uv run python -c "import jax; print(jax.devices())"
```

## Data

Download tokenized FineWeb shards:

```bash
uv run python recnanogpt/download_fineweb_tokens.py --data_dir /path/to/fineweb
```

## FineWeb Runs

Default small-budget run (`depth=1`, `num_experts=16`, `width=768`):

```bash
DATA_DIR=/path/to/fineweb
CKPT_DIR=/path/to/checkpoints/fineweb_default_d1_e16_w768

uv run python -u recnanogpt/train.py \
  --exp_name fineweb_default_d1_e16_w768 \
  --depth 1 \
  --num_experts 16 \
  --expert_hidden_dim 768 \
  --q_heads 8 \
  --kv_heads 4 \
  --per_device_batch_size 512 \
  --seqlen 64 \
  --memory_len 128 \
  --ticks 1 \
  --memory_aggregation_regime sum_div_sqrt_num_experts \
  --experts_aggregation_regime sum_div_sqrt_group_size \
  --checkpoint_token_step \
  --segment_local_kv_cache \
  --attn_output_gate \
  --qkv_input_gelu_proj \
  --muon_gates \
  --pre_output_reg_cost 1e-9 \
  --data_dir "${DATA_DIR}" \
  --ckpt_path "${CKPT_DIR}" \
  --checkpoint_save_steps 1000 \
  --max_checkpoints_to_keep 2
```

To reproduce the small- and medium-budget sweeps, keep the fixed flags above and
change only `depth`, `num_experts`, `expert_hidden_dim`, `q_heads`, `kv_heads`,
and `per_device_batch_size`. Use a distinct `exp_name` and `ckpt_path` for each
row.

Small budget:

| depth | num_experts | expert_hidden_dim | q_heads | kv_heads | per_device_batch_size |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 768 | 8 | 4 | 512 |
| 2 | 8 | 768 | 8 | 4 | 512 |
| 4 | 4 | 768 | 8 | 4 | 512 |
| 1 | 7 | 1152 | 12 | 6 | 512 |
| 1 | 5 | 1344 | 14 | 7 | 512 |
| 2 | 5 | 960 | 10 | 5 | 512 |
| 1 | 4 | 1536 | 16 | 8 | 512 |
| 2 | 2 | 1536 | 16 | 8 | 512 |
| 4 | 1 | 1536 | 16 | 8 | 512 |
| 1 | 2 | 2112 | 22 | 11 | 512 |
| 2 | 1 | 2112 | 22 | 11 | 512 |
| 1 | 1 | 3072 | 32 | 16 | 512 |

Medium budget:

| depth | num_experts | expert_hidden_dim | q_heads | kv_heads | per_device_batch_size |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 768 | 8 | 4 | 256 |
| 2 | 16 | 768 | 8 | 4 | 256 |
| 4 | 8 | 768 | 8 | 4 | 256 |
| 8 | 4 | 768 | 8 | 4 | 128 |
| 1 | 14 | 1152 | 12 | 6 | 256 |
| 1 | 10 | 1344 | 14 | 7 | 256 |
| 2 | 10 | 960 | 10 | 5 | 256 |
| 1 | 8 | 1536 | 16 | 8 | 256 |
| 2 | 4 | 1536 | 16 | 8 | 256 |
| 4 | 2 | 1536 | 16 | 8 | 128 |
| 8 | 1 | 1536 | 16 | 8 | 64 |
| 1 | 4 | 2112 | 22 | 11 | 256 |
| 2 | 2 | 2112 | 22 | 11 | 256 |
| 1 | 2 | 3072 | 32 | 16 | 256 |
| 2 | 1 | 3072 | 32 | 16 | 256 |
| 1 | 1 | 4416 | 46 | 23 | 256 |
| 8 | 16 | 384 | 4 | 2 | 128 |

## Entrypoints

- `recnanogpt/train.py`: PRE recurrent transformer training.
- `nanogpt/train.py`: non-recurrent GPT and MoE baselines.
