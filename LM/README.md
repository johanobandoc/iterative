# FineWeb Experiments

This repository contains implementations for NanoGPT-style baselines and recurrent-transformer FineWeb experiments.

The codebase is heavily based on
[nanoGPTJAX](https://github.com/AakashKumarNain/nanoGPTJAX).

`recnanogpt` contains the recurrent Transformer. `nanogpt` contains Transformer.

## Install

Install `uv`, then create the environment for an NVIDIA GPU with CUDA 12:

```bash
uv sync
```

## Data

Download tokenized FineWeb shards:

```bash
uv run --extra data python recnanogpt/download_fineweb_tokens.py --data_dir /path/to/LM
```

## FineWeb Runs

Recurrent small-budget run (`depth=1`, `num_experts=16`, `width=768`):

```bash
DATA_DIR=/path/to/LM
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
  --checkpoint_token_step \
  --segment_local_kv_cache \
  --data_dir "${DATA_DIR}" \
  --ckpt_path "${CKPT_DIR}"
```

Non-recurrent run with the same allocation:

```bash
DATA_DIR=/path/to/LM
CKPT_DIR=/path/to/checkpoints/fineweb_nonrecurrent_d1_e16_w768

uv run python -u nanogpt/train.py \
  --exp_name fineweb_nonrecurrent_d1_e16_w768 \
  --depth 1 \
  --num_experts 16 \
  --expert_hidden_dim 768 \
  --q_heads 8 \
  --kv_heads 4 \
  --per_device_batch_size 128 \
  --seqlen 256 \
  --data_dir "${DATA_DIR}" \
  --ckpt_path "${CKPT_DIR}"
```

Reduce `per_device_batch_size` if needed to fit GPU memory; gradient accumulation
is calculated automatically. 

For longer context, use `--memory_len 1024` with recurrent training, keeping `--seqlen 64`. 
For non-recurrent training, use `--seqlen 2048` and `--per_device_batch_size 16` on four GPUs.

To run an allocation sweep, repeat the corresponding command for each row below, 
changing `depth`, `num_experts`, `expert_hidden_dim`, `q_heads`, and `kv_heads`. 

Small budget:

| depth | num_experts | expert_hidden_dim | q_heads | kv_heads |
|---:|---:|---:|---:|---:|
| 1 | 16 | 768 | 8 | 4 |
| 1 | 6 | 1152 | 12 | 6 |
| 1 | 4 | 1344 | 14 | 7 |
| 1 | 3 | 1536 | 16 | 8 |
| 1 | 1 | 2112 | 22 | 11 |
| 2 | 8 | 768 | 8 | 4 |
| 2 | 5 | 960 | 10 | 5 |
| 2 | 3 | 1152 | 12 | 6 |
| 2 | 2 | 1344 | 14 | 7 |
| 2 | 1 | 1728 | 18 | 9 |
| 4 | 4 | 768 | 8 | 4 |
| 4 | 1 | 1344 | 14 | 7 |
| 8 | 2 | 768 | 8 | 4 |
| 16 | 1 | 768 | 8 | 4 |

Medium budget:

| depth | num_experts | expert_hidden_dim | q_heads | kv_heads |
|---:|---:|---:|---:|---:|
| 1 | 32 | 768 | 8 | 4 |
| 1 | 13 | 1152 | 12 | 6 |
| 1 | 7 | 1536 | 16 | 8 |
| 1 | 3 | 2112 | 22 | 11 |
| 1 | 1 | 3072 | 32 | 16 |
| 2 | 16 | 768 | 8 | 4 |
| 2 | 10 | 960 | 10 | 5 |
| 2 | 7 | 1152 | 12 | 6 |
| 2 | 2 | 1920 | 20 | 10 |
| 2 | 1 | 2496 | 26 | 13 |
| 4 | 8 | 768 | 8 | 4 |
| 4 | 5 | 960 | 10 | 5 |
| 4 | 3 | 1200 | 12 | 6 |
| 4 | 1 | 1920 | 20 | 10 |
| 7 | 2 | 1152 | 12 | 6 |
| 8 | 4 | 768 | 8 | 4 |
| 8 | 1 | 1440 | 16 | 8 |
| 13 | 1 | 1152 | 12 | 6 |
| 16 | 2 | 768 | 8 | 4 |
| 16 | 1 | 1056 | 12 | 6 |
| 32 | 1 | 768 | 8 | 4 |

## Sliding Window Evaluation

Evaluate a saved checkpoint with a rolling attention window:

```bash
uv run python -u nanogpt/eval_fineweb_sliding.py \
  --checkpoint "${CKPT_DIR}/10000" \
  --data_dir "${DATA_DIR}" \
  --window_size 128
```

 Use `--window_size 1024` for long-context evaluation. 
 The defaults use 2048 independent validation streams.
