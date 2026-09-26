"""Evaluate a non-recurrent checkpoint with a rolling KV cache."""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import model_moe
from checkpoint_utils import load_weights_from_checkpoint_with_validation
from config import BATCH_AXIS_NAME, Config, ModelConfig, ShardingRules
from fineweb_dataloader import load_shard_tokens, make_window_sampler
from jax_compat import mesh_context
from moe_sliding_eval import MoEKVCache, init_moe_kv_cache, score_chunk


def protocol_geometry(
    num_tokens, logical_stream_batch_size=2048, chunk_size=64, window_size=128
):
    if min(logical_stream_batch_size, chunk_size, window_size) < 1:
        raise ValueError("Stream count, chunk size, and window size must be positive.")
    stream_len = num_tokens // logical_stream_batch_size
    total_chunks = max(0, (stream_len - 1) // chunk_size)
    warmup_chunks = (window_size + chunk_size - 1) // chunk_size
    scored_chunks = total_chunks - warmup_chunks
    if scored_chunks <= 0:
        raise ValueError("Validation shard has no scored chunks after cache warm-up.")
    return {
        "total_chunks": total_chunks,
        "warmup_chunks": warmup_chunks,
        "scored_chunks": scored_chunks,
        "scored_tokens": scored_chunks * logical_stream_batch_size * chunk_size,
        "final_absolute_cursor": total_chunks * chunk_size,
    }


def load_model(checkpoint, devices=None):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    for item in ("metadata", "params"):
        if not (checkpoint / item).is_dir():
            raise ValueError(f"Checkpoint lacks {item}: {checkpoint}")
    with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as checkpointer:
        metadata = checkpointer.restore(
            checkpoint / "metadata", args=ocp.args.JsonRestore()
        )
    fields = {
        "num_layers",
        "num_experts",
        "d_emb",
        "q_heads",
        "kv_heads",
        "seqlen",
        "vocab_size",
    }
    saved_model = metadata.get("model", {})
    if (
        not isinstance(saved_model, dict)
        or not fields <= saved_model.keys()
        or "dtype" not in metadata
    ):
        raise ValueError("Checkpoint lacks model settings saved by nanogpt/train.py.")
    if saved_model.get("model_backend", "moe") != "moe":
        raise ValueError("Checkpoint must use stacked expert parameters.")
    model_cfg = ModelConfig(
        **{name: saved_model[name] for name in fields},
        dtype=jnp.dtype(metadata["dtype"]),
    )
    if model_cfg.attn.head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension.")
    devices = jax.devices() if devices is None else devices
    mesh = Mesh(np.asarray(devices), axis_names=BATCH_AXIS_NAME)
    rules = ShardingRules(batch=BATCH_AXIS_NAME)
    cfg = Config(mesh=mesh, rules=rules, model=model_cfg)
    model_class = model_moe.GPT
    sharding = model_class.shardings(mesh, rules, model_cfg)
    with mesh_context(mesh):
        template = jax.eval_shape(lambda: model_class.init(jax.random.PRNGKey(0), cfg))
        params = load_weights_from_checkpoint_with_validation(
            checkpoint / "params", template, sharding
        )
    return cfg, params


def cache_sharding(cache, mesh):
    array = NamedSharding(mesh, P(None, BATCH_AXIS_NAME, None, None, None))
    return MoEKVCache(
        k=[array for _ in cache.k],
        v=[array for _ in cache.v],
        end=NamedSharding(mesh, P()),
        window_size=cache.window_size,
    )


def evaluate_shards(
    params,
    cfg,
    files,
    *,
    window_size=128,
    logical_stream_batch_size=2048,
    chunk_size=64,
    lane_block_size=256,
):
    if min(window_size, logical_stream_batch_size, chunk_size, lane_block_size) < 1:
        raise ValueError(
            "Window, stream, chunk, and lane-block sizes must be positive."
        )
    if logical_stream_batch_size % lane_block_size:
        raise ValueError("lane_block_size must divide logical_stream_batch_size.")
    if lane_block_size % cfg.mesh.size:
        raise ValueError("lane_block_size must be divisible by the number of devices.")
    if not files:
        raise ValueError("No validation shards were selected.")
    replicated = NamedSharding(cfg.mesh, P())
    params_sharding = jax.tree.map(lambda _: replicated, params)
    params = jax.device_put(params, params_sharding)
    initial = init_moe_kv_cache(
        params, lane_block_size, window_size, cfg.model.dtype
    )
    cache_layout = cache_sharding(initial, cfg.mesh)
    del initial
    data_layout = NamedSharding(cfg.mesh, P(BATCH_AXIS_NAME, None))
    scorer = jax.jit(
        lambda weights, x, y, cache: score_chunk(
            weights, x, y, cache, cfg.model.attn.head_dim
        ),
        in_shardings=(params_sharding, data_layout, data_layout, cache_layout),
        out_shardings=(None, None, cache_layout),
        donate_argnums=(3,),
    )
    total_ce = 0.0
    total_targets = 0
    expected_targets = 0
    for path in files:
        loaded = load_shard_tokens(path)
        tokens = loaded["tokens"]
        try:
            geometry = protocol_geometry(
                loaded["size"], logical_stream_batch_size, chunk_size, window_size
            )
            sampler = make_window_sampler(tokens, size=loaded["size"])
            total_chunks = sampler.build(logical_stream_batch_size, chunk_size)
            if total_chunks != geometry["total_chunks"]:
                raise RuntimeError("Validation sampler geometry does not match.")
            expected_targets += geometry["scored_tokens"]
            host_buffer = np.empty((lane_block_size, chunk_size + 1), dtype=np.uint16)
            for lane_start in range(0, logical_stream_batch_size, lane_block_size):
                lane_stop = lane_start + lane_block_size
                cache = init_moe_kv_cache(
                    params,
                    lane_block_size,
                    window_size,
                    cfg.model.dtype,
                )
                cache = jax.device_put(cache, cache_layout)
                for chunk in range(total_chunks):
                    starts = sampler.built_starts[chunk, lane_start:lane_stop]
                    ends = sampler.built_ends[chunk, lane_start:lane_stop]
                    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
                        host_buffer[row] = tokens[int(start) : int(end)]
                    x = jax.device_put(
                        host_buffer[:, :-1].astype(np.int32), data_layout
                    )
                    y = jax.device_put(host_buffer[:, 1:].astype(np.int32), data_layout)
                    ce_sum, count, cache = scorer(params, x, y, cache)
                    if chunk >= geometry["warmup_chunks"]:
                        ce_value, count_value = jax.device_get((ce_sum, count))
                        total_ce += float(ce_value)
                        total_targets += int(count_value)
                if int(jax.device_get(cache.end)) != geometry["final_absolute_cursor"]:
                    raise RuntimeError("The absolute KV-cache position is incorrect.")
            print(
                f"Evaluated {Path(path).name}: {geometry['scored_tokens']:,} targets",
                flush=True,
            )
        finally:
            tokens.unlink_on_del()
    if total_targets != expected_targets:
        raise RuntimeError("The number of scored validation targets is incorrect.")
    loss = total_ce / total_targets
    if not math.isfinite(loss):
        raise FloatingPointError("Validation cross-entropy is not finite.")
    return {
        "cross_entropy": loss,
        "perplexity": math.exp(loss),
        "scored_tokens": total_targets,
        "window_size": window_size,
        "logical_stream_batch_size": logical_stream_batch_size,
        "chunk_size": chunk_size,
        "warmup_chunks": (window_size + chunk_size - 1) // chunk_size,
        "validation_shards": len(files),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Training checkpoint directory containing params and metadata.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing *val*.bin token shards.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=128,
        help="Attention window, including the current token.",
    )
    parser.add_argument(
        "--logical_stream_batch_size",
        type=int,
        default=2048,
        help="Number of independent streams used to split each shard.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="Tokens processed per stream in each chunk.",
    )
    parser.add_argument(
        "--lane_block_size",
        type=int,
        default=256,
        help="Streams evaluated at once; reduce to lower memory use.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    files = sorted(args.data_dir.glob("*val*.bin"))
    if not files:
        raise FileNotFoundError(f"No validation shards found in {args.data_dir}.")
    cfg, params = load_model(args.checkpoint)
    with mesh_context(cfg.mesh):
        result = evaluate_shards(
            params,
            cfg,
            files,
            window_size=args.window_size,
            logical_stream_batch_size=args.logical_stream_batch_size,
            chunk_size=args.chunk_size,
            lane_block_size=args.lane_block_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
