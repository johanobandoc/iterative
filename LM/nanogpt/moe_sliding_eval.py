"""Token-by-token evaluation with a rolling attention window."""

import dataclasses
import math

import jax
import jax.numpy as jnp

import model as base_model
import model_moe
from utils import jax_pytree_struct


@jax_pytree_struct
class MoEKVCache:
    """Per-stage, per-expert rolling KV state used only by this evaluator.

    Each item in ``k`` and ``v`` has shape
    ``[experts, batch, kv_heads, window, head_dim]``. ``end`` is the absolute
    position of the next token to write; unlike a physical ring index, it never
    wraps.
    """

    k: list[jax.Array]
    v: list[jax.Array]
    end: jax.Array
    window_size: int = dataclasses.field(metadata=dict(static=True))

    def next_write_index(self) -> jax.Array:
        return jnp.mod(self.end, self.window_size)

def _validate_params(params):
    if not params.stages:
        raise ValueError("NanoGPT-MoE sliding evaluation requires at least one stage.")
    if params.num_experts < 1:
        raise ValueError("NanoGPT-MoE sliding evaluation requires at least one expert.")

    first_wk = params.stages[0].experts.attn.wk
    if first_wk.ndim != 4 or first_wk.shape[0] != params.num_experts:
        raise ValueError(
            "Expected stacked MoE attention weights with shape "
            "[experts, d_emb, kv_heads, head_dim]."
        )
    expected = (params.num_experts, first_wk.shape[2], first_wk.shape[3])
    for stage in params.stages:
        wk = stage.experts.attn.wk
        actual = (wk.shape[0], wk.shape[2], wk.shape[3])
        if actual != expected:
            raise ValueError(
                "All MoE stages must use the same expert/KV-head dimensions, "
                f"got {actual} and expected {expected}."
            )


def init_moe_kv_cache(params, batch_size, window_size, dtype):
    """Create fresh, empty FIFO state without touching a checkpoint tree."""

    _validate_params(params)
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}.")

    first_wk = params.stages[0].experts.attn.wk
    shape = (
        params.num_experts,
        batch_size,
        first_wk.shape[2],
        window_size,
        first_wk.shape[3],
    )
    cache_dtype = jnp.dtype(dtype)
    return MoEKVCache(
        k=[jnp.zeros(shape, dtype=cache_dtype) for _ in params.stages],
        v=[jnp.zeros(shape, dtype=cache_dtype) for _ in params.stages],
        end=jnp.asarray(0, dtype=jnp.int32),
        window_size=window_size,
    )


def _attention_token(
    params,
    x,
    stage_k,
    stage_v,
    freqs,
    write_index,
    valid_slots,
):
    """Run one expert's attention and update one expert's physical ring."""

    orig_dtype = x.dtype
    sin, cos = freqs

    q = jnp.einsum("btd,dhq->bthq", x, params.wq)
    k = jnp.einsum("btd,dhq->bthq", x, params.wk)
    v = jnp.einsum("btd,dhq->bthq", x, params.wv)

    q = base_model.rmsnorm_forward(q)
    k = base_model.rmsnorm_forward(k)
    q = base_model.calculate_rope(q, sin, cos)
    k = base_model.calculate_rope(k, sin, cos)

    # stage_{k,v}: [B, KVH, W, HD], projected token: [B, 1, KVH, HD].
    stage_k = stage_k.at[:, :, write_index, :].set(k[:, 0, :, :])
    stage_v = stage_v.at[:, :, write_index, :].set(v[:, 0, :, :])
    key = jnp.copy(jnp.transpose(stage_k, (0, 2, 1, 3)))
    value = jnp.copy(jnp.transpose(stage_v, (0, 2, 1, 3)))
    mask = valid_slots[None, None, None, :]

    attn = jax.nn.dot_product_attention(
        q,
        key,
        value,
        mask=mask,
        scale=1.0 / math.sqrt(q.shape[-1]),
        is_causal=False,
        implementation=None,
    ).astype(orig_dtype)

    out = jnp.einsum("bthq,hqd->btd", attn, params.wo)
    return out, stage_k, stage_v


def _expert_block_token(
    params,
    x,
    stage_k,
    stage_v,
    freqs,
    write_index,
    valid_slots,
):
    attn_in = base_model.rmsnorm_forward(x)
    attn_out, stage_k, stage_v = _attention_token(
        params.attn,
        attn_in,
        stage_k,
        stage_v,
        freqs,
        write_index,
        valid_slots,
    )

    x = x + attn_out
    ffn_in = base_model.rmsnorm_forward(x)
    x = x + base_model.mlp_forward(params.mlp, ffn_in)
    return x, stage_k, stage_v




def forward_token(params, token_ids, cache, head_dim):
    """Evaluate one token and advance the FIFO by exactly one position.

    Args:
        params: Existing ``model_moe.GPT`` checkpoint parameter tree.
        token_ids: Integer token IDs with shape ``[batch, 1]``.
        cache: A cache returned by :func:`init_moe_kv_cache` or this function.
        head_dim: RoPE head dimension recorded in the checkpoint config.

    Returns:
        ``(logits, next_cache)`` where logits have shape ``[batch, 1, vocab]``.
    """

    if token_ids.ndim != 2 or token_ids.shape[1] != 1:
        raise ValueError(
            "forward_token accepts one token per lane; expected [batch, 1], "
            f"got {token_ids.shape}."
        )
    if token_ids.shape[0] != cache.k[0].shape[1]:
        raise ValueError(
            f"Token batch {token_ids.shape[0]} does not match cache batch "
            f"{cache.k[0].shape[1]}."
        )
    inferred_head_dim = params.stages[0].experts.attn.wk.shape[-1]
    if head_dim != inferred_head_dim:
        raise ValueError(
            f"head_dim={head_dim} does not match checkpoint head_dim="
            f"{inferred_head_dim}."
        )

    x = base_model.embedding_forward(params.embed, token_ids)
    positions = jnp.broadcast_to(cache.end, token_ids.shape).astype(jnp.int32)
    freqs = base_model.precompute_frequencies(
        positions,
        features=head_dim,
        dtype=x.dtype,
    )
    write_index = cache.next_write_index()
    end_after_write = cache.end + jnp.asarray(1, dtype=cache.end.dtype)
    valid_slots = jnp.arange(cache.window_size, dtype=cache.end.dtype) < jnp.minimum(
        end_after_write, cache.window_size
    )

    next_k = []
    next_v = []
    for stage_index, stage in enumerate(params.stages):
        expert_outputs, stage_k, stage_v = jax.vmap(
            _expert_block_token,
            in_axes=(0, None, 0, 0, None, None, None),
            out_axes=(0, 0, 0),
        )(
            stage.experts,
            x,
            cache.k[stage_index],
            cache.v[stage_index],
            freqs,
            write_index,
            valid_slots,
        )
        x = model_moe._aggregate_expert_outputs(params, expert_outputs)
        next_k.append(stage_k)
        next_v.append(stage_v)

    x = base_model.rmsnorm_forward(x)
    logits = base_model.linear_forward(params.lm_head, x)
    logits = 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)
    next_cache = dataclasses.replace(
        cache,
        k=next_k,
        v=next_v,
        end=end_after_write,
    )
    return logits, next_cache


def _cross_entropy_sum(logits, labels):
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    selected = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)
    return -jnp.sum(selected, dtype=jnp.float32)


def score_chunk(params, x, y, cache, head_dim):
    """Teacher-force a ``[B, T]`` chunk without materializing ``[B,T,V]``."""

    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same [batch, time] shape, got {x.shape} "
            f"and {y.shape}."
        )
    if x.shape[0] != cache.k[0].shape[1]:
        raise ValueError(
            f"Chunk batch {x.shape[0]} does not match cache batch "
            f"{cache.k[0].shape[1]}."
        )

    init = (
        cache,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def scan_token(carry, token_and_label):
        token_cache, ce_sum, token_count = carry
        token, label = token_and_label
        logits, token_cache = forward_token(
            params,
            token[:, None],
            token_cache,
            head_dim,
        )
        ce_sum = ce_sum + _cross_entropy_sum(logits[:, 0, :], label)
        token_count = token_count + jnp.asarray(label.size, dtype=jnp.int32)
        return (token_cache, ce_sum, token_count), None

    (next_cache, ce_sum, token_count), _ = jax.lax.scan(
        scan_token,
        init,
        (jnp.swapaxes(x, 0, 1), jnp.swapaxes(y, 0, 1)),
    )
    return ce_sum, token_count, next_cache
