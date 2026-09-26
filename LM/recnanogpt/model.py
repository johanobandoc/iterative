import math
import dataclasses

import jax
import jax.numpy as jnp

from config import LinearConfig
from layers import Embedding, Linear, GroupedQueryAttention
from utils import ParamInitializer
from utils import ParamSpec
from utils import is_param_spec
from utils import jax_pytree_struct
from utils import layer_repr


# The recurrent MoE path uses single-query attention (Q=1) against a short
# rolling KV cache. cuDNN fused attention does not support that shape here,
# so force the generic implementation on GPU in this fork.
if jax.default_backend() == "gpu":
    ATTN_IMPL = None
elif jax.default_backend() == "tpu":
    ATTN_IMPL = "xla"
else:
    ATTN_IMPL = None

@jax_pytree_struct
class MLP(ParamInitializer):
    fc1: Linear
    fc2: Linear

    @classmethod
    def param_specs(cls, cfg):
        fc1 = Linear.param_specs(cfg.fc1)
        fc2 = Linear.param_specs(cfg.fc2)
        return MLP(fc1=fc1, fc2=fc2)

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class RecurrentTransformerBlock(ParamInitializer):
    in_proj: Linear
    attn: GroupedQueryAttention
    mlp: MLP

    @classmethod
    def param_specs(cls, cfg):
        recurrent_input_width = 2 * cfg.expert_hidden_dim
        in_proj_cfg = LinearConfig(
            dtype=cfg.dtype,
            in_features=recurrent_input_width,
            out_features=cfg.expert_hidden_dim,
            use_bias=False,
            weight_logical_axes=("linear_in", "linear_out"),
        )
        in_proj = Linear.param_specs(in_proj_cfg)
        attn_cfg = dataclasses.replace(
            cfg.attn,
            d_in=cfg.expert_hidden_dim,
        )
        attn = GroupedQueryAttention.param_specs(attn_cfg)
        mlp = MLP.param_specs(cfg.mlp)
        return RecurrentTransformerBlock(
            in_proj=in_proj,
            attn=attn,
            mlp=mlp,
        )

    def __repr__(self):
        return layer_repr(self)


def _stack_param_specs(tree, axis_size: int):
    return jax.tree_util.tree_map(
        lambda spec: None
        if spec is None
        else ParamSpec(
            shape=(axis_size,) + spec.shape,
            dtype=spec.dtype,
            logical_axes=(None,) + spec.logical_axes,
            initializer=spec.initializer,
        ),
        tree,
        is_leaf=is_param_spec,
    )


@jax_pytree_struct
class GPT(ParamInitializer):
    embed: Embedding
    blocks: RecurrentTransformerBlock
    lm_head: Linear
    num_experts: int = dataclasses.field(metadata=dict(static=True))
    res_stream_width: int = dataclasses.field(metadata=dict(static=True))
    attn_head_dim: int = dataclasses.field(metadata=dict(static=True))
    attn_kv_heads: int = dataclasses.field(metadata=dict(static=True))
    memory_len: int = dataclasses.field(metadata=dict(static=True))
    checkpoint_token_step: bool = dataclasses.field(metadata=dict(static=True))
    segment_local_kv_cache: bool = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        validate_recurrent_model_config(cfg)
        res_stream_width = cfg.expert_hidden_dim
        block = RecurrentTransformerBlock.param_specs(cfg)
        lm_head_cfg = dataclasses.replace(
            cfg.lm_head,
            in_features=res_stream_width,
        )
        return GPT(
            embed=Embedding.param_specs(cfg.embed),
            blocks=_stack_param_specs(block, cfg.num_experts),
            lm_head=Linear.param_specs(lm_head_cfg),
            num_experts=cfg.num_experts,
            res_stream_width=res_stream_width,
            attn_head_dim=cfg.expert_hidden_dim // cfg.q_heads,
            attn_kv_heads=cfg.kv_heads,
            memory_len=cfg.memory_len,
            checkpoint_token_step=cfg.checkpoint_token_step,
            segment_local_kv_cache=cfg.segment_local_kv_cache,
        )

    def __repr__(self):
        return layer_repr(self)


def precompute_frequencies(
    positions: jax.Array, features: int, theta=10000.0, dtype=None
):
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    if dtype is not None:
        sin = sin.astype(dtype)
        cos = cos.astype(dtype)
    return sin, cos


def calculate_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    orig_dtype = x.dtype
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(
        orig_dtype
    )


def embedding_forward(params, x):
    return params.weight.at[x, :].get()


def rmsnorm_forward(x, scale=None, eps=1e-5):
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    inv_scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    x = x / inv_scale
    if scale is not None:
        x = x * scale.astype(jnp.float32)
    return x.astype(orig_dtype)


def linear_forward(params, x):
    out = jnp.einsum("...d,dv->...v", x, params.weight)
    if params.bias is not None:
        return out + params.bias
    return out


def mlp_forward(params, x):
    x = linear_forward(params.fc1, x)
    x = jnp.square(jax.nn.relu(x))
    x = linear_forward(params.fc2, x)
    return x


def validate_recurrent_model_config(cfg):
    if cfg.num_experts < 1:
        raise ValueError("`num_experts` must be >= 1.")
    if cfg.memory_len < 1:
        raise ValueError("`memory_len` must be >= 1.")


def _append_to_cache(k_cache, v_cache, k_cur, v_cur, cache_fill):
    memory_len = k_cache.shape[2]

    def append_one(k_mem, v_mem, k_new, v_new, fill):
        k_new = k_new[:, None, :]
        v_new = v_new[:, None, :]

        def write_into_free_slot(_):
            return (
                jax.lax.dynamic_update_slice_in_dim(k_mem, k_new, fill, axis=1),
                jax.lax.dynamic_update_slice_in_dim(v_mem, v_new, fill, axis=1),
            )

        def roll_and_append(_):
            return (
                jnp.concatenate([k_mem[:, 1:, :], k_new], axis=1),
                jnp.concatenate([v_mem[:, 1:, :], v_new], axis=1),
            )

        return jax.lax.cond(fill < memory_len, write_into_free_slot, roll_and_append, operand=None)

    return jax.vmap(append_one)(k_cache, v_cache, k_cur, v_cur, cache_fill)


def _build_attention_cache_sequence(k_cache, v_cache, k_cur, v_cur, cache_fill):
    next_k_cache, next_v_cache = _append_to_cache(
        k_cache, v_cache, k_cur, v_cur, cache_fill,
    )
    fill_next = jnp.minimum(cache_fill + 1, k_cache.shape[2])
    key_seq = jnp.transpose(next_k_cache, (0, 2, 1, 3))
    value_seq = jnp.transpose(next_v_cache, (0, 2, 1, 3))
    valid_mask = (
        jnp.arange(k_cache.shape[2])[None, None, None, :]
        < fill_next[:, None, None, None]
    )
    return next_k_cache, next_v_cache, key_seq, value_seq, valid_mask


def _build_segment_local_attention_cache_sequence(
    k_cache,
    v_cache,
    local_k_cache,
    local_v_cache,
    k_cur,
    v_cur,
    cache_fill,
    local_pos,
):
    if local_k_cache.shape[2] > 0:
        k_new = k_cur[:, :, None, :]
        v_new = v_cur[:, :, None, :]
        local_k_cache = jax.lax.dynamic_update_slice_in_dim(
            local_k_cache,
            k_new,
            local_pos,
            axis=2,
        )
        local_v_cache = jax.lax.dynamic_update_slice_in_dim(
            local_v_cache,
            v_new,
            local_pos,
            axis=2,
        )
    else:
        return local_k_cache, local_v_cache, k_cur[:, None, :, :], v_cur[:, None, :, :], None

    memory_len = k_cache.shape[2]
    segment_len = local_k_cache.shape[2]
    prefix_len = jnp.minimum(
        local_pos + jnp.asarray(1, dtype=cache_fill.dtype),
        jnp.asarray(segment_len, dtype=cache_fill.dtype),
    )

    def window_one(k_mem, v_mem, k_seg, v_seg, fill):
        combined_k = jnp.concatenate([k_mem, k_seg], axis=1)
        combined_v = jnp.concatenate([v_mem, v_seg], axis=1)
        valid_len = fill + prefix_len
        fill_next = jnp.minimum(valid_len, memory_len)
        start = jnp.maximum(valid_len - memory_len, 0)
        out_pos = jnp.arange(memory_len, dtype=fill.dtype)
        logical_pos = start + out_pos
        src_pos = jnp.where(
            logical_pos < fill,
            logical_pos,
            memory_len + (logical_pos - fill),
        )
        src_pos = jnp.clip(src_pos, 0, memory_len + segment_len - 1)
        k_next = jnp.take(combined_k, src_pos, axis=1)
        v_next = jnp.take(combined_v, src_pos, axis=1)
        valid_out = out_pos < fill_next
        k_next = jnp.where(valid_out[None, :, None], k_next, jnp.zeros_like(k_next))
        v_next = jnp.where(valid_out[None, :, None], v_next, jnp.zeros_like(v_next))
        return k_next, v_next, fill_next

    window_k, window_v, fill_next = jax.vmap(window_one)(
        k_cache,
        v_cache,
        local_k_cache,
        local_v_cache,
        cache_fill,
    )
    key_seq = jnp.transpose(window_k, (0, 2, 1, 3))
    value_seq = jnp.transpose(window_v, (0, 2, 1, 3))
    valid_mask = (
        jnp.arange(memory_len, dtype=jnp.int32)[None, None, None, :]
        < fill_next[:, None, None, None]
    )
    return local_k_cache, local_v_cache, key_seq, value_seq, valid_mask


def _append_segment_to_cache(k_cache, v_cache, local_k_cache, local_v_cache, cache_fill):
    memory_len = k_cache.shape[2]
    segment_len = local_k_cache.shape[2]
    if segment_len == 0:
        return k_cache, v_cache, cache_fill

    def append_one(k_mem, v_mem, k_seg, v_seg, fill):
        combined_k = jnp.concatenate([k_mem, k_seg], axis=1)
        combined_v = jnp.concatenate([v_mem, v_seg], axis=1)
        valid_len = fill + jnp.asarray(segment_len, dtype=fill.dtype)
        fill_next = jnp.minimum(valid_len, memory_len)
        start = jnp.maximum(valid_len - memory_len, 0)
        out_pos = jnp.arange(memory_len, dtype=fill.dtype)
        logical_pos = start + out_pos
        src_pos = jnp.where(
            logical_pos < fill,
            logical_pos,
            memory_len + (logical_pos - fill),
        )
        src_pos = jnp.clip(src_pos, 0, memory_len + segment_len - 1)
        k_next = jnp.take(combined_k, src_pos, axis=1)
        v_next = jnp.take(combined_v, src_pos, axis=1)
        valid_out = out_pos < fill_next
        k_next = jnp.where(valid_out[None, :, None], k_next, jnp.zeros_like(k_next))
        v_next = jnp.where(valid_out[None, :, None], v_next, jnp.zeros_like(v_next))
        return k_next, v_next, fill_next

    return jax.vmap(append_one)(
        k_cache,
        v_cache,
        local_k_cache,
        local_v_cache,
        cache_fill,
    )


def _prepare_recurrent_layer_attention(
    params,
    token_embed,
    res_stream,
    freqs,
):
    x_in = jnp.concatenate([token_embed, res_stream], axis=-1)
    x = linear_forward(params.in_proj, x_in)
    attn_in = x
    attn_in = rmsnorm_forward(attn_in)
    sin, cos = freqs

    q = jnp.einsum("bd,dhq->bhq", attn_in, params.attn.wq)
    k_cur = jnp.einsum("bd,dhq->bhq", attn_in, params.attn.wk)
    v_cur = jnp.einsum("bd,dhq->bhq", attn_in, params.attn.wv)

    q = rmsnorm_forward(q)
    k_cur = rmsnorm_forward(k_cur)
    q = jnp.squeeze(calculate_rope(q[:, None, :, :], sin, cos), axis=1)
    k_cur = jnp.squeeze(calculate_rope(k_cur[:, None, :, :], sin, cos), axis=1)
    return x, q, k_cur, v_cur


def _finish_recurrent_layer_attention(
    params,
    x,
    q,
    key_seq,
    value_seq,
    valid_mask,
):
    attn = jax.nn.dot_product_attention(
        q[:, None, :, :],
        key_seq,
        value_seq,
        mask=valid_mask,
        scale=1.0 / math.sqrt(q.shape[-1]),
        is_causal=False,
        implementation=ATTN_IMPL,
    ).astype(x.dtype)
    attn = jnp.squeeze(attn, axis=1)
    attn_out = jnp.einsum("bhq,hqd->bd", attn, params.attn.wo)

    x = x + attn_out
    ffn_out = mlp_forward(params.mlp, rmsnorm_forward(x))
    return rmsnorm_forward(x + ffn_out)


def _logits_from_res_stream(params, res_stream):
    head_in = rmsnorm_forward(res_stream)
    logits = linear_forward(params.lm_head, head_in)
    return 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)


def _pre_output_activation_l2(res_stream):
    return jnp.mean(jnp.square(res_stream.astype(jnp.float32)), axis=-1)


def _aggregate_layer_outputs(params, layer_outs):
    return (jnp.sum(layer_outs, axis=0) / math.sqrt(params.num_experts)).astype(layer_outs.dtype)


def _aggregate_shared_kv_candidates(params, kv_stack):
    merged = jnp.sum(kv_stack.astype(jnp.float32), axis=0) / math.sqrt(params.num_experts)
    return merged.astype(kv_stack.dtype)


def _merge_layer_kv_candidates(params, k_stack, v_stack):
    merged_k = _aggregate_shared_kv_candidates(params, k_stack)
    merged_v = _aggregate_shared_kv_candidates(params, v_stack)
    return merged_k, merged_v
