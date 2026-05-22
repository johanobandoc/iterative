import math
import dataclasses

import jax
import jax.numpy as jnp

from config import LinearConfig, MLPConfig
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
class RMSNorm(ParamInitializer):
    scale: jax.Array | ParamSpec
    width: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, width, dtype):
        return RMSNorm(
            scale=ParamSpec(
                shape=(width,),
                dtype=dtype,
                logical_axes=("norm_out",),
                initializer=jax.nn.initializers.ones,
            ),
            width=width,
        )

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class RoutedAggregationRouter(ParamInitializer):
    context_proj: Linear
    expert_proj: Linear
    expert_embedding: jax.Array | ParamSpec
    embed_dim: int = dataclasses.field(metadata=dict(static=True))
    num_experts: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg, *, context_width: int, num_experts: int):
        embed_dim = cfg.expert_hidden_dim
        context_proj_cfg = LinearConfig(
            dtype=cfg.dtype,
            in_features=context_width,
            out_features=embed_dim,
            use_bias=False,
            weight_logical_axes=("linear_in", "linear_out"),
        )
        expert_proj_cfg = LinearConfig(
            dtype=cfg.dtype,
            in_features=embed_dim,
            out_features=embed_dim,
            use_bias=False,
            weight_logical_axes=("linear_in", "linear_out"),
        )
        expert_embedding = ParamSpec(
            shape=(num_experts, embed_dim),
            dtype=cfg.dtype,
            logical_axes=(None, "linear_out"),
            initializer=jax.nn.initializers.normal(
                stddev=1.0 / math.sqrt(embed_dim)
            ),
        )
        return RoutedAggregationRouter(
            context_proj=Linear.param_specs(context_proj_cfg),
            expert_proj=Linear.param_specs(expert_proj_cfg),
            expert_embedding=expert_embedding,
            embed_dim=embed_dim,
            num_experts=num_experts,
        )

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class RecurrentTransformerBlock(ParamInitializer):
    in_proj: Linear
    attn: GroupedQueryAttention
    attn_input_proj: Linear | None
    mlp: MLP

    @classmethod
    def param_specs(cls, cfg):
        recurrent_input_width = (1 + cfg.num_groups) * cfg.expert_hidden_dim
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
            d_in=recurrent_input_width
            if cfg.direct_qkv_from_input
            else cfg.expert_hidden_dim,
            use_output_gate=cfg.attn_output_gate,
        )
        attn = GroupedQueryAttention.param_specs(attn_cfg)
        attn_input_proj = None
        if cfg.qkv_input_gelu_proj:
            attn_input_proj_cfg = LinearConfig(
                dtype=cfg.dtype,
                in_features=attn_cfg.d_in,
                out_features=attn_cfg.d_in,
                use_bias=False,
                weight_logical_axes=("linear_in", "linear_out"),
            )
            attn_input_proj = Linear.param_specs(attn_input_proj_cfg)
        mlp = MLP.param_specs(cfg.mlp)
        return RecurrentTransformerBlock(
            in_proj=in_proj,
            attn=attn,
            attn_input_proj=attn_input_proj,
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
    aggregation_router: RoutedAggregationRouter | None
    lm_head: Linear
    num_experts: int = dataclasses.field(metadata=dict(static=True))
    memory_aggregation_regime: str = dataclasses.field(metadata=dict(static=True))
    experts_aggregation_regime: str = dataclasses.field(metadata=dict(static=True))
    num_groups: int = dataclasses.field(metadata=dict(static=True))
    res_stream_width: int = dataclasses.field(metadata=dict(static=True))
    attn_head_dim: int = dataclasses.field(metadata=dict(static=True))
    attn_kv_heads: int = dataclasses.field(metadata=dict(static=True))
    memory_len: int = dataclasses.field(metadata=dict(static=True))
    ticks: int = dataclasses.field(metadata=dict(static=True))
    direct_qkv_from_input: bool = dataclasses.field(metadata=dict(static=True))
    soft_routed_experts_aggregation: bool = dataclasses.field(metadata=dict(static=True))
    softmax_expert_routing: bool = dataclasses.field(metadata=dict(static=True))
    checkpoint_token_step: bool = dataclasses.field(metadata=dict(static=True))
    detach_kv_cache_state: bool = dataclasses.field(metadata=dict(static=True))
    segment_local_kv_cache: bool = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        validate_recurrent_model_config(cfg)
        res_stream_width = cfg.expert_hidden_dim * cfg.num_groups
        router = None
        if cfg.soft_routed_experts_aggregation:
            router = RoutedAggregationRouter.param_specs(
                cfg,
                context_width=cfg.expert_hidden_dim + res_stream_width,
                num_experts=cfg.num_experts,
            )
        block = RecurrentTransformerBlock.param_specs(cfg)
        lm_head_cfg = dataclasses.replace(
            cfg.lm_head,
            in_features=res_stream_width,
        )
        return GPT(
            embed=Embedding.param_specs(cfg.embed),
            blocks=_stack_param_specs(block, cfg.num_experts),
            aggregation_router=router,
            lm_head=Linear.param_specs(lm_head_cfg),
            num_experts=cfg.num_experts,
            memory_aggregation_regime=cfg.memory_aggregation_regime,
            experts_aggregation_regime=cfg.experts_aggregation_regime,
            num_groups=cfg.num_groups,
            res_stream_width=res_stream_width,
            attn_head_dim=cfg.expert_hidden_dim // cfg.q_heads,
            attn_kv_heads=cfg.kv_heads,
            memory_len=cfg.memory_len,
            ticks=cfg.ticks,
            direct_qkv_from_input=cfg.direct_qkv_from_input,
            soft_routed_experts_aggregation=cfg.soft_routed_experts_aggregation,
            softmax_expert_routing=cfg.softmax_expert_routing,
            checkpoint_token_step=cfg.checkpoint_token_step,
            detach_kv_cache_state=cfg.detach_kv_cache_state,
            segment_local_kv_cache=cfg.segment_local_kv_cache,
        )

    @classmethod
    def init(cls, key, cfg):
        return cls._init_fn(key, cfg.mesh, cfg.rules, cfg.model)

    def __repr__(self):
        return layer_repr(self)


def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(model))


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
    if cfg.memory_aggregation_regime not in (
        "mean",
        "sum_div_sqrt_num_experts",
    ):
        raise ValueError(
            "`memory_aggregation_regime` must be one of "
            "`{'mean', 'sum_div_sqrt_num_experts'}`."
        )
    if cfg.experts_aggregation_regime not in (
        "mean",
        "sum_div_sqrt_group_size",
    ):
        raise ValueError(
            "`experts_aggregation_regime` must be one of "
            "`{'mean', 'sum_div_sqrt_group_size'}`."
        )
    if cfg.softmax_expert_routing and not (
        cfg.soft_routed_experts_aggregation
    ):
        raise ValueError(
            "`softmax_expert_routing` requires "
            "`soft_routed_experts_aggregation`."
        )
    if getattr(cfg, "segment_local_kv_cache", False) and cfg.ticks != 1:
        raise ValueError("`segment_local_kv_cache` currently requires `ticks=1`.")


def init_recurrent_state(params, batch_size: int):
    dtype = params.embed.weight.dtype
    head_dim = params.attn_head_dim
    kv_heads = params.attn_kv_heads
    # Keep one unused placeholder slot so Orbax can checkpoint the pytree
    # without zero-sized leaves.
    prev_layer_out = jnp.zeros(
        (1, batch_size, params.embed.d_emb),
        dtype=dtype,
    )
    res_stream = jnp.zeros((batch_size, params.res_stream_width), dtype=dtype)
    k_cache = jnp.zeros(
        (batch_size, kv_heads, params.memory_len, head_dim),
        dtype=dtype,
    )
    v_cache = jnp.zeros_like(k_cache)
    cache_fill = jnp.zeros((batch_size,), dtype=jnp.int32)
    abs_pos = jnp.zeros((batch_size,), dtype=jnp.int32)
    return (
        res_stream,
        prev_layer_out,
        k_cache,
        v_cache,
        cache_fill,
        abs_pos,
    )


def _unpack_recurrent_state(params, recurrent_state):
    del params
    res_stream, prev_layer_out = recurrent_state[:2]
    k_cache, v_cache, cache_fill, abs_pos = recurrent_state[2:6]
    return (
        res_stream,
        prev_layer_out,
        k_cache,
        v_cache,
        cache_fill,
        abs_pos,
    )


def _pack_recurrent_state(
    params,
    res_stream,
    prev_layer_out,
    k_cache,
    v_cache,
    cache_fill,
    abs_pos,
):
    del params
    return (res_stream, prev_layer_out, k_cache, v_cache, cache_fill, abs_pos)


def reset_recurrent_state_slots(params, recurrent_state, slot_reset_mask):
    if slot_reset_mask is None:
        return recurrent_state
    reset_mask = jnp.asarray(slot_reset_mask, dtype=jnp.bool_)
    if reset_mask.ndim == 0:
        reset_mask = jnp.broadcast_to(reset_mask[None], (recurrent_state[0].shape[0],))

    (
        res_stream,
        prev_layer_out,
        k_cache,
        v_cache,
        cache_fill,
        abs_pos,
    ) = _unpack_recurrent_state(params, recurrent_state)
    res_stream = jnp.where(reset_mask[:, None], jnp.zeros_like(res_stream), res_stream)
    prev_layer_out = jnp.where(
        reset_mask[None, :, None],
        jnp.zeros_like(prev_layer_out),
        prev_layer_out,
    )
    k_cache = jnp.where(
        reset_mask[:, None, None, None],
        jnp.zeros_like(k_cache),
        k_cache,
    )
    v_cache = jnp.where(
        reset_mask[:, None, None, None],
        jnp.zeros_like(v_cache),
        v_cache,
    )
    cache_fill = jnp.where(reset_mask, jnp.zeros_like(cache_fill), cache_fill)
    abs_pos = jnp.where(reset_mask, jnp.zeros_like(abs_pos), abs_pos)
    return _pack_recurrent_state(
        params,
        res_stream,
        prev_layer_out,
        k_cache,
        v_cache,
        cache_fill,
        abs_pos,
    )


@jax.jit
def reset_kv_cache_state(params, recurrent_state):
    (
        res_stream,
        prev_layer_out,
        k_cache,
        v_cache,
        cache_fill,
        abs_pos,
    ) = _unpack_recurrent_state(params, recurrent_state)
    k_cache = jnp.zeros_like(k_cache)
    v_cache = jnp.zeros_like(v_cache)
    cache_fill = jnp.zeros_like(cache_fill)
    abs_pos = jnp.zeros_like(abs_pos)
    return _pack_recurrent_state(
        params,
        res_stream,
        prev_layer_out,
        k_cache,
        v_cache,
        cache_fill,
        abs_pos,
    )


def _append_to_cache(k_cache, v_cache, k_cur, v_cur, cache_fill):
    memory_len = k_cache.shape[2]
    if memory_len == 0:
        return k_cache, v_cache

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


def _build_attention_cache_sequence(k_cache, v_cache, k_cur, v_cur, cache_fill, write_kv_cache):
    if k_cur.shape[1] > 0 and k_cache.shape[2] > 0:
        def write_cache(_):
            next_k_cache, next_v_cache = _append_to_cache(
                k_cache,
                v_cache,
                k_cur,
                v_cur,
                cache_fill,
            )
            fill_next = jnp.minimum(cache_fill + 1, k_cache.shape[2])
            key_seq = jnp.transpose(next_k_cache, (0, 2, 1, 3))
            value_seq = jnp.transpose(next_v_cache, (0, 2, 1, 3))
            valid_mask = (
                jnp.arange(k_cache.shape[2])[None, None, None, :]
                < fill_next[:, None, None, None]
            )
            return next_k_cache, next_v_cache, key_seq, value_seq, valid_mask

        def read_only_cache(_):
            temp_k_cache, temp_v_cache = _append_to_cache(
                k_cache,
                v_cache,
                k_cur,
                v_cur,
                cache_fill,
            )
            fill_next = jnp.minimum(cache_fill + 1, k_cache.shape[2])
            key_seq = jnp.transpose(temp_k_cache, (0, 2, 1, 3))
            value_seq = jnp.transpose(temp_v_cache, (0, 2, 1, 3))
            valid_mask = (
                jnp.arange(k_cache.shape[2])[None, None, None, :]
                < fill_next[:, None, None, None]
            )
            return k_cache, v_cache, key_seq, value_seq, valid_mask

        return jax.lax.cond(
            write_kv_cache,
            write_cache,
            read_only_cache,
            operand=None,
        )

    key_seq = k_cur[:, None, :, :]
    value_seq = v_cur[:, None, :, :]
    return k_cache, v_cache, key_seq, value_seq, None


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
    if k_cur.shape[1] == 0:
        return local_k_cache, local_v_cache, k_cur[:, None, :, :], v_cur[:, None, :, :], None

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

    if k_cache.shape[2] == 0:
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
    if memory_len == 0 or segment_len == 0:
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
    direct_qkv_from_input,
    freqs,
):
    x_in = jnp.concatenate([token_embed, res_stream], axis=-1)
    x = linear_forward(params.in_proj, x_in)
    attn_in = x_in if direct_qkv_from_input else x
    if params.attn_input_proj is not None:
        attn_in = jax.nn.gelu(linear_forward(params.attn_input_proj, attn_in))
    attn_in = rmsnorm_forward(attn_in)
    sin, cos = freqs

    q = jnp.einsum("bd,dhq->bhq", attn_in, params.attn.wq)
    k_cur = jnp.einsum("bd,dhq->bhq", attn_in, params.attn.wk)
    v_cur = jnp.einsum("bd,dhq->bhq", attn_in, params.attn.wv)
    gate_logits = (
        jnp.einsum("bd,dhq->bhq", attn_in, params.attn.wg)
        if params.attn.wg is not None
        else jnp.zeros_like(v_cur)
    )

    q = rmsnorm_forward(q)
    k_cur = rmsnorm_forward(k_cur)
    q = jnp.squeeze(calculate_rope(q[:, None, :, :], sin, cos), axis=1)
    k_cur = jnp.squeeze(calculate_rope(k_cur[:, None, :, :], sin, cos), axis=1)
    return x, q, k_cur, v_cur, gate_logits


def _finish_recurrent_layer_attention(
    params,
    x,
    q,
    gate_logits,
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
    if params.attn.wg is not None:
        attn = attn * jax.nn.sigmoid(gate_logits.astype(jnp.float32)).astype(attn.dtype)
    attn_out = jnp.einsum("bhq,hqd->bd", attn, params.attn.wo)

    x = x + attn_out
    ffn_out = mlp_forward(params.mlp, rmsnorm_forward(x))
    ungated_out = x + ffn_out
    out = rmsnorm_forward(ungated_out)
    return out, ungated_out


def _logits_from_res_stream(params, res_stream):
    head_in = rmsnorm_forward(res_stream)
    logits = linear_forward(params.lm_head, head_in)
    return 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)


def _res_stream_stats(res_stream):
    res_stream_f32 = res_stream.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(jnp.square(res_stream_f32), axis=-1))
    max_abs = jnp.max(jnp.abs(res_stream_f32), axis=-1)
    return jnp.stack([rms, max_abs], axis=-1)


def _pre_output_activation_l2(res_stream):
    return jnp.mean(jnp.square(res_stream.astype(jnp.float32)), axis=-1)


def _empty_layer_pairwise_stats(num_experts: int):
    return (
        jnp.zeros((num_experts, num_experts), dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def _layer_pairwise_cosine_matrix_sum_and_count(layer_outs):
    num_experts = layer_outs.shape[0]
    if num_experts < 2:
        return _empty_layer_pairwise_stats(num_experts)

    flat = layer_outs.reshape((num_experts, layer_outs.shape[1], -1)).astype(jnp.float32)
    eps = jnp.asarray(1e-8, dtype=flat.dtype)
    unit = flat / jnp.maximum(jnp.linalg.norm(flat, axis=-1, keepdims=True), eps)
    cos_per_batch = jnp.einsum("lbd,mbd->blm", unit, unit)
    return jnp.sum(cos_per_batch, axis=0), jnp.asarray(flat.shape[1], dtype=jnp.float32)


def _compute_router_gates(router, use_softmax, current_embed, res_stream):
    if router is None:
        return None
    router_context_in = jnp.concatenate([current_embed, res_stream], axis=-1)
    router_query = linear_forward(router.context_proj, router_context_in).astype(jnp.float32)
    router_keys = linear_forward(
        router.expert_proj,
        router.expert_embedding,
    ).astype(jnp.float32)
    gate_logits = jnp.einsum("...d,ed->...e", router_query, router_keys) / math.sqrt(
        router.embed_dim
    )
    if use_softmax:
        gates = jax.nn.softmax(gate_logits, axis=-1)
    else:
        gates = jax.nn.sigmoid(gate_logits)
    return jnp.moveaxis(gates, -1, 0)[..., None]


def _routed_aggregation_gates(params, current_embed, res_stream):
    if not params.soft_routed_experts_aggregation:
        return None
    return _compute_router_gates(
        params.aggregation_router,
        params.softmax_expert_routing,
        current_embed,
        res_stream,
    )


def _aggregate_layer_outputs(
    params,
    layer_outs,
    layer_output_gates=None,
):
    out_dtype = layer_outs.dtype

    if params.num_groups == 1:
        if layer_output_gates is None:
            if params.experts_aggregation_regime == "mean":
                aggregated = jnp.mean(layer_outs, axis=0)
            else:
                aggregated = jnp.sum(layer_outs, axis=0) / math.sqrt(params.num_experts)
        else:
            weighted_sum = jnp.sum(layer_output_gates * layer_outs, axis=0)
            if params.experts_aggregation_regime == "mean":
                denom = (
                    jnp.asarray(1e-6, dtype=weighted_sum.dtype)
                    + jnp.sum(layer_output_gates, axis=0)
                )
            else:
                denom = jnp.sqrt(
                    jnp.asarray(1e-6, dtype=weighted_sum.dtype)
                    + jnp.sum(jnp.square(layer_output_gates), axis=0)
                )
            aggregated = weighted_sum / denom
        return aggregated.astype(out_dtype)

    group_size = params.num_experts // params.num_groups
    grouped = layer_outs.reshape(
        (params.num_groups, group_size) + layer_outs.shape[1:]
    )
    grouped = jnp.moveaxis(grouped, 1, 0)
    if layer_output_gates is None:
        if params.experts_aggregation_regime == "mean":
            grouped = jnp.mean(grouped, axis=0)
        else:
            grouped = jnp.sum(grouped, axis=0) / math.sqrt(group_size)
    else:
        grouped_gates = layer_output_gates.reshape(
            (params.num_groups, group_size) + layer_output_gates.shape[1:]
        )
        grouped_gates = jnp.moveaxis(grouped_gates, 1, 0)
        weighted_sum = jnp.sum(grouped_gates * grouped, axis=0)
        if params.experts_aggregation_regime == "mean":
            denom = (
                jnp.asarray(1e-6, dtype=weighted_sum.dtype)
                + jnp.sum(grouped_gates, axis=0)
            )
        else:
            denom = jnp.sqrt(
                jnp.asarray(1e-6, dtype=weighted_sum.dtype)
                + jnp.sum(jnp.square(grouped_gates), axis=0)
            )
        grouped = weighted_sum / denom
    grouped = jnp.moveaxis(grouped, 0, -1)
    feature_dim = grouped.shape[-2]
    aggregated = grouped.reshape(
        grouped.shape[:-2] + (feature_dim * params.num_groups,)
    )
    return aggregated.astype(out_dtype)


def _aggregate_shared_kv_candidates(params, kv_stack):
    kv_stack_f32 = kv_stack.astype(jnp.float32)
    if params.memory_aggregation_regime == "mean":
        merged = jnp.mean(kv_stack_f32, axis=0)
    elif params.memory_aggregation_regime == "sum_div_sqrt_num_experts":
        merged = jnp.sum(kv_stack_f32, axis=0) / math.sqrt(params.num_experts)
    else:
        raise ValueError(
            "Unsupported memory aggregation regime: "
            f"{params.memory_aggregation_regime}"
        )
    return merged.astype(kv_stack.dtype)


def _merge_layer_kv_candidates(params, k_stack, v_stack):
    merged_k = _aggregate_shared_kv_candidates(params, k_stack)
    merged_v = _aggregate_shared_kv_candidates(params, v_stack)
    return merged_k, merged_v


def _run_shared_kv_recurrent_cell(params, current_embed, recurrent_state, write_kv_cache):
    (
        res_stream,
        prev_layer_out,
        k_cache,
        v_cache,
        cache_fill,
        abs_pos,
    ) = _unpack_recurrent_state(params, recurrent_state)
    if params.detach_kv_cache_state:
        k_cache = jax.lax.stop_gradient(k_cache)
        v_cache = jax.lax.stop_gradient(v_cache)
    aggregation_gates = _routed_aggregation_gates(
        params,
        current_embed,
        res_stream,
    )
    step_freqs = precompute_frequencies(
        abs_pos[:, None],
        features=params.blocks.attn.head_dim,
        dtype=current_embed.dtype,
    )
    x_stack, q_stack, k_stack, v_stack, gate_logits_stack = jax.vmap(
        _prepare_recurrent_layer_attention,
        in_axes=(0, None, None, None, None),
        out_axes=(0, 0, 0, 0, 0),
    )(
        params.blocks,
        current_embed,
        res_stream,
        params.direct_qkv_from_input,
        step_freqs,
    )
    merged_k, merged_v = _merge_layer_kv_candidates(params, k_stack, v_stack)
    next_k_cache, next_v_cache, key_seq, value_seq, valid_mask = _build_attention_cache_sequence(
        k_cache,
        v_cache,
        merged_k,
        merged_v,
        cache_fill,
        write_kv_cache,
    )
    if params.detach_kv_cache_state:
        next_k_cache = jax.lax.stop_gradient(next_k_cache)
        next_v_cache = jax.lax.stop_gradient(next_v_cache)
        key_seq = jax.lax.stop_gradient(key_seq)
        value_seq = jax.lax.stop_gradient(value_seq)
    layer_outs, _ = jax.vmap(
        _finish_recurrent_layer_attention,
        in_axes=(0, 0, 0, 0, None, None, None),
        out_axes=(0, 0),
    )(
        params.blocks,
        x_stack,
        q_stack,
        gate_logits_stack,
        key_seq,
        value_seq,
        valid_mask,
    )
    res_stream = _aggregate_layer_outputs(
        params,
        layer_outs,
        aggregation_gates,
    )
    layer_pairwise_stats = _layer_pairwise_cosine_matrix_sum_and_count(layer_outs)
    write_kv_cache = jnp.asarray(write_kv_cache)
    if params.memory_len > 0:
        cache_fill = jnp.minimum(
            cache_fill + write_kv_cache.astype(cache_fill.dtype),
            params.memory_len,
        )
    abs_pos = abs_pos + write_kv_cache.astype(abs_pos.dtype)
    return (
        _pack_recurrent_state(
            params,
            res_stream,
            prev_layer_out,
            next_k_cache,
            next_v_cache,
            cache_fill,
            abs_pos,
        ),
        layer_pairwise_stats,
    )


def _run_recurrent_cell(params, current_embed, recurrent_state, write_kv_cache):
    return _run_shared_kv_recurrent_cell(
        params,
        current_embed,
        recurrent_state,
        write_kv_cache,
    )


def _run_token_steps(params, recurrent_state, token_embed):
    def recurrent_body(step_idx, loop_state):
        loop_state, layer_pairwise_matrix_sum, layer_pairwise_count = loop_state
        current_embed = token_embed
        write_kv_cache = step_idx == (params.ticks - 1)
        loop_state, layer_pairwise_stats = _run_recurrent_cell(
            params,
            current_embed,
            loop_state,
            write_kv_cache,
        )
        step_pairwise_matrix_sum, step_pairwise_count = layer_pairwise_stats
        return (
            loop_state,
            layer_pairwise_matrix_sum + step_pairwise_matrix_sum,
            layer_pairwise_count + step_pairwise_count,
        )

    return jax.lax.fori_loop(
        0,
        params.ticks,
        recurrent_body,
        (recurrent_state, *_empty_layer_pairwise_stats(params.num_experts)),
    )


def _token_step(params, carry, token_ids):
    token_embed = embedding_forward(params.embed, token_ids)
    carry, layer_pairwise_matrix_sum, layer_pairwise_count = _run_token_steps(
        params,
        carry,
        token_embed,
    )
    res_stream = carry[0]
    return carry, (
        _logits_from_res_stream(params, res_stream),
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    )


def forward_with_state(params, x, recurrent_state, slot_reset_mask=None):
    recurrent_state = reset_recurrent_state_slots(params, recurrent_state, slot_reset_mask)
    def token_step_logits_only(carry, token_ids):
        carry, (logits, _, _) = _token_step(params, carry, token_ids)
        return carry, logits

    token_step_impl = (
        jax.checkpoint(token_step_logits_only)
        if params.checkpoint_token_step
        else token_step_logits_only
    )

    final_state, logits = jax.lax.scan(
        token_step_impl,
        recurrent_state,
        jnp.swapaxes(x, 0, 1),
        _split_transpose=params.checkpoint_token_step,
    )
    return jnp.swapaxes(logits, 0, 1), final_state


def forward(params, x, segment_ids, freqs):
    del segment_ids, freqs
    batch_size = x.shape[0]
    logits, _ = forward_with_state(
        params,
        x,
        init_recurrent_state(params, batch_size),
        slot_reset_mask=None,
    )
    return logits


def forward_loss_with_state(
    params,
    x,
    y,
    recurrent_state,
    slot_reset_mask=None,
    loss_mask=None,
    pre_output_reg_cost=0.0,
):
    loss, final_state, _, _, _ = forward_loss_with_state_and_stats(
        params,
        x,
        y,
        recurrent_state,
        slot_reset_mask=slot_reset_mask,
        loss_mask=loss_mask,
        pre_output_reg_cost=pre_output_reg_cost,
    )
    return loss, final_state


def forward_loss_with_state_and_stats(
    params,
    x,
    y,
    recurrent_state,
    slot_reset_mask=None,
    loss_mask=None,
    pre_output_reg_cost=0.0,
):
    recurrent_state = reset_recurrent_state_slots(params, recurrent_state, slot_reset_mask)
    token_step_impl = (
        jax.checkpoint(
            lambda carry, token_ids: _token_step(params, carry, token_ids)
        )
        if params.checkpoint_token_step
        else lambda carry, token_ids: _token_step(params, carry, token_ids)
    )
    x_tokens = jnp.swapaxes(x, 0, 1)
    y_tokens = jnp.swapaxes(y, 0, 1)
    mask_tokens = None if loss_mask is None else jnp.swapaxes(loss_mask, 0, 1)

    def loss_step(carry, inputs):
        (
            recurrent_state,
            ce_loss_sum,
            reg_loss_sum,
            weight_sum,
            layer_pairwise_matrix_sum,
            layer_pairwise_count,
        ) = carry
        if mask_tokens is None:
            token_ids, target_ids = inputs
            token_mask = jnp.ones_like(target_ids, dtype=jnp.float32)
        else:
            token_ids, target_ids, token_mask = inputs
            token_mask = token_mask.astype(jnp.float32)

        recurrent_state, (
            logits,
            token_pairwise_matrix_sum,
            token_pairwise_count,
        ) = token_step_impl(recurrent_state, token_ids)
        res_stats = _res_stream_stats(recurrent_state[0])
        token_reg_loss = _pre_output_activation_l2(recurrent_state[0])
        token_log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_loss = -jnp.take_along_axis(
            token_log_probs,
            target_ids[:, None],
            axis=-1,
        ).squeeze(-1)
        ce_loss_sum = ce_loss_sum + jnp.sum(token_loss * token_mask)
        reg_loss_sum = reg_loss_sum + jnp.sum(token_reg_loss * token_mask)
        weight_sum = weight_sum + jnp.sum(token_mask)
        layer_pairwise_matrix_sum = (
            layer_pairwise_matrix_sum + token_pairwise_matrix_sum
        )
        layer_pairwise_count = layer_pairwise_count + token_pairwise_count
        return (
            (
                recurrent_state,
                ce_loss_sum,
                reg_loss_sum,
                weight_sum,
                layer_pairwise_matrix_sum,
                layer_pairwise_count,
            ),
            res_stats,
        )

    scan_inputs = (
        (x_tokens, y_tokens)
        if mask_tokens is None
        else (x_tokens, y_tokens, mask_tokens)
    )
    carry0 = (
        recurrent_state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros((params.num_experts, params.num_experts), dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )
    (
        final_state,
        ce_loss_sum,
        reg_loss_sum,
        weight_sum,
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    ), res_stats_seq = jax.lax.scan(loss_step, carry0, scan_inputs, _split_transpose=params.checkpoint_token_step)
    ce_loss = ce_loss_sum / jnp.maximum(weight_sum, 1.0)
    reg_loss = reg_loss_sum / jnp.maximum(weight_sum, 1.0)
    loss = ce_loss + pre_output_reg_cost * reg_loss
    return (
        loss,
        final_state,
        res_stats_seq,
        jnp.stack([ce_loss, reg_loss]),
        (layer_pairwise_matrix_sum, layer_pairwise_count),
    )


def forward_loss(
    params,
    x,
    y,
    segment_ids=None,
    freqs=None,
    loss_mask=None,
    pre_output_reg_cost=0.0,
):
    del segment_ids, freqs
    batch_size = x.shape[0]
    loss, _ = forward_loss_with_state(
        params,
        x,
        y,
        init_recurrent_state(params, batch_size),
        slot_reset_mask=None,
        loss_mask=loss_mask,
        pre_output_reg_cost=pre_output_reg_cost,
    )
    return loss


def forward_v2(params, x, segment_ids, cache, head_dim):
    del params, x, segment_ids, cache, head_dim
    raise NotImplementedError("`recnanogpt` inference is not implemented yet.")
