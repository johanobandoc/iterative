import dataclasses
import math

import jax
import jax.numpy as jnp

from config import LinearConfig
import model as base_model
from layers import Embedding, Linear
from utils import ParamInitializer
from utils import ParamSpec
from utils import is_param_spec
from utils import jax_pytree_struct
from utils import layer_repr


def _stack_param_specs(tree, axis_size: int):
    def stack_spec(spec):
        initializer = spec.initializer
        if axis_size == 1:
            base_initializer = initializer

            def singleton_axis_initializer(key, shape, dtype):
                return base_initializer(key, shape[1:], dtype)[None, ...]

            initializer = singleton_axis_initializer
        return ParamSpec(
            shape=(axis_size,) + spec.shape,
            dtype=spec.dtype,
            logical_axes=(None,) + spec.logical_axes,
            initializer=initializer,
        )

    return jax.tree_util.tree_map(
        stack_spec,
        tree,
        is_leaf=is_param_spec,
    )


@jax_pytree_struct
class SharedKVMoEStage(ParamInitializer):
    experts: base_model.TransformerBlock
    attn_input_proj: Linear | None
    num_experts: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        block = base_model.TransformerBlock.param_specs(cfg)
        attn_input_proj = None
        if cfg.qkv_input_gelu_proj:
            attn_input_proj_cfg = LinearConfig(
                dtype=cfg.dtype,
                in_features=cfg.d_emb,
                out_features=cfg.d_emb,
                use_bias=False,
                weight_logical_axes=("linear_in", "linear_out"),
            )
            attn_input_proj = _stack_param_specs(
                Linear.param_specs(attn_input_proj_cfg),
                cfg.num_experts,
            )
        return SharedKVMoEStage(
            experts=_stack_param_specs(block, cfg.num_experts),
            attn_input_proj=attn_input_proj,
            num_experts=cfg.num_experts,
        )

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class GPT(ParamInitializer):
    embed: Embedding
    stages: list[SharedKVMoEStage]
    lm_head: Linear
    num_experts: int = dataclasses.field(metadata=dict(static=True))
    experts_aggregation_regime: str = dataclasses.field(metadata=dict(static=True))
    qkv_input_gelu_proj: bool = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        validate_shared_kv_config(cfg)
        return GPT(
            embed=Embedding.param_specs(cfg.embed),
            stages=[
                SharedKVMoEStage.param_specs(cfg) for _ in range(cfg.num_layers)
            ],
            lm_head=Linear.param_specs(cfg.lm_head),
            num_experts=cfg.num_experts,
            experts_aggregation_regime=cfg.experts_aggregation_regime,
            qkv_input_gelu_proj=cfg.qkv_input_gelu_proj,
        )

    @classmethod
    def init(cls, key, cfg):
        return cls._init_fn(key, cfg.mesh, cfg.rules, cfg.model)

    def __repr__(self):
        return layer_repr(self)


def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(model))


def validate_shared_kv_config(cfg):
    if cfg.num_experts < 1:
        raise ValueError("`num_experts` must be >= 1.")
    if cfg.q_heads < 1:
        raise ValueError("`q_heads` must be >= 1.")
    if cfg.kv_heads < 1:
        raise ValueError("`kv_heads` must be >= 1.")
    if cfg.experts_aggregation_regime not in (
        "mean",
        "sum_div_sqrt_num_experts",
    ):
        raise ValueError(
            "`experts_aggregation_regime` must be one of "
            "`{'mean', 'sum_div_sqrt_num_experts'}`."
        )
    if cfg.d_emb % cfg.q_heads != 0:
        raise ValueError(
            f"`d_emb` must be divisible by `q_heads`, got {cfg.d_emb} and {cfg.q_heads}."
        )
    if cfg.q_heads % cfg.kv_heads != 0:
        raise ValueError(
            f"`q_heads` must be divisible by `kv_heads`, got {cfg.q_heads} and {cfg.kv_heads}."
        )


def _aggregate_expert_axis(params, expert_values):
    if params.experts_aggregation_regime == "mean":
        return jnp.mean(expert_values, axis=0).astype(expert_values.dtype)
    if params.experts_aggregation_regime == "sum_div_sqrt_num_experts":
        return (
            jnp.sum(expert_values, axis=0) / math.sqrt(params.num_experts)
        ).astype(expert_values.dtype)
    raise ValueError(
        "Unsupported experts aggregation regime: "
        f"{params.experts_aggregation_regime}"
    )


def _stacked_linear_forward(params, x):
    out = jnp.einsum("ebtd,edv->ebtv", x, params.weight)
    if params.bias is not None:
        return out + params.bias[:, None, None, :]
    return out


def _stacked_mlp_forward(params, x):
    x = _stacked_linear_forward(params.fc1, x)
    x = jnp.square(jax.nn.relu(x))
    return _stacked_linear_forward(params.fc2, x)


def _squeeze_singleton_expert(block):
    return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), block)


def _rope_expert_axis(x, sin, cos):
    return jax.vmap(
        base_model.calculate_rope,
        in_axes=(0, None, None),
        out_axes=0,
    )(x, sin, cos)


def _shared_kv_attn_forward(params, stage_params, x, mask, freqs):
    orig_dtype = x.dtype
    sin, cos = freqs
    expert_params = stage_params.experts
    attn_params = expert_params.attn

    if stage_params.attn_input_proj is not None:
        x = jnp.broadcast_to(x[None, ...], (params.num_experts,) + x.shape)
        x = jax.nn.gelu(_stacked_linear_forward(stage_params.attn_input_proj, x))
        x = base_model.rmsnorm_forward(x)

    with jax.named_scope("qkv_matmul"):
        if x.ndim == 4:
            q = jnp.einsum("ebtd,edhq->ebthq", x, attn_params.wq)
            k = jnp.einsum("ebtd,edhq->ebthq", x, attn_params.wk)
            v = jnp.einsum("ebtd,edhq->ebthq", x, attn_params.wv)
        else:
            q = jnp.einsum("btd,edhq->ebthq", x, attn_params.wq)
            k = jnp.einsum("btd,edhq->ebthq", x, attn_params.wk)
            v = jnp.einsum("btd,edhq->ebthq", x, attn_params.wv)

    with jax.named_scope("qk_norm"):
        q = base_model.rmsnorm_forward(q)
        k = base_model.rmsnorm_forward(k)

    with jax.named_scope("rope"):
        q = _rope_expert_axis(q, sin, cos)
        k = _rope_expert_axis(k, sin, cos)

    with jax.named_scope("merge_kv"):
        k_shared = _aggregate_expert_axis(params, k)
        v_shared = _aggregate_expert_axis(params, v)

    with jax.named_scope("attention"):
        scale = 1.0 / math.sqrt(q.shape[-1])
        num_experts, batch_size = q.shape[:2]
        q_flat = jnp.reshape(q, (num_experts * batch_size,) + q.shape[2:])
        k_flat = jnp.reshape(
            jnp.broadcast_to(k_shared[None, ...], k.shape),
            (num_experts * batch_size,) + k_shared.shape[1:],
        )
        v_flat = jnp.reshape(
            jnp.broadcast_to(v_shared[None, ...], v.shape),
            (num_experts * batch_size,) + v_shared.shape[1:],
        )
        if mask is not None:
            mask_flat = jnp.reshape(
                jnp.broadcast_to(mask[None, ...], (num_experts,) + mask.shape),
                (num_experts * batch_size,) + mask.shape[1:],
            )
            attn = jax.nn.dot_product_attention(
                q_flat,
                k_flat,
                v_flat,
                mask=mask_flat,
                scale=scale,
                is_causal=True,
                implementation=None,
            ).astype(orig_dtype)
        else:
            attn = jax.nn.dot_product_attention(
                q_flat,
                k_flat,
                v_flat,
                scale=scale,
                is_causal=True,
                implementation=None,
            ).astype(orig_dtype)
        attn = jnp.reshape(attn, q.shape)

    with jax.named_scope("projection"):
        return jnp.einsum("ebthq,ehqd->ebtd", attn, attn_params.wo)


def _shared_kv_block_forward(params, stage_params, x, mask, freqs):
    with jax.named_scope("pre_attn_norm"):
        attn_in = base_model.rmsnorm_forward(x)

    attn_out = _shared_kv_attn_forward(params, stage_params, attn_in, mask, freqs)

    with jax.named_scope("residual"):
        expert_x = x[None, ...] + attn_out

    with jax.named_scope("pre_ffn_norm"):
        ffn_in = base_model.rmsnorm_forward(expert_x)

    with jax.named_scope("ffn"):
        ffn_out = _stacked_mlp_forward(stage_params.experts.mlp, ffn_in)

    with jax.named_scope("residual"):
        expert_x = expert_x + ffn_out

    return _aggregate_expert_axis(params, expert_x)


def stage_forward(params, stage, x, mask, freqs):
    if params.num_experts == 1 and stage.attn_input_proj is None:
        return base_model.block_forward(
            _squeeze_singleton_expert(stage.experts),
            x,
            mask,
            freqs,
        )
    return _shared_kv_block_forward(params, stage, x, mask, freqs)


def forward(params, x, segment_ids, freqs):
    if segment_ids is not None:
        with jax.named_scope("compute_mask"):
            mask = base_model.compute_segment_mask(segment_ids)
    else:
        mask = None

    with jax.named_scope("embedding"):
        x = base_model.embedding_forward(params.embed, x)

    for stage in params.stages:
        x = stage_forward(params, stage, x, mask, freqs)

    with jax.named_scope("norm"):
        x = base_model.rmsnorm_forward(x)

    with jax.named_scope("unembed"):
        logits = base_model.linear_forward(params.lm_head, x)

    with jax.named_scope("logit_soft_capping"):
        logits = logits.astype(jnp.float32)
        logits = 15.0 * jnp.tanh(logits / 15.0)
    return logits
