import dataclasses
import math

import jax
import jax.numpy as jnp

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
class MoEStage(ParamInitializer):
    experts: base_model.TransformerBlock
    num_experts: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        block = base_model.TransformerBlock.param_specs(cfg)
        return MoEStage(
            experts=_stack_param_specs(block, cfg.num_experts),
            num_experts=cfg.num_experts,
        )

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class GPT(ParamInitializer):
    embed: Embedding
    stages: list[MoEStage]
    lm_head: Linear
    num_experts: int = dataclasses.field(metadata=dict(static=True))
    experts_aggregation_regime: str = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        validate_moe_config(cfg)
        return GPT(
            embed=Embedding.param_specs(cfg.embed),
            stages=[
                MoEStage.param_specs(cfg) for _ in range(cfg.num_layers)
            ],
            lm_head=Linear.param_specs(cfg.lm_head),
            num_experts=cfg.num_experts,
            experts_aggregation_regime=cfg.experts_aggregation_regime,
        )

    @classmethod
    def init(cls, key, cfg):
        return cls._init_fn(key, cfg.mesh, cfg.rules, cfg.model)

    def __repr__(self):
        return layer_repr(self)


def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(model))


def validate_moe_config(cfg):
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


def _aggregate_expert_outputs(params, expert_outputs):
    if params.experts_aggregation_regime == "mean":
        return jnp.mean(expert_outputs, axis=0).astype(expert_outputs.dtype)
    if params.experts_aggregation_regime == "sum_div_sqrt_num_experts":
        return (
            jnp.sum(expert_outputs, axis=0) / math.sqrt(params.num_experts)
        ).astype(expert_outputs.dtype)
    raise ValueError(
        "Unsupported experts aggregation regime: "
        f"{params.experts_aggregation_regime}"
    )


def _attn_forward_no_cudnn(params, x, mask, freqs):
    orig_dtype = x.dtype
    sin, cos = freqs

    with jax.named_scope("qkv_matmul"):
        q = jnp.einsum("btd, dhq -> bthq", x, params.wq)
        k = jnp.einsum("btd, dhq -> bthq", x, params.wk)
        v = jnp.einsum("btd, dhq -> bthq", x, params.wv)

    with jax.named_scope("qk_norm"):
        q = base_model.rmsnorm_forward(q)
        k = base_model.rmsnorm_forward(k)

    with jax.named_scope("rope"):
        q = base_model.calculate_rope(q, sin, cos)
        k = base_model.calculate_rope(k, sin, cos)

    with jax.named_scope("attention"):
        scale = 1.0 / math.sqrt(q.shape[-1])
        if mask is not None:
            attn = jax.nn.dot_product_attention(
                q,
                k,
                v,
                mask=mask,
                scale=scale,
                is_causal=True,
                implementation=None,
            ).astype(orig_dtype)
        else:
            attn = jax.nn.dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                is_causal=True,
                implementation=None,
            ).astype(orig_dtype)

    with jax.named_scope("projection"):
        return jnp.einsum("bthq, hqd->btd", attn, params.wo)


def _block_forward_no_cudnn(
    params,
    x,
    mask,
    freqs,
):
    with jax.named_scope("pre_attn_norm"):
        attn_in = base_model.rmsnorm_forward(x)

    attn_out = _attn_forward_no_cudnn(params.attn, attn_in, mask, freqs)

    with jax.named_scope("residual"):
        x = x + attn_out

    with jax.named_scope("pre_ffn_norm"):
        ffn_in = base_model.rmsnorm_forward(x)

    with jax.named_scope("ffn"):
        ffn_out = base_model.mlp_forward(params.mlp, ffn_in)

    with jax.named_scope("residual"):
        return x + ffn_out


def _squeeze_singleton_expert(block):
    return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), block)


def stage_forward(params, stage, x, mask, freqs):
    if params.num_experts == 1:
        return base_model.block_forward(
            _squeeze_singleton_expert(stage.experts),
            x,
            mask,
            freqs,
        )
    expert_outputs = jax.vmap(
        lambda expert_params: _block_forward_no_cudnn(
            expert_params,
            x,
            mask,
            freqs,
        ),
        in_axes=0,
        out_axes=0,
    )(stage.experts)
    return _aggregate_expert_outputs(params, expert_outputs)


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
