import math

import jax
import jax.numpy as jnp

from utils import layer_repr
from utils import ParamInitializer
from utils import jax_pytree_struct
from layers import Linear, GroupedQueryAttention


if jax.default_backend() == "gpu":
    ATTN_IMPL = "cudnn"

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
class TransformerBlock(ParamInitializer):
    attn: GroupedQueryAttention
    mlp: MLP

    @classmethod
    def param_specs(cls, cfg):
        attn = GroupedQueryAttention.param_specs(cfg.attn)
        mlp = MLP.param_specs(cfg.mlp)
        return TransformerBlock(attn=attn, mlp=mlp)

    def __repr__(self):
        return layer_repr(self)


def precompute_frequencies(
    positions: jax.Array, features: int, theta=10000.0, dtype=None
):
    """Generate Sin/Cos for Rotary Embeddings."""
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


def rmsnorm_forward(x, eps=1e-5):
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x / scale).astype(orig_dtype)


def linear_forward(params, x):
    out = jnp.einsum("...d, dv-> ...v", x, params.weight)
    if params.bias is not None:
        return out + params.bias
    else:
        return out


def mlp_forward(params, x):
    x = linear_forward(params.fc1, x)
    x = jnp.square(jax.nn.relu(x))
    x = linear_forward(params.fc2, x)
    return x



#################################### For training ########################################


def attn_forward(params, x, mask, freqs):
    orig_dtype = x.dtype
    sin, cos = freqs

    with jax.named_scope("qkv_matmul"):
        q = jnp.einsum("btd, dhq -> bthq", x, params.wq)
        k = jnp.einsum("btd, dhq -> bthq", x, params.wk)
        v = jnp.einsum("btd, dhq -> bthq", x, params.wv)

    with jax.named_scope("qk_norm"):
        q = rmsnorm_forward(q)
        k = rmsnorm_forward(k)

    with jax.named_scope("rope"):
        q = calculate_rope(q, sin, cos)
        k = calculate_rope(k, sin, cos)

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
                implementation=ATTN_IMPL,
            ).astype(orig_dtype)
        else:
            attn = jax.nn.dot_product_attention(
                q, k, v, scale=scale, is_causal=True, implementation=ATTN_IMPL
            ).astype(orig_dtype)

    with jax.named_scope("projection"):
        out = jnp.einsum("bthq, hqd->btd", attn, params.wo)
    return out


def block_forward(
    params,
    x,
    mask,
    freqs,
):
    with jax.named_scope("pre_attn_norm"):
        attn_in = rmsnorm_forward(x)

    attn_out = attn_forward(params.attn, attn_in, mask, freqs)

    with jax.named_scope("residual"):
        x = x + attn_out

    with jax.named_scope("pre_ffn_norm"):
        ffn_in = rmsnorm_forward(x)

    with jax.named_scope("ffn"):
        ffn_out = mlp_forward(params.mlp, ffn_in)

    with jax.named_scope("residual"):
        return x + ffn_out


def compute_segment_mask(segment_ids):
    """Compute once, reuse across all layers. Returns (B, 1, T, S) bias or None."""
    if segment_ids is None:
        return None

    # (B, T) valid token positions (segment_id != 0)
    # We discard padding tokens as valid tokens.
    valid_segment_ids = jnp.where(segment_ids != 0, 1, 0)

    # (B, T, T) same segment
    same_segment = jnp.equal(segment_ids[:, :, None], segment_ids[:, None, :])

    # (B, T, T) valid on both query and key axes
    valid = valid_segment_ids[:, :, None] & valid_segment_ids[:, None, :]

    mask = (same_segment & valid).astype(jnp.bool)
    return mask[:, None, :, :]  # (B, 1, T, T)
