from typing import Any
import jax.numpy as jnp
from flax import linen as nn


def init_conv_hidden(
    x: jnp.ndarray,
    expert_type: str,
    expert_hidden_dim: int,
    num_experts: int,
    depth: int = 1,
) -> Any:

    dtype = jnp.float32

    if x.ndim < 3:
        return init_dense_hidden(
            x,
            expert_type=expert_type,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            depth=depth,
        )

    b, h, w = x.shape[:3]

    if expert_type == "stacked_lstm":
        res_stream = jnp.zeros((b, h, w, expert_hidden_dim), dtype=dtype)
        h0 = jnp.zeros((num_experts, b, depth, h, w, expert_hidden_dim), dtype=dtype)
        c0 = jnp.zeros_like(h0)
        return res_stream, h0, c0


def init_dense_hidden(
    x: jnp.ndarray,
    expert_type: str,
    expert_hidden_dim: int,
    num_experts: int,
    depth: int = 1,
) -> Any:

    dtype = jnp.float32

    if expert_type == "stacked_dense_lstm":
        b = x.shape[0]
        res_stream = jnp.zeros((b, expert_hidden_dim), dtype=dtype)
        h0 = jnp.zeros((num_experts, b, depth, expert_hidden_dim), dtype=dtype)
        c0 = jnp.zeros_like(h0)
        return res_stream, h0, c0
    return tuple()


def init_recurrent_hidden(
    x: jnp.ndarray,
    expert_type: str,
    expert_hidden_dim: int,
    num_experts: int,
    depth: int = 1,
) -> Any:
    if expert_type in ("stacked_dense_lstm", "glu", "none"):
        return init_dense_hidden(
            x,
            expert_type=expert_type,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            depth=depth,
        )

    return init_conv_hidden(
        x,
        expert_type=expert_type,
        expert_hidden_dim=expert_hidden_dim,
        num_experts=num_experts,
        depth=depth,
    )


class PreheadAggregation(nn.Module):
    args: Any

    @nn.compact
    def __call__(self, cell_out: jnp.ndarray, x_enc: jnp.ndarray) -> jnp.ndarray:
        if self.args.prehead_aggregation == "attn":
            B, H, W, C = cell_out.shape
            attn_logits = nn.Conv(1, (1, 1), padding='SAME', kernel_init=nn.initializers.xavier_uniform())(cell_out).reshape((B, H * W))
            attn_w = nn.softmax(attn_logits, axis=-1).reshape((B, H * W, 1))
            tokens = cell_out.reshape((B, H * W, C))
            attn_vec = (attn_w * tokens).sum(axis=1)
            gap_vec = tokens.mean(axis=1)
            core_out = jnp.concatenate([attn_vec, gap_vec], axis=-1)
        if self.args.prehead_aggregation == "enc_cell_flatten":
            B = cell_out.shape[0]
            core_out = jnp.concatenate([x_enc, cell_out], axis=-1)

        return core_out.reshape((B, -1))
