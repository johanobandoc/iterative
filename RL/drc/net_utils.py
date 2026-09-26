from typing import Any

import jax.numpy as jnp
from flax import linen as nn


def init_recurrent_hidden(
    x: jnp.ndarray,
    expert_type: str,
    expert_hidden_dim: int,
    num_experts: int,
    depth: int = 1,
) -> Any:
    batch, height, width = x.shape[:3]
    res_stream = jnp.zeros((batch, height, width, expert_hidden_dim), dtype=jnp.float32)
    if expert_type == "rec_conv_glu":
        return res_stream
    if expert_type == "stacked_lstm":
        h = jnp.zeros(
            (num_experts, batch, depth, height, width, expert_hidden_dim),
            dtype=jnp.float32,
        )
        return res_stream, h, jnp.zeros_like(h)
    raise ValueError(f"Unsupported Sokoban expert type: {expert_type}")


class PreheadAggregation(nn.Module):
    args: Any

    @nn.compact
    def __call__(self, cell_out: jnp.ndarray) -> jnp.ndarray:
        if self.args.prehead_aggregation == "attn":
            B, H, W, C = cell_out.shape
            attn_logits = nn.Conv(1, (1, 1), padding='SAME', kernel_init=nn.initializers.xavier_uniform())(cell_out).reshape((B, H * W))
            attn_w = nn.softmax(attn_logits, axis=-1).reshape((B, H * W, 1))
            tokens = cell_out.reshape((B, H * W, C))
            attn_vec = (attn_w * tokens).sum(axis=1)
            gap_vec = tokens.mean(axis=1)
            core_out = jnp.concatenate([attn_vec, gap_vec], axis=-1)

        return core_out.reshape((B, -1))
