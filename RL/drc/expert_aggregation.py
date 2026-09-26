import jax.numpy as jnp
from flax import linen as nn


class SumExpertsAggregator(nn.Module):
    def __call__(self, expert_outputs: jnp.ndarray) -> jnp.ndarray:
        num_experts = expert_outputs.shape[0]
        denom = jnp.sqrt(jnp.asarray(num_experts, dtype=expert_outputs.dtype))
        return jnp.sum(expert_outputs, axis=0) / denom
