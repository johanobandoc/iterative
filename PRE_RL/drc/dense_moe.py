import jax
import jax.numpy as jnp
from flax import linen as nn

from expert_aggregation import SumExpertsAggregator

class MoEDense(nn.Module):
    """Mixture-of-experts dense block for vector inputs."""

    expert_hidden_dim: int
    num_experts: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        VmappedDense = nn.vmap(
            DenseGLUCell,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_experts,
        )
        expert_dense = VmappedDense(
            features=self.expert_hidden_dim,
            use_bias=self.use_bias,
        )
        expert_outputs = expert_dense(x)
        expert_aggregator = SumExpertsAggregator()
        return expert_aggregator(expert_outputs)


class DenseGLUCell(nn.Module):
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = nn.Dense(self.features * 2, use_bias=self.use_bias)(x)
        a, b = jnp.split(out, 2, axis=-1)
        out = a * jax.nn.sigmoid(b)
        return out
