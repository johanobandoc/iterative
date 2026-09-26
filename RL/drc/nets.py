from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from encoders import build_observation_encoder
from recurrent_moe import RecurrentMoE
from net_utils import PreheadAggregation


class DenseGLU(nn.Module):
    """Single dense GLU shared by the policy and value projections."""

    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        value, gate = jnp.split(nn.Dense(2 * self.features)(x), 2, axis=-1)
        return value * jax.nn.sigmoid(gate)


class ActorCritic(nn.Module):
    """Parallel recurrent expert policy/value network."""

    args: Any

    def _build_recurrent_core(self) -> RecurrentMoE:
        return RecurrentMoE(
            expert_type=self.args.expert_type,
            num_experts=self.args.num_experts,
            expert_hidden_dim=self.args.expert_hidden_dim,
            depth=self.args.depth,
            reset_hidden_every_step=self.args.reset_hidden_every_step,
            name="recurrent_core",
        )

    @nn.compact
    def initial_hidden(self, x: jnp.ndarray) -> Any:
        x_enc = build_observation_encoder(self.args)(x)
        return self._build_recurrent_core().initial_hidden(x_enc)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        hidden: Any = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, Any]:
        x_enc = build_observation_encoder(self.args)(x)
        cell_out, hidden = self._build_recurrent_core()(x_enc, hidden)
        flat = PreheadAggregation(args=self.args)(cell_out)
        mid = DenseGLU(features=self.args.head_hidden_dim, name="head_core")(flat)
        logits = nn.Dense(self.args.action_dim)(mid)
        value = nn.Dense(1)(mid)
        return logits, value.squeeze(-1), hidden
