from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from encoders import build_observation_encoder
from recurrent_moe import RecurrentMoE
from dense_moe import MoEDense
from net_utils import PreheadAggregation


def transform_obs(x: jnp.ndarray, env_id: str) -> jnp.ndarray:
    """Normalize observations to float32 channels-last."""
    env_id_lower = env_id.lower()

    def _scale_pixels(obs: jnp.ndarray) -> jnp.ndarray:
        if jnp.issubdtype(obs.dtype, jnp.integer):
            return obs.astype(jnp.float32) / 255.0
        return obs.astype(jnp.float32)

    if "sokoban" in env_id_lower:
        return x
    if 'craftax' in env_id_lower and 'symbolic' in env_id_lower:
        return x.astype(jnp.float32)
    if "craftax" in env_id_lower and 'pixels' in env_id_lower:
        return _scale_pixels(x)
    if "minatar" in env_id_lower:
        return x.astype(jnp.float32)
    if "minigrid" in env_id_lower:
        return _scale_pixels(x)

    return _scale_pixels(x)


class SharedMoECore(nn.Module):
    """Shared MoE core for both recurrent and head stages."""

    expert_type: str
    expert_hidden_dim: int
    num_experts: int
    depth: int = 1
    ticks: int = 1

    def _build_recurrent_core(self) -> RecurrentMoE:
        return RecurrentMoE(
            expert_type=self.expert_type,
            num_experts=self.num_experts,
            expert_hidden_dim=self.expert_hidden_dim,
            depth=self.depth,
            ticks=self.ticks,
        )

    @nn.compact
    def initial_hidden(self, x: jnp.ndarray) -> Any:
        if self.expert_type in ("none", "glu"):
            return tuple()
        return self._build_recurrent_core().initial_hidden(x)

    @nn.compact
    def __call__(self, x: jnp.ndarray, hidden_acts: Any) -> Tuple[jnp.ndarray, Any]:
        if self.expert_type == "none":
            if hidden_acts is None:
                hidden_acts = tuple()
            return x, hidden_acts

        if self.expert_type == "glu":
            if hidden_acts is None:
                hidden_acts = tuple()
            head_core = MoEDense(
                num_experts=self.num_experts,
                expert_hidden_dim=self.expert_hidden_dim,
            )
            return head_core(x), hidden_acts

        return self._build_recurrent_core()(x, hidden_acts)


class PRENet(nn.Module):
    """Parallel recurrent expert policy/value network."""
    args: Any

    def _build_encoder(self) -> nn.Module:
        return build_observation_encoder(self.args)

    def _build_recurrent_core(self) -> SharedMoECore:
        return SharedMoECore(
            expert_type=self.args.expert_type,
            num_experts=self.args.num_experts,
            expert_hidden_dim=self.args.expert_hidden_dim,
            depth=self.args.depth,
            ticks=self.args.ticks,
            name="recurrent_core",
        )

    def _build_head_core(self) -> SharedMoECore:
        return SharedMoECore(
            expert_type=self.args.head_type,
            expert_hidden_dim=self.args.head_hidden_dim,
            num_experts=self.args.head_num_experts,
            depth=1,
            ticks=self.args.head_ticks,
            name="head_core",
        )

    @nn.compact
    def initial_hidden(self, x: jnp.ndarray) -> List[Any]:
        x_enc = self._build_encoder()(x)
        conv_hidden = self._build_recurrent_core().initial_hidden(x_enc)
        head_seed = jnp.zeros((x_enc.shape[0], 1), dtype=x_enc.dtype)
        head_hidden = self._build_head_core().initial_hidden(head_seed)
        return conv_hidden, head_hidden

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        hidden_acts: Optional[List[Any]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[Any]]:

        x_enc = self._build_encoder()(x)

        if hidden_acts is None:
            conv_hidden, head_hidden = None, None
        else:
            conv_hidden, head_hidden = hidden_acts

        cell_out, conv_hidden = self._build_recurrent_core()(x_enc, conv_hidden)

        flat = PreheadAggregation(args=self.args)(cell_out, x_enc)

        mid, head_hidden = self._build_head_core()(flat, head_hidden)

        logits = nn.Dense(self.args.action_dim,
                          )(mid)
        value = nn.Dense(1,
                         )(mid)
        hidden_acts = (conv_hidden, head_hidden)

        return logits, value.squeeze(-1), hidden_acts


@dataclass
class Agent:
    apply_fn: Any
    params: Any
    rng: jax.Array
    model: Any

    def get_action_and_value(
        self,
        x: jnp.ndarray,
        hidden_acts: Optional[List[Any]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List[Any]]:
        rng, step_key = jax.random.split(self.rng)
        action_key, dropout_key = jax.random.split(step_key)
        self.rng = rng
        logits, value, new_hidden = self.apply_fn(
            self.params,
            x,
            hidden_acts,
            rngs={"dropout": dropout_key},
        )
        # Sample actions and compute log-probs/entropy
        action = jax.random.categorical(action_key, logits)
        log_probs = jax.nn.log_softmax(logits)
        chosen_logprob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        return action, chosen_logprob, entropy, value, logits, new_hidden
