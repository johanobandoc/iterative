from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from expert_aggregation import SumExpertsAggregator
from net_utils import init_recurrent_hidden


def _all_to_all_expert_read_views(
    res_stream: jnp.ndarray,
    num_experts: int,
) -> jnp.ndarray:
    return jnp.broadcast_to(
        jnp.expand_dims(res_stream, axis=0),
        (num_experts,) + res_stream.shape,
    )


class ConvLSTMCell(nn.Module):
    embed_dim: int
    kernel_size: Tuple[int, int] = (3, 3)
    skip_zero_state_projection: bool = False

    @nn.compact
    def __call__(
        self,
        x_enc: jnp.ndarray,
        res_stream: jnp.ndarray,
        h_cur: jnp.ndarray,
        c_cur: jnp.ndarray,
    ) -> Any:
        if self.skip_zero_state_projection:
            combined = x_enc
        else:
            combined = jnp.concatenate([x_enc, res_stream, h_cur], axis=-1)
        gates = nn.Conv(
            features=4 * self.embed_dim,
            kernel_size=self.kernel_size,
            padding="SAME",
            use_bias=True,
        )(combined)
        gates = nn.RMSNorm(epsilon=1e-5)(gates)

        cc_i, cc_f, cc_o, cc_g = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(cc_i)
        f = jax.nn.sigmoid(cc_f)
        o = jax.nn.sigmoid(cc_o)
        g = jnp.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * jnp.tanh(c_next)
        return h_next, c_next


class ConvGLUCell(nn.Module):
    """A spatial expert whose only carried state is the shared residual stream."""

    embed_dim: int
    kernel_size: Tuple[int, int] = (3, 3)
    skip_zero_state_projection: bool = False

    @nn.compact
    def __call__(
        self,
        x_enc: jnp.ndarray,
        res_stream: jnp.ndarray,
    ) -> jnp.ndarray:
        combined = x_enc if self.skip_zero_state_projection else jnp.concatenate(
            [x_enc, res_stream], axis=-1
        )
        gates = nn.Conv(
            features=2 * self.embed_dim,
            kernel_size=self.kernel_size,
            padding="SAME",
            use_bias=True,
        )(combined)
        value, gate = jnp.split(gates, 2, axis=-1)
        output = value * jax.nn.sigmoid(gate)
        return nn.RMSNorm(reduction_axes=(1, 2, 3), epsilon=1e-5)(output)


class RecurrentMoE(nn.Module):
    """One pass through the expert depths per environment step."""

    expert_type: str
    num_experts: int
    expert_hidden_dim: int
    depth: int = 1
    reset_hidden_every_step: bool = False

    def _initial_hidden_impl(self, x_enc: jnp.ndarray) -> Any:
        return init_recurrent_hidden(
            x_enc,
            expert_type=self.expert_type,
            expert_hidden_dim=self.expert_hidden_dim,
            num_experts=self.num_experts,
            depth=self.depth,
        )

    @nn.compact
    def initial_hidden(self, x_enc: jnp.ndarray) -> Any:
        return self._initial_hidden_impl(x_enc)

    @nn.compact
    def __call__(
        self,
        x_enc: jnp.ndarray,
        hidden_acts: Optional[Any] = None,
    ) -> Tuple[jnp.ndarray, Any]:
        if hidden_acts is None or self.reset_hidden_every_step:
            hidden_acts = self._initial_hidden_impl(x_enc)

        sum_expert_aggregator = SumExpertsAggregator(name="sum_expert_aggregator")

        if self.expert_type == "stacked_lstm":
            DepthCell = nn.vmap(
                ConvLSTMCell,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=(None, 0, 0, 0),
                out_axes=(0, 0),
                axis_size=self.num_experts,
            )

            def run_cell(x_enc, hidden_acts):
                res_stream, h_stack, c_stack = hidden_acts
                h_cur = h_stack[:, :, -1, ...]
                next_h = []
                next_c = []
                for depth_idx in range(self.depth):
                    expert_res_stream = _all_to_all_expert_read_views(
                        res_stream,
                        num_experts=self.num_experts,
                    )
                    c_cur = c_stack[:, :, depth_idx, ...]
                    depth_cell = DepthCell(
                        embed_dim=self.expert_hidden_dim,
                        kernel_size=(3, 3),
                        skip_zero_state_projection=(
                            self.reset_hidden_every_step and depth_idx == 0
                        ),
                        name=f"depth_{depth_idx}",
                    )
                    h_next, c_next = depth_cell(
                        x_enc,
                        expert_res_stream,
                        h_cur,
                        c_cur,
                    )
                    next_h.append(h_next)
                    next_c.append(c_next)
                    h_cur = h_next
                    res_stream = sum_expert_aggregator(h_next)

                h_stack = jnp.stack(next_h, axis=2)
                c_stack = jnp.stack(next_c, axis=2)
                hidden_acts = (res_stream, h_stack, c_stack)
                return res_stream, hidden_acts

        elif self.expert_type == "rec_conv_glu":
            DepthCell = nn.vmap(
                ConvGLUCell,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=(None, 0),
                out_axes=0,
                axis_size=self.num_experts,
            )

            def run_cell(x_enc, hidden_acts):
                res_stream = hidden_acts
                for depth_idx in range(self.depth):
                    expert_res_stream = _all_to_all_expert_read_views(
                        res_stream,
                        num_experts=self.num_experts,
                    )
                    depth_cell = DepthCell(
                        embed_dim=self.expert_hidden_dim,
                        skip_zero_state_projection=(
                            self.reset_hidden_every_step and depth_idx == 0
                        ),
                        name=f"depth_{depth_idx}",
                    )
                    expert_outputs = depth_cell(x_enc, expert_res_stream)
                    res_stream = sum_expert_aggregator(expert_outputs)
                return res_stream, res_stream

        else:
            raise ValueError(f"Unsupported Sokoban expert type: {self.expert_type}")

        cell_out, hidden_acts = run_cell(x_enc, hidden_acts)
        cell_out = nn.RMSNorm(epsilon=1e-5)(cell_out)

        return cell_out, hidden_acts
