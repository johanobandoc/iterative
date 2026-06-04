from typing import Any, List, Optional, Tuple

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

    @nn.compact
    def __call__(
        self,
        x_enc: jnp.ndarray,
        res_stream: jnp.ndarray,
        h_cur: jnp.ndarray,
        c_cur: jnp.ndarray,
    ) -> Any:
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

class DenseLSTMCell(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(
        self,
        x_enc: jnp.ndarray,
        x_in: jnp.ndarray,
        h_cur: jnp.ndarray,
        c_cur: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:

        combined = jnp.concatenate([x_enc, x_in, h_cur], axis=-1)
        gates = nn.Dense(
            features=4 * self.embed_dim,
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


class RecurrentMoE(nn.Module):
    """Recurrent core over stacked layers."""

    expert_type: str
    num_experts: int
    expert_hidden_dim: int
    depth: int = 1
    ticks: int = 1

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
        hidden_acts: Optional[List[Any]] = None,
    ) -> Tuple[jnp.ndarray, List[Any]]:
        if hidden_acts is None:
            hidden_acts = self._initial_hidden_impl(x_enc)

        sum_expert_aggregator = SumExpertsAggregator(name="sum_expert_aggregator")

        def build_read_views(
            res_stream: jnp.ndarray,
            num_experts: int,
        ) -> jnp.ndarray:
            return _all_to_all_expert_read_views(res_stream, num_experts)

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
                expert_res_stream = build_read_views(
                    res_stream,
                    num_experts=self.num_experts,
                )

                for depth_idx in range(self.depth):
                    c_cur = c_stack[:, :, depth_idx, ...]
                    depth_cell = DepthCell(
                        embed_dim=self.expert_hidden_dim,
                        kernel_size=(3, 3),
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

                h_stack = jnp.stack(next_h, axis=2)
                c_stack = jnp.stack(next_c, axis=2)
                h_final_stack = h_stack[:, :, -1, ...]
                res_stream = sum_expert_aggregator(h_final_stack)
                hidden_acts = (res_stream, h_stack, c_stack)
                return res_stream, hidden_acts

        if self.expert_type == "stacked_dense_lstm":
            DepthCell = nn.vmap(
                DenseLSTMCell,
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
                expert_res_stream = build_read_views(
                    res_stream,
                    num_experts=self.num_experts,
                )

                for depth_idx in range(self.depth):
                    c_cur = c_stack[:, :, depth_idx, ...]
                    depth_cell = DepthCell(
                        embed_dim=self.expert_hidden_dim,
                        name=f"depth_{depth_idx}",
                    )
                    h_next, c_next = depth_cell(
                        x_enc,
                        expert_res_stream,
                        h_cur,
                        c_cur,
                    )
                    h_cur = h_next
                    next_h.append(h_next)
                    next_c.append(c_next)

                h_stack = jnp.stack(next_h, axis=2)
                c_stack = jnp.stack(next_c, axis=2)
                h_final_stack = h_stack[:, :, -1, ...]
                res_stream = sum_expert_aggregator(h_final_stack)
                hidden_acts = (res_stream, h_stack, c_stack)
                return res_stream, hidden_acts

        def tick_fn(carry, _):
            cell_out, next_hidden = run_cell(x_enc, carry)
            return next_hidden, cell_out

        class Tick(nn.Module):
            @nn.compact
            def __call__(self, carry, _):
                return tick_fn(carry, _)

        TickLoop = nn.scan(
            Tick,
            variable_broadcast="params",
            split_rngs={"params": False},
            length=self.ticks,
        )
        dummy_steps = jnp.arange(self.ticks)
        hidden_acts, scan_out = TickLoop()(hidden_acts, dummy_steps)
        cell_outs = scan_out
        cell_out = cell_outs[-1]
        cell_out = nn.RMSNorm(epsilon=1e-5)(cell_out)

        return cell_out, hidden_acts
