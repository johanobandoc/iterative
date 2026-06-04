from typing import Any
from collections import deque
import gc
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nets import transform_obs

AXIS_NAME = "devices"


def build_hidden_template(args, model, params, obs):
    x = transform_obs(obs, args.env_id)
    return model.apply(
        params,
        x,
        method=model.initial_hidden,
    )


def validate_recurrent_config(args):
    num_experts = int(args.num_experts)
    if num_experts < 1:
        raise ValueError(f"num_experts must be positive, got {num_experts}")

def _stat_axes(x: jax.Array) -> tuple[int, ...]:
    return tuple(range(1, x.ndim))


def _rms(x: jax.Array, axis: tuple[int, ...]) -> jax.Array:
    return jnp.sqrt(jnp.mean(jnp.square(x), axis=axis))


def _max_abs(x: jax.Array, axis: tuple[int, ...]) -> jax.Array:
    return jnp.max(jnp.abs(x), axis=axis)


def _res_stream_from_hidden(hidden) -> jax.Array:
    return hidden[0]


def _supports_res_stream_stats(expert_type: str) -> bool:
    return expert_type in (
        "stacked_lstm",
        "stacked_dense_lstm",
    )


def res_stream_stats(hidden, expert_type: str) -> tuple[jax.Array, jax.Array]:
    res_stream = _res_stream_from_hidden(hidden)
    axes = _stat_axes(res_stream)
    return _rms(res_stream, axes), _max_abs(res_stream, axes)


def packed_res_stream_stats(
    hidden: tuple[Any, Any],
    expert_type: str,
    head_type: str,
    done_like: jax.Array,
) -> jax.Array:
    core_hidden, head_hidden = hidden

    if _supports_res_stream_stats(expert_type):
        core_rms, core_max_abs = res_stream_stats(core_hidden, expert_type)
    else:
        core_rms = jnp.zeros_like(done_like)
        core_max_abs = jnp.zeros_like(done_like)

    if _supports_res_stream_stats(head_type):
        head_rms, head_max_abs = res_stream_stats(head_hidden, head_type)
    else:
        head_rms = jnp.zeros_like(done_like)
        head_max_abs = jnp.zeros_like(done_like)

    core_stats = jnp.stack([core_rms, core_max_abs], axis=-1)
    head_stats = jnp.stack([head_rms, head_max_abs], axis=-1)
    return jnp.stack([core_stats, head_stats], axis=-1)


def _supports_hstack_pairwise_cosine(expert_type: str) -> bool:
    return expert_type in (
        "stacked_lstm",
        "stacked_dense_lstm",
    )


def _pairwise_h_stack(hidden) -> jax.Array:
    h_stack = hidden[1]
    return h_stack[:, :, -1, ...]


def _hstack_pairwise_cosine_stats(hidden, expert_type: str, done_like: jax.Array) -> tuple[jax.Array, jax.Array]:
    if not _supports_hstack_pairwise_cosine(expert_type):
        return jnp.zeros_like(done_like), jnp.zeros_like(done_like)
    h_stack = _pairwise_h_stack(hidden)
    num_experts = h_stack.shape[0]
    if num_experts < 2:
        return jnp.zeros_like(done_like), jnp.zeros_like(done_like)
    flat = h_stack.reshape((num_experts, h_stack.shape[1], -1))
    eps = jnp.asarray(1e-8, dtype=flat.dtype)
    unit = flat / jnp.maximum(jnp.linalg.norm(flat, axis=-1, keepdims=True), eps)
    cos = jnp.einsum("ebd,fbd->efb", unit, unit)
    tri_i, tri_j = jnp.triu_indices(num_experts, k=1)
    pairwise = cos[tri_i, tri_j, :]
    return jnp.mean(pairwise, axis=0), jnp.ones_like(done_like)


def packed_hstack_pairwise_cosine_stats(
    hidden: tuple[Any, Any],
    expert_type: str,
    head_type: str,
    done_like: jax.Array,
) -> jax.Array:
    core_hidden, head_hidden = hidden
    core_cos, core_valid = _hstack_pairwise_cosine_stats(core_hidden, expert_type, done_like)
    head_cos, head_valid = _hstack_pairwise_cosine_stats(head_hidden, head_type, done_like)
    core_stats = jnp.stack([core_cos, core_valid], axis=-1)
    head_stats = jnp.stack([head_cos, head_valid], axis=-1)
    return jnp.stack([core_stats, head_stats], axis=-1)


def _hstack_pairwise_cosine_matrix_sum_and_count(hidden, expert_type: str) -> tuple[jax.Array, jax.Array]:
    if not _supports_hstack_pairwise_cosine(expert_type):
        return jnp.zeros((1, 1), dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32)
    h_stack = _pairwise_h_stack(hidden, expert_type)
    num_experts = h_stack.shape[0]
    flat = h_stack.reshape((num_experts, h_stack.shape[1], -1))
    if num_experts < 2:
        return jnp.zeros((num_experts, num_experts), dtype=flat.dtype), jnp.asarray(0.0, dtype=flat.dtype)
    eps = jnp.asarray(1e-8, dtype=flat.dtype)
    unit = flat / jnp.maximum(jnp.linalg.norm(flat, axis=-1, keepdims=True), eps)
    cos_per_env = jnp.einsum("ibf,jbf->bij", unit, unit)
    return jnp.sum(cos_per_env, axis=0), jnp.asarray(flat.shape[1], dtype=flat.dtype)


def packed_hstack_pairwise_cosine_matrix_stats(
    hidden: tuple[Any, Any],
    expert_type: str,
    head_type: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    core_hidden, head_hidden = hidden
    core_sum, core_count = _hstack_pairwise_cosine_matrix_sum_and_count(core_hidden, expert_type)
    head_sum, head_count = _hstack_pairwise_cosine_matrix_sum_and_count(head_hidden, head_type)
    return core_sum, core_count, head_sum, head_count


def render_res_stream_curve(
    values: np.ndarray,
    width: int,
    height: int,
    ylabel: str,
    title: str,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    fig_w = max(2.4, width / 100.0)
    fig_h = max(2.0, height / 100.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.plot(np.arange(vals.shape[0]), vals, color="black", linewidth=1.0)
    ax.set_xlabel("step_idx")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(0, vals.shape[0] - 1))
    fig.tight_layout()

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    img = img[:, :, :3].copy()
    plt.close(fig)
    return img


def render_pairwise_cosine_heatmap(
    matrix: np.ndarray,
    title: str,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=120)
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    n_i = mat.shape[0]
    n_j = mat.shape[1]
    max_tick_labels = 8
    x_step = max(1, int(np.ceil(n_j / max_tick_labels)))
    y_step = max(1, int(np.ceil(n_i / max_tick_labels)))
    x_ticks = np.arange(0, n_j, x_step)
    y_ticks = np.arange(0, n_i, y_step)
    if x_ticks.size == 0 or x_ticks[-1] != n_j - 1:
        x_ticks = np.append(x_ticks, n_j - 1)
    if y_ticks.size == 0 or y_ticks[-1] != n_i - 1:
        y_ticks = np.append(y_ticks, n_i - 1)

    ax.set_xlabel("expert_j", fontsize=8)
    ax.set_ylabel("expert_i", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    tick_font = 6 if max(n_i, n_j) > 12 else 7
    ax.tick_params(axis="x", labelsize=tick_font)
    ax.tick_params(axis="y", labelsize=tick_font)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    img = img[:, :, :3].copy()
    plt.close(fig)
    return img


def reset_hidden_states(hidden, done_mask: jax.Array, template):
    """Reset recurrent states for finished envs using a broadcasted template."""
    env_dim = done_mask.shape[0]

    def _reset(h, t):
        if h.ndim == 0:
            return h
        if h.shape[0] == env_dim:
            expand_shape = (env_dim,) + (1,) * (h.ndim - 1)
        elif h.ndim > 1 and h.shape[1] == env_dim:
            expand_shape = (1, env_dim) + (1,) * (h.ndim - 2)
        else:
            raise ValueError(f"Hidden state shape {h.shape} incompatible with done mask of size {env_dim}")
        mask = done_mask.reshape(expand_shape)
        return jnp.where(mask, t, h)

    return jax.tree_util.tree_map(_reset, hidden, template)

def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones_next: jax.Array,
    next_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    rewards = rewards.astype(jnp.float32)
    values = values.astype(jnp.float32)
    dones_next = dones_next.astype(jnp.float32)

    def scan_fn(carry, inputs):
        adv, next_val = carry
        reward, value, done = inputs
        next_nonterminal = 1.0 - done
        delta = reward + gamma * next_val * next_nonterminal - value
        adv = delta + gamma * gae_lambda * next_nonterminal * adv
        return (adv, value), adv

    init = (jnp.zeros_like(next_value), next_value)
    (_, _), adv_rev = jax.lax.scan(
        scan_fn,
        init,
        (rewards[::-1], values[::-1], dones_next[::-1]),
    )
    advantages = adv_rev[::-1]
    returns = advantages + values
    return advantages, returns


def train(
    args,
    envs,
    model,
    params: Any,
    tx,
    opt_state: Any,
    rng: jax.Array,
    writer=None,
    hidden_template=None,
):
    num_steps, num_envs = args.num_steps, args.num_envs
    num_devices = getattr(args, "num_devices", 1)
    envs_per_device = getattr(args, "envs_per_device", num_envs)
    if num_devices * envs_per_device != num_envs:
        raise ValueError(f"num_envs ({num_envs}) must equal num_devices ({num_devices}) * envs_per_device ({envs_per_device})")

    if hidden_template is None:
        raise ValueError("hidden_template must be provided for JAX training")
    devices = jax.local_devices()[:num_devices]

    def _flatten_pmap_time_first(x):
        """Convert pmap output (devices, steps, envs_per_device, ...) to (steps, envs, ...)."""
        x_host = np.asarray(jax.device_get(x))
        if x_host.ndim < 3:
            return x_host
        x_host = np.transpose(x_host, (1, 0, *range(2, x_host.ndim)))
        new_shape = (x_host.shape[0], x_host.shape[1] * x_host.shape[2], *x_host.shape[3:])
        return x_host.reshape(new_shape)

    def _maybe_add_channel(obs):
        if obs.ndim == 3:
            return obs[..., None]
        return obs

    def reset_env(rng_key):
        rng_key, reset_key = jax.random.split(rng_key)
        reset_keys = jax.random.split(reset_key, envs_per_device)
        env_state, obs = envs.reset(reset_keys)
        obs = _maybe_add_channel(obs)
        done = jnp.zeros((envs_per_device,), dtype=jnp.float32)
        return rng_key, env_state, obs, done

    def rollout_and_update(params, opt_state, rng, env_state, obs, done, hidden, hidden_template):
        def loss_and_aux(params, rng, env_state, obs, done, hidden, hidden_template):
            hidden = reset_hidden_states(hidden, done > 0.0, hidden_template)

            def step_fn(step_carry, _):
                env_state, obs, done, hidden, rng_key = step_carry
                hidden = reset_hidden_states(hidden, done > 0.0, hidden_template)
                x = transform_obs(obs, args.env_id)
                rng_key, step_key = jax.random.split(rng_key)
                action_key, dropout_key = jax.random.split(step_key)
                logits, value, new_hidden = model.apply(
                    params,
                    x,
                    hidden,
                    rngs={"dropout": dropout_key},
                )
                action = jax.random.categorical(action_key, logits)
                env_state, next_obs, reward, next_done = envs.step(env_state, action)
                next_obs = _maybe_add_channel(next_obs)
                next_done = next_done.astype(jnp.float32)
                if args.ignore_done_for_training:
                    next_done_train = jnp.zeros_like(next_done)
                else:
                    next_done_train = next_done
                logp = jax.nn.log_softmax(logits)
                entropy = -jnp.sum(jnp.exp(logp) * logp, axis=-1)
                step_carry = (env_state, next_obs, next_done_train, new_hidden, rng_key)
                step_out = (logits, logp, entropy, value, action, reward, done, next_done)
                if args.log_hidden_stats:
                    res_stats = packed_res_stream_stats(
                        new_hidden,
                        expert_type=args.expert_type,
                        head_type=args.head_type,
                        done_like=done,
                    )
                    pairwise_cos_stats = packed_hstack_pairwise_cosine_stats(
                        new_hidden,
                        expert_type=args.expert_type,
                        head_type=args.head_type,
                        done_like=done,
                    )
                    (
                        core_pairwise_matrix_sum,
                        core_pairwise_matrix_count,
                        head_pairwise_matrix_sum,
                        head_pairwise_matrix_count,
                    ) = packed_hstack_pairwise_cosine_matrix_stats(
                        new_hidden,
                        expert_type=args.expert_type,
                        head_type=args.head_type,
                    )
                    step_out = step_out + (
                        res_stats,
                        pairwise_cos_stats,
                        core_pairwise_matrix_sum,
                        core_pairwise_matrix_count,
                        head_pairwise_matrix_sum,
                        head_pairwise_matrix_count,
                    )
                return step_carry, step_out

            (env_state, obs, done, hidden, rng), traj = jax.lax.scan(
                step_fn,
                (env_state, obs, done, hidden, rng),
                None,
                length=num_steps,
            )
            logits_seq, logp_seq, entropy_seq, value_seq, act_seq, rew_seq, done_seq, done_post_seq = traj[:8]
            traj_idx = 8
            if args.log_hidden_stats:
                (
                    res_stats_seq,
                    pairwise_cos_stats_seq,
                    core_pairwise_matrix_sum_seq,
                    core_pairwise_matrix_count_seq,
                    head_pairwise_matrix_sum_seq,
                    head_pairwise_matrix_count_seq,
                ) = traj[traj_idx:traj_idx + 6]
                traj_idx += 6
            else:
                envs_per_device = done.shape[0]
                res_stats_seq = jnp.zeros((num_steps, envs_per_device, 2, 2), dtype=jnp.float32)
                pairwise_cos_stats_seq = jnp.zeros((num_steps, envs_per_device, 2, 2), dtype=jnp.float32)
                core_pairwise_matrix_sum_seq = jnp.zeros((num_steps, 1, 1), dtype=jnp.float32)
                core_pairwise_matrix_count_seq = jnp.zeros((num_steps,), dtype=jnp.float32)
                head_pairwise_matrix_sum_seq = jnp.zeros((num_steps, 1, 1), dtype=jnp.float32)
                head_pairwise_matrix_count_seq = jnp.zeros((num_steps,), dtype=jnp.float32)
            hidden = reset_hidden_states(hidden, done > 0.0, hidden_template)
            x = transform_obs(obs, args.env_id)
            rng, next_dropout_key = jax.random.split(rng)
            _, next_value, _ = model.apply(
                params,
                x,
                hidden,
                rngs={"dropout": next_dropout_key},
            )
            if args.ignore_done_for_training:
                done_post_seq_train = jnp.zeros_like(done_post_seq)
            else:
                done_post_seq_train = done_post_seq
            adv_seq, ret_seq = compute_gae(
                rew_seq,
                value_seq,
                done_post_seq_train,
                next_value,
                args.gamma,
                args.gae_lambda,
            )

            if args.norm_adv:
                adv_mean = jax.lax.pmean(jnp.mean(adv_seq), axis_name=AXIS_NAME)
                adv_var = jax.lax.pmean(jnp.mean((adv_seq - adv_mean) ** 2), axis_name=AXIS_NAME)
                adv_seq = (adv_seq - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
            # Match dp-jum behavior: advantages/returns are treated as constants w.r.t. params.
            adv_seq = jax.lax.stop_gradient(adv_seq)
            ret_seq = jax.lax.stop_gradient(ret_seq)

            act_logp = jnp.take_along_axis(logp_seq, act_seq[..., None], axis=-1).squeeze(-1)
            pg_loss = -(adv_seq) * act_logp
            v_loss = 0.5 * jnp.square(value_seq - ret_seq)
            reg_loss = jnp.sum(jnp.square(logits_seq), axis=-1)

            loss = (
                jnp.mean(pg_loss)
                - args.ent_coef * jnp.mean(entropy_seq)
                + args.vf_coef * jnp.mean(v_loss)
                + args.reg_cost * jnp.mean(reg_loss)
            )
            metrics = {
                "pg": jnp.mean(pg_loss),
                "v": jnp.mean(v_loss),
                "ent": jnp.mean(entropy_seq),
                "reg": jnp.mean(reg_loss),
            }
            if args.log_hidden_stats:
                pairwise_values = pairwise_cos_stats_seq[:, :, 0, :]
                pairwise_valid = pairwise_cos_stats_seq[:, :, 1, :]
                pairwise_counts = jnp.sum(pairwise_valid, axis=(0, 1))
                pairwise_sums = jnp.sum(pairwise_values * pairwise_valid, axis=(0, 1))
                pairwise_means = pairwise_sums / jnp.maximum(pairwise_counts, 1.0)
                metrics["core_hstack_pairwise_cosine"] = pairwise_means[0]
                metrics["head_hstack_pairwise_cosine"] = pairwise_means[1]
                metrics["core_hstack_pairwise_count"] = pairwise_counts[0]
                metrics["head_hstack_pairwise_count"] = pairwise_counts[1]

                core_matrix_sum = jnp.sum(core_pairwise_matrix_sum_seq, axis=0)
                head_matrix_sum = jnp.sum(head_pairwise_matrix_sum_seq, axis=0)
                core_matrix_count = jnp.sum(core_pairwise_matrix_count_seq)
                head_matrix_count = jnp.sum(head_pairwise_matrix_count_seq)
                metrics["core_hstack_pairwise_matrix"] = core_matrix_sum / jnp.maximum(core_matrix_count, 1.0)
                metrics["head_hstack_pairwise_matrix"] = head_matrix_sum / jnp.maximum(head_matrix_count, 1.0)
                metrics["core_hstack_pairwise_matrix_count"] = core_matrix_count
                metrics["head_hstack_pairwise_matrix_count"] = head_matrix_count
            aux = {
                "rng": rng,
                "env_state": env_state,
                "obs": obs,
                "done": done,
                "hidden": hidden,
                "metrics": metrics,
                "rew_seq": rew_seq,
                "done_post_seq": done_post_seq,
                "ret_seq": ret_seq,
                "res_stats_seq": res_stats_seq,
            }
            return loss, aux

        (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
            params, rng, env_state, obs, done, hidden, hidden_template
        )
        rng = aux["rng"]
        env_state = aux["env_state"]
        obs = aux["obs"]
        done = aux["done"]
        hidden = aux["hidden"]
        metrics = aux["metrics"]
        rew_seq = aux["rew_seq"]
        done_post_seq = aux["done_post_seq"]
        ret_seq = aux["ret_seq"]
        res_stats_seq = aux["res_stats_seq"]

        grads = jax.lax.pmean(grads, axis_name=AXIS_NAME)
        loss = jax.lax.pmean(loss, axis_name=AXIS_NAME)
        metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name=AXIS_NAME), metrics)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = {**metrics, "loss": loss}
        return (
            params,
            opt_state,
            rng,
            env_state,
            obs,
            done,
            hidden,
            metrics,
            rew_seq,
            done_post_seq,
            ret_seq,
            res_stats_seq,
        )

    def make_p_train_step():
        return jax.pmap(
            rollout_and_update,
            axis_name=AXIS_NAME,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
            out_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

    p_reset = jax.pmap(reset_env, axis_name=AXIS_NAME)
    p_train_step = make_p_train_step()

    rng, env_state, obs, done = p_reset(rng)
    hidden = hidden_template
    global_step = 0
    start_time = time.time()
    recent_returns = deque(maxlen=400)
    recent_lengths = deque(maxlen=400)
    running_returns = np.zeros((num_envs,), dtype=np.float32)
    running_lengths = np.zeros((num_envs,), dtype=np.int32)
    if args.log_hidden_stats:
        max_steps = int(args.max_episode_steps)
        if max_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {max_steps}")
        ema_decay = float(args.res_stream_ema_decay)
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"res_stream_ema_decay must be in (0, 1), got {ema_decay}")
        # Shape: (metric_idx, core_idx, step_idx)
        # metric_idx: 0=rms, 1=max_abs; core_idx: 0=core, 1=head
        res_ema = np.zeros((2, 2, max_steps), dtype=np.float32)
        metric_labels = ("rms", "max_abs")
        core_labels = ("core", "head")

    def _metric_to_host(x):
        x_host = jax.device_get(x)
        if isinstance(x_host, np.ndarray) and x_host.shape:
            x_host = x_host[0]
        if isinstance(x_host, np.ndarray) and x_host.shape == ():
            return x_host.item()
        return x_host

    print(f"[train] starting: iterations={args.num_iterations}, batch_size={args.batch_size}", flush=True)
    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.time()
        (
            params,
            opt_state,
            rng,
            env_state,
            obs,
            done,
            hidden,
            loss_info,
            rew_seq,
            done_post_seq,
            ret_seq,
            res_stats_seq,
        ) = p_train_step(
            params,
            opt_state,
            rng,
            env_state,
            obs,
            done,
            hidden,
            hidden_template,
        )
        global_step += num_envs * num_steps
        step_elapsed = time.time() - iter_start

        # Logging (TensorBoard-style), mirroring torch
        rewards = _flatten_pmap_time_first(rew_seq)
        done_flags = _flatten_pmap_time_first(done_post_seq) > 0.0
        if args.log_hidden_stats:
            res_stats_host = _flatten_pmap_time_first(res_stats_seq)
        for t in range(num_steps):
            running_returns += rewards[t]
            running_lengths += 1
            if args.log_hidden_stats:
                step_idx = running_lengths - 1
                in_range = step_idx < max_steps
                if np.any(in_range):
                    idx = step_idx[in_range].astype(np.int64)
                    counts = np.bincount(idx, minlength=max_steps)
                    has = counts > 0
                    if np.any(has):
                        step_stats = res_stats_host[t][in_range]
                        for metric_idx in range(2):
                            for core_idx in range(2):
                                vals = step_stats[:, metric_idx, core_idx]
                                sums = np.bincount(idx, weights=vals, minlength=max_steps)
                                means = sums[has] / counts[has]
                                res_ema[metric_idx, core_idx, has] = (
                                    ema_decay * res_ema[metric_idx, core_idx, has]
                                    + (1.0 - ema_decay) * means
                                )
            finished = done_flags[t]
            if np.any(finished):
                recent_returns.extend(running_returns[finished])
                recent_lengths.extend(running_lengths[finished])
                running_returns[finished] = 0.0
                running_lengths[finished] = 0

        should_log = iteration == 1 or (
            args.log_interval and iteration % args.log_interval == 0
        )
        if should_log:
            mean_ret = float(np.mean(recent_returns)) if len(recent_returns) else float("nan")
            mean_len = float(np.mean(recent_lengths)) if len(recent_lengths) else float("nan")
            sps = int(global_step / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0

            print(f"[iter {iteration}/{args.num_iterations}] step={step_elapsed:.2f}s SPS={sps} return={mean_ret if mean_ret==mean_ret else 'n/a'} length={mean_len if mean_len==mean_len else 'n/a'}", flush=True)

            loss_info_host = jax.tree_util.tree_map(_metric_to_host, loss_info)
            writer.add_scalar("charts/episodic_return", mean_ret, global_step)
            writer.add_scalar("charts/episodic_length", mean_len, global_step)
            writer.add_scalar("charts/active_num_experts", int(args.num_experts), global_step)
            if args.track:
                import wandb
                wandb.log(
                    {"charts/active_num_experts": int(args.num_experts)},
                    step=global_step,
                )
            writer.add_scalar("losses/policy_loss", float(loss_info_host.get("pg", np.nan)), global_step)
            writer.add_scalar("losses/value_loss", float(loss_info_host.get("v", np.nan)), global_step)
            writer.add_scalar("losses/entropy", float(loss_info_host.get("ent", np.nan)), global_step)
            returns_mean = float(np.asarray(jax.device_get(ret_seq)).mean())
            writer.add_scalar("losses/returns", returns_mean, global_step)
            reg_val = loss_info_host.get("reg")
            if reg_val is not None:
                writer.add_scalar("losses/reg_loss", float(reg_val), global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            if args.log_hidden_stats:
                core_pairwise = loss_info_host.get("core_hstack_pairwise_cosine")
                head_pairwise = loss_info_host.get("head_hstack_pairwise_cosine")
                core_pairwise_count = loss_info_host.get("core_hstack_pairwise_count")
                head_pairwise_count = loss_info_host.get("head_hstack_pairwise_count")
                core_pairwise_matrix = loss_info_host.get("core_hstack_pairwise_matrix")
                head_pairwise_matrix = loss_info_host.get("head_hstack_pairwise_matrix")
                core_pairwise_matrix_count = loss_info_host.get("core_hstack_pairwise_matrix_count")
                head_pairwise_matrix_count = loss_info_host.get("head_hstack_pairwise_matrix_count")

                if core_pairwise is not None and core_pairwise_count is not None and float(core_pairwise_count) > 0.0:
                    writer.add_scalar("debug/core_hstack_pairwise_cosine", float(core_pairwise), global_step)
                if head_pairwise is not None and head_pairwise_count is not None and float(head_pairwise_count) > 0.0:
                    writer.add_scalar("debug/head_hstack_pairwise_cosine", float(head_pairwise), global_step)

                if args.track:
                    import wandb
                    log_payload = {}
                    if core_pairwise is not None and core_pairwise_count is not None and float(core_pairwise_count) > 0.0:
                        log_payload["debug/core_hstack_pairwise_cosine"] = float(core_pairwise)
                    if head_pairwise is not None and head_pairwise_count is not None and float(head_pairwise_count) > 0.0:
                        log_payload["debug/head_hstack_pairwise_cosine"] = float(head_pairwise)
                    if (
                        core_pairwise_matrix is not None
                        and core_pairwise_matrix_count is not None
                        and float(core_pairwise_matrix_count) > 0.0
                    ):
                        log_payload["debug/core_hstack_pairwise_cosine_heatmap"] = wandb.Image(
                            render_pairwise_cosine_heatmap(
                                core_pairwise_matrix,
                                title="core h_stack pairwise cosine",
                            ),
                            caption=f"step={global_step}",
                        )
                    if (
                        head_pairwise_matrix is not None
                        and head_pairwise_matrix_count is not None
                        and float(head_pairwise_matrix_count) > 0.0
                    ):
                        log_payload["debug/head_hstack_pairwise_cosine_heatmap"] = wandb.Image(
                            render_pairwise_cosine_heatmap(
                                head_pairwise_matrix,
                                title="head h_stack pairwise cosine",
                            ),
                            caption=f"step={global_step}",
                        )
                    for core_idx, core_name in enumerate(core_labels):
                        for metric_idx, metric_name in enumerate(metric_labels):
                            key = f"debug/{core_name}_res_stream_{metric_name}_curve"
                            log_payload[key] = wandb.Image(
                                render_res_stream_curve(
                                    res_ema[metric_idx, core_idx],
                                    width=max(240, max_steps * 4),
                                    height=200,
                                    ylabel=f"{core_name}_{metric_name}_ema",
                                    title=f"{core_name} res stream {metric_name} EMA by step",
                                ),
                                caption=f"step={global_step}",
                            )
                    wandb.log(log_payload)
                np.savez(
                    f"{args.out_dir}/res_stream_ema.npz",
                    res_ema=res_ema,
                    core_res_rms_ema=res_ema[0, 0],
                    core_res_max_abs_ema=res_ema[1, 0],
                    head_res_rms_ema=res_ema[0, 1],
                    head_res_max_abs_ema=res_ema[1, 1],
                    res_rms_ema=res_ema[0, 0],
                    res_max_ema=res_ema[1, 0],
                    ema_decay=ema_decay,
                )

    final_history_mean_return = None
    if len(recent_returns):
        final_history_mean_return = float(np.mean(recent_returns))

    wandb_run_id = None
    if args.track:
        import wandb
        wandb_run_id = wandb.run.id
        if final_history_mean_return is not None:
            wandb.run.summary["final_history_mean_return"] = final_history_mean_return
        wandb.run.summary["return_history_count"] = int(len(recent_returns))

    summary = {
        "final_history_mean_return": final_history_mean_return,
        "return_history_count": int(len(recent_returns)),
        "run_name": args.run_name,
        "group_name": args.group_name,
        "out_dir": args.out_dir,
        "wandb_run_id": wandb_run_id,
        "seed": int(args.seed),
        "num_experts": int(args.num_experts),
        "head_num_experts": int(args.head_num_experts),
    }
    with open(f"{args.out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if args.track:
        wandb.finish()

    del params
    del opt_state
    del rng
    del env_state
    del obs
    del done
    del hidden
    del hidden_template
    del loss_info
    del rew_seq
    del done_post_seq
    del ret_seq
    del res_stats_seq
    gc.collect()
    return
