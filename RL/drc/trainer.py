from typing import Any
from collections import deque
import gc
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

AXIS_NAME = "devices"


def build_hidden_template(model, params, obs):
    return model.apply(params, obs, method=model.initial_hidden)


def validate_recurrent_config(args):
    num_experts = int(args.num_experts)
    if num_experts < 1:
        raise ValueError(f"num_experts must be positive, got {num_experts}")


def reset_hidden_states(hidden, done_mask: jax.Array, template, reset_all: bool = False):
    """Reset recurrent states for finished envs using a broadcasted template."""
    if reset_all:
        return template
    env_dim = done_mask.shape[0]

    def _reset(h, t):
        if h.ndim == 0:
            return h
        if h.ndim == 6 and h.shape[1] == env_dim:
            # ConvLSTM h/c are [experts, envs, depth, height, width, channels].
            # Prefer the known env axis even when num_experts == num_envs.
            expand_shape = (1, env_dim) + (1,) * (h.ndim - 2)
        elif h.shape[0] == env_dim:
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

    def reset_env(rng_key):
        rng_key, reset_key = jax.random.split(rng_key)
        reset_keys = jax.random.split(reset_key, envs_per_device)
        env_state, obs = envs.reset(reset_keys)
        done = jnp.zeros((envs_per_device,), dtype=jnp.float32)
        return rng_key, env_state, obs, done

    def rollout_and_update(params, opt_state, rng, env_state, obs, done, hidden, hidden_template):
        def loss_and_aux(params, rng, env_state, obs, done, hidden, hidden_template):
            hidden = reset_hidden_states(
                hidden, done > 0.0, hidden_template,
                reset_all=args.reset_hidden_every_step,
            )

            def step_fn(step_carry, _):
                env_state, obs, done, hidden, rng_key = step_carry
                hidden = reset_hidden_states(
                    hidden, done > 0.0, hidden_template,
                    reset_all=args.reset_hidden_every_step,
                )
                rng_key, action_key = jax.random.split(rng_key)
                logits, value, new_hidden = model.apply(
                    params,
                    obs,
                    hidden,
                )
                action = jax.random.categorical(action_key, logits)
                env_state, next_obs, reward, next_done = envs.step(env_state, action)
                next_done = next_done.astype(jnp.float32)
                logp = jax.nn.log_softmax(logits)
                entropy = -jnp.sum(jnp.exp(logp) * logp, axis=-1)
                step_carry = (env_state, next_obs, next_done, new_hidden, rng_key)
                step_out = (logits, logp, entropy, value, action, reward, next_done)
                return step_carry, step_out

            (env_state, obs, done, hidden, rng), traj = jax.lax.scan(
                step_fn,
                (env_state, obs, done, hidden, rng),
                None,
                length=num_steps,
            )
            logits_seq, logp_seq, entropy_seq, value_seq, act_seq, rew_seq, done_post_seq = traj
            hidden = reset_hidden_states(
                hidden, done > 0.0, hidden_template,
                reset_all=args.reset_hidden_every_step,
            )
            _, next_value, _ = model.apply(
                params,
                obs,
                hidden,
            )
            adv_seq, ret_seq = compute_gae(
                rew_seq,
                value_seq,
                done_post_seq,
                next_value,
                args.gamma,
                args.gae_lambda,
            )

            # Treat advantages and returns as constants when differentiating the loss.
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
        )

    def make_p_train_step():
        return jax.pmap(
            rollout_and_update,
            axis_name=AXIS_NAME,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
            out_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
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

        rewards = _flatten_pmap_time_first(rew_seq)
        done_flags = _flatten_pmap_time_first(done_post_seq) > 0.0
        for t in range(num_steps):
            running_returns += rewards[t]
            running_lengths += 1
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
    gc.collect()
    return
