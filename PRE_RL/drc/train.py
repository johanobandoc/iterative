import os
import random
import json
from dataclasses import dataclass
import time
import jax
import jax.numpy as jnp
import numpy as np
import tyro
try:
    from torch.utils.tensorboard import SummaryWriter  # if torch available
except Exception:  # fallback to tensorboardX
    from tensorboardX import SummaryWriter

from logging_utils import configure_run_logging
from nets import PRENet, transform_obs
from optimizer_utils import build_optimizer
from trainer import (
    build_hidden_template,
    train,
    validate_recurrent_config,
)
from utils import setup_env


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False  # reserved for future W&B integration
    wandb_project_name: str = "pre-rl"
    wandb_entity: str = ""
    wandb_tags: tuple[str, ...] = ()
    capture_video: bool = False

    env_id: str = "Sokoban-v0"
    use_noop_action: bool = False
    craftax_optimistic_resets: bool = False
    craftax_optimistic_reset_ratio: int = 16
    cuda: bool = True
    total_timesteps: int = 50_000_000
    learning_rate: float = 4e-4
    weight_decay: float = -1 #3e-5
    num_envs: int = 32
    num_steps: int = 20
    anneal_lr: bool = True
    gamma: float = 0.97
    gae_lambda: float = 0.97
    ignore_done_for_training: bool = False  # keep done only for stats/logging
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 1 #0.5
    reg_cost: float = 1e-4
    norm_adv: bool = False
    num_devices: int = 0  # 0 -> use all local devices

    # Model
    agent: str = "PRENet"
    action_dim: int = 0
    expert_hidden_dim: int = 32
    encoder_hidden_dim: int = 0  # 0 -> use expert_hidden_dim
    num_experts: int = 1
    depth: int = 1
    ticks: int = 1

    # encoder
    obs_encoder: str = "embedding_rms"  # "embedding_rms" for Sokoban, "mlp_glu" for Craftax

    # recurrent core
    expert_type: str = "stacked_lstm"  # "stacked_lstm" for Sokoban, "stacked_dense_lstm" for Craftax
    obs_vocab_size: int = 5
    # head MoE
    head_num_experts: int = 1
    head_type: str = "glu"
    head_hidden_dim: int = 256
    head_ticks: int = 1
    prehead_aggregation: str = "attn"  # "attn" for Sokoban, "enc_cell_flatten" for Craftax

    # runtime
    batch_size: int = 0
    num_iterations: int = 0
    log_interval: int = 200  # print/log interval in iterations
    log_hidden_stats: bool = False  # log recurrent/residual activation norms
    max_episode_steps: int = 120  # for res_stream trajectory stats
    res_stream_ema_decay: float = 0.99  # EMA decay for res_stream trajectory stats
    storage_base_dir: str = ""  # empty -> repo root, or $SCRATCH/<repo> when SCRATCH is set


MODEL_REGISTRY = {
    "PRENet": PRENet,
}


def shard_env_tree(tree, num_envs: int, num_devices: int, axis1_env_axis_size: int = 0):
    """Split tensors with an env axis into (device, env_per_device, ...)."""
    envs_per_device = num_envs // max(1, num_devices)

    def _reshape(x):
        if not hasattr(x, "shape"):
            return x
        if x.ndim >= 5 and x.shape[1] == num_envs:
            # Stacked hidden state: (layers, envs, ...), shard over env axis.
            reshaped = jnp.asarray(x).reshape(x.shape[0], num_devices, envs_per_device, *x.shape[2:])
            return jnp.swapaxes(reshaped, 0, 1)
        if axis1_env_axis_size > 0 and x.ndim > 1 and x.shape[0] == axis1_env_axis_size and x.shape[1] == num_envs:
            # Prefer axis 1 when both axes could match num_envs. This matters for
            # hidden states like (num_experts, envs, expert_hidden_dim) when num_experts == num_envs.
            # Restrict this to known expert-stacked tensors.
            reshaped = jnp.asarray(x).reshape(x.shape[0], num_devices, envs_per_device, *x.shape[2:])
            return jnp.swapaxes(reshaped, 0, 1)
        if x.shape[0] == num_envs:
            return jnp.asarray(x).reshape(num_devices, envs_per_device, *x.shape[1:])
        return x

    return jax.tree_util.tree_map(_reshape, tree)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.encoder_hidden_dim == 0:
        args.encoder_hidden_dim = args.expert_hidden_dim
    validate_recurrent_config(args)
    if not args.cuda:
        jax.config.update("jax_platform_name", "cpu")
    devices = jax.local_devices()
    available_devices = len(devices)
    num_devices = args.num_devices or available_devices or 1
    if num_devices > available_devices:
        raise ValueError(f"Requested {num_devices} devices but only {available_devices} available")
    args.num_devices = num_devices
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = max(1, args.total_timesteps // args.batch_size)
    if args.num_envs % num_devices != 0:
        raise ValueError(f"num_envs={args.num_envs} must be divisible by num_devices={num_devices} for pmap")
    envs_per_device = args.num_envs // num_devices
    args.envs_per_device = envs_per_device

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # Envs
    print(f"[setup] env={args.env_id}, num_envs={args.num_envs}, num_steps={args.num_steps}", flush=True)
    envs = setup_env(args)

    # Sample shape to init model
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    _, sample_obs = envs.reset(reset_keys)
    if sample_obs.ndim == 3:
        sample_obs = sample_obs[..., None]
    sample_x = transform_obs(sample_obs, args.env_id)

    # Build model
    model_cls = MODEL_REGISTRY[args.agent]
    action_spec = envs.action_spec() if callable(envs.action_spec) else envs.action_spec
    args.action_dim = action_spec.num_values
    model = model_cls(args=args)
    rng, params_key, init_dropout_key = jax.random.split(rng, 3)
    params = model.init(
        {"params": params_key, "dropout": init_dropout_key},
        sample_x,
        None,
    )
    print(f"[model] initialized agent={args.agent}")
    summary_root_key = jax.random.PRNGKey(args.seed + 42)
    summary_params_key, summary_dropout_key = jax.random.split(summary_root_key)
    summary = model.tabulate(
        {"params": summary_params_key, "dropout": summary_dropout_key},
        sample_x,
        None,
        console_kwargs={"force_jupyter": False},
    )
    print(summary)
    leaves, _ = jax.tree_util.tree_flatten(params)
    num_params = sum(int(np.prod(getattr(l, 'shape', ()))) for l in leaves)
    print(f"[setup] action_dim={action_spec.num_values}, obs_shape={tuple(sample_x.shape[1:])}, params={num_params/1e6:.2f}M", flush=True)

    # Hidden-state template for rollout startup and fast episode resets on device.
    hidden_template = build_hidden_template(args, model, params, sample_obs)
    hidden_template = shard_env_tree(
        hidden_template,
        args.num_envs,
        num_devices,
        axis1_env_axis_size=args.num_experts,
    )

    # Optimizer
    tx = build_optimizer(args, params)
    opt_state = tx.init(params)
    opt_state = jax.device_put_replicated(opt_state, devices[:num_devices])

    # Logging setup (TensorBoard + optional W&B), mirroring torch version
    group_name = f"{args.agent}_{args.exp_name}"
    run_name = f"{group_name}_s{args.seed}_{int(time.time())}"
    args.group_name = group_name
    args.run_name = run_name
    args.randint = random.randint(0, 1000)
    configure_run_logging(args)

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity or None,
            sync_tensorboard=True,
            config=vars(args),
            group=group_name,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            tags=list(args.wandb_tags),
        )

    writer = SummaryWriter(args.out_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    with open(f"{args.out_dir}/args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Train
    rngs = jax.random.split(rng, num_devices)
    rngs = jax.device_put_sharded([r for r in rngs], devices[:num_devices])
    params_repl = jax.device_put_replicated(params, devices[:num_devices])
    train(
        args=args,
        envs=envs,
        model=model,
        params=params_repl,
        tx=tx,
        opt_state=opt_state,
        rng=rngs,
        writer=writer,
        hidden_template=hidden_template,
    )
    del params_repl
    del rngs

    writer.close()
