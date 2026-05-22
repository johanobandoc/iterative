import argparse
import dataclasses
import os

# Set some GPU FLAGS. Use setdefault so launch environments can override
# transport choices, e.g. disabling NVLS on systems where NCCL NVLS fails.
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
os.environ.setdefault("NCCL_LL128_BUFFSIZE", "-2")
os.environ.setdefault("NCCL_LL_BUFFSIZE", "-2")
os.environ.setdefault("NCCL_PROTO", "SIMPLE,LL,LL128")
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_pipelined_all_reduce=true "
    "--xla_gpu_enable_pipelined_all_gather=true "
    "--xla_gpu_enable_pipelined_reduce_scatter=true "
    "--xla_gpu_enable_while_loop_double_buffering=true "
    "--xla_gpu_enable_pipelined_p2p=true "
    "--xla_gpu_collective_permute_decomposer_threshold=1024 "
)
import warnings
import logging
import time
from pathlib import Path
from functools import partial

import jax

jax.config.update("jax_optimization_level", "O1")

import optax
import grain
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.sharding import Mesh


import model as regular_model
import model_moe as moe_model
import model_shared_kv as shared_kv_model
from utils import logical_to_sharding
from optim import build_optimizer
from config import ShardingRules, Config, BATCH_AXIS_NAME, DEFAULT_FINEWEB_DIR
from fineweb_dataloader import make_grain_shard_loader, make_window_sampler
from logging_utils import init_wandb
from jax_compat import mesh_context
from checkpoint_utils import assert_checkpoint_payload_is_host
from checkpoint_utils import prepare_for_checkpoint_save


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


MODEL_BACKENDS = {
    "gpt": regular_model,
    "moe": moe_model,
    "shared_kv": shared_kv_model,
}


def _promote_inexact_leaves_to_f32(tree):
    return jax.tree.map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact)
        else x,
        tree,
    )


def forward_model(params, x_batch, segment_ids, freqs):
    if isinstance(params, moe_model.GPT):
        return moe_model.forward(params, x_batch, segment_ids, freqs)
    if isinstance(params, shared_kv_model.GPT):
        return shared_kv_model.forward(params, x_batch, segment_ids, freqs)
    return regular_model.forward(params, x_batch, segment_ids, freqs)


def compute_loss(params, x_batch, y_batch, segment_ids, freqs, loss_mask):
    logits = forward_model(params, x_batch, segment_ids, freqs)
    if loss_mask is not None:
        per_token_loss = optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=y_batch,
            where=loss_mask,
        )
        return jnp.sum(per_token_loss) / jnp.maximum(jnp.sum(loss_mask), 1.0)
    else:
        return jnp.mean(
            optax.losses.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=y_batch
            )
        )


@partial(
    jax.jit,
    static_argnames=("optim", "grad_accum_steps"),
    donate_argnums=(0, 1, 3, 4, 5),
)
def train_step_accum(
    params, x_batch, y_batch, segment_ids, freqs, optim_state, optim, grad_accum_steps
):
    def body(carry, xy):
        param, opt_state, lsum = carry
        xb, yb = xy
        loss, grad = jax.value_and_grad(compute_loss)(
            param, xb, yb, segment_ids, freqs, None
        )

        # MultiSteps accumulates grad internally and returns a zero-tree update on
        # every micro-step except the last, where it emits the real update.
        updates, new_opt_state = optim.update(grad, opt_state, param)
        new_param = optax.apply_updates(param, updates)
        return (new_param, new_opt_state, lsum + loss), None

    carry0 = (params, optim_state, jnp.array(0.0, dtype=jnp.result_type(0.0)))
    (params, optim_state, lsum), _ = jax.lax.scan(
        body, carry0, (x_batch, y_batch), length=grad_accum_steps
    )
    loss = lsum / grad_accum_steps
    return params, loss, optim_state


@partial(jax.jit, static_argnames=("optim",), donate_argnums=(0, 1, 3, 4, 5))
def train_step(params, x_batch, y_batch, segment_ids, freqs, optim_state, optim):
    loss, grads = jax.value_and_grad(compute_loss)(
        params, x_batch, y_batch, segment_ids, freqs, None
    )
    updates, optim_state = optim.update(grads, optim_state, params)
    updated_params = optax.apply_updates(params, updates)
    return updated_params, loss, optim_state


@jax.jit
def val_step(params, x_batch, y_batch, segment_ids, freqs):
    loss = compute_loss(params, x_batch, y_batch, segment_ids, freqs, None)
    return loss


def line(label, value, comma=False, label_w=30, colon_w=2, value_w=20):
    fmt = f">{value_w}," if comma else f">{value_w}"
    if value is None:
        value = "None"
    return f"{label:<{label_w}}{':':<{colon_w}}{value:{fmt}}"


def resolve_grad_accum_steps(desired_batch_size, global_batch_size, seqlen):
    micro_batch_tokens = global_batch_size * seqlen
    grad_accum_steps = max(1, desired_batch_size // micro_batch_tokens)
    effective_token_batch_size = grad_accum_steps * micro_batch_tokens
    if effective_token_batch_size > desired_batch_size:
        raise ValueError(
            "Configured desired_batch_size would be overshot because a single "
            "micro-batch is already too large: "
            f"desired_batch_size={desired_batch_size:,}, "
            f"global_batch_size={global_batch_size:,}, "
            f"seqlen={seqlen:,}, "
            f"micro_batch_tokens={micro_batch_tokens:,}, "
            f"grad_accum_steps={grad_accum_steps:,}, "
            f"effective_token_batch_size={effective_token_batch_size:,}."
        )
    return grad_accum_steps


def resolve_alias(primary_name, primary_value, alias_name, alias_value):
    if primary_value is not None and alias_value is not None:
        if primary_value != alias_value:
            raise ValueError(
                f"`--{primary_name}` and `--{alias_name}` were both set with "
                f"different values: {primary_value} vs {alias_value}."
            )
        return primary_value
    return primary_value if primary_value is not None else alias_value


def dense_layer_compute(width, seqlen, q_heads, kv_heads):
    kv_ratio = kv_heads / q_heads
    projection_and_mlp = (10.0 + 2.0 * kv_ratio) * width * width
    sequence_attention = 2.0 * seqlen * width
    return projection_and_mlp + sequence_attention


def estimate_compute_ratio(cfg):
    baseline = 16.0 * dense_layer_compute(768, cfg.model.seqlen, 8, 4)
    active_layers = cfg.model.num_layers
    if cfg.model.model_backend in ("moe", "shared_kv"):
        active_layers *= cfg.model.num_experts
    compute = active_layers * dense_layer_compute(
        cfg.model.d_emb,
        cfg.model.seqlen,
        cfg.model.q_heads,
        cfg.model.kv_heads,
    )
    if (
        cfg.model.model_backend == "shared_kv"
        and cfg.model.qkv_input_gelu_proj
    ):
        compute += active_layers * cfg.model.d_emb * cfg.model.d_emb
    return compute / baseline


def validate_train_config(cfg):
    if cfg.model.model_backend == "gpt" and cfg.model.num_experts != 1:
        raise ValueError(
            "`--num_experts` changes model behavior only with "
            "`--model_backend moe` or `--model_backend shared_kv`; "
            "gpt NanoGPT requires "
            "`num_experts=1`."
        )
    if cfg.model.qkv_input_gelu_proj and cfg.model.model_backend != "shared_kv":
        raise ValueError(
            "`--qkv_input_gelu_proj` is currently implemented only for "
            "`--model_backend shared_kv`."
        )


def get_next_batch(
    starts,
    ends,
    bsz,
    seqlen,
    tokens,
    data_sharding,
    buf_u16,
    transfer_to_device=False,
    create_new_buf=False,
):
    """Gathers batches of input-labels pairs.

    Given the `starts` and `ends` of sequences provided by the
    shard sampler, this method generates batches of inputs-labels efficiently.
    """
    if buf_u16 is None and create_new_buf:
        buf_u16 = np.empty((bsz, seqlen + 1), dtype=np.uint16)

    ptr = 0
    for i, j in zip(starts, ends):
        n = j - i
        row = ptr // (seqlen + 1)
        col = ptr % (seqlen + 1)
        buf_u16[row, col : col + n] = tokens[i:j]
        ptr += n

    # If no new array was created
    if not create_new_buf:
        return None
    else:
        if transfer_to_device:
            x = jax.device_put(buf_u16[:, :-1], data_sharding)
            y = jax.device_put(buf_u16[:, 1:], data_sharding)
        else:
            x = buf_u16[:, :-1]
            y = buf_u16[:, 1:]
        return x, y


def main():
    parser = argparse.ArgumentParser(description="nanoGPTJAX pretraining")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Optional W&B experiment name prefix.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=None,
        help="Override the per-device batch size from config.",
    )
    parser.add_argument(
        "--total_train_steps",
        type=int,
        default=None,
        help="Override the total number of train steps from config.",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=None,
        help="Override the peak LR used for non-embedding parameters.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Override the LR warmup length in optimizer steps.",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=None,
        help="Override global gradient clipping norm.",
    )
    parser.add_argument(
        "--disable_muon",
        action="store_true",
        help="Use AdamW for non-embedding parameters instead of Muon.",
    )
    parser.add_argument(
        "--muon_peak_lr_floor",
        type=float,
        default=None,
        help="Override the Muon peak-LR floor. Defaults preserve existing behavior.",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=None,
        help="Override the LR schedule horizon passed to optax. Defaults preserve existing behavior.",
    )
    parser.add_argument(
        "--data_dir",
        "--fineweb_dir",
        dest="data_dir",
        type=str,
        default=DEFAULT_FINEWEB_DIR,
        help="Path to the FineWeb token shards.",
    )
    parser.add_argument(
        "--ckpt_path",
        "--save_ckpt_dir",
        dest="ckpt_path",
        type=Path,
        default=None,
        help="Override the checkpoint save directory from config.",
    )
    parser.add_argument(
        "--last_checkpoint_step",
        type=int,
        default=None,
        help="Resume from this checkpoint step inside --ckpt_path.",
    )
    parser.add_argument(
        "--checkpoint_save_steps",
        type=int,
        default=None,
        help="Override checkpoint save interval in steps.",
    )
    parser.add_argument(
        "--max_checkpoints_to_keep",
        type=int,
        default=None,
        help="Override the number of checkpoints retained.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=None,
        help="Override the training sequence length from config.",
    )
    parser.add_argument(
        "--model_backend",
        type=str,
        default=None,
        choices=tuple(MODEL_BACKENDS.keys()),
        help="Select the gpt NanoGPT model or a non-recurrent MoE backend.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Override the number of transformer layers from config.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Paper-style alias for --num_layers.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="Number of parallel MoEs per depth stage for the MoE backend.",
    )
    parser.add_argument(
        "--d_emb",
        type=int,
        default=None,
        help="Override the model hidden dimension from config.",
    )
    parser.add_argument(
        "--expert_hidden_dim",
        type=int,
        default=None,
        help="Paper-style alias for --d_emb.",
    )
    parser.add_argument(
        "--q_heads",
        type=int,
        default=None,
        help="Override the number of query attention heads from config.",
    )
    parser.add_argument(
        "--kv_heads",
        type=int,
        default=None,
        help="Override the number of key/value attention heads from config.",
    )
    parser.add_argument(
        "--experts_aggregation_regime",
        type=str,
        default=None,
        choices=("mean", "sum_div_sqrt_num_experts"),
        help="Aggregation rule for MoE outputs.",
    )
    parser.add_argument(
        "--qkv_input_gelu_proj",
        action="store_true",
        help="For shared_kv, apply a per-expert learned GELU projection before q/k/v.",
    )
    cli_args = parser.parse_args()

    # Get the mesh, sharding rules, amd the config
    devices = np.array(jax.devices())
    print("Number of devices found:", len(devices))
    mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
    sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
    cfg = Config(mesh=mesh, rules=sharding_rules)
    if cli_args.exp_name:
        cfg.tracking.exp_name = cli_args.exp_name
    cfg.data_dir = cli_args.data_dir
    if cli_args.ckpt_path is not None:
        cfg.ckpt_cfg.save_ckpt_dir = cli_args.ckpt_path
    if cli_args.last_checkpoint_step is not None:
        cfg.ckpt_cfg.last_checkpoint_step = cli_args.last_checkpoint_step
    if cli_args.checkpoint_save_steps is not None:
        cfg.ckpt_cfg.checkpoint_save_steps = cli_args.checkpoint_save_steps
    if cli_args.max_checkpoints_to_keep is not None:
        cfg.ckpt_cfg.max_checkpoints_to_keep = cli_args.max_checkpoints_to_keep
    if cli_args.per_device_batch_size is not None:
        cfg.hparams.per_device_batch_size = cli_args.per_device_batch_size
    if cli_args.total_train_steps is not None:
        cfg.hparams.total_train_steps = cli_args.total_train_steps
    if cli_args.max_lr is not None:
        cfg.hparams.max_lr = cli_args.max_lr
    if cli_args.warmup_steps is not None:
        cfg.hparams.warmup_steps = cli_args.warmup_steps
    if cli_args.grad_clip_norm is not None:
        cfg.hparams.grad_clip_norm = cli_args.grad_clip_norm
    if cli_args.disable_muon:
        cfg.hparams.use_muon = False
    if cli_args.muon_peak_lr_floor is not None:
        cfg.hparams.muon_peak_lr_floor = cli_args.muon_peak_lr_floor
    if cli_args.lr_decay_steps is not None:
        cfg.hparams.lr_decay_steps = cli_args.lr_decay_steps
    if cli_args.seqlen is not None:
        cfg.model.seqlen = cli_args.seqlen
    depth = resolve_alias(
        "num_layers",
        cli_args.num_layers,
        "depth",
        cli_args.depth,
    )
    hidden_dim = resolve_alias(
        "d_emb",
        cli_args.d_emb,
        "expert_hidden_dim",
        cli_args.expert_hidden_dim,
    )
    model_updates = {}
    if cli_args.model_backend is not None:
        model_updates["model_backend"] = cli_args.model_backend
    if depth is not None:
        cfg.model.num_layers = depth
        cfg.model.embed.num_layers = depth
        cfg.model.attn.num_layers = depth
        model_updates["num_layers"] = depth
    if cli_args.num_experts is not None:
        model_updates["num_experts"] = cli_args.num_experts
    if hidden_dim is not None:
        model_updates["d_emb"] = hidden_dim
    if cli_args.q_heads is not None:
        model_updates["q_heads"] = cli_args.q_heads
    if cli_args.kv_heads is not None:
        model_updates["kv_heads"] = cli_args.kv_heads
    if cli_args.experts_aggregation_regime is not None:
        model_updates["experts_aggregation_regime"] = (
            cli_args.experts_aggregation_regime
        )
    if cli_args.qkv_input_gelu_proj:
        model_updates["qkv_input_gelu_proj"] = True
    if model_updates:
        cfg.model = dataclasses.replace(cfg.model, **model_updates)
    validate_train_config(cfg)

    train_files = list(Path(cfg.data_dir).glob("*train*.bin"))
    val_files = list(Path(cfg.data_dir).glob("*val*.bin"))
    num_train_files = len(train_files)
    num_val_files = len(val_files)
    print("\nNumber of train files found: ", num_train_files)
    print("Number of validation files found: ", num_val_files)
    if num_train_files == 0 or num_val_files == 0:
        raise FileNotFoundError(
            f"No FineWeb train/val shards found in {cfg.data_dir}. "
            "Pass --data_dir with a directory containing *train*.bin and *val*.bin files."
        )

    dataloader_mode = "stream_equal_chunks"
    train_dl = make_grain_shard_loader(train_files)
    val_dl = make_grain_shard_loader(val_files)
    train_iter = iter(train_dl)

    per_device_bsz = cfg.hparams.per_device_batch_size
    bsz = per_device_bsz * len(devices)
    seqlen = cfg.model.seqlen
    head_dim = cfg.model.attn.head_dim
    data_sharding = logical_to_sharding(("batch",), cfg.mesh, cfg.rules)
    data_accum_sharding = logical_to_sharding(
        (None, "batch", None), cfg.mesh, cfg.rules
    )

    max_lr = cfg.hparams.max_lr
    min_lr = 0.01 * max_lr
    warmup_steps = cfg.hparams.warmup_steps
    desired_batch_size = cfg.hparams.desired_batch_size
    grad_accum_steps = resolve_grad_accum_steps(desired_batch_size, bsz, seqlen)
    total_train_steps = cfg.hparams.total_train_steps
    max_checkpoints_to_keep = cfg.ckpt_cfg.max_checkpoints_to_keep
    checkpoint_save_steps = cfg.ckpt_cfg.checkpoint_save_steps
    wandb_run = None
    model_backend = MODEL_BACKENDS[cfg.model.model_backend]
    compute_ratio = estimate_compute_ratio(cfg)

    # Load the model
    print("Building GPT model based on the config...")
    model = model_backend.GPT.init(jax.random.PRNGKey(0), cfg)
    print("Model built successfully!")

    # Optimizer
    optim = optax.chain(
        optax.clip_by_global_norm(cfg.hparams.grad_clip_norm),
        build_optimizer(
            model,
            d_model=cfg.model.d_emb,
            other_peak_lr=max_lr,
            other_min_lr=min_lr,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            b1=cfg.hparams.b1,
            b2=cfg.hparams.b2,
            embedding_lr=cfg.hparams.embedding_lr,
            weight_decay=cfg.hparams.weight_decay,
            cautious_weight_decay=cfg.hparams.cautious_weight_decay,
            use_muon=cfg.hparams.use_muon,
            muon_peak_lr_floor=cfg.hparams.muon_peak_lr_floor,
            lr_decay_steps=cfg.hparams.lr_decay_steps,
        ),
    )

    if grad_accum_steps > 1:
        print("Using `MultiSteps` in optax for gradient accumulation...")
        optim = optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)

    optim_state = optim.init(model)
    if grad_accum_steps > 1:
        optim_state = _promote_inexact_leaves_to_f32(optim_state)

    # Checkpointing
    ckpt_path = Path(cfg.ckpt_cfg.save_ckpt_dir)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints_to_keep,
        save_interval_steps=checkpoint_save_steps,
        enable_async_checkpointing=True,
        enable_background_delete=True,
    )
    handlers = {
        "params": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "optim_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "ds": ocp.Checkpointer(grain.checkpoint.CheckpointHandler()),
    }

    mngr = ocp.CheckpointManager(ckpt_path, handlers, options=options)

    print("")
    print("-" * 75)
    print("")

    print(
        line(
            "Number of trainable params: ",
            regular_model.count_params(model),
            comma=True,
        )
    )
    print(line("Model backend", cfg.model.model_backend))
    print(line("Depth", cfg.model.num_layers))
    print(line("Number of experts", cfg.model.num_experts))
    print(line("Expert hidden dim", cfg.model.d_emb))
    print(line("Query heads", cfg.model.q_heads))
    print(line("KV heads", cfg.model.kv_heads))
    print(line("Experts aggregation", cfg.model.experts_aggregation_regime))
    print(line("QKV input GELU proj", cfg.model.qkv_input_gelu_proj))
    print(line("Compute ratio vs 16x768", f"{compute_ratio:.4f}"))
    print(line("Sequence length per sample", seqlen))
    print(line("Dataloader mode", dataloader_mode))
    print(line("Per device batch size", per_device_bsz))
    print(line("Total batch size", bsz))
    print(line("Grad accumulation steps", grad_accum_steps))
    print()
    print(line("LR (min, max)", str((min_lr, max_lr))))
    print(line("Warmup steps", cfg.hparams.warmup_steps))
    print(line("Grad clip norm", cfg.hparams.grad_clip_norm))
    print(line("Use Muon", cfg.hparams.use_muon))
    print(line("Muon peak LR floor", cfg.hparams.muon_peak_lr_floor))
    print(line("LR decay steps override", cfg.hparams.lr_decay_steps))
    print(line("Weight decay", cfg.hparams.weight_decay), "\n")
    print("-" * 75)

    if cfg.tracking.track:
        wandb_run, _, run_name = init_wandb(
            cfg,
            job_type="pretrain",
            extra_config={
                "exp_name": cfg.tracking.exp_name,
                "batch_size": bsz,
                "grad_accum_steps": grad_accum_steps,
                "num_devices": len(devices),
                "model_backend": cfg.model.model_backend,
                "depth": cfg.model.num_layers,
                "num_layers": cfg.model.num_layers,
                "num_experts": cfg.model.num_experts,
                "expert_hidden_dim": cfg.model.d_emb,
                "d_emb": cfg.model.d_emb,
                "q_heads": cfg.model.q_heads,
                "kv_heads": cfg.model.kv_heads,
                "experts_aggregation_regime": cfg.model.experts_aggregation_regime,
                "qkv_input_gelu_proj": cfg.model.qkv_input_gelu_proj,
                "compute_ratio_vs_16x768": compute_ratio,
                "max_lr": cfg.hparams.max_lr,
                "warmup_steps": cfg.hparams.warmup_steps,
                "grad_clip_norm": cfg.hparams.grad_clip_norm,
                "use_muon": cfg.hparams.use_muon,
                "muon_peak_lr_floor": cfg.hparams.muon_peak_lr_floor,
                "lr_decay_steps": cfg.hparams.lr_decay_steps,
                "train_files": num_train_files,
                "val_files": num_val_files,
                "dataloader_mode": dataloader_mode,
                "script": "nanogpt/train.py",
            },
        )
        if wandb_run is not None:
            print(f"W&B tracking enabled: {run_name}")
        else:
            print(f"W&B tracking disabled after init failure: {run_name}")

    # Compute the frequencies
    positions = jnp.arange(seqlen)[None, :]
    with mesh_context(cfg.mesh):
        freqs = regular_model.precompute_frequencies(
            positions=positions,
            features=head_dim,
        )

    # Because our dataloader already ensures that sequence in a batch have
    # tokens equal to the context window, we do not need sequence packing here
    # Hence, we can segment_ids to None for pretraining.
    segment_ids = None
    resume_from_step = cfg.ckpt_cfg.last_checkpoint_step

    if resume_from_step > 0:
        resume_ckpt_path = os.path.join(
            cfg.ckpt_cfg.save_ckpt_dir, str(resume_from_step)
        )
        if os.path.exists(resume_ckpt_path):
            from checkpoint_utils import load_checkpoint

            model, optim_state, train_iter = load_checkpoint(
                mngr, resume_from_step, model, optim_state, mesh, train_iter
            )
        else:
            resume_from_step = 0
            print(
                f"Checkpoint path {resume_ckpt_path} not found! Resuming training without restoring checkpoint..."
            )

    best_loss = float("inf")
    last_val_loss = float("inf")
    es_patience = cfg.hparams.es_patience
    es_patience_counter = 0
    best_step = 0
    num_shards_used = 0
    total_tokens_consumed = 0

    # Reusable data buffers
    grad_accum_batch = np.zeros((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)
    val_data_buf = np.zeros((bsz, seqlen + 1), dtype=np.uint16)

    step = resume_from_step
    print("Starting training (the first step will take some time for compilation...)\n")

    training_complete = False
    train_start_time = time.time()

    # Training loop with explicit counter
    for shard in train_iter:
        if step >= total_train_steps or training_complete:
            mngr.wait_until_finished()
            print("Finished checkpointing! Cleaned.")
            break

        tokens = shard["tokens"]
        size = shard["size"]
        shard_name = Path(shard["path"]).name

        try:
            batch_sampler = make_window_sampler(
                tokens,
                size=size,
            )
            shard_processed_fully = False

            # build the static index once per shard (on-demand)
            num_batches_in_shard = batch_sampler.build(bsz, seqlen)
            print(f"\n=== Processing Shard: {num_shards_used} with name: {shard_name}", end=" | ")  # fmt: off
            print(f"Indexed {num_batches_in_shard} batches ===")

            while not shard_processed_fully:
                try:
                    start = time.time()
                    for micro_step in range(grad_accum_steps):
                        starts, ends = batch_sampler.next_batch(bsz, seqlen)
                        get_next_batch(
                            starts,
                            ends,
                            bsz,
                            seqlen,
                            tokens,
                            data_accum_sharding,
                            grad_accum_batch[micro_step],
                            transfer_to_device=False,
                        )
                    with mesh_context(cfg.mesh):
                        stacked_batch = jnp.asarray(
                            grad_accum_batch, dtype=jnp.int32, device=data_accum_sharding
                        )
                        stacked_x = stacked_batch[:, :, :-1]
                        stacked_y = stacked_batch[:, :, 1:]
                        model, loss, optim_state = train_step_accum(
                            model,
                            stacked_x,
                            stacked_y,
                            segment_ids,
                            freqs,
                            optim_state,
                            optim,
                            grad_accum_steps,
                        )

                    # Block for accurate timing
                    jax.block_until_ready(loss)
                    end = time.time()
                    dt = end - start
                    train_time_elapsed = (end - train_start_time) / 60  # in minutes
                    tokens_processed = bsz * seqlen * grad_accum_steps
                    total_tokens_consumed += tokens_processed
                    tokens_per_sec = int(tokens_processed / dt)

                    # fmt: off
                    print(f"Step: [{str(step).zfill(len(str(total_train_steps)))}/{total_train_steps}] | loss: {loss:8.4f} | Step time: {dt:5.2f} s | Train time: {train_time_elapsed:6.2f} min | Tokens processed/s: {tokens_per_sec:>9,}")
                    # fmt: on
                    current_step = step
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": float(loss),
                                "train/step_time_sec": dt,
                                "train/train_time_min": train_time_elapsed,
                                "train/tokens_processed": tokens_processed,
                                "train/tokens_per_sec": tokens_per_sec,
                                "train/total_tokens_consumed": total_tokens_consumed,
                                "data/shards_used": num_shards_used,
                            },
                            step=current_step,
                        )

                    step += 1

                    if (step % options.save_interval_steps) == 0:
                        ckpt_items = {
                            "params": model,
                            "optim_state": optim_state,
                        }
                        save_items = {
                            name: prepare_for_checkpoint_save(tree, cfg.mesh)
                            for name, tree in ckpt_items.items()
                        }
                        for name, tree in save_items.items():
                            assert_checkpoint_payload_is_host(name, tree)
                        mngr.save(
                            step,
                            args=ocp.args.Composite(
                                params=ocp.args.PyTreeSave(save_items["params"]),
                                optim_state=ocp.args.PyTreeSave(
                                    save_items["optim_state"]
                                ),
                                ds=grain.checkpoint.CheckpointSave(train_iter),
                            ),
                        )

                    if step >= total_train_steps:
                        print(
                            f"\nReached maximum training steps  : {total_train_steps}"
                        )
                        print(f"Total number of shards consumed : {num_shards_used}")
                        print(f"Best loss : {best_loss:.4f} at step {best_step}")
                        mngr.wait_until_finished()
                        print("Finished checkpointing! Cleaned.")
                        training_complete = True
                        break

                except StopIteration:
                    # Once we have trained on one shard, let's validate the performance as well
                    shard_processed_fully = True
                    num_shards_used += 1
                    print("Shard exhausted")
                    print(f"Total shards consumed: {num_shards_used:<5}")
                    print(f"Total Tokens consumed: {total_tokens_consumed:>9,}")
                    print("-" * 75)

                    print("\nScoring model performance on validation data...\n")
                    val_loss = 0.0
                    val_steps_count = 0
                    val_iter = iter(val_dl)
                    for val_shard in val_iter:
                        val_tokens = val_shard["tokens"]
                        try:
                            val_batch_sampler = make_window_sampler(
                                val_tokens,
                                size=val_shard["size"],
                            )

                            num_val_batches = val_batch_sampler.build(bsz, seqlen)
                            if num_val_batches <= 0:
                                continue

                            for _ in range(num_val_batches):
                                starts, ends = val_batch_sampler.next_batch(bsz, seqlen)
                                get_next_batch(
                                    starts,
                                    ends,
                                    bsz,
                                    seqlen,
                                    val_tokens,
                                    data_sharding,
                                    val_data_buf,
                                )

                                with mesh_context(cfg.mesh):
                                    curr_val_data = jnp.asarray(
                                        val_data_buf, dtype=jnp.int32, device=data_sharding
                                    )
                                    x = curr_val_data[:, :-1]
                                    y = curr_val_data[:, 1:]
                                    loss = val_step(model, x, y, segment_ids, freqs)
                                val_loss += loss.item()
                                val_steps_count += 1
                        finally:
                            val_tokens.unlink_on_del()
                    avg_val_loss = val_loss / val_steps_count
                    avg_val_loss = jax.block_until_ready(avg_val_loss)
                    improved = avg_val_loss < best_loss
                    if improved:
                        best_loss = avg_val_loss
                        best_step = step
                        es_patience_counter = 0
                    else:
                        es_patience_counter += 1

                    if es_patience_counter > es_patience:
                         # fmt: off
                        print(f"\nEarly stopping triggered! No improvement for {es_patience_counter} steps.")
                        print(f"Total number of shards consumed : {num_shards_used}")
                        print(f"Best loss                       : {best_loss:.4f} at step {best_step}")
                         # fmt: on
                        mngr.wait_until_finished()
                        training_complete = True
                        break

                    print(f"last_val_loss : {last_val_loss:.4f}")
                    print(f"curr_val_loss : {avg_val_loss:.4f}")
                    print(f"Best loss     : {best_loss:.4f} at step {best_step}\n")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/loss": avg_val_loss,
                                "val/last_loss": last_val_loss,
                                "val/best_loss": best_loss,
                                "val/best_step": best_step,
                                "val/improved": float(improved),
                                "train/es_patience_counter": es_patience_counter,
                                "data/shards_used": num_shards_used,
                                "train/total_tokens_consumed": total_tokens_consumed,
                            },
                            step=step,
                        )
                    last_val_loss = avg_val_loss
        finally:
            tokens.unlink_on_del()
    train_end_time = time.time()
    total_train_time_min = (train_end_time - train_start_time) / 60
    if wandb_run is not None:
        wandb_run.summary["train/total_time_min"] = total_train_time_min
        wandb_run.summary["train/total_tokens_consumed"] = total_tokens_consumed
        wandb_run.summary["val/best_loss"] = best_loss
        wandb_run.summary["val/best_step"] = best_step
        wandb_run.finish()
    print(
        f"\nTotal time taken to train the model: {total_train_time_min:.2f} minutes"
    )


if __name__ == "__main__":
    main()
