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


from model import precompute_frequencies
from model_moe import GPT, count_params, forward
from utils import logical_to_sharding
from optim import build_optimizer
from config import ShardingRules, Config, BATCH_AXIS_NAME, DEFAULT_FINEWEB_DIR
from fineweb_dataloader import make_grain_shard_loader, make_window_sampler, load_shard_tokens
from logging_utils import init_wandb
from jax_compat import mesh_context
from checkpoint_utils import assert_checkpoint_payload_is_host
from checkpoint_utils import get_sharding_for_checkpoint
from checkpoint_utils import prepare_for_checkpoint_save


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


def _promote_inexact_leaves_to_f32(tree):
    return jax.tree.map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact)
        else x,
        tree,
    )


def compute_loss(params, x_batch, y_batch, segment_ids, freqs, loss_mask):
    logits = forward(params, x_batch, segment_ids, freqs)
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
    if effective_token_batch_size != desired_batch_size:
        raise ValueError(
            "Configured desired_batch_size must be exactly divisible by the "
            "global micro-batch token count: "
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


def make_stream_checkpoint_state(active_shard_index, active_batch_iter, mesh):
    """Record the active shard and next window at an optimizer-update boundary."""
    with mesh_context(mesh):
        return {
            "active_shard_index": jnp.array(active_shard_index, dtype=jnp.int32),
            "active_batch_iter": jnp.array(active_batch_iter, dtype=jnp.int32),
        }


def make_checkpoint_metadata(cfg, train_files, batch_size, grad_accum_steps):
    return {
        "version": 2,
        "train_files": [path.name for path in train_files],
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "model": {
            name: getattr(cfg.model, name)
            for name in (
                "num_layers",
                "num_experts",
                "d_emb",
                "q_heads",
                "kv_heads",
                "seqlen",
                "vocab_size",
            )
        },
        "dtype": str(jnp.dtype(cfg.model.dtype)),
        "optimizer": {
            name: getattr(cfg.hparams, name)
            for name in (
                "max_lr",
                "warmup_steps",
                "total_train_steps",
                "b1",
                "b2",
                "embedding_lr",
                "weight_decay",
                "cautious_weight_decay",
                "grad_clip_norm",
                "use_muon",
                "muon_peak_lr_floor",
            )
        },
    }


def validate_resume_checkpoint(stream_state, metadata, expected_metadata):
    if metadata.get("version") == 1:
        metadata = {
            **metadata,
            "version": 2,
            "model": dict(metadata["model"]),
            "optimizer": dict(metadata["optimizer"]),
        }
        if metadata["model"].pop("model_backend", None) != "moe":
            raise ValueError("Checkpoint must use stacked expert parameters.")
        decay_steps = metadata["optimizer"].pop("lr_decay_steps", None)
        default_decay_steps = max(
            1,
            metadata["optimizer"]["total_train_steps"]
            - metadata["optimizer"]["warmup_steps"],
        )
        if decay_steps not in (None, default_decay_steps):
            raise ValueError("Checkpoint uses a different learning-rate decay schedule.")
    if metadata != expected_metadata:
        raise ValueError(
            "Checkpoint training data, batch layout, model, or optimizer settings "
            "do not match this run."
        )
    shard_index = int(stream_state["active_shard_index"])
    batch_iter = int(stream_state["active_batch_iter"])
    if not 0 <= shard_index < len(expected_metadata["train_files"]):
        raise ValueError(f"Checkpoint has an invalid active shard index: {shard_index}")
    if batch_iter <= 0 or batch_iter % expected_metadata["grad_accum_steps"]:
        raise ValueError(f"Checkpoint has an invalid sampler cursor: {batch_iter}")


def dense_layer_compute(width, seqlen, q_heads, kv_heads):
    kv_ratio = kv_heads / q_heads
    projection_and_mlp = (10.0 + 2.0 * kv_ratio) * width * width
    sequence_attention = 2.0 * seqlen * width
    return projection_and_mlp + sequence_attention


def estimate_compute_ratio(cfg):
    baseline = 16.0 * dense_layer_compute(768, cfg.model.seqlen, 8, 4)
    active_layers = cfg.model.num_layers * cfg.model.num_experts
    compute = active_layers * dense_layer_compute(
        cfg.model.d_emb,
        cfg.model.seqlen,
        cfg.model.q_heads,
        cfg.model.kv_heads,
    )
    return compute / baseline


def get_next_batch(starts, ends, seqlen, tokens, buf_u16):
    """Fill a reusable buffer with input-label windows from the shard."""
    ptr = 0
    for i, j in zip(starts, ends):
        n = j - i
        row = ptr // (seqlen + 1)
        col = ptr % (seqlen + 1)
        buf_u16[row, col : col + n] = tokens[i:j]
        ptr += n


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
        "--num_layers",
        type=int,
        default=None,
        help="Override the number of transformer layers from config.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Alias for --num_layers.",
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
        help="Alias for --d_emb.",
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
    cli_args = parser.parse_args()

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
    if cfg.ckpt_cfg.last_checkpoint_step < 0:
        raise ValueError("--last_checkpoint_step must be non-negative.")
    if cfg.ckpt_cfg.last_checkpoint_step > 0:
        resume_checkpoint = (
            Path(cfg.ckpt_cfg.save_ckpt_dir) / str(cfg.ckpt_cfg.last_checkpoint_step)
        )
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Requested resume checkpoint is missing: {resume_checkpoint}"
            )
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
    if model_updates:
        cfg.model = dataclasses.replace(cfg.model, **model_updates)

    train_files = sorted(Path(cfg.data_dir).glob("*train*.bin"))
    val_files = sorted(Path(cfg.data_dir).glob("*val*.bin"))
    train_file_to_index = {
        str(path.resolve()): index for index, path in enumerate(train_files)
    }
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
    compute_ratio = estimate_compute_ratio(cfg)

    # Load the model
    print("Building GPT model based on the config...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
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
        ),
    )

    if grad_accum_steps > 1:
        print("Using `MultiSteps` in optax for gradient accumulation...")
        optim = optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)

    optim_state = optim.init(model)
    if grad_accum_steps > 1:
        optim_state = _promote_inexact_leaves_to_f32(optim_state)
    stream_ckpt_state = make_stream_checkpoint_state(-1, 0, cfg.mesh)
    checkpoint_metadata = make_checkpoint_metadata(cfg, train_files, bsz, grad_accum_steps)

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
        "stream_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "metadata": ocp.Checkpointer(ocp.JsonCheckpointHandler()),
    }

    mngr = ocp.CheckpointManager(ckpt_path, handlers, options=options)

    print("")
    print("-" * 75)
    print("")

    print(
        line(
            "Number of trainable params: ",
            count_params(model),
            comma=True,
        )
    )
    print(line("Depth", cfg.model.num_layers))
    print(line("Number of experts", cfg.model.num_experts))
    print(line("Expert hidden dim", cfg.model.d_emb))
    print(line("Query heads", cfg.model.q_heads))
    print(line("KV heads", cfg.model.kv_heads))
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
                "depth": cfg.model.num_layers,
                "num_layers": cfg.model.num_layers,
                "num_experts": cfg.model.num_experts,
                "expert_hidden_dim": cfg.model.d_emb,
                "d_emb": cfg.model.d_emb,
                "q_heads": cfg.model.q_heads,
                "kv_heads": cfg.model.kv_heads,
                "compute_ratio_vs_16x768": compute_ratio,
                "max_lr": cfg.hparams.max_lr,
                "warmup_steps": cfg.hparams.warmup_steps,
                "grad_clip_norm": cfg.hparams.grad_clip_norm,
                "use_muon": cfg.hparams.use_muon,
                "muon_peak_lr_floor": cfg.hparams.muon_peak_lr_floor,
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
        freqs = precompute_frequencies(
            positions=positions,
            features=head_dim,
        )

    # Because our dataloader already ensures that sequence in a batch have
    # tokens equal to the context window, we do not need sequence packing here
    # Hence, we can segment_ids to None for pretraining.
    segment_ids = None
    resume_from_step = cfg.ckpt_cfg.last_checkpoint_step
    resumed_active_shard = None

    if resume_from_step > 0:
        resume_ckpt_path = ckpt_path / str(resume_from_step)
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"Requested resume checkpoint is missing: {resume_ckpt_path}")
        if not all((resume_ckpt_path / name).exists() for name in ("stream_state", "metadata")):
            raise ValueError(
                "Checkpoint lacks the saved sampler cursor or training metadata "
                "required for exact resume."
            )
        preflight = mngr.restore(
            resume_from_step,
            args=ocp.args.Composite(
                stream_state=ocp.args.PyTreeRestore(),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        validate_resume_checkpoint(
            preflight.stream_state, preflight.metadata, checkpoint_metadata,
        )

        def restore_args(tree):
            return jax.tree.map(
                lambda leaf: ocp.ArrayRestoreArgs(
                    sharding=get_sharding_for_checkpoint(leaf, mesh)
                ),
                tree,
            )

        with mesh_context(cfg.mesh):
            restored = mngr.restore(
                resume_from_step,
                args=ocp.args.Composite(
                    params=ocp.args.PyTreeRestore(
                        item=model, restore_args=restore_args(model),
                    ),
                    optim_state=ocp.args.PyTreeRestore(
                        item=optim_state, restore_args=restore_args(optim_state),
                    ),
                    ds=grain.checkpoint.CheckpointRestore(train_iter),
                    stream_state=ocp.args.PyTreeRestore(
                        item=stream_ckpt_state, restore_args=restore_args(stream_ckpt_state),
                    ),
                ),
            )
        model = restored.params
        optim_state = restored.optim_state
        train_iter = restored.ds
        stream_ckpt_state = restored.stream_state
        resumed_active_shard = load_shard_tokens(
            train_files[int(stream_ckpt_state["active_shard_index"])]
        )

    best_loss = float("inf")
    last_val_loss = float("inf")
    best_step = 0
    num_shards_used = (
        int(stream_ckpt_state["active_shard_index"])
        if resumed_active_shard is not None else 0
    )
    tokens_per_train_step = bsz * seqlen * grad_accum_steps
    total_tokens_consumed = resume_from_step * tokens_per_train_step

    # Reusable data buffers
    grad_accum_batch = np.zeros((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)
    val_data_buf = np.zeros((bsz, seqlen + 1), dtype=np.uint16)

    step = resume_from_step
    print("Starting training (the first step will take some time for compilation...)\n")

    training_complete = False
    train_start_time = time.time()

    def iterate_train_shards():
        if resumed_active_shard is not None:
            yield resumed_active_shard, int(stream_ckpt_state["active_batch_iter"])
        for next_shard in train_iter:
            yield next_shard, 0

    # Training loop with explicit counter
    for shard, initial_batch_iter in iterate_train_shards():
        if step >= total_train_steps or training_complete:
            shard["tokens"].unlink_on_del()
            mngr.wait_until_finished()
            print("Finished checkpointing! Cleaned.")
            break

        tokens = shard["tokens"]
        size = shard["size"]
        shard_name = Path(shard["path"]).name
        shard_index = train_file_to_index[str(Path(shard["path"]).resolve())]

        try:
            batch_sampler = make_window_sampler(
                tokens,
                size=size,
            )
            shard_processed_fully = False

            # build the static index once per shard (on-demand)
            num_batches_in_shard = batch_sampler.build(bsz, seqlen)
            if not 0 <= initial_batch_iter <= num_batches_in_shard:
                raise ValueError(
                    "Checkpoint sampler cursor is outside the active shard: "
                    f"{initial_batch_iter} > {num_batches_in_shard}"
                )
            batch_sampler.batch_iter = initial_batch_iter
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
                            seqlen,
                            tokens,
                            grad_accum_batch[micro_step],
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
                    tokens_processed = tokens_per_train_step
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
                        stream_ckpt_state = make_stream_checkpoint_state(
                            shard_index, batch_sampler.batch_iter, cfg.mesh,
                        )
                        ckpt_items = {
                            "params": model,
                            "optim_state": optim_state,
                            "stream_state": stream_ckpt_state,
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
                                stream_state=ocp.args.PyTreeSave(save_items["stream_state"]),
                                metadata=ocp.args.JsonSave(checkpoint_metadata),
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
                                    seqlen,
                                    val_tokens,
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
                                "data/shards_used": num_shards_used,
                                "train/total_tokens_consumed": total_tokens_consumed,
                            },
                            step=step,
                        )
                    last_val_loss = avg_val_loss
        finally:
            tokens.unlink_on_del()
    mngr.wait_until_finished()
    mngr.close()
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
