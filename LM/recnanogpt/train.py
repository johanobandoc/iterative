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
import contextlib
from pathlib import Path
from functools import partial

import jax

jax.config.update("jax_optimization_level", "O1")

import optax
import grain
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.tree_util import DictKey, GetAttrKey, SequenceKey
from jax.sharding import Mesh

from utils import logical_to_sharding
from optim import build_optimizer
from config import ShardingRules, Config, BATCH_AXIS_NAME, DEFAULT_FINEWEB_DIR
from fineweb_dataloader import (
    make_grain_shard_loader,
    make_window_sampler,
    load_shard_tokens,
)
from checkpoint_utils import assert_checkpoint_payload_is_host
from checkpoint_utils import get_sharding_for_checkpoint
from checkpoint_utils import prepare_for_checkpoint_save
from logging_utils import init_wandb
from model_shared_kv import (
    GPT,
    count_params,
    forward_with_state,
    forward_loss_with_state_and_terms,
    init_recurrent_state,
    validate_recurrent_model_config,
)


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


try:
    from jax.sharding import set_mesh as _set_mesh
except ImportError:
    _set_mesh = getattr(jax, "set_mesh", None)


def mesh_context(mesh):
    if mesh is None or _set_mesh is None:
        return contextlib.nullcontext()

    ctx = _set_mesh(mesh)
    if hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__"):
        return ctx
    return contextlib.nullcontext()


def _path_names(path):
    names = []
    for key in path:
        if isinstance(key, GetAttrKey):
            names.append(key.name)
        elif isinstance(key, SequenceKey):
            names.append(str(key.idx))
        elif isinstance(key, DictKey):
            names.append(str(key.key))
        else:
            names.append(str(key))
    return names


def _zero_size_leaf_paths(tree):
    zero_paths = []

    def visit(path, leaf):
        if getattr(leaf, "size", None) == 0:
            zero_paths.append(".".join(_path_names(path)))
        return None

    jax.tree_util.tree_map_with_path(visit, tree)
    return zero_paths



def init_accumulated_state(
    init_recurrent_state, params, *, grad_accum_steps, physical_batch_size, mesh, rules
):
    """Initialize shared-K/V state with a leading accumulation-group axis."""
    shapes = jax.eval_shape(lambda: init_recurrent_state(params, physical_batch_size))
    state = []
    # Residual state is batch-first; the remaining shared-K/V state is batch-second.
    for batch_axis, shape in zip((0, 1, 1, 1, 1, 1), shapes, strict=True):
        if shape.shape[batch_axis] != physical_batch_size:
            raise ValueError("Unexpected recurrent-state batch axis.")
        grouped_shape = (grad_accum_steps, *shape.shape)
        axes = [None] * len(grouped_shape)
        axes[batch_axis + 1] = "batch"
        state.append(
            jnp.zeros(
                grouped_shape,
                dtype=shape.dtype,
                device=logical_to_sharding(tuple(axes), mesh, rules),
            )
        )
    return tuple(state)


def make_train_step_streaming_accum(forward_loss_with_state_and_terms):
    @partial(
        jax.jit,
        static_argnames=("optim", "grad_accum_steps"),
        donate_argnums=(0, 1, 3, 4, 5),
    )
    def train_step_streaming_accum(
        params,
        x_batch,
        y_batch,
        slot_reset_mask_batch,
        recurrent_state,
        optim_state,
        optim,
        grad_accum_steps,
        pre_output_reg_cost,
    ):
        if x_batch.shape[0] != grad_accum_steps:
            raise ValueError("Inputs must have one group per accumulation step.")
        if recurrent_state[0].shape[:2] != x_batch.shape[:2]:
            raise ValueError("Recurrent state must start with [group, batch, ...].")

        def accumulator_leaf(value):
            dtype = (
                jnp.float32 if jnp.issubdtype(value.dtype, jnp.inexact) else value.dtype
            )
            return jnp.zeros(value.shape, dtype=dtype)

        def body(carry, inputs):
            grad_sum, loss_sum, terms_sum = carry
            xb, yb, reset_mask, group_state = inputs

            def loss_fn(loss_params):
                loss, next_state, terms = forward_loss_with_state_and_terms(
                    loss_params,
                    xb,
                    yb,
                    group_state,
                    slot_reset_mask=reset_mask,
                    loss_mask=None,
                    pre_output_reg_cost=pre_output_reg_cost,
                )
                return loss, (next_state, terms)

            (loss, (next_state, terms)), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            grad_sum = jax.tree.map(
                lambda total, grad: total + grad.astype(total.dtype),
                grad_sum,
                grads,
            )
            return (grad_sum, loss_sum + loss, terms_sum + terms), jax.tree.map(
                jax.lax.stop_gradient,
                next_state,
            )

        carry0 = (
            jax.tree.map(accumulator_leaf, params),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
        )
        (grad_sum, loss_sum, terms_sum), next_state = jax.lax.scan(
            body,
            carry0,
            (x_batch, y_batch, slot_reset_mask_batch, recurrent_state),
            length=grad_accum_steps,
        )
        mean_grads = jax.tree.map(
            lambda total, param: (total / grad_accum_steps).astype(param.dtype),
            grad_sum,
            params,
        )
        updates, optim_state = optim.update(mean_grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return (
            params,
            loss_sum / grad_accum_steps,
            terms_sum / grad_accum_steps,
            optim_state,
            next_state,
        )

    return train_step_streaming_accum


def make_val_step_streaming(forward_loss_with_state_and_terms):
    @partial(jax.jit, donate_argnums=(3,))
    def val_step_streaming(
        params, x_batch, y_batch, recurrent_state, slot_reset_mask, pre_output_reg_cost
    ):
        def body(carry, inputs):
            loss_sum, terms_sum = carry
            xb, yb, group_state, reset_mask = inputs
            loss, next_state, terms = forward_loss_with_state_and_terms(
                params,
                xb,
                yb,
                group_state,
                slot_reset_mask=reset_mask,
                loss_mask=None,
                pre_output_reg_cost=pre_output_reg_cost,
            )
            return (loss_sum + loss, terms_sum + terms), jax.tree.map(
                jax.lax.stop_gradient,
                next_state,
            )

        (loss_sum, terms_sum), next_state = jax.lax.scan(
            body,
            (jnp.array(0.0, dtype=jnp.float32), jnp.zeros((2,), dtype=jnp.float32)),
            (x_batch, y_batch, recurrent_state, slot_reset_mask),
        )
        return loss_sum / x_batch.shape[0], next_state, terms_sum / x_batch.shape[0]

    return val_step_streaming


def make_stream_warmup_step(forward_with_state):
    @partial(jax.jit, donate_argnums=(3,))
    def stream_warmup_step(params, x_batch, slot_reset_mask_batch, recurrent_state):
        def group_body(unused, inputs):
            group_state, x_segments, reset_segments = inputs

            def segment_body(current_state, segment_inputs):
                xb, reset_mask = segment_inputs
                _, next_state = forward_with_state(
                    params,
                    xb,
                    current_state,
                    slot_reset_mask=reset_mask,
                )
                return jax.tree.map(jax.lax.stop_gradient, next_state), None

            next_state, _ = jax.lax.scan(
                segment_body, group_state, (x_segments, reset_segments)
            )
            return unused, next_state

        _, next_state = jax.lax.scan(
            group_body,
            None,
            (
                recurrent_state,
                jnp.swapaxes(x_batch, 0, 1),
                jnp.swapaxes(slot_reset_mask_batch, 0, 1),
            ),
        )
        return next_state

    return stream_warmup_step


def line(label, value, comma=False, label_w=30, colon_w=2, value_w=20):
    fmt = f">{value_w}," if comma else f">{value_w}"
    return f"{label:<{label_w}}{':':<{colon_w}}{value:{fmt}}"


def resolve_grad_accum_steps(desired_batch_size, global_batch_size, seqlen):
    if min(desired_batch_size, global_batch_size, seqlen) < 1:
        raise ValueError(
            "Token batch size, physical batch size, and sequence length must be positive."
        )
    micro_batch_tokens = global_batch_size * seqlen
    steps, remainder = divmod(desired_batch_size, micro_batch_tokens)
    if steps < 1 or remainder:
        raise ValueError(
            "desired_batch_size must be exactly divisible by the physical token batch "
            f"and at least one microbatch: got {desired_batch_size} and {micro_batch_tokens}."
        )
    return steps


def make_stream_checkpoint_state(
    active_shard_index, active_batch_iter, mesh, *, accumulation_layout
):
    with mesh_context(mesh):
        values = {
            "active_shard_index": active_shard_index,
            "active_batch_iter": active_batch_iter,
            "accumulation_layout_version": 1,
            **accumulation_layout,
        }
        return {key: jnp.array(value, dtype=jnp.int32) for key, value in values.items()}


def validate_stream_checkpoint_state(state, accumulation_layout):
    expected = {"accumulation_layout_version": 1, **accumulation_layout}
    mismatches = {
        key: (None if key not in state else int(state[key]), value)
        for key, value in expected.items()
        if key not in state or int(state[key]) != value
    }
    if mismatches:
        raise ValueError(
            f"Checkpoint accumulation layout does not match this run: {mismatches}"
        )


def get_next_batch(batch_sampler, tokens, buffer):
    """Fill [accumulation group, physical batch, sequence + 1] token windows."""
    groups, physical_batch_size, window_size = buffer.shape
    logical_batch_size = groups * physical_batch_size
    starts, ends, reset_mask = batch_sampler.next_batch_with_reset(
        logical_batch_size, window_size - 1
    )
    flat = buffer.reshape(logical_batch_size, window_size)
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        flat[row] = tokens[start:end]
    return reset_mask.reshape(groups, physical_batch_size)


def main():
    parser = argparse.ArgumentParser(description="nanoGPTJAX MoE pretraining")
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
        "--warmup_steps",
        type=int,
        default=None,
        help="Override optimizer warmup steps.",
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
        help="Override how many checkpoints to retain on disk.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=None,
        help="Override the training sequence length from config.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="Override the number of recurrent experts from config.",
    )
    parser.add_argument(
        "--expert_hidden_dim",
        type=int,
        default=None,
        help="Override the recurrent expert hidden dimension while rebuilding derived model subconfigs.",
    )
    parser.add_argument(
        "--q_heads",
        type=int,
        default=None,
        help="Override the number of query heads while rebuilding all derived model subconfigs.",
    )
    parser.add_argument(
        "--kv_heads",
        type=int,
        default=None,
        help="Override the number of KV heads while rebuilding all derived model subconfigs.",
    )
    parser.add_argument(
        "--memory_len",
        type=int,
        default=None,
        help="Override the recurrent KV memory length used by the main stack.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="For the depth-stacked shared-K/V backend, run this many independent shared-K/V cores in sequence on each recurrent sweep.",
    )
    parser.add_argument(
        "--checkpoint_token_step",
        action="store_true",
        help="Enable activation checkpointing for the recurrent per-token step.",
    )
    parser.add_argument(
        "--segment_local_kv_cache",
        action="store_true",
        help=(
            "Keep the long recurrent KV cache read-only inside a segment, "
            "cache only the segment-local K/V during the token scan, and "
            "append the segment to long memory once after the scan."
        ),
    )
    parser.add_argument(
        "--other_peak_lr",
        type=float,
        default=None,
        help="Override the Muon/other-weights peak learning rate from config.",
    )
    parser.add_argument(
        "--cautious_weight_decay",
        type=float,
        default=None,
        help="Override the cautious weight decay applied after the base optimizer.",
    )
    parser.add_argument(
        "--pre_output_reg_cost",
        type=float,
        default=None,
        help="L2 penalty coefficient on the final residual stream before the LM head.",
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
        cfg.hparams.warmup_steps = int(min(300, 0.01 * cfg.hparams.total_train_steps))
    if cli_args.warmup_steps is not None:
        cfg.hparams.warmup_steps = cli_args.warmup_steps
    model_updates = {}
    if cli_args.seqlen is not None:
        model_updates["seqlen"] = cli_args.seqlen
    if cli_args.num_experts is not None:
        model_updates["num_experts"] = cli_args.num_experts
    if cli_args.expert_hidden_dim is not None:
        model_updates["expert_hidden_dim"] = cli_args.expert_hidden_dim
    if cli_args.q_heads is not None:
        model_updates["q_heads"] = cli_args.q_heads
    if cli_args.kv_heads is not None:
        model_updates["kv_heads"] = cli_args.kv_heads
    if cli_args.memory_len is not None:
        model_updates["memory_len"] = cli_args.memory_len
    if cli_args.depth is not None:
        model_updates["depth"] = cli_args.depth
    if cli_args.checkpoint_token_step:
        model_updates["checkpoint_token_step"] = True
    if cli_args.segment_local_kv_cache:
        model_updates["segment_local_kv_cache"] = True
    if model_updates:
        cfg.model = dataclasses.replace(cfg.model, **model_updates)
    if cfg.model.expert_hidden_dim % cfg.model.q_heads != 0:
        raise ValueError(
            f"`expert_hidden_dim` must be divisible by `q_heads`, got {cfg.model.expert_hidden_dim} and {cfg.model.q_heads}."
        )
    if cfg.model.q_heads % cfg.model.kv_heads != 0:
        raise ValueError(
            f"`q_heads` must be divisible by `kv_heads`, got {cfg.model.q_heads} and {cfg.model.kv_heads}."
        )
    if cfg.model.depth < 1:
        raise ValueError(
            "`depth` must be >= 1, "
            f"got {cfg.model.depth}."
        )
    if cli_args.other_peak_lr is not None:
        cfg.hparams.other_peak_lr = cli_args.other_peak_lr
    if cli_args.cautious_weight_decay is not None:
        cfg.hparams.cautious_weight_decay = cli_args.cautious_weight_decay
    if cli_args.pre_output_reg_cost is not None:
        cfg.hparams.pre_output_reg_cost = cli_args.pre_output_reg_cost
    train_step_streaming_accum = make_train_step_streaming_accum(
        forward_loss_with_state_and_terms
    )
    val_step_streaming = make_val_step_streaming(forward_loss_with_state_and_terms)
    stream_warmup_step = make_stream_warmup_step(forward_with_state)
    validate_recurrent_model_config(cfg.model)
    dataloader_mode = "stream_equal_chunks"

    train_files = sorted(Path(cfg.data_dir).glob("*train*.bin"))
    val_files = sorted(Path(cfg.data_dir).glob("*val*.bin"))
    train_file_to_index = {str(path): idx for idx, path in enumerate(train_files)}
    num_train_files = len(train_files)
    num_val_files = len(val_files)
    print("\nNumber of train files found: ", num_train_files)
    print("Number of validation files found: ", num_val_files)
    if num_train_files == 0 or num_val_files == 0:
        raise FileNotFoundError(
            f"No FineWeb train/val shards found in {cfg.data_dir}. "
            "Pass --data_dir with a directory containing *train*.bin and *val*.bin files."
        )

    train_dl = make_grain_shard_loader(train_files)
    val_dl = make_grain_shard_loader(val_files)
    train_iter = iter(train_dl)

    per_device_bsz = cfg.hparams.per_device_batch_size
    bsz = per_device_bsz * len(devices)
    seqlen = cfg.model.seqlen
    data_accum_sharding = logical_to_sharding(
        (None, "batch", None), cfg.mesh, cfg.rules
    )
    reset_accum_sharding = logical_to_sharding((None, "batch"), cfg.mesh, cfg.rules)
    warmup_sharding = logical_to_sharding((None, None, "batch", None), cfg.mesh, cfg.rules)
    warmup_reset_sharding = logical_to_sharding((None, None, "batch"), cfg.mesh, cfg.rules)

    other_peak_lr = cfg.hparams.other_peak_lr
    other_min_lr = 0.01 * other_peak_lr
    warmup_steps = cfg.hparams.warmup_steps
    desired_batch_size = cfg.hparams.desired_batch_size
    grad_accum_steps = resolve_grad_accum_steps(desired_batch_size, bsz, seqlen)
    cfg.hparams.grad_accum_steps = grad_accum_steps
    logical_bsz = bsz * grad_accum_steps
    accumulation_layout = {
        "grad_accum_steps": grad_accum_steps,
        "physical_batch_size": bsz,
        "sequence_length": seqlen,
    }
    stream_warmup_segments = (cfg.model.memory_len + seqlen - 1) // seqlen
    total_train_steps = cfg.hparams.total_train_steps
    max_checkpoints_to_keep = cfg.ckpt_cfg.max_checkpoints_to_keep
    checkpoint_save_steps = cfg.ckpt_cfg.checkpoint_save_steps
    wandb_run = None

    # Load the model
    print("Building GPT model based on the config...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    print("Model built successfully!")

    # Optimizer
    optim = optax.chain(
        optax.clip_by_global_norm(cfg.hparams.grad_clip_norm),
        build_optimizer(
            model,
            d_model=cfg.model.expert_hidden_dim,
            other_peak_lr=other_peak_lr,
            other_min_lr=other_min_lr,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            b1=cfg.hparams.b1,
            b2=cfg.hparams.b2,
            embedding_lr=cfg.hparams.embedding_lr,
            weight_decay=cfg.hparams.weight_decay,
            cautious_weight_decay=cfg.hparams.cautious_weight_decay,
            use_muon=True,
        ),
    )
    optim_state = optim.init(model)

    def new_recurrent_state():
        return init_accumulated_state(
            init_recurrent_state, model, grad_accum_steps=grad_accum_steps,
            physical_batch_size=bsz, mesh=cfg.mesh, rules=cfg.rules,
        )

    train_recurrent_state = new_recurrent_state()
    stream_ckpt_state = make_stream_checkpoint_state(
        -1, 0, cfg.mesh, accumulation_layout=accumulation_layout,
    )

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
    handlers["stream_state"] = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    handlers["recurrent_state"] = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    mngr = ocp.CheckpointManager(ckpt_path, handlers, options=options)

    print("")
    print("-" * 75)
    print("")

    print(line("Number of trainable params: ", count_params(model), comma=True))
    print(line("Number of experts", cfg.model.num_experts))
    print(line("Expert hidden dim", cfg.model.expert_hidden_dim))
    print(line("Query heads", cfg.model.q_heads))
    print(line("KV heads", cfg.model.kv_heads))
    print(line("Head dim", cfg.model.attn.head_dim))
    print(line("Sequence length per sample", seqlen))
    print(line("Dataloader mode", dataloader_mode))
    print(
        line(
            "Shared-global depth",
            cfg.model.depth,
        )
    )
    print(line("Checkpoint save steps", checkpoint_save_steps))
    print(line("Max checkpoints to keep", max_checkpoints_to_keep))
    print(line("Per device batch size", per_device_bsz))
    print(line("Total batch size", bsz))
    print(line("Grad accumulation steps", grad_accum_steps))
    print(line("Independent token streams", logical_bsz))
    print(line("Stream warmup segments", stream_warmup_segments))
    print(line("Segment-local KV cache", cfg.model.segment_local_kv_cache))
    print()
    print(line("Other LR (min, peak)", str((other_min_lr, other_peak_lr))))
    print(line("Warmup steps", cfg.hparams.warmup_steps))
    print(line("Cautious weight decay", cfg.hparams.cautious_weight_decay))
    print(line("Pre-output reg cost", cfg.hparams.pre_output_reg_cost))
    print(line("Weight decay", cfg.hparams.weight_decay))
    print()
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
                "num_experts": cfg.model.num_experts,
                "expert_hidden_dim": cfg.model.expert_hidden_dim,
                "q_heads": cfg.model.q_heads,
                "kv_heads": cfg.model.kv_heads,
                "head_dim": cfg.model.attn.head_dim,
                "depth": cfg.model.depth,
                "train_files": num_train_files,
                "val_files": num_val_files,
                "dataloader_mode": dataloader_mode,
                "segment_local_kv_cache": cfg.model.segment_local_kv_cache,
                "stream_warmup_segments": stream_warmup_segments,
                "pre_output_reg_cost": cfg.hparams.pre_output_reg_cost,
                "script": "recnanogpt/train.py",
            },
        )
        if wandb_run is not None:
            print(f"W&B tracking enabled: {run_name}")
        else:
            print(f"W&B tracking disabled after init failure: {run_name}")
    resume_from_step = cfg.ckpt_cfg.last_checkpoint_step
    resumed_active_shard = None

    if resume_from_step > 0:
        resume_ckpt_path = os.path.join(
            cfg.ckpt_cfg.save_ckpt_dir, str(resume_from_step)
        )
        if os.path.exists(resume_ckpt_path):
            metadata = mngr.restore(
                resume_from_step,
                args=ocp.args.Composite(stream_state=ocp.args.PyTreeRestore()),
            )
            validate_stream_checkpoint_state(metadata.stream_state, accumulation_layout)
            params_restore_args = jax.tree.map(
                lambda s: ocp.ArrayRestoreArgs(
                    sharding=get_sharding_for_checkpoint(s, mesh)
                ),
                model,
            )
            optim_restore_args = jax.tree.map(
                lambda s: ocp.ArrayRestoreArgs(
                    sharding=get_sharding_for_checkpoint(s, mesh)
                ),
                optim_state,
            )
            recurrent_restore_args = jax.tree.map(
                lambda s: ocp.ArrayRestoreArgs(
                    sharding=get_sharding_for_checkpoint(s, mesh)
                ),
                train_recurrent_state,
            )
            stream_restore_args = jax.tree.map(
                lambda s: ocp.ArrayRestoreArgs(
                    sharding=get_sharding_for_checkpoint(s, mesh)
                ),
                stream_ckpt_state,
            )
            with mesh_context(cfg.mesh):
                restored = mngr.restore(
                    resume_from_step,
                    args=ocp.args.Composite(
                        params=ocp.args.PyTreeRestore(
                            item=model,
                            restore_args=params_restore_args,
                        ),
                        optim_state=ocp.args.PyTreeRestore(
                            item=optim_state,
                            restore_args=optim_restore_args,
                        ),
                        ds=grain.checkpoint.CheckpointRestore(train_iter),
                        stream_state=ocp.args.PyTreeRestore(
                            item=stream_ckpt_state,
                            restore_args=stream_restore_args,
                        ),
                        recurrent_state=ocp.args.PyTreeRestore(
                            item=train_recurrent_state,
                            restore_args=recurrent_restore_args,
                        ),
                    ),
                )
            model = restored.params
            optim_state = restored.optim_state
            train_iter = restored.ds
            stream_ckpt_state = restored.stream_state
            train_recurrent_state = restored.recurrent_state
            active_shard_index = int(stream_ckpt_state["active_shard_index"])
            if active_shard_index >= 0:
                resumed_active_shard = load_shard_tokens(train_files[active_shard_index])
        else:
            raise FileNotFoundError(
                f"Requested resume checkpoint is missing: {resume_ckpt_path}"
            )

    best_loss = float("inf")
    last_val_loss = float("inf")
    best_step = 0
    num_shards_used = 0
    tokens_per_train_step = logical_bsz * seqlen
    total_tokens_consumed = int(resume_from_step) * tokens_per_train_step

    # Reusable data buffers
    grad_accum_batch = np.zeros((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)
    grad_accum_reset_mask = np.ones((grad_accum_steps, bsz), dtype=np.bool_)
    stream_warmup_batch = np.zeros(
        (max(1, stream_warmup_segments), grad_accum_steps, bsz, seqlen + 1),
        dtype=np.uint16,
    )
    stream_warmup_reset_mask = np.ones(
        (max(1, stream_warmup_segments), grad_accum_steps, bsz),
        dtype=np.bool_,
    )
    val_data_buf = np.zeros((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)

    def warmup_state(sampler, shard_tokens, state):
        for index in range(stream_warmup_segments):
            stream_warmup_reset_mask[index] = get_next_batch(
                sampler, shard_tokens, stream_warmup_batch[index],
            )
        with mesh_context(cfg.mesh):
            batch = jnp.asarray(
                stream_warmup_batch[:stream_warmup_segments],
                dtype=jnp.int32, device=warmup_sharding,
            )
            resets = jnp.asarray(
                stream_warmup_reset_mask[:stream_warmup_segments],
                dtype=jnp.bool_, device=warmup_reset_sharding,
            )
        state = stream_warmup_step(model, batch[..., :-1], resets, state)
        jax.block_until_ready(state[0])
        return state

    step = resume_from_step
    print("Starting training (the first step will take some time for compilation...)\n")

    training_complete = False
    train_start_time = time.time()
    pending_shards = []
    if resumed_active_shard is not None:
        pending_shards.append(
            (resumed_active_shard, int(stream_ckpt_state["active_batch_iter"]))
        )

    def iterate_train_shards():
        for item in pending_shards:
            yield item
        for next_shard in train_iter:
            yield next_shard, 0

    # Training loop with explicit counter
    for shard, initial_batch_iter in iterate_train_shards():
        if step >= total_train_steps or training_complete:
            mngr.wait_until_finished()
            print("Finished checkpointing! Cleaned.")
            break

        tokens = shard["tokens"]
        size = shard["size"]
        shard_name = Path(shard["path"]).name
        shard_index = train_file_to_index[str(Path(shard["path"]))]

        try:
            batch_sampler = make_window_sampler(
                tokens,
                size=size,
            )
            shard_processed_fully = False

            # build the static index once per shard (on-demand)
            num_batches_in_shard = batch_sampler.build(logical_bsz, seqlen)
            if not 0 <= initial_batch_iter <= num_batches_in_shard:
                raise ValueError("Checkpoint stream cursor is outside the current shard.")
            if initial_batch_iter > 0:
                batch_sampler.batch_iter = initial_batch_iter
            if initial_batch_iter == 0:
                with mesh_context(cfg.mesh):
                    train_recurrent_state = new_recurrent_state()
            print(f"\n=== Processing Shard: {num_shards_used} with name: {shard_name}", end=" | ")  # fmt: off
            print(f"Indexed {num_batches_in_shard} batches ===")

            if initial_batch_iter == 0 and stream_warmup_segments > 0:
                try:
                    train_recurrent_state = warmup_state(batch_sampler, tokens, train_recurrent_state)
                    skipped_tokens = stream_warmup_segments * logical_bsz * seqlen
                    print(
                        "Stream warmup skipped "
                        f"{stream_warmup_segments} segment(s), "
                        f"{skipped_tokens:,} token(s); "
                        "no optimizer update or train-step increment."
                    )
                except StopIteration:
                    shard_processed_fully = True
                    with mesh_context(cfg.mesh):
                        train_recurrent_state = new_recurrent_state()
                    print(
                        "Shard exhausted during stream warmup; "
                        "no optimizer update or train-step increment."
                    )

            while not shard_processed_fully:
                try:
                    start = time.time()
                    grad_accum_reset_mask[:] = get_next_batch(batch_sampler, tokens, grad_accum_batch)
                    with mesh_context(cfg.mesh):
                        stacked_batch = jnp.asarray(
                            grad_accum_batch, dtype=jnp.int32, device=data_accum_sharding
                        )
                        stacked_x = stacked_batch[:, :, :-1]
                        stacked_y = stacked_batch[:, :, 1:]
                        stacked_reset_mask = jnp.asarray(
                            grad_accum_reset_mask,
                            dtype=jnp.bool_,
                            device=reset_accum_sharding,
                        )
                    (
                        model,
                        loss,
                        train_loss_terms,
                        optim_state,
                        train_recurrent_state,
                    ) = train_step_streaming_accum(
                        model,
                        stacked_x,
                        stacked_y,
                        stacked_reset_mask,
                        train_recurrent_state,
                        optim_state,
                        optim,
                        grad_accum_steps,
                        cfg.hparams.pre_output_reg_cost,
                    )
                    train_ce_loss = float(jax.device_get(train_loss_terms[0]))
                    train_pre_output_reg = float(jax.device_get(train_loss_terms[1]))

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
                        log_payload = {
                            "train/loss": float(loss),
                            "train/ce_loss": train_ce_loss,
                            "train/pre_output_reg": train_pre_output_reg,
                            "train/step_time_sec": dt,
                            "train/train_time_min": train_time_elapsed,
                            "train/tokens_processed": tokens_processed,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/total_tokens_consumed": total_tokens_consumed,
                            "data/shards_used": num_shards_used,
                        }
                        wandb_run.log(log_payload, step=current_step)

                    step += 1

                    if (step % options.save_interval_steps) == 0:
                        with mesh_context(cfg.mesh):
                            stream_ckpt_state = make_stream_checkpoint_state(
                                shard_index,
                                batch_sampler.batch_iter,
                                cfg.mesh,
                                accumulation_layout=accumulation_layout,
                            )
                            ckpt_items = {
                                "params": model,
                                "optim_state": optim_state,
                                "stream_state": stream_ckpt_state,
                                "recurrent_state": train_recurrent_state,
                            }
                            zero_leaf_report = {
                                name: _zero_size_leaf_paths(tree)
                                for name, tree in ckpt_items.items()
                            }
                            zero_leaf_report = {
                                name: paths
                                for name, paths in zero_leaf_report.items()
                                if paths
                            }
                            if zero_leaf_report:
                                raise ValueError(
                                    "Zero-sized arrays detected before checkpoint save: "
                                    f"{zero_leaf_report}"
                                )
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
                                    stream_state=ocp.args.PyTreeSave(
                                        save_items["stream_state"]
                                    ),
                                    recurrent_state=ocp.args.PyTreeSave(
                                        save_items["recurrent_state"]
                                    ),
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
                    with mesh_context(cfg.mesh):
                        train_recurrent_state = new_recurrent_state()
                    print("Shard exhausted")
                    print(f"Total shards consumed: {num_shards_used:<5}")
                    print(f"Total Tokens consumed: {total_tokens_consumed:>9,}")
                    print("-" * 75)

                    print("\nScoring model performance on validation data...\n")
                    val_loss = 0.0
                    val_ce_loss = 0.0
                    val_pre_output_reg = 0.0
                    val_steps_count = 0
                    val_iter = iter(val_dl)
                    for val_shard in val_iter:
                        val_tokens = val_shard["tokens"]
                        try:
                            val_batch_sampler = make_window_sampler(
                                val_tokens,
                                size=val_shard["size"],
                            )

                            num_val_batches = val_batch_sampler.build(logical_bsz, seqlen)
                            if num_val_batches <= 0:
                                continue

                            with mesh_context(cfg.mesh):
                                val_recurrent_state = new_recurrent_state()
                            remaining_val_batches = num_val_batches
                            if stream_warmup_segments > 0:
                                try:
                                    val_recurrent_state = warmup_state(
                                        val_batch_sampler, val_tokens, val_recurrent_state,
                                    )
                                    remaining_val_batches -= stream_warmup_segments
                                except StopIteration:
                                    continue
                            if remaining_val_batches <= 0:
                                continue

                            for _ in range(remaining_val_batches):
                                slot_reset_mask = get_next_batch(val_batch_sampler, val_tokens, val_data_buf)

                                with mesh_context(cfg.mesh):
                                    curr_val_data = jnp.asarray(
                                        val_data_buf, dtype=jnp.int32, device=data_accum_sharding
                                    )
                                    x = curr_val_data[..., :-1]
                                    y = curr_val_data[..., 1:]
                                    val_reset_mask = jnp.asarray(
                                        slot_reset_mask,
                                        dtype=jnp.bool_,
                                        device=reset_accum_sharding,
                                    )
                                    (
                                        loss,
                                        val_recurrent_state,
                                        val_loss_terms,
                                    ) = val_step_streaming(
                                        model,
                                        x,
                                        y,
                                        val_recurrent_state,
                                        val_reset_mask,
                                        cfg.hparams.pre_output_reg_cost,
                                    )
                                val_loss += loss.item()
                                val_ce_loss += float(val_loss_terms[0])
                                val_pre_output_reg += float(val_loss_terms[1])
                                val_steps_count += 1
                        finally:
                            val_tokens.unlink_on_del()
                    if val_steps_count == 0:
                        raise RuntimeError(
                            "No validation batches remained after stream warmup; "
                            "reduce memory_len or use larger validation shards."
                        )
                    avg_val_loss = val_loss / val_steps_count
                    avg_val_ce_loss = val_ce_loss / val_steps_count
                    avg_val_pre_output_reg = val_pre_output_reg / val_steps_count
                    avg_val_loss = jax.block_until_ready(avg_val_loss)
                    improved = avg_val_loss < best_loss
                    if improved:
                        best_loss = avg_val_loss
                        best_step = step

                    print(f"last_val_loss : {last_val_loss:.4f}")
                    print(f"curr_val_loss : {avg_val_loss:.4f}")
                    print(f"Best loss     : {best_loss:.4f} at step {best_step}\n")
                    if wandb_run is not None:
                        log_payload = {
                            "val/loss": avg_val_loss,
                            "val/ce_loss": avg_val_ce_loss,
                            "val/pre_output_reg": avg_val_pre_output_reg,
                            "val/last_loss": last_val_loss,
                            "val/best_loss": best_loss,
                            "val/best_step": best_step,
                            "val/improved": float(improved),
                            "data/shards_used": num_shards_used,
                            "train/total_tokens_consumed": total_tokens_consumed,
                        }
                        wandb_run.log(log_payload, step=step)
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
