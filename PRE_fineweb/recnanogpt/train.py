import argparse
import dataclasses
import importlib
import importlib.util
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


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


MODEL_BACKENDS = {
    "rec_shared_kv": "model_shared_kv",
}
MODEL_BACKEND_CHOICES = tuple(MODEL_BACKENDS)


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


def _promote_inexact_leaves_to_f32(tree):
    return jax.tree.map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact)
        else x,
        tree,
    )


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

def load_model_backend(architecture: str):
    try:
        module_name = MODEL_BACKENDS[architecture]
    except KeyError:
        raise ValueError(
            f"Unsupported recurrent architecture {architecture!r}; "
            f"expected one of {MODEL_BACKEND_CHOICES}."
        ) from None
    train_dir = Path(__file__).resolve().parent
    module_path = train_dir / f"{module_name}.py"
    if not module_path.exists():
        return importlib.import_module(module_name)
    spec = importlib.util.spec_from_file_location(
        f"_recnanogpt_local_{module_name}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load backend module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_train_step_streaming_accum(forward_loss_with_state_and_stats):
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
        def body(carry, batch_inputs):
            (
                current_params,
                current_optim_state,
                current_state,
                loss_sum,
                loss_terms_sum,
                layer_pairwise_matrix_sum,
                layer_pairwise_count,
            ) = carry
            xb, yb, reset_mask = batch_inputs

            def loss_fn(local_params):
                (
                    loss,
                    next_state,
                    res_stats,
                    loss_terms,
                    layer_pairwise_stats,
                ) = forward_loss_with_state_and_stats(
                    local_params,
                    xb,
                    yb,
                    current_state,
                    slot_reset_mask=reset_mask,
                    loss_mask=None,
                    pre_output_reg_cost=pre_output_reg_cost,
                )
                return loss, (next_state, res_stats, loss_terms, layer_pairwise_stats)

            (
                loss,
                (next_state, res_stats, loss_terms, layer_pairwise_stats),
            ), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params)
            token_pairwise_matrix_sum, token_pairwise_count = layer_pairwise_stats
            updates, current_optim_state = optim.update(
                grads,
                current_optim_state,
                current_params,
            )
            current_params = optax.apply_updates(current_params, updates)
            current_state = jax.tree.map(jax.lax.stop_gradient, next_state)
            return (
                current_params,
                current_optim_state,
                current_state,
                loss_sum + loss,
                loss_terms_sum + loss_terms,
                layer_pairwise_matrix_sum + token_pairwise_matrix_sum,
                layer_pairwise_count + token_pairwise_count,
            ), res_stats

        carry0 = (
            params,
            optim_state,
            recurrent_state,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((params.num_experts, params.num_experts), dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        )
        (
            params,
            optim_state,
            recurrent_state,
            loss_sum,
            loss_terms_sum,
            layer_pairwise_matrix_sum,
            layer_pairwise_count,
        ), res_stats_seq = jax.lax.scan(
            body,
            carry0,
            (x_batch, y_batch, slot_reset_mask_batch),
            length=grad_accum_steps,
        )
        return (
            params,
            loss_sum / grad_accum_steps,
            loss_terms_sum / grad_accum_steps,
            optim_state,
            recurrent_state,
            res_stats_seq,
            layer_pairwise_matrix_sum,
            layer_pairwise_count,
        )

    return train_step_streaming_accum


def make_val_step_streaming(forward_loss_with_state_and_stats):
    @jax.jit
    def val_step_streaming(
        params,
        x_batch,
        y_batch,
        recurrent_state,
        slot_reset_mask,
        pre_output_reg_cost,
    ):
        return forward_loss_with_state_and_stats(
            params,
            x_batch,
            y_batch,
            recurrent_state,
            slot_reset_mask=slot_reset_mask,
            loss_mask=None,
            pre_output_reg_cost=pre_output_reg_cost,
        )

    return val_step_streaming


def make_stream_warmup_step(forward_with_state):
    @partial(jax.jit, donate_argnums=(3,))
    def stream_warmup_step(
        params,
        x_batch,
        slot_reset_mask_batch,
        recurrent_state,
    ):
        def body(current_state, batch_inputs):
            xb, reset_mask = batch_inputs
            _, next_state = forward_with_state(
                params,
                xb,
                current_state,
                slot_reset_mask=reset_mask,
            )
            return jax.tree.map(jax.lax.stop_gradient, next_state), None

        recurrent_state, _ = jax.lax.scan(
            body,
            recurrent_state,
            (x_batch, slot_reset_mask_batch),
        )
        return recurrent_state

    return stream_warmup_step


def line(label, value, comma=False, label_w=30, colon_w=2, value_w=20):
    fmt = f">{value_w}," if comma else f">{value_w}"
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


def make_stream_checkpoint_state(active_shard_index, active_batch_iter, mesh):
    with mesh_context(mesh):
        return {
            "active_shard_index": jnp.array(active_shard_index, dtype=jnp.int32),
            "active_batch_iter": jnp.array(active_batch_iter, dtype=jnp.int32),
        }


def should_log_debug_images(step: int) -> bool:
    if step == 0:
        return True
    if step >= 2 and (step & (step - 1)) == 0:
        return True
    return step % 100 == 0


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
    ax.set_xlabel("stream_token_idx")
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

    ax.set_xlabel("layer_j", fontsize=8)
    ax.set_ylabel("layer_i", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="both", labelsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    img = img[:, :, :3].copy()
    plt.close(fig)
    return img


def wandb_image_from_array(image: np.ndarray, caption: str, name: str):
    """Materialize W&B media files under scratch-backed TMPDIR."""
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import wandb

    tmp_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "wandb_images"
    tmp_root.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=f"{name}_", suffix=".png", dir=tmp_root)
    os.close(fd)
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    mpimg.imsave(path, image)
    return wandb.Image(path, caption=caption)


def pairwise_cosine_mean_from_matrix(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return 0.0
    tri_i, tri_j = np.triu_indices(matrix.shape[0], k=1)
    if tri_i.size == 0:
        return 0.0
    return float(np.mean(matrix[tri_i, tri_j]))


def stream_token_positions(starts, stream_starts):
    starts = np.asarray(starts, dtype=np.int32)
    stream_starts = np.asarray(stream_starts, dtype=np.int32)
    return starts - stream_starts


def update_res_stream_ema(
    res_ema,
    res_stats_seq,
    batch_start_positions,
    seqlen,
    ema_decay,
):
    token_offsets = np.arange(seqlen, dtype=np.int32)
    res_stats_seq = np.asarray(res_stats_seq, dtype=np.float32)
    batch_start_positions = np.asarray(batch_start_positions, dtype=np.int32)
    max_tokens = res_ema.shape[1]

    if res_stats_seq.ndim == 3:
        res_stats_seq = res_stats_seq[None, ...]
        batch_start_positions = batch_start_positions[None, ...]

    for micro_idx in range(res_stats_seq.shape[0]):
        token_positions = batch_start_positions[micro_idx][:, None] + token_offsets[None, :]
        for token_idx in range(seqlen):
            step_idx = token_positions[:, token_idx]
            in_range = step_idx < max_tokens
            if not np.any(in_range):
                continue
            idx = step_idx[in_range].astype(np.int64)
            counts = np.bincount(idx, minlength=max_tokens)
            has = counts > 0
            if not np.any(has):
                continue
            step_stats = res_stats_seq[micro_idx, token_idx, in_range]
            for metric_idx in range(2):
                vals = step_stats[:, metric_idx]
                sums = np.bincount(idx, weights=vals, minlength=max_tokens)
                means = sums[has] / counts[has]
                res_ema[metric_idx, has] = (
                    ema_decay * res_ema[metric_idx, has]
                    + (1.0 - ema_decay) * means
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
        "--ticks",
        type=int,
        default=None,
        help="Override the number of recurrent steps per emitted token from config.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="For the depth-stacked shared-K/V backend, run this many independent shared-K/V cores in sequence on each recurrent sweep.",
    )
    parser.add_argument(
        "--recurrent_architecture",
        type=str,
        choices=MODEL_BACKEND_CHOICES,
        default=None,
        help="Select the recurrent backend implementation for the MoE model.",
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=None,
        help="Override grouped residual-stream aggregation width.",
    )
    parser.add_argument(
        "--checkpoint_token_step",
        action="store_true",
        help="Enable activation checkpointing for the recurrent per-token step.",
    )
    parser.add_argument(
        "--detach_kv_cache_state",
        action="store_true",
        help="Stop gradients through the rolling recurrent KV-cache state and K/V memory attention sequence.",
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
        "--attn_output_gate",
        action="store_true",
        help="Enable a sigmoid gate on the per-head attention output before the output projection.",
    )
    parser.add_argument(
        "--direct_qkv_from_input",
        action="store_true",
        help="Project q, k, and v directly from the concatenated recurrent input instead of the post-in-proj hidden state.",
    )
    parser.add_argument(
        "--qkv_input_gelu_proj",
        action="store_true",
        help="Apply a learned GELU projection on the attention source tensor before the q/k/v projections.",
    )
    parser.add_argument(
        "--soft_routed_experts_aggregation",
        action="store_true",
        help="Use routed expert gates as normalized weights in the shared residual-stream aggregation.",
    )
    parser.add_argument(
        "--softmax_expert_routing",
        action="store_true",
        help="Use softmax-normalized expert gate logits across experts when routed aggregation is enabled.",
    )
    parser.add_argument(
        "--experts_aggregation_regime",
        type=str,
        choices=("mean", "sum_div_sqrt_group_size"),
        default=None,
        help="Choose how per-expert outputs are merged into the shared residual stream.",
    )
    parser.add_argument(
        "--memory_aggregation_regime",
        type=str,
        choices=("mean", "sum_div_sqrt_num_experts"),
        default=None,
        help="Choose how per-expert K/V candidates are merged into the one shared K/V cache entry.",
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
        "--muon_gates",
        action="store_true",
        help="Keep gate parameters on Muon instead of routing them to AdamW.",
    )
    parser.add_argument(
        "--pre_output_reg_cost",
        type=float,
        default=None,
        help="L2 penalty coefficient on the final residual stream before the LM head.",
    )
    parser.add_argument(
        "--reset_kv_cache_after_step",
        action="store_true",
        help="Reset all KV caches and cache positions after each completed training step.",
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
    if cli_args.ticks is not None:
        model_updates["ticks"] = cli_args.ticks
    if cli_args.depth is not None:
        model_updates["depth"] = cli_args.depth
    if cli_args.recurrent_architecture is not None:
        model_updates["recurrent_architecture"] = (
            cli_args.recurrent_architecture
        )
    if cli_args.num_groups is not None:
        model_updates["num_groups"] = cli_args.num_groups
    if cli_args.checkpoint_token_step:
        model_updates["checkpoint_token_step"] = True
    if cli_args.detach_kv_cache_state:
        model_updates["detach_kv_cache_state"] = True
    if cli_args.segment_local_kv_cache:
        model_updates["segment_local_kv_cache"] = True
    if cli_args.attn_output_gate:
        model_updates["attn_output_gate"] = True
    if cli_args.direct_qkv_from_input:
        model_updates["direct_qkv_from_input"] = True
    if cli_args.qkv_input_gelu_proj:
        model_updates["qkv_input_gelu_proj"] = True
    if cli_args.soft_routed_experts_aggregation:
        model_updates["soft_routed_experts_aggregation"] = True
    if cli_args.softmax_expert_routing:
        model_updates["softmax_expert_routing"] = True
    if cli_args.experts_aggregation_regime is not None:
        model_updates["experts_aggregation_regime"] = (
            cli_args.experts_aggregation_regime
        )
    if cli_args.memory_aggregation_regime is not None:
        model_updates["memory_aggregation_regime"] = (
            cli_args.memory_aggregation_regime
        )
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
    if cli_args.reset_kv_cache_after_step:
        cfg.hparams.reset_kv_cache_after_step = True
    model_backend = load_model_backend(cfg.model.recurrent_architecture)
    count_params = model_backend.count_params
    GPT = model_backend.GPT
    forward_with_state = model_backend.forward_with_state
    forward_loss_with_state_and_stats = model_backend.forward_loss_with_state_and_stats
    init_recurrent_state = model_backend.init_recurrent_state
    reset_kv_cache_state = model_backend.reset_kv_cache_state
    canonicalize_recurrent_state_for_checkpoint = getattr(
        model_backend,
        "canonicalize_recurrent_state_for_checkpoint",
        lambda _params, state: state,
    )
    validate_recurrent_model_config = model_backend.validate_recurrent_model_config
    train_step_streaming_accum = make_train_step_streaming_accum(
        forward_loss_with_state_and_stats
    )
    val_step_streaming = make_val_step_streaming(forward_loss_with_state_and_stats)
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
    data_sharding = logical_to_sharding(("batch",), cfg.mesh, cfg.rules)
    data_accum_sharding = logical_to_sharding(
        (None, "batch", None), cfg.mesh, cfg.rules
    )
    reset_sharding = logical_to_sharding(("batch",), cfg.mesh, cfg.rules)
    reset_accum_sharding = logical_to_sharding((None, "batch"), cfg.mesh, cfg.rules)

    other_peak_lr = cfg.hparams.other_peak_lr
    other_min_lr = 0.01 * other_peak_lr
    adamw_gates = not cli_args.muon_gates
    warmup_steps = cfg.hparams.warmup_steps
    desired_batch_size = cfg.hparams.desired_batch_size
    grad_accum_steps = resolve_grad_accum_steps(desired_batch_size, bsz, seqlen)
    stream_warmup_segments = (
        (cfg.model.memory_len + seqlen - 1) // seqlen
        if cfg.model.memory_len > 0
        else 0
    )
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
            adamw_gates=adamw_gates,
        ),
    )
    if grad_accum_steps > 1:
        print(
            "Using `MultiSteps` in optax for gradient accumulation with detached streamed state..."
        )
        optim = optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)

    optim_state = optim.init(model)
    if grad_accum_steps > 1:
        optim_state = _promote_inexact_leaves_to_f32(optim_state)
    with mesh_context(cfg.mesh):
        train_recurrent_state = init_recurrent_state(model, bsz)
    stream_ckpt_state = make_stream_checkpoint_state(-1, 0, cfg.mesh)

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
            "Recurrent architecture",
            cfg.model.recurrent_architecture,
        )
    )
    print(
        line(
            "Shared-global depth",
            cfg.model.depth,
        )
    )
    print(line("Attention output gate", cfg.model.attn_output_gate))
    print(line("Direct QKV from input", cfg.model.direct_qkv_from_input))
    print(line("QKV input GELU proj", cfg.model.qkv_input_gelu_proj))
    print(
        line(
            "Soft-routed experts aggregation",
            cfg.model.soft_routed_experts_aggregation,
        )
    )
    print(
        line(
            "Softmax expert routing",
            cfg.model.softmax_expert_routing,
        )
    )
    print(
        line(
            "Memory aggregation regime",
            cfg.model.memory_aggregation_regime,
        )
    )
    print(line("Experts aggregation regime", cfg.model.experts_aggregation_regime))
    print(line("Number of groups", cfg.model.num_groups))
    print(line("Checkpoint save steps", checkpoint_save_steps))
    print(line("Max checkpoints to keep", max_checkpoints_to_keep))
    print(line("Per device batch size", per_device_bsz))
    print(line("Total batch size", bsz))
    print(line("Grad accumulation steps", grad_accum_steps))
    print(line("Stream warmup segments", stream_warmup_segments))
    print(line("Detach KV cache state", cfg.model.detach_kv_cache_state))
    print(line("Segment-local KV cache", cfg.model.segment_local_kv_cache))
    print()
    print(line("Other LR (min, peak)", str((other_min_lr, other_peak_lr))))
    print(line("Warmup steps", cfg.hparams.warmup_steps))
    print(line("Gates use AdamW", adamw_gates))
    print(line("Cautious weight decay", cfg.hparams.cautious_weight_decay))
    print(line("Pre-output reg cost", cfg.hparams.pre_output_reg_cost))
    print(line("Reset KV cache after step", cfg.hparams.reset_kv_cache_after_step))
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
                "recurrent_architecture": cfg.model.recurrent_architecture,
                "depth": cfg.model.depth,
                "train_files": num_train_files,
                "val_files": num_val_files,
                "dataloader_mode": dataloader_mode,
                "attn_output_gate": cfg.model.attn_output_gate,
                "direct_qkv_from_input": cfg.model.direct_qkv_from_input,
                "qkv_input_gelu_proj": cfg.model.qkv_input_gelu_proj,
                "soft_routed_experts_aggregation": cfg.model.soft_routed_experts_aggregation,
                "softmax_expert_routing": cfg.model.softmax_expert_routing,
                "experts_aggregation_regime": cfg.model.experts_aggregation_regime,
                "memory_aggregation_regime": cfg.model.memory_aggregation_regime,
                "num_groups": cfg.model.num_groups,
                "detach_kv_cache_state": cfg.model.detach_kv_cache_state,
                "segment_local_kv_cache": cfg.model.segment_local_kv_cache,
                "stream_warmup_segments": stream_warmup_segments,
                "reset_kv_cache_after_step": cfg.hparams.reset_kv_cache_after_step,
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
    tokens_per_train_step = bsz * seqlen * grad_accum_steps
    total_tokens_consumed = int(resume_from_step) * tokens_per_train_step
    res_stream_ema = None
    if wandb_run is not None:
        res_stream_ema = np.zeros(
            (2, cfg.tracking.res_stream_stats_tokens), dtype=np.float32
        )

    # Reusable data buffers
    grad_accum_batch = np.zeros((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)
    grad_accum_reset_mask = np.ones((grad_accum_steps, bsz), dtype=np.bool_)
    grad_accum_stream_positions = np.zeros((grad_accum_steps, bsz), dtype=np.int32)
    stream_warmup_batch = np.zeros(
        (max(1, stream_warmup_segments), bsz, seqlen + 1),
        dtype=np.uint16,
    )
    stream_warmup_reset_mask = np.ones(
        (max(1, stream_warmup_segments), bsz),
        dtype=np.bool_,
    )
    val_data_buf = np.zeros((bsz, seqlen + 1), dtype=np.uint16)

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
            num_batches_in_shard = batch_sampler.build(bsz, seqlen)
            slot_stream_starts = batch_sampler.slot_stream_starts
            if initial_batch_iter > 0:
                batch_sampler.batch_iter = initial_batch_iter
            if initial_batch_iter == 0:
                with mesh_context(cfg.mesh):
                    train_recurrent_state = init_recurrent_state(model, bsz)
            print(f"\n=== Processing Shard: {num_shards_used} with name: {shard_name}", end=" | ")  # fmt: off
            print(f"Indexed {num_batches_in_shard} batches ===")

            if initial_batch_iter == 0 and stream_warmup_segments > 0:
                try:
                    for warmup_idx in range(stream_warmup_segments):
                        starts, ends, slot_reset_mask = (
                            batch_sampler.next_batch_with_reset(bsz, seqlen)
                        )
                        get_next_batch(
                            starts,
                            ends,
                            bsz,
                            seqlen,
                            tokens,
                            data_accum_sharding,
                            stream_warmup_batch[warmup_idx],
                            transfer_to_device=False,
                        )
                        stream_warmup_reset_mask[warmup_idx] = slot_reset_mask

                    with mesh_context(cfg.mesh):
                        warmup_batch = jnp.asarray(
                            stream_warmup_batch[:stream_warmup_segments],
                            dtype=jnp.int32,
                            device=data_accum_sharding,
                        )
                        warmup_x = warmup_batch[:, :, :-1]
                        warmup_reset_mask = jnp.asarray(
                            stream_warmup_reset_mask[:stream_warmup_segments],
                            dtype=jnp.bool_,
                            device=reset_accum_sharding,
                        )
                    train_recurrent_state = stream_warmup_step(
                        model,
                        warmup_x,
                        warmup_reset_mask,
                        train_recurrent_state,
                    )
                    jax.block_until_ready(train_recurrent_state[0])
                    skipped_tokens = stream_warmup_segments * bsz * seqlen
                    print(
                        "Stream warmup skipped "
                        f"{stream_warmup_segments} segment(s), "
                        f"{skipped_tokens:,} token(s); "
                        "no optimizer update or train-step increment."
                    )
                except StopIteration:
                    shard_processed_fully = True
                    with mesh_context(cfg.mesh):
                        train_recurrent_state = init_recurrent_state(model, bsz)
                    print(
                        "Shard exhausted during stream warmup; "
                        "no optimizer update or train-step increment."
                    )

            while not shard_processed_fully:
                try:
                    start = time.time()
                    for micro_step in range(grad_accum_steps):
                        starts, ends, slot_reset_mask = (
                            batch_sampler.next_batch_with_reset(bsz, seqlen)
                        )
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
                        grad_accum_reset_mask[micro_step] = slot_reset_mask
                        grad_accum_stream_positions[micro_step] = stream_token_positions(
                            starts,
                            slot_stream_starts,
                        )
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
                        train_res_stats,
                        train_layer_pairwise_matrix_sum,
                        train_layer_pairwise_count,
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
                    if cfg.hparams.reset_kv_cache_after_step:
                        train_recurrent_state = reset_kv_cache_state(
                            model,
                            train_recurrent_state,
                        )
                    train_ce_loss = float(jax.device_get(train_loss_terms[0]))
                    train_pre_output_reg = float(jax.device_get(train_loss_terms[1]))
                    if res_stream_ema is not None:
                        update_res_stream_ema(
                            res_stream_ema,
                            jax.device_get(train_res_stats),
                            grad_accum_stream_positions,
                            seqlen,
                            cfg.tracking.res_stream_ema_decay,
                        )

                    # Block for accurate timing
                    jax.block_until_ready(loss)
                    end = time.time()
                    dt = end - start
                    train_time_elapsed = (end - train_start_time) / 60  # in minutes
                    tokens_processed = tokens_per_train_step
                    total_tokens_consumed += tokens_processed
                    tokens_per_sec = int(tokens_processed / dt)
                    train_layer_output_pairwise_cosine = None
                    train_layer_output_pairwise_matrix = None
                    train_layer_pairwise_count = float(
                        jax.device_get(train_layer_pairwise_count)
                    )
                    if train_layer_pairwise_count > 0.0:
                        train_layer_output_pairwise_matrix = np.asarray(
                            jax.device_get(train_layer_pairwise_matrix_sum),
                            dtype=np.float32,
                        ) / train_layer_pairwise_count
                        train_layer_output_pairwise_cosine = (
                            pairwise_cosine_mean_from_matrix(
                                train_layer_output_pairwise_matrix
                            )
                        )

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
                        if train_layer_output_pairwise_cosine is not None:
                            log_payload["debug/layer_output_pairwise_cosine"] = (
                                train_layer_output_pairwise_cosine
                            )
                        if (
                            res_stream_ema is not None
                            and should_log_debug_images(current_step)
                        ):
                            max_tokens = cfg.tracking.res_stream_stats_tokens
                            train_prev_pairwise_matrix = train_layer_output_pairwise_matrix
                            train_prev_pairwise_matrix_count = train_layer_pairwise_count
                            train_prev_heatmap_title = "train layer outputs pairwise cosine"
                            if train_prev_pairwise_matrix_count > 0.0:
                                train_prev_heatmap = wandb_image_from_array(
                                    render_pairwise_cosine_heatmap(
                                        train_prev_pairwise_matrix,
                                        title=train_prev_heatmap_title,
                                    ),
                                    caption=f"step={current_step}",
                                    name="layer_output_pairwise_cosine_heatmap",
                                )
                                log_payload["debug/layer_output_pairwise_cosine_heatmap"] = (
                                    train_prev_heatmap
                                )
                            log_payload["debug/res_stream_rms_curve"] = wandb_image_from_array(
                                render_res_stream_curve(
                                    res_stream_ema[0],
                                    width=max(240, max_tokens * 4),
                                    height=200,
                                    ylabel="res_rms_ema",
                                    title="res stream rms EMA by stream token",
                                ),
                                caption=f"step={current_step}",
                                name="res_stream_rms_curve",
                            )
                            log_payload["debug/res_stream_max_abs_curve"] = wandb_image_from_array(
                                render_res_stream_curve(
                                    res_stream_ema[1],
                                    width=max(240, max_tokens * 4),
                                    height=200,
                                    ylabel="res_max_abs_ema",
                                    title="res stream max_abs EMA by stream token",
                                ),
                                caption=f"step={current_step}",
                                name="res_stream_max_abs_curve",
                            )
                        wandb_run.log(log_payload, step=current_step)

                    step += 1

                    if (step % options.save_interval_steps) == 0:
                        with mesh_context(cfg.mesh):
                            stream_ckpt_state = make_stream_checkpoint_state(
                                shard_index,
                                batch_sampler.batch_iter,
                                cfg.mesh,
                            )
                            save_recurrent_state = (
                                canonicalize_recurrent_state_for_checkpoint(
                                    model,
                                    train_recurrent_state,
                                )
                            )
                            ckpt_items = {
                                "params": model,
                                "optim_state": optim_state,
                                "stream_state": stream_ckpt_state,
                                "recurrent_state": save_recurrent_state,
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
                    # Once we have trained on one shard, let's validate the performance as well
                    shard_processed_fully = True
                    num_shards_used += 1
                    with mesh_context(cfg.mesh):
                        train_recurrent_state = init_recurrent_state(model, bsz)
                    print("Shard exhausted")
                    print(f"Total shards consumed: {num_shards_used:<5}")
                    print(f"Total Tokens consumed: {total_tokens_consumed:>9,}")
                    print("-" * 75)

                    print("\nScoring model performance on validation data...\n")
                    val_loss = 0.0
                    val_ce_loss = 0.0
                    val_pre_output_reg = 0.0
                    val_steps_count = 0
                    val_layer_output_pairwise_cosine_sum = 0.0
                    val_layer_output_pairwise_count = 0.0
                    val_layer_output_pairwise_matrix_sum = np.zeros(
                        (cfg.model.num_experts, cfg.model.num_experts), dtype=np.float32
                    )
                    val_layer_output_pairwise_matrix_count = 0.0
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

                            with mesh_context(cfg.mesh):
                                val_recurrent_state = init_recurrent_state(model, bsz)
                            remaining_val_batches = num_val_batches
                            if stream_warmup_segments > 0:
                                try:
                                    for warmup_idx in range(stream_warmup_segments):
                                        starts, ends, slot_reset_mask = (
                                            val_batch_sampler.next_batch_with_reset(
                                                bsz, seqlen
                                            )
                                        )
                                        get_next_batch(
                                            starts,
                                            ends,
                                            bsz,
                                            seqlen,
                                            val_tokens,
                                            data_accum_sharding,
                                            stream_warmup_batch[warmup_idx],
                                            transfer_to_device=False,
                                        )
                                        stream_warmup_reset_mask[warmup_idx] = (
                                            slot_reset_mask
                                        )
                                    with mesh_context(cfg.mesh):
                                        warmup_batch = jnp.asarray(
                                            stream_warmup_batch[
                                                :stream_warmup_segments
                                            ],
                                            dtype=jnp.int32,
                                            device=data_accum_sharding,
                                        )
                                        warmup_x = warmup_batch[:, :, :-1]
                                        warmup_reset_mask = jnp.asarray(
                                            stream_warmup_reset_mask[
                                                :stream_warmup_segments
                                            ],
                                            dtype=jnp.bool_,
                                            device=reset_accum_sharding,
                                        )
                                    val_recurrent_state = stream_warmup_step(
                                        model,
                                        warmup_x,
                                        warmup_reset_mask,
                                        val_recurrent_state,
                                    )
                                    jax.block_until_ready(val_recurrent_state[0])
                                    remaining_val_batches -= stream_warmup_segments
                                except StopIteration:
                                    continue
                            if remaining_val_batches <= 0:
                                continue

                            for _ in range(remaining_val_batches):
                                starts, ends, slot_reset_mask = (
                                    val_batch_sampler.next_batch_with_reset(
                                        bsz, seqlen
                                    )
                                )
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
                                    val_reset_mask = jnp.asarray(
                                        slot_reset_mask,
                                        dtype=jnp.bool_,
                                        device=reset_sharding,
                                    )
                                    (
                                        loss,
                                        val_recurrent_state,
                                        _,
                                        val_loss_terms,
                                        val_layer_pairwise_stats,
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
                                layer_pairwise_matrix_sum, layer_pairwise_count = (
                                    val_layer_pairwise_stats
                                )
                                layer_pairwise_count = float(
                                    jax.device_get(layer_pairwise_count)
                                )
                                if layer_pairwise_count > 0.0:
                                    layer_pairwise_matrix = np.asarray(
                                        jax.device_get(layer_pairwise_matrix_sum),
                                        dtype=np.float32,
                                    ) / layer_pairwise_count
                                    val_layer_output_pairwise_cosine_sum += (
                                        pairwise_cosine_mean_from_matrix(
                                            layer_pairwise_matrix
                                        )
                                    )
                                    val_layer_output_pairwise_count += 1.0
                                    val_layer_output_pairwise_matrix_sum += np.asarray(
                                        jax.device_get(layer_pairwise_matrix_sum),
                                        dtype=np.float32,
                                    )
                                    val_layer_output_pairwise_matrix_count += (
                                        layer_pairwise_count
                                    )
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
                    avg_val_layer_output_pairwise_cosine = None
                    val_layer_output_pairwise_matrix = None
                    if val_layer_output_pairwise_count > 0.0:
                        avg_val_layer_output_pairwise_cosine = (
                            val_layer_output_pairwise_cosine_sum
                            / val_layer_output_pairwise_count
                        )
                        val_layer_output_pairwise_matrix = (
                            val_layer_output_pairwise_matrix_sum
                            / val_layer_output_pairwise_matrix_count
                        )
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
                        log_payload = {
                            "val/loss": avg_val_loss,
                            "val/ce_loss": avg_val_ce_loss,
                            "val/pre_output_reg": avg_val_pre_output_reg,
                            "val/last_loss": last_val_loss,
                            "val/best_loss": best_loss,
                            "val/best_step": best_step,
                            "val/improved": float(improved),
                            "train/es_patience_counter": es_patience_counter,
                            "data/shards_used": num_shards_used,
                            "train/total_tokens_consumed": total_tokens_consumed,
                        }
                        if avg_val_layer_output_pairwise_cosine is not None:
                            log_payload["debug/val_layer_output_pairwise_cosine"] = (
                                avg_val_layer_output_pairwise_cosine
                        )
                        if res_stream_ema is not None:
                            max_tokens = cfg.tracking.res_stream_stats_tokens
                            train_prev_pairwise_matrix = train_layer_output_pairwise_matrix
                            train_prev_pairwise_matrix_count = train_layer_pairwise_count
                            train_prev_heatmap_title = "train layer outputs pairwise cosine"
                            if train_prev_pairwise_matrix_count > 0.0:
                                train_prev_heatmap = wandb_image_from_array(
                                    render_pairwise_cosine_heatmap(
                                        train_prev_pairwise_matrix,
                                        title=train_prev_heatmap_title,
                                    ),
                                    caption=f"step={step}",
                                    name="layer_output_pairwise_cosine_heatmap",
                                )
                                log_payload["debug/layer_output_pairwise_cosine_heatmap"] = (
                                    train_prev_heatmap
                                )
                            if val_layer_output_pairwise_matrix is not None:
                                log_payload["debug/val_layer_output_pairwise_cosine_heatmap"] = (
                                    wandb_image_from_array(
                                        render_pairwise_cosine_heatmap(
                                            val_layer_output_pairwise_matrix,
                                            title="val layer outputs pairwise cosine",
                                        ),
                                        caption=f"step={step}",
                                        name="val_layer_output_pairwise_cosine_heatmap",
                                    )
                                )
                            log_payload["debug/res_stream_rms_curve"] = wandb_image_from_array(
                                render_res_stream_curve(
                                    res_stream_ema[0],
                                    width=max(240, max_tokens * 4),
                                    height=200,
                                    ylabel="res_rms_ema",
                                    title="res stream rms EMA by stream token",
                                ),
                                caption=f"step={step}",
                                name="res_stream_rms_curve",
                            )
                            log_payload["debug/res_stream_max_abs_curve"] = wandb_image_from_array(
                                render_res_stream_curve(
                                    res_stream_ema[1],
                                    width=max(240, max_tokens * 4),
                                    height=200,
                                    ylabel="res_max_abs_ema",
                                    title="res stream max_abs EMA by stream token",
                                ),
                                caption=f"step={step}",
                                name="res_stream_max_abs_curve",
                            )
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
