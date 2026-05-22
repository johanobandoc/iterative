import os
import jax
import jax.numpy as jnp
import dataclasses
from pathlib import Path
from typing import Callable, Tuple, Optional
from jax.sharding import Mesh
from utils import jax_pytree_struct


AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]

# Expected physical mesh axis names:
# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
EXPERT_AXIS_NAME = "z"
TENSOR_ONLY_AXIS_NAME = "y"
ATTN_HEADS_AXIS_NAME = "y"
TENSOR_AXIS_NAME = ("y", "z")


def _resolve_scratch_root() -> Path:
    env_root = os.environ.get("RECNANOGPT_SCRATCH_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    shared_scratch = os.environ.get("SCRATCH")
    if shared_scratch:
        return Path(shared_scratch).expanduser()
    return Path("runs")


def _resolve_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_SCRATCH_ROOT = _resolve_scratch_root()
DEFAULT_FINEWEB_DIR = os.environ.get(
    "RECNANOGPT_DATA_DIR",
    str(
        _resolve_existing_path(
            DEFAULT_SCRATCH_ROOT / "fineweb",
            DEFAULT_SCRATCH_ROOT / "fineweb10B",
        )
    ),
)
DEFAULT_CKPT_DIR = os.environ.get(
    "RECNANOGPT_CKPT_DIR",
    str(
        _resolve_existing_path(
            DEFAULT_SCRATCH_ROOT / "recnanogpt" / "ckpts",
            DEFAULT_SCRATCH_ROOT / "recnanogpt_checkpoints",
        )
    ),
)


def init_uniform(scale=1.0):
    def kernel_init(key, shape, dtype):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)

    return kernel_init


@dataclasses.dataclass
class EmbeddingConfig:
    dtype: jnp.dtype = jnp.bfloat16
    vocab_size: int = 50304
    d_emb: int = 768
    num_layers: int = 12
    weight_initializer: Callable = dataclasses.field(init=False)
    weight_logical_axes: Tuple[str, str] = ("embed_in", "embed_out")

    def __post_init__(self):
        self.weight_initializer = jax.nn.initializers.normal(stddev=1.0)


@dataclasses.dataclass
class MultiHeadAttentionConfig:
    dtype: jnp.dtype = jnp.bfloat16
    d_in: int = 768
    d_out: int = 768
    num_heads: int = 12
    num_layers: int = 12

    wq_initializer: Callable = dataclasses.field(init=False)
    wk_initializer: Callable = dataclasses.field(init=False)
    wv_initializer: Callable = dataclasses.field(init=False)
    wo_initializer: Callable = dataclasses.field(init=False)

    wq_logical_axes: Tuple[str, str] = ("attn_wq_in", "attn_wq_out")
    wk_logical_axes: Tuple[str, str] = ("attn_wk_in", "attn_wk_out")
    wv_logical_axes: Tuple[str, str] = ("attn_wv_in", "attn_wv_out")
    wo_logical_axes: Tuple[str, str] = ("attn_wo_in", "attn_wo_out")

    def __post_init__(self):
        init = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wq_initializer = init
        self.wk_initializer = init
        self.wv_initializer = init
        self.wo_initializer = jax.nn.initializers.zeros


@dataclasses.dataclass
class GroupedQueryAttentionConfig:
    dtype: jnp.dtype = jnp.bfloat16
    d_emb: int = 768
    d_in: int = 768
    q_heads: int = 8
    kv_heads: int = 4
    num_layers: int = 12
    use_output_gate: bool = False
    head_dim: int = dataclasses.field(init=False)

    wq_initializer: Callable = dataclasses.field(init=False)
    wk_initializer: Callable = dataclasses.field(init=False)
    wv_initializer: Callable = dataclasses.field(init=False)
    wg_initializer: Callable = dataclasses.field(init=False)
    wo_initializer: Callable = dataclasses.field(init=False)

    wq_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_q_heads", "attn_head_dim")
    wk_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_kv_heads", "attn_head_dim")
    wv_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_kv_heads", "attn_head_dim")
    wg_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_q_heads", "attn_head_dim")
    wo_logical_axes: Tuple[str, str, str] = ("attn_wo_in", "attn_head_dim", "attn_wo_out")

    def __post_init__(self):
        self.head_dim = self.d_emb // self.q_heads
        self.wq_initializer = init_uniform(scale=3**0.5 * self.d_in**-0.5)
        self.wk_initializer = init_uniform(scale=3**0.5 * self.d_in**-0.5)
        self.wv_initializer = init_uniform(scale=3**0.5 * self.d_in**-0.5)
        self.wg_initializer = jax.nn.initializers.zeros
        self.wo_initializer = jax.nn.initializers.zeros


@dataclasses.dataclass
class LinearConfig:
    dtype: jnp.dtype = jnp.bfloat16
    in_features: int = 768
    out_features: int = 50304
    use_bias: bool = False
    weight_initializer: Callable = None
    weight_logical_axes: Tuple[str, str] = ("linear_in", "linear_out")


@dataclasses.dataclass
class MLPConfig:
    d_emb: int = 768
    dtype: jnp.dtype = jnp.bfloat16
    fc1: LinearConfig = dataclasses.field(init=False)
    fc2: LinearConfig = dataclasses.field(init=False)

    def __post_init__(self):
        self.fc1 = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb,
            out_features=self.d_emb * 4,
            weight_initializer=init_uniform(scale=3**0.5 * self.d_emb**-0.5),
            weight_logical_axes=("mlp_fc1_in", "mlp_fc1_out"),
        )
        self.fc2 = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb * 4,
            out_features=self.d_emb,
            weight_initializer=jax.nn.initializers.zeros,
            weight_logical_axes=("mlp_fc2_in", "mlp_fc2_out"),
        )


@dataclasses.dataclass
class ModelConfig:
    seqlen: int = 256 #16 #16 #16 #64 #256 #2048
    vocab_size: int = 50304
    expert_hidden_dim: int = 768
    num_experts: int = 16
    q_heads: int = 8
    kv_heads: int = 4
    attn_type: str = "gqa"
    attn_output_gate: bool = False
    recurrent_architecture: str = "rec_shared_kv"
    direct_qkv_from_input: bool = False
    qkv_input_gelu_proj: bool = False
    soft_routed_experts_aggregation: bool = False
    softmax_expert_routing: bool = False
    experts_aggregation_regime: str = "sum_div_sqrt_group_size"
    depth: int = 1
    memory_aggregation_regime: str = "mean"
    num_groups: int = 1
    memory_len: int = 16
    ticks: int = 2
    checkpoint_token_step: bool = False
    detach_kv_cache_state: bool = False
    segment_local_kv_cache: bool = False
    num_heads: Optional[Tuple[int, None]] = dataclasses.field(init=False)
    dtype: jnp.dtype = jnp.bfloat16

    embed: EmbeddingConfig = dataclasses.field(init=False)
    mlp: MLPConfig = dataclasses.field(init=False)
    lm_head: LinearConfig = dataclasses.field(init=False)

    if attn_type == "mha":
        attn: MultiHeadAttentionConfig = dataclasses.field(init=False)
    elif attn_type == "gqa":
        attn: GroupedQueryAttentionConfig = dataclasses.field(init=False)
    else:
        raise ValueError(
            f"Only these attention types are supported for now `['gqa', 'mha']`. Received = {attn_type}"
        )

    def __post_init__(self):
        if self.q_heads == self.kv_heads and self.attn_type == "gqa":
            raise Warning(
                "When the number of query heads equals the number of kv heads, JAX computes MHA not GQA!"
            )

        self.embed = EmbeddingConfig(
            dtype=self.dtype,
            vocab_size=self.vocab_size,
            d_emb=self.expert_hidden_dim,
            num_layers=self.num_experts,
        )
        if self.attn_type == "mha":
            self.attn = MultiHeadAttentionConfig(
                dtype=self.dtype,
                d_in=self.expert_hidden_dim,
                d_out=self.expert_hidden_dim,
                num_heads=self.num_heads,
                num_layers=self.num_experts,
            )
        elif self.attn_type == "gqa":
            self.attn = GroupedQueryAttentionConfig(
                dtype=self.dtype,
                d_emb=self.expert_hidden_dim,
                d_in=self.expert_hidden_dim,
                q_heads=self.q_heads,
                kv_heads=self.kv_heads,
                num_layers=self.num_experts,
                use_output_gate=self.attn_output_gate,
            )
        self.mlp = MLPConfig(dtype=self.dtype, d_emb=self.expert_hidden_dim)
        self.lm_head = LinearConfig(
            dtype=self.dtype,
            in_features=self.expert_hidden_dim * self.num_groups,
            out_features=self.vocab_size,
            weight_initializer=jax.nn.initializers.normal(stddev=0.001),
        )


@dataclasses.dataclass
class ShardingRules:
    batch: AxisName = BATCH_AXIS_NAME
    sequence: AxisName = None
    act_embed: AxisName = None
    act_heads: AxisName = None

    embed_in: AxisName = None
    embed_out: AxisName = None

    attn_wqkv_in: AxisName = None
    attn_q_heads: AxisName = None
    attn_kv_heads: AxisName = None
    attn_head_dim: AxisName = None

    attn_wo_in: AxisName = None
    attn_wo_out: AxisName = None

    norm_in: AxisName = None
    norm_out: AxisName = None

    mlp_fc1_in: AxisName = None
    mlp_fc1_out: AxisName = None
    mlp_fc2_in: AxisName = None
    mlp_fc2_out: AxisName = None

    linear_in: AxisName = None
    linear_out: AxisName = None


@dataclasses.dataclass
class CheckpointConfig:
    # Checkpoint related
    max_checkpoints_to_keep: int = 5
    checkpoint_save_steps: int = 100
    last_checkpoint_step: int = 0
    # Directory where checkpoints will be saved
    save_ckpt_dir: Path | str = DEFAULT_CKPT_DIR
    # Path to params subdirectory within a checkpoint from which weights will be loaded
    load_params_ckpt_path: Path | str = ""


@dataclasses.dataclass
class HyperParams:
    # Batch size related
    per_device_batch_size: int = 32 #128 #256 #1024 #1024 #1024 #1024 #256 #512 #128
    desired_batch_size: int = 131072 #262144 #524288 #32768 #4048 #524288 #128
    grad_accum_steps: Optional[float] = dataclasses.field(init=False)

    # Optimizer related
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    other_peak_lr: float = 0.02
    b1: float = 0.8
    b2: float = 0.95
    weight_decay: float = 0.0
    cautious_weight_decay: float = 0.01
    pre_output_reg_cost: float = 1e-4
    grad_clip_norm: float = 1.0
    total_train_steps: int = 10000
    warmup_steps: int = int(min(300, 0.01 * total_train_steps))  # ~10% of total steps

    # For midtraining and SFT
    init_lr_frac: float = 0.2
    final_lr_frac: float = 0.0


    # Other
    reset_kv_cache_after_step: bool = False
    es_patience: int = 500
    val_interval: int = 50


@dataclasses.dataclass
class TrackingConfig:
    track: bool = True
    exp_name: str = ""
    wandb_project_name: str = "jaxMoEnanogpt"
    wandb_entity: str = ""
    wandb_tags: tuple[str, ...] = ()
    res_stream_stats_tokens: int = 100
    res_stream_ema_decay: float = 0.99


@jax_pytree_struct
class Config:
    seed: jax.Array = None
    mesh: Mesh = None
    rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    hparams: HyperParams = dataclasses.field(default_factory=HyperParams)
    ckpt_cfg: CheckpointConfig = dataclasses.field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = dataclasses.field(default_factory=TrackingConfig)
    data_dir: Path | str = DEFAULT_FINEWEB_DIR
