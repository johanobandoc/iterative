from typing import Any

import jax
import jax.numpy as jnp
import jax.nn.initializers as init
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import numpy as np


class ObservationEmbEncoderRMS(nn.Module):
    """Encode discrete grid observations into expert_hidden_dim feature maps."""
    args: Any
    obs_vocab_size: int
    expert_hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_idx = x.astype(jnp.int32)
        features_per_channel = self.expert_hidden_dim // 2
        scale = 1 / np.sqrt(2*features_per_channel)
        x_enc1 = nn.Embed(num_embeddings=self.obs_vocab_size,
                          features=features_per_channel,
                          embedding_init=init.variance_scaling(scale, "fan_in", "normal", out_axis=0)
                          )(x_idx[..., 0])
        x_enc2 = nn.Embed(num_embeddings=self.obs_vocab_size,
                          features=features_per_channel,
                          embedding_init=init.variance_scaling(scale, "fan_in", "normal", out_axis=0)
                          )(x_idx[..., 1])
        x_enc = jnp.concatenate([x_enc1, x_enc2], axis=-1)
        # x_enc = x_enc / np.sqrt(2)
        # x_enc = nn.RMSNorm(epsilon=1e-5)(x_enc)
        x_enc = nn.RMSNorm(reduction_axes=(-3, -2, -1), epsilon=1e-5, use_scale=False)(x_enc)
        return x_enc


class ObservationMLPGLUEncoder(nn.Module):
    """Encode observations with a per-position GLU MLP."""
    args: Any
    obs_vocab_size: int
    expert_hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_enc = x.astype(jnp.float32)
        x_enc = nn.Dense(self.expert_hidden_dim * 2,
                         kernel_init=orthogonal(1),
                         # kernel_init=orthogonal(np.sqrt(2)),
                         bias_init=constant(0.0),
                         )(x_enc)
        a, b = jnp.split(x_enc, 2, axis=-1)
        x_enc = a * jax.nn.sigmoid(b)
        x_enc = nn.RMSNorm(epsilon=1e-5)(x_enc)
        return x_enc


ENCODER_REGISTRY = {
    "embedding_rms": ObservationEmbEncoderRMS,
    "mlp_glu": ObservationMLPGLUEncoder,
}


def build_observation_encoder(args: Any) -> nn.Module:
    encoder_cls = ENCODER_REGISTRY[args.obs_encoder]
    encoder_hidden_dim = args.encoder_hidden_dim if args.encoder_hidden_dim > 0 else args.expert_hidden_dim
    return encoder_cls(args=args, obs_vocab_size=args.obs_vocab_size, expert_hidden_dim=encoder_hidden_dim)
