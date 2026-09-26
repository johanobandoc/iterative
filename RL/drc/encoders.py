from typing import Any

import jax.numpy as jnp
import jax.nn.initializers as init
from flax import linen as nn
import numpy as np


class ObservationEmbEncoderRMS(nn.Module):
    """Encode discrete grid observations into expert_hidden_dim feature maps."""
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
        x_enc = nn.RMSNorm(reduction_axes=(-3, -2, -1), epsilon=1e-5, use_scale=False)(x_enc)
        return x_enc


ENCODER_REGISTRY = {
    "embedding_rms": ObservationEmbEncoderRMS,
}


def build_observation_encoder(args: Any) -> nn.Module:
    encoder_cls = ENCODER_REGISTRY[args.obs_encoder]
    encoder_hidden_dim = args.encoder_hidden_dim or args.expert_hidden_dim
    return encoder_cls(obs_vocab_size=args.obs_vocab_size, expert_hidden_dim=encoder_hidden_dim)
