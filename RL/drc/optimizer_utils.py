from typing import Any

import optax


def build_optimizer(args: Any):
    """Adam with global gradient clipping, as used in the Sokoban sweeps."""
    if args.anneal_lr:
        learning_rate = optax.linear_schedule(
            init_value=args.learning_rate,
            end_value=0.0,
            transition_steps=max(1, args.num_iterations),
        )
    else:
        learning_rate = args.learning_rate

    return optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-6),
    )
