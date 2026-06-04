from typing import Any

import optax
from flax import traverse_util
from flax.core import FrozenDict, freeze


def _freeze_like(reference_tree, tree):
    if isinstance(reference_tree, FrozenDict):
        return freeze(tree)
    return tree


def _path_lower(path) -> tuple[str, ...]:
    return tuple(part.lower() for part in path)


def _is_last_head_param(path) -> bool:
    return (
        len(path) >= 3
        and path[0] == "params"
        and path[1] in ("Dense_0", "Dense_1")
    )


def build_weight_decay_mask(params):
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {}
    for path in flat_params:
        leaf_name = path[-1]
        path_lower = _path_lower(path)
        full_path = "/".join(path).lower()
        is_encoder_param = any("encoder" in part for part in path_lower)
        exclude = (
            leaf_name == "bias"
            or leaf_name == "scale"
            or leaf_name == "embedding"
            or "gate" in full_path
            or "router" in full_path
            or is_encoder_param
            or _is_last_head_param(path)
        )
        flat_mask[path] = not exclude
    return _freeze_like(params, traverse_util.unflatten_dict(flat_mask))


def build_optimizer(args: Any, params):
    if args.anneal_lr:
        schedule = optax.linear_schedule(
            init_value=args.learning_rate,
            end_value=0.0,
            transition_steps=max(1, args.num_iterations),
        )
    else:
        schedule = args.learning_rate

    if args.weight_decay <= 0.0:
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=schedule, eps=1e-6),
        )

    weight_decay_mask = build_weight_decay_mask(params)
    return optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            eps=1e-6,
            weight_decay=args.weight_decay,
            mask=weight_decay_mask,
        ),
    )
