import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def get_sharding_for_checkpoint(x, mesh):
    """Obtain checkpoint-safe shardings for pytree leaves on save and restore."""
    if hasattr(x, "ndim") and x.ndim == 0:
        return NamedSharding(mesh, P())
    if isinstance(x, jax.Array) and hasattr(x, "sharding"):
        from jax.sharding import SingleDeviceSharding

        sharding = x.sharding
        # Ensure small optimizer leaves (e.g., Muon scalars/vectors) are replicated,
        # not left on a single device, to match param shardings during train_step.
        if isinstance(sharding, SingleDeviceSharding):
            return NamedSharding(mesh, P())
        if getattr(sharding, "mesh", None) is None:
            return NamedSharding(mesh, P())
        return sharding
    else:
        return NamedSharding(mesh, P())


def prepare_for_checkpoint_save(tree, mesh):
    """Convert checkpoint arrays to host payloads so Orbax avoids jax.Array save."""
    del mesh
    return jax.tree.map(
        lambda leaf: (
            None
            if leaf is None
            else np.asarray(jax.device_get(leaf))
            if isinstance(leaf, jax.Array)
            else leaf
        ),
        tree,
        is_leaf=lambda x: x is None,
    )


def find_jax_array_leaves(tree):
    paths = []
    path_vals, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in path_vals:
        if isinstance(leaf, jax.Array):
            paths.append(tree_path_to_str(path))
    return paths


def assert_checkpoint_payload_is_host(name, tree):
    paths = find_jax_array_leaves(tree)
    if not paths:
        return

    preview = "\n".join(f"  {path}" for path in paths[:16])
    remaining = len(paths) - min(len(paths), 16)
    if remaining > 0:
        preview = f"{preview}\n  ... and {remaining} more"
    raise RuntimeError(
        f"Checkpoint tree `{name}` still contains jax.Array leaves after preparation:\n"
        f"{preview}"
    )


def tree_path_to_str(path):
    """Converts flattened tree path with leaves into a string.

    Args:
        path: leaf path
    Returns:
        A combined string representation of the path
    """

    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "name"):
            parts.append(str(p.name))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return "/".join(parts)
