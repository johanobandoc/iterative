import jax
import hashlib
import numpy as np
import orbax.checkpoint as ocp
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


def extract_shapes_and_dtypes(tree):
    """Extracts the shapes and dtypes of leaves (arrays) from a pytree.

    Args:
        tree: A PyTree instance
    Returns:
        A dictionary of leaves where the value corresponding to a leaf
        contains the shape and dtype of that leaf. A leaf here represents
        a jax array (e.g. model weights).
    """

    path_vals, _ = jax.tree_util.tree_flatten_with_path(tree)
    flattened = {}
    for path, leaf in path_vals:
        if leaf is None:
            continue
        if not hasattr(leaf, "shape"):
            continue
        flattened[tree_path_to_str(path)] = leaf
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in flattened.items()}


def get_schema_hash(tree):
    """Extracts the schema (shapes and dtypes) of a pytree, and calculates
    a hash of it."""
    schema = extract_shapes_and_dtypes(tree)
    entries = [f"{k}:{shape}:{dtype}" for k, (shape, dtype) in schema.items()]
    entries.sort()
    blob = "\n".join(entries)
    return hashlib.sha256(blob.encode()).hexdigest()


def print_diff(params_struct, ckpt_struct):
    """Prints the difference between two abstract pytrees structures."""

    params_schema = extract_shapes_and_dtypes(params_struct)
    ckpt_schema = extract_shapes_and_dtypes(ckpt_struct)

    param_schema_set = set(params_schema)
    ckpt_schema_set = set(ckpt_schema)

    missing = sorted(param_schema_set - ckpt_schema_set)
    extra = sorted(ckpt_schema_set - param_schema_set)
    mismatch = sorted(
        k
        for k in params_schema.keys() & ckpt_schema.keys()
        if params_schema[k] != ckpt_schema[k]
    )

    if not missing and not extra and not mismatch:
        print("Pytree match!")
        return True

    if missing:
        print("\nMissing in checkpoint:")
        for key in missing:
            print(" ", key, params_schema[key])

    if extra:
        print("\nExtra in checkpoint:")
        for key in extra:
            print(" ", key, ckpt_schema[key])

    if mismatch:
        print("\nShape or dtype mismatch found!")
        for key in mismatch:
            print("Key: ", key)
            print("    Param schema: ", params_schema[key])
            print("    Ckpt  schema: ", ckpt_schema[key], "\n")


def validate_checkpoint(params_struct, ckpt_struct):
    """Checks if the current param pytree is valid for a given checkpoint.

    Args:
        param_struct: Abstract pytree of the current params
        ckpt_struct : Abstract pytree of the checkpoint
    Returns:
        True/False depending on whether the param abstract pytree matches
        with the abstract pytree of the given checkpoint.
    """

    model_hash = get_schema_hash(params_struct)
    ckpt_hash = get_schema_hash(ckpt_struct)
    return model_hash == ckpt_hash


# Validate checkpoint structure, dtypes, and shapes before restoring weights.
def load_weights_from_checkpoint_with_validation(path, params, sharding):
    print(f"Reading checkpoint metadata from: {path}")
    with ocp.PyTreeCheckpointer() as ckptr:
        ckpt_metadata = ckptr.metadata(path)

    params_struct = jax.tree.map(ocp.utils.to_shape_dtype_struct, params)
    ckpt_struct = ckpt_metadata.item_metadata.tree

    print("Validating params structure and checkpoint being loaded...\n")
    is_valid = validate_checkpoint(params_struct, ckpt_struct)

    if not is_valid:
        print_diff(params_struct, ckpt_struct)
        raise RuntimeError(
            "\nThe model structure does not match with the checkpoint being loaded!"
        )

    print(f"Restoring params from: {path}")
    item, transforms = sharding, None
    restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            path,
            args=ocp.args.PyTreeRestore(
                item=item, transforms=transforms, restore_args=restore_args
            ),
        )
