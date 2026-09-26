import contextlib

import jax


try:
    from jax.sharding import set_mesh as _set_mesh
except ImportError:
    _set_mesh = getattr(jax, "set_mesh", None)


def mesh_context(mesh):
    if _set_mesh is None:
        return contextlib.nullcontext()

    ctx = _set_mesh(mesh)
    if hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__"):
        return ctx
    return contextlib.nullcontext()
