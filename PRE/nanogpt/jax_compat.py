import contextlib

import jax


try:
    from jax.sharding import auto_axes as _auto_axes
except ImportError:
    _auto_axes = None

try:
    from jax.sharding import set_mesh as _set_mesh
except ImportError:
    _set_mesh = getattr(jax, "set_mesh", None)

try:
    from jax.sharding import reshard as _reshard
except ImportError:
    _reshard = None


def auto_axes(*args, **kwargs):
    if _auto_axes is not None:
        return _auto_axes(*args, **kwargs)

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(fn):
        return fn

    return decorator


def mesh_context(mesh):
    if _set_mesh is None:
        return contextlib.nullcontext()

    ctx = _set_mesh(mesh)
    if hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__"):
        return ctx
    return contextlib.nullcontext()


def reshard_like(x, ref):
    if _reshard is None:
        return x
    return _reshard(x, jax.typeof(ref).sharding.spec)
