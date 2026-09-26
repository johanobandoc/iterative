import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from shared_fineweb_dataloader import (  # noqa: E402
    load_shard_tokens,
    make_window_sampler,
    make_grain_shard_loader,
)


__all__ = [
    "load_shard_tokens",
    "make_window_sampler",
    "make_grain_shard_loader",
]
