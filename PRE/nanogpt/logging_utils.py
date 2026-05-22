import os
import dataclasses
import inspect
import tempfile
import time
from pathlib import Path

import jax
import numpy as np


def _default_scratch_log_root():
    explicit = os.environ.get("JAXNANOGPT_LOG_ROOT")
    if explicit:
        return Path(explicit).expanduser()

    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch).expanduser() / "jaxnanogpt_logs"

    user = os.environ.get("USER")
    if user:
        shared_scratch = Path("/scratch") / user
        if shared_scratch.exists():
            return shared_scratch / "jaxnanogpt_logs"

    return None


def configure_scratch_logging():
    """Keep run-local logging artifacts off small home filesystems."""
    log_root = _default_scratch_log_root()
    if log_root is None:
        return None

    log_root.mkdir(parents=True, exist_ok=True)
    defaults = {
        "WANDB_DIR": log_root / "wandb",
        "WANDB_CACHE_DIR": log_root / "wandb_cache",
        "WANDB_DATA_DIR": log_root / "wandb_data",
        "TMPDIR": log_root / "tmp",
        "MPLCONFIGDIR": log_root / "matplotlib",
    }
    for env_name, path in defaults.items():
        path.mkdir(parents=True, exist_ok=True)
        if env_name in ("TMPDIR", "MPLCONFIGDIR"):
            os.environ[env_name] = str(path)
        else:
            os.environ.setdefault(env_name, str(path))
    tempfile.tempdir = os.environ["TMPDIR"]
    return log_root


configure_scratch_logging()


def serialize_for_logging(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: serialize_for_logging(value.__dict__[field.name])
            for field in dataclasses.fields(value)
            if field.name in value.__dict__
        }
    if isinstance(value, dict):
        return {str(key): serialize_for_logging(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_logging(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_run_metadata(cfg, job_type):
    seed = 0 if cfg.seed is None else int(cfg.seed)
    if cfg.tracking.exp_name:
        group_name = cfg.tracking.exp_name
        run_name = f"{cfg.tracking.exp_name}_s{seed}_{int(time.time())}"
    else:
        group_name = f"nanogpt_{job_type}"
        if getattr(cfg.model, "model_backend", "gpt") in ("moe", "shared_kv"):
            run_name = (
                f"{group_name}_{cfg.model.model_backend}_d{cfg.model.num_layers}"
                f"_e{cfg.model.num_experts}_h{cfg.model.d_emb}"
                f"_s{seed}_{int(time.time())}"
            )
        else:
            run_name = (
                f"{group_name}_layers{cfg.model.num_layers}_d{cfg.model.d_emb}"
                f"_s{seed}_{int(time.time())}"
            )
    return group_name, run_name


def init_wandb(cfg, job_type, extra_config=None):
    import wandb

    group_name, run_name = make_run_metadata(cfg, job_type)
    config_payload = serialize_for_logging(cfg)
    if extra_config is not None:
        config_payload.update(serialize_for_logging(extra_config))
    init_timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "120"))
    service_wait = float(
        os.environ.get("WANDB_SERVICE_WAIT", os.environ.get("WANDB__SERVICE_WAIT", "180"))
    )
    settings_kwargs = {"init_timeout": init_timeout}
    if "x_service_wait" in inspect.signature(wandb.Settings).parameters:
        settings_kwargs["x_service_wait"] = service_wait
    try:
        run = wandb.init(
            project=cfg.tracking.wandb_project_name,
            entity=cfg.tracking.wandb_entity or None,
            config=config_payload,
            group=group_name,
            name=run_name,
            job_type=job_type,
            tags=list(cfg.tracking.wandb_tags),
            save_code=True,
            settings=wandb.Settings(**settings_kwargs),
        )
    except Exception as exc:
        print(f"W&B init failed: {exc}. Continuing without W&B tracking.")
        return None, group_name, run_name
    return run, group_name, run_name
