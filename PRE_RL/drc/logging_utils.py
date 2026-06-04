import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def resolve_storage_base_dir(storage_base_dir: str, project_root: str = PROJECT_ROOT) -> str:
    if storage_base_dir:
        return os.path.abspath(os.path.expanduser(storage_base_dir))

    env_storage_base_dir = os.environ.get("PRE_RL_STORAGE_DIR")
    if env_storage_base_dir:
        return os.path.abspath(os.path.expanduser(env_storage_base_dir))

    scratch_root = os.environ.get("SCRATCH")
    if scratch_root and os.path.isdir(os.path.expanduser(scratch_root)):
        return os.path.join(os.path.abspath(os.path.expanduser(scratch_root)), os.path.basename(project_root))

    return project_root


def configure_run_logging(args, project_root: str = PROJECT_ROOT) -> str:
    args.storage_base_dir = resolve_storage_base_dir(args.storage_base_dir, project_root=project_root)

    runs_base_dir = os.path.join(args.storage_base_dir, "runs")
    wandb_base_dir = os.path.join(args.storage_base_dir, "wandb")
    args.out_dir = os.path.join(
        runs_base_dir,
        args.env_id,
        args.group_name,
        f"{args.run_name}_randint{args.randint}",
    )
    os.makedirs(args.out_dir, exist_ok=True)

    if args.track:
        wandb_cache_dir = os.path.join(wandb_base_dir, ".cache")
        wandb_data_dir = os.path.join(wandb_base_dir, ".data")
        wandb_artifact_dir = os.path.join(wandb_base_dir, "artifacts")
        os.makedirs(wandb_base_dir, exist_ok=True)
        os.makedirs(wandb_cache_dir, exist_ok=True)
        os.makedirs(wandb_data_dir, exist_ok=True)
        os.makedirs(wandb_artifact_dir, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", wandb_base_dir)
        os.environ.setdefault("WANDB_CACHE_DIR", wandb_cache_dir)
        os.environ.setdefault("WANDB_DATA_DIR", wandb_data_dir)
        os.environ.setdefault("WANDB_ARTIFACT_DIR", wandb_artifact_dir)

    print(
        f"[logging] storage_base_dir={args.storage_base_dir}, out_dir={args.out_dir}"
        + (f", wandb_dir={wandb_base_dir}" if args.track else ""),
        flush=True,
    )
    return wandb_base_dir
