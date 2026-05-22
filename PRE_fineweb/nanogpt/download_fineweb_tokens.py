# This file is directly taken from here:
# https://github.com/KellerJordan/modded-nanogpt/blob/master/data/cached_fineweb10B.py

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "num_chunks",
        nargs="?",
        type=int,
        default=103,
        help="Number of FineWeb training chunks to download.",
    )
    parser.add_argument(
        "--data_dir",
        "--fineweb_dir",
        dest="data_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fineweb10B",
        help="Directory where the FineWeb token shards will be stored.",
    )
    return parser.parse_args()


# Download the GPT-2 tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get(fname, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / fname).exists():
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=str(data_dir),
        )


def main():
    args = parse_args()

    get("fineweb_val_%06d.bin" % 0, args.data_dir)
    for i in range(1, args.num_chunks + 1):
        get("fineweb_train_%06d.bin" % i, args.data_dir)


if __name__ == "__main__":
    main()
