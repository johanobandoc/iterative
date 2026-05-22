import grain
import numpy as np
from pathlib import Path
from grain.multiprocessing import SharedMemoryArray


class ChunkStreamingBatchSampler:
    def __init__(self, tokens, size=None):
        self.tokens = tokens
        self.size = len(tokens) if size is None else size
        self.batch_iter = 0
        self.built_ready = False
        self.slot_stream_starts = np.empty((0,), dtype=np.int32)

    def build(self, batch_size, max_seq_len):
        target_len = max_seq_len + 1
        stride = max_seq_len
        empty_starts = np.empty((0, batch_size), dtype=np.int32)
        empty_resets = np.empty((0, batch_size), dtype=np.bool_)

        stream_len = self.size // batch_size
        self.slot_stream_starts = (
            np.arange(batch_size, dtype=np.int32) * np.int32(stream_len)
        )
        if stream_len < target_len:
            self.built_starts = empty_starts
            self.built_ends = empty_starts
            self.built_resets = empty_resets
            self.built_batch_size = batch_size
            self.built_max_seq_len = max_seq_len
            self.built_ready = True
            self.batch_iter = 0
            return 0

        num_batches = 1 + (stream_len - target_len) // stride
        batch_offsets = np.arange(num_batches, dtype=np.int32)[:, None] * np.int32(
            stride
        )
        self.built_starts = self.slot_stream_starts[None, :] + batch_offsets
        self.built_ends = self.built_starts + np.int32(target_len)
        self.built_resets = np.zeros((num_batches, batch_size), dtype=np.bool_)
        self.built_resets[0] = True
        self.built_batch_size = batch_size
        self.built_max_seq_len = max_seq_len
        self.built_ready = True
        self.batch_iter = 0
        return len(self.built_starts)

    def next_batch(self, batch_size: int, max_seq_len: int):
        starts, ends, _ = self.next_batch_with_reset(batch_size, max_seq_len)
        return starts, ends

    def next_batch_with_reset(self, batch_size: int, max_seq_len: int):
        if (
            not self.built_ready
            or self.built_batch_size != batch_size
            or self.built_max_seq_len != max_seq_len
        ):
            self.build(batch_size, max_seq_len)
        if self.batch_iter >= len(self.built_starts):
            raise StopIteration(
                "Insufficient equal-chunk streaming windows ahead; hit tail of shard."
            )
        batch_idx = self.batch_iter
        self.batch_iter += 1
        starts = self.built_starts[batch_idx].tolist()
        ends = self.built_ends[batch_idx].tolist()
        reset_mask = self.built_resets[batch_idx].copy()
        return starts, ends, reset_mask


def make_window_sampler(tokens, size=None):
    return ChunkStreamingBatchSampler(tokens, size=size)


class CustomSharedMemoryDataSource(grain.sources.SharedMemoryDataSource):
    def __init__(self, elements=None, *, name=None):
        if elements is not None:
            elements = [str(Path(p).resolve()) for p in elements]
        super().__init__(elements, name=name)
        self.files = [] if elements is None else elements
        self.name = name

    def __repr__(self):
        return f"Fineweb10BSharedMemoryData(name={self.name}, len={len(self.files)})"


class LoadShardTokens(grain.transforms.Map):
    def map(self, path):
        return load_shard_tokens(path)


def load_shard_tokens(path):
    file = Path(path)

    header = np.fromfile(str(file), count=256, dtype=np.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])

    with file.open("rb", buffering=0) as f:
        f.seek(256 * 4)
        tokens = SharedMemoryArray((num_tokens,), dtype=np.uint16)
        nbytes = f.readinto(tokens)
        assert nbytes == 2 * num_tokens, (
            "number of tokens read does not match header"
        )

    return {
        "path": str(file),
        "tokens": tokens,
        "size": num_tokens,
    }


def make_grain_shard_loader(files):
    ds = grain.MapDataset.source([str(p) for p in files]).map(LoadShardTokens())
    ds = ds.to_iter_dataset(read_options=grain.ReadOptions(prefetch_buffer_size=4))
    return ds


__all__ = [
    "ChunkStreamingBatchSampler",
    "CustomSharedMemoryDataSource",
    "LoadShardTokens",
    "load_shard_tokens",
    "make_window_sampler",
    "make_grain_shard_loader",
]
