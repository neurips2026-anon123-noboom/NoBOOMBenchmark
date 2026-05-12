"""Utilities for shared-memory cached preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import logging
import struct
import sys
import time
from typing import Callable, Tuple, Union, Dict, Any, Optional

from tqdm import trange

import torch

TransformFn = Callable[[torch.Tensor], torch.Tensor]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedCacheMetadata:
    """Metadata describing a preprocessed cache entry."""

    num_features: int
    seq_len: Union[int, Tuple[int, ...]]
    length: int
    ndim: int
    window_size: int


class SharedCacheFormat:
    """Binary layout helpers for shared-memory cache blobs."""

    MAGIC = b"NBCACHE1"
    VERSION = 1
    HEADER_FORMAT = "<8sQQQ"
    COUNT_FORMAT = "<QQ"
    TENSOR_PREFIX_FORMAT = "<BBH"
    OFFSET_FORMAT = "<QQ"

    @classmethod
    def header_size(cls) -> int:
        return struct.calcsize(cls.HEADER_FORMAT)


DTYPE_TO_CODE = {
    torch.float32: 1,
    torch.float64: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.int16: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bool: 8,
}
CODE_TO_DTYPE = {value: key for key, value in DTYPE_TO_CODE.items()}


def open_shared_memory(name: str, *, create: bool = False, size: Optional[int] = None) -> shared_memory.SharedMemory:
    """Open a shared-memory segment with Ray-compatible tracking defaults."""

    kwargs: Dict[str, Any] = {"name": name, "create": create}
    if size is not None:
        kwargs["size"] = size
    if sys.version_info >= (3, 13):
        kwargs["track"] = False
    return shared_memory.SharedMemory(**kwargs)


def build_shared_cache_payload(
    parent,
    transform: Optional[TransformFn],
    target_transform: Optional[TransformFn],
) -> Tuple[bytes, bytes, int]:
    """Build the shared-memory payload for a transformed dataset."""
    build_start = time.perf_counter()
    datapoint_read_seconds = 0.0
    transform_seconds = 0.0
    target_transform_seconds = 0.0
    tensor_serialize_seconds = 0.0
    metadata = bytearray()
    data = bytearray()

    for idx in trange(len(parent), desc="Building shared-memory payload"):
        datapoint_start = time.perf_counter()
        inputs, targets = parent.get_datapoint(idx)
        datapoint_read_seconds += time.perf_counter() - datapoint_start
        metadata.extend(struct.pack("<8s", SharedCacheFormat.MAGIC))
        if transform is not None:
            transform_start = time.perf_counter()
            inputs = (transform(inputs[0]),)
            transform_seconds += time.perf_counter() - transform_start
        if target_transform is not None:
            target_transform_start = time.perf_counter()
            targets = (target_transform(targets[0]),)
            target_transform_seconds += time.perf_counter() - target_transform_start
        metadata.extend(struct.pack(SharedCacheFormat.COUNT_FORMAT, len(inputs), len(targets)))
        tensor_serialize_start = time.perf_counter()
        for tensor in tuple(inputs) + tuple(targets):
            tensor = tensor.cpu().contiguous()
            dtype_code = DTYPE_TO_CODE.get(tensor.dtype)
            if dtype_code is None:
                raise ValueError(f"Unsupported tensor dtype for shared cache: {tensor.dtype}")
            shape = tensor.shape
            ndim = len(shape)
            metadata.extend(struct.pack(SharedCacheFormat.TENSOR_PREFIX_FORMAT, dtype_code, ndim, 0))
            if ndim:
                metadata.extend(struct.pack(f"<{ndim}Q", *shape))
            metadata.extend(struct.pack(SharedCacheFormat.OFFSET_FORMAT, len(data), tensor.numel()))
            data.extend(tensor.numpy().tobytes())
        tensor_serialize_seconds += time.perf_counter() - tensor_serialize_start
    num_items = len(parent)
    total_seconds = time.perf_counter() - build_start
    logger.info(
        "Built shared cache payload for %d items in %.2fs (read=%.2fs, transform=%.2fs, "
        "target_transform=%.2fs, tensor_serialize=%.2fs, metadata=%.2f MiB, data=%.2f MiB).",
        num_items,
        total_seconds,
        datapoint_read_seconds,
        transform_seconds,
        target_transform_seconds,
        tensor_serialize_seconds,
        len(metadata) / (1024 ** 2),
        len(data) / (1024 ** 2),
    )
    return bytes(metadata), bytes(data), num_items


def write_shared_cache(shm_name: str, payload: Tuple[bytes, bytes, int]) -> int:
    """Write the cache payload into shared memory.

    Args:
        shm_name (str): Shared memory name to create.
        payload (Tuple[bytes, bytes, int]): Serialized metadata, tensor data, and item count.

    Returns:
        int: Total size of the shared-memory segment in bytes.
    """

    write_start = time.perf_counter()
    metadata, data, num_items = payload
    header_size = SharedCacheFormat.header_size()
    metadata_size = len(metadata)
    total_size = header_size + metadata_size + len(data)

    try:
        shm = open_shared_memory(shm_name)
        shm.unlink()
        shm.close()
    except FileNotFoundError:
        pass

    shm = open_shared_memory(shm_name, create=True, size=total_size)
    header = struct.pack(
        SharedCacheFormat.HEADER_FORMAT,
        SharedCacheFormat.MAGIC,
        SharedCacheFormat.VERSION,
        num_items,
        metadata_size,
    )
    shm.buf[:header_size] = header
    shm.buf[header_size:header_size + metadata_size] = metadata
    data_start = header_size + metadata_size
    shm.buf[data_start:data_start + len(data)] = data
    shm.close()
    logger.info(
        "Wrote shared cache '%s' in %.2fs (items=%d, total=%.2f MiB, metadata=%.2f MiB, data=%.2f MiB).",
        shm_name,
        time.perf_counter() - write_start,
        num_items,
        total_size / (1024 ** 2),
        metadata_size / (1024 ** 2),
        len(data) / (1024 ** 2),
    )
    return total_size
