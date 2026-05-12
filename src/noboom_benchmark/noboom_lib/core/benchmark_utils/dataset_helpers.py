from typing import Callable, Literal, List, Optional, Sequence, Tuple, Union
from multiprocessing import shared_memory
import multiprocessing
import os
import struct
import logging
from dataclasses import dataclass
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import delayed
from lightning.fabric.utilities.seed import pl_worker_init_function
from tqdm_joblib import ParallelPbar

import torch
from torch.utils.data import ConcatDataset
import ray

from noboom.tsad.data import TorchDataset
from timesead.data.dataset import collate_fn
from timesead.data.transforms import (DatasetSource, make_dataset_split,
                                      PipelineDataset, Transform)

from ..dataset_variants import (
    DatasetVariant,
    OME_EXTENSION_TRAIN_MODE,
    list_ome_extension_csv_files,
    resolve_dataset_variant,
)
from ..logging import setup_logging
from ..tune.dataset_helper_actors import get_preprocessed_cache_registry
from ..tune.shared_cache import (
    CODE_TO_DTYPE,
    SharedCacheFormat,
    open_shared_memory,
)


logger = logging.getLogger(__name__)


TransformCallable = Callable[[Transform], Transform]


def _diagnostic_logging_enabled() -> bool:
    value = os.getenv("NOBOOM_WORKER_DEBUG", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class MetaInfo:
    dtype: int
    shape: Tuple[int, ...]
    offset: int
    numel: int


@dataclass(frozen=True)
class DatasetLoadResult:
    train_data: DatasetSource
    val_data: DatasetSource
    test_data: DatasetSource
    extra_train_data: Optional[DatasetSource] = None
    extra_val_data: Optional[DatasetSource] = None
    extra_test_data: Optional[DatasetSource] = None
    extra_train_mode: Optional[str] = None
    extra_val_mode: Optional[str] = None
    extra_test_mode: Optional[str] = None


def _record_timing_stat(
    logger_: logging.Logger,
    timing_stats: dict,
    *,
    stage: str,
    item: int,
    elapsed_seconds: float,
    log_interval: int,
    slow_call_threshold_s: float,
    extra: Optional[str] = None,
) -> None:
    if not _diagnostic_logging_enabled():
        return
    stats = timing_stats.setdefault(
        stage,
        {"count": 0, "total": 0.0, "max": 0.0},
    )
    stats["count"] += 1
    stats["total"] += elapsed_seconds
    stats["max"] = max(stats["max"], elapsed_seconds)
    count = stats["count"]
    should_log = (
        count <= 3
        or elapsed_seconds >= slow_call_threshold_s
        or count % log_interval == 0
    )
    if should_log:
        avg_seconds = stats["total"] / count
        suffix = f" {extra}" if extra else ""
        logger_.debug(
            "Timing[%s] count=%d item=%d elapsed=%.4fs avg=%.4fs max=%.4fs pid=%d%s",
            stage,
            count,
            item,
            elapsed_seconds,
            avg_seconds,
            stats["max"],
            os.getpid(),
            suffix,
        )


def _init_dataloader_worker(worker_id: int) -> None:
    """Apply Lightning worker seeding and configure logging for DataLoader subprocesses."""
    pl_worker_init_function(worker_id)
    setup_logging(verbosity=1 if _diagnostic_logging_enabled() else 0)


def _refresh_raw_data_root(
    dataset_name: str,
    *,
    include_generated: bool,
) -> str:
    from ..data_prep.pipeline import ensure_dataset_assets

    runtime_config = SimpleNamespace(
        seafile_username=os.getenv("SEAFILE_USERNAME"),
        seafile_password=os.getenv("SEAFILE_PASS"),
    )
    refreshed_root, _ = ensure_dataset_assets(
        dataset_name,
        runtime_config=runtime_config,
        include_generated=include_generated,
    )
    return str(refreshed_root)


class SequenceTensorDataset:
    def __init__(
        self,
        samples: Sequence[Union[np.ndarray, torch.Tensor]],
        targets: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
    ) -> None:
        self._samples = [self._to_numpy(sample, dtype=np.float32) for sample in samples]
        if not self._samples:
            raise ValueError("SequenceTensorDataset requires at least one sample.")
        if targets is None:
            self._targets = [
                np.zeros(sample.shape[0], dtype=np.int64)
                for sample in self._samples
            ]
        else:
            self._targets = [self._to_numpy(target) for target in targets]
            if len(self._samples) != len(self._targets):
                raise ValueError("SequenceTensorDataset samples/targets length mismatch.")

    @staticmethod
    def _to_numpy(
        array: Union[np.ndarray, torch.Tensor],
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        result = np.asarray(array)
        if dtype is not None and result.dtype != dtype:
            result = result.astype(dtype, copy=False)
        return result

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        sample = torch.from_numpy(self._samples[item])
        target = torch.from_numpy(self._targets[item])
        return (sample,), (target,)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def seq_len(self) -> List[int]:
        return [sample.shape[0] for sample in self._samples]

    @property
    def num_features(self) -> int:
        return int(self._samples[0].shape[-1])

    @property
    def ndim(self) -> int:
        return int(self._samples[0].ndim)

    @property
    def window_size(self) -> Optional[int]:
        return None


class SkipNaNWindows(Transform):
    """Filter out windows with NaNs prior to batching.

    Args:
        parent (Transform): Base transform that yields ``(x,), (y,)`` tuples.
        build_on_init (bool): Whether to immediately build the valid index mask.
    """
    def __init__(self, parent: Transform, *, build_on_init: bool = False):
        """Initialize the NaN-filtering transform.

        Args:
            parent (Transform): Base transform yielding ``(x,), (y,)`` tuples.
            build_on_init (bool): Whether to build the index immediately. Defaults to False.
        """
        super().__init__(parent)
        self.base = parent
        self._valid_indices = None
        self.stats = None
        if build_on_init:
            self.build_index(show_progress=True)

    def build_index(self, *, show_progress: bool = False):
        """Precompute valid indices by scanning for NaNs.

        Args:
            show_progress (bool): Enable tqdm progress output when available.
                Defaults to False.

        Returns:
            dict: Summary statistics about valid/invalid windows.
        """
        n = len(self.base)

        def _is_valid_datapoint(base, i: int) -> bool:
            x, y = base.get_datapoint(i)

            for x_ in x:
                if torch.isnan(x_).any():
                    return False
            for y_ in y:
                if torch.isnan(y_).any():
                    return False
            return True

        valid = ParallelPbar("Scanning windows for NaNs")(n_jobs=4)(
            delayed(_is_valid_datapoint)(self.base, i) for i in range(n)
        )
        valid = torch.tensor(valid, dtype=torch.bool)

        self._valid_indices = torch.nonzero(valid, as_tuple=False).flatten().to(torch.int64)
        n_valid = int(self._valid_indices.numel())
        self.stats = {"n_total": n, "n_valid": n_valid, "n_invalid": n - n_valid}
        return self.stats

    def __len__(self):
        """Return the number of valid windows after filtering.

        Returns:
            int: Number of valid windows.

        Raises:
            RuntimeError: If ``build_index`` has not been called.
        """
        if self._valid_indices is None:
            raise RuntimeError("Call build_index() before using this dataset.")
        return int(self._valid_indices.numel())

    def _get_datapoint_impl(self, i: int):
        """Retrieve a datapoint by mapping from filtered to base index.

        Args:
            i (int): Index into the filtered dataset.

        Returns:
            Any: Datapoint from the base dataset.

        Raises:
            RuntimeError: If ``build_index`` has not been called.
        """
        if self._valid_indices is None:
            raise RuntimeError("Call build_index() before using this dataset.")
        base_i = int(self._valid_indices[i].item())
        return self.base.get_datapoint(base_i)

class PreprocessDatasetWrapper(Transform):

    def __init__(self, parent: Transform,
                 n_features: Optional[int] = None,
                 transform: Callable[[torch.Tensor], torch.Tensor] = None,
                 target_transform: Callable[[torch.Tensor], torch.Tensor] = None,
                 cache_preprocessed: bool = False,
                 shared_cache_name: Optional[str] = None,
                 mode: str = "train",
                 verbosity: int = 0):
        """Wrap a dataset with optional input/target transforms.

        Args:
            parent (Transform): Base transform to wrap.
            n_features (Optional[int]): Explicit feature count. Defaults to None.
            transform (Callable[[torch.Tensor], torch.Tensor] | None): Input transform.
                Defaults to None.
            target_transform (Callable[[torch.Tensor], torch.Tensor] | None): Target transform.
                Defaults to None.
            cache_preprocessed (bool): Whether to cache transformed windows in shared memory.
            shared_cache_name (Optional[str]): Shared memory name to reuse across processes.
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbosity > 0 else logging.INFO)
        self.logger.info(
            "Initializing dataset wrapper with transform %s and target transform %s, shared cache: %s.",
            transform,
            target_transform,
            "enabled" if cache_preprocessed else "disabled",
        )
        self._n_features = n_features
        self.transform = transform
        self.target_transform = target_transform
        self.cache_preprocessed = cache_preprocessed
        self.shared_cache_name = shared_cache_name
        self.mode = mode
        self._shared_memory: Optional[shared_memory.SharedMemory] = None
        self._shared_index: Optional[list] = None
        self._shared_data_start: Optional[int] = None
        self._cache_initialized = False
        self._cached_tensor_views = []
        self._timing_enabled = _diagnostic_logging_enabled()
        self._timing_stats = {}
        self._timing_log_interval = 512
        self._timing_slow_call_threshold_s = 0.05
        if self.cache_preprocessed and not self.shared_cache_name:
            raise ValueError("shared_cache_name must be provided when cache_preprocessed is enabled.")
        if self.cache_preprocessed:
            self.logger.info("Ensuring shared cache is available for mode '%s'.", self.mode)
            self._ensure_cache()

    def _record_timing(
        self,
        stage: str,
        *,
        item: int,
        elapsed_seconds: float,
        extra: Optional[str] = None,
    ) -> None:
        """Record and periodically log per-item timing statistics."""
        if not self._timing_enabled:
            return
        _record_timing_stat(
            self.logger,
            self._timing_stats,
            stage=stage,
            item=item,
            elapsed_seconds=elapsed_seconds,
            log_interval=self._timing_log_interval,
            slow_call_threshold_s=self._timing_slow_call_threshold_s,
            extra=extra,
        )

    def _apply_transforms(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Apply transforms to a datapoint without using the cache."""
        if not self._timing_enabled:
            inputs, targets = self.parent.get_datapoint(item)
            if self.transform is not None:
                inputs = (self.transform(inputs[0]),)
            if self.target_transform is not None:
                targets = (self.target_transform(targets[0]),)
            return inputs, targets
        self.logger.debug("Applying transforms for item %d in mode '%s'.", item, self.mode)
        total_start = time.perf_counter()
        source_start = total_start
        inputs, targets = self.parent.get_datapoint(item)
        source_elapsed = time.perf_counter() - source_start
        self._record_timing(
            "source_get",
            item=item,
            elapsed_seconds=source_elapsed,
            extra=f"mode={self.mode}",
        )
        if self.transform is not None:
            transform_start = time.perf_counter()
            inputs = (self.transform(inputs[0]),)
            self._record_timing(
                "input_transform",
                item=item,
                elapsed_seconds=time.perf_counter() - transform_start,
                extra=f"mode={self.mode}",
            )
        if self.target_transform is not None:
            target_transform_start = time.perf_counter()
            targets = (self.target_transform(targets[0]),)
            self._record_timing(
                "target_transform",
                item=item,
                elapsed_seconds=time.perf_counter() - target_transform_start,
                extra=f"mode={self.mode}",
            )
        self._record_timing(
            "preprocess_total",
            item=item,
            elapsed_seconds=time.perf_counter() - total_start,
            extra=f"mode={self.mode}",
        )
        return inputs, targets

    def _parse_shared_cache(self, shm: shared_memory.SharedMemory) -> bool:
        self.logger.info("Parsing shared cache header for '%s'.", self._shared_memory_name)
        header_size = SharedCacheFormat.header_size()
        header = bytes(shm.buf[:header_size])
        magic, version, num_items, metadata_size = struct.unpack(SharedCacheFormat.HEADER_FORMAT, header)
        if magic != SharedCacheFormat.MAGIC or version != SharedCacheFormat.VERSION:
            raise ValueError("Shared cache '%s' has incompatible header.", self._shared_memory_name)
        if self.parent is not None and num_items != len(self.parent):
            raise ValueError(
                f"Shared cache '{self._shared_memory_name}' length mismatch (expected {len(self.parent)}, found {num_items})."
            )
        self.logger.debug(
            "Shared cache '%s' header ok (items=%d, metadata_size=%d).",
            self._shared_memory_name,
            num_items,
            metadata_size,
        )
        metadata = bytes(shm.buf[header_size:header_size + metadata_size])
        offset = 0
        index: list = []
        for i in range(num_items):
            magic, = struct.unpack_from("<8s", metadata, offset)
            if magic != SharedCacheFormat.MAGIC:
                raise ValueError(f"Shared cache '{self._shared_memory_name}' has incompatible header, item {i}., {magic}, {SharedCacheFormat.MAGIC}")
            offset += struct.calcsize("<8s")
            in_count, tgt_count = struct.unpack_from(SharedCacheFormat.COUNT_FORMAT, metadata, offset)
            offset += struct.calcsize(SharedCacheFormat.COUNT_FORMAT)
            inputs = []
            targets = []
            for _ in range(in_count + tgt_count):
                dtype_code, ndim, _reserved = struct.unpack_from(SharedCacheFormat.TENSOR_PREFIX_FORMAT, metadata, offset)
                offset += struct.calcsize(SharedCacheFormat.TENSOR_PREFIX_FORMAT)
                shape = struct.unpack_from(f"<{ndim}Q", metadata, offset) if ndim > 0 else ()
                offset += struct.calcsize(f"<{ndim}Q") if ndim > 0 else 0
                data_offset, numel = struct.unpack_from(SharedCacheFormat.OFFSET_FORMAT, metadata, offset)
                offset += struct.calcsize(SharedCacheFormat.OFFSET_FORMAT)
                meta = MetaInfo(dtype=dtype_code,
                                shape=shape,
                                offset=data_offset,
                                numel=numel) #{"dtype": dtype_code, "shape": shape, "offset": data_offset, "numel": numel}
                if len(inputs) < in_count:
                    inputs.append(meta)
                else:
                    targets.append(meta)
            index.append((tuple(inputs), tuple(targets)))
        self._shared_memory = shm
        self._shared_index = index
        self._shared_data_start = header_size + metadata_size
        self.logger.info("Loaded shared cache '%s' with %d entries.", self._shared_memory_name, num_items)
        return True

    @property
    def _shared_memory_name(self) -> str:
        return f"{self.shared_cache_name}__{self.mode}"

    def _load_shared_cache(self) -> bool:
        """Load a cache from shared memory if available."""
        self.logger.info("Attempting to load shared cache '%s'.", self._shared_memory_name)
        try:
            shm = open_shared_memory(self._shared_memory_name)
        except FileNotFoundError:
            self.logger.info("Shared cache '%s' not found.", self._shared_memory_name)
            return False
        if self._parse_shared_cache(shm):
            self.logger.debug("Shared cache '%s' successfully loaded.", self._shared_memory_name)
            return True
        shm.close()
        self.logger.warning("Shared cache '%s' failed to parse.", self._shared_memory_name)
        return False

    def _ensure_cache(self) -> None:
        """Ensure the cache is populated, optionally using shared memory."""
        if self._cache_initialized:
            self.logger.debug("Cache already initialized for '%s'.", self._shared_memory_name)
            return
        if self.cache_preprocessed:
            self.logger.info(
                "Preparing cache for dataset '%s' (mode=%s).",
                self.shared_cache_name,
                self.mode,
            )
            registry = get_preprocessed_cache_registry()
            node_id = ray.get_runtime_context().get_node_id()
            cache_metadata = ray.get(
                registry.prepare_cache_for_dataset.remote(
                    self.shared_cache_name,
                    self.mode,
                    self._shared_memory_name,
                    node_id,
                    parent=self.parent,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
            )
            self.logger.debug(
                "Cache metadata received for '%s' (length=%s, num_features=%s, ndim=%s, window_size=%s).",
                self._shared_memory_name,
                cache_metadata.length,
                cache_metadata.num_features,
                cache_metadata.ndim,
                cache_metadata.window_size,
            )
            self.parent = None
            self.transform = None
            self.target_transform = None

            if not self._load_shared_cache():
                raise RuntimeError(f"Failed to load shared cache '{self._shared_memory_name}'.")

            self._length = cache_metadata.length
            self._seq_len = cache_metadata.seq_len
            self._n_features = cache_metadata.num_features
            self._ndim = cache_metadata.ndim
            self._window_size = cache_metadata.window_size
            self._cached_tensor_views = [None] * self._length
            self.logger.info("Cache ready for '%s' (%d items).", self._shared_memory_name, self._length)
        self._cache_initialized = True

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Apply optional preprocessing to the inputs and targets.

        Args:
            item (int): Index inside the wrapped transform.

        Returns:
            Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]: Transformed inputs/targets.
        """
        if self.cache_preprocessed:
            if not self._cache_initialized:
                raise RuntimeError(f"Cache '{self._shared_memory_name}' is not initialized.")

            if item > len(self._shared_index):
                raise IndexError(f"Cache '{self._shared_memory_name}' index {item} out of range. [{len(self._shared_index)}, {len(self)}]")
            inputs_meta, targets_meta = self._shared_index[item]
            cached_views = self._cached_tensor_views[item]
            cache_hit = cached_views is not None
            materialize_elapsed = 0.0
            total_start: Optional[float] = None
            if self._timing_enabled:
                total_start = time.perf_counter()
            if cached_views is None:
                materialize_start: Optional[float] = None
                if self._timing_enabled:
                    materialize_start = time.perf_counter()
                cached_views = self._cached_tensor_views[item] = [[], []]
                for i, metas in enumerate([inputs_meta, targets_meta]):
                    for meta in metas:
                        tensor = self._load_shared_tensor(meta)
                        cached_views[i].append(tensor)
                if self._timing_enabled and materialize_start is not None:
                    materialize_elapsed = time.perf_counter() - materialize_start
                    self._record_timing(
                        "shared_cache_materialize",
                        item=item,
                        elapsed_seconds=materialize_elapsed,
                        extra=(
                            f"mode={self.mode} tensors={len(inputs_meta) + len(targets_meta)}"
                        ),
                    )
            inputs = tuple(cached_views[0])
            targets = tuple(cached_views[1])
            if self._timing_enabled and total_start is not None:
                self._record_timing(
                    "shared_cache_get",
                    item=item,
                    elapsed_seconds=time.perf_counter() - total_start,
                    extra=(
                        f"mode={self.mode} cached_views={'hit' if cache_hit else 'miss'} "
                        f"tensors={len(inputs_meta) + len(targets_meta)} "
                        f"materialize={materialize_elapsed:.4f}s"
                    ),
                )

            return inputs, targets
        if self._timing_enabled:
            self.logger.debug(
                "Using on-the-fly preprocessing for mode '%s'.",
                self.mode,
            )
        return self._apply_transforms(item)

    def _load_shared_tensor(self, meta: MetaInfo) -> torch.Tensor:
        dtype = CODE_TO_DTYPE.get(meta.dtype)
        if dtype is None:
            raise ValueError(f"Unknown dtype code in shared cache: {meta.dtype}")
        numel = int(meta.numel)
        offset = int(meta.offset)
        base_offset = self._shared_data_start or 0
        flat = torch.frombuffer(self._shared_memory.buf, dtype=dtype, count=numel, offset=base_offset + offset)
        shape = tuple(int(dim) for dim in meta.shape)
        if shape:
            return flat.view(shape)
        return flat

    def __len__(self) -> Optional[int]:
        if self.cache_preprocessed:
            return self._length
        return super().__len__()

    @property
    def seq_len(self) -> Union[int, List[int], None]:
        if self.cache_preprocessed:
            return self._seq_len
        return super().seq_len

    @property
    def num_features(self) -> Union[int, Tuple[int, ...], None]:
        return self._n_features

    def __del__(self) -> None:
        if self._shared_memory is not None:
            self._shared_memory.close()

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def window_size(self) -> int:
        return self._window_size


def _select_dataset_features_inplace(
    dataset: TorchDataset,
    feature_indices: Sequence[int],
) -> None:
    indices = np.asarray(feature_indices, dtype=np.int64)
    dataset._samples = [sample[:, indices] for sample in dataset._samples]


def _load_ome_extension_train_source(variant: DatasetVariant) -> DatasetSource:
    if variant.extension_dir is None or variant.reduced_feature_names is None:
        raise ValueError("OME extension variant is missing extension_dir or reduced_feature_names.")

    samples: List[np.ndarray] = []
    for csv_path in list_ome_extension_csv_files(variant.extension_dir):
        frame = pd.read_csv(csv_path, usecols=list(variant.reduced_feature_names))
        samples.append(frame.to_numpy(dtype=np.float32, copy=False))

    dataset = SequenceTensorDataset(samples)
    return DatasetSource(dataset)


def get_dataset_data(
    dataset_name: str,
    splits: Tuple[float, float],
    version: str = '1.0',
    data_dir: str = 'data/',
    fast_load: bool = False,
    data_source: Literal["real", "synthetic", "both"] = "real",
    gen_to_org_factor: float = -1,
) -> DatasetLoadResult:
    """Load train/val/test splits for a dataset version.

    Args:
        dataset_name (str): Dataset slug to retrieve.
        splits (Tuple[float, float]): Train/validation split fractions.
        version (str): Dataset version string. Defaults to "1.0".
        data_dir (str): Root directory where datasets are stored. Defaults to "data/".
        fast_load (bool): Whether to enable cached loading paths. Defaults to False.

    Returns:
        DatasetLoadResult: Train/val/test dataset sources plus optional auxiliary sources.
    """
    resolved_data_dir = data_dir
    try:
        variant = resolve_dataset_variant(dataset_name, resolved_data_dir)
    except FileNotFoundError:
        logger.warning(
            "Dataset '%s' is missing raw training files under '%s'. Refreshing the raw dataset cache once.",
            dataset_name,
            data_dir,
        )
        resolved_data_dir = _refresh_raw_data_root(
            dataset_name,
            include_generated=data_source != "real",
        )
        variant = resolve_dataset_variant(dataset_name, resolved_data_dir)
    base_dataset_name = variant.base_dataset_name

    presplit_data = TorchDataset(base_dataset_name, version, root=resolved_data_dir, train=True, fast_load=fast_load)
    if variant.base_feature_indices is not None:
        _select_dataset_features_inplace(presplit_data, variant.base_feature_indices)

    train_data, val_data = make_dataset_split(
        presplit_data,
        *splits,
        axis='time')
    test_data = TorchDataset(base_dataset_name, version, root=resolved_data_dir, train=False, fast_load=fast_load)
    if variant.base_feature_indices is not None:
        _select_dataset_features_inplace(test_data, variant.base_feature_indices)
    test_data = DatasetSource(test_data)

    extra_train_data: Optional[DatasetSource] = None
    extra_val_data: Optional[DatasetSource] = None
    extra_test_data: Optional[DatasetSource] = None
    extra_train_mode: Optional[str] = None
    extra_val_mode: Optional[str] = None
    extra_test_mode: Optional[str] = None

    if data_source != 'real':
        extra_presplit_data = TorchDataset(
            base_dataset_name + '_tsst',
            version,
            root=resolved_data_dir,
            train=True,
            fast_load=fast_load,
        )
        if gen_to_org_factor > 0:
            org_seq_len = presplit_data.seq_len
            if isinstance(org_seq_len, int):
                org_seq_len = [org_seq_len]
            org_seq_len = sum(org_seq_len)
            extra_seq_len_list = extra_presplit_data.seq_len
            if isinstance(extra_seq_len_list, int):
                extra_seq_len_list = [extra_seq_len_list]
            extra_seq_len = sum(extra_seq_len_list)
            new_len = int(min(gen_to_org_factor*org_seq_len, extra_seq_len))
            extra_seq_len_list = np.asarray(extra_seq_len_list)
            cum_seq_len = np.cumsum(extra_seq_len_list)
            new_len = int(np.searchsorted(cum_seq_len, new_len, side='left'))
            logger.info(f"Old extra dataset length: {len(extra_presplit_data)}, new extra dataset length: {new_len}")
            percent = new_len / len(extra_presplit_data)
            extra_presplit_data, _ = make_dataset_split(
                extra_presplit_data,
                *[percent, 1 - percent],
                axis='batch')

        extra_train_data, extra_val_data = make_dataset_split(
            extra_presplit_data,
            *splits,
            axis='batch')
        extra_train_mode = 'train_tsst'
        extra_val_mode = 'val_tsst'
        logger.info(f"Extra train data: {len(extra_train_data)}, val data: {len(extra_val_data)}")
        extra_test_data = TorchDataset(
            base_dataset_name + '_tsst',
            version,
            root=resolved_data_dir,
            train=False,
            fast_load=fast_load,
        )
        extra_test_data = DatasetSource(extra_test_data)
        extra_test_mode = 'test_tsst'

    if variant.include_extension_train:
        extra_train_data = _load_ome_extension_train_source(variant)
        extra_train_mode = OME_EXTENSION_TRAIN_MODE

    return DatasetLoadResult(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        extra_train_data=extra_train_data,
        extra_val_data=extra_val_data,
        extra_test_data=extra_test_data,
        extra_train_mode=extra_train_mode,
        extra_val_mode=extra_val_mode,
        extra_test_mode=extra_test_mode,
    )


def assert_picklable(obj, name: str):
    import pickle
    try:
        pickle.dumps(obj)
        print(f"[OK] picklable: {name} ({type(obj)})")
    except Exception:
        print(f"[FAIL] not picklable: {name} ({type(obj)})")
        raise


class WindowLastElement(Transform):
    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)
        inputs = tuple(inp[:, -1] if inp.ndim == 3 else inp for inp in inputs)
        targets = tuple(tgt[:, -1] if tgt.ndim == 2 else tgt for tgt in targets)
        return inputs, targets

    def window_size(self) -> Optional[int]:
        return None


class TimedTransformAccess(Transform):
    def __init__(
        self,
        parent: Transform,
        *,
        stage: str,
        loader_name: str,
        source_name: str,
        verbosity: int = 0,
    ) -> None:
        super().__init__(parent)
        self._timing_enabled = _diagnostic_logging_enabled()
        self.stage = stage
        self.loader_name = loader_name
        self.source_name = source_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbosity > 0 else logging.INFO)
        self._timing_stats = {}
        self._timing_log_interval = 512
        self._timing_slow_call_threshold_s = 0.05

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if not self._timing_enabled:
            return self.parent.get_datapoint(item)
        start = time.perf_counter()
        out = self.parent.get_datapoint(item)
        _record_timing_stat(
            self.logger,
            self._timing_stats,
            stage=self.stage,
            item=item,
            elapsed_seconds=time.perf_counter() - start,
            log_interval=self._timing_log_interval,
            slow_call_threshold_s=self._timing_slow_call_threshold_s,
            extra=f"loader={self.loader_name} source={self.source_name}",
        )
        return out


class TimedDatasetAccess:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        stage: str,
        loader_name: str,
        source_name: str,
        verbosity: int = 0,
    ) -> None:
        self._timing_enabled = _diagnostic_logging_enabled()
        self.dataset = dataset
        self.stage = stage
        self.loader_name = loader_name
        self.source_name = source_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbosity > 0 else logging.INFO)
        self._timing_stats = {}
        self._timing_log_interval = 512
        self._timing_slow_call_threshold_s = 0.05

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if not self._timing_enabled:
            return self.dataset[item]
        start = time.perf_counter()
        out = self.dataset[item]
        _record_timing_stat(
            self.logger,
            self._timing_stats,
            stage=self.stage,
            item=item,
            elapsed_seconds=time.perf_counter() - start,
            log_interval=self._timing_log_interval,
            slow_call_threshold_s=self._timing_slow_call_threshold_s,
            extra=f"loader={self.loader_name} source={self.source_name}",
        )
        return out

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def seq_len(self) -> Union[int, List[int], None]:
        return self.dataset.seq_len

    @property
    def num_features(self) -> Union[int, Tuple[int, ...], None]:
        return self.dataset.num_features

    @property
    def ndim(self) -> Optional[int]:
        return self.dataset.ndim

    @property
    def window_size(self) -> Optional[int]:
        return self.dataset.window_size

    def __getattr__(self, name: str):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)


class TimedCollate:
    def __init__(
        self,
        collate: Callable,
        *,
        loader_name: str,
        verbosity: int = 0,
    ) -> None:
        self._timing_enabled = _diagnostic_logging_enabled()
        self.collate = collate
        self.loader_name = loader_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbosity > 0 else logging.INFO)
        self._timing_stats = {}
        self._call_count = 0
        self._timing_log_interval = 64
        self._timing_slow_call_threshold_s = 0.05

    def __call__(self, batch):
        if not self._timing_enabled:
            return self.collate(batch)
        batch_index = self._call_count
        self._call_count += 1
        start = time.perf_counter()
        out = self.collate(batch)
        _record_timing_stat(
            self.logger,
            self._timing_stats,
            stage="collate",
            item=batch_index,
            elapsed_seconds=time.perf_counter() - start,
            log_interval=self._timing_log_interval,
            slow_call_threshold_s=self._timing_slow_call_threshold_s,
            extra=f"loader={self.loader_name} batch_size={len(batch)}",
        )
        return out


def get_dataloader(data: Transform, pipeline: Optional[TransformCallable], batch_dim: int, shuffle: bool,
                   batch_size: int = 64, num_workers: int = 0, drop_last: bool = False,
                   persistent_workers: Optional[bool] = None,
                   extra_data: Optional[Transform] = None,
                   extra_pipeline: Optional[TransformCallable] = None,
                   data_source: Literal["real", "synthetic", "both"] = "real",
                   gen_to_org_factor: float = -1,
                   remove_window_dim: bool = False,
                   loader_name: str = "loader") -> torch.utils.data.DataLoader:
    """Create a dataloader that applies the configured transform pipeline.

    Args:
        data (Transform): Dataset transform to wrap.
        pipeline (Optional[TransformCallable]): Callable that builds a pipeline.
        batch_dim (int): Dimension treated as the batch axis.
        shuffle (bool): Whether to shuffle batches.
        batch_size (int): Number of sequences per batch. Defaults to 64.
        num_workers (int): Worker processes for loading. Defaults to 0.
        drop_last (bool): Whether to drop the final partial batch. Defaults to False.
        persistent_workers (Optional[bool]): Whether to keep workers alive. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: Configured dataloader instance.

    Side Effects:
        Creates dataset wrapper transforms and a dataloader.
    """
    diagnostic_logging_enabled = _diagnostic_logging_enabled()
    pipes = []
    assert data is not None
    if pipeline is not None:
        pipe = (pipeline(data))
    else:
        pipe = data
    if diagnostic_logging_enabled:
        pipe = TimedTransformAccess(
            pipe,
            stage="pipeline_transform_get",
            loader_name=loader_name,
            source_name="real",
            verbosity=1,
        )
    pipe = PipelineDataset(pipe)
    if diagnostic_logging_enabled:
        pipe = TimedDatasetAccess(
            pipe,
            stage="pipeline_dataset_get",
            loader_name=loader_name,
            source_name="real",
            verbosity=1,
        )
    use_primary_dataset = data_source != "synthetic" or extra_data is None
    if use_primary_dataset:
        pipes.append(pipe)

    if data_source != "real":
        if extra_data is None:
            if data_source == "both":
                raise ValueError(
                    "data_source='both' requires extra_data so the primary and auxiliary datasets can be concatenated."
                )
        else:
            if extra_pipeline is not None:
                extra_pipe = (extra_pipeline(extra_data))
            else:
                extra_pipe = extra_data
            if remove_window_dim:
                extra_pipe = WindowLastElement(extra_pipe)
            if diagnostic_logging_enabled:
                extra_pipe = TimedTransformAccess(
                    extra_pipe,
                    stage="pipeline_transform_get",
                    loader_name=loader_name,
                    source_name="synthetic",
                    verbosity=1,
                )
            extra_pipe = PipelineDataset(extra_pipe)
            if diagnostic_logging_enabled:
                extra_pipe = TimedDatasetAccess(
                    extra_pipe,
                    stage="pipeline_dataset_get",
                    loader_name=loader_name,
                    source_name="synthetic",
                    verbosity=1,
                )
            pipes.append(extra_pipe)
    pipe = ConcatDataset(pipes)

    persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers
    multiprocessing_context = multiprocessing.get_context("forkserver") if num_workers > 0 else None
    if drop_last and len(pipe) // batch_size < 5:
        drop_last = False

    collate: Callable = collate_fn(batch_dim=batch_dim)
    if diagnostic_logging_enabled:
        collate = TimedCollate(
            collate,
            loader_name=loader_name,
            verbosity=1,
        )

    return torch.utils.data.DataLoader(
        pipe,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        pin_memory=True,
        shuffle=shuffle,
        multiprocessing_context=multiprocessing_context,
        worker_init_fn=_init_dataloader_worker if num_workers > 0 else None,
    )
