"""Arrow IPC + mmap runtime datasets for benchmark training and evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from timesead.data.transforms import DatasetSource

from ..data_prep.manifest import PreparedDatasetManifest


class SequenceTensorDataset:
    """In-memory sequence dataset used for inference overrides and caches."""

    def __init__(
        self,
        samples: Sequence[Union[np.ndarray, torch.Tensor]],
        targets: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
    ) -> None:
        self._samples = [self._to_numpy(sample, dtype=np.float32) for sample in samples]
        if not self._samples:
            raise ValueError("SequenceTensorDataset requires at least one sample.")
        if targets is None:
            self._targets = [np.zeros(sample.shape[0], dtype=np.int64) for sample in self._samples]
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
        return [int(sample.shape[0]) for sample in self._samples]

    @property
    def num_features(self) -> int:
        return int(self._samples[0].shape[-1])

    @property
    def ndim(self) -> int:
        return int(self._samples[0].ndim)

    @property
    def window_size(self) -> Optional[int]:
        return None


def _load_primitive_arrow_column(
    path: str,
    *,
    expected_column_name: str,
) -> Tuple[pa.MemoryMappedFile, List[np.ndarray]]:
    source = pa.memory_map(path, "r")
    reader = ipc.open_file(source)
    arrays: List[np.ndarray] = []
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        if batch.num_columns != 1:
            raise ValueError(f"Expected exactly one column in '{path}', found {batch.num_columns}.")
        if batch.schema.names[0] != expected_column_name:
            raise ValueError(
                f"Unexpected Arrow column name in '{path}': expected '{expected_column_name}', "
                f"found '{batch.schema.names[0]}'."
            )
        arrays.append(batch.column(0).to_numpy(zero_copy_only=True))
    if not arrays:
        raise ValueError(f"Arrow file '{path}' does not contain any record batches.")
    return source, arrays


def _resolve_sequence_batches(
    *,
    seq_len: np.ndarray,
    num_features: int,
    feature_batches: Sequence[np.ndarray],
    label_batches: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(feature_batches) != len(label_batches):
        raise ValueError(
            "Feature and label Arrow files contain different numbers of record batches: "
            f"{len(feature_batches)} vs {len(label_batches)}."
        )

    seq_batch_index = np.zeros(len(seq_len), dtype=np.int64)
    seq_local_feature_start = np.zeros(len(seq_len), dtype=np.int64)
    seq_local_label_start = np.zeros(len(seq_len), dtype=np.int64)

    if len(seq_len) == 0:
        for batch in feature_batches:
            if len(batch) != 0:
                raise ValueError("Feature Arrow batches must be empty when there are no sequences.")
        for batch in label_batches:
            if len(batch) != 0:
                raise ValueError("Label Arrow batches must be empty when there are no sequences.")
        return seq_batch_index, seq_local_feature_start, seq_local_label_start

    sequence_cursor = 0
    for batch_index, (feature_batch, label_batch) in enumerate(zip(feature_batches, label_batches)):
        feature_cursor = 0
        label_cursor = 0
        batch_sequence_count = 0
        while sequence_cursor < len(seq_len):
            current_seq_len = int(seq_len[sequence_cursor])
            feature_width = current_seq_len * int(num_features)
            label_width = current_seq_len
            if feature_cursor + feature_width > len(feature_batch) or label_cursor + label_width > len(label_batch):
                break
            seq_batch_index[sequence_cursor] = batch_index
            seq_local_feature_start[sequence_cursor] = feature_cursor
            seq_local_label_start[sequence_cursor] = label_cursor
            feature_cursor += feature_width
            label_cursor += label_width
            sequence_cursor += 1
            batch_sequence_count += 1
        if feature_cursor != len(feature_batch) or label_cursor != len(label_batch):
            raise ValueError(
                f"Arrow batch boundary mismatch for batch {batch_index}: "
                f"consumed feature values {feature_cursor}/{len(feature_batch)} and "
                f"label values {label_cursor}/{len(label_batch)}."
            )
        if batch_sequence_count == 0 and (len(feature_batch) > 0 or len(label_batch) > 0):
            raise ValueError(f"Arrow batch {batch_index} contains values but no complete sequences.")

    if sequence_cursor != len(seq_len):
        raise ValueError(
            f"Arrow batch coverage mismatch: resolved {sequence_cursor} sequences, expected {len(seq_len)}."
        )
    return seq_batch_index, seq_local_feature_start, seq_local_label_start


class ArrowSequenceDataset:
    """Map-style dataset that reads canonical sequences from memory-mapped Arrow IPC arrays."""

    def __init__(
        self,
        *,
        manifest: PreparedDatasetManifest,
        mode: str,
    ) -> None:
        if mode not in manifest.modes:
            raise KeyError(f"Prepared manifest does not contain mode '{mode}'.")
        self._manifest = manifest
        self._mode = mode
        self._mode_manifest = manifest.modes[mode]
        self._seq_len = np.asarray(self._mode_manifest.seq_len, dtype=np.int64)
        self._feature_source: Optional[pa.MemoryMappedFile] = None
        self._label_source: Optional[pa.MemoryMappedFile] = None
        self._feature_batches: Optional[List[np.ndarray]] = None
        self._label_batches: Optional[List[np.ndarray]] = None
        self._seq_batch_index: Optional[np.ndarray] = None
        self._feature_starts: Optional[np.ndarray] = None
        self._label_starts: Optional[np.ndarray] = None
        self._open_arrow_buffers()

    def _close_arrow_buffers(self) -> None:
        for source_name in ("_feature_source", "_label_source"):
            source = getattr(self, source_name, None)
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
                finally:
                    setattr(self, source_name, None)

    def _open_arrow_buffers(self) -> None:
        if self._feature_batches is not None and self._label_batches is not None:
            return

        self._close_arrow_buffers()
        feature_source, feature_batches = _load_primitive_arrow_column(
            self._mode_manifest.feature_values_path,
            expected_column_name="value",
        )
        label_source, label_batches = _load_primitive_arrow_column(
            self._mode_manifest.label_values_path,
            expected_column_name="value",
        )
        (
            seq_batch_index,
            feature_starts,
            label_starts,
        ) = _resolve_sequence_batches(
            seq_len=self._seq_len,
            num_features=int(self._mode_manifest.num_features),
            feature_batches=feature_batches,
            label_batches=label_batches,
        )
        expected_feature_values = int(self._seq_len.sum()) * int(self._mode_manifest.num_features)
        expected_label_values = int(self._seq_len.sum())
        feature_value_count = sum(len(batch) for batch in feature_batches)
        label_value_count = sum(len(batch) for batch in label_batches)
        if feature_value_count != expected_feature_values:
            raise ValueError(
                f"Arrow feature buffer length mismatch for mode '{self._mode}': "
                f"expected {expected_feature_values}, found {feature_value_count}."
            )
        if label_value_count != expected_label_values:
            raise ValueError(
                f"Arrow label buffer length mismatch for mode '{self._mode}': "
                f"expected {expected_label_values}, found {label_value_count}."
            )

        self._feature_source = feature_source
        self._label_source = label_source
        self._feature_batches = feature_batches
        self._label_batches = label_batches
        self._seq_batch_index = seq_batch_index
        self._feature_starts = feature_starts
        self._label_starts = label_starts

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        self._open_arrow_buffers()
        if self._feature_batches is None or self._label_batches is None:
            raise RuntimeError(f"Arrow buffers for mode '{self._mode}' are not available.")
        if self._seq_batch_index is None or self._feature_starts is None or self._label_starts is None:
            raise RuntimeError(f"Arrow index buffers for mode '{self._mode}' are not available.")
        seq_len = int(self._seq_len[item])
        batch_index = int(self._seq_batch_index[item])
        feature_start = int(self._feature_starts[item])
        label_start = int(self._label_starts[item])
        feature_stop = feature_start + seq_len * int(self._mode_manifest.num_features)
        label_stop = label_start + seq_len
        feature_batch = self._feature_batches[batch_index]
        label_batch = self._label_batches[batch_index]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable.*",
                category=UserWarning,
            )
            features = torch.from_numpy(
                feature_batch[feature_start:feature_stop].reshape(
                    seq_len,
                    int(self._mode_manifest.num_features),
                )
            )
            labels = torch.from_numpy(label_batch[label_start:label_stop])
        return (features,), (labels,)

    def __len__(self) -> int:
        return int(self._mode_manifest.item_count)

    @property
    def seq_len(self) -> List[int]:
        return list(self._mode_manifest.seq_len)

    @property
    def num_features(self) -> int:
        return int(self._mode_manifest.num_features)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def window_size(self) -> Optional[int]:
        return None

    @property
    def source_name(self) -> str:
        return str(self._mode_manifest.source_name)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_feature_source"] = None
        state["_label_source"] = None
        state["_feature_batches"] = None
        state["_label_batches"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        self._close_arrow_buffers()


def build_dataset_source(
    manifest: PreparedDatasetManifest,
    *,
    mode: str,
) -> DatasetSource:
    """Build a ``DatasetSource`` from one Arrow IPC manifest mode."""

    return DatasetSource(ArrowSequenceDataset(manifest=manifest, mode=mode))
