from functools import cached_property
import json
import logging
import struct
from typing import Any, Optional, Sequence, Tuple

from mlflow.data import DatasetSource as MLFlowDatasetSource
from mlflow.data.digest_utils import get_normalized_md5_digest
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.numpy_dataset import (Dataset, PyFuncConvertibleDatasetMixin,
                                       TensorDatasetSchema)
from mlflow.types.utils import _infer_schema
import numpy as np
import torch

from timesead.data.transforms import PipelineDataset

MAX_ROWS_TO_READ = 10000

logger = logging.getLogger(__name__)


class CombinedDatasetSource:
    """Present multiple dataset-like sources as one logical source."""

    def __init__(self, datasets: Sequence[Any]):
        self._datasets = [dataset for dataset in datasets if dataset is not None]
        if not self._datasets:
            raise ValueError("CombinedDatasetSource requires at least one dataset.")
        self._offsets = []
        total = 0
        for dataset in self._datasets:
            total += len(dataset)
            self._offsets.append(total)

    def __len__(self) -> int:
        return self._offsets[-1]

    def get_datapoint(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if item < 0 or item >= len(self):
            raise IndexError(item)
        start = 0
        for dataset, stop in zip(self._datasets, self._offsets):
            if item < stop:
                return dataset.get_datapoint(item - start)
            start = stop
        raise IndexError(item)

    @property
    def seq_len(self) -> list[int]:
        seq_lens: list[int] = []
        for dataset in self._datasets:
            dataset_seq_len = getattr(dataset, "seq_len", [])
            if isinstance(dataset_seq_len, int):
                seq_lens.extend([dataset_seq_len] * len(dataset))
            else:
                seq_lens.extend(list(dataset_seq_len))
        return seq_lens

    @property
    def num_features(self) -> int:
        return int(getattr(self._datasets[0], "num_features"))

    @property
    def window_size(self) -> Optional[int]:
        return getattr(self._datasets[0], "window_size", None)


class MLFlowDataSourceDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """Represents a dataset source for use with MLflow Tracking."""

    def __init__(
        self,
        dataset: Any,
        source: MLFlowDatasetSource,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """Initialize the MLflow dataset wrapper.

        Args:
            dataset (DatasetSource): Underlying dataset source.
            source (MLFlowDatasetSource): MLflow dataset source metadata.
            name (str | None): Dataset name. Defaults to None.
            digest (str | None): Optional digest hash. Defaults to None.
        """
        self._ds = dataset
        super().__init__(source=source, name=name, digest=digest)

    def _compute_dataset_digest(
        self,
        dataset: Any,
    ) -> str:
        """Compute a digest for the given pipeline dataset.

        Args:
            dataset (PipelineDataset): Pipeline dataset to hash.

        Returns:
            str: Digest string.
        """

        hashable_elements = []

        def hash_dataset_element(element: torch.Tensor):
            """Hash a single tensor element into the digest buffer.

            Args:
                element (torch.Tensor): Tensor to hash.

            Returns:
                None: Updates the hashable element list in place.
            """
            if element is None:
                return
            if element.numel() == 0:
                return
            logger.debug(f"hash_dataset_element({element.shape})")
            hash_val = torch.hash_tensor(element).item()
            hashable_elements.append(struct.pack(">Q", hash_val & ((1<<64)-1)))
        for inputs, targets in dataset:
            hash_dataset_element(inputs[0])
            hash_dataset_element(targets[0])

        return get_normalized_md5_digest(hashable_elements)


    def _compute_digest(self) -> str:
        """Compute a digest for the dataset when not provided.

        Returns:
            str: Dataset digest.
        """
        return self._compute_dataset_digest(self.ds)

    def to_dict(self) -> dict[str, str]:
        """Create config dictionary for the dataset.

        Returns:
            dict[str, str]: Config containing name, digest, source, schema, and profile.
        """
        schema = json.dumps(self.schema.to_dict()) if self.schema else None
        config = super().to_dict()
        config.update(
            {
                "schema": schema,
                "profile": json.dumps(self.profile),
            }
        )

        return config

    @property
    def source(self) -> MLFlowDatasetSource:
        """Return the MLflow dataset source.

        Returns:
            MLFlowDatasetSource: MLflow dataset source metadata.
        """
        return self._source

    @property
    def ds_source(self) -> Any:
        """Return the underlying dataset source.

        Returns:
            DatasetSource: Underlying dataset source.
        """
        return self._ds

    @property
    def ds(self) -> PipelineDataset:
        """Wrap the dataset source in a PipelineDataset.

        Returns:
            PipelineDataset: Pipeline dataset wrapper.
        """
        return PipelineDataset(self._ds)

    @property
    def profile(self) -> Optional[Any]:
        """Return a lightweight profile for the dataset.

        Returns:
            Any | None: Profile mapping or None if unavailable.
        """

        ds = self.ds

        profile = {
            "num_features": ds.num_features,
            "seq_len": ds.seq_len,
            "num_samples": len(ds)
        }

        return profile

    def _get_data_chunk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load a chunk of data for evaluation purposes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature and target arrays.
        """
        inputs = []
        targets = []
        total_len = 0
        for input, target in self.ds:
            input, target = input[0], target[0]
            if input.numel() == 0:
                continue
            rows_to_read = min(input.shape[0], max(0, MAX_ROWS_TO_READ - total_len))
            if rows_to_read > 0:
                inputs.append(input[:rows_to_read].numpy())
                targets.append(target[:rows_to_read].numpy())
            else:
                break
        inputs = np.concatenate(inputs)
        targets = np.concatenate(targets)
        return inputs, targets

    @cached_property
    def schema(self) -> Optional[TensorDatasetSchema]:
        """Infer the MLflow TensorSpec schema for features and targets.

        Returns:
            TensorDatasetSchema | None: Inferred schema, or None on failure.
        """
        try:
            inputs = []
            targets = []
            total_len = 0
            for input, target in self.ds:
                input, target = input[0], target[0]
                if input.numel() == 0:
                    continue
                rows_to_read = min(input.shape[0], max(0, MAX_ROWS_TO_READ - total_len))
                if rows_to_read > 0:
                    inputs.append(input[:rows_to_read].numpy())
                    targets.append(target[:rows_to_read].numpy())
                else:
                    break
            inputs = np.concatenate(inputs)
            targets = np.concatenate(targets)
            features_schema = _infer_schema(inputs)
            targets_schema = _infer_schema(targets)
            return TensorDatasetSchema(features=features_schema, targets=targets_schema)
        except Exception as exc:
            logger.warning("Failed to infer schema for MLFlowDataSourceDataset. Exception: %s", exc)
            return None


    def to_evaluation_dataset(self, path: Optional[str] = None, feature_names=None) -> EvaluationDataset:
        """Convert to an EvaluationDataset for MLflow model evaluation.

        Args:
            path (Any | None): Optional dataset path. Defaults to None.
            feature_names (Any | None): Optional feature names. Defaults to None.

        Returns:
            EvaluationDataset: Evaluation dataset wrapper.
        """
        inputs, targets = self._get_data_chunk()
        return EvaluationDataset(
            data=inputs,
            targets=targets,
            path=path,
            feature_names=feature_names,
            name=self.name,
            digest=self.digest,
        )


def from_datasource(
    dataset: Any,
    source: Optional[str | MLFlowDatasetSource] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> MLFlowDataSourceDataset:
    """Create an MLFlowDataSourceDataset from a dataset source.

    Args:
        dataset (DatasetSource): Dataset source to wrap.
        source (str | MLFlowDatasetSource | None): MLflow dataset source or URI.
            Defaults to None, which uses a code dataset source.
        name (str | None): Optional dataset name. Defaults to None.
        digest (str | None): Optional dataset digest. Defaults to None.

    Returns:
        MLFlowDataSourceDataset: Wrapped dataset instance.
    """

    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.exceptions import MlflowException
    from mlflow.tracking.context import registry

    if source is not None:
        if isinstance(source, MLFlowDatasetSource):
            resolved_source = source
        else:
            try:
                resolved_source = resolve_dataset_source(
                    source,
                )
            except MlflowException:
                # MLflow only resolves built-in dataset source schemes. Preserve
                # custom lineage URIs such as generated:// and derived:// as tags
                # on a code-backed source instead of failing task execution.
                context_tags = dict(registry.resolve_tags() or {})
                context_tags["noboom.dataset_source_uri"] = str(source)
                logger.warning(
                    "MLflow could not resolve dataset source '%s'; "
                    "falling back to CodeDatasetSource tags.",
                    source,
                )
                resolved_source = CodeDatasetSource(tags=context_tags)
    else:
        context_tags = registry.resolve_tags()
        resolved_source = CodeDatasetSource(tags=context_tags)
    return MLFlowDataSourceDataset(
        dataset=dataset, source=resolved_source, name=name, digest=digest
    )


def combine_datasets(*datasets: Any) -> CombinedDatasetSource:
    return CombinedDatasetSource(datasets)
