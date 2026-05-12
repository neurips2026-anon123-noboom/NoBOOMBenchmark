"""MLflow dataset lineage helpers for benchmark runs."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Mapping, Optional

from mlflow import MlflowClient
from mlflow.entities import DatasetInput, InputTag
from mlflow.data.evaluation_dataset import EvaluationDataset

from ..benchmark_utils.mlflow_dataset import combine_datasets, from_datasource


logger = logging.getLogger(__name__)

MLFLOW_DATASET_CONTEXT_TAG = "mlflow.data.context"


@dataclass(frozen=True)
class LoggedDatasetInfo:
    context: str
    split_name: str
    data_source: str
    name: str
    digest: str
    profile: Dict[str, Any]
    dataset: Any
    evaluation_dataset: EvaluationDataset


class DatasetLineageCache:
    def __init__(self) -> None:
        self._entries: Dict[str, LoggedDatasetInfo] = {}

    def get(self, key: str) -> Optional[LoggedDatasetInfo]:
        return self._entries.get(key)

    def set(self, key: str, value: LoggedDatasetInfo) -> LoggedDatasetInfo:
        self._entries[key] = value
        return value


def build_logged_dataset_name(
    requested_dataset_name: str,
    split_name: str,
    data_source: str,
) -> str:
    return f"{requested_dataset_name}__{split_name}__{data_source}"


def build_logged_dataset_source(
    *,
    base_source_uri: Optional[str],
    requested_dataset_name: str,
    split_name: str,
    data_source: str,
) -> str:
    if data_source == "real" and base_source_uri:
        return f"{base_source_uri}#{requested_dataset_name}:{split_name}"
    if data_source == "synthetic":
        return f"generated://{requested_dataset_name}/{split_name}"
    if data_source == "both":
        return f"derived://{requested_dataset_name}/{split_name}/combined"
    return f"code://{requested_dataset_name}/{split_name}"


def datamodule_split_dataset(datamodule: Any, split_name: str) -> Any:
    split_mapping = {
        "train": ("train_data", "extra_train_data"),
        "val": ("val_data", "extra_val_data"),
        "test": ("test_data", "extra_test_data"),
        "predict": ("test_data", "extra_test_data"),
    }
    primary_attr, extra_attr = split_mapping[split_name]
    primary_dataset = getattr(datamodule, primary_attr, None)
    extra_dataset = getattr(datamodule, extra_attr, None)
    if primary_dataset is not None and extra_dataset is not None:
        return combine_datasets(primary_dataset, extra_dataset)
    return primary_dataset or extra_dataset


def get_or_create_logged_dataset(
    cache: DatasetLineageCache,
    *,
    cache_key: str,
    dataset_source: Any,
    requested_dataset_name: str,
    split_name: str,
    data_source: str,
    source_uri: Optional[str],
    context: str,
) -> LoggedDatasetInfo:
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    dataset_name = build_logged_dataset_name(requested_dataset_name, split_name, data_source)
    mlflow_dataset = from_datasource(
        dataset_source,
        source=build_logged_dataset_source(
            base_source_uri=source_uri,
            requested_dataset_name=requested_dataset_name,
            split_name=split_name,
            data_source=data_source,
        ),
        name=dataset_name,
    )
    info = LoggedDatasetInfo(
        context=context,
        split_name=split_name,
        data_source=data_source,
        name=str(mlflow_dataset.name),
        digest=str(mlflow_dataset.digest),
        profile=dict(mlflow_dataset.profile or {}),
        dataset=mlflow_dataset,
        evaluation_dataset=mlflow_dataset.to_evaluation_dataset(),
    )
    return cache.set(cache_key, info)


def log_dataset_input(
    client: MlflowClient,
    run_id: str,
    dataset_info: LoggedDatasetInfo,
    *,
    tags: Optional[Mapping[str, str]] = None,
) -> None:
    input_tags = [InputTag(MLFLOW_DATASET_CONTEXT_TAG, dataset_info.context)]
    if tags is not None:
        input_tags.extend(InputTag(str(key), str(value)) for key, value in tags.items())
    dataset_input = DatasetInput(
        dataset=dataset_info.dataset._to_mlflow_entity(),
        tags=input_tags,
    )
    client.log_inputs(run_id=run_id, datasets=[dataset_input])


def datamodule_split_lineage(
    client: MlflowClient,
    cache: DatasetLineageCache,
    *,
    run_id: str,
    datamodule: Any,
    requested_dataset_name: str,
    split_name: str,
    context: str,
    source_uri: Optional[str],
    tags: Optional[Mapping[str, str]] = None,
) -> Optional[LoggedDatasetInfo]:
    dataset_source = datamodule_split_dataset(datamodule, split_name)
    if dataset_source is None:
        return None
    effective_source_uri = source_uri
    if effective_source_uri is None:
        manifest = getattr(datamodule, "_manifest", None)
        effective_source_uri = getattr(manifest, "source_uri", None)
    data_source = str(datamodule._loader_data_source(split_name))
    cache_key = json.dumps(
        {
            "run_id": run_id,
            "requested_dataset_name": requested_dataset_name,
            "split_name": split_name,
            "data_source": data_source,
            "context": context,
        },
        sort_keys=True,
    )
    dataset_info = get_or_create_logged_dataset(
        cache,
        cache_key=cache_key,
        dataset_source=dataset_source,
        requested_dataset_name=requested_dataset_name,
        split_name=split_name,
        data_source=data_source,
        source_uri=effective_source_uri,
        context=context,
    )
    log_dataset_input(client, run_id, dataset_info, tags=tags)
    logger.info(
        "Logged MLflow input dataset '%s' (split=%s, data_source=%s, digest=%s) to run '%s'.",
        dataset_info.name,
        split_name,
        data_source,
        dataset_info.digest,
        run_id,
    )
    return dataset_info
