"""Shared MLflow lineage helpers for benchmark orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import pandas as pd
import mlflow
from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from ..data_prep.pipeline import resolve_base_dataset_name


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControllerRunContext:
    experiment_id: str
    run_id: str
    run_name: str


def dataset_variant_label(requested_dataset_name: str) -> str:
    if requested_dataset_name.endswith("_tsst"):
        return "tsst"
    if requested_dataset_name.endswith("_ext"):
        return "ext"
    if requested_dataset_name.endswith("_red"):
        return "red"
    return "base"


def dataset_base_name(requested_dataset_name: str) -> str:
    normalized_name = requested_dataset_name.replace("_tsst", "")
    return resolve_base_dataset_name(normalized_name)


def build_benchmark_tags(
    *,
    timestamp: str,
    run_level: str,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    requested_dataset_name: Optional[str] = None,
    run_mode: Optional[str] = None,
    deployment_mode: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    seed: Optional[int] = None,
    stage: Optional[str] = None,
    source_experiment_id: Optional[str] = None,
    source_study_run_id: Optional[str] = None,
    selected_seed: Optional[int] = None,
    selected_stage: Optional[str] = None,
    extra_tags: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    tags: Dict[str, str] = {
        "run_level": run_level,
        "timestamp": timestamp,
    }

    effective_requested_dataset = requested_dataset_name or dataset_name
    if dataset_name is not None:
        tags["dataset"] = dataset_name
    if effective_requested_dataset is not None:
        tags["dataset_requested"] = effective_requested_dataset
        tags["dataset_base"] = dataset_base_name(effective_requested_dataset)
        tags["dataset_variant"] = dataset_variant_label(effective_requested_dataset)
    if model_name is not None:
        tags["model"] = model_name
    if run_mode is not None:
        tags["run_mode"] = run_mode
    if deployment_mode is not None:
        tags["deployment_mode"] = deployment_mode
    if parent_run_id is not None:
        tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
    if seed is not None:
        tags["seed"] = str(seed)
    if stage is not None:
        tags["stage"] = stage
    if source_experiment_id is not None:
        tags["source_experiment_id"] = source_experiment_id
    if source_study_run_id is not None:
        tags["source_study_run_id"] = source_study_run_id
    if selected_seed is not None:
        tags["selected_seed"] = str(selected_seed)
    if selected_stage is not None:
        tags["selected_stage"] = selected_stage
    if extra_tags is not None:
        for key, value in extra_tags.items():
            if value is not None:
                tags[str(key)] = str(value)
    return tags


def start_controller_run(
    client: MlflowClient,
    experiment_id: str,
    *,
    timestamp: str,
    tune_mode: bool,
    deployment_mode: str,
    datasets: Sequence[str],
    models: Sequence[str],
) -> ControllerRunContext:
    run_mode = "tune" if tune_mode else "single_train"
    run_name = f"controller__{timestamp}"
    controller_run = client.create_run(
        experiment_id,
        tags=build_benchmark_tags(
            timestamp=timestamp,
            run_level="controller",
            run_mode=run_mode,
            deployment_mode=deployment_mode,
            extra_tags={
                "datasets": json.dumps(sorted(datasets)),
                "models": json.dumps(sorted(models)),
            },
        ),
        run_name=run_name,
    )
    return ControllerRunContext(
        experiment_id=str(experiment_id),
        run_id=controller_run.info.run_id,
        run_name=run_name,
    )


def log_code_snapshot(
    client: MlflowClient,
    run_id: str,
    *,
    storage_path: str,
) -> str:
    project_root = Path(__file__).resolve().parents[5]
    archive_path = shutil.make_archive(
        base_name=str(Path(storage_path) / "source_code"),
        format="gztar",
        root_dir=project_root,
    )
    client.log_artifact(run_id, archive_path, artifact_path="code")
    return archive_path


def log_dependency_manifest(
    client: MlflowClient,
    run_id: str,
    *,
    manifest_path: str,
) -> None:
    client.log_artifact(run_id, manifest_path, artifact_path="metadata")


def log_dict_artifact(
    client: MlflowClient,
    run_id: str,
    *,
    artifact_file: str,
    payload: Mapping[str, Any],
) -> None:
    client.log_dict(run_id, dict(payload), artifact_file=artifact_file)


def log_table_artifact(
    client: MlflowClient,
    run_id: str,
    *,
    artifact_file: str,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    frame = pd.DataFrame(list(rows))
    client.log_table(run_id, frame, artifact_file=artifact_file)


def _pair_result_as_mapping(pair_result: Any) -> Mapping[str, Any]:
    if isinstance(pair_result, Mapping):
        return pair_result

    model_dump = getattr(pair_result, "model_dump", None)
    if callable(model_dump):
        dumped_pair_result = model_dump()
        if isinstance(dumped_pair_result, Mapping):
            return dumped_pair_result

    return {}


def build_pair_summary_rows(pair_results: Sequence[Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for pair_result in pair_results:
        pair_result_data = _pair_result_as_mapping(pair_result)
        result_payload = pair_result_data.get("result")
        row: Dict[str, Any] = {
            "model": pair_result_data.get("model_name"),
            "dataset": pair_result_data.get("dataset_name"),
            "study_run_id": pair_result_data.get("study_run_id"),
            "storage_path": pair_result_data.get("storage_path"),
            "status": pair_result_data.get("status"),
            "partial_result": pair_result_data.get("partial_result"),
            "job_status_message": pair_result_data.get("job_status_message"),
            "result_source": pair_result_data.get("result_source"),
            "seafile_synced": pair_result_data.get("seafile_synced"),
            "cleanup_performed": pair_result_data.get("cleanup_performed"),
        }
        if isinstance(result_payload, Mapping):
            for key, value in result_payload.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[str(key)] = value
            for key in ("hpo_seeds", "final_eval_seeds", "full_seeds"):
                value = result_payload.get(key)
                if value is not None and key not in row:
                    row[key] = json.dumps(value)
        rows.append(row)
    return rows


@contextmanager
def maybe_active_run(
    *,
    run_id: str,
    enable_system_metrics: bool,
) -> Iterator[None]:
    if not enable_system_metrics:
        yield
        return

    try:
        active_run = mlflow.start_run(run_id=run_id, log_system_metrics=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to start MLflow active run with system metrics for run '%s': %s. Continuing without system metrics.",
            run_id,
            exc,
        )
        yield
        return

    with active_run:
        yield
