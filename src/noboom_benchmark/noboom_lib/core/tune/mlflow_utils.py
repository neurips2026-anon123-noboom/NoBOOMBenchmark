"""MLflow helpers for tune orchestration."""
from __future__ import annotations

from collections import deque
import json
import logging
import math
from pathlib import Path
import shutil
import tempfile
from typing import Deque, Dict, List, Mapping, Optional, Tuple

import mlflow
from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

logger = logging.getLogger(__name__)


def log_code_snapshot(
    experiment_id: str, timestamp: str, storage_path: str | Path
) -> None:
    """Archive the current source tree and log it to MLflow.

    Args:
        experiment_id (str): MLflow experiment ID.
        timestamp (str): Timestamp for the run name.
        storage_path (str | Path): Directory where the archive is created.

    Returns:
        None: The archive is logged to MLflow.

    Side Effects:
        Creates a tar archive and logs it as an MLflow artifact.
    """
    project_root = Path(__file__).resolve().parents[2]
    logger.info(
        "Creating code snapshot of project root '%s' into '%s'.",
        project_root,
        storage_path,
    )

    archive_path = shutil.make_archive(
        base_name=str(Path(storage_path) / "source_code"),
        format="gztar",
        root_dir=project_root,
    )
    logger.info("Code archive created at '%s'. Logging to MLflow.", archive_path)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"code_archive__{timestamp}",
        tags={"type": "code_archive", "timestamp": timestamp},
    ):
        mlflow.log_artifact(archive_path, artifact_path="code")
        logger.info("Code archive logged as MLflow artifact.")


def log_dependency_manifest(
    experiment_id: str,
    timestamp: str,
    manifest_path: str,
    tracking_uri: Optional[str] = None,
) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    manifest_file = Path(manifest_path)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"dependency_manifest__{timestamp}",
        tags={"type": "dependency_manifest", "timestamp": timestamp},
    ):
        mlflow.log_artifact(str(manifest_file), artifact_path="metadata")
        logger.info("Dependency manifest logged from '%s'.", manifest_file)


def gather_runs(
    study_run_id: str,
    experiment_ids: Optional[List[str]] = None,
    sep: str = "/",
    max_results: int = 100000,
) -> Dict[str, str]:
    """Build a map of run IDs to hierarchical run paths.

    Args:
        study_run_id (str): MLflow study run ID to use as the root.
        experiment_ids (Optional[List[str]]): Experiment IDs to search. Defaults to None.
        sep (str): Path separator for run hierarchy. Defaults to "/".
        max_results (int): Maximum number of runs to fetch per query. Defaults to 100000.

    Returns:
        Dict[str, str]: Mapping of run_id to hierarchical path.

    Side Effects:
        Queries MLflow for run metadata.
    """
    study_run = mlflow.get_run(study_run_id)

    if experiment_ids is None:
        experiment_ids = [study_run.info.experiment_id]

    study_name = study_run.data.tags.get(MLFLOW_RUN_NAME) or study_run.info.run_id

    out: Dict[str, str] = {}
    q: Deque[Tuple[str, str]] = deque([(study_run_id, study_name)])

    parent_tag_col = f"tags.{MLFLOW_PARENT_RUN_ID}"
    run_name_col = f"tags.{MLFLOW_RUN_NAME}"

    while q:
        parent_id, parent_path = q.popleft()

        df = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"{parent_tag_col} = '{parent_id}'",
            output_format="pandas",
            max_results=max_results,
        )

        for _, row in df.iterrows():
            child_id = row["run_id"]
            child_name = row.get(run_name_col) or child_id
            child_path = f"{parent_path}{sep}{child_name}"

            out[child_id] = child_path
            q.append((child_id, child_path))

    return out


def _download_json_artifact_for_run(
    run_id: str,
    *,
    artifact_path: str,
) -> Optional[dict]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=tmp_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to download MLflow artifact '%s' for run_id=%s: %s",
                artifact_path,
                run_id,
                exc,
            )
            return None

        with open(local_path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def _metric_as_float(
    *,
    metrics: Mapping[str, object],
    metric_name: str,
) -> Optional[float]:
    metric_value = metrics.get(metric_name)
    if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
        return None

    resolved_metric = float(metric_value)
    if math.isnan(resolved_metric):
        return None
    return resolved_metric


def _load_ranked_trial_fallback(
    *,
    client: MlflowClient,
    experiment_id: str,
    study_run_id: str,
    min_completed_seeds: int,
) -> Optional[Tuple[dict, Optional[str]]]:
    trial_runs = client.search_runs(
        [experiment_id],
        filter_string=(
            "tags.run_level = 'trial' and "
            f"tags.{MLFLOW_PARENT_RUN_ID} = '{study_run_id}'"
        ),
        max_results=1000,
    )
    eligible_trials = []
    for trial_run in trial_runs:
        completed_seed_count = _metric_as_float(
            metrics=trial_run.data.metrics,
            metric_name="completed_seed_count",
        )
        alarm_score = _metric_as_float(
            metrics=trial_run.data.metrics,
            metric_name="alarm_score",
        )
        if completed_seed_count is None or completed_seed_count < float(min_completed_seeds):
            continue
        if alarm_score is None:
            continue

        aaf = _metric_as_float(
            metrics=trial_run.data.metrics,
            metric_name="aaf",
        )
        if aaf is None:
            aaf = float("inf")
        time_total_s = _metric_as_float(
            metrics=trial_run.data.metrics,
            metric_name="time_total_s",
        )
        if time_total_s is None:
            time_total_s = float("inf")

        eligible_trials.append(
            (
                -alarm_score,
                aaf,
                time_total_s,
                trial_run.info.run_id,
                trial_run,
            )
        )

    if not eligible_trials:
        return None

    for _rank_alarm, _rank_aaf, _rank_runtime, _rank_run_id, trial_run in sorted(eligible_trials):
        metadata_payload = _download_json_artifact_for_run(
            trial_run.info.run_id,
            artifact_path="metadata.json",
        )
        if not isinstance(metadata_payload, dict):
            continue
        if not isinstance(metadata_payload.get("hp_params"), dict):
            logger.warning(
                "Trial run %s was selected as a fallback candidate but its metadata.json had no hp_params.",
                trial_run.info.run_id,
            )
            continue
        logger.info(
            "Loaded source hyperparameters from fallback trial run %s under study run %s.",
            trial_run.info.run_id,
            study_run_id,
        )
        return metadata_payload, study_run_id

    return None


def load_json_artifact_from_mlflow(
    experiment_id: str,
    model_name: str,
    dataset_name: str,
    timestamp: str,
    artifact_path: str,
    tracking_uri: Optional[str] = None,
) -> Optional[Tuple[dict, str]]:
    """Download a JSON artifact from a study-level MLflow run.

    Args:
        experiment_id (str): MLflow experiment ID.
        model_name (str): Model name used to filter runs.
        dataset_name (str): Dataset name used to filter runs.
        timestamp (str): Timestamp used to filter runs.
        artifact_path (str): Artifact path to download.

    Returns:
        Tuple[dict, str] | None: JSON payload and run ID, or None if not found.

    Side Effects:
        Downloads artifacts from MLflow to a temporary directory.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    filter_string = (
        "tags.run_level = 'study' and "
        f"tags.model = '{model_name}' and "
        f"tags.dataset = '{dataset_name}' and "
        f"tags.timestamp = '{timestamp}'"
    )

    # mlflow.search_runs returns a pandas DataFrame
    df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=1,
        output_format="pandas",
    )

    if df.empty:
        logger.warning(
            "No MLflow study run found for model=%s, dataset=%s, timestamp=%s.",
            model_name,
            dataset_name,
            timestamp,
        )
        return None

    run_id = str(df.iloc[0]["run_id"])
    payload = _download_json_artifact_for_run(
        run_id,
        artifact_path=artifact_path,
    )
    if payload is None:
        return None
    return payload, run_id


def load_study_result_from_mlflow(
    experiment_id: str,
    model_name: str,
    dataset_name: str,
    artifact_path: str = "summary/result.json",
    tracking_uri: Optional[str] = None,
    min_completed_seeds: int = 2,
) -> Optional[Tuple[dict, Optional[str]]]:
    """Fetch the latest study-level result artifact for a model/dataset pair.

    Args:
        experiment_id (str): MLflow experiment ID.
        model_name (str): Model name used to filter runs.
        dataset_name (str): Dataset name used to filter runs.
        artifact_path (str): Artifact path to download. Defaults to summary/result.json.
        tracking_uri (Optional[str]): MLflow tracking URI override.
        min_completed_seeds (int): Minimum number of completed seeds required for
            a child trial to be considered as the fallback source.

    Returns:
        Tuple[dict, Optional[str]] | None: JSON payload and study run ID, or None
            if not found.

    Side Effects:
        Downloads artifacts from MLflow to a temporary directory and, when the
        study-level summary is missing, falls back to the best eligible child
        trial ranked by alarm_score desc, aaf asc, and time_total_s asc.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    filter_string = (
        "tags.run_level = 'study' and "
        f"tags.model = '{model_name}' and "
        f"tags.dataset = '{dataset_name}'"
    )

    df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=1000,
        output_format="pandas",
    )

    for run_id in df.get("run_id", []):
        resolved_run_id = str(run_id)
        payload = _download_json_artifact_for_run(
            resolved_run_id,
            artifact_path=artifact_path,
        )
        if payload is not None:
            return payload, resolved_run_id
        fallback_payload = _load_ranked_trial_fallback(
            client=client,
            experiment_id=experiment_id,
            study_run_id=resolved_run_id,
            min_completed_seeds=min_completed_seeds,
        )
        if fallback_payload is not None:
            return fallback_payload

    logger.warning(
        "No usable source study result found for model=%s, dataset=%s in experiment_id=%s.",
        model_name,
        dataset_name,
        experiment_id,
    )
    return None
