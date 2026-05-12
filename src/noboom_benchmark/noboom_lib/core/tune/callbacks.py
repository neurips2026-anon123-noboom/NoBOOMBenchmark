from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from filelock import FileLock
import mlflow
from omegaconf import OmegaConf
from tenacity import retry, stop_after_attempt, wait_fixed

from noboom_cluster.noboom_cli_lib.specs import PairResult, PairRunSpec

from ..config.runtime import RuntimeConfig
from .interfaces import PairOutputCallback
from .mlflow_utils import gather_runs, load_json_artifact_from_mlflow
from .report_generation import excel_score_table_has_metrics, update_score_excel_one
from .storage import delete_by_extension, upload_to_seafile


EXCEL_PATH = Path("noboom_experiments.xlsx")
EXCEL_SHEET = "scores"
RUNTIME_METRIC = "time_total_s"

logger = logging.getLogger(__name__)


class MlflowSeafilePairOutputCallback(PairOutputCallback):
    def __init__(
        self,
        *,
        experiment_root: Path,
        experiment_name: str,
        runtime_config: RuntimeConfig,
        config_dir: str,
        temp_dir: str,
        tracking_uri: str,
    ) -> None:
        self._experiment_root = experiment_root
        self._experiment_name = experiment_name
        self._runtime_config = runtime_config
        self._config_dir = config_dir
        self._temp_dir = temp_dir
        self._tracking_uri = tracking_uri

    def _should_sync_to_seafile(self) -> bool:
        if str(getattr(self._runtime_config, "artifact_storage_backend", "remote")) == "local":
            return False
        if str(getattr(self._runtime_config, "execution_backend", "ray")) == "local":
            return False
        if not bool(getattr(self._runtime_config, "seafile_upload_results", False)):
            return False
        return bool(
            getattr(self._runtime_config, "seafile_username", None)
            and getattr(self._runtime_config, "seafile_root_path", None)
        )

    def on_job_terminal(
        self,
        pair_spec: PairRunSpec,
        status: str,
        job_status_message: Optional[str] = None,
    ) -> PairResult:
        payload = self._load_result(pair_spec)
        status_upper = str(status).upper()
        result = dict(payload.result)
        study_run_id = payload.study_run_id
        excel_path = self._experiment_root / EXCEL_PATH

        self._update_excel_snapshot(
            pair_spec,
            result,
            excel_path=excel_path,
            status=status_upper,
            allow_empty_row=status_upper != "SUCCEEDED",
        )

        pair_result = PairResult(
            model_name=pair_spec.model_name,
            dataset_name=pair_spec.dataset_name,
            storage_path=pair_spec.storage_path,
            result=result,
            study_run_id=study_run_id,
            status=status_upper,
            partial_result=status_upper != "SUCCEEDED",
            job_status_message=job_status_message,
            result_source=self._resolve_result_source(
                payload.result_source,
                result=result,
                status=status_upper,
            ),
        )

        if self._should_sync_to_seafile():
            self._sync_pair_outputs(
                pair_spec,
                study_run_id,
                excel_path,
                include_checkpoints=status_upper == "SUCCEEDED"
                and bool(getattr(self._runtime_config, "seafile_upload_checkpoints", False)),
                include_run_artifacts=status_upper == "SUCCEEDED",
            )
            pair_result.seafile_synced = True
            if status_upper == "SUCCEEDED":
                self._cleanup_pair_artifacts(pair_spec, study_run_id)
                pair_result.cleanup_performed = True

        return pair_result

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def _load_result(self, pair_spec: PairRunSpec) -> PairResult:
        mlflow_result = load_json_artifact_from_mlflow(
            pair_spec.experiment_id,
            pair_spec.model_name,
            pair_spec.dataset_name,
            pair_spec.timestamp,
            "summary/result.json",
            tracking_uri=self._tracking_uri,
        )

        result = {}
        study_run_id = None
        if mlflow_result:
            result, study_run_id = mlflow_result
            return PairResult(
                model_name=pair_spec.model_name,
                dataset_name=pair_spec.dataset_name,
                storage_path=pair_spec.storage_path,
                result=result,
                study_run_id=study_run_id,
                result_source="mlflow_artifact",
            )

        local_result_path = self._resolve_local_storage_path(pair_spec.storage_path) / "result.json"
        if local_result_path.exists():
            raw_payload = json.loads(local_result_path.read_text(encoding="utf-8"))
            if isinstance(raw_payload, Mapping) and "result" in raw_payload:
                loaded_payload = PairResult.model_validate(raw_payload)
                if loaded_payload.result_source == "canonical":
                    loaded_payload.result_source = "local_snapshot"
                return loaded_payload
            if isinstance(raw_payload, Mapping):
                result = dict(raw_payload)
                return PairResult(
                    model_name=pair_spec.model_name,
                    dataset_name=pair_spec.dataset_name,
                    storage_path=pair_spec.storage_path,
                    result=result,
                    study_run_id=study_run_id,
                    result_source="local_snapshot",
                )

        return PairResult(
            model_name=pair_spec.model_name,
            dataset_name=pair_spec.dataset_name,
            storage_path=pair_spec.storage_path,
            result=result,
            study_run_id=study_run_id,
            result_source="empty",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def _sync_pair_outputs(
        self,
        pair_spec: PairRunSpec,
        study_run_id: Optional[str],
        excel_path: Path,
        include_checkpoints: Optional[bool] = None,
        include_run_artifacts: bool = True,
    ) -> None:
        del pair_spec
        study_run = None
        should_upload_checkpoints = (
            bool(getattr(self._runtime_config, "seafile_upload_checkpoints", False))
            if include_checkpoints is None
            else include_checkpoints
        )
        if study_run_id:
            mlflow.set_tracking_uri(self._tracking_uri)
            study_run = mlflow.get_run(study_run_id)
            if should_upload_checkpoints:
                artifact_path = study_run.info.artifact_uri.replace("s3://", "seaweed_s3:")
                upload_to_seafile(
                    artifact_path,
                    Path(self._experiment_name) / ("BEST_" + study_run.info.run_name),
                )

        lock_path = excel_path.with_suffix(excel_path.suffix + ".lock")
        with FileLock(str(lock_path)):
            upload_to_seafile(
                self._experiment_root,
                Path(self._experiment_name),
                extra_args=["--include", "*.xlsx", "--include", "*.json"],
            )

        if include_run_artifacts and study_run_id:
            experiment = mlflow.get_experiment_by_name(self._experiment_name)
            if experiment is not None and study_run is not None:
                run_paths = gather_runs(study_run_id, [experiment.experiment_id])
                if run_paths:
                    with tempfile.TemporaryDirectory(dir=self._temp_dir) as tmp_dir:
                        tmp_path = Path(tmp_dir)
                        for run_id, hierarchy in run_paths.items():
                            run_handle = mlflow.get_run(run_id)
                            artifact_uri = run_handle.info.artifact_uri.replace("s3://", "seaweed_s3:")
                            local_run_path = tmp_path / hierarchy
                            local_run_path.mkdir(parents=True, exist_ok=True)
                            subprocess.run(
                                [
                                    "rclone",
                                    "sync",
                                    artifact_uri,
                                    str(local_run_path),
                                    "-v",
                                    "--exclude",
                                    "'**/*.ckpt'",
                                    "--transfers",
                                    "32",
                                    "--checkers",
                                    "64",
                                    "--fast-list",
                                ],
                                check=True,
                            )

                        upload_to_seafile(
                            tmp_path,
                            Path(self._experiment_name) / "all",
                            compressed=True,
                            archive_name=study_run.info.run_name,
                        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def _cleanup_pair_artifacts(self, pair_spec: PairRunSpec, study_run_id: Optional[str]) -> None:
        if not self._runtime_config.s3_bucket:
            return

        mlflow.set_tracking_uri(self._tracking_uri)
        if study_run_id:
            experiment = mlflow.get_experiment_by_name(self._experiment_name)
            if experiment is not None:
                run_paths = gather_runs(study_run_id, [experiment.experiment_id])
                for run_id in run_paths:
                    run_handle = mlflow.get_run(run_id)
                    artifact_uri = run_handle.info.artifact_uri
                    parsed_artifact = urlparse(artifact_uri)
                    run_path = parsed_artifact.path.lstrip("/")
                    delete_by_extension(self._runtime_config.s3_bucket, run_path, ["ckpt"])

        parsed_storage = urlparse(pair_spec.storage_path.split("?", 1)[0])
        tune_path = parsed_storage.path.lstrip("/") or parsed_storage.geturl().lstrip("/")
        if tune_path:
            delete_by_extension(self._runtime_config.s3_bucket, tune_path)

    @staticmethod
    def _resolve_local_storage_path(storage_path: str) -> Path:
        return Path(storage_path.split("?", 1)[0])

    def _update_excel_snapshot(
        self,
        pair_spec: PairRunSpec,
        result: Mapping[str, Any],
        *,
        excel_path: Path,
        status: str,
        allow_empty_row: bool,
    ) -> None:
        best_metrics = self._best_metrics_for_excel(result)
        if not best_metrics and allow_empty_row and status == "FAILED":
            best_metrics = {"status": status.lower()}
        if best_metrics:
            update_score_excel_one(
                pair_spec.model_name,
                pair_spec.dataset_name,
                best_metrics,
                excel_path,
                EXCEL_SHEET,
                minimize_metrics=[RUNTIME_METRIC],
            )
            return

        if allow_empty_row and excel_score_table_has_metrics(excel_path, EXCEL_SHEET):
            update_score_excel_one(
                pair_spec.model_name,
                pair_spec.dataset_name,
                {},
                excel_path,
                EXCEL_SHEET,
                minimize_metrics=[RUNTIME_METRIC],
            )

    def _best_metrics_for_excel(self, result: Mapping[str, Any]) -> dict[str, Any]:
        common_cfg = OmegaConf.load(str(Path(self._config_dir) / "params" / "common.yaml"))
        common_cfg_resolved = OmegaConf.to_container(common_cfg, resolve=True)
        metrics_to_use = list(common_cfg_resolved["study"].get("metrics", [])) + [RUNTIME_METRIC]
        return {
            key: value
            for key, value in result.items()
            if any(metric_name in key for metric_name in metrics_to_use)
        }

    @staticmethod
    def _resolve_result_source(current_source: str, *, result: Mapping[str, Any], status: str) -> str:
        if status == "STOPPED" and not result:
            return "empty_stopped"
        if status == "STOPPED":
            return current_source if current_source != "canonical" else "stopped_snapshot"
        return current_source
