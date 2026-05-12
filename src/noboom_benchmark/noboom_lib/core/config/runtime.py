"""Runtime configuration helpers for NoBoom benchmark."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from noboom_cluster.noboom_cli_lib.settings import WorkerSettings


@dataclass(slots=True)
class RuntimeConfig:
    experiment_name: Optional[str]
    mapped_storage: Optional[str]
    s3_bucket: Optional[str]
    prepared_dataset_s3_path: Optional[str]
    mlflow_tracking_uri: Optional[str]
    optuna_storage_uri: Optional[str]
    s3_prefix: Path
    s3_endpoint_url: Optional[str]
    seafile_username: Optional[str]
    seafile_password: Optional[str]
    seafile_root_path: Optional[str]
    seafile_upload_checkpoints: bool
    seafile_upload_results: bool
    nooboom_exclusive: bool
    timestamp: str
    model: Optional[List[str]]
    dataset: Optional[List[str]]
    mode: Optional[str]
    verbose: Optional[int]
    config_dir: Optional[str]
    temp_dir: Optional[str]
    env_file: Optional[str]
    experiment_id: Optional[str]
    model_name: Optional[str]
    dataset_name: Optional[str]
    storage_path: Optional[str]
    gpus_per_run: Optional[float]
    deployment_mode: str
    mlflow_enable_controller_lineage: bool
    mlflow_enable_dataset_tracking: bool
    mlflow_enable_logged_models: bool
    mlflow_enable_evaluation: bool
    mlflow_enable_tables: bool
    mlflow_enable_system_metrics: bool
    mlflow_enable_registry: bool
    mlflow_controller_run_id: Optional[str]
    execution_backend: str
    artifact_storage_backend: str
    notification_email_enabled: bool = False
    notification_telegram_enabled: bool = False
    notification_ram_breach_ratio: Optional[str] = None

    @classmethod
    def from_env_and_args(cls, args: Namespace) -> "RuntimeConfig":
        settings = WorkerSettings()
        execution_backend = str(
            getattr(args, "execution_backend", None)
            or getattr(settings, "nooboom_execution_backend", None)
            or "ray"
        )
        artifact_storage_backend = str(getattr(args, "artifact_storage_backend", None) or "remote")
        if execution_backend != "local":
            settings.validate_required()

        timestamp = getattr(args, "timestamp", None) or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        s3_prefix = Path(settings.s3_prefix)
        if settings.experiment_name:
            s3_prefix = s3_prefix / settings.experiment_name

        tracking_uri = getattr(args, "tracking_uri", None) or settings.mlflow_tracking_uri
        optuna_storage_uri = getattr(args, "optuna_storage_uri", None) or settings.optuna_storage_uri

        return cls(
            experiment_name=settings.experiment_name,
            mapped_storage=settings.mapped_storage,
            s3_bucket=settings.s3_bucket,
            prepared_dataset_s3_path=(
                getattr(args, "prepared_dataset_s3_path", None) or settings.prepared_dataset_s3_path
            ),
            mlflow_tracking_uri=tracking_uri,
            optuna_storage_uri=optuna_storage_uri,
            s3_prefix=s3_prefix,
            s3_endpoint_url=settings.s3_endpoint_url,
            seafile_username=settings.seafile_username,
            seafile_password=settings.seafile_pass,
            seafile_root_path=settings.seafile_root_path,
            seafile_upload_checkpoints=settings.nooboom_seafile_upload_checkpoints,
            seafile_upload_results=settings.nooboom_seafile_upload_results,
            nooboom_exclusive=settings.nooboom_exclusive,
            timestamp=timestamp,
            model=getattr(args, "model", None),
            dataset=getattr(args, "dataset", None),
            mode=getattr(args, "mode", None),
            verbose=getattr(args, "verbose", None),
            config_dir=getattr(args, "config_dir", None),
            temp_dir=getattr(args, "temp_dir", "/tmp/ray"),
            env_file=getattr(args, "env_file", None),
            experiment_id=getattr(args, "experiment_id", None),
            model_name=getattr(args, "model_name", None),
            dataset_name=getattr(args, "dataset_name", None),
            storage_path=getattr(args, "storage_path", None),
            gpus_per_run=getattr(args, "gpus_per_run", None),
            deployment_mode=settings.nooboom_deployment_mode,
            mlflow_enable_controller_lineage=settings.nooboom_mlflow_enable_controller_lineage,
            mlflow_enable_dataset_tracking=settings.nooboom_mlflow_enable_dataset_tracking,
            mlflow_enable_logged_models=settings.nooboom_mlflow_enable_logged_models,
            mlflow_enable_evaluation=settings.nooboom_mlflow_enable_evaluation,
            mlflow_enable_tables=settings.nooboom_mlflow_enable_tables,
            mlflow_enable_system_metrics=settings.nooboom_mlflow_enable_system_metrics,
            mlflow_enable_registry=settings.nooboom_mlflow_enable_registry,
            mlflow_controller_run_id=settings.nooboom_mlflow_controller_run_id,
            execution_backend=execution_backend,
            artifact_storage_backend=artifact_storage_backend,
            notification_email_enabled=settings.nooboom_notify_email_enabled,
            notification_telegram_enabled=settings.nooboom_notify_telegram_enabled,
            notification_ram_breach_ratio=settings.nooboom_notify_ram_breach_ratio,
        )

    def validate(self) -> None:
        if self.execution_backend == "local":
            missing = []
            required = {
                "EXPERIMENT_NAME": self.experiment_name,
                "NOBOOM_MAPPED_STORAGE": self.mapped_storage,
                "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
                "OPTUNA_STORAGE_URI": self.optuna_storage_uri,
            }
            for name, value in required.items():
                if not value:
                    missing.append(name)
            if missing:
                raise RuntimeError(
                    "Missing required configuration values: "
                    + ", ".join(sorted(missing))
                    + ". Please set them in the environment or .env file."
                )
            return

        missing = []
        required = {
            "EXPERIMENT_NAME": self.experiment_name,
            "NOBOOM_MAPPED_STORAGE": self.mapped_storage,
            "NOBOOM_S3_BUCKET": self.s3_bucket,
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "OPTUNA_STORAGE_URI": self.optuna_storage_uri,
        }
        for name, value in required.items():
            if not value:
                missing.append(name)
        if missing:
            raise RuntimeError(
                "Missing required configuration values: "
                + ", ".join(sorted(missing))
                + ". Please set them in the environment or .env file."
            )
