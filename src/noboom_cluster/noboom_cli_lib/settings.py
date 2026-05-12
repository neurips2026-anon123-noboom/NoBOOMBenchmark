from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ControllerSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    experiment_name: Optional[str] = Field(default=None, validation_alias="EXPERIMENT_NAME")
    nooboom_mapped_storage: Optional[str] = Field(
        default=None, validation_alias="NOBOOM_MAPPED_STORAGE"
    )
    nooboom_s3_bucket: Optional[str] = Field(default=None, validation_alias="NOBOOM_S3_BUCKET")
    nooboom_s3_prefix: str = Field(default="experiment_data", validation_alias="NOBOOM_S3_PREFIX")
    nooboom_prepared_dataset_s3_path: Optional[str] = Field(
        default=None,
        validation_alias="NOBOOM_PREPARED_DATASET_S3_PATH",
    )
    mlflow_tracking_uri: Optional[str] = Field(
        default=None, validation_alias="MLFLOW_TRACKING_URI"
    )
    optuna_storage_uri: Optional[str] = Field(
        default=None, validation_alias="OPTUNA_STORAGE_URI"
    )
    s3_endpoint_url: Optional[str] = Field(default=None, validation_alias="S3_ENDPOINT_URL")
    seafile_username: str = Field(default="", validation_alias="SEAFILE_USERNAME")
    seafile_pass: str = Field(default="", validation_alias="SEAFILE_PASS")
    seafile_root_path: str = Field(default="", validation_alias="SEAFILE_ROOT_PATH")
    kaggle_api_token: Optional[str] = Field(default=None, validation_alias="KAGGLE_API_TOKEN")
    nooboom_worker_debug: Optional[str] = Field(
        default=None, validation_alias="NOBOOM_WORKER_DEBUG"
    )
    machine_nodes_b64: Optional[str] = Field(default=None, validation_alias="NOBOOM_MACHINE_NODES_B64")
    postgres_user: str = Field(default="noboom", validation_alias="POSTGRES_USER")
    postgres_password: Optional[str] = Field(default=None, validation_alias="POSTGRES_PASSWORD")
    ray_enable_autoscaler_v2: str = Field(
        default="0", validation_alias="RAY_ENABLE_AUTOSCALER_V2"
    )
    nooboom_exclusive: bool = Field(default=False, validation_alias="NOBOOM_EXCLUSIVE")
    nooboom_deployment_mode: str = Field(default="docker", validation_alias="NOBOOM_DEPLOYMENT_MODE")
    nooboom_mlflow_enable_controller_lineage: bool = Field(
        default=True,
        validation_alias="NOBOOM_MLFLOW_ENABLE_CONTROLLER_LINEAGE",
    )
    nooboom_mlflow_enable_dataset_tracking: bool = Field(
        default=True,
        validation_alias="NOBOOM_MLFLOW_ENABLE_DATASET_TRACKING",
    )
    nooboom_mlflow_enable_logged_models: bool = Field(
        default=False,
        validation_alias="NOBOOM_MLFLOW_ENABLE_LOGGED_MODELS",
    )
    nooboom_mlflow_enable_evaluation: bool = Field(
        default=False,
        validation_alias="NOBOOM_MLFLOW_ENABLE_EVALUATION",
    )
    nooboom_mlflow_enable_tables: bool = Field(
        default=True,
        validation_alias="NOBOOM_MLFLOW_ENABLE_TABLES",
    )
    nooboom_mlflow_enable_system_metrics: bool = Field(
        default=True,
        validation_alias="NOBOOM_MLFLOW_ENABLE_SYSTEM_METRICS",
    )
    nooboom_mlflow_enable_registry: bool = Field(
        default=False,
        validation_alias="NOBOOM_MLFLOW_ENABLE_REGISTRY",
    )
    nooboom_seafile_upload_checkpoints: bool = Field(
        default=False,
        validation_alias="NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS",
    )
    nooboom_seafile_upload_results: bool = Field(
        default=False,
        validation_alias="NOBOOM_SEAFILE_UPLOAD_RESULTS",
    )
    nooboom_mlflow_controller_run_id: Optional[str] = Field(
        default=None,
        validation_alias="NOBOOM_MLFLOW_CONTROLLER_RUN_ID",
    )
    nooboom_ram_guard_enabled: bool = Field(default=True, validation_alias="NOBOOM_RAM_GUARD_ENABLED")
    nooboom_ram_guard_max_used_fraction: float = Field(
        default=0.95,
        validation_alias="NOBOOM_RAM_GUARD_MAX_USED_FRACTION",
    )
    nooboom_ram_guard_poll_interval_s: float = Field(
        default=10.0,
        validation_alias="NOBOOM_RAM_GUARD_POLL_INTERVAL_S",
    )
    nooboom_ram_guard_resume_used_fraction: float = Field(
        default=0.90,
        validation_alias="NOBOOM_RAM_GUARD_RESUME_USED_FRACTION",
    )
    nooboom_ram_guard_cooldown_s: float = Field(
        default=120.0,
        validation_alias="NOBOOM_RAM_GUARD_COOLDOWN_S",
    )
    nooboom_notify_email_enabled: bool = Field(default=False, validation_alias="NOBOOM_NOTIFY_EMAIL_ENABLED")
    nooboom_notify_smtp_host: str = Field(default="", validation_alias="NOBOOM_NOTIFY_SMTP_HOST")
    nooboom_notify_smtp_port: int = Field(default=587, validation_alias="NOBOOM_NOTIFY_SMTP_PORT")
    nooboom_notify_smtp_username: str = Field(default="", validation_alias="NOBOOM_NOTIFY_SMTP_USERNAME")
    nooboom_notify_smtp_password: str = Field(
        default="",
        validation_alias="NOBOOM_NOTIFY_SMTP_PASSWORD",
        repr=False,
    )
    nooboom_notify_smtp_from: str = Field(default="", validation_alias="NOBOOM_NOTIFY_SMTP_FROM")
    nooboom_notify_smtp_to: str = Field(default="", validation_alias="NOBOOM_NOTIFY_SMTP_TO")
    nooboom_notify_smtp_use_tls: bool = Field(default=True, validation_alias="NOBOOM_NOTIFY_SMTP_USE_TLS")
    nooboom_notify_smtp_use_ssl: bool = Field(default=False, validation_alias="NOBOOM_NOTIFY_SMTP_USE_SSL")
    nooboom_notify_smtp_timeout_s: float = Field(
        default=10.0,
        validation_alias="NOBOOM_NOTIFY_SMTP_TIMEOUT_S",
    )
    nooboom_notify_telegram_enabled: bool = Field(
        default=False,
        validation_alias="NOBOOM_NOTIFY_TELEGRAM_ENABLED",
    )
    nooboom_notify_telegram_link_token: str = Field(
        default="",
        validation_alias="NOBOOM_NOTIFY_TELEGRAM_LINK_TOKEN",
        repr=False,
    )
    nooboom_notify_telegram_start_link: str = Field(
        default="",
        validation_alias="NOBOOM_NOTIFY_TELEGRAM_START_LINK",
    )
    nooboom_notify_telegram_relay_url: str = Field(
        default="",
        validation_alias="NOBOOM_NOTIFY_TELEGRAM_RELAY_URL",
    )
    nooboom_notify_telegram_relay_secret: str = Field(
        default="",
        validation_alias="NOBOOM_NOTIFY_TELEGRAM_RELAY_SECRET",
        repr=False,
    )
    nooboom_notify_telegram_timeout_s: float = Field(
        default=10.0,
        validation_alias="NOBOOM_NOTIFY_TELEGRAM_TIMEOUT_S",
    )
    nooboom_notify_ram_breach_ratio: Optional[str] = Field(
        default=None,
        validation_alias="NOBOOM_NOTIFY_RAM_BREACH_RATIO",
    )

    def validate_required(self) -> None:
        missing = []
        required = {
            "EXPERIMENT_NAME": self.experiment_name,
            "NOBOOM_MAPPED_STORAGE": self.nooboom_mapped_storage,
            "NOBOOM_S3_BUCKET": self.nooboom_s3_bucket,
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "OPTUNA_STORAGE_URI": self.optuna_storage_uri,
            "POSTGRES_PASSWORD": self.postgres_password,
        }
        for key, value in required.items():
            if not value:
                missing.append(key)
        if missing:
            raise RuntimeError(
                "Missing required configuration values: "
                + ", ".join(sorted(missing))
                + ". Please set them in the environment or .env file."
            )

    def to_shared_env(self) -> Dict[str, str]:
        return {
            "NOBOOM_MAPPED_STORAGE": self.nooboom_mapped_storage or "",
            "NOBOOM_S3_BUCKET": self.nooboom_s3_bucket or "",
            "NOBOOM_S3_PREFIX": self.nooboom_s3_prefix,
            "NOBOOM_PREPARED_DATASET_S3_PATH": self.nooboom_prepared_dataset_s3_path or "",
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri or "",
            "OPTUNA_STORAGE_URI": self.optuna_storage_uri or "",
            "SEAFILE_USERNAME": self.seafile_username,
            "SEAFILE_PASS": self.seafile_pass,
            "SEAFILE_ROOT_PATH": self.seafile_root_path,
            "POSTGRES_USER": self.postgres_user,
            "POSTGRES_PASSWORD": self.postgres_password or "",
            "RAY_ENABLE_AUTOSCALER_V2": self.ray_enable_autoscaler_v2,
            "NOBOOM_EXCLUSIVE": "1" if self.nooboom_exclusive else "0",
            "NOBOOM_DEPLOYMENT_MODE": self.nooboom_deployment_mode,
            "KAGGLE_API_TOKEN": self.kaggle_api_token or "",
            "NOBOOM_WORKER_DEBUG": self.nooboom_worker_debug or "",
            "S3_ENDPOINT_URL": self.s3_endpoint_url or "",
            "NOBOOM_MLFLOW_ENABLE_CONTROLLER_LINEAGE": (
                "1" if self.nooboom_mlflow_enable_controller_lineage else "0"
            ),
            "NOBOOM_MLFLOW_ENABLE_DATASET_TRACKING": (
                "1" if self.nooboom_mlflow_enable_dataset_tracking else "0"
            ),
            "NOBOOM_MLFLOW_ENABLE_LOGGED_MODELS": (
                "1" if self.nooboom_mlflow_enable_logged_models else "0"
            ),
            "NOBOOM_MLFLOW_ENABLE_EVALUATION": (
                "1" if self.nooboom_mlflow_enable_evaluation else "0"
            ),
            "NOBOOM_MLFLOW_ENABLE_TABLES": "1" if self.nooboom_mlflow_enable_tables else "0",
            "NOBOOM_MLFLOW_ENABLE_SYSTEM_METRICS": (
                "1" if self.nooboom_mlflow_enable_system_metrics else "0"
            ),
            "NOBOOM_MLFLOW_ENABLE_REGISTRY": (
                "1" if self.nooboom_mlflow_enable_registry else "0"
            ),
            "NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS": (
                "1" if self.nooboom_seafile_upload_checkpoints else "0"
            ),
            "NOBOOM_SEAFILE_UPLOAD_RESULTS": (
                "1" if self.nooboom_seafile_upload_results else "0"
            ),
            "NOBOOM_MLFLOW_CONTROLLER_RUN_ID": self.nooboom_mlflow_controller_run_id or "",
            "NOBOOM_RAM_GUARD_ENABLED": "1" if self.nooboom_ram_guard_enabled else "0",
            "NOBOOM_RAM_GUARD_MAX_USED_FRACTION": str(self.nooboom_ram_guard_max_used_fraction),
            "NOBOOM_RAM_GUARD_POLL_INTERVAL_S": str(self.nooboom_ram_guard_poll_interval_s),
            "NOBOOM_RAM_GUARD_RESUME_USED_FRACTION": str(self.nooboom_ram_guard_resume_used_fraction),
            "NOBOOM_RAM_GUARD_COOLDOWN_S": str(self.nooboom_ram_guard_cooldown_s),
            "NOBOOM_NOTIFY_EMAIL_ENABLED": "1" if self.nooboom_notify_email_enabled else "0",
            "NOBOOM_NOTIFY_SMTP_HOST": self.nooboom_notify_smtp_host,
            "NOBOOM_NOTIFY_SMTP_PORT": str(self.nooboom_notify_smtp_port),
            "NOBOOM_NOTIFY_SMTP_USERNAME": self.nooboom_notify_smtp_username,
            "NOBOOM_NOTIFY_SMTP_PASSWORD": self.nooboom_notify_smtp_password,
            "NOBOOM_NOTIFY_EMAIL_FROM": self.nooboom_notify_smtp_from,
            "NOBOOM_NOTIFY_EMAIL_TO": self.nooboom_notify_smtp_to,
            "NOBOOM_NOTIFY_SMTP_FROM": self.nooboom_notify_smtp_from,
            "NOBOOM_NOTIFY_SMTP_TO": self.nooboom_notify_smtp_to,
            "NOBOOM_NOTIFY_SMTP_USE_TLS": "1" if self.nooboom_notify_smtp_use_tls else "0",
            "NOBOOM_NOTIFY_SMTP_USE_SSL": "1" if self.nooboom_notify_smtp_use_ssl else "0",
            "NOBOOM_NOTIFY_SMTP_TIMEOUT_S": str(self.nooboom_notify_smtp_timeout_s),
            "NOBOOM_NOTIFY_TELEGRAM_ENABLED": "1" if self.nooboom_notify_telegram_enabled else "0",
            "NOBOOM_NOTIFY_TELEGRAM_LINK_TOKEN": self.nooboom_notify_telegram_link_token,
            "NOBOOM_NOTIFY_TELEGRAM_START_LINK": self.nooboom_notify_telegram_start_link,
            "NOBOOM_NOTIFY_TELEGRAM_RELAY_URL": self.nooboom_notify_telegram_relay_url,
            "NOBOOM_NOTIFY_TELEGRAM_RELAY_SECRET": self.nooboom_notify_telegram_relay_secret,
            "NOBOOM_NOTIFY_TELEGRAM_TIMEOUT_S": str(self.nooboom_notify_telegram_timeout_s),
            "NOBOOM_NOTIFY_RAM_BREACH_RATIO": self.nooboom_notify_ram_breach_ratio or "",
        }

    def to_pair_job_env(self) -> Dict[str, str]:
        self.validate_required()
        env = self.to_shared_env()
        env["EXPERIMENT_NAME"] = str(self.experiment_name)
        return env

    def to_local_pair_job_env(self) -> Dict[str, str]:
        env = self.to_shared_env()
        env["EXPERIMENT_NAME"] = str(self.experiment_name or "")
        env["NOBOOM_EXECUTION_BACKEND"] = "local"
        return env


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    experiment_name: Optional[str] = Field(default=None, validation_alias="EXPERIMENT_NAME")
    mapped_storage: Optional[str] = Field(default=None, validation_alias="NOBOOM_MAPPED_STORAGE")
    s3_bucket: Optional[str] = Field(default=None, validation_alias="NOBOOM_S3_BUCKET")
    s3_prefix: str = Field(default="experiment_data", validation_alias="NOBOOM_S3_PREFIX")
    prepared_dataset_s3_path: Optional[str] = Field(
        default=None,
        validation_alias="NOBOOM_PREPARED_DATASET_S3_PATH",
    )
    mlflow_tracking_uri: Optional[str] = Field(
        default=None, validation_alias="MLFLOW_TRACKING_URI"
    )
    optuna_storage_uri: Optional[str] = Field(
        default=None, validation_alias="OPTUNA_STORAGE_URI"
    )
    s3_endpoint_url: Optional[str] = Field(default=None, validation_alias="S3_ENDPOINT_URL")
    seafile_username: str = Field(default="", validation_alias="SEAFILE_USERNAME")
    seafile_pass: str = Field(default="", validation_alias="SEAFILE_PASS")
    seafile_root_path: str = Field(default="", validation_alias="SEAFILE_ROOT_PATH")
    nooboom_exclusive: bool = Field(default=False, validation_alias="NOBOOM_EXCLUSIVE")
    nooboom_deployment_mode: str = Field(default="docker", validation_alias="NOBOOM_DEPLOYMENT_MODE")
    nooboom_mlflow_enable_controller_lineage: bool = Field(
        default=True,
        validation_alias="NOBOOM_MLFLOW_ENABLE_CONTROLLER_LINEAGE",
    )
    nooboom_mlflow_enable_dataset_tracking: bool = Field(
        default=True,
        validation_alias="NOBOOM_MLFLOW_ENABLE_DATASET_TRACKING",
    )
    nooboom_mlflow_enable_logged_models: bool = Field(
        default=False,
        validation_alias="NOBOOM_MLFLOW_ENABLE_LOGGED_MODELS",
    )
    nooboom_mlflow_enable_evaluation: bool = Field(
        default=False,
        validation_alias="NOBOOM_MLFLOW_ENABLE_EVALUATION",
    )
    nooboom_mlflow_enable_tables: bool = Field(
        default=True,
        validation_alias="NOBOOM_MLFLOW_ENABLE_TABLES",
    )
    nooboom_mlflow_enable_system_metrics: bool = Field(
        default=True,
        validation_alias="NOBOOM_MLFLOW_ENABLE_SYSTEM_METRICS",
    )
    nooboom_mlflow_enable_registry: bool = Field(
        default=False,
        validation_alias="NOBOOM_MLFLOW_ENABLE_REGISTRY",
    )
    nooboom_seafile_upload_checkpoints: bool = Field(
        default=False,
        validation_alias="NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS",
    )
    nooboom_seafile_upload_results: bool = Field(
        default=False,
        validation_alias="NOBOOM_SEAFILE_UPLOAD_RESULTS",
    )
    nooboom_mlflow_controller_run_id: Optional[str] = Field(
        default=None,
        validation_alias="NOBOOM_MLFLOW_CONTROLLER_RUN_ID",
    )
    nooboom_notify_email_enabled: bool = Field(default=False, validation_alias="NOBOOM_NOTIFY_EMAIL_ENABLED")
    nooboom_notify_telegram_enabled: bool = Field(
        default=False,
        validation_alias="NOBOOM_NOTIFY_TELEGRAM_ENABLED",
    )
    nooboom_notify_ram_breach_ratio: Optional[str] = Field(
        default=None,
        validation_alias="NOBOOM_NOTIFY_RAM_BREACH_RATIO",
    )

    def validate_required(self) -> None:
        missing = []
        required = {
            "EXPERIMENT_NAME": self.experiment_name,
            "NOBOOM_MAPPED_STORAGE": self.mapped_storage,
            "NOBOOM_S3_BUCKET": self.s3_bucket,
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "OPTUNA_STORAGE_URI": self.optuna_storage_uri,
        }
        for key, value in required.items():
            if not value:
                missing.append(key)
        if missing:
            raise RuntimeError(
                "Missing required configuration values: "
                + ", ".join(sorted(missing))
                + ". Please set them in the environment or .env file."
            )

    def to_runtime_kwargs(self) -> Dict[str, Optional[str]]:
        self.validate_required()
        return {
            "experiment_name": self.experiment_name,
            "mapped_storage": self.mapped_storage,
            "s3_bucket": self.s3_bucket,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "optuna_storage_uri": self.optuna_storage_uri,
            "s3_endpoint_url": self.s3_endpoint_url,
            "prepared_dataset_s3_path": self.prepared_dataset_s3_path,
            "seafile_username": self.seafile_username,
            "seafile_password": self.seafile_pass,
            "seafile_root_path": self.seafile_root_path,
            "s3_prefix": str(Path(self.s3_prefix)),
            "deployment_mode": self.nooboom_deployment_mode,
            "mlflow_enable_controller_lineage": self.nooboom_mlflow_enable_controller_lineage,
            "mlflow_enable_dataset_tracking": self.nooboom_mlflow_enable_dataset_tracking,
            "mlflow_enable_logged_models": self.nooboom_mlflow_enable_logged_models,
            "mlflow_enable_evaluation": self.nooboom_mlflow_enable_evaluation,
            "mlflow_enable_tables": self.nooboom_mlflow_enable_tables,
            "mlflow_enable_system_metrics": self.nooboom_mlflow_enable_system_metrics,
            "mlflow_enable_registry": self.nooboom_mlflow_enable_registry,
            "seafile_upload_checkpoints": self.nooboom_seafile_upload_checkpoints,
            "seafile_upload_results": self.nooboom_seafile_upload_results,
            "mlflow_controller_run_id": self.nooboom_mlflow_controller_run_id,
            "notification_email_enabled": self.nooboom_notify_email_enabled,
            "notification_telegram_enabled": self.nooboom_notify_telegram_enabled,
            "notification_ram_breach_ratio": self.nooboom_notify_ram_breach_ratio,
        }
