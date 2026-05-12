from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field
import yaml

NOBOOM_REPOSITORY_URL = os.environ.get("NOBOOM_REPOSITORY_URL", "https://github.com/denix56/noboom.git")
TIMESEAD_REPOSITORY_URL = os.environ.get(
    "TIMESEAD_REPOSITORY_URL",
    "https://github.com/denix56/TimeSeAD.git",
)
TIMESEAD_EXTENSIONS_REPOSITORY_URL = os.environ.get(
    "TIMESEAD_EXTENSIONS_REPOSITORY_URL",
    "https://github.com/denix56/TimeSeAD-extensions.git",
)


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ip: str
    devices: Optional[str] = None
    ssh_user: Optional[str] = None

    def resolved_ssh_user(self, default_ssh_user: str) -> str:
        return self.ssh_user or default_ssh_user


class InventoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: List[NodeSpec]

    @staticmethod
    def resolve_path(path: str) -> Path:
        file_path = Path(path).expanduser()
        if file_path.exists():
            return file_path.resolve()

        if not file_path.is_absolute():
            package_root = Path(__file__).resolve().parents[1]
            package_candidate = package_root / file_path
            if package_candidate.exists():
                return package_candidate.resolve()

            repo_root = Path(__file__).resolve().parents[3]
            repo_candidate = repo_root / file_path
            if repo_candidate.exists():
                return repo_candidate.resolve()

        raise FileNotFoundError(f"Inventory file not found: {file_path}")

    @classmethod
    def load(cls, path: str) -> "InventoryConfig":
        file_path = cls.resolve_path(path)

        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        return cls.model_validate(data)


class ServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    head_local_ip: str
    storage_path: str
    mapped_storage: str
    workdir: str
    workdir_host: str
    root_dir: str
    ray_temp_dir: str
    mlflow_ui_port: int
    nooboom_s3_bucket: str
    nooboom_s3_prefix: str
    nooboom_prepared_dataset_s3_path: Optional[str] = None
    mlflow_tracking_uri: str
    optuna_storage_uri: str
    seafile_username: str = ""
    seafile_pass: str = ""
    seafile_root_path: str = ""
    kaggle_api_token: Optional[str] = None
    nooboom_worker_debug: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_endpoint_url: Optional[str] = None
    enable_seaweed: bool = True
    use_docker: bool = True
    postgres_user: str = "noboom"
    postgres_password: str
    ray_enable_autoscaler_v2: str = "0"
    deployment_mode: str = "docker"
    nooboom_mlflow_enable_controller_lineage: bool = True
    nooboom_mlflow_enable_dataset_tracking: bool = True
    nooboom_mlflow_enable_logged_models: bool = False
    nooboom_mlflow_enable_evaluation: bool = False
    nooboom_mlflow_enable_tables: bool = True
    nooboom_mlflow_enable_system_metrics: bool = True
    nooboom_mlflow_enable_registry: bool = False
    nooboom_seafile_upload_checkpoints: bool = False
    nooboom_seafile_upload_results: bool = False
    nooboom_mlflow_controller_run_id: Optional[str] = None
    nooboom_notify_email_enabled: bool = False
    nooboom_notify_smtp_host: str = ""
    nooboom_notify_smtp_port: int = 587
    nooboom_notify_smtp_username: str = ""
    nooboom_notify_smtp_password: str = Field(default="", repr=False)
    nooboom_notify_smtp_from: str = ""
    nooboom_notify_smtp_to: str = ""
    nooboom_notify_smtp_use_tls: bool = True
    nooboom_notify_smtp_use_ssl: bool = False
    nooboom_notify_smtp_timeout_s: float = 10.0
    nooboom_notify_telegram_enabled: bool = False
    nooboom_notify_telegram_link_token: str = Field(default="", repr=False)
    nooboom_notify_telegram_start_link: str = ""
    nooboom_notify_telegram_relay_url: str = ""
    nooboom_notify_telegram_relay_secret: str = Field(default="", repr=False)
    nooboom_notify_telegram_timeout_s: float = 10.0
    nooboom_notify_ram_breach_ratio: str = ""

    def to_env_dict(self) -> Dict[str, str]:
        env = {
            "POSTGRES_USER": self.postgres_user,
            "POSTGRES_PASSWORD": self.postgres_password,
            "OPTUNA_STORAGE_URI": self.optuna_storage_uri,
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "MLFLOW_STORAGE_URI": (
                f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@"
                f"{self.head_local_ip}:5432/mlflow_db"
            ),
            "NOBOOM_STORAGE": self.storage_path,
            "NOBOOM_ROOT_DIR": self.root_dir,
            "KAGGLE_API_TOKEN": self.kaggle_api_token or "",
            "MLFLOW_UI_LOCAL_PORT": str(self.mlflow_ui_port),
            "NOBOOM_MAPPED_STORAGE": self.mapped_storage,
            "NOBOOM_WORKER_DEBUG": self.nooboom_worker_debug or "",
            "NOBOOM_DOCKER_WORKDIR": self.workdir,
            "NOBOOM_DOCKER_WORKDIR_HOST": self.workdir_host,
            "HEAD_LOCAL_IP": self.head_local_ip,
            "NOBOOM_S3_BUCKET": self.nooboom_s3_bucket,
            "NOBOOM_S3_PREFIX": self.nooboom_s3_prefix,
            "NOBOOM_PREPARED_DATASET_S3_PATH": self.nooboom_prepared_dataset_s3_path or "",
            "SEAFILE_USERNAME": self.seafile_username,
            "SEAFILE_PASS": self.seafile_pass,
            "SEAFILE_ROOT_PATH": self.seafile_root_path,
            "NUMBA_CACHE_DIR": f"{self.workdir}/.numba",
            "NOBOOM_USE_SEAWEED": "1" if self.enable_seaweed else "0",
            "NOBOOM_SKIP_DOCKER_CLEANUP": "0" if self.use_docker else "1",
            "NOBOOM_SKIP_COMPOSE_UP": "0" if self.use_docker else "1",
            "NOBOOM_USE_DOCKER": "1" if self.use_docker else "0",
            "KAGGLEHUB_CACHE": f"{self.mapped_storage}/datasets",
            "RAY_AUTH_MODE": "token",
            "RAY_enable_autoscaler_v2": self.ray_enable_autoscaler_v2,
            "TORCHINDUCTOR_CACHE_DIR": str(Path(self.ray_temp_dir).parent / "inductor_cache"),
            "AWS_REGION": self.aws_region,
            "AWS_EC2_METADATA_DISABLED": "true",
            "NOBOOM_DEPLOYMENT_MODE": self.deployment_mode,
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
            "NOBOOM_NOTIFY_RAM_BREACH_RATIO": self.nooboom_notify_ram_breach_ratio,
        }

        if self.aws_access_key_id:
            env["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            env["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        if self.s3_endpoint_url:
            env["S3_ENDPOINT_URL"] = self.s3_endpoint_url
            env["MLFLOW_S3_ENDPOINT_URL"] = self.s3_endpoint_url

        return env


class ClusterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine_ip_file: str
    cluster_config: Optional[str] = None
    ssh_user: str = "cloud"
    ssh_key: str = str(Path("~/.ssh/id_ed25519").expanduser())
    root_dir: str = "~/noboom"
    ray_temp_dir: str = "/tmp/ray"
    dashboard_port: int = 8265
    mlflow_port: int = 5001
    deployment_mode: str = "docker"
    enable_seaweed: bool = True
    force_restart: bool = False
    exclusive: bool = False
    verbose: int = 0
    gpus_per_run: float = 0.125

    @property
    def use_docker(self) -> bool:
        return self.deployment_mode == "docker"


class DatasetModelPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    model: str

    def as_tuple(self) -> Tuple[str, str]:
        return self.dataset, self.model


def parse_dataset_model_pair(raw_pair: str) -> DatasetModelPair:
    parts = raw_pair.split(":")
    if len(parts) != 2:
        raise ValueError("Expected pair syntax DATASET:MODEL.")

    dataset = parts[0].strip()
    model = parts[1].strip()
    if not dataset or not model:
        raise ValueError("Expected pair syntax DATASET:MODEL with non-empty values.")

    return DatasetModelPair(dataset=dataset, model=model)


class ControllerRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    dataset_model_pairs: List[DatasetModelPair] = Field(default_factory=list)
    tune: bool = False
    source_experiment_id: Optional[str] = None
    config_dir: str = "configs"
    temp_dir: str = "/tmp/ray"
    gpus_per_run: float
    timestamp: str
    verbose: int = 0
    env_file: Optional[str] = None
    max_in_flight: int = 4
    save_checkpoints: bool = False
    hpo_seeds: Optional[List[int]] = None
    execution_backend: str = "ray"
    artifact_storage_backend: str = "remote"
    optuna_storage_backend: str = "configured"
    local_storage_path: Optional[str] = None
    optuna_sqlite_path: Optional[str] = None
    ram_guard_enabled: bool = True
    ram_guard_max_used_fraction: float = 0.95
    ram_guard_poll_interval_s: float = 10.0
    ram_guard_resume_used_fraction: float = 0.90
    ram_guard_cooldown_s: float = 120.0


class DependencyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    noboom_sha: str
    timesead_sha: str
    timesead_extensions_sha: str

    def to_uv_packages(self) -> List[str]:
        return [
            f"noboom[data,baselines] @ git+{NOBOOM_REPOSITORY_URL}@{self.noboom_sha}",
            f"timesead[experiments] @ git+{TIMESEAD_REPOSITORY_URL}@{self.timesead_sha}",
            (
                "timesead-extensions @ "
                f"git+{TIMESEAD_EXTENSIONS_REPOSITORY_URL}@{self.timesead_extensions_sha}"
            ),
        ]

    def write_json(self, path: str) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return str(output_path)


class PairRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    source_experiment_id: Optional[str] = None
    model_name: str
    dataset_name: str
    timestamp: str
    storage_path: str
    gpus_per_run: float
    optuna_storage_uri: str
    config_dir: str
    temp_dir: str
    data_manifest_path: Optional[str] = None
    verbose: int = 0
    tune: bool = False
    env_file: Optional[str] = None
    tracking_uri: str
    s3_endpoint_url: Optional[str] = None
    prepared_dataset_s3_path: Optional[str] = None
    save_checkpoints: bool = False
    hpo_seeds: Optional[List[int]] = None
    execution_backend: str = "ray"
    artifact_storage_backend: str = "remote"
    optuna_storage_backend: str = "configured"

    @property
    def submission_id(self) -> str:
        return f"{self.dataset_name}__{self.model_name}__{self.timestamp}"

    @property
    def pair_id(self) -> str:
        return f"{self.dataset_name}__{self.model_name}"

    def to_json(self) -> str:
        return self.model_dump_json()

    def to_base64(self) -> str:
        return base64.urlsafe_b64encode(self.to_json().encode("utf-8")).decode("ascii")

    def write_json(self, path: str) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return str(output_path)

    @classmethod
    def load(cls, path: str) -> "PairRunSpec":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.model_validate(json.load(handle))

    @classmethod
    def from_base64(cls, payload: str) -> "PairRunSpec":
        raw_payload = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
        return cls.model_validate_json(raw_payload)


class PairResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str
    dataset_name: str
    storage_path: str
    result: Dict[str, Any] = Field(default_factory=dict)
    study_run_id: Optional[str] = None
    status: str = "SUCCEEDED"
    partial_result: bool = False
    job_status_message: Optional[str] = None
    result_source: str = "canonical"
    seafile_synced: bool = False
    cleanup_performed: bool = False
