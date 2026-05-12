from __future__ import annotations

from argparse import Namespace

import pytest

from noboom_benchmark.noboom_lib.core.config.runtime import RuntimeConfig
from noboom_cluster.noboom_cli_lib.settings import ControllerSettings, WorkerSettings


def test_controller_settings_export_pair_job_env(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENT_NAME", "NoBoomBenchmark__20260311_120000")
    monkeypatch.setenv("NOBOOM_MAPPED_STORAGE", "/workspace/noboom/storage")
    monkeypatch.setenv("NOBOOM_S3_BUCKET", "noboom-ray")
    monkeypatch.setenv("NOBOOM_S3_PREFIX", "experiment_data")
    monkeypatch.setenv("NOBOOM_PREPARED_DATASET_S3_PATH", "s3://prepared-bucket/prepared-root")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setenv(
        "OPTUNA_STORAGE_URI",
        "postgresql+psycopg://noboom:example-password@127.0.0.1:5432/optuna_db",
    )
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-postgres-password")
    monkeypatch.setenv("NOBOOM_EXCLUSIVE", "true")
    monkeypatch.setenv("NOBOOM_DEPLOYMENT_MODE", "cluster")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_CONTROLLER_LINEAGE", "0")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_DATASET_TRACKING", "1")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_LOGGED_MODELS", "1")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_EVALUATION", "0")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_TABLES", "1")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_SYSTEM_METRICS", "0")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_REGISTRY", "1")
    monkeypatch.setenv("NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS", "1")
    monkeypatch.setenv("NOBOOM_SEAFILE_UPLOAD_RESULTS", "0")
    monkeypatch.setenv("NOBOOM_MLFLOW_CONTROLLER_RUN_ID", "controller-run-id")

    settings = ControllerSettings()

    assert settings.to_shared_env()["NOBOOM_EXCLUSIVE"] == "1"
    assert settings.to_shared_env()["NOBOOM_DEPLOYMENT_MODE"] == "cluster"
    assert settings.to_shared_env()["NOBOOM_MLFLOW_ENABLE_CONTROLLER_LINEAGE"] == "0"
    assert settings.to_shared_env()["NOBOOM_MLFLOW_ENABLE_DATASET_TRACKING"] == "1"
    assert settings.to_shared_env()["NOBOOM_MLFLOW_ENABLE_EVALUATION"] == "0"
    assert settings.to_shared_env()["NOBOOM_MLFLOW_ENABLE_SYSTEM_METRICS"] == "0"
    assert settings.to_shared_env()["NOBOOM_MLFLOW_ENABLE_REGISTRY"] == "1"
    assert settings.to_shared_env()["NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS"] == "1"
    assert settings.to_shared_env()["NOBOOM_SEAFILE_UPLOAD_RESULTS"] == "0"
    assert settings.to_shared_env()["NOBOOM_MLFLOW_CONTROLLER_RUN_ID"] == "controller-run-id"
    assert settings.to_shared_env()["NOBOOM_PREPARED_DATASET_S3_PATH"] == "s3://prepared-bucket/prepared-root"
    assert settings.to_shared_env()["POSTGRES_PASSWORD"] == "test-postgres-password"
    assert settings.to_pair_job_env()["EXPERIMENT_NAME"] == "NoBoomBenchmark__20260311_120000"


def test_controller_settings_validate_required_raises_for_missing_postgres_password(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXPERIMENT_NAME", "NoBoomBenchmark__20260311_120000")
    monkeypatch.setenv("NOBOOM_MAPPED_STORAGE", "/workspace/noboom/storage")
    monkeypatch.setenv("NOBOOM_S3_BUCKET", "noboom-ray")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setenv(
        "OPTUNA_STORAGE_URI",
        "postgresql+psycopg://noboom:test-password@127.0.0.1:5432/optuna_db",
    )
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        ControllerSettings().validate_required()


def test_controller_settings_local_pair_env_does_not_require_s3_or_postgres(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENT_NAME", "NoBoomBenchmark__local")
    monkeypatch.setenv("NOBOOM_MAPPED_STORAGE", "/tmp/noboom-local")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/noboom-local/mlruns")
    monkeypatch.setenv("OPTUNA_STORAGE_URI", "memory://")
    monkeypatch.delenv("NOBOOM_S3_BUCKET", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    env = ControllerSettings().to_local_pair_job_env()

    assert env["EXPERIMENT_NAME"] == "NoBoomBenchmark__local"
    assert env["NOBOOM_S3_BUCKET"] == ""
    assert env["OPTUNA_STORAGE_URI"] == "memory://"
    assert env["NOBOOM_EXECUTION_BACKEND"] == "local"


def test_worker_settings_validate_required_raises(monkeypatch) -> None:
    monkeypatch.delenv("EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("NOBOOM_MAPPED_STORAGE", raising=False)
    monkeypatch.delenv("NOBOOM_S3_BUCKET", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("OPTUNA_STORAGE_URI", raising=False)

    with pytest.raises(RuntimeError, match="EXPERIMENT_NAME"):
        WorkerSettings().validate_required()


def test_mlflow_logged_model_features_default_disabled(monkeypatch) -> None:
    monkeypatch.delenv("NOBOOM_MLFLOW_ENABLE_LOGGED_MODELS", raising=False)
    monkeypatch.delenv("NOBOOM_MLFLOW_ENABLE_EVALUATION", raising=False)
    monkeypatch.delenv("NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS", raising=False)
    monkeypatch.delenv("NOBOOM_SEAFILE_UPLOAD_RESULTS", raising=False)

    controller_settings = ControllerSettings()
    worker_settings = WorkerSettings()

    assert controller_settings.nooboom_mlflow_enable_logged_models is False
    assert controller_settings.nooboom_mlflow_enable_evaluation is False
    assert controller_settings.nooboom_seafile_upload_checkpoints is False
    assert controller_settings.nooboom_seafile_upload_results is False
    assert worker_settings.nooboom_mlflow_enable_logged_models is False
    assert worker_settings.nooboom_mlflow_enable_evaluation is False
    assert worker_settings.nooboom_seafile_upload_checkpoints is False
    assert worker_settings.nooboom_seafile_upload_results is False


def test_runtime_config_reads_worker_settings_and_arg_overrides(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENT_NAME", "NoBoomBenchmark__20260311_120000")
    monkeypatch.setenv("NOBOOM_MAPPED_STORAGE", "/workspace/noboom/storage")
    monkeypatch.setenv("NOBOOM_S3_BUCKET", "noboom-ray")
    monkeypatch.setenv("NOBOOM_S3_PREFIX", "experiment_data")
    monkeypatch.setenv("NOBOOM_PREPARED_DATASET_S3_PATH", "s3://prepared-bucket/from-env")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setenv(
        "OPTUNA_STORAGE_URI",
        "postgresql+psycopg://noboom:example-password@127.0.0.1:5432/optuna_db",
    )
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-postgres-password")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://127.0.0.1:8333")
    monkeypatch.setenv("NOBOOM_DEPLOYMENT_MODE", "cluster")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_CONTROLLER_LINEAGE", "0")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_DATASET_TRACKING", "1")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_LOGGED_MODELS", "1")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_EVALUATION", "1")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_TABLES", "0")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_SYSTEM_METRICS", "0")
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_REGISTRY", "1")
    monkeypatch.setenv("NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS", "1")
    monkeypatch.setenv("NOBOOM_SEAFILE_UPLOAD_RESULTS", "0")
    monkeypatch.setenv("NOBOOM_MLFLOW_CONTROLLER_RUN_ID", "controller-run-id")

    runtime_config = RuntimeConfig.from_env_and_args(
        Namespace(
            timestamp="20260311_120000",
            tracking_uri="http://10.0.0.1:5000",
            optuna_storage_uri=None,
            temp_dir="/tmp/ray",
            config_dir="configs",
            verbose=1,
            model=["gdn"],
            dataset=["ome"],
            mode="tune",
            env_file=None,
            experiment_id=None,
            model_name=None,
            dataset_name=None,
            storage_path=None,
            gpus_per_run=0.25,
            prepared_dataset_s3_path="s3://prepared-bucket/from-args",
        )
    )

    runtime_config.validate()

    assert runtime_config.mlflow_tracking_uri == "http://10.0.0.1:5000"
    assert runtime_config.optuna_storage_uri == (
        "postgresql+psycopg://noboom:example-password@127.0.0.1:5432/optuna_db"
    )
    assert runtime_config.s3_prefix.as_posix() == "experiment_data/NoBoomBenchmark__20260311_120000"
    assert runtime_config.prepared_dataset_s3_path == "s3://prepared-bucket/from-args"
    assert runtime_config.s3_endpoint_url == "http://127.0.0.1:8333"
    assert runtime_config.deployment_mode == "cluster"
    assert runtime_config.mlflow_enable_controller_lineage is False
    assert runtime_config.mlflow_enable_dataset_tracking is True
    assert runtime_config.mlflow_enable_logged_models is True
    assert runtime_config.mlflow_enable_evaluation is True
    assert runtime_config.mlflow_enable_tables is False
    assert runtime_config.mlflow_enable_system_metrics is False
    assert runtime_config.mlflow_enable_registry is True
    assert runtime_config.seafile_upload_checkpoints is True
    assert runtime_config.seafile_upload_results is False
    assert runtime_config.mlflow_controller_run_id == "controller-run-id"


def test_runtime_config_local_mode_does_not_require_s3(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EXPERIMENT_NAME", "NoBoomBenchmark__local")
    monkeypatch.setenv("NOBOOM_MAPPED_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setenv("OPTUNA_STORAGE_URI", "memory://")
    monkeypatch.delenv("NOBOOM_S3_BUCKET", raising=False)

    runtime_config = RuntimeConfig.from_env_and_args(
        Namespace(
            execution_backend="local",
            artifact_storage_backend="local",
            timestamp="20260311_120000",
            tracking_uri=None,
            optuna_storage_uri=None,
            temp_dir=str(tmp_path / "tmp"),
            config_dir="configs",
            verbose=0,
            model=None,
            dataset=None,
            mode=None,
            env_file=None,
            experiment_id=None,
            model_name="gdn",
            dataset_name="ome",
            storage_path=str(tmp_path / "storage" / "local" / "ome__gdn"),
            gpus_per_run=0.25,
            prepared_dataset_s3_path=None,
        )
    )

    runtime_config.validate()

    assert runtime_config.execution_backend == "local"
    assert runtime_config.artifact_storage_backend == "local"
    assert runtime_config.s3_bucket is None
