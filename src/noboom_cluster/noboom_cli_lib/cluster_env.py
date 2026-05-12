"""Environment preparation utilities for cluster nodes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .ray_utils.internal.env import update_env

MLFLOW_STORAGE_URI = "postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${HEAD_LOCAL_IP}:5432/mlflow_db"
OPTUNA_STORAGE_URI = "postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${HEAD_LOCAL_IP}:5432/optuna_db"
DEFAULT_S3_BUCKET = "noboom-ray"
DEFAULT_S3_PREFIX = "experiment_data"


def _require_env_var(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_env_vars(
    *,
    head_local_ip: str,
    aws_env: Mapping[str, str],
    root_dir: str,
    storage_path: str,
    mapped_storage: str,
    workdir: str,
    workdir_host: str,
    enable_seaweed: bool,
    use_docker: bool,
    mlflow_ui_port: int,
    ray_temp_dir: str,
) -> dict[str, str]:
    postgres_password = _require_env_var("POSTGRES_PASSWORD")

    env_vars = {
        "OPTUNA_STORAGE_URI": OPTUNA_STORAGE_URI,
        "MLFLOW_STORAGE_URI": MLFLOW_STORAGE_URI,
        "MLFLOW_TRACKING_URI": f"http://{head_local_ip}:5000",
        "NOBOOM_STORAGE": storage_path,
        # TODO: Remove on code release
        "NOBOOM_ROOT_DIR": root_dir,
        "KAGGLE_API_TOKEN": os.getenv("KAGGLE_API_TOKEN"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER", "noboom"),
        "POSTGRES_PASSWORD": postgres_password,
        "MLFLOW_UI_LOCAL_PORT": str(mlflow_ui_port),
        "NOBOOM_MAPPED_STORAGE": mapped_storage,
        "NOBOOM_WORKER_DEBUG": os.getenv("NOBOOM_WORKER_DEBUG"),
        "NOBOOM_DOCKER_WORKDIR": workdir,
        "NOBOOM_DOCKER_WORKDIR_HOST": workdir_host,
        "HEAD_LOCAL_IP": head_local_ip,
        "NOBOOM_S3_BUCKET": os.getenv("NOBOOM_S3_BUCKET", DEFAULT_S3_BUCKET),
        "NOBOOM_S3_PREFIX": os.getenv("NOBOOM_S3_PREFIX", DEFAULT_S3_PREFIX),
        "SEAFILE_USERNAME": os.getenv("SEAFILE_USERNAME", ""),
        "SEAFILE_PASS": os.getenv("SEAFILE_PASS", ""),
        "SEAFILE_ROOT_PATH": os.getenv("SEAFILE_ROOT_PATH", ""),
        "NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS": os.getenv("NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS", "0"),
        "NOBOOM_SEAFILE_UPLOAD_RESULTS": os.getenv("NOBOOM_SEAFILE_UPLOAD_RESULTS", "0"),
        "NUMBA_CACHE_DIR": f"{workdir}/.numba",
        "NOBOOM_USE_SEAWEED": "1" if enable_seaweed else "0",
        "NOBOOM_SKIP_DOCKER_CLEANUP": "1" if not use_docker else "0",
        "NOBOOM_SKIP_COMPOSE_UP": "1" if not use_docker else "0",
        "NOBOOM_USE_DOCKER": "1" if use_docker else "0",
        "KAGGLEHUB_CACHE": f"{mapped_storage}/datasets",
        "RAY_AUTH_MODE": "token",
        "RAY_enable_autoscaler_v2": os.getenv("RAY_ENABLE_AUTOSCALER_V2", "0"),
        "TORCHINDUCTOR_CACHE_DIR": str(Path(ray_temp_dir).parent / "inductor_cache"),
        **aws_env,
    }

    return env_vars


def prepare_cluster_env_file(
    env_template_path: str,
    env_output_path: str,
    head_local_ip: str,
    aws_env: Mapping[str, str],
    *,
    root_dir: str,
    storage_path: str,
    mapped_storage: str,
    workdir: str,
    workdir_host: str,
    enable_seaweed: bool,
    use_docker: bool,
    mlflow_ui_port: int,
    ray_temp_dir: str,
) -> str:
    """Create a cluster-wide environment file from a template on the local machine.

    Args:
        env_template_path (str): Path to the base `.env` template file.
        env_output_path (str): Destination path for the rendered `.env` file.
        head_local_ip (str): Local IP address of the head node.
        aws_env (Mapping[str, str]): AWS/SeaweedFS environment variables to inject.
        root_dir (str): Root directory on the node for cluster assets.
        storage_path (str): Path to shared storage on the node.
        mapped_storage (str): Container-mapped storage path.
        workdir (str): Working directory for the runtime or containers.
        enable_seaweed (bool): Whether to enable SeaweedFS configuration.
        use_docker (bool): Whether the node runs services inside Docker.
        mlflow_ui_port (int): Local port for the MLflow UI.

    Returns:
        str: Path to the newly generated environment file.

    Side Effects:
        Writes a new `.env` file to disk.
    """
    template_path = Path(env_template_path)
    output_path = Path(env_output_path + ".base")
    output_path.write_text(template_path.read_text())
    os.environ["HEAD_LOCAL_IP"] = head_local_ip
    env_vars = _build_env_vars(
        head_local_ip=head_local_ip,
        aws_env=aws_env,
        root_dir=root_dir,
        storage_path=storage_path,
        mapped_storage=mapped_storage,
        workdir=workdir,
        workdir_host=workdir_host,
        enable_seaweed=enable_seaweed,
        use_docker=use_docker,
        mlflow_ui_port=mlflow_ui_port,
        ray_temp_dir=ray_temp_dir
    )
    update_env(output_path, env_vars, suffix="")
    return str(output_path)
