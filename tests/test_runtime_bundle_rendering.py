from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from noboom_cluster.noboom_cli_lib.deployment import build_service_settings, get_backend
from noboom_cluster.noboom_cli_lib.runtime_bundle import (
    STYLE_TRANSFER_CHECKPOINT_RELATIVE_PATH,
    build_runtime_bundle,
)


def test_docker_bundle_renders_outside_repo() -> None:
    bundle = build_runtime_bundle(project_root=Path.cwd(), deployment_mode="docker")
    backend = get_backend("docker")
    service_settings = build_service_settings(
        controller_settings={
            "NOBOOM_S3_BUCKET": "noboom-ray",
            "NOBOOM_S3_PREFIX": "experiment_data",
            "POSTGRES_USER": "noboom",
            "POSTGRES_PASSWORD": "example-password",
        },
        head_local_ip="10.0.0.1",
        root_dir="/home/cloud/noboom",
        storage_path="/home/cloud/noboom/experiment_data",
        mapped_storage="/workspace/noboom/storage",
        workdir="/workspace/noboom",
        workdir_host="/tmp/ray_tmp_mount/noboom-benchmark-docker/workspace/noboom",
        ray_temp_dir="/tmp/ray",
        mlflow_ui_port=5001,
        enable_seaweed=True,
        use_docker=True,
        aws_access_key_id="access",
        aws_secret_access_key="secret",
    )

    backend.render_env_file(bundle, service_settings)
    backend.render_cluster_config(
        bundle,
        head_public_ip="1.2.3.4",
        head_local_ip="10.0.0.1",
        worker_local_ips=["10.0.0.2"],
        ssh_user="cloud",
        ssh_key="~/.ssh/id_ed25519",
        root_dir="/home/cloud/noboom",
        ray_temp_dir="/tmp/ray",
        storage_path="/home/cloud/noboom/experiment_data",
        mapped_storage="/workspace/noboom/storage",
        mlflow_ui_port=5001,
        workdir="/workspace/noboom",
        workdir_host_root="/tmp/ray_tmp_mount/noboom-benchmark-docker/workspace/noboom",
        mount_files_host="/tmp/ray_tmp_mount/noboom-benchmark-docker/workspace/noboom/mnt",
    )

    cluster_yaml = bundle.cluster_config_path.read_text(encoding="utf-8")
    cluster_config = yaml.safe_load(cluster_yaml)
    env_file = bundle.env_base_path.read_text(encoding="utf-8")

    assert bundle.cluster_config_path.exists()
    assert bundle.root_dir != Path.cwd()
    assert str(bundle.mount_files_dir.resolve()) in cluster_yaml
    assert cluster_config["setup_commands"] == ["BASE=/tmp/ray bash /workspace/noboom/mnt/scripts/cleanup_tmp.sh"]
    assert "MLFLOW_TRACKING_URI=http://10.0.0.1:5000" in env_file
    assert (bundle.root_dir / "noboom_benchmark" / "run_tune.py").exists()
    assert (bundle.root_dir / "noboom_cluster" / "noboom_cli_lib" / "specs.py").exists()


def test_native_bundle_renders_native_paths() -> None:
    bundle = build_runtime_bundle(project_root=Path.cwd(), deployment_mode="native")
    backend = get_backend("native")
    service_settings = build_service_settings(
        controller_settings={
            "NOBOOM_S3_BUCKET": "noboom-ray",
            "NOBOOM_S3_PREFIX": "experiment_data",
            "POSTGRES_USER": "noboom",
            "POSTGRES_PASSWORD": "example-password",
        },
        head_local_ip="10.0.0.1",
        root_dir="/work/user/noboom",
        storage_path="/work/user/noboom/experiment_data",
        mapped_storage="/work/user/noboom/experiment_data",
        workdir="/work/user/noboom",
        workdir_host="/work/user/noboom",
        ray_temp_dir="/work/user/noboom/tmp/ray",
        mlflow_ui_port=5001,
        enable_seaweed=True,
        use_docker=False,
        aws_access_key_id="access",
        aws_secret_access_key="secret",
    )

    backend.render_env_file(bundle, service_settings)
    backend.render_cluster_config(
        bundle,
        head_public_ip="203.0.113.10",
        head_local_ip="203.0.113.10",
        worker_local_ips=[],
        ssh_user="user",
        ssh_key="~/.ssh/id_ed25519",
        root_dir="/work/user/noboom",
        ray_temp_dir="/work/user/noboom/tmp/ray",
        storage_path="/work/user/noboom/experiment_data",
        mapped_storage="/work/user/noboom/experiment_data",
        mlflow_ui_port=5001,
        workdir="/work/user/noboom",
        workdir_host_root="/work/user/noboom",
        mount_files_host="/work/user/noboom/mnt",
    )

    cluster_yaml = bundle.cluster_config_path.read_text(encoding="utf-8")
    cluster_config = yaml.safe_load(cluster_yaml)

    assert bundle.cluster_config_path.exists()
    assert "/work/user/noboom/mnt" in cluster_yaml
    assert cluster_config["setup_commands"] == [
        "BASE=/work/user/noboom/tmp/ray bash /work/user/noboom/mnt/scripts/cleanup_tmp.sh"
    ]
    assert "ray stop --force" in cluster_yaml
    head_start_commands = cluster_config["head_start_ray_commands"]
    assert "ray stop --force" in head_start_commands[0]
    assert "filter_busy_gpus.py" in head_start_commands[1]
    assert "ray start --head" in head_start_commands[2]
    worker_start_commands = cluster_config["worker_start_ray_commands"]
    assert "ray stop --force" in worker_start_commands[0]
    assert "filter_busy_gpus.py" in worker_start_commands[1]
    assert "ray start --address=$RAY_HEAD_IP:6379" in worker_start_commands[2]


def test_build_service_settings_requires_postgres_password() -> None:
    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        build_service_settings(
            controller_settings={
                "NOBOOM_S3_BUCKET": "noboom-ray",
                "NOBOOM_S3_PREFIX": "experiment_data",
                "POSTGRES_USER": "noboom",
            },
            head_local_ip="10.0.0.1",
            root_dir="/work/user/noboom",
            storage_path="/work/user/noboom/experiment_data",
            mapped_storage="/work/user/noboom/experiment_data",
            workdir="/work/user/noboom",
            workdir_host="/work/user/noboom",
            ray_temp_dir="/work/user/noboom/tmp/ray",
            mlflow_ui_port=5001,
            enable_seaweed=True,
            use_docker=False,
        )


def test_runtime_bundle_includes_style_transfer_checkpoint() -> None:
    bundle = build_runtime_bundle(project_root=Path.cwd(), deployment_mode="native")

    assert (bundle.root_dir / STYLE_TRANSFER_CHECKPOINT_RELATIVE_PATH).exists()
