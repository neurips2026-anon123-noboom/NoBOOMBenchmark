from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, Mapping, Sequence, Tuple

import ray.autoscaler.sdk
import yaml

from .internal.ssh import get_remote_local_ip


def get_resolved_workdir(workdir: str, cluster_config: str, use_docker: bool) -> str:
    if use_docker:
        cluster_config = Path(cluster_config)
        if not cluster_config.exists():
            cluster_config = cluster_config.with_name(cluster_config.name + '.template')
        with cluster_config.open("r") as f:
            cfg = yaml.safe_load(f)
            cluster_name = cfg["cluster_name"]
            dir_name = f"{ray.autoscaler.sdk.get_docker_host_mount_location(cluster_name)}{workdir}"
    else:
        dir_name = workdir
    return dir_name


def head_sha(repo: str = "../..") -> str:
    """Get the Git SHA for the HEAD of the specified repository.

    Args:
        repo (str): Path to the Git repository. Defaults to "../..".

    Returns:
        str: The HEAD commit SHA.

    Raises:
        subprocess.CalledProcessError: If the git command fails.
    """
    r = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    return r.stdout.strip()


def preprocess_cluster_yaml(
    *,
    yaml_path: str | Path,
    head_ip: str,
    worker_ips: Sequence[str],
    ssh_user: str,
    ssh_key: str | Path,
    root_dir: str | Path,
    workdir: str | Path,
    ray_temp_dir: str,
    storage_path: str,
    remote_storage_path: str,
    aws_env: Mapping[str, str],
    enable_seaweed: bool,
    mlflow_ui_port: int,
    use_docker: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """Load and update a Ray cluster YAML with runtime metadata.

    Args:
        yaml_path (str | Path): Path to the Ray cluster config template.
        head_ip (str): Public IP of the head node.
        worker_ips (Sequence[str]): Public IPs of worker nodes.
        ssh_user (str): SSH username.
        ssh_key (str | Path): Path to the SSH private key.
        root_dir (str | Path): Root directory on remote hosts.
        workdir (str | Path): Work directory used by Ray.
        storage_path (str): Storage path used inside the runtime.
        remote_storage_path (str): Host path to map into containers.
        env_file_path (str | Path): Path to the prepared `.env` file.
        machine_file_path (str | Path): Path to the machine file with local IPs.
        aws_env (Mapping[str, str]): AWS/SeaweedFS environment variables.
        enable_seaweed (bool): Whether to enable SeaweedFS configuration.
        mlflow_ui_port (int): MLflow UI port to expose.
        use_docker (bool): Whether the cluster uses Docker. Defaults to True.

    Returns:
        Dict[str, Any]: Updated cluster configuration dictionary.

    Side Effects:
        Executes SSH commands and writes the updated YAML to disk.
    """
    yaml_path_template = Path(yaml_path).with_suffix(".yaml.template")

    with yaml_path_template.open("r") as f:
        cfg = yaml.safe_load(f)

    workdir_mount = f"{workdir}/mnt"
    workdir_host = get_resolved_workdir(workdir_mount, cluster_config=yaml_path, use_docker=use_docker)

    cfg["provider"] = {
        "type": "local",
        "head_ip": head_ip,
        "worker_ips": list(worker_ips),
        "external_head_ip": head_ip,
    }
    cfg["min_workers"] = len(worker_ips)
    cfg["max_workers"] = len(worker_ips)

    if "file_mounts" not in cfg or cfg["file_mounts"] is None:
        cfg["file_mounts"] = {}
    cfg["file_mounts"][str(Path("/root" if use_docker else f"/home/{ssh_user}") / ".ray/auth_token")] = str(Path("~/.ray/auth_token").expanduser())
    # .env is updated in initialization_commands but we need to prevent it from syncing
    cfg["file_mounts"][str(workdir_mount)] = str(
        (Path(__file__).resolve().parents[1] / "scripts" / "mount_files").resolve()
    )

    if "setup_commands" not in cfg or cfg["setup_commands"] is None:
        cfg["setup_commands"] = []

    cleanup_script_path = shlex.quote(f"{workdir_mount}/scripts/cleanup_tmp.sh")
    cleanup_base_dir = shlex.quote(ray_temp_dir)
    cfg["setup_commands"].append(f"BASE={cleanup_base_dir} bash {cleanup_script_path}")

    if "head_setup_commands" not in cfg or cfg["head_setup_commands"] is None:
        cfg["head_setup_commands"] = []
    head_setup_commands = cfg["head_setup_commands"]
    head_setup_cmd = f"bash {workdir_mount}/scripts/setup_head_container.sh"
    if not use_docker:
        head_setup_cmd = " ".join(
            [
                f"source {root_dir}/.venv/bin/activate &&",
                "set -a &&",
                f"source {workdir_host}/.env &&",
                "set +a &&",
                head_setup_cmd,
            ]
        )
    if head_setup_cmd not in head_setup_commands:
        head_setup_commands.append(head_setup_cmd)

    if "initialization_commands" not in cfg or cfg["initialization_commands"] is None:
        cfg["initialization_commands"] = []
    initialization_commands = cfg["initialization_commands"]
    cuda_venv_path = str(Path(workdir_host).parent / ".venv-set-cuda")
    cuda_venv_bin = f"{cuda_venv_path}/bin"
    cuda_venv_cmd = " ".join(
        [
            # Install uv and ensure it's on PATH for this non-interactive shell
            r'curl -LsSf https://astral.sh/uv/install.sh | sh;',
            r'export PATH="$HOME/.local/bin:$PATH";',
            r'command -v uv >/dev/null 2>&1 || { echo "uv not found after install" >&2; exit 1; };',

            # Create venv with Python 3.12 (only if missing)
            f'if [ ! -d "{cuda_venv_path}" ]; then uv venv "{cuda_venv_path}" --python=3.12 --clear; fi;',

            # Install deps using uv into that venv
            f'uv pip install --python "{cuda_venv_bin}/python" -q pyyaml;',
        ]
    )

    initialization_commands.append(cuda_venv_cmd)

    cuda_init_cmd = " ".join(
        [
            f"set -a; source \"{workdir_host}/.env.base\"; set +a;",
            f"\"{cuda_venv_bin}/python\" {workdir_host}/scripts/setup_env.py",
        ]
    )
    initialization_commands.append(cuda_init_cmd)

    env_export_vars = {
        "NOBOOM_ROOT_DIR": str(root_dir),
        "NOBOOM_STORAGE": storage_path,
        "NOBOOM_MAPPED_STORAGE": remote_storage_path,
        "NOBOOM_DOCKER_WORKDIR": str(workdir),
        "NOBOOM_USE_DOCKER": "1" if use_docker else "0",
        "NOBOOM_USE_SEAWEED": "1" if enable_seaweed else "0",
        "MLFLOW_UI_LOCAL_PORT": str(mlflow_ui_port),
        "RAY_ENABLE_AUTOSCALER_V2": os.getenv("RAY_ENABLE_AUTOSCALER_V2", "0"),
        **aws_env,
    }
    for key, value in env_export_vars.items():
        if value is None:
            continue
        os.environ.setdefault(key, str(value))

    if use_docker:
        # Ensure docker section exists
        if "docker" not in cfg or cfg["docker"] is None:
            cfg["docker"] = {}
        docker_cfg = cfg["docker"]
        docker_cfg.setdefault("run_options", [])

        if "run_options" not in docker_cfg or docker_cfg["run_options"] is None:
            docker_cfg["run_options"] = []
        docker_cfg["run_options"].append(f"--env-file={workdir_host}/.env")
        docker_cfg["run_options"].append(f"-v {root_dir}/datasets:/datasets")
        docker_cfg["run_options"].append(f"-v {remote_storage_path}:{storage_path}")

        if os.getenv("NOBOOM_WORKER_DEBUG") == "1":
            docker_cfg["run_options"].append(f"-v {root_dir}/logs:/workspace/logs")

        node_init_cmd = (f"set -a; source \"{workdir_host}/.env\"; set +a; "
                         f"sudo python3 {workdir_host}/scripts/setup_node.py")
        if node_init_cmd not in initialization_commands:
            initialization_commands.append(node_init_cmd)

        init_cmd = (f"set -a; source \"{workdir_host}/.env\"; set +a; "
                    f"sudo bash {workdir_host}/scripts/setup_docker_group.sh")
        if init_cmd not in initialization_commands:
            initialization_commands.append(init_cmd)

    head_start_cmd = (
        "RAY_enable_autoscaler_v2=0 ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 "
        f"--autoscaling-config=~/ray_bootstrap_config.yaml --temp-dir={ray_temp_dir} --resources='{{\"exclusive\": 1}}' --labels='{{\"role\":\"head\"}}'"
    )
    worker_start_cmd = f"RAY_enable_autoscaler_v2=0 ray start --address=$RAY_HEAD_IP:6379 --temp-dir={ray_temp_dir} --resources='{{\"exclusive\": 1}}'"
    env_cmd = f"set -a; source \"{workdir_host}/.env\"; set +a;"
    if not use_docker:
        extra_cmd = " ".join(
            [
                f"source \"{root_dir}/.venv/bin/activate\";",
                env_cmd,
            ]
        )
    else:
        extra_cmd = env_cmd
    filter_busy_gpus_cmd = (
        f"python3 \"{workdir_host}/scripts/filter_busy_gpus.py\" --env-file \"{workdir_host}/.env\""
    )
    for commands_list, start_cmd in zip(
        ["head_start_ray_commands", "worker_start_ray_commands"],
        (head_start_cmd, worker_start_cmd),
    ):
        cfg[commands_list].append(filter_busy_gpus_cmd)
        cfg[commands_list].append(start_cmd)
        for i, command in enumerate(cfg[commands_list]):
            cfg[commands_list][i] = " ".join(
                [
                    extra_cmd,
                    cfg[commands_list][i],
                ]
            ).strip()

    if "auth" not in cfg or cfg["auth"] is None:
        cfg["auth"] = {}
    auth_cfg = cfg["auth"]

    auth_cfg["ssh_user"] = ssh_user
    auth_cfg["ssh_private_key"] = ssh_key

    output_path = Path(yaml_path)
    with output_path.open("w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    return cfg, str(output_path)
