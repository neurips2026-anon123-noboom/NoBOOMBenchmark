from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
import shlex
from typing import Dict, List, Mapping, Optional, Sequence

import ray.autoscaler.sdk

from .runtime_bundle import RuntimeBundle, render_template
from .specs import ServiceSettings


RAY_DOCKER_WORKDIR = "/workspace/noboom"


def resolve_workdir_host(workdir: str, cluster_name: str, use_docker: bool) -> str:
    if use_docker:
        return f"{ray.autoscaler.sdk.get_docker_host_mount_location(cluster_name)}{workdir}"
    return workdir


class DeploymentBackend(ABC):
    cluster_name: str
    use_docker: bool

    @property
    @abstractmethod
    def template_name(self) -> str:
        raise NotImplementedError

    @property
    def remote_workdir(self) -> str:
        return RAY_DOCKER_WORKDIR if self.use_docker else ""

    def render_cluster_config(
        self,
        bundle: RuntimeBundle,
        *,
        head_public_ip: str,
        head_local_ip: str,
        worker_local_ips: Sequence[str],
        ssh_user: str,
        ssh_key: str,
        root_dir: str,
        ray_temp_dir: str,
        storage_path: str,
        mapped_storage: str,
        mlflow_ui_port: int,
        workdir: str,
        workdir_host_root: str,
        mount_files_host: str,
        ray_head_ip: Optional[str] = None,
        ray_worker_ips: Optional[Sequence[str]] = None,
        force_public_node_ip_map: Optional[Mapping[str, str]] = None,
    ) -> Path:
        forced_public_node_ips = force_public_node_ip_map or {}
        workdir_mount_remote = f"{workdir}/mnt"
        auth_token_remote_path = str(
            Path("/root" if self.use_docker else f"/home/{ssh_user}") / ".ray" / "auth_token"
        )

        cleanup_script_path = shlex.quote(f"{workdir_mount_remote}/scripts/cleanup_tmp.sh")
        cleanup_base_dir = shlex.quote(ray_temp_dir)
        setup_commands = [f"BASE={cleanup_base_dir} bash {cleanup_script_path}"]
        head_setup_commands = [
            self._head_setup_command(root_dir, workdir_mount_remote, workdir_host_root)
        ]
        initialization_commands = self._initialization_commands(
            root_dir=root_dir,
            mount_files_host=mount_files_host,
            workdir_mount_remote=workdir_mount_remote,
            force_public_node_ip_map=forced_public_node_ips,
        )
        head_start_ray_commands, worker_start_ray_commands = self._ray_start_commands(
            root_dir=root_dir,
            workdir_mount_remote=workdir_mount_remote,
            ray_temp_dir=ray_temp_dir,
            force_public_node_ip_map=forced_public_node_ips,
        )

        provider_head_ip = ray_head_ip or head_public_ip
        provider_worker_ips = (
            list(ray_worker_ips) if ray_worker_ips is not None else list(worker_local_ips)
        )

        context: Dict[str, object] = {
            "cluster_name": self.cluster_name,
            "docker_image": os.getenv("NOBOOM_DOCKER_IMAGE", "ghcr.io/denix56/noboom:latest"),
            "docker_run_options": self._docker_run_options(
                root_dir=root_dir,
                storage_path=storage_path,
                mapped_storage=mapped_storage,
                workdir_host_root=workdir_host_root,
                force_public_node_ip_map=forced_public_node_ips,
            ),
            "head_ip": provider_head_ip,
            "worker_ips": provider_worker_ips,
            "external_head_ip": head_public_ip,
            "ssh_user": ssh_user,
            "ssh_private_key": ssh_key,
            "setup_commands": setup_commands,
            "head_setup_commands": head_setup_commands,
            "initialization_commands": initialization_commands,
            "head_start_ray_commands": head_start_ray_commands,
            "worker_start_ray_commands": worker_start_ray_commands,
            "max_workers": len(worker_local_ips),
            "min_workers": len(worker_local_ips),
            "auth_token_remote_path": auth_token_remote_path,
            "auth_token_local_path": str(Path("~/.ray/auth_token").expanduser()),
            "workdir_mount_remote_path": workdir_mount_remote,
            "mount_files_local_path": str(bundle.mount_files_dir.resolve()),
            "mlflow_ui_port": mlflow_ui_port,
        }
        render_template(self.template_name, bundle.cluster_config_path, context)
        return bundle.cluster_config_path

    def render_env_file(self, bundle: RuntimeBundle, service_settings: ServiceSettings) -> Path:
        env_vars = service_settings.to_env_dict()
        render_template("cluster.env.j2", bundle.env_base_path, {"env_vars": env_vars})
        return bundle.env_base_path

    def _docker_run_options(
        self,
        *,
        root_dir: str,
        storage_path: str,
        mapped_storage: str,
        workdir_host_root: str,
        force_public_node_ip_map: Mapping[str, str],
    ) -> List[str]:
        if not self.use_docker:
            return []
        options = [
            "--ulimit nofile=65536:65536",
            "--gpus=all",
            "--runtime=nvidia",
            f"--env-file={workdir_host_root}/mnt/.env",
            f"-v {root_dir}/datasets:/datasets",
            f"-v {storage_path}:{mapped_storage}",
        ]
        return options

    def _head_setup_command(self, root_dir: str, workdir_mount_remote: str, workdir_host: str) -> str:
        head_setup_cmd = f"bash {workdir_mount_remote}/scripts/setup_head_container.sh"
        if self.use_docker:
            return head_setup_cmd
        return " ".join(
            [
                f"source {root_dir}/.venv/bin/activate &&",
                "set -a &&",
                f"source {workdir_host}/.env &&",
                "set +a &&",
                head_setup_cmd,
            ]
        )

    def _initialization_commands(
        self,
        *,
        root_dir: str,
        mount_files_host: str,
        workdir_mount_remote: str,
        force_public_node_ip_map: Mapping[str, str],
    ) -> List[str]:
        cuda_venv_path = str(Path(mount_files_host).parent / ".venv-set-cuda")
        cuda_venv_bin = f"{cuda_venv_path}/bin"
        commands = [
            " ".join(
                [
                    'curl -LsSf https://astral.sh/uv/install.sh | sh;',
                    'export PATH="$HOME/.local/bin:$PATH";',
                    'command -v uv >/dev/null 2>&1 || { echo "uv not found after install" >&2; exit 1; };',
                    f'if [ ! -d "{cuda_venv_path}" ]; then uv venv "{cuda_venv_path}" --python=3.12 --clear; fi;',
                    f'uv pip install --python "{cuda_venv_bin}/python" -q pyyaml;',
                ]
            ),
            " ".join(
                [
                    f'set -a; source "{mount_files_host}/.env.base"; set +a;',
                    f'"{cuda_venv_bin}/python" {mount_files_host}/scripts/setup_env.py',
                ]
            ),
        ]
        if self.use_docker:
            commands.append(
                f'set -a; source "{mount_files_host}/.env"; set +a; sudo python3 {mount_files_host}/scripts/setup_node.py'
            )
            commands.append(
                f'set -a; source "{mount_files_host}/.env"; set +a; sudo bash {mount_files_host}/scripts/setup_docker_group.sh'
            )
            alias_command = self._host_loopback_alias_command(force_public_node_ip_map)
            if alias_command:
                commands.append(alias_command)
        return commands

    def _host_loopback_alias_command(self, force_public_node_ip_map: Mapping[str, str]) -> str:
        forced_pairs = " ".join(
            f"{local_ip}={public_ip}"
            for local_ip, public_ip in force_public_node_ip_map.items()
            if local_ip and public_ip
        )
        if not forced_pairs:
            return ""
        return " ".join(
            [
                f"FORCED_RAY_NODE_IP_MAP={shlex.quote(forced_pairs)};",
                'detected_node_ip="$(ip route get 1.1.1.1 2>/dev/null | awk \'/src/ {print $7; exit}\')";',
                'if [ -z "$detected_node_ip" ]; then detected_node_ip="$(hostname -I | awk \'{print $1}\')"; fi;',
                'for forced_node_ip_pair in $FORCED_RAY_NODE_IP_MAP; do',
                'forced_local_ip="${forced_node_ip_pair%%=*}";',
                'forced_public_ip="${forced_node_ip_pair#*=}";',
                'if [ "$detected_node_ip" = "$forced_local_ip" ]; then',
                'if ! ip -o addr show dev lo | grep -q " $forced_public_ip/"; then',
                'sudo ip addr add "$forced_public_ip/32" dev lo;',
                "fi;",
                'echo "[WARN] Bound forced Ray public IP $forced_public_ip to host loopback for Docker host networking.";',
                "fi;",
                "done;",
            ]
        )

    def _ray_start_commands(
        self,
        *,
        root_dir: str,
        workdir_mount_remote: str,
        ray_temp_dir: str,
        force_public_node_ip_map: Mapping[str, str],
    ) -> tuple[List[str], List[str]]:
        env_prefix = f'set -a; source "{workdir_mount_remote}/.env"; set +a;'
        prefix = env_prefix
        if not self.use_docker:
            prefix = " ".join(
                [
                    f'source "{root_dir}/.venv/bin/activate";',
                    env_prefix,
                ]
            )
        filter_busy_gpus_cmd = " ".join(
            [
                prefix,
                f'python3 "{workdir_mount_remote}/scripts/filter_busy_gpus.py"',
                f'--env-file "{workdir_mount_remote}/.env"',
            ]
        ).strip()
        node_ip_arg_cmd = self._node_ip_arg_command(force_public_node_ip_map)
        ray_gcs_port = int(os.getenv("NOBOOM_RAY_GCS_PORT", "6379"))
        ray_dashboard_port = int(os.getenv("NOBOOM_RAY_DASHBOARD_REMOTE_PORT", "8265"))
        head_commands = [
            " ".join([prefix, "ray stop --force"]).strip(),
            filter_busy_gpus_cmd,
            " ".join(
                [
                    prefix,
                    node_ip_arg_cmd,
                    f"RAY_enable_autoscaler_v2=0 ray start --head --port={ray_gcs_port} --dashboard-host=0.0.0.0 "
                    f"--dashboard-port={ray_dashboard_port} --autoscaling-config=~/ray_bootstrap_config.yaml --temp-dir={ray_temp_dir} "
                    "$node_ip_arg --resources='{\"exclusive\": 1}' --labels='{\"role\":\"head\"}'",
                ]
            ).strip(),
        ]
        worker_commands = [
            " ".join([prefix, "ray stop --force"]).strip(),
            filter_busy_gpus_cmd,
            " ".join(
                [
                    prefix,
                    node_ip_arg_cmd,
                    f"RAY_enable_autoscaler_v2=0 ray start --address=$RAY_HEAD_IP:{ray_gcs_port} "
                    f"--temp-dir={ray_temp_dir} $node_ip_arg --resources='{{\"exclusive\": 1}}'",
                ]
            ).strip(),
        ]
        return head_commands, worker_commands

    def _node_ip_arg_command(self, force_public_node_ip_map: Mapping[str, str]) -> str:
        forced_pairs = " ".join(
            f"{local_ip}={public_ip}"
            for local_ip, public_ip in force_public_node_ip_map.items()
            if local_ip and public_ip
        )
        return " ".join(
            [
                f"FORCED_RAY_NODE_IP_MAP={shlex.quote(forced_pairs)};",
                'node_ip_arg="";',
                'detected_node_ip="$(ip route get 1.1.1.1 2>/dev/null | awk \'/src/ {print $7; exit}\')";',
                'if [ -z "$detected_node_ip" ]; then detected_node_ip="$(hostname -I | awk \'{print $1}\')"; fi;',
                'for forced_node_ip_pair in $FORCED_RAY_NODE_IP_MAP; do',
                'forced_local_ip="${forced_node_ip_pair%%=*}";',
                'forced_public_ip="${forced_node_ip_pair#*=}";',
                'if [ "$detected_node_ip" = "$forced_local_ip" ]; then',
                'node_ip_arg="--node-ip-address=$forced_public_ip";',
                "break;",
                "fi;",
                "done;",
            ]
        )


class DockerBackend(DeploymentBackend):
    cluster_name = "noboom-benchmark-docker"
    use_docker = True

    @property
    def template_name(self) -> str:
        return "ray-cluster-docker.yaml.j2"


class NativeBackend(DeploymentBackend):
    cluster_name = "noboom-benchmark-native"
    use_docker = False

    @property
    def template_name(self) -> str:
        return "ray-cluster-native.yaml.j2"


def get_backend(deployment_mode: str) -> DeploymentBackend:
    if deployment_mode == "docker":
        return DockerBackend()
    if deployment_mode == "native":
        return NativeBackend()
    raise ValueError(f"Unsupported deployment mode: {deployment_mode}")


def build_service_settings(
    *,
    controller_settings: Mapping[str, Optional[str]],
    head_local_ip: str,
    root_dir: str,
    storage_path: str,
    mapped_storage: str,
    workdir: str,
    workdir_host: str,
    ray_temp_dir: str,
    mlflow_ui_port: int,
    enable_seaweed: bool,
    use_docker: bool,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> ServiceSettings:
    postgres_password_value = controller_settings.get("POSTGRES_PASSWORD")
    if not postgres_password_value:
        raise RuntimeError(
            "Missing required controller setting: POSTGRES_PASSWORD. "
            "Please define POSTGRES_PASSWORD in the environment."
        )

    s3_endpoint_url = None
    if enable_seaweed:
        s3_endpoint_url = f"http://{head_local_ip}:8333"

    nooboom_s3_bucket = str(controller_settings.get("NOBOOM_S3_BUCKET") or "noboom-ray")
    postgres_user = str(controller_settings.get("POSTGRES_USER") or "noboom")
    postgres_password = str(postgres_password_value)
    mlflow_tracking_uri = str(
        controller_settings.get("MLFLOW_TRACKING_URI") or f"http://{head_local_ip}:5000"
    )
    optuna_storage_uri = str(
        controller_settings.get("OPTUNA_STORAGE_URI")
        or f"postgresql+psycopg://{postgres_user}:{postgres_password}@{head_local_ip}:5432/optuna_db"
    )

    return ServiceSettings(
        head_local_ip=head_local_ip,
        storage_path=storage_path,
        mapped_storage=mapped_storage,
        workdir=workdir,
        workdir_host=workdir_host,
        root_dir=root_dir,
        ray_temp_dir=ray_temp_dir,
        mlflow_ui_port=mlflow_ui_port,
        nooboom_s3_bucket=nooboom_s3_bucket,
        nooboom_s3_prefix=str(controller_settings.get("NOBOOM_S3_PREFIX") or "experiment_data"),
        nooboom_prepared_dataset_s3_path=controller_settings.get("NOBOOM_PREPARED_DATASET_S3_PATH"),
        mlflow_tracking_uri=mlflow_tracking_uri,
        optuna_storage_uri=optuna_storage_uri,
        seafile_username=str(controller_settings.get("SEAFILE_USERNAME") or ""),
        seafile_pass=str(controller_settings.get("SEAFILE_PASS") or ""),
        seafile_root_path=str(controller_settings.get("SEAFILE_ROOT_PATH") or ""),
        kaggle_api_token=controller_settings.get("KAGGLE_API_TOKEN"),
        nooboom_worker_debug=controller_settings.get("NOBOOM_WORKER_DEBUG"),
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        s3_endpoint_url=s3_endpoint_url,
        enable_seaweed=enable_seaweed,
        use_docker=use_docker,
        postgres_user=postgres_user,
        postgres_password=postgres_password,
        ray_enable_autoscaler_v2=str(controller_settings.get("RAY_ENABLE_AUTOSCALER_V2") or "0"),
        deployment_mode="docker" if use_docker else "native",
        nooboom_mlflow_enable_controller_lineage=(
            str(controller_settings.get("NOBOOM_MLFLOW_ENABLE_CONTROLLER_LINEAGE") or "1") == "1"
        ),
        nooboom_mlflow_enable_dataset_tracking=(
            str(controller_settings.get("NOBOOM_MLFLOW_ENABLE_DATASET_TRACKING") or "1") == "1"
        ),
        nooboom_mlflow_enable_logged_models=(
            str(controller_settings.get("NOBOOM_MLFLOW_ENABLE_LOGGED_MODELS") or "0") == "1"
        ),
        nooboom_mlflow_enable_evaluation=(
            str(controller_settings.get("NOBOOM_MLFLOW_ENABLE_EVALUATION") or "0") == "1"
        ),
        nooboom_mlflow_enable_tables=(
            str(controller_settings.get("NOBOOM_MLFLOW_ENABLE_TABLES") or "1") == "1"
        ),
        nooboom_mlflow_enable_system_metrics=(
            str(controller_settings.get("NOBOOM_MLFLOW_ENABLE_SYSTEM_METRICS") or "1") == "1"
        ),
        nooboom_mlflow_enable_registry=(
            str(controller_settings.get("NOBOOM_MLFLOW_ENABLE_REGISTRY") or "0") == "1"
        ),
        nooboom_seafile_upload_checkpoints=(
            str(controller_settings.get("NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS") or "0") == "1"
        ),
        nooboom_seafile_upload_results=(
            str(controller_settings.get("NOBOOM_SEAFILE_UPLOAD_RESULTS") or "0") == "1"
        ),
        nooboom_mlflow_controller_run_id=controller_settings.get("NOBOOM_MLFLOW_CONTROLLER_RUN_ID"),
        nooboom_notify_email_enabled=(
            str(controller_settings.get("NOBOOM_NOTIFY_EMAIL_ENABLED") or "0") == "1"
        ),
        nooboom_notify_smtp_host=str(controller_settings.get("NOBOOM_NOTIFY_SMTP_HOST") or ""),
        nooboom_notify_smtp_port=int(controller_settings.get("NOBOOM_NOTIFY_SMTP_PORT") or 587),
        nooboom_notify_smtp_username=str(
            controller_settings.get("NOBOOM_NOTIFY_SMTP_USERNAME") or ""
        ),
        nooboom_notify_smtp_password=str(
            controller_settings.get("NOBOOM_NOTIFY_SMTP_PASSWORD") or ""
        ),
        nooboom_notify_smtp_from=str(
            controller_settings.get("NOBOOM_NOTIFY_SMTP_FROM")
            or controller_settings.get("NOBOOM_NOTIFY_EMAIL_FROM")
            or ""
        ),
        nooboom_notify_smtp_to=str(
            controller_settings.get("NOBOOM_NOTIFY_SMTP_TO")
            or controller_settings.get("NOBOOM_NOTIFY_EMAIL_TO")
            or ""
        ),
        nooboom_notify_smtp_use_tls=(
            str(controller_settings.get("NOBOOM_NOTIFY_SMTP_USE_TLS") or "1") == "1"
        ),
        nooboom_notify_smtp_use_ssl=(
            str(controller_settings.get("NOBOOM_NOTIFY_SMTP_USE_SSL") or "0") == "1"
        ),
        nooboom_notify_smtp_timeout_s=float(
            controller_settings.get("NOBOOM_NOTIFY_SMTP_TIMEOUT_S") or 10.0
        ),
        nooboom_notify_telegram_enabled=(
            str(controller_settings.get("NOBOOM_NOTIFY_TELEGRAM_ENABLED") or "0") == "1"
        ),
        nooboom_notify_telegram_link_token=str(
            controller_settings.get("NOBOOM_NOTIFY_TELEGRAM_LINK_TOKEN") or ""
        ),
        nooboom_notify_telegram_start_link=str(
            controller_settings.get("NOBOOM_NOTIFY_TELEGRAM_START_LINK") or ""
        ),
        nooboom_notify_telegram_relay_url=str(
            controller_settings.get("NOBOOM_NOTIFY_TELEGRAM_RELAY_URL") or ""
        ),
        nooboom_notify_telegram_relay_secret=str(
            controller_settings.get("NOBOOM_NOTIFY_TELEGRAM_RELAY_SECRET") or ""
        ),
        nooboom_notify_telegram_timeout_s=float(
            controller_settings.get("NOBOOM_NOTIFY_TELEGRAM_TIMEOUT_S") or 10.0
        ),
        nooboom_notify_ram_breach_ratio=str(
            controller_settings.get("NOBOOM_NOTIFY_RAM_BREACH_RATIO") or ""
        ),
    )
