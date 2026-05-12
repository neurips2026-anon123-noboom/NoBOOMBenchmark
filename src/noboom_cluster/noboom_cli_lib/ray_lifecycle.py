"""Cluster orchestration for Ray lifecycle and job submission."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Optional, Sequence

from .cluster_io import (
    build_aws_env,
    ensure_ray_ssh_user_on_nodes,
    load_remote_aws_env,
    prepare_node_environment,
    transfer_head_files,
    write_local_machine_file,
)
from .allowlist import AllowlistConfig, AllowlistStatus, AllowlistUpdateResponse
from .cluster_validation import (
    ClusterValidationConfig,
    build_expected_cluster_nodes,
    validate_cluster_before_submission,
)
from .deployment import build_service_settings, get_backend, resolve_workdir_host
from .runtime_bundle import RuntimeBundle, build_runtime_bundle, render_template
from .ray_utils import (
    StartScriptConfig,
    forward_ports,
    is_job_server_running,
    kill_user_ssh,
    refresh_live_mlflow_service,
    run_ray_command,
    start_script,
)
from .ray_utils.internal.ssh import run_ssh_capture
from .settings import ControllerSettings
from .specs import InventoryConfig, NodeSpec

logger = logging.getLogger(__name__)

RAY_DASHBOARD_LOCAL_PORT = 8265
RAY_GCS_PORT = int(os.getenv("NOBOOM_RAY_GCS_PORT", "6379"))
PRE_SUBMIT_VALIDATION_TIMEOUT_S = 300.0
PRE_SUBMIT_VALIDATION_INTERVAL_S = 10.0
MLFLOW_UI_LOCAL_PORT = 5001
NOBOOM_MAPPED_STORAGE = "/workspace/noboom/storage"
NOBOOM_DOCKER_WORKDIR = "/workspace/noboom"


@dataclass(frozen=True)
class ClusterRunConfig:
    machine_ip_file: str
    cluster_config: Optional[str]
    datasets: Optional[Sequence[str]]
    models: Optional[Sequence[str]]
    tune: bool
    experiment_id: Optional[str]
    force_restart: bool
    exclusive: bool
    ssh_user: str
    ssh_key: str
    root_dir: str
    ray_temp_dir: str
    dashboard_port: int
    mlflow_port: int
    deployment_mode: str
    verbose: int
    gpus_per_run: float
    max_in_flight: int = 4
    pairs: Optional[Sequence[str]] = None
    hpo_seeds: Optional[Sequence[int]] = None


@dataclass(frozen=True)
class UpdateAllowedConfig:
    machine_ip_file: str
    ssh_user: str
    ssh_key: str
    dashboard_port: int
    mlflow_port: int
    root_dir: str
    ray_temp_dir: str
    deployment_mode: str
    poll_timeout_s: float = 60.0


@dataclass(frozen=True)
class RayProviderAddressPlan:
    head_ip: str
    worker_ips: list[str]
    mode: str


def resolve_storage_paths(root_dir: str, *, use_docker: bool) -> tuple[str, str]:
    storage_path = f"{root_dir}/experiment_data"
    mapped_storage = NOBOOM_MAPPED_STORAGE if use_docker else storage_path
    return storage_path, mapped_storage


def build_node_ssh_user_map(
    public_inventory: InventoryConfig,
    local_inventory: InventoryConfig,
    default_ssh_user: str,
) -> dict[str, str]:
    host_map: dict[str, str] = {}
    for public_node, local_node in zip(public_inventory.nodes, local_inventory.nodes):
        node_ssh_user = public_node.resolved_ssh_user(default_ssh_user)
        host_map[public_node.ip] = node_ssh_user
        host_map[local_node.ip] = node_ssh_user
    return host_map


def ray_host_mount_paths(workdir_host_root: str, *, use_docker: bool) -> list[str]:
    if not use_docker:
        return []
    mount_root = Path(workdir_host_root).parents[1]
    return [
        str(mount_root / "root" / ".ray"),
        str(Path(workdir_host_root)),
    ]


def force_public_node_ip_map_for_cross_region_nodes(
    local_inventory: InventoryConfig,
    selected_ray_addresses: Sequence[str],
) -> dict[str, str]:
    forced: dict[str, str] = {}
    for local_node, selected_address in zip(local_inventory.nodes, selected_ray_addresses):
        if selected_address and selected_address != local_node.ip:
            forced[local_node.ip] = selected_address
    return forced


def configured_ray_provider_address_plan(inventory: InventoryConfig) -> RayProviderAddressPlan:
    return RayProviderAddressPlan(
        head_ip=inventory.nodes[0].ip,
        worker_ips=[node.ip for node in inventory.nodes[1:]],
        mode="configured",
    )


def choose_ray_provider_address_plan(
    public_inventory: InventoryConfig,
    local_inventory: InventoryConfig,
    *,
    default_ssh_user: str,
    ssh_key: str,
    configured_addresses_failed: bool = False,
    failure_reason: Optional[BaseException] = None,
) -> RayProviderAddressPlan:
    """Choose Ray provider addresses without extending the machine YAML schema."""
    configured_plan = configured_ray_provider_address_plan(public_inventory)
    if not configured_addresses_failed:
        return configured_plan

    private_plan = RayProviderAddressPlan(
        head_ip=local_inventory.nodes[0].ip,
        worker_ips=[node.ip for node in local_inventory.nodes[1:]],
        mode="private-fallback",
    )
    if _validate_private_ray_provider_fallback(
        public_inventory,
        local_inventory,
        default_ssh_user=default_ssh_user,
        ssh_key=ssh_key,
        selected_head_ray_address=private_plan.head_ip,
    ):
        logger.warning(
            "Configured Ray provider addresses failed%s; falling back to validated "
            "auto-detected private addresses for Ray provider config.",
            f" ({failure_reason})" if failure_reason else "",
        )
        return private_plan

    raise RuntimeError(
        "Configured Ray provider addresses failed, and auto-detected private "
        "Ray addresses were not validated. Refusing to start a benchmark with "
        "an incomplete Ray cluster."
    )


def _validate_private_ray_provider_fallback(
    public_inventory: InventoryConfig,
    local_inventory: InventoryConfig,
    *,
    default_ssh_user: str,
    ssh_key: str,
    selected_head_ray_address: str,
) -> bool:
    if len(public_inventory.nodes) != len(local_inventory.nodes):
        logger.warning(
            "Cannot validate private Ray fallback: public and private inventories "
            "have different node counts."
        )
        return False

    if len(public_inventory.nodes) <= 1:
        return True

    public_head = public_inventory.nodes[0]
    public_workers = public_inventory.nodes[1:]
    private_workers = local_inventory.nodes[1:]
    head_ssh_user = public_head.resolved_ssh_user(default_ssh_user)

    for public_worker, private_worker in zip(public_workers, private_workers):
        if not _remote_tcp_probe(
            ssh_host=public_head.ip,
            ssh_user=head_ssh_user,
            ssh_key=ssh_key,
            target_host=private_worker.ip,
            target_port=22,
        ):
            logger.warning(
                "Private Ray fallback failed validation: head %s cannot reach "
                "worker private address %s on TCP/22.",
                public_head.ip,
                private_worker.ip,
            )
            return False

        worker_ssh_user = public_worker.resolved_ssh_user(default_ssh_user)
        if not _remote_tcp_probe(
            ssh_host=public_worker.ip,
            ssh_user=worker_ssh_user,
            ssh_key=ssh_key,
            target_host=selected_head_ray_address,
            target_port=RAY_GCS_PORT,
        ):
            logger.warning(
                "Private Ray fallback failed validation: worker %s cannot reach "
                "selected head Ray address %s on TCP/%d.",
                public_worker.ip,
                selected_head_ray_address,
                RAY_GCS_PORT,
            )
            return False

    return True


def _remote_tcp_probe(
    *,
    ssh_host: str,
    ssh_user: str,
    ssh_key: str,
    target_host: str,
    target_port: int,
) -> bool:
    command = (
        "python3 - <<'PY'\n"
        "import socket\n"
        "import sys\n"
        f"target = ({target_host!r}, {target_port})\n"
        "try:\n"
        "    sock = socket.create_connection(target, timeout=5.0)\n"
        "    sock.close()\n"
        "except OSError as exc:\n"
        "    print(f'tcp probe failed: {exc}', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "PY"
    )
    try:
        run_ssh_capture(ssh_user, ssh_host, ssh_key, command)
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        return False
    return True


class ClusterBootstrapper:
    def __init__(self, config: ClusterRunConfig) -> None:
        self._config = config
        self._backend = get_backend(config.deployment_mode)
        self._bundle = build_runtime_bundle(deployment_mode=config.deployment_mode)
        self._controller_settings = ControllerSettings()

    @property
    def bundle(self) -> RuntimeBundle:
        return self._bundle

    def bootstrap(self) -> int:
        kill_user_ssh()
        storage_path, mapped_storage = resolve_storage_paths(
            self._config.root_dir,
            use_docker=self._backend.use_docker,
        )
        workdir = NOBOOM_DOCKER_WORKDIR if self._backend.use_docker else self._config.root_dir

        inventory = InventoryConfig.load(self._config.machine_ip_file)
        head_node = inventory.nodes[0]
        worker_nodes = inventory.nodes[1:]
        head_ip = head_node.ip
        worker_ips = [node.ip for node in worker_nodes]
        head_ssh_user = head_node.resolved_ssh_user(self._config.ssh_user)
        logger.info("Head node: %s | Workers: %s", head_ip, ", ".join(worker_ips))

        write_local_machine_file(
            inventory.nodes,
            ssh_user=self._config.ssh_user,
            ssh_key=self._config.ssh_key,
            output_path=str(self._bundle.machine_file_path),
        )
        local_inventory = InventoryConfig.load(str(self._bundle.machine_file_path))
        head_local_ip = local_inventory.nodes[0].ip
        worker_local_ips = [node.ip for node in local_inventory.nodes[1:]]
        node_ssh_user_map = build_node_ssh_user_map(
            inventory,
            local_inventory,
            self._config.ssh_user,
        )
        provider_address_plan = choose_ray_provider_address_plan(
            inventory,
            local_inventory,
            default_ssh_user=self._config.ssh_user,
            ssh_key=self._config.ssh_key,
        )
        force_public_node_ip_map = force_public_node_ip_map_for_cross_region_nodes(
            local_inventory,
            [provider_address_plan.head_ip, *provider_address_plan.worker_ips],
        )

        job_server_address = f"http://127.0.0.1:{self._config.dashboard_port}"
        forward_ports(
            head_ip,
            head_ssh_user,
            self._config.ssh_key,
            self._config.dashboard_port,
            self._config.mlflow_port,
        )
        job_server_running = (
            False if self._config.force_restart else is_job_server_running(job_server_address)
        )

        workdir_host_root = resolve_workdir_host(
            workdir,
            self._backend.cluster_name,
            self._backend.use_docker,
        )
        mount_files_host = f"{workdir_host_root}/mnt"
        ray_mount_paths = ray_host_mount_paths(
            workdir_host_root,
            use_docker=self._backend.use_docker,
        )
        ensure_ray_ssh_user_on_nodes(
            nodes=inventory.nodes,
            default_ssh_user=self._config.ssh_user,
            target_ssh_user=head_ssh_user,
            ssh_key=self._config.ssh_key,
            ray_mount_paths=ray_mount_paths,
        )

        aws_env = load_remote_aws_env(
            head_ip=head_ip,
            ssh_user=head_ssh_user,
            ssh_key=self._config.ssh_key,
            root_dir=self._config.root_dir,
            head_local_ip=head_local_ip,
            enable_seaweed=True,
        )
        if aws_env is None:
            aws_env, _ = build_aws_env(
                head_local_ip,
                enable_seaweed=True,
                s3_config_path=str(self._bundle.s3_config_path),
            )
        service_settings = build_service_settings(
            controller_settings=self._controller_settings.to_shared_env(),
            head_local_ip=head_local_ip,
            root_dir=self._config.root_dir,
            storage_path=storage_path,
            mapped_storage=mapped_storage,
            workdir=workdir,
            workdir_host=workdir_host_root,
            ray_temp_dir=self._config.ray_temp_dir,
            mlflow_ui_port=self._config.mlflow_port,
            enable_seaweed=True,
            use_docker=self._backend.use_docker,
            aws_access_key_id=aws_env.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_env.get("AWS_SECRET_ACCESS_KEY"),
        )
        controller_env_vars = service_settings.to_env_dict()
        self._backend.render_env_file(self._bundle, service_settings)
        render_template(
            "docker-compose.yaml.j2",
            self._bundle.head_files_dir / "scripts" / "docker-compose.yaml",
            {
                "env_file_path": f"{workdir_host_root}/mnt/.env",
                "head_local_ip": head_local_ip,
                "seaweed_data_path": f"{storage_path}/seaweedfs",
                "s3_config_host_path": f"{self._config.root_dir}/s3.json",
                "postgres_data_path": f"{self._config.root_dir}/postgres",
                "pg_hba_host_path": f"{self._config.root_dir}/scripts/pg_hba.conf",
                "nooboom_s3_bucket": service_settings.nooboom_s3_bucket,
                "nooboom_s3_prefix": service_settings.nooboom_s3_prefix,
                "allowed_hosts": (
                    f"{head_local_ip}:5000,127.0.0.1:5000,"
                    f"127.0.0.1:{self._config.mlflow_port},localhost"
                ),
            },
        )
        self._backend.render_cluster_config(
            self._bundle,
            head_public_ip=head_ip,
            head_local_ip=head_local_ip,
            worker_local_ips=worker_local_ips,
            ssh_user=head_ssh_user,
            ssh_key=self._config.ssh_key,
            root_dir=self._config.root_dir,
            ray_temp_dir=self._config.ray_temp_dir,
            storage_path=storage_path,
            mapped_storage=mapped_storage,
            mlflow_ui_port=self._config.mlflow_port,
            workdir=workdir,
            workdir_host_root=workdir_host_root,
            mount_files_host=mount_files_host,
            ray_head_ip=provider_address_plan.head_ip,
            ray_worker_ips=provider_address_plan.worker_ips,
            force_public_node_ip_map=force_public_node_ip_map,
        )

        if not job_server_running:
            self._prepare_nodes(inventory.nodes)
            transfer_head_files(
                head_ip=head_ip,
                ssh_user=head_ssh_user,
                ssh_key=self._config.ssh_key,
                root_dir=self._config.root_dir,
                local_head_files_dir=str(self._bundle.head_files_dir),
            )
        else:
            logger.info("Ray job server already running; syncing updated files to the live cluster.")
            transfer_head_files(
                head_ip=head_ip,
                ssh_user=head_ssh_user,
                ssh_key=self._config.ssh_key,
                root_dir=self._config.root_dir,
                local_head_files_dir=str(self._bundle.head_files_dir),
            )
            run_ray_command(
                [
                    "ray",
                    "up",
                    "-v",
                    "-y",
                    "--no-restart",
                    "--no-config-cache",
                    str(self._bundle.cluster_config_path),
                ]
            )
            refresh_live_mlflow_service(
                self._bundle.cluster_config_path,
                root_dir=self._config.root_dir,
                workdir_host=workdir_host_root,
                mlflow_port=5000,
                enable_docker=self._backend.use_docker,
            )

        validation_config = ClusterValidationConfig(
            config_path=str(self._bundle.cluster_config_path),
            expected_nodes=build_expected_cluster_nodes(
                inventory.nodes,
                local_inventory.nodes,
                [provider_address_plan.head_ip, *provider_address_plan.worker_ips],
            ),
            gpus_per_run=self._config.gpus_per_run,
            max_in_flight=self._config.max_in_flight,
            datasets=self._config.datasets or (),
            models=self._config.models or (),
            pairs=self._config.pairs or (),
            poll_timeout_s=PRE_SUBMIT_VALIDATION_TIMEOUT_S,
            poll_interval_s=PRE_SUBMIT_VALIDATION_INTERVAL_S,
            head_ssh_host=provider_address_plan.head_ip,
            head_ssh_user=head_ssh_user,
            head_ssh_key=self._config.ssh_key,
        )

        try:
            self._start_benchmark(
                provider_address_plan=provider_address_plan,
                storage_path=storage_path,
                workdir=workdir,
                job_server_address=job_server_address,
                controller_env_vars=controller_env_vars,
                node_ssh_user_map=node_ssh_user_map,
                validation_config=validation_config,
            )
        except Exception as exc:
            if not worker_nodes or provider_address_plan.mode != "configured":
                raise
            fallback_plan = choose_ray_provider_address_plan(
                inventory,
                local_inventory,
                default_ssh_user=self._config.ssh_user,
                ssh_key=self._config.ssh_key,
                configured_addresses_failed=True,
                failure_reason=exc,
            )
            self._backend.render_cluster_config(
                self._bundle,
                head_public_ip=head_ip,
                head_local_ip=head_local_ip,
                worker_local_ips=worker_local_ips,
                ssh_user=head_ssh_user,
                ssh_key=self._config.ssh_key,
                root_dir=self._config.root_dir,
                ray_temp_dir=self._config.ray_temp_dir,
                storage_path=storage_path,
                mapped_storage=mapped_storage,
                mlflow_ui_port=self._config.mlflow_port,
                workdir=workdir,
                workdir_host_root=workdir_host_root,
                mount_files_host=mount_files_host,
                ray_head_ip=fallback_plan.head_ip,
                ray_worker_ips=fallback_plan.worker_ips,
                force_public_node_ip_map=force_public_node_ip_map_for_cross_region_nodes(
                    local_inventory,
                    [fallback_plan.head_ip, *fallback_plan.worker_ips],
                ),
            )
            fallback_validation_config = ClusterValidationConfig(
                config_path=str(self._bundle.cluster_config_path),
                expected_nodes=build_expected_cluster_nodes(
                    inventory.nodes,
                    local_inventory.nodes,
                    [fallback_plan.head_ip, *fallback_plan.worker_ips],
                ),
                gpus_per_run=self._config.gpus_per_run,
                max_in_flight=self._config.max_in_flight,
                datasets=self._config.datasets or (),
                models=self._config.models or (),
                pairs=self._config.pairs or (),
                poll_timeout_s=PRE_SUBMIT_VALIDATION_TIMEOUT_S,
                poll_interval_s=PRE_SUBMIT_VALIDATION_INTERVAL_S,
                head_ssh_host=fallback_plan.head_ip,
                head_ssh_user=head_ssh_user,
                head_ssh_key=self._config.ssh_key,
            )
            self._start_benchmark(
                provider_address_plan=fallback_plan,
                storage_path=storage_path,
                workdir=workdir,
                job_server_address=job_server_address,
                controller_env_vars=controller_env_vars,
                node_ssh_user_map=node_ssh_user_map,
                validation_config=fallback_validation_config,
            )
        return 0

    def _prepare_nodes(self, nodes: Sequence[NodeSpec]) -> None:
        for node in nodes:
            prepare_node_environment(
                node=node,
                ssh_user=self._config.ssh_user,
                ssh_key=self._config.ssh_key,
                root_dir=self._config.root_dir,
                use_docker=self._backend.use_docker,
            )

    def _start_benchmark(
        self,
        *,
        provider_address_plan: RayProviderAddressPlan,
        storage_path: str,
        workdir: str,
        job_server_address: str,
        controller_env_vars: dict[str, str],
        node_ssh_user_map: dict[str, str],
        validation_config: ClusterValidationConfig,
    ) -> None:
        start_script(
            StartScriptConfig(
                ssh_user=node_ssh_user_map.get(
                    provider_address_plan.head_ip,
                    self._config.ssh_user,
                ),
                ssh_key_path=self._config.ssh_key,
                root_dir=self._config.root_dir,
                storage_dir=storage_path,
                workdir=workdir,
                head_ip=provider_address_plan.head_ip,
                worker_ips=provider_address_plan.worker_ips,
                config_path=str(self._bundle.cluster_config_path),
                ray_temp_dir=self._config.ray_temp_dir,
                models=self._config.models or [],
                datasets=self._config.datasets or [],
                pairs=self._config.pairs or [],
                tune=self._config.tune,
                experiment_id=self._config.experiment_id,
                job_server_address=job_server_address,
                verbose=self._config.verbose,
                force_restart=self._config.force_restart,
                exclusive=self._config.exclusive,
                enable_docker=self._backend.use_docker,
                mlflow_port=5000,
                mlflow_local_port=self._config.mlflow_port,
                dashboard_local_port=self._config.dashboard_port,
                gpus_per_run=self._config.gpus_per_run,
                bundle_root=str(self._bundle.root_dir),
                controller_env_vars=controller_env_vars,
                max_in_flight=self._config.max_in_flight,
                hpo_seeds=self._config.hpo_seeds,
                node_ssh_users=node_ssh_user_map,
                pre_submit_validation=lambda: validate_cluster_before_submission(
                    validation_config,
                ),
            )
        )


def run_cluster_job(config: ClusterRunConfig) -> int:
    return ClusterBootstrapper(config).bootstrap()


def update_cluster_allowlist(config: UpdateAllowedConfig) -> AllowlistUpdateResponse:
    backend = get_backend(config.deployment_mode)
    bundle = build_runtime_bundle(deployment_mode=config.deployment_mode)
    inventory = InventoryConfig.load(config.machine_ip_file)
    head_ip = inventory.nodes[0].ip
    head_ssh_user = inventory.nodes[0].resolved_ssh_user(config.ssh_user)

    write_local_machine_file(
        inventory.nodes,
        ssh_user=config.ssh_user,
        ssh_key=config.ssh_key,
        output_path=str(bundle.machine_file_path),
    )
    local_inventory = InventoryConfig.load(str(bundle.machine_file_path))
    head_local_ip = local_inventory.nodes[0].ip
    worker_local_ips = [node.ip for node in local_inventory.nodes[1:]]
    local_allowlist = AllowlistConfig.from_inventory(local_inventory)
    provider_address_plan = choose_ray_provider_address_plan(
        inventory,
        local_inventory,
        default_ssh_user=config.ssh_user,
        ssh_key=config.ssh_key,
    )
    force_public_node_ip_map = force_public_node_ip_map_for_cross_region_nodes(
        local_inventory,
        [provider_address_plan.head_ip, *provider_address_plan.worker_ips],
    )

    storage_path, mapped_storage = resolve_storage_paths(
        config.root_dir,
        use_docker=backend.use_docker,
    )
    workdir = NOBOOM_DOCKER_WORKDIR if backend.use_docker else config.root_dir
    workdir_host_root = resolve_workdir_host(
        workdir,
        backend.cluster_name,
        backend.use_docker,
    )
    mount_files_host = f"{workdir_host_root}/mnt"
    ensure_ray_ssh_user_on_nodes(
        nodes=inventory.nodes,
        default_ssh_user=config.ssh_user,
        target_ssh_user=head_ssh_user,
        ssh_key=config.ssh_key,
        ray_mount_paths=ray_host_mount_paths(workdir_host_root, use_docker=backend.use_docker),
    )

    controller_settings = ControllerSettings()
    aws_env = load_remote_aws_env(
        head_ip=head_ip,
        ssh_user=head_ssh_user,
        ssh_key=config.ssh_key,
        root_dir=config.root_dir,
        head_local_ip=head_local_ip,
        enable_seaweed=True,
    )
    if aws_env is None:
        aws_env, _ = build_aws_env(
            head_local_ip,
            enable_seaweed=True,
            s3_config_path=str(bundle.s3_config_path),
        )
    service_settings = build_service_settings(
        controller_settings=controller_settings.to_shared_env(),
        head_local_ip=head_local_ip,
        root_dir=config.root_dir,
        storage_path=storage_path,
        mapped_storage=mapped_storage,
        workdir=workdir,
        workdir_host=workdir_host_root,
        ray_temp_dir=config.ray_temp_dir,
        mlflow_ui_port=config.mlflow_port,
        enable_seaweed=True,
        use_docker=backend.use_docker,
        aws_access_key_id=aws_env.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_env.get("AWS_SECRET_ACCESS_KEY"),
    )
    backend.render_env_file(bundle, service_settings)
    backend.render_cluster_config(
        bundle,
        head_public_ip=head_ip,
        head_local_ip=head_local_ip,
        worker_local_ips=worker_local_ips,
        ssh_user=head_ssh_user,
        ssh_key=config.ssh_key,
        root_dir=config.root_dir,
        ray_temp_dir=config.ray_temp_dir,
        storage_path=storage_path,
        mapped_storage=mapped_storage,
        mlflow_ui_port=config.mlflow_port,
        workdir=workdir,
        workdir_host_root=workdir_host_root,
        mount_files_host=mount_files_host,
        ray_head_ip=provider_address_plan.head_ip,
        ray_worker_ips=provider_address_plan.worker_ips,
        force_public_node_ip_map=force_public_node_ip_map,
    )

    kill_user_ssh()
    run_ray_command(
        [
            "ray",
            "up",
            "-v",
            "-y",
            "--no-restart",
            "--no-config-cache",
            str(bundle.cluster_config_path),
        ]
    )

    forward_ports(
        head_ip,
        head_ssh_user,
        config.ssh_key,
        config.dashboard_port,
        config.mlflow_port,
    )

    job_server_address = f"http://127.0.0.1:{config.dashboard_port}"
    deadline = time.time() + config.poll_timeout_s
    while time.time() < deadline:
        if is_job_server_running(job_server_address):
            return AllowlistUpdateResponse(
                accepted=True,
                status=AllowlistStatus(
                    revision=time.time_ns(),
                    allowed=local_allowlist.to_node_map(),
                ),
            )
        time.sleep(1.0)

    raise RuntimeError(f"Ray job server is not reachable at {job_server_address}.")
