from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import shlex
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ray.job_submission import JobSubmissionClient

from .internal.ssh_utils import forward_ports, kill_user_ssh
from .internal.utils import (
    NATIVE_MLFLOW_TMUX_SESSION,
    NATIVE_SEAWEED_TMUX_SESSION,
    run_named_tmux_session,
    run_ray_command,
    run_ray_exec,
    setup_ray_ufw,
    stop_mlflow_gracefully,
    stop_weed_gracefully,
)
from .cluster_yaml import get_resolved_workdir
from ..runtime_env import build_uv_runtime


logger = logging.getLogger(__name__)
DEFAULT_JOB_SERVER_TIMEOUT_S = 5.0


def _local_ray_auth_token() -> Optional[str]:
    token_path = Path.home() / ".ray" / "auth_token"
    if not token_path.is_file():
        return None

    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        return None

    return token


def _job_submission_client(address: str) -> JobSubmissionClient:
    token = _local_ray_auth_token()
    headers = {"authorization": f"Bearer {token}"} if token else None
    return JobSubmissionClient(address=address, headers=headers)


@dataclass(frozen=True)
class StartScriptConfig:
    ssh_user: str
    ssh_key_path: str | Path
    root_dir: str
    storage_dir: str
    workdir: str
    head_ip: str
    worker_ips: Sequence[str]
    config_path: str | Path
    ray_temp_dir: str
    models: Sequence[str]
    datasets: Sequence[str]
    tune: bool
    experiment_id: Optional[str]
    job_server_address: str
    gpus_per_run: int
    verbose: int = 0
    force_restart: bool = False
    exclusive: bool = False
    enable_docker: bool = True
    mlflow_port: int = 5000
    mlflow_local_port: int = 5001
    dashboard_local_port: int = 8265
    bundle_root: Optional[str] = None
    controller_env_vars: Optional[Mapping[str, str]] = None
    max_in_flight: int = 4
    pairs: Sequence[str] = ()
    save_checkpoints: bool = False
    hpo_seeds: Optional[Sequence[int]] = None
    node_ssh_users: Optional[Mapping[str, str]] = None
    pre_submit_validation: Optional[Callable[[], None]] = None


def _ssh_user_for_host(config: StartScriptConfig, host: str) -> str:
    if config.node_ssh_users and host in config.node_ssh_users:
        return config.node_ssh_users[host]
    return config.ssh_user


def _start_native_services(
    config_path: str | Path,
    *,
    root_dir: str,
    workdir_host: str,
    storage_dir: str,
    mlflow_port: int,
) -> None:
    """Start Postgres, SeaweedFS, and MLflow directly on the head node.

    Args:
        config_path (str | Path): Ray cluster config path for `ray exec`.
        root_dir (str): Root directory on the head node.
        storage_dir (str): Storage directory for SeaweedFS volumes.
        mlflow_port (int): Port for the MLflow server.
        env (Mapping[str, str]): Environment variables to inherit for commands.

    Returns:
        None: Services are started via remote commands.

    Raises:
        RuntimeError: If SeaweedFS fails health checks after startup.

    Side Effects:
        Executes Ray commands, starts tmux sessions, and restarts services.
    """

    stop_weed_gracefully(str(config_path), root_dir=root_dir, kill_after_timeout=True)
    stop_mlflow_gracefully(str(config_path), root_dir=root_dir, kill_after_timeout=True)

    postgres_data_dir = f"{root_dir}/postgres/pgsql/data"
    postgres_log_file = f"{root_dir}/postgres/logfile"
    postgres_shell_cmd = " ".join(
        [
            "set -euo pipefail;",
            f"source \"{root_dir}/.venv/bin/activate\";",
            f"set -a && source \"{workdir_host}/mnt/.env\" && set +a;",
            "export PGPASSWORD=${POSTGRES_PASSWORD};",
            f"export POSTGRES_DATA_DIR={shlex.quote(postgres_data_dir)};",
            f"export POSTGRES_LOG_FILE={shlex.quote(postgres_log_file)};",
            f"cp {shlex.quote(f'{workdir_host}/scripts/pg_hba.conf')} \"$POSTGRES_DATA_DIR/pg_hba.conf\";",
            "test -d \"$POSTGRES_DATA_DIR\";",
            "pg_ctl -D \"$POSTGRES_DATA_DIR\" -m fast stop >/dev/null 2>&1 || true;",
            "POSTGRES_PID=$(ss -ltnp | awk '/:5432 / && /pid=/ {match($0,/pid=([0-9]+)/,m); if (m[1] != \"\") print m[1]}' | head -n1);",
            "if [ -n \"$POSTGRES_PID\" ]; then kill \"$POSTGRES_PID\" >/dev/null 2>&1 || true; sleep 2; fi;",
            "pg_ctl -D \"$POSTGRES_DATA_DIR\" -l \"$POSTGRES_LOG_FILE\" "
            "-o \"-c listen_addresses='*' -p 5432\" start;",
            "for i in {1..30}; do "
            "pg_isready -h 127.0.0.1 -p 5432 -U \"${POSTGRES_USER:-noboom}\" && break; "
            "sleep 1; "
            "if [ \"$i\" -eq 30 ]; then exit 1; fi; "
            "done;",
        ]
    )
    postgres_cmd = shlex.quote(postgres_shell_cmd)

    seaweed_shell_cmd = " ".join(
        [
            "set -euo pipefail;",
            f"source \"{root_dir}/.venv/bin/activate\";",
            f"set -a && source \"{root_dir}/mnt/.env\" && set +a &&",
            f"mkdir -p {storage_dir}/seaweedfs && "
            f"weed server -ip=${{HEAD_LOCAL_IP:-0.0.0.0}} -master.port=9333 -volume.port=8080 -volume.max=64 -filer "
            f"-filer.port=8888 -s3 -s3.port=8333 -s3.config={root_dir}/s3.json -dir={storage_dir}/seaweedfs",
        ]
    )
    mlflow_shell_cmd = (
        " ".join(
            [
                "set -euo pipefail;",
                f"source \"{root_dir}/.venv/bin/activate\" &&",
                f"set -a && source \"{workdir_host}/mnt/.env\" && set +a &&",
                f"mlflow server --backend-store-uri ${{MLFLOW_STORAGE_URI}} --default-artifact-root "
                f"s3://${{NOBOOM_S3_BUCKET}}/${{NOBOOM_S3_PREFIX}}/artifacts --host 0.0.0.0 "
                f"--port {mlflow_port} --serve-artifacts --allowed-hosts "
                f"${{HEAD_LOCAL_IP}}:{mlflow_port},127.0.0.1:{mlflow_port},"
                f"127.0.0.1:${{MLFLOW_UI_LOCAL_PORT}},localhost",
            ]
        )
    )

    run_ray_exec(
        str(config_path),
        f"bash -lc {postgres_cmd}",
    )

    run_named_tmux_session(
        str(config_path),
        NATIVE_SEAWEED_TMUX_SESSION,
        seaweed_shell_cmd,
    )

    health_cmd = (
        f"set -a && source \"{workdir_host}/mnt/.env\" && set +a && "
        "for i in {1..30}; do "
        "curl -fsS http://${HEAD_LOCAL_IP}:8333/status && exit 0; "
        "sleep 5; "
        "done; exit 1"
    )
    res = run_ray_exec(
        str(config_path),
        health_cmd,
    )
    if res != 0:
        raise RuntimeError("Seaweedfs failed to start.")
    run_named_tmux_session(
        str(config_path),
        NATIVE_MLFLOW_TMUX_SESSION,
        mlflow_shell_cmd,
    )

    logger.info(
        "Native services started via named tmux sessions: noboom-seaweedfs, noboom-mlflow.",
    )


def refresh_live_mlflow_service(
    config_path: str | Path,
    *,
    root_dir: str,
    workdir_host: str,
    mlflow_port: int,
    enable_docker: bool,
) -> None:
    """Refresh MLflow on a live cluster after syncing updated env files.

    This path is used when the Ray job server is already running. We regenerate
    the head-node `.env` from the synced `.env.base` and restart only MLflow so
    artifact access picks up new S3-related environment variables.
    """

    logger.info("Refreshing MLflow service on the live cluster.")

    setup_env_script = shlex.quote(f"{workdir_host}/mnt/scripts/setup_env.py")
    setup_env_cmd = shlex.quote(
        " ".join(
            [
                "set -euo pipefail;",
                f"python3 {setup_env_script};",
            ]
        )
    )
    run_ray_exec(str(config_path), f"bash -lc {setup_env_cmd}")

    if enable_docker:
        env_file = shlex.quote(f"{workdir_host}/mnt/.env")
        compose_file = shlex.quote(f"{root_dir}/scripts/docker-compose.yaml")
        restart_cmd = shlex.quote(
            " ".join(
                [
                    "set -euo pipefail;",
                    f"set -a && source {env_file} && set +a;",
                    "if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then",
                    f"docker compose -f {compose_file} up -d --no-deps --force-recreate mlflow;",
                    "elif command -v docker-compose >/dev/null 2>&1; then",
                    f"docker-compose -f {compose_file} up -d --no-deps --force-recreate mlflow;",
                    "else",
                    "echo '[ERROR] Neither docker compose nor docker-compose found in PATH.' >&2;",
                    "exit 1;",
                    "fi;",
                ]
            )
        )
        run_ray_exec(str(config_path), f"bash -lc {restart_cmd}")
        return

    stop_mlflow_gracefully(str(config_path), root_dir=root_dir, kill_after_timeout=True)
    mlflow_shell_cmd = " ".join(
        [
            "set -euo pipefail;",
            f"source \"{root_dir}/.venv/bin/activate\" &&",
            f"set -a && source \"{workdir_host}/mnt/.env\" && set +a &&",
            f"mlflow server --backend-store-uri ${{MLFLOW_STORAGE_URI}} --default-artifact-root "
            f"s3://${{NOBOOM_S3_BUCKET}}/${{NOBOOM_S3_PREFIX}}/artifacts --host 0.0.0.0 "
            f"--port {mlflow_port} --serve-artifacts --allowed-hosts "
            f"${{HEAD_LOCAL_IP}}:{mlflow_port},127.0.0.1:{mlflow_port},"
            f"127.0.0.1:${{MLFLOW_UI_LOCAL_PORT}},localhost",
        ]
    )
    run_named_tmux_session(
        str(config_path),
        NATIVE_MLFLOW_TMUX_SESSION,
        mlflow_shell_cmd,
    )


def is_job_server_running(
    address: str,
    request_timeout_s: float = DEFAULT_JOB_SERVER_TIMEOUT_S,
) -> bool:
    """Check whether a Ray job server is reachable.

    Args:
        address (str): Ray job server address, including protocol and port.

    Returns:
        bool: True if the job server responds, otherwise False.
    """
    try:
        response = _job_submission_client(address=address)._do_request(
            "GET",
            "/api/jobs/",
            timeout=(request_timeout_s, request_timeout_s),
        )
        if response.status_code == 200:
            return True
        logger.info(
            "Ray job server probe at %s returned HTTP %s.",
            address,
            response.status_code,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "Ray job server not reachable at %s within %.1fs: %s",
            address,
            request_timeout_s,
            exc,
        )
    return False


def start_script(config: StartScriptConfig) -> None:
    """Start or reuse a Ray cluster and submit the benchmark job.

    Args:
        config (StartScriptConfig): Configuration for cluster startup and job submission.

    Returns:
        None: Submits the job to the Ray job server.

    Side Effects:
        Runs SSH commands, starts/stops Ray services, and submits a Ray job.
    """
    # Build CLI args for run_tune.py
    mode_label = "tune" if config.tune else "single-train"
    experiment_label = (
        f"source_experiment_id={config.experiment_id}"
        if config.experiment_id and not config.tune
        else "no source_experiment_id"
    )
    logger.info("Preparing %s run (%s).", mode_label, experiment_label)
    args: List[str] = []
    if config.models:
        args.extend(["--model"] + list(config.models))
    if config.datasets:
        args.extend(["--dataset"] + list(config.datasets))
    for pair in config.pairs:
        args.extend(["--pair", pair])
    if config.tune:
        args.append("--tune")
    if config.experiment_id and not config.tune:
        args.extend(["--experiment-id", config.experiment_id])
    args.extend(["--temp-dir", config.ray_temp_dir])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.extend(["--timestamp", timestamp])
    args.extend(["--gpus-per-run", str(config.gpus_per_run)])
    args.extend(["--max-in-flight", str(config.max_in_flight)])
    for seed in config.hpo_seeds or ():
        args.extend(["--hpo-seed", str(seed)])
    if config.save_checkpoints:
        args.append("--save-checkpoints")
    if config.verbose:
        args.extend(["-v"] * config.verbose)

    workdir_host = get_resolved_workdir(
        config.workdir,
        cluster_config=config.config_path,
        use_docker=config.enable_docker,
    )
    if not config.enable_docker:
        args.extend(["--env-file", f"{workdir_host}/mnt/.env"])

    job_server_running = False
    if not config.force_restart:
        job_server_running = is_job_server_running(config.job_server_address)
        if not job_server_running:
            logger.info("Ray job server not reachable, will (re)start it.")

    if config.force_restart or not job_server_running:
        logger.info("Starting Ray cluster%s.", " (forced)" if config.force_restart else "")
        if config.enable_docker:
            setup_ray_ufw(
                config.head_ip,
                config.worker_ips,
                config.ssh_user,
                config.ssh_key_path,
                ssh_user_by_host=config.node_ssh_users,
            )
        else:
            logger.info("Skipping Docker and firewall setup for native deployment mode.")

        logger.info("Generating Ray auth token and bringing up the cluster.")
        run_ray_command(["ray", "down", "-v", "-y", str(config.config_path)])
        run_ray_command(["ray", "get-auth-token", "--generate"])
        for i in range(2):
            try:
                run_ray_command(["ray", "up", "-v", "-y", "--no-config-cache", str(config.config_path)])
                break
            # In case of docker group setup, we need to relogin
            except Exception as e:
                kill_user_ssh()
                forward_ports(
                    config.head_ip,
                    config.ssh_user,
                    config.ssh_key_path,
                    config.dashboard_local_port,
                    config.mlflow_local_port,
                )
                logger.warning("Failed to bring up Ray cluster: %s", e)
                if i == 1:
                    raise e

        selected_ray_ips = [config.head_ip, *list(config.worker_ips)]

        setup_head_cmd = shlex.quote(
            " ".join(
                [
                    "set -euo pipefail;",
                    f"set -a && source \"{workdir_host}/mnt/.env\" && set +a &&",
                    f"bash {config.root_dir}/scripts/setup_head.sh {' '.join(selected_ray_ips)}",
                ]
            )
        )

        run_ray_exec(str(config.config_path), f"bash -lc {setup_head_cmd}")

        if not config.enable_docker:
            _start_native_services(
                config.config_path,
                root_dir=config.root_dir,
                workdir_host=workdir_host,
                storage_dir=config.storage_dir,
                mlflow_port=config.mlflow_port,
            )
    else:
        logger.info("Ray job server already running; skipping cluster start.")
    if config.pre_submit_validation is not None:
        config.pre_submit_validation()
    client = _job_submission_client(address=config.job_server_address)
    if config.exclusive:
        logger.info("Exclusive mode enabled; stopping all active Ray jobs before submission.")
        jobs = client.list_jobs()
        for job in jobs:
            if job.status.is_terminal():
                continue
            logger.info("Stopping job %s (status=%s)", job.job_id, job.status)
            client.stop_job(job.job_id)

    def build_runtime_env(exclusive: bool, timestamp: str) -> Dict[str, Any]:
        env_vars = dict(config.controller_env_vars or {})
        env_vars.setdefault("EXPERIMENT_NAME", f"NoBoomBenchmark__{timestamp}")
        env_vars["NOBOOM_JOB_ROLE"] = "controller"
        env_vars["NOBOOM_CONTROLLER_SUBMISSION_ID"] = f"main_tune_job__{timestamp}"
        if exclusive:
            env_vars["NOBOOM_EXCLUSIVE"] = "1"
        runtime_env: Dict[str, Any] = {
            "env_vars": env_vars,
            "uv": build_uv_runtime(),
        }
        if config.bundle_root:
            runtime_env["working_dir"] = config.bundle_root
        return runtime_env

    runtime_env = build_runtime_env(config.exclusive, timestamp)
    telegram_link = runtime_env["env_vars"].get("NOBOOM_NOTIFY_TELEGRAM_START_LINK")
    if telegram_link:
        logger.info("Telegram notifications enrollment link: %s", telegram_link)
    logger.info("Executing noboom_benchmark.run_tune with args: %s", " ".join(args))

    client.submit_job(
        entrypoint=(f"python -m noboom_benchmark.run_tune {' '.join(args)}"),
        submission_id=f"main_tune_job__{timestamp}",
        runtime_env=runtime_env,
    )
