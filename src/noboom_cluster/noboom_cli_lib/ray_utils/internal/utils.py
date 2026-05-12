from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from typing import Iterable, List, Mapping, Optional, Sequence

from .ssh import get_remote_local_ip, run_ssh

logger = logging.getLogger(__name__)

NATIVE_SEAWEED_TMUX_SESSION = "noboom-seaweedfs"
NATIVE_MLFLOW_TMUX_SESSION = "noboom-mlflow"


def run_ray_command(
    command: Sequence[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
) -> int:
    """Run a Ray CLI command and log stdout/stderr.

    Args:
        command (Sequence[str]): Command and arguments to execute.
        env (Optional[Mapping[str, str]]): Environment variables override. Defaults to None.
        check (bool): Whether to raise on non-zero exit. Defaults to True.

    Returns:
        subprocess.CompletedProcess: Result from ``subprocess.run``.

    Raises:
        subprocess.CalledProcessError: If ``check`` is True and the command fails.

    Side Effects:
        Executes a subprocess and logs output.
    """
    logger.info("Running Ray command: %s", shlex.join(command))
    if env is None:
        env = {}
    env["RAY_enable_autoscaler_v2"] = "0"
    result = subprocess.run(command, env=dict(os.environ) | dict(env) if env else dict(os.environ), check=check)
    return result.returncode


def run_ray_exec(
    cluster_yaml: str,
    remote_cmd: str,
    *,
    tmux: bool = False,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
) -> int:
    """Run a shell command via ``ray exec`` without parallelization.

    Args:
        cluster_yaml (str): Path to the Ray cluster YAML config.
        remote_cmd (str): Remote command to execute.
        tmux (bool): Whether to run the command in a tmux session. Defaults to False.
        env (Optional[Mapping[str, str]]): Environment variables override. Defaults to None.
        check (bool): Whether to raise on non-zero exit. Defaults to True.

    Returns:
        subprocess.CompletedProcess: Result of the ``ray exec`` command.

    Side Effects:
        Executes a Ray subprocess.
    """
    cmd = ["ray", "exec", "--run-env", "host"]
    if tmux:
        cmd.append("--tmux")
    cmd.extend([cluster_yaml, remote_cmd])
    return run_ray_command(cmd, env=env, check=check)


def run_named_tmux_session(
    cluster_yaml: str,
    session_name: str,
    command: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
) -> int:
    """Run a command in a deterministic tmux session on the cluster head."""
    session_name_quoted = shlex.quote(session_name)
    inner_command = shlex.quote(command)
    tmux_command = shlex.quote(
        " ".join(
            [
                "set -euo pipefail;",
                f"tmux kill-session -t {session_name_quoted} >/dev/null 2>&1 || true;",
                f"tmux new-session -d -s {session_name_quoted} bash -lc {inner_command};",
            ]
        )
    )
    return run_ray_exec(cluster_yaml, f"bash -lc {tmux_command}", env=env, check=check)


def _extract_pids_from_text(text: str) -> List[int]:
    """Extract PID integers from textual output.

    Args:
        text (str): Text containing process IDs.

    Returns:
        List[int]: List of extracted PID values.
    """
    # Best-effort PID extraction (Ray output format varies).
    return [int(x) for x in re.findall(r"(?<!\d)(\d{2,})(?!\d)", text)]


def _stop_by_pgrep_regex_gracefully(
    cluster_yaml: str,
    root_dir: str,
    pgrep_regex: str,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.5,
    kill_after_timeout: bool = False,
    tmux_sessions: Optional[Sequence[str]] = None,
) -> bool:
    """Stop processes matching a regex via ``ray submit`` and wait for shutdown.

    Args:
        cluster_yaml (str): Path to the Ray cluster YAML config.
        pgrep_regex (str): Regex passed to ``pgrep -f`` for matching processes.
        timeout_s (float): Timeout before giving up. Defaults to 30.0.
        poll_interval_s (float): Poll interval in seconds. Defaults to 0.5.
        kill_after_timeout (bool): Whether to SIGKILL after timeout. Defaults to False.

    Returns:
        bool: True if the matching process set disappears.

    Side Effects:
        Executes remote process termination commands via Ray.
    """
    pgrep_regex = shlex.quote(pgrep_regex)
    cmd = [
        "python3",
        f"{root_dir}/scripts/stop_by_pgrep_regex.py",
        "--pgrep-regex",
        pgrep_regex,
        "--timeout-s",
        str(timeout_s),
        "--poll-interval-s",
        str(poll_interval_s),
    ]
    if kill_after_timeout:
        cmd.append("--kill-after-timeout")
    for session_name in tmux_sessions or ():
        cmd.extend(["--tmux-session", shlex.quote(session_name)])
    result = run_ray_exec(cluster_yaml, " ".join(cmd), check=True)
    return result == 0


def stop_weed_gracefully(
    cluster_yaml: str,
    root_dir: str,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.5,
    kill_after_timeout: bool = False,
) -> bool:
    """Stop SeaweedFS services gracefully on the cluster.

    Args:
        cluster_yaml (str): Path to the Ray cluster YAML config.
        timeout_s (float): Timeout before giving up. Defaults to 30.0.
        poll_interval_s (float): Poll interval in seconds. Defaults to 0.5.
        kill_after_timeout (bool): Whether to SIGKILL after timeout. Defaults to False.

    Returns:
        bool: True if the SeaweedFS processes disappear.
    """
    # weed executable + a server role token to avoid killing "weed version", etc.
    weed_re = r'(^|/)(weed)(\s|$)'
    role_re = r'\b(server|master|volume|filer|s3|iam|mq|replication|mount)\b'
    grep_expr = f"{weed_re}.*{role_re}|{role_re}.*{weed_re}"
    return _stop_by_pgrep_regex_gracefully(
        cluster_yaml=cluster_yaml,
        root_dir=root_dir,
        pgrep_regex=grep_expr,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        kill_after_timeout=kill_after_timeout,
        tmux_sessions=[NATIVE_SEAWEED_TMUX_SESSION],
    )


def stop_mlflow_gracefully(
    cluster_yaml: str,
    root_dir: str,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.5,
    kill_after_timeout: bool = False,
) -> bool:
    """Stop MLflow server processes gracefully on the cluster.

    Args:
        cluster_yaml (str): Path to the Ray cluster YAML config.
        timeout_s (float): Timeout before giving up. Defaults to 30.0.
        poll_interval_s (float): Poll interval in seconds. Defaults to 0.5.
        kill_after_timeout (bool): Whether to SIGKILL after timeout. Defaults to False.

    Returns:
        bool: True if MLflow processes disappear.
    """
    # (mlflow executable OR python -m mlflow) AND "server"
    grep_expr = r'(\bmlflow\b|\bpython(\d+(\.\d+)*)?\b.*\s-m\s+mlflow\b).*\bserver\b'
    return _stop_by_pgrep_regex_gracefully(
        cluster_yaml=cluster_yaml,
        root_dir=root_dir,
        pgrep_regex=grep_expr,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        kill_after_timeout=kill_after_timeout,
        tmux_sessions=[NATIVE_MLFLOW_TMUX_SESSION],
    )


def setup_ray_ufw(
    head_public_ip: str,
    worker_public_ips: Iterable[str],
    ssh_user: str,
    ssh_key_path: str,
    ssh_user_by_host: Optional[Mapping[str, str]] = None,
) -> None:
    """Configure UFW rules for Ray nodes by resolving local IPs.

    Args:
        head_public_ip (str): Public IP or hostname of the head node.
        worker_public_ips (Iterable[str]): Public IPs or hostnames of worker nodes.
        ssh_user (str): SSH username for all nodes.
        ssh_key_path (str): Path to the SSH private key.

    Returns:
        None: UFW rules are applied via remote commands.

    Side Effects:
        Executes SSH commands and modifies UFW configuration on remote hosts.
    """
    if os.getenv("NOBOOM_DISABLE_RAY_UFW_SSH") == "1":
        logger.info(
            "Skipping SSH-based UFW setup because NOBOOM_DISABLE_RAY_UFW_SSH=1."
        )
        return

    worker_public_ips = list(worker_public_ips)
    ssh_user_by_host = dict(ssh_user_by_host or {})
    head_ssh_user = ssh_user_by_host.get(head_public_ip, ssh_user)

    logger.info("Head public IP: %s", head_public_ip)
    logger.info("Worker public IPs: %s", ", ".join(worker_public_ips))

    # 1) Determine head local IP
    logger.info("Detecting head local IP...")
    head_local_ip = get_remote_local_ip(
        host=head_public_ip,
        ssh_user=head_ssh_user,
        ssh_key_path=ssh_key_path
    )
    #ufw_cmd_head = f"sudo ufw allow from {head_public_ip}"
    #run_ssh(ssh_user, head_public_ip, ssh_key_path, ufw_cmd_head)
    logger.info("Head local IP: %s", head_local_ip)

    # 2) Process each worker
    for worker_public_ip in worker_public_ips:
        logger.info("%s", "-" * 60)
        logger.info("Processing worker (public): %s", worker_public_ip)

        # 2a) Detect worker local IP
        logger.info("Detecting worker local IP...")
        worker_local_ip = get_remote_local_ip(
            host=worker_public_ip,
            ssh_user=ssh_user_by_host.get(worker_public_ip, ssh_user),
            ssh_key_path=ssh_key_path,
        )
        logger.info("Worker local IP: %s", worker_local_ip)

        # 2b) On HEAD: allow from worker local and configured/public IP.
        # You can restrict to specific ports by changing the ufw commands here.
        for worker_source_ip in dict.fromkeys([worker_local_ip, worker_public_ip]):
            logger.info("On HEAD: ufw allow from worker IP %s...", worker_source_ip)
            ufw_cmd_head = f"sudo ufw allow from {worker_source_ip}"
            run_ssh(head_ssh_user, head_public_ip, ssh_key_path, ufw_cmd_head)

        # 2c) On WORKER: allow from head local and configured/public IP.
        for head_source_ip in dict.fromkeys([head_local_ip, head_public_ip]):
            logger.info("On WORKER: ufw allow from head IP %s...", head_source_ip)
            ufw_cmd_worker = f"sudo ufw allow from {head_source_ip}"
            run_ssh(
                ssh_user_by_host.get(worker_public_ip, ssh_user),
                worker_public_ip,
                ssh_key_path,
                ufw_cmd_worker,
            )

        logger.info(
            "Finished worker %s (local %s).",
            worker_public_ip,
            worker_local_ip,
        )

    # Optional: show UFW status on head for verification
    logger.info("%s", "-" * 60)
    logger.info("Fetching UFW status from head...")
    status = run_ssh(
        head_ssh_user,
        head_public_ip,
        ssh_key_path,
        "sudo ufw status verbose || sudo ufw status",
    )
    logger.info("UFW status on head:\n%s", status)
