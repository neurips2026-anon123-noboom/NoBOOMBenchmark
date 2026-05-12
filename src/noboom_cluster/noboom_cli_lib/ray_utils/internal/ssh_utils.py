import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import logging
import socket
import time
from typing import Mapping, Optional
import psutil

logger = logging.getLogger(__name__)


def ensure_ssh_wrapper(
    extra_ssh_opts=None,
    host_user_map: Optional[Mapping[str, str]] = None,
    wrapper_dir: Optional[Path] = None,
) -> Path:
    """Create a cross-platform SSH wrapper that injects extra options.

    Args:
        extra_ssh_opts (Iterable[str] | None): Extra options to prepend to SSH calls.
        wrapper_dir (Path | None): Directory to place the wrapper in. Defaults to a
            temporary directory when None.

    Returns:
        Path: Path to the wrapper script.

    Raises:
        RuntimeError: If the `ssh` binary cannot be found on PATH.

    Side Effects:
        Creates a wrapper script on disk and may set execute permissions.
    """
    # Find the real ssh binary
    real_ssh = shutil.which("ssh")
    if real_ssh is None:
        raise RuntimeError("ssh binary not found in PATH")

    # Directory for the wrapper
    if wrapper_dir is None:
        wrapper_dir = Path(tempfile.mkdtemp(prefix="sshwrap-"))
    else:
        wrapper_dir = Path(wrapper_dir)
        wrapper_dir.mkdir(parents=True, exist_ok=True)

    is_windows = os.name == "nt"

    host_user_map = {
        str(host): str(user)
        for host, user in (host_user_map or {}).items()
        if host and user
    }

    if is_windows and not host_user_map:
        # On Windows, a .cmd wrapper is the safest choice
        wrapper_path = wrapper_dir / "ssh.cmd"
        # %* passes all original args
        opts_str = " ".join(extra_ssh_opts)
        script = f'@echo off\r\n"{real_ssh}" {opts_str} %*\r\n'
        wrapper_path.write_text(script, encoding="utf-8")
        # No chmod needed for .cmd on Windows
    else:
        # POSIX: use Python so we can rewrite user@host targets for mixed-user clusters.
        wrapper_path = wrapper_dir / "ssh"
        extra_opts = list(extra_ssh_opts or [])
        script = f"""#!/usr/bin/env python3
# Auto-generated ssh wrapper.
import os
import sys

REAL_SSH = {real_ssh!r}
EXTRA_OPTS = {extra_opts!r}
HOST_USER_MAP = {host_user_map!r}


def rewrite_target(arg):
    if "@" not in arg:
        return arg
    user, target = arg.rsplit("@", 1)
    host, sep, suffix = target.partition(":")
    mapped_user = HOST_USER_MAP.get(host)
    if not mapped_user:
        return arg
    return f"{{mapped_user}}@{{host}}{{sep}}{{suffix}}"


args = [rewrite_target(arg) for arg in sys.argv[1:]]
os.execv(REAL_SSH, [REAL_SSH, *EXTRA_OPTS, *args])
"""
        wrapper_path.write_text(script, encoding="utf-8")
        # Make executable
        wrapper_path.chmod(
            wrapper_path.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

    return wrapper_path


def activate_ssh_wrapper(extra_ssh_opts=None) -> Path:
    """Create an SSH wrapper and prepend it to the process PATH.

    Args:
        extra_ssh_opts (Iterable[str] | None): Extra options to inject into SSH.

    Returns:
        Path: Path to the wrapper script.

    Side Effects:
        Modifies the current process PATH environment variable.
    """
    wrapper_path = ensure_ssh_wrapper(extra_ssh_opts=extra_ssh_opts)
    wrapper_dir = str(wrapper_path.parent)

    # Prepend to PATH for *this* Python process and all subprocesses
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = wrapper_dir + os.pathsep + old_path

    return wrapper_path


def activate_node_ssh_wrapper(
    *,
    host_user_map: Mapping[str, str],
    extra_ssh_opts=None,
) -> Path:
    """Create an SSH wrapper that rewrites user@host for mixed-user clusters."""
    wrapper_path = ensure_ssh_wrapper(
        extra_ssh_opts=extra_ssh_opts,
        host_user_map=host_user_map,
    )
    wrapper_dir = str(wrapper_path.parent)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = wrapper_dir + os.pathsep + old_path
    logger.info(
        "Activated SSH wrapper with per-host users for %d hosts.",
        len(host_user_map),
    )
    return wrapper_path

def kill_user_ssh(timeout_s: float = 5.0) -> None:
    uid = os.getuid()
    processes = []
    for p in psutil.process_iter(["pid", "name", "uids"]):
        try:
            if (
                p.info["uids"].real == uid and
                p.info["name"] in ("ssh", "sshd")
            ):
                p.terminate()
                processes.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if not processes:
        return

    _, alive = psutil.wait_procs(processes, timeout=timeout_s)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=timeout_s)

def is_port_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_s)
        return s.connect_ex((host, port)) == 0

def forward_ports(
    head_host: str,
    ssh_user: str,
    ssh_key: str,
    local_dashboard: int,
    local_mlflow: int,
    startup_timeout_s: float = 5.0,
) -> None:
    """Start background SSH port forwarding to the Ray dashboard and MLflow UI.

    Args:
        head_host (str): Hostname or IP of the head node.
        ssh_user (str): SSH username.
        ssh_key (str): Path to the SSH private key.
        local_dashboard (int): Local port to bind for the Ray dashboard.
        local_mlflow (int): Local port to bind for the MLflow UI.

    Returns:
        None: This function launches a background process and returns immediately.

    Side Effects:
        Spawns an `ssh` subprocess that keeps port forwards open.
    """

    logger.info(
        "Forwarding Ray dashboard %s and MLflow %s from %s for user %s.",
        local_dashboard,
        local_mlflow,
        head_host,
        ssh_user,
    )

    remote_dashboard = int(os.getenv("NOBOOM_RAY_DASHBOARD_REMOTE_PORT", "8265"))
    process = subprocess.Popen(
        [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            f"{local_dashboard}:127.0.0.1:{remote_dashboard}",
            "-L",
            f"{local_mlflow}:127.0.0.1:5000",
            f"{ssh_user}@{head_host}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "SSH port forwarding exited before the local Ray dashboard and "
                "MLflow ports became ready."
            )
        if (
            is_port_listening("127.0.0.1", local_dashboard)
            and is_port_listening("127.0.0.1", local_mlflow)
        ):
            return
        time.sleep(0.1)

    process.terminate()
    raise RuntimeError(
        "Timed out waiting for local SSH port forwarding to bind the Ray dashboard "
        f"port {local_dashboard} and MLflow port {local_mlflow}."
    )
