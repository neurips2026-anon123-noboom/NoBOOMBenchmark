from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading
from typing import List, Sequence, Tuple


def run_ssh(ssh_user: str, remote_host: str, ssh_key: str, cmd: str) -> str:
    """Run a remote command via SSH and stream output locally.

    Args:
        ssh_user (str): SSH username.
        remote_host (str): Remote hostname or IP address.
        ssh_key (str): Path to the SSH private key.
        cmd (str): Command to execute on the remote host.

    Returns:
        str: Captured stdout from the remote command.

    Side Effects:
        Spawns an SSH subprocess and streams stdout/stderr to local terminals.
    """
    ssh_target = f"{ssh_user}@{remote_host}"
    ssh_key_path = Path(ssh_key).expanduser()
    p = subprocess.Popen(
        [
            "ssh",
            "-i",
            str(ssh_key_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            ssh_target,
            cmd,  # IMPORTANT: cmd is a single argument, not split
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered in text mode
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
    )

    assert p.stdout is not None
    assert p.stderr is not None

    out_lines: List[str] = []

    def pump_stdout() -> None:
        """Stream stdout lines from the SSH process to local stdout."""
        for line in iter(p.stdout.readline, ""):
            out_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        p.stdout.close()

    def pump_stderr() -> None:
        """Stream stderr lines from the SSH process to local stderr."""
        for line in iter(p.stderr.readline, ""):
            sys.stderr.write(line)
            sys.stderr.flush()
        p.stderr.close()

    t_out = threading.Thread(target=pump_stdout, daemon=True)
    t_err = threading.Thread(target=pump_stderr, daemon=True)

    t_out.start()
    t_err.start()

    p.wait()
    t_out.join()
    t_err.join()

    return "".join(out_lines)


def run_ssh_capture(ssh_user: str, remote_host: str, ssh_key: str, cmd: str) -> str:
    """Run a remote command via SSH and capture stdout without streaming it locally.

    Args:
        ssh_user (str): SSH username.
        remote_host (str): Remote hostname or IP address.
        ssh_key (str): Path to the SSH private key.
        cmd (str): Command to execute on the remote host.

    Returns:
        str: Captured stdout from the remote command.

    Raises:
        subprocess.CalledProcessError: If the SSH command exits with a non-zero status.
    """
    ssh_target = f"{ssh_user}@{remote_host}"
    ssh_key_path = Path(ssh_key).expanduser()
    result = subprocess.run(
        [
            "ssh",
            "-i",
            str(ssh_key_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            ssh_target,
            cmd,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def create_dir(ssh_user: str, remote_host: str, ssh_key: str, path: str) -> None:
    """Create a directory on a remote host via SSH.

    Args:
        ssh_user (str): SSH username.
        remote_host (str): Remote hostname or IP.
        ssh_key (str): Path to the SSH private key.
        path (str): Directory path to create.

    Returns:
        None: The directory is created remotely.

    Side Effects:
        Executes a remote SSH command.
    """
    run_ssh(ssh_user, remote_host, ssh_key, f"mkdir -p {path}; chmod 755 {path} || true")


def reset_ufw(ssh_user: str, remote_host: str, ssh_key: str) -> None:
    """Reset and re-enable UFW on a remote host.

    Args:
        ssh_user (str): SSH username.
        remote_host (str): Remote hostname or IP.
        ssh_key (str): Path to the SSH private key.

    Returns:
        None: UFW rules are reset remotely.

    Side Effects:
        Executes a remote SSH command that modifies firewall configuration.
    """
    run_ssh(
        ssh_user,
        remote_host,
        ssh_key,
        "sudo ufw --force reset && sudo ufw default deny incoming && "
        "sudo ufw default allow outgoing && sudo ufw allow OpenSSH && sudo ufw --force enable",
    )


def remove_old_container(ssh_user: str, remote_host: str, ssh_key: str, name: str) -> None:
    """Remove a Docker container by name on a remote host.

    Args:
        ssh_user (str): SSH username.
        remote_host (str): Remote hostname or IP.
        ssh_key (str): Path to the SSH private key.
        name (str): Container name to remove.

    Returns:
        None: Docker removal is executed remotely.

    Side Effects:
        Executes a remote SSH command and may stop/remove containers.
    """
    run_ssh(ssh_user, remote_host, ssh_key, f"docker rm -f {name} 2>/dev/null || true")


def transfer_files(
    ssh_user: str,
    remote_host: str,
    ssh_key_path: str | Path,
    files: Sequence[Tuple[str, str]],
) -> None:
    """Transfer files to a remote host via SCP.

    Args:
        ssh_user (str): SSH username.
        remote_host (str): Remote hostname or IP.
        ssh_key_path (str | Path): Path to the SSH private key.
        files (Sequence[Tuple[str, str]]): Sequence of (local, remote) file pairs.

    Returns:
        None: Files are transferred to the remote host.

    Side Effects:
        Creates remote directories and executes SCP subprocesses.
    """
    ssh_target = f"{ssh_user}@{remote_host}"
    ssh_key_path = str(Path(ssh_key_path).expanduser())

    for local, remote in files:
        create_dir(ssh_user, remote_host, ssh_key_path, str(Path(remote).parent))

        cmd = [
            "scp",
            "-i",
            ssh_key_path,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            str(Path(local).expanduser()),
            f"{ssh_target}:{remote}",
        ]
        subprocess.run(cmd, check=True)


def get_remote_local_ip(
    host: str,
    ssh_user: str,
    ssh_key_path: str,
) -> str:
    """Query a remote host for its local (LAN) IP using SSH.

    Args:
        host (str): Public hostname or IP of the remote machine.
        ssh_user (str): SSH username.
        ssh_key_path (str): Path to the private SSH key.

    Returns:
        str: Detected local IP (first IPv4 address).

    Raises:
        RuntimeError: If the local IP cannot be determined.

    Side Effects:
        Executes a remote command via SSH.
    """
    cmd = (
        "hostname -I | awk '{print $1}'"
    )
    out = run_ssh(ssh_user, host, ssh_key_path, cmd)
    local_ip = out.strip()
    if not local_ip:
        raise RuntimeError(f"Failed to determine local IP on host {host}")
    return local_ip
