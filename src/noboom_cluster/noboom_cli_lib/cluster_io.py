"""Cluster IO helpers (node discovery, file transfer, port forwarding)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
from typing import Iterable, List, Optional, Sequence

import yaml

from .cluster_env import prepare_cluster_env_file
from .ray_utils.cluster_yaml import (get_remote_local_ip,
                                                                  preprocess_cluster_yaml,
                                                                  get_resolved_workdir)
from .ray_utils.internal.env import write_seaweedfs_s3_auth_json
from .ray_utils.internal.ssh import remove_old_container, reset_ufw, run_ssh_capture
from .ray_utils.internal.ssh_utils import activate_ssh_wrapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MachineNode:
    ip: str
    devices: Optional[str] = None
    ssh_user: Optional[str] = None

    def resolved_ssh_user(self, default_ssh_user: str) -> str:
        return self.ssh_user or default_ssh_user


def resolve_node_ssh_user(node: object, default_ssh_user: str) -> str:
    ssh_user = getattr(node, "ssh_user", None)
    return str(ssh_user) if ssh_user else default_ssh_user


def read_ip_list(path: str) -> List[MachineNode]:
    """Load cluster node definitions from a YAML file.

    Args:
        path (str): Path to the YAML file containing a `nodes` list.

    Returns:
        List[MachineNode]: Parsed list of machine nodes (head + workers).

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML structure does not match the expected schema.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"IP list file not found: {file_path}")

    logger.info("Reading cluster nodes from %s", file_path)
    with file_path.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    if not isinstance(content, dict):
        raise ValueError(f"Cluster node file must be a YAML mapping, got: {type(content).__name__}")
    nodes = content.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Cluster node file must define a non-empty 'nodes' list.")

    parsed_nodes: List[MachineNode] = []
    for index, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            raise ValueError(f"Node entry {index} must be a mapping, got: {type(node).__name__}")
        ip = node.get("ip")
        if not isinstance(ip, str) or not ip.strip():
            raise ValueError(f"Node entry {index} must include a non-empty 'ip' string.")
        devices = node.get("devices")
        if devices is not None and not isinstance(devices, str):
            raise ValueError(f"Node entry {index} 'devices' must be a string when provided.")
        ssh_user = node.get("ssh_user")
        if ssh_user is not None and not isinstance(ssh_user, str):
            raise ValueError(f"Node entry {index} 'ssh_user' must be a string when provided.")
        parsed_nodes.append(MachineNode(ip=ip, devices=devices, ssh_user=ssh_user))

    logger.info("Discovered %d nodes (head + workers).", len(parsed_nodes))
    return parsed_nodes


def write_local_machine_file(
    nodes: Iterable[MachineNode],
    ssh_user: str,
    ssh_key: str,
    output_path: str,
) -> str:
    """Write a machine file with local IPs for each node.

    Args:
        nodes (Iterable[MachineNode]): Nodes with public IPs and optional device mappings.
        ssh_user (str): SSH username.
        ssh_key (str): SSH key path.
        output_path (str): Destination path for the local-IP machine file.

    Returns:
        str: Path to the written machine file.
    """
    local_nodes = resolve_nodes_to_local_ips(nodes, ssh_user=ssh_user, ssh_key=ssh_key)

    data = {"nodes": local_nodes}
    Path(output_path).write_text(yaml.safe_dump(data, sort_keys=False))
    return output_path


def resolve_nodes_to_local_ips(
    nodes: Iterable[MachineNode],
    *,
    ssh_user: str,
    ssh_key: str,
) -> List[dict[str, str]]:
    local_nodes = []
    for node in nodes:
        node_ssh_user = resolve_node_ssh_user(node, ssh_user)
        local_ip = get_remote_local_ip(node.ip, node_ssh_user, ssh_key)
        entry = {"ip": local_ip}
        if node.devices:
            entry["devices"] = node.devices
        if getattr(node, "ssh_user", None):
            entry["ssh_user"] = node_ssh_user
        local_nodes.append(entry)
    return local_nodes


def load_ssh_public_key(ssh_key: str) -> str:
    """Load the public key matching an SSH private key without exposing the private key."""
    key_path = Path(ssh_key).expanduser()
    public_key_path = Path(f"{key_path}.pub")
    if public_key_path.is_file():
        public_key = public_key_path.read_text(encoding="utf-8").strip()
    else:
        result = subprocess.run(
            ["ssh-keygen", "-y", "-f", str(key_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        public_key = result.stdout.strip()
    if not public_key:
        raise RuntimeError(f"Unable to load public key for SSH key {key_path}.")
    return public_key


def ensure_ray_ssh_user(
    *,
    node: MachineNode,
    default_ssh_user: str,
    target_ssh_user: str,
    ssh_key: str,
    public_key: str,
    ray_mount_paths: Sequence[str] = (),
) -> None:
    """Ensure Ray's global SSH user exists and can log in on a node."""
    node_ssh_user = resolve_node_ssh_user(node, default_ssh_user)
    logger.info(
        "Ensuring Ray SSH user %s exists on node %s using configured user %s.",
        target_ssh_user,
        node.ip,
        node_ssh_user,
    )
    if node_ssh_user == target_ssh_user:
        logger.info(
            "Configured SSH user already matches Ray SSH user on node %s; skipping sudo user setup.",
            node.ip,
        )
        return
    mount_path_entries = " ".join(shlex.quote(path) for path in ray_mount_paths if path)
    script = "\n".join(
        [
            "set -euo pipefail",
            f"TARGET_USER={shlex.quote(target_ssh_user)}",
            f"PUBLIC_KEY={shlex.quote(public_key)}",
            f"RAY_MOUNT_PATHS=({mount_path_entries})",
            'sudo -n true',
            'if ! id -u "$TARGET_USER" >/dev/null 2>&1; then',
            '  sudo useradd -m -s /bin/bash "$TARGET_USER"',
            "fi",
            'TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"',
            'if [ -z "$TARGET_HOME" ]; then',
            '  echo "Unable to resolve target user home" >&2',
            "  exit 1",
            "fi",
            'sudo install -d -m 700 -o "$TARGET_USER" -g "$TARGET_USER" "$TARGET_HOME/.ssh"',
            'AUTH_KEYS="$TARGET_HOME/.ssh/authorized_keys"',
            'TMP_KEY="$(mktemp)"',
            'trap \'rm -f "$TMP_KEY"\' EXIT',
            'printf "%s\\n" "$PUBLIC_KEY" > "$TMP_KEY"',
            'sudo touch "$AUTH_KEYS"',
            'if ! sudo grep -qxF "$PUBLIC_KEY" "$AUTH_KEYS"; then',
            '  sudo tee -a "$AUTH_KEYS" < "$TMP_KEY" >/dev/null',
            "fi",
            'sudo chown "$TARGET_USER:$TARGET_USER" "$AUTH_KEYS"',
            'sudo chmod 600 "$AUTH_KEYS"',
            'SUDOERS_FILE="/etc/sudoers.d/noboom-ray-$TARGET_USER"',
            'printf "%s ALL=(ALL) NOPASSWD:ALL\\n" "$TARGET_USER" | sudo tee "$SUDOERS_FILE" >/dev/null',
            'sudo chmod 440 "$SUDOERS_FILE"',
            'if getent group docker >/dev/null 2>&1; then',
            '  sudo usermod -aG docker "$TARGET_USER"',
            "fi",
            'for RAY_MOUNT_PATH in "${RAY_MOUNT_PATHS[@]}"; do',
            '  sudo install -d -m 755 -o "$TARGET_USER" -g "$TARGET_USER" "$RAY_MOUNT_PATH"',
            '  sudo chown -R "$TARGET_USER:$TARGET_USER" "$RAY_MOUNT_PATH"',
            "done",
        ]
    )
    run_ssh_capture(
        node_ssh_user,
        node.ip,
        ssh_key,
        f"bash -lc {shlex.quote(script)}",
    )


def ensure_ray_ssh_user_on_nodes(
    *,
    nodes: Iterable[MachineNode],
    default_ssh_user: str,
    target_ssh_user: str,
    ssh_key: str,
    ray_mount_paths: Sequence[str] = (),
) -> None:
    public_key = load_ssh_public_key(ssh_key)
    for node in nodes:
        ensure_ray_ssh_user(
            node=node,
            default_ssh_user=default_ssh_user,
            target_ssh_user=target_ssh_user,
            ssh_key=ssh_key,
            public_key=public_key,
            ray_mount_paths=ray_mount_paths,
        )


def build_aws_env(head_local_ip: str, *, enable_seaweed: bool, s3_config_path: str) -> tuple[dict[str, str], Optional[str]]:
    """Build AWS/SeaweedFS environment variables and optional config path.

    Args:
        head_local_ip (str): Local IP of the head node hosting SeaweedFS.
        enable_seaweed (bool): Whether SeaweedFS is enabled.
        s3_config_path (str): Path to write the SeaweedFS S3 config JSON.

    Returns:
        tuple[dict[str, str], str | None]: Env mapping and config path (or None).

    Side Effects:
        Writes SeaweedFS credentials JSON when SeaweedFS is enabled.
    """
    if not enable_seaweed:
        return {}, None

    logger.info("Generating SeaweedFS S3 credentials at %s", s3_config_path)
    access_key, secret_key = write_seaweedfs_s3_auth_json(s3_config_path, overwrite=True)
    aws_env = {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "AWS_REGION": "us-east-1",
        "S3_ENDPOINT_URL": f"http://{head_local_ip}:8333",
        "AWS_EC2_METADATA_DISABLED": "true",
        "MLFLOW_S3_ENDPOINT_URL": f"http://{head_local_ip}:8333",
    }
    return aws_env, s3_config_path


def load_remote_aws_env(
    *,
    head_ip: str,
    ssh_user: str,
    ssh_key: str,
    root_dir: str,
    head_local_ip: str,
    enable_seaweed: bool,
) -> Optional[dict[str, str]]:
    """Load persisted SeaweedFS credentials from the head node when available.

    Args:
        head_ip (str): Public IP of the head node.
        ssh_user (str): SSH username.
        ssh_key (str): Path to the SSH private key.
        root_dir (str): Root directory on the head node.
        head_local_ip (str): Local IP of the head node hosting SeaweedFS.
        enable_seaweed (bool): Whether SeaweedFS is enabled.

    Returns:
        Optional[dict[str, str]]: AWS-compatible SeaweedFS environment variables, or
            ``None`` when no persisted credentials are available.
    """
    if not enable_seaweed:
        return {}

    python_snippet = (
        "import json; "
        "from pathlib import Path; "
        f"path = Path({root_dir!r}).expanduser() / 's3.json'; "
        "payload = json.loads(path.read_text(encoding='utf-8')); "
        "identities = payload.get('identities') or []; "
        "credentials = (identities[0].get('credentials') or [])[0]; "
        "print(json.dumps({"
        "'AWS_ACCESS_KEY_ID': credentials['accessKey'], "
        "'AWS_SECRET_ACCESS_KEY': credentials['secretKey']"
        "}))"
    )
    remote_cmd = f"python3 -c {shlex.quote(python_snippet)}"

    try:
        payload = run_ssh_capture(ssh_user, head_ip, ssh_key, remote_cmd).strip()
    except subprocess.CalledProcessError:
        logger.info(
            "No persisted SeaweedFS credentials found at %s/s3.json on %s.",
            root_dir,
            head_ip,
        )
        return None

    try:
        credentials = json.loads(payload)
        access_key = str(credentials["AWS_ACCESS_KEY_ID"]).strip()
        secret_key = str(credentials["AWS_SECRET_ACCESS_KEY"]).strip()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to parse persisted SeaweedFS credentials from %s/s3.json on %s: %s",
            root_dir,
            head_ip,
            exc,
        )
        return None

    if not access_key or not secret_key:
        logger.warning(
            "Persisted SeaweedFS credentials at %s/s3.json on %s are incomplete.",
            root_dir,
            head_ip,
        )
        return None

    logger.info("Reusing persisted SeaweedFS credentials from %s/s3.json.", root_dir)
    return {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "AWS_REGION": "us-east-1",
        "S3_ENDPOINT_URL": f"http://{head_local_ip}:8333",
        "AWS_EC2_METADATA_DISABLED": "true",
        "MLFLOW_S3_ENDPOINT_URL": f"http://{head_local_ip}:8333",
    }


def prepare_node_environment(
    *,
    node: MachineNode,
    ssh_user: str,
    ssh_key: str,
    root_dir: str,
    use_docker: bool,
) -> None:
    """Prepare environment files and prerequisites on a single node.

    Args:
        node (MachineNode): Node metadata containing IP and device mapping.
        ssh_user (str): SSH username.
        ssh_key (str): Path to the SSH private key.
        root_dir (str): Root directory on the node.
        use_docker (bool): Whether Docker is used for services.

    Returns:
        None: Environment is prepared via remote commands and file transfers.

    Side Effects:
        Resets UFW, creates directories, and removes stale containers.
    """
    node_ssh_user = resolve_node_ssh_user(node, ssh_user)
    logger.info("Preparing environment for node %s as %s", node.ip, node_ssh_user)
    if use_docker:
        reset_ufw(node_ssh_user, node.ip, ssh_key)
    if use_docker:
        remove_old_container(node_ssh_user, node.ip, ssh_key, "ray-noboom-container")


def transfer_head_files(
    *,
    head_ip: str,
    ssh_user: str,
    ssh_key: str,
    root_dir: str,
    local_head_files_dir: str,
):
    """Transfer head-node scripts and prepared configuration files.

    Args:
        head_ip (str): IP address of the head node.
        worker_ips (Sequence[str]): Worker node IPs for cluster YAML generation.
        ssh_user (str): SSH username.
        ssh_key (str): Path to the SSH private key.
        root_dir (str): Root directory on the head node.
        workdir (str): Work directory used by Ray containers.
        mapped_storage (str): Container-mapped storage path.
        storage_path (str): Storage path on the host.
        cluster_config (str): Path to the Ray cluster YAML template.
        use_docker (bool): Whether Docker is used on the cluster.
        s3_config_path (str | None): Path to generated S3 config, if any.
        s3_config_filename (str): Filename to use for the S3 config on the head node.
        env_file_path (str): Path to the prepared `.env` file.
        machine_file_path (str): Path to the local-IP machine file.
        aws_env (Mapping[str, str]): AWS/SeaweedFS environment variables.
        enable_seaweed (bool): Whether to enable SeaweedFS configuration.
        mlflow_ui_port (int): MLflow UI port to expose.

    Returns:
        None: Transfers files to the head node.

    Side Effects:
        Writes a preprocessed cluster YAML file and uploads files via rsync.
    """

    ssh_key_path = str(Path(ssh_key).expanduser())
    local_expanded = str(Path(local_head_files_dir).resolve()) + "/"
    ssh_target = f"{ssh_user}@{head_ip}:{root_dir+"/"}"
    rsync_cmd = [
        "rsync",
        "--progress",
        "-az",
        "-e",
        f"ssh -i {ssh_key_path} -o BatchMode=yes -o StrictHostKeyChecking=no",
        local_expanded,
        ssh_target,
    ]
    subprocess.run(rsync_cmd, check=True)


def prepare_cluster_environment(
    *,
    nodes: Iterable[MachineNode],
    head_ip: str,
    worker_ips: Sequence[str],
    ssh_user: str,
    ssh_key: str,
    root_dir: str,
    ray_temp_dir: str,
    storage_path: str,
    mapped_storage: str,
    workdir: str,
    cluster_config: str,
    env_template_path: str,
    enable_seaweed: bool,
    use_docker: bool,
    mlflow_ui_port: int,
    s3_config_filename: str,
) -> None:
    """Prepare environment files and scripts across all cluster nodes.

    Args:
        nodes (Iterable[MachineNode]): Nodes participating in the cluster.
        head_ip (str): IP of the head node.
        worker_ips (Sequence[str]): IPs of worker nodes.
        ssh_user (str): SSH username for all nodes.
        ssh_key (str): Path to the SSH private key.
        root_dir (str): Root directory on the nodes.
        storage_path (str): Storage path on the nodes.
        mapped_storage (str): Container-mapped storage path.
        workdir (str): Work directory used for Ray.
        cluster_config (str): Path to the Ray cluster YAML template.
        env_template_path (str): Path to the `.env` template file.
        machine_ip_file (str): Path to the cluster machine YAML file.
        enable_seaweed (bool): Whether to enable SeaweedFS configuration.
        use_docker (bool): Whether Docker is used on the nodes.
        mlflow_ui_port (int): MLflow UI port to expose.
        s3_config_filename (str): File name for the SeaweedFS config.

    Returns:
        None: All nodes are prepared and head files transferred.

    Raises:
        FileNotFoundError: If the env template path does not exist.

    Side Effects:
        Modifies environment variables, runs SSH commands, and transfers files.
    """
    if not os.path.exists(env_template_path):
        raise FileNotFoundError(f".env file not found at {env_template_path}")

    nodes_list = list(nodes)
    os.environ["HEAD_IP"] = head_ip

    logger.debug("Preparing SSH wrapper and environment files for nodes.")
    activate_ssh_wrapper(extra_ssh_opts=["-o", "StrictHostKeyChecking=no"])

    head_local_ip = get_remote_local_ip(head_ip, ssh_user, ssh_key)
    aws_env, s3_config_path = build_aws_env(
        head_local_ip,
        enable_seaweed=enable_seaweed,
        s3_config_path=s3_config_filename,
    )
    env_file_path = Path(env_template_path).with_suffix("")
    env_file_path = str(env_file_path.parent / "mount_files" / env_file_path.name)

    workdir_host = get_resolved_workdir(workdir, cluster_config=cluster_config, use_docker=use_docker)

    prepare_cluster_env_file(
        env_template_path,
        env_file_path,
        head_local_ip,
        aws_env,
        root_dir=root_dir,
        storage_path=storage_path,
        mapped_storage=mapped_storage,
        workdir=workdir,
        workdir_host=workdir_host,
        enable_seaweed=enable_seaweed,
        use_docker=use_docker,
        mlflow_ui_port=mlflow_ui_port,
        ray_temp_dir=ray_temp_dir,
    )
    scripts_root = Path(__file__).resolve().parent / "scripts"
    machine_file_path = str(scripts_root / "mount_files" / "machine_nodes.yaml")
    write_local_machine_file(
        nodes_list,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        output_path=machine_file_path,
    )
    for node in nodes_list:
        prepare_node_environment(
            node=node,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            root_dir=root_dir,
            use_docker=use_docker,
        )

    preprocess_cluster_yaml(
        yaml_path=cluster_config,
        head_ip=head_ip,
        worker_ips=worker_ips,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        root_dir=root_dir,
        workdir=workdir,
        ray_temp_dir=ray_temp_dir,
        storage_path=mapped_storage,
        remote_storage_path=storage_path,
        aws_env=aws_env,
        enable_seaweed=enable_seaweed,
        mlflow_ui_port=mlflow_ui_port,
        use_docker=use_docker,
    )

    transfer_head_files(
        head_ip=head_ip,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        root_dir=root_dir,
        local_head_files_dir=str(scripts_root / "head_files"),
    )
