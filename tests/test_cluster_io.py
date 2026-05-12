from __future__ import annotations

import json

from noboom_cluster.noboom_cli_lib.cluster_io import (
    MachineNode,
    ensure_ray_ssh_user,
    ensure_ray_ssh_user_on_nodes,
    load_remote_aws_env,
    resolve_nodes_to_local_ips,
)
from noboom_cluster.noboom_cli_lib.ray_utils.internal import utils as ray_internal_utils


def test_load_remote_aws_env_reuses_persisted_seaweed_credentials(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_run_ssh_capture(ssh_user: str, remote_host: str, ssh_key: str, cmd: str) -> str:
        captured["ssh_user"] = ssh_user
        captured["remote_host"] = remote_host
        captured["ssh_key"] = ssh_key
        captured["cmd"] = cmd
        return json.dumps(
            {
                "AWS_ACCESS_KEY_ID": "persisted-access-key",
                "AWS_SECRET_ACCESS_KEY": "persisted-secret-key",
            }
        )

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.cluster_io.run_ssh_capture",
        fake_run_ssh_capture,
    )

    aws_env = load_remote_aws_env(
        head_ip="203.0.113.10",
        ssh_user="cloud",
        ssh_key="~/.ssh/id_ed25519",
        root_dir="~/noboom",
        head_local_ip="10.0.0.10",
        enable_seaweed=True,
    )

    assert aws_env == {
        "AWS_ACCESS_KEY_ID": "persisted-access-key",
        "AWS_SECRET_ACCESS_KEY": "persisted-secret-key",
        "AWS_REGION": "us-east-1",
        "S3_ENDPOINT_URL": "http://10.0.0.10:8333",
        "AWS_EC2_METADATA_DISABLED": "true",
        "MLFLOW_S3_ENDPOINT_URL": "http://10.0.0.10:8333",
    }
    assert captured["ssh_user"] == "cloud"
    assert captured["remote_host"] == "203.0.113.10"
    assert captured["ssh_key"] == "~/.ssh/id_ed25519"
    assert "s3.json" in captured["cmd"]


def test_resolve_nodes_to_local_ips_uses_per_node_ssh_user(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_get_remote_local_ip(host: str, ssh_user: str, ssh_key_path: str) -> str:
        calls.append((host, ssh_user, ssh_key_path))
        return f"10.0.0.{len(calls)}"

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.cluster_io.get_remote_local_ip",
        fake_get_remote_local_ip,
    )

    local_nodes = resolve_nodes_to_local_ips(
        [
            MachineNode(ip="203.0.113.10", devices="0", ssh_user="ubuntu"),
            MachineNode(ip="203.0.113.11", devices="0"),
        ],
        ssh_user="cloud",
        ssh_key="~/.ssh/id_ed25519",
    )

    assert local_nodes == [
        {"ip": "10.0.0.1", "devices": "0", "ssh_user": "ubuntu"},
        {"ip": "10.0.0.2", "devices": "0"},
    ]
    assert calls == [
        ("203.0.113.10", "ubuntu", "~/.ssh/id_ed25519"),
        ("203.0.113.11", "cloud", "~/.ssh/id_ed25519"),
    ]


def test_ensure_ray_ssh_user_uses_configured_user_and_installs_target_user(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_run_ssh_capture(ssh_user: str, remote_host: str, ssh_key: str, cmd: str) -> str:
        captured["ssh_user"] = ssh_user
        captured["remote_host"] = remote_host
        captured["ssh_key"] = ssh_key
        captured["cmd"] = cmd
        return ""

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.cluster_io.run_ssh_capture",
        fake_run_ssh_capture,
    )

    ensure_ray_ssh_user(
        node=MachineNode(ip="203.0.113.11", ssh_user="cloud"),
        default_ssh_user="ubuntu",
        target_ssh_user="ubuntu",
        ssh_key="~/.ssh/id_ed25519",
        public_key="ssh-ed25519 AAAATEST test@example",
        ray_mount_paths=["/tmp/ray_tmp_mount/noboom-benchmark-docker/root/.ray"],
    )

    assert captured["ssh_user"] == "cloud"
    assert captured["remote_host"] == "203.0.113.11"
    assert captured["ssh_key"] == "~/.ssh/id_ed25519"
    assert "TARGET_USER=ubuntu" in captured["cmd"]
    assert "useradd -m -s /bin/bash" in captured["cmd"]
    assert "authorized_keys" in captured["cmd"]
    assert "NOPASSWD:ALL" in captured["cmd"]
    assert "/tmp/ray_tmp_mount/noboom-benchmark-docker/root/.ray" in captured["cmd"]
    assert "sudo chown -R" in captured["cmd"]


def test_ensure_ray_ssh_user_on_nodes_loads_public_key_once(
    monkeypatch,
    tmp_path,
) -> None:
    key_path = tmp_path / "id_ed25519"
    key_path.write_text("private-key-placeholder", encoding="utf-8")
    key_path.with_suffix(key_path.suffix + ".pub").write_text(
        "ssh-ed25519 AAAATEST test@example\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []

    def fake_ensure_ray_ssh_user(**kwargs) -> None:
        calls.append((kwargs["node"].ip, kwargs["public_key"]))

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.cluster_io.ensure_ray_ssh_user",
        fake_ensure_ray_ssh_user,
    )

    ensure_ray_ssh_user_on_nodes(
        nodes=[
            MachineNode(ip="203.0.113.10", ssh_user="ubuntu"),
            MachineNode(ip="203.0.113.11", ssh_user="cloud"),
        ],
        default_ssh_user="ubuntu",
        target_ssh_user="ubuntu",
        ssh_key=str(key_path),
    )

    assert calls == [
        ("203.0.113.10", "ssh-ed25519 AAAATEST test@example"),
        ("203.0.113.11", "ssh-ed25519 AAAATEST test@example"),
    ]


def test_setup_ray_ufw_allows_public_and_local_peer_ips(monkeypatch) -> None:
    monkeypatch.delenv("NOBOOM_DISABLE_RAY_UFW_SSH", raising=False)
    local_ips = {
        "203.0.113.10": "10.0.0.10",
        "203.0.113.11": "10.0.0.11",
    }
    commands: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        ray_internal_utils,
        "get_remote_local_ip",
        lambda host, ssh_user, ssh_key_path: local_ips[host],
    )
    monkeypatch.setattr(
        ray_internal_utils,
        "run_ssh",
        lambda ssh_user, remote_host, ssh_key, cmd: commands.append(
            (ssh_user, remote_host, cmd)
        )
        or "",
    )

    ray_internal_utils.setup_ray_ufw(
        head_public_ip="203.0.113.10",
        worker_public_ips=["203.0.113.11"],
        ssh_user="ubuntu",
        ssh_key_path="~/.ssh/id_ed25519",
        ssh_user_by_host={
            "203.0.113.10": "ubuntu",
            "203.0.113.11": "cloud",
        },
    )

    assert ("ubuntu", "203.0.113.10", "sudo ufw allow from 10.0.0.11") in commands
    assert ("ubuntu", "203.0.113.10", "sudo ufw allow from 203.0.113.11") in commands
    assert ("cloud", "203.0.113.11", "sudo ufw allow from 10.0.0.10") in commands
    assert ("cloud", "203.0.113.11", "sudo ufw allow from 203.0.113.10") in commands
