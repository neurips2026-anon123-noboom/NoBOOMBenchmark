from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from noboom_cluster.noboom_cli_lib.deployment import get_backend
from noboom_cluster.noboom_cli_lib.ray_lifecycle import (
    choose_ray_provider_address_plan,
    force_public_node_ip_map_for_cross_region_nodes,
)
from noboom_cluster.noboom_cli_lib.runtime_bundle import build_runtime_bundle
from noboom_cluster.noboom_cli_lib.specs import InventoryConfig


def test_existing_machine_yaml_parses_without_address_extensions(tmp_path: Path) -> None:
    inventory_path = tmp_path / "machines.yaml"
    inventory_path.write_text(
        "nodes:\n"
        "  - ip: 203.0.113.10\n"
        "    devices: 0,1\n"
        "    ssh_user: ubuntu\n"
        "  - ip: 203.0.113.11\n"
        "    devices: \"2\"\n",
        encoding="utf-8",
    )

    inventory = InventoryConfig.load(str(inventory_path))

    assert [node.ip for node in inventory.nodes] == ["203.0.113.10", "203.0.113.11"]
    assert inventory.nodes[0].resolved_ssh_user("cloud") == "ubuntu"


def test_machine_yaml_rejects_stale_ray_ip_schema_field() -> None:
    with pytest.raises(ValidationError):
        InventoryConfig.model_validate(
            {
                "nodes": [
                    {
                        "ip": "203.0.113.10",
                        "ray_ip": "10.0.0.10",
                    }
                ]
            }
        )


def test_configured_ip_is_rendered_as_ray_provider_address() -> None:
    bundle = build_runtime_bundle(project_root=Path.cwd(), deployment_mode="native")
    backend = get_backend("native")

    backend.render_cluster_config(
        bundle,
        head_public_ip="203.0.113.10",
        head_local_ip="10.0.0.10",
        worker_local_ips=["10.0.0.11"],
        ssh_user="cloud",
        ssh_key="~/.ssh/id_ed25519",
        root_dir="/tmp/noboom",
        ray_temp_dir="/tmp/ray",
        storage_path="/tmp/noboom/experiment_data",
        mapped_storage="/tmp/noboom/experiment_data",
        mlflow_ui_port=5001,
        workdir="/tmp/noboom",
        workdir_host_root="/tmp/noboom",
        mount_files_host="/tmp/noboom/mnt",
        ray_head_ip="203.0.113.10",
        ray_worker_ips=["203.0.113.11"],
    )

    cluster_config = yaml.safe_load(bundle.cluster_config_path.read_text(encoding="utf-8"))

    assert cluster_config["provider"]["head_ip"] == "203.0.113.10"
    assert cluster_config["provider"]["worker_ips"] == ["203.0.113.11"]
    assert cluster_config["provider"]["external_head_ip"] == "203.0.113.10"


def test_rendered_ray_start_forces_public_node_ip_only_for_selected_nodes() -> None:
    bundle = build_runtime_bundle(project_root=Path.cwd(), deployment_mode="native")
    backend = get_backend("native")

    backend.render_cluster_config(
        bundle,
        head_public_ip="203.0.113.10",
        head_local_ip="203.0.113.10",
        worker_local_ips=["10.0.0.11"],
        ssh_user="cloud",
        ssh_key="~/.ssh/id_ed25519",
        root_dir="/tmp/noboom",
        ray_temp_dir="/tmp/ray",
        storage_path="/tmp/noboom/experiment_data",
        mapped_storage="/tmp/noboom/experiment_data",
        mlflow_ui_port=5001,
        workdir="/tmp/noboom",
        workdir_host_root="/tmp/noboom",
        mount_files_host="/tmp/noboom/mnt",
        ray_head_ip="203.0.113.10",
        ray_worker_ips=["203.0.113.11"],
        force_public_node_ip_map={"10.0.0.11": "203.0.113.11"},
    )

    cluster_config = yaml.safe_load(bundle.cluster_config_path.read_text(encoding="utf-8"))
    worker_start = cluster_config["worker_start_ray_commands"][2]
    head_start = cluster_config["head_start_ray_commands"][2]

    assert "FORCED_RAY_NODE_IP_MAP=10.0.0.11=203.0.113.11" in worker_start
    assert "--node-ip-address=$forced_public_ip" in worker_start
    assert "detected_node_ip" in worker_start
    assert "FORCED_RAY_NODE_IP_MAP=10.0.0.11=203.0.113.11" in head_start


def test_docker_forced_public_node_ip_uses_loopback_alias() -> None:
    bundle = build_runtime_bundle(project_root=Path.cwd(), deployment_mode="docker")
    backend = get_backend("docker")

    backend.render_cluster_config(
        bundle,
        head_public_ip="203.0.113.10",
        head_local_ip="203.0.113.10",
        worker_local_ips=["10.0.0.11"],
        ssh_user="cloud",
        ssh_key="~/.ssh/id_ed25519",
        root_dir="/tmp/noboom",
        ray_temp_dir="/tmp/ray",
        storage_path="/tmp/noboom/experiment_data",
        mapped_storage="/workspace/noboom/storage",
        mlflow_ui_port=5001,
        workdir="/workspace/noboom",
        workdir_host_root="/tmp/ray_tmp_mount/noboom-benchmark-docker/workspace/noboom",
        mount_files_host="/tmp/ray_tmp_mount/noboom-benchmark-docker/workspace/noboom/mnt",
        ray_head_ip="203.0.113.10",
        ray_worker_ips=["203.0.113.11"],
        force_public_node_ip_map={"10.0.0.11": "203.0.113.11"},
    )

    cluster_config = yaml.safe_load(bundle.cluster_config_path.read_text(encoding="utf-8"))
    initialization_commands = cluster_config["initialization_commands"]

    assert "--network=host" not in cluster_config["docker"]["run_options"]
    assert "--net=host" not in cluster_config["docker"]["run_options"]
    assert any(
        "FORCED_RAY_NODE_IP_MAP=10.0.0.11=203.0.113.11" in command
        and 'ip addr add "$forced_public_ip/32" dev lo' in command
        for command in initialization_commands
    )


def test_force_public_node_ip_map_only_when_selected_address_differs_from_local() -> None:
    local_inventory = InventoryConfig.model_validate(
        {
            "nodes": [
                {"ip": "203.0.113.10"},
                {"ip": "203.0.113.11"},
                {"ip": "192.168.0.16"},
            ]
        }
    )

    forced = force_public_node_ip_map_for_cross_region_nodes(
        local_inventory,
        ["203.0.113.10", "203.0.113.11", "203.0.113.12"],
    )

    assert forced == {"192.168.0.16": "203.0.113.12"}


def test_configured_public_addresses_preferred_over_detected_private() -> None:
    public_inventory = InventoryConfig.model_validate(
        {
            "nodes": [
                {"ip": "203.0.113.10"},
                {"ip": "203.0.113.11"},
            ]
        }
    )
    private_inventory = InventoryConfig.model_validate(
        {
            "nodes": [
                {"ip": "10.0.0.10"},
                {"ip": "10.0.0.11"},
            ]
        }
    )

    plan = choose_ray_provider_address_plan(
        public_inventory,
        private_inventory,
        default_ssh_user="cloud",
        ssh_key="~/.ssh/id_ed25519",
    )

    assert plan.mode == "configured"
    assert plan.head_ip == "203.0.113.10"
    assert plan.worker_ips == ["203.0.113.11"]


def test_private_fallback_warns_when_configured_fails_and_private_validates(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    public_inventory = InventoryConfig.model_validate(
        {
            "nodes": [
                {"ip": "203.0.113.10", "ssh_user": "head-user"},
                {"ip": "203.0.113.11", "ssh_user": "worker-user"},
            ]
        }
    )
    private_inventory = InventoryConfig.model_validate(
        {
            "nodes": [
                {"ip": "10.0.0.10"},
                {"ip": "10.0.0.11"},
            ]
        }
    )
    probes: list[tuple[str, str, str]] = []

    def fake_run_ssh_capture(ssh_user: str, remote_host: str, ssh_key: str, cmd: str) -> str:
        probes.append((ssh_user, remote_host, cmd))
        return ""

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.run_ssh_capture",
        fake_run_ssh_capture,
    )

    with caplog.at_level("WARNING"):
        plan = choose_ray_provider_address_plan(
            public_inventory,
            private_inventory,
            default_ssh_user="cloud",
            ssh_key="~/.ssh/id_ed25519",
            configured_addresses_failed=True,
            failure_reason=RuntimeError("public Ray join failed"),
        )

    assert plan.mode == "private-fallback"
    assert plan.head_ip == "10.0.0.10"
    assert plan.worker_ips == ["10.0.0.11"]
    assert "falling back to validated auto-detected private addresses" in caplog.text
    assert probes[0][0:2] == ("head-user", "203.0.113.10")
    assert "10.0.0.11" in probes[0][2]
    assert probes[1][0:2] == ("worker-user", "203.0.113.11")
    assert "10.0.0.10" in probes[1][2]
    assert "6379" in probes[1][2]


def test_unreachable_configured_and_private_addresses_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_inventory = InventoryConfig.model_validate(
        {
            "nodes": [
                {"ip": "203.0.113.10"},
                {"ip": "203.0.113.11"},
            ]
        }
    )
    private_inventory = InventoryConfig.model_validate(
        {
            "nodes": [
                {"ip": "10.0.0.10"},
                {"ip": "10.0.0.11"},
            ]
        }
    )

    def fake_run_ssh_capture(ssh_user: str, remote_host: str, ssh_key: str, cmd: str) -> str:
        raise subprocess.CalledProcessError(255, ["ssh"])

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.run_ssh_capture",
        fake_run_ssh_capture,
    )

    with pytest.raises(RuntimeError, match="Refusing to start a benchmark"):
        choose_ray_provider_address_plan(
            public_inventory,
            private_inventory,
            default_ssh_user="cloud",
            ssh_key="~/.ssh/id_ed25519",
            configured_addresses_failed=True,
        )
