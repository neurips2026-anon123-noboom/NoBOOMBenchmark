from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from noboom_cluster import cli
from noboom_cluster.noboom_cli_lib.allowlist import (
    AllowlistConfig,
    AllowlistNode,
    AllowlistStatus,
    AllowlistUpdateResponse,
    ClusterNodeGpuView,
    TrialGpuLease,
)
from noboom_cluster.noboom_cli_lib.ray_lifecycle import (
    ClusterRunConfig,
    UpdateAllowedConfig,
    run_cluster_job,
    update_cluster_allowlist,
)
from noboom_cluster.noboom_cli_lib.runtime_bundle import RuntimeBundle
from noboom_cluster.noboom_cli_lib.specs import InventoryConfig
from noboom_benchmark.noboom_lib.core.tune import runtime_allowlist


def test_update_cluster_allowlist_normalizes_local_ips_and_reestablishes_forwarding(
    monkeypatch,
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.yaml"
    inventory_path.write_text(
        "nodes:\n"
        "  - ip: 203.0.113.10\n"
        "    devices: 0,1\n"
        "  - ip: 203.0.113.11\n"
        "    devices: \"2\"\n",
        encoding="utf-8",
    )

    calls = {
        "forwarded": [],
        "ray_commands": [],
    }
    bundle_root = tmp_path / "bundle"
    mount_files_dir = bundle_root / "mount"
    head_files_dir = bundle_root / "head"
    runtime_dir = bundle_root / "runtime"
    mount_files_dir.mkdir(parents=True)
    head_files_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)

    fake_bundle = RuntimeBundle(
        root_dir=bundle_root,
        runtime_dir=runtime_dir,
        pair_specs_dir=runtime_dir / "pair_specs",
        mount_files_dir=mount_files_dir,
        head_files_dir=head_files_dir,
        cluster_config_path=runtime_dir / "ray-cluster-native.yaml",
        env_base_path=mount_files_dir / ".env.base",
        machine_file_path=mount_files_dir / "machine_nodes.yaml",
        s3_config_path=head_files_dir / "s3.json",
    )

    class FakeBackend:
        cluster_name = "noboom-benchmark-native"
        use_docker = False

        def render_env_file(self, bundle: RuntimeBundle, service_settings: object) -> Path:
            calls["service_settings"] = service_settings
            bundle.env_base_path.write_text("HEAD_LOCAL_IP=10.0.0.10\n", encoding="utf-8")
            return bundle.env_base_path

        def render_cluster_config(self, bundle: RuntimeBundle, **kwargs: object) -> Path:
            calls["cluster_config_kwargs"] = kwargs
            bundle.cluster_config_path.write_text("cluster_name: test\n", encoding="utf-8")
            return bundle.cluster_config_path

    def fake_write_local_machine_file(nodes, ssh_user: str, ssh_key: str, output_path: str) -> str:
        Path(output_path).write_text(
            "nodes:\n"
            "  - ip: 10.0.0.10\n"
            "    devices: 0,1\n"
            "  - ip: 10.0.0.11\n"
            "    devices: \"2\"\n",
            encoding="utf-8",
        )
        calls["write_local_machine_file"] = {
            "nodes": [(node.ip, node.devices) for node in nodes],
            "ssh_user": ssh_user,
            "ssh_key": ssh_key,
            "output_path": output_path,
        }
        return output_path

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.build_runtime_bundle",
        lambda deployment_mode: fake_bundle,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.get_backend",
        lambda deployment_mode: FakeBackend(),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.write_local_machine_file",
        fake_write_local_machine_file,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.ensure_ray_ssh_user_on_nodes",
        lambda **kwargs: calls.setdefault("ensure_ray_ssh_user_on_nodes", kwargs),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.ControllerSettings",
        lambda: SimpleNamespace(to_shared_env=lambda: {}),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.build_aws_env",
        lambda *args, **kwargs: ({}, None),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.load_remote_aws_env",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.build_service_settings",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.resolve_workdir_host",
        lambda workdir, cluster_name, use_docker: workdir,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.kill_user_ssh",
        lambda: calls.__setitem__("kill_user_ssh", calls.get("kill_user_ssh", 0) + 1),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.run_ray_command",
        lambda command: calls["ray_commands"].append(command),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.forward_ports",
        lambda head_ip, ssh_user, ssh_key, dashboard_port, mlflow_port: calls["forwarded"].append(
            (head_ip, ssh_user, ssh_key, dashboard_port, mlflow_port)
        ),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.is_job_server_running",
        lambda address: True,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.time.time_ns",
        lambda: 2,
    )

    result = update_cluster_allowlist(
        UpdateAllowedConfig(
            machine_ip_file=str(inventory_path),
            ssh_user="cloud",
            ssh_key="~/.ssh/id_ed25519",
            dashboard_port=8265,
            mlflow_port=5001,
            root_dir="~/noboom",
            ray_temp_dir="/tmp/ray",
            deployment_mode="native",
        )
    )

    assert result == AllowlistUpdateResponse(
        accepted=True,
        status=AllowlistStatus(
            revision=2,
            allowed={
                "10.0.0.10": ["0", "1"],
                "10.0.0.11": ["2"],
            },
        ),
    )
    assert AllowlistConfig.from_inventory(
        InventoryConfig.load(str(fake_bundle.machine_file_path))
    ).to_node_map() == {
        "10.0.0.10": ["0", "1"],
        "10.0.0.11": ["2"],
    }
    assert calls["kill_user_ssh"] == 1
    assert calls["forwarded"] == [("203.0.113.10", "cloud", "~/.ssh/id_ed25519", 8265, 5001)]
    assert calls["ray_commands"] == [
        [
            "ray",
            "up",
            "-v",
            "-y",
            "--no-restart",
            "--no-config-cache",
            str(fake_bundle.cluster_config_path),
        ]
    ]
    assert calls["cluster_config_kwargs"]["head_local_ip"] == "10.0.0.10"
    assert calls["cluster_config_kwargs"]["worker_local_ips"] == ["10.0.0.11"]
    assert calls["ensure_ray_ssh_user_on_nodes"]["target_ssh_user"] == "cloud"


def test_cli_update_allowed_command_returns_error_on_rejected_update(monkeypatch, tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.yaml"
    inventory_path.write_text("nodes:\n  - ip: 203.0.113.10\n", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "update_cluster_allowlist",
        lambda config: AllowlistUpdateResponse(accepted=False, errors=["bad request"]),
    )

    exit_code = cli.main(
        [
            "update-allowed",
            "--machine-ip-file",
            str(inventory_path),
        ]
    )

    assert exit_code == 1


def test_cli_main_routes_top_level_options_to_run_command(monkeypatch, tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.yaml"
    inventory_path.write_text("nodes:\n  - ip: 203.0.113.10\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_execute_run(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_execute_run", fake_execute_run)

    exit_code = cli.main(
        [
            "--machine-ip-file",
            str(inventory_path),
            "--dataset",
            "ome",
            "--model",
            "gdn",
        ]
    )

    assert exit_code == 0
    assert captured["machine_ip_file"] == str(inventory_path)
    assert captured["dataset"] == ["ome"]
    assert captured["model"] == ["gdn"]


def test_run_cluster_job_forwards_ports_before_probing_job_server(
    monkeypatch,
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.yaml"
    inventory_path.write_text(
        "nodes:\n"
        "  - ip: 203.0.113.10\n"
        "  - ip: 203.0.113.11\n",
        encoding="utf-8",
    )

    bundle_root = tmp_path / "bundle"
    mount_files_dir = bundle_root / "mount"
    head_files_dir = bundle_root / "head"
    runtime_dir = bundle_root / "runtime"
    mount_files_dir.mkdir(parents=True)
    head_files_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)

    fake_bundle = RuntimeBundle(
        root_dir=bundle_root,
        runtime_dir=runtime_dir,
        pair_specs_dir=runtime_dir / "pair_specs",
        mount_files_dir=mount_files_dir,
        head_files_dir=head_files_dir,
        cluster_config_path=runtime_dir / "ray-cluster-native.yaml",
        env_base_path=mount_files_dir / ".env.base",
        machine_file_path=mount_files_dir / "machine_nodes.yaml",
        s3_config_path=head_files_dir / "s3.json",
    )

    events: list[str] = []

    class FakeBackend:
        cluster_name = "noboom-benchmark-native"
        use_docker = False

        def render_env_file(self, bundle: RuntimeBundle, service_settings: object) -> Path:
            bundle.env_base_path.write_text("HEAD_LOCAL_IP=10.0.0.10\n", encoding="utf-8")
            return bundle.env_base_path

        def render_cluster_config(self, bundle: RuntimeBundle, **kwargs: object) -> Path:
            bundle.cluster_config_path.write_text("cluster_name: test\n", encoding="utf-8")
            return bundle.cluster_config_path

    local_ips = iter(["10.0.0.10", "10.0.0.11"])

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.build_runtime_bundle",
        lambda deployment_mode: fake_bundle,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.get_backend",
        lambda deployment_mode: FakeBackend(),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.ControllerSettings",
        lambda: SimpleNamespace(to_shared_env=lambda: {}),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.build_aws_env",
        lambda *args, **kwargs: ({}, None),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.load_remote_aws_env",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.build_service_settings",
        lambda **kwargs: SimpleNamespace(
            to_env_dict=lambda: {},
            nooboom_s3_bucket="bucket",
            nooboom_s3_prefix="prefix",
        ),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.resolve_workdir_host",
        lambda workdir, cluster_name, use_docker: workdir,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.render_template",
        lambda *args, **kwargs: None,
    )

    def fake_write_local_machine_file(*args, **kwargs) -> str:
        output_path = Path(kwargs["output_path"])
        output_path.write_text(
            "nodes:\n"
            f"  - ip: {next(local_ips)}\n"
            f"  - ip: {next(local_ips)}\n",
            encoding="utf-8",
        )
        return str(output_path)

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.write_local_machine_file",
        fake_write_local_machine_file,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.ensure_ray_ssh_user_on_nodes",
        lambda **kwargs: events.append("ensure-user"),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.kill_user_ssh",
        lambda: events.append("kill"),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.forward_ports",
        lambda *args, **kwargs: events.append("forward"),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.is_job_server_running",
        lambda address: events.append("probe") or True,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.transfer_head_files",
        lambda **kwargs: events.append("transfer"),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.run_ray_command",
        lambda command: events.append("ray-up"),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.refresh_live_mlflow_service",
        lambda config_path, **kwargs: events.append("refresh-mlflow"),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.start_script",
        lambda config: events.append("start"),
    )

    exit_code = run_cluster_job(
        ClusterRunConfig(
            machine_ip_file=str(inventory_path),
            cluster_config=None,
            datasets=["ome"],
            models=["gdn"],
            tune=False,
            experiment_id=None,
            force_restart=False,
            exclusive=False,
            ssh_user="cloud",
            ssh_key="~/.ssh/id_ed25519",
            root_dir="~/noboom",
            ray_temp_dir="/tmp/ray",
            dashboard_port=8265,
            mlflow_port=5001,
            deployment_mode="native",
            verbose=0,
            gpus_per_run=0.125,
        )
    )

    assert exit_code == 0
    assert events == [
        "kill",
        "forward",
        "probe",
        "ensure-user",
        "transfer",
        "ray-up",
        "refresh-mlflow",
        "start",
    ]


def test_run_cluster_job_reuses_persisted_seaweed_credentials_for_live_cluster(
    monkeypatch,
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.yaml"
    inventory_path.write_text(
        "nodes:\n"
        "  - ip: 203.0.113.10\n",
        encoding="utf-8",
    )

    bundle_root = tmp_path / "bundle"
    mount_files_dir = bundle_root / "mount"
    head_files_dir = bundle_root / "head"
    runtime_dir = bundle_root / "runtime"
    mount_files_dir.mkdir(parents=True)
    head_files_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)

    fake_bundle = RuntimeBundle(
        root_dir=bundle_root,
        runtime_dir=runtime_dir,
        pair_specs_dir=runtime_dir / "pair_specs",
        mount_files_dir=mount_files_dir,
        head_files_dir=head_files_dir,
        cluster_config_path=runtime_dir / "ray-cluster-native.yaml",
        env_base_path=mount_files_dir / ".env.base",
        machine_file_path=mount_files_dir / "machine_nodes.yaml",
        s3_config_path=head_files_dir / "s3.json",
    )

    recorded: dict[str, object] = {}

    class FakeBackend:
        cluster_name = "noboom-benchmark-native"
        use_docker = False

        def render_env_file(self, bundle: RuntimeBundle, service_settings: object) -> Path:
            bundle.env_base_path.write_text("HEAD_LOCAL_IP=10.0.0.10\n", encoding="utf-8")
            recorded["service_settings"] = service_settings
            return bundle.env_base_path

        def render_cluster_config(self, bundle: RuntimeBundle, **kwargs: object) -> Path:
            bundle.cluster_config_path.write_text("cluster_name: test\n", encoding="utf-8")
            return bundle.cluster_config_path

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.build_runtime_bundle",
        lambda deployment_mode: fake_bundle,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.get_backend",
        lambda deployment_mode: FakeBackend(),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.ControllerSettings",
        lambda: SimpleNamespace(
            to_shared_env=lambda: {
                "POSTGRES_PASSWORD": "postgres-password",
            }
        ),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.resolve_workdir_host",
        lambda workdir, cluster_name, use_docker: workdir,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.render_template",
        lambda *args, **kwargs: None,
    )

    def fake_write_local_machine_file(*args, **kwargs) -> str:
        output_path = Path(kwargs["output_path"])
        output_path.write_text("nodes:\n  - ip: 10.0.0.10\n", encoding="utf-8")
        return str(output_path)

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.write_local_machine_file",
        fake_write_local_machine_file,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.ensure_ray_ssh_user_on_nodes",
        lambda **kwargs: recorded.setdefault("ensure_ray_ssh_user_on_nodes", kwargs),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.kill_user_ssh",
        lambda: None,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.forward_ports",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.is_job_server_running",
        lambda address: True,
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.load_remote_aws_env",
        lambda **kwargs: {
            "AWS_ACCESS_KEY_ID": "persisted-access-key",
            "AWS_SECRET_ACCESS_KEY": "persisted-secret-key",
            "AWS_REGION": "us-east-1",
            "S3_ENDPOINT_URL": "http://10.0.0.10:8333",
            "AWS_EC2_METADATA_DISABLED": "true",
            "MLFLOW_S3_ENDPOINT_URL": "http://10.0.0.10:8333",
        },
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.build_aws_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("build_aws_env should not run")),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.transfer_head_files",
        lambda **kwargs: recorded.setdefault("transfer_head_files", []).append(kwargs),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.run_ray_command",
        lambda command: recorded.setdefault("ray_commands", []).append(command),
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.refresh_live_mlflow_service",
        lambda config_path, **kwargs: recorded.setdefault(
            "refresh_live_mlflow_service",
            [],
        ).append((str(config_path), kwargs)),
    )

    def fake_start_script(config) -> None:
        recorded["controller_env_vars"] = dict(config.controller_env_vars or {})

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.ray_lifecycle.start_script",
        fake_start_script,
    )

    exit_code = run_cluster_job(
        ClusterRunConfig(
            machine_ip_file=str(inventory_path),
            cluster_config=None,
            datasets=["ome"],
            models=["gdn"],
            tune=False,
            experiment_id=None,
            force_restart=False,
            exclusive=False,
            ssh_user="cloud",
            ssh_key="~/.ssh/id_ed25519",
            root_dir="~/noboom",
            ray_temp_dir="/tmp/ray",
            dashboard_port=8265,
            mlflow_port=5001,
            deployment_mode="native",
            verbose=0,
            gpus_per_run=0.125,
        )
    )

    assert exit_code == 0
    assert recorded["controller_env_vars"]["AWS_ACCESS_KEY_ID"] == "persisted-access-key"
    assert recorded["controller_env_vars"]["AWS_SECRET_ACCESS_KEY"] == "persisted-secret-key"
    assert recorded["transfer_head_files"]
    assert recorded["ray_commands"] == [
        [
            "ray",
            "up",
            "-v",
            "-y",
            "--no-restart",
            "--no-config-cache",
            str(fake_bundle.cluster_config_path),
        ]
    ]
    assert recorded["refresh_live_mlflow_service"] == [
        (
            str(fake_bundle.cluster_config_path),
            {
                "root_dir": "~/noboom",
                "workdir_host": "~/noboom",
                "mlflow_port": 5000,
                "enable_docker": False,
            },
        )
    ]


def test_runtime_allowlist_prune_dead_leases_keeps_live_trials(monkeypatch) -> None:
    state = runtime_allowlist.RuntimeGpuAllowlistState()
    state.initialize(
        [
            ClusterNodeGpuView(
                ip="10.0.0.10",
                node_id="node-a",
                visible_gpu_ids=["0", "1"],
            )
        ],
        AllowlistConfig(nodes=[AllowlistNode(ip="10.0.0.10", devices=["0", "1"])]),
    )
    state._active_leases = {
        "dead-owned": TrialGpuLease(
            trial_id="dead-owned",
            ip="10.0.0.10",
            node_id="node-a",
            gpu_id="0",
            gpu_fraction=0.5,
            revision=1,
            owner_job_id="job-dead",
        ),
        "live-owned": TrialGpuLease(
            trial_id="live-owned",
            ip="10.0.0.10",
            node_id="node-a",
            gpu_id="1",
            gpu_fraction=0.5,
            revision=1,
            owner_job_id="job-live",
        ),
        "live-ownerless": TrialGpuLease(
            trial_id="live-ownerless",
            ip="10.0.0.10",
            node_id="node-a",
            gpu_id="0",
            gpu_fraction=0.25,
            revision=1,
        ),
        "dead-ownerless": TrialGpuLease(
            trial_id="dead-ownerless",
            ip="10.0.0.10",
            node_id="node-a",
            gpu_id="1",
            gpu_fraction=0.25,
            revision=1,
        ),
    }
    monkeypatch.setattr(runtime_allowlist, "_list_live_job_ids", lambda: {"job-live"})

    status = state.prune_dead_leases(
        live_trial_ids=["live-ownerless"],
        drop_ownerless=True,
    )

    assert {lease.trial_id for lease in status.active_leases} == {
        "live-owned",
        "live-ownerless",
    }
