from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from noboom_cluster.noboom_cli_lib.ray_utils import job_lifecycle


def test_local_ray_auth_token_reads_generated_token(monkeypatch, tmp_path: Path) -> None:
    ray_dir = tmp_path / ".ray"
    ray_dir.mkdir()
    token_path = ray_dir / "auth_token"
    token_path.write_text("secret-token\n", encoding="utf-8")

    monkeypatch.setattr(job_lifecycle.Path, "home", staticmethod(lambda: tmp_path))

    token = job_lifecycle._local_ray_auth_token()

    assert token == "secret-token"


def test_job_submission_client_exports_local_auth(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, address: str, headers=None) -> None:
            captured["address"] = address
            captured["authorization"] = (headers or {}).get("authorization", "")

    monkeypatch.setattr(
        job_lifecycle,
        "_local_ray_auth_token",
        lambda: "secret-token",
    )
    monkeypatch.setattr(job_lifecycle, "JobSubmissionClient", FakeClient)

    client = job_lifecycle._job_submission_client("http://127.0.0.1:8265")

    assert isinstance(client, FakeClient)
    assert captured == {
        "address": "http://127.0.0.1:8265",
        "authorization": "Bearer secret-token",
    }


def test_is_job_server_running_uses_bounded_request_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def _do_request(self, method: str, endpoint: str, **kwargs: object) -> SimpleNamespace:
            captured["method"] = method
            captured["endpoint"] = endpoint
            captured["kwargs"] = kwargs
            return SimpleNamespace(status_code=200)

    def fake_client(address: str) -> FakeClient:
        captured["address"] = address
        return FakeClient()

    monkeypatch.setattr(job_lifecycle, "_job_submission_client", fake_client)

    assert job_lifecycle.is_job_server_running(
        "http://127.0.0.1:8265",
        request_timeout_s=2.5,
    )
    assert captured == {
        "address": "http://127.0.0.1:8265",
        "method": "GET",
        "endpoint": "/api/jobs/",
        "kwargs": {"timeout": (2.5, 2.5)},
    }


def test_is_job_server_running_returns_false_when_probe_raises(monkeypatch) -> None:
    class FakeClient:
        def _do_request(self, method: str, endpoint: str, **kwargs: object) -> SimpleNamespace:
            raise TimeoutError("timed out")

    monkeypatch.setattr(job_lifecycle, "_job_submission_client", lambda address: FakeClient())

    assert not job_lifecycle.is_job_server_running("http://127.0.0.1:8265", request_timeout_s=1.0)


def test_start_script_uses_bounded_probe_instead_of_direct_list_jobs(monkeypatch) -> None:
    probe_calls: list[str] = []
    submitted: dict[str, object] = {}

    class FakeClient:
        def list_jobs(self) -> list[object]:
            raise AssertionError("start_script should use is_job_server_running for probing")

        def submit_job(self, **kwargs: object) -> None:
            submitted.update(kwargs)

    monkeypatch.setattr(
        job_lifecycle,
        "is_job_server_running",
        lambda address: probe_calls.append(address) or True,
    )
    monkeypatch.setattr(job_lifecycle, "_job_submission_client", lambda address: FakeClient())
    monkeypatch.setattr(job_lifecycle, "build_uv_runtime", lambda: {})
    monkeypatch.setattr(
        job_lifecycle,
        "get_resolved_workdir",
        lambda workdir, cluster_config, use_docker: workdir,
    )

    job_lifecycle.start_script(
        job_lifecycle.StartScriptConfig(
            ssh_user="cloud",
            ssh_key_path="~/.ssh/id_ed25519",
            root_dir="~/noboom",
            storage_dir="~/noboom/experiment_data",
            workdir="~/noboom",
            head_ip="203.0.113.10",
            worker_ips=["203.0.113.11"],
            config_path="cluster.yaml",
            ray_temp_dir="/tmp/ray",
            models=["gdn"],
            datasets=["ome"],
            tune=False,
            experiment_id=None,
            job_server_address="http://127.0.0.1:8265",
            gpus_per_run=1,
            enable_docker=False,
        )
    )

    assert probe_calls == ["http://127.0.0.1:8265"]
    assert submitted["entrypoint"].startswith("python -m noboom_benchmark.run_tune --model gdn --dataset ome")


def test_start_script_forwards_explicit_pairs(monkeypatch) -> None:
    submitted: dict[str, object] = {}

    class FakeClient:
        def submit_job(self, **kwargs: object) -> None:
            submitted.update(kwargs)

    monkeypatch.setattr(job_lifecycle, "is_job_server_running", lambda address: True)
    monkeypatch.setattr(job_lifecycle, "_job_submission_client", lambda address: FakeClient())
    monkeypatch.setattr(job_lifecycle, "build_uv_runtime", lambda: {})
    monkeypatch.setattr(
        job_lifecycle,
        "get_resolved_workdir",
        lambda workdir, cluster_config, use_docker: workdir,
    )

    job_lifecycle.start_script(
        job_lifecycle.StartScriptConfig(
            ssh_user="cloud",
            ssh_key_path="~/.ssh/id_ed25519",
            root_dir="~/noboom",
            storage_dir="~/noboom/experiment_data",
            workdir="~/noboom",
            head_ip="203.0.113.10",
            worker_ips=[],
            config_path="cluster.yaml",
            ray_temp_dir="/tmp/ray",
            models=["gdn"],
            datasets=["ome"],
            pairs=["srb:lstm_ae", "ome:gdn"],
            tune=False,
            experiment_id=None,
            job_server_address="http://127.0.0.1:8265",
            gpus_per_run=1,
            enable_docker=False,
        )
    )

    entrypoint = str(submitted["entrypoint"])
    assert "--model gdn --dataset ome --pair srb:lstm_ae --pair ome:gdn" in entrypoint


def test_start_script_forwards_save_checkpoints(monkeypatch) -> None:
    submitted: dict[str, object] = {}

    class FakeClient:
        def submit_job(self, **kwargs: object) -> None:
            submitted.update(kwargs)

    monkeypatch.setattr(job_lifecycle, "is_job_server_running", lambda address: True)
    monkeypatch.setattr(job_lifecycle, "_job_submission_client", lambda address: FakeClient())
    monkeypatch.setattr(job_lifecycle, "build_uv_runtime", lambda: {})
    monkeypatch.setattr(
        job_lifecycle,
        "get_resolved_workdir",
        lambda workdir, cluster_config, use_docker: workdir,
    )

    job_lifecycle.start_script(
        job_lifecycle.StartScriptConfig(
            ssh_user="cloud",
            ssh_key_path="~/.ssh/id_ed25519",
            root_dir="~/noboom",
            storage_dir="~/noboom/experiment_data",
            workdir="~/noboom",
            head_ip="203.0.113.10",
            worker_ips=[],
            config_path="cluster.yaml",
            ray_temp_dir="/tmp/ray",
            models=["gdn"],
            datasets=["ome"],
            tune=False,
            experiment_id=None,
            job_server_address="http://127.0.0.1:8265",
            gpus_per_run=1,
            enable_docker=False,
            save_checkpoints=True,
        )
    )

    assert "--save-checkpoints" in str(submitted["entrypoint"])


def test_start_script_validation_failure_prevents_job_submission(monkeypatch) -> None:
    class FakeClient:
        def submit_job(self, **kwargs: object) -> None:
            raise AssertionError("submit_job must not run after validation failure")

    monkeypatch.setattr(job_lifecycle, "is_job_server_running", lambda address: True)
    monkeypatch.setattr(job_lifecycle, "_job_submission_client", lambda address: FakeClient())
    monkeypatch.setattr(job_lifecycle, "build_uv_runtime", lambda: {})
    monkeypatch.setattr(
        job_lifecycle,
        "get_resolved_workdir",
        lambda workdir, cluster_config, use_docker: workdir,
    )

    def fail_validation() -> None:
        raise RuntimeError("cluster validation failed")

    with pytest.raises(RuntimeError, match="cluster validation failed"):
        job_lifecycle.start_script(
            job_lifecycle.StartScriptConfig(
                ssh_user="cloud",
                ssh_key_path="~/.ssh/id_ed25519",
                root_dir="~/noboom",
                storage_dir="~/noboom/experiment_data",
                workdir="~/noboom",
                head_ip="203.0.113.10",
                worker_ips=["203.0.113.11"],
                config_path="cluster.yaml",
                ray_temp_dir="/tmp/ray",
                models=["gdn"],
                datasets=["ome"],
                tune=False,
                experiment_id=None,
                job_server_address="http://127.0.0.1:8265",
                gpus_per_run=1,
                enable_docker=False,
                pre_submit_validation=fail_validation,
            )
        )


def test_start_script_passes_selected_ray_ips_to_setup_head(monkeypatch) -> None:
    submitted: dict[str, object] = {}
    ray_exec_commands: list[str] = []

    class FakeClient:
        def submit_job(self, **kwargs: object) -> None:
            submitted.update(kwargs)

    monkeypatch.setattr(job_lifecycle, "run_ray_command", lambda command: None)
    monkeypatch.setattr(
        job_lifecycle,
        "run_ray_exec",
        lambda config_path, command, tmux=False: ray_exec_commands.append(command) or 0,
    )
    monkeypatch.setattr(job_lifecycle, "_start_native_services", lambda *args, **kwargs: None)
    monkeypatch.setattr(job_lifecycle, "_job_submission_client", lambda address: FakeClient())
    monkeypatch.setattr(job_lifecycle, "build_uv_runtime", lambda: {})
    monkeypatch.setattr(
        job_lifecycle,
        "get_resolved_workdir",
        lambda workdir, cluster_config, use_docker: workdir,
    )

    job_lifecycle.start_script(
        job_lifecycle.StartScriptConfig(
            ssh_user="ubuntu",
            ssh_key_path="~/.ssh/id_ed25519",
            root_dir="/tmp/noboom",
            storage_dir="/tmp/noboom/experiment_data",
            workdir="/tmp/noboom",
            head_ip="203.0.113.10",
            worker_ips=["203.0.113.11"],
            config_path="cluster.yaml",
            ray_temp_dir="/tmp/ray",
            models=["gdn"],
            datasets=["ome"],
            tune=False,
            experiment_id=None,
            job_server_address="http://127.0.0.1:8265",
            gpus_per_run=1,
            force_restart=True,
            enable_docker=False,
        )
    )

    setup_head_commands = [command for command in ray_exec_commands if "setup_head.sh" in command]
    assert setup_head_commands
    assert "203.0.113.10 203.0.113.11" in setup_head_commands[0]
    assert "192.168." not in setup_head_commands[0]
    assert submitted["entrypoint"].startswith("python -m noboom_benchmark.run_tune")


def test_start_native_services_uses_root_dir_postgres_data(monkeypatch) -> None:
    recorded_commands: list[tuple[str, bool]] = []
    recorded_tmux_sessions: list[tuple[str, str, str]] = []

    monkeypatch.setattr(job_lifecycle, "stop_weed_gracefully", lambda *args, **kwargs: None)
    monkeypatch.setattr(job_lifecycle, "stop_mlflow_gracefully", lambda *args, **kwargs: None)

    def fake_run_ray_exec(config_path: str, command: str, tmux: bool = False) -> int:
        recorded_commands.append((command, tmux))
        return 0

    def fake_run_named_tmux_session(config_path: str, session_name: str, command: str) -> int:
        recorded_tmux_sessions.append((config_path, session_name, command))
        return 0

    monkeypatch.setattr(job_lifecycle, "run_ray_exec", fake_run_ray_exec)
    monkeypatch.setattr(job_lifecycle, "run_named_tmux_session", fake_run_named_tmux_session)

    job_lifecycle._start_native_services(
        "cluster.yaml",
        root_dir="/work/user/test_noboom",
        workdir_host="/work/user/test_noboom",
        storage_dir="/work/user/test_noboom/experiment_data",
        mlflow_port=5000,
    )

    postgres_command, postgres_tmux = recorded_commands[0]

    assert postgres_tmux is False
    assert "/work/user/test_noboom/postgres/pgsql/data" in postgres_command
    assert "SHOW data_directory" not in postgres_command
    assert "ss -ltnp" in postgres_command
    assert recorded_tmux_sessions[0][1] == job_lifecycle.NATIVE_SEAWEED_TMUX_SESSION
    assert "weed server" in recorded_tmux_sessions[0][2]
    assert recorded_tmux_sessions[1][1] == job_lifecycle.NATIVE_MLFLOW_TMUX_SESSION
    assert "mlflow server" in recorded_tmux_sessions[1][2]


def test_refresh_live_mlflow_service_recreates_docker_mlflow(monkeypatch) -> None:
    recorded_commands: list[tuple[str, bool]] = []

    def fake_run_ray_exec(config_path: str, command: str, tmux: bool = False) -> int:
        recorded_commands.append((command, tmux))
        return 0

    monkeypatch.setattr(job_lifecycle, "run_ray_exec", fake_run_ray_exec)

    job_lifecycle.refresh_live_mlflow_service(
        "cluster.yaml",
        root_dir="~/noboom",
        workdir_host="/tmp/ray/session_latest/runtime_resources/working_dir_files/_ray_pkg_123/workspace/noboom",
        mlflow_port=5000,
        enable_docker=True,
    )

    setup_env_command, setup_env_tmux = recorded_commands[0]
    restart_command, restart_tmux = recorded_commands[1]

    assert setup_env_tmux is False
    assert "setup_env.py" in setup_env_command
    assert restart_tmux is False
    assert "docker compose -f" in restart_command
    assert "--force-recreate mlflow" in restart_command
    assert "docker-compose -f" in restart_command


def test_refresh_live_mlflow_service_restarts_native_mlflow(monkeypatch) -> None:
    recorded_commands: list[tuple[str, bool]] = []
    recorded_tmux_sessions: list[tuple[str, str, str]] = []
    stop_calls: list[tuple[str, str, bool]] = []

    def fake_run_ray_exec(config_path: str, command: str, tmux: bool = False) -> int:
        recorded_commands.append((command, tmux))
        return 0

    def fake_run_named_tmux_session(config_path: str, session_name: str, command: str) -> int:
        recorded_tmux_sessions.append((config_path, session_name, command))
        return 0

    def fake_stop_mlflow_gracefully(
        cluster_yaml: str,
        root_dir: str,
        kill_after_timeout: bool = False,
    ) -> bool:
        stop_calls.append((cluster_yaml, root_dir, kill_after_timeout))
        return True

    monkeypatch.setattr(job_lifecycle, "run_ray_exec", fake_run_ray_exec)
    monkeypatch.setattr(job_lifecycle, "run_named_tmux_session", fake_run_named_tmux_session)
    monkeypatch.setattr(job_lifecycle, "stop_mlflow_gracefully", fake_stop_mlflow_gracefully)

    job_lifecycle.refresh_live_mlflow_service(
        "cluster.yaml",
        root_dir="/work/user/test_noboom",
        workdir_host="/work/user/test_noboom",
        mlflow_port=5000,
        enable_docker=False,
    )

    setup_env_command, setup_env_tmux = recorded_commands[0]

    assert setup_env_tmux is False
    assert "setup_env.py" in setup_env_command
    assert stop_calls == [("cluster.yaml", "/work/user/test_noboom", True)]
    assert recorded_tmux_sessions[0][1] == job_lifecycle.NATIVE_MLFLOW_TMUX_SESSION
    assert "mlflow server" in recorded_tmux_sessions[0][2]
    assert "/work/user/test_noboom/mnt/.env" in recorded_tmux_sessions[0][2]
