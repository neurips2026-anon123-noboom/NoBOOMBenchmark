from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_local.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_local", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_commands_runs_non_ray_local_mode_with_local_outputs(tmp_path: Path) -> None:
    module = _load_module()
    args = argparse.Namespace(
        repo_root=str(REPO_ROOT),
        container_repo_dir="/workspace/noboom",
        container_output_dir=None,
        output_dir=str(tmp_path / "outputs"),
        dataset=["cont_reactive_ome"],
        model=["gdn"],
        pair=[],
        timestamp="20260311_120000",
        experiment_name=None,
        config_dir="src/noboom_cluster/cluster_files/configs",
        gpus_per_run=0.25,
        gpus="all",
        tune=True,
        optuna_storage_backend="memory",
        optuna_sqlite_path=str(tmp_path / "outputs" / "optuna.db"),
        extra_run_arg=[],
    )

    commands = module.build_commands(args)

    run_command = commands.run_command
    assert run_command[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in run_command
    assert "all" in run_command
    assert f"{REPO_ROOT}:/workspace/noboom" in run_command
    assert f"{tmp_path / 'outputs'}:/workspace/noboom/.noboom_local" in run_command
    assert "NOBOOM_SEAFILE_UPLOAD_RESULTS=0" in run_command
    assert "NOBOOM_S3_BUCKET=" in run_command
    assert "PYTHONPATH=/workspace/noboom/src" in run_command
    assert "--execution-backend" in run_command
    assert run_command[run_command.index("--execution-backend") + 1] == "local"
    assert "--artifact-storage-backend" in run_command
    assert run_command[run_command.index("--artifact-storage-backend") + 1] == "local"
    assert "--optuna-storage-backend" in run_command
    assert run_command[run_command.index("--optuna-storage-backend") + 1] == "memory"
    assert commands.excel_path == (
        tmp_path / "outputs" / "NoBoomBenchmark__20260311_120000" / "noboom_experiments.xlsx"
    )


def test_build_commands_maps_sqlite_storage_inside_output_mount(tmp_path: Path) -> None:
    module = _load_module()
    args = argparse.Namespace(
        repo_root=str(REPO_ROOT),
        container_repo_dir="/workspace/noboom",
        container_output_dir=None,
        output_dir=str(tmp_path / "outputs"),
        dataset=[],
        model=[],
        pair=["cont_reactive_ome:gdn"],
        timestamp="local_docker",
        experiment_name="NoBoomBenchmark__docker",
        config_dir="src/noboom_cluster/cluster_files/configs",
        gpus_per_run=1.0,
        gpus="device=0",
        tune=False,
        optuna_storage_backend="sqlite",
        optuna_sqlite_path=str(tmp_path / "outputs" / "state" / "optuna.db"),
        extra_run_arg=["--verbose"],
    )

    commands = module.build_commands(args)

    assert "--optuna-sqlite-path" in commands.run_command
    assert commands.run_command[commands.run_command.index("--optuna-sqlite-path") + 1] == (
        "/workspace/noboom/.noboom_local/state/optuna.db"
    )
    assert "EXPERIMENT_NAME=NoBoomBenchmark__docker" in commands.run_command
    assert commands.excel_path == tmp_path / "outputs" / "NoBoomBenchmark__docker" / "noboom_experiments.xlsx"


def test_build_commands_can_use_custom_mount(tmp_path: Path) -> None:
    module = _load_module()
    args = argparse.Namespace(
        repo_root=str(REPO_ROOT),
        container_repo_dir="/workspace/noboom-src",
        container_output_dir=None,
        output_dir=str(tmp_path / "outputs"),
        dataset=[],
        model=[],
        pair=["cont_reactive_ome:neutralad"],
        timestamp="local_docker",
        experiment_name=None,
        config_dir="src/noboom_cluster/cluster_files/configs",
        gpus_per_run=1.0,
        gpus="all",
        tune=False,
        optuna_storage_backend="memory",
        optuna_sqlite_path=str(tmp_path / "outputs" / "optuna.db"),
        extra_run_arg=[],
    )

    commands = module.build_commands(args)

    assert "ghcr.io/denix56/noboom-benchmark-non-ray:latest" in commands.run_command
    assert f"{REPO_ROOT}:/workspace/noboom-src" in commands.run_command
    assert f"{tmp_path / 'outputs'}:/workspace/noboom-src/.noboom_local" in commands.run_command
    assert commands.run_command[commands.run_command.index("-w") + 1] == "/workspace/noboom-src"
    assert "PYTHONPATH=/workspace/noboom-src/src" in commands.run_command


def test_parser_accepts_positional_pair_with_common_defaults(tmp_path: Path) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "cont_reactive_ome:neutralad",
            "--tune",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )
    module.apply_cli_convenience_defaults(args)
    module.validate_args(args)

    commands = module.build_commands(args)

    run_command = commands.run_command
    assert "ghcr.io/denix56/noboom-benchmark-non-ray:latest" in run_command
    assert "--pair" in run_command
    assert run_command[run_command.index("--pair") + 1] == "cont_reactive_ome:neutralad"
    assert "--tune" in run_command
    assert run_command[run_command.index("--gpus") + 1] == "all"
    assert run_command[run_command.index("--gpus-per-run") + 1] == "1.0"
    assert run_command[run_command.index("--optuna-storage-backend") + 1] == "memory"
    assert f"{REPO_ROOT}:/workspace/noboom-source" in run_command
    assert f"{tmp_path / 'outputs'}:/workspace/noboom-source/.noboom_local" in run_command
    assert "PYTHONPATH=/workspace/noboom-source/src" in run_command


def test_parser_accepts_native_deployment_mode(tmp_path: Path) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "cont_reactive_ome:neutralad",
            "--deployment-mode",
            "native",
            "--tune",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    commands = module.build_commands(args)

    run_command = commands.run_command
    assert run_command[:3] == [sys.executable, "-m", "noboom_benchmark.run_tune"]
    assert "--pair" in run_command
    assert run_command[run_command.index("--pair") + 1] == "cont_reactive_ome:neutralad"
    assert run_command[run_command.index("--timestamp") + 1] == "local_native"
    assert run_command[run_command.index("--execution-backend") + 1] == "local"
    assert run_command[run_command.index("--artifact-storage-backend") + 1] == "local"
    assert run_command[run_command.index("--local-storage-path") + 1] == str(tmp_path / "outputs")
    assert commands.run_env is not None
    assert commands.run_env["NOBOOM_SEAFILE_UPLOAD_RESULTS"] == "0"
    assert commands.run_env["MLFLOW_TRACKING_URI"] == f"file://{tmp_path / 'outputs' / 'mlruns'}"
    assert commands.excel_path == (
        tmp_path / "outputs" / "NoBoomBenchmark__local_native" / "noboom_experiments.xlsx"
    )


def test_dry_run_uses_published_image_without_build(tmp_path: Path, capsys) -> None:
    module = _load_module()

    result = module.main(
        [
            "cont_reactive_ome:neutralad",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "docker build" not in output
    assert "ghcr.io/denix56/noboom-benchmark-non-ray:latest" in output
    assert f"{REPO_ROOT}:/workspace/noboom-source" in output
    assert "--pair cont_reactive_ome:neutralad" in output
    assert f"Output directory: {tmp_path / 'outputs'}" in output
    assert "Head Excel path:" in output


@pytest.mark.parametrize("removed_option", ["--prebuilt-image", "--image"])
def test_removed_image_options_are_not_accepted(removed_option: str) -> None:
    module = _load_module()
    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([removed_option, "cont_reactive_ome:neutralad"])


def test_parser_preserves_explicit_custom_mount(tmp_path: Path) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "cont_reactive_ome:neutralad",
            "--container-repo-dir",
            "/workspace/custom-src",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )
    module.apply_cli_convenience_defaults(args)

    commands = module.build_commands(args)

    assert "ghcr.io/denix56/noboom-benchmark-non-ray:latest" in commands.run_command
    assert f"{REPO_ROOT}:/workspace/custom-src" in commands.run_command
    assert "PYTHONPATH=/workspace/custom-src/src" in commands.run_command


def test_validate_args_requires_dataset_model_or_pair() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module.validate_args(argparse.Namespace(pair=[], dataset=[], model=[]))
