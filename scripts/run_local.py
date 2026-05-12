from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "ghcr.io/denix56/noboom-benchmark-non-ray:latest"
DEFAULT_CONTAINER_REPO_DIR = "/workspace/noboom-source"


@dataclass(frozen=True)
class LocalRunCommand:
    run_command: List[str]
    output_dir: Path
    excel_path: Path
    run_env: Optional[Dict[str, str]] = None


def _repo_relative_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _build_pair_args(args: argparse.Namespace) -> List[str]:
    pair_args: List[str] = []
    for dataset in args.dataset or []:
        pair_args.extend(["--dataset", dataset])
    for model in args.model or []:
        pair_args.extend(["--model", model])
    for pair in getattr(args, "positional_pair", []) or []:
        pair_args.extend(["--pair", pair])
    for pair in args.pair or []:
        pair_args.extend(["--pair", pair])
    return pair_args


def _local_only_env() -> Dict[str, str]:
    return {
        "NOBOOM_S3_BUCKET": "",
        "NOBOOM_PREPARED_DATASET_S3_PATH": "",
        "S3_ENDPOINT_URL": "",
        "SEAFILE_USERNAME": "",
        "SEAFILE_PASS": "",
        "SEAFILE_ROOT_PATH": "",
        "NOBOOM_SEAFILE_UPLOAD_RESULTS": "0",
        "NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS": "0",
    }


def _build_run_tune_args(
    args: argparse.Namespace,
    *,
    storage_path: Path,
    sqlite_path: Optional[Path],
) -> List[str]:
    run_tune_args = [
        *_build_pair_args(args),
        "--gpus-per-run",
        str(args.gpus_per_run),
        "--timestamp",
        args.timestamp,
        "--config-dir",
        args.config_dir,
        "--temp-dir",
        str(storage_path / "tmp"),
        "--execution-backend",
        "local",
        "--artifact-storage-backend",
        "local",
        "--local-storage-path",
        str(storage_path),
        "--optuna-storage-backend",
        args.optuna_storage_backend,
    ]
    if args.tune:
        run_tune_args.append("--tune")
    if sqlite_path is not None:
        run_tune_args.extend(["--optuna-sqlite-path", str(sqlite_path)])
    if args.extra_run_arg:
        run_tune_args.extend(args.extra_run_arg)
    return run_tune_args


def _excel_path(args: argparse.Namespace, output_dir: Path) -> Path:
    experiment_name = args.experiment_name or f"NoBoomBenchmark__{args.timestamp}"
    return output_dir / experiment_name / "noboom_experiments.xlsx"


def _build_native_command(args: argparse.Namespace) -> LocalRunCommand:
    output_dir = _repo_relative_path(args.output_dir).resolve()
    sqlite_path: Optional[Path] = None
    if args.optuna_storage_backend == "sqlite":
        sqlite_path = _repo_relative_path(args.optuna_sqlite_path).resolve()

    run_env = os.environ.copy()
    run_env.update(_local_only_env())
    run_env["MLFLOW_TRACKING_URI"] = f"file://{output_dir / 'mlruns'}"
    if args.experiment_name:
        run_env["EXPERIMENT_NAME"] = args.experiment_name

    return LocalRunCommand(
        run_command=[
            sys.executable,
            "-m",
            "noboom_benchmark.run_tune",
            *_build_run_tune_args(args, storage_path=output_dir, sqlite_path=sqlite_path),
        ],
        output_dir=output_dir,
        excel_path=_excel_path(args, output_dir),
        run_env=run_env,
    )


def _build_docker_command(args: argparse.Namespace) -> LocalRunCommand:
    repo_root = _repo_relative_path(args.repo_root).resolve()
    output_dir = _repo_relative_path(args.output_dir).resolve()
    sqlite_path: Optional[Path] = None
    if args.optuna_storage_backend == "sqlite":
        sqlite_host_path = _repo_relative_path(args.optuna_sqlite_path).resolve()
        try:
            sqlite_path = Path(args.container_output_dir) / sqlite_host_path.relative_to(output_dir)
        except ValueError:
            sqlite_path = Path(args.container_output_dir) / sqlite_host_path.name

    container_repo_dir = Path(args.container_repo_dir)
    container_output_dir = Path(args.container_output_dir)
    docker_env: List[str] = []
    for key, value in _local_only_env().items():
        docker_env.extend(["-e", f"{key}={value}"])
    docker_env.extend(["-e", f"PYTHONPATH={container_repo_dir / 'src'}"])
    if args.experiment_name:
        docker_env.extend(["-e", f"EXPERIMENT_NAME={args.experiment_name}"])

    return LocalRunCommand(
        run_command=[
            "docker",
            "run",
            "--rm",
            "--gpus",
            args.gpus,
            "-v",
            f"{repo_root}:{container_repo_dir}",
            "-v",
            f"{output_dir}:{container_output_dir}",
            "-w",
            str(container_repo_dir),
            *docker_env,
            DEFAULT_IMAGE,
            "python",
            "-m",
            "noboom_benchmark.run_tune",
            *_build_run_tune_args(args, storage_path=container_output_dir, sqlite_path=sqlite_path),
        ],
        output_dir=output_dir,
        excel_path=_excel_path(args, output_dir),
    )


def build_commands(args: argparse.Namespace) -> LocalRunCommand:
    apply_cli_convenience_defaults(args)
    if args.deployment_mode == "native":
        return _build_native_command(args)
    return _build_docker_command(args)


def shell_join(command: Sequence[str], env: Optional[Dict[str, str]] = None) -> str:
    if not env:
        return " ".join(shlex.quote(part) for part in command)
    env_keys = [
        "NOBOOM_S3_BUCKET",
        "NOBOOM_PREPARED_DATASET_S3_PATH",
        "S3_ENDPOINT_URL",
        "SEAFILE_USERNAME",
        "SEAFILE_PASS",
        "SEAFILE_ROOT_PATH",
        "NOBOOM_SEAFILE_UPLOAD_RESULTS",
        "NOBOOM_SEAFILE_UPLOAD_CHECKPOINTS",
        "MLFLOW_TRACKING_URI",
        "EXPERIMENT_NAME",
    ]
    env_parts = [f"{key}={shlex.quote(env[key])}" for key in env_keys if key in env]
    return " ".join([*env_parts, *(shlex.quote(part) for part in command)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NoBoom sequential/non-Ray mode locally.")
    parser.add_argument(
        "positional_pair",
        nargs="*",
        metavar="DATASET:MODEL",
        help="Convenience positional pair. Example: cont_reactive_ome:neutralad",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--deployment-mode", choices=["native", "docker"], default="docker")
    parser.add_argument(
        "--container-repo-dir",
        default=None,
        help="Container source mount used with --deployment-mode=docker.",
    )
    parser.add_argument("--container-output-dir", default=None)
    parser.add_argument("--output-dir", default=".noboom_local")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--config-dir", default="src/noboom_cluster/cluster_files/configs")
    parser.add_argument("--gpus-per-run", type=float, default=1.0)
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--optuna-storage-backend",
        choices=["memory", "sqlite"],
        default="memory",
    )
    parser.add_argument("--optuna-sqlite-path", default=".noboom_local/optuna.db")
    parser.add_argument("--extra-run-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def apply_cli_convenience_defaults(args: argparse.Namespace) -> None:
    if not hasattr(args, "deployment_mode"):
        args.deployment_mode = "docker"
    if args.timestamp is None:
        args.timestamp = f"local_{args.deployment_mode}"
    if args.container_repo_dir is None:
        args.container_repo_dir = DEFAULT_CONTAINER_REPO_DIR
    if args.container_output_dir is None:
        args.container_output_dir = str(Path(args.container_repo_dir) / ".noboom_local")


def validate_args(args: argparse.Namespace) -> None:
    positional_pair = getattr(args, "positional_pair", []) or []
    if not args.pair and not positional_pair and (not args.dataset or not args.model):
        raise SystemExit("Provide --dataset with --model, one or more --pair values, or both.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_cli_convenience_defaults(args)
    validate_args(args)
    commands = build_commands(args)
    commands.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(shell_join(commands.run_command, commands.run_env))
        print(f"Output directory: {commands.output_dir}")
        print(f"Head Excel path: {commands.excel_path}")
        return 0

    subprocess.run(commands.run_command, check=True, env=commands.run_env)
    print(f"Output directory: {commands.output_dir}")
    print(f"Head Excel path: {commands.excel_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
