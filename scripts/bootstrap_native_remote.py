#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass, field
from typing import IO, Any, Mapping, Optional, Sequence

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency
    tqdm = None

logging.addLevelName(logging.WARNING, "WARN")


USAGE_TEXT = """Usage:
  scripts/bootstrap_native_remote.sh \\
    --root-dir <root_dir> \\
    [--local | (--host <host> --ssh-user <ssh_user> --ssh-key <ssh_key>)] \\
    [--head] \\
    [--skip-uv] \\
    [--skip-venv] \\
    [--skip-dependency-sync] \\
    [--skip-treeple] \\
    [--skip-torch-cluster] \\
    [--skip-postgres] \\
    [--skip-seaweed] \\
    [--skip-symlinks] \\
    [--rotate-postgres-password] \\
    [--uninstall]

Modes:
  install     Default mode. Provision a native runtime under the given root dir.
  rotate-postgres-password
              Rotate the managed PostgreSQL password in place for an existing head bootstrap.
  uninstall   Remove only the assets previously created by this bootstrap script.

Examples:
  scripts/bootstrap_native_remote.sh \\
    --local \\
    --root-dir ~/noboom-native-local \\
    --skip-torch-cluster

  scripts/bootstrap_native_remote.sh \\
    --local \\
    --root-dir ~/noboom-native-local-head \\
    --head

  scripts/bootstrap_native_remote.sh \\
    --host 203.0.113.10 \\
    --ssh-user <ssh_user> \\
    --ssh-key <path_to_private_ssh_key> \\
    --root-dir <remote_root_dir>/noboom-native-smoke \\
    --skip-torch-cluster \\
    --skip-postgres \\
    --skip-seaweed \\
    --skip-symlinks

  scripts/bootstrap_native_remote.sh \\
    --host 203.0.113.10 \\
    --ssh-user <ssh_user> \\
    --ssh-key <path_to_private_ssh_key> \\
    --root-dir <remote_root_dir>/noboom-native-smoke \\
    --head \\
    --rotate-postgres-password

  scripts/bootstrap_native_remote.sh \\
    --host 203.0.113.10 \\
    --ssh-user <ssh_user> \\
    --ssh-key <path_to_private_ssh_key> \\
    --root-dir <remote_root_dir>/noboom-native-smoke \\
    --uninstall
"""

TREEPLE_REF = "ef417e47e12caaa70ecc027db53f18b70b895930"
TORCH_CLUSTER_REF = "6dabb048d29c2bc788fce885d0e13f888c8987a2"
POSTGRES_VERSION = "18.1"
SEAWEED_VERSION = "4.05"
PYTHON_VERSION = "3.13"
POSTGRES_ARCHIVE_URL = (
    f"https://ftp.postgresql.org/pub/source/v{POSTGRES_VERSION}/postgresql-{POSTGRES_VERSION}.tar.bz2"
)
SEAWEED_ARCHIVE_URL = (
    f"https://github.com/seaweedfs/seaweedfs/releases/download/{SEAWEED_VERSION}/linux_amd64_full.tar.gz"
)
POSTGRES_USER = "noboom"
POSTGRES_PASSWORD_ENV_VAR = "POSTGRES_PASSWORD"
BOOTSTRAP_SECRETS_FILENAME = "bootstrap-secrets.json"
TREEPLE_BUILD_PACKAGES: tuple[str, ...] = (
    "meson>=1.4.0",
    "meson-python>=0.16.0",
    "ninja",
    "wheel",
    "setuptools<=65.5",
    "packaging",
    "Cython>=3.0.10",
    "numpy>=1.25.0",
    "scipy>=1.5.0",
    "scikit-learn>=1.6.0",
    "spin>=0.12",
)
SKIP_STAGE_FLAGS: tuple[tuple[str, str], ...] = (
    ("skip_uv", "--skip-uv"),
    ("skip_venv", "--skip-venv"),
    ("skip_dependency_sync", "--skip-dependency-sync"),
    ("skip_treeple", "--skip-treeple"),
    ("skip_torch_cluster", "--skip-torch-cluster"),
    ("skip_postgres", "--skip-postgres"),
    ("skip_seaweed", "--skip-seaweed"),
    ("skip_symlinks", "--skip-symlinks"),
)


class ScriptError(RuntimeError):
    """Raised for user-facing bootstrap failures."""


@dataclass(frozen=True)
class FrontendConfig:
    mode: str
    local_mode: bool
    head_mode: bool
    skip_uv: bool
    skip_venv: bool
    skip_dependency_sync: bool
    skip_treeple: bool
    skip_torch_cluster: bool
    skip_postgres: bool
    skip_seaweed: bool
    skip_symlinks: bool
    host: str
    ssh_user: str
    ssh_key: Optional[Path]
    root_dir: str


@dataclass(frozen=True)
class TargetConfig:
    mode: str
    root_dir: Path
    request_host: str
    head_mode: bool
    skip_uv: bool
    skip_venv: bool
    skip_dependency_sync: bool
    skip_treeple: bool
    skip_torch_cluster: bool
    skip_postgres: bool
    skip_seaweed: bool
    skip_symlinks: bool


@dataclass
class ProgressTracker:
    total_steps: int
    label: str
    logger: "LineLogger"
    current_step: int = 0
    _bar: Optional[Any] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if tqdm is None or not sys.stderr.isatty():
            return
        self._bar = tqdm(
            total=self.total_steps,
            desc=self.label,
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )

    def step(self, description: str) -> None:
        self.current_step += 1
        if self._bar is not None:
            self._bar.set_postfix_str(description, refresh=False)
            self._bar.update(1)
        self.logger.info(f"[STEP {self.current_step}/{self.total_steps}] {description}")

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class LineLogger:
    def __init__(self, log_file_path: Path) -> None:
        self._log_file_path = log_file_path
        self._handle: IO[str] = log_file_path.open("a", encoding="utf-8")
        logger_name = f"bootstrap_native_remote.{id(self)}"
        self._logger = logging.getLogger(logger_name)
        self._logger.handlers.clear()
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.StreamHandler(self._handle)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        self._logger.addHandler(stdout_handler)
        self._logger.addHandler(file_handler)

    def close(self) -> None:
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
        self._handle.close()

    def emit_raw(self, message: str) -> None:
        sys.stdout.write(message)
        sys.stdout.flush()
        self._handle.write(message)
        self._handle.flush()

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warn(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)


@dataclass
class BootstrapState:
    install_host: str
    remote_hostname: str
    root_dir: str
    install_timestamp: str
    pins: dict[str, str]
    is_head: bool
    postgres_port: str
    created_core_paths: list[str]
    created_auxiliary_paths: list[str]
    preexisting_auxiliary_paths: list[str]
    created_symlinks: list[dict[str, str]]

    @classmethod
    def load(cls, state_path: Path) -> "BootstrapState":
        raw_data = json.loads(state_path.read_text(encoding="utf-8"))
        created_core_paths = [str(path) for path in raw_data.get("created_core_paths", [])]
        is_head_value = raw_data.get("is_head")
        if is_head_value is None:
            root_dir = str(raw_data.get("root_dir", ""))
            head_paths = {f"{root_dir}/postgres", f"{root_dir}/seaweedfs"}
            is_head_value = any(path in head_paths for path in created_core_paths)

        return cls(
            install_host=str(raw_data.get("install_host", "")),
            remote_hostname=str(raw_data.get("remote_hostname", "")),
            root_dir=str(raw_data.get("root_dir", "")),
            install_timestamp=str(raw_data.get("install_timestamp", "")),
            pins={
                "treeple_ref": str(raw_data.get("pins", {}).get("treeple_ref", TREEPLE_REF)),
                "torch_cluster_ref": str(raw_data.get("pins", {}).get("torch_cluster_ref", TORCH_CLUSTER_REF)),
                "postgres_version": str(raw_data.get("pins", {}).get("postgres_version", POSTGRES_VERSION)),
                "seaweed_version": str(raw_data.get("pins", {}).get("seaweed_version", SEAWEED_VERSION)),
            },
            is_head=bool(is_head_value),
            postgres_port=str(raw_data.get("postgres_port", "")),
            created_core_paths=created_core_paths,
            created_auxiliary_paths=[str(path) for path in raw_data.get("created_auxiliary_paths", [])],
            preexisting_auxiliary_paths=[str(path) for path in raw_data.get("preexisting_auxiliary_paths", [])],
            created_symlinks=[
                {
                    "path": str(entry["path"]),
                    "target": str(entry["target"]),
                }
                for entry in raw_data.get("created_symlinks", [])
            ],
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "install_host": self.install_host,
                "remote_hostname": self.remote_hostname,
                "root_dir": self.root_dir,
                "install_timestamp": self.install_timestamp,
                "pins": self.pins,
                "is_head": self.is_head,
                "postgres_port": self.postgres_port,
                "created_core_paths": self.created_core_paths,
                "created_auxiliary_paths": self.created_auxiliary_paths,
                "preexisting_auxiliary_paths": self.preexisting_auxiliary_paths,
                "created_symlinks": self.created_symlinks,
            },
            indent=2,
        ) + "\n"


def print_usage(stream: IO[str]) -> None:
    stream.write(USAGE_TEXT)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise ScriptError(f"Required local command not found: {command_name}")


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def merge_env(extra_env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    env = dict(os.environ)
    if extra_env is not None:
        env.update({key: value for key, value in extra_env.items()})
    return env


def run_frontend_command(
    command: Sequence[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(part) for part in command],
        input=input_text,
        capture_output=True,
        text=True,
        env=merge_env(env),
        check=False,
    )
    if result.returncode != 0:
        raise ScriptError(result.stderr.strip() or result.stdout.strip() or f"Command failed: {format_command(command)}")
    return result


def run_frontend_ssh(
    ssh_base: Sequence[str],
    remote_command: str,
    *,
    input_text: Optional[str] = None,
) -> str:
    result = subprocess.run(
        [*ssh_base, remote_command],
        input=input_text,
        capture_output=True,
        text=True,
        env=merge_env(),
        check=False,
    )
    if result.returncode != 0:
        raise ScriptError(result.stderr.strip() or result.stdout.strip() or f"Remote command failed: {remote_command}")
    return result.stdout.strip()


def stream_frontend_ssh(
    ssh_base: Sequence[str],
    remote_command: str,
    *,
    input_text: Optional[str] = None,
) -> None:
    result = subprocess.run(
        [*ssh_base, remote_command],
        input=input_text,
        text=True,
        env=merge_env(),
        check=False,
    )
    if result.returncode != 0:
        raise ScriptError(f"Remote command failed: {remote_command}")


def normalize_local_root(requested_root: str) -> Path:
    return Path(requested_root).expanduser().resolve()


def normalize_remote_root(ssh_base: Sequence[str], requested_root: str) -> Path:
    remote_command = " ".join(
        [
            "python3",
            "-c",
            shlex.quote("import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))"),
            shlex.quote(requested_root),
        ]
    )
    return Path(run_frontend_ssh(ssh_base, remote_command))


def requires_postgres_password(config: Any) -> bool:
    mode = getattr(config, "mode", "")
    if mode == "rotate-postgres-password":
        return bool(getattr(config, "head_mode", False))
    return bool(
        mode == "install"
        and getattr(config, "head_mode", False)
        and not getattr(config, "skip_postgres", False)
    )


def has_skip_stage_flags(config: Any) -> bool:
    return any(getattr(config, attr_name, False) for attr_name, _ in SKIP_STAGE_FLAGS)


def replace_postgres_uri_password(uri: str, password: str) -> str:
    if "://" not in uri or "@" not in uri:
        return uri
    scheme, remainder = uri.split("://", 1)
    credentials, suffix = remainder.split("@", 1)
    if ":" not in credentials:
        return uri
    username, _ = credentials.split(":", 1)
    return f"{scheme}://{username}:{password}@{suffix}"


def rewrite_env_file_postgres_password(env_path: Path, password: str) -> bool:
    if not env_path.is_file():
        return False

    original_mode = env_path.stat().st_mode & 0o777
    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    found_password_line = False

    for line in lines:
        if line.startswith("POSTGRES_PASSWORD="):
            updated_lines.append(f"POSTGRES_PASSWORD={password}")
            found_password_line = True
            continue
        if line.startswith("MLFLOW_STORAGE_URI="):
            current_uri = line.split("=", 1)[1]
            updated_lines.append(
                f"MLFLOW_STORAGE_URI={replace_postgres_uri_password(current_uri, password)}"
            )
            continue
        if line.startswith("OPTUNA_STORAGE_URI="):
            current_uri = line.split("=", 1)[1]
            updated_lines.append(
                f"OPTUNA_STORAGE_URI={replace_postgres_uri_password(current_uri, password)}"
            )
            continue
        updated_lines.append(line)

    if not found_password_line:
        updated_lines.append(f"POSTGRES_PASSWORD={password}")

    os.chmod(env_path, 0o600)
    env_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    os.chmod(env_path, original_mode or 0o600)
    return True


def build_postgres_password_rotation_sql(password: str) -> str:
    escaped_password = password.replace("'", "''")
    return (
        "DO $$ "
        "BEGIN "
        f"IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '{POSTGRES_USER}') THEN "
        f"CREATE ROLE {POSTGRES_USER} LOGIN PASSWORD '{escaped_password}'; "
        f"ELSE ALTER ROLE {POSTGRES_USER} WITH PASSWORD '{escaped_password}'; "
        "END IF; "
        "END $$;"
    )


def get_required_env_value(env_var_name: str, env: Mapping[str, str]) -> str:
    value = str(env.get(env_var_name, "")).strip()
    if not value:
        raise FrontendUsageError(f"Missing required environment variable: {env_var_name}")
    return value


def build_bootstrap_secret_payload(config: FrontendConfig) -> dict[str, str]:
    secrets: dict[str, str] = {}
    if requires_postgres_password(config):
        secrets[POSTGRES_PASSWORD_ENV_VAR] = get_required_env_value(POSTGRES_PASSWORD_ENV_VAR, os.environ)
    return secrets


def needs_staged_inputs(config: FrontendConfig) -> bool:
    return config.mode == "install" or requires_postgres_password(config)


def stage_local_inputs(normalized_root_dir: Path, repo_dir: Path, config: FrontendConfig) -> None:
    metadata_dir = normalized_root_dir / ".noboom-native-bootstrap"
    staging_dir = metadata_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    if config.mode == "install":
        for source_path in (
            repo_dir / "pyproject.toml",
            repo_dir / "uv.lock",
            repo_dir / "docker" / "db" / "scripts" / "init-db.sql",
        ):
            shutil.copy2(source_path, staging_dir / source_path.name)

    secret_payload = build_bootstrap_secret_payload(config)
    if secret_payload:
        secret_path = staging_dir / BOOTSTRAP_SECRETS_FILENAME
        secret_path.write_text(json.dumps(secret_payload) + "\n", encoding="utf-8")
        secret_path.chmod(0o600)


def stage_remote_inputs(
    ssh_base: Sequence[str],
    ssh_target: str,
    rsync_ssh: str,
    normalized_root_dir: Path,
    repo_dir: Path,
    config: FrontendConfig,
) -> None:
    metadata_dir = normalized_root_dir / ".noboom-native-bootstrap"
    staging_dir = metadata_dir / "staging"
    mkdir_command = "mkdir -p {root} {metadata} {staging}".format(
        root=shlex.quote(str(normalized_root_dir)),
        metadata=shlex.quote(str(metadata_dir)),
        staging=shlex.quote(str(staging_dir)),
    )
    run_frontend_ssh(ssh_base, mkdir_command)

    secret_payload = build_bootstrap_secret_payload(config)
    temp_secret_dir: Optional[Path] = None
    try:
        if secret_payload:
            temp_secret_dir = Path(tempfile.mkdtemp(prefix="bootstrap-native-secrets-"))
            temp_secret_path = temp_secret_dir / BOOTSTRAP_SECRETS_FILENAME
            temp_secret_path.write_text(json.dumps(secret_payload) + "\n", encoding="utf-8")
            temp_secret_path.chmod(0o600)

        command = ["rsync", "-az", "-e", rsync_ssh]
        if config.mode == "install":
            command.extend(
                [
                    str(repo_dir / "pyproject.toml"),
                    str(repo_dir / "uv.lock"),
                    str(repo_dir / "docker" / "db" / "scripts" / "init-db.sql"),
                ]
            )
        if temp_secret_dir is not None:
            temp_secret_path = temp_secret_dir / BOOTSTRAP_SECRETS_FILENAME
            command.append(str(temp_secret_path))
        command.append(f"{ssh_target}:{staging_dir}/")
        run_frontend_command(command)
    finally:
        if temp_secret_dir is not None and temp_secret_dir.exists():
            shutil.rmtree(temp_secret_dir)


def current_script_source() -> str:
    return Path(__file__).read_text(encoding="utf-8")


def append_skip_stage_flags(command_parts: list[str], config: Any) -> None:
    for attr_name, flag_name in SKIP_STAGE_FLAGS:
        if getattr(config, attr_name):
            command_parts.append(flag_name)


def build_target_command(config: FrontendConfig, normalized_root_dir: Path, request_host: str) -> str:
    command_parts = [
        "python3",
        "-",
        "--internal-target-mode",
        "--mode",
        config.mode,
        "--root-dir",
        str(normalized_root_dir),
        "--request-host",
        request_host,
    ]
    if config.head_mode:
        command_parts.append("--head")
    append_skip_stage_flags(command_parts, config)
    return format_command(command_parts)


def frontend_main(argv: Sequence[str]) -> int:
    config = parse_frontend_args(argv)
    validate_frontend_config(config)

    repo_dir = repo_root()
    if config.mode == "install":
        required_files = (
            repo_dir / "pyproject.toml",
            repo_dir / "uv.lock",
            repo_dir / "docker" / "db" / "scripts" / "init-db.sql",
        )
        for source_path in required_files:
            if not source_path.is_file():
                raise ScriptError(f"Required local file not found: {source_path}")

    if config.local_mode:
        normalized_root_dir = normalize_local_root(config.root_dir)
        if needs_staged_inputs(config):
            stage_local_inputs(normalized_root_dir, repo_dir, config)
        target = BootstrapTarget(
            TargetConfig(
                mode=config.mode,
                root_dir=normalized_root_dir,
                request_host="local",
                head_mode=config.head_mode,
                skip_uv=config.skip_uv,
                skip_venv=config.skip_venv,
                skip_dependency_sync=config.skip_dependency_sync,
                skip_treeple=config.skip_treeple,
                skip_torch_cluster=config.skip_torch_cluster,
                skip_postgres=config.skip_postgres,
                skip_seaweed=config.skip_seaweed,
                skip_symlinks=config.skip_symlinks,
            )
        )
        target.run()
        return 0

    assert config.ssh_key is not None
    ssh_key_path = config.ssh_key.expanduser().resolve()
    if not ssh_key_path.is_file():
        raise ScriptError(f"SSH key not found: {ssh_key_path}")

    ssh_target = f"{config.ssh_user}@{config.host}"
    ssh_base = [
        "ssh",
        "-T",
        "-i",
        str(ssh_key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        ssh_target,
    ]
    rsync_ssh = f"ssh -i {shlex.quote(str(ssh_key_path))} -o BatchMode=yes -o StrictHostKeyChecking=no"
    normalized_root_dir = normalize_remote_root(ssh_base, config.root_dir)

    if needs_staged_inputs(config):
        stage_remote_inputs(ssh_base, ssh_target, rsync_ssh, normalized_root_dir, repo_dir, config)

    remote_command = build_target_command(config, normalized_root_dir, config.host)
    stream_frontend_ssh(ssh_base, remote_command, input_text=current_script_source())
    return 0


def parse_frontend_args(argv: Sequence[str]) -> FrontendConfig:
    mode = "install"
    head_mode = False
    skip_uv = False
    skip_venv = False
    skip_dependency_sync = False
    skip_treeple = False
    skip_torch_cluster = False
    skip_postgres = False
    skip_seaweed = False
    skip_symlinks = False
    local_mode = False
    host = ""
    ssh_user = ""
    ssh_key = ""
    root_dir = ""

    def set_mode(new_mode: str, triggering_flag: str) -> None:
        nonlocal mode
        if mode != "install" and mode != new_mode:
            raise FrontendUsageError(
                f"Conflicting mode flags: {triggering_flag} cannot be combined with the current mode."
            )
        mode = new_mode

    index = 0
    argv_list = list(argv)
    while index < len(argv_list):
        argument = argv_list[index]
        if argument in ("-h", "--help"):
            print_usage(sys.stdout)
            raise SystemExit(0)
        if argument == "--host":
            value = require_value(argument, argv_list, index)
            host = value
            index += 2
            continue
        if argument == "--ssh-user":
            value = require_value(argument, argv_list, index)
            ssh_user = value
            index += 2
            continue
        if argument == "--ssh-key":
            value = require_value(argument, argv_list, index)
            ssh_key = value
            index += 2
            continue
        if argument == "--root-dir":
            value = require_value(argument, argv_list, index)
            root_dir = value
            index += 2
            continue
        if argument == "--local":
            local_mode = True
            index += 1
            continue
        if argument == "--head":
            head_mode = True
            index += 1
            continue
        if argument == "--skip-uv":
            skip_uv = True
            index += 1
            continue
        if argument == "--skip-venv":
            skip_venv = True
            index += 1
            continue
        if argument == "--skip-dependency-sync":
            skip_dependency_sync = True
            index += 1
            continue
        if argument == "--skip-treeple":
            skip_treeple = True
            index += 1
            continue
        if argument == "--skip-torch-cluster":
            skip_torch_cluster = True
            index += 1
            continue
        if argument == "--skip-postgres":
            skip_postgres = True
            index += 1
            continue
        if argument == "--skip-seaweed":
            skip_seaweed = True
            index += 1
            continue
        if argument == "--skip-symlinks":
            skip_symlinks = True
            index += 1
            continue
        if argument == "--uninstall":
            set_mode("uninstall", argument)
            index += 1
            continue
        if argument == "--rotate-postgres-password":
            set_mode("rotate-postgres-password", argument)
            index += 1
            continue
        raise FrontendUsageError(f"Unknown argument: {argument}")

    return FrontendConfig(
        mode=mode,
        local_mode=local_mode,
        head_mode=head_mode,
        skip_uv=skip_uv,
        skip_venv=skip_venv,
        skip_dependency_sync=skip_dependency_sync,
        skip_treeple=skip_treeple,
        skip_torch_cluster=skip_torch_cluster,
        skip_postgres=skip_postgres,
        skip_seaweed=skip_seaweed,
        skip_symlinks=skip_symlinks,
        host=host,
        ssh_user=ssh_user,
        ssh_key=Path(ssh_key) if ssh_key else None,
        root_dir=root_dir,
    )


class FrontendUsageError(ScriptError):
    """Raised for argument/usage failures that should print help."""


def require_value(flag: str, argv: Sequence[str], index: int) -> str:
    try:
        value = argv[index + 1]
    except IndexError as exc:
        raise FrontendUsageError(f"Missing value for {flag}") from exc
    if value.startswith("--"):
        raise FrontendUsageError(f"Missing value for {flag}")
    return value


def validate_frontend_config(config: FrontendConfig) -> None:
    if config.local_mode and any((config.host, config.ssh_user, config.ssh_key is not None)):
        raise FrontendUsageError("The --local flag cannot be combined with --host, --ssh-user, or --ssh-key.")

    missing_flags: list[str] = []
    if not config.root_dir:
        missing_flags.append("--root-dir")
    if not config.local_mode:
        if not config.host:
            missing_flags.append("--host")
        if not config.ssh_user:
            missing_flags.append("--ssh-user")
        if config.ssh_key is None:
            missing_flags.append("--ssh-key")
    if missing_flags:
        raise FrontendUsageError(f"Missing required arguments: {' '.join(missing_flags)}")

    if config.mode == "rotate-postgres-password" and not config.head_mode:
        raise FrontendUsageError("The --rotate-postgres-password mode requires --head.")

    if config.mode != "install" and has_skip_stage_flags(config):
        raise FrontendUsageError("Skip-stage flags can only be used with install mode.")

    if requires_postgres_password(config):
        get_required_env_value(POSTGRES_PASSWORD_ENV_VAR, os.environ)

    require_command("python3")
    if not config.local_mode:
        require_command("ssh")
        require_command("rsync")


def parse_internal_args(argv: Sequence[str]) -> TargetConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-target-mode", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("install", "rotate-postgres-password", "uninstall"),
        required=True,
    )
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--request-host", required=True)
    parser.add_argument("--head", action="store_true")
    parser.add_argument("--skip-uv", action="store_true")
    parser.add_argument("--skip-venv", action="store_true")
    parser.add_argument("--skip-dependency-sync", action="store_true")
    parser.add_argument("--skip-treeple", action="store_true")
    parser.add_argument("--skip-torch-cluster", action="store_true")
    parser.add_argument("--skip-postgres", action="store_true")
    parser.add_argument("--skip-seaweed", action="store_true")
    parser.add_argument("--skip-symlinks", action="store_true")
    namespace = parser.parse_args(list(argv))
    return TargetConfig(
        mode=str(namespace.mode),
        root_dir=Path(str(namespace.root_dir)).expanduser().resolve(),
        request_host=str(namespace.request_host),
        head_mode=bool(namespace.head),
        skip_uv=bool(namespace.skip_uv),
        skip_venv=bool(namespace.skip_venv),
        skip_dependency_sync=bool(namespace.skip_dependency_sync),
        skip_treeple=bool(namespace.skip_treeple),
        skip_torch_cluster=bool(namespace.skip_torch_cluster),
        skip_postgres=bool(namespace.skip_postgres),
        skip_seaweed=bool(namespace.skip_seaweed),
        skip_symlinks=bool(namespace.skip_symlinks),
    )


@dataclass
class BootstrapTarget:
    config: TargetConfig
    metadata_dir: Path = field(init=False)
    staging_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    state_file: Path = field(init=False)
    remote_hostname: str = field(init=False)
    install_timestamp: str = field(init=False)
    logger: Optional[LineLogger] = field(init=False, default=None)
    runtime_env: dict[str, str] = field(init=False)
    progress: Optional[ProgressTracker] = field(init=False, default=None)
    created_core_paths: list[Path] = field(init=False, default_factory=list)
    created_auxiliary_paths: list[Path] = field(init=False, default_factory=list)
    preexisting_auxiliary_paths: list[Path] = field(init=False, default_factory=list)
    created_symlinks: list[tuple[Path, Path]] = field(init=False, default_factory=list)
    loaded_state: Optional[BootstrapState] = field(init=False, default=None)
    postgres_port: str = field(init=False, default="")
    postgres_password: Optional[str] = field(init=False, default=None)
    install_completed: bool = field(init=False, default=False)

    common_core_rel_paths: tuple[str, ...] = (".venv", "treeple", "cluster")
    head_core_rel_paths: tuple[str, ...] = ("postgres", "seaweedfs")
    aux_rel_paths: tuple[str, ...] = (".numba", "tmp", "tmp/ray", "experiment_data")

    def __post_init__(self) -> None:
        self.metadata_dir = self.config.root_dir / ".noboom-native-bootstrap"
        self.staging_dir = self.metadata_dir / "staging"
        self.logs_dir = self.metadata_dir / "logs"
        self.state_file = self.metadata_dir / "state.json"
        self.remote_hostname = socket.getfqdn() or socket.gethostname()
        self.install_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.runtime_env = dict(os.environ)
        local_bin = str(Path.home() / ".local" / "bin")
        self.runtime_env["PATH"] = f"{local_bin}:{self.runtime_env.get('PATH', '')}"

    def run(self) -> None:
        if self.config.mode == "install":
            self.run_install()
            return
        if self.config.mode == "rotate-postgres-password":
            self.run_rotate_postgres_password()
            return
        if self.config.mode == "uninstall":
            self.run_uninstall()
            return
        raise ScriptError(f"Unsupported mode: {self.config.mode}")

    def start_logging(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{self.config.mode}.log"
        self.logger = LineLogger(log_file)
        self.logger.info(f"Mode: {self.config.mode}")
        self.logger.info(f"Root dir: {self.config.root_dir}")
        self.logger.info(f"Request host: {self.config.request_host}")
        self.logger.info(f"Head mode: {int(self.config.head_mode)}")
        self.logger.info(f"Remote host: {self.remote_hostname}")

    def close_logging(self) -> None:
        if self.progress is not None:
            self.progress.close()
            self.progress = None
        if self.logger is not None:
            self.logger.close()
            self.logger = None

    def require_logger(self) -> LineLogger:
        if self.logger is None:
            raise ScriptError("Logger has not been initialized.")
        return self.logger

    def load_staged_secret(self, key: str, *, required: bool = False) -> Optional[str]:
        secret_path = self.staging_dir / BOOTSTRAP_SECRETS_FILENAME
        if not secret_path.is_file():
            if required:
                raise ScriptError(f"Missing staged bootstrap secret file: {secret_path}")
            return None

        raw_data = json.loads(secret_path.read_text(encoding="utf-8"))
        value = str(raw_data.get(key, "")).strip()
        if not value:
            if required:
                raise ScriptError(f"Missing required bootstrap secret: {key}")
            return None
        return value

    def require_postgres_password(self) -> str:
        if self.postgres_password is None:
            self.postgres_password = self.load_staged_secret(POSTGRES_PASSWORD_ENV_VAR, required=True)
        return self.postgres_password

    def managed_postgres_paths(self) -> dict[str, Path]:
        postgres_root = self.config.root_dir / "postgres"
        postgres_prefix = postgres_root / "pgsql"
        postgres_data = postgres_prefix / "data"
        return {
            "root": postgres_root,
            "prefix": postgres_prefix,
            "data": postgres_data,
            "logfile": postgres_root / "logfile",
            "pg_ctl": postgres_prefix / "bin" / "pg_ctl",
            "postgres": postgres_prefix / "bin" / "postgres",
            "pg_isready": postgres_prefix / "bin" / "pg_isready",
            "psql": postgres_prefix / "bin" / "psql",
        }

    def detect_running_postgres_runtime(self, postgres_data: Path) -> tuple[Optional[str], Optional[str]]:
        pid_file = postgres_data / "postmaster.pid"
        if not pid_file.is_file():
            return None, None

        lines = pid_file.read_text(encoding="utf-8").splitlines()
        process_id = lines[0].strip() if lines else ""
        current_port = lines[3].strip() if len(lines) > 3 and lines[3].strip() else None
        listen_addresses = None

        if process_id.isdigit():
            cmdline_path = Path(f"/proc/{process_id}/cmdline")
            if cmdline_path.is_file():
                arguments = [
                    part.decode("utf-8", errors="replace")
                    for part in cmdline_path.read_bytes().split(b"\0")
                    if part
                ]
                for index, argument in enumerate(arguments):
                    if (
                        argument == "-c"
                        and index + 1 < len(arguments)
                        and arguments[index + 1].startswith("listen_addresses=")
                    ):
                        listen_addresses = arguments[index + 1].split("=", 1)[1]
                        break
                if current_port is None:
                    for index, argument in enumerate(arguments):
                        if argument == "-p" and index + 1 < len(arguments):
                            current_port = arguments[index + 1]
                            break

        return current_port, listen_addresses

    def wait_for_postgres_ready(self, pg_isready: Path, port: str) -> None:
        for attempt in range(30):
            ready_result = subprocess.run(
                [str(pg_isready), "-h", "127.0.0.1", "-p", port, "-U", POSTGRES_USER],
                capture_output=True,
                text=True,
                env=merge_env(self.runtime_env),
                check=False,
            )
            if ready_result.returncode == 0:
                return
            if attempt == 29:
                raise ScriptError("PostgreSQL did not become ready in time.")
            time.sleep(1)

    def validate_postgres_password(self, psql: Path, port: str, password: str) -> None:
        postgres_env = dict(self.runtime_env)
        postgres_env["PGPASSWORD"] = password
        self.run_process(
            [
                str(psql),
                "-h",
                "127.0.0.1",
                "-p",
                port,
                "-U",
                POSTGRES_USER,
                "-d",
                "postgres",
                "-Atqc",
                "SELECT 1",
            ],
            env=postgres_env,
        )

    def rotate_managed_postgres_password(self) -> None:
        postgres_paths = self.managed_postgres_paths()
        pg_ctl = postgres_paths["pg_ctl"]
        postgres_binary = postgres_paths["postgres"]
        pg_isready = postgres_paths["pg_isready"]
        psql = postgres_paths["psql"]
        postgres_data = postgres_paths["data"]
        postgres_password = self.require_postgres_password()

        if not pg_ctl.is_file() or not postgres_binary.is_file() or not psql.is_file() or not pg_isready.is_file():
            raise ScriptError(
                f"Managed PostgreSQL binaries not found under {postgres_paths['prefix']}. "
                "Bootstrap the head node first."
            )
        if not (postgres_data / "PG_VERSION").is_file():
            raise ScriptError(f"Managed PostgreSQL data directory not found: {postgres_data}")

        status_result = subprocess.run(
            [str(pg_ctl), "-D", str(postgres_data), "status"],
            capture_output=True,
            text=True,
            env=merge_env(self.runtime_env),
            check=False,
        )
        was_running = status_result.returncode == 0
        current_port, current_listen_addresses = self.detect_running_postgres_runtime(postgres_data)
        restart_port = current_port or self.loaded_state.postgres_port or self.select_available_postgres_port()
        restart_listen_addresses = current_listen_addresses or ("*" if was_running else "127.0.0.1")

        if was_running:
            self.require_logger().info(
                f"Stopping managed PostgreSQL for password rotation (port {restart_port})"
            )
            self.run_process(
                [str(pg_ctl), "-D", str(postgres_data), "-m", "fast", "stop"],
                env=self.runtime_env,
            )

        rotation_sql = build_postgres_password_rotation_sql(postgres_password)
        self.require_logger().info("Applying in-place PostgreSQL role password rotation")
        self.run_process(
            "printf '%s\\n' {sql} | {postgres} --single -D {data} postgres >/dev/null".format(
                sql=shlex.quote(rotation_sql),
                postgres=shlex.quote(str(postgres_binary)),
                data=shlex.quote(str(postgres_data)),
            ),
            env=self.runtime_env,
            shell=True,
        )

        self.require_logger().info(f"Starting PostgreSQL on port {restart_port} for validation")
        self.run_process(
            [
                str(pg_ctl),
                "-D",
                str(postgres_data),
                "-l",
                str(postgres_paths["logfile"]),
                "-o",
                f"-c listen_addresses='{restart_listen_addresses}' -p {restart_port}",
                "start",
            ],
            env=self.runtime_env,
        )
        self.wait_for_postgres_ready(pg_isready, restart_port)
        self.validate_postgres_password(psql, restart_port, postgres_password)

        if not was_running:
            self.require_logger().info("Stopping temporary PostgreSQL validation instance")
            self.run_process(
                [str(pg_ctl), "-D", str(postgres_data), "-m", "fast", "stop"],
                env=self.runtime_env,
            )

    def sync_postgres_password_to_env_files(self) -> list[Path]:
        updated_paths: list[Path] = []
        postgres_password = self.require_postgres_password()
        for env_path in (
            self.config.root_dir / ".env",
            self.config.root_dir / "mnt" / ".env",
            self.config.root_dir / "mnt" / ".env.base",
        ):
            if rewrite_env_file_postgres_password(env_path, postgres_password):
                updated_paths.append(env_path)
        return updated_paths

    def run_process(
        self,
        command: Sequence[str] | str,
        *,
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        shell: bool = False,
        check: bool = True,
    ) -> str:
        logger = self.require_logger()
        if shell:
            if not isinstance(command, str):
                raise ScriptError("Shell commands must be passed as a string.")
            popen_command: Sequence[str] = ["bash", "-c", command]
            display_command = command
        else:
            if isinstance(command, str):
                raise ScriptError("Non-shell commands must be passed as a sequence.")
            popen_command = [str(part) for part in command]
            display_command = format_command(popen_command)

        process = subprocess.Popen(
            popen_command,
            cwd=str(cwd) if cwd is not None else None,
            env=merge_env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )

        output_chunks: list[str] = []
        assert process.stdout is not None
        for chunk in process.stdout:
            output_chunks.append(chunk)
            logger.emit_raw(chunk)
        return_code = process.wait()
        output_text = "".join(output_chunks)
        if check and return_code != 0:
            raise ScriptError(f"Command failed ({return_code}): {display_command}")
        return output_text

    def command_exists(self, command_name: str, *, env: Optional[Mapping[str, str]] = None) -> bool:
        path_value = None if env is None else env.get("PATH")
        return shutil.which(command_name, path=path_value) is not None

    def ensure_target_command(self, command_name: str) -> None:
        if not self.command_exists(command_name, env=self.runtime_env):
            raise ScriptError(f"Required target command not found: {command_name}")

    def venv_env(self) -> dict[str, str]:
        env = dict(self.runtime_env)
        venv_bin = str(self.config.root_dir / ".venv" / "bin")
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(self.config.root_dir / ".venv")
        return env

    def load_state(self) -> BootstrapState:
        if not self.state_file.is_file():
            raise ScriptError(f"Bootstrap state file not found: {self.state_file}")
        state = BootstrapState.load(self.state_file)
        if state.root_dir != str(self.config.root_dir):
            raise ScriptError(
                f"State root_dir mismatch: expected {self.config.root_dir!r}, found {state.root_dir!r}"
            )
        if state.remote_hostname != self.remote_hostname:
            raise ScriptError(
                f"State remote_hostname mismatch: expected {self.remote_hostname!r}, found {state.remote_hostname!r}"
            )
        self.loaded_state = state
        self.postgres_port = state.postgres_port
        return state

    def write_state(self) -> None:
        state = BootstrapState(
            install_host=self.config.request_host,
            remote_hostname=self.remote_hostname,
            root_dir=str(self.config.root_dir),
            install_timestamp=self.install_timestamp,
            pins={
                "treeple_ref": TREEPLE_REF,
                "torch_cluster_ref": TORCH_CLUSTER_REF,
                "postgres_version": POSTGRES_VERSION,
                "seaweed_version": SEAWEED_VERSION,
            },
            is_head=self.config.head_mode,
            postgres_port=self.postgres_port,
            created_core_paths=[str(path) for path in self.created_core_paths],
            created_auxiliary_paths=[str(path) for path in self.created_auxiliary_paths],
            preexisting_auxiliary_paths=[str(path) for path in self.preexisting_auxiliary_paths],
            created_symlinks=[
                {
                    "path": str(link_path),
                    "target": str(target_path),
                }
                for link_path, target_path in self.created_symlinks
            ],
        )
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(state.to_json(), encoding="utf-8")

    def mark_auxiliary_path_state(self, relative_path: str) -> None:
        absolute_path = self.config.root_dir / relative_path
        if absolute_path.exists() or absolute_path.is_symlink():
            self.preexisting_auxiliary_paths.append(absolute_path)
        else:
            self.created_auxiliary_paths.append(absolute_path)

    def ensure_auxiliary_directories_first_install(self) -> None:
        for relative_path in self.aux_rel_paths:
            self.mark_auxiliary_path_state(relative_path)
        (self.config.root_dir / ".numba").mkdir(parents=True, exist_ok=True)
        (self.config.root_dir / "tmp" / "ray").mkdir(parents=True, exist_ok=True)
        (self.config.root_dir / "experiment_data").mkdir(parents=True, exist_ok=True)

    def ensure_uv(self) -> None:
        logger = self.require_logger()
        if self.command_exists("uv", env=self.runtime_env):
            return
        logger.info("Installing uv in user space")
        self.run_process("curl -LsSf https://astral.sh/uv/install.sh | sh", env=self.runtime_env, shell=True)
        if not self.command_exists("uv", env=self.runtime_env):
            raise ScriptError("uv installation failed")

    def ensure_managed_python(self) -> None:
        self.require_logger().info(f"Ensuring uv-managed CPython {PYTHON_VERSION} is available")
        self.run_process(
            ["uv", "python", "install", "--managed-python", PYTHON_VERSION],
            env=self.runtime_env,
        )

    def ensure_main_venv(self) -> None:
        python_path = self.config.root_dir / ".venv" / "bin" / "python"
        if python_path.is_file():
            return
        self.require_logger().info("Creating main virtual environment")
        self.ensure_managed_python()
        self.run_process(
            [
                "uv",
                "venv",
                str(self.config.root_dir / ".venv"),
                f"--python={PYTHON_VERSION}",
                "--managed-python",
                "--seed",
            ],
            env=self.runtime_env,
        )

    def sync_locked_dependencies(self) -> None:
        self.require_logger().info("Syncing locked benchmark dependencies")
        command = (
            f"source {shlex.quote(str(self.config.root_dir / '.venv' / 'bin' / 'activate'))} && "
            "UV_LINK_MODE=copy uv sync --active --locked --no-install-project --extra benchmark"
        )
        self.run_process(command, cwd=self.staging_dir, env=self.venv_env(), shell=True)

    def validate_treeple(self) -> None:
        self.run_process(
            [
                str(self.config.root_dir / ".venv" / "bin" / "python"),
                "-c",
                "import treeple; print(treeple.__version__)",
            ],
            env=self.venv_env(),
        )

    def detect_cuda_home(self) -> Path:
        preferred = Path("/usr/local/cuda")
        candidate = preferred if (preferred / "bin" / "nvcc").is_file() else None
        if candidate is None:
            matches = sorted(Path("/usr/local").glob("cuda-*"))
            for match in reversed(matches):
                if (match / "bin" / "nvcc").is_file():
                    candidate = match
                    break
        if candidate is None:
            raise ScriptError("CUDA toolkit with nvcc not found under /usr/local/cuda or /usr/local/cuda-*.")
        self.runtime_env["CUDA_HOME"] = str(candidate)
        self.runtime_env["PATH"] = f"{candidate / 'bin'}:{self.runtime_env.get('PATH', '')}"
        lib64_path = str(candidate / "lib64")
        ld_library_path = self.runtime_env.get("LD_LIBRARY_PATH")
        self.runtime_env["LD_LIBRARY_PATH"] = (
            f"{lib64_path}:{ld_library_path}" if ld_library_path else lib64_path
        )
        return candidate

    def validate_torch_cluster(self) -> None:
        self.run_process(
            [
                str(self.config.root_dir / ".venv" / "bin" / "python"),
                "-c",
                (
                    "import torch; import torch_cluster; "
                    "x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device='cuda' if torch.cuda.is_available() else 'cpu'); "
                    "batch = torch.tensor([0, 0], device=x.device); "
                    "edge_index = torch_cluster.knn_graph(x, k=1, batch=batch, loop=False); "
                    "print(torch_cluster.__version__, tuple(edge_index.shape), edge_index.device.type)"
                ),
            ],
            env=self.venv_env(),
        )

    def ensure_git_checkout(
        self,
        repo_url: str,
        repo_dir: Path,
        ref: str,
        *,
        recurse_submodules: bool = False,
    ) -> None:
        if not (repo_dir / ".git").is_dir():
            self.require_logger().info(f"Cloning {repo_url} into {repo_dir}")
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            clone_command = ["git", "clone"]
            if recurse_submodules:
                clone_command.append("--recurse-submodules")
            clone_command.extend([repo_url, str(repo_dir)])
            self.run_process(clone_command, env=self.runtime_env)

        self.run_process(["git", "-C", str(repo_dir), "fetch", "--tags", "origin"], env=self.runtime_env)
        self.run_process(["git", "-C", str(repo_dir), "checkout", "--detach", ref], env=self.runtime_env)
        if (repo_dir / ".gitmodules").is_file():
            self.run_process(
                ["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"],
                env=self.runtime_env,
            )

    def build_treeple(self) -> None:
        build_venv = self.config.root_dir / "treeple" / ".build-venv"
        wheelhouse = self.config.root_dir / "treeple" / "wheelhouse"
        treeple_dir = self.config.root_dir / "treeple"
        build_python = build_venv / "bin" / "python"
        self.require_logger().info("Building treeple")
        self.ensure_git_checkout(
            os.environ.get("TREEPLE_REPOSITORY_URL", "https://github.com/denix56/treeple.git"),
            treeple_dir,
            TREEPLE_REF,
            recurse_submodules=True,
        )

        if not (build_venv / "bin" / "python").is_file():
            self.ensure_managed_python()
            self.run_process(
                [
                    "uv",
                    "venv",
                    str(build_venv),
                    f"--python={PYTHON_VERSION}",
                    "--managed-python",
                    "--seed",
                ],
                env=self.runtime_env,
            )

        self.run_process(["uv", "pip", "install", "--python", str(build_python), "pip"], env=self.runtime_env)
        self.run_process(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(build_python),
                *TREEPLE_BUILD_PACKAGES,
            ],
            env=self.runtime_env,
        )
        numpy_include_dir = self.run_process(
            [
                str(build_python),
                "-c",
                "import numpy; print(numpy.get_include())",
            ],
            env=self.runtime_env,
        ).strip()
        python_pkgconfig_dir = self.ensure_python_pkg_config_metadata(build_venv, build_python)
        treeple_build_env = dict(self.runtime_env)
        treeple_build_env["PATH"] = f"{build_venv / 'bin'}:{treeple_build_env.get('PATH', '')}"
        treeple_build_env["VIRTUAL_ENV"] = str(build_venv)
        treeple_build_env["PYTHON"] = str(build_python)
        treeple_build_env["Python_EXECUTABLE"] = str(build_python)
        treeple_build_env["Python3_EXECUTABLE"] = str(build_python)
        treeple_build_env["Python_ROOT_DIR"] = str(build_venv)
        treeple_build_env["Python3_ROOT_DIR"] = str(build_venv)
        treeple_build_env["Python_FIND_VIRTUALENV"] = "ONLY"
        treeple_build_env["Python3_FIND_VIRTUALENV"] = "ONLY"
        treeple_build_env["NUMPY_INCLUDE_DIR"] = numpy_include_dir
        existing_prefix_path = treeple_build_env.get("CMAKE_PREFIX_PATH", "")
        treeple_build_env["CMAKE_PREFIX_PATH"] = (
            f"{build_venv}{os.pathsep}{existing_prefix_path}" if existing_prefix_path else str(build_venv)
        )
        # Build treeple against the generated pkg-config metadata for the managed
        # build interpreter only. Host PKG_CONFIG_PATH entries can point at a
        # different Python install and make CMake resolve the wrong headers/libs.
        treeple_build_env["PKG_CONFIG_PATH"] = str(python_pkgconfig_dir)
        cmake_arg_parts = [
            treeple_build_env.get("CMAKE_ARGS", "").strip(),
            f"-DPython_EXECUTABLE={build_python}",
            f"-DPython3_EXECUTABLE={build_python}",
            f"-DPython_ROOT_DIR={build_venv}",
            f"-DPython3_ROOT_DIR={build_venv}",
            "-DPython_FIND_VIRTUALENV=ONLY",
            "-DPython3_FIND_VIRTUALENV=ONLY",
        ]
        treeple_build_env["CMAKE_ARGS"] = " ".join(part for part in cmake_arg_parts if part)
        self.run_process(
            f"source {shlex.quote(str(build_venv / 'bin' / 'activate'))} && spin build --clean -j4",
            cwd=treeple_dir,
            env=treeple_build_env,
            shell=True,
        )

        if wheelhouse.exists():
            shutil.rmtree(wheelhouse)
        wheelhouse.mkdir(parents=True, exist_ok=True)
        self.run_process(
            [
                str(build_python),
                "-m",
                "pip",
                "wheel",
                "--wheel-dir",
                str(wheelhouse),
                "--no-deps",
                str(treeple_dir),
            ],
            env=treeple_build_env,
        )
        wheel_paths = sorted(wheelhouse.glob("*.whl"))
        if not wheel_paths:
            raise ScriptError(f"No wheel produced for treeple in {wheelhouse}")
        self.run_process(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.config.root_dir / ".venv" / "bin" / "python"),
                "--no-deps",
                "--reinstall",
                str(wheel_paths[-1]),
            ],
            env=self.runtime_env,
        )
        self.validate_treeple()

    def ensure_python_pkg_config_metadata(self, build_venv: Path, build_python: Path) -> Path:
        pkgconfig_dir = build_venv / ".pkgconfig"
        pkgconfig_dir.mkdir(parents=True, exist_ok=True)

        python_metadata = json.loads(
            self.run_process(
                [
                    str(build_python),
                    "-c",
                    (
                        "import json, sysconfig; "
                        "metadata = {"
                        "'base_prefix': sysconfig.get_config_var('prefix') or '', "
                        "'exec_prefix': sysconfig.get_config_var('exec_prefix') or '', "
                        "'include_dir': sysconfig.get_path('include') or '', "
                        "'platinclude_dir': sysconfig.get_path('platinclude') or '', "
                        "'libdir': sysconfig.get_config_var('LIBDIR') or '', "
                        "'ldlibrary': sysconfig.get_config_var('LDLIBRARY') or '', "
                        "'version': sysconfig.get_config_var('VERSION') or '', "
                        "'libs': sysconfig.get_config_var('LIBS') or '', "
                        "'syslibs': sysconfig.get_config_var('SYSLIBS') or '', "
                        "'linkforshared': sysconfig.get_config_var('LINKFORSHARED') or ''"
                        "}; "
                        "print(json.dumps(metadata))"
                    ),
                ],
                env=self.runtime_env,
            ).strip()
        )

        version = str(python_metadata.get("version", "")).strip()
        if not version:
            version = f"{sysconfig.get_python_version()}"
        major_minor = ".".join(version.split(".")[:2]) or version

        ld_library = str(python_metadata.get("ldlibrary", "")).strip()
        if ld_library.startswith("lib") and "." in ld_library[3:]:
            library_name = ld_library[3:].split(".", 1)[0]
        elif ld_library.startswith("lib"):
            library_name = ld_library[3:]
        else:
            library_name = f"python{major_minor}"

        libdir = str(python_metadata.get("libdir", "")).strip()
        include_dir = str(python_metadata.get("include_dir", "")).strip()
        platinclude_dir = str(python_metadata.get("platinclude_dir", "")).strip()
        include_flags = [f"-I{include_dir}"] if include_dir else []
        if platinclude_dir and platinclude_dir != include_dir:
            include_flags.append(f"-I{platinclude_dir}")

        libs_parts = []
        if libdir:
            libs_parts.append(f"-L{libdir}")
        libs_parts.append(f"-l{library_name}")
        for extra_part in (
            str(python_metadata.get("libs", "")).strip(),
            str(python_metadata.get("syslibs", "")).strip(),
            str(python_metadata.get("linkforshared", "")).strip(),
        ):
            if extra_part:
                libs_parts.append(extra_part)

        pc_template = "\n".join(
            [
                f"prefix={python_metadata.get('base_prefix', '')}",
                f"exec_prefix={python_metadata.get('exec_prefix', '')}",
                f"libdir={libdir}",
                f"includedir={include_dir}",
                "",
                "Name: Python",
                "Description: Bootstrap-managed Python metadata",
                f"Version: {major_minor}",
                "Requires:",
                "Libs: " + " ".join(part for part in libs_parts if part),
                "Cflags: " + " ".join(include_flags),
                "",
            ]
        )

        for file_name in ("python.pc", "python3.pc", f"python-{major_minor}.pc"):
            (pkgconfig_dir / file_name).write_text(pc_template, encoding="utf-8")

        return pkgconfig_dir

    def detect_torch_cuda_arch_list(self) -> str:
        nvidia_smi = shutil.which("nvidia-smi", path=self.runtime_env.get("PATH"))
        if nvidia_smi is not None:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                env=merge_env(self.runtime_env),
                check=False,
            )
            if result.returncode == 0:
                caps = sorted(
                    {line.strip().replace(" ", "") for line in result.stdout.splitlines() if line.strip()},
                    key=lambda item: tuple(int(part) for part in item.split(".")),
                )
                if caps:
                    caps[-1] = f"{caps[-1]}+PTX"
                    return " ".join(caps)

        result = subprocess.run(
            [
                str(self.config.root_dir / ".venv" / "bin" / "python"),
                "-c",
                (
                    "import torch; "
                    "assert torch.cuda.is_available(); "
                    "caps = sorted({f'{major}.{minor}' for index in range(torch.cuda.device_count()) "
                    "for major, minor in [torch.cuda.get_device_capability(index)]}, "
                    "key=lambda item: tuple(int(part) for part in item.split('.'))); "
                    "assert caps; caps[-1] = f'{caps[-1]}+PTX'; print(' '.join(caps))"
                ),
            ],
            capture_output=True,
            text=True,
            env=merge_env(self.venv_env()),
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

        return "8.0 8.6 8.9 9.0+PTX"

    def build_torch_cluster(self) -> None:
        build_venv = self.config.root_dir / "cluster" / ".build-venv"
        wheelhouse = self.config.root_dir / "cluster" / "wheelhouse"
        cluster_dir = self.config.root_dir / "cluster"

        self.require_logger().info("Building torch-cluster")
        self.detect_cuda_home()
        self.ensure_git_checkout(
            "https://github.com/rusty1s/pytorch_cluster.git",
            cluster_dir,
            TORCH_CLUSTER_REF,
        )

        if not (build_venv / "bin" / "python").is_file():
            self.ensure_managed_python()
            self.run_process(
                [
                    "uv",
                    "venv",
                    str(build_venv),
                    f"--python={PYTHON_VERSION}",
                    "--managed-python",
                    "--seed",
                ],
                env=self.runtime_env,
            )

        self.run_process(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(build_venv / "bin" / "python"),
                "setuptools",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            env=self.runtime_env,
        )

        if wheelhouse.exists():
            shutil.rmtree(wheelhouse)
        wheelhouse.mkdir(parents=True, exist_ok=True)
        torch_cuda_arch_list = self.detect_torch_cuda_arch_list()
        self.require_logger().info(f"Using TORCH_CUDA_ARCH_LIST={torch_cuda_arch_list}")
        build_env = dict(self.runtime_env)
        build_env["TORCH_CUDA_ARCH_LIST"] = torch_cuda_arch_list
        build_env["FORCE_CUDA"] = "1"
        self.run_process(
            [
                "uv",
                "build",
                "--python",
                str(build_venv / "bin" / "python"),
                "--out-dir",
                str(wheelhouse),
            ],
            cwd=cluster_dir,
            env=build_env,
        )
        wheel_paths = sorted(wheelhouse.glob("*.whl"))
        if not wheel_paths:
            raise ScriptError(f"No wheel produced for torch-cluster in {wheelhouse}")
        self.run_process(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.config.root_dir / ".venv" / "bin" / "python"),
                "--no-deps",
                "--reinstall",
                str(wheel_paths[-1]),
            ],
            env=self.runtime_env,
        )
        self.validate_torch_cluster()

    def ensure_treeple_ready(self) -> None:
        try:
            self.validate_treeple()
        except ScriptError:
            self.build_treeple()

    def ensure_torch_cluster_ready(self) -> None:
        try:
            self.validate_torch_cluster()
        except ScriptError:
            self.build_torch_cluster()

    def select_available_postgres_port(self) -> str:
        for port in range(5432, 5501):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return str(port)
        raise ScriptError("No available PostgreSQL port found in 5432-5500.")

    def resolve_postgres_port(self) -> None:
        if self.postgres_port:
            return
        if self.loaded_state is not None and self.loaded_state.postgres_port:
            self.postgres_port = self.loaded_state.postgres_port
        else:
            self.postgres_port = self.select_available_postgres_port()
        self.require_logger().info(f"Using managed PostgreSQL port {self.postgres_port}")

    def ensure_postgres_ready(self) -> None:
        postgres_root = self.config.root_dir / "postgres"
        postgres_prefix = postgres_root / "pgsql"
        postgres_data = postgres_prefix / "data"
        postgres_source = postgres_root / f"postgresql-{POSTGRES_VERSION}"
        postgres_archive = postgres_root / f"postgresql-{POSTGRES_VERSION}.tar.bz2"
        password_file = self.metadata_dir / "postgres-password.txt"
        pg_ctl = postgres_prefix / "bin" / "pg_ctl"
        pg_isready = postgres_prefix / "bin" / "pg_isready"
        psql = postgres_prefix / "bin" / "psql"
        postgres_password = self.require_postgres_password()

        self.resolve_postgres_port()

        if not pg_ctl.is_file():
            self.require_logger().info(f"Building PostgreSQL {POSTGRES_VERSION}")
            postgres_root.mkdir(parents=True, exist_ok=True)
            self.run_process(["curl", "-fL", POSTGRES_ARCHIVE_URL, "-o", str(postgres_archive)], env=self.runtime_env)
            if postgres_source.exists():
                shutil.rmtree(postgres_source)
            self.run_process(["tar", "-xjf", str(postgres_archive), "-C", str(postgres_root)], env=self.runtime_env)
            self.run_process(
                "./configure --prefix={prefix} --without-icu --without-readline && "
                "make -j\"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)\" && "
                "make install".format(prefix=shlex.quote(str(postgres_prefix))),
                cwd=postgres_source,
                env=self.runtime_env,
                shell=True,
            )

        if not (postgres_data / "PG_VERSION").is_file():
            self.require_logger().info("Initializing PostgreSQL data directory")
            postgres_data.mkdir(parents=True, exist_ok=True)
            password_file.write_text(f"{postgres_password}\n", encoding="utf-8")
            password_file.chmod(0o600)
            try:
                self.run_process(
                    [
                        str(postgres_prefix / "bin" / "initdb"),
                        "-D",
                        str(postgres_data),
                        f"--username={POSTGRES_USER}",
                        f"--pwfile={password_file}",
                        "--auth-local=scram-sha-256",
                        "--auth-host=scram-sha-256",
                    ],
                    env=self.runtime_env,
                )
            finally:
                if password_file.exists():
                    password_file.unlink()

        status_result = subprocess.run(
            [str(pg_ctl), "-D", str(postgres_data), "status"],
            capture_output=True,
            text=True,
            env=merge_env(self.runtime_env),
            check=False,
        )
        if status_result.returncode != 0:
            self.require_logger().info("Starting PostgreSQL")
            self.run_process(
                [
                    str(pg_ctl),
                    "-D",
                    str(postgres_data),
                    "-l",
                    str(postgres_root / "logfile"),
                    "-o",
                    f"-c listen_addresses='127.0.0.1' -p {self.postgres_port}",
                    "start",
                ],
                env=self.runtime_env,
            )

        for attempt in range(30):
            ready_result = subprocess.run(
                [str(pg_isready), "-h", "127.0.0.1", "-p", self.postgres_port, "-U", POSTGRES_USER],
                capture_output=True,
                text=True,
                env=merge_env(self.runtime_env),
                check=False,
            )
            if ready_result.returncode == 0:
                break
            if attempt == 29:
                raise ScriptError("PostgreSQL did not become ready in time.")
            time.sleep(1)

        self.require_logger().info("Applying database bootstrap SQL")
        postgres_env = dict(self.runtime_env)
        postgres_env["PGPASSWORD"] = postgres_password
        postgres_env[POSTGRES_PASSWORD_ENV_VAR] = postgres_password
        self.run_process(
            [
                str(psql),
                "-h",
                "127.0.0.1",
                "-p",
                self.postgres_port,
                "-U",
                POSTGRES_USER,
                "-d",
                "postgres",
                "-v",
                "ON_ERROR_STOP=1",
                "-f",
                str(self.staging_dir / "init-db.sql"),
            ],
            env=postgres_env,
        )

    def ensure_seaweed_binary(self) -> None:
        seaweed_root = self.config.root_dir / "seaweedfs"
        archive_path = seaweed_root / "linux_amd64_full.tar.gz"
        weed_binary = seaweed_root / "weed"

        seaweed_root.mkdir(parents=True, exist_ok=True)
        if weed_binary.is_file():
            version_result = subprocess.run(
                [str(weed_binary), "version"],
                capture_output=True,
                text=True,
                env=merge_env(self.runtime_env),
                check=False,
            )
            if version_result.returncode == 0 and f" {SEAWEED_VERSION} " in version_result.stdout:
                self.require_logger().info(f"SeaweedFS {SEAWEED_VERSION} already present")
                return

        self.require_logger().info(f"Downloading SeaweedFS {SEAWEED_VERSION}")
        self.run_process(["curl", "-fL", SEAWEED_ARCHIVE_URL, "-o", str(archive_path)], env=self.runtime_env)
        self.run_process(["tar", "-xzf", str(archive_path), "-C", str(seaweed_root), "weed"], env=self.runtime_env)
        weed_binary.chmod(0o755)
        self.run_process([str(weed_binary), "version"], env=self.runtime_env)

    def ensure_managed_symlink(self, link_path: Path, target_path: Path) -> None:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink():
            current_target = Path(os.path.realpath(link_path))
            expected_target = Path(os.path.realpath(target_path))
            if current_target == expected_target:
                return
            link_path.unlink()
        elif link_path.exists():
            if link_path.is_dir():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()

        link_path.symlink_to(target_path)
        pair = (link_path, target_path)
        if pair not in self.created_symlinks:
            self.created_symlinks.append(pair)

    def ensure_symlinked_binaries(self) -> None:
        self.ensure_managed_symlink(self.config.root_dir / ".venv" / "bin" / "psql", self.config.root_dir / "postgres" / "pgsql" / "bin" / "psql")
        self.ensure_managed_symlink(self.config.root_dir / ".venv" / "bin" / "pg_ctl", self.config.root_dir / "postgres" / "pgsql" / "bin" / "pg_ctl")
        self.ensure_managed_symlink(self.config.root_dir / ".venv" / "bin" / "initdb", self.config.root_dir / "postgres" / "pgsql" / "bin" / "initdb")
        self.ensure_managed_symlink(self.config.root_dir / ".venv" / "bin" / "postgres", self.config.root_dir / "postgres" / "pgsql" / "bin" / "postgres")
        self.ensure_managed_symlink(self.config.root_dir / ".venv" / "bin" / "weed", self.config.root_dir / "seaweedfs" / "weed")

    def discover_preexisting_managed_paths(self) -> list[Path]:
        conflicts: list[Path] = []
        for relative_path in self.common_core_rel_paths:
            absolute_path = self.config.root_dir / relative_path
            if absolute_path.exists() or absolute_path.is_symlink():
                conflicts.append(absolute_path)
        if self.config.head_mode:
            for relative_path in self.head_core_rel_paths:
                absolute_path = self.config.root_dir / relative_path
                if absolute_path.exists() or absolute_path.is_symlink():
                    conflicts.append(absolute_path)
        return conflicts

    def remove_stale_managed_paths(self, conflicts: Sequence[Path]) -> None:
        if not conflicts:
            return
        self.require_logger().info("Removing preexisting managed paths before provisioning.")
        self.stop_managed_services()
        for path in conflicts:
            self.safe_remove_path(path)

    def first_install_preflight(self) -> None:
        self.created_core_paths = [self.metadata_dir]
        conflicts = self.discover_preexisting_managed_paths()
        if conflicts:
            self.remove_stale_managed_paths(conflicts)
        self.ensure_auxiliary_directories_first_install()

    def mark_first_install_core_paths(self) -> None:
        for relative_path in self.common_core_rel_paths:
            self.created_core_paths.append(self.config.root_dir / relative_path)
        if self.config.head_mode:
            for relative_path in self.head_core_rel_paths:
                self.created_core_paths.append(self.config.root_dir / relative_path)

    def safe_remove_path(self, absolute_path: Path) -> None:
        logger = self.require_logger()
        if not absolute_path.exists() and not absolute_path.is_symlink():
            logger.info(f"Path already absent: {absolute_path}")
            return

        if absolute_path.is_symlink():
            resolved_path = Path(os.path.realpath(absolute_path))
            if str(resolved_path) and not str(resolved_path).startswith(str(self.config.root_dir)):
                logger.warn(f"Skipping path replaced with external symlink target: {absolute_path}")
                return
            absolute_path.unlink()
            logger.info(f"Removed managed path: {absolute_path}")
            return

        if absolute_path == self.config.root_dir / "tmp":
            if absolute_path.is_dir() and any(absolute_path.iterdir()):
                logger.warn(f"Skipping non-empty bootstrap-created tmp directory: {absolute_path}")
                return

        if absolute_path.is_dir():
            shutil.rmtree(absolute_path)
        else:
            absolute_path.unlink()
        logger.info(f"Removed managed path: {absolute_path}")

    def remove_recorded_symlink(self, link_path: Path, target_path: Path) -> None:
        logger = self.require_logger()
        if not link_path.exists() and not link_path.is_symlink():
            logger.info(f"Symlink already absent: {link_path}")
            return
        if not link_path.is_symlink():
            logger.warn(f"Skipping non-symlink path recorded as managed symlink: {link_path}")
            return
        current_target = Path(os.path.realpath(link_path))
        expected_target = Path(os.path.realpath(target_path))
        if current_target == expected_target:
            link_path.unlink()
            logger.info(f"Removed managed symlink: {link_path}")
            return
        logger.warn(f"Skipping symlink with unexpected target: {link_path}")

    def stop_managed_services(self) -> None:
        pg_ctl = self.config.root_dir / "postgres" / "pgsql" / "bin" / "pg_ctl"
        postgres_data = self.config.root_dir / "postgres" / "pgsql" / "data"
        if pg_ctl.is_file():
            status_result = subprocess.run(
                [str(pg_ctl), "-D", str(postgres_data), "status"],
                capture_output=True,
                text=True,
                env=merge_env(self.runtime_env),
                check=False,
            )
            if status_result.returncode == 0:
                self.require_logger().info("Stopping managed PostgreSQL")
                subprocess.run(
                    [str(pg_ctl), "-D", str(postgres_data), "-m", "fast", "stop"],
                    capture_output=True,
                    text=True,
                    env=merge_env(self.runtime_env),
                    check=False,
                )

        pgrep_path = shutil.which("pgrep", path=self.runtime_env.get("PATH"))
        pkill_path = shutil.which("pkill", path=self.runtime_env.get("PATH"))
        if pgrep_path is None or pkill_path is None:
            return

        seaweed_binary = self.config.root_dir / "seaweedfs" / "weed"
        pgrep_result = subprocess.run(
            [pgrep_path, "-f", str(seaweed_binary)],
            capture_output=True,
            text=True,
            env=merge_env(self.runtime_env),
            check=False,
        )
        if pgrep_result.returncode == 0:
            self.require_logger().info("Stopping managed SeaweedFS")
            subprocess.run(
                [pkill_path, "-f", str(seaweed_binary)],
                capture_output=True,
                text=True,
                env=merge_env(self.runtime_env),
                check=False,
            )
            time.sleep(1)

    def remove_managed_symlinks_from_state(self, state: BootstrapState) -> None:
        for entry in state.created_symlinks:
            self.remove_recorded_symlink(Path(entry["path"]), Path(entry["target"]))

    def remove_managed_auxiliary_paths_from_state(self, state: BootstrapState) -> None:
        for path_value in state.created_auxiliary_paths:
            path = Path(path_value)
            if path == self.config.root_dir / "tmp":
                continue
            self.safe_remove_path(path)
        if str(self.config.root_dir / "tmp") in state.created_auxiliary_paths:
            self.safe_remove_path(self.config.root_dir / "tmp")

    def remove_managed_core_paths_from_state(self, state: BootstrapState, *, preserve_metadata: bool) -> None:
        for path_value in state.created_core_paths:
            path = Path(path_value)
            if preserve_metadata and path == self.metadata_dir:
                continue
            self.safe_remove_path(path)

        if not preserve_metadata:
            self.safe_remove_path(self.metadata_dir)

    def clear_existing_bootstrap_for_reinstall(self, state: BootstrapState) -> None:
        self.require_logger().info("Removing existing bootstrap-managed install before reprovisioning.")
        self.stop_managed_services()
        self.remove_managed_symlinks_from_state(state)
        self.remove_managed_auxiliary_paths_from_state(state)
        self.remove_managed_core_paths_from_state(state, preserve_metadata=True)
        if self.state_file.exists():
            self.state_file.unlink()
        self.loaded_state = None
        self.created_core_paths.clear()
        self.created_auxiliary_paths.clear()
        self.preexisting_auxiliary_paths.clear()
        self.created_symlinks.clear()

    def cleanup_failed_first_install(self) -> None:
        if self.state_file.exists() or self.install_completed:
            return
        logger = self.require_logger()
        logger.warn("Install failed before bootstrap ownership was recorded; removing bootstrap-created paths.")

        pg_ctl = self.config.root_dir / "postgres" / "pgsql" / "bin" / "pg_ctl"
        postgres_data = self.config.root_dir / "postgres" / "pgsql" / "data"
        if pg_ctl.is_file():
            status_result = subprocess.run(
                [str(pg_ctl), "-D", str(postgres_data), "status"],
                capture_output=True,
                text=True,
                env=merge_env(self.runtime_env),
                check=False,
            )
            if status_result.returncode == 0:
                subprocess.run(
                    [str(pg_ctl), "-D", str(postgres_data), "-m", "fast", "stop"],
                    capture_output=True,
                    text=True,
                    env=merge_env(self.runtime_env),
                    check=False,
                )

        for path in self.created_auxiliary_paths:
            self.safe_remove_path(path)
        for path in self.created_core_paths:
            self.safe_remove_path(path)

    def compute_install_step_total(self) -> int:
        total = 1
        if not self.config.skip_uv:
            total += 1
        if not self.config.skip_venv:
            total += 1
        if not self.config.skip_dependency_sync:
            total += 1
        if not self.config.skip_treeple:
            total += 1
        if not self.config.skip_torch_cluster:
            total += 1
        if self.config.head_mode:
            if not self.config.skip_postgres:
                total += 1
            if not self.config.skip_seaweed:
                total += 1
            if not self.config.skip_symlinks:
                total += 1
        return total

    def required_install_commands(self) -> tuple[str, ...]:
        required: list[str] = ["python3"]
        if not self.config.skip_uv or not self.config.skip_postgres or not self.config.skip_seaweed:
            required.append("curl")
        if not self.config.skip_treeple or not self.config.skip_torch_cluster:
            required.append("git")
        if not self.config.skip_treeple or not self.config.skip_postgres or not self.config.skip_seaweed:
            required.append("tar")
        if not self.config.skip_treeple or not self.config.skip_torch_cluster or not self.config.skip_postgres:
            required.append("g++")
        if not self.config.skip_treeple:
            required.append("cmake")
        if not self.config.skip_postgres:
            required.append("make")
        return tuple(dict.fromkeys(required))

    def run_rotate_postgres_password(self) -> None:
        self.start_logging()
        try:
            self.progress = ProgressTracker(3, "Bootstrap Rotate PG Password", self.require_logger())
            self.progress.step("Loading bootstrap state")
            state = self.load_state()
            if not state.is_head:
                raise ScriptError(
                    "The managed bootstrap state for this root is not a head installation. "
                    "PostgreSQL password rotation only applies to head roots."
                )
            self.postgres_password = self.load_staged_secret(POSTGRES_PASSWORD_ENV_VAR, required=True)

            self.progress.step("Rotating managed PostgreSQL password")
            self.rotate_managed_postgres_password()

            self.progress.step("Updating managed environment files")
            updated_paths = self.sync_postgres_password_to_env_files()
            if updated_paths:
                for updated_path in updated_paths:
                    self.require_logger().info(f"Updated PostgreSQL password references in {updated_path}")
            else:
                self.require_logger().info("No managed env files were present to update.")
        finally:
            self.close_logging()

    def run_install(self) -> None:
        self.start_logging()
        try:
            had_existing_state = False
            stale_managed_paths: list[Path] = []
            for command_name in self.required_install_commands():
                self.ensure_target_command(command_name)

            if not (self.staging_dir / "pyproject.toml").is_file() or not (self.staging_dir / "uv.lock").is_file() or not (self.staging_dir / "init-db.sql").is_file():
                raise ScriptError(f"Missing staged bootstrap inputs under {self.staging_dir}")
            if requires_postgres_password(self.config):
                self.postgres_password = self.load_staged_secret(POSTGRES_PASSWORD_ENV_VAR, required=True)

            if self.state_file.is_file():
                had_existing_state = True
                state = self.load_state()
                if state.is_head:
                    if not self.config.head_mode:
                        self.require_logger().info("Existing head bootstrap detected; keeping head-specific setup enabled.")
                    self.config = TargetConfig(
                        mode=self.config.mode,
                        root_dir=self.config.root_dir,
                        request_host=self.config.request_host,
                        head_mode=True,
                        skip_uv=self.config.skip_uv,
                        skip_venv=self.config.skip_venv,
                        skip_dependency_sync=self.config.skip_dependency_sync,
                        skip_treeple=self.config.skip_treeple,
                        skip_torch_cluster=self.config.skip_torch_cluster,
                        skip_postgres=self.config.skip_postgres,
                        skip_seaweed=self.config.skip_seaweed,
                        skip_symlinks=self.config.skip_symlinks,
                    )
                elif self.config.head_mode:
                    self.require_logger().info(
                        "Existing worker bootstrap detected; reinstalling with head-specific setup enabled."
                    )
            else:
                stale_managed_paths = self.discover_preexisting_managed_paths()

            self.progress = ProgressTracker(
                self.compute_install_step_total() + (1 if had_existing_state or stale_managed_paths else 0),
                "Bootstrap Install",
                self.require_logger(),
            )
            if had_existing_state:
                self.progress.step("Removing existing bootstrap-managed install")
                self.clear_existing_bootstrap_for_reinstall(state)
            elif stale_managed_paths:
                self.progress.step("Removing preexisting managed paths")
                self.remove_stale_managed_paths(stale_managed_paths)

            self.first_install_preflight()
            self.mark_first_install_core_paths()

            if self.config.skip_uv:
                self.require_logger().info("Skipping uv bootstrap by request.")
            else:
                self.progress.step("Ensuring uv is available")
                self.ensure_uv()

            if self.config.skip_venv:
                self.require_logger().info("Skipping main virtual environment creation by request.")
            else:
                self.progress.step("Ensuring the main virtual environment exists")
                self.ensure_main_venv()

            if self.config.skip_dependency_sync:
                self.require_logger().info("Skipping dependency sync by request.")
            else:
                self.progress.step("Syncing locked benchmark dependencies")
                self.sync_locked_dependencies()

            if self.config.skip_treeple:
                self.require_logger().info("Skipping treeple build and validation by request.")
            else:
                self.progress.step("Validating or building treeple")
                self.ensure_treeple_ready()

            if self.config.skip_torch_cluster:
                self.require_logger().info("Skipping torch-cluster build and validation by request.")
            else:
                self.progress.step("Validating or building torch-cluster")
                self.ensure_torch_cluster_ready()
            if self.config.head_mode:
                if self.config.skip_postgres:
                    self.require_logger().info("Skipping PostgreSQL setup by request.")
                else:
                    self.progress.step("Installing or validating PostgreSQL")
                    self.ensure_postgres_ready()
                if self.config.skip_seaweed:
                    self.require_logger().info("Skipping SeaweedFS setup by request.")
                else:
                    self.progress.step("Installing or validating SeaweedFS")
                    self.ensure_seaweed_binary()
                if self.config.skip_symlinks:
                    self.require_logger().info("Skipping head-specific symlink creation by request.")
                else:
                    self.progress.step("Linking head-specific binaries into the virtual environment")
                    self.ensure_symlinked_binaries()
            else:
                self.require_logger().info("Skipping head-specific setup. Use --head on the Ray head node.")

            self.progress.step("Finalizing bootstrap state")
            if self.loaded_state is None:
                self.write_state()
                self.require_logger().info(f"Wrote bootstrap state to {self.state_file}")
            else:
                self.require_logger().info("Reconciliation install completed without changing ownership metadata.")

            self.install_completed = True
        except Exception:
            if self.loaded_state is None:
                self.cleanup_failed_first_install()
            raise
        finally:
            self.close_logging()

    def run_uninstall(self) -> None:
        state = self.load_state()
        self.start_logging()
        try:
            self.progress = ProgressTracker(4, "Bootstrap Uninstall", self.require_logger())
            self.progress.step("Stopping managed services")
            self.stop_managed_services()

            self.progress.step("Removing managed symlinks")
            self.remove_managed_symlinks_from_state(state)

            self.progress.step("Removing managed auxiliary paths")
            self.remove_managed_auxiliary_paths_from_state(state)

            self.progress.step("Removing managed core paths and metadata")
            self.remove_managed_core_paths_from_state(state, preserve_metadata=False)
        finally:
            self.close_logging()


def main(argv: Sequence[str]) -> int:
    if argv and argv[0] == "--internal-target-mode":
        target = BootstrapTarget(parse_internal_args(argv))
        target.run()
        return 0

    try:
        return frontend_main(argv)
    except FrontendUsageError as exc:
        print(str(exc), file=sys.stderr)
        print_usage(sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ScriptError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
