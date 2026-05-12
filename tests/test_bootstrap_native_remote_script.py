from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "bootstrap_native_remote.sh"
PYTHON_SCRIPT_PATH = REPO_ROOT / "scripts" / "bootstrap_native_remote.py"


def load_bootstrap_python_module() -> Any:
    module_name = "bootstrap_native_remote_test_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, PYTHON_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load bootstrap module from {PYTHON_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_script(*args: str, env: Optional[Mapping[str, str]] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=dict(env) if env is not None else None,
        check=False,
    )


def test_bootstrap_native_remote_script_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_bootstrap_native_remote_python_script_compiles() -> None:
    result = subprocess.run(
        ["python3", "-m", "py_compile", str(PYTHON_SCRIPT_PATH)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_init_db_sql_keeps_postgres_password_interpolation_outside_do_blocks() -> None:
    init_db_path = REPO_ROOT / "docker" / "db" / "scripts" / "init-db.sql"
    init_db_sql = init_db_path.read_text(encoding="utf-8")

    assert "SELECT format('CREATE ROLE noboom LOGIN PASSWORD %L', :'postgres_password')" in init_db_sql
    assert "EXECUTE format('CREATE ROLE noboom LOGIN PASSWORD %L', :'postgres_password');" not in init_db_sql


def test_bootstrap_native_remote_treeple_build_uses_explicit_build_packages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PKG_CONFIG_PATH", "/tmp/ci/lib/pkgconfig")
    bootstrap_module = load_bootstrap_python_module()
    target = bootstrap_module.BootstrapTarget(
        bootstrap_module.TargetConfig(
            mode="install",
            root_dir=tmp_path / "native-root",
            request_host="local",
            head_mode=False,
            skip_uv=False,
            skip_venv=False,
            skip_dependency_sync=False,
            skip_treeple=False,
            skip_torch_cluster=False,
            skip_postgres=False,
            skip_seaweed=False,
            skip_symlinks=False,
        )
    )

    calls: list[dict[str, object]] = []
    dummy_logger = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        warn=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        emit_raw=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(target, "require_logger", lambda: dummy_logger)
    monkeypatch.setattr(target, "ensure_git_checkout", lambda *args, **kwargs: None)
    monkeypatch.setattr(target, "validate_treeple", lambda: None)

    def fake_run_process(
        command: object,
        *,
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        shell: bool = False,
        check: bool = True,
    ) -> str:
        del cwd, check
        calls.append(
            {
                "command": command,
                "env": dict(env) if env is not None else None,
                "shell": shell,
            }
        )
        if shell:
            return ""
        if not isinstance(command, list):
            return ""
        if command == [
            str(target.config.root_dir / "treeple" / ".build-venv" / "bin" / "python"),
            "-c",
            "import numpy; print(numpy.get_include())",
        ]:
            return "/fake/numpy/include\n"
        if (
            len(command) == 3
            and command[0] == str(target.config.root_dir / "treeple" / ".build-venv" / "bin" / "python")
            and command[1] == "-c"
            and "sysconfig.get_config_var('LDLIBRARY')" in command[2]
        ):
            return json.dumps(
                {
                    "base_prefix": str(target.config.root_dir / "treeple" / ".build-venv"),
                    "exec_prefix": str(target.config.root_dir / "treeple" / ".build-venv"),
                    "include_dir": "/fake/python/include",
                    "platinclude_dir": "/fake/python/include",
                    "libdir": "/fake/python/lib",
                    "ldlibrary": f"libpython{bootstrap_module.PYTHON_VERSION}.so",
                    "version": bootstrap_module.PYTHON_VERSION,
                    "libs": "-ldl",
                    "syslibs": "-lm",
                    "linkforshared": "",
                }
            )
        if command[:3] == [str(target.config.root_dir / "treeple" / ".build-venv" / "bin" / "python"), "-m", "pip"] and "wheel" in command:
            wheelhouse = target.config.root_dir / "treeple" / "wheelhouse"
            wheelhouse.mkdir(parents=True, exist_ok=True)
            (wheelhouse / "treeple-0.10.3-py3-none-any.whl").write_text("", encoding="utf-8")
        return ""

    monkeypatch.setattr(target, "run_process", fake_run_process)

    target.build_treeple()

    managed_python_install_calls = [
        call["command"]
        for call in calls
        if call["command"] == ["uv", "python", "install", "--managed-python", bootstrap_module.PYTHON_VERSION]
    ]
    assert managed_python_install_calls

    dependency_install_commands = [
        call["command"]
        for call in calls
        if isinstance(call["command"], list)
        and call["command"][:4] == ["uv", "pip", "install", "--python"]
        and any(package in call["command"] for package in bootstrap_module.TREEPLE_BUILD_PACKAGES)
    ]

    assert dependency_install_commands
    dependency_install_command = dependency_install_commands[0]
    assert "-r" not in dependency_install_command
    assert not any(str(target.config.root_dir / "treeple" / "build_requirements.txt") == str(part) for part in dependency_install_command)
    for package in bootstrap_module.TREEPLE_BUILD_PACKAGES:
        assert package in dependency_install_command

    spin_build_calls = [
        call
        for call in calls
        if call["shell"] is True and isinstance(call["command"], str) and "spin build --clean -j4" in call["command"]
    ]
    assert spin_build_calls
    spin_build_call = spin_build_calls[0]
    assert f"source {target.config.root_dir / 'treeple' / '.build-venv' / 'bin' / 'activate'}" in str(spin_build_call["command"])
    spin_build_env = spin_build_call["env"]
    assert spin_build_env is not None
    assert spin_build_env["PYTHON"] == str(target.config.root_dir / "treeple" / ".build-venv" / "bin" / "python")
    assert spin_build_env["Python_EXECUTABLE"] == spin_build_env["PYTHON"]
    assert spin_build_env["Python3_EXECUTABLE"] == spin_build_env["PYTHON"]
    assert spin_build_env["Python_FIND_VIRTUALENV"] == "ONLY"
    assert spin_build_env["Python3_FIND_VIRTUALENV"] == "ONLY"
    assert spin_build_env["NUMPY_INCLUDE_DIR"] == "/fake/numpy/include"
    pkgconfig_dir = target.config.root_dir / "treeple" / ".build-venv" / ".pkgconfig"
    assert spin_build_env["PKG_CONFIG_PATH"] == str(pkgconfig_dir)
    assert (pkgconfig_dir / "python.pc").is_file()
    assert (pkgconfig_dir / "python3.pc").is_file()
    assert (pkgconfig_dir / f"python-{bootstrap_module.PYTHON_VERSION}.pc").is_file()
    assert "Name: Python" in (pkgconfig_dir / "python.pc").read_text(encoding="utf-8")


def test_bootstrap_native_remote_script_requires_all_flags() -> None:
    result = run_script()

    assert result.returncode != 0
    assert "Missing required arguments" in result.stderr
    assert "--host" in result.stderr
    assert "--ssh-user" in result.stderr
    assert "--ssh-key" in result.stderr
    assert "--root-dir" in result.stderr


def test_bootstrap_native_remote_script_rejects_unknown_argument() -> None:
    result = run_script("--bogus")

    assert result.returncode != 0
    assert "Unknown argument" in result.stderr


def test_bootstrap_native_remote_script_accepts_skip_torch_cluster_flag() -> None:
    result = run_script("--skip-torch-cluster")

    assert result.returncode != 0
    assert "Missing required arguments" in result.stderr
    assert "Unknown argument" not in result.stderr


def test_bootstrap_native_remote_script_accepts_all_skip_flags() -> None:
    skip_flags = (
        "--skip-uv",
        "--skip-venv",
        "--skip-dependency-sync",
        "--skip-treeple",
        "--skip-torch-cluster",
        "--skip-postgres",
        "--skip-seaweed",
        "--skip-symlinks",
    )
    result = run_script(*skip_flags)

    assert result.returncode != 0
    assert "Missing required arguments" in result.stderr
    assert "Unknown argument" not in result.stderr


def test_bootstrap_native_remote_script_accepts_head_flag() -> None:
    result = run_script("--head")

    assert result.returncode != 0
    assert "Missing required arguments" in result.stderr
    assert "Unknown argument" not in result.stderr


def test_bootstrap_native_remote_script_accepts_rotate_postgres_password_flag() -> None:
    result = run_script("--rotate-postgres-password")

    assert result.returncode != 0
    assert "Missing required arguments" in result.stderr
    assert "Unknown argument" not in result.stderr


def test_bootstrap_native_remote_script_local_mode_only_requires_root_dir() -> None:
    result = run_script("--local")
    error_line = result.stderr.splitlines()[0]

    assert result.returncode != 0
    assert error_line == "Missing required arguments: --root-dir"


def test_bootstrap_native_remote_script_rejects_local_mode_with_ssh_flags() -> None:
    result = run_script(
        "--local",
        "--host",
        "127.0.0.1",
        "--root-dir",
        "/tmp/noboom-local",
    )

    assert result.returncode != 0
    assert "cannot be combined" in result.stderr


def test_bootstrap_native_remote_script_local_uninstall_requires_existing_state(tmp_path: Path) -> None:
    local_root = tmp_path / "local-root"

    result = run_script(
        "--local",
        "--root-dir",
        str(local_root),
        "--uninstall",
    )

    assert result.returncode != 0
    assert "Bootstrap state file not found" in result.stderr
    assert not (local_root / ".noboom-native-bootstrap").exists()


def test_bootstrap_native_remote_script_local_can_skip_every_install_stage(tmp_path: Path) -> None:
    env = prepare_local_target_tools(tmp_path)
    local_root = tmp_path / "local-root"

    result = run_script(
        "--local",
        "--root-dir",
        str(local_root),
        "--skip-uv",
        "--skip-venv",
        "--skip-dependency-sync",
        "--skip-treeple",
        "--skip-torch-cluster",
        "--skip-postgres",
        "--skip-seaweed",
        "--skip-symlinks",
        env=env,
    )

    combined_output = result.stdout + result.stderr
    state_file = local_root / ".noboom-native-bootstrap" / "state.json"

    assert result.returncode == 0, combined_output
    assert "Skipping uv bootstrap by request." in combined_output
    assert "Skipping treeple build and validation by request." in combined_output
    assert "Wrote bootstrap state" in combined_output
    assert state_file.is_file()


def test_bootstrap_native_remote_script_local_seaweed_stage_handles_missing_binary(tmp_path: Path) -> None:
    env = prepare_local_target_tools(tmp_path)
    local_root = tmp_path / "local-root"

    result = run_script(
        "--local",
        "--head",
        "--root-dir",
        str(local_root),
        "--skip-uv",
        "--skip-venv",
        "--skip-dependency-sync",
        "--skip-treeple",
        "--skip-torch-cluster",
        "--skip-postgres",
        "--skip-symlinks",
        env=env,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Downloading SeaweedFS 4.05" in combined_output
    assert "FileNotFoundError" not in combined_output


def test_bootstrap_native_remote_script_local_head_mode_requires_postgres_password(
    tmp_path: Path,
) -> None:
    env = prepare_local_target_tools(tmp_path)
    env.pop("POSTGRES_PASSWORD", None)
    local_root = tmp_path / "local-root"

    result = run_script(
        "--local",
        "--head",
        "--root-dir",
        str(local_root),
        env=env,
    )

    assert result.returncode != 0
    assert "Missing required environment variable: POSTGRES_PASSWORD" in result.stderr


def test_bootstrap_native_remote_script_rotate_postgres_password_requires_head(tmp_path: Path) -> None:
    env = prepare_local_target_tools(tmp_path)
    local_root = tmp_path / "local-root"

    result = run_script(
        "--local",
        "--root-dir",
        str(local_root),
        "--rotate-postgres-password",
        env=env,
    )

    assert result.returncode != 0
    assert "requires --head" in result.stderr


def test_bootstrap_native_remote_script_rejects_conflicting_rotate_and_uninstall_modes(
    tmp_path: Path,
) -> None:
    env = prepare_local_target_tools(tmp_path)
    env["POSTGRES_PASSWORD"] = "test-postgres-password"
    local_root = tmp_path / "local-root"

    result = run_script(
        "--local",
        "--root-dir",
        str(local_root),
        "--head",
        "--rotate-postgres-password",
        "--uninstall",
        env=env,
    )

    assert result.returncode != 0
    assert "Conflicting mode flags" in result.stderr


def test_bootstrap_native_remote_script_rejects_skip_flags_for_rotate_mode(tmp_path: Path) -> None:
    env = prepare_local_target_tools(tmp_path)
    env["POSTGRES_PASSWORD"] = "test-postgres-password"
    local_root = tmp_path / "local-root"

    result = run_script(
        "--local",
        "--root-dir",
        str(local_root),
        "--head",
        "--rotate-postgres-password",
        "--skip-postgres",
        env=env,
    )

    assert result.returncode != 0
    assert "Skip-stage flags can only be used with install mode." in result.stderr


def prepare_fake_remote_tools(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    ssh_key_path = tmp_path / "id_test"
    ssh_key_path.write_text("dummy", encoding="utf-8")

    fake_ssh = bin_dir / "ssh"
    fake_ssh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
command_string="${@: -1}"
exec /bin/bash -c "$command_string"
""",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)

    fake_rsync = bin_dir / "rsync"
    fake_rsync.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

sources=()
while (($# > 0)); do
  case "$1" in
    -e)
      shift 2
      ;;
    -*)
      shift
      ;;
    *)
      sources+=("$1")
      shift
      ;;
  esac
done

last_index=$((${#sources[@]} - 1))
destination="${sources[$last_index]}"
unset "sources[$last_index]"
destination="${destination#*:}"
mkdir -p "$destination"
for source_path in "${sources[@]}"; do
  cp "$source_path" "$destination/"
done
""",
        encoding="utf-8",
    )
    fake_rsync.chmod(0o755)

    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        "#!/usr/bin/env bash\nexit 1\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)

    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\nexit 1\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    return env, ssh_key_path


def prepare_local_target_tools(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    for command_name in (
        "bash",
        "sh",
        "python3",
        "dirname",
        "mkdir",
        "cp",
        "cat",
        "touch",
        "tee",
        "date",
        "hostname",
        "git",
        "tar",
        "make",
        "cmake",
        "g++",
    ):
        command_path = shutil.which(command_name)
        if command_path is None:
            raise AssertionError(f"Required test command is unavailable: {command_name}")
        (bin_dir / command_name).symlink_to(command_path)

    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        "#!/usr/bin/env bash\nexit 1\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = str(bin_dir)
    env["HOME"] = str(tmp_path)
    return env


def write_bootstrap_state(
    root_dir: Path,
    *,
    is_head: bool = False,
    created_core_paths: Optional[list[Path]] = None,
    postgres_port: str = "",
) -> None:
    normalized_root = root_dir.resolve()
    metadata_dir = normalized_root / ".noboom-native-bootstrap"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    state_path = metadata_dir / "state.json"
    core_paths = created_core_paths or [metadata_dir, normalized_root / ".venv"]
    state_path.write_text(
        json.dumps(
            {
                "install_host": "local",
                "remote_hostname": socket.getfqdn() or socket.gethostname(),
                "root_dir": str(normalized_root),
                "install_timestamp": "2026-03-16T00:00:00Z",
                "pins": {
                    "treeple_ref": "ef417e47e12caaa70ecc027db53f18b70b895930",
                    "torch_cluster_ref": "6dabb048d29c2bc788fce885d0e13f888c8987a2",
                    "postgres_version": "18.1",
                    "seaweed_version": "4.05",
                },
                "is_head": is_head,
                "postgres_port": postgres_port,
                "created_core_paths": [str(path) for path in core_paths],
                "created_auxiliary_paths": [],
                "preexisting_auxiliary_paths": [],
                "created_symlinks": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_bootstrap_native_remote_script_uninstall_requires_existing_state(tmp_path: Path) -> None:
    env, ssh_key_path = prepare_fake_remote_tools(tmp_path)
    remote_root = tmp_path / "remote-root"

    result = run_script(
        "--host",
        "127.0.0.1",
        "--ssh-user",
        "tester",
        "--ssh-key",
        str(ssh_key_path),
        "--root-dir",
        str(remote_root),
        "--uninstall",
        env=env,
    )

    assert result.returncode != 0
    assert "Bootstrap state file not found" in result.stderr
    assert not (remote_root / ".noboom-native-bootstrap").exists()


def test_bootstrap_native_remote_script_cleans_metadata_after_conflict(tmp_path: Path) -> None:
    env, ssh_key_path = prepare_fake_remote_tools(tmp_path)
    remote_root = tmp_path / "remote-root"
    (remote_root / ".venv").mkdir()

    result = run_script(
        "--host",
        "127.0.0.1",
        "--ssh-user",
        "tester",
        "--ssh-key",
        str(ssh_key_path),
        "--root-dir",
        str(remote_root),
        env=env,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Refusing to bootstrap into a root with preexisting managed paths:" not in combined_output
    assert "Removing preexisting managed paths before provisioning." in combined_output
    assert not (remote_root / ".venv").exists()


def test_bootstrap_native_remote_script_local_cleans_metadata_after_conflict(tmp_path: Path) -> None:
    env = prepare_local_target_tools(tmp_path)
    local_root = tmp_path / "local-root"
    (local_root / ".venv").mkdir(parents=True)

    result = run_script(
        "--local",
        "--root-dir",
        str(local_root),
        env=env,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Refusing to bootstrap into a root with preexisting managed paths:" not in combined_output
    assert "Removing preexisting managed paths before provisioning." in combined_output
    assert not (local_root / ".venv").exists()


def test_bootstrap_native_remote_script_local_worker_mode_skips_head_only_conflicts(tmp_path: Path) -> None:
    env = prepare_local_target_tools(tmp_path)
    local_root = tmp_path / "local-root"
    (local_root / "postgres").mkdir(parents=True)

    result = run_script(
        "--local",
        "--root-dir",
        str(local_root),
        env=env,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "preexisting managed paths" not in combined_output
    assert "Required target command not found: curl" not in combined_output
    assert "[INFO] Installing uv in user space" in combined_output


def test_bootstrap_native_remote_script_local_head_mode_detects_head_only_conflicts(tmp_path: Path) -> None:
    env = prepare_local_target_tools(tmp_path)
    env["POSTGRES_PASSWORD"] = "test-postgres-password"
    local_root = tmp_path / "local-root"
    (local_root / "postgres").mkdir(parents=True)

    result = run_script(
        "--local",
        "--head",
        "--root-dir",
        str(local_root),
        env=env,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Refusing to bootstrap into a root with preexisting managed paths:" not in combined_output
    assert "Removing preexisting managed paths before provisioning." in combined_output
    assert not (local_root / "postgres").exists()


def test_bootstrap_native_remote_script_remote_head_stages_postgres_secret(tmp_path: Path) -> None:
    env, ssh_key_path = prepare_fake_remote_tools(tmp_path)
    env["POSTGRES_PASSWORD"] = "test-postgres-password"
    remote_root = tmp_path / "remote-root"
    fake_ssh_path = tmp_path / "bin" / "ssh"
    fake_ssh_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
command_string="${@: -1}"
if [[ "$command_string" == *"--internal-target-mode"* ]]; then
  exit 17
fi
exec /bin/bash -c "$command_string"
""",
        encoding="utf-8",
    )
    fake_ssh_path.chmod(0o755)

    result = run_script(
        "--host",
        "127.0.0.1",
        "--ssh-user",
        "tester",
        "--ssh-key",
        str(ssh_key_path),
        "--root-dir",
        str(remote_root),
        "--head",
        env=env,
    )

    secret_path = remote_root / ".noboom-native-bootstrap" / "staging" / "bootstrap-secrets.json"

    assert result.returncode != 0
    assert secret_path.is_file()
    assert json.loads(secret_path.read_text(encoding="utf-8")) == {
        "POSTGRES_PASSWORD": "test-postgres-password"
    }


def test_bootstrap_native_remote_script_remote_rotate_stages_postgres_secret(tmp_path: Path) -> None:
    env, ssh_key_path = prepare_fake_remote_tools(tmp_path)
    env["POSTGRES_PASSWORD"] = "rotated-postgres-password"
    remote_root = tmp_path / "remote-root"
    write_bootstrap_state(remote_root, is_head=True)
    fake_ssh_path = tmp_path / "bin" / "ssh"
    fake_ssh_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
command_string="${@: -1}"
if [[ "$command_string" == *"--internal-target-mode"* ]]; then
  exit 17
fi
exec /bin/bash -c "$command_string"
""",
        encoding="utf-8",
    )
    fake_ssh_path.chmod(0o755)

    result = run_script(
        "--host",
        "127.0.0.1",
        "--ssh-user",
        "tester",
        "--ssh-key",
        str(ssh_key_path),
        "--root-dir",
        str(remote_root),
        "--head",
        "--rotate-postgres-password",
        env=env,
    )

    secret_path = remote_root / ".noboom-native-bootstrap" / "staging" / "bootstrap-secrets.json"

    assert result.returncode != 0
    assert secret_path.is_file()
    assert json.loads(secret_path.read_text(encoding="utf-8")) == {
        "POSTGRES_PASSWORD": "rotated-postgres-password"
    }


def test_bootstrap_native_remote_script_local_reinstalls_existing_managed_root(tmp_path: Path) -> None:
    env = prepare_local_target_tools(tmp_path)
    local_root = tmp_path / "local-root"
    (local_root / ".venv").mkdir(parents=True)
    write_bootstrap_state(local_root)

    result = run_script(
        "--local",
        "--root-dir",
        str(local_root),
        env=env,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Refusing to bootstrap into a root with preexisting managed paths:" not in combined_output
    assert "Removing existing bootstrap-managed install before reprovisioning." in combined_output
    assert not (local_root / ".venv").exists()


def test_bootstrap_native_remote_script_local_rotates_postgres_password_in_place(
    tmp_path: Path,
) -> None:
    env = prepare_local_target_tools(tmp_path)
    env["POSTGRES_PASSWORD"] = "rotated-postgres-password"
    local_root = tmp_path / "local-root"
    write_bootstrap_state(
        local_root,
        is_head=True,
        created_core_paths=[local_root / ".noboom-native-bootstrap", local_root / "postgres"],
        postgres_port="5449",
    )

    postgres_prefix = local_root / "postgres" / "pgsql"
    postgres_bin = postgres_prefix / "bin"
    postgres_data = postgres_prefix / "data"
    postgres_bin.mkdir(parents=True, exist_ok=True)
    postgres_data.mkdir(parents=True, exist_ok=True)
    (postgres_data / "PG_VERSION").write_text("18\n", encoding="utf-8")

    sql_capture = local_root / "postgres-single.sql"
    pg_ctl_log = local_root / "pg_ctl.log"
    psql_log = local_root / "psql.log"
    psql_password_log = local_root / "psql-password.log"

    (postgres_bin / "pg_ctl").write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{pg_ctl_log}"
if [[ "${{@: -1}}" == "status" ]]; then
  exit 3
fi
exit 0
""",
        encoding="utf-8",
    )
    (postgres_bin / "postgres").write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cat > "{sql_capture}"
exit 0
""",
        encoding="utf-8",
    )
    (postgres_bin / "pg_isready").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
        encoding="utf-8",
    )
    (postgres_bin / "psql").write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" > "{psql_log}"
printf '%s\\n' "${{PGPASSWORD:-}}" > "{psql_password_log}"
exit 0
""",
        encoding="utf-8",
    )

    for script_path in (
        postgres_bin / "pg_ctl",
        postgres_bin / "postgres",
        postgres_bin / "pg_isready",
        postgres_bin / "psql",
    ):
        script_path.chmod(0o755)

    root_env = local_root / ".env"
    mount_dir = local_root / "mnt"
    mount_dir.mkdir(parents=True, exist_ok=True)
    mount_env = mount_dir / ".env"
    base_env = mount_dir / ".env.base"
    for env_path in (root_env, mount_env, base_env):
        env_path.write_text(
            "\n".join(
                (
                    "POSTGRES_PASSWORD=old-password",
                    "MLFLOW_STORAGE_URI=postgresql://noboom:old-password@127.0.0.1:5432/mlflow_db",
                    "OPTUNA_STORAGE_URI=postgresql://noboom:old-password@127.0.0.1:5432/optuna_db",
                )
            )
            + "\n",
            encoding="utf-8",
        )

    result = run_script(
        "--local",
        "--root-dir",
        str(local_root),
        "--head",
        "--rotate-postgres-password",
        env=env,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode == 0, combined_output
    assert "Rotating managed PostgreSQL password" in combined_output
    assert "Updated PostgreSQL password references" in combined_output
    assert "ALTER ROLE noboom WITH PASSWORD 'rotated-postgres-password'" in sql_capture.read_text(
        encoding="utf-8"
    )
    assert "rotated-postgres-password" in psql_password_log.read_text(encoding="utf-8")
    assert "-p 5449" in pg_ctl_log.read_text(encoding="utf-8")

    for env_path in (root_env, mount_env, base_env):
        env_text = env_path.read_text(encoding="utf-8")
        assert "POSTGRES_PASSWORD=rotated-postgres-password" in env_text
        assert "old-password" not in env_text
