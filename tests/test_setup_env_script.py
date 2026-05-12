from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SOURCE = (
    REPO_ROOT / "src/noboom_cluster/noboom_cli_lib/scripts/mount_files/scripts/setup_env.py"
)


def test_setup_env_writes_env_to_mount_and_root(tmp_path: Path) -> None:
    mount_dir = tmp_path / "mnt"
    scripts_dir = mount_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    script_path = scripts_dir / "setup_env.py"
    shutil.copyfile(SCRIPT_SOURCE, script_path)

    (mount_dir / ".env.base").write_text(
        "HEAD_LOCAL_IP=10.0.0.1\nNOBOOM_STORAGE=/tmp/storage\n",
        encoding="utf-8",
    )
    (mount_dir / "machine_nodes.yaml").write_text(
        "nodes:\n"
        "  - ip: 10.0.0.1\n"
        "    devices: 2,3\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ip = fake_bin / "ip"
    fake_ip.write_text(
        "#!/usr/bin/env bash\n"
        "echo '1.1.1.1 via 1.1.1.1 dev eth0 src 10.0.0.1 uid 1000'\n",
        encoding="utf-8",
    )
    fake_ip.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    mount_env = mount_dir / ".env"
    root_env = tmp_path / ".env"
    assert mount_env.is_file()
    assert root_env.is_file()
    assert mount_env.read_text(encoding="utf-8") == root_env.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2,3" in root_env.read_text(encoding="utf-8")
