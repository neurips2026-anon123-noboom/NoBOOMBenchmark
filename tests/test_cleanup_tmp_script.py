from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "src" / "noboom_cluster" / "noboom_cli_lib" / "scripts" / "mount_files" / "scripts" / "cleanup_tmp.sh"
)


def test_cleanup_tmp_script_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_cleanup_tmp_script_ignores_permission_denied_entries(tmp_path: Path) -> None:
    ray_tmp_dir = tmp_path / "ray"
    inaccessible_session = ray_tmp_dir / "session_inaccessible"
    inaccessible_events_dir = inaccessible_session / "logs" / "events"
    inaccessible_events_dir.mkdir(parents=True)
    (inaccessible_events_dir / "event_GCS.log").write_text("blocked", encoding="utf-8")

    removable_session = ray_tmp_dir / "session_removable"
    removable_session.mkdir(parents=True)
    (removable_session / "temp.txt").write_text("cleanup", encoding="utf-8")

    os.chmod(inaccessible_session / "logs", 0o500)
    os.chmod(inaccessible_events_dir, 0o500)

    try:
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "BASE": str(ray_tmp_dir)},
            check=False,
        )
    finally:
        os.chmod(inaccessible_events_dir, 0o700)
        os.chmod(inaccessible_session / "logs", 0o700)

    assert result.returncode == 0, result.stderr
    assert "failed to fully remove" in result.stderr
    assert inaccessible_session.exists()
    assert not removable_session.exists()
