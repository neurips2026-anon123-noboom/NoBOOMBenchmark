from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Optional

from noboom_cluster.noboom_cli_lib.ray_utils.internal import utils


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "src"
    / "noboom_cluster"
    / "noboom_cli_lib"
    / "scripts"
    / "head_files"
    / "scripts"
    / "stop_by_pgrep_regex.py"
)

SPEC = importlib.util.spec_from_file_location("stop_by_pgrep_regex", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
stop_by_pgrep_regex = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stop_by_pgrep_regex)


def test_stop_mlflow_gracefully_targets_named_tmux_session(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    def fake_run_ray_exec(
        cluster_yaml: str,
        remote_cmd: str,
        *,
        tmux: bool = False,
        env: Optional[dict[str, str]] = None,
        check: bool = True,
    ) -> int:
        recorded["cluster_yaml"] = cluster_yaml
        recorded["remote_cmd"] = remote_cmd
        recorded["tmux"] = tmux
        recorded["env"] = env
        recorded["check"] = check
        return 0

    monkeypatch.setattr(utils, "run_ray_exec", fake_run_ray_exec)

    assert utils.stop_mlflow_gracefully("cluster.yaml", root_dir="/work/user/test_noboom")
    assert recorded["cluster_yaml"] == "cluster.yaml"
    assert recorded["tmux"] is False
    assert recorded["check"] is True
    assert "--tmux-session noboom-mlflow" in str(recorded["remote_cmd"])


def test_main_kills_requested_tmux_session_without_matching_processes(monkeypatch) -> None:
    killed_sessions: list[str] = []

    monkeypatch.setattr(stop_by_pgrep_regex, "_pgrep", lambda regex: [])
    monkeypatch.setattr(stop_by_pgrep_regex, "_collect_tmux_sessions", lambda pids: [])
    monkeypatch.setattr(
        stop_by_pgrep_regex,
        "_kill_tmux_sessions",
        lambda sessions: killed_sessions.extend(sessions),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stop_by_pgrep_regex.py",
            "--pgrep-regex",
            "mlflow",
            "--tmux-session",
            "noboom-mlflow",
        ],
    )

    assert stop_by_pgrep_regex.main() == 0
    assert killed_sessions == ["noboom-mlflow"]
