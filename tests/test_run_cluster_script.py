from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_cluster.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_cluster", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_cluster_script_delegates_to_cluster_cli(monkeypatch) -> None:
    module = _load_module()
    captured: dict[str, object] = {}

    def fake_cluster_main(
        argv: Optional[List[str]] = None,
        prog_name: Optional[str] = None,
    ) -> int:
        captured["argv"] = argv
        captured["prog_name"] = prog_name
        return 17

    monkeypatch.setattr(module, "cluster_main", fake_cluster_main)

    result = module.main(["--deployment-mode", "native", "--pair", "ome:gdn"])

    assert result == 17
    assert captured["argv"] == ["--deployment-mode", "native", "--pair", "ome:gdn"]
    assert captured["prog_name"] == "run_cluster.py"
