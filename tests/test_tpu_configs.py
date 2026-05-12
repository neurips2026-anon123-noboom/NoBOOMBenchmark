from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs"
SCRIPT_PATH = ROOT / "scripts" / "smoke_tpu_models.py"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dict(current, value)
        else:
            merged[key] = value
    return merged


def _plain_default_model_config(model_name: str) -> dict[str, Any]:
    common = _load_yaml(CONFIG_ROOT / "models" / "common.yaml")
    model = _load_yaml(CONFIG_ROOT / "models" / f"{model_name}.yaml")
    return _deep_merge_dict(common, model)


def _load_smoke_module() -> Any:
    spec = importlib.util.spec_from_file_location("smoke_tpu_models_for_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_model_configs_do_not_opt_into_tpu() -> None:
    for config_path in sorted((CONFIG_ROOT / "models").glob("*.yaml")):
        content = config_path.read_text(encoding="utf-8").lower()
        assert "accelerator: tpu" not in content


def test_tpu_overlays_are_sidecar_and_opt_in() -> None:
    smoke = _load_smoke_module()

    assert tuple(smoke.DEFAULT_TPU_SMOKE_MODELS) == ("autoencoder", "dlinear", "nlinear")
    for model_name in smoke.DEFAULT_TPU_SMOKE_MODELS:
        default_config = _plain_default_model_config(model_name)
        script_default_config = smoke.load_base_model_config(model_name)
        tpu_config = smoke.load_tpu_model_config(model_name)
        tpu_params = smoke.load_tpu_param_config(model_name)

        assert script_default_config == default_config
        assert default_config["trainer"]["accelerator"] != "tpu"
        assert default_config["model"]["compile_torch_model"] is True
        assert tpu_config["trainer"]["accelerator"] == "tpu"
        assert tpu_config["trainer"]["precision"] == "bf16-mixed"
        assert tpu_config["model"]["compile_torch_model"] is False
        assert tpu_params["study"]["n_trials"] == 1
        assert tpu_params["search_space"]["window_size"]["choices"] == [16]


def test_tpu_smoke_harness_cpu_path_does_not_launch_benchmark(tmp_path: Path) -> None:
    json_out = tmp_path / "tpu_smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--device",
            "cpu",
            "--models",
            "autoencoder",
            "--steps",
            "1",
            "--json-out",
            str(json_out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode == 0, combined_output
    assert "run_tune" not in combined_output
    assert "noboom_cluster.cli" not in combined_output
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload == [
        {
            "compile_torch_model": False,
            "device": "cpu",
            "last_loss": payload[0]["last_loss"],
            "model": "autoencoder",
            "precision": "bf16-mixed",
            "status": "passed",
            "steps": 1,
            "tpu_window_choices": [16],
        }
    ]
