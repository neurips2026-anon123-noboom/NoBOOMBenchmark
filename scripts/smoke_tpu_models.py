#!/usr/bin/env python3
"""Tiny opt-in TPU smoke harness for curated NoBoom model configs.

This script merges the default model/param configs with sidecar overlays from
``cluster_files/configs/tpu`` and runs direct PyTorch training steps. It does
not call Ray, MLflow, the cluster CLI, or benchmark launch code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
CONFIG_ROOT = SRC_ROOT / "noboom_cluster" / "cluster_files" / "configs"
TPU_OVERLAY_ROOT = CONFIG_ROOT / "tpu"
DEFAULT_TPU_SMOKE_MODELS = ("autoencoder", "dlinear", "nlinear")

for path in (str(SRC_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from smoke_compile_precision_models import (  # noqa: E402
    _call_loss,
    _compact_error,
    _filter_init_args,
    _finite_gradients,
    _install_timesead_stubs_if_missing,
    _make_batch,
    _resolve_class,
    _small_init_args,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
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


def _merge_yaml_files(paths: Sequence[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        if path.exists():
            merged = _deep_merge_dict(merged, _load_yaml(path))
    return merged


def _overlay_path(kind: str, name: str) -> Path:
    return TPU_OVERLAY_ROOT / kind / f"{name}.yaml"


def load_base_model_config(model_name: str) -> dict[str, Any]:
    return _merge_yaml_files(
        [
            CONFIG_ROOT / "models" / "common.yaml",
            CONFIG_ROOT / "models" / f"{model_name}.yaml",
        ]
    )


def load_tpu_model_config(model_name: str) -> dict[str, Any]:
    model_overlay = _overlay_path("models", model_name)
    if not model_overlay.exists():
        raise FileNotFoundError(f"No TPU model overlay for '{model_name}': {model_overlay}")
    return _merge_yaml_files(
        [
            CONFIG_ROOT / "models" / "common.yaml",
            CONFIG_ROOT / "models" / f"{model_name}.yaml",
            _overlay_path("models", "common"),
            model_overlay,
        ]
    )


def load_tpu_param_config(model_name: str) -> dict[str, Any]:
    param_overlay = _overlay_path("params", model_name)
    if not param_overlay.exists():
        raise FileNotFoundError(f"No TPU param overlay for '{model_name}': {param_overlay}")
    return _merge_yaml_files(
        [
            CONFIG_ROOT / "params" / "common.yaml",
            CONFIG_ROOT / "params" / f"{model_name}.yaml",
            _overlay_path("params", "common"),
            param_overlay,
        ]
    )


def _validate_tpu_config(model_name: str, config: Mapping[str, Any]) -> None:
    trainer = config.get("trainer")
    model = config.get("model")
    if not isinstance(trainer, Mapping) or not isinstance(model, Mapping):
        raise ValueError(f"{model_name} TPU config must include trainer and model mappings.")
    if trainer.get("accelerator") != "tpu":
        raise ValueError(f"{model_name} TPU overlay did not set trainer.accelerator=tpu.")
    if trainer.get("precision") != "bf16-mixed":
        raise ValueError(f"{model_name} TPU overlay did not set trainer.precision=bf16-mixed.")
    if model.get("compile_torch_model") is not False:
        raise ValueError(f"{model_name} TPU overlay must disable torch.compile.")


def _instantiate_model_from_config(model_name: str, config: Mapping[str, Any]) -> nn.Module:
    model_config = config["model"]
    if not isinstance(model_config, Mapping):
        raise TypeError("model config must be a mapping")
    network_config = model_config["network"]
    if not isinstance(network_config, Mapping):
        raise TypeError("model.network config must be a mapping")
    cls = _resolve_class(str(network_config["class_path"]))
    init_args = _small_init_args(model_name, dict(network_config.get("init_args", {})))
    return cls(**_filter_init_args(cls, init_args))


def _instantiate_loss_from_config(config: Mapping[str, Any]) -> nn.Module:
    model_config = config["model"]
    if not isinstance(model_config, Mapping):
        raise TypeError("model config must be a mapping")
    loss_config = model_config.get("losses")
    if not isinstance(loss_config, Mapping):
        return nn.MSELoss()
    cls = _resolve_class(str(loss_config["class_path"]))
    init_args = dict(loss_config.get("init_args") or {})
    return cls(**_filter_init_args(cls, init_args))


def _select_device(name: str) -> tuple[torch.device, Optional[Callable[[], None]], str]:
    if name == "cpu":
        return torch.device("cpu"), None, "cpu"
    if name == "auto":
        if not os.environ.get("PJRT_DEVICE") and not os.environ.get("TPU_NAME"):
            return torch.device("cpu"), None, "cpu:auto-no-tpu-env"
    try:
        import torch_xla.core.xla_model as xm
    except ModuleNotFoundError as error:
        if name == "auto":
            return torch.device("cpu"), None, "cpu:auto-no-torch-xla"
        raise RuntimeError("TPU device requested but torch_xla is not installed.") from error
    return xm.xla_device(), xm.mark_step, "xla"


def _precision_from_config(config: Mapping[str, Any]) -> str:
    trainer = config.get("trainer", {})
    if isinstance(trainer, Mapping):
        return str(trainer.get("precision", "32"))
    return "32"


def _autocast_device_type(device: torch.device) -> str:
    if device.type == "xla":
        return "xla"
    return device.type


def run_model(model_name: str, device: torch.device, mark_step: Optional[Callable[[], None]], steps: int) -> dict[str, Any]:
    config = load_tpu_model_config(model_name)
    params = load_tpu_param_config(model_name)
    _validate_tpu_config(model_name, config)

    torch.manual_seed(23)
    model = _instantiate_model_from_config(model_name, config).to(device=device)
    loss = _instantiate_loss_from_config(config).to(device=device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1.0e-3) if trainable else None
    precision = _precision_from_config(config)
    use_bf16 = precision == "bf16-mixed"
    losses = []

    for _step in range(steps):
        inputs, targets = _make_batch(model_name, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=_autocast_device_type(device),
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            output = model(inputs)
            value = _call_loss(loss, output, targets)
        if not torch.isfinite(value.detach()).all():
            raise FloatingPointError("non-finite loss")
        value.backward()
        if not _finite_gradients(trainable):
            raise FloatingPointError("non-finite gradient")
        if optimizer is not None:
            optimizer.step()
        if mark_step is not None:
            mark_step()
        losses.append(float(value.detach().cpu()))

    window_choices = params.get("search_space", {}).get("window_size", {}).get("choices", [])
    return {
        "model": model_name,
        "device": device.type,
        "precision": precision,
        "steps": steps,
        "last_loss": losses[-1] if losses else None,
        "compile_torch_model": config["model"]["compile_torch_model"],
        "tpu_window_choices": window_choices,
    }


def _model_names(requested: Optional[Sequence[str]]) -> list[str]:
    if requested:
        return list(requested)
    return list(DEFAULT_TPU_SMOKE_MODELS)


def _print_effective_config(model_name: str) -> None:
    payload = {
        "model": model_name,
        "model_config": load_tpu_model_config(model_name),
        "param_config": load_tpu_param_config(model_name),
    }
    print(yaml.safe_dump(payload, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "tpu"], default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    _install_timesead_stubs_if_missing()
    model_names = _model_names(args.models)
    for model_name in model_names:
        config = load_tpu_model_config(model_name)
        load_tpu_param_config(model_name)
        _validate_tpu_config(model_name, config)
        if args.print_effective_config:
            _print_effective_config(model_name)

    if args.dry_run:
        results = [{"model": model_name, "status": "validated"} for model_name in model_names]
    else:
        device, mark_step, device_label = _select_device(args.device)
        print(json.dumps({"selected_device": str(device), "device_label": device_label}), flush=True)
        results = []
        for model_name in model_names:
            print(f"==> {model_name}", flush=True)
            try:
                result = run_model(model_name, device=device, mark_step=mark_step, steps=args.steps)
            except Exception as error:
                result = {"model": model_name, "status": "failed", "error": _compact_error(error)}
            else:
                result["status"] = "passed"
            print(json.dumps(result, sort_keys=True), flush=True)
            results.append(result)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    failed = [result for result in results if result.get("status") == "failed"]
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
