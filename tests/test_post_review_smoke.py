from __future__ import annotations

import contextlib
import inspect
import importlib
import importlib.util
import io
from pathlib import Path
import random
import sys
import types
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_ROOT = ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "models"
MODEL_UTILS_PATH = (
    ROOT
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
    / "model_utils.py"
)

MODEL_NAMES = [
    "anomaly_transformer",
    "dcdetector",
    "catch",
    "dada",
    "rtdetector",
    "carots",
    "paano",
    "oraclead",
]


def _install_timesead_stubs_if_missing() -> None:
    try:
        timesead_spec = importlib.util.find_spec("timesead")
    except ValueError:
        timesead_spec = None
    if timesead_spec is not None:
        return

    timesead_module = sys.modules.setdefault("timesead", types.ModuleType("timesead"))
    models_module = sys.modules.setdefault("timesead.models", types.ModuleType("timesead.models"))
    common_module = sys.modules.setdefault("timesead.models.common", types.ModuleType("timesead.models.common"))
    anomaly_detector_module = sys.modules.setdefault(
        "timesead.models.common.anomaly_detector",
        types.ModuleType("timesead.models.common.anomaly_detector"),
    )

    class BaseModel(torch.nn.Module):
        pass

    class AnomalyDetector(torch.nn.Module):
        pass

    models_module.BaseModel = BaseModel
    common_module.AnomalyDetector = AnomalyDetector
    anomaly_detector_module.AnomalyDetector = AnomalyDetector
    timesead_module.models = models_module
    models_module.common = common_module


def _load_get_args():
    spec = importlib.util.spec_from_file_location("post_review_model_utils", MODEL_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_args


get_args = _load_get_args()


def _set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    current = config
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _load_model_config(model_name: str) -> dict[str, Any]:
    with (MODEL_CONFIG_ROOT / f"{model_name}.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for source, target, apply_on in get_args(model_name):
        if source == "window_size" and apply_on == "parse":
            _set_dotted(config, target, 16)
    return config


def _resolve_class(class_path: str):
    _install_timesead_stubs_if_missing()
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    if (
        module_name.startswith("noboom_benchmark.noboom_lib.core.models.")
        and module_name != "noboom_benchmark.noboom_lib.core.models.dada"
    ):
        module = importlib.reload(module)
    return getattr(module, class_name)


def _smoke_init_args(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    init_args = dict(config["model"]["detector"].get("init_args", {}))

    if model_name in {"anomaly_transformer", "dcdetector"}:
        init_args["d_model"] = 16
        init_args["n_heads"] = 1
        init_args["e_layers"] = 1
    if model_name == "dcdetector":
        init_args["patch_size"] = [4]
    elif model_name == "catch":
        init_args.update(
            cf_dim=8,
            d_ff=16,
            d_model=16,
            e_layers=1,
            head_dim=8,
            inference_patch_size=4,
            n_heads=1,
            patch_size=4,
            patch_stride=2,
        )
    elif model_name == "dada":
        init_args["model_path"] = str(ROOT)
    elif model_name == "carots":
        init_args.update(
            cuts_hidden_dim=4,
            cuts_num_epochs=1,
            eval_period=1,
            hidden_dim=8,
            input_step=15,
            max_eval_windows=8,
            max_train_windows=8,
            num_epochs=1,
            pred_step=1,
        )
    elif model_name == "paano":
        init_args["memory_bank_ratio"] = None
    elif model_name == "oraclead":
        init_args.update(hidden_dim=8, num_heads=1, num_layers=1)

    return init_args


def _filter_init_args(cls: type, init_args: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return init_args
    return {key: value for key, value in init_args.items() if key in parameters}


def _causal_windows(points: torch.Tensor, window_size: int) -> torch.Tensor:
    windows = []
    for end in range(points.shape[0]):
        start = max(0, end - window_size + 1)
        window = points[start : end + 1]
        if window.shape[0] < window_size:
            pad = points[:1].repeat(window_size - window.shape[0], 1)
            window = torch.cat([pad, window], dim=0)
        windows.append(window)
    return torch.stack(windows)


def _network_init_args(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    init_args = dict(config["model"]["network"]["init_args"])
    for key in ("input_dim", "enc_in", "c_out", "feats"):
        if key in init_args:
            init_args[key] = 3
    for key in ("win_size", "seq_len", "window_size"):
        if key in init_args:
            init_args[key] = 16

    if model_name == "anomaly_transformer":
        init_args.update(d_ff=16, d_model=16, e_layers=1, n_heads=1)
    elif model_name == "dcdetector":
        init_args.update(d_model=16, e_layers=1, n_heads=1, patch_size=[4])
    elif model_name == "catch":
        init_args.update(
            cf_dim=8,
            d_ff=16,
            d_model=16,
            e_layers=1,
            head_dim=8,
            inference_patch_size=4,
            n_heads=1,
            patch_size=4,
            patch_stride=2,
        )
    elif model_name == "carots":
        init_args.update(
            cuts_hidden_dim=4,
            hidden_dim=8,
            input_step=15,
            pred_step=1,
            projector_hidden_dim=8,
            projector_output_dim=8,
        )
    elif model_name == "oraclead":
        init_args.update(hidden_dim=8, num_heads=1, num_layers=1)
    elif model_name == "paano":
        init_args.update(
            input_dim=3,
            kernel_sizes=[3],
            layers=[8],
            patch_size=16,
            projection_dim=8,
        )
    elif model_name == "rtdetector":
        init_args.update(dropout=0.0, feats=3, window_size=16)
    return init_args


def _instantiate_loss(config: dict[str, Any]):
    loss_config = config["model"].get("losses")
    if loss_config is None:
        return None
    loss_cls = _resolve_class(loss_config["class_path"])
    loss_init_args = dict(loss_config.get("init_args", {}))
    return loss_cls(**_filter_init_args(loss_cls, loss_init_args))


class _FakeDADACore(torch.nn.Module):
    def infer(
        self,
        x: torch.Tensor,
        norm: int = 0,
        mask_mode: str = "c",
        copies: int = 10,
    ) -> torch.Tensor:
        del norm, mask_mode
        return x.unsqueeze(0).repeat(copies, 1, 1, 1)

    def cal_anomaly_score(self, batch_x: torch.Tensor, batch_out_copies: torch.Tensor) -> torch.Tensor:
        return (batch_x - batch_out_copies.mean(dim=0)).pow(2).mean(dim=-1)


class _FakeDADA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeDADACore()
        self.config = types.SimpleNamespace(seq_len=16)


@pytest.fixture(autouse=True)
def _stub_dada_checkpoint_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module(
        "noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.DADA.DADA"
    )
    monkeypatch.setattr(module, "_load_model_from_path", lambda _model_path: _FakeDADA())
    native_module = importlib.import_module("noboom_benchmark.noboom_lib.core.models.dada")
    monkeypatch.setattr(native_module, "_load_model_from_path", lambda _model_path: _FakeDADA())


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_recent_model_registry_fit_and_score_smoke(model_name: str) -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    config = _load_model_config(model_name)
    detector_cls = _resolve_class(config["model"]["detector"]["class_path"])
    detector_init_args = _smoke_init_args(model_name, config)
    if "network" in config["model"]:
        network_cls = _resolve_class(config["model"]["network"]["class_path"])
        network = network_cls(**_network_init_args(model_name, config))
        detector_init_args["model"] = network
        loss = _instantiate_loss(config)
        if loss is not None:
            detector_init_args["loss"] = loss
    else:
        detector_init_args["seq_len"] = 16

    detector = detector_cls(**_filter_init_args(detector_cls, detector_init_args))

    frame_tensor = torch.linspace(0.0, 1.0, 64 * 3, dtype=torch.float32).reshape(64, 3)
    if model_name in {"anomaly_transformer", "dcdetector", "rtdetector"}:
        score_tensor = _causal_windows(frame_tensor, 16)
    else:
        score_tensor = frame_tensor
    loader = DataLoader(TensorDataset(score_tensor), batch_size=64, shuffle=False)

    with contextlib.redirect_stdout(io.StringIO()):
        detector.fit(loader)
        scores = detector.compute_online_anomaly_score((score_tensor,))

    assert scores.shape == (64,)
    assert torch.isfinite(scores).all()
