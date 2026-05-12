from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs"
MODEL_CONFIG_DIR = CONFIG_ROOT / "models"
PARAM_CONFIG_DIR = CONFIG_ROOT / "params"

REQUESTED_NEURAL_NAMES = [
    "itransformer",
    "modern_tcn",
    "dualtf",
    "patchtst",
    "dlinear",
    "nlinear",
    "tfad",
    "autoencoder",
]
REQUESTED_DETECTOR_NAMES = ["ocsvm", "pca", "hbos"]
REQUESTED_NAMES = REQUESTED_NEURAL_NAMES + REQUESTED_DETECTOR_NAMES


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _import_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _first_importable(candidates: list[tuple[str, str]]) -> Optional[type]:
    for module_name, attr_name in candidates:
        try:
            return _import_attr(module_name, attr_name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            continue
    return None


class TinyWindowDataset(Dataset):
    def __init__(self, x: torch.Tensor) -> None:
        self.x = x
        self.y = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        return (self.x[idx],), self.y[idx]


@pytest.mark.parametrize("model_name", REQUESTED_NAMES)
def test_requested_model_config_loads_when_shipped(model_name: str) -> None:
    config_path = MODEL_CONFIG_DIR / f"{model_name}.yaml"
    if not config_path.exists():
        pytest.skip(f"{model_name} model config is not present in this checkout.")

    config = _load_yaml(config_path)

    assert config
    assert config_path.stat().st_size > 0


@pytest.mark.parametrize("model_name", REQUESTED_NAMES)
def test_requested_param_config_loads_when_shipped(model_name: str) -> None:
    config_path = PARAM_CONFIG_DIR / f"{model_name}.yaml"
    if not config_path.exists():
        pytest.skip(f"{model_name} parameter config is not present in this checkout.")

    config = _load_yaml(config_path)

    assert "search_space" in config


def test_current_variant_names_are_declared_in_lightweight_param_configs() -> None:
    lnt_params = _load_yaml(PARAM_CONFIG_DIR / "lnt.yaml")
    neutralad_old_params = _load_yaml(PARAM_CONFIG_DIR / "neutralad_old.yaml")

    lnt_encoder_choices = lnt_params["search_space"]["model"]["network"]["init_args"]["encoder_type"]["choices"]
    neutralad_encoder_choices = neutralad_old_params["search_space"]["model"]["network"]["encoder_type"]["choices"]

    assert "modern_tcn" in lnt_encoder_choices
    assert "patchtst" in neutralad_encoder_choices
    assert "itransformer" in neutralad_encoder_choices


@pytest.mark.parametrize("model_name", REQUESTED_NAMES)
def test_requested_config_references_parse_where_declared(model_name: str) -> None:
    matched_paths = [
        path
        for path in sorted(CONFIG_ROOT.rglob("*.yaml"))
        if model_name in path.read_text(encoding="utf-8").lower()
    ]
    if not matched_paths:
        pytest.skip(f"{model_name} is not declared in shipped YAML configs yet.")

    for path in matched_paths:
        _load_yaml(path)


@pytest.mark.parametrize(
    "model_name,candidates,init_kwargs,input_shape,expected_shape",
    [
        (
            "itransformer",
            [("timesead_ext.models.encoders.itransformer", "ITransformerEncoder")],
            {"seq_len": 8, "d_model": 16, "num_heads": 4, "num_layers": 1, "d_ff": 32},
            (3, 8, 2),
            (3, 16),
        ),
        (
            "modern_tcn",
            [
                ("timesead_ext.models.encoders.modern_tcn", "ModernTCNEncoder"),
                ("timesead_ext.models.encoders.modern_tcn", "ModernTCN"),
            ],
            {"seq_len": 8, "d_model": 16},
            (3, 8, 2),
            None,
        ),
        (
            "dualtf",
            [
                ("timesead_ext.models.encoders.dualtf", "DualTFEncoder"),
                ("timesead_ext.models.other.dualtf", "DualTF"),
            ],
            {"seq_len": 8, "d_model": 16},
            (3, 8, 2),
            None,
        ),
        (
            "patchtst",
            [("timesead_ext.models.encoders.patchtst", "PatchTSTEncoder")],
            {
                "seq_len": 8,
                "patch_len": 4,
                "patch_stride": 2,
                "d_model": 16,
                "num_heads": 4,
                "num_layers": 1,
                "d_ff": 32,
            },
            (3, 8, 2),
            (3, 16),
        ),
        (
            "dlinear",
            [
                ("timesead_ext.models.encoders.dlinear", "DLinear"),
                ("timesead_ext.models.other.dlinear", "DLinear"),
            ],
            {"seq_len": 8, "pred_len": 8, "enc_in": 2},
            (3, 8, 2),
            None,
        ),
        (
            "nlinear",
            [
                ("timesead_ext.models.encoders.nlinear", "NLinear"),
                ("timesead_ext.models.other.nlinear", "NLinear"),
            ],
            {"seq_len": 8, "pred_len": 8, "enc_in": 2},
            (3, 8, 2),
            None,
        ),
        (
            "tfad",
            [
                ("timesead_ext.models.other.tfad", "TFAD"),
                ("noboom_benchmark.noboom_lib.core.models.tfad", "TFAD"),
            ],
            {"input_dim": 2, "window_size": 8},
            (3, 8, 2),
            None,
        ),
    ],
)
def test_neural_imports_run_tiny_forward_when_available(
    model_name: str,
    candidates: list[tuple[str, str]],
    init_kwargs: dict[str, Any],
    input_shape: tuple[int, int, int],
    expected_shape: Optional[tuple[int, ...]],
) -> None:
    model_cls = _first_importable(candidates)
    if model_cls is None:
        pytest.skip(f"{model_name} import path is not present in this checkout.")

    torch.manual_seed(0)
    model = model_cls(**init_kwargs)
    model.eval()
    x = torch.randn(*input_shape)

    with torch.no_grad():
        output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == input_shape[0]
    if expected_shape is not None:
        assert tuple(output.shape) == expected_shape


def test_autoencoder_tiny_forward_round_trips_window_shape() -> None:
    from noboom_benchmark.noboom_lib.core.models.igad import MLPAutoEncoder

    torch.manual_seed(0)
    model = MLPAutoEncoder(window_size=8, input_dim=2, hidden_dims=[8], latent_dim=4)
    x = torch.randn(3, 8, 2)
    output = model(x)

    assert output.shape == x.shape


def test_ocsvm_detector_scores_tiny_windows_without_benchmark_launch() -> None:
    detector_cls = pytest.importorskip("timesead.models.other.lstm_ae_ocsvm").LSTMAEOCSVMAnomalyDetector

    class TinyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return x.transpose(0, 1).contiguous()

    torch.manual_seed(1)
    x = torch.randn(8, 4, 2)
    loader = DataLoader(TinyWindowDataset(x), batch_size=4)
    detector = detector_cls(TinyEncoder(), gamma="scale", nu=0.2)

    detector.fit(loader)
    scores = detector.compute_online_anomaly_score((x,))

    assert scores.shape == (8,)
    assert torch.isfinite(scores).all()


def test_pca_detector_scores_tiny_windows_without_benchmark_launch() -> None:
    detector_cls = pytest.importorskip("timesead.models.baselines.pcas").PCAAnomalyDetector

    torch.manual_seed(2)
    x = torch.randn(6, 4, 2)
    loader = DataLoader(TinyWindowDataset(x), batch_size=3)
    detector = detector_cls(n_components=1)

    detector.fit(loader)
    scores = detector.compute_online_anomaly_score((x,))

    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()


def test_hbos_detector_scores_tiny_windows_without_benchmark_launch() -> None:
    detector_cls = pytest.importorskip("timesead.models.baselines.hbos").HBOSAD

    torch.manual_seed(3)
    x = torch.randn(6, 4, 2)
    loader = DataLoader(TinyWindowDataset(x), batch_size=3)
    detector = detector_cls(n_bins=3)

    detector.fit(loader)
    scores = detector.compute_online_anomaly_score((x,))

    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()
