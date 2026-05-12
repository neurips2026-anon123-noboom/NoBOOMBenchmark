from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import types

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from noboom_benchmark.noboom_lib.core.tune_constants import uses_lightning_optimizer

MODEL_UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
    / "model_utils.py"
)


def _load_get_args():
    spec = importlib.util.spec_from_file_location("recent_model_ports_model_utils", MODEL_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_args


get_args = _load_get_args()


def _install_timesead_stubs() -> None:
    timesead_module = sys.modules.setdefault("timesead", types.ModuleType("timesead"))
    models_module = sys.modules.setdefault("timesead.models", types.ModuleType("timesead.models"))
    common_module = sys.modules.setdefault("timesead.models.common", types.ModuleType("timesead.models.common"))
    anomaly_detector_module = sys.modules.setdefault(
        "timesead.models.common.anomaly_detector",
        types.ModuleType("timesead.models.common.anomaly_detector"),
    )
    optim_module = sys.modules.setdefault("timesead.optim", types.ModuleType("timesead.optim"))
    loss_module = sys.modules.setdefault("timesead.optim.loss", types.ModuleType("timesead.optim.loss"))

    class BaseModel(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            super().__init__()

    class AnomalyDetector(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            super().__init__()

    class Loss(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            super().__init__()

    models_module.BaseModel = BaseModel
    common_module.AnomalyDetector = AnomalyDetector
    anomaly_detector_module.AnomalyDetector = AnomalyDetector
    loss_module.Loss = Loss
    timesead_module.models = models_module
    timesead_module.optim = optim_module
    models_module.common = common_module
    optim_module.loss = loss_module


def test_registry_wires_new_window_sizes() -> None:
    expected = {
        "anomaly_transformer": ("window_size", "model.network.init_args.win_size", "parse"),
        "dcdetector": ("window_size", "model.network.init_args.win_size", "parse"),
        "catch": ("window_size", "model.network.init_args.seq_len", "parse"),
        "dada": ("window_size", "model.detector.init_args.seq_len", "parse"),
        "rtdetector": ("window_size", "model.network.init_args.window_size", "parse"),
        "carots": ("window_size", "model.network.init_args.win_size", "parse"),
        "paano": ("window_size", "model.network.init_args.patch_size", "parse"),
        "oraclead": ("window_size", "model.network.init_args.win_size", "parse"),
    }

    for model_name, mapping in expected.items():
        assert mapping in get_args(model_name)


def test_anomaly_transformer_registry_links_native_network_dimensions() -> None:
    mappings = get_args("anomaly_transformer")

    assert ("data.num_features", "model.network.init_args.enc_in", "instantiate") in mappings
    assert ("data.num_features", "model.network.init_args.c_out", "instantiate") in mappings
    assert ("window_size", "model.network.init_args.win_size", "parse") in mappings


def test_dcdetector_registry_links_native_network_dimensions() -> None:
    mappings = get_args("dcdetector")

    assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings
    assert ("window_size", "model.network.init_args.win_size", "parse") in mappings


def test_rtdetector_registry_links_native_network_dimensions() -> None:
    mappings = get_args("rtdetector")

    assert ("data.num_features", "model.network.init_args.feats", "instantiate") in mappings
    assert ("window_size", "model.network.init_args.window_size", "parse") in mappings


def test_catch_registry_links_native_network_dimensions() -> None:
    mappings = get_args("catch")

    assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings
    assert ("window_size", "model.network.init_args.seq_len", "parse") in mappings
    assert ("window_size", "model.detector.init_args.seq_len", "parse") in mappings


def test_carots_registry_links_native_network_dimensions_and_loss() -> None:
    mappings = get_args("carots")

    assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings
    assert ("model.losses", "model.detector.init_args.loss", "parse") in mappings
    assert ("window_size", "model.network.init_args.win_size", "parse") in mappings
    assert ("window_size", "model.detector.init_args.win_size", "parse") in mappings


def test_paano_registry_links_native_network_dimensions() -> None:
    mappings = get_args("paano")

    assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings
    assert ("window_size", "model.network.init_args.patch_size", "parse") in mappings
    assert ("window_size", "model.detector.init_args.patch_size", "parse") in mappings


def test_oraclead_registry_links_native_network_dimensions_and_loss() -> None:
    mappings = get_args("oraclead")

    assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings
    assert ("window_size", "model.network.init_args.win_size", "parse") in mappings
    assert ("model.losses", "model.detector.init_args.loss", "parse") in mappings


def test_dada_registry_keeps_window_size_on_zero_shot_detector() -> None:
    mappings = get_args("dada")

    assert ("window_size", "model.detector.init_args.seq_len", "parse") in mappings
    targets = [target for _source, target, _apply_on in mappings]
    assert not any(target.startswith("model.network.") for target in targets)


def test_shipped_configs_use_local_class_paths_and_predict_on_end() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )

    model_files = {
        "carots.yaml": "noboom_benchmark.noboom_lib.core.models.carots.CAROTSAnomalyDetector",
    }

    for filename, expected_class_path in model_files.items():
        with (config_root / filename).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert config["model"]["predict_on_end"] is True
        assert config["trainer"]["precision"] == "bf16-mixed"
        assert config["model"]["detector"]["class_path"] == expected_class_path


def test_dada_config_uses_native_detector_owned_zero_shot_model() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "dada.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["predict_on_end"] is True
    assert config["trainer"]["precision"] == 32
    assert "network" not in config["model"]
    assert (
        config["model"]["detector"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.dada.DADAAnomalyDetector"
    )
    assert config["model"]["detector"]["init_args"]["seq_len"] is None
    assert config["model"]["detector"]["init_args"]["copies"] == 10
    assert config["model"]["detector"]["init_args"]["auto_download"] is True
    assert (
        config["model"]["detector"]["init_args"]["download_base_url"]
        == "https://raw.githubusercontent.com/iambowen/DADA/main/DADA"
    )


def test_anomaly_transformer_config_uses_native_network_loss_and_optimizer() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "anomaly_transformer.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["predict_on_end"] is True
    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["max_epochs"] == 25
    assert (
        config["model"]["network"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.anomaly_transformer.AnomalyTransformer"
    )
    assert (
        config["model"]["detector"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.anomaly_transformer.AnomalyTransformerAnomalyDetector"
    )
    assert config["model"]["detector"]["init_args"]["batch_size"] == 128
    assert (
        config["model"]["losses"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.anomaly_transformer.AnomalyTransformerLoss"
    )
    assert config["model"]["optimizer"]["class_path"] == "torch.optim.AdamW"


def test_dcdetector_config_uses_native_network_loss_and_optimizer() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "dcdetector.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["predict_on_end"] is True
    assert config["window_size"] == 100
    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["max_epochs"] == 10
    assert (
        config["model"]["network"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.dcdetector.DCdetector"
    )
    assert (
        config["model"]["detector"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.dcdetector.DCdetectorAnomalyDetector"
    )
    assert config["model"]["detector"]["init_args"]["batch_size"] == 128
    assert (
        config["model"]["losses"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.dcdetector.DCdetectorLoss"
    )
    assert config["model"]["optimizer"]["class_path"] == "torch.optim.AdamW"


def test_catch_config_uses_native_network_loss_and_optimizer() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "catch.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["predict_on_end"] is True
    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["max_epochs"] == 25
    assert config["model"]["network"]["class_path"] == "noboom_benchmark.noboom_lib.core.models.catch.CATCH"
    assert (
        config["model"]["detector"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.catch.CATCHAnomalyDetector"
    )
    assert config["model"]["losses"]["class_path"] == "noboom_benchmark.noboom_lib.core.models.catch.CATCHLoss"
    assert config["model"]["optimizer"]["class_path"] == "torch.optim.AdamW"


def _estimate_catch_flatten_head_params(
    *,
    window_size: int,
    patch_size: int,
    patch_stride: int,
    d_model: int,
) -> int:
    patch_num = int((window_size - patch_size) / patch_stride + 1)
    nf = 2 * d_model * patch_num
    return 2 * (3 * (nf * nf + nf) + (nf * window_size + window_size))


def test_catch_tuning_space_avoids_billion_parameter_heads() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
    )
    with (config_root / "params" / "catch.yaml").open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    with (config_root / "models" / "catch.yaml").open("r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)

    network_space = params["search_space"]["model"]["network"]["init_args"]
    max_head_params = max(
        _estimate_catch_flatten_head_params(
            window_size=window_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            d_model=d_model,
        )
        for window_size in params["search_space"]["window_size"]["choices"]
        for patch_size in network_space["patch_size"]["choices"]
        for patch_stride in network_space["patch_stride"]["choices"]
        for d_model in network_space["d_model"]["choices"]
    )

    assert max_head_params <= 250_000_000
    assert network_space["d_model"]["choices"] == [64]
    assert network_space["e_layers"]["choices"] == [2]
    assert model_config["data"]["batch_size"] <= 16
    assert model_config["model"]["detector"]["init_args"]["batch_size"] <= 64


def test_carots_config_uses_native_network_loss_and_detector_managed_training() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "carots.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["predict_on_end"] is True
    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["max_epochs"] == 1
    assert config["trainer"]["limit_train_batches"] == 0
    assert config["trainer"]["limit_val_batches"] == 0
    assert config["model"]["network"]["class_path"] == "noboom_benchmark.noboom_lib.core.models.carots.CAROTS"
    assert (
        config["model"]["detector"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.carots.CAROTSAnomalyDetector"
    )
    assert config["model"]["losses"]["class_path"] == "noboom_benchmark.noboom_lib.core.models.carots.CAROTSLoss"
    assert config["model"]["optimizer"]["class_path"] == "torch.optim.AdamW"


def test_rtdetector_config_uses_native_network_loss_and_optimizer() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "rtdetector.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["max_epochs"] == 25
    assert "predict_on_end" not in config["model"]
    assert (
        config["model"]["network"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.rtdetector.RTdetector"
    )
    assert (
        config["model"]["detector"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.rtdetector.RTdetectorAnomalyDetector"
    )
    assert (
        config["model"]["losses"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.rtdetector.RTdetectorLoss"
    )
    assert config["model"]["optimizer"]["class_path"] == "torch.optim.AdamW"


def test_paano_config_uses_native_network_loss_and_optimizer() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "paano.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["predict_on_end"] is True
    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["max_steps"] == 200
    assert config["model"]["network"]["class_path"] == "noboom_benchmark.noboom_lib.core.models.paano.PaAno"
    assert (
        config["model"]["detector"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.paano.PaAnoAnomalyDetector"
    )
    assert config["model"]["losses"]["class_path"] == "noboom_benchmark.noboom_lib.core.models.paano.PaAnoLoss"
    assert config["model"]["optimizer"]["class_path"] == "torch.optim.AdamW"


def test_oraclead_config_uses_native_network_loss_and_optimizer() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "oraclead.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["predict_on_end"] is True
    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["max_epochs"] == 25
    assert config["data"]["batch_size"] == 1024
    assert config["model"]["network"]["class_path"] == "noboom_benchmark.noboom_lib.core.models.oraclead.OracleAD"
    assert (
        config["model"]["detector"]["class_path"]
        == "noboom_benchmark.noboom_lib.core.models.oraclead.OracleADAnomalyDetector"
    )
    assert config["model"]["losses"]["class_path"] == "noboom_benchmark.noboom_lib.core.models.oraclead.OracleADLoss"
    assert config["model"]["optimizer"]["class_path"] == "torch.optim.AdamW"
    assert config["model"]["optimizer"]["init_args"]["lr"] == 5.0e-4
    assert config["model"]["optimizer"]["init_args"]["weight_decay"] == 1.0e-5


def test_oraclead_config_exposes_declared_native_init_args() -> None:
    _install_timesead_stubs()
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "oraclead.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for section in ("network", "detector", "losses"):
        section_config = config["model"][section]
        module_name, _, class_name = section_config["class_path"].rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        accepted_args = set(inspect.signature(cls.__init__).parameters)
        declared_args = set(section_config.get("init_args", {}))
        assert declared_args <= accepted_args


def test_catch_config_exposes_declared_native_init_args() -> None:
    _install_timesead_stubs()
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "catch.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for section in ("network", "detector", "losses"):
        section_config = config["model"][section]
        module_name, _, class_name = section_config["class_path"].rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        accepted_args = set(inspect.signature(cls.__init__).parameters)
        declared_args = set(section_config.get("init_args", {}))
        assert declared_args <= accepted_args


def test_carots_config_exposes_declared_native_init_args() -> None:
    _install_timesead_stubs()
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "carots.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for section in ("network", "detector", "losses"):
        section_config = config["model"][section]
        module_name, _, class_name = section_config["class_path"].rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        accepted_args = set(inspect.signature(cls.__init__).parameters)
        declared_args = set(section_config.get("init_args", {}))
        assert declared_args <= accepted_args


def test_dada_config_exposes_declared_detector_init_args() -> None:
    _install_timesead_stubs()
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )
    with (config_root / "dada.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    detector_config = config["model"]["detector"]
    module_name, _, class_name = detector_config["class_path"].rpartition(".")
    detector_cls = getattr(importlib.import_module(module_name), class_name)
    accepted_args = set(inspect.signature(detector_cls.__init__).parameters)
    declared_args = set(detector_config.get("init_args", {}))

    assert declared_args <= accepted_args


def test_carots_uses_detector_managed_optimizer_for_cuts_plus_stage() -> None:
    assert not uses_lightning_optimizer("carots")


def test_dada_zero_shot_model_does_not_receive_lightning_optimizer_configs() -> None:
    assert not uses_lightning_optimizer("dada")


def test_catch_uses_lightning_optimizer() -> None:
    assert uses_lightning_optimizer("catch")


def test_anomaly_transformer_uses_lightning_optimizer() -> None:
    assert uses_lightning_optimizer("anomaly_transformer")


def test_dcdetector_uses_lightning_optimizer() -> None:
    assert uses_lightning_optimizer("dcdetector")


def test_rtdetector_uses_lightning_optimizer() -> None:
    assert uses_lightning_optimizer("rtdetector")


def test_oraclead_uses_lightning_optimizer() -> None:
    assert uses_lightning_optimizer("oraclead")


def test_paano_uses_lightning_optimizer() -> None:
    assert uses_lightning_optimizer("paano")


def test_paano_memory_bank_score_increases_with_patch_novelty() -> None:
    from noboom_benchmark.noboom_lib.core.models.paano import PaAnoAnomalyDetector

    class MeanEmbeddingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def embedding(self, patches: torch.Tensor) -> torch.Tensor:
            means = patches.mean(dim=(1, 2)) * self.weight
            return torch.stack([torch.ones_like(means), means], dim=1)

    train_windows = torch.zeros(4, 2, 1)
    train_loader = DataLoader(TensorDataset(train_windows), batch_size=2, shuffle=False)
    detector = PaAnoAnomalyDetector(
        model=MeanEmbeddingModel(),
        patch_size=2,
        stride=1,
        batch_size=2,
        memory_bank_ratio=None,
        top_k=1,
    )

    detector.fit(train_loader)
    test_points = torch.tensor([0.0, 0.0, 4.0, 4.0], dtype=torch.float32)
    test_windows = test_points.view(-1, 1, 1).repeat(1, 2, 1)
    scores = detector.compute_online_anomaly_score((test_windows,))

    assert scores.shape == test_points.shape
    assert torch.isfinite(scores).all()
    assert scores[-1] > scores[0]


def test_dada_detector_scores_last_step_of_each_causal_window() -> None:
    from noboom_benchmark.noboom_lib.core.models.dada import DADAAnomalyDetector

    class LastStepMagnitudeDADA:
        seq_len = 3

        def __init__(self) -> None:
            self.seen_batches = []

        def infer(self, batch: torch.Tensor, norm: int = 0, copies: int = 10) -> torch.Tensor:
            self.seen_batches.append(batch.detach().clone())
            return batch.unsqueeze(0)

        def anomaly_score(self, batch: torch.Tensor, out_copies: torch.Tensor) -> torch.Tensor:
            assert out_copies.shape[1:] == batch.shape
            return batch.abs().sum(dim=2)

    fake_model = LastStepMagnitudeDADA()
    detector = DADAAnomalyDetector(model=fake_model, seq_len=3, batch_size=2, copies=4, norm=0)
    train_loader = DataLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2, shuffle=False)
    detector.fit(train_loader)

    points = torch.tensor([[0.0], [2.0], [-3.0]], dtype=torch.float32)
    scores = detector.compute_online_anomaly_score((points,))

    assert scores.tolist() == [0.0, 2.0, 3.0]
    assert torch.cat(fake_model.seen_batches, dim=0).shape == (3, 3, 1)


def test_dada_downloads_external_checkpoint_when_local_paths_are_missing(tmp_path, monkeypatch) -> None:
    from noboom_benchmark.noboom_lib.core.models import dada as dada_module

    downloaded = []

    def fake_download_file(url: str, destination: Path) -> None:
        downloaded.append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"checkpoint-file")

    monkeypatch.delenv("DADA_MODEL_PATH", raising=False)
    monkeypatch.delenv("DADA_REPO_PATH", raising=False)
    monkeypatch.setenv("DADA_AUTO_DOWNLOAD", "1")
    monkeypatch.setattr(dada_module, "ROOT_PATH", str(tmp_path / "source" / "constant.py"))
    monkeypatch.setattr(dada_module, "_download_file", fake_download_file)

    cache_dir = tmp_path / "cache"
    model = dada_module.DADA(
        model_path=str(tmp_path / "missing-model"),
        repo_path=str(tmp_path / "missing-repo"),
        cache_dir=str(cache_dir),
    )

    resolved = model._resolve_model_path()

    assert resolved == cache_dir / "DADA"
    assert [destination.name for _, destination in downloaded] == list(dada_module.DADA_REQUIRED_FILES)
    assert all((resolved / filename).is_file() for filename in dada_module.DADA_REQUIRED_FILES)


def test_catch_detector_score_combines_temporal_and_frequency_terms() -> None:
    from noboom_benchmark.noboom_lib.core.models.catch import CATCHAnomalyDetector

    class ZeroReconstructionModel(torch.nn.Module):
        seq_len = 3

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(
            self,
            inputs: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            window = inputs[0]
            reconstruction = torch.zeros_like(window) * self.weight
            return reconstruction, torch.empty(0, device=window.device), torch.zeros((), device=window.device), window

    class ConstantFrequencyCriterion(torch.nn.Module):
        def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.full_like(outputs, 2.0)

    windows = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 4.0], [0.0, 2.0]],
            [[2.0, 0.0], [1.0, 1.0], [3.0, 1.0]],
        ]
    )
    detector = CATCHAnomalyDetector(
        model=ZeroReconstructionModel(),
        seq_len=3,
        batch_size=2,
        score_lambda=0.5,
    )
    detector.frequency_criterion = ConstantFrequencyCriterion()

    scores = detector._score_windows(windows)
    expected_temporal = windows.pow(2).mean(dim=-1).reshape(-1)
    expected = expected_temporal + 1.0

    assert torch.allclose(scores, expected)
