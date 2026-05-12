import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
)
PACKAGE_NAME = "noboom_benchmark.noboom_lib.core.benchmark_utils"


def _install_benchmark_model_stubs() -> None:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules[PACKAGE_NAME] = package
    sys.modules.pop(f"{PACKAGE_NAME}.benchmark_model", None)

    style_transfer_module = types.ModuleType(f"{PACKAGE_NAME}.style_transfer")

    class _StyleTransfer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def setup(self) -> None:
            return None

        def state_dict(self, prefix: str = ""):
            return {}

    style_transfer_module.StyleTransfer = _StyleTransfer
    sys.modules[style_transfer_module.__name__] = style_transfer_module

    training_ingredient_module = types.ModuleType("timesead_experiments.utils.training_ingredient")
    training_ingredient_module.instantiate_loss = lambda loss: loss
    sys.modules[training_ingredient_module.__name__] = training_ingredient_module
    sys.modules.setdefault("timesead_experiments", types.ModuleType("timesead_experiments"))
    sys.modules.setdefault("timesead_experiments.utils", types.ModuleType("timesead_experiments.utils"))

    timesead_models_module = types.ModuleType("timesead.models")

    class _BaseModel(torch.nn.Module):
        def grouped_parameters(self):
            return (self.parameters(),)

    timesead_models_module.BaseModel = _BaseModel
    sys.modules["timesead.models"] = timesead_models_module

    timesead_common_module = types.ModuleType("timesead.models.common")

    class _AnomalyDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, inputs):
            return self.compute_online_anomaly_score(inputs)

    timesead_common_module.AnomalyDetector = _AnomalyDetector
    sys.modules["timesead.models.common"] = timesead_common_module

    timesead_loss_module = types.ModuleType("timesead.optim.loss")

    class _Loss(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    timesead_loss_module.Loss = _Loss
    sys.modules["timesead.optim.loss"] = timesead_loss_module
    sys.modules.setdefault("timesead.optim", types.ModuleType("timesead.optim"))

    timesead_utils_module = types.ModuleType("timesead.utils.utils")

    def _pack_tuple(value):
        return value if isinstance(value, tuple) else (value,)

    timesead_utils_module.pack_tuple = _pack_tuple
    sys.modules["timesead.utils.utils"] = timesead_utils_module
    sys.modules.setdefault("timesead.utils", types.ModuleType("timesead.utils"))


_install_benchmark_model_stubs()
benchmark_model_module = importlib.import_module(f"{PACKAGE_NAME}.benchmark_model")
BenchmarkModel = benchmark_model_module.BenchmarkModel
for module_name in [
    PACKAGE_NAME,
    f"{PACKAGE_NAME}.benchmark_model",
    f"{PACKAGE_NAME}.style_transfer",
    "timesead_experiments.utils.training_ingredient",
    "timesead.models",
    "timesead.models.common",
    "timesead.optim.loss",
    "timesead.utils.utils",
]:
    sys.modules.pop(module_name, None)


class _SeqLenCheckingNetwork(torch.nn.Module):
    def __init__(self, seq_len: int, num_features: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.linear = torch.nn.Linear(num_features, num_features)

    def forward(self, inputs):
        input_batch, = inputs
        assert input_batch.shape[1] == self.seq_len
        return self.linear(input_batch)

    def grouped_parameters(self):
        return [list(self.parameters())]


class _TopKCheckingNetwork(_SeqLenCheckingNetwork):
    def __init__(self, seq_len: int, num_features: int, topk: int) -> None:
        super().__init__(seq_len=seq_len, num_features=num_features)
        self.embedding = torch.nn.Embedding(num_features, 1)
        self.topk = topk


class _TupleLoss(torch.nn.Module):
    def forward(self, outputs, targets, epoch=None, num_epochs=None):
        output_batch, = outputs
        target_batch, = targets
        return (output_batch - target_batch).sum()


def test_setup_uses_network_seq_len_for_meta_shapes(monkeypatch) -> None:
    def fake_measure_flops(meta_network, forward_fn, loss_fn=None):
        out = forward_fn()
        if loss_fn is not None:
            loss_fn(out)
        return 0

    monkeypatch.setattr(benchmark_model_module, "instantiate_losses", lambda losses: losses)
    monkeypatch.setattr(benchmark_model_module, "measure_flops", fake_measure_flops)

    model = BenchmarkModel(
        detector=lambda network: object(),
        network=_SeqLenCheckingNetwork(seq_len=127, num_features=3),
        losses=[_TupleLoss()],
        batch_dim=0,
        window_size=128,
        num_features=3,
        metrics=[],
        model_name="energyad_fedformer",
    )
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(batch_size=2))

    model.setup("fit")

    assert model.flops_per_batch == 0


def test_hpad_setup_uses_network_seq_len_for_meta_shapes(monkeypatch) -> None:
    def fake_measure_flops(meta_network, forward_fn, loss_fn=None):
        out = forward_fn()
        if loss_fn is not None:
            loss_fn(out)
        return 0

    monkeypatch.setattr(benchmark_model_module, "instantiate_losses", lambda losses: losses)
    monkeypatch.setattr(benchmark_model_module, "measure_flops", fake_measure_flops)

    model = BenchmarkModel(
        detector=lambda network: object(),
        network=_SeqLenCheckingNetwork(seq_len=95, num_features=3),
        losses=[_TupleLoss()],
        batch_dim=0,
        window_size=128,
        num_features=3,
        metrics=[],
        model_name="hpad",
    )
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(batch_size=2))

    model.setup("fit")

    assert model.flops_per_batch == 0


def test_init_allows_missing_metrics() -> None:
    model = BenchmarkModel(
        detector=object(),
        metrics=None,
    )

    assert model.metrics == {}


def test_align_prediction_outputs_honors_label_index_offset() -> None:
    model = BenchmarkModel(
        detector=object(),
        window_size=2,
        prediction_horizon=1,
        label_index_offset=1,
        metrics=[],
    )
    scores = torch.tensor([0.4, 0.9], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)

    padded_scores, aligned_labels = model.align_prediction_outputs(scores, labels, [5])

    np.testing.assert_allclose(
        padded_scores,
        np.array([-np.inf, -np.inf, -np.inf, 0.4, 0.9], dtype=np.float32),
    )
    np.testing.assert_allclose(aligned_labels, labels.numpy())


def test_setup_clamps_gdn_topk_to_available_neighbors(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_model_module, "instantiate_losses", lambda losses: losses)
    monkeypatch.setattr(benchmark_model_module, "measure_flops", lambda *_args, **_kwargs: 0)

    model = BenchmarkModel(
        detector=lambda network: object(),
        network=_TopKCheckingNetwork(seq_len=127, num_features=13, topk=15),
        losses=[_TupleLoss()],
        batch_dim=0,
        window_size=128,
        num_features=13,
        metrics=[],
        model_name="gdn",
    )
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(batch_size=2))

    model.setup("fit")

    assert model.network.topk == 12


def test_setup_clamps_gdn_topk_from_network_embedding_when_num_features_missing(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_model_module, "instantiate_losses", lambda losses: losses)
    monkeypatch.setattr(benchmark_model_module, "measure_flops", lambda *_args, **_kwargs: 0)

    model = BenchmarkModel(
        detector=lambda network: object(),
        network=_TopKCheckingNetwork(seq_len=127, num_features=13, topk=15),
        losses=[_TupleLoss()],
        batch_dim=0,
        window_size=128,
        num_features=None,
        metrics=[],
        model_name="gdn",
    )
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(batch_size=2))

    model.setup("fit")

    assert model.network.topk == 12


def test_init_defaults_evaluation_postprocessing_to_disabled() -> None:
    model = BenchmarkModel(
        detector=object(),
        metrics=[],
    )

    assert model.evaluation_postprocessing_config.enabled is False


def test_on_predict_epoch_end_uses_event_postprocessing_when_enabled(monkeypatch) -> None:
    helper_calls = []

    def fake_postprocess(score_sequences, label_sequences, *, metric_names, config, fix_threshold=False, dataset_name=None):
        del label_sequences, metric_names, config, fix_threshold, dataset_name
        helper_calls.append([sequence.copy() for sequence in score_sequences])
        return SimpleNamespace(
            raw_scores=np.array([-np.inf, 0.1, 0.8], dtype=np.float64),
            metric_values={"alarm_score": 0.5},
            predictions=np.array([0, 0, 1], dtype=np.int32),
        )

    monkeypatch.setattr(
        benchmark_model_module,
        "evaluate_sequences_with_optional_postprocessing",
        fake_postprocess,
    )
    monkeypatch.setattr(
        benchmark_model_module,
        "select_threshold",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy threshold selection should not run")
        ),
    )

    model = BenchmarkModel(
        detector=object(),
        batch_dim=0,
        window_size=1,
        metrics=["alarm_score"],
        evaluation_postprocessing={"enabled": True},
    )
    model._trainer = SimpleNamespace(
        predict_loop=SimpleNamespace(
            predictions=[torch.tensor([0.1, 0.8], dtype=torch.float32)],
        ),
        datamodule=SimpleNamespace(
            dataset_name="cont_reactive_ome",
            seq_len=lambda split: [2],
            predict_orig_dataloader=lambda: [
                (
                    None,
                    (torch.tensor([[0.0, 1.0]], dtype=torch.float32),),
                )
            ],
        ),
    )

    model.on_predict_epoch_end()

    assert len(helper_calls) == 1
    np.testing.assert_allclose(helper_calls[0][0], np.array([0.1, 0.8], dtype=np.float32))
    np.testing.assert_allclose(model.anomaly_scores, np.array([-np.inf, 0.1, 0.8], dtype=np.float64))
    assert model.metrics["alarm_score"][0] == 0.5


def test_on_predict_epoch_end_allows_edf_to_exceed_reference_metric(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_model_module,
        "get_metric_by_name",
        lambda metric_name: (
            lambda predictions, targets: 0.20833333333333334
            if metric_name == "edf"
            else (_ for _ in ()).throw(AssertionError(f"unexpected metric: {metric_name}"))
        ),
    )
    monkeypatch.setattr(
        benchmark_model_module,
        "evaluate_sequences_with_optional_postprocessing",
        lambda *_args, **_kwargs: SimpleNamespace(
            metric_values={"edf": 0.25},
            predictions=np.array([0, 1, 1], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        benchmark_model_module,
        "select_threshold",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy threshold selection should not run")
        ),
    )

    model = BenchmarkModel(
        detector=object(),
        batch_dim=0,
        window_size=1,
        metrics=["edf"],
        evaluation_postprocessing={"enabled": True},
    )
    model._trainer = SimpleNamespace(
        predict_loop=SimpleNamespace(
            predictions=[torch.tensor([0.1, 0.8], dtype=torch.float32)],
        ),
        datamodule=SimpleNamespace(
            dataset_name="cont_reactive_ome",
            seq_len=lambda split: [2],
            predict_orig_dataloader=lambda: [
                (
                    None,
                    (torch.tensor([[0.0, 1.0]], dtype=torch.float32),),
                )
            ],
        ),
    )

    model.on_predict_epoch_end()

    assert model.metrics["edf"] == (0.25, 0.20833333333333334)
