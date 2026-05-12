from argparse import Namespace
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import torch

import noboom_benchmark.noboom_lib.core.benchmark_utils.threshold as threshold_module


def test_objective_threshold_returns_raw_scores_for_best_feature(
    monkeypatch,
) -> None:
    class FakeDetector:
        def __init__(
            self,
            window_size: int,
            use_mean: bool,
            std_coefficient: float,
            *,
            two_sided: bool,
        ) -> None:
            del window_size, use_mean, std_coefficient, two_sided

        def score(self, series: torch.Tensor) -> torch.Tensor:
            del series
            return torch.tensor(
                [
                    [0.1, 0.9],
                    [0.8, 0.2],
                ],
                dtype=torch.float32,
            )

    fake_noboom = ModuleType("noboom")
    fake_noboom.__path__ = []
    fake_tsad = ModuleType("noboom.tsad")
    fake_tsad.__path__ = []
    fake_baselines = ModuleType("noboom.tsad.baselines")
    fake_baselines.MovingMeanDifferenceAnomalyDetector = FakeDetector
    fake_noboom.tsad = fake_tsad
    fake_tsad.baselines = fake_baselines

    monkeypatch.setitem(sys.modules, "noboom", fake_noboom)
    monkeypatch.setitem(sys.modules, "noboom.tsad", fake_tsad)
    monkeypatch.setitem(sys.modules, "noboom.tsad.baselines", fake_baselines)

    monkeypatch.setattr(
        threshold_module,
        "manifest_path_for_stage_from_args",
        lambda args, stage: "/tmp/fake-manifest.json",
    )
    monkeypatch.setattr(
        threshold_module.PreparedDatasetManifest,
        "load",
        staticmethod(lambda _path: SimpleNamespace()),
    )
    monkeypatch.setattr(
        threshold_module,
        "build_dataset_source",
        lambda manifest, mode: "eval-data",
    )
    monkeypatch.setattr(
        threshold_module,
        "PipelineDataset",
        lambda dataset: [
            (
                (torch.tensor([[1.0], [2.0]], dtype=torch.float32),),
                (torch.tensor([0.0, 1.0], dtype=torch.float32),),
            )
        ],
    )
    monkeypatch.setattr(
        threshold_module,
        "get_metric_by_name",
        lambda name: lambda predictions, labels: float(np.mean(predictions == labels)),
    )
    monkeypatch.setattr(threshold_module, "is_metric_binary", lambda _name: True)
    monkeypatch.setattr(
        threshold_module,
        "select_threshold",
        lambda scores, labels: SimpleNamespace(best_threshold=0.5),
    )

    class FakeParallel:
        def __init__(self, n_jobs: int, verbose: int) -> None:
            del n_jobs, verbose

        def __call__(self, tasks):
            return [task() for task in tasks]

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(threshold_module, "Parallel", FakeParallel)
    monkeypatch.setattr(threshold_module, "delayed", fake_delayed)

    metrics, anomaly_scores, anomaly_labels, threshold, feature = threshold_module.objective_threshold(
        args=Namespace(),
        hp_params={
            "window_size": 4,
            "use_mean": True,
            "std_coefficient": 1.0,
            "two_sided": False,
        },
        metrics=["alarm_score"],
        target_metric="alarm_score",
    )

    assert metrics["alarm_score"] == (1.0, 1.0)
    np.testing.assert_allclose(anomaly_scores, np.array([0.0, 0.1, 0.8], dtype=np.float32))
    np.testing.assert_array_equal(anomaly_labels, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert threshold == 0.5
    assert feature == 0


def test_objective_threshold_uses_event_postprocessing_when_enabled(
    monkeypatch,
) -> None:
    class FakeDetector:
        def __init__(
            self,
            window_size: int,
            use_mean: bool,
            std_coefficient: float,
            *,
            two_sided: bool,
        ) -> None:
            del window_size, use_mean, std_coefficient, two_sided

        def score(self, series: torch.Tensor) -> torch.Tensor:
            del series
            return torch.tensor(
                [
                    [0.1, 0.9],
                    [0.8, 0.2],
                ],
                dtype=torch.float32,
            )

    fake_noboom = ModuleType("noboom")
    fake_noboom.__path__ = []
    fake_tsad = ModuleType("noboom.tsad")
    fake_tsad.__path__ = []
    fake_baselines = ModuleType("noboom.tsad.baselines")
    fake_baselines.MovingMeanDifferenceAnomalyDetector = FakeDetector
    fake_noboom.tsad = fake_tsad
    fake_tsad.baselines = fake_baselines

    monkeypatch.setitem(sys.modules, "noboom", fake_noboom)
    monkeypatch.setitem(sys.modules, "noboom.tsad", fake_tsad)
    monkeypatch.setitem(sys.modules, "noboom.tsad.baselines", fake_baselines)

    monkeypatch.setattr(
        threshold_module,
        "manifest_path_for_stage_from_args",
        lambda args, stage: "/tmp/fake-manifest.json",
    )
    monkeypatch.setattr(
        threshold_module.PreparedDatasetManifest,
        "load",
        staticmethod(lambda _path: SimpleNamespace()),
    )
    monkeypatch.setattr(
        threshold_module,
        "build_dataset_source",
        lambda manifest, mode: "eval-data",
    )
    monkeypatch.setattr(
        threshold_module,
        "PipelineDataset",
        lambda dataset: [
            (
                (torch.tensor([[1.0], [2.0]], dtype=torch.float32),),
                (torch.tensor([0.0, 1.0], dtype=torch.float32),),
            )
        ],
    )
    monkeypatch.setattr(
        threshold_module,
        "get_metric_by_name",
        lambda name: lambda predictions, labels: float(np.mean(predictions == labels)),
    )
    monkeypatch.setattr(threshold_module, "is_metric_binary", lambda _name: True)
    monkeypatch.setattr(
        threshold_module,
        "select_threshold",
        lambda scores, labels: (_ for _ in ()).throw(AssertionError("legacy threshold path should not run")),
    )

    class FakeParallel:
        def __init__(self, n_jobs: int, verbose: int) -> None:
            del n_jobs, verbose

        def __call__(self, tasks):
            return [task() for task in tasks]

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    helper_calls = []

    def fake_postprocess(score_sequences, label_sequences, *, metric_names, config, fix_threshold=False, dataset_name=None):
        del label_sequences, metric_names, config, fix_threshold, dataset_name
        helper_calls.append([sequence.copy() for sequence in score_sequences])
        metric_value = float(score_sequences[0][0] < score_sequences[0][1])
        return SimpleNamespace(
            threshold=0.75,
            predictions=np.array([0, 0, 1], dtype=np.int32),
            metric_values={"alarm_score": metric_value},
        )

    monkeypatch.setattr(threshold_module, "Parallel", FakeParallel)
    monkeypatch.setattr(threshold_module, "delayed", fake_delayed)
    monkeypatch.setattr(
        threshold_module,
        "evaluate_sequences_with_optional_postprocessing",
        fake_postprocess,
    )

    metrics, anomaly_scores, anomaly_labels, threshold, feature = threshold_module.objective_threshold(
        args=Namespace(
            study_params={
                "evaluation_postprocessing": {
                    "enabled": True,
                }
            }
        ),
        hp_params={
            "window_size": 4,
            "use_mean": True,
            "std_coefficient": 1.0,
            "two_sided": False,
        },
        metrics=["alarm_score"],
        target_metric="alarm_score",
    )

    assert len(helper_calls) == 2
    np.testing.assert_allclose(helper_calls[0][0], np.array([0.1, 0.8], dtype=np.float32))
    np.testing.assert_allclose(helper_calls[1][0], np.array([0.9, 0.2], dtype=np.float32))
    assert metrics["alarm_score"] == (1.0, 1.0)
    np.testing.assert_allclose(anomaly_scores, np.array([0.0, 0.1, 0.8], dtype=np.float32))
    np.testing.assert_array_equal(anomaly_labels, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert threshold == 0.75
    assert feature == 0
