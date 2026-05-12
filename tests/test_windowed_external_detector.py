from __future__ import annotations

import sys
import types

import torch


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
        def grouped_parameters(self):
            return (self.parameters(),)

    class AnomalyDetector(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, inputs):
            return self.compute_online_anomaly_score(inputs)

    class Loss(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    models_module.BaseModel = BaseModel
    common_module.AnomalyDetector = AnomalyDetector
    anomaly_detector_module.AnomalyDetector = AnomalyDetector
    loss_module.Loss = Loss
    timesead_module.models = models_module
    timesead_module.optim = optim_module
    models_module.common = common_module
    optim_module.loss = loss_module


_install_timesead_stubs()

from noboom_benchmark.noboom_lib.core.models._windowed_external import (  # noqa: E402
    SourceBackedAnomalyDetector,
    build_causal_windows,
    dataloader_to_frame,
    frame_to_tensor,
    tensor_to_frame,
    train_val_split,
)


def test_shared_helpers_split_and_window_frames() -> None:
    frame = torch.arange(18, dtype=torch.float32).view(6, 3).numpy()
    import pandas as pd

    df = pd.DataFrame(frame, columns=["a", "b", "c"])
    train, valid = train_val_split(df, 0.5, 1)
    assert len(train) == 3
    assert len(valid) == 3

    tensor = frame_to_tensor(df)
    assert tensor.shape == (6, 3)
    round_trip = tensor_to_frame(tensor, df.columns, df.index)
    assert round_trip.equals(df.astype("float32"))

    windows = build_causal_windows(frame, 4)
    assert windows.shape == (6, 4, 3)
    assert torch.allclose(torch.tensor(windows[0]), torch.tensor([[0.0, 1.0, 2.0]]).repeat(4, 1))


def test_source_backed_detector_converts_windows_and_aligns_scores() -> None:
    class DummySource:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.fit_shape = None

        def detect_fit(self, train_data):
            self.fit_shape = train_data.shape

        def detect_score(self, test_data):
            return test_data.sum(axis=1).to_numpy(), None

    class DummyDetector(SourceBackedAnomalyDetector):
        source_cls = DummySource

    inputs = torch.randn(4, 12, 3)
    detector = DummyDetector(example=True)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs), batch_size=2)
    detector.fit(loader)
    scores = detector((inputs,))

    assert dataloader_to_frame(loader).shape == (4, 3)
    assert detector.impl.kwargs == {"example": True}
    assert detector.impl.fit_shape == (4, 3)
    assert scores.shape == (4,)
