"""Tests for the NoBoom-specific timesead Transform classes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Tuple

import torch


def _install_timesead_stubs() -> None:
    """Provide a minimal ``timesead.data.transforms.transform_base`` stub.

    The stub mirrors the parts of the real ``Transform`` base class that our
    NoBoom transforms depend on (``parent``, ``get_datapoint``, ``seq_len``)
    so the tests can be run in lightweight CI environments without pulling
    the heavy ML stack.
    """
    if "timesead.data.transforms.transform_base" in sys.modules:
        return

    timesead_module = sys.modules.setdefault("timesead", types.ModuleType("timesead"))
    data_module = sys.modules.setdefault("timesead.data", types.ModuleType("timesead.data"))
    transforms_pkg = sys.modules.setdefault(
        "timesead.data.transforms", types.ModuleType("timesead.data.transforms")
    )
    base_module = sys.modules.setdefault(
        "timesead.data.transforms.transform_base",
        types.ModuleType("timesead.data.transforms.transform_base"),
    )

    class Transform:
        def __init__(self, parent):
            self.parent = parent

        def get_datapoint(self, item: int):
            return self._get_datapoint_impl(item)

        def _get_datapoint_impl(self, item: int):
            raise NotImplementedError

        @property
        def seq_len(self):
            if self.parent is not None:
                return self.parent.seq_len
            return None

    base_module.Transform = Transform
    transforms_pkg.Transform = Transform
    data_module.transforms = transforms_pkg
    timesead_module.data = data_module


_install_timesead_stubs()


_NOBOOM_TRANSFORMS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
    / "noboom_transforms.py"
)


def _load_noboom_transforms_module() -> types.ModuleType:
    """Load `noboom_transforms.py` directly from its source file.

    The package-level ``benchmark_utils.__init__`` eagerly imports the full
    Lightning + Ray stack which is not available in CI's lightweight test
    environment. Loading by path keeps this test self-contained.
    """
    spec = importlib.util.spec_from_file_location(
        "noboom_transforms_under_test", _NOBOOM_TRANSFORMS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_module = _load_noboom_transforms_module()
FirstOrderDifferenceTransform = _module.FirstOrderDifferenceTransform
PerWindowStandardiseInputTransform = _module.PerWindowStandardiseInputTransform

from timesead.data.transforms.transform_base import Transform  # noqa: E402


class _StubSource(Transform):
    """In-memory parent that simply yields a recorded (inputs, targets) pair."""

    def __init__(
        self,
        inputs: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...],
        seq_len_value=None,
    ):
        super().__init__(parent=None)
        self._inputs = inputs
        self._targets = targets
        self._seq_len = seq_len_value

    def _get_datapoint_impl(self, item: int):
        return self._inputs, self._targets

    def __len__(self):
        return 1

    @property
    def seq_len(self):
        return self._seq_len


def test_first_order_diff_replaces_inputs_with_diff() -> None:
    x = torch.arange(10, dtype=torch.float32).unsqueeze(-1)  # (10, 1)
    source = _StubSource(inputs=(x,), targets=())
    transform = FirstOrderDifferenceTransform(parent=source)

    inputs, targets = transform.get_datapoint(0)

    assert len(inputs) == 1
    assert inputs[0].shape == (9, 1)
    assert torch.allclose(inputs[0], torch.ones(9, 1))
    assert targets == ()


def test_first_order_diff_truncates_per_timestep_labels() -> None:
    x = torch.randn(8, 2)
    labels = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0])  # length 8 — same as input time-axis
    scalar_label = torch.tensor(1)
    source = _StubSource(inputs=(x,), targets=(labels, scalar_label))
    transform = FirstOrderDifferenceTransform(parent=source)

    _, new_targets = transform.get_datapoint(0)

    assert new_targets[0].shape == (7,)
    assert torch.equal(new_targets[0], labels[1:])
    # 0-D tensors have no time axis to truncate.
    assert torch.equal(new_targets[1], scalar_label)


def test_first_order_diff_seq_len_drops_one() -> None:
    source = _StubSource(inputs=(torch.zeros(10, 1),), targets=(), seq_len_value=10)
    transform = FirstOrderDifferenceTransform(parent=source)
    assert transform.seq_len == 9

    list_source = _StubSource(inputs=(torch.zeros(5, 1),), targets=(), seq_len_value=[10, 7, 4])
    list_transform = FirstOrderDifferenceTransform(parent=list_source)
    assert list_transform.seq_len == [9, 6, 3]


def test_per_window_standardiser_zero_means_and_unit_std() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 3) * 4.5 + 1.7  # non-zero mean, non-unit std
    source = _StubSource(inputs=(x,), targets=())
    transform = PerWindowStandardiseInputTransform(parent=source)

    inputs, _ = transform.get_datapoint(0)

    standardised = inputs[0]
    assert torch.allclose(standardised.mean(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(standardised.std(dim=0), torch.ones(3), atol=1e-5)


def test_per_window_standardiser_applies_same_stats_to_dx_targets() -> None:
    torch.manual_seed(1)
    input_window = torch.randn(8, 2) * 3.0 + 2.0
    target_window = torch.randn(2, 2) * 3.0 + 2.0
    source = _StubSource(inputs=(input_window,), targets=(target_window,))
    transform = PerWindowStandardiseInputTransform(parent=source)

    inputs, targets = transform.get_datapoint(0)

    expected_mean = input_window.mean(dim=0, keepdim=True)
    expected_std = input_window.std(dim=0, keepdim=True).clamp_min(1e-6)
    expected_target = (target_window - expected_mean) / expected_std

    assert torch.allclose(targets[0], expected_target, atol=1e-6)


def test_per_window_standardiser_passes_labels_through_unchanged() -> None:
    input_window = torch.randn(8, 3)
    labels = torch.tensor([0, 1, 0, 0, 1, 1, 0, 0])  # ndim=1, no trailing-dim match
    source = _StubSource(inputs=(input_window,), targets=(labels,))
    transform = PerWindowStandardiseInputTransform(parent=source)

    _, targets = transform.get_datapoint(0)

    assert torch.equal(targets[0], labels)


def test_per_window_standardiser_handles_test_pipeline_label_plus_dx_target() -> None:
    """Mirrors the (label, dx_target) pair produced by PredictionTargetTransform
    when ``replace_labels=False``."""
    input_window = torch.randn(6, 4)
    label = torch.tensor([0, 1])  # shape (K,)
    dx_target = torch.randn(2, 4) * 5.0
    source = _StubSource(inputs=(input_window,), targets=(label, dx_target))
    transform = PerWindowStandardiseInputTransform(parent=source)

    _, targets = transform.get_datapoint(0)

    expected_mean = input_window.mean(dim=0, keepdim=True)
    expected_std = input_window.std(dim=0, keepdim=True).clamp_min(1e-6)
    expected_dx = (dx_target - expected_mean) / expected_std

    assert torch.equal(targets[0], label)  # labels untouched
    assert torch.allclose(targets[1], expected_dx, atol=1e-6)


def test_first_order_diff_then_standardiser_compose() -> None:
    """End-to-end CTE: raw window -> dx -> z-scored dx."""
    torch.manual_seed(42)
    raw = torch.cumsum(torch.randn(20, 2), dim=0) * 10.0  # autocorrelated, non-unit
    source = _StubSource(inputs=(raw,), targets=())
    diff_transform = FirstOrderDifferenceTransform(parent=source)
    chain = PerWindowStandardiseInputTransform(parent=diff_transform)

    inputs, _ = chain.get_datapoint(0)

    assert inputs[0].shape == (19, 2)
    assert torch.allclose(inputs[0].mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(inputs[0].std(dim=0), torch.ones(2), atol=1e-5)
