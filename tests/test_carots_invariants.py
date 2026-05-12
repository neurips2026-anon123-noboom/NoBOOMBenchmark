"""Invariant tests for the patched CAROTS implementation.

These tests cover the corrections made against the official reference
(https://github.com/kimanki/CAROTS) and the paper (arXiv:2506.03964):

* ``CAROTSCore.forward`` now produces the 3B layout
  ``[anchors; positives; negatives]`` (paper §3.3) instead of the 4B layout
  used in the official ``models/carots/modeling_carots.py``.
* ``carots_loss`` is rewritten for that 3B layout while preserving the
  symmetric similarity filtering of the official ``models/carots/loss.py``
  (lines 14-21, 27).
* ``CAROTS._init_centroids`` slices anchors only — fixing the official
  ``models/carots/scorer_carots.py:30`` slice that mixed in positives.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.CAROTS.CAROTS import (  # noqa: E402
    CAROTS,
    CAROTSConfig,
)
from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.CAROTS.model import (  # noqa: E402
    CAROTSCore,
    carots_loss,
)
from noboom_benchmark.noboom_lib.core.models._windowed_external import (  # noqa: E402
    SourceBackedAnomalyDetector,
)
from noboom_benchmark.noboom_lib.core.models.carots import (  # noqa: E402
    CAROTS as NativeCAROTS,
    CAROTSAnomalyDetector,
    CAROTSLoss,
)


def _make_core(input_dim: int = 3, projector_output_dim: int = 16) -> CAROTSCore:
    core = CAROTSCore(
        input_dim=input_dim,
        input_step=9,
        pred_step=1,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        projector_hidden_dim=32,
        projector_output_dim=projector_output_dim,
        data_dim=1,
        cuts_hidden_dim=8,
        cuts_gru_layers=1,
        cuts_concat_h=True,
        cuts_shared_weights_decoder=False,
        noise_level=0.1,
        bias_candidates=(0.5, -0.5),
        cutoff_probability=0.1,
        disturb_all=True,
        transform_percent=0.5,
    )
    # PositiveAugmentor requires a fitted causal discoverer. For tests we
    # reuse the freshly-initialised CUTS+ network from the same core; its
    # graph is uniform sigmoid(0)=0.5 which is fine for shape-level checks.
    core.positive_augmentor.set_causal_discoverer(core.causal_discoverer)
    return core


def _toy_frame(n_rows: int = 32, n_channels: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_rows, n_channels)).astype(np.float32)
    # Add a smooth signal so the loss has structure to learn.
    t = np.linspace(0.0, 4 * np.pi, n_rows, dtype=np.float32)
    base[:, 0] += np.sin(t)
    if n_channels > 1:
        base[:, 1] += np.cos(t)
    return base


def test_carots_core_forward_layout_is_3b() -> None:
    torch.manual_seed(0)
    core = _make_core(input_dim=3, projector_output_dim=8).eval()
    batch = torch.randn(4, 10, 3)
    out_full = core(batch, positive_augment=True, negative_augment=True)
    assert out_full.shape == (12, 8), out_full.shape

    out_neg_only = core(batch, positive_augment=False, negative_augment=True)
    assert out_neg_only.shape == (8, 8), out_neg_only.shape


def test_init_centroids_shape_invariant_under_augmentor_flags(tmp_path) -> None:
    projection_dim = 16

    def _build(positive_augment: bool) -> CAROTS:
        torch.manual_seed(0)
        config = CAROTSConfig(
            projector_output_dim=projection_dim,
            projector_hidden_dim=32,
            hidden_dim=32,
            num_layers=1,
            input_step=9,
            pred_step=1,
            win_size=10,
            batch_size=8,
            cuts_hidden_dim=8,
            positive_augment=positive_augment,
        )
        wrapper = CAROTS.__new__(CAROTS)
        wrapper.config = config
        wrapper.device = torch.device("cpu")
        wrapper.model = _make_core(input_dim=3, projector_output_dim=projection_dim).eval()
        wrapper.cuts_best_state = None
        wrapper.best_state = None
        wrapper.centroid_wo_norm = None
        wrapper.centroid_norm = None
        wrapper.train_cl_stats = (0.0, 1.0)
        wrapper.train_cd_stats = (0.0, 1.0)
        return wrapper

    frame = _toy_frame(n_rows=32, n_channels=3)
    # 23 windows of 10x3 from 32 rows.
    windows = np.stack(
        [frame[start : start + 10] for start in range(frame.shape[0] - 10 + 1)],
        axis=0,
    ).astype(np.float32)

    for positive_augment in (True, False):
        wrapper = _build(positive_augment=positive_augment)
        wrapper._init_centroids(windows)
        assert wrapper.centroid_wo_norm is not None
        assert wrapper.centroid_norm is not None
        # Both centroid tensors must expose the projection dimension on their
        # trailing axis regardless of the augmentor flags. The leading axis is
        # a singleton so the centroid broadcasts cleanly during scoring.
        assert wrapper.centroid_wo_norm.shape[-1] == projection_dim
        assert wrapper.centroid_norm.shape[-1] == projection_dim
        assert wrapper.centroid_wo_norm.numel() == projection_dim
        assert wrapper.centroid_norm.numel() == projection_dim


def test_carots_loss_finite_and_decreases_on_toy_frame() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    projection_dim = 16
    core = _make_core(input_dim=3, projector_output_dim=projection_dim).train()

    # Freeze the causal discoverer so its un-trained graph does not destabilise
    # the contrastive head while we measure loss decrease over a few steps.
    for param in core.causal_discoverer.parameters():
        param.requires_grad = False

    frame = _toy_frame(n_rows=32, n_channels=3)
    windows = np.stack(
        [frame[start : start + 10] for start in range(frame.shape[0] - 10 + 1)],
        axis=0,
    ).astype(np.float32)
    batch = torch.from_numpy(windows[:8])

    optim = torch.optim.Adam(
        [p for n, p in core.named_parameters() if not n.startswith("causal_discoverer.")],
        lr=1e-3,
    )

    losses: list[float] = []
    initial_loss: Optional[float] = None
    for _ in range(5):
        optim.zero_grad()
        out = core(batch, positive_augment=True, negative_augment=True)
        assert out.shape[0] == batch.shape[0] * 3
        loss = carots_loss(
            out,
            sim_threshold=0.0,
            temperature=0.1,
            positive_augment=True,
        )
        assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if initial_loss is None:
            initial_loss = loss.item()

    assert all(np.isfinite(value) for value in losses), losses
    assert losses[-1] < losses[0], (
        f"expected loss to decrease over 5 toy steps; got {losses}"
    )


def test_native_carots_components_do_not_use_source_backed_adapter() -> None:
    network = NativeCAROTS(
        input_dim=3,
        win_size=10,
        input_step=9,
        pred_step=1,
        hidden_dim=16,
        projector_hidden_dim=16,
        projector_output_dim=8,
        cuts_hidden_dim=4,
    )
    loss = CAROTSLoss(sim_threshold=0.0)
    detector = CAROTSAnomalyDetector(
        model=network,
        loss=loss,
        win_size=10,
        input_step=9,
        pred_step=1,
        num_epochs=1,
        cuts_num_epochs=1,
        batch_size=4,
    )

    assert isinstance(network, torch.nn.Module)
    assert isinstance(loss, torch.nn.Module)
    assert not issubclass(type(detector), SourceBackedAnomalyDetector)
    assert detector.model is network
    assert detector.loss is loss


def test_carots_loss_class_matches_function_for_fixed_threshold() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(12, 8)
    loss = CAROTSLoss(
        sim_threshold=0.0,
        sim_threshold_schedule=False,
        temperature=0.2,
        positive_augment=True,
    )

    expected = carots_loss(
        embeddings,
        sim_threshold=0.0,
        temperature=0.2,
        positive_augment=True,
    )
    actual = loss((embeddings,), (torch.empty(0),), epoch=3, num_epochs=5)

    assert torch.allclose(actual, expected)
