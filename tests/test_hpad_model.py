import sys
import types

import torch
import torch.nn as nn


def _install_timesead_stubs() -> None:
    timesead_module = sys.modules.setdefault("timesead", types.ModuleType("timesead"))

    models_module = sys.modules.setdefault("timesead.models", types.ModuleType("timesead.models"))
    common_module = sys.modules.setdefault("timesead.models.common", types.ModuleType("timesead.models.common"))
    anomaly_detector_module = sys.modules.setdefault(
        "timesead.models.common.anomaly_detector",
        types.ModuleType("timesead.models.common.anomaly_detector"),
    )
    layers_module = sys.modules.setdefault("timesead.models.layers", types.ModuleType("timesead.models.layers"))
    autoformer_module = sys.modules.setdefault(
        "timesead.models.layers.autoformer_encdec",
        types.ModuleType("timesead.models.layers.autoformer_encdec"),
    )
    autocorrelation_module = sys.modules.setdefault(
        "timesead.models.layers.autocorrelation",
        types.ModuleType("timesead.models.layers.autocorrelation"),
    )
    embed_module = sys.modules.setdefault(
        "timesead.models.layers.embed",
        types.ModuleType("timesead.models.layers.embed"),
    )
    fourier_module = sys.modules.setdefault(
        "timesead.models.layers.fourier_correlation",
        types.ModuleType("timesead.models.layers.fourier_correlation"),
    )
    optim_module = sys.modules.setdefault("timesead.optim", types.ModuleType("timesead.optim"))
    loss_module = sys.modules.setdefault("timesead.optim.loss", types.ModuleType("timesead.optim.loss"))

    class AnomalyDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, inputs):
            return self.compute_online_anomaly_score(inputs)

    class BaseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class Loss(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class DataEmbedding(nn.Module):
        def __init__(self, c_in: int, d_model: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.linear = nn.Linear(c_in, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(self.linear(x))

    class FourierBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, seq_len: int, **kwargs) -> None:
            super().__init__()
            self.seq_len = seq_len

        def forward(self, q, k, v, mask):
            return v, None

    class AutoCorrelationLayer(nn.Module):
        def __init__(self, correlation: nn.Module, d_model: int, n_heads: int, **kwargs) -> None:
            super().__init__()
            self.inner = correlation

        def forward(self, queries, keys, values, attn_mask):
            out, attn = self.inner(queries, keys, values, attn_mask)
            return out, attn

    class EncoderLayer(nn.Module):
        def __init__(self, attention: nn.Module, d_model: int, d_ff=None, **kwargs) -> None:
            super().__init__()
            self.attention = attention
            self.feed_forward = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor, attn_mask=None):
            attended, attn = self.attention(x, x, x, attn_mask)
            return self.feed_forward(attended), attn

    class Encoder(nn.Module):
        def __init__(self, attn_layers, conv_layers=None, norm_layer=None) -> None:
            super().__init__()
            self.attn_layers = nn.ModuleList(attn_layers)
            self.norm = norm_layer

        def forward(self, x: torch.Tensor, attn_mask=None):
            attns = []
            for layer in self.attn_layers:
                x, attn = layer(x, attn_mask=attn_mask)
                attns.append(attn)
            if self.norm is not None:
                x = self.norm(x)
            return x, attns

    class CustomLayerNorm(nn.LayerNorm):
        pass

    anomaly_detector_module.AnomalyDetector = AnomalyDetector
    loss_module.Loss = Loss
    embed_module.DataEmbedding = DataEmbedding
    fourier_module.FourierBlock = FourierBlock
    autocorrelation_module.AutoCorrelationLayer = AutoCorrelationLayer
    autoformer_module.CustomLayerNorm = CustomLayerNorm
    autoformer_module.Encoder = Encoder
    autoformer_module.EncoderLayer = EncoderLayer
    models_module.BaseModel = BaseModel

    timesead_module.models = models_module
    timesead_module.optim = optim_module
    models_module.common = common_module
    models_module.layers = layers_module
    common_module.anomaly_detector = anomaly_detector_module
    layers_module.autoformer_encdec = autoformer_module
    layers_module.autocorrelation = autocorrelation_module
    layers_module.embed = embed_module
    layers_module.fourier_correlation = fourier_module
    optim_module.loss = loss_module


_install_timesead_stubs()

from noboom_benchmark.noboom_lib.core.models.hpad import HPAD, HPADAnomalyDetector, HPADLoss  # noqa: E402


def test_hpad_forward_returns_expected_shapes() -> None:
    model = HPAD(
        window_size=16,
        input_dim=3,
        model_dim=32,
        num_heads=4,
        fcn_dim=64,
        encoder_layers=2,
        modes=8,
        patch_scales=[1, 2, 4],
        num_patch_prototypes=4,
        top_k_periods=2,
        num_period_prototypes=3,
    )
    inputs = torch.randn(2, 16, 3)

    outputs = model((inputs,))

    assert len(outputs) == 6
    reconstruction, patch_score_seq, period_score_seq, patch_entropy_term, patch_feature_term, period_feature_term = outputs
    assert reconstruction.shape == (2, 16, 3)
    assert patch_score_seq.shape == (2, 16)
    assert period_score_seq.shape == (2, 16)
    assert patch_entropy_term.ndim == 0
    assert patch_feature_term.ndim == 0
    assert period_feature_term.ndim == 0


def test_hpad_detector_returns_batch_scores_and_loss_is_scalar() -> None:
    model = HPAD(
        window_size=16,
        input_dim=3,
        model_dim=32,
        num_heads=4,
        fcn_dim=64,
        encoder_layers=2,
        modes=8,
        patch_scales=[1, 2, 4],
        num_patch_prototypes=4,
        top_k_periods=2,
        num_period_prototypes=3,
    )
    detector = HPADAnomalyDetector(model, beta=1.0)
    loss = HPADLoss()
    inputs = torch.randn(2, 16, 3)

    outputs = model((inputs,))
    scores = detector((inputs,))
    loss_value = loss(outputs, (inputs,))

    assert scores.shape == (2,)
    assert loss_value.ndim == 0


def test_hpad_loss_accepts_nested_prediction_tuple() -> None:
    model = HPAD(
        window_size=16,
        input_dim=3,
        model_dim=32,
        num_heads=4,
        fcn_dim=64,
        encoder_layers=2,
        modes=8,
        patch_scales=[1, 2, 4],
        num_patch_prototypes=4,
        top_k_periods=2,
        num_period_prototypes=3,
    )
    loss = HPADLoss()
    inputs = torch.randn(2, 16, 3)

    outputs = model((inputs,))
    loss_value = loss((outputs,), (inputs,))

    assert loss_value.ndim == 0
