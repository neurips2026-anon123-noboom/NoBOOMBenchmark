"""Classical detector-only anomaly models for NoBoom benchmark configs."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from timesead.models.common import AnomalyDetector
except ImportError:  # pragma: no cover - allows lightweight config tests without TimeSeAD.
    try:
        from timesead.models.common.anomaly_detector import AnomalyDetector  # type: ignore
    except ImportError:
        class AnomalyDetector(torch.nn.Module):  # type: ignore[no-redef]
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("dummy", torch.tensor([]), persistent=False)

            def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
                return self.compute_online_anomaly_score(inputs)


def _extract_inputs(batch: Any) -> Tuple[torch.Tensor, ...]:
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
        batch = batch[0]
    if isinstance(batch, torch.Tensor):
        return (batch,)
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
        return tuple(batch)
    raise ValueError("Expected a Tensor batch or a (inputs, targets) batch with Tensor inputs.")


class _FlattenedWindowDetector(AnomalyDetector):
    """Base adapter for sklearn-style detectors fitted on flattened windows."""

    def __init__(self, input_shape: str = "btf", random_state: Optional[int] = None) -> None:
        super().__init__()
        if input_shape[0] not in {"b", "t"}:
            raise ValueError("input_shape must start with 'b' or 't'.")
        self.input_shape = input_shape
        self.random_state = random_state
        self.window_size: Optional[int] = None

    def _batch_input_to_btf(self, value: torch.Tensor) -> torch.Tensor:
        data = value.detach().to(dtype=torch.float32)
        if data.ndim == 2:
            return data.unsqueeze(1)
        if data.ndim != 3:
            raise ValueError("Expected input tensor with shape [batch, features] or [batch, window, features].")
        if self.input_shape[0] == "t":
            return data.permute(1, 0, 2)
        return data

    def _flatten_inputs(self, inputs: Tuple[torch.Tensor, ...], *, fitting: bool = False) -> np.ndarray:
        data = self._batch_input_to_btf(inputs[0])
        if fitting:
            self.window_size = data.shape[1]
        elif self.window_size is None:
            raise RuntimeError('Run "fit" before trying to compute anomaly scores.')
        else:
            data = data[:, -self.window_size :, :]
        flat = data.reshape(data.shape[0], -1).cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(flat).all():
            raise RuntimeError("NaNs or infinities in input data.")
        return flat

    def _fit_array(self, data: np.ndarray) -> None:
        raise NotImplementedError

    def _score_array(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs: Any) -> None:
        del kwargs
        batches: List[np.ndarray] = []
        for batch in dataset:
            batches.append(self._flatten_inputs(_extract_inputs(batch), fitting=True))
        if not batches:
            raise ValueError("Cannot fit detector on an empty dataset.")
        self._fit_array(np.concatenate(batches, axis=0))

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        data = self._flatten_inputs(inputs)
        scores = self._score_array(data)
        return torch.as_tensor(scores, dtype=torch.float32, device=inputs[0].device).reshape(-1)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1] if self.input_shape[0] == "b" else target[-1]


class OCSVMAD(_FlattenedWindowDetector):
    """One-class SVM anomaly detector over flattened windows."""

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        nu: float = 0.1,
        normalize: bool = True,
        input_shape: str = "btf",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(input_shape=input_shape, random_state=random_state)
        try:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import OneClassSVM
        except ImportError as exc:  # pragma: no cover - dependency is in benchmark extras.
            raise ImportError("OCSVMAD requires scikit-learn.") from exc

        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.normalize = normalize
        estimator = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.model = make_pipeline(StandardScaler(), estimator) if normalize else estimator

    def _fit_array(self, data: np.ndarray) -> None:
        self.model.fit(data)

    def _score_array(self, data: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(data)


class PCAAD(_FlattenedWindowDetector):
    """PCA reconstruction-error detector over flattened windows."""

    def __init__(
        self,
        n_components: Union[int, float] = 0.95,
        svd_solver: str = "full",
        whiten: bool = False,
        input_shape: str = "btf",
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(input_shape=input_shape, random_state=random_state)
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:  # pragma: no cover - dependency is in benchmark extras.
            raise ImportError("PCAAD requires scikit-learn.") from exc

        self._pca_cls = PCA
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.model: Optional[PCA] = None

    def _resolved_n_components(self, data: np.ndarray) -> Union[int, float]:
        if isinstance(self.n_components, int):
            return max(1, min(self.n_components, min(data.shape)))
        return self.n_components

    def _fit_array(self, data: np.ndarray) -> None:
        self.model = self._pca_cls(
            n_components=self._resolved_n_components(data),
            svd_solver=self.svd_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        self.model.fit(data)

    def _score_array(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('Run "fit" before trying to compute anomaly scores.')
        projected = self.model.transform(data)
        reconstructed = self.model.inverse_transform(projected)
        return np.mean(np.square(data - reconstructed), axis=1)


class HBOSAD(_FlattenedWindowDetector):
    """Histogram-based outlier score detector over flattened windows."""

    def __init__(
        self,
        n_bins: Optional[int] = 10,
        alpha: float = 0.1,
        bin_tol: float = 0.5,
        input_shape: str = "btf",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(input_shape=input_shape, random_state=random_state)
        try:
            from pyod.models.hbos import HBOS
        except ImportError as exc:  # pragma: no cover - dependency is in benchmark extras.
            raise ImportError("HBOSAD requires pyod.") from exc

        self.n_bins = n_bins
        self.alpha = alpha
        self.bin_tol = bin_tol
        self.model = HBOS(n_bins=n_bins or "auto", alpha=alpha, tol=bin_tol)

    def _fit_array(self, data: np.ndarray) -> None:
        self.model.fit(data)

    def _score_array(self, data: np.ndarray) -> np.ndarray:
        return self.model.decision_function(data)
