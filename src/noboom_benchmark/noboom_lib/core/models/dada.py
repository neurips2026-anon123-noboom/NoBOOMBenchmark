"""Native DADA detector integration for the benchmark."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.request import urlopen

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

try:
    from timesead.models.common import AnomalyDetector
except ImportError:  # pragma: no cover - test stubs install this when needed.
    from timesead.models.common.anomaly_detector import AnomalyDetector  # type: ignore

try:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
except ImportError:  # pragma: no cover - only needed when loading an official checkpoint.
    get_class_from_dynamic_module = None

from ._source.baselines.self_impl.recent_utils import (
    build_causal_windows,
    first_existing_path,
    make_window_loader,
    transform_frame,
)
from ._source.common.constant import ROOT_PATH
from ._windowed_external import dataloader_to_frame, tensor_points_to_frame


DEFAULT_HYPER_PARAMS = {
    # The official zero-shot scripts mostly use batch size 32 outside MSL.
    "auto_download": True,
    "batch_size": 32,
    "cache_dir": None,
    "copies": 10,
    "download_base_url": "https://raw.githubusercontent.com/iambowen/DADA/main/DADA",
    "model_path": None,
    "norm": 0,
    "repo_path": None,
    "seq_len": 100,
    "trust_remote_code": True,
}
DADA_REQUIRED_FILES = (
    "config.json",
    "configuration_DADA.py",
    "modeling_DADA.py",
    "pytorch_model.bin",
)


class DADAConfig:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_model_from_path(model_path: Path) -> torch.nn.Module:
    """Load the official DADA model via HuggingFace's dynamic module loader."""

    if get_class_from_dynamic_module is None:
        raise ImportError("Loading a DADA checkpoint requires the optional `transformers` dependency.")
    missing = _missing_checkpoint_files(model_path)
    if missing:
        raise FileNotFoundError(
            f"DADA checkpoint directory `{model_path}` is missing required files: {', '.join(missing)}"
        )

    config_class = get_class_from_dynamic_module(
        "configuration_DADA.DADAConfig",
        str(model_path),
        local_files_only=True,
    )
    model_class = get_class_from_dynamic_module(
        "modeling_DADA.DADA",
        str(model_path),
        local_files_only=True,
    )

    config_data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config_keys = [
        "seq_len",
        "hidden_dim",
        "d_model",
        "bn_dims",
        "k",
        "patch_len",
        "mask_mode",
        "depth",
        "max_iters",
    ]
    config = config_class(**{key: config_data[key] for key in config_keys if key in config_data})
    model = model_class(config)
    state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


def _missing_checkpoint_files(model_path: Path) -> list[str]:
    return [name for name in DADA_REQUIRED_FILES if not (model_path / name).is_file()]


def _is_complete_checkpoint_dir(model_path: Path) -> bool:
    return model_path.is_dir() and not _missing_checkpoint_files(model_path)


def _default_cache_dir() -> Path:
    explicit = os.environ.get("DADA_CACHE_DIR") or os.environ.get("NOBOOM_DADA_CACHE_DIR")
    if explicit:
        return Path(explicit)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "noboom" / "DADA"
    return Path.home() / ".cache" / "noboom" / "DADA"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with urlopen(url, timeout=120) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temp_path.replace(destination)


def _patched_dada_infer(
    self: Any,
    x: torch.Tensor,
    norm: int = 0,
    mask_mode: Optional[str] = None,
    copies: int = 10,
) -> torch.Tensor:
    """Patch the official DADA infer path so it works on both CPU and GPU."""

    if mask_mode is None:
        mask_mode = self.mask_mode
    batch_size, seq_len, dims = x.size()

    if norm:
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
    else:
        means = None
        stdev = None

    x = x.permute(0, 2, 1).reshape(batch_size * dims, seq_len)
    if seq_len % self.patch_len != 0:
        full_length = self.patch_num * self.patch_len
        padding = torch.zeros((x.shape[0], full_length - seq_len), device=x.device, dtype=x.dtype)
        patch_input = torch.cat([x, padding], dim=1)
    else:
        patch_input = x

    patch_input = patch_input.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
    patch_input = self.input_embed(patch_input)

    if mask_mode == "c":
        if copies % 2 != 0:
            raise ValueError("DADA symmetric masking requires an even number of copies.")
        mask_1 = torch.from_numpy(
            np.random.binomial(1, 0.5, size=(batch_size * dims * (copies // 2), self.patch_num))
        ).to(device=patch_input.device, dtype=torch.bool)
        mask = torch.cat([mask_1, ~mask_1], dim=0)
        patch_input = patch_input.repeat(copies, 1, 1)
        patch_input[mask] = 0
    elif mask_mode == "random":
        mask = torch.from_numpy(
            np.random.binomial(1, 0.5, size=(batch_size * dims * copies, self.patch_num))
        ).to(device=patch_input.device, dtype=torch.bool)
        patch_input = patch_input.repeat(copies, 1, 1)
        patch_input[mask] = 0
    elif mask_mode == "nomask":
        copies = 1
    else:
        raise ValueError(f"Unknown DADA mask_mode: {mask_mode}")

    repr_tensor = self.encoder(patch_input)
    repr_tensor = torch.reshape(repr_tensor, (-1, self.patch_num * self.repr_dim))
    repr_tensor, _ = self.adaptive_bottleneck(repr_tensor, repr_tensor)
    repr_tensor = torch.reshape(repr_tensor, (-1, self.patch_num, self.repr_dim))

    out = self.decoder(repr_tensor)
    out = out.reshape(copies, batch_size * dims, seq_len)
    out = out[:, :, :seq_len].reshape(copies, batch_size, dims, seq_len)
    out = out.permute(0, 1, 3, 2)

    if norm:
        out = out * stdev.unsqueeze(0).repeat(copies, 1, 1, 1) + means.unsqueeze(0).repeat(copies, 1, 1, 1)
    return out


class DADA:
    """Official DADA checkpoint wrapper used by the native detector.

    DADA is a zero-shot checkpoint method in this benchmark. There is no clean
    Lightning training objective to expose without changing the algorithm, so
    the detector owns this wrapper and uses it only for inference/scoring.
    """

    def __init__(
        self,
        seq_len: Optional[int] = None,
        model_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        trust_remote_code: bool = True,
        auto_download: bool = True,
        cache_dir: Optional[str] = None,
        download_base_url: Optional[str] = None,
    ) -> None:
        self.config = DADAConfig(
            seq_len=DEFAULT_HYPER_PARAMS["seq_len"] if seq_len is None else int(seq_len),
            model_path=model_path,
            repo_path=repo_path,
            trust_remote_code=trust_remote_code,
            auto_download=auto_download,
            cache_dir=cache_dir,
            download_base_url=download_base_url or DEFAULT_HYPER_PARAMS["download_base_url"],
        )
        self.model: Optional[torch.nn.Module] = None

    @property
    def seq_len(self) -> int:
        return int(self.config.seq_len)

    def _resolve_model_path(self) -> Path:
        candidates = []
        if self.config.model_path:
            candidates.append(Path(self.config.model_path))
        if self.config.repo_path:
            repo_path = Path(self.config.repo_path)
            candidates.extend([repo_path / "DADA", repo_path])

        env_model_path = os.environ.get("DADA_MODEL_PATH")
        env_repo_path = os.environ.get("DADA_REPO_PATH")
        if env_model_path:
            candidates.append(Path(env_model_path))
        if env_repo_path:
            repo_path = Path(env_repo_path)
            candidates.extend([repo_path / "DADA", repo_path])

        candidates.extend(
            [
                Path(ROOT_PATH).parent / "DADA" / "DADA",
                Path(ROOT_PATH).parent / "DADA",
            ]
        )
        for candidate in candidates:
            if _is_complete_checkpoint_dir(candidate):
                return candidate

        auto_download = bool(self.config.auto_download) and _env_flag("DADA_AUTO_DOWNLOAD", True)
        if auto_download:
            return self._download_checkpoint()

        resolved = first_existing_path(candidates)
        if resolved is not None:
            missing = _missing_checkpoint_files(resolved)
            missing_message = f" Existing directory `{resolved}` is missing: {', '.join(missing)}."
        else:
            missing_message = ""
        raise FileNotFoundError(
            "Could not resolve a complete DADA model directory. Set `model_path`, "
            "`repo_path`, `DADA_MODEL_PATH`, enable `auto_download`, or clone the "
            f"official DADA repo next to the CATCH workspace.{missing_message}"
        )

    def _download_checkpoint(self) -> Path:
        cache_root = Path(self.config.cache_dir) if self.config.cache_dir else _default_cache_dir()
        checkpoint_dir = cache_root / "DADA"
        if _is_complete_checkpoint_dir(checkpoint_dir):
            return checkpoint_dir

        base_url = str(self.config.download_base_url).rstrip("/")
        missing = _missing_checkpoint_files(checkpoint_dir)
        for filename in missing:
            _download_file(f"{base_url}/{filename}", checkpoint_dir / filename)

        still_missing = _missing_checkpoint_files(checkpoint_dir)
        if still_missing:
            raise FileNotFoundError(
                "Downloaded DADA checkpoint is incomplete; missing files: "
                f"{', '.join(still_missing)} in `{checkpoint_dir}`."
            )
        return checkpoint_dir

    def load(self, device: torch.device) -> None:
        if self.model is None:
            if not self.config.trust_remote_code:
                raise ValueError("DADA checkpoint loading requires `trust_remote_code=True`.")
            model_path = self._resolve_model_path()
            self.model = _load_model_from_path(model_path)
            core_model = getattr(self.model, "model", None)
            if core_model is not None and hasattr(core_model, "input_embed") and hasattr(core_model, "encoder"):
                core_model.infer = types.MethodType(_patched_dada_infer, core_model)

            loaded_seq_len = getattr(getattr(self.model, "config", None), "seq_len", self.config.seq_len)
            if isinstance(loaded_seq_len, (list, tuple)):
                loaded_seq_len = loaded_seq_len[0]
            self.config.seq_len = int(loaded_seq_len)

        self.model.to(device)
        self.model.eval()

    def infer(self, batch: torch.Tensor, norm: int = 0, copies: int = 10) -> torch.Tensor:
        self.load(batch.device)
        if self.model is None:
            raise RuntimeError("DADA checkpoint was not loaded.")
        return self.model.model.infer(batch, norm=norm, copies=copies)

    def anomaly_score(self, batch: torch.Tensor, out_copies: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("DADA checkpoint was not loaded.")
        return self.model.model.cal_anomaly_score(batch_x=batch, batch_out_copies=out_copies)


class DADAAnomalyDetector(AnomalyDetector):
    """DADA zero-shot scorer with native fit/score/alignment logic."""

    def __init__(
        self,
        model: Optional[DADA] = None,
        seq_len: Optional[int] = None,
        batch_size: int = 128,
        copies: int = 10,
        norm: int = 0,
        model_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        trust_remote_code: bool = True,
        auto_download: bool = True,
        cache_dir: Optional[str] = None,
        download_base_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = model if model is not None else DADA(
            seq_len=seq_len,
            model_path=model_path,
            repo_path=repo_path,
            trust_remote_code=trust_remote_code,
            auto_download=auto_download,
            cache_dir=cache_dir,
            download_base_url=download_base_url,
        )
        self.seq_len = int(seq_len) if seq_len is not None else None
        self.batch_size = int(batch_size)
        self.copies = int(copies)
        self.norm = int(norm)
        self.scaler = StandardScaler()

    def _window_size(self) -> int:
        if self.seq_len is not None:
            return int(self.seq_len)
        return self.model.seq_len

    def fit(self, dataset: DataLoader, **kwargs: Any) -> None:
        del kwargs
        frame = dataloader_to_frame(dataset)
        self.scaler.fit(frame.to_numpy())

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        frame = tensor_points_to_frame(inputs)
        scaled = transform_frame(self.scaler, frame)
        windows = build_causal_windows(scaled.to_numpy(), self._window_size())
        loader = make_window_loader(windows, batch_size=self.batch_size, shuffle=False)

        scores = []
        device = inputs[0].device
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                out_copies = self.model.infer(batch, norm=self.norm, copies=self.copies)
                batch_scores = self.model.anomaly_score(batch, out_copies)
                scores.append(batch_scores[:, -1].detach().cpu().numpy())
        if not scores:
            return torch.empty(0, dtype=torch.float32, device=device)
        all_scores = np.concatenate(scores, axis=0)
        return torch.as_tensor(all_scores, dtype=torch.float32, device=device)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


DADAAD = DADAAnomalyDetector

__all__ = ["DADA", "DADAAD", "DADAAnomalyDetector"]
