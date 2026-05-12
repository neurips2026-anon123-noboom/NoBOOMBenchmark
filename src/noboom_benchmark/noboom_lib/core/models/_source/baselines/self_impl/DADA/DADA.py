"""HuggingFace-backed DADA wrapper for the CATCH anomaly benchmark."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
try:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
except ImportError:  # pragma: no cover - only needed when loading an official checkpoint.
    get_class_from_dynamic_module = None

from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.recent_utils import (
    build_causal_windows,
    first_existing_path,
    make_window_loader,
    transform_frame,
)
from noboom_benchmark.noboom_lib.core.models._source.common.constant import ROOT_PATH


DEFAULT_HYPER_PARAMS = {
    # The official zero-shot scripts mostly use batch size 32 outside MSL.
    "batch_size": 32,
    "copies": 10,
    "model_path": None,
    "norm": 0,
    "repo_path": None,
    "seq_len": 100,
    "trust_remote_code": True,
}


class DADAConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_model_from_path(model_path: Path):
    """Load the official DADA model via HuggingFace's dynamic module loader."""

    if get_class_from_dynamic_module is None:
        raise ImportError("Loading a DADA checkpoint requires the optional `transformers` dependency.")

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


def _patched_dada_infer(self, x: torch.Tensor, norm: int = 0, mask_mode: Optional[str] = None, copies: int = 10):
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
    """Zero-shot DADA scorer.

    The model weights are resolved from a local DADA checkout or an explicit
    `model_path`. We keep only the benchmark-facing wrapper here and delegate
    the architecture loading to HuggingFace's dynamic-module loader so the
    official repo code can be reused without vendoring the full model file.
    """

    def __init__(self, **kwargs):
        self.config = DADAConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def detect_fit(self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None) -> None:
        self.scaler.fit(train_data.to_numpy())

    def _resolve_model_path(self) -> Path:
        explicit = []
        if self.config.model_path:
            explicit.append(Path(self.config.model_path))
        if self.config.repo_path:
            repo_path = Path(self.config.repo_path)
            explicit.extend([repo_path / "DADA", repo_path])

        env_model_path = os.environ.get("DADA_MODEL_PATH")
        env_repo_path = os.environ.get("DADA_REPO_PATH")
        if env_model_path:
            explicit.append(Path(env_model_path))
        if env_repo_path:
            repo_path = Path(env_repo_path)
            explicit.extend([repo_path / "DADA", repo_path])

        explicit.extend(
            [
                Path(ROOT_PATH).parent / "DADA" / "DADA",
                Path(ROOT_PATH).parent / "DADA",
            ]
        )
        resolved = first_existing_path(explicit)
        if resolved is None:
            raise FileNotFoundError(
                "Could not resolve a DADA model directory. Set `model_path`, "
                "`repo_path`, `DADA_MODEL_PATH`, or clone the official DADA repo "
                "next to the CATCH workspace."
            )
        return resolved

    def _load_model(self) -> None:
        if self.model is not None:
            return

        model_path = self._resolve_model_path()
        self.model = _load_model_from_path(model_path).to(self.device)
        self.model.eval()

        core_model = getattr(self.model, "model", None)
        if core_model is not None and hasattr(core_model, "input_embed") and hasattr(core_model, "encoder"):
            core_model.infer = types.MethodType(_patched_dada_infer, core_model)

        loaded_seq_len = getattr(getattr(self.model, "config", None), "seq_len", self.config.seq_len)
        if isinstance(loaded_seq_len, (list, tuple)):
            loaded_seq_len = loaded_seq_len[0]
        self.config.seq_len = int(loaded_seq_len)

    def detect_score(self, test: pd.DataFrame):
        self._load_model()
        scaled = transform_frame(self.scaler, test)
        windows = build_causal_windows(scaled.to_numpy(), self.config.seq_len)
        loader = make_window_loader(windows, batch_size=self.config.batch_size, shuffle=False)

        scores = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                out_copies = self.model.model.infer(batch, norm=self.config.norm, copies=self.config.copies)
                batch_scores = self.model.model.cal_anomaly_score(batch_x=batch, batch_out_copies=out_copies)
                scores.append(batch_scores[:, -1].detach().cpu().numpy())
        all_scores = np.concatenate(scores, axis=0)
        return all_scores, all_scores
