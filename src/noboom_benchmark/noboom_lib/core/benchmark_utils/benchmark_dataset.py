"""Lightning data module backed by prepared Arrow IPC dataset manifests."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import tempfile
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import torch
from sklearn.base import TransformerMixin
from timesead.data.transforms import DatasetSource

from .dataset_helpers import TransformCallable, get_dataloader
from .arrow_runtime import SequenceTensorDataset, build_dataset_source
from .style_transfer import StyleTransfer
from ..data_prep.manifest import PreparedDatasetManifest


logger = logging.getLogger(__name__)

MAX_STYLED_PREDICT_BATCH_SIZE = 1024
STYLED_PREDICT_CACHE_FORMAT_VERSION = 3


class BenchmarkDataset(L.LightningDataModule):
    """Manifest-backed benchmark datamodule."""

    def __init__(
        self,
        dataset_name: str,
        requested_dataset_name: Optional[str] = None,
        data_manifest_path: Optional[str] = None,
        splits: Optional[Tuple[float, float]] = None,
        batch_dim: Optional[int] = None,
        batch_size: int = 64,
        predict_batch_size: Optional[int] = None,
        predict_num_workers: Optional[int] = None,
        num_workers: int = 0,
        train_transform: Optional[TransformCallable] = None,
        val_transform: Optional[TransformCallable] = None,
        test_transform: Optional[TransformCallable] = None,
        version: str = "1.0",
        scaler: Optional[TransformerMixin] = None,
        data_source: Literal["real", "synthetic", "both"] = "real",
        gen_to_org_factor: float = -1,
        test_on_org: bool = False,
        styled_predict_batch_size: Optional[int] = None,
        inference_dataset_path: Optional[str] = None,
        scaler_state_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.requested_dataset_name = requested_dataset_name or dataset_name
        self.data_manifest_path = data_manifest_path
        self.splits = splits
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.predict_num_workers = predict_num_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.extra_train_transform = copy.deepcopy(train_transform) if train_transform is not None else None
        self.val_transform = val_transform
        self.extra_val_transform = copy.deepcopy(val_transform) if val_transform is not None else None
        self.test_transform = test_transform
        self.extra_test_transform = copy.deepcopy(test_transform) if test_transform is not None else None
        self.version = version
        self.scaler = scaler
        self.data_source = data_source
        self.gen_to_org_factor = gen_to_org_factor
        self.test_on_org = test_on_org
        self.styled_predict_batch_size = styled_predict_batch_size
        self.inference_dataset_path = inference_dataset_path
        self.scaler_state_path = scaler_state_path

        self.train_data: Optional[DatasetSource] = None
        self.val_data: Optional[DatasetSource] = None
        self.test_data: Optional[DatasetSource] = None
        self.extra_train_data: Optional[DatasetSource] = None
        self.extra_val_data: Optional[DatasetSource] = None
        self.extra_test_data: Optional[DatasetSource] = None
        self._predict_data_override: Optional[DatasetSource] = None
        self._predict_cache_path: Optional[Path] = None
        self._manifest: Optional[PreparedDatasetManifest] = None
        self._seq_len: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        self._num_features = 0

        if self.scaler_state_path is not None:
            with open(self.scaler_state_path, "rb") as handle:
                self.scaler = pickle.load(handle)

        if self.inference_dataset_path is None:
            self._manifest = self._load_or_prepare_manifest()
            self._num_features = self._manifest.modes["train"].num_features if "train" in self._manifest.modes else self._manifest.modes["test"].num_features
            if self.scaler is None and self._manifest.scaler_path is not None:
                with open(self._manifest.scaler_path, "rb") as handle:
                    self.scaler = pickle.load(handle)
        elif self.scaler is not None and hasattr(self.scaler, "feature_names_out_"):
            self._num_features = len(self.scaler.feature_names_out_)

    def _load_or_prepare_manifest(self) -> PreparedDatasetManifest:
        if self.data_manifest_path is None:
            raise ValueError(
                "BenchmarkDataset requires data_manifest_path unless inference_dataset_path is configured. "
                "Prepare dataset manifests before constructing the datamodule."
            )
        return PreparedDatasetManifest.load(self.data_manifest_path)

    def _load_inference_dataset_source(self) -> DatasetSource:
        if self.inference_dataset_path is None:
            raise RuntimeError("inference_dataset_path is not configured.")
        payload = torch.load(self.inference_dataset_path, map_location="cpu")
        samples = payload["samples"]
        targets = payload.get("targets")
        dataset = SequenceTensorDataset(samples, targets)
        if self.scaler is not None:
            scaled_samples = []
            for sample in dataset._samples:
                transformed = self.scaler.transform(sample)
                scaled_samples.append(torch.as_tensor(transformed, dtype=torch.float32))
            dataset = SequenceTensorDataset(scaled_samples, dataset._targets)
            if hasattr(self.scaler, "feature_names_out_"):
                self._num_features = len(self.scaler.feature_names_out_)
        else:
            self._num_features = dataset.num_features
        return DatasetSource(dataset)

    def _attached_trainer(self) -> Optional[L.Trainer]:
        return getattr(self, "trainer", None)

    def _barrier(self, name: str) -> None:
        trainer = self._attached_trainer()
        if trainer is None or trainer.strategy is None:
            return
        trainer.strategy.barrier(name)

    def _is_global_zero(self) -> bool:
        trainer = self._attached_trainer()
        if trainer is None:
            return True
        return bool(trainer.is_global_zero)

    def _styled_predict_cache_dir(self) -> Path:
        manifest_root = Path(self.data_manifest_path).expanduser().resolve().parent if self.data_manifest_path else Path(tempfile.gettempdir())
        return manifest_root / ".noboom_cache" / "styled_predict"

    def _predict_source_signature(self) -> Dict[str, object]:
        if self.test_data is None:
            return {}
        predict_source = self.test_transform(self.test_data) if self.test_transform is not None else self.test_data
        source_length = len(predict_source)
        signature: Dict[str, object] = {"length": source_length}
        if source_length == 0:
            return signature

        def summarize_item(item_index: int, prefix: str) -> None:
            item_inputs, item_targets = predict_source.get_datapoint(item_index)
            input_tensor = item_inputs[0].to(torch.float32)
            target_tensor = item_targets[0].to(torch.float32)
            signature[f"{prefix}_input_shape"] = tuple(input_tensor.shape)
            signature[f"{prefix}_target_shape"] = tuple(target_tensor.shape)
            signature[f"{prefix}_input_sum"] = round(float(input_tensor.sum().item()), 6)
            signature[f"{prefix}_target_sum"] = round(float(target_tensor.sum().item()), 6)

        summarize_item(0, "first")
        summarize_item(source_length - 1, "last")
        return signature

    def _styled_predict_cache_path(self, style_transfer: StyleTransfer) -> Path:
        cache_params = {
            "cache_format_version": STYLED_PREDICT_CACHE_FORMAT_VERSION,
            "dataset_name": self.dataset_name,
            "requested_dataset_name": self.requested_dataset_name,
            "data_manifest_path": self.data_manifest_path,
            "batch_dim": self.batch_dim,
            "test_on_org": self.test_on_org,
            "predict_source_signature": self._predict_source_signature(),
            "style_transfer": style_transfer.cache_signature(),
        }
        payload = json.dumps(cache_params, sort_keys=True, default=str).encode("utf-8")
        cache_key = hashlib.sha256(payload).hexdigest()
        return self._styled_predict_cache_dir() / f"{self.dataset_name}__{cache_key}.pt"

    def _load_styled_predict_dataset(self, cache_path: Path) -> DatasetSource:
        payload = torch.load(cache_path, map_location="cpu")
        dataset = SequenceTensorDataset(payload["samples"], payload["targets"])
        return DatasetSource(dataset)

    def _build_styled_predict_cache(
        self,
        style_transfer: StyleTransfer,
        cache_path: Path,
        device: Optional[torch.device] = None,
    ) -> None:
        if self._loader_data_source("predict") != "real":
            raise RuntimeError("Styled predict cache only supports predict loaders backed by real test data.")

        style_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        style_transfer.setup()
        style_transfer.eval()
        style_transfer.to(style_device)

        build_batch_size = self._resolve_styled_predict_batch_size(style_transfer, style_device)
        build_loader = get_dataloader(
            self.test_data,
            self.test_transform,
            self.batch_dim,
            shuffle=False,
            batch_size=build_batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            extra_data=None,
            extra_pipeline=None,
            data_source="real",
            loader_name="predict_style_cache_build",
        )

        styled_batches: List[torch.Tensor] = []
        target_batches: List[torch.Tensor] = []
        with torch.inference_mode():
            for batch_inputs, batch_targets in build_loader:
                styled_inputs = style_transfer(batch_inputs[0].to(style_device)).cpu().contiguous()
                styled_targets = batch_targets[0].cpu().contiguous()
                styled_batches.append(styled_inputs)
                target_batches.append(styled_targets)

        if style_device.type == "cuda":
            style_transfer.to("cpu")
            torch.cuda.empty_cache()

        if not styled_batches:
            raise RuntimeError(f"Styled predict cache for '{self.dataset_name}' produced no windows.")

        payload = {
            "samples": torch.cat(styled_batches, dim=0),
            "targets": torch.cat(target_batches, dim=0),
            "metadata": {
                "cache_format_version": STYLED_PREDICT_CACHE_FORMAT_VERSION,
                "dataset_name": self.dataset_name,
                "style_transfer": style_transfer.cache_signature(),
            },
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f"{cache_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _repeat_probe_batch(x: torch.Tensor, batch_dim: int, batch_size: int) -> torch.Tensor:
        repeat_factors = [1] * x.dim()
        repeat_factors[batch_dim] = batch_size
        return x.repeat(*repeat_factors).contiguous()

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower()

    def _styled_predict_probe_input(self) -> torch.Tensor:
        probe_loader = get_dataloader(
            self.test_data,
            self.test_transform,
            self.batch_dim,
            shuffle=False,
            batch_size=1,
            num_workers=0,
            persistent_workers=False,
            extra_data=None,
            extra_pipeline=None,
            data_source="real",
            loader_name="predict_style_cache_probe",
        )
        batch_inputs, _ = next(iter(probe_loader))
        return batch_inputs[0]

    def _probe_styled_predict_batch_size(
        self,
        style_transfer: StyleTransfer,
        probe_input: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Dict[str, float]]:
        batch_dim = self.batch_dim if self.batch_dim is not None else 0
        repeated_probe = self._repeat_probe_batch(probe_input, batch_dim, batch_size)
        try:
            return style_transfer.probe_generation_memory(repeated_probe.to(device, non_blocking=True))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        except RuntimeError as exc:
            if device.type == "cuda" and self._is_cuda_oom(exc):
                torch.cuda.empty_cache()
                return None
            raise

    @staticmethod
    def _bytes_to_gib(num_bytes: float) -> float:
        return float(num_bytes) / float(1024 ** 3)

    def _resolve_styled_predict_batch_size(
        self,
        style_transfer: StyleTransfer,
        device: torch.device,
    ) -> int:
        requested_batch_size = self.styled_predict_batch_size if self.styled_predict_batch_size is not None else self.batch_size
        base_batch_size = min(requested_batch_size, MAX_STYLED_PREDICT_BATCH_SIZE)
        if device.type != "cuda" or not torch.cuda.is_available():
            return base_batch_size

        probe_input = self._styled_predict_probe_input()
        successful_batch_size = base_batch_size
        successful_metrics = self._probe_styled_predict_batch_size(style_transfer, probe_input, successful_batch_size, device)
        while successful_metrics is None and successful_batch_size > 1:
            successful_batch_size = max(1, successful_batch_size // 2)
            successful_metrics = self._probe_styled_predict_batch_size(style_transfer, probe_input, successful_batch_size, device)
        if successful_metrics is None:
            return 1

        total_memory_bytes = float(torch.cuda.get_device_properties(device).total_memory)
        peak_allocated_bytes = max(successful_metrics["peak_allocated_bytes"], 1.0)
        estimated_batch_cap = int(successful_batch_size * (total_memory_bytes / peak_allocated_bytes))
        estimated_batch_cap = max(successful_batch_size + 1, estimated_batch_cap)
        estimated_batch_cap = min(estimated_batch_cap, successful_batch_size * 64)
        estimated_batch_cap = min(estimated_batch_cap, MAX_STYLED_PREDICT_BATCH_SIZE)

        failed_batch_size: Optional[int] = None
        candidate_batch_size = min(successful_batch_size * 2, estimated_batch_cap)
        while failed_batch_size is None and candidate_batch_size <= estimated_batch_cap:
            candidate_metrics = self._probe_styled_predict_batch_size(style_transfer, probe_input, candidate_batch_size, device)
            if candidate_metrics is None:
                failed_batch_size = candidate_batch_size
            else:
                successful_batch_size = candidate_batch_size
                successful_metrics = candidate_metrics
                if candidate_batch_size == estimated_batch_cap:
                    break
                candidate_batch_size = min(candidate_batch_size * 2, estimated_batch_cap)

        if failed_batch_size is None:
            failed_batch_size = estimated_batch_cap + 1

        low = successful_batch_size
        high = failed_batch_size - 1
        while low < high:
            mid = (low + high + 1) // 2
            mid_metrics = self._probe_styled_predict_batch_size(style_transfer, probe_input, mid, device)
            if mid_metrics is None:
                high = mid - 1
            else:
                low = mid
                successful_metrics = mid_metrics

        logger.info(
            "DiffStyle generation memory probe for '%s': batch_size=%d peak_allocated=%.2f GiB peak_reserved=%.2f GiB.",
            self.dataset_name,
            low,
            self._bytes_to_gib(successful_metrics["peak_allocated_bytes"]),
            self._bytes_to_gib(successful_metrics["peak_reserved_bytes"]),
        )
        return low

    def enable_styled_predict_cache(
        self,
        style_transfer: StyleTransfer,
        device: Optional[torch.device] = None,
        load_dataset: bool = True,
    ) -> Path:
        if self.test_data is None:
            self.setup("predict_cache")
        cache_path = self._styled_predict_cache_path(style_transfer)
        if self._predict_data_override is not None and self._predict_cache_path == cache_path:
            return cache_path

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            rebuild_cache = not cache_path.exists()
            if cache_path.exists() and load_dataset:
                try:
                    dataset = self._load_styled_predict_dataset(cache_path)
                except Exception:
                    cache_path.unlink(missing_ok=True)
                    rebuild_cache = True
                else:
                    self._predict_data_override = dataset
                    self._predict_cache_path = cache_path
                    return cache_path

            if rebuild_cache:
                self._build_styled_predict_cache(style_transfer, cache_path, device=device)
                if load_dataset:
                    self._predict_data_override = self._load_styled_predict_dataset(cache_path)
                    self._predict_cache_path = cache_path
        return cache_path

    def _setup_manifest_backed_data(self) -> None:
        if self._manifest is None:
            self._manifest = self._load_or_prepare_manifest()

        self.train_data = build_dataset_source(self._manifest, mode="train")
        self.val_data = build_dataset_source(self._manifest, mode="val")
        self.test_data = build_dataset_source(self._manifest, mode="test")

        self.extra_train_data = None
        self.extra_val_data = None
        self.extra_test_data = None
        if "train_ext" in self._manifest.modes:
            self.extra_train_data = build_dataset_source(self._manifest, mode="train_ext")

        self._seq_len = {
            "train": list(self._manifest.modes["train"].seq_len) + (
                list(self._manifest.modes["train_ext"].seq_len) if "train_ext" in self._manifest.modes else []
            ),
            "val": list(self._manifest.modes["val"].seq_len),
            "test": list(self._manifest.modes["test"].seq_len),
        }
        self._num_features = int(self._manifest.modes["train"].num_features)

    def _setup_impl(self, stage: Optional[str]) -> None:
        del stage
        if self.inference_dataset_path is not None:
            inference_source = self._load_inference_dataset_source()
            seq_lengths = inference_source.seq_len
            if isinstance(seq_lengths, int):
                seq_lengths = [seq_lengths] * len(inference_source)
            self.train_data = None
            self.val_data = None
            self.test_data = inference_source
            self.extra_train_data = None
            self.extra_val_data = None
            self.extra_test_data = None
            self._seq_len = {"train": [], "val": [], "test": list(seq_lengths)}
            return

        self._setup_manifest_backed_data()

    def setup(self, stage: Optional[str] = None) -> None:
        trainer = self._attached_trainer()
        if trainer is None:
            self._setup_impl(stage)
            return
        if trainer.is_global_zero:
            self._setup_impl(stage)
        self._barrier("setup_barrier_1")
        if not trainer.is_global_zero:
            self._setup_impl(stage)
        self._barrier("setup_barrier_2")

    def _loader_data_source(self, split: str) -> Literal["real", "synthetic", "both"]:
        if split == "train" and self.extra_train_data is not None and self.data_source == "real":
            return "both"
        if split in {"test", "predict"} and self._manifest is not None:
            test_source = self._manifest.modes["test"].source_name
            if test_source == "real":
                return "real"
            if test_source == "synthetic":
                return "synthetic"
        return self.data_source

    def train_dataloader(self):
        return get_dataloader(
            self.train_data,
            self.train_transform,
            self.batch_dim,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            extra_data=self.extra_train_data,
            extra_pipeline=self.extra_train_transform,
            data_source=self._loader_data_source("train"),
            gen_to_org_factor=self.gen_to_org_factor,
            persistent_workers=self.num_workers > 0,
            loader_name="train",
        )

    def val_dataloader(self):
        return get_dataloader(
            self.val_data,
            self.val_transform,
            self.batch_dim,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            extra_data=self.extra_val_data,
            extra_pipeline=self.extra_val_transform,
            data_source=self._loader_data_source("val"),
            gen_to_org_factor=self.gen_to_org_factor,
            loader_name="val",
        )

    def test_dataloader(self):
        return get_dataloader(
            self.test_data,
            self.test_transform,
            self.batch_dim,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            extra_data=self.extra_test_data,
            extra_pipeline=self.extra_test_transform,
            data_source=self._loader_data_source("test"),
            remove_window_dim=True,
            loader_name="test",
        )

    def predict_dataloader(self):
        if self._predict_data_override is not None:
            return get_dataloader(
                self._predict_data_override,
                None,
                self.batch_dim,
                shuffle=False,
                batch_size=self.predict_batch_size or self.batch_size,
                num_workers=0,
                persistent_workers=False,
                data_source="real",
                loader_name="predict_cached",
            )
        return get_dataloader(
            self.test_data,
            self.test_transform,
            self.batch_dim,
            shuffle=False,
            batch_size=self.predict_batch_size or self.batch_size,
            num_workers=self.predict_num_workers if self.predict_num_workers is not None else self.num_workers,
            persistent_workers=False,
            extra_data=self.extra_test_data,
            extra_pipeline=self.extra_test_transform,
            data_source=self._loader_data_source("predict"),
            loader_name="predict",
        )

    def predict_orig_dataloader(self):
        return get_dataloader(
            self.test_data,
            None,
            self.batch_dim,
            shuffle=False,
            batch_size=1,
            num_workers=0,
            persistent_workers=False,
            extra_data=self.extra_test_data,
            extra_pipeline=None,
            data_source=self._loader_data_source("predict"),
            loader_name="predict_orig",
        )

    @property
    def num_features(self) -> int:
        return self._num_features

    def num_samples(self, split: str) -> int:
        split_data = {
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data,
        }
        dataset = split_data[split]
        extra_data = {
            "train": self.extra_train_data,
            "val": self.extra_val_data,
            "test": self.extra_test_data,
        }[split]
        total = 0 if dataset is None else len(dataset)
        if extra_data is not None:
            total += len(extra_data)
        return total

    def seq_len(self, mode: str) -> List[int]:
        if mode not in self._seq_len:
            raise ValueError(f"Unknown sequence length mode: {mode}")
        return list(self._seq_len[mode])
