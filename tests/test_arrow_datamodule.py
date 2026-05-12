from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle

import lightning.pytorch as pl
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from torch import nn

from noboom_benchmark.noboom_lib.core.benchmark_utils.arrow_runtime import build_dataset_source
from noboom_benchmark.noboom_lib.core.benchmark_utils.benchmark_dataset import BenchmarkDataset
from noboom_benchmark.noboom_lib.core.data_prep.manifest import (
    PreparedDatasetManifest,
    PreparedModeManifest,
)
from noboom_benchmark.noboom_lib.core.data_prep.pipeline import (
    CanonicalSequence,
    PreparationRecipe,
    _write_arrow_mode_arrays,
    build_manifest_export,
    fit_and_transform_sequences,
    run_preparation_pipeline,
)


class FeatureExpandingScaler:
    def __init__(self) -> None:
        self.feature_names_in_ = ["f1", "f2"]
        self.feature_names_out_ = ["f1", "f2", "sum", "diff"]

    def transform(self, frame) -> np.ndarray:
        values = np.asarray(frame, dtype=np.float32)
        return np.column_stack(
            [
                values,
                values[:, 0] + values[:, 1],
                values[:, 0] - values[:, 1],
            ]
        )


def _write_mode(root: Path, mode: str, rows: list[dict[str, object]], source_name: str) -> PreparedModeManifest:
    mode_dir = root / mode
    mode_paths = _write_arrow_mode_arrays(
        mode_dir,
        [
            {
                "sequence_id": str(row["sequence_id"]),
                "features": np.asarray(row["features"], dtype=np.float32),
                "labels": np.asarray(row["labels"], dtype=np.int64),
                "requested_dataset_name": "test_dataset",
                "base_dataset_name": "test_dataset",
                "mode": str(row["mode"]),
                "source_name": str(row["source_name"]),
            }
            for row in rows
        ],
    )
    return PreparedModeManifest(
        mode=mode,
        source_name=source_name,
        data_source="real",
        item_count=len(rows),
        num_features=int(np.asarray(rows[0]["features"]).shape[-1]) if rows else 0,
        seq_len=[int(np.asarray(row["features"]).shape[0]) for row in rows],
        feature_values_path=mode_paths["feature_values_path"],
        label_values_path=mode_paths["label_values_path"],
    )


def _build_manifest(tmp_path: Path) -> str:
    root = tmp_path / "prepared" / "fingerprint"
    root.mkdir(parents=True, exist_ok=True)

    train_rows = [
        {
            "sequence_id": "train:0",
            "features": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "labels": np.asarray([0, 0], dtype=np.int64),
            "mode": "train",
            "source_name": "real",
        },
        {
            "sequence_id": "train:1",
            "features": np.asarray([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
            "labels": np.asarray([0, 1, 0], dtype=np.int64),
            "mode": "train",
            "source_name": "real",
        },
    ]
    val_rows = [
        {
            "sequence_id": "val:0",
            "features": np.asarray([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32),
            "labels": np.asarray([0, 0], dtype=np.int64),
            "mode": "val",
            "source_name": "real",
        },
    ]
    test_rows = [
        {
            "sequence_id": "test:0",
            "features": np.asarray([[15.0, 16.0], [17.0, 18.0], [19.0, 20.0]], dtype=np.float32),
            "labels": np.asarray([0, 1, 1], dtype=np.int64),
            "mode": "test",
            "source_name": "real",
        },
    ]
    ext_rows = [
        {
            "sequence_id": "train_ext:0",
            "features": np.asarray([[21.0, 22.0], [23.0, 24.0]], dtype=np.float32),
            "labels": np.asarray([0, 0], dtype=np.int64),
            "mode": "train_ext",
            "source_name": "extension",
        },
    ]

    manifest = PreparedDatasetManifest(
        manifest_version=2,
        fingerprint="fingerprint",
        exporter_version="arrow-ipc-v1",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome_ext",
        base_dataset_name="cont_reactive_ome",
        stage_data_source="real",
        version="1.0",
        source_uri="kaggle://faebs94/noboom-anomaly-detection-in-chemical-processes/versions/1",
        recipe={
            "splits": [0.7, 0.3],
            "gen_to_org_factor": -1.0,
            "test_on_org": False,
            "scaler_spec": None,
        },
        feature_names=["f1", "f2"],
        scaler_path=None,
        feature_names_path=str(root / "feature_names.json"),
        modes={
            "train": _write_mode(root, "train", train_rows, "real"),
            "val": _write_mode(root, "val", val_rows, "real"),
            "test": _write_mode(root, "test", test_rows, "real"),
            "train_ext": _write_mode(root, "train_ext", ext_rows, "extension"),
        },
    )
    feature_names_path = Path(manifest.feature_names_path)
    feature_names_path.write_text(json.dumps(manifest.feature_names), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest.write(str(manifest_path))
    return str(manifest_path)


def _build_synthetic_manifest(tmp_path: Path) -> str:
    root = tmp_path / "prepared" / "synthetic_fingerprint"
    root.mkdir(parents=True, exist_ok=True)

    def _row(sequence_id: str, mode: str, values: list[list[float]]) -> dict[str, object]:
        return {
            "sequence_id": sequence_id,
            "features": np.asarray(values, dtype=np.float32),
            "labels": np.zeros(len(values), dtype=np.int64),
            "mode": mode,
            "source_name": "synthetic",
        }

    train_rows = [_row("train_tsst:0", "train_tsst", [[1.0, 2.0], [3.0, 4.0]])]
    val_rows = [_row("val_tsst:0", "val_tsst", [[5.0, 6.0], [7.0, 8.0]])]
    test_rows = [_row("test_tsst:0", "test_tsst", [[9.0, 10.0], [11.0, 12.0]])]

    manifest = PreparedDatasetManifest(
        manifest_version=2,
        fingerprint="synthetic_fingerprint",
        exporter_version="arrow-ipc-v1",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome_tsst",
        base_dataset_name="cont_reactive_ome",
        stage_data_source="synthetic",
        version="1.0",
        source_uri="kaggle://faebs94/noboom-anomaly-detection-in-chemical-processes/versions/1",
        recipe={
            "splits": [0.7, 0.3],
            "gen_to_org_factor": -1.0,
            "test_on_org": False,
            "scaler_spec": None,
        },
        feature_names=["f1", "f2"],
        scaler_path=None,
        feature_names_path=str(root / "feature_names.json"),
        modes={
            "train": _write_mode(root, "train", train_rows, "synthetic"),
            "val": _write_mode(root, "val", val_rows, "synthetic"),
            "test": _write_mode(root, "test", test_rows, "synthetic"),
        },
    )
    feature_names_path = Path(manifest.feature_names_path)
    feature_names_path.write_text(json.dumps(manifest.feature_names), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest.write(str(manifest_path))
    return str(manifest_path)


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    manifest = PreparedDatasetManifest.load(manifest_path)

    assert manifest.requested_dataset_name == "cont_reactive_ome_ext"
    assert set(manifest.modes) == {"train", "val", "test", "train_ext"}
    assert manifest.modes["train"].seq_len == [2, 3]
    assert manifest.feature_names == ["f1", "f2"]

def test_benchmark_dataset_reads_manifest_and_exposes_expected_lengths(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)

    datamodule = BenchmarkDataset(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome_ext",
        data_manifest_path=manifest_path,
        batch_dim=0,
        batch_size=1,
        num_workers=0,
        data_source="real",
    )
    datamodule.setup("fit")

    assert datamodule.num_features == 2
    assert datamodule.seq_len("train") == [2, 3, 2]
    assert datamodule.seq_len("val") == [2]
    assert datamodule.seq_len("test") == [3]
    assert datamodule.num_samples("train") == 3
    assert datamodule.extra_train_data is not None

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert len(train_loader.dataset) == 3
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1

    train_inputs, train_targets = next(iter(train_loader))
    assert train_inputs[0].shape[-1] == 2
    assert train_targets[0].dtype.is_floating_point is False


def test_benchmark_dataset_uses_predict_batch_size_for_prediction_loader(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)

    datamodule = BenchmarkDataset(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome_ext",
        data_manifest_path=manifest_path,
        batch_dim=0,
        batch_size=1,
        predict_batch_size=2,
        num_workers=0,
        data_source="real",
    )
    datamodule.setup("fit")

    assert datamodule.train_dataloader().batch_size == 1
    assert datamodule.predict_dataloader().batch_size == 2


def test_benchmark_dataset_supports_synthetic_only_manifest(tmp_path: Path) -> None:
    manifest_path = _build_synthetic_manifest(tmp_path)

    datamodule = BenchmarkDataset(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome_tsst",
        data_manifest_path=manifest_path,
        batch_dim=0,
        batch_size=1,
        num_workers=0,
        data_source="synthetic",
    )
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    batch_inputs, batch_targets = next(iter(train_loader))

    assert len(train_loader.dataset) == 1
    assert batch_inputs[0].shape[-1] == 2
    assert batch_targets[0].dtype.is_floating_point is False


def test_arrow_writer_is_safe_from_background_thread(tmp_path: Path) -> None:
    mode_dir = tmp_path / "threaded"
    rows = [
        {
            "sequence_id": "train:0",
            "features": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "labels": np.asarray([0, 1], dtype=np.int64),
            "requested_dataset_name": "test_dataset",
            "base_dataset_name": "test_dataset",
            "mode": "train",
            "source_name": "real",
        }
    ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_write_arrow_mode_arrays, mode_dir, rows)
        paths = future.result()

    assert Path(paths["feature_values_path"]).exists()
    assert Path(paths["label_values_path"]).exists()


def test_arrow_runtime_supports_multi_record_batch_export(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "noboom_benchmark.noboom_lib.core.data_prep.pipeline.ARROW_EXPORT_TARGET_BATCH_BYTES",
        32,
    )
    manifest_path = _build_manifest(tmp_path)
    manifest = PreparedDatasetManifest.load(manifest_path)

    feature_source = pa.memory_map(manifest.modes["train"].feature_values_path, "r")
    feature_reader = ipc.open_file(feature_source)
    try:
        assert feature_reader.num_record_batches > 1
    finally:
        feature_source.close()

    datamodule = BenchmarkDataset(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome_ext",
        data_manifest_path=manifest_path,
        batch_dim=0,
        batch_size=1,
        num_workers=0,
        data_source="real",
    )
    datamodule.setup("fit")

    train_inputs, train_targets = next(iter(datamodule.train_dataloader()))
    assert train_inputs[0].shape[-1] == 2
    assert train_targets[0].dtype.is_floating_point is False


def test_arrow_runtime_dataset_round_trips_through_pickle(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    manifest = PreparedDatasetManifest.load(manifest_path)

    dataset = build_dataset_source(manifest, mode="train")
    restored_dataset = pickle.loads(pickle.dumps(dataset))
    batch_inputs, batch_targets = restored_dataset[0]

    assert batch_inputs[0].shape == (2, 2)
    assert batch_targets[0].dtype.is_floating_point is False


def test_benchmark_dataset_supports_lightning_fit_with_worker_processes(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)

    datamodule = BenchmarkDataset(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome_ext",
        data_manifest_path=manifest_path,
        batch_dim=0,
        batch_size=1,
        num_workers=2,
        data_source="real",
    )

    class TinyModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(2, 2)

        def training_step(self, batch, batch_idx):  # type: ignore[override]
            del batch_idx
            x = batch[0][0].float().mean(dim=1)
            return self.layer(x).sum()

        def validation_step(self, batch, batch_idx):  # type: ignore[override]
            del batch
            del batch_idx
            return None

        def configure_optimizers(self):  # type: ignore[override]
            return torch.optim.SGD(self.parameters(), lr=0.1)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    trainer.fit(TinyModule(), datamodule=datamodule)


def test_preparation_pipeline_parallelizes_mode_transform_and_export(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NOBOOM_PREP_MAX_WORKERS", "2")
    raw_dataset_dir = tmp_path / "raw" / "cont_reactive_ome"
    raw_dataset_dir.mkdir(parents=True, exist_ok=True)
    (raw_dataset_dir / "train.csv").write_text("f1,f2\n1.0,2.0\n3.0,4.0\n", encoding="utf-8")

    recipe = PreparationRecipe(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        base_dataset_name="cont_reactive_ome",
        version="1.0",
        stage_data_source="real",
        splits=(0.7, 0.3),
        gen_to_org_factor=-1.0,
        test_on_org=False,
        scaler_spec={
            "class_path": "sklearn.preprocessing.StandardScaler",
            "init_args": {},
        },
        raw_data_root=str(tmp_path / "raw"),
        source_uri="kaggle://example/dataset",
        raw_provenance=[],
    )
    mode_rows = {
        "scaler_train_real": [
            CanonicalSequence(
                sequence_id="train:0",
                features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                labels=np.asarray([0, 0], dtype=np.int64),
                requested_dataset_name="cont_reactive_ome",
                base_dataset_name="cont_reactive_ome",
                mode="train",
                source_name="real",
            )
        ],
        "train": [
            CanonicalSequence(
                sequence_id="train:0",
                features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                labels=np.asarray([0, 0], dtype=np.int64),
                requested_dataset_name="cont_reactive_ome",
                base_dataset_name="cont_reactive_ome",
                mode="train",
                source_name="real",
            )
        ],
        "val": [
            CanonicalSequence(
                sequence_id="val:0",
                features=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
                labels=np.asarray([0, 1], dtype=np.int64),
                requested_dataset_name="cont_reactive_ome",
                base_dataset_name="cont_reactive_ome",
                mode="val",
                source_name="real",
            )
        ],
        "test": [
            CanonicalSequence(
                sequence_id="test:0",
                features=np.asarray([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
                labels=np.asarray([1, 1], dtype=np.int64),
                requested_dataset_name="cont_reactive_ome",
                base_dataset_name="cont_reactive_ome",
                mode="test",
                source_name="real",
            )
        ],
    }

    scaler, feature_names, transformed_modes = fit_and_transform_sequences(recipe, mode_rows)

    assert scaler is not None
    assert feature_names == ["f1", "f2"]
    assert set(transformed_modes) == {"train", "val", "test"}
    assert transformed_modes["test"][0].features.shape == (2, 2)

    manifest_payload = build_manifest_export(
        recipe=recipe,
        fingerprint="parallel_fingerprint",
        prepared_root=str(tmp_path / "prepared"),
        mode_rows=transformed_modes,
        feature_names=feature_names,
        scaler=scaler,
    )
    manifest = PreparedDatasetManifest.load(str(manifest_payload["manifest_path"]))

    assert set(manifest.modes) == {"train", "val", "test"}
    assert Path(manifest.modes["train"].feature_values_path).exists()
    assert Path(manifest.modes["val"].feature_values_path).exists()
    assert Path(manifest.modes["test"].feature_values_path).exists()


def test_run_preparation_pipeline_returns_generated_manifest_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "prepared" / "fingerprint" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    recipe = PreparationRecipe(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        base_dataset_name="cont_reactive_ome",
        version="1.0",
        stage_data_source="real",
        splits=(0.7, 0.3),
        gen_to_org_factor=-1.0,
        test_on_org=False,
        scaler_spec=None,
        raw_data_root=str(tmp_path / "raw"),
        source_uri="kaggle://example/dataset",
        raw_provenance=[],
    )

    def fake_execute_preparation_steps(*, recipe: PreparationRecipe, fingerprint: str, prepared_root: Path):
        del recipe
        del fingerprint
        del prepared_root
        return {"manifest_path": str(manifest_path)}

    monkeypatch.setattr(
        "noboom_benchmark.noboom_lib.core.data_prep.pipeline._execute_preparation_steps",
        fake_execute_preparation_steps,
    )

    resolved_path = run_preparation_pipeline(
        recipe=recipe,
        prepared_root=tmp_path / "prepared",
    )

    assert resolved_path == str(manifest_path)


def test_build_manifest_export_tracks_transformed_feature_width_in_manifest(tmp_path: Path) -> None:
    recipe = PreparationRecipe(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        base_dataset_name="cont_reactive_ome",
        version="1.0",
        stage_data_source="real",
        splits=(0.7, 0.3),
        gen_to_org_factor=-1.0,
        test_on_org=False,
        scaler_spec=None,
        raw_data_root=str(tmp_path / "raw"),
        source_uri="kaggle://example/dataset",
        raw_provenance=[],
    )
    mode_rows = {
        "train": [
            CanonicalSequence(
                sequence_id="train:0",
                features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                labels=np.asarray([0, 0], dtype=np.int64),
                requested_dataset_name="cont_reactive_ome",
                base_dataset_name="cont_reactive_ome",
                mode="train",
                source_name="real",
            )
        ],
        "val": [
            CanonicalSequence(
                sequence_id="val:0",
                features=np.asarray([[5.0, 6.0]], dtype=np.float32),
                labels=np.asarray([0], dtype=np.int64),
                requested_dataset_name="cont_reactive_ome",
                base_dataset_name="cont_reactive_ome",
                mode="val",
                source_name="real",
            )
        ],
        "test": [
            CanonicalSequence(
                sequence_id="test:0",
                features=np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
                labels=np.asarray([1, 1], dtype=np.int64),
                requested_dataset_name="cont_reactive_ome",
                base_dataset_name="cont_reactive_ome",
                mode="test",
                source_name="real",
            )
        ],
    }

    manifest_payload = build_manifest_export(
        recipe=recipe,
        fingerprint="expanding_scaler_fingerprint",
        prepared_root=str(tmp_path / "prepared"),
        mode_rows=mode_rows,
        feature_names=["f1", "f2", "sum", "diff"],
        scaler=FeatureExpandingScaler(),
        input_feature_names=["f1", "f2"],
    )
    manifest = PreparedDatasetManifest.load(str(manifest_payload["manifest_path"]))

    assert manifest.modes["train"].num_features == 4
    train_source = build_dataset_source(manifest, mode="train")
    train_inputs, _ = train_source.dataset[0]
    assert tuple(train_inputs[0].shape) == (2, 4)
