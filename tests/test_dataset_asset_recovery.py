from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from noboom_benchmark.noboom_lib.core.benchmark_utils import dataset_helpers
from noboom_benchmark.noboom_lib.core.data_prep import pipeline


def test_ensure_dataset_assets_retries_with_forced_download_when_cached_copy_is_incomplete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    initial_root = tmp_path / "initial-download"
    (initial_root / "noboom" / "industry_process").mkdir(parents=True, exist_ok=True)

    refreshed_root = tmp_path / "refreshed-download"
    refreshed_dataset_dir = refreshed_root / "noboom" / "industry_process"
    refreshed_dataset_dir.mkdir(parents=True, exist_ok=True)
    (refreshed_dataset_dir / "train_normal.csv").write_text(
        "f1,Anomaly\n1.0,0\n",
        encoding="utf-8",
    )

    download_calls = []

    def fake_dataset_download(
        dataset_slug: str,
        path: Optional[str] = None,
        *,
        force_download: bool = False,
    ) -> str:
        del path
        assert dataset_slug == pipeline.KAGGLE_DATASET_SLUG
        download_calls.append(force_download)
        return str(refreshed_root if force_download else initial_root)

    monkeypatch.setattr(pipeline.kagglehub, "dataset_download", fake_dataset_download)

    noboom_root, source_uri = pipeline.ensure_dataset_assets(
        "industry_process",
        runtime_config=SimpleNamespace(seafile_username=None, seafile_password=None),
        include_generated=False,
    )

    assert noboom_root == refreshed_root / "noboom"
    assert source_uri == f"kaggle://{pipeline.KAGGLE_DATASET_SLUG}"
    assert download_calls == [False, True]


def test_get_dataset_data_refreshes_raw_assets_once_after_missing_training_file(
    monkeypatch,
) -> None:
    resolve_calls = []

    def fake_resolve_dataset_variant(dataset_name: str, data_dir: str):
        resolve_calls.append((dataset_name, data_dir))
        if data_dir == "broken-root":
            raise FileNotFoundError("missing training file")
        return dataset_helpers.DatasetVariant(
            requested_name=dataset_name,
            base_dataset_name="cont_reactive_ome",
        )

    refresh_calls = []

    def fake_refresh_raw_data_root(dataset_name: str, *, include_generated: bool) -> str:
        refresh_calls.append((dataset_name, include_generated))
        return "repaired-root"

    torch_dataset_calls = []

    class FakeTorchDataset:
        def __init__(self, name: str, version: str, *, root: str, train: bool, fast_load: bool) -> None:
            del version
            del fast_load
            torch_dataset_calls.append((name, root, train))
            self.name = name
            self.root = root
            self.train = train
            self.seq_len = [4]

        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(dataset_helpers, "resolve_dataset_variant", fake_resolve_dataset_variant)
    monkeypatch.setattr(dataset_helpers, "_refresh_raw_data_root", fake_refresh_raw_data_root)
    monkeypatch.setattr(dataset_helpers, "TorchDataset", FakeTorchDataset)
    monkeypatch.setattr(dataset_helpers, "make_dataset_split", lambda dataset, *args, axis: ("train-split", "val-split"))
    monkeypatch.setattr(dataset_helpers, "DatasetSource", lambda dataset: ("dataset-source", dataset.root, dataset.train))

    result = dataset_helpers.get_dataset_data(
        dataset_name="cont_reactive_ome_ext",
        splits=(0.7, 0.3),
        data_dir="broken-root",
        data_source="real",
    )

    assert resolve_calls == [
        ("cont_reactive_ome_ext", "broken-root"),
        ("cont_reactive_ome_ext", "repaired-root"),
    ]
    assert refresh_calls == [("cont_reactive_ome_ext", False)]
    assert torch_dataset_calls == [
        ("cont_reactive_ome", "repaired-root", True),
        ("cont_reactive_ome", "repaired-root", False),
    ]
    assert result.train_data == "train-split"
    assert result.val_data == "val-split"
    assert result.test_data == ("dataset-source", "repaired-root", False)
