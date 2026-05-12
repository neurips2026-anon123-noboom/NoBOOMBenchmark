from __future__ import annotations

from typing import Any

from noboom_benchmark.noboom_lib.core.tune import mlflow_datasets


class _FakeMlflowDataset:
    def __init__(self, *, name: str, digest: str) -> None:
        self.name = name
        self.digest = digest
        self.profile = {"rows": 3}

    def to_evaluation_dataset(self) -> str:
        return "evaluation-dataset"

    def _to_mlflow_entity(self) -> dict[str, Any]:
        return {"name": self.name, "digest": self.digest}


def test_get_or_create_logged_dataset_caches_entries(monkeypatch) -> None:
    created = []

    def fake_from_datasource(dataset_source, *, source: str, name: str):
        del dataset_source
        created.append((source, name))
        return _FakeMlflowDataset(name=name, digest="digest-1")

    monkeypatch.setattr(mlflow_datasets, "from_datasource", fake_from_datasource)

    cache = mlflow_datasets.DatasetLineageCache()
    first = mlflow_datasets.get_or_create_logged_dataset(
        cache,
        cache_key="seed-1::predict",
        dataset_source=object(),
        requested_dataset_name="cont_reactive_ome_tsst",
        split_name="test",
        data_source="real",
        source_uri="https://example.test/dataset",
        context="testing",
    )
    second = mlflow_datasets.get_or_create_logged_dataset(
        cache,
        cache_key="seed-1::predict",
        dataset_source=object(),
        requested_dataset_name="cont_reactive_ome_tsst",
        split_name="test",
        data_source="real",
        source_uri="https://example.test/dataset",
        context="testing",
    )

    assert first is second
    assert created == [
        (
            "https://example.test/dataset#cont_reactive_ome_tsst:test",
            "cont_reactive_ome_tsst__test__real",
        )
    ]
    assert first.evaluation_dataset == "evaluation-dataset"


def test_log_dataset_input_adds_context_and_custom_tags() -> None:
    captured = {}

    class FakeClient:
        def log_inputs(self, *, run_id: str, datasets=None, models=None) -> None:
            captured["run_id"] = run_id
            captured["datasets"] = datasets
            captured["models"] = models

    dataset_info = mlflow_datasets.LoggedDatasetInfo(
        context="evaluation",
        split_name="test",
        data_source="real",
        name="cont_reactive_ome__test__real",
        digest="digest-1",
        profile={"rows": 5},
        dataset=_FakeMlflowDataset(name="cont_reactive_ome__test__real", digest="digest-1"),
        evaluation_dataset="evaluation-dataset",
    )

    mlflow_datasets.log_dataset_input(
        FakeClient(),
        "study-run-id",
        dataset_info,
        tags={"split": "test", "phase": "evaluation"},
    )

    dataset_input = captured["datasets"][0]
    assert captured["run_id"] == "study-run-id"
    assert captured["models"] is None
    assert dataset_input.dataset["digest"] == "digest-1"
    assert {(tag.key, tag.value) for tag in dataset_input.tags} == {
        ("mlflow.data.context", "evaluation"),
        ("split", "test"),
        ("phase", "evaluation"),
    }
