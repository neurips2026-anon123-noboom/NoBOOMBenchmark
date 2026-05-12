from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.exceptions import MlflowException

from noboom_benchmark.noboom_lib.core.benchmark_utils import mlflow_dataset


def test_from_datasource_falls_back_for_custom_uri_scheme(monkeypatch) -> None:
    import mlflow.data.dataset_source_registry as dataset_source_registry
    from mlflow.tracking.context import registry

    def fake_resolve_dataset_source(raw_source):
        raise MlflowException(f"Unsupported source: {raw_source}")

    monkeypatch.setattr(dataset_source_registry, "resolve_dataset_source", fake_resolve_dataset_source)
    monkeypatch.setattr(registry, "resolve_tags", lambda: {"existing": "tag"})

    dataset = mlflow_dataset.from_datasource(
        object(),
        source="generated://cont_reactive_ome_tsst/train",
        name="cont_reactive_ome_tsst__train__synthetic",
        digest="digest-1",
    )

    assert isinstance(dataset.source, CodeDatasetSource)
    assert dataset.source.to_dict()["tags"] == {
        "existing": "tag",
        "noboom.dataset_source_uri": "generated://cont_reactive_ome_tsst/train",
    }
