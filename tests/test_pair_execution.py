from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

from noboom_cluster.noboom_cli_lib.specs import PairRunSpec

from noboom_benchmark.tuning_workflow import parse_args
from noboom_benchmark.noboom_lib.core.tune.pair_execution import (
    pair_spec_to_namespace,
    run_pair_spec,
    run_pair_spec_file,
)


def _build_pair_spec() -> PairRunSpec:
    return PairRunSpec(
        experiment_id="1",
        source_experiment_id="2",
        model_name="gdn",
        dataset_name="ome",
        timestamp="20260311_120000",
        storage_path="/tmp/storage/ray/ome__gdn",
        gpus_per_run=0.25,
        optuna_storage_uri="postgresql+psycopg://noboom:example-password@127.0.0.1:5432/optuna_db",
        config_dir="configs",
        temp_dir="/tmp/ray",
        verbose=1,
        tune=True,
        env_file="/tmp/.env",
        tracking_uri="http://127.0.0.1:5000",
        s3_endpoint_url="http://127.0.0.1:8333",
        prepared_dataset_s3_path="s3://prepared-bucket/prepared-root",
    )


def test_pair_spec_to_namespace_maps_fields() -> None:
    pair_spec = _build_pair_spec()

    namespace = pair_spec_to_namespace(pair_spec)

    assert namespace.model_name == "gdn"
    assert namespace.dataset_name == "ome"
    assert namespace.source_experiment_id == "2"
    assert namespace.env_file == "/tmp/.env"
    assert namespace.prepared_dataset_s3_path == "s3://prepared-bucket/prepared-root"
    assert namespace.save_checkpoints is False


def test_pair_spec_to_namespace_maps_save_checkpoints() -> None:
    pair_spec = _build_pair_spec().model_copy(update={"save_checkpoints": True})

    namespace = pair_spec_to_namespace(pair_spec)

    assert namespace.save_checkpoints is True


def test_pair_spec_to_namespace_maps_hpo_seeds() -> None:
    pair_spec = _build_pair_spec().model_copy(update={"hpo_seeds": [42, 44]})

    namespace = pair_spec_to_namespace(pair_spec)

    assert namespace.hpo_seeds == [42, 44]


def test_pair_spec_to_namespace_maps_local_backend_fields() -> None:
    pair_spec = _build_pair_spec().model_copy(
        update={
            "execution_backend": "local",
            "artifact_storage_backend": "local",
            "optuna_storage_backend": "sqlite",
            "optuna_storage_uri": "sqlite:////tmp/optuna.db",
            "s3_endpoint_url": None,
            "prepared_dataset_s3_path": None,
        }
    )

    namespace = pair_spec_to_namespace(pair_spec)

    assert namespace.execution_backend == "local"
    assert namespace.artifact_storage_backend == "local"
    assert namespace.optuna_storage_backend == "sqlite"
    assert namespace.optuna_storage_uri == "sqlite:////tmp/optuna.db"
    assert namespace.s3_endpoint_url is None
    assert namespace.prepared_dataset_s3_path is None


def test_run_pair_spec_uses_tuning_runner_module_from_package(monkeypatch, tmp_path: Path) -> None:
    pair_spec = _build_pair_spec()
    pair_spec_path = tmp_path / "pair-spec.json"
    pair_spec.write_json(str(pair_spec_path))
    captured = {}

    fake_module = ModuleType("noboom_benchmark.noboom_lib.core.tuning_runner")

    def fake_run_tune_or_train(args):
        captured["model_name"] = args.model_name
        captured["dataset_name"] = args.dataset_name
        captured["tracking_uri"] = args.tracking_uri
        return ("gdn", "ome", {"score": 0.99})

    fake_module.run_tune_or_train = fake_run_tune_or_train  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "noboom_benchmark.noboom_lib.core.tuning_runner",
        fake_module,
    )

    assert run_pair_spec(pair_spec) == ("gdn", "ome", {"score": 0.99})
    assert run_pair_spec_file(str(pair_spec_path)) == ("gdn", "ome", {"score": 0.99})
    assert captured == {
        "model_name": "gdn",
        "dataset_name": "ome",
        "tracking_uri": "http://127.0.0.1:5000",
    }


def test_parse_args_accepts_pair_spec_without_legacy_required_args() -> None:
    namespace = parse_args(["--pair-spec", "/tmp/pair-spec.json"])

    assert namespace.pair_spec == "/tmp/pair-spec.json"
    assert namespace.experiment_id is None
    assert namespace.model_name is None
    assert namespace.optuna_storage_uri is None


def test_parse_args_accepts_prepared_dataset_s3_path() -> None:
    namespace = parse_args(
        [
            "--pair-spec",
            "/tmp/pair-spec.json",
            "--prepared-dataset-s3-path",
            "s3://prepared-bucket/prepared-root",
        ]
    )

    assert namespace.prepared_dataset_s3_path == "s3://prepared-bucket/prepared-root"


def test_parse_args_accepts_save_checkpoints() -> None:
    namespace = parse_args(
        [
            "--experiment-id",
            "1",
            "--model-name",
            "gdn",
            "--dataset-name",
            "ome",
            "--timestamp",
            "20260311_120000",
            "--storage-path",
            "/tmp/storage",
            "--gpus-per-run",
            "0.25",
            "--optuna-storage-uri",
            "sqlite:///optuna.db",
            "--save-checkpoints",
        ]
    )

    assert namespace.save_checkpoints is True


def test_parse_args_accepts_inline_pair_spec_without_legacy_required_args() -> None:
    namespace = parse_args(["--pair-spec-b64", _build_pair_spec().to_base64()])

    assert namespace.pair_spec_b64 is not None
    assert namespace.experiment_id is None
    assert namespace.model_name is None
    assert namespace.optuna_storage_uri is None


def test_pair_spec_base64_round_trip() -> None:
    pair_spec = _build_pair_spec()

    loaded = PairRunSpec.from_base64(pair_spec.to_base64())

    assert loaded == pair_spec
