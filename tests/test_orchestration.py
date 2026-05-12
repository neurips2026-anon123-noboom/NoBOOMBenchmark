from __future__ import annotations

from argparse import Namespace
import gc
from pathlib import Path
from types import SimpleNamespace

import pytest

from noboom_cluster.noboom_cli_lib.specs import DependencyManifest, PairResult

from noboom_benchmark.run_tune import parse_args as parse_run_tune_args
from noboom_benchmark.noboom_lib.core.tune import orchestration


def _controller_args(
    *,
    dataset: list[str],
    model: list[str],
    pair: list[str],
) -> Namespace:
    return Namespace(
        dataset=dataset,
        model=model,
        pair=pair,
        tune=True,
        experiment_id=None,
        config_dir="configs",
        temp_dir="/tmp/ray",
        gpus_per_run=0.25,
        timestamp="20260311_120000",
        verbose=0,
        env_file=None,
        max_in_flight=1,
    )


def test_controller_spec_canonicalizes_model_names() -> None:
    args = Namespace(
        dataset=["ome"],
        model=["Anomaly_transformer", "GDN", "gdn"],
        pair=[],
        tune=True,
        experiment_id=None,
        config_dir="configs",
        temp_dir="/tmp/ray",
        gpus_per_run=0.25,
        timestamp="20260311_120000",
        verbose=0,
        env_file=None,
        max_in_flight=1,
    )

    spec = orchestration._build_controller_spec(args)

    assert spec.models == ["anomaly_transformer", "gdn"]


def test_controller_spec_expands_selected_datasets_and_models() -> None:
    spec = orchestration._build_controller_spec(
        _controller_args(
            dataset=["srb", "ome", "ome"],
            model=["LSTM_AE", "GDN", "gdn"],
            pair=[],
        )
    )

    assert [pair.as_tuple() for pair in spec.dataset_model_pairs] == [
        ("ome", "gdn"),
        ("ome", "lstm_ae"),
        ("srb", "gdn"),
        ("srb", "lstm_ae"),
    ]


def test_run_tune_parser_and_controller_support_only_explicit_pairs() -> None:
    namespace = parse_run_tune_args(["--pair", "ome:GDN", "--gpus-per-run", "0.25"])

    assert namespace.dataset == []
    assert namespace.model == []
    assert namespace.pair == ["ome:GDN"]

    spec = orchestration._build_controller_spec(
        _controller_args(dataset=[], model=[], pair=["ome:GDN", "srb:lstm_ae"])
    )

    assert [pair.as_tuple() for pair in spec.dataset_model_pairs] == [
        ("ome", "gdn"),
        ("srb", "lstm_ae"),
    ]


def test_run_tune_parser_and_controller_support_save_checkpoints() -> None:
    namespace = parse_run_tune_args(
        ["--pair", "ome:GDN", "--gpus-per-run", "0.25", "--save-checkpoints"]
    )

    assert namespace.save_checkpoints is True

    args = _controller_args(dataset=[], model=[], pair=["ome:GDN"])
    args.save_checkpoints = True
    spec = orchestration._build_controller_spec(args)

    assert spec.save_checkpoints is True


def test_run_tune_parser_supports_local_execution_defaults(tmp_path: Path) -> None:
    namespace = parse_run_tune_args(
        [
            "--pair",
            "ome:GDN",
            "--gpus-per-run",
            "0.25",
            "--execution-backend",
            "local",
            "--local-storage-path",
            str(tmp_path / "runs"),
        ]
    )

    assert namespace.execution_backend == "local"
    assert namespace.artifact_storage_backend is None
    assert namespace.optuna_storage_backend is None

    spec = orchestration._build_controller_spec(namespace)

    assert spec.execution_backend == "local"
    assert spec.artifact_storage_backend == "local"
    assert spec.optuna_storage_backend == "memory"
    assert spec.local_storage_path == str(tmp_path / "runs")


def test_controller_spec_mixes_cartesian_product_with_explicit_pairs() -> None:
    spec = orchestration._build_controller_spec(
        _controller_args(
            dataset=["ome"],
            model=["GDN", "lstm_ae"],
            pair=["srb:GDN"],
        )
    )

    assert [pair.as_tuple() for pair in spec.dataset_model_pairs] == [
        ("ome", "gdn"),
        ("ome", "lstm_ae"),
        ("srb", "gdn"),
    ]


def test_controller_spec_removes_duplicate_pairs_before_submission() -> None:
    spec = orchestration._build_controller_spec(
        _controller_args(
            dataset=["ome", "ome"],
            model=["GDN", "gdn"],
            pair=["ome:gdn", "srb:lstm_ae", "srb:LSTM_AE"],
        )
    )

    assert [pair.as_tuple() for pair in spec.dataset_model_pairs] == [
        ("ome", "gdn"),
        ("srb", "lstm_ae"),
    ]


def test_run_tune_parser_rejects_invalid_explicit_pair_syntax() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_run_tune_args(["--pair", "ome-gdn", "--gpus-per-run", "0.25"])

    assert exc_info.value.code == 2


class _FakeRuntimeConfig:
    @classmethod
    def from_env_and_args(cls, args: Namespace) -> "_FakeRuntimeConfig":
        return cls()

    def validate(self) -> None:
        return None


class _FakeControllerSettings:
    def __init__(self, storage_root: Path) -> None:
        self.mlflow_tracking_uri = "http://127.0.0.1:5000"
        self.experiment_name = "NoBoomBenchmark__20260311_120000"
        self.nooboom_mapped_storage = storage_root
        self.optuna_storage_uri = "postgresql+psycopg://noboom:example-password@127.0.0.1:5432/optuna_db"
        self.s3_endpoint_url = "http://127.0.0.1:8333"
        self.seafile_username = ""
        self.machine_nodes_b64 = None

    def validate_required(self) -> None:
        return None

    def to_pair_job_env(self) -> dict[str, str]:
        return {"EXPERIMENT_NAME": self.experiment_name}


class _LineageRuntimeConfig(_FakeRuntimeConfig):
    def __init__(self) -> None:
        self.mlflow_enable_controller_lineage = True
        self.mlflow_enable_tables = True
        self.deployment_mode = "cluster"

    @classmethod
    def from_env_and_args(cls, args: Namespace) -> "_LineageRuntimeConfig":
        del args
        return cls()


def test_run_tune_submits_pair_jobs_without_legacy_cache_bootstrap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    args = Namespace(
        dataset=["ome"],
        model=["gdn"],
        tune=True,
        experiment_id=None,
        config_dir="configs",
        temp_dir=str(tmp_path / "ray"),
        gpus_per_run=0.25,
        timestamp="20260311_120000",
        verbose=0,
        env_file=None,
        max_in_flight=1,
    )

    captured: dict[str, object] = {}

    class FakePairJobController:
        def __init__(self, **kwargs: object) -> None:
            captured["controller_kwargs"] = kwargs

        def submit_jobs(self, pair_specs: object) -> list[object]:
            gc.collect()
            captured["pair_specs"] = list(pair_specs)
            return []

    monkeypatch.setattr(orchestration, "setup_logging", lambda verbosity: None)
    monkeypatch.setattr(orchestration, "ControllerSettings", lambda: _FakeControllerSettings(tmp_path))
    monkeypatch.setattr(orchestration, "RuntimeConfig", _FakeRuntimeConfig)
    monkeypatch.setattr(orchestration, "MlflowSeafilePairOutputCallback", lambda **kwargs: object())
    monkeypatch.setattr(orchestration, "PairJobController", FakePairJobController)
    monkeypatch.setattr(orchestration, "log_code_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestration, "log_dependency_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestration, "upload_to_seafile", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestration.ray, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestration.ray, "get", lambda value: None)
    monkeypatch.setattr(
        orchestration.DependencyResolver,
        "resolve",
        lambda self: DependencyManifest(
            noboom_sha="abc123",
            timesead_sha="def456",
            timesead_extensions_sha="ghi789",
        ),
    )
    monkeypatch.setattr(orchestration.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(orchestration.mlflow, "get_experiment_by_name", lambda name: None)
    monkeypatch.setattr(orchestration.mlflow, "create_experiment", lambda name: "123")
    monkeypatch.setattr(
        orchestration.mlflow,
        "set_experiment",
        lambda name: SimpleNamespace(experiment_id="123"),
    )

    assert orchestration.run_tune(args) == 0
    assert len(captured["pair_specs"]) == 1
    assert captured["pair_specs"][0].prepared_dataset_s3_path is None


def test_run_tune_logs_controller_lineage_and_propagates_controller_run_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = Namespace(
        dataset=["ome"],
        model=["gdn"],
        tune=True,
        experiment_id=None,
        config_dir="configs",
        temp_dir=str(tmp_path / "ray"),
        gpus_per_run=0.25,
        timestamp="20260311_120000",
        verbose=0,
        env_file=None,
        max_in_flight=1,
    )

    captured: dict[str, object] = {
        "artifacts": [],
        "tables": [],
        "dicts": [],
    }

    experiment_name = "NoBoomBenchmark__20260311_120000"

    class FakeControllerSettings(_FakeControllerSettings):
        def __init__(self, storage_root: Path) -> None:
            super().__init__(storage_root)
            self.nooboom_mlflow_controller_run_id = None

        def to_pair_job_env(self) -> dict[str, str]:
            env = {"EXPERIMENT_NAME": self.experiment_name}
            if self.nooboom_mlflow_controller_run_id:
                env["NOBOOM_MLFLOW_CONTROLLER_RUN_ID"] = self.nooboom_mlflow_controller_run_id
            return env

    class FakeClient:
        def log_artifact(self, run_id: str, path: str, artifact_path: str = "") -> None:
            captured["artifacts"].append((run_id, Path(path).name, artifact_path))

        def log_dict(self, run_id: str, payload, artifact_file: str) -> None:
            captured["dicts"].append((run_id, artifact_file, payload))

        def log_table(self, run_id: str, data, artifact_file: str) -> None:
            captured["tables"].append((run_id, artifact_file, data.to_dict(orient="records")))

        def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
            captured["terminated"] = (run_id, status)

    class FakePairJobController:
        def __init__(self, **kwargs: object) -> None:
            captured["env_vars"] = kwargs["env_vars"]

        def submit_jobs(self, pair_specs: object) -> list[object]:
            excel_path = tmp_path / experiment_name / "noboom_experiments.xlsx"
            excel_path.parent.mkdir(parents=True, exist_ok=True)
            excel_path.write_text("excel", encoding="utf-8")
            return [
                PairResult(
                    model_name="gdn",
                    dataset_name="ome",
                    storage_path="/tmp/storage",
                    study_run_id="study-run-id",
                    result={"alarm_score": 0.9},
                    seafile_synced=False,
                    cleanup_performed=False,
                )
            ]

    monkeypatch.setattr(orchestration, "setup_logging", lambda verbosity: None)
    monkeypatch.setattr(orchestration, "ControllerSettings", lambda: FakeControllerSettings(tmp_path))
    monkeypatch.setattr(orchestration, "RuntimeConfig", _LineageRuntimeConfig)
    monkeypatch.setattr(orchestration, "MlflowSeafilePairOutputCallback", lambda **kwargs: object())
    monkeypatch.setattr(orchestration, "PairJobController", FakePairJobController)
    monkeypatch.setattr(
        orchestration,
        "start_controller_run",
        lambda *args, **kwargs: SimpleNamespace(run_id="controller-run-id", run_name="controller__20260311_120000"),
    )
    monkeypatch.setattr(orchestration, "log_code_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestration, "log_dependency_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestration, "upload_to_seafile", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestration, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(orchestration.ray, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestration.ray, "get", lambda value: None)
    monkeypatch.setattr(
        orchestration.DependencyResolver,
        "resolve",
        lambda self: DependencyManifest(
            noboom_sha="abc123",
            timesead_sha="def456",
            timesead_extensions_sha="ghi789",
        ),
    )
    monkeypatch.setattr(orchestration.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(orchestration.mlflow, "get_experiment_by_name", lambda name: None)
    monkeypatch.setattr(orchestration.mlflow, "create_experiment", lambda name: "123")
    monkeypatch.setattr(
        orchestration.mlflow,
        "set_experiment",
        lambda name: SimpleNamespace(experiment_id="123"),
    )

    assert orchestration.run_tune(args) == 0
    assert captured["env_vars"]["NOBOOM_MLFLOW_CONTROLLER_RUN_ID"] == "controller-run-id"
    assert captured["dicts"][0][1] == "summary/controller.json"
    assert captured["tables"][0][1] == "tables/pairs.json"
    assert captured["tables"][0][2][0]["model"] == "gdn"
    assert captured["tables"][0][2][0]["alarm_score"] == 0.9
    assert captured["tables"][0][2][0]["status"] == "SUCCEEDED"
    assert captured["dicts"][0][2]["pair_count_succeeded"] == 1
    assert ("controller-run-id", "noboom_experiments.xlsx", "reports") in captured["artifacts"]
    assert captured["terminated"] == ("controller-run-id", "FINISHED")


def test_run_tune_local_backend_skips_ray_and_uses_local_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOBOOM_MLFLOW_ENABLE_CONTROLLER_LINEAGE", "0")
    monkeypatch.setenv("OPTUNA_STORAGE_URI", "postgresql+psycopg://noboom:secret@127.0.0.1:5432/optuna_db")
    monkeypatch.setenv("NOBOOM_S3_BUCKET", "remote-bucket-from-env")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://127.0.0.1:8333")
    monkeypatch.setenv("SEAFILE_USERNAME", "user-from-env")
    args = Namespace(
        dataset=["ome"],
        model=["gdn"],
        pair=[],
        tune=True,
        experiment_id=None,
        config_dir="configs",
        temp_dir=str(tmp_path / "tmp"),
        gpus_per_run=0.25,
        timestamp="20260311_120000",
        verbose=0,
        env_file=None,
        max_in_flight=4,
        save_checkpoints=False,
        hpo_seeds=None,
        execution_backend="local",
        artifact_storage_backend=None,
        local_storage_path=str(tmp_path / "local-runs"),
        optuna_storage_backend=None,
        optuna_sqlite_path=None,
        prepared_dataset_s3_path=None,
    )
    captured: dict[str, object] = {}

    class FakeLocalPairJobController:
        def __init__(self, **kwargs: object) -> None:
            captured["controller_kwargs"] = kwargs

        def submit_jobs(self, pair_specs: object) -> list[PairResult]:
            specs = list(pair_specs)
            captured["pair_specs"] = specs
            return [
                PairResult(
                    model_name=specs[0].model_name,
                    dataset_name=specs[0].dataset_name,
                    storage_path=specs[0].storage_path,
                    result={"alarm_score": 0.9},
                )
            ]

    def fail_ray_init(*args: object, **kwargs: object) -> None:
        raise AssertionError("ray.init should not run for local backend")

    monkeypatch.setattr(orchestration, "setup_logging", lambda verbosity: None)
    monkeypatch.setattr(orchestration, "LocalPairJobController", FakeLocalPairJobController)
    monkeypatch.setattr(orchestration, "MlflowSeafilePairOutputCallback", lambda **kwargs: object())
    monkeypatch.setattr(
        orchestration,
        "start_controller_run",
        lambda *args, **kwargs: SimpleNamespace(run_id="controller-run-id", run_name="controller"),
    )
    monkeypatch.setattr(orchestration, "log_code_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestration, "log_dependency_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestration,
        "upload_to_seafile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("local backend should not upload to Seafile")
        ),
    )
    monkeypatch.setattr(orchestration.ray, "init", fail_ray_init)
    monkeypatch.setattr(
        orchestration.DependencyResolver,
        "resolve",
        lambda self: DependencyManifest(
            noboom_sha="abc123",
            timesead_sha="def456",
            timesead_extensions_sha="ghi789",
        ),
    )
    monkeypatch.setattr(orchestration.mlflow, "set_tracking_uri", lambda uri: captured.setdefault("tracking_uri", uri))
    monkeypatch.setattr(orchestration.mlflow, "get_experiment_by_name", lambda name: None)
    monkeypatch.setattr(orchestration.mlflow, "create_experiment", lambda name: "123")
    monkeypatch.setattr(
        orchestration.mlflow,
        "set_experiment",
        lambda name: SimpleNamespace(experiment_id="123"),
    )

    assert orchestration.run_tune(args) == 0
    pair_specs = captured["pair_specs"]
    assert len(pair_specs) == 1
    assert pair_specs[0].execution_backend == "local"
    assert pair_specs[0].artifact_storage_backend == "local"
    assert pair_specs[0].optuna_storage_uri == "memory://"
    assert pair_specs[0].s3_endpoint_url is None
    assert str(pair_specs[0].storage_path).endswith("/local-runs/NoBoomBenchmark__20260311_120000/local/ome__gdn")
    assert captured["controller_kwargs"]["env_vars"]["NOBOOM_S3_BUCKET"] == ""
    assert captured["controller_kwargs"]["env_vars"]["S3_ENDPOINT_URL"] == ""
    assert captured["controller_kwargs"]["env_vars"]["SEAFILE_USERNAME"] == ""
    assert captured["controller_kwargs"]["env_vars"]["NOBOOM_SEAFILE_UPLOAD_RESULTS"] == "0"
