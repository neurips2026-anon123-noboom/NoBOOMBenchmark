from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
from ray.job_submission import JobStatus

from noboom_cluster.noboom_cli_lib.specs import DependencyManifest, PairResult, PairRunSpec

from noboom_benchmark.noboom_lib.core.tune.interfaces import PairOutputCallback
from noboom_benchmark.noboom_lib.core.tune.job_submission import (
    LocalPairJobController,
    PairJobController,
    build_pair_run_spec,
)


class RecordingCallback(PairOutputCallback):
    def __init__(self, fail_on: Optional[str] = None) -> None:
        self.calls: List[tuple[str, str, Optional[str]]] = []
        self.fail_on = fail_on

    def on_job_terminal(
        self,
        pair_spec: PairRunSpec,
        status: str,
        job_status_message: Optional[str] = None,
    ) -> PairResult:
        self.calls.append((pair_spec.pair_id, status, job_status_message))
        if self.fail_on == pair_spec.pair_id:
            raise RuntimeError("callback failure")
        return PairResult(
            model_name=pair_spec.model_name,
            dataset_name=pair_spec.dataset_name,
            storage_path=pair_spec.storage_path,
            result={"score": 1.0},
            status=status,
            partial_result=status != "SUCCEEDED",
            job_status_message=job_status_message,
        )


class FakeJobClient:
    def __init__(self) -> None:
        self.submitted: List[dict] = []
        self._jobs: dict[str, SimpleNamespace] = {}

    def submit_job(self, entrypoint: str, runtime_env: dict, submission_id: str) -> None:
        self.submitted.append(
            {
                "entrypoint": entrypoint,
                "runtime_env": runtime_env,
                "submission_id": submission_id,
            }
        )
        self._jobs[submission_id] = SimpleNamespace(
            submission_id=submission_id,
            status=JobStatus.SUCCEEDED,
            message=None,
        )

    def list_jobs(self) -> List[SimpleNamespace]:
        return list(self._jobs.values())


def _build_pair_spec(dataset_name: str, model_name: str) -> PairRunSpec:
    return PairRunSpec(
        experiment_id="1",
        source_experiment_id=None,
        model_name=model_name,
        dataset_name=dataset_name,
        timestamp="20260311_120000",
        storage_path=f"/tmp/storage/ray/{dataset_name}__{model_name}",
        gpus_per_run=0.25,
        optuna_storage_uri="postgresql+psycopg://noboom:example-password@127.0.0.1:5432/optuna_db",
        config_dir="configs",
        temp_dir="/tmp/ray",
        verbose=0,
        tune=True,
        env_file=None,
        tracking_uri="http://127.0.0.1:5000",
        s3_endpoint_url="http://127.0.0.1:8333",
    )


def test_pair_job_controller_submits_one_job_per_pair(tmp_path: Path) -> None:
    callback = RecordingCallback()
    client = FakeJobClient()
    controller = PairJobController(
        dependency_manifest=DependencyManifest(
            noboom_sha="abc123",
            timesead_sha="def456",
            timesead_extensions_sha="ghi789",
        ),
        pair_specs_dir=tmp_path,
        callback=callback,
        env_vars={"EXPERIMENT_NAME": "NoBoomBenchmark__20260311_120000"},
        max_in_flight=2,
        client=client,
    )

    pair_specs = [_build_pair_spec("ome", "gdn"), _build_pair_spec("srb", "lstm_ae")]
    results = controller.submit_jobs(pair_specs)

    assert [entry["submission_id"] for entry in client.submitted] == [
        "ome__gdn__20260311_120000",
        "srb__lstm_ae__20260311_120000",
    ]
    assert all(
        entry["entrypoint"].startswith("python -m noboom_benchmark.tuning_workflow --pair-spec-b64 ")
        for entry in client.submitted
    )
    assert callback.calls == [
        ("ome__gdn", "SUCCEEDED", None),
        ("srb__lstm_ae", "SUCCEEDED", None),
    ]
    assert [result.model_name for result in results] == ["gdn", "lstm_ae"]
    assert ".noboom_runtime/pair_specs" not in client.submitted[0]["entrypoint"]


def test_pair_job_controller_raises_when_callback_fails(tmp_path: Path) -> None:
    callback = RecordingCallback(fail_on="ome__gdn")
    client = FakeJobClient()
    controller = PairJobController(
        dependency_manifest=DependencyManifest(
            noboom_sha="abc123",
            timesead_sha="def456",
            timesead_extensions_sha="ghi789",
        ),
        pair_specs_dir=tmp_path,
        callback=callback,
        env_vars={"EXPERIMENT_NAME": "NoBoomBenchmark__20260311_120000"},
        max_in_flight=1,
        client=client,
    )

    with pytest.raises(RuntimeError, match="pair output callbacks failed"):
        controller.submit_jobs([_build_pair_spec("ome", "gdn")])


def test_pair_job_controller_returns_stopped_pair_result(tmp_path: Path) -> None:
    callback = RecordingCallback()
    client = FakeJobClient()
    controller = PairJobController(
        dependency_manifest=DependencyManifest(
            noboom_sha="abc123",
            timesead_sha="def456",
            timesead_extensions_sha="ghi789",
        ),
        pair_specs_dir=tmp_path,
        callback=callback,
        env_vars={"EXPERIMENT_NAME": "NoBoomBenchmark__20260311_120000"},
        max_in_flight=1,
        client=client,
    )

    pair_spec = _build_pair_spec("ome", "gdn")

    def submit_stopped_job(entrypoint: str, runtime_env: dict, submission_id: str) -> None:
        del entrypoint, runtime_env
        client.submitted.append({"submission_id": submission_id})
        client._jobs[submission_id] = SimpleNamespace(
            submission_id=submission_id,
            status=JobStatus.STOPPED,
            message="Stopped by user",
        )

    client.submit_job = submit_stopped_job  # type: ignore[method-assign]

    results = controller.submit_jobs([pair_spec])

    assert callback.calls == [("ome__gdn", "STOPPED", "Stopped by user")]
    assert len(results) == 1
    assert results[0].status == "STOPPED"
    assert results[0].partial_result is True
    assert results[0].job_status_message == "Stopped by user"


def test_pair_job_controller_records_failed_terminal_job_on_head_excel_path(tmp_path: Path) -> None:
    callback = RecordingCallback()
    client = FakeJobClient()
    controller = PairJobController(
        dependency_manifest=DependencyManifest(
            noboom_sha="abc123",
            timesead_sha="def456",
            timesead_extensions_sha="ghi789",
        ),
        pair_specs_dir=tmp_path,
        callback=callback,
        env_vars={"EXPERIMENT_NAME": "NoBoomBenchmark__20260311_120000"},
        max_in_flight=1,
        client=client,
    )

    pair_spec = _build_pair_spec("ome", "gdn")

    def submit_failed_job(entrypoint: str, runtime_env: dict, submission_id: str) -> None:
        del entrypoint, runtime_env
        client.submitted.append({"submission_id": submission_id})
        client._jobs[submission_id] = SimpleNamespace(
            submission_id=submission_id,
            status=JobStatus.FAILED,
            message="worker failed",
        )

    client.submit_job = submit_failed_job  # type: ignore[method-assign]

    results = controller.submit_jobs([pair_spec])

    assert callback.calls == [("ome__gdn", "FAILED", "worker failed")]
    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert results[0].partial_result is True
    assert results[0].job_status_message == "worker failed"


def test_local_pair_job_controller_runs_pairs_sequentially(tmp_path: Path) -> None:
    callback = RecordingCallback()
    calls: List[str] = []

    def fake_runner(pair_spec: PairRunSpec) -> tuple[str, str, dict[str, float]]:
        calls.append(pair_spec.pair_id)
        return pair_spec.model_name, pair_spec.dataset_name, {"score": 1.0}

    controller = LocalPairJobController(
        dependency_manifest=DependencyManifest(
            noboom_sha="abc123",
            timesead_sha="def456",
            timesead_extensions_sha="ghi789",
        ),
        pair_specs_dir=tmp_path,
        callback=callback,
        env_vars={"EXPERIMENT_NAME": "NoBoomBenchmark__20260311_120000"},
        runner=fake_runner,
    )

    pair_specs = [
        _build_pair_spec("ome", "gdn").model_copy(update={"execution_backend": "local"}),
        _build_pair_spec("srb", "lstm_ae").model_copy(update={"execution_backend": "local"}),
    ]

    results = controller.submit_jobs(pair_specs)

    assert calls == ["ome__gdn", "srb__lstm_ae"]
    assert callback.calls == [
        ("ome__gdn", "SUCCEEDED", None),
        ("srb__lstm_ae", "SUCCEEDED", None),
    ]
    assert [result.status for result in results] == ["SUCCEEDED", "SUCCEEDED"]


def test_local_pair_job_controller_reports_failed_pair_and_continues(tmp_path: Path) -> None:
    callback = RecordingCallback()
    calls: List[str] = []

    def fake_runner(pair_spec: PairRunSpec) -> tuple[str, str, dict[str, float]]:
        calls.append(pair_spec.pair_id)
        if pair_spec.pair_id == "ome__gdn":
            raise RuntimeError("local failure")
        return pair_spec.model_name, pair_spec.dataset_name, {"score": 1.0}

    controller = LocalPairJobController(
        dependency_manifest=DependencyManifest(
            noboom_sha="abc123",
            timesead_sha="def456",
            timesead_extensions_sha="ghi789",
        ),
        pair_specs_dir=tmp_path,
        callback=callback,
        env_vars={"EXPERIMENT_NAME": "NoBoomBenchmark__20260311_120000"},
        runner=fake_runner,
    )

    results = controller.submit_jobs(
        [
            _build_pair_spec("ome", "gdn").model_copy(update={"execution_backend": "local"}),
            _build_pair_spec("srb", "lstm_ae").model_copy(update={"execution_backend": "local"}),
        ]
    )

    assert calls == ["ome__gdn", "srb__lstm_ae"]
    assert [result.status for result in results] == ["FAILED", "SUCCEEDED"]
    assert results[0].partial_result is True
    assert results[0].job_status_message == "local failure"


def test_build_pair_run_spec_keeps_local_storage_path_clean(tmp_path: Path) -> None:
    spec = build_pair_run_spec(
        experiment_id="1",
        source_experiment_id=None,
        model_name="gdn",
        dataset_name="ome",
        timestamp="20260311_120000",
        local_storage_path=tmp_path / "storage",
        optuna_storage_uri="postgresql+psycopg://noboom:example-password@127.0.0.1:5432/optuna_db",
        tracking_uri="http://127.0.0.1:5000",
        config_dir="configs",
        temp_dir="/tmp/ray",
        data_manifest_path=None,
        verbose=0,
        tune_mode=True,
        gpus_per_run=0.25,
        s3_endpoint_url="http://127.0.0.1:8333",
        prepared_dataset_s3_path="s3://prepared-bucket/prepared-root",
        env_file=None,
        hpo_seeds=[42, 44],
    )

    assert spec.storage_path == str(tmp_path / "storage" / "ray" / "ome__gdn")
    assert spec.s3_endpoint_url == "http://127.0.0.1:8333"
    assert spec.hpo_seeds == [42, 44]


def test_build_pair_run_spec_uses_local_storage_dir_for_local_backend(tmp_path: Path) -> None:
    spec = build_pair_run_spec(
        experiment_id="1",
        source_experiment_id=None,
        model_name="gdn",
        dataset_name="ome",
        timestamp="20260311_120000",
        local_storage_path=tmp_path / "storage",
        optuna_storage_uri="memory://",
        tracking_uri=f"file://{tmp_path / 'mlruns'}",
        config_dir="configs",
        temp_dir="/tmp/ray",
        data_manifest_path=None,
        verbose=0,
        tune_mode=True,
        gpus_per_run=0.25,
        s3_endpoint_url=None,
        prepared_dataset_s3_path=None,
        env_file=None,
        execution_backend="local",
        artifact_storage_backend="local",
        optuna_storage_backend="memory",
    )

    assert spec.storage_path == str(tmp_path / "storage" / "local" / "ome__gdn")
    assert spec.execution_backend == "local"
    assert spec.artifact_storage_backend == "local"
    assert spec.optuna_storage_backend == "memory"
    assert spec.s3_endpoint_url is None
