from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from noboom_benchmark.noboom_lib.core.tune import mlflow_utils


def test_load_study_result_from_mlflow_skips_newest_run_without_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mlflow_utils,
        "MlflowClient",
        lambda tracking_uri=None: SimpleNamespace(search_runs=lambda *args, **kwargs: []),
    )
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "search_runs",
        lambda **_kwargs: pd.DataFrame([{"run_id": "newest-run"}, {"run_id": "older-run"}]),
    )

    def fake_download_artifacts(*, run_id: str, artifact_path: str, dst_path: str) -> str:
        del artifact_path
        if run_id == "newest-run":
            raise FileNotFoundError("artifact not uploaded yet")

        local_path = Path(dst_path) / f"{run_id}.json"
        local_path.write_text(
            json.dumps({"hp_params": {"window_size": 64, "lr": 1e-3}}),
            encoding="utf-8",
        )
        return str(local_path)

    monkeypatch.setattr(mlflow_utils.mlflow.artifacts, "download_artifacts", fake_download_artifacts)

    result = mlflow_utils.load_study_result_from_mlflow(
        "exp-1",
        "neutralad",
        "cont_reactive_ome",
    )

    assert result == ({"hp_params": {"window_size": 64, "lr": 1e-3}}, "older-run")


def test_load_study_result_from_mlflow_falls_back_to_best_eligible_trial(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "search_runs",
        lambda **_kwargs: pd.DataFrame([{"run_id": "study-run-1"}]),
    )
    download_calls = []

    def fake_download_json_artifact_for_run(run_id: str, *, artifact_path: str):
        download_calls.append((run_id, artifact_path))
        if run_id == "study-run-1":
            return None
        if run_id == "trial-fast":
            return {"hp_params": {"window_size": 77, "lr": 1e-4}}
        return None

    class FakeClient:
        def __init__(self, tracking_uri=None) -> None:
            del tracking_uri

        def search_runs(self, experiment_ids, filter_string: str, max_results: int = 1000):
            del experiment_ids, filter_string, max_results
            return [
                SimpleNamespace(
                    info=SimpleNamespace(run_id="trial-insufficient"),
                    data=SimpleNamespace(
                        metrics={
                            "completed_seed_count": 1.0,
                            "alarm_score": 0.99,
                            "aaf": 0.01,
                            "time_total_s": 1.0,
                        }
                    ),
                ),
                SimpleNamespace(
                    info=SimpleNamespace(run_id="trial-slower"),
                    data=SimpleNamespace(
                        metrics={
                            "completed_seed_count": 2.0,
                            "alarm_score": 0.95,
                            "aaf": 0.2,
                            "time_total_s": 12.0,
                        }
                    ),
                ),
                SimpleNamespace(
                    info=SimpleNamespace(run_id="trial-worse-aaf"),
                    data=SimpleNamespace(
                        metrics={
                            "completed_seed_count": 2.0,
                            "alarm_score": 0.95,
                            "aaf": 0.3,
                            "time_total_s": 5.0,
                        }
                    ),
                ),
                SimpleNamespace(
                    info=SimpleNamespace(run_id="trial-fast"),
                    data=SimpleNamespace(
                        metrics={
                            "completed_seed_count": 2.0,
                            "alarm_score": 0.95,
                            "aaf": 0.2,
                            "time_total_s": 9.0,
                        }
                    ),
                ),
            ]

    monkeypatch.setattr(mlflow_utils, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        mlflow_utils,
        "_download_json_artifact_for_run",
        fake_download_json_artifact_for_run,
    )

    result = mlflow_utils.load_study_result_from_mlflow(
        "exp-1",
        "neutralad",
        "cont_reactive_ome",
    )

    assert result == ({"hp_params": {"window_size": 77, "lr": 1e-4}}, "study-run-1")
    assert download_calls == [
        ("study-run-1", "summary/result.json"),
        ("trial-fast", "metadata.json"),
    ]
