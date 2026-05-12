from argparse import Namespace
from contextlib import nullcontext
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest

from noboom_benchmark.noboom_lib.core import model_objectives
from noboom_benchmark.noboom_lib.core.tune.pruning import SeedAwareMetricReporter
from noboom_benchmark.noboom_lib.core.tune_constants import TUNE_PROGRESS_ATTR


CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_cluster"
    / "cluster_files"
    / "configs"
)


def test_run_pytorch_objective_rejects_more_than_one_gpu_per_trial() -> None:
    args = Namespace(
        config_dir=str(CONFIG_DIR),
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        gpus_per_run=2,
    )

    with pytest.raises(ValueError, match="single GPU per trial"):
        model_objectives.tune_resource_request_dict(args)


def test_tune_resource_request_dict_reserves_full_gpu_for_neutralad_on_industry_process() -> None:
    args = Namespace(
        config_dir=str(CONFIG_DIR),
        model_name="neutralad",
        dataset_name="industry_process",
        gpus_per_run=0.25,
    )

    resources = model_objectives.tune_resource_request_dict(args)

    assert resources["GPU"] == pytest.approx(1.0)


def test_tune_resource_request_dict_reserves_half_gpu_for_physdiff() -> None:
    args = Namespace(
        config_dir=str(CONFIG_DIR),
        model_name="physdiff",
        dataset_name="batch_dist_ternary_acetone_1_butanol_methanol",
        gpus_per_run=0.33,
    )

    resources = model_objectives.tune_resource_request_dict(args)

    assert resources["GPU"] == pytest.approx(0.5)
    assert resources["exclusive"] == pytest.approx(0.001)


@pytest.mark.parametrize("model_name", ["hmm", "gmmhmm"])
def test_tune_resource_request_dict_reserves_full_exclusive_for_industry_hmm_models(
    model_name: str,
) -> None:
    args = Namespace(
        config_dir=str(CONFIG_DIR),
        model_name=model_name,
        dataset_name="industry_process",
        gpus_per_run=0.33,
    )

    resources = model_objectives.tune_resource_request_dict(args)

    assert resources["exclusive"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("dataset_name", "expected_memory_gib"),
    [
        ("cont_reactive_ome", 16),
        ("batch_dist_ternary_acetone_1_butanol_methanol", 50),
        ("industry_process", 64),
    ],
)
def test_tune_resource_request_dict_uses_dataset_trial_memory_gib(
    dataset_name: str,
    expected_memory_gib: int,
) -> None:
    args = Namespace(
        config_dir=str(CONFIG_DIR),
        model_name="neutralad",
        dataset_name=dataset_name,
        gpus_per_run=0.25,
        study_params={
            "trial_memory_gib_by_dataset": {
                "cont_reactive_ome": 16,
                "batch_dist_ternary_acetone_1_butanol_methanol": 50,
                "industry_process": 64,
            },
        },
    )

    resources = model_objectives.tune_resource_request_dict(args)

    assert resources["memory"] == expected_memory_gib * 1024**3


def test_tune_resource_request_dict_uses_base_dataset_for_tsst_memory() -> None:
    args = Namespace(
        config_dir=str(CONFIG_DIR),
        model_name="neutralad",
        requested_dataset_name="cont_reactive_ome_tsst",
        dataset_name="cont_reactive_ome",
        gpus_per_run=0.25,
        study_params={"trial_memory_gib_by_dataset": {"cont_reactive_ome": 16}},
    )

    resources = model_objectives.tune_resource_request_dict(args)

    assert resources["memory"] == 16 * 1024**3


def test_tune_resource_request_dict_omits_memory_without_dataset_setting() -> None:
    args = Namespace(
        config_dir=str(CONFIG_DIR),
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        gpus_per_run=0.25,
    )

    resources = model_objectives.tune_resource_request_dict(args)

    assert "memory" not in resources


def test_log_seed_artifacts_skips_checkpoints_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "seed_42"
    seed_dir.mkdir()
    (seed_dir / "last.ckpt").write_text("checkpoint", encoding="utf-8")
    (seed_dir / "metrics.json").write_text("{}", encoding="utf-8")
    logged_artifacts = []
    logged_files = []

    monkeypatch.setattr(
        model_objectives.mlflow,
        "log_artifacts",
        lambda *args, **kwargs: logged_artifacts.append((args, kwargs)),
    )
    monkeypatch.setattr(
        model_objectives.mlflow,
        "log_artifact",
        lambda *args, **kwargs: logged_files.append((args, kwargs)),
    )

    model_objectives._log_seed_artifacts_to_trial_run(
        trial_run_id="trial-run-id",
        seed=42,
        seed_ckpt_dir=seed_dir,
    )

    assert logged_artifacts == []
    assert logged_files == [
        (
            (str(seed_dir / "metrics.json"),),
            {"artifact_path": "artifacts/ckpts/seed_42", "run_id": "trial-run-id"},
        )
    ]


def test_log_seed_artifacts_includes_checkpoints_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "seed_42"
    seed_dir.mkdir()
    (seed_dir / "last.ckpt").write_text("checkpoint", encoding="utf-8")
    logged_artifacts = []
    logged_files = []

    monkeypatch.setattr(
        model_objectives.mlflow,
        "log_artifacts",
        lambda *args, **kwargs: logged_artifacts.append((args, kwargs)),
    )
    monkeypatch.setattr(
        model_objectives.mlflow,
        "log_artifact",
        lambda *args, **kwargs: logged_files.append((args, kwargs)),
    )

    model_objectives._log_seed_artifacts_to_trial_run(
        trial_run_id="trial-run-id",
        seed=42,
        seed_ckpt_dir=seed_dir,
        include_checkpoints=True,
    )

    assert logged_artifacts == [
        (
            (str(seed_dir),),
            {"artifact_path": "artifacts/ckpts/seed_42", "run_id": "trial-run-id"},
        )
    ]
    assert logged_files == []


def test_run_seed_task_logs_scalar_metrics_to_seed_mlflow_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logged_metrics = []
    terminated_runs = []
    saved_artifacts = []
    uploaded_artifacts = []

    class FakeMlflowClient:
        def __init__(self, tracking_uri: str) -> None:
            assert tracking_uri == "file:///tmp/mlruns"

        def set_terminated(self, run_id: str) -> None:
            terminated_runs.append(run_id)

    def fake_run_seed(*_args, **_kwargs):
        return (
            {
                "alarm_score": (np.float64(1.25), np.float64(5.0)),
                "aaf": (0.5, 1.0),
            },
            np.array([0.1, 0.2]),
            np.array([0, 1]),
            {"threshold": np.float64(0.4), "feature": "P101"},
            "seed-run-id",
            None,
        )

    monkeypatch.setattr(model_objectives, "MlflowClient", FakeMlflowClient)
    monkeypatch.setattr(model_objectives, "run_seed", fake_run_seed)
    monkeypatch.setattr(model_objectives.mlflow, "set_tracking_uri", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model_objectives.mlflow,
        "log_metrics",
        lambda metrics, run_id: logged_metrics.append((dict(metrics), run_id)),
    )
    monkeypatch.setattr(
        model_objectives,
        "save_seed_artifacts",
        lambda *args, **kwargs: saved_artifacts.append((args, kwargs)),
    )
    monkeypatch.setattr(
        model_objectives,
        "_log_seed_artifacts_to_trial_run",
        lambda **kwargs: uploaded_artifacts.append(kwargs),
    )

    args = Namespace(
        tracking_uri="file:///tmp/mlruns",
        save_checkpoints=False,
    )

    result = model_objectives.run_seed_task._function(  # noqa: SLF001
        {},
        args,
        42,
        str(tmp_path / "trial"),
        "trial-run-id",
        "trial-name",
    )

    assert result == {
        "alarm_score": (np.float64(1.25), np.float64(5.0)),
        "aaf": (0.5, 1.0),
    }
    assert logged_metrics == [
        (
            {
                "alarm_score": 1.25,
                "theory_best_alarm_score": 5.0,
                "aaf": 0.5,
                "theory_best_aaf": 1.0,
                "threshold": 0.4,
            },
            "seed-run-id",
        )
    ]
    assert terminated_runs == ["seed-run-id"]
    assert saved_artifacts[0][0][0] == tmp_path / "trial" / "artifacts" / "ckpts" / "seed_42"
    assert uploaded_artifacts == [
        {
            "trial_run_id": "trial-run-id",
            "seed": 42,
            "seed_ckpt_dir": tmp_path / "trial" / "artifacts" / "ckpts" / "seed_42",
            "include_checkpoints": False,
        }
    ]


def test_predict_with_cli_clears_compiled_wrapper_before_checkpoint_restore(
    tmp_path: Path,
) -> None:
    predict_calls: list[dict[str, object]] = []
    ckpt_path = tmp_path / "last.ckpt"
    ckpt_path.write_text("checkpoint", encoding="utf-8")

    class FakeTrainer:
        def predict(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            del datamodule
            predict_calls.append(
                {
                    "ckpt_path": ckpt_path,
                    "compiled_network": getattr(model, "_compiled_network", "missing"),
                    "compile_invocation_count": getattr(model, "_compile_invocation_count", "missing"),
                }
            )

    cli = SimpleNamespace(
        trainer=FakeTrainer(),
        datamodule=SimpleNamespace(scaler=None),
        model=SimpleNamespace(
            _compiled_network=object(),
            _compile_invocation_count=3,
            style_transfer=None,
            use_test_style_transfer=False,
            metrics={"alarm_score": 0.9},
            anomaly_scores=np.array([1.0]),
            anomaly_labels=np.array([0]),
        ),
    )

    result = model_objectives._predict_with_cli(
        cli=cli,
        args=Namespace(model_name="neutralad", mlflow_enable_dataset_tracking=False),
        stage="real",
        seed=7,
        seed_run_id="seed-run-id",
        seed_ckpt_dir=tmp_path,
        predict_ckpt_path=str(ckpt_path),
        dataset_lineage_cache=None,
    )

    assert predict_calls == [
        {
            "ckpt_path": str(ckpt_path),
            "compiled_network": None,
            "compile_invocation_count": 0,
        }
    ]
    assert result.metrics == {"alarm_score": 0.9}


def test_predict_with_cli_creates_seed_dir_before_saving_scaler(tmp_path: Path) -> None:
    seed_dir = tmp_path / "missing" / "seed_7"
    scaler = {"drop_features": ["A"]}

    class FakeTrainer:
        def predict(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            del model, datamodule, ckpt_path

    cli = SimpleNamespace(
        trainer=FakeTrainer(),
        datamodule=SimpleNamespace(scaler=scaler),
        model=SimpleNamespace(
            style_transfer=None,
            use_test_style_transfer=False,
            metrics={"alarm_score": 0.9},
            anomaly_scores=np.array([1.0]),
            anomaly_labels=np.array([0]),
        ),
    )

    result = model_objectives._predict_with_cli(
        cli=cli,
        args=Namespace(model_name="dcdetector", mlflow_enable_dataset_tracking=False),
        stage="real",
        seed=7,
        seed_run_id="seed-run-id",
        seed_ckpt_dir=seed_dir,
        predict_ckpt_path=None,
        dataset_lineage_cache=None,
    )

    assert (seed_dir / "scaler.pkl").is_file()
    assert result.metrics == {"alarm_score": 0.9}


def test_seed_aware_metric_reporter_builds_completed_payload() -> None:
    reporter = SeedAwareMetricReporter(
        tune_metric="alarm_score",
    )
    reporter.note_completed_seed(0.6)
    reporter.note_completed_seed(0.4)

    payload = reporter.build_completed_payload(
        metric_history={
            "alarm_score": [0.6, 0.4],
            "aaf": [0.2, 0.1],
            "edf": [0.2, 0.8],
            "ldf": [0.4, 0.1],
        },
        best_metric_history={
            "alarm_score": [0.9, 0.9],
            "aaf": [0.2, 0.2],
            "edf": [1.0, 1.0],
            "ldf": [0.0, 0.0],
        },
        trial_run_id="trial-run-id",
        seed=45,
    )

    assert payload["alarm_score"] == pytest.approx(0.5)
    assert payload["aaf"] == pytest.approx(0.15)
    assert payload["edf"] == pytest.approx(0.5)
    assert payload["ldf"] == pytest.approx(0.25)
    assert payload["mean_aaf"] == pytest.approx(0.15)
    assert payload["mean_edf"] == pytest.approx(0.5)
    assert payload["mean_ldf"] == pytest.approx(0.25)
    assert "mean_alarm_score" not in payload
    assert payload["best_alarm_score"] == pytest.approx(0.6)
    assert payload["best_aaf"] == pytest.approx(0.1)
    assert payload["best_edf"] == pytest.approx(0.8)
    assert payload["best_ldf"] == pytest.approx(0.4)
    assert payload["theory_best_alarm_score"] == pytest.approx(0.9)
    assert payload["theory_best_aaf"] == pytest.approx(0.2)
    assert payload["theory_best_edf"] == pytest.approx(1.0)
    assert payload["theory_best_ldf"] == pytest.approx(0.0)
    assert payload["completed_seed_count"] == 2
    assert payload["seed"] == 45
    assert payload["mlflow_run_id"] == "trial-run-id"
    assert payload[TUNE_PROGRESS_ATTR] == 2


def _run_tune_trainable_seed_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    study_params: dict[str, Any],
) -> list[int]:
    observed_seeds: list[int] = []

    class FakeTuneContext:
        def get_trial_name(self) -> str:
            return "neutralad__cont_reactive_ome__trial_0001"

        def get_trial_dir(self) -> str:
            trial_dir = tmp_path / "trial"
            trial_dir.mkdir()
            return str(trial_dir)

        def get_trial_id(self) -> str:
            return "trial-id"

    class FakeClient:
        def search_runs(self, *args, **kwargs):
            del args, kwargs
            return [SimpleNamespace(info=SimpleNamespace(run_id="trial-run-id"))]

        def log_batch(self, *args, **kwargs) -> None:
            del args, kwargs

        def set_terminated(self, *args, **kwargs) -> None:
            del args, kwargs

        def set_tag(self, *args, **kwargs) -> None:
            del args, kwargs

    def fake_run_seed(
        hp_params,
        args,
        seed_ckpt_dir,
        seed: int,
        trial_run_id: Optional[str],
        trial_name: Optional[str],
        trial_metric_reporter=None,
    ):
        del hp_params, args, seed_ckpt_dir, trial_run_id, trial_name, trial_metric_reporter
        observed_seeds.append(seed)
        return (
            {"alarm_score": (float(seed), 100.0)},
            np.array([float(seed)]),
            np.array([0]),
            {},
            f"seed-run-{seed}",
            None,
        )

    def raise_missing_allowlist_actor():
        raise ValueError("runtime GPU allowlist actor unavailable")

    monkeypatch.setattr(model_objectives.tune, "get_context", lambda: FakeTuneContext())
    monkeypatch.setattr(model_objectives.tune, "report", lambda payload: None)
    monkeypatch.setattr(model_objectives, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(model_objectives, "get_runtime_gpu_allowlist_actor", raise_missing_allowlist_actor)
    monkeypatch.setattr(model_objectives, "validate_model_hparams", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_objectives,
        "prepare_model_dataset_manifests",
        lambda **kwargs: {"real": str(tmp_path / "manifest.json")},
    )
    monkeypatch.setattr(model_objectives, "run_seed", fake_run_seed)
    monkeypatch.setattr(model_objectives.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_objectives, "save_seed_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_objectives, "_log_seed_artifacts_to_trial_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_objectives, "save_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_objectives, "_log_trial_metadata_to_mlflow", lambda *args, **kwargs: None)

    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        timestamp="20260311_000000",
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        study_run_id="study-run-id",
        config_dir=str(CONFIG_DIR),
        runtime_config=SimpleNamespace(),
        use_generated=False,
        data_manifest_path_explicit=False,
        verbose=0,
        tune=True,
        deployment_mode="native",
        study_params=study_params,
        mlflow_enable_system_metrics=False,
        save_checkpoints=False,
        temp_dir=str(tmp_path),
    )

    model_objectives.run_tune_trainable({"window_size": 128}, args=args)
    return observed_seeds


def test_run_tune_trainable_uses_hpo_seeds_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_seeds = _run_tune_trainable_seed_probe(
        monkeypatch,
        tmp_path,
        {
            "tune_metric": "alarm_score",
            "seeds": [42, 43, 44, 45, 46],
            "hpo_seeds": [43, 45],
        },
    )

    assert observed_seeds == [43, 45]


def test_run_tune_trainable_falls_back_to_study_seeds_without_hpo_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_seeds = _run_tune_trainable_seed_probe(
        monkeypatch,
        tmp_path,
        {
            "tune_metric": "alarm_score",
            "seeds": [42, 43, 44],
        },
    )

    assert observed_seeds == [42, 43, 44]


def test_run_tune_trainable_rejects_empty_hpo_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="hpo_seeds"):
        _run_tune_trainable_seed_probe(
            monkeypatch,
            tmp_path,
            {
                "tune_metric": "alarm_score",
                "seeds": [42, 43, 44],
                "hpo_seeds": [],
            },
        )


def test_run_tune_trainable_reports_seed_context_and_stops_after_prune(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    reported_payloads = []
    terminated_runs = []
    run_seed_calls = []
    received_hparams = []
    prepared_manifest_calls = []

    class FakeContext:
        def get_trial_name(self) -> str:
            return "neutralad__cont_reactive_ome__20260311_000000__trial_0001"

        def get_trial_dir(self) -> str:
            return str(trial_dir)

    class FakeClient:
        def search_runs(self, *_args, **_kwargs):
            return [SimpleNamespace(info=SimpleNamespace(run_id="trial-run-id"))]

        def log_batch(self, *_args, **_kwargs) -> None:
            return None

        def set_terminated(
            self,
            run_id: str,
            status: Optional[str] = None,
            end_time: Optional[int] = None,
        ) -> None:
            terminated_runs.append((run_id, status))

    def fake_run_seed(
        hp_params,
        args,
        seed_ckpt_dir,
        seed,
        trial_run_id,
        trial_name,
        trial_metric_reporter=None,
    ):
        received_hparams.append(dict(hp_params))
        run_seed_calls.append(seed)
        return (
            {
                "alarm_score": (0.75, 0.9),
                "aaf": (0.2, 0.2),
            },
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            {},
            f"seed-run-{seed}",
        )

    def fake_report(payload):
        reported_payloads.append(payload)
        raise SystemExit()

    monkeypatch.setattr(model_objectives.tune, "get_context", lambda: FakeContext())
    monkeypatch.setattr(model_objectives.tune, "report", fake_report)
    monkeypatch.setattr(model_objectives, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(model_objectives, "run_seed", fake_run_seed)
    monkeypatch.setattr(model_objectives, "save_seed_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_objectives, "_log_seed_artifacts_to_trial_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_objectives, "save_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_objectives.mlflow, "log_metrics", lambda *args, **kwargs: None)

    def fake_prepare_model_dataset_manifests(**kwargs):
        prepared_manifest_calls.append(kwargs)
        return {"real": "/prepared/real/manifest.json"}

    monkeypatch.setattr(
        model_objectives,
        "prepare_model_dataset_manifests",
        fake_prepare_model_dataset_manifests,
    )

    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        config_dir=str(CONFIG_DIR),
        runtime_config=SimpleNamespace(),
        use_generated=False,
        verbose=0,
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        study_run_id="study-run-id",
        timestamp="20260311_000000",
        study_params={
            "tune_metric": "alarm_score",
            "seeds": [42, 43],
        },
    )

    result = model_objectives.run_tune_trainable({"lr": 1e-3, "window_size": 38}, args)

    assert run_seed_calls == [42]
    assert received_hparams == [{"lr": 1e-3, "window_size": 38}]
    assert prepared_manifest_calls[0]["hp_params"] == {"lr": 1e-3, "window_size": 38}
    assert args.prepared_manifest_paths == {"real": "/prepared/real/manifest.json"}
    assert args.data_manifest_path == "/prepared/real/manifest.json"
    assert terminated_runs == [("seed-run-42", None)]
    assert len(reported_payloads) == 1
    assert reported_payloads[0]["seed"] == 42
    assert reported_payloads[0]["completed_seed_count"] == 1
    assert reported_payloads[0][TUNE_PROGRESS_ATTR] == 1
    assert reported_payloads[0]["alarm_score"] == pytest.approx(0.75)
    assert result == reported_payloads[0]


def test_run_tune_trainable_short_circuits_invalid_lnt_window_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {
        "tags": [],
        "saved_metadata": [],
    }
    run_seed_calls: list[dict[str, object]] = []

    class FakeContext:
        def get_trial_name(self) -> str:
            return "lnt__industry_process__20260318_000000__trial_deadbeef"

        def get_trial_dir(self) -> str:
            return str(tmp_path / "trial")

    class FakeClient:
        def search_runs(self, *_args, **_kwargs):
            return [SimpleNamespace(info=SimpleNamespace(run_id="trial-run-id"))]

        def log_batch(self, *_args, **_kwargs) -> None:
            return None

        def set_tag(self, run_id: str, key: str, value: str) -> None:
            captured["tags"].append((run_id, key, value))

    def fake_run_seed(*args, **kwargs):
        run_seed_calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("run_seed should not be called for invalid LNT configs")

    monkeypatch.setattr(model_objectives.tune, "get_context", lambda: FakeContext())
    monkeypatch.setattr(model_objectives, "MlflowClient", lambda *_args, **_kwargs: FakeClient())
    monkeypatch.setattr(model_objectives, "run_seed", fake_run_seed)
    monkeypatch.setattr(
        model_objectives,
        "save_metadata",
        lambda args, artifact_dir, hp_params: captured["saved_metadata"].append(
            (Path(artifact_dir), dict(hp_params))
        ),
    )

    args = Namespace(
        model_name="lnt",
        dataset_name="industry_process",
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        study_run_id="study-run-id",
        timestamp="20260318_000000",
        config_dir=str(CONFIG_DIR),
        study_params={
            "tune_metric": "alarm_score",
            "direction": "maximize",
            "seeds": [42],
        },
    )

    result = model_objectives.run_tune_trainable(
        {
            "window_size": 38,
            "model": {
                "network": {
                    "init_args": {
                        "encoder_type": "bosch_cpc",
                    }
                }
            },
        },
        args,
    )

    assert run_seed_calls == []
    assert result["invalid_config"] is True
    assert result["completed_seed_count"] == 0
    assert result["alarm_score"] == pytest.approx(-1e18)
    assert "window_size >= 41" in result["invalid_config_reason"]
    assert captured["saved_metadata"] == [
        (
            tmp_path / "trial",
            {
                "window_size": 38,
                "model": {
                    "network": {
                        "init_args": {
                            "encoder_type": "bosch_cpc",
                        }
                    }
                },
            },
        )
    ]
    assert ("trial-run-id", "invalid_config", "true") in captured["tags"]


def test_maybe_active_run_propagates_body_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_runs: list[dict[str, object]] = []
    events: list[object] = []
    module_globals = model_objectives.maybe_active_run.__wrapped__.__globals__

    class FakeRun:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, _tb):
            events.append(("exit", exc_type, str(exc) if exc is not None else None))
            return False

    def fake_start_run(*, run_id: str, log_system_metrics: bool) -> FakeRun:
        started_runs.append(
            {
                "run_id": run_id,
                "log_system_metrics": log_system_metrics,
            }
        )
        return FakeRun()

    monkeypatch.setitem(
        module_globals,
        "mlflow",
        SimpleNamespace(start_run=fake_start_run),
    )

    with pytest.raises(RuntimeError, match="train failed"):
        with model_objectives.maybe_active_run(
            run_id="seed-run-id",
            enable_system_metrics=True,
        ):
            raise RuntimeError("train failed")

    assert started_runs == [{"run_id": "seed-run-id", "log_system_metrics": True}]
    assert events == ["enter", ("exit", RuntimeError, "train failed")]


def test_maybe_active_run_warns_and_continues_when_system_metrics_start_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    started_runs: list[dict[str, object]] = []
    module_globals = model_objectives.maybe_active_run.__wrapped__.__globals__

    def fake_start_run(*, run_id: str, log_system_metrics: bool) -> None:
        started_runs.append(
            {
                "run_id": run_id,
                "log_system_metrics": log_system_metrics,
            }
        )
        raise RuntimeError("system metrics unsupported")

    monkeypatch.setitem(
        module_globals,
        "mlflow",
        SimpleNamespace(start_run=fake_start_run),
    )

    with caplog.at_level(logging.WARNING):
        with model_objectives.maybe_active_run(
            run_id="seed-run-id",
            enable_system_metrics=True,
        ):
            executed = True

    assert executed is True
    assert started_runs == [{"run_id": "seed-run-id", "log_system_metrics": True}]
    assert "seed-run-id" in caplog.text
    assert "Continuing without system metrics" in caplog.text


def test_resolve_last_checkpoint_path_raises_clear_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_download_artifacts(*, run_id: str, artifact_path: str) -> str:
        del run_id, artifact_path
        raise model_objectives.mlflow.MlflowException("missing artifact")

    monkeypatch.setattr(
        model_objectives.mlflow.artifacts,
        "download_artifacts",
        fake_download_artifacts,
    )

    with pytest.raises(FileNotFoundError, match="predict stage 'real' for seed 7"):
        model_objectives._resolve_last_checkpoint_path(
            local_root=tmp_path / "real",
            run_id="seed-run-id",
            artifact_path="real/last",
            required=True,
            purpose="predict stage 'real' for seed 7",
        )


def test_predict_func_prefers_local_last_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seed_ckpt_dir = tmp_path / "ckpts" / "seed_7"
    local_stage_ckpt_dir = seed_ckpt_dir / "real"
    local_stage_ckpt_dir.mkdir(parents=True)
    local_checkpoint = local_stage_ckpt_dir / "last.ckpt"
    local_checkpoint.write_text("checkpoint", encoding="utf-8")
    manifest_path = tmp_path / "prepared" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    hp_path = tmp_path / "params.yaml"
    hp_path.write_text("trainer: {}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeTrainer:
        def predict(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            captured["model"] = model
            captured["datamodule"] = datamodule
            captured["ckpt_path"] = ckpt_path

    class FakeCLI:
        def __init__(self, *args, **kwargs) -> None:
            del args
            captured["cli_ckpt_dir"] = kwargs["ckpt_dir"]
            self.trainer = FakeTrainer()
            self.datamodule = SimpleNamespace()
            self.model = SimpleNamespace(
                style_transfer=None,
                use_test_style_transfer=False,
                style_transfer_ckpt_path=None,
                metrics={"alarm_score": 0.9},
                anomaly_scores=np.array([1.0]),
                anomaly_labels=np.array([0]),
            )

    def fake_create_extra_params_for_lightning(*args, **kwargs) -> str:
        del args, kwargs
        return str(hp_path)

    def fail_download_artifacts(*args, **kwargs) -> str:
        raise AssertionError("predict_func should not download MLflow artifacts when local checkpoint exists")

    monkeypatch.setattr(model_objectives, "BenchmarkCLI", FakeCLI)
    monkeypatch.setattr(
        model_objectives,
        "create_extra_params_for_lightning",
        fake_create_extra_params_for_lightning,
    )
    monkeypatch.setattr(
        model_objectives.mlflow.artifacts,
        "download_artifacts",
        fail_download_artifacts,
    )

    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        config_dir=str(CONFIG_DIR),
        requested_dataset_name="cont_reactive_ome",
        data_manifest_path=str(manifest_path),
        tracking_uri="file:///tmp/mlruns",
        mlflow_enable_dataset_tracking=False,
        remove_ckpt=False,
    )
    config = {
        "logger_params": {},
        "hp_params": {},
        "seed": 7,
        "seed_run_id": "seed-run-id",
        "args": args,
        "stage": "real",
        "ckpt_dir": seed_ckpt_dir,
    }

    metrics, scores, labels, dataset_info = model_objectives.predict_func(config)

    assert captured["cli_ckpt_dir"] == local_stage_ckpt_dir
    assert captured["ckpt_path"] == str(local_checkpoint)
    assert metrics == {"alarm_score": 0.9}
    assert np.array_equal(scores, np.array([1.0]))
    assert np.array_equal(labels, np.array([0]))
    assert dataset_info is None


def test_predict_func_skips_checkpoint_lookup_for_detector_managed_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seed_ckpt_dir = tmp_path / "ckpts" / "seed_42"
    manifest_path = tmp_path / "prepared" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    hp_path = tmp_path / "params.yaml"
    hp_path.write_text("trainer: {}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeTrainer:
        def predict(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            captured["model"] = model
            captured["datamodule"] = datamodule
            captured["ckpt_path"] = ckpt_path

    class FakeCLI:
        def __init__(self, *args, **kwargs) -> None:
            del args
            captured["cli_ckpt_dir"] = kwargs["ckpt_dir"]
            self.trainer = FakeTrainer()
            self.datamodule = SimpleNamespace()
            self.model = SimpleNamespace(
                style_transfer=None,
                use_test_style_transfer=False,
                style_transfer_ckpt_path=None,
                metrics={"alarm_score": 0.8},
                anomaly_scores=np.array([0.2, 0.4]),
                anomaly_labels=np.array([0, 1]),
            )

    def fake_create_extra_params_for_lightning(*args, **kwargs) -> str:
        del args, kwargs
        return str(hp_path)

    def fail_checkpoint_lookup(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("detector-managed models do not produce Lightning checkpoints")

    monkeypatch.setattr(model_objectives, "BenchmarkCLI", FakeCLI)
    monkeypatch.setattr(
        model_objectives,
        "create_extra_params_for_lightning",
        fake_create_extra_params_for_lightning,
    )
    monkeypatch.setattr(
        model_objectives,
        "_resolve_last_checkpoint_path",
        fail_checkpoint_lookup,
    )

    args = Namespace(
        model_name="dada",
        dataset_name="batch_dist_ternary_acetone_1_butanol_methanol",
        config_dir=str(CONFIG_DIR),
        requested_dataset_name="batch_dist_ternary_acetone_1_butanol_methanol",
        data_manifest_path=str(manifest_path),
        tracking_uri="file:///tmp/mlruns",
        mlflow_enable_dataset_tracking=False,
        remove_ckpt=False,
    )
    config = {
        "logger_params": {},
        "hp_params": {},
        "seed": 42,
        "seed_run_id": "seed-run-id",
        "args": args,
        "stage": "real",
        "ckpt_dir": seed_ckpt_dir,
    }

    metrics, scores, labels, dataset_info = model_objectives.predict_func(config)

    assert captured["cli_ckpt_dir"] is None
    assert captured["ckpt_path"] is None
    assert metrics == {"alarm_score": 0.8}
    assert np.array_equal(scores, np.array([0.2, 0.4]))
    assert np.array_equal(labels, np.array([0, 1]))
    assert dataset_info is None


def test_train_func_skips_completed_checkpoint_for_detector_managed_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hp_path = tmp_path / "params.yaml"
    hp_path.write_text("trainer: {}\n", encoding="utf-8")
    captured: dict[str, object] = {
        "fit_ckpt_paths": [],
        "predict_ckpt_paths": [],
        "checkpoint_required_flags": [],
        "terminated_runs": [],
    }

    class FakeTrainer:
        def fit(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            del model, datamodule
            captured["fit_ckpt_paths"].append(ckpt_path)

        def predict(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            del model, datamodule
            captured["predict_ckpt_paths"].append(ckpt_path)

    class FakeCLI:
        def __init__(self, *args, **kwargs) -> None:
            del args
            captured["cli_ckpt_dir"] = kwargs["ckpt_dir"]
            self.trainer = FakeTrainer()
            self.datamodule = SimpleNamespace(scaler=None)
            self.model = SimpleNamespace(
                style_transfer=None,
                use_test_style_transfer=False,
                style_transfer_ckpt_path=None,
                metrics={"alarm_score": 0.7},
                anomaly_scores=np.array([0.3]),
                anomaly_labels=np.array([1]),
            )

    class FakeMlflowClient:
        def __init__(self, tracking_uri: str) -> None:
            captured["tracking_uri"] = tracking_uri

        def create_run(self, experiment_id, tags=None, run_name=None):
            captured["experiment_id"] = experiment_id
            captured["stage_tags"] = tags
            captured["stage_run_name"] = run_name
            return SimpleNamespace(info=SimpleNamespace(run_id="stage-run-id"))

        def set_terminated(self, run_id: str) -> None:
            captured["terminated_runs"].append(run_id)

    def fake_create_extra_params_for_lightning(*args, **kwargs) -> str:
        del args, kwargs
        return str(hp_path)

    def fake_resolve_last_checkpoint_path(*, required: bool, purpose: str, **kwargs):
        del kwargs
        captured["checkpoint_required_flags"].append((required, purpose))
        if required:
            raise AssertionError("detector-managed models do not create completed-stage checkpoints")
        return None

    monkeypatch.setattr(model_objectives, "BenchmarkCLI", FakeCLI)
    monkeypatch.setattr(model_objectives, "MlflowClient", FakeMlflowClient)
    monkeypatch.setattr(
        model_objectives,
        "create_extra_params_for_lightning",
        fake_create_extra_params_for_lightning,
    )
    monkeypatch.setattr(
        model_objectives,
        "_resolve_last_checkpoint_path",
        fake_resolve_last_checkpoint_path,
    )
    monkeypatch.setattr(
        model_objectives,
        "_prepared_manifest_path",
        lambda args, stage_data_source: str(tmp_path / f"{stage_data_source}.manifest.json"),
    )
    monkeypatch.setattr(
        model_objectives,
        "maybe_active_run",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        model_objectives.mlflow,
        "log_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("detector-managed models should not log Lightning checkpoints")
        ),
    )

    args = Namespace(
        model_name="carots",
        dataset_name="batch_dist_ternary_acetone_1_butanol_methanol",
        requested_dataset_name="batch_dist_ternary_acetone_1_butanol_methanol",
        config_dir=str(CONFIG_DIR),
        data_manifest_path=str(tmp_path / "manifest.json"),
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        timestamp="20260430_014932",
        tune=True,
        use_generated=False,
        deployment_mode="native",
        remove_ckpt=False,
        mlflow_enable_dataset_tracking=False,
        mlflow_enable_system_metrics=False,
    )
    config = {
        "logger_params": {},
        "hp_params": {},
        "seed": 42,
        "args": args,
        "ckpt_dir": tmp_path / "ckpts" / "seed_42",
        "seed_run_id": "seed-run-id",
        "trial_name": "trial-name",
    }

    stage_results = model_objectives.train_func(config)

    assert captured["fit_ckpt_paths"] == [None]
    assert captured["predict_ckpt_paths"] == [None]
    assert captured["checkpoint_required_flags"] == [
        (False, "resume stage 'real' for seed 42"),
    ]
    assert captured["terminated_runs"] == ["stage-run-id"]
    assert stage_results["real"].metrics == {"alarm_score": 0.7}
    assert np.array_equal(stage_results["real"].anomaly_scores, np.array([0.3]))
    assert np.array_equal(stage_results["real"].anomaly_labels, np.array([1]))


def test_train_func_keeps_default_lightning_checkpoint_out_of_seed_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seed_ckpt_dir = tmp_path / "ckpts" / "seed_42"
    hp_path = tmp_path / "params.yaml"
    hp_path.write_text("trainer: {}\n", encoding="utf-8")
    captured: dict[str, object] = {
        "cli_ckpt_dirs": [],
        "fit_ckpt_paths": [],
        "predict_ckpt_paths": [],
        "mlflow_artifact_uploads": [],
    }

    class FakeTrainer:
        def __init__(self, ckpt_dir: Path) -> None:
            self.ckpt_dir = ckpt_dir

        def fit(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            del model, datamodule
            captured["fit_ckpt_paths"].append(ckpt_path)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            (self.ckpt_dir / "last.ckpt").write_text("checkpoint", encoding="utf-8")

        def predict(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            del model, datamodule
            captured["predict_ckpt_paths"].append(ckpt_path)

    class FakeCLI:
        def __init__(self, *args, **kwargs) -> None:
            del args
            ckpt_dir = Path(kwargs["ckpt_dir"])
            captured["cli_ckpt_dirs"].append(ckpt_dir)
            self.trainer = FakeTrainer(ckpt_dir)
            self.datamodule = SimpleNamespace(scaler=None)
            self.model = SimpleNamespace(
                style_transfer=None,
                use_test_style_transfer=False,
                style_transfer_ckpt_path=None,
                metrics={"alarm_score": 0.9},
                anomaly_scores=np.array([0.5]),
                anomaly_labels=np.array([0]),
            )

    class FakeMlflowClient:
        def __init__(self, tracking_uri: str) -> None:
            captured["tracking_uri"] = tracking_uri

        def create_run(self, experiment_id, tags=None, run_name=None):
            del experiment_id, tags, run_name
            return SimpleNamespace(info=SimpleNamespace(run_id="stage-run-id"))

        def set_terminated(self, run_id: str) -> None:
            captured["terminated_run"] = run_id

    monkeypatch.setattr(model_objectives, "BenchmarkCLI", FakeCLI)
    monkeypatch.setattr(model_objectives, "MlflowClient", FakeMlflowClient)
    monkeypatch.setattr(
        model_objectives,
        "create_extra_params_for_lightning",
        lambda *args, **kwargs: str(hp_path),
    )
    monkeypatch.setattr(
        model_objectives,
        "_prepared_manifest_path",
        lambda args, stage_data_source: str(tmp_path / f"{stage_data_source}.manifest.json"),
    )
    monkeypatch.setattr(model_objectives, "maybe_active_run", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        model_objectives.mlflow,
        "log_artifacts",
        lambda *args, **kwargs: captured["mlflow_artifact_uploads"].append((args, kwargs)),
    )

    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        config_dir=str(CONFIG_DIR),
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        timestamp="20260430_014932",
        tune=True,
        use_generated=False,
        deployment_mode="native",
        remove_ckpt=False,
        mlflow_enable_dataset_tracking=False,
        mlflow_enable_system_metrics=False,
        temp_dir=str(tmp_path / "ray_tmp"),
    )
    config = {
        "logger_params": {},
        "hp_params": {},
        "seed": 42,
        "args": args,
        "ckpt_dir": seed_ckpt_dir,
        "seed_run_id": "seed-run-id",
        "trial_name": "trial-name",
    }

    stage_results = model_objectives.train_func(config)

    artifact_stage_dir = seed_ckpt_dir / "real"
    runtime_stage_dir = captured["cli_ckpt_dirs"][0]
    assert runtime_stage_dir != artifact_stage_dir
    assert not (artifact_stage_dir / "last.ckpt").exists()
    assert not runtime_stage_dir.exists()
    assert captured["fit_ckpt_paths"] == [None]
    assert captured["predict_ckpt_paths"] == [str(runtime_stage_dir / "last.ckpt")]
    assert captured["mlflow_artifact_uploads"] == []
    assert stage_results["real"].metrics == {"alarm_score": 0.9}


def test_train_func_save_checkpoints_preserves_artifact_checkpoint_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seed_ckpt_dir = tmp_path / "ckpts" / "seed_42"
    hp_path = tmp_path / "params.yaml"
    hp_path.write_text("trainer: {}\n", encoding="utf-8")
    captured: dict[str, object] = {
        "cli_ckpt_dirs": [],
        "predict_ckpt_paths": [],
        "mlflow_artifact_uploads": [],
    }

    class FakeTrainer:
        def __init__(self, ckpt_dir: Path) -> None:
            self.ckpt_dir = ckpt_dir

        def fit(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            del model, datamodule, ckpt_path
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            (self.ckpt_dir / "last.ckpt").write_text("checkpoint", encoding="utf-8")

        def predict(self, model, datamodule, ckpt_path: Optional[str] = None) -> None:
            del model, datamodule
            captured["predict_ckpt_paths"].append(ckpt_path)

    class FakeCLI:
        def __init__(self, *args, **kwargs) -> None:
            del args
            ckpt_dir = Path(kwargs["ckpt_dir"])
            captured["cli_ckpt_dirs"].append(ckpt_dir)
            self.trainer = FakeTrainer(ckpt_dir)
            self.datamodule = SimpleNamespace(scaler=None)
            self.model = SimpleNamespace(
                style_transfer=None,
                use_test_style_transfer=False,
                style_transfer_ckpt_path=None,
                metrics={"alarm_score": 0.95},
                anomaly_scores=np.array([0.7]),
                anomaly_labels=np.array([1]),
            )

    class FakeMlflowClient:
        def __init__(self, tracking_uri: str) -> None:
            del tracking_uri

        def create_run(self, experiment_id, tags=None, run_name=None):
            del experiment_id, tags, run_name
            return SimpleNamespace(info=SimpleNamespace(run_id="stage-run-id"))

        def set_terminated(self, run_id: str) -> None:
            captured["terminated_run"] = run_id

    monkeypatch.setattr(model_objectives, "BenchmarkCLI", FakeCLI)
    monkeypatch.setattr(model_objectives, "MlflowClient", FakeMlflowClient)
    monkeypatch.setattr(
        model_objectives,
        "create_extra_params_for_lightning",
        lambda *args, **kwargs: str(hp_path),
    )
    monkeypatch.setattr(
        model_objectives,
        "_prepared_manifest_path",
        lambda args, stage_data_source: str(tmp_path / f"{stage_data_source}.manifest.json"),
    )
    monkeypatch.setattr(model_objectives, "maybe_active_run", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        model_objectives.mlflow,
        "log_artifacts",
        lambda *args, **kwargs: captured["mlflow_artifact_uploads"].append((args, kwargs)),
    )

    args = Namespace(
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        config_dir=str(CONFIG_DIR),
        tracking_uri="file:///tmp/mlruns",
        experiment_id="exp-1",
        timestamp="20260430_014932",
        tune=True,
        use_generated=False,
        deployment_mode="native",
        remove_ckpt=False,
        mlflow_enable_dataset_tracking=False,
        mlflow_enable_system_metrics=False,
        temp_dir=str(tmp_path / "ray_tmp"),
        save_checkpoints=True,
    )
    config = {
        "logger_params": {},
        "hp_params": {},
        "seed": 42,
        "args": args,
        "ckpt_dir": seed_ckpt_dir,
        "seed_run_id": "seed-run-id",
        "trial_name": "trial-name",
    }

    stage_results = model_objectives.train_func(config)

    artifact_stage_dir = seed_ckpt_dir / "real"
    assert captured["cli_ckpt_dirs"] == [artifact_stage_dir]
    assert (artifact_stage_dir / "last.ckpt").exists()
    assert captured["predict_ckpt_paths"] == [str(artifact_stage_dir / "last.ckpt")]
    assert captured["mlflow_artifact_uploads"] == [
        (
            (str(artifact_stage_dir),),
            {"artifact_path": "real/last", "run_id": "seed-run-id"},
        )
    ]
    assert stage_results["real"].metrics == {"alarm_score": 0.95}
