from __future__ import annotations

import contextlib
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from noboom_benchmark.noboom_lib.core.tune import mlflow_models


def test_select_canonical_model_prefers_smaller_seed_on_tie(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    (trial_dir / "artifacts" / "ckpts" / "seed_42").mkdir(parents=True)
    (trial_dir / "artifacts" / "ckpts" / "seed_43").mkdir(parents=True)

    class FakeClient:
        def search_runs(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    info=SimpleNamespace(run_id="seed-run-43"),
                    data=SimpleNamespace(metrics={"alarm_score": 0.8}, tags={"seed": "43"}),
                ),
                SimpleNamespace(
                    info=SimpleNamespace(run_id="seed-run-42"),
                    data=SimpleNamespace(metrics={"alarm_score": 0.8}, tags={"seed": "42"}),
                ),
            ]

    selection = mlflow_models.select_canonical_model(
        FakeClient(),
        experiment_id="exp-1",
        trial_run_id="trial-run-id",
        tune_metric="alarm_score",
        tune_mode="max",
        trial_artifact_dir=trial_dir,
    )

    assert selection.seed == 42
    assert selection.seed_run_id == "seed-run-42"
    assert selection.stage == "real"
    assert selection.seed_artifact_dir == trial_dir / "artifacts" / "ckpts" / "seed_42"


def test_select_canonical_model_supports_legacy_checkpoint_layout(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    (trial_dir / "ckpts" / "seed_42").mkdir(parents=True)

    class FakeClient:
        def search_runs(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    info=SimpleNamespace(run_id="seed-run-42"),
                    data=SimpleNamespace(metrics={"alarm_score": 0.8}, tags={"seed": "42"}),
                ),
            ]

    selection = mlflow_models.select_canonical_model(
        FakeClient(),
        experiment_id="exp-1",
        trial_run_id="trial-run-id",
        tune_metric="alarm_score",
        tune_mode="max",
        trial_artifact_dir=trial_dir,
    )

    assert selection.seed_artifact_dir == trial_dir / "ckpts" / "seed_42"


def test_resolve_canonical_checkpoint_path_supports_flat_stage_layout(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_42"
    checkpoint_path = seed_dir / "real" / "last.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    resolved = mlflow_models._resolve_canonical_checkpoint_path(seed_dir, "real")  # noqa: SLF001

    assert resolved == checkpoint_path


def test_feature_names_for_run_falls_back_to_manifest_when_scaler_names_are_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scaler_path = tmp_path / "scaler.pkl"
    with scaler_path.open("wb") as handle:
        pickle.dump(
            SimpleNamespace(feature_names_in_=[], feature_names_out_=[]),
            handle,
        )

    monkeypatch.setattr(
        mlflow_models,
        "_load_real_stage_manifest",
        lambda args: SimpleNamespace(feature_names=["f1", "f2", "f3"]),
    )

    feature_names = mlflow_models._feature_names_for_run(  # noqa: SLF001
        args=SimpleNamespace(),
        scaler_path=scaler_path,
    )

    assert feature_names == ["f1", "f2", "f3"]


def test_split_model_input_groups_dataframe_by_sequence_id() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 10.0],
            "feature_b": [3.0, 4.0, 20.0],
            "sequence_id": [0, 0, 1],
            "label": [0, 0, 1],
        }
    )

    sequences, original_frame = mlflow_models._split_model_input(  # noqa: SLF001
        frame,
        expected_feature_names=["feature_a", "feature_b"],
    )

    assert original_frame is frame
    assert len(sequences) == 2
    assert sequences[0].shape == (2, 2)
    assert sequences[1].shape == (1, 2)


def test_evaluation_rows_convert_numpy_scalars() -> None:
    rows = mlflow_models.evaluation_rows(
        {
            "alarm_score": np.float32(0.9),
            "evaluation_dataset_name": "cont_reactive_ome__test__real",
        }
    )

    assert rows[0]["metric"] == "alarm_score"
    assert rows[0]["value"] == pytest.approx(0.9)
    assert rows[1] == {
        "metric": "evaluation_dataset_name",
        "value": "cont_reactive_ome__test__real",
    }


def test_log_study_model_registers_candidate_alias_when_enabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    alias_calls = []

    monkeypatch.setattr(
        mlflow_models,
        "select_canonical_model",
        lambda *args, **kwargs: mlflow_models.CanonicalModelSelection(
            trial_run_id="trial-run-id",
            trial_artifact_dir=tmp_path,
            seed_run_id="seed-run-id",
            seed=42,
            stage="real",
            seed_artifact_dir=tmp_path / "seed_42",
        ),
    )
    monkeypatch.setattr(
        mlflow_models,
        "_build_model_bundle",
        lambda *args, **kwargs: (
            bundle_dir,
            object(),
            {"feature_names": ["f1"], "hp_params": {"lr": 1e-3}},
        ),
    )
    monkeypatch.setattr(
        mlflow_models,
        "_build_example_signature",
        lambda feature_names: (
            pd.DataFrame({"f1": [0.0]}),
            pd.DataFrame({"score": [0.0]}),
            "signature",
        ),
    )
    monkeypatch.setattr(mlflow_models.mlflow, "start_run", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        mlflow_models.mlflow.pyfunc,
        "log_model",
        lambda **kwargs: SimpleNamespace(model_id="model-id", model_uri="models:/benchmark_model/1"),
    )
    monkeypatch.setattr(
        mlflow_models.mlflow,
        "register_model",
        lambda model_uri, name: SimpleNamespace(version="3"),
    )

    class FakeClient:
        def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
            alias_calls.append((name, alias, version))

    args = SimpleNamespace(
        study_params={"tune_metric": "alarm_score", "direction": "maximize"},
        experiment_id="exp-1",
        model_name="gdn",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome_tsst",
        mlflow_enable_registry=True,
    )

    info = mlflow_models.log_study_model(
        args,
        FakeClient(),
        study_run_id="study-run-id",
        trial_run_id="trial-run-id",
        trial_artifact_dir=tmp_path,
        resolved_hparams={"lr": 1e-3},
    )

    assert info.registered_model_name == "noboom__gdn__cont_reactive_ome_tsst"
    assert info.registered_model_version == "3"
    assert alias_calls == [("noboom__gdn__cont_reactive_ome_tsst", "candidate", "3")]


def test_log_study_evaluation_uses_sequence_aware_postprocessing_when_enabled(
    monkeypatch,
) -> None:
    set_tags = []
    helper_calls = []

    evaluation_frame = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "sequence_id": [0, 0, 1],
            "label": [0, 1, 0],
        }
    )

    monkeypatch.setattr(mlflow_models, "build_evaluation_frame", lambda args, feature_names: evaluation_frame)
    monkeypatch.setattr(mlflow_models, "_load_real_stage_manifest", lambda args: SimpleNamespace())
    monkeypatch.setattr(mlflow_models, "build_dataset_source", lambda manifest, mode: "dataset-source")
    monkeypatch.setattr(
        mlflow_models,
        "get_or_create_logged_dataset",
        lambda *args, **kwargs: SimpleNamespace(name="dataset-name", digest="dataset-digest"),
    )
    monkeypatch.setattr(mlflow_models, "log_dataset_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mlflow_models.mlflow.pyfunc,
        "load_model",
        lambda uri: SimpleNamespace(
            predict=lambda frame: pd.DataFrame({"score": [0.1, 0.9, 0.2]})
        ),
    )
    monkeypatch.setattr(
        mlflow_models.mlflow.models,
        "evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("mlflow.models.evaluate should not run when postprocessing is enabled")
        ),
    )

    def fake_postprocess(score_sequences, label_sequences, *, metric_names, config, fix_threshold=False, dataset_name=None):
        del metric_names, config, fix_threshold, dataset_name
        helper_calls.append((score_sequences, label_sequences))
        return SimpleNamespace(metric_values={"alarm_score": 0.75})

    monkeypatch.setattr(
        mlflow_models,
        "evaluate_sequences_with_optional_postprocessing",
        fake_postprocess,
    )

    class FakeClient:
        def set_tag(self, run_id: str, key: str, value: str) -> None:
            set_tags.append((run_id, key, value))

    args = SimpleNamespace(
        study_params={
            "metrics": ["alarm_score"],
            "evaluation_postprocessing": {"enabled": True},
        },
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        datasource=None,
    )
    model_info = mlflow_models.LoggedStudyModelInfo(
        model_id="model-id",
        model_uri="models:/benchmark/1",
        selected_seed=42,
        selected_stage="real",
        feature_names=["f1"],
    )

    result = mlflow_models.log_study_evaluation(
        args,
        FakeClient(),
        study_run_id="study-run-id",
        model_info=model_info,
    )

    assert len(helper_calls) == 1
    score_sequences, label_sequences = helper_calls[0]
    np.testing.assert_allclose(score_sequences[0], np.array([0.1, 0.9]))
    np.testing.assert_allclose(score_sequences[1], np.array([0.2]))
    np.testing.assert_array_equal(label_sequences[0], np.array([0, 1]))
    np.testing.assert_array_equal(label_sequences[1], np.array([0]))
    assert result["alarm_score"] == pytest.approx(0.75)
    assert result["evaluation_postprocessing_enabled"] is True
    assert result["evaluation_dataset_name"] == "dataset-name"
    assert ("study-run-id", "evaluation_postprocessing_enabled", "true") in set_tags


def test_log_study_evaluation_uses_mlflow_evaluate_when_postprocessing_disabled(
    monkeypatch,
) -> None:
    set_tags = []

    evaluation_frame = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "sequence_id": [0, 0],
            "label": [0, 1],
        }
    )

    monkeypatch.setattr(mlflow_models, "build_evaluation_frame", lambda args, feature_names: evaluation_frame)
    monkeypatch.setattr(mlflow_models, "_load_real_stage_manifest", lambda args: SimpleNamespace())
    monkeypatch.setattr(mlflow_models, "build_dataset_source", lambda manifest, mode: "dataset-source")
    monkeypatch.setattr(
        mlflow_models,
        "get_or_create_logged_dataset",
        lambda *args, **kwargs: SimpleNamespace(name="dataset-name", digest="dataset-digest"),
    )
    monkeypatch.setattr(mlflow_models, "log_dataset_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow_models.mlflow, "start_run", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        mlflow_models.mlflow.models,
        "evaluate",
        lambda **kwargs: SimpleNamespace(metrics={"alarm_score": 0.6}),
    )
    monkeypatch.setattr(
        mlflow_models.mlflow.pyfunc,
        "load_model",
        lambda uri: (_ for _ in ()).throw(
            AssertionError("pyfunc.load_model should not run when postprocessing is disabled")
        ),
    )

    class FakeClient:
        def set_tag(self, run_id: str, key: str, value: str) -> None:
            set_tags.append((run_id, key, value))

    args = SimpleNamespace(
        study_params={
            "metrics": ["alarm_score"],
            "evaluation_postprocessing": {"enabled": False},
        },
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        datasource=None,
    )
    model_info = mlflow_models.LoggedStudyModelInfo(
        model_id="model-id",
        model_uri="models:/benchmark/1",
        selected_seed=42,
        selected_stage="real",
        feature_names=["f1"],
    )

    result = mlflow_models.log_study_evaluation(
        args,
        FakeClient(),
        study_run_id="study-run-id",
        model_info=model_info,
    )

    assert result["alarm_score"] == pytest.approx(0.6)
    assert result["evaluation_postprocessing_enabled"] is False
    assert ("study-run-id", "evaluation_postprocessing_enabled", "false") in set_tags
