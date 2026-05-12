"""Study-level MLflow model logging, evaluation, and registry helpers."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature, make_metric
import numpy as np
import pandas as pd
import torch
import yaml
from timesead.data.transforms import PipelineDataset

from ..benchmark_utils import BenchmarkCLI
from ..benchmark_utils.arrow_runtime import build_dataset_source
from ..metric_utils.alarm_threshold_search import select_threshold
from ..metric_utils.evaluation_postprocessing import (
    evaluate_sequences_with_optional_postprocessing,
    resolve_evaluation_postprocessing_config,
)
from ..metric_utils.metrics import get_metric_by_name, is_metric_binary
from ..data_prep.manifest import PreparedDatasetManifest
from ..data_prep.runtime import manifest_path_for_stage_from_args
from ..tune_constants import CKPTS_DIR
from ..tune_utils import find_matching_metadata
from .mlflow_datasets import DatasetLineageCache, get_or_create_logged_dataset, log_dataset_input


logger = logging.getLogger(__name__)

SEQUENCE_ID_COLUMN = "sequence_id"
LABEL_COLUMN = "label"
LOGGED_MODEL_NAME = "benchmark_model"


@dataclass(frozen=True)
class CanonicalModelSelection:
    trial_run_id: str
    trial_artifact_dir: Path
    seed_run_id: str
    seed: int
    stage: str
    seed_artifact_dir: Path


@dataclass(frozen=True)
class LoggedStudyModelInfo:
    model_id: str
    model_uri: str
    selected_seed: int
    selected_stage: str
    feature_names: list[str]
    registered_model_name: Optional[str] = None
    registered_model_version: Optional[str] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None


def _metric_greater_is_better(metric_name: str) -> bool:
    return metric_name not in {"aaf"}


def _flatten_score_array(scores: np.ndarray) -> pd.DataFrame:
    score_array = np.asarray(scores)
    if score_array.ndim == 1:
        return pd.DataFrame({"score": score_array})
    if score_array.ndim == 2 and score_array.shape[1] == 1:
        return pd.DataFrame({"score": score_array[:, 0]})
    frame = pd.DataFrame(score_array, columns=[f"score_{idx}" for idx in range(score_array.shape[1])])
    frame.insert(0, "score", frame.max(axis=1))
    return frame


def _feature_columns_from_frame(
    frame: pd.DataFrame,
    expected_feature_names: Sequence[str],
) -> list[str]:
    if expected_feature_names and all(column in frame.columns for column in expected_feature_names):
        return list(expected_feature_names)
    reserved = {SEQUENCE_ID_COLUMN, LABEL_COLUMN, "target", "y_true"}
    numeric_columns = [
        column
        for column in frame.columns
        if column not in reserved and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if expected_feature_names and len(numeric_columns) != len(expected_feature_names):
        raise ValueError(
            f"Expected {len(expected_feature_names)} feature columns, found {len(numeric_columns)}."
        )
    return numeric_columns


def _split_model_input(
    model_input: Any,
    expected_feature_names: Sequence[str],
) -> tuple[list[np.ndarray], Optional[pd.DataFrame]]:
    if isinstance(model_input, pd.DataFrame):
        feature_columns = _feature_columns_from_frame(model_input, expected_feature_names)
        if SEQUENCE_ID_COLUMN in model_input.columns:
            grouped = model_input.groupby(SEQUENCE_ID_COLUMN, sort=False)
            sequences = [
                group.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=False)
                for _, group in grouped
            ]
        else:
            sequences = [model_input.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=False)]
        return sequences, model_input

    array_input = np.asarray(model_input, dtype=np.float32)
    if array_input.ndim != 2:
        raise ValueError("Logged benchmark models expect a 2-D array shaped [timesteps, features].")
    return [array_input], None


def _result_frame_for_input(
    input_frame: Optional[pd.DataFrame],
    scores: np.ndarray,
) -> pd.DataFrame:
    score_frame = _flatten_score_array(scores)
    if input_frame is None:
        return score_frame

    result_frame = pd.DataFrame(index=input_frame.index)
    if SEQUENCE_ID_COLUMN in input_frame.columns:
        result_frame[SEQUENCE_ID_COLUMN] = input_frame[SEQUENCE_ID_COLUMN]
    for column in score_frame.columns:
        result_frame[column] = score_frame[column].to_numpy()
    return result_frame


def _write_override_config(
    bundle_dir: Path,
    *,
    base_hparams: Mapping[str, Any],
    inference_dataset_path: Path,
    scaler_state_path: Path,
) -> Path:
    override_path = bundle_dir / "inference_overrides.yaml"
    payload = dict(base_hparams)
    payload.update({
        "model.compile_torch_model": False,
        "trainer.accelerator": "cpu",
        "trainer.devices": 1,
        "trainer.precision": 32,
        "trainer.logger": False,
        "trainer.enable_checkpointing": False,
        "data.num_workers": 0,
        "data.inference_dataset_path": str(inference_dataset_path),
        "data.scaler_state_path": str(scaler_state_path),
        "data.data_source": "real",
        "data.data_manifest_path": None,
        "model.use_test_style_transfer": False,
        "model.ckpt_path": None,
        "model.predict_on_end_streaming": False,
        "model.prediction_stream.enabled": False,
    })
    with open(override_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)
    return override_path


class LoggedBenchmarkPyfuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self._bundle_dir = Path(context.artifacts["bundle"])
        self._manifest = json.loads((self._bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        self._feature_names = list(self._manifest.get("feature_names", []))

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        del context, params
        sequences, input_frame = _split_model_input(model_input, self._feature_names)
        zero_targets = [np.zeros(sequence.shape[0], dtype=np.int64) for sequence in sequences]

        with tempfile.TemporaryDirectory(prefix="noboom-logged-model-") as temp_dir:
            temp_root = Path(temp_dir)
            inference_dataset_path = temp_root / "inference_dataset.pt"
            torch.save({"samples": sequences, "targets": zero_targets}, inference_dataset_path)
            override_path = _write_override_config(
                temp_root,
                base_hparams=dict(self._manifest["hp_params"]),
                inference_dataset_path=inference_dataset_path,
                scaler_state_path=self._bundle_dir / "scaler.pkl",
            )
            cli = BenchmarkCLI(
                self._manifest["model_name"],
                self._manifest["dataset_name"],
                data_manifest_path=None,
                config_dir=str(self._bundle_dir / "configs"),
                ckpt_dir=str(temp_root / "checkpoints"),
                requested_dataset_name=self._manifest.get("requested_dataset_name"),
                hp_cfg=str(override_path),
            )
            checkpoint = torch.load(self._bundle_dir / "checkpoint.ckpt", map_location="cpu")
            cli.model.on_load_checkpoint(checkpoint)
            cli.model.load_state_dict(checkpoint["state_dict"], strict=False)
            cli.model.to("cpu")
            cli.model.eval()
            if cli.model.network is not None and hasattr(cli.model.detector, "model"):
                cli.model.detector.model = cli.model.network

            cli.datamodule.setup("predict")
            raw_batches = []
            with torch.inference_mode():
                for batch_idx, batch in enumerate(cli.datamodule.predict_dataloader()):
                    raw_batches.append(cli.model.predict_step(batch, batch_idx))
            if not raw_batches:
                raise RuntimeError("Logged benchmark model produced no prediction batches.")

            raw_scores = torch.cat(raw_batches, dim=0)
            labels = []
            for _, label in cli.datamodule.predict_orig_dataloader():
                labels.append(label[0].select(cli.model.batch_dimension, 0))
            label_tensor = torch.cat(labels, dim=0)
            aligned_scores, _aligned_labels = cli.model.align_prediction_outputs(
                raw_scores,
                label_tensor,
                cli.datamodule.seq_len("test"),
            )
        return _result_frame_for_input(input_frame, aligned_scores)


class LoggedThresholdPyfuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self._bundle_dir = Path(context.artifacts["bundle"])
        self._manifest = json.loads((self._bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        self._feature_names = list(self._manifest.get("feature_names", []))

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        del context, params
        sequences, input_frame = _split_model_input(model_input, self._feature_names)

        from noboom.tsad.baselines import MovingMeanDifferenceAnomalyDetector

        detector = MovingMeanDifferenceAnomalyDetector(
            self._manifest["hp_params"]["window_size"],
            self._manifest["hp_params"]["use_mean"],
            self._manifest["hp_params"]["std_coefficient"],
            two_sided=self._manifest["hp_params"]["two_sided"],
        )
        feature_index = int(self._manifest["feature_index"])
        flattened_scores = []
        for sequence in sequences:
            detector_scores = detector.score(torch.from_numpy(sequence)).cpu().numpy()
            row_scores = np.concatenate(
                [
                    np.zeros(1, dtype=np.float32),
                    detector_scores[:, feature_index].astype(np.float32, copy=False),
                ]
            )
            flattened_scores.append(row_scores)
        aligned_scores = np.concatenate(flattened_scores, axis=0)
        return _result_frame_for_input(input_frame, aligned_scores)


def _resolve_trial_artifact_dir(
    best_result: Any,
    *,
    experiment_path: Path,
    best_hp_params: Mapping[str, Any],
) -> Path:
    for attr_name in ("path", "log_dir"):
        candidate = getattr(best_result, attr_name, None)
        if candidate:
            return Path(candidate)

    metadata_path, _metadata = find_matching_metadata(
        experiment_path,
        dict(best_hp_params),
        exact=True,
        verbose=False,
    )
    if metadata_path is None:
        raise FileNotFoundError(f"Could not resolve best trial artifact directory under {experiment_path}.")
    return metadata_path.parent


def _resolve_seed_artifact_dir(trial_artifact_dir: Path, selected_seed: int) -> Path:
    seed_suffix = Path(f"seed_{selected_seed}")
    candidate_dirs = [
        trial_artifact_dir / CKPTS_DIR / seed_suffix,
        trial_artifact_dir / "ckpts" / seed_suffix,
    ]
    for candidate_dir in candidate_dirs:
        if candidate_dir.exists():
            return candidate_dir

    candidate_display = ", ".join(str(candidate_dir) for candidate_dir in candidate_dirs)
    raise FileNotFoundError(
        f"Selected seed artifact directory does not exist under {trial_artifact_dir}: {candidate_display}"
    )


def _resolve_canonical_checkpoint_path(seed_artifact_dir: Path, stage: str) -> Path:
    stage_dir = seed_artifact_dir / stage
    candidate_paths = [
        stage_dir / "last.ckpt",
        stage_dir / "last" / "last.ckpt",
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    candidate_display = ", ".join(str(candidate_path) for candidate_path in candidate_paths)
    raise FileNotFoundError(f"Canonical checkpoint not found. Searched: {candidate_display}")


def select_canonical_model(
    client: MlflowClient,
    *,
    experiment_id: str,
    trial_run_id: str,
    tune_metric: str,
    tune_mode: str,
    trial_artifact_dir: Path,
) -> CanonicalModelSelection:
    seed_runs = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.run_level = 'seed' and tags.mlflow.parentRunId = '{trial_run_id}'"
        ),
        max_results=1000,
    )
    if not seed_runs:
        raise RuntimeError(f"No MLflow seed runs found for trial run '{trial_run_id}'.")

    reverse = tune_mode == "max"
    ranked_seed_runs = sorted(
        seed_runs,
        key=lambda run: (
            float(run.data.metrics.get(tune_metric, float("-inf") if reverse else float("inf"))),
            -int(run.data.tags.get("seed", "0")) if reverse else int(run.data.tags.get("seed", "0")),
        ),
        reverse=reverse,
    )
    selected_seed_run = ranked_seed_runs[0]
    selected_seed = int(selected_seed_run.data.tags["seed"])
    seed_artifact_dir = _resolve_seed_artifact_dir(trial_artifact_dir, selected_seed)
    return CanonicalModelSelection(
        trial_run_id=trial_run_id,
        trial_artifact_dir=trial_artifact_dir,
        seed_run_id=selected_seed_run.info.run_id,
        seed=selected_seed,
        stage="real",
        seed_artifact_dir=seed_artifact_dir,
    )


def _copy_config_bundle(args: Any, bundle_dir: Path) -> None:
    configs_root = bundle_dir / "configs" / "models"
    configs_root.mkdir(parents=True, exist_ok=True)
    for file_name in ("common.yaml", f"{args.model_name}.yaml"):
        source_path = Path(args.config_dir) / "models" / file_name
        target_path = configs_root / file_name
        target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def _feature_names_for_run(
    *,
    args: Any,
    scaler_path: Optional[Path] = None,
) -> list[str]:
    if scaler_path is not None and scaler_path.exists():
        with open(scaler_path, "rb") as handle:
            scaler = pickle.load(handle)
        for attr_name in ("feature_names_out_", "feature_names_in_"):
            feature_names = getattr(scaler, attr_name, None)
            if feature_names:
                return [str(name) for name in feature_names]
    manifest = _load_real_stage_manifest(args)
    return list(manifest.feature_names)


def _build_model_bundle(
    args: Any,
    *,
    selection: CanonicalModelSelection,
    resolved_hparams: Mapping[str, Any],
) -> tuple[Path, mlflow.pyfunc.PythonModel, Dict[str, Any]]:
    bundle_root = Path(tempfile.mkdtemp(prefix="noboom-mlflow-model-"))
    _copy_config_bundle(args, bundle_root)

    manifest: Dict[str, Any] = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "requested_dataset_name": getattr(args, "requested_dataset_name", args.dataset_name),
        "selected_seed": selection.seed,
        "selected_stage": selection.stage,
        "hp_params": dict(resolved_hparams),
        "tune_metric": args.study_params["tune_metric"],
    }

    if args.model_name == "threshold":
        metrics_payload = json.loads((selection.seed_artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        manifest["feature_index"] = int(metrics_payload["feature"])
        manifest["threshold"] = float(metrics_payload["threshold"])
        manifest["feature_names"] = _feature_names_for_run(args=args)
        python_model: mlflow.pyfunc.PythonModel = LoggedThresholdPyfuncModel()
    else:
        checkpoint_path = _resolve_canonical_checkpoint_path(
            selection.seed_artifact_dir,
            selection.stage,
        )
        scaler_path = selection.seed_artifact_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Canonical scaler not found: {scaler_path}")
        (bundle_root / "checkpoint.ckpt").write_bytes(checkpoint_path.read_bytes())
        (bundle_root / "scaler.pkl").write_bytes(scaler_path.read_bytes())
        manifest["feature_names"] = _feature_names_for_run(args=args, scaler_path=scaler_path)
        python_model = LoggedBenchmarkPyfuncModel()

    with open(bundle_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return bundle_root, python_model, manifest


def _build_example_signature(feature_names: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    example_input = pd.DataFrame(
        np.zeros((4, len(feature_names)), dtype=np.float32),
        columns=list(feature_names),
    )
    example_output = pd.DataFrame({"score": np.zeros(4, dtype=np.float32)})
    return example_input, example_output, infer_signature(example_input, example_output)


def _load_real_stage_manifest(args: Any) -> PreparedDatasetManifest:
    manifest_path = manifest_path_for_stage_from_args(args, "real")
    return PreparedDatasetManifest.load(manifest_path)


def log_study_model(
    args: Any,
    client: MlflowClient,
    *,
    study_run_id: str,
    trial_run_id: str,
    trial_artifact_dir: Path,
    resolved_hparams: Mapping[str, Any],
) -> LoggedStudyModelInfo:
    tune_mode = "max" if args.study_params.get("direction") == "maximize" else "min"
    selection = select_canonical_model(
        client,
        experiment_id=args.experiment_id,
        trial_run_id=trial_run_id,
        tune_metric=args.study_params["tune_metric"],
        tune_mode=tune_mode,
        trial_artifact_dir=trial_artifact_dir,
    )
    bundle_dir, python_model, manifest = _build_model_bundle(
        args,
        selection=selection,
        resolved_hparams=resolved_hparams,
    )
    example_input, _example_output, signature = _build_example_signature(manifest["feature_names"])
    try:
        with mlflow.start_run(run_id=study_run_id):
            model_info = mlflow.pyfunc.log_model(
                name=LOGGED_MODEL_NAME,
                python_model=python_model,
                artifacts={"bundle": str(bundle_dir)},
                code_paths=[str(Path(__file__).resolve().parents[4])],
                input_example=example_input,
                signature=signature,
                model_type="timeseries_anomaly_detection",
                metadata=manifest,
                params=dict(resolved_hparams),
                tags={
                    "selected_seed": str(selection.seed),
                    "selected_stage": selection.stage,
                },
            )
    finally:
        shutil.rmtree(bundle_dir, ignore_errors=True)

    registered_model_name = None
    registered_model_version = None
    if getattr(args, "mlflow_enable_registry", False):
        registered_model_name = (
            f"noboom__{args.model_name}__{getattr(args, 'requested_dataset_name', args.dataset_name)}"
        )
        model_version = mlflow.register_model(model_info.model_uri, registered_model_name)
        registered_model_version = str(model_version.version)
        client.set_registered_model_alias(registered_model_name, "candidate", registered_model_version)

    return LoggedStudyModelInfo(
        model_id=str(model_info.model_id),
        model_uri=str(model_info.model_uri),
        selected_seed=selection.seed,
        selected_stage=selection.stage,
        feature_names=list(manifest["feature_names"]),
        registered_model_name=registered_model_name,
        registered_model_version=registered_model_version,
    )


def build_evaluation_frame(args: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    manifest = _load_real_stage_manifest(args)
    test_source = build_dataset_source(manifest, mode="test")
    rows = []
    for sequence_id, (inputs, targets) in enumerate(PipelineDataset(test_source)):
        sequence_frame = pd.DataFrame(inputs[0].numpy(), columns=list(feature_names))
        sequence_frame[SEQUENCE_ID_COLUMN] = sequence_id
        sequence_frame[LABEL_COLUMN] = targets[0].numpy()
        rows.append(sequence_frame)
    if not rows:
        raise RuntimeError(f"Evaluation frame is empty for dataset '{args.dataset_name}'.")
    return pd.concat(rows, axis=0, ignore_index=True)


def build_extra_metrics(metric_names: Iterable[str]) -> list[Any]:
    extra_metrics = []
    for metric_name in metric_names:
        metric_fn = get_metric_by_name(metric_name)

        def _eval_fn(
            predictions: pd.Series,
            targets: pd.Series,
            metrics: Dict[str, Any],
            *,
            metric_name: str = metric_name,
            metric_fn: Any = metric_fn,
        ) -> float:
            del metrics
            labels = targets.to_numpy()
            threshold_labels = (labels > 0).astype(int)
            threshold = select_threshold(predictions.to_numpy(), threshold_labels).best_threshold
            predicted_labels = (predictions.to_numpy() > threshold).astype(int)
            metric_targets = threshold_labels if is_metric_binary(metric_name) else labels
            return float(metric_fn(predicted_labels, metric_targets))

        extra_metrics.append(
            make_metric(
                eval_fn=_eval_fn,
                greater_is_better=_metric_greater_is_better(metric_name),
                name=metric_name,
            )
        )
    return extra_metrics


def _evaluation_postprocessing_config_for_args(args: Any):
    study_params = getattr(args, "study_params", None)
    if study_params is None:
        return resolve_evaluation_postprocessing_config(None)
    return resolve_evaluation_postprocessing_config(study_params.get("evaluation_postprocessing"))


def _prediction_frame_from_output(
    evaluation_frame: pd.DataFrame,
    model_output: Any,
) -> pd.DataFrame:
    if isinstance(model_output, pd.DataFrame):
        return model_output
    return _result_frame_for_input(evaluation_frame, np.asarray(model_output))


def _sequence_scores_and_labels_from_frames(
    evaluation_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if "score" not in prediction_frame.columns:
        raise ValueError("Prediction frame must contain a 'score' column for evaluation.")

    aligned = pd.DataFrame(index=evaluation_frame.index)
    if SEQUENCE_ID_COLUMN in evaluation_frame.columns:
        aligned[SEQUENCE_ID_COLUMN] = evaluation_frame[SEQUENCE_ID_COLUMN].to_numpy()
    else:
        aligned[SEQUENCE_ID_COLUMN] = 0
    aligned[LABEL_COLUMN] = evaluation_frame[LABEL_COLUMN].to_numpy()
    aligned["score"] = prediction_frame["score"].to_numpy()

    score_sequences = []
    label_sequences = []
    for _, group in aligned.groupby(SEQUENCE_ID_COLUMN, sort=False):
        score_sequences.append(group["score"].to_numpy(dtype=np.float64, copy=False))
        label_sequences.append(group[LABEL_COLUMN].to_numpy(copy=False))
    return score_sequences, label_sequences


def log_study_evaluation(
    args: Any,
    client: MlflowClient,
    *,
    study_run_id: str,
    model_info: LoggedStudyModelInfo,
) -> Dict[str, Any]:
    feature_names = list(model_info.feature_names) or _feature_names_for_run(args=args)
    postprocessing_config = _evaluation_postprocessing_config_for_args(args)

    evaluation_frame = build_evaluation_frame(args, feature_names)

    dataset_cache = DatasetLineageCache()
    manifest = _load_real_stage_manifest(args)
    evaluation_dataset_source = build_dataset_source(manifest, mode="test")
    dataset_info = get_or_create_logged_dataset(
        dataset_cache,
        cache_key=f"{study_run_id}::evaluation",
        dataset_source=evaluation_dataset_source,
        requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
        split_name="test",
        data_source="real",
        source_uri=getattr(args, "datasource", None),
        context="evaluation",
    )
    log_dataset_input(client, study_run_id, dataset_info, tags={"split": "test", "phase": "evaluation"})
    client.set_tag(
        study_run_id,
        "evaluation_postprocessing_enabled",
        str(postprocessing_config.enabled).lower(),
    )
    client.set_tag(
        study_run_id,
        "evaluation_postprocessing_config",
        json.dumps(postprocessing_config.to_dict(), sort_keys=True),
    )

    if postprocessing_config.enabled:
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        model_output = loaded_model.predict(evaluation_frame)
        prediction_frame = _prediction_frame_from_output(evaluation_frame, model_output)
        score_sequences, label_sequences = _sequence_scores_and_labels_from_frames(
            evaluation_frame,
            prediction_frame,
        )
        processed = evaluate_sequences_with_optional_postprocessing(
            score_sequences,
            label_sequences,
            metric_names=tuple(args.study_params["metrics"]),
            config=postprocessing_config,
        )
        merged_metrics = {
            **processed.metric_values,
            "evaluation_dataset_name": dataset_info.name,
            "evaluation_dataset_digest": dataset_info.digest,
            "evaluation_postprocessing_enabled": True,
            "evaluation_postprocessing_config": json.dumps(
                postprocessing_config.to_dict(),
                sort_keys=True,
            ),
        }
    else:
        evaluation_metrics = build_extra_metrics(args.study_params["metrics"])
        with mlflow.start_run(run_id=study_run_id):
            evaluation_result = mlflow.models.evaluate(
                model=model_info.model_uri,
                data=evaluation_frame,
                targets=LABEL_COLUMN,
                predictions="score",
                model_type="regressor",
                evaluator_config={"log_model_explainability": False},
                extra_metrics=evaluation_metrics,
                model_id=model_info.model_id,
            )

        merged_metrics = {
            **evaluation_result.metrics,
            "evaluation_dataset_name": dataset_info.name,
            "evaluation_dataset_digest": dataset_info.digest,
            "evaluation_postprocessing_enabled": False,
            "evaluation_postprocessing_config": json.dumps(
                postprocessing_config.to_dict(),
                sort_keys=True,
            ),
        }
    return merged_metrics


def trial_rows_for_study(client: MlflowClient, *, experiment_id: str, study_run_id: str) -> list[Dict[str, Any]]:
    trial_runs = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.run_level = 'trial' and tags.mlflow.parentRunId = '{study_run_id}'"
        ),
        max_results=1000,
    )
    rows = []
    for run in trial_runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
        }
        row.update(run.data.metrics)
        rows.append(row)
    return rows


def seed_rows_for_trial(client: MlflowClient, *, experiment_id: str, trial_run_id: str) -> list[Dict[str, Any]]:
    seed_runs = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.run_level = 'seed' and tags.mlflow.parentRunId = '{trial_run_id}'"
        ),
        max_results=1000,
    )
    rows = []
    for run in seed_runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "seed": int(run.data.tags.get("seed", "0")),
            "status": run.info.status,
        }
        row.update(run.data.metrics)
        rows.append(row)
    return sorted(rows, key=lambda row: row["seed"])


def resolve_best_trial_dir(
    best_result: Any,
    *,
    experiment_path: Path,
) -> Path:
    best_config = getattr(best_result, "config", None)
    if best_config is None:
        raise ValueError("Best Ray result does not expose a config payload.")
    return _resolve_trial_artifact_dir(
        best_result,
        experiment_path=experiment_path,
        best_hp_params=best_config,
    )


def evaluation_rows(metrics: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for metric_name, metric_value in sorted(metrics.items()):
        if isinstance(metric_value, (np.generic,)):
            metric_value = metric_value.item()
        rows.append(
            {
                "metric": str(metric_name),
                "value": metric_value,
            }
        )
    return rows
