import math
import os
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional
import yaml

import mlflow.artifacts
import torch
from mlflow import MlflowClient
from mlflow.entities import RunTag
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
import numpy as np
try:
    import ray
    from ray import tune
except ModuleNotFoundError:  # pragma: no cover - exercised only in Ray-free environments
    class _LocalRemoteFunction:
        def __init__(self, func: Any) -> None:
            self._func = func

        def options(self, **kwargs: Any) -> "_LocalRemoteFunction":
            del kwargs
            return self

        def remote(self, *args: Any, **kwargs: Any) -> Any:
            return self._func(*args, **kwargs)

    class _LocalRayShim:
        @staticmethod
        def remote(func: Optional[Any] = None, **kwargs: Any) -> Any:
            del kwargs
            if func is None:
                return lambda inner: _LocalRemoteFunction(inner)
            return _LocalRemoteFunction(func)

    class _LocalTuneShim:
        class Checkpoint:
            pass

        @staticmethod
        def get_context() -> Any:
            raise RuntimeError("Ray Tune context is unavailable in local execution.")

    ray = _LocalRayShim()  # type: ignore[assignment]
    tune = _LocalTuneShim()  # type: ignore[assignment]

from tqdm import tqdm

from .benchmark_utils import BenchmarkCLI, objective_threshold
from .benchmark_utils.arrow_runtime import build_dataset_source
from .data_prep.manifest import PreparedDatasetManifest
from .data_prep.runtime import manifest_path_for_stage_from_args, prepare_model_dataset_manifests
from .benchmark_utils.style_transfer import StyleTransfer
from .metric_utils.evaluation_postprocessing import resolve_evaluation_postprocessing_config
from .tune.pruning import (
    SeedAwareMetricReporter,
    TrialPrunedError,
    build_reduced_hpo_seed_context,
    has_explicit_hpo_seeds,
    resolve_hpo_seeds,
)
try:
    from .tune.runtime_allowlist import apply_cuda_visible_devices_for_lease, get_runtime_gpu_allowlist_actor
except ModuleNotFoundError:  # pragma: no cover - exercised only in Ray-free environments
    def apply_cuda_visible_devices_for_lease(lease: Any) -> None:
        del lease

    def get_runtime_gpu_allowlist_actor() -> Any:
        raise ValueError("Runtime GPU allowlist actor is unavailable without Ray.")
from .tune_constants import (
    CKPTS_DIR,
    TORCH_FLOAT_PREC,
    CPU_ONLY_MODELS,
    NUM_CPUS_PER_JOB,
    TUNE_PROGRESS_ATTR,
    uses_lightning_optimizer,
)
from .tune.mlflow_datasets import (
    DatasetLineageCache,
    LoggedDatasetInfo,
    datamodule_split_lineage,
    get_or_create_logged_dataset,
    log_dataset_input,
)
from .tune.mlflow_tracking import build_benchmark_tags
from .tune.mlflow_tracking import maybe_active_run
from .tune_utils import (
    InvalidHyperparameterConfiguration,
    create_extra_params_for_lightning,
    save_metadata,
    save_seed_artifacts,
    setup_seed_mlflow,
    validate_model_hparams,
)

logger = logging.getLogger(__name__)
BYTES_PER_GIB = 1024**3

RAY_TRIAL_PROGRESS_FILENAME = "trial_progress.json"
_XLA_DEVICE_ENV_KEYS = (
    "PJRT_DEVICE",
    "TPU_NAME",
    "TPU_VISIBLE_CHIPS",
    "XRT_TPU_CONFIG",
    "CLOUD_TPU_TASK_ID",
)


@dataclass(frozen=True)
class StagePredictionResult:
    metrics: Dict[str, Any]
    anomaly_scores: np.ndarray
    anomaly_labels: np.ndarray
    dataset_info: Optional[LoggedDatasetInfo]


def _should_save_checkpoint_artifacts(args: Namespace) -> bool:
    return bool(getattr(args, "save_checkpoints", False))


def _safe_xla_device() -> Optional[torch.device]:
    try:
        import torch_xla.core.xla_model as xm
    except Exception:
        return None

    try:
        return xm.xla_device()
    except Exception as exc:
        logger.warning(
            "XLA/TPU environment was detected, but no XLA device could be resolved for style transfer; "
            "falling back to CPU. Error: %s",
            exc,
        )
        return None


def _resolve_style_transfer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    xla_requested = (
        any(os.getenv(key) for key in _XLA_DEVICE_ENV_KEYS)
        or any(name == "torch_xla" or name.startswith("torch_xla.") for name in sys.modules)
    )
    if xla_requested:
        xla_device = _safe_xla_device()
        if xla_device is not None:
            return xla_device
        logger.warning(
            "Style-transfer cache generation is using CPU because XLA/TPU device discovery was unavailable."
        )
    return torch.device("cpu")


def _runtime_checkpoint_root(config: Dict[str, Any], args: Namespace, seed: int) -> Path:
    configured_root = config.get("runtime_checkpoint_root")
    if configured_root is not None:
        return Path(configured_root)

    temp_dir = Path(getattr(args, "temp_dir", tempfile.gettempdir()))
    return (
        temp_dir
        / "noboom_runtime_checkpoints"
        / str(getattr(args, "timestamp", "run"))
        / str(config.get("trial_name", "trial"))
        / f"seed_{seed}"
    )


def _log_seed_artifacts_to_trial_run(
    *,
    trial_run_id: Optional[str],
    seed: int,
    seed_ckpt_dir: Path,
    include_checkpoints: bool = False,
) -> None:
    if trial_run_id is None:
        logger.debug(
            "Skipping direct seed artifact upload because trial_run_id is not available for seed=%d.",
            seed,
        )
        return

    artifact_path = CKPTS_DIR / f"seed_{seed}"
    logger.info(
        "Uploading seed artifacts from '%s' to MLflow trial run '%s' at '%s'.",
        seed_ckpt_dir,
        trial_run_id,
        artifact_path,
    )
    if include_checkpoints:
        mlflow.log_artifacts(
            str(seed_ckpt_dir),
            artifact_path=str(artifact_path),
            run_id=trial_run_id,
        )
        return

    for file_path in sorted(seed_ckpt_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix == ".ckpt":
            continue
        relative_parent = file_path.relative_to(seed_ckpt_dir).parent
        file_artifact_path = artifact_path
        if str(relative_parent) != ".":
            file_artifact_path = artifact_path / relative_parent
        mlflow.log_artifact(
            str(file_path),
            artifact_path=str(file_artifact_path),
            run_id=trial_run_id,
        )


def _log_trial_metadata_to_mlflow(
    *,
    trial_run_id: Optional[str],
    artifact_dir: Path,
) -> None:
    if trial_run_id is None:
        logger.debug("Skipping trial metadata upload because trial_run_id is not available.")
        return

    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        logger.debug(
            "Skipping trial metadata upload because '%s' does not exist.",
            metadata_path,
        )
        return

    logger.info(
        "Uploading trial metadata '%s' to MLflow run '%s'.",
        metadata_path,
        trial_run_id,
    )
    mlflow.log_artifact(str(metadata_path), run_id=trial_run_id)


def _prepared_manifest_path(args: Namespace, stage_data_source: str) -> str:
    return manifest_path_for_stage_from_args(args, stage_data_source)


def _load_stage_manifest(args: Namespace, stage_data_source: str) -> PreparedDatasetManifest:
    return PreparedDatasetManifest.load(_prepared_manifest_path(args, stage_data_source))


def _find_last_checkpoint(root: Path) -> Optional[Path]:
    if root.is_file():
        return root if root.suffix == ".ckpt" else None

    for candidate in (root / "last.ckpt", root / "last" / "last.ckpt"):
        if candidate.exists():
            return candidate
    return None


def _resolve_last_checkpoint_path(
    *,
    local_root: Optional[Path],
    run_id: Optional[str],
    artifact_path: Optional[str],
    required: bool,
    purpose: str,
) -> Optional[Path]:
    searched_locations: list[str] = []

    if local_root is not None:
        local_checkpoint = _find_last_checkpoint(local_root)
        if local_checkpoint is not None:
            return local_checkpoint
        searched_locations.append(f"local path '{local_root}'")

    if run_id is not None and artifact_path is not None:
        searched_locations.append(f"MLflow run '{run_id}' artifact '{artifact_path}'")
        try:
            downloaded_root = Path(
                mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
            )
        except mlflow.MlflowException as exc:
            if required:
                raise FileNotFoundError(
                    f"Could not resolve last checkpoint for {purpose}. "
                    f"Searched {', '.join(searched_locations)}."
                ) from exc
            logger.warning(
                "Skipping checkpoint restore for %s because artifact '%s' was not available on run '%s': %s",
                purpose,
                artifact_path,
                run_id,
                exc,
            )
            return None

        downloaded_checkpoint = _find_last_checkpoint(downloaded_root)
        if downloaded_checkpoint is not None:
            return downloaded_checkpoint
        searched_locations.append(f"downloaded path '{downloaded_root}'")

    if required:
        raise FileNotFoundError(
            f"Could not resolve last checkpoint for {purpose}. "
            f"Searched {', '.join(searched_locations)}."
        )
    return None


def _serialize_metric_history(
    metric_history: Dict[str, list],
) -> Dict[str, list[float]]:
    return {
        str(name): [float(value) for value in values]
        for name, values in metric_history.items()
    }


def _build_trial_restore_state(
    *,
    completed_seeds: list[int],
    metric_history: Dict[str, list],
    best_metric_history: Dict[str, list],
    last_payload: Dict[str, Any],
    trial_run_id: str,
) -> Dict[str, Any]:
    return {
        "completed_seeds": [int(seed) for seed in completed_seeds],
        "metric_history": _serialize_metric_history(metric_history),
        "best_metric_history": _serialize_metric_history(best_metric_history),
        "last_payload": dict(last_payload),
        "trial_run_id": str(trial_run_id),
    }


def _load_trial_restore_state(
    checkpoint: Optional[tune.Checkpoint],
) -> Dict[str, Any]:
    if checkpoint is None:
        return {}

    with checkpoint.as_directory() as checkpoint_dir:
        state_path = Path(checkpoint_dir) / RAY_TRIAL_PROGRESS_FILENAME
        if not state_path.exists():
            logger.warning(
                "Ray Tune checkpoint '%s' did not include '%s'; starting the trial from scratch.",
                checkpoint_dir,
                RAY_TRIAL_PROGRESS_FILENAME,
            )
            return {}
        return json.loads(state_path.read_text(encoding="utf-8"))


def _build_trial_checkpoint(
    *,
    trial_root_dir: Path,
    completed_seeds: list[int],
    metric_history: Dict[str, list],
    best_metric_history: Dict[str, list],
    last_payload: Dict[str, Any],
    trial_run_id: str,
) -> tune.Checkpoint:
    restore_state = _build_trial_restore_state(
        completed_seeds=completed_seeds,
        metric_history=metric_history,
        best_metric_history=best_metric_history,
        last_payload=last_payload,
        trial_run_id=trial_run_id,
    )

    checkpoint_dir = trial_root_dir / ".ray_trial_progress"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / RAY_TRIAL_PROGRESS_FILENAME
    checkpoint_path.write_text(json.dumps(restore_state, sort_keys=True), encoding="utf-8")
    return tune.Checkpoint.from_directory(checkpoint_dir)


def _unpack_run_seed_result(
    result: tuple[Any, ...],
) -> tuple[
    Dict[str, Any],
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
    str,
    Optional[LoggedDatasetInfo],
]:
    if len(result) == 6:
        return result
    if len(result) == 5:
        raw_metrics, anomaly_scores, anomaly_labels, extra_metrics, seed_run_id = result
        return raw_metrics, anomaly_scores, anomaly_labels, extra_metrics, seed_run_id, None
    raise ValueError(f"Unexpected run_seed result length: {len(result)}")


def _seed_run_name(trial_name: str, seed: int) -> str:
    return f"{trial_name}__seed_{seed}"


def _stage_run_name(trial_name: str, seed: int, stage: str) -> str:
    return f"{_seed_run_name(trial_name, seed)}__{stage}"


def _invalid_trial_metric_value(direction: Optional[str]) -> float:
    return -1e18 if direction == "maximize" else 1e18


def _build_invalid_trial_payload(
    *,
    args: Namespace,
    trial_run_id: str,
    reason: str,
) -> Dict[str, Any]:
    tune_metric = args.study_params["tune_metric"]
    invalid_metric_value = _invalid_trial_metric_value(args.study_params.get("direction"))
    payload = {
        tune_metric: invalid_metric_value,
        f"best_{tune_metric}": invalid_metric_value,
        f"theory_best_{tune_metric}": invalid_metric_value,
        "completed_seed_count": 0,
        "n_seeds": 0,
        "mlflow_run_id": trial_run_id,
        TUNE_PROGRESS_ATTR: 0,
        "invalid_config": True,
        "invalid_config_reason": reason,
    }
    payload.update(build_reduced_hpo_seed_context(args.study_params))
    if tune_metric != "alarm_score":
        payload[f"mean_{tune_metric}"] = invalid_metric_value
    return payload


def _log_stage_dataset_lineage(
    *,
    client: MlflowClient,
    dataset_lineage_cache: Optional[DatasetLineageCache],
    args: Namespace,
    run_id: str,
    datamodule: Any,
    stage: str,
) -> None:
    if not getattr(args, "mlflow_enable_dataset_tracking", False) or dataset_lineage_cache is None:
        return

    datamodule_split_lineage(
        client,
        dataset_lineage_cache,
        run_id=run_id,
        datamodule=datamodule,
        requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
        split_name="train",
        context="training",
        source_uri=getattr(args, "datasource", None),
        tags={"stage": stage, "split": "train"},
    )
    datamodule_split_lineage(
        client,
        dataset_lineage_cache,
        run_id=run_id,
        datamodule=datamodule,
        requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
        split_name="val",
        context="validation",
        source_uri=getattr(args, "datasource", None),
        tags={"stage": stage, "split": "val"},
    )


def _cleanup_stage_run_artifacts(stage_run_id: str) -> None:
    try:
        stage_run = mlflow.get_run(run_id=stage_run_id)
        repository = get_artifact_repository(stage_run.info.artifact_uri)
        for entry in repository.list_artifacts():
            if entry.is_dir and "epoch" in entry.path:
                repository.delete_artifacts(entry.path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(exc)


def _cleanup_seed_stage_artifacts(seed_run_id: str, stage: str) -> None:
    try:
        run = mlflow.get_run(seed_run_id)
        repository = get_artifact_repository(run.info.artifact_uri)
        repository.delete_artifacts(f"{stage}/last/last.ckpt")
        for entry in repository.list_artifacts(stage):
            if entry.is_dir and "epoch" in entry.path:
                repository.delete_artifacts(entry.path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(exc)


def _reset_model_state_for_checkpoint_restore(model: Any) -> None:
    compiled_network = getattr(model, "_compiled_network", None)
    if compiled_network is None:
        return

    logger.info("Clearing compiled network wrapper before checkpoint restore.")
    model._compiled_network = None
    if hasattr(model, "_compile_invocation_count"):
        model._compile_invocation_count = 0


def _predict_with_cli(
    *,
    cli: BenchmarkCLI,
    args: Namespace,
    stage: str,
    seed: int,
    seed_run_id: str,
    seed_ckpt_dir: Path,
    predict_ckpt_path: Optional[str],
    dataset_lineage_cache: Optional[DatasetLineageCache],
) -> StagePredictionResult:
    trainer = cli.trainer
    logger.info("Running final predict for seed=%d, stage=%s.", seed, stage)
    if stage == "synthetic" and cli.model.style_transfer is not None:
        predict_device = _resolve_style_transfer_device()
        cache_path = cli.datamodule.enable_styled_predict_cache(
            cli.model.style_transfer,
            device=predict_device,
        )
        logger.info("Using styled predict cache '%s' for synthetic prediction.", cache_path)
        cli.model.style_transfer = None
        cli.model.use_test_style_transfer = False
    elif stage == "synthetic":
        logger.info(
            "Skipping styled predict cache creation for synthetic prediction because test style transfer "
            "is disabled or no style_transfer_ckpt_path is configured for model '%s'.",
            args.model_name,
        )

    if predict_ckpt_path is not None:
        _reset_model_state_for_checkpoint_restore(cli.model)

    trainer.predict(cli.model, datamodule=cli.datamodule, ckpt_path=predict_ckpt_path)
    scaler_path = seed_ckpt_dir / "scaler.pkl"
    if getattr(cli.datamodule, "scaler", None) is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as handle:
            pickle.dump(cli.datamodule.scaler, handle, pickle.HIGHEST_PROTOCOL)

    dataset_info = None
    if getattr(args, "mlflow_enable_dataset_tracking", False) and dataset_lineage_cache is not None:
        dataset_info = datamodule_split_lineage(
            MlflowClient(args.tracking_uri),
            dataset_lineage_cache,
            run_id=seed_run_id,
            datamodule=cli.datamodule,
            requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
            split_name="predict",
            context="testing",
            source_uri=getattr(args, "datasource", None),
            tags={"stage": stage, "split": "predict"},
        )

    return StagePredictionResult(
        metrics=cli.model.metrics,
        anomaly_scores=cli.model.anomaly_scores,
        anomaly_labels=cli.model.anomaly_labels,
        dataset_info=dataset_info,
    )

def _dataset_memory_lookup_names(args: Namespace) -> list[str]:
    names = []
    for attr in ("requested_dataset_name", "dataset_name"):
        value = getattr(args, attr, None)
        if isinstance(value, str) and value not in names:
            names.append(value)
        if isinstance(value, str) and value.endswith("_tsst"):
            base_value = value[: -len("_tsst")]
            if base_value not in names:
                names.append(base_value)
    return names


def _trial_memory_gib_for_dataset(args: Namespace) -> Optional[float]:
    study_params = getattr(args, "study_params", None)
    if not isinstance(study_params, Mapping):
        return None

    memory_by_dataset = study_params.get("trial_memory_gib_by_dataset")
    if not isinstance(memory_by_dataset, Mapping):
        return None

    memory_gib = None
    for dataset_name in _dataset_memory_lookup_names(args):
        if dataset_name in memory_by_dataset:
            memory_gib = memory_by_dataset[dataset_name]
            break
    if memory_gib is None:
        memory_gib = memory_by_dataset.get("default")
    if memory_gib is None:
        return None

    return _validate_trial_memory_gib(
        memory_gib,
        setting_name="study.trial_memory_gib_by_dataset",
        args=args,
    )


def _validate_trial_memory_gib(
    value: Any,
    *,
    setting_name: str,
    args: Namespace,
) -> float:
    memory_gib = value
    memory_gib = float(memory_gib)
    if memory_gib <= 0:
        raise ValueError(
            f"{setting_name} must contain positive GiB values; "
            f"got {memory_gib} for dataset '{getattr(args, 'dataset_name', '')}'."
        )
    return memory_gib


def _trial_memory_bytes_for_dataset(args: Namespace) -> Optional[int]:
    memory_gib = _trial_memory_gib_for_dataset(args)
    if memory_gib is None:
        return None
    return int(memory_gib * BYTES_PER_GIB)


def tune_resource_request_dict(
    args: Namespace,
    *,
    include_gpu: bool = True,
) -> Dict[str, Any]:
    """Return Ray Tune resource hints that respect CPU-only baselines.

    Args:
        args (Namespace): Parsed arguments with model and resource settings.

    Returns:
        Dict[str, Any]: Resource request for the direct Ray Tune trainable.

    Raises:
        ValueError: If a GPU-backed model resolves to more than one GPU.
    """
    requires_gpu = args.model_name not in CPU_ONLY_MODELS

    with open(Path(args.config_dir) / "models/common.yaml", "r", encoding="utf-8") as f:
        common_params = yaml.safe_load(f)
        cpu_per_job = max(NUM_CPUS_PER_JOB, int(common_params["data"].get("num_workers", 1)))

    gpu_per_job = args.gpus_per_run
    if not requires_gpu:
        gpu_per_job = 0
    elif gpu_per_job < 1:
        if "physdiff" in args.model_name:
            gpu_per_job = 0.5
        elif "industry_process" in args.dataset_name and "neutralad" in args.model_name:
            gpu_per_job = 1.0
        elif "industry_process" in args.dataset_name:
            gpu_per_job = min(gpu_per_job * 4, 0.5)
        else:
            if "timesnet" in args.model_name:
                gpu_per_job = 0.5
            elif "lstm" in args.model_name:
                gpu_per_job = min(gpu_per_job * 2, 0.3)
            elif "neutralad" in args.model_name:
                gpu_per_job = min(gpu_per_job * 2, 0.5)
    else:
        gpu_per_job = float(math.ceil(gpu_per_job))

    if requires_gpu and gpu_per_job > 1:
        raise ValueError(
            f"Resolved GPU request for model '{args.model_name}' is {gpu_per_job}, "
            "but the tuning path only supports a single GPU per trial."
        )

    if "eif" in args.model_name:
        with open(Path(args.config_dir) / "models/eif.yaml", "r", encoding="utf-8") as f:
            model_params = yaml.safe_load(f)
            cpu_per_job = max(cpu_per_job, model_params["model"]["detector"]["init_args"].get("n_jobs", 0))
    elif "threshold" in args.model_name:
        cpu_per_job = 4

    resources = {"CPU": cpu_per_job + 1}
    if include_gpu:
        resources["GPU"] = gpu_per_job
    if "industry_process" in args.dataset_name and str(args.model_name) in {"gmmhmm", "hmm", "kan_ad"}:
        resources["exclusive"] = 1.0
    elif "physdiff" in args.model_name:
        resources["exclusive"] = 0.001
    elif "eif" in args.model_name and "industry_process" in args.dataset_name:
        resources["exclusive"] = 0.5
    else:
        resources["exclusive"] = 0.001
    trial_memory_bytes = _trial_memory_bytes_for_dataset(args)
    if trial_memory_bytes is not None:
        resources["memory"] = trial_memory_bytes

    logger.info(
        "Resource request for model '%s': requires_gpu=%s, resources=%s",
        args.model_name,
        requires_gpu,
        resources,
    )
    return resources


def train_func(config: Dict[str, Any]) -> Dict[str, StagePredictionResult]:
    logger_params = config["logger_params"]
    hp_params = config["hp_params"]
    seed = config["seed"]
    args = config["args"]
    ckpt_dir = config["ckpt_dir"]
    seed_run_id = config["seed_run_id"]
    dataset_lineage_cache = config.get("dataset_lineage_cache")
    torch.set_float32_matmul_precision(TORCH_FLOAT_PREC)

    ckpt_path = None
    stages = ["real"]
    if args.use_generated:
        stages.insert(0, "synthetic")

    lr_scale = 1.0
    max_epochs = None
    stage_results: Dict[str, StagePredictionResult] = {}
    client = MlflowClient(args.tracking_uri)
    save_checkpoint_artifacts = _should_save_checkpoint_artifacts(args)
    runtime_checkpoint_root = _runtime_checkpoint_root(config, args, seed)

    try:
        with tqdm(enumerate(stages), total=len(stages), desc="Stage", unit="stage", dynamic_ncols=True) as pbar:
            for i, stage in pbar:
                pbar.set_description_str(f"Stage: {stage}")
                restore_ckpt_path = None
                stage_tags = {
                    **build_benchmark_tags(
                        timestamp=args.timestamp,
                        run_level="stage",
                        model_name=args.model_name,
                        dataset_name=args.dataset_name,
                        requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
                        run_mode="tune" if args.tune else "single_train",
                        deployment_mode=getattr(args, "deployment_mode", None),
                        parent_run_id=seed_run_id,
                        seed=seed,
                        stage=stage,
                    ),
                }
                stage_artifact_ckpt_dir = Path(ckpt_dir) / stage
                stage_runtime_ckpt_dir = (
                    stage_artifact_ckpt_dir
                    if save_checkpoint_artifacts
                    else runtime_checkpoint_root / stage
                )
                restore_checkpoint = _resolve_last_checkpoint_path(
                    local_root=stage_runtime_ckpt_dir,
                    run_id=None,
                    artifact_path=None,
                    required=False,
                    purpose=f"resume stage '{stage}' for seed {seed}",
                )
                if restore_checkpoint is not None:
                    restore_ckpt_path = str(restore_checkpoint)
                stage_run_name = _stage_run_name(config["trial_name"], seed, stage)
                stage_run = client.create_run(args.experiment_id, tags=stage_tags, run_name=stage_run_name)
                stage_run_id = stage_run.info.run_id

                logger_params["run_id"] = stage_run_id
                stage_manifest_path = _prepared_manifest_path(args, stage)

                hp_params["data.data_source"] = stage
                hp_params["data.data_manifest_path"] = stage_manifest_path
                hp_params["model.ckpt_path"] = ckpt_path
                hp_params["model.save_hparams"] = True
                if uses_lightning_optimizer(args.model_name):
                    hp_params["model.optimizer.init_args.lr"] = 1e-4
                hp_path = create_extra_params_for_lightning(args, logger_params, seed, hp_params)
                cli = BenchmarkCLI(
                    args.model_name,
                    args.dataset_name,
                    data_manifest_path=stage_manifest_path,
                    config_dir=args.config_dir,
                    hp_cfg=hp_path,
                    ckpt_dir=stage_runtime_ckpt_dir,
                    requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
                    lr_scale=lr_scale,
                    max_epochs=max_epochs
                )
                Path(hp_path).unlink()

                if (
                    stage == "synthetic"
                    and cli.model.use_test_style_transfer
                    and cli.model.style_transfer_ckpt_path is not None
                ):
                    predict_device = _resolve_style_transfer_device()
                    cache_style_transfer = StyleTransfer(
                        cli.model.style_transfer_ckpt_path,
                        batch_dim=cli.datamodule.batch_dim,
                    )
                    cache_path = cli.datamodule.enable_styled_predict_cache(
                        cache_style_transfer,
                        device=predict_device,
                        load_dataset=False,
                    )
                    logger.info("Prepared styled predict cache '%s' before synthetic training.", cache_path)
                elif stage == "synthetic":
                    logger.info(
                        "Skipping styled predict cache preparation before synthetic training because "
                        "test style transfer is disabled or no style_transfer_ckpt_path is configured for model '%s'.",
                        args.model_name,
                    )

                trainer = cli.trainer
                with maybe_active_run(
                    run_id=stage_run_id,
                    enable_system_metrics=bool(getattr(args, "mlflow_enable_system_metrics", False)),
                ):
                    trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=restore_ckpt_path)

                _log_stage_dataset_lineage(
                    client=client,
                    dataset_lineage_cache=dataset_lineage_cache,
                    args=args,
                    run_id=stage_run_id,
                    datamodule=cli.datamodule,
                    stage=stage,
                )

                if args.remove_ckpt:
                    _cleanup_stage_run_artifacts(stage_run_id)

                predict_ckpt_path: Optional[str] = None
                if uses_lightning_optimizer(args.model_name):
                    stage_last_checkpoint = _resolve_last_checkpoint_path(
                        local_root=stage_runtime_ckpt_dir,
                        run_id=None,
                        artifact_path=None,
                        required=True,
                        purpose=f"completed stage '{stage}' for seed {seed}",
                    )
                    predict_ckpt_path = str(stage_last_checkpoint)

                stage_results[stage] = _predict_with_cli(
                    cli=cli,
                    args=args,
                    stage=stage,
                    seed=seed,
                    seed_run_id=seed_run_id,
                    seed_ckpt_dir=Path(ckpt_dir),
                    predict_ckpt_path=predict_ckpt_path,
                    dataset_lineage_cache=dataset_lineage_cache,
                )

                if uses_lightning_optimizer(args.model_name) and predict_ckpt_path is not None:
                    if save_checkpoint_artifacts:
                        stage_last_checkpoint = Path(predict_ckpt_path)
                        mlflow.log_artifacts(
                            str(stage_last_checkpoint.parent),
                            artifact_path=f"{stage}/last",
                            run_id=seed_run_id,
                        )
                        if args.remove_ckpt:
                            _cleanup_seed_stage_artifacts(seed_run_id, stage)
                    if i < len(stages) - 1:
                        ckpt_path = predict_ckpt_path
                        lr_scale *= 0.1
                        max_epochs = 20

                client.set_terminated(stage_run_id)
    finally:
        if not save_checkpoint_artifacts:
            shutil.rmtree(runtime_checkpoint_root, ignore_errors=True)

    return stage_results


def predict_func(
    config: Dict[str, Any],
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray, Optional[LoggedDatasetInfo]]:
    logger_params = config["logger_params"]
    hp_params = config["hp_params"]
    seed = config["seed"]
    seed_run_id = config["seed_run_id"]
    args = config["args"]
    stage = config["stage"]
    dataset_lineage_cache = config.get("dataset_lineage_cache")

    torch.set_float32_matmul_precision(TORCH_FLOAT_PREC)
    ckpt_dir = None
    predict_ckpt_path: Optional[str] = "last" if args.model_name in CPU_ONLY_MODELS else None
    if uses_lightning_optimizer(args.model_name):
        local_stage_ckpt_dir = Path(config["ckpt_dir"]) / stage
        resolved_checkpoint = _resolve_last_checkpoint_path(
            local_root=local_stage_ckpt_dir,
            run_id=seed_run_id,
            artifact_path=f"{stage}/last",
            required=True,
            purpose=f"predict stage '{stage}' for seed {seed}",
        )
        ckpt_dir = resolved_checkpoint.parent
        predict_ckpt_path = str(resolved_checkpoint)
    hp_params["data.data_source"] = stage
    stage_manifest_path = _prepared_manifest_path(args, stage)
    hp_params["data.data_manifest_path"] = stage_manifest_path
    hp_path = create_extra_params_for_lightning(args, logger_params, seed, hp_params)

    cli = BenchmarkCLI(
        args.model_name,
        args.dataset_name,
        data_manifest_path=stage_manifest_path,
        config_dir=args.config_dir,
        hp_cfg=hp_path,
        ckpt_dir=ckpt_dir,
        requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
    )

    Path(hp_path).unlink()

    trainer = cli.trainer
    logger.info("Running final predict for seed=%d, stage=%s.", seed, stage)
    if stage == "synthetic" and cli.model.style_transfer is not None:
        predict_device = _resolve_style_transfer_device()
        cache_path = cli.datamodule.enable_styled_predict_cache(
            cli.model.style_transfer,
            device=predict_device,
        )
        logger.info("Using styled predict cache '%s' for synthetic prediction.", cache_path)
        cli.model.style_transfer = None
        cli.model.use_test_style_transfer = False
    elif stage == "synthetic":
        logger.info(
            "Skipping styled predict cache creation for synthetic prediction because test style transfer "
            "is disabled or no style_transfer_ckpt_path is configured for model '%s'.",
            args.model_name,
        )
    trainer.predict(cli.model, datamodule=cli.datamodule, ckpt_path=predict_ckpt_path)
    scaler_path = Path(config["ckpt_dir"]) / "scaler.pkl"
    if getattr(cli.datamodule, "scaler", None) is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as handle:
            pickle.dump(cli.datamodule.scaler, handle, pickle.HIGHEST_PROTOCOL)

    dataset_info = None
    if getattr(args, "mlflow_enable_dataset_tracking", False) and dataset_lineage_cache is not None:
        dataset_info = datamodule_split_lineage(
            MlflowClient(args.tracking_uri),
            dataset_lineage_cache,
            run_id=seed_run_id,
            datamodule=cli.datamodule,
            requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
            split_name="predict",
            context="testing",
            source_uri=getattr(args, "datasource", None),
            tags={"stage": stage, "split": "predict"},
        )

    if args.remove_ckpt:
        try:
            run = mlflow.get_run(seed_run_id)
            repository = get_artifact_repository(run.info.artifact_uri)
            repository.delete_artifacts(f"{stage}/last/last.ckpt")
            for entry in repository.list_artifacts(stage):
                if entry.is_dir and "epoch" in entry.path:
                    repository.delete_artifacts(entry.path)
        except Exception as e:
            logger.warning(e)

    return cli.model.metrics, cli.model.anomaly_scores, cli.model.anomaly_labels, dataset_info


def run_pytorch_objective(
    hp_params: Dict[str, Any],
    args: Namespace,
    ckpt_dir: Path,
    seed: int,
    seed_run_id: str,
    trial_name: Optional[str] = None,
    trial_metric_reporter: Optional[SeedAwareMetricReporter] = None,
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray, Optional[LoggedDatasetInfo]]:
    """Run a PyTorch-based objective for a specific seed and trial.

    Args:
        args (Namespace): Parsed CLI arguments for the trial.
        ckpt_dir (Path): Checkpoint directory for the seed.
        seed (int): Seed value used for the trial.
        seed_run_id (str): MLflow run ID for the seed.
        hp_params (Dict[str, Any]): Hyperparameters for the model.

    Returns:
        tuple: Seed metrics, anomaly scores, labels, model ID, dataset name, dataset digest.

    Raises:
        FileNotFoundError: If expected checkpoint files are missing.

    Side Effects:
        Trains models, logs to MLflow, and reads/writes checkpoints.
    """
    experiment_id = args.experiment_id
    tracking_uri = args.tracking_uri
    torch.set_float32_matmul_precision(TORCH_FLOAT_PREC)

    logger.info(
        "Starting PyTorch objective for model=%s, dataset=%s, seed=%d.",
        args.model_name,
        args.dataset_name,
        seed,
    )

    client = MlflowClient(tracking_uri)
    experiment_name = client.get_experiment(experiment_id).name

    logger_params = {
        "experiment_name": experiment_name,
        "run_id": seed_run_id,
        "tracking_uri": tracking_uri,
    }

    tune_resource_request_dict(args)

    hp_params = hp_params.copy()
    hp_params["data.gen_to_org_factor"] = -1

    config = {
        "logger_params": logger_params,
        "hp_params": hp_params,
        "seed": seed,
        "seed_run_id": seed_run_id,
        "args": args,
        "ckpt_dir": ckpt_dir,
        "trial_name": trial_name or seed_run_id,
        "trial_metric_reporter": trial_metric_reporter,
        "dataset_lineage_cache": DatasetLineageCache(),
    }

    stage_results = train_func(config)
    stages = ["real"]
    if args.use_generated:
        stages.insert(0, "synthetic")
    total_metrics: Dict[str, Any] = {}
    for stage in stages:
        result = stage_results[stage]
        metrics = result.metrics
        if stage != "real":
            metrics = {f"{k}__{stage}": v for k, v in metrics.items()}
        total_metrics |= metrics
    final_stage_result = stage_results[stages[-1]]
    anomaly_scores = final_stage_result.anomaly_scores
    anomaly_labels = final_stage_result.anomaly_labels
    dataset_info = final_stage_result.dataset_info
    logger.debug("Seed metrics: %s", total_metrics)
    return total_metrics, anomaly_scores, anomaly_labels, dataset_info


def log_threshold_dataset_lineage(
    *,
    args: Namespace,
    client: MlflowClient,
    run_id: str,
    cache: DatasetLineageCache,
) -> Optional[LoggedDatasetInfo]:
    if not getattr(args, "mlflow_enable_dataset_tracking", False):
        return None

    manifest = _load_stage_manifest(args, "real")
    dataset_source = build_dataset_source(manifest, mode="test")
    dataset_info = get_or_create_logged_dataset(
        cache,
        cache_key=f"{run_id}::threshold::predict",
        dataset_source=dataset_source,
        requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
        split_name="predict",
        data_source="real",
        source_uri=getattr(args, "datasource", None),
        context="testing",
    )
    log_dataset_input(
        client,
        run_id,
        dataset_info,
        tags={"split": "predict", "model_family": "threshold"},
    )
    return dataset_info


def run_seed(
    hp_params: Dict[str, Any],
    args: Namespace,
    seed_ckpt_dir: Path,
    seed: int,
    trial_run_id: Optional[str],
    trial_name: Optional[str],
    trial_metric_reporter: Optional[SeedAwareMetricReporter] = None,
) -> tuple[
    Dict[str, Any],
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
    str,
    Optional[LoggedDatasetInfo],
]:
    """Execute the model objective for a single seed.

    Args:
        args (Namespace): Parsed CLI arguments for the trial.
        hp_params (Dict[str, Any]): Hyperparameters to evaluate.
        seed_ckpt_dir (Path): Checkpoint directory for the seed.
        seed (int): Seed value used for the run.
        trial_run_id (str | None): MLflow trial run ID.
        trial_name (str | None): Trial name for MLflow tags.

    Returns:
        tuple: Raw metrics, anomaly scores, anomaly labels, extra metrics, seed run ID,
        model ID, dataset name, dataset digest.

    Raises:
        RuntimeError: If MLflow parameters are missing for tuning runs.
    """

    client = MlflowClient(args.tracking_uri)
    experiment_id = args.experiment_id
    tune_metric = args.study_params["tune_metric"]

    if client is None or experiment_id is None or trial_run_id is None or trial_name is None:
        raise RuntimeError(
            "MLflow client/experiment/trial info must be provided when tune_model=True."
        )
    seed_run_id = setup_seed_mlflow(
        args, hp_params, client, experiment_id, trial_name, trial_run_id, seed
    )

    extra_metrics: Dict[str, Any] = {}
    dataset_lineage_cache = DatasetLineageCache()
    dataset_info = None
    postprocessing_config = resolve_evaluation_postprocessing_config(
        args.study_params.get("evaluation_postprocessing")
    )
    client.set_tag(seed_run_id, "evaluation_postprocessing_enabled", str(postprocessing_config.enabled).lower())
    client.set_tag(
        seed_run_id,
        "evaluation_postprocessing_config",
        json.dumps(postprocessing_config.to_dict(), sort_keys=True),
    )
    try:
        if args.model_name == "threshold":
            logger.info("Using threshold objective for model '%s'.", args.model_name)
            with maybe_active_run(
                run_id=seed_run_id,
                enable_system_metrics=bool(getattr(args, "mlflow_enable_system_metrics", False)),
            ):
                raw_metrics, anomaly_scores, anomaly_labels, threshold, feature = objective_threshold(
                    args,
                    hp_params,
                    args.study_params["metrics"],
                    tune_metric,
                )
            extra_metrics["threshold"] = threshold
            extra_metrics["feature"] = feature
            dataset_info = log_threshold_dataset_lineage(
                args=args,
                client=client,
                run_id=seed_run_id,
                cache=dataset_lineage_cache,
            )
        else:
            raw_metrics, anomaly_scores, anomaly_labels, dataset_info = run_pytorch_objective(
                hp_params,
                args=args,
                seed=seed,
                ckpt_dir=seed_ckpt_dir,
                seed_run_id=seed_run_id,
                trial_name=trial_name,
                trial_metric_reporter=trial_metric_reporter,
            )
    except TrialPrunedError:
        client.set_terminated(seed_run_id, status="KILLED")
        raise

    return raw_metrics, anomaly_scores, anomaly_labels, extra_metrics, seed_run_id, dataset_info


def _seed_metric_payload(
    raw_metrics: Mapping[str, Any],
    extra_metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    seed_metrics: Dict[str, float] = {}
    for name, metric in raw_metrics.items():
        value, theory_best_value = metric
        seed_metrics[str(name)] = float(value)
        seed_metrics[f"theory_best_{name}"] = float(theory_best_value)

    for name, value in (extra_metrics or {}).items():
        if isinstance(value, (bool, str, bytes)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            seed_metrics[str(name)] = float(value)
    return seed_metrics


def run_seed_task_local(
    hp_params: Dict[str, Any],
    args: Namespace,
    seed: int,
    trial_root_dir: str,
    trial_run_id: str,
    trial_name: str,
) -> Dict[str, Any]:
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)

    seed_ckpt_dir = Path(trial_root_dir) / CKPTS_DIR / f"seed_{seed}"
    (
        raw_metrics,
        anomaly_scores,
        anomaly_labels,
        extra_metrics,
        seed_run_id,
        _dataset_info,
    ) = _unpack_run_seed_result(
        run_seed(
            hp_params,
            args,
            seed_ckpt_dir,
            seed,
            trial_run_id,
            trial_name,
        )
    )
    seed_metrics = _seed_metric_payload(raw_metrics, extra_metrics)
    if seed_run_id is not None and seed_metrics:
        mlflow.log_metrics(seed_metrics, run_id=seed_run_id)

    save_seed_artifacts(
        seed_ckpt_dir,
        anomaly_scores,
        anomaly_labels,
        seed_metrics,
        **extra_metrics,
    )
    _log_seed_artifacts_to_trial_run(
        trial_run_id=trial_run_id,
        seed=seed,
        seed_ckpt_dir=seed_ckpt_dir,
        include_checkpoints=_should_save_checkpoint_artifacts(args),
    )

    if seed_run_id is not None:
        client.set_terminated(seed_run_id)

    return raw_metrics


@ray.remote
def run_seed_task(
    hp_params: Dict[str, Any],
    args: Namespace,
    seed: int,
    trial_root_dir: str,
    trial_run_id: str,
    trial_name: str,
) -> Dict[str, Any]:
    return run_seed_task_local(
        hp_params,
        args,
        seed,
        trial_root_dir,
        trial_run_id,
        trial_name,
    )


def run_tune_trainable(
    hp_params: Dict[str, Any],
    args: Namespace,
) -> Dict[str, Any]:
    """Evaluate a trial configuration across multiple seeds.

    Args:
        hp_params (Dict[str, Any]): Hyperparameters for the trial.
        args (Namespace): Parsed CLI arguments for the trial.
        experiment_id (str | None): MLflow experiment ID. Defaults to None.
        trial_run_id (str | None): MLflow trial run ID. Defaults to None.

    Returns:
        Dict[str, Any]: Aggregated metrics across seeds.

    Side Effects:
        Runs model training/evaluation, logs to MLflow, and writes artifacts.
    """
    ctx = tune.get_context()
    logger.info(
        "Starting evaluate_trial_across_seeds for model=%s, dataset=%s.",
        args.model_name,
        args.dataset_name,
    )
    logger.debug("Hyperparameters: %s", hp_params)

    client = MlflowClient(args.tracking_uri)
    trial_name = ctx.get_trial_name()
    trial_root_dir = Path(ctx.get_trial_dir())
    trial_id = ctx.get_trial_id() if hasattr(ctx, "get_trial_id") else trial_name
    tune_metric = args.study_params["tune_metric"]
    experiment_id = args.experiment_id

    logger.info("Starting Ray Tune trainable with config: %s", hp_params)

    assert args.model_name in trial_name and args.dataset_name in trial_name, (
            args.model_name + " " + args.dataset_name + " " + trial_name
    )
    logger.info("Ray Tune trial name: '%s', trial_dir='%s'.", trial_name, trial_root_dir)

    runs = client.search_runs(
        [experiment_id],
        f'tags.{MLFLOW_RUN_NAME} = "{trial_name}"',
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found for trial_name='{trial_name}'.")
    trial_run_id = runs[0].info.run_id
    logger.info("Found trial run_id=%s for trial_name=%s.", trial_run_id, trial_name)
    try:
        lease = ray.get(get_runtime_gpu_allowlist_actor().activate_lease.remote(trial_id))
        if lease is not None:
            apply_cuda_visible_devices_for_lease(lease)
    except ValueError:
        logger.debug("Runtime GPU allowlist actor is unavailable for trial %s.", trial_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to activate GPU lease for trial %s.", trial_id)

    tags = build_benchmark_tags(
        timestamp=args.timestamp,
        run_level="trial",
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
        run_mode="tune",
        deployment_mode=getattr(args, "deployment_mode", None),
        parent_run_id=args.study_run_id,
    )

    tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
    client.log_batch(trial_run_id, tags=tags_arr)

    artifact_dir = Path(trial_root_dir)
    ckpt_dir = artifact_dir / CKPTS_DIR

    logger.info(
        "Trial root dir: '%s', artifact dir: '%s', ckpt dir: '%s'.",
        trial_root_dir,
        artifact_dir,
        ckpt_dir,
    )

    try:
        validate_model_hparams(
            args.model_name,
            hp_params,
            config_dir=str(Path(getattr(args, "config_dir", ".")).resolve()),
        )
    except InvalidHyperparameterConfiguration as exc:
        reason = str(exc)
        logger.warning("Skipping invalid trial configuration '%s': %s", trial_name, reason)
        client.set_tag(trial_run_id, "invalid_config", "true")
        client.set_tag(trial_run_id, "invalid_config_reason", reason)
        save_metadata(args, artifact_dir, hp_params)
        _log_trial_metadata_to_mlflow(
            trial_run_id=trial_run_id,
            artifact_dir=artifact_dir,
        )
        return _build_invalid_trial_payload(
            args=args,
            trial_run_id=trial_run_id,
            reason=reason,
        )

    args.preparation_hp_params = hp_params
    if getattr(args, "data_manifest_path_explicit", False):
        prepared_manifest_paths = {
            str(key): str(value)
            for key, value in getattr(args, "prepared_manifest_paths", {}).items()
            if value is not None
        }
        if getattr(args, "use_generated", False) and "synthetic" not in prepared_manifest_paths:
            generated_manifest_paths = prepare_model_dataset_manifests(
                dataset_name=args.dataset_name,
                requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
                model_name=args.model_name,
                config_dir=str(Path(getattr(args, "config_dir", ".")).resolve()),
                runtime_config=args.runtime_config,
                use_generated=True,
                hp_params=hp_params,
                verbosity=int(getattr(args, "verbose", 0) or 0),
            )
            prepared_manifest_paths["synthetic"] = generated_manifest_paths["synthetic"]
        args.prepared_manifest_paths = prepared_manifest_paths
    else:
        args.prepared_manifest_paths = prepare_model_dataset_manifests(
            dataset_name=args.dataset_name,
            requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
            model_name=args.model_name,
            config_dir=str(Path(getattr(args, "config_dir", ".")).resolve()),
            runtime_config=args.runtime_config,
            use_generated=args.use_generated,
            hp_params=hp_params,
            verbosity=int(getattr(args, "verbose", 0) or 0),
        )
        args.data_manifest_path = args.prepared_manifest_paths["real"]

    metrics: Dict[str, list] = defaultdict(list)
    theory_best_metrics: Dict[str, list] = defaultdict(list)
    dataset_name = None
    dataset_digest = None
    total_metrics: Dict[str, Any] = {}
    metric_reporter = SeedAwareMetricReporter(
        tune_metric=tune_metric,
        seed_context=build_reduced_hpo_seed_context(args.study_params),
    )
    completed_seeds: list[int] = []
    completed_seed_set: set[int] = set()
    hpo_seeds = resolve_hpo_seeds(args.study_params)
    if has_explicit_hpo_seeds(args.study_params):
        logger.info(
            "Using %d HPO seeds from study.hpo_seeds instead of %d full study seeds.",
            len(hpo_seeds),
            len(args.study_params.get("seeds", [])),
        )

    for step, seed in enumerate(hpo_seeds):
        logger.info("Processing seed=%d.", seed)
        seed_ckpt_dir = Path(ckpt_dir) / f"seed_{seed}"
        logger.info("Seed checkpoint dir: '%s'.", seed_ckpt_dir)

        try:
            (
                raw_metrics,
                anomaly_scores,
                anomaly_labels,
                extra_metrics,
                seed_run_id,
                current_dataset_info,
            ) = _unpack_run_seed_result(
                run_seed(
                    hp_params,
                    args,
                    seed_ckpt_dir,
                    seed,
                    trial_run_id,
                    trial_name,
                    trial_metric_reporter=metric_reporter,
                )
            )
        except TrialPrunedError as exc:
            total_metrics = exc.payload
            logger.info("Trial pruned during seed=%d with payload=%s", seed, total_metrics)
            break

        for name, metric in raw_metrics.items():
            metrics[name].append(metric[0])
            theory_best_metrics[name].append(metric[1])
            # Ensure the theoretical best value is consistent across seeds.
            assert (max(theory_best_metrics[name]) - min(theory_best_metrics[name])) <= np.finfo(
                np.float32
            ).eps, "Theoretical best value differs across seeds"

        seed_metrics = _seed_metric_payload(raw_metrics, extra_metrics)
        logger.debug("Seed metrics: %s", seed_metrics)
        logger.debug("Accumulated metrics: %s", metrics)
        logger.debug("Accumulated theory-best metrics: %s", theory_best_metrics)
        if seed_run_id is not None and seed_metrics:
            mlflow.log_metrics(seed_metrics, run_id=seed_run_id)

        save_seed_artifacts(
            seed_ckpt_dir,
            anomaly_scores,
            anomaly_labels,
            metrics | {f"theory_best_{name}": val for name, val in theory_best_metrics.items()},
            **extra_metrics,
        )
        _log_seed_artifacts_to_trial_run(
            trial_run_id=trial_run_id,
            seed=seed,
            seed_ckpt_dir=seed_ckpt_dir,
            include_checkpoints=_should_save_checkpoint_artifacts(args),
        )

        logger.info("Marking MLflow run '%s' as terminated.", seed_run_id)
        client.set_terminated(seed_run_id)
        metric_reporter.note_completed_seed(float(raw_metrics[tune_metric][0]))
        if seed not in completed_seed_set:
            completed_seeds.append(seed)
            completed_seed_set.add(seed)
        if current_dataset_info is not None:
            dataset_name = current_dataset_info.name
            dataset_digest = current_dataset_info.digest

        total_metrics = metric_reporter.build_completed_payload(
            metric_history=metrics,
            best_metric_history=theory_best_metrics,
            trial_run_id=trial_run_id,
            seed=seed,
            dataset_name=dataset_name,
            dataset_digest=dataset_digest,
        )
        if current_dataset_info is not None:
            total_metrics["dataset_name"] = dataset_name
            total_metrics["dataset_digest"] = dataset_digest

        logger.info("Aggregated metrics after seed=%d: %s", seed, total_metrics)

        if step < len(hpo_seeds) - 1:
            logger.info(
                "Reporting finalized seed metrics to Ray Tune (completed_seed_count=%d).",
                metric_reporter.completed_seed_count,
            )
            try:
                metric_reporter.emit_completed_payload(total_metrics)
            except TrialPrunedError as exc:
                total_metrics = exc.payload
                logger.info("Trial pruned after seed=%d with payload=%s", seed, total_metrics)
                break

    save_metadata(
        args,
        artifact_dir,
        hp_params,
    )
    _log_trial_metadata_to_mlflow(
        trial_run_id=trial_run_id,
        artifact_dir=artifact_dir,
    )

    logger.info("Returning tune metric '%s' = %s.", tune_metric, str(total_metrics))
    return total_metrics
