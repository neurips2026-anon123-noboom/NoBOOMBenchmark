from __future__ import annotations

import functools
import logging.config
import math
from argparse import Namespace
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

from filelock import FileLock
import mlflow
from mlflow import MlflowClient
import optuna
from omegaconf import OmegaConf
try:
    import ray
    from ray import tune
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from ray.tune.callback import Callback
    from ray.tune.experiment import Trial
    from ray.tune.search.optuna import OptunaSearch
except ModuleNotFoundError:  # pragma: no cover - exercised only in Ray-free environments
    ray = None  # type: ignore[assignment]
    tune = None  # type: ignore[assignment]
    NodeAffinitySchedulingStrategy = None  # type: ignore[assignment]
    Trial = Any  # type: ignore[misc, assignment]

    class Callback:  # type: ignore[no-redef]
        pass

    OptunaSearch = None  # type: ignore[assignment]
import numpy as np

from noboom_cluster.noboom_cli_lib.specs import PairResult

from .benchmark_utils import build_sampler, load_config
from .logging import setup_logging
from .config.runtime import RuntimeConfig
from .data_prep.runtime import prepare_model_dataset_manifests
try:
    from .mlflow_logging_utils import MLFlowLoggerCallbackSW
except ModuleNotFoundError:  # pragma: no cover - exercised only in Ray-free environments
    MLFlowLoggerCallbackSW = None  # type: ignore[assignment]
from .model_objectives import (
    _build_invalid_trial_payload,
    _invalid_trial_metric_value,
    _log_trial_metadata_to_mlflow,
    run_seed_task,
    run_seed_task_local,
    run_tune_trainable,
    tune_resource_request_dict,
)
from .tune.mlflow_utils import load_study_result_from_mlflow
from .tune.report_generation import excel_score_table_has_metrics, update_score_excel_one
from .tune_utils import (
    InvalidHyperparameterConfiguration,
    build_tune_scheduler,
    build_tune_search_space,
    create_study_name,
    jsonify_params,
    save_metadata,
    validate_model_hparams,
)
from .tune.mlflow_models import (
    resolve_best_trial_dir,
    seed_rows_for_trial,
    trial_rows_for_study,
)
from .tune.mlflow_tracking import build_benchmark_tags, log_table_artifact
from .tune.notification_callbacks import (
    RayTuneNotificationCallback,
    emit_trial_start_from_args,
    emit_trial_terminal_from_args,
)
from .tune.pruning import (
    aggregate_completed_metrics,
    build_final_eval_seed_context,
    build_reduced_hpo_seed_context,
    has_explicit_hpo_seeds,
    resolve_full_seeds,
    resolve_hpo_seeds,
)

logger = logging.getLogger(__name__)
EXCEL_PATH = Path("noboom_experiments.xlsx")
EXCEL_SHEET = "scores"
RUNTIME_METRIC = "time_total_s"
KAGGLE_DATASET_URI = "kaggle://faebs94/noboom-anomaly-detection-in-chemical-processes/versions/1"
FINAL_EVAL_AFTER_HPO_KEY = "final_eval_after_hpo"
FINAL_EVAL_AFTER_HPO_LEGACY_KEY = "final_full_seed_eval"


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dict(current, value)
        else:
            merged[key] = value
    return merged


def _dataset_search_space_override(args: Namespace) -> Optional[Dict[str, Any]]:
    common_params_cfg = OmegaConf.load(str(Path(args.config_dir) / "params" / "common.yaml"))
    common_params = OmegaConf.to_container(common_params_cfg, resolve=True)
    if not isinstance(common_params, Mapping):
        return None
    dataset_search_spaces = common_params.get("dataset_search_spaces")
    if not isinstance(dataset_search_spaces, Mapping):
        return None

    dataset_candidates = [
        getattr(args, "dataset_name", None),
        getattr(args, "requested_dataset_name", None),
    ]
    seen = set()
    for dataset_name in dataset_candidates:
        if dataset_name is None or dataset_name in seen:
            continue
        seen.add(dataset_name)
        override = dataset_search_spaces.get(str(dataset_name))
        if isinstance(override, Mapping):
            return dict(override)
    return None


def _resolve_local_storage_path(storage_path: str) -> Path:
    parsed_storage = urlparse(str(storage_path))
    if parsed_storage.path:
        return Path(parsed_storage.path)
    return Path(str(storage_path).split("?", 1)[0])


def _study_setup_lock(study_name: str, storage_path: str, temp_dir: str) -> FileLock:
    """Build a filesystem lock for study setup without relying on Ray named actors."""

    lock_root = _resolve_local_storage_path(storage_path)
    if not str(lock_root):
        lock_root = Path(temp_dir)
    lock_dir = lock_root / ".noboom_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_dir / f"{study_name}.setup.lock"))


def _load_study_params(args: Namespace, model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    study_params, hp_params = load_config(
        f"{args.config_dir}/params/common.yaml",
        f"{args.config_dir}/params/{model_name}.yaml",
    )
    dataset_override = _dataset_search_space_override(args)
    if dataset_override is not None:
        hp_params = _deep_merge_dict(hp_params, dataset_override)
    model_common_cfg = OmegaConf.load(str(Path(args.config_dir) / "models" / "common.yaml"))
    model_cfg = OmegaConf.load(str(Path(args.config_dir) / "models" / f"{model_name}.yaml"))
    merged_model_cfg = OmegaConf.to_container(
        OmegaConf.merge(model_common_cfg, model_cfg),
        resolve=True,
    )
    if isinstance(merged_model_cfg, Mapping):
        model_section = merged_model_cfg.get("model")
        if isinstance(model_section, Mapping):
            evaluation_postprocessing = model_section.get("evaluation_postprocessing")
            if isinstance(evaluation_postprocessing, Mapping):
                study_params["evaluation_postprocessing"] = dict(evaluation_postprocessing)

    logger.info("Study params loaded: %s", study_params)
    logger.debug("HP params loaded: %s", hp_params)
    return study_params, hp_params


def _select_hpo_study_params(args: Namespace, study_params: Dict[str, Any]) -> Dict[str, Any]:
    cli_hpo_seeds = getattr(args, "hpo_seeds", None)
    if cli_hpo_seeds is None:
        return study_params

    selected_params = dict(study_params)
    selected_params["hpo_seeds"] = list(cli_hpo_seeds)
    hpo_seeds = resolve_hpo_seeds(selected_params)
    logger.info(
        "Using HPO seed subset %s from --hpo-seed. Full study seeds remain %s for single-train runs.",
        hpo_seeds,
        selected_params.get("seeds", []),
    )
    return selected_params


def _is_local_execution(args: Namespace) -> bool:
    return str(getattr(args, "execution_backend", "ray")) == "local"


def _optuna_storage_from_uri(storage_uri: Optional[str]) -> Any:
    if storage_uri is None or storage_uri == "" or storage_uri == "memory://":
        return None
    return optuna.storages.RDBStorage(url=storage_uri)


def _sample_optuna_search_space(search_space: Mapping[str, Any], trial: optuna.Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        if isinstance(spec, Mapping) and "distribution" in spec:
            dist = str(spec["distribution"])
            if dist == "uniform":
                params[name] = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    step=float(spec["step"]) if spec.get("step") is not None else None,
                )
            elif dist == "loguniform":
                params[name] = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=True,
                )
                if spec.get("step") is not None:
                    step = float(spec["step"])
                    params[name] = round(float(params[name]) / step) * step
            elif dist == "int":
                params[name] = trial.suggest_int(
                    name,
                    int(spec["low"]),
                    int(spec["high"]),
                    step=int(spec["step"]) if spec.get("step") is not None else 1,
                )
            elif dist == "categorical":
                params[name] = trial.suggest_categorical(name, list(spec["choices"]))
            else:
                raise ValueError(f"Unknown distribution type: {dist}")
        elif isinstance(spec, Mapping):
            params[name] = _sample_optuna_search_space(spec, trial)
        else:
            raise ValueError(f"Invalid spec at '{name}': expected mapping, got {type(spec)}")
    return params


def _run_local_hpo_trial(
    *,
    args: Namespace,
    client: MlflowClient,
    hp_params: Dict[str, Any],
    trial_name: str,
    trial_root_dir: Path,
) -> Dict[str, Any]:
    tune_metric = args.study_params["tune_metric"]
    trial_run = client.create_run(
        args.experiment_id,
        tags=build_benchmark_tags(
            timestamp=args.timestamp,
            run_level="trial",
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
            run_mode="tune",
            deployment_mode=getattr(args, "deployment_mode", None),
            parent_run_id=args.study_run_id,
        ),
        run_name=trial_name,
    )
    trial_run_id = trial_run.info.run_id
    artifact_dir = trial_root_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        validate_model_hparams(
            args.model_name,
            hp_params,
            config_dir=str(Path(getattr(args, "config_dir", ".")).resolve()),
        )
    except InvalidHyperparameterConfiguration as exc:
        reason = str(exc)
        logger.warning("Skipping invalid local trial configuration '%s': %s", trial_name, reason)
        client.set_tag(trial_run_id, "invalid_config", "true")
        client.set_tag(trial_run_id, "invalid_config_reason", reason)
        save_metadata(args, artifact_dir, hp_params)
        client.set_terminated(trial_run_id)
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

    metrics: Dict[str, list] = {}
    theory_best_metrics: Dict[str, list] = {}
    total_metrics: Dict[str, Any] = {}
    hpo_seeds = resolve_hpo_seeds(args.study_params)
    completed_seed_count = 0

    for seed in hpo_seeds:
        seed_result = run_seed_task_local(
            hp_params,
            args,
            seed,
            str(trial_root_dir),
            trial_run_id,
            trial_name,
        )
        for name, metric in seed_result.items():
            metrics.setdefault(name, []).append(metric[0])
            theory_best_metrics.setdefault(name, []).append(metric[1])
            assert (max(theory_best_metrics[name]) - min(theory_best_metrics[name])) <= np.finfo(
                np.float32
            ).eps, "Theoretical best value differs across seeds"

        completed_seed_count += 1
        total_metrics = aggregate_completed_metrics(
            metric_history=metrics,
            best_metric_history=theory_best_metrics,
        )
        total_metrics |= {
            "n_seeds": completed_seed_count - 1,
            "completed_seed_count": completed_seed_count,
            "mlflow_run_id": trial_run_id,
        }
        total_metrics.update(build_reduced_hpo_seed_context(args.study_params))

    save_metadata(args, artifact_dir, hp_params)
    _log_trial_metadata_to_mlflow(trial_run_id=trial_run_id, artifact_dir=artifact_dir)
    client.set_terminated(trial_run_id)
    if tune_metric not in total_metrics:
        total_metrics[tune_metric] = _invalid_trial_metric_value(args.study_params.get("direction"))
    return total_metrics


def _search_space_size(search_space: Mapping[str, Any]) -> float:
    size: float = 1
    for spec in search_space.values():
        if isinstance(spec, Mapping) and "distribution" in spec:
            dist = str(spec["distribution"])
            if dist == "categorical":
                size *= len(spec["choices"])
            elif dist == "int":
                step = int(spec.get("step") or 1)
                size *= int(math.floor((int(spec["high"]) - int(spec["low"])) / step) + 1)
            else:
                size *= math.inf
        elif isinstance(spec, Mapping):
            size *= _search_space_size(spec)
        else:
            raise ValueError(f"Invalid search space entry: expected mapping, got {type(spec)}")
    return size


def execute_local_tuning_workflow(args: Namespace) -> Tuple[str, str, Dict[str, Any]]:
    logger.debug("Starting execute_local_tuning_workflow.")
    model_name = args.model_name
    dataset_name = args.dataset_name
    timestamp = args.timestamp
    args.remove_ckpt = True
    args.config_dir = str(Path(args.config_dir).resolve())

    client = MlflowClient(args.tracking_uri)
    experiment_name = client.get_experiment(args.experiment_id).name
    study_name = create_study_name(model_name, dataset_name, timestamp)
    setup_lock = _study_setup_lock(study_name, str(args.storage_path), str(args.temp_dir))

    with setup_lock:
        study_params, hp_params = _load_study_params(args, model_name)
        study_params = _select_hpo_study_params(args, study_params)
        tune_metric = str(study_params["tune_metric"])
        tune_mode = "maximize" if study_params.get("direction") == "maximize" else "minimize"
        search_space_size = _search_space_size(hp_params)
        num_samples = int(min(float(study_params["n_trials"]), search_space_size))
        sampler = build_sampler(study_params.get("sampler"))

        args.study_params = study_params
        study_run = client.create_run(
            args.experiment_id,
            tags=build_benchmark_tags(
                timestamp=args.timestamp,
                run_level="study",
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
                run_mode="tune",
                deployment_mode=getattr(args, "deployment_mode", None),
                parent_run_id=getattr(args, "mlflow_controller_run_id", None),
                extra_tags=build_reduced_hpo_seed_context(study_params),
            ),
            run_name=study_name,
        )
        args.study_run_id = study_run.info.run_id

        optuna_study = optuna.create_study(
            study_name=study_name,
            direction=tune_mode,
            sampler=sampler,
            storage=_optuna_storage_from_uri(args.optuna_storage_uri),
            load_if_exists=True,
        )

    logger.info(
        "Starting local Optuna search for experiment='%s', model='%s', dataset='%s', trials=%d.",
        experiment_name,
        model_name,
        dataset_name,
        num_samples,
    )
    best_result: Optional[Dict[str, Any]] = None
    best_hparams: Optional[Dict[str, Any]] = None
    best_value: Optional[float] = None
    best_trial_run_id: Optional[str] = None
    local_study_root = _resolve_local_storage_path(str(args.storage_path)) / study_name

    for _ in range(num_samples):
        trial = optuna_study.ask()
        resolved_hparams = _sample_optuna_search_space(hp_params, trial)
        trial_name = f"{study_name}__trial_{trial.number}"
        emit_trial_start_from_args(args, trial_name=trial_name)
        try:
            trial_result = _run_local_hpo_trial(
                args=args,
                client=client,
                hp_params=resolved_hparams,
                trial_name=trial_name,
                trial_root_dir=local_study_root / trial_name,
            )
        except Exception as exc:  # noqa: BLE001
            emit_trial_terminal_from_args(
                args,
                trial_name=trial_name,
                status="FAILED",
                message=str(exc) or exc.__class__.__name__,
            )
            raise
        emit_trial_terminal_from_args(
            args,
            trial_name=trial_name,
            status="SUCCEEDED",
            result=trial_result,
        )
        metric_value = float(trial_result[tune_metric])
        trial.set_user_attr("resolved_hparams", jsonify_params(resolved_hparams))
        trial.set_user_attr("result", jsonify_params(trial_result))
        optuna_study.tell(trial, metric_value)

        improved = best_value is None
        if best_value is not None and tune_mode == "maximize":
            improved = metric_value > best_value
        elif best_value is not None:
            improved = metric_value < best_value
        if improved:
            best_value = metric_value
            best_result = dict(trial_result)
            best_hparams = dict(resolved_hparams)
            run_id = trial_result.get("mlflow_run_id")
            best_trial_run_id = str(run_id) if run_id is not None else None
            _persist_pair_snapshot(
                client,
                args=args,
                result=trial_result,
                resolved_hparams=resolved_hparams,
                status="RUNNING",
                partial_result=True,
                result_source="incremental_best_trial",
                write_mlflow=True,
            )

    if best_result is None:
        raise RuntimeError("Local Optuna study completed without any trial results.")

    args.hpo_study_run_id = study_run.info.run_id
    args.hpo_best_trial_run_id = best_trial_run_id
    result = _finalize_study_outputs(
        client,
        args=args,
        trial_run_id=str(best_trial_run_id),
        trial_artifact_dir=None,
        resolved_hparams=best_hparams,
        result=best_result,
    )
    client.set_terminated(study_run.info.run_id)

    if _should_run_full_seed_final_eval(study_params):
        best_config_payload = _best_config_payload(best_hparams)
        if best_config_payload is None:
            raise RuntimeError(
                "Cannot run full-seed final evaluation because the selected local Optuna trial has no config payload."
            )
        return execute_single_train_workflow(
            _full_seed_final_eval_args(
                args,
                hpo_result=result,
                hpo_study_run_id=study_run.info.run_id,
                hpo_best_trial_run_id=best_trial_run_id,
            ),
            hparams_payload=best_config_payload,
        )

    return model_name, dataset_name, result


def _extract_hparams_payload(result_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not result_json:
        return None

    if isinstance(result_json.get("config"), dict):
        return result_json["config"]

    hp_candidate = result_json.get("hp_params")
    if isinstance(hp_candidate, dict):
        return hp_candidate

    params_candidate = result_json.get("params")
    if isinstance(params_candidate, dict):
        return params_candidate

    return None


def _flag_enabled(args: Namespace, attr_name: str, default: bool = False) -> bool:
    return bool(getattr(args, attr_name, default))


def _should_run_full_seed_final_eval(study_params: Mapping[str, Any]) -> bool:
    final_eval_enabled = bool(
        study_params.get(FINAL_EVAL_AFTER_HPO_KEY, False)
        or study_params.get(FINAL_EVAL_AFTER_HPO_LEGACY_KEY, False)
    )
    if not final_eval_enabled:
        return False
    return bool(build_reduced_hpo_seed_context(study_params))


def _full_seed_final_eval_args(
    args: Namespace,
    *,
    hpo_result: Mapping[str, Any],
    hpo_study_run_id: str,
    hpo_best_trial_run_id: Optional[str],
) -> Namespace:
    eval_args = Namespace(**vars(args))
    eval_args.tune = False
    eval_args.source_experiment_id = None
    eval_args.hpo_seeds = None
    eval_args.study_run_id = None
    eval_args.hpo_result = dict(hpo_result)
    eval_args.hpo_study_run_id = hpo_study_run_id
    eval_args.hpo_best_trial_run_id = hpo_best_trial_run_id
    eval_args.result_source = "final_full_seed_eval_after_hpo"
    eval_args.incremental_result_source = "incremental_final_full_seed_eval_after_hpo"
    eval_args.result_seed_context = build_final_eval_seed_context(args.study_params)
    eval_args.result_seed_context["hpo_study_run_id"] = hpo_study_run_id
    if hpo_best_trial_run_id is not None:
        eval_args.result_seed_context["hpo_best_trial_run_id"] = hpo_best_trial_run_id
    return eval_args


def _max_concurrent_trials(num_samples: int, trial_resources: Mapping[str, Any]) -> int:
    if (
        float(trial_resources.get("GPU", 0.0)) >= 1.0
        and float(trial_resources.get("exclusive", 0.0)) >= 1.0
    ):
        return 1
    return max(1, num_samples // 2)


def _scheduler_config_for_hpo_seeds(study_params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    scheduler_cfg = study_params.get("scheduler")
    if scheduler_cfg is None:
        return None
    if not isinstance(scheduler_cfg, Mapping):
        raise ValueError("study.scheduler must be a mapping when provided.")
    if not has_explicit_hpo_seeds(study_params):
        return dict(scheduler_cfg)

    scheduler_copy = dict(scheduler_cfg)
    if scheduler_copy.get("name", "ASHA") != "ASHA":
        return scheduler_copy

    scheduler_args = scheduler_copy.get("args", {})
    if not isinstance(scheduler_args, Mapping):
        scheduler_args = {}
    scheduler_args = dict(scheduler_args)
    hpo_seed_count = len(resolve_hpo_seeds(study_params))
    previous_max_t = scheduler_args.get("max_t")
    previous_grace_period = scheduler_args.get("grace_period")
    scheduler_args["max_t"] = hpo_seed_count
    scheduler_args["grace_period"] = 1
    scheduler_copy["args"] = scheduler_args
    logger.info(
        "Adjusted ASHA scheduler for study.hpo_seeds: max_t=%s -> %d, grace_period=%s -> 1.",
        previous_max_t,
        hpo_seed_count,
        previous_grace_period,
    )
    return scheduler_copy


def _pair_snapshot_dataset_name(args: Namespace) -> str:
    return str(getattr(args, "requested_dataset_name", args.dataset_name))


def _pair_experiment_root(storage_path: str) -> Path:
    return _resolve_local_storage_path(storage_path).parents[1]


def _pair_result_path(storage_path: str) -> Path:
    return _resolve_local_storage_path(storage_path) / "result.json"


def _build_study_result_payload(
    *,
    result: Mapping[str, Any],
    resolved_hparams: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    final_result = dict(result)
    if resolved_hparams is not None:
        final_result["hp_params"] = jsonify_params(dict(resolved_hparams))
    return final_result


def _is_final_eval_result_source(result_source: str) -> bool:
    return result_source in {
        "final_eval",
        "incremental_final_eval",
        "incremental_single_train",
        "final_full_seed_eval_after_hpo",
        "incremental_final_full_seed_eval_after_hpo",
    }


def _seed_context_for_snapshot(args: Namespace, result_source: str) -> Dict[str, Any]:
    explicit_context = getattr(args, "result_seed_context", None)
    if isinstance(explicit_context, Mapping):
        return dict(explicit_context)

    study_params = getattr(args, "study_params", {})
    if not isinstance(study_params, Mapping) or not has_explicit_hpo_seeds(study_params):
        return {}
    if _is_final_eval_result_source(result_source) or not _flag_enabled(args, "tune"):
        return build_final_eval_seed_context(study_params)
    return build_reduced_hpo_seed_context(study_params)


def _attach_lineage_fields(
    final_result: Dict[str, Any],
    *,
    args: Namespace,
    result_source: str,
) -> None:
    final_result["result_source"] = result_source
    hpo_best_trial_run_id = final_result.get("hpo_best_trial_run_id")
    if hpo_best_trial_run_id is None:
        hpo_best_trial_run_id = getattr(args, "hpo_best_trial_run_id", None)
    if hpo_best_trial_run_id is None and not _is_final_eval_result_source(result_source):
        hpo_best_trial_run_id = final_result.get("mlflow_run_id")
    if hpo_best_trial_run_id is not None:
        final_result["hpo_best_trial_run_id"] = str(hpo_best_trial_run_id)

    hpo_study_run_id = getattr(args, "hpo_study_run_id", None)
    if hpo_study_run_id is not None:
        final_result["hpo_study_run_id"] = str(hpo_study_run_id)

    if _is_final_eval_result_source(result_source):
        final_eval_run_id = final_result.get("final_eval_run_id", final_result.get("mlflow_run_id"))
        if final_eval_run_id is not None:
            final_result["final_eval_run_id"] = str(final_eval_run_id)


def _best_config_payload(resolved_hparams: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if resolved_hparams is None:
        return None
    return {"config": dict(resolved_hparams)}


def _prepare_single_train_manifests(
    args: Namespace,
    resolved_hparams: Mapping[str, Any],
) -> None:
    prepared_manifest_paths = getattr(args, "prepared_manifest_paths", None)
    if (
        isinstance(prepared_manifest_paths, Mapping)
        and getattr(args, "data_manifest_path", None)
        and (not getattr(args, "use_generated", False) or "synthetic" in prepared_manifest_paths)
    ):
        return
    if not hasattr(args, "runtime_config"):
        logger.debug(
            "Runtime config is unavailable; assuming single-train manifests were prepared by caller."
        )
        return

    args.preparation_hp_params = dict(resolved_hparams)
    if getattr(args, "data_manifest_path_explicit", False):
        prepared_manifest_paths = {
            str(key): str(value)
            for key, value in getattr(args, "prepared_manifest_paths", {}).items()
            if value is not None
        }
        explicit_manifest_path = getattr(args, "data_manifest_path", None)
        if explicit_manifest_path is not None:
            prepared_manifest_paths.setdefault("real", str(explicit_manifest_path))
        if getattr(args, "use_generated", False) and "synthetic" not in prepared_manifest_paths:
            generated_manifest_paths = prepare_model_dataset_manifests(
                dataset_name=args.dataset_name,
                requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
                model_name=args.model_name,
                config_dir=args.config_dir,
                runtime_config=args.runtime_config,
                use_generated=True,
                hp_params=dict(resolved_hparams),
                verbosity=int(getattr(args, "verbose", 0) or 0),
            )
            prepared_manifest_paths["synthetic"] = generated_manifest_paths["synthetic"]
        args.prepared_manifest_paths = prepared_manifest_paths
        if "real" in prepared_manifest_paths:
            args.data_manifest_path = prepared_manifest_paths["real"]
        return

    args.prepared_manifest_paths = prepare_model_dataset_manifests(
        dataset_name=args.dataset_name,
        requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
        model_name=args.model_name,
        config_dir=args.config_dir,
        runtime_config=args.runtime_config,
        use_generated=bool(getattr(args, "use_generated", False)),
        hp_params=dict(resolved_hparams),
        verbosity=int(getattr(args, "verbose", 0) or 0),
    )
    args.data_manifest_path = args.prepared_manifest_paths["real"]


def _excel_metrics_for_result(args: Namespace, result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics_to_use = list(args.study_params.get("metrics", [])) + [RUNTIME_METRIC]
    return {
        key: value
        for key, value in result.items()
        if any(metric_name in key for metric_name in metrics_to_use)
    }


def _persist_pair_snapshot(
    client: MlflowClient,
    *,
    args: Namespace,
    result: Mapping[str, Any],
    resolved_hparams: Optional[Mapping[str, Any]],
    status: str,
    partial_result: bool,
    result_source: str,
    write_mlflow: bool,
) -> Dict[str, Any]:
    final_result = _build_study_result_payload(
        result=result,
        resolved_hparams=resolved_hparams,
    )
    seed_context = {}
    if not (
        not _flag_enabled(args, "tune")
        and result.get("uses_reduced_hpo_seeds") is False
    ):
        seed_context = _seed_context_for_snapshot(args, result_source)
    final_result = seed_context | final_result
    _attach_lineage_fields(
        final_result,
        args=args,
        result_source=result_source,
    )
    if final_result.get("uses_reduced_hpo_seeds") is False:
        final_result.pop("hpo_seeds", None)
        final_result.pop("hpo_seed_count", None)

    local_result_path = _pair_result_path(str(args.storage_path))
    local_result_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = PairResult(
        model_name=str(args.model_name),
        dataset_name=_pair_snapshot_dataset_name(args),
        storage_path=str(args.storage_path),
        result=final_result,
        study_run_id=getattr(args, "study_run_id", None),
        status=status,
        partial_result=partial_result,
        result_source=result_source,
    )
    local_result_path.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")

    if write_mlflow and getattr(args, "study_run_id", None):
        client.log_dict(args.study_run_id, final_result, artifact_file="summary/result.json")

    excel_path = _pair_experiment_root(str(args.storage_path)) / EXCEL_PATH
    excel_metrics = _excel_metrics_for_result(args, final_result)
    if excel_metrics:
        update_score_excel_one(
            str(args.model_name),
            _pair_snapshot_dataset_name(args),
            excel_metrics,
            excel_path,
            EXCEL_SHEET,
            minimize_metrics=[RUNTIME_METRIC],
        )
    elif excel_score_table_has_metrics(excel_path, EXCEL_SHEET):
        update_score_excel_one(
            str(args.model_name),
            _pair_snapshot_dataset_name(args),
            {},
            excel_path,
            EXCEL_SHEET,
            minimize_metrics=[RUNTIME_METRIC],
        )

    return final_result


class BestTrialSnapshotCallback(Callback):
    def __init__(
        self,
        *,
        args: Namespace,
        client: MlflowClient,
        mode: str,
    ) -> None:
        self._args = args
        self._client = client
        self._mode = mode
        self._best_metric_value: Optional[float] = None

    def on_trial_result(
        self,
        iteration: int,
        trials: List["Trial"],
        trial: "Trial",
        result: Dict[str, Any],
        **info: Any,
    ) -> None:
        del iteration, trials, info

        tune_metric = str(self._args.study_params["tune_metric"])
        metric_value = result.get(tune_metric)
        if not isinstance(metric_value, (int, float)):
            return

        metric_value = float(metric_value)
        if np.isnan(metric_value):
            return

        if self._best_metric_value is not None:
            if self._mode == "max" and metric_value <= self._best_metric_value:
                return
            if self._mode == "min" and metric_value >= self._best_metric_value:
                return

        self._best_metric_value = metric_value
        resolved_hparams = getattr(trial, "config", None)
        if not isinstance(resolved_hparams, Mapping):
            resolved_hparams = None

        _persist_pair_snapshot(
            self._client,
            args=self._args,
            result=result,
            resolved_hparams=resolved_hparams,
            status="RUNNING",
            partial_result=True,
            result_source="incremental_best_trial",
            write_mlflow=True,
        )


def _promote_selected_trial_artifacts(
    client: MlflowClient,
    *,
    study_run_id: str,
    trial_artifact_dir: Optional[Path],
) -> None:
    if trial_artifact_dir is None:
        return

    metadata_path = trial_artifact_dir / "metadata.json"
    if metadata_path.exists():
        client.log_artifact(study_run_id, str(metadata_path), artifact_path="selected_trial")


def _log_study_tables(
    client: MlflowClient,
    *,
    args: Namespace,
    trial_run_id: str,
) -> None:
    if not _flag_enabled(args, "mlflow_enable_tables"):
        return

    seed_rows = seed_rows_for_trial(
        client,
        experiment_id=args.experiment_id,
        trial_run_id=trial_run_id,
    )
    if seed_rows:
        log_table_artifact(
            client,
            args.study_run_id,
            artifact_file="tables/seeds.json",
            rows=seed_rows,
        )

    trial_rows = trial_rows_for_study(
        client,
        experiment_id=args.experiment_id,
        study_run_id=args.study_run_id,
    )
    if trial_rows:
        log_table_artifact(
            client,
            args.study_run_id,
            artifact_file="tables/trials.json",
            rows=trial_rows,
        )


def _finalize_study_outputs(
    client: MlflowClient,
    *,
    args: Namespace,
    trial_run_id: str,
    trial_artifact_dir: Optional[Path],
    resolved_hparams: Optional[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    if _flag_enabled(args, "mlflow_enable_logged_models") or _flag_enabled(args, "mlflow_enable_evaluation"):
        logger.warning(
            "MLflow logged-model packaging and evaluation are disabled; skipping post-run model logging."
        )

    if _flag_enabled(args, "mlflow_promote_trial_artifacts") and trial_artifact_dir is not None:
        _promote_selected_trial_artifacts(
            client,
            study_run_id=args.study_run_id,
            trial_artifact_dir=trial_artifact_dir,
        )

    _log_study_tables(
        client,
        args=args,
        trial_run_id=trial_run_id,
    )
    result_source = getattr(args, "result_source", None)
    if result_source is None:
        result_source = "hpo_best_trial" if _flag_enabled(args, "tune") else "final_eval"

    return _persist_pair_snapshot(
        client,
        args=args,
        result=result,
        resolved_hparams=resolved_hparams,
        status="SUCCEEDED",
        partial_result=False,
        result_source=result_source,
        write_mlflow=True,
    )


def execute_tuning_workflow(args: Namespace) -> Tuple[str, str, Dict[str, Any]]:
    """Run the full hyperparameter tuning workflow.

    Args:
        args (Namespace): Parsed CLI arguments for tuning.

    Returns:
        Tuple[str, str, Dict[str, Any]]: Model name, dataset name, and best metrics.

    Side Effects:
        Creates MLflow runs, launches Ray Tune trials, and writes artifacts.
    """
    if _is_local_execution(args):
        return execute_local_tuning_workflow(args)
    if tune is None or OptunaSearch is None or MLFlowLoggerCallbackSW is None:
        raise RuntimeError("Ray Tune is required for the default tuning backend.")

    logger.debug("Starting execute_tuning_workflow.")
    optuna_storage_uri = args.optuna_storage_uri
    model_name = args.model_name
    dataset_name = args.dataset_name
    timestamp = args.timestamp
    args.remove_ckpt = True
    args.config_dir = str(Path(args.config_dir).resolve())

    client = MlflowClient(args.tracking_uri)
    experiment_name = client.get_experiment(args.experiment_id).name

    logger.info(
        "Running hyperparameter tuning for experiment='%s', model='%s', dataset='%s', timestamp='%s'.",
        experiment_name,
        model_name,
        dataset_name,
        timestamp,
    )

    study_name = create_study_name(model_name, dataset_name, timestamp)
    setup_lock = _study_setup_lock(study_name, str(args.storage_path), str(args.temp_dir))

    with setup_lock:
        optuna_storage = optuna.storages.RDBStorage(url=optuna_storage_uri)
        logger.info("Optuna storage URL: %s", optuna_storage_uri)

        study_params, hp_params = _load_study_params(args, model_name)
        study_params = _select_hpo_study_params(args, study_params)

        tune_metric = study_params.get("tune_metric")
        tune_mode = "max" if study_params.get("direction") == "maximize" else "min"
        logger.info("Tune metric: '%s', mode: '%s'.", tune_metric, tune_mode)

        tune_search_space, search_space_size = build_tune_search_space(hp_params)
        num_samples = int(min(float(study_params["n_trials"]), search_space_size))
        logger.info("Number of trials: %s", num_samples)

        logger.info("Optuna/Ray study name: '%s'.", study_name)

        sampler = build_sampler(study_params.get("sampler"))
        hpo_seeds = resolve_hpo_seeds(study_params)
        if has_explicit_hpo_seeds(study_params):
            logger.info(
                "Ray Tune HPO will use %d seeds from study.hpo_seeds; single-train remains on %d study seeds.",
                len(hpo_seeds),
                len(study_params.get("seeds", [])),
            )
        scheduler = build_tune_scheduler(_scheduler_config_for_hpo_seeds(study_params))

        # Enrich args with config info before deriving resource requests.
        args.study_params = study_params
        full_trial_resources = tune_resource_request_dict(args)

        optuna_search = OptunaSearch(
            storage=optuna_storage,
            study_name=study_name,
            sampler=sampler,
        )
        logger.info("OptunaSearch initialized.")

        mlflow_callback = MLFlowLoggerCallbackSW(
            tracking_uri=args.tracking_uri,
            experiment_name=experiment_name,
            save_artifact=True,
            tags={
                "dataset": dataset_name,
                "dataset_requested": getattr(args, "requested_dataset_name", dataset_name),
                "model": model_name,
                "timestamp": timestamp,
                "deployment_mode": getattr(args, "deployment_mode", ""),
            },
        )

        logger.info("MLflowLoggerCallback initialized.")

        study_run = client.create_run(
            args.experiment_id,
            tags=build_benchmark_tags(
                timestamp=args.timestamp,
                run_level="study",
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
                run_mode="tune",
                deployment_mode=getattr(args, "deployment_mode", None),
                parent_run_id=getattr(args, "mlflow_controller_run_id", None),
                extra_tags=build_reduced_hpo_seed_context(study_params),
            ),
            run_name=study_name,
        )
        args.study_run_id = study_run.info.run_id
        best_trial_snapshot_callback = BestTrialSnapshotCallback(
            args=args,
            client=client,
            mode=tune_mode,
        )
        notification_callback = RayTuneNotificationCallback(args=args)
        tuner = tune.Tuner(
            tune.with_resources(
                functools.partial(run_tune_trainable, args=args),
                resources=full_trial_resources,
            ),
            param_space=tune_search_space,
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                metric=tune_metric,
                mode=tune_mode,
                scheduler=scheduler,
                num_samples=num_samples,
                max_concurrent_trials=_max_concurrent_trials(num_samples, full_trial_resources),
                trial_name_creator=lambda trial: f"{study_name}__trial_{trial.trial_id}",
            ),
            run_config=tune.RunConfig(
                name=study_name,
                storage_path=str(_resolve_local_storage_path(str(args.storage_path))),
                callbacks=[mlflow_callback, best_trial_snapshot_callback, notification_callback],
                failure_config=tune.FailureConfig(max_failures=3),
            )
        )
    logger.info("Starting Ray Tune hyperparameter search.")
    result_grid = tuner.fit()
    #with worker_lock():
    logger.info("Ray Tune finished; selecting best result.")
    best_result = result_grid.get_best_result()
    result = dict(best_result.metrics)
    logger.info("Best result metrics: %s", result)
    best_run_id = result.get("mlflow_run_id")
    args.hpo_study_run_id = study_run.info.run_id
    args.hpo_best_trial_run_id = str(best_run_id) if best_run_id is not None else None

    best_trial_dir = None
    resolved_hparams = None
    best_result_config = getattr(best_result, "config", None)
    if isinstance(best_result_config, Mapping):
        resolved_hparams = dict(best_result_config)
    if _flag_enabled(args, "mlflow_promote_trial_artifacts"):
        best_trial_dir = resolve_best_trial_dir(
            best_result,
            experiment_path=Path(result_grid.experiment_path),
        )
    result = _finalize_study_outputs(
        client,
        args=args,
        trial_run_id=str(best_run_id),
        trial_artifact_dir=best_trial_dir,
        resolved_hparams=resolved_hparams,
        result=result,
    )

    shutil.rmtree(result_grid.experiment_path, ignore_errors=True)

    logger.info("Hyperparameter tuning completed successfully.")
    client.set_terminated(study_run.info.run_id)
    if _should_run_full_seed_final_eval(study_params):
        best_config_payload = _best_config_payload(resolved_hparams)
        if best_config_payload is None:
            raise RuntimeError(
                "Cannot run full-seed final evaluation because the selected Tune result has no config payload."
            )
        logger.info(
            "Reduced-seed HPO completed; starting final evaluation with %d full study seeds "
            "instead of %d HPO seeds.",
            len(resolve_full_seeds(study_params)),
            len(resolve_hpo_seeds(study_params)),
        )
        return execute_single_train_workflow(
            _full_seed_final_eval_args(
                args,
                hpo_result=result,
                hpo_study_run_id=study_run.info.run_id,
                hpo_best_trial_run_id=str(best_run_id) if best_run_id is not None else None,
            ),
            hparams_payload=best_config_payload,
        )
    return model_name, dataset_name, result


def execute_single_train_workflow(
    args: Namespace,
    hparams_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Run a single-train workflow without hyperparameter search."""
    logger.debug("Starting execute_single_train_workflow.")
    model_name = args.model_name
    dataset_name = args.dataset_name
    timestamp = args.timestamp
    source_experiment_id = getattr(args, "source_experiment_id", None)

    client = MlflowClient(args.tracking_uri)
    experiment_name = client.get_experiment(args.experiment_id).name

    logger.info(
        "Running single-train workflow for experiment='%s', model='%s', dataset='%s', timestamp='%s'.",
        experiment_name,
        model_name,
        dataset_name,
        timestamp,
    )
    args.config_dir = str(Path(args.config_dir).resolve())

    study_params, hp_params = _load_study_params(args, model_name)
    study_params = _select_hpo_study_params(args, study_params)
    args.study_params = study_params
    args.remove_ckpt = False
    resolved_hparams = None
    study_run_id = None
    if hparams_payload:
        resolved_hparams = _extract_hparams_payload(hparams_payload)
        if resolved_hparams:
            logger.info(f"Using resolved hyperparameters from provided payload. Params: {resolved_hparams}")

    if resolved_hparams is None and source_experiment_id:
        try:
            source_result = load_study_result_from_mlflow(
                source_experiment_id,
                model_name,
                dataset_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load hyperparameters from source experiment %s: %s",
                source_experiment_id,
                exc,
            )
            source_result = None
        if source_result:
            result_json, study_run_id = source_result
            resolved_hparams = _extract_hparams_payload(result_json)
            if resolved_hparams:
                if study_run_id:
                    logger.info(
                        "Loaded hyperparameters from source study run %s. Params: %s.",
                        study_run_id,
                        resolved_hparams,
                    )
                else:
                    logger.info(
                        "Loaded hyperparameters from source experiment snapshot. Params: %s.",
                        resolved_hparams,
                    )
            else:
                logger.warning(
                    "Source study result %s did not include a usable config payload; falling back to defaults.",
                    study_run_id,
                )

    if resolved_hparams is None:
        resolved_hparams = hp_params
        if source_experiment_id:
            logger.info(
                "No source experiment hyperparameters found; using default config parameters.",
            )
        else:
            logger.info(
                "Source experiment ID not provided; using default config parameters."
            )

    try:
        validate_model_hparams(
            model_name,
            resolved_hparams,
            config_dir=args.config_dir,
        )
    except InvalidHyperparameterConfiguration as exc:
        raise ValueError(
            f"Invalid resolved hyperparameters for model='{model_name}' on dataset='{dataset_name}': {exc}"
        ) from exc

    _prepare_single_train_manifests(args, resolved_hparams)

    study_name = create_study_name(model_name, dataset_name, timestamp)
    trial_name = f"{study_name}__single_train"

    study_tags = build_benchmark_tags(
        timestamp=timestamp,
        run_level="study",
        model_name=model_name,
        dataset_name=dataset_name,
        requested_dataset_name=getattr(args, "requested_dataset_name", dataset_name),
        run_mode="single_train",
        deployment_mode=getattr(args, "deployment_mode", None),
        parent_run_id=getattr(args, "mlflow_controller_run_id", None),
        extra_tags=_seed_context_for_snapshot(args, getattr(args, "result_source", "final_eval")),
    )
    if study_run_id:
        study_tags["source_study_run_id"] = study_run_id
    if source_experiment_id:
        study_tags["source_experiment_id"] = source_experiment_id
    if getattr(args, "hpo_study_run_id", None):
        study_tags["hpo_study_run_id"] = str(args.hpo_study_run_id)
    if getattr(args, "hpo_best_trial_run_id", None):
        study_tags["hpo_best_trial_run_id"] = str(args.hpo_best_trial_run_id)

    study_run = client.create_run(
        args.experiment_id,
        tags=study_tags,
        run_name=study_name,
    )
    args.study_run_id = study_run.info.run_id

    trial_run = client.create_run(
        args.experiment_id,
        tags=build_benchmark_tags(
            timestamp=timestamp,
            run_level="trial",
            model_name=model_name,
            dataset_name=dataset_name,
            requested_dataset_name=getattr(args, "requested_dataset_name", dataset_name),
            run_mode="single_train",
            deployment_mode=getattr(args, "deployment_mode", None),
            parent_run_id=args.study_run_id,
            source_experiment_id=source_experiment_id,
            source_study_run_id=study_run_id,
            extra_tags=_seed_context_for_snapshot(args, getattr(args, "result_source", "final_eval")),
        ),
        run_name=trial_name,
    )
    trial_run_id = trial_run.info.run_id

    trial_root_base = _resolve_local_storage_path(str(args.storage_path)) / study_name
    trial_root_dir = trial_root_base / trial_name
    trial_root_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting single-train run across %d seeds.", len(args.study_params["seeds"]))
    emit_trial_start_from_args(args, trial_name=trial_name, run_id=trial_run_id)
    metrics = {}
    theory_best_metrics = {}
    completed_seed_count = 0
    result: Dict[str, Any] = {}

    if _is_local_execution(args):
        seed_results = [
            run_seed_task_local(
                resolved_hparams,
                args,
                seed,
                str(trial_root_dir),
                trial_run_id,
                trial_name,
            )
            for seed in args.study_params["seeds"]
        ]
    else:
        node_id = ray.get_runtime_context().get_node_id()
        strat = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        seed_resources = tune_resource_request_dict(args).copy()
        seed_num_cpus = float(seed_resources.pop("CPU", 0))
        seed_num_gpus = float(seed_resources.pop("GPU", 0))
        seed_memory = float(seed_resources.pop("memory", 0))
        seed_task_options = {
            "scheduling_strategy": strat,
            "num_cpus": seed_num_cpus,
            "num_gpus": seed_num_gpus,
            "resources": seed_resources.copy(),
        }
        if seed_memory > 0:
            seed_task_options["memory"] = seed_memory
        seed_futures = [
            run_seed_task.options(**seed_task_options).remote(
                resolved_hparams,
                args,
                seed,
                trial_root_dir,
                trial_run_id,
                trial_name,
            )
            for step, seed in enumerate(args.study_params["seeds"])
        ]
        pending_seed_futures = list(zip(args.study_params["seeds"], seed_futures))
        seed_results = []
        while pending_seed_futures:
            pending_futures = [future for _seed, future in pending_seed_futures]
            ready_futures, _remaining = ray.wait(pending_futures, num_returns=1)
            ready_future = ready_futures[0]
            ready_index = next(
                index
                for index, (_seed, future) in enumerate(pending_seed_futures)
                if future is ready_future or future == ready_future
            )
            pending_seed_futures.pop(ready_index)
            seed_results.append(ray.get(ready_future))

    for seed_result in seed_results:
        raw_metrics = seed_result
        for name, metric in raw_metrics.items():
            metrics.setdefault(name, []).append(metric[0])
            theory_best_metrics.setdefault(name, []).append(metric[1])
            assert (max(theory_best_metrics[name]) - min(theory_best_metrics[name])) <= np.finfo(
                np.float32
            ).eps, "Theoretical best value differs across seeds"

        completed_seed_count += 1
        result = aggregate_completed_metrics(
            metric_history=metrics,
            best_metric_history=theory_best_metrics,
        )
        result |= {
            "n_seeds": completed_seed_count - 1,
            "completed_seed_count": completed_seed_count,
            "mlflow_run_id": trial_run_id,
        }
        _persist_pair_snapshot(
            client,
            args=args,
            result=result,
            resolved_hparams=resolved_hparams,
            status="RUNNING",
            partial_result=True,
            result_source=getattr(args, "incremental_result_source", "incremental_single_train"),
            write_mlflow=True,
        )

    metric_result = aggregate_completed_metrics(
        metric_history=metrics,
        best_metric_history=theory_best_metrics,
    )
    mlflow.log_metrics(metric_result, run_id=args.study_run_id)

    save_metadata(
        args,
        trial_root_dir,
        resolved_hparams,
    )
    logger.info("Single-train result metrics: %s", result)

    result = _finalize_study_outputs(
        client,
        args=args,
        trial_run_id=trial_run_id,
        trial_artifact_dir=trial_root_dir,
        resolved_hparams=resolved_hparams,
        result=result,
    )
    logger.info("Single-train workflow completed successfully.")
    client.set_terminated(trial_run_id)
    client.set_terminated(study_run.info.run_id)
    emit_trial_terminal_from_args(
        args,
        trial_name=trial_name,
        status="SUCCEEDED",
        result=result,
    )
    return model_name, dataset_name, result


def run_tune_or_train(args: Namespace) -> Tuple[str, str, Dict[str, Any]]:
    """Run hyperparameter tuning or evaluation depending on CLI flags.

    Args:
        args (Namespace): Parsed CLI arguments with run configuration.

    Returns:
        Tuple[str, str, Dict[str, Any]]: Model name, dataset name, and metrics.

    Side Effects:
        Initializes Ray, configures MLflow, and executes tuning/evaluation.
    """
    args.requested_dataset_name = args.dataset_name
    args.use_generated = "tsst" in args.dataset_name
    args.dataset_name = args.dataset_name.replace("_tsst", "")
    args.config_dir = str(Path(args.config_dir).resolve())
    runtime_config = RuntimeConfig.from_env_and_args(args)
    runtime_config.validate()

    args.runtime_config = runtime_config
    args.tracking_uri = runtime_config.mlflow_tracking_uri
    args.deployment_mode = runtime_config.deployment_mode
    args.mlflow_enable_controller_lineage = runtime_config.mlflow_enable_controller_lineage
    args.mlflow_enable_dataset_tracking = runtime_config.mlflow_enable_dataset_tracking
    args.mlflow_enable_logged_models = runtime_config.mlflow_enable_logged_models
    args.mlflow_enable_evaluation = runtime_config.mlflow_enable_evaluation
    args.mlflow_enable_tables = runtime_config.mlflow_enable_tables
    args.mlflow_enable_system_metrics = runtime_config.mlflow_enable_system_metrics
    args.mlflow_enable_registry = runtime_config.mlflow_enable_registry
    args.mlflow_controller_run_id = runtime_config.mlflow_controller_run_id
    setup_logging(verbosity=args.verbose)
    if getattr(args, "data_manifest_path", None):
        args.data_manifest_path_explicit = True
        args.prepared_manifest_paths = {"real": str(args.data_manifest_path)}
        if args.use_generated and not getattr(args, "tune", False):
            generated_manifest_paths = prepare_model_dataset_manifests(
                dataset_name=args.dataset_name,
                requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
                model_name=args.model_name,
                config_dir=args.config_dir,
                runtime_config=runtime_config,
                use_generated=True,
                verbosity=args.verbose,
            )
            args.prepared_manifest_paths["synthetic"] = generated_manifest_paths["synthetic"]
    elif getattr(args, "tune", False):
        args.data_manifest_path_explicit = False
    else:
        args.data_manifest_path_explicit = False
        args.prepared_manifest_paths = prepare_model_dataset_manifests(
            dataset_name=args.dataset_name,
            requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
            model_name=args.model_name,
            config_dir=args.config_dir,
            runtime_config=runtime_config,
            use_generated=args.use_generated,
            verbosity=args.verbose,
        )
        args.data_manifest_path = args.prepared_manifest_paths["real"]

    if not _is_local_execution(args):
        if ray is None:
            raise RuntimeError("Ray is required for the default execution backend.")

        def setup_func():
            setup_logging(verbosity=args.verbose)

        ray.init('auto',
                 runtime_env={"worker_process_setup_hook": setup_func},
                 _temp_dir=args.temp_dir)
    logger.info("CLI arguments: %s", args)

    args.datasource = KAGGLE_DATASET_URI

    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            text=True  # returns string instead of bytes
        )
        logger.debug(output)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.debug("nvidia-smi unavailable: %s", exc)

    # Attach tracking_uri for Ray Tune trainable function
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(experiment_id=args.experiment_id)

    if args.tune:
        logger.info("Tuning mode selected.")
        return execute_tuning_workflow(args)

    logger.info("Single-train mode selected.")
    return execute_single_train_workflow(args)
