"""Orchestrate per-pair tune runs for NoBoom benchmark."""
from __future__ import annotations

from argparse import Namespace
import itertools
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient
try:
    import ray
except ModuleNotFoundError:  # pragma: no cover - exercised only in Ray-free environments
    ray = None  # type: ignore[assignment]

from noboom_cluster.noboom_cli_lib.dependency_resolver import DependencyResolver
from noboom_cluster.noboom_cli_lib.settings import ControllerSettings
from noboom_cluster.noboom_cli_lib.specs import (
    ControllerRunSpec,
    DatasetModelPair,
    parse_dataset_model_pair,
)

from ..config.runtime import RuntimeConfig
from ..logging import setup_logging
from ..notifications import get_shared_dispatcher
from .callbacks import MlflowSeafilePairOutputCallback
from .job_submission import LocalPairJobController, PairJobController, build_pair_run_spec
from .mlflow_tracking import (
    build_pair_summary_rows,
    log_code_snapshot,
    log_dependency_manifest,
    log_dict_artifact,
    log_table_artifact,
    start_controller_run,
)
from .notification_callbacks import (
    NotificationPairOutputCallback,
    emit_controller_start,
    emit_controller_terminal,
)
from .storage import upload_to_seafile

logger = logging.getLogger(__name__)


def _canonical_model_name(model_name: str) -> str:
    return model_name.strip().lower()


def _canonical_pair(pair: DatasetModelPair) -> DatasetModelPair:
    return DatasetModelPair(
        dataset=pair.dataset.strip(),
        model=_canonical_model_name(pair.model),
    )


def _parse_explicit_pairs(raw_pairs: Iterable[object]) -> List[DatasetModelPair]:
    pairs: List[DatasetModelPair] = []
    for raw_pair in raw_pairs:
        if isinstance(raw_pair, DatasetModelPair):
            pairs.append(_canonical_pair(raw_pair))
            continue
        pairs.append(_canonical_pair(parse_dataset_model_pair(str(raw_pair))))
    return pairs


def _dedupe_pairs(pairs: Iterable[Tuple[str, str]]) -> List[DatasetModelPair]:
    seen: set[Tuple[str, str]] = set()
    deduped: List[DatasetModelPair] = []
    for dataset_name, model_name in pairs:
        key = (dataset_name, model_name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(DatasetModelPair(dataset=dataset_name, model=model_name))
    return deduped


def _build_controller_spec(args: Namespace) -> ControllerRunSpec:
    datasets = sorted(set(getattr(args, "dataset", None) or []))
    raw_models = getattr(args, "model", None) or []
    models = sorted({_canonical_model_name(model_name) for model_name in raw_models})
    explicit_pairs = _parse_explicit_pairs(getattr(args, "pair", None) or [])
    dataset_model_pairs = _dedupe_pairs(
        itertools.chain(
            itertools.product(datasets, models),
            (pair.as_tuple() for pair in explicit_pairs),
        )
    )
    if not dataset_model_pairs:
        raise ValueError("Provide at least one dataset-model pair via --dataset/--model or --pair.")

    execution_backend = str(getattr(args, "execution_backend", "ray") or "ray")
    artifact_storage_backend = getattr(args, "artifact_storage_backend", None)
    if artifact_storage_backend is None:
        artifact_storage_backend = "local" if execution_backend == "local" else "remote"
    optuna_storage_backend = getattr(args, "optuna_storage_backend", None)
    if optuna_storage_backend is None:
        optuna_storage_backend = "memory" if execution_backend == "local" else "configured"

    return ControllerRunSpec(
        datasets=datasets,
        models=models,
        dataset_model_pairs=dataset_model_pairs,
        tune=args.tune,
        source_experiment_id=args.experiment_id if not args.tune else None,
        config_dir=args.config_dir,
        temp_dir=args.temp_dir,
        gpus_per_run=args.gpus_per_run,
        timestamp=args.timestamp,
        verbose=args.verbose,
        env_file=args.env_file,
        max_in_flight=getattr(args, "max_in_flight", 4),
        save_checkpoints=bool(getattr(args, "save_checkpoints", False)),
        hpo_seeds=getattr(args, "hpo_seeds", None),
        execution_backend=execution_backend,
        artifact_storage_backend=str(artifact_storage_backend),
        optuna_storage_backend=str(optuna_storage_backend),
        local_storage_path=getattr(args, "local_storage_path", None),
        optuna_sqlite_path=getattr(args, "optuna_sqlite_path", None),
        ram_guard_enabled=bool(getattr(args, "ram_guard", True)),
        ram_guard_max_used_fraction=float(getattr(args, "ram_guard_max_used_fraction", 0.95)),
        ram_guard_poll_interval_s=float(getattr(args, "ram_guard_poll_interval_s", 10.0)),
        ram_guard_resume_used_fraction=float(getattr(args, "ram_guard_resume_used_fraction", 0.90)),
        ram_guard_cooldown_s=float(getattr(args, "ram_guard_cooldown_s", 120.0)),
    )


def _local_storage_root(args: Namespace, timestamp: str) -> Path:
    configured_path = getattr(args, "local_storage_path", None)
    if configured_path:
        return Path(str(configured_path)).expanduser().resolve()
    return (Path.cwd() / ".noboom_local" / timestamp).resolve()


def _resolve_optuna_storage_uri(args: Namespace, storage_root: Path) -> str:
    backend = str(getattr(args, "optuna_storage_backend", None) or "configured")
    if backend == "memory":
        return "memory://"
    if backend == "sqlite":
        sqlite_path = getattr(args, "optuna_sqlite_path", None)
        db_path = Path(str(sqlite_path)).expanduser() if sqlite_path else storage_root / "optuna.db"
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"
    configured = os.getenv("OPTUNA_STORAGE_URI")
    if configured:
        return configured
    return "memory://"


def _configure_local_controller_environment(
    args: Namespace,
    controller_settings: ControllerSettings,
    controller_spec: ControllerRunSpec,
) -> Path:
    args.execution_backend = controller_spec.execution_backend
    args.artifact_storage_backend = controller_spec.artifact_storage_backend
    args.optuna_storage_backend = controller_spec.optuna_storage_backend
    storage_root = _local_storage_root(args, controller_spec.timestamp)
    storage_root.mkdir(parents=True, exist_ok=True)
    experiment_name = (
        controller_settings.experiment_name
        or os.getenv("EXPERIMENT_NAME")
        or f"NoBoomBenchmark__{controller_spec.timestamp}"
    )
    mlflow_tracking_uri = (
        controller_settings.mlflow_tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or f"file://{storage_root / 'mlruns'}"
    )
    optuna_storage_uri = _resolve_optuna_storage_uri(args, storage_root)

    controller_settings.experiment_name = experiment_name
    controller_settings.nooboom_mapped_storage = str(storage_root)
    controller_settings.mlflow_tracking_uri = mlflow_tracking_uri
    controller_settings.optuna_storage_uri = optuna_storage_uri
    controller_settings.nooboom_s3_bucket = None
    controller_settings.nooboom_prepared_dataset_s3_path = None
    controller_settings.s3_endpoint_url = None
    controller_settings.seafile_username = ""
    controller_settings.seafile_pass = ""
    controller_settings.seafile_root_path = ""
    controller_settings.nooboom_seafile_upload_results = False

    os.environ["EXPERIMENT_NAME"] = str(experiment_name)
    os.environ["NOBOOM_MAPPED_STORAGE"] = str(storage_root)
    os.environ["MLFLOW_TRACKING_URI"] = str(mlflow_tracking_uri)
    os.environ["OPTUNA_STORAGE_URI"] = str(optuna_storage_uri)
    os.environ["NOBOOM_EXECUTION_BACKEND"] = "local"
    os.environ["NOBOOM_DEPLOYMENT_MODE"] = "local"
    os.environ["NOBOOM_S3_BUCKET"] = ""
    os.environ["NOBOOM_PREPARED_DATASET_S3_PATH"] = ""
    os.environ["S3_ENDPOINT_URL"] = ""
    os.environ["SEAFILE_USERNAME"] = ""
    os.environ["SEAFILE_PASS"] = ""
    os.environ["SEAFILE_ROOT_PATH"] = ""
    os.environ["NOBOOM_SEAFILE_UPLOAD_RESULTS"] = "0"
    return storage_root


def run_tune(args: Namespace) -> int:
    if args.env_file is not None:
        load_dotenv(args.env_file)

    setup_logging(args.verbose)
    controller_settings = ControllerSettings()
    controller_spec = _build_controller_spec(args)
    controller_settings_overrides = {
        "NOBOOM_RAM_GUARD_ENABLED": "1" if controller_spec.ram_guard_enabled else "0",
        "NOBOOM_RAM_GUARD_MAX_USED_FRACTION": str(controller_spec.ram_guard_max_used_fraction),
        "NOBOOM_RAM_GUARD_POLL_INTERVAL_S": str(controller_spec.ram_guard_poll_interval_s),
        "NOBOOM_RAM_GUARD_RESUME_USED_FRACTION": str(controller_spec.ram_guard_resume_used_fraction),
        "NOBOOM_RAM_GUARD_COOLDOWN_S": str(controller_spec.ram_guard_cooldown_s),
    }
    os.environ.update(controller_settings_overrides)
    controller_settings.nooboom_ram_guard_enabled = controller_spec.ram_guard_enabled
    controller_settings.nooboom_ram_guard_max_used_fraction = controller_spec.ram_guard_max_used_fraction
    controller_settings.nooboom_ram_guard_poll_interval_s = controller_spec.ram_guard_poll_interval_s
    controller_settings.nooboom_ram_guard_resume_used_fraction = controller_spec.ram_guard_resume_used_fraction
    controller_settings.nooboom_ram_guard_cooldown_s = controller_spec.ram_guard_cooldown_s
    local_mode = controller_spec.execution_backend == "local"
    if local_mode:
        _configure_local_controller_environment(args, controller_settings, controller_spec)
    else:
        controller_settings.validate_required()
    runtime_config = RuntimeConfig.from_env_and_args(args)
    runtime_config.validate()
    notification_dispatcher = get_shared_dispatcher(reset=True)
    run_datasets = sorted({pair.dataset for pair in controller_spec.dataset_model_pairs})
    run_models = sorted({pair.model for pair in controller_spec.dataset_model_pairs})

    logger.info("Controller run spec: %s", controller_spec.model_dump())
    mlflow.set_tracking_uri(str(controller_settings.mlflow_tracking_uri))

    experiment_name = str(controller_settings.experiment_name)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name)
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    client = MlflowClient(str(controller_settings.mlflow_tracking_uri))
    prepared_dataset_s3_path = (
        getattr(args, "prepared_dataset_s3_path", None)
        or getattr(controller_settings, "nooboom_prepared_dataset_s3_path", None)
    )

    controller_run_context = None
    if bool(getattr(runtime_config, "mlflow_enable_controller_lineage", False)):
        controller_run_context = start_controller_run(
            client,
            experiment_id,
            timestamp=controller_spec.timestamp,
            tune_mode=controller_spec.tune,
            deployment_mode=getattr(runtime_config, "deployment_mode", ""),
            datasets=run_datasets,
            models=run_models,
        )
        controller_settings.nooboom_mlflow_controller_run_id = controller_run_context.run_id

    local_storage_path = Path(str(controller_settings.nooboom_mapped_storage)) / experiment_name
    local_storage_path.mkdir(parents=True, exist_ok=True)

    if controller_run_context is not None:
        log_code_snapshot(
            client,
            controller_run_context.run_id,
            storage_path=str(local_storage_path),
        )

    dependency_manifest = DependencyResolver().resolve()
    manifest_path = local_storage_path / "dependency-manifest.json"
    dependency_manifest.write_json(str(manifest_path))
    if controller_run_context is not None:
        log_dependency_manifest(
            client,
            controller_run_context.run_id,
            manifest_path=str(manifest_path),
        )

    if (
        not local_mode
        and bool(getattr(controller_settings, "nooboom_seafile_upload_results", False))
        and controller_settings.seafile_username
        and controller_settings.seafile_root_path
    ):
        upload_to_seafile(manifest_path, Path(experiment_name) / "metadata")

    if not local_mode:
        if ray is None:
            raise RuntimeError("Ray is required for the default execution backend.")
        ray.init(
            "auto",
            runtime_env={"worker_process_setup_hook": lambda: setup_logging(verbosity=args.verbose)},
            _temp_dir=args.temp_dir,
        )

    try:
        emit_controller_start(
            timestamp=controller_spec.timestamp,
            tune_mode=controller_spec.tune,
            datasets=run_datasets,
            models=run_models,
            pair_count=len(controller_spec.dataset_model_pairs),
            controller_run_id=(
                controller_run_context.run_id if controller_run_context is not None else None
            ),
        )
        pair_output_callback = MlflowSeafilePairOutputCallback(
            experiment_root=local_storage_path,
            experiment_name=experiment_name,
            runtime_config=runtime_config,
            config_dir=controller_spec.config_dir,
            temp_dir=controller_spec.temp_dir,
            tracking_uri=str(controller_settings.mlflow_tracking_uri),
        )
        callback = NotificationPairOutputCallback(
            pair_output_callback,
            dispatcher=notification_dispatcher,
            controller_run_id=(
                controller_run_context.run_id
                if controller_run_context is not None
                else getattr(controller_settings, "nooboom_mlflow_controller_run_id", None)
            ),
        )

        pair_specs_dir = Path(".noboom_runtime") / "pair_specs"
        pair_specs_dir.mkdir(parents=True, exist_ok=True)

        pair_specs = [
            build_pair_run_spec(
                experiment_id=experiment_id,
                source_experiment_id=controller_spec.source_experiment_id,
                model_name=model_name,
                dataset_name=dataset_name,
                timestamp=controller_spec.timestamp,
                local_storage_path=local_storage_path,
                optuna_storage_uri=str(controller_settings.optuna_storage_uri),
                tracking_uri=str(controller_settings.mlflow_tracking_uri),
                config_dir=controller_spec.config_dir,
                temp_dir=controller_spec.temp_dir,
                data_manifest_path=None,
                verbose=controller_spec.verbose,
                tune_mode=controller_spec.tune,
                gpus_per_run=controller_spec.gpus_per_run,
                s3_endpoint_url=controller_settings.s3_endpoint_url,
                prepared_dataset_s3_path=prepared_dataset_s3_path,
                env_file=controller_spec.env_file,
                save_checkpoints=controller_spec.save_checkpoints,
                hpo_seeds=controller_spec.hpo_seeds,
                execution_backend=controller_spec.execution_backend,
                artifact_storage_backend=controller_spec.artifact_storage_backend,
                optuna_storage_backend=controller_spec.optuna_storage_backend,
            )
            for pair in controller_spec.dataset_model_pairs
            for dataset_name, model_name in [pair.as_tuple()]
        ]

        controller_cls = LocalPairJobController if local_mode else PairJobController
        env_vars = (
            controller_settings.to_local_pair_job_env()
            if local_mode
            else controller_settings.to_pair_job_env()
        )
        job_controller = controller_cls(
            dependency_manifest=dependency_manifest,
            pair_specs_dir=pair_specs_dir,
            callback=callback,
            env_vars=env_vars,
            max_in_flight=controller_spec.max_in_flight,
        )
        pair_results = job_controller.submit_jobs(pair_specs)
        if controller_run_context is not None:
            status_counts = {}
            for pair_result in pair_results:
                status = str(getattr(pair_result, "status", "SUCCEEDED")).lower()
                status_counts[f"pair_count_{status}"] = status_counts.get(f"pair_count_{status}", 0) + 1
            controller_summary = {
                "datasets": run_datasets,
                "models": run_models,
                "dataset_model_pairs": [
                    {"dataset": pair.dataset, "model": pair.model}
                    for pair in controller_spec.dataset_model_pairs
                ],
                "timestamp": controller_spec.timestamp,
                "run_mode": "tune" if controller_spec.tune else "single_train",
                "deployment_mode": getattr(runtime_config, "deployment_mode", ""),
                "pair_count": len(pair_results),
                "controller_run_id": controller_run_context.run_id,
            } | status_counts
            log_dict_artifact(
                client,
                controller_run_context.run_id,
                artifact_file="summary/controller.json",
                payload=controller_summary,
            )
            if bool(getattr(runtime_config, "mlflow_enable_tables", False)):
                log_table_artifact(
                    client,
                    controller_run_context.run_id,
                    artifact_file="tables/pairs.json",
                    rows=build_pair_summary_rows(pair_results),
                )
            excel_path = local_storage_path / "noboom_experiments.xlsx"
            if excel_path.exists():
                client.log_artifact(
                    controller_run_context.run_id,
                    str(excel_path),
                    artifact_path="reports",
                )
            client.set_terminated(controller_run_context.run_id)
        emit_controller_terminal(
            status="SUCCEEDED",
            timestamp=controller_spec.timestamp,
            tune_mode=controller_spec.tune,
            datasets=run_datasets,
            models=run_models,
            pair_count=len(pair_results),
            controller_run_id=(
                controller_run_context.run_id if controller_run_context is not None else None
            ),
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled controller exception: %s", exc)
        if controller_run_context is not None:
            client.set_terminated(controller_run_context.run_id, status="FAILED")
        emit_controller_terminal(
            status="FAILED",
            timestamp=controller_spec.timestamp,
            tune_mode=controller_spec.tune,
            datasets=run_datasets,
            models=run_models,
            pair_count=len(controller_spec.dataset_model_pairs),
            controller_run_id=(
                controller_run_context.run_id if controller_run_context is not None else None
            ),
            message=exc.__class__.__name__,
        )
        return 1
