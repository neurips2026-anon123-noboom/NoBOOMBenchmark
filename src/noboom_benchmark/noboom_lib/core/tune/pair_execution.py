from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict, Tuple

from noboom_cluster.noboom_cli_lib.specs import PairRunSpec


def pair_spec_to_namespace(pair_spec: PairRunSpec) -> Namespace:
    return Namespace(
        experiment_id=pair_spec.experiment_id,
        source_experiment_id=pair_spec.source_experiment_id,
        model_name=pair_spec.model_name,
        dataset_name=pair_spec.dataset_name,
        timestamp=pair_spec.timestamp,
        storage_path=pair_spec.storage_path,
        gpus_per_run=pair_spec.gpus_per_run,
        optuna_storage_uri=pair_spec.optuna_storage_uri,
        config_dir=pair_spec.config_dir,
        temp_dir=pair_spec.temp_dir,
        data_manifest_path=pair_spec.data_manifest_path,
        verbose=pair_spec.verbose,
        tune=pair_spec.tune,
        env_file=pair_spec.env_file,
        tracking_uri=pair_spec.tracking_uri,
        s3_endpoint_url=pair_spec.s3_endpoint_url,
        prepared_dataset_s3_path=pair_spec.prepared_dataset_s3_path,
        save_checkpoints=pair_spec.save_checkpoints,
        hpo_seeds=pair_spec.hpo_seeds,
        execution_backend=pair_spec.execution_backend,
        artifact_storage_backend=pair_spec.artifact_storage_backend,
        optuna_storage_backend=pair_spec.optuna_storage_backend,
    )


def run_pair_spec(pair_spec: PairRunSpec) -> Tuple[str, str, Dict[str, Any]]:
    from ..tuning_runner import run_tune_or_train

    return run_tune_or_train(pair_spec_to_namespace(pair_spec))


def run_pair_spec_file(path: str) -> Tuple[str, str, Dict[str, Any]]:
    return run_pair_spec(PairRunSpec.load(path))
