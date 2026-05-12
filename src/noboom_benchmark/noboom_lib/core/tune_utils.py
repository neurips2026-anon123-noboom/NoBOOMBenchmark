from argparse import Namespace
from collections import OrderedDict
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import yaml

import boto3
import mlflow
from mlflow import MlflowClient
from mlflow.sklearn import (DEFAULT_AWAIT_MAX_SLEEP_SECONDS, Model,
                            ModelInputExample, ModelSignature,
                            SERIALIZATION_FORMAT_CLOUDPICKLE)
import numpy as np
try:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
except ModuleNotFoundError:  # pragma: no cover - exercised only in Ray-free environments
    tune = None  # type: ignore[assignment]
    ASHAScheduler = None  # type: ignore[assignment]
    FIFOScheduler = None  # type: ignore[assignment]

from .data_prep.runtime import manifest_paths_from_args
from .tune_constants import TUNE_PROGRESS_ATTR
from .tune.mlflow_tracking import build_benchmark_tags
from .tune.pruning import (
    build_final_eval_seed_context,
    build_reduced_hpo_seed_context,
    has_explicit_hpo_seeds,
    resolve_full_seeds,
    resolve_hpo_seeds,
)

logger = logging.getLogger(__name__)


class InvalidHyperparameterConfiguration(ValueError):
    """Raised when a sampled hyperparameter payload cannot instantiate the model."""


@lru_cache(maxsize=None)
def _load_lnt_model_init_args(config_dir: str) -> Dict[str, Any]:
    """Load the default LNT init args used to fill in unspecified encoder settings."""
    model_config_path = Path(config_dir).resolve() / "models" / "lnt.yaml"
    with model_config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    try:
        return dict(payload["model"]["network"]["init_args"])
    except (KeyError, TypeError) as exc:
        raise InvalidHyperparameterConfiguration(
            f"Could not load LNT init_args from '{model_config_path}'."
        ) from exc


def _lookup_config_value(payload: Mapping[str, Any], dotted_key: str) -> Any:
    """Resolve either a flattened dotted key or a nested mapping path."""
    if dotted_key in payload:
        return payload[dotted_key]

    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_lnt_init_arg(
    hp_params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    relative_key: str,
) -> Any:
    """Return an LNT init arg from the sampled payload or the model defaults."""
    for candidate in (
        f"model.network.init_args.{relative_key}",
        relative_key,
    ):
        resolved = _lookup_config_value(hp_params, candidate)
        if resolved is not None:
            return resolved
    return _lookup_config_value(defaults, relative_key)


def _coerce_int_sequence(value: Any, *, field_name: str) -> list[int]:
    """Convert an arbitrary sequence into a list of ints for encoder validation."""
    if isinstance(value, Mapping) or not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise InvalidHyperparameterConfiguration(
            f"LNT field '{field_name}' must be a concrete integer sequence; received {type(value)}."
        )

    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise InvalidHyperparameterConfiguration(
            f"LNT field '{field_name}' must contain only integers."
        ) from exc


def _minimum_input_length_for_strided_conv_stack(
    *,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    paddings: Sequence[int],
) -> int:
    """Compute the minimum input length that keeps every Conv1d stage non-empty."""
    if len(kernel_sizes) != len(strides) or len(strides) != len(paddings):
        raise InvalidHyperparameterConfiguration(
            "LNT Bosch CPC encoder config is inconsistent: filters, strides, and padding must have the same length."
        )

    required_input = 1
    for kernel_size, stride, padding in reversed(list(zip(kernel_sizes, strides, paddings))):
        required_input = stride * (required_input - 1) - (2 * padding) + kernel_size

    return max(1, required_input)


def validate_model_hparams(
    model_name: str,
    hp_params: Dict[str, Any],
    *,
    config_dir: str,
) -> None:
    """Validate a concrete model hyperparameter payload before launching training."""
    if model_name != "lnt" or not isinstance(hp_params, Mapping):
        return

    defaults = _load_lnt_model_init_args(config_dir)
    encoder_type = _resolve_lnt_init_arg(hp_params, defaults, "encoder_type")
    window_size = _lookup_config_value(hp_params, "window_size")
    if encoder_type is None or window_size is None:
        return
    if isinstance(encoder_type, Mapping) or isinstance(window_size, Mapping):
        return
    if str(encoder_type) != "bosch_cpc":
        return

    try:
        numeric_window_size = int(window_size)
    except (TypeError, ValueError) as exc:
        raise InvalidHyperparameterConfiguration(
            f"LNT window_size must be an integer, received {window_size!r}."
        ) from exc

    kernel_sizes = _coerce_int_sequence(
        _resolve_lnt_init_arg(hp_params, defaults, "encoder_cfg.filters"),
        field_name="encoder_cfg.filters",
    )
    strides = _coerce_int_sequence(
        _resolve_lnt_init_arg(hp_params, defaults, "encoder_cfg.strides"),
        field_name="encoder_cfg.strides",
    )
    paddings = _coerce_int_sequence(
        _resolve_lnt_init_arg(hp_params, defaults, "encoder_cfg.padding"),
        field_name="encoder_cfg.padding",
    )
    min_window_size = _minimum_input_length_for_strided_conv_stack(
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
    )
    if numeric_window_size < min_window_size:
        raise InvalidHyperparameterConfiguration(
            "Invalid LNT configuration: encoder_type='bosch_cpc' requires "
            f"window_size >= {min_window_size} with filters={kernel_sizes}, "
            f"strides={strides}, padding={paddings}; received window_size={numeric_window_size}."
        )


def is_s3_path(path: str) -> bool:
    """Check whether a path is an S3 URI.

    Args:
        path (str): Path or URI to evaluate.

    Returns:
        bool: True if the path starts with ``s3://``.
    """
    return path.startswith("s3://")


def create_study_name(model: str, dataset: str, timestamp: str) -> str:
    """Derive a deterministic study name for logging and reuse.

    Args:
        model (str): Model name.
        dataset (str): Dataset name.
        timestamp (str): Timestamp string used for uniqueness.

    Returns:
        str: Combined study name string.
    """
    return f"{model}__{dataset}__{timestamp}"


def is_storage_persistent() -> bool:
    """Check whether storage persistence is enabled via environment.

    Returns:
        bool: True if ``NOBOOM_PERSISTENT`` is set to "1".
    """
    val = os.environ.get("NOBOOM_PERSISTENT", "0") == "1"
    logger.debug(
        "NOBOOM_PERSISTENT=%s -> is_storage_persistent=%s",
        os.getenv("NOBOOM_PERSISTENT"),
        val,
    )
    return val


def jsonify_params(params: dict) -> dict:
    """Convert hyperparameter values into JSON-serializable primitives.

    Args:
        params (dict): Hyperparameter mapping containing possibly complex values.

    Returns:
        dict: JSON-serializable hyperparameter mapping.
    """

    def _convert(value: Any) -> Any:
        """Convert a single value into a JSON-serializable type.

        Args:
            value (Any): Value to convert.

        Returns:
            Any: JSON-serializable representation.
        """
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    converted = {k: _convert(v) for k, v in params.items()}
    logger.debug("JSON-serializable params: %s", converted)
    return converted

def copy_prefix(
    bucket: str,
    src_prefix: str,
    dst_prefix: str,
    *,
    s3_client=None,
    extra_args: Optional[dict] = None,
) -> int:
    """Copy all objects under one prefix to another within the same bucket.

    Args:
        bucket (str): S3 bucket name.
        src_prefix (str): Source prefix to copy from.
        dst_prefix (str): Destination prefix to copy to.
        s3_client (Any | None): Boto3 S3 client override. Defaults to None.
        extra_args (dict | None): Extra arguments for ``copy_object``. Defaults to None.

    Returns:
        int: Number of copied objects.

    Side Effects:
        Performs network I/O against S3.
    """
    s3 = s3_client or boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"))
    extra_args = extra_args or {}

    # Normalize prefixes
    if src_prefix and not src_prefix.endswith("/"):
        src_prefix += "/"
    if dst_prefix and not dst_prefix.endswith("/"):
        dst_prefix += "/"

    paginator = s3.get_paginator("list_objects_v2")
    copied = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=src_prefix):
        for obj in page.get("Contents", []):
            src_key = obj["Key"]
            # Preserve relative path under src_prefix
            rel = src_key[len(src_prefix):]
            dst_key = f"{dst_prefix}{rel}"

            if dst_key == src_key:
                continue

            copy_source = {"Bucket": bucket, "Key": src_key}

            # Copy object; preserve metadata by default
            s3.copy_object(
                Bucket=bucket,
                Key=dst_key,
                CopySource=copy_source,
                MetadataDirective="COPY",
                **extra_args,
            )
            copied += 1

    return copied


def load_hp_params(obj: Any) -> Optional[Dict[str, Any]]:
    """Extract hyperparameter params from metadata.

    Args:
        obj (Any): Raw ``hp_params`` value from metadata.

    Returns:
        Dict[str, Any] | None: Parsed hyperparameter mapping or None.
    """
    if obj is None:
        logger.debug("hp_params is None in metadata.")
        return None

    if isinstance(obj, dict):
        logger.debug("hp_params already dict.")
        return obj

    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                logger.debug("hp_params loaded from JSON string.")
                return parsed
            logger.debug("hp_params JSON string did not contain a dict.")
        except json.JSONDecodeError as e:
            logger.warning("Failed to decode hp_params JSON string: %s", e)
            return None

    logger.debug("hp_params is neither dict nor JSON-string, returning None.")
    return None


def dicts_match(candidate: Dict[str, Any], target: Dict[str, Any], *, exact: bool = True) -> bool:
    """Compare two dictionaries for exact or subset matches.

    Args:
        candidate (Dict[str, Any]): Candidate mapping to compare against.
        target (Dict[str, Any]): Target mapping to match.
        exact (bool): Require exact equality when True. Defaults to True.

    Returns:
        bool: True if the mappings match based on ``exact``.
    """
    if exact:
        match = candidate == target
        logger.debug("Exact dict comparison -> %s", match)
        return match

    for k, v in target.items():
        if k not in candidate or candidate[k] != v:
            logger.debug("Subset check failed at key '%s'.", k)
            return False
    logger.debug("Subset comparison succeeded.")
    return True


def find_matching_metadata(
    root_dir: str | Path,
    target_hp: Dict[str, Any],
    *,
    exact: bool = True,
    verbose: bool = True,
) -> Tuple[Optional[Path], Any]:
    """Search for metadata files whose hyperparameters match a target.

    Args:
        root_dir (str | Path): Root directory to search recursively.
        target_hp (Dict[str, Any]): Target hyperparameter mapping.
        exact (bool): Require exact dict equality when True. Defaults to True.
        verbose (bool): Whether to log warnings and info. Defaults to True.

    Returns:
        tuple[Path | None, Any]: Matching metadata path and parsed JSON content.

    Side Effects:
        Reads files from disk and logs progress.
    """
    root_dir = Path(root_dir)
    logger.info(
        "Searching for matching metadata.json under '%s' (exact=%s).",
        root_dir,
        exact,
    )
    logger.debug("Target hyperparameters: %s", target_hp)

    for meta_path in root_dir.rglob("metadata.json"):
        logger.debug("Checking metadata file: %s", meta_path)
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            if verbose:
                logger.warning("Failed to read %s: %s", meta_path, e)
            continue

        hp_raw = meta.get("hp_params")
        hp_dict = load_hp_params(hp_raw)
        if hp_dict is None:
            if verbose:
                logger.warning("No valid hp_params in %s", meta_path)
            continue

        if dicts_match(hp_dict, target_hp, exact=exact):
            if verbose:
                logger.info("Found matching metadata.json: %s", meta_path)
            return meta_path, meta

    if verbose:
        logger.info("No matching metadata.json files found.")
    return None, None


def save_seed_artifacts(
    seed_dir: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    metrics: Dict[str, Any],
    **kwargs,
) -> None:
    """Persist checkpoints and anomaly scores for a single seed.

    Args:
        seed_dir (Path): Directory to store artifacts.
        scores (np.ndarray): Array of anomaly scores.
        labels (np.ndarray): Array of labels aligned to scores.
        metrics (Dict[str, Any]): Metrics to serialize to JSON.
        **kwargs: Additional metrics to include in the JSON output.

    Returns:
        None: Artifacts are written to disk.

    Side Effects:
        Creates directories and writes JSON/NPZ files.
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving seed artifacts to '%s.", seed_dir)

    metrics_path = seed_dir / "metrics.json"
    full_metrics = metrics | kwargs
    logger.debug("Seed metrics: %s", full_metrics)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2, default=str)

    if scores is not None and labels is not None:
        scores_path = seed_dir / "anomaly_scores.npz"
        logger.info("Saving anomaly scores to '%s'.", scores_path)
        np.savez_compressed(scores_path, scores=scores, labels=labels)


def build_tune_search_space(search_space: dict, _top_level: bool = True) -> Tuple[dict, float]:
    """Recursively convert Optuna-style search space into Ray Tune format.

    Args:
        search_space (dict): Optuna-style search space configuration.
        _top_level (bool): Internal parameter to disable logging for low-level recursive calls.

    Returns:
        Tuple[dict, float]: Ray Tune-compatible search space and total combinations.

    Raises:
        TypeError: If ``search_space`` is not a mapping.
        ValueError: If an entry has an unknown distribution or invalid type.
    """
    if tune is None:
        raise RuntimeError("Ray Tune is required to build a Ray Tune search space.")

    if _top_level:
        logger.info("Building Ray Tune search space from Optuna-style config.")
    if not isinstance(search_space, dict):
        raise TypeError(f"search_space must be a mapping, got {type(search_space)}")

    def _count_discrete_range(low: float, high: float, step: float, *, inclusive: bool) -> int:
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        span = high - low
        if span < 0:
            return 0
        if inclusive:
            return max(0, int(math.floor(span / step) + 1))
        return max(0, int(math.ceil(span / step)))

    result = {}
    total_size: float = 1
    for name, spec in search_space.items():
        logger.debug("Processing search space entry '%s': %s", name, spec)
        if isinstance(spec, dict) and "distribution" in spec:
            dist = spec["distribution"]
            step = spec.get("step", None)
            size: float

            if dist == "uniform":
                if step is not None:
                    result[name] = tune.quniform(float(spec["low"]), float(spec["high"]), q=float(step))
                else:
                    result[name] = tune.uniform(float(spec["low"]), float(spec["high"]))
                size = math.inf
            elif dist == "loguniform":
                if step is not None:
                    result[name] = tune.qloguniform(float(spec["low"]), float(spec["high"]), q=float(step))
                else:
                    result[name] = tune.loguniform(float(spec["low"]), float(spec["high"]))
                size = math.inf
            elif dist == "int":
                if step is not None:
                    result[name] = tune.qrandint(int(spec["low"]), int(spec["high"]), q=int(step))
                    size = _count_discrete_range(
                        int(spec["low"]), int(spec["high"]), int(step), inclusive=True
                    )
                else:
                    result[name] = tune.randint(int(spec["low"]), int(spec["high"]) + 1)
                    size = _count_discrete_range(
                        int(spec["low"]), int(spec["high"]), 1, inclusive=True
                    )
            elif dist == "categorical":
                result[name] = tune.choice(spec["choices"])
                size = len(spec["choices"])
            else:
                raise ValueError(f"Unknown distribution type: {dist}")
            total_size *= size
        elif isinstance(spec, dict):
            subspace, sub_size = build_tune_search_space(spec, _top_level=False)
            result[name] = subspace
            total_size *= sub_size
        else:
            raise ValueError(
                f"Invalid spec at '{name}': expected mapping, got {type(spec)}"
            )

    if _top_level:
        logger.info("Search space construction completed.")
    return result, total_size


def build_tune_scheduler(
    pruner_cfg: Optional[dict],
):
    """Create a Ray Tune scheduler from the provided pruner configuration.

    Args:
        pruner_cfg (dict | None): Pruner configuration mapping. Defaults to None.

    Returns:
        ASHAScheduler | FIFOScheduler: Configured scheduler instance.
    """
    if ASHAScheduler is None or FIFOScheduler is None:
        raise RuntimeError("Ray Tune is required to build a Ray Tune scheduler.")

    if pruner_cfg is None:
        logger.info("No pruner_cfg provided -> using FIFOScheduler.")
        return FIFOScheduler()

    scheduler_name = pruner_cfg.get("name", "ASHA")
    scheduler_args = pruner_cfg.get("args", {})
    if scheduler_name != "ASHA":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    max_t = int(scheduler_args.get("max_t", 100))
    grace_period = int(scheduler_args.get("grace_period", 1))

    logger.info(
        "Creating ASHAScheduler with pruner_cfg=%s (max_t=%d, grace_period=%d)",
        pruner_cfg,
        max_t,
        grace_period,
    )
    return ASHAScheduler(
        time_attr=TUNE_PROGRESS_ATTR,
        max_t=max_t,
        grace_period=grace_period,
    )


def load_saved_best_params(args: Namespace, study_dir: Path, model_name: str, dataset_name: str) -> dict:
    """Load persisted best hyperparameters and metadata for reuse.

    Args:
        args (Namespace): Parsed arguments for the study.
        study_dir (Path): Study directory containing saved metadata.
        model_name (str): Model name (for logging context).
        dataset_name (str): Dataset name (for logging context).

    Returns:
        dict: Saved best parameters metadata.

    Raises:
        FileNotFoundError: If the best params file does not exist.
    """
    metadata_path = study_dir / "best_params.json"

    logger.info("Loading saved best params from '%s'.", metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No saved best parameters found at {metadata_path}. Run hyperparameter search first."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.debug("Loaded best params: %s", metadata)
    return metadata


def save_metadata(
    args: Namespace,
    artifact_dir,
    hp_params: dict,
) -> OrderedDict:
    """Build and return metadata describing a tuning trial.

    Args:
        args (Namespace): Parsed arguments containing model and dataset info.
        artifact_dir (Any): Directory in which metadata is stored.
        hp_params (dict): Hyperparameter mapping for the trial.

    Returns:
        OrderedDict: Metadata payload ready to serialize.

    Side Effects:
        Creates the artifact directory if needed.
    """
    logger.info("Saving metadata to '%s'.", artifact_dir)
    full_seeds = resolve_full_seeds(args.study_params)
    metadata = OrderedDict(
        [
            ("model", args.model_name),
            ("dataset", args.dataset_name),
            ("timestamp", args.timestamp),
            ("seeds", full_seeds),
            ("tune_metric", args.study_params["tune_metric"]),
            (
                "tune_mode",
                "max" if args.study_params.get("direction") == "maximize" else "min",
            ),
            (
                "evaluation_postprocessing",
                jsonify_params(args.study_params.get("evaluation_postprocessing", {})),
            ),
            ("checkpoint_subdir", args.timestamp),
            ("hp_params", jsonify_params(hp_params)),
        ]
    )
    if has_explicit_hpo_seeds(args.study_params):
        metadata["hpo_seeds"] = resolve_hpo_seeds(args.study_params)
        metadata["final_eval_seeds"] = full_seeds
        if bool(getattr(args, "tune", False)):
            metadata.update(build_reduced_hpo_seed_context(args.study_params))
        else:
            metadata.update(build_final_eval_seed_context(args.study_params))
    result_seed_context = getattr(args, "result_seed_context", None)
    if isinstance(result_seed_context, Mapping):
        metadata.update(dict(result_seed_context))
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = artifact_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    logger.info("Metadata written to '%s'.", metadata_path)
    logger.debug("Metadata content: %s", metadata)
    return metadata


def create_extra_params_for_lightning(
    args: Namespace,
    logger_params: Dict[str, Any],
    seed: int,
    hp_params: Dict[str, Any],
) -> str:
    """Create a temporary Lightning config file with seed and logger params.

    Args:
        args (Namespace): Parsed arguments with study and model info.
        logger_params (Dict[str, Any]): Logger-specific parameters to inject.
        seed (int): Seed value to set for the run.
        hp_params (Dict[str, Any]): Hyperparameters to write to the config file.

    Returns:
        str: Path to the temporary YAML file.

    Side Effects:
        Writes a temporary YAML file to disk.
    """
    logger.info(
        "Creating temporary Lightning params for seed=%d, model=%s, dataset=%s.",
        seed,
        args.model_name,
        args.dataset_name,
    )
    hp_params_s = hp_params.copy()
    hp_params_s["seed_everything"] = seed
    hp_params_s["model.metrics"] = args.study_params["metrics"]
    if args.study_params.get("evaluation_postprocessing") is not None:
        hp_params_s["model.evaluation_postprocessing"] = args.study_params["evaluation_postprocessing"]
    if "data.data_manifest_path" not in hp_params_s:
        stage_data_source = str(hp_params_s.get("data.data_source", "real"))
        try:
            hp_params_s["data.data_manifest_path"] = manifest_paths_from_args(args)[stage_data_source]
        except (KeyError, ValueError):
            if stage_data_source == "real" and getattr(args, "data_manifest_path", None):
                hp_params_s["data.data_manifest_path"] = args.data_manifest_path

    for name, value in logger_params.items():
        hp_params_s[f"trainer.logger.init_args.{name}"] = value
    save_checkpoints = bool(getattr(args, "save_checkpoints", False))
    hp_params_s["trainer.logger.init_args.log_model"] = save_checkpoints
    hp_params_s["trainer.logger.init_args.log_checkpoints"] = save_checkpoints

    if args.model_name in ["kmeans", "eif", "ocsvm", "pca", "hbos"]:
        logger.info("Model '%s' is CPU-only baseline; adjusting trainer params.", args.model_name)
        hp_params_s["trainer.max_epochs"] = 1
        hp_params_s["trainer.limit_train_batches"] = 0
        hp_params_s["trainer.limit_val_batches"] = 0
        hp_params_s["trainer.accelerator"] = "cpu"

    fd, hp_path = tempfile.mkstemp(suffix=".yaml", text=True)
    with os.fdopen(fd, "w") as f:
        yaml.dump(hp_params_s, f)
    logger.debug("Temporary Lightning config written to '%s'.", hp_path)
    return hp_path


def flatten_dict(nested: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary into a single-level mapping.

    Args:
        nested (dict): Nested mapping to flatten.
        parent_key (str): Prefix to apply to keys. Defaults to "".
        sep (str): Separator to use between key segments. Defaults to "/".

    Returns:
        dict: Flattened mapping.
    """
    flat = {}
    for key, value in nested.items():
        key = str(key)
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key, sep=sep))
        else:
            flat[new_key] = value
    return flat


def setup_seed_mlflow(
    args: Namespace,
    hp_params: Dict[str, Any],
    client: MlflowClient,
    experiment_id: str,
    trial_name: str,
    trial_run_id: str,
    seed: int,
) -> str:
    """Create an MLflow run for a specific seed and log parameters.

    Args:
        args (Namespace): Parsed arguments containing dataset/model metadata.
        hp_params (Dict[str, Any]): Hyperparameters to log.
        client (MlflowClient): MLflow client used to create and log runs.
        experiment_id (str): MLflow experiment ID.
        trial_name (str): Trial name used for the run.
        trial_run_id (str): Parent trial run ID.
        seed (int): Seed value for the run.

    Returns:
        str: MLflow run ID for the seed run.

    Side Effects:
        Creates an MLflow run and logs parameters.
    """
    seed_run_name = f"{trial_name}__seed_{seed}"
    seed_context = getattr(args, "result_seed_context", None)
    if isinstance(seed_context, Mapping):
        seed_extra_tags = dict(seed_context)
    elif has_explicit_hpo_seeds(args.study_params):
        seed_extra_tags = (
            build_reduced_hpo_seed_context(args.study_params)
            if bool(getattr(args, "tune", False))
            else build_final_eval_seed_context(args.study_params)
        )
    else:
        seed_extra_tags = {}

    seed_tags = build_benchmark_tags(
        timestamp=args.timestamp,
        run_level="seed",
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        requested_dataset_name=getattr(args, "requested_dataset_name", args.dataset_name),
        run_mode="tune" if args.tune else "single_train",
        deployment_mode=getattr(args, "deployment_mode", None),
        parent_run_id=trial_run_id,
        seed=seed,
        extra_tags=seed_extra_tags,
    )

    logger.info(
        "Creating MLflow seed run: experiment_id=%s, trial_name=%s, seed=%d.",
        experiment_id,
        trial_name,
        seed,
    )
    seed_run = client.create_run(experiment_id, tags=seed_tags, run_name=seed_run_name)
    run_id = seed_run.info.run_id
    client.log_param(run_id, "seed", seed)
    mlflow.log_params(flatten_dict(hp_params), run_id=run_id)
    logger.debug("Created seed run with run_id=%s.", run_id)
    return run_id


def sklearn_log_model(
    sk_model,
    artifact_path: Optional[str] = None,
    conda_env=None,
    code_paths=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn="predict",
    metadata=None,
    # New arguments
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, Any]] = None,
    model_type: Optional[str] = None,
    step: int = 0,
    model_id: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs,
):
    """Log a scikit-learn model as an MLflow artifact.

    Args:
        sk_model (Any): scikit-learn model to be saved.
        artifact_path (str | None): Deprecated MLflow artifact path. Defaults to None.
        conda_env (Any | None): Conda environment spec for MLflow. Defaults to None.
        code_paths (Any | None): Code paths to include. Defaults to None.
        serialization_format (Any): Serialization format to use. Defaults to
            ``SERIALIZATION_FORMAT_CLOUDPICKLE``.
        registered_model_name (Any | None): Registered model name. Defaults to None.
        signature (ModelSignature | None): Model signature. Defaults to None.
        input_example (ModelInputExample | None): Input example. Defaults to None.
        await_registration_for (Any): Wait time for registration. Defaults to
            ``DEFAULT_AWAIT_MAX_SLEEP_SECONDS``.
        pip_requirements (Any | None): Pip requirements. Defaults to None.
        extra_pip_requirements (Any | None): Extra pip requirements. Defaults to None.
        pyfunc_predict_fn (str): Prediction function name. Defaults to "predict".
        metadata (Any | None): MLflow metadata. Defaults to None.
        params (dict[str, Any] | None): Params to log with the model. Defaults to None.
        tags (dict[str, Any] | None): Tags to attach to the model. Defaults to None.
        model_type (str | None): Optional model type metadata. Defaults to None.
        step (int): Step value for logging. Defaults to 0.
        model_id (str | None): Optional model ID. Defaults to None.
        name (str | None): Model name to use. Defaults to None.
        **kwargs: Additional keyword arguments forwarded to ``Model.log``.

    Returns:
        ModelInfo: Metadata about the logged model.

    Side Effects:
        Logs a model artifact to the active MLflow run.
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=mlflow.sklearn,
        sk_model=sk_model,
        conda_env=conda_env,
        code_paths=code_paths,
        serialization_format=serialization_format,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        pyfunc_predict_fn=pyfunc_predict_fn,
        metadata=metadata,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
        **kwargs
    )
