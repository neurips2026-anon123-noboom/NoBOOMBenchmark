"""Runtime helpers for building and resolving prepared dataset manifests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from filelock import FileLock
from omegaconf import OmegaConf

from ..config.runtime import RuntimeConfig
from ..tune.storage import create_s3_client, ensure_bucket_exists
from .manifest import PreparedDatasetManifest
from .pipeline import (
    build_preparation_recipe,
    fingerprint_recipe,
    run_preparation_pipeline,
)


logger = logging.getLogger(__name__)

PREPARED_BUNDLE_REMOTE_ROOT = "prepared_datasets"
PREPARED_BUNDLE_MANIFEST_NAME = "manifest.json"
PREPARED_BUNDLE_BUILD_LOCK_NAME = ".build.lock"
PREPARED_BUNDLE_REMOTE_WAIT_TIMEOUT_S = 3600.0
PREPARED_BUNDLE_REMOTE_WAIT_POLL_S = 2.0


def _normalize_scaler_spec(spec: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    return {
        "class_path": str(spec["class_path"]),
        "init_args": dict(spec.get("init_args", {})),
    }


def _merge_scaler_init_args_from_hparams(
    scaler_spec: Optional[Mapping[str, Any]],
    hp_params: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    normalized_scaler_spec = _normalize_scaler_spec(scaler_spec)
    if hp_params is None:
        return normalized_scaler_spec

    data_params = hp_params.get("data")
    if not isinstance(data_params, Mapping):
        return normalized_scaler_spec
    scaler_params = data_params.get("scaler")
    if not isinstance(scaler_params, Mapping):
        return normalized_scaler_spec
    init_args = scaler_params.get("init_args")
    if not isinstance(init_args, Mapping):
        return normalized_scaler_spec
    if normalized_scaler_spec is None:
        return None

    merged_scaler_spec = {
        "class_path": normalized_scaler_spec["class_path"],
        "init_args": dict(normalized_scaler_spec.get("init_args", {})),
    }
    merged_scaler_spec["init_args"].update(dict(init_args))
    return merged_scaler_spec


def _spec_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _spec_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_spec_to_dict(item) for item in value]
    if hasattr(value, "as_dict"):
        return _spec_to_dict(value.as_dict())
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return {
            key: _spec_to_dict(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _load_model_data_config(config_dir: str, model_name: str) -> Dict[str, Any]:
    common_cfg = OmegaConf.load(str(Path(config_dir) / "models" / "common.yaml"))
    model_cfg = OmegaConf.load(str(Path(config_dir) / "models" / f"{model_name}.yaml"))
    merged_cfg = OmegaConf.merge(common_cfg, model_cfg)
    resolved_cfg = OmegaConf.to_container(merged_cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        raise ValueError(f"Unexpected resolved config payload for model '{model_name}'.")
    data_cfg = resolved_cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError(f"Unexpected data config payload for model '{model_name}'.")
    return data_cfg


def _prepared_bundle_remote_location(
    runtime_config: RuntimeConfig,
    *,
    fingerprint: str,
) -> Optional[Tuple[str, str]]:
    bucket: Optional[str]
    root_prefix: str
    override_path = getattr(runtime_config, "prepared_dataset_s3_path", None)
    if override_path:
        normalized_override = str(override_path).strip()
        if normalized_override.startswith("s3://"):
            parsed = urlparse(normalized_override)
            bucket = parsed.netloc.strip("/") or None
            if bucket is None:
                logger.warning(
                    "Ignoring invalid prepared dataset S3 path '%s' because it does not include a bucket.",
                    normalized_override,
                )
                return None
            root_prefix = parsed.path.strip("/")
        else:
            bucket = runtime_config.s3_bucket
            root_prefix = normalized_override.strip("/")
    else:
        bucket = runtime_config.s3_bucket
        root_prefix = "/".join(
            part
            for part in [str(runtime_config.s3_prefix).strip("/"), PREPARED_BUNDLE_REMOTE_ROOT]
            if part
        )

    if not bucket:
        return None

    prefix_parts = [root_prefix, fingerprint]
    remote_prefix = "/".join(part for part in prefix_parts if part)
    return str(bucket), remote_prefix


def _portable_bundle_path(path_value: Optional[str], bundle_root: Path) -> Optional[str]:
    if path_value is None:
        return None

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        return path.as_posix()

    try:
        relative_path = path.resolve(strict=False).relative_to(bundle_root.resolve(strict=False))
    except ValueError:
        return str(path)
    return relative_path.as_posix()


def _local_bundle_path(path_value: Optional[str], bundle_root: Path) -> Optional[str]:
    if path_value is None:
        return None

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((bundle_root.resolve(strict=False) / path).resolve(strict=False))


def _portable_manifest_payload(manifest_path: Path, bundle_root: Path) -> Dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    portable_payload = dict(payload)
    portable_payload["scaler_path"] = _portable_bundle_path(
        payload.get("scaler_path"),
        bundle_root,
    )
    portable_payload["feature_names_path"] = _portable_bundle_path(
        payload.get("feature_names_path"),
        bundle_root,
    )

    portable_modes: Dict[str, Any] = {}
    for mode_name, mode_payload in dict(payload["modes"]).items():
        mode_dict = dict(mode_payload)
        mode_dict["feature_values_path"] = _portable_bundle_path(
            mode_dict["feature_values_path"],
            bundle_root,
        )
        mode_dict["label_values_path"] = _portable_bundle_path(
            mode_dict["label_values_path"],
            bundle_root,
        )
        portable_modes[str(mode_name)] = mode_dict
    portable_payload["modes"] = portable_modes
    return portable_payload


def _localized_manifest_payload(payload: Mapping[str, Any], bundle_root: Path) -> Dict[str, Any]:
    localized_payload = dict(payload)
    localized_payload["scaler_path"] = _local_bundle_path(
        payload.get("scaler_path"),
        bundle_root,
    )
    localized_payload["feature_names_path"] = _local_bundle_path(
        payload.get("feature_names_path"),
        bundle_root,
    )

    localized_modes: Dict[str, Any] = {}
    for mode_name, mode_payload in dict(payload["modes"]).items():
        mode_dict = dict(mode_payload)
        mode_dict["feature_values_path"] = _local_bundle_path(
            mode_dict["feature_values_path"],
            bundle_root,
        )
        mode_dict["label_values_path"] = _local_bundle_path(
            mode_dict["label_values_path"],
            bundle_root,
        )
        localized_modes[str(mode_name)] = mode_dict
    localized_payload["modes"] = localized_modes
    return localized_payload


def _write_localized_manifest_file(manifest_path: Path, *, bundle_root: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    localized_payload = _localized_manifest_payload(payload, bundle_root)
    manifest_path.write_text(
        json.dumps(localized_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _s3_object_exists(
    *,
    s3_client: Any,
    bucket: str,
    key: str,
) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchBucket", "NoSuchKey", "NotFound"}:
            return False
        raise


def _acquire_prepared_bundle_build_lock(
    *,
    runtime_config: RuntimeConfig,
    fingerprint: str,
) -> bool:
    remote_location = _prepared_bundle_remote_location(
        runtime_config,
        fingerprint=fingerprint,
    )
    if remote_location is None:
        return False

    bucket, remote_prefix = remote_location
    manifest_key = f"{remote_prefix}/{PREPARED_BUNDLE_MANIFEST_NAME}"
    lock_key = f"{remote_prefix}/{PREPARED_BUNDLE_BUILD_LOCK_NAME}"

    try:
        ensure_bucket_exists(bucket)
        s3_client = create_s3_client()
        s3_client.put_object(
            Bucket=bucket,
            Key=lock_key,
            Body=f"pid={Path.cwd()} time={time.time():.6f}\n".encode("utf-8"),
            ContentType="text/plain",
            IfNoneMatch="*",
        )
        logger.info(
            "Acquired remote prepared bundle build lock at s3://%s/%s.",
            bucket,
            lock_key,
        )
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code not in {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}:
            logger.warning(
                "Failed to acquire prepared bundle build lock at s3://%s/%s: %s",
                bucket,
                lock_key,
                exc,
            )
            return False

        deadline = time.monotonic() + PREPARED_BUNDLE_REMOTE_WAIT_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if _s3_object_exists(
                    s3_client=s3_client,
                    bucket=bucket,
                    key=manifest_key,
                ):
                    logger.info(
                        "Prepared bundle for fingerprint '%s' became available at s3://%s/%s while waiting.",
                        fingerprint,
                        bucket,
                        manifest_key,
                    )
                    return False
            except Exception as wait_exc:  # noqa: BLE001
                logger.warning(
                    "Failed while waiting on prepared bundle manifest at s3://%s/%s: %s",
                    bucket,
                    manifest_key,
                    wait_exc,
                )
                return False
            time.sleep(PREPARED_BUNDLE_REMOTE_WAIT_POLL_S)

        logger.warning(
            "Timed out waiting for prepared bundle manifest at s3://%s/%s; falling back to local build.",
            bucket,
            manifest_key,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to acquire prepared bundle build lock at s3://%s/%s: %s",
            bucket,
            lock_key,
            exc,
        )
        return False


def _release_prepared_bundle_build_lock(
    *,
    runtime_config: RuntimeConfig,
    fingerprint: str,
) -> None:
    remote_location = _prepared_bundle_remote_location(
        runtime_config,
        fingerprint=fingerprint,
    )
    if remote_location is None:
        return

    bucket, remote_prefix = remote_location
    lock_key = f"{remote_prefix}/{PREPARED_BUNDLE_BUILD_LOCK_NAME}"
    try:
        create_s3_client().delete_object(Bucket=bucket, Key=lock_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to release prepared bundle build lock at s3://%s/%s: %s",
            bucket,
            lock_key,
            exc,
        )


def _download_prepared_bundle_from_s3(
    *,
    bundle_root: Path,
    runtime_config: RuntimeConfig,
    fingerprint: str,
) -> bool:
    remote_location = _prepared_bundle_remote_location(
        runtime_config,
        fingerprint=fingerprint,
    )
    if remote_location is None:
        return False

    bucket, remote_prefix = remote_location
    manifest_key = f"{remote_prefix}/{PREPARED_BUNDLE_MANIFEST_NAME}"
    download_prefix = f"{remote_prefix}/"
    s3_client = create_s3_client()

    try:
        if not _s3_object_exists(
            s3_client=s3_client,
            bucket=bucket,
            key=manifest_key,
        ):
            return False
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to probe prepared bundle at s3://%s/%s: %s",
            bucket,
            manifest_key,
            exc,
        )
        return False

    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    temp_bundle_root = Path(
        tempfile.mkdtemp(
            prefix=f".{fingerprint}-download-",
            dir=str(bundle_root.parent),
        )
    )
    try:
        downloaded_any = False
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=download_prefix):
            for obj in page.get("Contents", []):
                key = str(obj["Key"])
                relative_key = key[len(download_prefix):]
                if not relative_key:
                    continue
                local_path = temp_bundle_root / relative_key
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(bucket, key, str(local_path))
                downloaded_any = True

        manifest_path = temp_bundle_root / PREPARED_BUNDLE_MANIFEST_NAME
        if not downloaded_any or not manifest_path.exists():
            shutil.rmtree(temp_bundle_root, ignore_errors=True)
            return False

        _write_localized_manifest_file(manifest_path, bundle_root=bundle_root)

        if bundle_root.exists():
            manifest_path = bundle_root / PREPARED_BUNDLE_MANIFEST_NAME
            if manifest_path.exists():
                shutil.rmtree(temp_bundle_root, ignore_errors=True)
                return True
            shutil.rmtree(bundle_root, ignore_errors=True)

        temp_bundle_root.rename(bundle_root)
        logger.info(
            "Downloaded prepared bundle for fingerprint '%s' from s3://%s/%s.",
            fingerprint,
            bucket,
            remote_prefix,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(temp_bundle_root, ignore_errors=True)
        logger.warning(
            "Failed to download prepared bundle for fingerprint '%s' from s3://%s/%s: %s",
            fingerprint,
            bucket,
            remote_prefix,
            exc,
        )
        return False


def _upload_prepared_bundle_to_s3(
    *,
    bundle_root: Path,
    runtime_config: RuntimeConfig,
    fingerprint: str,
) -> None:
    remote_location = _prepared_bundle_remote_location(
        runtime_config,
        fingerprint=fingerprint,
    )
    if remote_location is None:
        return

    manifest_path = bundle_root / PREPARED_BUNDLE_MANIFEST_NAME
    if not manifest_path.exists():
        logger.warning(
            "Skipping prepared bundle upload for fingerprint '%s' because %s is missing.",
            fingerprint,
            manifest_path,
        )
        return

    bucket, remote_prefix = remote_location
    remote_dir_prefix = f"{remote_prefix}/"

    try:
        ensure_bucket_exists(bucket)
        s3_client = create_s3_client()
        for local_path in sorted(bundle_root.rglob("*")):
            if not local_path.is_file():
                continue
            if local_path.name == PREPARED_BUNDLE_MANIFEST_NAME or local_path.suffix == ".lock":
                continue
            s3_client.upload_file(
                str(local_path),
                bucket,
                f"{remote_dir_prefix}{local_path.relative_to(bundle_root).as_posix()}",
            )

        portable_manifest = _portable_manifest_payload(manifest_path, bundle_root)
        manifest_body = json.dumps(portable_manifest, indent=2, sort_keys=True, default=str) + "\n"
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{remote_dir_prefix}{PREPARED_BUNDLE_MANIFEST_NAME}",
            Body=manifest_body.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "Uploaded prepared bundle for fingerprint '%s' to s3://%s/%s.",
            fingerprint,
            bucket,
            remote_prefix,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to upload prepared bundle for fingerprint '%s' to s3://%s/%s: %s",
            fingerprint,
            bucket,
            remote_prefix,
            exc,
        )


def default_prepared_root(runtime_config: RuntimeConfig) -> Path:
    shared_root = runtime_config.mapped_storage
    if shared_root:
        return Path(shared_root) / "prepared"
    return Path(tempfile.gettempdir()) / "noboom_prepared"


def resolve_model_preparation_settings(
    *,
    config_dir: str,
    model_name: str,
    hp_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve dataset preparation settings from the merged model config."""

    data_cfg = _load_model_data_config(config_dir, model_name)
    raw_splits = data_cfg.get("splits", [0.7, 0.3])
    if not isinstance(raw_splits, Sequence) or len(raw_splits) != 2:
        raise ValueError(f"Model '{model_name}' has invalid data.splits: {raw_splits}")
    return {
        "splits": (float(raw_splits[0]), float(raw_splits[1])),
        "version": str(data_cfg.get("version", "1.0")),
        "gen_to_org_factor": float(data_cfg.get("gen_to_org_factor", -1.0)),
        "test_on_org": bool(data_cfg.get("test_on_org", False)),
        "scaler_spec": _merge_scaler_init_args_from_hparams(
            _spec_to_dict(data_cfg.get("scaler")),
            hp_params,
        ),
    }


def ensure_prepared_dataset_manifest(
    *,
    dataset_name: str,
    requested_dataset_name: Optional[str],
    runtime_config: RuntimeConfig,
    splits: Sequence[float],
    version: str,
    stage_data_source: str,
    gen_to_org_factor: float,
    test_on_org: bool,
    scaler_spec: Optional[Mapping[str, Any]],
    prepared_root: Optional[str] = None,
    verbosity: int = 0,
) -> str:
    """Prepare a dataset bundle if needed and return its manifest path."""

    effective_requested_name = requested_dataset_name or dataset_name
    normalized_scaler_spec = _normalize_scaler_spec(scaler_spec)
    recipe = build_preparation_recipe(
        dataset_name=dataset_name,
        requested_dataset_name=effective_requested_name,
        runtime_config=runtime_config,
        splits=(float(splits[0]), float(splits[1])),
        version=version,
        stage_data_source=stage_data_source,
        gen_to_org_factor=float(gen_to_org_factor),
        test_on_org=bool(test_on_org),
        scaler_spec=normalized_scaler_spec,
    )
    root = Path(prepared_root) if prepared_root is not None else default_prepared_root(runtime_config)
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = fingerprint_recipe(recipe)
    manifest_path = root / fingerprint / "manifest.json"
    lock_root = root / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{fingerprint}.lock"

    logger.info(
        "Resolving prepared manifest for dataset '%s' (requested='%s', stage_source=%s, fingerprint=%s).",
        dataset_name,
        effective_requested_name,
        stage_data_source,
        fingerprint,
    )
    with FileLock(str(lock_path)):
        if not manifest_path.exists():
            bundle_root = root / fingerprint
            remote_build_lock_acquired = False
            try:
                restored_from_remote = _download_prepared_bundle_from_s3(
                    bundle_root=bundle_root,
                    runtime_config=runtime_config,
                    fingerprint=fingerprint,
                )
                if not restored_from_remote:
                    remote_build_lock_acquired = _acquire_prepared_bundle_build_lock(
                        runtime_config=runtime_config,
                        fingerprint=fingerprint,
                    )
                    if not remote_build_lock_acquired:
                        restored_from_remote = _download_prepared_bundle_from_s3(
                            bundle_root=bundle_root,
                            runtime_config=runtime_config,
                            fingerprint=fingerprint,
                        )

                if restored_from_remote:
                    logger.info(
                        "Prepared manifest for fingerprint '%s' restored to %s from remote storage.",
                        fingerprint,
                        manifest_path,
                    )
                else:
                    logger.info(
                        "Prepared manifest missing at %s; starting preparation.",
                        manifest_path,
                    )
                    run_preparation_pipeline(recipe=recipe, prepared_root=root)
                    _upload_prepared_bundle_to_s3(
                        bundle_root=bundle_root,
                        runtime_config=runtime_config,
                        fingerprint=fingerprint,
                    )
            finally:
                if remote_build_lock_acquired:
                    _release_prepared_bundle_build_lock(
                        runtime_config=runtime_config,
                        fingerprint=fingerprint,
                    )
        elif verbosity > 0:
            logger.info("Prepared manifest already exists at %s.", manifest_path)
    return str(manifest_path)


def prepare_model_dataset_manifests(
    *,
    dataset_name: str,
    requested_dataset_name: Optional[str],
    model_name: str,
    config_dir: str,
    runtime_config: RuntimeConfig,
    use_generated: bool,
    hp_params: Optional[Mapping[str, Any]] = None,
    prepared_root: Optional[str] = None,
    verbosity: int = 0,
) -> Dict[str, str]:
    """Prepare the manifests needed for a model run before training starts."""

    settings = resolve_model_preparation_settings(
        config_dir=config_dir,
        model_name=model_name,
        hp_params=hp_params,
    )
    manifest_paths = {
        "real": ensure_prepared_dataset_manifest(
            dataset_name=dataset_name,
            requested_dataset_name=requested_dataset_name,
            runtime_config=runtime_config,
            splits=settings["splits"],
            version=settings["version"],
            stage_data_source="real",
            gen_to_org_factor=settings["gen_to_org_factor"],
            test_on_org=settings["test_on_org"],
            scaler_spec=settings["scaler_spec"],
            prepared_root=prepared_root,
            verbosity=verbosity,
        )
    }
    if use_generated:
        manifest_paths["synthetic"] = ensure_prepared_dataset_manifest(
            dataset_name=dataset_name,
            requested_dataset_name=requested_dataset_name,
            runtime_config=runtime_config,
            splits=settings["splits"],
            version=settings["version"],
            stage_data_source="synthetic",
            gen_to_org_factor=settings["gen_to_org_factor"],
            test_on_org=settings["test_on_org"],
            scaler_spec=settings["scaler_spec"],
            prepared_root=prepared_root,
            verbosity=verbosity,
        )
    return manifest_paths


def manifest_paths_from_args(args: Any) -> Dict[str, str]:
    """Normalize manifest paths stored on a run Namespace."""

    prepared_manifest_paths = getattr(args, "prepared_manifest_paths", None)
    if isinstance(prepared_manifest_paths, Mapping):
        normalized = {
            str(key): str(value)
            for key, value in prepared_manifest_paths.items()
            if value is not None
        }
        if normalized:
            return normalized

    data_manifest_path = getattr(args, "data_manifest_path", None)
    if data_manifest_path is not None:
        return {"real": str(data_manifest_path)}

    raise ValueError(
        "No prepared dataset manifest paths are configured. "
        "Prepare manifests before launching training or evaluation."
    )


def _rebuild_missing_stage_manifest(args: Any, stage_data_source: str) -> str:
    runtime_config = getattr(args, "runtime_config", None)
    config_dir = getattr(args, "config_dir", None)
    model_name = getattr(args, "model_name", None)
    dataset_name = getattr(args, "dataset_name", None)
    if runtime_config is None or config_dir is None or model_name is None or dataset_name is None:
        raise FileNotFoundError(
            "Prepared dataset manifest is missing on this worker, and the run does not carry enough "
            "context to rebuild it locally."
        )

    settings = resolve_model_preparation_settings(
        config_dir=str(config_dir),
        model_name=str(model_name),
        hp_params=getattr(args, "preparation_hp_params", None),
    )
    manifest_path = ensure_prepared_dataset_manifest(
        dataset_name=str(dataset_name),
        requested_dataset_name=getattr(args, "requested_dataset_name", dataset_name),
        runtime_config=runtime_config,
        splits=settings["splits"],
        version=settings["version"],
        stage_data_source=stage_data_source,
        gen_to_org_factor=settings["gen_to_org_factor"],
        test_on_org=settings["test_on_org"],
        scaler_spec=settings["scaler_spec"],
        verbosity=int(getattr(args, "verbose", 0) or 0),
    )
    prepared_manifest_paths = getattr(args, "prepared_manifest_paths", None)
    if isinstance(prepared_manifest_paths, Mapping):
        updated_manifest_paths = {
            str(key): str(value)
            for key, value in prepared_manifest_paths.items()
            if value is not None
        }
        updated_manifest_paths[stage_data_source] = str(manifest_path)
        args.prepared_manifest_paths = updated_manifest_paths
    if stage_data_source == "real":
        args.data_manifest_path = str(manifest_path)
    return str(manifest_path)


def manifest_path_for_stage_from_args(args: Any, stage_data_source: str) -> str:
    """Resolve the prepared manifest path for one stage."""

    manifest_paths = manifest_paths_from_args(args)
    if stage_data_source not in manifest_paths:
        raise ValueError(
            f"No prepared dataset manifest was configured for stage '{stage_data_source}'. "
            f"Available stages: {sorted(manifest_paths)}."
        )
    manifest_path = str(manifest_paths[stage_data_source])
    if Path(manifest_path).exists():
        return manifest_path
    return _rebuild_missing_stage_manifest(args, stage_data_source)


def load_prepared_manifest(path: str) -> PreparedDatasetManifest:
    return PreparedDatasetManifest.load(path)
