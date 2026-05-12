"""Dataset preparation helpers for Arrow IPC dataset bundles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import pickle
import shutil
import sys
import tempfile
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.request import urlopen
import zipfile

from filelock import FileLock
import kagglehub
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from sklearn.base import TransformerMixin
from tqdm import tqdm

from noboom.tsad.data import TorchDataset
from timesead.data.transforms import DatasetSource, PipelineDataset, make_dataset_split

from ..config.runtime import RuntimeConfig
from ..webdav_utils import download_from_webdav
from .manifest import PreparedDatasetManifest, PreparedModeManifest


logger = logging.getLogger(__name__)

KAGGLE_DATASET_SLUG = "faebs94/noboom-anomaly-detection-in-chemical-processes/versions/1"
TSST_REMOTE_FOLDER = "NOBOOM_TSST"
OME_BASE_DATASET_NAME = "cont_reactive_ome"
OME_EXTENDED_DATASET_NAME = "cont_reactive_ome_ext"
OME_REDUCED_DATASET_NAME = "cont_reactive_ome_red"
OME_EXTENSION_DOWNLOAD_URL = os.environ.get(
    "NOBOOM_OME_EXTENSION_URL",
    "https://anonymous.example.org/noboom/ome-extension/",
)
TSST_WEBDAV_HOSTNAME = os.environ.get("NOBOOM_TSST_WEBDAV_URL", "https://anonymous.example.org/seafdav/")
OME_EXTENSION_SUPPORT_DIRNAME = "_cont_reactive_ome_extension"
OME_EXTENSION_TRAIN_MODE = "train_ext"
PREPARED_MANIFEST_VERSION = 2
PREPARED_EXPORTER_VERSION = "arrow-ipc-v2"
ARROW_EXPORT_TARGET_BATCH_BYTES = 64 * 1024 * 1024
PREPARATION_MAX_WORKERS_ENV = "NOBOOM_PREP_MAX_WORKERS"
DEFAULT_PREPARATION_MAX_WORKERS = 4
PREPARATION_PROGRESS_LOG_INTERVAL_S = 10.0


@dataclass(frozen=True)
class CanonicalSequence:
    """Canonical sequence row exported to Arrow IPC bundles."""

    sequence_id: str
    features: np.ndarray
    labels: np.ndarray
    requested_dataset_name: str
    base_dataset_name: str
    mode: str
    source_name: str


@dataclass(frozen=True)
class DatasetVariant:
    """Variant policy resolved from the requested dataset name."""

    requested_name: str
    base_dataset_name: str
    reduced_feature_names: Optional[Tuple[str, ...]] = None
    base_feature_indices: Optional[Tuple[int, ...]] = None
    extension_dir: Optional[Path] = None
    extension_mode: Optional[str] = None
    include_extension_train: bool = False


@dataclass(frozen=True)
class PreparationRecipe:
    """Preparation recipe captured in the manifest fingerprint."""

    dataset_name: str
    requested_dataset_name: str
    base_dataset_name: str
    version: str
    stage_data_source: str
    splits: Tuple[float, float]
    gen_to_org_factor: float
    test_on_org: bool
    scaler_spec: Optional[Dict[str, Any]]
    raw_data_root: str
    source_uri: Optional[str]
    raw_provenance: List[Dict[str, Any]]

    def to_fingerprint_payload(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "requested_dataset_name": self.requested_dataset_name,
            "base_dataset_name": self.base_dataset_name,
            "version": self.version,
            "stage_data_source": self.stage_data_source,
            "splits": list(self.splits),
            "gen_to_org_factor": self.gen_to_org_factor,
            "test_on_org": self.test_on_org,
            "scaler_spec": self.scaler_spec,
            "raw_provenance": self.raw_provenance,
            "exporter_version": PREPARED_EXPORTER_VERSION,
        }


def instantiate_object(spec: Optional[Mapping[str, Any]]) -> Optional[Any]:
    """Instantiate an object from ``class_path`` / ``init_args`` config."""

    if spec is None:
        return None
    class_path = spec.get("class_path")
    if class_path is None:
        raise ValueError(f"Invalid object spec without class_path: {spec}")
    module_name, _, attr_name = str(class_path).rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid class path: {class_path}")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr_name)
    init_args = dict(spec.get("init_args", {}))
    return cls(**init_args)


def _count_sequence_steps(rows: Sequence[CanonicalSequence]) -> int:
    return sum(int(row.features.shape[0]) for row in rows)


def _format_mode_summary(mode_rows: Mapping[str, Any]) -> str:
    if not mode_rows:
        return "no modes"
    parts = []
    for mode_name, rows in mode_rows.items():
        if isinstance(rows, PreparedModeManifest):
            parts.append(f"{mode_name}={rows.item_count} seq/{sum(rows.seq_len)} steps")
        else:
            parts.append(f"{mode_name}={len(rows)} seq/{_count_sequence_steps(rows)} steps")
    return ", ".join(parts)


def _preparation_tqdm(*, total: int, desc: str, unit: str) -> tqdm:
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=PREPARATION_PROGRESS_LOG_INTERVAL_S,
        file=sys.stdout,
    )


def get_dataset_cache_root() -> Path:
    """Resolve the shared dataset cache root."""

    cache_env = os.getenv("KAGGLEHUB_CACHE")
    if cache_env:
        return Path(cache_env)
    return Path.home() / ".cache" / "kagglehub"


def get_ome_extension_support_dir(data_dir: Path) -> Path:
    return data_dir / OME_EXTENSION_SUPPORT_DIRNAME


def resolve_base_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.replace("_tsst", "")
    if normalized in {OME_EXTENDED_DATASET_NAME, OME_REDUCED_DATASET_NAME}:
        return OME_BASE_DATASET_NAME
    return normalized


def resolve_download_dataset_name(dataset_name: str) -> str:
    return resolve_base_dataset_name(dataset_name)


def get_noboom_meta_columns(dataset_name: str) -> Tuple[str, ...]:
    if dataset_name == "industry_process":
        return ("Anomaly",)
    return (
        "Time",
        "Label (common/hard fault)",
        "Label (common/soft fault)",
        "Label (common/controller fault)",
        "Label (common/hard and soft)",
        "Label (common/all)",
        "Label (advanced/hard fault)",
        "Label (advanced/soft fault)",
        "Label (advanced/controller fault)",
    )


def preprocess_noboom_dataset(data_dir: Path, dataset_name: str) -> None:
    """Convert source CSV or NPY files into Parquet once."""

    def has_ancestor_named(path: Path) -> bool:
        return any(parent.name == dataset_name for parent in path.parents)

    csv_files = sorted(path for path in data_dir.rglob("*.csv") if has_ancestor_named(path))
    if csv_files:
        input_files = csv_files
        loader = pd.read_csv
    else:
        input_files = sorted(path for path in data_dir.rglob("*.npy") if has_ancestor_named(path))

        def loader(path: Path) -> pd.DataFrame:
            return pd.DataFrame(np.load(path))

    pending_conversions = [file_path for file_path in input_files if not file_path.with_suffix(".parquet").exists()]
    if input_files:
        logger.info(
            "Checking %d raw files for parquet preprocessing for dataset '%s' (%d conversions needed).",
            len(input_files),
            dataset_name,
            len(pending_conversions),
        )
    for file_path in input_files:
        parquet_path = file_path.with_suffix(".parquet")
        if parquet_path.exists():
            continue
        frame = loader(file_path)
        frame.to_parquet(parquet_path)


def _is_allowed_ome_extension_member(member_name: str) -> bool:
    member_path = PurePosixPath(member_name)
    if "__MACOSX" in member_path.parts:
        return False
    if member_path.name.startswith("._") or member_path.name == ".DS_Store":
        return False
    return member_path.name.endswith("_styled_features.csv")


def _extract_ome_extension_archive(archive_path: Path, destination_dir: Path) -> List[Path]:
    extracted_files: List[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not _is_allowed_ome_extension_member(member.filename):
                continue
            member_path = PurePosixPath(member.filename)
            target_path = destination_dir.joinpath(*member_path.parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_files.append(target_path)
    if not extracted_files:
        raise RuntimeError(f"No styled OME extension CSV files were extracted from {archive_path}.")
    return extracted_files


def list_ome_extension_csv_files(extension_dir: Path) -> List[Path]:
    csv_files: List[Path] = []
    for path in sorted(extension_dir.rglob("*")):
        if not path.is_file():
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._") or path.name == ".DS_Store":
            continue
        if path.name.endswith("_styled_features.csv"):
            csv_files.append(path)
    if not csv_files:
        raise RuntimeError(f"No styled feature CSV files found in {extension_dir}.")
    return csv_files


def prepare_ome_extension_support(data_dir: Path) -> Path:
    support_dir = get_ome_extension_support_dir(data_dir)
    if support_dir.exists():
        try:
            list_ome_extension_csv_files(support_dir)
            return support_dir
        except RuntimeError:
            shutil.rmtree(support_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        archive_path = temp_dir / "ome_generated.zip"
        extracted_root = temp_dir / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)
        with urlopen(OME_EXTENSION_DOWNLOAD_URL) as response, archive_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
        _extract_ome_extension_archive(archive_path, extracted_root)
        list_ome_extension_csv_files(extracted_root)
        shutil.copytree(extracted_root, support_dir, dirs_exist_ok=True)
    return support_dir


def read_ome_extension_feature_names(extension_dir: Path) -> Tuple[str, ...]:
    header_variants: Dict[Tuple[str, ...], List[Path]] = {}
    for csv_path in list_ome_extension_csv_files(extension_dir):
        header = tuple(pd.read_csv(csv_path, nrows=0).columns.tolist())
        header_variants.setdefault(header, []).append(csv_path)
    if len(header_variants) != 1:
        details = {
            tuple(columns): [str(path) for path in paths]
            for columns, paths in header_variants.items()
        }
        raise RuntimeError(f"OME extension CSV schema mismatch: {details}")
    feature_names = next(iter(header_variants))
    if not feature_names:
        raise RuntimeError("OME extension CSV files have no feature columns.")
    return feature_names


def read_base_dataset_feature_names(data_root: Path, dataset_name: str) -> Tuple[str, ...]:
    dataset_dir = data_root / dataset_name
    csv_files = sorted(dataset_dir.rglob("train_normal*.csv"))
    if not csv_files:
        csv_files = sorted(dataset_dir.rglob("train*.csv"))
    if csv_files:
        columns = pd.read_csv(csv_files[0], nrows=0).columns.tolist()
    else:
        parquet_files = sorted(dataset_dir.rglob("train_normal*.parquet"))
        if not parquet_files:
            parquet_files = sorted(dataset_dir.rglob("train*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"Could not locate a training file to infer features for dataset '{dataset_name}' in {dataset_dir}."
            )
        columns = pd.read_parquet(parquet_files[0]).columns.tolist()

    meta_columns = set(get_noboom_meta_columns(dataset_name))
    feature_names = tuple(column for column in columns if column not in meta_columns)
    if not feature_names:
        raise RuntimeError(f"Dataset '{dataset_name}' has no non-metadata feature columns.")
    return feature_names


def _download_noboom_dataset(*, force_download: bool) -> Path:
    action = "Refreshing" if force_download else "Fetching or reusing"
    logger.info("%s Kaggle dataset '%s'.", action, KAGGLE_DATASET_SLUG)
    return Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG, force_download=force_download))


def _validate_requested_dataset_assets(data_root: Path, requested_dataset_name: str) -> None:
    requested_base_dataset = resolve_download_dataset_name(requested_dataset_name)
    read_base_dataset_feature_names(data_root, requested_base_dataset)

    if requested_dataset_name.replace("_tsst", "") in {OME_EXTENDED_DATASET_NAME, OME_REDUCED_DATASET_NAME}:
        logger.info("Preparing OME extension support files for '%s'.", requested_dataset_name)
        prepare_ome_extension_support(data_root)
        resolve_dataset_variant(requested_dataset_name, data_root)


def resolve_dataset_variant(dataset_name: str, data_root: Path) -> DatasetVariant:
    base_dataset_name = resolve_base_dataset_name(dataset_name)
    if dataset_name.replace("_tsst", "") not in {OME_EXTENDED_DATASET_NAME, OME_REDUCED_DATASET_NAME}:
        return DatasetVariant(
            requested_name=dataset_name,
            base_dataset_name=base_dataset_name,
        )

    base_feature_names = read_base_dataset_feature_names(data_root, base_dataset_name)
    extension_dir = prepare_ome_extension_support(data_root)
    extension_feature_names = read_ome_extension_feature_names(extension_dir)
    base_feature_index = {name: idx for idx, name in enumerate(base_feature_names)}
    reduced_feature_names = tuple(
        feature_name
        for feature_name in extension_feature_names
        if feature_name in base_feature_index
    )
    if not reduced_feature_names:
        raise RuntimeError(
            f"No overlapping features found between {base_dataset_name} and the OME extension dataset."
        )

    normalized_name = dataset_name.replace("_tsst", "")
    return DatasetVariant(
        requested_name=dataset_name,
        base_dataset_name=base_dataset_name,
        reduced_feature_names=reduced_feature_names,
        base_feature_indices=tuple(base_feature_index[name] for name in reduced_feature_names),
        extension_dir=extension_dir,
        extension_mode=OME_EXTENSION_TRAIN_MODE,
        include_extension_train=normalized_name == OME_EXTENDED_DATASET_NAME,
    )


def ensure_dataset_assets(
    requested_dataset_name: str,
    *,
    runtime_config: RuntimeConfig,
    include_generated: bool,
) -> Tuple[Path, Optional[str]]:
    """Ensure raw sources are available and return the NoBOOM data root."""

    logger.info(
        "Resolving raw assets for requested dataset '%s' (stage_source=%s).",
        requested_dataset_name,
        "synthetic" if include_generated else "real",
    )
    cache_root = get_dataset_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / "dataset_download.lock"
    requested_base_dataset = resolve_download_dataset_name(requested_dataset_name)

    source_uri = f"kaggle://{KAGGLE_DATASET_SLUG}"
    with FileLock(str(lock_path)):
        last_missing_training_error: Optional[FileNotFoundError] = None
        noboom_root: Optional[Path] = None
        for force_download in (False, True):
            if force_download:
                logger.warning(
                    "Dataset '%s' is missing expected training files in the cached Kaggle copy. "
                    "Retrying once with a forced re-download.",
                    requested_base_dataset,
                )

            dataset_root = _download_noboom_dataset(force_download=force_download)
            noboom_root = dataset_root / "noboom"
            preprocess_noboom_dataset(dataset_root, requested_base_dataset)
            try:
                _validate_requested_dataset_assets(noboom_root, requested_dataset_name)
            except FileNotFoundError as exc:
                last_missing_training_error = exc
                if not force_download:
                    continue
                raise
            else:
                break

        if noboom_root is None:
            if last_missing_training_error is not None:
                raise last_missing_training_error
            raise RuntimeError(f"Failed to resolve raw assets for dataset '{requested_dataset_name}'.")

        if include_generated:
            logger.info("Fetching or reusing generated TSST dataset for '%s'.", requested_base_dataset)
            generated_root = Path(
                download_from_webdav(
                    webdav_hostname=TSST_WEBDAV_HOSTNAME,
                    webdav_login=runtime_config.seafile_username or "",
                    webdav_password=runtime_config.seafile_password or "",
                    remote_folder=TSST_REMOTE_FOLDER,
                    out_dir=cache_root,
                )
            )
            generated_dataset_name = f"{requested_base_dataset}_tsst"
            generated_source_dir = generated_root / "noboom" / generated_dataset_name
            generated_link = noboom_root / generated_dataset_name
            if generated_link.exists() or generated_link.is_symlink():
                generated_link.unlink(missing_ok=True)
            generated_link.symlink_to(generated_source_dir, target_is_directory=True)
            source_uri = f"{source_uri}+webdav://{TSST_REMOTE_FOLDER}/{generated_dataset_name}"

    logger.info(
        "Raw assets ready for '%s' at %s.",
        requested_dataset_name,
        noboom_root,
    )
    return noboom_root, source_uri


def collect_raw_provenance(data_root: Path, variant: DatasetVariant, include_generated: bool) -> List[Dict[str, Any]]:
    """Collect a lightweight provenance signature for relevant raw files."""

    roots = [data_root / variant.base_dataset_name]
    if include_generated:
        roots.append(data_root / f"{variant.base_dataset_name}_tsst")
    if variant.extension_dir is not None:
        roots.append(variant.extension_dir)

    provenance: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in {".csv", ".parquet", ".npy"}:
                continue
            stat_result = path.stat()
            provenance.append(
                {
                    "path": str(path.relative_to(data_root)),
                    "size": int(stat_result.st_size),
                    "mtime_ns": int(stat_result.st_mtime_ns),
                }
            )
    return provenance


def _write_arrow_column(
    output_path: Path,
    *,
    field_name: str,
    values: np.ndarray,
    data_type: pa.DataType,
) -> None:
    """Write one primitive Arrow IPC column to disk."""

    schema = pa.schema([pa.field(field_name, data_type, nullable=False)])
    record_batch = pa.record_batch(
        [pa.array(values, type=data_type)],
        schema=schema,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ipc.RecordBatchFileWriter(output_path, schema) as writer:
        writer.write_batch(record_batch)


def _row_array(row: Any, key: str, dtype: np.dtype) -> np.ndarray:
    if isinstance(row, Mapping):
        value = row[key]
    else:
        value = getattr(row, key)
    return np.asarray(value, dtype=dtype)


def _iter_arrow_row_batches(
    rows: Sequence[Any],
    *,
    target_batch_bytes: int,
) -> Iterable[Sequence[Any]]:
    if not rows:
        yield []
        return

    batch_start = 0
    batch_bytes = 0
    for index, row in enumerate(rows):
        row_bytes = _row_array(row, "features", np.float32).nbytes + _row_array(row, "labels", np.int64).nbytes
        if index > batch_start and batch_bytes + row_bytes > target_batch_bytes:
            yield rows[batch_start:index]
            batch_start = index
            batch_bytes = 0
        batch_bytes += row_bytes
    yield rows[batch_start:]


def _flatten_arrow_row_batch(
    rows: Sequence[Any],
    *,
    key: str,
    dtype: np.dtype,
) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=dtype)

    arrays = [_row_array(row, key, dtype).reshape(-1) for row in rows]
    total_length = sum(int(array.size) for array in arrays)
    flat = np.empty(total_length, dtype=dtype)
    offset = 0
    for array in arrays:
        next_offset = offset + int(array.size)
        flat[offset:next_offset] = array
        offset = next_offset
    return flat


def _resolve_preparation_max_workers(task_count: int) -> int:
    if task_count <= 1:
        return 1

    configured_workers: Optional[int] = None
    configured_env = os.getenv(PREPARATION_MAX_WORKERS_ENV)
    if configured_env is not None:
        try:
            configured_workers = int(configured_env)
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; expected a positive integer.",
                PREPARATION_MAX_WORKERS_ENV,
                configured_env,
            )
        else:
            if configured_workers < 1:
                logger.warning(
                    "Ignoring invalid %s=%r; expected a positive integer.",
                    PREPARATION_MAX_WORKERS_ENV,
                    configured_env,
                )
                configured_workers = None

    if configured_workers is None:
        configured_workers = min(DEFAULT_PREPARATION_MAX_WORKERS, os.cpu_count() or 1)
    return max(1, min(task_count, configured_workers))


def _clone_fitted_scaler(scaler: TransformerMixin) -> TransformerMixin:
    return pickle.loads(pickle.dumps(scaler, pickle.HIGHEST_PROTOCOL))


def _write_arrow_mode_arrays(output_dir: Path, rows: Sequence[Any]) -> Dict[str, str]:
    """Write one mode as flat primitive Arrow IPC arrays in bounded batches."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_values_path = output_dir / "feature_values.arrow"
    label_values_path = output_dir / "label_values.arrow"
    feature_schema = pa.schema([pa.field("value", pa.float32(), nullable=False)])
    label_schema = pa.schema([pa.field("value", pa.int64(), nullable=False)])
    with ipc.RecordBatchFileWriter(feature_values_path, feature_schema) as feature_writer, ipc.RecordBatchFileWriter(
        label_values_path,
        label_schema,
    ) as label_writer:
        for row_batch in _iter_arrow_row_batches(
            rows,
            target_batch_bytes=ARROW_EXPORT_TARGET_BATCH_BYTES,
        ):
            feature_values = _flatten_arrow_row_batch(
                row_batch,
                key="features",
                dtype=np.float32,
            )
            label_values = _flatten_arrow_row_batch(
                row_batch,
                key="labels",
                dtype=np.int64,
            )
            feature_writer.write_batch(
                pa.record_batch(
                    [pa.array(feature_values, type=pa.float32())],
                    schema=feature_schema,
                )
            )
            label_writer.write_batch(
                pa.record_batch(
                    [pa.array(label_values, type=pa.int64())],
                    schema=label_schema,
                )
            )
    return {
        "feature_values_path": str(feature_values_path),
        "label_values_path": str(label_values_path),
    }


def _transform_feature_batch(
    rows: Sequence[CanonicalSequence],
    *,
    input_feature_names: Sequence[str],
    scaler: TransformerMixin,
) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=np.float32)

    fit_frame = pd.DataFrame(
        np.concatenate([row.features for row in rows], axis=0),
        columns=input_feature_names,
    )
    transformed = scaler.transform(fit_frame)
    return np.asarray(transformed, dtype=np.float32)


def _select_dataset_features_inplace(dataset: TorchDataset, feature_indices: Sequence[int]) -> None:
    indices = np.asarray(feature_indices, dtype=np.int64)
    dataset._samples = [sample[:, indices] for sample in dataset._samples]


def _dataset_source_sequences(
    dataset_source: DatasetSource,
    *,
    requested_dataset_name: str,
    base_dataset_name: str,
    mode: str,
    source_name: str,
) -> List[CanonicalSequence]:
    rows: List[CanonicalSequence] = []
    started_at = perf_counter()
    last_logged_at = started_at
    total_steps = 0
    pipeline_dataset = PipelineDataset(dataset_source)
    logger.info(
        "Loading canonical sequences for mode '%s' from source '%s'.",
        mode,
        source_name,
    )
    with _preparation_tqdm(
        total=len(pipeline_dataset),
        desc=f"Prepare {mode}",
        unit="seq",
    ) as progress:
        for sequence_index, (inputs, targets) in enumerate(pipeline_dataset):
            feature_array = np.asarray(inputs[0].numpy(), dtype=np.float32)
            label_array = np.asarray(targets[0].numpy(), dtype=np.int64)
            total_steps += int(feature_array.shape[0])
            rows.append(
                CanonicalSequence(
                    sequence_id=f"{mode}:{sequence_index}",
                    features=feature_array,
                    labels=label_array,
                    requested_dataset_name=requested_dataset_name,
                    base_dataset_name=base_dataset_name,
                    mode=mode,
                    source_name=source_name,
                )
            )
            progress.update(1)
            now = perf_counter()
            if now - last_logged_at >= PREPARATION_PROGRESS_LOG_INTERVAL_S:
                progress.set_postfix_str(f"steps={total_steps}", refresh=False)
                logger.info(
                    "Loaded mode '%s': %d sequences / %d steps so far.",
                    mode,
                    sequence_index + 1,
                    total_steps,
                )
                last_logged_at = now
        progress.set_postfix_str(f"steps={total_steps}", refresh=False)
    logger.info(
        "Loaded mode '%s': %d sequences / %d steps in %.1fs.",
        mode,
        len(rows),
        total_steps,
        perf_counter() - started_at,
    )
    return rows


def _load_ome_extension_sequences(variant: DatasetVariant) -> List[CanonicalSequence]:
    if variant.extension_dir is None or variant.reduced_feature_names is None:
        raise ValueError("OME extension variant is missing extension_dir or reduced_feature_names.")
    rows: List[CanonicalSequence] = []
    csv_files = list_ome_extension_csv_files(variant.extension_dir)
    started_at = perf_counter()
    last_logged_at = started_at
    total_steps = 0
    logger.info(
        "Loading OME extension sequences from %d CSV files.",
        len(csv_files),
    )
    with _preparation_tqdm(
        total=len(csv_files),
        desc="Prepare train_ext",
        unit="seq",
    ) as progress:
        for sequence_index, csv_path in enumerate(csv_files):
            frame = pd.read_csv(csv_path, usecols=list(variant.reduced_feature_names))
            feature_array = frame.to_numpy(dtype=np.float32, copy=False)
            label_array = np.zeros(feature_array.shape[0], dtype=np.int64)
            total_steps += int(feature_array.shape[0])
            rows.append(
                CanonicalSequence(
                    sequence_id=f"{OME_EXTENSION_TRAIN_MODE}:{sequence_index}",
                    features=feature_array,
                    labels=label_array,
                    requested_dataset_name=variant.requested_name,
                    base_dataset_name=variant.base_dataset_name,
                    mode=OME_EXTENSION_TRAIN_MODE,
                    source_name="extension",
                )
            )
            progress.update(1)
            now = perf_counter()
            if now - last_logged_at >= PREPARATION_PROGRESS_LOG_INTERVAL_S:
                progress.set_postfix_str(f"steps={total_steps}", refresh=False)
                logger.info(
                    "Loaded OME extension: %d/%d sequences / %d steps so far.",
                    sequence_index + 1,
                    len(csv_files),
                    total_steps,
                )
                last_logged_at = now
        progress.set_postfix_str(f"steps={total_steps}", refresh=False)
    logger.info(
        "Loaded OME extension: %d sequences / %d steps in %.1fs.",
        len(rows),
        total_steps,
        perf_counter() - started_at,
    )
    return rows


def load_sequence_modes(recipe: PreparationRecipe) -> Dict[str, List[CanonicalSequence]]:
    """Load the selected stage modes into canonical sequence rows."""

    data_root = Path(recipe.raw_data_root)
    variant = resolve_dataset_variant(recipe.requested_dataset_name, data_root)
    presplit_data = TorchDataset(
        variant.base_dataset_name,
        recipe.version,
        root=str(data_root),
        train=True,
        fast_load=True,
    )
    if variant.base_feature_indices is not None:
        _select_dataset_features_inplace(presplit_data, variant.base_feature_indices)
    train_data, val_data = make_dataset_split(presplit_data, *recipe.splits, axis="time")

    test_data = TorchDataset(
        variant.base_dataset_name,
        recipe.version,
        root=str(data_root),
        train=False,
        fast_load=True,
    )
    if variant.base_feature_indices is not None:
        _select_dataset_features_inplace(test_data, variant.base_feature_indices)
    real_test = DatasetSource(test_data)

    modes: Dict[str, List[CanonicalSequence]] = {}

    # Scaler is always fit on the real train set, even for synthetic stages.
    modes["scaler_train_real"] = _dataset_source_sequences(
        train_data,
        requested_dataset_name=recipe.requested_dataset_name,
        base_dataset_name=variant.base_dataset_name,
        mode="train",
        source_name="real",
    )

    if recipe.stage_data_source == "real":
        modes["train"] = modes["scaler_train_real"]
        modes["val"] = _dataset_source_sequences(
            val_data,
            requested_dataset_name=recipe.requested_dataset_name,
            base_dataset_name=variant.base_dataset_name,
            mode="val",
            source_name="real",
        )
        modes["test"] = _dataset_source_sequences(
            real_test,
            requested_dataset_name=recipe.requested_dataset_name,
            base_dataset_name=variant.base_dataset_name,
            mode="test",
            source_name="real",
        )
        if variant.include_extension_train:
            modes[OME_EXTENSION_TRAIN_MODE] = _load_ome_extension_sequences(variant)
    elif recipe.stage_data_source == "synthetic":
        generated_dataset_name = f"{variant.base_dataset_name}_tsst"
        generated_presplit = TorchDataset(
            generated_dataset_name,
            recipe.version,
            root=str(data_root),
            train=True,
            fast_load=True,
        )
        if recipe.gen_to_org_factor > 0:
            org_seq_len = sum(int(row.features.shape[0]) for row in modes["scaler_train_real"])
            synthetic_lengths = np.asarray([sequence.shape[0] for sequence in generated_presplit._samples], dtype=np.int64)
            max_total_length = int(min(recipe.gen_to_org_factor * org_seq_len, int(synthetic_lengths.sum())))
            cutoff_index = int(np.searchsorted(np.cumsum(synthetic_lengths), max_total_length, side="left")) + 1
            generated_presplit._samples = generated_presplit._samples[:cutoff_index]
            generated_presplit._labels = generated_presplit._labels[:cutoff_index]

        generated_train, generated_val = make_dataset_split(
            generated_presplit,
            *recipe.splits,
            axis="batch",
        )
        generated_test = DatasetSource(
            TorchDataset(
                generated_dataset_name,
                recipe.version,
                root=str(data_root),
                train=False,
                fast_load=True,
            )
        )
        modes["train"] = _dataset_source_sequences(
            generated_train,
            requested_dataset_name=recipe.requested_dataset_name,
            base_dataset_name=variant.base_dataset_name,
            mode="train_tsst",
            source_name="synthetic",
        )
        modes["val"] = _dataset_source_sequences(
            generated_val,
            requested_dataset_name=recipe.requested_dataset_name,
            base_dataset_name=variant.base_dataset_name,
            mode="val_tsst",
            source_name="synthetic",
        )
        test_mode = "test" if recipe.test_on_org else "test_tsst"
        test_source = real_test if recipe.test_on_org else generated_test
        test_source_name = "real" if recipe.test_on_org else "synthetic"
        modes["test"] = _dataset_source_sequences(
            test_source,
            requested_dataset_name=recipe.requested_dataset_name,
            base_dataset_name=variant.base_dataset_name,
            mode=test_mode,
            source_name=test_source_name,
        )
    else:
        raise ValueError(f"Unsupported stage_data_source: {recipe.stage_data_source}")

    return modes


def fit_scaler_for_export(
    recipe: PreparationRecipe,
    mode_rows: Dict[str, List[CanonicalSequence]],
) -> Tuple[Optional[TransformerMixin], List[str], List[str]]:
    """Fit the configured scaler once and return export feature metadata."""

    started_at = perf_counter()
    scaler = instantiate_object(recipe.scaler_spec) if recipe.scaler_spec is not None else None
    variant = resolve_dataset_variant(recipe.requested_dataset_name, Path(recipe.raw_data_root))
    input_feature_names = (
        list(variant.reduced_feature_names)
        if variant.reduced_feature_names is not None
        else list(read_base_dataset_feature_names(Path(recipe.raw_data_root), recipe.base_dataset_name))
    )
    scaler_fit_rows = list(mode_rows.get("scaler_train_real", []))
    if OME_EXTENSION_TRAIN_MODE in mode_rows:
        scaler_fit_rows.extend(mode_rows[OME_EXTENSION_TRAIN_MODE])

    if scaler is None:
        logger.info(
            "No scaler configured for '%s'; exporting %d input features without fitting.",
            recipe.requested_dataset_name,
            len(input_feature_names),
        )
        return None, list(input_feature_names), list(input_feature_names)

    logger.info(
        "Fitting export scaler for '%s' on %d sequences / %d steps with %d input features.",
        recipe.requested_dataset_name,
        len(scaler_fit_rows),
        _count_sequence_steps(scaler_fit_rows),
        len(input_feature_names),
    )
    fit_frame = pd.DataFrame(
        np.concatenate([row.features for row in scaler_fit_rows], axis=0),
        columns=input_feature_names if mode_rows.get("scaler_train_real") else None,
    )
    scaler.fit(fit_frame)
    feature_names = [str(name) for name in getattr(scaler, "feature_names_out_", [])]
    if not feature_names:
        feature_names = [str(name) for name in getattr(scaler, "feature_names_in_", [])]
    logger.info(
        "Finished fitting export scaler for '%s' in %.1fs; output features=%d.",
        recipe.requested_dataset_name,
        perf_counter() - started_at,
        len(feature_names),
    )
    return scaler, feature_names, list(input_feature_names)


def fit_and_transform_sequences(
    recipe: PreparationRecipe,
    mode_rows: Dict[str, List[CanonicalSequence]],
) -> Tuple[Optional[TransformerMixin], List[str], Dict[str, List[CanonicalSequence]]]:
    """Compatibility wrapper that materializes transformed rows in memory."""

    scaler, feature_names, input_feature_names = fit_scaler_for_export(recipe, mode_rows)
    export_mode_names = [mode for mode in mode_rows if mode != "scaler_train_real"]
    if scaler is None:
        return None, feature_names, {mode: list(mode_rows[mode]) for mode in export_mode_names}

    transformed_modes: Dict[str, List[CanonicalSequence]] = {}
    for mode in export_mode_names:
        mode_scaler = _clone_fitted_scaler(scaler)
        if mode == "test" and hasattr(mode_scaler, "extreme_action"):
            mode_scaler.extreme_action = "none"

        transformed_rows: List[CanonicalSequence] = []
        for row_batch in _iter_arrow_row_batches(
            mode_rows[mode],
            target_batch_bytes=ARROW_EXPORT_TARGET_BATCH_BYTES,
        ):
            if not row_batch:
                continue
            transformed_values = _transform_feature_batch(
                row_batch,
                input_feature_names=input_feature_names,
                scaler=mode_scaler,
            )
            offset = 0
            for row in row_batch:
                next_offset = offset + int(row.features.shape[0])
                transformed_rows.append(
                    CanonicalSequence(
                        sequence_id=row.sequence_id,
                        features=np.asarray(transformed_values[offset:next_offset], dtype=np.float32),
                        labels=np.asarray(row.labels, dtype=np.int64),
                        requested_dataset_name=row.requested_dataset_name,
                        base_dataset_name=row.base_dataset_name,
                        mode=row.mode,
                        source_name=row.source_name,
                    )
                )
                offset = next_offset

        transformed_modes[mode] = transformed_rows

    return scaler, feature_names, transformed_modes


def _export_mode_manifest(
    *,
    bundle_root: Path,
    recipe: PreparationRecipe,
    mode_name: str,
    rows: Sequence[CanonicalSequence],
    input_feature_names: Sequence[str],
    scaler: Optional[TransformerMixin],
    mode_rows_pretransformed: bool = False,
) -> Tuple[str, PreparedModeManifest]:
    mode_dir = bundle_root / mode_name
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)

    feature_values_path = mode_dir / "feature_values.arrow"
    label_values_path = mode_dir / "label_values.arrow"
    feature_schema = pa.schema([pa.field("value", pa.float32(), nullable=False)])
    label_schema = pa.schema([pa.field("value", pa.int64(), nullable=False)])
    seq_len: List[int] = []
    item_count = 0
    num_features = 0
    source_name = rows[0].source_name if rows else recipe.stage_data_source
    mode_scaler = _clone_fitted_scaler(scaler) if scaler is not None else None
    if mode_scaler is not None and mode_name == "test" and hasattr(mode_scaler, "extreme_action"):
        mode_scaler.extreme_action = "none"
    started_at = perf_counter()
    last_logged_at = started_at
    total_items = len(rows)
    total_steps = _count_sequence_steps(rows)
    logger.info(
        "Exporting mode '%s' for '%s': %d sequences / %d steps.",
        mode_name,
        recipe.requested_dataset_name,
        total_items,
        total_steps,
    )

    with ipc.RecordBatchFileWriter(feature_values_path, feature_schema) as feature_writer, ipc.RecordBatchFileWriter(
        label_values_path,
        label_schema,
    ) as label_writer:
        for row_batch in _iter_arrow_row_batches(
            rows,
            target_batch_bytes=ARROW_EXPORT_TARGET_BATCH_BYTES,
        ):
            if not row_batch:
                continue
            batch_seq_len = [int(row.features.shape[0]) for row in row_batch]
            batch_step_count = sum(batch_seq_len)
            seq_len.extend(batch_seq_len)
            item_count += len(row_batch)

            if mode_rows_pretransformed or mode_scaler is None:
                feature_values = _flatten_arrow_row_batch(
                    row_batch,
                    key="features",
                    dtype=np.float32,
                )
            else:
                feature_values = _transform_feature_batch(
                    row_batch,
                    input_feature_names=input_feature_names,
                    scaler=mode_scaler,
                ).reshape(-1)
            if batch_step_count == 0:
                batch_num_features = int(row_batch[0].features.shape[-1])
            else:
                if len(feature_values) % batch_step_count != 0:
                    raise ValueError(
                        f"Mode '{mode_name}' produced {len(feature_values)} feature values for "
                        f"{batch_step_count} steps; cannot infer a consistent feature width."
                    )
                batch_num_features = int(len(feature_values) // batch_step_count)
            if item_count == len(row_batch):
                num_features = batch_num_features
            elif num_features != batch_num_features:
                raise ValueError(
                    f"Mode '{mode_name}' mixes feature widths: {num_features} vs {batch_num_features}."
                )
            label_values = _flatten_arrow_row_batch(
                row_batch,
                key="labels",
                dtype=np.int64,
            )

            feature_writer.write_batch(
                pa.record_batch(
                    [pa.array(feature_values, type=pa.float32())],
                    schema=feature_schema,
                )
            )
            label_writer.write_batch(
                pa.record_batch(
                    [pa.array(label_values, type=pa.int64())],
                    schema=label_schema,
                )
            )
            now = perf_counter()
            if item_count == total_items or now - last_logged_at >= PREPARATION_PROGRESS_LOG_INTERVAL_S:
                logger.info(
                    "Exported mode '%s' for '%s': %d/%d sequences.",
                    mode_name,
                    recipe.requested_dataset_name,
                    item_count,
                    total_items,
                )
                last_logged_at = now

    logger.info(
        "Finished exporting mode '%s' for '%s' in %.1fs.",
        mode_name,
        recipe.requested_dataset_name,
        perf_counter() - started_at,
    )
    return mode_name, PreparedModeManifest(
        mode=mode_name,
        source_name=source_name,
        data_source=recipe.stage_data_source,
        item_count=item_count,
        num_features=num_features,
        seq_len=seq_len,
        feature_values_path=str(feature_values_path),
        label_values_path=str(label_values_path),
    )


def build_manifest_export(
    recipe: PreparationRecipe,
    fingerprint: str,
    prepared_root: str,
    mode_rows: Dict[str, List[CanonicalSequence]],
    feature_names: List[str],
    scaler: Optional[TransformerMixin],
    input_feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Export Arrow IPC arrays and build the manifest payload."""

    started_at = perf_counter()
    bundle_root = Path(prepared_root) / fingerprint
    bundle_root.mkdir(parents=True, exist_ok=True)

    scaler_path: Optional[Path] = None
    if scaler is not None:
        scaler_path = bundle_root / "scaler.pkl"
        with scaler_path.open("wb") as handle:
            pickle.dump(scaler, handle, pickle.HIGHEST_PROTOCOL)

    feature_names_path = bundle_root / "feature_names.json"
    with feature_names_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)
        handle.write("\n")

    manifest_modes: Dict[str, PreparedModeManifest] = {}
    export_mode_names = list(mode_rows)
    effective_input_feature_names = list(input_feature_names) if input_feature_names is not None else list(feature_names)
    mode_rows_pretransformed = input_feature_names is None
    worker_count = _resolve_preparation_max_workers(len(export_mode_names))
    if worker_count == 1:
        for mode_name in export_mode_names:
            resolved_mode_name, mode_manifest = _export_mode_manifest(
                bundle_root=bundle_root,
                recipe=recipe,
                mode_name=mode_name,
                rows=mode_rows[mode_name],
                input_feature_names=effective_input_feature_names,
                scaler=scaler,
                mode_rows_pretransformed=mode_rows_pretransformed,
            )
            manifest_modes[resolved_mode_name] = mode_manifest
    else:
        logger.info(
            "Exporting prepared dataset '%s' with %d Arrow workers across %d modes.",
            recipe.requested_dataset_name,
            worker_count,
            len(export_mode_names),
        )
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="prep-export") as executor:
            future_by_mode = {
                mode_name: executor.submit(
                    _export_mode_manifest,
                    bundle_root=bundle_root,
                    recipe=recipe,
                    mode_name=mode_name,
                    rows=mode_rows[mode_name],
                    input_feature_names=effective_input_feature_names,
                    scaler=scaler,
                    mode_rows_pretransformed=mode_rows_pretransformed,
                )
                for mode_name in export_mode_names
            }
            for mode_name in export_mode_names:
                resolved_mode_name, mode_manifest = future_by_mode[mode_name].result()
                manifest_modes[resolved_mode_name] = mode_manifest

    manifest = PreparedDatasetManifest(
        manifest_version=PREPARED_MANIFEST_VERSION,
        fingerprint=fingerprint,
        exporter_version=PREPARED_EXPORTER_VERSION,
        dataset_name=recipe.dataset_name,
        requested_dataset_name=recipe.requested_dataset_name,
        base_dataset_name=recipe.base_dataset_name,
        stage_data_source=recipe.stage_data_source,
        version=recipe.version,
        source_uri=recipe.source_uri,
        recipe={
            "splits": list(recipe.splits),
            "gen_to_org_factor": recipe.gen_to_org_factor,
            "test_on_org": recipe.test_on_org,
            "scaler_spec": recipe.scaler_spec,
        },
        feature_names=feature_names,
        scaler_path=None if scaler_path is None else str(scaler_path),
        feature_names_path=str(feature_names_path),
        modes=manifest_modes,
    )
    manifest_path = bundle_root / "manifest.json"
    manifest.write(str(manifest_path))
    logger.info(
        "Prepared manifest ready for '%s' at %s in %.1fs (%s).",
        recipe.requested_dataset_name,
        manifest_path,
        perf_counter() - started_at,
        _format_mode_summary(manifest_modes),
    )
    return {
        "manifest": manifest.to_dict(),
        "manifest_path": str(manifest_path),
    }


def build_preparation_recipe(
    *,
    dataset_name: str,
    requested_dataset_name: str,
    runtime_config: RuntimeConfig,
    splits: Tuple[float, float],
    version: str,
    stage_data_source: str,
    gen_to_org_factor: float,
    test_on_org: bool,
    scaler_spec: Optional[Dict[str, Any]],
) -> PreparationRecipe:
    """Resolve raw assets and build the fingerprint recipe."""

    include_generated = stage_data_source == "synthetic"
    data_root, source_uri = ensure_dataset_assets(
        requested_dataset_name,
        runtime_config=runtime_config,
        include_generated=include_generated,
    )
    variant = resolve_dataset_variant(requested_dataset_name, data_root)
    provenance = collect_raw_provenance(data_root, variant, include_generated)
    return PreparationRecipe(
        dataset_name=dataset_name,
        requested_dataset_name=requested_dataset_name,
        base_dataset_name=variant.base_dataset_name,
        version=version,
        stage_data_source=stage_data_source,
        splits=splits,
        gen_to_org_factor=gen_to_org_factor,
        test_on_org=test_on_org,
        scaler_spec=scaler_spec,
        raw_data_root=str(data_root),
        source_uri=source_uri,
        raw_provenance=provenance,
    )


def fingerprint_recipe(recipe: PreparationRecipe) -> str:
    payload = json.dumps(
        {
            "recipe": recipe.to_fingerprint_payload(),
            "manifest_version": PREPARED_MANIFEST_VERSION,
            "exporter_version": PREPARED_EXPORTER_VERSION,
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _execute_preparation_steps(
    *,
    recipe: PreparationRecipe,
    fingerprint: str,
    prepared_root: Path,
) -> Dict[str, Any]:
    """Run the preparation steps in process and return the manifest payload."""

    started_at = perf_counter()
    logger.info(
        "Starting dataset preparation for '%s' (stage_source=%s, fingerprint=%s).",
        recipe.requested_dataset_name,
        recipe.stage_data_source,
        fingerprint,
    )
    logger.info("Preparation step 1/3: loading canonical sequences.")
    load_started_at = perf_counter()
    mode_rows = load_sequence_modes(recipe)
    logger.info(
        "Preparation step 1/3 completed in %.1fs (%s).",
        perf_counter() - load_started_at,
        _format_mode_summary(mode_rows),
    )
    scaler_started_at = perf_counter()
    logger.info("Preparation step 2/3: fitting export scaler and resolving output features.")
    fitted_scaler, feature_names, input_feature_names = fit_scaler_for_export(recipe, mode_rows)
    logger.info(
        "Preparation step 2/3 completed in %.1fs (output_features=%d).",
        perf_counter() - scaler_started_at,
        len(feature_names),
    )
    export_mode_rows = {
        mode_name: rows
        for mode_name, rows in mode_rows.items()
        if mode_name != "scaler_train_real"
    }
    export_started_at = perf_counter()
    logger.info(
        "Preparation step 3/3: exporting Arrow bundles for %d modes (%s).",
        len(export_mode_rows),
        _format_mode_summary(export_mode_rows),
    )
    result = build_manifest_export(
        recipe,
        fingerprint,
        str(prepared_root),
        export_mode_rows,
        feature_names,
        fitted_scaler,
        input_feature_names,
    )
    logger.info(
        "Finished dataset preparation for '%s' in %.1fs (export stage %.1fs).",
        recipe.requested_dataset_name,
        perf_counter() - started_at,
        perf_counter() - export_started_at,
    )
    return result


def run_preparation_pipeline(
    *,
    recipe: PreparationRecipe,
    prepared_root: Path,
) -> str:
    """Execute the preparation steps and return the manifest path."""

    fingerprint = fingerprint_recipe(recipe)
    manifest_path = prepared_root / fingerprint / "manifest.json"
    if manifest_path.exists():
        logger.info(
            "Reusing prepared manifest for '%s' at %s.",
            recipe.requested_dataset_name,
            manifest_path,
        )
        return str(manifest_path)

    manifest_result = _execute_preparation_steps(
        recipe=recipe,
        fingerprint=fingerprint,
        prepared_root=prepared_root,
    )
    return str(manifest_result["manifest_path"])
