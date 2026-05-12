import argparse
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Sequence
from urllib.request import urlopen
import zipfile

from filelock import FileLock
import kagglehub
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config.runtime import RuntimeConfig
from .dataset_variants import (
    OME_BASE_DATASET_NAME,
    OME_EXTENDED_DATASET_NAME,
    OME_EXTENSION_DOWNLOAD_URL,
    OME_REDUCED_DATASET_NAME,
    get_ome_extension_support_dir,
    list_ome_extension_csv_files,
    resolve_download_dataset_name,
)
from .logging import setup_logging
from .webdav_utils import download_from_webdav

logger = logging.getLogger(__name__)

KAGGLE_DATASET_SLUG = "faebs94/noboom-anomaly-detection-in-chemical-processes/versions/1"
TSST_WEBDAV_HOSTNAME = os.environ.get("NOBOOM_TSST_WEBDAV_URL", "https://anonymous.example.org/seafdav/")


def get_kagglehub_cache_path() -> Path:
    """Resolve the KaggleHub cache directory."""
    cache_env = os.getenv("KAGGLEHUB_CACHE")
    if not cache_env:
        logger.warning(
            "KAGGLEHUB_CACHE is not set; kagglehub will use its default cache location."
        )
        return Path.home() / ".cache" / "kagglehub"
    return Path(cache_env)


def get_dataset_mapping_path() -> Path:
    """Return the JSON mapping path for cached dataset locations."""
    return get_kagglehub_cache_path() / "dataset_paths.json"


def load_dataset_paths(datasets: Sequence[str]) -> Dict[str, str]:
    """Load dataset paths from the cached mapping file.

    Args:
        datasets (Sequence[str]): Dataset names to resolve.

    Returns:
        Dict[str, str]: Mapping of dataset names to local paths.

    Raises:
        FileNotFoundError: If the mapping file is missing.
        KeyError: If any requested datasets are not present in the mapping.
    """
    mapping_path = get_dataset_mapping_path()
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Dataset mapping not found at {mapping_path}. "
            "Run the orchestration dataset setup first."
        )

    with mapping_path.open("r", encoding="utf-8") as handle:
        dataset_paths = json.load(handle)

    missing = [dataset for dataset in datasets if dataset not in dataset_paths]
    if missing:
        raise KeyError(
            f"Dataset mapping missing entries for: {', '.join(missing)}."
        )

    return {dataset: dataset_paths[dataset] for dataset in datasets}


def preprocess_noboom_dataset(data_dir: str | Path, ds_name: str) -> None:
    """Convert dataset CSV files to Parquet format.

    Args:
        data_dir (str | Path): Root directory containing dataset files.
        ds_name (str): Dataset name used for logging.

    Returns:
        None: Files are converted in place.

    Side Effects:
        Reads CSV files and writes Parquet files to disk.
    """
    root_path = Path(data_dir)
    logger.info("Starting preprocessing for dataset '%s' in directory '%s'.", ds_name, root_path)

    def has_ancestor_named(p: Path) -> bool:
        # checks all parents up to filesystem root
        return any(parent.name == ds_name for parent in p.parents)

    dataset_files = sorted(path for path in root_path.rglob("*.csv") if has_ancestor_named(path))
    logger.info("Found %d CSV files to process for dataset '%s'.", len(dataset_files), ds_name)
    if dataset_files:
        def load_func(path: Path) -> pd.DataFrame:
            return pd.read_csv(path)

        file_mode = 'CSV'
    else:
        dataset_files = sorted(path for path in root_path.rglob("*.npy") if has_ancestor_named(path))
        def load_func(path: Path) -> pd.DataFrame:
            return pd.DataFrame(np.load(path))

        file_mode = 'NPY'

    for file_path in tqdm(dataset_files, desc=f"Preprocessing {ds_name} dataset"):
        parquet_path = file_path.with_suffix(".parquet")
        if parquet_path.exists():
            logger.debug("Parquet already exists, skipping: %s", parquet_path)
            continue

        logger.debug(f"Reading {file_mode} file: %s", file_path)
        time_series = load_func(file_path)

        logger.debug("Writing Parquet file: %s", parquet_path)
        time_series.to_parquet(parquet_path)

    logger.info("Finished preprocessing dataset '%s'.", ds_name)


def _download_public_file(url: str, destination: Path) -> None:
    logger.info("Downloading public dataset archive from '%s'.", url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


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


def prepare_ome_extension_support(
    data_dir: str | Path,
    download_url: str = OME_EXTENSION_DOWNLOAD_URL,
) -> Path:
    support_dir = get_ome_extension_support_dir(data_dir)
    try:
        list_ome_extension_csv_files(support_dir)
        logger.info("OME extension support already prepared at '%s'.", support_dir)
        return support_dir
    except (FileNotFoundError, RuntimeError):
        logger.info("Preparing OME extension support directory at '%s'.", support_dir)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "ome_generated.zip"
        extracted_root = tmp_dir / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)

        _download_public_file(download_url, archive_path)
        _extract_ome_extension_archive(archive_path, extracted_root)
        list_ome_extension_csv_files(extracted_root)

        if support_dir.exists():
            shutil.rmtree(support_dir)
        shutil.copytree(extracted_root, support_dir)

    logger.info("OME extension support prepared at '%s'.", support_dir)
    return support_dir


def ensure_datasets_downloaded(
    datasets: Sequence[str],
    runtime_config: RuntimeConfig,
    output_path: Optional[str | Path] = None,
    verbosity: int = 0,
) -> Dict[str, str]:
    """Download datasets (if needed) and return their local paths.

    Args:
        datasets (Sequence[str]): Dataset names to preprocess.
        output_path (str | Path | None): Optional path to write JSON mapping.
            Defaults to None.
        verbosity (int): Verbosity level for logging. Defaults to 0.

    Returns:
        Dict[str, str]: Mapping of dataset names to local paths.

    Side Effects:
        Downloads data via KaggleHub, writes files, and may write a JSON mapping.
    """
    logger.info("Ensuring datasets are downloaded: %s", list(datasets))

    cache_path = get_kagglehub_cache_path()

    cache_path.mkdir(parents=True, exist_ok=True)
    logger.info("Using Kaggle cache directory: %s", cache_path)

    lock_path = cache_path / "dataset_download.lock"
    logger.debug("Lock file path: %s", lock_path)

    dataset_paths: Dict[str, str] = {}

    with FileLock(str(lock_path)):
        logger.info("Acquiring exclusive lock for dataset download.")

        requested_datasets = list(datasets)
        org_datasets = sorted({resolve_download_dataset_name(dataset) for dataset in requested_datasets})
        logger.info(
            "Downloading/using cached Kaggle dataset '%s'.",
            KAGGLE_DATASET_SLUG,
        )
        dataset_root = Path(
            kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
        )
        logger.info("Original dataset root resolved to: %s", dataset_root)

        for dataset in org_datasets:
            logger.info("Processing dataset '%s'.", dataset)
            preprocess_noboom_dataset(dataset_root, dataset)
            ds_path = dataset_root / "noboom" / dataset
            dataset_paths[dataset] = str(ds_path)
            logger.info("Dataset '%s' path set to: %s", dataset, ds_path)

        if any(dataset in {OME_EXTENDED_DATASET_NAME, OME_REDUCED_DATASET_NAME} for dataset in requested_datasets):
            ome_data_root = dataset_root / "noboom"
            prepare_ome_extension_support(ome_data_root)
            base_path = dataset_paths[OME_BASE_DATASET_NAME]
            dataset_paths[OME_EXTENDED_DATASET_NAME] = base_path
            dataset_paths[OME_REDUCED_DATASET_NAME] = base_path
            logger.info(
                "Registered OME extension dataset aliases: %s -> %s, %s -> %s",
                OME_EXTENDED_DATASET_NAME,
                base_path,
                OME_REDUCED_DATASET_NAME,
                base_path,
            )

        tsst_datasets = [d for d in requested_datasets if 'tsst' in d]
        if tsst_datasets:
            dataset_root = Path(download_from_webdav(
                webdav_hostname=TSST_WEBDAV_HOSTNAME,
                webdav_login=runtime_config.seafile_username,
                webdav_password=runtime_config.seafile_password,
                remote_folder="NOBOOM_TSST",
                out_dir=cache_path))

            for dataset in tsst_datasets:
                logger.info("Processing dataset '%s'.", dataset)
                ds_path = dataset_root / "noboom" / dataset
                dataset_paths[dataset] = str(ds_path)
                try:
                    Path(dataset_paths[dataset.replace("_tsst", "")] + '_tsst').symlink_to(ds_path,
                                                                                           target_is_directory=True)
                except FileExistsError:
                    pass
                logger.info("Dataset '%s' path set to: %s", dataset, ds_path)

        if output_path is not None:
            output_path = Path(output_path)
            logger.info("Writing dataset paths mapping to: %s", output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as output_file:
                json.dump(dataset_paths, output_file)
            logger.info("Dataset paths written successfully.")

    logger.info("All requested datasets processed. Paths: %s", dataset_paths)
    return dataset_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", help="Dataset names.")
    parser.add_argument("--output", type=str, help="Write dataset paths to this file.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (INFO by default, DEBUG with -v).",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("CLI arguments: %s", args)

    if not args.dataset:
        logger.error("--dataset is required.")
        sys.exit(1)
    if not args.output:
        logger.error("--output is required.")
        sys.exit(1)

    ensure_datasets_downloaded(args.dataset, args.output)
