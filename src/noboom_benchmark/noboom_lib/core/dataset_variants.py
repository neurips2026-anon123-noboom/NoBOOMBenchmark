from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


OME_BASE_DATASET_NAME = "cont_reactive_ome"
OME_EXTENDED_DATASET_NAME = "cont_reactive_ome_ext"
OME_REDUCED_DATASET_NAME = "cont_reactive_ome_red"
OME_VARIANT_DATASET_NAMES = frozenset(
    {
        OME_EXTENDED_DATASET_NAME,
        OME_REDUCED_DATASET_NAME,
    }
)
OME_EXTENSION_DOWNLOAD_URL = os.environ.get(
    "NOBOOM_OME_EXTENSION_URL",
    "https://anonymous.example.org/noboom/ome-extension/",
)
OME_EXTENSION_SUPPORT_DIRNAME = "_cont_reactive_ome_extension"
OME_EXTENSION_TRAIN_MODE = "train_ext"


@dataclass(frozen=True)
class DatasetVariant:
    requested_name: str
    base_dataset_name: str
    reduced_feature_names: Optional[Tuple[str, ...]] = None
    base_feature_indices: Optional[Tuple[int, ...]] = None
    extension_dir: Optional[Path] = None
    extension_mode: Optional[str] = None
    include_extension_train: bool = False


def resolve_base_dataset_name(dataset_name: str) -> str:
    if dataset_name in OME_VARIANT_DATASET_NAMES:
        return OME_BASE_DATASET_NAME
    return dataset_name


def resolve_download_dataset_name(dataset_name: str) -> str:
    return resolve_base_dataset_name(dataset_name.replace("_tsst", ""))


def is_ome_extension_variant(dataset_name: str) -> bool:
    return dataset_name in OME_VARIANT_DATASET_NAMES


def get_ome_extension_support_dir(data_dir: Union[str, Path]) -> Path:
    return Path(data_dir) / OME_EXTENSION_SUPPORT_DIRNAME


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


def list_ome_extension_csv_files(extension_dir: Union[str, Path]) -> List[Path]:
    root = Path(extension_dir)
    if not root.exists():
        raise FileNotFoundError(f"OME extension directory does not exist: {root}")

    csv_files: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__MACOSX" in path.parts:
            continue
        if path.name.startswith("._") or path.name == ".DS_Store":
            continue
        if path.name.endswith("_styled_features.csv"):
            csv_files.append(path)

    if not csv_files:
        raise RuntimeError(f"No styled feature CSV files found in {root}.")

    return csv_files


def read_ome_extension_feature_names(extension_dir: Union[str, Path]) -> Tuple[str, ...]:
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


def read_base_dataset_feature_names(
    data_dir: Union[str, Path],
    dataset_name: str,
) -> Tuple[str, ...]:
    dataset_dir = Path(data_dir) / dataset_name
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


def resolve_dataset_variant(
    dataset_name: str,
    data_dir: Union[str, Path],
) -> DatasetVariant:
    base_dataset_name = resolve_base_dataset_name(dataset_name)
    if dataset_name not in OME_VARIANT_DATASET_NAMES:
        return DatasetVariant(
            requested_name=dataset_name,
            base_dataset_name=base_dataset_name,
        )

    base_feature_names = read_base_dataset_feature_names(data_dir, base_dataset_name)
    extension_dir = get_ome_extension_support_dir(data_dir)
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

    return DatasetVariant(
        requested_name=dataset_name,
        base_dataset_name=base_dataset_name,
        reduced_feature_names=reduced_feature_names,
        base_feature_indices=tuple(base_feature_index[name] for name in reduced_feature_names),
        extension_dir=extension_dir,
        extension_mode=OME_EXTENSION_TRAIN_MODE,
        include_extension_train=dataset_name == OME_EXTENDED_DATASET_NAME,
    )
