"""Manifest types for prepared Arrow IPC datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class PreparedModeManifest:
    """Metadata for one prepared dataset mode."""

    mode: str
    source_name: str
    data_source: str
    item_count: int
    num_features: int
    seq_len: List[int]
    feature_values_path: str
    label_values_path: str


@dataclass(frozen=True)
class PreparedDatasetManifest:
    """Top-level manifest describing one prepared dataset bundle."""

    manifest_version: int
    fingerprint: str
    exporter_version: str
    dataset_name: str
    requested_dataset_name: str
    base_dataset_name: str
    stage_data_source: str
    version: str
    source_uri: Optional[str]
    recipe: Dict[str, Any]
    feature_names: List[str]
    scaler_path: Optional[str]
    feature_names_path: Optional[str]
    modes: Dict[str, PreparedModeManifest]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreparedDatasetManifest":
        modes = {
            str(name): PreparedModeManifest(**dict(mode_payload))
            for name, mode_payload in dict(payload["modes"]).items()
        }
        return cls(
            manifest_version=int(payload["manifest_version"]),
            fingerprint=str(payload["fingerprint"]),
            exporter_version=str(payload["exporter_version"]),
            dataset_name=str(payload["dataset_name"]),
            requested_dataset_name=str(payload["requested_dataset_name"]),
            base_dataset_name=str(payload["base_dataset_name"]),
            stage_data_source=str(payload["stage_data_source"]),
            version=str(payload["version"]),
            source_uri=None if payload.get("source_uri") is None else str(payload["source_uri"]),
            recipe=dict(payload["recipe"]),
            feature_names=[str(name) for name in payload["feature_names"]],
            scaler_path=None if payload.get("scaler_path") is None else str(payload["scaler_path"]),
            feature_names_path=(
                None if payload.get("feature_names_path") is None else str(payload["feature_names_path"])
            ),
            modes=modes,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["modes"] = {
            mode_name: asdict(mode_manifest)
            for mode_name, mode_manifest in self.modes.items()
        }
        return payload

    @classmethod
    def load(cls, path: str) -> "PreparedDatasetManifest":
        return _load_manifest_cached(str(Path(path).expanduser().resolve()))

    def write(self, path: str) -> str:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        return str(manifest_path)


@lru_cache(maxsize=64)
def _load_manifest_cached(path: str) -> PreparedDatasetManifest:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return PreparedDatasetManifest.from_dict(payload)
