from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterator
import json
from pathlib import Path
from types import SimpleNamespace

from botocore.exceptions import ClientError

from noboom_benchmark.noboom_lib.core.data_prep import runtime
from noboom_benchmark.noboom_lib.core.data_prep.manifest import PreparedDatasetManifest


def _write_prepared_bundle(bundle_root: Path, *, fingerprint: str) -> Path:
    train_dir = bundle_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    feature_names_path = bundle_root / "feature_names.json"
    feature_names_path.write_text(json.dumps(["f1"]), encoding="utf-8")
    feature_values_path = train_dir / "feature_values.arrow"
    label_values_path = train_dir / "label_values.arrow"
    feature_values_path.write_bytes(b"feature-bytes")
    label_values_path.write_bytes(b"label-bytes")

    manifest_payload = {
        "manifest_version": 2,
        "fingerprint": fingerprint,
        "exporter_version": "arrow-ipc-v1",
        "dataset_name": "cont_reactive_ome",
        "requested_dataset_name": "cont_reactive_ome",
        "base_dataset_name": "cont_reactive_ome",
        "stage_data_source": "real",
        "version": "1.0",
        "source_uri": None,
        "recipe": {
            "splits": [0.7, 0.3],
            "gen_to_org_factor": -1.0,
            "test_on_org": False,
            "scaler_spec": None,
        },
        "feature_names": ["f1"],
        "scaler_path": None,
        "feature_names_path": str(feature_names_path),
        "modes": {
            "train": {
                "mode": "train",
                "source_name": "real",
                "data_source": "real",
                "item_count": 1,
                "num_features": 1,
                "seq_len": [1],
                "feature_values_path": str(feature_values_path),
                "label_values_path": str(label_values_path),
            }
        },
    }
    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


class _FakePaginator:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def paginate(self, *, Bucket: str, Prefix: str) -> Iterator[dict[str, object]]:
        del Bucket
        contents = [
            {"Key": key}
            for key in sorted(self._objects)
            if key.startswith(Prefix)
        ]
        yield {"Contents": contents}


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        del Bucket
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {}

    def get_paginator(self, operation_name: str) -> _FakePaginator:
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self.objects)

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        del Bucket
        destination = Path(Filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[Key])

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        del Bucket
        self.objects[Key] = Path(Filename).read_bytes()

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        IfNoneMatch: str = "",
    ) -> dict[str, object]:
        del Bucket
        del ContentType
        if IfNoneMatch == "*" and Key in self.objects:
            raise ClientError(
                {"Error": {"Code": "412", "Message": "Precondition Failed"}},
                "PutObject",
            )
        self.objects[Key] = bytes(Body)
        return {}

    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        del Bucket
        self.objects.pop(Key, None)
        return {}


def test_manifest_path_for_stage_rebuilds_missing_manifest_locally(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rebuilt_manifest_path = tmp_path / "rebuilt" / "manifest.json"

    monkeypatch.setattr(
        runtime,
        "resolve_model_preparation_settings",
        lambda **kwargs: {
            "splits": (0.7, 0.3),
            "version": "1.0",
            "gen_to_org_factor": -1.0,
            "test_on_org": False,
            "scaler_spec": None,
        },
    )

    def fake_ensure_prepared_dataset_manifest(**kwargs) -> str:
        rebuilt_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rebuilt_manifest_path.write_text("{}", encoding="utf-8")
        return str(rebuilt_manifest_path)

    monkeypatch.setattr(
        runtime,
        "ensure_prepared_dataset_manifest",
        fake_ensure_prepared_dataset_manifest,
    )

    args = Namespace(
        prepared_manifest_paths={"real": str(tmp_path / "missing" / "manifest.json")},
        runtime_config=SimpleNamespace(),
        config_dir="/configs",
        model_name="neutralad",
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        verbose=1,
    )

    resolved_path = runtime.manifest_path_for_stage_from_args(args, "real")

    assert resolved_path == str(rebuilt_manifest_path)
    assert args.prepared_manifest_paths["real"] == str(rebuilt_manifest_path)
    assert args.data_manifest_path == str(rebuilt_manifest_path)


def test_resolve_model_preparation_settings_merges_scaler_init_args_from_hparams(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "_load_model_data_config",
        lambda config_dir, model_name: {
            "splits": [0.7, 0.3],
            "version": "1.0",
            "gen_to_org_factor": -1.0,
            "test_on_org": False,
            "scaler": {
                "class_path": "pkg.Scaler",
                "init_args": {
                    "enable_asinh": True,
                    "drop_features": [],
                },
            },
        },
    )

    settings = runtime.resolve_model_preparation_settings(
        config_dir="/configs",
        model_name="neutralad",
        hp_params={
            "data": {
                "scaler": {
                    "init_args": {
                        "drop_features": ["LS701", "LS702"],
                    },
                },
            },
        },
    )

    assert settings["scaler_spec"] == {
        "class_path": "pkg.Scaler",
        "init_args": {
            "enable_asinh": True,
            "drop_features": ["LS701", "LS702"],
        },
    }


def test_prepared_bundle_remote_location_supports_explicit_s3_path_override() -> None:
    runtime_config = SimpleNamespace(
        s3_bucket="default-bucket",
        s3_prefix=Path("experiment_data"),
        prepared_dataset_s3_path="s3://prepared-bucket/custom/root",
    )

    assert runtime._prepared_bundle_remote_location(
        runtime_config,
        fingerprint="fingerprint",
    ) == ("prepared-bucket", "custom/root/fingerprint")

    runtime_config.prepared_dataset_s3_path = "custom/root"

    assert runtime._prepared_bundle_remote_location(
        runtime_config,
        fingerprint="fingerprint",
    ) == ("default-bucket", "custom/root/fingerprint")


def test_ensure_prepared_dataset_manifest_downloads_bundle_from_s3_before_rebuild(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fingerprint = "fingerprint"
    source_bundle_root = tmp_path / "source" / fingerprint
    manifest_path = _write_prepared_bundle(source_bundle_root, fingerprint=fingerprint)
    portable_payload = runtime._portable_manifest_payload(manifest_path, source_bundle_root)

    fake_s3_client = _FakeS3Client()
    remote_prefix = "experiment_data/prepared_datasets/fingerprint"
    for local_path in sorted(source_bundle_root.rglob("*")):
        if not local_path.is_file() or local_path.name == "manifest.json":
            continue
        fake_s3_client.objects[
            f"{remote_prefix}/{local_path.relative_to(source_bundle_root).as_posix()}"
        ] = local_path.read_bytes()
    fake_s3_client.objects[f"{remote_prefix}/manifest.json"] = (
        json.dumps(portable_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    monkeypatch.setattr(runtime, "create_s3_client", lambda: fake_s3_client)
    monkeypatch.setattr(runtime, "ensure_bucket_exists", lambda bucket: None)
    monkeypatch.setattr(runtime, "build_preparation_recipe", lambda **kwargs: {"recipe": kwargs})
    monkeypatch.setattr(runtime, "fingerprint_recipe", lambda recipe: fingerprint)
    monkeypatch.setattr(
        runtime,
        "run_preparation_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("bundle should be downloaded, not rebuilt")),
    )

    runtime_config = SimpleNamespace(
        mapped_storage=str(tmp_path / "node-storage"),
        s3_bucket="bucket",
        s3_prefix=Path("experiment_data"),
    )

    resolved_manifest_path = runtime.ensure_prepared_dataset_manifest(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        runtime_config=runtime_config,
        splits=(0.7, 0.3),
        version="1.0",
        stage_data_source="real",
        gen_to_org_factor=-1.0,
        test_on_org=False,
        scaler_spec=None,
    )

    bundle_root = Path(resolved_manifest_path).parent
    manifest = PreparedDatasetManifest.load(resolved_manifest_path)
    assert resolved_manifest_path == str(bundle_root / "manifest.json")
    assert manifest.feature_names_path == str((bundle_root / "feature_names.json").resolve(strict=False))
    assert manifest.modes["train"].feature_values_path == str(
        (bundle_root / "train" / "feature_values.arrow").resolve(strict=False)
    )
    assert Path(manifest.modes["train"].feature_values_path).read_bytes() == b"feature-bytes"
    assert Path(manifest.modes["train"].label_values_path).read_bytes() == b"label-bytes"


def test_ensure_prepared_dataset_manifest_uploads_new_bundle_to_s3_after_build(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fingerprint = "fingerprint"
    fake_s3_client = _FakeS3Client()
    ensured_buckets: list[str] = []

    monkeypatch.setattr(runtime, "create_s3_client", lambda: fake_s3_client)
    monkeypatch.setattr(runtime, "ensure_bucket_exists", lambda bucket: ensured_buckets.append(bucket))
    monkeypatch.setattr(runtime, "build_preparation_recipe", lambda **kwargs: {"recipe": kwargs})
    monkeypatch.setattr(runtime, "fingerprint_recipe", lambda recipe: fingerprint)

    def fake_run_preparation_pipeline(*, recipe, prepared_root: Path) -> str:
        del recipe
        bundle_root = prepared_root / fingerprint
        manifest_path = _write_prepared_bundle(bundle_root, fingerprint=fingerprint)
        return str(manifest_path)

    monkeypatch.setattr(runtime, "run_preparation_pipeline", fake_run_preparation_pipeline)

    runtime_config = SimpleNamespace(
        mapped_storage=str(tmp_path / "node-storage"),
        s3_bucket="bucket",
        s3_prefix=Path("experiment_data"),
    )

    resolved_manifest_path = runtime.ensure_prepared_dataset_manifest(
        dataset_name="cont_reactive_ome",
        requested_dataset_name="cont_reactive_ome",
        runtime_config=runtime_config,
        splits=(0.7, 0.3),
        version="1.0",
        stage_data_source="real",
        gen_to_org_factor=-1.0,
        test_on_org=False,
        scaler_spec=None,
    )

    assert Path(resolved_manifest_path).exists()
    assert ensured_buckets == ["bucket", "bucket"]

    remote_prefix = "experiment_data/prepared_datasets/fingerprint"
    assert f"{remote_prefix}/train/feature_values.arrow" in fake_s3_client.objects
    assert f"{remote_prefix}/train/label_values.arrow" in fake_s3_client.objects
    assert f"{remote_prefix}/feature_names.json" in fake_s3_client.objects
    assert f"{remote_prefix}/manifest.json" in fake_s3_client.objects
    assert f"{remote_prefix}/.build.lock" not in fake_s3_client.objects

    remote_manifest = json.loads(fake_s3_client.objects[f"{remote_prefix}/manifest.json"].decode("utf-8"))
    assert remote_manifest["feature_names_path"] == "feature_names.json"
    assert remote_manifest["modes"]["train"]["feature_values_path"] == "train/feature_values.arrow"
    assert remote_manifest["modes"]["train"]["label_values_path"] == "train/label_values.arrow"
