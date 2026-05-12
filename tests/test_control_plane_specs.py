from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from noboom_cluster.noboom_cli_lib.dependency_resolver import DependencyResolver
from noboom_cluster.noboom_cli_lib.dependency_resolver import (
    NOBOOM_REPOSITORY_URL,
    TIMESEAD_EXTENSIONS_REPOSITORY_URL,
    TIMESEAD_REPOSITORY_URL,
)
from noboom_cluster.noboom_cli_lib.specs import DependencyManifest, InventoryConfig, PairRunSpec


def test_inventory_config_load_preserves_device_mapping(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.yaml"
    inventory_path.write_text(
        "nodes:\n"
        "  - ip: 1.2.3.4\n"
        "    devices: 0,1\n"
        "    ssh_user: ubuntu\n"
        "  - ip: 5.6.7.8\n",
        encoding="utf-8",
    )

    inventory = InventoryConfig.load(str(inventory_path))

    assert [node.ip for node in inventory.nodes] == ["1.2.3.4", "5.6.7.8"]
    assert inventory.nodes[0].devices == "0,1"
    assert inventory.nodes[0].ssh_user == "ubuntu"
    assert inventory.nodes[0].resolved_ssh_user("cloud") == "ubuntu"
    assert inventory.nodes[1].devices is None
    assert inventory.nodes[1].ssh_user is None
    assert inventory.nodes[1].resolved_ssh_user("cloud") == "cloud"


def test_inventory_config_load_resolves_package_relative_inventory() -> None:
    inventory = InventoryConfig.load("a100.yaml")

    assert [node.ip for node in inventory.nodes] == ["203.0.113.10", "203.0.113.11", "203.0.113.12"]
    assert inventory.nodes[0].devices == "0,1,2,3,4,5,6,7"


def test_dependency_manifest_round_trip(tmp_path: Path) -> None:
    manifest = DependencyManifest(
        noboom_sha="abc123",
        timesead_sha="def456",
        timesead_extensions_sha="ghi789",
    )
    manifest_path = tmp_path / "dependency-manifest.json"
    manifest.write_json(str(manifest_path))

    loaded = DependencyResolver.load(str(manifest_path))

    assert loaded == manifest


def test_dependency_resolver_uses_expected_refs(monkeypatch) -> None:
    calls: List[Tuple[str, str]] = []

    def fake_resolve_remote_sha(repo_url: str, ref: str = "HEAD") -> str:
        calls.append((repo_url, ref))
        return f"sha-for-{len(calls)}"

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.dependency_resolver.resolve_remote_sha",
        fake_resolve_remote_sha,
    )

    manifest = DependencyResolver().resolve()

    assert manifest == DependencyManifest(
        noboom_sha="sha-for-1",
        timesead_sha="sha-for-2",
        timesead_extensions_sha="sha-for-3",
    )
    assert calls == [
        (NOBOOM_REPOSITORY_URL, "HEAD"),
        (TIMESEAD_REPOSITORY_URL, "HEAD"),
        (TIMESEAD_EXTENSIONS_REPOSITORY_URL, "HEAD"),
    ]


def test_pair_run_spec_round_trip(tmp_path: Path) -> None:
    pair_spec = PairRunSpec(
        experiment_id="1",
        source_experiment_id="2",
        model_name="gdn",
        dataset_name="ome",
        timestamp="20260311_120000",
        storage_path="/tmp/storage/ray/ome__gdn?scheme=http&endpoint_override=http://127.0.0.1:8333",
        gpus_per_run=0.25,
        optuna_storage_uri="postgresql+psycopg://noboom:example-password@127.0.0.1:5432/optuna_db",
        config_dir="configs",
        temp_dir="/tmp/ray",
        verbose=1,
        tune=True,
        env_file=None,
        tracking_uri="http://127.0.0.1:5000",
        s3_endpoint_url="http://127.0.0.1:8333",
        prepared_dataset_s3_path="s3://prepared-bucket/prepared-root",
    )
    pair_spec_path = tmp_path / "pair-spec.json"
    pair_spec.write_json(str(pair_spec_path))

    loaded = PairRunSpec.load(str(pair_spec_path))

    assert loaded == pair_spec
    assert loaded.submission_id == "ome__gdn__20260311_120000"
