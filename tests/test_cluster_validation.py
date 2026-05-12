from __future__ import annotations

from dataclasses import replace

import pytest

from noboom_cluster.noboom_cli_lib.cluster_validation import (
    AUTOSCALER_UNSATISFIABLE_PHRASE,
    ClusterValidationConfig,
    ClusterValidationError,
    ExpectedClusterNode,
    RayNodeView,
    build_expected_cluster_nodes,
    validate_cluster_before_submission,
    validate_cluster_snapshot,
)
from noboom_cluster.noboom_cli_lib.specs import NodeSpec


def _config() -> ClusterValidationConfig:
    return ClusterValidationConfig(
        config_path="cluster.yaml",
        expected_nodes=[
            ExpectedClusterNode(
                label="head",
                ssh_host="203.0.113.10",
                selected_ray_address="10.0.0.10",
                expected_gpus=0.0,
            ),
            ExpectedClusterNode(
                label="worker-1",
                ssh_host="203.0.113.11",
                selected_ray_address="10.0.0.11",
                expected_gpus=2.0,
            ),
        ],
        gpus_per_run=0.5,
        max_in_flight=2,
        datasets=["ome"],
        models=["gdn"],
    )


def test_validation_passes_when_expected_workers_and_resources_are_active() -> None:
    rows = validate_cluster_snapshot(
        config=_config(),
        ray_nodes=[
            RayNodeView(
                selected_ray_address="10.0.0.10",
                alive=True,
                resources={"CPU": 8, "exclusive": 1},
            ),
            RayNodeView(
                selected_ray_address="10.0.0.11",
                alive=True,
                resources={"CPU": 16, "GPU": 2, "exclusive": 1},
            ),
        ],
        cluster_resources={"CPU": 24, "GPU": 2, "exclusive": 2},
    )

    assert rows == []


def test_expected_nodes_report_selected_address_but_match_generated_local_address() -> None:
    expected_nodes = build_expected_cluster_nodes(
        public_nodes=[
            NodeSpec(ip="203.0.113.10"),
            NodeSpec(ip="203.0.113.11", devices="0,1"),
        ],
        local_nodes=[
            NodeSpec(ip="10.0.0.10"),
            NodeSpec(ip="10.0.0.11", devices="0,1"),
        ],
    )
    config = ClusterValidationConfig(
        config_path="cluster.yaml",
        expected_nodes=expected_nodes,
        gpus_per_run=0.5,
        max_in_flight=1,
        datasets=["ome"],
        models=["gdn"],
    )

    rows = validate_cluster_snapshot(
        config=config,
        ray_nodes=[
            RayNodeView(
                selected_ray_address="10.0.0.10",
                alive=True,
                resources={"CPU": 8, "exclusive": 1},
            ),
            RayNodeView(
                selected_ray_address="10.0.0.11",
                alive=True,
                resources={"CPU": 16, "GPU": 2, "exclusive": 1},
            ),
        ],
        cluster_resources={"CPU": 24, "GPU": 2, "exclusive": 2},
    )

    assert expected_nodes[1].selected_ray_address == "203.0.113.11"
    assert rows == []


def test_validation_fails_when_only_head_is_active() -> None:
    rows = validate_cluster_snapshot(
        config=_config(),
        ray_nodes=[
            RayNodeView(
                selected_ray_address="10.0.0.10",
                alive=True,
                resources={"CPU": 8, "exclusive": 1},
            ),
        ],
        cluster_resources={"CPU": 8, "exclusive": 1},
        ray_status_text="10.0.0.11 pending: uninitialized",
    )

    assert [row.label for row in rows] == ["worker-1", "cluster"]
    assert rows[0].status == "pending"
    assert "expected Ray node is not active" in rows[0].reason
    assert "active Ray nodes 1 < expected 2" in rows[1].reason


def test_validation_fails_when_gpu_and_exclusive_resources_are_low() -> None:
    rows = validate_cluster_snapshot(
        config=_config(),
        ray_nodes=[
            RayNodeView(
                selected_ray_address="10.0.0.10",
                alive=True,
                resources={"CPU": 8, "exclusive": 1},
            ),
            RayNodeView(
                selected_ray_address="10.0.0.11",
                alive=True,
                resources={"CPU": 16, "GPU": 1, "exclusive": 0.5},
            ),
        ],
        cluster_resources={"CPU": 24, "GPU": 1, "exclusive": 1.5},
    )

    assert [row.label for row in rows] == ["worker-1", "cluster"]
    assert "GPU resources 1 < expected 2" in rows[0].reason
    assert "exclusive resources 0.5 < expected 1" in rows[0].reason
    assert "total GPUs 1 < expected 2" in rows[1].reason
    assert "total exclusive resources 1.5 < expected 2" in rows[1].reason


def test_validation_fails_when_placement_bundle_cannot_be_satisfied() -> None:
    config = ClusterValidationConfig(
        config_path="cluster.yaml",
        expected_nodes=[
            ExpectedClusterNode(
                label="head",
                ssh_host="203.0.113.10",
                selected_ray_address="10.0.0.10",
                expected_gpus=0.0,
            ),
        ],
        gpus_per_run=1.0,
        max_in_flight=1,
        datasets=["ome"],
        models=["gdn"],
    )

    rows = validate_cluster_snapshot(
        config=config,
        ray_nodes=[
            RayNodeView(
                selected_ray_address="10.0.0.10",
                alive=True,
                resources={"CPU": 8, "GPU": 0.5, "exclusive": 1},
            ),
        ],
        cluster_resources={"CPU": 8, "GPU": 0.5, "exclusive": 1},
    )

    assert [row.label for row in rows] == ["cluster"]
    assert "no active node can satisfy placement bundle" in rows[0].reason


def test_validation_fails_on_pre_submit_autoscaler_unsatisfiable_message(monkeypatch) -> None:
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.cluster_validation.query_ray_cluster_snapshot",
        lambda config_path, **kwargs: {
            "ray_status_stdout": AUTOSCALER_UNSATISFIABLE_PHRASE,
            "nodes": [
                {
                    "NodeManagerAddress": "10.0.0.10",
                    "Alive": True,
                    "Resources": {"CPU": 8, "exclusive": 1},
                },
                {
                    "NodeManagerAddress": "10.0.0.11",
                    "Alive": True,
                    "Resources": {"CPU": 16, "GPU": 2, "exclusive": 1},
                },
            ],
            "cluster_resources": {"CPU": 24, "GPU": 2, "exclusive": 2},
        },
    )

    with pytest.raises(ClusterValidationError, match=AUTOSCALER_UNSATISFIABLE_PHRASE):
        validate_cluster_before_submission(_config())


def test_validation_retries_until_cluster_becomes_ready(monkeypatch) -> None:
    snapshots = iter(
        [
            {
                "ray_status_stdout": "10.0.0.11 pending: setting-up",
                "nodes": [
                    {
                        "NodeManagerAddress": "10.0.0.10",
                        "Alive": True,
                        "Resources": {"CPU": 8, "exclusive": 1},
                    },
                ],
                "cluster_resources": {"CPU": 8, "exclusive": 1},
            },
            {
                "ray_status_stdout": "",
                "nodes": [
                    {
                        "NodeManagerAddress": "10.0.0.10",
                        "Alive": True,
                        "Resources": {"CPU": 8, "exclusive": 1},
                    },
                    {
                        "NodeManagerAddress": "10.0.0.11",
                        "Alive": True,
                        "Resources": {"CPU": 16, "GPU": 2, "exclusive": 1},
                    },
                ],
                "cluster_resources": {"CPU": 24, "GPU": 2, "exclusive": 2},
            },
        ]
    )
    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.cluster_validation.query_ray_cluster_snapshot",
        lambda config_path, **kwargs: next(snapshots),
    )

    validate_cluster_before_submission(
        replace(
            _config(),
            poll_timeout_s=1.0,
            poll_interval_s=0.0,
        )
    )


def test_query_validation_falls_back_to_head_ssh_when_ray_exec_fails(
    monkeypatch,
    tmp_path,
) -> None:
    cluster_config = tmp_path / "ray.yaml"
    cluster_config.write_text(
        "provider:\n"
        "  head_ip: 203.0.113.10\n"
        "auth:\n"
        "  ssh_user: ubuntu\n"
        "  ssh_private_key: ~/.ssh/id_ed25519\n"
        "docker:\n"
        "  container_name: ray-noboom-container\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(command, capture_output, text, check):
        calls.append(command)
        if command[0:2] == ["ray", "exec"]:
            return type(
                "Result",
                (),
                {"returncode": 1, "stdout": "", "stderr": "ray exec failed"},
            )()
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": (
                    "__NOBOOM_CLUSTER_VALIDATION_JSON_START__\n"
                    '{"ray_status_stdout":"","nodes":[],"cluster_resources":{}}\n'
                    "__NOBOOM_CLUSTER_VALIDATION_JSON_END__\n"
                ),
                "stderr": "",
            },
        )()

    monkeypatch.setattr(
        "noboom_cluster.noboom_cli_lib.cluster_validation.subprocess.run",
        fake_run,
    )

    from noboom_cluster.noboom_cli_lib.cluster_validation import query_ray_cluster_snapshot

    payload = query_ray_cluster_snapshot(
        str(cluster_config),
        head_ssh_host="198.51.100.20",
        head_ssh_user="user",
        head_ssh_key="~/.ssh/id_ed25519",
    )

    assert payload["nodes"] == []
    assert calls[0][0:2] == ["ray", "exec"]
    assert calls[1][0] == "ssh"
    assert "user@198.51.100.20" in calls[1]
    assert "~/.ssh/id_ed25519" not in calls[1]
    assert any("id_ed25519" in part for part in calls[1])
    assert "docker exec -i ray-noboom-container" in calls[1][-1]
