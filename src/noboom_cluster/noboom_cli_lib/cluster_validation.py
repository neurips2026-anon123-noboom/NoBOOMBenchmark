"""Fail-fast validation for Ray cluster state before benchmark submission."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any, Mapping, Optional, Sequence

import yaml

from .specs import NodeSpec

logger = logging.getLogger(__name__)

AUTOSCALER_UNSATISFIABLE_PHRASE = "No available node types can fulfill resource request"


@dataclass(frozen=True)
class ExpectedClusterNode:
    label: str
    ssh_host: str
    selected_ray_address: str
    expected_gpus: float
    match_addresses: Sequence[str] = ()


@dataclass(frozen=True)
class ClusterValidationConfig:
    config_path: str
    expected_nodes: Sequence[ExpectedClusterNode]
    gpus_per_run: float
    max_in_flight: int
    datasets: Sequence[str] = ()
    models: Sequence[str] = ()
    pairs: Sequence[str] = ()
    poll_timeout_s: float = 0.0
    poll_interval_s: float = 5.0
    head_ssh_host: Optional[str] = None
    head_ssh_user: Optional[str] = None
    head_ssh_key: Optional[str] = None


@dataclass(frozen=True)
class RayNodeView:
    selected_ray_address: str
    alive: bool
    resources: Mapping[str, float]


@dataclass(frozen=True)
class ValidationRow:
    label: str
    ssh_host: str
    selected_ray_address: str
    status: str
    reason: str


class ClusterValidationError(RuntimeError):
    pass


def build_expected_cluster_nodes(
    public_nodes: Sequence[NodeSpec],
    local_nodes: Sequence[NodeSpec],
    selected_ray_addresses: Optional[Sequence[str]] = None,
) -> list[ExpectedClusterNode]:
    expected: list[ExpectedClusterNode] = []
    for index, (public_node, local_node) in enumerate(zip(public_nodes, local_nodes)):
        label = "head" if index == 0 else f"worker-{index}"
        selected_address = (
            selected_ray_addresses[index]
            if selected_ray_addresses is not None and index < len(selected_ray_addresses)
            else public_node.ip
        )
        expected.append(
            ExpectedClusterNode(
                label=label,
                ssh_host=public_node.ip,
                selected_ray_address=selected_address,
                expected_gpus=_expected_gpu_count(public_node.devices),
                match_addresses=tuple(
                    _dedupe([selected_address, public_node.ip, local_node.ip])
                ),
            )
        )
    return expected


def validate_cluster_before_submission(config: ClusterValidationConfig) -> None:
    deadline = time.monotonic() + max(0.0, config.poll_timeout_s)
    last_rows: list[ValidationRow] = []
    last_error: Optional[ClusterValidationError] = None
    while True:
        try:
            snapshot = query_ray_cluster_snapshot(
                config.config_path,
                head_ssh_host=config.head_ssh_host,
                head_ssh_user=config.head_ssh_user,
                head_ssh_key=config.head_ssh_key,
            )
        except ClusterValidationError as exc:
            last_error = exc
            last_rows = []
        else:
            last_error = None
            last_rows = _validate_snapshot_payload(config, snapshot)
            if not last_rows:
                return

        if time.monotonic() >= deadline:
            break
        if config.poll_interval_s > 0:
            time.sleep(config.poll_interval_s)

    if last_error is not None:
        raise last_error
    message = "Ray cluster validation failed before job submission:\n" + format_validation_table(
        last_rows
    )
    logger.error(message)
    raise ClusterValidationError(message)


def _validate_snapshot_payload(
    config: ClusterValidationConfig,
    snapshot: Mapping[str, Any],
) -> list[ValidationRow]:
    nodes = [
        RayNodeView(
            selected_ray_address=str(
                node.get("NodeManagerAddress") or node.get("node_manager_address") or ""
            ),
            alive=bool(node.get("Alive") if "Alive" in node else node.get("alive")),
            resources=_float_mapping(node.get("Resources") or node.get("resources") or {}),
        )
        for node in snapshot.get("nodes", [])
    ]
    resources = _float_mapping(snapshot.get("cluster_resources") or {})
    status_text = str(snapshot.get("ray_status_stdout") or "")
    if snapshot.get("ray_status_stderr"):
        status_text = f"{status_text}\n{snapshot['ray_status_stderr']}"

    result = validate_cluster_snapshot(
        config=config,
        ray_nodes=nodes,
        cluster_resources=resources,
        ray_status_text=status_text,
    )
    if not result:
        logger.info(
            "Ray cluster validation passed: %d active nodes, %.3g GPUs, %.3g exclusive resources.",
            len([node for node in nodes if node.alive]),
            resources.get("GPU", 0.0),
            resources.get("exclusive", 0.0),
        )
    return result


def validate_cluster_snapshot(
    *,
    config: ClusterValidationConfig,
    ray_nodes: Sequence[RayNodeView],
    cluster_resources: Mapping[str, float],
    ray_status_text: str = "",
) -> list[ValidationRow]:
    active_nodes_by_address = {
        node.selected_ray_address: node
        for node in ray_nodes
        if node.alive and node.selected_ray_address
    }
    all_nodes_by_address = {
        node.selected_ray_address: node
        for node in ray_nodes
        if node.selected_ray_address
    }
    status_text_lower = ray_status_text.lower()
    autoscaler_failures = _autoscaler_failures(ray_status_text)
    rows: list[ValidationRow] = []

    for expected in config.expected_nodes:
        node = _find_expected_node(expected, all_nodes_by_address)
        reasons: list[str] = []
        status = "active"
        if node is None:
            status = _status_for_missing_node(expected, status_text_lower)
            reasons.append("expected Ray node is not active")
        elif not node.alive:
            status = _status_for_missing_node(expected, status_text_lower)
            reasons.append("Ray node is registered but not alive")
        else:
            actual_gpus = float(node.resources.get("GPU", 0.0))
            if actual_gpus + 1e-9 < expected.expected_gpus:
                reasons.append(
                    f"GPU resources {actual_gpus:g} < expected {expected.expected_gpus:g}"
                )
            actual_exclusive = float(node.resources.get("exclusive", 0.0))
            if actual_exclusive + 1e-9 < 1.0:
                reasons.append(f"exclusive resources {actual_exclusive:g} < expected 1")

        node_text = _node_status_lines(expected, ray_status_text)
        if node_text:
            reasons.extend(node_text)
        if autoscaler_failures:
            reasons.extend(autoscaler_failures)

        if reasons:
            rows.append(
                ValidationRow(
                    label=expected.label,
                    ssh_host=expected.ssh_host,
                    selected_ray_address=expected.selected_ray_address,
                    status=status,
                    reason="; ".join(_dedupe(reasons)),
                )
            )

    active_count = len(active_nodes_by_address)
    expected_count = len(config.expected_nodes)
    total_expected_gpus = sum(node.expected_gpus for node in config.expected_nodes)
    total_expected_exclusive = float(expected_count)
    total_gpus = float(cluster_resources.get("GPU", 0.0))
    total_exclusive = float(cluster_resources.get("exclusive", 0.0))

    cluster_reasons: list[str] = []
    if active_count < expected_count:
        cluster_reasons.append(f"active Ray nodes {active_count} < expected {expected_count}")
    if total_gpus + 1e-9 < total_expected_gpus:
        cluster_reasons.append(f"total GPUs {total_gpus:g} < expected {total_expected_gpus:g}")
    if total_exclusive + 1e-9 < total_expected_exclusive:
        cluster_reasons.append(
            f"total exclusive resources {total_exclusive:g} < expected {total_expected_exclusive:g}"
        )
    placement_reason = _placement_failure_reason(
        config=config,
        active_nodes=list(active_nodes_by_address.values()),
        cluster_resources=cluster_resources,
    )
    if placement_reason:
        cluster_reasons.append(placement_reason)

    if cluster_reasons:
        rows.append(
            ValidationRow(
                label="cluster",
                ssh_host="-",
                selected_ray_address="-",
                status="invalid",
                reason="; ".join(cluster_reasons),
            )
        )

    return rows


def format_validation_table(rows: Sequence[ValidationRow]) -> str:
    headers = ["node", "ssh host", "selected Ray address", "status", "failure reason"]
    table = [
        [
            row.label,
            row.ssh_host,
            row.selected_ray_address,
            row.status,
            row.reason,
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in table))
        for column in range(len(headers))
    ]
    lines = [
        "  ".join(headers[column].ljust(widths[column]) for column in range(len(headers))),
        "  ".join("-" * widths[column] for column in range(len(headers))),
    ]
    for row in table:
        lines.append("  ".join(row[column].ljust(widths[column]) for column in range(len(row))))
    return "\n".join(lines)


def query_ray_cluster_snapshot(
    config_path: str,
    *,
    head_ssh_host: Optional[str] = None,
    head_ssh_user: Optional[str] = None,
    head_ssh_key: Optional[str] = None,
) -> dict[str, Any]:
    remote_script = _validation_remote_script()
    command = [
        "ray",
        "exec",
        str(config_path),
        "python - <<'PY'\n" + remote_script + "\nPY",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    combined = f"{result.stdout}\n{result.stderr}"
    payload = _extract_validation_payload(combined)
    if result.returncode != 0 and payload is None:
        fallback_payload = query_ray_cluster_snapshot_via_head_ssh(
            config_path,
            head_ssh_host=head_ssh_host,
            head_ssh_user=head_ssh_user,
            head_ssh_key=head_ssh_key,
        )
        if fallback_payload is not None:
            return fallback_payload
        raise ClusterValidationError(
            "Unable to query Ray cluster before job submission. "
            f"`ray exec` exited with code {result.returncode}."
        )
    if payload is None:
        fallback_payload = query_ray_cluster_snapshot_via_head_ssh(
            config_path,
            head_ssh_host=head_ssh_host,
            head_ssh_user=head_ssh_user,
            head_ssh_key=head_ssh_key,
        )
        if fallback_payload is not None:
            return fallback_payload
        raise ClusterValidationError("Unable to parse Ray cluster validation response.")
    if payload.get("query_error"):
        raise ClusterValidationError(f"Unable to query Ray cluster: {payload['query_error']}")
    return payload


def query_ray_cluster_snapshot_via_head_ssh(
    config_path: str,
    *,
    head_ssh_host: Optional[str] = None,
    head_ssh_user: Optional[str] = None,
    head_ssh_key: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    cluster_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    provider = cluster_config.get("provider") or {}
    auth = cluster_config.get("auth") or {}
    head_ip = str(head_ssh_host or provider.get("head_ip") or provider.get("external_head_ip") or "")
    ssh_user = str(head_ssh_user or auth.get("ssh_user") or "")
    ssh_key = str(head_ssh_key or auth.get("ssh_private_key") or "")
    if not head_ip or not ssh_user:
        return None

    remote_script = _validation_remote_script()
    docker_config = cluster_config.get("docker") or {}
    container_name = str(docker_config.get("container_name") or "")
    if container_name:
        remote_command = (
            f"sudo docker exec -i {shlex.quote(container_name)} "
            f"python - <<'PY'\n{remote_script}\nPY"
        )
    else:
        remote_command = f"python - <<'PY'\n{remote_script}\nPY"

    ssh_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]
    if ssh_key:
        ssh_command.extend(["-i", str(Path(ssh_key).expanduser())])
    ssh_command.extend([f"{ssh_user}@{head_ip}", remote_command])
    result = subprocess.run(ssh_command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    payload = _extract_validation_payload(f"{result.stdout}\n{result.stderr}")
    if payload is None or payload.get("query_error"):
        return None
    return payload


def _validation_remote_script() -> str:
    return r"""
import json
import subprocess
import sys

payload = {}
try:
    status = subprocess.run(
        [sys.executable, "-m", "ray.scripts.scripts", "status"],
        capture_output=True,
        text=True,
        check=False,
    )
except Exception as exc:
    payload["ray_status_returncode"] = -1
    payload["ray_status_stdout"] = ""
    payload["ray_status_stderr"] = f"{type(exc).__name__}: {exc}"
else:
    payload["ray_status_returncode"] = status.returncode
    payload["ray_status_stdout"] = status.stdout
    payload["ray_status_stderr"] = status.stderr
try:
    import ray

    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    payload["nodes"] = ray.nodes()
    payload["cluster_resources"] = ray.cluster_resources()
    payload["available_resources"] = ray.available_resources()
except Exception as exc:
    payload["query_error"] = f"{type(exc).__name__}: {exc}"
print("__NOBOOM_CLUSTER_VALIDATION_JSON_START__")
print(json.dumps(payload, default=str))
print("__NOBOOM_CLUSTER_VALIDATION_JSON_END__")
"""


def _extract_validation_payload(output: str) -> Optional[dict[str, Any]]:
    match = re.search(
        r"__NOBOOM_CLUSTER_VALIDATION_JSON_START__\s*(\{.*?\})\s*__NOBOOM_CLUSTER_VALIDATION_JSON_END__",
        output,
        flags=re.DOTALL,
    )
    if not match:
        return None
    return json.loads(match.group(1))


def _expected_gpu_count(devices: Optional[str]) -> float:
    if devices is None or not devices.strip():
        return 0.0
    selected = [part.strip() for part in devices.split(",") if part.strip()]
    if len(selected) == 1 and selected[0].lower() in {"none", "no", "false"}:
        return 0.0
    return float(len(selected))


def _float_mapping(raw: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _find_expected_node(
    expected: ExpectedClusterNode,
    nodes_by_address: Mapping[str, RayNodeView],
) -> Optional[RayNodeView]:
    for address in _candidate_addresses(expected):
        node = nodes_by_address.get(address)
        if node is not None:
            return node
    return None


def _status_for_missing_node(expected: ExpectedClusterNode, status_text_lower: str) -> str:
    for selected_address in _candidate_addresses(expected):
        if selected_address.lower() in status_text_lower:
            if "pending" in status_text_lower:
                return "pending"
            if "uninitialized" in status_text_lower:
                return "pending"
    return "missing"


def _node_status_lines(expected: ExpectedClusterNode, ray_status_text: str) -> list[str]:
    messages: list[str] = []
    addresses = _candidate_addresses(expected)
    for line in ray_status_text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if any(address in stripped for address in addresses) and (
            "pending" in lowered or "uninitialized" in lowered
        ):
            messages.append(stripped)
    return messages


def _candidate_addresses(expected: ExpectedClusterNode) -> list[str]:
    return _dedupe([expected.selected_ray_address, *expected.match_addresses])


def _autoscaler_failures(ray_status_text: str) -> list[str]:
    failures: list[str] = []
    for line in ray_status_text.splitlines():
        stripped = line.strip()
        if AUTOSCALER_UNSATISFIABLE_PHRASE in stripped:
            failures.append(stripped)
    return failures


def _placement_failure_reason(
    *,
    config: ClusterValidationConfig,
    active_nodes: Sequence[RayNodeView],
    cluster_resources: Mapping[str, float],
) -> Optional[str]:
    demands = _placement_demands(config)
    if not demands:
        return None

    for demand in demands:
        if not any(_node_satisfies(node, demand) for node in active_nodes):
            return f"no active node can satisfy placement bundle {_format_resources(demand)}"

    concurrent_count = min(max(1, config.max_in_flight), len(demands))
    largest_gpu = max(demand.get("GPU", 0.0) for demand in demands)
    largest_exclusive = max(demand.get("exclusive", 0.0) for demand in demands)
    aggregate = {
        "GPU": largest_gpu * concurrent_count,
        "exclusive": largest_exclusive * concurrent_count,
    }
    missing = {
        key: value
        for key, value in aggregate.items()
        if value > 0 and float(cluster_resources.get(key, 0.0)) + 1e-9 < value
    }
    if missing:
        return (
            "cluster resources cannot satisfy "
            f"{concurrent_count} in-flight placement bundles {_format_resources(missing)}"
        )
    return None


def _placement_demands(config: ClusterValidationConfig) -> list[dict[str, float]]:
    pairs = _dataset_model_pairs(config.datasets, config.models, config.pairs)
    if not pairs:
        pairs = [("", "")]
    return [
        {
            "GPU": _gpu_request_for_pair(dataset, model, config.gpus_per_run),
            "exclusive": _exclusive_request_for_pair(dataset, model),
        }
        for dataset, model in pairs
    ]


def _dataset_model_pairs(
    datasets: Sequence[str],
    models: Sequence[str],
    pairs: Sequence[str],
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for dataset in datasets:
        for model in models:
            item = (dataset.lower(), model.lower())
            if item not in seen:
                normalized.append(item)
                seen.add(item)
    for raw_pair in pairs:
        parts = raw_pair.split(":", 1)
        if len(parts) != 2:
            continue
        item = (parts[0].strip().lower(), parts[1].strip().lower())
        if item[0] and item[1] and item not in seen:
            normalized.append(item)
            seen.add(item)
    return normalized


def _gpu_request_for_pair(dataset: str, model: str, requested_gpu: float) -> float:
    if not model or model in {"eif", "gmmhmm", "hbos", "hmm", "kmeans", "ocsvm", "pca", "threshold"}:
        return 0.0 if model else requested_gpu
    if requested_gpu >= 1:
        return float(int(requested_gpu + 0.999999))
    if "physdiff" in model or "timesnet" in model:
        return max(requested_gpu, 0.5)
    if "industry_process" in dataset and "neutralad" in model:
        return 1.0
    if "industry_process" in dataset:
        return min(requested_gpu * 4, 0.5)
    if "lstm" in model:
        return min(requested_gpu * 2, 0.3)
    if "neutralad" in model:
        return min(requested_gpu * 2, 0.5)
    return requested_gpu


def _exclusive_request_for_pair(dataset: str, model: str) -> float:
    if "industry_process" in dataset and model in {"gmmhmm", "hmm"}:
        return 1.0
    if "industry_process" in dataset and "eif" in model:
        return 0.5
    return 0.001


def _node_satisfies(node: RayNodeView, demand: Mapping[str, float]) -> bool:
    return all(float(node.resources.get(key, 0.0)) + 1e-9 >= value for key, value in demand.items())


def _format_resources(resources: Mapping[str, float]) -> str:
    payload = ", ".join(f"{key}={value:g}" for key, value in sorted(resources.items()))
    return "{" + payload + "}"


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = " ".join(value.split())
        if compact and compact not in seen:
            result.append(compact)
            seen.add(compact)
    return result


def validation_command_preview(config_path: str) -> str:
    """Return a short diagnostic command without embedding environment values."""
    return " ".join(["ray", "exec", shlex.quote(config_path), shlex.quote("ray status")])
