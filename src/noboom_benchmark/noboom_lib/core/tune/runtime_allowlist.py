from __future__ import annotations

from pathlib import Path
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Set, TYPE_CHECKING

import ray
from ray.util import state as ray_state
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment import Trial
from ray.tune.schedulers import TrialScheduler
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from noboom_cluster.noboom_cli_lib.allowlist import (
    AllowlistConfig,
    AllowlistStatus,
    AllowlistUpdateResponse,
    ClusterNodeGpuView,
    TrialGpuLease,
)

from .tune_helpers import RAY_NAMESPACE

if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController

logger = logging.getLogger(__name__)

RUNTIME_ALLOWLIST_ACTOR_NAME = "runtime_gpu_allowlist"
LEASE_CONFIG_KEY = "__noboom_gpu_lease"
LEASE_STATE_RESERVED = "reserved"
LEASE_STATE_ACTIVE = "active"
NODE_PIN_RESOURCE_QUANTITY = 0.001
TERMINAL_JOB_STATUSES = frozenset({"STOPPED", "SUCCEEDED", "FAILED"})


def _discover_visible_gpu_ids() -> List[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip() != "":
        return [item.strip() for item in visible_devices.split(",") if item.strip()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _discover_local_ip() -> str:
    result = subprocess.run(
        ["bash", "-lc", "ip route get 1.1.1.1 | awk '{for (i=1;i<=NF;i++) if ($i==\"src\") print $(i+1)}'"],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _current_job_id() -> Optional[str]:
    try:
        return str(ray.get_runtime_context().get_job_id())
    except Exception:  # noqa: BLE001
        logger.exception("Failed to resolve the current Ray job id.")
        return None


def _list_live_job_ids() -> Optional[Set[str]]:
    try:
        jobs = ray_state.list_jobs(detail=True, raise_on_missing_output=False)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to list Ray jobs for allowlist lease cleanup.")
        return None

    live_job_ids: Set[str] = set()
    for job in jobs:
        job_id = getattr(job, "job_id", None)
        status = str(getattr(job, "status", "") or "")
        if not job_id or status in TERMINAL_JOB_STATUSES:
            continue
        live_job_ids.add(str(job_id))
    return live_job_ids


@ray.remote
def _probe_cluster_node() -> ClusterNodeGpuView:
    return ClusterNodeGpuView(
        ip=_discover_local_ip(),
        node_id=ray.get_runtime_context().get_node_id(),
        visible_gpu_ids=_discover_visible_gpu_ids(),
    )


class RuntimeGpuAllowlistState:
    def __init__(self) -> None:
        self._cluster_nodes: Dict[str, ClusterNodeGpuView] = {}
        self._allowed: Dict[str, List[str]] = {}
        self._active_leases: Dict[str, TrialGpuLease] = {}
        self._revision = 0

    def initialize(
        self,
        cluster_nodes: List[ClusterNodeGpuView],
        initial_allowlist: AllowlistConfig,
    ) -> AllowlistStatus:
        self._cluster_nodes = {node.ip: node for node in cluster_nodes}
        if not self._allowed:
            expanded, errors = self._validate_and_expand_allowlist(initial_allowlist)
            if errors:
                raise RuntimeError("Invalid startup allowlist: " + "; ".join(errors))
            self._allowed = expanded
            self._revision = 1
        else:
            self._drop_invalid_reserved_leases()
        self._save_state()
        return self.get_status()

    def acquire_lease(
        self,
        trial_id: str,
        gpu_fraction: float,
        owner_job_id: Optional[str] = None,
    ) -> Optional[TrialGpuLease]:
        if gpu_fraction <= 0:
            return None

        self._prune_dead_job_leases()

        existing = self._active_leases.get(trial_id)
        if existing is not None:
            if existing.state == LEASE_STATE_RESERVED and not self._is_lease_allowed(existing):
                self._active_leases.pop(trial_id, None)
            else:
                return existing

        candidate = self._choose_lease_candidate(gpu_fraction)
        if candidate is None:
            return None

        lease = TrialGpuLease(
            trial_id=trial_id,
            ip=candidate["ip"],
            node_id=candidate["node_id"],
            gpu_id=candidate["gpu_id"],
            gpu_fraction=gpu_fraction,
            revision=self._revision,
            state=LEASE_STATE_RESERVED,
            owner_job_id=owner_job_id,
        )
        self._active_leases[trial_id] = lease
        self._save_state()
        return lease

    def activate_lease(self, trial_id: str) -> Optional[TrialGpuLease]:
        lease = self._active_leases.get(trial_id)
        if lease is None:
            return None
        if not self._is_lease_allowed(lease):
            self._active_leases.pop(trial_id, None)
            self._save_state()
            return None
        if lease.state != LEASE_STATE_ACTIVE:
            lease.state = LEASE_STATE_ACTIVE
            self._save_state()
        return lease

    def release_lease(self, trial_id: str) -> Optional[TrialGpuLease]:
        lease = self._active_leases.pop(trial_id, None)
        if lease is not None:
            self._save_state()
        return lease

    def prune_dead_leases(
        self,
        *,
        live_trial_ids: Optional[Sequence[str]] = None,
        drop_ownerless: bool = False,
    ) -> AllowlistStatus:
        self._prune_dead_job_leases(
            live_trial_ids=live_trial_ids,
            drop_ownerless=drop_ownerless,
        )
        self._save_state()
        return self.get_status()

    def apply_allowlist(self, allowlist: AllowlistConfig) -> AllowlistUpdateResponse:
        if not self._cluster_nodes:
            return AllowlistUpdateResponse(
                accepted=False,
                errors=["Runtime GPU allowlist actor has not been initialized."],
            )

        expanded, errors = self._validate_and_expand_allowlist(allowlist)
        if errors:
            return AllowlistUpdateResponse(
                accepted=False,
                status=self.get_status(),
                errors=errors,
            )

        changed = expanded != self._allowed
        self._allowed = expanded
        if changed:
            self._revision += 1
        self._drop_invalid_reserved_leases()
        self._save_state()
        return AllowlistUpdateResponse(
            accepted=True,
            status=self.get_status(),
        )

    def get_status(self) -> AllowlistStatus:
        cluster_nodes = sorted(self._cluster_nodes.values(), key=lambda item: item.ip)
        active_leases = sorted(self._active_leases.values(), key=lambda item: item.trial_id)
        draining_leases = [lease for lease in active_leases if not self._is_lease_allowed(lease)]
        return AllowlistStatus(
            revision=self._revision,
            cluster_nodes=cluster_nodes,
            allowed={ip: list(gpu_ids) for ip, gpu_ids in sorted(self._allowed.items())},
            active_leases=active_leases,
            draining_leases=draining_leases,
        )

    def _validate_and_expand_allowlist(
        self,
        allowlist: AllowlistConfig,
    ) -> tuple[Dict[str, List[str]], List[str]]:
        expanded: Dict[str, List[str]] = {}
        errors: List[str] = []
        seen_ips: set[str] = set()

        for node in allowlist.nodes:
            if node.ip in seen_ips:
                errors.append(f"Duplicate node entry for {node.ip}.")
                continue
            seen_ips.add(node.ip)

            cluster_node = self._cluster_nodes.get(node.ip)
            if cluster_node is None:
                errors.append(f"Unknown node IP: {node.ip}.")
                continue

            requested_devices = node.expanded_devices(cluster_node.visible_gpu_ids)
            unknown_devices = [
                device_id
                for device_id in requested_devices
                if device_id not in cluster_node.visible_gpu_ids
            ]
            if unknown_devices:
                errors.append(
                    f"Node {node.ip} requested unknown GPU IDs: {', '.join(sorted(set(unknown_devices)))}.",
                )
                continue

            deduped_devices: List[str] = []
            for device_id in requested_devices:
                if device_id not in deduped_devices:
                    deduped_devices.append(device_id)
            expanded[node.ip] = deduped_devices

        return expanded, errors

    def _choose_lease_candidate(self, gpu_fraction: float) -> Optional[Dict[str, Any]]:
        candidates: List[tuple[float, str, str, str]] = []
        usage_by_gpu = self._usage_by_gpu()

        for ip, allowed_gpu_ids in self._allowed.items():
            cluster_node = self._cluster_nodes.get(ip)
            if cluster_node is None:
                continue
            for gpu_id in allowed_gpu_ids:
                current_usage = usage_by_gpu.get((ip, gpu_id), 0.0)
                if current_usage + gpu_fraction <= 1.000001:
                    candidates.append((current_usage, ip, gpu_id, cluster_node.node_id))

        if not candidates:
            return None

        _, ip, gpu_id, node_id = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        return {
            "ip": ip,
            "gpu_id": gpu_id,
            "node_id": node_id,
        }

    def _usage_by_gpu(self) -> Dict[tuple[str, str], float]:
        usage: Dict[tuple[str, str], float] = {}
        for lease in self._active_leases.values():
            usage[(lease.ip, lease.gpu_id)] = usage.get((lease.ip, lease.gpu_id), 0.0) + lease.gpu_fraction
        return usage

    def _is_lease_allowed(self, lease: TrialGpuLease) -> bool:
        allowed_gpu_ids = self._allowed.get(lease.ip)
        if allowed_gpu_ids is None:
            return False
        return lease.gpu_id in allowed_gpu_ids

    def _drop_invalid_reserved_leases(self) -> None:
        for trial_id, lease in list(self._active_leases.items()):
            if lease.state == LEASE_STATE_RESERVED and not self._is_lease_allowed(lease):
                self._active_leases.pop(trial_id, None)

    def _prune_dead_job_leases(
        self,
        *,
        live_trial_ids: Optional[Sequence[str]] = None,
        drop_ownerless: bool = False,
    ) -> None:
        live_job_ids = _list_live_job_ids()
        live_trial_id_set = {str(trial_id) for trial_id in live_trial_ids or []}
        changed = False

        for trial_id, lease in list(self._active_leases.items()):
            if lease.trial_id in live_trial_id_set:
                continue

            if lease.owner_job_id:
                if live_job_ids is None or lease.owner_job_id in live_job_ids:
                    continue
                self._active_leases.pop(trial_id, None)
                changed = True
                continue

            if drop_ownerless:
                self._active_leases.pop(trial_id, None)
                changed = True

        if changed:
            self._save_state()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "cluster_nodes": [
                node.model_dump(mode="json")
                for node in sorted(self._cluster_nodes.values(), key=lambda item: item.ip)
            ],
            "allowed": {ip: list(gpu_ids) for ip, gpu_ids in sorted(self._allowed.items())},
            "active_leases": [
                lease.model_dump(mode="json")
                for lease in sorted(self._active_leases.values(), key=lambda item: item.trial_id)
            ],
            "revision": self._revision,
        }

    def load_payload(self, payload: Dict[str, Any]) -> None:
        self._cluster_nodes = {
            node["ip"]: ClusterNodeGpuView.model_validate(node)
            for node in payload.get("cluster_nodes", [])
        }
        self._allowed = {
            str(ip): list(gpu_ids)
            for ip, gpu_ids in payload.get("allowed", {}).items()
        }
        self._active_leases = {
            lease["trial_id"]: TrialGpuLease.model_validate(lease)
            for lease in payload.get("active_leases", [])
        }
        self._revision = int(payload.get("revision", 0))

    def _restore_state(self) -> None:
        return None

    def _save_state(self) -> None:
        return None


@ray.remote(label_selector={"role": "head"}, max_restarts=-1, max_task_retries=-1)
class RuntimeGpuAllowlistActor(RuntimeGpuAllowlistState):
    def __init__(self) -> None:
        super().__init__()
        self._restore_state()

    def _state_path(self) -> Optional[Path]:
        global_node = ray._private.worker._global_node
        if global_node is None:
            return None
        session_dir = Path(global_node.get_session_dir_path())
        return session_dir / "actor_ckpts" / "runtime_gpu_allowlist.json"

    def _restore_state(self) -> None:
        state_path = self._state_path()
        if state_path is None or not state_path.exists():
            return

        self.load_payload(json.loads(state_path.read_text(encoding="utf-8")))

    def _save_state(self) -> None:
        state_path = self._state_path()
        if state_path is None:
            return

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(self.to_payload(), sort_keys=True), encoding="utf-8")


def get_runtime_gpu_allowlist_actor() -> Any:
    return ray.get_actor(RUNTIME_ALLOWLIST_ACTOR_NAME, namespace=RAY_NAMESPACE)


def apply_cuda_visible_devices_for_lease(lease: TrialGpuLease) -> None:
    previous_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if previous_devices != lease.gpu_id:
        logger.info(
            "Applying GPU lease for trial '%s': CUDA_VISIBLE_DEVICES=%s (previous=%s).",
            lease.trial_id,
            lease.gpu_id,
            previous_devices,
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = lease.gpu_id


def ensure_runtime_gpu_allowlist_actor() -> Any:
    try:
        return get_runtime_gpu_allowlist_actor()
    except ValueError:
        return RuntimeGpuAllowlistActor.options(
            name=RUNTIME_ALLOWLIST_ACTOR_NAME,
            namespace=RAY_NAMESPACE,
            lifetime="detached",
        ).remote()


def discover_cluster_nodes() -> List[ClusterNodeGpuView]:
    refs = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_id = node.get("NodeID")
        if not node_id:
            continue
        refs.append(
            _probe_cluster_node.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote()
        )
    return sorted(ray.get(refs), key=lambda item: item.ip)


def initialize_runtime_gpu_allowlist(initial_allowlist: AllowlistConfig) -> AllowlistStatus:
    actor = ensure_runtime_gpu_allowlist_actor()
    return ray.get(actor.initialize.remote(discover_cluster_nodes(), initial_allowlist))


class LeaseAwareTrialScheduler(TrialScheduler):
    def __init__(
        self,
        base_scheduler: TrialScheduler,
        *,
        gpu_fraction: float,
        base_resources: Dict[str, float],
    ) -> None:
        super().__init__()
        self._base_scheduler = base_scheduler
        self._default_gpu_fraction = gpu_fraction
        self._default_base_resources = dict(base_resources)
        self._trial_gpu_fractions: Dict[str, float] = {}
        self._trial_base_resources: Dict[str, Dict[str, float]] = {}
        self._allowlist_actor = get_runtime_gpu_allowlist_actor()
        self._owner_job_id = _current_job_id()

    @property
    def metric(self) -> Optional[str]:
        return self._base_scheduler.metric

    @property
    def supports_buffered_results(self) -> bool:
        return self._base_scheduler.supports_buffered_results

    def set_search_properties(
        self,
        metric: Optional[str],
        mode: Optional[str],
        **spec: Any,
    ) -> bool:
        return self._base_scheduler.set_search_properties(metric, mode, **spec)

    def on_trial_add(self, tune_controller: "TuneController", trial: Trial) -> None:
        self._capture_trial_resource_request(trial)
        self._base_scheduler.on_trial_add(tune_controller, trial)

    def on_trial_error(self, tune_controller: "TuneController", trial: Trial) -> None:
        self._release_lease(trial.trial_id)
        self._base_scheduler.on_trial_error(tune_controller, trial)

    def on_trial_result(
        self,
        tune_controller: "TuneController",
        trial: Trial,
        result: Dict[str, Any],
    ) -> str:
        return self._base_scheduler.on_trial_result(tune_controller, trial, result)

    def on_trial_complete(
        self,
        tune_controller: "TuneController",
        trial: Trial,
        result: Dict[str, Any],
    ) -> None:
        self._release_lease(trial.trial_id)
        self._forget_trial_resource_request(trial.trial_id)
        self._base_scheduler.on_trial_complete(tune_controller, trial, result)

    def on_trial_remove(self, tune_controller: "TuneController", trial: Trial) -> None:
        self._release_lease(trial.trial_id)
        self._forget_trial_resource_request(trial.trial_id)
        self._base_scheduler.on_trial_remove(tune_controller, trial)

    def choose_trial_to_run(self, tune_controller: "TuneController") -> Optional[Trial]:
        trial = self._base_scheduler.choose_trial_to_run(tune_controller)
        if trial is None:
            return trial

        gpu_fraction = self._trial_gpu_fraction(trial)
        if gpu_fraction <= 0:
            return trial

        lease = ray.get(
            self._allowlist_actor.acquire_lease.remote(
                trial.trial_id,
                gpu_fraction,
                self._owner_job_id,
            )
        )
        if lease is None:
            return None

        self._apply_trial_lease(trial, lease)
        return trial

    def debug_string(self) -> str:
        return f"(LeaseAware) {self._base_scheduler.debug_string()}"

    def save(self, checkpoint_path: str) -> None:
        payload = {
            "base_scheduler": self._base_scheduler,
            "default_gpu_fraction": self._default_gpu_fraction,
            "default_base_resources": self._default_base_resources,
            "trial_gpu_fractions": self._trial_gpu_fractions,
            "trial_base_resources": self._trial_base_resources,
        }
        with Path(checkpoint_path).open("wb") as handle:
            import pickle

            pickle.dump(payload, handle)

    def restore(self, checkpoint_path: str) -> None:
        with Path(checkpoint_path).open("rb") as handle:
            import pickle

            payload = pickle.load(handle)
        self._base_scheduler = payload["base_scheduler"]
        self._default_gpu_fraction = float(payload["default_gpu_fraction"])
        self._default_base_resources = dict(payload["default_base_resources"])
        self._trial_gpu_fractions = {
            str(trial_id): float(gpu_fraction)
            for trial_id, gpu_fraction in payload.get("trial_gpu_fractions", {}).items()
        }
        self._trial_base_resources = {
            str(trial_id): {str(key): float(value) for key, value in resources.items()}
            for trial_id, resources in payload.get("trial_base_resources", {}).items()
        }
        self._allowlist_actor = get_runtime_gpu_allowlist_actor()

    def _apply_trial_lease(self, trial: Trial, lease: TrialGpuLease) -> None:
        trial.config[LEASE_CONFIG_KEY] = lease.model_dump(mode="json")
        trial.invalidate_json_state()
        pinned_resources = self._trial_base_resources_for(trial)
        pinned_resources[f"node:{lease.ip}"] = NODE_PIN_RESOURCE_QUANTITY
        placement_group = PlacementGroupFactory(
            [pinned_resources],
            strategy="STRICT_PACK",
        )
        trial.update_resources(placement_group)

    def _capture_trial_resource_request(self, trial: Trial) -> None:
        if trial.trial_id in self._trial_gpu_fractions and trial.trial_id in self._trial_base_resources:
            return

        trial_resources = self._extract_trial_resources(trial)
        lease_payload = trial.config.get(LEASE_CONFIG_KEY)
        gpu_fraction = float(trial_resources.pop("GPU", 0.0))
        if gpu_fraction <= 0 and isinstance(lease_payload, dict):
            gpu_fraction = float(lease_payload.get("gpu_fraction", 0.0))
        if gpu_fraction <= 0:
            gpu_fraction = self._default_gpu_fraction

        base_resources = {
            str(key): float(value)
            for key, value in trial_resources.items()
        }
        if not base_resources:
            base_resources = dict(self._default_base_resources)

        self._trial_gpu_fractions[trial.trial_id] = gpu_fraction
        self._trial_base_resources[trial.trial_id] = base_resources

    def _extract_trial_resources(self, trial: Trial) -> Dict[str, float]:
        placement_group = getattr(trial, "placement_group_factory", None)
        required_resources = getattr(placement_group, "required_resources", None)
        if isinstance(required_resources, dict):
            return {
                str(key): float(value)
                for key, value in required_resources.items()
            }

        trial_resources = dict(self._default_base_resources)
        if self._default_gpu_fraction > 0:
            trial_resources["GPU"] = self._default_gpu_fraction
        return trial_resources

    def _trial_gpu_fraction(self, trial: Trial) -> float:
        self._capture_trial_resource_request(trial)
        return self._trial_gpu_fractions.get(trial.trial_id, self._default_gpu_fraction)

    def _trial_base_resources_for(self, trial: Trial) -> Dict[str, float]:
        self._capture_trial_resource_request(trial)
        return dict(self._trial_base_resources.get(trial.trial_id, self._default_base_resources))

    def _forget_trial_resource_request(self, trial_id: str) -> None:
        self._trial_gpu_fractions.pop(trial_id, None)
        self._trial_base_resources.pop(trial_id, None)

    def _release_lease(self, trial_id: str) -> None:
        try:
            ray.get(self._allowlist_actor.release_lease.remote(trial_id))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to release GPU lease for trial %s", trial_id)


def build_lease_aware_tune_scheduler(
    base_scheduler: TrialScheduler,
    *,
    full_resources: Dict[str, Any],
    base_resources: Dict[str, Any],
) -> Optional[TrialScheduler]:
    if float(full_resources.get("GPU", 0.0)) <= 0:
        return None

    try:
        get_runtime_gpu_allowlist_actor()
    except ValueError:
        logger.warning("Runtime GPU allowlist actor is unavailable; falling back to default Ray GPU scheduling.")
        return None

    return LeaseAwareTrialScheduler(
        base_scheduler,
        gpu_fraction=float(full_resources.get("GPU", 0.0)),
        base_resources={key: float(value) for key, value in base_resources.items()},
    )
