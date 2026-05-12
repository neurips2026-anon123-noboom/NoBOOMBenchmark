"""Ray node RAM guard for per-pair jobs."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Dict, Mapping, Optional, Protocol

try:
    import ray
except ModuleNotFoundError:  # pragma: no cover - Ray-free unit environments
    ray = None  # type: ignore[assignment]

try:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
except ModuleNotFoundError:  # pragma: no cover - Ray-free unit environments
    NodeAffinitySchedulingStrategy = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PAIR_JOB_ROLE = "pair"
PAIR_JOB_ROLE_ENV = "NOBOOM_JOB_ROLE"
PAIR_JOB_ROLE_ENV_ALIASES = ("NOBOOM_RAY_JOB_ROLE",)


class PairRamGuard(Protocol):
    def stop_hot_pending_jobs(self, pending_pair_specs: Mapping[str, Any]) -> list[Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class RamUsage:
    node_id: str
    used_fraction: float
    total_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class NodeRamGuardEvent:
    node_id: str
    used_fraction: float
    resume_used_fraction: float
    total_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class RamGuardStopEvent:
    node_id: str
    submission_id: str
    used_fraction: float
    pair_id: Optional[str] = None


def parse_linux_meminfo(text: str) -> tuple[int, int]:
    values: Dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        parts = raw_value.strip().split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        multiplier = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
        values[key] = value * multiplier
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    return total, available


def memory_used_fraction(total_bytes: int, available_bytes: int) -> float:
    if total_bytes <= 0:
        return 0.0
    used = max(0, total_bytes - max(0, available_bytes))
    return min(1.0, used / total_bytes)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "on", "true", "yes"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    return float(value)


def _is_terminal_status(status: Any) -> bool:
    is_terminal = getattr(status, "is_terminal", None)
    if callable(is_terminal):
        try:
            return bool(is_terminal())
        except Exception:  # noqa: BLE001
            return False
    return str(status).upper() in {"SUCCEEDED", "FAILED", "STOPPED"}


def _job_submission_id(job_info: Any) -> Optional[str]:
    value = getattr(job_info, "submission_id", None)
    if value:
        return str(value)
    metadata = getattr(job_info, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("submission_id")
        if value:
            return str(value)
    return None


def _job_node_id(job_info: Any) -> Optional[str]:
    for attr in ("node_id", "driver_node_id"):
        value = getattr(job_info, attr, None)
        if value:
            return str(value)
    metadata = getattr(job_info, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("node_id", "driver_node_id", "ray_node_id"):
            value = metadata.get(key)
            if value:
                return str(value)
    return None


class RayNodeRamGuard:
    def __init__(
        self,
        *,
        client: Any,
        enabled: bool = True,
        max_used_fraction: float = 0.95,
        resume_used_fraction: float = 0.90,
        poll_interval_s: float = 10.0,
        cooldown_s: float = 120.0,
        notifier: Optional[Any] = None,
        clock: Optional[Any] = None,
    ) -> None:
        self._client = client
        self._enabled = enabled
        self._max_used_fraction = max_used_fraction
        self._resume_used_fraction = resume_used_fraction
        self._poll_interval_s = poll_interval_s
        self._cooldown_s = cooldown_s
        self._notifier = notifier
        self._clock = clock or time.monotonic
        self._last_poll = 0.0
        self._last_stop_by_submission: Dict[str, float] = {}
        self._hot_nodes: Dict[str, RamUsage] = {}

    @classmethod
    def from_env(cls, *, client: Any, notifier: Optional[Any] = None) -> Optional["RayNodeRamGuard"]:
        enabled = _env_bool("NOBOOM_RAM_GUARD_ENABLED", True)
        if _env_bool("NOBOOM_RAM_GUARD_DISABLED", False):
            enabled = False
        if not enabled:
            return None
        return cls(
            client=client,
            enabled=enabled,
            max_used_fraction=_env_float("NOBOOM_RAM_GUARD_MAX_USED_FRACTION", 0.95),
            resume_used_fraction=_env_float("NOBOOM_RAM_GUARD_RESUME_USED_FRACTION", 0.90),
            poll_interval_s=_env_float("NOBOOM_RAM_GUARD_POLL_INTERVAL_S", 10.0),
            cooldown_s=_env_float("NOBOOM_RAM_GUARD_COOLDOWN_S", 120.0),
            notifier=notifier,
        )

    def stop_hot_pending_jobs(self, pending_pair_specs: Mapping[str, Any]) -> list[RamGuardStopEvent]:
        if not self._enabled or not pending_pair_specs:
            return []
        now = float(self._clock())
        if self._last_poll and now - self._last_poll < self._poll_interval_s:
            return []
        self._last_poll = now
        hot_nodes = self.hot_nodes()
        if not hot_nodes:
            return []
        stopped: list[RamGuardStopEvent] = []
        jobs_by_submission = self._pending_jobs_by_submission(pending_pair_specs)
        for submission_id, job_info in jobs_by_submission.items():
            if submission_id not in pending_pair_specs:
                continue
            node_id = _job_node_id(job_info)
            if not node_id or node_id not in hot_nodes:
                continue
            last_stop = self._last_stop_by_submission.get(submission_id)
            if last_stop is not None and now - last_stop < self._cooldown_s:
                continue
            try:
                self._client.stop_job(submission_id)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to stop pair job %s after RAM breach.", submission_id, exc_info=True)
                continue
            self._last_stop_by_submission[submission_id] = now
            pair_spec = pending_pair_specs[submission_id]
            event = RamGuardStopEvent(
                node_id=node_id,
                submission_id=submission_id,
                used_fraction=hot_nodes[node_id].used_fraction,
                pair_id=getattr(pair_spec, "pair_id", None),
            )
            stopped.append(event)
            self._notify(event)
        return stopped

    def hot_nodes(self) -> Dict[str, RamUsage]:
        usages = self.probe_node_memory()
        hot: Dict[str, RamUsage] = {}
        for usage in usages:
            if usage.used_fraction >= self._max_used_fraction:
                hot[usage.node_id] = usage
                if usage.node_id not in self._hot_nodes:
                    self._notify(
                        NodeRamGuardEvent(
                            node_id=usage.node_id,
                            used_fraction=usage.used_fraction,
                            resume_used_fraction=self._resume_used_fraction,
                            total_bytes=usage.total_bytes,
                            available_bytes=usage.available_bytes,
                        )
                    )
            elif usage.used_fraction <= self._resume_used_fraction:
                self._hot_nodes.pop(usage.node_id, None)
        self._hot_nodes.update(hot)
        return hot

    def probe_node_memory(self) -> list[RamUsage]:
        if ray is None:
            return []
        try:
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
            nodes = [node for node in ray.nodes() if node.get("Alive", True)]
        except Exception:  # noqa: BLE001
            logger.debug("Unable to probe Ray nodes for RAM guard.", exc_info=True)
            return []
        usages: list[RamUsage] = []
        for node in nodes:
            node_id = str(node.get("NodeID") or node.get("node_id") or "")
            if not node_id:
                continue
            usage = self._probe_single_node(node_id)
            if usage is not None:
                usages.append(usage)
        return usages

    def _probe_single_node(self, node_id: str) -> Optional[RamUsage]:
        if ray is None or NodeAffinitySchedulingStrategy is None:
            return None

        @ray.remote(num_cpus=0)
        def _read_meminfo() -> tuple[int, int]:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                return parse_linux_meminfo(handle.read())

        try:
            strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            total, available = ray.get(_read_meminfo.options(scheduling_strategy=strategy).remote())
        except Exception:  # noqa: BLE001
            logger.debug("Unable to probe RAM on Ray node %s.", node_id, exc_info=True)
            return None
        return RamUsage(
            node_id=node_id,
            used_fraction=memory_used_fraction(total, available),
            total_bytes=total,
            available_bytes=available,
        )

    def _pending_jobs_by_submission(self, pending_pair_specs: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            jobs = self._client.list_jobs()
        except Exception:  # noqa: BLE001
            logger.debug("Unable to list Ray jobs for RAM guard.", exc_info=True)
            return {}
        pending: Dict[str, Any] = {}
        for job_info in jobs:
            submission_id = _job_submission_id(job_info)
            if submission_id not in pending_pair_specs:
                continue
            status = getattr(job_info, "status", None)
            if status is not None and _is_terminal_status(status):
                continue
            pending[submission_id] = job_info
        return pending

    def _notify(self, event: Any) -> None:
        if self._notifier is None:
            return
        notify = getattr(self._notifier, "notify", None)
        if callable(notify):
            try:
                notify(event)
            except Exception:  # noqa: BLE001
                logger.debug("RAM guard notifier failed.", exc_info=True)
