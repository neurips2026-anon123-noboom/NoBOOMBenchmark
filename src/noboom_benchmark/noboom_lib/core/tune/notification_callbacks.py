"""Notification integration for pair, trial, and controller lifecycle events."""
from __future__ import annotations

from argparse import Namespace
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    from ray.tune.callback import Callback
    from ray.tune.experiment import Trial
except ModuleNotFoundError:  # pragma: no cover - Ray-free unit environments
    Trial = Any  # type: ignore[misc, assignment]

    class Callback:  # type: ignore[no-redef]
        pass

from noboom_cluster.noboom_cli_lib.specs import PairResult, PairRunSpec

from ..notifications import BestEffortNotificationDispatcher, get_shared_dispatcher
from ..notifications.events import (
    NotificationEvent,
    NotificationEventType,
    NotificationSeverity,
)
from .interfaces import PairOutputCallback

logger = logging.getLogger(__name__)

_RAM_BREACH_MARKERS = (
    "outofmemory",
    "out of memory",
    "oom",
    "memory usage",
    "ray out of memory",
    "worker killed due to memory",
    "exceeds the memory",
)


def _safe_message(message: Optional[str]) -> Optional[str]:
    if message is None:
        return None
    text = str(message)
    if len(text) > 1000:
        return text[:997] + "..."
    return text


def _is_ram_breach_message(message: Optional[str]) -> bool:
    if message is None:
        return False
    normalized = str(message).lower().replace("_", "").replace("-", " ")
    return any(marker in normalized for marker in _RAM_BREACH_MARKERS)


def _status_severity(status: str) -> NotificationSeverity:
    status_upper = status.upper()
    if status_upper in {"SUCCEEDED", "COMPLETED", "TERMINATED", "STARTED"}:
        return NotificationSeverity.INFO
    if status_upper in {"STOPPED", "CANCELLED", "PRUNED"}:
        return NotificationSeverity.WARNING
    return NotificationSeverity.ERROR


def _event_message(prefix: str, status: str, message: Optional[str]) -> str:
    base = f"{prefix} finished with status {status.upper()}."
    safe = _safe_message(message)
    if safe:
        return f"{base} Message: {safe}"
    return base


def pair_start_event(
    pair_spec: PairRunSpec,
    *,
    controller_run_id: Optional[str] = None,
) -> NotificationEvent:
    return NotificationEvent(
        event_type=NotificationEventType.PAIR_START,
        severity=NotificationSeverity.INFO,
        title="Pair started",
        message=f"Pair {pair_spec.pair_id} started.",
        status="STARTED",
        model_name=pair_spec.model_name,
        dataset_name=pair_spec.dataset_name,
        pair_id=pair_spec.pair_id,
        controller_run_id=controller_run_id,
        details={
            "timestamp": pair_spec.timestamp,
            "submission_id": pair_spec.submission_id,
            "storage_path": pair_spec.storage_path,
            "tune": pair_spec.tune,
        },
    )


def pair_terminal_event(
    pair_spec: PairRunSpec,
    *,
    status: str,
    job_status_message: Optional[str] = None,
    result: Optional[PairResult] = None,
    controller_run_id: Optional[str] = None,
) -> NotificationEvent:
    details: Dict[str, Any] = {
        "timestamp": pair_spec.timestamp,
        "submission_id": pair_spec.submission_id,
        "storage_path": pair_spec.storage_path,
        "tune": pair_spec.tune,
    }
    if result is not None:
        details.update(
            {
                "partial_result": result.partial_result,
                "result_source": result.result_source,
                "study_run_id": result.study_run_id,
            }
        )
    return NotificationEvent(
        event_type=NotificationEventType.PAIR_TERMINAL,
        severity=_status_severity(status),
        title="Pair terminal",
        message=_event_message(f"Pair {pair_spec.pair_id}", status, job_status_message),
        status=status.upper(),
        model_name=pair_spec.model_name,
        dataset_name=pair_spec.dataset_name,
        pair_id=pair_spec.pair_id,
        controller_run_id=controller_run_id,
        run_id=result.study_run_id if result is not None else None,
        details=details,
    )


def ram_breach_event(
    *,
    title: str,
    message: Optional[str],
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    pair_id: Optional[str] = None,
    trial_name: Optional[str] = None,
    controller_run_id: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> NotificationEvent:
    return NotificationEvent(
        event_type=NotificationEventType.RAM_BREACH,
        severity=NotificationSeverity.CRITICAL,
        title=title,
        message=_safe_message(message) or "A benchmark task reported a RAM breach.",
        status="FAILED",
        model_name=model_name,
        dataset_name=dataset_name,
        pair_id=pair_id,
        trial_name=trial_name,
        controller_run_id=controller_run_id,
        details=dict(details or {}),
    )


def trial_start_event(args: Namespace, *, trial_name: str, run_id: Optional[str] = None) -> NotificationEvent:
    return NotificationEvent(
        event_type=NotificationEventType.TRIAL_START,
        severity=NotificationSeverity.INFO,
        title="Trial started",
        message=f"Trial {trial_name} started.",
        status="STARTED",
        model_name=getattr(args, "model_name", None),
        dataset_name=getattr(args, "requested_dataset_name", getattr(args, "dataset_name", None)),
        trial_name=trial_name,
        pair_id=(
            f"{getattr(args, 'requested_dataset_name', getattr(args, 'dataset_name', ''))}"
            f"__{getattr(args, 'model_name', '')}"
        ),
        controller_run_id=getattr(args, "mlflow_controller_run_id", None),
        run_id=run_id,
        details={"timestamp": getattr(args, "timestamp", None), "tune": bool(getattr(args, "tune", False))},
    )


def trial_terminal_event(
    args: Namespace,
    *,
    trial_name: str,
    status: str,
    message: Optional[str] = None,
    result: Optional[Mapping[str, Any]] = None,
) -> NotificationEvent:
    details: Dict[str, Any] = {
        "timestamp": getattr(args, "timestamp", None),
        "tune": bool(getattr(args, "tune", False)),
    }
    if result is not None:
        details["result_keys"] = sorted(str(key) for key in result.keys())
        run_id = result.get("mlflow_run_id")
    else:
        run_id = None
    return NotificationEvent(
        event_type=NotificationEventType.TRIAL_TERMINAL,
        severity=_status_severity(status),
        title="Trial terminal",
        message=_event_message(f"Trial {trial_name}", status, message),
        status=status.upper(),
        model_name=getattr(args, "model_name", None),
        dataset_name=getattr(args, "requested_dataset_name", getattr(args, "dataset_name", None)),
        trial_name=trial_name,
        pair_id=(
            f"{getattr(args, 'requested_dataset_name', getattr(args, 'dataset_name', ''))}"
            f"__{getattr(args, 'model_name', '')}"
        ),
        controller_run_id=getattr(args, "mlflow_controller_run_id", None),
        run_id=str(run_id) if run_id is not None else None,
        details=details,
    )


def controller_start_event(
    *,
    timestamp: str,
    tune_mode: bool,
    datasets: List[str],
    models: List[str],
    pair_count: int,
    controller_run_id: Optional[str] = None,
) -> NotificationEvent:
    return NotificationEvent(
        event_type=NotificationEventType.CONTROLLER_START,
        severity=NotificationSeverity.INFO,
        title="Controller started",
        message=f"Controller run {timestamp} started.",
        status="STARTED",
        controller_run_id=controller_run_id,
        details={
            "timestamp": timestamp,
            "run_mode": "tune" if tune_mode else "single_train",
            "datasets": datasets,
            "models": models,
            "pair_count": pair_count,
        },
    )


def controller_terminal_event(
    *,
    status: str,
    timestamp: str,
    tune_mode: bool,
    datasets: List[str],
    models: List[str],
    pair_count: int,
    controller_run_id: Optional[str] = None,
    message: Optional[str] = None,
) -> NotificationEvent:
    return NotificationEvent(
        event_type=NotificationEventType.CONTROLLER_TERMINAL,
        severity=_status_severity(status),
        title="Controller terminal",
        message=_event_message(f"Controller run {timestamp}", status, message),
        status=status.upper(),
        controller_run_id=controller_run_id,
        details={
            "timestamp": timestamp,
            "run_mode": "tune" if tune_mode else "single_train",
            "datasets": datasets,
            "models": models,
            "pair_count": pair_count,
        },
    )


class NotificationPairOutputCallback(PairOutputCallback):
    def __init__(
        self,
        delegate: PairOutputCallback,
        *,
        dispatcher: Optional[BestEffortNotificationDispatcher] = None,
        controller_run_id: Optional[str] = None,
    ) -> None:
        self._delegate = delegate
        self._dispatcher = dispatcher or get_shared_dispatcher()
        self._controller_run_id = controller_run_id

    def on_job_start(self, pair_spec: PairRunSpec) -> None:
        try:
            self._delegate.on_job_start(pair_spec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pair output start callback failed: %s", exc)
        self._dispatcher.dispatch(pair_start_event(pair_spec, controller_run_id=self._controller_run_id))

    def on_job_terminal(
        self,
        pair_spec: PairRunSpec,
        status: str,
        job_status_message: Optional[str] = None,
    ) -> PairResult:
        try:
            result = self._delegate.on_job_terminal(pair_spec, status, job_status_message)
        except Exception as exc:  # noqa: BLE001
            self._dispatcher.dispatch(
                pair_terminal_event(
                    pair_spec,
                    status="FAILED",
                    job_status_message=f"Pair output callback failed with {exc.__class__.__name__}",
                    controller_run_id=self._controller_run_id,
                )
            )
            raise

        self._dispatcher.dispatch(
            pair_terminal_event(
                pair_spec,
                status=result.status,
                job_status_message=result.job_status_message or job_status_message,
                result=result,
                controller_run_id=self._controller_run_id,
            )
        )
        message = result.job_status_message or job_status_message
        if _is_ram_breach_message(message):
            self._dispatcher.dispatch(
                ram_breach_event(
                    title="Pair RAM breach",
                    message=message,
                    model_name=pair_spec.model_name,
                    dataset_name=pair_spec.dataset_name,
                    pair_id=pair_spec.pair_id,
                    controller_run_id=self._controller_run_id,
                    details={"submission_id": pair_spec.submission_id},
                )
            )
        return result


class RayTuneNotificationCallback(Callback):
    def __init__(
        self,
        *,
        args: Namespace,
        dispatcher: Optional[BestEffortNotificationDispatcher] = None,
    ) -> None:
        self._args = args
        self._dispatcher = dispatcher or get_shared_dispatcher()

    def on_trial_start(
        self,
        iteration: int,
        trials: List["Trial"],
        trial: "Trial",
        **info: Any,
    ) -> None:
        del iteration, trials, info
        self._dispatcher.dispatch(trial_start_event(self._args, trial_name=_trial_name(trial)))

    def on_trial_complete(
        self,
        iteration: int,
        trials: List["Trial"],
        trial: "Trial",
        **info: Any,
    ) -> None:
        del iteration, trials, info
        self._dispatcher.dispatch(
            trial_terminal_event(
                self._args,
                trial_name=_trial_name(trial),
                status="SUCCEEDED",
            )
        )

    def on_trial_error(
        self,
        iteration: int,
        trials: List["Trial"],
        trial: "Trial",
        **info: Any,
    ) -> None:
        del iteration, trials
        message = _trial_error_message(trial, info)
        trial_name = _trial_name(trial)
        self._dispatcher.dispatch(
            trial_terminal_event(
                self._args,
                trial_name=trial_name,
                status="FAILED",
                message=message,
            )
        )
        if _is_ram_breach_message(message):
            self._dispatcher.dispatch(
                ram_breach_event(
                    title="Trial RAM breach",
                    message=message,
                    model_name=getattr(self._args, "model_name", None),
                    dataset_name=getattr(
                        self._args,
                        "requested_dataset_name",
                        getattr(self._args, "dataset_name", None),
                    ),
                    trial_name=trial_name,
                    controller_run_id=getattr(self._args, "mlflow_controller_run_id", None),
                )
            )


def _trial_name(trial: "Trial") -> str:
    for attr_name in ("trial_id", "experiment_tag", "trainable_name"):
        value = getattr(trial, attr_name, None)
        if value:
            return str(value)
    return str(trial)


def _trial_error_message(trial: "Trial", info: Mapping[str, Any]) -> Optional[str]:
    for key in ("error_msg", "error_message", "message"):
        value = info.get(key)
        if value:
            return str(value)
    error_file = getattr(trial, "error_file", None)
    if error_file:
        path = Path(str(error_file))
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="replace")[-1000:]
            except OSError:
                return f"Trial failed; error file unavailable: {path.name}"
    return getattr(trial, "status", None)


def emit_trial_start_from_args(
    args: Namespace,
    *,
    trial_name: str,
    run_id: Optional[str] = None,
) -> None:
    get_shared_dispatcher().dispatch(trial_start_event(args, trial_name=trial_name, run_id=run_id))


def emit_trial_terminal_from_args(
    args: Namespace,
    *,
    trial_name: str,
    status: str,
    message: Optional[str] = None,
    result: Optional[Mapping[str, Any]] = None,
) -> None:
    dispatcher = get_shared_dispatcher()
    dispatcher.dispatch(
        trial_terminal_event(
            args,
            trial_name=trial_name,
            status=status,
            message=message,
            result=result,
        )
    )
    if _is_ram_breach_message(message):
        dispatcher.dispatch(
            ram_breach_event(
                title="Trial RAM breach",
                message=message,
                model_name=getattr(args, "model_name", None),
                dataset_name=getattr(args, "requested_dataset_name", getattr(args, "dataset_name", None)),
                trial_name=trial_name,
                controller_run_id=getattr(args, "mlflow_controller_run_id", None),
            )
        )


def emit_controller_start(
    *,
    timestamp: str,
    tune_mode: bool,
    datasets: List[str],
    models: List[str],
    pair_count: int,
    controller_run_id: Optional[str] = None,
) -> None:
    get_shared_dispatcher().dispatch(
        controller_start_event(
            timestamp=timestamp,
            tune_mode=tune_mode,
            datasets=datasets,
            models=models,
            pair_count=pair_count,
            controller_run_id=controller_run_id,
        )
    )


def emit_controller_terminal(
    *,
    status: str,
    timestamp: str,
    tune_mode: bool,
    datasets: List[str],
    models: List[str],
    pair_count: int,
    controller_run_id: Optional[str] = None,
    message: Optional[str] = None,
) -> None:
    get_shared_dispatcher().dispatch(
        controller_terminal_event(
            status=status,
            timestamp=timestamp,
            tune_mode=tune_mode,
            datasets=datasets,
            models=models,
            pair_count=pair_count,
            controller_run_id=controller_run_id,
            message=message,
        )
    )
