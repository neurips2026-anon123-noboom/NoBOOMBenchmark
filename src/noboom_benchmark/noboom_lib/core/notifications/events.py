"""Notification event model and safe rendering helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, Mapping, Optional


class NotificationEventType(str, Enum):
    CONTROLLER_START = "controller_start"
    CONTROLLER_TERMINAL = "controller_terminal"
    PAIR_START = "pair_start"
    PAIR_TERMINAL = "pair_terminal"
    TRIAL_START = "trial_start"
    TRIAL_TERMINAL = "trial_terminal"
    RAM_BREACH = "ram_breach"


class NotificationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


_SECRET_MARKERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
)


def _is_secret_key(key: str) -> bool:
    key_lower = key.lower()
    return any(marker in key_lower for marker in _SECRET_MARKERS)


def _safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[redacted]" if _is_secret_key(str(key)) else _safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class NotificationEvent:
    event_type: NotificationEventType
    severity: NotificationSeverity
    title: str
    message: str
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: Optional[str] = None
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    trial_name: Optional[str] = None
    pair_id: Optional[str] = None
    controller_run_id: Optional[str] = None
    run_id: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def safe_details(self) -> Dict[str, Any]:
        safe = _safe_value(self.details)
        if isinstance(safe, dict):
            return safe
        return {"details": safe}

    def to_safe_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp_utc": self.timestamp_utc.isoformat(),
        }
        optional_fields = {
            "status": self.status,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "trial_name": self.trial_name,
            "pair_id": self.pair_id,
            "controller_run_id": self.controller_run_id,
            "run_id": self.run_id,
        }
        payload.update({key: value for key, value in optional_fields.items() if value is not None})
        details = self.safe_details()
        if details:
            payload["details"] = details
        return payload

    def subject(self) -> str:
        status = f" {self.status}" if self.status else ""
        return f"[NoBoom] {self.title}{status}"

    def render_text(self) -> str:
        lines = [
            self.subject(),
            "",
            self.message,
            "",
            f"event_type: {self.event_type.value}",
            f"severity: {self.severity.value}",
            f"timestamp_utc: {self.timestamp_utc.isoformat()}",
        ]
        for key, value in self.to_safe_dict().items():
            if key in {"event_type", "severity", "title", "message", "timestamp_utc", "details"}:
                continue
            lines.append(f"{key}: {value}")
        details = self.safe_details()
        if details:
            lines.extend(["", "details:", json.dumps(details, indent=2, sort_keys=True)])
        return "\n".join(lines)
