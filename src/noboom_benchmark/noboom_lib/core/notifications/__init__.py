"""Best-effort notifications for NoBoom benchmark runs."""
from __future__ import annotations

from .dispatcher import (
    BestEffortNotificationDispatcher,
    build_dispatcher_from_env,
    get_shared_dispatcher,
    reset_shared_dispatcher,
)
from .events import NotificationEvent, NotificationEventType, NotificationSeverity
from .settings import NotificationSettings
from .telegram import (
    TELEGRAM_BOT_DISPLAY_NAME,
    ensure_telegram_enrollment_env,
    telegram_start_link,
)

__all__ = [
    "BestEffortNotificationDispatcher",
    "NotificationEvent",
    "NotificationEventType",
    "NotificationSettings",
    "NotificationSeverity",
    "TELEGRAM_BOT_DISPLAY_NAME",
    "build_dispatcher_from_env",
    "ensure_telegram_enrollment_env",
    "get_shared_dispatcher",
    "reset_shared_dispatcher",
    "telegram_start_link",
]
