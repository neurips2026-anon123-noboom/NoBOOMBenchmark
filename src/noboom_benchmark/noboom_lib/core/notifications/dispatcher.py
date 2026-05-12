"""Shared best-effort notification dispatcher."""
from __future__ import annotations

import logging
from threading import Lock
from typing import List, Optional

from .events import NotificationEvent
from .settings import NotificationSettings
from .sinks import NotificationSink, SmtpEmailSink, TelegramNotifierBotSink

logger = logging.getLogger(__name__)


class BestEffortNotificationDispatcher:
    def __init__(self, sinks: Optional[List[NotificationSink]] = None) -> None:
        self._sinks = list(sinks or [])

    @property
    def enabled(self) -> bool:
        return bool(self._sinks)

    def dispatch(self, event: NotificationEvent) -> None:
        for sink in self._sinks:
            try:
                sink.send(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Notification sink '%s' failed for event '%s' with %s.",
                    getattr(sink, "name", sink.__class__.__name__),
                    event.event_type.value,
                    exc.__class__.__name__,
                )


def build_dispatcher_from_env() -> BestEffortNotificationDispatcher:
    settings = NotificationSettings.from_env()
    sinks: List[NotificationSink] = []
    if settings.should_send_email():
        sinks.append(
            SmtpEmailSink(
                host=str(settings.smtp_host),
                port=settings.smtp_port,
                sender=str(settings.smtp_from),
                recipients=settings.smtp_to,
                username=settings.smtp_username,
                password=settings.smtp_password,
                use_tls=settings.smtp_use_tls,
                use_ssl=settings.smtp_use_ssl,
                timeout_s=settings.smtp_timeout_s,
            )
        )
    if settings.should_send_telegram():
        sinks.append(
            TelegramNotifierBotSink(
                relay_url=str(settings.telegram_relay_url),
                link_token=str(settings.telegram_link_token),
                relay_secret=settings.telegram_relay_secret,
                timeout_s=settings.telegram_timeout_s,
            )
        )
    return BestEffortNotificationDispatcher(sinks=sinks)


_shared_lock = Lock()
_shared_dispatcher: Optional[BestEffortNotificationDispatcher] = None


def get_shared_dispatcher(reset: bool = False) -> BestEffortNotificationDispatcher:
    global _shared_dispatcher
    with _shared_lock:
        if reset or _shared_dispatcher is None:
            _shared_dispatcher = build_dispatcher_from_env()
        return _shared_dispatcher


def reset_shared_dispatcher() -> None:
    global _shared_dispatcher
    with _shared_lock:
        _shared_dispatcher = None
