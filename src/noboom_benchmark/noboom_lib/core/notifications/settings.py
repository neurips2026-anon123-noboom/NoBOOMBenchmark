"""Environment-backed notification settings."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import List, Optional


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "on", "yes", "true"}


def parse_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if value is None or value.strip() == "":
        return default
    return float(value)


def parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class NotificationSettings:
    enabled: bool = False
    email_enabled: bool = False
    telegram_enabled: bool = False
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = field(default=None, repr=False)
    smtp_from: Optional[str] = None
    smtp_to: List[str] = field(default_factory=list)
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout_s: float = 10.0
    telegram_link_token: Optional[str] = None
    telegram_start_link: Optional[str] = None
    telegram_relay_url: Optional[str] = None
    telegram_relay_secret: Optional[str] = field(default=None, repr=False)
    telegram_timeout_s: float = 10.0
    ram_breach_ratio: Optional[float] = None

    @classmethod
    def from_env(cls) -> "NotificationSettings":
        email_enabled = parse_bool(os.getenv("NOBOOM_NOTIFY_EMAIL_ENABLED"), False)
        telegram_enabled = parse_bool(os.getenv("NOBOOM_NOTIFY_TELEGRAM_ENABLED"), False)
        enabled = parse_bool(
            os.getenv("NOBOOM_NOTIFICATIONS_ENABLED"),
            email_enabled or telegram_enabled,
        )
        return cls(
            enabled=enabled,
            email_enabled=email_enabled,
            telegram_enabled=telegram_enabled,
            smtp_host=os.getenv("NOBOOM_NOTIFY_SMTP_HOST"),
            smtp_port=int(os.getenv("NOBOOM_NOTIFY_SMTP_PORT", "587")),
            smtp_username=os.getenv("NOBOOM_NOTIFY_SMTP_USERNAME"),
            smtp_password=os.getenv("NOBOOM_NOTIFY_SMTP_PASSWORD"),
            smtp_from=os.getenv("NOBOOM_NOTIFY_EMAIL_FROM") or os.getenv("NOBOOM_NOTIFY_SMTP_FROM"),
            smtp_to=parse_csv(os.getenv("NOBOOM_NOTIFY_EMAIL_TO") or os.getenv("NOBOOM_NOTIFY_SMTP_TO")),
            smtp_use_tls=parse_bool(os.getenv("NOBOOM_NOTIFY_SMTP_USE_TLS"), True),
            smtp_use_ssl=parse_bool(os.getenv("NOBOOM_NOTIFY_SMTP_USE_SSL"), False),
            smtp_timeout_s=float(os.getenv("NOBOOM_NOTIFY_SMTP_TIMEOUT_S", "10")),
            telegram_link_token=os.getenv("NOBOOM_NOTIFY_TELEGRAM_LINK_TOKEN"),
            telegram_start_link=os.getenv("NOBOOM_NOTIFY_TELEGRAM_START_LINK"),
            telegram_relay_url=os.getenv("NOBOOM_NOTIFY_TELEGRAM_RELAY_URL"),
            telegram_relay_secret=os.getenv("NOBOOM_NOTIFY_TELEGRAM_RELAY_SECRET"),
            telegram_timeout_s=float(os.getenv("NOBOOM_NOTIFY_TELEGRAM_TIMEOUT_S", "10")),
            ram_breach_ratio=parse_float(os.getenv("NOBOOM_NOTIFY_RAM_BREACH_RATIO")),
        )

    def should_send_email(self) -> bool:
        return bool(
            self.enabled
            and self.email_enabled
            and self.smtp_host
            and self.smtp_from
            and self.smtp_to
        )

    def should_send_telegram(self) -> bool:
        return bool(
            self.enabled
            and self.telegram_enabled
            and self.telegram_link_token
            and self.telegram_relay_url
        )
