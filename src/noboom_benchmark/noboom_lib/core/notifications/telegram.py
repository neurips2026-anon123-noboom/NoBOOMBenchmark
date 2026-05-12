"""Telegram notification enrollment helpers."""
from __future__ import annotations

import json
import os
import secrets
from typing import Dict, Optional
from urllib.parse import quote

TELEGRAM_BOT_USERNAME = "noboom_notifier_bot"
TELEGRAM_BOT_DISPLAY_NAME = "@noboom_notifier_bot"
TELEGRAM_START_LINK_ENV = "NOBOOM_NOTIFY_TELEGRAM_START_LINK"
TELEGRAM_LINK_TOKEN_ENV = "NOBOOM_NOTIFY_TELEGRAM_LINK_TOKEN"
TELEGRAM_RELAY_URL_ENV = "NOBOOM_NOTIFY_TELEGRAM_RELAY_URL"


def generate_telegram_link_token() -> str:
    return secrets.token_urlsafe(24)


def telegram_start_link(token: str) -> str:
    return f"https://t.me/{TELEGRAM_BOT_USERNAME}?start={quote(token, safe='')}"


def ensure_telegram_enrollment_env(env: Optional[Dict[str, str]] = None) -> str:
    target = os.environ if env is None else env
    token = target.get(TELEGRAM_LINK_TOKEN_ENV)
    if not token:
        token = generate_telegram_link_token()
    link = telegram_start_link(token)
    target[TELEGRAM_LINK_TOKEN_ENV] = token
    target[TELEGRAM_START_LINK_ENV] = link
    return link


def telegram_relay_payload(token: str, text: str, event: Dict[str, object]) -> bytes:
    payload = {"token": token, "text": text, "event": event}
    return json.dumps(payload, sort_keys=True).encode("utf-8")
