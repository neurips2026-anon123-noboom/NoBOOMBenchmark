"""Notification sink implementations."""
from __future__ import annotations

from email.message import EmailMessage
import smtplib
from typing import Optional, Protocol
from urllib.request import Request, urlopen

from .events import NotificationEvent
from .telegram import telegram_relay_payload


class NotificationSink(Protocol):
    name: str

    def send(self, event: NotificationEvent) -> None:
        raise NotImplementedError


class SmtpEmailSink:
    name = "smtp_email"

    def __init__(
        self,
        *,
        host: str,
        port: int,
        sender: str,
        recipients: list[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        use_ssl: bool = False,
        timeout_s: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._sender = sender
        self._recipients = recipients
        self._username = username
        self._password = password
        self._use_tls = use_tls
        self._use_ssl = use_ssl
        self._timeout_s = timeout_s

    def send(self, event: NotificationEvent) -> None:
        message = EmailMessage()
        message["Subject"] = event.subject()
        message["From"] = self._sender
        message["To"] = ", ".join(self._recipients)
        message.set_content(event.render_text())

        smtp_cls = smtplib.SMTP_SSL if self._use_ssl else smtplib.SMTP
        with smtp_cls(self._host, self._port, timeout=self._timeout_s) as smtp:
            if self._use_tls and not self._use_ssl:
                smtp.starttls()
            if self._username:
                smtp.login(self._username, self._password or "")
            smtp.send_message(message)


class TelegramNotifierBotSink:
    name = "telegram_notifier_bot"

    def __init__(
        self,
        *,
        relay_url: str,
        link_token: str,
        relay_secret: Optional[str] = None,
        timeout_s: float = 10.0,
    ) -> None:
        self._relay_url = relay_url
        self._link_token = link_token
        self._relay_secret = relay_secret
        self._timeout_s = timeout_s

    def send(self, event: NotificationEvent) -> None:
        text = event.render_text()
        if len(text) > 4000:
            text = text[:3997] + "..."
        headers = {"Content-Type": "application/json"}
        if self._relay_secret:
            headers["X-NoBoom-Relay-Secret"] = self._relay_secret
        request = Request(
            self._relay_url,
            data=telegram_relay_payload(self._link_token, text, event.to_safe_dict()),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_s) as response:
            response.read()
