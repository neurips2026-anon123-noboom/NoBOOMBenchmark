"""Standalone Telegram relay for @noboom_notifier_bot."""
from __future__ import annotations

from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_DB_PATH = "/data/telegram-relay.sqlite3"
BOT_USERNAME = "noboom_notifier_bot"

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def parse_start_token(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    parts = text.strip().split(maxsplit=1)
    if len(parts) == 2 and parts[0] == "/start" and parts[1].strip():
        return parts[1].strip()
    if len(parts) == 1 and parts[0].startswith("/start="):
        token = parts[0].split("=", 1)[1].strip()
        return token or None
    return None


class RelayStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists subscriptions (
                    token text primary key,
                    chat_id text not null,
                    updated_at text default current_timestamp
                )
                """
            )

    def subscribe(self, token: str, chat_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into subscriptions(token, chat_id, updated_at)
                values (?, ?, current_timestamp)
                on conflict(token) do update set
                    chat_id = excluded.chat_id,
                    updated_at = current_timestamp
                """,
                (token, chat_id),
            )

    def chat_id_for_token(self, token: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "select chat_id from subscriptions where token = ?",
                (token,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])


class TelegramClient:
    def __init__(self, bot_token: str, *, timeout_s: float = 10.0) -> None:
        self._bot_token = bot_token
        self._timeout_s = timeout_s

    def _api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self._bot_token}/{method}"

    def call(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = Request(
            self._api_url(method),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def send_message(self, chat_id: str, text: str) -> Dict[str, Any]:
        return self.call(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
        )

    def set_webhook(self, url: str) -> Dict[str, Any]:
        return self.call("setWebhook", {"url": url})

    def get_updates(self, limit: int = 10) -> Dict[str, Any]:
        request = Request(self._api_url(f"getUpdates?limit={limit}"), method="GET")
        with urlopen(request, timeout=self._timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))


class RelayApp:
    def __init__(
        self,
        *,
        store: RelayStore,
        telegram: TelegramClient,
        webhook_secret: Optional[str] = None,
        relay_secret: Optional[str] = None,
    ) -> None:
        self.store = store
        self.telegram = telegram
        self.webhook_secret = webhook_secret
        self.relay_secret = relay_secret

    def webhook_path(self) -> str:
        if self.webhook_secret:
            return f"/telegram/webhook/{self.webhook_secret}"
        return "/telegram/webhook"


def make_handler(app: RelayApp) -> type[BaseHTTPRequestHandler]:
    class RelayHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - " + format, self.address_string(), *args)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/healthz":
                _json_response(self, 200, {"ok": True})
                return
            if path == "/":
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "bot": f"@{BOT_USERNAME}",
                        "webhook_path": app.webhook_path(),
                        "send_path": "/noboom/telegram/send",
                    },
                )
                return
            _json_response(self, 404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = _read_json(self)
            except json.JSONDecodeError:
                _json_response(self, 400, {"ok": False, "error": "invalid json"})
                return

            if path == app.webhook_path():
                self._handle_webhook(payload)
                return
            if path == "/noboom/telegram/send":
                self._handle_send(payload)
                return
            _json_response(self, 404, {"ok": False, "error": "not found"})

        def _handle_webhook(self, payload: Dict[str, Any]) -> None:
            message = payload.get("message") or payload.get("edited_message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            token = parse_start_token(message.get("text"))
            if chat_id is not None and token:
                app.store.subscribe(token, str(chat_id))
                app.telegram.send_message(
                    str(chat_id),
                    "NoBoom notifications are enabled for this benchmark token.",
                )
            _json_response(self, 200, {"ok": True})

        def _handle_send(self, payload: Dict[str, Any]) -> None:
            if app.relay_secret:
                actual_secret = self.headers.get("X-NoBoom-Relay-Secret")
                if actual_secret != app.relay_secret:
                    _json_response(self, 403, {"ok": False, "error": "forbidden"})
                    return
            token = str(payload.get("token") or "")
            text = str(payload.get("text") or "")
            if not token or not text:
                _json_response(self, 400, {"ok": False, "error": "missing token or text"})
                return
            chat_id = app.store.chat_id_for_token(token)
            if not chat_id:
                _json_response(self, 202, {"ok": True, "delivered": False, "reason": "not subscribed"})
                return
            app.telegram.send_message(chat_id, text)
            _json_response(self, 200, {"ok": True, "delivered": True})

    return RelayHandler


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve")
    webhook = subparsers.add_parser("set-webhook")
    webhook.add_argument("--base-url", required=True)
    updates = subparsers.add_parser("updates")
    updates.add_argument("--limit", type=int, default=10)
    return parser


def _telegram_from_env() -> TelegramClient:
    token = _env("NOBOOM_TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set NOBOOM_TELEGRAM_BOT_TOKEN for the relay process.")
    timeout_s = float(_env("NOBOOM_TELEGRAM_REQUEST_TIMEOUT_S", "10"))
    return TelegramClient(token, timeout_s=timeout_s)


def serve() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    host = _env("NOBOOM_TELEGRAM_RELAY_HOST", DEFAULT_HOST)
    port = int(_env("NOBOOM_TELEGRAM_RELAY_PORT", str(DEFAULT_PORT)))
    app = RelayApp(
        store=RelayStore(_env("NOBOOM_TELEGRAM_RELAY_DB", DEFAULT_DB_PATH)),
        telegram=_telegram_from_env(),
        webhook_secret=_env("NOBOOM_TELEGRAM_WEBHOOK_SECRET") or None,
        relay_secret=_env("NOBOOM_TELEGRAM_RELAY_SHARED_SECRET") or None,
    )
    logger.info("Starting NoBoom Telegram relay on %s:%s for @%s.", host, port, BOT_USERNAME)
    logger.info("Telegram webhook path: %s", app.webhook_path())
    server = ThreadingHTTPServer((host, port), make_handler(app))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


def set_webhook(base_url: str) -> int:
    webhook_secret = _env("NOBOOM_TELEGRAM_WEBHOOK_SECRET") or None
    path = f"/telegram/webhook/{webhook_secret}" if webhook_secret else "/telegram/webhook"
    url = base_url.rstrip("/") + path
    result = _telegram_from_env().set_webhook(url)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Bot link template: https://t.me/noboom_notifier_bot?start=<token>")
    return 0 if result.get("ok") else 1


def print_updates(limit: int) -> int:
    try:
        result = _telegram_from_env().get_updates(limit=limit)
    except HTTPError as exc:
        print(exc.read().decode("utf-8"), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    command = args.command or "serve"
    if command == "serve":
        return serve()
    if command == "set-webhook":
        return set_webhook(args.base_url)
    if command == "updates":
        return print_updates(args.limit)
    raise RuntimeError(f"Unknown command: {command}")
