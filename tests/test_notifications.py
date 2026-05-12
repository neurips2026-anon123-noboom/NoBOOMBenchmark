from __future__ import annotations

from email.message import EmailMessage
import json
from pathlib import Path
from typing import List, Optional

from noboom_benchmark.noboom_lib.core.notifications import (
    BestEffortNotificationDispatcher,
    NotificationEvent,
    NotificationEventType,
    NotificationSettings,
    NotificationSeverity,
    ensure_telegram_enrollment_env,
    telegram_start_link,
)
from noboom_benchmark.noboom_lib.core.notifications import sinks
from noboom_benchmark.noboom_lib.core.tune.notification_callbacks import NotificationPairOutputCallback
from noboom_cluster.noboom_cli_lib.specs import PairResult, PairRunSpec


class RecordingSink:
    name = "recording"

    def __init__(self) -> None:
        self.events: List[NotificationEvent] = []

    def send(self, event: NotificationEvent) -> None:
        self.events.append(event)


class FailingSink:
    name = "failing"

    def send(self, event: NotificationEvent) -> None:
        del event
        raise RuntimeError("sink failed")


class FakeSmtp:
    instances: List["FakeSmtp"] = []

    def __init__(self, host: str, port: int, timeout: Optional[float] = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.login_calls: List[tuple[str, str]] = []
        self.messages: List[EmailMessage] = []
        FakeSmtp.instances.append(self)

    def __enter__(self) -> "FakeSmtp":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def starttls(self) -> None:
        self.started_tls = True

    def login(self, username: str, password: str) -> None:
        self.login_calls.append((username, password))

    def send_message(self, message: EmailMessage) -> None:
        self.messages.append(message)


class RecordingPairCallback:
    def on_job_terminal(
        self,
        pair_spec: PairRunSpec,
        status: str,
        job_status_message: Optional[str] = None,
    ) -> PairResult:
        return PairResult(
            model_name=pair_spec.model_name,
            dataset_name=pair_spec.dataset_name,
            storage_path=pair_spec.storage_path,
            status=status,
            partial_result=status != "SUCCEEDED",
            job_status_message=job_status_message,
        )


def _event() -> NotificationEvent:
    return NotificationEvent(
        event_type=NotificationEventType.PAIR_TERMINAL,
        severity=NotificationSeverity.ERROR,
        title="Pair failed",
        message="neutralad on ome failed",
        status="FAILED",
        pair_id="ome__neutralad",
        details={"telegram_token": "secret-token", "nested": {"smtp_password": "secret-password"}},
    )


def _pair_spec(tmp_path: Path) -> PairRunSpec:
    return PairRunSpec(
        experiment_id="1",
        source_experiment_id=None,
        model_name="neutralad",
        dataset_name="ome",
        timestamp="20260311_120000",
        storage_path=str(tmp_path / "storage"),
        gpus_per_run=0.25,
        optuna_storage_uri="memory://",
        config_dir="configs",
        temp_dir=str(tmp_path),
        verbose=0,
        tune=True,
        env_file=None,
        tracking_uri=f"file://{tmp_path / 'mlruns'}",
        s3_endpoint_url=None,
    )


def test_notification_settings_from_env_enables_configured_channels(monkeypatch) -> None:
    monkeypatch.setenv("NOBOOM_NOTIFY_EMAIL_ENABLED", "1")
    monkeypatch.setenv("NOBOOM_NOTIFY_SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("NOBOOM_NOTIFY_SMTP_FROM", "noboom@example.test")
    monkeypatch.setenv("NOBOOM_NOTIFY_SMTP_TO", "ops@example.test,ml@example.test")
    monkeypatch.setenv("NOBOOM_NOTIFY_TELEGRAM_ENABLED", "1")
    monkeypatch.setenv("NOBOOM_NOTIFY_TELEGRAM_LINK_TOKEN", "link-token")
    monkeypatch.setenv("NOBOOM_NOTIFY_TELEGRAM_RELAY_URL", "https://relay.example.test/send")

    settings = NotificationSettings.from_env()

    assert settings.should_send_email() is True
    assert settings.should_send_telegram() is True
    assert settings.smtp_to == ["ops@example.test", "ml@example.test"]


def test_telegram_enrollment_link_uses_notifier_bot() -> None:
    env: dict[str, str] = {}

    link = ensure_telegram_enrollment_env(env)

    assert link.startswith("https://t.me/noboom_notifier_bot?start=")
    assert telegram_start_link(env["NOBOOM_NOTIFY_TELEGRAM_LINK_TOKEN"]) == link


def test_notification_event_redacts_secret_details() -> None:
    rendered = _event().render_text()

    assert "secret-token" not in rendered
    assert "secret-password" not in rendered
    assert "[redacted]" in rendered


def test_dispatcher_is_best_effort_when_one_sink_fails(caplog) -> None:
    recording = RecordingSink()
    dispatcher = BestEffortNotificationDispatcher(sinks=[FailingSink(), recording])

    dispatcher.dispatch(_event())

    assert len(recording.events) == 1
    assert "Notification sink 'failing' failed" in caplog.text


def test_smtp_email_sink_sends_without_real_smtp(monkeypatch) -> None:
    FakeSmtp.instances = []
    monkeypatch.setattr(sinks.smtplib, "SMTP", FakeSmtp)
    sink = sinks.SmtpEmailSink(
        host="smtp.example.test",
        port=2525,
        sender="noboom@example.test",
        recipients=["ops@example.test"],
        username="smtp-user",
        password="smtp-password",
    )

    sink.send(_event())

    smtp = FakeSmtp.instances[0]
    assert smtp.started_tls is True
    assert smtp.login_calls == [("smtp-user", "smtp-password")]
    assert smtp.messages[0]["Subject"] == "[NoBoom] Pair failed FAILED"


def test_telegram_sink_posts_relay_payload_without_network(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["full_url"] = request.full_url
        captured["data"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(sinks, "urlopen", fake_urlopen)
    sink = sinks.TelegramNotifierBotSink(
        relay_url="https://relay.example.test/send",
        link_token="link-token",
        timeout_s=3.0,
    )

    sink.send(_event())

    payload = json.loads(captured["data"].decode("utf-8"))
    assert captured["full_url"] == "https://relay.example.test/send"
    assert payload["token"] == "link-token"
    assert payload["event"]["event_type"] == "pair_terminal"


def test_pair_notification_wrapper_emits_start_terminal_and_ram_breach(tmp_path: Path) -> None:
    recording = RecordingSink()
    dispatcher = BestEffortNotificationDispatcher(sinks=[recording])
    callback = NotificationPairOutputCallback(
        RecordingPairCallback(),  # type: ignore[arg-type]
        dispatcher=dispatcher,
        controller_run_id="controller-run-id",
    )
    pair_spec = _pair_spec(tmp_path)

    callback.on_job_start(pair_spec)
    callback.on_job_terminal(pair_spec, "FAILED", "Ray OutOfMemoryError")

    assert [event.event_type for event in recording.events] == [
        NotificationEventType.PAIR_START,
        NotificationEventType.PAIR_TERMINAL,
        NotificationEventType.RAM_BREACH,
    ]
    assert recording.events[0].status == "STARTED"
    assert recording.events[0].controller_run_id == "controller-run-id"
