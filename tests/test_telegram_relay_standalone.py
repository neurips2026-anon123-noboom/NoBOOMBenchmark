from __future__ import annotations

from noboom_telegram_relay import relay


def test_parse_start_token() -> None:
    assert relay.parse_start_token("/start abc123") == "abc123"
    assert relay.parse_start_token("/start=abc123") == "abc123"
    assert relay.parse_start_token("/help") is None


def test_relay_store_subscription_round_trip(tmp_path) -> None:
    store = relay.RelayStore(str(tmp_path / "relay.sqlite3"))

    store.subscribe("token-1", "chat-1")

    assert store.chat_id_for_token("token-1") == "chat-1"
    assert store.chat_id_for_token("missing") is None
