# NoBoom Telegram Relay

Standalone relay for `@noboom_notifier_bot`. It maps NoBoom-generated
`/start <token>` links to Telegram chat IDs and forwards notifications from
benchmarks via:

```bash
NOBOOM_NOTIFY_TELEGRAM_RELAY_URL=https://<relay>/noboom/telegram/send
```

Run locally:

```bash
export NOBOOM_TELEGRAM_BOT_TOKEN="<botfather-token>"
PYTHONPATH=src python -m noboom_telegram_relay serve
```

Build/push Docker:

```bash
python src/noboom_telegram_relay/build_push.py --image ghcr.io/denix56/noboom-telegram-relay:latest --push
```

Jarvis CPU-only relay helper:

```bash
export JL_API_KEY="<jarvis-token>"
uv run --with git+https://github.com/jarvislabsai/JLClient.git \
  python src/noboom_telegram_relay/deploy_jarvis.py
```
