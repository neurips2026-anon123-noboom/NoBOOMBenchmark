FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    NOBOOM_TELEGRAM_RELAY_HOST=0.0.0.0 \
    NOBOOM_TELEGRAM_RELAY_PORT=8080 \
    NOBOOM_TELEGRAM_RELAY_DB=/data/telegram-relay.sqlite3

WORKDIR /app
COPY . /app/noboom_telegram_relay

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8080

CMD ["python", "-m", "noboom_telegram_relay", "serve"]
