FROM ghcr.io/mlflow/mlflow:latest

ENV DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1

ARG TARGETPLATFORM

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
    uv pip install psycopg[binary,pool] boto3
