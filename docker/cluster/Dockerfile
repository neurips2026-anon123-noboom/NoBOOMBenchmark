FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# Only runtime dependencies (no compilers, no build toolchain)
RUN apt-get update \
    && apt-get install -y --quiet --no-install-recommends \
       python3 libpq-dev pkg-config libssl-dev openssh-client ca-certificates \
       git \
    && rm -fr /var/lib/apt/lists/*

# Install uv (useful if you want to install extra packages at runtime)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace/NoBoomBenchmark

ADD src/noboom_cluster /workspace/NoBoomBenchmark

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --extra cluster

ENV RAY_UP_enable_autoscaler_v2=0

# Default entrypoint (you can override with `docker run --entrypoint` or CMD)
ENTRYPOINT ["python cli.py"]
