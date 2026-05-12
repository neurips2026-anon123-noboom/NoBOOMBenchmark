##############################
# Stage 0: uv-enabled Python base
##############################
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS uv-python-base

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=240 \
    UV_LOCK_TIMEOUT=240 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --quiet --no-install-recommends \
       python3 python3-venv python3-pip python3-dev \
       git curl build-essential cmake \
       libssl-dev ca-certificates meson ninja-build cython3 \
    && rm -fr /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG PYTHON_VERSION=3.13
ARG TARGETPLATFORM

##############################
# Stage 1: build treeple wheel
##############################
FROM uv-python-base AS treeple-builder

# IMPORTANT: pin the source to an immutable ref for Docker caching
ARG TREEPLE_REF=ef417e47e12caaa70ecc027db53f18b70b895930
ARG TREEPLE_REPOSITORY_URL=https://github.com/denix56/treeple.git

# Clone at pinned ref (do NOT build from a moving branch)
RUN git clone --recurse-submodules "$TREEPLE_REPOSITORY_URL" /treeple \
    && cd /treeple \
    && git checkout --detach "$TREEPLE_REF" \
    && git submodule update --init --recursive

WORKDIR /treeple
RUN uv venv --python=${PYTHON_VERSION} /treeple/.venv

RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
    uv pip install pip

# 2) Now you can use --locked reliably
RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
    uv pip install --python=/treeple/.venv/bin/python -r build_requirements.txt

RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
    uv run --active --no-editable spin build -j4

RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
    /treeple/.venv/bin/python -m pip wheel --wheel-dir /build/wheelhouse --no-deps .

############################

FROM uv-python-base AS torch-cluster-builder

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX"

# IMPORTANT: pin to an immutable ref to avoid cache busts when the default
# branch advances. Builds were frequently recompiling this stage because the
# git+https source pointer moved between runs.
ARG TORCH_CLUSTER_REF=6dabb048d29c2bc788fce885d0e13f888c8987a2

RUN git clone https://github.com/rusty1s/pytorch_cluster.git /cluster \
    && cd /cluster \
    && git checkout --detach "$TORCH_CLUSTER_REF"

WORKDIR /cluster
RUN uv venv --python=${PYTHON_VERSION} /cluster/.venv

RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
     uv pip install --python=/cluster/.venv/bin/python setuptools torch --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
     FORCE_CUDA=1 uv build --python=/cluster/.venv/bin/python --out-dir /build/wheelhouse


#############################
# Stage 2: main builder image
#############################
FROM uv-python-base AS builder

# System deps needed for building Python packages and CUDA extensions
RUN apt-get update \
    && apt-get install -y --quiet --no-install-recommends \
       libpq-dev pkg-config openssh-client cython3 \
    && rm -fr /var/lib/apt/lists/*

##############################
# Install Python dependencies (NoBoom)
##############################
WORKDIR /workspace/noboom

# Compile and install dependencies into the system Python (UV_SYSTEM_PYTHON=1)
RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --python=${PYTHON_VERSION} --locked --no-install-project --extra benchmark

##############################
# Install prebuilt treeple + torch_cluster libraries into this env
##############################
# Copy only the installed packages from the builder venvs after uv sync.

COPY --from=treeple-builder /build/wheelhouse /wheelhouse
COPY --from=torch-cluster-builder /build/wheelhouse /wheelhouse

RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
    uv pip install --python=.venv/bin/python --no-deps /wheelhouse/*.whl

RUN --mount=type=cache,id=uv-cache-$TARGETPLATFORM,target=/root/.cache/uv \
    uv pip install pip

FROM chrislusf/seaweedfs:latest AS seaweed

##############################
# Runtime image
##############################
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Only runtime dependencies
RUN apt-get update \
    && apt-get install -y --quiet --no-install-recommends \
       python3 python3-dev cython3 libpq-dev pkg-config libssl-dev openssh-client ca-certificates \
       rclone rsync git iputils-ping tmux build-essential \
    && rm -fr /var/lib/apt/lists/*

# Install uv for Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY --from=seaweed /usr/bin/weed /usr/bin/weed
RUN chmod +x /usr/bin/weed && weed version

# Copy project (including everything uv installed into system Python in the builder)
COPY --from=builder /workspace/noboom/.venv /opt/noboom/.venv
COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python


ENV PATH="/opt/noboom/.venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/noboom/.venv" \
    PYTHONPATH="/workspace/noboom/src"

# Workdir where you'll mount / copy your project sources
WORKDIR /workspace/noboom
