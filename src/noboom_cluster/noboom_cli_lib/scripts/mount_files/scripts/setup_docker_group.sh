#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${NOBOOM_SSH_USER:-${SUDO_USER:-}}"

if [[ -z "$USER_NAME" || "$USER_NAME" == "root" ]]; then
  for candidate in ubuntu cloud; do
    if id -u "$candidate" >/dev/null 2>&1; then
      USER_NAME="$candidate"
      break
    fi
  done
fi

# Must be run as root
if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root (use sudo)." >&2
  exit 1
fi

if ! id -u "$USER_NAME" >/dev/null 2>&1; then
  echo "User '$USER_NAME' does not exist." >&2
  exit 1
fi

echo "[DOCKER] Ensuring Docker is installed..."

PKG_MGR=""

if ! command -v docker >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="apt"
    echo "[DOCKER] Detected apt-based system. Installing docker.io..."
    apt-get update -y
    apt-get install -y docker.io
    systemctl enable docker
    systemctl start docker
  elif command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
    echo "[DOCKER] Detected dnf-based system. Installing docker..."
    dnf install -y docker
    systemctl enable docker
    systemctl start docker
  elif command -v yum >/dev/null 2>&1; then
    PKG_MGR="yum"
    echo "[DOCKER] Detected yum-based system. Installing docker..."
    yum install -y docker
    systemctl enable docker
    systemctl start docker
  else
    echo "[DOCKER] Could not detect supported package manager (apt, dnf, yum). Install Docker manually." >&2
    exit 1
  fi
else
  echo "[DOCKER] Docker is already installed."
  if command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="apt"
  elif command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
  elif command -v yum >/dev/null 2>&1; then
    PKG_MGR="yum"
  fi
fi

echo "[DOCKER] Ensuring Docker Compose is installed..."

get_docker_major() {
  docker version --format '{{.Server.Version}}' 2>/dev/null \
    | awk -F. '{print $1}'
}

have_docker() {
  command -v docker >/dev/null 2>&1
}

have_compose_v2() {
  have_docker && docker compose version >/dev/null 2>&1
}

have_compose_v1() {
  command -v docker-compose >/dev/null 2>&1
}

if have_compose_v2; then
  echo "[DOCKER] 'docker compose' plugin is available."
else
  # No v2 yet; check engine version and possible v1
  if have_docker; then
    DOCKER_MAJOR="$(get_docker_major || echo 0)"
  else
    DOCKER_MAJOR=0
  fi

  if have_compose_v1; then
    if [ "$DOCKER_MAJOR" -ge 25 ]; then
      echo "[DOCKER] Legacy 'docker-compose' v1 detected with Docker Engine ${DOCKER_MAJOR}.x."
      echo "[DOCKER] This combination is known to be broken (KeyError: 'ContainerConfig')."
      echo "[DOCKER] Attempting to install the v2 plugin instead..."
    else
      echo "[DOCKER] Legacy 'docker-compose' v1 is available and Docker Engine < 25; using it is possible but v2 is still preferred."
    fi
  else
    echo "[DOCKER] Docker Compose not found; attempting installation..."
  fi

  # Try to install v2 plugin first
  if [[ "${PKG_MGR}" == "apt" ]]; then
    apt-get update -y || echo "[DOCKER] Warning: apt-get update failed."
    apt-get install -y docker-compose-plugin || {
      echo "[DOCKER] Failed to install docker-compose-plugin via apt; trying legacy docker-compose..." >&2
      apt-get install -y docker-compose || {
        echo "[DOCKER] Failed to install docker-compose or docker-compose-plugin via apt." >&2
      }
    }
  elif [[ "${PKG_MGR}" == "dnf" ]]; then
    dnf install -y docker-compose-plugin || {
      echo "[DOCKER] Failed to install docker-compose-plugin via dnf; trying legacy docker-compose..." >&2
      dnf install -y docker-compose || {
        echo "[DOCKER] Failed to install docker-compose or docker-compose-plugin via dnf." >&2
      }
    }
  elif [[ "${PKG_MGR}" == "yum" ]]; then
    yum install -y docker-compose-plugin || {
      echo "[DOCKER] Failed to install docker-compose-plugin via yum; trying legacy docker-compose..." >&2
      yum install -y docker-compose || {
        echo "[DOCKER] Failed to install docker-compose or docker-compose-plugin via yum." >&2
      }
    }
  else
    echo "[DOCKER] Cannot auto-install Docker Compose: unknown package manager. Please install manually." >&2
  fi

  # Final detection after installation attempts
  if have_compose_v2; then
    echo "[DOCKER] 'docker compose' plugin is now available."
  elif have_compose_v1; then
    if [ "$DOCKER_MAJOR" -ge 25 ]; then
      echo "[DOCKER] Warning: only legacy 'docker-compose' v1 is available with Docker Engine ${DOCKER_MAJOR}.x." >&2
      echo "[DOCKER] This may fail with KeyError: 'ContainerConfig'. Prefer installing the v2 plugin." >&2
    else
      echo "[DOCKER] Only legacy 'docker-compose' v1 is available; using it."
    fi
  else
    echo "[DOCKER] Warning: Docker Compose still not available after installation attempt." >&2
  fi
fi

echo "[DOCKER] Ensuring 'docker' group exists..."
if ! getent group docker >/dev/null 2>&1; then
  groupadd docker
  echo "[DOCKER] Created group 'docker'."
else
  echo "[DOCKER] Group 'docker' already exists."
fi

echo "[DOCKER] Adding user '$USER_NAME' to 'docker' group..."
usermod -aG docker "$USER_NAME"

echo
echo "[DOCKER] Done."
echo "User '$USER_NAME' has been added to the 'docker' group."
