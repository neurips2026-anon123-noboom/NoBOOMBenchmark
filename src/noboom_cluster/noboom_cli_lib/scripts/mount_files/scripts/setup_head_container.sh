#!/usr/bin/env bash
set -euo pipefail

# 1) Re-clone the repo into /workspace/NoBoomBenchmark
# cd /workspace

###############################################################################
# X) SeaweedFS S3 / rclone configuration for current user
###############################################################################
if [[ -n "${NOBOOM_SKIP_SEAWEED_S3_SETUP:-}" && "${NOBOOM_SKIP_SEAWEED_S3_SETUP}" != "0" ]]; then
  echo "[SKIP] Skipping SeaweedFS S3 rclone setup..."
else
  echo "[seaweed-s3] Checking SeaweedFS S3 rclone configuration..."

  REMOTE_NAME="${RCLONE_SEAWEED_S3_REMOTE_NAME:-seaweed_s3}"

  # SeaweedFS S3 endpoint (examples):
  #   http://127.0.0.1:8333
  #   http://seaweedfs-s3:8333
  #   https://seaweed.example.com:8333
  S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-http://${HEAD_LOCAL_IP}:8333}"
  S3_REGION="${S3_REGION:-us-east-1}"

  if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "[seaweed-s3] AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not set, skipping rclone setup."
  else
    if ! command -v rclone >/dev/null 2>&1; then
      echo "[seaweed-s3] rclone not found, skipping rclone setup."
    else
      mkdir -p "${HOME}/.config/rclone"

      if rclone listremotes 2>/dev/null | grep -qx "${REMOTE_NAME}:"; then
        echo "[seaweed-s3] rclone remote '${REMOTE_NAME}' already exists, nothing to do."
      else
        echo "[seaweed-s3] Creating rclone remote '${REMOTE_NAME}' for SeaweedFS S3..."

        # NOTE: rclone's S3 backend supports S3-compatible servers via provider=Other + endpoint=...
        rclone config create "${REMOTE_NAME}" s3 \
          provider="Other" \
          env_auth="true" \
          endpoint="${S3_ENDPOINT_URL}" \
          region="${S3_REGION}" \
          acl="private" \
          --non-interactive

        echo "[seaweed-s3] rclone remote '${REMOTE_NAME}' configured."
        echo "[seaweed-s3] Tip: use --s3-force-path-style when syncing/copying if you hit bucket/redirect issues."
      fi
    fi
  fi
fi


###############################################################################
# 3) Seafile / rclone configuration for current user
###############################################################################
if [[ -n "${NOBOOM_SKIP_SEAFILE_SETUP:-}" && "${NOBOOM_SKIP_SEAFILE_SETUP}" != "0" ]]; then
  echo "[SKIP] Skipping Seafile rclone setup..."
else
  echo "[seafile] Checking Seafile rclone configuration..."

  REMOTE_NAME="${RCLONE_SEAFILE_REMOTE_NAME:-seafile}"

  if [[ -z "${SEAFILE_USERNAME:-}" || -z "${SEAFILE_PASS:-}" ]]; then
    echo "[seafile] SEAFILE_USERNAME or SEAFILE_PASS not set, skipping rclone setup."
  else
    if ! command -v rclone >/dev/null 2>&1; then
      echo "[seafile] rclone not found, skipping rclone setup."
    else
      mkdir -p "${HOME}/.config/rclone"

      if rclone listremotes 2>/dev/null | grep -qx "${REMOTE_NAME}:"; then
        echo "[seafile] rclone remote '${REMOTE_NAME}' already exists, nothing to do."
      else
        echo "[seafile] Creating rclone remote '${REMOTE_NAME}' for Seafile WebDAV..."

        OBSCURED_PASS="$(rclone obscure -- "${SEAFILE_PASS}")"

        rclone config create "${REMOTE_NAME}" webdav \
          url="${NOBOOM_WEBDAV_URL:-https://anonymous.example.org/seafdav/}" \
          vendor="webdav" \
          user="${SEAFILE_USERNAME}" \
          pass="${OBSCURED_PASS}" \
          --non-interactive

        echo "[seafile] rclone remote '${REMOTE_NAME}' configured."
      fi
    fi
  fi
fi

#rm -rf /workspace/NoBoomBenchmark/*
#
## 4) Clone and checkout the desired branch using a token from env
#git clone "${NOBOOM_REPOSITORY_URL:-https://github.com/denix56/NoBoomBenchmark.git}" /workspace/NoBoomBenchmark
#git -C /workspace/NoBoomBenchmark checkout ray_cluster_tune_docker
#
## 5) Run run_tune.py with all arguments passed to this script
#cd /workspace/NoBoomBenchmark/src/noboom_benchmark
#echo "$@"
#exec python ./run_tune.py "$@" -v
