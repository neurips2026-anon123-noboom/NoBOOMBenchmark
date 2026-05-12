set -euo pipefail

BASE="${BASE:-/tmp/ray}"
[[ -d "$BASE" ]] || exit 0

SL="$BASE/session_latest"

remove_path() {
  local path="$1"
  if rm -rf -- "$path" 2>/dev/null; then
    return 0
  fi

  printf 'Warning: failed to fully remove %s; continuing.\n' "$path" >&2
}

# If session_latest exists and is a symlink, resolve its target (absolute path)
TARGET=""
if [[ -L "$SL" ]]; then
  # readlink -f resolves to an absolute canonical path
  TARGET="$(readlink -f -- "$SL" || true)"
fi

# If session_latest exists but is NOT a symlink, delete all session_* dirs and exit
if [[ -e "$SL" && ! -L "$SL" ]]; then
  find "$BASE" -mindepth 1 -maxdepth 1 -name 'session_*' -print0 \
  | while IFS= read -r -d '' p; do
      remove_path "$p"
    done
  exit 0
fi

# Otherwise: delete everything except session_latest and (if present) its target dir
find "$BASE" -mindepth 1 -maxdepth 1 -print0 \
| while IFS= read -r -d '' p; do
    bn="$(basename -- "$p")"

    # Keep the session_latest symlink itself
    if [[ "$bn" == "session_latest" ]]; then
      continue
    fi

    # Keep the symlink target, but only if it's inside BASE and matches this entry
    if [[ -n "$TARGET" && "$TARGET" == "$BASE/"* && "$p" == "$TARGET" ]]; then
      continue
    fi

    remove_path "$p"
  done
