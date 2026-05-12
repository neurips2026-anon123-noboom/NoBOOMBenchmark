#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import pwd
import subprocess
from typing import Dict, List, Optional, Set


logger = logging.getLogger(__name__)


def _run(command: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        logger.warning("Command failed with exit code %s: %s", result.returncode, " ".join(command))
        if result.stderr.strip():
            logger.warning(result.stderr.strip())
        return None
    return result.stdout


def _read_env(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _upsert_env(env_path: Path, key: str, value: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[index] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    if env_path.exists():
        os.chmod(env_path, 0o600)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(env_path, 0o400)


def _current_user() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return str(os.getuid())


def _gpu_index_by_uuid() -> Dict[str, str]:
    output = _run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    if output is None:
        return {}

    by_uuid: Dict[str, str] = {}
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        by_uuid[parts[1]] = parts[0]
    return by_uuid


def _all_gpu_indices() -> List[str]:
    output = _run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"])
    if output is None:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _process_owner(pid: str) -> Optional[str]:
    output = _run(["ps", "-o", "user=", "-p", pid])
    if output is None:
        return None
    owner = output.strip()
    return owner or None


def _busy_gpus_by_other_users(gpu_index_by_uuid: Dict[str, str]) -> Dict[str, Set[str]]:
    output = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return {}

    current_user = _current_user()
    busy: Dict[str, Set[str]] = {}
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 2:
            continue
        gpu_uuid, pid = parts[0], parts[1]
        gpu_id = gpu_index_by_uuid.get(gpu_uuid)
        if gpu_id is None:
            continue
        owner = _process_owner(pid)
        if owner is None or owner == current_user:
            continue
        busy.setdefault(gpu_id, set()).add(owner)
    return busy


def _parse_devices(raw_devices: Optional[str]) -> List[str]:
    if raw_devices is None or raw_devices.strip() == "":
        return _all_gpu_indices()
    return [item.strip() for item in raw_devices.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove GPUs used by other users from CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--env-file", required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    env = _read_env(args.env_file)
    selected_devices = _parse_devices(env.get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES")))
    if not selected_devices:
        logger.info("No selected GPUs were found; leaving CUDA_VISIBLE_DEVICES empty.")
        _upsert_env(args.env_file, "CUDA_VISIBLE_DEVICES", "")
        return 0

    gpu_index_by_uuid = _gpu_index_by_uuid()
    if not gpu_index_by_uuid:
        logger.warning("Unable to inspect GPU UUIDs with nvidia-smi; keeping CUDA_VISIBLE_DEVICES=%s.", ",".join(selected_devices))
        return 0

    busy_by_gpu = _busy_gpus_by_other_users(gpu_index_by_uuid)
    allowed_devices: List[str] = []
    for device in selected_devices:
        owners = busy_by_gpu.get(device)
        if owners:
            logger.warning(
                "Excluding GPU %s from CUDA_VISIBLE_DEVICES because it is used by other user(s): %s.",
                device,
                ", ".join(sorted(owners)),
            )
            continue
        allowed_devices.append(device)

    if not allowed_devices:
        logger.warning("All selected GPUs are busy by other users; Ray will start with no visible GPUs.")

    new_value = ",".join(allowed_devices)
    _upsert_env(args.env_file, "CUDA_VISIBLE_DEVICES", new_value)
    logger.info("CUDA_VISIBLE_DEVICES after busy-GPU filtering: %s", new_value or "<empty>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
