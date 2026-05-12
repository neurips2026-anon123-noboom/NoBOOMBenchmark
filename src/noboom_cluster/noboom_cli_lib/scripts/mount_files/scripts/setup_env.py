#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    cmd = "ip route get 1.1.1.1 | awk '{for (i=1;i<=NF;i++) if ($i==\"src\") print $(i+1)}'"
    logger.debug("Resolving local IP with command: %s", cmd)
    output = os.popen(cmd).read().strip()
    logger.debug("Local IP command output: %s", output if output else "<empty>")
    return output


def upsert_env(env_path: Path, key: str, value: str) -> None:
    logger.debug("Upserting %s in env file: %s", key, env_path)
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    logger.debug("Loaded %d existing env lines.", len(lines))
    updated = False
    lines_org = lines.copy()
    for idx, line in enumerate(lines_org):
        if line.startswith(f"{key}="):
            if value is None:
                lines.pop(idx)
                logger.info("Removed %s from %s", key, env_path)
            else:
                lines[idx] = f"{key}={value}"
            updated = True
            break
    if not updated:
        if value is not None:
            lines.append(f"{key}={value}")
            logger.info("Added %s=%s to %s", key, value, env_path)
    else:
        logger.info("Updated %s=%s in %s", key, value, env_path)
    # Remove .base
    env_path = env_path.with_suffix("")
    if env_path.exists():
        os.chmod(env_path, 0o600)
    env_path.write_text("\n".join(lines) + "\n")
    os.chmod(env_path, 0o400)
    logger.debug("Wrote %d env lines to %s", len(lines), env_path)



def find_node_devices(machine_file: Path, local_ip: str) -> Optional[str]:
    logger.debug("Loading machine file: %s", machine_file)
    data = yaml.safe_load(machine_file.read_text()) or {}
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("Machine file must define a 'nodes' list.")
    logger.debug("Machine file contains %d nodes.", len(nodes))
    for node in nodes:
        if not isinstance(node, dict):
            logger.debug("Skipping non-dict node entry: %s", node)
            continue
        ip = node.get("ip")
        if str(ip) == local_ip:
            logger.info("Matched local IP %s in machine file.", local_ip)
            devices = node.get("devices")
            if devices:
                logger.info("Found CUDA devices for %s: %s", local_ip, devices)
                return str(devices)
            logger.info("No CUDA devices listed for %s", local_ip)
            return None
        logger.debug("Node IP %s does not match local IP %s", ip, local_ip)
    return None


def main() -> int:
    logging.basicConfig(
        filename=str(Path(__file__).parents[2] / "set_cuda_visible_devices.py.log"),
        filemode="w",  # "a" = append, "w" = overwrite
        level=logging.DEBUG,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    )

    mount_dir = Path(__file__).resolve().parents[1]
    root_dir = mount_dir.parent

    env_output = mount_dir / ".env.base"
    machine_file = mount_dir / "machine_nodes.yaml"
    mount_env_path = mount_dir / ".env"
    root_env_path = root_dir / ".env"

    logger.info("Using env output: %s", str(env_output))
    logger.info("Using machine file: %s", str(machine_file))

    if not machine_file.exists():
        raise SystemExit(f"Machine file not found at {machine_file}")

    local_ip = get_local_ip()
    if not local_ip:
        raise SystemExit("Unable to determine local IP address")
    logger.info("Local IP detected: %s", local_ip)

    devices = find_node_devices(machine_file, local_ip)
    upsert_env(env_output, "CUDA_VISIBLE_DEVICES", devices)
    if not mount_env_path.exists():
        raise SystemExit(f"Expected generated env file at {mount_env_path}")
    if root_env_path.exists():
        os.chmod(root_env_path, 0o600)
        root_env_path.unlink()
    shutil.copyfile(mount_env_path, root_env_path)
    os.chmod(root_env_path, 0o400)
    logger.info("Copied generated env file to %s", root_env_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
