#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import yaml


def run_command(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def get_local_ip() -> str:
    route = run_command(
        ["bash", "-lc", 'ip route get 1.1.1.1 2>/dev/null | awk \'/src/ {print $7; exit}\'']
    )
    if route.stdout.strip():
        return route.stdout.strip()
    hostname = run_command(["bash", "-lc", "hostname -I | awk '{print $1}'"])
    return hostname.stdout.strip()


def load_bootstrap_ips(bootstrap_path: Path) -> tuple[str, list[str]]:
    if not bootstrap_path.exists():
        return "", []
    data = yaml.safe_load(bootstrap_path.read_text()) or {}
    provider = data.get("provider") or {}
    head_ip = provider.get("head_ip") or ""
    worker_ips = provider.get("worker_ips") or []
    if not isinstance(worker_ips, list):
        worker_ips = []
    return str(head_ip), [str(ip) for ip in worker_ips]


def main() -> int:
    if os.getenv("NOBOOM_SKIP_FIREWALL"):
        print("[SKIP] Skipping firewall configuration because NOBOOM_SKIP_FIREWALL is set.")
        return 0

    if shutil.which("ufw") is None:
        print("[WARN] ufw is not installed; skipping firewall configuration.")
        return 0

    bootstrap_config = Path(
        os.getenv("RAY_BOOTSTRAP_CONFIG", str(Path.home() / "ray_bootstrap_config.yaml"))
    )
    head_ip, worker_ips = load_bootstrap_ips(bootstrap_config)

    if not head_ip and os.getenv("RAY_HEAD_IP"):
        head_ip = os.environ["RAY_HEAD_IP"]

    local_ip = get_local_ip()
    role = "unknown"
    if head_ip and local_ip == head_ip:
        role = "head"
    elif local_ip in worker_ips:
        role = "worker"
    elif head_ip:
        role = "worker"

    if role == "unknown":
        print("[WARN] Unable to determine node role; skipping ufw updates.")
        return 0

    print(f"[INFO] Local IP: {local_ip}")
    print(f"[INFO] Head IP: {head_ip or '<unknown>'}")
    print(f"[INFO] Worker IPs: {' '.join(worker_ips) or '<none>'}")
    print(f"[INFO] Node role: {role}")

    if role == "head":
        if worker_ips:
            print("[INFO] Adding ufw rules on head for worker IPs.")
            for ip in worker_ips:
                run_command(["sudo", "ufw", "allow", "from", ip])
        else:
            print("[INFO] No worker IPs available; skipping ufw rules on head.")
    else:
        if head_ip:
            print("[INFO] Adding ufw rule on worker for head IP.")
            run_command(["sudo", "ufw", "allow", "from", head_ip])
        else:
            print("[WARN] Head IP unavailable; skipping ufw rule on worker.")

    run_command(["sudo", "ufw", "status", "verbose"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
