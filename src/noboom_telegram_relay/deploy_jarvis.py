#!/usr/bin/env python3
"""Create a small CPU JarvisLabs relay instance and print/start relay commands."""
from __future__ import annotations

from argparse import ArgumentParser
import os
from typing import List, Optional


DEFAULT_IMAGE = "ghcr.io/denix56/noboom-telegram-relay:latest"
DEFAULT_NAME = "noboom-telegram-relay-cpu"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Relay Docker image.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="JarvisLabs instance name.")
    parser.add_argument("--storage-gb", type=int, default=20, help="Instance storage size.")
    parser.add_argument("--template", default="pytorch", help="JarvisLabs template name.")
    parser.add_argument("--port", type=int, default=8080, help="Relay HTTP port.")
    parser.add_argument("--dry-run", action="store_true", help="Print intended instance shape only.")
    return parser


def _require_jarvis_key() -> str:
    value = os.environ.get("JL_API_KEY", "").strip() or os.environ.get("JARVIS_API_TOKEN", "").strip()
    if not value:
        raise RuntimeError("Set JL_API_KEY before running this deploy helper.")
    return value


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(
            "Jarvis CPU relay instance: "
            f"name={args.name} instance_type=CPU num_cpus=1 storage={args.storage_gb} "
            f"template={args.template} http_ports={args.port} image={args.image}"
        )
        return 0
    from jlclient import jarvisclient
    from jlclient.jarvisclient import Instance

    jarvisclient.token = _require_jarvis_key()
    instance = Instance.create(
        "CPU",
        num_cpus=1,
        storage=args.storage_gb,
        template=args.template,
        name=args.name,
        http_ports=str(args.port),
    )
    print(instance)
    print("Copy src/noboom_telegram_relay to the instance and run python -m noboom_telegram_relay serve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
