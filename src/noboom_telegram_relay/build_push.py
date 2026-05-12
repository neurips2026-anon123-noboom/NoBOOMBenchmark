#!/usr/bin/env python3
"""Build and optionally push the standalone Telegram relay Docker image."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

RELAY_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = "ghcr.io/denix56/noboom-telegram-relay:latest"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image tag.")
    parser.add_argument("--platform", default=None, help="Optional docker build platform.")
    parser.add_argument("--push", action="store_true", help="Push the image after build.")
    parser.add_argument("--no-cache", action="store_true", help="Disable Docker build cache.")
    return parser


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(RELAY_DIR), check=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    build_command = ["docker", "build", "-f", str(RELAY_DIR / "Dockerfile"), "-t", args.image]
    if args.platform:
        build_command.extend(["--platform", args.platform])
    if args.no_cache:
        build_command.append("--no-cache")
    build_command.append(".")
    run(build_command)
    if args.push:
        run(["docker", "push", args.image])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
