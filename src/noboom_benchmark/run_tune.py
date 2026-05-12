"""CLI utilities for orchestrating NoBoom Optuna hyperparameter studies."""
from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import datetime
import logging
import os
import sys
import warnings
from typing import List, Optional

try:
    from .noboom_lib.core.tune.orchestration import run_tune
    from .noboom_lib.core.notifications import ensure_telegram_enrollment_env
    from noboom_cluster.noboom_cli_lib.specs import parse_dataset_model_pair
except ImportError:  # pragma: no cover - compatibility for direct script execution
    from noboom_benchmark.noboom_lib.core.tune.orchestration import run_tune
    from noboom_benchmark.noboom_lib.core.notifications import ensure_telegram_enrollment_env
    from noboom_cluster.noboom_cli_lib.specs import parse_dataset_model_pair


def _pair_argument(raw_pair: str) -> str:
    try:
        parse_dataset_model_pair(raw_pair)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc
    return raw_pair


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=[],
        help="Model name. Multiple models can be specified.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=[],
        help="Dataset name. Multiple datasets can be specified.",
    )
    parser.add_argument(
        "--pair",
        type=_pair_argument,
        action="append",
        default=[],
        help="Explicit dataset-model pair in DATASET:MODEL format. May be repeated.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (INFO by default, DEBUG with -v).",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing params/common.yaml.",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="/tmp/ray",
        help="Temporary directory for storing Ray logs and checkpoints.",
    )
    parser.add_argument(
        "--gpus-per-run",
        type=float,
        required=True,
        help="GPUs to allocate per single-train run.",
    )

    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Timestamp to use for experiment name and storage paths. Defaults to current time.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (default: true).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Source MLflow experiment ID for single-train parameter/checkpoint lookup.",
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Environment file to use. (Important for native mode.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=4,
        help="Maximum number of pair jobs to keep in flight.",
    )
    parser.add_argument(
        "--prepared-dataset-s3-path",
        type=str,
        default=None,
        help=(
            "Optional S3 root for prepared dataset bundles. Accepts either "
            "'s3://bucket/prefix' or a prefix relative to NOBOOM_S3_BUCKET."
        ),
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save and log full Lightning checkpoint artifacts.",
    )
    parser.add_argument(
        "--hpo-seed",
        dest="hpo_seeds",
        action="append",
        type=int,
        default=None,
        help=(
            "Seed to use during HPO. Repeat to provide multiple seeds. "
            "Overrides study.hpo_seeds when set."
        ),
    )
    parser.add_argument(
        "--execution-backend",
        choices=["ray", "local"],
        default="ray",
        help="Execution backend for model/dataset pair jobs. Defaults to Ray.",
    )
    parser.add_argument(
        "--artifact-storage-backend",
        choices=["remote", "local"],
        default=None,
        help=(
            "Artifact/result storage backend. Defaults to remote for Ray and local "
            "for --execution-backend=local."
        ),
    )
    parser.add_argument(
        "--local-storage-path",
        type=str,
        default=None,
        help="Local filesystem root for sequential/local benchmark outputs.",
    )
    parser.add_argument(
        "--optuna-storage-backend",
        choices=["configured", "memory", "sqlite"],
        default=None,
        help=(
            "Optuna storage for local/sequential runs. Defaults to configured storage "
            "for Ray and in-memory storage for --execution-backend=local."
        ),
    )
    parser.add_argument(
        "--optuna-sqlite-path",
        type=str,
        default=None,
        help="SQLite database path used when --optuna-storage-backend=sqlite.",
    )
    ram_guard_default = os.getenv("NOBOOM_RAM_GUARD_ENABLED", "1").strip().lower() in {
        "1",
        "on",
        "true",
        "yes",
    }
    parser.add_argument("--ram-guard", dest="ram_guard", action="store_true", default=ram_guard_default)
    parser.add_argument("--no-ram-guard", dest="ram_guard", action="store_false")
    parser.add_argument(
        "--ram-guard-max-used-fraction",
        type=float,
        default=float(os.getenv("NOBOOM_RAM_GUARD_MAX_USED_FRACTION", "0.95")),
    )
    parser.add_argument(
        "--ram-guard-poll-interval-s",
        type=float,
        default=float(os.getenv("NOBOOM_RAM_GUARD_POLL_INTERVAL_S", "10.0")),
    )
    parser.add_argument(
        "--ram-guard-resume-used-fraction",
        type=float,
        default=float(os.getenv("NOBOOM_RAM_GUARD_RESUME_USED_FRACTION", "0.90")),
    )
    parser.add_argument(
        "--ram-guard-cooldown-s",
        type=float,
        default=float(os.getenv("NOBOOM_RAM_GUARD_COOLDOWN_S", "120.0")),
    )
    parser.add_argument(
        "--notify-channel",
        action="append",
        choices=["email", "telegram"],
        default=[],
    )
    parser.add_argument("--notify-email-to", action="append", default=[])
    parser.add_argument("--notify-telegram-chat-id", default=None)
    return parser


def parse_args(argv: Optional[List[str]] = None) -> Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.pair and (not args.dataset or not args.model):
        parser.error("Provide --dataset with --model, one or more --pair values, or both.")
    channels = {channel.strip().lower() for channel in args.notify_channel or []}
    if "email" in channels:
        os.environ["NOBOOM_NOTIFY_EMAIL_ENABLED"] = "1"
        recipients = [
            item.strip()
            for raw in args.notify_email_to or []
            for item in str(raw).split(",")
            if item.strip()
        ]
        if recipients:
            os.environ["NOBOOM_NOTIFY_EMAIL_TO"] = ",".join(recipients)
            os.environ["NOBOOM_NOTIFY_SMTP_TO"] = ",".join(recipients)
    if "telegram" in channels or os.getenv("NOBOOM_NOTIFY_TELEGRAM_ENABLED") == "1":
        os.environ["NOBOOM_NOTIFY_TELEGRAM_ENABLED"] = "1"
        link = ensure_telegram_enrollment_env(os.environ)
        print(f"Telegram notifications enrollment link: {link}", flush=True)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """Parse CLI options and launch Optuna studies across Ray-managed GPUs.

    Returns:
        int: Exit code returned by the tuning orchestration.

    Side Effects:
        Parses CLI arguments and may launch distributed Ray workloads.
    """
    warnings.warn(
        "run_tune.py is a compatibility shim. Prefer the new cluster/controller flow.",
        DeprecationWarning,
        stacklevel=2,
    )
    args = parse_args(argv)
    return run_tune(args)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    sys.exit(main())
