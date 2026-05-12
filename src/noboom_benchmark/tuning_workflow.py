import argparse
from argparse import Namespace
from typing import List, Optional
import logging
import sys
import warnings
from typing import Callable, Dict, Tuple

from dotenv import load_dotenv
from noboom_cluster.noboom_cli_lib.specs import PairRunSpec

logger = logging.getLogger(__name__)


def _load_pair_runner() -> Callable[[PairRunSpec], Tuple[str, str, Dict[str, object]]]:
    try:
        from .noboom_lib.core.tune.pair_execution import run_pair_spec
    except ImportError:  # pragma: no cover - compatibility for direct script execution
        from noboom_benchmark.noboom_lib.core.tune.pair_execution import run_pair_spec
    return run_pair_spec


def _load_legacy_runner() -> Callable[[Namespace], Tuple[str, str, Dict[str, object]]]:
    try:
        from .noboom_lib.core.tuning_runner import run_tune_or_train
    except ImportError:  # pragma: no cover - compatibility for direct script execution
        from noboom_benchmark.noboom_lib.core.tuning_runner import run_tune_or_train
    return run_tune_or_train


def parse_args(argv: Optional[List[str]] = None) -> Namespace:
    """Parse CLI arguments for the tuning workflow.

    Args:
        argv (Optional[List[str]]): Optional argv list. Defaults to None.

    Returns:
        Namespace: Parsed CLI arguments.
    """
    pair_parser = argparse.ArgumentParser(add_help=False)
    pair_parser.add_argument("--pair-spec", default=None, type=str)
    pair_parser.add_argument("--pair-spec-b64", default=None, type=str)
    pair_args, _ = pair_parser.parse_known_args(argv)
    required_legacy_args = pair_args.pair_spec is None and pair_args.pair_spec_b64 is None

    parser = argparse.ArgumentParser(description="Run tuning or evaluation for a single model/dataset pair.")
    parser.add_argument("--experiment-id", required=required_legacy_args, type=str)
    parser.add_argument("--source-experiment-id", default=None, type=str)
    parser.add_argument("--model-name", required=required_legacy_args, type=str)
    parser.add_argument("--dataset-name", required=required_legacy_args, type=str)
    parser.add_argument("--timestamp", required=required_legacy_args, type=str)
    parser.add_argument("--storage-path", required=required_legacy_args, type=str)
    parser.add_argument("--gpus-per-run", required=required_legacy_args, type=float)
    parser.add_argument("--optuna-storage-uri", required=required_legacy_args, type=str)
    parser.add_argument("--config-dir", default="configs", type=str)
    parser.add_argument("--verbose", default=0, type=int)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--env-file", type=str, default=None)
    parser.add_argument("--temp-dir", default="/tmp/ray", type=str)
    parser.add_argument("--prepared-dataset-s3-path", default=None, type=str)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--hpo-seed", dest="hpo_seeds", action="append", type=int, default=None)
    parser.add_argument("--execution-backend", choices=["ray", "local"], default="ray")
    parser.add_argument("--artifact-storage-backend", choices=["remote", "local"], default="remote")
    parser.add_argument(
        "--optuna-storage-backend",
        choices=["configured", "memory", "sqlite"],
        default="configured",
    )
    parser.add_argument("--pair-spec", default=None, type=str)
    parser.add_argument("--pair-spec-b64", default=None, type=str)

    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.pair_spec is not None or cli_args.pair_spec_b64 is not None:
        warnings.warn(
            "tuning_workflow.py is a compatibility shim. Pair jobs should keep using inline pair specs.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            if cli_args.pair_spec_b64 is not None:
                pair_spec = PairRunSpec.from_base64(cli_args.pair_spec_b64)
            else:
                pair_spec = PairRunSpec.load(cli_args.pair_spec)
            _load_pair_runner()(pair_spec)
            sys.exit(0)
        except Exception:
            logger.exception("run_pair_spec failed")
            sys.exit(1)

    if cli_args.env_file is not None:
        load_dotenv(cli_args.env_file)

    try:
        warnings.warn(
            "Legacy tuning_workflow.py CLI arguments are deprecated. Prefer --pair-spec.",
            DeprecationWarning,
            stacklevel=2,
        )
        _load_legacy_runner()(cli_args)
        sys.exit(0)
    except Exception:
        logger.exception("run_tune_or_train failed")
        sys.exit(1)
