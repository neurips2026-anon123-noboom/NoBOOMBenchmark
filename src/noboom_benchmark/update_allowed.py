from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import ray

from noboom_cluster.noboom_cli_lib.allowlist import AllowlistConfig

from .noboom_lib.core.tune.runtime_allowlist import get_runtime_gpu_allowlist_actor

RESULT_PREFIX = "ALLOWLIST_UPDATE_RESULT="

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a live node/GPU allowlist update.")
    parser.add_argument("--allowlist-b64", required=True, type=str)
    args = parser.parse_args(argv)

    ray.init("auto")
    response = ray.get(
        get_runtime_gpu_allowlist_actor().apply_allowlist.remote(
            AllowlistConfig.from_base64(args.allowlist_b64)
        )
    )
    print(f"{RESULT_PREFIX}{response.to_json()}")
    return 0 if response.accepted else 2


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    sys.exit(main())
