from __future__ import annotations

import sys
from typing import List, Optional

from noboom_cluster.cli import main as cluster_main


def main(argv: Optional[List[str]] = None) -> int:
    return cluster_main(argv, prog_name="run_cluster.py")


if __name__ == "__main__":
    sys.exit(main())
