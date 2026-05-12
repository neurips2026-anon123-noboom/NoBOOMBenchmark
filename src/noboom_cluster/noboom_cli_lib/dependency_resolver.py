from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from .specs import DependencyManifest

logger = logging.getLogger(__name__)

NOBOOM_REPOSITORY_URL = os.environ.get("NOBOOM_REPOSITORY_URL", "https://github.com/denix56/noboom.git")
TIMESEAD_REPOSITORY_URL = os.environ.get(
    "TIMESEAD_REPOSITORY_URL",
    "https://github.com/denix56/TimeSeAD.git",
)
TIMESEAD_EXTENSIONS_REPOSITORY_URL = os.environ.get(
    "TIMESEAD_EXTENSIONS_REPOSITORY_URL",
    "https://github.com/denix56/TimeSeAD-extensions.git",
)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def resolve_remote_sha(repo_url: str, ref: str = "HEAD") -> str:
    output = subprocess.check_output(
        ["git", "ls-remote", repo_url, ref],
        text=True,
    ).strip()
    if not output:
        raise RuntimeError(f"No output from git ls-remote for {repo_url} {ref}")
    return output.split()[0]


class DependencyResolver:
    def __init__(self, extensions_ref: str = "HEAD") -> None:
        self._extensions_ref = extensions_ref

    def resolve(self) -> DependencyManifest:
        logger.info("Resolving dependency SHAs with git ls-remote.")
        manifest = DependencyManifest(
            noboom_sha=resolve_remote_sha(NOBOOM_REPOSITORY_URL),
            timesead_sha=resolve_remote_sha(TIMESEAD_REPOSITORY_URL),
            timesead_extensions_sha=resolve_remote_sha(
                TIMESEAD_EXTENSIONS_REPOSITORY_URL,
                ref=self._extensions_ref,
            ),
        )
        logger.info(
            "Resolved dependency manifest: noboom=%s timesead=%s timesead_extensions=%s",
            manifest.noboom_sha,
            manifest.timesead_sha,
            manifest.timesead_extensions_sha,
        )
        return manifest

    def write(self, output_path: str, manifest: Optional[DependencyManifest] = None) -> str:
        resolved_manifest = manifest or self.resolve()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resolved_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return str(path)

    @staticmethod
    def load(path: str) -> DependencyManifest:
        with Path(path).open("r", encoding="utf-8") as handle:
            return DependencyManifest.model_validate(json.load(handle))
