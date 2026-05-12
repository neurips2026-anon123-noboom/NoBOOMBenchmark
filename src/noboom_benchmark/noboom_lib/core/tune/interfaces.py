from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from noboom_cluster.noboom_cli_lib.specs import PairResult, PairRunSpec


class PairOutputCallback(ABC):
    def on_job_start(self, pair_spec: PairRunSpec) -> None:
        del pair_spec
        return None

    @abstractmethod
    def on_job_terminal(
        self,
        pair_spec: PairRunSpec,
        status: str,
        job_status_message: Optional[str] = None,
    ) -> PairResult:
        raise NotImplementedError
