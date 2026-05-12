from __future__ import annotations

from types import SimpleNamespace
from typing import List

from ray.job_submission import JobStatus

from noboom_benchmark.noboom_lib.core.tune.node_ram_guard import (
    RamUsage,
    RayNodeRamGuard,
    memory_used_fraction,
    parse_linux_meminfo,
)


def test_parse_linux_meminfo_and_used_fraction() -> None:
    total, available = parse_linux_meminfo(
        """
        MemTotal:       1000 kB
        MemAvailable:    50 kB
        """
    )

    assert total == 1000 * 1024
    assert available == 50 * 1024
    assert memory_used_fraction(total, available) == 0.95


def test_ram_guard_stops_only_pending_pair_jobs_on_hot_node() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.stopped: List[str] = []

        def list_jobs(self):
            return [
                SimpleNamespace(
                    submission_id="pair-1",
                    status=JobStatus.RUNNING,
                    node_id="node-hot",
                ),
                SimpleNamespace(
                    submission_id="controller",
                    status=JobStatus.RUNNING,
                    node_id="node-hot",
                ),
                SimpleNamespace(
                    submission_id="pair-2",
                    status=JobStatus.RUNNING,
                    node_id="node-cool",
                ),
            ]

        def stop_job(self, submission_id: str) -> None:
            self.stopped.append(submission_id)

    class FakeGuard(RayNodeRamGuard):
        def probe_node_memory(self) -> list[RamUsage]:
            return [
                RamUsage("node-hot", 0.97, 100, 3),
                RamUsage("node-cool", 0.10, 100, 90),
            ]

    client = FakeClient()
    guard = FakeGuard(client=client, poll_interval_s=0.0, cooldown_s=0.0)

    stopped = guard.stop_hot_pending_jobs(
        {
            "pair-1": SimpleNamespace(pair_id="dataset__model"),
            "pair-2": SimpleNamespace(pair_id="dataset__other"),
        }
    )

    assert client.stopped == ["pair-1"]
    assert stopped[0].submission_id == "pair-1"
    assert stopped[0].pair_id == "dataset__model"
