from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Optional

from ray.tune.execution.placement_groups import PlacementGroupFactory

from noboom_cluster.noboom_cli_lib.allowlist import AllowlistConfig, AllowlistNode, ClusterNodeGpuView, TrialGpuLease
from noboom_benchmark.noboom_lib.core.tune.runtime_allowlist import (
    LEASE_CONFIG_KEY,
    LeaseAwareTrialScheduler,
    NODE_PIN_RESOURCE_QUANTITY,
    RuntimeGpuAllowlistState,
    apply_cuda_visible_devices_for_lease,
)


def test_runtime_allowlist_state_initializes_and_expands_devices() -> None:
    state = RuntimeGpuAllowlistState()

    status = state.initialize(
        [
            ClusterNodeGpuView(ip="10.0.0.1", node_id="node-a", visible_gpu_ids=["0", "1"]),
            ClusterNodeGpuView(ip="10.0.0.2", node_id="node-b", visible_gpu_ids=["2"]),
        ],
        AllowlistConfig(
            nodes=[
                AllowlistNode(ip="10.0.0.1", devices=["1"]),
                AllowlistNode(ip="10.0.0.2", devices=None),
            ]
        ),
    )

    assert status.revision == 1
    assert status.allowed == {
        "10.0.0.1": ["1"],
        "10.0.0.2": ["2"],
    }


def test_runtime_allowlist_rejects_invalid_update_atomically() -> None:
    state = RuntimeGpuAllowlistState()
    state.initialize(
        [ClusterNodeGpuView(ip="10.0.0.1", node_id="node-a", visible_gpu_ids=["0", "1"])],
        AllowlistConfig(nodes=[AllowlistNode(ip="10.0.0.1", devices=None)]),
    )

    response = state.apply_allowlist(
        AllowlistConfig(
            nodes=[
                AllowlistNode(ip="10.0.0.1", devices=["9"]),
                AllowlistNode(ip="10.0.0.9", devices=None),
            ]
        )
    )

    assert response.accepted is False
    assert "Unknown node IP: 10.0.0.9." in response.errors
    assert "Node 10.0.0.1 requested unknown GPU IDs: 9." in response.errors
    assert state.get_status().allowed == {"10.0.0.1": ["0", "1"]}


def test_runtime_allowlist_drops_reserved_leases_and_drains_active_leases() -> None:
    state = RuntimeGpuAllowlistState()
    state.initialize(
        [ClusterNodeGpuView(ip="10.0.0.1", node_id="node-a", visible_gpu_ids=["0"])],
        AllowlistConfig(nodes=[AllowlistNode(ip="10.0.0.1", devices=None)]),
    )

    reserved = state.acquire_lease("trial-reserved", 1.0)
    assert reserved is not None

    response = state.apply_allowlist(AllowlistConfig(nodes=[]))

    assert response.accepted is True
    assert response.status is not None
    assert response.status.active_leases == []

    state = RuntimeGpuAllowlistState()
    state.initialize(
        [ClusterNodeGpuView(ip="10.0.0.1", node_id="node-a", visible_gpu_ids=["0"])],
        AllowlistConfig(nodes=[AllowlistNode(ip="10.0.0.1", devices=None)]),
    )
    active = state.acquire_lease("trial-active", 1.0)
    assert active is not None
    activated = state.activate_lease("trial-active")
    assert activated is not None
    assert activated.state == "active"

    response = state.apply_allowlist(AllowlistConfig(nodes=[]))

    assert response.accepted is True
    assert response.status is not None
    assert [lease.trial_id for lease in response.status.active_leases] == ["trial-active"]
    assert [lease.trial_id for lease in response.status.draining_leases] == ["trial-active"]


def test_runtime_allowlist_state_payload_round_trip() -> None:
    state = RuntimeGpuAllowlistState()
    state.initialize(
        [ClusterNodeGpuView(ip="10.0.0.1", node_id="node-a", visible_gpu_ids=["0", "1"])],
        AllowlistConfig(nodes=[AllowlistNode(ip="10.0.0.1", devices=["1"])]),
    )
    lease = state.acquire_lease("trial-1", 0.5)
    assert lease is not None
    state.activate_lease("trial-1")

    restored = RuntimeGpuAllowlistState()
    restored.load_payload(state.to_payload())

    assert restored.get_status() == state.get_status()


def test_apply_cuda_visible_devices_for_lease_sets_worker_gpu(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    apply_cuda_visible_devices_for_lease(
        TrialGpuLease(
            trial_id="trial-1",
            ip="10.0.0.1",
            node_id="node-a",
            gpu_id="1",
            gpu_fraction=1.0,
            revision=1,
        )
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


class _FakeRemoteMethod:
    def __init__(self, fn):
        self.remote = fn


class _FakeAllowlistActor:
    def __init__(self, lease: Optional[TrialGpuLease]) -> None:
        self.lease = lease
        self.acquired: List[tuple[str, float]] = []
        self.released: List[str] = []
        self.acquire_lease = _FakeRemoteMethod(self._acquire)
        self.release_lease = _FakeRemoteMethod(self._release)

    def _acquire(
        self,
        trial_id: str,
        gpu_fraction: float,
        owner_job_id: Optional[str] = None,
    ) -> Optional[TrialGpuLease]:
        del owner_job_id
        self.acquired.append((trial_id, gpu_fraction))
        return self.lease

    def _release(self, trial_id: str) -> None:
        self.released.append(trial_id)
        return None


class _FakeTrial:
    def __init__(self, trial_id: str, resources: Optional[dict[str, float]] = None) -> None:
        self.trial_id = trial_id
        self.config = {}
        self.placement_group_factory = PlacementGroupFactory(
            [resources or {"CPU": 1.0, "GPU": 1.0}],
            strategy="PACK",
        )
        self.invalidated = False

    def invalidate_json_state(self) -> None:
        self.invalidated = True

    def update_resources(self, resources) -> None:
        self.placement_group_factory = resources


class _FakeBaseScheduler:
    def __init__(self, trial) -> None:
        self.trial = trial
        self.metric = "score"
        self.supports_buffered_results = True

    def set_search_properties(self, metric, mode, **spec):
        return True

    def on_trial_add(self, tune_controller, trial) -> None:
        return None

    def on_trial_error(self, tune_controller, trial) -> None:
        return None

    def on_trial_result(self, tune_controller, trial, result):
        return "CONTINUE"

    def on_trial_complete(self, tune_controller, trial, result) -> None:
        return None

    def on_trial_remove(self, tune_controller, trial) -> None:
        return None

    def choose_trial_to_run(self, tune_controller):
        return self.trial

    def debug_string(self) -> str:
        return "fake"


def test_lease_aware_scheduler_assigns_trial_resources(monkeypatch) -> None:
    from noboom_benchmark.noboom_lib.core.tune import runtime_allowlist

    lease = TrialGpuLease(
        trial_id="trial-1",
        ip="10.0.0.1",
        node_id="node-a",
        gpu_id="3",
        gpu_fraction=0.75,
        revision=2,
    )
    fake_actor = _FakeAllowlistActor(lease)
    monkeypatch.setattr(runtime_allowlist, "get_runtime_gpu_allowlist_actor", lambda: fake_actor)
    monkeypatch.setattr(runtime_allowlist.ray, "get", lambda value: value)
    monkeypatch.setattr(runtime_allowlist, "_current_job_id", lambda: None)

    trial = _FakeTrial("trial-1", resources={"CPU": 3.0, "GPU": 0.75, "exclusive": 0.5})
    scheduler = LeaseAwareTrialScheduler(
        _FakeBaseScheduler(trial),
        gpu_fraction=0.25,
        base_resources={"CPU": 2.0, "exclusive": 0.001},
    )

    selected = scheduler.choose_trial_to_run(SimpleNamespace())

    assert selected is trial
    assert fake_actor.acquired == [("trial-1", 0.75)]
    assert trial.invalidated is True
    assert trial.config[LEASE_CONFIG_KEY]["gpu_id"] == "3"
    assert trial.config[LEASE_CONFIG_KEY]["gpu_fraction"] == 0.75
    assert trial.placement_group_factory.required_resources == {
        "CPU": 3.0,
        "exclusive": 0.5,
        "node:10.0.0.1": NODE_PIN_RESOURCE_QUANTITY,
    }


def test_lease_aware_scheduler_blocks_when_no_gpu_lease(monkeypatch) -> None:
    from noboom_benchmark.noboom_lib.core.tune import runtime_allowlist

    fake_actor = _FakeAllowlistActor(None)
    monkeypatch.setattr(runtime_allowlist, "get_runtime_gpu_allowlist_actor", lambda: fake_actor)
    monkeypatch.setattr(runtime_allowlist.ray, "get", lambda value: value)
    monkeypatch.setattr(runtime_allowlist, "_current_job_id", lambda: None)

    trial = _FakeTrial("trial-1", resources={"CPU": 2.0, "GPU": 0.5})
    scheduler = LeaseAwareTrialScheduler(
        _FakeBaseScheduler(trial),
        gpu_fraction=0.5,
        base_resources={"CPU": 2.0},
    )

    assert scheduler.choose_trial_to_run(SimpleNamespace()) is None
    assert fake_actor.acquired == [("trial-1", 0.5)]
