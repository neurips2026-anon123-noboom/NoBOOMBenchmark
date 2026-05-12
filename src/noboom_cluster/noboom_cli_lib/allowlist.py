from __future__ import annotations

import base64
import json
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .specs import InventoryConfig


def parse_device_string(devices: Optional[str]) -> Optional[List[str]]:
    if devices is None:
        return None

    parsed = [device.strip() for device in devices.split(",") if device.strip()]
    if not parsed:
        return []
    return parsed


class AllowlistNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ip: str
    devices: Optional[List[str]] = None

    @classmethod
    def from_inventory_entry(cls, ip: str, devices: Optional[str]) -> "AllowlistNode":
        return cls(ip=ip, devices=parse_device_string(devices))

    def expanded_devices(self, visible_devices: Sequence[str]) -> List[str]:
        if self.devices is None:
            return list(visible_devices)
        return list(self.devices)


class AllowlistConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: List[AllowlistNode] = Field(default_factory=list)

    @classmethod
    def from_inventory(cls, inventory: InventoryConfig) -> "AllowlistConfig":
        return cls(
            nodes=[
                AllowlistNode.from_inventory_entry(node.ip, node.devices)
                for node in inventory.nodes
            ]
        )

    def to_base64(self) -> str:
        raw_payload = self.model_dump_json().encode("utf-8")
        return base64.urlsafe_b64encode(raw_payload).decode("ascii")

    @classmethod
    def from_base64(cls, payload: str) -> "AllowlistConfig":
        raw_payload = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
        return cls.model_validate_json(raw_payload)

    def to_node_map(self) -> Dict[str, Optional[List[str]]]:
        return {node.ip: list(node.devices) if node.devices is not None else None for node in self.nodes}


class ClusterNodeGpuView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ip: str
    node_id: str
    visible_gpu_ids: List[str] = Field(default_factory=list)


class TrialGpuLease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    ip: str
    node_id: str
    gpu_id: str
    gpu_fraction: float
    revision: int
    state: str = "reserved"
    owner_job_id: Optional[str] = None


class AllowlistStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = 0
    cluster_nodes: List[ClusterNodeGpuView] = Field(default_factory=list)
    allowed: Dict[str, List[str]] = Field(default_factory=dict)
    active_leases: List[TrialGpuLease] = Field(default_factory=list)
    draining_leases: List[TrialGpuLease] = Field(default_factory=list)


class AllowlistUpdateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    status: Optional[AllowlistStatus] = None
    errors: List[str] = Field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)
