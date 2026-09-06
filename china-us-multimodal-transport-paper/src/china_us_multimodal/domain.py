from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    ROAD = "road"
    RAIL = "rail"
    OCEAN = "ocean"


class NodeKind(str, Enum):
    CN_ORIGIN = "cn_origin"
    CN_PORT = "cn_port"
    CANAL = "canal"
    US_PORT = "us_port"
    US_HUB = "us_hub"
    DESTINATION = "destination"


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    kind: NodeKind
    country: str
    capacity_feu_per_day: float | None = None


@dataclass(frozen=True)
class Arc:
    id: str
    origin: str
    destination: str
    mode: Mode
    transit_time_h: float
    cost_usd_per_feu: float
    distance_km: float | None = None
    capacity_feu_per_day: float | None = None
    service_ids: tuple[str, ...] = ()
    planning_wait_h: float = 0.0
    via_panama: bool = False


@dataclass(frozen=True)
class Service:
    id: str
    arc_id: str
    departures_h: tuple[float, ...]
    repeat_every_h: float
    capacity_feu: float
    cutoff_h: float = 0.0


@dataclass(frozen=True)
class TransferRule:
    node_id: str
    from_mode: Mode
    to_mode: Mode
    duration_h: float
    cost_usd_per_feu: float


@dataclass(frozen=True)
class Shipment:
    id: str
    origin: str
    destination: str
    quantity_feu: float
    release_h: float = 0.0
    due_h: float | None = None
    tardiness_usd_per_feu_h: float | None = None
    case_id: str | None = None


@dataclass(frozen=True)
class RouteAllocation:
    shipment_id: str
    quantity_feu: float
    arc_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModelData:
    nodes: dict[str, Node]
    arcs: dict[str, Arc]
    services: dict[str, Service]
    shipments: dict[str, Shipment]
    transfer_rules: dict[tuple[str, Mode, Mode], TransferRule] = field(default_factory=dict)

    def outgoing(self) -> dict[str, list[Arc]]:
        result: dict[str, list[Arc]] = {node_id: [] for node_id in self.nodes}
        for arc in self.arcs.values():
            result.setdefault(arc.origin, []).append(arc)
        return result
