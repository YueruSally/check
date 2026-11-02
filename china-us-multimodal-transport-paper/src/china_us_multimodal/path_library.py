from __future__ import annotations

from dataclasses import dataclass, replace

from .domain import ModelData, NodeKind
from .martins import Label, candidate_paths


PathLibrary = dict[str, tuple[tuple[str, ...], ...]]


@dataclass(frozen=True)
class StructuralPathEnumeration:
    """Cycle-free paths satisfying only the model's structural restrictions.

    Unlike the Martins library, these paths are not removed by static
    cost--time dominance.  ``complete`` is false when the explicit safety cap
    is reached, so a capped enumeration can never be reported as exhaustive.
    """

    labels: tuple[Label, ...]
    complete: bool
    explored_labels: int
    enumeration_cap: int


def _gateway_subnetwork(data: ModelData, gateway_id: str) -> ModelData:
    """Keep one US gateway so dominated all-water alternatives remain represented."""
    arcs = {}
    for arc_id, arc in data.arcs.items():
        origin_kind = data.nodes[arc.origin].kind
        destination_kind = data.nodes[arc.destination].kind
        if destination_kind == NodeKind.US_PORT and arc.destination != gateway_id:
            continue
        if origin_kind == NodeKind.US_PORT and arc.origin != gateway_id:
            continue
        arcs[arc_id] = arc
    service_ids = {item for arc in arcs.values() for item in arc.service_ids}
    services = {
        service_id: service
        for service_id, service in data.services.items()
        if service_id in service_ids
    }
    return replace(data, arcs=arcs, services=services)


def enumerate_structural_paths(
    data: ModelData,
    origin: str,
    destination: str,
    *,
    max_legs: int,
    max_mode_changes: int,
    enumeration_cap: int = 100_000,
) -> StructuralPathEnumeration:
    """Enumerate simple paths without applying static dominance pruning."""
    if enumeration_cap <= 0:
        raise ValueError("enumeration_cap must be positive.")
    if origin not in data.nodes or destination not in data.nodes:
        raise KeyError("Unknown origin or destination.")
    outgoing = {
        node_id: tuple(sorted(arcs, key=lambda arc: arc.id))
        for node_id, arcs in data.outgoing().items()
    }
    start = Label(origin, 0.0, 0.0, (), frozenset({origin}), None, 0)
    stack = [start]
    complete: list[Label] = []
    explored = 0
    while stack:
        label = stack.pop()
        explored += 1
        if label.node_id == destination:
            if len(complete) >= enumeration_cap:
                return StructuralPathEnumeration(
                    tuple(sorted(complete, key=lambda item: (
                        item.cost_usd_per_feu, item.elapsed_h, item.arc_ids
                    ))),
                    False,
                    explored,
                    enumeration_cap,
                )
            complete.append(label)
            continue
        if len(label.arc_ids) >= max_legs:
            continue
        next_labels: list[Label] = []
        for arc in outgoing.get(label.node_id, ()):
            if arc.destination in label.visited_nodes:
                continue
            mode_changes = label.mode_changes + int(
                label.last_mode is not None and label.last_mode != arc.mode
            )
            if mode_changes > max_mode_changes:
                continue
            next_labels.append(
                Label(
                    node_id=arc.destination,
                    cost_usd_per_feu=label.cost_usd_per_feu + arc.cost_usd_per_feu,
                    elapsed_h=(
                        label.elapsed_h
                        + arc.transit_time_h
                        + arc.planning_wait_h
                    ),
                    arc_ids=label.arc_ids + (arc.id,),
                    visited_nodes=label.visited_nodes | {arc.destination},
                    last_mode=arc.mode,
                    mode_changes=mode_changes,
                )
            )
        stack.extend(reversed(next_labels))
    return StructuralPathEnumeration(
        tuple(sorted(complete, key=lambda item: (
            item.cost_usd_per_feu, item.elapsed_h, item.arc_ids
        ))),
        True,
        explored,
        enumeration_cap,
    )


def enumerate_gateway_structural_paths(
    data: ModelData,
    origin: str,
    destination: str,
    gateway_id: str,
    *,
    max_legs: int,
    max_mode_changes: int,
    enumeration_cap: int = 100_000,
) -> StructuralPathEnumeration:
    """Enumerate all structurally feasible paths through one US gateway."""
    if gateway_id not in data.nodes or data.nodes[gateway_id].kind != NodeKind.US_PORT:
        raise ValueError(f"Unknown US gateway: {gateway_id!r}.")
    return enumerate_structural_paths(
        _gateway_subnetwork(data, gateway_id),
        origin,
        destination,
        max_legs=max_legs,
        max_mode_changes=max_mode_changes,
        enumeration_cap=enumeration_cap,
    )


def gateway_martins_paths(
    data: ModelData,
    origin: str,
    destination: str,
    gateway_id: str,
    *,
    max_legs: int,
    max_mode_changes: int,
    max_paths: int,
) -> tuple[Label, ...]:
    """Expose the gateway-stratified Martins result for coverage auditing."""
    return tuple(
        candidate_paths(
            _gateway_subnetwork(data, gateway_id),
            origin,
            destination,
            max_legs=max_legs,
            max_mode_changes=max_mode_changes,
            max_paths=max_paths,
        )
    )


def build_gateway_aware_library(
    data: ModelData,
    *,
    max_legs: int,
    max_mode_changes: int,
    max_paths_per_gateway: int = 4,
    disabled_gateways: set[str] | None = None,
) -> PathLibrary:
    """Run Martins separately by US gateway, then merge unique candidate paths."""
    excluded = disabled_gateways or set()
    gateways = sorted(
        node_id
        for node_id, node in data.nodes.items()
        if node.kind == NodeKind.US_PORT and node_id not in excluded
    )
    if not gateways:
        raise ValueError("The pilot network has no US gateway nodes.")

    od_cache: dict[tuple[str, str], tuple[tuple[str, ...], ...]] = {}
    library: PathLibrary = {}
    for shipment_id in sorted(data.shipments):
        shipment = data.shipments[shipment_id]
        od = (shipment.origin, shipment.destination)
        if od not in od_cache:
            found: dict[tuple[str, ...], tuple[float, float]] = {}
            for gateway_id in gateways:
                subnetwork = _gateway_subnetwork(data, gateway_id)
                labels = candidate_paths(
                    subnetwork,
                    shipment.origin,
                    shipment.destination,
                    max_legs=max_legs,
                    max_mode_changes=max_mode_changes,
                    max_paths=max_paths_per_gateway,
                )
                for label in labels:
                    found[label.arc_ids] = (label.cost_usd_per_feu, label.elapsed_h)
            if not found:
                raise ValueError(
                    f"No Martins candidate path for OD {shipment.origin}->{shipment.destination}."
                )
            od_cache[od] = tuple(
                path for path, _ in sorted(found.items(), key=lambda item: (*item[1], item[0]))
            )
        library[shipment_id] = od_cache[od]
    return library
