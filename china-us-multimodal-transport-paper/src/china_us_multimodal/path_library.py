from __future__ import annotations

from dataclasses import replace

from .domain import ModelData, NodeKind
from .martins import candidate_paths


PathLibrary = dict[str, tuple[tuple[str, ...], ...]]


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


def build_gateway_aware_library(
    data: ModelData,
    *,
    max_legs: int,
    max_mode_changes: int,
    max_paths_per_gateway: int = 4,
) -> PathLibrary:
    """Run Martins separately by US gateway, then merge unique candidate paths."""
    gateways = sorted(
        node_id for node_id, node in data.nodes.items() if node.kind == NodeKind.US_PORT
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
