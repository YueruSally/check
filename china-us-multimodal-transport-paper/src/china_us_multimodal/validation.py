from __future__ import annotations

from collections import deque

from .domain import ModelData, NodeKind


class DataValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


_ALLOWED_STAGE_TRANSITIONS: set[tuple[NodeKind, NodeKind]] = {
    (NodeKind.CN_ORIGIN, NodeKind.CN_ORIGIN),
    (NodeKind.CN_ORIGIN, NodeKind.CN_PORT),
    (NodeKind.CN_PORT, NodeKind.CN_PORT),
    (NodeKind.CN_PORT, NodeKind.CANAL),
    (NodeKind.CN_PORT, NodeKind.US_PORT),
    (NodeKind.CANAL, NodeKind.US_PORT),
    (NodeKind.US_PORT, NodeKind.US_PORT),
    (NodeKind.US_PORT, NodeKind.US_HUB),
    (NodeKind.US_PORT, NodeKind.DESTINATION),
    (NodeKind.US_HUB, NodeKind.US_HUB),
    (NodeKind.US_HUB, NodeKind.DESTINATION),
}


def _reachable(data: ModelData, origin: str, destination: str) -> bool:
    outgoing = data.outgoing()
    queue = deque([origin])
    seen = {origin}
    while queue:
        node = queue.popleft()
        if node == destination:
            return True
        for arc in outgoing.get(node, []):
            if arc.destination not in seen:
                seen.add(arc.destination)
                queue.append(arc.destination)
    return False


def validate_model_data(data: ModelData) -> None:
    errors: list[str] = []
    for node_id, node in data.nodes.items():
        if node_id != node.id:
            errors.append(f"Node key {node_id!r} differs from node.id {node.id!r}.")
        if node.capacity_feu_per_day is not None and node.capacity_feu_per_day <= 0:
            errors.append(f"Node {node.id} capacity must be positive.")

    for arc_id, arc in data.arcs.items():
        if arc_id != arc.id:
            errors.append(f"Arc key {arc_id!r} differs from arc.id {arc.id!r}.")
        if arc.origin not in data.nodes or arc.destination not in data.nodes:
            errors.append(f"Arc {arc.id} refers to an unknown endpoint.")
            continue
        if arc.transit_time_h <= 0 or arc.cost_usd_per_feu < 0:
            errors.append(f"Arc {arc.id} has invalid time or cost.")
        if arc.capacity_feu_per_day is not None and arc.capacity_feu_per_day <= 0:
            errors.append(f"Arc {arc.id} daily capacity must be positive.")
        stage = (data.nodes[arc.origin].kind, data.nodes[arc.destination].kind)
        if stage not in _ALLOWED_STAGE_TRANSITIONS:
            errors.append(
                f"Arc {arc.id} violates the China-US stage sequence: "
                f"{stage[0].value} -> {stage[1].value}."
            )
        for service_id in arc.service_ids:
            if service_id not in data.services:
                errors.append(f"Arc {arc.id} references missing service {service_id!r}.")

    for service_id, service in data.services.items():
        if service_id != service.id:
            errors.append(f"Service key {service_id!r} differs from service.id {service.id!r}.")
        if service.arc_id not in data.arcs:
            errors.append(f"Service {service.id} refers to missing arc {service.arc_id!r}.")
        elif service.id not in data.arcs[service.arc_id].service_ids:
            errors.append(f"Service {service.id} is not listed by arc {service.arc_id}.")
        if service.capacity_feu <= 0:
            errors.append(f"Service {service.id} capacity must be positive.")
        if service.repeat_every_h <= 0 or not service.departures_h:
            errors.append(f"Service {service.id} needs a valid recurring timetable.")
        if any(value < 0 or value >= service.repeat_every_h for value in service.departures_h):
            errors.append(f"Service {service.id} departures must lie within one period.")

    for shipment_id, shipment in data.shipments.items():
        if shipment_id != shipment.id:
            errors.append(f"Shipment key {shipment_id!r} differs from shipment.id {shipment.id!r}.")
        if shipment.origin not in data.nodes or shipment.destination not in data.nodes:
            errors.append(f"Shipment {shipment.id} refers to an unknown endpoint.")
            continue
        if shipment.quantity_feu <= 0 or shipment.release_h < 0:
            errors.append(f"Shipment {shipment.id} has invalid quantity or release time.")
        if shipment.due_h is not None and shipment.due_h < shipment.release_h:
            errors.append(f"Shipment {shipment.id} due time precedes release time.")
        if (
            shipment.tardiness_usd_per_feu_h is not None
            and shipment.tardiness_usd_per_feu_h < 0
        ):
            errors.append(f"Shipment {shipment.id} has a negative tardiness rate.")
        if not _reachable(data, shipment.origin, shipment.destination):
            errors.append(f"Shipment {shipment.id} has no topological route.")

    for key, rule in data.transfer_rules.items():
        if key != (rule.node_id, rule.from_mode, rule.to_mode):
            errors.append(f"Transfer-rule key {key!r} does not match its fields.")
        if rule.node_id not in data.nodes:
            errors.append(f"Transfer rule refers to missing node {rule.node_id!r}.")
        if rule.duration_h < 0 or rule.cost_usd_per_feu < 0:
            errors.append(f"Transfer rule {key!r} has a negative value.")

    if errors:
        raise DataValidationError(errors)
