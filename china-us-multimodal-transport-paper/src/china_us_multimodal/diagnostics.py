from __future__ import annotations

from collections import Counter
import math

from .config import ModelConfig
from .domain import ModelData, NodeKind, RouteAllocation
from .evaluator import (
    _scenario_arc_cost,
    _scenario_ocean_departure_delay_h,
    _scenario_port_capacity_multiplier,
)
from .schedule import ranked_departure


def _departure_rank(allocation: RouteAllocation, arc_index: int) -> int:
    if not allocation.service_departure_ranks:
        return 0
    if len(allocation.service_departure_ranks) != len(allocation.arc_ids):
        raise ValueError(
            "A route allocation must provide either no departure ranks or one per arc."
        )
    rank = int(allocation.service_departure_ranks[arc_index])
    if rank < 0:
        raise ValueError("Service-departure ranks must be non-negative.")
    return rank


def capacity_usage_rows(
    data: ModelData,
    allocations: list[RouteAllocation],
    config: ModelConfig,
) -> list[dict[str, object]]:
    """Reconstruct the capacity accounting used by ``evaluate_solution``.

    The returned rows expose service-departure, arc-day, and node-day utilization
    without changing the optimization objectives or feasibility rules.
    """
    service_usage: Counter[tuple[str, float]] = Counter()
    daily_arc_usage: Counter[tuple[str, int]] = Counter()
    daily_node_usage: Counter[tuple[str, int]] = Counter()

    for allocation in allocations:
        shipment = data.shipments[allocation.shipment_id]
        arcs = [data.arcs[arc_id] for arc_id in allocation.arc_ids]
        current_h = shipment.release_h
        previous = None
        for arc_index, arc in enumerate(arcs):
            if previous is not None:
                rule = data.transfer_rules.get((arc.origin, previous.mode, arc.mode))
                if rule is not None:
                    current_h += rule.duration_h

            departure_h = current_h
            if config.constraints.enforce_timetable and arc.service_ids:
                services = [data.services[service_id] for service_id in arc.service_ids]
                scheduled = ranked_departure(
                    services,
                    current_h,
                    rank=_departure_rank(allocation, arc_index),
                )
                scheduled_departure_h = scheduled.departure_h
                departure_h = (
                    scheduled_departure_h
                    + _scenario_ocean_departure_delay_h(arc, config)
                )
                service_usage[(scheduled.service_id, scheduled_departure_h)] += (
                    allocation.quantity_feu
                )

            day = math.floor(departure_h / 24)
            daily_arc_usage[(arc.id, day)] += allocation.quantity_feu
            daily_node_usage[(arc.origin, day)] += allocation.quantity_feu
            current_h = departure_h + arc.transit_time_h
            previous = arc

    rows: list[dict[str, object]] = []
    for (service_id, departure_h), used in sorted(service_usage.items()):
        service = data.services[service_id]
        arc = data.arcs[service.arc_id]
        multiplier = (
            config.scenario.panama_availability_multiplier if arc.via_panama else 1.0
        )
        limit = service.capacity_feu * multiplier
        rows.append(
            _usage_row(
                "service_departure",
                service_id,
                f"{departure_h:g}",
                used,
                limit,
                arc.origin,
                arc.destination,
            )
        )

    for (arc_id, day), used in sorted(daily_arc_usage.items()):
        arc = data.arcs[arc_id]
        if arc.capacity_feu_per_day is None:
            continue
        rows.append(
            _usage_row(
                "arc_day",
                arc_id,
                str(day),
                used,
                arc.capacity_feu_per_day,
                arc.origin,
                arc.destination,
            )
        )

    for (node_id, day), used in sorted(daily_node_usage.items()):
        node = data.nodes[node_id]
        if node.capacity_feu_per_day is None:
            continue
        limit = node.capacity_feu_per_day
        if node.kind.value in {"cn_port", "us_port"}:
            limit *= _scenario_port_capacity_multiplier(node_id, config)
        rows.append(
            _usage_row(
                "node_day",
                node_id,
                str(day),
                used,
                limit,
                node_id,
                "",
            )
        )

    return rows


def _usage_row(
    resource_type: str,
    resource_id: str,
    period: str,
    used_feu: float,
    capacity_feu: float,
    origin: str,
    destination: str,
) -> dict[str, object]:
    utilization = used_feu / capacity_feu if capacity_feu > 0 else math.inf
    return {
        "resource_type": resource_type,
        "resource_id": resource_id,
        "period": period,
        "origin": origin,
        "destination": destination,
        "used_feu": used_feu,
        "capacity_feu": capacity_feu,
        "slack_feu": capacity_feu - used_feu,
        "utilization_pct": 100.0 * utilization,
        "binding_90pct": int(utilization >= 0.90),
        "binding_95pct": int(utilization >= 0.95),
        "binding_100pct": int(utilization >= 1.0 - 1e-9),
    }


def route_component_rows(
    data: ModelData,
    allocations: list[RouteAllocation],
    config: ModelConfig,
) -> list[dict[str, object]]:
    """Decompose route time and cost by transport stage and gateway."""
    rows: list[dict[str, object]] = []
    for allocation in allocations:
        shipment = data.shipments[allocation.shipment_id]
        arcs = [data.arcs[arc_id] for arc_id in allocation.arc_ids]
        gateway = next(
            (
                node_id
                for arc in arcs
                for node_id in (arc.origin, arc.destination)
                if data.nodes[node_id].kind == NodeKind.US_PORT
            ),
            "unclassified",
        )
        current_h = shipment.release_h
        previous = None
        reached_gateway = False
        totals = {
            "china_inland_transit_h": 0.0,
            "ocean_transit_h": 0.0,
            "us_inland_transit_h": 0.0,
            "waiting_h": 0.0,
            "transfer_h": 0.0,
            "china_inland_transport_cost_usd": 0.0,
            "ocean_transport_cost_usd": 0.0,
            "us_inland_transport_cost_usd": 0.0,
            "holding_cost_usd": 0.0,
            "transfer_cost_usd": 0.0,
            "processing_cost_usd": 0.0,
        }

        for arc_index, arc in enumerate(arcs):
            if previous is not None:
                rule = data.transfer_rules.get((arc.origin, previous.mode, arc.mode))
                if rule is not None:
                    current_h += rule.duration_h
                    totals["transfer_h"] += rule.duration_h
                    totals["transfer_cost_usd"] += (
                        rule.cost_usd_per_feu * allocation.quantity_feu
                    )
                    totals["processing_cost_usd"] += (
                        config.operational_costs.processing_usd_per_feu_h
                        * rule.duration_h
                        * allocation.quantity_feu
                    )

            departure_h = current_h
            if config.constraints.enforce_timetable and arc.service_ids:
                services = [data.services[service_id] for service_id in arc.service_ids]
                scheduled = ranked_departure(
                    services,
                    current_h,
                    rank=_departure_rank(allocation, arc_index),
                )
                departure_h = (
                    scheduled.departure_h
                    + _scenario_ocean_departure_delay_h(arc, config)
                )
            waiting_h = max(0.0, departure_h - current_h)
            totals["waiting_h"] += waiting_h
            totals["holding_cost_usd"] += (
                config.operational_costs.holding_usd_per_feu_h
                * waiting_h
                * allocation.quantity_feu
            )

            arc_cost = _scenario_arc_cost(arc, config) * allocation.quantity_feu
            if arc.mode.value == "ocean":
                stage = "ocean"
            elif reached_gateway or arc.origin == gateway:
                stage = "us_inland"
            else:
                stage = "china_inland"
            totals[f"{stage}_transit_h"] += arc.transit_time_h
            totals[f"{stage}_transport_cost_usd"] += arc_cost
            current_h = departure_h + arc.transit_time_h
            if arc.destination == gateway:
                reached_gateway = True
            previous = arc

        tardiness_h = max(0.0, current_h - shipment.due_h) if shipment.due_h else 0.0
        tardiness_rate = (
            shipment.tardiness_usd_per_feu_h
            if shipment.tardiness_usd_per_feu_h is not None
            else config.penalties.tardiness_usd_per_feu_h
        )
        tardiness_cost = tardiness_h * allocation.quantity_feu * tardiness_rate
        total_cost = (
            totals["china_inland_transport_cost_usd"]
            + totals["ocean_transport_cost_usd"]
            + totals["us_inland_transport_cost_usd"]
            + totals["holding_cost_usd"]
            + totals["transfer_cost_usd"]
            + totals["processing_cost_usd"]
            + tardiness_cost
        )
        rows.append(
            {
                "shipment_id": allocation.shipment_id,
                "gateway": gateway,
                "quantity_feu": allocation.quantity_feu,
                "lead_time_h": current_h - shipment.release_h,
                "tardiness_h": tardiness_h,
                "tardiness_cost_usd": tardiness_cost,
                "total_cost_usd": total_cost,
                **totals,
            }
        )
    return rows
