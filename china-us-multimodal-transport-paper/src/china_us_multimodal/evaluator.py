from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from .config import ModelConfig
from .domain import Arc, ModelData, RouteAllocation
from .schedule import earliest_departure


@dataclass(frozen=True)
class RouteOutcome:
    shipment_id: str
    quantity_feu: float
    arrival_h: float
    cost_usd: float
    tardiness_h: float


@dataclass(frozen=True)
class EvaluationResult:
    total_cost_usd: float
    makespan_h: float
    max_lead_time_h: float
    total_tardiness_feu_h: float
    feasible: bool
    violations: tuple[str, ...]
    routes: tuple[RouteOutcome, ...]

    @property
    def objectives(self) -> tuple[float, float]:
        return self.total_cost_usd, self.makespan_h


def _scenario_arc_cost(arc: Arc, config: ModelConfig) -> float:
    if arc.mode.value != "ocean":
        return arc.cost_usd_per_feu
    multiplier = (
        config.scenario.usec_ocean_cost_multiplier
        if arc.via_panama
        else config.scenario.uswc_ocean_cost_multiplier
    )
    return arc.cost_usd_per_feu * multiplier


def _path_arcs(data: ModelData, allocation: RouteAllocation) -> list[Arc]:
    shipment = data.shipments[allocation.shipment_id]
    arcs = [data.arcs[arc_id] for arc_id in allocation.arc_ids]
    if not arcs:
        raise ValueError(f"Shipment {shipment.id} has an empty path.")
    if arcs[0].origin != shipment.origin or arcs[-1].destination != shipment.destination:
        raise ValueError(f"Path endpoints do not match shipment {shipment.id}.")
    for previous, current in zip(arcs, arcs[1:]):
        if previous.destination != current.origin:
            raise ValueError(f"Non-contiguous path: {previous.id} -> {current.id}.")
    return arcs


def evaluate_solution(
    data: ModelData,
    allocations: list[RouteAllocation],
    config: ModelConfig,
) -> EvaluationResult:
    violations: list[str] = []
    by_shipment: dict[str, list[RouteAllocation]] = defaultdict(list)
    for allocation in allocations:
        if allocation.shipment_id not in data.shipments:
            violations.append(f"Unknown shipment {allocation.shipment_id!r}.")
        elif allocation.quantity_feu <= 0:
            violations.append(f"Allocation for {allocation.shipment_id} is not positive.")
        else:
            by_shipment[allocation.shipment_id].append(allocation)

    tolerance = config.constraints.allocation_tolerance_feu
    for shipment in data.shipments.values():
        plans = by_shipment.get(shipment.id, [])
        allocated = sum(plan.quantity_feu for plan in plans)
        demand = shipment.quantity_feu * config.scenario.demand_multiplier
        if abs(allocated - demand) > tolerance:
            violations.append(
                f"Shipment {shipment.id}: allocated {allocated:g} FEU, demand {demand:g} FEU."
            )
        if not config.constraints.allow_split and len(plans) > 1:
            violations.append(f"Shipment {shipment.id} is split while splitting is disabled.")
        if len(plans) > config.constraints.max_paths_per_shipment:
            violations.append(f"Shipment {shipment.id} uses too many paths.")

    service_usage: Counter[tuple[str, float]] = Counter()
    daily_arc_usage: Counter[tuple[str, int]] = Counter()
    daily_node_usage: Counter[tuple[str, int]] = Counter()
    outcomes: list[RouteOutcome] = []

    for shipment_id, plans in by_shipment.items():
        shipment = data.shipments[shipment_id]
        for plan in plans:
            try:
                arcs = _path_arcs(data, plan)
            except (KeyError, ValueError) as exc:
                violations.append(str(exc))
                continue
            if len(arcs) > config.constraints.max_legs_per_path:
                violations.append(f"Shipment {shipment_id} exceeds the path leg limit.")
            changes = sum(a.mode != b.mode for a, b in zip(arcs, arcs[1:]))
            if changes > config.constraints.max_mode_changes:
                violations.append(f"Shipment {shipment_id} exceeds the mode-change limit.")

            current_h = shipment.release_h
            route_cost = 0.0
            previous: Arc | None = None
            for arc in arcs:
                if previous is not None:
                    rule = data.transfer_rules.get((arc.origin, previous.mode, arc.mode))
                    if previous.mode != arc.mode and rule is None:
                        violations.append(
                            f"Missing transfer rule at {arc.origin}: "
                            f"{previous.mode.value}->{arc.mode.value}."
                        )
                    if rule is not None:
                        current_h += rule.duration_h
                        route_cost += rule.cost_usd_per_feu * plan.quantity_feu

                departure_h = current_h
                if config.constraints.enforce_timetable and arc.service_ids:
                    services = [data.services[item] for item in arc.service_ids]
                    scheduled = earliest_departure(services, current_h)
                    departure_h = scheduled.departure_h
                    service_usage[(scheduled.service_id, departure_h)] += plan.quantity_feu
                elif config.constraints.enforce_timetable and arc.mode.value in {"rail", "ocean"}:
                    violations.append(f"Scheduled arc {arc.id} has no service.")

                day = math.floor(departure_h / 24)
                daily_arc_usage[(arc.id, day)] += plan.quantity_feu
                daily_node_usage[(arc.origin, day)] += plan.quantity_feu
                current_h = departure_h + arc.transit_time_h
                route_cost += _scenario_arc_cost(arc, config) * plan.quantity_feu
                previous = arc

            tardiness_h = max(0.0, current_h - shipment.due_h) if shipment.due_h else 0.0
            if tardiness_h and config.constraints.hard_due_time:
                violations.append(f"Shipment {shipment_id} is {tardiness_h:.2f} h late.")
            route_cost += (
                tardiness_h * plan.quantity_feu * config.penalties.tardiness_usd_per_feu_h
            )
            outcomes.append(RouteOutcome(shipment_id, plan.quantity_feu, current_h, route_cost, tardiness_h))

    if config.constraints.enforce_capacity:
        for (service_id, departure_h), used in service_usage.items():
            service = data.services[service_id]
            arc = data.arcs[service.arc_id]
            multiplier = config.scenario.panama_availability_multiplier if arc.via_panama else 1.0
            limit = service.capacity_feu * multiplier
            if used > limit + tolerance:
                violations.append(
                    f"Service {service_id} at {departure_h:g} h: {used:g}>{limit:g} FEU."
                )
        for (arc_id, day), used in daily_arc_usage.items():
            limit = data.arcs[arc_id].capacity_feu_per_day
            if limit is not None and used > limit + tolerance:
                violations.append(f"Arc {arc_id} day {day}: {used:g}>{limit:g} FEU.")
        for (node_id, day), used in daily_node_usage.items():
            node = data.nodes[node_id]
            if node.capacity_feu_per_day is None:
                continue
            limit = node.capacity_feu_per_day
            if node.kind.value in {"cn_port", "us_port"}:
                limit *= config.scenario.port_capacity_multiplier
            if used > limit + tolerance:
                violations.append(f"Node {node_id} day {day}: {used:g}>{limit:g} FEU.")

    total_cost = sum(item.cost_usd for item in outcomes)
    total_tardiness = sum(item.tardiness_h * item.quantity_feu for item in outcomes)
    if outcomes:
        earliest_release = min(data.shipments[item.shipment_id].release_h for item in outcomes)
        makespan = max(item.arrival_h for item in outcomes) - earliest_release
        max_lead = max(
            item.arrival_h - data.shipments[item.shipment_id].release_h for item in outcomes
        )
    else:
        makespan = max_lead = 0.0

    feasible = not violations
    if not feasible:
        floor = config.penalties.infeasible_objective
        total_cost, makespan = max(total_cost, floor), max(makespan, floor)
    return EvaluationResult(
        total_cost,
        makespan,
        max_lead,
        total_tardiness,
        feasible,
        tuple(violations),
        tuple(outcomes),
    )
