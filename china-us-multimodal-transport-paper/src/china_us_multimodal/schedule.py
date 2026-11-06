from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from .domain import Service


@dataclass(frozen=True)
class ScheduledDeparture:
    service_id: str
    departure_h: float


def next_departure(service: Service, ready_h: float) -> ScheduledDeparture:
    """Return the first recurring departure whose cutoff can be met."""
    if service.repeat_every_h <= 0:
        raise ValueError(f"Service {service.id} has a non-positive repeat period.")
    if not service.departures_h:
        raise ValueError(f"Service {service.id} has no departures.")
    eligible_h = ready_h + service.cutoff_h
    best: float | None = None
    for base_h in service.departures_h:
        cycles = max(0, math.ceil((eligible_h - base_h) / service.repeat_every_h))
        candidate = base_h + cycles * service.repeat_every_h
        if candidate + 1e-9 < eligible_h:
            candidate += service.repeat_every_h
        best = candidate if best is None else min(best, candidate)
    return ScheduledDeparture(service.id, float(best))


def earliest_departure(services: list[Service], ready_h: float) -> ScheduledDeparture:
    return ranked_departure(services, ready_h, rank=0)


def ranked_departure(
    services: list[Service], ready_h: float, *, rank: int
) -> ScheduledDeparture:
    """Return the zero-based ``rank``-th feasible scheduled departure.

    The ordered event stream contains every recurring departure pattern of
    every service attached to an arc.  Departures from different services at
    the same time remain distinct capacity choices.  Duplicate events within
    one service are removed.  ``rank=0`` is exactly the historical earliest-
    reachable-service rule.
    """
    if not services:
        raise ValueError("At least one service is required.")
    if rank < 0:
        raise ValueError("Departure rank must be non-negative.")

    queue: list[tuple[float, int, int, str, float]] = []
    by_id = {service.id: service for service in services}
    for service_order, service in enumerate(services):
        if service.repeat_every_h <= 0:
            raise ValueError(f"Service {service.id} has a non-positive repeat period.")
        if not service.departures_h:
            raise ValueError(f"Service {service.id} has no departures.")
        eligible_h = ready_h + service.cutoff_h
        for pattern_index, base_h in enumerate(service.departures_h):
            cycles = max(
                0,
                math.ceil((eligible_h - base_h) / service.repeat_every_h),
            )
            candidate = base_h + cycles * service.repeat_every_h
            if candidate + 1e-9 < eligible_h:
                candidate += service.repeat_every_h
            heapq.heappush(
                queue,
                (
                    float(candidate),
                    service_order,
                    pattern_index,
                    service.id,
                    float(service.repeat_every_h),
                ),
            )

    seen: set[tuple[str, float]] = set()
    found = -1
    while queue:
        departure_h, service_order, pattern_index, service_id, repeat_h = heapq.heappop(
            queue
        )
        heapq.heappush(
            queue,
            (
                departure_h + repeat_h,
                service_order,
                pattern_index,
                service_id,
                repeat_h,
            ),
        )
        signature = (service_id, departure_h)
        if signature in seen:
            continue
        seen.add(signature)
        found += 1
        if found == rank:
            return ScheduledDeparture(by_id[service_id].id, departure_h)

    raise RuntimeError("No recurring departure could be generated.")
