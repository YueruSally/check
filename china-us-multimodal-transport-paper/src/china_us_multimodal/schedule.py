from __future__ import annotations

from dataclasses import dataclass
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
    if not services:
        raise ValueError("At least one service is required.")
    return min(
        (next_departure(service, ready_h) for service in services),
        key=lambda item: item.departure_h,
    )
