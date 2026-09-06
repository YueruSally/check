from __future__ import annotations

import math
from typing import Iterable

Point = tuple[float, float]


def nondominated_points(points: Iterable[Point]) -> tuple[Point, ...]:
    """Return the unique two-objective minimization front, sorted by cost."""
    best_by_cost: dict[float, float] = {}
    for cost, makespan in points:
        if not math.isfinite(cost) or not math.isfinite(makespan):
            raise ValueError("Objective values must be finite.")
        best_by_cost[cost] = min(makespan, best_by_cost.get(cost, makespan))

    front: list[Point] = []
    best_makespan = math.inf
    for cost, makespan in sorted(best_by_cost.items()):
        if makespan < best_makespan:
            front.append((cost, makespan))
            best_makespan = makespan
    return tuple(front)


def normalize_points(
    points: Iterable[Point],
    ideal: Point,
    nadir: Point,
) -> tuple[Point, ...]:
    spans = tuple(max(high - low, 1e-12) for low, high in zip(ideal, nadir))
    return tuple(
        (
            (point[0] - ideal[0]) / spans[0],
            (point[1] - ideal[1]) / spans[1],
        )
        for point in points
    )


def hypervolume_2d(
    points: Iterable[Point],
    reference: Point = (1.1, 1.1),
) -> float:
    """Exact dominated hypervolume for a two-objective minimization front."""
    filtered = [
        point
        for point in nondominated_points(points)
        if point[0] <= reference[0] and point[1] <= reference[1]
    ]
    previous_makespan = reference[1]
    area = 0.0
    for cost, makespan in filtered:
        if makespan < previous_makespan:
            area += max(0.0, reference[0] - cost) * (
                previous_makespan - makespan
            )
            previous_makespan = makespan
    return area


def spacing(points: Iterable[Point]) -> float:
    """Standard deviation of nearest-neighbour Manhattan distances."""
    values = tuple(nondominated_points(points))
    if len(values) < 2:
        return 0.0
    nearest = [
        min(
            abs(left[0] - right[0]) + abs(left[1] - right[1])
            for index, right in enumerate(values)
            if index != left_index
        )
        for left_index, left in enumerate(values)
    ]
    mean = sum(nearest) / len(nearest)
    return math.sqrt(sum((value - mean) ** 2 for value in nearest) / len(nearest))
