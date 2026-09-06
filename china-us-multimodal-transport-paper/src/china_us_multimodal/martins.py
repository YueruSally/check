from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools

from .domain import ModelData, Mode


@dataclass(frozen=True)
class Label:
    node_id: str
    cost_usd_per_feu: float
    elapsed_h: float
    arc_ids: tuple[str, ...]
    visited_nodes: frozenset[str]
    last_mode: Mode | None
    mode_changes: int


def dominates(left: Label, right: Label, tolerance: float = 1e-9) -> bool:
    no_worse = (
        left.cost_usd_per_feu <= right.cost_usd_per_feu + tolerance
        and left.elapsed_h <= right.elapsed_h + tolerance
    )
    better = (
        left.cost_usd_per_feu < right.cost_usd_per_feu - tolerance
        or left.elapsed_h < right.elapsed_h - tolerance
    )
    return no_worse and better


def candidate_paths(
    data: ModelData,
    origin: str,
    destination: str,
    *,
    max_legs: int = 8,
    max_mode_changes: int = 4,
    max_labels_per_node: int = 100,
    max_paths: int = 50,
) -> list[Label]:
    """Build a static bi-objective path library with Martins label setting.

    The shared evaluator subsequently applies exact timetable and capacity
    constraints. planning_wait_h is used only to rank candidate paths.
    """
    if origin not in data.nodes or destination not in data.nodes:
        raise KeyError("Unknown origin or destination.")
    outgoing = data.outgoing()
    counter = itertools.count()
    start = Label(origin, 0.0, 0.0, (), frozenset({origin}), None, 0)
    labels: dict[str, list[Label]] = {node_id: [] for node_id in data.nodes}
    labels[origin].append(start)
    queue: list[tuple[float, float, int, Label]] = [(0.0, 0.0, next(counter), start)]
    complete: list[Label] = []

    while queue and len(complete) < max_paths:
        _, _, _, label = heapq.heappop(queue)
        if any(dominates(other, label) for other in labels[label.node_id] if other != label):
            continue
        if label.node_id == destination:
            complete.append(label)
            continue
        if len(label.arc_ids) >= max_legs:
            continue
        for arc in outgoing.get(label.node_id, []):
            if arc.destination in label.visited_nodes:
                continue
            changes = label.mode_changes + int(
                label.last_mode is not None and label.last_mode != arc.mode
            )
            if changes > max_mode_changes:
                continue
            proposed = Label(
                arc.destination,
                label.cost_usd_per_feu + arc.cost_usd_per_feu,
                label.elapsed_h + arc.transit_time_h + arc.planning_wait_h,
                label.arc_ids + (arc.id,),
                label.visited_nodes | {arc.destination},
                arc.mode,
                changes,
            )
            bucket = labels[arc.destination]
            if any(dominates(existing, proposed) for existing in bucket):
                continue
            bucket[:] = [existing for existing in bucket if not dominates(proposed, existing)]
            bucket.append(proposed)
            bucket.sort(key=lambda item: (item.cost_usd_per_feu, item.elapsed_h))
            del bucket[max_labels_per_node:]
            if proposed in bucket:
                heapq.heappush(
                    queue,
                    (proposed.cost_usd_per_feu, proposed.elapsed_h, next(counter), proposed),
                )

    complete.sort(key=lambda item: (item.cost_usd_per_feu, item.elapsed_h))
    return [
        label
        for label in complete
        if not any(dominates(other, label) for other in complete if other != label)
    ][:max_paths]
