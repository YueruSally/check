from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import tomllib
from typing import Iterable

from .config import ModelConfig
from .domain import ModelData, RouteAllocation
from .evaluator import EvaluationResult, evaluate_solution
from .path_library import PathLibrary, build_gateway_aware_library


@dataclass(frozen=True)
class NSGA2Config:
    population_size: int = 48
    generations: int = 40
    crossover_rate: float = 0.9
    mutation_rate: float = 0.12
    max_paths_per_gateway: int = 4
    seed: int = 20260906

    def validate(self) -> None:
        if self.population_size < 4:
            raise ValueError("NSGA-II population_size must be at least 4.")
        if self.generations < 1:
            raise ValueError("NSGA-II generations must be positive.")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("NSGA-II crossover_rate must be in [0, 1].")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("NSGA-II mutation_rate must be in [0, 1].")
        if self.max_paths_per_gateway < 1:
            raise ValueError("max_paths_per_gateway must be positive.")


def load_nsga2_config(path: str | Path, **overrides: int | float | None) -> NSGA2Config:
    with Path(path).open("rb") as handle:
        values = dict(tomllib.load(handle).get("nsga2", {}))
    values.update({key: value for key, value in overrides.items() if value is not None})
    config = NSGA2Config(**values)
    config.validate()
    return config


@dataclass
class Individual:
    genome: tuple[tuple[float, ...], ...]
    evaluation: EvaluationResult | None = None
    allocations: tuple[RouteAllocation, ...] = ()
    rank: int = 0
    crowding_distance: float = 0.0

    @property
    def objectives(self) -> tuple[float, float]:
        if self.evaluation is None:
            raise RuntimeError("Individual has not been evaluated.")
        return self.evaluation.objectives


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    feasible_count: int
    first_front_size: int
    min_cost_usd: float | None
    min_weighted_mean_delivery_time_h: float | None
    min_constraint_violation: float


@dataclass(frozen=True)
class NSGA2Result:
    shipment_ids: tuple[str, ...]
    candidate_library: PathLibrary
    population: tuple[Individual, ...]
    pareto_front: tuple[Individual, ...]
    history: tuple[GenerationRecord, ...]


def _normalize_sparse(
    values: Iterable[float],
    max_nonzero: int,
    min_share: float = 0.0,
) -> tuple[float, ...]:
    """Keep at most max_nonzero paths and prune shares below min_share."""
    cleaned = [max(0.0, float(value)) for value in values]
    if not cleaned:
        raise ValueError("A shipment must have at least one candidate path.")
    keep = sorted(range(len(cleaned)), key=lambda index: cleaned[index], reverse=True)[
        :max_nonzero
    ]
    active = [index for index in keep if cleaned[index] > 1e-15]
    if not active:
        active = [keep[0]]
        cleaned[keep[0]] = 1.0

    while len(active) > 1:
        total = sum(cleaned[index] for index in active)
        below = [
            index
            for index in active
            if cleaned[index] / total < min_share - 1e-12
        ]
        if not below:
            break
        active = [index for index in active if index not in below]

    sparse = [0.0] * len(cleaned)
    total = sum(cleaned[index] for index in active)
    for index in active:
        sparse[index] = cleaned[index] / total
    return tuple(sparse)


def _random_gene(
    path_count: int,
    max_nonzero: int,
    rng: random.Random,
    min_share: float = 0.0,
) -> tuple[float, ...]:
    active_count = rng.randint(1, min(path_count, max_nonzero))
    active = rng.sample(range(path_count), active_count)
    raw = [0.0] * path_count
    for index in active:
        raw[index] = rng.expovariate(1.0)
    return _normalize_sparse(raw, max_nonzero, min_share)


def _decode(
    data: ModelData,
    config: ModelConfig,
    shipment_ids: tuple[str, ...],
    library: PathLibrary,
    genome: tuple[tuple[float, ...], ...],
) -> tuple[RouteAllocation, ...]:
    allocations: list[RouteAllocation] = []
    for shipment_id, weights in zip(shipment_ids, genome):
        demand = data.shipments[shipment_id].quantity_feu * config.scenario.demand_multiplier
        active = [(index, weight) for index, weight in enumerate(weights) if weight > 1e-12]
        assigned = 0.0
        for position, (path_index, weight) in enumerate(active):
            quantity = demand - assigned if position == len(active) - 1 else demand * weight
            assigned += quantity
            allocations.append(
                RouteAllocation(shipment_id, quantity, library[shipment_id][path_index])
            )
    return tuple(allocations)


def _evaluate(
    individual: Individual,
    data: ModelData,
    config: ModelConfig,
    shipment_ids: tuple[str, ...],
    library: PathLibrary,
) -> None:
    allocations = _decode(data, config, shipment_ids, library, individual.genome)
    individual.allocations = allocations
    individual.evaluation = evaluate_solution(data, list(allocations), config)


def _dominates(left: Individual, right: Individual, tolerance: float = 1e-9) -> bool:
    if left.evaluation is None or right.evaluation is None:
        raise RuntimeError("Dominance requires evaluated individuals.")
    if left.evaluation.feasible != right.evaluation.feasible:
        return left.evaluation.feasible
    if not left.evaluation.feasible:
        return (
            left.evaluation.constraint_violation
            < right.evaluation.constraint_violation - tolerance
        )
    left_values, right_values = left.objectives, right.objectives
    no_worse = all(a <= b + tolerance for a, b in zip(left_values, right_values))
    better = any(a < b - tolerance for a, b in zip(left_values, right_values))
    return no_worse and better


def fast_non_dominated_sort(population: list[Individual]) -> list[list[Individual]]:
    dominates_sets: list[list[int]] = [[] for _ in population]
    dominated_counts = [0] * len(population)
    fronts: list[list[int]] = [[]]
    for p, left in enumerate(population):
        for q, right in enumerate(population):
            if p == q:
                continue
            if _dominates(left, right):
                dominates_sets[p].append(q)
            elif _dominates(right, left):
                dominated_counts[p] += 1
        if dominated_counts[p] == 0:
            left.rank = 0
            fronts[0].append(p)

    index = 0
    while index < len(fronts) and fronts[index]:
        next_front: list[int] = []
        for p in fronts[index]:
            for q in dominates_sets[p]:
                dominated_counts[q] -= 1
                if dominated_counts[q] == 0:
                    population[q].rank = index + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        index += 1
    return [[population[item] for item in front] for front in fronts if front]


def assign_crowding_distance(front: list[Individual]) -> None:
    for individual in front:
        individual.crowding_distance = 0.0
    if len(front) <= 2:
        for individual in front:
            individual.crowding_distance = math.inf
        return
    if all(item.evaluation and item.evaluation.feasible for item in front):
        dimensions = [
            lambda item: item.objectives[0],
            lambda item: item.objectives[1],
        ]
    else:
        dimensions = [lambda item: item.evaluation.constraint_violation]  # type: ignore[union-attr]
    for value_of in dimensions:
        ordered = sorted(front, key=value_of)
        ordered[0].crowding_distance = math.inf
        ordered[-1].crowding_distance = math.inf
        low, high = value_of(ordered[0]), value_of(ordered[-1])
        if abs(high - low) <= 1e-12:
            continue
        for index in range(1, len(ordered) - 1):
            if math.isinf(ordered[index].crowding_distance):
                continue
            ordered[index].crowding_distance += (
                value_of(ordered[index + 1]) - value_of(ordered[index - 1])
            ) / (high - low)


def _rank_population(population: list[Individual]) -> list[list[Individual]]:
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        assign_crowding_distance(front)
    return fronts


def _tournament(population: list[Individual], rng: random.Random) -> Individual:
    left, right = rng.sample(population, 2)
    left_key = (left.rank, -left.crowding_distance)
    right_key = (right.rank, -right.crowding_distance)
    if left_key < right_key:
        return left
    if right_key < left_key:
        return right
    return left if rng.random() < 0.5 else right


def _crossover(
    left: Individual,
    right: Individual,
    rate: float,
    max_nonzero: int,
    min_share: float,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    if rng.random() > rate:
        return tuple(tuple(gene) for gene in left.genome)
    child = []
    for left_gene, right_gene in zip(left.genome, right.genome):
        alpha = rng.random()
        mixed = [alpha * a + (1 - alpha) * b for a, b in zip(left_gene, right_gene)]
        child.append(_normalize_sparse(mixed, max_nonzero, min_share))
    return tuple(child)


def _mutate(
    genome: tuple[tuple[float, ...], ...],
    rate: float,
    max_nonzero: int,
    min_share: float,
    rng: random.Random,
) -> tuple[tuple[float, ...], ...]:
    mutated = []
    for gene in genome:
        values = list(gene)
        if rng.random() < rate:
            if len(values) == 1:
                values[0] = 1.0
            elif rng.random() < 0.25:
                values = list(_random_gene(len(values), max_nonzero, rng, min_share))
            else:
                source = rng.randrange(len(values))
                target = rng.randrange(len(values) - 1)
                if target >= source:
                    target += 1
                amount = rng.random() * values[source]
                values[source] -= amount
                values[target] += amount
        mutated.append(_normalize_sparse(values, max_nonzero, min_share))
    return tuple(mutated)


def _environmental_selection(combined: list[Individual], size: int) -> list[Individual]:
    selected: list[Individual] = []
    for front in _rank_population(combined):
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            selected.extend(
                sorted(front, key=lambda item: item.crowding_distance, reverse=True)[
                    : size - len(selected)
                ]
            )
            break
    _rank_population(selected)
    return selected


def _history_record(generation: int, population: list[Individual]) -> GenerationRecord:
    fronts = _rank_population(population)
    feasible = [item for item in population if item.evaluation and item.evaluation.feasible]
    return GenerationRecord(
        generation=generation,
        feasible_count=len(feasible),
        first_front_size=len(fronts[0]),
        min_cost_usd=min((item.objectives[0] for item in feasible), default=None),
        min_weighted_mean_delivery_time_h=min(
            (item.objectives[1] for item in feasible), default=None
        ),
        min_constraint_violation=min(
            item.evaluation.constraint_violation  # type: ignore[union-attr]
            for item in population
        ),
    )


def run_nsga2(
    data: ModelData,
    model_config: ModelConfig,
    algorithm_config: NSGA2Config,
    candidate_library: PathLibrary | None = None,
) -> NSGA2Result:
    algorithm_config.validate()
    rng = random.Random(algorithm_config.seed)
    shipment_ids = tuple(sorted(data.shipments))
    library = candidate_library or build_gateway_aware_library(
        data,
        max_legs=model_config.constraints.max_legs_per_path,
        max_mode_changes=model_config.constraints.max_mode_changes,
        max_paths_per_gateway=algorithm_config.max_paths_per_gateway,
    )
    for shipment_id in shipment_ids:
        if shipment_id not in library or not library[shipment_id]:
            raise ValueError(f"Shipment {shipment_id} has no candidate paths.")

    max_nonzero = model_config.constraints.max_paths_per_shipment
    min_share = model_config.constraints.min_path_share
    population: list[Individual] = []
    for seed_kind in ("cost", "time"):
        genome = []
        for shipment_id in shipment_ids:
            paths = library[shipment_id]
            key = 0 if seed_kind == "cost" else 1
            scores = []
            for path in paths:
                cost = sum(data.arcs[arc_id].cost_usd_per_feu for arc_id in path)
                time = sum(
                    data.arcs[arc_id].transit_time_h + data.arcs[arc_id].planning_wait_h
                    for arc_id in path
                )
                scores.append((cost, time))
            chosen = min(range(len(paths)), key=lambda index: scores[index][key])
            gene = [0.0] * len(paths)
            gene[chosen] = 1.0
            genome.append(tuple(gene))
        population.append(Individual(tuple(genome)))
    while len(population) < algorithm_config.population_size:
        genome = tuple(
            _random_gene(len(library[shipment_id]), max_nonzero, rng, min_share)
            for shipment_id in shipment_ids
        )
        population.append(Individual(genome))
    for individual in population:
        _evaluate(individual, data, model_config, shipment_ids, library)
    _rank_population(population)
    history = [_history_record(0, population)]

    for generation in range(1, algorithm_config.generations + 1):
        offspring: list[Individual] = []
        while len(offspring) < algorithm_config.population_size:
            parent_a = _tournament(population, rng)
            parent_b = _tournament(population, rng)
            genome = _crossover(
                parent_a,
                parent_b,
                algorithm_config.crossover_rate,
                max_nonzero,
                min_share,
                rng,
            )
            genome = _mutate(
                genome,
                algorithm_config.mutation_rate,
                max_nonzero,
                min_share,
                rng,
            )
            child = Individual(genome)
            _evaluate(child, data, model_config, shipment_ids, library)
            offspring.append(child)
        population = _environmental_selection(
            population + offspring,
            algorithm_config.population_size,
        )
        history.append(_history_record(generation, population))

    fronts = _rank_population(population)
    pareto = tuple(item for item in fronts[0] if item.evaluation and item.evaluation.feasible)
    return NSGA2Result(
        shipment_ids=shipment_ids,
        candidate_library=library,
        population=tuple(population),
        pareto_front=pareto,
        history=tuple(history),
    )
