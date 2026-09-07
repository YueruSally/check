from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Iterable

from .config import load_config
from .io import load_model_data, select_case
from .metrics import hypervolume_2d, nondominated_points, normalize_points, spacing
from .nsga2 import NSGA2Config, load_nsga2_config, run_nsga2
from .validation import validate_model_data


@dataclass(frozen=True)
class BatchSolution:
    total_cost_usd: float
    quantity_weighted_mean_delivery_time_h: float
    makespan_h: float
    max_lead_time_h: float
    total_tardiness_feu_h: float
    allocations: tuple[tuple[str, float, tuple[str, ...]], ...]


@dataclass(frozen=True)
class BatchRun:
    case_id: str
    scenario: str
    seed: int
    elapsed_seconds: float
    population_size: int
    generations: int
    feasible_population: int
    pareto_size: int
    min_cost_usd: float | None
    min_weighted_mean_delivery_time_h: float | None
    points: tuple[tuple[float, float], ...]
    solutions: tuple[BatchSolution, ...] = ()


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _run_one(payload: tuple[str, str, str, str, int, NSGA2Config]) -> BatchRun:
    data_directory, config_path, case_id, scenario, seed, base_algorithm = payload
    model_config = load_config(config_path, scenario)
    data = load_model_data(data_directory)
    case_data = select_case(data, case_id)
    algorithm = replace(base_algorithm, seed=seed)
    started = time.perf_counter()
    result = run_nsga2(case_data, model_config, algorithm)
    elapsed = time.perf_counter() - started
    feasible = [
        item for item in result.population if item.evaluation and item.evaluation.feasible
    ]
    solution_by_point: dict[tuple[float, float], BatchSolution] = {}
    for item in result.pareto_front:
        evaluation = item.evaluation
        if evaluation is None:
            continue
        point = item.objectives
        solution_by_point.setdefault(
            point,
            BatchSolution(
                total_cost_usd=evaluation.total_cost_usd,
                quantity_weighted_mean_delivery_time_h=(
                    evaluation.quantity_weighted_mean_delivery_time_h
                ),
                makespan_h=evaluation.makespan_h,
                max_lead_time_h=evaluation.max_lead_time_h,
                total_tardiness_feu_h=evaluation.total_tardiness_feu_h,
                allocations=tuple(
                    (
                        allocation.shipment_id,
                        allocation.quantity_feu,
                        tuple(allocation.arc_ids),
                    )
                    for allocation in item.allocations
                ),
            ),
        )
    points = nondominated_points(solution_by_point)
    solutions = tuple(solution_by_point[point] for point in points)
    return BatchRun(
        case_id=case_id,
        scenario=scenario,
        seed=seed,
        elapsed_seconds=elapsed,
        population_size=algorithm.population_size,
        generations=algorithm.generations,
        feasible_population=len(feasible),
        pareto_size=len(points),
        min_cost_usd=min((point[0] for point in points), default=None),
        min_weighted_mean_delivery_time_h=min(
            (point[1] for point in points), default=None
        ),
        points=points,
        solutions=solutions,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(runs: Iterable[BatchRun]) -> tuple[list[dict], list[dict], list[dict]]:
    runs = list(runs)
    run_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    union_rows: list[dict] = []
    groups = sorted({(run.case_id, run.scenario) for run in runs})
    for case_id, scenario in groups:
        group = [
            run for run in runs if run.case_id == case_id and run.scenario == scenario
        ]
        pooled = [point for run in group for point in run.points]
        if pooled:
            ideal = (min(p[0] for p in pooled), min(p[1] for p in pooled))
            nadir = (max(p[0] for p in pooled), max(p[1] for p in pooled))
        else:
            ideal = nadir = (0.0, 0.0)

        hypervolumes: list[float] = []
        spacings: list[float] = []
        for run in group:
            normalized = normalize_points(run.points, ideal, nadir) if run.points else ()
            hv = hypervolume_2d(normalized) if normalized else 0.0
            space = spacing(normalized)
            hypervolumes.append(hv)
            spacings.append(space)
            row = asdict(run)
            row.pop("points")
            row.pop("solutions")
            row["hypervolume_normalized"] = hv
            row["spacing_normalized"] = space
            run_rows.append(row)

        union_front = nondominated_points(pooled)
        sources: dict[tuple[float, float], list[int]] = {}
        for run in group:
            for point in run.points:
                sources.setdefault(point, []).append(run.seed)
        for index, point in enumerate(union_front, start=1):
            union_rows.append(
                {
                    "case_id": case_id,
                    "scenario": scenario,
                    "point_id": index,
                    "total_cost_usd": point[0],
                    "quantity_weighted_mean_delivery_time_h": point[1],
                    "source_seeds": "|".join(map(str, sorted(set(sources[point])))),
                }
            )

        costs = [run.min_cost_usd for run in group if run.min_cost_usd is not None]
        delivery_times = [
            run.min_weighted_mean_delivery_time_h
            for run in group
            if run.min_weighted_mean_delivery_time_h is not None
        ]
        elapsed = [run.elapsed_seconds for run in group]
        successful = sum(bool(run.points) for run in group)
        aggregate_rows.append(
            {
                "case_id": case_id,
                "scenario": scenario,
                "runs": len(group),
                "successful_runs": successful,
                "success_rate": successful / len(group),
                "total_front_points": len(pooled),
                "union_front_size": len(union_front),
                "best_cost_usd": min(costs) if costs else None,
                "cost_median_usd": statistics.median(costs) if costs else None,
                "cost_q1_usd": _percentile(costs, 0.25),
                "cost_q3_usd": _percentile(costs, 0.75),
                "best_weighted_mean_delivery_time_h": (
                    min(delivery_times) if delivery_times else None
                ),
                "weighted_mean_delivery_time_median_h": (
                    statistics.median(delivery_times) if delivery_times else None
                ),
                "weighted_mean_delivery_time_q1_h": _percentile(
                    delivery_times, 0.25
                ),
                "weighted_mean_delivery_time_q3_h": _percentile(
                    delivery_times, 0.75
                ),
                "hypervolume_median": statistics.median(hypervolumes),
                "hypervolume_q1": _percentile(hypervolumes, 0.25),
                "hypervolume_q3": _percentile(hypervolumes, 0.75),
                "spacing_median": statistics.median(spacings),
                "elapsed_median_seconds": statistics.median(elapsed),
                "elapsed_total_seconds": sum(elapsed),
                "ideal_cost_usd": ideal[0],
                "ideal_weighted_mean_delivery_time_h": ideal[1],
                "nadir_cost_usd": nadir[0],
                "nadir_weighted_mean_delivery_time_h": nadir[1],
                "hypervolume_reference_normalized": "1.1|1.1",
            }
        )
    return run_rows, aggregate_rows, union_rows


def representative_rows(runs: Iterable[BatchRun]) -> tuple[list[dict], list[dict]]:
    """Select union-front solutions and expose their complete route allocations."""
    runs = list(runs)
    solution_rows: list[dict] = []
    allocation_rows: list[dict] = []
    groups = sorted({(run.case_id, run.scenario) for run in runs})
    role_order = ("min_cost", "min_delivery_time", "balanced")
    for case_id, scenario in groups:
        group = [
            run for run in runs if run.case_id == case_id and run.scenario == scenario
        ]
        union_front = nondominated_points(
            point for run in group for point in run.points
        )
        if not union_front:
            continue
        ideal = (min(p[0] for p in union_front), min(p[1] for p in union_front))
        nadir = (max(p[0] for p in union_front), max(p[1] for p in union_front))
        normalized = normalize_points(union_front, ideal, nadir)
        min_cost = min(union_front, key=lambda point: (point[0], point[1]))
        min_delivery_time = min(
            union_front, key=lambda point: (point[1], point[0])
        )
        balanced = union_front[
            min(
                range(len(union_front)),
                key=lambda index: normalized[index][0] ** 2
                + normalized[index][1] ** 2,
            )
        ]
        roles_by_point: dict[tuple[float, float], set[str]] = {
            point: set() for point in union_front
        }
        roles_by_point[min_cost].add("min_cost")
        roles_by_point[min_delivery_time].add("min_delivery_time")
        roles_by_point[balanced].add("balanced")

        candidates: dict[
            tuple[float, float], list[tuple[int, BatchSolution]]
        ] = {point: [] for point in union_front}
        for run in group:
            for solution in run.solutions:
                point = (
                    solution.total_cost_usd,
                    solution.quantity_weighted_mean_delivery_time_h,
                )
                if point in candidates:
                    candidates[point].append((run.seed, solution))

        for point_index, point in enumerate(union_front, start=1):
            if not candidates[point]:
                continue
            seed, solution = min(candidates[point], key=lambda item: item[0])
            solution_id = f"{case_id}_{scenario}_{point_index:03d}"
            roles = "|".join(
                role for role in role_order if role in roles_by_point[point]
            )
            solution_rows.append(
                {
                    "solution_id": solution_id,
                    "case_id": case_id,
                    "scenario": scenario,
                    "roles": roles,
                    "source_seed": seed,
                    "total_cost_usd": solution.total_cost_usd,
                    "quantity_weighted_mean_delivery_time_h": (
                        solution.quantity_weighted_mean_delivery_time_h
                    ),
                    "makespan_h": solution.makespan_h,
                    "max_lead_time_h": solution.max_lead_time_h,
                    "total_tardiness_feu_h": solution.total_tardiness_feu_h,
                    "route_allocations": len(solution.allocations),
                }
            )
            for allocation_index, (
                shipment_id,
                quantity_feu,
                arc_ids,
            ) in enumerate(solution.allocations, start=1):
                allocation_rows.append(
                    {
                        "solution_id": solution_id,
                        "allocation_id": allocation_index,
                        "shipment_id": shipment_id,
                        "quantity_feu": quantity_feu,
                        "arc_ids": "|".join(arc_ids),
                    }
                )
    return solution_rows, allocation_rows


def _report_progress(run: BatchRun, completed: int, total: int) -> None:
    print(
        f"[{completed}/{total}] {run.case_id} {run.scenario} "
        f"seed={run.seed} feasible={run.feasible_population}/"
        f"{run.population_size} front={run.pareto_size} "
        f"elapsed={run.elapsed_seconds:.2f}s",
        flush=True,
    )


def run_batch_experiment(
    data_directory: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
    cases: Iterable[str],
    scenarios: Iterable[str],
    seeds: Iterable[int],
    workers: int = 1,
    population_size: int | None = None,
    generations: int | None = None,
) -> list[BatchRun]:
    data_directory = Path(data_directory)
    config_path = Path(config_path)
    output_directory = Path(output_directory)
    cases, scenarios, seeds = tuple(cases), tuple(scenarios), tuple(seeds)
    if not cases or not scenarios or not seeds:
        raise ValueError("Cases, scenarios and seeds must all be non-empty.")
    if workers < 1:
        raise ValueError("workers must be positive.")

    data = load_model_data(data_directory)
    validate_model_data(data)
    available_cases = {item.case_id for item in data.shipments.values()}
    unknown_cases = set(cases) - available_cases
    if unknown_cases:
        raise ValueError(f"Unknown cases: {sorted(unknown_cases)}")
    for scenario in scenarios:
        load_config(config_path, scenario)
    algorithm = load_nsga2_config(
        config_path,
        population_size=population_size,
        generations=generations,
    )
    payloads = [
        (
            str(data_directory),
            str(config_path),
            case_id,
            scenario,
            seed,
            algorithm,
        )
        for case_id in cases
        for scenario in scenarios
        for seed in seeds
    ]
    runs: list[BatchRun] = []
    if workers == 1:
        for completed, payload in enumerate(payloads, start=1):
            run = _run_one(payload)
            runs.append(run)
            _report_progress(run, completed, len(payloads))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one, payload) for payload in payloads]
            for completed, future in enumerate(as_completed(futures), start=1):
                run = future.result()
                runs.append(run)
                _report_progress(run, completed, len(payloads))
    runs.sort(key=lambda run: (run.case_id, run.scenario, run.seed))

    output_directory.mkdir(parents=True, exist_ok=True)
    run_rows, aggregate_rows, union_rows = summarize_runs(runs)
    _write_csv(
        output_directory / "runs.csv",
        [
            "case_id", "scenario", "seed", "elapsed_seconds", "population_size",
            "generations", "feasible_population", "pareto_size", "min_cost_usd",
            "min_weighted_mean_delivery_time_h", "hypervolume_normalized",
            "spacing_normalized",
        ],
        run_rows,
    )
    front_rows = [
        {
            "case_id": run.case_id,
            "scenario": run.scenario,
            "seed": run.seed,
            "point_id": index,
            "total_cost_usd": point[0],
            "quantity_weighted_mean_delivery_time_h": point[1],
        }
        for run in runs
        for index, point in enumerate(run.points, start=1)
    ]
    _write_csv(
        output_directory / "fronts.csv",
        [
            "case_id", "scenario", "seed", "point_id", "total_cost_usd",
            "quantity_weighted_mean_delivery_time_h",
        ],
        front_rows,
    )
    _write_csv(output_directory / "aggregate.csv", list(aggregate_rows[0]), aggregate_rows)
    _write_csv(
        output_directory / "combined_pareto.csv",
        [
            "case_id", "scenario", "point_id", "total_cost_usd",
            "quantity_weighted_mean_delivery_time_h", "source_seeds",
        ],
        union_rows,
    )
    solution_rows, allocation_rows = representative_rows(runs)
    _write_csv(
        output_directory / "representative_solutions.csv",
        [
            "solution_id", "case_id", "scenario", "roles", "source_seed",
            "total_cost_usd", "quantity_weighted_mean_delivery_time_h",
            "makespan_h", "max_lead_time_h", "total_tardiness_feu_h",
            "route_allocations",
        ],
        solution_rows,
    )
    _write_csv(
        output_directory / "representative_allocations.csv",
        [
            "solution_id", "allocation_id", "shipment_id", "quantity_feu",
            "arc_ids",
        ],
        allocation_rows,
    )

    conversion_path = data_directory / "conversion_report.json"
    conversion = (
        json.loads(conversion_path.read_text(encoding="utf-8"))
        if conversion_path.exists()
        else {}
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": (
            "multi-seed NSGA-II with cost and quantity-weighted delivery time"
        ),
        "cases": list(cases),
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "workers": workers,
        "algorithm": asdict(algorithm),
        "evaluations_per_run": algorithm.population_size
        * (algorithm.generations + 1),
        "source_workbook": conversion.get("source_workbook"),
        "source_sha256": conversion.get("source_sha256"),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "minimum_path_share": load_config(
            config_path, scenarios[0]
        ).constraints.min_path_share,
        "objectives": [
            "total_cost_usd",
            "quantity_weighted_mean_delivery_time_h",
        ],
        "files": {
            "runs.csv": "one row per independent run",
            "fronts.csv": "all final feasible non-dominated objective points",
            "aggregate.csv": "cross-seed stability summary",
            "combined_pareto.csv": "non-dominated union over seeds",
            "representative_solutions.csv": (
                "objective values and roles for union-front solutions"
            ),
            "representative_allocations.csv": (
                "shipment quantities and arc sequences for union-front solutions"
            ),
        },
        "note": (
            "Calibration baseline, not a paper conclusion. Comparator algorithms "
            "and scenario experiments must use the same evaluation budget and "
            "shared normalization bounds."
        ),
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return runs
