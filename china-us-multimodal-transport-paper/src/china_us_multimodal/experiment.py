from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
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
    min_makespan_h: float | None
    points: tuple[tuple[float, float], ...]


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
    points = nondominated_points(
        item.objectives for item in result.pareto_front if item.evaluation
    )
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
        min_makespan_h=min((point[1] for point in points), default=None),
        points=points,
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
                    "makespan_h": point[1],
                    "source_seeds": "|".join(map(str, sorted(set(sources[point])))),
                }
            )

        costs = [run.min_cost_usd for run in group if run.min_cost_usd is not None]
        times = [
            run.min_makespan_h for run in group if run.min_makespan_h is not None
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
                "best_makespan_h": min(times) if times else None,
                "makespan_median_h": statistics.median(times) if times else None,
                "makespan_q1_h": _percentile(times, 0.25),
                "makespan_q3_h": _percentile(times, 0.75),
                "hypervolume_median": statistics.median(hypervolumes),
                "hypervolume_q1": _percentile(hypervolumes, 0.25),
                "hypervolume_q3": _percentile(hypervolumes, 0.75),
                "spacing_median": statistics.median(spacings),
                "elapsed_median_seconds": statistics.median(elapsed),
                "elapsed_total_seconds": sum(elapsed),
                "ideal_cost_usd": ideal[0],
                "ideal_makespan_h": ideal[1],
                "nadir_cost_usd": nadir[0],
                "nadir_makespan_h": nadir[1],
                "hypervolume_reference_normalized": "1.1|1.1",
            }
        )
    return run_rows, aggregate_rows, union_rows


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
    if workers == 1:
        runs = [_run_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            runs = list(executor.map(_run_one, payloads))

    output_directory.mkdir(parents=True, exist_ok=True)
    run_rows, aggregate_rows, union_rows = summarize_runs(runs)
    _write_csv(
        output_directory / "runs.csv",
        [
            "case_id", "scenario", "seed", "elapsed_seconds", "population_size",
            "generations", "feasible_population", "pareto_size", "min_cost_usd",
            "min_makespan_h", "hypervolume_normalized", "spacing_normalized",
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
            "makespan_h": point[1],
        }
        for run in runs
        for index, point in enumerate(run.points, start=1)
    ]
    _write_csv(
        output_directory / "fronts.csv",
        ["case_id", "scenario", "seed", "point_id", "total_cost_usd", "makespan_h"],
        front_rows,
    )
    _write_csv(output_directory / "aggregate.csv", list(aggregate_rows[0]), aggregate_rows)
    _write_csv(
        output_directory / "combined_pareto.csv",
        [
            "case_id", "scenario", "point_id", "total_cost_usd", "makespan_h",
            "source_seeds",
        ],
        union_rows,
    )

    conversion_path = data_directory / "conversion_report.json"
    conversion = (
        json.loads(conversion_path.read_text(encoding="utf-8"))
        if conversion_path.exists()
        else {}
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "multi-seed NSGA-II calibration baseline",
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
        "files": {
            "runs.csv": "one row per independent run",
            "fronts.csv": "all final feasible non-dominated objective points",
            "aggregate.csv": "cross-seed stability summary",
            "combined_pareto.csv": "non-dominated union over seeds",
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
