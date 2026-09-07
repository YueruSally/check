from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

from .config import load_config
from .evidence_workbook import read_baseline_inputs
from .experiment import run_batch_experiment
from .io import load_model_data, select_case
from .nsga2 import load_nsga2_config, run_nsga2
from .validation import validate_model_data


def _write_nsga2_result(result, output_directory: Path, metadata: dict) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    front_rows = []
    solutions = []
    for solution_id, individual in enumerate(result.pareto_front, start=1):
        evaluation = individual.evaluation
        assert evaluation is not None
        front_rows.append(
            {
                "solution_id": solution_id,
                "total_cost_usd": evaluation.total_cost_usd,
                "quantity_weighted_mean_delivery_time_h": (
                    evaluation.quantity_weighted_mean_delivery_time_h
                ),
                "makespan_h": evaluation.makespan_h,
                "max_lead_time_h": evaluation.max_lead_time_h,
                "total_tardiness_feu_h": evaluation.total_tardiness_feu_h,
                "route_allocations": len(individual.allocations),
            }
        )
        solutions.append(
            {
                "solution_id": solution_id,
                "objectives": {
                    "total_cost_usd": evaluation.total_cost_usd,
                    "quantity_weighted_mean_delivery_time_h": (
                        evaluation.quantity_weighted_mean_delivery_time_h
                    ),
                },
                "max_lead_time_h": evaluation.max_lead_time_h,
                "total_tardiness_feu_h": evaluation.total_tardiness_feu_h,
                "allocations": [
                    {
                        "shipment_id": item.shipment_id,
                        "quantity_feu": item.quantity_feu,
                        "arc_ids": list(item.arc_ids),
                    }
                    for item in individual.allocations
                ],
            }
        )
    with (output_directory / "pareto_front.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "solution_id",
            "total_cost_usd",
            "quantity_weighted_mean_delivery_time_h",
            "makespan_h",
            "max_lead_time_h",
            "total_tardiness_feu_h",
            "route_allocations",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(front_rows)

    payload = {
        "metadata": metadata,
        "summary": {
            "population_size": len(result.population),
            "pareto_front_size": len(result.pareto_front),
            "candidate_paths_per_shipment": {
                shipment_id: len(paths)
                for shipment_id, paths in result.candidate_library.items()
            },
        },
        "history": [asdict(item) for item in result.history],
        "solutions": solutions,
    }
    (output_directory / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="China-US multimodal model utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evidence = subparsers.add_parser("validate-evidence")
    evidence.add_argument("workbook")

    network = subparsers.add_parser("validate-network")
    network.add_argument("data_directory")
    network.add_argument("--config", default="configs/nsga2_pilot.toml")
    network.add_argument("--scenario")

    optimize = subparsers.add_parser("run-nsga2")
    optimize.add_argument("data_directory")
    optimize.add_argument("--config", default="configs/nsga2_pilot.toml")
    optimize.add_argument("--scenario")
    optimize.add_argument("--case", required=True, choices=("D1", "D2", "D3"))
    optimize.add_argument("--population", type=int)
    optimize.add_argument("--generations", type=int)
    optimize.add_argument("--seed", type=int)
    optimize.add_argument("--output", type=Path, required=True)

    batch = subparsers.add_parser("run-experiment")
    batch.add_argument("data_directory")
    batch.add_argument("--config", default="configs/nsga2_pilot.toml")
    batch.add_argument("--cases", nargs="+", default=["D1", "D2", "D3"])
    batch.add_argument("--scenarios", nargs="+", default=["pilot_baseline"])
    batch.add_argument("--seed-start", type=int, default=20260906)
    batch.add_argument("--runs", type=int, default=30)
    batch.add_argument("--workers", type=int, default=1)
    batch.add_argument("--population", type=int)
    batch.add_argument("--generations", type=int)
    batch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate-evidence":
        values = read_baseline_inputs(args.workbook)
        print(f"Validated {len(values)} baseline evidence inputs.")
        return

    if args.command == "run-experiment":
        if args.runs < 1:
            parser.error("--runs must be positive")
        seeds = range(args.seed_start, args.seed_start + args.runs)
        runs = run_batch_experiment(
            args.data_directory,
            args.config,
            args.output,
            args.cases,
            args.scenarios,
            seeds,
            workers=args.workers,
            population_size=args.population,
            generations=args.generations,
        )
        print(
            f"Completed {len(runs)} independent NSGA-II runs; "
            f"summary written to {args.output}."
        )
        return

    config = load_config(args.config, args.scenario)
    data = load_model_data(args.data_directory)
    validate_model_data(data)
    if args.command == "validate-network":
        cases = sorted({item.case_id for item in data.shipments.values() if item.case_id})
        print(
            f"Validated {len(data.nodes)} nodes, {len(data.arcs)} arcs, "
            f"{len(data.services)} services and {len(data.shipments)} shipments "
            f"across cases {cases} for scenario {config.scenario.name}."
        )
        return

    case_data = select_case(data, args.case)
    algorithm = load_nsga2_config(
        args.config,
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )
    started = time.perf_counter()
    result = run_nsga2(case_data, config, algorithm)
    elapsed_seconds = time.perf_counter() - started
    conversion_path = Path(args.data_directory) / "conversion_report.json"
    conversion = (
        json.loads(conversion_path.read_text(encoding="utf-8"))
        if conversion_path.exists()
        else {}
    )
    _write_nsga2_result(
        result,
        args.output,
        {
            "case": args.case,
            "scenario": config.scenario.name,
            "quantity_unit": config.quantity_unit,
            "algorithm": asdict(algorithm),
            "elapsed_seconds": elapsed_seconds,
            "source_workbook": conversion.get("source_workbook"),
            "source_sha256": conversion.get("source_sha256"),
            "note": "Pipeline validation run; not a paper result unless replicated under the experiment protocol.",
        },
    )
    print(
        f"NSGA-II completed for {args.case}: {len(result.pareto_front)} feasible "
        f"non-dominated solutions written to {args.output}."
    )


if __name__ == "__main__":
    main()
