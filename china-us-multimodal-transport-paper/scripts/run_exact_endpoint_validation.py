#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import capacity_usage_rows
from china_us_multimodal.exact_endpoint import (
    build_exact_endpoint_model,
    solve_lexicographic_endpoint,
)
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.nsga2 import load_nsga2_config


CONDITIONS = (
    # The first condition is the evaluator/MILP consistency preflight.
    ("D2", "delay_264h", "allowed", "uswc_departure_delay_264h"),
    ("D2", "delay_264h", "restricted", "uswc_departure_delay_264h_no_ny_nj"),
    ("D1", "delay_288h", "allowed", "uswc_departure_delay_288h"),
    ("D1", "delay_288h", "restricted", "uswc_departure_delay_288h_no_ny_nj"),
    ("D2", "delay_288h", "allowed", "uswc_departure_delay_288h"),
    ("D2", "delay_288h", "restricted", "uswc_departure_delay_288h_no_ny_nj"),
    ("D3", "delay_240h", "allowed", "uswc_departure_delay_240h"),
    ("D3", "delay_240h", "restricted", "uswc_departure_delay_240h_no_ny_nj"),
    (
        "D1",
        "compound_30pct_delay_168h",
        "allowed",
        "uswc_compound_30pct_delay_168h",
    ),
    (
        "D1",
        "compound_30pct_delay_168h",
        "restricted",
        "uswc_compound_30pct_delay_168h_no_ny_nj",
    ),
)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gateway_for_path(data, arc_ids: tuple[str, ...]) -> str:
    for arc_id in arc_ids:
        arc = data.arcs[arc_id]
        for node_id in (arc.origin, arc.destination):
            if data.nodes[node_id].kind.value == "us_port":
                return node_id
    return "unclassified"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate selected minimum-cost and minimum-time endpoints with a "
            "two-stage path-allocation MILP and the independent evaluator."
        )
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--heuristic-final-results",
        type=Path,
        default=Path("results/v12_final_closed_fronts_30seeds"),
    )
    parser.add_argument(
        "--historical-v07-results",
        type=Path,
        default=Path("results/v07_activation_scenarios_30seeds"),
    )
    parser.add_argument(
        "--historical-v10-results",
        type=Path,
        default=Path("results/v10_ny_nj_counterfactual_30seeds"),
    )
    parser.add_argument("--time-limit-seconds", type=float, default=600.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--primary-relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--primary-absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.time_limit_seconds <= 0:
        parser.error("--time-limit-seconds must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    full_data = load_model_data(args.data)
    algorithm = load_nsga2_config(args.config)
    summary_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []

    completed_tasks = 0
    planned_tasks = len(CONDITIONS) * 2
    for condition_index, (case_id, comparison, access, scenario) in enumerate(
        CONDITIONS, start=1
    ):
        data = select_case(full_data, case_id)
        config = load_config(args.config, scenario)
        model = build_exact_endpoint_model(
            data,
            config,
            max_paths_per_gateway=algorithm.max_paths_per_gateway,
        )
        for primary in ("time", "cost"):
            completed_tasks += 1
            endpoint = "min_time_then_cost" if primary == "time" else "min_cost_then_time"
            print(
                f"[{completed_tasks}/{planned_tasks}] {case_id} {comparison} "
                f"{access} {endpoint}",
                flush=True,
            )
            result = solve_lexicographic_endpoint(
                model,
                primary_objective=primary,
                time_limit_seconds=args.time_limit_seconds,
                mip_relative_gap=args.mip_relative_gap,
                primary_relative_tolerance=args.primary_relative_tolerance,
                primary_absolute_tolerance=args.primary_absolute_tolerance,
            )
            validation_id = f"{case_id}_{comparison}_{access}_{endpoint}"
            evaluation = result.evaluation
            gateway_quantities: dict[str, float] = {}
            for allocation in result.allocations:
                gateway = gateway_for_path(data, allocation.arc_ids)
                gateway_quantities[gateway] = (
                    gateway_quantities.get(gateway, 0.0) + allocation.quantity_feu
                )
            total_quantity = sum(gateway_quantities.values())
            summary_rows.append(
                {
                    "validation_id": validation_id,
                    "case_id": case_id,
                    "comparison": comparison,
                    "gateway_access": access,
                    "scenario": scenario,
                    "endpoint": endpoint,
                    "primary_objective": result.primary_objective,
                    "secondary_objective": result.secondary_objective,
                    "candidate_paths": model.quantity_variable_count,
                    "binary_variables": model.quantity_variable_count,
                    "stage_one_proven": int(result.stage_one_proven),
                    "lexicographic_proven": int(result.lexicographic_proven),
                    "evaluator_consistent": int(result.evaluator_consistent),
                    "primary_optimum": result.primary_optimum,
                    "primary_tolerance": result.primary_tolerance,
                    "total_cost_usd": (
                        evaluation.total_cost_usd if evaluation is not None else ""
                    ),
                    "quantity_weighted_mean_delivery_time_h": (
                        evaluation.quantity_weighted_mean_delivery_time_h
                        if evaluation is not None
                        else ""
                    ),
                    "feasible": int(evaluation.feasible) if evaluation is not None else 0,
                    "violations": (
                        " | ".join(evaluation.violations)
                        if evaluation is not None
                        else "no incumbent"
                    ),
                    "evaluator_cost_abs_error": result.evaluator_cost_abs_error,
                    "evaluator_time_abs_error": result.evaluator_time_abs_error,
                    "resource_max_abs_error_feu": result.resource_max_abs_error_feu,
                    "ny_nj_share": (
                        gateway_quantities.get("ny_nj", 0.0) / total_quantity
                        if total_quantity
                        else ""
                    ),
                    "la_lb_share": (
                        gateway_quantities.get("la_lb", 0.0) / total_quantity
                        if total_quantity
                        else ""
                    ),
                    "seattle_tacoma_share": (
                        gateway_quantities.get("seattle_tacoma", 0.0) / total_quantity
                        if total_quantity
                        else ""
                    ),
                    "solver_calls": len(result.stage_calls),
                }
            )
            for call in result.stage_calls:
                solver_rows.append(
                    {
                        "validation_id": validation_id,
                        **asdict(call),
                        "time_limit_seconds": args.time_limit_seconds,
                        "requested_mip_relative_gap": args.mip_relative_gap,
                    }
                )
            for allocation_index, allocation in enumerate(result.allocations, start=1):
                allocation_rows.append(
                    {
                        "validation_id": validation_id,
                        "allocation_id": allocation_index,
                        "shipment_id": allocation.shipment_id,
                        "quantity_feu": allocation.quantity_feu,
                        "gateway": gateway_for_path(data, allocation.arc_ids),
                        "arc_ids": "|".join(allocation.arc_ids),
                    }
                )
            if result.allocations:
                for row in capacity_usage_rows(data, list(result.allocations), config):
                    capacity_rows.append({"validation_id": validation_id, **row})

            # The first endpoint is the preflight. Do not spend the remaining
            # solver budget if the independent evaluator/accounting disagrees.
            if condition_index == 1 and primary == "time" and not result.evaluator_consistent:
                write_outputs(
                    args.output,
                    summary_rows,
                    solver_rows,
                    allocation_rows,
                    capacity_rows,
                )
                raise RuntimeError(
                    "MILP/evaluator preflight failed; remaining endpoint tasks were not run."
                )

    write_outputs(
        args.output,
        summary_rows,
        solver_rows,
        allocation_rows,
        capacity_rows,
    )
    comparison_rows: list[dict[str, object]] = []
    indexed = {
        (row["case_id"], row["comparison"], row["gateway_access"], row["endpoint"]): row
        for row in summary_rows
    }
    for case_id, comparison in sorted(
        {(row["case_id"], row["comparison"]) for row in summary_rows}
    ):
        for endpoint in ("min_cost_then_time", "min_time_then_cost"):
            allowed = indexed.get((case_id, comparison, "allowed", endpoint))
            restricted = indexed.get((case_id, comparison, "restricted", endpoint))
            if allowed is None or restricted is None:
                continue
            allowed_cost = allowed["total_cost_usd"]
            restricted_cost = restricted["total_cost_usd"]
            allowed_time = allowed["quantity_weighted_mean_delivery_time_h"]
            restricted_time = restricted["quantity_weighted_mean_delivery_time_h"]
            comparison_rows.append(
                {
                    "case_id": case_id,
                    "comparison": comparison,
                    "endpoint": endpoint,
                    "allowed_lexicographic_proven": allowed["lexicographic_proven"],
                    "restricted_lexicographic_proven": restricted[
                        "lexicographic_proven"
                    ],
                    "comparison_status": (
                        "proven_lexicographic_endpoints"
                        if int(allowed["lexicographic_proven"])
                        and int(restricted["lexicographic_proven"])
                        else "incumbent_or_partially_proven_only"
                    ),
                    "cost_avoided_by_allowing_ny_nj_usd": (
                        float(restricted_cost) - float(allowed_cost)
                        if restricted_cost != "" and allowed_cost != ""
                        else ""
                    ),
                    "time_avoided_by_allowing_ny_nj_h": (
                        float(restricted_time) - float(allowed_time)
                        if restricted_time != "" and allowed_time != ""
                        else ""
                    ),
                    "allowed_ny_nj_share": allowed["ny_nj_share"],
                }
            )
    write_csv(
        args.output / "exact_counterfactual_value.csv",
        comparison_rows,
        [
            "case_id", "comparison", "endpoint", "allowed_lexicographic_proven",
            "restricted_lexicographic_proven", "comparison_status",
            "cost_avoided_by_allowing_ny_nj_usd",
            "time_avoided_by_allowing_ny_nj_h", "allowed_ny_nj_share",
        ],
    )
    heuristic_rows_by_case: dict[str, list[dict[str, str]]] = {}
    for case_id in {condition[0] for condition in CONDITIONS}:
        heuristic_rows_by_case[case_id] = read_csv(
            args.heuristic_final_results / case_id / "runs.csv"
        )
    historical_v07 = read_csv(args.historical_v07_results / "runs.csv")
    historical_v10 = {
        case_id: read_csv(args.historical_v10_results / case_id / "runs.csv")
        for case_id in {condition[0] for condition in CONDITIONS}
        if (args.historical_v10_results / case_id / "runs.csv").exists()
    }
    quality_rows: list[dict[str, object]] = []
    for exact_row in summary_rows:
        case_id = str(exact_row["case_id"])
        scenario = str(exact_row["scenario"])
        access = str(exact_row["gateway_access"])
        comparison = str(exact_row["comparison"])
        if comparison == "compound_30pct_delay_168h":
            source_name = "v07" if access == "allowed" else "v10"
            source_rows = (
                historical_v07 if access == "allowed" else historical_v10[case_id]
            )
        else:
            source_name = "v12_targeted_rerun"
            source_rows = heuristic_rows_by_case[case_id]
        feasible_rows = [
            row
            for row in source_rows
            if row["case_id"] == case_id
            and row["scenario"] == scenario
            and int(row.get("feasible_population", "0") or 0) > 0
        ]
        endpoint = str(exact_row["endpoint"])
        field = (
            "min_cost_usd"
            if endpoint == "min_cost_then_time"
            else "min_weighted_mean_delivery_time_h"
        )
        heuristic_best = min(
            (float(row[field]) for row in feasible_rows if row.get(field)),
            default=None,
        )
        exact_value_raw = exact_row["primary_optimum"]
        exact_value = (
            float(exact_value_raw) if exact_value_raw not in {"", None} else None
        )
        absolute_gap = (
            heuristic_best - exact_value
            if heuristic_best is not None and exact_value is not None
            else None
        )
        relative_gap = (
            absolute_gap / abs(exact_value)
            if absolute_gap is not None and exact_value not in {None, 0.0}
            else None
        )
        quality_rows.append(
            {
                "validation_id": exact_row["validation_id"],
                "case_id": case_id,
                "comparison": comparison,
                "gateway_access": access,
                "scenario": scenario,
                "endpoint": endpoint,
                "heuristic_source": source_name,
                "heuristic_feasible_runs": len(feasible_rows),
                "heuristic_best_primary_value": heuristic_best,
                "milp_primary_value": exact_value,
                "heuristic_minus_milp_absolute_gap": absolute_gap,
                "heuristic_minus_milp_relative_gap": relative_gap,
                "milp_lexicographic_proven": exact_row["lexicographic_proven"],
                "comparison_interpretation": (
                    "optimality_gap_against_proven_endpoint"
                    if int(exact_row["lexicographic_proven"])
                    else "incumbent_comparison_only"
                ),
                "negative_gap_flag": int(
                    absolute_gap is not None
                    and absolute_gap
                    < -max(1e-5, 1e-9 * abs(exact_value or 0.0))
                ),
            }
        )
    write_csv(
        args.output / "heuristic_endpoint_quality.csv",
        quality_rows,
        [
            "validation_id", "case_id", "comparison", "gateway_access",
            "scenario", "endpoint", "heuristic_source",
            "heuristic_feasible_runs", "heuristic_best_primary_value",
            "milp_primary_value", "heuristic_minus_milp_absolute_gap",
            "heuristic_minus_milp_relative_gap", "milp_lexicographic_proven",
            "comparison_interpretation", "negative_gap_flag",
        ],
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": [
            {
                "case_id": case_id,
                "comparison": comparison,
                "gateway_access": access,
                "scenario": scenario,
            }
            for case_id, comparison, access, scenario in CONDITIONS
        ],
        "endpoint_tasks": planned_tasks,
        "maximum_solver_calls": planned_tasks * 2,
        "time_limit_seconds_per_call": args.time_limit_seconds,
        "mip_relative_gap": args.mip_relative_gap,
        "primary_relative_tolerance": args.primary_relative_tolerance,
        "primary_absolute_tolerance": args.primary_absolute_tolerance,
        "interpretation": (
            "lexicographic_proven=1 requires both solver stages to terminate "
            "optimally within the recorded tolerance and the decoded allocation "
            "to match the independent evaluator and all three capacity accounts"
        ),
        "quality_comparison": (
            "heuristic_endpoint_quality.csv compares each best 30-seed NSGA-II "
            "primary endpoint with the corresponding MILP value; a formal gap is "
            "claimed only when the MILP endpoint is proven"
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exact endpoint validation written to {args.output}.", flush=True)


def write_outputs(
    output: Path,
    summary_rows: list[dict[str, object]],
    solver_rows: list[dict[str, object]],
    allocation_rows: list[dict[str, object]],
    capacity_rows: list[dict[str, object]],
) -> None:
    write_csv(
        output / "exact_endpoint_summary.csv",
        summary_rows,
        [
            "validation_id", "case_id", "comparison", "gateway_access",
            "scenario", "endpoint", "primary_objective", "secondary_objective",
            "candidate_paths", "binary_variables", "stage_one_proven",
            "lexicographic_proven", "evaluator_consistent", "primary_optimum",
            "primary_tolerance", "total_cost_usd",
            "quantity_weighted_mean_delivery_time_h", "feasible", "violations",
            "evaluator_cost_abs_error", "evaluator_time_abs_error",
            "resource_max_abs_error_feu", "ny_nj_share", "la_lb_share",
            "seattle_tacoma_share", "solver_calls",
        ],
    )
    write_csv(
        output / "exact_solver_calls.csv",
        solver_rows,
        [
            "validation_id", "stage", "objective", "status_code", "status",
            "success", "objective_value", "dual_bound", "relative_gap",
            "node_count", "elapsed_seconds", "message", "time_limit_seconds",
            "requested_mip_relative_gap",
        ],
    )
    write_csv(
        output / "exact_endpoint_allocations.csv",
        allocation_rows,
        [
            "validation_id", "allocation_id", "shipment_id", "quantity_feu",
            "gateway", "arc_ids",
        ],
    )
    write_csv(
        output / "exact_endpoint_capacity_usage.csv",
        capacity_rows,
        [
            "validation_id", "resource_type", "resource_id", "period", "origin",
            "destination", "used_feu", "capacity_feu", "slack_feu",
            "utilization_pct", "binding_90pct", "binding_95pct", "binding_100pct",
        ],
    )


if __name__ == "__main__":
    main()
