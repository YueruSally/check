#!/usr/bin/env python3
"""Run the pre-registered exact NY/NJ value and endpoint-share experiment.

The runner is intentionally resumable.  Every endpoint, share-bound solve, and
feasibility check is written to ``checkpoints/`` immediately.  Re-running the
same command skips completed tasks.  Compatible v1.2 endpoint records are
imported rather than solved again.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import capacity_usage_rows, route_component_rows
from china_us_multimodal.domain import RouteAllocation
from china_us_multimodal.evaluator import _scenario_ocean_departure_delay_h
from china_us_multimodal.exact_endpoint import (
    build_exact_endpoint_model,
    solve_bounded_milp,
    solve_feasibility,
    solve_lexicographic_endpoint,
)
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.schedule import earliest_departure
from china_us_multimodal.supplementary_design import (
    CASES,
    ExactCondition,
    condition_config,
    core_conditions,
)


ENDPOINTS = (
    ("min_cost_then_time", "cost"),
    ("min_time_then_cost", "time"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def gateway_for_path(data, arc_ids: Iterable[str]) -> str:
    for arc_id in arc_ids:
        arc = data.arcs[arc_id]
        for node_id in (arc.origin, arc.destination):
            if data.nodes[node_id].kind.value == "us_port":
                return node_id
    return "unclassified"


def allocations_from_payload(payload: dict[str, object]) -> tuple[RouteAllocation, ...]:
    return tuple(
        RouteAllocation(
            shipment_id=str(row["shipment_id"]),
            quantity_feu=float(row["quantity_feu"]),
            arc_ids=tuple(str(item) for item in row["arc_ids"]),
        )
        for row in payload.get("allocations", [])
    )


def gateway_shares(data, allocations: Iterable[RouteAllocation]) -> dict[str, float]:
    quantities: dict[str, float] = {}
    total = 0.0
    for allocation in allocations:
        gateway = gateway_for_path(data, allocation.arc_ids)
        quantities[gateway] = quantities.get(gateway, 0.0) + allocation.quantity_feu
        total += allocation.quantity_feu
    return {
        gateway: quantities.get(gateway, 0.0) / total if total else 0.0
        for gateway in ("ny_nj", "la_lb", "seattle_tacoma")
    }


def endpoint_id(
    case_id: str, condition_id: str, gateway_access: str, endpoint: str
) -> str:
    return f"{case_id}_{condition_id}_{gateway_access}_{endpoint}"


def endpoint_payload(
    *,
    condition: ExactCondition,
    gateway_access: str,
    endpoint: str,
    model,
    result,
    reused_from: str = "",
    max_paths_per_gateway: int = 4,
) -> dict[str, object]:
    evaluation = result.evaluation
    shares = gateway_shares(model.data, result.allocations)
    calls = [asdict(call) for call in result.stage_calls]
    return {
        "validation_id": endpoint_id(
            condition.case_id, condition.condition_id, gateway_access, endpoint
        ),
        "case_id": condition.case_id,
        "condition_id": condition.condition_id,
        "family": condition.family,
        "gateway_access": gateway_access,
        "scenario": model.config.scenario.name,
        "uswc_capacity_multiplier": condition.uswc_capacity_multiplier,
        "uswc_delay_h": condition.uswc_delay_h,
        "endpoint": endpoint,
        "primary_objective": result.primary_objective,
        "secondary_objective": result.secondary_objective,
        "max_paths_per_gateway": max_paths_per_gateway,
        "candidate_paths": model.quantity_variable_count,
        "binary_variables": model.quantity_variable_count,
        "stage_one_proven": int(result.stage_one_proven),
        "lexicographic_proven": int(result.lexicographic_proven),
        "evaluator_consistent": int(result.evaluator_consistent),
        "primary_optimum": result.primary_optimum,
        "primary_tolerance": result.primary_tolerance,
        "total_cost_usd": evaluation.total_cost_usd if evaluation else None,
        "quantity_weighted_mean_delivery_time_h": (
            evaluation.quantity_weighted_mean_delivery_time_h if evaluation else None
        ),
        "feasible": int(evaluation.feasible) if evaluation else 0,
        "violations": "|".join(evaluation.violations) if evaluation else "",
        "evaluator_cost_abs_error": result.evaluator_cost_abs_error,
        "evaluator_time_abs_error": result.evaluator_time_abs_error,
        "resource_max_abs_error_feu": result.resource_max_abs_error_feu,
        "ny_nj_share": shares["ny_nj"],
        "la_lb_share": shares["la_lb"],
        "seattle_tacoma_share": shares["seattle_tacoma"],
        "solver_calls": calls,
        "solver_call_count": len(calls),
        "solver_runtime_seconds": sum(
            float(call["elapsed_seconds"] or 0.0) for call in calls
        ),
        "maximum_relative_gap": max(
            (
                float(call["relative_gap"])
                for call in calls
                if call["relative_gap"] is not None
            ),
            default=None,
        ),
        "reused_from": reused_from,
        "allocations": [
            {
                "shipment_id": allocation.shipment_id,
                "quantity_feu": allocation.quantity_feu,
                "gateway": gateway_for_path(model.data, allocation.arc_ids),
                "arc_ids": list(allocation.arc_ids),
            }
            for allocation in result.allocations
        ],
    }


def load_reuse_payloads(root: Path) -> dict[tuple[str, str, str, str], dict[str, object]]:
    summary_rows = read_csv(root / "exact_endpoint_summary.csv")
    allocation_rows = read_csv(root / "exact_endpoint_allocations.csv")
    solver_rows = read_csv(root / "exact_solver_calls.csv")
    allocations: dict[str, list[dict[str, object]]] = {}
    for row in allocation_rows:
        allocations.setdefault(row["validation_id"], []).append(
            {
                "shipment_id": row["shipment_id"],
                "quantity_feu": float(row["quantity_feu"]),
                "gateway": row["gateway"],
                "arc_ids": row["arc_ids"].split("|"),
            }
        )
    calls: dict[str, list[dict[str, object]]] = {}
    for row in solver_rows:
        calls.setdefault(row["validation_id"], []).append(
            {
                "stage": int(row["stage"]),
                "objective": row["objective"],
                "status_code": int(row["status_code"]),
                "status": row["status"],
                "success": row["success"].lower() == "true",
                "objective_value": float(row["objective_value"])
                if row["objective_value"]
                else None,
                "dual_bound": float(row["dual_bound"])
                if row["dual_bound"]
                else None,
                "relative_gap": float(row["relative_gap"])
                if row["relative_gap"]
                else None,
                "node_count": int(row["node_count"]) if row["node_count"] else None,
                "elapsed_seconds": float(row["elapsed_seconds"])
                if row["elapsed_seconds"]
                else None,
                "message": row["message"],
            }
        )
    reused: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in summary_rows:
        key = (
            row["case_id"],
            row["comparison"],
            row["gateway_access"],
            row["endpoint"],
        )
        validation_id = row["validation_id"]
        reused[key] = {
            "source_validation_id": validation_id,
            "summary": row,
            "allocations": allocations.get(validation_id, []),
            "solver_calls": calls.get(validation_id, []),
        }
    return reused


def import_reused_endpoint(
    *,
    source: dict[str, object],
    condition: ExactCondition,
    gateway_access: str,
    endpoint: str,
) -> dict[str, object]:
    row = source["summary"]
    calls = source["solver_calls"]
    return {
        "validation_id": endpoint_id(
            condition.case_id, condition.condition_id, gateway_access, endpoint
        ),
        "case_id": condition.case_id,
        "condition_id": condition.condition_id,
        "family": condition.family,
        "gateway_access": gateway_access,
        "scenario": f"supp_{condition.condition_id}"
        + ("_no_ny_nj" if gateway_access == "restricted" else ""),
        "uswc_capacity_multiplier": condition.uswc_capacity_multiplier,
        "uswc_delay_h": condition.uswc_delay_h,
        "endpoint": endpoint,
        "primary_objective": row["primary_objective"],
        "secondary_objective": row["secondary_objective"],
        "max_paths_per_gateway": 4,
        "candidate_paths": int(row["candidate_paths"]),
        "binary_variables": int(row["binary_variables"]),
        "stage_one_proven": int(row["stage_one_proven"]),
        "lexicographic_proven": int(row["lexicographic_proven"]),
        "evaluator_consistent": int(row["evaluator_consistent"]),
        "primary_optimum": float(row["primary_optimum"])
        if row["primary_optimum"]
        else None,
        "primary_tolerance": float(row["primary_tolerance"])
        if row["primary_tolerance"]
        else None,
        "total_cost_usd": float(row["total_cost_usd"])
        if row["total_cost_usd"]
        else None,
        "quantity_weighted_mean_delivery_time_h": float(
            row["quantity_weighted_mean_delivery_time_h"]
        )
        if row["quantity_weighted_mean_delivery_time_h"]
        else None,
        "feasible": int(row["feasible"]),
        "violations": row["violations"],
        "evaluator_cost_abs_error": float(row["evaluator_cost_abs_error"])
        if row["evaluator_cost_abs_error"]
        else None,
        "evaluator_time_abs_error": float(row["evaluator_time_abs_error"])
        if row["evaluator_time_abs_error"]
        else None,
        "resource_max_abs_error_feu": float(row["resource_max_abs_error_feu"])
        if row["resource_max_abs_error_feu"]
        else None,
        "ny_nj_share": float(row["ny_nj_share"]),
        "la_lb_share": float(row["la_lb_share"]),
        "seattle_tacoma_share": float(row["seattle_tacoma_share"]),
        "solver_calls": calls,
        "solver_call_count": len(calls),
        "solver_runtime_seconds": sum(
            float(call["elapsed_seconds"] or 0.0) for call in calls
        ),
        "maximum_relative_gap": max(
            (
                float(call["relative_gap"])
                for call in calls
                if call["relative_gap"] is not None
            ),
            default=None,
        ),
        "reused_from": (
            "results/v12_exact_endpoint_validation/"
            + str(source["source_validation_id"])
        ),
        "allocations": source["allocations"],
    }


def numerical_tolerance(
    value: float, absolute_tolerance: float, relative_tolerance: float
) -> float:
    return absolute_tolerance + relative_tolerance * abs(value)


def endpoint_region_bounds(
    payload: dict[str, object],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[float, float]:
    cost = float(payload["total_cost_usd"])
    time_h = float(payload["quantity_weighted_mean_delivery_time_h"])
    primary = str(payload["primary_objective"])
    primary_optimum = float(payload["primary_optimum"])
    primary_tolerance = float(payload["primary_tolerance"])
    if primary == "cost":
        return (
            primary_optimum + primary_tolerance,
            time_h + numerical_tolerance(
                time_h, absolute_tolerance, relative_tolerance
            ),
        )
    return (
        cost + numerical_tolerance(cost, absolute_tolerance, relative_tolerance),
        primary_optimum + primary_tolerance,
    )


def endpoints_coincident(
    left: dict[str, object],
    right: dict[str, object],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    left_cost = float(left["total_cost_usd"])
    right_cost = float(right["total_cost_usd"])
    left_time = float(left["quantity_weighted_mean_delivery_time_h"])
    right_time = float(right["quantity_weighted_mean_delivery_time_h"])
    cost_tolerance = max(
        numerical_tolerance(left_cost, absolute_tolerance, relative_tolerance),
        numerical_tolerance(right_cost, absolute_tolerance, relative_tolerance),
    )
    time_tolerance = max(
        numerical_tolerance(left_time, absolute_tolerance, relative_tolerance),
        numerical_tolerance(right_time, absolute_tolerance, relative_tolerance),
    )
    return (
        abs(left_cost - right_cost) <= cost_tolerance
        and abs(left_time - right_time) <= time_tolerance
    )


def share_bound_payload(
    *,
    bound_id: str,
    condition: ExactCondition,
    endpoint_region: str,
    direction: str,
    cost_upper_bound: float,
    time_upper_bound: float,
    result,
    data,
) -> dict[str, object]:
    shares = gateway_shares(data, result.allocations)
    return {
        "bound_id": bound_id,
        "case_id": condition.case_id,
        "condition_id": condition.condition_id,
        "family": condition.family,
        "endpoint_region": endpoint_region,
        "direction": direction,
        "cost_upper_bound_usd": cost_upper_bound,
        "time_upper_bound_h": time_upper_bound,
        "proven_optimal": int(result.proven_optimal),
        "proven_infeasible": int(result.proven_infeasible),
        "evaluator_consistent": int(result.evaluator_consistent),
        "ny_nj_share": shares["ny_nj"] if result.evaluation else None,
        "total_cost_usd": (
            result.evaluation.total_cost_usd if result.evaluation else None
        ),
        "quantity_weighted_mean_delivery_time_h": (
            result.evaluation.quantity_weighted_mean_delivery_time_h
            if result.evaluation
            else None
        ),
        "solver_call": asdict(result.solver_call),
        "allocations": [
            {
                "shipment_id": allocation.shipment_id,
                "quantity_feu": allocation.quantity_feu,
                "gateway": gateway_for_path(data, allocation.arc_ids),
                "arc_ids": list(allocation.arc_ids),
            }
            for allocation in result.allocations
        ],
    }


def service_leg_rows(data, allocations, config, validation_id: str):
    rows: list[dict[str, object]] = []
    for allocation_number, allocation in enumerate(allocations, start=1):
        current_h = data.shipments[allocation.shipment_id].release_h
        previous = None
        gateway = gateway_for_path(data, allocation.arc_ids)
        for leg_number, arc_id in enumerate(allocation.arc_ids, start=1):
            arc = data.arcs[arc_id]
            transfer_h = 0.0
            if previous is not None:
                rule = data.transfer_rules.get((arc.origin, previous.mode, arc.mode))
                if rule is not None:
                    transfer_h = rule.duration_h
                    current_h += transfer_h
            ready_h = current_h
            service_id = ""
            published_departure_h = ready_h
            actual_departure_h = ready_h
            if config.constraints.enforce_timetable and arc.service_ids:
                services = [data.services[item] for item in arc.service_ids]
                scheduled = earliest_departure(services, ready_h)
                service_id = scheduled.service_id
                published_departure_h = scheduled.departure_h
                actual_departure_h = published_departure_h + (
                    _scenario_ocean_departure_delay_h(arc, config)
                )
            arrival_h = actual_departure_h + arc.transit_time_h
            rows.append(
                {
                    "validation_id": validation_id,
                    "allocation_id": allocation_number,
                    "shipment_id": allocation.shipment_id,
                    "quantity_feu": allocation.quantity_feu,
                    "gateway": gateway,
                    "leg_number": leg_number,
                    "arc_id": arc_id,
                    "origin": arc.origin,
                    "destination": arc.destination,
                    "mode": arc.mode.value,
                    "service_id": service_id,
                    "transfer_before_leg_h": transfer_h,
                    "ready_h": ready_h,
                    "published_departure_h": published_departure_h,
                    "actual_departure_h": actual_departure_h,
                    "arrival_h": arrival_h,
                }
            )
            current_h = arrival_h
            previous = arc
    return rows


def endpoint_summary_row(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key not in {"solver_calls", "allocations"}}


def run_endpoints(args, data, base_config, conditions, checkpoints):
    reused = load_reuse_payloads(args.reuse_exact_root)
    total = len(conditions) * 2 * len(ENDPOINTS)
    completed = 0
    for condition in conditions:
        case_data = select_case(data, condition.case_id)
        for access in ("allowed", "restricted"):
            pending: list[tuple[str, str, Path]] = []
            for endpoint, primary in ENDPOINTS:
                checkpoint = checkpoints / "endpoints" / (
                    endpoint_id(condition.case_id, condition.condition_id, access, endpoint)
                    + ".json"
                )
                if checkpoint.exists():
                    completed += 1
                    print(f"[{completed}/{total}] resume {checkpoint.stem}", flush=True)
                    continue
                key = (condition.case_id, condition.condition_id, access, endpoint)
                if key in reused:
                    payload = import_reused_endpoint(
                        source=reused[key],
                        condition=condition,
                        gateway_access=access,
                        endpoint=endpoint,
                    )
                    write_json_atomic(checkpoint, payload)
                    completed += 1
                    print(f"[{completed}/{total}] reused {checkpoint.stem}", flush=True)
                else:
                    pending.append((endpoint, primary, checkpoint))
            if not pending:
                continue
            config = condition_config(base_config, condition, gateway_access=access)
            model = build_exact_endpoint_model(
                case_data, config, max_paths_per_gateway=4
            )
            for endpoint, primary, checkpoint in pending:
                result = solve_lexicographic_endpoint(
                    model,
                    primary_objective=primary,
                    time_limit_seconds=args.time_limit_seconds,
                    mip_relative_gap=args.mip_relative_gap,
                    primary_relative_tolerance=args.relative_tolerance,
                    primary_absolute_tolerance=args.absolute_tolerance,
                )
                payload = endpoint_payload(
                    condition=condition,
                    gateway_access=access,
                    endpoint=endpoint,
                    model=model,
                    result=result,
                )
                write_json_atomic(checkpoint, payload)
                completed += 1
                print(
                    f"[{completed}/{total}] {checkpoint.stem} "
                    f"proven={payload['lexicographic_proven']} "
                    f"runtime={payload['solver_runtime_seconds']:.3f}s",
                    flush=True,
                )


def run_share_bounds(args, data, base_config, conditions, checkpoints):
    planned_regions: list[tuple[ExactCondition, str, dict[str, object]]] = []
    for condition in conditions:
        endpoint_payloads = {
            endpoint: load_json(
                checkpoints
                / "endpoints"
                / (
                    endpoint_id(
                        condition.case_id, condition.condition_id, "allowed", endpoint
                    )
                    + ".json"
                )
            )
            for endpoint, _ in ENDPOINTS
        }
        if not all(
            int(payload["lexicographic_proven"]) == 1
            for payload in endpoint_payloads.values()
        ):
            continue
        if endpoints_coincident(
            endpoint_payloads["min_cost_then_time"],
            endpoint_payloads["min_time_then_cost"],
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        ):
            planned_regions.append(
                (
                    condition,
                    "coincident_endpoints",
                    endpoint_payloads["min_cost_then_time"],
                )
            )
        else:
            planned_regions.extend(
                (condition, endpoint, payload)
                for endpoint, payload in endpoint_payloads.items()
            )
    total = len(planned_regions) * 2
    completed = 0
    for condition, region, payload in planned_regions:
        case_data = select_case(data, condition.case_id)
        config = condition_config(base_config, condition, gateway_access="allowed")
        model = None
        cost_bound, time_bound = endpoint_region_bounds(
            payload,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        for direction in ("min", "max"):
            bound_id = f"{condition.case_id}_{condition.condition_id}_{region}_{direction}_ny_nj_share"
            checkpoint = checkpoints / "share_bounds" / f"{bound_id}.json"
            if checkpoint.exists():
                completed += 1
                print(f"[share {completed}/{total}] resume {bound_id}", flush=True)
                continue
            if model is None:
                model = build_exact_endpoint_model(
                    case_data, config, max_paths_per_gateway=4
                )
            result = solve_bounded_milp(
                model,
                objective_coefficients=model.gateway_share_coefficients("ny_nj"),
                objective_name="ny_nj_share",
                upper_bounds=(
                    (model.objective_coefficients("cost"), cost_bound),
                    (model.objective_coefficients("time"), time_bound),
                ),
                maximize=direction == "max",
                time_limit_seconds=args.time_limit_seconds,
                mip_relative_gap=args.mip_relative_gap,
            )
            bound_payload = share_bound_payload(
                bound_id=bound_id,
                condition=condition,
                endpoint_region=region,
                direction=direction,
                cost_upper_bound=cost_bound,
                time_upper_bound=time_bound,
                result=result,
                data=case_data,
            )
            write_json_atomic(checkpoint, bound_payload)
            completed += 1
            print(
                f"[share {completed}/{total}] {bound_id} "
                f"proven={bound_payload['proven_optimal']} "
                f"share={bound_payload['ny_nj_share']}",
                flush=True,
            )


def run_feasibility_checks(args, data, base_config, checkpoints):
    for index, case_id in enumerate(CASES, start=1):
        checkpoint = checkpoints / "feasibility" / f"{case_id}_demand150_capacity60.json"
        if checkpoint.exists():
            print(f"[feasibility {index}/3] resume {case_id}", flush=True)
            continue
        condition = ExactCondition(
            case_id=case_id,
            condition_id="demand_150pct_capacity_60pct",
            family="feasibility_diagnostic",
            uswc_capacity_multiplier=0.60,
        )
        case_data = select_case(data, case_id)
        config = condition_config(
            base_config,
            condition,
            gateway_access="allowed",
            demand_multiplier=1.50,
        )
        model = build_exact_endpoint_model(case_data, config, max_paths_per_gateway=4)
        result = solve_feasibility(
            model,
            time_limit_seconds=args.time_limit_seconds,
            mip_relative_gap=args.mip_relative_gap,
        )
        if result.evaluator_consistent and result.evaluation is not None:
            classification = "feasible_solution_found"
        elif result.proven_infeasible:
            classification = "proven_infeasible"
        else:
            classification = "indeterminate_within_time_limit"
        payload = {
            "case_id": case_id,
            "condition_id": condition.condition_id,
            "classification": classification,
            "feasible_solution_found": int(
                result.evaluator_consistent and result.evaluation is not None
            ),
            "proven_infeasible": int(result.proven_infeasible),
            "solver_call": asdict(result.solver_call),
            "allocations": [
                {
                    "shipment_id": item.shipment_id,
                    "quantity_feu": item.quantity_feu,
                    "gateway": gateway_for_path(case_data, item.arc_ids),
                    "arc_ids": list(item.arc_ids),
                }
                for item in result.allocations
            ],
        }
        write_json_atomic(checkpoint, payload)
        print(
            f"[feasibility {index}/3] {case_id} {classification}", flush=True
        )


def finalize_outputs(args, data, base_config, conditions, checkpoints):
    endpoint_payloads = [
        load_json(path)
        for path in sorted((checkpoints / "endpoints").glob("*.json"))
    ]
    endpoint_payloads.sort(
        key=lambda row: (
            row["case_id"],
            row["family"],
            float(row["uswc_delay_h"]),
            float(row["uswc_capacity_multiplier"]),
            row["gateway_access"],
            row["endpoint"],
        )
    )
    summary_rows = [endpoint_summary_row(row) for row in endpoint_payloads]
    summary_fields = [
        "validation_id", "case_id", "condition_id", "family", "gateway_access",
        "scenario", "uswc_capacity_multiplier", "uswc_delay_h", "endpoint",
        "primary_objective", "secondary_objective", "max_paths_per_gateway",
        "candidate_paths", "binary_variables", "stage_one_proven",
        "lexicographic_proven", "evaluator_consistent", "primary_optimum",
        "primary_tolerance", "total_cost_usd",
        "quantity_weighted_mean_delivery_time_h", "feasible", "violations",
        "evaluator_cost_abs_error", "evaluator_time_abs_error",
        "resource_max_abs_error_feu", "ny_nj_share", "la_lb_share",
        "seattle_tacoma_share", "solver_call_count", "solver_runtime_seconds",
        "maximum_relative_gap", "reused_from",
    ]
    write_csv(args.output / "exact_endpoint_summary.csv", summary_rows, summary_fields)

    solver_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    cost_summary_rows: list[dict[str, object]] = []
    condition_index = {
        (item.case_id, item.condition_id): item for item in conditions
    }
    case_cache = {case_id: select_case(data, case_id) for case_id in CASES}
    for payload in endpoint_payloads:
        validation_id = str(payload["validation_id"])
        for call in payload["solver_calls"]:
            solver_rows.append({"validation_id": validation_id, **call})
        allocations = allocations_from_payload(payload)
        for allocation_number, allocation in enumerate(allocations, start=1):
            allocation_rows.append(
                {
                    "validation_id": validation_id,
                    "allocation_id": allocation_number,
                    "shipment_id": allocation.shipment_id,
                    "quantity_feu": allocation.quantity_feu,
                    "gateway": gateway_for_path(
                        case_cache[str(payload["case_id"])], allocation.arc_ids
                    ),
                    "arc_ids": "|".join(allocation.arc_ids),
                }
            )
        condition = condition_index[(str(payload["case_id"]), str(payload["condition_id"]))]
        config = condition_config(
            base_config, condition, gateway_access=str(payload["gateway_access"])
        )
        case_data = case_cache[str(payload["case_id"])]
        leg_rows.extend(service_leg_rows(case_data, allocations, config, validation_id))
        usage = capacity_usage_rows(case_data, list(allocations), config)
        capacity_rows.extend(
            {"validation_id": validation_id, **row} for row in usage
        )
        components = route_component_rows(case_data, list(allocations), config)
        for allocation_number, row in enumerate(components, start=1):
            component_rows.append(
                {
                    "validation_id": validation_id,
                    "allocation_id": allocation_number,
                    **row,
                }
            )
        transport = sum(
            float(row["china_inland_transport_cost_usd"])
            + float(row["ocean_transport_cost_usd"])
            + float(row["us_inland_transport_cost_usd"])
            for row in components
        )
        transfer = sum(float(row["transfer_cost_usd"]) for row in components)
        waiting = sum(float(row["holding_cost_usd"]) for row in components)
        processing = sum(float(row["processing_cost_usd"]) for row in components)
        tardiness = sum(float(row["tardiness_cost_usd"]) for row in components)
        cost_summary_rows.append(
            {
                "validation_id": validation_id,
                "case_id": payload["case_id"],
                "condition_id": payload["condition_id"],
                "gateway_access": payload["gateway_access"],
                "endpoint": payload["endpoint"],
                "transport_cost_usd": transport,
                "transfer_cost_usd": transfer,
                "waiting_holding_cost_usd": waiting,
                "processing_cost_usd": processing,
                "tardiness_cost_usd": tardiness,
                "transport_and_transfer_cost_usd": transport + transfer,
                "waiting_and_processing_cost_usd": waiting + processing,
                "reconstructed_total_cost_usd": (
                    transport + transfer + waiting + processing + tardiness
                ),
                "reported_total_cost_usd": payload["total_cost_usd"],
            }
        )

    write_csv(
        args.output / "exact_solver_calls.csv",
        solver_rows,
        [
            "validation_id", "stage", "objective", "status_code", "status",
            "success", "objective_value", "dual_bound", "relative_gap",
            "node_count", "elapsed_seconds", "message",
        ],
    )
    write_csv(
        args.output / "exact_endpoint_allocations.csv",
        allocation_rows,
        [
            "validation_id", "allocation_id", "shipment_id", "quantity_feu",
            "gateway", "arc_ids",
        ],
    )
    write_csv(
        args.output / "exact_endpoint_service_legs.csv",
        leg_rows,
        [
            "validation_id", "allocation_id", "shipment_id", "quantity_feu",
            "gateway", "leg_number", "arc_id", "origin", "destination", "mode",
            "service_id", "transfer_before_leg_h", "ready_h",
            "published_departure_h", "actual_departure_h", "arrival_h",
        ],
    )
    write_csv(
        args.output / "exact_endpoint_capacity_usage.csv",
        capacity_rows,
        [
            "validation_id", "resource_type", "resource_id", "period", "origin",
            "destination", "used_feu", "capacity_feu", "slack_feu",
            "utilization_pct", "binding_90pct", "binding_95pct", "binding_100pct",
        ],
    )
    component_fields = (
        ["validation_id", "allocation_id"]
        + list(component_rows[0].keys())[2:]
        if component_rows
        else ["validation_id", "allocation_id"]
    )
    write_csv(args.output / "exact_route_components.csv", component_rows, component_fields)
    write_csv(
        args.output / "exact_cost_component_summary.csv",
        cost_summary_rows,
        [
            "validation_id", "case_id", "condition_id", "gateway_access",
            "endpoint", "transport_cost_usd", "transfer_cost_usd",
            "waiting_holding_cost_usd", "processing_cost_usd",
            "tardiness_cost_usd", "transport_and_transfer_cost_usd",
            "waiting_and_processing_cost_usd", "reconstructed_total_cost_usd",
            "reported_total_cost_usd",
        ],
    )

    by_key = {
        (row["case_id"], row["condition_id"], row["gateway_access"], row["endpoint"]): row
        for row in endpoint_payloads
    }
    value_rows: list[dict[str, object]] = []
    for condition in conditions:
        for endpoint, _ in ENDPOINTS:
            allowed = by_key[(condition.case_id, condition.condition_id, "allowed", endpoint)]
            restricted = by_key[(condition.case_id, condition.condition_id, "restricted", endpoint)]
            cost_benefit = (
                float(restricted["total_cost_usd"]) - float(allowed["total_cost_usd"])
                if allowed["total_cost_usd"] is not None
                and restricted["total_cost_usd"] is not None
                else None
            )
            time_benefit = (
                float(restricted["quantity_weighted_mean_delivery_time_h"])
                - float(allowed["quantity_weighted_mean_delivery_time_h"])
                if allowed["quantity_weighted_mean_delivery_time_h"] is not None
                and restricted["quantity_weighted_mean_delivery_time_h"] is not None
                else None
            )
            value_rows.append(
                {
                    "case_id": condition.case_id,
                    "condition_id": condition.condition_id,
                    "family": condition.family,
                    "uswc_capacity_multiplier": condition.uswc_capacity_multiplier,
                    "uswc_delay_h": condition.uswc_delay_h,
                    "endpoint": endpoint,
                    "allowed_proven": allowed["lexicographic_proven"],
                    "restricted_proven": restricted["lexicographic_proven"],
                    "cost_avoided_by_allowing_ny_nj_usd": cost_benefit,
                    "time_avoided_by_allowing_ny_nj_h": time_benefit,
                    "allowed_ny_nj_share": allowed["ny_nj_share"],
                    "allowed_total_cost_usd": allowed["total_cost_usd"],
                    "restricted_total_cost_usd": restricted["total_cost_usd"],
                    "allowed_weighted_time_h": allowed[
                        "quantity_weighted_mean_delivery_time_h"
                    ],
                    "restricted_weighted_time_h": restricted[
                        "quantity_weighted_mean_delivery_time_h"
                    ],
                }
            )
    value_fields = list(value_rows[0].keys())
    write_csv(args.output / "exact_ny_nj_value.csv", value_rows, value_fields)

    delay_rows: list[dict[str, object]] = []
    for condition in conditions:
        if condition.family not in {"baseline", "regional_delay"}:
            continue
        cost_row = next(
            row
            for row in value_rows
            if row["case_id"] == condition.case_id
            and row["condition_id"] == condition.condition_id
            and row["endpoint"] == "min_cost_then_time"
        )
        time_row = next(
            row
            for row in value_rows
            if row["case_id"] == condition.case_id
            and row["condition_id"] == condition.condition_id
            and row["endpoint"] == "min_time_then_cost"
        )
        delay_rows.append(
            {
                "case_id": condition.case_id,
                "delay_h": condition.uswc_delay_h,
                "minimum_cost_benefit_usd": cost_row[
                    "cost_avoided_by_allowing_ny_nj_usd"
                ],
                "minimum_cost_endpoint_ny_nj_share": cost_row[
                    "allowed_ny_nj_share"
                ],
                "minimum_time_benefit_h": time_row[
                    "time_avoided_by_allowing_ny_nj_h"
                ],
                "minimum_time_endpoint_ny_nj_share": time_row[
                    "allowed_ny_nj_share"
                ],
                "different_endpoints_warning": (
                    "cost and time benefits refer to different lexicographic endpoints"
                ),
            }
        )
    write_csv(
        args.output / "delay_value_curves.csv",
        delay_rows,
        list(delay_rows[0].keys()),
    )
    capacity_rows_out = [
        row
        for row in value_rows
        if row["family"] in {"capacity_reduction", "compound_disruption"}
    ]
    write_csv(
        args.output / "capacity_compound_value_summary.csv",
        capacity_rows_out,
        value_fields,
    )

    share_payloads = [
        load_json(path)
        for path in sorted((checkpoints / "share_bounds").glob("*.json"))
    ]
    share_groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    share_solver_rows: list[dict[str, object]] = []
    share_allocation_rows: list[dict[str, object]] = []
    for payload in share_payloads:
        key = (
            str(payload["case_id"]),
            str(payload["condition_id"]),
            str(payload["endpoint_region"]),
        )
        share_groups.setdefault(key, {})[str(payload["direction"])] = payload
        share_solver_rows.append(
            {"bound_id": payload["bound_id"], **payload["solver_call"]}
        )
        for allocation_number, row in enumerate(payload["allocations"], start=1):
            share_allocation_rows.append(
                {
                    "bound_id": payload["bound_id"],
                    "allocation_id": allocation_number,
                    "shipment_id": row["shipment_id"],
                    "quantity_feu": row["quantity_feu"],
                    "gateway": row["gateway"],
                    "arc_ids": "|".join(row["arc_ids"]),
                }
            )
    share_range_rows: list[dict[str, object]] = []
    for (case_id, condition_id, region), bounds in sorted(share_groups.items()):
        minimum = bounds.get("min")
        maximum = bounds.get("max")
        min_share = minimum.get("ny_nj_share") if minimum else None
        max_share = maximum.get("ny_nj_share") if maximum else None
        share_range_rows.append(
            {
                "case_id": case_id,
                "condition_id": condition_id,
                "endpoint_region": region,
                "ny_nj_min_share": min_share,
                "ny_nj_max_share": max_share,
                "min_proven": minimum.get("proven_optimal") if minimum else 0,
                "max_proven": maximum.get("proven_optimal") if maximum else 0,
                "crosses_10pct": int(
                    min_share is not None
                    and max_share is not None
                    and float(min_share) < 0.10 <= float(max_share)
                ),
                "all_endpoint_equivalent_solutions_at_least_10pct": int(
                    min_share is not None and float(min_share) >= 0.10 - 1e-9
                ),
                "share_unique_within_tolerance": int(
                    min_share is not None
                    and max_share is not None
                    and math.isclose(
                        float(min_share), float(max_share), rel_tol=0.0, abs_tol=1e-8
                    )
                ),
                "interpretation_limit": (
                    "equal bounds identify only aggregate gateway-share uniqueness, "
                    "not full allocation uniqueness"
                ),
            }
        )
    write_csv(
        args.output / "exact_endpoint_ny_nj_share_ranges.csv",
        share_range_rows,
        [
            "case_id", "condition_id", "endpoint_region", "ny_nj_min_share",
            "ny_nj_max_share", "min_proven", "max_proven", "crosses_10pct",
            "all_endpoint_equivalent_solutions_at_least_10pct",
            "share_unique_within_tolerance", "interpretation_limit",
        ],
    )
    write_csv(
        args.output / "share_bound_solver_calls.csv",
        share_solver_rows,
        [
            "bound_id", "stage", "objective", "status_code", "status", "success",
            "objective_value", "dual_bound", "relative_gap", "node_count",
            "elapsed_seconds", "message",
        ],
    )
    write_csv(
        args.output / "share_bound_allocations.csv",
        share_allocation_rows,
        [
            "bound_id", "allocation_id", "shipment_id", "quantity_feu",
            "gateway", "arc_ids",
        ],
    )

    feasibility_payloads = [
        load_json(path)
        for path in sorted((checkpoints / "feasibility").glob("*.json"))
    ]
    feasibility_rows = [
        {
            "case_id": row["case_id"],
            "condition_id": row["condition_id"],
            "classification": row["classification"],
            "feasible_solution_found": row["feasible_solution_found"],
            "proven_infeasible": row["proven_infeasible"],
            "status_code": row["solver_call"]["status_code"],
            "status": row["solver_call"]["status"],
            "relative_gap": row["solver_call"]["relative_gap"],
            "elapsed_seconds": row["solver_call"]["elapsed_seconds"],
            "message": row["solver_call"]["message"],
        }
        for row in feasibility_payloads
    ]
    write_csv(
        args.output / "demand150_capacity60_feasibility.csv",
        feasibility_rows,
        list(feasibility_rows[0].keys()),
    )

    ny_allocations = [
        row for row in allocation_rows if row["gateway"] == "ny_nj"
    ]
    shipment_metadata = {
        shipment_id: shipment
        for case_id in CASES
        for shipment_id, shipment in case_cache[case_id].shipments.items()
    }
    ny_shipment_rows = [
        {
            **row,
            "shipment_total_quantity_feu": shipment_metadata[
                str(row["shipment_id"])
            ].quantity_feu,
            "release_h": shipment_metadata[str(row["shipment_id"])].release_h,
            "due_h": shipment_metadata[str(row["shipment_id"])].due_h,
            "tardiness_usd_per_feu_h": shipment_metadata[
                str(row["shipment_id"])
            ].tardiness_usd_per_feu_h,
        }
        for row in ny_allocations
    ]
    write_csv(
        args.output / "ny_nj_shipment_diagnostic.csv",
        ny_shipment_rows,
        [
            "validation_id", "allocation_id", "shipment_id", "quantity_feu",
            "gateway", "arc_ids", "shipment_total_quantity_feu", "release_h",
            "due_h", "tardiness_usd_per_feu_h",
        ],
    )
    ny_validation_ids = {str(row["validation_id"]) for row in ny_allocations}
    ny_resource_rows = [
        row
        for row in capacity_rows
        if row["validation_id"] in ny_validation_ids
        and (
            row["origin"] == "ny_nj"
            or row["destination"] == "ny_nj"
            or row["resource_id"] == "ny_nj"
        )
    ]
    write_csv(
        args.output / "ny_nj_resource_utilization.csv",
        ny_resource_rows,
        [
            "validation_id", "resource_type", "resource_id", "period", "origin",
            "destination", "used_feu", "capacity_feu", "slack_feu",
            "utilization_pct", "binding_90pct", "binding_95pct", "binding_100pct",
        ],
    )
    binding_rows = [row for row in capacity_rows if int(row["binding_95pct"]) == 1]
    write_csv(
        args.output / "binding_resource_diagnostic.csv",
        binding_rows,
        [
            "validation_id", "resource_type", "resource_id", "period", "origin",
            "destination", "used_feu", "capacity_feu", "slack_feu",
            "utilization_pct", "binding_90pct", "binding_95pct", "binding_100pct",
        ],
    )
    batch_sets: list[dict[str, object]] = []
    for payload in endpoint_payloads:
        if payload["gateway_access"] != "allowed":
            continue
        rows = [
            row
            for row in ny_shipment_rows
            if row["validation_id"] == payload["validation_id"]
        ]
        batch_sets.append(
            {
                "validation_id": payload["validation_id"],
                "case_id": payload["case_id"],
                "condition_id": payload["condition_id"],
                "endpoint": payload["endpoint"],
                "ny_nj_quantity_feu": sum(float(row["quantity_feu"]) for row in rows),
                "ny_nj_share": payload["ny_nj_share"],
                "ny_nj_shipment_count": len({row["shipment_id"] for row in rows}),
                "ny_nj_shipment_ids": "|".join(
                    sorted({str(row["shipment_id"]) for row in rows})
                ),
            }
        )
    write_csv(
        args.output / "ny_nj_batch_set_diagnostic.csv",
        batch_sets,
        list(batch_sets[0].keys()),
    )
    repetition_groups: dict[float, list[dict[str, object]]] = {}
    for row in batch_sets:
        repetition_groups.setdefault(round(float(row["ny_nj_quantity_feu"]), 6), []).append(row)
    repetitions = [
        {
            "ny_nj_quantity_feu": quantity,
            "occurrence_count": len(rows),
            "validation_ids": "|".join(str(row["validation_id"]) for row in rows),
            "distinct_batch_sets": len({str(row["ny_nj_shipment_ids"]) for row in rows}),
        }
        for quantity, rows in sorted(repetition_groups.items())
        if quantity > 0 and len(rows) > 1
    ]
    write_csv(
        args.output / "ny_nj_repeated_quantity_diagnostic.csv",
        repetitions,
        [
            "ny_nj_quantity_feu", "occurrence_count", "validation_ids",
            "distinct_batch_sets",
        ],
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": "pre-registered final non-carbon exact endpoint and share-bound suite",
        "conditions": [asdict(condition) for condition in conditions],
        "endpoint_tasks": len(conditions) * 2 * len(ENDPOINTS),
        "reused_endpoint_tasks": sum(bool(row["reused_from"]) for row in endpoint_payloads),
        "share_bound_tasks": len(share_payloads),
        "feasibility_tasks": len(feasibility_rows),
        "time_limit_seconds_per_solver_call": args.time_limit_seconds,
        "mip_relative_gap": args.mip_relative_gap,
        "endpoint_absolute_tolerance": args.absolute_tolerance,
        "endpoint_relative_tolerance": args.relative_tolerance,
        "activation_threshold": 0.10,
        "interpretation": {
            "value_curves": (
                "minimum-cost and minimum-time benefit curves compare different "
                "lexicographic endpoints and do not claim simultaneous benefits"
            ),
            "share_bounds": (
                "equal NY/NJ share bounds establish only aggregate share uniqueness "
                "within an endpoint-equivalent objective region"
            ),
            "endpoint_scope": (
                "an endpoint without NY/NJ does not establish that the full Pareto "
                "front excludes NY/NJ"
            ),
        },
    }
    write_json_atomic(args.output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final exact endpoint, share-bound, and feasibility suite."
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/v14_core_exact_suite")
    )
    parser.add_argument(
        "--reuse-exact-root",
        type=Path,
        default=Path("results/v12_exact_endpoint_validation"),
    )
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.time_limit_seconds <= 0:
        parser.error("--time-limit-seconds must be positive")
    if args.mip_relative_gap < 0:
        parser.error("--mip-relative-gap must be non-negative")
    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    conditions = core_conditions()
    checkpoints = args.output / "checkpoints"
    run_endpoints(args, data, base_config, conditions, checkpoints)
    run_share_bounds(args, data, base_config, conditions, checkpoints)
    run_feasibility_checks(args, data, base_config, checkpoints)
    finalize_outputs(args, data, base_config, conditions, checkpoints)
    print(f"Final exact core suite written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
