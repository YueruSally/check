#!/usr/bin/env python3
"""Exhaustive structural-path coverage audit and exact endpoint revalidation.

This is the dependency-gating experiment for the remaining paper analyses.  It
enumerates simple paths under the model's structural constraints without
static cost--time dominance pruning.  Exact endpoints are then compared for
the current Martins-8 library and the complete structural library.  If the
enumeration safety cap is reached, the script writes the audit and stops before
claiming or solving a complete-library experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.exact_endpoint import (
    build_exact_endpoint_model,
    solve_bounded_milp,
    solve_lexicographic_endpoint,
)
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.martins import dominates
from china_us_multimodal.path_library import (
    enumerate_gateway_structural_paths,
    gateway_martins_paths,
)
from china_us_multimodal.supplementary_design import ExactCondition, condition_config
from run_supplementary_core_exact import (
    ENDPOINTS,
    allocations_from_payload,
    endpoint_payload,
    endpoint_region_bounds,
    endpoints_coincident,
    gateway_for_path,
    load_json,
    read_csv,
    share_bound_payload,
    write_csv,
    write_json_atomic,
)


GATEWAYS = ("la_lb", "seattle_tacoma", "ny_nj")
MARTINS_LIMITS = (4, 8, 12, 16)


def coverage_conditions() -> tuple[ExactCondition, ...]:
    requested = (
        ("D1", "delay_0h", "baseline", 1.0, 0.0),
        ("D2", "delay_0h", "baseline", 1.0, 0.0),
        ("D3", "delay_0h", "baseline", 1.0, 0.0),
        ("D1", "capacity_35pct", "capacity_reduction", 0.35, 0.0),
        ("D2", "capacity_35pct", "capacity_reduction", 0.35, 0.0),
        ("D3", "capacity_35pct", "capacity_reduction", 0.35, 0.0),
        ("D1", "compound_30pct_delay_168h", "compound_disruption", 0.30, 168.0),
        ("D2", "compound_30pct_delay_168h", "compound_disruption", 0.30, 168.0),
        ("D3", "compound_30pct_delay_168h", "compound_disruption", 0.30, 168.0),
        ("D1", "delay_264h", "regional_delay", 1.0, 264.0),
        ("D1", "delay_288h", "regional_delay", 1.0, 288.0),
        ("D2", "delay_264h", "regional_delay", 1.0, 264.0),
        ("D2", "delay_288h", "regional_delay", 1.0, 288.0),
        ("D3", "delay_168h", "regional_delay", 1.0, 168.0),
        ("D3", "delay_240h", "regional_delay", 1.0, 240.0),
        ("D3", "delay_264h", "regional_delay", 1.0, 264.0),
    )
    return tuple(
        ExactCondition(
            case_id=case_id,
            condition_id=condition_id,
            family=family,
            uswc_capacity_multiplier=capacity,
            uswc_delay_h=delay,
        )
        for case_id, condition_id, family, capacity, delay in requested
    )


def arc_resource_signature(data, arc_ids: tuple[str, ...]) -> str:
    parts = []
    for arc_id in arc_ids:
        arc = data.arcs[arc_id]
        parts.append(
            ":".join(
                (
                    arc_id,
                    arc.mode.value,
                    (
                        "none"
                        if arc.capacity_feu_per_day is None
                        else f"{arc.capacity_feu_per_day:.12g}"
                    ),
                    "+".join(arc.service_ids) if arc.service_ids else "unscheduled",
                )
            )
        )
    return "|".join(parts)


def enumerate_case_paths(case_data, config, enumeration_cap: int):
    shipments = list(case_data.shipments.values())
    od_pairs = sorted({(item.origin, item.destination) for item in shipments})
    enumerations = {}
    for origin, destination in od_pairs:
        for gateway in GATEWAYS:
            enumerations[(origin, destination, gateway)] = (
                enumerate_gateway_structural_paths(
                    case_data,
                    origin,
                    destination,
                    gateway,
                    max_legs=config.constraints.max_legs_per_path,
                    max_mode_changes=config.constraints.max_mode_changes,
                    enumeration_cap=enumeration_cap,
                )
            )
    return enumerations


def full_required_library(case_data, enumerations, disabled_gateways: set[str]):
    library: dict[str, tuple[tuple[str, ...], ...]] = {}
    max_paths = 0
    for shipment_id, shipment in sorted(case_data.shipments.items()):
        paths: list[tuple[str, ...]] = []
        for gateway in GATEWAYS:
            if gateway in disabled_gateways:
                continue
            labels = enumerations[(shipment.origin, shipment.destination, gateway)].labels
            max_paths = max(max_paths, len(labels))
            paths.extend(label.arc_ids for label in labels)
        library[shipment_id] = tuple(paths)
    if not library or max_paths <= 0:
        raise RuntimeError("Complete structural path library is empty.")
    return library, max_paths


def model_signatures(model) -> set[tuple[str, tuple[str, ...]]]:
    return {(profile.shipment_id, profile.arc_ids) for profile in model.profiles}


def endpoint_fields() -> list[str]:
    return [
        "validation_id", "library_variant", "case_id", "condition_id", "family",
        "gateway_access", "scenario", "uswc_capacity_multiplier", "uswc_delay_h",
        "endpoint", "primary_objective", "secondary_objective",
        "max_paths_per_gateway", "candidate_paths", "binary_variables",
        "stage_one_proven", "lexicographic_proven", "evaluator_consistent",
        "primary_optimum", "primary_tolerance", "total_cost_usd",
        "quantity_weighted_mean_delivery_time_h", "feasible", "violations",
        "evaluator_cost_abs_error", "evaluator_time_abs_error",
        "resource_max_abs_error_feu", "ny_nj_share", "la_lb_share",
        "seattle_tacoma_share", "solver_call_count", "solver_runtime_seconds",
        "maximum_relative_gap", "reused_from",
    ]


def build_full_model(case_data, config, enumerations):
    required, max_paths = full_required_library(
        case_data,
        enumerations,
        set(config.scenario.disabled_gateways),
    )
    model = build_exact_endpoint_model(
        case_data,
        config,
        max_paths_per_gateway=max_paths,
        required_paths_by_shipment=required,
    )
    required_signatures = {
        (shipment_id, path)
        for shipment_id, paths in required.items()
        for path in paths
    }
    if model_signatures(model) != required_signatures:
        missing = required_signatures - model_signatures(model)
        extra = model_signatures(model) - required_signatures
        raise RuntimeError(
            "Full structural library mismatch: "
            f"missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
        )
    return model, max_paths


def write_coverage_audit(args, data, base_config, case_enumerations):
    summary_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for case_id in ("D1", "D2", "D3"):
        case_data = select_case(data, case_id)
        enumerations = case_enumerations[case_id]
        od_pairs = sorted({
            (item.origin, item.destination) for item in case_data.shipments.values()
        })
        for origin, destination in od_pairs:
            for gateway in GATEWAYS:
                enumeration = enumerations[(origin, destination, gateway)]
                labels = enumeration.labels
                nondominated = {
                    label.arc_ids
                    for label in labels
                    if not any(
                        dominates(other, label)
                        for other in labels
                        if other != label
                    )
                }
                martins = {
                    limit: gateway_martins_paths(
                        case_data,
                        origin,
                        destination,
                        gateway,
                        max_legs=base_config.constraints.max_legs_per_path,
                        max_mode_changes=base_config.constraints.max_mode_changes,
                        max_paths=limit,
                    )
                    for limit in MARTINS_LIMITS
                }
                signatures = {
                    limit: {label.arc_ids for label in found}
                    for limit, found in martins.items()
                }
                missing_nondominated = nondominated - signatures[16]
                dominated_count = len(labels) - len(nondominated)
                summary_rows.append(
                    {
                        "case_id": case_id,
                        "origin": origin,
                        "destination": destination,
                        "gateway": gateway,
                        "structural_enumeration_complete": int(enumeration.complete),
                        "structural_path_count": len(labels),
                        "structural_nondominated_count": len(nondominated),
                        "statically_dominated_count": dominated_count,
                        "martins_4_count": len(signatures[4]),
                        "martins_8_count": len(signatures[8]),
                        "martins_12_count": len(signatures[12]),
                        "martins_16_count": len(signatures[16]),
                        "martins_4_subset_8": int(signatures[4] <= signatures[8]),
                        "martins_8_subset_12": int(signatures[8] <= signatures[12]),
                        "martins_12_subset_16": int(signatures[12] <= signatures[16]),
                        "martins_16_subset_structural_full": int(
                            signatures[16] <= {label.arc_ids for label in labels}
                        ),
                        "martins_8_cap_reached": int(len(signatures[8]) == 8),
                        "nondominated_missing_at_8": len(nondominated - signatures[8]),
                        "nondominated_missing_at_16": len(missing_nondominated),
                        "full_paths_missing_at_8": len(
                            {label.arc_ids for label in labels} - signatures[8]
                        ),
                        "explored_labels": enumeration.explored_labels,
                        "enumeration_cap": enumeration.enumeration_cap,
                        "coverage_reason": (
                            "enumeration_cap_reached"
                            if not enumeration.complete
                            else "martins_missing_nondominated_paths"
                            if missing_nondominated
                            else "additional_structural_paths_removed_by_static_dominance"
                            if dominated_count
                            else "martins_library_is_structurally_complete"
                        ),
                    }
                )
                for path_number, label in enumerate(labels, start=1):
                    dominators = [
                        other for other in labels
                        if other != label and dominates(other, label)
                    ]
                    signature = arc_resource_signature(case_data, label.arc_ids)
                    inventory_rows.append(
                        {
                            "case_id": case_id,
                            "origin": origin,
                            "destination": destination,
                            "gateway": gateway,
                            "structural_path_number": path_number,
                            "arc_ids": "|".join(label.arc_ids),
                            "mode_sequence": "|".join(
                                case_data.arcs[arc_id].mode.value
                                for arc_id in label.arc_ids
                            ),
                            "service_ids": "|".join(
                                "+".join(case_data.arcs[arc_id].service_ids)
                                if case_data.arcs[arc_id].service_ids else "unscheduled"
                                for arc_id in label.arc_ids
                            ),
                            "resource_signature": signature,
                            "static_cost_usd_per_feu": label.cost_usd_per_feu,
                            "static_elapsed_h": label.elapsed_h,
                            "statically_nondominated": int(label.arc_ids in nondominated),
                            "static_dominator_arc_ids": "||".join(
                                "|".join(item.arc_ids) for item in dominators
                            ),
                            "dominated_but_resource_distinct": int(
                                bool(dominators)
                                and all(
                                    arc_resource_signature(case_data, item.arc_ids)
                                    != signature
                                    for item in dominators
                                )
                            ),
                            "present_martins_4": int(label.arc_ids in signatures[4]),
                            "present_martins_8": int(label.arc_ids in signatures[8]),
                            "present_martins_12": int(label.arc_ids in signatures[12]),
                            "present_martins_16": int(label.arc_ids in signatures[16]),
                        }
                    )
    write_csv(
        args.output / "path_coverage_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )
    write_csv(
        args.output / "structural_path_inventory.csv",
        inventory_rows,
        list(inventory_rows[0].keys()),
    )
    return summary_rows


def run_full_endpoints(args, data, base_config, conditions, case_enumerations):
    checkpoints = args.output / "checkpoints" / "endpoints"
    total = len(conditions) * 2 * len(ENDPOINTS)
    completed = 0
    for condition in conditions:
        case_data = select_case(data, condition.case_id)
        enumerations = case_enumerations[condition.case_id]
        for access in ("allowed", "restricted"):
            config = condition_config(base_config, condition, gateway_access=access)
            model = None
            max_paths = None
            for endpoint, primary in ENDPOINTS:
                task_id = (
                    f"{condition.case_id}_{condition.condition_id}_{access}_"
                    f"{endpoint}_structural_full"
                )
                checkpoint = checkpoints / f"{task_id}.json"
                if checkpoint.exists():
                    existing = load_json(checkpoint)
                    if int(existing.get("lexicographic_proven", 0)) == 1:
                        completed += 1
                        print(f"[{completed}/{total}] resume {task_id}", flush=True)
                        continue
                if model is None:
                    model, max_paths = build_full_model(
                        case_data, config, enumerations
                    )
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
                    max_paths_per_gateway=int(max_paths),
                )
                payload["validation_id"] = task_id
                payload["library_variant"] = "structural_full"
                write_json_atomic(checkpoint, payload)
                completed += 1
                print(
                    f"[{completed}/{total}] {task_id} "
                    f"proven={payload['lexicographic_proven']} "
                    f"paths={payload['candidate_paths']}",
                    flush=True,
                )


def run_full_share_bounds(args, data, base_config, conditions, case_enumerations):
    endpoint_root = args.output / "checkpoints" / "endpoints"
    bound_root = args.output / "checkpoints" / "share_bounds"
    planned = []
    for condition in conditions:
        endpoints = {
            endpoint: load_json(
                endpoint_root / (
                    f"{condition.case_id}_{condition.condition_id}_allowed_"
                    f"{endpoint}_structural_full.json"
                )
            )
            for endpoint, _ in ENDPOINTS
        }
        if not all(int(row["lexicographic_proven"]) == 1 for row in endpoints.values()):
            continue
        if endpoints_coincident(
            endpoints["min_cost_then_time"],
            endpoints["min_time_then_cost"],
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        ):
            planned.append((condition, "coincident_endpoints", endpoints["min_cost_then_time"]))
        else:
            planned.extend(
                (condition, endpoint, payload)
                for endpoint, payload in endpoints.items()
            )
    total = len(planned) * 2
    completed = 0
    for condition, region, endpoint in planned:
        case_data = select_case(data, condition.case_id)
        config = condition_config(base_config, condition, gateway_access="allowed")
        model = None
        cost_bound, time_bound = endpoint_region_bounds(
            endpoint,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        for direction in ("min", "max"):
            bound_id = (
                f"{condition.case_id}_{condition.condition_id}_{region}_"
                f"{direction}_ny_nj_share_structural_full"
            )
            checkpoint = bound_root / f"{bound_id}.json"
            if checkpoint.exists():
                existing = load_json(checkpoint)
                if share_bound_complete(existing):
                    completed += 1
                    print(f"[share {completed}/{total}] resume {bound_id}", flush=True)
                    continue
                print(
                    f"[share retry] {bound_id} has an incomplete checkpoint "
                    f"(proven={existing.get('proven_optimal')}, "
                    f"consistent={existing.get('evaluator_consistent')}, "
                    f"share={existing.get('ny_nj_share')}); recomputing",
                    flush=True,
                )
            if direction == "min" and float(endpoint.get("ny_nj_share", 1.0)) <= 1e-12:
                payload = zero_share_lower_bound_certificate(
                    bound_id=bound_id,
                    condition=condition,
                    endpoint_region=region,
                    cost_upper_bound=cost_bound,
                    time_upper_bound=time_bound,
                    endpoint=endpoint,
                )
                write_json_atomic(checkpoint, payload)
                completed += 1
                print(
                    f"[share {completed}/{total}] {bound_id} "
                    "proven=1 certificate=nonnegative_share_lower_bound",
                    flush=True,
                )
                continue
            if model is None:
                model, _ = build_full_model(
                    case_data,
                    config,
                    case_enumerations[condition.case_id],
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
            payload = share_bound_payload(
                bound_id=bound_id,
                condition=condition,
                endpoint_region=region,
                direction=direction,
                cost_upper_bound=cost_bound,
                time_upper_bound=time_bound,
                result=result,
                data=case_data,
            )
            payload["library_variant"] = "structural_full"
            write_json_atomic(checkpoint, payload)
            completed += 1
            print(
                f"[share {completed}/{total}] {bound_id} "
                f"proven={payload['proven_optimal']} "
                f"consistent={payload['evaluator_consistent']} "
                f"share={payload['ny_nj_share']}",
                flush=True,
            )


def zero_share_lower_bound_certificate(
    *,
    bound_id: str,
    condition: ExactCondition,
    endpoint_region: str,
    cost_upper_bound: float,
    time_upper_bound: float,
    endpoint: dict[str, object],
) -> dict[str, object]:
    """Certify a zero minimum share from a proven zero-share endpoint.

    Gateway share is a sum of nonnegative allocation variables divided by
    positive total demand, so zero is a global lower bound.  A proven,
    evaluator-consistent endpoint with zero NY/NJ allocation attains that
    bound inside its own endpoint-equivalent region; no additional MILP call
    is required.
    """
    if int(endpoint.get("lexicographic_proven", 0)) != 1:
        raise ValueError("A zero-share certificate requires a proven endpoint.")
    if int(endpoint.get("evaluator_consistent", 0)) != 1:
        raise ValueError(
            "A zero-share certificate requires an evaluator-consistent endpoint."
        )
    if abs(float(endpoint.get("ny_nj_share", 1.0))) > 1e-12:
        raise ValueError("The endpoint does not attain the zero-share lower bound.")
    return {
        "bound_id": bound_id,
        "case_id": condition.case_id,
        "condition_id": condition.condition_id,
        "family": condition.family,
        "endpoint_region": endpoint_region,
        "direction": "min",
        "cost_upper_bound_usd": cost_upper_bound,
        "time_upper_bound_h": time_upper_bound,
        "proven_optimal": 1,
        "proven_infeasible": 0,
        "evaluator_consistent": 1,
        "ny_nj_share": 0.0,
        "total_cost_usd": endpoint["total_cost_usd"],
        "quantity_weighted_mean_delivery_time_h": endpoint[
            "quantity_weighted_mean_delivery_time_h"
        ],
        "solver_call": {
            "stage": 0,
            "objective": "min_ny_nj_share",
            "status_code": 0,
            "status": "analytically_certified",
            "success": True,
            "objective_value": 0.0,
            "dual_bound": 0.0,
            "relative_gap": 0.0,
            "node_count": 0,
            "elapsed_seconds": 0.0,
            "message": (
                "A proven endpoint attains zero NY/NJ share; nonnegativity "
                "therefore certifies the global minimum."
            ),
        },
        "allocations": endpoint["allocations"],
        "certificate_type": "nonnegative_share_lower_bound",
        "certificate_source_validation_id": endpoint["validation_id"],
        "library_variant": "structural_full",
    }


def share_bound_complete(payload: dict[str, object]) -> bool:
    """Return whether a share-bound checkpoint is safe to reuse.

    A solver status alone is insufficient for downstream gateway-share
    analysis.  The decoded allocation must also pass the common evaluator and
    supply the objective values used to define the endpoint-equivalent region.
    """
    return (
        int(payload.get("proven_optimal", 0)) == 1
        and int(payload.get("evaluator_consistent", 0)) == 1
        and payload.get("ny_nj_share") is not None
        and payload.get("total_cost_usd") is not None
        and payload.get("quantity_weighted_mean_delivery_time_h") is not None
    )


def finalize(args, data, conditions):
    endpoint_payloads = [
        load_json(path)
        for path in sorted((args.output / "checkpoints" / "endpoints").glob("*.json"))
    ]
    endpoint_payloads.sort(key=lambda row: str(row["validation_id"]))
    write_csv(
        args.output / "full_exact_endpoint_summary.csv",
        [
            {key: value for key, value in row.items() if key not in {"solver_calls", "allocations"}}
            for row in endpoint_payloads
        ],
        endpoint_fields(),
    )
    solver_rows = []
    allocation_rows = []
    for payload in endpoint_payloads:
        for call in payload["solver_calls"]:
            solver_rows.append({"validation_id": payload["validation_id"], **call})
        case_data = select_case(data, str(payload["case_id"]))
        for index, allocation in enumerate(allocations_from_payload(payload), start=1):
            allocation_rows.append(
                {
                    "validation_id": payload["validation_id"],
                    "allocation_id": index,
                    "shipment_id": allocation.shipment_id,
                    "quantity_feu": allocation.quantity_feu,
                    "gateway": gateway_for_path(case_data, allocation.arc_ids),
                    "arc_ids": "|".join(allocation.arc_ids),
                }
            )
    write_csv(
        args.output / "full_exact_solver_calls.csv",
        solver_rows,
        [
            "validation_id", "stage", "objective", "status_code", "status",
            "success", "objective_value", "dual_bound", "relative_gap",
            "node_count", "elapsed_seconds", "message",
        ],
    )
    write_csv(
        args.output / "full_exact_endpoint_allocations.csv",
        allocation_rows,
        [
            "validation_id", "allocation_id", "shipment_id", "quantity_feu",
            "gateway", "arc_ids",
        ],
    )

    share_payloads = [
        load_json(path)
        for path in sorted((args.output / "checkpoints" / "share_bounds").glob("*.json"))
    ]
    share_solver_rows = []
    share_allocation_rows = []
    for payload in share_payloads:
        share_solver_rows.append(
            {"bound_id": payload["bound_id"], **payload["solver_call"]}
        )
        for index, allocation in enumerate(payload["allocations"], start=1):
            share_allocation_rows.append(
                {
                    "bound_id": payload["bound_id"],
                    "allocation_id": index,
                    "shipment_id": allocation["shipment_id"],
                    "quantity_feu": allocation["quantity_feu"],
                    "gateway": allocation["gateway"],
                    "arc_ids": "|".join(allocation["arc_ids"]),
                }
            )
    write_csv(
        args.output / "full_share_bound_solver_calls.csv",
        share_solver_rows,
        [
            "bound_id", "stage", "objective", "status_code", "status",
            "success", "objective_value", "dual_bound", "relative_gap",
            "node_count", "elapsed_seconds", "message",
        ],
    )
    write_csv(
        args.output / "full_share_bound_allocations.csv",
        share_allocation_rows,
        [
            "bound_id", "allocation_id", "shipment_id", "quantity_feu",
            "gateway", "arc_ids",
        ],
    )
    share_groups = {}
    for payload in share_payloads:
        key = (payload["case_id"], payload["condition_id"], payload["endpoint_region"])
        share_groups.setdefault(key, {})[payload["direction"]] = payload
    unresolved_rows = []
    for (case_id, condition_id, region), directions in sorted(share_groups.items()):
        for direction in ("min", "max"):
            payload = directions.get(direction)
            if payload is not None and share_bound_complete(payload):
                continue
            unresolved_rows.append(
                {
                    "case_id": case_id,
                    "condition_id": condition_id,
                    "endpoint_region": region,
                    "direction": direction,
                    "bound_id": payload.get("bound_id", "") if payload else "",
                    "status": (
                        payload.get("solver_call", {}).get("status", "")
                        if payload
                        else "missing_checkpoint"
                    ),
                    "proven_optimal": (
                        payload.get("proven_optimal", "") if payload else ""
                    ),
                    "proven_infeasible": (
                        payload.get("proven_infeasible", "") if payload else ""
                    ),
                    "evaluator_consistent": (
                        payload.get("evaluator_consistent", "") if payload else ""
                    ),
                    "ny_nj_share": payload.get("ny_nj_share", "") if payload else "",
                    "solver_message": (
                        payload.get("solver_call", {}).get("message", "")
                        if payload
                        else ""
                    ),
                }
            )
    write_csv(
        args.output / "unresolved_share_bounds.csv",
        unresolved_rows,
        [
            "case_id", "condition_id", "endpoint_region", "direction",
            "bound_id", "status", "proven_optimal", "proven_infeasible",
            "evaluator_consistent", "ny_nj_share", "solver_message",
        ],
    )
    if unresolved_rows:
        identifiers = ", ".join(
            str(row["bound_id"] or (
                f"{row['case_id']}_{row['condition_id']}_"
                f"{row['endpoint_region']}_{row['direction']}"
            ))
            for row in unresolved_rows
        )
        raise RuntimeError(
            f"{len(unresolved_rows)} share-bound task(s) lack a proven, "
            "evaluator-consistent NY/NJ share: " + identifiers + ". "
            f"See {args.output / 'unresolved_share_bounds.csv'}."
        )
    share_rows = []
    for (case_id, condition_id, region), directions in sorted(share_groups.items()):
        minimum = directions["min"]
        maximum = directions["max"]
        low = float(minimum["ny_nj_share"])
        high = float(maximum["ny_nj_share"])
        share_rows.append(
            {
                "case_id": case_id,
                "condition_id": condition_id,
                "endpoint_region": region,
                "ny_nj_min_share": low,
                "ny_nj_max_share": high,
                "min_proven": minimum["proven_optimal"],
                "max_proven": maximum["proven_optimal"],
                "crosses_10pct": int(low < 0.10 - 1e-9 and high >= 0.10 - 1e-9),
                "all_endpoint_equivalent_solutions_at_least_10pct": int(low >= 0.10 - 1e-9),
            }
        )
    write_csv(
        args.output / "full_endpoint_share_ranges.csv",
        share_rows,
        list(share_rows[0].keys()),
    )

    condition_keys = {(item.case_id, item.condition_id) for item in conditions}
    path8_share_rows = [
        row
        for row in read_csv(
            args.paths8_results / "exact_endpoint_ny_nj_share_ranges.csv"
        )
        if (row["case_id"], row["condition_id"]) in condition_keys
    ]
    path8_share_index = {
        (row["case_id"], row["condition_id"], row["endpoint_region"]): row
        for row in path8_share_rows
    }
    full_share_index = {
        (row["case_id"], row["condition_id"], row["endpoint_region"]): row
        for row in share_rows
    }
    share_comparison_rows = []
    for key in sorted(set(path8_share_index) | set(full_share_index)):
        eight = path8_share_index.get(key, {})
        full = full_share_index.get(key, {})
        share_comparison_rows.append(
            {
                "case_id": key[0],
                "condition_id": key[1],
                "endpoint_region": key[2],
                "region_present_paths8": int(bool(eight)),
                "region_present_full": int(bool(full)),
                "paths8_min_share": eight.get("ny_nj_min_share", ""),
                "paths8_max_share": eight.get("ny_nj_max_share", ""),
                "full_min_share": full.get("ny_nj_min_share", ""),
                "full_max_share": full.get("ny_nj_max_share", ""),
                "paths8_min_proven": eight.get("min_proven", ""),
                "paths8_max_proven": eight.get("max_proven", ""),
                "full_min_proven": full.get("min_proven", ""),
                "full_max_proven": full.get("max_proven", ""),
                "endpoint_region_structure_unchanged": int(
                    bool(eight) and bool(full)
                ),
            }
        )
    write_csv(
        args.output / "share_range_difference_8_vs_full.csv",
        share_comparison_rows,
        list(share_comparison_rows[0].keys()),
    )

    path8_rows = read_csv(args.paths8_results / "exact_endpoint_summary.csv")
    path8_index = {
        (row["case_id"], row["condition_id"], row["gateway_access"], row["endpoint"]): row
        for row in path8_rows
    }
    full_index = {
        (row["case_id"], row["condition_id"], row["gateway_access"], row["endpoint"]): row
        for row in endpoint_payloads
    }
    comparison_rows = []
    for condition in conditions:
        for access in ("allowed", "restricted"):
            for endpoint, _ in ENDPOINTS:
                key = (condition.case_id, condition.condition_id, access, endpoint)
                eight = path8_index[key]
                full = full_index[key]
                eight_cost = float(eight["total_cost_usd"])
                full_cost = float(full["total_cost_usd"])
                eight_time = float(eight["quantity_weighted_mean_delivery_time_h"])
                full_time = float(full["quantity_weighted_mean_delivery_time_h"])
                eight_share = float(eight["ny_nj_share"])
                full_share = float(full["ny_nj_share"])
                comparison_rows.append(
                    {
                        "case_id": condition.case_id,
                        "condition_id": condition.condition_id,
                        "gateway_access": access,
                        "endpoint": endpoint,
                        "paths8_proven": eight["lexicographic_proven"],
                        "full_proven": full["lexicographic_proven"],
                        "paths8_candidate_paths": eight["candidate_paths"],
                        "full_candidate_paths": full["candidate_paths"],
                        "paths8_cost_usd": eight_cost,
                        "full_cost_usd": full_cost,
                        "cost_change_full_minus_8_usd": full_cost - eight_cost,
                        "paths8_time_h": eight_time,
                        "full_time_h": full_time,
                        "time_change_full_minus_8_h": full_time - eight_time,
                        "paths8_ny_nj_share": eight_share,
                        "full_ny_nj_share": full_share,
                        "ny_nj_share_change": full_share - eight_share,
                        "material_use_conclusion_unchanged": int(
                            (eight_share >= 0.10 - 1e-9)
                            == (full_share >= 0.10 - 1e-9)
                        ),
                    }
                )
    write_csv(
        args.output / "endpoint_difference_8_vs_full.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )

    value_rows = []
    for condition in conditions:
        for endpoint, _ in ENDPOINTS:
            for variant, index in (("paths8", path8_index), ("structural_full", full_index)):
                allowed = index[(condition.case_id, condition.condition_id, "allowed", endpoint)]
                restricted = index[(condition.case_id, condition.condition_id, "restricted", endpoint)]
                value_rows.append(
                    {
                        "case_id": condition.case_id,
                        "condition_id": condition.condition_id,
                        "endpoint": endpoint,
                        "library_variant": variant,
                        "cost_avoided_by_allowing_ny_nj_usd": (
                            float(restricted["total_cost_usd"])
                            - float(allowed["total_cost_usd"])
                        ),
                        "time_avoided_by_allowing_ny_nj_h": (
                            float(restricted["quantity_weighted_mean_delivery_time_h"])
                            - float(allowed["quantity_weighted_mean_delivery_time_h"])
                        ),
                        "allowed_ny_nj_share": allowed["ny_nj_share"],
                    }
                )
    write_csv(
        args.output / "counterfactual_value_8_vs_full.csv",
        value_rows,
        list(value_rows[0].keys()),
    )
    return endpoint_payloads, share_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate structural paths and compare exact 8-path/full endpoints."
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--paths8-results",
        type=Path,
        default=Path("results/v17_core_exact_paths8"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/v20_final_path_coverage")
    )
    parser.add_argument("--enumeration-cap-per-gateway", type=int, default=100_000)
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    required_paths8_files = (
        args.paths8_results / "exact_endpoint_summary.csv",
        args.paths8_results / "exact_endpoint_ny_nj_share_ranges.csv",
    )
    missing_paths8 = [str(path) for path in required_paths8_files if not path.exists()]
    if missing_paths8:
        raise FileNotFoundError(
            "Run or restore the v1.7 eight-path exact results first: "
            + ", ".join(missing_paths8)
        )
    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    conditions = coverage_conditions()
    case_enumerations = {
        case_id: enumerate_case_paths(
            select_case(data, case_id),
            base_config,
            args.enumeration_cap_per_gateway,
        )
        for case_id in ("D1", "D2", "D3")
    }
    audit_rows = write_coverage_audit(
        args, data, base_config, case_enumerations
    )
    incomplete = [
        row for row in audit_rows
        if int(row["structural_enumeration_complete"]) != 1
    ]
    if incomplete:
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "structural_enumeration_complete": False,
            "enumeration_cap_per_gateway": args.enumeration_cap_per_gateway,
            "incomplete_od_gateway_rows": incomplete,
            "exact_endpoint_experiment_run": False,
            "interpretation": (
                "The safety cap was reached. Outputs are an inventory audit only "
                "and must not be described as exhaustive path coverage."
            ),
        }
        write_json_atomic(args.output / "manifest.json", manifest)
        raise RuntimeError(
            "Structural enumeration cap reached; exact full-library run was not started."
        )

    run_full_endpoints(
        args, data, base_config, conditions, case_enumerations
    )
    run_full_share_bounds(
        args, data, base_config, conditions, case_enumerations
    )
    endpoints, shares = finalize(args, data, conditions)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "structural_enumeration_complete": True,
        "enumeration_cap_per_gateway": args.enumeration_cap_per_gateway,
        "conditions": [asdict(item) for item in conditions],
        "martins_limits_audited": list(MARTINS_LIMITS),
        "full_endpoint_tasks": len(endpoints),
        "full_share_regions": len(shares),
        "paths8_source": str(args.paths8_results),
        "interpretation": (
            "The full library enumerates all simple paths satisfying the stated "
            "leg, mode-change, and no-node-revisit restrictions; it does not "
            "apply static cost--time dominance pruning."
        ),
    }
    write_json_atomic(args.output / "manifest.json", manifest)
    print(f"Final path coverage written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
