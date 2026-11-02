#!/usr/bin/env python3
"""Targeted capacity sensitivity for the residual 727.5 FEU NY/NJ quantity.

The eight-path diagnostic identifies one binding LA/LB-to-inland daily arc in
the two conditions where 727.5 FEU remains at an exact endpoint.  This script
changes only that identified arc capacity; it does not introduce a general
capacity, timetable, cost, or due-date scan.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.exact_endpoint import (
    build_exact_endpoint_model,
    solve_bounded_milp,
    solve_lexicographic_endpoint,
)
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import ExactCondition, condition_config
from run_supplementary_core_exact import (
    ENDPOINTS,
    allocations_from_payload,
    endpoint_payload,
    endpoint_region_bounds,
    gateway_for_path,
    load_json,
    share_bound_payload,
    write_csv,
    write_json_atomic,
)


CAPACITY_MULTIPLIERS = (0.50, 1.00, 1.50, 2.00)
TARGETS = (
    ("D2", "delay_288h", 288.0, "us-d-02"),
    ("D3", "delay_240h", 240.0, "us-d-03"),
)


def scaled_arc_capacity(data, arc_id: str, multiplier: float):
    arc = data.arcs[arc_id]
    if arc.capacity_feu_per_day is None:
        raise ValueError(f"Target arc {arc_id!r} has no daily capacity.")
    arcs = dict(data.arcs)
    arcs[arc_id] = replace(
        arc,
        capacity_feu_per_day=arc.capacity_feu_per_day * multiplier,
    )
    return replace(data, arcs=arcs), float(arc.capacity_feu_per_day)


def condition_for(
    case_id: str,
    base_condition_id: str,
    delay_h: float,
    multiplier: float,
) -> ExactCondition:
    label = int(round(100.0 * multiplier))
    return ExactCondition(
        case_id=case_id,
        condition_id=f"{base_condition_id}_la_inland_capacity_{label}pct",
        family="binding_arc_capacity_sensitivity",
        uswc_delay_h=delay_h,
    )


def endpoint_task_id(condition: ExactCondition, access: str, endpoint: str) -> str:
    return f"{condition.case_id}_{condition.condition_id}_{access}_{endpoint}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the targeted LA/LB inland daily-arc capacity sensitivity."
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v19_binding_arc_capacity_sensitivity"),
    )
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    checkpoints = args.output / "checkpoints"
    target_metadata: dict[str, dict[str, object]] = {}
    conditions: list[tuple[ExactCondition, object]] = []
    for case_id, base_condition_id, delay_h, arc_id in TARGETS:
        original_case = select_case(data, case_id)
        for multiplier in CAPACITY_MULTIPLIERS:
            condition = condition_for(
                case_id, base_condition_id, delay_h, multiplier
            )
            case_data, original_capacity = scaled_arc_capacity(
                original_case, arc_id, multiplier
            )
            target_metadata[condition.condition_id] = {
                "binding_arc_id": arc_id,
                "binding_arc_capacity_multiplier": multiplier,
                "original_binding_arc_capacity_feu_per_day": original_capacity,
                "binding_arc_capacity_feu_per_day": original_capacity * multiplier,
            }
            conditions.append((condition, case_data))

    total = len(conditions) * 2 * len(ENDPOINTS)
    completed = 0
    for condition, case_data in conditions:
        for access in ("allowed", "restricted"):
            config = condition_config(base_config, condition, gateway_access=access)
            model = None
            for endpoint, primary in ENDPOINTS:
                task_id = endpoint_task_id(condition, access, endpoint)
                checkpoint = checkpoints / "endpoints" / f"{task_id}.json"
                if checkpoint.exists():
                    existing = load_json(checkpoint)
                    if int(existing.get("lexicographic_proven", 0)) == 1:
                        completed += 1
                        print(f"[{completed}/{total}] resume {task_id}", flush=True)
                        continue
                if model is None:
                    model = build_exact_endpoint_model(
                        case_data, config, max_paths_per_gateway=8
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
                    max_paths_per_gateway=8,
                )
                payload["validation_id"] = task_id
                payload.update(target_metadata[condition.condition_id])
                write_json_atomic(checkpoint, payload)
                completed += 1
                print(
                    f"[{completed}/{total}] {task_id} "
                    f"proven={payload['lexicographic_proven']}",
                    flush=True,
                )

    endpoint_payloads = [
        load_json(path)
        for path in sorted((checkpoints / "endpoints").glob("*.json"))
    ]
    endpoint_index = {
        (
            str(row["case_id"]),
            str(row["condition_id"]),
            str(row["gateway_access"]),
            str(row["endpoint"]),
        ): row
        for row in endpoint_payloads
    }

    share_total = len(conditions) * len(ENDPOINTS) * 2
    share_completed = 0
    for condition, case_data in conditions:
        config = condition_config(base_config, condition, gateway_access="allowed")
        model = None
        for endpoint, _ in ENDPOINTS:
            endpoint_row = endpoint_index[
                (condition.case_id, condition.condition_id, "allowed", endpoint)
            ]
            if int(endpoint_row["lexicographic_proven"]) != 1:
                continue
            cost_bound, time_bound = endpoint_region_bounds(
                endpoint_row,
                absolute_tolerance=args.absolute_tolerance,
                relative_tolerance=args.relative_tolerance,
            )
            for direction in ("min", "max"):
                bound_id = (
                    f"{condition.case_id}_{condition.condition_id}_{endpoint}_"
                    f"{direction}_ny_nj_share"
                )
                checkpoint = checkpoints / "share_bounds" / f"{bound_id}.json"
                if checkpoint.exists():
                    existing = load_json(checkpoint)
                    if int(existing.get("proven_optimal", 0)) == 1:
                        share_completed += 1
                        print(
                            f"[share {share_completed}/{share_total}] resume {bound_id}",
                            flush=True,
                        )
                        continue
                if model is None:
                    model = build_exact_endpoint_model(
                        case_data, config, max_paths_per_gateway=8
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
                    endpoint_region=endpoint,
                    direction=direction,
                    cost_upper_bound=cost_bound,
                    time_upper_bound=time_bound,
                    result=result,
                    data=case_data,
                )
                payload.update(target_metadata[condition.condition_id])
                write_json_atomic(checkpoint, payload)
                share_completed += 1
                print(
                    f"[share {share_completed}/{share_total}] {bound_id} "
                    f"proven={payload['proven_optimal']}",
                    flush=True,
                )

    endpoint_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    for payload in endpoint_payloads:
        endpoint_rows.append(
            {
                key: value
                for key, value in payload.items()
                if key not in {"solver_calls", "allocations"}
            }
        )
        for call in payload["solver_calls"]:
            solver_rows.append({"validation_id": payload["validation_id"], **call})
        case_data = select_case(data, str(payload["case_id"]))
        for index, allocation in enumerate(
            allocations_from_payload(payload), start=1
        ):
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

    endpoint_fields = list(endpoint_rows[0].keys())
    write_csv(args.output / "exact_endpoint_summary.csv", endpoint_rows, endpoint_fields)
    write_csv(
        args.output / "exact_endpoint_allocations.csv",
        allocation_rows,
        [
            "validation_id",
            "allocation_id",
            "shipment_id",
            "quantity_feu",
            "gateway",
            "arc_ids",
        ],
    )
    write_csv(
        args.output / "exact_solver_calls.csv",
        solver_rows,
        [
            "validation_id",
            "stage",
            "objective",
            "status_code",
            "status",
            "success",
            "objective_value",
            "dual_bound",
            "relative_gap",
            "node_count",
            "elapsed_seconds",
            "message",
        ],
    )

    share_payloads = [
        load_json(path)
        for path in sorted((checkpoints / "share_bounds").glob("*.json"))
    ]
    share_index = {
        (
            str(row["case_id"]),
            str(row["condition_id"]),
            str(row["endpoint_region"]),
            str(row["direction"]),
        ): row
        for row in share_payloads
    }
    share_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []
    for condition, _ in conditions:
        metadata = target_metadata[condition.condition_id]
        for endpoint, _ in ENDPOINTS:
            allowed = endpoint_index[
                (condition.case_id, condition.condition_id, "allowed", endpoint)
            ]
            restricted = endpoint_index[
                (condition.case_id, condition.condition_id, "restricted", endpoint)
            ]
            minimum = share_index[
                (condition.case_id, condition.condition_id, endpoint, "min")
            ]
            maximum = share_index[
                (condition.case_id, condition.condition_id, endpoint, "max")
            ]
            share_rows.append(
                {
                    "case_id": condition.case_id,
                    "condition_id": condition.condition_id,
                    "endpoint": endpoint,
                    **metadata,
                    "ny_nj_min_share": minimum["ny_nj_share"],
                    "ny_nj_max_share": maximum["ny_nj_share"],
                    "min_proven": minimum["proven_optimal"],
                    "max_proven": maximum["proven_optimal"],
                    "crosses_10pct": int(
                        float(minimum["ny_nj_share"]) < 0.10
                        <= float(maximum["ny_nj_share"])
                    ),
                }
            )
            value_rows.append(
                {
                    "case_id": condition.case_id,
                    "condition_id": condition.condition_id,
                    "endpoint": endpoint,
                    **metadata,
                    "allowed_proven": allowed["lexicographic_proven"],
                    "restricted_proven": restricted["lexicographic_proven"],
                    "allowed_ny_nj_share": allowed["ny_nj_share"],
                    "cost_avoided_by_allowing_ny_nj_usd": (
                        float(restricted["total_cost_usd"])
                        - float(allowed["total_cost_usd"])
                    ),
                    "time_avoided_by_allowing_ny_nj_h": (
                        float(restricted["quantity_weighted_mean_delivery_time_h"])
                        - float(allowed["quantity_weighted_mean_delivery_time_h"])
                    ),
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
    write_csv(
        args.output / "binding_arc_share_ranges.csv",
        share_rows,
        list(share_rows[0].keys()),
    )
    write_csv(
        args.output / "binding_arc_value_summary.csv",
        value_rows,
        list(value_rows[0].keys()),
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": "targeted binding LA/LB inland daily-arc capacity sensitivity",
        "capacity_multipliers": CAPACITY_MULTIPLIERS,
        "targets": [
            {
                "case_id": case_id,
                "base_condition_id": condition_id,
                "delay_h": delay_h,
                "binding_arc_id": arc_id,
            }
            for case_id, condition_id, delay_h, arc_id in TARGETS
        ],
        "max_paths_per_gateway": 8,
        "endpoint_tasks": total,
        "share_bound_tasks": share_total,
        "time_limit_seconds_per_solver_call": args.time_limit_seconds,
        "mip_relative_gap": args.mip_relative_gap,
        "interpretation_limit": (
            "Only the diagnosed binding LA/LB-to-inland daily arc is varied; "
            "this is not a general capacity or timetable sensitivity grid."
        ),
    }
    write_json_atomic(args.output / "manifest.json", manifest)
    print(f"Targeted capacity sensitivity written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
