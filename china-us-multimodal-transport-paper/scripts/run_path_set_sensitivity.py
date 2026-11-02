#!/usr/bin/env python3
"""Exact 4-vs-8 candidate-path sensitivity for the fixed core conditions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.exact_endpoint import (
    build_exact_endpoint_model,
    solve_lexicographic_endpoint,
)
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import (
    condition_config,
    path_sensitivity_conditions,
)
from run_supplementary_core_exact import (
    ENDPOINTS,
    allocations_from_payload,
    endpoint_payload,
    gateway_for_path,
    load_json,
    read_csv,
    write_csv,
    write_json_atomic,
)


def path_signature(profile) -> tuple[str, tuple[str, ...]]:
    return profile.shipment_id, profile.arc_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exact endpoints with four and eight paths per gateway."
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--core-results", type=Path, default=Path("results/v14_core_exact_suite")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/v15_path_set_sensitivity")
    )
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    conditions = path_sensitivity_conditions()
    core_rows = read_csv(args.core_results / "exact_endpoint_summary.csv")
    if not core_rows:
        raise RuntimeError("Run the v1.4 exact core suite before path sensitivity.")
    core_index = {
        (row["case_id"], row["condition_id"], row["gateway_access"], row["endpoint"]): row
        for row in core_rows
    }
    checkpoints = args.output / "checkpoints"
    inventory_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    total = len(conditions) * 2 * len(ENDPOINTS)
    completed = 0
    for condition in conditions:
        case_data = select_case(data, condition.case_id)
        for access in ("allowed", "restricted"):
            config = condition_config(base_config, condition, gateway_access=access)
            model4 = build_exact_endpoint_model(
                case_data, config, max_paths_per_gateway=4
            )
            required_paths: dict[str, list[tuple[str, ...]]] = {}
            for profile in model4.profiles:
                required_paths.setdefault(profile.shipment_id, []).append(
                    profile.arc_ids
                )
            model8 = build_exact_endpoint_model(
                case_data,
                config,
                max_paths_per_gateway=8,
                required_paths_by_shipment={
                    shipment_id: tuple(paths)
                    for shipment_id, paths in required_paths.items()
                },
            )
            signatures4 = {path_signature(profile) for profile in model4.profiles}
            signatures8 = {path_signature(profile) for profile in model8.profiles}
            missing = signatures4 - signatures8
            if missing:
                example = sorted(missing)[:3]
                raise RuntimeError(
                    "The four-path library is not nested in the eight-path library: "
                    f"{condition.case_id}/{condition.condition_id}/{access}: {example}"
                )
            gateways4: dict[str, int] = {}
            gateways8: dict[str, int] = {}
            for profile in model4.profiles:
                gateway = gateway_for_path(case_data, profile.arc_ids)
                gateways4[gateway] = gateways4.get(gateway, 0) + 1
            for profile in model8.profiles:
                gateway = gateway_for_path(case_data, profile.arc_ids)
                gateways8[gateway] = gateways8.get(gateway, 0) + 1
                signature = path_signature(profile)
                inventory_rows.append(
                    {
                        "case_id": condition.case_id,
                        "condition_id": condition.condition_id,
                        "gateway_access": access,
                        "shipment_id": profile.shipment_id,
                        "gateway": gateway,
                        "arc_ids": "|".join(profile.arc_ids),
                        "unit_cost_usd": profile.unit_cost_usd,
                        "lead_time_h": profile.lead_time_h,
                        "present_in_4_path_library": int(signature in signatures4),
                        "present_in_8_path_library": 1,
                        "new_when_limit_is_8": int(signature not in signatures4),
                    }
                )
            count_rows.append(
                {
                    "case_id": condition.case_id,
                    "condition_id": condition.condition_id,
                    "gateway_access": access,
                    "paths_in_4_library": len(signatures4),
                    "paths_in_8_library": len(signatures8),
                    "new_paths": len(signatures8 - signatures4),
                    "four_is_subset_of_eight": 1,
                    "la_lb_paths_4": gateways4.get("la_lb", 0),
                    "la_lb_paths_8": gateways8.get("la_lb", 0),
                    "seattle_tacoma_paths_4": gateways4.get("seattle_tacoma", 0),
                    "seattle_tacoma_paths_8": gateways8.get("seattle_tacoma", 0),
                    "ny_nj_paths_4": gateways4.get("ny_nj", 0),
                    "ny_nj_paths_8": gateways8.get("ny_nj", 0),
                    "coverage_interpretation": (
                        "expanded_library"
                        if signatures8 - signatures4
                        else "no_new_paths_not_a_coverage_validation"
                    ),
                }
            )
            for endpoint, primary in ENDPOINTS:
                task_id = (
                    f"{condition.case_id}_{condition.condition_id}_{access}_"
                    f"{endpoint}_paths8"
                )
                checkpoint = checkpoints / f"{task_id}.json"
                if checkpoint.exists():
                    completed += 1
                    print(f"[{completed}/{total}] resume {task_id}", flush=True)
                    continue
                result = solve_lexicographic_endpoint(
                    model8,
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
                    model=model8,
                    result=result,
                    max_paths_per_gateway=8,
                )
                payload["validation_id"] = task_id
                write_json_atomic(checkpoint, payload)
                completed += 1
                print(
                    f"[{completed}/{total}] {task_id} "
                    f"proven={payload['lexicographic_proven']}",
                    flush=True,
                )

    payloads = [load_json(path) for path in sorted(checkpoints.glob("*.json"))]
    payload_index = {
        (row["case_id"], row["condition_id"], row["gateway_access"], row["endpoint"]): row
        for row in payloads
    }
    summary_rows = [
        {key: value for key, value in row.items() if key not in {"solver_calls", "allocations"}}
        for row in payloads
    ]
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
    write_csv(args.output / "path8_exact_endpoint_summary.csv", summary_rows, summary_fields)
    allocation_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    for payload in payloads:
        for call in payload["solver_calls"]:
            solver_rows.append({"validation_id": payload["validation_id"], **call})
        for index, allocation in enumerate(allocations_from_payload(payload), start=1):
            allocation_rows.append(
                {
                    "validation_id": payload["validation_id"],
                    "allocation_id": index,
                    "shipment_id": allocation.shipment_id,
                    "quantity_feu": allocation.quantity_feu,
                    "gateway": gateway_for_path(
                        select_case(data, str(payload["case_id"])), allocation.arc_ids
                    ),
                    "arc_ids": "|".join(allocation.arc_ids),
                }
            )
    write_csv(
        args.output / "path8_exact_endpoint_allocations.csv",
        allocation_rows,
        [
            "validation_id", "allocation_id", "shipment_id", "quantity_feu",
            "gateway", "arc_ids",
        ],
    )
    write_csv(
        args.output / "path8_solver_calls.csv",
        solver_rows,
        [
            "validation_id", "stage", "objective", "status_code", "status",
            "success", "objective_value", "dual_bound", "relative_gap",
            "node_count", "elapsed_seconds", "message",
        ],
    )
    write_csv(
        args.output / "candidate_path_inventory.csv",
        inventory_rows,
        [
            "case_id", "condition_id", "gateway_access", "shipment_id", "gateway",
            "arc_ids", "unit_cost_usd", "lead_time_h",
            "present_in_4_path_library", "present_in_8_path_library",
            "new_when_limit_is_8",
        ],
    )
    write_csv(
        args.output / "candidate_path_count_summary.csv",
        count_rows,
        list(count_rows[0].keys()),
    )

    comparison_rows: list[dict[str, object]] = []
    for condition in conditions:
        for access in ("allowed", "restricted"):
            for endpoint, _ in ENDPOINTS:
                key = (condition.case_id, condition.condition_id, access, endpoint)
                four = core_index[key]
                eight = payload_index[key]
                four_cost = float(four["total_cost_usd"])
                eight_cost = float(eight["total_cost_usd"])
                four_time = float(four["quantity_weighted_mean_delivery_time_h"])
                eight_time = float(eight["quantity_weighted_mean_delivery_time_h"])
                four_share = float(four["ny_nj_share"])
                eight_share = float(eight["ny_nj_share"])
                comparison_rows.append(
                    {
                        "case_id": condition.case_id,
                        "condition_id": condition.condition_id,
                        "gateway_access": access,
                        "endpoint": endpoint,
                        "paths4_proven": four["lexicographic_proven"],
                        "paths8_proven": eight["lexicographic_proven"],
                        "paths4_cost_usd": four_cost,
                        "paths8_cost_usd": eight_cost,
                        "cost_change_8_minus_4_usd": eight_cost - four_cost,
                        "paths4_time_h": four_time,
                        "paths8_time_h": eight_time,
                        "time_change_8_minus_4_h": eight_time - four_time,
                        "paths4_ny_nj_share": four_share,
                        "paths8_ny_nj_share": eight_share,
                        "ny_nj_share_change": eight_share - four_share,
                        "paths4_reaches_10pct": int(four_share >= 0.10 - 1e-9),
                        "paths8_reaches_10pct": int(eight_share >= 0.10 - 1e-9),
                        "gateway_use_conclusion_unchanged": int(
                            (four_share >= 0.10 - 1e-9)
                            == (eight_share >= 0.10 - 1e-9)
                        ),
                    }
                )
    write_csv(
        args.output / "path_limit_endpoint_comparison.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )

    value_rows: list[dict[str, object]] = []
    endpoint_comparison_index = {
        (row["case_id"], row["condition_id"], row["gateway_access"], row["endpoint"]): row
        for row in comparison_rows
    }
    for condition in conditions:
        for endpoint, _ in ENDPOINTS:
            allowed = endpoint_comparison_index[
                (condition.case_id, condition.condition_id, "allowed", endpoint)
            ]
            restricted = endpoint_comparison_index[
                (condition.case_id, condition.condition_id, "restricted", endpoint)
            ]
            for limit in (4, 8):
                value_rows.append(
                    {
                        "case_id": condition.case_id,
                        "condition_id": condition.condition_id,
                        "endpoint": endpoint,
                        "max_paths_per_gateway": limit,
                        "cost_avoided_by_allowing_ny_nj_usd": (
                            float(restricted[f"paths{limit}_cost_usd"])
                            - float(allowed[f"paths{limit}_cost_usd"])
                        ),
                        "time_avoided_by_allowing_ny_nj_h": (
                            float(restricted[f"paths{limit}_time_h"])
                            - float(allowed[f"paths{limit}_time_h"])
                        ),
                        "allowed_ny_nj_share": allowed[f"paths{limit}_ny_nj_share"],
                    }
                )
    write_csv(
        args.output / "path_limit_counterfactual_value.csv",
        value_rows,
        list(value_rows[0].keys()),
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": [asdict(condition) for condition in conditions],
        "endpoint_tasks_paths8": total,
        "paths4_source": str(args.core_results),
        "nested_library_required": True,
        "conditions_without_new_paths": [
            {
                "case_id": row["case_id"],
                "condition_id": row["condition_id"],
                "gateway_access": row["gateway_access"],
            }
            for row in count_rows
            if int(row["new_paths"]) == 0
        ],
        "interpretation": (
            "A condition with no added path is reported as no effective coverage "
            "expansion, not as evidence that four paths are sufficient."
        ),
    }
    write_json_atomic(args.output / "manifest.json", manifest)
    print(f"Path-set sensitivity written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
