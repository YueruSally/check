#!/usr/bin/env python3
"""Calibrate baseline NY/NJ competitiveness on the exhaustive path library."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.domain import Mode
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import ExactCondition, condition_config
from run_final_path_coverage import (
    GATEWAYS,
    build_full_model,
    enumerate_case_paths,
    share_bound_complete,
    solve_share_bound_with_numerical_retry,
    zero_share_lower_bound_certificate,
)
from run_supplementary_core_exact import (
    ENDPOINTS,
    endpoint_payload,
    endpoint_region_bounds,
    endpoints_coincident,
    load_json,
    share_bound_payload,
    write_csv,
    write_json_atomic,
)


CASES = ("D1", "D2", "D3")
USEC_COST_MULTIPLIERS = (1.0, 0.90, 0.80, 0.70)
PANAMA_TIME_REDUCTIONS_H = (48.0, 96.0)


@dataclass(frozen=True)
class CalibrationSpec:
    condition: ExactCondition
    dimension: str
    value: float


def specs() -> tuple[CalibrationSpec, ...]:
    rows: list[CalibrationSpec] = []
    for case_id in CASES:
        rows.extend(
            CalibrationSpec(
                ExactCondition(
                    case_id=case_id,
                    condition_id=f"baseline_usec_cost_{int(round(100 * multiplier)):03d}pct",
                    family="baseline_gateway_calibration",
                ),
                "usec_ocean_cost_multiplier",
                multiplier,
            )
            for multiplier in USEC_COST_MULTIPLIERS
        )
        rows.extend(
            CalibrationSpec(
                ExactCondition(
                    case_id=case_id,
                    condition_id=f"baseline_panama_time_minus_{int(hours)}h",
                    family="baseline_gateway_calibration",
                ),
                "panama_transit_time_reduction_h",
                hours,
            )
            for hours in PANAMA_TIME_REDUCTIONS_H
        )
    return tuple(rows)


def calibrated_case_data(case_data, spec: CalibrationSpec):
    if spec.dimension != "panama_transit_time_reduction_h":
        return case_data
    arcs = {}
    for arc_id, arc in case_data.arcs.items():
        if arc.via_panama and arc.mode == Mode.OCEAN:
            reduced = arc.transit_time_h - spec.value
            if reduced <= 0:
                raise ValueError(
                    f"Panama time reduction makes {arc_id} non-positive: {reduced}"
                )
            arcs[arc_id] = replace(arc, transit_time_h=reduced)
        else:
            arcs[arc_id] = arc
    return replace(case_data, arcs=arcs)


def calibrated_config(base_config, spec: CalibrationSpec, access: str):
    config = condition_config(base_config, spec.condition, gateway_access=access)
    if spec.dimension != "usec_ocean_cost_multiplier":
        return config
    return replace(
        config,
        scenario=replace(
            config.scenario,
            usec_ocean_cost_multiplier=spec.value,
        ),
    )


def validation_id(spec: CalibrationSpec, access: str, endpoint: str) -> str:
    return f"{spec.condition.case_id}_{spec.condition.condition_id}_{access}_{endpoint}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact baseline gateway calibration with all structural paths."
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--structural-results",
        type=Path,
        default=Path("results/v21_full_structural_core_exact"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v23_full_structural_baseline_calibration"),
    )
    parser.add_argument("--enumeration-cap-per-gateway", type=int, default=100_000)
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-extra-service-departures", type=int, default=0)
    args = parser.parse_args()

    structural_manifest = load_json(args.structural_results / "manifest.json")
    if not structural_manifest.get("structural_enumeration_complete"):
        raise RuntimeError("The structural path source is not exhaustive.")

    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    experiment_specs = specs()
    base_cases = {case_id: select_case(data, case_id) for case_id in CASES}
    enumerations = {
        case_id: enumerate_case_paths(
            case_data, base_config, args.enumeration_cap_per_gateway
        )
        for case_id, case_data in base_cases.items()
    }
    if any(
        not enumeration.complete
        for by_gateway in enumerations.values()
        for enumeration in by_gateway.values()
    ):
        raise RuntimeError("Structural enumeration cap reached.")

    inventory_rows: list[dict[str, object]] = []
    for case_id, case_data in base_cases.items():
        for (origin, destination, gateway), enumeration in sorted(
            enumerations[case_id].items()
        ):
            for path_number, label in enumerate(enumeration.labels, start=1):
                inventory_rows.append(
                    {
                        "case_id": case_id,
                        "origin": origin,
                        "destination": destination,
                        "gateway": gateway,
                        "path_number": path_number,
                        "arc_ids": "|".join(label.arc_ids),
                        "static_cost_usd_per_feu": label.cost_usd_per_feu,
                        "static_elapsed_h": label.elapsed_h,
                        "minimum_static_cost_for_gateway": int(
                            label.cost_usd_per_feu
                            == min(x.cost_usd_per_feu for x in enumeration.labels)
                        ),
                        "minimum_static_time_for_gateway": int(
                            label.elapsed_h
                            == min(x.elapsed_h for x in enumeration.labels)
                        ),
                    }
                )
    write_csv(
        args.output / "baseline_gateway_path_inventory.csv",
        inventory_rows,
        list(inventory_rows[0]),
    )

    checkpoint_root = args.output / "checkpoints"
    total = len(experiment_specs) * 2 * len(ENDPOINTS)
    completed = 0
    for spec in experiment_specs:
        case_data = calibrated_case_data(base_cases[spec.condition.case_id], spec)
        for access in ("allowed", "restricted"):
            config = calibrated_config(base_config, spec, access)
            model = None
            max_paths = None
            for endpoint, primary in ENDPOINTS:
                task_id = validation_id(spec, access, endpoint)
                checkpoint = checkpoint_root / "endpoints" / f"{task_id}.json"
                if checkpoint.exists():
                    existing = load_json(checkpoint)
                    if int(existing.get("lexicographic_proven", 0)) == 1:
                        completed += 1
                        print(f"[{completed}/{total}] resume {task_id}", flush=True)
                        continue
                if model is None:
                    model, max_paths = build_full_model(
                        case_data,
                        config,
                        enumerations[spec.condition.case_id],
                        max_extra_service_departures=args.max_extra_service_departures,
                    )
                from china_us_multimodal.exact_endpoint import solve_lexicographic_endpoint

                result = solve_lexicographic_endpoint(
                    model,
                    primary_objective=primary,
                    time_limit_seconds=args.time_limit_seconds,
                    mip_relative_gap=args.mip_relative_gap,
                    primary_relative_tolerance=args.relative_tolerance,
                    primary_absolute_tolerance=args.absolute_tolerance,
                )
                payload = endpoint_payload(
                    condition=spec.condition,
                    gateway_access=access,
                    endpoint=endpoint,
                    model=model,
                    result=result,
                    max_paths_per_gateway=int(max_paths),
                )
                payload.update(
                    {
                        "validation_id": task_id,
                        "library_variant": (
                            "structural_full" if args.max_extra_service_departures == 0
                            else f"structural_full_service_plus_{args.max_extra_service_departures}"
                        ),
                        "calibration_dimension": spec.dimension,
                        "calibration_value": spec.value,
                    }
                )
                write_json_atomic(checkpoint, payload)
                completed += 1
                print(
                    f"[{completed}/{total}] {task_id} "
                    f"proven={payload['lexicographic_proven']}",
                    flush=True,
                )

    endpoint_payloads = [
        load_json(path)
        for path in sorted((checkpoint_root / "endpoints").glob("*.json"))
    ]
    endpoint_index = {
        (
            row["case_id"],
            row["condition_id"],
            row["gateway_access"],
            row["endpoint"],
        ): row
        for row in endpoint_payloads
    }

    planned = []
    for spec in experiment_specs:
        endpoints = {
            endpoint: endpoint_index[
                (spec.condition.case_id, spec.condition.condition_id, "allowed", endpoint)
            ]
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
            planned.append((spec, "coincident_endpoints", endpoints["min_cost_then_time"]))
        else:
            planned.extend((spec, endpoint, row) for endpoint, row in endpoints.items())

    share_total = len(planned) * 2
    share_completed = 0
    for spec, region, endpoint in planned:
        case_data = calibrated_case_data(base_cases[spec.condition.case_id], spec)
        config = calibrated_config(base_config, spec, "allowed")
        model = None
        cost_bound, time_bound = endpoint_region_bounds(
            endpoint,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        for direction in ("min", "max"):
            bound_id = (
                f"{spec.condition.case_id}_{spec.condition.condition_id}_{region}_"
                f"{direction}_ny_nj_share_structural_full"
            )
            checkpoint = checkpoint_root / "share_bounds" / f"{bound_id}.json"
            if checkpoint.exists():
                existing = load_json(checkpoint)
                if share_bound_complete(existing):
                    share_completed += 1
                    print(
                        f"[share {share_completed}/{share_total}] resume {bound_id}",
                        flush=True,
                    )
                    continue
            if direction == "min" and float(endpoint.get("ny_nj_share", 1.0)) <= 1e-12:
                payload = zero_share_lower_bound_certificate(
                    bound_id=bound_id,
                    condition=spec.condition,
                    endpoint_region=region,
                    cost_upper_bound=cost_bound,
                    time_upper_bound=time_bound,
                    endpoint=endpoint,
                )
            else:
                if model is None:
                    model, _ = build_full_model(
                        case_data,
                        config,
                        enumerations[spec.condition.case_id],
                        max_extra_service_departures=args.max_extra_service_departures,
                    )
                result, applied_cost, applied_time, attempts, retry_mode = (
                    solve_share_bound_with_numerical_retry(
                        model=model,
                        direction=direction,
                        cost_bound=cost_bound,
                        time_bound=time_bound,
                        time_limit_seconds=args.time_limit_seconds,
                        mip_relative_gap=args.mip_relative_gap,
                    )
                )
                payload = share_bound_payload(
                    bound_id=bound_id,
                    condition=spec.condition,
                    endpoint_region=region,
                    direction=direction,
                    cost_upper_bound=applied_cost,
                    time_upper_bound=applied_time,
                    result=result,
                    data=case_data,
                )
                payload.update(
                    {
                        "strict_cost_upper_bound_usd": cost_bound,
                        "strict_time_upper_bound_h": time_bound,
                        "share_bound_retry_mode": retry_mode,
                        "solver_attempts": attempts,
                    }
                )
            payload.update(
                {
                    "library_variant": (
                        "structural_full" if args.max_extra_service_departures == 0
                        else f"structural_full_service_plus_{args.max_extra_service_departures}"
                    ),
                    "calibration_dimension": spec.dimension,
                    "calibration_value": spec.value,
                }
            )
            write_json_atomic(checkpoint, payload)
            share_completed += 1
            print(
                f"[share {share_completed}/{share_total}] {bound_id} "
                f"proven={payload['proven_optimal']} share={payload['ny_nj_share']}",
                flush=True,
            )

    share_payloads = [
        load_json(path)
        for path in sorted((checkpoint_root / "share_bounds").glob("*.json"))
    ]
    unresolved = [row for row in share_payloads if not share_bound_complete(row)]
    if unresolved:
        raise RuntimeError(
            "Unresolved share bounds: "
            + ", ".join(str(row["bound_id"]) for row in unresolved)
        )

    endpoint_rows = [
        {key: value for key, value in row.items() if key not in {"solver_calls", "allocations"}}
        for row in endpoint_payloads
    ]
    write_csv(args.output / "exact_endpoint_summary.csv", endpoint_rows, list(endpoint_rows[0]))

    allocation_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    for payload in endpoint_payloads:
        for call in payload["solver_calls"]:
            solver_rows.append({"validation_id": payload["validation_id"], **call})
        for allocation_id, allocation in enumerate(payload["allocations"], start=1):
            allocation_rows.append(
                {
                    "validation_id": payload["validation_id"],
                    "allocation_id": allocation_id,
                    "shipment_id": allocation["shipment_id"],
                    "quantity_feu": allocation["quantity_feu"],
                    "gateway": allocation["gateway"],
                    "arc_ids": "|".join(allocation["arc_ids"]),
                    "service_departure_ranks": "|".join(
                        str(item)
                        for item in allocation.get("service_departure_ranks", [])
                    ),
                }
            )
    write_csv(args.output / "exact_endpoint_allocations.csv", allocation_rows, list(allocation_rows[0]))
    write_csv(args.output / "exact_endpoint_solver_calls.csv", solver_rows, list(solver_rows[0]))

    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in share_payloads:
        key = (str(row["case_id"]), str(row["condition_id"]), str(row["endpoint_region"]))
        groups.setdefault(key, {})[str(row["direction"])] = row
    range_rows = []
    for (case_id, condition_id, region), directions in sorted(groups.items()):
        minimum, maximum = directions["min"], directions["max"]
        range_rows.append(
            {
                "case_id": case_id,
                "condition_id": condition_id,
                "endpoint_region": region,
                "ny_nj_min_share": minimum["ny_nj_share"],
                "ny_nj_max_share": maximum["ny_nj_share"],
                "min_proven": minimum["proven_optimal"],
                "max_proven": maximum["proven_optimal"],
            }
        )
    write_csv(args.output / "endpoint_ny_nj_share_ranges.csv", range_rows, list(range_rows[0]))

    value_rows = []
    for spec in experiment_specs:
        for endpoint, _ in ENDPOINTS:
            allowed = endpoint_index[
                (spec.condition.case_id, spec.condition.condition_id, "allowed", endpoint)
            ]
            restricted = endpoint_index[
                (spec.condition.case_id, spec.condition.condition_id, "restricted", endpoint)
            ]
            value_rows.append(
                {
                    "case_id": spec.condition.case_id,
                    "condition_id": spec.condition.condition_id,
                    "calibration_dimension": spec.dimension,
                    "calibration_value": spec.value,
                    "endpoint": endpoint,
                    "allowed_cost_usd": allowed["total_cost_usd"],
                    "restricted_cost_usd": restricted["total_cost_usd"],
                    "cost_avoided_by_allowing_ny_nj_usd": (
                        float(restricted["total_cost_usd"])
                        - float(allowed["total_cost_usd"])
                    ),
                    "allowed_time_h": allowed["quantity_weighted_mean_delivery_time_h"],
                    "restricted_time_h": restricted["quantity_weighted_mean_delivery_time_h"],
                    "time_avoided_by_allowing_ny_nj_h": (
                        float(restricted["quantity_weighted_mean_delivery_time_h"])
                        - float(allowed["quantity_weighted_mean_delivery_time_h"])
                    ),
                    "allowed_ny_nj_share": allowed["ny_nj_share"],
                }
            )
    write_csv(args.output / "baseline_calibration_value.csv", value_rows, list(value_rows[0]))

    write_json_atomic(
        args.output / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "library_variant": (
                "structural_full" if args.max_extra_service_departures == 0
                else f"structural_full_service_plus_{args.max_extra_service_departures}"
            ),
            "max_extra_service_departures": args.max_extra_service_departures,
            "structural_source": str(args.structural_results),
            "usec_ocean_cost_multipliers": list(USEC_COST_MULTIPLIERS),
            "panama_transit_time_reductions_h": [0.0, *PANAMA_TIME_REDUCTIONS_H],
            "cost_and_time_changes_are_one_factor_at_a_time": True,
            "specifications": [asdict(item) for item in experiment_specs],
            "endpoint_tasks": len(endpoint_payloads),
            "share_bound_tasks": len(share_payloads),
            "interpretation_limits": [
                "The tested levels identify a sampled transition interval, not a continuous market threshold.",
                "Static path inventory omits schedule waiting and is descriptive only; optimization results include timetable and capacity constraints.",
                "Gateway share uniqueness does not imply allocation uniqueness.",
            ],
        },
    )
    print(f"Full structural baseline calibration written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
