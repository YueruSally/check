#!/usr/bin/env python3
"""Exact delay-by-inland-capacity interaction on the full path library."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import capacity_usage_rows
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import ExactCondition, condition_config
from run_final_path_coverage import (
    build_full_model,
    enumerate_case_paths,
    share_bound_complete,
    solve_share_bound_with_numerical_retry,
    zero_share_lower_bound_certificate,
)
from run_supplementary_core_exact import (
    ENDPOINTS,
    allocations_from_payload,
    endpoint_payload,
    endpoint_region_bounds,
    endpoints_coincident,
    load_json,
    share_bound_payload,
    write_csv,
    write_json_atomic,
)


CAPACITY_MULTIPLIERS = (0.50, 1.0, 1.50)
TARGETS = {
    "D2": {"arc_id": "us-d-02", "delays_h": (264.0, 276.0, 288.0)},
    "D3": {"arc_id": "us-d-03", "delays_h": (216.0, 240.0, 264.0)},
}
CORE_REUSED = {
    ("D2", 264.0, 1.0),
    ("D2", 288.0, 1.0),
    ("D3", 240.0, 1.0),
    ("D3", 264.0, 1.0),
}


@dataclass(frozen=True)
class InlandSpec:
    condition: ExactCondition
    binding_arc_id: str
    capacity_multiplier: float
    base_capacity_feu_per_day: float


def label(value: float) -> str:
    return f"{int(round(100 * value)):03d}pct"


def specs(base_cases) -> tuple[InlandSpec, ...]:
    rows: list[InlandSpec] = []
    for case_id, target in TARGETS.items():
        arc_id = str(target["arc_id"])
        capacity = base_cases[case_id].arcs[arc_id].capacity_feu_per_day
        if capacity is None:
            raise ValueError(f"Target inland arc {arc_id} has no daily capacity.")
        for delay_h in target["delays_h"]:
            for multiplier in CAPACITY_MULTIPLIERS:
                if (case_id, float(delay_h), multiplier) in CORE_REUSED:
                    continue
                rows.append(
                    InlandSpec(
                        condition=ExactCondition(
                            case_id=case_id,
                            condition_id=(
                                f"delay_{int(delay_h)}h_inland_{arc_id}_capacity_"
                                f"{label(multiplier)}"
                            ),
                            family="delay_inland_capacity_interaction",
                            uswc_delay_h=float(delay_h),
                        ),
                        binding_arc_id=arc_id,
                        capacity_multiplier=multiplier,
                        base_capacity_feu_per_day=float(capacity),
                    )
                )
    return tuple(rows)


def adjusted_case_data(base_case, spec: InlandSpec):
    arc = base_case.arcs[spec.binding_arc_id]
    arcs = dict(base_case.arcs)
    arcs[spec.binding_arc_id] = replace(
        arc,
        capacity_feu_per_day=(
            spec.base_capacity_feu_per_day * spec.capacity_multiplier
        ),
    )
    return replace(base_case, arcs=arcs)


def validation_id(spec: InlandSpec, access: str, endpoint: str) -> str:
    return f"{spec.condition.case_id}_{spec.condition.condition_id}_{access}_{endpoint}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path("results/v26_full_structural_inland_bottleneck"),
    )
    parser.add_argument("--enumeration-cap-per-gateway", type=int, default=100_000)
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    structural_manifest = load_json(args.structural_results / "manifest.json")
    if not structural_manifest.get("structural_enumeration_complete"):
        raise RuntimeError("The structural path source is not exhaustive.")

    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    base_cases = {case_id: select_case(data, case_id) for case_id in TARGETS}
    experiment_specs = specs(base_cases)
    enumerations = {
        case_id: enumerate_case_paths(
            base_cases[case_id], base_config, args.enumeration_cap_per_gateway
        )
        for case_id in TARGETS
    }
    if any(
        not enumeration.complete
        for by_gateway in enumerations.values()
        for enumeration in by_gateway.values()
    ):
        raise RuntimeError("Structural enumeration cap reached.")

    checkpoint_root = args.output / "checkpoints"
    total = len(experiment_specs) * 2 * len(ENDPOINTS)
    completed = 0
    for spec in experiment_specs:
        case_id = spec.condition.case_id
        case_data = adjusted_case_data(base_cases[case_id], spec)
        for access in ("allowed", "restricted"):
            config = condition_config(base_config, spec.condition, gateway_access=access)
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
                        case_data, config, enumerations[case_id]
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
                        "library_variant": "structural_full",
                        "binding_arc_id": spec.binding_arc_id,
                        "binding_arc_capacity_multiplier": spec.capacity_multiplier,
                        "binding_arc_base_capacity_feu_per_day": (
                            spec.base_capacity_feu_per_day
                        ),
                        "binding_arc_capacity_feu_per_day": (
                            spec.base_capacity_feu_per_day
                            * spec.capacity_multiplier
                        ),
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
    spec_index = {
        (spec.condition.case_id, spec.condition.condition_id): spec
        for spec in experiment_specs
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
        case_id = spec.condition.case_id
        case_data = adjusted_case_data(base_cases[case_id], spec)
        config = condition_config(base_config, spec.condition, gateway_access="allowed")
        model = None
        cost_bound, time_bound = endpoint_region_bounds(
            endpoint,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        for direction in ("min", "max"):
            bound_id = (
                f"{case_id}_{spec.condition.condition_id}_{region}_{direction}_"
                "ny_nj_share_structural_full"
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
                        case_data, config, enumerations[case_id]
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
                    "library_variant": "structural_full",
                    "binding_arc_id": spec.binding_arc_id,
                    "binding_arc_capacity_multiplier": spec.capacity_multiplier,
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
    capacity_rows: list[dict[str, object]] = []
    for payload in endpoint_payloads:
        validation = str(payload["validation_id"])
        spec = spec_index[(str(payload["case_id"]), str(payload["condition_id"]))]
        case_data = adjusted_case_data(base_cases[spec.condition.case_id], spec)
        config = condition_config(
            base_config,
            spec.condition,
            gateway_access=str(payload["gateway_access"]),
        )
        allocations = list(allocations_from_payload(payload))
        for call in payload["solver_calls"]:
            solver_rows.append({"validation_id": validation, **call})
        for allocation_id, allocation in enumerate(payload["allocations"], start=1):
            allocation_rows.append(
                {
                    "validation_id": validation,
                    "allocation_id": allocation_id,
                    "shipment_id": allocation["shipment_id"],
                    "quantity_feu": allocation["quantity_feu"],
                    "gateway": allocation["gateway"],
                    "arc_ids": "|".join(allocation["arc_ids"]),
                }
            )
        for row in capacity_usage_rows(case_data, allocations, config):
            capacity_rows.append({"validation_id": validation, **row})
    write_csv(args.output / "exact_endpoint_allocations.csv", allocation_rows, list(allocation_rows[0]))
    write_csv(args.output / "exact_endpoint_solver_calls.csv", solver_rows, list(solver_rows[0]))
    write_csv(args.output / "exact_endpoint_capacity_usage.csv", capacity_rows, list(capacity_rows[0]))

    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in share_payloads:
        key = (str(row["case_id"]), str(row["condition_id"]), str(row["endpoint_region"]))
        groups.setdefault(key, {})[str(row["direction"])] = row
    range_rows = []
    for (case_id, condition_id, region), directions in sorted(groups.items()):
        minimum, maximum = directions["min"], directions["max"]
        spec = spec_index[(case_id, condition_id)]
        range_rows.append(
            {
                "case_id": case_id,
                "condition_id": condition_id,
                "delay_h": spec.condition.uswc_delay_h,
                "binding_arc_id": spec.binding_arc_id,
                "binding_arc_capacity_multiplier": spec.capacity_multiplier,
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
                    "delay_h": spec.condition.uswc_delay_h,
                    "binding_arc_id": spec.binding_arc_id,
                    "binding_arc_capacity_multiplier": spec.capacity_multiplier,
                    "binding_arc_capacity_feu_per_day": (
                        spec.base_capacity_feu_per_day
                        * spec.capacity_multiplier
                    ),
                    "endpoint": endpoint,
                    "allowed_ny_nj_share": allowed["ny_nj_share"],
                    "cost_avoided_by_allowing_ny_nj_usd": (
                        float(restricted["total_cost_usd"])
                        - float(allowed["total_cost_usd"])
                    ),
                    "time_avoided_by_allowing_ny_nj_h": (
                        float(restricted["quantity_weighted_mean_delivery_time_h"])
                        - float(allowed["quantity_weighted_mean_delivery_time_h"])
                    ),
                }
            )
    write_csv(args.output / "inland_bottleneck_gateway_value.csv", value_rows, list(value_rows[0]))

    write_json_atomic(
        args.output / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "library_variant": "structural_full",
            "structural_source": str(args.structural_results),
            "capacity_multipliers": list(CAPACITY_MULTIPLIERS),
            "targets": TARGETS,
            "core_exact_conditions_reused_in_analysis": [
                {"case_id": case, "delay_h": delay, "capacity_multiplier": mult}
                for case, delay, mult in sorted(CORE_REUSED)
            ],
            "endpoint_tasks": len(endpoint_payloads),
            "share_bound_tasks": len(share_payloads),
            "interpretation_limits": [
                "Only one diagnosed LA/LB-to-inland daily arc is changed per destination.",
                "The experiment identifies an interaction mechanism and is not a general inland-network capacity grid.",
                "The four already available 100% capacity conditions are reused from v2.1.",
            ],
        },
    )
    print(f"Full structural inland-bottleneck interaction written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
