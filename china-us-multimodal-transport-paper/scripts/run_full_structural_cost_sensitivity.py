#!/usr/bin/env python3
"""Exact tardiness-penalty and holding-cost sensitivity on the full path set."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import route_component_rows
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


PENALTY_MULTIPLIERS = (0.0, 0.25, 0.50, 1.0, 1.50, 2.0)
HOLDING_MULTIPLIERS = (0.0, 0.50, 2.0)
TARGETS = (("D1", 288.0), ("D2", 288.0), ("D3", 240.0))


@dataclass(frozen=True)
class SensitivitySpec:
    condition: ExactCondition
    dimension: str
    multiplier: float
    base_delay_h: float


def label(multiplier: float) -> str:
    return f"{int(round(100.0 * multiplier)):03d}pct"


def specs() -> tuple[SensitivitySpec, ...]:
    result: list[SensitivitySpec] = []
    for case_id, delay_h in TARGETS:
        for multiplier in PENALTY_MULTIPLIERS:
            result.append(
                SensitivitySpec(
                    ExactCondition(
                        case_id=case_id,
                        condition_id=(
                            f"delay_{int(delay_h)}h_penalty_{label(multiplier)}"
                        ),
                        family="tardiness_penalty_sensitivity",
                        uswc_delay_h=delay_h,
                    ),
                    "tardiness_penalty",
                    multiplier,
                    delay_h,
                )
            )
        for multiplier in HOLDING_MULTIPLIERS:
            result.append(
                SensitivitySpec(
                    ExactCondition(
                        case_id=case_id,
                        condition_id=(
                            f"delay_{int(delay_h)}h_holding_{label(multiplier)}"
                        ),
                        family="holding_cost_sensitivity",
                        uswc_delay_h=delay_h,
                    ),
                    "holding_cost",
                    multiplier,
                    delay_h,
                )
            )
    return tuple(result)


def scaled_case_data(case_data, spec: SensitivitySpec):
    if spec.dimension != "tardiness_penalty":
        return case_data
    shipments = {
        shipment_id: replace(
            shipment,
            tardiness_usd_per_feu_h=(
                None
                if shipment.tardiness_usd_per_feu_h is None
                else shipment.tardiness_usd_per_feu_h * spec.multiplier
            ),
        )
        for shipment_id, shipment in case_data.shipments.items()
    }
    return replace(case_data, shipments=shipments)


def sensitivity_config(base_config, spec: SensitivitySpec, access: str):
    config = condition_config(base_config, spec.condition, gateway_access=access)
    if spec.dimension == "tardiness_penalty":
        return replace(
            config,
            penalties=replace(
                config.penalties,
                tardiness_usd_per_feu_h=(
                    config.penalties.tardiness_usd_per_feu_h * spec.multiplier
                ),
            ),
        )
    return replace(
        config,
        operational_costs=replace(
            config.operational_costs,
            holding_usd_per_feu_h=(
                config.operational_costs.holding_usd_per_feu_h * spec.multiplier
            ),
        ),
    )


def validation_id(spec: SensitivitySpec, access: str, endpoint: str) -> str:
    return f"{spec.condition.case_id}_{spec.condition.condition_id}_{access}_{endpoint}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact cost-parameter sensitivity with structural paths."
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
        default=Path("results/v22_full_structural_cost_sensitivity"),
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
    experiment_specs = specs()
    base_cases = {case_id: select_case(data, case_id) for case_id, _ in TARGETS}
    enumerations = {
        case_id: enumerate_case_paths(
            case_data, base_config, args.enumeration_cap_per_gateway
        )
        for case_id, case_data in base_cases.items()
    }
    if any(
        not enumeration.complete
        for case_enumerations in enumerations.values()
        for enumeration in case_enumerations.values()
    ):
        raise RuntimeError("Structural enumeration cap reached.")

    checkpoint_root = args.output / "checkpoints"
    total = len(experiment_specs) * 2 * len(ENDPOINTS)
    completed = 0
    for spec in experiment_specs:
        case_data = scaled_case_data(base_cases[spec.condition.case_id], spec)
        for access in ("allowed", "restricted"):
            config = sensitivity_config(base_config, spec, access)
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
                        case_data, config, enumerations[spec.condition.case_id]
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
                        "sensitivity_dimension": spec.dimension,
                        "sensitivity_multiplier": spec.multiplier,
                        "base_delay_h": spec.base_delay_h,
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
        case_data = scaled_case_data(base_cases[spec.condition.case_id], spec)
        config = sensitivity_config(base_config, spec, "allowed")
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
                        case_data, config, enumerations[spec.condition.case_id]
                    )
                (
                    result,
                    applied_cost,
                    applied_time,
                    attempts,
                    retry_mode,
                ) = solve_share_bound_with_numerical_retry(
                    model=model,
                    direction=direction,
                    cost_bound=cost_bound,
                    time_bound=time_bound,
                    time_limit_seconds=args.time_limit_seconds,
                    mip_relative_gap=args.mip_relative_gap,
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
                    "sensitivity_dimension": spec.dimension,
                    "sensitivity_multiplier": spec.multiplier,
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
    endpoint_allocation_rows = []
    endpoint_solver_rows = []
    for payload in endpoint_payloads:
        for call in payload["solver_calls"]:
            endpoint_solver_rows.append(
                {"validation_id": payload["validation_id"], **call}
            )
        for allocation_id, allocation in enumerate(payload["allocations"], start=1):
            endpoint_allocation_rows.append(
                {
                    "validation_id": payload["validation_id"],
                    "allocation_id": allocation_id,
                    "shipment_id": allocation["shipment_id"],
                    "quantity_feu": allocation["quantity_feu"],
                    "gateway": allocation["gateway"],
                    "arc_ids": "|".join(allocation["arc_ids"]),
                }
            )
    write_csv(
        args.output / "exact_endpoint_allocations.csv",
        endpoint_allocation_rows,
        list(endpoint_allocation_rows[0]),
    )
    write_csv(
        args.output / "exact_endpoint_solver_calls.csv",
        endpoint_solver_rows,
        list(endpoint_solver_rows[0]),
    )

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
    component_rows = []
    spec_index = {
        (spec.condition.case_id, spec.condition.condition_id): spec
        for spec in experiment_specs
    }
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
                    "sensitivity_dimension": spec.dimension,
                    "sensitivity_multiplier": spec.multiplier,
                    "base_delay_h": spec.base_delay_h,
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
    write_csv(args.output / "counterfactual_value.csv", value_rows, list(value_rows[0]))

    for payload in endpoint_payloads:
        spec = spec_index[(str(payload["case_id"]), str(payload["condition_id"]))]
        case_data = scaled_case_data(base_cases[spec.condition.case_id], spec)
        config = sensitivity_config(base_config, spec, str(payload["gateway_access"]))
        rows = route_component_rows(
            case_data, list(allocations_from_payload(payload)), config
        )
        totals = {
            "transport_cost_usd": 0.0,
            "holding_cost_usd": 0.0,
            "transfer_cost_usd": 0.0,
            "processing_cost_usd": 0.0,
            "tardiness_cost_usd": 0.0,
        }
        for row in rows:
            totals["transport_cost_usd"] += sum(
                float(row[field])
                for field in (
                    "china_inland_transport_cost_usd",
                    "ocean_transport_cost_usd",
                    "us_inland_transport_cost_usd",
                )
            )
            for field in (
                "holding_cost_usd",
                "transfer_cost_usd",
                "processing_cost_usd",
                "tardiness_cost_usd",
            ):
                totals[field] += float(row[field])
        component_rows.append(
            {
                "validation_id": payload["validation_id"],
                "case_id": payload["case_id"],
                "condition_id": payload["condition_id"],
                "gateway_access": payload["gateway_access"],
                "endpoint": payload["endpoint"],
                "sensitivity_dimension": spec.dimension,
                "sensitivity_multiplier": spec.multiplier,
                **totals,
                "component_total_cost_usd": sum(totals.values()),
                "evaluated_total_cost_usd": payload["total_cost_usd"],
            }
        )
    write_csv(args.output / "endpoint_cost_components.csv", component_rows, list(component_rows[0]))

    write_json_atomic(
        args.output / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "library_variant": "structural_full",
            "structural_source": str(args.structural_results),
            "penalty_multipliers": list(PENALTY_MULTIPLIERS),
            "holding_multipliers": [0.0, 0.5, 1.0, 2.0],
            "holding_multiplier_1_represented_by_penalty_multiplier_1": True,
            "targets": [asdict(spec) for spec in experiment_specs],
            "endpoint_tasks": len(endpoint_payloads),
            "share_bound_tasks": len(share_payloads),
        },
    )
    print(f"Full structural cost sensitivity written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
