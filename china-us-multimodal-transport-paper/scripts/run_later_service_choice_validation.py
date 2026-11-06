#!/usr/bin/env python3
"""Validate the earliest-reachable-service assumption with exact endpoints.

The historical rule fixes every scheduled arc to departure rank zero.  The
expanded rule makes ranks zero, one, and two available as separate itinerary
choices on every scheduled leg, without adding services or capacity.  Proven
historical endpoints are copied as immutable rank-zero controls; only the
expanded-rule endpoints are newly solved.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.exact_endpoint import solve_lexicographic_endpoint
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import ExactCondition, condition_config
from run_final_path_coverage import build_full_model, enumerate_case_paths
from run_regional_inland_capacity import target_arcs
from run_supplementary_core_exact import (
    ENDPOINTS,
    allocations_from_payload,
    endpoint_payload,
    gateway_for_path,
    load_json,
    service_leg_rows,
    write_csv,
    write_json_atomic,
)


ACCESS_LEVELS = ("allowed", "restricted")


@dataclass(frozen=True)
class ValidationSpec:
    condition: ExactCondition
    current_source: str
    regional_inland_capacity_multiplier: float | None = None


def validation_specs() -> tuple[ValidationSpec, ...]:
    return (
        ValidationSpec(
            ExactCondition("D1", "delay_288h", "regional_delay", uswc_delay_h=288.0),
            "core",
        ),
        ValidationSpec(
            ExactCondition("D2", "delay_288h", "regional_delay", uswc_delay_h=288.0),
            "core",
        ),
        ValidationSpec(
            ExactCondition("D3", "delay_216h", "regional_delay", uswc_delay_h=216.0),
            "inland_interaction",
        ),
        ValidationSpec(
            ExactCondition("D3", "delay_240h", "regional_delay", uswc_delay_h=240.0),
            "core",
        ),
        *(
            ValidationSpec(
                ExactCondition(
                    case_id,
                    "capacity_35pct",
                    "gateway_node_capacity",
                    uswc_capacity_multiplier=0.35,
                ),
                "core",
            )
            for case_id in ("D1", "D2", "D3")
        ),
        ValidationSpec(
            ExactCondition(
                "D3",
                "delay_168h_regional_inland_capacity_030pct",
                "regional_inland_capacity",
                uswc_delay_h=168.0,
            ),
            "regional_inland",
            regional_inland_capacity_multiplier=0.30,
        ),
    )


def checkpoint_id(
    spec: ValidationSpec, access: str, endpoint: str, rule: str
) -> str:
    return (
        f"{spec.condition.case_id}_{spec.condition.condition_id}_{access}_"
        f"{endpoint}_{rule}"
    )


def historical_checkpoint(
    spec: ValidationSpec,
    access: str,
    endpoint: str,
    *,
    core_root: Path,
    inland_root: Path,
    regional_root: Path,
) -> Path:
    case_id = spec.condition.case_id
    if spec.current_source == "core":
        name = (
            f"{case_id}_{spec.condition.condition_id}_{access}_{endpoint}_"
            "structural_full.json"
        )
        return core_root / "checkpoints" / "endpoints" / name
    if spec.current_source == "inland_interaction":
        name = (
            f"D3_delay_216h_inland_us-d-03_capacity_100pct_{access}_{endpoint}.json"
        )
        return inland_root / "checkpoints" / "endpoints" / name
    if spec.current_source == "regional_inland":
        name = (
            "D3_delay_168h_regional_inland_capacity_030pct_"
            f"{access}_{endpoint}.json"
        )
        return regional_root / "checkpoints" / "endpoints" / name
    raise ValueError(f"Unknown current endpoint source: {spec.current_source!r}.")


def seed_current_rule(
    args: argparse.Namespace,
    specs: tuple[ValidationSpec, ...],
) -> None:
    root = args.output / "checkpoints" / "endpoints"
    for spec in specs:
        for access in ACCESS_LEVELS:
            for endpoint, _ in ENDPOINTS:
                validation_id = checkpoint_id(
                    spec, access, endpoint, "earliest_only"
                )
                target = root / f"{validation_id}.json"
                if target.exists():
                    payload = load_json(target)
                    if int(payload.get("lexicographic_proven", 0)) == 1:
                        continue
                source = historical_checkpoint(
                    spec,
                    access,
                    endpoint,
                    core_root=args.core_results,
                    inland_root=args.inland_results,
                    regional_root=args.regional_inland_results,
                )
                if not source.exists():
                    raise FileNotFoundError(
                        f"Missing proven current-rule endpoint: {source}"
                    )
                payload = load_json(source)
                if int(payload.get("lexicographic_proven", 0)) != 1:
                    raise RuntimeError(f"Current-rule endpoint is not proven: {source}")
                if int(payload.get("evaluator_consistent", 0)) != 1:
                    raise RuntimeError(
                        f"Current-rule endpoint is not evaluator-consistent: {source}"
                    )
                payload.update(
                    {
                        "validation_id": validation_id,
                        "condition_id": spec.condition.condition_id,
                        "family": spec.condition.family,
                        "service_choice_rule": "earliest_only",
                        "max_extra_service_departures": 0,
                        "source_validation_id": payload.get("validation_id", ""),
                        "reused_from": str(source),
                    }
                )
                for allocation in payload.get("allocations", []):
                    allocation.setdefault(
                        "service_departure_ranks",
                        [0] * len(allocation.get("arc_ids", [])),
                    )
                write_json_atomic(target, payload)


def adjusted_data(base_case, spec: ValidationSpec):
    multiplier = spec.regional_inland_capacity_multiplier
    if multiplier is None:
        return base_case
    arc_ids = target_arcs(base_case)
    arcs = dict(base_case.arcs)
    for arc_id in arc_ids:
        base_capacity = base_case.arcs[arc_id].capacity_feu_per_day
        if base_capacity is None:
            raise RuntimeError(f"Target inland arc {arc_id} has no daily capacity.")
        arcs[arc_id] = replace(
            base_case.arcs[arc_id],
            capacity_feu_per_day=float(base_capacity) * multiplier,
        )
    return replace(base_case, arcs=arcs)


def run_expanded_endpoints(
    args: argparse.Namespace,
    data,
    base_config,
    specs: tuple[ValidationSpec, ...],
    enumerations,
) -> None:
    root = args.output / "checkpoints" / "endpoints"
    total = len(specs) * len(ACCESS_LEVELS) * len(ENDPOINTS)
    completed = 0
    for spec in specs:
        base_case = select_case(data, spec.condition.case_id)
        case_data = adjusted_data(base_case, spec)
        for access in ACCESS_LEVELS:
            config = condition_config(
                base_config,
                spec.condition,
                gateway_access=access,
            )
            model = None
            max_paths = None
            for endpoint, primary in ENDPOINTS:
                validation_id = checkpoint_id(
                    spec, access, endpoint, "earliest_plus_two"
                )
                checkpoint = root / f"{validation_id}.json"
                if checkpoint.exists():
                    existing = load_json(checkpoint)
                    if int(existing.get("lexicographic_proven", 0)) == 1:
                        completed += 1
                        print(f"[{completed}/{total}] resume {validation_id}", flush=True)
                        continue
                if model is None:
                    model, max_paths = build_full_model(
                        case_data,
                        config,
                        enumerations[spec.condition.case_id],
                        max_extra_service_departures=2,
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
                    condition=spec.condition,
                    gateway_access=access,
                    endpoint=endpoint,
                    model=model,
                    result=result,
                    max_paths_per_gateway=int(max_paths),
                )
                payload.update(
                    {
                        "validation_id": validation_id,
                        "library_variant": "structural_full_service_expanded",
                        "service_choice_rule": "earliest_plus_two",
                        "max_extra_service_departures": 2,
                        "candidate_itineraries": model.quantity_variable_count,
                        "regional_inland_capacity_multiplier": (
                            spec.regional_inland_capacity_multiplier
                        ),
                    }
                )
                write_json_atomic(checkpoint, payload)
                completed += 1
                print(
                    f"[{completed}/{total}] {validation_id} "
                    f"proven={payload['lexicographic_proven']} "
                    f"itineraries={payload['candidate_itineraries']} "
                    f"runtime={payload['solver_runtime_seconds']:.2f}s",
                    flush=True,
                )


def clean_payload_row(payload: dict[str, object]) -> dict[str, object]:
    excluded = {"solver_calls", "allocations"}
    return {key: value for key, value in payload.items() if key not in excluded}


def endpoint_index(payloads):
    return {
        (
            row["case_id"],
            row["condition_id"],
            row["gateway_access"],
            row["endpoint"],
            row["service_choice_rule"],
        ): row
        for row in payloads
    }


def finalize(
    args: argparse.Namespace,
    data,
    specs: tuple[ValidationSpec, ...],
) -> None:
    payloads = [
        load_json(path)
        for path in sorted(
            (args.output / "checkpoints" / "endpoints").glob("*.json")
        )
    ]
    expected = len(specs) * len(ACCESS_LEVELS) * len(ENDPOINTS) * 2
    if len(payloads) != expected:
        raise RuntimeError(f"Expected {expected} endpoint records, found {len(payloads)}.")
    unresolved = [
        row
        for row in payloads
        if int(row.get("lexicographic_proven", 0)) != 1
        or int(row.get("evaluator_consistent", 0)) != 1
    ]
    if unresolved:
        write_csv(
            args.output / "unresolved_endpoints.csv",
            [clean_payload_row(row) for row in unresolved],
            list(clean_payload_row(unresolved[0])),
        )
        raise RuntimeError(
            f"{len(unresolved)} endpoint(s) are not proven and evaluator-consistent."
        )

    summary_rows = [clean_payload_row(row) for row in payloads]
    fields = sorted({key for row in summary_rows for key in row})
    write_csv(args.output / "endpoint_summary.csv", summary_rows, fields)

    solver_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    service_rows: list[dict[str, object]] = []
    for payload in payloads:
        for call in payload.get("solver_calls", []):
            solver_rows.append({"validation_id": payload["validation_id"], **call})
        case_data = adjusted_data(
            select_case(data, str(payload["case_id"])),
            next(
                spec
                for spec in specs
                if spec.condition.case_id == payload["case_id"]
                and spec.condition.condition_id == payload["condition_id"]
            ),
        )
        allocations = allocations_from_payload(payload)
        for allocation_id, allocation in enumerate(allocations, start=1):
            allocation_rows.append(
                {
                    "validation_id": payload["validation_id"],
                    "allocation_id": allocation_id,
                    "shipment_id": allocation.shipment_id,
                    "quantity_feu": allocation.quantity_feu,
                    "gateway": gateway_for_path(case_data, allocation.arc_ids),
                    "arc_ids": "|".join(allocation.arc_ids),
                    "service_departure_ranks": "|".join(
                        str(item) for item in allocation.service_departure_ranks
                    ),
                }
            )
        config = condition_config(
            load_config(args.config, "pilot_baseline"),
            next(
                spec.condition
                for spec in specs
                if spec.condition.case_id == payload["case_id"]
                and spec.condition.condition_id == payload["condition_id"]
            ),
            gateway_access=str(payload["gateway_access"]),
        )
        for row in service_leg_rows(
            case_data,
            allocations,
            config,
            str(payload["validation_id"]),
        ):
            service_rows.append(row)

    write_csv(
        args.output / "solver_calls.csv",
        solver_rows,
        list(solver_rows[0]),
    )
    write_csv(
        args.output / "endpoint_allocations.csv",
        allocation_rows,
        list(allocation_rows[0]),
    )
    write_csv(
        args.output / "endpoint_service_legs.csv",
        service_rows,
        list(service_rows[0]),
    )

    index = endpoint_index(payloads)
    comparison_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []
    for spec in specs:
        key_prefix = (spec.condition.case_id, spec.condition.condition_id)
        for access in ACCESS_LEVELS:
            for endpoint, _ in ENDPOINTS:
                current = index[(*key_prefix, access, endpoint, "earliest_only")]
                expanded = index[
                    (*key_prefix, access, endpoint, "earliest_plus_two")
                ]
                comparison_rows.append(
                    {
                        "case_id": key_prefix[0],
                        "condition_id": key_prefix[1],
                        "gateway_access": access,
                        "endpoint": endpoint,
                        "current_cost_usd": current["total_cost_usd"],
                        "expanded_cost_usd": expanded["total_cost_usd"],
                        "expanded_minus_current_cost_usd": (
                            float(expanded["total_cost_usd"])
                            - float(current["total_cost_usd"])
                        ),
                        "current_time_h": current[
                            "quantity_weighted_mean_delivery_time_h"
                        ],
                        "expanded_time_h": expanded[
                            "quantity_weighted_mean_delivery_time_h"
                        ],
                        "expanded_minus_current_time_h": (
                            float(expanded["quantity_weighted_mean_delivery_time_h"])
                            - float(current["quantity_weighted_mean_delivery_time_h"])
                        ),
                        "current_ny_nj_share": current["ny_nj_share"],
                        "expanded_ny_nj_share": expanded["ny_nj_share"],
                        "ny_nj_share_change": (
                            float(expanded["ny_nj_share"])
                            - float(current["ny_nj_share"])
                        ),
                        "current_candidate_itineraries": current.get(
                            "candidate_itineraries", current.get("candidate_paths")
                        ),
                        "expanded_candidate_itineraries": expanded.get(
                            "candidate_itineraries", expanded.get("candidate_paths")
                        ),
                        "expanded_solver_runtime_seconds": expanded[
                            "solver_runtime_seconds"
                        ],
                        "expanded_maximum_relative_gap": expanded[
                            "maximum_relative_gap"
                        ],
                    }
                )
        for rule in ("earliest_only", "earliest_plus_two"):
            cost_allowed = index[
                (*key_prefix, "allowed", "min_cost_then_time", rule)
            ]
            cost_without = index[
                (*key_prefix, "restricted", "min_cost_then_time", rule)
            ]
            time_allowed = index[
                (*key_prefix, "allowed", "min_time_then_cost", rule)
            ]
            time_without = index[
                (*key_prefix, "restricted", "min_time_then_cost", rule)
            ]
            value_rows.append(
                {
                    "case_id": key_prefix[0],
                    "condition_id": key_prefix[1],
                    "service_choice_rule": rule,
                    "minimum_cost_gateway_value_usd": (
                        float(cost_without["total_cost_usd"])
                        - float(cost_allowed["total_cost_usd"])
                    ),
                    "minimum_time_gateway_value_h": (
                        float(
                            time_without["quantity_weighted_mean_delivery_time_h"]
                        )
                        - float(
                            time_allowed["quantity_weighted_mean_delivery_time_h"]
                        )
                    ),
                    "minimum_cost_endpoint_ny_nj_share": cost_allowed[
                        "ny_nj_share"
                    ],
                    "minimum_time_endpoint_ny_nj_share": time_allowed[
                        "ny_nj_share"
                    ],
                }
            )
    write_csv(
        args.output / "current_vs_later_service_endpoints.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )
    write_csv(
        args.output / "gateway_value_by_service_choice_rule.csv",
        value_rows,
        list(value_rows[0]),
    )

    write_json_atomic(
        args.output / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "later_service_choice_validation",
            "conditions": [asdict(spec) for spec in specs],
            "historical_rule": "earliest feasible scheduled departure only",
            "expanded_rule": (
                "each scheduled leg may choose its earliest feasible departure "
                "or either of the next two feasible departure events"
            ),
            "new_expanded_endpoint_tasks": len(specs)
            * len(ACCESS_LEVELS)
            * len(ENDPOINTS),
            "historical_endpoint_tasks_reused": len(specs)
            * len(ACCESS_LEVELS)
            * len(ENDPOINTS),
            "structural_path_set": "complete under the registered restrictions",
            "interpretation_limit": (
                "The experiment validates two additional departure events per "
                "scheduled leg; it does not enumerate every later departure."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--core-results",
        type=Path,
        default=Path("results/v21_full_structural_core_exact"),
    )
    parser.add_argument(
        "--inland-results",
        type=Path,
        default=Path("results/v26_full_structural_inland_bottleneck"),
    )
    parser.add_argument(
        "--regional-inland-results",
        type=Path,
        default=Path("results/v28_regional_inland_capacity"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v31_later_service_choice_validation"),
    )
    parser.add_argument("--enumeration-cap-per-gateway", type=int, default=100_000)
    parser.add_argument("--time-limit-seconds", type=float, default=1800.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    specs = validation_specs()
    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    case_data = {
        case_id: select_case(data, case_id) for case_id in ("D1", "D2", "D3")
    }
    enumerations = {
        case_id: enumerate_case_paths(
            selected,
            base_config,
            args.enumeration_cap_per_gateway,
        )
        for case_id, selected in case_data.items()
    }
    if any(
        not enumeration.complete
        for by_gateway in enumerations.values()
        for enumeration in by_gateway.values()
    ):
        raise RuntimeError("Structural enumeration cap reached.")

    seed_current_rule(args, specs)
    run_expanded_endpoints(args, data, base_config, specs, enumerations)
    finalize(args, data, specs)
    print(f"Later-service choice validation written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
