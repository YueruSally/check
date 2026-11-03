#!/usr/bin/env python3
"""Solve exact epsilon-constrained trade-off points on the full path library."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.exact_endpoint import solve_bounded_milp
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
    allocations_from_payload,
    gateway_for_path,
    gateway_shares,
    load_json,
    numerical_tolerance,
    share_bound_payload,
    write_csv,
    write_json_atomic,
)


DESIGNS = (
    ExactCondition(
        case_id="D1",
        condition_id="delay_288h",
        family="epsilon_tradeoff",
        uswc_delay_h=288.0,
    ),
    ExactCondition(
        case_id="D3",
        condition_id="delay_240h",
        family="epsilon_tradeoff",
        uswc_delay_h=240.0,
    ),
)


def endpoint_index(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["case_id"], row["condition_id"], row["gateway_access"], row["endpoint"]): row
        for row in rows
    }


def tradeoff_complete(payload: dict[str, object]) -> bool:
    if int(payload.get("stage_one_proven_infeasible", 0)) == 1:
        return True
    return (
        int(payload.get("tradeoff_proven", 0)) == 1
        and int(payload.get("evaluator_consistent", 0)) == 1
        and payload.get("total_cost_usd") is not None
        and payload.get("quantity_weighted_mean_delivery_time_h") is not None
    )


def solve_tradeoff(
    *,
    model,
    condition: ExactCondition,
    access: str,
    level: int,
    time_cap_h: float,
    time_limit_seconds: float,
    mip_relative_gap: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    cost_coefficients = model.objective_coefficients("cost")
    time_coefficients = model.objective_coefficients("time")
    stage_one = solve_bounded_milp(
        model,
        objective_coefficients=cost_coefficients,
        objective_name="cost_under_time_cap",
        upper_bounds=((time_coefficients, time_cap_h),),
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
    )
    identifier = f"{condition.case_id}_{condition.condition_id}_{access}_epsilon_{level:02d}"
    base: dict[str, object] = {
        "tradeoff_id": identifier,
        "case_id": condition.case_id,
        "condition_id": condition.condition_id,
        "gateway_access": access,
        "epsilon_level": level,
        "time_cap_h": time_cap_h,
        "stage_one_proven": int(stage_one.proven_optimal),
        "stage_one_proven_infeasible": int(stage_one.proven_infeasible),
        "stage_one_call": asdict(stage_one.solver_call),
    }
    if not stage_one.proven_optimal or stage_one.objective_value is None:
        return {
            **base,
            "tradeoff_proven": 0,
            "evaluator_consistent": int(stage_one.evaluator_consistent),
            "cost_optimum_usd": stage_one.objective_value,
            "cost_tolerance_usd": None,
            "time_optimum_h": None,
            "time_tolerance_h": None,
            "total_cost_usd": None,
            "quantity_weighted_mean_delivery_time_h": None,
            "ny_nj_share": None,
            "la_lb_share": None,
            "seattle_tacoma_share": None,
            "stage_two_call": None,
            "allocations": [],
        }

    cost_tolerance = max(
        numerical_tolerance(
            float(stage_one.objective_value), absolute_tolerance, relative_tolerance
        ),
        abs(float(stage_one.evaluator_cost_abs_error or 0.0)),
    )
    cost_bound = float(stage_one.objective_value) + cost_tolerance
    def run_stage_two(
        current_cost_bound: float, current_time_cap: float, *, presolve: bool
    ):
        return solve_bounded_milp(
            model,
            objective_coefficients=time_coefficients,
            objective_name="time_tiebreak",
            upper_bounds=(
                (time_coefficients, current_time_cap),
                (cost_coefficients, current_cost_bound),
            ),
            time_limit_seconds=time_limit_seconds,
            mip_relative_gap=mip_relative_gap,
            presolve=presolve,
        )

    applied_cost_bound = cost_bound
    applied_time_cap = time_cap_h
    stage_two = run_stage_two(cost_bound, time_cap_h, presolve=True)
    stage_two_attempts = [
        {
            "mode": "strict_with_presolve",
            "cost_upper_bound_usd": cost_bound,
            "time_upper_bound_h": time_cap_h,
            **asdict(stage_two.solver_call),
        }
    ]
    if stage_two.proven_infeasible:
        stage_two = run_stage_two(cost_bound, time_cap_h, presolve=False)
        stage_two_attempts.append(
            {
                "mode": "strict_without_presolve",
                "cost_upper_bound_usd": cost_bound,
                "time_upper_bound_h": time_cap_h,
                **asdict(stage_two.solver_call),
            }
        )
    if stage_two.proven_infeasible:
        applied_cost_bound = cost_bound + 1.0
        applied_time_cap = time_cap_h + (1.0 / 3600.0)
        stage_two = run_stage_two(
            applied_cost_bound, applied_time_cap, presolve=False
        )
        stage_two_attempts.append(
            {
                "mode": "one_dollar_one_second_without_presolve",
                "cost_upper_bound_usd": applied_cost_bound,
                "time_upper_bound_h": applied_time_cap,
                **asdict(stage_two.solver_call),
            }
        )
    evaluation = stage_two.evaluation
    shares = gateway_shares(model.data, stage_two.allocations)
    time_tolerance = (
        max(
            numerical_tolerance(
                float(stage_two.objective_value),
                absolute_tolerance,
                relative_tolerance,
            ),
            abs(float(stage_two.evaluator_time_abs_error or 0.0)),
        )
        if stage_two.objective_value is not None
        else None
    )
    return {
        **base,
        "tradeoff_proven": int(stage_one.proven_optimal and stage_two.proven_optimal),
        "evaluator_consistent": int(stage_two.evaluator_consistent),
        "cost_optimum_usd": stage_one.objective_value,
        "cost_tolerance_usd": cost_tolerance,
        "applied_cost_upper_bound_usd": applied_cost_bound,
        "applied_time_cap_h": applied_time_cap,
        "time_optimum_h": stage_two.objective_value,
        "time_tolerance_h": time_tolerance,
        "total_cost_usd": evaluation.total_cost_usd if evaluation else None,
        "quantity_weighted_mean_delivery_time_h": (
            evaluation.quantity_weighted_mean_delivery_time_h if evaluation else None
        ),
        "ny_nj_share": shares["ny_nj"],
        "la_lb_share": shares["la_lb"],
        "seattle_tacoma_share": shares["seattle_tacoma"],
        "stage_two_call": asdict(stage_two.solver_call),
        "stage_two_attempts": stage_two_attempts,
        "allocations": [
            {
                "shipment_id": allocation.shipment_id,
                "quantity_feu": allocation.quantity_feu,
                "gateway": gateway_for_path(model.data, allocation.arc_ids),
                "arc_ids": list(allocation.arc_ids),
            }
            for allocation in stage_two.allocations
        ],
    }


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
        "--output",
        type=Path,
        default=Path("results/v27_full_structural_epsilon_tradeoffs"),
    )
    parser.add_argument("--internal-levels", type=int, default=8)
    parser.add_argument("--enumeration-cap-per-gateway", type=int, default=100_000)
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.internal_levels < 1:
        raise ValueError("--internal-levels must be positive.")

    core_manifest = load_json(args.core_results / "manifest.json")
    if not core_manifest.get("structural_enumeration_complete"):
        raise RuntimeError("The structural path source is not exhaustive.")
    endpoints = endpoint_index(args.core_results / "full_exact_endpoint_summary.csv")
    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    base_cases = {condition.case_id: select_case(data, condition.case_id) for condition in DESIGNS}
    enumerations = {
        case_id: enumerate_case_paths(
            case_data, base_config, args.enumeration_cap_per_gateway
        )
        for case_id, case_data in base_cases.items()
    }
    if any(
        not item.complete
        for by_gateway in enumerations.values()
        for item in by_gateway.values()
    ):
        raise RuntimeError("Structural enumeration cap reached.")

    caps: dict[tuple[str, str], list[float]] = {}
    for condition in DESIGNS:
        minimum_time = float(
            endpoints[
                (
                    condition.case_id,
                    condition.condition_id,
                    "allowed",
                    "min_time_then_cost",
                )
            ]["quantity_weighted_mean_delivery_time_h"]
        )
        cost_endpoint_time = float(
            endpoints[
                (
                    condition.case_id,
                    condition.condition_id,
                    "allowed",
                    "min_cost_then_time",
                )
            ]["quantity_weighted_mean_delivery_time_h"]
        )
        if cost_endpoint_time <= minimum_time:
            raise RuntimeError(
                f"No non-degenerate trade-off interval for {condition.case_id}."
            )
        caps[(condition.case_id, condition.condition_id)] = [
            minimum_time
            + (level / (args.internal_levels + 1))
            * (cost_endpoint_time - minimum_time)
            for level in range(1, args.internal_levels + 1)
        ]

    checkpoint_root = args.output / "checkpoints"
    task_total = len(DESIGNS) * 2 * args.internal_levels
    completed = 0
    for condition in DESIGNS:
        case_data = base_cases[condition.case_id]
        for access in ("allowed", "restricted"):
            config = condition_config(base_config, condition, gateway_access=access)
            model = None
            for level, cap in enumerate(
                caps[(condition.case_id, condition.condition_id)], start=1
            ):
                identifier = (
                    f"{condition.case_id}_{condition.condition_id}_{access}_epsilon_{level:02d}"
                )
                checkpoint = checkpoint_root / "tradeoffs" / f"{identifier}.json"
                if checkpoint.exists() and tradeoff_complete(load_json(checkpoint)):
                    completed += 1
                    print(f"[{completed}/{task_total}] resume {identifier}", flush=True)
                    continue
                if model is None:
                    model, _ = build_full_model(
                        case_data, config, enumerations[condition.case_id]
                    )
                payload = solve_tradeoff(
                    model=model,
                    condition=condition,
                    access=access,
                    level=level,
                    time_cap_h=cap,
                    time_limit_seconds=args.time_limit_seconds,
                    mip_relative_gap=args.mip_relative_gap,
                    relative_tolerance=args.relative_tolerance,
                    absolute_tolerance=args.absolute_tolerance,
                )
                write_json_atomic(checkpoint, payload)
                completed += 1
                print(
                    f"[{completed}/{task_total}] {identifier} "
                    f"proven={payload['tradeoff_proven']} "
                    f"infeasible={payload['stage_one_proven_infeasible']}",
                    flush=True,
                )

    payloads = [
        load_json(path)
        for path in sorted((checkpoint_root / "tradeoffs").glob("*.json"))
    ]
    allowed = [
        row
        for row in payloads
        if row["gateway_access"] == "allowed" and int(row["tradeoff_proven"]) == 1
    ]
    share_total = len(allowed) * 2
    share_completed = 0
    condition_index = {
        (condition.case_id, condition.condition_id): condition for condition in DESIGNS
    }
    models: dict[tuple[str, str], object] = {}
    for point in allowed:
        condition = condition_index[(point["case_id"], point["condition_id"])]
        key = (condition.case_id, condition.condition_id)
        if key not in models:
            config = condition_config(base_config, condition, gateway_access="allowed")
            models[key], _ = build_full_model(
                base_cases[condition.case_id], config, enumerations[condition.case_id]
            )
        model = models[key]
        cost_bound = float(point["applied_cost_upper_bound_usd"])
        time_bound = float(point["time_optimum_h"]) + float(
            point["time_tolerance_h"]
        )
        for direction in ("min", "max"):
            bound_id = f"{point['tradeoff_id']}_{direction}_ny_nj_share"
            checkpoint = checkpoint_root / "share_bounds" / f"{bound_id}.json"
            if checkpoint.exists() and share_bound_complete(load_json(checkpoint)):
                share_completed += 1
                print(
                    f"[share {share_completed}/{share_total}] resume {bound_id}",
                    flush=True,
                )
                continue
            if direction == "min" and float(point["ny_nj_share"]) <= 1e-12:
                certificate_point = {
                    **point,
                    "validation_id": point["tradeoff_id"],
                    "lexicographic_proven": point["tradeoff_proven"],
                }
                bound = zero_share_lower_bound_certificate(
                    bound_id=bound_id,
                    condition=condition,
                    endpoint_region=str(point["tradeoff_id"]),
                    cost_upper_bound=cost_bound,
                    time_upper_bound=time_bound,
                    endpoint=certificate_point,
                )
            else:
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
                bound = share_bound_payload(
                    bound_id=bound_id,
                    condition=condition,
                    endpoint_region=str(point["tradeoff_id"]),
                    direction=direction,
                    cost_upper_bound=applied_cost,
                    time_upper_bound=applied_time,
                    result=result,
                    data=base_cases[condition.case_id],
                )
                bound.update(
                    {
                        "strict_cost_upper_bound_usd": cost_bound,
                        "strict_time_upper_bound_h": time_bound,
                        "share_bound_retry_mode": retry_mode,
                        "solver_attempts": attempts,
                    }
                )
            bound["library_variant"] = "structural_full"
            write_json_atomic(checkpoint, bound)
            share_completed += 1
            print(
                f"[share {share_completed}/{share_total}] {bound_id} "
                f"proven={bound['proven_optimal']} share={bound['ny_nj_share']}",
                flush=True,
            )

    share_payloads = [
        load_json(path)
        for path in sorted((checkpoint_root / "share_bounds").glob("*.json"))
    ]
    unresolved = [row for row in share_payloads if not share_bound_complete(row)]
    if unresolved:
        raise RuntimeError(
            "Unresolved trade-off share bounds: "
            + ", ".join(str(row["bound_id"]) for row in unresolved)
        )

    summary_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "stage_one_call",
                "stage_two_call",
                "stage_two_attempts",
                "allocations",
            }
        }
        for row in payloads
    ]
    write_csv(args.output / "epsilon_tradeoff_summary.csv", summary_rows, list(summary_rows[0]))
    solver_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    for row in payloads:
        if row.get("stage_one_call"):
            solver_rows.append(
                {
                    "tradeoff_id": row["tradeoff_id"],
                    "stage_name": "stage_one_call",
                    "attempt_mode": "strict_with_presolve",
                    **row["stage_one_call"],
                }
            )
        for attempt in row.get("stage_two_attempts", []):
            solver_rows.append(
                {
                    "tradeoff_id": row["tradeoff_id"],
                    "stage_name": "stage_two_call",
                    "attempt_mode": attempt["mode"],
                    **{
                        key: value
                        for key, value in attempt.items()
                        if key
                        not in {
                            "mode",
                            "cost_upper_bound_usd",
                            "time_upper_bound_h",
                        }
                    },
                    "cost_upper_bound_usd": attempt["cost_upper_bound_usd"],
                    "time_upper_bound_h": attempt["time_upper_bound_h"],
                }
            )
        for allocation_id, allocation in enumerate(
            allocations_from_payload(row), start=1
        ):
            allocation_rows.append(
                {
                    "tradeoff_id": row["tradeoff_id"],
                    "allocation_id": allocation_id,
                    "shipment_id": allocation.shipment_id,
                    "quantity_feu": allocation.quantity_feu,
                    "arc_ids": "|".join(allocation.arc_ids),
                }
            )
    solver_fields = list(solver_rows[0])
    for solver_row in solver_rows[1:]:
        for field in solver_row:
            if field not in solver_fields:
                solver_fields.append(field)
    write_csv(
        args.output / "epsilon_tradeoff_solver_calls.csv",
        solver_rows,
        solver_fields,
    )
    write_csv(args.output / "epsilon_tradeoff_allocations.csv", allocation_rows, list(allocation_rows[0]))

    groups: dict[str, dict[str, dict[str, object]]] = {}
    for row in share_payloads:
        point_id = str(row["endpoint_region"])
        groups.setdefault(point_id, {})[str(row["direction"])] = row
    range_rows = []
    by_id = {str(row["tradeoff_id"]): row for row in payloads}
    for point_id, directions in sorted(groups.items()):
        point = by_id[point_id]
        minimum, maximum = directions["min"], directions["max"]
        range_rows.append(
            {
                "tradeoff_id": point_id,
                "case_id": point["case_id"],
                "condition_id": point["condition_id"],
                "epsilon_level": point["epsilon_level"],
                "time_cap_h": point["time_cap_h"],
                "ny_nj_min_share": minimum["ny_nj_share"],
                "ny_nj_max_share": maximum["ny_nj_share"],
                "min_proven": minimum["proven_optimal"],
                "max_proven": maximum["proven_optimal"],
            }
        )
    write_csv(args.output / "epsilon_ny_nj_share_ranges.csv", range_rows, list(range_rows[0]))

    write_json_atomic(
        args.output / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "library_variant": "structural_full",
            "structural_source": str(args.core_results),
            "internal_time_caps_per_condition": args.internal_levels,
            "conditions": [asdict(condition) for condition in DESIGNS],
            "tradeoff_tasks": len(payloads),
            "share_bound_tasks": len(share_payloads),
            "interpretation": (
                "Each internal point minimizes cost subject to a fixed delivery-time "
                "upper bound and then minimizes time within the proven cost optimum."
            ),
            "interpretation_limit": (
                "These are exactly solved epsilon-constrained trade-off solutions, "
                "not a claim that the entire Pareto front has been enumerated."
            ),
        },
    )
    print(f"Exact epsilon trade-off solutions written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
