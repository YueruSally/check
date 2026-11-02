#!/usr/bin/env python3
"""Exact robustness experiment for ten pre-registered demand instances."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import random
from statistics import median

from china_us_multimodal.config import load_config
from china_us_multimodal.domain import ModelData
from china_us_multimodal.exact_endpoint import (
    build_exact_endpoint_model,
    solve_lexicographic_endpoint,
)
from china_us_multimodal.io import load_model_data, select_case
from china_us_multimodal.supplementary_design import (
    CASES,
    condition_config,
    demand_robustness_conditions,
)
from run_supplementary_core_exact import (
    ENDPOINTS,
    allocations_from_payload,
    endpoint_payload,
    gateway_for_path,
    load_json,
    write_csv,
    write_json_atomic,
)


INSTANCE_SEEDS = tuple(range(20261101, 20261111))
CASE_SEED_OFFSETS = {"D1": 101, "D2": 202, "D3": 303}
QUANTITY_LOG_SD = 0.18
QUANTITY_FACTOR_CLIP = (0.875, 1.125)
FINAL_QUANTITY_RATIO_CHECK = (0.75, 1.30)
DUE_JITTER_CHOICES_H = (-48, -36, -24, -12, 0, 12, 24, 36, 48)
MINIMUM_DUE_ALLOWANCE_H = 24.0
MATERIAL_COST_USD = 10_000.0
MATERIAL_COST_RELATIVE = 0.0001
MATERIAL_TIME_H = 0.25
NUMERICAL_COST_USD = 1.0
NUMERICAL_TIME_H = 1e-5
ACTIVATION_SHARE = 0.10


def generated_case_data(
    base_case: ModelData,
    *,
    case_id: str,
    instance_index: int,
    seed: int,
) -> tuple[ModelData, list[dict[str, object]]]:
    rng = random.Random(seed + CASE_SEED_OFFSETS[case_id])
    shipment_ids = sorted(base_case.shipments)
    base_quantities = [base_case.shipments[item].quantity_feu for item in shipment_ids]
    factors = [
        min(
            QUANTITY_FACTOR_CLIP[1],
            max(
                QUANTITY_FACTOR_CLIP[0],
                math.exp(rng.gauss(0.0, QUANTITY_LOG_SD)),
            ),
        )
        for _ in shipment_ids
    ]
    raw = [quantity * factor for quantity, factor in zip(base_quantities, factors)]
    total = sum(base_quantities)
    scale = total / sum(raw)
    quantities = [value * scale for value in raw]
    residual = total - sum(quantities)
    largest = max(range(len(quantities)), key=quantities.__getitem__)
    quantities[largest] += residual
    ratios = [value / base for value, base in zip(quantities, base_quantities)]
    if min(ratios) < FINAL_QUANTITY_RATIO_CHECK[0] - 1e-12 or max(ratios) > (
        FINAL_QUANTITY_RATIO_CHECK[1] + 1e-12
    ):
        raise RuntimeError(
            f"Generated quantity ratios left the pre-registered range for {case_id}."
        )

    raw_jitters = [float(rng.choice(DUE_JITTER_CHOICES_H)) for _ in shipment_ids]
    centered_mean = sum(raw_jitters) / len(raw_jitters)
    centered_jitters = [value - centered_mean for value in raw_jitters]
    shipments = dict(base_case.shipments)
    manifest_rows: list[dict[str, object]] = []
    for shipment_id, quantity, factor, jitter in zip(
        shipment_ids, quantities, ratios, centered_jitters
    ):
        base = base_case.shipments[shipment_id]
        if base.due_h is None:
            due_h = None
            applied_jitter = 0.0
        else:
            allowance = base.due_h - base.release_h
            new_allowance = max(MINIMUM_DUE_ALLOWANCE_H, allowance + jitter)
            due_h = base.release_h + new_allowance
            applied_jitter = new_allowance - allowance
        shipments[shipment_id] = replace(base, quantity_feu=quantity, due_h=due_h)
        manifest_rows.append(
            {
                "demand_instance_id": f"demand_{instance_index:02d}",
                "generator_seed": seed,
                "case_id": case_id,
                "shipment_id": shipment_id,
                "base_quantity_feu": base.quantity_feu,
                "generated_quantity_feu": quantity,
                "quantity_ratio": factor,
                "release_h": base.release_h,
                "base_due_h": base.due_h,
                "generated_due_h": due_h,
                "due_jitter_h": applied_jitter,
                "tardiness_usd_per_feu_h": base.tardiness_usd_per_feu_h,
            }
        )
    generated = replace(base_case, shipments=shipments)
    if len(generated.shipments) != len(base_case.shipments):
        raise RuntimeError("Demand generation changed the batch count.")
    if not math.isclose(
        sum(item.quantity_feu for item in generated.shipments.values()),
        total,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Demand generation changed total quantity.")
    if any(
        generated.shipments[item].release_h != base_case.shipments[item].release_h
        for item in shipment_ids
    ):
        raise RuntimeError("Demand generation changed a release time.")
    return generated, manifest_rows


def material_cost_threshold(reference_cost: float) -> float:
    return max(MATERIAL_COST_USD, MATERIAL_COST_RELATIVE * abs(reference_cost))


def classify_performance(
    min_cost_allowed: dict[str, object],
    min_cost_restricted: dict[str, object],
    min_time_allowed: dict[str, object],
    min_time_restricted: dict[str, object],
) -> tuple[str, dict[str, object]]:
    cost_threshold = material_cost_threshold(
        float(min_cost_restricted["total_cost_usd"])
    )
    cost_endpoint_cost_benefit = (
        float(min_cost_restricted["total_cost_usd"])
        - float(min_cost_allowed["total_cost_usd"])
    )
    cost_endpoint_time_benefit = (
        float(min_cost_restricted["quantity_weighted_mean_delivery_time_h"])
        - float(min_cost_allowed["quantity_weighted_mean_delivery_time_h"])
    )
    time_endpoint_cost_benefit = (
        float(min_time_restricted["total_cost_usd"])
        - float(min_time_allowed["total_cost_usd"])
    )
    time_endpoint_time_benefit = (
        float(min_time_restricted["quantity_weighted_mean_delivery_time_h"])
        - float(min_time_allowed["quantity_weighted_mean_delivery_time_h"])
    )
    same_endpoint_joint = (
        cost_endpoint_cost_benefit >= cost_threshold
        and cost_endpoint_time_benefit >= MATERIAL_TIME_H
    ) or (
        time_endpoint_cost_benefit >= cost_threshold
        and time_endpoint_time_benefit >= MATERIAL_TIME_H
    )
    cost_primary_material = cost_endpoint_cost_benefit >= cost_threshold
    time_primary_material = time_endpoint_time_benefit >= MATERIAL_TIME_H
    paid_speed = (
        time_endpoint_time_benefit >= MATERIAL_TIME_H
        and time_endpoint_cost_benefit <= -cost_threshold
    )
    if same_endpoint_joint:
        performance_type = "simultaneously_lower_cost_and_faster"
    elif paid_speed:
        performance_type = "paid_speed"
    elif cost_primary_material and time_primary_material:
        performance_type = "separate_endpoint_benefits"
    elif cost_primary_material:
        performance_type = "cost_only"
    elif time_primary_material:
        performance_type = "time_only"
    else:
        performance_type = "no_material_improvement"
    return performance_type, {
        "material_cost_threshold_usd": cost_threshold,
        "material_time_threshold_h": MATERIAL_TIME_H,
        "min_cost_endpoint_cost_benefit_usd": cost_endpoint_cost_benefit,
        "min_cost_endpoint_time_benefit_h": cost_endpoint_time_benefit,
        "min_time_endpoint_cost_benefit_usd": time_endpoint_cost_benefit,
        "min_time_endpoint_time_benefit_h": time_endpoint_time_benefit,
    }


def share_state(value: float) -> str:
    if value <= 1e-9:
        return "not_used"
    if value < ACTIVATION_SHARE - 1e-9:
        return "used_below_10pct"
    return "used_at_least_10pct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact endpoints for ten fixed alternative demand instances."
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/v16_demand_robustness")
    )
    parser.add_argument("--time-limit-seconds", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-paths-per-gateway", type=int, default=4)
    args = parser.parse_args()
    if args.max_paths_per_gateway <= 0:
        parser.error("--max-paths-per-gateway must be positive")

    data = load_model_data(args.data)
    base_config = load_config(args.config, "pilot_baseline")
    base_cases = {case_id: select_case(data, case_id) for case_id in CASES}
    conditions = demand_robustness_conditions()
    conditions_by_case = {
        case_id: tuple(item for item in conditions if item.case_id == case_id)
        for case_id in CASES
    }
    checkpoints = args.output / "checkpoints"
    generated_cases: dict[tuple[str, str], ModelData] = {}
    demand_manifest_rows: list[dict[str, object]] = []
    for instance_index, seed in enumerate(INSTANCE_SEEDS, start=1):
        instance_id = f"demand_{instance_index:02d}"
        for case_id in CASES:
            generated, rows = generated_case_data(
                base_cases[case_id],
                case_id=case_id,
                instance_index=instance_index,
                seed=seed,
            )
            generated_cases[(instance_id, case_id)] = generated
            demand_manifest_rows.extend(rows)
    write_csv(
        args.output / "demand_instance_shipments.csv",
        demand_manifest_rows,
        list(demand_manifest_rows[0].keys()),
    )

    total = len(INSTANCE_SEEDS) * len(conditions) * 2 * len(ENDPOINTS)
    completed = 0
    for instance_index, seed in enumerate(INSTANCE_SEEDS, start=1):
        instance_id = f"demand_{instance_index:02d}"
        for case_id in CASES:
            case_data = generated_cases[(instance_id, case_id)]
            for condition in conditions_by_case[case_id]:
                for access in ("allowed", "restricted"):
                    config = condition_config(
                        base_config, condition, gateway_access=access
                    )
                    model = None
                    for endpoint, primary in ENDPOINTS:
                        task_id = (
                            f"{instance_id}_{case_id}_{condition.condition_id}_"
                            f"{access}_{endpoint}"
                        )
                        checkpoint = checkpoints / f"{task_id}.json"
                        if checkpoint.exists():
                            completed += 1
                            print(f"[{completed}/{total}] resume {task_id}", flush=True)
                            continue
                        if model is None:
                            model = build_exact_endpoint_model(
                                case_data,
                                config,
                                max_paths_per_gateway=args.max_paths_per_gateway,
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
                            max_paths_per_gateway=args.max_paths_per_gateway,
                        )
                        payload["validation_id"] = task_id
                        payload["demand_instance_id"] = instance_id
                        payload["generator_seed"] = seed
                        write_json_atomic(checkpoint, payload)
                        completed += 1
                        print(
                            f"[{completed}/{total}] {task_id} "
                            f"proven={payload['lexicographic_proven']}",
                            flush=True,
                        )

    payloads = [load_json(path) for path in sorted(checkpoints.glob("*.json"))]
    payload_index = {
        (
            row["demand_instance_id"],
            row["case_id"],
            row["condition_id"],
            row["gateway_access"],
            row["endpoint"],
        ): row
        for row in payloads
    }
    summary_rows = [
        {key: value for key, value in row.items() if key not in {"solver_calls", "allocations"}}
        for row in payloads
    ]
    summary_fields = [
        "validation_id", "demand_instance_id", "generator_seed", "case_id",
        "condition_id", "family", "gateway_access", "scenario",
        "uswc_capacity_multiplier", "uswc_delay_h", "endpoint",
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
    write_csv(
        args.output / "demand_exact_endpoint_summary.csv", summary_rows, summary_fields
    )
    allocation_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    for payload in payloads:
        case_data = generated_cases[
            (str(payload["demand_instance_id"]), str(payload["case_id"]))
        ]
        for call in payload["solver_calls"]:
            solver_rows.append({"validation_id": payload["validation_id"], **call})
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
        args.output / "demand_exact_endpoint_allocations.csv",
        allocation_rows,
        [
            "validation_id", "allocation_id", "shipment_id", "quantity_feu",
            "gateway", "arc_ids",
        ],
    )
    write_csv(
        args.output / "demand_solver_calls.csv",
        solver_rows,
        [
            "validation_id", "stage", "objective", "status_code", "status",
            "success", "objective_value", "dual_bound", "relative_gap",
            "node_count", "elapsed_seconds", "message",
        ],
    )

    classification_rows: list[dict[str, object]] = []
    for instance_index, seed in enumerate(INSTANCE_SEEDS, start=1):
        instance_id = f"demand_{instance_index:02d}"
        for condition in conditions:
            rows = {
                (access, endpoint): payload_index[
                    (
                        instance_id,
                        condition.case_id,
                        condition.condition_id,
                        access,
                        endpoint,
                    )
                ]
                for access in ("allowed", "restricted")
                for endpoint, _ in ENDPOINTS
            }
            all_proven = all(
                int(row["lexicographic_proven"]) == 1 for row in rows.values()
            )
            if all_proven:
                performance_type, metrics = classify_performance(
                    rows[("allowed", "min_cost_then_time")],
                    rows[("restricted", "min_cost_then_time")],
                    rows[("allowed", "min_time_then_cost")],
                    rows[("restricted", "min_time_then_cost")],
                )
            else:
                performance_type = "unclassified_unproven_endpoint"
                metrics = {
                    "material_cost_threshold_usd": "",
                    "material_time_threshold_h": MATERIAL_TIME_H,
                    "min_cost_endpoint_cost_benefit_usd": "",
                    "min_cost_endpoint_time_benefit_h": "",
                    "min_time_endpoint_cost_benefit_usd": "",
                    "min_time_endpoint_time_benefit_h": "",
                }
            min_cost_share = float(rows[("allowed", "min_cost_then_time")]["ny_nj_share"])
            min_time_share = float(rows[("allowed", "min_time_then_cost")]["ny_nj_share"])
            classification_rows.append(
                {
                    "demand_instance_id": instance_id,
                    "generator_seed": seed,
                    "case_id": condition.case_id,
                    "condition_id": condition.condition_id,
                    "all_four_endpoints_proven": int(all_proven),
                    "performance_type": performance_type,
                    **metrics,
                    "minimum_cost_numerically_improved": int(
                        all_proven
                        and float(metrics["min_cost_endpoint_cost_benefit_usd"])
                        > NUMERICAL_COST_USD
                    ),
                    "minimum_time_numerically_improved": int(
                        all_proven
                        and float(metrics["min_time_endpoint_time_benefit_h"])
                        > NUMERICAL_TIME_H
                    ),
                    "min_cost_endpoint_ny_nj_share": min_cost_share,
                    "min_time_endpoint_ny_nj_share": min_time_share,
                    "min_cost_endpoint_usage_state": share_state(min_cost_share),
                    "min_time_endpoint_usage_state": share_state(min_time_share),
                    "any_endpoint_uses_ny_nj": int(
                        min_cost_share > 1e-9 or min_time_share > 1e-9
                    ),
                    "any_endpoint_reaches_10pct": int(
                        min_cost_share >= ACTIVATION_SHARE - 1e-9
                        or min_time_share >= ACTIVATION_SHARE - 1e-9
                    ),
                }
            )
    write_csv(
        args.output / "demand_robustness_classification.csv",
        classification_rows,
        list(classification_rows[0].keys()),
    )
    count_groups: dict[tuple[str, str, str], int] = {}
    for row in classification_rows:
        key = (str(row["case_id"]), str(row["condition_id"]), str(row["performance_type"]))
        count_groups[key] = count_groups.get(key, 0) + 1
    count_rows = [
        {
            "case_id": case_id,
            "condition_id": condition_id,
            "performance_type": performance_type,
            "instance_count": count,
            "total_instances": len(INSTANCE_SEEDS),
        }
        for (case_id, condition_id, performance_type), count in sorted(count_groups.items())
    ]
    write_csv(
        args.output / "demand_robustness_type_counts.csv",
        count_rows,
        list(count_rows[0].keys()),
    )
    usage_groups: dict[tuple[str, str, str, str], int] = {}
    for row in classification_rows:
        key = (
            str(row["case_id"]),
            str(row["condition_id"]),
            str(row["min_cost_endpoint_usage_state"]),
            str(row["min_time_endpoint_usage_state"]),
        )
        usage_groups[key] = usage_groups.get(key, 0) + 1
    usage_count_rows = [
        {
            "case_id": case_id,
            "condition_id": condition_id,
            "min_cost_endpoint_usage_state": cost_state,
            "min_time_endpoint_usage_state": time_state,
            "instance_count": count,
            "total_instances": len(INSTANCE_SEEDS),
        }
        for (case_id, condition_id, cost_state, time_state), count in sorted(
            usage_groups.items()
        )
    ]
    write_csv(
        args.output / "demand_robustness_usage_counts.csv",
        usage_count_rows,
        list(usage_count_rows[0].keys()),
    )
    distribution_rows: list[dict[str, object]] = []
    for condition in conditions:
        selected = [
            row
            for row in classification_rows
            if row["case_id"] == condition.case_id
            and row["condition_id"] == condition.condition_id
            and int(row["all_four_endpoints_proven"]) == 1
        ]
        cost_values = [
            float(row["min_cost_endpoint_cost_benefit_usd"]) for row in selected
        ]
        time_values = [
            float(row["min_time_endpoint_time_benefit_h"]) for row in selected
        ]
        distribution_rows.append(
            {
                "case_id": condition.case_id,
                "condition_id": condition.condition_id,
                "proven_instance_count": len(selected),
                "minimum_cost_benefit_min_usd": min(cost_values) if cost_values else "",
                "minimum_cost_benefit_median_usd": median(cost_values)
                if cost_values
                else "",
                "minimum_cost_benefit_max_usd": max(cost_values) if cost_values else "",
                "minimum_time_benefit_min_h": min(time_values) if time_values else "",
                "minimum_time_benefit_median_h": median(time_values)
                if time_values
                else "",
                "minimum_time_benefit_max_h": max(time_values) if time_values else "",
                "interpretation_limit": "constructed-instance range, not probability",
            }
        )
    write_csv(
        args.output / "demand_robustness_benefit_distribution.csv",
        distribution_rows,
        list(distribution_rows[0].keys()),
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "instance_seeds": INSTANCE_SEEDS,
            "case_seed_offsets": CASE_SEED_OFFSETS,
            "quantity_log_sd": QUANTITY_LOG_SD,
            "initial_quantity_factor_clip": QUANTITY_FACTOR_CLIP,
            "final_quantity_ratio_check": FINAL_QUANTITY_RATIO_CHECK,
            "due_jitter_choices_h": DUE_JITTER_CHOICES_H,
            "jitter_centering": "subtract within-case instance mean",
            "minimum_due_allowance_h": MINIMUM_DUE_ALLOWANCE_H,
            "total_quantity_held_fixed": True,
            "batch_count_held_fixed": True,
            "release_times_held_fixed": True,
            "tardiness_rates_held_fixed": True,
        },
        "classification_fixed_before_results": {
            "material_cost_usd": MATERIAL_COST_USD,
            "material_cost_relative": MATERIAL_COST_RELATIVE,
            "material_time_h": MATERIAL_TIME_H,
            "numerical_cost_usd": NUMERICAL_COST_USD,
            "numerical_time_h": NUMERICAL_TIME_H,
            "activation_share": ACTIVATION_SHARE,
            "performance_types": [
                "simultaneously_lower_cost_and_faster",
                "paid_speed",
                "separate_endpoint_benefits",
                "cost_only",
                "time_only",
                "no_material_improvement",
                "unclassified_unproven_endpoint",
            ],
        },
        "conditions": [asdict(condition) for condition in conditions],
        "endpoint_tasks": total,
        "max_paths_per_gateway": args.max_paths_per_gateway,
        "interpretation_limit": (
            "Counts describe robustness across constructed instances and must not "
            "be interpreted as real-world probabilities."
        ),
    }
    write_json_atomic(args.output / "manifest.json", manifest)
    print(f"Demand-instance robustness written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
