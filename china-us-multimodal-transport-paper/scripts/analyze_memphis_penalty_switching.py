#!/usr/bin/env python3
"""Trace Memphis minimum-cost shipment changes across tardiness penalties."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from china_us_multimodal.domain import RouteAllocation
from china_us_multimodal.evaluator import evaluate_solution
from china_us_multimodal.io import load_model_data, select_case
from run_full_structural_cost_sensitivity import (
    scaled_case_data,
    sensitivity_config,
    specs,
)
from run_supplementary_core_exact import gateway_for_path
from china_us_multimodal.config import load_config


LEVELS = (0.0, 0.25, 0.50, 1.0, 1.50, 2.0)
CASE_ID = "D2"
ENDPOINT = "min_cost_then_time"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/v22_full_structural_cost_sensitivity"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v29_memphis_penalty_switching"),
    )
    args = parser.parse_args()

    all_data = load_model_data(args.data)
    base_case = select_case(all_data, CASE_ID)
    base_config = load_config(args.config, "pilot_baseline")
    allocation_rows = read_csv(args.results / "exact_endpoint_allocations.csv")
    endpoint_rows = read_csv(args.results / "exact_endpoint_summary.csv")
    component_rows = read_csv(args.results / "endpoint_cost_components.csv")

    endpoint_index = {row["validation_id"]: row for row in endpoint_rows}
    component_index = {row["validation_id"]: row for row in component_rows}
    allocations_by_validation: dict[str, list[RouteAllocation]] = defaultdict(list)
    for row in allocation_rows:
        allocations_by_validation[row["validation_id"]].append(
            RouteAllocation(
                shipment_id=row["shipment_id"],
                quantity_feu=float(row["quantity_feu"]),
                arc_ids=tuple(row["arc_ids"].split("|")),
            )
        )

    target_specs = {
        spec.multiplier: spec
        for spec in specs()
        if spec.condition.case_id == CASE_ID
        and spec.dimension == "tardiness_penalty"
        and spec.multiplier in LEVELS
    }
    if set(target_specs) != set(LEVELS):
        raise RuntimeError(f"Missing penalty specifications: {set(LEVELS)-set(target_specs)}")

    level_rows: list[dict[str, object]] = []
    shipment_rows: list[dict[str, object]] = []
    profiles: dict[float, dict[str, dict[str, object]]] = {}

    for multiplier in LEVELS:
        spec = target_specs[multiplier]
        validation_id = (
            f"{CASE_ID}_{spec.condition.condition_id}_allowed_{ENDPOINT}"
        )
        if validation_id not in allocations_by_validation:
            raise KeyError(f"Missing allocations for {validation_id}.")
        case_data = scaled_case_data(base_case, spec)
        config = sensitivity_config(base_config, spec, "allowed")
        allocations = allocations_by_validation[validation_id]
        evaluation = evaluate_solution(case_data, allocations, config)
        if not evaluation.feasible:
            raise RuntimeError(
                f"Stored allocation {validation_id} failed evaluation: "
                + "; ".join(evaluation.violations)
            )

        gateway_quantities: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        path_rows: dict[str, list[str]] = defaultdict(list)
        for allocation in allocations:
            gateway = gateway_for_path(case_data, allocation.arc_ids)
            gateway_quantities[allocation.shipment_id][gateway] += (
                allocation.quantity_feu
            )
            path_rows[allocation.shipment_id].append(
                f"{gateway}:{'|'.join(allocation.arc_ids)}={allocation.quantity_feu:.6f}"
            )

        outcomes: dict[str, list] = defaultdict(list)
        for outcome in evaluation.routes:
            outcomes[outcome.shipment_id].append(outcome)

        profile: dict[str, dict[str, object]] = {}
        base_weighted_tardiness_cost = 0.0
        for shipment_id in sorted(case_data.shipments):
            shipment_outcomes = outcomes[shipment_id]
            quantity = sum(row.quantity_feu for row in shipment_outcomes)
            mean_arrival = sum(
                row.arrival_h * row.quantity_feu for row in shipment_outcomes
            ) / quantity
            tardiness_feu_h = sum(
                row.tardiness_h * row.quantity_feu for row in shipment_outcomes
            )
            base_shipment = base_case.shipments[shipment_id]
            base_tardiness_rate = (
                base_shipment.tardiness_usd_per_feu_h
                if base_shipment.tardiness_usd_per_feu_h is not None
                else base_config.penalties.tardiness_usd_per_feu_h
            )
            base_weighted_tardiness = tardiness_feu_h * base_tardiness_rate
            base_weighted_tardiness_cost += base_weighted_tardiness
            mean_tardiness = tardiness_feu_h / quantity
            record = {
                "shipment_id": shipment_id,
                "penalty_multiplier": multiplier,
                "shipment_quantity_feu": quantity,
                "ny_nj_quantity_feu": gateway_quantities[shipment_id].get(
                    "ny_nj", 0.0
                ),
                "la_lb_quantity_feu": gateway_quantities[shipment_id].get(
                    "la_lb", 0.0
                ),
                "seattle_tacoma_quantity_feu": gateway_quantities[shipment_id].get(
                    "seattle_tacoma", 0.0
                ),
                "quantity_weighted_arrival_h": mean_arrival,
                "tardiness_feu_h": tardiness_feu_h,
                "base_tardiness_rate_usd_per_feu_h": base_tardiness_rate,
                "base_weighted_tardiness_cost_usd": base_weighted_tardiness,
                "quantity_weighted_tardiness_h": mean_tardiness,
                "route_allocations": ";".join(sorted(path_rows[shipment_id])),
            }
            profile[shipment_id] = record
            shipment_rows.append(record)
        profiles[multiplier] = profile

        endpoint = endpoint_index[validation_id]
        components = component_index[validation_id]
        level_rows.append(
            {
                "penalty_multiplier": multiplier,
                "validation_id": validation_id,
                "total_cost_usd": endpoint["total_cost_usd"],
                "weighted_mean_delivery_time_h": endpoint[
                    "quantity_weighted_mean_delivery_time_h"
                ],
                "ny_nj_share": endpoint["ny_nj_share"],
                "total_tardiness_feu_h": evaluation.total_tardiness_feu_h,
                "base_weighted_tardiness_cost_usd": (
                    base_weighted_tardiness_cost
                ),
                "transport_cost_usd": components["transport_cost_usd"],
                "holding_cost_usd": components["holding_cost_usd"],
                "transfer_cost_usd": components["transfer_cost_usd"],
                "processing_cost_usd": components["processing_cost_usd"],
                "tardiness_cost_usd": components["tardiness_cost_usd"],
            }
        )

    write_csv(args.output / "penalty_level_summary.csv", level_rows)
    write_csv(args.output / "shipment_penalty_profiles.csv", shipment_rows)

    switch_rows: list[dict[str, object]] = []
    monotonicity_rows: list[dict[str, object]] = []
    for lower, upper in zip(LEVELS, LEVELS[1:]):
        lower_total = next(
            float(row["base_weighted_tardiness_cost_usd"])
            for row in level_rows
            if float(row["penalty_multiplier"]) == lower
        )
        upper_total = next(
            float(row["base_weighted_tardiness_cost_usd"])
            for row in level_rows
            if float(row["penalty_multiplier"]) == upper
        )
        monotonicity_rows.append(
            {
                "lower_penalty_multiplier": lower,
                "upper_penalty_multiplier": upper,
                "lower_base_weighted_tardiness_cost_usd": lower_total,
                "upper_base_weighted_tardiness_cost_usd": upper_total,
                "change_base_weighted_tardiness_cost_usd": upper_total - lower_total,
                "nonincreasing_within_tolerance": int(
                    upper_total <= lower_total + 1e-5
                ),
            }
        )
        for shipment_id in sorted(profiles[lower]):
            before = profiles[lower][shipment_id]
            after = profiles[upper][shipment_id]
            fields = (
                "ny_nj_quantity_feu",
                "la_lb_quantity_feu",
                "seattle_tacoma_quantity_feu",
                "quantity_weighted_arrival_h",
                "tardiness_feu_h",
            )
            if not any(
                abs(float(after[field]) - float(before[field])) > 1e-6
                for field in fields
            ):
                continue
            switch_rows.append(
                {
                    "shipment_id": shipment_id,
                    "lower_penalty_multiplier": lower,
                    "upper_penalty_multiplier": upper,
                    "ny_nj_quantity_change_feu": float(
                        after["ny_nj_quantity_feu"]
                    )
                    - float(before["ny_nj_quantity_feu"]),
                    "la_lb_quantity_change_feu": float(
                        after["la_lb_quantity_feu"]
                    )
                    - float(before["la_lb_quantity_feu"]),
                    "seattle_tacoma_quantity_change_feu": float(
                        after["seattle_tacoma_quantity_feu"]
                    )
                    - float(before["seattle_tacoma_quantity_feu"]),
                    "mean_arrival_change_h": float(
                        after["quantity_weighted_arrival_h"]
                    )
                    - float(before["quantity_weighted_arrival_h"]),
                    "tardiness_change_feu_h": float(after["tardiness_feu_h"])
                    - float(before["tardiness_feu_h"]),
                    "routes_before": before["route_allocations"],
                    "routes_after": after["route_allocations"],
                }
            )

    write_csv(args.output / "adjacent_penalty_switches.csv", switch_rows)
    write_csv(args.output / "tardiness_monotonicity_check.csv", monotonicity_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "results_source": str(args.results),
                "case_id": CASE_ID,
                "endpoint": ENDPOINT,
                "penalty_multipliers": list(LEVELS),
                "all_tardiness_monotonicity_checks_passed": all(
                    int(row["nonincreasing_within_tolerance"]) == 1
                    for row in monotonicity_rows
                ),
                "switch_rows": len(switch_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Memphis penalty-switching analysis written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
