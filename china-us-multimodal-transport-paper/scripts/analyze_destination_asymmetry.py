#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import route_component_rows
from china_us_multimodal.domain import RouteAllocation
from china_us_multimodal.io import load_model_data, select_case


SELECTIONS = (
    ("D1", "below_activation", "v08", "uswc_departure_delay_264h"),
    ("D1", "above_activation", "v08", "uswc_departure_delay_288h"),
    ("D2", "below_activation", "v07", "uswc_departure_delay_240h"),
    ("D2", "above_activation", "v08", "uswc_departure_delay_264h"),
    ("D3", "below_activation", "v06", "uswc_departure_delay_168h"),
    ("D3", "above_activation", "v07", "uswc_departure_delay_240h"),
)
COST_FIELDS = (
    "china_inland_transport_cost_usd",
    "ocean_transport_cost_usd",
    "us_inland_transport_cost_usd",
    "holding_cost_usd",
    "transfer_cost_usd",
    "processing_cost_usd",
    "tardiness_cost_usd",
    "total_cost_usd",
)
TIME_FIELDS = (
    "china_inland_transit_h",
    "ocean_transit_h",
    "us_inland_transit_h",
    "waiting_h",
    "transfer_h",
    "lead_time_h",
    "tardiness_h",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def balanced_solution(
    results: Path,
    case_id: str,
    scenario: str,
) -> tuple[dict[str, str], list[RouteAllocation]]:
    solutions = read_csv(results / "representative_solutions.csv")
    matches = [
        row
        for row in solutions
        if row["case_id"] == case_id
        and row["scenario"] == scenario
        and "balanced" in row.get("roles", "").lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one balanced solution for {case_id}/{scenario} in {results}; "
            f"found {len(matches)}"
        )
    meta = matches[0]
    allocations = [
        RouteAllocation(
            shipment_id=row["shipment_id"],
            quantity_feu=float(row["quantity_feu"]),
            arc_ids=tuple(row["arc_ids"].split("|")),
        )
        for row in read_csv(results / "representative_allocations.csv")
        if row["solution_id"] == meta["solution_id"]
    ]
    return meta, allocations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose destination-specific gateway activation mechanisms."
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v06", type=Path, required=True)
    parser.add_argument("--v07", type=Path, required=True)
    parser.add_argument("--v08", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {"v06": args.v06, "v07": args.v07, "v08": args.v08}
    data = load_model_data(args.data)
    output_rows: list[dict[str, object]] = []

    for case_id, activation_side, source, scenario in SELECTIONS:
        meta, allocations = balanced_solution(
            sources[source], case_id, scenario
        )
        case_data = select_case(data, case_id)
        config = load_config(args.config, scenario)
        components = route_component_rows(case_data, allocations, config)
        by_gateway: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in components:
            by_gateway[str(row["gateway"])].append(row)
        total_quantity = sum(float(row["quantity_feu"]) for row in components)
        reconstructed_solution_cost = sum(
            float(row["total_cost_usd"]) for row in components
        )
        reported_solution_cost = float(meta["total_cost_usd"])

        for gateway, rows in sorted(by_gateway.items()):
            gateway_quantity = sum(float(row["quantity_feu"]) for row in rows)
            result: dict[str, object] = {
                "case_id": case_id,
                "activation_side": activation_side,
                "scenario": scenario,
                "source_results": source,
                "solution_id": meta["solution_id"],
                "reported_solution_cost_usd": reported_solution_cost,
                "reconstructed_solution_cost_usd": reconstructed_solution_cost,
                "cost_reconciliation_residual_usd": (
                    reconstructed_solution_cost - reported_solution_cost
                ),
                "gateway": gateway,
                "quantity_feu": gateway_quantity,
                "gateway_share_pct": 100.0 * gateway_quantity / total_quantity,
            }
            for field in COST_FIELDS:
                total = sum(float(row[field]) for row in rows)
                result[field] = total
                result[field.replace("_usd", "_usd_per_feu")] = (
                    total / gateway_quantity
                )
            for field in TIME_FIELDS:
                result[f"quantity_weighted_{field}"] = sum(
                    float(row[field]) * float(row["quantity_feu"]) for row in rows
                ) / gateway_quantity
            output_rows.append(result)

    fields = list(output_rows[0]) if output_rows else []
    write_csv(args.output / "gateway_mechanism_components.csv", output_rows, fields)
    print(
        f"Wrote {len(output_rows)} destination-asymmetry mechanism rows to {args.output}."
    )


if __name__ == "__main__":
    main()
