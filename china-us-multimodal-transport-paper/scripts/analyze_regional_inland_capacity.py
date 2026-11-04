#!/usr/bin/env python3
"""Create manuscript-ready summaries for the regional inland-capacity test."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


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
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/v28_regional_inland_capacity"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v28_regional_inland_capacity"),
    )
    args = parser.parse_args()

    endpoints = read_csv(args.results / "exact_endpoint_summary.csv")
    ranges = read_csv(args.results / "endpoint_ny_nj_share_ranges.csv")
    values = read_csv(args.results / "regional_inland_gateway_value.csv")
    usage = read_csv(args.results / "exact_endpoint_capacity_usage.csv")

    endpoint_index = {
        (
            row["case_id"],
            row["condition_id"],
            row["gateway_access"],
            row["endpoint"],
        ): row
        for row in endpoints
    }
    range_index = {
        (row["case_id"], row["condition_id"], row["endpoint_region"]): row
        for row in ranges
    }

    comparison_rows: list[dict[str, object]] = []
    for value in values:
        key = (value["case_id"], value["condition_id"])
        endpoint = value["endpoint"]
        allowed = endpoint_index[(*key, "allowed", endpoint)]
        restricted = endpoint_index[(*key, "restricted", endpoint)]
        share = range_index.get((*key, endpoint))
        if share is None:
            share = range_index[(*key, "coincident_endpoints")]
        low = float(share["ny_nj_min_share"])
        high = float(share["ny_nj_max_share"])
        comparison_rows.append(
            {
                "case_id": value["case_id"],
                "condition_id": value["condition_id"],
                "delay_h": value["delay_h"],
                "regional_inland_capacity_multiplier": value[
                    "regional_inland_capacity_multiplier"
                ],
                "regional_inland_arc_ids": value["regional_inland_arc_ids"],
                "endpoint": endpoint,
                "allowed_proven": allowed["lexicographic_proven"],
                "restricted_proven": restricted["lexicographic_proven"],
                "allowed_total_cost_usd": allowed["total_cost_usd"],
                "restricted_total_cost_usd": restricted["total_cost_usd"],
                "allowed_mean_delivery_time_h": allowed[
                    "quantity_weighted_mean_delivery_time_h"
                ],
                "restricted_mean_delivery_time_h": restricted[
                    "quantity_weighted_mean_delivery_time_h"
                ],
                "ny_nj_min_share": low,
                "ny_nj_max_share": high,
                "ny_nj_state": (
                    "not_used"
                    if high <= 1e-9
                    else "below_10pct"
                    if high < 0.10 - 1e-9
                    else "all_equivalent_at_least_10pct"
                    if low >= 0.10 - 1e-9
                    else "crosses_10pct"
                ),
                "cost_avoided_usd": value[
                    "cost_avoided_by_allowing_ny_nj_usd"
                ],
                "time_avoided_h": value["time_avoided_by_allowing_ny_nj_h"],
            }
        )
    write_csv(args.output / "regional_inland_capacity_comparison.csv", comparison_rows)

    meta = {row["validation_id"]: row for row in endpoints}
    target_by_condition = {
        (row["case_id"], row["condition_id"]): set(
            row["regional_inland_arc_ids"].split("|")
        )
        for row in values
    }
    binding_rows: list[dict[str, object]] = []
    for row in usage:
        if row["resource_type"] != "arc_day":
            continue
        endpoint = meta[row["validation_id"]]
        targets = target_by_condition[(endpoint["case_id"], endpoint["condition_id"])]
        if row["resource_id"] not in targets:
            continue
        if float(row["utilization_pct"]) < 90.0 - 1e-9:
            continue
        binding_rows.append(
            {
                "validation_id": row["validation_id"],
                "case_id": endpoint["case_id"],
                "condition_id": endpoint["condition_id"],
                "gateway_access": endpoint["gateway_access"],
                "endpoint": endpoint["endpoint"],
                "resource_id": row["resource_id"],
                "origin": row["origin"],
                "destination": row["destination"],
                "model_day": row["period"],
                "used_feu": row["used_feu"],
                "capacity_feu": row["capacity_feu"],
                "utilization_pct": row["utilization_pct"],
                "binding_100pct": row["binding_100pct"],
            }
        )
    if binding_rows:
        write_csv(args.output / "binding_regional_inland_arc_days.csv", binding_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "results_source": str(args.results),
                "comparison_rows": len(comparison_rows),
                "binding_target_arc_day_rows": len(binding_rows),
                "interpretation": (
                    "All rail and road daily-arc capacities from LA/LB and "
                    "Seattle/Tacoma to each independently solved destination are "
                    "scaled together; service and gateway-node capacities remain fixed."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Regional inland-capacity analysis written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
