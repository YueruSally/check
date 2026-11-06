#!/usr/bin/env python3
"""Create manuscript-ready tables for the inland-capacity interaction."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


ENDPOINTS = ("min_cost_then_time", "min_time_then_cost")
TARGETS = {
    "D2": {"arc_id": "us-d-02", "delays_h": (264, 276, 288)},
    "D3": {"arc_id": "us-d-03", "delays_h": (216, 240, 264)},
}
REUSED = {
    ("D2", 264),
    ("D2", 288),
    ("D3", 240),
    ("D3", 264),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def state(low: float, high: float) -> str:
    if high <= 1e-9:
        return "not_used"
    if high < 0.10 - 1e-9:
        return "used_below_10pct"
    if low >= 0.10 - 1e-9:
        return "all_equivalent_solutions_at_least_10pct"
    return "crosses_10pct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-analysis",
        type=Path,
        default=Path("analysis/v21_full_structural_core_exact"),
    )
    parser.add_argument(
        "--inland-results",
        type=Path,
        default=Path("results/v26_full_structural_inland_bottleneck"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v26_full_structural_inland_bottleneck"),
    )
    args = parser.parse_args()

    new_values = read_csv(args.inland_results / "inland_bottleneck_gateway_value.csv")
    new_ranges = read_csv(args.inland_results / "endpoint_ny_nj_share_ranges.csv")
    core_values = read_csv(args.core_analysis / "delay_gateway_value.csv")
    capacities = read_csv(args.inland_results / "exact_endpoint_capacity_usage.csv")
    range_index = {
        (row["case_id"], row["condition_id"], row["endpoint_region"]): row
        for row in new_ranges
    }

    rows: list[dict[str, object]] = []
    for row in new_values:
        share = range_index.get(
            (row["case_id"], row["condition_id"], row["endpoint"])
        ) or range_index.get(
            (row["case_id"], row["condition_id"], "coincident_endpoints")
        )
        if share is None:
            raise KeyError(
                "Missing endpoint share range for "
                f"{row['case_id']}/{row['condition_id']}/{row['endpoint']}"
            )
        low = float(share["ny_nj_min_share"])
        high = float(share["ny_nj_max_share"])
        rows.append(
            {
                "case_id": row["case_id"],
                "delay_h": int(float(row["delay_h"])),
                "binding_arc_id": row["binding_arc_id"],
                "inland_capacity_multiplier": float(
                    row["binding_arc_capacity_multiplier"]
                ),
                "endpoint": row["endpoint"],
                "ny_nj_min_share": low,
                "ny_nj_max_share": high,
                "ny_nj_state": state(low, high),
                "cost_avoided_usd": float(
                    row["cost_avoided_by_allowing_ny_nj_usd"]
                ),
                "time_avoided_h": float(
                    row["time_avoided_by_allowing_ny_nj_h"]
                ),
                "source": str(args.inland_results),
            }
        )

    for row in core_values:
        case_id = row["case_id"]
        delay_h = int(float(row["delay_h"]))
        if (case_id, delay_h) not in REUSED:
            continue
        low = float(row["endpoint_equivalent_min_ny_nj_share"])
        high = float(row["endpoint_equivalent_max_ny_nj_share"])
        rows.append(
            {
                "case_id": case_id,
                "delay_h": delay_h,
                "binding_arc_id": TARGETS[case_id]["arc_id"],
                "inland_capacity_multiplier": 1.0,
                "endpoint": row["endpoint"],
                "ny_nj_min_share": low,
                "ny_nj_max_share": high,
                "ny_nj_state": state(low, high),
                "cost_avoided_usd": float(
                    row["cost_avoided_by_allowing_ny_nj_usd"]
                ),
                "time_avoided_h": float(
                    row["time_avoided_by_allowing_ny_nj_h"]
                ),
                "source": str(args.core_analysis),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["case_id"]),
            int(row["delay_h"]),
            float(row["inland_capacity_multiplier"]),
            ENDPOINTS.index(str(row["endpoint"])),
        )
    )
    write_csv(args.output / "inland_capacity_delay_interaction.csv", rows)

    maximum: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in capacities:
        if row["resource_type"] != "arc_day":
            continue
        if row["resource_id"] not in {"us-d-02", "us-d-03"}:
            continue
        key = (row["validation_id"], row["resource_type"], row["resource_id"])
        if key not in maximum or float(row["utilization_pct"]) > float(
            maximum[key]["utilization_pct"]
        ):
            maximum[key] = {
                "validation_id": row["validation_id"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "binding_period": row["period"],
                "used_feu": float(row["used_feu"]),
                "capacity_feu": float(row["capacity_feu"]),
                "utilization_pct": float(row["utilization_pct"]),
                "binding_95pct": int(row["binding_95pct"]),
                "binding_100pct": int(row["binding_100pct"]),
            }
    utilization = sorted(maximum.values(), key=lambda row: str(row["validation_id"]))
    write_csv(args.output / "target_arc_maximum_utilization.csv", utilization)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "core_source": str(args.core_analysis),
                "inland_source": str(args.inland_results),
                "findings": [
                    "Inland capacity and West Coast delay interact; neither has a destination-independent effect on NY/NJ use.",
                    "For Memphis, 50% inland capacity activates the minimum-cost endpoint at 264 h and both endpoints at 288 h, whereas 150% capacity removes NY/NJ at 288 h.",
                    "For Columbus, the minimum-cost endpoint uses NY/NJ at lower delays than the minimum-time endpoint, and the share response varies materially with inland capacity.",
                    "The mechanism is endpoint-dependent: tightening inland capacity can make NY/NJ cost-effective before it makes NY/NJ time-optimal.",
                ],
                "limits": [
                    "One diagnosed LA/LB-to-destination daily arc is varied per destination.",
                    "The design identifies a local interaction mechanism and is not a full inland-network capacity grid.",
                    "Maximum-utilization diagnostics in this table cover newly solved v2.6 tasks; reused v2.1 values remain exact but are labelled by source.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Inland-bottleneck analysis written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
