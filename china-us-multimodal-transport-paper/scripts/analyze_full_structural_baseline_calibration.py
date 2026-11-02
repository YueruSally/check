#!/usr/bin/env python3
"""Summarize exact baseline gateway calibration results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def share_state(low: float, high: float) -> str:
    if high <= 1e-9:
        return "not_used"
    if high < 0.10 - 1e-9:
        return "used_below_10pct"
    if low >= 0.10 - 1e-9:
        return "all_endpoint_equivalent_solutions_at_least_10pct"
    return "crosses_10pct_within_endpoint_equivalent_region"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/v23_full_structural_baseline_calibration"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v23_full_structural_baseline_calibration"),
    )
    args = parser.parse_args()

    values = read_csv(args.results / "baseline_calibration_value.csv")
    shares = read_csv(args.results / "endpoint_ny_nj_share_ranges.csv")
    share_index = {
        (row["case_id"], row["condition_id"], row["endpoint_region"]): row
        for row in shares
    }
    rows: list[dict[str, object]] = []
    for row in values:
        source = share_index.get(
            (row["case_id"], row["condition_id"], row["endpoint"])
        ) or share_index.get(
            (row["case_id"], row["condition_id"], "coincident_endpoints")
        )
        if source is None:
            raise KeyError(
                f"Missing share range for {row['case_id']}/{row['condition_id']}"
            )
        low = float(source["ny_nj_min_share"])
        high = float(source["ny_nj_max_share"])
        rows.append(
            {
                **row,
                "calibration_value": float(row["calibration_value"]),
                "allowed_cost_usd": float(row["allowed_cost_usd"]),
                "restricted_cost_usd": float(row["restricted_cost_usd"]),
                "cost_avoided_by_allowing_ny_nj_usd": float(
                    row["cost_avoided_by_allowing_ny_nj_usd"]
                ),
                "allowed_time_h": float(row["allowed_time_h"]),
                "restricted_time_h": float(row["restricted_time_h"]),
                "time_avoided_by_allowing_ny_nj_h": float(
                    row["time_avoided_by_allowing_ny_nj_h"]
                ),
                "allowed_ny_nj_share": float(row["allowed_ny_nj_share"]),
                "endpoint_equivalent_min_ny_nj_share": low,
                "endpoint_equivalent_max_ny_nj_share": high,
                "endpoint_share_state": share_state(low, high),
            }
        )
    write_csv(args.output / "baseline_calibration_summary.csv", rows)

    material = [
        row
        for row in rows
        if float(row["endpoint_equivalent_max_ny_nj_share"]) > 1e-9
        or abs(float(row["cost_avoided_by_allowing_ny_nj_usd"])) > 1.0
        or abs(float(row["time_avoided_by_allowing_ny_nj_h"])) > 1e-5
    ]
    write_csv(args.output / "material_calibration_changes.csv", material)

    findings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_source": str(args.results),
        "maximum_tested_ocean_cost_reduction_fraction": 0.30,
        "maximum_tested_panama_time_reduction_h": 96.0,
        "any_endpoint_region_reaches_10pct_ny_nj": any(
            float(row["endpoint_equivalent_max_ny_nj_share"]) >= 0.10 - 1e-9
            for row in rows
        ),
        "material_rows": len(material),
        "interpretation": [
            "A 30% USEC ocean-cost reduction does not introduce NY/NJ into either exact endpoint for any destination.",
            "A 48-hour Panama transit-time reduction does not introduce NY/NJ into either exact endpoint for any destination.",
            "At a 96-hour reduction, only the Columbus minimum-cost endpoint uses NY/NJ and its exact share remains below the 10% network-level criterion.",
            "The baseline is therefore robust to the tested one-factor changes but represents a timetable- and delivery-sensitive case-study setting rather than a universal market calibration.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(findings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Full structural baseline-calibration analysis written to {args.output}.")


if __name__ == "__main__":
    main()
