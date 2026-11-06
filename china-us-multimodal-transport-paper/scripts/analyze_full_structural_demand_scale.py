#!/usr/bin/env python3
"""Combine 80%, 100%, and 120% exact total-demand evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


ENDPOINTS = ("min_cost_then_time", "min_time_then_cost")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-analysis",
        type=Path,
        default=Path("analysis/v21_full_structural_core_exact"),
    )
    parser.add_argument(
        "--demand-results",
        type=Path,
        default=Path("results/v25_full_structural_demand_scale"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v25_full_structural_demand_scale"),
    )
    args = parser.parse_args()

    core_delay = read_csv(args.core_analysis / "delay_gateway_value.csv")
    core_other = read_csv(args.core_analysis / "capacity_compound_value.csv")
    values = read_csv(args.demand_results / "demand_scale_gateway_value.csv")
    shares = read_csv(args.demand_results / "endpoint_ny_nj_share_ranges.csv")
    capacities = read_csv(args.demand_results / "exact_endpoint_capacity_usage.csv")
    share_index = {
        (row["case_id"], row["condition_id"], row["endpoint_region"]): row
        for row in shares
    }

    evidence: list[dict[str, object]] = []
    requested = {
        (row["case_id"], row["base_condition_id"])
        for row in values
    }
    for row in [*core_delay, *core_other]:
        if (row["case_id"], row["condition_id"]) not in requested:
            continue
        evidence.append(
            {
                "case_id": row["case_id"],
                "base_condition_id": row["condition_id"],
                "demand_multiplier": 1.0,
                "endpoint": row["endpoint"],
                "ny_nj_min_share": float(row["endpoint_equivalent_min_ny_nj_share"]),
                "ny_nj_max_share": float(row["endpoint_equivalent_max_ny_nj_share"]),
                "cost_avoided_usd": float(
                    row["cost_avoided_by_allowing_ny_nj_usd"]
                ),
                "time_avoided_h": float(
                    row["time_avoided_by_allowing_ny_nj_h"]
                ),
                "source": str(args.core_analysis),
            }
        )

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
        evidence.append(
            {
                "case_id": row["case_id"],
                "base_condition_id": row["base_condition_id"],
                "demand_multiplier": float(row["demand_multiplier"]),
                "endpoint": row["endpoint"],
                "ny_nj_min_share": float(source["ny_nj_min_share"]),
                "ny_nj_max_share": float(source["ny_nj_max_share"]),
                "cost_avoided_usd": float(
                    row["cost_avoided_by_allowing_ny_nj_usd"]
                ),
                "time_avoided_h": float(
                    row["time_avoided_by_allowing_ny_nj_h"]
                ),
                "source": str(args.demand_results),
            }
        )
    evidence.sort(
        key=lambda row: (
            str(row["case_id"]),
            str(row["base_condition_id"]),
            str(row["endpoint"]),
            float(row["demand_multiplier"]),
        )
    )
    write_csv(args.output / "demand_scale_comparison.csv", evidence)

    utilization: dict[tuple[str, str], float] = {}
    for row in capacities:
        key = (row["validation_id"], row["resource_type"])
        utilization[key] = max(
            utilization.get(key, 0.0), float(row["utilization_pct"])
        )
    utilization_rows = [
        {
            "validation_id": validation_id,
            "resource_type": resource_type,
            "maximum_utilization_pct": maximum,
            "binding_95pct": int(maximum >= 95.0 - 1e-9),
            "binding_100pct": int(maximum >= 100.0 - 1e-9),
        }
        for (validation_id, resource_type), maximum in sorted(utilization.items())
    ]
    write_csv(args.output / "demand_scale_maximum_utilization.csv", utilization_rows)

    capacity35 = [
        row for row in evidence if row["base_condition_id"] == "capacity_35pct"
    ]
    findings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "core_source": str(args.core_analysis),
        "demand_scale_source": str(args.demand_results),
        "demand_multipliers": [0.8, 1.0, 1.2],
        "all_capacity35_endpoints_exclude_ny_nj": all(
            float(row["ny_nj_max_share"]) <= 1e-9 for row in capacity35
        ),
        "interpretation": [
            "At 35% West Coast port capacity, neither endpoint uses NY/NJ at 80%, 100%, or 120% demand for any destination, despite binding West Coast resources in the scaled cases.",
            "At the tested severe delays, demand intensity changes the exact NY/NJ share and the value of access but does not erase destination and endpoint dependence.",
            "Memphis at 288 h uses NY/NJ at both endpoints across all three demand levels, with a larger share at higher total demand.",
            "Columbus at 240 h uses NY/NJ in the minimum-cost endpoint across all three demand levels but not in the minimum-time endpoint.",
        ],
        "limits": [
            "Demand is proportionally scaled and is not an empirical forecast distribution.",
            "Port-capacity robustness does not by itself identify the effect of a specific inland bottleneck under delay.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(findings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Full structural demand-scale analysis written to {args.output}.")


if __name__ == "__main__":
    main()
