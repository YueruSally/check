#!/usr/bin/env python3
"""Summarize exact epsilon trade-off solutions and same-cap gateway value."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/v27_full_structural_epsilon_tradeoffs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v27_full_structural_epsilon_tradeoffs"),
    )
    args = parser.parse_args()

    points = read_csv(args.results / "epsilon_tradeoff_summary.csv")
    shares = read_csv(args.results / "epsilon_ny_nj_share_ranges.csv")
    share_index = {row["tradeoff_id"]: row for row in shares}
    index = {
        (row["case_id"], row["condition_id"], row["epsilon_level"], row["gateway_access"]): row
        for row in points
    }
    rows: list[dict[str, object]] = []
    keys = sorted(
        {
            (row["case_id"], row["condition_id"], row["epsilon_level"])
            for row in points
        }
    )
    for case_id, condition_id, level in keys:
        allowed = index[(case_id, condition_id, level, "allowed")]
        restricted = index[(case_id, condition_id, level, "restricted")]
        allowed_proven = int(allowed["tradeoff_proven"]) == 1
        restricted_proven = int(restricted["tradeoff_proven"]) == 1
        share = share_index.get(allowed["tradeoff_id"])
        rows.append(
            {
                "case_id": case_id,
                "condition_id": condition_id,
                "epsilon_level": int(level),
                "time_cap_h": float(allowed["time_cap_h"]),
                "allowed_proven": int(allowed_proven),
                "restricted_proven": int(restricted_proven),
                "restricted_proven_infeasible": int(
                    restricted["stage_one_proven_infeasible"]
                ),
                "allowed_cost_usd": float(allowed["total_cost_usd"])
                if allowed_proven
                else None,
                "allowed_time_h": float(
                    allowed["quantity_weighted_mean_delivery_time_h"]
                )
                if allowed_proven
                else None,
                "restricted_cost_usd": float(restricted["total_cost_usd"])
                if restricted_proven
                else None,
                "restricted_time_h": float(
                    restricted["quantity_weighted_mean_delivery_time_h"]
                )
                if restricted_proven
                else None,
                "ny_nj_min_share": float(share["ny_nj_min_share"])
                if share
                else None,
                "ny_nj_max_share": float(share["ny_nj_max_share"])
                if share
                else None,
                "cost_avoided_at_same_time_cap_usd": (
                    float(restricted["total_cost_usd"])
                    - float(allowed["total_cost_usd"])
                    if allowed_proven and restricted_proven
                    else None
                ),
                "time_difference_restricted_minus_allowed_h": (
                    float(restricted["quantity_weighted_mean_delivery_time_h"])
                    - float(allowed["quantity_weighted_mean_delivery_time_h"])
                    if allowed_proven and restricted_proven
                    else None
                ),
            }
        )
    write_csv(args.output / "epsilon_tradeoff_comparison.csv", rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "results_source": str(args.results),
                "description": (
                    "Exact cost-minimizing solutions at common delivery-time caps, "
                    "with NY/NJ share bounds for the allowed model."
                ),
                "limits": [
                    "Internal epsilon points support preference-dependence analysis but are not labelled a complete Pareto front.",
                    "Cost savings compare allowed and restricted models at the same time cap; infeasible restricted caps are reported rather than imputed.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Epsilon trade-off analysis written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
