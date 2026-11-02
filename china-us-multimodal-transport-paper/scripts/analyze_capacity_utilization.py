#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.diagnostics import capacity_usage_rows
from china_us_multimodal.domain import RouteAllocation
from china_us_multimodal.io import load_model_data, select_case


DETAIL_FIELDS = [
    "solution_id",
    "case_id",
    "scenario",
    "roles",
    "resource_type",
    "resource_id",
    "period",
    "origin",
    "destination",
    "used_feu",
    "capacity_feu",
    "slack_feu",
    "utilization_pct",
    "binding_90pct",
    "binding_95pct",
    "binding_100pct",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct capacity utilization for representative solutions or for "
            "all distinct configurations found at retained objective points."
        )
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all-found-configurations",
        action="store_true",
        help=(
            "Use found_configuration_*.csv instead of only role-labelled "
            "representative solutions."
        ),
    )
    args = parser.parse_args()

    if args.all_found_configurations:
        solution_rows = read_csv(args.results / "found_configuration_solutions.csv")
        allocation_rows = read_csv(
            args.results / "found_configuration_allocations.csv"
        )
        id_field = "configuration_id"
        representative = {row[id_field]: row for row in solution_rows}
    else:
        solution_rows = read_csv(args.results / "representative_solutions.csv")
        allocation_rows = read_csv(args.results / "representative_allocations.csv")
        id_field = "solution_id"
        representative = {
            row[id_field]: row
            for row in solution_rows
            if row.get("roles", "").strip()
        }
    grouped: dict[str, list[RouteAllocation]] = defaultdict(list)
    for row in allocation_rows:
        if row[id_field] not in representative:
            continue
        grouped[row[id_field]].append(
            RouteAllocation(
                shipment_id=row["shipment_id"],
                quantity_feu=float(row["quantity_feu"]),
                arc_ids=tuple(row["arc_ids"].split("|")),
            )
        )

    all_data = load_model_data(args.data)
    detail: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for solution_id, meta in sorted(representative.items()):
        case_data = select_case(all_data, meta["case_id"])
        config = load_config(args.config, meta["scenario"])
        usages = capacity_usage_rows(case_data, grouped[solution_id], config)
        for row in usages:
            detail.append(
                {
                    "solution_id": solution_id,
                    "case_id": meta["case_id"],
                    "scenario": meta["scenario"],
                    "roles": meta["roles"],
                    **row,
                }
            )

        for resource_type in ("service_departure", "arc_day", "node_day"):
            typed = [row for row in usages if row["resource_type"] == resource_type]
            uswc = [
                row
                for row in typed
                if {
                    str(row["resource_id"]),
                    str(row["origin"]),
                    str(row["destination"]),
                }
                & {"la_lb", "seattle_tacoma"}
            ]
            summary.append(
                {
                    "solution_id": solution_id,
                    "case_id": meta["case_id"],
                    "scenario": meta["scenario"],
                    "roles": meta["roles"],
                    "resource_type": resource_type,
                    "resources_used": len(typed),
                    "maximum_utilization_pct": max(
                        (float(row["utilization_pct"]) for row in typed), default=0.0
                    ),
                    "binding_90pct_count": sum(
                        int(row["binding_90pct"]) for row in typed
                    ),
                    "binding_95pct_count": sum(
                        int(row["binding_95pct"]) for row in typed
                    ),
                    "binding_100pct_count": sum(
                        int(row["binding_100pct"]) for row in typed
                    ),
                    "uswc_maximum_utilization_pct": max(
                        (float(row["utilization_pct"]) for row in uswc), default=0.0
                    ),
                }
            )

    write_csv(args.output / "capacity_utilization.csv", detail, DETAIL_FIELDS)
    write_csv(
        args.output / "capacity_utilization_summary.csv",
        summary,
        [
            "solution_id",
            "case_id",
            "scenario",
            "roles",
            "resource_type",
            "resources_used",
            "maximum_utilization_pct",
            "binding_90pct_count",
            "binding_95pct_count",
            "binding_100pct_count",
            "uswc_maximum_utilization_pct",
        ],
    )
    print(
        f"Wrote capacity diagnostics for {len(representative)} selected configurations "
        f"to {args.output}."
    )


if __name__ == "__main__":
    main()
