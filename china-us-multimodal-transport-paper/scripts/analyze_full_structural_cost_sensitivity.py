#!/usr/bin/env python3
"""Create manuscript-ready tables from the exact cost sensitivity experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


COMPONENTS = (
    "transport_cost_usd",
    "holding_cost_usd",
    "transfer_cost_usd",
    "processing_cost_usd",
    "tardiness_cost_usd",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty analysis table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
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
    parser = argparse.ArgumentParser(
        description="Analyze exact full-path penalty and holding-cost sensitivity."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/v22_full_structural_cost_sensitivity"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v22_full_structural_cost_sensitivity"),
    )
    args = parser.parse_args()

    values = read_csv(args.results / "counterfactual_value.csv")
    shares = read_csv(args.results / "endpoint_ny_nj_share_ranges.csv")
    components = read_csv(args.results / "endpoint_cost_components.csv")

    share_index = {
        (row["case_id"], row["condition_id"], row["endpoint_region"]): row
        for row in shares
    }

    def share_range(row: dict[str, str]) -> tuple[float, float]:
        key = (row["case_id"], row["condition_id"], row["endpoint"])
        source = share_index.get(key)
        if source is None:
            source = share_index.get(
                (row["case_id"], row["condition_id"], "coincident_endpoints")
            )
        if source is None:
            raise KeyError(f"Missing share range for {key}")
        return float(source["ny_nj_min_share"]), float(source["ny_nj_max_share"])

    value_rows: list[dict[str, object]] = []
    for row in values:
        low, high = share_range(row)
        value_rows.append(
            {
                "case_id": row["case_id"],
                "condition_id": row["condition_id"],
                "sensitivity_dimension": row["sensitivity_dimension"],
                "sensitivity_multiplier": float(row["sensitivity_multiplier"]),
                "base_delay_h": float(row["base_delay_h"]),
                "endpoint": row["endpoint"],
                "allowed_solution_ny_nj_share": float(row["allowed_ny_nj_share"]),
                "endpoint_equivalent_min_ny_nj_share": low,
                "endpoint_equivalent_max_ny_nj_share": high,
                "endpoint_share_state": share_state(low, high),
                "cost_avoided_by_allowing_ny_nj_usd": float(
                    row["cost_avoided_by_allowing_ny_nj_usd"]
                ),
                "time_avoided_by_allowing_ny_nj_h": float(
                    row["time_avoided_by_allowing_ny_nj_h"]
                ),
            }
        )
    write_csv(args.output / "cost_sensitivity_value.csv", value_rows)

    component_index = {
        (
            row["case_id"],
            row["condition_id"],
            row["gateway_access"],
            row["endpoint"],
        ): row
        for row in components
    }
    component_rows: list[dict[str, object]] = []
    for row in values:
        key = (row["case_id"], row["condition_id"])
        allowed = component_index[(*key, "allowed", row["endpoint"])]
        restricted = component_index[(*key, "restricted", row["endpoint"])]
        deltas = {
            f"{field.removesuffix('_usd')}_avoided_usd": (
                float(restricted[field]) - float(allowed[field])
            )
            for field in COMPONENTS
        }
        component_rows.append(
            {
                "case_id": row["case_id"],
                "condition_id": row["condition_id"],
                "sensitivity_dimension": row["sensitivity_dimension"],
                "sensitivity_multiplier": float(row["sensitivity_multiplier"]),
                "endpoint": row["endpoint"],
                **deltas,
                "component_total_avoided_usd": sum(deltas.values()),
                "reported_total_avoided_usd": float(
                    row["cost_avoided_by_allowing_ny_nj_usd"]
                ),
            }
        )
    write_csv(args.output / "counterfactual_cost_components.csv", component_rows)

    penalty_zero = [
        row
        for row in value_rows
        if row["sensitivity_dimension"] == "tardiness_penalty"
        and row["sensitivity_multiplier"] == 0.0
        and row["endpoint"] == "min_cost_then_time"
    ]
    holding_rows = [
        row
        for row in value_rows
        if row["sensitivity_dimension"] == "holding_cost"
    ]
    penalty_sequences: dict[str, list[dict[str, object]]] = {}
    for case_id in sorted({str(row["case_id"]) for row in value_rows}):
        sequence = sorted(
            (
                row
                for row in value_rows
                if row["case_id"] == case_id
                and row["sensitivity_dimension"] == "tardiness_penalty"
                and row["endpoint"] == "min_cost_then_time"
            ),
            key=lambda row: float(row["sensitivity_multiplier"]),
        )
        penalty_sequences[case_id] = [
            {
                "multiplier": row["sensitivity_multiplier"],
                "min_share": row["endpoint_equivalent_min_ny_nj_share"],
                "max_share": row["endpoint_equivalent_max_ny_nj_share"],
                "cost_avoided_usd": row["cost_avoided_by_allowing_ny_nj_usd"],
            }
            for row in sequence
        ]

    findings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_source": str(args.results),
        "zero_tardiness_penalty_min_cost_endpoint": {
            str(row["case_id"]): {
                "ny_nj_min_share": row["endpoint_equivalent_min_ny_nj_share"],
                "ny_nj_max_share": row["endpoint_equivalent_max_ny_nj_share"],
                "cost_avoided_usd": row["cost_avoided_by_allowing_ny_nj_usd"],
            }
            for row in penalty_zero
        },
        "penalty_min_cost_sequences": penalty_sequences,
        "all_zero_penalty_min_cost_regions_use_ny_nj": all(
            float(row["endpoint_equivalent_min_ny_nj_share"]) > 1e-9
            for row in penalty_zero
        ),
        "holding_sensitivity_has_any_endpoint_activation_change": any(
            share_state(
                float(row["endpoint_equivalent_min_ny_nj_share"]),
                float(row["endpoint_equivalent_max_ny_nj_share"]),
            )
            != share_state(
                float(reference["endpoint_equivalent_min_ny_nj_share"]),
                float(reference["endpoint_equivalent_max_ny_nj_share"]),
            )
            for row in holding_rows
            for reference in holding_rows
            if row["case_id"] == reference["case_id"]
            and row["endpoint"] == reference["endpoint"]
        ),
        "interpretation": [
            "At zero tardiness penalty, NY/NJ remains in the D1 and D3 minimum-cost endpoint regions but not the D2 region; its cost value in D1 and D3 is therefore not generated solely by tardiness penalties.",
            "Minimum-cost gateway shares need not change monotonically with the tardiness multiplier because discrete path, service, and minimum-share choices change jointly.",
            "Cost-component differences are restricted minus allowed at the same lexicographic endpoint; positive values are avoided costs and negative values are added costs.",
            "Cost and time benefits remain endpoint-specific and must not be presented as simultaneous unless they occur in the same endpoint row.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(findings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Full structural cost-sensitivity analysis written to {args.output}.")


if __name__ == "__main__":
    main()
