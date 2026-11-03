#!/usr/bin/env python3
"""Create manuscript-ready tables from the full structural exact core grid."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


CASES = ("D1", "D2", "D3")
ENDPOINTS = ("min_cost_then_time", "min_time_then_cost")
METRICS = (
    ("cost", "total_cost_usd", 1.0),
    ("time", "quantity_weighted_mean_delivery_time_h", 1.0 / 3600.0),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty analysis table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def delay_hours(condition_id: str) -> int | None:
    if not condition_id.startswith("delay_") or not condition_id.endswith("h"):
        return None
    return int(condition_id.removeprefix("delay_").removesuffix("h"))


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
        description="Analyze exact endpoints from the exhaustive structural library."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/v21_full_structural_core_exact"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v21_full_structural_core_exact"),
    )
    args = parser.parse_args()

    endpoint_rows = read_csv(args.results / "full_exact_endpoint_summary.csv")
    share_rows = read_csv(args.results / "full_endpoint_share_ranges.csv")
    difference_rows = read_csv(args.results / "endpoint_difference_8_vs_full.csv")
    endpoint_index = {
        (r["case_id"], r["condition_id"], r["gateway_access"], r["endpoint"]): r
        for r in endpoint_rows
    }
    share_index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in share_rows:
        key = (row["case_id"], row["condition_id"], row["endpoint_region"])
        share_index[key] = row

    def endpoint_share_range(case_id: str, condition_id: str, endpoint: str):
        row = share_index.get((case_id, condition_id, endpoint))
        if row is None:
            row = share_index.get((case_id, condition_id, "coincident_endpoints"))
        if row is None:
            raise KeyError(f"Missing share range for {case_id}/{condition_id}/{endpoint}")
        return float(row["ny_nj_min_share"]), float(row["ny_nj_max_share"])

    value_rows: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    for case_id in CASES:
        conditions = sorted(
            {
                row["condition_id"]
                for row in endpoint_rows
                if row["case_id"] == case_id
            }
        )
        for condition_id in conditions:
            for endpoint in ENDPOINTS:
                allowed = endpoint_index[(case_id, condition_id, "allowed", endpoint)]
                restricted = endpoint_index[
                    (case_id, condition_id, "restricted", endpoint)
                ]
                low, high = endpoint_share_range(case_id, condition_id, endpoint)
                record = {
                    "case_id": case_id,
                    "condition_id": condition_id,
                    "delay_h": delay_hours(condition_id),
                    "endpoint": endpoint,
                    "allowed_cost_usd": float(allowed["total_cost_usd"]),
                    "restricted_cost_usd": float(restricted["total_cost_usd"]),
                    "cost_avoided_by_allowing_ny_nj_usd": (
                        float(restricted["total_cost_usd"])
                        - float(allowed["total_cost_usd"])
                    ),
                    "allowed_time_h": float(
                        allowed["quantity_weighted_mean_delivery_time_h"]
                    ),
                    "restricted_time_h": float(
                        restricted["quantity_weighted_mean_delivery_time_h"]
                    ),
                    "time_avoided_by_allowing_ny_nj_h": (
                        float(restricted["quantity_weighted_mean_delivery_time_h"])
                        - float(allowed["quantity_weighted_mean_delivery_time_h"])
                    ),
                    "allowed_solution_ny_nj_share": float(allowed["ny_nj_share"]),
                    "endpoint_equivalent_min_ny_nj_share": low,
                    "endpoint_equivalent_max_ny_nj_share": high,
                    "endpoint_share_state": share_state(low, high),
                }
                value_rows.append(record)
                activation_rows.append(
                    {
                        "case_id": case_id,
                        "condition_id": condition_id,
                        "delay_h": delay_hours(condition_id),
                        "endpoint": endpoint,
                        "ny_nj_min_share": low,
                        "ny_nj_max_share": high,
                        "activation_threshold_share": 0.10,
                        "state": share_state(low, high),
                    }
                )

    delay_values = [row for row in value_rows if row["delay_h"] is not None]
    scenario_values = [row for row in value_rows if row["delay_h"] is None]
    write_csv(args.output / "delay_gateway_value.csv", delay_values)
    write_csv(args.output / "capacity_compound_value.csv", scenario_values)
    write_csv(args.output / "endpoint_activation_status.csv", activation_rows)

    recovery_rows: list[dict[str, object]] = []
    for row in value_rows:
        if row["condition_id"] == "delay_0h":
            continue
        baseline = endpoint_index[
            (row["case_id"], "delay_0h", "allowed", row["endpoint"])
        ]
        for metric, field, denominator_tolerance in METRICS:
            baseline_value = float(baseline[field])
            restricted_value = float(row[f"restricted_{metric}_usd"] if metric == "cost" else row["restricted_time_h"])
            allowed_value = float(row[f"allowed_{metric}_usd"] if metric == "cost" else row["allowed_time_h"])
            disruption_loss = restricted_value - baseline_value
            avoided_loss = restricted_value - allowed_value
            primary_metric = (
                (row["endpoint"] == "min_cost_then_time" and metric == "cost")
                or (row["endpoint"] == "min_time_then_cost" and metric == "time")
            )
            recovery = (
                avoided_loss / disruption_loss
                if primary_metric and disruption_loss > denominator_tolerance
                else None
            )
            recovery_rows.append(
                {
                    "case_id": row["case_id"],
                    "condition_id": row["condition_id"],
                    "delay_h": row["delay_h"],
                    "endpoint": row["endpoint"],
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "disrupted_restricted_value": restricted_value,
                    "disrupted_allowed_value": allowed_value,
                    "disruption_loss": disruption_loss,
                    "ny_nj_avoided_loss": avoided_loss,
                    "recovery_ratio": recovery,
                    "ratio_reportable": int(recovery is not None),
                }
            )
    write_csv(args.output / "disruption_loss_recovery.csv", recovery_rows)

    material_endpoint_changes = [
        row
        for row in difference_rows
        if abs(float(row["cost_change_full_minus_8_usd"])) > 1.0
        or abs(float(row["time_change_full_minus_8_h"])) > (1.0 / 3600.0)
        or abs(float(row["ny_nj_share_change"])) > 1e-9
    ]
    write_csv(args.output / "material_path_library_changes.csv", material_endpoint_changes)

    first_use: dict[str, dict[str, object]] = {}
    for case_id in CASES:
        candidates = [
            row
            for row in activation_rows
            if row["case_id"] == case_id
            and row["delay_h"] is not None
            and float(row["ny_nj_max_share"]) >= 0.10 - 1e-9
        ]
        first = min(candidates, key=lambda row: int(row["delay_h"]))
        first_use[case_id] = {
            "delay_h": first["delay_h"],
            "endpoint": first["endpoint"],
            "state": first["state"],
        }

    findings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_source": str(args.results),
        "path_library_decision": "exhaustive_structural_library_required",
        "materially_changed_endpoint_count": len(material_endpoint_changes),
        "total_endpoint_count": len(difference_rows),
        "first_tested_delay_with_at_least_10pct_ny_nj_by_case": first_use,
        "interpretation_limits": [
            "Delay values are tested grid points, not continuous real-world thresholds.",
            "Cost and time benefits are endpoint-specific and are not simultaneous unless the same endpoint row demonstrates both.",
            "Recovery ratios are suppressed when the no-NY/NJ disruption loss is numerically negligible.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(findings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Full structural core analysis written to {args.output}.", flush=True)


if __name__ == "__main__":
    main()
