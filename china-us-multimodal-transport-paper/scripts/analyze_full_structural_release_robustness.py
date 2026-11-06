#!/usr/bin/env python3
"""Combine single- and multi-wave exact evidence into robustness tables."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median


ENDPOINTS = ("min_cost_then_time", "min_time_then_cost")
RELEASE_DELAY_GRID = (0.0, 168.0, 240.0, 264.0, 288.0, 336.0)


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
        "--single-wave-analysis",
        type=Path,
        default=Path("analysis/v21_full_structural_core_exact"),
    )
    parser.add_argument(
        "--release-results",
        type=Path,
        default=Path("results/v24_full_structural_release_robustness"),
    )
    parser.add_argument(
        "--single-wave-inland-analysis",
        type=Path,
        default=Path("analysis/v26_full_structural_inland_bottleneck"),
        help=(
            "Additional exact single-wave delay levels solved in the inland "
            "interaction experiment; only the unchanged 100% inland-capacity "
            "rows are merged."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/v24_full_structural_release_robustness"),
    )
    args = parser.parse_args()

    single = read_csv(args.single_wave_analysis / "delay_gateway_value.csv")
    single_additional = read_csv(
        args.single_wave_inland_analysis / "inland_capacity_delay_interaction.csv"
    )
    values = read_csv(args.release_results / "release_pattern_gateway_value.csv")
    shares = read_csv(args.release_results / "endpoint_ny_nj_share_ranges.csv")
    capacities = read_csv(args.release_results / "exact_endpoint_capacity_usage.csv")
    share_index = {
        (
            row["release_instance_id"],
            row["case_id"],
            float(row["delay_h"]),
            row["endpoint_region"],
        ): row
        for row in shares
    }

    evidence: list[dict[str, object]] = []
    for row in single:
        delay = float(row["delay_h"])
        if delay not in RELEASE_DELAY_GRID:
            continue
        evidence.append(
            {
                "release_pattern": "single_wave",
                "release_instance_id": "single_wave_selectable_service",
                "release_generator_seed": "",
                "case_id": row["case_id"],
                "delay_h": delay,
                "endpoint": row["endpoint"],
                "ny_nj_min_share": float(row["endpoint_equivalent_min_ny_nj_share"]),
                "ny_nj_max_share": float(row["endpoint_equivalent_max_ny_nj_share"]),
                "cost_avoided_usd": float(
                    row["cost_avoided_by_allowing_ny_nj_usd"]
                ),
                "time_avoided_h": float(
                    row["time_avoided_by_allowing_ny_nj_h"]
                ),
            }
        )

    # The inland-interaction design solved two additional delay levels without
    # changing inland capacity: Memphis at 276 h and Columbus at 216 h.  These
    # are exact single-wave observations and must be included when identifying
    # the first tested material-use level.
    for row in single_additional:
        delay = float(row["delay_h"])
        if abs(float(row["inland_capacity_multiplier"]) - 1.0) > 1e-9:
            continue
        if (row["case_id"], delay) not in {("D2", 276.0), ("D3", 216.0)}:
            continue
        evidence.append(
            {
                "release_pattern": "single_wave",
                "release_instance_id": "single_wave_selectable_service",
                "release_generator_seed": "",
                "case_id": row["case_id"],
                "delay_h": delay,
                "endpoint": row["endpoint"],
                "ny_nj_min_share": float(row["ny_nj_min_share"]),
                "ny_nj_max_share": float(row["ny_nj_max_share"]),
                "cost_avoided_usd": float(row["cost_avoided_usd"]),
                "time_avoided_h": float(row["time_avoided_h"]),
            }
        )

    for row in values:
        instance_id = row["release_instance_id"]
        case_id = row["case_id"]
        delay = float(row["delay_h"])
        endpoint = row["endpoint"]
        source = share_index.get((instance_id, case_id, delay, endpoint))
        if source is None:
            source = share_index.get(
                (instance_id, case_id, delay, "coincident_endpoints")
            )
        if source is None:
            raise KeyError(
                f"Missing share range for {instance_id}/{case_id}/{delay}/{endpoint}"
            )
        evidence.append(
            {
                "release_pattern": row["release_pattern"],
                "release_instance_id": instance_id,
                "release_generator_seed": row["release_generator_seed"],
                "case_id": case_id,
                "delay_h": delay,
                "endpoint": endpoint,
                "ny_nj_min_share": float(source["ny_nj_min_share"]),
                "ny_nj_max_share": float(source["ny_nj_max_share"]),
                "cost_avoided_usd": float(
                    row["cost_avoided_by_allowing_ny_nj_usd"]
                ),
                "time_avoided_h": float(
                    row["time_avoided_by_allowing_ny_nj_h"]
                ),
            }
        )
    write_csv(args.output / "combined_release_evidence.csv", evidence)

    instance_groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in evidence:
        instance_groups[
            (
                str(row["release_pattern"]),
                str(row["release_instance_id"]),
                str(row["case_id"]),
                str(row["endpoint"]),
            )
        ].append(row)

    activation_rows: list[dict[str, object]] = []
    for (pattern, instance, case_id, endpoint), rows in sorted(
        instance_groups.items()
    ):
        any_delays = sorted(
            float(row["delay_h"])
            for row in rows
            if float(row["ny_nj_max_share"]) >= 0.10 - 1e-9
        )
        all_delays = sorted(
            float(row["delay_h"])
            for row in rows
            if float(row["ny_nj_min_share"]) >= 0.10 - 1e-9
        )
        activation_rows.append(
            {
                "release_pattern": pattern,
                "release_instance_id": instance,
                "case_id": case_id,
                "endpoint": endpoint,
                "first_tested_delay_any_endpoint_equivalent_solution_at_10pct_h": (
                    any_delays[0] if any_delays else None
                ),
                "first_tested_delay_all_endpoint_equivalent_solutions_at_10pct_h": (
                    all_delays[0] if all_delays else None
                ),
            }
        )
    write_csv(args.output / "activation_by_release_instance.csv", activation_rows)

    activation_summary: list[dict[str, object]] = []
    summary_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in activation_rows:
        summary_groups[
            (str(row["release_pattern"]), str(row["case_id"]), str(row["endpoint"]))
        ].append(row)
    for (pattern, case_id, endpoint), rows in sorted(summary_groups.items()):
        any_counts = Counter(
            row["first_tested_delay_any_endpoint_equivalent_solution_at_10pct_h"]
            for row in rows
        )
        all_counts = Counter(
            row["first_tested_delay_all_endpoint_equivalent_solutions_at_10pct_h"]
            for row in rows
        )
        activation_summary.append(
            {
                "release_pattern": pattern,
                "case_id": case_id,
                "endpoint": endpoint,
                "instances": len(rows),
                "first_any_delay_counts": "|".join(
                    f"{key}:{value}" for key, value in sorted(any_counts.items())
                ),
                "first_all_delay_counts": "|".join(
                    f"{key}:{value}" for key, value in sorted(all_counts.items())
                ),
            }
        )
    write_csv(args.output / "activation_frequency_summary.csv", activation_summary)

    distribution_groups: dict[
        tuple[str, str, float, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in evidence:
        distribution_groups[
            (
                str(row["release_pattern"]),
                str(row["case_id"]),
                float(row["delay_h"]),
                str(row["endpoint"]),
            )
        ].append(row)
    distribution_rows: list[dict[str, object]] = []
    for (pattern, case_id, delay, endpoint), rows in sorted(
        distribution_groups.items()
    ):
        lows = [float(row["ny_nj_min_share"]) for row in rows]
        highs = [float(row["ny_nj_max_share"]) for row in rows]
        costs = [float(row["cost_avoided_usd"]) for row in rows]
        times = [float(row["time_avoided_h"]) for row in rows]
        distribution_rows.append(
            {
                "release_pattern": pattern,
                "case_id": case_id,
                "delay_h": delay,
                "endpoint": endpoint,
                "instances": len(rows),
                "ny_nj_min_share_min": min(lows),
                "ny_nj_min_share_median": median(lows),
                "ny_nj_min_share_max": max(lows),
                "ny_nj_max_share_min": min(highs),
                "ny_nj_max_share_median": median(highs),
                "ny_nj_max_share_max": max(highs),
                "cost_avoided_min_usd": min(costs),
                "cost_avoided_median_usd": median(costs),
                "cost_avoided_max_usd": max(costs),
                "time_avoided_min_h": min(times),
                "time_avoided_median_h": median(times),
                "time_avoided_max_h": max(times),
            }
        )
    write_csv(args.output / "release_value_distributions.csv", distribution_rows)

    utilization: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in capacities:
        utilization[(row["validation_id"], row["resource_type"])].append(
            float(row["utilization_pct"])
        )
    utilization_rows = [
        {
            "validation_id": validation_id,
            "resource_type": resource_type,
            "maximum_utilization_pct": max(values),
            "binding_95pct": int(max(values) >= 95.0 - 1e-9),
        }
        for (validation_id, resource_type), values in sorted(utilization.items())
    ]
    write_csv(args.output / "endpoint_maximum_resource_utilization.csv", utilization_rows)

    findings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_wave_source": str(args.single_wave_analysis),
        "single_wave_additional_delay_source": str(
            args.single_wave_inland_analysis
        ),
        "multi_wave_source": str(args.release_results),
        "activation_share": 0.10,
        "interpretation": [
            "Release dispersion preserves the qualitative value of an alternative gateway but materially changes its first tested use and its objective dependence.",
            "Chicago minimum-cost use remains at 288 h across multi-wave designs, while its minimum-time use moves to 336 h.",
            "Memphis generally moves from 288 h under the single wave to 336 h under multi-wave release; one daily instance reaches 10% in the minimum-cost endpoint at 288 h.",
            "Columbus minimum-cost use varies between 168 h and 240 h across daily assignments and occurs at 264 h in the weekly-wave designs, whereas minimum-time use occurs at 336 h in every tested multi-wave instance.",
            "The repeated 32.66% single-wave share is not a universal capacity ceiling: multi-wave exact shares cover multiple intermediate values and 100%.",
            "Most highlighted multi-wave solutions do not bind service, daily-arc, or node capacities, so gateway switching cannot be attributed solely to capacity overflow.",
        ],
        "limits": [
            "Constructed release instances are robustness cases, not empirical occurrence probabilities.",
            "First-use statements refer only to the tested delay grid.",
            "Cost and time values remain endpoint-specific.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(findings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Full structural release-robustness analysis written to {args.output}.")


if __name__ == "__main__":
    main()
