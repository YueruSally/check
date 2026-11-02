#!/usr/bin/env python3
"""Build manuscript-facing tables from completed experiment outputs.

This script performs post-processing only. It does not run the routing model or
NSGA-II. In particular, it closes the NY/NJ-allowed archive with the matching
NY/NJ-restricted archive so that the known feasible-set inclusion relation is
respected before counterfactual endpoint and hypervolume comparisons are made.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_paired_statistics import bootstrap_interval, stable_seed
from china_us_multimodal.metrics import (
    hypervolume_2d,
    nondominated_points,
    normalize_points,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "paper_revised"
V06 = ROOT / "analysis" / "v06_threshold_scenarios"
V07 = ROOT / "analysis" / "v07_activation_scenarios"
V08 = ROOT / "analysis" / "v08_focused_activation_thresholds"
V09 = ROOT / "analysis" / "v09_activation_regime"
V10 = ROOT / "results" / "v10_ny_nj_counterfactual_30seeds"
ALLOWED_RESULTS = {
    "v07": ROOT / "results" / "v07_activation_scenarios_30seeds",
    "v08": ROOT / "results" / "v08_focused_activation_thresholds_30seeds",
}
COMPARISONS = {
    "D1": (
        ("delay_activation", "uswc_departure_delay_288h", "v08", "uswc_departure_delay_288h_no_ny_nj"),
        ("extreme_compound", "uswc_compound_30pct_delay_168h", "v07", "uswc_compound_30pct_delay_168h_no_ny_nj"),
    ),
    "D2": (
        ("delay_activation", "uswc_departure_delay_264h", "v08", "uswc_departure_delay_264h_no_ny_nj"),
        ("extreme_compound", "uswc_compound_30pct_delay_168h", "v07", "uswc_compound_30pct_delay_168h_no_ny_nj"),
    ),
    "D3": (
        ("delay_activation", "uswc_departure_delay_240h", "v07", "uswc_departure_delay_240h_no_ny_nj"),
        ("extreme_compound", "uswc_compound_30pct_delay_168h", "v07", "uswc_compound_30pct_delay_168h_no_ny_nj"),
    ),
}
METRICS = (
    ("min_cost_usd", "cost_avoided_by_ny_nj", "USD"),
    ("min_weighted_mean_delivery_time_h", "delivery_time_avoided_by_ny_nj", "h"),
)
GATEWAY_ARCS = {
    "la_lb": {"sea-01", "sea-02"},
    "seattle_tacoma": {"sea-03", "sea-04"},
    "ny_nj": {"sea-05", "sea-06"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def feasible(row: dict[str, str]) -> bool:
    return int(row.get("feasible_population", "0") or 0) > 0


def objective_points(rows: list[dict[str, str]], case_id: str, scenario: str) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            float(row["total_cost_usd"]),
            float(row["quantity_weighted_mean_delivery_time_h"]),
        )
        for row in rows
        if row["case_id"] == case_id and row["scenario"] == scenario
    )


def ny_share_by_objective(results: Path) -> dict[tuple[str, str, float, float], float]:
    solutions = read_csv(results / "representative_solutions.csv")
    allocations = read_csv(results / "representative_allocations.csv")
    quantities: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in allocations:
        arcs = set(row["arc_ids"].split("|"))
        gateway = "unclassified"
        for name, gateway_arcs in GATEWAY_ARCS.items():
            if arcs & gateway_arcs:
                gateway = name
                break
        quantities[row["solution_id"]][gateway] += float(row["quantity_feu"])

    shares: dict[tuple[str, str, float, float], float] = {}
    for row in solutions:
        gateway_quantities = quantities[row["solution_id"]]
        total = sum(gateway_quantities.values())
        shares[
            (
                row["case_id"],
                row["scenario"],
                float(row["total_cost_usd"]),
                float(row["quantity_weighted_mean_delivery_time_h"]),
            )
        ] = 100.0 * gateway_quantities["ny_nj"] / total if total else 0.0
    return shares


def build_counterfactual_tables() -> None:
    summaries: list[dict[str, object]] = []
    differences: list[dict[str, object]] = []
    front_summaries: list[dict[str, object]] = []
    front_points: list[dict[str, object]] = []

    allowed_runs = {
        name: read_csv(path / "runs.csv") for name, path in ALLOWED_RESULTS.items()
    }
    allowed_fronts = {
        name: read_csv(path / "combined_pareto.csv") for name, path in ALLOWED_RESULTS.items()
    }
    allowed_shares = {
        name: ny_share_by_objective(path) for name, path in ALLOWED_RESULTS.items()
    }

    for case_id, comparisons in COMPARISONS.items():
        restricted_runs = read_csv(V10 / case_id / "runs.csv")
        restricted_front_rows = read_csv(V10 / case_id / "combined_pareto.csv")
        for comparison, allowed_scenario, allowed_source, restricted_scenario in comparisons:
            allowed_index = {
                (row["scenario"], int(row["seed"])): row
                for row in allowed_runs[allowed_source]
                if row["case_id"] == case_id
            }
            restricted_index = {
                (row["scenario"], int(row["seed"])): row
                for row in restricted_runs
                if row["case_id"] == case_id
            }
            allowed_seeds = {
                seed
                for (scenario, seed), row in allowed_index.items()
                if scenario == allowed_scenario and feasible(row)
            }
            restricted_seeds = {
                seed
                for (scenario, seed), row in restricted_index.items()
                if scenario == restricted_scenario and feasible(row)
            }
            common_seeds = sorted(allowed_seeds & restricted_seeds)

            for source_field, metric, unit in METRICS:
                benefits: list[float] = []
                benefit_percentages: list[float] = []
                raw_allowed_values: list[float] = []
                restricted_values: list[float] = []
                closed_allowed_values: list[float] = []
                for seed in common_seeds:
                    raw_allowed = float(allowed_index[(allowed_scenario, seed)][source_field])
                    restricted = float(restricted_index[(restricted_scenario, seed)][source_field])
                    closed_allowed = min(raw_allowed, restricted)
                    benefit = restricted - closed_allowed
                    benefit_pct = 100.0 * benefit / restricted if restricted else 0.0
                    raw_allowed_values.append(raw_allowed)
                    restricted_values.append(restricted)
                    closed_allowed_values.append(closed_allowed)
                    benefits.append(benefit)
                    benefit_percentages.append(benefit_pct)
                    differences.append(
                        {
                            "case_id": case_id,
                            "comparison": comparison,
                            "allowed_scenario": allowed_scenario,
                            "restricted_scenario": restricted_scenario,
                            "seed": seed,
                            "metric": metric,
                            "unit": unit,
                            "raw_allowed_value": raw_allowed,
                            "restricted_value": restricted,
                            "archive_closed_allowed_value": closed_allowed,
                            "raw_restricted_minus_allowed": restricted - raw_allowed,
                            "benefit_from_allowing_ny_nj": benefit,
                            "benefit_pct_of_restricted_value": benefit_pct,
                        }
                    )

                mean_low = mean_high = median_low = median_high = ""
                mean_pct_low = mean_pct_high = ""
                if benefits:
                    boot_seed = stable_seed(case_id, comparison, metric, "archive_closed")
                    mean_low, mean_high = bootstrap_interval(
                        benefits, statistics.mean, 10000, boot_seed
                    )
                    median_low, median_high = bootstrap_interval(
                        benefits, statistics.median, 10000, boot_seed + 1
                    )
                    mean_pct_low, mean_pct_high = bootstrap_interval(
                        benefit_percentages, statistics.mean, 10000, boot_seed + 2
                    )
                summaries.append(
                    {
                        "case_id": case_id,
                        "comparison": comparison,
                        "allowed_scenario": allowed_scenario,
                        "restricted_scenario": restricted_scenario,
                        "metric": metric,
                        "unit": unit,
                        "allowed_feasible_runs": len(allowed_seeds),
                        "restricted_feasible_runs": len(restricted_seeds),
                        "paired_feasible_runs": len(common_seeds),
                        "feasibility_gain_runs": len(allowed_seeds) - len(restricted_seeds),
                        "raw_allowed_mean": statistics.mean(raw_allowed_values),
                        "restricted_mean": statistics.mean(restricted_values),
                        "archive_closed_allowed_mean": statistics.mean(closed_allowed_values),
                        "mean_benefit_from_allowing_ny_nj": statistics.mean(benefits),
                        "mean_benefit_pct": statistics.mean(benefit_percentages),
                        "median_benefit_from_allowing_ny_nj": statistics.median(benefits),
                        "median_benefit_pct": statistics.median(benefit_percentages),
                        "bootstrap_mean_ci_low": mean_low,
                        "bootstrap_mean_ci_high": mean_high,
                        "bootstrap_median_ci_low": median_low,
                        "bootstrap_median_ci_high": median_high,
                        "bootstrap_mean_pct_ci_low": mean_pct_low,
                        "bootstrap_mean_pct_ci_high": mean_pct_high,
                        "positive_benefit_runs": sum(value > 0 for value in benefits),
                        "zero_benefit_runs": sum(value == 0 for value in benefits),
                        "negative_benefit_runs": sum(value < 0 for value in benefits),
                        "inference_scope": "paired stochastic-search variability under fixed inputs",
                    }
                )

            raw_allowed_front = nondominated_points(
                objective_points(allowed_fronts[allowed_source], case_id, allowed_scenario)
            )
            restricted_front = nondominated_points(
                objective_points(restricted_front_rows, case_id, restricted_scenario)
            )
            archive_closed_front = nondominated_points(raw_allowed_front + restricted_front)
            pooled = raw_allowed_front + restricted_front
            ideal = (min(point[0] for point in pooled), min(point[1] for point in pooled))
            nadir = (max(point[0] for point in pooled), max(point[1] for point in pooled))
            restricted_hv = hypervolume_2d(normalize_points(restricted_front, ideal, nadir))
            closed_hv = hypervolume_2d(normalize_points(archive_closed_front, ideal, nadir))
            front_summaries.append(
                {
                    "case_id": case_id,
                    "comparison": comparison,
                    "allowed_scenario": allowed_scenario,
                    "restricted_scenario": restricted_scenario,
                    "raw_allowed_front_size": len(raw_allowed_front),
                    "restricted_front_size": len(restricted_front),
                    "archive_closed_allowed_front_size": len(archive_closed_front),
                    "shared_ideal_cost_usd": ideal[0],
                    "shared_ideal_delivery_time_h": ideal[1],
                    "shared_nadir_cost_usd": nadir[0],
                    "shared_nadir_delivery_time_h": nadir[1],
                    "restricted_hypervolume_normalized": restricted_hv,
                    "archive_closed_allowed_hypervolume_normalized": closed_hv,
                    "hypervolume_gain": closed_hv - restricted_hv,
                    "hypervolume_gain_pct_of_restricted": (
                        100.0 * (closed_hv - restricted_hv) / restricted_hv
                        if restricted_hv
                        else 0.0
                    ),
                    "reference_point_normalized": "1.1|1.1",
                }
            )

            raw_allowed_set = set(raw_allowed_front)
            restricted_set = set(restricted_front)
            for condition, points in (
                ("restricted", restricted_front),
                ("archive_closed_allowed", archive_closed_front),
            ):
                for point_id, (cost, delivery_time) in enumerate(points, start=1):
                    in_allowed = (cost, delivery_time) in raw_allowed_set
                    in_restricted = (cost, delivery_time) in restricted_set
                    share = 0.0
                    if in_allowed:
                        share = allowed_shares[allowed_source].get(
                            (case_id, allowed_scenario, cost, delivery_time), 0.0
                        )
                    source = (
                        "both" if in_allowed and in_restricted
                        else "raw_allowed" if in_allowed
                        else "restricted"
                    )
                    front_points.append(
                        {
                            "case_id": case_id,
                            "comparison": comparison,
                            "condition": condition,
                            "point_id": point_id,
                            "total_cost_usd": cost,
                            "quantity_weighted_mean_delivery_time_h": delivery_time,
                            "ny_nj_share_pct": share,
                            "point_source": source,
                        }
                    )

    write_csv(OUTPUT / "counterfactual_value_summary.csv", summaries)
    write_csv(OUTPUT / "counterfactual_paired_differences.csv", differences)
    write_csv(OUTPUT / "counterfactual_front_summary.csv", front_summaries)
    write_csv(OUTPUT / "counterfactual_front_points.csv", front_points)


def activation_state(union_size: int, using_size: int, minimum_when_used: float, maximum_when_used: float, threshold: float) -> tuple[float, float, str]:
    minimum_all = minimum_when_used if union_size and using_size == union_size else 0.0
    maximum_all = maximum_when_used if using_size else 0.0
    if maximum_all < threshold:
        state = "no_meaningful_activation"
    elif minimum_all >= threshold:
        state = "front_wide_activation"
    else:
        state = "transition_regime"
    return minimum_all, maximum_all, state


def build_activation_tables() -> None:
    classification: list[dict[str, object]] = []
    for row in read_csv(V09 / "activation_regime.csv"):
        union_size = int(row["union_solutions"])
        using_size = int(row["ny_nj_using_solutions"])
        conditional_mean = float(row["ny_nj_share_mean_pct_when_used"])
        minimum_all, maximum_all, state = activation_state(
            union_size,
            using_size,
            float(row["ny_nj_share_min_pct_when_used"]),
            float(row["ny_nj_share_max_pct_when_used"]),
            10.0,
        )
        classification.append(
            {
                "case_id": row["case_id"],
                "scenario": row["scenario"],
                "remaining_uswc_capacity_pct": 100.0 * float(row["uswc_capacity_multiplier"]),
                "uswc_delay_h": float(row["uswc_delay_h"]),
                "nominal_capacity_days_required": float(row["scenario_demand_feu"])
                / float(row["available_combined_uswc_capacity_feu_per_day"]),
                "balanced_ny_nj_share_pct": float(row["balanced_ny_nj_share_pct"]),
                "union_solutions": union_size,
                "ny_nj_using_solutions": using_size,
                "ny_nj_share_min_pct_all_solutions": minimum_all,
                "ny_nj_share_mean_pct_all_solutions": (
                    conditional_mean * using_size / union_size if union_size else 0.0
                ),
                "ny_nj_share_max_pct_all_solutions": maximum_all,
                "activation_threshold_pct": 10.0,
                "activation_state": state,
            }
        )
    write_csv(OUTPUT / "activation_classification.csv", classification)

    sensitivity: list[dict[str, object]] = []
    selected: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}
    for version, directory in ((6, V06), (7, V07), (8, V08)):
        for row in read_csv(directory / "ny_activation.csv"):
            selected[(row["case_id"], row["scenario"])] = (version, row)
    for (case_id, scenario), (version, row) in sorted(selected.items()):
        union_size = int(row["union_solutions"])
        using_size = int(row["ny_nj_using_solutions"])
        for threshold in (5.0, 10.0, 20.0):
            minimum_all, maximum_all, state = activation_state(
                union_size,
                using_size,
                float(row["ny_nj_share_min_pct_when_used"]),
                float(row["ny_nj_share_max_pct_when_used"]),
                threshold,
            )
            sensitivity.append(
                {
                    "case_id": case_id,
                    "scenario": scenario,
                    "source_version": f"v0.{version}",
                    "threshold_pct": threshold,
                    "union_solutions": union_size,
                    "ny_nj_using_solutions": using_size,
                    "ny_nj_share_min_pct_all_solutions": minimum_all,
                    "ny_nj_share_max_pct_all_solutions": maximum_all,
                    "activation_state": state,
                }
            )
    write_csv(OUTPUT / "activation_threshold_sensitivity.csv", sensitivity)


def build_capacity_table() -> None:
    rows = read_csv(V09 / "capacity_utilization_summary.csv")
    output: list[dict[str, object]] = []
    scenarios = {
        "pilot_baseline",
        "uswc_port_capacity_60pct",
        "uswc_port_capacity_40pct",
        "uswc_port_capacity_35pct",
    }
    for row in rows:
        if (
            row["scenario"] in scenarios
            and "balanced" in row["roles"].lower()
            and row["resource_type"] == "node_day"
        ):
            output.append(
                {
                    "case_id": row["case_id"],
                    "scenario": row["scenario"],
                    "uswc_maximum_node_day_utilization_pct": float(
                        row["uswc_maximum_utilization_pct"]
                    ),
                    "ny_nj_share_pct": 0.0,
                }
            )
    write_csv(OUTPUT / "capacity_saturation_summary.csv", output)


def build_mechanism_table() -> None:
    source = read_csv(
        ROOT / "analysis" / "v11_destination_asymmetry" / "gateway_mechanism_components.csv"
    )
    above = [row for row in source if row["activation_side"] == "above_activation"]
    by_case_gateway = {(row["case_id"], row["gateway"]): row for row in above}
    output: list[dict[str, object]] = []
    for case_id in ("D1", "D2", "D3"):
        west = by_case_gateway[(case_id, "la_lb")]
        east = by_case_gateway[(case_id, "ny_nj")]
        output.append(
            {
                "case_id": case_id,
                "scenario": east["scenario"],
                "ny_nj_share_pct": float(east["gateway_share_pct"]),
                "ny_minus_la_ocean_transit_h": float(east["quantity_weighted_ocean_transit_h"])
                - float(west["quantity_weighted_ocean_transit_h"]),
                "ny_minus_la_waiting_h": float(east["quantity_weighted_waiting_h"])
                - float(west["quantity_weighted_waiting_h"]),
                "ny_minus_la_us_inland_transit_h": float(
                    east["quantity_weighted_us_inland_transit_h"]
                )
                - float(west["quantity_weighted_us_inland_transit_h"]),
                "ny_minus_la_total_lead_time_h": float(
                    east["quantity_weighted_lead_time_h"]
                )
                - float(west["quantity_weighted_lead_time_h"]),
                "ny_minus_la_total_cost_usd_per_feu": float(east["total_cost_usd_per_feu"])
                - float(west["total_cost_usd_per_feu"]),
                "interpretation_scope": "balanced union-front solution; descriptive mechanism decomposition",
            }
        )
    write_csv(OUTPUT / "destination_mechanism_summary.csv", output)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_counterfactual_tables()
    build_activation_tables()
    build_capacity_table()
    build_mechanism_table()
    print(f"Revised manuscript analysis written to {OUTPUT}.")


if __name__ == "__main__":
    main()
