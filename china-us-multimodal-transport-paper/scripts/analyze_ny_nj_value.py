#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from analyze_paired_statistics import (
    bootstrap_interval,
    exact_sign_test,
    holm_adjust,
    paired_cohens_dz,
    stable_seed,
)


COMPARISONS = {
    "D1": (
        (
            "delay_activation",
            "uswc_departure_delay_288h",
            "v08",
            "uswc_departure_delay_288h_no_ny_nj",
        ),
        (
            "extreme_compound",
            "uswc_compound_30pct_delay_168h",
            "v07",
            "uswc_compound_30pct_delay_168h_no_ny_nj",
        ),
    ),
    "D2": (
        (
            "delay_activation",
            "uswc_departure_delay_264h",
            "v08",
            "uswc_departure_delay_264h_no_ny_nj",
        ),
        (
            "extreme_compound",
            "uswc_compound_30pct_delay_168h",
            "v07",
            "uswc_compound_30pct_delay_168h_no_ny_nj",
        ),
    ),
    "D3": (
        (
            "delay_activation",
            "uswc_departure_delay_240h",
            "v07",
            "uswc_departure_delay_240h_no_ny_nj",
        ),
        (
            "extreme_compound",
            "uswc_compound_30pct_delay_168h",
            "v07",
            "uswc_compound_30pct_delay_168h_no_ny_nj",
        ),
    ),
}
METRICS = (
    ("min_cost_usd", "cost_avoided_by_ny_nj", "USD"),
    (
        "min_weighted_mean_delivery_time_h",
        "delivery_time_avoided_by_ny_nj",
        "h",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def feasible(row: dict[str, str]) -> bool:
    return int(row.get("feasible_population", "0") or 0) > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify the counterfactual value of allowing the NY/NJ gateway."
    )
    parser.add_argument(
        "results", type=Path, help="Root containing the new disabled-gateway D1/D2/D3 folders"
    )
    parser.add_argument("--v07", type=Path, required=True)
    parser.add_argument("--v08", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1000:
        parser.error("--bootstrap-repetitions must be at least 1000")

    summary: list[dict[str, object]] = []
    differences: list[dict[str, object]] = []
    allowed_sources = {
        "v07": read_csv(args.v07 / "runs.csv"),
        "v08": read_csv(args.v08 / "runs.csv"),
    }
    for case_id, comparisons in COMPARISONS.items():
        disabled_rows = read_csv(args.results / case_id / "runs.csv")
        disabled_indexed = {
            (row["scenario"], int(row["seed"])): row
            for row in disabled_rows
            if row["case_id"] == case_id
        }
        for (
            comparison_name,
            allowed_scenario,
            allowed_source,
            disabled_scenario,
        ) in comparisons:
            allowed_indexed = {
                (row["scenario"], int(row["seed"])): row
                for row in allowed_sources[allowed_source]
                if row["case_id"] == case_id
            }
            allowed_seeds = {
                seed
                for (scenario, seed), row in allowed_indexed.items()
                if scenario == allowed_scenario and feasible(row)
            }
            disabled_seeds = {
                seed
                for (scenario, seed), row in disabled_indexed.items()
                if scenario == disabled_scenario and feasible(row)
            }
            common_seeds = sorted(allowed_seeds & disabled_seeds)

            for source_field, metric, unit in METRICS:
                paired_values: list[float] = []
                paired_percentages: list[float] = []
                allowed_values: list[float] = []
                disabled_values: list[float] = []
                for seed in common_seeds:
                    allowed = float(
                        allowed_indexed[(allowed_scenario, seed)][source_field]
                    )
                    disabled = float(
                        disabled_indexed[(disabled_scenario, seed)][source_field]
                    )
                    benefit = disabled - allowed
                    benefit_pct = 100.0 * benefit / disabled if disabled else math.nan
                    allowed_values.append(allowed)
                    disabled_values.append(disabled)
                    paired_values.append(benefit)
                    paired_percentages.append(benefit_pct)
                    differences.append(
                        {
                            "case_id": case_id,
                            "comparison": comparison_name,
                            "allowed_scenario": allowed_scenario,
                            "allowed_source_results": allowed_source,
                            "disabled_scenario": disabled_scenario,
                            "seed": seed,
                            "metric": metric,
                            "unit": unit,
                            "allowed_value": allowed,
                            "disabled_value": disabled,
                            "benefit_from_allowing_ny_nj": benefit,
                            "benefit_pct_of_disabled_value": benefit_pct,
                        }
                    )

                positive = negative = ties = 0
                p_value = 1.0
                mean_low = mean_high = median_low = median_high = ""
                effect_size: float | None = None
                if paired_values:
                    positive, negative, ties, p_value = exact_sign_test(paired_values)
                    bootstrap_seed = stable_seed(
                        case_id,
                        comparison_name,
                        metric,
                        args.bootstrap_repetitions,
                    )
                    mean_low, mean_high = bootstrap_interval(
                        paired_values,
                        statistics.mean,
                        args.bootstrap_repetitions,
                        bootstrap_seed,
                    )
                    median_low, median_high = bootstrap_interval(
                        paired_values,
                        statistics.median,
                        args.bootstrap_repetitions,
                        bootstrap_seed + 1,
                    )
                    effect_size = paired_cohens_dz(paired_values)

                summary.append(
                    {
                        "case_id": case_id,
                        "comparison": comparison_name,
                        "allowed_scenario": allowed_scenario,
                        "allowed_source_results": allowed_source,
                        "disabled_scenario": disabled_scenario,
                        "metric": metric,
                        "unit": unit,
                        "planned_runs_per_condition": len(
                            {
                                seed
                                for scenario, seed in allowed_indexed
                                if scenario == allowed_scenario
                            }
                        ),
                        "allowed_feasible_runs": len(allowed_seeds),
                        "disabled_feasible_runs": len(disabled_seeds),
                        "feasibility_gain_runs": len(allowed_seeds) - len(disabled_seeds),
                        "paired_feasible_runs": len(common_seeds),
                        "allowed_mean": statistics.mean(allowed_values)
                        if allowed_values
                        else "",
                        "disabled_mean": statistics.mean(disabled_values)
                        if disabled_values
                        else "",
                        "mean_benefit_from_allowing_ny_nj": statistics.mean(paired_values)
                        if paired_values
                        else "",
                        "mean_benefit_pct": statistics.mean(paired_percentages)
                        if paired_percentages
                        else "",
                        "median_benefit_from_allowing_ny_nj": statistics.median(paired_values)
                        if paired_values
                        else "",
                        "median_benefit_pct": statistics.median(paired_percentages)
                        if paired_percentages
                        else "",
                        "bootstrap_mean_ci_low": mean_low,
                        "bootstrap_mean_ci_high": mean_high,
                        "bootstrap_median_ci_low": median_low,
                        "bootstrap_median_ci_high": median_high,
                        "paired_cohens_dz": effect_size if effect_size is not None else "",
                        "positive_benefits": positive,
                        "negative_benefits": negative,
                        "ties": ties,
                        "exact_sign_test_p_value": p_value,
                        "holm_adjusted_p_value": "",
                    }
                )

    tested = [index for index, row in enumerate(summary) if row["paired_feasible_runs"]]
    adjusted = holm_adjust(
        [float(summary[index]["exact_sign_test_p_value"]) for index in tested]
    )
    for index, value in zip(tested, adjusted):
        summary[index]["holm_adjusted_p_value"] = value

    summary_fields = list(summary[0]) if summary else []
    difference_fields = list(differences[0]) if differences else [
        "case_id",
        "comparison",
        "allowed_scenario",
        "allowed_source_results",
        "disabled_scenario",
        "seed",
        "metric",
        "unit",
        "allowed_value",
        "disabled_value",
        "benefit_from_allowing_ny_nj",
        "benefit_pct_of_disabled_value",
    ]
    write_csv(args.output / "ny_nj_value_summary.csv", summary, summary_fields)
    write_csv(args.output / "ny_nj_value_paired_differences.csv", differences, difference_fields)
    print(
        f"Wrote {len(summary)} NY/NJ value summaries and {len(differences)} "
        f"paired differences to {args.output}."
    )


if __name__ == "__main__":
    main()
