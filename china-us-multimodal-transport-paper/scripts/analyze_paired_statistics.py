#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable

BASELINE = "pilot_baseline"
METRICS = (
    ("min_cost_usd", "total_cost_usd", "USD"),
    (
        "min_weighted_mean_delivery_time_h",
        "weighted_mean_delivery_time_h",
        "h",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_sign_test(differences: list[float], tolerance: float = 1e-12) -> tuple[int, int, int, float]:
    positive = sum(value > tolerance for value in differences)
    negative = sum(value < -tolerance for value in differences)
    ties = len(differences) - positive - negative
    trials = positive + negative
    if trials == 0:
        return positive, negative, ties, 1.0
    smaller = min(positive, negative)
    lower_tail = sum(math.comb(trials, index) for index in range(smaller + 1))
    p_value = min(1.0, 2.0 * lower_tail / (2**trials))
    return positive, negative, ties, p_value


def paired_cohens_dz(differences: list[float]) -> float | None:
    if len(differences) < 2:
        return None
    standard_deviation = statistics.stdev(differences)
    if standard_deviation == 0:
        return 0.0 if statistics.mean(differences) == 0 else None
    return statistics.mean(differences) / standard_deviation


def bootstrap_interval(
    values: list[float],
    statistic: Callable[[list[float]], float],
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    generator = random.Random(seed)
    size = len(values)
    estimates = [
        statistic([values[generator.randrange(size)] for _ in range(size)])
        for _ in range(repetitions)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    running_maximum = 0.0
    ordered = sorted(range(len(p_values)), key=p_values.__getitem__)
    total = len(p_values)
    for rank, original_index in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[original_index])
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = running_maximum
    return adjusted


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def valid_run(row: dict[str, str]) -> bool:
    return int(row.get("feasible_population", "0") or 0) > 0


def analyze(
    rows: list[dict[str, str]],
    baseline_name: str,
    bootstrap_repetitions: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["case_id"], row["scenario"], int(row["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate run row for {key}")
        indexed[key] = row

    cases = sorted({row["case_id"] for row in rows})
    scenarios = sorted(
        {row["scenario"] for row in rows if row["scenario"] != baseline_name}
    )
    summary_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []

    for case_id in cases:
        baseline_seeds = {
            int(row["seed"])
            for row in rows
            if row["case_id"] == case_id
            and row["scenario"] == baseline_name
            and valid_run(row)
        }
        if not baseline_seeds:
            raise ValueError(f"no feasible baseline runs for {case_id}")

        for scenario in scenarios:
            scenario_seeds = {
                int(row["seed"])
                for row in rows
                if row["case_id"] == case_id
                and row["scenario"] == scenario
                and valid_run(row)
            }
            if scenario_seeds != baseline_seeds:
                missing = sorted(baseline_seeds - scenario_seeds)
                extra = sorted(scenario_seeds - baseline_seeds)
                raise ValueError(
                    f"seed mismatch for {case_id}/{scenario}: "
                    f"missing={missing}, extra={extra}"
                )

            for source_field, metric_name, unit in METRICS:
                baseline_values: list[float] = []
                scenario_values: list[float] = []
                differences: list[float] = []
                percentage_differences: list[float] = []

                for seed in sorted(baseline_seeds):
                    baseline_row = indexed[(case_id, baseline_name, seed)]
                    scenario_row = indexed[(case_id, scenario, seed)]
                    baseline_value = float(baseline_row[source_field])
                    scenario_value = float(scenario_row[source_field])
                    difference = scenario_value - baseline_value
                    percentage = (
                        100.0 * difference / baseline_value
                        if baseline_value != 0
                        else math.nan
                    )
                    baseline_values.append(baseline_value)
                    scenario_values.append(scenario_value)
                    differences.append(difference)
                    percentage_differences.append(percentage)
                    difference_rows.append(
                        {
                            "case_id": case_id,
                            "scenario": scenario,
                            "seed": seed,
                            "metric": metric_name,
                            "unit": unit,
                            "baseline_value": baseline_value,
                            "scenario_value": scenario_value,
                            "paired_difference": difference,
                            "paired_difference_pct": percentage,
                        }
                    )

                positive, negative, ties, p_value = exact_sign_test(differences)
                bootstrap_seed = stable_seed(
                    case_id,
                    scenario,
                    metric_name,
                    bootstrap_repetitions,
                )
                mean_low, mean_high = bootstrap_interval(
                    differences,
                    statistics.mean,
                    bootstrap_repetitions,
                    bootstrap_seed,
                )
                median_low, median_high = bootstrap_interval(
                    differences,
                    statistics.median,
                    bootstrap_repetitions,
                    bootstrap_seed + 1,
                )
                summary_rows.append(
                    {
                        "case_id": case_id,
                        "scenario": scenario,
                        "metric": metric_name,
                        "unit": unit,
                        "paired_runs": len(differences),
                        "baseline_mean": statistics.mean(baseline_values),
                        "scenario_mean": statistics.mean(scenario_values),
                        "mean_paired_difference": statistics.mean(differences),
                        "mean_paired_difference_pct": statistics.mean(
                            percentage_differences
                        ),
                        "median_paired_difference": statistics.median(differences),
                        "median_paired_difference_pct": statistics.median(
                            percentage_differences
                        ),
                        "bootstrap_mean_ci_low": mean_low,
                        "bootstrap_mean_ci_high": mean_high,
                        "bootstrap_median_ci_low": median_low,
                        "bootstrap_median_ci_high": median_high,
                        "paired_cohens_dz": paired_cohens_dz(differences),
                        "positive_differences": positive,
                        "negative_differences": negative,
                        "ties": ties,
                        "exact_sign_test_p_value": p_value,
                        "holm_adjusted_p_value": None,
                    }
                )

    adjusted = holm_adjust(
        [float(row["exact_sign_test_p_value"]) for row in summary_rows]
    )
    for row, adjusted_p_value in zip(summary_rows, adjusted):
        row["holm_adjusted_p_value"] = adjusted_p_value
    return summary_rows, difference_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pair scenario runs to the baseline by case and seed, then calculate "
            "distribution-free tests, effect sizes and deterministic bootstrap intervals."
        )
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()

    if args.bootstrap_repetitions < 1000:
        parser.error("--bootstrap-repetitions must be at least 1000")
    runs_path = args.results / "runs.csv"
    if not runs_path.is_file():
        parser.error(f"missing result file: {runs_path}")

    summary_rows, difference_rows = analyze(
        read_csv(runs_path),
        args.baseline,
        args.bootstrap_repetitions,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output / "paired_statistics.csv",
        summary_rows,
        [
            "case_id",
            "scenario",
            "metric",
            "unit",
            "paired_runs",
            "baseline_mean",
            "scenario_mean",
            "mean_paired_difference",
            "mean_paired_difference_pct",
            "median_paired_difference",
            "median_paired_difference_pct",
            "bootstrap_mean_ci_low",
            "bootstrap_mean_ci_high",
            "bootstrap_median_ci_low",
            "bootstrap_median_ci_high",
            "paired_cohens_dz",
            "positive_differences",
            "negative_differences",
            "ties",
            "exact_sign_test_p_value",
            "holm_adjusted_p_value",
        ],
    )
    write_csv(
        args.output / "paired_differences.csv",
        difference_rows,
        [
            "case_id",
            "scenario",
            "seed",
            "metric",
            "unit",
            "baseline_value",
            "scenario_value",
            "paired_difference",
            "paired_difference_pct",
        ],
    )
    manifest = {
        "baseline": args.baseline,
        "pairing_keys": ["case_id", "seed"],
        "metrics": [metric for _, metric, _ in METRICS],
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "confidence_interval": "95% percentile bootstrap",
        "test": "two-sided exact sign test; zero differences omitted",
        "effect_size": "paired Cohen's dz",
        "multiplicity": "Holm adjustment across every case/scenario/metric comparison",
        "randomness": "deterministic SHA-256-derived bootstrap seeds",
    }
    (args.output / "paired_statistics_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "
",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(summary_rows)} paired summaries and "
        f"{len(difference_rows)} seed-level differences to {args.output}."
    )


if __name__ == "__main__":
    main()
