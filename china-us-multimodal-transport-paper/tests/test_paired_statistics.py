from scripts.analyze_paired_statistics import (
    analyze,
    exact_sign_test,
    holm_adjust,
    paired_cohens_dz,
)


def test_exact_sign_test_counts_ties_and_uses_two_sided_probability():
    positive, negative, ties, p_value = exact_sign_test([1, 2, 3, 4, 0])
    assert (positive, negative, ties) == (4, 0, 1)
    assert p_value == 0.125


def test_holm_adjustment_is_monotone_in_rank_order():
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_paired_cohens_dz_uses_within_seed_differences():
    assert paired_cohens_dz([1.0, 2.0, 3.0]) == 2.0


def test_analysis_pairs_each_scenario_to_baseline_by_case_and_seed():
    rows = []
    for seed, baseline_cost, scenario_cost in [
        (11, 100.0, 110.0),
        (12, 120.0, 126.0),
    ]:
        rows.extend(
            [
                {
                    "case_id": "D1",
                    "scenario": "pilot_baseline",
                    "seed": str(seed),
                    "feasible_population": "48",
                    "min_cost_usd": str(baseline_cost),
                    "min_weighted_mean_delivery_time_h": "20",
                },
                {
                    "case_id": "D1",
                    "scenario": "stress",
                    "seed": str(seed),
                    "feasible_population": "48",
                    "min_cost_usd": str(scenario_cost),
                    "min_weighted_mean_delivery_time_h": "25",
                },
            ]
        )

    summaries, differences = analyze(rows, "pilot_baseline", 1000)

    assert len(summaries) == 2
    assert len(differences) == 4
    cost = next(row for row in summaries if row["metric"] == "total_cost_usd")
    assert cost["paired_runs"] == 2
    assert cost["mean_paired_difference"] == 8.0
    assert cost["positive_differences"] == 2
    assert cost["negative_differences"] == 0
