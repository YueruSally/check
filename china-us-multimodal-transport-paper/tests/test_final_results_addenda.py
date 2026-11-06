from scripts.run_final_results_addenda import (
    absolute_recovery_rows,
    all_single_wave_criterion_rows,
    core_validation_id,
    structural_library_variant,
    supplemental_216_rows,
)


def test_selectable_service_library_variant_and_validation_id():
    assert structural_library_variant(0) == "structural_full"
    assert structural_library_variant(2) == "structural_full_service_plus_2"
    assert core_validation_id("D1", "delay_288h", "min_cost_then_time", 2) == (
        "D1_delay_288h_restricted_min_cost_then_time_"
        "structural_full_service_plus_2"
    )


def test_supplemental_share_rows_normalise_endpoint_region():
    source = "supplemental.csv"
    rows = supplemental_216_rows(
        [
            {
                "case_id": "D3",
                "condition_id": "delay_216h_inland_us-d-03_capacity_100pct",
                "endpoint_region": "min_cost_then_time",
                "ny_nj_min_share": "0.1596",
            },
            {
                "case_id": "D3",
                "condition_id": "delay_216h_inland_us-d-03_capacity_100pct",
                "endpoint_region": "min_time_then_cost",
                "ny_nj_min_share": "0.0",
            },
        ],
        source,
    )

    assert [row["endpoint"] for row in rows] == [
        "min_cost_then_time",
        "min_time_then_cost",
    ]
    assert rows[0]["meets_10pct"] == 1
    assert rows[0]["meets_20pct"] == 0
    combined = all_single_wave_criterion_rows([], rows)
    cost_10pct = next(
        row
        for row in combined
        if row["case_id"] == "D3"
        and row["endpoint"] == "min_cost_then_time"
        and row["criterion_share"] == 0.10
    )
    assert cost_10pct["first_tested_single_wave_delay_h"] == 216.0


def test_absolute_recovery_uses_current_core_baseline():
    recovery_rows = [
        {
            "case_id": case_id,
            "condition_id": "delay_288h",
            "endpoint": endpoint,
            "metric": metric,
        }
        for case_id in ("D1", "D2")
        for endpoint, metric in (
            ("min_cost_then_time", "cost"),
            ("min_time_then_cost", "time"),
        )
    ]
    inland_endpoints = [
        {
            "case_id": "D3",
            "condition_id": "delay_216h_inland_us-d-03_capacity_100pct",
            "gateway_access": gateway_access,
            "endpoint": endpoint,
            "total_cost_usd": cost,
            "quantity_weighted_mean_delivery_time_h": time,
        }
        for gateway_access, endpoint, cost, time in (
            ("allowed", "min_cost_then_time", "130", "15"),
            ("restricted", "min_cost_then_time", "150", "14"),
            ("allowed", "min_time_then_cost", "170", "12"),
            ("restricted", "min_time_then_cost", "180", "13"),
        )
    ]
    core_endpoints = [
        {
            "case_id": "D3",
            "condition_id": "delay_0h",
            "gateway_access": "restricted",
            "endpoint": "min_cost_then_time",
            "total_cost_usd": "100",
            "quantity_weighted_mean_delivery_time_h": "10",
        },
        {
            "case_id": "D3",
            "condition_id": "delay_0h",
            "gateway_access": "restricted",
            "endpoint": "min_time_then_cost",
            "total_cost_usd": "110",
            "quantity_weighted_mean_delivery_time_h": "10",
        },
    ]

    rows = absolute_recovery_rows(
        recovery_rows, inland_endpoints, core_endpoints
    )
    d3_cost = next(
        row
        for row in rows
        if row["case_id"] == "D3" and row["metric"] == "cost"
    )
    d3_time = next(
        row
        for row in rows
        if row["case_id"] == "D3" and row["metric"] == "time"
    )
    assert d3_cost["baseline_value"] == 100.0
    assert d3_cost["disruption_loss"] == 50.0
    assert d3_time["baseline_value"] == 10.0
    assert d3_time["disruption_loss"] == 3.0
