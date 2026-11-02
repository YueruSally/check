#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from china_us_multimodal.config import load_config
from china_us_multimodal.io import load_model_data, select_case

from analyze_scenarios import allocation_summary, read_csv


GRID = {
    (0.60, 0.0): "uswc_port_capacity_60pct",
    (0.40, 0.0): "uswc_port_capacity_40pct",
    (0.35, 0.0): "uswc_port_capacity_35pct",
    (0.60, 168.0): "uswc_capacity_60pct_delay_168h",
    (0.40, 168.0): "uswc_compound_40pct_delay_168h",
    (0.35, 168.0): "uswc_compound_35pct_delay_168h",
    (0.60, 288.0): "uswc_capacity_60pct_delay_288h",
    (0.40, 288.0): "uswc_capacity_40pct_delay_288h",
    (0.35, 288.0): "uswc_capacity_35pct_delay_288h",
}


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preferred_gateway_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    preferred: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["scenario"]))
        role = str(row["roles"]).lower()
        if key not in preferred or "balanced" in role or "knee" in role:
            preferred[key] = row
    return preferred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the v0.9 load-pressure by delay activation regime."
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    solution_rows = read_csv(args.results / "representative_solutions.csv")
    allocation_rows = read_csv(args.results / "representative_allocations.csv")
    gateway_rows, ny_rows = allocation_summary(solution_rows, allocation_rows)
    preferred = preferred_gateway_rows(gateway_rows)
    ny_by_key = {
        (str(row["case_id"]), str(row["scenario"])): row for row in ny_rows
    }

    all_data = load_model_data(args.data)
    base_uswc_capacity = sum(
        all_data.nodes[node_id].capacity_feu_per_day or 0.0
        for node_id in ("la_lb", "seattle_tacoma")
    )
    rows: list[dict[str, object]] = []
    scenarios = sorted({row["scenario"] for row in solution_rows})
    cases = sorted({row["case_id"] for row in solution_rows})
    for case_id in cases:
        case_data = select_case(all_data, case_id)
        base_demand = sum(item.quantity_feu for item in case_data.shipments.values())
        for scenario in scenarios:
            key = (case_id, scenario)
            if key not in preferred or key not in ny_by_key:
                continue
            config = load_config(args.config, scenario)
            demand = base_demand * config.scenario.demand_multiplier
            available_capacity = (
                (all_data.nodes["la_lb"].capacity_feu_per_day or 0.0)
                * config.scenario.port_capacity_multiplier
                * config.scenario.uswc_port_capacity_multiplier
                * config.scenario.la_lb_port_capacity_multiplier
                + (all_data.nodes["seattle_tacoma"].capacity_feu_per_day or 0.0)
                * config.scenario.port_capacity_multiplier
                * config.scenario.uswc_port_capacity_multiplier
                * config.scenario.seattle_tacoma_port_capacity_multiplier
            )
            gateway = preferred[key]
            activation = ny_by_key[key]
            all_solutions_use_ny = (
                int(activation["union_solutions"])
                == int(activation["ny_nj_using_solutions"])
            )
            minimum_ny_share = float(activation["ny_nj_share_min_pct_when_used"])
            rows.append(
                {
                    "case_id": case_id,
                    "scenario": scenario,
                    "demand_multiplier": config.scenario.demand_multiplier,
                    "uswc_capacity_multiplier": config.scenario.uswc_port_capacity_multiplier,
                    "uswc_delay_h": config.scenario.uswc_ocean_departure_delay_h,
                    "base_demand_feu": base_demand,
                    "scenario_demand_feu": demand,
                    "base_combined_uswc_capacity_feu_per_day": base_uswc_capacity,
                    "available_combined_uswc_capacity_feu_per_day": available_capacity,
                    "nominal_load_pressure": demand / available_capacity,
                    "balanced_la_lb_share_pct": gateway["la_lb_share_pct"],
                    "balanced_seattle_tacoma_share_pct": gateway[
                        "seattle_tacoma_share_pct"
                    ],
                    "balanced_ny_nj_share_pct": gateway["ny_nj_share_pct"],
                    "union_solutions": activation["union_solutions"],
                    "ny_nj_using_solutions": activation["ny_nj_using_solutions"],
                    "ny_nj_activation_pct": activation["ny_nj_activation_pct"],
                    "ny_nj_share_min_pct_when_used": minimum_ny_share,
                    "ny_nj_share_mean_pct_when_used": activation[
                        "ny_nj_share_mean_pct_when_used"
                    ],
                    "ny_nj_share_max_pct_when_used": activation[
                        "ny_nj_share_max_pct_when_used"
                    ],
                    "stable_ny_nj_10pct": int(
                        all_solutions_use_ny and minimum_ny_share >= 10.0
                    ),
                }
            )

    fields = [
        "case_id",
        "scenario",
        "demand_multiplier",
        "uswc_capacity_multiplier",
        "uswc_delay_h",
        "base_demand_feu",
        "scenario_demand_feu",
        "base_combined_uswc_capacity_feu_per_day",
        "available_combined_uswc_capacity_feu_per_day",
        "nominal_load_pressure",
        "balanced_la_lb_share_pct",
        "balanced_seattle_tacoma_share_pct",
        "balanced_ny_nj_share_pct",
        "union_solutions",
        "ny_nj_using_solutions",
        "ny_nj_activation_pct",
        "ny_nj_share_min_pct_when_used",
        "ny_nj_share_mean_pct_when_used",
        "ny_nj_share_max_pct_when_used",
        "stable_ny_nj_10pct",
    ]
    rows.sort(
        key=lambda row: (
            str(row["case_id"]),
            float(row["uswc_delay_h"]),
            -float(row["uswc_capacity_multiplier"]),
            float(row["demand_multiplier"]),
        )
    )
    write_csv(args.output / "activation_regime.csv", rows, fields)

    equivalent = [
        row
        for row in rows
        if row["scenario"]
        in {"uswc_port_capacity_40pct", "demand_150pct_uswc_capacity_60pct"}
    ]
    write_csv(args.output / "load_pressure_equivalence.csv", equivalent, fields)
    plot_heatmap(rows, args.output)
    print(f"Wrote activation-regime diagnostics to {args.output}.")


def plot_heatmap(rows: list[dict[str, object]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; activation-regime SVG was skipped.")
        return

    lookup = {
        (str(row["case_id"]), str(row["scenario"])): float(
            row["balanced_ny_nj_share_pct"]
        )
        for row in rows
    }
    cases = sorted({str(row["case_id"]) for row in rows})
    capacities = [0.60, 0.40, 0.35]
    delays = [0.0, 168.0, 288.0]
    figure, axes = plt.subplots(1, len(cases), figsize=(4.8 * len(cases), 4.2), squeeze=False)
    image = None
    for index, case_id in enumerate(cases):
        values = [
            [lookup.get((case_id, GRID[(capacity, delay)]), float("nan")) for capacity in capacities]
            for delay in delays
        ]
        axis = axes[0][index]
        image = axis.imshow(values, vmin=0.0, vmax=100.0, cmap="Blues", aspect="auto")
        axis.set_title(case_id)
        axis.set_xticks(range(len(capacities)), ["60%", "40%", "35%"])
        axis.set_yticks(range(len(delays)), ["0", "168", "288"])
        axis.set_xlabel("Remaining USWC capacity")
        if index == 0:
            axis.set_ylabel("USWC departure delay (h)")
        for row_index, line in enumerate(values):
            for column_index, value in enumerate(line):
                if value == value:
                    axis.text(column_index, row_index, f"{value:.1f}%", ha="center", va="center")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Balanced NY/NJ share (%)")
    figure.subplots_adjust(left=0.07, right=0.90, bottom=0.15, top=0.88, wspace=0.28)
    figure.savefig(output / "activation_regime_heatmap.svg", format="svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
