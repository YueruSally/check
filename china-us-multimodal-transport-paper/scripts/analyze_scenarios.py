#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SCENARIO_ORDER = [
    "pilot_baseline",
    "uswc_port_capacity_70pct",
    "uswc_port_capacity_65pct",
    "uswc_port_capacity_60pct",
    "la_lb_port_capacity_50pct",
    "seattle_tacoma_port_capacity_50pct",
    "la_lb_departure_delay_72h",
    "seattle_tacoma_departure_delay_72h",
    "uswc_departure_delay_96h",
    "uswc_departure_delay_120h",
    "uswc_departure_delay_168h",
    "uswc_compound_50pct_delay_120h",
]
GATEWAY_ARCS = {
    "la_lb": {"sea-01", "sea-02"},
    "seattle_tacoma": {"sea-03", "sea-04"},
    "ny_nj": {"sea-05", "sea-06"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else default


def first_value(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        if row.get(name):
            return row[name]
    return ""


def gateway_for_path(path_text: str) -> str:
    for gateway, arc_ids in GATEWAY_ARCS.items():
        if any(arc_id in path_text for arc_id in arc_ids):
            return gateway
    return "unclassified"


def build_scenario_effects(aggregate_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    baseline = {
        row["case_id"]: row
        for row in aggregate_rows
        if row["scenario"] == "pilot_baseline"
    }
    output: list[dict[str, object]] = []
    for row in aggregate_rows:
        base = baseline[row["case_id"]]
        base_cost = number(base, "cost_median_usd")
        base_time = number(base, "weighted_mean_delivery_time_median_h")
        cost = number(row, "cost_median_usd")
        delivery_time = number(row, "weighted_mean_delivery_time_median_h")
        output.append(
            {
                "case_id": row["case_id"],
                "scenario": row["scenario"],
                "runs": row["runs"],
                "successful_runs": row["successful_runs"],
                "success_rate": row["success_rate"],
                "union_front_size": row["union_front_size"],
                "cost_median_usd": cost,
                "cost_change_pct_vs_baseline": 100.0 * (cost / base_cost - 1.0),
                "weighted_mean_delivery_time_median_h": delivery_time,
                "delivery_time_change_h_vs_baseline": delivery_time - base_time,
                "best_cost_usd": row["best_cost_usd"],
                "best_weighted_mean_delivery_time_h": row[
                    "best_weighted_mean_delivery_time_h"
                ],
            }
        )
    order = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    output.sort(key=lambda item: (item["case_id"], order.get(str(item["scenario"]), 999)))
    return output


def allocation_summary(
    solution_rows: list[dict[str, str]],
    allocation_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metadata = {row["solution_id"]: row for row in solution_rows}
    gateway_quantities: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    shipment_paths: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in allocation_rows:
        solution_id = row["solution_id"]
        quantity = number(
            row,
            first_value(
                {name: name for name in row if row.get(name)},
                ("quantity_feu", "allocated_quantity_feu", "allocation_feu"),
            ),
        )
        path_text = first_value(row, ("arc_ids", "path_arc_ids", "path"))
        gateway = gateway_for_path(path_text)
        gateway_quantities[solution_id][gateway] += quantity
        shipment_id = first_value(row, ("shipment_id", "batch_id"))
        shipment_paths[(solution_id, shipment_id)].append(quantity)

    minimum_path_share: dict[str, float] = {}
    for (solution_id, _), quantities in shipment_paths.items():
        total = sum(quantities)
        if total <= 0:
            continue
        share = min(quantities) / total
        if solution_id not in minimum_path_share:
            minimum_path_share[solution_id] = share
        else:
            minimum_path_share[solution_id] = min(
                minimum_path_share[solution_id], share
            )

    gateway_rows: list[dict[str, object]] = []
    for solution_id, meta in metadata.items():
        if not meta.get("roles", "").strip():
            continue
        quantities = gateway_quantities[solution_id]
        total = sum(quantities.values())
        gateway_rows.append(
            {
                "solution_id": solution_id,
                "case_id": meta["case_id"],
                "scenario": meta["scenario"],
                "roles": meta["roles"],
                "source_seed": meta["source_seed"],
                "total_cost_usd": meta["total_cost_usd"],
                "quantity_weighted_mean_delivery_time_h": meta[
                    "quantity_weighted_mean_delivery_time_h"
                ],
                "makespan_h": meta["makespan_h"],
                "la_lb_share_pct": 100.0 * quantities["la_lb"] / total if total else 0.0,
                "seattle_tacoma_share_pct": (
                    100.0 * quantities["seattle_tacoma"] / total if total else 0.0
                ),
                "ny_nj_share_pct": 100.0 * quantities["ny_nj"] / total if total else 0.0,
                "unclassified_share_pct": (
                    100.0 * quantities["unclassified"] / total if total else 0.0
                ),
                "minimum_path_share_pct": 100.0 * minimum_path_share.get(
                    solution_id, 0.0
                ),
            }
        )

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for solution_id, meta in metadata.items():
        groups[(meta["case_id"], meta["scenario"])].append(solution_id)

    ny_rows: list[dict[str, object]] = []
    for (case_id, scenario), solution_ids in groups.items():
        shares: list[float] = []
        for solution_id in solution_ids:
            quantities = gateway_quantities[solution_id]
            total = sum(quantities.values())
            share = quantities["ny_nj"] / total if total else 0.0
            if share > 0:
                shares.append(share)
        ny_rows.append(
            {
                "case_id": case_id,
                "scenario": scenario,
                "union_solutions": len(solution_ids),
                "ny_nj_using_solutions": len(shares),
                "ny_nj_activation_pct": 100.0 * len(shares) / len(solution_ids),
                "ny_nj_share_min_pct_when_used": 100.0 * min(shares) if shares else 0.0,
                "ny_nj_share_mean_pct_when_used": (
                    100.0 * sum(shares) / len(shares) if shares else 0.0
                ),
                "ny_nj_share_max_pct_when_used": 100.0 * max(shares) if shares else 0.0,
            }
        )
    order = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    gateway_rows.sort(
        key=lambda item: (item["case_id"], order.get(str(item["scenario"]), 999))
    )
    ny_rows.sort(key=lambda item: (item["case_id"], order.get(str(item["scenario"]), 999)))
    return gateway_rows, ny_rows


def plot_effects(
    rows: list[dict[str, object]],
    gateway_rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV tables were written and SVG plots were skipped.")
        return

    labels = [name.replace("uswc_", "").replace("_", " ") for name in SCENARIO_ORDER]
    cases = sorted({str(row["case_id"]) for row in rows})
    by_key = {(str(row["case_id"]), str(row["scenario"])): row for row in rows}

    for field, ylabel, filename in [
        ("cost_change_pct_vs_baseline", "Median cost change vs baseline (%)", "scenario_cost_change.svg"),
        ("delivery_time_change_h_vs_baseline", "Median delivery-time change vs baseline (h)", "scenario_time_change.svg"),
    ]:
        figure, axis = plt.subplots(figsize=(13, 5.5))
        for case_id in cases:
            values = [
                float(by_key[(case_id, scenario)][field])
                for scenario in SCENARIO_ORDER
            ]
            axis.plot(range(len(labels)), values, marker="o", linewidth=1.8, label=case_id)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / filename, format="svg")
        plt.close(figure)

    preferred: dict[tuple[str, str], dict[str, object]] = {}
    for row in gateway_rows:
        key = (str(row["case_id"]), str(row["scenario"]))
        role = str(row["roles"]).lower()
        if key not in preferred or "balanced" in role or "knee" in role:
            preferred[key] = row
    plot_rows = [preferred[key] for key in sorted(preferred)]
    xlabels = [
        str(row["case_id"]) + " / " + str(row["scenario"]).replace("uswc_", "")
        for row in plot_rows
    ]
    left = [float(row["la_lb_share_pct"]) for row in plot_rows]
    middle = [float(row["seattle_tacoma_share_pct"]) for row in plot_rows]
    right = [float(row["ny_nj_share_pct"]) for row in plot_rows]
    figure, axis = plt.subplots(figsize=(18, 6))
    positions = range(len(plot_rows))
    axis.bar(positions, left, label="LA/LB")
    axis.bar(positions, middle, bottom=left, label="Seattle/Tacoma")
    axis.bar(
        positions,
        right,
        bottom=[a + b for a, b in zip(left, middle)],
        label="NY/NJ",
    )
    axis.set_xticks(list(positions), xlabels, rotation=75, ha="right", fontsize=7)
    axis.set_ylabel("Allocated FEU share (%)")
    axis.set_ylim(0, 100)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(output_dir / "balanced_gateway_shares.svg", format="svg")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create v0.6 non-carbon scenario tables and paper-ready SVG figures."
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    required = [
        "aggregate.csv",
        "representative_solutions.csv",
        "representative_allocations.csv",
    ]
    missing = [name for name in required if not (args.results / name).is_file()]
    if missing:
        parser.error("missing result files: " + ", ".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    aggregate_rows = read_csv(args.results / "aggregate.csv")
    solution_rows = read_csv(args.results / "representative_solutions.csv")
    allocation_rows = read_csv(args.results / "representative_allocations.csv")

    effects = build_scenario_effects(aggregate_rows)
    gateway_rows, ny_rows = allocation_summary(solution_rows, allocation_rows)

    write_csv(
        args.output / "scenario_effects.csv",
        effects,
        [
            "case_id",
            "scenario",
            "runs",
            "successful_runs",
            "success_rate",
            "union_front_size",
            "cost_median_usd",
            "cost_change_pct_vs_baseline",
            "weighted_mean_delivery_time_median_h",
            "delivery_time_change_h_vs_baseline",
            "best_cost_usd",
            "best_weighted_mean_delivery_time_h",
        ],
    )
    write_csv(
        args.output / "gateway_shares.csv",
        gateway_rows,
        [
            "solution_id",
            "case_id",
            "scenario",
            "roles",
            "source_seed",
            "total_cost_usd",
            "quantity_weighted_mean_delivery_time_h",
            "makespan_h",
            "la_lb_share_pct",
            "seattle_tacoma_share_pct",
            "ny_nj_share_pct",
            "unclassified_share_pct",
            "minimum_path_share_pct",
        ],
    )
    write_csv(
        args.output / "ny_activation.csv",
        ny_rows,
        [
            "case_id",
            "scenario",
            "union_solutions",
            "ny_nj_using_solutions",
            "ny_nj_activation_pct",
            "ny_nj_share_min_pct_when_used",
            "ny_nj_share_mean_pct_when_used",
            "ny_nj_share_max_pct_when_used",
        ],
    )

    plot_effects(effects, gateway_rows, args.output)
    print(f"Analysis written to {args.output}.")


if __name__ == "__main__":
    main()
