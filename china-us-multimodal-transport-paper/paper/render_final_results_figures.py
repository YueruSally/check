"""Render final Results figures from v2.1--v2.7 stored outputs only."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CASES = (("D1", "Chicago"), ("D2", "Memphis"), ("D3", "Columbus"))
COLORS = {"D1": "#0072B2", "D2": "#D55E00", "D3": "#009E73"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(figure: plt.Figure, name: str) -> None:
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            OUT / f"{name}.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def clean(axis: plt.Axes, direction: str = "both") -> None:
    axis.grid(axis=direction, color="#dddddd", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


# Figure 1: exact gateway value over the delay grid.
delay = read(ROOT / "analysis/v21_full_structural_core_exact/delay_gateway_value.csv")
delay_additional = read(
    ROOT
    / "analysis/v26_full_structural_inland_bottleneck/inland_capacity_delay_interaction.csv"
)
delay_index = {
    (row["case_id"], row["endpoint"], float(row["delay_h"])): row for row in delay
}
for row in delay_additional:
    delay_h = float(row["delay_h"])
    if abs(float(row["inland_capacity_multiplier"]) - 1.0) > 1e-9:
        continue
    if (row["case_id"], delay_h) not in {("D2", 276.0), ("D3", 216.0)}:
        continue
    delay_index[(row["case_id"], row["endpoint"], delay_h)] = {
        "case_id": row["case_id"],
        "endpoint": row["endpoint"],
        "delay_h": row["delay_h"],
        "cost_avoided_by_allowing_ny_nj_usd": row["cost_avoided_usd"],
        "time_avoided_by_allowing_ny_nj_h": row["time_avoided_h"],
    }
delay = list(delay_index.values())
figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
for case_id, city in CASES:
    cost_rows = sorted(
        (
            row
            for row in delay
            if row["case_id"] == case_id
            and row["endpoint"] == "min_cost_then_time"
        ),
        key=lambda row: float(row["delay_h"]),
    )
    time_rows = sorted(
        (
            row
            for row in delay
            if row["case_id"] == case_id
            and row["endpoint"] == "min_time_then_cost"
        ),
        key=lambda row: float(row["delay_h"]),
    )
    axes[0].plot(
        [float(row["delay_h"]) for row in cost_rows],
        [float(row["cost_avoided_by_allowing_ny_nj_usd"]) / 1e6 for row in cost_rows],
        marker="o",
        linewidth=1.5,
        markersize=4,
        color=COLORS[case_id],
        label=city,
    )
    axes[1].plot(
        [float(row["delay_h"]) for row in time_rows],
        [float(row["time_avoided_by_allowing_ny_nj_h"]) for row in time_rows],
        marker="o",
        linewidth=1.5,
        markersize=4,
        color=COLORS[case_id],
        label=city,
    )
for axis in axes:
    axis.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Additional West Coast sailing delay (hours)")
    axis.set_xticks((0, 72, 120, 168, 240, 288, 336))
    clean(axis)
axes[0].set_title("(a) Minimum-cost endpoint")
axes[0].set_ylabel("Cost avoided by NYNJ access (USD million)")
axes[1].set_title("(b) Minimum-time endpoint")
axes[1].set_ylabel("Mean delivery time avoided (hours)")
axes[1].legend(frameon=False, loc="upper left")
figure.tight_layout()
save(figure, "fig07_final_gateway_value_curves")


# Figure 2: first material-use delay under release dispersion.
release = read(
    ROOT
    / "analysis/v24_full_structural_release_robustness/activation_by_release_instance.csv"
)
patterns = (
    ("single_wave", "Single\nwave"),
    ("two_weekly_waves", "Two weekly\nwaves"),
    ("four_weekly_waves", "Four weekly\nwaves"),
    ("daily_over_four_weeks", "Daily over\nfour weeks"),
)
figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
for axis, (endpoint, title) in zip(
    axes,
    (
        ("min_cost_then_time", "(a) Minimum-cost endpoint"),
        ("min_time_then_cost", "(b) Minimum-time endpoint"),
    ),
):
    for case_offset, (case_id, city) in enumerate(CASES):
        for pattern_index, (pattern, _) in enumerate(patterns):
            values = [
                float(row["first_tested_delay_all_endpoint_equivalent_solutions_at_10pct_h"])
                for row in release
                if row["case_id"] == case_id
                and row["endpoint"] == endpoint
                and row["release_pattern"] == pattern
                and row["first_tested_delay_all_endpoint_equivalent_solutions_at_10pct_h"]
            ]
            # The 216 h Columbus observation was added later as a focused
            # diagnostic and was not evaluated for the multi-wave designs.
            # Figure 2 therefore uses only the common release-design grid;
            # the supplemental 216 h result remains in the text and table note.
            if (
                case_id == "D3"
                and endpoint == "min_cost_then_time"
                and pattern == "single_wave"
            ):
                values = [240.0 if abs(value - 216.0) < 1e-9 else value for value in values]
            x = pattern_index + (case_offset - 1) * 0.18
            axis.scatter(
                [x] * len(values),
                values,
                s=29,
                color=COLORS[case_id],
                edgecolors="white",
                linewidths=0.35,
                alpha=0.85,
                label=city if pattern_index == 0 else None,
            )
    axis.set_xticks(range(len(patterns)), [label for _, label in patterns])
    axis.set_xlabel("Shipment-release design")
    axis.set_title(title)
    clean(axis)
axes[0].set_ylabel("First tested delay with NYNJ share ≥10% (hours)")
axes[1].legend(frameon=False, loc="lower right")
figure.tight_layout()
save(figure, "fig08_release_robustness")


# Figure 3: inland-capacity-by-delay interaction.
interaction = read(
    ROOT
    / "analysis/v26_full_structural_inland_bottleneck/inland_capacity_delay_interaction.csv"
)
capacity_colors = {0.5: "#CC79A7", 1.0: "#0072B2", 1.5: "#009E73"}
figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
for axis, (case_id, city) in zip(axes, (("D2", "Memphis"), ("D3", "Columbus"))):
    for multiplier in (0.5, 1.0, 1.5):
        for endpoint, linestyle, marker in (
            ("min_cost_then_time", "-", "o"),
            ("min_time_then_cost", "--", "s"),
        ):
            rows = sorted(
                (
                    row
                    for row in interaction
                    if row["case_id"] == case_id
                    and abs(float(row["inland_capacity_multiplier"]) - multiplier) < 1e-9
                    and row["endpoint"] == endpoint
                ),
                key=lambda row: float(row["delay_h"]),
            )
            axis.plot(
                [float(row["delay_h"]) for row in rows],
                [100 * float(row["ny_nj_min_share"]) for row in rows],
                color=capacity_colors[multiplier],
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                linewidth=1.4,
            )
    axis.axhline(10, color="#555555", linewidth=0.8, linestyle=":")
    axis.set_title(city)
    axis.set_xlabel("Additional sailing delay (hours)")
    clean(axis)
axes[0].set_ylabel("Proven minimum NYNJ share (%)")
capacity_handles = [
    Line2D([0], [0], color=capacity_colors[value], linewidth=2, label=f"{int(value*100)}% inland capacity")
    for value in (0.5, 1.0, 1.5)
]
endpoint_handles = [
    Line2D([0], [0], color="#333333", linestyle="-", marker="o", label="Minimum cost"),
    Line2D([0], [0], color="#333333", linestyle="--", marker="s", label="Minimum time"),
]
figure.legend(
    handles=capacity_handles + endpoint_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=5,
    frameon=False,
    fontsize=8,
)
figure.tight_layout(rect=(0, 0.08, 1, 1))
save(figure, "fig09_inland_capacity_interaction")


# Figure 4: exact epsilon-constrained internal trade-off points.
epsilon = read(
    ROOT
    / "analysis/v27_full_structural_epsilon_tradeoffs/epsilon_tradeoff_comparison.csv"
)
core = read(
    ROOT / "results/v21_full_structural_core_exact/full_exact_endpoint_summary.csv"
)
figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
last_scatter = None
for axis, (case_id, city, condition_id) in zip(
    axes,
    (("D1", "Chicago", "delay_288h"), ("D3", "Columbus", "delay_240h")),
):
    internal = sorted(
        (row for row in epsilon if row["case_id"] == case_id),
        key=lambda row: float(row["allowed_time_h"]),
    )
    allowed_endpoints = [
        row
        for row in core
        if row["case_id"] == case_id
        and row["condition_id"] == condition_id
        and row["gateway_access"] == "allowed"
    ]
    restricted_endpoints = [
        row
        for row in core
        if row["case_id"] == case_id
        and row["condition_id"] == condition_id
        and row["gateway_access"] == "restricted"
    ]
    allowed_points = [
        (
            float(row["quantity_weighted_mean_delivery_time_h"]),
            float(row["total_cost_usd"]) / 1e6,
            100 * float(row["ny_nj_share"]),
        )
        for row in allowed_endpoints
    ] + [
        (
            float(row["allowed_time_h"]),
            float(row["allowed_cost_usd"]) / 1e6,
            100 * (float(row["ny_nj_min_share"]) + float(row["ny_nj_max_share"])) / 2,
        )
        for row in internal
    ]
    allowed_points.sort()
    axis.plot(
        [point[0] for point in allowed_points],
        [point[1] for point in allowed_points],
        color="#666666",
        linewidth=1.0,
        zorder=1,
    )
    last_scatter = axis.scatter(
        [point[0] for point in allowed_points],
        [point[1] for point in allowed_points],
        c=[point[2] for point in allowed_points],
        cmap="viridis",
        vmin=0,
        vmax=70,
        s=48,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
        label="With NYNJ",
    )
    restricted_points = [
        (
            float(row["quantity_weighted_mean_delivery_time_h"]),
            float(row["total_cost_usd"]) / 1e6,
        )
        for row in restricted_endpoints
    ] + [
        (float(row["restricted_time_h"]), float(row["restricted_cost_usd"]) / 1e6)
        for row in internal
        if int(row["restricted_proven"]) == 1
    ]
    restricted_points = sorted(set(restricted_points))
    axis.plot(
        [point[0] for point in restricted_points],
        [point[1] for point in restricted_points],
        color="#999999",
        linewidth=0.9,
        linestyle="--",
        zorder=1,
    )
    axis.scatter(
        [point[0] for point in restricted_points],
        [point[1] for point in restricted_points],
        color="#555555",
        marker="x",
        s=35,
        linewidths=1.0,
        zorder=2,
        label="Without NYNJ",
    )
    axis.set_title(city)
    axis.set_xlabel("Weighted mean delivery time (hours)")
    axis.set_ylabel("Total operational cost (USD million)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    axis.legend(frameon=False, fontsize=8)
    clean(axis)
# Reserve a dedicated margin for the colour bar.  Creating it from the two
# subplot axes and then shrinking the figure caused the bar to overlap the
# Columbus legend and the rightmost observations in the publication export.
figure.subplots_adjust(wspace=0.28, right=0.84, bottom=0.16, top=0.90)
if last_scatter is not None:
    colorbar_axis = figure.add_axes((0.87, 0.16, 0.018, 0.74))
    colorbar = figure.colorbar(last_scatter, cax=colorbar_axis)
    colorbar.set_label("NYNJ share in solution with NYNJ (%)")
save(figure, "fig10_exact_epsilon_tradeoffs")

print(f"Final Results figures written to {OUT}.")
