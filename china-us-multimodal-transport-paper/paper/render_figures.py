"""Render publication figures from stored outputs; no optimisation is run."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)
SUITES = {
    6: "v06_threshold_scenarios",
    7: "v07_activation_scenarios",
    8: "v08_focused_activation_thresholds",
}
RESULTS = {
    7: "v07_activation_scenarios_30seeds",
    8: "v08_focused_activation_thresholds_30seeds",
}
REVISED = ROOT / "analysis" / "paper_revised"
CASES = (("D1", "Chicago"), ("D2", "Memphis"), ("D3", "Columbus"))
COLORS = {"D1": "#0072B2", "D2": "#D55E00", "D3": "#009E73"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def analysis_rows(version: int, filename: str) -> list[dict[str, str]]:
    return read(ROOT / "analysis" / SUITES[version] / filename)


def select(data: list[dict[str, str]], case_id: str, scenario: str) -> dict[str, str]:
    return next(
        row
        for row in data
        if row["case_id"] == case_id and row["scenario"] == scenario
    )


def balanced_share(version: int, case_id: str, scenario: str) -> float:
    matches = [
        row
        for row in analysis_rows(version, "gateway_shares.csv")
        if row["case_id"] == case_id
        and row["scenario"] == scenario
        and "balanced" in row["roles"].lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one balanced solution for {case_id}/{scenario} in v0.{version}"
        )
    return float(matches[0]["ny_nj_share_pct"])


def all_solution_range(row: dict[str, str]) -> tuple[float, float]:
    union_size = int(row["union_solutions"])
    using_size = int(row["ny_nj_using_solutions"])
    minimum = (
        float(row["ny_nj_share_min_pct_when_used"])
        if union_size and using_size == union_size
        else 0.0
    )
    maximum = (
        float(row["ny_nj_share_max_pct_when_used"])
        if using_size
        else 0.0
    )
    return minimum, maximum


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def clean(axis: plt.Axes, direction: str = "y") -> None:
    axis.grid(axis=direction, color="#dddddd", linewidth=0.5)
    axis.set_axisbelow(True)


def save(figure: plt.Figure, name: str) -> None:
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            OUT / f"{name}.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


# Figure 1: cost-time trade-offs under baseline and severe disruptions.
fronts = {
    version: read(ROOT / "results" / directory / "combined_pareto.csv")
    for version, directory in RESULTS.items()
}
figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True, sharey=True)
scenario_colors = ("#4D4D4D", "#0072B2", "#D55E00")
for index, (case_id, city) in enumerate(CASES):
    delay = {"D1": 288, "D2": 264, "D3": 240}[case_id]
    settings = (
        (7, "pilot_baseline", "Baseline"),
        (8 if case_id != "D3" else 7, f"uswc_departure_delay_{delay}h", f"Delay: {delay} h"),
        (7, "uswc_compound_30pct_delay_168h", "30% capacity + 168 h delay"),
    )
    for setting_index, (version, scenario, label) in enumerate(settings):
        points = [
            row
            for row in fronts[version]
            if row["case_id"] == case_id and row["scenario"] == scenario
        ]
        axes[index].scatter(
            [float(row["quantity_weighted_mean_delivery_time_h"]) for row in points],
            [float(row["total_cost_usd"]) / 1e6 for row in points],
            s=19,
            color=scenario_colors[setting_index],
            marker=("o", "^", "s")[setting_index],
            label=label,
            alpha=0.85,
            linewidths=0.2,
        )
        if setting_index == 1:
            inset = axes[index].inset_axes([0.48, 0.19, 0.47, 0.29])
            inset.scatter(
                [float(row["quantity_weighted_mean_delivery_time_h"]) for row in points],
                [float(row["total_cost_usd"]) / 1e6 for row in points],
                color=scenario_colors[setting_index],
                s=8,
                marker="^",
            )
            inset.set_title("Delay scenario: enlarged", fontsize=7, pad=3)
            inset.tick_params(labelsize=6, pad=1)
            inset.ticklabel_format(useOffset=False, style="plain")
            inset.xaxis.set_major_locator(MaxNLocator(3))
            inset.yaxis.set_major_locator(MaxNLocator(3))
            inset.grid(color="#dddddd", linewidth=0.4)
    axes[index].set_title(f"({chr(97 + index)}) {city}")
    axes[index].set_xlabel("Weighted mean delivery time (hours)")
    axes[index].legend(fontsize=7, loc="upper left", frameon=False)
    clean(axes[index])
axes[0].set_ylabel("Total operational cost (USD million)")
figure.tight_layout()
save(figure, "fig01_cost_time_fronts")


# Figure 2: balanced solution and full archive range under delay-only scenarios.
activation = {version: analysis_rows(version, "ny_activation.csv") for version in SUITES}
figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharex=True, sharey=True)
for index, (case_id, city) in enumerate(CASES):
    axis = axes[index]
    lower, upper = {"D1": (264, 288), "D2": (240, 264), "D3": (168, 240)}[case_id]
    axis.axvspan(lower, upper, facecolor=COLORS[case_id], alpha=0.10)
    axis.axhline(10.0, color="#555555", linestyle="--", linewidth=0.8)
    tested = [(6, 0), (6, 96), (6, 120), (6, 168), (7, 240)]
    if case_id != "D3":
        tested.extend((8, delay_value) for delay_value in (264, 288, 312))
    tested.append((7, 336))
    for version, delay_value in tested:
        scenario = (
            "pilot_baseline"
            if delay_value == 0
            else f"uswc_departure_delay_{delay_value}h"
        )
        row = select(activation[version], case_id, scenario)
        minimum, maximum = all_solution_range(row)
        point = balanced_share(version, case_id, scenario)
        axis.vlines(delay_value, minimum, maximum, color=COLORS[case_id], linewidth=1.4)
        axis.plot(delay_value, point, "o", color=COLORS[case_id], markersize=5)
        axis.plot((delay_value, delay_value), (minimum, maximum), "_", color=COLORS[case_id], markersize=8)
    axis.text(
        (lower + upper) / 2,
        92,
        f"Tested interval: ({lower}, {upper}] h",
        ha="center",
        fontsize=8,
        color=COLORS[case_id],
    )
    axis.set_title(f"({chr(97 + index)}) {city}")
    axis.set_xlabel("Additional sailing delay (hours)")
    axis.set_xticks((0, 96, 168, 240, 288, 336))
    axis.set_ylim(-3, 100)
    clean(axis)
axes[0].set_ylabel("New York/New Jersey freight share (%)")
figure.legend(
    handles=(
        Line2D([0], [0], marker="o", color="#333333", linestyle="none", label="Balanced solution"),
        Line2D([0], [0], color="#333333", linewidth=1.4, label="Full pooled-front range"),
        Line2D([0], [0], color="#555555", linestyle="--", label="10% criterion"),
    ),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=3,
    frameon=False,
    fontsize=8,
)
figure.tight_layout(rect=(0, 0.08, 1, 1))
save(figure, "fig02_delay_activation")


# Figure 3: categorical capacity-delay activation regimes.
regime_rows = read(REVISED / "activation_classification.csv")
state_code = {
    "no_meaningful_activation": 0,
    "transition_regime": 1,
    "front_wide_activation": 2,
}
capacities = (60, 40, 35)
delays = (0, 168, 288)
cmap = ListedColormap(("#D9D9D9", "#E69F00", "#0072B2"))
norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), cmap.N)
figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.8), sharey=True)
for index, (case_id, city) in enumerate(CASES):
    lookup = {
        (
            round(float(row["remaining_uswc_capacity_pct"])),
            round(float(row["uswc_delay_h"])),
        ): row
        for row in regime_rows
        if row["case_id"] == case_id
        and round(float(row["remaining_uswc_capacity_pct"])) in capacities
        and round(float(row["uswc_delay_h"])) in delays
    }
    values = [
        [state_code[lookup[(capacity, delay_value)]["activation_state"]] for capacity in capacities]
        for delay_value in delays
    ]
    axes[index].imshow(values, cmap=cmap, norm=norm, aspect="auto")
    for row_index, delay_value in enumerate(delays):
        for column_index, capacity in enumerate(capacities):
            share = float(lookup[(capacity, delay_value)]["balanced_ny_nj_share_pct"])
            text_color = "white" if values[row_index][column_index] == 2 else "#222222"
            axes[index].text(column_index, row_index, f"{share:.1f}%", ha="center", va="center", color=text_color)
    axes[index].set_title(f"({chr(97 + index)}) {city}")
    axes[index].set_xticks(range(len(capacities)), [f"{value}%" for value in capacities])
    axes[index].set_yticks(range(len(delays)), [str(value) for value in delays])
    axes[index].set_xlabel("Remaining West Coast capacity")
axes[0].set_ylabel("Additional sailing delay (hours)")
figure.legend(
    handles=(
        Patch(facecolor="#D9D9D9", label="No meaningful activation"),
        Patch(facecolor="#E69F00", label="Transition regime"),
        Patch(facecolor="#0072B2", label="Front-wide activation"),
    ),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    frameon=False,
    fontsize=8,
)
figure.tight_layout(rect=(0, 0.10, 1, 1))
save(figure, "fig03_activation_regime")


# Figure 4: archive-closed allowed and restricted Pareto sets.
counterfactual_points = read(REVISED / "counterfactual_front_points.csv")
figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
normalizer = Normalize(vmin=0.0, vmax=60.0)
for index, (case_id, city) in enumerate(CASES):
    rows = [
        row
        for row in counterfactual_points
        if row["case_id"] == case_id and row["comparison"] == "delay_activation"
    ]
    restricted = [row for row in rows if row["condition"] == "restricted"]
    allowed = [row for row in rows if row["condition"] == "archive_closed_allowed"]
    image = axes[index].scatter(
        [float(row["quantity_weighted_mean_delivery_time_h"]) for row in allowed],
        [float(row["total_cost_usd"]) / 1e6 for row in allowed],
        c=[float(row["ny_nj_share_pct"]) for row in allowed],
        cmap="viridis",
        norm=normalizer,
        marker="o",
        s=24,
        edgecolors="black",
        linewidths=0.2,
        label="NY/NJ allowed, archive closed",
    )
    axes[index].scatter(
        [float(row["quantity_weighted_mean_delivery_time_h"]) for row in restricted],
        [float(row["total_cost_usd"]) / 1e6 for row in restricted],
        marker="x",
        color="#777777",
        s=28,
        linewidth=1.0,
        label="NY/NJ restricted",
    )
    axes[index].set_title(f"({chr(97 + index)}) {city}")
    axes[index].set_xlabel("Weighted mean delivery time (hours)")
    axes[index].ticklabel_format(useOffset=False, style="plain")
    clean(axes[index])
    axes[index].legend(frameon=False, fontsize=7)
axes[0].set_ylabel("Total operational cost (USD million)")
figure.subplots_adjust(left=0.07, right=0.88, bottom=0.16, top=0.90, wspace=0.28)
# Reserve a dedicated axis outside the three panels.  Calling subplots_adjust
# after figure.colorbar moves the panels without moving the colorbar and can
# leave the scale overlapping the Columbus panel in the exported figure.
colorbar_axis = figure.add_axes((0.91, 0.20, 0.015, 0.64))
colorbar = figure.colorbar(image, cax=colorbar_axis)
colorbar.set_label("NY/NJ freight share (%)")
save(figure, "fig04_counterfactual_fronts")


# Figure 5: paired archive-closed endpoint benefits.
benefits = read(REVISED / "counterfactual_value_summary.csv")
order = [
    (case_id, comparison)
    for case_id, _ in CASES
    for comparison in ("delay_activation", "extreme_compound")
]
labels = [
    f"{dict(CASES)[case_id]}: "
    + ("delay activation" if comparison == "delay_activation" else "30% capacity + 168 h")
    for case_id, comparison in order
]
figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for axis, metric, percent_scale, title, xlabel in (
    (axes[0], "cost_avoided_by_ny_nj", True, "(a) Cost endpoint", "Cost avoided relative to restricted case (%)"),
    (axes[1], "delivery_time_avoided_by_ny_nj", False, "(b) Delivery-time endpoint", "Delivery time avoided (hours)"),
):
    for position, (case_id, comparison) in enumerate(order):
        row = next(
            item
            for item in benefits
            if item["case_id"] == case_id
            and item["comparison"] == comparison
            and item["metric"] == metric
        )
        if percent_scale:
            estimate = float(row["mean_benefit_pct"])
            low = float(row["bootstrap_mean_pct_ci_low"])
            high = float(row["bootstrap_mean_pct_ci_high"])
        else:
            estimate = float(row["mean_benefit_from_allowing_ny_nj"])
            low = float(row["bootstrap_mean_ci_low"])
            high = float(row["bootstrap_mean_ci_high"])
        axis.errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            capsize=4,
            markersize=5,
            color=COLORS[case_id],
        )
        axis.text(
            high,
            position - 0.18,
            f"{row['positive_benefit_runs']}/30",
            fontsize=7,
            ha="left",
            color=COLORS[case_id],
        )
    axis.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.margins(x=0.15)
    clean(axis, "x")
axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
axes[0].invert_yaxis()
figure.tight_layout()
save(figure, "fig05_counterfactual_benefits")


# Figure 6: destination-specific mechanism differences (NY/NJ minus LA/LB).
mechanism = read(REVISED / "destination_mechanism_summary.csv")
positions = range(len(CASES))
figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
width = 0.22
time_fields = (
    ("ny_minus_la_ocean_transit_h", "Ocean transit", "#0072B2"),
    ("ny_minus_la_waiting_h", "Waiting", "#D55E00"),
    ("ny_minus_la_us_inland_transit_h", "US inland", "#009E73"),
)
for offset, (field, label, color) in zip((-width, 0.0, width), time_fields):
    axes[0].bar(
        [position + offset for position in positions],
        [float(next(row for row in mechanism if row["case_id"] == case_id)[field]) for case_id, _ in CASES],
        width=width,
        label=label,
        color=color,
    )
axes[0].plot(
    list(positions),
    [
        float(next(row for row in mechanism if row["case_id"] == case_id)["ny_minus_la_total_lead_time_h"])
        for case_id, _ in CASES
    ],
    marker="D",
    color="#222222",
    linestyle="none",
    label="Total lead time",
)
axes[0].axhline(0.0, color="#555555", linewidth=0.8)
axes[0].set_title("(a) Time components")
axes[0].set_ylabel("NY/NJ minus LA/LB (hours)")
axes[0].legend(frameon=False, fontsize=8, ncol=2)
clean(axes[0])

cost_values = [
    float(next(row for row in mechanism if row["case_id"] == case_id)["ny_minus_la_total_cost_usd_per_feu"])
    for case_id, _ in CASES
]
axes[1].bar(
    list(positions),
    cost_values,
    color=["#009E73" if value < 0 else "#D55E00" for value in cost_values],
    width=0.55,
)
axes[1].axhline(0.0, color="#555555", linewidth=0.8)
axes[1].set_title("(b) Unit cost")
axes[1].set_ylabel("NY/NJ minus LA/LB (USD/FEU)")
clean(axes[1])
for axis in axes:
    axis.set_xticks(list(positions), [city for _, city in CASES])
figure.tight_layout()
save(figure, "fig06_destination_mechanism")


# Supplementary Figure S1: 168-hour compound disruption.
figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True, sharey=True)
for index, (case_id, city) in enumerate(CASES):
    axis = axes[index]
    axis.axhline(10.0, color="#555555", linestyle="--", linewidth=0.8)
    for capacity in (40, 35, 30):
        if case_id == "D3" and capacity == 35:
            axis.text(35, 3, "Not tested", ha="center", fontsize=8, color="#666666")
            continue
        version = 8 if capacity == 35 else 7
        scenario = f"uswc_compound_{capacity}pct_delay_168h"
        row = select(activation[version], case_id, scenario)
        minimum, maximum = all_solution_range(row)
        point = balanced_share(version, case_id, scenario)
        axis.vlines(capacity, minimum, maximum, color=COLORS[case_id], linewidth=1.4)
        axis.plot(capacity, point, "o", color=COLORS[case_id], markersize=5)
    axis.set_title(f"({chr(97 + index)}) {city}")
    axis.set_xlim(42, 28)
    axis.set_xticks((40, 35, 30))
    axis.set_ylim(0, 60)
    axis.set_xlabel("Remaining West Coast capacity (%)")
    clean(axis)
axes[0].set_ylabel("New York/New Jersey freight share (%)")
figure.tight_layout()
save(figure, "figS01_compound_activation")

print(f"Publication figures written to {OUT}.")
