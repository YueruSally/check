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
FINAL_VALIDATION = ROOT / "analysis" / "v12_final_validation"
EXACT_VALIDATION = ROOT / "results" / "v12_exact_endpoint_validation"
V12_RESULTS = ROOT / "results" / "v12_final_closed_fronts_30seeds"
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
v12_fronts = {
    case_id: read(V12_RESULTS / case_id / "combined_pareto.csv")
    for case_id, _ in CASES
}
figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True, sharey=True)
scenario_colors = ("#4D4D4D", "#0072B2", "#D55E00")
for index, (case_id, city) in enumerate(CASES):
    delay = {"D1": 288, "D2": 288, "D3": 240}[case_id]
    settings = (
        (fronts[7], "pilot_baseline", "Baseline"),
        (v12_fronts[case_id], f"uswc_departure_delay_{delay}h", f"Delay: {delay} h"),
        (fronts[7], "uswc_compound_30pct_delay_168h", "30% capacity + 168 h delay"),
    )
    for setting_index, (source, scenario, label) in enumerate(settings):
        points = [
            row
            for row in source
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
closed_activation = {
    (row["case_id"], row["allowed_scenario"]): row
    for row in read(FINAL_VALIDATION / "closed_front_classification.csv")
}
figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharex=True, sharey=True)
for index, (case_id, city) in enumerate(CASES):
    axis = axes[index]
    lower, upper = {"D1": (264, 288), "D2": (264, 288), "D3": (168, 240)}[case_id]
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
        final_row = closed_activation.get((case_id, scenario))
        if final_row is None:
            row = select(activation[version], case_id, scenario)
            minimum, maximum = all_solution_range(row)
        else:
            minimum = 100.0 * float(final_row["ny_nj_share_min"])
            maximum = 100.0 * float(final_row["ny_nj_share_max"])
        point = (minimum + maximum) / 2.0
        axis.vlines(delay_value, minimum, maximum, color=COLORS[case_id], linewidth=1.4)
        axis.plot(delay_value, point, "o", color=COLORS[case_id], markersize=5)
        axis.plot((delay_value, delay_value), (minimum, maximum), "_", color=COLORS[case_id], markersize=8)
    axis.text(
        (lower + upper) / 2,
        92,
        f"Front-wide interval: ({lower}, {upper}] h",
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
        Line2D([0], [0], marker="o", color="#333333", linestyle="none", label="Range midpoint"),
        Line2D([0], [0], color="#333333", linewidth=1.4, label="Retained-configuration range"),
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
        Patch(facecolor="#D9D9D9", label="No allowed configuration ≥10%"),
        Patch(facecolor="#E69F00", label="Allowed range crosses 10%"),
        Patch(facecolor="#0072B2", label="All allowed configurations ≥10%"),
    ),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    frameon=False,
    fontsize=8,
)
figure.tight_layout(rect=(0, 0.10, 1, 1))
save(figure, "fig03_activation_regime")


# Figure 4: configuration-preserving closed fronts from the targeted rerun.
counterfactual_points = read(FINAL_VALIDATION / "closed_front_points.csv")
figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
normalizer = Normalize(vmin=0.0, vmax=60.0)
for index, (case_id, city) in enumerate(CASES):
    comparison = {"D1": "delay_288h", "D2": "delay_264h", "D3": "delay_240h"}[case_id]
    restricted_scenario = {
        "D1": "uswc_departure_delay_288h_no_ny_nj",
        "D2": "uswc_departure_delay_264h_no_ny_nj",
        "D3": "uswc_departure_delay_240h_no_ny_nj",
    }[case_id]
    rows = [
        row
        for row in counterfactual_points
        if row["case_id"] == case_id and row["comparison"] == comparison
    ]
    restricted = [
        row
        for row in v12_fronts[case_id]
        if row["case_id"] == case_id and row["scenario"] == restricted_scenario
    ]
    image = axes[index].scatter(
        [float(row["quantity_weighted_mean_delivery_time_h"]) for row in rows],
        [float(row["total_cost_usd"]) / 1e6 for row in rows],
        c=[
            50.0 * (float(row["ny_nj_share_min"]) + float(row["ny_nj_share_max"]))
            for row in rows
        ],
        cmap="viridis",
        norm=normalizer,
        marker="o",
        s=24,
        edgecolors="black",
        linewidths=0.2,
        label="Closed non-dominated union",
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
colorbar.set_label("NY/NJ share range midpoint (%)")
save(figure, "fig04_counterfactual_fronts")


# Figure 5: proven lexicographic endpoint effects.
benefits = read(EXACT_VALIDATION / "exact_counterfactual_value.csv")
order = [
    ("D1", "delay_288h", "Chicago: 288 h delay"),
    ("D1", "compound_30pct_delay_168h", "Chicago: 30% + 168 h"),
    ("D2", "delay_264h", "Memphis: 264 h delay"),
    ("D2", "delay_288h", "Memphis: 288 h delay"),
    ("D3", "delay_240h", "Columbus: 240 h delay"),
]
labels = [label for _, _, label in order]
figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
endpoint_styles = (
    ("min_cost_then_time", "o", "Minimum cost, then time"),
    ("min_time_then_cost", "D", "Minimum time, then cost"),
)
for axis, field, scale, title, xlabel in (
    (axes[0], "cost_avoided_by_allowing_ny_nj_usd", 1e-6, "(a) Cost difference", "Restricted minus allowed cost (USD million)"),
    (axes[1], "time_avoided_by_allowing_ny_nj_h", 1.0, "(b) Delivery-time difference", "Restricted minus allowed time (hours)"),
):
    for position, (case_id, comparison, _) in enumerate(order):
        for endpoint, marker, endpoint_label in endpoint_styles:
            row = next(
                item
                for item in benefits
                if item["case_id"] == case_id
                and item["comparison"] == comparison
                and item["endpoint"] == endpoint
            )
            axis.plot(
                float(row[field]) * scale,
                position,
                marker=marker,
                color=COLORS[case_id],
                markerfacecolor=("white" if marker == "D" else COLORS[case_id]),
                linestyle="none",
                markersize=6,
                label=endpoint_label if position == 0 else None,
            )
    axis.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.margins(x=0.15)
    clean(axis, "x")
axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
axes[0].invert_yaxis()
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
figure.tight_layout()
save(figure, "fig05_counterfactual_benefits")


# Figure 6: seed-paired shipment-matched time decomposition.
mechanism = read(FINAL_VALIDATION / "batch_matched_time_decomposition_summary.csv")
mechanism_order = [
    ("D1", "delay_288h", "Chicago\n288 h"),
    ("D2", "delay_264h", "Memphis\n264 h"),
    ("D2", "delay_288h", "Memphis\n288 h"),
    ("D3", "delay_240h", "Columbus\n240 h"),
]
positions = list(range(len(mechanism_order)))
figure, axis = plt.subplots(figsize=(9.6, 4.6))
width = 0.30
for offset, component, label, color in (
    (-width / 2, "ny_nj_contribution_h", "NY/NJ-assigned component", "#D55E00"),
    (width / 2, "west_coast_contribution_h", "West Coast-assigned component", "#0072B2"),
):
    estimates = []
    lower = []
    upper = []
    for case_id, comparison, _ in mechanism_order:
        row = next(
            item
            for item in mechanism
            if item["case_id"] == case_id
            and item["comparison"] == comparison
            and item["component"] == component
        )
        estimate = float(row["mean_h"])
        estimates.append(estimate)
        lower.append(estimate - float(row["mean_bootstrap_95_low_h"]))
        upper.append(float(row["mean_bootstrap_95_high_h"]) - estimate)
    axis.bar(
        [position + offset for position in positions],
        estimates,
        width=width,
        color=color,
        label=label,
        yerr=[lower, upper],
        capsize=3,
        error_kw={"linewidth": 0.8},
    )
totals = []
total_low = []
total_high = []
for case_id, comparison, _ in mechanism_order:
    row = next(
        item
        for item in mechanism
        if item["case_id"] == case_id
        and item["comparison"] == comparison
        and item["component"] == "allowed_minus_restricted_time_h"
    )
    estimate = float(row["mean_h"])
    totals.append(estimate)
    total_low.append(estimate - float(row["mean_bootstrap_95_low_h"]))
    total_high.append(float(row["mean_bootstrap_95_high_h"]) - estimate)
axis.errorbar(
    positions,
    totals,
    yerr=[total_low, total_high],
    fmt="D",
    color="#222222",
    markerfacecolor="white",
    capsize=3,
    markersize=5,
    label="System total",
)
axis.axhline(0.0, color="#555555", linewidth=0.8)
axis.set_ylabel("Allowed minus restricted mean time (hours)")
axis.set_xticks(positions, [label for _, _, label in mechanism_order])
axis.legend(frameon=False, fontsize=8, ncol=3)
clean(axis)
figure.tight_layout()
save(figure, "fig06_destination_mechanism")


# Supplementary Figure S1: 168-hour compound disruption.
historical_corrections = {
    (row["case_id"], row["comparison"]): row
    for row in read(FINAL_VALIDATION / "historical_closed_front_corrections.csv")
}
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
        correction = historical_corrections.get(
            (case_id, f"compound_{capacity}pct_delay_168h")
        )
        if correction is not None:
            minimum = 100.0 * float(correction["ny_nj_share_min"])
            maximum = 100.0 * float(correction["ny_nj_share_max"])
        point = (minimum + maximum) / 2.0
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
